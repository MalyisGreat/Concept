"""
Train student model on Qwen3-extracted concept vectors.

Usage:
    python train_on_qwen3.py --vectors qwen3_vectors_qwen3_1.7b.pt --epochs 200
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
from pathlib import Path
import json
from dataclasses import dataclass
import random
import numpy as np

# ============================================================================
# Configuration
# ============================================================================

@dataclass
class Config:
    # Student model - using MLP since it worked best
    student_type: str = "mlp"  # "mlp" or "transformer"
    student_hidden_dim: int = 512
    student_num_layers: int = 4
    student_dropout: float = 0.3

    # Training
    batch_size: int = 64
    learning_rate: float = 3e-4
    num_epochs: int = 200
    warmup_steps: int = 200
    weight_decay: float = 0.1

    # Data
    train_split: float = 0.7
    val_split: float = 0.15
    # test_split = 1 - train_split - val_split

    # Regularization
    label_smoothing: float = 0.1

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================================
# Models
# ============================================================================

class MLPStudent(nn.Module):
    """MLP-based student model - best performing architecture."""

    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int,
                 output_dim: int, num_layers: int, dropout: float = 0.3):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)

        layers = []
        in_dim = embed_dim

        for i in range(num_layers):
            out_dim = hidden_dim if i < num_layers - 1 else output_dim
            layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.LayerNorm(out_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            in_dim = out_dim

        # Final projection
        layers.append(nn.Linear(output_dim, output_dim))

        self.mlp = nn.Sequential(*layers)

    def forward(self, input_ids, attention_mask=None):
        # Embed
        x = self.embedding(input_ids)  # [B, L, E]

        # Mean pool over sequence
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            x = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            x = x.mean(dim=1)

        # MLP
        x = self.mlp(x)

        return x


class TransformerStudent(nn.Module):
    """Tiny transformer student model."""

    def __init__(self, vocab_size: int, embed_dim: int, num_layers: int,
                 num_heads: int, ff_dim: int, output_dim: int,
                 max_seq_len: int = 64, dropout: float = 0.1):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Embedding(max_seq_len, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(embed_dim, output_dim)
        self.layer_norm = nn.LayerNorm(output_dim)

    def forward(self, input_ids, attention_mask=None):
        B, L = input_ids.shape

        # Embeddings
        pos = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, -1)
        x = self.embedding(input_ids) + self.pos_embedding(pos)

        # Create attention mask for transformer
        if attention_mask is not None:
            src_key_padding_mask = (attention_mask == 0)
        else:
            src_key_padding_mask = None

        # Transform
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)

        # Mean pool
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            x = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            x = x.mean(dim=1)

        # Project to output
        x = self.output_proj(x)
        x = self.layer_norm(x)

        return x


# ============================================================================
# Dataset
# ============================================================================

SENTENCE_TEMPLATES = [
    "The concept of {concept} is important.",
    "{concept} plays a significant role.",
    "Understanding {concept} is essential.",
    "The meaning of {concept} varies.",
    "{concept} can be described as follows.",
    "In the context of {concept}, we see that",
    "The word {concept} represents a key idea.",
    "{concept} is fundamental to understanding.",
    "When discussing {concept}, experts note",
    "The nature of {concept} is complex.",
    "{concept} has many interpretations.",
    "Studying {concept} reveals insights.",
    "The essence of {concept} lies in",
    "{concept} encompasses multiple aspects.",
    "Defining {concept} requires careful thought.",
]


class ConceptDataset(Dataset):
    def __init__(self, concepts: list, direction_vectors: dict,
                 tokenizer, max_length: int = 64,
                 sentences_per_concept: int = 10):
        self.concepts = concepts
        self.direction_vectors = direction_vectors
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.sentences_per_concept = sentences_per_concept

        # Pre-generate all data
        self.data = []
        for concept in concepts:
            vec = direction_vectors[concept]
            for _ in range(sentences_per_concept):
                template = random.choice(SENTENCE_TEMPLATES)
                sentence = template.format(concept=concept)
                self.data.append((sentence, vec, concept))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sentence, vector, concept = self.data[idx]

        # Tokenize
        encoded = self.tokenizer(
            sentence,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "target_vector": vector,
            "concept": concept,
        }


# ============================================================================
# Training
# ============================================================================

class Trainer:
    def __init__(self, model, config: Config, tokenizer):
        self.model = model
        self.config = config
        self.tokenizer = tokenizer
        self.device = config.device

        self.model.to(self.device)

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

        # Cosine schedule with warmup
        self.scheduler = None  # Set after we know total steps

    def compute_loss(self, predictions, targets, label_smoothing=0.0):
        """Compute cosine similarity loss with optional label smoothing."""
        pred_norm = F.normalize(predictions, p=2, dim=-1)
        target_norm = F.normalize(targets, p=2, dim=-1)

        cos_sim = (pred_norm * target_norm).sum(dim=-1)

        # Target is 1.0 (or 1-label_smoothing for smoothing)
        target_sim = 1.0 - label_smoothing
        loss = (target_sim - cos_sim).pow(2).mean()

        return loss, cos_sim.mean()

    def train_epoch(self, dataloader, epoch: int):
        self.model.train()
        total_loss = 0
        total_cos_sim = 0
        num_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        for batch in pbar:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            targets = batch["target_vector"].to(self.device)

            # Forward
            predictions = self.model(input_ids, attention_mask)

            # Loss
            loss, cos_sim = self.compute_loss(
                predictions, targets,
                label_smoothing=self.config.label_smoothing
            )

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            if self.scheduler is not None:
                self.scheduler.step()

            total_loss += loss.item()
            total_cos_sim += cos_sim.item()
            num_batches += 1

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "cos_sim": f"{cos_sim.item():.4f}"
            })

        return total_loss / num_batches, total_cos_sim / num_batches

    @torch.no_grad()
    def evaluate(self, dataloader, desc="Eval"):
        self.model.eval()
        total_loss = 0
        total_cos_sim = 0
        num_batches = 0

        all_predictions = []
        all_targets = []
        all_concepts = []

        for batch in tqdm(dataloader, desc=desc):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            targets = batch["target_vector"].to(self.device)

            predictions = self.model(input_ids, attention_mask)

            loss, cos_sim = self.compute_loss(predictions, targets)

            total_loss += loss.item()
            total_cos_sim += cos_sim.item()
            num_batches += 1

            all_predictions.append(predictions.cpu())
            all_targets.append(targets.cpu())
            all_concepts.extend(batch["concept"])

        # Compute retrieval accuracy
        all_predictions = torch.cat(all_predictions)
        all_targets = torch.cat(all_targets)

        # Normalize
        pred_norm = F.normalize(all_predictions, p=2, dim=-1)
        target_norm = F.normalize(all_targets, p=2, dim=-1)

        # Compute all pairwise similarities
        sim_matrix = pred_norm @ target_norm.T

        # Top-1 and Top-5 accuracy
        top1_correct = 0
        top5_correct = 0
        for i in range(len(all_predictions)):
            sorted_indices = sim_matrix[i].argsort(descending=True)
            if sorted_indices[0] == i:
                top1_correct += 1
            if i in sorted_indices[:5]:
                top5_correct += 1

        return {
            "loss": total_loss / num_batches,
            "cos_sim": total_cos_sim / num_batches,
            "top1_acc": top1_correct / len(all_predictions),
            "top5_acc": top5_correct / len(all_predictions),
        }


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vectors", type=str, required=True,
                        help="Path to extracted vectors (.pt file)")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--model-type", type=str, default="mlp",
                        choices=["mlp", "transformer"])
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--output", type=str, default="trained_student.pt")

    args = parser.parse_args()

    # Load extracted vectors
    print(f"Loading vectors from {args.vectors}...")
    data = torch.load(args.vectors)

    print(f"\nVector file info:")
    print(f"  Model: {data['model']}")
    print(f"  Hidden dim: {data['hidden_dim']}")
    print(f"  Extraction layer: {data['extraction_layer']}/{data['num_layers']}")
    print(f"  Num concepts: {data['num_concepts']}")

    concepts = data["concepts"]
    direction_vectors = data["direction_vectors"]
    output_dim = data["hidden_dim"]

    # Split concepts
    random.shuffle(concepts)
    n = len(concepts)
    n_train = int(n * 0.7)
    n_val = int(n * 0.15)

    train_concepts = concepts[:n_train]
    val_concepts = concepts[n_train:n_train+n_val]
    test_concepts = concepts[n_train+n_val:]

    print(f"\nData split:")
    print(f"  Train: {len(train_concepts)} concepts")
    print(f"  Val: {len(val_concepts)} concepts")
    print(f"  Test: {len(test_concepts)} concepts")

    # Load tokenizer (use same as extraction)
    tokenizer = AutoTokenizer.from_pretrained(data["model_id"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    vocab_size = tokenizer.vocab_size

    # Create config
    config = Config()
    config.num_epochs = args.epochs
    config.batch_size = args.batch_size
    config.learning_rate = args.lr
    config.student_type = args.model_type
    config.student_hidden_dim = args.hidden_dim
    config.student_num_layers = args.num_layers

    # Create datasets
    train_dataset = ConceptDataset(train_concepts, direction_vectors, tokenizer)
    val_dataset = ConceptDataset(val_concepts, direction_vectors, tokenizer, sentences_per_concept=3)
    test_dataset = ConceptDataset(test_concepts, direction_vectors, tokenizer, sentences_per_concept=3)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size)

    # Create model
    if config.student_type == "mlp":
        model = MLPStudent(
            vocab_size=vocab_size,
            embed_dim=256,
            hidden_dim=config.student_hidden_dim,
            output_dim=output_dim,
            num_layers=config.student_num_layers,
            dropout=config.student_dropout,
        )
    else:
        model = TransformerStudent(
            vocab_size=vocab_size,
            embed_dim=128,
            num_layers=config.student_num_layers,
            num_heads=4,
            ff_dim=512,
            output_dim=output_dim,
            dropout=config.student_dropout,
        )

    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nStudent model: {config.student_type}")
    print(f"  Parameters: {num_params:,}")
    print(f"  Hidden dim: {config.student_hidden_dim}")
    print(f"  Num layers: {config.student_num_layers}")
    print(f"  Output dim: {output_dim}")

    # Trainer
    trainer = Trainer(model, config, tokenizer)

    # Training loop
    print(f"\nStarting training for {config.num_epochs} epochs...")

    history = {
        "train_loss": [], "train_cos_sim": [],
        "val_loss": [], "val_cos_sim": [],
    }

    best_val_cos_sim = -1
    best_epoch = 0

    for epoch in range(1, config.num_epochs + 1):
        # Train
        train_loss, train_cos_sim = trainer.train_epoch(train_loader, epoch)
        history["train_loss"].append(train_loss)
        history["train_cos_sim"].append(train_cos_sim)

        # Validate every 5 epochs
        if epoch % 5 == 0 or epoch == 1:
            val_metrics = trainer.evaluate(val_loader, "Validating")
            history["val_loss"].append(val_metrics["loss"])
            history["val_cos_sim"].append(val_metrics["cos_sim"])

            print(f"\nEpoch {epoch}:")
            print(f"  Train - Loss: {train_loss:.4f}, Cos Sim: {train_cos_sim:.4f}")
            print(f"  Val   - Loss: {val_metrics['loss']:.4f}, Cos Sim: {val_metrics['cos_sim']:.4f}")
            print(f"  Val   - Top-1: {val_metrics['top1_acc']:.4f}, Top-5: {val_metrics['top5_acc']:.4f}")

            if val_metrics["cos_sim"] > best_val_cos_sim:
                best_val_cos_sim = val_metrics["cos_sim"]
                best_epoch = epoch
                # Save best model
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "epoch": epoch,
                    "val_cos_sim": val_metrics["cos_sim"],
                }, args.output)
                print(f"  New best! Saved to {args.output}")

    # Final test evaluation
    print("\n" + "="*60)
    print("FINAL TEST EVALUATION")
    print("="*60)

    # Load best model
    checkpoint = torch.load(args.output)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_metrics = trainer.evaluate(test_loader, "Testing")

    print(f"\nTest Results (best model from epoch {best_epoch}):")
    print(f"  Cosine Similarity: {test_metrics['cos_sim']:.4f}")
    print(f"  Top-1 Accuracy: {test_metrics['top1_acc']:.4f}")
    print(f"  Top-5 Accuracy: {test_metrics['top5_acc']:.4f}")

    # Plot training curves
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Train")
    epochs_with_val = list(range(0, len(history["train_loss"]), 5))
    if 0 not in epochs_with_val:
        epochs_with_val = [0] + epochs_with_val
    plt.plot(epochs_with_val[:len(history["val_loss"])], history["val_loss"], label="Val", marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history["train_cos_sim"], label="Train")
    plt.plot(epochs_with_val[:len(history["val_cos_sim"])], history["val_cos_sim"], label="Val", marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Cosine Similarity")
    plt.title("Cosine Similarity")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("training_curves_qwen3.png", dpi=150)
    print(f"\nSaved training curves to training_curves_qwen3.png")

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Teacher model: {data['model']}")
    print(f"Teacher hidden dim: {output_dim}")
    print(f"Student type: {config.student_type}")
    print(f"Student params: {num_params:,}")
    print(f"Training concepts: {len(train_concepts)}")
    print(f"Test concepts: {len(test_concepts)}")
    print(f"Best val cos_sim: {best_val_cos_sim:.4f} (epoch {best_epoch})")
    print(f"Test cos_sim: {test_metrics['cos_sim']:.4f}")
    print(f"Test top-1 acc: {test_metrics['top1_acc']:.4f}")
    print(f"Test top-5 acc: {test_metrics['top5_acc']:.4f}")
    print("="*60)


if __name__ == "__main__":
    main()
