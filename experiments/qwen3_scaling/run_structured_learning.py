"""
Structured Learning Experiment: Force learning geometry, not memorization.

Key techniques:
1. CONTRASTIVE STRUCTURE - Learn "betrayal closer to deception than loyalty"
2. BOTTLENECK ARCHITECTURE - Narrow intermediate layers force compression
3. TRIPLET RELATIONSHIPS - A:B :: C:D analogies
4. MULTI-LAYER SUPERVISION - Match multiple teacher layers (if available)
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
from datetime import datetime

# ============================================================================
# Configuration
# ============================================================================

@dataclass
class Config:
    # Bottleneck architecture
    embed_dim: int = 64
    bottleneck_dim: int = 32      # NARROW - forces compression
    hidden_dim: int = 128
    num_layers: int = 2

    # Loss weights
    cosine_weight: float = 1.0      # Direct vector matching
    contrastive_weight: float = 1.0  # Relational structure
    triplet_margin: float = 0.3     # Margin for triplet loss

    # Training
    batch_size: int = 32
    learning_rate: float = 1e-3
    num_epochs: int = 150
    weight_decay: float = 0.01

    # Data
    train_split: float = 0.7
    val_split: float = 0.15
    sentences_per_concept: int = 3

    device: str = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================================
# Bottleneck Model - Forces compression through narrow layer
# ============================================================================

class BottleneckModel(nn.Module):
    """
    Architecture with narrow bottleneck that forces learning compressed representation.

    Input → Embed → Compress to bottleneck → Expand → Output

    The bottleneck is MUCH smaller than input/output, so the model
    must learn efficient encoding of concept structure.
    """

    def __init__(self, vocab_size: int, embed_dim: int, bottleneck_dim: int,
                 hidden_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # Encoder: compress to bottleneck
        self.encoder = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, bottleneck_dim),  # COMPRESS
            nn.LayerNorm(bottleneck_dim),
        )

        # Decoder: expand from bottleneck to output
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

        self.bottleneck_dim = bottleneck_dim

    def encode(self, input_ids, attention_mask=None):
        """Get bottleneck representation."""
        x = self.embedding(input_ids)

        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            x = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            x = x.mean(dim=1)

        return self.encoder(x)  # [B, bottleneck_dim]

    def decode(self, bottleneck):
        """Expand bottleneck to output."""
        return self.decoder(bottleneck)

    def forward(self, input_ids, attention_mask=None):
        bottleneck = self.encode(input_ids, attention_mask)
        output = self.decode(bottleneck)
        return output, bottleneck

# ============================================================================
# Datasets with relational structure
# ============================================================================

TEMPLATES = [
    "{concept}",
    "the concept {concept}",
    "{concept} means",
    "understanding {concept}",
]


class ConceptDataset(Dataset):
    """Basic concept → vector dataset."""

    def __init__(self, concepts, direction_vectors, tokenizer, max_length=24):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []

        for concept in concepts:
            vec = direction_vectors[concept]
            for template in TEMPLATES:
                text = template.format(concept=concept)
                self.data.append((text, vec, concept))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text, vector, concept = self.data[idx]
        encoded = self.tokenizer(text, max_length=self.max_length,
                                 padding="max_length", truncation=True,
                                 return_tensors="pt")
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "target_vector": vector,
            "concept": concept,
        }


class TripletDataset(Dataset):
    """
    Triplet dataset for learning relational structure.

    Each sample: (anchor, positive, negative)
    - anchor and positive should be semantically similar
    - anchor and negative should be dissimilar

    We determine similarity by cosine similarity in teacher space.
    """

    def __init__(self, concepts, direction_vectors, tokenizer, max_length=24,
                 num_triplets=5000):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.concepts = concepts
        self.direction_vectors = direction_vectors

        # Precompute similarity matrix
        vecs = torch.stack([direction_vectors[c] for c in concepts])
        vecs_norm = F.normalize(vecs, p=2, dim=-1)
        self.sim_matrix = vecs_norm @ vecs_norm.T  # [N, N]

        # Generate triplets
        self.triplets = self._generate_triplets(num_triplets)

    def _generate_triplets(self, num_triplets):
        """Generate (anchor, positive, negative) triplets based on similarity."""
        triplets = []
        n = len(self.concepts)

        for _ in range(num_triplets):
            # Random anchor
            anchor_idx = random.randint(0, n-1)

            # Find positive (high similarity) and negative (low similarity)
            sims = self.sim_matrix[anchor_idx].numpy()

            # Positive: top 20% similar (excluding self)
            sorted_idx = np.argsort(sims)[::-1]
            pos_candidates = sorted_idx[1:max(2, n//5)]  # Skip self
            pos_idx = random.choice(pos_candidates)

            # Negative: bottom 50%
            neg_candidates = sorted_idx[n//2:]
            neg_idx = random.choice(neg_candidates)

            triplets.append((anchor_idx, pos_idx, neg_idx))

        return triplets

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        anchor_idx, pos_idx, neg_idx = self.triplets[idx]

        def encode(concept):
            template = random.choice(TEMPLATES)
            text = template.format(concept=concept)
            enc = self.tokenizer(text, max_length=self.max_length,
                                padding="max_length", truncation=True,
                                return_tensors="pt")
            return enc["input_ids"].squeeze(0), enc["attention_mask"].squeeze(0)

        anchor_ids, anchor_mask = encode(self.concepts[anchor_idx])
        pos_ids, pos_mask = encode(self.concepts[pos_idx])
        neg_ids, neg_mask = encode(self.concepts[neg_idx])

        return {
            "anchor_ids": anchor_ids,
            "anchor_mask": anchor_mask,
            "anchor_vec": self.direction_vectors[self.concepts[anchor_idx]],
            "pos_ids": pos_ids,
            "pos_mask": pos_mask,
            "pos_vec": self.direction_vectors[self.concepts[pos_idx]],
            "neg_ids": neg_ids,
            "neg_mask": neg_mask,
            "neg_vec": self.direction_vectors[self.concepts[neg_idx]],
        }


# ============================================================================
# Training with structured losses
# ============================================================================

class StructuredTrainer:
    def __init__(self, model, config: Config):
        self.model = model
        self.config = config
        self.device = config.device

        self.model.to(self.device)

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

    def cosine_loss(self, predictions, targets):
        """Direct vector matching loss."""
        pred_norm = F.normalize(predictions, p=2, dim=-1)
        target_norm = F.normalize(targets, p=2, dim=-1)
        cos_sim = (pred_norm * target_norm).sum(dim=-1)
        return (1 - cos_sim).mean(), cos_sim.mean()

    def triplet_loss(self, anchor_out, pos_out, neg_out,
                     anchor_target, pos_target, neg_target):
        """
        Triplet loss in BOTTLENECK space.

        The key insight: we want the bottleneck representations to preserve
        the relational structure of the teacher space.

        If teacher says: sim(anchor, pos) > sim(anchor, neg)
        Then student bottleneck should also have: sim(anchor, pos) > sim(anchor, neg)
        """
        # Normalize in bottleneck space
        anchor_norm = F.normalize(anchor_out, p=2, dim=-1)
        pos_norm = F.normalize(pos_out, p=2, dim=-1)
        neg_norm = F.normalize(neg_out, p=2, dim=-1)

        # Distances in student space
        pos_dist = 1 - (anchor_norm * pos_norm).sum(dim=-1)  # Cosine distance
        neg_dist = 1 - (anchor_norm * neg_norm).sum(dim=-1)

        # Triplet loss: pos should be closer than neg by margin
        loss = F.relu(pos_dist - neg_dist + self.config.triplet_margin).mean()

        # Also track if ordering is correct
        correct = (pos_dist < neg_dist).float().mean()

        return loss, correct

    def train_epoch(self, concept_loader, triplet_loader):
        self.model.train()

        total_cosine_loss = 0
        total_triplet_loss = 0
        total_cos_sim = 0
        total_ordering = 0
        num_batches = 0

        # Zip the two loaders
        for concept_batch, triplet_batch in zip(concept_loader, triplet_loader):
            # === Cosine loss on concepts ===
            input_ids = concept_batch["input_ids"].to(self.device)
            attention_mask = concept_batch["attention_mask"].to(self.device)
            targets = concept_batch["target_vector"].to(self.device)

            output, bottleneck = self.model(input_ids, attention_mask)
            cos_loss, cos_sim = self.cosine_loss(output, targets)

            # === Triplet loss on relationships ===
            anchor_out, _ = self.model(
                triplet_batch["anchor_ids"].to(self.device),
                triplet_batch["anchor_mask"].to(self.device)
            )
            pos_out, _ = self.model(
                triplet_batch["pos_ids"].to(self.device),
                triplet_batch["pos_mask"].to(self.device)
            )
            neg_out, _ = self.model(
                triplet_batch["neg_ids"].to(self.device),
                triplet_batch["neg_mask"].to(self.device)
            )

            trip_loss, ordering = self.triplet_loss(
                anchor_out, pos_out, neg_out,
                triplet_batch["anchor_vec"].to(self.device),
                triplet_batch["pos_vec"].to(self.device),
                triplet_batch["neg_vec"].to(self.device),
            )

            # Combined loss
            loss = (self.config.cosine_weight * cos_loss +
                    self.config.contrastive_weight * trip_loss)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_cosine_loss += cos_loss.item()
            total_triplet_loss += trip_loss.item()
            total_cos_sim += cos_sim.item()
            total_ordering += ordering.item()
            num_batches += 1

        return {
            "cosine_loss": total_cosine_loss / num_batches,
            "triplet_loss": total_triplet_loss / num_batches,
            "cos_sim": total_cos_sim / num_batches,
            "ordering_acc": total_ordering / num_batches,
        }

    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval()

        total_cos_sim = 0
        num_batches = 0

        for batch in loader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            targets = batch["target_vector"].to(self.device)

            output, _ = self.model(input_ids, attention_mask)
            _, cos_sim = self.cosine_loss(output, targets)

            total_cos_sim += cos_sim.item()
            num_batches += 1

        return total_cos_sim / num_batches


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser()

    default_vectors = Path(__file__).parent.parent.parent / "data" / "vectors" / "qwen3_vectors_qwen3_1.7b.pt"
    parser.add_argument("--vectors", type=str, default=str(default_vectors))
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--num-concepts", type=int, default=500)
    parser.add_argument("--bottleneck-dim", type=int, default=32)

    default_output = Path(__file__).parent.parent.parent / "results"
    parser.add_argument("--output-dir", type=str, default=str(default_output))

    args = parser.parse_args()

    print("="*70)
    print("STRUCTURED LEARNING EXPERIMENT")
    print("="*70)
    print("\nTechniques:")
    print("  1. Bottleneck architecture (forces compression)")
    print("  2. Triplet contrastive loss (learns relational structure)")
    print("  3. Combined cosine + triplet training")

    # Load vectors
    print(f"\nLoading vectors from {args.vectors}...")
    data = torch.load(args.vectors)

    print(f"  Model: {data['model']}")
    print(f"  Teacher dim: {data['hidden_dim']}")
    print(f"  Available concepts: {data['num_concepts']}")

    all_concepts = data["concepts"][:args.num_concepts]
    direction_vectors = data["direction_vectors"]
    output_dim = data["hidden_dim"]

    random.seed(42)
    random.shuffle(all_concepts)

    # Split
    n = len(all_concepts)
    n_train = int(n * 0.7)
    n_val = int(n * 0.15)

    train_concepts = all_concepts[:n_train]
    val_concepts = all_concepts[n_train:n_train+n_val]
    test_concepts = all_concepts[n_train+n_val:]

    print(f"\nData split:")
    print(f"  Train: {len(train_concepts)} concepts")
    print(f"  Val: {len(val_concepts)} concepts")
    print(f"  Test: {len(test_concepts)} concepts")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(data["model_id"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Config
    config = Config()
    config.num_epochs = args.epochs
    config.bottleneck_dim = args.bottleneck_dim

    # Datasets
    train_concept_ds = ConceptDataset(train_concepts, direction_vectors, tokenizer)
    train_triplet_ds = TripletDataset(train_concepts, direction_vectors, tokenizer,
                                       num_triplets=len(train_concept_ds))
    val_ds = ConceptDataset(val_concepts, direction_vectors, tokenizer)
    test_ds = ConceptDataset(test_concepts, direction_vectors, tokenizer)

    train_concept_loader = DataLoader(train_concept_ds, batch_size=config.batch_size, shuffle=True)
    train_triplet_loader = DataLoader(train_triplet_ds, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size)
    test_loader = DataLoader(test_ds, batch_size=config.batch_size)

    # Model
    model = BottleneckModel(
        vocab_size=len(tokenizer),
        embed_dim=config.embed_dim,
        bottleneck_dim=config.bottleneck_dim,
        hidden_dim=config.hidden_dim,
        output_dim=output_dim,
    )

    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel:")
    print(f"  Embed dim: {config.embed_dim}")
    print(f"  Bottleneck dim: {config.bottleneck_dim} (NARROW)")
    print(f"  Hidden dim: {config.hidden_dim}")
    print(f"  Output dim: {output_dim}")
    print(f"  Total params: {num_params:,}")
    print(f"  Params per concept: {num_params / len(train_concepts):.0f}")

    # Training
    trainer = StructuredTrainer(model, config)

    print(f"\nTraining for {config.num_epochs} epochs...")
    print(f"  Cosine weight: {config.cosine_weight}")
    print(f"  Contrastive weight: {config.contrastive_weight}")
    print(f"  Triplet margin: {config.triplet_margin}")

    history = {"train_cos": [], "val_cos": [], "ordering": []}
    best_val = -1
    best_epoch = 0

    for epoch in range(1, config.num_epochs + 1):
        metrics = trainer.train_epoch(train_concept_loader, train_triplet_loader)
        history["train_cos"].append(metrics["cos_sim"])
        history["ordering"].append(metrics["ordering_acc"])

        if epoch % 10 == 0:
            val_cos = trainer.evaluate(val_loader)
            history["val_cos"].append(val_cos)

            print(f"Epoch {epoch}: train_cos={metrics['cos_sim']:.4f}, "
                  f"val_cos={val_cos:.4f}, ordering={metrics['ordering_acc']:.4f}")

            if val_cos > best_val:
                best_val = val_cos
                best_epoch = epoch

    # Final test
    test_cos = trainer.evaluate(test_loader)

    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"Best validation cos_sim: {best_val:.4f} (epoch {best_epoch})")
    print(f"Test cosine similarity: {test_cos:.4f}")
    print(f"Final ordering accuracy: {history['ordering'][-1]:.4f}")

    # Generalization gap
    gap = history["train_cos"][-1] - test_cos
    print(f"Generalization gap: {gap:.4f}")

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    results = {
        "config": {
            "num_concepts": args.num_concepts,
            "bottleneck_dim": config.bottleneck_dim,
            "hidden_dim": config.hidden_dim,
            "epochs": config.num_epochs,
        },
        "results": {
            "best_val_cos_sim": best_val,
            "test_cos_sim": test_cos,
            "final_ordering_acc": history["ordering"][-1],
            "generalization_gap": gap,
            "num_params": num_params,
        }
    }

    results_file = output_dir / f"structured_learning_{timestamp}.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax1 = axes[0]
    ax1.plot(history["train_cos"], label="Train", alpha=0.7)
    epochs_val = list(range(9, len(history["train_cos"]), 10))
    ax1.plot(epochs_val, history["val_cos"], 'o-', label="Val")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cosine Similarity")
    ax1.set_title("Training Progress")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.plot(history["ordering"], label="Triplet Ordering Accuracy")
    ax2.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label="Random baseline")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Relational Structure Learning")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    plot_file = output_dir / f"structured_learning_{timestamp}.png"
    plt.savefig(plot_file, dpi=150)
    print(f"\nPlots saved to {plot_file}")
    print(f"Results saved to {results_file}")


if __name__ == "__main__":
    main()
