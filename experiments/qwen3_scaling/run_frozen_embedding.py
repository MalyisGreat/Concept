"""
Frozen Embedding Experiment: Use Qwen3's pretrained embeddings (frozen) + small MLP.

Key insight: Qwen3's embeddings already encode semantic meaning.
We just need to learn the transformation: embedding_space → direction_vector_space.

This should have WAY fewer parameters since we don't train the embedding layer.
"""

import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
from pathlib import Path
import json
from dataclasses import dataclass, asdict
import random
import numpy as np
from datetime import datetime


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class Config:
    # MLP dimensions to test (these are now TINY since embedding is frozen)
    hidden_dims: list = None
    num_layers_list: list = None

    # Concept counts
    num_concepts_list: list = None

    # Training
    batch_size: int = 64
    learning_rate: float = 1e-3
    num_epochs: int = 100
    weight_decay: float = 0.01
    dropout: float = 0.2

    # Loss weights
    triplet_weight: float = 0.1  # Weight for relationship loss
    triplet_margin: float = 0.2

    # Data splits
    train_split: float = 0.7
    val_split: float = 0.15

    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [128, 256, 512]
        if self.num_layers_list is None:
            self.num_layers_list = [1, 2, 3]
        if self.num_concepts_list is None:
            self.num_concepts_list = [1000, 5000, 10000, 25000, 50000]


# ============================================================================
# Model with Frozen Embeddings
# ============================================================================

class FrozenEmbeddingMLP(nn.Module):
    """
    Uses Qwen3's frozen embedding layer + trainable MLP.

    The embedding layer is NOT trained - it already knows semantics.
    We only train the MLP to transform embedding → direction vector.
    """

    def __init__(self, frozen_embedding: nn.Embedding, hidden_dim: int,
                 output_dim: int, num_layers: int, dropout: float = 0.2):
        super().__init__()

        # Frozen embedding from Qwen3
        self.embedding = frozen_embedding
        self.embedding.requires_grad_(False)  # Freeze!

        embed_dim = frozen_embedding.embedding_dim

        # Small MLP (this is all we train)
        layers = []
        in_dim = embed_dim

        for i in range(num_layers):
            out_dim = hidden_dim if i < num_layers - 1 else hidden_dim
            layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.LayerNorm(out_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            in_dim = out_dim

        self.mlp = nn.Sequential(*layers)

        # Output projection
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, input_ids, attention_mask=None):
        # Get frozen embeddings
        with torch.no_grad():
            x = self.embedding(input_ids)  # [B, L, embed_dim]

        # Mean pool (keep this simple)
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            x = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            x = x.mean(dim=1)

        # MLP transformation
        x = self.mlp(x)

        # Output
        x = self.output(x)

        return x

    def trainable_params(self):
        """Count only trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================================
# Dataset
# ============================================================================

TEMPLATES = [
    "The concept {concept} means",
    "{concept} refers to",
    "Understanding {concept} involves",
    "The word {concept} represents",
    "{concept} is",
]

class ConceptDataset(Dataset):
    def __init__(self, concepts, direction_vectors, tokenizer, max_length=32):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.concepts = concepts
        self.direction_vectors = direction_vectors

        # Pre-compute all (concept, template) pairs
        self.data = []
        for concept in concepts:
            template = random.choice(TEMPLATES)
            sentence = template.format(concept=concept)
            self.data.append((concept, sentence))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        concept, sentence = self.data[idx]

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
            "target_vector": self.direction_vectors[concept],
            "concept": concept,
        }


# ============================================================================
# Training
# ============================================================================

def compute_cosine_loss(predictions, targets):
    """Cosine similarity loss."""
    pred_norm = F.normalize(predictions, p=2, dim=-1)
    target_norm = F.normalize(targets, p=2, dim=-1)
    cos_sim = (pred_norm * target_norm).sum(dim=-1)
    loss = (1.0 - cos_sim).mean()
    return loss, cos_sim.mean()


def compute_triplet_loss(predictions, targets, margin=0.2):
    """
    Triplet loss to encourage learning relationships.
    For each anchor, use a random positive (similar) and negative (different).
    """
    batch_size = predictions.shape[0]
    if batch_size < 3:
        return torch.tensor(0.0, device=predictions.device)

    pred_norm = F.normalize(predictions, p=2, dim=-1)
    target_norm = F.normalize(targets, p=2, dim=-1)

    # Compute target similarities to find positive/negative pairs
    target_sims = torch.mm(target_norm, target_norm.t())

    total_loss = 0
    count = 0

    for i in range(batch_size):
        # Find most similar (positive) and least similar (negative) in batch
        sims = target_sims[i].clone()
        sims[i] = -1  # Exclude self

        pos_idx = sims.argmax().item()
        neg_idx = sims.argmin().item()

        if pos_idx != neg_idx:
            anchor = pred_norm[i]
            positive = pred_norm[pos_idx]
            negative = pred_norm[neg_idx]

            pos_dist = 1 - (anchor * positive).sum()
            neg_dist = 1 - (anchor * negative).sum()

            loss = F.relu(pos_dist - neg_dist + margin)
            total_loss += loss
            count += 1

    if count > 0:
        return total_loss / count
    return torch.tensor(0.0, device=predictions.device)


def train_single_config(
    num_concepts: int,
    hidden_dim: int,
    num_layers: int,
    all_concepts: list,
    direction_vectors: dict,
    frozen_embedding: nn.Embedding,
    tokenizer,
    output_dim: int,
    config: Config,
):
    """Train a single configuration."""

    device = config.device

    # Select concepts
    concepts = all_concepts[:num_concepts]

    # Split
    n = len(concepts)
    n_train = int(n * config.train_split)
    n_val = int(n * config.val_split)

    train_concepts = concepts[:n_train]
    val_concepts = concepts[n_train:n_train+n_val]
    test_concepts = concepts[n_train+n_val:]

    # Datasets
    train_dataset = ConceptDataset(train_concepts, direction_vectors, tokenizer)
    val_dataset = ConceptDataset(val_concepts, direction_vectors, tokenizer)
    test_dataset = ConceptDataset(test_concepts, direction_vectors, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size)

    # Create model with frozen embedding
    model = FrozenEmbeddingMLP(
        frozen_embedding=frozen_embedding,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_layers=num_layers,
        dropout=config.dropout,
    ).to(device)

    trainable_params = model.trainable_params()

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.num_epochs
    )

    # Training
    best_val_cos_sim = -1
    best_epoch = 0

    for epoch in range(1, config.num_epochs + 1):
        # Train
        model.train()
        train_cos_sim = 0
        num_batches = 0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = batch["target_vector"].to(device)

            predictions = model(input_ids, attention_mask)

            # Combined loss
            cos_loss, cos_sim = compute_cosine_loss(predictions, targets)
            triplet_loss = compute_triplet_loss(predictions, targets, config.triplet_margin)

            loss = cos_loss + config.triplet_weight * triplet_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_cos_sim += cos_sim.item()
            num_batches += 1

        scheduler.step()
        train_cos_sim /= num_batches

        # Validate every 10 epochs
        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_cos_sim = 0
                num_batches = 0

                for batch in val_loader:
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    targets = batch["target_vector"].to(device)

                    predictions = model(input_ids, attention_mask)
                    _, cos_sim = compute_cosine_loss(predictions, targets)

                    val_cos_sim += cos_sim.item()
                    num_batches += 1

                val_cos_sim /= num_batches

                if val_cos_sim > best_val_cos_sim:
                    best_val_cos_sim = val_cos_sim
                    best_epoch = epoch

    # Test
    model.eval()
    with torch.no_grad():
        test_cos_sim = 0
        num_batches = 0

        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = batch["target_vector"].to(device)

            predictions = model(input_ids, attention_mask)
            _, cos_sim = compute_cosine_loss(predictions, targets)

            test_cos_sim += cos_sim.item()
            num_batches += 1

        test_cos_sim /= num_batches

    return {
        "num_concepts": num_concepts,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "trainable_params": trainable_params,
        "params_per_concept": trainable_params / num_concepts,
        "best_val_cos_sim": best_val_cos_sim,
        "best_epoch": best_epoch,
        "test_cos_sim": test_cos_sim,
        "train_concepts": len(train_concepts),
        "test_concepts": len(test_concepts),
        "generalization_gap": best_val_cos_sim - test_cos_sim,
    }


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser()

    default_vectors = Path(__file__).parent.parent.parent / "data" / "vectors" / "qwen3_vectors_qwen3_0.6b_50000concepts.pt"
    parser.add_argument("--vectors", type=str, default=str(default_vectors))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--triplet-weight", type=float, default=0.1)

    default_output = Path(__file__).parent.parent.parent / "results"
    parser.add_argument("--output-dir", type=str, default=str(default_output))

    args = parser.parse_args()

    print("="*70)
    print("FROZEN EMBEDDING EXPERIMENT")
    print("="*70)
    print("Using Qwen3's pretrained embeddings (FROZEN) + small trainable MLP")
    print("="*70)

    # Load vectors
    print(f"\nLoading vectors from {args.vectors}...")
    data = torch.load(args.vectors)

    print(f"  Model: {data['model']}")
    print(f"  Hidden dim: {data['hidden_dim']}")
    print(f"  Num concepts: {data['num_concepts']}")

    all_concepts = data["concepts"]
    direction_vectors = data["direction_vectors"]
    output_dim = data["hidden_dim"]
    model_id = data["model_id"]

    random.seed(42)
    random.shuffle(all_concepts)

    # Load Qwen3 model to get frozen embeddings
    print(f"\nLoading Qwen3 embeddings from {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load only the embedding layer
    qwen_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True
    )

    # Extract and copy the embedding layer
    frozen_embedding = nn.Embedding(
        qwen_model.model.embed_tokens.num_embeddings,
        qwen_model.model.embed_tokens.embedding_dim
    )
    frozen_embedding.weight.data = qwen_model.model.embed_tokens.weight.data.clone()
    frozen_embedding = frozen_embedding.to(args.vectors.split("cuda")[0] if "cuda" in str(args.vectors) else "cuda" if torch.cuda.is_available() else "cpu")

    # Delete the full model to save memory
    del qwen_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    embed_dim = frozen_embedding.embedding_dim
    print(f"  Embedding dim: {embed_dim}")
    print(f"  Vocab size: {frozen_embedding.num_embeddings}")
    print(f"  Embedding params (FROZEN): {frozen_embedding.num_embeddings * embed_dim:,}")

    # Config
    config = Config()
    config.num_epochs = args.epochs
    config.triplet_weight = args.triplet_weight

    print(f"\nExperiment grid:")
    print(f"  Concept counts: {config.num_concepts_list}")
    print(f"  Hidden dims: {config.hidden_dims}")
    print(f"  Num layers: {config.num_layers_list}")
    print(f"  Triplet loss weight: {config.triplet_weight}")

    # Generate all configs
    import itertools
    configs = list(itertools.product(
        config.num_concepts_list,
        config.hidden_dims,
        config.num_layers_list,
    ))

    print(f"  Total experiments: {len(configs)}")

    # Move embedding to device
    device = config.device
    frozen_embedding = frozen_embedding.to(device)

    # Run experiments
    results = []

    for i, (num_concepts, hidden_dim, num_layers) in enumerate(configs):
        print(f"\n[{i+1}/{len(configs)}] concepts={num_concepts}, hidden={hidden_dim}, layers={num_layers}")

        result = train_single_config(
            num_concepts=num_concepts,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            all_concepts=all_concepts,
            direction_vectors=direction_vectors,
            frozen_embedding=frozen_embedding,
            tokenizer=tokenizer,
            output_dim=output_dim,
            config=config,
        )

        results.append(result)

        print(f"  Trainable params: {result['trainable_params']:,} ({result['params_per_concept']:.1f}/concept)")
        print(f"  Val cos_sim: {result['best_val_cos_sim']:.4f}, Test: {result['test_cos_sim']:.4f}")
        print(f"  Gap: {result['generalization_gap']:.4f}")

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f"frozen_embedding_results_{timestamp}.json"

    with open(results_file, "w") as f:
        json.dump({
            "config": asdict(config),
            "teacher_model": data["model"],
            "teacher_dim": output_dim,
            "embed_dim": embed_dim,
            "results": results,
        }, f, indent=2)

    print(f"\nResults saved to {results_file}")

    # Summary
    print("\n" + "="*70)
    print("RESULTS SUMMARY (sorted by test cos_sim)")
    print("="*70)
    print(f"{'Concepts':<10} {'Hidden':<8} {'Layers':<8} {'Params':<12} {'P/C':<10} {'Val':<10} {'Test':<10} {'Gap':<10}")
    print("-"*88)

    sorted_results = sorted(results, key=lambda x: x["test_cos_sim"], reverse=True)

    for r in sorted_results[:15]:
        print(f"{r['num_concepts']:<10} {r['hidden_dim']:<8} {r['num_layers']:<8} "
              f"{r['trainable_params']:<12,} {r['params_per_concept']:<10.1f} "
              f"{r['best_val_cos_sim']:<10.4f} {r['test_cos_sim']:<10.4f} {r['generalization_gap']:<10.4f}")

    print("-"*88)

    best = sorted_results[0]
    print(f"\nBest config:")
    print(f"  Concepts: {best['num_concepts']}, Hidden: {best['hidden_dim']}, Layers: {best['num_layers']}")
    print(f"  Trainable params: {best['trainable_params']:,} ({best['params_per_concept']:.1f}/concept)")
    print(f"  Test cosine sim: {best['test_cos_sim']:.4f}")
    print(f"  Generalization gap: {best['generalization_gap']:.4f}")

    # Compare params
    print(f"\n  vs Original approach: ~19M params → {best['trainable_params']:,} params")
    print(f"  Parameter reduction: {19_000_000 / best['trainable_params']:.1f}x smaller")


if __name__ == "__main__":
    main()
