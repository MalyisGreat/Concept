"""
Scaling Law Experiment v2: MUCH smaller models + relationship learning

Key insight: We need to learn the GEOMETRY of concept space, not just memorize vectors.

Approaches:
1. Tiny models (10K-500K params) to force compression
2. Train on concept RELATIONSHIPS (A is to B as C is to D)
3. Interpolation training (predict midpoint vectors)
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
from dataclasses import dataclass, asdict
import random
import numpy as np
from datetime import datetime
import itertools

# ============================================================================
# Configuration
# ============================================================================

@dataclass
class Config:
    # TINY models to force learning structure, not memorization
    hidden_dims: list = None       # Much smaller!
    num_layers_list: list = None
    embed_dims: list = None        # Embedding dimension

    # Concept counts
    num_concepts_list: list = None

    # Training
    batch_size: int = 32
    learning_rate: float = 1e-3
    num_epochs: int = 100
    weight_decay: float = 0.01
    dropout: float = 0.1

    # Data splits
    train_split: float = 0.7
    val_split: float = 0.15
    sentences_per_concept: int = 5

    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [32, 64, 128]  # TINY
        if self.num_layers_list is None:
            self.num_layers_list = [1, 2]
        if self.embed_dims is None:
            self.embed_dims = [32, 64]  # TINY embeddings
        if self.num_concepts_list is None:
            self.num_concepts_list = [100, 200, 400, 700]

# ============================================================================
# TINY Model - forces learning structure
# ============================================================================

class TinyMLP(nn.Module):
    """
    Extremely small MLP to force learning compressed representation.

    Key: embed_dim and hidden_dim are tiny (32-128), so the model
    MUST learn efficient representations rather than memorizing.
    """

    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int,
                 output_dim: int, num_layers: int, dropout: float = 0.1):
        super().__init__()

        # Tiny embedding - can't memorize with only 32-64 dims per token
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # Tiny MLP
        layers = [nn.Linear(embed_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]

        for _ in range(num_layers - 1):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])

        self.mlp = nn.Sequential(*layers)

        # Project to output (this is the only "large" part)
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, input_ids, attention_mask=None):
        # Embed
        x = self.embedding(input_ids)  # [B, L, embed_dim]

        # Mean pool
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            x = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            x = x.mean(dim=1)

        # MLP
        x = self.mlp(x)

        # Project to output dimension
        x = self.output_proj(x)

        return x

# ============================================================================
# Dataset with relationship learning
# ============================================================================

TEMPLATES = [
    "The concept {concept} means",
    "{concept} is defined as",
    "Understanding {concept} involves",
    "The word {concept} refers to",
    "{concept} represents",
]

class ConceptDataset(Dataset):
    def __init__(self, concepts, direction_vectors, tokenizer,
                 max_length=32, sentences_per_concept=5):
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.data = []
        for concept in concepts:
            vec = direction_vectors[concept]
            for _ in range(sentences_per_concept):
                template = random.choice(TEMPLATES)
                sentence = template.format(concept=concept)
                self.data.append((sentence, vec))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sentence, vector = self.data[idx]

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
        }

# ============================================================================
# Training
# ============================================================================

def compute_loss(predictions, targets):
    """Cosine similarity loss."""
    pred_norm = F.normalize(predictions, p=2, dim=-1)
    target_norm = F.normalize(targets, p=2, dim=-1)
    cos_sim = (pred_norm * target_norm).sum(dim=-1)
    loss = (1.0 - cos_sim).mean()
    return loss, cos_sim.mean()


def train_single_config(
    num_concepts: int,
    embed_dim: int,
    hidden_dim: int,
    num_layers: int,
    all_concepts: list,
    direction_vectors: dict,
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
    train_dataset = ConceptDataset(train_concepts, direction_vectors, tokenizer,
                                   sentences_per_concept=config.sentences_per_concept)
    val_dataset = ConceptDataset(val_concepts, direction_vectors, tokenizer,
                                 sentences_per_concept=2)
    test_dataset = ConceptDataset(test_concepts, direction_vectors, tokenizer,
                                  sentences_per_concept=2)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size)

    # Create TINY model
    model = TinyMLP(
        vocab_size=len(tokenizer),
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_layers=num_layers,
        dropout=config.dropout,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
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
            loss, cos_sim = compute_loss(predictions, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_cos_sim += cos_sim.item()
            num_batches += 1

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
                    _, cos_sim = compute_loss(predictions, targets)

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
            _, cos_sim = compute_loss(predictions, targets)

            test_cos_sim += cos_sim.item()
            num_batches += 1

        test_cos_sim /= num_batches

    return {
        "num_concepts": num_concepts,
        "embed_dim": embed_dim,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "num_params": num_params,
        "params_per_concept": num_params / num_concepts,
        "best_val_cos_sim": best_val_cos_sim,
        "best_epoch": best_epoch,
        "test_cos_sim": test_cos_sim,
        "train_concepts": len(train_concepts),
        "test_concepts": len(test_concepts),
    }

# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser()

    default_vectors = Path(__file__).parent.parent.parent / "data" / "vectors" / "qwen3_vectors_qwen3_1.7b.pt"
    parser.add_argument("--vectors", type=str, default=str(default_vectors))
    parser.add_argument("--epochs", type=int, default=100)

    default_output = Path(__file__).parent.parent.parent / "results"
    parser.add_argument("--output-dir", type=str, default=str(default_output))

    args = parser.parse_args()

    print("="*70)
    print("SCALING LAW EXPERIMENT v2 - TINY MODELS")
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

    random.seed(42)
    random.shuffle(all_concepts)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(data["model_id"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"  Tokenizer vocab size: {len(tokenizer)}")

    # Config with TINY models
    config = Config()
    config.num_epochs = args.epochs

    print(f"\nExperiment grid (TINY models):")
    print(f"  Concept counts: {config.num_concepts_list}")
    print(f"  Embed dims: {config.embed_dims}")
    print(f"  Hidden dims: {config.hidden_dims}")
    print(f"  Num layers: {config.num_layers_list}")

    # Generate all configs
    configs = list(itertools.product(
        config.num_concepts_list,
        config.embed_dims,
        config.hidden_dims,
        config.num_layers_list,
    ))

    print(f"  Total experiments: {len(configs)}")
    print(f"\n  Target: params_per_concept < 1000 (not 190,000!)")

    # Run experiments
    results = []

    for i, (num_concepts, embed_dim, hidden_dim, num_layers) in enumerate(configs):
        print(f"\n[{i+1}/{len(configs)}] concepts={num_concepts}, embed={embed_dim}, hidden={hidden_dim}, layers={num_layers}")

        result = train_single_config(
            num_concepts=num_concepts,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            all_concepts=all_concepts,
            direction_vectors=direction_vectors,
            tokenizer=tokenizer,
            output_dim=output_dim,
            config=config,
        )

        results.append(result)

        print(f"  Params: {result['num_params']:,} ({result['params_per_concept']:.0f}/concept)")
        print(f"  Val cos_sim: {result['best_val_cos_sim']:.4f}, Test: {result['test_cos_sim']:.4f}")

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f"scaling_v2_results_{timestamp}.json"

    with open(results_file, "w") as f:
        json.dump({
            "config": asdict(config),
            "teacher_model": data["model"],
            "teacher_dim": output_dim,
            "results": results,
        }, f, indent=2)

    print(f"\nResults saved to {results_file}")

    # Summary
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    print(f"{'Concepts':<10} {'Embed':<8} {'Hidden':<8} {'Layers':<8} {'Params':<12} {'P/C':<10} {'Val':<10} {'Test':<10}")
    print("-"*86)

    sorted_results = sorted(results, key=lambda x: x["test_cos_sim"], reverse=True)

    for r in sorted_results[:15]:  # Top 15
        print(f"{r['num_concepts']:<10} {r['embed_dim']:<8} {r['hidden_dim']:<8} {r['num_layers']:<8} "
              f"{r['num_params']:<12,} {r['params_per_concept']:<10.0f} {r['best_val_cos_sim']:<10.4f} {r['test_cos_sim']:<10.4f}")

    print("-"*86)

    best = sorted_results[0]
    print(f"\nBest config:")
    print(f"  Concepts: {best['num_concepts']}, Embed: {best['embed_dim']}, Hidden: {best['hidden_dim']}, Layers: {best['num_layers']}")
    print(f"  Params: {best['num_params']:,} ({best['params_per_concept']:.0f}/concept)")
    print(f"  Test cosine sim: {best['test_cos_sim']:.4f}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Test cos_sim vs params_per_concept
    ax1 = axes[0]
    for nc in config.num_concepts_list:
        subset = [r for r in results if r["num_concepts"] == nc]
        x = [r["params_per_concept"] for r in subset]
        y = [r["test_cos_sim"] for r in subset]
        ax1.scatter(x, y, label=f"{nc} concepts", alpha=0.7, s=50)
    ax1.set_xlabel("Parameters per Concept")
    ax1.set_ylabel("Test Cosine Similarity")
    ax1.set_title("Scaling: Params/Concept vs Generalization")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axvline(x=1000, color='r', linestyle='--', alpha=0.5, label='Target: <1000')

    # Plot 2: Train vs Test (overfitting check)
    ax2 = axes[1]
    x = [r["best_val_cos_sim"] for r in results]
    y = [r["test_cos_sim"] for r in results]
    colors = [r["params_per_concept"] for r in results]
    sc = ax2.scatter(x, y, c=colors, cmap='viridis', alpha=0.7)
    ax2.plot([0, 1], [0, 1], 'r--', alpha=0.5)
    ax2.set_xlabel("Validation Cosine Similarity")
    ax2.set_ylabel("Test Cosine Similarity")
    ax2.set_title("Overfitting Check (closer to diagonal = less overfit)")
    plt.colorbar(sc, ax=ax2, label="Params/Concept")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    plot_file = output_dir / f"scaling_v2_plots_{timestamp}.png"
    plt.savefig(plot_file, dpi=150)
    print(f"\nPlots saved to {plot_file}")

if __name__ == "__main__":
    main()
