"""
Scaling Law Experiment: Vary # concepts and model size to find optimal configurations.

This script runs a grid search over:
- Number of training concepts: [100, 200, 400, 700, 1000]
- Student model hidden dimensions: [128, 256, 512, 1024]
- Student model layers: [2, 4, 6]

Usage:
    python run_scaling_experiment.py --vectors ../../data/vectors/qwen3_vectors_qwen3_1.7b.pt
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
class ExperimentConfig:
    # Scaling parameters to sweep
    num_concepts_list: list = None
    hidden_dim_list: list = None
    num_layers_list: list = None

    # Fixed training params
    batch_size: int = 64
    learning_rate: float = 3e-4
    num_epochs: int = 100
    weight_decay: float = 0.1
    dropout: float = 0.3
    label_smoothing: float = 0.1

    # Data
    train_split: float = 0.7
    val_split: float = 0.15
    sentences_per_concept: int = 10

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def __post_init__(self):
        if self.num_concepts_list is None:
            # Much larger concept counts for 50K dataset
            self.num_concepts_list = [1000, 5000, 10000, 25000, 50000]
        if self.hidden_dim_list is None:
            self.hidden_dim_list = [128, 256, 512]
        if self.num_layers_list is None:
            self.num_layers_list = [2, 4]

# ============================================================================
# Model
# ============================================================================

class MLPStudent(nn.Module):
    """MLP-based student model."""

    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int,
                 output_dim: int, num_layers: int, dropout: float = 0.3):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)

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

        # Final projection to match teacher dimension
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.mlp = nn.Sequential(*layers)
        self.output_dim = output_dim

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

def compute_loss(predictions, targets, label_smoothing=0.0):
    """Compute cosine similarity loss."""
    pred_norm = F.normalize(predictions, p=2, dim=-1)
    target_norm = F.normalize(targets, p=2, dim=-1)

    cos_sim = (pred_norm * target_norm).sum(dim=-1)
    target_sim = 1.0 - label_smoothing
    loss = (target_sim - cos_sim).pow(2).mean()

    return loss, cos_sim.mean()


def train_single_config(
    num_concepts: int,
    hidden_dim: int,
    num_layers: int,
    all_concepts: list,
    direction_vectors: dict,
    tokenizer,
    teacher_dim: int,
    config: ExperimentConfig,
):
    """Train a single configuration and return results."""

    device = config.device

    # Select subset of concepts
    concepts = all_concepts[:num_concepts]

    # Split
    n = len(concepts)
    n_train = int(n * config.train_split)
    n_val = int(n * config.val_split)

    train_concepts = concepts[:n_train]
    val_concepts = concepts[n_train:n_train+n_val]
    test_concepts = concepts[n_train+n_val:]

    # Create datasets
    train_dataset = ConceptDataset(
        train_concepts, direction_vectors, tokenizer,
        sentences_per_concept=config.sentences_per_concept
    )
    val_dataset = ConceptDataset(
        val_concepts, direction_vectors, tokenizer,
        sentences_per_concept=3
    )
    test_dataset = ConceptDataset(
        test_concepts, direction_vectors, tokenizer,
        sentences_per_concept=3
    )

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size)

    # Create model
    vocab_size = len(tokenizer)
    embed_dim = min(256, hidden_dim)

    model = MLPStudent(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        output_dim=teacher_dim,
        num_layers=num_layers,
        dropout=config.dropout,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    # Training loop
    best_val_cos_sim = -1
    best_epoch = 0
    train_history = []
    val_history = []

    for epoch in range(1, config.num_epochs + 1):
        # Train
        model.train()
        total_loss = 0
        total_cos_sim = 0
        num_batches = 0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = batch["target_vector"].to(device)

            predictions = model(input_ids, attention_mask)
            loss, cos_sim = compute_loss(predictions, targets, config.label_smoothing)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            total_cos_sim += cos_sim.item()
            num_batches += 1

        train_loss = total_loss / num_batches
        train_cos_sim = total_cos_sim / num_batches
        train_history.append(train_cos_sim)

        # Validate every 10 epochs
        if epoch % 10 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                total_cos_sim = 0
                num_batches = 0

                for batch in val_loader:
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    targets = batch["target_vector"].to(device)

                    predictions = model(input_ids, attention_mask)
                    _, cos_sim = compute_loss(predictions, targets)

                    total_cos_sim += cos_sim.item()
                    num_batches += 1

                val_cos_sim = total_cos_sim / num_batches
                val_history.append(val_cos_sim)

                if val_cos_sim > best_val_cos_sim:
                    best_val_cos_sim = val_cos_sim
                    best_epoch = epoch

    # Final test evaluation
    model.eval()
    with torch.no_grad():
        total_cos_sim = 0
        num_batches = 0

        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = batch["target_vector"].to(device)

            predictions = model(input_ids, attention_mask)
            _, cos_sim = compute_loss(predictions, targets)

            total_cos_sim += cos_sim.item()
            num_batches += 1

        test_cos_sim = total_cos_sim / num_batches

    return {
        "num_concepts": num_concepts,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "num_params": num_params,
        "train_concepts": len(train_concepts),
        "test_concepts": len(test_concepts),
        "best_val_cos_sim": best_val_cos_sim,
        "best_epoch": best_epoch,
        "test_cos_sim": test_cos_sim,
        "final_train_cos_sim": train_history[-1] if train_history else 0,
        "train_history": train_history,
        "val_history": val_history,
    }

# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser()

    # Default vectors path
    default_vectors = Path(__file__).parent.parent.parent / "data" / "vectors" / "qwen3_vectors_qwen3_1.7b.pt"

    parser.add_argument("--vectors", type=str, default=str(default_vectors),
                        help="Path to extracted vectors (.pt file)")
    parser.add_argument("--epochs", type=int, default=100)

    default_output = Path(__file__).parent.parent.parent / "results"
    parser.add_argument("--output-dir", type=str, default=str(default_output))

    args = parser.parse_args()

    print("="*70)
    print("SCALING LAW EXPERIMENT")
    print("="*70)

    # Load extracted vectors
    print(f"\nLoading vectors from {args.vectors}...")
    data = torch.load(args.vectors)

    print(f"  Model: {data['model']}")
    print(f"  Hidden dim: {data['hidden_dim']}")
    print(f"  Num concepts: {data['num_concepts']}")

    all_concepts = data["concepts"]
    direction_vectors = data["direction_vectors"]
    teacher_dim = data["hidden_dim"]

    # Shuffle concepts once at the start for consistent splits
    random.seed(42)
    random.shuffle(all_concepts)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(data["model_id"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"  Tokenizer vocab size: {len(tokenizer)}")

    # Configuration
    config = ExperimentConfig()
    config.num_epochs = args.epochs

    # Filter concept counts to available concepts
    max_concepts = len(all_concepts)
    config.num_concepts_list = [n for n in config.num_concepts_list if n <= max_concepts]

    print(f"\nExperiment grid:")
    print(f"  Concept counts: {config.num_concepts_list}")
    print(f"  Hidden dims: {config.hidden_dim_list}")
    print(f"  Num layers: {config.num_layers_list}")

    total_experiments = (
        len(config.num_concepts_list) *
        len(config.hidden_dim_list) *
        len(config.num_layers_list)
    )
    print(f"  Total experiments: {total_experiments}")

    # Run experiments
    results = []
    experiment_num = 0

    for num_concepts, hidden_dim, num_layers in itertools.product(
        config.num_concepts_list,
        config.hidden_dim_list,
        config.num_layers_list
    ):
        experiment_num += 1
        print(f"\n[{experiment_num}/{total_experiments}] "
              f"concepts={num_concepts}, hidden={hidden_dim}, layers={num_layers}")

        result = train_single_config(
            num_concepts=num_concepts,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            all_concepts=all_concepts,
            direction_vectors=direction_vectors,
            tokenizer=tokenizer,
            teacher_dim=teacher_dim,
            config=config,
        )

        results.append(result)

        print(f"  Params: {result['num_params']:,}")
        print(f"  Best val cos_sim: {result['best_val_cos_sim']:.4f} (epoch {result['best_epoch']})")
        print(f"  Test cos_sim: {result['test_cos_sim']:.4f}")

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f"scaling_results_{timestamp}.json"

    # Remove non-serializable items
    for r in results:
        r.pop("train_history", None)
        r.pop("val_history", None)

    with open(results_file, "w") as f:
        json.dump({
            "config": asdict(config),
            "teacher_model": data["model"],
            "teacher_dim": teacher_dim,
            "results": results,
            "timestamp": timestamp,
        }, f, indent=2)

    print(f"\nResults saved to {results_file}")

    # Print summary table
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    print(f"{'Concepts':<10} {'Hidden':<10} {'Layers':<8} {'Params':<12} {'Val cos':<10} {'Test cos':<10}")
    print("-"*70)

    # Sort by test cosine similarity
    sorted_results = sorted(results, key=lambda x: x["test_cos_sim"], reverse=True)

    for r in sorted_results:
        print(f"{r['num_concepts']:<10} {r['hidden_dim']:<10} {r['num_layers']:<8} "
              f"{r['num_params']:<12,} {r['best_val_cos_sim']:<10.4f} {r['test_cos_sim']:<10.4f}")

    print("-"*70)
    print(f"\nBest configuration:")
    best = sorted_results[0]
    print(f"  Concepts: {best['num_concepts']}")
    print(f"  Hidden dim: {best['hidden_dim']}")
    print(f"  Layers: {best['num_layers']}")
    print(f"  Parameters: {best['num_params']:,}")
    print(f"  Test cosine similarity: {best['test_cos_sim']:.4f}")

    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot 1: Test cos_sim vs num_concepts (colored by model size)
    ax1 = axes[0]
    for hidden_dim in config.hidden_dim_list:
        subset = [r for r in results if r["hidden_dim"] == hidden_dim]
        x = [r["num_concepts"] for r in subset]
        y = [r["test_cos_sim"] for r in subset]
        ax1.plot(x, y, 'o-', label=f"hidden={hidden_dim}", markersize=8)
    ax1.set_xlabel("Number of Concepts")
    ax1.set_ylabel("Test Cosine Similarity")
    ax1.set_title("Scaling with Data")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Test cos_sim vs num_params
    ax2 = axes[1]
    for num_concepts in config.num_concepts_list:
        subset = [r for r in results if r["num_concepts"] == num_concepts]
        x = [r["num_params"] for r in subset]
        y = [r["test_cos_sim"] for r in subset]
        ax2.plot(x, y, 'o-', label=f"concepts={num_concepts}", markersize=8)
    ax2.set_xlabel("Number of Parameters")
    ax2.set_ylabel("Test Cosine Similarity")
    ax2.set_title("Scaling with Model Size")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale("log")

    # Plot 3: Heatmap of best configs
    ax3 = axes[2]

    # Create matrix
    concepts_idx = {n: i for i, n in enumerate(config.num_concepts_list)}
    hidden_idx = {h: i for i, h in enumerate(config.hidden_dim_list)}

    matrix = np.zeros((len(config.num_concepts_list), len(config.hidden_dim_list)))
    for r in results:
        i = concepts_idx[r["num_concepts"]]
        j = hidden_idx[r["hidden_dim"]]
        # Average over layers
        matrix[i, j] = max(matrix[i, j], r["test_cos_sim"])

    im = ax3.imshow(matrix, cmap="YlOrRd", aspect="auto")
    ax3.set_xticks(range(len(config.hidden_dim_list)))
    ax3.set_xticklabels(config.hidden_dim_list)
    ax3.set_yticks(range(len(config.num_concepts_list)))
    ax3.set_yticklabels(config.num_concepts_list)
    ax3.set_xlabel("Hidden Dimension")
    ax3.set_ylabel("Number of Concepts")
    ax3.set_title("Test Cosine Similarity (best across layers)")
    plt.colorbar(im, ax=ax3)

    # Add text annotations
    for i in range(len(config.num_concepts_list)):
        for j in range(len(config.hidden_dim_list)):
            ax3.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center", fontsize=9)

    plt.tight_layout()

    plot_file = output_dir / f"scaling_plots_{timestamp}.png"
    plt.savefig(plot_file, dpi=150)
    print(f"Plots saved to {plot_file}")

    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
