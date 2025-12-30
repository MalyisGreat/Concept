"""
Compare different approaches with SMALL models on ALL 50K concepts.

Approaches:
1. Trainable embeddings (tiny embed dim)
2. Frozen Qwen3 embeddings + tiny MLP
3. Linear probe (just linear layer on frozen embeddings)

All use minimal parameters to test if structure can be learned.
"""

import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import argparse
from pathlib import Path
import random
from datetime import datetime
import json


# ============================================================================
# Models
# ============================================================================

class TinyTrainableModel(nn.Module):
    """Approach 1: Tiny trainable embeddings."""
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids)
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            x = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            x = x.mean(dim=1)
        return self.mlp(x)


class FrozenEmbedMLP(nn.Module):
    """Approach 2: Frozen Qwen3 embeddings + tiny MLP."""
    def __init__(self, frozen_embedding, hidden_dim, output_dim):
        super().__init__()
        self.embedding = frozen_embedding
        self.embedding.requires_grad_(False)
        embed_dim = frozen_embedding.embedding_dim

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, input_ids, attention_mask=None):
        with torch.no_grad():
            x = self.embedding(input_ids)
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            x = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            x = x.mean(dim=1)
        return self.mlp(x)


class LinearProbe(nn.Module):
    """Approach 3: Just a linear layer on frozen embeddings."""
    def __init__(self, frozen_embedding, output_dim):
        super().__init__()
        self.embedding = frozen_embedding
        self.embedding.requires_grad_(False)
        self.linear = nn.Linear(frozen_embedding.embedding_dim, output_dim)

    def forward(self, input_ids, attention_mask=None):
        with torch.no_grad():
            x = self.embedding(input_ids)
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            x = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            x = x.mean(dim=1)
        return self.linear(x)


# ============================================================================
# Dataset
# ============================================================================

class ConceptDataset(Dataset):
    def __init__(self, concepts, direction_vectors, tokenizer):
        self.tokenizer = tokenizer
        self.data = [(c, direction_vectors[c]) for c in concepts]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        concept, vec = self.data[idx]
        text = f"The concept {concept} means"
        enc = self.tokenizer(text, max_length=32, padding="max_length",
                            truncation=True, return_tensors="pt")
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "target": vec,
        }


# ============================================================================
# Training
# ============================================================================

def train_and_eval(model, train_loader, val_loader, test_loader, epochs, lr, device):
    """Train and return results."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=0.01
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    best_val = -1

    for epoch in range(1, epochs + 1):
        model.train()
        train_sim = 0
        n = 0

        for batch in train_loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            target = batch["target"].to(device)

            pred = model(ids, mask)
            pred_n = F.normalize(pred, dim=-1)
            tgt_n = F.normalize(target, dim=-1)
            loss = (1 - (pred_n * tgt_n).sum(dim=-1)).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_sim += (pred_n * tgt_n).sum(dim=-1).mean().item()
            n += 1

        scheduler.step()
        train_sim /= n

        # Validate every 10 epochs
        if epoch % 10 == 0:
            model.eval()
            val_sim = 0
            n = 0
            with torch.no_grad():
                for batch in val_loader:
                    ids = batch["input_ids"].to(device)
                    mask = batch["attention_mask"].to(device)
                    target = batch["target"].to(device)
                    pred = model(ids, mask)
                    pred_n = F.normalize(pred, dim=-1)
                    tgt_n = F.normalize(target, dim=-1)
                    val_sim += (pred_n * tgt_n).sum(dim=-1).mean().item()
                    n += 1
            val_sim /= n
            if val_sim > best_val:
                best_val = val_sim
            print(f"    Epoch {epoch}: train={train_sim:.4f}, val={val_sim:.4f}")

    # Test
    model.eval()
    test_sim = 0
    n = 0
    with torch.no_grad():
        for batch in test_loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            target = batch["target"].to(device)
            pred = model(ids, mask)
            pred_n = F.normalize(pred, dim=-1)
            tgt_n = F.normalize(target, dim=-1)
            test_sim += (pred_n * tgt_n).sum(dim=-1).mean().item()
            n += 1
    test_sim /= n

    return {
        "trainable_params": trainable,
        "best_val": best_val,
        "test": test_sim,
        "gap": best_val - test_sim,
    }


def main():
    parser = argparse.ArgumentParser()
    default_vectors = Path(__file__).parent.parent.parent / "data" / "vectors" / "qwen3_vectors_qwen3_0.6b_50000concepts.pt"
    parser.add_argument("--vectors", type=str, default=str(default_vectors))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("="*70)
    print("APPROACH COMPARISON - Small models, ALL concepts")
    print("="*70)

    # Load data
    print("\nLoading vectors...")
    data = torch.load(args.vectors)
    all_concepts = data["concepts"]
    direction_vectors = data["direction_vectors"]
    output_dim = data["hidden_dim"]
    model_id = data["model_id"]

    print(f"  Total concepts: {len(all_concepts)}")

    # Split
    random.seed(42)
    random.shuffle(all_concepts)
    n = len(all_concepts)
    train_concepts = all_concepts[:int(n*0.7)]
    val_concepts = all_concepts[int(n*0.7):int(n*0.85)]
    test_concepts = all_concepts[int(n*0.85):]

    print(f"  Train: {len(train_concepts)}, Val: {len(val_concepts)}, Test: {len(test_concepts)}")

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    vocab_size = len(tokenizer)

    # Load frozen embedding
    print("Loading Qwen3 embedding layer...")
    qwen = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float32, device_map="cpu", trust_remote_code=True
    )
    frozen_emb = nn.Embedding(
        qwen.model.embed_tokens.num_embeddings,
        qwen.model.embed_tokens.embedding_dim
    )
    frozen_emb.weight.data = qwen.model.embed_tokens.weight.data.clone()
    frozen_emb = frozen_emb.to(device)
    del qwen
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Create dataloaders
    train_loader = DataLoader(
        ConceptDataset(train_concepts, direction_vectors, tokenizer),
        batch_size=args.batch_size, shuffle=True
    )
    val_loader = DataLoader(
        ConceptDataset(val_concepts, direction_vectors, tokenizer),
        batch_size=args.batch_size
    )
    test_loader = DataLoader(
        ConceptDataset(test_concepts, direction_vectors, tokenizer),
        batch_size=args.batch_size
    )

    results = []

    # ========================================
    # Approach 1: Tiny trainable embeddings
    # ========================================
    for embed_dim in [32, 64]:
        for hidden_dim in [128, 256]:
            print(f"\n[Trainable] embed={embed_dim}, hidden={hidden_dim}")
            model = TinyTrainableModel(
                vocab_size, embed_dim, hidden_dim, output_dim
            ).to(device)

            res = train_and_eval(model, train_loader, val_loader, test_loader,
                                args.epochs, lr=1e-3, device=device)
            res["approach"] = "trainable"
            res["embed_dim"] = embed_dim
            res["hidden_dim"] = hidden_dim
            results.append(res)

            print(f"  Params: {res['trainable_params']:,}, Val: {res['best_val']:.4f}, Test: {res['test']:.4f}, Gap: {res['gap']:.4f}")

    # ========================================
    # Approach 2: Frozen embeddings + MLP
    # ========================================
    for hidden_dim in [64, 128, 256]:
        print(f"\n[Frozen+MLP] hidden={hidden_dim}")
        model = FrozenEmbedMLP(frozen_emb, hidden_dim, output_dim).to(device)

        res = train_and_eval(model, train_loader, val_loader, test_loader,
                            args.epochs, lr=1e-3, device=device)
        res["approach"] = "frozen_mlp"
        res["hidden_dim"] = hidden_dim
        results.append(res)

        print(f"  Params: {res['trainable_params']:,}, Val: {res['best_val']:.4f}, Test: {res['test']:.4f}, Gap: {res['gap']:.4f}")

    # ========================================
    # Approach 3: Linear probe
    # ========================================
    print(f"\n[Linear Probe]")
    model = LinearProbe(frozen_emb, output_dim).to(device)

    res = train_and_eval(model, train_loader, val_loader, test_loader,
                        args.epochs, lr=1e-3, device=device)
    res["approach"] = "linear_probe"
    results.append(res)

    print(f"  Params: {res['trainable_params']:,}, Val: {res['best_val']:.4f}, Test: {res['test']:.4f}, Gap: {res['gap']:.4f}")

    # ========================================
    # Summary
    # ========================================
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    print(f"{'Approach':<20} {'Config':<15} {'Params':<12} {'Val':<10} {'Test':<10} {'Gap':<10}")
    print("-"*77)

    sorted_results = sorted(results, key=lambda x: x["test"], reverse=True)
    for r in sorted_results:
        if r["approach"] == "trainable":
            config = f"e={r['embed_dim']},h={r['hidden_dim']}"
        elif r["approach"] == "frozen_mlp":
            config = f"h={r['hidden_dim']}"
        else:
            config = "-"

        print(f"{r['approach']:<20} {config:<15} {r['trainable_params']:<12,} {r['best_val']:<10.4f} {r['test']:<10.4f} {r['gap']:<10.4f}")

    # Save
    output_dir = Path(__file__).parent.parent.parent / "results"
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(output_dir / f"approach_comparison_{timestamp}.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_dir}")

    # Best
    best = sorted_results[0]
    print(f"\nBest: {best['approach']} with test={best['test']:.4f}")


if __name__ == "__main__":
    main()
