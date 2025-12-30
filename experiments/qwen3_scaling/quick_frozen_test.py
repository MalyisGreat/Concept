"""
Quick test: Frozen Qwen3 embeddings + small MLP on ALL 50K concepts.
Single run, no grid search. Just see if it works.
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


class FrozenEmbeddingMLP(nn.Module):
    def __init__(self, frozen_embedding: nn.Embedding, hidden_dim: int,
                 output_dim: int, num_layers: int = 2):
        super().__init__()

        self.embedding = frozen_embedding
        self.embedding.requires_grad_(False)

        embed_dim = frozen_embedding.embedding_dim

        layers = []
        in_dim = embed_dim
        for _ in range(num_layers):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1),
            ])
            in_dim = hidden_dim

        self.mlp = nn.Sequential(*layers)
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, input_ids, attention_mask=None):
        with torch.no_grad():
            x = self.embedding(input_ids)

        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            x = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            x = x.mean(dim=1)

        x = self.mlp(x)
        return self.output(x)


class SimpleDataset(Dataset):
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


def main():
    parser = argparse.ArgumentParser()
    default_vectors = Path(__file__).parent.parent.parent / "data" / "vectors" / "qwen3_vectors_qwen3_0.6b_50000concepts.pt"
    parser.add_argument("--vectors", type=str, default=str(default_vectors))
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("="*60)
    print("QUICK FROZEN EMBEDDING TEST")
    print("="*60)

    # Load vectors
    print(f"\nLoading vectors...")
    data = torch.load(args.vectors)
    all_concepts = data["concepts"]
    direction_vectors = data["direction_vectors"]
    output_dim = data["hidden_dim"]
    model_id = data["model_id"]

    print(f"  Concepts: {len(all_concepts)}")
    print(f"  Output dim: {output_dim}")

    # Shuffle and split
    random.seed(42)
    random.shuffle(all_concepts)

    n = len(all_concepts)
    n_train = int(n * 0.7)
    n_val = int(n * 0.15)

    train_concepts = all_concepts[:n_train]
    val_concepts = all_concepts[n_train:n_train+n_val]
    test_concepts = all_concepts[n_train+n_val:]

    print(f"  Train: {len(train_concepts)}, Val: {len(val_concepts)}, Test: {len(test_concepts)}")

    # Load tokenizer and frozen embedding
    print(f"\nLoading Qwen3 embedding layer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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

    print(f"  Embedding dim: {frozen_emb.embedding_dim}")

    # Create model
    model = FrozenEmbeddingMLP(
        frozen_emb, args.hidden, output_dim, args.layers
    ).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Hidden: {args.hidden}, Layers: {args.layers}")
    print(f"  Trainable params: {trainable:,}")
    print(f"  Params per concept: {trainable / len(train_concepts):.1f}")

    # Data
    train_loader = DataLoader(
        SimpleDataset(train_concepts, direction_vectors, tokenizer),
        batch_size=args.batch_size, shuffle=True
    )
    val_loader = DataLoader(
        SimpleDataset(val_concepts, direction_vectors, tokenizer),
        batch_size=args.batch_size
    )
    test_loader = DataLoader(
        SimpleDataset(test_concepts, direction_vectors, tokenizer),
        batch_size=args.batch_size
    )

    # Train
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=0.01
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    print(f"\nTraining for {args.epochs} epochs...")

    best_val = -1

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_sim = 0
        n_batch = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)
        for batch in pbar:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            target = batch["target"].to(device)

            pred = model(ids, mask)

            # Cosine loss
            pred_n = F.normalize(pred, dim=-1)
            tgt_n = F.normalize(target, dim=-1)
            cos_sim = (pred_n * tgt_n).sum(dim=-1)
            loss = (1 - cos_sim).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_sim += cos_sim.mean().item()
            n_batch += 1
            pbar.set_postfix({"cos_sim": f"{train_sim/n_batch:.4f}"})

        scheduler.step()
        train_sim /= n_batch

        # Validate
        if epoch % 5 == 0 or epoch == args.epochs:
            model.eval()
            val_sim = 0
            n_batch = 0

            with torch.no_grad():
                for batch in val_loader:
                    ids = batch["input_ids"].to(device)
                    mask = batch["attention_mask"].to(device)
                    target = batch["target"].to(device)

                    pred = model(ids, mask)
                    pred_n = F.normalize(pred, dim=-1)
                    tgt_n = F.normalize(target, dim=-1)
                    cos_sim = (pred_n * tgt_n).sum(dim=-1)

                    val_sim += cos_sim.mean().item()
                    n_batch += 1

            val_sim /= n_batch

            if val_sim > best_val:
                best_val = val_sim

            print(f"Epoch {epoch:3d}: Train={train_sim:.4f}, Val={val_sim:.4f}, Best={best_val:.4f}")

    # Test
    model.eval()
    test_sim = 0
    n_batch = 0

    with torch.no_grad():
        for batch in test_loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            target = batch["target"].to(device)

            pred = model(ids, mask)
            pred_n = F.normalize(pred, dim=-1)
            tgt_n = F.normalize(target, dim=-1)
            cos_sim = (pred_n * tgt_n).sum(dim=-1)

            test_sim += cos_sim.mean().item()
            n_batch += 1

    test_sim /= n_batch

    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"  Train concepts: {len(train_concepts)}")
    print(f"  Test concepts:  {len(test_concepts)}")
    print(f"  Trainable params: {trainable:,}")
    print(f"  Params/concept: {trainable / len(train_concepts):.1f}")
    print(f"\n  Best Val cos_sim: {best_val:.4f}")
    print(f"  Test cos_sim:     {test_sim:.4f}")
    print(f"  Gap:              {best_val - test_sim:.4f}")
    print("="*60)


if __name__ == "__main__":
    main()
