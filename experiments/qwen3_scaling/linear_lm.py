"""
Linear Language Model Experiment

Uses the insight that embedding → direction is approximately linear.
Therefore direction → embedding should also be approximately linear (inverse).

Flow:
1. Learn linear transform: embedding → direction (we proved this works at 0.906)
2. Compute inverse: direction → embedding
3. Train tiny model to predict next direction vector
4. Generate: predict_direction → inverse → embedding → nearest_token
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


class LinearTransform(nn.Module):
    """Linear transform: embedding → direction"""
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim)

    def forward(self, x):
        return self.linear(x)


class NextVectorPredictor(nn.Module):
    """Tiny model to predict next direction vector given sequence."""
    def __init__(self, dim, hidden_dim=256, num_layers=2):
        super().__init__()

        self.pos_embed = nn.Parameter(torch.randn(1, 512, dim) * 0.02)

        layers = []
        for _ in range(num_layers):
            layers.append(nn.TransformerEncoderLayer(
                d_model=dim, nhead=8, dim_feedforward=hidden_dim*4,
                dropout=0.1, batch_first=True
            ))
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=dim, nhead=8, dim_feedforward=hidden_dim*4, batch_first=True),
            num_layers=num_layers
        )

    def forward(self, directions, mask=None):
        """
        directions: [batch, seq_len, dim]
        Returns: predicted next direction for each position
        """
        seq_len = directions.shape[1]
        x = directions + self.pos_embed[:, :seq_len, :]

        # Causal mask
        causal = torch.triu(torch.ones(seq_len, seq_len, device=directions.device), diagonal=1).bool()

        x = self.transformer(x, mask=causal)
        return x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--num-sequences", type=int, default=5000)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--hidden-dim", type=int, default=256)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("="*70)
    print("LINEAR LANGUAGE MODEL")
    print("="*70)
    print("Using linear transform insight for generation")
    print("="*70)

    # Load model for extraction
    print("\nLoading Qwen3...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    model.eval()

    dim = model.config.hidden_size
    extraction_layer = model.config.num_hidden_layers // 2

    print(f"  Hidden dim: {dim}")
    print(f"  Extraction layer: {extraction_layer}")

    # Get embedding matrix
    embedding_matrix = model.model.embed_tokens.weight.data.float().cpu()
    print(f"  Vocab size: {embedding_matrix.shape[0]}")

    # Hook for extraction
    activations = None
    def hook_fn(module, input, output):
        nonlocal activations
        if isinstance(output, tuple):
            activations = output[0].detach()
        else:
            activations = output.detach()

    model.model.layers[extraction_layer].register_forward_hook(hook_fn)

    # Generate LOTS of varied sentences
    subjects = [
        "The man", "The woman", "A child", "The cat", "The dog", "A bird",
        "The teacher", "The doctor", "A student", "The scientist", "An artist",
        "The boy", "The girl", "A person", "The worker", "The chef",
        "My friend", "Her brother", "His sister", "Their mother", "Our father",
        "The king", "The queen", "A soldier", "The farmer", "The writer",
        "Someone", "Everyone", "Nobody", "The leader", "A stranger",
    ]

    verbs = [
        "walked", "ran", "jumped", "sat", "looked", "said", "wanted", "needed",
        "thought", "believed", "knew", "understood", "learned", "taught",
        "made", "created", "built", "fixed", "broke", "found", "lost",
        "opened", "closed", "started", "stopped", "continued", "finished",
        "loved", "hated", "feared", "hoped", "wished", "dreamed",
        "ate", "drank", "slept", "woke", "worked", "played", "rested",
        "spoke", "listened", "watched", "heard", "felt", "touched",
        "gave", "took", "brought", "sent", "received", "bought", "sold",
    ]

    objects = [
        "the book", "the door", "the window", "the food", "the water",
        "the letter", "the phone", "the car", "the house", "the money",
        "the truth", "the answer", "the question", "the problem", "the solution",
        "a gift", "a message", "a story", "a song", "a picture",
        "something", "everything", "nothing", "anything", "the idea",
    ]

    adverbs = [
        "slowly", "quickly", "carefully", "happily", "sadly", "quietly",
        "loudly", "softly", "gently", "roughly", "easily", "hardly",
        "suddenly", "gradually", "immediately", "eventually", "finally",
        "always", "never", "often", "sometimes", "rarely", "usually",
    ]

    locations = [
        "in the room", "at the park", "on the street", "by the river",
        "near the house", "inside the building", "outside the door",
        "at home", "at work", "at school", "in the garden", "on the roof",
    ]

    texts = []

    # Pattern 1: Subject + Verb + Adverb
    for s in subjects:
        for v in verbs:
            for a in adverbs[:10]:
                texts.append(f"{s} {v} {a}")

    # Pattern 2: Subject + Verb + Object
    for s in subjects[:15]:
        for v in verbs[:20]:
            for o in objects[:10]:
                texts.append(f"{s} {v} {o}")

    # Pattern 3: Subject + Verb + Location
    for s in subjects[:10]:
        for v in verbs[:15]:
            for l in locations:
                texts.append(f"{s} {v} {l}")

    # Pattern 4: Longer sentences
    for s in subjects[:10]:
        for v in verbs[:10]:
            for o in objects[:8]:
                for a in adverbs[:5]:
                    texts.append(f"{s} {v} {o} {a}")

    random.shuffle(texts)
    texts = texts[:args.num_sequences]
    print(f"\nUsing {len(texts)} sequences")

    # Extract embeddings and directions for all sequences
    print("\nExtracting embeddings and directions...")

    all_embeddings = []  # List of [seq_len, dim] tensors
    all_directions = []  # List of [seq_len, dim] tensors

    with torch.no_grad():
        for text in tqdm(texts, desc="Extracting"):
            inputs = tokenizer(text, return_tensors="pt").to(device)
            _ = model(**inputs)

            # Get embeddings (input to model)
            emb = model.model.embed_tokens(inputs["input_ids"]).squeeze(0).float().cpu()

            # Get directions (from middle layer)
            dirs = activations.squeeze(0).float().cpu()

            if emb.shape[0] >= 3:  # Need at least 3 tokens
                all_embeddings.append(emb)
                all_directions.append(dirs)

    print(f"Extracted {len(all_embeddings)} valid sequences")

    # Compute mean direction for centering
    all_dirs_flat = torch.cat(all_directions, dim=0)
    mean_direction = all_dirs_flat.mean(dim=0)
    print(f"Mean direction computed from {all_dirs_flat.shape[0]} tokens")

    # Center directions
    for i in range(len(all_directions)):
        all_directions[i] = all_directions[i] - mean_direction

    # Step 1: Train linear transform embedding → direction
    print("\n" + "="*70)
    print("STEP 1: Training linear transform (embedding → direction)")
    print("="*70)

    linear_model = LinearTransform(dim).to(device)
    optimizer = torch.optim.AdamW(linear_model.parameters(), lr=1e-3)

    # Prepare data for linear transform training
    emb_flat = torch.cat(all_embeddings, dim=0).to(device)
    dir_flat = torch.cat(all_directions, dim=0).to(device)

    n_samples = emb_flat.shape[0]
    batch_size = 256

    for epoch in range(20):
        perm = torch.randperm(n_samples)
        total_loss = 0
        n_batches = 0

        for i in range(0, n_samples, batch_size):
            idx = perm[i:i+batch_size]
            emb_batch = emb_flat[idx]
            dir_batch = dir_flat[idx]

            pred = linear_model(emb_batch)

            # Cosine loss
            pred_n = F.normalize(pred, dim=-1)
            dir_n = F.normalize(dir_batch, dim=-1)
            loss = (1 - (pred_n * dir_n).sum(dim=-1)).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        if (epoch + 1) % 5 == 0:
            cos_sim = (pred_n * dir_n).sum(dim=-1).mean().item()
            print(f"Epoch {epoch+1}: loss={total_loss/n_batches:.4f}, cos_sim={cos_sim:.4f}")

    # Get the learned linear transform weight
    W = linear_model.linear.weight.data.cpu()  # [dim, dim]
    b = linear_model.linear.bias.data.cpu()    # [dim]

    # Compute pseudo-inverse for direction → embedding
    print("\nComputing inverse transform...")
    W_inv = torch.linalg.pinv(W)
    print(f"  W shape: {W.shape}, W_inv shape: {W_inv.shape}")

    # Test inverse
    test_emb = emb_flat[:100].cpu()
    test_dir = dir_flat[:100].cpu()

    with torch.no_grad():
        # Forward: emb → dir
        pred_dir = test_emb @ W.T + b

        # Inverse: dir → emb
        reconstructed_emb = (test_dir - b) @ W_inv.T

        # Check reconstruction
        recon_cos = F.cosine_similarity(reconstructed_emb, test_emb, dim=-1).mean()
        print(f"  Reconstruction cosine similarity: {recon_cos:.4f}")

    # Step 2: Train next-direction predictor
    print("\n" + "="*70)
    print("STEP 2: Training next-direction predictor")
    print("="*70)

    predictor = NextVectorPredictor(dim, args.hidden_dim).to(device)
    num_params = sum(p.numel() for p in predictor.parameters())
    print(f"Predictor params: {num_params:,}")

    optimizer = torch.optim.AdamW(predictor.parameters(), lr=1e-4)

    # Split sequences
    n_seqs = len(all_directions)
    train_dirs = all_directions[:int(n_seqs*0.8)]
    val_dirs = all_directions[int(n_seqs*0.8):]

    print(f"Train sequences: {len(train_dirs)}, Val: {len(val_dirs)}")

    best_val = -1

    for epoch in range(args.epochs):
        predictor.train()
        random.shuffle(train_dirs)

        total_loss = 0
        n_batches = 0

        for seq in train_dirs:
            if seq.shape[0] < 3:
                continue

            seq = seq.unsqueeze(0).to(device)  # [1, seq_len, dim]

            # Input: positions 0 to n-2, Target: positions 1 to n-1
            input_seq = seq[:, :-1, :]
            target_seq = seq[:, 1:, :]

            pred = predictor(input_seq)

            # Cosine loss
            pred_n = F.normalize(pred, dim=-1)
            tgt_n = F.normalize(target_seq, dim=-1)
            loss = (1 - (pred_n * tgt_n).sum(dim=-1)).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        # Validate
        if (epoch + 1) % 5 == 0:
            predictor.eval()
            val_cos = 0
            n_val = 0

            with torch.no_grad():
                for seq in val_dirs:
                    if seq.shape[0] < 3:
                        continue
                    seq = seq.unsqueeze(0).to(device)
                    input_seq = seq[:, :-1, :]
                    target_seq = seq[:, 1:, :]

                    pred = predictor(input_seq)
                    pred_n = F.normalize(pred, dim=-1)
                    tgt_n = F.normalize(target_seq, dim=-1)
                    val_cos += (pred_n * tgt_n).sum(dim=-1).mean().item()
                    n_val += 1

            val_cos /= n_val
            if val_cos > best_val:
                best_val = val_cos
            print(f"Epoch {epoch+1}: train_loss={total_loss/n_batches:.4f}, val_cos={val_cos:.4f}")

    # Step 3: Generation test
    print("\n" + "="*70)
    print("STEP 3: Generation test")
    print("="*70)

    W_inv = W_inv.to(device)
    b_dev = b.to(device)
    mean_dir_dev = mean_direction.to(device)
    emb_matrix = embedding_matrix.to(device)

    def generate(prompt, max_tokens=10):
        predictor.eval()

        # Get initial directions from prompt
        with torch.no_grad():
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            _ = model(**inputs)
            dirs = activations.squeeze(0).float() - mean_dir_dev  # [seq_len, dim]
            dirs = dirs.unsqueeze(0)  # [1, seq_len, dim]

        generated = []

        for _ in range(max_tokens):
            with torch.no_grad():
                # Predict next direction
                pred_dir = predictor(dirs)[:, -1, :]  # [1, dim]

                # Add back mean
                pred_dir_raw = pred_dir + mean_dir_dev

                # Inverse transform: direction → embedding
                pred_emb = (pred_dir_raw - b_dev) @ W_inv.T  # [1, dim]

                # Find nearest token
                pred_emb_n = F.normalize(pred_emb, dim=-1)
                emb_n = F.normalize(emb_matrix, dim=-1)
                sims = pred_emb_n @ emb_n.T  # [1, vocab]

                token_id = sims.argmax(dim=-1).item()
                token = tokenizer.decode([token_id])
                generated.append(token)

                # Get actual direction for this token and append
                # (For next iteration, use the predicted direction)
                dirs = torch.cat([dirs, pred_dir.unsqueeze(1)], dim=1)

        return generated

    # Test generation
    prompts = [
        "The cat sat on",
        "She walked to the",
        "He opened the door",
        "They went to",
        "The sun was",
    ]

    for prompt in prompts:
        tokens = generate(prompt, max_tokens=8)
        print(f"\nPrompt: '{prompt}'")
        print(f"Generated: {' '.join(tokens)}")

    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE")
    print("="*70)
    print(f"Linear transform cos_sim: ~0.90 (trained earlier)")
    print(f"Reconstruction cos_sim: {recon_cos:.4f}")
    print(f"Next-direction prediction val_cos: {best_val:.4f}")
    print(f"Predictor params: {num_params:,}")


if __name__ == "__main__":
    main()
