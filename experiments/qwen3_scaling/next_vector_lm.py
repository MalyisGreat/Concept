"""
Next-Vector Language Model Experiment

Instead of predicting single concept vectors, predict the NEXT vector in a sequence.
This is like a language model but operating in semantic vector space.

Architecture:
1. Encode each token in a sequence to a vector (using our trained encoder or Qwen3 embeddings)
2. Use a small transformer/RNN to predict the next vector
3. Decode by finding nearest token in embedding space

This could be a "compressed semantic language model".
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
# Extract sequence vectors from Qwen3
# ============================================================================

class SequenceVectorExtractor:
    """Extract hidden state vectors for each token in sequences."""

    def __init__(self, model_id: str, device: str = "cuda"):
        self.device = device
        self.model_id = model_id

        print(f"Loading model: {model_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        self.model.eval()

        self.num_layers = self.model.config.num_hidden_layers
        self.hidden_dim = self.model.config.hidden_size
        self.extraction_layer = self.num_layers // 2

        print(f"  Hidden dim: {self.hidden_dim}")
        print(f"  Extraction layer: {self.extraction_layer}")

        # Get embedding matrix for decoding
        self.embedding_matrix = self.model.model.embed_tokens.weight.data.float().cpu()

        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                self.activations = output[0].detach()
            else:
                self.activations = output.detach()

        layer = self.model.model.layers[self.extraction_layer]
        layer.register_forward_hook(hook_fn)

    @torch.no_grad()
    def extract_sequence(self, text: str) -> torch.Tensor:
        """Extract vector for each token position."""
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        _ = self.model(**inputs)
        # Return all positions, not just mean
        return self.activations.squeeze(0).float().cpu()  # [seq_len, hidden_dim]

    def decode_vector(self, vector: torch.Tensor, top_k: int = 5) -> list:
        """Find nearest tokens to a vector."""
        vector = vector.float().cpu()
        # Cosine similarity with all embeddings
        vector_norm = F.normalize(vector.unsqueeze(0), dim=-1)
        emb_norm = F.normalize(self.embedding_matrix, dim=-1)
        sims = torch.mm(vector_norm, emb_norm.t()).squeeze(0)

        top_indices = sims.topk(top_k).indices.tolist()
        return [(self.tokenizer.decode([idx]), sims[idx].item()) for idx in top_indices]


# ============================================================================
# Next-Vector Prediction Model
# ============================================================================

class NextVectorTransformer(nn.Module):
    """
    Tiny transformer that predicts the next vector given previous vectors.

    Input: sequence of vectors [v1, v2, ..., vn]
    Output: predicted next vector v_{n+1}
    """

    def __init__(self, hidden_dim: int, model_dim: int = 256,
                 num_heads: int = 4, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.model_dim = model_dim

        # Project input vectors to model dimension
        self.input_proj = nn.Linear(hidden_dim, model_dim)

        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, 512, model_dim) * 0.02)

        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=model_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output projection back to hidden_dim
        self.output_proj = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, hidden_dim),
        )

    def forward(self, vectors: torch.Tensor, mask: torch.Tensor = None):
        """
        vectors: [batch, seq_len, hidden_dim]
        Returns: [batch, seq_len, hidden_dim] - predicted next vectors for each position
        """
        batch_size, seq_len, _ = vectors.shape

        # Project to model dim
        x = self.input_proj(vectors)

        # Add positional encoding
        x = x + self.pos_encoding[:, :seq_len, :]

        # Causal mask for autoregressive prediction
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=vectors.device), diagonal=1).bool()

        # Transformer
        x = self.transformer(x, mask=causal_mask)

        # Project back to hidden_dim
        x = self.output_proj(x)

        return x


class NextVectorRNN(nn.Module):
    """Simpler RNN version for comparison."""

    def __init__(self, hidden_dim: int, model_dim: int = 256, num_layers: int = 2):
        super().__init__()

        self.input_proj = nn.Linear(hidden_dim, model_dim)
        self.rnn = nn.GRU(model_dim, model_dim, num_layers=num_layers, batch_first=True)
        self.output_proj = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, hidden_dim),
        )

    def forward(self, vectors: torch.Tensor, mask: torch.Tensor = None):
        x = self.input_proj(vectors)
        x, _ = self.rnn(x)
        return self.output_proj(x)


# ============================================================================
# Dataset
# ============================================================================

class SequenceDataset(Dataset):
    """Dataset of token sequences with their vectors."""

    def __init__(self, sequences: list, max_len: int = 32):
        self.sequences = sequences
        self.max_len = max_len

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        vectors = self.sequences[idx]  # [seq_len, hidden_dim]
        seq_len = vectors.shape[0]

        # Pad or truncate
        if seq_len > self.max_len:
            vectors = vectors[:self.max_len]
            seq_len = self.max_len
        elif seq_len < self.max_len:
            pad = torch.zeros(self.max_len - seq_len, vectors.shape[1])
            vectors = torch.cat([vectors, pad], dim=0)

        # Input: positions 0 to n-1, Target: positions 1 to n
        input_vecs = vectors[:-1]  # [max_len-1, hidden_dim]
        target_vecs = vectors[1:]  # [max_len-1, hidden_dim]

        # Mask for valid positions
        mask = torch.zeros(self.max_len - 1)
        mask[:min(seq_len - 1, self.max_len - 1)] = 1

        return {
            "input": input_vecs,
            "target": target_vecs,
            "mask": mask,
        }


# ============================================================================
# Training
# ============================================================================

def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    total_cos_sim = 0
    n_batches = 0

    for batch in dataloader:
        inputs = batch["input"].to(device)
        targets = batch["target"].to(device)
        mask = batch["mask"].to(device)

        predictions = model(inputs)

        # Cosine loss only on valid positions
        pred_norm = F.normalize(predictions, dim=-1)
        tgt_norm = F.normalize(targets, dim=-1)
        cos_sim = (pred_norm * tgt_norm).sum(dim=-1)  # [batch, seq_len]

        # Masked loss
        cos_sim_masked = cos_sim * mask
        loss = (1 - cos_sim_masked).sum() / mask.sum()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        total_cos_sim += (cos_sim_masked.sum() / mask.sum()).item()
        n_batches += 1

    return total_loss / n_batches, total_cos_sim / n_batches


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    total_cos_sim = 0
    n_batches = 0

    for batch in dataloader:
        inputs = batch["input"].to(device)
        targets = batch["target"].to(device)
        mask = batch["mask"].to(device)

        predictions = model(inputs)

        pred_norm = F.normalize(predictions, dim=-1)
        tgt_norm = F.normalize(targets, dim=-1)
        cos_sim = (pred_norm * tgt_norm).sum(dim=-1)

        cos_sim_masked = cos_sim * mask
        total_cos_sim += (cos_sim_masked.sum() / mask.sum()).item()
        n_batches += 1

    return total_cos_sim / n_batches


# ============================================================================
# Text generation
# ============================================================================

@torch.no_grad()
def generate_text(model, extractor, prompt: str, max_new_tokens: int = 20, device: str = "cuda"):
    """Generate text by predicting next vectors and decoding."""
    model.eval()

    # Get initial vectors from prompt
    vectors = extractor.extract_sequence(prompt).to(device)
    vectors = vectors.unsqueeze(0)  # [1, seq_len, hidden_dim]

    generated_tokens = []

    for _ in range(max_new_tokens):
        # Predict next vector
        pred = model(vectors)
        next_vec = pred[0, -1]  # Last position prediction

        # Decode to token
        decoded = extractor.decode_vector(next_vec.cpu(), top_k=1)
        token, score = decoded[0]
        generated_tokens.append((token, score))

        # Get actual vector for this token and append
        # (In practice, we'd use the model's embedding, but for now use predicted)
        next_vec = next_vec.unsqueeze(0).unsqueeze(0)  # [1, 1, hidden_dim]
        vectors = torch.cat([vectors, next_vec], dim=1)

    return generated_tokens


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--num-sequences", type=int, default=10000)
    parser.add_argument("--max-len", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--model-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--arch", type=str, default="transformer", choices=["transformer", "rnn"])
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("="*70)
    print("NEXT-VECTOR LANGUAGE MODEL")
    print("="*70)

    # Sample sentences for training
    sample_texts = [
        "The quick brown fox jumps over the lazy dog",
        "In the beginning there was nothing but darkness",
        "She walked down the street looking for her keys",
        "The sun rose slowly over the mountains",
        "He opened the door and stepped inside",
        "It was a dark and stormy night",
        "The cat sat on the windowsill watching birds",
        "They decided to go to the beach for vacation",
        "The old house creaked in the wind",
        "She picked up the phone and dialed the number",
        "The children played in the garden all afternoon",
        "He cooked dinner while listening to music",
        "The train arrived at the station on time",
        "She read the book from cover to cover",
        "The dog barked at the mailman",
        "They watched the sunset from the rooftop",
        "The rain fell gently on the leaves",
        "He fixed the broken chair with glue",
        "She planted flowers in the garden",
        "The bird flew south for the winter",
    ]

    # Generate more varied sentences
    subjects = ["The man", "The woman", "The child", "The cat", "The dog", "A bird", "The teacher", "The doctor"]
    verbs = ["walked", "ran", "jumped", "sat", "stood", "looked", "thought", "said", "wanted", "needed"]
    objects = ["the book", "the door", "the window", "home", "outside", "carefully", "slowly", "quickly"]

    for subj in subjects:
        for verb in verbs:
            for obj in objects:
                sample_texts.append(f"{subj} {verb} {obj}")

    # Shuffle and limit
    random.shuffle(sample_texts)
    sample_texts = sample_texts[:args.num_sequences]

    print(f"\nNumber of training sequences: {len(sample_texts)}")

    # Extract vectors
    print("\nExtracting sequence vectors from Qwen3...")
    extractor = SequenceVectorExtractor(args.model, device)

    sequences = []
    for text in tqdm(sample_texts, desc="Extracting"):
        try:
            vecs = extractor.extract_sequence(text)
            if vecs.shape[0] >= 3:  # Need at least 3 tokens for input/target
                sequences.append(vecs)
        except:
            continue

    print(f"Extracted {len(sequences)} valid sequences")

    hidden_dim = extractor.hidden_dim

    # Split data
    random.shuffle(sequences)
    n = len(sequences)
    train_seqs = sequences[:int(n*0.8)]
    val_seqs = sequences[int(n*0.8):int(n*0.9)]
    test_seqs = sequences[int(n*0.9):]

    print(f"Train: {len(train_seqs)}, Val: {len(val_seqs)}, Test: {len(test_seqs)}")

    # Datasets
    train_loader = DataLoader(
        SequenceDataset(train_seqs, args.max_len),
        batch_size=args.batch_size, shuffle=True
    )
    val_loader = DataLoader(
        SequenceDataset(val_seqs, args.max_len),
        batch_size=args.batch_size
    )
    test_loader = DataLoader(
        SequenceDataset(test_seqs, args.max_len),
        batch_size=args.batch_size
    )

    # Create model
    if args.arch == "transformer":
        model = NextVectorTransformer(
            hidden_dim=hidden_dim,
            model_dim=args.model_dim,
            num_layers=args.num_layers,
        ).to(device)
    else:
        model = NextVectorRNN(
            hidden_dim=hidden_dim,
            model_dim=args.model_dim,
            num_layers=args.num_layers,
        ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {args.arch}")
    print(f"Model dim: {args.model_dim}")
    print(f"Layers: {args.num_layers}")
    print(f"Parameters: {num_params:,}")

    # Train
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    best_val = -1

    print("\nTraining...")
    for epoch in range(1, args.epochs + 1):
        train_loss, train_cos = train_epoch(model, train_loader, optimizer, device)
        scheduler.step()

        if epoch % 5 == 0:
            val_cos = evaluate(model, val_loader, device)
            if val_cos > best_val:
                best_val = val_cos
            print(f"Epoch {epoch}: train_cos={train_cos:.4f}, val_cos={val_cos:.4f}")

    # Test
    test_cos = evaluate(model, test_loader, device)

    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"Architecture: {args.arch}")
    print(f"Parameters: {num_params:,}")
    print(f"Best Val cos_sim: {best_val:.4f}")
    print(f"Test cos_sim: {test_cos:.4f}")
    print(f"Gap: {best_val - test_cos:.4f}")

    # Try generation
    print("\n" + "="*70)
    print("GENERATION TEST")
    print("="*70)

    test_prompts = [
        "The cat sat on",
        "She walked to the",
        "He opened the",
    ]

    for prompt in test_prompts:
        print(f"\nPrompt: '{prompt}'")
        generated = generate_text(model, extractor, prompt, max_new_tokens=10, device=device)
        tokens = [t[0] for t in generated]
        print(f"Generated: {' '.join(tokens)}")

    # Save results
    output_dir = Path(__file__).parent.parent.parent / "results"
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    results = {
        "arch": args.arch,
        "model_dim": args.model_dim,
        "num_layers": args.num_layers,
        "num_params": num_params,
        "num_sequences": len(sequences),
        "best_val": best_val,
        "test": test_cos,
        "gap": best_val - test_cos,
    }

    with open(output_dir / f"next_vector_lm_{timestamp}.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
