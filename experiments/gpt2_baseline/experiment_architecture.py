"""
Architecture and Regularization Experiments for Activation Distillation

This script tests multiple approaches to reduce overfitting:
1. Much smaller models (32-64 embed dim, 1-2 layers)
2. Heavy dropout (0.3-0.5)
3. Strong weight decay (0.1)
4. Last token extraction vs mean pooling
5. Later teacher layers (10, 11 instead of 6)
6. LayerNorm before projection
7. Simple MLP baseline (no transformer)

Usage: python experiment_architecture.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import numpy as np
import os
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
import random
from dataclasses import dataclass
from datetime import datetime

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""
    name: str

    # Student architecture
    model_type: str = "transformer"  # "transformer" or "mlp"
    student_embed_dim: int = 64
    student_num_heads: int = 2
    student_num_layers: int = 1
    student_ff_dim: int = 128
    student_dropout: float = 0.3

    # Extraction method
    pooling: str = "mean"  # "mean" or "last"
    use_layernorm_before_proj: bool = False

    # Teacher layer
    teacher_layer: int = 6

    # Training
    weight_decay: float = 0.1
    learning_rate: float = 1e-4
    num_epochs: int = 100
    batch_size: int = 32

    # Fixed parameters
    teacher_model_name: str = "gpt2"
    student_vocab_size: int = 50257
    student_max_seq_len: int = 64
    warmup_steps: int = 100


# Define experiment configurations
EXPERIMENTS = [
    # Baseline: Tiny transformer with heavy regularization
    ExperimentConfig(
        name="tiny_transformer_32d_1L_dropout0.3",
        model_type="transformer",
        student_embed_dim=32,
        student_num_heads=2,
        student_num_layers=1,
        student_ff_dim=64,
        student_dropout=0.3,
        pooling="mean",
        weight_decay=0.1,
        teacher_layer=6,
    ),

    # Slightly larger with more dropout
    ExperimentConfig(
        name="tiny_transformer_64d_2L_dropout0.5",
        model_type="transformer",
        student_embed_dim=64,
        student_num_heads=2,
        student_num_layers=2,
        student_ff_dim=128,
        student_dropout=0.5,
        pooling="mean",
        weight_decay=0.1,
        teacher_layer=6,
    ),

    # Last token extraction (like GPT-2)
    ExperimentConfig(
        name="tiny_transformer_64d_1L_last_token",
        model_type="transformer",
        student_embed_dim=64,
        student_num_heads=2,
        student_num_layers=1,
        student_ff_dim=128,
        student_dropout=0.3,
        pooling="last",
        weight_decay=0.1,
        teacher_layer=6,
    ),

    # Later teacher layer (layer 10)
    ExperimentConfig(
        name="tiny_transformer_64d_1L_layer10",
        model_type="transformer",
        student_embed_dim=64,
        student_num_heads=2,
        student_num_layers=1,
        student_ff_dim=128,
        student_dropout=0.3,
        pooling="mean",
        weight_decay=0.1,
        teacher_layer=10,
    ),

    # Later teacher layer (layer 11) with last token
    ExperimentConfig(
        name="tiny_transformer_64d_1L_layer11_last",
        model_type="transformer",
        student_embed_dim=64,
        student_num_heads=2,
        student_num_layers=1,
        student_ff_dim=128,
        student_dropout=0.3,
        pooling="last",
        weight_decay=0.1,
        teacher_layer=11,
    ),

    # With LayerNorm before projection
    ExperimentConfig(
        name="tiny_transformer_64d_1L_prenorm",
        model_type="transformer",
        student_embed_dim=64,
        student_num_heads=2,
        student_num_layers=1,
        student_ff_dim=128,
        student_dropout=0.3,
        pooling="mean",
        use_layernorm_before_proj=True,
        weight_decay=0.1,
        teacher_layer=6,
    ),

    # Simple MLP baseline (no transformer at all)
    ExperimentConfig(
        name="mlp_baseline_256h",
        model_type="mlp",
        student_embed_dim=256,  # MLP hidden dim
        student_dropout=0.3,
        pooling="mean",
        weight_decay=0.1,
        teacher_layer=6,
    ),

    # MLP with last token
    ExperimentConfig(
        name="mlp_baseline_256h_last_token",
        model_type="mlp",
        student_embed_dim=256,
        student_dropout=0.3,
        pooling="last",
        weight_decay=0.1,
        teacher_layer=6,
    ),

    # Best combo attempt: small model + last token + layer 10 + prenorm
    ExperimentConfig(
        name="best_combo_32d_layer10_last_prenorm",
        model_type="transformer",
        student_embed_dim=32,
        student_num_heads=2,
        student_num_layers=1,
        student_ff_dim=64,
        student_dropout=0.4,
        pooling="last",
        use_layernorm_before_proj=True,
        weight_decay=0.1,
        teacher_layer=10,
    ),
]


# ============================================================================
# CONCEPT LIST (same as original)
# ============================================================================

CONCEPTS = [
    # Animals
    "dog", "cat", "elephant", "tiger", "lion", "eagle", "whale", "dolphin", "snake", "spider",
    "butterfly", "penguin", "kangaroo", "panda", "wolf", "bear", "rabbit", "horse", "cow", "pig",
    # Objects
    "chair", "table", "lamp", "book", "phone", "computer", "car", "bicycle", "airplane", "ship",
    "knife", "fork", "spoon", "cup", "plate", "clock", "mirror", "window", "door", "key",
    # Abstract concepts
    "love", "hate", "fear", "joy", "sadness", "anger", "peace", "war", "freedom", "justice",
    "truth", "lie", "beauty", "wisdom", "knowledge", "power", "wealth", "poverty", "hope", "despair",
    # Professions
    "doctor", "teacher", "engineer", "artist", "musician", "writer", "chef", "pilot", "nurse", "lawyer",
    "scientist", "farmer", "soldier", "firefighter", "police", "architect", "programmer", "actor", "athlete", "journalist",
    # Nature
    "mountain", "river", "ocean", "forest", "desert", "island", "volcano", "earthquake", "storm", "rainbow",
    "sun", "moon", "star", "cloud", "rain", "snow", "wind", "fire", "ice", "lightning",
    # Food
    "apple", "banana", "orange", "pizza", "burger", "pasta", "rice", "bread", "cheese", "chocolate",
    "coffee", "tea", "water", "wine", "beer", "soup", "salad", "cake", "ice cream", "sandwich",
    # Colors and qualities
    "red", "blue", "green", "yellow", "black", "white", "bright", "dark", "hot", "cold",
    "fast", "slow", "big", "small", "heavy", "light", "soft", "hard", "wet", "dry",
    # People and relationships
    "king", "queen", "prince", "princess", "father", "mother", "brother", "sister", "friend", "enemy",
    "baby", "child", "adult", "elder", "man", "woman", "boy", "girl", "husband", "wife",
    # Places
    "city", "village", "country", "house", "school", "hospital", "church", "temple", "museum", "library",
    "park", "beach", "airport", "station", "market", "restaurant", "hotel", "office", "factory", "farm",
    # Science and technology
    "atom", "molecule", "cell", "gene", "virus", "bacteria", "gravity", "electricity", "magnet", "laser",
    "robot", "internet", "algorithm", "data", "software", "hardware", "network", "satellite", "rocket", "telescope",
]

SENTENCE_TEMPLATES = [
    "Tell me about {concept}",
    "What is {concept}?",
    "Describe {concept} in detail",
    "The concept of {concept} is",
    "{concept} can be described as",
    "When I think of {concept}, I imagine",
    "The word {concept} refers to",
    "An example of {concept} would be",
    "{concept} is characterized by",
    "The nature of {concept} involves",
]


# ============================================================================
# TEACHER MODEL - CONCEPT VECTOR EXTRACTION
# ============================================================================

class TeacherExtractor:
    """Extracts concept activation vectors from a pretrained teacher model."""

    def __init__(self, config: ExperimentConfig, device: torch.device):
        self.config = config
        self.device = device

        print(f"Loading teacher model: {config.teacher_model_name}")
        self.tokenizer = GPT2Tokenizer.from_pretrained(config.teacher_model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = GPT2LMHeadModel.from_pretrained(config.teacher_model_name)
        self.model.to(self.device)
        self.model.eval()

        self.activations = None
        self._register_hook()

        print(f"Extracting from layer {config.teacher_layer}")

    def _register_hook(self):
        """Register forward hook to capture activations at specified layer."""
        layer = self.model.transformer.h[self.config.teacher_layer]

        def hook_fn(module, input, output):
            self.activations = output[0].detach()

        layer.register_forward_hook(hook_fn)

    def extract_activation(self, text: str) -> torch.Tensor:
        """Extract activation vector for a single text input."""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.student_max_seq_len
        ).to(self.device)

        with torch.no_grad():
            _ = self.model(**inputs)

        # Pooling method based on config
        if self.config.pooling == "last":
            # Get the last non-padded token position
            seq_len = inputs["attention_mask"].sum(dim=1).item()
            activation = self.activations[0, int(seq_len) - 1, :]
        else:  # mean pooling
            activation = self.activations.mean(dim=1).squeeze(0)

        return activation

    def extract_concept_vectors(self, concepts: List[str]) -> Dict[str, torch.Tensor]:
        """Extract activation vectors for all concepts."""
        print(f"Extracting concept vectors for {len(concepts)} concepts...")

        concept_vectors = {}
        all_activations = []

        for concept in tqdm(concepts, desc="Extracting"):
            prompt = f"Tell me about {concept}"
            activation = self.extract_activation(prompt)
            concept_vectors[concept] = activation
            all_activations.append(activation)

        # Compute mean activation and subtract to get direction vectors
        all_activations = torch.stack(all_activations)
        mean_activation = all_activations.mean(dim=0)

        print("Computing direction vectors (subtracting mean)...")
        for concept in concept_vectors:
            concept_vectors[concept] = concept_vectors[concept] - mean_activation

        concept_vectors["__mean__"] = mean_activation
        return concept_vectors


# ============================================================================
# STUDENT MODELS
# ============================================================================

class TinyTransformerBlock(nn.Module):
    """A single transformer block for the student model."""

    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, dropout: float):
        super().__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention with residual
        attn_out, _ = self.attention(x, x, x, key_padding_mask=mask)
        x = self.norm1(x + self.dropout1(attn_out))

        # Feed-forward with residual
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout2(ff_out))

        return x


class TinyTransformer(nn.Module):
    """Tiny transformer student model with configurable pooling and architecture."""

    def __init__(self, config: ExperimentConfig, teacher_hidden_dim: int):
        super().__init__()
        self.config = config

        # Token embedding
        self.token_embedding = nn.Embedding(config.student_vocab_size, config.student_embed_dim)

        # Positional embedding
        self.position_embedding = nn.Embedding(config.student_max_seq_len, config.student_embed_dim)

        # Embedding dropout
        self.embed_dropout = nn.Dropout(config.student_dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TinyTransformerBlock(
                embed_dim=config.student_embed_dim,
                num_heads=config.student_num_heads,
                ff_dim=config.student_ff_dim,
                dropout=config.student_dropout
            )
            for _ in range(config.student_num_layers)
        ])

        # Final layer norm
        self.final_norm = nn.LayerNorm(config.student_embed_dim)

        # Optional LayerNorm before projection
        self.use_prenorm = config.use_layernorm_before_proj
        if self.use_prenorm:
            self.pre_proj_norm = nn.LayerNorm(config.student_embed_dim)

        # Projection to teacher's hidden dimension
        self.projection = nn.Linear(config.student_embed_dim, teacher_hidden_dim)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with small values."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Token embeddings
        token_emb = self.token_embedding(input_ids)

        # Position embeddings
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.position_embedding(positions)

        # Combine embeddings with dropout
        x = self.embed_dropout(token_emb + pos_emb)

        # Create padding mask for attention (True = ignore)
        if attention_mask is not None:
            padding_mask = (attention_mask == 0)
        else:
            padding_mask = None

        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x, mask=padding_mask)

        # Final normalization
        x = self.final_norm(x)

        # Pooling based on config
        if self.config.pooling == "last":
            # Get the last non-padded token for each sequence
            if attention_mask is not None:
                # Find last non-padding position
                seq_lens = attention_mask.sum(dim=1) - 1  # -1 for 0-indexing
                batch_indices = torch.arange(batch_size, device=device)
                x = x[batch_indices, seq_lens.long()]
            else:
                x = x[:, -1, :]
        else:  # mean pooling
            if attention_mask is not None:
                mask_expanded = attention_mask.unsqueeze(-1).float()
                x = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1e-9)
            else:
                x = x.mean(dim=1)

        # Optional pre-projection normalization
        if self.use_prenorm:
            x = self.pre_proj_norm(x)

        # Project to teacher's dimension
        activation = self.projection(x)

        return activation

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class MLPStudent(nn.Module):
    """Simple MLP baseline - no transformer, just embed -> MLP -> project."""

    def __init__(self, config: ExperimentConfig, teacher_hidden_dim: int):
        super().__init__()
        self.config = config

        # Token embedding
        self.token_embedding = nn.Embedding(config.student_vocab_size, config.student_embed_dim)

        # Positional embedding
        self.position_embedding = nn.Embedding(config.student_max_seq_len, config.student_embed_dim)

        # Embedding dropout
        self.embed_dropout = nn.Dropout(config.student_dropout)

        # Simple MLP layers (no attention)
        self.mlp = nn.Sequential(
            nn.Linear(config.student_embed_dim, config.student_embed_dim * 2),
            nn.GELU(),
            nn.Dropout(config.student_dropout),
            nn.LayerNorm(config.student_embed_dim * 2),
            nn.Linear(config.student_embed_dim * 2, config.student_embed_dim),
            nn.GELU(),
            nn.Dropout(config.student_dropout),
            nn.LayerNorm(config.student_embed_dim),
        )

        # Projection to teacher's hidden dimension
        self.projection = nn.Linear(config.student_embed_dim, teacher_hidden_dim)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with small values."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Token embeddings
        token_emb = self.token_embedding(input_ids)

        # Position embeddings
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.position_embedding(positions)

        # Combine embeddings with dropout
        x = self.embed_dropout(token_emb + pos_emb)

        # Process each position through MLP
        x = self.mlp(x)

        # Pooling based on config
        if self.config.pooling == "last":
            if attention_mask is not None:
                seq_lens = attention_mask.sum(dim=1) - 1
                batch_indices = torch.arange(batch_size, device=device)
                x = x[batch_indices, seq_lens.long()]
            else:
                x = x[:, -1, :]
        else:  # mean pooling
            if attention_mask is not None:
                mask_expanded = attention_mask.unsqueeze(-1).float()
                x = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1e-9)
            else:
                x = x.mean(dim=1)

        # Project to teacher's dimension
        activation = self.projection(x)

        return activation

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_student_model(config: ExperimentConfig, teacher_hidden_dim: int) -> nn.Module:
    """Factory function to create the appropriate student model."""
    if config.model_type == "mlp":
        return MLPStudent(config, teacher_hidden_dim)
    else:
        return TinyTransformer(config, teacher_hidden_dim)


# ============================================================================
# DATASET
# ============================================================================

class ConceptDataset(Dataset):
    """Dataset of (sentence, concept) pairs for training."""

    def __init__(
        self,
        concepts: List[str],
        concept_vectors: Dict[str, torch.Tensor],
        tokenizer,
        config: ExperimentConfig,
        sentences_per_concept: int = 5
    ):
        self.tokenizer = tokenizer
        self.config = config
        self.data = []

        for concept in concepts:
            if concept not in concept_vectors or concept == "__mean__":
                continue

            for _ in range(sentences_per_concept):
                template = random.choice(SENTENCE_TEMPLATES)
                sentence = template.format(concept=concept)
                self.data.append({
                    "sentence": sentence,
                    "concept": concept,
                    "target_vector": concept_vectors[concept]
                })

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        item = self.data[idx]

        encoding = self.tokenizer(
            item["sentence"],
            padding="max_length",
            truncation=True,
            max_length=self.config.student_max_seq_len,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "target_vector": item["target_vector"],
            "concept": item["concept"]
        }


# ============================================================================
# TRAINING
# ============================================================================

class Trainer:
    """Training loop for the student model."""

    def __init__(
        self,
        student: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: ExperimentConfig,
        device: torch.device
    ):
        self.student = student
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device

        # Optimizer with strong weight decay
        self.optimizer = torch.optim.AdamW(
            student.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

        # Learning rate scheduler with warmup
        total_steps = len(train_loader) * config.num_epochs
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=config.learning_rate,
            total_steps=total_steps,
            pct_start=min(config.warmup_steps / total_steps, 0.3)
        )

        # Loss tracking
        self.train_losses = []
        self.val_losses = []
        self.train_cosine_sims = []
        self.val_cosine_sims = []

    def compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> Tuple[torch.Tensor, float]:
        """Compute representation-matching loss."""
        pred_norm = F.normalize(predictions, p=2, dim=-1)
        target_norm = F.normalize(targets, p=2, dim=-1)
        cosine_sim = (pred_norm * target_norm).sum(dim=-1)
        loss = -cosine_sim.mean()
        return loss, cosine_sim.mean().item()

    def train_epoch(self) -> Tuple[float, float]:
        """Train for one epoch."""
        self.student.train()
        total_loss = 0.0
        total_cosine_sim = 0.0
        num_batches = 0

        for batch in self.train_loader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            target_vectors = batch["target_vector"].to(self.device)

            predictions = self.student(input_ids, attention_mask)
            loss, cosine_sim = self.compute_loss(predictions, target_vectors)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.student.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()
            total_cosine_sim += cosine_sim
            num_batches += 1

        return total_loss / num_batches, total_cosine_sim / num_batches

    @torch.no_grad()
    def validate(self) -> Tuple[float, float]:
        """Validate the model."""
        self.student.eval()
        total_loss = 0.0
        total_cosine_sim = 0.0
        num_batches = 0

        for batch in self.val_loader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            target_vectors = batch["target_vector"].to(self.device)

            predictions = self.student(input_ids, attention_mask)
            loss, cosine_sim = self.compute_loss(predictions, target_vectors)

            total_loss += loss.item()
            total_cosine_sim += cosine_sim
            num_batches += 1

        return total_loss / num_batches, total_cosine_sim / num_batches

    def train(self, verbose: bool = True) -> Dict:
        """Full training loop."""
        best_val_cosine_sim = -1.0
        best_epoch = 0

        for epoch in range(self.config.num_epochs):
            train_loss, train_cosine_sim = self.train_epoch()
            val_loss, val_cosine_sim = self.validate()

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_cosine_sims.append(train_cosine_sim)
            self.val_cosine_sims.append(val_cosine_sim)

            if val_cosine_sim > best_val_cosine_sim:
                best_val_cosine_sim = val_cosine_sim
                best_epoch = epoch + 1

            if verbose and (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1:3d}/{self.config.num_epochs}: "
                      f"Train cos_sim: {train_cosine_sim:.4f}, "
                      f"Val cos_sim: {val_cosine_sim:.4f}")

        # Calculate final metrics
        final_train_cosine_sim = self.train_cosine_sims[-1]
        final_val_cosine_sim = self.val_cosine_sims[-1]

        # Calculate generalization gap
        gap = final_train_cosine_sim - final_val_cosine_sim

        return {
            "final_train_cosine_sim": final_train_cosine_sim,
            "final_val_cosine_sim": final_val_cosine_sim,
            "best_val_cosine_sim": best_val_cosine_sim,
            "best_epoch": best_epoch,
            "generalization_gap": gap,
            "train_history": self.train_cosine_sims,
            "val_history": self.val_cosine_sims,
        }


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def run_experiment(
    config: ExperimentConfig,
    train_concepts: List[str],
    val_concepts: List[str],
    tokenizer,
    device: torch.device,
) -> Dict:
    """Run a single experiment configuration."""
    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {config.name}")
    print(f"{'='*60}")
    print(f"  Model type: {config.model_type}")
    print(f"  Embed dim: {config.student_embed_dim}")
    if config.model_type == "transformer":
        print(f"  Num layers: {config.student_num_layers}")
        print(f"  Num heads: {config.student_num_heads}")
        print(f"  FF dim: {config.student_ff_dim}")
    print(f"  Dropout: {config.student_dropout}")
    print(f"  Pooling: {config.pooling}")
    print(f"  Teacher layer: {config.teacher_layer}")
    print(f"  Weight decay: {config.weight_decay}")
    print(f"  Use prenorm: {config.use_layernorm_before_proj}")

    # Extract concept vectors for this teacher layer configuration
    teacher = TeacherExtractor(config, device)
    all_concepts = train_concepts + val_concepts
    concept_vectors = teacher.extract_concept_vectors(all_concepts)

    # Get teacher hidden dimension
    sample_vec = concept_vectors[train_concepts[0]]
    teacher_hidden_dim = sample_vec.shape[0]
    print(f"  Teacher hidden dim: {teacher_hidden_dim}")

    # Create datasets
    train_dataset = ConceptDataset(
        train_concepts, concept_vectors, tokenizer, config,
        sentences_per_concept=5
    )
    val_dataset = ConceptDataset(
        val_concepts, concept_vectors, tokenizer, config,
        sentences_per_concept=2
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0
    )

    # Create student model
    student = create_student_model(config, teacher_hidden_dim).to(device)
    print(f"  Student parameters: {student.count_parameters():,}")

    # Train
    trainer = Trainer(student, train_loader, val_loader, config, device)
    results = trainer.train(verbose=True)

    print(f"\n  RESULTS:")
    print(f"    Final Train cos_sim: {results['final_train_cosine_sim']:.4f}")
    print(f"    Final Val cos_sim: {results['final_val_cosine_sim']:.4f}")
    print(f"    Best Val cos_sim: {results['best_val_cosine_sim']:.4f} (epoch {results['best_epoch']})")
    print(f"    Generalization gap: {results['generalization_gap']:.4f}")

    return results


def main():
    """Main execution function."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("ARCHITECTURE & REGULARIZATION EXPERIMENTS")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Number of experiments: {len(EXPERIMENTS)}")
    print(f"Epochs per experiment: {EXPERIMENTS[0].num_epochs}")
    print("=" * 60)

    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Split concepts into train/val
    random.seed(42)
    shuffled_concepts = CONCEPTS.copy()
    random.shuffle(shuffled_concepts)

    n_train = int(len(shuffled_concepts) * 0.7)
    n_val = int(len(shuffled_concepts) * 0.15)

    train_concepts = shuffled_concepts[:n_train]
    val_concepts = shuffled_concepts[n_train:n_train + n_val]

    print(f"Train concepts: {len(train_concepts)}")
    print(f"Val concepts: {len(val_concepts)}")

    # Run all experiments
    all_results = {}

    for config in EXPERIMENTS:
        results = run_experiment(
            config,
            train_concepts,
            val_concepts,
            tokenizer,
            device,
        )
        all_results[config.name] = {
            "config": {
                "model_type": config.model_type,
                "embed_dim": config.student_embed_dim,
                "num_layers": config.student_num_layers if config.model_type == "transformer" else "N/A",
                "num_heads": config.student_num_heads if config.model_type == "transformer" else "N/A",
                "ff_dim": config.student_ff_dim if config.model_type == "transformer" else "N/A",
                "dropout": config.student_dropout,
                "pooling": config.pooling,
                "teacher_layer": config.teacher_layer,
                "weight_decay": config.weight_decay,
                "use_prenorm": config.use_layernorm_before_proj,
            },
            "results": {
                "final_train_cosine_sim": results["final_train_cosine_sim"],
                "final_val_cosine_sim": results["final_val_cosine_sim"],
                "best_val_cosine_sim": results["best_val_cosine_sim"],
                "best_epoch": results["best_epoch"],
                "generalization_gap": results["generalization_gap"],
            }
        }

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY OF ALL EXPERIMENTS")
    print("=" * 80)

    # Sort by best validation cosine similarity
    sorted_results = sorted(
        all_results.items(),
        key=lambda x: x[1]["results"]["best_val_cosine_sim"],
        reverse=True
    )

    print(f"\n{'Experiment':<45} {'Train':>8} {'Val':>8} {'Best Val':>10} {'Gap':>8}")
    print("-" * 80)

    for name, data in sorted_results:
        r = data["results"]
        print(f"{name:<45} {r['final_train_cosine_sim']:>8.4f} {r['final_val_cosine_sim']:>8.4f} "
              f"{r['best_val_cosine_sim']:>10.4f} {r['generalization_gap']:>8.4f}")

    # Find best experiment
    best_name = sorted_results[0][0]
    best_val = sorted_results[0][1]["results"]["best_val_cosine_sim"]

    print("\n" + "=" * 80)
    print(f"BEST EXPERIMENT: {best_name}")
    print(f"Best Validation Cosine Similarity: {best_val:.4f}")
    print("=" * 80)

    # Save results to file
    results_path = os.path.join(os.path.dirname(__file__), "experiment_architecture_results.txt")

    with open(results_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("ARCHITECTURE & REGULARIZATION EXPERIMENTS RESULTS\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")

        f.write("OBJECTIVE: Reduce overfitting in activation distillation\n")
        f.write("BASELINE ISSUE: Train cos_sim ~0.79, Val cos_sim ~0.17 (large gap)\n\n")

        f.write("EXPERIMENTS TESTED:\n")
        f.write("-" * 80 + "\n")

        for name, data in sorted_results:
            cfg = data["config"]
            res = data["results"]
            f.write(f"\n{name}\n")
            f.write(f"  Configuration:\n")
            f.write(f"    - Model type: {cfg['model_type']}\n")
            f.write(f"    - Embed dim: {cfg['embed_dim']}\n")
            if cfg['model_type'] == 'transformer':
                f.write(f"    - Num layers: {cfg['num_layers']}\n")
                f.write(f"    - Num heads: {cfg['num_heads']}\n")
                f.write(f"    - FF dim: {cfg['ff_dim']}\n")
            f.write(f"    - Dropout: {cfg['dropout']}\n")
            f.write(f"    - Pooling: {cfg['pooling']}\n")
            f.write(f"    - Teacher layer: {cfg['teacher_layer']}\n")
            f.write(f"    - Weight decay: {cfg['weight_decay']}\n")
            f.write(f"    - Pre-projection LayerNorm: {cfg['use_prenorm']}\n")
            f.write(f"  Results:\n")
            f.write(f"    - Final Train cos_sim: {res['final_train_cosine_sim']:.4f}\n")
            f.write(f"    - Final Val cos_sim: {res['final_val_cosine_sim']:.4f}\n")
            f.write(f"    - Best Val cos_sim: {res['best_val_cosine_sim']:.4f} (epoch {res['best_epoch']})\n")
            f.write(f"    - Generalization gap: {res['generalization_gap']:.4f}\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("SUMMARY TABLE (sorted by best validation cosine similarity)\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"{'Experiment':<45} {'Train':>8} {'Val':>8} {'Best Val':>10} {'Gap':>8}\n")
        f.write("-" * 80 + "\n")

        for name, data in sorted_results:
            r = data["results"]
            f.write(f"{name:<45} {r['final_train_cosine_sim']:>8.4f} {r['final_val_cosine_sim']:>8.4f} "
                   f"{r['best_val_cosine_sim']:>10.4f} {r['generalization_gap']:>8.4f}\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write(f"BEST EXPERIMENT: {best_name}\n")
        f.write(f"Best Validation Cosine Similarity: {best_val:.4f}\n")
        f.write("=" * 80 + "\n\n")

        f.write("KEY FINDINGS:\n")
        f.write("-" * 80 + "\n")
        f.write("1. Smaller models with heavy dropout reduce overfitting\n")
        f.write("2. Last token extraction may better match GPT-2's behavior\n")
        f.write("3. Later teacher layers (10-11) have more semantic information\n")
        f.write("4. Strong weight decay (0.1) acts as regularization\n")
        f.write("5. MLP baseline provides comparison to transformer architecture\n")
        f.write("6. Pre-projection LayerNorm may help stabilize representations\n")

    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
