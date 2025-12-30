"""
Activation Vector Distillation: Train a tiny transformer to produce target activation vectors
from a larger teacher model (GPT-2 small).

This script:
1. Extracts concept vectors from GPT-2 small (teacher)
2. Trains a tiny transformer (student) to match these vectors
3. Evaluates generalization and composition
4. Visualizes alignment in 2D/3D

Usage: python activation_distillation.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import json
import os
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
import random

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    # Teacher model
    teacher_model_name = "gpt2"  # GPT-2 small (124M params)
    teacher_layer = 6  # Layer to extract activations from (GPT-2 small has 12 layers)

    # Student model
    student_vocab_size = 50257  # Same as GPT-2
    student_embed_dim = 128  # Narrow hidden dimension
    student_num_heads = 4
    student_num_layers = 3  # 2-4 layers
    student_ff_dim = 512
    student_max_seq_len = 64
    student_dropout = 0.1

    # Training
    batch_size = 32
    learning_rate = 1e-4
    num_epochs = 100
    warmup_steps = 100
    weight_decay = 0.01

    # Data
    num_concepts = 200  # Number of concepts to use
    train_split = 0.8
    sentences_per_concept = 5  # Training sentences per concept

    # Paths
    save_dir = "checkpoints"
    concept_vectors_path = "concept_vectors.pt"

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================================
# CONCEPT LIST
# ============================================================================

# ~200 diverse concepts for training and evaluation
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

# Composition test pairs (A + B should be near C)
COMPOSITION_TESTS = [
    ("king", "female", "queen"),
    ("man", "royalty", "king"),
    ("woman", "royalty", "queen"),
    ("dog", "small", "puppy"),
    ("cat", "small", "kitten"),
    ("father", "female", "mother"),
    ("brother", "female", "sister"),
    ("hot", "water", "steam"),
    ("cold", "water", "ice"),
    ("fast", "car", "racing"),
]

# Templates for generating training sentences
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

    def __init__(self, config: Config):
        self.config = config
        self.device = config.device

        print(f"Loading teacher model: {config.teacher_model_name}")
        self.tokenizer = GPT2Tokenizer.from_pretrained(config.teacher_model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = GPT2LMHeadModel.from_pretrained(config.teacher_model_name)
        self.model.to(self.device)
        self.model.eval()

        # Storage for activations captured by hook
        self.activations = None

        # Register hook to capture residual stream
        self._register_hook()

        print(f"Teacher model loaded on {self.device}")
        print(f"Extracting from layer {config.teacher_layer}")

    def _register_hook(self):
        """Register forward hook to capture activations at specified layer."""
        layer = self.model.transformer.h[self.config.teacher_layer]

        def hook_fn(module, input, output):
            # output is a tuple, first element is hidden states
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

        # Take the mean of the sequence dimension to get a single vector
        activation = self.activations.mean(dim=1).squeeze(0)  # [hidden_dim]
        return activation

    def extract_concept_vectors(self, concepts: List[str]) -> Dict[str, torch.Tensor]:
        """Extract activation vectors for all concepts."""
        print(f"Extracting concept vectors for {len(concepts)} concepts...")

        concept_vectors = {}
        all_activations = []

        for concept in tqdm(concepts):
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

        # Store mean for later use
        concept_vectors["__mean__"] = mean_activation

        return concept_vectors

    def save_concept_vectors(self, concept_vectors: Dict[str, torch.Tensor], path: str):
        """Save concept vectors to disk."""
        torch.save(concept_vectors, path)
        print(f"Saved concept vectors to {path}")

    @staticmethod
    def load_concept_vectors(path: str) -> Dict[str, torch.Tensor]:
        """Load concept vectors from disk."""
        return torch.load(path)


# ============================================================================
# STUDENT MODEL - TINY TRANSFORMER
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
        x = self.norm1(x + attn_out)

        # Feed-forward with residual
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)

        return x


class TinyTransformer(nn.Module):
    """
    Tiny transformer student model.
    Designed to produce activation vectors matching the teacher's concept vectors.
    """

    def __init__(self, config: Config, teacher_hidden_dim: int):
        super().__init__()
        self.config = config

        # Token embedding
        self.token_embedding = nn.Embedding(config.student_vocab_size, config.student_embed_dim)

        # Positional embedding
        self.position_embedding = nn.Embedding(config.student_max_seq_len, config.student_embed_dim)

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
        """
        Forward pass through the student model.

        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len] (1 for real tokens, 0 for padding)

        Returns:
            activation: [batch_size, teacher_hidden_dim] - the output activation vector
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Token embeddings
        token_emb = self.token_embedding(input_ids)  # [batch, seq, embed]

        # Position embeddings
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.position_embedding(positions)  # [batch, seq, embed]

        # Combine embeddings
        x = token_emb + pos_emb

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

        # Pool over sequence (mean of non-padded tokens)
        if attention_mask is not None:
            # Mask padded positions
            mask_expanded = attention_mask.unsqueeze(-1).float()
            x = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1e-9)
        else:
            x = x.mean(dim=1)

        # Project to teacher's dimension
        activation = self.projection(x)  # [batch, teacher_hidden_dim]

        return activation

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


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
        config: Config,
        sentences_per_concept: int = 5
    ):
        self.tokenizer = tokenizer
        self.config = config
        self.data = []

        for concept in concepts:
            if concept not in concept_vectors or concept == "__mean__":
                continue

            # Generate multiple training sentences per concept
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

        # Tokenize
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
        student: TinyTransformer,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Config
    ):
        self.student = student
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = config.device

        # Optimizer
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
            pct_start=config.warmup_steps / total_steps
        )

        # Loss tracking
        self.train_losses = []
        self.val_losses = []
        self.val_cosine_sims = []

    def compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> Tuple[torch.Tensor, float]:
        """
        Compute representation-matching loss.
        Loss = negative cosine similarity (we want to maximize similarity)
        """
        # Normalize vectors
        pred_norm = F.normalize(predictions, p=2, dim=-1)
        target_norm = F.normalize(targets, p=2, dim=-1)

        # Cosine similarity
        cosine_sim = (pred_norm * target_norm).sum(dim=-1)

        # Loss is negative cosine similarity (minimize to maximize similarity)
        loss = -cosine_sim.mean()

        return loss, cosine_sim.mean().item()

    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.student.train()
        total_loss = 0.0
        num_batches = 0

        for batch in tqdm(self.train_loader, desc="Training"):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            target_vectors = batch["target_vector"].to(self.device)

            # Forward pass
            predictions = self.student(input_ids, attention_mask)

            # Compute loss
            loss, _ = self.compute_loss(predictions, target_vectors)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.student.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches

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

    def train(self):
        """Full training loop."""
        print(f"\nStarting training for {self.config.num_epochs} epochs")
        print(f"Student model parameters: {self.student.count_parameters():,}")

        best_val_loss = float("inf")

        for epoch in range(self.config.num_epochs):
            # Train
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)

            # Validate
            val_loss, val_cosine_sim = self.validate()
            self.val_losses.append(val_loss)
            self.val_cosine_sims.append(val_cosine_sim)

            print(f"Epoch {epoch+1}/{self.config.num_epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Val Cosine Sim: {val_cosine_sim:.4f}")

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint("best_model.pt")

        print(f"\nTraining complete. Best validation loss: {best_val_loss:.4f}")

    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        os.makedirs(self.config.save_dir, exist_ok=True)
        path = os.path.join(self.config.save_dir, filename)
        torch.save({
            "model_state_dict": self.student.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "val_cosine_sims": self.val_cosine_sims
        }, path)

    def plot_training_curves(self):
        """Plot training and validation loss curves."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        ax1.plot(self.train_losses, label="Train Loss")
        ax1.plot(self.val_losses, label="Val Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss (Neg Cosine Sim)")
        ax1.set_title("Training Curves")
        ax1.legend()
        ax1.grid(True)

        ax2.plot(self.val_cosine_sims, label="Val Cosine Similarity", color="green")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Cosine Similarity")
        ax2.set_title("Validation Cosine Similarity")
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig("training_curves.png", dpi=150)
        plt.show()


# ============================================================================
# EVALUATION
# ============================================================================

class Evaluator:
    """Evaluate student model on various tasks."""

    def __init__(
        self,
        student: TinyTransformer,
        concept_vectors: Dict[str, torch.Tensor],
        tokenizer,
        config: Config
    ):
        self.student = student
        self.concept_vectors = concept_vectors
        self.tokenizer = tokenizer
        self.config = config
        self.device = config.device
        self.student.eval()

    @torch.no_grad()
    def get_student_vector(self, text: str) -> torch.Tensor:
        """Get student's activation vector for a text input."""
        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.config.student_max_seq_len,
            return_tensors="pt"
        )

        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        return self.student(input_ids, attention_mask).squeeze(0)

    def evaluate_concept_matching(
        self,
        concepts: List[str],
        name: str = "Test"
    ) -> Dict:
        """Evaluate how well student matches teacher vectors for concepts."""
        results = {
            "concept_similarities": {},
            "mean_similarity": 0.0,
            "top_1_accuracy": 0.0,
            "top_5_accuracy": 0.0
        }

        all_target_vectors = []
        all_target_concepts = []
        for c in concepts:
            if c in self.concept_vectors and c != "__mean__":
                all_target_vectors.append(self.concept_vectors[c])
                all_target_concepts.append(c)

        all_target_vectors = torch.stack(all_target_vectors).to(self.device)
        all_target_vectors_norm = F.normalize(all_target_vectors, p=2, dim=-1)

        similarities = []
        top_1_correct = 0
        top_5_correct = 0

        for concept in tqdm(concepts, desc=f"Evaluating {name}"):
            if concept not in self.concept_vectors or concept == "__mean__":
                continue

            prompt = f"Tell me about {concept}"
            student_vec = self.get_student_vector(prompt)
            student_vec_norm = F.normalize(student_vec.unsqueeze(0), p=2, dim=-1)

            # Similarity to target
            target_vec = self.concept_vectors[concept].to(self.device)
            target_vec_norm = F.normalize(target_vec.unsqueeze(0), p=2, dim=-1)
            sim = (student_vec_norm * target_vec_norm).sum().item()

            results["concept_similarities"][concept] = sim
            similarities.append(sim)

            # Retrieval accuracy: find nearest neighbor
            all_sims = (student_vec_norm @ all_target_vectors_norm.T).squeeze(0)
            top_5_indices = all_sims.topk(5).indices.tolist()
            top_5_concepts = [all_target_concepts[i] for i in top_5_indices]

            if top_5_concepts[0] == concept:
                top_1_correct += 1
            if concept in top_5_concepts:
                top_5_correct += 1

        results["mean_similarity"] = np.mean(similarities)
        results["top_1_accuracy"] = top_1_correct / len(similarities)
        results["top_5_accuracy"] = top_5_correct / len(similarities)

        print(f"\n{name} Evaluation Results:")
        print(f"  Mean Cosine Similarity: {results['mean_similarity']:.4f}")
        print(f"  Top-1 Retrieval Accuracy: {results['top_1_accuracy']:.4f}")
        print(f"  Top-5 Retrieval Accuracy: {results['top_5_accuracy']:.4f}")

        return results

    def evaluate_composition(self) -> Dict:
        """Test if student learns compositional representations."""
        results = {
            "composition_tests": [],
            "mean_similarity_to_target": 0.0
        }

        print("\nComposition Evaluation:")
        similarities = []

        for concept_a, concept_b, expected in COMPOSITION_TESTS:
            if (concept_a not in self.concept_vectors or
                concept_b not in self.concept_vectors or
                expected not in self.concept_vectors):
                continue

            # Get vectors
            vec_a = self.concept_vectors[concept_a].to(self.device)
            vec_b = self.concept_vectors[concept_b].to(self.device)
            vec_expected = self.concept_vectors[expected].to(self.device)

            # Simple vector addition for composition
            composed = vec_a + vec_b

            # Normalize
            composed_norm = F.normalize(composed.unsqueeze(0), p=2, dim=-1)
            expected_norm = F.normalize(vec_expected.unsqueeze(0), p=2, dim=-1)

            sim = (composed_norm * expected_norm).sum().item()
            similarities.append(sim)

            results["composition_tests"].append({
                "a": concept_a,
                "b": concept_b,
                "expected": expected,
                "similarity": sim
            })

            print(f"  {concept_a} + {concept_b} -> {expected}: sim = {sim:.4f}")

        results["mean_similarity_to_target"] = np.mean(similarities) if similarities else 0.0
        print(f"  Mean Composition Similarity: {results['mean_similarity_to_target']:.4f}")

        return results


# ============================================================================
# VISUALIZATION
# ============================================================================

class Visualizer:
    """Visualize teacher and student activation spaces."""

    def __init__(
        self,
        student: TinyTransformer,
        concept_vectors: Dict[str, torch.Tensor],
        tokenizer,
        config: Config
    ):
        self.student = student
        self.concept_vectors = concept_vectors
        self.tokenizer = tokenizer
        self.config = config
        self.device = config.device
        self.student.eval()

    @torch.no_grad()
    def get_student_vector(self, text: str) -> torch.Tensor:
        """Get student's activation vector for a text input."""
        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.config.student_max_seq_len,
            return_tensors="pt"
        )

        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        return self.student(input_ids, attention_mask).squeeze(0).cpu()

    def visualize_alignment_2d(self, concepts: List[str], method: str = "pca"):
        """
        Visualize teacher targets and student outputs in 2D.

        Args:
            concepts: List of concepts to visualize
            method: 'pca' or 'tsne'
        """
        teacher_vectors = []
        student_vectors = []
        valid_concepts = []

        for concept in concepts:
            if concept in self.concept_vectors and concept != "__mean__":
                teacher_vec = self.concept_vectors[concept].cpu().numpy()

                prompt = f"Tell me about {concept}"
                student_vec = self.get_student_vector(prompt).numpy()

                teacher_vectors.append(teacher_vec)
                student_vectors.append(student_vec)
                valid_concepts.append(concept)

        teacher_vectors = np.array(teacher_vectors)
        student_vectors = np.array(student_vectors)

        # Combine for joint dimensionality reduction
        all_vectors = np.vstack([teacher_vectors, student_vectors])

        if method == "pca":
            reducer = PCA(n_components=2)
        else:
            reducer = TSNE(n_components=2, perplexity=min(30, len(valid_concepts)-1))

        all_2d = reducer.fit_transform(all_vectors)
        n = len(valid_concepts)
        teacher_2d = all_2d[:n]
        student_2d = all_2d[n:]

        # Plot
        fig, ax = plt.subplots(figsize=(12, 10))

        # Plot teacher vectors
        ax.scatter(teacher_2d[:, 0], teacher_2d[:, 1], c='blue', marker='o',
                   s=100, alpha=0.6, label='Teacher')

        # Plot student vectors
        ax.scatter(student_2d[:, 0], student_2d[:, 1], c='red', marker='x',
                   s=100, alpha=0.6, label='Student')

        # Draw lines connecting teacher-student pairs
        for i in range(n):
            ax.plot([teacher_2d[i, 0], student_2d[i, 0]],
                    [teacher_2d[i, 1], student_2d[i, 1]],
                    'gray', alpha=0.3, linewidth=1)

        # Add labels for a subset of concepts
        for i, concept in enumerate(valid_concepts[:30]):  # Label first 30
            ax.annotate(concept, (teacher_2d[i, 0], teacher_2d[i, 1]),
                        fontsize=8, alpha=0.7)

        ax.set_title(f"Teacher vs Student Activation Space ({method.upper()})")
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"alignment_2d_{method}.png", dpi=150)
        plt.show()

    def visualize_alignment_3d(self, concepts: List[str]):
        """Visualize teacher targets and student outputs in 3D using PCA."""
        teacher_vectors = []
        student_vectors = []
        valid_concepts = []

        for concept in concepts:
            if concept in self.concept_vectors and concept != "__mean__":
                teacher_vec = self.concept_vectors[concept].cpu().numpy()

                prompt = f"Tell me about {concept}"
                student_vec = self.get_student_vector(prompt).numpy()

                teacher_vectors.append(teacher_vec)
                student_vectors.append(student_vec)
                valid_concepts.append(concept)

        teacher_vectors = np.array(teacher_vectors)
        student_vectors = np.array(student_vectors)

        # Joint PCA
        all_vectors = np.vstack([teacher_vectors, student_vectors])
        pca = PCA(n_components=3)
        all_3d = pca.fit_transform(all_vectors)

        n = len(valid_concepts)
        teacher_3d = all_3d[:n]
        student_3d = all_3d[n:]

        # Plot
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Plot teacher vectors
        ax.scatter(teacher_3d[:, 0], teacher_3d[:, 1], teacher_3d[:, 2],
                   c='blue', marker='o', s=100, alpha=0.6, label='Teacher')

        # Plot student vectors
        ax.scatter(student_3d[:, 0], student_3d[:, 1], student_3d[:, 2],
                   c='red', marker='x', s=100, alpha=0.6, label='Student')

        # Draw lines connecting pairs
        for i in range(n):
            ax.plot([teacher_3d[i, 0], student_3d[i, 0]],
                    [teacher_3d[i, 1], student_3d[i, 1]],
                    [teacher_3d[i, 2], student_3d[i, 2]],
                    'gray', alpha=0.3, linewidth=1)

        ax.set_title("Teacher vs Student Activation Space (3D PCA)")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")
        ax.legend()

        plt.tight_layout()
        plt.savefig("alignment_3d.png", dpi=150)
        plt.show()

    def visualize_similarity_heatmap(self, concepts: List[str]):
        """Visualize similarity matrix between teacher and student vectors."""
        teacher_vectors = []
        student_vectors = []
        valid_concepts = []

        for concept in concepts[:50]:  # Limit for visibility
            if concept in self.concept_vectors and concept != "__mean__":
                teacher_vec = self.concept_vectors[concept]

                prompt = f"Tell me about {concept}"
                student_vec = self.get_student_vector(prompt)

                teacher_vectors.append(teacher_vec)
                student_vectors.append(student_vec)
                valid_concepts.append(concept)

        teacher_matrix = torch.stack(teacher_vectors)
        student_matrix = torch.stack(student_vectors)

        # Normalize
        teacher_norm = F.normalize(teacher_matrix, p=2, dim=-1)
        student_norm = F.normalize(student_matrix, p=2, dim=-1)

        # Compute similarity matrix
        sim_matrix = (student_norm @ teacher_norm.T).numpy()

        # Plot
        fig, ax = plt.subplots(figsize=(14, 12))
        im = ax.imshow(sim_matrix, cmap='RdYlBu', vmin=-1, vmax=1)

        ax.set_xticks(range(len(valid_concepts)))
        ax.set_yticks(range(len(valid_concepts)))
        ax.set_xticklabels(valid_concepts, rotation=90, fontsize=8)
        ax.set_yticklabels(valid_concepts, fontsize=8)

        ax.set_xlabel("Teacher Concepts")
        ax.set_ylabel("Student Concepts")
        ax.set_title("Student-Teacher Similarity Matrix")

        plt.colorbar(im, ax=ax, label="Cosine Similarity")
        plt.tight_layout()
        plt.savefig("similarity_heatmap.png", dpi=150)
        plt.show()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    config = Config()

    print("=" * 60)
    print("ACTIVATION VECTOR DISTILLATION")
    print("=" * 60)
    print(f"Device: {config.device}")
    print(f"Teacher: {config.teacher_model_name}")
    print(f"Number of concepts: {len(CONCEPTS)}")
    print("=" * 60)

    # Step 1: Extract concept vectors from teacher
    print("\n[STEP 1] Extracting concept vectors from teacher model...")

    if os.path.exists(config.concept_vectors_path):
        print(f"Loading existing concept vectors from {config.concept_vectors_path}")
        concept_vectors = TeacherExtractor.load_concept_vectors(config.concept_vectors_path)
    else:
        teacher = TeacherExtractor(config)
        concept_vectors = teacher.extract_concept_vectors(CONCEPTS)
        teacher.save_concept_vectors(concept_vectors, config.concept_vectors_path)

    # Get teacher hidden dimension
    sample_vec = concept_vectors[CONCEPTS[0]]
    teacher_hidden_dim = sample_vec.shape[0]
    print(f"Teacher hidden dimension: {teacher_hidden_dim}")

    # Step 2: Prepare datasets
    print("\n[STEP 2] Preparing datasets...")

    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(config.teacher_model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Split concepts into train/val/test
    random.seed(42)
    shuffled_concepts = CONCEPTS.copy()
    random.shuffle(shuffled_concepts)

    n_train = int(len(shuffled_concepts) * 0.7)
    n_val = int(len(shuffled_concepts) * 0.15)

    train_concepts = shuffled_concepts[:n_train]
    val_concepts = shuffled_concepts[n_train:n_train + n_val]
    test_concepts = shuffled_concepts[n_train + n_val:]

    print(f"Train concepts: {len(train_concepts)}")
    print(f"Val concepts: {len(val_concepts)}")
    print(f"Test concepts: {len(test_concepts)}")

    # Create datasets
    train_dataset = ConceptDataset(
        train_concepts, concept_vectors, tokenizer, config,
        sentences_per_concept=config.sentences_per_concept
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

    # Step 3: Initialize student model
    print("\n[STEP 3] Initializing student model...")

    student = TinyTransformer(config, teacher_hidden_dim).to(config.device)
    print(f"Student parameters: {student.count_parameters():,}")
    print(f"Student architecture:")
    print(f"  - Embedding dim: {config.student_embed_dim}")
    print(f"  - Num layers: {config.student_num_layers}")
    print(f"  - Num heads: {config.student_num_heads}")
    print(f"  - FF dim: {config.student_ff_dim}")

    # Step 4: Train
    print("\n[STEP 4] Training student model...")

    trainer = Trainer(student, train_loader, val_loader, config)
    trainer.train()
    trainer.plot_training_curves()

    # Step 5: Evaluate
    print("\n[STEP 5] Evaluation...")

    # Load best model
    checkpoint = torch.load(os.path.join(config.save_dir, "best_model.pt"))
    student.load_state_dict(checkpoint["model_state_dict"])

    evaluator = Evaluator(student, concept_vectors, tokenizer, config)

    # Evaluate on training concepts
    train_results = evaluator.evaluate_concept_matching(train_concepts, "Training")

    # Evaluate on held-out concepts (generalization)
    test_results = evaluator.evaluate_concept_matching(test_concepts, "Held-out (Generalization)")

    # Evaluate composition
    composition_results = evaluator.evaluate_composition()

    # Step 6: Visualization
    print("\n[STEP 6] Visualization...")

    visualizer = Visualizer(student, concept_vectors, tokenizer, config)

    # 2D visualization
    all_concepts = train_concepts + test_concepts
    visualizer.visualize_alignment_2d(all_concepts[:100], method="pca")
    visualizer.visualize_alignment_2d(all_concepts[:100], method="tsne")

    # 3D visualization
    visualizer.visualize_alignment_3d(all_concepts[:100])

    # Similarity heatmap
    visualizer.visualize_similarity_heatmap(all_concepts)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Training Set Performance:")
    print(f"  - Mean Cosine Similarity: {train_results['mean_similarity']:.4f}")
    print(f"  - Top-1 Accuracy: {train_results['top_1_accuracy']:.4f}")
    print(f"  - Top-5 Accuracy: {train_results['top_5_accuracy']:.4f}")
    print(f"\nGeneralization (Held-out Concepts):")
    print(f"  - Mean Cosine Similarity: {test_results['mean_similarity']:.4f}")
    print(f"  - Top-1 Accuracy: {test_results['top_1_accuracy']:.4f}")
    print(f"  - Top-5 Accuracy: {test_results['top_5_accuracy']:.4f}")
    print(f"\nComposition:")
    print(f"  - Mean Similarity to Expected: {composition_results['mean_similarity_to_target']:.4f}")
    print("=" * 60)

    print("\nGenerated files:")
    print("  - concept_vectors.pt (teacher concept vectors)")
    print("  - checkpoints/best_model.pt (trained student model)")
    print("  - training_curves.png")
    print("  - alignment_2d_pca.png")
    print("  - alignment_2d_tsne.png")
    print("  - alignment_3d.png")
    print("  - similarity_heatmap.png")


if __name__ == "__main__":
    main()
