"""
Contrastive Learning for Activation Vector Distillation

This experiment addresses overfitting in the original activation distillation by:
1. Using InfoNCE/contrastive loss with negative sampling
2. Temperature scaling for sharper/softer distributions
3. Hard negative mining (semantically similar concepts as negatives)
4. Smaller model capacity (64 embed dim, 2 layers)

The hypothesis is that contrastive learning will force the student to learn
discriminative features rather than memorizing specific training examples.
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
from datetime import datetime

# ============================================================================
# CONFIGURATION
# ============================================================================

class ContrastiveConfig:
    # Teacher model
    teacher_model_name = "gpt2"
    teacher_layer = 6

    # Student model - REDUCED CAPACITY to fight overfitting
    student_vocab_size = 50257
    student_embed_dim = 64  # Reduced from 128
    student_num_heads = 2   # Reduced from 4
    student_num_layers = 2  # Reduced from 3
    student_ff_dim = 256    # Reduced from 512
    student_max_seq_len = 64
    student_dropout = 0.2   # Increased from 0.1

    # Contrastive learning parameters
    num_negatives = 15        # Number of negative samples per positive
    temperature = 0.1         # Temperature for InfoNCE (will try 0.07-0.5)
    hard_negative_ratio = 0.5 # Ratio of hard negatives vs random negatives
    cosine_loss_weight = 0.5  # Weight for direct cosine similarity loss (NEW FIX)

    # Training
    batch_size = 32
    learning_rate = 3e-4
    num_epochs = 100
    warmup_steps = 200
    weight_decay = 0.05  # Increased regularization

    # Data
    num_concepts = 200
    train_split = 0.7
    sentences_per_concept = 5

    # Paths
    save_dir = "checkpoints_contrastive"
    concept_vectors_path = "concept_vectors.pt"
    results_path = "experiment_contrastive_results.txt"

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

# Semantic categories for hard negative mining
SEMANTIC_CATEGORIES = {
    "animals": ["dog", "cat", "elephant", "tiger", "lion", "eagle", "whale", "dolphin", "snake", "spider",
                "butterfly", "penguin", "kangaroo", "panda", "wolf", "bear", "rabbit", "horse", "cow", "pig"],
    "objects": ["chair", "table", "lamp", "book", "phone", "computer", "car", "bicycle", "airplane", "ship",
                "knife", "fork", "spoon", "cup", "plate", "clock", "mirror", "window", "door", "key"],
    "abstract": ["love", "hate", "fear", "joy", "sadness", "anger", "peace", "war", "freedom", "justice",
                 "truth", "lie", "beauty", "wisdom", "knowledge", "power", "wealth", "poverty", "hope", "despair"],
    "professions": ["doctor", "teacher", "engineer", "artist", "musician", "writer", "chef", "pilot", "nurse", "lawyer",
                    "scientist", "farmer", "soldier", "firefighter", "police", "architect", "programmer", "actor", "athlete", "journalist"],
    "nature": ["mountain", "river", "ocean", "forest", "desert", "island", "volcano", "earthquake", "storm", "rainbow",
               "sun", "moon", "star", "cloud", "rain", "snow", "wind", "fire", "ice", "lightning"],
    "food": ["apple", "banana", "orange", "pizza", "burger", "pasta", "rice", "bread", "cheese", "chocolate",
             "coffee", "tea", "water", "wine", "beer", "soup", "salad", "cake", "ice cream", "sandwich"],
    "qualities": ["red", "blue", "green", "yellow", "black", "white", "bright", "dark", "hot", "cold",
                  "fast", "slow", "big", "small", "heavy", "light", "soft", "hard", "wet", "dry"],
    "people": ["king", "queen", "prince", "princess", "father", "mother", "brother", "sister", "friend", "enemy",
               "baby", "child", "adult", "elder", "man", "woman", "boy", "girl", "husband", "wife"],
    "places": ["city", "village", "country", "house", "school", "hospital", "church", "temple", "museum", "library",
               "park", "beach", "airport", "station", "market", "restaurant", "hotel", "office", "factory", "farm"],
    "science": ["atom", "molecule", "cell", "gene", "virus", "bacteria", "gravity", "electricity", "magnet", "laser",
                "robot", "internet", "algorithm", "data", "software", "hardware", "network", "satellite", "rocket", "telescope"],
}

# Build reverse mapping: concept -> category
CONCEPT_TO_CATEGORY = {}
for category, concepts in SEMANTIC_CATEGORIES.items():
    for concept in concepts:
        CONCEPT_TO_CATEGORY[concept] = category

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
# TEACHER MODEL
# ============================================================================

class TeacherExtractor:
    """Extracts concept activation vectors from teacher model."""

    def __init__(self, config: ContrastiveConfig):
        self.config = config
        self.device = config.device

        print(f"Loading teacher model: {config.teacher_model_name}")
        self.tokenizer = GPT2Tokenizer.from_pretrained(config.teacher_model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = GPT2LMHeadModel.from_pretrained(config.teacher_model_name)
        self.model.to(self.device)
        self.model.eval()

        self.activations = None
        self._register_hook()

        print(f"Teacher model loaded on {self.device}")

    def _register_hook(self):
        layer = self.model.transformer.h[self.config.teacher_layer]

        def hook_fn(module, input, output):
            self.activations = output[0].detach()

        layer.register_forward_hook(hook_fn)

    def extract_activation(self, text: str) -> torch.Tensor:
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.student_max_seq_len
        ).to(self.device)

        with torch.no_grad():
            _ = self.model(**inputs)

        activation = self.activations.mean(dim=1).squeeze(0)
        return activation

    def extract_concept_vectors(self, concepts: List[str]) -> Dict[str, torch.Tensor]:
        print(f"Extracting concept vectors for {len(concepts)} concepts...")

        concept_vectors = {}
        all_activations = []

        for concept in tqdm(concepts):
            prompt = f"Tell me about {concept}"
            activation = self.extract_activation(prompt)
            concept_vectors[concept] = activation
            all_activations.append(activation)

        all_activations = torch.stack(all_activations)
        mean_activation = all_activations.mean(dim=0)

        print("Computing direction vectors (subtracting mean)...")
        for concept in concept_vectors:
            concept_vectors[concept] = concept_vectors[concept] - mean_activation

        concept_vectors["__mean__"] = mean_activation

        return concept_vectors

    def save_concept_vectors(self, concept_vectors: Dict[str, torch.Tensor], path: str):
        torch.save(concept_vectors, path)
        print(f"Saved concept vectors to {path}")

    @staticmethod
    def load_concept_vectors(path: str) -> Dict[str, torch.Tensor]:
        return torch.load(path)


# ============================================================================
# HARD NEGATIVE MINING
# ============================================================================

class HardNegativeMiner:
    """
    Mines hard negatives for contrastive learning.
    Hard negatives are concepts from the same semantic category.
    """

    def __init__(self, concept_vectors: Dict[str, torch.Tensor], all_concepts: List[str]):
        self.concept_vectors = concept_vectors
        self.all_concepts = [c for c in all_concepts if c in concept_vectors and c != "__mean__"]

        # Precompute similarity matrix for vector-based hard negatives
        self._precompute_similarities()

    def _precompute_similarities(self):
        """Precompute pairwise similarities between all concept vectors."""
        vectors = []
        for concept in self.all_concepts:
            vectors.append(self.concept_vectors[concept])

        vectors = torch.stack(vectors)
        vectors_norm = F.normalize(vectors, p=2, dim=-1)

        # Similarity matrix
        self.similarity_matrix = vectors_norm @ vectors_norm.T
        self.concept_to_idx = {c: i for i, c in enumerate(self.all_concepts)}

    def get_hard_negatives(
        self,
        target_concept: str,
        num_hard: int,
        num_random: int,
        exclude: List[str] = None
    ) -> List[str]:
        """
        Get hard and random negative concepts.

        Hard negatives: concepts from same semantic category OR
                        concepts with high embedding similarity
        Random negatives: randomly sampled concepts
        """
        exclude = set(exclude or [])
        exclude.add(target_concept)

        available = [c for c in self.all_concepts if c not in exclude]

        hard_negatives = []
        random_negatives = []

        # Strategy 1: Same semantic category
        target_category = CONCEPT_TO_CATEGORY.get(target_concept)
        if target_category:
            category_concepts = [
                c for c in SEMANTIC_CATEGORIES.get(target_category, [])
                if c in available and c != target_concept
            ]
            hard_negatives.extend(category_concepts[:num_hard // 2])

        # Strategy 2: High embedding similarity (but not the same concept)
        if target_concept in self.concept_to_idx:
            idx = self.concept_to_idx[target_concept]
            sims = self.similarity_matrix[idx].clone()

            # Mask out unavailable concepts
            for i, c in enumerate(self.all_concepts):
                if c in exclude or c in hard_negatives:
                    sims[i] = -float('inf')

            # Get top-k similar concepts
            remaining_hard = num_hard - len(hard_negatives)
            if remaining_hard > 0:
                top_indices = sims.topk(remaining_hard).indices.tolist()
                for i in top_indices:
                    if self.all_concepts[i] not in exclude:
                        hard_negatives.append(self.all_concepts[i])

        # Random negatives from remaining concepts
        remaining = [c for c in available if c not in hard_negatives]
        if remaining:
            random_negatives = random.sample(remaining, min(num_random, len(remaining)))

        return hard_negatives[:num_hard] + random_negatives[:num_random]


# ============================================================================
# STUDENT MODEL (SMALLER)
# ============================================================================

class TinyTransformerBlock(nn.Module):
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
        attn_out, _ = self.attention(x, x, x, key_padding_mask=mask)
        x = self.norm1(x + attn_out)
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        return x


class TinyTransformerContrastive(nn.Module):
    """Smaller transformer for contrastive learning."""

    def __init__(self, config: ContrastiveConfig, teacher_hidden_dim: int):
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config.student_vocab_size, config.student_embed_dim)
        self.position_embedding = nn.Embedding(config.student_max_seq_len, config.student_embed_dim)

        self.blocks = nn.ModuleList([
            TinyTransformerBlock(
                embed_dim=config.student_embed_dim,
                num_heads=config.student_num_heads,
                ff_dim=config.student_ff_dim,
                dropout=config.student_dropout
            )
            for _ in range(config.student_num_layers)
        ])

        self.final_norm = nn.LayerNorm(config.student_embed_dim)
        self.projection = nn.Linear(config.student_embed_dim, teacher_hidden_dim)

        self._init_weights()

    def _init_weights(self):
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

        token_emb = self.token_embedding(input_ids)
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.position_embedding(positions)

        x = token_emb + pos_emb

        if attention_mask is not None:
            padding_mask = (attention_mask == 0)
        else:
            padding_mask = None

        for block in self.blocks:
            x = block(x, mask=padding_mask)

        x = self.final_norm(x)

        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).float()
            x = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1e-9)
        else:
            x = x.mean(dim=1)

        activation = self.projection(x)
        return activation

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================================
# CONTRASTIVE DATASET
# ============================================================================

class ContrastiveDataset(Dataset):
    """
    Dataset that provides positive pairs and samples negative concepts for contrastive learning.
    """

    def __init__(
        self,
        concepts: List[str],
        concept_vectors: Dict[str, torch.Tensor],
        all_concepts: List[str],
        tokenizer,
        config: ContrastiveConfig,
        sentences_per_concept: int = 5
    ):
        self.tokenizer = tokenizer
        self.config = config
        self.concept_vectors = concept_vectors
        self.data = []

        # Initialize hard negative miner
        self.negative_miner = HardNegativeMiner(concept_vectors, all_concepts)

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
        concept = item["concept"]

        # Get negative samples
        num_hard = int(self.config.num_negatives * self.config.hard_negative_ratio)
        num_random = self.config.num_negatives - num_hard

        negative_concepts = self.negative_miner.get_hard_negatives(
            concept, num_hard, num_random
        )

        # Pad if we don't have enough negatives
        while len(negative_concepts) < self.config.num_negatives:
            available = [c for c in self.concept_vectors.keys()
                        if c != concept and c != "__mean__" and c not in negative_concepts]
            if available:
                negative_concepts.append(random.choice(available))
            else:
                break

        # Get negative vectors
        negative_vectors = torch.stack([
            self.concept_vectors[c] for c in negative_concepts[:self.config.num_negatives]
        ])

        # Tokenize sentence
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
            "target_vector": item["target_vector"],  # Positive
            "negative_vectors": negative_vectors,     # Negatives
            "concept": item["concept"],
            "negative_concepts": negative_concepts[:self.config.num_negatives]
        }


# ============================================================================
# CONTRASTIVE LOSS (InfoNCE)
# ============================================================================

class InfoNCELoss(nn.Module):
    """
    InfoNCE / Contrastive Loss

    Loss = -log(exp(sim(anchor, positive) / T) / sum(exp(sim(anchor, all) / T)))

    where:
    - anchor: student's prediction
    - positive: teacher's target concept vector
    - all: positive + all negatives
    - T: temperature (lower = sharper distribution)
    """

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        predictions: torch.Tensor,     # [batch, hidden_dim]
        positive_targets: torch.Tensor, # [batch, hidden_dim]
        negative_targets: torch.Tensor  # [batch, num_neg, hidden_dim]
    ) -> Tuple[torch.Tensor, float, float]:
        """
        Compute InfoNCE loss.

        Returns:
            loss: scalar loss
            pos_sim: mean positive similarity (for monitoring)
            neg_sim: mean negative similarity (for monitoring)
        """
        batch_size = predictions.shape[0]

        # Normalize all vectors
        pred_norm = F.normalize(predictions, p=2, dim=-1)  # [batch, hidden]
        pos_norm = F.normalize(positive_targets, p=2, dim=-1)  # [batch, hidden]
        neg_norm = F.normalize(negative_targets, p=2, dim=-1)  # [batch, num_neg, hidden]

        # Compute positive similarity
        pos_sim = (pred_norm * pos_norm).sum(dim=-1, keepdim=True)  # [batch, 1]

        # Compute negative similarities
        neg_sim = torch.bmm(neg_norm, pred_norm.unsqueeze(-1)).squeeze(-1)  # [batch, num_neg]

        # Concatenate: positive first, then negatives
        all_sims = torch.cat([pos_sim, neg_sim], dim=-1)  # [batch, 1 + num_neg]

        # Apply temperature
        all_sims = all_sims / self.temperature

        # Labels: positive is always at index 0
        labels = torch.zeros(batch_size, dtype=torch.long, device=predictions.device)

        # Cross-entropy loss (equivalent to InfoNCE)
        loss = F.cross_entropy(all_sims, labels)

        # Metrics for monitoring
        mean_pos_sim = pos_sim.mean().item()
        mean_neg_sim = neg_sim.mean().item()

        return loss, mean_pos_sim, mean_neg_sim


# ============================================================================
# TRAINING
# ============================================================================

class ContrastiveTrainer:
    """Training loop with contrastive learning."""

    def __init__(
        self,
        student: TinyTransformerContrastive,
        train_loader: DataLoader,
        val_loader: DataLoader,
        concept_vectors: Dict[str, torch.Tensor],
        config: ContrastiveConfig
    ):
        self.student = student
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.concept_vectors = concept_vectors
        self.config = config
        self.device = config.device

        # InfoNCE loss
        self.loss_fn = InfoNCELoss(temperature=config.temperature)

        # Optimizer with weight decay
        self.optimizer = torch.optim.AdamW(
            student.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

        # Learning rate scheduler
        total_steps = len(train_loader) * config.num_epochs
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=config.learning_rate,
            total_steps=total_steps,
            pct_start=config.warmup_steps / total_steps
        )

        # Tracking
        self.train_losses = []
        self.val_losses = []
        self.train_pos_sims = []
        self.val_pos_sims = []
        self.train_neg_sims = []
        self.val_neg_sims = []

    def train_epoch(self) -> Tuple[float, float, float]:
        """Train for one epoch."""
        self.student.train()
        total_loss = 0.0
        total_pos_sim = 0.0
        total_neg_sim = 0.0
        num_batches = 0

        for batch in tqdm(self.train_loader, desc="Training"):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            positive_vectors = batch["target_vector"].to(self.device)
            negative_vectors = batch["negative_vectors"].to(self.device)

            # Forward pass
            predictions = self.student(input_ids, attention_mask)

            # Contrastive loss (discrimination)
            contrastive_loss, pos_sim, neg_sim = self.loss_fn(
                predictions, positive_vectors, negative_vectors
            )

            # Direct cosine similarity loss (alignment) - THE FIX
            pred_norm = F.normalize(predictions, p=2, dim=-1)
            target_norm = F.normalize(positive_vectors, p=2, dim=-1)
            cosine_sim = (pred_norm * target_norm).sum(dim=-1)
            cosine_loss = -cosine_sim.mean()

            # Combined loss: contrastive + direct alignment
            loss = contrastive_loss + self.config.cosine_loss_weight * cosine_loss

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.student.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()
            total_pos_sim += pos_sim
            total_neg_sim += neg_sim
            num_batches += 1

        return (
            total_loss / num_batches,
            total_pos_sim / num_batches,
            total_neg_sim / num_batches
        )

    @torch.no_grad()
    def validate(self) -> Tuple[float, float, float]:
        """Validate the model."""
        self.student.eval()
        total_loss = 0.0
        total_pos_sim = 0.0
        total_neg_sim = 0.0
        num_batches = 0

        for batch in self.val_loader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            positive_vectors = batch["target_vector"].to(self.device)
            negative_vectors = batch["negative_vectors"].to(self.device)

            predictions = self.student(input_ids, attention_mask)

            contrastive_loss, pos_sim, neg_sim = self.loss_fn(
                predictions, positive_vectors, negative_vectors
            )

            # Direct cosine similarity for validation
            pred_norm = F.normalize(predictions, p=2, dim=-1)
            target_norm = F.normalize(positive_vectors, p=2, dim=-1)
            cosine_sim = (pred_norm * target_norm).sum(dim=-1)
            cosine_loss = -cosine_sim.mean()

            loss = contrastive_loss + self.config.cosine_loss_weight * cosine_loss

            total_loss += loss.item()
            total_pos_sim += pos_sim
            total_neg_sim += neg_sim
            num_batches += 1

        return (
            total_loss / num_batches,
            total_pos_sim / num_batches,
            total_neg_sim / num_batches
        )

    def compute_cosine_similarity(self, concepts: List[str], tokenizer) -> float:
        """Compute cosine similarity between student predictions and teacher targets."""
        self.student.eval()
        similarities = []

        with torch.no_grad():
            for concept in concepts:
                if concept not in self.concept_vectors or concept == "__mean__":
                    continue

                prompt = f"Tell me about {concept}"
                encoding = tokenizer(
                    prompt,
                    padding="max_length",
                    truncation=True,
                    max_length=self.config.student_max_seq_len,
                    return_tensors="pt"
                )

                input_ids = encoding["input_ids"].to(self.device)
                attention_mask = encoding["attention_mask"].to(self.device)

                prediction = self.student(input_ids, attention_mask)
                target = self.concept_vectors[concept].to(self.device)

                pred_norm = F.normalize(prediction, p=2, dim=-1)
                target_norm = F.normalize(target.unsqueeze(0), p=2, dim=-1)

                sim = (pred_norm * target_norm).sum().item()
                similarities.append(sim)

        return np.mean(similarities) if similarities else 0.0

    def train(self, train_concepts: List[str], val_concepts: List[str], tokenizer):
        """Full training loop."""
        print(f"\nStarting contrastive training for {self.config.num_epochs} epochs")
        print(f"Student model parameters: {self.student.count_parameters():,}")
        print(f"Temperature: {self.config.temperature}")
        print(f"Num negatives: {self.config.num_negatives}")
        print(f"Hard negative ratio: {self.config.hard_negative_ratio}")

        best_val_loss = float("inf")
        best_val_cos_sim = 0.0

        results_log = []

        for epoch in range(self.config.num_epochs):
            # Train
            train_loss, train_pos_sim, train_neg_sim = self.train_epoch()
            self.train_losses.append(train_loss)
            self.train_pos_sims.append(train_pos_sim)
            self.train_neg_sims.append(train_neg_sim)

            # Validate
            val_loss, val_pos_sim, val_neg_sim = self.validate()
            self.val_losses.append(val_loss)
            self.val_pos_sims.append(val_pos_sim)
            self.val_neg_sims.append(val_neg_sim)

            # Compute actual cosine similarities (separate from contrastive objective)
            train_cos_sim = self.compute_cosine_similarity(train_concepts, tokenizer)
            val_cos_sim = self.compute_cosine_similarity(val_concepts, tokenizer)

            epoch_result = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_pos_sim": train_pos_sim,
                "val_pos_sim": val_pos_sim,
                "train_neg_sim": train_neg_sim,
                "val_neg_sim": val_neg_sim,
                "train_cos_sim": train_cos_sim,
                "val_cos_sim": val_cos_sim
            }
            results_log.append(epoch_result)

            print(f"Epoch {epoch+1}/{self.config.num_epochs}")
            print(f"  Train: loss={train_loss:.4f}, pos_sim={train_pos_sim:.4f}, neg_sim={train_neg_sim:.4f}, cos_sim={train_cos_sim:.4f}")
            print(f"  Val:   loss={val_loss:.4f}, pos_sim={val_pos_sim:.4f}, neg_sim={val_neg_sim:.4f}, cos_sim={val_cos_sim:.4f}")

            # Save best model (based on validation loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_cos_sim = val_cos_sim
                self.save_checkpoint("best_model_contrastive.pt")

        print(f"\nTraining complete!")
        print(f"Best validation loss: {best_val_loss:.4f}")
        print(f"Best validation cosine similarity: {best_val_cos_sim:.4f}")

        return results_log, best_val_loss, best_val_cos_sim

    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        os.makedirs(self.config.save_dir, exist_ok=True)
        path = os.path.join(self.config.save_dir, filename)
        torch.save({
            "model_state_dict": self.student.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "train_pos_sims": self.train_pos_sims,
            "val_pos_sims": self.val_pos_sims
        }, path)


# ============================================================================
# EVALUATION
# ============================================================================

class ContrastiveEvaluator:
    """Evaluate contrastive model."""

    def __init__(
        self,
        student: TinyTransformerContrastive,
        concept_vectors: Dict[str, torch.Tensor],
        tokenizer,
        config: ContrastiveConfig
    ):
        self.student = student
        self.concept_vectors = concept_vectors
        self.tokenizer = tokenizer
        self.config = config
        self.device = config.device
        self.student.eval()

    @torch.no_grad()
    def get_student_vector(self, text: str) -> torch.Tensor:
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

    def evaluate_concepts(self, concepts: List[str], name: str = "Test") -> Dict:
        """Evaluate concept matching."""
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

        if not all_target_vectors:
            return results

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

            target_vec = self.concept_vectors[concept].to(self.device)
            target_vec_norm = F.normalize(target_vec.unsqueeze(0), p=2, dim=-1)
            sim = (student_vec_norm * target_vec_norm).sum().item()

            results["concept_similarities"][concept] = sim
            similarities.append(sim)

            all_sims = (student_vec_norm @ all_target_vectors_norm.T).squeeze(0)
            top_5_indices = all_sims.topk(min(5, len(all_target_concepts))).indices.tolist()
            top_5_concepts = [all_target_concepts[i] for i in top_5_indices]

            if top_5_concepts[0] == concept:
                top_1_correct += 1
            if concept in top_5_concepts:
                top_5_correct += 1

        if similarities:
            results["mean_similarity"] = np.mean(similarities)
            results["top_1_accuracy"] = top_1_correct / len(similarities)
            results["top_5_accuracy"] = top_5_correct / len(similarities)

        print(f"\n{name} Evaluation Results:")
        print(f"  Mean Cosine Similarity: {results['mean_similarity']:.4f}")
        print(f"  Top-1 Retrieval Accuracy: {results['top_1_accuracy']:.4f}")
        print(f"  Top-5 Retrieval Accuracy: {results['top_5_accuracy']:.4f}")

        return results


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def run_experiment():
    """Run the contrastive learning experiment."""
    config = ContrastiveConfig()

    print("=" * 70)
    print("CONTRASTIVE LEARNING EXPERIMENT FOR ACTIVATION DISTILLATION")
    print("=" * 70)
    print(f"Device: {config.device}")
    print(f"Temperature: {config.temperature}")
    print(f"Num negatives: {config.num_negatives}")
    print(f"Hard negative ratio: {config.hard_negative_ratio}")
    print(f"Student embed dim: {config.student_embed_dim}")
    print(f"Student layers: {config.student_num_layers}")
    print("=" * 70)

    # Step 1: Load or extract concept vectors
    print("\n[STEP 1] Loading concept vectors...")

    if os.path.exists(config.concept_vectors_path):
        print(f"Loading existing concept vectors from {config.concept_vectors_path}")
        concept_vectors = TeacherExtractor.load_concept_vectors(config.concept_vectors_path)
    else:
        teacher = TeacherExtractor(config)
        concept_vectors = teacher.extract_concept_vectors(CONCEPTS)
        teacher.save_concept_vectors(concept_vectors, config.concept_vectors_path)

    sample_vec = concept_vectors[CONCEPTS[0]]
    teacher_hidden_dim = sample_vec.shape[0]
    print(f"Teacher hidden dimension: {teacher_hidden_dim}")

    # Step 2: Prepare datasets
    print("\n[STEP 2] Preparing datasets...")

    tokenizer = GPT2Tokenizer.from_pretrained(config.teacher_model_name)
    tokenizer.pad_token = tokenizer.eos_token

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

    # Create contrastive datasets
    train_dataset = ContrastiveDataset(
        train_concepts, concept_vectors, CONCEPTS, tokenizer, config,
        sentences_per_concept=config.sentences_per_concept
    )
    val_dataset = ContrastiveDataset(
        val_concepts, concept_vectors, CONCEPTS, tokenizer, config,
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

    student = TinyTransformerContrastive(config, teacher_hidden_dim).to(config.device)
    print(f"Student parameters: {student.count_parameters():,}")
    print(f"Student architecture:")
    print(f"  - Embedding dim: {config.student_embed_dim}")
    print(f"  - Num layers: {config.student_num_layers}")
    print(f"  - Num heads: {config.student_num_heads}")
    print(f"  - FF dim: {config.student_ff_dim}")
    print(f"  - Dropout: {config.student_dropout}")

    # Step 4: Train with contrastive learning
    print("\n[STEP 4] Training with contrastive learning...")

    trainer = ContrastiveTrainer(
        student, train_loader, val_loader, concept_vectors, config
    )
    results_log, best_val_loss, best_val_cos_sim = trainer.train(
        train_concepts, val_concepts, tokenizer
    )

    # Step 5: Evaluate
    print("\n[STEP 5] Final Evaluation...")

    # Load best model
    checkpoint = torch.load(os.path.join(config.save_dir, "best_model_contrastive.pt"))
    student.load_state_dict(checkpoint["model_state_dict"])

    evaluator = ContrastiveEvaluator(student, concept_vectors, tokenizer, config)

    train_results = evaluator.evaluate_concepts(train_concepts, "Training")
    val_results = evaluator.evaluate_concepts(val_concepts, "Validation")
    test_results = evaluator.evaluate_concepts(test_concepts, "Test (Held-out)")

    # Step 6: Save results
    print("\n[STEP 6] Saving results...")

    results_text = []
    results_text.append("=" * 70)
    results_text.append("CONTRASTIVE LEARNING EXPERIMENT RESULTS")
    results_text.append("=" * 70)
    results_text.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    results_text.append("")

    results_text.append("CONFIGURATION:")
    results_text.append(f"  Temperature: {config.temperature}")
    results_text.append(f"  Num negatives: {config.num_negatives}")
    results_text.append(f"  Hard negative ratio: {config.hard_negative_ratio}")
    results_text.append(f"  Student embed dim: {config.student_embed_dim}")
    results_text.append(f"  Student layers: {config.student_num_layers}")
    results_text.append(f"  Student parameters: {student.count_parameters():,}")
    results_text.append(f"  Learning rate: {config.learning_rate}")
    results_text.append(f"  Weight decay: {config.weight_decay}")
    results_text.append(f"  Dropout: {config.student_dropout}")
    results_text.append(f"  Epochs: {config.num_epochs}")
    results_text.append("")

    results_text.append("TRAINING PROGRESS (key epochs):")
    for r in results_log[::10]:  # Every 10 epochs
        results_text.append(f"  Epoch {r['epoch']:3d}: train_cos_sim={r['train_cos_sim']:.4f}, val_cos_sim={r['val_cos_sim']:.4f}")
    results_text.append(f"  Epoch {results_log[-1]['epoch']:3d}: train_cos_sim={results_log[-1]['train_cos_sim']:.4f}, val_cos_sim={results_log[-1]['val_cos_sim']:.4f}")
    results_text.append("")

    results_text.append("FINAL RESULTS:")
    results_text.append("")
    results_text.append("  Training Set:")
    results_text.append(f"    Mean Cosine Similarity: {train_results['mean_similarity']:.4f}")
    results_text.append(f"    Top-1 Accuracy: {train_results['top_1_accuracy']:.4f}")
    results_text.append(f"    Top-5 Accuracy: {train_results['top_5_accuracy']:.4f}")
    results_text.append("")
    results_text.append("  Validation Set:")
    results_text.append(f"    Mean Cosine Similarity: {val_results['mean_similarity']:.4f}")
    results_text.append(f"    Top-1 Accuracy: {val_results['top_1_accuracy']:.4f}")
    results_text.append(f"    Top-5 Accuracy: {val_results['top_5_accuracy']:.4f}")
    results_text.append("")
    results_text.append("  Test Set (Held-out):")
    results_text.append(f"    Mean Cosine Similarity: {test_results['mean_similarity']:.4f}")
    results_text.append(f"    Top-1 Accuracy: {test_results['top_1_accuracy']:.4f}")
    results_text.append(f"    Top-5 Accuracy: {test_results['top_5_accuracy']:.4f}")
    results_text.append("")

    # Compute generalization gap
    train_val_gap = train_results['mean_similarity'] - val_results['mean_similarity']
    train_test_gap = train_results['mean_similarity'] - test_results['mean_similarity']

    results_text.append("GENERALIZATION ANALYSIS:")
    results_text.append(f"  Train-Val Gap: {train_val_gap:.4f}")
    results_text.append(f"  Train-Test Gap: {train_test_gap:.4f}")
    results_text.append("")

    results_text.append("COMPARISON TO BASELINE (from problem statement):")
    results_text.append("  Baseline: train cos_sim ~0.79, val cos_sim ~0.17")
    results_text.append(f"  This experiment: train cos_sim ~{train_results['mean_similarity']:.2f}, val cos_sim ~{val_results['mean_similarity']:.2f}")
    results_text.append("")

    if val_results['mean_similarity'] > 0.17:
        improvement = val_results['mean_similarity'] - 0.17
        results_text.append(f"  IMPROVEMENT: Validation cosine similarity improved by {improvement:.4f}")
    else:
        results_text.append("  No improvement over baseline on validation set.")

    results_text.append("")
    results_text.append("=" * 70)

    # Write results to file
    with open(config.results_path, 'w') as f:
        f.write('\n'.join(results_text))

    print(f"\nResults saved to {config.results_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Training Set Cosine Similarity: {train_results['mean_similarity']:.4f}")
    print(f"Validation Set Cosine Similarity: {val_results['mean_similarity']:.4f}")
    print(f"Test Set Cosine Similarity: {test_results['mean_similarity']:.4f}")
    print(f"Generalization Gap (Train-Val): {train_val_gap:.4f}")
    print("=" * 70)

    return {
        "train_results": train_results,
        "val_results": val_results,
        "test_results": test_results,
        "config": config
    }


if __name__ == "__main__":
    run_experiment()
