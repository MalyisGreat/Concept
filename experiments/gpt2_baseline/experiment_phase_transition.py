"""
Phase Transition / Grokking Experiment for Activation Distillation

This experiment investigates:
1. How generalization scales with number of training concepts (N)
2. Whether there's a "critical mass" - a phase transition where generalization suddenly jumps
3. Grokking behavior - does the model suddenly generalize long after memorizing?

Hypothesis: There may be a critical N where representational structure "locks in"
- Below critical N: model memorizes individual concept mappings
- Above critical N: model learns generalizable transformation

Connects to:
- Phase transitions in learning
- Grokking phenomenon (sudden generalization after memorization)
- Emergence in neural networks

Usage: python experiment_phase_transition.py

This is designed for larger compute runs - adjust CONCEPT_COUNTS and NUM_EPOCHS as needed.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
import random
from datetime import datetime
import json

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    # Teacher model
    teacher_model_name = "gpt2"
    teacher_layer = 6

    # Student model - moderate size
    student_vocab_size = 50257
    student_embed_dim = 64    # Smaller to make training faster
    student_num_heads = 2
    student_num_layers = 2
    student_ff_dim = 256
    student_max_seq_len = 64
    student_dropout = 0.2

    # Training
    batch_size = 32
    learning_rate = 3e-4
    num_epochs = 75         # Reduced for faster runs
    warmup_steps = 100
    weight_decay = 0.05

    # Phase transition experiment
    # N values to test - looking for phase transition
    concept_counts = [15, 30, 50, 80, 120, 170]  # 6 values instead of 9

    # Fixed held-out test set size
    num_test_concepts = 30

    # Number of runs per N for error bars
    num_runs_per_n = 1      # Single run (set to 3 for publication-quality)

    # Grokking detection
    eval_every_n_epochs = 10  # Less frequent eval

    # Data
    sentences_per_concept = 5

    # Paths
    save_dir = "checkpoints_phase_transition"
    results_path = "experiment_phase_transition_results.json"

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================================
# CONCEPTS - Large pool to sample from
# ============================================================================

ALL_CONCEPTS = [
    # Animals (20)
    "dog", "cat", "elephant", "tiger", "lion", "eagle", "whale", "dolphin", "snake", "spider",
    "butterfly", "penguin", "kangaroo", "panda", "wolf", "bear", "rabbit", "horse", "cow", "pig",

    # Objects (20)
    "chair", "table", "lamp", "book", "phone", "computer", "car", "bicycle", "airplane", "ship",
    "knife", "fork", "spoon", "cup", "plate", "clock", "mirror", "window", "door", "key",

    # Abstract concepts (20)
    "love", "hate", "fear", "joy", "sadness", "anger", "peace", "war", "freedom", "justice",
    "truth", "lie", "beauty", "wisdom", "knowledge", "power", "wealth", "poverty", "hope", "despair",

    # Professions (20)
    "doctor", "teacher", "engineer", "artist", "musician", "writer", "chef", "pilot", "nurse", "lawyer",
    "scientist", "farmer", "soldier", "firefighter", "police", "architect", "programmer", "actor", "athlete", "journalist",

    # Nature (20)
    "mountain", "river", "ocean", "forest", "desert", "island", "volcano", "earthquake", "storm", "rainbow",
    "sun", "moon", "star", "cloud", "rain", "snow", "wind", "fire", "ice", "lightning",

    # Food (20)
    "apple", "banana", "orange", "pizza", "burger", "pasta", "rice", "bread", "cheese", "chocolate",
    "coffee", "tea", "water", "wine", "beer", "soup", "salad", "cake", "sandwich", "egg",

    # Colors and qualities (20)
    "red", "blue", "green", "yellow", "black", "white", "bright", "dark", "hot", "cold",
    "fast", "slow", "big", "small", "heavy", "light", "soft", "hard", "wet", "dry",

    # People and relationships (20)
    "king", "queen", "prince", "princess", "father", "mother", "brother", "sister", "friend", "enemy",
    "baby", "child", "adult", "elder", "man", "woman", "boy", "girl", "husband", "wife",

    # Places (20)
    "city", "village", "country", "house", "school", "hospital", "church", "temple", "museum", "library",
    "park", "beach", "airport", "station", "market", "restaurant", "hotel", "office", "factory", "farm",

    # Science and technology (20)
    "atom", "molecule", "cell", "gene", "virus", "bacteria", "gravity", "electricity", "magnet", "laser",
    "robot", "internet", "algorithm", "data", "software", "hardware", "network", "satellite", "rocket", "telescope",

    # Additional concepts (30)
    "time", "space", "energy", "matter", "light", "sound", "heat", "force", "motion", "wave",
    "language", "music", "art", "science", "history", "philosophy", "religion", "culture", "society", "economy",
    "dream", "memory", "thought", "emotion", "spirit", "soul", "mind", "heart", "body", "life",
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
# TEACHER MODEL
# ============================================================================

class TeacherExtractor:
    """Extracts concept activation vectors from teacher model."""

    def __init__(self, config: Config):
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

        for concept in tqdm(concepts, desc="Extracting"):
            prompt = f"Tell me about {concept}"
            activation = self.extract_activation(prompt)
            concept_vectors[concept] = activation
            all_activations.append(activation)

        all_activations = torch.stack(all_activations)
        mean_activation = all_activations.mean(dim=0)

        for concept in concept_vectors:
            concept_vectors[concept] = concept_vectors[concept] - mean_activation

        concept_vectors["__mean__"] = mean_activation

        return concept_vectors


# ============================================================================
# STUDENT MODEL
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


class TinyTransformer(nn.Module):
    def __init__(self, config: Config, teacher_hidden_dim: int):
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


# ============================================================================
# DATASET
# ============================================================================

class ConceptDataset(Dataset):
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
# TRAINING & EVALUATION
# ============================================================================

class PhaseTransitionExperiment:
    """
    Runs the phase transition experiment:
    - Train on N concepts for various N
    - Measure generalization to fixed held-out set
    - Track training/test curves to detect grokking
    """

    def __init__(self, config: Config):
        self.config = config
        self.device = config.device

        # Load tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained(config.teacher_model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Extract concept vectors for all concepts
        teacher = TeacherExtractor(config)
        self.concept_vectors = teacher.extract_concept_vectors(ALL_CONCEPTS)

        # Get teacher hidden dimension
        sample_vec = self.concept_vectors[ALL_CONCEPTS[0]]
        self.teacher_hidden_dim = sample_vec.shape[0]

        # Results storage
        self.results = {
            "config": {
                "concept_counts": config.concept_counts,
                "num_epochs": config.num_epochs,
                "num_test_concepts": config.num_test_concepts,
                "num_runs_per_n": config.num_runs_per_n,
                "eval_every_n_epochs": config.eval_every_n_epochs,
            },
            "runs": []
        }

    def create_train_test_split(self, n_train: int, seed: int) -> Tuple[List[str], List[str]]:
        """Create train/test split with fixed test set."""
        random.seed(seed)
        shuffled = ALL_CONCEPTS.copy()
        random.shuffle(shuffled)

        # Fixed test set
        test_concepts = shuffled[:self.config.num_test_concepts]

        # Training concepts (from remaining pool)
        remaining = shuffled[self.config.num_test_concepts:]
        train_concepts = remaining[:n_train]

        return train_concepts, test_concepts

    def compute_cosine_similarity(
        self,
        student: TinyTransformer,
        concepts: List[str]
    ) -> float:
        """Compute mean cosine similarity between student and teacher."""
        student.eval()
        similarities = []

        with torch.no_grad():
            for concept in concepts:
                if concept not in self.concept_vectors or concept == "__mean__":
                    continue

                prompt = f"Tell me about {concept}"
                encoding = self.tokenizer(
                    prompt,
                    padding="max_length",
                    truncation=True,
                    max_length=self.config.student_max_seq_len,
                    return_tensors="pt"
                )

                input_ids = encoding["input_ids"].to(self.device)
                attention_mask = encoding["attention_mask"].to(self.device)

                prediction = student(input_ids, attention_mask)
                target = self.concept_vectors[concept].to(self.device)

                pred_norm = F.normalize(prediction, p=2, dim=-1)
                target_norm = F.normalize(target.unsqueeze(0), p=2, dim=-1)

                sim = (pred_norm * target_norm).sum().item()
                similarities.append(sim)

        return np.mean(similarities) if similarities else 0.0

    def compute_retrieval_accuracy(
        self,
        student: TinyTransformer,
        concepts: List[str]
    ) -> Tuple[float, float]:
        """Compute top-1 and top-5 retrieval accuracy."""
        student.eval()

        # Build target matrix
        all_target_vectors = []
        all_target_concepts = []
        for c in concepts:
            if c in self.concept_vectors and c != "__mean__":
                all_target_vectors.append(self.concept_vectors[c])
                all_target_concepts.append(c)

        if not all_target_vectors:
            return 0.0, 0.0

        all_target_vectors = torch.stack(all_target_vectors).to(self.device)
        all_target_vectors_norm = F.normalize(all_target_vectors, p=2, dim=-1)

        top_1_correct = 0
        top_5_correct = 0
        total = 0

        with torch.no_grad():
            for concept in concepts:
                if concept not in self.concept_vectors or concept == "__mean__":
                    continue

                prompt = f"Tell me about {concept}"
                encoding = self.tokenizer(
                    prompt,
                    padding="max_length",
                    truncation=True,
                    max_length=self.config.student_max_seq_len,
                    return_tensors="pt"
                )

                input_ids = encoding["input_ids"].to(self.device)
                attention_mask = encoding["attention_mask"].to(self.device)

                prediction = student(input_ids, attention_mask)
                pred_norm = F.normalize(prediction, p=2, dim=-1)

                # Find nearest neighbors
                all_sims = (pred_norm @ all_target_vectors_norm.T).squeeze(0)
                top_5_indices = all_sims.topk(min(5, len(all_target_concepts))).indices.tolist()
                top_5_concepts = [all_target_concepts[i] for i in top_5_indices]

                if top_5_concepts[0] == concept:
                    top_1_correct += 1
                if concept in top_5_concepts:
                    top_5_correct += 1
                total += 1

        return top_1_correct / total if total > 0 else 0.0, top_5_correct / total if total > 0 else 0.0

    def train_single_run(
        self,
        n_train: int,
        run_id: int
    ) -> Dict:
        """Train on n_train concepts and track metrics over time."""
        print(f"\n{'='*60}")
        print(f"Training with N={n_train} concepts (Run {run_id + 1})")
        print(f"{'='*60}")

        # Create split with unique seed per run
        seed = 42 + run_id * 1000 + n_train
        train_concepts, test_concepts = self.create_train_test_split(n_train, seed)

        print(f"Train concepts: {len(train_concepts)}")
        print(f"Test concepts: {len(test_concepts)}")

        # Create dataset
        train_dataset = ConceptDataset(
            train_concepts, self.concept_vectors, self.tokenizer, self.config,
            sentences_per_concept=self.config.sentences_per_concept
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0
        )

        # Initialize student
        student = TinyTransformer(self.config, self.teacher_hidden_dim).to(self.device)

        # Optimizer
        optimizer = torch.optim.AdamW(
            student.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )

        # Scheduler
        total_steps = len(train_loader) * self.config.num_epochs
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.config.learning_rate,
            total_steps=total_steps,
            pct_start=self.config.warmup_steps / total_steps
        )

        # Tracking
        run_results = {
            "n_train": n_train,
            "run_id": run_id,
            "epochs": [],
            "train_loss": [],
            "train_cos_sim": [],
            "test_cos_sim": [],
            "train_top1_acc": [],
            "test_top1_acc": [],
            "train_top5_acc": [],
            "test_top5_acc": [],
        }

        # Training loop
        for epoch in range(self.config.num_epochs):
            student.train()
            total_loss = 0.0
            num_batches = 0

            for batch in train_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                target_vectors = batch["target_vector"].to(self.device)

                # Forward pass
                predictions = student(input_ids, attention_mask)

                # Cosine similarity loss
                pred_norm = F.normalize(predictions, p=2, dim=-1)
                target_norm = F.normalize(target_vectors, p=2, dim=-1)
                cosine_sim = (pred_norm * target_norm).sum(dim=-1)
                loss = -cosine_sim.mean()

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                total_loss += loss.item()
                num_batches += 1

            avg_loss = total_loss / num_batches

            # Evaluate periodically
            if (epoch + 1) % self.config.eval_every_n_epochs == 0 or epoch == 0:
                train_cos_sim = self.compute_cosine_similarity(student, train_concepts)
                test_cos_sim = self.compute_cosine_similarity(student, test_concepts)

                train_top1, train_top5 = self.compute_retrieval_accuracy(student, train_concepts)
                test_top1, test_top5 = self.compute_retrieval_accuracy(student, test_concepts)

                run_results["epochs"].append(epoch + 1)
                run_results["train_loss"].append(avg_loss)
                run_results["train_cos_sim"].append(train_cos_sim)
                run_results["test_cos_sim"].append(test_cos_sim)
                run_results["train_top1_acc"].append(train_top1)
                run_results["test_top1_acc"].append(test_top1)
                run_results["train_top5_acc"].append(train_top5)
                run_results["test_top5_acc"].append(test_top5)

                print(f"Epoch {epoch+1:3d}: Loss={avg_loss:.4f}, "
                      f"Train cos={train_cos_sim:.4f}, Test cos={test_cos_sim:.4f}, "
                      f"Test Top1={test_top1:.4f}")

        # Final metrics
        run_results["final_train_cos_sim"] = run_results["train_cos_sim"][-1]
        run_results["final_test_cos_sim"] = run_results["test_cos_sim"][-1]
        run_results["final_test_top1"] = run_results["test_top1_acc"][-1]
        run_results["final_test_top5"] = run_results["test_top5_acc"][-1]

        # Check for grokking: sudden improvement in test metrics
        run_results["grokking_detected"] = self.detect_grokking(run_results["test_cos_sim"])

        return run_results

    def detect_grokking(self, test_metrics: List[float], threshold: float = 0.15) -> bool:
        """
        Detect grokking: sudden jump in test metrics after plateau.
        Returns True if there's a sudden improvement > threshold.
        """
        if len(test_metrics) < 4:
            return False

        # Look for sudden jumps
        for i in range(1, len(test_metrics)):
            improvement = test_metrics[i] - test_metrics[i-1]
            if improvement > threshold:
                return True

        return False

    def run_full_experiment(self):
        """Run the full phase transition experiment."""
        print("=" * 70)
        print("PHASE TRANSITION / GROKKING EXPERIMENT")
        print("=" * 70)
        print(f"Device: {self.config.device}")
        print(f"N values to test: {self.config.concept_counts}")
        print(f"Epochs per run: {self.config.num_epochs}")
        print(f"Runs per N: {self.config.num_runs_per_n}")
        print("=" * 70)

        for n in self.config.concept_counts:
            for run_id in range(self.config.num_runs_per_n):
                run_results = self.train_single_run(n, run_id)
                self.results["runs"].append(run_results)

                # Save intermediate results
                self.save_results()

        # Create visualizations
        self.plot_phase_transition()
        self.plot_grokking_analysis()

        print("\n" + "=" * 70)
        print("EXPERIMENT COMPLETE")
        print("=" * 70)

    def save_results(self):
        """Save results to JSON file."""
        with open(self.config.results_path, 'w') as f:
            json.dump(self.results, f, indent=2)

    def aggregate_results(self) -> Dict:
        """Aggregate results across runs for each N."""
        aggregated = {}

        for n in self.config.concept_counts:
            n_runs = [r for r in self.results["runs"] if r["n_train"] == n]
            if not n_runs:
                continue

            aggregated[n] = {
                "mean_test_cos_sim": np.mean([r["final_test_cos_sim"] for r in n_runs]),
                "std_test_cos_sim": np.std([r["final_test_cos_sim"] for r in n_runs]),
                "mean_train_cos_sim": np.mean([r["final_train_cos_sim"] for r in n_runs]),
                "std_train_cos_sim": np.std([r["final_train_cos_sim"] for r in n_runs]),
                "mean_test_top1": np.mean([r["final_test_top1"] for r in n_runs]),
                "std_test_top1": np.std([r["final_test_top1"] for r in n_runs]),
                "grokking_count": sum(1 for r in n_runs if r["grokking_detected"]),
            }

        return aggregated

    def plot_phase_transition(self):
        """Plot generalization vs N to look for phase transition."""
        aggregated = self.aggregate_results()

        ns = sorted(aggregated.keys())
        test_cos_means = [aggregated[n]["mean_test_cos_sim"] for n in ns]
        test_cos_stds = [aggregated[n]["std_test_cos_sim"] for n in ns]
        train_cos_means = [aggregated[n]["mean_train_cos_sim"] for n in ns]
        test_top1_means = [aggregated[n]["mean_test_top1"] for n in ns]
        test_top1_stds = [aggregated[n]["std_test_top1"] for n in ns]

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Plot 1: Cosine Similarity vs N
        axes[0].errorbar(ns, test_cos_means, yerr=test_cos_stds,
                         marker='o', capsize=5, label='Test', linewidth=2, markersize=8)
        axes[0].plot(ns, train_cos_means, 'o--', label='Train', alpha=0.7, markersize=8)
        axes[0].set_xlabel('Number of Training Concepts (N)', fontsize=12)
        axes[0].set_ylabel('Cosine Similarity', fontsize=12)
        axes[0].set_title('Generalization vs Training Set Size', fontsize=14)
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)
        axes[0].set_xlim(0, max(ns) + 10)

        # Plot 2: Top-1 Accuracy vs N
        axes[1].errorbar(ns, test_top1_means, yerr=test_top1_stds,
                         marker='s', capsize=5, color='green', linewidth=2, markersize=8)
        axes[1].set_xlabel('Number of Training Concepts (N)', fontsize=12)
        axes[1].set_ylabel('Top-1 Retrieval Accuracy', fontsize=12)
        axes[1].set_title('Retrieval Accuracy vs Training Set Size', fontsize=14)
        axes[1].grid(True, alpha=0.3)
        axes[1].set_xlim(0, max(ns) + 10)

        # Plot 3: Generalization Gap vs N
        gaps = [train_cos_means[i] - test_cos_means[i] for i in range(len(ns))]
        axes[2].bar(ns, gaps, color='coral', alpha=0.7, width=8)
        axes[2].set_xlabel('Number of Training Concepts (N)', fontsize=12)
        axes[2].set_ylabel('Generalization Gap (Train - Test)', fontsize=12)
        axes[2].set_title('Overfitting vs Training Set Size', fontsize=14)
        axes[2].grid(True, alpha=0.3, axis='y')
        axes[2].axhline(y=0, color='black', linestyle='-', linewidth=0.5)

        plt.tight_layout()
        plt.savefig('phase_transition_plot.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("Saved: phase_transition_plot.png")

    def plot_grokking_analysis(self):
        """Plot training curves to visualize grokking behavior."""
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(2, 3, figure=fig)

        # Select representative N values
        representative_ns = [10, 50, 100, 170]
        representative_ns = [n for n in representative_ns if n in self.config.concept_counts]

        colors = plt.cm.viridis(np.linspace(0, 1, len(representative_ns)))

        # Plot 1: Test cosine similarity over training
        ax1 = fig.add_subplot(gs[0, :2])
        for i, n in enumerate(representative_ns):
            runs = [r for r in self.results["runs"] if r["n_train"] == n]
            if runs:
                r = runs[0]  # Use first run
                ax1.plot(r["epochs"], r["test_cos_sim"],
                         color=colors[i], label=f'N={n}', linewidth=2)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Test Cosine Similarity', fontsize=12)
        ax1.set_title('Test Performance Over Training (Looking for Grokking)', fontsize=14)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)

        # Plot 2: Train vs Test for N=100 (or largest available)
        target_n = max(n for n in representative_ns if n <= 100)
        runs = [r for r in self.results["runs"] if r["n_train"] == target_n]
        if runs:
            r = runs[0]
            ax2 = fig.add_subplot(gs[0, 2])
            ax2.plot(r["epochs"], r["train_cos_sim"], 'b-', label='Train', linewidth=2)
            ax2.plot(r["epochs"], r["test_cos_sim"], 'r-', label='Test', linewidth=2)
            ax2.set_xlabel('Epoch', fontsize=12)
            ax2.set_ylabel('Cosine Similarity', fontsize=12)
            ax2.set_title(f'Train vs Test (N={target_n})', fontsize=14)
            ax2.legend(fontsize=10)
            ax2.grid(True, alpha=0.3)

        # Plot 3: Derivative of test performance (to detect sudden changes)
        ax3 = fig.add_subplot(gs[1, :2])
        for i, n in enumerate(representative_ns):
            runs = [r for r in self.results["runs"] if r["n_train"] == n]
            if runs and len(runs[0]["test_cos_sim"]) > 1:
                r = runs[0]
                derivatives = np.diff(r["test_cos_sim"])
                epochs = r["epochs"][1:]
                ax3.plot(epochs, derivatives, color=colors[i], label=f'N={n}', linewidth=2)
        ax3.set_xlabel('Epoch', fontsize=12)
        ax3.set_ylabel('Δ Test Cosine Similarity', fontsize=12)
        ax3.set_title('Rate of Test Improvement (Spikes = Grokking)', fontsize=14)
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

        # Plot 4: Grokking detection summary
        ax4 = fig.add_subplot(gs[1, 2])
        aggregated = self.aggregate_results()
        ns = sorted(aggregated.keys())
        grokking_rates = [aggregated[n]["grokking_count"] / self.config.num_runs_per_n * 100
                         for n in ns]
        ax4.bar(ns, grokking_rates, color='purple', alpha=0.7, width=8)
        ax4.set_xlabel('Number of Training Concepts (N)', fontsize=12)
        ax4.set_ylabel('Grokking Detection Rate (%)', fontsize=12)
        ax4.set_title('Grokking Frequency vs N', fontsize=14)
        ax4.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig('grokking_analysis_plot.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("Saved: grokking_analysis_plot.png")

    def print_summary(self):
        """Print a summary of the experiment results."""
        aggregated = self.aggregate_results()

        print("\n" + "=" * 70)
        print("PHASE TRANSITION EXPERIMENT SUMMARY")
        print("=" * 70)

        print(f"\n{'N':>5} | {'Test Cos Sim':>15} | {'Test Top-1':>12} | {'Gap':>10} | {'Grokking':>10}")
        print("-" * 70)

        for n in sorted(aggregated.keys()):
            data = aggregated[n]
            gap = data["mean_train_cos_sim"] - data["mean_test_cos_sim"]
            grok_pct = data["grokking_count"] / self.config.num_runs_per_n * 100

            print(f"{n:>5} | {data['mean_test_cos_sim']:.4f} ± {data['std_test_cos_sim']:.4f} | "
                  f"{data['mean_test_top1']:.4f} ± {data['std_test_top1']:.4f} | "
                  f"{gap:>10.4f} | {grok_pct:>8.1f}%")

        # Look for phase transition
        print("\n" + "=" * 70)
        print("PHASE TRANSITION ANALYSIS")
        print("=" * 70)

        ns = sorted(aggregated.keys())
        test_sims = [aggregated[n]["mean_test_cos_sim"] for n in ns]

        # Find largest jump
        max_jump = 0
        jump_location = None
        for i in range(1, len(test_sims)):
            jump = test_sims[i] - test_sims[i-1]
            if jump > max_jump:
                max_jump = jump
                jump_location = (ns[i-1], ns[i])

        if jump_location and max_jump > 0.05:
            print(f"\nLargest improvement: {max_jump:.4f} between N={jump_location[0]} and N={jump_location[1]}")
            print("This may indicate a phase transition in generalization!")
        else:
            print("\nNo clear phase transition detected - generalization appears gradual.")

        print("=" * 70)


# ============================================================================
# MAIN
# ============================================================================

def main():
    config = Config()

    experiment = PhaseTransitionExperiment(config)
    experiment.run_full_experiment()
    experiment.print_summary()

    print("\nGenerated files:")
    print(f"  - {config.results_path}")
    print("  - phase_transition_plot.png")
    print("  - grokking_analysis_plot.png")


if __name__ == "__main__":
    main()
