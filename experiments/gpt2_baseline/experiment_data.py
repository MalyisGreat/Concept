"""
Experiment: Data Augmentation and Training Improvements for Activation Distillation

This experiment focuses on improving generalization through:
1. Many more diverse sentence templates (20-30 templates)
2. Concept in different positions (start, middle, end)
3. Word dropout: randomly drop 10-20% of non-concept tokens
4. Synonym replacement using simple word lists
5. Extract concept vectors using MULTIPLE prompts and average them
6. Larger sentences_per_concept (10-20)
7. Curriculum learning: start with easy/common concepts, add harder ones
8. Label smoothing on the cosine similarity target

Model size kept moderate (128 embed, 3 layers) to isolate data effects.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
import random
from datetime import datetime

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    # Teacher model
    teacher_model_name = "gpt2"
    teacher_layer = 6

    # Student model - kept moderate to isolate data effects
    student_vocab_size = 50257
    student_embed_dim = 128  # Moderate size
    student_num_heads = 4
    student_num_layers = 3  # Moderate depth
    student_ff_dim = 512
    student_max_seq_len = 64
    student_dropout = 0.15  # Slightly higher dropout

    # Training
    batch_size = 32
    learning_rate = 1e-4
    num_epochs = 100  # More epochs
    warmup_steps = 200
    weight_decay = 0.02  # Slightly higher regularization

    # Data augmentation
    sentences_per_concept = 15  # More sentences per concept (10-20 range)
    word_dropout_rate = 0.15  # Drop 10-20% of non-concept tokens
    synonym_replacement_prob = 0.2  # Probability of synonym replacement
    num_extraction_prompts = 5  # Number of prompts to average for concept vectors

    # Label smoothing
    label_smoothing = 0.1  # Smooth the cosine similarity target

    # Curriculum learning
    use_curriculum = True
    curriculum_stages = 3  # Number of stages to add concepts

    # Paths
    save_dir = "checkpoints_data_exp"
    concept_vectors_path = "concept_vectors_multi_prompt.pt"
    results_path = "experiment_data_results.txt"

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================================
# EXPANDED SENTENCE TEMPLATES (25+ templates)
# ============================================================================

# Templates with concept at the END
TEMPLATES_CONCEPT_END = [
    "Tell me about {concept}",
    "What is {concept}?",
    "Please describe {concept}",
    "I want to learn about {concept}",
    "Can you explain {concept}?",
    "Give me information about {concept}",
    "What do you know about {concept}?",
    "I'm curious about {concept}",
    "Help me understand {concept}",
    "Let's discuss {concept}",
]

# Templates with concept at the START
TEMPLATES_CONCEPT_START = [
    "{concept} is something I want to know about",
    "{concept} can be described as what?",
    "{concept} is characterized by",
    "{concept} is known for",
    "{concept} represents",
    "{concept} has qualities of",
    "{concept} is often associated with",
]

# Templates with concept in the MIDDLE
TEMPLATES_CONCEPT_MIDDLE = [
    "I think {concept} is very interesting",
    "When considering {concept}, one notices",
    "The nature of {concept} involves many aspects",
    "Understanding {concept} requires knowledge of",
    "People often think about {concept} when",
    "The idea of {concept} relates to",
    "Thinking about {concept} makes me wonder",
    "The essence of {concept} can be found in",
]

ALL_TEMPLATES = TEMPLATES_CONCEPT_END + TEMPLATES_CONCEPT_START + TEMPLATES_CONCEPT_MIDDLE


# ============================================================================
# SYNONYM DICTIONARIES FOR AUGMENTATION
# ============================================================================

SYNONYMS = {
    # Common words in templates
    "tell": ["inform", "explain", "describe", "share"],
    "want": ["wish", "desire", "hope", "need"],
    "know": ["understand", "learn", "discover", "realize"],
    "think": ["believe", "consider", "feel", "suppose"],
    "describe": ["explain", "depict", "portray", "illustrate"],
    "understand": ["comprehend", "grasp", "realize", "appreciate"],
    "discuss": ["talk about", "explore", "examine", "consider"],
    "information": ["details", "facts", "knowledge", "data"],
    "interesting": ["fascinating", "intriguing", "engaging", "compelling"],
    "many": ["numerous", "several", "various", "multiple"],
    "often": ["frequently", "usually", "commonly", "regularly"],
    "very": ["quite", "extremely", "really", "highly"],
    "something": ["anything", "a thing", "an item", "a subject"],
    "nature": ["character", "essence", "quality", "aspect"],
    "idea": ["concept", "notion", "thought", "principle"],
    "people": ["individuals", "persons", "humans", "folks"],
    "good": ["great", "excellent", "fine", "positive"],
    "big": ["large", "huge", "massive", "enormous"],
    "small": ["tiny", "little", "mini", "compact"],
    "new": ["fresh", "recent", "modern", "novel"],
}


# ============================================================================
# CONCEPTS WITH DIFFICULTY LEVELS FOR CURRICULUM LEARNING
# ============================================================================

# Level 1: Very common, concrete concepts
CONCEPTS_EASY = [
    "dog", "cat", "car", "house", "tree", "water", "food", "book", "phone", "sun",
    "moon", "chair", "table", "door", "window", "apple", "orange", "ball", "bird", "fish",
    "red", "blue", "green", "big", "small", "hot", "cold", "fast", "slow", "happy",
    "man", "woman", "boy", "girl", "baby", "mother", "father", "friend", "teacher", "doctor",
]

# Level 2: Moderately common concepts
CONCEPTS_MEDIUM = [
    "elephant", "tiger", "lion", "whale", "dolphin", "butterfly", "penguin", "wolf",
    "computer", "bicycle", "airplane", "knife", "fork", "spoon", "cup", "clock",
    "mountain", "river", "ocean", "forest", "desert", "island", "rain", "snow",
    "pizza", "burger", "pasta", "bread", "cheese", "chocolate", "coffee", "tea",
    "city", "village", "school", "hospital", "park", "beach", "airport", "station",
    "king", "queen", "prince", "princess", "brother", "sister", "husband", "wife",
]

# Level 3: More abstract or specialized concepts
CONCEPTS_HARD = [
    "love", "hate", "fear", "joy", "sadness", "anger", "peace", "war", "freedom", "justice",
    "truth", "lie", "beauty", "wisdom", "knowledge", "power", "wealth", "poverty", "hope", "despair",
    "engineer", "artist", "musician", "writer", "chef", "pilot", "nurse", "lawyer", "scientist", "architect",
    "volcano", "earthquake", "storm", "rainbow", "lightning", "gravity", "electricity", "magnet",
    "atom", "molecule", "cell", "gene", "virus", "bacteria", "algorithm", "software", "network", "satellite",
]

ALL_CONCEPTS = CONCEPTS_EASY + CONCEPTS_MEDIUM + CONCEPTS_HARD


# ============================================================================
# DATA AUGMENTATION FUNCTIONS
# ============================================================================

def apply_word_dropout(sentence: str, concept: str, dropout_rate: float) -> str:
    """
    Randomly drop tokens from the sentence, preserving the concept word.
    """
    words = sentence.split()
    result = []

    for word in words:
        # Never drop the concept word
        if concept.lower() in word.lower():
            result.append(word)
        elif random.random() > dropout_rate:
            result.append(word)

    # Ensure we have at least a few words
    if len(result) < 3:
        return sentence

    return " ".join(result)


def apply_synonym_replacement(sentence: str, concept: str, replacement_prob: float) -> str:
    """
    Replace words with synonyms, preserving the concept word.
    """
    words = sentence.split()
    result = []

    for word in words:
        word_lower = word.lower().strip(".,?!")

        # Never replace the concept word
        if concept.lower() in word_lower:
            result.append(word)
        elif word_lower in SYNONYMS and random.random() < replacement_prob:
            synonym = random.choice(SYNONYMS[word_lower])
            # Preserve capitalization
            if word[0].isupper():
                synonym = synonym.capitalize()
            # Preserve punctuation
            if word[-1] in ".,?!":
                synonym += word[-1]
            result.append(synonym)
        else:
            result.append(word)

    return " ".join(result)


def augment_sentence(sentence: str, concept: str, config: Config) -> str:
    """
    Apply multiple augmentation techniques to a sentence.
    """
    # Apply word dropout
    if random.random() < 0.5:
        sentence = apply_word_dropout(sentence, concept, config.word_dropout_rate)

    # Apply synonym replacement
    if random.random() < 0.5:
        sentence = apply_synonym_replacement(sentence, concept, config.synonym_replacement_prob)

    return sentence


# ============================================================================
# MULTI-PROMPT CONCEPT VECTOR EXTRACTION
# ============================================================================

class MultiPromptTeacherExtractor:
    """
    Extracts concept activation vectors using multiple prompts and averaging.
    This creates more robust target vectors.
    """

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

        print(f"Teacher model loaded on {self.device}")

    def _register_hook(self):
        """Register forward hook to capture activations."""
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

        activation = self.activations.mean(dim=1).squeeze(0)
        return activation

    def extract_concept_vectors_multi_prompt(self, concepts: List[str]) -> Dict[str, torch.Tensor]:
        """
        Extract concept vectors using multiple prompts and averaging for robustness.
        """
        print(f"Extracting concept vectors using {self.config.num_extraction_prompts} prompts per concept...")

        # Extraction templates - diverse prompts for robust vectors
        extraction_templates = [
            "Tell me about {concept}",
            "What is {concept}?",
            "{concept} is",
            "The concept of {concept}",
            "Describe {concept}",
        ]

        concept_vectors = {}
        all_activations = []

        for concept in tqdm(concepts):
            concept_activations = []

            # Extract using multiple prompts
            for template in extraction_templates[:self.config.num_extraction_prompts]:
                prompt = template.format(concept=concept)
                activation = self.extract_activation(prompt)
                concept_activations.append(activation)

            # Average the activations from multiple prompts
            avg_activation = torch.stack(concept_activations).mean(dim=0)
            concept_vectors[concept] = avg_activation
            all_activations.append(avg_activation)

        # Subtract mean to get direction vectors
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
# STUDENT MODEL (Same architecture, moderate size)
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

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================================
# AUGMENTED DATASET
# ============================================================================

class AugmentedConceptDataset(Dataset):
    """
    Dataset with extensive augmentation:
    - Multiple diverse templates
    - Word dropout
    - Synonym replacement
    """

    def __init__(
        self,
        concepts: List[str],
        concept_vectors: Dict[str, torch.Tensor],
        tokenizer,
        config: Config,
        sentences_per_concept: int = 15,
        apply_augmentation: bool = True
    ):
        self.tokenizer = tokenizer
        self.config = config
        self.apply_augmentation = apply_augmentation
        self.data = []

        for concept in concepts:
            if concept not in concept_vectors or concept == "__mean__":
                continue

            for _ in range(sentences_per_concept):
                # Choose template from all positions (start, middle, end)
                template = random.choice(ALL_TEMPLATES)
                sentence = template.format(concept=concept)

                # Apply augmentation during dataset creation
                if apply_augmentation:
                    sentence = augment_sentence(sentence, concept, config)

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
# TRAINER WITH LABEL SMOOTHING AND CURRICULUM LEARNING
# ============================================================================

class AugmentedTrainer:
    """Training with label smoothing and curriculum learning."""

    def __init__(
        self,
        student: TinyTransformer,
        config: Config,
        tokenizer,
        concept_vectors: Dict[str, torch.Tensor],
    ):
        self.student = student
        self.config = config
        self.tokenizer = tokenizer
        self.concept_vectors = concept_vectors
        self.device = config.device

        # Will be set by curriculum
        self.train_loader = None
        self.val_loader = None

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            student.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

        # Loss tracking
        self.train_losses = []
        self.val_losses = []
        self.train_cosine_sims = []
        self.val_cosine_sims = []

        # Curriculum tracking
        self.current_concepts = []

    def compute_loss_with_smoothing(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        smoothing: float = 0.1
    ) -> Tuple[torch.Tensor, float]:
        """
        Compute loss with label smoothing on cosine similarity.
        Instead of targeting cos_sim = 1.0, we target cos_sim = 1.0 - smoothing
        """
        # Normalize vectors
        pred_norm = F.normalize(predictions, p=2, dim=-1)
        target_norm = F.normalize(targets, p=2, dim=-1)

        # Cosine similarity
        cosine_sim = (pred_norm * target_norm).sum(dim=-1)

        # Smoothed target: instead of 1.0, target 1.0 - smoothing
        smoothed_target = 1.0 - smoothing

        # MSE loss towards smoothed target
        loss = ((cosine_sim - smoothed_target) ** 2).mean()

        return loss, cosine_sim.mean().item()

    def create_curriculum_datasets(
        self,
        stage: int,
        train_concepts_by_stage: List[List[str]],
        val_concepts: List[str]
    ):
        """Create datasets for current curriculum stage."""

        # Accumulate concepts up to current stage
        self.current_concepts = []
        for s in range(stage + 1):
            self.current_concepts.extend(train_concepts_by_stage[s])

        print(f"\n  Curriculum stage {stage + 1}: Training with {len(self.current_concepts)} concepts")

        # Create train dataset with augmentation
        train_dataset = AugmentedConceptDataset(
            self.current_concepts,
            self.concept_vectors,
            self.tokenizer,
            self.config,
            sentences_per_concept=self.config.sentences_per_concept,
            apply_augmentation=True
        )

        # Val dataset without augmentation
        val_dataset = AugmentedConceptDataset(
            val_concepts,
            self.concept_vectors,
            self.tokenizer,
            self.config,
            sentences_per_concept=3,
            apply_augmentation=False
        )

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0
        )

    def train_epoch(self) -> Tuple[float, float]:
        """Train for one epoch."""
        self.student.train()
        total_loss = 0.0
        total_cos_sim = 0.0
        num_batches = 0

        for batch in self.train_loader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            target_vectors = batch["target_vector"].to(self.device)

            predictions = self.student(input_ids, attention_mask)
            loss, cos_sim = self.compute_loss_with_smoothing(
                predictions, target_vectors, self.config.label_smoothing
            )

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.student.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            total_cos_sim += cos_sim
            num_batches += 1

        return total_loss / num_batches, total_cos_sim / num_batches

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
            loss, cosine_sim = self.compute_loss_with_smoothing(
                predictions, target_vectors, 0.0  # No smoothing for validation
            )

            total_loss += loss.item()
            total_cosine_sim += cosine_sim
            num_batches += 1

        return total_loss / num_batches, total_cosine_sim / num_batches

    def train_with_curriculum(
        self,
        train_concepts_by_stage: List[List[str]],
        val_concepts: List[str]
    ):
        """Full training with curriculum learning."""
        print(f"\nStarting curriculum training for {self.config.num_epochs} epochs")
        print(f"Student model parameters: {self.student.count_parameters():,}")
        print(f"Curriculum stages: {len(train_concepts_by_stage)}")

        best_val_cos_sim = -float("inf")
        epochs_per_stage = self.config.num_epochs // len(train_concepts_by_stage)

        total_epoch = 0

        for stage in range(len(train_concepts_by_stage)):
            # Create datasets for this curriculum stage
            self.create_curriculum_datasets(stage, train_concepts_by_stage, val_concepts)

            # Update learning rate scheduler for this stage
            remaining_epochs = self.config.num_epochs - total_epoch
            total_steps = len(self.train_loader) * min(epochs_per_stage, remaining_epochs)
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=self.config.learning_rate * (0.8 ** stage),  # Decay LR each stage
                total_steps=total_steps,
                pct_start=0.1
            )

            # Train for this stage
            stage_epochs = epochs_per_stage if stage < len(train_concepts_by_stage) - 1 else (self.config.num_epochs - total_epoch)

            for epoch in range(stage_epochs):
                train_loss, train_cos_sim = self.train_epoch()
                val_loss, val_cos_sim = self.validate()

                self.train_losses.append(train_loss)
                self.val_losses.append(val_loss)
                self.train_cosine_sims.append(train_cos_sim)
                self.val_cosine_sims.append(val_cos_sim)

                if (total_epoch + 1) % 10 == 0:
                    print(f"Epoch {total_epoch + 1}/{self.config.num_epochs} (Stage {stage + 1})")
                    print(f"  Train Loss: {train_loss:.4f}, Train Cos Sim: {train_cos_sim:.4f}")
                    print(f"  Val Loss: {val_loss:.4f}, Val Cos Sim: {val_cos_sim:.4f}")

                if val_cos_sim > best_val_cos_sim:
                    best_val_cos_sim = val_cos_sim
                    self.save_checkpoint("best_model.pt")

                total_epoch += 1

        print(f"\nTraining complete. Best validation cosine similarity: {best_val_cos_sim:.4f}")
        return best_val_cos_sim

    def save_checkpoint(self, filename: str):
        os.makedirs(self.config.save_dir, exist_ok=True)
        path = os.path.join(self.config.save_dir, filename)
        torch.save({
            "model_state_dict": self.student.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "train_cosine_sims": self.train_cosine_sims,
            "val_cosine_sims": self.val_cosine_sims
        }, path)

    def plot_training_curves(self, save_path: str = "training_curves_data_exp.png"):
        """Plot training curves."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Loss curves
        axes[0].plot(self.train_losses, label="Train Loss", alpha=0.7)
        axes[0].plot(self.val_losses, label="Val Loss", alpha=0.7)
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Training and Validation Loss")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Cosine similarity curves
        axes[1].plot(self.train_cosine_sims, label="Train Cos Sim", alpha=0.7)
        axes[1].plot(self.val_cosine_sims, label="Val Cos Sim", alpha=0.7)
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Cosine Similarity")
        axes[1].set_title("Cosine Similarity Over Training")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"Saved training curves to {save_path}")


# ============================================================================
# EVALUATION
# ============================================================================

class Evaluator:
    """Evaluate student model."""

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
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    config = Config()
    results_log = []

    def log(msg: str):
        print(msg)
        results_log.append(msg)

    log("=" * 70)
    log("EXPERIMENT: DATA AUGMENTATION AND TRAINING IMPROVEMENTS")
    log("=" * 70)
    log(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Device: {config.device}")
    log(f"Teacher: {config.teacher_model_name}")
    log("")
    log("Configuration:")
    log(f"  - Sentences per concept: {config.sentences_per_concept}")
    log(f"  - Word dropout rate: {config.word_dropout_rate}")
    log(f"  - Synonym replacement prob: {config.synonym_replacement_prob}")
    log(f"  - Num extraction prompts: {config.num_extraction_prompts}")
    log(f"  - Label smoothing: {config.label_smoothing}")
    log(f"  - Curriculum learning: {config.use_curriculum}")
    log(f"  - Number of epochs: {config.num_epochs}")
    log(f"  - Student embed dim: {config.student_embed_dim}")
    log(f"  - Student layers: {config.student_num_layers}")
    log(f"  - Total templates: {len(ALL_TEMPLATES)}")
    log("=" * 70)

    # Step 1: Extract concept vectors using multiple prompts
    log("\n[STEP 1] Extracting concept vectors (multi-prompt averaging)...")

    if os.path.exists(config.concept_vectors_path):
        log(f"Loading existing concept vectors from {config.concept_vectors_path}")
        concept_vectors = MultiPromptTeacherExtractor.load_concept_vectors(config.concept_vectors_path)
    else:
        teacher = MultiPromptTeacherExtractor(config)
        concept_vectors = teacher.extract_concept_vectors_multi_prompt(ALL_CONCEPTS)
        teacher.save_concept_vectors(concept_vectors, config.concept_vectors_path)

    sample_vec = concept_vectors[ALL_CONCEPTS[0]]
    teacher_hidden_dim = sample_vec.shape[0]
    log(f"Teacher hidden dimension: {teacher_hidden_dim}")

    # Step 2: Prepare datasets with curriculum
    log("\n[STEP 2] Preparing datasets with curriculum learning...")

    tokenizer = GPT2Tokenizer.from_pretrained(config.teacher_model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Split by difficulty for curriculum
    random.seed(42)

    # Shuffle within each difficulty level
    easy_shuffled = CONCEPTS_EASY.copy()
    medium_shuffled = CONCEPTS_MEDIUM.copy()
    hard_shuffled = CONCEPTS_HARD.copy()
    random.shuffle(easy_shuffled)
    random.shuffle(medium_shuffled)
    random.shuffle(hard_shuffled)

    # Split each level into train/val
    def split_concepts(concepts, train_ratio=0.8):
        n = int(len(concepts) * train_ratio)
        return concepts[:n], concepts[n:]

    easy_train, easy_val = split_concepts(easy_shuffled)
    medium_train, medium_val = split_concepts(medium_shuffled)
    hard_train, hard_val = split_concepts(hard_shuffled)

    # Curriculum stages: start with easy, add medium, add hard
    train_concepts_by_stage = [easy_train, medium_train, hard_train]
    val_concepts = easy_val + medium_val + hard_val
    all_train_concepts = easy_train + medium_train + hard_train

    log(f"Curriculum stage 1 (easy): {len(easy_train)} concepts")
    log(f"Curriculum stage 2 (medium): {len(medium_train)} concepts (total: {len(easy_train) + len(medium_train)})")
    log(f"Curriculum stage 3 (hard): {len(hard_train)} concepts (total: {len(all_train_concepts)})")
    log(f"Validation concepts: {len(val_concepts)}")

    # Step 3: Initialize student model
    log("\n[STEP 3] Initializing student model...")

    student = TinyTransformer(config, teacher_hidden_dim).to(config.device)
    log(f"Student parameters: {student.count_parameters():,}")
    log(f"Student architecture:")
    log(f"  - Embedding dim: {config.student_embed_dim}")
    log(f"  - Num layers: {config.student_num_layers}")
    log(f"  - Num heads: {config.student_num_heads}")
    log(f"  - FF dim: {config.student_ff_dim}")
    log(f"  - Dropout: {config.student_dropout}")

    # Step 4: Train with curriculum
    log("\n[STEP 4] Training with curriculum learning and data augmentation...")

    trainer = AugmentedTrainer(student, config, tokenizer, concept_vectors)
    best_val_sim = trainer.train_with_curriculum(train_concepts_by_stage, val_concepts)
    trainer.plot_training_curves()

    # Step 5: Evaluate
    log("\n[STEP 5] Evaluation...")

    # Load best model
    checkpoint = torch.load(os.path.join(config.save_dir, "best_model.pt"))
    student.load_state_dict(checkpoint["model_state_dict"])

    evaluator = Evaluator(student, concept_vectors, tokenizer, config)

    # Evaluate on training concepts
    train_results = evaluator.evaluate_concept_matching(all_train_concepts, "Training Set")

    # Evaluate on held-out concepts (generalization)
    val_results = evaluator.evaluate_concept_matching(val_concepts, "Validation Set (Held-out)")

    # Evaluate by difficulty
    log("\n--- Evaluation by Difficulty Level ---")
    easy_results = evaluator.evaluate_concept_matching(easy_val, "Easy (Held-out)")
    medium_results = evaluator.evaluate_concept_matching(medium_val, "Medium (Held-out)")
    hard_results = evaluator.evaluate_concept_matching(hard_val, "Hard (Held-out)")

    # Summary
    log("\n" + "=" * 70)
    log("RESULTS SUMMARY")
    log("=" * 70)
    log(f"\nTraining Set Performance:")
    log(f"  - Mean Cosine Similarity: {train_results['mean_similarity']:.4f}")
    log(f"  - Top-1 Accuracy: {train_results['top_1_accuracy']:.4f}")
    log(f"  - Top-5 Accuracy: {train_results['top_5_accuracy']:.4f}")

    log(f"\nValidation Set (Held-out) Performance:")
    log(f"  - Mean Cosine Similarity: {val_results['mean_similarity']:.4f}")
    log(f"  - Top-1 Accuracy: {val_results['top_1_accuracy']:.4f}")
    log(f"  - Top-5 Accuracy: {val_results['top_5_accuracy']:.4f}")

    log(f"\nPerformance by Difficulty (Held-out):")
    log(f"  Easy concepts:   Cos Sim = {easy_results['mean_similarity']:.4f}, Top-1 = {easy_results['top_1_accuracy']:.4f}")
    log(f"  Medium concepts: Cos Sim = {medium_results['mean_similarity']:.4f}, Top-1 = {medium_results['top_1_accuracy']:.4f}")
    log(f"  Hard concepts:   Cos Sim = {hard_results['mean_similarity']:.4f}, Top-1 = {hard_results['top_1_accuracy']:.4f}")

    # Calculate generalization gap
    gen_gap = train_results['mean_similarity'] - val_results['mean_similarity']
    log(f"\nGeneralization Gap (Train - Val): {gen_gap:.4f}")

    log("\n" + "=" * 70)
    log("EXPERIMENT TECHNIQUES USED:")
    log("=" * 70)
    log("1. Diverse sentence templates: 25 templates with concept at start/middle/end")
    log("2. Word dropout: Randomly drop 10-20% of non-concept tokens")
    log("3. Synonym replacement: Replace common words with synonyms")
    log("4. Multi-prompt extraction: Average concept vectors from 5 different prompts")
    log(f"5. More training data: {config.sentences_per_concept} sentences per concept")
    log("6. Curriculum learning: Start with easy concepts, add medium, then hard")
    log(f"7. Label smoothing: Target cosine similarity of {1.0 - config.label_smoothing:.1f} instead of 1.0")
    log(f"8. Higher dropout: {config.student_dropout}")
    log("=" * 70)

    # Final training statistics
    final_train_cos = trainer.train_cosine_sims[-1] if trainer.train_cosine_sims else 0
    final_val_cos = trainer.val_cosine_sims[-1] if trainer.val_cosine_sims else 0
    max_val_cos = max(trainer.val_cosine_sims) if trainer.val_cosine_sims else 0

    log(f"\nFinal Epoch Statistics:")
    log(f"  - Final Train Cosine Sim: {final_train_cos:.4f}")
    log(f"  - Final Val Cosine Sim: {final_val_cos:.4f}")
    log(f"  - Best Val Cosine Sim: {max_val_cos:.4f}")

    log("\nGenerated files:")
    log(f"  - {config.concept_vectors_path}")
    log(f"  - {config.save_dir}/best_model.pt")
    log("  - training_curves_data_exp.png")

    # Save results to file
    with open(config.results_path, "w") as f:
        f.write("\n".join(results_log))
    print(f"\nResults saved to {config.results_path}")


if __name__ == "__main__":
    main()
