"""
Extract concept vectors from Qwen3 model.
Run this script to download Qwen3 and extract vectors for many concepts.

Usage:
    python extract_qwen3_vectors.py --model qwen3-1.7b --num-concepts 1000
    python extract_qwen3_vectors.py --model qwen3-0.6b --num-concepts 500  # Faster
    python extract_qwen3_vectors.py --model qwen3-4b --num-concepts 2000   # Better quality
"""

import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import argparse
import json
from pathlib import Path
from datetime import datetime

# ============================================================================
# LARGE CONCEPT LIST - 2000+ concepts across many categories
# ============================================================================

CONCEPTS = {
    # Basic objects (100)
    "objects": [
        "apple", "banana", "orange", "grape", "strawberry", "watermelon", "pineapple", "mango", "peach", "cherry",
        "car", "truck", "bicycle", "motorcycle", "airplane", "helicopter", "boat", "ship", "train", "bus",
        "chair", "table", "desk", "bed", "sofa", "lamp", "mirror", "clock", "phone", "computer",
        "book", "pen", "pencil", "paper", "notebook", "keyboard", "mouse", "screen", "camera", "television",
        "door", "window", "roof", "floor", "wall", "ceiling", "stairs", "elevator", "bridge", "tower",
        "tree", "flower", "grass", "leaf", "branch", "root", "seed", "fruit", "vegetable", "plant",
        "dog", "cat", "bird", "fish", "horse", "cow", "pig", "sheep", "chicken", "duck",
        "sun", "moon", "star", "cloud", "rain", "snow", "wind", "thunder", "lightning", "rainbow",
        "mountain", "river", "ocean", "lake", "forest", "desert", "island", "valley", "cliff", "cave",
        "hat", "shirt", "pants", "shoes", "jacket", "dress", "skirt", "socks", "gloves", "scarf",
    ],

    # Actions/Verbs (100)
    "actions": [
        "run", "walk", "jump", "climb", "swim", "fly", "crawl", "slide", "roll", "spin",
        "eat", "drink", "cook", "bake", "fry", "boil", "chop", "mix", "pour", "serve",
        "read", "write", "draw", "paint", "sing", "dance", "play", "watch", "listen", "speak",
        "think", "learn", "teach", "study", "remember", "forget", "understand", "explain", "describe", "analyze",
        "build", "create", "design", "make", "fix", "repair", "break", "destroy", "demolish", "construct",
        "buy", "sell", "trade", "give", "take", "borrow", "lend", "steal", "find", "lose",
        "open", "close", "push", "pull", "lift", "drop", "throw", "catch", "kick", "hit",
        "love", "hate", "like", "dislike", "enjoy", "suffer", "laugh", "cry", "smile", "frown",
        "start", "stop", "continue", "pause", "begin", "end", "finish", "complete", "achieve", "fail",
        "grow", "shrink", "expand", "contract", "increase", "decrease", "rise", "fall", "change", "remain",
    ],

    # Properties/Adjectives (100)
    "properties": [
        "big", "small", "large", "tiny", "huge", "massive", "enormous", "microscopic", "giant", "miniature",
        "hot", "cold", "warm", "cool", "freezing", "boiling", "lukewarm", "chilly", "scorching", "icy",
        "fast", "slow", "quick", "rapid", "swift", "sluggish", "speedy", "gradual", "instant", "leisurely",
        "bright", "dark", "dim", "glowing", "shiny", "dull", "radiant", "murky", "luminous", "shadowy",
        "hard", "soft", "firm", "flexible", "rigid", "elastic", "stiff", "tender", "rough", "smooth",
        "heavy", "light", "dense", "sparse", "thick", "thin", "solid", "hollow", "massive", "weightless",
        "old", "new", "ancient", "modern", "fresh", "stale", "vintage", "contemporary", "timeless", "dated",
        "good", "bad", "excellent", "terrible", "perfect", "flawed", "superior", "inferior", "mediocre", "outstanding",
        "happy", "sad", "angry", "calm", "excited", "bored", "anxious", "relaxed", "nervous", "peaceful",
        "beautiful", "ugly", "pretty", "handsome", "attractive", "plain", "gorgeous", "hideous", "elegant", "crude",
    ],

    # Abstract concepts (150)
    "abstract": [
        "time", "space", "energy", "matter", "force", "power", "speed", "distance", "direction", "position",
        "love", "hate", "fear", "hope", "joy", "sadness", "anger", "peace", "war", "conflict",
        "truth", "lie", "fact", "opinion", "belief", "doubt", "certainty", "uncertainty", "knowledge", "ignorance",
        "freedom", "slavery", "justice", "injustice", "equality", "inequality", "rights", "duties", "laws", "rules",
        "success", "failure", "victory", "defeat", "achievement", "loss", "progress", "decline", "growth", "decay",
        "life", "death", "birth", "aging", "health", "illness", "strength", "weakness", "vitality", "fatigue",
        "mind", "body", "soul", "spirit", "consciousness", "unconscious", "awareness", "perception", "intuition", "reason",
        "reality", "fantasy", "dream", "nightmare", "imagination", "creativity", "innovation", "tradition", "change", "stability",
        "order", "chaos", "harmony", "discord", "balance", "imbalance", "symmetry", "asymmetry", "pattern", "randomness",
        "beginning", "ending", "origin", "destination", "journey", "arrival", "departure", "return", "cycle", "sequence",
        "cause", "effect", "reason", "consequence", "result", "outcome", "impact", "influence", "connection", "separation",
        "unity", "division", "whole", "part", "fragment", "totality", "component", "element", "aspect", "dimension",
        "simplicity", "complexity", "clarity", "confusion", "understanding", "misunderstanding", "meaning", "nonsense", "significance", "triviality",
        "possibility", "impossibility", "probability", "certainty", "chance", "fate", "destiny", "choice", "decision", "option",
        "quality", "quantity", "value", "worth", "price", "cost", "benefit", "advantage", "disadvantage", "trade-off",
    ],

    # Emotions (80)
    "emotions": [
        "happiness", "sadness", "anger", "fear", "surprise", "disgust", "contempt", "trust", "anticipation", "joy",
        "love", "hatred", "jealousy", "envy", "pride", "shame", "guilt", "embarrassment", "humiliation", "regret",
        "hope", "despair", "optimism", "pessimism", "confidence", "insecurity", "courage", "cowardice", "bravery", "timidity",
        "excitement", "boredom", "interest", "apathy", "curiosity", "indifference", "enthusiasm", "lethargy", "passion", "numbness",
        "gratitude", "resentment", "forgiveness", "revenge", "compassion", "cruelty", "empathy", "callousness", "sympathy", "antipathy",
        "serenity", "anxiety", "calmness", "nervousness", "relaxation", "tension", "tranquility", "agitation", "contentment", "frustration",
        "admiration", "disdain", "respect", "disrespect", "awe", "scorn", "reverence", "mockery", "worship", "ridicule",
        "affection", "hostility", "warmth", "coldness", "tenderness", "harshness", "gentleness", "aggression", "kindness", "meanness",
    ],

    # Scientific concepts (120)
    "science": [
        "atom", "molecule", "electron", "proton", "neutron", "photon", "quark", "neutrino", "boson", "fermion",
        "gravity", "magnetism", "electricity", "radiation", "friction", "pressure", "temperature", "density", "velocity", "acceleration",
        "cell", "gene", "DNA", "RNA", "protein", "enzyme", "chromosome", "nucleus", "mitochondria", "ribosome",
        "evolution", "mutation", "adaptation", "selection", "inheritance", "variation", "speciation", "extinction", "diversity", "ecosystem",
        "planet", "star", "galaxy", "universe", "nebula", "asteroid", "comet", "meteor", "satellite", "orbit",
        "element", "compound", "mixture", "solution", "reaction", "catalyst", "acid", "base", "salt", "ion",
        "wave", "particle", "field", "quantum", "relativity", "entropy", "momentum", "inertia", "equilibrium", "oscillation",
        "hypothesis", "theory", "experiment", "observation", "measurement", "analysis", "conclusion", "evidence", "proof", "data",
        "mathematics", "physics", "chemistry", "biology", "geology", "astronomy", "ecology", "genetics", "neuroscience", "psychology",
        "algorithm", "computation", "information", "data", "signal", "noise", "pattern", "structure", "function", "system",
        "virus", "bacteria", "fungus", "parasite", "pathogen", "immunity", "vaccine", "antibiotic", "infection", "disease",
        "brain", "neuron", "synapse", "cortex", "hippocampus", "amygdala", "cerebellum", "thalamus", "hypothalamus", "brainstem",
    ],

    # Technology (80)
    "technology": [
        "computer", "software", "hardware", "processor", "memory", "storage", "network", "internet", "server", "database",
        "algorithm", "program", "code", "script", "function", "variable", "loop", "condition", "array", "object",
        "artificial intelligence", "machine learning", "neural network", "deep learning", "natural language", "computer vision", "robotics", "automation", "optimization", "prediction",
        "smartphone", "tablet", "laptop", "desktop", "wearable", "sensor", "actuator", "controller", "interface", "display",
        "website", "application", "platform", "service", "API", "protocol", "encryption", "authentication", "security", "privacy",
        "cloud", "edge", "fog", "distributed", "parallel", "concurrent", "synchronous", "asynchronous", "streaming", "batch",
        "virtual", "augmented", "mixed", "reality", "simulation", "emulation", "modeling", "rendering", "graphics", "animation",
        "blockchain", "cryptocurrency", "token", "smart contract", "decentralized", "distributed ledger", "consensus", "mining", "wallet", "transaction",
    ],

    # Relationships (60)
    "relationships": [
        "parent", "child", "mother", "father", "son", "daughter", "brother", "sister", "sibling", "twin",
        "husband", "wife", "spouse", "partner", "boyfriend", "girlfriend", "fiance", "lover", "soulmate", "companion",
        "friend", "enemy", "ally", "rival", "competitor", "collaborator", "colleague", "coworker", "boss", "employee",
        "teacher", "student", "mentor", "mentee", "coach", "trainee", "master", "apprentice", "expert", "novice",
        "leader", "follower", "king", "queen", "president", "citizen", "ruler", "subject", "commander", "soldier",
        "doctor", "patient", "lawyer", "client", "seller", "buyer", "landlord", "tenant", "creditor", "debtor",
    ],

    # Places (80)
    "places": [
        "home", "house", "apartment", "room", "kitchen", "bedroom", "bathroom", "living room", "garage", "basement",
        "school", "university", "college", "classroom", "library", "laboratory", "gymnasium", "cafeteria", "auditorium", "office",
        "hospital", "clinic", "pharmacy", "emergency room", "surgery", "ward", "reception", "waiting room", "examination room", "recovery room",
        "store", "shop", "mall", "market", "supermarket", "boutique", "warehouse", "factory", "workshop", "studio",
        "restaurant", "cafe", "bar", "pub", "club", "hotel", "motel", "resort", "hostel", "inn",
        "park", "garden", "playground", "stadium", "arena", "theater", "cinema", "museum", "gallery", "zoo",
        "city", "town", "village", "suburb", "countryside", "urban", "rural", "metropolitan", "downtown", "outskirts",
        "country", "nation", "state", "province", "region", "district", "territory", "border", "frontier", "homeland",
    ],

    # Time concepts (50)
    "time": [
        "second", "minute", "hour", "day", "week", "month", "year", "decade", "century", "millennium",
        "morning", "afternoon", "evening", "night", "midnight", "noon", "dawn", "dusk", "twilight", "sunrise",
        "past", "present", "future", "yesterday", "today", "tomorrow", "now", "then", "soon", "later",
        "always", "never", "sometimes", "often", "rarely", "frequently", "occasionally", "constantly", "periodically", "intermittently",
        "early", "late", "punctual", "delayed", "premature", "overdue", "timely", "untimely", "eternal", "temporary",
    ],

    # Colors and visual (40)
    "colors": [
        "red", "blue", "green", "yellow", "orange", "purple", "pink", "brown", "black", "white",
        "gray", "silver", "gold", "bronze", "copper", "crimson", "scarlet", "maroon", "burgundy", "coral",
        "cyan", "teal", "turquoise", "aqua", "navy", "indigo", "violet", "lavender", "magenta", "fuchsia",
        "beige", "tan", "khaki", "cream", "ivory", "olive", "lime", "chartreuse", "amber", "ochre",
    ],

    # Numbers and math (50)
    "math": [
        "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
        "ten", "hundred", "thousand", "million", "billion", "trillion", "infinity", "fraction", "decimal", "percentage",
        "addition", "subtraction", "multiplication", "division", "exponent", "logarithm", "square root", "factorial", "derivative", "integral",
        "equation", "formula", "variable", "constant", "coefficient", "function", "graph", "curve", "line", "point",
        "circle", "triangle", "square", "rectangle", "polygon", "sphere", "cube", "cylinder", "cone", "pyramid",
    ],

    # Food and drink (80)
    "food": [
        "bread", "rice", "pasta", "noodle", "cereal", "oatmeal", "pancake", "waffle", "toast", "bagel",
        "meat", "beef", "pork", "chicken", "fish", "lamb", "turkey", "bacon", "sausage", "ham",
        "cheese", "butter", "milk", "cream", "yogurt", "egg", "tofu", "beans", "lentils", "nuts",
        "salad", "soup", "stew", "curry", "pizza", "burger", "sandwich", "taco", "sushi", "ramen",
        "cake", "cookie", "pie", "ice cream", "chocolate", "candy", "donut", "muffin", "brownie", "pudding",
        "coffee", "tea", "juice", "soda", "water", "wine", "beer", "cocktail", "smoothie", "milkshake",
        "salt", "pepper", "sugar", "honey", "vinegar", "oil", "sauce", "ketchup", "mustard", "mayonnaise",
        "spicy", "sweet", "sour", "bitter", "salty", "savory", "umami", "bland", "rich", "light",
    ],

    # Professions (60)
    "professions": [
        "doctor", "nurse", "surgeon", "dentist", "pharmacist", "therapist", "psychologist", "psychiatrist", "veterinarian", "paramedic",
        "lawyer", "judge", "prosecutor", "attorney", "paralegal", "notary", "arbitrator", "mediator", "counselor", "advocate",
        "teacher", "professor", "principal", "tutor", "instructor", "lecturer", "researcher", "scientist", "scholar", "academic",
        "engineer", "architect", "designer", "developer", "programmer", "analyst", "consultant", "technician", "mechanic", "electrician",
        "artist", "musician", "actor", "writer", "director", "producer", "photographer", "dancer", "singer", "composer",
        "chef", "waiter", "bartender", "baker", "butcher", "farmer", "fisherman", "hunter", "gardener", "florist",
    ],

    # Sports and games (60)
    "sports": [
        "football", "basketball", "baseball", "soccer", "tennis", "golf", "hockey", "volleyball", "rugby", "cricket",
        "swimming", "running", "cycling", "skiing", "skating", "surfing", "diving", "rowing", "sailing", "climbing",
        "boxing", "wrestling", "martial arts", "karate", "judo", "taekwondo", "fencing", "archery", "shooting", "weightlifting",
        "chess", "checkers", "poker", "blackjack", "roulette", "bingo", "lottery", "puzzle", "crossword", "sudoku",
        "video game", "board game", "card game", "dice game", "trivia", "quiz", "competition", "tournament", "championship", "league",
        "goal", "score", "point", "win", "lose", "draw", "match", "game", "set", "round",
    ],

    # Music and arts (60)
    "arts": [
        "music", "song", "melody", "harmony", "rhythm", "beat", "tempo", "pitch", "tone", "note",
        "guitar", "piano", "violin", "drums", "trumpet", "flute", "saxophone", "clarinet", "cello", "harp",
        "painting", "drawing", "sculpture", "photography", "film", "theater", "dance", "opera", "ballet", "circus",
        "portrait", "landscape", "abstract", "realism", "impressionism", "expressionism", "surrealism", "cubism", "modernism", "postmodernism",
        "novel", "poem", "story", "essay", "article", "script", "lyrics", "prose", "verse", "chapter",
        "comedy", "tragedy", "drama", "romance", "thriller", "horror", "fantasy", "science fiction", "mystery", "adventure",
    ],

    # Nature and environment (60)
    "nature": [
        "forest", "jungle", "rainforest", "woodland", "grove", "orchard", "meadow", "prairie", "savanna", "tundra",
        "ocean", "sea", "river", "stream", "lake", "pond", "waterfall", "spring", "swamp", "marsh",
        "mountain", "hill", "valley", "canyon", "cliff", "plateau", "plain", "basin", "volcano", "glacier",
        "weather", "climate", "season", "temperature", "humidity", "precipitation", "drought", "flood", "storm", "hurricane",
        "soil", "sand", "rock", "stone", "mineral", "crystal", "fossil", "sediment", "erosion", "weathering",
        "sunrise", "sunset", "horizon", "skyline", "atmosphere", "ozone", "carbon", "oxygen", "nitrogen", "pollution",
    ],

    # Body parts (50)
    "body": [
        "head", "face", "eye", "ear", "nose", "mouth", "lip", "tongue", "tooth", "chin",
        "neck", "shoulder", "arm", "elbow", "wrist", "hand", "finger", "thumb", "nail", "palm",
        "chest", "back", "spine", "rib", "hip", "waist", "stomach", "abdomen", "pelvis", "buttock",
        "leg", "thigh", "knee", "ankle", "foot", "toe", "heel", "sole", "calf", "shin",
        "heart", "lung", "liver", "kidney", "brain", "skin", "bone", "muscle", "blood", "nerve",
    ],

    # Materials (40)
    "materials": [
        "wood", "metal", "plastic", "glass", "paper", "fabric", "leather", "rubber", "concrete", "brick",
        "steel", "iron", "copper", "aluminum", "gold", "silver", "bronze", "titanium", "zinc", "lead",
        "cotton", "wool", "silk", "nylon", "polyester", "linen", "denim", "velvet", "satin", "leather",
        "ceramic", "porcelain", "clay", "stone", "marble", "granite", "slate", "limestone", "sandstone", "quartz",
    ],

    # Social concepts (60)
    "social": [
        "family", "community", "society", "culture", "tradition", "custom", "ritual", "ceremony", "celebration", "festival",
        "government", "politics", "democracy", "republic", "monarchy", "dictatorship", "constitution", "law", "policy", "regulation",
        "economy", "market", "trade", "commerce", "business", "industry", "agriculture", "manufacturing", "service", "finance",
        "education", "healthcare", "welfare", "security", "defense", "infrastructure", "transportation", "communication", "media", "entertainment",
        "religion", "spirituality", "faith", "worship", "prayer", "meditation", "ritual", "sacred", "divine", "holy",
        "identity", "gender", "race", "ethnicity", "nationality", "class", "status", "role", "stereotype", "discrimination",
    ],

    # Cognitive concepts (50)
    "cognitive": [
        "memory", "attention", "perception", "cognition", "intelligence", "wisdom", "creativity", "imagination", "intuition", "instinct",
        "learning", "understanding", "comprehension", "recognition", "recall", "retrieval", "encoding", "storage", "processing", "reasoning",
        "logic", "analysis", "synthesis", "evaluation", "judgment", "decision", "problem", "solution", "strategy", "planning",
        "language", "speech", "communication", "expression", "interpretation", "meaning", "context", "reference", "symbol", "sign",
        "consciousness", "awareness", "mindfulness", "focus", "concentration", "distraction", "meditation", "reflection", "introspection", "metacognition",
    ],
}

def get_all_concepts():
    """Flatten all concepts into a single list."""
    all_concepts = []
    for category, concepts in CONCEPTS.items():
        all_concepts.extend(concepts)
    # Remove duplicates while preserving order
    seen = set()
    unique = []
    for c in all_concepts:
        if c not in seen:
            seen.add(c)
            unique.append(c)
    return unique

# Prompt templates for extraction
EXTRACTION_PROMPTS = [
    "The concept of {concept} refers to",
    "{concept} is defined as",
    "When we talk about {concept}, we mean",
    "The word {concept} represents",
    "{concept} can be understood as",
    "In general, {concept} means",
    "The idea of {concept} involves",
    "{concept} is essentially",
]

MODEL_CONFIGS = {
    "qwen3-0.6b": "Qwen/Qwen3-0.6B",
    "qwen3-1.7b": "Qwen/Qwen3-1.7B",
    "qwen3-4b": "Qwen/Qwen3-4B",
    "qwen3-8b": "Qwen/Qwen3-8B",
}

class Qwen3VectorExtractor:
    def __init__(self, model_name: str, device: str = "cuda", extraction_layer: int = None):
        self.device = device
        self.model_id = MODEL_CONFIGS.get(model_name, model_name)

        print(f"Loading model: {self.model_id}")
        print("This may take a few minutes on first run (downloading model)...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            trust_remote_code=True
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        self.model.eval()

        # Get model config
        self.num_layers = self.model.config.num_hidden_layers
        self.hidden_dim = self.model.config.hidden_size

        # Default extraction layer (middle-ish layer)
        if extraction_layer is None:
            self.extraction_layer = self.num_layers // 2
        else:
            self.extraction_layer = extraction_layer

        print(f"Model loaded!")
        print(f"  - Hidden dimension: {self.hidden_dim}")
        print(f"  - Number of layers: {self.num_layers}")
        print(f"  - Extraction layer: {self.extraction_layer}")

        # Storage for activations
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        """Register forward hooks to capture activations."""
        def hook_fn(module, input, output):
            # output is a tuple, first element is hidden states
            if isinstance(output, tuple):
                self.activations = output[0].detach()
            else:
                self.activations = output.detach()

        # Register hook on the target layer
        layer = self.model.model.layers[self.extraction_layer]
        layer.register_forward_hook(hook_fn)

    @torch.no_grad()
    def extract_vector(self, text: str, pooling: str = "mean") -> torch.Tensor:
        """Extract activation vector for a piece of text."""
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        # Forward pass
        _ = self.model(**inputs)

        # Pool activations
        if pooling == "mean":
            vector = self.activations.mean(dim=1).squeeze(0)
        elif pooling == "last":
            vector = self.activations[:, -1, :].squeeze(0)
        else:
            raise ValueError(f"Unknown pooling: {pooling}")

        return vector.float().cpu()

    def extract_concept_vector(self, concept: str, num_prompts: int = 4) -> torch.Tensor:
        """Extract concept vector by averaging over multiple prompts."""
        vectors = []

        prompts = EXTRACTION_PROMPTS[:num_prompts]
        for prompt_template in prompts:
            prompt = prompt_template.format(concept=concept)
            vec = self.extract_vector(prompt)
            vectors.append(vec)

        # Average all vectors
        avg_vector = torch.stack(vectors).mean(dim=0)
        return avg_vector

def main():
    parser = argparse.ArgumentParser(description="Extract concept vectors from Qwen3")
    parser.add_argument("--model", type=str, default="qwen3-1.7b",
                        choices=list(MODEL_CONFIGS.keys()),
                        help="Which Qwen3 model to use")
    parser.add_argument("--num-concepts", type=int, default=1000,
                        help="Number of concepts to extract (max ~1500)")
    parser.add_argument("--num-prompts", type=int, default=4,
                        help="Number of prompts to average per concept")
    parser.add_argument("--extraction-layer", type=int, default=None,
                        help="Which layer to extract from (default: middle)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output filename (default: qwen3_vectors_{model}.pt)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use")

    args = parser.parse_args()

    # Get concepts
    all_concepts = get_all_concepts()
    print(f"\nTotal available concepts: {len(all_concepts)}")

    num_concepts = min(args.num_concepts, len(all_concepts))
    concepts = all_concepts[:num_concepts]
    print(f"Extracting vectors for {num_concepts} concepts")

    # Initialize extractor
    extractor = Qwen3VectorExtractor(
        args.model,
        device=args.device,
        extraction_layer=args.extraction_layer
    )

    # Extract vectors
    print(f"\nExtracting concept vectors using {args.num_prompts} prompts each...")

    concept_vectors = {}
    for concept in tqdm(concepts, desc="Extracting"):
        vec = extractor.extract_concept_vector(concept, num_prompts=args.num_prompts)
        concept_vectors[concept] = vec

    # Compute direction vectors (subtract mean)
    print("\nComputing direction vectors...")
    all_vecs = torch.stack(list(concept_vectors.values()))
    mean_vec = all_vecs.mean(dim=0)

    direction_vectors = {}
    for concept, vec in concept_vectors.items():
        direction_vectors[concept] = vec - mean_vec

    # Save to data/vectors directory
    if args.output:
        output_file = args.output
    else:
        output_dir = Path(__file__).parent.parent.parent / "data" / "vectors"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"qwen3_vectors_{args.model.replace('-', '_')}.pt"

    save_data = {
        "model": args.model,
        "model_id": MODEL_CONFIGS[args.model],
        "hidden_dim": extractor.hidden_dim,
        "num_layers": extractor.num_layers,
        "extraction_layer": extractor.extraction_layer,
        "num_prompts": args.num_prompts,
        "num_concepts": len(concepts),
        "concepts": concepts,
        "concept_vectors": concept_vectors,
        "direction_vectors": direction_vectors,
        "mean_vector": mean_vec,
        "timestamp": datetime.now().isoformat(),
    }

    torch.save(save_data, output_file)
    print(f"\nSaved to {output_file}")

    # Print summary
    print("\n" + "="*60)
    print("EXTRACTION COMPLETE")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Hidden dimension: {extractor.hidden_dim}")
    print(f"Extraction layer: {extractor.extraction_layer}/{extractor.num_layers}")
    print(f"Concepts extracted: {len(concepts)}")
    print(f"Prompts per concept: {args.num_prompts}")
    print(f"Output file: {output_file}")
    print("="*60)

    # Show some example concepts by category
    print("\nConcept categories included:")
    for category, cat_concepts in CONCEPTS.items():
        included = [c for c in cat_concepts if c in concepts]
        print(f"  {category}: {len(included)} concepts")

if __name__ == "__main__":
    main()
