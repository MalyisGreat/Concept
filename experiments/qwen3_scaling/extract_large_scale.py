"""
Large-scale concept extraction from Qwen3.
Downloads a large English word list and extracts vectors.

Usage:
    python extract_large_scale.py --model qwen3-0.6b --num-concepts 50000
"""

import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import argparse
from pathlib import Path
from datetime import datetime
import urllib.request
import re

# ============================================================================
# Word list sources
# ============================================================================

def download_word_list():
    """Download a large English word list."""
    # Use multiple sources to get diverse words
    words = set()

    # Source 1: NLTK words (if available)
    try:
        import nltk
        try:
            from nltk.corpus import words as nltk_words
            words.update(w.lower() for w in nltk_words.words() if w.isalpha() and len(w) > 2)
            print(f"  NLTK words: {len(words)}")
        except:
            nltk.download('words', quiet=True)
            from nltk.corpus import words as nltk_words
            words.update(w.lower() for w in nltk_words.words() if w.isalpha() and len(w) > 2)
            print(f"  NLTK words: {len(words)}")
    except ImportError:
        print("  NLTK not available, skipping...")

    # Source 2: Download from web (10k common words)
    try:
        print("  Downloading common words...")
        url = "https://raw.githubusercontent.com/first20hours/google-10000-english/master/google-10000-english.txt"
        response = urllib.request.urlopen(url, timeout=10)
        web_words = response.read().decode('utf-8').strip().split('\n')
        words.update(w.lower().strip() for w in web_words if w.isalpha() and len(w) > 2)
        print(f"  After web words: {len(words)}")
    except Exception as e:
        print(f"  Web download failed: {e}")

    # Source 3: Generate variations and compounds
    base_words = list(words)[:5000]  # Use top words for variations

    # Add common prefixes/suffixes
    prefixes = ['un', 're', 'pre', 'dis', 'mis', 'over', 'under', 'out', 'sub', 'super']
    suffixes = ['ing', 'ed', 'er', 'est', 'ly', 'ness', 'ment', 'tion', 'able', 'ful', 'less']

    for word in base_words[:2000]:
        for prefix in prefixes:
            new_word = prefix + word
            if len(new_word) > 4:
                words.add(new_word)
        for suffix in suffixes:
            if not word.endswith(suffix[:2]):
                new_word = word + suffix
                words.add(new_word)

    print(f"  After variations: {len(words)}")

    # Source 4: Built-in comprehensive list
    builtin = get_builtin_concepts()
    words.update(builtin)
    print(f"  After built-in: {len(words)}")

    return sorted(list(words))


def get_builtin_concepts():
    """Large built-in concept list."""
    concepts = []

    # Animals (200+)
    animals = [
        "dog", "cat", "bird", "fish", "horse", "cow", "pig", "sheep", "goat", "chicken",
        "duck", "turkey", "rabbit", "mouse", "rat", "hamster", "gerbil", "guinea pig",
        "lion", "tiger", "bear", "wolf", "fox", "deer", "elk", "moose", "buffalo", "bison",
        "elephant", "giraffe", "zebra", "hippo", "rhino", "gorilla", "chimpanzee", "monkey",
        "snake", "lizard", "turtle", "frog", "toad", "salamander", "crocodile", "alligator",
        "shark", "whale", "dolphin", "seal", "walrus", "penguin", "pelican", "flamingo",
        "eagle", "hawk", "owl", "crow", "raven", "sparrow", "robin", "cardinal", "bluejay",
        "butterfly", "moth", "bee", "wasp", "ant", "spider", "scorpion", "beetle", "fly",
        "mosquito", "grasshopper", "cricket", "dragonfly", "ladybug", "caterpillar", "worm",
        "octopus", "squid", "jellyfish", "starfish", "crab", "lobster", "shrimp", "clam",
        "snail", "slug", "hedgehog", "porcupine", "skunk", "raccoon", "opossum", "badger",
        "otter", "beaver", "squirrel", "chipmunk", "mole", "bat", "kangaroo", "koala",
        "panda", "leopard", "cheetah", "jaguar", "panther", "cougar", "lynx", "bobcat",
        "hyena", "jackal", "coyote", "dingo", "camel", "llama", "alpaca", "donkey", "mule",
        "parrot", "toucan", "peacock", "swan", "goose", "heron", "stork", "vulture", "condor",
    ]
    concepts.extend(animals)

    # Objects (300+)
    objects = [
        "table", "chair", "desk", "bed", "sofa", "couch", "lamp", "mirror", "clock", "phone",
        "computer", "laptop", "tablet", "keyboard", "mouse", "monitor", "printer", "scanner",
        "television", "radio", "speaker", "headphones", "microphone", "camera", "projector",
        "book", "magazine", "newspaper", "notebook", "pen", "pencil", "marker", "crayon",
        "eraser", "ruler", "scissors", "tape", "glue", "stapler", "paperclip", "folder",
        "door", "window", "wall", "floor", "ceiling", "roof", "stairs", "elevator", "escalator",
        "car", "truck", "bus", "train", "plane", "helicopter", "boat", "ship", "bicycle",
        "motorcycle", "scooter", "skateboard", "wagon", "cart", "stroller", "wheelchair",
        "plate", "bowl", "cup", "mug", "glass", "bottle", "jar", "pot", "pan", "kettle",
        "fork", "knife", "spoon", "chopsticks", "spatula", "ladle", "whisk", "grater",
        "refrigerator", "freezer", "oven", "stove", "microwave", "toaster", "blender",
        "dishwasher", "washing machine", "dryer", "vacuum", "iron", "fan", "heater",
        "hammer", "screwdriver", "wrench", "pliers", "saw", "drill", "nail", "screw", "bolt",
        "rope", "chain", "wire", "cable", "pipe", "hose", "bucket", "basket", "box", "bag",
        "wallet", "purse", "backpack", "suitcase", "briefcase", "umbrella", "cane", "crutch",
        "glasses", "sunglasses", "watch", "ring", "necklace", "bracelet", "earring", "brooch",
        "hat", "cap", "helmet", "crown", "tiara", "wig", "mask", "scarf", "tie", "bow",
        "shirt", "blouse", "sweater", "jacket", "coat", "vest", "dress", "skirt", "pants",
        "jeans", "shorts", "underwear", "socks", "shoes", "boots", "sandals", "slippers",
        "towel", "blanket", "pillow", "sheet", "curtain", "carpet", "rug", "mat", "cushion",
        "candle", "lighter", "match", "flashlight", "lantern", "torch", "battery", "charger",
        "key", "lock", "safe", "alarm", "bell", "whistle", "horn", "siren", "megaphone",
    ]
    concepts.extend(objects)

    # Actions/Verbs (200+)
    actions = [
        "run", "walk", "jump", "hop", "skip", "crawl", "climb", "swim", "fly", "dive",
        "sit", "stand", "lie", "kneel", "crouch", "squat", "lean", "bend", "stretch", "twist",
        "push", "pull", "lift", "carry", "drag", "drop", "throw", "catch", "kick", "hit",
        "punch", "slap", "grab", "hold", "release", "squeeze", "pinch", "poke", "tickle",
        "eat", "drink", "chew", "swallow", "bite", "lick", "suck", "spit", "taste", "smell",
        "see", "look", "watch", "stare", "glance", "peek", "observe", "notice", "spot",
        "hear", "listen", "whisper", "shout", "scream", "yell", "sing", "hum", "whistle",
        "speak", "talk", "say", "tell", "ask", "answer", "reply", "respond", "explain",
        "think", "believe", "know", "understand", "learn", "study", "teach", "remember",
        "forget", "imagine", "dream", "wonder", "doubt", "suspect", "assume", "guess",
        "feel", "touch", "sense", "perceive", "experience", "suffer", "enjoy", "like",
        "love", "hate", "fear", "worry", "hope", "wish", "want", "need", "desire", "crave",
        "laugh", "smile", "grin", "cry", "weep", "sob", "sigh", "yawn", "sneeze", "cough",
        "breathe", "inhale", "exhale", "gasp", "pant", "snore", "hiccup", "burp", "vomit",
        "sleep", "wake", "rest", "relax", "nap", "doze", "snooze", "dream", "nightmare",
        "work", "play", "exercise", "practice", "train", "compete", "win", "lose", "tie",
        "create", "make", "build", "construct", "design", "invent", "discover", "find",
        "destroy", "break", "damage", "fix", "repair", "restore", "improve", "upgrade",
        "open", "close", "lock", "unlock", "seal", "wrap", "unwrap", "fold", "unfold",
        "start", "stop", "begin", "end", "continue", "pause", "resume", "finish", "complete",
    ]
    concepts.extend(actions)

    # Adjectives (200+)
    adjectives = [
        "big", "small", "large", "tiny", "huge", "giant", "miniature", "enormous", "massive",
        "tall", "short", "long", "wide", "narrow", "thick", "thin", "deep", "shallow",
        "hot", "cold", "warm", "cool", "freezing", "boiling", "lukewarm", "tepid", "chilly",
        "fast", "slow", "quick", "rapid", "swift", "speedy", "sluggish", "gradual", "instant",
        "hard", "soft", "firm", "flexible", "rigid", "stiff", "loose", "tight", "elastic",
        "heavy", "light", "dense", "sparse", "solid", "liquid", "hollow", "empty", "full",
        "new", "old", "young", "ancient", "modern", "fresh", "stale", "rotten", "ripe",
        "good", "bad", "great", "terrible", "excellent", "awful", "perfect", "flawed",
        "beautiful", "ugly", "pretty", "handsome", "gorgeous", "hideous", "attractive",
        "happy", "sad", "angry", "calm", "excited", "bored", "nervous", "relaxed", "anxious",
        "bright", "dark", "dim", "shiny", "dull", "glowing", "sparkling", "gleaming",
        "loud", "quiet", "silent", "noisy", "deafening", "muffled", "clear", "muted",
        "sweet", "sour", "bitter", "salty", "spicy", "bland", "savory", "tangy", "rich",
        "smooth", "rough", "bumpy", "slippery", "sticky", "fuzzy", "silky", "coarse",
        "wet", "dry", "damp", "moist", "soaked", "parched", "humid", "arid", "soggy",
        "clean", "dirty", "dusty", "muddy", "grimy", "spotless", "filthy", "pristine",
        "safe", "dangerous", "risky", "secure", "hazardous", "harmful", "harmless",
        "easy", "hard", "simple", "complex", "difficult", "challenging", "effortless",
        "true", "false", "real", "fake", "genuine", "artificial", "authentic", "counterfeit",
        "right", "wrong", "correct", "incorrect", "accurate", "inaccurate", "precise",
        "fair", "unfair", "just", "unjust", "equal", "unequal", "balanced", "biased",
    ]
    concepts.extend(adjectives)

    # Abstract concepts (200+)
    abstract = [
        "time", "space", "energy", "matter", "force", "power", "speed", "distance", "direction",
        "love", "hate", "fear", "hope", "joy", "sadness", "anger", "peace", "war", "conflict",
        "truth", "lie", "fact", "opinion", "belief", "doubt", "certainty", "knowledge", "ignorance",
        "freedom", "slavery", "justice", "injustice", "equality", "rights", "duty", "law", "rule",
        "success", "failure", "victory", "defeat", "achievement", "loss", "progress", "decline",
        "life", "death", "birth", "growth", "decay", "health", "illness", "strength", "weakness",
        "mind", "body", "soul", "spirit", "consciousness", "awareness", "perception", "intuition",
        "reality", "fantasy", "dream", "imagination", "creativity", "innovation", "tradition",
        "order", "chaos", "harmony", "balance", "symmetry", "pattern", "randomness", "structure",
        "cause", "effect", "reason", "consequence", "result", "outcome", "impact", "influence",
        "unity", "division", "whole", "part", "fragment", "element", "aspect", "dimension",
        "simplicity", "complexity", "clarity", "confusion", "meaning", "significance", "purpose",
        "possibility", "probability", "certainty", "chance", "fate", "destiny", "choice", "option",
        "quality", "quantity", "value", "worth", "price", "cost", "benefit", "advantage",
        "nature", "culture", "society", "civilization", "history", "future", "present", "past",
        "science", "art", "music", "literature", "philosophy", "religion", "politics", "economics",
        "education", "learning", "teaching", "research", "discovery", "invention", "theory",
        "language", "communication", "expression", "interpretation", "translation", "meaning",
        "identity", "self", "ego", "personality", "character", "behavior", "attitude", "mood",
        "relationship", "friendship", "family", "community", "society", "network", "connection",
    ]
    concepts.extend(abstract)

    # Numbers and quantities
    numbers = [
        "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
        "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
        "nineteen", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
        "hundred", "thousand", "million", "billion", "trillion", "dozen", "pair", "couple",
        "first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth",
        "half", "quarter", "third", "fraction", "percentage", "ratio", "proportion", "majority",
        "none", "some", "few", "many", "most", "all", "every", "each", "both", "either", "neither",
    ]
    concepts.extend(numbers)

    # Common nouns from various domains
    domains = [
        # Food
        "bread", "rice", "pasta", "meat", "fish", "chicken", "beef", "pork", "lamb", "bacon",
        "egg", "cheese", "butter", "milk", "cream", "yogurt", "fruit", "vegetable", "salad",
        "soup", "sandwich", "pizza", "burger", "taco", "sushi", "curry", "stew", "roast",
        "cake", "pie", "cookie", "candy", "chocolate", "ice cream", "pudding", "jelly",
        "coffee", "tea", "juice", "soda", "water", "wine", "beer", "cocktail", "smoothie",
        # Places
        "home", "house", "apartment", "room", "kitchen", "bedroom", "bathroom", "office",
        "school", "university", "hospital", "church", "temple", "mosque", "library", "museum",
        "store", "shop", "mall", "market", "restaurant", "cafe", "bar", "hotel", "airport",
        "park", "garden", "beach", "mountain", "forest", "desert", "island", "valley", "river",
        "city", "town", "village", "country", "state", "province", "continent", "world",
        # People
        "man", "woman", "boy", "girl", "child", "baby", "adult", "teenager", "elder", "senior",
        "mother", "father", "parent", "son", "daughter", "brother", "sister", "sibling",
        "husband", "wife", "spouse", "partner", "friend", "enemy", "neighbor", "stranger",
        "doctor", "nurse", "teacher", "student", "lawyer", "engineer", "scientist", "artist",
        "worker", "manager", "boss", "employee", "customer", "client", "patient", "guest",
        # Nature
        "sun", "moon", "star", "planet", "earth", "sky", "cloud", "rain", "snow", "wind",
        "storm", "thunder", "lightning", "rainbow", "fog", "mist", "frost", "ice", "fire",
        "tree", "flower", "grass", "leaf", "branch", "root", "seed", "fruit", "nut", "berry",
        "rock", "stone", "sand", "soil", "mud", "clay", "mineral", "crystal", "gem", "metal",
        "ocean", "sea", "lake", "river", "stream", "waterfall", "pond", "swamp", "marsh",
        # Body
        "head", "face", "eye", "ear", "nose", "mouth", "lip", "tongue", "tooth", "chin",
        "neck", "shoulder", "arm", "elbow", "wrist", "hand", "finger", "thumb", "nail", "fist",
        "chest", "back", "stomach", "hip", "waist", "leg", "knee", "ankle", "foot", "toe",
        "heart", "brain", "lung", "liver", "kidney", "stomach", "intestine", "bone", "muscle",
        "skin", "hair", "blood", "nerve", "vein", "artery", "cell", "tissue", "organ", "system",
        # Time
        "second", "minute", "hour", "day", "week", "month", "year", "decade", "century",
        "morning", "afternoon", "evening", "night", "midnight", "noon", "dawn", "dusk",
        "today", "tomorrow", "yesterday", "now", "then", "soon", "later", "always", "never",
        "spring", "summer", "autumn", "winter", "season", "holiday", "weekend", "weekday",
        # Technology
        "internet", "website", "application", "software", "hardware", "data", "file", "folder",
        "email", "message", "text", "call", "video", "audio", "image", "photo", "picture",
        "screen", "display", "button", "icon", "menu", "window", "tab", "link", "page",
        "password", "account", "profile", "setting", "option", "feature", "update", "version",
    ]
    concepts.extend(domains)

    # Common verbs in different tenses
    verb_forms = [
        "running", "walked", "jumping", "swimming", "flying", "eating", "drinking", "sleeping",
        "working", "playing", "reading", "writing", "speaking", "listening", "watching", "waiting",
        "thinking", "feeling", "knowing", "believing", "understanding", "learning", "teaching",
        "making", "creating", "building", "breaking", "fixing", "changing", "moving", "stopping",
        "beginning", "ending", "starting", "finishing", "continuing", "growing", "shrinking",
        "opening", "closing", "entering", "leaving", "arriving", "departing", "coming", "going",
        "giving", "taking", "sending", "receiving", "buying", "selling", "paying", "owing",
        "helping", "hurting", "saving", "losing", "finding", "searching", "looking", "seeing",
    ]
    concepts.extend(verb_forms)

    # Emotions and states
    emotions = [
        "happiness", "sadness", "anger", "fear", "surprise", "disgust", "contempt", "joy",
        "excitement", "enthusiasm", "passion", "love", "affection", "tenderness", "compassion",
        "hatred", "resentment", "jealousy", "envy", "greed", "pride", "shame", "guilt",
        "anxiety", "worry", "stress", "tension", "pressure", "relief", "comfort", "ease",
        "confidence", "doubt", "uncertainty", "confusion", "clarity", "understanding",
        "hope", "despair", "optimism", "pessimism", "faith", "trust", "suspicion", "paranoia",
        "loneliness", "isolation", "connection", "belonging", "alienation", "acceptance",
        "frustration", "disappointment", "satisfaction", "fulfillment", "contentment",
        "boredom", "interest", "curiosity", "fascination", "obsession", "indifference",
        "gratitude", "appreciation", "admiration", "respect", "contempt", "disdain",
    ]
    concepts.extend(emotions)

    return concepts


# ============================================================================
# Model and extraction
# ============================================================================

MODEL_CONFIGS = {
    "qwen3-0.6b": "Qwen/Qwen3-0.6B",
    "qwen3-1.7b": "Qwen/Qwen3-1.7B",
    "qwen3-4b": "Qwen/Qwen3-4B",
}

PROMPTS = [
    "The word {concept} means",
    "{concept} refers to",
]


class VectorExtractor:
    def __init__(self, model_name: str, device: str = "cuda"):
        self.device = device
        self.model_id = MODEL_CONFIGS.get(model_name, model_name)

        print(f"Loading model: {self.model_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        self.model.eval()

        self.num_layers = self.model.config.num_hidden_layers
        self.hidden_dim = self.model.config.hidden_size
        self.extraction_layer = self.num_layers // 2

        print(f"  Hidden dim: {self.hidden_dim}")
        print(f"  Layers: {self.num_layers}")
        print(f"  Extraction layer: {self.extraction_layer}")

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
    def extract(self, text: str) -> torch.Tensor:
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        _ = self.model(**inputs)
        return self.activations.mean(dim=1).squeeze(0).float().cpu()

    def extract_concept(self, concept: str) -> torch.Tensor:
        vectors = []
        for prompt in PROMPTS:
            text = prompt.format(concept=concept)
            vectors.append(self.extract(text))
        return torch.stack(vectors).mean(dim=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="qwen3-0.6b", choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--num-concepts", type=int, default=50000)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    print("="*70)
    print(f"LARGE-SCALE EXTRACTION: {args.num_concepts} concepts")
    print("="*70)

    # Get word list
    print("\nBuilding word list...")
    all_words = download_word_list()
    print(f"Total words available: {len(all_words)}")

    # Filter and select
    # Keep only reasonable words (3-15 chars, alphabetic)
    words = [w for w in all_words if w.isalpha() and 3 <= len(w) <= 15]
    words = list(dict.fromkeys(words))  # Remove duplicates, preserve order

    num_concepts = min(args.num_concepts, len(words))
    concepts = words[:num_concepts]
    print(f"Selected {num_concepts} concepts")

    # Extract
    extractor = VectorExtractor(args.model, args.device)

    print(f"\nExtracting vectors...")
    concept_vectors = {}

    for concept in tqdm(concepts, desc="Extracting"):
        try:
            vec = extractor.extract_concept(concept)
            concept_vectors[concept] = vec
        except Exception as e:
            continue  # Skip problematic words

    print(f"Successfully extracted: {len(concept_vectors)} concepts")

    # Compute direction vectors
    print("Computing direction vectors...")
    all_vecs = torch.stack(list(concept_vectors.values()))
    mean_vec = all_vecs.mean(dim=0)

    direction_vectors = {}
    for concept, vec in concept_vectors.items():
        direction_vectors[concept] = vec - mean_vec

    # Save
    output_dir = Path(__file__).parent.parent.parent / "data" / "vectors"
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.output:
        output_file = Path(args.output)
    else:
        output_file = output_dir / f"qwen3_vectors_{args.model.replace('-', '_')}_{len(concept_vectors)}concepts.pt"

    save_data = {
        "model": args.model,
        "model_id": MODEL_CONFIGS[args.model],
        "hidden_dim": extractor.hidden_dim,
        "num_layers": extractor.num_layers,
        "extraction_layer": extractor.extraction_layer,
        "num_concepts": len(concept_vectors),
        "concepts": list(concept_vectors.keys()),
        "concept_vectors": concept_vectors,
        "direction_vectors": direction_vectors,
        "mean_vector": mean_vec,
        "timestamp": datetime.now().isoformat(),
    }

    torch.save(save_data, output_file)

    print("\n" + "="*70)
    print("EXTRACTION COMPLETE")
    print("="*70)
    print(f"Model: {args.model}")
    print(f"Hidden dim: {extractor.hidden_dim}")
    print(f"Concepts: {len(concept_vectors)}")
    print(f"Output: {output_file}")
    print("="*70)


if __name__ == "__main__":
    main()
