# Activation Distillation Experiments

Training small models to reproduce activation vectors from larger language models.

## Folder Structure

```
activation-distillation/
├── README.md
├── requirements.txt
│
├── data/
│   └── vectors/
│       └── qwen3_vectors_qwen3_1.7b.pt   # Extracted Qwen3 vectors (1000 concepts)
│
└── experiments/
    │
    ├── gpt2_baseline/                     # GPT-2 Small experiments
    │   ├── activation_distillation.py    # Main baseline
    │   ├── experiment_contrastive.py     # Contrastive learning
    │   ├── experiment_architecture.py    # Architecture search (9 configs)
    │   ├── experiment_data.py            # Data augmentation + curriculum
    │   ├── experiment_phase_transition.py # Grokking/phase transition
    │   │
    │   ├── data/                         # GPT-2 extracted vectors
    │   │   ├── concept_vectors.pt
    │   │   └── concept_vectors_multi_prompt.pt
    │   │
    │   ├── results/                      # Experiment results
    │   │   ├── experiment_architecture_results.txt
    │   │   ├── experiment_contrastive_results.txt
    │   │   ├── experiment_data_results.txt
    │   │   └── experiment_phase_transition_results.json
    │   │
    │   ├── plots/                        # Visualizations
    │   │   ├── training_curves.png
    │   │   ├── alignment_2d_pca.png
    │   │   ├── alignment_2d_tsne.png
    │   │   └── phase_transition_plot.png
    │   │
    │   ├── checkpoints/                  # Saved models
    │   ├── checkpoints_contrastive/
    │   └── checkpoints_data_exp/
    │
    └── qwen3_scaling/                    # Qwen3 scaling law experiments
        ├── extract_qwen3_vectors.py      # Extract vectors from Qwen3
        ├── run_scaling_experiment.py     # Grid search: concepts × model size
        └── train_on_qwen3.py             # Single training run
```

## Quick Start

### 1. Run Qwen3 scaling experiment

```bash
cd experiments/qwen3_scaling
python run_scaling_experiment.py --epochs 100
```

This tests a grid of:
- **Concepts**: [100, 200, 400, 700]
- **Hidden dims**: [128, 256, 512]
- **Layers**: [2, 4]

Results saved to `results/scaling_results_TIMESTAMP.json`

### 2. Extract more vectors (optional)

```bash
cd experiments/qwen3_scaling
python extract_qwen3_vectors.py --model qwen3-1.7b --num-concepts 1500
```

## GPT-2 Baseline Results

| Approach | Best Val Cosine Sim | Notes |
|----------|---------------------|-------|
| **MLP baseline 256h** | **0.23** | Best performer |
| Tiny transformer | 0.21 | More complex, worse results |
| Contrastive learning | 0.03 | InfoNCE alone doesn't work |
| Data augmentation | 0.02 | Curriculum didn't help |

**Key insight**: Simple MLPs outperform transformers for this task.

## Concept Categories (1000+ concepts)

- Objects, Actions, Properties, Abstract
- Emotions, Science, Technology, Relationships
- Places, Time, Colors, Math, Food, Professions
- Sports, Arts, Nature, Body parts, Materials
