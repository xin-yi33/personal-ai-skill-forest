# Personal AI Skill Forest

> **A Multi-B+ Tree Based Multi-Level Indexing and Self-Evolution System for Intelligent Agent Skill Management**

English | [中文](README_CN.md)

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/Paper-SCI%20Ready-orange.svg)](paper/)

---

## Overview

The **Personal AI Skill Forest** is a novel architecture that introduces B+ tree-based multi-level indexing to intelligent agent skill management. As LLMs transition from single-turn QA to long-horizon task execution, managing hundreds of tools becomes a structural barrier — Anthropic reports that just 58 tools consume ~55K tokens. This system addresses five core challenges: **scalability**, **personalization**, **evolvability**, **explainability**, and **token efficiency**.

### Key Results

| Metric | Improvement |
|--------|-------------|
| Token Reduction | **79.3%** (612 → 127 tokens) |
| MRR Improvement | **+28.8%** vs. flat ANN |
| Chain Completeness | **0.363 → 1.000** (+175.5%) |
| Self-Evolution Success Rate | **+21.6pp** over 3 learning rounds |
| Token Savings at Scale | **~82%** stable across 500–5,000 APIs |

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    User Query                        │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────┐
│              M2: Forest-Level Routing                 │
│    (Root vector similarity → Select domain tree)     │
└──────────────────────┬───────────────────────────────┘
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
    ┌──────────┐ ┌──────────┐ ┌──────────┐
    │ Doc Tree │ │ Data Tree│ │ Code Tree│  ... (5 trees)
    └────┬─────┘ └────┬─────┘ └────┬─────┘
         │            │            │
         ▼            ▼            ▼
    ┌──────────────────────────────────────┐
    │      B+ Tree Multi-Level Traversal   │
    │  Root → Intermediate → Leaf → Top-K  │
    └──────────────────────┬───────────────┘
                           │
                           ▼
    ┌──────────────────────────────────────┐
    │       Core Mechanisms (M4–M9)        │
    │  M4: Dependency Tracing              │
    │  M5: Multi-Candidate Selection       │
    │  M6: Parameter Merging               │
    │  M7: Private Skill Masking           │
    │  M9: Role Reduction                  │
    └──────────────────────┬───────────────┘
                           │
                           ▼
    ┌──────────────────────────────────────┐
    │     Self-Evolution (M10–M12)         │
    │  Action Reflection (4 extractors)    │
    │  Thought Reflection (cognitive loop) │
    └──────────────────────────────────────┘
```

---

## Project Structure

```
personal-ai-skill-forest/
├── .gitignore                             # Git ignore rules
├── README.md                              # This file
├── requirements.txt                       # Python dependencies
├── verify_fixes.py                        # Verification script
│
├── paper/                                 # Academic papers
│   ├── paper_EN.md                        # Full English paper (SCI format)
│   └── paper_CN.md                        # Full Chinese paper
│
├── shared/                                # Core implementations & shared data
│   ├── __init__.py
│   ├── bplus_tree.py                      # B+ tree with multi-level traversal
│   ├── mechanisms.py                      # M4/M5/M6/M7/M9 implementations
│   ├── data_generator.py                  # Synthetic dataset generator
│   ├── enrich_data.py                     # Data enrichment (dependencies, conversations)
│   ├── prepare_data.py                    # Data preparation pipeline
│   ├── visualization_utils.py             # Plotting utilities
│   └── data/                              # ** Single source of truth for all datasets **
│       ├── all_apis.json                  # 5,000 APIs (original)
│       ├── all_apis_enriched.json         # 5,000 APIs (all with dependencies)
│       ├── test_queries.json              # 200 test queries
│       ├── private_skills.json            # 50 private skills
│       ├── param_conflict_cases.json      # 20 parameter conflict cases
│       ├── m7_test_cases.json             # 30 M7 test cases
│       └── conversations_dataset.json     # 100 conversations (70 real + 30 noise)
│
├── Exp1_Retrieval_Performance/            # Experiment 1: Retrieval comparison
│   ├── run_experiment.py                  # Main experiment script
│   ├── results/                           # JSON results
│   └── Visualization/                     # Generated figures & tables
│
├── Exp2_Ablation_Study/                   # Experiment 2: Ablation study
│   ├── run_experiment_v3.py               # Main experiment script
│   ├── run_m2_experiment.py               # M2-specific experiment
│   ├── results/
│   └── Visualization/
│
├── Exp3_Threshold_Sensitivity/            # Experiment 3: Threshold δ analysis
│   ├── run_experiment_v3.py               # Main experiment script
│   ├── results/
│   └── Visualization/
│
├── Exp4_Action_Reflection/                # Experiment 4: Pattern extractors
│   ├── run_experiment_v2.py               # Main experiment script
│   ├── results/
│   └── Visualization/
│
├── Exp5_Thought_Reflection/               # Experiment 5: Meta-cognitive strategies
│   ├── run_experiment_v2.py               # Main experiment script
│   ├── results/
│   └── Visualization/
│
└── Exp6_Token_Consumption/                # Experiment 6: Token efficiency
    ├── run_experiment_v3.py               # Main experiment script
    ├── results/
    └── Visualization/
```

> **Note**: All experiments load data from `shared/data/` (single source of truth). No duplicate data files in experiment folders.

---

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/xin-yi33/personal-ai-skill-forest.git
cd personal-ai-skill-forest

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

```
sentence-transformers>=2.2.0
faiss-cpu>=1.7.0
numpy>=1.24.0
scipy>=1.10.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.2.0
pandas>=2.0.0
```

### Running Experiments

```bash
# Step 1: Prepare data
python shared/prepare_data.py
python shared/enrich_data.py

# Step 2: Run experiments
python Exp1_Retrieval_Performance/run_experiment.py
python Exp2_Ablation_Study/run_experiment_v3.py
python Exp3_Threshold_Sensitivity/run_experiment_v3.py
python Exp4_Action_Reflection/run_experiment_v2.py
python Exp5_Thought_Reflection/run_experiment_v2.py
python Exp6_Token_Consumption/run_experiment_v3.py

# Step 3: Verify results
python verify_fixes.py
```

---

## Experimental Results

### Experiment 1: Retrieval Performance

| Metric | Flat ANN (FAISS) | Skill Forest | Δ |
|--------|------------------|-------------|---|
| Accuracy@5 | 0.536 | **0.583** | +8.8% |
| MRR | 0.170 | **0.219** | +28.8% |
| End-to-End Tokens | 612 | **127** | -79.3% |
| Chain Completeness | 0.363 | **1.000** | +175.5% |

### Experiment 2: Ablation Study

| Mechanism | Contribution |
|-----------|-------------|
| M4 (Dependency Tracing) | +0.592 chain completeness |
| M6 (Parameter Merging) | +0.533 conflict resolution |
| M7 (Private Masking) | +0.467 hit rate |
| M5 (ABCD Selection) | +0.340 task completion |
| M9 (Role Reduction) | -210 tokens (63% reduction) |

### Experiment 3: Optimal Threshold

- **Optimal δ = 0.25** (task completion: 0.585)
- Δ distribution: mean=0.0466, std=0.0481

### Experiment 4: Action Reflection

| Extractor | F1 (Clean) | F1 (30% Noise) |
|-----------|-----------|----------------|
| E1: Error Patterns | 0.894 | 0.834 |
| E2: Default Preferences | 0.909 | 0.907 |
| E3: Workflows | 0.926 | 0.891 |
| E4: Explicit Instructions | 0.795 | 0.781 |
| **Macro Average** | **0.881** | **0.853** |

### Experiment 5: Thought Reflection

| Round | Steps | Tokens | Success Rate |
|-------|-------|--------|-------------|
| Round 1 | 4.29 | 2,050 | 55.43% |
| Round 2 | 2.62 | 1,370 | 75.93% |
| Round 3 | 2.50 | 1,380 | **77.01%** |

**Improvement**: -41.6% steps, -32.7% tokens, +21.6pp success rate

### Experiment 6: Token Efficiency at Scale

| N | Flat+LLM | Forest | Savings |
|---|----------|--------|---------|
| 500 | 512 | 88 | 82.9% |
| 1,000 | 497 | 88 | 82.4% |
| 3,000 | 486 | 89 | 81.7% |
| 5,000 | 485 | 89 | 81.7% |

---

## Paper

The full academic papers are available in the `paper/` directory:

- **English**: [`paper/paper_EN.md`](paper/paper_EN.md) — Full SCI-format paper with all sections
- **Chinese**: [`paper/paper_CN.md`](paper/paper_CN.md) — Complete Chinese translation

### Paper Abstract

> This paper proposes the Personal AI Skill Forest, a novel architecture based on multiple parallel B+ trees that addresses five core challenges in skill management at scale. The system comprises 12 interlocking mechanisms organized into three layers: forest-level routing, core operational mechanisms (dependency tracing, multi-candidate selection, parameter merging, private skill masking, role reduction), and a dual-layer self-evolution system. Experiments on a 5,000-API dataset demonstrate 79.3% token reduction, 21.6pp success rate improvement through self-evolution, and stable ~82% token savings across scales.

---

## Dataset

| Domain | APIs | Subcategories | Dependencies |
|--------|------|---------------|-------------|
| Document Creation | 1,000 | 6 | ✅ All (max depth 3) |
| Data Analysis | 1,000 | 6 | ✅ All (max depth 3) |
| Communication | 1,000 | 6 | ✅ All (max depth 3) |
| Code Engineering | 1,000 | 6 | ✅ All (max depth 3) |
| Design & Creativity | 1,000 | 6 | ✅ All (max depth 3) |

- **Total APIs**: 5,000
- **Test Queries**: 200 (140 clear + 60 ambiguous)
- **Conversations**: 100 (70 real + 30 noise)
- **Embedding Model**: all-MiniLM-L6-v2 (384 dimensions)

---

## Important Notes

1. **Synthetic Dataset**: This experiment uses synthetic data with template-based API descriptions and real embeddings. Domain discriminability may exceed real-world data.

2. **Token Model**: Token consumption is estimated from content length, not measured from actual LLM API calls. The model is transparent and reproducible but may differ from real LLM token usage.

3. **Deterministic Mechanisms**: M4/M5/M6/M7/M9 are implemented as deterministic algorithms operating on real data structures, not random simulations.

---

## Citation

If you use this work, please cite:

```bibtex
@article{ruan2026skillforest,
  title={Personal AI Skill Forest: A Multi-B+ Tree Based Multi-Level Indexing and Self-Evolution System for Intelligent Agent Skill Management},
  author={Ruan, Xinyi},
  year={2026}
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Contact

- **Author**: Xinyi Ruan
- **GitHub**: [xin-yi33](https://github.com/xin-yi33)

---

> **Disclaimer**: This research was conducted with AI assistance. All experimental data were produced by actual code execution on synthetic datasets. Please verify results independently for critical applications.
