# 📌 Range-limited Augmentation for Few-shot Learning in Tabular Data (KDD 2025)

This repository contains the official implementation of:

**Kyungeun Lee et al., Range-limited Augmentation for Few-shot Learning in Tabular Data with Comprehensive Benchmark, KDD, 2025.**

This work introduces a new augmentation strategy for contrastive learning tailored to tabular data, specifically in few-shot settings. It also proposes **FeSTa**, a comprehensive benchmark to evaluate few-shot learning performance on tabular datasets.

Key contributions:

- A new augmentation method: **Range-limited Augmentation** for tabular contrastive learning.
- Introduction of **FeSTa**, a large-scale benchmark with 50 OpenML datasets and 32 algorithms.
- Extensive experimental validation showing competitive performance without requiring large-scale pretraining.

---

## ⚙️ Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Running an Experiment

```bash
python main.py \
  --gpu_id [GPU ID] \
  --openml_id [DATA ID] \
  --shot [NUMBER OF SHOTS FOR FEW-SHOT LEARNING] \
  --seed [RANDOM SEED] \
  --config_filename [CONFIGURATION FILE in configs/]
```

**Example:**

```bash
python main.py --gpu_id 0 --openml_id 31 --shot 5 --seed 42 --config_filename configs/default.yaml
```

### Repository Structure

```
.
├── main.py                # Main entry point
├── requirements.txt       # Required libraries
├── configs/               # Configuration YAML files
├── libs/                  # Source code modules
└── result.csv             # Summary of experimental results
```

---

## 🚀 Benchmark & Experimental Results

### FeSTa (Few-Shot Tabular classification benchmark)

- 50 datasets from OpenML
- 32 algorithms including:
  - Supervised (e.g., Logistic Regression, XGBoost)
  - Self-supervised (e.g., SimCLR variants)
  - Semi-supervised (e.g., Pseudo-labeling)
  - Foundation models (e.g., TabPFN)

### Results

- Stored in `result.csv`
- Evaluated on 20 random splits
- Metrics: **Accuracy** and **AUROC**

---
## 📚 Citation & Contact

### Citation

```bibtex
@article{leerange,
  title={Range-limited Augmentation for Few-shot Learning in Tabular Data},
  author={Lee, Kyungeun and Eo, Moonjung and Cho, Hye-Seung and Yoon, Suhee and Yoon, Sanghyu and Sim, Ye Seul and Lim, Woohyung}
}
```

### Contact

For questions or collaborations, please contact the authors.
