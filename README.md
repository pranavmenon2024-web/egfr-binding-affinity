# EGFR Binding Affinity Prediction

A molecular machine learning pipeline predicting the binding affinity 
of small molecules against EGFR kinase (ChEMBL ID: CHEMBL203) using 
both a classical Random Forest baseline and a Graph Convolutional Network.

## Overview

1. **Data** — IC50 measurements fetched from ChEMBL, converted to pIC50, 
   filtered by molecular weight, deduplicated. ~3,491 compounds.
2. **Splitting** — Bemis–Murcko scaffold split (80/10/10) to evaluate 
   generalization to structurally novel compounds.
3. **Baseline** — RandomForestRegressor on 2048-bit Morgan fingerprints 
   (ECFP, radius=2).
4. **GCN** — 3-layer Graph Convolutional Network with global mean pooling, 
   trained with Adam optimizer, ReduceLROnPlateau scheduler, and early stopping.

## Results

| Model | RMSE | MAE | R² | Pearson | Spearman |
|---|---|---|---|---|---|
| Random Forest (baseline) | 0.929 | 0.711 | 0.613 | 0.792 | 0.761 |
| GCN (seed 42) | 1.378 | 1.122 | 0.148 | 0.402 | 0.383 |

The Random Forest substantially outperforms the GCN on this dataset. 
This is consistent with literature showing classical fingerprint-based 
models remain competitive on small molecular datasets under scaffold 
split conditions, where structural diversity between train and test sets 
is high. GCN results vary with random initialization; across seeds 0, 1 and 42 the
GCN reached R² 0.12–0.15 and Pearson 0.39–0.40, so the gap to the Random
Forest is consistent rather than an artifact of one run.

**Bug fix (Sep 2026):** an earlier version of the training loop updated the
best validation loss before checking whether to reset the early-stopping
counter, so the counter never reset and training always stopped at epoch 15.
With the fix, training runs until validation loss stops improving (epoch
44–54 depending on seed), and GCN R² rose from ~0.05 to ~0.15. The Random
Forest result is unaffected.

## Limitations and Future Work

- GCN performance is limited by dataset size and scaffold split difficulty.
  Larger datasets or transfer learning from pretrained molecular encoders 
  (e.g. ChemBERTa, GIN pretrained on ZINC) would likely improve results.
- The Random Forest is reported for one seed; GCN results were checked
  across three seeds. More seeds and confidence intervals would give more
  reliable estimates.
- Additional baselines (XGBoost, SVM with RBF kernel) would strengthen 
  the comparison.
- Atom features are minimal; incorporating partial charges, 3D coordinates, 
  or MMFF-minimized geometries could improve GCN performance.

## Directory Structure
egfr_project/
├── main.py         # Full pipeline
├── README.md       # This file
├── data/           # Cleaned dataset (egfr_pic50.csv)
└── outputs/        # Model weights and prediction plots
## Requirements

- Python 3.10+
- RDKit
- PyTorch + PyTorch Geometric
- pandas, scikit-learn, scipy, matplotlib
- chembl-webresource-client

## Usage

```bash
conda activate egfr-env
python3 main.py
```

If `data/egfr_pic50.csv` exists, the script uses it instead of re-downloading
from ChEMBL. Delete it to fetch fresh data.
