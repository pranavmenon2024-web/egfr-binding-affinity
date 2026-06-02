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
| GCN | ~1.45 | ~1.21 | ~0.05 | ~0.29 | ~0.25 |

The Random Forest substantially outperforms the GCN on this dataset. 
This is consistent with literature showing classical fingerprint-based 
models remain competitive on small molecular datasets under scaffold 
split conditions, where structural diversity between train and test sets 
is high. GCN results vary across runs due to random initialization; 
reported values are representative single-run results.

## Limitations and Future Work

- GCN performance is limited by dataset size and scaffold split difficulty.
  Larger datasets or transfer learning from pretrained molecular encoders 
  (e.g. ChemBERTa, GIN pretrained on ZINC) would likely improve results.
- Results represent single-run performance. Multi-seed averaging would 
  give more reliable estimates.
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
