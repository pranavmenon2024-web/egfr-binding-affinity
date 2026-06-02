"""
Enhanced EGFR binding affinity prediction with GCN and baseline models.

This script extends the basic GCN prototype by adding several improvements:

* **Baseline model:** A classical machine‑learning baseline is trained on
  Morgan (ECFP) fingerprints using a random forest regressor. This
  provides a point of comparison for the graph neural network.
* **Scaffold splitting:** Instead of a simple random split, the data
  is split using Bemis–Murcko scaffolds to better evaluate
  generalization to unseen chemical scaffolds. The dataset is
  partitioned into train/validation/test sets by scaffold.
* **Additional metrics:** Besides MSE, RMSE, MAE and R2, Pearson and
  Spearman correlations are computed to assess monotonic relationships
  between predicted and true values.

The script follows these steps:

1. Download EGFR activity data from ChEMBL and clean it.
2. Convert SMILES to RDKit molecules and compute pIC50.
3. Generate Morgan fingerprints and build graph representations.
4. Perform a scaffold split to create train/val/test partitions.
5. Train a RandomForestRegressor on fingerprints and evaluate.
6. Train a GCN on graphs and evaluate.
7. Output metrics and save prediction plots.

Run this script from the project root with:

    python main.py

The results will be saved in the `data` and `outputs` directories.
"""
import warnings
warnings.filterwarnings("ignore")
import math
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from chembl_webresource_client.new_client import new_client
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr, spearmanr
from torch import nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool


def fetch_egfr_data(limit: int = 5000) -> pd.DataFrame:
    """Fetch EGFR IC50 data from ChEMBL and remove duplicates."""
    activity = new_client.activity
    res = activity.filter(
        target_chembl_id="CHEMBL203",
        standard_type="IC50",
        standard_units="nM",
        standard_relation="=",
    ).only([
        "molecule_chembl_id",
        "canonical_smiles",
        "standard_value",
    ])
    rows: List[Dict[str, float]] = []
    for i, item in enumerate(res):
        if i >= limit:
            break
        smiles = item.get("canonical_smiles")
        value = item.get("standard_value")
        chembl_id = item.get("molecule_chembl_id")
        if not smiles or not value:
            continue
        try:
            ic50 = float(value)
        except (TypeError, ValueError):
            continue
        if ic50 <= 0:
            continue
        rows.append({"chembl_id": chembl_id, "smiles": smiles, "ic50_nM": ic50})
    df = pd.DataFrame(rows).drop_duplicates(subset=["smiles"]).reset_index(drop=True)
    return df


def ic50_to_pic50(ic50_nM: float) -> float:
    """Convert IC50 in nM to pIC50."""
    molar = ic50_nM * 1e-9
    return -math.log10(molar)


# Fixed atomic numbers for one‑hot encoding
ATOM_LIST: List[int] = [1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53]


def atom_features(atom: Chem.Atom) -> List[float]:
    atomic_num = atom.GetAtomicNum()
    one_hot = [1.0 if atomic_num == x else 0.0 for x in ATOM_LIST]
    unknown = [1.0] if atomic_num not in ATOM_LIST else [0.0]
    return one_hot + unknown + [
        float(atom.GetDegree()),
        float(atom.GetFormalCharge()),
        float(atom.GetHybridization()),
        float(atom.GetIsAromatic()),
        float(atom.GetTotalNumHs()),
    ]


def mol_to_graph(smiles: str, y_value: float) -> Data:
    mol = Chem.MolFromSmiles(smiles)
    # Node features
    x = torch.tensor([atom_features(a) for a in mol.GetAtoms()], dtype=torch.float)
    # Edges
    edge_indices: List[List[int]] = []
    edge_attrs: List[List[float]] = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bond_type = float(bond.GetBondTypeAsDouble())
        edge_indices.append([i, j])
        edge_attrs.append([bond_type])
        edge_indices.append([j, i])
        edge_attrs.append([bond_type])
    if edge_indices:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 1), dtype=torch.float)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.y = torch.tensor([y_value], dtype=torch.float)
    data.smiles = smiles
    return data


def compute_morgan_fp(mol: Chem.Mol, radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    """Compute a Morgan fingerprint as a numpy array of bits."""
    from rdkit.Chem import rdFingerprintGenerator
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    fp = generator.GetFingerprint(mol)
    return np.array(fp)


def generate_scaffold(smiles: str) -> str:
    """Return the Bemis–Murcko scaffold for a SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    try:
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
    except Exception:
        scaffold = ""
    return scaffold


def scaffold_split(indices: List[int], smiles_list: List[str], train_frac: float = 0.8, val_frac: float = 0.1) -> Tuple[List[int], List[int], List[int]]:
    """Split molecule indices into train/val/test sets by scaffold.

    Scaffolds are sorted by the number of molecules they contain. Entire
    scaffolds are assigned to train, then validation, then test sets until
    the desired fractions are reached.

    Args:
        indices: List of dataset indices.
        smiles_list: List of SMILES strings corresponding to indices.
        train_frac: Fraction of molecules to include in the training set.
        val_frac: Fraction to include in the validation set.

    Returns:
        (train_indices, val_indices, test_indices)
    """
    scaffolds: Dict[str, List[int]] = defaultdict(list)
    for idx in indices:
        scaf = generate_scaffold(smiles_list[idx])
        scaffolds[scaf].append(idx)
    # Sort scaffolds by descending size
    sorted_scaffolds = sorted(scaffolds.items(), key=lambda kv: len(kv[1]), reverse=True)
    total = len(indices)
    train_indices: List[int] = []
    val_indices: List[int] = []
    test_indices: List[int] = []
    for scaf, idxs in sorted_scaffolds:
        if len(train_indices) / total < train_frac:
            train_indices += idxs
        elif len(val_indices) / total < val_frac:
            val_indices += idxs
        else:
            test_indices += idxs
    return train_indices, val_indices, test_indices


class GCNRegressor(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 64):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.regressor = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1),
        )

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        x = global_mean_pool(x, batch)
        out = self.regressor(x)
        return out.view(-1)


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device) -> float:
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        pred = model(batch.x, batch.edge_index, batch.batch)
        loss = F.mse_loss(pred, batch.y.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[Dict[str, float], List[float], List[float]]:
    model.eval()
    preds: List[float] = []
    trues: List[float] = []
    for batch in loader:
        batch = batch.to(device)
        pred = model(batch.x, batch.edge_index, batch.batch)
        preds.extend(pred.cpu().numpy().tolist())
        trues.extend(batch.y.view(-1).cpu().numpy().tolist())
    mse = mean_squared_error(trues, preds)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(trues, preds)
    r2 = r2_score(trues, preds)
    # Additional correlations
    try:
        pearson_corr = pearsonr(trues, preds)[0]
        spearman_corr = spearmanr(trues, preds)[0]
    except Exception:
        pearson_corr = float('nan')
        spearman_corr = float('nan')
    metrics = {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "pearson": pearson_corr,
        "spearman": spearman_corr,
    }
    return metrics, preds, trues


def plot_predictions(y_true: List[float], y_pred: List[float], title: str, out_path: str) -> None:
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.6)
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")
    plt.xlabel("Actual pIC50")
    plt.ylabel("Predicted pIC50")
    plt.title(title)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def main() -> None:
    os.makedirs("data", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    print("Fetching EGFR data from ChEMBL...")
    df = fetch_egfr_data(limit=5000)
    print(f"Fetched {len(df)} raw rows")

    df["mol"] = df["smiles"].apply(Chem.MolFromSmiles)
    df = df[df["mol"].notnull()].copy()
    df["mw"] = df["mol"].apply(Descriptors.MolWt)
    df = df[(df["mw"] > 100) & (df["mw"] < 900)].copy()
    df["pIC50"] = df["ic50_nM"].apply(ic50_to_pic50)
    df = df.dropna(subset=["pIC50"]).reset_index(drop=True)

    # Save cleaned data
    df[["chembl_id", "smiles", "ic50_nM", "pIC50"]].to_csv(os.path.join("data", "egfr_pic50.csv"), index=False)
    print(f"Cleaned dataset has {len(df)} compounds")

    # Compute Morgan fingerprints
    print("Computing Morgan fingerprints...")
    fingerprints = []
    for mol in df["mol"]:
        fp = compute_morgan_fp(mol)
        fingerprints.append(fp)
    fingerprints = np.array(fingerprints)

    # Build graph data objects
    print("Building graph objects...")
    graph_data: List[Data] = []
    for idx, row in df.iterrows():
        g = mol_to_graph(row["smiles"], row["pIC50"])
        graph_data.append(g)

    # Perform scaffold split
    print("Performing scaffold split...")
    indices = list(range(len(df)))
    train_idx, val_idx, test_idx = scaffold_split(indices, df["smiles"].tolist(), train_frac=0.8, val_frac=0.1)
    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

    # Split data for baseline
    X_train, y_train = fingerprints[train_idx], df.loc[train_idx, "pIC50"].values
    X_val, y_val = fingerprints[val_idx], df.loc[val_idx, "pIC50"].values
    X_test, y_test = fingerprints[test_idx], df.loc[test_idx, "pIC50"].values

    # Train baseline model
    print("Training RandomForest baseline...")
    rf = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    # Evaluate baseline
    y_pred_rf = rf.predict(X_test)
    baseline_mse = mean_squared_error(y_test, y_pred_rf)
    baseline_rmse = math.sqrt(baseline_mse)
    baseline_mae = mean_absolute_error(y_test, y_pred_rf)
    baseline_r2 = r2_score(y_test, y_pred_rf)
    try:
        baseline_pearson = pearsonr(y_test, y_pred_rf)[0]
        baseline_spearman = spearmanr(y_test, y_pred_rf)[0]
    except Exception:
        baseline_pearson = float('nan')
        baseline_spearman = float('nan')

    print("Baseline metrics (test):")
    print(f"RMSE: {baseline_rmse:.4f}")
    print(f"MAE:  {baseline_mae:.4f}")
    print(f"R2:   {baseline_r2:.4f}")
    print(f"Pearson: {baseline_pearson:.4f}")
    print(f"Spearman: {baseline_spearman:.4f}")

    # Prepare data loaders for GCN
    train_graphs = [graph_data[i] for i in train_idx]
    val_graphs = [graph_data[i] for i in val_idx]
    test_graphs = [graph_data[i] for i in test_idx]

    train_loader = DataLoader(train_graphs, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=32)
    test_loader = DataLoader(test_graphs, batch_size=32)

    # Determine device
    device = torch.device(
        "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    print(f"Using device: {device}")

    # Instantiate model
    in_channels = graph_data[0].x.shape[1]
    model = GCNRegressor(in_channels=in_channels, hidden_channels=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    best_val_mse = float('inf')
    patience_counter = 0
    PATIENCE = 15
    best_model_path = os.path.join("outputs", "best_model_gcn.pt")

    # Train GCN with scaffold split
    for epoch in range(1, 101):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_metrics, _, _ = evaluate(model, val_loader, device)
        if val_metrics["mse"] < best_val_mse:
            best_val_mse = val_metrics["mse"]
            torch.save(model.state_dict(), best_model_path)
        print(
            f"Epoch {epoch:02d} | Train MSE: {train_loss:.4f} | "
            f"Val RMSE: {val_metrics['rmse']:.4f} | "
            f"Val MAE: {val_metrics['mae']:.4f} | "
            f"Val R2: {val_metrics['r2']:.4f} | "
            f"Val Pearson: {val_metrics['pearson']:.4f} | "
            f"Val Spearman: {val_metrics['spearman']:.4f}"
        )
        scheduler.step(val_metrics["mse"])
        if val_metrics["mse"] < best_val_mse:
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping triggered at epoch {epoch}")
                break

    # Load best model and evaluate on test set
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    test_metrics, gcn_pred, gcn_true = evaluate(model, test_loader, device)
    print("\nGCN test metrics:")
    for k, v in test_metrics.items():
        print(f"{k}: {v:.4f}")

    # Save prediction plots
    plot_predictions(
        y_true=y_test.tolist(),
        y_pred=y_pred_rf.tolist(),
        title="RandomForest Baseline: Predicted vs Actual",
        out_path=os.path.join("outputs", "baseline_pred_vs_actual.png"),
    )
    plot_predictions(
        y_true=gcn_true,
        y_pred=gcn_pred,
        title="GCN: Predicted vs Actual",
        out_path=os.path.join("outputs", "gcn_pred_vs_actual.png"),
    )
    print("Saved prediction plots to outputs/ directory")


if __name__ == "__main__":
    main()
