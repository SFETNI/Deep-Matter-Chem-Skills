---
name: molecular-property-gnn
domain: geometric-deep-learning
maturity: draft
tools:
  - RDKit
  - PyTorch Geometric
  - DGL-LifeSci
  - DeepChem
  - scikit-learn
  - matplotlib
tested_versions:
  RDKit: "2023.9.4"
  PyTorch Geometric: "2.5.0"
  DGL-LifeSci: "0.3.2"
  DeepChem: "2.8.0"
  scikit-learn: "1.4.0"
requires_gpu: true
estimated_runtime: hours-to-days
---

# Molecular Property GNN

## Description

This skill covers graph neural networks for molecular property prediction: molecular graph construction from SMILES, SDF, RDKit molecules, and 3D conformers; node and edge featurization; 2D message-passing models; 3D continuous-filter and directional models; validation under scaffold and chemistry shifts; uncertainty estimation; and integration with molecular featurization, surrogate validation, and scientific visualization. Invoke this skill when predicting molecular or small-cluster properties such as solubility, toxicity, HOMO/LUMO gaps, dipoles, reaction barriers, binding affinity, or materials-relevant molecular descriptors from molecular graphs or conformers.

## Domain Context

Molecular GNNs encode a molecule as a graph where atoms are nodes and bonds, distances, or neighbor relations are edges. A 2D molecular GNN operates on chemical connectivity: atom types, formal charges, aromaticity, bond orders, stereochemistry, and ring membership. A 3D molecular GNN also uses conformer coordinates and must respect Euclidean symmetry: scalar properties should be invariant to translation and rotation, while vector or tensor properties require equivariant outputs.

The graph is an approximation to the molecular information available at prediction time. A SMILES-derived graph has no explicit electron density, solvent environment, crystal packing, conformer ensemble, or assay protocol. A conformer-derived graph captures geometry, but only for the chosen conformation and coordinate generation workflow. For flexible molecules, properties such as solvation free energy, binding affinity, or dipole moment may depend on an ensemble rather than a single conformer. [EXPERT REVIEW NEEDED]

The validation split often matters more than the architecture. Random splits measure interpolation over molecules similar to those already seen. Scaffold splits, assay splits, time splits, or chemistry-aware splits measure more realistic transfer to new chemotypes, experimental campaigns, or molecular families. Reporting only random-split benchmark metrics is usually insufficient for scientific use.

Molecular GNNs are not automatically superior to fingerprints. Morgan fingerprints with ridge, random forest, or gradient boosting can be strong baselines for small 2D QSAR datasets. Learned GNN representations are most useful when the dataset is large enough, when atom/bond context matters, when 3D geometry is relevant, or when the model will be reused for transfer learning. For tens to hundreds of molecules, classical descriptors and Gaussian processes may be more reliable unless strong priors or pretraining are available.

## When to Use This Skill

- Predicting molecular properties from known molecular structures, SMILES, SDF files, or conformer ensembles.
- Comparing learned graph representations against RDKit fingerprints, physicochemical descriptors, or Gaussian-process baselines.
- Building models where atom-level features, bond topology, stereochemistry, or conformer geometry are expected to matter.
- Validating molecular ML models under scaffold, chemistry-family, assay, or time-based distribution shift.
- Estimating applicability domains or uncertainty for screening, active learning, or Bayesian optimization over molecules.
- Auditing molecular datasets for leakage through duplicated compounds, near-identical scaffolds, target-derived descriptors, or conformer-generation artifacts.

## When Not to Use This Skill

- **When the target depends on an unavailable experimental protocol.** Binding affinity, toxicity, and assay readouts may depend on pH, concentration, cell line, protein construct, or measurement conditions absent from the molecular graph.
- **When only a few dozen labeled molecules are available.** Start with fingerprints, regularized linear models, random forests, Gaussian processes, and nested cross-validation before training a high-capacity GNN.
- **When the molecular identity is ambiguous.** Mixtures, salts, tautomers, protonation states, organometallics, polymers, and partially specified structures require curation before graph construction.
- **When stereochemistry or conformers are essential but missing.** A 2D graph that drops stereocenters cannot predict stereospecific properties; a single generated conformer may be insufficient for flexible molecules.
- **When periodic packing controls the property.** Molecular crystals, surface adsorbates, and condensed-phase interfaces may require crystal or atomistic graph models rather than isolated-molecule GNNs.
- **When the deployment chemistry is outside the training domain.** A GNN trained on neutral drug-like molecules will not reliably extrapolate to charged radicals, transition-metal complexes, ionic liquids, or reactive intermediates without validation. [EXPERT REVIEW NEEDED]

## Core Concepts

- **Molecular graph:** A graph representation with atoms as nodes and chemical bonds as edges. Built from SMILES, SDF/MOL files, RDKit `Mol` objects, or bond perception from coordinates.
- **Node features:** Per-atom inputs such as atomic number, degree, valence, formal charge, aromaticity, hybridization, ring membership, chirality, isotope, partial charge, donor/acceptor flags, or 3D coordinates.
- **Edge features:** Per-bond inputs such as bond type, conjugation, aromaticity, ring membership, stereo configuration, graph distance, interatomic distance, or radial basis expansion.
- **2D GNN:** Message passing over the topological bond graph. Appropriate for many QSAR tasks where connectivity and local chemistry dominate.
- **3D GNN:** Message passing over spatial neighbor graphs or bonded graphs with geometric features. Needed when conformation, nonbonded interactions, or directional geometry influence the target.
- **Message passing neural network (MPNN):** A model that updates atom embeddings by aggregating neighbor messages and then pools atom embeddings to a molecular representation.
- **Graph convolution and graph attention:** Message-passing variants where neighbor contributions are fixed by learned filters or attention weights. Attention weights are not automatically chemically interpretable.
- **SchNet-style continuous-filter model:** A 3D model using distance-based continuous filters and radial basis functions. Rotationally invariant for scalar targets.
- **DimeNet-style directional model:** A 3D model using distances and angles to encode directional interactions. Useful when angular geometry controls the property.
- **Equivariant molecular model:** A 3D model whose internal vector/tensor features transform consistently under rotation. Needed for vector properties and often useful for data-efficient scalar prediction.
- **Fingerprint baseline:** Fixed descriptors such as Morgan/ECFP, MACCS, RDKit physicochemical descriptors, or Mordred descriptors. These are strong baselines and should be reported before claiming a GNN advantage.
- **Scaffold split:** A split that holds out Bemis-Murcko scaffolds or chemistry families. More realistic than random splits for new-chemotype generalization.
- **Applicability domain:** The region of molecular space where the model has supporting training data, measured by scaffold membership, fingerprint distance, descriptor distance, ensemble uncertainty, or conformal calibration.

## Key Workflows

### Workflow 1: Build leakage-aware molecular graphs from SMILES

Start with explicit molecule standardization. Invalid molecules, disconnected salts, missing stereochemistry, and duplicated structures should be handled before graph conversion.

```python
from dataclasses import dataclass
from collections import Counter
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from sklearn.model_selection import GroupKFold


@dataclass
class MoleculeRecord:
    smiles: str
    mol: Chem.Mol
    inchikey: str
    scaffold: str
    target: float | None = None


def standardize_smiles(smiles: str) -> Chem.Mol | None:
    """
    Parse a SMILES string into a sanitized RDKit molecule.

    This is intentionally conservative. Production workflows should use a
    project-specific salt/protonation/tautomer policy and record it.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None
    return mol


def murcko_scaffold(smiles: str) -> str:
    from rdkit.Chem.Scaffolds import MurckoScaffold
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "invalid"
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    return Chem.MolToSmiles(scaffold, isomericSmiles=True) or "acyclic"


def build_records(df: pd.DataFrame, smiles_col: str, target_col: str) -> list[MoleculeRecord]:
    records = []
    failures = []
    for i, row in df.iterrows():
        mol = standardize_smiles(row[smiles_col])
        if mol is None:
            failures.append(i)
            continue
        canonical = Chem.MolToSmiles(mol, isomericSmiles=True)
        inchikey = Chem.MolToInchiKey(mol)
        records.append(MoleculeRecord(
            smiles=canonical,
            mol=mol,
            inchikey=inchikey,
            scaffold=murcko_scaffold(canonical),
            target=None if pd.isna(row[target_col]) else float(row[target_col]),
        ))

    print(f"Parsed {len(records)} molecules; failed {len(failures)}")
    duplicate_counts = Counter(r.inchikey for r in records)
    n_duplicates = sum(c - 1 for c in duplicate_counts.values() if c > 1)
    if n_duplicates:
        print(f"WARNING: {n_duplicates} duplicate InChIKey entries before splitting")
    return records


def atom_features(atom: Chem.Atom) -> dict:
    return {
        "atomic_num": atom.GetAtomicNum(),
        "degree": atom.GetDegree(),
        "formal_charge": atom.GetFormalCharge(),
        "hybridization": str(atom.GetHybridization()),
        "is_aromatic": atom.GetIsAromatic(),
        "chiral_tag": str(atom.GetChiralTag()),
        "num_hs": atom.GetTotalNumHs(),
        "is_in_ring": atom.IsInRing(),
    }


def bond_features(bond: Chem.Bond) -> dict:
    return {
        "bond_type": str(bond.GetBondType()),
        "is_conjugated": bond.GetIsConjugated(),
        "is_aromatic": bond.GetIsAromatic(),
        "is_in_ring": bond.IsInRing(),
        "stereo": str(bond.GetStereo()),
    }


def audit_graph_inputs(records: list[MoleculeRecord]) -> None:
    n_missing_chiral = 0
    n_charged = 0
    n_radical = 0
    for rec in records:
        for atom in rec.mol.GetAtoms():
            if atom.HasProp("_ChiralityPossible") and atom.GetChiralTag() == Chem.ChiralType.CHI_UNSPECIFIED:
                n_missing_chiral += 1
            if atom.GetFormalCharge() != 0:
                n_charged += 1
            if atom.GetNumRadicalElectrons() > 0:
                n_radical += 1
    print(f"Potential unspecified stereocenters: {n_missing_chiral}")
    print(f"Charged atom count: {n_charged}")
    print(f"Radical atom count: {n_radical}")
    if n_missing_chiral:
        print("WARNING: stereochemistry may be incomplete; scaffold metrics can be optimistic")


def scaffold_group_splits(records: list[MoleculeRecord], n_splits: int = 5):
    groups = np.array([r.scaffold for r in records])
    targets = np.array([r.target for r in records], dtype=float)
    valid = ~np.isnan(targets)
    splitter = GroupKFold(n_splits=min(n_splits, len(set(groups[valid]))))
    for fold, (train_idx, test_idx) in enumerate(splitter.split(np.zeros(valid.sum()), targets[valid], groups[valid]), 1):
        valid_indices = np.flatnonzero(valid)
        yield fold, valid_indices[train_idx], valid_indices[test_idx]
```

### Workflow 2: Convert RDKit molecules to PyTorch Geometric data

This example keeps the featurizer explicit so the feature vocabulary can be versioned. Production projects should freeze the exact atom and bond feature schema in metadata.

```python
import torch
from torch_geometric.data import Data
from rdkit import Chem


ATOM_TYPES = [1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53]
BOND_TYPES = [
    Chem.BondType.SINGLE,
    Chem.BondType.DOUBLE,
    Chem.BondType.TRIPLE,
    Chem.BondType.AROMATIC,
]


def one_hot(value, choices):
    return [value == c for c in choices] + [value not in choices]


def atom_feature_vector(atom: Chem.Atom) -> list[float]:
    return (
        one_hot(atom.GetAtomicNum(), ATOM_TYPES)
        + one_hot(atom.GetDegree(), [0, 1, 2, 3, 4, 5])
        + one_hot(atom.GetFormalCharge(), [-2, -1, 0, 1, 2])
        + [atom.GetIsAromatic(), atom.IsInRing()]
        + one_hot(str(atom.GetHybridization()), ["SP", "SP2", "SP3", "SP3D", "SP3D2"])
        + one_hot(str(atom.GetChiralTag()), ["CHI_UNSPECIFIED", "CHI_TETRAHEDRAL_CW", "CHI_TETRAHEDRAL_CCW"])
    )


def bond_feature_vector(bond: Chem.Bond) -> list[float]:
    return (
        one_hot(bond.GetBondType(), BOND_TYPES)
        + [bond.GetIsConjugated(), bond.GetIsAromatic(), bond.IsInRing()]
        + one_hot(str(bond.GetStereo()), ["STEREONONE", "STEREOZ", "STEREOE"])
    )


def mol_to_pyg_data(mol: Chem.Mol, y: float | None = None) -> Data:
    x = torch.tensor(
        [atom_feature_vector(atom) for atom in mol.GetAtoms()],
        dtype=torch.float32,
    )

    edge_index = []
    edge_attr = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        feat = bond_feature_vector(bond)
        edge_index.extend([[i, j], [j, i]])
        edge_attr.extend([feat, feat])

    if edge_index:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float32)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, len(bond_feature_vector(Chem.MolFromSmiles("CC").GetBondWithIdx(0)))), dtype=torch.float32)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    if y is not None:
        data.y = torch.tensor([float(y)], dtype=torch.float32)
    data.smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
    data.inchikey = Chem.MolToInchiKey(mol)
    return data


def validate_pyg_dataset(graphs: list[Data]) -> None:
    n_empty_edges = sum(g.edge_index.numel() == 0 for g in graphs)
    n_nan_y = sum(hasattr(g, "y") and torch.isnan(g.y).any().item() for g in graphs)
    n_single_atom = sum(g.num_nodes == 1 for g in graphs)
    print(f"Graphs: {len(graphs)}")
    print(f"Single-atom graphs: {n_single_atom}")
    print(f"Graphs with no bonds: {n_empty_edges}")
    print(f"Graphs with NaN target: {n_nan_y}")
    if n_empty_edges > 0:
        print("WARNING: no-bond molecules may need special handling in message passing")
```

### Workflow 3: Train a 2D molecular MPNN with scaffold validation

Use scaffold or chemistry-aware validation for model selection. Keep preprocessing, splitting, and early stopping metadata with the model artifact.

```python
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, global_mean_pool


class MolecularGNN(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int = 128, depth: int = 4):
        super().__init__()
        self.node_encoder = nn.Linear(node_dim, hidden_dim)
        self.edge_encoder = nn.Linear(edge_dim, hidden_dim)
        self.convs = nn.ModuleList()
        for _ in range(depth):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(GINEConv(mlp))
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, data):
        h = self.node_encoder(data.x)
        e = self.edge_encoder(data.edge_attr)
        for conv in self.convs:
            h = h + conv(h, data.edge_index, e)
        pooled = global_mean_pool(h, data.batch)
        return self.readout(pooled).squeeze(-1)


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    n = 0
    for batch in loader:
        batch = batch.to(device)
        pred = model(batch)
        y = batch.y.view(-1)
        loss = F.mse_loss(pred, y)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        total_loss += loss.item() * len(y)
        n += len(y)
    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    y_true = []
    y_pred = []
    for batch in loader:
        batch = batch.to(device)
        pred = model(batch).detach().cpu()
        y_true.append(batch.y.view(-1).cpu())
        y_pred.append(pred)
    y_true = torch.cat(y_true)
    y_pred = torch.cat(y_pred)
    mae = torch.mean(torch.abs(y_pred - y_true)).item()
    rmse = torch.sqrt(torch.mean((y_pred - y_true) ** 2)).item()
    return {"mae": mae, "rmse": rmse, "y_true": y_true.numpy(), "y_pred": y_pred.numpy()}


def train_scaffold_fold(
    graphs: list,
    train_idx,
    val_idx,
    output_dir: str,
    batch_size: int = 64,
    max_epochs: int = 300,
    patience: int = 40,
    device: str = "cuda",
) -> dict:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    train_loader = DataLoader([graphs[i] for i in train_idx], batch_size=batch_size, shuffle=True)
    val_loader = DataLoader([graphs[i] for i in val_idx], batch_size=batch_size, shuffle=False)

    node_dim = graphs[0].x.shape[1]
    edge_dim = graphs[0].edge_attr.shape[1]
    model = MolecularGNN(node_dim=node_dim, edge_dim=edge_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)

    best_mae = float("inf")
    best_epoch = 0
    history = []
    for epoch in range(1, max_epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val = evaluate(model, val_loader, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_mae": val["mae"], "val_rmse": val["rmse"]})
        if val["mae"] < best_mae:
            best_mae = val["mae"]
            best_epoch = epoch
            torch.save(model.state_dict(), Path(output_dir) / "best_model.pt")
        if epoch - best_epoch > patience:
            break

    Path(output_dir, "history.json").write_text(json.dumps(history, indent=2))
    return {"best_val_mae": best_mae, "best_epoch": best_epoch, "n_epochs": len(history)}
```

### Workflow 4: Generate and audit 3D conformers for geometry-aware models

3D molecular GNNs are only as reliable as the conformers. Track conformer generation settings, minimization status, charge/protonation assumptions, and whether the property is conformer-specific or ensemble-averaged.

```python
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np


def generate_conformers(
    mol: Chem.Mol,
    n_conformers: int = 20,
    max_attempts: int = 200,
    prune_rms_thresh: float = 0.5,
    random_seed: int = 17,
) -> Chem.Mol | None:
    """
    Generate and MMFF-optimize conformers.

    Use a quantum-chemistry or force-field protocol appropriate to the target
    if conformer energy ordering matters. RDKit/MMFF is a screening-level step.
    [EXPERT REVIEW NEEDED]
    """
    mol_h = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = random_seed
    params.pruneRmsThresh = prune_rms_thresh
    params.maxAttempts = max_attempts
    conf_ids = AllChem.EmbedMultipleConfs(mol_h, numConfs=n_conformers, params=params)
    if len(conf_ids) == 0:
        return None

    results = AllChem.MMFFOptimizeMoleculeConfs(mol_h, maxIters=500)
    energies = []
    for conf_id, (status, energy) in zip(conf_ids, results):
        if status != 0:
            print(f"WARNING: conformer {conf_id} did not converge in MMFF optimization")
        energies.append(float(energy))
        mol_h.GetConformer(conf_id).SetDoubleProp("mmff_energy", float(energy))

    min_energy = min(energies)
    for conf_id, energy in zip(conf_ids, energies):
        mol_h.GetConformer(conf_id).SetDoubleProp("relative_energy_kcal_mol", float(energy - min_energy))
    return mol_h


def conformer_distance_edges(mol: Chem.Mol, conf_id: int, cutoff_A: float = 5.0):
    """
    Build a spatial neighbor list for 3D GNNs.
    Bonded edges and distance edges should be encoded separately in production.
    """
    conf = mol.GetConformer(conf_id)
    positions = np.array(conf.GetPositions(), dtype=float)
    src, dst, dist = [], [], []
    for i in range(len(positions)):
        for j in range(len(positions)):
            if i == j:
                continue
            d = np.linalg.norm(positions[i] - positions[j])
            if d <= cutoff_A:
                src.append(i)
                dst.append(j)
                dist.append(d)
    return np.array([src, dst], dtype=int), np.array(dist, dtype=float)


def audit_conformer_ensemble(mol: Chem.Mol) -> None:
    energies = [
        conf.GetDoubleProp("relative_energy_kcal_mol")
        for conf in mol.GetConformers()
        if conf.HasProp("relative_energy_kcal_mol")
    ]
    print(f"Conformers: {mol.GetNumConformers()}")
    if energies:
        print(f"Relative energy range: {min(energies):.2f} to {max(energies):.2f} kcal/mol")
        n_high = sum(e > 10.0 for e in energies)
        if n_high:
            print(f"WARNING: {n_high} conformers are >10 kcal/mol above minimum")
```

### Workflow 5: Fingerprint baselines and scaffold-split comparison

Do not claim a GNN improves molecular prediction until it beats strong baselines on the same split. For small datasets, the baseline may be the more defensible model.

```python
import numpy as np
import pandas as pd
from rdkit.Chem import rdFingerprintGenerator
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


def morgan_fingerprint_matrix(mols: list[Chem.Mol], radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    arr = np.zeros((len(mols), n_bits), dtype=np.float32)
    for i, mol in enumerate(mols):
        fp = generator.GetFingerprint(mol)
        arr[i, list(fp.GetOnBits())] = 1.0
    return arr


def compare_fingerprint_baselines(records, train_idx, test_idx) -> pd.DataFrame:
    mols = [r.mol for r in records]
    y = np.array([r.target for r in records], dtype=float)
    X = morgan_fingerprint_matrix(mols)

    models = {
        "ridge_morgan": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler(with_mean=False)),
            ("ridge", Ridge(alpha=1.0)),
        ]),
        "random_forest_morgan": RandomForestRegressor(
            n_estimators=500,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),
    }

    rows = []
    for name, model in models.items():
        model.fit(X[train_idx], y[train_idx])
        pred = model.predict(X[test_idx])
        rows.append({
            "model": name,
            "mae": float(mean_absolute_error(y[test_idx], pred)),
            "r2": float(r2_score(y[test_idx], pred)),
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
        })
    return pd.DataFrame(rows)
```

### Workflow 6: Applicability-domain and uncertainty checks

Uncertainty estimates must be validated out-of-fold. Ensemble variance, MC dropout, GP uncertainty, and conformal intervals are decision aids only after calibration.

```python
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances


def tanimoto_distance_matrix(fp_a: np.ndarray, fp_b: np.ndarray) -> np.ndarray:
    # Binary Morgan fingerprints: distance = 1 - intersection / union.
    inter = fp_a @ fp_b.T
    a_sum = fp_a.sum(axis=1, keepdims=True)
    b_sum = fp_b.sum(axis=1, keepdims=True).T
    union = a_sum + b_sum - inter
    return 1.0 - inter / np.maximum(union, 1.0)


def label_applicability_domain(records, train_idx, eval_idx, distance_threshold: float = 0.6) -> pd.DataFrame:
    mols = [r.mol for r in records]
    X = morgan_fingerprint_matrix(mols)
    d = tanimoto_distance_matrix(X[eval_idx], X[train_idx])
    nearest = d.min(axis=1)
    rows = []
    for idx, dist in zip(eval_idx, nearest):
        rows.append({
            "index": int(idx),
            "smiles": records[idx].smiles,
            "nearest_train_tanimoto_distance": float(dist),
            "coverage_class": "in_domain" if dist <= distance_threshold else "out_of_domain",
        })
    out = pd.DataFrame(rows)
    print(out["coverage_class"].value_counts())
    return out


def calibration_by_uncertainty_quantile(pred_df: pd.DataFrame, n_bins: int = 5) -> pd.DataFrame:
    """
    pred_df columns: y_true, y_pred, y_std. Predictions must be out-of-fold.
    """
    df = pred_df.copy()
    df["abs_error"] = (df["y_pred"] - df["y_true"]).abs()
    df["uncertainty_bin"] = pd.qcut(df["y_std"], q=n_bins, duplicates="drop")
    summary = df.groupby("uncertainty_bin").agg(
        n=("abs_error", "size"),
        mean_std=("y_std", "mean"),
        mae=("abs_error", "mean"),
        p90_error=("abs_error", lambda x: x.quantile(0.90)),
    ).reset_index()
    print(summary)
    if summary["mae"].corr(summary["mean_std"]) < 0.5:
        print("WARNING: uncertainty is weakly correlated with error; do not use it for acquisition")
    return summary
```

## Common Failure Modes

| Failure | Symptom | Likely Cause | Diagnostic | Fix |
|---------|---------|--------------|------------|-----|
| Scaffold leakage | Random split looks strong but scaffold split fails | Same core scaffold appears in train and test | Compare random, scaffold, and time splits; inspect held-out scaffolds | Report scaffold split for new-chemotype claims; group duplicated scaffolds before splitting |
| Stereochemistry loss | Enantiomers or diastereomers receive identical predictions | SMILES canonicalization or featurizer dropped chiral tags | Count unspecified stereocenters; compare isomeric and non-isomeric SMILES | Preserve isomeric SMILES and chiral atom/bond features; curate missing stereochemistry |
| Invalid conformers | 3D model gives unstable or inconsistent predictions | ETKDG/MMFF failed, high-energy conformers retained, wrong protonation state | Audit conformer optimization status and relative energy distribution | Regenerate conformers; use ensemble pooling; align conformer protocol to target property |
| Target leakage through descriptors | Test metrics are implausibly good | Features encode experimental labels, DFT outputs, or assay-derived quantities | Trace every descriptor to information available at prediction time | Remove leaking descriptors; rebuild pipeline and rerun validation |
| Overfitting benchmark datasets | Leaderboard performance does not reproduce on project data | Hyperparameters tuned repeatedly on public test sets | Use untouched external test set or nested CV | Freeze model-selection protocol; report confidence intervals |
| Assay noise ignored | Model cannot exceed apparent noise floor but is treated as failing | Replicate variability or inconsistent experimental conditions dominate labels | Estimate replicate error and per-assay variance | Model assay/source effects; report error relative to label noise |
| Poor extrapolation to new chemotypes | High errors for held-out scaffold families | Training data lack similar chemistry | Nearest-neighbor fingerprint distance and scaffold error breakdown | Expand training set or restrict applicability domain |
| Random-split metric compared to scaffold baseline | GNN appears superior by using an easier split | Baselines and GNN evaluated under different split protocols | Recompute all models on identical folds | Compare only same split, same target, same preprocessing |
| Aromaticity perception mismatch | Same molecule gets different graph features from different file sources | SMILES/SDF sanitization and aromaticity model differ | Canonicalize inputs and compare atom aromatic flags | Standardize RDKit sanitization and version the parser |
| Molecule duplicates across splits | Test MAE nearly zero for repeated compounds | Duplicate InChIKeys or tautomer variants split apart | Deduplicate by InChIKey and tautomer/protonation policy before split | Collapse duplicates or group them into the same fold |
| Uncalibrated uncertainty | Ensemble variance does not increase on bad predictions | Ensemble members share data and architecture biases | Reliability diagram and uncertainty-error rank correlation | Calibrate with conformal methods or use uncertainty only as a heuristic |

## Best Practices

- **Freeze the molecular standardization policy before splitting.** Salt stripping, neutralization, tautomer canonicalization, protonation, and stereochemistry handling determine what the model sees.
- **Deduplicate before train/test split.** Use InChIKey, canonical SMILES, scaffold grouping, and where needed tautomer-aware matching.
- **Report random and scaffold splits separately.** Random-split results are useful for interpolation; scaffold splits are the minimum for new-chemotype claims.
- **Use fingerprint baselines.** Morgan fingerprints with ridge, random forest, or gradient boosting are strong controls and often outperform GNNs on small datasets.
- **Track every feature schema.** Atom feature order, bond feature order, allowable atom types, aromaticity model, and RDKit version are part of the model definition.
- **Treat 3D conformers as data, not decoration.** Store conformer generation settings, force field, optimization status, relative energies, and whether the target is conformer-specific or ensemble-averaged.
- **Do not overinterpret attention weights.** Graph attention can highlight message weights, but those weights are not equivalent to chemical causality without perturbation or counterfactual tests.
- **Evaluate per scaffold, assay, source, charge state, and size range.** Global MAE hides systematic failures on rare but scientifically important subdomains.
- **Validate uncertainty out-of-fold.** In-sample posterior or ensemble uncertainty is not a deployment calibration estimate.
- **Preserve negative and failed results.** Screening and active-learning campaigns need full candidate logs, not only successful or promising molecules.

## Integration with Other Skills

- **[materials-featurization](../../scientific-data/materials-featurization/SKILL.md):** Molecular GNNs are learned featurizers. Fixed descriptors such as Morgan fingerprints, RDKit descriptors, SOAP, and Coulomb matrices provide baselines and leakage diagnostics.
- **[scientific-visualization](../../scientific-data/scientific-visualization/SKILL.md):** Parity plots, residual distributions, scaffold-level error bars, uncertainty calibration plots, and molecule-highlight figures should follow reproducible plotting conventions.
- **[reproducibility-workflow](../../scientific-data/reproducibility-workflow/SKILL.md):** Molecular standardization rules, split files, featurizer schemas, random seeds, package versions, model checkpoints, and prediction artifacts must be versioned.
- **[small-data-ml](../../scientific-ml/small-data-ml/SKILL.md):** Small molecular datasets require nested CV, regularization, uncertainty reporting, and strong non-GNN baselines. This is an intentional forward reference until that skill is added.
- **[surrogate-validation](../../surrogate-active-learning/surrogate-validation/SKILL.md):** Scaffold splits, applicability-domain labels, calibration checks, and closed-loop validation follow the surrogate validation principles in that skill.
- **[gaussian-process-materials](../../surrogate-active-learning/gaussian-process-materials/SKILL.md):** Gaussian processes on fingerprints or molecular descriptors are strong small-data baselines and can provide calibrated uncertainty for molecular screening.
- **[molecular-featurization](../../cheminformatics/molecular-featurization/SKILL.md):** Graph construction, fingerprints, conformer features, and descriptor provenance should be defined by the molecular featurization workflow. This is an intentional forward reference until that skill is added.
- **[molecular-datasets](../../cheminformatics/molecular-datasets/SKILL.md):** Dataset curation, assay metadata, duplicate handling, and benchmark split definitions should be handled upstream. This is an intentional forward reference until that skill is added.
- **[crystal-property-gnn](../crystal-property-gnn/SKILL.md):** Crystal GNNs share message-passing and split-validation concepts but use periodic neighbor graphs, lattice information, and prototype-aware validation. This is an intentional forward reference until that skill is added.

## Key References

1. Gilmer, J., et al. (2017). *Neural message passing for quantum chemistry.* Proceedings of the 34th International Conference on Machine Learning. https://arxiv.org/abs/1704.01212
2. Yang, K., et al. (2019). *Analyzing learned molecular representations for property prediction.* Journal of Chemical Information and Modeling, 59, 3370-3388. https://doi.org/10.1021/acs.jcim.9b00237
3. Wu, Z., et al. (2018). *MoleculeNet: A benchmark for molecular machine learning.* Chemical Science, 9, 513-530. https://doi.org/10.1039/C7SC02664A
4. Schutt, K. T., et al. (2017). *SchNet: A continuous-filter convolutional neural network for modeling quantum interactions.* Advances in Neural Information Processing Systems, 30. https://arxiv.org/abs/1706.08566
5. Klicpera, J., et al. (2020). *Directional message passing for molecular graphs.* International Conference on Learning Representations. https://arxiv.org/abs/2003.03123
6. Wieder, O., et al. (2020). *A compact review of molecular property prediction with graph neural networks.* Drug Discovery Today: Technologies, 37, 1-12. https://doi.org/10.1016/j.ddtec.2020.11.009
