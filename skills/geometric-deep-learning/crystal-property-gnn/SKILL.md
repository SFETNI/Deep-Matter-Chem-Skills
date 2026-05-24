---
name: crystal-property-gnn
domain: geometric-deep-learning
maturity: draft
tools:
  - pymatgen
  - PyTorch Geometric
  - DGL
  - matgl
  - CHGNet
  - scikit-learn
  - matplotlib
tested_versions:
  pymatgen: "2024.11.13"
  PyTorch Geometric: "2.5.0"
  DGL: "2.1.0"
  matgl: "1.1.3"
  CHGNet: "0.3.0"
  scikit-learn: "1.4.0"
requires_gpu: true
estimated_runtime: hours-to-days
---

# Crystal Property GNN

## Description

This skill covers graph neural networks for crystal and periodic-material property prediction: crystal graph construction from periodic structures, node/edge/lattice/distance/angle features, periodic neighbor finding, cutoff selection, invariant and equivariant representations, leakage-aware splitting, uncertainty estimation, applicability-domain checks, and validation for properties such as formation energy, band gap, elastic constants, phonon-derived quantities, phase stability, conductivity proxies, and related materials targets. Invoke this skill when a model must learn directly from crystal structures rather than fixed composition or local-environment descriptors.

## Domain Context

Crystal GNNs represent a periodic structure as a graph whose nodes are atomic sites and whose edges connect sites through periodic images. Unlike molecular graphs, crystal graphs are not defined by chemical bonds alone. A single atom can have several periodic neighbors of the same site, and the edge identity depends on the lattice vector used to connect the pair. The graph therefore encodes both chemistry and geometry: species, fractional and Cartesian positions, lattice, neighbor distances, angles, and periodic image offsets.

The graph construction defines the physical approximation. A cutoff graph assumes that the target property is controlled by interactions within a finite radius. This is often reasonable for local structural contributions to formation energy or many learned representations, but it is incomplete for long-range electrostatics, polar response, magnetism, excitons, or transport properties unless the model includes appropriate global features or long-range corrections. [EXPERT REVIEW NEEDED]

Crystal structures also have representational ambiguity. The same material may appear as a primitive cell, conventional cell, supercell, symmetrized structure, relaxed structure, or database-derived standardized structure. A good crystal GNN should be invariant to cell choice and site ordering, but finite cutoffs, pooling, and preprocessing can make predictions sensitive to representation. Always test primitive/conventional and supercell consistency before trusting a model on mixed-source datasets.

Validation is inseparable from dataset provenance. Random train/test splits over Materials Project-like data often place polymorphs, near-duplicates, same-prototype compounds, or same chemical systems on both sides of the split. That measures interpolation over familiar chemistry, not transfer to new elements, new prototypes, or new database sources. Composition, structure-family, time, and leave-one-chemical-system-out splits are required for stronger claims.

## When to Use This Skill

- Predicting crystal or periodic-material properties from full structures, including atomic species, coordinates, and lattice.
- Comparing graph-based crystal representations against composition descriptors, SOAP/Voronoi descriptors, Gaussian processes, or small-data baselines.
- Training or evaluating CGCNN, MEGNet, SchNet-style, DimeNet-style, GemNet-style, ALIGNN-style, M3GNet-style, or CHGNet-style models.
- Auditing database-derived crystal ML datasets for duplicate structures, polymorph leakage, target correction inconsistency, or biased splits.
- Estimating uncertainty and applicability domains before screening hypothetical crystals, substitutions, defects, or high-throughput DFT candidates.
- Extracting pretrained crystal model embeddings as fixed descriptors for small-data property models.

## When Not to Use This Skill

- **When composition alone is the only available input.** Use composition descriptors or small-data ML baselines rather than inventing a structure.
- **When the property depends on an unmodeled electronic state.** Magnetic order, charged defects, polarons, excitons, and correlated-electron states need metadata or physics not present in a neutral relaxed structure graph. [EXPERT REVIEW NEEDED]
- **When the structure is disordered or partially occupied without a chosen representation.** Standard graph libraries generally require ordered sites. Enumerate or otherwise represent disorder intentionally.
- **When the target mixes incompatible references.** Formation energies or band gaps from different functionals, correction schemes, pseudopotentials, or database snapshots cannot be treated as one label without correction metadata.
- **When only a handful of labeled crystals are available.** Use fixed descriptors, pretrained embeddings, Gaussian processes, and rigorous small-data validation before training a high-capacity GNN.
- **When extrapolating to new chemistries without held-out evidence.** A crystal GNN trained on common oxides can give confident but unreliable predictions for nitrides, halides, intermetallics, or molecular crystals outside the training domain.

## Core Concepts

- **Crystal graph:** A graph whose nodes are periodic atomic sites and whose edges connect sites within a cutoff, often including periodic image offsets.
- **Periodic image feature:** Integer lattice translation vector identifying which periodic copy of a neighbor produced an edge. Needed to reconstruct the edge vector.
- **Node features:** Atomic number, learned element embedding, group/period, electronegativity, covalent radius, oxidation state, magnetic moment, site property tags, or one-hot species encoding.
- **Edge features:** Distance, radial basis expansion, bond type heuristic, Voronoi weight, image vector, edge direction, or edge embedding.
- **Lattice features:** Cell lengths, angles, volume, density, reciprocal lattice, strain tensor, or global state features. Some architectures include them explicitly; others rely on edge vectors.
- **Angle features:** Triplet information such as bond angles or line-graph edges. Used by DimeNet, GemNet, and ALIGNN-style models to capture directional geometry.
- **Cutoff radius:** Maximum neighbor distance included in the graph. It controls physical locality, graph connectivity, and compute cost.
- **Neighbor cap:** Maximum number of neighbors retained per site. Useful for batching but dangerous if it truncates physically relevant neighbors in dense structures.
- **Invariant representation:** Model output for scalar properties is unchanged under translation, rotation, site permutation, and cell representation.
- **Equivariant representation:** Vector or tensor features transform predictably under rotations. Important for forces, tensors, and data-efficient geometry learning.
- **Structure-family split:** A validation split that holds out prototypes, space groups, structure clusters, or crystal families to measure transfer beyond near-duplicates.
- **Leave-one-chemical-system-out split:** A split that holds out all compounds containing a particular element set or chemical system, e.g. all Li-Fe-O materials.
- **Applicability domain:** Region of chemical and structural space covered by training data, measured by composition distance, structure similarity, graph embedding distance, or ensemble uncertainty.

## Key Workflows

### Workflow 1: Build periodic crystal graphs with pymatgen

Graph construction should be deterministic and auditable. Record cutoff, neighbor strategy, structure standardization policy, and whether periodic image vectors are stored.

```python
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
from pymatgen.core import Structure
from pymatgen.analysis.local_env import CrystalNN


@dataclass
class CrystalGraph:
    node_species: list[str]
    frac_coords: np.ndarray
    lattice: np.ndarray
    edge_index: np.ndarray
    edge_image: np.ndarray
    edge_distance: np.ndarray
    metadata: dict


def build_cutoff_crystal_graph(
    structure: Structure,
    cutoff_A: float = 6.0,
    max_neighbors: int | None = 24,
    include_self_edges: bool = False,
) -> CrystalGraph:
    """
    Build a periodic cutoff graph from a pymatgen Structure.

    The cutoff should be tested for connectivity and target sensitivity.
    Typical values are 5-8 A for general crystal property models, but the
    correct value is chemistry- and architecture-dependent. [EXPERT REVIEW NEEDED]
    """
    edge_src, edge_dst, edge_img, edge_dist = [], [], [], []

    for i, site in enumerate(structure):
        neigh = structure.get_neighbors(site, cutoff_A, include_index=True, include_image=True)
        neigh = sorted(neigh, key=lambda n: float(n.nn_distance))
        if max_neighbors is not None:
            neigh = neigh[:max_neighbors]

        for n in neigh:
            j = int(n.index)
            if not include_self_edges and i == j and np.all(np.asarray(n.image) == 0):
                continue
            edge_src.append(i)
            edge_dst.append(j)
            edge_img.append([int(x) for x in n.image])
            edge_dist.append(float(n.nn_distance))

    graph = CrystalGraph(
        node_species=[str(site.specie) for site in structure],
        frac_coords=np.asarray(structure.frac_coords, dtype=float),
        lattice=np.asarray(structure.lattice.matrix, dtype=float),
        edge_index=np.asarray([edge_src, edge_dst], dtype=np.int64),
        edge_image=np.asarray(edge_img, dtype=np.int64),
        edge_distance=np.asarray(edge_dist, dtype=float),
        metadata={
            "formula": structure.composition.reduced_formula,
            "n_sites": len(structure),
            "cutoff_A": cutoff_A,
            "max_neighbors": max_neighbors,
            "volume_A3": float(structure.volume),
        },
    )
    return graph


def audit_graph_connectivity(graph: CrystalGraph) -> dict:
    """Report graph connectivity diagnostics before training."""
    n = len(graph.node_species)
    degrees = np.bincount(graph.edge_index[0], minlength=n) if graph.edge_index.size else np.zeros(n, dtype=int)
    report = {
        "n_sites": n,
        "n_edges": int(graph.edge_index.shape[1]),
        "min_degree": int(degrees.min()) if len(degrees) else 0,
        "median_degree": float(np.median(degrees)) if len(degrees) else 0.0,
        "max_degree": int(degrees.max()) if len(degrees) else 0,
        "min_distance_A": float(np.min(graph.edge_distance)) if len(graph.edge_distance) else None,
        "max_distance_A": float(np.max(graph.edge_distance)) if len(graph.edge_distance) else None,
    }
    if report["min_degree"] == 0:
        print("WARNING: at least one site has zero neighbors; cutoff is too small or structure is malformed")
    if report["max_degree"] == graph.metadata.get("max_neighbors"):
        print("WARNING: neighbor cap reached; physically relevant neighbors may be truncated")
    return report


def write_graph_manifest(structure_file: str, graph: CrystalGraph, output_json: str) -> None:
    Path(output_json).write_text(json.dumps({
        "structure_file": structure_file,
        "metadata": graph.metadata,
        "connectivity": audit_graph_connectivity(graph),
        "feature_policy": {
            "node_features": "species embeddings plus optional elemental properties",
            "edge_features": "distance plus periodic image vector",
            "lattice_features": "stored separately for reproducibility",
        },
    }, indent=2))
```

### Workflow 2: Convert crystal graphs to PyTorch Geometric data

This minimal conversion preserves periodic image and lattice information. Production models may use library-specific graph converters, but the same invariants should be checked.

```python
import torch
from torch_geometric.data import Data
from pymatgen.core import Element


def element_feature_vector(symbol: str) -> list[float]:
    el = Element(symbol)
    return [
        float(el.Z),
        float(el.X) if el.X is not None else 0.0,
        float(el.row),
        float(el.group) if el.group is not None else 0.0,
        float(el.atomic_radius or 0.0),
    ]


def radial_basis(distances: np.ndarray, r_max: float, n_basis: int = 64) -> np.ndarray:
    centers = np.linspace(0.0, r_max, n_basis)
    width = (centers[1] - centers[0]) if n_basis > 1 else r_max
    return np.exp(-((distances[:, None] - centers[None, :]) ** 2) / max(width, 1e-6) ** 2)


def crystal_graph_to_pyg(
    graph: CrystalGraph,
    target: float | None = None,
    n_rbf: int = 64,
) -> Data:
    x = torch.tensor(
        [element_feature_vector(sym) for sym in graph.node_species],
        dtype=torch.float32,
    )
    edge_index = torch.tensor(graph.edge_index, dtype=torch.long)
    edge_attr = torch.tensor(
        radial_basis(graph.edge_distance, r_max=graph.metadata["cutoff_A"], n_basis=n_rbf),
        dtype=torch.float32,
    )
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.edge_distance = torch.tensor(graph.edge_distance, dtype=torch.float32)
    data.edge_image = torch.tensor(graph.edge_image, dtype=torch.long)
    data.frac_coords = torch.tensor(graph.frac_coords, dtype=torch.float32)
    data.lattice = torch.tensor(graph.lattice, dtype=torch.float32).view(1, 3, 3)
    data.formula = graph.metadata["formula"]
    data.n_sites = graph.metadata["n_sites"]
    if target is not None:
        data.y = torch.tensor([float(target)], dtype=torch.float32)
    return data


def verify_cell_representation_consistency(structures: list[Structure], model_predict) -> None:
    """
    Compare predictions for primitive, conventional, and 2x supercell forms.

    model_predict should return an intensive scalar, e.g. formation energy per atom.
    """
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    for i, s in enumerate(structures):
        sga = SpacegroupAnalyzer(s, symprec=1e-3)
        variants = {
            "input": s,
            "primitive": sga.get_primitive_standard_structure(),
            "conventional": sga.get_conventional_standard_structure(),
            "supercell_2x": s * [2, 1, 1],
        }
        preds = {name: float(model_predict(v)) for name, v in variants.items()}
        spread = max(preds.values()) - min(preds.values())
        print(f"{i}: representation spread = {spread:.5f}; {preds}")
        if spread > 0.02:
            print("WARNING: prediction depends on cell representation; check pooling and graph construction")
```

### Workflow 3: Train and evaluate with chemistry-aware splits

Use composition or structure-family splits to match deployment. Random splits are useful as an easy interpolation baseline, not as the only result.

```python
import json
from pathlib import Path
import numpy as np
import pandas as pd
from pymatgen.core import Composition
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.metrics import mean_absolute_error, r2_score


def chemical_system(formula: str) -> str:
    comp = Composition(formula)
    return "-".join(sorted(str(el) for el in comp.elements))


def add_split_groups(df: pd.DataFrame, formula_col: str = "formula") -> pd.DataFrame:
    out = df.copy()
    out["chemical_system"] = out[formula_col].map(chemical_system)
    out["anonymous_formula"] = out[formula_col].map(lambda f: Composition(f).anonymized_formula)
    if "prototype" not in out.columns:
        out["prototype"] = out["anonymous_formula"]
    return out


def summarize_split(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    train_systems = set(train_df["chemical_system"])
    test_systems = set(test_df["chemical_system"])
    train_elements = set().union(*[set(s.split("-")) for s in train_systems]) if train_systems else set()
    test_elements = set().union(*[set(s.split("-")) for s in test_systems]) if test_systems else set()
    return {
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "n_train_systems": int(len(train_systems)),
        "n_test_systems": int(len(test_systems)),
        "overlap_chemical_systems": sorted(train_systems & test_systems),
        "new_test_elements": sorted(test_elements - train_elements),
    }


def evaluate_grouped_predictions(
    df: pd.DataFrame,
    pred_col: str,
    target_col: str,
    group_col: str,
    output_json: str,
) -> dict:
    rows = []
    for group, g in df.groupby(group_col):
        rows.append({
            "group": str(group),
            "n": int(len(g)),
            "mae": float(mean_absolute_error(g[target_col], g[pred_col])),
            "r2": float(r2_score(g[target_col], g[pred_col])) if len(g) > 1 else None,
        })
    report = {
        "global_mae": float(mean_absolute_error(df[target_col], df[pred_col])),
        "global_r2": float(r2_score(df[target_col], df[pred_col])),
        "by_group": sorted(rows, key=lambda r: r["mae"], reverse=True),
    }
    Path(output_json).write_text(json.dumps(report, indent=2))
    return report


def leave_one_chemical_system_splits(df: pd.DataFrame):
    df = add_split_groups(df)
    splitter = LeaveOneGroupOut()
    groups = df["chemical_system"].to_numpy()
    for fold, (train_idx, test_idx) in enumerate(splitter.split(df, groups=groups), 1):
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]
        yield fold, train_idx, test_idx, summarize_split(train_df, test_df)
```

### Workflow 4: Detect duplicates, polymorph leakage, and target inconsistency

Run these audits before training. A powerful GNN cannot fix inconsistent labels or duplicated structures split across folds.

```python
from pymatgen.analysis.structure_matcher import StructureMatcher


def find_structure_duplicates(
    structures: list[Structure],
    ids: list[str],
    ltol: float = 0.2,
    stol: float = 0.3,
    angle_tol: float = 5.0,
) -> list[dict]:
    matcher = StructureMatcher(ltol=ltol, stol=stol, angle_tol=angle_tol)
    duplicates = []
    assigned = set()
    for i, s_i in enumerate(structures):
        if i in assigned:
            continue
        group = [ids[i]]
        for j in range(i + 1, len(structures)):
            if j in assigned:
                continue
            if matcher.fit(s_i, structures[j]):
                group.append(ids[j])
                assigned.add(j)
        if len(group) > 1:
            duplicates.append({"representative": ids[i], "duplicates": group})
    print(f"Duplicate groups: {len(duplicates)}")
    return duplicates


def audit_polymorph_targets(df: pd.DataFrame, formula_col: str, target_col: str, id_col: str) -> pd.DataFrame:
    """
    Report formulas with multiple structures and inconsistent target spread.
    Large spread may be physical polymorphism or mixed calculation settings.
    """
    rows = []
    for formula, g in df.groupby(formula_col):
        if len(g) < 2:
            continue
        rows.append({
            "formula": formula,
            "n_entries": int(len(g)),
            "target_min": float(g[target_col].min()),
            "target_max": float(g[target_col].max()),
            "target_range": float(g[target_col].max() - g[target_col].min()),
            "ids": list(g[id_col].astype(str)),
        })
    out = pd.DataFrame(rows).sort_values("target_range", ascending=False)
    print(out.head(20))
    return out


def check_target_compatibility(df: pd.DataFrame) -> None:
    required = ["functional", "pseudopotential", "correction_scheme", "database_snapshot"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"WARNING: missing target-provenance columns: {missing}")
        return
    for col in required:
        values = sorted(df[col].dropna().astype(str).unique())
        print(f"{col}: {values[:10]}{'...' if len(values) > 10 else ''}")
        if len(values) > 1:
            print(f"WARNING: multiple {col} values; target labels may be incompatible")
```

### Workflow 5: Applicability-domain and uncertainty diagnostics

For screening, a prediction should carry a domain label. Use composition distance, embedding distance, ensemble disagreement, and calibration checks together.

```python
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


def composition_fraction_matrix(formulas: list[str]) -> tuple[np.ndarray, list[str]]:
    comps = [Composition(f).fractional_composition for f in formulas]
    elements = sorted({str(el) for comp in comps for el in comp.elements})
    X = np.zeros((len(comps), len(elements)), dtype=float)
    for i, comp in enumerate(comps):
        for j, el in enumerate(elements):
            X[i, j] = comp.get_atomic_fraction(el)
    return X, elements


def add_embedding_applicability_domain(
    train_embeddings: np.ndarray,
    eval_embeddings: np.ndarray,
    eval_df: pd.DataFrame,
    output_csv: str,
    threshold_quantile: float = 0.95,
) -> pd.DataFrame:
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_embeddings)
    X_eval = scaler.transform(eval_embeddings)

    nn_train = NearestNeighbors(n_neighbors=2).fit(X_train)
    train_dists, _ = nn_train.kneighbors(X_train)
    threshold = float(np.quantile(train_dists[:, 1], threshold_quantile))

    nn = NearestNeighbors(n_neighbors=1).fit(X_train)
    eval_dists, eval_idx = nn.kneighbors(X_eval)
    out = eval_df.copy()
    out["nearest_train_embedding_distance"] = eval_dists[:, 0]
    out["nearest_train_index"] = eval_idx[:, 0]
    out["coverage_class"] = np.where(eval_dists[:, 0] <= threshold, "in_domain", "out_of_domain")
    out.to_csv(output_csv, index=False)
    print(f"Embedding-domain threshold: {threshold:.4f}")
    print(out["coverage_class"].value_counts())
    return out


def summarize_ensemble_uncertainty(predictions: np.ndarray, y_true: np.ndarray | None = None) -> pd.DataFrame:
    """
    predictions shape: (n_models, n_samples). Use out-of-fold or external-test
    predictions for calibration, not training-set predictions.
    """
    mean = predictions.mean(axis=0)
    std = predictions.std(axis=0)
    out = pd.DataFrame({"y_pred": mean, "y_std": std})
    if y_true is not None:
        out["y_true"] = y_true
        out["abs_error"] = np.abs(mean - y_true)
        corr = out["abs_error"].corr(out["y_std"])
        print(f"Correlation between ensemble std and abs error: {corr:.3f}")
        if corr < 0.5:
            print("WARNING: ensemble uncertainty is weakly correlated with error")
    return out
```

### Workflow 6: Compare against descriptor and pretrained-embedding baselines

Graph models should be compared against simpler or pretrained representations under identical splits.

```python
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error


def baseline_regressors() -> dict:
    return {
        "ridge": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("ridge", Ridge(alpha=1.0)),
        ]),
        "random_forest": RandomForestRegressor(
            n_estimators=500,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),
    }


def compare_tabular_baselines(
    X: np.ndarray,
    y: np.ndarray,
    train_idx,
    test_idx,
    output_csv: str,
) -> pd.DataFrame:
    rows = []
    for name, model in baseline_regressors().items():
        model.fit(X[train_idx], y[train_idx])
        pred = model.predict(X[test_idx])
        rows.append({
            "model": name,
            "mae": float(mean_absolute_error(y[test_idx], pred)),
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
        })
    out = pd.DataFrame(rows).sort_values("mae")
    out.to_csv(output_csv, index=False)
    return out


def extract_chgnet_embeddings(structures: list[Structure]) -> np.ndarray:
    """
    Extract frozen CHGNet graph embeddings for downstream small-data models.

    API details can change across CHGNet versions; pin the version and validate
    embedding shape before using this in production.
    [EXPERT REVIEW NEEDED]
    """
    from chgnet.model import CHGNet
    model = CHGNet.load()
    embeddings = []
    for s in structures:
        pred = model.predict_structure(s, return_atom_feas=True)
        atom_fea = np.asarray(pred["atom_fea"])
        embeddings.append(atom_fea.mean(axis=0))
    return np.asarray(embeddings)
```

## Common Failure Modes

| Failure | Symptom | Likely Cause | Diagnostic | Fix |
|---------|---------|--------------|------------|-----|
| Duplicate structures across splits | Near-perfect validation metrics; poor external transfer | Same structure appears under multiple database IDs or cell choices | Run `StructureMatcher` before splitting; inspect duplicate groups | Deduplicate or group duplicates into the same fold |
| Random split overestimates generalization | Random MAE is low; leave-one-system MAE is much higher | Same chemistry and prototypes appear in train and test | Compare random, prototype, composition, and leave-one-chemical-system splits | Report split-specific claims; use deployment-matched split |
| Wrong primitive/conventional cell handling | Predictions differ for equivalent cells or supercells | Pooling, normalization, or graph construction depends on cell size | Run representation consistency check | Normalize intensive targets; test primitive/conventional/supercell invariance |
| Lost oxidation states or site properties | Oxidation-dependent features become zero or inconsistent | Format conversion dropped species metadata, magmoms, or charge labels | Compare pymatgen site properties before and after conversion | Preserve sidecar metadata or avoid oxidation-dependent node features |
| Inconsistent target corrections | Model learns database correction artifacts | Formation energies mix raw and corrected values or database snapshots | Print correction scheme, functional, and snapshot counts | Use one compatible target definition; store correction metadata |
| Polymorphism ignored | Same formula has multiple labels; model averages incompatible structures | Polymorphs treated as duplicate compositions or split across folds | Audit target spread by formula and structure match | Keep structures distinct; group polymorph families for validation |
| Disorder mishandled as ordered | Model sees arbitrary ordered approximants as exact crystals | Partial occupancies or occupational disorder not represented | Check `structure.is_ordered` and site occupancies | Enumerate ordered structures or use disorder-aware descriptors |
| Poor extrapolation to unseen chemistries | Confident predictions for new elements are wrong | Training set lacks element or chemical-system support | Leave-one-element/system validation; embedding-distance domain labels | Restrict domain or acquire labels in the new chemistry |
| Cutoff graph disconnected | Model misses isolated sites or open-framework interactions | Cutoff too small or neighbor cap too strict | Connectivity audit: min degree, neighbor cap hits | Increase cutoff; remove cap; inspect edge distances |
| Database bias mistaken for physics | Learned trends reproduce which materials were computed, not physical laws | Training set overrepresents stable oxides or common prototypes | Plot composition/prototype/source distributions | Rebalance, stratify, or state database-domain limitation |
| Magnetic structures flattened | Magnetic-property predictions fail or vary by source | Magnetic order, spin state, or magmom metadata absent | Check site properties and calculation metadata | Include magnetic descriptors or separate magnetic datasets |

## Best Practices

- **Define the graph construction contract.** Record cutoff, neighbor cap, image vectors, radial basis, angle features, lattice features, and standardization policy.
- **Validate cell-representation invariance.** Predictions for primitive, conventional, and equivalent supercell representations should agree for intensive scalar targets.
- **Deduplicate before splitting.** Use `StructureMatcher`, formula checks, database IDs, and relaxed-structure matching before train/test assignment.
- **Use split protocols that match the scientific claim.** Random split for interpolation, composition split for new compositions, prototype split for new structures, and leave-one-chemical-system split for stronger extrapolation.
- **Always compare against fixed-descriptor baselines.** Magpie, SOAP, Voronoi, random forest, ridge, GP, and pretrained embeddings are essential controls.
- **Preserve target provenance.** Formation energy, band gap, elastic, phonon, and stability labels must carry functional, correction, convergence, and database snapshot metadata.
- **Audit per-group errors.** Report errors by chemical system, element, prototype, space group, structure source, stability class, and coverage class.
- **Treat uncertainty as a calibrated artifact.** Ensemble or dropout variance is not enough; validate reliability on held-out structures.
- **Handle disorder and magnetism explicitly.** Do not silently coerce partial occupancies or magnetic states into ordinary ordered nonmagnetic structures.
- **Visualize coverage before screening.** Embedding maps, composition-simplex coverage, and nearest-neighbor distances reveal where predictions are extrapolative.

## Integration with Other Skills

- **[molecular-property-gnn](../molecular-property-gnn/SKILL.md):** Molecular and crystal GNNs share message passing, uncertainty, and split-validation concepts, but crystal GNNs must handle periodic images, lattice geometry, and cell representation invariance.
- **[materials-featurization](../../scientific-data/materials-featurization/SKILL.md):** Fixed descriptors such as Magpie, SOAP, Voronoi, and Coulomb/sine matrices provide baselines and leakage diagnostics for graph models.
- **[pymatgen-analysis](../../scientific-data/pymatgen-analysis/SKILL.md):** pymatgen provides the periodic structures, standardization, `StructureMatcher` deduplication, symmetry analysis, and site-property metadata used before graph construction.
- **[materials-databases-access](../../scientific-data/materials-databases-access/SKILL.md):** Database queries supply structures, entries, task IDs, correction schemes, and provenance needed to build reliable crystal GNN datasets.
- **[scientific-visualization](../../scientific-data/scientific-visualization/SKILL.md):** Parity plots, residual maps, embedding projections, element-wise error plots, and structure visualizations should follow reproducible plotting conventions.
- **[reproducibility-workflow](../../scientific-data/reproducibility-workflow/SKILL.md):** Graph construction metadata, split files, model checkpoints, prediction tables, and package versions must be archived for auditability.
- **[small-data-ml](../../scientific-ml/small-data-ml/SKILL.md):** Small crystal datasets need nested CV, grouped splits, baselines, confidence intervals, and careful reporting before neural models are justified.
- **[surrogate-validation](../../surrogate-active-learning/surrogate-validation/SKILL.md):** Applicability-domain checks, calibration, learning curves, group errors, and closed-loop screening validation follow the surrogate validation framework.
- **[gaussian-process-materials](../../surrogate-active-learning/gaussian-process-materials/SKILL.md):** GPs on fixed descriptors or pretrained graph embeddings are strong baselines and uncertainty models for small crystal datasets.
- **[high-throughput-dft](../../electronic-structure/high-throughput-dft/SKILL.md):** Crystal GNNs often triage high-throughput DFT queues; validation must prove they are reliable enough for candidate prioritization.
- **[dft-convergence](../../electronic-structure/dft-convergence/SKILL.md):** DFT targets used to train or validate crystal GNNs must share convergence settings and compatible reference definitions.
- **[molecular-featurization](../../cheminformatics/molecular-featurization/SKILL.md):** Molecular featurization is relevant for molecular crystals, organic salts, and hybrid datasets where molecules and periodic structures are connected. This is an intentional forward reference until that skill is added.
- **[molecular-datasets](../../cheminformatics/molecular-datasets/SKILL.md):** Molecular dataset curation and assay metadata matter when crystal datasets include molecular solids, salts, or organic semiconductors. This is an intentional forward reference until that skill is added.

## Key References

1. Xie, T., & Grossman, J. C. (2018). *Crystal graph convolutional neural networks for an accurate and interpretable prediction of material properties.* Physical Review Letters, 120, 145301. https://doi.org/10.1103/PhysRevLett.120.145301
2. Chen, C., Ye, W., Zuo, Y., Zheng, C., & Ong, S. P. (2019). *Graph networks as a universal machine learning framework for molecules and crystals.* Chemistry of Materials, 31, 3564-3572. https://doi.org/10.1021/acs.chemmater.9b01294
3. Choudhary, K., & DeCost, B. (2021). *Atomistic line graph neural network for improved materials property predictions.* npj Computational Materials, 7, 185. https://doi.org/10.1038/s41524-021-00650-1
4. Chen, C., & Ong, S. P. (2022). *A universal graph deep learning interatomic potential for the periodic table.* Nature Computational Science, 2, 718-728. https://doi.org/10.1038/s43588-022-00349-3
5. Deng, B., et al. (2023). *CHGNet as a pretrained universal neural network potential for charge-informed atomistic modelling.* Nature Machine Intelligence, 5, 1031-1041. https://doi.org/10.1038/s42256-023-00716-3
6. Dunn, A., et al. (2020). *Benchmarking materials property prediction methods: The Matbench test set and Automatminer reference algorithm.* npj Computational Materials, 6, 138. https://doi.org/10.1038/s41524-020-00406-3
