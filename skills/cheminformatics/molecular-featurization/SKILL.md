---
name: molecular-featurization
domain: cheminformatics
maturity: draft
tools:
  - RDKit
  - pandas
  - numpy
  - scikit-learn
  - mordred
  - matplotlib
tested_versions:
  RDKit: "2023.9.4"
  pandas: "2.2.0"
  numpy: "1.26.4"
  scikit-learn: "1.4.0"
  mordred: "1.2.0"
requires_gpu: false
estimated_runtime: minutes-to-hours
---

# Molecular Featurization

## Description

This skill covers molecular featurization for chemistry and molecular machine learning: parsing SMILES, SMARTS, InChI, SDF, and MOL2-like records; RDKit sanitization and standardization; fingerprints such as Morgan/ECFP, MACCS, atom-pair, torsion, pharmacophore, and topological fingerprints; physicochemical descriptors; 2D versus 3D descriptors; conformer-dependent features; descriptor scaling, missing values, and leakage-aware feature selection. Invoke this skill before training molecular property models, molecular GNNs, small-data surrogates, or screening workflows that depend on chemically meaningful molecular inputs.

## Domain Context

Molecular featurization converts a chemical representation into numerical inputs for machine learning. That conversion is not neutral. A SMILES string encodes molecular graph connectivity and stereochemistry if specified, but not a conformer ensemble, protonation microstate, solvent, assay conditions, or experimental uncertainty. A 3D SDF file includes coordinates, but those coordinates may be generated, force-field-minimized, crystallographic, docked, or experimentally derived. Different coordinate sources can produce different descriptors for the same molecule.

The first scientific decision is molecular identity. Salts, mixtures, counterions, tautomers, zwitterions, protonation states, isotopes, stereoisomers, and resonance forms may all map to different graphs or descriptors. Standardization choices must match the target property. For a pH-dependent solubility or binding assay, neutralizing every molecule may remove signal; for a gas-phase quantum-chemistry target, keeping a counterion may be wrong. [EXPERT REVIEW NEEDED]

Fingerprints and descriptors encode different assumptions. Morgan/ECFP fingerprints capture circular atom neighborhoods and are strong baselines for QSAR and small molecular datasets. MACCS keys encode a fixed set of substructure patterns. Atom-pair and topological torsion fingerprints encode longer graph-distance relationships. Physicochemical descriptors such as logP, TPSA, HBD/HBA counts, rotatable bonds, molecular weight, and formal charge are interpretable but lower-dimensional. 3D descriptors capture shape, conformer geometry, pharmacophores, Coulomb interactions, or surface properties, but depend strongly on conformer generation and alignment.

Validation must account for correlated molecules. Random splits often place analogs, stereoisomers, tautomers, or near-duplicate structures on both sides of the split. That can make descriptor pipelines look more predictive than they are for new chemotypes. Scaffold, time, assay, series, or chemistry-aware splits are usually needed when the deployment goal is screening beyond a known analog series.

## When to Use This Skill

- Building feature matrices for molecular property prediction, QSAR, molecular screening, or small-data molecular surrogates.
- Standardizing molecular identifiers before deduplication, splitting, descriptor generation, or model training.
- Comparing fingerprint, physicochemical, 3D conformer, and graph-based representations for a molecular dataset.
- Auditing molecular ML data for invalid structures, lost stereochemistry, inconsistent protonation, tautomer leakage, salts, mixtures, or duplicate molecules.
- Preparing fixed-descriptor baselines for molecular GNNs or Gaussian-process models.
- Generating descriptor provenance metadata for reproducible molecular screening workflows.

## When Not to Use This Skill

- **When the target is determined by a full periodic structure.** Molecular descriptors are not enough for molecular crystals, salts in a lattice, surfaces, or interfaces where packing controls the property.
- **When assay metadata define the label.** Toxicity, binding affinity, and activity values depend on protocol, target, cell line, pH, and units; featurization alone cannot make mixed assays compatible.
- **When the molecule is not chemically specified.** Generic R-groups, polymers with distributional composition, mixtures, unresolved stereochemistry, or reaction recipes require curation before descriptor generation.
- **When 3D geometry is essential but conformers are unreliable.** Do not use single-conformer descriptors for flexible molecules without conformer protocol validation and sensitivity checks.
- **When descriptors are computed from information unavailable at prediction time.** Experimental labels, docked poses, DFT results, or assay-derived annotations can leak target information.

## Core Concepts

- **SMILES:** Compact line notation for molecular graph connectivity. Isomeric SMILES can include stereochemistry; non-isomeric canonical SMILES may collapse distinct stereoisomers.
- **SMARTS:** Substructure query language used for pattern matching and chemical filters. SMARTS features are useful only if the pattern library is documented.
- **InChI and InChIKey:** Standardized molecular identifiers useful for deduplication. InChIKey layers can collapse or distinguish stereochemistry, isotopes, and protonation depending on how they are used.
- **SDF and MOL files:** Structure-data formats that can store atoms, bonds, coordinates, charges, and property fields. Parsing quality depends on valence, bond-order, and sanitization consistency.
- **MOL2:** Common docking and force-field format with atom types and charges. RDKit support is more limited than for SMILES/SDF; validate atom typing and bond perception after parsing.
- **Sanitization:** RDKit process that checks valence, aromaticity, conjugation, ring information, and implicit hydrogens. Sanitization failures are data-quality signals.
- **Standardization:** Project-specific conversion of molecular records into a canonical modeling form: salt stripping, charge normalization, tautomer selection, isotope handling, and stereochemistry policy.
- **Fingerprint:** Fixed-length or sparse bit/count vector representing substructures. Common families include Morgan/ECFP, MACCS, atom-pair, topological torsion, pharmacophore, and pattern fingerprints.
- **Physicochemical descriptor:** Interpretable scalar feature such as molecular weight, logP, TPSA, rotatable bonds, HBD/HBA counts, ring counts, formal charge, or fraction sp3.
- **2D descriptor:** Feature computed from graph topology and atom/bond attributes, independent of 3D coordinates.
- **3D descriptor:** Feature computed from conformer coordinates, such as shape, volume, distance distributions, Coulomb terms, pharmacophore geometry, or 3D autocorrelation.
- **Conformer ensemble:** Multiple plausible molecular geometries. Flexible molecules often require aggregation over an ensemble rather than a single conformer.
- **Descriptor leakage:** A feature that encodes label or test-set information, often through preprocessing fit on all data, post-assay annotations, or target-derived calculations.

## Key Workflows

### Workflow 1: Parse, sanitize, standardize, and deduplicate molecules

Build a curation table before computing descriptors. Keep invalid records instead of silently dropping them so failure rates can be audited.

```python
from dataclasses import dataclass
from collections import Counter
from pathlib import Path
import json
import pandas as pd
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize


@dataclass
class ParsedMolecule:
    source_id: str
    input_text: str
    mol: Chem.Mol | None
    canonical_smiles: str | None
    inchikey: str | None
    status: str
    notes: list[str]


def parse_molecule_text(text: str, fmt: str = "smiles") -> Chem.Mol | None:
    """Parse one molecular record with conservative sanitization."""
    if pd.isna(text) or not str(text).strip():
        return None
    text = str(text).strip()
    if fmt == "smiles":
        return Chem.MolFromSmiles(text, sanitize=True)
    if fmt == "inchi":
        return Chem.MolFromInchi(text, sanitize=True)
    if fmt == "molblock":
        return Chem.MolFromMolBlock(text, sanitize=True, removeHs=False)
    raise ValueError(f"Unsupported format: {fmt}")


def standardize_rdkit_mol(
    mol: Chem.Mol,
    largest_fragment: bool = True,
    normalize: bool = True,
    reionize: bool = False,
    uncharge: bool = False,
) -> tuple[Chem.Mol | None, list[str]]:
    """
    Apply an explicit project standardization policy.

    These choices are target-dependent. For pH-specific assays, uncharging or
    reionizing may remove real signal. [EXPERT REVIEW NEEDED]
    """
    notes = []
    try:
        out = Chem.Mol(mol)
        if largest_fragment:
            out = rdMolStandardize.LargestFragmentChooser().choose(out)
            notes.append("largest_fragment")
        if normalize:
            out = rdMolStandardize.Normalizer().normalize(out)
            notes.append("normalize")
        if reionize:
            out = rdMolStandardize.Reionizer().reionize(out)
            notes.append("reionize")
        if uncharge:
            out = rdMolStandardize.Uncharger().uncharge(out)
            notes.append("uncharge")
        Chem.SanitizeMol(out)
        return out, notes
    except Exception as exc:
        return None, notes + [f"standardization_failed:{type(exc).__name__}"]


def build_molecule_manifest(
    df: pd.DataFrame,
    id_col: str,
    molecule_col: str,
    output_csv: str,
    fmt: str = "smiles",
) -> pd.DataFrame:
    rows = []
    parsed_records = []
    for _, row in df.iterrows():
        source_id = str(row[id_col])
        raw = row[molecule_col]
        notes = []
        mol = parse_molecule_text(raw, fmt=fmt)
        if mol is None:
            rows.append({
                "source_id": source_id,
                "input": raw,
                "status": "parse_failed",
                "canonical_smiles": None,
                "inchikey": None,
                "notes": "parse_failed",
            })
            continue

        std_mol, std_notes = standardize_rdkit_mol(mol)
        notes.extend(std_notes)
        if std_mol is None:
            rows.append({
                "source_id": source_id,
                "input": raw,
                "status": "standardization_failed",
                "canonical_smiles": None,
                "inchikey": None,
                "notes": ";".join(notes),
            })
            continue

        smiles = Chem.MolToSmiles(std_mol, isomericSmiles=True)
        inchikey = Chem.MolToInchiKey(std_mol)
        rows.append({
            "source_id": source_id,
            "input": raw,
            "status": "ok",
            "canonical_smiles": smiles,
            "inchikey": inchikey,
            "n_atoms": std_mol.GetNumAtoms(),
            "formal_charge": Chem.GetFormalCharge(std_mol),
            "notes": ";".join(notes),
        })
        parsed_records.append(ParsedMolecule(source_id, str(raw), std_mol, smiles, inchikey, "ok", notes))

    out = pd.DataFrame(rows)
    out.to_csv(output_csv, index=False)
    print(out["status"].value_counts(dropna=False))
    duplicate_counts = out.loc[out["status"] == "ok", "inchikey"].value_counts()
    n_duplicate_rows = int((duplicate_counts[duplicate_counts > 1] - 1).sum())
    if n_duplicate_rows:
        print(f"WARNING: {n_duplicate_rows} duplicate molecule rows by InChIKey")
    return out
```

### Workflow 2: Compute fingerprint families with versioned parameters

Fingerprints are strong baselines. Use count fingerprints when substructure multiplicity matters; use bit fingerprints when dimensionality and sparsity are more important.

```python
from pathlib import Path
import json
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, MACCSkeys, rdMolDescriptors
from rdkit.Chem import rdFingerprintGenerator


def fp_to_numpy(fp, n_bits: int) -> np.ndarray:
    arr = np.zeros((n_bits,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def compute_fingerprint_matrix(
    smiles: list[str],
    fp_type: str,
    n_bits: int = 2048,
    radius: int = 2,
    use_counts: bool = False,
) -> tuple[np.ndarray, dict]:
    """
    Compute a fixed-size molecular fingerprint matrix.

    fp_type: "morgan", "maccs", "atom_pair", "torsion", or "pattern".
    """
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    if any(m is None for m in mols):
        raise ValueError("Invalid SMILES present; parse and filter before fingerprinting")

    rows = []
    for mol in mols:
        if fp_type == "morgan":
            generator = rdFingerprintGenerator.GetMorganGenerator(
                radius=radius,
                fpSize=n_bits,
                countSimulation=use_counts,
            )
            fp = generator.GetFingerprint(mol)
            rows.append(fp_to_numpy(fp, n_bits))
        elif fp_type == "maccs":
            fp = MACCSkeys.GenMACCSKeys(mol)
            rows.append(fp_to_numpy(fp, 167))
        elif fp_type == "atom_pair":
            fp = rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=n_bits)
            rows.append(fp_to_numpy(fp, n_bits))
        elif fp_type == "torsion":
            fp = rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(mol, nBits=n_bits)
            rows.append(fp_to_numpy(fp, n_bits))
        elif fp_type == "pattern":
            fp = Chem.PatternFingerprint(mol, fpSize=n_bits)
            rows.append(fp_to_numpy(fp, n_bits))
        else:
            raise ValueError(f"Unknown fingerprint type: {fp_type}")

    X = np.vstack(rows)
    metadata = {
        "fp_type": fp_type,
        "n_bits": int(X.shape[1]),
        "radius": radius if fp_type == "morgan" else None,
        "use_counts": use_counts,
        "rdkit_note": "Record RDKit version and standardization policy with this matrix.",
    }
    return X, metadata


def write_fingerprint_dataset(
    manifest_csv: str,
    output_npz: str,
    metadata_json: str,
    fp_type: str = "morgan",
    n_bits: int = 2048,
    radius: int = 2,
) -> None:
    manifest = pd.read_csv(manifest_csv)
    ok = manifest[manifest["status"] == "ok"].copy()
    X, metadata = compute_fingerprint_matrix(
        ok["canonical_smiles"].tolist(),
        fp_type=fp_type,
        n_bits=n_bits,
        radius=radius,
    )
    np.savez_compressed(output_npz, X=X, source_id=ok["source_id"].to_numpy(), inchikey=ok["inchikey"].to_numpy())
    metadata["n_molecules"] = int(len(ok))
    Path(metadata_json).write_text(json.dumps(metadata, indent=2))
    print(f"Wrote {X.shape} fingerprint matrix to {output_npz}")
```

### Workflow 3: Physicochemical descriptor table with missing-value audit

Physicochemical descriptors are interpretable, fast, and excellent baselines for small datasets. Treat NaN and extreme values as curation signals.

```python
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, Crippen, rdMolDescriptors


DESCRIPTOR_FUNCTIONS = {
    "mol_wt": Descriptors.MolWt,
    "exact_mol_wt": Descriptors.ExactMolWt,
    "logp": Crippen.MolLogP,
    "tpsa": rdMolDescriptors.CalcTPSA,
    "hbd": Lipinski.NumHDonors,
    "hba": Lipinski.NumHAcceptors,
    "rotatable_bonds": Lipinski.NumRotatableBonds,
    "ring_count": rdMolDescriptors.CalcNumRings,
    "aromatic_rings": rdMolDescriptors.CalcNumAromaticRings,
    "fraction_csp3": rdMolDescriptors.CalcFractionCSP3,
    "formal_charge": Chem.GetFormalCharge,
    "heavy_atoms": Descriptors.HeavyAtomCount,
}


def compute_physchem_descriptors(smiles: list[str]) -> pd.DataFrame:
    rows = []
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        row = {"row": i, "canonical_smiles": smi}
        if mol is None:
            row["status"] = "parse_failed"
            rows.append(row)
            continue
        row["status"] = "ok"
        for name, fn in DESCRIPTOR_FUNCTIONS.items():
            try:
                row[name] = float(fn(mol))
            except Exception:
                row[name] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def audit_descriptor_table(df: pd.DataFrame, feature_cols: list[str]) -> None:
    print(f"Descriptor table: {len(df)} rows x {len(feature_cols)} features")
    nan_frac = df[feature_cols].isna().mean().sort_values(ascending=False)
    high_nan = nan_frac[nan_frac > 0.01]
    if len(high_nan):
        print("WARNING: descriptors with >1% missing values:")
        print(high_nan.head(20))

    zero_var = [c for c in feature_cols if df[c].nunique(dropna=True) <= 1]
    if zero_var:
        print(f"WARNING: zero-variance descriptors: {zero_var}")

    checks = {
        "mol_wt": (0, 3000),
        "logp": (-20, 20),
        "tpsa": (0, 1000),
        "rotatable_bonds": (0, 200),
        "formal_charge": (-10, 10),
    }
    for col, (lo, hi) in checks.items():
        if col in df:
            bad = df[(df[col] < lo) | (df[col] > hi)]
            if len(bad):
                print(f"WARNING: {col} has {len(bad)} out-of-range values")
```

### Workflow 4: Generate conformers and conformer-dependent descriptors

3D descriptors require a conformer policy. Store random seeds, embedding settings, force-field status, and whether descriptors are single-conformer or ensemble-aggregated.

```python
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors


def generate_conformer_ensemble(
    mol: Chem.Mol,
    n_conformers: int = 25,
    prune_rms_thresh: float = 0.5,
    random_seed: int = 13,
    optimize: bool = True,
) -> Chem.Mol | None:
    mol_h = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = random_seed
    params.pruneRmsThresh = prune_rms_thresh
    conf_ids = AllChem.EmbedMultipleConfs(mol_h, numConfs=n_conformers, params=params)
    if len(conf_ids) == 0:
        return None
    if optimize:
        results = AllChem.MMFFOptimizeMoleculeConfs(mol_h, maxIters=500)
        for conf_id, (status, energy) in zip(conf_ids, results):
            mol_h.GetConformer(conf_id).SetIntProp("mmff_status", int(status))
            mol_h.GetConformer(conf_id).SetDoubleProp("mmff_energy", float(energy))
    return mol_h


def compute_3d_descriptor_summary(mol_h: Chem.Mol) -> dict:
    """
    Compute simple conformer-dependent descriptors and summarize across conformers.
    These descriptors are sensitive to conformer generation quality.
    """
    rows = []
    for conf in mol_h.GetConformers():
        cid = conf.GetId()
        rows.append({
            "conf_id": cid,
            "radius_gyration": float(rdMolDescriptors.CalcRadiusOfGyration(mol_h, confId=cid)),
            "asphericity": float(rdMolDescriptors.CalcAsphericity(mol_h, confId=cid)),
            "spherocity": float(rdMolDescriptors.CalcSpherocityIndex(mol_h, confId=cid)),
            "inertial_shape_factor": float(rdMolDescriptors.CalcInertialShapeFactor(mol_h, confId=cid)),
            "mmff_status": conf.GetIntProp("mmff_status") if conf.HasProp("mmff_status") else None,
            "mmff_energy": conf.GetDoubleProp("mmff_energy") if conf.HasProp("mmff_energy") else None,
        })
    df = pd.DataFrame(rows)
    summary = {}
    for col in ["radius_gyration", "asphericity", "spherocity", "inertial_shape_factor"]:
        summary[f"{col}_mean"] = float(df[col].mean())
        summary[f"{col}_std"] = float(df[col].std(ddof=0))
        summary[f"{col}_min"] = float(df[col].min())
    if "mmff_energy" in df and df["mmff_energy"].notna().any():
        summary["mmff_energy_range"] = float(df["mmff_energy"].max() - df["mmff_energy"].min())
    if df["mmff_status"].fillna(0).astype(int).ne(0).any():
        summary["warning"] = "one or more conformers failed MMFF convergence"
    return summary


def conformer_descriptor_table(smiles: list[str], output_csv: str) -> pd.DataFrame:
    rows = []
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            rows.append({"row": i, "canonical_smiles": smi, "status": "parse_failed"})
            continue
        mol_h = generate_conformer_ensemble(mol)
        if mol_h is None:
            rows.append({"row": i, "canonical_smiles": smi, "status": "conformer_failed"})
            continue
        row = {"row": i, "canonical_smiles": smi, "status": "ok"}
        row.update(compute_3d_descriptor_summary(mol_h))
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(output_csv, index=False)
    print(out["status"].value_counts(dropna=False))
    return out
```

### Workflow 5: Leakage-safe preprocessing, scaling, and feature selection

All preprocessing that learns from feature values or targets must be fit inside the training fold. Save the fitted pipeline with the split metadata.

```python
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold, SelectKBest, mutual_info_regression
from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.ensemble import RandomForestRegressor


def build_descriptor_pipeline(model=None, k_best: int | None = None) -> Pipeline:
    steps = [
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("variance", VarianceThreshold(threshold=0.0)),
        ("scale", StandardScaler()),
    ]
    if k_best is not None:
        steps.append(("select", SelectKBest(score_func=mutual_info_regression, k=k_best)))
    if model is None:
        model = RandomForestRegressor(
            n_estimators=500,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )
    steps.append(("model", model))
    return Pipeline(steps)


def evaluate_descriptor_model(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    group_col: str,
    k_best: int | None = None,
) -> pd.DataFrame:
    data = df.dropna(subset=[target_col, group_col]).copy()
    X = data[feature_cols]
    y = data[target_col].to_numpy()
    groups = data[group_col].astype(str).to_numpy()
    cv = GroupKFold(n_splits=min(5, len(set(groups))))
    pipe = build_descriptor_pipeline(k_best=k_best)
    scores = cross_validate(
        pipe,
        X,
        y,
        groups=groups,
        cv=cv,
        scoring=("neg_mean_absolute_error", "r2"),
        return_train_score=True,
        n_jobs=1,
    )
    out = pd.DataFrame(scores)
    out["test_mae"] = -out["test_neg_mean_absolute_error"]
    out["train_mae"] = -out["train_neg_mean_absolute_error"]
    print(out[["test_mae", "train_mae", "test_r2", "train_r2"]].describe())
    return out
```

### Workflow 6: Applicability-domain checks with fingerprints

Before screening new molecules, label whether each candidate is close to the training set in the chosen representation.

```python
import numpy as np
import pandas as pd


def tanimoto_distance_matrix(X_a: np.ndarray, X_b: np.ndarray) -> np.ndarray:
    """Binary fingerprint Tanimoto distance matrix."""
    inter = X_a @ X_b.T
    a_sum = X_a.sum(axis=1, keepdims=True)
    b_sum = X_b.sum(axis=1, keepdims=True).T
    union = a_sum + b_sum - inter
    return 1.0 - inter / np.maximum(union, 1.0)


def label_fingerprint_domain(
    train_smiles: list[str],
    candidate_smiles: list[str],
    output_csv: str,
    distance_threshold: float = 0.65,
) -> pd.DataFrame:
    X_train, meta = compute_fingerprint_matrix(train_smiles, fp_type="morgan", n_bits=2048, radius=2)
    X_cand, _ = compute_fingerprint_matrix(candidate_smiles, fp_type="morgan", n_bits=2048, radius=2)
    distances = tanimoto_distance_matrix(X_cand, X_train)
    nearest = distances.min(axis=1)
    out = pd.DataFrame({
        "canonical_smiles": candidate_smiles,
        "nearest_train_tanimoto_distance": nearest,
        "coverage_class": np.where(nearest <= distance_threshold, "in_domain", "out_of_domain"),
    })
    out.to_csv(output_csv, index=False)
    print(out["coverage_class"].value_counts())
    return out
```

## Common Failure Modes

| Failure | Symptom | Likely Cause | Diagnostic | Fix |
|---------|---------|--------------|------------|-----|
| Invalid SMILES | Rows disappear before modeling; descriptor matrix has fewer rows than labels | Parse failures, unsupported valence, malformed strings, or non-molecular records | Count parse failures and retain source IDs in manifest | Fix source data or exclude with explicit failure report |
| Lost stereochemistry | Enantiomers or diastereomers collapse to identical features | Non-isomeric SMILES, SDF conversion, or canonicalization dropped chiral tags | Compare isomeric and non-isomeric canonical SMILES; count unspecified stereocenters | Preserve isomeric SMILES and chiral descriptors; curate ambiguous stereochemistry |
| Inconsistent protonation | Same compound appears with several charges and descriptors | Mixed pH assumptions, neutralization policy, or database source conventions | Group by parent InChIKey and inspect formal charge distribution | Define pH/protonation policy and record it in metadata |
| Tautomer leakage | Train and test contain tautomer variants of the same compound | Deduplication by exact SMILES misses tautomer relationships | Apply tautomer canonicalization and group by tautomer parent | Keep tautomer families in one split or model tautomers explicitly |
| Conformer artifacts | 3D descriptors unstable across seeds or strongly predictive for wrong reason | Failed embedding, high-energy conformers, or single-conformer bias | Repeat conformer generation with different seeds; inspect energy range | Use ensemble summaries; filter failed conformers; document protocol |
| Descriptor leakage | Validation metrics are implausibly strong | Feature selection, scaling, or imputation fit on all data; descriptor encodes assay or target | Trace all preprocessing and feature provenance | Put preprocessing in sklearn `Pipeline`; remove target-derived features |
| Duplicate molecules under different representations | Same compound appears in train and test with different SMILES, salt, or tautomer | Deduplication did not use InChIKey/standardization | Count duplicate InChIKeys and parent forms before splitting | Collapse or group duplicates before splitting |
| Mixed assay-dependent labels | Model learns assay source rather than molecular property | Different measurement protocols or units combined | Stratify metrics by assay/source/unit and inspect target distributions | Separate assays, harmonize units, or include protocol metadata |
| Missing-value artifacts | Imputed rows cluster or dominate high-error cases | Descriptor failures are not random across chemistry | Plot missingness by scaffold, charge, and source | Drop fragile descriptors or add missing indicators and report failure rates |
| Fingerprint collision or saturation | Many molecules have very similar bit counts; model cannot distinguish analogs | Too few bits, wrong radius, or bit fingerprint loses counts | Compare bit density and nearest-neighbor distances | Increase bit length, use count fingerprints, or add physicochemical descriptors |

## Best Practices

- **Freeze the molecule standardization policy before splitting.** Salt handling, tautomer canonicalization, protonation, stereochemistry, isotopes, and charge treatment define the dataset.
- **Keep parse failures in an audit table.** Silent row drops break label alignment and hide chemistry classes that the featurizer cannot handle.
- **Deduplicate before train/test split.** Use canonical SMILES, InChIKey layers, tautomer parents, and salt-parent identifiers as appropriate.
- **Use descriptor families as baselines.** Morgan fingerprints, MACCS keys, physicochemical descriptors, and random forest/ridge baselines should be compared before adopting a complex model.
- **Fit preprocessing inside CV folds.** Imputation, scaling, feature selection, PCA, and target transformations must be trained on the training fold only.
- **Report descriptor provenance.** RDKit version, standardization policy, fingerprint radius, bit length, count/bit mode, descriptor list, and conformer settings are part of the model.
- **Treat 3D descriptors as protocol-dependent.** Store conformer seeds, embedding algorithm, force field, optimization status, and ensemble aggregation method.
- **Use scaffold, time, assay, or chemistry-aware splits for deployment claims.** Random splits mainly measure interpolation within known analogs.
- **Visualize coverage and failure modes.** Plot missingness, descriptor distributions, fingerprint nearest-neighbor distances, and scaffold-level errors.
- **Keep assay metadata with labels.** A molecular graph alone cannot explain mixed protocols, concentration units, or target variants.

## Integration with Other Skills

- **[molecular-property-gnn](../../geometric-deep-learning/molecular-property-gnn/SKILL.md):** Molecular GNNs consume molecular graphs, atom/bond features, stereochemistry, and conformer policies defined by molecular featurization. Fixed fingerprints and physicochemical descriptors are also required baselines.
- **[molecular-datasets](../molecular-datasets/SKILL.md):** Dataset curation, assay metadata, benchmark split definitions, and source provenance should be prepared upstream. This is an intentional forward reference until that skill is added.
- **[small-data-ml](../../scientific-ml/small-data-ml/SKILL.md):** Molecular datasets often sit in the small-data regime, where nested CV, strong baselines, confidence intervals, and leakage-aware preprocessing dominate model quality.
- **[surrogate-validation](../../surrogate-active-learning/surrogate-validation/SKILL.md):** Applicability-domain labels, residual diagnostics, calibration checks, and leakage-safe splits are downstream validation requirements for descriptor-based surrogates.
- **[materials-featurization](../../scientific-data/materials-featurization/SKILL.md):** Molecular featurization overlaps with materials descriptors when molecules, adsorbates, molecular crystals, or hybrid organic-inorganic systems are part of a materials dataset.
- **[scientific-visualization](../../scientific-data/scientific-visualization/SKILL.md):** Descriptor distributions, PCA/UMAP maps, scaffold error plots, missingness heatmaps, parity plots, and molecule depictions should follow reproducible plotting conventions.
- **[reproducibility-workflow](../../scientific-data/reproducibility-workflow/SKILL.md):** Standardization policies, descriptor metadata, split files, feature matrices, and model artifacts must be versioned so molecular ML results can be audited.

## Key References

1. Rogers, D., & Hahn, M. (2010). *Extended-connectivity fingerprints.* Journal of Chemical Information and Modeling, 50, 742-754. https://doi.org/10.1021/ci100050t
2. Durant, J. L., et al. (2002). *Reoptimization of MDL keys for use in drug discovery.* Journal of Chemical Information and Computer Sciences, 42, 1273-1280. https://doi.org/10.1021/ci010132r
3. Landrum, G. (2024). *RDKit: Open-source cheminformatics software.* https://www.rdkit.org/
4. Moriwaki, H., Tian, Y.-S., Kawashita, N., & Takagi, T. (2018). *Mordred: a molecular descriptor calculator.* Journal of Cheminformatics, 10, 4. https://doi.org/10.1186/s13321-018-0258-y
5. Wu, Z., et al. (2018). *MoleculeNet: A benchmark for molecular machine learning.* Chemical Science, 9, 513-530. https://doi.org/10.1039/C7SC02664A
6. Bemis, G. W., & Murcko, M. A. (1996). *The properties of known drugs. 1. Molecular frameworks.* Journal of Medicinal Chemistry, 39, 2887-2893. https://doi.org/10.1021/jm9602928
