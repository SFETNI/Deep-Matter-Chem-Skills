---
name: molecular-datasets
domain: cheminformatics
maturity: draft
tools:
  - RDKit
  - pandas
  - numpy
  - scikit-learn
  - DeepChem
  - matplotlib
tested_versions:
  RDKit: "2023.9.4"
  pandas: "2.2.0"
  numpy: "1.26.4"
  scikit-learn: "1.4.0"
  DeepChem: "2.8.0"
requires_gpu: false
estimated_runtime: minutes-to-hours
---

# Molecular Datasets

## Description

This skill covers molecular datasets for chemistry and molecular machine learning: selecting appropriate sources, preserving identifiers and metadata, standardizing molecular records, deduplicating structures, harmonizing units and labels, creating leakage-aware splits, auditing provenance and licenses, and preparing curated datasets for molecular featurization, molecular GNNs, small-data ML, and surrogate validation. Invoke this skill before training or comparing molecular property models, especially when combining public benchmarks, database exports, quantum-chemistry datasets, or internal experimental measurements.

## Domain Context

A molecular dataset is more than a table of SMILES and labels. Each row carries assumptions about molecular identity, measurement protocol, target definition, and data provenance. The same compound may appear as a salt, neutral parent, tautomer, stereoisomer, InChIKey duplicate, SDF entry, or assay-specific record. Two rows with the same canonical SMILES may have different labels because they were measured at different pH, temperature, solvent, assay target, experimental protocol, or quantum-chemistry level.

Dataset source defines what claims a model can support. QM9 is a small-molecule quantum-chemistry dataset for equilibrium gas-phase organic molecules. MoleculeNet aggregates many benchmark tasks with different label types and split conventions. Tox21, ESOL, FreeSolv, and Lipophilicity are useful molecular-property benchmarks but are small and can be overfit by repeated leaderboard tuning. ChEMBL, BindingDB, PubChem, ZINC, and internal screening campaigns contain richer chemistry but require heavier curation, metadata harmonization, and licensing checks. OC20/OC22 are primarily catalyst/surface datasets rather than ordinary isolated-molecule datasets, but they are relevant when molecules are adsorbates on materials surfaces.

The core risk is accidental incompatibility. Mixing pIC50, Ki, Kd, percent inhibition, and qualitative activity labels as if they are one target creates a model of assay artifacts. Mixing gas-phase DFT barriers with solvent-corrected experimental rates creates a multi-fidelity target that must be labeled as such. Combining public and internal data without source tags makes it impossible to diagnose whether a model learned chemistry or source bias.

Splits must match deployment. Random splits measure interpolation among similar molecules. Scaffold splits estimate transfer to new cores. Time splits estimate prospective performance when new measurements arrive later. Assay or source splits test protocol transfer. Out-of-distribution splits test new chemotypes or property regimes. Comparing models across incompatible splits is scientifically meaningless.

## When to Use This Skill

- Building a molecular ML dataset from QM9, MoleculeNet, Tox21, ESOL, FreeSolv, Lipophilicity, ChEMBL, PubChem, ZINC, BindingDB, internal experiments, or quantum-chemistry calculations.
- Combining multiple molecular data sources while preserving provenance, units, method metadata, and license information.
- Designing train/validation/test splits for molecular property prediction, bioactivity modeling, toxicity screening, quantum-property prediction, or molecular surrogate modeling.
- Auditing benchmark results for duplicate molecules, scaffold leakage, inconsistent labels, or incompatible split protocols.
- Preparing curated inputs for RDKit featurization, molecular GNNs, Gaussian processes, small-data ML, or Bayesian optimization.
- Recording dataset manifests so molecular ML results can be reproduced and extended.

## When Not to Use This Skill

- **When the data are periodic structures rather than molecules.** Use materials database and crystal GNN workflows for crystals, surfaces, defects, and periodic adsorbate systems.
- **When the target is not tied to molecular identity.** Reaction yields, synthesis outcomes, or formulation properties may require reaction, process, or mixture datasets rather than single-molecule tables.
- **When raw assay records lack enough metadata to define the target.** If units, assay type, target protein, pH, temperature, or concentration are missing, labels may be unsuitable for quantitative modeling.
- **When licensing forbids derived dataset redistribution.** Some sources permit model training but not redistribution of structures, labels, or merged tables. Check before committing data artifacts.
- **When benchmark splits are fixed by a publication or competition.** Do not replace official splits if the goal is direct benchmark comparison; add additional splits separately.

## Core Concepts

- **Dataset source:** Public or internal origin of records, such as QM9, MoleculeNet, PubChem, ChEMBL, ZINC, BindingDB, OC20/OC22, or an experimental campaign.
- **Molecular identifier:** A representation or key for molecular identity: SMILES, canonical SMILES, isomeric SMILES, InChI, InChIKey, SDF, or MOL2.
- **Record identifier:** Stable source-specific row ID, such as ChEMBL molecule ID, PubChem CID, BindingDB ligand ID, QM9 index, or internal sample ID. Keep this even after standardization.
- **Target type:** Label family, such as quantum property, solubility, toxicity, bioactivity, binding affinity, redox potential, reaction barrier, or materials-relevant molecular property.
- **Assay metadata:** Experimental context for a label: assay target, protocol, units, pH, temperature, solvent, organism, cell line, method, replicate, and uncertainty.
- **Quantum metadata:** Computational context for a label: method, basis set, functional, charge, spin, geometry source, solvent model, convergence thresholds, and code version.
- **Standardization policy:** Rules for salts, tautomers, protonation, stereochemistry, isotopes, metal complexes, mixtures, and disconnected fragments.
- **Deduplication key:** Identifier used to collapse or group records, often InChIKey, first InChIKey block, canonical tautomer parent, or assay-specific compound ID.
- **Label conflict:** Multiple labels for the same molecular identity that disagree beyond expected uncertainty. Conflicts can be real protocol differences or data errors.
- **Split protocol:** Rule assigning rows or molecule groups to train, validation, and test sets. The protocol is part of the dataset definition.
- **Provenance manifest:** Machine-readable file recording source, query, timestamp, license, filtering, standardization, deduplication, split seed, and checksums.

## Key Workflows

### Workflow 1: Build a dataset manifest from heterogeneous sources

Start with a manifest that preserves source IDs and raw labels. Do not collapse records until source, units, and target definitions are audited.

```python
from pathlib import Path
import json
import hashlib
from datetime import datetime
import pandas as pd


REQUIRED_COLUMNS = [
    "source",
    "source_id",
    "raw_molecule",
    "molecule_format",
    "target_name",
    "target_value",
    "target_units",
    "target_type",
]


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_source_table(path: str, source: str, column_map: dict) -> pd.DataFrame:
    """
    Load one source table and map it into the canonical manifest schema.

    column_map example:
        {"smiles": "raw_molecule", "activity": "target_value"}
    """
    df = pd.read_csv(path)
    out = pd.DataFrame()
    for src_col, dst_col in column_map.items():
        out[dst_col] = df[src_col]
    out["source"] = source
    out["source_file"] = path
    for col in REQUIRED_COLUMNS:
        if col not in out:
            out[col] = None
    return out[REQUIRED_COLUMNS + ["source_file"]]


def write_dataset_manifest(
    rows: pd.DataFrame,
    output_csv: str,
    output_metadata_json: str,
    source_files: list[str],
    license_notes: dict,
    curation_notes: list[str],
) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in rows.columns]
    if missing:
        raise ValueError(f"Manifest missing required columns: {missing}")

    rows.to_csv(output_csv, index=False)
    metadata = {
        "created_utc": datetime.utcnow().isoformat() + "Z",
        "manifest_csv": output_csv,
        "n_rows": int(len(rows)),
        "sources": sorted(rows["source"].dropna().astype(str).unique()),
        "target_names": sorted(rows["target_name"].dropna().astype(str).unique()),
        "source_file_sha256": {path: sha256_file(path) for path in source_files if Path(path).exists()},
        "license_notes": license_notes,
        "curation_notes": curation_notes,
    }
    Path(output_metadata_json).write_text(json.dumps(metadata, indent=2))
    print(f"Wrote {len(rows)} raw dataset records to {output_csv}")
```

### Workflow 2: Standardize identifiers and preserve molecular identity layers

Keep raw and standardized identifiers side by side. The raw representation is provenance; the standardized representation is a modeling choice.

```python
import pandas as pd
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize


def standardize_for_dataset(mol: Chem.Mol, strip_salts: bool = True) -> tuple[Chem.Mol | None, list[str]]:
    notes = []
    try:
        out = Chem.Mol(mol)
        if strip_salts:
            out = rdMolStandardize.LargestFragmentChooser().choose(out)
            notes.append("largest_fragment")
        out = rdMolStandardize.Normalizer().normalize(out)
        notes.append("normalize")
        Chem.SanitizeMol(out)
        return out, notes
    except Exception as exc:
        return None, [f"standardization_failed:{type(exc).__name__}"]


def add_identifier_columns(
    manifest_csv: str,
    output_csv: str,
    molecule_col: str = "raw_molecule",
    format_col: str = "molecule_format",
) -> pd.DataFrame:
    df = pd.read_csv(manifest_csv)
    rows = []
    for _, row in df.iterrows():
        raw = row[molecule_col]
        fmt = str(row.get(format_col, "smiles")).lower()
        out = row.to_dict()
        mol = None
        if pd.notna(raw):
            if fmt in {"smiles", "canonical_smiles", "isomeric_smiles"}:
                mol = Chem.MolFromSmiles(str(raw))
            elif fmt == "inchi":
                mol = Chem.MolFromInchi(str(raw))
            elif fmt in {"sdf", "mol", "molblock"}:
                mol = Chem.MolFromMolBlock(str(raw), sanitize=True, removeHs=False)

        if mol is None:
            out.update({
                "parse_status": "parse_failed",
                "canonical_smiles": None,
                "isomeric_smiles": None,
                "inchi": None,
                "inchikey": None,
                "inchikey14": None,
                "standardization_notes": "parse_failed",
            })
            rows.append(out)
            continue

        std_mol, notes = standardize_for_dataset(mol)
        if std_mol is None:
            out.update({
                "parse_status": "standardization_failed",
                "canonical_smiles": None,
                "isomeric_smiles": None,
                "inchi": None,
                "inchikey": None,
                "inchikey14": None,
                "standardization_notes": ";".join(notes),
            })
            rows.append(out)
            continue

        inchi = Chem.MolToInchi(std_mol)
        inchikey = Chem.MolToInchiKey(std_mol)
        out.update({
            "parse_status": "ok",
            "canonical_smiles": Chem.MolToSmiles(std_mol, isomericSmiles=False),
            "isomeric_smiles": Chem.MolToSmiles(std_mol, isomericSmiles=True),
            "inchi": inchi,
            "inchikey": inchikey,
            "inchikey14": inchikey.split("-")[0],
            "formal_charge": Chem.GetFormalCharge(std_mol),
            "n_heavy_atoms": std_mol.GetNumHeavyAtoms(),
            "standardization_notes": ";".join(notes),
        })
        rows.append(out)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(output_csv, index=False)
    print(out_df["parse_status"].value_counts(dropna=False))
    return out_df
```

### Workflow 3: Harmonize target units and detect label conflicts

Unit harmonization must happen before model training. Keep original units and converted units so conversions can be audited.

```python
import numpy as np
import pandas as pd


def convert_bioactivity_to_pvalue(value, units: str) -> float | None:
    """
    Convert molar concentration labels to pX = -log10(M).

    Only use when value is a concentration such as IC50, Ki, Kd, or EC50.
    Do not mix pIC50 and pKi without target-type metadata. [EXPERT REVIEW NEEDED]
    """
    if pd.isna(value):
        return None
    units = str(units).lower().strip()
    factor_to_molar = {
        "m": 1.0,
        "mm": 1e-3,
        "um": 1e-6,
        "micromolar": 1e-6,
        "nm": 1e-9,
        "nanomolar": 1e-9,
    }.get(units)
    if factor_to_molar is None:
        return None
    molar = float(value) * factor_to_molar
    if molar <= 0:
        return None
    return float(-np.log10(molar))


def harmonize_targets(df: pd.DataFrame, output_csv: str) -> pd.DataFrame:
    out = df.copy()
    out["target_value_original"] = out["target_value"]
    out["target_units_original"] = out["target_units"]
    out["target_value_model"] = np.nan
    out["target_units_model"] = None
    out["target_harmonization_status"] = "unhandled"

    for i, row in out.iterrows():
        target_type = str(row.get("target_type", "")).lower()
        units = str(row.get("target_units", "")).lower()
        value = row.get("target_value")
        if target_type in {"ic50", "ki", "kd", "ec50"}:
            pval = convert_bioactivity_to_pvalue(value, units)
            if pval is not None:
                out.loc[i, "target_value_model"] = pval
                out.loc[i, "target_units_model"] = f"p{target_type.upper()}"
                out.loc[i, "target_harmonization_status"] = "converted_to_pvalue"
        elif target_type in {"solubility", "logsolubility", "logp", "logd", "toxicity", "quantum_property"}:
            out.loc[i, "target_value_model"] = float(value) if pd.notna(value) else np.nan
            out.loc[i, "target_units_model"] = row.get("target_units")
            out.loc[i, "target_harmonization_status"] = "kept_original_units"

    out.to_csv(output_csv, index=False)
    print(out["target_harmonization_status"].value_counts(dropna=False))
    return out


def find_label_conflicts(
    df: pd.DataFrame,
    group_col: str = "inchikey",
    value_col: str = "target_value_model",
    target_col: str = "target_name",
    source_col: str = "source",
    threshold: float = 1.0,
) -> pd.DataFrame:
    """
    Report duplicate molecular identities with target spread above threshold.
    For pIC50-like labels, threshold=1 means a 10x concentration difference.
    """
    rows = []
    valid = df.dropna(subset=[group_col, value_col])
    for (mol_id, target), g in valid.groupby([group_col, target_col]):
        if len(g) < 2:
            continue
        spread = float(g[value_col].max() - g[value_col].min())
        if spread >= threshold:
            rows.append({
                "molecule_id": mol_id,
                "target_name": target,
                "n_records": int(len(g)),
                "value_min": float(g[value_col].min()),
                "value_max": float(g[value_col].max()),
                "value_spread": spread,
                "sources": sorted(g[source_col].dropna().astype(str).unique()),
            })
    conflicts = pd.DataFrame(rows).sort_values("value_spread", ascending=False) if rows else pd.DataFrame(rows)
    print(f"Label conflict groups: {len(conflicts)}")
    return conflicts
```

### Workflow 4: Create leakage-aware molecular splits

Store split assignments as dataset artifacts. Never regenerate splits implicitly from a random seed alone after a dataset changes.

```python
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import GroupShuffleSplit


def murcko_scaffold_from_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "invalid"
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    return Chem.MolToSmiles(scaffold, isomericSmiles=True) or "acyclic"


def add_split_groups(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["scaffold"] = out["isomeric_smiles"].fillna("").map(murcko_scaffold_from_smiles)
    out["molecule_family"] = out["inchikey14"].fillna(out["inchikey"]).fillna(out["source_id"])
    if "assay_id" not in out:
        out["assay_id"] = out["target_name"].astype(str) + "::" + out["source"].astype(str)
    return out


def assign_group_split(
    df: pd.DataFrame,
    group_col: str,
    output_csv: str,
    test_size: float = 0.20,
    val_size: float = 0.10,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Assign train/val/test by group. group_col can be scaffold, assay_id,
    source, molecule_family, or time bucket.
    """
    data = df.dropna(subset=[group_col]).copy()
    data["split"] = "train"
    groups = data[group_col].astype(str).to_numpy()

    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_val_idx, test_idx = next(splitter.split(data, groups=groups))
    data.iloc[test_idx, data.columns.get_loc("split")] = "test"

    train_val = data.iloc[train_val_idx].copy()
    val_fraction_of_train_val = val_size / max(1e-12, 1.0 - test_size)
    splitter2 = GroupShuffleSplit(n_splits=1, test_size=val_fraction_of_train_val, random_state=seed + 1)
    train_idx2, val_idx2 = next(splitter2.split(train_val, groups=train_val[group_col].astype(str).to_numpy()))
    val_global_index = train_val.iloc[val_idx2].index
    data.loc[val_global_index, "split"] = "val"

    data.to_csv(output_csv, index=False)
    print(data["split"].value_counts())
    print(f"Split group: {group_col}; unique test groups = {data.loc[data['split'] == 'test', group_col].nunique()}")
    return data


def audit_split_overlap(df: pd.DataFrame, group_cols: list[str]) -> None:
    for col in group_cols:
        if col not in df:
            continue
        train = set(df.loc[df["split"] == "train", col].dropna().astype(str))
        val = set(df.loc[df["split"] == "val", col].dropna().astype(str))
        test = set(df.loc[df["split"] == "test", col].dropna().astype(str))
        print(f"{col}: train-val overlap={len(train & val)}, train-test overlap={len(train & test)}, val-test overlap={len(val & test)}")
```

### Workflow 5: Audit metadata completeness and license constraints

Metadata completeness is part of data quality. A dataset with 50,000 labels but no units can be less useful than 500 well-annotated labels.

```python
from pathlib import Path
import json
import pandas as pd


METADATA_BY_TARGET = {
    "bioactivity": ["target_name", "target_units", "assay_id", "assay_type", "organism", "source"],
    "toxicity": ["target_name", "target_units", "assay_id", "organism", "endpoint", "source"],
    "solubility": ["target_units", "pH", "temperature_K", "solvent", "source"],
    "binding_affinity": ["target_name", "target_units", "protein_id", "assay_id", "temperature_K", "pH"],
    "quantum_property": ["method", "basis_or_functional", "charge", "spin", "geometry_source", "source"],
    "redox_potential": ["target_units", "reference_electrode", "solvent", "temperature_K", "method", "source"],
}


def audit_metadata_completeness(df: pd.DataFrame, output_csv: str) -> pd.DataFrame:
    rows = []
    for target_type, required in METADATA_BY_TARGET.items():
        subset = df[df["target_type"].astype(str).str.lower() == target_type]
        if len(subset) == 0:
            continue
        for col in required:
            present = col in subset.columns
            completeness = float(subset[col].notna().mean()) if present else 0.0
            rows.append({
                "target_type": target_type,
                "column": col,
                "n_rows": int(len(subset)),
                "completeness": completeness,
                "status": "ok" if completeness >= 0.95 else "incomplete",
            })
    report = pd.DataFrame(rows)
    report.to_csv(output_csv, index=False)
    print(report.sort_values("completeness").head(20))
    return report


def write_license_manifest(
    dataset_name: str,
    sources: list[dict],
    output_json: str,
) -> None:
    """
    sources entries should include source, url, license, access_date,
    redistribution_allowed, commercial_use_allowed, and citation.
    """
    manifest = {
        "dataset_name": dataset_name,
        "license_audit_required": True,
        "sources": sources,
        "notes": [
            "Do not redistribute merged data unless all source licenses permit it.",
            "For internal datasets, record data owner and approved use.",
            "For benchmark comparisons, preserve official split and citation metadata.",
        ],
    }
    Path(output_json).write_text(json.dumps(manifest, indent=2))
```

### Workflow 6: Dataset card and reproducibility bundle

Every curated dataset should ship with a short dataset card, split files, manifest metadata, and checksums.

```python
from pathlib import Path
import json
import pandas as pd


def make_dataset_card(
    curated_csv: str,
    split_csv: str,
    metadata_json: str,
    output_md: str,
) -> None:
    df = pd.read_csv(curated_csv)
    split_df = pd.read_csv(split_csv)
    meta = json.loads(Path(metadata_json).read_text()) if Path(metadata_json).exists() else {}

    lines = [
        f"# Dataset Card: {meta.get('dataset_name', Path(curated_csv).stem)}",
        "",
        "## Summary",
        f"- Rows: {len(df)}",
        f"- Unique molecules by InChIKey: {df['inchikey'].nunique() if 'inchikey' in df else 'unknown'}",
        f"- Sources: {', '.join(sorted(df['source'].dropna().astype(str).unique())) if 'source' in df else 'unknown'}",
        f"- Target names: {', '.join(sorted(df['target_name'].dropna().astype(str).unique())) if 'target_name' in df else 'unknown'}",
        "",
        "## Splits",
    ]
    if "split" in split_df:
        for split, count in split_df["split"].value_counts().items():
            lines.append(f"- {split}: {count}")
    lines.extend([
        "",
        "## Required Metadata",
        "- Molecular standardization policy",
        "- Target units and conversion rules",
        "- Source and license information",
        "- Split protocol and random seed",
        "- Known limitations and excluded records",
        "",
        "## Known Risks",
        "- Duplicate molecules under alternate representations",
        "- Assay-specific labels mixed across protocols",
        "- Scaffold leakage if official split is not used",
        "- Benchmark overfitting from repeated public test-set use",
    ])
    Path(output_md).write_text("\n".join(lines) + "\n")
```

## Common Failure Modes

| Failure | Symptom | Likely Cause | Diagnostic | Fix |
|---------|---------|--------------|------------|-----|
| Duplicate molecules under different representations | Same compound appears in train and test with different SMILES, salt, tautomer, or stereoisomer labels | Deduplication used raw SMILES only | Count InChIKey and tautomer-parent duplicates before splitting | Standardize first; group duplicates into the same split |
| Label conflicts | Same molecule has several incompatible target values | Mixed assays, unit errors, replicate variability, or source disagreement | Run label conflict report by InChIKey and target | Keep replicate metadata, average only compatible labels, or model assay separately |
| Missing assay metadata | Bioactivity model performs well globally but fails by source | Target protein, assay type, pH, or units missing | Metadata completeness audit by target type | Exclude under-specified labels or downgrade to qualitative task |
| Unit mismatch | Activity or solubility labels differ by orders of magnitude | nM, uM, M, log units, or percent activity mixed | Inspect units and converted target ranges by source | Convert units explicitly and keep original units |
| Invalid structures | Descriptor generation drops rows or crashes | Malformed SMILES, valence errors, mixtures, polymers, or unsupported metals | Parse-status counts and source-level failure rate | Fix source records or exclude with documented reason |
| Stereochemistry loss | Stereoisomers collapse to one identity or share labels incorrectly | Non-isomeric SMILES or conversion stripped chiral information | Compare canonical and isomeric SMILES; count unspecified chiral centers | Preserve isomeric identifiers and group ambiguous records |
| Scaffold leakage | Scaffold split performance much worse than random split | Analog series split across train and test | Murcko scaffold overlap audit | Use scaffold or time split for deployment claims |
| Benchmark overfitting | Public benchmark score improves but external dataset fails | Repeated tuning on known test set or leaderboard feedback | Compare to untouched internal or temporal split | Freeze benchmark use; report external validation |
| Incompatible splits compared | One model appears better only because it used an easier split | Random split result compared to scaffold split result | Check split metadata and row IDs | Recompute all models on identical split files |
| License violation risk | Dataset cannot be shared despite strong model results | Source license forbids redistribution or commercial use | License manifest review | Share derived model only if allowed; cite source and follow terms |

## Best Practices

- **Preserve raw records.** Raw identifiers, source IDs, units, and labels are provenance, even if standardized fields are used for modeling.
- **Standardize before splitting.** Salt stripping, tautomer handling, stereochemistry, protonation, and deduplication must happen before split assignment.
- **Keep split files immutable.** Store row IDs and split labels as a committed artifact; do not regenerate splits implicitly.
- **Use split protocols that match deployment.** Random split for interpolation, scaffold split for new cores, time split for prospective claims, assay/source split for protocol transfer.
- **Never mix target definitions silently.** pIC50, pKi, percent inhibition, solubility in different solvents, and DFT levels are different labels unless harmonized.
- **Audit metadata completeness.** Units, pH, temperature, solvent, method, source, and uncertainty determine whether labels are scientifically usable.
- **Track licenses and citations.** Dataset provenance includes legal permission and required attribution.
- **Report benchmark limitations.** Public benchmark scores are useful only with official splits, exact preprocessing, and citation of prior tuning exposure.
- **Keep negative and failed records.** Invalid structures and failed assays reveal dataset boundaries and prevent silent bias.
- **Version the complete curation pipeline.** Queries, raw exports, standardization settings, filters, splits, feature matrices, and dataset cards should all be reproducible.

## Integration with Other Skills

- **[molecular-featurization](../molecular-featurization/SKILL.md):** Molecular datasets provide the curated identifiers, standardization policy, and split files consumed by descriptor and fingerprint generation.
- **[molecular-property-gnn](../../geometric-deep-learning/molecular-property-gnn/SKILL.md):** Molecular GNN training depends on clean graph inputs, preserved stereochemistry, conformer metadata, and scaffold or time splits from the dataset layer.
- **[crystal-property-gnn](../../geometric-deep-learning/crystal-property-gnn/SKILL.md):** Molecular datasets connect to crystal GNNs for molecular crystals, organic semiconductors, salts, and adsorbate/material hybrid datasets.
- **[small-data-ml](../../scientific-ml/small-data-ml/SKILL.md):** Many molecular datasets are small or effectively small after grouping by scaffold or assay; use nested CV, baselines, and confidence intervals from that skill.
- **[surrogate-validation](../../surrogate-active-learning/surrogate-validation/SKILL.md):** Split integrity, label conflict audits, applicability-domain checks, and negative-result tracking feed directly into surrogate validation.
- **[scientific-visualization](../../scientific-data/scientific-visualization/SKILL.md):** Dataset coverage maps, scaffold distributions, label conflict plots, missingness heatmaps, and split diagnostics should use reproducible figure conventions.
- **[reproducibility-workflow](../../scientific-data/reproducibility-workflow/SKILL.md):** Dataset manifests, checksums, license manifests, split files, and dataset cards are reproducibility artifacts that should be versioned.
- **[materials-featurization](../../scientific-data/materials-featurization/SKILL.md):** Molecular datasets overlap with materials featurization when molecules, adsorbates, molecular crystals, redox molecules, or organic electronics candidates enter materials ML workflows.

## Key References

1. Wu, Z., et al. (2018). *MoleculeNet: A benchmark for molecular machine learning.* Chemical Science, 9, 513-530. https://doi.org/10.1039/C7SC02664A
2. Ramakrishnan, R., et al. (2014). *Quantum chemistry structures and properties of 134 kilo molecules.* Scientific Data, 1, 140022. https://doi.org/10.1038/sdata.2014.22
3. Gaulton, A., et al. (2017). *The ChEMBL database in 2017.* Nucleic Acids Research, 45, D945-D954. https://doi.org/10.1093/nar/gkw1074
4. Kim, S., et al. (2023). *PubChem 2023 update.* Nucleic Acids Research, 51, D1373-D1380. https://doi.org/10.1093/nar/gkac956
5. Sterling, T., & Irwin, J. J. (2015). *ZINC 15 - ligand discovery for everyone.* Journal of Chemical Information and Modeling, 55, 2324-2337. https://doi.org/10.1021/acs.jcim.5b00559
6. Gilson, M. K., et al. (2016). *BindingDB in 2015: A public database for medicinal chemistry, computational chemistry and systems pharmacology.* Nucleic Acids Research, 44, D1045-D1053. https://doi.org/10.1093/nar/gkv1072
7. Chanussot, L., et al. (2021). *Open Catalyst 2020 (OC20) dataset and community challenges.* ACS Catalysis, 11, 6059-6072. https://doi.org/10.1021/acscatal.0c04525
