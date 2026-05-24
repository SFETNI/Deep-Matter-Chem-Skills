---
name: materials-featurization
domain: scientific-data
maturity: draft
tools:
  - matminer
  - pymatgen
  - ASE
  - DScribe
  - RDKit
tested_versions:
  matminer: "0.9.2"
  pymatgen: "2024.1.26"
  ASE: "3.23.0"
  DScribe: "2.1.0"
  RDKit: "2023.9.4"
requires_gpu: false
estimated_runtime: minutes-to-hours
---

# Materials Featurization

## Description

This skill covers the conversion of atomic structures, compositions, and trajectories into numerical feature vectors for physical-science ML: composition featurizers (Magpie-style elemental statistics, oxidation-state features), structure featurizers (SOAP, Coulomb matrix, Voronoi, radial distribution functions, bond orientational order), site-level featurizers with structure-level aggregation, and descriptor selection for crystals, molecules, surfaces, defects, amorphous structures, and MD trajectories. Covers matminer, pymatgen, ASE, DScribe, and RDKit. Invoke this skill when building a feature matrix for regression or classification on materials datasets, when selecting between hand-crafted descriptors and learned graph representations, or when auditing a feature pipeline for leakage, NaN patterns, or physical inconsistencies.

## Domain Context

Materials featurization is the translation of atomic structures and compositions into vectors of numbers that a statistical model can process. Unlike image or text data where pixels and tokens have a natural numerical representation, materials structures have no canonical encoding: a crystal can be described by its unit cell (arbitrary choice of conventional vs. primitive cell), its composition (continuous or categorical species fractions), its local environments (per-site descriptors aggregated to structure level), or its global topology (graph connectivity). Each encoding captures different physical information and has different sensitivity to symmetry operations, cell conventions, and disorder.

**Hand-crafted descriptors vs. learned representations.** Composition-based descriptors (Magpie, CGCNN input) require only element symbols and fractions — cheap to compute, insensitive to structural details, and appropriate for screening or when structures are unavailable. Structure-based descriptors (SOAP, Voronoi, Coulomb matrix) use full atomic positions but are O(N²) or worse for global descriptors. Learned graph neural network representations (CGCNN, MEGNet, MatFormer) jointly learn the featurization and the regression, but require training data, are less interpretable, and cannot be used as feature vectors for arbitrary downstream models without fine-tuning. The choice between them depends on data quantity, interpretability requirements, computational budget, and whether the target property is dominated by local chemistry (SOAP is then sufficient) or long-range interactions (learned models with long-range attention are preferable).

**Featurization is not neutral.** The descriptor encodes assumptions about what physical information is relevant for the target property. SOAP captures local bonding environments up to a cutoff radius; it is blind to long-range electrostatics and topological features. Coulomb matrix eigenvalues capture intermolecular distances but are sensitive to the molecular geometry in ways that may not correlate with the property of interest. Magpie features average elemental properties by composition and are entirely blind to the crystal structure. Using the wrong descriptor class for a property introduces systematic prediction errors that cannot be fixed by model tuning.

**Leakage is the primary failure mode in materials ML.** Target leakage occurs when a feature encodes information that would not be available at prediction time — for example, including the DFT-computed band gap as a feature for predicting another DFT-computed property where the band gap is also the target or strongly correlated with it. Feature leakage occurs when preprocessing steps (scaling, imputation, feature selection) are fit on the combined train+test set rather than only the train set, making test performance optimistic. Materials datasets have additional structure: many crystals sharing a prototype differ only in lattice parameters, so a random train/test split may put near-duplicates on both sides. All of these require explicit mitigation.

**Missing values and physical invalidity.** Matminer featurizers can return NaN for structures where a required property is undefined — for example, `OxidationStateDecorator` fails on structures with ambiguous charge balance, and `CrystalNNFingerprint` can fail on structures with unusual coordination environments. A feature matrix with 20% NaN in a critical column that is imputed by mean produces a model that silently degrades for those structures without any warning.

## When to Use This Skill

- Building composition- or structure-based feature matrices for surrogate property models (formation energy, band gap, thermal conductivity, defect formation energy).
- Constructing diversity metrics and coverage maps for active learning or dataset curation (see `active-learning-mlp`): selecting which structures to label with DFT.
- Featurizing trajectories for per-frame property prediction, structure classification, or change-point detection in MD or active learning.
- Auditing an inherited feature matrix for leakage, NaN patterns, duplicate structures, or descriptor inconsistencies before using it for modeling.
- Comparing the information content of different descriptor families on a new chemical space.

## When Not to Use This Skill

- **When a graph neural network trained on the same dataset is available and more accurate** — learned representations from CGCNN, MEGNet, or CHGNet will typically outperform hand-crafted descriptors for the same training data if sufficient data (> 1,000 structures) is available. Use this skill when interpretability, speed, or small-data constraints favor fixed descriptors.
- **For properties requiring electron density or wavefunction information** — local environment descriptors are blind to quantum chemical effects like Mott localization, charge transfer, or orbital hybridization. DFT-derived features (PDOS moments, Bader charges, orbital character) require DFT calculations and are a different class of featurizer.
- **For long-timescale dynamical properties (diffusion coefficients, viscosity, glass transition)** — these require MD trajectories, not static structure features. Featurizing only the equilibrium structure for predicting dynamical properties will produce unreliable models. [EXPERT REVIEW NEEDED]
- **For molecular systems where topology matters more than geometry** — RDKit fingerprints (Morgan, MACCS, topological torsion) encode graph-based connectivity and are more appropriate than SOAP or Coulomb matrices for drug-like molecules. Use RDKit for molecules with known connectivity; use DScribe for periodic systems or molecules where only coordinates are available.

## Core Concepts

- **Featurizer:** A function that maps a material (as a pymatgen Structure/Composition, ASE Atoms, or RDKit Mol) to a fixed-length numerical vector. Featurizers can be composition-level (require only species and fractions), site-level (one vector per atom, then aggregated), or structure-level (require full atomic positions and cell).
- **SOAP (Smooth Overlap of Atomic Positions):** An atom-centered descriptor that encodes the local chemical environment within a cutoff radius `r_cut` using radial basis functions and spherical harmonics. Invariant to rotation, reflection, and translation. Produces a per-site vector of length `n_max × (l_max+1)² × (n_species+1)×n_species/2`. Averaged across sites to produce a structure-level vector. The most widely used descriptor for MLP active learning and structure similarity metrics.
- **Magpie descriptors:** A set of 145 composition-level features introduced in the Materials Agnostic Platform for Informatics and Exploration (Ward et al. 2016). Each feature is a statistical aggregate (mean, std, min, max, range, mode) of an elemental property (atomic number, electronegativity, atomic radius, cohesive energy, melting point, etc.) weighted by composition. Available via matminer's `ElementProperty` featurizer with `preset="magpie"`.
- **Coulomb matrix:** A pairwise matrix M_ij = Z_i Z_j / |r_i - r_j| (i ≠ j), M_ii = 0.5 Z_i^2.4. Encodes all pairwise distances weighted by nuclear charges. Not invariant to atom ordering — requires eigenvalue sorting or random rotation averaging. Scales as O(N²) in system size. Suited for molecules; less practical for large crystals.
- **Sine matrix (periodic Coulomb matrix):** Extension of the Coulomb matrix to periodic systems using the minimum image convention for distances. Avoids the vacuum problem of the Coulomb matrix for crystals. Implemented in DScribe.
- **Voronoi tessellation descriptors:** Partitions space into Voronoi cells; each atom's cell encodes its coordination geometry. matminer's `VoronoiFingerprint` computes statistics of the Voronoi polyhedron (n_faces, volume, area ratios). Robust to disordered structures; less sensitive to cutoff radius than SOAP.
- **CrystalNNFingerprint:** matminer featurizer that uses the CrystalNN neighbor-finding algorithm (distance + angle heuristics) to assign coordination environments and compute a fingerprint encoding the local geometry. More expensive than Voronoi but captures bond angle information. Can fail on unusual coordination environments (returns NaN).
- **Bond orientational order (BOO):** Steinhardt order parameters Q_l measuring the orientational symmetry of the nearest-neighbor shell. Q_6 ≈ 0.574 for FCC, ≈ 0.510 for BCC, ≈ 0.485 for HCP. Useful for detecting crystalline vs. amorphous environments in MD trajectories. Implemented in freud and OVITO.
- **Oxidation state assignment:** pymatgen's `OxidationStateDecorator` uses a bond-valence-based heuristic to assign formal charges to sites. Required for featurizers that encode oxidation-state-specific properties. Fails for metallic systems (cannot distribute charges) and for ambiguous mixed-valence systems. Always check the success rate before including oxidation-state features.
- **Feature leakage:** A feature that inadvertently encodes the target variable or is preprocessed using target-set information. Examples: including DFT formation energy as a feature for a model predicting DFT formation energy from composition; fitting a StandardScaler on train+test data; using the target property to select the most predictive features before splitting.
- **Train/test leakage in materials datasets:** Two structures that share a crystal prototype (e.g., both are rocksalt AB compounds differing only in composition) or that are related by a small structural perturbation will have nearly identical descriptor vectors. A random split assigns them to different folds, producing over-optimistic test error. Materials-aware splitting (by composition family, by crystal prototype, or by descriptor-space distance) is required for honest evaluation.

## Key Workflows

### Workflow 1: Composition featurizers — Magpie-style descriptors with matminer

Generate composition-level feature matrices. These require only chemical formulas; no structural information is needed. The fastest featurizer class and the only option when structures are unavailable.

```python
import numpy as np
import pandas as pd
from pymatgen.core import Composition
from matminer.featurizers.composition import ElementProperty, Stoichiometry, IonProperty
from matminer.featurizers.base import MultipleFeaturizer
from matminer.utils.data import DemlData, PymatgenData, MagpieData


def build_composition_feature_matrix(
    formulas: list[str],
    preset: str = "magpie",
    include_stoichiometry: bool = True,
    include_ion_property: bool = False,
) -> pd.DataFrame:
    """
    Build a composition-level feature matrix for a list of chemical formulas.

    preset: 'magpie' (145 features, recommended for general use),
            'deml' (DeML properties),
            'matscholar_el' (embedding-based elemental features) [EXPERT REVIEW NEEDED]
    include_stoichiometry: adds L2, L3, L7, L10 norm features of the element fractions.
    include_ion_property: adds average ionic radius and maximum possible oxidation states.
        Requires pymatgen's oxidation state data; may fail for unusual compositions.

    Returns a DataFrame with one row per formula.
    NaN values indicate featurization failures; check and handle before modeling.
    """
    df = pd.DataFrame({"formula": formulas})
    df["composition"] = df["formula"].map(lambda f: Composition(f))

    featurizers = [ElementProperty.from_preset(preset)]
    if include_stoichiometry:
        featurizers.append(Stoichiometry())
    if include_ion_property:
        # IonProperty can fail for metallic or mixed-valence compositions
        featurizers.append(IonProperty(fast=True))

    multi_feat = MultipleFeaturizer(featurizers)
    multi_feat.set_n_jobs(1)  # Avoid multiprocessing issues in HPC environments

    df = multi_feat.featurize_dataframe(
        df, col_id="composition", ignore_errors=True
    )

    # Report NaN fraction per feature
    nan_fractions = df.drop(columns=["formula", "composition"]).isnull().mean()
    high_nan_cols = nan_fractions[nan_fractions > 0.05].sort_values(ascending=False)
    if len(high_nan_cols) > 0:
        print(f"WARNING: {len(high_nan_cols)} features have > 5% NaN values:")
        for col, frac in high_nan_cols.head(10).items():
            print(f"  {col}: {frac:.1%}")

    return df


def check_composition_features(df: pd.DataFrame) -> None:
    """Sanity checks for a composition feature matrix before modeling."""
    feature_cols = [c for c in df.columns if c not in ("formula", "composition")]
    X = df[feature_cols].values

    print(f"\nComposition feature matrix: {X.shape[0]} structures × {X.shape[1]} features")

    # Check 1: NaN fraction
    nan_frac = np.isnan(X).mean()
    print(f"  NaN fraction: {nan_frac:.3%}")
    if nan_frac > 0.01:
        print(f"  WARNING: {nan_frac:.1%} NaN — inspect high-NaN columns before imputation")

    # Check 2: Zero-variance features (constant columns)
    variances = np.nanvar(X, axis=0)
    n_zero_var = (variances == 0).sum()
    if n_zero_var > 0:
        print(f"  WARNING: {n_zero_var} zero-variance features (constant across all structures)")
        print(f"    → These carry no information; drop before training")

    # Check 3: Feature value ranges for physical plausibility
    # Magpie mean electronegativity (Pauling) should be in [0.7, 4.0]
    en_cols = [c for c in feature_cols if "ElectroNeg" in c and "mean" in c.lower()]
    if en_cols:
        en_vals = df[en_cols[0]].dropna().values
        if en_vals.min() < 0.5 or en_vals.max() > 5.0:
            print(f"  WARNING: {en_cols[0]} has out-of-range values: "
                  f"[{en_vals.min():.2f}, {en_vals.max():.2f}]")

    print("  Composition feature check complete.")
```

### Workflow 2: Structure featurizers — SOAP descriptors with DScribe

SOAP is the workhorse descriptor for structure-based materials ML. Used for surrogate models, FPS-based diversity filtering, and MLP active learning.

```python
import numpy as np
from ase.io import read
from dscribe.descriptors import SOAP
from sklearn.preprocessing import normalize


def compute_soap_features(
    structures: list,           # List of ASE Atoms objects
    species: list[str],
    r_cut: float = 5.0,
    n_max: int = 9,
    l_max: int = 6,
    sigma: float = 0.5,
    average: str = "inner",     # "inner" (before power spectrum) or "outer" (after)
    sparse: bool = False,
    normalize_vectors: bool = True,
) -> np.ndarray:
    """
    Compute SOAP descriptors for a list of structures.

    average: "inner" averages the density expansion coefficients before
        computing the power spectrum — the result is a single vector per
        structure. "outer" averages the power spectra — subtly different
        normalization. "inner" is standard for structure-level features;
        use "off" for per-site descriptors.
    sigma: Gaussian broadening width in Å. Controls sensitivity to atomic
        displacements. Smaller sigma → more sensitive to small displacements.
        [EXPERT REVIEW NEEDED: sigma choice is system-dependent]

    Returns: (N_structures, D) float64 array.
    D = n_max² × (l_max+1) × n_species_pairs for real-valued power spectrum.
    """
    soap = SOAP(
        species=species,
        r_cut=r_cut,
        n_max=n_max,
        l_max=l_max,
        sigma=sigma,
        average=average,
        sparse=sparse,
        periodic=True,     # Must match the structure's PBC
    )

    # Check PBC consistency
    for i, atoms in enumerate(structures):
        if not all(atoms.pbc):
            print(f"WARNING: structure {i} has non-periodic PBC but SOAP is set to periodic=True")
            print(f"  → Set periodic=False for molecular systems, or fix PBC flags")

    descs = soap.create(structures)  # (N, D) or list of (N_i, D) for per-site

    if normalize_vectors:
        descs = normalize(descs, norm="l2")

    print(f"SOAP descriptor shape: {descs.shape}")
    print(f"  n_features = n_max²·(l_max+1)·n_pairs = "
          f"{n_max}²·{l_max+1}·{len(species)*(len(species)+1)//2} = {descs.shape[1]}")

    return descs


def soap_pairwise_distances(descs: np.ndarray, n_sample: int = None) -> np.ndarray:
    """
    Compute pairwise Euclidean distances in SOAP space.
    Used for duplicate detection, FPS, and train/test leakage assessment.

    For large datasets, n_sample limits the comparison to a random subset.
    """
    if n_sample and n_sample < len(descs):
        idx = np.random.choice(len(descs), n_sample, replace=False)
        descs_sub = descs[idx]
    else:
        descs_sub = descs

    # Euclidean distance in normalized SOAP space
    # = sqrt(2 - 2*cos_similarity) for normalized vectors
    dot = descs_sub @ descs_sub.T
    dot = np.clip(dot, -1, 1)
    dist = np.sqrt(np.maximum(0, 2 - 2 * dot))
    np.fill_diagonal(dist, 0.0)
    return dist


def soap_cutoff_sensitivity_check(
    structures: list,
    species: list[str],
    r_cut_values: list[float],
) -> None:
    """
    Test how descriptor distances change with r_cut.
    If the pairwise distance matrix changes significantly between r_cut=5.0
    and r_cut=6.0, the descriptor is sensitive to the cutoff in this range
    and the choice should be physically motivated (e.g., beyond the second
    coordination shell is usually sufficient).
    """
    from scipy.spatial.distance import pdist

    print("SOAP cutoff sensitivity test:")
    prev_dists = None
    for r_cut in r_cut_values:
        soap = SOAP(species=species, r_cut=r_cut, n_max=9, l_max=6,
                    sigma=0.5, average="inner", periodic=True)
        descs = normalize(soap.create(structures), norm="l2")
        dists = pdist(descs, metric="euclidean")
        if prev_dists is not None:
            corr = np.corrcoef(prev_dists, dists)[0, 1]
            delta = np.abs(dists - prev_dists).mean()
            print(f"  r_cut={r_cut:.1f} Å: corr with prev = {corr:.4f}, "
                  f"mean |Δdist| = {delta:.4f}")
        else:
            print(f"  r_cut={r_cut:.1f} Å: baseline")
        prev_dists = dists

    print("  Interpretation: corr > 0.99 → insensitive to cutoff at this scale.")
    print("  corr < 0.95 → cutoff spans a coordination shell boundary; "
          "choose r_cut to include or exclude a full shell.")
```

### Workflow 3: Structure-level featurization with matminer

matminer provides a large library of site-level and structure-level featurizers. Site-level featurizers produce per-atom features that must be aggregated to structure-level.

```python
import numpy as np
import pandas as pd
from pymatgen.core import Structure
from matminer.featurizers.structure import (
    SiteStatsFingerprint,
    DensityFeatures,
    RadialDistributionFunction,
)
from matminer.featurizers.site import (
    CrystalNNFingerprint,
    VoronoiFingerprint,
    ChemicalSRO,
    CoordinationNumber,
)


def build_structure_feature_matrix(
    structures: list[Structure],
    featurizer_set: str = "voronoi",
) -> pd.DataFrame:
    """
    Build a structure-level feature matrix using matminer site featurizers.

    featurizer_set:
        "voronoi"  — Voronoi-based fingerprint, robust, fast, handles disorder.
                     Default for metals and oxides.
        "crystalnn" — CrystalNN-based fingerprint, captures angular geometry.
                      More accurate for molecular and covalent systems but slower
                      and more likely to fail on unusual environments (returns NaN).
        "density"  — Cell density, packing fraction, vpa. Fastest; least informative.

    Returns: DataFrame with one row per structure.
    """
    df = pd.DataFrame({"structure": structures})

    if featurizer_set == "voronoi":
        # VoronoiFingerprint: statistics of Voronoi polyhedron per site
        # aggregated to structure-level via SiteStatsFingerprint
        site_feat = VoronoiFingerprint(use_symm_weights=True, use_weights="area")
        feat = SiteStatsFingerprint(
            site_feat,
            stats=["mean", "std_dev", "minimum", "maximum"],
        )
        feat.fit(df["structure"].tolist())
        df = feat.featurize_dataframe(df, col_id="structure", ignore_errors=True)

    elif featurizer_set == "crystalnn":
        # CrystalNNFingerprint: requires oxidation state decoration first
        # for best results (uses oxidation-state-aware bond valence distances)
        from pymatgen.analysis.local_env import CrystalNN
        site_feat = CrystalNNFingerprint.from_preset("ops")  # Orbital-projection fingerprint
        feat = SiteStatsFingerprint(
            site_feat,
            stats=["mean", "std_dev", "minimum", "maximum"],
        )
        feat.fit(df["structure"].tolist())
        df = feat.featurize_dataframe(df, col_id="structure", ignore_errors=True)
        # CrystalNNFingerprint fails silently on unusual environments → returns NaN row
        n_nan_rows = df.isnull().any(axis=1).sum()
        if n_nan_rows > 0:
            print(f"WARNING: {n_nan_rows} structures produced NaN features with CrystalNNFingerprint")
            print(f"  Inspect these structures for unusual coordination environments.")
            print(f"  Consider falling back to VoronoiFingerprint for failed structures.")

    elif featurizer_set == "density":
        feat = DensityFeatures()
        df = feat.featurize_dataframe(df, col_id="structure", ignore_errors=True)

    # Always report NaN statistics
    feature_cols = [c for c in df.columns if c != "structure"]
    nan_counts = df[feature_cols].isnull().sum()
    if nan_counts.sum() > 0:
        print(f"\nNaN summary after {featurizer_set} featurization:")
        for col in nan_counts[nan_counts > 0].index:
            print(f"  {col}: {nan_counts[col]} NaN ({nan_counts[col]/len(df):.1%})")

    return df


def add_density_and_rdf_features(
    df: pd.DataFrame,
    rdf_cutoff: float = 10.0,
    rdf_bin_size: float = 0.1,
) -> pd.DataFrame:
    """
    Append density features and radial distribution function statistics to
    an existing structure feature DataFrame.

    RDF statistics (mean peak position, peak height, integrated area in
    first and second shell) are informative for amorphous vs. crystalline
    classification and for detecting structural phase transitions.
    [EXPERT REVIEW NEEDED: RDF cutoff and bin size are system-dependent]
    """
    density_feat = DensityFeatures()
    df = density_feat.featurize_dataframe(df, col_id="structure", ignore_errors=True)

    # Note: RadialDistributionFunction in matminer computes the full g(r) vector
    # (not scalar statistics). For regression, extract summary statistics from g(r).
    rdf_feat = RadialDistributionFunction(cutoff=rdf_cutoff, bin_size=rdf_bin_size)
    rdf_results = []
    for s in df["structure"]:
        try:
            rdf_dict = rdf_feat.featurize(s)
            # rdf_dict contains {"distances": [...], "distribution": [...]}
            # Extract first-shell peak position and height
            g = np.array(rdf_dict[0]["distribution"])
            r = np.array(rdf_dict[0]["distances"])
            first_peak_idx = np.argmax(g[r < 4.0])  # First shell typically < 4 Å
            rdf_results.append({
                "rdf_first_peak_r": float(r[r < 4.0][first_peak_idx]),
                "rdf_first_peak_g": float(g[r < 4.0][first_peak_idx]),
                "rdf_mean_g": float(g.mean()),
            })
        except Exception as e:
            rdf_results.append({"rdf_first_peak_r": np.nan,
                                 "rdf_first_peak_g": np.nan,
                                 "rdf_mean_g": np.nan})

    rdf_df = pd.DataFrame(rdf_results, index=df.index)
    return pd.concat([df, rdf_df], axis=1)
```

### Workflow 4: Feature matrix assembly, preprocessing, and NaN handling

Combining multiple featurizers, scaling, and preparing the feature matrix for a downstream model.

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from pathlib import Path
import json


def assemble_feature_matrix(
    composition_df: pd.DataFrame,
    structure_df: pd.DataFrame,
    target_col: str,
    drop_cols: list[str] = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Merge composition and structure feature DataFrames into a single
    clean feature matrix X and target vector y.

    drop_cols: columns to exclude (e.g., formula, structure object, intermediate
               columns that would constitute target leakage).
    Returns (X, y) as DataFrame and Series.
    """
    # Merge on index (assumes same order and length)
    df = pd.concat(
        [composition_df.drop(columns=["composition"], errors="ignore"),
         structure_df.drop(columns=["structure"], errors="ignore")],
        axis=1,
    )

    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in assembled DataFrame. "
                       f"Available: {list(df.columns[:10])}")

    y = df.pop(target_col)

    # Drop non-feature columns
    always_drop = ["formula", "composition", "structure", "material_id",
                   "icsd_id", "source_id"]
    if drop_cols:
        always_drop.extend(drop_cols)
    df = df.drop(columns=[c for c in always_drop if c in df.columns], errors="ignore")

    # Drop columns that are not numeric
    non_numeric = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        print(f"WARNING: Dropping {len(non_numeric)} non-numeric columns: {non_numeric[:5]}...")
        df = df.drop(columns=non_numeric)

    print(f"Feature matrix: {df.shape[0]} structures × {df.shape[1]} features")
    print(f"Target: '{target_col}', range=[{y.min():.3f}, {y.max():.3f}]")
    print(f"NaN fraction (X): {df.isnull().values.mean():.3%}")

    return df, y


def build_preprocessing_pipeline(
    nan_strategy: str = "median",
    scaler: str = "robust",
    var_threshold: float = 0.0,
) -> Pipeline:
    """
    Build a sklearn preprocessing Pipeline for materials feature matrices.

    nan_strategy: "median" (robust to outliers), "mean" (standard),
                  or "constant" with fill_value=0 (for sparse feature sets).
    scaler: "robust" (RobustScaler, recommended for materials descriptors with outliers),
            "standard" (StandardScaler, assumes approximately Gaussian features).
    var_threshold: drop features with variance < threshold (after imputation).
                   0.0 drops constant features only.

    IMPORTANT: Fit this pipeline on the TRAINING SET ONLY. Never fit on test set.
    Apply the fitted pipeline to both train and test sets.
    """
    steps = [
        ("imputer", SimpleImputer(strategy=nan_strategy, keep_empty_features=True)),
        ("var_filter", VarianceThreshold(threshold=var_threshold)),
    ]
    if scaler == "robust":
        steps.append(("scaler", RobustScaler()))
    elif scaler == "standard":
        steps.append(("scaler", StandardScaler()))

    return Pipeline(steps)


def detect_leakage_risk(
    X: pd.DataFrame,
    y: pd.Series,
    correlation_threshold: float = 0.95,
) -> list[str]:
    """
    Identify features that are suspiciously highly correlated with the target.
    A correlation > 0.95 with the target warrants investigation for leakage.

    This is a heuristic check, not a definitive leakage detector.
    A high correlation is NOT proof of leakage — it may be physically legitimate.
    But it should trigger manual inspection of how the feature was computed.
    """
    feature_corrs = X.corrwith(y).abs().sort_values(ascending=False)
    high_corr = feature_corrs[feature_corrs > correlation_threshold]

    if len(high_corr) > 0:
        print(f"\nLeakage risk: {len(high_corr)} features with |corr(X_i, y)| > {correlation_threshold}:")
        for feat, corr in high_corr.head(10).items():
            print(f"  {feat}: corr = {corr:.4f}")
        print("  → Manually inspect: was this feature derived from or using the target?")
    else:
        print(f"No features with |corr| > {correlation_threshold} with target (leakage risk low)")

    return list(high_corr.index)


def save_feature_metadata(
    feature_names: list[str],
    preprocessing_pipeline: Pipeline,
    output_file: str = "feature_metadata.json",
) -> None:
    """
    Save feature names and preprocessing parameters for reproducibility.
    Required: the same feature list and preprocessing pipeline must be
    applied consistently across all future predictions. See reproducibility-workflow.
    """
    metadata = {
        "n_features_input": len(feature_names),
        "feature_names": feature_names,
        "preprocessing": {
            "nan_strategy": preprocessing_pipeline.named_steps["imputer"].strategy,
            "scaler_type": type(preprocessing_pipeline.named_steps["scaler"]).__name__,
        },
    }
    Path(output_file).write_text(json.dumps(metadata, indent=2))
    print(f"Feature metadata saved to {output_file}")
```

### Workflow 5: Train/test splitting and leakage prevention

Materials datasets have structural redundancy that makes random splits misleading. Use materials-aware splits.

```python
import numpy as np
import pandas as pd
from pymatgen.core import Structure
from sklearn.model_selection import train_test_split


def check_structural_duplicates(
    structures: list[Structure],
    soap_descs: np.ndarray,
    distance_threshold: float = 0.02,
) -> tuple[list[int], list[tuple[int, int]]]:
    """
    Detect near-duplicate structures in the dataset using SOAP distances.
    Duplicates should be resolved before any train/test split to prevent
    the same structure appearing in both sets.

    Returns (duplicate_indices, duplicate_pairs):
        duplicate_indices: indices of structures that have at least one near-duplicate.
        duplicate_pairs: list of (i, j) pairs with SOAP distance < threshold.
    """
    n = len(soap_descs)
    duplicate_pairs = []
    duplicate_indices = set()

    # O(N²) comparison — feasible for < 10,000 structures
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(soap_descs[i] - soap_descs[j])
            if d < distance_threshold:
                duplicate_pairs.append((i, j))
                duplicate_indices.add(i)
                duplicate_indices.add(j)

    if duplicate_pairs:
        print(f"Found {len(duplicate_pairs)} near-duplicate pairs "
              f"({len(duplicate_indices)} structures involved)")
        print(f"  Threshold: SOAP distance < {distance_threshold}")
        print(f"  Resolve before splitting: keep one, remove the other, or group into a single set")
    else:
        print(f"No near-duplicate structures detected (threshold={distance_threshold})")

    return list(duplicate_indices), duplicate_pairs


def prototype_aware_split(
    df: pd.DataFrame,
    target_col: str,
    prototype_col: str,
    test_fraction: float = 0.10,
    val_fraction: float = 0.10,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split a materials dataset by crystal prototype to prevent prototype leakage.
    All structures sharing a prototype are assigned to the same split.

    prototype_col: column containing prototype labels (e.g., "spacegroup_number",
        "crystal_system", "anonymous_formula" from pymatgen). Using
        "anonymous_formula" (e.g., "AB2") groups isostructural compounds.

    This split is conservative: test error reflects generalization to new
    prototype families, which is harder than new compositions within known prototypes.
    [EXPERT REVIEW NEEDED: appropriate split granularity depends on scientific question]
    """
    rng = np.random.default_rng(seed)

    prototypes = df[prototype_col].unique()
    rng.shuffle(prototypes)

    n = len(prototypes)
    n_test = max(1, int(n * test_fraction))
    n_val = max(1, int(n * val_fraction))

    test_protos = set(prototypes[:n_test])
    val_protos = set(prototypes[n_test:n_test + n_val])
    train_protos = set(prototypes[n_test + n_val:])

    train_df = df[df[prototype_col].isin(train_protos)].copy()
    val_df = df[df[prototype_col].isin(val_protos)].copy()
    test_df = df[df[prototype_col].isin(test_protos)].copy()

    print(f"Prototype-aware split (seed={seed}):")
    print(f"  Prototypes: {len(train_protos)} train / {len(val_protos)} val / "
          f"{len(test_protos)} test")
    print(f"  Structures: {len(train_df)} train / {len(val_df)} val / {len(test_df)} test")

    return train_df, val_df, test_df


def descriptor_space_split(
    soap_descs: np.ndarray,
    indices: np.ndarray,
    test_fraction: float = 0.15,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Split structures into train and test sets such that test structures are
    maximally dissimilar from the training set in SOAP descriptor space.
    This is the most conservative split — test structures are the hardest
    to predict because they are most different from training.

    Uses farthest-point sampling on the full dataset to select test structures.
    Returns (train_indices, test_indices).
    """
    rng = np.random.default_rng(seed)
    n_test = max(1, int(len(indices) * test_fraction))

    # Start from a random structure as the first test point
    test_idx = [rng.integers(len(indices))]
    remaining = list(range(len(indices)))
    remaining.pop(test_idx[0])

    while len(test_idx) < n_test:
        test_descs = soap_descs[test_idx]
        # Distance from each remaining structure to the nearest test structure
        remaining_descs = soap_descs[remaining]
        dists_to_test = np.min(
            np.linalg.norm(remaining_descs[:, None, :] - test_descs[None, :, :], axis=2),
            axis=1,
        )
        farthest = remaining[int(np.argmax(dists_to_test))]
        test_idx.append(farthest)
        remaining.remove(farthest)

    train_indices = indices[remaining]
    test_indices = indices[test_idx]

    print(f"Descriptor-space split: {len(train_indices)} train, {len(test_indices)} test")
    print(f"  Test structures are maximally diverse from training set in SOAP space")
    return train_indices, test_indices
```

### Workflow 6: Featurization for trajectories, surfaces, and amorphous structures

Applying featurizers consistently to non-equilibrium structures, MD trajectories, and disordered systems.

```python
import numpy as np
from ase.io import read
from dscribe.descriptors import SOAP
from sklearn.preprocessing import normalize


def featurize_trajectory(
    trajectory_file: str,
    species: list[str],
    r_cut: float = 5.0,
    subsample_interval: int = 10,
    average: str = "inner",
) -> np.ndarray:
    """
    Compute per-frame SOAP descriptors for an MD or AIMD trajectory.
    subsample_interval: use every N-th frame to reduce correlation.
    Returns (N_frames_subsampled, D) array.

    Usage: structure classification, change-point detection, identifying
    frames with unusual environments for active learning query.
    """
    frames = read(trajectory_file, index=f"::{subsample_interval}")
    soap = SOAP(species=species, r_cut=r_cut, n_max=9, l_max=6,
                sigma=0.5, average=average, periodic=True)
    descs = soap.create(frames)
    descs = normalize(descs, norm="l2")
    print(f"Trajectory featurization: {len(frames)} frames (every {subsample_interval}th), "
          f"descriptor shape {descs.shape}")
    return descs


def featurize_surface_slab(
    slab_atoms,            # ASE Atoms: slab with vacuum, pbc=[T,T,F] or [T,T,T]
    species: list[str],
    r_cut: float = 5.0,
    surface_depth_A: float = 4.0,
    return_per_site: bool = True,
) -> dict:
    """
    Compute SOAP descriptors for a surface slab, distinguishing surface
    and bulk-like sites by depth from the vacuum interface.

    surface_depth_A: sites within this depth of the slab surface are classified
    as "surface" sites; deeper sites are "bulk" sites.

    Returns dict with per-site descriptors and site classification.
    [EXPERT REVIEW NEEDED: surface_depth_A threshold is slab-geometry-dependent]
    """
    soap = SOAP(species=species, r_cut=r_cut, n_max=9, l_max=6,
                sigma=0.5, average="off", periodic=True)
    per_site_descs = soap.create(slab_atoms)  # (N_atoms, D)

    # Classify sites by z-coordinate
    z_coords = slab_atoms.get_positions()[:, 2]
    z_surface_top = z_coords.max()
    z_surface_bot = z_coords.min()

    is_surface = (
        (z_coords > z_surface_top - surface_depth_A) |
        (z_coords < z_surface_bot + surface_depth_A)
    )

    return {
        "all_sites": per_site_descs,
        "surface_sites": per_site_descs[is_surface],
        "bulk_sites": per_site_descs[~is_surface],
        "surface_mask": is_surface,
        "n_surface": is_surface.sum(),
        "n_bulk": (~is_surface).sum(),
    }


def check_oxidation_state_consistency(
    structures,   # list of pymatgen Structure objects
    strategy: str = "BVAnalyzer",
) -> dict:
    """
    Attempt oxidation state assignment for all structures and report
    success rate. A low success rate invalidates oxidation-state-based features.

    strategy: "BVAnalyzer" (bond valence, default), "ICSD" (ICSD-based heuristics).
    [EXPERT REVIEW NEEDED: BVAnalyzer works for oxides and simple ionic compounds;
     fails for metals, sulfides, and many intermetallics]
    """
    from pymatgen.analysis.bond_valence import BVAnalyzer

    analyzer = BVAnalyzer()
    n_success, n_fail, n_ambiguous = 0, 0, 0
    failed_indices = []

    for i, s in enumerate(structures):
        try:
            decorated = analyzer.get_oxi_state_decorated_structure(s)
            n_success += 1
        except Exception as e:
            err_str = str(e).lower()
            if "ambiguous" in err_str or "multiple" in err_str:
                n_ambiguous += 1
            else:
                n_fail += 1
            failed_indices.append(i)

    total = len(structures)
    print(f"\nOxidation state assignment ({strategy}):")
    print(f"  Success:   {n_success}/{total} ({n_success/total:.1%})")
    print(f"  Ambiguous: {n_ambiguous}/{total} ({n_ambiguous/total:.1%})")
    print(f"  Failed:    {n_fail}/{total} ({n_fail/total:.1%})")
    if (n_fail + n_ambiguous) / total > 0.10:
        print(f"  WARNING: > 10% of structures failed oxidation state assignment.")
        print(f"  Oxidation-state-dependent features will have NaN for these structures.")
        print(f"  Consider: (a) drop oxidation features, (b) impute, or (c) use a model")
        print(f"  that handles missing values natively.")

    return {
        "n_success": n_success, "n_ambiguous": n_ambiguous,
        "n_fail": n_fail, "failed_indices": failed_indices,
    }
```

## Common Failure Modes

| Failure | Symptom | Likely Cause | Diagnostic | Fix |
|---------|---------|--------------|------------|-----|
| Target leakage via correlated feature | Test R² appears near-perfect (> 0.99) for a dataset where this seems implausible; feature importance shows one dominant feature | A feature was computed using or is strongly correlated with the target; common example: including DFT total energy per atom as a feature for a model predicting formation energy | Run `detect_leakage_risk`; inspect any feature with |corr(X_i, y)| > 0.95; trace its computation back to the raw data source | Remove the leaking feature; audit the entire feature pipeline for any property that could be derived from the target; re-evaluate model performance |
| Preprocessing fitted on train+test (feature leakage) | Test error is lower than expected; model performance degrades significantly when deployed on truly unseen data | `StandardScaler` or `SimpleImputer` fit on the full dataset before splitting; feature selection using the full dataset | Audit the preprocessing pipeline: every `.fit()` or `.fit_transform()` call must only see training data; test should only see `.transform()` | Wrap all preprocessing in a sklearn `Pipeline`; call `pipe.fit(X_train)` only; apply with `pipe.transform(X_test)` |
| Inconsistent oxidation state assignment across dataset | IonProperty or CrystalNNFingerprint features show 15–30% NaN; imputed structures cluster separately from non-imputed | `OxidationStateDecorator` or `BVAnalyzer` fails for metallic or complex compositions; NaN imputed by mean introduces a systematic bias | Call `check_oxidation_state_consistency`; plot NaN-imputed vs. non-imputed structures in PCA space | For datasets with > 10% oxidation failure, drop oxidation-state-dependent features or train a separate model for structures where assignment succeeds; do not silently impute the NaN |
| Descriptor sensitivity to cell choice (conventional vs. primitive) | Two descriptions of the same crystal produce different feature vectors; model predicts different properties for the same material depending on input file origin | Matminer site featurizers produce different numbers of sites and different per-site statistics for a 4-atom conventional BCC cell vs. a 2-atom primitive cell | Standardize all structures to the same representation before featurization: `pymatgen Structure.get_primitive_structure()` or `get_conventional_standard_structure()` consistently | Choose one convention (primitive is preferred for efficiency) and apply it to all structures via `SpacegroupAnalyzer(s).get_primitive_standard_structure()` before featurization |
| SOAP cutoff truncating a coordination shell | Model predictions are insensitive to second-shell chemistry; accuracy for transition metals worse than expected; active learning selects structures that differ only in second shell | r_cut set below the second coordination shell boundary; cutoff falls in the middle of a shell, creating an artifact | Run `soap_cutoff_sensitivity_check`; plot RDF and identify shell boundaries; r_cut should be above the local minimum after the second shell | Set r_cut to a local minimum in the RDF (between coordination shells); typical: > 4.5 Å for metals, > 5.5 Å for open frameworks [EXPERT REVIEW NEEDED] |
| Near-duplicate structures in train and test sets | Test MAE is anomalously low; learning curve plateaus immediately; model does not generalize to truly novel compositions | Random train/test split applied to a dataset with many isostructural compounds or closely related polymorphs | Run `check_structural_duplicates` on the full dataset; compute pairwise SOAP distances between train and nearest test neighbor — median nearest-neighbor distance should be > 0.05 | Remove duplicates before splitting; use `prototype_aware_split` or `descriptor_space_split` for datasets with many related structures |
| NaN-heavy descriptors silently degrading imputed rows | Rows with imputed features cluster in a region of feature space that does not correspond to any real structure class; model learns to predict the imputation artifact | Imputation by mean or median fills NaN with an unphysical representative value; for features with 30%+ NaN, imputed rows dominate the distribution of imputed values | Visualize imputed vs. non-imputed rows in PCA space; if they separate cleanly, imputation is introducing a systematic artifact | Drop features with > 20% NaN; or use an explicit missing-value indicator column alongside the imputed column; or use a model that handles NaN natively (XGBoost, LightGBM) |
| PBC mishandling in molecular descriptors | SOAP for molecules in a periodic box produces wrong features when the molecule is close to the box boundary or when r_cut > half the box length | ASE Atoms for molecules has pbc=False but the periodic SOAP implementation wraps coordinates; or pbc=True with a box too small (box < 2 × r_cut) produces ghost atom artifacts | Check `atoms.pbc` before featurizing; verify box size ≥ 2 × r_cut in all directions for periodic SOAP; run forces check on a displaced molecule to verify descriptor invariance | Set `periodic=False` in SOAP for molecular systems; use a molecular box ≥ 2 × r_cut per side; or use non-periodic Coulomb matrix / Morgan fingerprint for molecules |

## Best Practices

- **Choose the featurizer class that matches the physical information relevant to the target property.** For properties dominated by local bonding (formation energy, band gap of simple materials), SOAP or Voronoi is appropriate. For properties requiring long-range information (Madelung energy, piezoelectricity in polar materials), global descriptors or learned representations are needed. Mismatched featurizer–property pairs cannot be rescued by hyperparameter tuning.
- **Always check the NaN fraction per feature before any preprocessing.** A feature with 25% NaN that is imputed by median creates a bimodal distribution: real values and imputed values. This is tolerable only if the imputed rows are physically similar to the median row. For structure featurizers that fail on unusual environments, NaN is a diagnostic, not noise — it tells you the featurizer does not understand the structure.
- **Fit all preprocessing (imputation, scaling, feature selection) exclusively on the training set.** Every `.fit()` call that uses test data inflates test performance. Use sklearn `Pipeline` and `GridSearchCV` with `fit_transform(X_train)` and `transform(X_test)` to enforce this.
- **Report SOAP descriptor hyperparameters (r_cut, n_max, l_max, sigma, average) as part of the model card.** A SOAP vector computed with r_cut=5.0 is not the same feature as r_cut=6.0; models trained on one are not transferable to the other. Include these in the `feature_metadata.json` (see `reproducibility-workflow`).
- **Standardize cell conventions before structure featurization.** Always apply either `get_primitive_standard_structure()` or `get_conventional_standard_structure()` uniformly across the entire dataset. Mixing conventions produces multiple descriptor vectors for the same crystal.
- **Use prototype-aware or descriptor-space splitting for honest model evaluation on materials datasets.** Random splitting overestimates generalization to new chemical spaces because it leaks correlated structures. The harder but more informative splits reveal whether the model extrapolates to new crystal prototypes or new composition families.
- **Document and version the exact set of feature names** alongside the model checkpoint. Feature names can change silently when matminer or pymatgen versions change their featurizer output ordering or add/remove features. See `reproducibility-workflow` for how to record this.
- **For oxidation-state features, always validate the assignment success rate before including them.** An oxide-specific featurizer (e.g., cation field strength, anion charge) trained on a mixed dataset where only 60% of structures have valid oxidation states will learn from an imputed majority that is not physically meaningful.

## Integration with Other Skills

- **[reproducibility-workflow](../reproducibility-workflow/SKILL.md):** The featurizer hyperparameters (r_cut, n_max, l_max, sigma, preset name), the exact feature names output by each featurizer version, and the preprocessing pipeline parameters must all be recorded as provenance metadata. `save_feature_metadata` in Workflow 4 is the featurization layer of the general reproducibility framework described there.
- **[dft-dataset-generation](../../electronic-structure/dft-dataset-generation/SKILL.md):** Structural descriptors (SOAP, Voronoi) drive the FPS diversity filtering in that skill's Workflow 1. The descriptor hyperparameters used for FPS must be consistent with those used for the downstream surrogate model — computing FPS with r_cut=5.0 and training a surrogate with r_cut=6.0 may include structurally redundant training data. Use the same descriptor for both steps.
- **[dft-convergence](../../electronic-structure/dft-convergence/SKILL.md):** DFT-derived features (band gap, formation energy, bulk modulus from DFT) are target variables, not input features — do not use them as features for models predicting those same quantities. However, DFT-derived structural relaxation outputs (lattice parameters, bond lengths) are valid input features for models predicting other DFT quantities computed at the same convergence level.
- **[vasp-workflow](../../electronic-structure/vasp-workflow/SKILL.md):** When VASP outputs (relaxed structures, OUTCAR-derived properties) are used as the basis for featurization, the pymatgen parser must be consistent with the VASP settings. Structures relaxed with different ENCUT or POTCAR may have systematically different bond lengths and volumes, which translate directly into different descriptor vectors.
- **[quantum-espresso](../../electronic-structure/quantum-espresso/SKILL.md):** QE-relaxed structures use Bohr or Angstrom depending on the output format; ASE handles the conversion, but verify that `atoms.cell` and `atoms.positions` are in Angstrom before passing to DScribe or matminer. A cell in Bohr produces descriptor vectors that are shifted in r-space by a factor of 1.89 relative to the expected range.
- **[mlp-validation](../../ml-interatomic-potentials/mlp-validation/SKILL.md):** SOAP descriptors are used in that skill's extrapolation detection workflow to compute distances between production MD frames and the training set. The descriptor parameters (r_cut, n_max, l_max) must be identical between the active learning FPS step, the MLP validation step, and any downstream surrogate — use a single `descriptor_config.json` shared across all pipelines.
- **[active-learning-mlp](../../ml-interatomic-potentials/active-learning-mlp/SKILL.md):** Descriptor-space diversity filtering (FPS with SOAP) is the primary redundancy-reduction step in that skill's Workflow 1. The SOAP cutoff must be at or above the second coordination shell to capture the full local environment that the MLP sees. Using a shorter cutoff for FPS than for the MLP training descriptor is inconsistent and may select redundant structures.
- **[surrogate-validation](../../surrogate-active-learning/surrogate-validation/SKILL.md):** Surrogate model validation requires knowing which test structures are interpolation vs. extrapolation cases. The descriptor-space distances computed here (SOAP Euclidean distance to the nearest training structure) quantify extrapolation for any fixed-descriptor surrogate model. A test structure with no training neighbor within distance δ is an extrapolation candidate regardless of the model's predicted uncertainty.
- **[gaussian-process-materials](../../surrogate-active-learning/gaussian-process-materials/SKILL.md):** Gaussian processes use the descriptor as the kernel input — the kernel function is the inner product or distance in descriptor space. The choice of descriptor defines the GP's inductive bias. SOAP-based kernels (dot-product kernel on normalized SOAP vectors) are a physically motivated choice for local property prediction; Matminer features with an RBF kernel are appropriate for composition-based screening where interpretability matters.
- **[crystal-property-gnn](../../geometric-deep-learning/crystal-property-gnn/SKILL.md):** Graph neural networks jointly learn the featurization and the regression; they do not use a fixed descriptor. However, the same train/test leakage considerations apply: prototype-aware or composition-aware splits are required for honest evaluation. Fixed descriptors (SOAP, Voronoi) are useful as baselines against which GNN accuracy should be compared before committing to the more complex model.
- **[molecular-property-gnn](../../geometric-deep-learning/molecular-property-gnn/SKILL.md):** For molecular systems, the featurization choice is between graph-based (atom-bond graph, as in GNNs), topology-based (Morgan/ECFP fingerprints from RDKit), geometry-based (SOAP, Coulomb matrix), or learned (e.g., ChemBERTa from SMILES). RDKit fingerprints are the most widely used fixed descriptors for drug-like molecules; SOAP is preferred for conformer-sensitive properties where 3D geometry matters.

## Key References

1. Ward, L., et al. (2016). *A general-purpose machine learning framework for predicting properties of inorganic materials.* npj Computational Materials, 2, 16028. https://doi.org/10.1038/npjcompumats.2016.28
2. Bartók, A. P., et al. (2013). *On representing chemical environments.* Physical Review B, 87(18), 184115. https://doi.org/10.1103/PhysRevB.87.184115
3. Himanen, L., et al. (2020). *DScribe: Library of descriptors for machine learning in materials science.* Computer Physics Communications, 247, 106949. https://doi.org/10.1016/j.cpc.2019.106949
4. Dunn, A., et al. (2020). *Benchmarking materials property prediction methods: the Matbench test set and Automatminer reference algorithm.* npj Computational Materials, 6, 138. https://doi.org/10.1038/s41524-020-00406-3
5. De, S., et al. (2016). *Comparing molecules and solids across structural and alchemical space.* Physical Chemistry Chemical Physics, 18(20), 13754–13769. https://doi.org/10.1039/C6CP00415F
