---
name: uncertainty-mlp
domain: ml-interatomic-potentials
maturity: draft
tools:
  - MACE
  - ASE
  - PyTorch
  - LAMMPS
  - matplotlib
tested_versions:
  MACE: "0.3.6"
  ASE: "3.23.0"
  PyTorch: "2.2.0"
  LAMMPS: "2Aug2023"
  matplotlib: "3.8.0"
requires_gpu: true
estimated_runtime: hours-to-days
---

# Uncertainty Quantification for ML Interatomic Potentials

## Description

This skill covers practical uncertainty quantification (UQ) for machine-learned interatomic potentials: committee (ensemble) models, model-deviation computation for forces, energies, and stresses, calibration of uncertainty thresholds against DFT errors, SOAP-based structural dissimilarity as a complementary out-of-distribution (OOD) detector, conformal prediction for rigorous confidence intervals, uncertainty propagation through MD trajectories, and stopping/extrapolation criteria. Invoke this skill when designing the active learning acquisition strategy for a new MLP campaign, when diagnosing silent extrapolation in production MD, or when quantifying the reliability of MLP predictions for properties that depend on the model's generalization behavior.

## Domain Context

An MLP evaluated on a structure outside its training distribution will produce incorrect forces with no error message and no indication of failure. The only way to detect this before a catastrophic MD trajectory is through an uncertainty estimate that flags the structure as novel relative to the training set. UQ for MLPs is therefore a safety mechanism, not a statistical nicety.

**Epistemic versus aleatoric uncertainty.** Epistemic uncertainty arises from insufficient training data: the model has not seen enough structures to constrain its predictions in a region of configuration space. This type of uncertainty can be reduced by adding more training data — it is the signal that drives active learning. Aleatoric uncertainty arises from irreducible noise in the labels (DFT numerical noise, finite k-mesh errors, SCF threshold effects). For well-converged DFT labels (EDIFF=1e-6, PREC=Accurate), aleatoric uncertainty is typically < 1 meV/Å for forces and is negligible relative to model extrapolation errors. In practice, committee model disagreement primarily measures epistemic uncertainty, but cannot cleanly separate it from aleatoric contributions.

**The committee model approach.** The dominant practical UQ method for neural network potentials is the ensemble (committee): N independently trained models with identical architecture but different random seeds or different training data subsets. The standard deviation of their force predictions at a given structure, σ_F, is the uncertainty proxy. Structures with high σ_F are candidates for DFT labeling in the active learning loop. Structures with catastrophically high σ_F are likely extrapolation cases where at least one committee member has produced garbage forces — these should be inspected before being sent to DFT.

**What committee models do not detect.** The committee approach has a fundamental blind spot: if all committee members agree on the same wrong answer, σ_F is low and the structure passes the confidence filter. This happens in two well-known failure modes. First, if the committee was trained on the same data with only different random seeds, the members are all trained to the same DFT reference surface and will agree on any structure that is similar to the training distribution but whose correct DFT energy is systematically different from what the training surface predicts (e.g., a structure with a subtle structural distortion not covered by training). Second, for universal potentials (MACE-MP-0, CHGNet), a "committee" of fine-tuned versions of the same foundation model will share the same base representation and will not disagree enough in regions near the fine-tuning data. In both cases, SOAP-based structural dissimilarity is a complementary check that can flag structures as novel even when the committee is falsely confident.

**Calibration.** A committee model that reports σ_F = 0.10 eV/Å is only useful if that number is interpretable: does it mean the true force error is likely < 0.10 eV/Å, or < 1.0 eV/Å? Calibration maps σ_F (or any other uncertainty proxy) to actual DFT force errors on a held-out validation set. A calibrated uncertainty estimate satisfies: the true error is below the predicted bound with the stated probability. Uncalibrated thresholds (set by intuition rather than by reference to DFT errors on the same system) routinely cause either excessive DFT queries (σ_low set too low) or silent extrapolation (σ_high set too high).

**Uncertainty in production MD.** A structure visited in production MD may have any uncertainty level. Structures with σ_F > σ_high should halt or restart the MD trajectory; structures with σ_low < σ_F < σ_high are candidates for active learning; structures with σ_F < σ_low are confident interpolation. Continuously monitoring σ_F during MD adds computational overhead (each step requires N force evaluations instead of 1), which is why fast committee evaluation and selective monitoring strategies are important in practice. [EXPERT REVIEW NEEDED: overhead tolerance depends on system size and committee size]

## When to Use This Skill

- Designing the active learning query strategy: computing σ_F on exploration MD frames to select which structures to label with DFT.
- Diagnosing whether a production MD trajectory is extrapolating before or after the trajectory diverges.
- Deciding whether to deploy a system-specific MLP or a universal potential for a new chemical system, by checking whether the universal potential's training distribution covers the target.
- Quantifying MLP prediction reliability for a property report or paper: committee variance provides an estimate of how much the prediction would change if the training data were different.
- Detecting the onset of phase transitions or structural transformations in MD where the model begins sampling uncharted configuration space.

## When Not to Use This Skill

- **As a substitute for `mlp-validation`.** Committee uncertainty and structural dissimilarity are indicators of likely error, not measurements of actual error. The validation trinity (static test-set errors, physical property reproduction, MD stability) from that skill is required to characterize actual accuracy. A low σ_F does not certify the model is correct — it certifies the committee is consistent.
- **For aleatoric noise estimation.** If the goal is to quantify DFT numerical uncertainty (label noise), that requires repeating DFT calculations with perturbed settings, not running a committee. Committee variance is not a reliable estimate of DFT label noise.
- **For universal potentials without fine-tuning.** A committee of frozen universal potential checkpoints (e.g., multiple MACE-MP-0 release versions) spans model-version uncertainty, not the uncertainty of the potential for a specific chemical system. This is useful for benchmarking but is not the same as epistemic uncertainty relative to the target DFT level.
- **When the computational budget for N×inference is prohibitive.** For very large systems (> 100,000 atoms) where a single MLP evaluation costs hours, running a 4-member committee is impractical. Use single-model last-layer uncertainty, latent-space distance, or SOAP dissimilarity instead. [EXPERT REVIEW NEEDED]

## Core Concepts

- **Committee model (deep ensemble):** N independently trained MLP instances with identical architecture but different random seeds. Each member sees the same training data but converges to a different local minimum of the loss surface. The spread of their predictions measures epistemic uncertainty. Minimum committee size: N=4; recommended: N=8 for well-calibrated estimates. Training N models multiplies training cost by N.
- **Model deviation (σ_F):** The standard deviation of per-component force predictions across committee members at a given structure. Computed per atom as the Frobenius norm of the force standard deviation tensor. Aggregated to structure level as the maximum or mean per-atom σ_F. Typical units: eV/Å. Related but distinct from `σ_E` (energy deviation) and `σ_S` (stress deviation).
- **Uncertainty thresholds (σ_low, σ_high):** The two-tier classification scheme. Structures with σ_F < σ_low are well-converged (confident interpolation); the MD trajectory is continued without DFT labeling. Structures with σ_low ≤ σ_F < σ_high are candidates for active learning. Structures with σ_F ≥ σ_high are likely extrapolation or unphysical; the MD step is rejected or the structure is flagged. Thresholds must be calibrated against actual DFT errors on a validation set, not set by intuition. [EXPERT REVIEW NEEDED: typical ranges are 0.05–0.20 eV/Å for σ_low and 0.50–2.0 eV/Å for σ_high for metallic systems]
- **Calibration:** The process of mapping σ_F to actual DFT force errors using a held-out validation set with known DFT labels. A calibrated committee satisfies: P(|F_DFT - F̄_committee| < α · σ_F) ≥ 1 - δ, where α and δ are calibration parameters. Calibration curves plot σ_F against actual error for validation set structures.
- **SOAP dissimilarity:** Euclidean distance between a structure's SOAP descriptor and the nearest training set structure in SOAP space. A complementary OOD detector that is nonzero even when the committee unanimously agrees on a wrong answer. Requires a precomputed SOAP descriptor database of the training set.
- **Coverage radius (δ_cover):** The maximum SOAP distance from a training structure within which the MLP is expected to interpolate reliably. Estimated from the validation set: δ_cover ≈ max SOAP distance to nearest training neighbor across validation structures that pass all MLP validation checks.
- **Conformal prediction:** A distribution-free framework for constructing prediction intervals with rigorous marginal coverage guarantees. For a calibration set of size n, conformal intervals contain the true value with probability ≥ 1 - α on exchangeable test points. Applied to forces: fit a nonconformity score (e.g., committee σ_F normalized by magnitude), find the 1-α quantile on the calibration set, use this as the force interval half-width for test structures.
- **Per-atom vs. structure-level uncertainty:** Committee σ_F is computed per atom. Structure-level aggregation options: maximum (worst-case; most conservative for active learning), mean (average environment uncertainty), 90th percentile (robust to single-atom outliers). Maximum aggregation is standard for stopping/extrapolation criteria; mean is more appropriate for diversity-weighted active learning queries.
- **Energy deviation (σ_E):** Per-atom energy standard deviation across committee. Less sensitive to local environment changes than σ_F; useful for detecting systematic PES curvature disagreements that affect thermodynamic properties without triggering large force errors. Monitor alongside σ_F for NPT MD validation.
- **Ensemble collapse:** The failure mode where committee members converge to nearly identical models despite different seeds. Occurs when the training set is small and homogeneous, the loss landscape has a deep and narrow global minimum, or the optimizer is deterministic enough to overcome seed diversity. Detected by checking that committee pairwise test-set force MAE differences are > 2× the individual member's test-set noise floor.

## Key Workflows

### Workflow 1: Training a MACE committee and verifying member diversity

A properly diverse committee is the foundation of all downstream UQ. Members trained with different seeds on the same dataset will differ only if the training is genuinely stochastic and the loss landscape has multiple distinct minima.

```python
import subprocess
import hashlib
import json
import numpy as np
from pathlib import Path
from ase.io import read
from mace.calculators import MACECalculator


def train_committee(
    train_file: str,
    val_file: str,
    n_members: int = 8,
    base_seed: int = 100,
    model_prefix: str = "committee",
    mace_kwargs: dict = None,
    device: str = "cuda",
) -> list[str]:
    """
    Train N committee members with identical hyperparameters and different seeds.

    n_members: 4 is a minimum; 8 gives better-calibrated σ_F estimates.
        Training cost scales linearly with n_members.
    base_seed: seeds are base_seed + i*17 (separated to avoid correlated RNG sequences).
    Returns: list of trained model file paths.
    """
    if mace_kwargs is None:
        # Example: W-Ta alloy, metallic system
        mace_kwargs = {
            "model": "MACE",
            "r_max": 5.5,
            "correlation": 3,
            "num_interactions": 2,
            "hidden_irreps": "128x0e + 128x1o + 128x2e",
            "max_ell": 3,
            "energy_weight": 1.0,
            "forces_weight": 10.0,
            "stress_weight": 1.0,
            "batch_size": 8,
            "lr": 0.01,
            "max_num_epochs": 1500,
            "patience": 250,
            "ema": True,
            "ema_decay": 0.99,
            "scaling": "rms_forces_scaling",
            "default_dtype": "float32",
            "E0s": "average",
        }

    model_paths = []
    for i in range(n_members):
        seed = base_seed + i * 17
        name = f"{model_prefix}_s{seed}"
        cmd = (
            ["mace_run_train",
             f"--name={name}",
             f"--train_file={train_file}",
             f"--valid_file={val_file}",
             f"--device={device}",
             f"--seed={seed}"]
            + [f"--{k}={str(v).lower() if isinstance(v, bool) else v}"
               for k, v in mace_kwargs.items()]
        )
        print(f"Training member {i+1}/{n_members} (seed={seed})...")
        subprocess.run(cmd, check=True)

        # MACE writes: {name}_run-{seed}_stagetwo.model
        model_file = f"{name}_run-{seed}_stagetwo.model"
        if not Path(model_file).exists():
            raise FileNotFoundError(
                f"Expected model file {model_file} not found after training. "
                f"Check MACE log for errors."
            )
        model_paths.append(model_file)

    # Record committee manifest for reproducibility
    manifest = {
        "n_members": n_members,
        "model_paths": model_paths,
        "model_sha256": {
            mp: hashlib.sha256(open(mp, "rb").read()).hexdigest()
            for mp in model_paths
        },
        "train_file": train_file,
        "train_sha256": hashlib.sha256(open(train_file, "rb").read()).hexdigest(),
        "mace_kwargs": {k: str(v) for k, v in mace_kwargs.items()},
        "seeds": [base_seed + i * 17 for i in range(n_members)],
    }
    Path(f"{model_prefix}_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Committee manifest written to {model_prefix}_manifest.json")
    return model_paths


def verify_committee_diversity(
    model_paths: list[str],
    test_file: str,
    min_pairwise_force_mae_meV: float = 5.0,
    device: str = "cpu",
) -> bool:
    """
    Check that committee members are genuinely diverse by computing pairwise
    force MAE differences on the test set. If all pairs agree too closely,
    the committee has collapsed and σ_F will underestimate true uncertainty.

    min_pairwise_force_mae_meV: minimum acceptable mean pairwise force MAE
        difference (meV/Å). If < this threshold, the committee may be collapsed.
        [EXPERT REVIEW NEEDED: system-dependent threshold]
    """
    calcs = [
        MACECalculator(model_paths=mp, device=device, default_dtype="float64")
        for mp in model_paths
    ]
    frames = read(test_file, index=":100", format="extxyz")  # First 100 frames
    all_forces = []  # (n_members, n_frames, n_atoms, 3) — ragged; stored as list

    for calc in calcs:
        member_forces = []
        for atoms in frames:
            a = atoms.copy()
            a.calc = calc
            member_forces.append(a.get_forces())
        all_forces.append(member_forces)

    # Pairwise MAE between committee members
    n = len(model_paths)
    pairwise_maes = []
    for i in range(n):
        for j in range(i + 1, n):
            diffs = []
            for fi, fj in zip(all_forces[i], all_forces[j]):
                diffs.append(np.abs(fi - fj).mean() * 1000)  # meV/Å
            pairwise_maes.append(np.mean(diffs))

    mean_pairwise = np.mean(pairwise_maes)
    print(f"\nCommittee diversity check:")
    print(f"  Mean pairwise force MAE: {mean_pairwise:.2f} meV/Å")
    print(f"  Min pairwise force MAE: {np.min(pairwise_maes):.2f} meV/Å")
    print(f"  Max pairwise force MAE: {np.max(pairwise_maes):.2f} meV/Å")

    if mean_pairwise < min_pairwise_force_mae_meV:
        print(f"  WARNING: Pairwise MAE < {min_pairwise_force_mae_meV} meV/Å — "
              f"committee may have collapsed. Diversity is insufficient for reliable UQ.")
        print(f"  Possible causes: training set too small/homogeneous; "
              f"optimizer overcoming seed diversity; epochs too many (over-converged).")
        return False

    print(f"  PASS: Committee members are sufficiently diverse.")
    return True
```

### Workflow 2: Per-frame force, energy, and stress model deviation

Computing σ_F, σ_E, and σ_S efficiently during MD or for a batch of candidate structures. The aggregation from atom-level to structure-level is a critical design choice.

```python
import numpy as np
from ase import Atoms
from mace.calculators import MACECalculator


class CommitteeCalculator:
    """
    Wraps a list of MACECalculators to provide per-frame uncertainty estimates.
    Computes the mean prediction (used as the MLP prediction) and the
    standard deviation (used as the uncertainty estimate).
    """

    def __init__(
        self,
        model_paths: list[str],
        device: str = "cpu",
        dtype: str = "float32",
    ):
        self.calcs = [
            MACECalculator(model_paths=mp, device=device, default_dtype=dtype)
            for mp in model_paths
        ]
        self.n_members = len(model_paths)

    def compute(self, atoms: Atoms) -> dict:
        """
        Evaluate all committee members on a single structure.
        Returns a dict with mean predictions and per-quantity deviations.
        """
        energies, forces_list, stresses = [], [], []

        for calc in self.calcs:
            a = atoms.copy()
            a.calc = calc
            energies.append(a.get_potential_energy())
            forces_list.append(a.get_forces())           # (N_atoms, 3)
            s = a.get_stress(voigt=True)
            if s is not None:
                stresses.append(s)

        n_atoms = len(atoms)
        F = np.stack(forces_list, axis=0)                # (N_committee, N_atoms, 3)

        # ─── Force deviation ──────────────────────────────────────────────────
        # Per-atom: norm of the per-component std vector
        sigma_F_per_atom = np.linalg.norm(F.std(axis=0), axis=1)  # (N_atoms,) eV/Å

        # Structure-level aggregations
        sigma_F_max  = sigma_F_per_atom.max()
        sigma_F_mean = sigma_F_per_atom.mean()
        sigma_F_p90  = np.percentile(sigma_F_per_atom, 90)

        # ─── Energy deviation ─────────────────────────────────────────────────
        E_arr = np.array(energies)
        sigma_E_per_atom = E_arr.std() / n_atoms  # eV/atom

        # ─── Stress deviation ─────────────────────────────────────────────────
        sigma_S = None
        if len(stresses) == self.n_members:
            S = np.stack(stresses, axis=0)              # (N_committee, 6)
            sigma_S_GPa = S.std(axis=0) * 160.218       # eV/Å³ → GPa
            sigma_S = float(sigma_S_GPa.max())

        return {
            # Mean predictions
            "energy_eV": float(np.mean(energies)),
            "forces_eV_A": F.mean(axis=0),               # (N_atoms, 3)
            # Per-atom uncertainty
            "sigma_F_per_atom_eV_A": sigma_F_per_atom,   # (N_atoms,) eV/Å
            # Structure-level uncertainty aggregations
            "sigma_F_max_eV_A":  sigma_F_max,
            "sigma_F_mean_eV_A": sigma_F_mean,
            "sigma_F_p90_eV_A":  sigma_F_p90,
            "sigma_E_per_atom_eV": sigma_E_per_atom,
            "sigma_S_max_GPa": sigma_S,
            # Metadata
            "n_atoms": n_atoms,
            "config_type": atoms.info.get("config_type", "unknown"),
        }

    def batch_compute(self, frames: list[Atoms]) -> list[dict]:
        """Evaluate committee on a list of structures."""
        return [self.compute(atoms) for atoms in frames]

    def classify(
        self,
        result: dict,
        sigma_low: float = 0.05,
        sigma_high: float = 0.50,
        aggregation: str = "max",
    ) -> str:
        """
        Classify a structure by uncertainty tier using the chosen aggregation.
        aggregation: "max" (conservative) | "mean" | "p90" (robust to outliers)
        Returns: "confident" | "query" | "extrapolation"
        """
        key = f"sigma_F_{aggregation}_eV_A"
        if key not in result:
            raise KeyError(f"Aggregation '{aggregation}' not in result dict. "
                           f"Use 'max', 'mean', or 'p90'.")
        sigma = result[key]

        if sigma < sigma_low:
            return "confident"
        elif sigma < sigma_high:
            return "query"
        else:
            return "extrapolation"


def separate_high_force_atoms(
    result: dict,
    force_magnitude_threshold_eV_A: float = 5.0,
) -> dict:
    """
    Separate atoms with genuinely high uncertainty from atoms that happen
    to have large force magnitudes. A high-force atom (e.g., an atom
    compressed close to a neighbor) typically has high per-atom σ_F simply
    because forces scale with the force magnitude — this is not a sign that
    the model is uncertain about the environment, only that it disagrees in
    proportion to the forces themselves.

    Returns dict with per-atom classifications: "high_force", "high_uncertainty",
    "both", or "normal".
    """
    forces_mean = result["forces_eV_A"]           # (N, 3)
    sigma_F = result["sigma_F_per_atom_eV_A"]     # (N,)

    force_magnitudes = np.linalg.norm(forces_mean, axis=1)  # (N,)
    # Normalize: relative uncertainty = σ_F / |F|
    # High absolute σ_F on a high-|F| atom is expected; flag only high *relative* uncertainty
    # Use a floor to avoid division-by-zero on near-zero forces
    relative_sigma = sigma_F / np.maximum(force_magnitudes, 0.1)  # dimensionless

    is_high_force = force_magnitudes > force_magnitude_threshold_eV_A
    is_high_rel_uncertainty = relative_sigma > 0.5  # > 50% relative std [EXPERT REVIEW NEEDED]

    labels = np.where(
        is_high_force & is_high_rel_uncertainty, "both",
        np.where(is_high_force, "high_force",
                 np.where(is_high_rel_uncertainty, "high_uncertainty", "normal"))
    )

    n_genuine = (labels == "high_uncertainty").sum() + (labels == "both").sum()
    n_trivial = (labels == "high_force").sum()

    if n_trivial > 0 and n_genuine == 0:
        print(f"  {n_trivial} atoms have high σ_F purely due to large |F|. "
              f"Structure-level σ_F is inflated but does not indicate genuine uncertainty.")
    elif n_genuine > 0:
        print(f"  {n_genuine} atoms show genuinely high relative uncertainty "
              f"(σ_F/|F| > 0.5) independent of force magnitude.")

    return {
        "labels": labels,
        "force_magnitudes": force_magnitudes,
        "relative_sigma": relative_sigma,
        "n_high_force": int(is_high_force.sum()),
        "n_high_rel_uncertainty": int(is_high_rel_uncertainty.sum()),
    }
```

### Workflow 3: Calibrating σ_F thresholds against actual DFT errors

Thresholds that are not calibrated against the actual DFT force errors on the target system are guesses. This workflow uses the held-out test set (with known DFT labels) to fit calibration parameters.

```python
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ase.io import read


def calibrate_uncertainty_thresholds(
    committee: "CommitteeCalculator",
    test_file: str,
    aggregation: str = "max",
    target_coverage: float = 0.95,
    output_plot: str = "calibration_curve.png",
) -> dict:
    """
    Fit calibration between σ_F and actual |F_MLP - F_DFT| on a held-out test set.

    For each test structure:
    - Compute σ_F (committee model deviation) using the chosen aggregation
    - Compute actual force MAE vs. DFT reference labels

    Outputs:
    - Calibration plot: σ_F vs actual F_MAE with a 1:1 line for reference
    - Recommended σ_low, σ_high thresholds
    - Miscalibration factor α: actual error ≈ α × σ_F on average

    target_coverage: desired fraction of structures for which σ_F ≥ actual error.
        0.95 → 95% of structures have actual force error ≤ fitted bound.
    """
    frames = read(test_file, index=":", format="extxyz")
    results = committee.batch_compute(frames)

    sigma_F_vals = []
    actual_F_maes = []

    for atoms, result in zip(frames, results):
        ref_forces = atoms.arrays.get("forces")
        if ref_forces is None:
            continue

        pred_forces = result["forces_eV_A"]
        actual_mae = np.abs(pred_forces - ref_forces).mean() * 1000  # meV/Å
        sigma_meV = result[f"sigma_F_{aggregation}_eV_A"] * 1000     # meV/Å

        sigma_F_vals.append(sigma_meV)
        actual_F_maes.append(actual_mae)

    sigma_arr = np.array(sigma_F_vals)
    error_arr = np.array(actual_F_maes)

    # ─── Calibration: find α such that σ_F × α ≥ actual error in target_coverage fraction ───
    # For each test structure, compute the ratio actual_error / σ_F.
    # The target_coverage quantile of these ratios is the calibration factor α.
    safe_sigma = np.maximum(sigma_arr, 0.1)  # avoid div/0
    ratios = error_arr / safe_sigma
    alpha = np.quantile(ratios, target_coverage)

    # ─── Recommend thresholds ────────────────────────────────────────────────
    # σ_low: below this, the model is confident (expected error < 20 meV/Å for metals)
    # σ_high: above this, the model is extrapolating (expected error > 200 meV/Å)
    # These are physical targets; adjust for the system and property of interest.
    force_target_low_meV  = 20.0   # eV/Å; "confident" threshold
    force_target_high_meV = 200.0  # meV/Å; "extrapolation" threshold
    sigma_low_rec  = force_target_low_meV  / alpha
    sigma_high_rec = force_target_high_meV / alpha

    print(f"\nCalibration results (aggregation='{aggregation}', coverage={target_coverage:.0%}):")
    print(f"  Calibration factor α = {alpha:.3f}  "
          f"(actual error ≤ α × σ_F with probability {target_coverage:.0%})")
    print(f"  Recommended σ_low  = {sigma_low_rec:.2f} meV/Å  "
          f"(expected actual F error < {force_target_low_meV:.0f} meV/Å)")
    print(f"  Recommended σ_high = {sigma_high_rec:.2f} meV/Å  "
          f"(expected actual F error < {force_target_high_meV:.0f} meV/Å)")
    print(f"  NOTE: Thresholds are in meV/Å for committee σ_F × 1000")

    # ─── Calibration plot ────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.scatter(sigma_arr, error_arr, alpha=0.4, s=10, color="steelblue")
    lim = max(sigma_arr.max(), error_arr.max()) * 1.1
    ax.plot([0, lim], [0, lim], "k--", lw=1, label="σ = error (ideal)")
    ax.plot([0, lim], [0, lim * alpha], "r-", lw=1.5,
            label=f"α·σ = error ({target_coverage:.0%} coverage)")
    ax.axvline(sigma_low_rec,  color="green", ls=":", lw=1.5, label=f"σ_low={sigma_low_rec:.1f}")
    ax.axvline(sigma_high_rec, color="orange", ls=":", lw=1.5, label=f"σ_high={sigma_high_rec:.1f}")
    ax.set_xlabel(f"Committee σ_F ({aggregation}) [meV/Å]")
    ax.set_ylabel("Actual F MAE vs DFT [meV/Å]")
    ax.set_title("Calibration: σ_F vs. actual force error")
    ax.legend(fontsize=8)
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)

    # Reliability diagram: coverage fraction as a function of α·σ bound
    ax2 = axes[1]
    alphas = np.linspace(0.1, 3.0, 50)
    coverages = [(error_arr <= a * safe_sigma).mean() for a in alphas]
    ax2.plot(alphas, coverages, "b-")
    ax2.axhline(target_coverage, color="r", ls="--", label=f"Target {target_coverage:.0%}")
    ax2.axvline(alpha, color="r", ls=":", label=f"α={alpha:.2f}")
    ax2.set_xlabel("α (multiplier on σ_F)")
    ax2.set_ylabel("Fraction of structures with error ≤ α·σ_F")
    ax2.set_title("Coverage vs. confidence multiplier")
    ax2.legend()
    ax2.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(output_plot, dpi=150)
    print(f"Calibration plot saved to {output_plot}")

    return {
        "alpha": float(alpha),
        "sigma_low_meV_A": float(sigma_low_rec),
        "sigma_high_meV_A": float(sigma_high_rec),
        "n_test_structures": len(sigma_arr),
        "median_sigma_F_meV": float(np.median(sigma_arr)),
        "median_actual_F_mae_meV": float(np.median(error_arr)),
    }
```

### Workflow 4: SOAP-based structural dissimilarity as complementary OOD detector

Committee variance is blind to confident-but-wrong extrapolation. SOAP dissimilarity catches structures that are geometrically novel relative to the training set, even if the committee happens to agree.

```python
import numpy as np
from dscribe.descriptors import SOAP
from sklearn.preprocessing import normalize
from ase.io import read


class SOAPDissimilarityDetector:
    """
    Detects out-of-distribution structures by measuring their SOAP descriptor
    distance to the nearest training set structure. Uses an approximate
    nearest-neighbor approach for large training sets.

    Complements committee σ_F: catches cases where the committee collapses
    to a wrong-but-consistent answer on geometrically novel structures.
    """

    def __init__(
        self,
        species: list[str],
        r_cut: float = 5.0,
        n_max: int = 9,
        l_max: int = 6,
        sigma: float = 0.5,
    ):
        self.soap = SOAP(
            species=species, r_cut=r_cut, n_max=n_max, l_max=l_max,
            sigma=sigma, average="inner", periodic=True,
        )
        self.train_descs = None    # (N_train, D) normalized SOAP vectors
        self.coverage_radius = None

    def fit(self, train_frames: list, val_frames: list = None) -> None:
        """
        Compute and store SOAP descriptors for the training set.
        Optionally fit coverage_radius from a validation set.

        val_frames: held-out frames that are known to be within the training
            distribution (passed mlp-validation). The 95th percentile of their
            SOAP distances to training neighbors is the coverage_radius.
        """
        print(f"Building SOAP descriptor database for {len(train_frames)} training structures...")
        descs = self.soap.create(train_frames)
        self.train_descs = normalize(descs, norm="l2")
        print(f"  SOAP descriptor shape: {self.train_descs.shape}")

        if val_frames is not None:
            val_descs = normalize(self.soap.create(val_frames), norm="l2")
            dists = self._min_distances(val_descs)
            self.coverage_radius = float(np.percentile(dists, 95))
            print(f"  Coverage radius (95th pct of val distances): "
                  f"{self.coverage_radius:.4f}")
        else:
            # Conservative default; must be calibrated for the system [EXPERT REVIEW NEEDED]
            self.coverage_radius = 0.10
            print(f"  Coverage radius not calibrated (no val_frames); "
                  f"using default {self.coverage_radius}")

    def _min_distances(self, query_descs: np.ndarray) -> np.ndarray:
        """
        Brute-force minimum distance from each query descriptor to training set.
        For large training sets (> 10,000), use approximate methods (FAISS, AnnoyIndex).
        [EXPERT REVIEW NEEDED: use approximate nearest-neighbor for large datasets]
        """
        # dot product between normalized vectors = cosine similarity
        # Euclidean distance = sqrt(2 - 2 * cos_sim) for unit vectors
        cos_sim = query_descs @ self.train_descs.T   # (N_query, N_train)
        cos_sim = np.clip(cos_sim, -1, 1)
        distances = np.sqrt(np.maximum(0, 2 - 2 * cos_sim))  # (N_query, N_train)
        return distances.min(axis=1)                           # (N_query,)

    def score(self, frames: list) -> np.ndarray:
        """
        Return SOAP dissimilarity (distance to nearest training neighbor)
        for each structure in frames. Higher = more out-of-distribution.
        """
        if self.train_descs is None:
            raise RuntimeError("Call fit() before score()")
        descs = normalize(self.soap.create(frames), norm="l2")
        return self._min_distances(descs)

    def classify(
        self, frames: list, coverage_radius: float = None
    ) -> list[str]:
        """
        Classify structures as 'interpolation' or 'extrapolation' based on
        SOAP distance to the training set.
        """
        radius = coverage_radius or self.coverage_radius
        dists = self.score(frames)
        return [
            "interpolation" if d <= radius else "extrapolation"
            for d in dists
        ]


def combined_uncertainty_report(
    frames: list,
    committee_results: list[dict],
    soap_dists: np.ndarray,
    sigma_low: float,
    sigma_high: float,
    coverage_radius: float,
) -> None:
    """
    Report structures where committee and SOAP disagree: these are the most
    dangerous cases — the committee is falsely confident about a novel structure.
    """
    n_committee_confident_but_ood = 0
    n_committee_uncertain_but_familiar = 0
    for i, (result, dist) in enumerate(zip(committee_results, soap_dists)):
        sigma = result["sigma_F_max_eV_A"]
        committee_ok = sigma < sigma_low
        soap_ok = dist <= coverage_radius

        if committee_ok and not soap_ok:
            n_committee_confident_but_ood += 1
            print(f"  WARN frame {i}: committee confident (σ={sigma*1000:.1f} meV/Å) "
                  f"but SOAP OOD (dist={dist:.4f} > r={coverage_radius:.4f}). "
                  f"Possible ensemble collapse on novel geometry.")
        elif not committee_ok and soap_ok:
            n_committee_uncertain_but_familiar += 1

    print(f"\nCombined UQ report:")
    print(f"  {n_committee_confident_but_ood} structures: committee confident but SOAP OOD "
          f"→ potential silent extrapolation (CRITICAL)")
    print(f"  {n_committee_uncertain_but_familiar} structures: committee uncertain but "
          f"SOAP familiar → ensemble variance in a known geometry family")
```

### Workflow 5: Conformal prediction for rigorous per-structure force intervals

Conformal prediction produces intervals with guaranteed marginal coverage without distributional assumptions. Applicable to force prediction as a post-hoc wrapper around any MLP.

```python
import numpy as np
from ase.io import read


def fit_conformal_scores(
    committee: "CommitteeCalculator",
    calibration_file: str,
    aggregation: str = "max",
) -> np.ndarray:
    """
    Fit nonconformity scores on a calibration set.
    Nonconformity score: actual force MAE / committee σ_F.
    A high score means the committee σ_F underestimated the true error.

    Returns sorted nonconformity scores for use in conformal prediction.
    The calibration set must be SEPARATE from the training set — it must not
    have been used for model selection or hyperparameter tuning.
    """
    cal_frames = read(calibration_file, index=":", format="extxyz")
    results = committee.batch_compute(cal_frames)

    scores = []
    for atoms, result in zip(cal_frames, results):
        ref_forces = atoms.arrays.get("forces")
        if ref_forces is None:
            continue

        pred_forces = result["forces_eV_A"]
        actual_mae = np.abs(pred_forces - ref_forces).mean()   # eV/Å
        sigma = result[f"sigma_F_{aggregation}_eV_A"]
        score = actual_mae / max(sigma, 1e-6)                  # dimensionless
        scores.append(score)

    scores = np.array(scores)
    print(f"Conformal calibration: {len(scores)} structures")
    print(f"  Nonconformity score quantiles:")
    for q in [0.50, 0.80, 0.90, 0.95, 0.99]:
        print(f"    {q:.0%}: {np.quantile(scores, q):.3f}")

    return np.sort(scores)


def conformal_force_bound(
    result: dict,
    sorted_scores: np.ndarray,
    alpha: float = 0.10,
    aggregation: str = "max",
) -> float:
    """
    Return the conformal prediction bound on force MAE for a new structure.

    With probability ≥ 1 - alpha over exchangeable draws from the data
    distribution, the actual force MAE is ≤ the returned bound.

    alpha: miscoverage rate. alpha=0.05 → 95% coverage guarantee.
    Returns bound in eV/Å.

    Assumes exchangeability (structures are i.i.d. from the same distribution).
    This assumption breaks if the test structure is from a distribution shift
    (new temperature, composition, or phase). Check SOAP dissimilarity first.
    [EXPERT REVIEW NEEDED: exchangeability assumption is rarely exactly satisfied
     in active learning settings with distribution shift]
    """
    n = len(sorted_scores)
    # Conformal quantile: (n+1)(1-alpha)/n quantile of calibration scores
    q_idx = int(np.ceil((n + 1) * (1 - alpha))) - 1
    q_idx = min(q_idx, n - 1)
    quantile = sorted_scores[q_idx]

    sigma = result[f"sigma_F_{aggregation}_eV_A"]
    bound = quantile * sigma  # eV/Å
    return bound
```

### Workflow 6: Uncertainty monitoring in LAMMPS trajectories and stopping criteria

Post-processing a LAMMPS trajectory through the committee to identify extrapolation onset, without running the committee inline during MD.

```python
import numpy as np
from ase.io import read
from pathlib import Path


def evaluate_trajectory_uncertainty(
    trajectory_file: str,
    committee: "CommitteeCalculator",
    subsample_interval: int = 10,
    sigma_high: float = 0.50,
    sigma_low: float = 0.05,
    window_size: int = 50,
) -> dict:
    """
    Post-process a LAMMPS dump trajectory through the committee to identify
    when and where the MLP began extrapolating.

    subsample_interval: evaluate every N-th frame (reduces cost).
    sigma_high: extrapolation threshold (eV/Å).
    window_size: rolling window for computing running extrapolation fraction.

    Returns timestamps, σ_F time series, and the extrapolation onset step.
    """
    frames = read(trajectory_file, index=f"::{subsample_interval}")
    results = committee.batch_compute(frames)

    sigma_max_series = np.array([r["sigma_F_max_eV_A"] for r in results])
    sigma_mean_series = np.array([r["sigma_F_mean_eV_A"] for r in results])
    is_extrapolating = sigma_max_series >= sigma_high
    is_uncertain = (sigma_max_series >= sigma_low) & ~is_extrapolating

    # Rolling extrapolation fraction
    rolling_extrap_frac = np.convolve(
        is_extrapolating.astype(float),
        np.ones(window_size) / window_size,
        mode="valid",
    )

    # Find onset: first window where rolling fraction > 1%
    onset_window_idx = next(
        (i for i, f in enumerate(rolling_extrap_frac) if f > 0.01),
        None,
    )
    onset_frame = onset_window_idx * subsample_interval if onset_window_idx else None

    print(f"\nTrajectory uncertainty analysis:")
    print(f"  Frames evaluated: {len(frames)} (every {subsample_interval} steps)")
    print(f"  Fraction confident (σ < {sigma_low:.2f} eV/Å):   "
          f"{(~is_uncertain & ~is_extrapolating).mean():.1%}")
    print(f"  Fraction uncertain (query zone):  {is_uncertain.mean():.1%}")
    print(f"  Fraction extrapolating (σ ≥ {sigma_high:.2f}): {is_extrapolating.mean():.1%}")

    if onset_frame is not None:
        print(f"  Extrapolation onset (> 1% rolling): frame {onset_frame}")
        print(f"  → Trajectory should have been stopped or re-trained before this point")
    else:
        print(f"  No persistent extrapolation detected")

    return {
        "sigma_max_series": sigma_max_series,
        "sigma_mean_series": sigma_mean_series,
        "is_extrapolating": is_extrapolating,
        "extrapolation_onset_frame": onset_frame,
        "extrapolation_fraction": float(is_extrapolating.mean()),
        "n_frames": len(frames),
    }


def write_uncertainty_annotated_trajectory(
    trajectory_file: str,
    committee: "CommitteeCalculator",
    output_file: str,
    sigma_high: float = 0.50,
    subsample_interval: int = 1,
) -> None:
    """
    Write a trajectory in extxyz format with per-frame σ_F annotations.
    Per-atom uncertainty stored as an extxyz property for visualization
    in OVITO or ASE GUI: color atoms by σ_F to identify uncertain local
    environments.
    """
    from ase.io import write as ase_write

    frames = read(trajectory_file, index=f"::{subsample_interval}")
    annotated = []

    for atoms, result in zip(frames, committee.batch_compute(frames)):
        a = atoms.copy()
        a.info["sigma_F_max_eV_A"] = result["sigma_F_max_eV_A"]
        a.info["sigma_F_mean_eV_A"] = result["sigma_F_mean_eV_A"]
        a.info["sigma_E_per_atom_eV"] = result["sigma_E_per_atom_eV"]
        a.info["uncertainty_class"] = (
            "confident" if result["sigma_F_max_eV_A"] < 0.05
            else "extrapolation" if result["sigma_F_max_eV_A"] >= sigma_high
            else "query"
        )
        # Per-atom σ_F as an array property for visualization
        a.arrays["sigma_F_per_atom"] = result["sigma_F_per_atom_eV_A"]
        annotated.append(a)

    ase_write(output_file, annotated, format="extxyz")
    n_extrap = sum(1 for a in annotated
                   if a.info.get("uncertainty_class") == "extrapolation")
    print(f"Wrote {len(annotated)} annotated frames to {output_file}")
    print(f"  {n_extrap} frames classified as extrapolation "
          f"(σ_F_max ≥ {sigma_high} eV/Å)")
```

## Common Failure Modes

| Failure | Symptom | Likely Cause | Diagnostic | Fix |
|---------|---------|--------------|------------|-----|
| Overconfident extrapolation — ensemble collapse | σ_F is uniformly low (< σ_low) for all MD frames including frames where the MD trajectory later diverges; the model fails catastrophically without any warning from the committee | Committee members trained on the same homogeneous dataset with only seed diversity — all members learned the same function and agree on wrong predictions outside training | Run `verify_committee_diversity`; if mean pairwise force MAE between members is < 5 meV/Å, the committee has collapsed; also check SOAP dissimilarity for the failing frames | Increase training set diversity before training the committee; alternatively use partially non-overlapping training subsets for each member (jackknife or bootstrap); supplement with SOAP dissimilarity for OOD detection |
| σ_F dominated by trivially high-force atoms | Structure-level σ_F (max aggregation) is high but all high-σ atoms are the same ones that have large |F|; σ_F (mean aggregation) is low; the structure is physically reasonable | Per-atom σ_F scales with force magnitude for atoms near other atoms (short contact); the committee disagrees proportionally to the force magnitude, not because the environment is novel | Run `separate_high_force_atoms`; if n_trivial >> n_genuine, switch from max to mean or p90 aggregation for the uncertainty threshold | Use mean or 90th-percentile σ_F for active learning queries; use relative σ_F/|F| to flag atoms that are genuinely uncertain independent of force magnitude |
| Miscalibrated thresholds — too low σ_low | Active learning queries excessive DFT calculations per AL iteration; convergence never reached; DFT budget exhausted; majority of selected frames turn out to have actual force errors well below acceptable tolerance | σ_low set by intuition or copied from a different system without calibration against DFT errors on the target system | Run `calibrate_uncertainty_thresholds` on the test set; if the recommended σ_low is 3× higher than the current setting, thresholds are too conservative | Recalibrate using Workflow 3; accept a higher σ_low to reduce DFT spending; verify that at the new σ_low the fraction of false-negatives (missed errors) is acceptable |
| Miscalibrated thresholds — too high σ_high | Production MD runs without triggering any warnings; model silently extrapolates; trajectory diverges only after hundreds of ps | σ_high set to an unrealistically large value; or σ_high was calibrated on a near-equilibrium dataset that does not represent the MD temperature and pressure | Check: at the target MD temperature, what fraction of frames have σ_F_max > current σ_high? If < 0.01%, the threshold is too permissive; run short committee-evaluated MD and inspect frames near σ_high | Lower σ_high until the extrapolation fraction matches expectation for the system; validate by running active learning iterations at the target temperature |
| Missing species — committee agrees on garbage | A new element (e.g., a dopant, adsorbed molecule) appears in the production structure but was absent from training; committee produces large and wrong forces; but σ_F is not elevated because all members produce the same garbage | Element not in any committee member's training set; MLP's embedding for unseen elements is random but consistent across members initialized with the same architecture | Check: does `model.atomic_numbers` (MACE) contain all elements in the production structure? If not, the model will use a fallback embedding that is consistent across members | Always add at least a few DFT-labeled structures containing every element that will appear in production, even if that element is a trace dopant; verify species list in committee manifest before deployment |
| Inconsistent DFT labels across committee training | Committee members trained on data from different DFT settings (different ENCUT, POTCAR, or functional); pairwise committee disagreement is elevated but not because of genuine model uncertainty — it reflects inconsistent labels | DFT settings were changed between active learning iterations; new data added with a different settings hash; committee uses all data including the inconsistent labels | Run `DatasetManifest.report()` and check settings_hash consistency; compare POTCAR checksums across iterations; run `verify_dft_provenance` on all calc directories | Recompute inconsistent structures; enforce settings hash check in `collect_vasp_labels` and `DatasetManifest.add_iteration`; maintain a single canonical DFT settings file |
| Universal potential committee — falsely low uncertainty | A committee of fine-tuned versions of MACE-MP-0 (or another foundation model) shows low σ_F on structures that are genuinely extrapolation for the target DFT level | Committee members share the same pre-trained representation; fine-tuning with different seeds produces similar models in the fine-tuning data distribution; disagreement in the base pre-training distribution is not reflective of the target DFT level uncertainty | Compare SOAP dissimilarity of production frames against the fine-tuning training set (not the pre-training set); high dissimilarity despite low σ_F confirms false confidence | Do not rely on committee σ_F alone for fine-tuned universal potentials; always combine with SOAP dissimilarity; run a system-specific validation (mlp-validation) of the fine-tuned model before production |
| Confusing force standard deviation with force error | σ_F values are reported as if they were force errors; thresholds are set as if σ_F = actual force MAE; calibration factor α is assumed to be 1.0 | Conflation of model disagreement (σ_F) with accuracy (|F_MLP - F_DFT|); the two are correlated but σ_F ≠ error without calibration | Compute the calibration factor α from Workflow 3; if α >> 1, σ_F systematically underestimates the actual error | Always calibrate α; report σ_F and the calibrated bound α·σ_F separately; in publications, state whether reported uncertainty is raw σ_F or calibrated |

## Best Practices

- **Train at least 4 committee members, preferably 8.** With N=2 or N=3, the standard deviation has high statistical noise (std of 2 values is poorly estimated). N=8 provides a reliable estimate at 8× training cost. For large models where 8× is unaffordable, use N=4 and increase the seed spacing.
- **Verify committee diversity immediately after training** with `verify_committee_diversity`. A committee where members agree within 3 meV/Å on a diverse test set is collapsed and will not provide useful uncertainty estimates. Do not proceed to active learning with a collapsed committee.
- **Calibrate σ_low and σ_high on every new chemical system** using Workflow 3. Thresholds calibrated for a W-Ta alloy at 1000 K will not transfer to an oxide at 500 K or an organic molecule. The calibration takes < 1 hour on a pre-existing test set and prevents both excessive DFT spending and silent extrapolation.
- **Always supplement committee σ_F with SOAP dissimilarity for critical production runs.** The two signals are complementary: σ_F is sensitive to epistemic uncertainty within the training distribution; SOAP dissimilarity is sensitive to structural novelty regardless of committee agreement. Use both in parallel and flag structures where they disagree (Workflow 4).
- **Use per-atom σ_F for active learning queries, not structure-level max σ_F.** Selecting structures where a single atom drives up σ_F_max often targets high-force short-contact events that are edge cases, not the most informative structures. Filter by normalized (relative) σ_F or by mean σ_F before FPS diversity selection.
- **Record the committee manifest (model paths, SHA-256 hashes, seeds, and calibration parameters) in the project log.** A committee evaluated with a different model file than what was calibrated against will have different σ_F values and different threshold semantics. See `reproducibility-workflow` for how to track this.
- **Run the NVE energy conservation test from `mlp-validation` before each new AL iteration's model.** A model with poor NVE stability will produce diverging trajectories before σ_F rises above σ_high; NVE drift is an independent check that the model is physically stable in the well-sampled regime.
- **Do not use committee uncertainty as the sole stopping criterion for active learning.** The fraction of frames with σ_F > σ_low dropping below 1% is a necessary but not sufficient stopping criterion. Always run the full `mlp-validation` protocol on the final model before declaring convergence.

## Integration with Other Skills

- **[mace-training](../mace-training/SKILL.md):** Committee members are trained using the same hyperparameters as the single-model workflow described there. The `--seed` argument, `--ema`, `--scaling`, and `--default_dtype` flags all apply identically to committee training. The best-practice note there to record the model checksum extends to every committee member.
- **[active-learning-mlp](../active-learning-mlp/SKILL.md):** Committee σ_F is the primary uncertainty signal in that skill's Workflow 2 and Workflow 3. The σ_low, σ_high, and σ_query thresholds defined by calibration in Workflow 3 above map directly to the `SIGMA_LOW`, `SIGMA_HIGH`, and `SIGMA_QUERY` constants in that skill. The FPS diversity selection in that skill's Workflow 3 operates on the candidate pool filtered by this skill's uncertainty classification.
- **[mlp-validation](../mlp-validation/SKILL.md):** Static test-set error analysis (Workflow 1 of that skill) provides the ground truth against which committee σ_F is calibrated (Workflow 3 above). The held-out test set used for static validation must also be the calibration set for uncertainty thresholds. The `check_extrapolation_coverage` function in that skill uses committee σ_F with the thresholds produced here.
- **[mlp-to-lammps](../mlp-to-lammps/SKILL.md):** Post-processing LAMMPS trajectories through the Python committee (Workflow 6 above) uses structures extracted from LAMMPS dump files. The committee force evaluation is done in Python/ASE after the LAMMPS run; LAMMPS itself uses only a single model for production. The model SHA-256 from the deployment validation in that skill must match one of the committee member checksums.
- **[universal-potentials](../universal-potentials/SKILL.md):** The limitations of committee UQ for fine-tuned universal potentials are described in the Domain Context and in the failure mode table. That skill covers when universal potentials are appropriate without fine-tuning; the SOAP dissimilarity approach described here is the recommended OOD check for any universal potential deployment.
- **[fine-tuning-mlp](../fine-tuning-mlp/SKILL.md):** When fine-tuning a universal potential on system-specific DFT data, the committee for UQ should be built from fine-tuned members, not from the foundation model's checkpoints. The SOAP dissimilarity coverage radius must be fitted against the fine-tuning training set, not the pre-training set.
- **[dft-dataset-generation](../../electronic-structure/dft-dataset-generation/SKILL.md):** The structures flagged for DFT labeling by committee uncertainty are the inputs to that skill's VASP and QE labeling pipelines. The settings hash from the canonical INCAR must be consistent across all iterations; the `DatasetManifest` hash check prevents accidental label inconsistency between committee members trained on data from different iterations.
- **[dft-convergence](../../electronic-structure/dft-convergence/SKILL.md):** Inconsistent DFT settings between committee training iterations is a silent failure mode that elevates committee σ_F artifactually. The settings hash established by that skill's convergence protocol must be frozen before the first AL iteration and checked in every subsequent oracle call.
- **[lammps-workflow](../../atomistic-md/lammps-workflow/SKILL.md):** Exploration MD trajectories used for active learning are run via LAMMPS. The dump frequency and dump format settings from that skill determine how many frames are available for committee post-processing. A dump frequency that is too low (every 1000 steps) may miss short-lived high-uncertainty configurations.
- **[md-equilibration](../../atomistic-md/md-equilibration/SKILL.md):** Committee σ_F during the equilibration phase provides an early warning that the MLP is extrapolating before the full production run is committed. The NVE energy conservation test from that skill and committee σ_F are complementary: NVE tests stability within the sampled configurations; σ_F flags the onset of novel configurations.
- **[reproducibility-workflow](../../scientific-data/reproducibility-workflow/SKILL.md):** The committee manifest (model paths, SHA-256, seeds, calibration parameters σ_low/σ_high, calibration factor α, calibration set checksum) is the per-campaign provenance record for the UQ pipeline. It belongs alongside the dataset manifest and the MLflow run records described in that skill.
- **[materials-featurization](../../scientific-data/materials-featurization/SKILL.md):** The SOAP descriptor parameters used for structural dissimilarity in Workflow 4 must be consistent with those used for FPS diversity filtering in `active-learning-mlp` and `dft-dataset-generation`. That skill's `save_feature_metadata` pattern should be applied to the SOAP parameters used in `SOAPDissimilarityDetector` to ensure the same descriptor is used at every stage.

## Key References

1. Schran, C., et al. (2020). *Committee neural network potentials control generalization errors and enable active learning.* The Journal of Chemical Physics, 153(10), 104105. https://doi.org/10.1063/5.0016004
2. Imbalzano, G., et al. (2021). *Uncertainty estimation for molecular dynamics and sampling.* The Journal of Chemical Physics, 154(7), 074102. https://doi.org/10.1063/5.0036522
3. Vandermause, J., et al. (2020). *On-the-fly active learning of interpretable Bayesian force fields for atomistic rare events.* npj Computational Materials, 6, 20. https://doi.org/10.1038/s41524-020-0283-z
4. Batatia, I., et al. (2022). *MACE: Higher order equivariant message passing neural networks for fast and accurate force fields.* Advances in Neural Information Processing Systems, 35. https://arxiv.org/abs/2206.07697
5. Angelopoulos, A. N., & Bates, S. (2023). *Conformal prediction: A gentle introduction.* Foundations and Trends in Machine Learning, 16(4), 494–591. https://doi.org/10.1561/2200000101
