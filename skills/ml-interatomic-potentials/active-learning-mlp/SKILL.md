---
name: active-learning-mlp
domain: ml-interatomic-potentials
maturity: draft
tools:
  - MACE
  - LAMMPS
  - VASP
  - Quantum ESPRESSO
  - ASE
  - PyTorch
tested_versions:
  MACE: "0.3.6"
  LAMMPS: "2Aug2023"
  VASP: "6.3.2"
  ASE: "3.23.0"
requires_gpu: true
estimated_runtime: days-to-weeks
---

# Active Learning for ML Interatomic Potentials

## Description

This skill covers the systematic active learning workflow for iteratively building ML interatomic potential training datasets: initial dataset construction, uncertainty-guided exploration MD, DFT oracle labeling, query strategy, dataset versioning, retraining cadence, and stopping criteria. Active learning is the primary method for constructing compact, non-redundant MLP training datasets that generalize across the thermodynamic conditions and structural environments targeted for production MD. Invoke this skill when building a new MLP training dataset from scratch, when a trained MLP fails in MD outside its training distribution, or when designing a data generation campaign for a new chemical system.

## Domain Context

An MLP trained on a finite DFT dataset is accurate only within the distribution of configurations that dataset samples. The critical challenge is that the target distribution — all configurations visited during production MD — is unknown before the potential exists. Active learning resolves this circularity by iterating:

1. **Explore:** Run MD with the current provisional MLP to generate candidate structures.
2. **Query:** Select from the candidates those most likely to improve the model (high uncertainty, high diversity, or in undersampled regions).
3. **Label:** Compute DFT energies, forces, and stresses for the selected structures.
4. **Retrain:** Add the new DFT data to the training set and retrain or fine-tune the MLP.
5. **Evaluate:** Test whether the model has reached the target accuracy and stability.

The loop continues until a stopping criterion is met: typically when the fraction of exploration MD frames with high uncertainty falls below a threshold, indicating that the MLP has learned the accessible configuration space.

**Why not random sampling?** DFT calculations are expensive (minutes to hours each). Random structures are overwhelmingly drawn from high-energy, irrelevant regions of configuration space. Active learning focuses the DFT budget on structures that are both physically accessible (from MD) and maximally informative (high model uncertainty).

**The committee model approach** is the dominant practical uncertainty estimator for neural network potentials. Train N models (typically 4–8) with identical architecture but different random seeds or data subsets. Frames where the N models disagree strongly (high force standard deviation) are likely extrapolation candidates — structures where the model is uncertain because it has seen few similar configurations during training.

**A key risk:** An MLP that is overconfident — one whose uncertainty estimates are well-calibrated on the training distribution but systematically low for a new class of structures — will fail silently. The committee disagreement is only meaningful if the committee members have genuinely diverse representations; identical architectures trained on the same data with different seeds produce limited diversity. Uncertainty quantification does not prove the model is correct; it only identifies structures where the current models disagree.

## When to Use This Skill

- Building a system-specific MLP from scratch for a new composition, phase, or temperature range.
- Extending an existing MLP to new conditions (higher temperature, new phase, surface, defect, interface) where the model is known to be extrapolating.
- Generating training datasets for fine-tuning a universal potential (MACE-MP-0) on a specific system.
- Detecting and recovering from MLP failure during production MD (runaway trajectory diagnosis and repair).
- Designing a DFT data collection campaign with a fixed oracle budget (e.g., N DFT calculations per week on a shared HPC cluster).

## When Not to Use This Skill

- **When a validated universal potential already covers the system** — benchmark first; if accuracy is sufficient, active learning is unnecessary overhead.
- **When the target configuration space is fully enumerable** — for a simple bulk crystal with one equilibrium structure and small thermal displacements, a well-chosen set of rattled structures plus volumetrically strained configurations may be sufficient without an iterative loop.
- **When DFT oracle cost is negligible** — for systems where DFT calculations are extremely cheap (< 1 minute each), generating a large random dataset and training once is often more practical than a complex AL pipeline.
- **For systems requiring excited-state or non-adiabatic dynamics** — active learning on Born-Oppenheimer surfaces does not cover conical intersections or photochemical pathways.

## Core Concepts

- **Committee model:** N independently trained MLP instances (same architecture, different random seeds or different data subsets) used jointly for uncertainty estimation. The force disagreement across committee members at a given structure is used as the uncertainty proxy.
- **Model deviation (σ_F):** The standard deviation of per-atom force predictions across committee members, typically taken as the per-atom maximum or mean. Threshold-based selection: frames with σ_F > σ_high are labeled "extrapolation" (do not use for MD, always send to DFT); frames with σ_low < σ_F < σ_high are "interpolation with uncertainty" (candidates for DFT labeling); frames with σ_F < σ_low are "well-converged" (keep in MD, do not relabel).
- **Extrapolation grade (γ̃):** In Gaussian process and FLARE-based approaches, a Bayesian measure of how far a new structure departs from the training set in the kernel feature space. Equivalent in spirit to committee model deviation but with principled calibration.
- **Query budget:** The number of DFT oracle calls per active learning iteration. Constrained by HPC allocation and wall-time. Typical: 50–500 new DFT structures per iteration. [EXPERT REVIEW NEEDED]
- **Redundancy filtering:** Even among high-uncertainty frames, many may be structurally similar. Selecting all of them wastes the DFT budget. Diversity-based filtering (farthest-point sampling in descriptor space, clustering) selects a non-redundant subset.
- **Descriptor for diversity:** A fast-to-compute structural fingerprint used to measure pairwise structure similarity. Common choices: SOAP (Smooth Overlap of Atomic Positions) descriptor via DScribe, per-atom ACE descriptors, or MACE latent features. The descriptor space must be computed consistently across all candidate frames.
- **Dataset versioning:** Each iteration of the active learning loop produces a new dataset version. Versions must be tracked with checksums, metadata (DFT settings, iteration number, added structure count), and provenance (which MD run generated each structure). DVC (Data Version Control) or a custom JSON manifest are common solutions.
- **Retraining from scratch vs. fine-tuning:** Retraining from scratch on the full updated dataset is the safest option (no artifacts from previous iterations). Fine-tuning on only the new data is faster but risks catastrophic forgetting. For MACE, full retraining with consistent hyperparameters and a fixed random seed is recommended for reproducibility. [EXPERT REVIEW NEEDED]
- **Stopping criterion:** Common choices: (1) fraction of exploration frames with σ_F > σ_high falls below 1%; (2) MLP validation error on a held-out test set stops improving between iterations; (3) DFT budget is exhausted. Criterion (2) is the most physically meaningful but requires maintaining an independent test set that was not influenced by the active learning queries.
- **Oracle consistency:** All DFT labels across all iterations must be computed at identical settings: same code, same functional, same ENCUT (VASP) / ecutwfc (QE), same POTCAR version, same smearing. Mixing settings introduces systematic energy offsets that corrupt the training signal.

## Key Workflows

### Workflow 1: Initial dataset construction

The first training set must be physically reasonable and cover the target composition and a wide range of energetically accessible structures. It does not need to be large — 200–1000 structures is typical for a starting point.

```python
from ase.build import bulk, surface
from ase.io import write
from ase import Atoms
import numpy as np
from pathlib import Path

RNG = np.random.default_rng(42)


def rattle_structure(atoms: Atoms, sigma: float, n_copies: int) -> list[Atoms]:
    """
    Create thermally distorted copies by displacing atoms from equilibrium.
    sigma: displacement magnitude in Å. Typical: 0.05–0.3 Å.
    Larger sigma = higher effective temperature; always validate forces
    before including high-sigma structures in training.
    """
    copies = []
    for _ in range(n_copies):
        rattled = atoms.copy()
        rattled.rattle(stdev=sigma)
        copies.append(rattled)
    return copies


def volumetric_strain_set(atoms: Atoms,
                           strains: list[float]) -> list[Atoms]:
    """
    Isotropically scale the cell to probe the EOS and pressure range.
    strains: list of volumetric strain factors, e.g. [0.90, 0.95, 1.00, 1.05, 1.10].
    """
    return [
        Atoms(symbols=atoms.symbols,
              positions=atoms.positions * s**(1/3),
              cell=atoms.cell * s**(1/3),
              pbc=atoms.pbc)
        for s in strains
    ]


def build_initial_dataset(
    equilibrium_structures: list[Atoms],
    output_file: str,
    rattle_sigmas: list[float] = (0.05, 0.10, 0.20),
    n_rattled_per_sigma: int = 20,
    volume_strains: list[float] = (0.90, 0.94, 0.97, 1.00, 1.03, 1.06, 1.10),
) -> int:
    """
    Build a minimal initial dataset from equilibrium structures.
    Includes: rattled copies at multiple sigma, volumetrically strained cells.
    All structures are tagged with config_type for stratified splitting.
    """
    all_frames: list[Atoms] = []
    for base in equilibrium_structures:
        ctype = base.info.get("config_type", "bulk")
        # Rattled structures at escalating distortions
        for sigma in rattle_sigmas:
            for frame in rattle_structure(base, sigma, n_rattled_per_sigma):
                frame.info["config_type"] = f"{ctype}_rattle_{sigma:.2f}"
                frame.info["al_iteration"] = 0
                all_frames.append(frame)
        # Volumetrically strained for EOS coverage
        for frame in volumetric_strain_set(base, volume_strains):
            frame.info["config_type"] = f"{ctype}_strain"
            frame.info["al_iteration"] = 0
            all_frames.append(frame)

    write(output_file, all_frames, format="extxyz")
    print(f"Initial dataset: {len(all_frames)} structures → {output_file}")
    return len(all_frames)


# Example: W-Ta binary alloy
W_bcc = bulk("W", "bcc", a=3.165)
W_bcc.info["config_type"] = "W_bulk"
Ta_bcc = bulk("Ta", "bcc", a=3.303)
Ta_bcc.info["config_type"] = "Ta_bulk"

# Also include representative alloy compositions
# (SQS or random substitutions; compute DFT on these too)
build_initial_dataset([W_bcc, Ta_bcc], "al_iter0_candidates.xyz")
# Send al_iter0_candidates.xyz for DFT labeling (see Workflow 4),
# then train the first provisional committee.
```

### Workflow 2: Committee model training and force uncertainty estimation

```python
import subprocess
from pathlib import Path


def train_committee(
    train_file: str,
    val_file: str,
    n_members: int = 4,
    base_seed: int = 100,
    model_prefix: str = "committee",
    r_max: float = 5.5,
    hidden_irreps: str = "128x0e + 128x1o + 128x2e",
    max_num_epochs: int = 1000,
    device: str = "cuda",
) -> list[str]:
    """
    Train N committee members with identical hyperparameters but different random seeds.
    Returns list of trained model file paths.
    """
    model_paths = []
    for i in range(n_members):
        seed = base_seed + i * 17   # Separated seeds for genuine independence
        name = f"{model_prefix}_seed{seed}"
        cmd = [
            "mace_run_train",
            f"--name={name}",
            f"--train_file={train_file}",
            f"--valid_file={val_file}",
            f"--E0s=average",
            f"--model=MACE",
            f"--r_max={r_max}",
            f"--correlation=3",
            f"--num_interactions=2",
            f"--hidden_irreps={hidden_irreps}",
            f"--max_ell=3",
            f"--energy_weight=1.0",
            f"--forces_weight=10.0",
            f"--stress_weight=1.0",
            f"--batch_size=8",
            f"--lr=0.01",
            f"--max_num_epochs={max_num_epochs}",
            f"--patience=200",
            f"--ema",
            f"--ema_decay=0.99",
            f"--scaling=rms_forces_scaling",
            f"--default_dtype=float32",
            f"--device={device}",
            f"--seed={seed}",
        ]
        print(f"Training committee member {i+1}/{n_members} (seed={seed})")
        subprocess.run(cmd, check=True)
        # MACE saves to {name}_run-{seed}_stagetwo.model
        model_file = f"{name}_run-{seed}_stagetwo.model"
        model_paths.append(model_file)
    return model_paths


def compute_force_uncertainty(
    frames: list,
    model_paths: list[str],
    device: str = "cpu",
) -> list[dict]:
    """
    Compute per-frame force standard deviation across committee members.
    Returns a list of dicts with index, max_sigma_F, mean_sigma_F, and config_type.
    """
    import numpy as np
    from mace.calculators import MACECalculator

    calcs = [
        MACECalculator(model_paths=mp, device=device, default_dtype="float32")
        for mp in model_paths
    ]
    results = []
    for i, atoms in enumerate(frames):
        forces_per_model = []
        for calc in calcs:
            a = atoms.copy()
            a.calc = calc
            forces_per_model.append(a.get_forces())  # (N_atoms, 3)
        # Stack: shape (n_committee, N_atoms, 3)
        F = np.stack(forces_per_model, axis=0)
        sigma_F = F.std(axis=0)               # (N_atoms, 3)
        max_sigma = sigma_F.max()             # Worst-case per-component disagreement
        mean_sigma = sigma_F.mean()
        results.append({
            "index": i,
            "max_sigma_F_eV_A": float(max_sigma),
            "mean_sigma_F_eV_A": float(mean_sigma),
            "config_type": atoms.info.get("config_type", "unknown"),
            "natoms": len(atoms),
        })
    return results
```

### Workflow 3: Query strategy — uncertainty filtering and diversity selection

Selecting which uncertain frames to send to DFT. High uncertainty alone is insufficient — many high-uncertainty frames may be redundant copies of the same environment.

```python
import numpy as np
from ase.io import read, write
from dscribe.descriptors import SOAP
from sklearn.preprocessing import normalize


# --- Uncertainty thresholds (system-dependent; adjust empirically) ---
SIGMA_LOW = 0.05    # eV/Å: below this → model confident, skip DFT
SIGMA_HIGH = 0.50   # eV/Å: above this → likely unphysical extrapolation,
                    # do NOT include in training; investigate structure
SIGMA_QUERY = 0.15  # eV/Å: above this (and below SIGMA_HIGH) → candidate for DFT
# [EXPERT REVIEW NEEDED: thresholds are highly system-dependent;
#  calibrate on a held-out validation set]


def classify_frames(
    uncertainty_results: list[dict],
    sigma_low: float = SIGMA_LOW,
    sigma_high: float = SIGMA_HIGH,
    sigma_query: float = SIGMA_QUERY,
) -> dict[str, list[int]]:
    """Classify frames by uncertainty tier."""
    confident, candidates, extrapolation = [], [], []
    for r in uncertainty_results:
        s = r["max_sigma_F_eV_A"]
        if s < sigma_low:
            confident.append(r["index"])
        elif s < sigma_query:
            pass   # Uncertain but below query threshold — continue MD
        elif s < sigma_high:
            candidates.append(r["index"])
        else:
            extrapolation.append(r["index"])
    return {"confident": confident,
            "candidates": candidates,
            "extrapolation": extrapolation}


def farthest_point_sampling(
    frames: list,
    indices: list[int],
    n_select: int,
    species: list[str],
) -> list[int]:
    """
    Select n_select structures from candidates using farthest-point sampling
    in SOAP descriptor space. Ensures diversity: selected structures are
    maximally spread across the accessible configuration space.
    """
    if len(indices) <= n_select:
        return indices

    # Compute SOAP descriptors
    soap = SOAP(
        species=species,
        r_cut=5.0,
        n_max=8,
        l_max=6,
        average="inner",  # Global descriptor per structure
        sparse=False,
    )
    candidate_frames = [frames[i] for i in indices]
    descs = soap.create(candidate_frames)              # (N_candidates, D)
    descs = normalize(descs, norm="l2")

    # Greedy farthest-point sampling
    selected = [0]  # Start from first candidate (arbitrary)
    min_dists = np.full(len(indices), np.inf)

    for _ in range(n_select - 1):
        last = descs[selected[-1]]
        dists_to_last = np.linalg.norm(descs - last, axis=1)
        min_dists = np.minimum(min_dists, dists_to_last)
        min_dists[selected] = -1   # Already selected
        next_idx = int(np.argmax(min_dists))
        selected.append(next_idx)

    return [indices[i] for i in selected]


def select_dft_candidates(
    frames: list,
    uncertainty_results: list[dict],
    n_dft_budget: int,
    species: list[str],
) -> tuple[list, list[int]]:
    """
    Full query pipeline: classify → filter → diversity select → return candidates.
    Returns (selected_frames, selected_indices_in_original_list).
    """
    classes = classify_frames(uncertainty_results)
    candidates = classes["candidates"]
    extrapolation = classes["extrapolation"]

    print(f"  Confident:      {len(classes['confident'])} frames (skip DFT)")
    print(f"  Query zone:     {len(candidates)} frames (candidates for DFT)")
    print(f"  Extrapolation:  {len(extrapolation)} frames (flag for investigation)")

    if extrapolation:
        print(f"  WARNING: {len(extrapolation)} frames are likely unphysical "
              f"— do not include in training without manual inspection")

    if not candidates:
        print("  No candidates in query zone — model may have converged "
              "or exploration is stuck")
        return [], []

    # Farthest-point sampling to select diverse subset within budget
    selected_indices = farthest_point_sampling(
        frames, candidates, n_dft_budget, species
    )
    selected_frames = [frames[i] for i in selected_indices]
    print(f"  Selected {len(selected_frames)} structures for DFT labeling "
          f"(from {len(candidates)} candidates)")
    return selected_frames, selected_indices
```

### Workflow 4: DFT oracle — labeling pipeline and consistency checks

```python
from pathlib import Path
from pymatgen.io.vasp.sets import MPStaticSet
from pymatgen.core import Structure
import json

# DFT SETTINGS MUST BE FROZEN FOR THE ENTIRE ACTIVE LEARNING CAMPAIGN.
# Any change to ENCUT, POTCAR, ISMEAR, SIGMA, or functional after Iteration 0
# invalidates the reference energy surface and corrupts the training set.

DFT_SETTINGS = {
    "ENCUT": 520,
    "EDIFF": 1e-6,
    "NSW": 0,
    "IBRION": -1,
    "ISMEAR": 0,
    "SIGMA": 0.10,
    "PREC": "Accurate",
    "ALGO": "Normal",
    "LWAVE": False,
    "LCHARG": False,
    "ISIF": 2,     # Compute stress tensor
    "LREAL": "Auto",
    "NELM": 200,
}


def prepare_oracle_batch(
    candidate_frames: list,
    output_base: Path,
    al_iteration: int,
    dft_settings: dict = DFT_SETTINGS,
) -> Path:
    """
    Write VASP input files for each candidate structure.
    Returns path to a manifest JSON documenting the batch.
    """
    batch_dir = output_base / f"al_iter{al_iteration:03d}_oracle"
    batch_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"iteration": al_iteration, "structures": []}

    for j, atoms in enumerate(candidate_frames):
        from ase.io import write as ase_write
        calc_dir = batch_dir / f"struct_{j:04d}"
        calc_dir.mkdir(exist_ok=True)

        # Convert ASE Atoms → pymatgen Structure
        from pymatgen.io.ase import AseAtomsAdaptor
        structure = AseAtomsAdaptor.get_structure(atoms)
        vis = MPStaticSet(structure, user_incar_settings=dft_settings)
        vis.write_input(str(calc_dir))

        manifest["structures"].append({
            "index": j,
            "config_type": atoms.info.get("config_type", "unknown"),
            "calc_dir": str(calc_dir.relative_to(output_base)),
            "natoms": len(atoms),
        })

    manifest_file = batch_dir / "manifest.json"
    manifest_file.write_text(json.dumps(manifest, indent=2))
    print(f"Prepared {len(candidate_frames)} DFT calculations in {batch_dir}")
    return manifest_file


def collect_oracle_results(
    batch_dir: Path,
    al_iteration: int,
    check_convergence: bool = True,
) -> list:
    """
    Parse completed DFT calculations and return labeled ASE Atoms.
    Filters out unconverged runs.
    """
    from pymatgen.io.vasp.outputs import Vasprun
    from ase import Atoms as AseAtoms
    import numpy as np

    KBAR_TO_EV_A3 = 1.0 / 160.21766208
    labeled_frames = []
    n_failed = 0

    for calc_dir in sorted(batch_dir.glob("struct_*")):
        vr_path = calc_dir / "vasprun.xml"
        if not vr_path.exists():
            n_failed += 1
            continue
        try:
            v = Vasprun(str(vr_path), parse_potcar_file=False)
        except Exception:
            n_failed += 1
            continue
        if check_convergence and not v.converged_electronic:
            n_failed += 1
            print(f"  Unconverged: {calc_dir.name} — excluded")
            continue

        structure = v.final_structure
        last_step = v.ionic_steps[-1]
        atoms = AseAtoms(
            symbols=[str(s) for s in structure.species],
            positions=structure.cart_coords,
            cell=structure.lattice.matrix,
            pbc=True,
        )
        atoms.info["energy"] = v.final_energy
        atoms.arrays["forces"] = np.array(last_step["forces"])
        stress_kbar = np.array(last_step["stress"])
        stress_voigt = np.array([
            stress_kbar[0,0], stress_kbar[1,1], stress_kbar[2,2],
            stress_kbar[1,2], stress_kbar[0,2], stress_kbar[0,1],
        ])
        atoms.info["stress"] = -stress_voigt * KBAR_TO_EV_A3
        atoms.info["al_iteration"] = al_iteration
        labeled_frames.append(atoms)

    print(f"Collected {len(labeled_frames)} labeled structures "
          f"({n_failed} failed/unconverged)")
    return labeled_frames
```

### Workflow 5: Dataset versioning, retraining, and the full iteration loop

```python
import hashlib
import json
from pathlib import Path
from ase.io import read, write
from datetime import datetime


class ActiveLearningDataset:
    """
    Versioned dataset manager for active learning iterations.
    Maintains a manifest with provenance of each structure (iteration, source).
    """

    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        self.manifest_file = self.base_dir / "dataset_manifest.json"
        if self.manifest_file.exists():
            self.manifest = json.loads(self.manifest_file.read_text())
        else:
            self.manifest = {"iterations": [], "total_structures": 0,
                             "dft_settings_hash": None}

    def _hash_settings(self, settings: dict) -> str:
        return hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:16]

    def add_iteration(
        self, new_frames: list, al_iteration: int,
        dft_settings: dict, train_file: str, val_file: str,
    ) -> None:
        """
        Add new labeled structures to the dataset.
        Raises ValueError if DFT settings have changed since the first iteration.
        """
        settings_hash = self._hash_settings(dft_settings)
        if self.manifest["dft_settings_hash"] is None:
            self.manifest["dft_settings_hash"] = settings_hash
        elif self.manifest["dft_settings_hash"] != settings_hash:
            raise ValueError(
                "DFT settings have changed since iteration 0! "
                "This invalidates the reference energy surface. "
                "Do not mix datasets with different settings."
            )

        # Append to master dataset
        master_file = self.base_dir / "all_labeled.xyz"
        existing = read(str(master_file), index=":", format="extxyz") \
                   if master_file.exists() else []
        all_frames = existing + new_frames
        write(str(master_file), all_frames, format="extxyz")

        # Update manifest
        self.manifest["iterations"].append({
            "iteration": al_iteration,
            "n_new_structures": len(new_frames),
            "timestamp": datetime.utcnow().isoformat(),
            "settings_hash": settings_hash,
        })
        self.manifest["total_structures"] = len(all_frames)
        self.manifest_file.write_text(json.dumps(self.manifest, indent=2))

        # Write versioned train/val split for this iteration
        from sklearn.model_selection import train_test_split  # or use custom stratified split
        train, val = train_test_split(all_frames, test_size=0.10, random_state=al_iteration)
        write(train_file, train, format="extxyz")
        write(val_file, val, format="extxyz")
        print(f"Iteration {al_iteration}: added {len(new_frames)} structures. "
              f"Total: {len(all_frames)} (train={len(train)}, val={len(val)})")


def check_stopping_criteria(
    uncertainty_results: list[dict],
    val_force_mae_history: list[float],
    sigma_high: float = 0.50,
    convergence_fraction: float = 0.01,
    mae_patience: int = 3,
) -> dict:
    """
    Evaluate whether the active learning loop should terminate.
    Returns a dict with 'converged' flag and reasons.
    """
    # Criterion 1: Fraction of exploration frames above extrapolation threshold
    n_extrap = sum(1 for r in uncertainty_results
                   if r["max_sigma_F_eV_A"] > sigma_high)
    extrap_fraction = n_extrap / max(len(uncertainty_results), 1)

    # Criterion 2: Validation force MAE has plateaued
    mae_plateau = False
    if len(val_force_mae_history) >= mae_patience + 1:
        recent = val_force_mae_history[-mae_patience:]
        prev = val_force_mae_history[-(mae_patience + 1):-1]
        relative_improvement = (np.mean(prev) - np.mean(recent)) / np.mean(prev)
        mae_plateau = relative_improvement < 0.005  # < 0.5% improvement

    converged = extrap_fraction < convergence_fraction or mae_plateau
    return {
        "converged": converged,
        "extrapolation_fraction": extrap_fraction,
        "mae_plateau": mae_plateau,
        "reason": (
            f"extrap_fraction={extrap_fraction:.3f} "
            f"(threshold={convergence_fraction}); "
            f"mae_plateau={mae_plateau}"
        ),
    }


# --- Main active learning loop (orchestration sketch) ---
def run_active_learning_loop(
    system_name: str,
    initial_dataset: str,
    species: list[str],
    n_iterations: int = 10,
    n_dft_per_iter: int = 100,
    exploration_steps: int = 200000,
    target_temperature_K: float = 1000.0,
) -> None:
    """
    Orchestrates the full active learning loop.
    Each iteration: explore → query → label → retrain → evaluate.
    """
    import numpy as np
    ds = ActiveLearningDataset(f"{system_name}_al_dataset")
    val_mae_history = []

    for iteration in range(1, n_iterations + 1):
        print(f"\n{'='*60}")
        print(f"Active Learning Iteration {iteration}/{n_iterations}")
        print(f"{'='*60}")

        # 1. Retrain committee on current dataset
        model_paths = train_committee(
            train_file=f"{system_name}_train.xyz",
            val_file=f"{system_name}_val.xyz",
            n_members=4,
            base_seed=iteration * 100,
            model_prefix=f"{system_name}_iter{iteration:03d}",
        )

        # 2. Run exploration MD with uncertainty monitoring
        # (Use LAMMPS with committee average for dynamics;
        #  evaluate committee uncertainty every N steps)
        exploration_frames = run_exploration_md(  # defined separately
            model_paths=model_paths,
            initial_structure=f"{system_name}_equil.lammps",
            temperature_K=target_temperature_K,
            n_steps=exploration_steps,
            sample_every=100,
        )

        # 3. Compute uncertainty
        uncertainty_results = compute_force_uncertainty(
            exploration_frames, model_paths
        )

        # 4. Check stopping
        status = check_stopping_criteria(uncertainty_results, val_mae_history)
        print(f"Stopping check: {status['reason']}")
        if status["converged"]:
            print("Active learning converged. Proceeding to final validation.")
            break

        # 5. Select candidates for DFT
        candidates, _ = select_dft_candidates(
            exploration_frames, uncertainty_results, n_dft_per_iter, species
        )

        # 6. Label with DFT (submit to HPC, wait for completion)
        batch_dir = Path(f"{system_name}_oracle/iter{iteration:03d}")
        prepare_oracle_batch(candidates, batch_dir.parent, iteration)
        # ... HPC submission and wait ...
        labeled = collect_oracle_results(batch_dir, iteration)

        # 7. Update versioned dataset and prepare next train/val
        ds.add_iteration(labeled, iteration, DFT_SETTINGS,
                         f"{system_name}_train.xyz",
                         f"{system_name}_val.xyz")
```

### Workflow 6: Validation before production MD deployment

```python
def pre_deployment_validation(
    model_path: str,
    test_file: str,
    equilibrium_structure: str,
    target_temperature_K: float,
    device: str = "cpu",
) -> bool:
    """
    Systematic validation checklist before committing to production MD.
    Returns True if all checks pass.
    """
    import numpy as np
    from ase.io import read
    from mace.calculators import MACECalculator
    from ase.md.verlet import VelocityVerlet
    from ase import units

    calc = MACECalculator(model_paths=model_path, device=device,
                          default_dtype="float64")
    passed = []

    # Check 1: Force MAE on held-out test set
    frames = read(test_file, index=":", format="extxyz")
    force_errors = []
    for atoms in frames:
        ref_f = atoms.arrays["forces"]
        a = atoms.copy(); a.calc = calc
        pred_f = a.get_forces()
        force_errors.append(np.abs(ref_f - pred_f).mean() * 1000)
    fmae = np.mean(force_errors)
    check1 = fmae < 100.0  # meV/Å; threshold is system-dependent [EXPERT REVIEW NEEDED]
    print(f"[{'PASS' if check1 else 'FAIL'}] Test force MAE: {fmae:.1f} meV/Å")
    passed.append(check1)

    # Check 2: NVE energy drift from equilibrated structure
    eq = read(equilibrium_structure); eq.calc = calc
    from ase.md.verlet import VelocityVerlet
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
    MaxwellBoltzmannDistribution(eq, temperature_K=target_temperature_K)
    dyn = VelocityVerlet(eq, timestep=2.0 * units.fs)
    e0 = eq.get_total_energy()
    energies = [e0]
    for _ in range(5000):
        dyn.run(1)
        energies.append(eq.get_total_energy())
    slope = np.polyfit(range(len(energies)), energies, 1)[0]
    drift_meV_per_step = abs(slope) * 1000
    check2 = drift_meV_per_step < 0.01  # meV/step [EXPERT REVIEW NEEDED]
    print(f"[{'PASS' if check2 else 'FAIL'}] NVE drift: {drift_meV_per_step:.4f} meV/step")
    passed.append(check2)

    # Check 3: Committee uncertainty on exploration trajectory
    # (should be < sigma_low for > 99% of production-condition frames)

    all_passed = all(passed)
    print(f"\nPre-deployment: {'ALL CHECKS PASSED' if all_passed else 'SOME CHECKS FAILED'}")
    return all_passed
```

## Common Failure Modes

| Failure | Symptom | Likely Cause | Diagnostic | Fix |
|---------|---------|--------------|------------|-----|
| Runaway MD during exploration | Energy explosion within first ps; max force > 100 eV/Å; atoms escape simulation box | Provisional MLP explores unphysical configuration; model extrapolates far outside training domain | Monitor max force every 10 steps during exploration; halt trajectory if max force exceeds threshold (e.g., 50 eV/Å) | Set force cap in exploration MD; send the last valid frame before divergence to DFT rather than discarding; add minimum-image distance check to detect unphysical atom separations |
| Uncertainty miscalibration | Committee deviation is uniformly low; exploration MD appears confident but model fails in validation | Committee members are too similar (same architecture + same data + nearly same seed); overfit to training distribution | Compare committee member test MAEs independently; test on deliberately out-of-distribution structures | Increase committee diversity: use different random seeds with larger separation, different data subsets (jackknife), or different architectures; calibrate thresholds on a held-out validation set not used in training |
| Biased exploration — stuck in one phase | Dataset grows but model fails for temperatures or phases not yet explored; exploration only samples near-equilibrium | Exploration MD initialized from a single starting configuration or run at a single temperature | Visualize sampled configuration space with SOAP-PCA or UMAP; check if all config_types are represented in new acquisitions | Use multiple exploration trajectories from diverse starting points: different compositions, temperatures, defect types, surfaces; include AIMD snapshots at high temperature |
| Duplicate configurations inflate dataset | Dataset grows but model accuracy does not improve; training is slow and expensive for diminishing returns | Farthest-point sampling not applied; or applied with too coarse a descriptor; trajectory sampled too frequently | Compute pairwise SOAP distances for new acquisitions; if > 50% of pairs have distance < 0.01, dataset is redundant | Apply farthest-point sampling or clustering before DFT labeling; increase sampling interval in exploration MD; remove near-duplicates with distance threshold |
| Inconsistent DFT settings across iterations | Systematic energy discontinuity between early and late iterations; training loss increases after adding new data | ENCUT, POTCAR version, ISMEAR, or functional changed mid-campaign | Hash INCAR + POTCAR settings for every DFT calculation; plot E/atom distribution by iteration | Recompute affected structures at the canonical settings; reject calculations with non-matching settings hash; use dataset manager with settings validation (Workflow 5) |
| Overfitting to queried configurations | Model achieves low loss on training set and actively acquired structures but fails on unseen configurations | Dataset dominated by structures from one AL iteration; query strategy too narrowly focused | Compare error on a held-out test set never seen by the AL loop vs. error on the training set | Maintain a clean held-out test set from Iteration 0 (never include in AL queries); ensure all config_types are represented in each iteration's acquisitions |
| Missing rare events in training domain | MLP stable for bulk dynamics but fails for nucleation, phase transition, or rare defect creation | Standard temperature-driven exploration does not access rare events on MD timescales | Check validation error on structures from specialized calculations (TS searches, NEB, defect configurations) | Supplement exploration with targeted structure generation: rattled transition-state geometries, NEB images, short-range compressed configurations for fracture |
| False confidence — model stops querying too early | Stopping criterion triggers; subsequent production MD fails; model was overconfident | Committee uncertainty drops below threshold before true convergence; model memorized training distribution | Continue monitoring uncertainty on held-out structures after stopping; check production MD stability for 1+ ns before declaring success | Tighten stopping criterion; require both uncertainty convergence AND validation MAE plateau for ≥ 3 iterations; run production MD on a modest system first before committing to large-scale deployment |

## Best Practices

- **Freeze DFT settings at Iteration 0 and never change them.** Create a canonical `dft_settings.json` file at the start of the campaign and validate every new DFT batch against its hash before adding it to the training set. A single ENCUT change or POTCAR version change mid-campaign can corrupt the entire dataset.
- **Maintain a clean held-out test set from Iteration 0.** These structures are labeled with DFT but never shown to the model and never targeted by the query strategy. They provide an unbiased accuracy estimate throughout the campaign. Never add held-out structures to the training set.
- **Monitor max force during exploration MD.** Set a hard force cap (e.g., 30–50 eV/Å) that halts the trajectory and logs the divergence point. The last frame before divergence contains information about the failure mode — send it for DFT labeling rather than discarding.
- **Use at least 4 committee members for meaningful uncertainty estimates.** Two members provide a binary agreement/disagreement signal, not a calibrated uncertainty. Four to eight gives a more reliable estimate of the force standard deviation.
- **Calibrate uncertainty thresholds on a validation set, not by intuition.** The optimal σ_low and σ_query values depend on the system, architecture, and training set size. Compute the fraction of training-distribution structures with σ_F > σ_query; if it is > 20%, the threshold is too low. [EXPERT REVIEW NEEDED]
- **Diversify exploration temperatures and conditions.** Run exploration MD at multiple temperatures spanning the target application range. For solids, also include slightly above-melt structures to probe the disordered limit. For surfaces, include adsorbed and clean configurations separately.
- **Version every dataset update with full provenance.** Record: iteration number, DFT settings hash, list of new structure file names, committee model checksum used for the query, and timestamp. Without this, tracking down dataset bugs becomes intractable.
- **Document and report the active learning protocol in publications.** Specify the committee size, uncertainty thresholds, query strategy, total DFT count per iteration, number of iterations, and convergence criterion. This is the minimum needed for reproducibility of MLP papers.

## Integration with Other Skills

- **[mace-training](../mace-training/SKILL.md):** The committee training and dataset preparation workflows here extend that skill's single-model training. The extxyz format, DFT conversion routines, and hyperparameter choices transfer directly; this skill adds the iteration and uncertainty layer.
- **[mlp-validation](../mlp-validation/SKILL.md):** Workflow 6 (pre-deployment validation) is a prerequisite to that skill's full validation suite. After the AL loop converges, formal validation (RDF, phonons, elastic constants, long MD stability) should be performed before production deployment.
- **[mlp-to-lammps](../mlp-to-lammps/SKILL.md):** Exploration MD in the AL loop runs the provisional MACE models in LAMMPS or ASE. The LAMMPS compilation requirements and `no_domain_decomposition` constraints described there apply to every exploration run.
- **[uncertainty-mlp](../uncertainty-mlp/SKILL.md):** Committee deviation is the simplest uncertainty estimator. That skill covers more principled alternatives (deep ensembles, conformal prediction, last-layer Bayesian approaches) that may be warranted when committee diversity is insufficient.
- **[dft-dataset-generation](../../electronic-structure/dft-dataset-generation/SKILL.md):** Covers the broader dataset design problem: which structures to include beyond what AL generates (defects, surfaces, NEB images, AIRSS structures). AL is one strategy within that skill's framework.
- **[dft-convergence](../../electronic-structure/dft-convergence/SKILL.md):** The DFT oracle settings frozen in Workflow 4 must be established using that skill's convergence protocol before the AL campaign begins. Any change discovered during the campaign (e.g., ENCUT too low for a new species) requires recomputing affected structures.
- **[vasp-workflow](../../electronic-structure/vasp-workflow/SKILL.md):** Workflow 4 uses VASP as the DFT oracle. ISIF=2, PREC=Accurate, and tight EDIFF settings from that skill are required for accurate force and stress labels.
- **[quantum-espresso](../../electronic-structure/quantum-espresso/SKILL.md):** When QE is the oracle, the same consistency requirements apply. Forces in QE output are in Ry/bohr and must be converted via ASE before adding to the extxyz dataset.
- **[lammps-workflow](../../atomistic-md/lammps-workflow/SKILL.md):** Exploration MD uses LAMMPS with MACE pair style. The input script mechanics and timestep selection from that skill apply to every exploration trajectory.
- **[md-equilibration](../../atomistic-md/md-equilibration/SKILL.md):** Exploration trajectories should start from properly equilibrated structures to avoid sampling the transient equilibration phase. The protocol and diagnostics from that skill apply to each new starting configuration.
- **[reproducibility-workflow](../../scientific-data/reproducibility-workflow/SKILL.md):** The dataset versioning, DFT settings hashing, and provenance tracking described in Workflow 5 are a specialization of the general reproducibility framework covered there.

## Key References

1. Zhang, L., et al. (2019). *Active learning of uniformly accurate interatomic potentials for materials simulation.* Physical Review Materials, 3(2), 023804. https://doi.org/10.1103/PhysRevMaterials.3.023804
2. Vandermause, J., et al. (2020). *On-the-fly active learning of interpretable Bayesian force fields for atomistic simulations.* npj Computational Materials, 6(1), 20. https://doi.org/10.1038/s41524-020-0283-z
3. Schran, C., et al. (2020). *Committee neural network potentials control generalization errors and enable active learning.* Journal of Chemical Physics, 153(10), 104105. https://doi.org/10.1063/5.0016004
4. Smith, J. S., et al. (2018). *Less is more: Sampling chemical space with active learning.* Journal of Chemical Physics, 148(24), 241733. https://doi.org/10.1063/1.5023802
5. Imbalzano, G., et al. (2021). *Uncertainty estimation for molecular dynamics and sampling.* Journal of Chemical Physics, 154(7), 074102. https://doi.org/10.1063/5.0036522
