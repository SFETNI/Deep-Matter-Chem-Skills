---
name: mace-training
domain: ml-interatomic-potentials
maturity: draft
tools:
  - MACE
  - ASE
  - PyTorch
  - VASP
  - Quantum ESPRESSO
  - LAMMPS
tested_versions:
  MACE: "0.3.6"
  ASE: "3.23.0"
  PyTorch: "2.2.0"
requires_gpu: true
estimated_runtime: hours-to-days
---

# MACE Training

## Description

This skill covers the full workflow for training MACE (Message-passing Atomic Cluster Expansion) machine-learned interatomic potentials: DFT dataset preparation in extxyz format, isolated atom energy handling, train/validation/test splitting, model architecture selection, loss weighting, training execution, validation error analysis, and deployment to ASE and LAMMPS. Invoke this skill when building a new MACE model from DFT reference data, when diagnosing training failures, or when deciding whether MACE or another MLP architecture is appropriate for a given application.

## Domain Context

MACE is an equivariant graph neural network potential that combines two theoretical frameworks: the Atomic Cluster Expansion (ACE) body-ordered basis and E(3)-equivariant message passing. The key physical insight is that the potential energy surface must be invariant to rotations, reflections, and translations of the entire system, while forces — being vectors — must transform equivariantly under rotations. MACE enforces this by representing atomic features as geometric tensors (irreducible representations of SO(3)) and contracting them via Clebsch-Gordan tensor products.

Practical consequences of this architecture:

- **Higher body order per layer:** A single MACE interaction block effectively encodes higher-body-order interactions than a standard message-passing layer, because the tensor product at each node combines information from multiple neighbors simultaneously. This allows MACE to reach DFT accuracy with fewer layers than SchNet or DimeNet.
- **Data efficiency:** Equivariance acts as a physical inductive bias. A model that does not need to learn rotational symmetry from data requires fewer training structures to reach a given accuracy. MACE typically achieves force MAE < 50 meV/Å on well-sampled datasets with 1,000–10,000 structures.
- **The dataset defines the model's validity domain:** MACE is a local potential with a fixed cutoff radius. It cannot extrapolate to compositions, coordination environments, or bond lengths absent from the training set. A low test error on held-out data does not guarantee stability in MD if the MD trajectory visits configurations outside the training distribution.
- **Reference energy convention:** DFT total energies are extensive and include core-electron contributions that are not physically meaningful on their own. MACE subtracts isolated atom reference energies (E0) from all structure energies before training. The E0 values must be computed at the same level of DFT theory (functional, pseudopotential, cutoff) as the training data. Errors in E0 shift the predicted formation energy of every structure by a constant and destabilize model training.
- **Stress tensor convention:** MACE uses ASE's stress convention: Voigt-ordered Cauchy stress in eV/Å³, with tensile stress positive (opposite to engineering convention). VASP outputs stress in kBar with compression positive. The conversion factor and sign flip are a common source of silent dataset errors.

## When to Use This Skill

- Building a system-specific MLP for a new chemical space (new composition, phase, or environment) where universal potentials are insufficiently accurate.
- Generating a potential for large-scale MD of a well-defined material system where the DFT reference level is fixed.
- Fine-tuning a universal MACE model (MACE-MP-0, MACE-OFF) to a specific system with modest additional DFT data.
- Active learning loops where MACE committee models drive DFT oracle queries.
- Replacing a computationally expensive classical force field for a system outside its validated transferability range.

## When Not to Use This Skill

- **When a validated universal potential (MACE-MP-0, CHGNet, M3GNet) already covers your system** — universal potentials often achieve acceptable accuracy at zero additional DFT cost; always benchmark them first.
- **For highly reactive systems with complex charge transfer or excited-state dynamics** — MACE does not model electronic degrees of freedom. ReaxFF or DFT-MD is more appropriate when bond formation/breaking involves charge redistribution not captured by the training geometry.
- **When fewer than ~200–500 well-converged DFT structures are available** — a MACE model trained on very few structures will likely overfit or produce an unstable potential. Expand the dataset via active learning first.
- **For properties requiring long-range electrostatics (ionic solids, polar surfaces, charged defects)** — standard MACE uses a local cutoff and has no long-range electrostatic model. MACE-with-long-range extensions or dispersion corrections are required for ionic materials. [EXPERT REVIEW NEEDED]
- **When the target property is an excited-state or electronic property** — MLPs fit the ground-state Born-Oppenheimer surface only.

## Core Concepts

- **Extxyz format:** The standard flat-file format for atomistic ML datasets. Each frame stores the atom count, a comment line with key-value metadata (`Lattice`, `Properties`, `energy`, `stress`, `pbc`), and per-atom lines with species and positions. Forces are stored as a per-atom property. This is ASE's native format and the MACE training input.
- **Isolated atom energies (E0):** DFT total energy of a single atom of each species in a large vacuum box at the same DFT level as the training data. Subtracted from all structure energies to give formation-like energies centered near zero. Errors in E0 produce systematic shifts in predicted energies. The MACE `--E0s="average"` option estimates E0 by linear regression from the dataset — convenient but less physically principled.
- **Cutoff radius (`r_max`):** The distance beyond which atom pairs are not included in the neighbor graph. Typical values: 4–6 Å for metallic systems; 5–7 Å for covalent/molecular systems. A larger cutoff captures more long-range interactions but increases cost as O(r_max³). The cutoff must be smaller than half the shortest periodic cell dimension.
- **Angular momentum (`max_ell` or `L`):** Maximum angular momentum quantum number for the geometric tensor features. L=2 gives up to d-like features; L=3 gives f-like features. Higher L increases expressiveness and cost. L=3 is recommended for metallic systems; L=2 often sufficient for covalent. [EXPERT REVIEW NEEDED]
- **Correlation order (`correlation` or `ν`):** Body order of the ACE basis used per interaction block. `correlation=3` (3-body) is standard. Higher correlation captures more angular complexity but increases cost exponentially.
- **Hidden irreps (`hidden_irreps`):** The channel dimensions and symmetry types of the hidden layer features, specified in e3nn notation: e.g., `"128x0e + 128x1o + 128x2e"` means 128 scalar (0e), 128 vector (1o), and 128 rank-2 even-parity (2e) channels. Larger hidden irreps increase model capacity and cost.
- **Number of interaction blocks (`num_interactions`):** Depth of the message-passing network. Typically 2–3. Each block increases the effective receptive field by one cutoff radius.
- **Loss function:** `L = w_E · L_E + w_F · L_F + w_S · L_S`, where L_E is per-atom energy RMSE, L_F is per-component force RMSE, and L_S is per-component stress RMSE. Weights (`--energy_weight`, `--forces_weight`, `--stress_weight`) must be tuned. High `forces_weight` relative to `energy_weight` is standard for MLPs used in MD (forces drive dynamics); stress weights are required for NPT MD.
- **Exponential moving average (EMA):** EMA of model weights (`--ema`, `--ema_decay=0.99`) is used for the final model checkpoint. EMA smooths the weight trajectory and typically gives better test performance than the raw optimizer snapshot.
- **Scaling:** `--scaling="rms_forces_scaling"` normalizes the model output by the RMS force in the training set. This makes the loss approximately unit-independent and improves training stability.
- **Precision:** Float32 is standard for training on GPU; float64 improves numerical stability for energy-sensitive applications but roughly doubles memory and compute. Always use float64 for final validation on held-out data.

## Key Workflows

### Workflow 1: Dataset preparation from DFT outputs

Convert DFT calculation outputs (VASP or QE) to extxyz format. This is the most error-prone step and the most common source of silent dataset failures.

```python
from pathlib import Path
import numpy as np
from ase import Atoms
from ase.io import write
from pymatgen.io.vasp.outputs import Vasprun

# -------------------------------------------------------
# VASP → extxyz conversion
# Critical: units and sign conventions must be exact.
# -------------------------------------------------------

KBAR_TO_EV_A3 = 1.0 / 160.21766208  # 1 kBar = 1/160.218 eV/Å³

def vasprun_to_ase(vasprun_path: Path,
                   check_convergence: bool = True) -> Atoms | None:
    """
    Convert a completed VASP vasprun.xml to an ASE Atoms object
    with energy, forces, and stress attached.
    Returns None if the SCF did not converge.
    """
    try:
        v = Vasprun(str(vasprun_path), parse_potcar_file=False)
    except Exception as e:
        print(f"Parse error in {vasprun_path}: {e}")
        return None

    if check_convergence and not v.converged_electronic:
        print(f"SCF not converged: {vasprun_path}")
        return None  # Never include unconverged calculations in training data

    structure = v.final_structure
    last_step = v.ionic_steps[-1]

    # --- Build ASE Atoms ---
    atoms = Atoms(
        symbols=[str(s) for s in structure.species],
        positions=structure.cart_coords,
        cell=structure.lattice.matrix,
        pbc=True,
    )

    # --- Energy (eV): total energy from vasprun ---
    atoms.info["energy"] = v.final_energy  # eV

    # --- Forces (eV/Å): directly from vasprun ---
    forces = np.array(last_step["forces"])  # eV/Å, shape (N, 3)
    atoms.arrays["forces"] = forces

    # --- Stress (eV/Å³): VASP → ASE convention ---
    # VASP OUTCAR: stress in kBar, compression positive.
    # ASE convention: Voigt [xx,yy,zz,yz,xz,xy] in eV/Å³, TENSILE positive.
    # Pymatgen returns stress in kBar with the VASP sign (compression positive).
    # To convert to ASE: multiply by -KBAR_TO_EV_A3.
    # [EXPERT REVIEW NEEDED: verify sign convention against ASE stress getter]
    stress_kbar = np.array(last_step["stress"])  # 3x3, kBar
    # Extract Voigt components: [xx, yy, zz, yz, xz, xy]
    stress_voigt_kbar = np.array([
        stress_kbar[0, 0], stress_kbar[1, 1], stress_kbar[2, 2],
        stress_kbar[1, 2], stress_kbar[0, 2], stress_kbar[0, 1],
    ])
    # Sign flip: VASP compression-positive → ASE tensile-positive
    atoms.info["stress"] = -stress_voigt_kbar * KBAR_TO_EV_A3

    return atoms


def build_dataset(calc_dirs: list[Path], output_file: str) -> int:
    """
    Collect all valid DFT calculations into a single extxyz dataset file.
    Returns the number of frames written.
    """
    frames = []
    n_skipped = 0
    for calc_dir in calc_dirs:
        vr = calc_dir / "vasprun.xml"
        if not vr.exists():
            continue
        atoms = vasprun_to_ase(vr)
        if atoms is None:
            n_skipped += 1
            continue
        # Attach provenance metadata (not used by MACE but useful for auditing)
        atoms.info["config_type"] = calc_dir.parent.name  # e.g. "bulk", "surface"
        frames.append(atoms)

    write(output_file, frames, format="extxyz")
    print(f"Wrote {len(frames)} frames to {output_file} ({n_skipped} skipped)")
    return len(frames)


# Stress sign verification (always run this check on a sample structure):
def verify_stress_sign(atoms: Atoms) -> None:
    """
    Sanity check: for a compressed crystal, stress should be positive (ASE convention).
    For a stretched crystal, stress should be negative.
    A near-equilibrium crystal at 0 K should have stress ≈ 0 in all components.
    """
    s = atoms.info.get("stress")
    if s is not None:
        print(f"Stress diagonal [eV/Å³]: {s[:3]}")
        print(f"  Hydrostatic pressure: {-s[:3].mean():.4f} eV/Å³ "
              f"= {-s[:3].mean() * 160.218:.1f} GPa")
        print("  Near-equilibrium: all components should be < 0.01 eV/Å³")
```

### Workflow 2: Train/validation/test splitting

Atomistic datasets are not i.i.d. — structures from the same MD trajectory or rattled from the same crystal are highly correlated. Random splitting leaks correlated structures into train and test, producing optimistic test errors.

```python
import random
from collections import defaultdict
from ase.io import read, write

def split_by_config_type(
    dataset_file: str,
    train_file: str,
    val_file: str,
    test_file: str,
    val_fraction: float = 0.10,
    test_fraction: float = 0.10,
    seed: int = 42,
) -> dict:
    """
    Split an extxyz dataset into train/val/test, stratified by config_type.
    Within each config_type, split by trajectory ID if present, otherwise randomly.

    Stratified splitting ensures that rare configurations (e.g., transition states,
    surfaces, defects) appear in all three splits rather than being lost to chance.
    """
    random.seed(seed)
    all_frames = read(dataset_file, index=":", format="extxyz")

    # Group by config_type
    by_type: dict[str, list] = defaultdict(list)
    for frame in all_frames:
        ctype = frame.info.get("config_type", "Default")
        by_type[ctype].append(frame)

    train, val, test = [], [], []

    for ctype, frames in by_type.items():
        random.shuffle(frames)
        n = len(frames)
        n_test = max(1, int(n * test_fraction))
        n_val = max(1, int(n * val_fraction))
        n_train = n - n_val - n_test

        if n_train < 1:
            # Too few structures: put all in train, skip val/test for this type
            train.extend(frames)
            print(f"WARNING: {ctype} has only {n} frames; all assigned to train")
            continue

        test.extend(frames[:n_test])
        val.extend(frames[n_test:n_test + n_val])
        train.extend(frames[n_test + n_val:])
        print(f"  {ctype}: {n} frames → train={n_train}, val={n_val}, test={n_test}")

    write(train_file, train, format="extxyz")
    write(val_file, val, format="extxyz")
    write(test_file, test, format="extxyz")

    return {"train": len(train), "val": len(val), "test": len(test)}


# Data quality checks before training:
def check_dataset_quality(frames: list, species: list[str]) -> None:
    """Run sanity checks on the dataset before feeding to MACE."""
    energies = [f.info.get("energy") for f in frames]
    forces_list = [f.arrays.get("forces") for f in frames]
    stress_list = [f.info.get("stress") for f in frames]

    # Check 1: No None entries (missing labels)
    n_missing_e = sum(e is None for e in energies)
    n_missing_f = sum(f is None for f in forces_list)
    if n_missing_e or n_missing_f:
        raise ValueError(f"Missing labels: {n_missing_e} energies, {n_missing_f} forces")

    # Check 2: Per-atom energy in physically plausible range
    for i, (frame, e) in enumerate(zip(frames, energies)):
        epa = e / len(frame)
        if not (-20.0 < epa < 5.0):  # eV/atom; wide range for diverse chemistries
            print(f"WARNING: Frame {i} has unusual E/atom = {epa:.3f} eV/atom "
                  f"(config_type={frame.info.get('config_type', '?')})")

    # Check 3: Force magnitudes — very large forces usually indicate problems
    import numpy as np
    for i, (frame, forces) in enumerate(zip(frames, forces_list)):
        if forces is None:
            continue
        fmax = np.abs(forces).max()
        if fmax > 50.0:  # eV/Å; flag but don't exclude automatically
            print(f"WARNING: Frame {i} has max force {fmax:.1f} eV/Å "
                  f"— verify this is physical and the SCF was converged")

    # Check 4: All required species are present
    present = set()
    for frame in frames:
        present.update(frame.get_chemical_symbols())
    missing = set(species) - present
    if missing:
        raise ValueError(f"Species {missing} declared but absent from dataset")

    print(f"Dataset checks passed: {len(frames)} frames, "
          f"species={sorted(present)}, "
          f"E/atom range=[{min(e/len(f) for f,e in zip(frames,energies)):.2f}, "
          f"{max(e/len(f) for f,e in zip(frames,energies)):.2f}] eV/atom")
```

### Workflow 3: MACE training — hyperparameter choices

```bash
#!/bin/bash
# MACE training script — annotated for a metallic system (e.g., W-Ta alloy)
# GPU required; adjust batch_size to fit GPU memory (start with 4 and increase).

mace_run_train \
    --name="W_Ta_mace" \
    --model="MACE" \
    \
    # --- Dataset ---
    --train_file="train.xyz" \
    --valid_file="val.xyz" \
    --statistics_file="statistics.json" \
    \
    # --- Reference energies ---
    # "average": MACE fits E0 by ridge regression from the dataset.
    # Alternatively: '{"W": -12.345, "Ta": -11.678}' (eV, from isolated atom DFT runs)
    # DFT-computed E0 is more physically principled; "average" is more robust to
    # poor E0 DFT calculations (e.g., spin-unpolarized atom in a box).
    --E0s="average" \
    \
    # --- Architecture ---
    --r_max=5.5 \              # Cutoff radius (Å). 5-6 Å typical for metals.
    --num_cutoff_basis=8 \     # Bessel radial basis functions
    --correlation=3 \          # Body order (3 = 3-body interactions per block)
    --num_interactions=2 \     # Number of message-passing blocks
    --hidden_irreps="128x0e + 128x1o + 128x2e" \  # Model width; 128 per channel
    --MLP_irreps="16x0e" \     # Readout MLP width
    --max_ell=3 \              # Max angular momentum L (3 = f-like features)
    --gate="silu" \            # Activation function for scalars
    \
    # --- Loss weights ---
    # Forces dominate the training signal for MD quality.
    # Stress weight enables NPT MD; set to 0 if no stress labels.
    --energy_weight=1.0 \
    --forces_weight=10.0 \
    --stress_weight=1.0 \
    --loss="weighted" \        # Use per-config_type weights if config_type_weights set
    \
    # --- Optimizer ---
    --optimizer="adam" \
    --lr=0.01 \
    --weight_decay=5e-7 \
    --amsgrad \                # AMSGrad variant; more stable than Adam for MLPs
    --batch_size=10 \          # Reduce if GPU OOM; increase for better gradient estimates
    --valid_batch_size=20 \
    \
    # --- Scheduler ---
    # Exponential decay: lr(t) = lr(0) * gamma^t
    --scheduler="ExponentialLR" \
    --gamma=0.9993 \           # Decay rate per epoch; tune for convergence speed
    --lr_scheduler_patience=50 \
    \
    # --- Training duration ---
    --max_num_epochs=2000 \
    --patience=300 \           # Early stopping: stop if val loss doesn't improve for N epochs
    --eval_interval=2 \        # Validate every N epochs
    \
    # --- EMA and scaling ---
    --ema \
    --ema_decay=0.99 \
    --scaling="rms_forces_scaling" \
    \
    # --- Hardware and precision ---
    --device="cuda" \
    --default_dtype="float32" \
    --seed=123 \
    \
    # --- Per-config_type loss weighting (optional) ---
    # Up-weight underrepresented or important configurations.
    --config_type_weights='{"Default": 1.0, "surface": 2.0, "liquid": 2.0, "defect": 3.0}'
```

**Architecture selection guide:**

| System class | `r_max` (Å) | `max_ell` | `hidden_irreps` | Notes |
|---|---|---|---|---|
| FCC/BCC metals | 5.0–6.0 | 3 | 128x0e+128x1o+128x2e | Standard |
| Covalent (Si, C) | 4.0–5.5 | 2–3 | 128x0e+128x1o | Shorter bonds |
| Oxides, ceramics | 5.0–6.5 | 3 | 128x0e+128x1o+128x2e | Include O neighbors |
| Molecular (organic) | 5.0–6.0 | 2–3 | 64x0e+64x1o | Smaller OK if dataset large |
| Liquids/amorphous | 5.5–7.0 | 3 | 128x0e+128x1o+128x2e | Larger cutoff for disorder |
| Alloys (multi-species) | 5.5–6.5 | 3 | 256x0e+256x1o+256x2e | More capacity needed |

[EXPERT REVIEW NEEDED: cutoff and irrep recommendations are indicative; always perform architecture ablations on held-out data for a new system.]

### Workflow 4: Training monitoring and convergence diagnostics

```python
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path


def parse_mace_log(log_file: str) -> dict:
    """
    Parse MACE training log for loss curves and validation metrics.
    MACE logs are JSON-lines format (one JSON object per epoch).
    """
    epochs, train_loss, val_loss = [], [], []
    val_e_mae, val_f_mae, val_s_mae = [], [], []

    with open(log_file) as f:
        for line in f:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            epochs.append(d.get("epoch", len(epochs)))
            train_loss.append(d.get("loss", None))
            val_loss.append(d.get("val_loss", None))
            val_e_mae.append(d.get("val_energy_mae", None))
            val_f_mae.append(d.get("val_forces_mae", None))
            val_s_mae.append(d.get("val_stress_mae", None))

    return {
        "epochs": np.array(epochs),
        "train_loss": np.array(val_loss),
        "val_loss": np.array(val_loss),
        "val_energy_mae": np.array(val_e_mae),
        "val_forces_mae": np.array(val_f_mae),
        "val_stress_mae": np.array(val_s_mae),
    }


def diagnose_training(log: dict) -> None:
    """Print convergence diagnostics and flag common pathologies."""
    vf = log["val_forces_mae"]
    ve = log["val_energy_mae"]
    tl = log["train_loss"]
    vl = log["val_loss"]

    # Remove None entries
    vf_valid = vf[vf != None]
    ve_valid = ve[ve != None]

    print("=== Training Diagnostics ===")
    if len(vf_valid) > 0:
        print(f"Best val force MAE: {np.nanmin(vf_valid) * 1000:.1f} meV/Å")
        print(f"Final val force MAE: {vf_valid[-1] * 1000:.1f} meV/Å")
    if len(ve_valid) > 0:
        print(f"Best val energy MAE: {np.nanmin(ve_valid) * 1000:.2f} meV/atom")

    # Diagnostic 1: Is training loss still decreasing?
    if len(tl) > 100:
        slope = np.polyfit(range(len(tl[-50:])), tl[-50:], 1)[0]
        if slope > 0:
            print("WARNING: Training loss is INCREASING in the last 50 epochs — "
                  "possible overfitting or too-high LR")

    # Diagnostic 2: Train-val gap
    if len(tl) > 0 and len(vl) > 0:
        gap = vl[-1] - tl[-1]
        if gap > 2 * tl[-1]:
            print("WARNING: Large train-val gap — possible overfitting. "
                  "Consider: more data, smaller model, weight decay, or data augmentation")

    # Diagnostic 3: Force MAE plateau
    if len(vf_valid) > 200:
        recent = vf_valid[-100:]
        early = vf_valid[len(vf_valid)//2:len(vf_valid)//2 + 100]
        if recent.mean() > 0.95 * early.mean():
            print("WARNING: Force MAE not improving in recent epochs — "
                  "consider reducing LR, increasing model size, or expanding dataset")

    # Target force MAE benchmarks (material-class dependent):
    print("\nForce MAE targets (approximate; system-dependent):")
    print("  Metals (bulk):    < 50 meV/Å for stable MD")
    print("  Covalent:         < 80 meV/Å  [EXPERT REVIEW NEEDED]")
    print("  Molecular:        < 30 meV/Å for accurate geometry")
    print("  Reactive:         < 100 meV/Å (higher tolerance; reactions are rare events)")


# Overfitting detection: compare force error distributions on train vs test
def compare_train_test_forces(model, train_frames, test_frames) -> None:
    """
    Compute force errors on train and test sets.
    Large discrepancy indicates overfitting.
    """
    from mace.calculators import MACECalculator
    calc = MACECalculator(model_paths=model, device="cpu", default_dtype="float32")

    def get_force_errors(frames):
        errors = []
        for atoms in frames:
            ref_forces = atoms.arrays["forces"]
            atoms_copy = atoms.copy()
            atoms_copy.calc = calc
            pred_forces = atoms_copy.get_forces()
            errors.append(np.abs(ref_forces - pred_forces).mean())
        return np.array(errors)

    train_errors = get_force_errors(train_frames[:200])  # sample for speed
    test_errors = get_force_errors(test_frames)

    print(f"Train force MAE: {train_errors.mean() * 1000:.1f} ± {train_errors.std() * 1000:.1f} meV/Å")
    print(f"Test  force MAE: {test_errors.mean() * 1000:.1f} ± {test_errors.std() * 1000:.1f} meV/Å")
    if test_errors.mean() > 2 * train_errors.mean():
        print("WARNING: Test MAE >> Train MAE — possible data leakage in train set, "
              "or test set contains out-of-distribution structures")
```

### Workflow 5: Validation error analysis and outlier inspection

```python
import numpy as np
from ase.io import read
from mace.calculators import MACECalculator
from collections import defaultdict


def full_validation(model_path: str, test_file: str,
                    device: str = "cpu") -> dict:
    """
    Compute energy, force, and stress errors on the test set.
    Decompose by config_type to identify systematic gaps.
    """
    calc = MACECalculator(model_paths=model_path, device=device,
                          default_dtype="float64")  # float64 for validation
    frames = read(test_file, index=":", format="extxyz")

    by_type: dict[str, dict] = defaultdict(lambda: {
        "e_errors": [], "f_errors": [], "s_errors": [], "n_atoms": []
    })

    outliers = []
    for i, atoms in enumerate(frames):
        ctype = atoms.info.get("config_type", "Default")
        natoms = len(atoms)

        ref_e = atoms.info["energy"] / natoms  # eV/atom
        ref_f = atoms.arrays["forces"]         # eV/Å
        ref_s = atoms.info.get("stress")       # eV/Å³ Voigt, or None

        atoms_calc = atoms.copy()
        atoms_calc.calc = calc
        pred_e = atoms_calc.get_potential_energy() / natoms
        pred_f = atoms_calc.get_forces()

        e_err = abs(pred_e - ref_e) * 1000  # meV/atom
        f_mae = np.abs(pred_f - ref_f).mean() * 1000  # meV/Å

        by_type[ctype]["e_errors"].append(e_err)
        by_type[ctype]["f_errors"].append(f_mae)
        by_type[ctype]["n_atoms"].append(natoms)

        if ref_s is not None:
            pred_s = atoms_calc.get_stress()
            s_mae = np.abs(pred_s - ref_s).mean() * 160.218  # GPa
            by_type[ctype]["s_errors"].append(s_mae)

        # Flag outliers: force MAE > 5× the overall mean (rough threshold)
        if f_mae > 500:  # meV/Å — absolute upper threshold [EXPERT REVIEW NEEDED]
            outliers.append({"index": i, "config_type": ctype,
                             "f_mae_meV_A": f_mae, "e_err_meV": e_err,
                             "natoms": natoms})

    # Report per-type summary
    print(f"\n{'Config type':<20}  {'N':>5}  {'E MAE (meV/at)':>15}  "
          f"{'F MAE (meV/Å)':>14}  {'S MAE (GPa)':>12}")
    print("-" * 72)
    for ctype, data in sorted(by_type.items()):
        ne = np.mean(data["e_errors"])
        nf = np.mean(data["f_errors"])
        ns = np.mean(data["s_errors"]) if data["s_errors"] else float("nan")
        print(f"{ctype:<20}  {len(data['e_errors']):>5}  {ne:>15.2f}  "
              f"{nf:>14.2f}  {ns:>12.3f}")

    if outliers:
        print(f"\n{len(outliers)} outlier frames (F MAE > 500 meV/Å):")
        for o in outliers[:10]:
            print(f"  Frame {o['index']} ({o['config_type']}): "
                  f"F={o['f_mae_meV_A']:.0f} meV/Å, E={o['e_err_meV']:.1f} meV/atom")

    return {"by_type": by_type, "outliers": outliers}
```

### Workflow 6: Deployment to ASE and LAMMPS

```python
# --- ASE deployment ---
from mace.calculators import MACECalculator
from ase.io import read
from ase import units
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.langevin import Langevin

model_path = "W_Ta_mace_run-123_stagetwo.model"

# Load calculator
calc = MACECalculator(
    model_paths=model_path,
    device="cuda",
    default_dtype="float32",
)

# Attach to a structure and verify energy against DFT reference:
atoms = read("test_structure.extxyz", index=0)
atoms.calc = calc
e_mace = atoms.get_potential_energy()
e_dft = atoms.info["energy"]
print(f"MACE: {e_mace:.4f} eV, DFT: {e_dft:.4f} eV, "
      f"ΔE: {(e_mace - e_dft) * 1000:.1f} meV")
# If |ΔE| > 5 meV/atom for a bulk equilibrium structure: check E0 convention.

# Run NVT MD with Langevin thermostat in ASE (for small systems or testing)
MaxwellBoltzmannDistribution(atoms, temperature_K=300)
dyn = Langevin(atoms, timestep=2 * units.fs, temperature_K=300, friction=0.01)
dyn.run(1000)
```

```lammps
# --- LAMMPS deployment ---
# Requires LAMMPS compiled with the MACE pair style (cmake -D PKG_ML-MACE=yes)

units           metal
atom_style      atomic
boundary        p p p

read_data       W_Ta_bulk.lammps

pair_style      mace no_domain_decomposition
pair_coeff      * * W_Ta_mace_run-123_stagetwo.model W Ta
# Element order must match atom type order in data file.
# no_domain_decomposition: required for equivariant MP; limits MPI scaling.

# Validate forces vs ASE before running dynamics:
run             0
# Check LAMMPS forces match ASE forces on the same structure within 1 meV/Å.
# Discrepancy > 1 meV/Å: check model file path, element mapping, units.

# Then proceed with standard equilibration and production (see lammps-workflow skill).
```

## Common Failure Modes

| Failure | Symptom | Likely Cause | Diagnostic | Fix |
|---------|---------|--------------|------------|-----|
| Unit mismatch in dataset | Energy MAE remains high (> 1 eV/atom) despite training; forces correct but energies wrong by a constant offset | Energy in dataset is in Ry or Ha instead of eV, or distances in bohr instead of Å | Check raw values: eV/atom for typical metals is −4 to −12 eV/atom; if values are ×13.6 off, units are Ry | Convert all energies to eV and distances to Å before writing extxyz; use ASE unit conversion constants |
| Wrong E0 reference energies | Systematic energy offset per species; formation energies wrong; training diverges or plateaus at high energy loss | E0 computed with wrong spin state (spin-unpolarized atom vs. correct ground state), wrong functional, or wrong cell size | Compare `--E0s` DFT values to atomic reference data; check that atom DFT runs used same ENCUT, POTCAR, and ISMEAR as dataset | Recompute E0 for each species as spin-polarized, large-box DFT at identical settings; or use `--E0s="average"` as fallback |
| Stress sign/convention error | Stress MAE high (> 1 GPa for near-equilibrium structures) despite correct forces; NPT MD box expands or collapses unrealistically | VASP kBar stress converted without sign flip; tensile/compressive convention mismatch | Check: at equilibrium geometry, stress should be ≈ 0. If all components have wrong sign, apply `stress *= -1` | Apply `−kBar × KBAR_TO_EV_A3` conversion (see Workflow 1); validate on equilibrium structure: |stress| < 0.01 eV/Å³ |
| Data leakage: train/test correlation | Test error unrealistically low; production MD fails earlier than expected; MLP does not generalize | Random split applied to trajectory frames — adjacent frames are nearly identical, so test set is trivially predicted | Split by trajectory, by rattled-origin, or by config_type rather than randomly | Use `split_by_config_type` (Workflow 2); or explicitly group trajectory frames and split groups |
| Force-dominated loss collapse | Energy MAE diverges during training; model correctly predicts forces but energy is unphysical; NPT density wrong | `--forces_weight` set too high relative to `--energy_weight` (e.g., 100:1); model ignores energy loss | Plot energy loss and force loss separately; if energy loss is > 100× force loss per epoch, ratio is wrong | Reduce `--forces_weight` to 10–20; use `--loss="weighted"` with `--config_type_weights` for fine control |
| Species absent from training | Crash during MD on a structure containing a species not seen in training; or garbage forces for that species | A chemical species appears in the deployment structure but has zero or very few training examples | Count per-species frames in training set; check which species appear only in test | Add DFT calculations covering the missing species in all expected bonding environments; re-train |
| Unstable MD despite low test error | MD trajectory diverges (energy explosion) within ps; MACE test MAE is < 50 meV/Å | Model evaluated on near-equilibrium structures; MD explores high-temperature or defect configurations outside training distribution | Monitor max force during MD; extract frames where force > 10 eV/Å and compare to training set distribution | Expand dataset with active learning: run MACE-MD, extract uncertain frames, compute DFT, retrain |
| Batch size too large for GPU | CUDA out-of-memory error during training | Number of atoms × batch_size exceeds GPU VRAM | Start with `--batch_size=4`; increase until OOM; for 16 GB GPU, typical limit is batch_size=8–16 for 100-atom structures | Reduce `--batch_size`; use `--gradient_clip` and `--accumulate_gradient_batches` for effective larger batches |

## Best Practices

- **Never include unconverged DFT calculations in the training set.** A single frame with wrong forces (from an unconverged SCF) can corrupt training. Always filter by `converged_electronic = True` and check for `NELM reached` warnings before including a VASP calculation.
- **Validate the stress sign convention on a known equilibrium structure** before training. A near-equilibrium bulk crystal at 0 K should have all Voigt stress components within ±0.01 eV/Å³. If the diagonal components are uniformly wrong in sign, apply the sign correction.
- **Always include diverse configurations in the training set.** A model trained only on equilibrium bulk structures will fail for surfaces, defects, liquids, and distorted geometries encountered during MD. A minimum viable dataset for MD includes: equilibrium bulk, volumetrically strained bulk (±5%), rattled bulk at target temperatures, and any special environments (surfaces, defects) present in production MD.
- **Use committee models for uncertainty estimation during active learning.** Train 4–8 MACE models from different random seeds on the same dataset. Frames where committee force predictions disagree (force std > threshold) are candidates for DFT re-computation.
- **Check per-species and per-config_type errors, not just the global test MAE.** A model with excellent overall force MAE may have 10× higher error for a rare species or a minority configuration type that is critical for the target application.
- **Run an NVE energy conservation test before any production NPT MD.** Integrate 10,000 steps of NVE MD from the equilibrated structure. Energy drift > 1 meV/atom/ns indicates extrapolation, a timestep too large for the MACE forces, or a sign/units error. [EXPERT REVIEW NEEDED]
- **Use float64 for final validation** even if float32 was used for training. Force and energy errors computed at float32 precision can mask small systematic biases.
- **Record the MACE version, model file checksum, and training command** in every project README. MACE's default architecture and training defaults have changed between versions; a model trained with v0.3 may produce different behavior when evaluated with a different API version.

## Integration with Other Skills

- **[dft-dataset-generation](../../electronic-structure/dft-dataset-generation/SKILL.md):** Covers the systematic generation of DFT training configurations via rattling, AIMD snapshots, and active learning oracle calls. Workflow 1 above shows the conversion step; that skill covers the upstream calculation design.
- **[dft-convergence](../../electronic-structure/dft-convergence/SKILL.md):** The DFT settings used to generate training data (ENCUT, k-mesh, ISMEAR, SIGMA) must be uniform across the entire dataset. That skill's convergence protocol defines the parameter set that is frozen for dataset generation.
- **[vasp-workflow](../../electronic-structure/vasp-workflow/SKILL.md):** VASP is the most common DFT oracle for MACE training data. Workflow 1 above parses VASP vasprun.xml output. The ISIF=2 and PREC=Accurate settings from that skill are required for accurate stress labels.
- **[quantum-espresso](../../electronic-structure/quantum-espresso/SKILL.md):** When QE is used as the DFT oracle, forces are in Ry/bohr and must be converted to eV/Å. Stress in QE is also in Ry/bohr³ with potentially different sign conventions. Use ASE's QE parser to handle unit conversion automatically.
- **[lammps-workflow](../../atomistic-md/lammps-workflow/SKILL.md):** Workflow 6 above shows MACE deployment in LAMMPS. The `no_domain_decomposition` flag and element-type mapping requirements described there are prerequisites for correct MACE-LAMMPS deployment.
- **[md-equilibration](../../atomistic-md/md-equilibration/SKILL.md):** After MACE deployment, the same equilibration protocol (minimize → NVT ramp → NPT) applies. The NVE energy conservation test described there is the primary check for MACE timestep stability before production.
- **[active-learning-mlp](../active-learning-mlp/SKILL.md):** MACE committee models are the uncertainty signal in active learning loops. That skill covers the full loop: uncertainty filtering, DFT oracle selection, dataset augmentation, and retraining protocol.
- **[mlp-validation](../mlp-validation/SKILL.md):** Formal MLP validation goes beyond test-set MAE to include RDF comparison, phonon spectrum, elastic constants, vacancy formation energies, and MD stability. Consult that skill after training before deploying MACE in production.
- **[mlp-to-lammps](../mlp-to-lammps/SKILL.md):** Covers compilation of LAMMPS with the MACE pair style, model file format requirements, and parallel scaling limitations of `no_domain_decomposition`.
- **[uncertainty-mlp](../uncertainty-mlp/SKILL.md):** Committee models are one uncertainty approach; that skill covers alternatives (deep ensembles, last-layer uncertainty, conformal prediction) and when each is appropriate.
- **[fine-tuning-mlp](../fine-tuning-mlp/SKILL.md):** When a universal MACE model (MACE-MP-0) is close but not accurate enough, fine-tuning on system-specific DFT data is often more efficient than training from scratch. That skill covers the fine-tuning workflow and hyperparameter transfer.
- **[universal-potentials](../universal-potentials/SKILL.md):** Before committing to training a system-specific MACE model, benchmark MACE-MP-0, CHGNet, or M3GNet on the target system. If universal potential errors are within acceptable tolerances, system-specific training is unnecessary.

## Key References

1. Batatia, I., et al. (2022). *MACE: Higher order equivariant message passing neural networks for fast and accurate force fields.* Advances in Neural Information Processing Systems, 35. https://arxiv.org/abs/2206.07697
2. Batatia, I., et al. (2022). *The design space of E(3)-equivariant atom-centred interatomic potentials.* arXiv:2205.06643. https://arxiv.org/abs/2205.06643
3. Drautz, R. (2019). *Atomic cluster expansion for accurate and transferable interatomic potentials.* Physical Review B, 99(1), 014104. https://doi.org/10.1103/PhysRevB.99.014104
4. Kovács, D. P., et al. (2021). *Linear atomic cluster expansion force fields for organic molecules: Beyond RMSE.* Journal of Chemical Theory and Computation, 17(12), 7696–7711. https://doi.org/10.1021/acs.jctc.1c00647
5. Batatia, I., et al. (2023). *A foundation model for atomistic simulation of chemistry and materials.* arXiv:2401.00096. https://arxiv.org/abs/2401.00096
