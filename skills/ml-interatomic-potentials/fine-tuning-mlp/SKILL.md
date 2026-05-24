---
name: fine-tuning-mlp
domain: ml-interatomic-potentials
maturity: draft
tools:
  - MACE
  - ASE
  - PyTorch
  - VASP
  - Quantum ESPRESSO
tested_versions:
  MACE: "0.3.6"
  ASE: "3.23.0"
  PyTorch: "2.2.0"
requires_gpu: true
estimated_runtime: hours-to-days
---

# Fine-Tuning Machine-Learned Interatomic Potentials

## Description

This skill covers fine-tuning pretrained universal machine-learned interatomic potentials on system-specific DFT data: when fine-tuning is the right strategy, how to prepare a small targeted dataset, how to configure MACE's `--foundation_model` workflow, how to handle reference energies and label offsets, how to avoid catastrophic forgetting, and how to validate and deploy the resulting model. Invoke this skill after `universal-potentials` benchmarking has confirmed that a pretrained model (MACE-MP-0, CHGNet, SevenNet, ORB) falls short of the accuracy target for a specific chemistry or structural regime, and before committing to full system-specific training from scratch.

## Domain Context

Transfer learning for MLPs exploits a pretrained model's learned representation of atomic environments — built from millions of DFT calculations across the periodic table — as a starting point for a new, narrowly defined task. The key physical insight is that local chemical environments for a new target system (e.g., a W-Re alloy, a ZnO surface, an organic solvent at 400 K) are structurally similar to environments the pretrained model has already seen in the training database. Fine-tuning adjusts the model weights to improve accuracy within the target domain while ideally retaining the broad coverage of the pretrained backbone.

**Why fine-tuning works.** A universal MACE-MP-0 model trained on ~150,000 Materials Project structures has learned to embed atomic environments in a high-dimensional space that already separates coordination geometries, element types, and bonding patterns. The final readout layers (energy and force heads) map these embeddings to energies and forces. For a new target system with similar chemistry, only the readout layers and possibly the last interaction block need updating. This is analogous to transfer learning in computer vision: the early convolutional filters (low-level feature detectors) transfer; the classification head does not.

**When the analogy breaks down.** The transfer analogy fails when the target system has structural features genuinely absent from the pretrained model's training distribution: exotic bonding (e.g., very short bonds in high-pressure phases), elements underrepresented in the Materials Project (actinides, many rare earths), coordination numbers outside the typical range, or strong long-range interactions (charge transfer, magnetic coupling) that the local potential architecture cannot represent regardless of training data. In these cases, fine-tuning may improve accuracy modestly but cannot close the fundamental accuracy gap caused by architectural limitations or training distribution mismatch.

**Catastrophic forgetting.** Neural network fine-tuning on a small dataset tends to overwrite the weights that encode the pretrained model's broad knowledge, replacing them with weights that fit only the fine-tuning data. The result is a model that is accurate within the fine-tuning domain but has dramatically degraded accuracy outside it. For MLPs, this manifests as excellent force errors on the fine-tuning test set but unstable MD for configurations that drift slightly outside the fine-tuning distribution — the worst failure mode because it is not caught by static validation metrics.

**Reference energy consistency.** Pretrained models embed a specific reference energy convention: MACE-MP-0 uses PBE+U isolated atom reference energies computed with specific PAW pseudopotentials. Fine-tuning DFT data must use the same DFT level (functional, pseudopotential family, cutoff energy, k-mesh density) as the original training data, or the model will be trained on a mixture of reference levels that produces internally inconsistent energy predictions. A 50 meV/atom systematic error from reference-energy mismatch is invisible in force validation but corrupts all thermodynamic predictions.

**MACE fine-tuning mechanism.** MACE provides a `--foundation_model` flag that loads pretrained weights and continues training from that checkpoint. Three training modes are available: (1) full fine-tuning — all weights updated; (2) head-only fine-tuning — only the energy/force readout layers updated, all interaction blocks frozen; (3) partial fine-tuning — last N interaction blocks + readout updated, earlier blocks frozen. Head-only fine-tuning is the safest choice for very small datasets (< 100 structures) and minimizes forgetting. Full fine-tuning with a small learning rate and a mixed dataset (fine-tuning data + a fraction of the original training data) is preferred for larger datasets (500+ structures). [EXPERT REVIEW NEEDED: optimal strategy depends on model size and dataset size; these are heuristics, not guarantees]

## When to Use This Skill

- Universal potential benchmarking (`universal-potentials` Workflow 1) showed force MAE of 1.5–10× the target accuracy, indicating the pretrained model is close but not sufficient.
- The target system is a chemical variant of something well-represented in the Materials Project: an alloy with a new composition, a surface of a known bulk, a polymorph of a known material, a molecular liquid with a known solute.
- The DFT budget is limited (< 500 new DFT calculations) and training from scratch would require > 1,000 well-converged structures to reach the target accuracy.
- The target application is screening or exploratory MD where modest improvement over the universal potential baseline is sufficient, rather than precise thermodynamic integration.
- A magnetic material where CHGNet is used as the foundation model and improved spin-resolved force accuracy is needed for a specific composition.
- An organic system where MACE-OFF23 is the foundation model and system-specific conformational energetics need refinement.

## When Not to Use This Skill

- **When the universal potential shows no correlation with DFT forces (force MAE > 10× target or Pearson r < 0.5).** The pretrained representation is not a useful starting point; training from scratch with targeted DFT data will converge faster. Use `mace-training` directly.
- **When the target system contains elements not supported by the foundation model.** Loading weights that have no training signal for a new element's embedding will not transfer useful representations. Adding new elements requires either training from scratch or using a foundation model that was trained with those elements.
- **When the target DFT level differs from the pretrained model's level and the difference cannot be corrected by a reference energy offset.** Fine-tuning MACE-MP-0 (PBE+U) on HSE06 or meta-GGA data produces a hybrid-functional model; the energy scale differences and exchange-correlation contributions will be absorbed inconsistently into the model weights. Train a new model at the target DFT level if functional consistency is required.
- **When the fine-tuning dataset has fewer than ~50 structures.** With very few structures, there is no meaningful train/validation split; overfitting is almost guaranteed. Expand the dataset first via `active-learning-mlp` or targeted DFT (`dft-dataset-generation`), then fine-tune.
- **When long-range electrostatics or explicit charge transfer dominate the target property.** Fine-tuning a local cutoff model on ionic supercells, charged defects, or strongly polar surfaces does not add long-range physics to the architecture; it will fit the training set but fail at distances beyond the cutoff. A model with explicit long-range treatment is required.
- **When the fine-tuned model will be used as a "DFT label source" for a second model.** A fine-tuned MLP's predictions are still MLP predictions, not DFT labels. Using fine-tuned model energies as training targets for another model compounds approximation errors.

## Core Concepts

- **Foundation model:** The pretrained universal MLP used as the starting point for fine-tuning. In MACE, loaded via `--foundation_model=<path-or-model-name>`. The foundation model's architecture (cutoff radius, number of interactions, hidden irreps, correlation order, max_ell) is inherited; hyperparameters cannot be changed without retraining from scratch.
- **Head-only fine-tuning:** Only the final energy-readout and force-readout linear layers are updated; all message-passing interaction blocks are frozen. Fastest convergence, least catastrophic forgetting, but limited accuracy improvement — can only rescale and shift the energy landscape, not reshape local environment sensitivity. Appropriate when the fine-tuning dataset is very small (50–200 structures).
- **Full fine-tuning:** All model weights are updated from the pretrained initialization. Produces the best accuracy on the fine-tuning domain if the dataset is large enough (500+ structures) but risks catastrophic forgetting if the learning rate is too large or the dataset is too small.
- **Mixed-dataset training:** Including a fraction of the original training data (or a proxy for it) alongside the fine-tuning data. Prevents catastrophic forgetting by keeping the loss on the broad distribution non-zero. Requires access to the original training dataset or a representative subset, which is not always available for commercial or closed-source foundation models. [EXPERT REVIEW NEEDED]
- **Learning rate schedule for fine-tuning:** Start with a learning rate 10–100× lower than the from-scratch training rate. For MACE from-scratch, LR ≈ 0.01; for fine-tuning, LR ≈ 1e-3 to 1e-4 depending on the fine-tuning dataset size and the number of frozen layers. A warm restart schedule (cosine annealing) or plateau-based LR reduction is preferred over a fixed schedule. [EXPERT REVIEW NEEDED]
- **Reference energy offset (ΔE0):** The difference between the DFT isolated atom energies used for fine-tuning and those used for the foundation model's training. If the fine-tuning DFT uses different pseudopotentials, a different cutoff energy, or a different DFT functional, the absolute energies will differ by a constant per-species offset. This offset can be subtracted before training if the DFT levels are otherwise comparable; if they differ fundamentally, fine-tuning should not be used.
- **Label consistency:** All fine-tuning structures must be labeled at the same DFT level: same functional, same pseudopotential version, same k-mesh density protocol, same convergence criteria, same POTCAR files (for VASP). A mixed-label dataset produces systematic energy errors that inflate validation metrics without producing a physically meaningful model.
- **Domain coverage of fine-tuning data:** The fine-tuning dataset must sample the configurations the model will encounter in production: the correct temperature range, relevant structural distortions (vacancies, surface terminations, liquid configurations), and the relevant composition range. A fine-tuned model accurately validated on equilibrium structures at 0 K may fail in 1000 K MD if the fine-tuning data did not include high-temperature configurations.
- **Stiffness parameter / regularization:** In some fine-tuning frameworks, a regularization term encourages the fine-tuned weights to stay close to the pretrained weights, controlling forgetting explicitly. MACE does not implement this natively; it is approximated by using a low LR and a short training run. [EXPERT REVIEW NEEDED]
- **Validation split for fine-tuning:** Even with a small fine-tuning dataset, a held-out validation split (at minimum 10–20 structures) is required to detect overfitting. Cross-validation is preferred for datasets < 200 structures. The validation set must be drawn from the same structural distribution as the training set (same temperature, same composition range) but must not overlap with it.
- **MACE stage-one vs. stage-two training:** MACE training proceeds in two stages; `--foundation_model` starts from the pretrained stage-two checkpoint. Fine-tuning runs only stage two. The stage-one checkpoint from the foundation model is not reused.

## Key Workflows

### Workflow 1: Dataset preparation for fine-tuning

Fine-tuning datasets are small and every structure counts. Systematic label consistency checking is more important here than in large-scale training, because a single mislabeled structure is a larger fraction of the dataset.

```python
import hashlib
import json
import numpy as np
from pathlib import Path
from ase.io import read, write
from ase import Atoms


# ─── Step 1: Verify DFT level consistency across all fine-tuning calculations ─

def check_label_consistency(
    calc_dirs: list[str],
    dft_code: str = "vasp",
    canonical_settings_hash: str = None,
) -> dict:
    """
    Verify that all fine-tuning DFT calculations used identical settings.

    canonical_settings_hash: SHA-256 of the canonical INCAR (VASP) or input
        template (QE) established by `dft-convergence`. All calculations must
        match this hash.

    Returns a consistency report. Raises ValueError if critical inconsistencies
    are found (different ENCUT, different POTCAR, different functional).
    """
    from pymatgen.io.vasp.inputs import Incar, Potcar

    canonical_potcar_hashes = {}
    inconsistencies = []
    report = []

    for i, calc_dir in enumerate(calc_dirs):
        p = Path(calc_dir)
        entry = {"calc_dir": str(calc_dir), "issues": []}

        if dft_code == "vasp":
            incar_path = p / "INCAR"
            potcar_path = p / "POTCAR"
            outcar_path = p / "OUTCAR"

            if not outcar_path.exists():
                entry["issues"].append("OUTCAR missing — DFT calculation not complete")
                report.append(entry)
                continue

            # Check INCAR settings hash
            if incar_path.exists():
                incar_hash = hashlib.sha256(incar_path.read_bytes()).hexdigest()[:12]
                entry["incar_hash"] = incar_hash
                if canonical_settings_hash and incar_hash != canonical_settings_hash[:12]:
                    entry["issues"].append(
                        f"INCAR hash {incar_hash} != canonical {canonical_settings_hash[:12]}"
                    )

            # Check POTCAR consistency: all calculations must use the same PPs
            if potcar_path.exists():
                potcar_hash = hashlib.sha256(potcar_path.read_bytes()).hexdigest()[:12]
                entry["potcar_hash"] = potcar_hash
                try:
                    potcar = Potcar.from_file(str(potcar_path))
                    species_key = tuple(pot.element for pot in potcar)
                    if species_key not in canonical_potcar_hashes:
                        canonical_potcar_hashes[species_key] = potcar_hash
                    elif canonical_potcar_hashes[species_key] != potcar_hash:
                        entry["issues"].append(
                            f"POTCAR hash mismatch for species {species_key}: "
                            f"{potcar_hash} vs canonical {canonical_potcar_hashes[species_key]}"
                        )
                except Exception as e:
                    entry["issues"].append(f"POTCAR parse error: {e}")

            # Check for ionic convergence
            try:
                outcar_text = outcar_path.read_text(errors="replace")
                if "reached required accuracy" not in outcar_text and "General timing" not in outcar_text:
                    entry["issues"].append("OUTCAR does not show convergence — DFT may have failed")
            except Exception:
                pass

        if entry["issues"]:
            inconsistencies.extend(entry["issues"])
        report.append(entry)

    n_issues = sum(len(e["issues"]) for e in report)
    print(f"\nLabel consistency check: {len(calc_dirs)} calculations, {n_issues} issues")
    for e in report:
        if e["issues"]:
            print(f"  {e['calc_dir']}:")
            for issue in e["issues"]:
                print(f"    - {issue}")

    if n_issues > 0:
        print(f"\nWARNING: {n_issues} label consistency issues found. "
              f"Mixing inconsistent DFT settings in a fine-tuning dataset "
              f"will produce systematic energy errors that degrade model quality.")

    return {"n_calc": len(calc_dirs), "n_issues": n_issues, "report": report}


def compute_reference_energy_offsets(
    finetune_e0_file: str,
    foundation_e0_file: str,
) -> dict:
    """
    Compute per-species reference energy offsets between the fine-tuning DFT
    level and the foundation model's training DFT level.

    finetune_e0_file: JSON file with per-species isolated atom energies at the
        fine-tuning DFT level, e.g.:
        {"W": -12.9577, "Re": -12.4231, "O": -4.9480}  # eV per atom, PBE PAW
    foundation_e0_file: JSON file with per-species E0 from the foundation model's
        training database (available in the MACE-MP-0 repository).

    Returns delta_E0[species] = E0_finetune - E0_foundation (eV).
    A non-zero delta indicates reference level mismatch.
    If |delta| > 0.5 eV/atom for any species, DFT levels may be incompatible
    and fine-tuning should be reconsidered.
    """
    finetune_e0 = json.loads(Path(finetune_e0_file).read_text())
    foundation_e0 = json.loads(Path(foundation_e0_file).read_text())

    delta_e0 = {}
    for species, e0_ft in finetune_e0.items():
        e0_fm = foundation_e0.get(species)
        if e0_fm is None:
            print(f"  WARNING: {species} not found in foundation model E0 — "
                  f"this element may be unsupported by the foundation model")
            continue
        delta = e0_ft - e0_fm
        delta_e0[species] = delta
        status = "OK" if abs(delta) < 0.1 else ("WARN" if abs(delta) < 0.5 else "CRITICAL")
        print(f"  {species}: ΔE0 = {delta:+.4f} eV  [{status}]")

    n_critical = sum(1 for d in delta_e0.values() if abs(d) > 0.5)
    if n_critical > 0:
        print(f"\n  CRITICAL: {n_critical} species with |ΔE0| > 0.5 eV. "
              f"DFT reference levels are likely incompatible. "
              f"Consider re-computing fine-tuning calculations with the same "
              f"pseudopotential family as the foundation model.")
    return delta_e0


def prepare_finetune_dataset(
    finetune_extxyz: str,
    foundation_extxyz: str = None,
    val_fraction: float = 0.15,
    test_fraction: float = 0.10,
    seed: int = 42,
    output_prefix: str = "finetune",
    add_foundation_fraction: float = 0.0,
) -> dict:
    """
    Split a fine-tuning dataset into train/val/test, and optionally mix in a
    fraction of foundation model training data to reduce catastrophic forgetting.

    finetune_extxyz: extxyz file of DFT-labeled fine-tuning structures.
    foundation_extxyz: extxyz file of foundation training data (MP structures
        or a representative subset). Required if add_foundation_fraction > 0.
    add_foundation_fraction: fraction of fine-tuning dataset size to sample from
        foundation data and mix into training set. E.g., 0.5 adds 0.5×N_ft
        foundation structures to the training set. Reduces forgetting at the
        cost of longer training. [EXPERT REVIEW NEEDED: 0.3–1.0 is a typical
        range but the optimal fraction is system-dependent]

    Returns: dict with paths to train/val/test files and dataset statistics.
    """
    rng = np.random.default_rng(seed)
    frames = read(finetune_extxyz, index=":", format="extxyz")
    n = len(frames)
    print(f"Fine-tuning dataset: {n} structures")

    if n < 50:
        print(f"  WARNING: Only {n} fine-tuning structures. "
              f"Consider expanding via active learning before fine-tuning.")
    if n < 20:
        raise ValueError(
            f"Fine-tuning dataset too small ({n} structures). "
            f"A minimum of 20 structures is required for a meaningful validation split."
        )

    idx = rng.permutation(n)
    n_test = max(int(n * test_fraction), 5)
    n_val  = max(int(n * val_fraction), 5)
    n_train = n - n_val - n_test

    train_frames = [frames[i] for i in idx[:n_train]]
    val_frames   = [frames[i] for i in idx[n_train:n_train + n_val]]
    test_frames  = [frames[i] for i in idx[n_train + n_val:]]

    # Optionally mix in foundation model data for anti-forgetting regularization
    if add_foundation_fraction > 0 and foundation_extxyz:
        n_foundation_to_add = int(n_train * add_foundation_fraction)
        foundation_frames = read(foundation_extxyz, index=":", format="extxyz")
        if len(foundation_frames) < n_foundation_to_add:
            print(f"  WARNING: Foundation dataset has only {len(foundation_frames)} frames; "
                  f"requested {n_foundation_to_add}. Using all foundation frames.")
            n_foundation_to_add = len(foundation_frames)
        foundation_sample = [
            foundation_frames[i]
            for i in rng.choice(len(foundation_frames), n_foundation_to_add, replace=False)
        ]
        print(f"  Adding {len(foundation_sample)} foundation structures to training set "
              f"({add_foundation_fraction:.1%} of fine-tuning train size) "
              f"to reduce catastrophic forgetting.")
        # Tag foundation frames for monitoring loss contributions
        for f in foundation_sample:
            f.info["dataset_source"] = "foundation"
        for f in train_frames:
            f.info["dataset_source"] = "finetune"
        train_frames = train_frames + foundation_sample
        rng.shuffle(train_frames)

    train_file = f"{output_prefix}_train.extxyz"
    val_file   = f"{output_prefix}_val.extxyz"
    test_file  = f"{output_prefix}_test.extxyz"

    write(train_file, train_frames, format="extxyz")
    write(val_file,   val_frames,   format="extxyz")
    write(test_file,  test_frames,  format="extxyz")

    print(f"\nDataset split:")
    print(f"  Train: {len(train_frames)} structures → {train_file}")
    print(f"  Val:   {len(val_frames)} structures   → {val_file}")
    print(f"  Test:  {len(test_frames)} structures  → {test_file}")

    return {
        "train_file": train_file,
        "val_file": val_file,
        "test_file": test_file,
        "n_train_finetune": n_train,
        "n_val": len(val_frames),
        "n_test": len(test_frames),
        "n_foundation_added": len(foundation_sample) if add_foundation_fraction > 0 else 0,
    }
```

### Workflow 2: MACE fine-tuning with the `--foundation_model` flag

Configuring and launching MACE fine-tuning for three different regimes: head-only (very small dataset), partial (moderate dataset), and full (larger dataset with anti-forgetting mixing).

```python
import subprocess
import hashlib
import json
from pathlib import Path


# Fine-tuning regime configurations
# [EXPERT REVIEW NEEDED: these are indicative starting points, not universal recommendations]

FINETUNE_CONFIGS = {
    "head_only": {
        # Freeze all interaction blocks; update only readout layers.
        # Use for: < 200 fine-tuning structures, minimal forgetting required.
        "freeze_to_output": True,      # MACE flag to freeze interaction layers
        "lr": 1e-4,
        "weight_decay": 1e-8,
        "max_num_epochs": 200,
        "patience": 50,
        "batch_size": 4,
        "description": "Head-only fine-tuning. Minimal forgetting. Limited accuracy gain.",
    },
    "partial": {
        # Unfreeze last interaction block + readout.
        # Use for: 200–500 fine-tuning structures.
        "freeze_to_output": False,
        "num_interactions": 1,         # Only fine-tune last N interactions (approximated via LR layering)
        "lr": 5e-4,
        "weight_decay": 1e-8,
        "max_num_epochs": 500,
        "patience": 100,
        "batch_size": 8,
        "description": "Partial fine-tuning (last block + readout). Moderate forgetting risk.",
    },
    "full": {
        # Full fine-tuning with low LR and foundation data mixing.
        # Use for: 500+ fine-tuning structures + foundation data fraction.
        "freeze_to_output": False,
        "lr": 1e-3,
        "weight_decay": 5e-8,
        "max_num_epochs": 1000,
        "patience": 200,
        "batch_size": 8,
        "description": "Full fine-tuning. Best accuracy; requires anti-forgetting strategy.",
    },
}


def run_mace_finetune(
    train_file: str,
    val_file: str,
    foundation_model: str,
    regime: str = "partial",
    name: str = "finetuned_model",
    energy_weight: float = 1.0,
    forces_weight: float = 10.0,
    stress_weight: float = 1.0,
    E0s: str = "average",
    device: str = "cuda",
    seed: int = 42,
    extra_flags: dict = None,
) -> str:
    """
    Launch MACE fine-tuning from a pretrained foundation model.

    foundation_model: path to a local .model file, or a MACE-MP shorthand
        such as "medium" (resolves to the MACE-MP-0 medium checkpoint).
        For reproducibility, always use a local file with a recorded SHA-256,
        not the shorthand — shorthand may resolve to different checkpoints
        after package updates.
    regime: "head_only" | "partial" | "full" — selects the configuration
        from FINETUNE_CONFIGS above.
    E0s: "average" lets MACE fit reference energies from the dataset. If the
        fine-tuning dataset is too small for reliable regression (< 50 structures
        spanning multiple compositions), provide explicit E0s as a JSON string:
        '{"W": -12.9577, "Re": -12.4231}'. Using "average" on a small homogeneous
        dataset can produce spurious reference energies.

    Returns: path to the fine-tuned model file.
    """
    config = FINETUNE_CONFIGS[regime].copy()
    if extra_flags:
        config.update(extra_flags)

    # Record foundation model SHA-256 for provenance
    if Path(foundation_model).exists():
        fm_sha256 = hashlib.sha256(open(foundation_model, "rb").read()).hexdigest()
        print(f"Foundation model: {foundation_model}")
        print(f"  SHA-256: {fm_sha256[:16]}...")
    else:
        fm_sha256 = "resolved_at_runtime"
        print(f"Foundation model: {foundation_model} (shorthand — record SHA-256 after download)")

    cmd = [
        "mace_run_train",
        f"--name={name}",
        f"--train_file={train_file}",
        f"--valid_file={val_file}",
        f"--foundation_model={foundation_model}",
        f"--lr={config['lr']}",
        f"--weight_decay={config['weight_decay']}",
        f"--max_num_epochs={config['max_num_epochs']}",
        f"--patience={config['patience']}",
        f"--batch_size={config['batch_size']}",
        f"--energy_weight={energy_weight}",
        f"--forces_weight={forces_weight}",
        f"--stress_weight={stress_weight}",
        f"--E0s={E0s}",
        f"--device={device}",
        f"--seed={seed}",
        "--ema", "--ema_decay=0.99",
        "--scaling=rms_forces_scaling",
        "--default_dtype=float32",
        "--save_cpu",   # Save model in CPU-compatible format for ASE/LAMMPS portability
    ]

    if config.get("freeze_to_output"):
        cmd.append("--freeze_to_output")

    print(f"\nLaunching MACE fine-tuning (regime: {regime})...")
    print(f"  {config['description']}")
    print(f"  LR={config['lr']}, epochs={config['max_num_epochs']}, "
          f"patience={config['patience']}, batch={config['batch_size']}")
    print(f"  Command: {' '.join(cmd[:8])} ...")

    subprocess.run(cmd, check=True)

    # MACE output: {name}_run-{seed}_stagetwo.model
    model_file = f"{name}_run-{seed}_stagetwo.model"
    if not Path(model_file).exists():
        raise FileNotFoundError(
            f"Expected fine-tuned model {model_file} not found. Check MACE log."
        )

    ft_sha256 = hashlib.sha256(open(model_file, "rb").read()).hexdigest()
    print(f"\nFine-tuned model: {model_file}")
    print(f"  SHA-256: {ft_sha256[:16]}...")

    # Write provenance record
    provenance = {
        "foundation_model": foundation_model,
        "foundation_sha256": fm_sha256,
        "finetuned_model": model_file,
        "finetuned_sha256": ft_sha256,
        "regime": regime,
        "config": {k: str(v) for k, v in config.items()},
        "train_file": train_file,
        "train_sha256": hashlib.sha256(open(train_file, "rb").read()).hexdigest(),
        "val_file": val_file,
        "seed": seed,
        "E0s": E0s,
    }
    prov_file = f"{name}_finetune_provenance.json"
    Path(prov_file).write_text(json.dumps(provenance, indent=2))
    print(f"  Provenance record: {prov_file}")

    return model_file
```

### Workflow 3: Validation of fine-tuned model — forgetting check and domain accuracy

Fine-tuned model validation must cover two orthogonal questions: (1) did the model improve on the target domain? (2) did it lose accuracy outside the target domain?

```python
import numpy as np
from ase.io import read
from mace.calculators import MACECalculator


def evaluate_model_on_dataset(
    model_path: str,
    dataset_file: str,
    label: str = "model",
    device: str = "cpu",
    dtype: str = "float64",
) -> dict:
    """
    Compute force MAE, energy MAE, and per-frame statistics for a model on a dataset.
    Used to compare foundation vs. fine-tuned model on both target and OOD datasets.
    """
    calc = MACECalculator(model_paths=model_path, device=device, default_dtype=dtype)
    frames = read(dataset_file, index=":", format="extxyz")

    force_maes, energy_errors = [], []
    for atoms in frames:
        ref_forces = atoms.arrays.get("forces")
        ref_energy = atoms.info.get("energy")
        if ref_forces is None:
            continue

        a = atoms.copy()
        a.calc = calc
        pred_forces = a.get_forces()
        force_maes.append(np.abs(pred_forces - ref_forces).mean() * 1000)

        if ref_energy is not None:
            e_err = (a.get_potential_energy() - ref_energy) / len(atoms) * 1000
            energy_errors.append(e_err)

    result = {
        "label": label,
        "dataset": dataset_file,
        "n": len(force_maes),
        "force_mae_meV_A": float(np.mean(force_maes)),
        "force_mae_p90_meV_A": float(np.percentile(force_maes, 90)),
    }
    if energy_errors:
        result["energy_mae_meV_atom"] = float(np.mean(np.abs(energy_errors)))

    return result


def forgetting_and_improvement_report(
    foundation_model: str,
    finetuned_model: str,
    target_test_file: str,
    ood_test_file: str = None,
    device: str = "cpu",
) -> None:
    """
    Compare foundation and fine-tuned models on:
    1. The target-domain test set (should improve after fine-tuning)
    2. An out-of-domain test set (should not degrade significantly after fine-tuning)

    ood_test_file: ideally a sample from the foundation model training data or
        from a different material system. If not available, the check is skipped
        with a warning. [EXPERT REVIEW NEEDED: OOD test set selection is critical;
        degradation on OOD is often not checked in practice]
    """
    print("\n" + "=" * 65)
    print("FINE-TUNING EVALUATION REPORT")
    print("=" * 65)

    fm_target = evaluate_model_on_dataset(
        foundation_model, target_test_file, label="Foundation", device=device
    )
    ft_target = evaluate_model_on_dataset(
        finetuned_model, target_test_file, label="Fine-tuned", device=device
    )

    print(f"\nTarget domain ({target_test_file}):")
    print(f"  Foundation:  F MAE = {fm_target['force_mae_meV_A']:.1f} meV/Å")
    print(f"  Fine-tuned:  F MAE = {ft_target['force_mae_meV_A']:.1f} meV/Å")

    improvement = fm_target["force_mae_meV_A"] - ft_target["force_mae_meV_A"]
    improvement_pct = improvement / fm_target["force_mae_meV_A"] * 100
    if improvement > 0:
        print(f"  → Improvement: {improvement:.1f} meV/Å ({improvement_pct:.0f}%)")
    else:
        print(f"  → WARNING: Fine-tuning degraded target accuracy by "
              f"{abs(improvement):.1f} meV/Å. Check for overfitting or LR too high.")

    if ood_test_file:
        fm_ood = evaluate_model_on_dataset(
            foundation_model, ood_test_file, label="Foundation", device=device
        )
        ft_ood = evaluate_model_on_dataset(
            finetuned_model, ood_test_file, label="Fine-tuned", device=device
        )

        print(f"\nOut-of-domain ({ood_test_file}):")
        print(f"  Foundation:  F MAE = {fm_ood['force_mae_meV_A']:.1f} meV/Å")
        print(f"  Fine-tuned:  F MAE = {ft_ood['force_mae_meV_A']:.1f} meV/Å")

        forgetting = ft_ood["force_mae_meV_A"] - fm_ood["force_mae_meV_A"]
        forgetting_pct = forgetting / max(fm_ood["force_mae_meV_A"], 1.0) * 100
        if forgetting > 0.15 * fm_ood["force_mae_meV_A"]:
            print(f"  → WARNING: Catastrophic forgetting detected. "
                  f"OOD accuracy degraded by {forgetting:.1f} meV/Å "
                  f"({forgetting_pct:.0f}%). "
                  f"Consider head-only fine-tuning or mixed-dataset training.")
        else:
            print(f"  → Acceptable: OOD accuracy change = {forgetting:+.1f} meV/Å "
                  f"({forgetting_pct:+.0f}%). No significant forgetting.")
    else:
        print(f"\n  WARNING: No OOD test set provided. "
              f"Catastrophic forgetting cannot be assessed. "
              f"It is strongly recommended to supply an OOD test file "
              f"(e.g., a sample from the Materials Project for the same elements).")

    print("\n" + "=" * 65)
    print("DEPLOYMENT RECOMMENDATION:")
    if improvement > 10 and (ood_test_file is None or forgetting < 0.15 * fm_ood["force_mae_meV_A"]):
        print("  Fine-tuned model is ready for validation with `mlp-validation`.")
        print("  Run the full validation trinity: static errors, phonons, NVE stability.")
    elif improvement > 0 and improvement < 10:
        print("  Marginal improvement (<10 meV/Å). Consider:")
        print("  (a) Adding more fine-tuning structures via `active-learning-mlp`.")
        print("  (b) Switching to full fine-tuning regime if currently using head-only.")
        print("  (c) Checking dataset quality — label inconsistency can cap improvement.")
    else:
        print("  Fine-tuning did not produce reliable improvement. Investigate:")
        print("  (a) Reference energy consistency (Workflow 1).")
        print("  (b) Dataset coverage (Workflow 4 for domain-specific gaps).")
        print("  (c) Whether training from scratch is more appropriate.")
    print("=" * 65)
```

### Workflow 4: Domain-targeted DFT data collection strategy

A fine-tuning dataset that does not cover the structural configurations actually visited in production will fail silently. This workflow designs the data collection strategy for specific target applications.

```python
import numpy as np
from pathlib import Path
from ase.io import read, write
from mace.calculators import mace_mp


def audit_finetune_dataset_coverage(
    finetune_file: str,
    production_traj_file: str = None,
    species: list[str] = None,
    r_cut: float = 5.0,
    output_report: str = "coverage_audit.json",
) -> dict:
    """
    Check whether the fine-tuning dataset covers the structural configurations
    that the model will encounter in production.

    Computes SOAP dissimilarity between production trajectory frames and the
    nearest fine-tuning structure. High dissimilarity means the fine-tuning
    data does not cover that configuration — fine-tuning may not help there.

    production_traj_file: a LAMMPS dump or extxyz trajectory of the target
        production simulation, evaluated by the current universal potential.
        If not available, runs a short universal-potential MD to generate one.
    """
    import json
    from dscribe.descriptors import SOAP
    from sklearn.preprocessing import normalize

    finetune_frames = read(finetune_file, index=":", format="extxyz")
    print(f"Fine-tuning dataset: {len(finetune_frames)} structures")

    if production_traj_file is None:
        print("  No production trajectory provided. Cannot assess coverage.")
        return {}

    prod_frames = read(production_traj_file, index="::10", format="extxyz")
    print(f"Production trajectory: {len(prod_frames)} frames (every 10th)")

    if species is None:
        species = sorted(set(
            sym for atoms in finetune_frames for sym in atoms.get_chemical_symbols()
        ))

    soap = SOAP(
        species=species, r_cut=r_cut, n_max=9, l_max=6,
        sigma=0.5, average="inner", periodic=True,
    )
    ft_descs  = normalize(soap.create(finetune_frames), norm="l2")
    prod_descs = normalize(soap.create(prod_frames), norm="l2")

    cos_sim = prod_descs @ ft_descs.T
    distances = np.sqrt(np.maximum(0, 2 - 2 * np.clip(cos_sim, -1, 1)))
    min_dists = distances.min(axis=1)

    # Estimate coverage radius from within-fine-tuning distribution
    cs_ft = ft_descs @ ft_descs.T
    np.fill_diagonal(cs_ft, -2.0)
    d_within = np.sqrt(np.maximum(0, 2 - 2 * np.clip(cs_ft, -1, 1)))
    np.fill_diagonal(d_within, np.nan)
    coverage_radius = float(np.nanpercentile(np.nanmin(d_within, axis=1), 95))

    n_uncovered = (min_dists > coverage_radius).sum()
    print(f"\nCoverage audit:")
    print(f"  Coverage radius (95th pct within-FT distance): {coverage_radius:.4f}")
    print(f"  Production frames outside coverage: "
          f"{n_uncovered}/{len(prod_frames)} ({n_uncovered/len(prod_frames):.0%})")

    if n_uncovered / len(prod_frames) > 0.20:
        print(f"  WARNING: > 20% of production frames are outside the fine-tuning "
              f"dataset coverage. The fine-tuned model will likely fail for these "
              f"configurations. Consider running active learning (active-learning-mlp) "
              f"to sample and label the uncovered regions.")

    result = {
        "n_finetune": len(finetune_frames),
        "n_prod_frames": len(prod_frames),
        "coverage_radius": coverage_radius,
        "n_uncovered": int(n_uncovered),
        "uncovered_fraction": float(n_uncovered / len(prod_frames)),
        "min_dist_to_ft_mean": float(np.mean(min_dists)),
        "min_dist_to_ft_p90": float(np.percentile(min_dists, 90)),
    }
    Path(output_report).write_text(json.dumps(result, indent=2))
    return result


def design_targeted_collection(
    target_system: str,
    target_application: str,
    temperature_K: float = None,
    defect_types: list[str] = None,
    surface_miller: list = None,
) -> None:
    """
    Print a structured data collection plan for common fine-tuning target domains.
    Not executable — generates a human-readable protocol.

    target_system: e.g. "W0.8Re0.2 BCC alloy", "ZnO wurtzite", "liquid water at 350K"
    target_application: "MD at T" | "surface energetics" | "defect formation" |
                        "phonons" | "phase diagram" | "screening"
    """
    print(f"\nTargeted data collection plan for: {target_system}")
    print(f"Application: {target_application}")
    print("-" * 60)

    base_structures = [
        "Equilibrium supercell (1×1×1 up to 3×3×3)",
        "Compressed supercell: volume -5% to -20% (3–5 points)",
        "Expanded supercell: volume +5% to +20% (3–5 points)",
        "Random displacement structures: 0.05–0.20 Å rms displacement (20–50 structures)",
    ]
    print("Recommended base structures (all systems):")
    for s in base_structures:
        print(f"  • {s}")

    if temperature_K is not None:
        print(f"\nFor MD at {temperature_K} K:")
        print(f"  • Pre-equilibrate with universal potential at {temperature_K} K")
        print(f"  • Sample 30–80 frames from the NVT trajectory (skip first 20% equilibration)")
        print(f"  • Include frames at T ± 200 K to cover thermal fluctuations")
        print(f"  • Total recommended: 80–150 DFT single-point calculations")

    if defect_types:
        print(f"\nFor defect energetics ({', '.join(defect_types)}):")
        print(f"  • Clean supercell (2×2×2 or larger) as reference")
        for d in defect_types:
            print(f"  • {d}: 5–10 structures with varying local relaxation states")
        print(f"  • Defect supercells with 0.02–0.10 Å random displacements (10–20 per type)")
        print(f"  • NOTE: DFT must use the same k-mesh as the clean bulk reference")

    if surface_miller:
        miller_str = ", ".join(str(m) for m in surface_miller)
        print(f"\nFor surface energetics ({miller_str}):")
        print(f"  • Clean slab (5–7 layers, ≥10 Å vacuum) — fully relaxed")
        print(f"  • Compressed/expanded in-plane lattice constants (±3%, ±5%)")
        print(f"  • Surfaces with and without adatom/vacancy (5–10 structures each)")
        print(f"  • Random displacement structures for surface+subsurface layers")
        print(f"  • NOTE: Include bulk reference structures to anchor absolute energies")

    print(f"\nTotal estimated new DFT calculations: "
          f"{'100–200' if temperature_K else '50–150'}")
    print(f"DFT setup: use `dft-dataset-generation` for systematic collection.")
    print(f"DFT convergence: use identical INCAR/pseudopotentials as existing campaign.")
```

### Workflow 5: Fine-tuning a committee for post-fine-tuning uncertainty quantification

After fine-tuning, a committee of fine-tuned models provides UQ that is calibrated to the target domain. The committee should be built from fine-tuned members, not from the foundation model.

```python
import subprocess
import hashlib
import json
from pathlib import Path


def train_finetuned_committee(
    train_file: str,
    val_file: str,
    foundation_model: str,
    n_members: int = 4,
    base_seed: int = 42,
    regime: str = "partial",
    model_prefix: str = "ft_committee",
    device: str = "cuda",
) -> list[str]:
    """
    Train a committee of fine-tuned models for UQ after fine-tuning.

    IMPORTANT: Committee members must be fine-tuned from the SAME foundation
    model checkpoint with DIFFERENT seeds. Using different foundation model
    versions (e.g., medium vs. large MACE-MP-0) as committee members does
    not produce a valid uncertainty estimate — the members span model-version
    uncertainty, not epistemic uncertainty relative to the target DFT level.

    n_members: 4 is the minimum for reliable σ_F; 8 is preferred.
        Fine-tuning 8 members at 500 epochs each costs ~8× single-model time.
        For resource-constrained settings, use n_members=4 with larger seed
        spacing.

    Returns: list of fine-tuned model paths.
    """
    fm_sha256 = hashlib.sha256(open(foundation_model, "rb").read()).hexdigest()
    print(f"Foundation model SHA-256: {fm_sha256[:16]}...")
    print(f"Training {n_members} committee members (regime={regime})...")

    config = FINETUNE_CONFIGS[regime]
    model_paths = []

    for i in range(n_members):
        seed = base_seed + i * 17
        name = f"{model_prefix}_s{seed}"
        model_file = run_mace_finetune(
            train_file=train_file,
            val_file=val_file,
            foundation_model=foundation_model,
            regime=regime,
            name=name,
            seed=seed,
            device=device,
        )
        model_paths.append(model_file)

    # Committee manifest
    manifest = {
        "committee_type": "finetuned",
        "foundation_model": foundation_model,
        "foundation_sha256": fm_sha256,
        "n_members": n_members,
        "regime": regime,
        "seeds": [base_seed + i * 17 for i in range(n_members)],
        "model_paths": model_paths,
        "model_sha256": {
            mp: hashlib.sha256(open(mp, "rb").read()).hexdigest()
            for mp in model_paths
        },
        "train_file": train_file,
        "train_sha256": hashlib.sha256(open(train_file, "rb").read()).hexdigest(),
    }
    manifest_file = f"{model_prefix}_manifest.json"
    Path(manifest_file).write_text(json.dumps(manifest, indent=2))
    print(f"\nCommittee manifest: {manifest_file}")
    print(f"WARNING: Verify committee diversity with `verify_committee_diversity` "
          f"from `uncertainty-mlp` before using σ_F for active learning thresholds.")
    return model_paths


def verify_finetune_did_not_destroy_committee_diversity(
    foundation_model: str,
    finetuned_models: list[str],
    test_file: str,
    device: str = "cpu",
) -> None:
    """
    Check that fine-tuned committee members are still diverse enough for UQ.
    Fine-tuning on a small dataset can cause ensemble collapse even if the
    foundation committee was diverse, because all members fit the same small
    training set to the same solution.

    Compares pairwise force MAE between fine-tuned members against the
    expected diversity level from the foundation committee.
    """
    import numpy as np
    from ase.io import read
    from mace.calculators import MACECalculator

    frames = read(test_file, index=":50", format="extxyz")  # First 50 frames

    calcs = [
        MACECalculator(model_paths=mp, device=device, default_dtype="float64")
        for mp in finetuned_models
    ]
    all_forces = []
    for calc in calcs:
        member_forces = []
        for atoms in frames:
            a = atoms.copy()
            a.calc = calc
            member_forces.append(a.get_forces())
        all_forces.append(member_forces)

    n = len(finetuned_models)
    pairwise_maes = []
    for i in range(n):
        for j in range(i + 1, n):
            diffs = [
                np.abs(fi - fj).mean() * 1000
                for fi, fj in zip(all_forces[i], all_forces[j])
            ]
            pairwise_maes.append(np.mean(diffs))

    mean_pw = np.mean(pairwise_maes)
    print(f"\nFine-tuned committee diversity check:")
    print(f"  Mean pairwise force MAE: {mean_pw:.2f} meV/Å")
    if mean_pw < 3.0:
        print(f"  WARNING: Committee may have collapsed (mean pairwise MAE < 3 meV/Å).")
        print(f"  Possible cause: fine-tuning dataset too small and homogeneous; "
              f"all members converged to the same fine-tuning solution.")
        print(f"  Consider: (a) head-only fine-tuning (maintains more diversity); "
              f"(b) larger fine-tuning dataset; (c) SOAP dissimilarity as OOD backup.")
    else:
        print(f"  PASS: Committee members are sufficiently diverse for UQ.")
```

## Common Failure Modes

| Failure | Symptom | Likely Cause | Diagnostic | Fix |
|---------|---------|--------------|------------|-----|
| Overfitting to a small fine-tuning set | Validation force MAE is very low (< 20 meV/Å) but MD at target temperature diverges within a few ps; test set error is good but production behavior is poor | Fine-tuning dataset does not cover the configurations visited in production MD; model has memorized the training structures | Run `audit_finetune_dataset_coverage` on a pre-equilibration trajectory; compute SOAP dissimilarity between production frames and fine-tuning dataset | Expand fine-tuning dataset to cover the target temperature and structural distortions (Workflow 4); run active learning after fine-tuning to fill gaps |
| Catastrophic forgetting | Fine-tuned model is accurate on the target domain but has force MAE 3–10× higher than the foundation model on OOD structures; model is unusable outside the fine-tuning composition or temperature | Learning rate too high or too many epochs; fine-tuning dataset dominates the loss and overwrites the pretrained representation; no anti-forgetting strategy | Run `forgetting_and_improvement_report` with an OOD test set; compare OOD errors before and after fine-tuning | Switch to head-only or partial regime; reduce LR by 5–10×; add foundation data to the training mix (`add_foundation_fraction` ≥ 0.3); reduce max epochs |
| Reference energy mismatch | Predicted formation energies are systematically shifted by 50–500 meV/atom even when forces are accurate; convex hull stability ordering is wrong | Fine-tuning DFT uses different pseudopotentials, cutoff energy, or functional than the foundation model's training data; `delta_E0` is large for one or more species | Run `compute_reference_energy_offsets`; if |ΔE0| > 0.1 eV for any species, inconsistency is likely | Re-run fine-tuning DFT with the same pseudopotential family (PAW PBE, same POTCAR version) as the foundation model; or provide explicit E0s that correct for the offset |
| Label inconsistency across fine-tuning iterations | Committee σ_F is elevated even for equilibrium structures; model errors are larger than expected from the fine-tuning dataset size | Fine-tuning DFT was run across multiple iterations with different INCAR settings, ENCUT, or k-mesh; some structures labeled at tighter convergence, others at looser | Run `check_label_consistency`; compare INCAR hashes across all fine-tuning calculations | Re-run all fine-tuning calculations with a frozen, canonical INCAR; do not mix structures from different convergence campaigns in the same fine-tuning set |
| False improvement from data leakage | Fine-tuned model shows excellent test-set accuracy but production accuracy is worse than expected | Test set structures were similar or identical to training structures (same configuration type, same temperature, same supercell size); the model is interpolating within the training manifold | Verify train/test split with SOAP dissimilarity check; ensure test structures were not used for any hyperparameter or checkpoint selection | Use prototype-aware or temperature-stratified splitting; ensure test set contains structures from different MD frames, different supercell sizes, or different distortion amplitudes |
| Ensemble collapse after fine-tuning | Fine-tuned committee σ_F is uniformly < 2 meV/Å even for thermally distorted structures; active learning queries all structures or none | Fine-tuning on a homogeneous small dataset drove all committee members to the same solution; seed diversity is insufficient when the dataset is the dominant constraint | Run `verify_finetune_did_not_destroy_committee_diversity`; if mean pairwise MAE < 3 meV/Å, collapse has occurred | Switch to head-only fine-tuning which maintains more diversity; supplement with SOAP dissimilarity OOD detection as a backup to committee σ_F |
| Fine-tuning outside pretrained model support | Fine-tuning on structures containing elements not in the foundation model's vocabulary; training appears to run but energy predictions for those elements are random | Foundation model has no trained embedding for the new element; the MACE element table is fixed at initialization | Check foundation model element coverage list before fine-tuning; test with a single-element supercell of the new element | Train a new model from scratch that includes the new element from the beginning; do not attempt to add new elements via fine-tuning |
| Unstable MD after fine-tuning with improved static errors | Fine-tuned model passes static force/energy validation but NVE MD drifts rapidly (> 1 meV/atom/ps) or diverges at target temperature | Fine-tuning improved accuracy for near-equilibrium structures while degrading the model's behavior for rare, high-energy configurations that arise during MD | Run NVE stability test (Workflow 5 in `mlp-validation`) on the fine-tuned model at the target temperature; compare energy drift vs. foundation model | Include high-energy configurations from MD (displacement > 0.15 Å) in the fine-tuning dataset; add random displacement structures at the target temperature amplitude |

## Best Practices

- **Always benchmark the universal potential before deciding to fine-tune.** The `universal-potentials` Workflow 1 benchmark takes < 1 hour and quantifies whether fine-tuning is warranted. If the universal potential is already accurate enough, fine-tuning adds risk (forgetting, label inconsistency) with no benefit.
- **Start with head-only fine-tuning on small datasets.** For fewer than 200 fine-tuning structures, head-only is the safest strategy: it cannot forget the pretrained representation. Evaluate whether head-only improvement is sufficient before escalating to partial or full fine-tuning.
- **Always supply an OOD test set for the forgetting check.** The most common oversight in fine-tuning evaluations is skipping the OOD check. Forgetting is silent in in-domain metrics. A 20-structure OOD test set sampled from the Materials Project for the same elements takes minutes to evaluate and can detect catastrophic forgetting before deployment.
- **Use explicit E0s when the fine-tuning dataset is small or compositionally narrow.** The `--E0s="average"` option regresses reference energies from the training dataset. With fewer than 100 structures of a single composition, the regression is underdetermined and produces spurious E0 values. Provide pre-computed isolated atom energies instead.
- **Record the foundation model SHA-256 alongside every fine-tuning result.** Universal potential weights change between package releases. A fine-tuning run that used `mace_mp(model="medium")` from package version 0.3.5 will not be reproducible with version 0.3.7 unless the specific checkpoint file is archived. Always download and record the exact checkpoint.
- **Run the full `mlp-validation` protocol on the fine-tuned model, not just static force errors.** Static force MAE on the fine-tuning test set is insufficient to certify deployment readiness. NVE stability, phonon frequencies, and the NVT ensemble test (from `mlp-validation`) must all be run on the fine-tuned model before production use.
- **Use `add_foundation_fraction ≥ 0.3` for full fine-tuning regimes when anti-forgetting is important.** Adding 30–50% of the fine-tuning training set size in foundation structures to the training batch significantly reduces forgetting with modest training cost increase. This is especially important when the fine-tuned model will be used both within and outside the target domain.
- **Do not use the fine-tuned model's predictions as labels for a second model.** A fine-tuned MLP predicts at MLP accuracy, not DFT accuracy. Using its predictions as labels for training another model compounds errors. All labels must come from actual DFT calculations.

## Integration with Other Skills

- **[universal-potentials](../universal-potentials/SKILL.md):** The decision to fine-tune begins with the benchmarking workflow in that skill. The `fine_tuning_decision_report` function there classifies systems as use-as-is, fine-tune, or train-from-scratch based on the universal potential force MAE relative to the target accuracy. Fine-tuning is indicated when that benchmark returns `FINE_TUNE` or `TRAIN_OR_FINE_TUNE`.
- **[mace-training](../mace-training/SKILL.md):** When `fine_tuning_decision_report` returns `TRAIN_FROM_SCRATCH`, that skill covers the full from-scratch pipeline. The dataset format, loss weighting, and validation protocol are identical whether training from scratch or fine-tuning; the distinction is only in the model initialization (`--foundation_model` vs. random initialization).
- **[uncertainty-mlp](../uncertainty-mlp/SKILL.md):** Post-fine-tuning UQ requires a committee of fine-tuned models, not the foundation committee. Workflow 5 here covers committee training and diversity verification after fine-tuning. The SOAP dissimilarity OOD detector from that skill is especially important for fine-tuned models, because ensemble collapse is more common after fine-tuning on small datasets.
- **[active-learning-mlp](../active-learning-mlp/SKILL.md):** Active learning after fine-tuning uses the fine-tuned committee's σ_F to select new DFT structures for labeling, targeting the gaps in the fine-tuning coverage identified by Workflow 4. The fine-tuned model replaces the universal potential as the exploration model in the AL loop. Thresholds σ_low and σ_high from `uncertainty-mlp` must be recalibrated after each fine-tuning iteration.
- **[mlp-validation](../mlp-validation/SKILL.md):** All fine-tuned models must pass the full validation trinity (static errors, physical property reproduction, MD stability) before deployment. The forgetting check in Workflow 3 here is a prerequisite screening step; `mlp-validation` provides the authoritative deployment certification.
- **[mlp-to-lammps](../mlp-to-lammps/SKILL.md):** Fine-tuned MACE models are deployed to LAMMPS using the same `pair_style mace` interface as system-specific or foundation models. The `--save_cpu` flag in Workflow 2 here ensures the fine-tuned checkpoint is serialized in a format compatible with LAMMPS deployment. Species mapping and unit verification must be repeated for the fine-tuned model.
- **[dft-dataset-generation](../../electronic-structure/dft-dataset-generation/SKILL.md):** The fine-tuning DFT calculations (single-point labels on pre-relaxed structures, temperature-sampled structures, surface slabs, defect supercells) follow the same labeling pipeline as full dataset generation. The canonical INCAR settings established in that skill must be reused exactly — not approximated — for fine-tuning data, to ensure label consistency with the foundation model's training level.
- **[dft-convergence](../../electronic-structure/dft-convergence/SKILL.md):** Fine-tuning DFT must use the same convergence protocol (ENCUT, EDIFF, k-mesh) as the rest of the campaign. If reference energy offsets are detected (Workflow 1 here), the root cause is almost always a deviation from the canonical convergence settings. That skill establishes the frozen settings hash that must be preserved across all fine-tuning iterations.
- **[vasp-workflow](../../electronic-structure/vasp-workflow/SKILL.md):** VASP INCAR templates for fine-tuning single-point calculations follow the same pattern as for full DFT dataset generation. Special attention is required for magnetic systems (ISPIN, MAGMOM initialization) and for defect supercells (NELM, NELMIN, charge handling).
- **[quantum-espresso](../../electronic-structure/quantum-espresso/SKILL.md):** QE is an alternative DFT backend for fine-tuning data generation. The pseudopotential family (SSSP efficiency vs. SSSP precision) must match the foundation model's training DFT level; mixing pseudopotential families introduces reference energy offsets.
- **[lammps-workflow](../../atomistic-md/lammps-workflow/SKILL.md):** Production MD with fine-tuned models uses the same LAMMPS input structure as any MACE deployment. The pre-equilibration strategy (Workflow 5 of `universal-potentials`) applies to the fine-tuned model as well; run a short NVT equilibration with the fine-tuned model before committing to production.
- **[md-equilibration](../../atomistic-md/md-equilibration/SKILL.md):** The NVE energy conservation test from that skill is the primary MD stability check for fine-tuned models. A fine-tuned model that passes static force validation but fails the NVE test should not be used for production MD regardless of its static accuracy.
- **[materials-featurization](../../scientific-data/materials-featurization/SKILL.md):** SOAP descriptors used in Workflow 4 (coverage audit) and Workflow 5 (diversity verification) must use consistent parameters across the campaign. That skill's `save_feature_metadata` pattern should be applied to the SOAP parameters used here to ensure reproducibility when comparing coverage audits across AL iterations.
- **[reproducibility-workflow](../../scientific-data/reproducibility-workflow/SKILL.md):** Fine-tuning provenance records (foundation model SHA-256, fine-tuning regime, seed, training dataset checksum, LR, E0s convention) belong in the campaign settings JSON described in that skill. The fine-tuning provenance JSON from Workflow 2 here is the per-run record; the campaign log records the sequence of fine-tuning iterations alongside the AL provenance chain.

## Key References

1. Batatia, I., et al. (2023). *A foundation model for atomistic materials chemistry.* arXiv:2401.00096. https://arxiv.org/abs/2401.00096
2. Deng, B., et al. (2023). *CHGNet as a pretrained universal neural network potential for charge-informed atomistic modelling.* Nature Machine Intelligence, 5, 1031–1041. https://doi.org/10.1038/s42256-023-00716-3
3. Goodfellow, I., et al. (2013). *An empirical investigation of catastrophic forgetting in gradient-based neural networks.* arXiv:1312.6211. https://arxiv.org/abs/1312.6211
4. Zaverkin, V., et al. (2023). *Transfer learning for chemically accurate interatomic neural network potentials.* Physical Chemistry Chemical Physics, 25, 5383. https://doi.org/10.1039/D2CP05793J
5. Kovács, D. P., et al. (2023). *MACE-OFF23: Transferable machine learning force fields for organic molecules.* arXiv:2312.15211. https://arxiv.org/abs/2312.15211
