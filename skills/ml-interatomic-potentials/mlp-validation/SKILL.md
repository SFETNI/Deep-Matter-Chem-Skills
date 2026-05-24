---
name: mlp-validation
domain: ml-interatomic-potentials
maturity: draft
tools:
  - MACE
  - ASE
  - LAMMPS
  - phonopy
  - pymatgen
  - matplotlib
tested_versions:
  MACE: "0.3.6"
  ASE: "3.23.0"
  phonopy: "2.21.0"
  pymatgen: "2024.1.26"
  LAMMPS: "2Aug2023"
requires_gpu: false
estimated_runtime: hours-to-days
---

# MLP Validation

## Description

This skill covers the systematic validation of machine-learned interatomic potentials (MLPs) before production deployment: static test-set error analysis, physical property reproduction (equation of state, phonons, elastic constants, surface and defect energies), MD stability testing, extrapolation detection, and deployment consistency verification between training code, ASE, and LAMMPS. Invoke this skill after completing MLP training (see `mace-training`) and before using the MLP for production MD or property prediction. A low test-set error is necessary but not sufficient for a production-quality MLP — this skill documents the additional checks required to catch the failure modes that pass the training metrics.

## Domain Context

An MLP is a function approximator fit to a finite set of DFT-labeled structures. The training loss and test-set error measure how well the model interpolates within the sampled configuration space. They do not measure:

- Whether the sampled space is representative of the target application (MD at a given temperature and pressure, surface chemistry, defect migration)
- Whether the model produces physically correct macroscopic properties (thermal expansion, phonon frequencies, elastic moduli) that emerge from the collective behavior of the potential energy surface
- Whether the model is numerically stable over nanosecond timescales, where rare high-energy configurations may be sampled with non-negligible probability
- Whether the LAMMPS and ASE implementations of the same model file produce identical forces (they should, but unit conversions and element-order conventions have introduced silent bugs in practice)

**The validation trinity** structures this skill:

1. **Static errors:** Energy, force, and stress MAE on held-out DFT structures, decomposed by chemical environment and configuration family. This is the fastest check and narrows down systematic biases.
2. **Physical property reproduction:** Comparison of MLP-predicted properties against DFT references computed at the same level of theory. EOS, phonon frequencies, elastic constants, and defect energies are the most diagnostic. These catch errors in the many-body and stress response that per-atom force errors miss.
3. **MD stability:** Short NVE energy conservation runs, NVT temperature stability, and NPT density and structural reproduction. MD integrates the potential repeatedly and amplifies any nonphysical region of the PES that was not covered by training.

**Reference standard: MLP predictions must be compared against DFT, not against another MLP.** Benchmarking one MLP against another (e.g., comparing MACE to a Behler–Parrinello network on the same test set) measures model agreement, not physical accuracy. The DFT level used for training data is the correct reference for all validation checks.

**Held-out test set integrity:** The test set used for final validation must be completely isolated from the training process, the active learning query strategy, and the hyperparameter search. Structures that informed any training decision — including model architecture selection — are contaminated for accuracy estimation purposes.

## When to Use This Skill

- After training any MLP (MACE, NequIP, Allegro, DeepMD, GAP) and before production deployment.
- When an active learning campaign signals convergence and a final model is being commissioned.
- When an existing MLP is being extended to new conditions (new temperature range, new phase, surface chemistry) and the domain extension must be validated.
- When an MLP from the literature is being reused on a new system and its transferability must be verified.
- When deploying to a new code (switching from ASE to LAMMPS, or upgrading MACE version) and deployment consistency must be confirmed.
- When quantifying the reliability of MLP-predicted properties for a paper or report.

## When Not to Use This Skill

- **As a substitute for active learning.** If the model fails validation, the correct response is to expand the training dataset (see `active-learning-mlp`), not to adjust validation thresholds.
- **When the model has not yet converged.** Training-run validation metrics should be monitored during training via `mace-training` Workflow 4. This skill is for post-training, post-convergence validation.
- **For properties outside the Born-Oppenheimer surface** — electronic excitations, photochemistry, non-adiabatic dynamics. The MLP models the ground-state PES; no validation on this surface certifies accuracy on excited states.
- **As the sole assessment for high-stakes applications.** For safety-critical or regulatory applications (e.g., nuclear materials, pharmaceutical molecular dynamics) MLP validation is a necessary but not sufficient quality assurance step; additional independent expert review and experimental benchmarks are required.

## Core Concepts

- **Validation trinity:** Static errors + physical property reproduction + MD stability. All three are required; passing one does not compensate for failing another.
- **Per-config-type decomposition:** Global test-set MAE conceals systematic failures in minority or critical configuration families. A model with 30 meV/Å global force MAE may have 200 meV/Å for surface configurations if surfaces are 5% of the test set. Always report errors broken down by `config_type`.
- **Per-species error decomposition:** For multi-element systems, force errors per chemical species reveal gaps in chemical coverage. A high force error for a low-abundance species (e.g., a dopant) indicates insufficient training coverage of that species' bonding environments.
- **Learning curve:** A plot of test-set error (force MAE) as a function of training set size. A flat learning curve at the current training size indicates that data quantity is no longer the bottleneck; a still-decreasing curve indicates more data would improve accuracy. Computed by training sub-models on nested subsets of the training data.
- **Equation of state (EOS):** Energy vs. volume curve for a bulk structure, fit to the Birch-Murnaghan or Vinet equation. MLP should reproduce DFT equilibrium volume V₀, bulk modulus B₀, and pressure derivative B₀' within specified tolerances. EOS errors are the most sensitive indicator of stress tensor quality.
- **Phonon spectrum:** Normal mode frequencies from the harmonic approximation, computed via finite displacements (phonopy) or DFPT. MLP phonon errors are typically dominated by the accuracy of forces on symmetry-displaced supercell structures. Imaginary modes in stable crystals indicate a locally unstable PES.
- **Elastic constants (Cᵢⱼ):** Second derivatives of the strain energy with respect to applied strain. Computed by applying uniform strains and fitting the energy-strain curve. Sensitive to the stress response of the MLP — elastic constants are the most stringent test of stress label accuracy.
- **Radial distribution function (RDF):** Pair correlation function g(r) from MD trajectory. Comparison of MLP-MD RDF to DFT-MD or experimental data validates structural properties at finite temperature. Peak positions should agree within 0.05 Å; peak heights within ~10%. [EXPERT REVIEW NEEDED]
- **NVE energy drift:** Rate of total energy drift in microcanonical MD. Drift arises from numerical integration error (timestep too large), unphysical configurations in the trajectory (model extrapolation), or implementation bugs. Acceptable drift is system-dependent but typically < 1 meV/atom/ps. [EXPERT REVIEW NEEDED]
- **Deployment consistency:** The same MACE model file should produce bitwise-identical forces when called from MACECalculator (ASE), from LAMMPS pair_style mace, and from MACE's own evaluation script. Discrepancies above 1 meV/Å indicate element-order mismatches, unit errors, or precision differences.
- **Extrapolation detection:** Uncertainty estimation using committee models (force std across N trained models) or structural dissimilarity metrics (SOAP distances to the training set). A structure is flagged for potential extrapolation if its committee σ_F exceeds a calibrated threshold or if its minimum SOAP distance to the training set exceeds a maximum-coverage radius.

## Key Workflows

### Workflow 1: Static error analysis — energy, force, and stress decomposition

The baseline check. Run before any physical property validation. Identify whether errors are globally acceptable and which configuration families or species are problematic.

```python
import numpy as np
from ase.io import read
from mace.calculators import MACECalculator
from collections import defaultdict
from pathlib import Path


def static_error_analysis(
    model_path: str,
    test_file: str,
    device: str = "cpu",
    outlier_force_threshold_meV: float = 500.0,
) -> dict:
    """
    Compute energy, force, and stress MAE on the test set.
    Decompose by config_type and chemical species.
    Flag outliers for inspection.

    Always use float64 for validation: float32 precision can mask
    small systematic errors in energy (< 1 meV/atom) and stress.
    """
    calc = MACECalculator(
        model_paths=model_path,
        device=device,
        default_dtype="float64",  # float64 for unbiased accuracy assessment
    )
    frames = read(test_file, index=":", format="extxyz")

    # Per-config-type accumulators
    by_type: dict[str, dict] = defaultdict(lambda: {
        "e_errors": [], "f_errors_all": [], "s_errors": [], "natoms": []
    })
    # Per-species force errors
    species_force_errors: dict[str, list] = defaultdict(list)
    outliers = []

    for i, atoms in enumerate(frames):
        ctype = atoms.info.get("config_type", "Default")
        natoms = len(atoms)

        ref_e = atoms.info["energy"] / natoms           # eV/atom
        ref_f = atoms.arrays["forces"]                  # (N, 3) eV/Å
        ref_s = atoms.info.get("stress")                # Voigt eV/Å³ or None

        a = atoms.copy()
        a.calc = calc
        pred_e = a.get_potential_energy() / natoms      # eV/atom
        pred_f = a.get_forces()                         # (N, 3) eV/Å

        e_err_meV = abs(pred_e - ref_e) * 1000          # meV/atom
        f_mae_meV = np.abs(pred_f - ref_f).mean() * 1000  # meV/Å (mean over atoms × components)
        f_rmse_meV = np.sqrt(np.mean((pred_f - ref_f)**2)) * 1000

        by_type[ctype]["e_errors"].append(e_err_meV)
        by_type[ctype]["f_errors_all"].append(f_mae_meV)
        by_type[ctype]["natoms"].append(natoms)

        if ref_s is not None:
            pred_s = a.get_stress()   # ASE Voigt eV/Å³
            s_mae_GPa = np.abs(pred_s - ref_s).mean() * 160.2176  # convert eV/Å³ → GPa
            by_type[ctype]["s_errors"].append(s_mae_GPa)

        # Per-species force errors
        syms = atoms.get_chemical_symbols()
        atom_f_errors = np.abs(pred_f - ref_f).mean(axis=1) * 1000  # (N,) meV/Å per atom
        for sym, ferr in zip(syms, atom_f_errors):
            species_force_errors[sym].append(float(ferr))

        # Flag outliers
        if f_mae_meV > outlier_force_threshold_meV:
            outliers.append({
                "index": i, "config_type": ctype,
                "f_mae_meV_A": f_mae_meV, "e_err_meV_atom": e_err_meV,
                "natoms": natoms,
            })

    # ─── Per-config-type report ──────────────────────────────────────────────
    print(f"\n{'Config type':<22}  {'N':>5}  {'E MAE (meV/at)':>16}  "
          f"{'F MAE (meV/Å)':>15}  {'S MAE (GPa)':>13}")
    print("─" * 78)
    all_e, all_f = [], []
    for ctype, d in sorted(by_type.items()):
        ne = np.mean(d["e_errors"]); nf = np.mean(d["f_errors_all"])
        ns = np.mean(d["s_errors"]) if d["s_errors"] else float("nan")
        all_e.extend(d["e_errors"]); all_f.extend(d["f_errors_all"])
        print(f"{ctype:<22}  {len(d['e_errors']):>5}  {ne:>16.2f}  "
              f"{nf:>15.2f}  {ns:>13.3f}")
    print("─" * 78)
    print(f"{'TOTAL':<22}  {sum(len(d['e_errors']) for d in by_type.values()):>5}  "
          f"{np.mean(all_e):>16.2f}  {np.mean(all_f):>15.2f}")

    # ─── Per-species force MAE ───────────────────────────────────────────────
    print(f"\nPer-species force MAE:")
    for sym, errs in sorted(species_force_errors.items()):
        print(f"  {sym:>4}: {np.mean(errs):>8.2f} ± {np.std(errs):>6.2f} meV/Å  "
              f"(N_atom_envs={len(errs)})")

    # ─── Outlier report ──────────────────────────────────────────────────────
    if outliers:
        print(f"\n{len(outliers)} outlier frames (F MAE > {outlier_force_threshold_meV:.0f} meV/Å):")
        for o in outliers[:10]:
            print(f"  Frame {o['index']:>4d}  {o['config_type']:<20}  "
                  f"F={o['f_mae_meV_A']:>7.1f} meV/Å  "
                  f"E={o['e_err_meV_atom']:>7.2f} meV/at")
        print("  → Inspect these structures: are they physically reasonable?")
        print("  → Common causes: unconverged DFT, high-energy rattled configs,")
        print("    structures from MD at T >> training temperature")
    else:
        print("\nNo outliers above threshold.")

    return {"by_type": dict(by_type), "species": dict(species_force_errors),
            "outliers": outliers}


# ─── Reference targets for common material classes ────────────────────────────
# [EXPERT REVIEW NEEDED: these targets are indicative; the appropriate threshold
#  depends strongly on the system and target application precision]
FORCE_MAE_TARGETS = {
    "metals_bulk_MD":        50,   # meV/Å: typical target for stable MD
    "covalent_bulk_MD":      80,   # meV/Å
    "molecular_dynamics":    30,   # meV/Å: geometry-sensitive
    "surfaces_adsorption":   60,   # meV/Å
    "reactive_bonds":       120,   # meV/Å: higher tolerance; reactions are rare events
}
ENERGY_MAE_TARGETS = {
    "bulk":      5,    # meV/atom: tight; required for EOS, phonons
    "surface":  10,    # meV/atom: surface formation energies sensitive
    "molecular": 1,    # meV/atom: conformer energies
}
```

### Workflow 2: Learning curves and data sufficiency

Determines whether the training set is large enough or whether more DFT data would meaningfully reduce errors. Trains a sequence of sub-models on nested subsets of increasing size.

```python
import subprocess
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ase.io import read, write
import random
import tempfile
from pathlib import Path


def compute_learning_curve(
    train_file: str,
    val_file: str,
    test_file: str,
    model_name: str,
    subset_sizes: list[int],
    base_mace_kwargs: dict,
    device: str = "cuda",
    seed: int = 42,
) -> dict:
    """
    Train MACE sub-models on nested subsets of the training set.
    Plot force MAE vs. training set size to assess data saturation.

    subset_sizes: list of training set sizes to probe,
        e.g. [100, 200, 500, 1000, 2000, 5000]
    base_mace_kwargs: dict of flags for mace_run_train (without name/train_file)
    """
    random.seed(seed)
    all_train = read(train_file, index=":", format="extxyz")
    test_frames = read(test_file, index=":", format="extxyz")

    results = {}

    for n in subset_sizes:
        if n > len(all_train):
            print(f"Subset size {n} > training set size {len(all_train)}, skipping")
            continue

        # Sample a stratified subset by config_type
        from collections import defaultdict
        by_type = defaultdict(list)
        for f in all_train:
            by_type[f.info.get("config_type", "Default")].append(f)
        subset = []
        for frames in by_type.values():
            random.shuffle(frames)
            frac = len(frames) / len(all_train)
            take = max(1, int(n * frac))
            subset.extend(frames[:take])
        subset = subset[:n]  # Trim to exact size

        with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as tf:
            subset_file = tf.name
        write(subset_file, subset, format="extxyz")

        name = f"{model_name}_lc_n{n}"
        cmd = (
            ["mace_run_train", f"--name={name}",
             f"--train_file={subset_file}", f"--valid_file={val_file}",
             f"--device={device}", f"--seed={seed}"]
            + [f"--{k}={v}" for k, v in base_mace_kwargs.items()]
        )
        print(f"Training subset n={n} ({len(subset)} structures)...")
        subprocess.run(cmd, check=True)

        # Evaluate on test set
        from mace.calculators import MACECalculator
        model_file = f"{name}_run-{seed}_stagetwo.model"
        calc = MACECalculator(model_paths=model_file, device="cpu",
                              default_dtype="float64")
        f_maes = []
        for atoms in test_frames:
            ref_f = atoms.arrays["forces"]
            a = atoms.copy(); a.calc = calc
            f_maes.append(np.abs(a.get_forces() - ref_f).mean() * 1000)

        mae = float(np.mean(f_maes))
        results[n] = mae
        print(f"  n={n:>6}: test force MAE = {mae:.2f} meV/Å")

    # Plot
    sizes = sorted(results.keys())
    maes = [results[s] for s in sizes]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.loglog(sizes, maes, "o-", color="steelblue")
    ax.set_xlabel("Training set size (structures)")
    ax.set_ylabel("Test force MAE (meV/Å)")
    ax.set_title(f"Learning curve — {model_name}")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{model_name}_learning_curve.png", dpi=150)
    print(f"Learning curve saved: {model_name}_learning_curve.png")

    # Assess saturation: if last two points improve by < 5%, saturated
    if len(sizes) >= 2:
        improvement = (results[sizes[-2]] - results[sizes[-1]]) / results[sizes[-2]]
        if improvement < 0.05:
            print(f"NOTE: Learning curve is saturating (last improvement = "
                  f"{improvement*100:.1f}%) — more data may not help significantly. "
                  f"Consider: larger model, better data diversity, active learning.")
        else:
            print(f"NOTE: Curve still declining (last improvement = "
                  f"{improvement*100:.1f}%) — adding more data likely to reduce errors.")

    return results
```

### Workflow 3: Equation of state and elastic constant validation

The EOS validates the bulk modulus, equilibrium volume, and pressure response — the most sensitive tests of stress tensor accuracy. Elastic constants are the most stringent check: they require accurate second derivatives of the energy with respect to strain.

```python
import numpy as np
from ase.io import read
from ase.eos import EquationOfState
from mace.calculators import MACECalculator
from pymatgen.core import Structure
from pymatgen.analysis.elasticity import ElasticTensor, DeformedStructureSet
from pymatgen.io.ase import AseAtomsAdaptor


def validate_eos(
    equilibrium_atoms,
    calc,
    dft_reference: dict,
    volume_range: float = 0.10,
    n_points: int = 11,
) -> dict:
    """
    Compute MLP equation of state and compare to DFT reference.

    dft_reference: dict with keys "V0_A3" (Å³/atom), "B0_GPa", "E0_eV_atom"
    Returns dict with MLP EOS parameters and errors.
    """
    volumes, energies = [], []
    V0_cell = equilibrium_atoms.get_volume()
    natoms = len(equilibrium_atoms)

    for strain in np.linspace(1 - volume_range, 1 + volume_range, n_points):
        a = equilibrium_atoms.copy()
        a.set_cell(a.cell * strain**(1/3), scale_atoms=True)
        a.calc = calc
        volumes.append(a.get_volume() / natoms)   # Å³/atom
        energies.append(a.get_potential_energy() / natoms)  # eV/atom

    eos = EquationOfState(volumes, energies, eos="birch-murnaghan")
    V0, E0, B = eos.fit()          # B in eV/Å³
    B_GPa = B * 160.2176           # Convert eV/Å³ → GPa

    # Compare to DFT
    dV = (V0 - dft_reference["V0_A3"]) / dft_reference["V0_A3"] * 100
    dB = (B_GPa - dft_reference["B0_GPa"]) / dft_reference["B0_GPa"] * 100

    print(f"\n{'Property':<25}  {'DFT':>10}  {'MLP':>10}  {'Error':>10}")
    print("─" * 58)
    print(f"{'V0 (Å³/atom)':<25}  {dft_reference['V0_A3']:>10.4f}  "
          f"{V0:>10.4f}  {dV:>+9.2f}%")
    print(f"{'B0 (GPa)':<25}  {dft_reference['B0_GPa']:>10.2f}  "
          f"{B_GPa:>10.2f}  {dB:>+9.2f}%")
    if "E0_eV_atom" in dft_reference:
        dE = (E0 - dft_reference["E0_eV_atom"]) * 1000
        print(f"{'E0 (meV/atom offset)':<25}  {0:>10.1f}  {dE:>10.1f}  {dE:>+9.1f} meV")

    # Tolerances (system-dependent) [EXPERT REVIEW NEEDED]
    if abs(dV) > 1.0:
        print(f"WARNING: V0 error {dV:.2f}% exceeds 1% tolerance — check stress labels")
    if abs(dB) > 5.0:
        print(f"WARNING: B0 error {dB:.2f}% exceeds 5% tolerance — EOS curvature incorrect")

    eos.plot(filename="eos_mlp_vs_dft.png")
    return {"V0": V0, "B0_GPa": B_GPa, "E0": E0, "dV_pct": dV, "dB_pct": dB}


def validate_elastic_constants(
    equilibrium_atoms,
    calc,
    dft_Cij_GPa: dict,
    strain_magnitudes: list[float] = (0.005, 0.010, 0.015, 0.020),
) -> dict:
    """
    Compute elastic constants by energy-strain fitting and compare to DFT.
    Uses pymatgen's DeformedStructureSet for symmetry-adapted strains.

    dft_Cij_GPa: dict of {(i,j): value_GPa} from DFT calculation.
    Strain magnitudes must be small enough to stay in the harmonic regime
    but large enough to exceed numerical noise. [EXPERT REVIEW NEEDED]
    """
    adaptor = AseAtomsAdaptor()
    pymat_structure = adaptor.get_structure(equilibrium_atoms)

    # Generate symmetry-adapted deformations
    deform_set = DeformedStructureSet(
        pymat_structure,
        norm_strains=strain_magnitudes,
        shear_strains=strain_magnitudes,
    )

    stresses_Voigt = []  # (n_deformations, 6) in GPa
    deformations = []

    for deform, deformed_struct in zip(deform_set.deformations,
                                       deform_set.deformed_structures):
        atoms_def = adaptor.get_atoms(deformed_struct)
        atoms_def.calc = calc
        # ASE stress: Voigt eV/Å³, tensile positive
        stress_eV_A3 = atoms_def.get_stress(voigt=True)
        stress_GPa = stress_eV_A3 * 160.2176  # eV/Å³ → GPa
        stresses_Voigt.append(stress_GPa)
        deformations.append(deform)

    # Fit elastic tensor from stress-strain
    elastic_tensor = ElasticTensor.from_independent_strains(
        deformations, stresses_Voigt, eq_stress=None
    )
    Cij_mlp = elastic_tensor.voigt  # (6,6) in GPa

    # Report key components
    print(f"\nElastic constants (GPa):")
    print(f"{'Component':<10}  {'DFT':>10}  {'MLP':>10}  {'Error (GPa)':>12}")
    print("─" * 46)
    for (i, j), val_dft in sorted(dft_Cij_GPa.items()):
        val_mlp = float(Cij_mlp[i, j])
        err = val_mlp - val_dft
        flag = "  ← WARNING" if abs(err) > 10 else ""
        print(f"C{i+1}{j+1}        {val_dft:>10.1f}  {val_mlp:>10.1f}  {err:>+12.1f}{flag}")

    return {"Cij_mlp": Cij_mlp, "elastic_tensor": elastic_tensor}
```

### Workflow 4: Phonon validation using finite displacements

Phonon frequencies are among the most sensitive tests of the local curvature of the MLP PES. Imaginary modes for a known stable crystal are a definitive sign of a locally incorrect potential.

```python
import numpy as np
from ase.io import read
from mace.calculators import MACECalculator
import phonopy
from phonopy.structure.atoms import PhonopyAtoms


def validate_phonons(
    equilibrium_atoms,
    calc,
    supercell_matrix: list,
    displacement_distance: float = 0.01,
    dft_phonon_yaml: str = None,
    output_prefix: str = "phonon_mlp",
) -> dict:
    """
    Compute MLP phonon band structure via finite displacements using phonopy.
    Optionally compare to DFT-computed phonopy YAML.

    supercell_matrix: e.g. [[3,0,0],[0,3,0],[0,0,3]] for 3×3×3 supercell.
    displacement_distance: 0.01 Å is standard for metals; use 0.03 Å for
        soft systems where the harmonic regime is harder to resolve.
        [EXPERT REVIEW NEEDED]
    """
    # Build phonopy object from ASE Atoms
    cell = PhonopyAtoms(
        symbols=equilibrium_atoms.get_chemical_symbols(),
        scaled_positions=equilibrium_atoms.get_scaled_positions(),
        cell=equilibrium_atoms.get_cell(),
    )
    ph = phonopy.Phonopy(cell, supercell_matrix)
    ph.generate_displacements(distance=displacement_distance)

    displaced_supercells = ph.supercells_with_displacements
    print(f"Computing forces on {len(displaced_supercells)} displaced supercells...")

    forces_set = []
    for sc in displaced_supercells:
        if sc is None:
            forces_set.append(None)
            continue
        from ase import Atoms as AseAtoms
        atoms_sc = AseAtoms(
            symbols=sc.symbols,
            positions=sc.positions,
            cell=sc.cell,
            pbc=True,
        )
        atoms_sc.calc = calc
        forces_set.append(atoms_sc.get_forces())

    ph.set_forces(forces_set)
    ph.produce_force_constants()

    # Compute phonon band structure along high-symmetry path
    ph.auto_band_structure(
        npoints=51,
        write_yaml=True,
        filename=f"{output_prefix}_band.yaml",
    )
    ph.auto_total_dos(
        sigma=0.1,
        write_dat=True,
        filename=f"{output_prefix}_dos.dat",
    )

    # Compute thermal properties
    ph.run_mesh([20, 20, 20])
    ph.run_thermal_properties(t_min=0, t_max=1500, t_step=50)
    ph.write_yaml_thermal_properties(filename=f"{output_prefix}_thermal.yaml")

    # Check for imaginary modes at Gamma — definitive instability indicator
    ph.run_qpoints([[0, 0, 0]])
    gamma_freqs = ph.get_qpoints_and_frequencies()[1][0]  # THz
    n_imaginary = (gamma_freqs < -0.1).sum()  # threshold: -0.1 THz [EXPERT REVIEW NEEDED]
    if n_imaginary > 0:
        print(f"\nWARNING: {n_imaginary} imaginary mode(s) at Gamma:")
        for f in gamma_freqs[gamma_freqs < -0.1]:
            print(f"  {f:.3f} THz")
        print("  This indicates a locally unstable PES. Possible causes:")
        print("  - MLP is extrapolating at the equilibrium geometry")
        print("  - Equilibrium structure is wrong (wrong symmetry or relaxation)")
        print("  - DFT training data was computed at a different geometry")
    else:
        print(f"Gamma-point frequencies: all real. Lowest = {gamma_freqs.min():.3f} THz")

    # Compare to DFT phonons if reference is provided
    if dft_phonon_yaml:
        ph_dft = phonopy.load(dft_phonon_yaml)
        # Compute RMS frequency error across high-symmetry path
        # This requires consistent q-point sampling — see phonopy documentation
        print(f"DFT phonon reference loaded: {dft_phonon_yaml}")
        print("Overlay band structures and compute per-mode frequency errors.")
        print("Target: < 5% RMS frequency error for acoustic branches. [EXPERT REVIEW NEEDED]")

    return {"phonopy_object": ph, "n_imaginary_at_gamma": int(n_imaginary),
            "gamma_frequencies_THz": gamma_freqs.tolist()}


def validate_defect_and_surface_energies(
    calc,
    dft_references: dict,
) -> None:
    """
    Compare MLP defect formation and surface energies to DFT references.
    These probe the MLP's accuracy outside bulk equilibrium environments.

    dft_references: dict mapping label to {
        "atoms": ASE Atoms (relaxed with DFT),
        "dft_energy": float (eV),
        "n_atoms": int,
        "bulk_e_per_atom": float (eV/atom, DFT),
        "description": str
    }
    """
    print(f"\n{'Property':<30}  {'DFT (eV)':>10}  {'MLP (eV)':>10}  {'Error (eV)':>12}")
    print("─" * 65)

    for label, ref in dft_references.items():
        atoms = ref["atoms"].copy()
        atoms.calc = calc
        mlp_e = atoms.get_potential_energy()
        dft_e = ref["dft_energy"]
        err = mlp_e - dft_e
        flag = "  ← WARNING" if abs(err) > 0.1 else ""
        print(f"{label:<30}  {dft_e:>10.4f}  {mlp_e:>10.4f}  {err:>+12.4f}{flag}")
```

### Workflow 5: MD stability — NVE conservation, NVT, and RDF comparison

MD stability tests reveal whether the MLP is usable for the target application conditions. Failures here are not fixed by retraining hyperparameters — they require expanding the training set via active learning.

```python
import numpy as np
from ase.io import read
from ase.md.verlet import VelocityVerlet
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase import units
from mace.calculators import MACECalculator


def nve_energy_conservation_test(
    equilibrium_atoms,
    calc,
    n_steps: int = 10_000,
    timestep_fs: float = 2.0,
    temperature_K: float = 1000.0,
) -> dict:
    """
    Run NVE MD and measure energy drift. The total energy should be conserved
    to within numerical integration error. Drift indicates:
    (a) timestep too large for the MACE force landscape at this temperature, or
    (b) the MLP is exploring configurations outside the training distribution,
        where forces become unphysical and inject energy.

    Reference: for metals at 1000 K with 2 fs timestep, drift < 1 meV/atom/ps
    is acceptable. Tighter tolerances for production property calculations.
    [EXPERT REVIEW NEEDED]
    """
    atoms = equilibrium_atoms.copy()
    atoms.calc = calc
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_K, rng=np.random.default_rng(42))

    # Zero center-of-mass momentum
    mom = atoms.get_momenta()
    mom -= mom.mean(axis=0)
    atoms.set_momenta(mom)

    dyn = VelocityVerlet(atoms, timestep=timestep_fs * units.fs)

    step_energies = []
    max_forces = []
    n_report = max(1, n_steps // 100)

    for step in range(n_steps):
        dyn.run(1)
        e_tot = atoms.get_total_energy()
        f_max = np.abs(atoms.get_forces()).max()
        step_energies.append(e_tot)
        max_forces.append(f_max)

        if f_max > 50.0:  # eV/Å: likely extrapolation; halt
            print(f"HALT at step {step}: max force {f_max:.1f} eV/Å exceeds 50 eV/Å.")
            print("  MLP is likely extrapolating. Collect this frame for active learning.")
            break

    energies = np.array(step_energies)
    natoms = len(atoms)
    dt_ps = timestep_fs / 1000.0
    time_ps = np.arange(len(energies)) * dt_ps

    # Linear fit to detect drift
    slope_eV_ps, intercept = np.polyfit(time_ps, energies, 1)
    drift_meV_atom_ps = slope_eV_ps / natoms * 1000

    print(f"\nNVE energy conservation ({temperature_K:.0f} K, {timestep_fs:.1f} fs):")
    print(f"  Steps completed:    {len(energies)}")
    print(f"  Energy range:       {energies.min():.4f} – {energies.max():.4f} eV")
    print(f"  Std dev:            {energies.std() * 1000 / natoms:.3f} meV/atom")
    print(f"  Linear drift:       {drift_meV_atom_ps:+.4f} meV/atom/ps")
    print(f"  Max force reached:  {max(max_forces):.2f} eV/Å")

    if abs(drift_meV_atom_ps) > 1.0:
        print("  WARNING: drift > 1 meV/atom/ps — consider reducing timestep or "
              "expanding the training set at this temperature")

    return {
        "drift_meV_atom_ps": drift_meV_atom_ps,
        "energy_std_meV_atom": energies.std() * 1000 / natoms,
        "max_force_eV_A": max(max_forces),
        "n_steps_completed": len(energies),
    }


def compute_rdf(
    trajectory_file: str,
    species_pair: tuple[str, str],
    r_max: float = 8.0,
    n_bins: int = 200,
    skip: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute radial distribution function g(r) from an ASE trajectory or extxyz.
    Compare MLP-MD RDF to DFT-MD or experimental reference to validate
    structural properties at finite temperature.

    skip: number of initial frames to discard (equilibration frames).
    Returns (r_centers, g_r) arrays.
    """
    from ase.io import read as ase_read
    frames = ase_read(trajectory_file, index=f"{skip}:", format="extxyz")

    bins = np.linspace(0, r_max, n_bins + 1)
    r_centers = 0.5 * (bins[1:] + bins[:-1])
    counts = np.zeros(n_bins)
    n_ref_atoms = 0
    total_n_pair = 0

    sp1, sp2 = species_pair
    for atoms in frames:
        syms = np.array(atoms.get_chemical_symbols())
        pos = atoms.get_positions()
        cell = atoms.get_cell()
        idx1 = np.where(syms == sp1)[0]
        idx2 = np.where(syms == sp2)[0]
        n_ref_atoms += len(idx1)

        for i in idx1:
            diffs = pos[idx2] - pos[i]
            # Minimum image convention
            diffs -= np.round(diffs @ np.linalg.inv(cell)) @ cell
            r = np.linalg.norm(diffs, axis=1)
            r_in_range = r[(r > 0) & (r < r_max)]
            counts += np.histogram(r_in_range, bins=bins)[0]

    # Normalize to ideal gas density
    V = np.array([a.get_volume() for a in frames]).mean()
    n_frames = len(frames)
    rho = len(np.where(np.array(frames[0].get_chemical_symbols()) == sp2)[0]) / V
    shell_vols = (4/3) * np.pi * (bins[1:]**3 - bins[:-1]**3)
    g_r = counts / (n_frames * n_ref_atoms * rho * shell_vols)

    return r_centers, g_r
```

### Workflow 6: Deployment consistency — ASE vs. LAMMPS force comparison

The same model file must produce identical forces in ASE and LAMMPS. This check must pass before trusting LAMMPS production runs. Element-type ordering, units, and floating-point precision are all sources of silent discrepancies.

```python
import numpy as np
from ase.io import read, write
from mace.calculators import MACECalculator
from pathlib import Path
import subprocess
import tempfile


def check_ase_lammps_consistency(
    model_path: str,
    test_structure,
    element_order: list[str],
    lammps_executable: str = "lmp",
    tolerance_meV_A: float = 1.0,
) -> bool:
    """
    Verify that ASE (MACECalculator) and LAMMPS (pair_style mace) produce
    identical forces on a test structure. Tolerance 1 meV/Å is tight but
    achievable when units, element order, and model file are consistent.

    element_order: LAMMPS atom type ordering, e.g. ["W", "Ta"].
        Must match the order in the LAMMPS data file.
    """
    # ─── ASE forces ────────────────────────────────────────────────────────
    atoms_ase = test_structure.copy()
    calc = MACECalculator(
        model_paths=model_path,
        device="cpu",
        default_dtype="float64",
    )
    atoms_ase.calc = calc
    f_ase = atoms_ase.get_forces()  # (N, 3) eV/Å
    e_ase = atoms_ase.get_potential_energy()

    # ─── LAMMPS forces ─────────────────────────────────────────────────────
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Write LAMMPS data file (atom ordering must match element_order)
        data_file = tmpdir / "test.lammps"
        write(str(data_file), test_structure, format="lammps-data",
              specorder=element_order)

        lammps_input = tmpdir / "test.in"
        lammps_input.write_text(f"""\
units           metal
atom_style      atomic
boundary        p p p
read_data       {data_file}

pair_style      mace no_domain_decomposition
pair_coeff      * * {Path(model_path).resolve()} {' '.join(element_order)}

run             0
write_dump      all custom {tmpdir}/forces.dump id type x y z fx fy fz
""")
        result = subprocess.run(
            [lammps_executable, "-in", str(lammps_input)],
            capture_output=True, text=True, cwd=str(tmpdir),
        )
        if result.returncode != 0:
            print(f"LAMMPS failed:\n{result.stderr[:2000]}")
            return False

        # Parse dump file
        dump_file = tmpdir / "forces.dump"
        f_lammps = np.zeros_like(f_ase)
        with open(dump_file) as df:
            lines = df.readlines()
        # LAMMPS dump: header + N atom lines
        data_start = next(i for i, l in enumerate(lines) if l.startswith("ITEM: ATOMS"))
        for line in lines[data_start + 1:]:
            cols = line.split()
            if len(cols) < 8:
                continue
            atom_id = int(cols[0]) - 1  # 0-indexed
            f_lammps[atom_id] = [float(cols[5]), float(cols[6]), float(cols[7])]

    # ─── Compare ────────────────────────────────────────────────────────────
    diff = np.abs(f_ase - f_lammps)
    max_diff_meV = diff.max() * 1000
    mean_diff_meV = diff.mean() * 1000

    print(f"\nASE vs. LAMMPS force comparison:")
    print(f"  Max  |ΔF|: {max_diff_meV:.4f} meV/Å")
    print(f"  Mean |ΔF|: {mean_diff_meV:.4f} meV/Å")
    print(f"  ASE energy: {e_ase:.6f} eV")

    if max_diff_meV > tolerance_meV_A:
        print(f"  FAIL: max difference {max_diff_meV:.3f} meV/Å exceeds "
              f"tolerance {tolerance_meV_A:.1f} meV/Å")
        print("  Common causes:")
        print("   - Element type order in LAMMPS does not match model training order")
        print("   - pair_coeff element list does not match atom types in data file")
        print("   - Model file path has spaces or special characters")
        print("   - LAMMPS was compiled without no_domain_decomposition support")
        print("   - float32 vs float64 precision mismatch (use float64 in both)")
        return False

    print(f"  PASS: ASE and LAMMPS agree within {tolerance_meV_A:.1f} meV/Å")
    return True


def check_extrapolation_coverage(
    model_paths: list[str],
    production_frames,
    training_file: str,
    sigma_low: float = 0.05,
    sigma_high: float = 0.50,
    device: str = "cpu",
) -> dict:
    """
    Assess how much of a production trajectory is within the MLP's training domain.
    Uses committee model force std as the uncertainty proxy.

    Returns fraction of frames in each uncertainty tier.
    """
    from mace.calculators import MACECalculator

    calcs = [
        MACECalculator(model_paths=mp, device=device, default_dtype="float32")
        for mp in model_paths
    ]

    tier_counts = {"confident": 0, "uncertain": 0, "extrapolation": 0}

    for atoms in production_frames:
        forces_per_model = []
        for calc in calcs:
            a = atoms.copy(); a.calc = calc
            forces_per_model.append(a.get_forces())
        F = np.stack(forces_per_model, axis=0)    # (n_committee, N, 3)
        max_sigma = F.std(axis=0).max()

        if max_sigma < sigma_low:
            tier_counts["confident"] += 1
        elif max_sigma < sigma_high:
            tier_counts["uncertain"] += 1
        else:
            tier_counts["extrapolation"] += 1

    n_total = sum(tier_counts.values())
    print(f"\nExtrapolation coverage ({n_total} production frames):")
    for tier, count in tier_counts.items():
        print(f"  {tier:<15}: {count:>6}  ({count/n_total*100:>5.1f}%)")
    if tier_counts["extrapolation"] / n_total > 0.01:
        print("  WARNING: > 1% extrapolation — model needs active learning refinement")

    return {k: v / n_total for k, v in tier_counts.items()}
```

## Common Failure Modes

| Failure | Symptom | Likely Cause | Diagnostic | Fix |
|---------|---------|--------------|------------|-----|
| Low test error but unstable MD | Test force MAE < 50 meV/Å but MD diverges within 10 ps; energy explosion; max force > 100 eV/Å during MD | Training structures do not cover the dynamically accessible configurations at the target temperature or density; high-energy region of PES is unphysical | Monitor max force during 5 ps NVE test run at target temperature; if any frame exceeds 30 eV/Å the model is extrapolating | Run active learning at the target temperature and conditions; add rattled structures with larger displacements; expand the DFT temperature range |
| Wrong stress sign — NPT box explosion or collapse | NPT MD box volume diverges monotonically; density wrong by a factor of 2 or more; bulk modulus from EOS has wrong sign | Stress labels in training set have wrong sign convention (VASP compression-positive not converted to ASE tensile-positive) | Check equilibrium structure: ASE stress diagonal should be ≈ 0 and positive for a compressed cell; if all components have wrong sign, convention is inverted | Apply `stress = -kBar * KBAR_TO_EV_A3` in the DFT conversion (see mace-training Workflow 1); recompute stress for all training structures |
| Missing chemistry — garbage forces for one species | Forces on one atom type are physically impossible (> 10 eV/Å in equilibrium geometry); energy wrong by > 1 eV/atom for any structure containing that species | Species is absent or nearly absent from the training set; or species appears only in one bonding environment | Report per-species force MAE (Workflow 1); count training frames per species; check species in every config_type | Add DFT calculations covering the missing species in all relevant bonding environments (bulk, surface, defect, alloy neighbor) |
| Poor high-energy behavior — energy catastrophe | Energy correct near equilibrium but diverges sharply at bond compression (< 0.8 × equilibrium bond length); reactive MD or NPT at high P fails | Training data does not cover short interatomic distances; MLP predicts unphysical attractive forces at short range | Sample g(r) during MD; compute energy vs bond length for a dimer at sub-equilibrium separations and compare to DFT | Include structures with compressed bonds in training (volumetric strain series ± 15%); use ZBL-style short-range correction if available for the architecture |
| Data leakage — optimistic test error | Test force MAE appears very low (< 10 meV/Å) but production MD fails quickly; active learning loops overestimate convergence | Adjacent frames from the same MD trajectory split randomly into train and test; near-identical structures in both splits | Compute SOAP distance between every test frame and its nearest training neighbor; if > 50% of test frames have SOAP distance < 0.001, test set is contaminated | Re-split by trajectory or by rattled-origin (see mace-training Workflow 2); ensure test set contains only structures drawn from distinct initial conditions |
| Bad unit conversion — systematic energy offset | Energy MAE is > 0.5 eV/atom despite correct forces; formation energies wrong by integer multiples of a known factor (×13.6 = Ry→eV, ×27.2 = Ha→eV, ×29.3 = Ry/bohr³ stress) | Energies or forces written in wrong units (Hartree or Rydberg instead of eV); distances in bohr instead of Å; stress in Ry/bohr³ | Check raw extxyz energy values: for metals, E/atom should be −4 to −12 eV/atom; for oxides −5 to −10 eV/atom; flag outliers ×13.6 or ×27.2 off | Convert all labels to eV and Å before writing extxyz; use ASE `units.Ry` and `units.Bohr` conversion constants |
| False confidence outside training domain | Committee uncertainty is uniformly low; model reports low σ_F for structures it has never seen; production fails with no uncertainty warning | Committee members have insufficient diversity — same architecture, same data, nearly same random seed; model has memorized the training distribution | Deliberately query a structure well outside the training domain (e.g., 2× compressed unit cell) and check σ_F; if σ_F < σ_low, the committee is overconfident | Increase committee diversity: larger seed separation, jackknife data subsets, or different architectures; calibrate thresholds using held-out out-of-distribution structures |
| EOS curvature wrong — bulk modulus error > 10% | Phonon frequencies systematically too high or too low; thermal expansion coefficient wrong sign or magnitude; NPT density drifts slowly | Stress tensor errors in training data; stress labels computed at different ENCUT or without converged basis; or stress labels excluded from training | Compute EOS at 11 volumes spanning ±10% and compare to DFT; check if stress labels are present in training extxyz (`stress` key in `atoms.info`) | Verify stress labels with `verify_stress_sign` on equilibrium structure; ensure `--stress_weight > 0` in training; recompute stress if ENCUT differs from bulk convergence run |

## Best Practices

- **Validate against DFT, never against another MLP.** Comparing MACE to a classical force field or another neural network measures model agreement, not physical accuracy. Every validation check in this skill requires a DFT reference.
- **Maintain a clean held-out test set from Iteration 0.** Structures targeted by the active learning query strategy are contaminated for unbiased accuracy estimation. A clean test set — assembled before active learning begins, never shown to the model — is the only reliable accuracy gauge across all iterations.
- **Always use float64 precision for validation.** Float32 can mask energy errors below 1 meV/atom and force errors below 5 meV/Å. Enable `default_dtype="float64"` in MACECalculator for all validation evaluations even if training used float32.
- **Run the ASE vs. LAMMPS consistency check before any LAMMPS production run.** Element type ordering, the `pair_coeff` element list, and the `no_domain_decomposition` flag must all be correct. A silent error here produces wrong forces with no error message.
- **Check per-species and per-config-type errors, not just the global MAE.** A 30 meV/Å global MAE may hide 300 meV/Å errors for a dopant species or a surface configuration that appears in < 5% of the test set. Report and inspect every config_type separately.
- **Run the NVE energy conservation test at the target temperature before committing to NPT.** A drift > 1 meV/atom/ps at target temperature is a contraindication for production MD. Reduce the timestep first (try 1 fs); if drift persists, the model needs active learning at that temperature.
- **Validate EOS before running NPT.** The NPT barostat uses forces and stresses to adjust the cell. A wrong stress sign or a B₀ error > 20% will cause unphysical density behavior. Run Workflow 3 and confirm V₀ and B₀ before enabling pressure coupling.
- **Check for imaginary phonon modes at Γ.** Even one imaginary mode in a nominally stable crystal is a definitive indicator of a locally incorrect MLP PES. Compute Γ-point phonons at negligible cost and flag any imaginary frequency < −0.1 THz.
- **Document the validation suite results in the project README** with model checksum, MACE version, test set size and composition, and achieved metrics. This is the minimum reproducibility record for MLP deployment.
- **For surfaces, defects, and reactions, validate each separately.** Bulk validation passing does not certify surface or defect accuracy. Include per-environment validation whenever the target application involves environments that are structurally distinct from bulk. [EXPERT REVIEW NEEDED]

## Integration with Other Skills

- **[mace-training](../mace-training/SKILL.md):** This skill follows mace-training in the MLP development workflow. The test set from that skill's stratified split is the primary input to Workflow 1. The stress sign convention and float64 validation prescription from mace-training carry directly into this skill.
- **[active-learning-mlp](../active-learning-mlp/SKILL.md):** Workflow 6 here (NVE stability test and extrapolation coverage check) is the pre-deployment gate before declaring active learning convergence. When validation fails, the active learning loop must be resumed with the failing structures as targets for new DFT labels.
- **[mlp-to-lammps](../mlp-to-lammps/SKILL.md):** Workflow 6 (ASE vs. LAMMPS consistency check) verifies the LAMMPS deployment described in detail in that skill. The element-type ordering requirements and `no_domain_decomposition` flag covered there directly affect the consistency check.
- **[uncertainty-mlp](../uncertainty-mlp/SKILL.md):** The committee-based extrapolation coverage check in Workflow 6 is the simplest uncertainty quantification approach. That skill covers principled alternatives (conformal prediction, last-layer Bayesian estimation) and when calibration of uncertainty estimates is required beyond the σ_F threshold approach.
- **[dft-dataset-generation](../../electronic-structure/dft-dataset-generation/SKILL.md):** When validation reveals systematic failures (missing config_type, wrong chemistry for a species, poor surface accuracy), the remedy is always to generate more targeted DFT data. That skill covers how to design the supplementary DFT calculations.
- **[dft-convergence](../../electronic-structure/dft-convergence/SKILL.md):** DFT reference calculations for phonon, EOS, and elastic constant validation must be computed at the same level of theory as the training data. If a discrepancy appears between MLP and DFT property references, verify that the reference DFT settings are identical to those used for training before concluding the MLP is wrong.
- **[vasp-workflow](../../electronic-structure/vasp-workflow/SKILL.md):** DFT reference calculations for EOS, phonon, and elastic property validation are typically run with VASP. The ISIF settings (ISIF=3 for cell relaxation, ISIF=2 for stress at fixed cell) from that skill are required for computing accurate reference stress tensors and elastic constants.
- **[quantum-espresso](../../electronic-structure/quantum-espresso/SKILL.md):** When QE is the reference code for validation, phonopy can use QE force outputs directly. Workflow 4's finite displacement approach applies identically; only the DFT interface changes.
- **[lammps-workflow](../../atomistic-md/lammps-workflow/SKILL.md):** The NVE and NVT MD stability tests in Workflow 5 can be run in LAMMPS as well as ASE. The equilibration protocol, timestep conventions, and thermostat settings from that skill apply to MACE-driven LAMMPS validation runs.
- **[md-equilibration](../../atomistic-md/md-equilibration/SKILL.md):** The pre-validation equilibration of the starting structure for NVE tests must follow the protocol from that skill. Starting from a poorly equilibrated structure inflates the apparent energy drift and produces a pessimistic NVE assessment.
- **[dft-phonons](../../electronic-structure/dft-phonons/SKILL.md):** The DFT reference phonon calculation that MLP phonons are validated against is described in that skill. The supercell construction, displacement protocol, and BORN charge handling for polar materials described there must match the phonopy setup in Workflow 4.
- **[eos-fitting](../../electronic-structure/eos-fitting/SKILL.md):** The Birch-Murnaghan EOS fitting and the DFT reference EOS required for Workflow 3 follow the protocol in that skill. When DFT and MLP EOS are computed consistently, discrepancies in V₀ and B₀ are unambiguously attributable to the MLP.

## Key References

1. Zuo, Y., et al. (2020). *Performance and cost assessment of machine learning interatomic potentials.* The Journal of Physical Chemistry A, 124(4), 731–745. https://doi.org/10.1021/acs.jpca.9b08723
2. Behler, J. (2016). *Perspective: Machine learning potentials for atomistic simulations.* The Journal of Chemical Physics, 145(17), 170901. https://doi.org/10.1063/1.4966192
3. Musil, F., et al. (2021). *Physics-inspired structural representations for molecules and materials.* Chemical Reviews, 121(16), 9759–9815. https://doi.org/10.1021/acs.chemrev.1c00021
4. Miksch, A. M., et al. (2021). *Strategies for the construction of machine-learning potentials for accurate and efficient atomic-scale simulations.* Machine Learning: Science and Technology, 2(3), 031001. https://doi.org/10.1088/2632-2153/abfd96
5. Fu, X., et al. (2023). *Forces are not enough: Benchmark and critical evaluation of machine learning force fields for molecular simulations.* Transactions on Machine Learning Research. https://openreview.net/forum?id=A8pqQipwkt
