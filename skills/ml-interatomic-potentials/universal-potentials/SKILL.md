---
name: universal-potentials
domain: ml-interatomic-potentials
maturity: draft
tools:
  - MACE
  - CHGNet
  - ASE
  - pymatgen
  - LAMMPS
tested_versions:
  MACE: "0.3.6"
  CHGNet: "0.3.8"
  ASE: "3.23.0"
  pymatgen: "2024.3.1"
  LAMMPS: "2Aug2023"
requires_gpu: false
estimated_runtime: minutes-to-hours
---

# Universal Machine-Learned Interatomic Potentials

## Description

This skill covers the selection, evaluation, and deployment of pretrained universal machine-learned interatomic potentials (universal MLPs) — models trained on large multi-element datasets that can be applied to new chemical systems without additional DFT labeling. It covers the major model families (MACE-MP, CHGNet, M3GNet, MatterSim, ORB, SevenNet), their supported chemistry and known limitations, practical use for relaxation, screening, pre-equilibration, structure ranking, and dataset triage, and the decision framework for choosing between a universal potential, fine-tuning, and training from scratch. Invoke this skill before committing to any new DFT dataset generation campaign, to establish whether a universal potential already achieves acceptable accuracy for the target application.

## Domain Context

Universal MLPs are trained on databases of DFT calculations spanning a large fraction of the periodic table and a wide range of crystal structure types. The two dominant training databases are the Materials Project (MP) — approximately 150,000 relaxed inorganic crystal structures computed with PBE+U — and the Alexandria/OMAT24 databases that extend coverage to higher-energy configurations, surfaces, and molecular systems. [EXPERT REVIEW NEEDED: database coverage evolves rapidly; verify against the model release notes before deployment]

**What universal training databases cover.** Most universal potentials are trained on near-equilibrium inorganic crystals at 0 K (or modest finite-temperature MD snapshots) computed at the PBE or PBE+U level of DFT with PAW pseudopotentials. The training set is diverse in composition but concentrated in low-energy, periodic, charge-neutral, diamagnetic or ferromagnetic phases. Coverage of molecules, surfaces, defects, liquids, high-pressure polymorphs, mixed-valence compounds, and charged supercells is sparse or absent in the original training data of most models, though newer models (MatterSim, ORB-v2, MACE-MP-0b) have broader coverage. [EXPERT REVIEW NEEDED]

**The core trade-off: breadth versus depth.** A universal potential can be evaluated on any supported element combination without additional training, but its accuracy for any specific system is generally lower than a well-converged system-specific MLP trained on targeted DFT data. The practical question is whether the universal potential's accuracy is already sufficient for the intended application (screening, qualitative ranking, pre-relaxation) or whether system-specific accuracy (force MAE < 50 meV/Å, phonon frequencies within 5%, correct phase ordering) is required.

**DFT level consistency.** Universal potentials implicitly represent the DFT level of their training data. MACE-MP-0 and CHGNet target PBE+U energetics. Applying them to compute formation energies for systems with correlated electrons (late transition metal oxides, rare earths) will inherit PBE+U errors plus any model approximation errors. If the target application requires a different DFT functional (HSE06, meta-GGA, dispersion-corrected PBE-D3), a universal potential trained at PBE level is not a substitute for DFT at the target level — it is an approximation to the wrong reference.

**Magnetic and spin degrees of freedom.** Standard universal potentials do not model spin explicitly. CHGNet includes a magnetic moment prediction head and was trained on spin-polarized calculations, making it the preferred choice for magnetic materials. MACE-MP-0 was trained on non-spin-polarized PBE; its energetics for magnetic systems are unreliable unless the magnetic ground state is well-captured by a non-magnetic approximation. [EXPERT REVIEW NEEDED]

**Long-range electrostatics.** All current widely deployed universal MLPs use local message passing with a fixed cutoff radius (5–6 Å typically). They cannot reproduce the Madelung energy of ionic crystals, the dipole-dipole interaction in polar molecules, or the image-charge interaction at charged surfaces. This is a fundamental architectural limitation, not a training-data limitation. For systems where electrostatic energy is a significant fraction of the total (ionic oxides, polar surfaces, charged defects, zwitterions), predictions will be qualitatively wrong at separations beyond the cutoff. Long-range equivariant models (MACE-off, NequIP-LR, PaiNN-LR) address this but are not yet universally deployed. [EXPERT REVIEW NEEDED]

**Version instability.** Universal potential model weights change between releases, and each new release typically changes predicted energies, forces, and phase stability for at least a subset of materials. Always record the exact model name, version, and checkpoint SHA-256 in any result that will be compared to future calculations. Using different model versions across an active learning campaign is a common source of label inconsistency.

## When to Use This Skill

- **Structure pre-relaxation before DFT.** Running a universal potential relaxation before the DFT step eliminates poor starting geometries, reduces DFT ionic steps, and prevents DFT crashes from unphysical initial structures. Even a model with modest force errors substantially improves DFT convergence rates.
- **High-throughput screening of large structure sets.** Screening tens of thousands of hypothetical structures for stability, formation energy ordering, or approximate lattice parameters before committing any to DFT. Universal MLPs run in seconds per structure and often correctly identify the top-N candidates.
- **MD pre-equilibration before DFT-MD or system-specific MLP-MD.** Equilibrating a large supercell at the target temperature and pressure using a universal potential, then sampling starting configurations for DFT or MACE training. Reduces the number of DFT steps required to reach equilibrium.
- **Dataset triage and coverage analysis.** Using universal potential force errors as a proxy for whether a new composition is inside or outside the existing DFT dataset coverage, before deciding whether to run new DFT calculations.
- **Quick sanity checks on new structures.** Checking whether a proposed structure is mechanically stable (positive definite elastic tensor, positive phonon frequencies at Gamma) at near-zero DFT cost, to filter out physically unreasonable structures before DFT.
- **Baseline model for benchmarking system-specific MLPs.** Establishing a reference accuracy floor: if a system-specific MLP does not outperform MACE-MP-0 on the target validation metrics, the system-specific training has likely gone wrong.
- **Systems where rough accuracy suffices.** Applications where formation energy errors of 50–200 meV/atom are acceptable: coarse phase diagram sketching, composition-space exploration, initial structure generation for crystal structure prediction.

## When Not to Use This Skill

- **As a substitute for system-specific DFT accuracy.** Universal potentials are not certified replacements for DFT on any specific system. Force errors of 100–500 meV/Å are common outside the training distribution. Never use universal potential energies as DFT labels for MLP training without validation. See the failure mode: "using universal potential labels as if they were DFT labels."
- **For charged systems, ionically stabilized phases, or explicitly protonated/deprotonated molecules.** Current universal MLPs have no mechanism for representing net charge. Proton transfer, charged defects (V_O^{2+}, h^+), and ionic dissociation are not reliably described.
- **For magnetic phase transitions, magnetically frustrated systems, or antiferromagnets.** Standard universal potentials (MACE-MP-0, M3GNet, ORB) are trained without spin and will report incorrect energy ordering between ferromagnetic and antiferromagnetic configurations. Use CHGNet for systems where magnetic structure matters, and validate the magnetic ground state independently.
- **For molecular crystals, polymers, and organic molecules where dispersion is dominant.** PBE-level universal MLPs without dispersion correction underestimate intermolecular binding, giving incorrect lattice constants, packing, and cohesive energies for organic solids. MACE-OFF23 is specifically trained for organic molecules and molecular crystals.
- **For properties at extreme conditions (P > 100 GPa, T > 3000 K)** that are poorly represented in the training databases. High-pressure polymorphs, shock equations of state, and laser-heated DAC configurations are outside most universal potentials' training distribution.
- **For aqueous electrochemistry, liquid electrolytes, or solid-liquid interfaces.** Explicit solvation, electrode-electrolyte charge transfer, and ion solvation energetics require models trained on liquid configurations at the relevant temperature — not present in the standard MP-based training sets.
- **When the result will be published as a DFT property.** If a paper claims a DFT-calculated property, the result must come from actual DFT. Universal MLP predictions are MLP predictions, not DFT results. This is an integrity issue, not only an accuracy issue.

## Core Concepts

- **Universal MLP:** A machine-learned interatomic potential trained on a large, multi-system, multi-element DFT database, intended to generalize across chemical space without system-specific retraining. Distinguished from system-specific MLPs by breadth of training, not by architecture. The same MACE architecture can be used as either a system-specific or universal potential depending on training data.
- **MACE-MP-0 (and variants):** A family of universal MACE models trained by Batatia et al. on the Materials Project PBE+U dataset. Multiple model sizes (small, medium, large) with different parameter counts and accuracy/cost trade-offs. MACE-MP-0b/0b3 are improved versions with updated training protocols. [EXPERT REVIEW NEEDED: the model family is actively updated; check the mace-mp repository for the latest release]
- **CHGNet (Crystal Hamiltonian Graph Network):** A universal GNN trained by Deng et al. on the Materials Project, including a magnetic moment prediction head trained on spin-polarized calculations. The only widely deployed universal MLP that explicitly models magnetic moments. Recommended for any system where magnetic ordering matters.
- **M3GNet:** A universal GNN by Chen and Ong trained on the Materials Project. Less accurate than MACE-MP-0 per benchmark but lighter-weight and part of the MatGL stack that also includes the PotNet and CHGNet implementations.
- **MatterSim:** A universal MLP from Microsoft Research trained on a broader dataset (MP + additional high-temperature and high-pressure configurations). Aims to improve accuracy at non-zero temperatures and extreme conditions. [EXPERT REVIEW NEEDED]
- **ORB (Open-source Research-grade B-NN):** A family of models trained on MP + additional datasets, designed for fast inference. ORB-v2 includes improved elemental coverage. [EXPERT REVIEW NEEDED]
- **SevenNet:** A universal GNN (NequIP-based) from KAIST trained on the OMAT24 dataset (MPtraj + Alexandria MD snapshots). High accuracy on non-equilibrium structures. [EXPERT REVIEW NEEDED]
- **MACE-OFF23:** A universal MACE model for organic molecules and molecular crystals, trained on SPICE and related databases at the ωB97M-D3(BJ)/def2-TZVPPD level. The appropriate choice for organic chemistry applications instead of MACE-MP (which targets inorganic crystals at PBE level).
- **Zero-shot application:** Applying a universal potential to a new system without any additional training or fine-tuning. Accuracy depends entirely on how well the new system is represented in the training distribution.
- **In-domain vs. out-of-domain (OOD):** A universal potential is in-domain for a new system if that system's structural and chemical features are well-represented in the training database. OOD indicators: rare elements, unusual coordination geometries, compositions outside the training convex hull, phases absent from MP. Check with SOAP dissimilarity to the training set or use the model's own uncertainty estimate if available.
- **Fine-tuning decision boundary:** Fine-tuning a universal potential on system-specific DFT data is efficient when the universal potential is close but not accurate enough — typically when force MAE is < 5× the target accuracy. If the universal potential shows no correlation with DFT forces (MAE > 1 eV/Å for metallic systems), training from scratch on system-specific data may converge faster. [EXPERT REVIEW NEEDED]
- **Training-set overlap (data leakage):** When validating a universal potential on a structure that was part of the training database (e.g., a Materials Project entry), measured errors reflect interpolation, not generalization. Always test on structures not in the training database when reporting generalization accuracy.
- **Equivariant vs. invariant universal potentials:** MACE-MP and SevenNet are equivariant (forces computed as equivariant tensor derivatives). M3GNet and CHGNet are invariant (use only distance and angle information). Equivariant models are generally more accurate per parameter but slower per evaluation.

## Key Workflows

### Workflow 1: Benchmarking a universal potential on a target system before committing to DFT

The first step for any new system is to run a held-out DFT validation before relying on the universal potential for any downstream use. This workflow computes force, energy, and stress errors on a set of DFT-labeled structures and produces a decision report.

```python
import numpy as np
import json
from pathlib import Path
from ase.io import read
from mace.calculators import mace_mp
from chgnet.model import CHGNet
from chgnet.model.dynamics import CHGNetCalculator


def load_dft_frames(test_file: str) -> list:
    """Load DFT-labeled frames from extxyz. Expects 'energy', 'forces' in metadata."""
    frames = read(test_file, index=":", format="extxyz")
    valid = []
    for atoms in frames:
        has_forces = atoms.arrays.get("forces") is not None
        has_energy = atoms.info.get("energy") is not None or atoms.get_potential_energy is not None
        if has_forces:
            valid.append(atoms)
    print(f"Loaded {len(valid)} frames with DFT forces from {test_file}")
    return valid


def evaluate_universal_potential(
    frames: list,
    model_name: str = "mace-mp",
    model_size: str = "medium",
    device: str = "cpu",
) -> dict:
    """
    Evaluate a universal potential on DFT-labeled frames.
    Returns per-structure energy and force errors.

    model_name: "mace-mp" | "chgnet" | "m3gnet"
    model_size: for mace-mp: "small" | "medium" | "large" (ignored for chgnet/m3gnet)
    """
    if model_name == "mace-mp":
        calc = mace_mp(model=model_size, dispersion=False, default_dtype="float64", device=device)
        model_label = f"MACE-MP-0 ({model_size})"
    elif model_name == "chgnet":
        model = CHGNet.load()
        calc = CHGNetCalculator(model=model, use_device=device)
        model_label = f"CHGNet {model.version}"
    elif model_name == "m3gnet":
        # Requires matgl package
        import matgl
        from matgl.ext.ase import PESCalculator
        pot = matgl.load_model("M3GNet-MP-2021.2.8-PES")
        calc = PESCalculator(pot)
        model_label = "M3GNet-MP-2021.2.8"
    else:
        raise ValueError(f"Unknown model_name '{model_name}'. Use 'mace-mp', 'chgnet', or 'm3gnet'.")

    energy_errors = []   # meV/atom
    force_maes = []      # meV/Å
    force_rmses = []     # meV/Å
    stress_maes = []     # GPa (if available)

    for atoms in frames:
        n_atoms = len(atoms)
        ref_forces = atoms.arrays["forces"]  # (N, 3) eV/Å
        ref_energy = atoms.info.get("energy")  # eV

        a = atoms.copy()
        a.calc = calc
        pred_energy = a.get_potential_energy()
        pred_forces = a.get_forces()

        if ref_energy is not None:
            energy_errors.append((pred_energy - ref_energy) / n_atoms * 1000)  # meV/atom

        force_diff = pred_forces - ref_forces
        force_maes.append(np.abs(force_diff).mean() * 1000)    # meV/Å
        force_rmses.append(np.sqrt((force_diff**2).mean()) * 1000)

        ref_stress = atoms.info.get("stress")
        if ref_stress is not None:
            try:
                pred_stress = a.get_stress(voigt=True)
                # Convert eV/Å³ → GPa: 1 eV/Å³ = 160.218 GPa
                stress_maes.append(np.abs(pred_stress - ref_stress).mean() * 160.218)
            except Exception:
                pass

    results = {
        "model": model_label,
        "n_structures": len(frames),
        "force_mae_meV_A": float(np.mean(force_maes)),
        "force_rmse_meV_A": float(np.mean(force_rmses)),
        "force_mae_p90_meV_A": float(np.percentile(force_maes, 90)),
        "force_mae_max_meV_A": float(np.max(force_maes)),
    }
    if energy_errors:
        results["energy_mae_meV_atom"] = float(np.mean(np.abs(energy_errors)))
        results["energy_rmse_meV_atom"] = float(np.sqrt(np.mean(np.array(energy_errors)**2)))
    if stress_maes:
        results["stress_mae_GPa"] = float(np.mean(stress_maes))

    print(f"\n{model_label} — validation on {len(frames)} structures:")
    print(f"  Force MAE:    {results['force_mae_meV_A']:.1f} meV/Å")
    print(f"  Force RMSE:   {results['force_rmse_meV_A']:.1f} meV/Å")
    print(f"  Force MAE p90:{results['force_mae_p90_meV_A']:.1f} meV/Å  (worst 10% of frames)")
    if "energy_mae_meV_atom" in results:
        print(f"  Energy MAE:   {results['energy_mae_meV_atom']:.2f} meV/atom")
    if "stress_mae_GPa" in results:
        print(f"  Stress MAE:   {results['stress_mae_GPa']:.3f} GPa")

    return results


def benchmark_universal_potentials(
    test_file: str,
    models: list[tuple[str, str]] = None,
    output_json: str = "universal_potential_benchmark.json",
    device: str = "cpu",
) -> dict:
    """
    Benchmark multiple universal potentials on the same DFT test set.
    Produces a decision report: use as-is, fine-tune, or train from scratch.

    models: list of (model_name, model_size) tuples.
        Default: [("mace-mp", "medium"), ("chgnet", ""), ("mace-mp", "large")]
    """
    if models is None:
        models = [("mace-mp", "small"), ("mace-mp", "medium"), ("chgnet", "")]

    frames = load_dft_frames(test_file)
    if len(frames) < 20:
        print(f"WARNING: Only {len(frames)} test structures. "
              f"Benchmarks on < 20 structures are statistically unreliable.")

    all_results = {}
    for model_name, model_size in models:
        key = f"{model_name}_{model_size}" if model_size else model_name
        try:
            all_results[key] = evaluate_universal_potential(
                frames, model_name=model_name, model_size=model_size, device=device
            )
        except Exception as e:
            print(f"WARNING: {model_name} ({model_size}) failed: {e}")
            all_results[key] = {"error": str(e)}

    # ─── Decision logic ──────────────────────────────────────────────────────
    # Heuristic thresholds — must be overridden for system-specific requirements
    # [EXPERT REVIEW NEEDED: system-dependent thresholds]
    print("\n" + "=" * 60)
    print("DECISION REPORT")
    print("=" * 60)
    for key, res in all_results.items():
        if "error" in res:
            print(f"\n{key}: FAILED — {res['error']}")
            continue
        f_mae = res["force_mae_meV_A"]
        print(f"\n{res['model']}  force MAE = {f_mae:.0f} meV/Å")
        if f_mae < 100:
            print("  → Acceptable for screening/pre-relaxation without fine-tuning")
        elif f_mae < 300:
            print("  → Consider fine-tuning on 200–1000 system-specific DFT structures")
        elif f_mae < 1000:
            print("  → Fine-tuning may help; benchmark after 500+ structures")
        else:
            print("  → Consider training from scratch; universal potential shows no correlation")

    Path(output_json).write_text(json.dumps(all_results, indent=2))
    print(f"\nFull results written to {output_json}")
    return all_results
```

### Workflow 2: Universal potential relaxation and screening pipeline

Using a universal potential for high-throughput structure relaxation and stability screening of a large set of candidate structures.

```python
import numpy as np
from pathlib import Path
from ase.io import read, write
from ase.optimize import FIRE
from ase.constraints import ExpCellFilter
from mace.calculators import mace_mp


def relax_with_universal_potential(
    atoms,
    calc,
    fmax: float = 0.05,
    max_steps: int = 300,
    relax_cell: bool = True,
    pressure_GPa: float = 0.0,
) -> tuple:
    """
    Relax a structure using a universal potential.

    fmax: convergence criterion in eV/Å.
    relax_cell: also optimize cell shape and volume.
    pressure_GPa: target hydrostatic pressure (0 = ambient).
    Returns: (relaxed_atoms, converged: bool, n_steps: int)
    """
    a = atoms.copy()
    a.calc = calc

    if relax_cell and a.pbc.all():
        # ExpCellFilter allows joint atom+cell optimization
        # pressure in ASE units: eV/Å³ (1 GPa = 0.006242 eV/Å³)
        pressure_eV_A3 = pressure_GPa * 0.006242
        ecf = ExpCellFilter(a, scalar_pressure=pressure_eV_A3)
        opt = FIRE(ecf, logfile=None)
    else:
        opt = FIRE(a, logfile=None)

    try:
        opt.run(fmax=fmax, steps=max_steps)
        converged = opt.get_number_of_steps() < max_steps
    except Exception as e:
        print(f"  WARNING: relaxation failed ({e})")
        converged = False

    return a, converged, opt.get_number_of_steps()


def screen_structures(
    input_file: str,
    output_relaxed: str = "relaxed_structures.extxyz",
    output_report: str = "screening_report.json",
    model_size: str = "medium",
    fmax: float = 0.05,
    max_steps: int = 300,
    energy_cutoff_meV_atom: float = 500.0,
    device: str = "cpu",
) -> None:
    """
    Relax and screen a set of candidate structures using MACE-MP.

    energy_cutoff_meV_atom: structures with formation energy above this value
        (relative to the lowest-energy structure in the set) are discarded.
        This is a rough stability filter — it does not compute the convex hull.

    Stores per-structure metadata for downstream DFT prioritization:
        relaxed geometry, predicted energy, convergence status, n_steps.
    """
    structures = read(input_file, index=":", format="extxyz")
    print(f"Screening {len(structures)} structures with MACE-MP-0 ({model_size})...")

    calc = mace_mp(model=model_size, dispersion=False, default_dtype="float32", device=device)

    relaxed_frames = []
    report = []

    for i, atoms in enumerate(structures):
        tag = atoms.info.get("config_type", f"structure_{i:04d}")
        relaxed, converged, n_steps = relax_with_universal_potential(
            atoms, calc, fmax=fmax, max_steps=max_steps
        )

        e_per_atom = relaxed.get_potential_energy() / len(relaxed)
        relaxed.info["mace_mp_energy_per_atom_eV"] = e_per_atom
        relaxed.info["mace_mp_relaxation_converged"] = converged
        relaxed.info["mace_mp_relaxation_steps"] = n_steps
        relaxed.info["config_type"] = tag
        relaxed.info["source_index"] = i
        relaxed_frames.append(relaxed)

        report.append({
            "index": i,
            "tag": tag,
            "n_atoms": len(atoms),
            "energy_per_atom_eV": e_per_atom,
            "converged": converged,
            "n_steps": n_steps,
        })

        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(structures)}] done...")

    # ─── Stability filter ────────────────────────────────────────────────────
    energies = np.array([r["energy_per_atom_eV"] for r in report])
    e_ref = energies.min()
    e_above_min_meV = (energies - e_ref) * 1000  # meV/atom

    for r, dE in zip(report, e_above_min_meV):
        r["meV_above_minimum"] = float(dE)
        r["passes_energy_filter"] = dE < energy_cutoff_meV_atom

    n_pass = sum(r["passes_energy_filter"] for r in report)
    n_converged = sum(r["converged"] for r in report)

    print(f"\nScreening complete:")
    print(f"  {n_converged}/{len(structures)} relaxations converged")
    print(f"  {n_pass}/{len(structures)} pass energy filter "
          f"(< {energy_cutoff_meV_atom:.0f} meV/atom above minimum)")
    print(f"  NOTE: This energy filter uses MACE-MP internal energies, "
          f"not formation energies vs. elemental references. "
          f"Use pymatgen ComputedEntry for convex hull stability.")

    # Write output
    selected = [f for f, r in zip(relaxed_frames, report) if r["passes_energy_filter"]]
    write(output_relaxed, selected, format="extxyz")
    print(f"  Relaxed and filtered structures written to {output_relaxed}")

    import json
    Path(output_report).write_text(json.dumps(report, indent=2))
    print(f"  Screening report written to {output_report}")
```

### Workflow 3: Out-of-domain detection for universal potential predictions

Before trusting a universal potential prediction, check whether the target structures fall within the model's training distribution. Uses SOAP dissimilarity against a reference set of Materials Project structures.

```python
import numpy as np
from ase.io import read
from dscribe.descriptors import SOAP
from sklearn.preprocessing import normalize


def check_universal_potential_domain(
    query_structures: list,
    reference_file: str,
    species: list[str],
    r_cut: float = 5.0,
    n_max: int = 9,
    l_max: int = 6,
    sigma: float = 0.5,
    ood_threshold: float = None,
    output_report: str = "ood_report.json",
) -> dict:
    """
    Estimate how in-distribution query structures are relative to a reference
    set (typically a sample of the training database or a set of validated
    structures for the target composition space).

    query_structures: list of ASE Atoms objects to check.
    reference_file: extxyz file of reference structures (e.g., MP structures
        for the same composition space, or the actual training set sample).
    ood_threshold: SOAP distance above which a structure is flagged as OOD.
        If None, uses the 95th percentile of within-reference pairwise distances
        as a heuristic (conservative default). [EXPERT REVIEW NEEDED]

    Returns dict with per-structure SOAP distances and OOD flags.
    """
    ref_frames = read(reference_file, index=":", format="extxyz")
    print(f"Reference set: {len(ref_frames)} structures")
    print(f"Query set:     {len(query_structures)} structures")

    soap = SOAP(
        species=species, r_cut=r_cut, n_max=n_max, l_max=l_max,
        sigma=sigma, average="inner", periodic=True,
    )

    print("Computing SOAP descriptors for reference set...")
    ref_descs = normalize(soap.create(ref_frames), norm="l2")

    print("Computing SOAP descriptors for query set...")
    query_descs = normalize(soap.create(query_structures), norm="l2")

    # Min distance from each query to reference set
    cos_sim = query_descs @ ref_descs.T            # (N_query, N_ref)
    cos_sim = np.clip(cos_sim, -1.0, 1.0)
    distances = np.sqrt(np.maximum(0, 2 - 2 * cos_sim))  # Euclidean on unit sphere
    min_dists = distances.min(axis=1)               # (N_query,)

    if ood_threshold is None:
        # Estimate threshold from within-reference spread
        # Use a random subsample for large reference sets
        n_ref_sample = min(len(ref_frames), 500)
        idx = np.random.choice(len(ref_frames), n_ref_sample, replace=False)
        ref_sub = ref_descs[idx]
        cs = ref_sub @ ref_sub.T
        np.fill_diagonal(cs, -2.0)  # Exclude self-distances
        d_within = np.sqrt(np.maximum(0, 2 - 2 * np.clip(cs, -1, 1)))
        np.fill_diagonal(d_within, np.nan)
        min_within = np.nanmin(d_within, axis=1)
        ood_threshold = float(np.percentile(min_within, 95))
        print(f"  Auto-threshold (95th pct of within-reference min-distances): "
              f"{ood_threshold:.4f}")
        print(f"  NOTE: This threshold is system-dependent. Validate against "
              f"known in-distribution and out-of-distribution structures before use.")

    n_ood = (min_dists > ood_threshold).sum()
    print(f"\nOOD detection results (threshold={ood_threshold:.4f}):")
    print(f"  {n_ood}/{len(query_structures)} structures flagged as OOD")
    print(f"  Median SOAP distance to reference: {np.median(min_dists):.4f}")
    print(f"  Max SOAP distance to reference:    {max(min_dists):.4f}")

    if n_ood > 0:
        print(f"  WARNING: {n_ood} structures are likely outside the model's "
              f"training distribution. Universal potential predictions for these "
              f"structures should be treated as qualitative estimates only.")

    report = {
        "ood_threshold": float(ood_threshold),
        "structures": [
            {
                "index": i,
                "tag": getattr(q, "info", {}).get("config_type", f"s{i}"),
                "soap_distance_to_reference": float(d),
                "is_ood": bool(d > ood_threshold),
            }
            for i, (q, d) in enumerate(zip(query_structures, min_dists))
        ],
    }

    import json
    Path(output_report).write_text(json.dumps(report, indent=2))
    return report


def summarize_ood_by_element(
    query_structures: list,
    ood_flags: list[bool],
) -> None:
    """
    Report which elements are most associated with OOD structures.
    Helps identify whether OOD is driven by a specific element or coordination type.
    """
    from collections import Counter

    ood_elements = Counter()
    all_elements = Counter()

    for atoms, is_ood in zip(query_structures, ood_flags):
        symbols = atoms.get_chemical_symbols()
        all_elements.update(symbols)
        if is_ood:
            ood_elements.update(symbols)

    print("\nElements in OOD structures vs. all structures:")
    for elem in sorted(ood_elements, key=lambda e: -ood_elements[e]):
        ood_frac = ood_elements[elem] / max(all_elements[elem], 1)
        print(f"  {elem}: {ood_elements[elem]}/{all_elements[elem]} atoms "
              f"in OOD structures ({ood_frac:.0%})")
```

### Workflow 4: Pre-relaxation pipeline for DFT input preparation

Using a universal potential to pre-relax unoptimized structures (from CIF databases, crystal structure prediction, or random substitutions) before DFT, reducing DFT ionic steps and preventing failed calculations.

```python
import numpy as np
from pathlib import Path
from ase.io import read, write
from ase.optimize import FIRE
from ase.constraints import ExpCellFilter
from mace.calculators import mace_mp


def prerelax_for_dft(
    input_file: str,
    output_file: str = "prerelaxed_for_dft.extxyz",
    model_size: str = "medium",
    fmax_coarse: float = 0.10,
    max_steps: int = 200,
    min_interatomic_distance: float = 1.2,
    device: str = "cpu",
) -> None:
    """
    Pre-relax structures with MACE-MP before DFT submission.

    fmax_coarse: convergence threshold for pre-relaxation. Use 0.10 eV/Å (not
        the DFT target 0.01–0.02 eV/Å); over-converging wastes compute and
        the DFT will finish the job.
    min_interatomic_distance: discard structures with any atom pair closer than
        this value (Å) after relaxation — likely unphysical and will cause DFT
        crashes. Typical clash threshold for common elements: 1.2 Å.
    """
    structures = read(input_file, index=":", format="extxyz")
    calc = mace_mp(model=model_size, dispersion=False, default_dtype="float32", device=device)

    output_frames = []
    n_clash = 0
    n_unconverged = 0

    for i, atoms in enumerate(structures):
        relaxed, converged, n_steps = relax_with_universal_potential(
            atoms, calc, fmax=fmax_coarse, max_steps=max_steps, relax_cell=True
        )
        if not converged:
            n_unconverged += 1

        # Check for atomic clashes after relaxation
        from ase.geometry import get_distances
        dists = relaxed.get_all_distances(mic=True)
        np.fill_diagonal(dists, np.inf)
        min_dist = dists.min()

        if min_dist < min_interatomic_distance:
            n_clash += 1
            print(f"  Structure {i} ({atoms.info.get('config_type','?')}): "
                  f"min dist={min_dist:.2f} Å < {min_interatomic_distance} Å after relaxation. "
                  f"Skipping — likely unphysical geometry.")
            continue

        relaxed.info["prerelaxation_converged"] = converged
        relaxed.info["prerelaxation_steps"] = n_steps
        relaxed.info["prerelaxation_min_dist_A"] = float(min_dist)
        relaxed.info["prerelaxation_model"] = f"MACE-MP-0 {model_size}"
        output_frames.append(relaxed)

    write(output_file, output_frames, format="extxyz")
    print(f"\nPre-relaxation summary:")
    print(f"  Input structures: {len(structures)}")
    print(f"  Output structures: {len(output_frames)}")
    print(f"  Discarded (clashes): {n_clash}")
    print(f"  Unconverged (check geometry): {n_unconverged}")
    print(f"  Written to {output_file}")
    print(f"  NOTE: These structures are ready for DFT INCAR submission. "
          f"Use ISTART=0, NSW>=50, EDIFF=1e-5 for the first DFT step. "
          f"Do NOT use as DFT labels — they are MACE-MP geometries, not DFT geometries.")


def validate_prerelaxed_vs_dft(
    prerelaxed_file: str,
    dft_relaxed_file: str,
) -> None:
    """
    After running DFT on pre-relaxed structures, compare geometries to check
    whether the pre-relaxation was a useful starting point.

    Reports: mean displacement per atom between MLP and DFT geometries,
    and flags structures where the MLP geometry was far from the DFT result
    (indicating the MLP relaxed to a different basin).
    """
    mlp_frames = read(prerelaxed_file, index=":", format="extxyz")
    dft_frames = read(dft_relaxed_file, index=":", format="extxyz")

    if len(mlp_frames) != len(dft_frames):
        print(f"WARNING: Frame count mismatch: {len(mlp_frames)} MLP vs "
              f"{len(dft_frames)} DFT. Check file ordering.")
        return

    displacements = []
    for mlp_a, dft_a in zip(mlp_frames, dft_frames):
        if len(mlp_a) != len(dft_a):
            displacements.append(np.nan)
            continue
        diff = mlp_a.get_positions() - dft_a.get_positions()
        displacements.append(np.linalg.norm(diff, axis=1).mean())

    displacements = np.array(displacements)
    print(f"\nMLP pre-relaxation vs DFT relaxation geometry comparison:")
    print(f"  Mean per-atom displacement: {np.nanmean(displacements):.3f} Å")
    print(f"  Median per-atom displacement: {np.nanmedian(displacements):.3f} Å")
    n_large = (displacements > 0.5).sum()
    if n_large > 0:
        print(f"  WARNING: {n_large} structures have MLP→DFT displacement > 0.5 Å/atom.")
        print(f"  These may have relaxed to different energy basins. "
              f"Inspect manually and consider longer DFT relaxation or manual geometry editing.")
```

### Workflow 5: MD pre-equilibration with a universal potential

Running NVT or NPT equilibration with MACE-MP before a system-specific MLP or DFT-MD run, to sample thermally accessible configurations as seeds for DFT labeling.

```python
import numpy as np
from pathlib import Path
from ase.io import read, write
from ase import units
from ase.md.langevin import Langevin
from ase.md.npt import NPT
from ase.io.trajectory import Trajectory
from mace.calculators import mace_mp


def preequilibrate_nvt(
    atoms_file: str,
    temperature_K: float,
    n_steps: int = 50_000,
    timestep_fs: float = 1.0,
    friction: float = 0.01,
    output_traj: str = "preequilibrated.traj",
    dump_interval: int = 100,
    model_size: str = "medium",
    device: str = "cpu",
) -> str:
    """
    NVT Langevin pre-equilibration using MACE-MP.

    For metallic systems: timestep_fs=2.0, friction=0.005.
    For molecular systems: timestep_fs=0.5–1.0, friction=0.02.
    n_steps=50,000 (50 ps at 1 fs/step) is typically sufficient to
    reach thermal equilibrium for a metallic crystal supercell.

    Returns path to the trajectory file.
    """
    atoms = read(atoms_file)
    calc = mace_mp(model=model_size, dispersion=False, default_dtype="float32", device=device)
    atoms.calc = calc

    # Assign initial velocities from Maxwell-Boltzmann distribution
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_K)

    dyn = Langevin(
        atoms,
        timestep=timestep_fs * units.fs,
        temperature_K=temperature_K,
        friction=friction / units.fs,
    )

    traj = Trajectory(output_traj, "w", atoms)
    dyn.attach(traj.write, interval=dump_interval)

    # Monitor temperature
    def print_md_info():
        e_kin = atoms.get_kinetic_energy()
        T = e_kin / (1.5 * len(atoms) * units.kB)
        step = dyn.get_number_of_steps()
        if step % 5000 == 0:
            print(f"  Step {step:6d}  T = {T:.0f} K  E_pot = {atoms.get_potential_energy():.3f} eV")

    dyn.attach(print_md_info, interval=100)

    print(f"Running NVT equilibration: {n_steps} steps at {temperature_K} K...")
    dyn.run(n_steps)
    traj.close()

    print(f"Trajectory written to {output_traj}")
    print(f"NOTE: This is a MACE-MP trajectory. Structures sampled from it should "
          f"be validated with DFT before use as training data. Universal-potential "
          f"geometries may contain subtle structural errors not visible in energy alone.")
    return output_traj


def sample_diverse_frames_from_traj(
    traj_file: str,
    n_frames: int = 50,
    skip_initial_fraction: float = 0.2,
    method: str = "fps",
    species: list[str] = None,
    output_file: str = "sampled_for_dft.extxyz",
) -> None:
    """
    Sample diverse frames from a pre-equilibration trajectory for DFT labeling.

    skip_initial_fraction: skip the first fraction of the trajectory
        (pre-equilibration phase). 0.2 → skip first 20% of frames.
    method: "fps" (farthest-point sampling in SOAP space) | "uniform"
        FPS is preferred for maximizing structural diversity.
    """
    traj = read(traj_file, index=":")
    n_skip = int(len(traj) * skip_initial_fraction)
    traj = traj[n_skip:]

    if method == "uniform":
        indices = np.linspace(0, len(traj) - 1, n_frames, dtype=int)
        selected = [traj[i] for i in indices]
    elif method == "fps":
        if species is None:
            # Collect all species from trajectory
            species = sorted(set(
                sym for atoms in traj for sym in atoms.get_chemical_symbols()
            ))
        from dscribe.descriptors import SOAP
        from sklearn.preprocessing import normalize

        soap = SOAP(
            species=species, r_cut=5.0, n_max=9, l_max=6,
            sigma=0.5, average="inner", periodic=True,
        )
        descs = normalize(soap.create(traj), norm="l2")  # (N_frames, D)

        # Greedy FPS: start from random frame, iteratively select most distant
        rng = np.random.default_rng(42)
        selected_idx = [int(rng.integers(len(traj)))]
        min_dists = np.full(len(traj), np.inf)

        for _ in range(n_frames - 1):
            last = descs[selected_idx[-1]]
            cos_sim = descs @ last
            dists = np.sqrt(np.maximum(0, 2 - 2 * np.clip(cos_sim, -1, 1)))
            min_dists = np.minimum(min_dists, dists)
            min_dists[selected_idx] = -np.inf  # Exclude already-selected
            selected_idx.append(int(np.argmax(min_dists)))

        selected = [traj[i] for i in sorted(selected_idx)]
        print(f"FPS selected {n_frames} diverse frames from {len(traj)} equilibrated frames")
    else:
        raise ValueError(f"Unknown sampling method '{method}'")

    write(output_file, selected, format="extxyz")
    print(f"Sampled frames written to {output_file}")
    print(f"Next: run DFT single-point calculations on these frames with "
          f"`dft-dataset-generation` to produce DFT-labeled extxyz for MACE training.")
```

### Workflow 6: Fine-tuning decision and system-specific accuracy assessment

Deciding whether to use a universal potential as-is, fine-tune it, or train from scratch, based on a systematic accuracy comparison.

```python
import numpy as np
import json
from ase.io import read


def fine_tuning_decision_report(
    benchmark_results: dict,
    system_description: str,
    target_application: str,
    target_force_mae_meV_A: float = 50.0,
    target_energy_mae_meV_atom: float = 5.0,
) -> str:
    """
    Produce a structured recommendation for whether to use the universal
    potential as-is, fine-tune it, or train from scratch.

    benchmark_results: output of benchmark_universal_potentials() — dict with
        per-model force MAE, energy MAE, and stress MAE.
    system_description: human-readable description of the target system
        (e.g., "W0.8Ta0.2 BCC alloy at 1500 K").
    target_application: e.g., "MD at 1500 K" | "structure screening" | "phase diagram"
    target_force_mae_meV_A: accuracy target for the application.
        MD at high T requires 30–100 meV/Å; structure screening 100–500 meV/Å;
        thermodynamic properties 10–50 meV/Å. [EXPERT REVIEW NEEDED]
    """
    best_model = None
    best_fmae = np.inf

    for key, res in benchmark_results.items():
        if "error" in res:
            continue
        if res["force_mae_meV_A"] < best_fmae:
            best_fmae = res["force_mae_meV_A"]
            best_model = res["model"]

    if best_model is None:
        return "ERROR: No models produced valid benchmarks."

    print(f"\n{'='*60}")
    print(f"FINE-TUNING DECISION REPORT")
    print(f"System:      {system_description}")
    print(f"Application: {target_application}")
    print(f"Target F MAE:{target_force_mae_meV_A:.0f} meV/Å")
    print(f"Best model:  {best_model}  (F MAE = {best_fmae:.0f} meV/Å)")
    print(f"{'='*60}")

    if best_fmae < target_force_mae_meV_A:
        recommendation = "USE_AS_IS"
        print(f"\nRECOMMENDATION: Use {best_model} without fine-tuning.")
        print(f"  Force MAE ({best_fmae:.0f} meV/Å) is below the target "
              f"({target_force_mae_meV_A:.0f} meV/Å).")
        print(f"  Validate with the full mlp-validation protocol before production use.")

    elif best_fmae < 5 * target_force_mae_meV_A:
        recommendation = "FINE_TUNE"
        print(f"\nRECOMMENDATION: Fine-tune {best_model} on system-specific DFT data.")
        print(f"  Force MAE ({best_fmae:.0f} meV/Å) is {best_fmae/target_force_mae_meV_A:.1f}× "
              f"above target but within 5× — fine-tuning is likely efficient.")
        print(f"  Suggested approach:")
        print(f"  1. Generate 200–500 DFT structures covering the target T/P/composition range")
        print(f"     using pre-equilibration trajectories from `universal-potentials`.")
        print(f"  2. Fine-tune with MACE `--foundation_model` flag and reduced LR (1e-3 → 1e-4).")
        print(f"  3. Validate fine-tuned model with mlp-validation before production.")
        print(f"  See `fine-tuning-mlp` skill for the complete workflow.")

    elif best_fmae < 20 * target_force_mae_meV_A:
        recommendation = "TRAIN_OR_FINE_TUNE"
        print(f"\nRECOMMENDATION: Fine-tune or train from scratch.")
        print(f"  Force MAE ({best_fmae:.0f} meV/Å) is {best_fmae/target_force_mae_meV_A:.1f}× "
              f"above target. Fine-tuning may converge slowly.")
        print(f"  If the universal potential shows correct force directions "
              f"(Pearson r > 0.7 between MLP and DFT forces), prefer fine-tuning.")
        print(f"  If force correlation is weak, start from scratch with MACE training.")

    else:
        recommendation = "TRAIN_FROM_SCRATCH"
        print(f"\nRECOMMENDATION: Train from scratch.")
        print(f"  Force MAE ({best_fmae:.0f} meV/Å) is > 20× above target. "
              f"The universal potential is not a useful starting point for this system.")
        print(f"  Typical cause: elements or bonding types absent from the training database.")
        print(f"  See `mace-training` skill for the from-scratch workflow.")

    return recommendation
```

## Common Failure Modes

| Failure | Symptom | Likely Cause | Diagnostic | Fix |
|---------|---------|--------------|------------|-----|
| Blind trust in a pretrained model | Universal potential used for a new system without validation; published results rely on MLP energies without DFT confirmation; systematic errors in phase stability or formation energies go undetected | Assumption that "trained on the Materials Project" means accurate for any inorganic crystal; no benchmarking step | Run `benchmark_universal_potentials` on 20–50 DFT-labeled structures for the target system; if none exist, run DFT on a small set specifically for validation before using the model in production | Always run Workflow 1 on a held-out DFT set for the target system before any downstream use; never skip this step |
| Unsupported or poorly represented elements | Universal potential predicts anomalously high formation energies or negative phonon frequencies for structures containing rare elements (e.g., Pa, Am, Tc) or unusual oxidation states; relaxations diverge or converge to unphysical geometries | Element absent from or severely underrepresented in the training database; the model's embedding for that element is trained on very few structures | Check the model's published element coverage map; run `check_universal_potential_domain` on structures containing suspect elements; compute the 90th-percentile force error for structures containing each element separately | Use a model with explicit coverage of the element (e.g., OMAT24-trained models have better coverage of actinides [EXPERT REVIEW NEEDED]); or train a system-specific MLP |
| Incorrect magnetic ground state | Universal potential (MACE-MP-0, M3GNet, ORB) predicts the wrong energy ordering between ferromagnetic, antiferromagnetic, and paramagnetic phases; formation energies of magnetic oxides are wrong by 0.5–2 eV/atom | Model trained without spin polarization; non-magnetic DFT reference fails to capture exchange-correlation energy of strongly magnetic systems | Compare MLP formation energies against spin-polarized DFT for a set of magnetic reference compounds (Fe, Co, Ni, FeO, Fe3O4, NiO); check if the model assigns the correct lattice constant for BCC Fe | Switch to CHGNet for magnetic systems; run spin-polarized DFT validation; do not use MACE-MP-0 formation energies for any system where magnetic ordering contributes to stability |
| Using universal potential energies as DFT labels | MACE committee trained on structures labeled by MACE-MP-0 instead of DFT; committee σ_F is low but the model predicts systematically wrong forces; all committee members agree on the wrong answer | Confusion between MLP energy predictions and DFT reference labels; occurs when "pre-relaxed" structures are wrongly flagged as "DFT-computed" in the dataset | Inspect the config_type and provenance fields in the extxyz dataset; check if structures labeled as DFT-computed have energy values consistent with the DFT code's output format | Always compute DFT labels by running VASP/QE on the prerelaxed geometries; never copy MLP energies into the DFT label fields; see `dft-dataset-generation` for the correct labeling pipeline |
| Spurious phase stability from training-set overlap | Universal potential correctly predicts formation energies for a set of structures because those exact structures are in the Materials Project training set; test set error appears low but generalization to modified compositions is poor | Validation dataset was drawn from the MP training set without removing training-set entries; model is interpolating, not generalizing | Check structure IDs against the MP training set (using `pymatgen.ext.matproj.MPRester` structure matching or composition fingerprinting); compute errors separately for MP entries vs. non-MP structures | Always test on structures not in the training database; use prototyped or slightly distorted structures as test cases; report errors separately for in-distribution and out-of-distribution test sets |
| Unstable MD despite low validation error | Production MD trajectory at target temperature diverges (atoms explosively leave the box, energy spikes); but static force validation showed acceptable MAE | Low static force MAE does not guarantee MD stability; the model may produce unstable forces for configurations that arise during MD but are absent from the test set; high-frequency vibrations not captured by low-timestep sampling | Run a short 1 ps NVE MD trajectory; compute energy drift per ps; run `verify_committee_diversity` if a committee is available; check if high-σ_F frames appear in early MD | If using a universal potential for MD, start with a short NVE stability test (Workflow 5 in `mlp-validation`); use a shorter timestep initially; consider fine-tuning on MD-sampled configurations at the target temperature |
| Wrong surface or defect energetics | Surface formation energy or defect formation energy from universal potential disagrees with DFT by 0.5–2 J/m² or 1–3 eV respectively; universal potential over- or underestimates the energy cost of creating a surface/defect | Surface and defect structures are rare in MP-based training sets; the model has limited exposure to broken-symmetry, under-coordinated environments; vacuum and defect supercell configurations are OOD | Run `check_universal_potential_domain` comparing surface/defect slabs against the reference training distribution; compute SOAP dissimilarity for surface atoms vs. bulk atoms | Collect targeted DFT calculations for surface or defect structures before trusting universal potential predictions; fine-tune on explicitly surface-containing structures using the `fine-tuning-mlp` workflow |
| Organic or molecular system treated with an inorganic universal potential | MACE-MP-0 or CHGNet applied to an organic molecule, polymer, or molecular crystal; predicted bond lengths wrong by 5–15%; dihedral barriers and intramolecular potentials unreliable | MACE-MP-0 is trained on periodic inorganic crystals at PBE level; organic bonding patterns and the importance of dispersion are not represented in the training data | Run MACE-OFF23 on the same system for comparison; check if molecule-level properties (bond lengths, vibrational modes) match quantum chemistry reference | Use MACE-OFF23 for organic systems; use a dispersion-corrected model if long-range van der Waals interactions are important (molecular crystals, adsorption); for organometallic systems, verify which universal potential covers the specific bonding type |

## Best Practices

- **Always run a DFT validation set before trusting a universal potential for any new system.** There is no substitute for comparing predictions to the ground truth for the specific chemistry and structural regime of interest. A 20-structure DFT validation set takes hours and prevents days of wasted downstream work.
- **Record the exact model name, version, release date, and SHA-256 of the model weights in every result.** Universal potential weights change between versions, and models with the same name (e.g., "MACE-MP-0 medium") can produce different results after a minor release. This is essential for reproducibility. Use the pattern from `reproducibility-workflow` to register the model checkpoint.
- **Use MACE-MP-0 for inorganic crystals, MACE-OFF23 for organic molecules, and CHGNet for magnetic systems as starting defaults.** These assignments reflect the training data coverage of each model. Cross-validate any such assignment by checking the force MAE on a representative test set.
- **Do not use a universal potential's internal energies as absolute formation energies without checking the reference state convention.** MACE-MP-0 energies reference PBE+U isolated atoms; computing formation energies against experimental or different-functional references requires careful reference energy handling.
- **Run the OOD detector (Workflow 3) for any system containing elements or structural types that appear rarely in the Materials Project.** OOD detection via SOAP dissimilarity is inexpensive and catches the cases most likely to produce catastrophically wrong predictions.
- **For MD, always run a short NVE stability test before production runs.** A universal potential that passes static force validation can still produce unstable MD for thermally accessible configurations outside the training distribution. A 5 ps NVE test at the target temperature costs minutes and can prevent a diverged trajectory.
- **Use `fmax=0.10 eV/Å` for pre-relaxation, not `0.01 eV/Å`.** Over-converging the MLP pre-relaxation wastes compute. The DFT step will complete the geometry optimization; the MLP only needs to remove the worst geometrical problems.
- **Do not skip the `validate_prerelaxed_vs_dft` check (Workflow 4) for a new composition.** A universal potential can relax into a metastable basin that is adjacent to the correct DFT minimum but different enough that DFT ionic relaxation is slow or converges to the wrong structure. For new compositions, verify that MLP and DFT relaxed geometries agree within 0.2–0.3 Å/atom.
- **For high-throughput screening, report results as "MACE-MP predictions" not "DFT predictions" or "calculated energies".** The distinction matters for reproducibility and scientific integrity. Universal potential predictions should always be labeled with their source model in any figure, table, or publication.
- **Prefer the largest available model size for accuracy-critical applications; use small models for exploratory screening.** MACE-MP-0 medium/large are typically 2–5× more accurate than small but 5–10× slower. The choice depends on the number of structures and the tolerable error.

## Integration with Other Skills

- **[mace-training](../mace-training/SKILL.md):** If the universal potential benchmark (Workflow 1 here) shows force MAE > 5× the target accuracy and force correlation with DFT is poor, system-specific MACE training is indicated. That skill covers the full training pipeline from DFT dataset preparation to validated deployment.
- **[fine-tuning-mlp](../fine-tuning-mlp/SKILL.md):** When force MAE is 2–5× above target (fine-tuning regime, Workflow 6 here), fine-tuning the universal potential on system-specific DFT data is more efficient than training from scratch. That skill covers the `--foundation_model` flag, learning rate schedules for fine-tuning, and when to unfreeze different layers.
- **[uncertainty-mlp](../uncertainty-mlp/SKILL.md):** Committee UQ applied to fine-tuned universal potentials requires special care: shared base representations across committee members reduce committee diversity. The SOAP dissimilarity OOD detector from that skill is the recommended complement to committee σ_F for universal-potential deployment. That skill also warns against conflating fine-tuned-model committee variance with true epistemic uncertainty.
- **[mlp-validation](../mlp-validation/SKILL.md):** The full validation trinity (static test-set errors, physical property reproduction, MD stability) from that skill must be run before deploying any universal potential — or fine-tuned model — in production. The static force/energy validation in Workflow 1 here is a prerequisite screening step, not a substitute for full validation.
- **[mlp-to-lammps](../mlp-to-lammps/SKILL.md):** MACE-MP-0 and SevenNet have documented LAMMPS interfaces (`pair_style mace` and `pair_style sevennet`). That skill covers the LAMMPS deployment of MLP models, including unit verification and force validation. Universal potentials deployed in LAMMPS require the same species-mapping and unit-verification checks as system-specific models.
- **[active-learning-mlp](../active-learning-mlp/SKILL.md):** Universal potentials are the natural starting point for the first AL iteration: pre-equilibrate with a universal potential to generate structurally diverse candidate frames, run DFT on the most uncertain ones, and train a system-specific model. The committee σ_F computed in that skill uses the system-specific model; the universal potential serves only for initial exploration.
- **[dft-dataset-generation](../../electronic-structure/dft-dataset-generation/SKILL.md):** Pre-relaxed structures from Workflow 4 here feed directly into the DFT labeling pipeline in that skill. The provenance metadata written by `prerelax_for_dft` (model name, version, convergence status) should be preserved in the extxyz dataset for reproducibility.
- **[dft-convergence](../../electronic-structure/dft-convergence/SKILL.md):** Universal potential pre-relaxation reduces DFT ionic steps but does not change the DFT convergence settings required. The ENCUT, EDIFF, k-mesh, and smearing settings from that skill must still be applied to all DFT calculations on pre-relaxed structures.
- **[vasp-workflow](../../electronic-structure/vasp-workflow/SKILL.md):** VASP relaxations of structures pre-relaxed with universal potentials typically converge in fewer ionic steps (NSW=20–50 instead of 100+). The INCAR settings for spin initialization (ISPIN, MAGMOM) must still be set manually — the universal potential does not provide initial magnetic moments (except CHGNet).
- **[quantum-espresso](../../electronic-structure/quantum-espresso/SKILL.md):** QE relaxations of pre-relaxed structures follow the same logic as VASP. The `system` block parameters (`ecutwfc`, `ecutrho`, `occupations`, `smearing`) are unchanged by the pre-relaxation step.
- **[lammps-workflow](../../atomistic-md/lammps-workflow/SKILL.md):** Pre-equilibrated structures from Workflow 5 here can be used as starting configurations for LAMMPS production runs with system-specific MLPs. The extxyz trajectory from MACE-MP pre-equilibration can be converted to LAMMPS dump format for restart.
- **[md-equilibration](../../atomistic-md/md-equilibration/SKILL.md):** Universal potential pre-equilibration (Workflow 5) is a cheap substitute for the full equilibration protocol in that skill only for generating starting configurations. For production MD accuracy validation, the NVE conservation test and ensemble analysis in that skill must be run on the final system-specific model.
- **[materials-featurization](../../scientific-data/materials-featurization/SKILL.md):** SOAP descriptors used for OOD detection in Workflow 3 here use the same parameters as the featurization pipeline in that skill. Keeping the SOAP parameters consistent (same r_cut, n_max, l_max, sigma, species list) ensures that distances computed here are comparable to those used in downstream ML models.
- **[reproducibility-workflow](../../scientific-data/reproducibility-workflow/SKILL.md):** Model version, checkpoint SHA-256, and benchmark results from Workflow 1 here are provenance records that belong in the campaign settings JSON described in that skill. Any result that uses a universal potential prediction should log the model identifier and the benchmark outcome alongside the result.

## Key References

1. Batatia, I., et al. (2023). *A foundation model for atomistic materials chemistry.* arXiv:2401.00096. https://arxiv.org/abs/2401.00096
2. Deng, B., et al. (2023). *CHGNet as a pretrained universal neural network potential for charge-informed atomistic modelling.* Nature Machine Intelligence, 5, 1031–1041. https://doi.org/10.1038/s42256-023-00716-3
3. Chen, C., & Ong, S. P. (2022). *A universal graph deep learning interatomic potential for the periodic table.* Nature Computational Science, 2, 718–728. https://doi.org/10.1038/s43588-022-00349-3
4. Kovács, D. P., et al. (2023). *MACE-OFF23: Transferable machine learning force fields for organic molecules.* arXiv:2312.15211. https://arxiv.org/abs/2312.15211
5. Merchant, A., et al. (2023). *Scaling deep learning for materials discovery.* Nature, 624, 80–85. https://doi.org/10.1038/s41586-023-06735-9
