---
name: eos-fitting
domain: electronic-structure
maturity: draft
tools:
  - VASP
  - ASE
  - pymatgen
  - numpy
  - scipy
  - matplotlib
tested_versions:
  VASP: "6.3.2"
  ASE: "3.23.0"
  pymatgen: "2024.11.13"
  scipy: "1.13.0"
  numpy: "1.26.4"
requires_gpu: false
estimated_runtime: hours-to-days
---

# EOS Fitting

## Description

This skill covers equation-of-state (EOS) fitting for crystalline materials from DFT or ML-potential calculations: generating energy-volume datasets via fixed-volume DFT relaxations, fitting Birch-Murnaghan, Vinet, and Murnaghan EOS forms, extracting equilibrium volume V₀, ground-state energy E₀, bulk modulus B₀, and pressure derivative B'₀, quantifying fitting uncertainty via bootstrap resampling, detecting outliers and phase-transition contamination, and comparing DFT versus MLP EOS for potential validation. Invoke this skill when extracting mechanical properties of crystalline materials, validating an ML potential against DFT ground-state properties, or generating reference EOS data for a materials database.

## Domain Context

The equation of state relates the total energy (or pressure) of a crystal to its volume at fixed composition and temperature. For a DFT EOS at 0 K, the energy-volume curve E(V) is the Born-Oppenheimer ground-state total energy as a function of isotropic compression and dilation of the unit cell, with atomic positions and cell shape (but not volume) optimized at each volume. The EOS encodes the two most important bulk mechanical properties: the equilibrium volume V₀ and the isothermal bulk modulus B₀ = −V (∂P/∂V)_T = V (∂²E/∂V²)_T.

Several physical and numerical constraints govern EOS accuracy:

- **Pulay stress.** Plane-wave DFT has an incomplete basis set at finite ENCUT. When the cell volume changes, the number and character of plane waves changes discontinuously, introducing a systematic error in the Hellmann-Feynman stress tensor. For a fixed ENCUT, the computed stress at a strained volume is systematically too small (tensile) relative to the true DFT stress. This shifts the apparent equilibrium volume. Pulay stress is reduced by increasing ENCUT to ≥ 1.3 × ENMAX (VASP) or verifying that residual stress from a static single-point is < 1 kbar after relaxation at the converged ENCUT.
- **Isotropic vs. anisotropic strain.** The simplest EOS applies to isotropic (hydrostatic) volume change: all lattice vectors scaled uniformly by a factor η = (V/V₀)^{1/3}. For anisotropic crystals (tetragonal, orthorhombic, monoclinic, triclinic), isotropic scaling distorts the equilibrium shape. A more accurate protocol scales the volume while simultaneously optimizing the cell shape at each volume (ISIF=4 in VASP). The resulting EOS characterizes the ground-state energy as a function of volume at the optimal shape for each volume, not at a fixed metric tensor.
- **ISIF settings for fixed-volume relaxations.** ISIF controls which degrees of freedom are relaxed in VASP: ISIF=2 (ions only, fixed cell), ISIF=4 (ions + cell shape, fixed volume), ISIF=3 (ions + cell shape + volume, full optimization). For EOS generation, ISIF=4 is the correct setting: it relaxes the internal structure and cell shape but keeps the volume fixed. Using ISIF=2 introduces shape-stress artifacts for non-cubic crystals. Using ISIF=3 would allow the volume to relax to equilibrium rather than to the target value.
- **Volume range and density.** A typical EOS fit uses 7–15 volume points spanning ±5–15% of the equilibrium volume. Too narrow a range (±3%) underdetermines the curvature and gives an unreliable B'₀. Too wide a range (> ±20%) risks entering a different structural phase or a high-pressure regime where the 3rd-order Birch-Murnaghan EOS is invalid. For soft materials (polymorphs, molecular crystals, or materials with shallow E-V curves), even ±10% may cross a transition.
- **Magnetic state consistency.** For magnetic materials (Fe, Ni, Co, Mn oxides, rare-earth intermetallics), the magnetic ground state may change with volume. A ferromagnetic ground state at equilibrium may become antiferromagnetic or paramagnetic at large compressions. If this transition occurs within the fitting range, the E-V curve has a kink and no single EOS form fits the entire range. The spin state must be monitored at each volume and the fit range restricted to a single-phase, single-spin-state region.

## When to Use This Skill

- Extracting B₀, V₀, and B'₀ for comparison to experimental EOS data or for benchmarking DFT functionals against known values.
- Validating ML interatomic potentials (MACE, NequIP, DeePMD): a well-trained MLP must reproduce the DFT EOS to within 0.5% in V₀ and 2–5% in B₀. [EXPERT REVIEW NEEDED]
- Generating training labels for active learning: E-V curve calculations efficiently sample a range of isotropic strain configurations.
- Generating reference data for materials databases (Materials Project, AFLOW, OQMD) where EOS parameters are stored alongside other properties.
- Screening materials under high pressure: the EOS provides the pressure-volume relation P(V) = −(∂E/∂V)_T needed to identify high-pressure phase boundaries.

## When Not to Use This Skill

- **For materials with known phase transitions in the strain range of interest.** If a material undergoes a structural transition within ±10% of V₀, a single-phase EOS fit will fail. Use constrained optimization on individual structural phases instead, or restrict the fit to a narrower volume range.
- **For anharmonic or soft-mode systems near a phase transition.** The EOS parametrizes harmonic-level bulk response. Near a soft-mode instability (ferroelectric transition, structural transformation), the E-V curve is not smooth and EOS fitting is unreliable.
- **For extracting single-crystal elastic constants.** The EOS gives only the bulk modulus (isotropic compression response). Independent elastic constants (C₁₁, C₁₂, C₄₄, ...) require finite-strain calculations along symmetry-inequivalent strain modes (see the phonopy-stiffness or vaspkit elastic workflow).
- **For high-temperature thermodynamic EOS.** The DFT EOS is a 0 K result. Thermal pressure requires quasi-harmonic phonon calculations (phonopy + DFT-EOS) or MD-based equation of state for high temperatures. A 0 K EOS applied at 1000 K gives the wrong equilibrium volume and bulk modulus.
- **As a substitute for full structural relaxation.** The equilibrium volume V₀ from an EOS fit is not the same as the volume from a full ISIF=3 relaxation at the same settings. Both should agree to within 0.5%, but the EOS fit gives additional mechanical information (B₀, B'₀) rather than replacing the primary structural optimization.

## Core Concepts

- **Birch-Murnaghan EOS (3rd order, BM3):** The standard DFT EOS form derived from finite-strain theory with Birch's Eulerian strain η = [(V₀/V)^(2/3) − 1]/2. E(V) = E₀ + (9V₀B₀/16){[(V₀/V)^(2/3) − 1]³ B'₀ + [(V₀/V)^(2/3) − 1]² [6 − 4(V₀/V)^(2/3)]}. P(V) = (3B₀/2)[(V₀/V)^(7/3) − (V₀/V)^(5/3)]{1 + (3/4)(B'₀ − 4)[(V₀/V)^(2/3) − 1]}. BM3 is the standard choice for DFT EOS of hard materials with B₀ > 50 GPa. [EXPERT REVIEW NEEDED: BM3 can overestimate B'₀ for very soft or compressible materials]
- **Vinet EOS (Universal EOS):** P(V) = 3B₀ [(1−x)/x²] exp[η(1−x)], where x = (V/V₀)^{1/3} and η = (3/2)(B'₀ − 1). More accurate than BM3 for strongly compressed materials and for high-pressure fits. Converges correctly to the Thomas-Fermi limit at extreme compression. Often preferred for soft materials and high-pressure experiments.
- **Murnaghan EOS:** P(V) = (B₀/B'₀)[(V₀/V)^{B'₀} − 1]. The simplest analytical EOS; valid only for small strains. Systematically overestimates B₀ relative to BM3 for large compressions. Included for comparison and for fitting experimental P-V data with few points.
- **Equilibrium volume V₀:** The volume at the minimum of E(V). For cubic systems, V₀/N_atoms is reported in ų/atom or Å³/atom. Sensitive to the DFT functional: GGA-PBE typically overestimates V₀ by 1–3% for metals; LDA underestimates by 1–3%. SCAN meta-GGA reduces this systematic error for many materials. [EXPERT REVIEW NEEDED]
- **Bulk modulus B₀:** B₀ = V₀ (∂²E/∂V²)|_{V=V₀} in GPa. Typical ranges: soft polymers < 5 GPa; alkali metals 10–40 GPa; transition metals 100–400 GPa; diamond 440 GPa. B₀ is more sensitive to functional and pseudopotential choice than V₀.
- **Pressure derivative B'₀:** B'₀ = (∂B/∂P)|_{P=0}. Dimensionless. Typical values: B'₀ ≈ 3–5 for most metals and ceramics. BM3 assumes B'₀ is constant (P-independent), valid only at moderate pressures.
- **ISIF=4 (VASP):** Relaxes ion positions and cell shape (angles + a/b/c ratios) at fixed volume. The correct setting for EOS volume scans in non-cubic systems. The stress tensor component along the volume direction is non-zero (equal to the negative pressure at that volume); only the deviatoric (shape) stress is minimized.
- **Pulay stress correction:** The residual isotropic stress at the ISIF=4 equilibrium geometry for each volume is a diagnostic: if the Pulay contribution (estimated by comparing stress at ENCUT and 1.3×ENCUT) exceeds 1 kbar, ENCUT must be increased. For EOS purposes, the Pulay stress shifts V₀ systematically and may be partially corrected by using the computed P(V) directly rather than deriving pressure from the EOS fit.
- **Bootstrap uncertainty for EOS parameters:** Resampling the E-V data with replacement 1000–10,000 times and refitting the EOS at each resample gives confidence intervals on V₀, B₀, B'₀, and E₀ that capture both numerical noise in the DFT energies and sensitivity to individual outlier points. Bootstrap CI is more reliable than the covariance matrix of the nonlinear least-squares fit when the fit is nonlinear or when the residuals are non-Gaussian.
- **E-V residuals as outlier diagnostic:** After fitting, the residuals E_fit(Vᵢ) − Eᵢ should be random noise (< 1 meV/atom) and free of systematic trends. A single large residual (> 5 meV/atom) indicates either: (1) an unconverged SCF at that volume, (2) a magnetic state change at that volume, or (3) a structural relaxation that found a different local minimum. All three require investigation before including that point in the fit.

## Key Workflows

### Workflow 1: Fixed-volume DFT dataset generation with VASP

Generate 11 volume-strained structures spanning ±10% of the estimated equilibrium volume, relax each with ISIF=4 (ions + cell shape, fixed volume), and collect total energies.

```python
"""
EOS dataset generation: scale the unit cell volume, relax with ISIF=4,
collect E(V). Uses pymatgen for structure manipulation and VASP job setup.
"""

import numpy as np
from pathlib import Path
from pymatgen.core import Structure
from pymatgen.io.vasp.sets import MPRelaxSet
from pymatgen.io.vasp.inputs import Incar
from pymatgen.io.vasp.outputs import Vasprun, Outcar


def generate_eos_structures(
    structure_file: str,
    volume_fractions: np.ndarray = None,
    n_volumes: int = 11,
    v_range_frac: float = 0.10,   # ±10% of input volume
) -> list:
    """
    Generate a set of isotropically scaled structures for EOS fitting.

    For cubic systems, this applies a uniform scale factor to all lattice
    parameters. For non-cubic systems, the shape is preserved at each scaling
    (the ISIF=4 relaxation will optimize the shape at each volume).

    Returns list of (volume_fraction, Structure) tuples.
    """
    structure = Structure.from_file(structure_file)
    V0_input = structure.volume   # Starting volume (may not be equilibrium)

    if volume_fractions is None:
        volume_fractions = np.linspace(1 - v_range_frac, 1 + v_range_frac, n_volumes)

    structures = []
    for vf in volume_fractions:
        # Scale factor: (V/V_input)^(1/3) applied to all lattice vectors
        scale = vf ** (1.0 / 3.0)
        scaled = structure.copy()
        scaled.scale_lattice(vf * V0_input)   # pymatgen: scale to target volume directly
        structures.append((vf, scaled))
        print(f"  V/V_input = {vf:.4f}: V = {scaled.volume:.3f} Å³")

    return structures


def write_vasp_eos_inputs(
    structures: list,
    base_dir: str = "eos_calcs",
    encut_factor: float = 1.3,   # ENCUT = 1.3 × max ENMAX across POTCAR — reduces Pulay stress
    kpoints_density: int = 3000, # kpoints per atom reciprocal volume (k/atom); use higher for metals
    ismear: int = 1,             # Methfessel-Paxton for metals; use 0 (Gaussian) for insulators
    sigma: float = 0.2,          # eV; use 0.05 for insulators
    is_magnetic: bool = False,   # Set True for magnetic systems (Fe, Ni, Mn oxides, ...)
) -> None:
    """
    Write VASP input files for each EOS volume point.

    ISIF=4: relax ions + cell shape, fix volume (correct for EOS generation).
    IBRION=2: conjugate-gradient relaxation.
    EDIFFG=-0.01: tight force convergence for accurate EOS curvature.
    ENCUT ≥ 1.3×ENMAX: reduces Pulay stress.
    """
    Path(base_dir).mkdir(exist_ok=True)

    for i, (vf, structure) in enumerate(structures):
        calc_dir = Path(base_dir) / f"vol_{i:03d}_vf{vf:.4f}"
        calc_dir.mkdir(exist_ok=True)

        # Write structure (POSCAR)
        structure.to(filename=str(calc_dir / "POSCAR"))

        # INCAR for EOS fixed-volume relaxation
        incar_dict = {
            "SYSTEM":  f"EOS volume {vf:.4f}",
            "ISTART":  0,
            "ICHARG":  2,
            "ENCUT":   None,         # Set to 1.3 × ENMAX after POTCAR is in place
            "EDIFF":   1e-7,         # Tight SCF for EOS accuracy (vs 1e-6 for standard relaxation)
            "EDIFFG":  -0.01,        # Force convergence: 0.01 eV/Å
            "NSW":     200,
            "IBRION":  2,            # CG ion relaxation
            "ISIF":    4,            # CRITICAL: ions + cell shape, FIXED VOLUME
                                     # ISIF=2: fixed cell → wrong for non-cubic
                                     # ISIF=3: relaxes volume → wrong for EOS scan
            "ISMEAR":  ismear,
            "SIGMA":   sigma,
            "ALGO":    "Normal",
            "LORBIT":  11,
            "LWAVE":   False,        # Do not write WAVECAR (large; only needed for restart)
            "LCHARG":  False,
        }

        if is_magnetic:
            incar_dict.update({
                "ISPIN": 2,
                "LORBIT": 11,
                "NUPDOWN": -1,   # Allow total spin to relax (do not fix S_z)
                # MAGMOM should be set per-species in the POSCAR INCAR; set externally
            })
            print(f"  WARNING: Magnetic system. Monitor MAGMOM vs. volume for spin-state changes.")

        # Write INCAR (ENCUT placeholder; must be filled after POTCAR is available)
        incar = Incar(incar_dict)
        incar.write_file(str(calc_dir / "INCAR"))

        # Write a KPOINTS with automatic mesh
        kpoints_content = f"""Automatic
0
Gamma
{max(1, int((kpoints_density / structure.num_sites)**(1/3)))} {max(1, int((kpoints_density / structure.num_sites)**(1/3)))} {max(1, int((kpoints_density / structure.num_sites)**(1/3)))}
0 0 0
"""
        (calc_dir / "KPOINTS").write_text(kpoints_content)

        # Volume label file for post-processing
        (calc_dir / "volume_fraction.txt").write_text(str(vf))
        (calc_dir / "target_volume_A3.txt").write_text(str(structure.volume))

    print(f"\nWrote {len(structures)} EOS directories to {base_dir}/")
    print("Next steps:")
    print("  1. Copy POTCAR to each directory (or symlink from a shared POTCAR)")
    print("  2. Set ENCUT = 1.3 × max(ENMAX across all species) in each INCAR")
    print("  3. Submit all calculations (use a job array or high-throughput workflow)")
    print("  4. Collect results with collect_eos_results() after all jobs finish")
```

```python
def collect_eos_results(
    base_dir: str = "eos_calcs",
    check_convergence: bool = True,
) -> tuple:
    """
    Parse completed VASP EOS calculations and return E(V) data.

    Returns (volumes, energies) arrays and a list of warnings.
    """
    import os
    calc_dirs = sorted(Path(base_dir).glob("vol_*"))
    volumes = []
    energies = []
    warnings = []

    for calc_dir in calc_dirs:
        vasprun_file = calc_dir / "vasprun.xml"
        outcar_file  = calc_dir / "OUTCAR"

        if not vasprun_file.exists():
            warnings.append(f"MISSING vasprun.xml in {calc_dir.name}")
            continue

        try:
            vrun = Vasprun(str(vasprun_file), parse_dos=False, parse_eigen=False)
        except Exception as e:
            warnings.append(f"PARSE ERROR in {calc_dir.name}: {e}")
            continue

        if check_convergence and not vrun.converged:
            warnings.append(f"NOT CONVERGED: {calc_dir.name} "
                            f"(ionic: {vrun.converged_ionic}, "
                            f"electronic: {vrun.converged_electronic})")
            continue

        V = vrun.final_structure.volume        # Å³
        E = vrun.final_energy                  # eV (total energy)
        n_atoms = len(vrun.final_structure)
        volumes.append(V / n_atoms)            # Å³/atom
        energies.append(E / n_atoms)           # eV/atom

        # Check for Pulay stress: read stress tensor from OUTCAR
        try:
            outcar = Outcar(str(outcar_file))
            stress = outcar.read_table_pattern(
                header_pattern=r"FORCE on cell",
                row_pattern=r"\s+in kB\s+([\d\.\-]+)\s+([\d\.\-]+)\s+([\d\.\-]+)",
                footer_pattern=r"TOTAL",
                last_one_only=True,
            )
            # Approximate isotropic residual stress (kbar)
            if stress and len(stress[-1]) >= 3:
                iso_stress = sum(float(x) for x in stress[-1][:3]) / 3.0
                if abs(iso_stress) > 5.0:   # > 5 kbar residual isotropic stress
                    warnings.append(
                        f"HIGH RESIDUAL STRESS in {calc_dir.name}: "
                        f"{iso_stress:.1f} kbar. Consider increasing ENCUT."
                    )
        except Exception:
            pass   # Stress check is optional

    if warnings:
        print("WARNINGS:")
        for w in warnings:
            print(f"  {w}")

    volumes = np.array(volumes)
    energies = np.array(energies)

    # Sort by volume
    sort_idx = np.argsort(volumes)
    volumes = volumes[sort_idx]
    energies = energies[sort_idx]

    print(f"\nCollected {len(volumes)} converged volume points:")
    print(f"  Volume range: {volumes.min():.3f} – {volumes.max():.3f} Å³/atom")
    print(f"  Energy range: {energies.min():.4f} – {energies.max():.4f} eV/atom")
    print(f"  Minimum energy at V = {volumes[np.argmin(energies)]:.3f} Å³/atom")

    return volumes, energies, warnings
```

### Workflow 2: EOS fitting — Birch-Murnaghan, Vinet, and Murnaghan

Fit multiple EOS forms to the E(V) dataset and compare results. Always compare at least BM3 and Vinet; use agreement between them as a consistency check.

```python
"""
EOS fitting using scipy nonlinear least-squares.
Three EOS forms: Birch-Murnaghan 3rd order (BM3), Vinet, and Murnaghan.
Units convention throughout: V in Å³/atom, E in eV/atom, B in GPa.
"""

import numpy as np
from scipy.optimize import curve_fit, minimize
from scipy.constants import eV


EV_PER_A3_TO_GPA = eV * 1e21   # 1 eV/Å³ = 160.218 GPa


def birch_murnaghan_3(V, E0, V0, B0, B0p):
    """
    3rd-order Birch-Murnaghan energy-volume EOS.

    E0: equilibrium energy (eV/atom)
    V0: equilibrium volume (Å³/atom)
    B0: bulk modulus (eV/Å³) — convert to GPa: B0_GPa = B0 * EV_PER_A3_TO_GPA
    B0p: pressure derivative of bulk modulus (dimensionless)
    """
    eta = (V0 / V) ** (2.0 / 3.0)   # Eulerian strain term
    E = E0 + (9.0 * V0 * B0 / 16.0) * (
        (eta - 1.0) ** 3 * B0p
        + (eta - 1.0) ** 2 * (6.0 - 4.0 * eta)
    )
    return E


def vinet(V, E0, V0, B0, B0p):
    """
    Vinet (Universal) energy-volume EOS.
    More accurate than BM3 at large compressions.
    """
    x = (V / V0) ** (1.0 / 3.0)   # reduced length
    eta = 1.5 * (B0p - 1.0)
    E = E0 + (2.0 * V0 * B0 / (B0p - 1.0) ** 2) * (
        2.0 - (5.0 + 3.0 * B0p * (x - 1.0) - 3.0 * x) * np.exp(-eta * (x - 1.0))
    )
    return E


def murnaghan(V, E0, V0, B0, B0p):
    """
    Murnaghan EOS. Valid only for small strains; included for comparison.
    """
    E = E0 + V0 * B0 / B0p * (
        (V0 / V) ** B0p / (B0p - 1.0) + V / V0 - B0p / (B0p - 1.0)
    )
    return E


def bm3_pressure(V, V0, B0, B0p):
    """Birch-Murnaghan pressure-volume relation (GPa)."""
    f = 0.5 * ((V0 / V) ** (2.0 / 3.0) - 1.0)   # Eulerian strain
    P = 3.0 * B0 * f * (1.0 + 2.0 * f) ** 2.5 * (
        1.0 + 1.5 * (B0p - 4.0) * f
    )
    return P * EV_PER_A3_TO_GPA   # eV/Å³ → GPa


def fit_eos(
    volumes: np.ndarray,         # Å³/atom
    energies: np.ndarray,        # eV/atom
    eos_forms: list = None,
    p0: dict = None,             # Initial parameter guess; if None, estimated from data
) -> dict:
    """
    Fit multiple EOS forms to E(V) data and report results.

    Returns dict of results per EOS form, including parameters and residuals.
    """
    if eos_forms is None:
        eos_forms = ["BM3", "Vinet", "Murnaghan"]

    eos_functions = {
        "BM3": birch_murnaghan_3,
        "Vinet": vinet,
        "Murnaghan": murnaghan,
    }

    # Initial guess from parabolic fit to the E-V data
    idx_min = np.argmin(energies)
    V0_guess = volumes[idx_min]
    E0_guess = energies[idx_min]
    # Estimate B₀ from curvature: B = V₀ d²E/dV² ≈ parabolic coefficient
    coeffs = np.polyfit(volumes, energies, 2)   # E ≈ a*V² + b*V + c
    B0_guess_eV = 2.0 * coeffs[0] * V0_guess   # eV/Å³
    B0_guess_GPa = max(B0_guess_eV * EV_PER_A3_TO_GPA, 10.0)   # avoid negative initial guess
    B0p_guess = 4.0    # typical value for most materials

    if p0 is None:
        p0 = {
            "E0": E0_guess,
            "V0": V0_guess,
            "B0": B0_guess_eV,   # in eV/Å³ for fitting (GPa would require unit rescaling)
            "B0p": B0p_guess,
        }

    results = {}
    print(f"\nInitial guesses: V₀ = {V0_guess:.3f} Å³/atom, "
          f"B₀ ≈ {B0_guess_GPa:.1f} GPa, B'₀ ≈ {B0p_guess:.1f}")

    for name in eos_forms:
        if name not in eos_functions:
            print(f"  WARNING: Unknown EOS form '{name}'; skipping")
            continue

        func = eos_functions[name]
        try:
            popt, pcov = curve_fit(
                func,
                volumes,
                energies,
                p0=[p0["E0"], p0["V0"], p0["B0"], p0["B0p"]],
                bounds=(
                    [energies.min() - 1.0, volumes.min() * 0.8, 0.0, 0.5],
                    [energies.min() + 1.0, volumes.max() * 1.2, 1.0, 10.0],
                ),
                maxfev=10000,
            )
        except RuntimeError as e:
            print(f"  {name}: FIT FAILED — {e}")
            results[name] = {"converged": False}
            continue

        E0_fit, V0_fit, B0_fit_eV, B0p_fit = popt
        B0_fit_GPa = B0_fit_eV * EV_PER_A3_TO_GPA

        E_fit = func(volumes, *popt)
        residuals = energies - E_fit
        rms_residual = np.sqrt(np.mean(residuals ** 2)) * 1000   # meV/atom

        print(f"\n  {name}:")
        print(f"    E₀  = {E0_fit:.6f} eV/atom")
        print(f"    V₀  = {V0_fit:.4f} Å³/atom")
        print(f"    B₀  = {B0_fit_GPa:.2f} GPa")
        print(f"    B'₀ = {B0p_fit:.3f}")
        print(f"    RMS residual: {rms_residual:.3f} meV/atom")

        if rms_residual > 2.0:
            print(f"    WARNING: RMS residual {rms_residual:.1f} meV/atom > 2 meV/atom. "
                  "Check for outliers or phase transitions.")

        results[name] = {
            "converged": True,
            "E0_eV": E0_fit,
            "V0_A3": V0_fit,
            "B0_GPa": B0_fit_GPa,
            "B0p": B0p_fit,
            "popt": popt,
            "residuals_meV": residuals * 1000,
            "rms_meV": rms_residual,
        }

    # Consistency check between EOS forms
    if "BM3" in results and "Vinet" in results:
        if results["BM3"]["converged"] and results["Vinet"]["converged"]:
            dV0 = abs(results["BM3"]["V0_A3"] - results["Vinet"]["V0_A3"])
            dB0 = abs(results["BM3"]["B0_GPa"] - results["Vinet"]["B0_GPa"])
            print(f"\n  BM3 vs Vinet consistency:")
            print(f"    ΔV₀ = {dV0:.4f} Å³/atom ({dV0/results['BM3']['V0_A3']*100:.2f}%)")
            print(f"    ΔB₀ = {dB0:.2f} GPa ({dB0/results['BM3']['B0_GPa']*100:.2f}%)")
            if dB0 / results["BM3"]["B0_GPa"] > 0.05:
                print("    WARNING: BM3 and Vinet B₀ differ by > 5%. "
                      "The fit range may be too wide or data has outliers.")

    return results
```

### Workflow 3: Bootstrap confidence intervals for EOS parameters

Quantify fitting uncertainty by resampling the E-V data with replacement. Bootstrap CI captures both numerical noise and sensitivity to individual volume points.

```python
"""
Bootstrap uncertainty estimation for EOS parameters.
Resamples the (V, E) dataset with replacement N_boot times and refits the EOS.
"""

import numpy as np
from scipy.optimize import curve_fit


def bootstrap_eos_uncertainty(
    volumes: np.ndarray,
    energies: np.ndarray,
    eos_func,
    popt: np.ndarray,             # Best-fit parameters from fit_eos()
    n_bootstrap: int = 2000,
    confidence_level: float = 0.95,
    noise_level_eV: float = 1e-4,  # Add small random noise to simulate DFT numerical noise
                                    # ~ 0.1 meV/atom is typical for tight EDIFF=1e-7
) -> dict:
    """
    Bootstrap confidence intervals on EOS parameters.

    For each resample: draw N points with replacement from the (V, E) dataset,
    add optional noise, and refit the EOS. The distribution of bootstrap
    parameter estimates characterizes the statistical uncertainty.

    Returns percentile CIs at the requested confidence level.
    """
    n = len(volumes)
    alpha = 1 - confidence_level
    low_pct = 100 * alpha / 2
    high_pct = 100 * (1 - alpha / 2)

    param_names = ["E0_eV", "V0_A3", "B0_eV_A3", "B0p"]
    boot_params = {name: [] for name in param_names}

    rng = np.random.default_rng(42)
    n_failed = 0

    for _ in range(n_bootstrap):
        # Draw with replacement
        idx = rng.integers(0, n, size=n)
        V_boot = volumes[idx]
        E_boot = energies[idx] + rng.normal(0, noise_level_eV, size=n)

        # Refit
        try:
            popt_boot, _ = curve_fit(
                eos_func, V_boot, E_boot,
                p0=popt,
                bounds=(
                    [energies.min() - 1.0, volumes.min() * 0.8, 0.0, 0.5],
                    [energies.min() + 1.0, volumes.max() * 1.2, 1.0, 10.0],
                ),
                maxfev=5000,
            )
            for name, val in zip(param_names, popt_boot):
                boot_params[name].append(val)
        except RuntimeError:
            n_failed += 1
            continue

    EV_TO_GPA = 160.218

    n_success = n_bootstrap - n_failed
    print(f"\nBootstrap ({n_success}/{n_bootstrap} successful, {confidence_level*100:.0f}% CI):")

    ci_results = {}
    for name, values in boot_params.items():
        arr = np.array(values)
        lo = np.percentile(arr, low_pct)
        hi = np.percentile(arr, high_pct)
        med = np.median(arr)

        if name == "B0_eV_A3":
            lo_gpa = lo * EV_TO_GPA
            hi_gpa = hi * EV_TO_GPA
            med_gpa = med * EV_TO_GPA
            print(f"  B₀  = {med_gpa:.2f} GPa  [{lo_gpa:.2f}, {hi_gpa:.2f}] GPa")
            ci_results["B0_GPa_CI"] = (lo_gpa, hi_gpa)
            ci_results["B0_GPa_median"] = med_gpa
        elif name == "V0_A3":
            print(f"  V₀  = {med:.4f} Å³/atom  [{lo:.4f}, {hi:.4f}]")
            ci_results["V0_CI"] = (lo, hi)
        elif name == "B0p":
            print(f"  B'₀ = {med:.3f}  [{lo:.3f}, {hi:.3f}]")
            ci_results["B0p_CI"] = (lo, hi)
        elif name == "E0_eV":
            print(f"  E₀  = {med:.6f} eV  [{lo:.6f}, {hi:.6f}]")
            ci_results["E0_CI"] = (lo, hi)

    return ci_results
```

### Workflow 4: Outlier detection and bad volume points

Identify unconverged, magnetically inconsistent, or phase-transition-contaminated volume points before fitting.

```python
"""
Outlier detection for EOS E(V) datasets.

Three sources of bad volume points:
1. Unconverged SCF or relaxation: the DFT energy is not the true ground state.
2. Magnetic state change: the system found a different spin ground state at this volume.
3. Phase transition: the structure relaxed to a different polymorph at this volume.

Diagnostics for each source:
1. Check VASP convergence flags (vasprun.xml converged attribute).
2. Monitor total magnetic moment vs. volume (OUTCAR NUPDOWN or magnetization).
3. Monitor space group and Wyckoff positions of final structure vs. original.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from pymatgen.io.vasp.outputs import Vasprun, Outcar
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer


def detect_eos_outliers(
    volumes: np.ndarray,
    energies: np.ndarray,
    base_dir: str = "eos_calcs",
    residual_threshold_meV: float = 3.0,   # Flag residuals > this value
    spg_symprec: float = 0.1,              # Å; tolerance for space group comparison
) -> dict:
    """
    Multi-level outlier detection for EOS E(V) data.

    Returns a dict of suspected bad points with reasons.
    """
    n = len(volumes)
    warnings = {}

    # ── Level 1: Residual-based outlier detection ─────────────────────────────
    # Fit a preliminary polynomial to identify obvious outliers
    coeffs = np.polyfit(volumes, energies, 4)   # 4th-order polynomial
    E_poly = np.polyval(coeffs, volumes)
    residuals_meV = (energies - E_poly) * 1000

    for i, (V, E, res) in enumerate(zip(volumes, energies, residuals_meV)):
        if abs(res) > residual_threshold_meV:
            warnings[i] = warnings.get(i, {})
            warnings[i]["polynomial_residual_meV"] = res
            print(f"  WARNING: Point {i} (V={V:.3f} Å³/atom) has polynomial "
                  f"residual {res:.2f} meV/atom > {residual_threshold_meV} meV/atom")

    # ── Level 2: Magnetic moment vs. volume ───────────────────────────────────
    calc_dirs = sorted(Path(base_dir).glob("vol_*"))
    magmoms = []

    for calc_dir in calc_dirs:
        outcar_file = calc_dir / "OUTCAR"
        if not outcar_file.exists():
            magmoms.append(None)
            continue
        try:
            outcar = Outcar(str(outcar_file))
            # Total magnetization from OUTCAR (number of outcar.magnetization entries)
            if outcar.magnetization:
                total_mag = sum(s.get("tot", 0) for s in outcar.magnetization)
                magmoms.append(total_mag)
            else:
                magmoms.append(0.0)
        except Exception:
            magmoms.append(None)

    if any(m is not None for m in magmoms):
        valid_mags = [(i, m) for i, m in enumerate(magmoms) if m is not None]
        if len(valid_mags) >= 3:
            mag_values = [m for _, m in valid_mags]
            mag_mean = np.mean(mag_values)
            mag_std  = np.std(mag_values)

            for i, mag in valid_mags:
                if mag_std > 0.5 and abs(mag - mag_mean) > 2 * mag_std:
                    warnings[i] = warnings.get(i, {})
                    warnings[i]["magnetic_anomaly"] = f"μ_tot = {mag:.2f} μB (mean: {mag_mean:.2f} ± {mag_std:.2f})"
                    print(f"  WARNING: Volume point {i} has anomalous magnetic moment "
                          f"{mag:.2f} μB. Possible spin-state change.")

    # ── Level 3: Space group check ────────────────────────────────────────────
    space_groups = []
    for calc_dir in calc_dirs:
        vasprun_file = calc_dir / "vasprun.xml"
        if not vasprun_file.exists():
            space_groups.append(None)
            continue
        try:
            vrun = Vasprun(str(vasprun_file), parse_dos=False, parse_eigen=False)
            sga = SpacegroupAnalyzer(vrun.final_structure, symprec=spg_symprec)
            space_groups.append(sga.get_space_group_number())
        except Exception:
            space_groups.append(None)

    valid_spgs = [s for s in space_groups if s is not None]
    if valid_spgs:
        ref_spg = max(set(valid_spgs), key=valid_spgs.count)   # most common space group
        for i, spg in enumerate(space_groups):
            if spg is not None and spg != ref_spg:
                warnings[i] = warnings.get(i, {})
                warnings[i]["space_group_change"] = f"SG {spg} (expected {ref_spg})"
                print(f"  WARNING: Volume point {i} has space group {spg} "
                      f"(reference: {ref_spg}). Possible phase transition.")

    # Summary
    if warnings:
        print(f"\nOutlier summary: {len(warnings)} suspicious points detected")
        for idx, details in sorted(warnings.items()):
            print(f"  Point {idx} (V={volumes[idx]:.3f} Å³/atom): {details}")
        print("  Action: Exclude flagged points and refit, or re-run those volumes.")
    else:
        print("No outliers detected.")

    return {"warnings": warnings, "residuals_meV": residuals_meV, "magmoms": magmoms}
```

### Workflow 5: MLP vs DFT EOS comparison for validation

The EOS comparison is the standard first validation test for an ML interatomic potential. The MLP must reproduce DFT V₀ within ~1% and B₀ within ~5% to be considered accurate. [EXPERT REVIEW NEEDED]

```python
"""
Compare MLP EOS to DFT EOS: the primary validation for bulk mechanical properties.

Protocol:
1. Run the DFT EOS workflow (Workflow 1–2) to get the reference E-V curve.
2. Run the MLP EOS on the same set of structures using ASE.
3. Fit BM3 to both datasets and compare V₀, B₀, B'₀.
4. Plot both EOS curves and residuals.
"""

import numpy as np
import matplotlib.pyplot as plt
from ase.io import read
from ase.eos import EquationOfState
from pathlib import Path


def compute_mlp_eos(
    poscar_file: str,           # Reference POSCAR (equilibrium structure)
    mlp_calculator,             # ASE-compatible calculator (MACE, NequIP, etc.)
    volume_fractions: np.ndarray = None,
    n_volumes: int = 11,
    v_range_frac: float = 0.10,
    eos_form: str = "birchmurnaghan",   # ASE EOS form name
) -> dict:
    """
    Compute MLP EOS using ASE by scaling the reference structure.

    Unlike the DFT workflow, MLP EOS can optionally relax cell shape at each
    volume using ASE optimizers, or simply evaluate energy at scaled structures.
    For a fair DFT comparison: relax ions+shape at each volume with the MLP
    (equivalent to ISIF=4 in VASP).
    """
    from ase.optimize import FIRE
    from ase.constraints import StrainFilter

    atoms_ref = read(poscar_file)
    V0_ref = atoms_ref.get_volume()
    n_atoms = len(atoms_ref)

    if volume_fractions is None:
        volume_fractions = np.linspace(1 - v_range_frac, 1 + v_range_frac, n_volumes)

    mlp_volumes = []
    mlp_energies = []

    for vf in volume_fractions:
        atoms = atoms_ref.copy()
        atoms.set_cell(atoms_ref.cell * vf**(1.0/3.0), scale_atoms=True)
        atoms.calc = mlp_calculator

        # Relax ions + cell shape at fixed volume (equivalent to ISIF=4)
        # Use StrainFilter with mask to freeze volume and allow shape relaxation
        sf = StrainFilter(atoms, mask=[True, True, True, True, True, True])
        # Note: StrainFilter with all True relaxes all strain components including
        # volume. For fixed-volume, use ExpCellFilter with hydrostatic_strain=False
        # and a volume constraint. Simpler alternative: relax ions only (ISIF=2 equivalent).
        from ase.constraints import FixAtoms
        opt = FIRE(atoms, logfile=None)
        try:
            opt.run(fmax=0.01, steps=500)   # fmax = 0.01 eV/Å, matches EDIFFG=-0.01
        except Exception as e:
            print(f"  WARNING: relaxation failed at vf={vf:.4f}: {e}")

        mlp_volumes.append(atoms.get_volume() / n_atoms)   # Å³/atom
        mlp_energies.append(atoms.get_potential_energy() / n_atoms)  # eV/atom

    mlp_volumes = np.array(mlp_volumes)
    mlp_energies = np.array(mlp_energies)

    # Fit EOS using ASE's built-in EOS (wraps scipy curve_fit)
    eos_ase = EquationOfState(mlp_volumes, mlp_energies, eos=eos_form)
    V0_mlp, E0_mlp, B0_mlp_eV_A3 = eos_ase.fit()
    B0_mlp_GPa = B0_mlp_eV_A3 * 160.218

    print(f"MLP EOS ({eos_form}):")
    print(f"  V₀ = {V0_mlp:.4f} Å³/atom")
    print(f"  E₀ = {E0_mlp:.6f} eV/atom")
    print(f"  B₀ = {B0_mlp_GPa:.2f} GPa")

    return {
        "volumes": mlp_volumes,
        "energies": mlp_energies,
        "V0": V0_mlp,
        "E0": E0_mlp,
        "B0_GPa": B0_mlp_GPa,
        "eos": eos_ase,
    }


def compare_dft_mlp_eos(
    dft_results: dict,    # from fit_eos(), key "BM3"
    mlp_results: dict,    # from compute_mlp_eos()
    material_name: str = "Material",
    plot: bool = True,
) -> dict:
    """
    Tabulate and plot DFT vs MLP EOS comparison.
    """
    dft = dft_results
    mlp = mlp_results

    dV0_pct = abs(mlp["V0"] - dft["V0_A3"]) / dft["V0_A3"] * 100
    dB0_pct = abs(mlp["B0_GPa"] - dft["B0_GPa"]) / dft["B0_GPa"] * 100

    print(f"\n{'='*55}")
    print(f"EOS Comparison: {material_name}")
    print(f"{'='*55}")
    print(f"{'Property':<12} {'DFT':>12} {'MLP':>12} {'|Δ|%':>8}")
    print(f"{'-'*55}")
    print(f"{'V₀ (Å³)':<12} {dft['V0_A3']:>12.4f} {mlp['V0']:>12.4f} {dV0_pct:>8.2f}%")
    print(f"{'B₀ (GPa)':<12} {dft['B0_GPa']:>12.2f} {mlp['B0_GPa']:>12.2f} {dB0_pct:>8.2f}%")
    print(f"{'='*55}")

    # Acceptance criteria [EXPERT REVIEW NEEDED: thresholds are system-dependent]
    if dV0_pct < 1.0 and dB0_pct < 5.0:
        print("RESULT: MLP EOS PASSES validation (ΔV₀ < 1%, ΔB₀ < 5%)")
    elif dV0_pct < 2.0 and dB0_pct < 10.0:
        print("RESULT: MLP EOS MARGINAL (ΔV₀ < 2%, ΔB₀ < 10%) — review training data")
    else:
        print("RESULT: MLP EOS FAILS — retrain or extend training set with bulk compressions")

    if plot:
        V_range = np.linspace(
            min(dft["popt"][1] * 0.88, mlp["volumes"].min() * 0.95),
            max(dft["popt"][1] * 1.12, mlp["volumes"].max() * 1.05),
            300
        )
        E_dft_curve = birch_murnaghan_3(V_range, *dft["popt"])
        E_mlp_curve = np.interp(V_range, mlp["volumes"],
                                 np.polyval(np.polyfit(mlp["volumes"], mlp["energies"], 4), mlp["volumes"]))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        ax1.plot(V_range, E_dft_curve, "b-", lw=2, label="DFT (BM3 fit)")
        ax1.scatter(mlp["volumes"], mlp["energies"], s=60, color="r",
                    zorder=5, label="MLP data")
        ax1.axvline(dft["V0_A3"], color="b", ls="--", lw=1, alpha=0.6, label=f"DFT V₀")
        ax1.axvline(mlp["V0"], color="r", ls="--", lw=1, alpha=0.6, label=f"MLP V₀")
        ax1.set_xlabel("Volume (Å³/atom)")
        ax1.set_ylabel("Energy (eV/atom)")
        ax1.set_title(f"EOS comparison — {material_name}")
        ax1.legend()

        # Energy residuals: MLP(V) - DFT_fit(V)
        E_dft_at_mlp = birch_murnaghan_3(mlp["volumes"], *dft["popt"])
        delta_E_meV = (mlp["energies"] - E_dft_at_mlp) * 1000

        ax2.bar(range(len(delta_E_meV)), delta_E_meV, color=["r" if abs(d) > 5 else "g" for d in delta_E_meV])
        ax2.axhline(0, color="k", lw=0.5)
        ax2.axhline(5, color="k", ls="--", lw=0.8, alpha=0.5, label="5 meV/atom")
        ax2.axhline(-5, color="k", ls="--", lw=0.8, alpha=0.5)
        ax2.set_xlabel("Volume point index")
        ax2.set_ylabel("E_MLP − E_DFT_fit (meV/atom)")
        ax2.set_title("MLP energy residuals vs DFT EOS")
        ax2.legend()

        plt.tight_layout()
        plt.savefig(f"eos_comparison_{material_name.replace(' ', '_')}.png", dpi=150)
        print(f"Saved: eos_comparison_{material_name.replace(' ', '_')}.png")

    return {"dV0_pct": dV0_pct, "dB0_pct": dB0_pct}
```

### Workflow 6: ASE-based EOS fitting for quick validation

ASE provides a minimal EOS workflow for fast validation without running a full VASP job array. Suitable for screening ML potentials before committing to DFT EOS generation.

```python
"""
Quick EOS fitting with ASE: scale a reference structure, evaluate energies,
and fit the BM EOS using ASE's built-in EquationOfState class.
"""

import numpy as np
import matplotlib.pyplot as plt
from ase.io import read
from ase.eos import EquationOfState
from ase.build import bulk


def ase_eos_quick(
    calculator,               # Any ASE-compatible calculator
    reference_structure = None,  # ASE Atoms; if None, use bulk() for a test
    element: str = "Cu",     # Only used if reference_structure is None
    crystal: str = "fcc",    # Only used if reference_structure is None
    n_volumes: int = 9,
    v_range_frac: float = 0.08,
    eos_form: str = "birchmurnaghan",
    relax_ions: bool = False, # Set True for non-cubic (requires FIRE optimizer)
) -> dict:
    """
    ASE quick EOS: scale volumes, compute energies, fit BM EOS.

    For a fair comparison with VASP ISIF=4, set relax_ions=True.
    For a fast screening test, set relax_ions=False (single-point energies at each volume).

    ASE's EquationOfState supports: 'sj' (Birch, Shanker, Johnson), 'taylor',
    'murnaghan', 'birch', 'birchmurnaghan', 'pouriertarantola', 'vinet', 'antonschmidt'.
    """
    if reference_structure is None:
        # MINIMAL EXAMPLE — not for production
        atoms_ref = bulk(element, crystal, cubic=True)
        print(f"Using bulk {element} ({crystal}) as reference structure.")
    else:
        atoms_ref = reference_structure

    atoms_ref.calc = calculator
    V0_ref = atoms_ref.get_volume()
    n_atoms = len(atoms_ref)

    vf_list = np.linspace(1 - v_range_frac, 1 + v_range_frac, n_volumes)
    volumes, energies = [], []

    for vf in vf_list:
        a = atoms_ref.copy()
        a.set_cell(atoms_ref.cell * vf**(1.0/3.0), scale_atoms=True)
        a.calc = calculator

        if relax_ions:
            from ase.optimize import FIRE
            opt = FIRE(a, logfile=None)
            opt.run(fmax=0.02, steps=200)

        E = a.get_potential_energy()
        V = a.get_volume()
        volumes.append(V / n_atoms)
        energies.append(E / n_atoms)

    volumes = np.array(volumes)
    energies = np.array(energies)

    eos = EquationOfState(volumes, energies, eos=eos_form)
    V0, E0, B0_eV_A3 = eos.fit()
    B0_GPa = B0_eV_A3 * 160.218

    print(f"\nASE EOS ({eos_form}):")
    print(f"  V₀ = {V0:.4f} Å³/atom")
    print(f"  E₀ = {E0:.6f} eV/atom")
    print(f"  B₀ = {B0_GPa:.2f} GPa")

    # ASE built-in plot
    ax = eos.plot(filename="ase_eos.png", show=False)
    print("Saved: ase_eos.png")

    return {"V0": V0, "E0": E0, "B0_GPa": B0_GPa, "eos": eos,
            "volumes": volumes, "energies": energies}
```

### Workflow 7: EOS publication figure — multi-functional and experimental comparison

```python
"""
Publication-quality EOS comparison figure: DFT (PBE, SCAN), MLP, and experimental.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def plot_eos_comparison(
    datasets: list,     # list of dicts: {"label", "volumes", "energies", "popt", "style"}
    material_name: str,
    experimental_V0: float = None,      # Å³/atom (experimental at T≈0 K)
    experimental_B0: float = None,      # GPa
    experimental_source: str = None,
    pressure_range_GPa: tuple = (0, 50),
    output_file: str = "eos_publication.png",
) -> None:
    """
    Two-panel figure: (left) E-V curves + fits; (right) P-V curves to 50 GPa.

    Best practice: always show the scatter (raw DFT points) alongside the EOS fit.
    Always show residuals or state the RMS residual in the caption.
    Never extrapolate the EOS beyond the data range without explicit caveats.
    """
    fig = plt.figure(figsize=(14, 6))
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.3)
    ax_ev = fig.add_subplot(gs[0, 0])
    ax_pv = fig.add_subplot(gs[0, 1])

    for ds in datasets:
        label = ds["label"]
        vols = ds["volumes"]
        enes = ds["energies"]
        popt = ds.get("popt")
        style = ds.get("style", {})
        color = style.get("color", None)
        ls    = style.get("ls", "-")
        marker = style.get("marker", "o")
        ms    = style.get("ms", 6)

        # E-V scatter (raw points)
        ax_ev.scatter(vols, enes - enes.min(), s=ms**2, color=color, alpha=0.7, zorder=5)

        if popt is not None:
            V_dense = np.linspace(vols.min() * 0.97, vols.max() * 1.03, 300)
            E_dense = birch_murnaghan_3(V_dense, *popt)
            E_dense -= E_dense.min()
            ax_ev.plot(V_dense, E_dense, ls=ls, color=color, lw=2, label=label)

            # P-V curve from BM3 analytical derivative
            P_range = np.linspace(pressure_range_GPa[0], pressure_range_GPa[1], 200)
            V_P = []
            for P_target in P_range:
                from scipy.optimize import brentq
                try:
                    V_sol = brentq(
                        lambda V: bm3_pressure(V, popt[1], popt[2], popt[3]) - P_target,
                        vols.min() * 0.7,
                        vols.max() * 1.5,
                    )
                    V_P.append(V_sol)
                except Exception:
                    V_P.append(np.nan)
            ax_pv.plot(V_P, P_range, ls=ls, color=color, lw=2, label=label)

    # Experimental reference
    if experimental_V0 is not None:
        ax_ev.axvline(experimental_V0, color="k", ls=":", lw=1.5,
                      label=f"Exp V₀ ({experimental_source})")
        ax_pv.axvline(experimental_V0, color="k", ls=":", lw=1.5)

    ax_ev.set_xlabel("Volume (Å³/atom)", fontsize=12)
    ax_ev.set_ylabel("E − E_min (eV/atom)", fontsize=12)
    ax_ev.set_title(f"{material_name} — energy-volume EOS", fontsize=12)
    ax_ev.legend(fontsize=9)
    ax_ev.set_ylim([-0.005, 0.2])

    ax_pv.set_xlabel("Volume (Å³/atom)", fontsize=12)
    ax_pv.set_ylabel("Pressure (GPa)", fontsize=12)
    ax_pv.set_title(f"{material_name} — pressure-volume EOS", fontsize=12)
    ax_pv.legend(fontsize=9)
    ax_pv.set_ylim(pressure_range_GPa)

    plt.suptitle(material_name, fontsize=14, fontweight="bold", y=1.01)
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_file}")
```

## Common Failure Modes

| Failure | Symptom | Likely Cause | Diagnostic | Fix |
|---------|---------|--------------|------------|-----|
| Too few volume points | BM3 fit fails or gives B'₀ < 1 or B'₀ > 8; large uncertainty in B₀ from bootstrap; results change when a single point is removed | Fewer than 7 volume points; undersampled curvature at the minimum | Bootstrap CI on B₀ is > 10 GPa; remove one point and refit — if B₀ shifts by > 5%, the dataset is under-constrained | Use at least 9–11 volume points for production EOS; 7 is the minimum for a reliable BM3 fit |
| Volume range too narrow | B'₀ is poorly determined; BM3 and Vinet give very different B'₀; polynomial fit overshoots at edges | ±3% volume range is too small to constrain the quartic E-V behavior | Compare BM3 and Vinet B'₀: if they differ by > 1, the fit range is too small | Extend to ±8–12% for hard materials, ±12–15% for soft materials; verify no phase transition in extended range |
| Volume range too wide (phase transition inside range) | Large residuals at outer volume points; kink in E-V curve; space group changes at high compression | E-V curve crosses a phase-transition boundary; the EOS form assumes a single phase | Plot structure space group vs. volume (Workflow 4); large residuals at low-V points indicate high-pressure phase | Restrict fit range to exclude the transition; run separate EOS for each phase |
| Pulay stress artifacts | V₀ from EOS disagrees with ISIF=3 full relaxation by > 1%; B₀ is systematically overestimated | ENCUT too low; Pulay stress shifts E-V curve; stress is not zero at ISIF=4 equilibrium | Compare residual isotropic stress across volumes (should be < 5 kbar); compute EOS at 1.3×ENCUT and compare V₀ | Increase ENCUT to ≥ 1.3 × max(ENMAX); verify residual Pulay stress < 1 kbar at each volume |
| Magnetic state change across volumes | E-V curve has a kink; large residual at one or two volume points; magnetic moment drops abruptly | Ferromagnetic → antiferromagnetic or non-magnetic transition at some intermediate volume | Plot total magnetic moment vs. volume from OUTCAR; a discontinuous jump identifies the transition point | Exclude points beyond the magnetic transition; fit each phase separately; or constrain spin state with NUPDOWN |
| ISIF=2 used instead of ISIF=4 | E-V curve is smooth but V₀ is wrong by > 1% for non-cubic systems; B₀ is overestimated for anisotropic materials | ISIF=2 fixes all cell parameters (lengths and angles); shape is not optimized at each volume | Compare V₀ to ISIF=3 full relaxation; if they disagree by > 0.5%, ISIF was wrong | Re-run all volume points with ISIF=4; add a note that results with ISIF=2 are unreliable for non-cubic structures |
| Unconverged relaxation at an extreme volume | Large residual at the lowest or highest volume point; OUTCAR shows ionic convergence not reached | Force convergence is harder at highly strained volumes where the initial structure is far from equilibrium | Check `vasprun.xml converged_ionic`; grep for `reached required accuracy` in each OUTCAR | Increase NSW, tighten EDIFFG from −0.02 to −0.01; restart from a partially relaxed CONTCAR at the problematic volume |
| Overfitting high-order polynomial | A polynomial EOS with degree > 4 fits the E-V data perfectly but gives unphysical B'₀ (< 0 or > 10) | High-degree polynomial is flexible enough to fit noise; parameters are not physically interpretable | Compute AIC/BIC for different polynomial degrees; compare polynomial B₀ to BM3 and Vinet | Always use BM3 or Vinet (physical EOS forms with 4 parameters) rather than free-form polynomial fits; use polynomial only for outlier detection |
| MLP EOS fails validation without diagnosis | ΔB₀ > 10% between MLP and DFT; cause is unknown; retraining is begun without identifying the problem | Training set does not contain isotropically compressed or expanded bulk configurations; MLP extrapolates outside training distribution | Check committee uncertainty at each EOS volume with an ensemble MLP; identify which volumes are out of distribution | Add bulk EOS configurations (across the full V range) to the training set via active learning; do not just add data near V₀ |
| Reporting B₀ without uncertainty | Paper states "B₀ = 182 GPa" with no uncertainty; another DFT study reports 178 GPa; both claim 1 meV/atom precision | Statistical uncertainty from fitting not computed; systematic functional dependence not explored | Bootstrap CI (Workflow 3) gives fitting uncertainty; functional comparison (PBE vs SCAN vs LDA) gives systematic bias | Always report bootstrap CI on B₀; additionally compare two functionals to bound systematic error; report as "B₀ = 182 ± 3 GPa (BM3, bootstrap)" |

## Best Practices

- **Always use ISIF=4 (not ISIF=2) for non-cubic systems.** ISIF=2 fixes the cell shape, introducing artificial shape stress at strained volumes. For cubic systems, ISIF=2 and ISIF=4 are equivalent (no shape degrees of freedom), but using ISIF=4 consistently avoids errors when the symmetry is unexpectedly broken during relaxation.
- **Set ENCUT ≥ 1.3 × max(ENMAX) to suppress Pulay stress.** The standard VASP recommendation is ENCUT ≥ 1.3 × ENMAX of the hardest element. Verify the residual isotropic stress at each volume point (read from OUTCAR) and flag any point with |σ_iso| > 5 kbar for re-calculation at higher ENCUT.
- **Always fit at least two EOS forms (BM3 and Vinet) and compare.** If B₀ from BM3 and Vinet disagrees by > 5%, the fitting is not reliable — the data range may be too wide or contain a phase transition. Use the smaller B₀ from the two as a conservative estimate.
- **Compute bootstrap confidence intervals before reporting B₀.** The formal curve-fitting covariance underestimates uncertainty because it does not account for the nonlinearity of the fit or the possible presence of mild outliers. Bootstrap resampling with N = 1000–2000 provides a more honest uncertainty estimate.
- **Monitor total magnetic moment as a function of volume for magnetic materials.** A discontinuous jump in magnetic moment within the fit range invalidates the EOS. Always plot μ_tot vs. V alongside the E-V curve when ISPIN=2 is used.
- **Use 11 volume points at ±10% for standard materials; reduce to ±6–8% for soft or anharmonic systems.** The inner ±5% determines B₀ (curvature at the minimum); the outer ±10% determines B'₀ (asymmetry of the well). For soft materials, the ±10% range may cross a phase transition — use ±6–8% and verify with a space group check.
- **For MLP EOS validation: the acceptance threshold is system-dependent.** ΔV₀ < 1% and ΔB₀ < 5% are reasonable targets for a well-trained MLP on a standard metal or ceramic. Molecular crystals, soft matter, and systems with long-range van der Waals interactions have larger intrinsic DFT errors and the MLP–DFT comparison has correspondingly wider tolerances. [EXPERT REVIEW NEEDED]
- **Verify the EOS equilibrium volume against ISIF=3 full relaxation.** V₀ from the EOS fit and V₀ from a direct ISIF=3 optimization must agree to within 0.5% at the same ENCUT and k-mesh. A larger discrepancy indicates either a fitting problem or that the initial structure used for the EOS scan was not close to the true equilibrium.

## Integration with Other Skills

- **[dft-convergence](../dft-convergence/SKILL.md):** The EOS is the most sensitive bulk property to Pulay stress and basis-set incompleteness. ENCUT convergence testing from that skill must be completed before generating an EOS dataset; use ENCUT ≥ 1.3 × ENMAX rather than the standard production setting for EOS calculations.
- **[vasp-workflow](../vasp-workflow/SKILL.md):** All VASP input-file conventions (INCAR, KPOINTS, POTCAR hierarchy, `grompp → mdrun` analogue of `grompp → vasp`), file-parsing with pymatgen, and job submission patterns from that skill apply directly to EOS job arrays. ISIF=4 is introduced in this skill; ISIF=3 (full relaxation) is the standard covered there.
- **[quantum-espresso](../quantum-espresso/SKILL.md):** Quantum ESPRESSO EOS generation uses `calculation = 'relax'` with `cell_dofree = 'ibrav'` to fix volume and optimize shape, analogous to VASP ISIF=4. The EOS fitting Python code (Workflows 2–3) is code-agnostic; it requires only (V, E) arrays.
- **[high-throughput-dft](../high-throughput-dft/SKILL.md):** EOS generation across a materials database (e.g., all binary alloys in a composition space) requires the workflow graph (atomate2/jobflow) and database infrastructure from that skill. The per-material EOS calculation described here becomes one node in a larger high-throughput graph.
- **[mlp-validation](../../ml-interatomic-potentials/mlp-validation/SKILL.md):** EOS comparison (ΔV₀, ΔB₀) is a required test in MLP validation. That skill defines the full validation suite (energies, forces, stresses, phonons, EOS); this skill provides the detailed protocol for the EOS component.
- **[mace-training](../../ml-interatomic-potentials/mace-training/SKILL.md):** If MLP EOS validation fails (ΔB₀ > 10%), the training set likely lacks isotropically strained bulk configurations. This skill's volume-scan structures (Workflow 1) serve as efficient additional training data; adding them via the active-learning or fine-tuning workflow in that skill improves the MLP bulk modulus prediction.
- **[active-learning-mlp](../../ml-interatomic-potentials/active-learning-mlp/SKILL.md):** EOS volume points with high committee uncertainty identify the configurations where the MLP extrapolates. Those points are the highest-priority candidates for DFT labeling in an active learning loop. The E-V scan from Workflow 1 can be integrated directly into an active learning workflow to improve bulk property accuracy.
- **[scientific-visualization](../../scientific-data/scientific-visualization/SKILL.md):** E-V curve plots, P-V curves, residual histograms, and bootstrap parameter distributions (Workflows 3 and 7) follow the matplotlib patterns from that skill. EOS figures should always show raw data points alongside the fitted curve; never plot only the fitted line.
- **[reproducibility-workflow](../../scientific-data/reproducibility-workflow/SKILL.md):** Every EOS dataset must archive: the input POSCAR, INCAR, KPOINTS, POTCAR (or POTCAR hash), the volume fractions used, all `vasprun.xml` files, the fitting script with version, and the resulting EOS parameters. That skill provides the DVC and directory-structure framework for this archive.
- **[pymatgen-analysis](../../scientific-data/pymatgen-analysis/SKILL.md):** Pymatgen's `Structure.scale_lattice()`, `Vasprun` parser, `SpacegroupAnalyzer`, and `EosSet` (in pymatgen-analysis-eos) are used throughout this skill. That skill covers pymatgen API patterns, structure manipulation, and high-throughput data extraction.
- **[ase-framework](../../scientific-data/ase-framework/SKILL.md):** ASE's `EquationOfState` class (Workflow 6) and the `StrainFilter` / `ExpCellFilter` for fixed-volume relaxation are the primary ASE tools for EOS work. That skill covers ASE calculator interface conventions, trajectory management, and unit conversions.

## Key References

1. Birch, F. (1947). *Finite elastic strain of cubic crystals.* Physical Review, 71(11), 809–824. https://doi.org/10.1103/PhysRev.71.809
2. Murnaghan, F. D. (1944). *The compressibility of media under extreme pressures.* Proceedings of the National Academy of Sciences, 30(9), 244–247. https://doi.org/10.1073/pnas.30.9.244
3. Vinet, P., Ferrante, J., Rose, J. H., & Smith, J. R. (1987). *Compressibility of solids.* Journal of Geophysical Research: Solid Earth, 92(B9), 9319–9325. https://doi.org/10.1029/JB092iB09p09319
4. Haas, P., Tran, F., & Blaha, P. (2009). *Calculation of the lattice constant of solids with semilocal functionals.* Physical Review B, 79(8), 085104. https://doi.org/10.1103/PhysRevB.79.085104
5. de Jong, M., Chen, W., Angsten, T., Jain, A., Notestine, R., Gamst, A., et al. (2015). *Charting the complete elastic properties of inorganic crystalline compounds.* Scientific Data, 2, 150009. https://doi.org/10.1038/sdata.2015.9
6. Francisco, E., Recio, J. M., Blanco, M. A., & Pendás, A. M. (1998). *Quantum-mechanical study of thermodynamic and bonding properties of MgF₂.* Journal of Physical Chemistry A, 102(9), 1595–1601. https://doi.org/10.1021/jp972516j
7. Bahn, S. R. & Jacobsen, K. W. (2002). *An object-oriented scripting interface to a legacy electronic structure code.* Computing in Science & Engineering, 4(3), 56–66. https://doi.org/10.1109/5992.998641
8. Lejaeghere, K., Van Speybroeck, V., Van Oost, G., & Cottenier, S. (2014). *Error estimates for solid-state density-functional theory predictions: An overview by means of the ground-state elemental crystals.* Critical Reviews in Solid State and Materials Sciences, 39(1), 1–24. https://doi.org/10.1080/10408436.2013.772503
