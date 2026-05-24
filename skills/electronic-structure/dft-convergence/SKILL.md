---
name: dft-convergence
domain: electronic-structure
maturity: draft
tools:
  - VASP
  - Quantum ESPRESSO
  - ASE
  - pymatgen
  - vaspkit
tested_versions:
  VASP: "6.3.2"
  Quantum ESPRESSO: "7.3"
  ASE: "3.23.0"
  pymatgen: "2024.11.13"
  vaspkit: "1.3.5"
requires_gpu: false
estimated_runtime: minutes-to-hours
---

# DFT Convergence

## Description

This skill covers systematic convergence testing for plane-wave DFT calculations: cutoff energy (ENCUT / ecutwfc), k-point mesh density, smearing scheme and width, and SCF thresholds. Convergence requirements change significantly depending on the property of interest, the code, and the downstream use of the data. Invoke this skill before any production DFT run, whenever introducing a new element or pseudopotential family, and whenever changing codes or functional.

## Domain Context

Plane-wave DFT is variational: the energy decreases monotonically toward the exact result as the basis set (ENCUT) and Brillouin zone sampling (k-points) are increased. This means convergence can always be tested systematically. However, two important subtleties apply:

- **Energy differences converge faster than absolute energies.** Systematic errors in ENCUT and k-points largely cancel when comparing two structures with the same composition and a similar geometry. Reaction energies and adsorption energies can be converged to 1 meV/atom with a lower ENCUT than is needed for the absolute total energy at the same tolerance.
- **Different properties have different convergence rates.** Forces, stresses, phonon frequencies, band gaps, magnetic moments, and dielectric constants each have their own sensitivity to basis set completeness and k-mesh density. A calculation converged for total energy is not automatically converged for phonon frequencies or surface energies.

The pseudopotential or PAW dataset defines a recommended cutoff (ENMAX in VASP POTCAR; wfc_cutoff or ecutwfc in Quantum ESPRESSO input), but these recommendations are conservative lower bounds, not validated production defaults. The density cutoff (ecutrho in QE, controlled implicitly by VASP) must also be checked for ultrasoft pseudopotentials.

Smearing introduces a fictitious electronic temperature to smooth the Fermi surface and improve SCF convergence in metals. The smearing width (SIGMA in VASP; degauss in QE) must be small enough that the physical result is not distorted, but large enough to achieve stable convergence. The correct smearing scheme depends on whether the system is metallic, semiconducting, or insulating.

## When to Use This Skill

- Before any production DFT run on a new material or element combination.
- When switching between pseudopotential families, PAW datasets, or functional.
- When comparing energies across structures with different unit cell sizes or compositions.
- When generating DFT training data for ML potentials (consistent convergence across the entire dataset is mandatory).
- When results are suspected to carry Pulay stress or basis-set incompleteness errors.
- When submitting a DFT dataset to a materials database or as supporting data for a publication.

## When Not to Use This Skill

- **Exploratory screening at intentionally loose settings** — if you are ranking hundreds of structures by relative stability and 10 meV/atom accuracy is acceptable, a fixed coarse grid is often used deliberately. Document this choice explicitly and do not compare those energies to tightly converged values.
- **When a validated high-throughput workflow (e.g., MP, AFLOW) has already established material-class-specific settings** — re-running convergence is redundant if you are operating strictly within their protocol and comparing only to data generated with the same settings.
- **For systems where the unconverged quantity is irrelevant** — e.g., computing adsorption geometries where a moderate k-mesh error in absolute energy cancels in the energy difference.

## Core Concepts

- **ENCUT / ecutwfc:** Plane-wave kinetic energy cutoff. Controls how many plane waves are included in the basis. Higher values improve completeness but increase cost as O(ENCUT^(3/2)) for the number of plane waves. Must be identical across all calculations that will be compared.
- **ecutrho (QE) / implicit rho cutoff (VASP):** Charge density cutoff. For norm-conserving pseudopotentials, ecutrho = 4 × ecutwfc. For ultrasoft (USPP) or PAW, typical ratio is 8–12×. VASP handles this automatically; in QE it must be set explicitly and tested for USPP.
- **Monkhorst-Pack mesh:** Uniform k-point grid in reciprocal space. For a given lattice, the effective k-point density is mesh / reciprocal lattice vector length (units: points per Å⁻¹). Convergence is material-dependent and must be tested separately for each compound.
- **Gamma-centered vs. MP-centered mesh:** Hexagonal, trigonal, FCC, and BCC Bravais lattices require Gamma-centered grids for correct symmetry. Off-Gamma MP grids on hexagonal systems produce wrong total energies and band structures.
- **Methfessel-Paxton smearing (ISMEAR=1/2 in VASP, smearing='mp' in QE):** Correct polynomial expansion of the Fermi function for metals. Use order 1 or 2. Fails for semiconductors/insulators — produces unphysical negative occupations.
- **Gaussian smearing (ISMEAR=0 in VASP, smearing='gauss' in QE):** Safe for all systems. Required for gap-opening Mott insulators or systems of unknown character. Less accurate than MP for metals.
- **Tetrahedron method with Blöchl corrections (ISMEAR=-5 in VASP, smearing='tetrahedra_opt' in QE):** Most accurate for density of states and total energy of non-metals with dense k-meshes. Cannot be used for metals (Fermi surface integration fails) or with too few k-points.
- **Absolute vs. relative convergence:** A calculation is absolutely converged when the property is insensitive to further increases in ENCUT or k-mesh. It is relatively converged when two comparable calculations have the same systematic error, so their difference is accurate. Relative convergence is sufficient for energy differences but not for absolute quantities.
- **Pulay stress:** Basis-set-incompleteness artifact in stress tensor calculations. Arises because the basis set changes as the lattice is strained. Fixed by using ENCUT ≥ 1.3 × ENMAX (VASP) or by relaxing with higher cutoff and verifying residual stress is negligible.

## Key Workflows

### Workflow 1: ENCUT convergence scan (VASP)

Test convergence of total energy and force magnitude as a function of ENCUT. Run at fixed geometry (NSW=0) to isolate the basis-set effect.

```python
from pathlib import Path
from pymatgen.core import Structure
from pymatgen.io.vasp.sets import MPStaticSet
from pymatgen.io.vasp.outputs import Vasprun

structure = Structure.from_file("POSCAR")

# Scan from 1.0x to 1.5x the highest ENMAX across all species.
# Read ENMAX from POTCAR headers or from pymatgen's POTCAR data.
# Example range for a system where max ENMAX = 400 eV:
encut_values = [400, 450, 500, 550, 600, 650]  # eV

for encut in encut_values:
    calc_dir = Path(f"conv_encut_{encut}")
    calc_dir.mkdir(exist_ok=True)
    vis = MPStaticSet(structure, user_incar_settings={
        "ENCUT": encut,
        "EDIFF": 1e-7,      # Tight SCF for convergence testing; looser in production
        "NSW": 0,
        "IBRION": -1,
        "ISMEAR": 0,        # Gaussian: safe for both metals and insulators during testing
        "SIGMA": 0.05,
        "LWAVE": False,
        "LCHARG": False,
        "PREC": "Accurate",
        "ALGO": "Normal",
    })
    vis.write_input(str(calc_dir))
    # Submit each calculation; collect results after all complete.

# Parse results
results = {}
for encut in encut_values:
    path = Path(f"conv_encut_{encut}/vasprun.xml")
    if not path.exists():
        continue
    v = Vasprun(str(path))
    natoms = len(v.final_structure)
    results[encut] = {
        "energy_per_atom": v.final_energy / natoms,
        "max_force": max(
            sum(f**2 for f in fvec)**0.5
            for fvec in v.ionic_steps[-1]["forces"]
        ) if v.ionic_steps[-1].get("forces") else None,
    }

# Report convergence
encuts = sorted(results)
print(f"{'ENCUT':>8}  {'E/atom (eV)':>14}  {'ΔE (meV/atom)':>15}  {'F_max (eV/Å)':>14}")
for i, e in enumerate(encuts):
    de = (results[e]["energy_per_atom"] - results[encuts[i-1]]["energy_per_atom"]) * 1000 if i > 0 else 0.0
    fmax = results[e]["max_force"] or float("nan")
    print(f"{e:>8}  {results[e]['energy_per_atom']:>14.6f}  {de:>15.3f}  {fmax:>14.5f}")

# Convergence criterion (property-dependent):
# Total energy:       |ΔE/atom| < 1 meV/atom between successive ENCUT steps
# Forces (relaxation): |ΔE/atom| < 0.1–0.5 meV/atom  [EXPERT REVIEW NEEDED]
# Stress / Pulay:     use ENCUT >= 1.3 * ENMAX and verify residual stress < 0.5 kBar
# Phonons:            |ΔE/atom| < 0.1 meV/atom; validate specific phonon modes [EXPERT REVIEW NEEDED]
# ML potential data:  consistent ENCUT across all dataset structures is more important
#                     than the absolute value; 1 meV/atom tolerance on total energy
```

### Workflow 2: k-point convergence scan (VASP)

After fixing ENCUT, scan k-point mesh density. Express convergence in terms of k-point linear density (points per Å⁻¹) to make the result portable across differently-shaped cells.

```python
from pathlib import Path
from pymatgen.core import Structure
from pymatgen.io.vasp.inputs import Kpoints
from pymatgen.io.vasp.sets import MPStaticSet
from pymatgen.io.vasp.outputs import Vasprun
import numpy as np

structure = Structure.from_file("POSCAR")
recip_lengths = np.linalg.norm(structure.lattice.reciprocal_lattice.matrix, axis=1)  # Å⁻¹

# Define meshes in terms of target linear density (k-points per Å⁻¹).
# Translate to grid dimensions for this specific cell.
target_densities = [0.03, 0.04, 0.05, 0.07, 0.10]  # Å⁻¹ per point

for density in target_densities:
    grid = tuple(max(1, int(np.ceil(L / density))) for L in recip_lengths)
    label = "x".join(map(str, grid))
    calc_dir = Path(f"conv_kpts_{label}")
    calc_dir.mkdir(exist_ok=True)

    vis = MPStaticSet(structure, user_incar_settings={
        "ENCUT": 520,          # Use converged value from Workflow 1
        "EDIFF": 1e-7,
        "NSW": 0,
        "IBRION": -1,
        "ISMEAR": 0,
        "SIGMA": 0.05,
        "LWAVE": False,
        "LCHARG": False,
        "PREC": "Accurate",
        "ALGO": "Normal",
    })
    # Override the k-points
    kpoints = Kpoints.gamma_automatic(grid, (0, 0, 0))
    vis.write_input(str(calc_dir))
    kpoints.write_file(str(calc_dir / "KPOINTS"))

# Parse and report
results = {}
for density in target_densities:
    grid = tuple(max(1, int(np.ceil(L / density))) for L in recip_lengths)
    label = "x".join(map(str, grid))
    path = Path(f"conv_kpts_{label}/vasprun.xml")
    if not path.exists():
        continue
    v = Vasprun(str(path))
    natoms = len(v.final_structure)
    nkpts = v.actual_kpoints_weights.__len__() if hasattr(v, "actual_kpoints_weights") else None
    results[label] = {
        "grid": grid,
        "nkpts_irreducible": nkpts,
        "energy_per_atom": v.final_energy / natoms,
    }

labels = list(results)
print(f"{'Grid':>14}  {'NK_irred':>10}  {'E/atom (eV)':>14}  {'ΔE (meV/atom)':>15}")
for i, lab in enumerate(labels):
    de = (results[lab]["energy_per_atom"] - results[labels[i-1]]["energy_per_atom"]) * 1000 if i > 0 else 0.0
    nk = results[lab]["nkpts_irreducible"] or "?"
    print(f"{lab:>14}  {nk:>10}  {results[lab]['energy_per_atom']:>14.6f}  {de:>15.3f}")
```

**Property-dependent k-mesh guidance:**

| Property | Guidance | Notes |
|----------|----------|-------|
| Bulk energy (insulator/semiconductor) | Converge to < 1 meV/atom; typically 4–8 pts per Å⁻¹ | Fewer k-points needed than metals |
| Bulk energy (metal) | Converge to < 1 meV/atom; typically 8–16 pts per Å⁻¹ | Fermi surface sampling is critical |
| Forces for relaxation | Same as bulk energy tolerance | Energy differences between ionic steps |
| Stress / EOS | Same as bulk energy; verify with ENCUT | Pulay stress can dominate k-mesh error |
| DOS and Fermi surface | 2–4× denser than energy-converged mesh | Use tetrahedron method (ISMEAR=-5) |
| Band structure | High-symmetry path; use charge density from separate dense MP run | k-path sampling separate from MP grid |
| Phonons (DFPT or finite displacement) | 2× denser than energy; test individual phonon branches | Acoustic sum rule violations signal under-sampling |
| Surface slab energy | 1D: Gamma-only along surface normal; 2D: converge in-plane | Surface k-mesh decoupled from bulk |
| Adsorption energy | Converge the difference; same mesh for slab+adsorbate and bare slab | Absolute error cancels; relative error must be < 5 meV |
| ML potential training | Same mesh for all structures; use slightly denser than energy-converged | Inconsistent meshes corrupt force labels |

### Workflow 3: Smearing scheme and SIGMA selection

Choosing the wrong smearing scheme produces wrong energies, wrong DOS, or poor SCF convergence.

```python
# Decision tree for smearing scheme selection:
#
# 1. Is the system definitely an insulator or semiconductor (band gap > ~0.5 eV)?
#    -> ISMEAR = 0, SIGMA = 0.01-0.05 eV  (Gaussian, small sigma)
#    -> ISMEAR = -5 is better for final energy if k-mesh is dense (>= ~6x6x6)
#    -> Do NOT use ISMEAR = -5 with fewer than ~6 k-points per direction or with metals
#
# 2. Is the system a metal?
#    -> ISMEAR = 1 or 2, SIGMA = 0.10-0.20 eV  (Methfessel-Paxton)
#    -> For noble metals (Au, Ag, Cu): SIGMA = 0.05-0.10 eV [EXPERT REVIEW NEEDED]
#    -> For transition metals with sharp DOS features: use ISMEAR=1, test SIGMA
#    -> Do NOT use ISMEAR = -5 for metals
#
# 3. Is the system of unknown character (e.g., high-throughput screening)?
#    -> ISMEAR = 0, SIGMA = 0.05 eV  (safe universal choice)
#    -> Post-hoc: check if NELEC converges non-integer; if gap exists, switch to ISMEAR=-5 for final run
#
# 4. Is the system magnetic or near a metal-insulator transition?
#    -> Start with ISMEAR = 0, SIGMA = 0.05 eV
#    -> After convergence, check character and refine
#
# SIGMA validation: the free energy F = E - T*S should approach E0 (total energy
# extrapolated to T=0) as SIGMA -> 0. In VASP, E0 is reported in OUTCAR.
# If |E - E0| > 1 meV/atom, SIGMA is too large.

# SIGMA scan example (VASP)
sigma_values = [0.20, 0.10, 0.05, 0.02, 0.01]
for sigma in sigma_values:
    # Run single-point with each SIGMA, read E and E0 from OUTCAR
    pass

# In OUTCAR, look for:
#   energy  without entropy=   -12.345678  energy(sigma->0) =   -12.345670
# The difference |E_without_entropy - E(sigma->0)| / N_atoms should be < 1 meV/atom.
# If it is not, reduce SIGMA.
```

**Quantum ESPRESSO equivalents:**

```python
# QE smearing settings (in &ELECTRONS / &SYSTEM namelist):
#
# Metals:
#   occupations = 'smearing'
#   smearing = 'mp'          ! Methfessel-Paxton
#   degauss = 0.01           ! Ry (~ 0.14 eV); test convergence
#
# Insulators (final energy):
#   occupations = 'smearing'
#   smearing = 'tetrahedra_opt'  ! Blöchl tetrahedron (requires ibrav/k-grid setup)
#   OR:
#   occupations = 'fixed'        ! Only if gap is guaranteed
#
# Safe universal (screening):
#   occupations = 'smearing'
#   smearing = 'gaussian'
#   degauss = 0.005          ! Ry (~ 0.07 eV)
#
# QE degauss units are Rydberg (1 Ry = 13.606 eV).
# A degauss of 0.01 Ry ~ 0.14 eV. Do not confuse with VASP SIGMA in eV.
```

### Workflow 4: ecutwfc and ecutrho convergence (Quantum ESPRESSO)

QE requires both the wavefunction cutoff (ecutwfc) and the charge density cutoff (ecutrho) to be set and converged.

```python
# Minimal QE convergence scan using ASE + subprocess

from pathlib import Path
import subprocess

QE_TEMPLATE = """\
&CONTROL
  calculation = 'scf'
  outdir = './out'
  prefix = 'conv'
  pseudo_dir = '{pseudo_dir}'
/
&SYSTEM
  ibrav = 0
  nat = {nat}
  ntyp = {ntyp}
  ecutwfc = {ecutwfc}
  ecutrho = {ecutrho}
  occupations = 'smearing'
  smearing = 'mp'
  degauss = 0.01
/
&ELECTRONS
  conv_thr = 1.0d-9
  mixing_beta = 0.5
/
ATOMIC_SPECIES
{atomic_species}
ATOMIC_POSITIONS angstrom
{positions}
K_POINTS automatic
{kgrid}
CELL_PARAMETERS angstrom
{cell}
"""

pseudo_dir = "/path/to/pseudopotentials"  # Set to your SSSP or PseudoDojo location
nat = 2
ntyp = 1

# For norm-conserving pseudopotentials (PseudoDojo NC):
#   ecutrho = 4 * ecutwfc  (automatic)
# For ultrasoft (SSSP Efficiency):
#   ecutrho = 8-12 * ecutwfc  (must be tested)

ecutwfc_values = [40, 50, 60, 70, 80, 90]  # Ry

for ewfc in ecutwfc_values:
    erho = 8 * ewfc   # For ultrasoft PP; use 4*ewfc for NC [EXPERT REVIEW NEEDED for specific PP]
    calc_dir = Path(f"conv_ewfc_{ewfc}")
    calc_dir.mkdir(exist_ok=True)
    inp = QE_TEMPLATE.format(
        pseudo_dir=pseudo_dir,
        nat=nat, ntyp=ntyp,
        ecutwfc=ewfc, ecutrho=erho,
        atomic_species="Fe  55.845  Fe.pbesol-spn-kjpaw_psl.1.0.0.UPF",
        positions="Fe  0.0  0.0  0.0\nFe  1.435  1.435  1.435",
        kgrid="8 8 8  0 0 0",
        cell="2.87 0.0 0.0\n0.0 2.87 0.0\n0.0 0.0 2.87",
    )
    (calc_dir / "scf.in").write_text(inp)
    # subprocess.run(["pw.x", "-in", "scf.in"], cwd=calc_dir, ...)

# Parse energy from pw.x output:
# grep "!    total energy" conv_ewfc_*/scf.out
# Look for convergence: |ΔE/atom| < 1 meV/atom (= 0.0735 mRy/atom)
```

### Workflow 5: Convergence for specific downstream properties

Some workflows need tighter or different convergence criteria than standard bulk energy convergence.

```python
# --- PHONONS (finite displacement or DFPT) ---
# Forces must be converged, not just energies.
# Recommended settings for phonon calculations:
phonon_incar = {
    "ENCUT": 550,           # Higher than energy-converged; Pulay stress affects dynamical matrix
    "EDIFF": 1e-8,          # Tight SCF; loose EDIFF corrupts forces via Pulay forces
    "PREC": "Accurate",
    "NSW": 0,
    "IBRION": -1,
    "ISMEAR": 0,
    "SIGMA": 0.01,          # Small sigma; phonons sensitive to smearing via band occupation
    "ALGO": "Normal",
    "LREAL": False,          # Real-space projection off: required for accurate forces
    # Note: LREAL = Auto or True introduces small errors in forces acceptable for
    # relaxation but not for phonon dynamical matrices. [EXPERT REVIEW NEEDED]
}
# k-mesh for phonons: 2x denser than energy-converged mesh in each direction.
# Validate: acoustic modes at Gamma should be < 1 cm⁻¹ (ideally < 0.1 cm⁻¹).

# --- BAND GAPS ---
# Band gaps (from eigenvalues) are not variational. They converge with k-mesh
# but not necessarily monotonically.
# - Use HSE06 or GW for accurate gaps; GGA gaps are systematically underestimated.
# - k-mesh for gap: use band structure k-path + dense SCF mesh.
# - ENCUT: standard energy convergence is usually sufficient.

# --- ADSORPTION ENERGIES ---
# E_ads = E(slab+adsorbate) - E(slab) - E(adsorbate_reference)
# Both slab and adsorbate must use identical ENCUT, POTCAR, and ISMEAR.
# k-mesh: Gamma-only along surface normal; converge in-plane independently.
# Vacuum: >= 15 Å; test with dipole correction (LDIPOL=True, IDIPOL=3).
# Counterpoise correction for BSSE is rarely done in plane-wave codes but
# BSSE is much smaller than in Gaussian basis sets. [EXPERT REVIEW NEEDED]

# --- ML POTENTIAL TRAINING DATA ---
# The most important convergence requirement is CONSISTENCY, not absolute tightness.
# All structures in a training set must use identical:
#   - ENCUT
#   - POTCAR (same version, same PP choice per element)
#   - ISMEAR and SIGMA
#   - EDIFF
# Mixing settings introduces systematic energy offsets that appear as noise in force training.
mlp_incar = {
    "ENCUT": 520,
    "EDIFF": 1e-6,       # Tight enough for accurate forces; not so tight it's slow
    "PREC": "Accurate",
    "NSW": 0,
    "IBRION": -1,
    "ISMEAR": 0,
    "SIGMA": 0.10,       # Slightly larger sigma acceptable; forces less sensitive than DOS
    "ISIF": 2,           # Compute stress tensor
    "LWAVE": False,
    "LCHARG": False,
    "ALGO": "Normal",
    "LREAL": "Auto",     # Acceptable for large supercells (>= ~50 atoms)
}
```

## Common Failure Modes

| Failure | Symptom | Likely Cause | Diagnostic | Fix |
|---------|---------|--------------|------------|-----|
| Pulay stress artifact | Relaxed volume too large or small; residual stress after convergence; lattice parameters deviate from experiment | ENCUT too close to POTCAR ENMAX; basis set changes as cell deforms | Check stress tensor in OUTCAR; increase ENCUT by 30% and re-relax; compare lattice parameters | Set ENCUT ≥ 1.3 × ENMAX; for stress-sensitive properties use even higher ENCUT |
| Non-monotonic k-mesh convergence | Energy oscillates as k-mesh increases; no clear convergence plateau | Symmetry breaking between meshes; Gamma vs MP centering inconsistency; band crossing through Fermi level | Plot E vs. mesh density; check that all meshes use same centering; examine band structure near Fermi level | Switch to Gamma-centered mesh; test both centeringOptions; increase mesh more slowly |
| ISMEAR=-5 on a metal | Wildly wrong total energy; negative DOS occupations; failed SCF | Tetrahedron method applied to a system with a Fermi surface or sparse k-mesh | Check system is insulating/semiconducting; count irreducible k-points | Switch to ISMEAR=1, SIGMA=0.10-0.20 for metals |
| SIGMA too large (thermal smearing error) | |E - E0| / N_atoms > 1 meV/atom in VASP OUTCAR; free energy and total energy differ | SIGMA set too high relative to electronic temperature scale of the system | Check OUTCAR: `energy without entropy` vs `energy(sigma->0)`; compute the difference | Reduce SIGMA by 2x and retest; typical safe range: 0.01–0.05 eV for insulators |
| Inconsistent ENCUT across training dataset | ML potential training loss does not decrease; force RMSE remains high; energy labels show systematic scatter | Different calculations in the dataset used different ENCUT or POTCAR versions | Grep ENCUT from all INCAR files; compare POTCAR TITEL strings | Re-run offending calculations with uniform settings; never mix POTCAR vintages |
| Incorrect ecutrho for ultrasoft PP (QE) | QE SCF fails to converge; charge density oscillations; forces wrong | ecutrho too small for ultrasoft pseudopotential | Check PP header for recommended ecutrho; compare to input; try ecutrho = 10 × ecutwfc | Set ecutrho ≥ 8 × ecutwfc for ultrasoft PP; retest convergence |
| k-mesh symmetry mismatch on hexagonal cell | Wrong energy; VASP warning about symmetry reduction; Fermi surface features missing | Off-Gamma Monkhorst-Pack mesh on hexagonal or trigonal lattice | Check KPOINTS file for Gamma vs. MP centering | Use Gamma-centered grid (KGAMMA = .TRUE. or `G` in KPOINTS); re-run convergence scan |
| Convergence plateau not reached (too narrow scan) | Apparently converged at low ENCUT but energy shifts again at higher values; common for f-electron systems | ENCUT scan range too narrow; PP with high ENMAX | Extend scan to 1.6–1.8 × ENMAX for actinides and lanthanides | Extend ENCUT scan range; lanthanide and actinide PAW datasets often require ENCUT ≥ 600–800 eV [EXPERT REVIEW NEEDED] |

## Best Practices

- **Always report the converged ENCUT and k-mesh** in publications and data deposits. Stating only the code name and functional is insufficient for reproducibility.
- **Keep ENCUT identical across all calculations in a thermodynamic cycle.** A reaction energy A → B is only meaningful if A and B were computed at the same ENCUT. This applies even when A and B have different compositions, unless the error is known to cancel.
- **Use the same POTCAR version across an entire project.** VASP PAW datasets change between releases (PBE.52, PBE.54, PBE.64). Mixing versions introduces systematic energy offsets that corrupt formation energy calculations.
- **For energy differences, test convergence on the difference, not on each total energy separately.** Adsorption energies, reaction energies, and defect formation energies converge faster than absolute totals because systematic errors cancel.
- **Test SIGMA independently from ENCUT.** Reducing SIGMA while ENCUT is too low can give a false appearance of convergence; both must be converged to their respective tolerances.
- **For high-throughput workflows, define and freeze a validated parameter set per element group** rather than running full convergence for every structure. Validate the set on representative structures from each class (oxide, nitride, intermetallic, etc.).
- **For surface slabs, converge the k-mesh in the in-plane directions independently** from the bulk. The out-of-plane direction can usually be sampled at Gamma only (1 k-point) once the vacuum is large enough.
- **Validate LREAL settings carefully for forces.** LREAL=Auto or True is faster for large cells but introduces projection errors. For phonon calculations or ML training data generation, use LREAL=False unless the cell is very large (≥ 100 atoms) and force accuracy has been validated. [EXPERT REVIEW NEEDED]
- **Record absolute and E0 energies for metals.** In VASP, `energy without entropy` is the free energy F and `energy(sigma->0)` is the extrapolated T→0 energy. Always use E0 for thermodynamic comparisons.

## Integration with Other Skills

- **[vasp-workflow](../vasp-workflow/SKILL.md):** This skill provides the convergence testing that must precede any production VASP run described in that skill. ENCUT and k-mesh values established here feed directly into the INCAR and KPOINTS files used there.
- **[quantum-espresso](../quantum-espresso/SKILL.md):** Workflows 3 and 4 above cover QE-specific convergence parameters (ecutwfc, ecutrho, degauss). The conceptual framework is identical; only the input syntax and typical parameter ranges differ.
- **[dft-phonons](../dft-phonons/SKILL.md):** Phonon calculations are among the most sensitive to basis-set and k-mesh incompleteness. The tighter ENCUT, EDIFF, and k-mesh recommendations in Workflow 5 apply directly to finite-displacement and DFPT inputs described there.
- **[dft-surfaces](../dft-surfaces/SKILL.md):** Surface slab convergence requires separate k-mesh testing in the in-plane directions, vacuum convergence testing, and dipole correction choices that interact with the ENCUT and smearing settings established here.
- **[dft-defects](../dft-defects/SKILL.md):** Defect formation energies use large supercells where k-mesh requirements drop (often Gamma-only), but ENCUT requirements do not. Charge corrections depend on accurate dielectric constants, which themselves require tight convergence.
- **[high-throughput-dft](../high-throughput-dft/SKILL.md):** High-throughput frameworks fix convergence parameters per material class. This skill provides the validation methodology for establishing and auditing those fixed parameter sets.
- **[mlp-validation](../../ml-interatomic-potentials/mlp-validation/SKILL.md):** The consistency of DFT convergence settings across a training dataset is one of the first checks in MLP validation. Force and energy errors that appear to be MLP failures often originate from inconsistent DFT reference settings.

## Key References

1. Kresse, G. & Furthmüller, J. (1996). *Efficient iterative schemes for ab initio total-energy calculations using a plane-wave basis set.* Physical Review B, 54(16), 11169. https://doi.org/10.1103/PhysRevB.54.11169
2. Methfessel, M. & Paxton, A. T. (1989). *High-precision sampling for Brillouin-zone integration in metals.* Physical Review B, 40(6), 3616. https://doi.org/10.1103/PhysRevB.40.3616
3. Blöchl, P. E., Jepsen, O., & Andersen, O. K. (1994). *Improved tetrahedron method for Brillouin-zone integrations.* Physical Review B, 49(23), 16223. https://doi.org/10.1103/PhysRevB.49.16223
4. Lejaeghere, K., et al. (2016). *Reproducibility in density functional theory calculations of solids.* Science, 351(6280), aad3000. https://doi.org/10.1126/science.aad3000
5. Prandini, G., et al. (2018). *Precision and efficiency in solid-state pseudopotential calculations.* npj Computational Materials, 4, 72. https://doi.org/10.1038/s41524-018-0127-2
