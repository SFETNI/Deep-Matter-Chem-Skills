---
name: cp2k-workflow
domain: electronic-structure
maturity: draft
tools:
  - CP2K
  - ASE
  - pymatgen
  - numpy
  - matplotlib
tested_versions:
  CP2K: "2023.2"
  ASE: "3.23.0"
  pymatgen: "2024.11.13"
  numpy: "1.26.4"
requires_gpu: false
estimated_runtime: "minutes-to-days (DFT SCF: minutes; AIMD: hours–days)"
---

# CP2K Workflow

## Description

This skill covers end-to-end DFT, AIMD, and atomistic simulation workflows using CP2K, from input file construction through convergence testing, geometry optimization, and ab initio molecular dynamics. CP2K uses the Gaussian and Plane Waves (GPW) method, which combines a Gaussian-type orbital (GTO) basis with an auxiliary plane-wave density grid, making it efficient for large systems, liquid-phase simulations, and condensed-phase AIMD. Invoke this skill when setting up CP2K calculations, selecting basis sets and GTH pseudopotentials, converging CUTOFF and REL_CUTOFF parameters, diagnosing SCF convergence failures, generating AIMD trajectories, or producing DFT reference data for ML potential training with an open-source code.

## Domain Context

CP2K solves the Kohn-Sham DFT equations using the GPW (Gaussian and Plane Waves) or GAPW (Gaussian Augmented Plane Waves) method. The key idea is a dual representation of the electron density: the wavefunction and density matrix are represented in a localized Gaussian basis, while the Hartree potential is evaluated on an auxiliary plane-wave grid using fast Fourier transforms. This avoids the O(N³) cost of dielectric screening in pure plane-wave codes for very large systems, and enables linear-scaling DFT (LS-DFT) via the linear-scaling module.

Physical approximations and structural concepts that must be understood before using this workflow:

- **GPW method:** The total electron density is expanded as a sum of atom-centered Gaussian functions, then mapped onto a regular real-space grid at a grid spacing controlled by `CUTOFF` (in Ry). The Hartree energy is evaluated in reciprocal space via FFT. The accuracy of this mapping depends on both `CUTOFF` and `REL_CUTOFF` — both must be converged explicitly. Unlike VASP or QE, a high `CUTOFF` alone is insufficient if `REL_CUTOFF` is too low.
- **GAPW method:** Augments GPW with atom-centered hard and soft auxiliary density expansions near nuclei (analogous to PAW), allowing all-electron calculations or harder core regions without increasing the plane-wave grid cost. Used for core-level spectroscopy (XPS, XANES) and when soft pseudopotentials are insufficient. Higher cost than GPW for the same basis.
- **GTH pseudopotentials:** Goedecker-Teter-Hutter norm-conserving pseudopotentials are the standard for CP2K. They are available for LDA, PBE, BLYP, PBE0, and several other functionals. The pseudopotential and basis set must be generated for and matched to the same XC functional — using a GTH-PBE pseudopotential with a BLYP functional is inconsistent and produces wrong results.
- **Gaussian basis sets:** CP2K ships MOLOPT-optimized basis sets (short-range `SR` variants for solids, standard for molecules). Quality tiers: SZV (minimal, screening only), DZVP (double-ζ + polarization, standard production), TZV2P (triple-ζ, higher accuracy), TZV2PX (extended). Short-range SR basis sets are preferred for periodic solids because they minimize linear dependence and basis set superposition error in condensed phases.
- **CUTOFF (Ry):** Controls the plane-wave grid density for the auxiliary density. Analogous to ecutrho in QE but expressed per-calculation, not per-element. Higher CUTOFF increases grid density and accuracy of the Hartree term. Typical production values: 400–800 Ry for most GTH basis sets.
- **REL_CUTOFF (Ry):** Controls the accuracy of the Gaussian-to-grid mapping for each atom type. Each Gaussian is mapped onto a subgrid whose fineness is determined by REL_CUTOFF. If REL_CUTOFF is too small, Gaussian functions are mapped poorly regardless of CUTOFF. Default: 60 Ry; recommended production: 60–100 Ry. Converge by scanning REL_CUTOFF at fixed CUTOFF.
- **OT (Orbital Transformation) method:** An iterative direct minimization approach for finding the ground-state wavefunction without diagonalizing the Kohn-Sham matrix. Efficient and stable for insulators and semiconductors with a band gap. Requires a good preconditioner. Cannot handle fractional occupations and must not be used for metallic systems. `OUTER_SCF` is required alongside OT to converge the total energy, not just the orbital gradient.
- **Diagonalization + smearing for metals:** For metallic systems, OT must be replaced with standard diagonalization (`&DIAGONALIZATION ON`) plus `ADDED_MOS` (extra molecular orbitals above the HOMO) and Fermi-Dirac or Gaussian smearing. Broyden or Pulay mixing is then required for charge density convergence.
- **Periodic boundary conditions and Poisson solver:** CP2K supports fully periodic (3D), slab (2D), wire (1D), and isolated (0D) boundary conditions via the `POISSON` section. The correct Poisson solver must match the periodicity: `PERIODIC` for bulk, Martyna-Tuckerman (`MT`) for isolated molecules or charged slabs, and wavelet-based `WAVELET` for general non-orthorhombic cells with mixed periodicity.
- **AIMD integration:** CP2K uses the Verlet algorithm for NVE, and thermostat-augmented integrators for NVT (CSVR, NOSE) and NPT (BAROSTAT + thermostat). The timestep must be ≤ 1/10 of the period of the fastest vibrational mode — typically 0.5 fs for hydrogen-containing systems, 1–2 fs for heavy-element systems.

## When to Use This Skill

- AIMD simulations of liquids, solutions, interfaces, and disordered systems, where the Gaussian basis and Born-Oppenheimer dynamics are well-suited.
- Large-scale DFT calculations (> 200–1000 atoms) where CP2K's O(N log N) Coulomb solver is more efficient than pure plane-wave codes.
- Linear-scaling DFT for very large systems (> 1000 atoms) using CP2K's LS-DFT module.
- DFT calculations for molecular systems, complex interfaces, or solution-phase chemistry where the Gaussian basis is advantageous.
- Open-source DFT data generation for ML potential training, as an alternative or complement to QE or VASP.
- Hybrid functional calculations (PBE0, HSE06) with ADMM acceleration, which is often faster than VASP for large systems.
- Core-level XPS and XANES spectra using the GAPW method.
- Condensed-phase thermodynamics via metadynamics or enhanced sampling (CP2K integrates with PLUMED).

## When Not to Use This Skill

- **Routine plane-wave DFT benchmarking** where VASP or QE are the established standard — CP2K energy comparisons require careful basis-set-superposition correction and are not directly comparable to plane-wave energies without cross-validation.
- **Band structure calculations along high-symmetry paths** — CP2K's localized basis does not naturally produce dispersion curves; use VASP or QE with a Wannier interpolation instead.
- **Phonon calculations via DFPT** — CP2K lacks a mature DFPT phonon module; use QE (ph.x) or VASP with phonopy/DFPT instead.
- **Strongly correlated systems requiring DMFT or high-level multireference methods** — CP2K's DFT+U is available but limited; use dedicated correlated-electron codes.
- **Systems with very heavy elements (actinides, 5f)** without validated GTH pseudopotentials — pseudopotential availability and accuracy should be verified against all-electron benchmarks. [EXPERT REVIEW NEEDED]
- **High-throughput DFT screening at the > 10,000 structure scale** — CP2K input file generation and output parsing pipelines are less mature than pymatgen/VASP or AiiDA/QE; use those stacks for large-scale screening.

## Core Concepts

- **GPW dual representation:** Wavefunction expanded in GTOs (Gaussian-type orbitals); density mapped to auxiliary PW grid for Coulomb/XC integration. Both CUTOFF and REL_CUTOFF must converge to achieve accurate energies and forces.
- **MOLOPT basis sets:** Optimized to minimize basis set superposition error (BSSE) in condensed phases. `SR` (short-range) variants suppress long-range tails for better linear independence in periodic systems. Always prefer `MOLOPT-SR-GTH` for periodic solids and `MOLOPT-GTH` for molecules.
- **GTH pseudopotential naming:** `GTH-{FUNCTIONAL}-q{NVALENCE}`. The `q` number gives the number of valence electrons. Example: `GTH-PBE-q8` for Fe (3d⁶4s² = 8 valence electrons). The pseudopotential file (`GTH_POTENTIALS`) and basis set file (`BASIS_MOLOPT`) are separate and must be consistently paired.
- **OUTER_SCF:** An outer loop that restarts the OT minimization with updated density after each outer iteration. Required with OT to ensure the total energy (not just the orbital gradient) converges to the requested `EPS_SCF`. Without `OUTER_SCF`, OT may produce a wavefunction satisfying the orbital gradient criterion but with a non-self-consistent Hamiltonian.
- **OT preconditioner:** `FULL_ALL` (full diagonalization-based) is most robust for insulators with large band gaps. `FULL_KINETIC` is faster for medium-gap systems. `FULL_SINGLE_INVERSE` is the default and balances cost and stability. A poor preconditioner choice causes OT to converge slowly or not at all.
- **ADDED_MOS:** Number of additional molecular orbitals beyond the occupied states. Required for metallic systems using diagonalization + smearing. A too-small value causes unoccupied states to fall below the Fermi level, producing an unconverged electron density.
- **ADMM (Auxiliary Density Matrix Method):** Approximates the Hartree-Fock exchange by projecting it onto a smaller auxiliary basis, reducing hybrid functional cost from O(N⁴) toward O(N³) or lower. Controlled by the `AUXILIARY_DENSITY_MATRIX_METHOD` subsection. The ADMM basis must be smaller than the primary basis and chosen to minimize the exchange approximation error.
- **Poisson solver:** The `MT` (Martyna-Tuckerman) solver enforces zero potential boundary conditions for isolated systems and charged slabs by extending the cell artificially. The `PERIODIC` solver assumes 3D periodicity and wraps the potential. Using the wrong solver for a charged system or a molecule-in-vacuum causes electrostatic errors that do not converge with cell size.
- **Restart logic:** CP2K restart files (`.wfn` for wavefunction, `.xyz` and `.vel` for AIMD state) must match the system size, basis set, and calculation type. Restarting from a wavefunction generated with a different basis or cell produces noise or divergence.

## Key Workflows

### Workflow 1: CP2K input anatomy — DFT SCF calculation

The CP2K input is hierarchical, using `&SECTION ... &END SECTION` blocks. The minimum input for a DFT energy calculation requires `GLOBAL`, `FORCE_EVAL`, `DFT`, and `SUBSYS` sections. All energy units are Hartree internally; CUTOFF and REL_CUTOFF are in Ry; distances and cell parameters in Å by default (or Bohr with explicit tag).

```
# cp2k_scf.inp — annotated DFT single-point energy for an oxide (e.g., MgO, 8 atoms)
# Run: cp2k.popt -i cp2k_scf.inp -o cp2k_scf.out

&GLOBAL
  PROJECT MgO_scf
  RUN_TYPE ENERGY          # ENERGY | GEO_OPT | CELL_OPT | MD | VIBRATIONAL_ANALYSIS
  PRINT_LEVEL MEDIUM       # LOW | MEDIUM | HIGH; MEDIUM gives convergence diagnostics
&END GLOBAL

&FORCE_EVAL
  METHOD Quickstep         # GPW/GAPW DFT via Quickstep module
  STRESS_TENSOR ANALYTICAL # Required for CELL_OPT and ML training data

  &DFT
    BASIS_SET_FILE_NAME  BASIS_MOLOPT          # shipped with CP2K
    POTENTIAL_FILE_NAME  GTH_POTENTIALS        # shipped with CP2K

    &MGRID
      CUTOFF      600     # Ry; plane-wave density grid cutoff — must be converged
      REL_CUTOFF   60     # Ry; Gaussian-to-grid mapping accuracy — must be converged
      NGRIDS        4     # Number of multi-grid levels; 4 is standard
    &END MGRID

    &QS
      METHOD GPW          # GPW (default) or GAPW (all-electron-like)
      EPS_DEFAULT 1.0E-12 # Global accuracy threshold for integrals
      EPS_PGF_ORB 1.0E-32 # Threshold for Gaussian orbital screening; tight for forces
    &END QS

    &SCF
      SCF_GUESS ATOMIC    # ATOMIC | RESTART | RANDOM; ATOMIC is robust for cold starts
      EPS_SCF   1.0E-6    # Electronic convergence threshold (Hartree); tighten to 1E-7
                          # for forces in ML training data or phonon-quality forces
      MAX_SCF   50        # Inner OT iterations per outer SCF cycle

      &OT ON              # Orbital Transformation — use for insulators and semiconductors
        MINIMIZER  DIIS              # DIIS | CG; DIIS is faster near convergence
        PRECONDITIONER FULL_ALL      # FULL_ALL | FULL_KINETIC | FULL_SINGLE_INVERSE
                                     # FULL_ALL: best for wide-gap insulators; expensive
                                     # FULL_KINETIC: faster for medium-gap systems
        ENERGY_GAP 0.1              # Estimated HOMO-LUMO gap (Hartree); helps preconditioner
      &END OT

      &OUTER_SCF ON       # Required with OT; converges total energy not just orbital gradient
        EPS_SCF   1.0E-6  # Must match or be looser than inner EPS_SCF
        MAX_SCF   20      # Outer restart iterations; increase to 50 for difficult cases
      &END OUTER_SCF

      &PRINT
        &RESTART
          FILENAME MgO_scf
          BACKUP_COPIES 2
          &EACH
            QS_SCF 0      # Write restart wavefunction after SCF converges
          &END EACH
        &END RESTART
      &END PRINT
    &END SCF

    &XC
      &XC_FUNCTIONAL PBE
      &END XC_FUNCTIONAL
      &VDW_POTENTIAL      # D3 dispersion — recommended for most systems
        POTENTIAL_TYPE PAIR_POTENTIAL
        &PAIR_POTENTIAL
          TYPE DFTD3
          PARAMETER_FILE_NAME dftd3.dat    # shipped with CP2K
          REFERENCE_FUNCTIONAL PBE
          R_CUTOFF 15.0   # Å; interaction cutoff
          D3_SCALING_PARAMETERS            # Becke-Johnson damping (default)
          CALCULATE_C9_TERM .TRUE.
        &END PAIR_POTENTIAL
      &END VDW_POTENTIAL
    &END XC

    &POISSON
      PERIODIC XYZ        # XYZ for bulk; NONE for isolated molecules; XY for slabs
      PSOLVER  PERIODIC   # PERIODIC | MT | WAVELET; must match PERIODIC above
    &END POISSON

    &PRINT
      &MO_CUBES OFF
      &END MO_CUBES
    &END PRINT
  &END DFT

  &SUBSYS
    &CELL
      ABC [angstrom] 4.212 4.212 4.212   # Cell lengths; use explicit unit tag
      ANGLES 90.0 90.0 90.0
      PERIODIC XYZ
    &END CELL

    &TOPOLOGY
      COORD_FILE_NAME  MgO.xyz
      COORD_FILE_FORMAT XYZ
    &END TOPOLOGY

    # One &KIND section per element species
    &KIND Mg
      BASIS_SET  DZVP-MOLOPT-SR-GTH         # Double-ζ + polarization, short-range
      POTENTIAL  GTH-PBE-q2                 # Mg: 2 valence electrons (3s²)
    &END KIND
    &KIND O
      BASIS_SET  DZVP-MOLOPT-SR-GTH
      POTENTIAL  GTH-PBE-q6                 # O: 6 valence electrons (2s²2p⁴)
    &END KIND
  &END SUBSYS
&END FORCE_EVAL
```

Parsing CP2K output with ASE:

```python
from ase.io import read
from pathlib import Path

# ASE reads CP2K output files directly
atoms = read("cp2k_scf.out", format="cp2k-output")
print(f"Energy: {atoms.get_potential_energy():.6f} eV")
print(f"Max force: {atoms.get_forces().max():.5f} eV/Å")

# For AIMD trajectories, read the .xyz file written by CP2K
trajectory = read("MgO-pos-1.xyz", index=":", format="extxyz")
print(f"Number of frames: {len(trajectory)}")

# pymatgen can read CP2K output via:
from pymatgen.io.cp2k.outputs import Cp2kOutput
out = Cp2kOutput("cp2k_scf.out", auto_load=True)
print(f"Final energy: {out.final_energy} eV")
print(f"Converged: {out.converged}")
```

### Workflow 2: CUTOFF and REL_CUTOFF convergence testing

Both CUTOFF and REL_CUTOFF must be converged independently. CUTOFF controls the plane-wave grid density; REL_CUTOFF controls the Gaussian mapping accuracy. A common mistake is converging only CUTOFF while leaving REL_CUTOFF at the default 60 Ry, which under-resolves contracted Gaussians for some basis sets.

Recommended protocol:
1. Fix REL_CUTOFF = 60 Ry; scan CUTOFF from 200 to 1000 Ry.
2. Fix CUTOFF at the converged value; scan REL_CUTOFF from 40 to 120 Ry.
3. Accept the lowest values where the total energy change is < 1 meV/atom.

```python
import subprocess
from pathlib import Path

TEMPLATE = """\
&GLOBAL
  PROJECT conv_{tag}
  RUN_TYPE ENERGY
  PRINT_LEVEL LOW
&END GLOBAL

&FORCE_EVAL
  METHOD Quickstep
  &DFT
    BASIS_SET_FILE_NAME BASIS_MOLOPT
    POTENTIAL_FILE_NAME GTH_POTENTIALS
    &MGRID
      CUTOFF     {cutoff}
      REL_CUTOFF {rel_cutoff}
      NGRIDS 4
    &END MGRID
    &QS
      METHOD GPW
      EPS_DEFAULT 1.0E-12
    &END QS
    &SCF
      SCF_GUESS ATOMIC
      EPS_SCF 1.0E-7
      MAX_SCF 50
      &OT ON
        MINIMIZER DIIS
        PRECONDITIONER FULL_ALL
      &END OT
      &OUTER_SCF ON
        EPS_SCF 1.0E-7
        MAX_SCF 20
      &END OUTER_SCF
    &END SCF
    &XC
      &XC_FUNCTIONAL PBE
      &END XC_FUNCTIONAL
    &END XC
    &POISSON
      PERIODIC XYZ
    &END POISSON
  &END DFT
  &SUBSYS
    &CELL
      ABC [angstrom] 4.212 4.212 4.212
    &END CELL
    &TOPOLOGY
      COORD_FILE_NAME MgO.xyz
      COORD_FILE_FORMAT XYZ
    &END TOPOLOGY
    &KIND Mg
      BASIS_SET DZVP-MOLOPT-SR-GTH
      POTENTIAL GTH-PBE-q2
    &END KIND
    &KIND O
      BASIS_SET DZVP-MOLOPT-SR-GTH
      POTENTIAL GTH-PBE-q6
    &END KIND
  &END SUBSYS
&END FORCE_EVAL
"""

def run_convergence_scan(cutoff_values: list[int], rel_cutoff: int, label: str):
    results = {}
    for cutoff in cutoff_values:
        tag = f"{label}_c{cutoff}_rc{rel_cutoff}"
        calc_dir = Path(f"conv_{tag}")
        calc_dir.mkdir(exist_ok=True)
        inp = TEMPLATE.format(tag=tag, cutoff=cutoff, rel_cutoff=rel_cutoff)
        (calc_dir / "cp2k.inp").write_text(inp)
        # Copy MgO.xyz into calc_dir or use absolute path
        subprocess.run(
            ["cp2k.popt", "-i", "cp2k.inp", "-o", "cp2k.out"],
            cwd=calc_dir, check=True
        )
        results[cutoff] = parse_cp2k_energy(calc_dir / "cp2k.out")
    return results

def parse_cp2k_energy(outfile: Path) -> float:
    """Extract the final total energy in eV from a CP2K output file."""
    HARTREE_TO_EV = 27.211386245988
    for line in reversed(outfile.read_text().splitlines()):
        if "ENERGY| Total FORCE_EVAL" in line:
            return float(line.split()[-1]) * HARTREE_TO_EV
    raise RuntimeError(f"Energy not found in {outfile}")

# Step 1: CUTOFF scan at REL_CUTOFF = 60 Ry
cutoff_results = run_convergence_scan([200, 300, 400, 500, 600, 800], rel_cutoff=60, label="cutoff_scan")

n_atoms = 8  # MgO unit cell
print(f"{'CUTOFF (Ry)':>12}  {'E/atom (eV)':>14}  {'ΔE (meV/atom)':>15}")
cutoffs = sorted(cutoff_results)
for i, c in enumerate(cutoffs):
    e = cutoff_results[c] / n_atoms
    de = (cutoff_results[c] - cutoff_results[cutoffs[i-1]]) / n_atoms * 1000 if i > 0 else 0.0
    print(f"{c:>12}  {e:>14.6f}  {de:>15.3f}")

# Step 2: REL_CUTOFF scan at converged CUTOFF (e.g., 600 Ry)
rel_results = {}
for rc in [40, 50, 60, 80, 100, 120]:
    tag = f"rel_c600_rc{rc}"
    calc_dir = Path(f"conv_{tag}")
    calc_dir.mkdir(exist_ok=True)
    inp = TEMPLATE.format(tag=tag, cutoff=600, rel_cutoff=rc)
    (calc_dir / "cp2k.inp").write_text(inp)
    subprocess.run(["cp2k.popt", "-i", "cp2k.inp", "-o", "cp2k.out"], cwd=calc_dir, check=True)
    rel_results[rc] = parse_cp2k_energy(calc_dir / "cp2k.out")

# Convergence criterion: |ΔE/atom| < 1 meV/atom between successive values.
# For forces (ML training or phonons): tighten to < 0.1 meV/atom. [EXPERT REVIEW NEEDED]
```

### Workflow 3: SCF convergence — OT, diagonalization, and mixing

The choice between OT (insulators) and diagonalization (metals) is mandatory, not a performance optimization.

```
# For METALLIC systems: replace the OT block with diagonalization + smearing

&SCF
  SCF_GUESS ATOMIC
  EPS_SCF   1.0E-6
  MAX_SCF   300           # Higher than OT; diagonalization SCF may need more iterations

  ADDED_MOS 100           # Extra MOs above the HOMO for smearing; increase if smearing
                          # places some states near or above Fermi level without ADDED_MOS
  &SMEAR ON
    METHOD FERMI_DIRAC
    ELECTRONIC_TEMPERATURE [K] 300   # kT smearing; use 300–1000 K for metals
                                      # Higher T aids convergence but blurs DOS
  &END SMEAR

  &DIAGONALIZATION ON
    ALGORITHM STANDARD    # STANDARD | OT_PRECOND; STANDARD is more robust
  &END DIAGONALIZATION

  &MIXING
    METHOD BROYDEN_MIXING  # BROYDEN_MIXING | PULAY_MIXING | DIRECT_P_MIXING
    ALPHA 0.4              # Mixing coefficient; reduce to 0.1–0.2 for oscillating SCF
    NBUFFER 8              # History length for Broyden mixing
    BETA  1.5              # Broyden inverse Jacobian estimate; default 1.5
  &END MIXING
&END SCF
```

OT preconditioner selection guide:

```python
# OT preconditioner selection (heuristic — always test)
#
# FULL_ALL:
#   Best accuracy and convergence rate for wide-gap insulators (gap > ~3 eV).
#   Cost: requires a full diagonalization step per outer SCF cycle.
#   Use for: oxides, ionic solids, wide-gap semiconductors.
#
# FULL_KINETIC:
#   Cheaper than FULL_ALL; uses kinetic energy operator as preconditioner.
#   Good for medium-gap semiconductors (0.5–3 eV gap).
#   Use for: Si, Ge, narrow-gap oxides.
#   [EXPERT REVIEW NEEDED: crossover gap value is system-dependent]
#
# FULL_SINGLE_INVERSE:
#   CP2K default. Intermediate cost and robustness.
#   Adequate starting point when system character is unknown.
#
# NONE:
#   Unpreconditioned gradient descent. Almost never optimal; kept for debugging.
#
# If OT stalls (energy decreases by < 1E-10 Hartree for > 20 steps without converging):
#   1. Try FULL_ALL (if not already in use)
#   2. Increase OUTER_SCF MAX_SCF to 50
#   3. Reduce ENERGY_GAP to 0.01 (for near-metallic systems that should use OT)
#   4. If none work: the system is likely metallic; switch to diagonalization
```

SCF convergence diagnostic patterns:

```python
def parse_scf_convergence(outfile: str) -> list[dict]:
    """Parse CP2K outer SCF convergence from output file."""
    import re
    pattern = re.compile(
        r"outer SCF iter\s+=\s+(\d+)\s+RMS gradient\s+=\s+(\S+)\s+energy\s+=\s+(\S+)"
    )
    steps = []
    for line in open(outfile):
        m = pattern.search(line)
        if m:
            steps.append({
                "outer_iter": int(m.group(1)),
                "rms_gradient": float(m.group(2)),
                "energy_hartree": float(m.group(3)),
            })
    return steps

steps = parse_scf_convergence("cp2k.out")
for s in steps[-10:]:
    print(f"Iter {s['outer_iter']:3d}  RMS={s['rms_gradient']:.3e}  E={s['energy_hartree']:.8f} Ha")

# Healthy convergence: RMS gradient decreases monotonically by ~10x every few outer steps.
# Oscillating RMS gradient: mixing coefficient too large (reduce ALPHA from 0.4 to 0.2).
# Flat RMS gradient: wrong preconditioner or metallic system using OT — switch algorithm.
# Sudden energy jump: SCF_GUESS RESTART reading incompatible wavefunction — use ATOMIC.
```

### Workflow 4: Geometry optimization (GEO_OPT and CELL_OPT)

CP2K supports atomic position relaxation (`GEO_OPT`) and simultaneous position + cell relaxation (`CELL_OPT`), both via the BFGS optimizer.

```
&GLOBAL
  PROJECT MgO_relax
  RUN_TYPE GEO_OPT         # GEO_OPT | CELL_OPT | MD | ENERGY
&END GLOBAL

&MOTION
  &GEO_OPT
    TYPE MINIMIZATION       # MINIMIZATION | TRANSITION_STATE (for NEB/dimer)
    OPTIMIZER BFGS          # BFGS | CG | LBFGS; BFGS is standard
    MAX_ITER 200
    MAX_FORCE 4.5E-4        # Hartree/Bohr (~0.023 eV/Å); tighten to 1E-4 for phonons
    RMS_FORCE 3.0E-4        # Hartree/Bohr; convergence when both MAX_FORCE and RMS_FORCE met
    MAX_DR    3.0E-3        # Maximum step size (Bohr)
    RMS_DR    1.5E-3
    &BFGS
      TRUST_RADIUS 0.2      # Initial trust radius (Bohr); reduce to 0.1 for soft structures
    &END BFGS
  &END GEO_OPT

  &PRINT
    &TRAJECTORY
      FILENAME MgO_relax
      &EACH
        GEO_OPT 1
      &END EACH
    &END TRAJECTORY
    &FORCES
      &EACH
        GEO_OPT 1
      &END EACH
    &END FORCES
    &RESTART
      BACKUP_COPIES 3
      &EACH
        GEO_OPT 5
      &END EACH
    &END RESTART
  &END PRINT
&END MOTION
```

For variable-cell (CELL_OPT) relaxation including lattice parameters:

```
# Replace RUN_TYPE GEO_OPT with CELL_OPT and add in &MOTION:
&CELL_OPT
  TYPE DIRECT_CELL_OPT    # DIRECT_CELL_OPT | GEO_OPT (iterative); DIRECT is standard
  OPTIMIZER BFGS
  MAX_ITER 200
  MAX_FORCE 4.5E-4        # Hartree/Bohr
  PRESSURE_TOLERANCE 100  # bar; convergence on stress for CELL_OPT
  KEEP_ANGLES .FALSE.     # .TRUE. constrains cell shape (angles fixed)
  KEEP_SYMMETRY .FALSE.   # .TRUE. enforces symmetry during cell relaxation [EXPERT REVIEW NEEDED]
  &BFGS
    TRUST_RADIUS 0.2
  &END BFGS
&END CELL_OPT
```

Post-optimization checks using ASE:

```python
from ase.io import read
import numpy as np

# Read all relaxation steps
traj = read("MgO_relax-pos-1.xyz", index=":", format="extxyz")

# Check force convergence
forces_per_step = []
for atoms in traj:
    if atoms.calc is not None and "forces" in atoms.calc.results:
        fmax = np.sqrt((atoms.get_forces()**2).sum(axis=1)).max()
        forces_per_step.append(fmax)

print(f"Initial max force: {forces_per_step[0]:.4f} eV/Å")
print(f"Final max force:   {forces_per_step[-1]:.4f} eV/Å")
if forces_per_step[-1] > 0.05:
    print("WARNING: Final forces > 0.05 eV/Å — geometry may not be well-converged")
```

### Workflow 5: Ab initio molecular dynamics (AIMD)

AIMD in CP2K uses Born-Oppenheimer dynamics: the electronic structure is fully converged at each ionic step. The cost is one CP2K SCF per timestep.

```
&GLOBAL
  PROJECT water_NVT
  RUN_TYPE MD
  PRINT_LEVEL MEDIUM
&END GLOBAL

&MOTION
  &MD
    ENSEMBLE NVT            # NVE | NVT | NPT_F | NPT_I | NPT_MTK
    STEPS  20000            # Total MD steps
    TIMESTEP [fs] 0.5       # fs; 0.5 fs for H-containing systems; 1-2 fs for heavy atoms
    TEMPERATURE  330        # K target; use 330 K for liquid water (slightly above 300 K
                            # to account for nuclear quantum effects underestimation) [EXPERT REVIEW NEEDED]
    TEMP_TOL    30          # K; tolerance for temperature deviation warning

    &THERMOSTAT
      TYPE CSVR             # CSVR (canonical sampling via velocity rescaling) or NOSE
      REGION GLOBAL         # GLOBAL | MOLECULE | MASSIVE
      &CSVR
        TIMECON [fs] 100    # Thermostat coupling time constant; 50–200 fs typical
      &END CSVR
      # Alternative: Nosé-Hoover chain (more physical NVT, slower to equilibrate)
      # &NOSE
      #   TIMECON [fs] 200
      #   LENGTH 3          # Chain length; 3 is standard
      #   YOSHIDA 3
      #   MTS 2
      # &END NOSE
    &END THERMOSTAT
  &END MD

  &PRINT
    &TRAJECTORY
      FILENAME water_NVT
      FORMAT XYZ
      &EACH
        MD 1                # Write every step; reduce to 10–50 for long runs
      &END EACH
    &END TRAJECTORY
    &VELOCITIES ON
      FORMAT XYZ
      &EACH
        MD 1
      &END EACH
    &END VELOCITIES
    &FORCES ON
      FORMAT XYZ
      &EACH
        MD 1
      &END EACH
    &END FORCES
    &ENERGIES
      &EACH
        MD 1
      &END EACH
    &END ENERGIES
    &RESTART
      BACKUP_COPIES 5       # Keep 5 backup .restart files; MD restarts are fragile
      &EACH
        MD 500              # Checkpoint every 500 steps (~250 fs at 0.5 fs timestep)
      &END EACH
    &END RESTART
    &RESTART_HISTORY
      &EACH
        MD 5000
      &END EACH
    &END RESTART_HISTORY
  &END PRINT
&END MOTION
```

AIMD restart:

```
# To restart from a checkpoint, modify &GLOBAL:
&GLOBAL
  PROJECT water_NVT_restart
  RUN_TYPE MD
  EXTENDED_FFT_LENGTHS .TRUE.
&END GLOBAL

# And add to &FORCE_EVAL/&DFT/&SCF:
&SCF
  SCF_GUESS RESTART          # Read wavefunction from .wfn file
  WFN_RESTART_FILE_NAME water_NVT-RESTART.wfn   # Adjust to latest checkpoint
  ...
&END SCF

# The .restart file in the project directory contains the full MD state (positions,
# velocities, thermostat state). CP2K automatically detects and reads it on restart
# if the PROJECT name matches. Rename or copy if changing project names.
```

Temperature and energy monitoring:

```python
import numpy as np
from pathlib import Path

def parse_cp2k_md_energies(outfile: str) -> dict:
    """Parse MD energies, temperature, and step from CP2K MD output."""
    steps, temps, ekins, epoots = [], [], [], []
    for line in open(outfile):
        if line.startswith(" i ="):
            # Format: " i =    1, time =    0.000, E =    -xxx.xxx, T = xxx.x"
            parts = line.split(",")
            step = int(parts[0].split("=")[1])
            temp = float(parts[3].split("=")[1].strip())
            e_total = float(parts[2].split("=")[1].strip())
            steps.append(step)
            temps.append(temp)
            epoots.append(e_total)
    return {"steps": np.array(steps), "temperature_K": np.array(temps),
            "energy_ha": np.array(epoots)}

md = parse_cp2k_md_energies("cp2k.out")
n_equil = len(md["steps"]) // 5  # Skip first 20% as equilibration
prod_temps = md["temperature_K"][n_equil:]
print(f"Production T: {prod_temps.mean():.1f} ± {prod_temps.std():.1f} K")
print(f"Target: 330 K  Drift: {prod_temps[-100:].mean() - prod_temps[:100].mean():.1f} K")
# Acceptable: T fluctuations within ±50 K; drift < 5 K over production run.
```

For NPT (constant pressure) ensembles:

```
&MD
  ENSEMBLE NPT_F            # NPT_F: isotropic barostat + thermostat
  STEPS  10000
  TIMESTEP [fs] 0.5
  TEMPERATURE 300
  PRESSURE [bar] 1.0        # Target pressure
  &THERMOSTAT
    TYPE CSVR
    REGION GLOBAL
    &CSVR
      TIMECON [fs] 100
    &END CSVR
  &END THERMOSTAT
  &BAROSTAT
    TIMECON [fs] 1000       # Barostat time constant; ~10x thermostat is a common starting point
    PRESSURE [bar] 1.0
    &THERMOSTAT
      TYPE CSVR
      &CSVR
        TIMECON [fs] 100
      &END CSVR
    &END THERMOSTAT
  &END BAROSTAT
&END MD
```

### Workflow 6: Generating DFT training data for ML potentials

CP2K can generate energy, force, and stress data for ML potential training. The output format must be postprocessed into the training set format (e.g., extxyz for MACE, LAMMPS dump for GRACE).

```
# For ML training data, key INCAR-equivalent settings:
# - EPS_SCF: 1E-6 (tight enough for accurate forces; tighten to 1E-7 if force noise is observed)
# - STRESS_TENSOR ANALYTICAL in &FORCE_EVAL (required for stress labels)
# - Forces are printed via &PRINT/&FORCES in &MOTION or via &PRINT_FORCES in &FORCE_EVAL
# - Use RUN_TYPE ENERGY for single-points; use MD or GEO_OPT for trajectory generation
```

```python
import subprocess
from pathlib import Path
from ase.io import read, write
import numpy as np

HARTREE_TO_EV = 27.211386245988
BOHR_TO_ANG   = 0.529177210903
# Stress conversion: CP2K outputs stress in GPa in some versions, in a.u. in others;
# always check the output header. For extxyz, stress should be in eV/Å³.
# 1 a.u. of pressure = 29421.01 bar; convert if needed. [EXPERT REVIEW NEEDED: check CP2K version]

def extract_cp2k_frame(out_dir: Path, n_atoms: int) -> dict | None:
    """
    Extract energy, forces, and stress from a completed CP2K single-point run.
    Returns None if the calculation did not converge.
    """
    outfile = out_dir / "cp2k.out"
    if not outfile.exists():
        return None
    text = outfile.read_text()
    if "SCF run NOT converged" in text:
        return None
    if "ENERGY| Total FORCE_EVAL" not in text:
        return None

    # Parse energy
    for line in reversed(text.splitlines()):
        if "ENERGY| Total FORCE_EVAL" in line:
            energy_ha = float(line.split()[-1])
            break
    energy_ev = energy_ha * HARTREE_TO_EV

    # Parse forces from -frc-1.xyz file (if PRINT/FORCES is enabled)
    frc_file = out_dir / f"{out_dir.name}-frc-1.xyz"
    forces_ev_ang = None
    if frc_file.exists():
        atoms_frc = read(str(frc_file), format="extxyz", index=-1)
        # CP2K force units in -frc-1.xyz are Hartree/Bohr
        forces_ev_ang = atoms_frc.get_positions() * HARTREE_TO_EV / BOHR_TO_ANG

    return {
        "energy_eV": energy_ev,
        "forces_eV_Ang": forces_ev_ang,
        "n_atoms": n_atoms,
        "energy_per_atom_eV": energy_ev / n_atoms,
    }


def batch_cp2k_single_points(structures, base_template: str, n_workers: int = 4):
    """
    Run CP2K single-point calculations for a list of ASE Atoms objects.
    Returns list of result dicts.
    """
    from concurrent.futures import ProcessPoolExecutor
    results = []
    calc_dirs = []
    for i, atoms in enumerate(structures):
        calc_dir = Path(f"sp_{i:05d}")
        calc_dir.mkdir(exist_ok=True)
        write(str(calc_dir / "structure.xyz"), atoms)
        (calc_dir / "cp2k.inp").write_text(base_template)
        calc_dirs.append(calc_dir)
    # Submit all; in production use a scheduler or FireWorks
    for d in calc_dirs:
        subprocess.run(
            ["cp2k.popt", "-i", "cp2k.inp", "-o", "cp2k.out"],
            cwd=d, check=False  # Don't raise; check convergence in extract step
        )
        result = extract_cp2k_frame(d, n_atoms=len(read(str(d / "structure.xyz"))))
        results.append(result)
    return results
```

Postprocessing to extxyz for MACE/GRACE training:

```python
from ase.io import read, write
from ase import Atoms
import numpy as np

def cp2k_results_to_extxyz(structures, results, output_file: str):
    """Write CP2K results as extxyz for ML potential training."""
    HARTREE_TO_EV = 27.211386245988
    BOHR_TO_ANG   = 0.529177210903

    frames = []
    for atoms, res in zip(structures, results):
        if res is None:
            continue  # Skip unconverged
        atoms = atoms.copy()
        atoms.info["energy"] = res["energy_eV"]
        if res["forces_eV_Ang"] is not None:
            atoms.arrays["forces"] = res["forces_eV_Ang"]
        frames.append(atoms)
    write(output_file, frames, format="extxyz")
    print(f"Wrote {len(frames)} frames to {output_file}")
```

### Workflow 7: Hybrid functionals with ADMM acceleration

Hybrid functional calculations (PBE0, HSE06) in CP2K use ADMM to project Hartree-Fock exchange onto a smaller auxiliary basis, reducing cost from O(N⁴) toward O(N² log N). ADMM is CP2K's most efficient path for hybrid DFT on periodic solids.

```
&FORCE_EVAL
  METHOD Quickstep
  &DFT
    BASIS_SET_FILE_NAME  BASIS_MOLOPT
    BASIS_SET_FILE_NAME  BASIS_ADMM          # Auxiliary basis for ADMM
    POTENTIAL_FILE_NAME  GTH_POTENTIALS

    &AUXILIARY_DENSITY_MATRIX_METHOD
      METHOD BASIS_PROJECTION               # Project primary basis onto auxiliary basis
      ADMM_PURIFICATION_METHOD NONE         # NONE | MCWEENY; NONE is faster; MCWEENY more accurate
      EXCH_SCALING_MODEL NONE
      EXCH_CORRECTION_FUNC PBEX            # Correction functional: PBE exchange
    &END AUXILIARY_DENSITY_MATRIX_METHOD

    &MGRID
      CUTOFF 600
      REL_CUTOFF 60
    &END MGRID

    &QS
      METHOD GPW
      EPS_DEFAULT 1.0E-12
    &END QS

    &SCF
      SCF_GUESS ATOMIC
      EPS_SCF 1.0E-6
      MAX_SCF 50
      &OT ON
        MINIMIZER DIIS
        PRECONDITIONER FULL_ALL
      &END OT
      &OUTER_SCF ON
        EPS_SCF 1.0E-6
        MAX_SCF 20
      &END OUTER_SCF
    &END SCF

    &XC
      &XC_FUNCTIONAL
        &PBE
          SCALE_X 0.75    ! 75% PBE exchange (PBE0 uses 25% HFX)
          SCALE_C 1.0
        &END PBE
      &END XC_FUNCTIONAL
      &HF
        FRACTION 0.25     ! 25% Hartree-Fock exchange for PBE0
        &SCREENING
          EPS_SCHWARZ 1.0E-10   ! Screening threshold for 2-electron integrals
        &END SCREENING
        &MEMORY
          MAX_MEMORY 4000       ! MB per MPI process for integral storage
          EPS_STORAGE_SCALING 0.1
        &END MEMORY
        &INTERACTION_POTENTIAL
          POTENTIAL_TYPE TRUNCATED           ! For PBE0 in solids; truncates HFX at R_C
          CUTOFF_RADIUS  8.0                 ! Å; must fit within cell — use ~min_cell/2
          ! For HSE06: use POTENTIAL_TYPE SHORTRANGE, OMEGA 0.11 bohr⁻¹
        &END INTERACTION_POTENTIAL
      &END HF
    &END XC
    &POISSON
      PERIODIC XYZ
    &END POISSON
  &END DFT

  &SUBSYS
    ...
    &KIND Mg
      BASIS_SET  DZVP-MOLOPT-SR-GTH
      BASIS_SET  AUX_FIT cFIT3               ! ADMM auxiliary basis for Mg; must exist in BASIS_ADMM
      POTENTIAL  GTH-PBE-q2
    &END KIND
    &KIND O
      BASIS_SET  DZVP-MOLOPT-SR-GTH
      BASIS_SET  AUX_FIT cFIT3               ! ADMM auxiliary basis for O
      POTENTIAL  GTH-PBE-q6
    &END KIND
  &END SUBSYS
&END FORCE_EVAL
```

ADMM accuracy validation:

```python
# ADMM introduces an approximation to HFX.
# Validate by comparing ADMM hybrid to exact hybrid on a small system:
# |E_ADMM - E_exact| / N_atoms should be < 1 meV/atom for well-chosen auxiliary basis.
# Common auxiliary basis choices for ADMM:
#   cFIT3:  small, fast, modest accuracy
#   cFIT9:  medium, good accuracy for most systems
#   cpFIT3: polarized, for systems with significant polarization [EXPERT REVIEW NEEDED]
# If ADMM errors are > 5 meV/atom, try a larger auxiliary basis (cFIT9 instead of cFIT3).
```

## Common Failure Modes

| Failure | Symptom | Likely Cause | Diagnostic | Fix |
|---------|---------|--------------|------------|-----|
| Wrong basis/pseudopotential pair | SCF converges but energies are systematically wrong; forces inconsistent; atom-type errors in log | GTH pseudopotential generated for a different functional than the XC in use (e.g., GTH-PBE-q with BLYP XC) | Check `&KIND` sections: `POTENTIAL GTH-{FUNC}-q{N}` must match `&XC_FUNCTIONAL {FUNC}` | Use matching PP and XC functional throughout; `GTH-PBE` with PBE, `GTH-BLYP` with BLYP |
| Unconverged CUTOFF | Total energy drifts as cell deforms; stress tensor unreliable; forces noisy for large basis functions | CUTOFF too low for the contracted Gaussians in the basis set | Run CUTOFF convergence scan (Workflow 2); plot E/atom vs CUTOFF | Increase CUTOFF until ΔE/atom < 1 meV/atom; typical production: 400–800 Ry for DZVP-MOLOPT-SR |
| Low REL_CUTOFF causing grid mapping error | Total energy converged with CUTOFF but shows steps/kinks in relaxation; forces inconsistent | REL_CUTOFF too small for tight Gaussians; atoms mapped onto coarse subgrid | Run REL_CUTOFF scan at fixed CUTOFF; check `CUTOFF` / `REL_CUTOFF` ratio (should give grid spacing < orbital radius) | Increase REL_CUTOFF to 60–100 Ry; re-run convergence; ratio CUTOFF/REL_CUTOFF > 8 is a warning sign |
| OT stalls on metallic system | RMS gradient plateaus above EPS_SCF; energy barely decreases; outer SCF does not converge | Fractional occupations at Fermi level violate OT assumptions; OT requires a finite gap | Check if system is metallic (DOS at Fermi level from a prior GGA run); check if `ENERGY_GAP` is unrealistic | Switch to diagonalization + ADDED_MOS + SMEAR; use Fermi-Dirac smearing |
| SCF oscillation with diagonalization | Energy oscillates ±0.1 eV between steps; Broyden mixing diverges | Mixing coefficient too large; strong charge sloshing in metallic or near-metallic system | Plot energy vs SCF iteration; check Broyden NBUFFER is adequate | Reduce ALPHA to 0.1–0.2; increase NBUFFER to 16; try PULAY_MIXING |
| Wrong Poisson solver for isolated system | Total energy depends on cell size anomalously; electrostatic potential diverges; spurious image interactions | PERIODIC Poisson solver applied to a molecule, cluster, or charged slab | Compare energy with two cell sizes; for isolated: energy should be cell-size-independent with MT | Use `PSOLVER MT` for molecules and clusters; for charged slabs use `PSOLVER WAVELET` or add compensating charge |
| AIMD timestep too large | Simulation diverges after a few steps; total energy increases without bound; forces grow explosively | Born-Oppenheimer timestep exceeds stability limit for fastest vibrational mode | Check highest frequency from a prior phonon or normal mode analysis; for H-containing systems, OH stretch is ~3600 cm⁻¹ → period ~9.3 fs | Use TIMESTEP ≤ 0.5 fs for H-containing systems; 1–2 fs for heavy-element systems without H |
| AIMD restart file mismatch | Restart fails with basis set error or atom count mismatch; geometry jumps discontinuously | Restart wavefunction from wrong run (different cell, basis, or atom count) | Check PROJECT name and `WFN_RESTART_FILE_NAME` path; verify atom count matches | Ensure PROJECT names and basis sets match between original and restart run; use `SCF_GUESS ATOMIC` if wfn file is suspect |
| Treating CP2K energy as directly comparable to VASP/QE | Apparent formation energy errors of 0.1–1 eV; systematic energy offsets between codes | Different pseudopotential definitions, core-electron count, and basis set incompleteness correction | Cross-validate on a known compound (e.g., MgO) against VASP or QE; compare lattice constants and bulk modulus | Never mix CP2K and VASP/QE energies in the same thermodynamic cycle without explicit cross-validation and energy alignment |
| ADMM HFX memory overflow | CP2K crashes with out-of-memory error during hybrid calculation | `MAX_MEMORY` in `&HF/&MEMORY` too small; 2-electron integrals exceed available RAM | Check available RAM per process; count MPI ranks; estimate memory as `MAX_MEMORY * n_MPI_ranks` | Increase `MAX_MEMORY`; reduce MPI processes per node; enable `EPS_STORAGE_SCALING` to discard small integrals |

## Best Practices

- **Always converge both CUTOFF and REL_CUTOFF** for each new basis set family and element combination. A single CUTOFF scan at fixed REL_CUTOFF is insufficient — run both scans systematically using Workflow 2.
- **Match pseudopotential to XC functional exactly.** `GTH-PBE` must be paired with PBE XC; `GTH-BLYP` with BLYP. Mismatches produce wrong energies without an error message.
- **Use MOLOPT-SR basis sets for periodic solids.** The short-range (SR) variants have smaller basis set superposition error and better linear independence in condensed phases than the standard MOLOPT sets.
- **Always include `OUTER_SCF` when using OT.** Without it, the OT loop converges the orbital gradient but not the total energy self-consistency; the resulting energy is not variational with respect to the density.
- **For AIMD, always enable `RESTART` printing with `BACKUP_COPIES ≥ 3`.** AIMD restarts are fragile — a corrupt or missing restart file loses the entire trajectory. Multiple backup copies guard against filesystem failures.
- **Do not compare CP2K total energies to VASP or QE energies without cross-validation.** Different pseudopotential core definitions and basis set incompleteness mean raw energies are not comparable across codes. Only energy differences within the same code and settings are meaningful.
- **For ML potential training data, use `EPS_SCF 1E-6` minimum and tighten to `1E-7` if forces are noisy.** Loose SCF convergence introduces force noise that corrupts ML training signals. Always enable `STRESS_TENSOR ANALYTICAL` in `&FORCE_EVAL`.
- **Enable dispersion correction (D3 or D4) for all condensed-phase and surface calculations.** Bare PBE misses London dispersion. D3(BJ) is the standard choice for most systems; D4 improves on D3 for highly polarizable systems. [EXPERT REVIEW NEEDED: D3 vs D4 crossover for specific material classes]
- **For charged supercells or defects, use the Martyna-Tuckerman (`MT`) or `WAVELET` Poisson solver.** The periodic `PERIODIC` solver wraps periodic images of charges and produces divergent or spurious electrostatics for charged systems.
- **Validate ADMM accuracy on a small test case** before committing to production hybrid calculations. Compare ADMM and non-ADMM energies on a 10–20 atom cell; acceptable discrepancy is < 1 meV/atom.
- **Record the full CP2K version, BASIS_SET file checksums, and POTENTIAL file checksums** for reproducibility. CP2K basis set files are updated between releases and can affect energies.

## Integration with Other Skills

- **[dft-convergence](../dft-convergence/SKILL.md):** The convergence philosophy (systematic scanning, property-dependent tolerances, property-specific tightening) applies directly to CP2K's CUTOFF, REL_CUTOFF, and EPS_SCF parameters. Workflow 2 in this skill is the CP2K analog of that skill's ENCUT and ecutwfc scans.
- **[vasp-workflow](../vasp-workflow/SKILL.md):** VASP and CP2K are complementary codes. VASP uses plane-wave basis sets (PAW); CP2K uses Gaussian basis sets (GTH). Cross-validating energies requires careful alignment. Use VASP for band structures and phonons; use CP2K for AIMD and large-scale liquid-phase simulations.
- **[quantum-espresso](../quantum-espresso/SKILL.md):** QE is the preferred open-source code for phonon (DFPT) calculations and for high-throughput DFT via AiiDA. CP2K and QE complement each other: use QE for phonons and linear-response; use CP2K for AIMD and large-scale DFT.
- **[high-throughput-dft](../high-throughput-dft/SKILL.md):** CP2K input generation and output parsing can be integrated into high-throughput frameworks via ASE's CP2K calculator or pymatgen's `Cp2kOutput`. The convergence parameters (CUTOFF, REL_CUTOFF, EPS_SCF) established by this skill become the fixed protocol for automated workflows.
- **[dft-dataset-generation](../dft-dataset-generation/SKILL.md):** CP2K AIMD trajectories and single-point calculations are a primary source of DFT training data for ML potentials. Workflow 6 here produces data in the format needed by that skill. Consistent CUTOFF, REL_CUTOFF, and EPS_SCF across the entire dataset are mandatory.
- **[dft-phonons](../dft-phonons/SKILL.md):** CP2K lacks a mature DFPT phonon module; for phonon calculations starting from CP2K-relaxed geometries, use the finite-displacement method (phonopy) with forces computed by CP2K single-points, or transfer the relaxed structure to QE for DFPT.
- **[dft-surfaces](../dft-surfaces/SKILL.md):** CP2K is well-suited for surface AIMD and large slab calculations. The Poisson solver selection (MT for isolated slabs, PERIODIC for dipole-corrected periodic slabs) described in this skill's Domain Context is essential for slab calculations without spurious image interactions.
- **[dft-defects](../dft-defects/SKILL.md):** Charged defect supercells in CP2K require the MT or WAVELET Poisson solver and explicit finite-size corrections. CP2K does not automatically apply Madelung or image-charge corrections — these must be computed externally (e.g., with pydefect or SHAKENBREAK).
- **[ase-framework](../../scientific-data/ase-framework/SKILL.md):** ASE's `CP2K` calculator wraps `cp2k.popt` and handles input file generation, subprocess execution, and output parsing. It supports AIMD via `ASE`'s molecular dynamics module with CP2K as the energy/force evaluator, enabling Python-driven workflows.
- **[pymatgen-analysis](../../scientific-data/pymatgen-analysis/SKILL.md):** pymatgen's `pymatgen.io.cp2k` module generates CP2K inputs from `Structure` objects and parses `Cp2kOutput`. It is the recommended path for high-throughput CP2K workflows driven by Materials Project structures.
- **[materials-databases-access](../../scientific-data/materials-databases-access/SKILL.md):** Starting structures for CP2K calculations can be retrieved from the Materials Project, ICSD, or COD via pymatgen or ASE. The retrieved `Structure` or `Atoms` object feeds directly into the CP2K input generation workflow.
- **[mlp-validation](../../ml-interatomic-potentials/mlp-validation/SKILL.md):** CP2K-generated energies, forces, and stresses from AIMD trajectories are a primary training data source for MACE, NequIP, and GRACE potentials. Validation of MLPs trained on CP2K data must use the same CUTOFF, REL_CUTOFF, and XC settings as the training set, with explicit checks for systematic CP2K vs. VASP offsets if mixing codes.
- **[active-learning-mlp](../../ml-interatomic-potentials/active-learning-mlp/SKILL.md):** Active learning loops can use CP2K as the DFT oracle for single-point labeling. Workflow 6 provides the batch single-point infrastructure. CP2K is particularly useful as the oracle when the target application involves liquid-phase or disordered systems.
- **[scientific-visualization](../../scientific-data/scientific-visualization/SKILL.md):** CP2K AIMD trajectories (`.xyz` output), cube files (electron density, ELF), and projected DOS files require visualization via VESTA (cube files), VMD/OVITO (trajectories), or matplotlib (DOS, temperature, energy time series).
- **[reproducibility-workflow](../../scientific-data/reproducibility-workflow/SKILL.md):** CP2K calculations require versioned basis set and pseudopotential files, checksummed input files, and logged output files to be reproducible. The CP2K version, BASIS_MOLOPT and GTH_POTENTIALS file versions, and all CUTOFF/REL_CUTOFF/EPS_SCF settings must be archived alongside each calculation.

## Key References

1. Kühne, T. D., et al. (2020). *CP2K: An electronic structure and molecular dynamics software package — Quickstep: Efficient and accurate electronic structure calculations.* Journal of Chemical Physics, 152(19), 194103. https://doi.org/10.1063/5.0007045
2. Lippert, G., Hutter, J., & Parrinello, M. (1997). *A hybrid Gaussian and plane wave density functional scheme.* Molecular Physics, 92(3), 477–487. https://doi.org/10.1080/002689797170220
3. VandeVondele, J., & Hutter, J. (2007). *Gaussian basis sets for accurate calculations on molecular systems in gas and condensed phases.* Journal of Chemical Physics, 127(11), 114105. https://doi.org/10.1063/1.2770708
4. Goedecker, S., Teter, M., & Hutter, J. (1996). *Separable dual-space Gaussian pseudopotentials.* Physical Review B, 54(3), 1703. https://doi.org/10.1103/PhysRevB.54.1703
5. Guidon, M., Hutter, J., & VandeVondele, J. (2010). *Auxiliary density matrix methods for Hartree-Fock exchange calculations.* Journal of Chemical Theory and Computation, 6(8), 2348–2364. https://doi.org/10.1021/ct1002225
6. Bussi, G., Donadio, D., & Parrinello, M. (2007). *Canonical sampling through velocity rescaling.* Journal of Chemical Physics, 126(1), 014101. https://doi.org/10.1063/1.2408420
7. Grimme, S., Antony, J., Ehrlich, S., & Krieg, H. (2010). *A consistent and accurate ab initio parametrization of density functional dispersion correction (DFT-D) for the 94 elements H–Pu.* Journal of Chemical Physics, 132(15), 154104. https://doi.org/10.1063/1.3382344
