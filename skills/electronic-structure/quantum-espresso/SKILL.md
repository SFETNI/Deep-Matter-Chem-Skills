---
name: quantum-espresso
domain: electronic-structure
maturity: draft
tools:
  - Quantum ESPRESSO
  - ASE
  - pymatgen
  - phonopy
  - AiiDA
tested_versions:
  Quantum ESPRESSO: "7.3"
  ASE: "3.23.0"
  pymatgen: "2024.11.13"
  phonopy: "2.26.6"
requires_gpu: false
estimated_runtime: minutes-to-days
---

# Quantum ESPRESSO Workflow

## Description

This skill covers end-to-end DFT calculations using Quantum ESPRESSO (QE), from input file construction and pseudopotential selection through SCF convergence, structural relaxation, band structure, and DOS workflows. QE is the dominant open-source plane-wave DFT code and the standard backend for many high-throughput databases, phonon codes (PHonon/DFPT), and ML potential pipelines. Invoke this skill when setting up or debugging a QE calculation, selecting pseudopotentials, parallelizing across HPC resources, or generating DFT reference data with an open-source code.

## Domain Context

Quantum ESPRESSO solves the Kohn-Sham DFT equations using a plane-wave basis set on a regular real-space grid, with pseudopotentials or PAW datasets representing core electrons. The plane-wave expansion is controlled by `ecutwfc` (wavefunction cutoff, in Ry) and `ecutrho` (charge density cutoff, in Ry). Unlike VASP, QE uses open pseudopotential libraries (SSSP, PseudoDojo) rather than bundled PAW datasets, making pseudopotential choice an explicit user responsibility.

Key physical approximations and their consequences:

- **Pseudopotential approximation:** Core electrons are replaced by a smooth effective potential. The cutoff radius and the number of valence electrons depend on the pseudopotential. Harder (smaller cutoff radius) pseudopotentials require higher `ecutwfc` but are more transferable. Softer (larger cutoff radius) pseudopotentials are cheaper but may fail for highly compressed lattices or unusual oxidation states.
- **Ultrasoft vs. norm-conserving vs. PAW:** Ultrasoft pseudopotentials (USPP) achieve low `ecutwfc` at the cost of requiring a separately converged `ecutrho` (typically 8–12× `ecutwfc`). Norm-conserving (NC) pseudopotentials use `ecutrho = 4 × ecutwfc` and are required for linear-response properties and many GW/TDDFT applications. PAW reconstructs all-electron wavefunctions and gives accuracy close to all-electron at moderate cost.
- **Periodic boundary conditions:** QE assumes 3D periodicity. Molecules, surfaces, and wires in a box must be handled with sufficient vacuum and, for charged systems or polar slabs, Makov-Payne or Martyna-Tuckerman corrections.
- **Density functional:** The functional is selected via the pseudopotential — each PP is generated for a specific functional (PBE, PBEsol, LDA, etc.). Unlike VASP, the `input_dft` tag in `&SYSTEM` can override this, but should only be used with care and for compatible PP/functional pairs.

QE outputs are in Ry and bohr units internally, but most output files report in eV and Å or in atomic units depending on the context. Unit confusion is a frequent source of errors.

## When to Use This Skill

- Open-source DFT calculations where code transparency or reproducibility with public PP libraries is required.
- Phonon calculations via DFPT (`ph.x`) — QE's PHonon module is mature and widely benchmarked.
- High-throughput workflows using AiiDA, which has first-class QE plugins.
- Linear-response properties: dielectric constants, Born effective charges, electron-phonon coupling (`ph.x` + `elph`).
- Systems where the SSSP or PseudoDojo pseudopotential accuracy has been validated and VASP PAW datasets have not.
- When comparing against literature results computed with QE at specific convergence settings.
- Generating DFT reference data for ML potential training with an auditable, version-pinned open-source code.

## When Not to Use This Skill

- **When VASP is the institutional standard and license is available** — mixing codes in a dataset without careful cross-validation introduces systematic energy offsets.
- **Hybrid functionals (HSE06) at scale** — QE's hybrid implementation (`input_dft = 'hse'`) is correct but substantially slower than VASP for large supercells; VASP or CP2K are preferred for routine hybrid calculations on periodic solids.
- **Strongly correlated systems requiring DMFT** — use codes with DMFT interfaces (TRIQS, DFT+DMFT workflow via Wannier90) rather than QE alone.
- **Very large systems (> 500 atoms) in production** — for large-scale MD or ML-potential training data generation with many structures, codes with better task-farm integration (VASP + AIMD or CP2K) may be more practical.
- **When the required pseudopotential does not exist** in SSSP or PseudoDojo — constructing a new pseudopotential requires expertise beyond this skill.

## Core Concepts

- **`ecutwfc` (Ry):** Wavefunction kinetic energy cutoff. Controls the plane-wave basis size. Always specify in Ry (1 Ry = 13.606 eV). The SSSP and PseudoDojo libraries provide verified minimum cutoffs per element and functional; use these as a starting point, not as a final value.
- **`ecutrho` (Ry):** Charge density cutoff. For NC pseudopotentials: `ecutrho = 4 × ecutwfc` (enforced automatically if not specified). For USPP or PAW: `ecutrho` must be explicitly set and converged; typical range is 6–12× `ecutwfc`. Undersetting `ecutrho` for USPP produces wrong energies and forces without an error message.
- **`ibrav`:** Bravais lattice index. Using `ibrav=0` with explicit `CELL_PARAMETERS` is strongly recommended over `ibrav > 0` — the lattice vector convention for `ibrav > 0` has caused many silent errors in the community due to non-obvious angle definitions. [EXPERT REVIEW NEEDED: specific ibrav values where sign conventions are ambiguous]
- **`degauss` (Ry):** Smearing width. Specified in Ry, not eV. 0.01 Ry ≈ 0.14 eV. A common mistake is setting `degauss` to a value intended as eV — this applies a smearing ~14× too large.
- **Pseudopotential libraries:**
  - *SSSP (Standard Solid-State Pseudopotentials):* Curated, version-controlled, benchmarked against all-electron codes. Two tiers: `efficiency` (lower cutoffs, good for structure/forces) and `precision` (higher cutoffs, suitable for phonons and EOS). Always use a specific SSSP version for reproducibility.
  - *PseudoDojo:* Systematic NC-FR (fully relativistic) and scalar-relativistic pseudopotentials. Required for spin-orbit coupling. Provides delta-factor benchmarks for quality control.
- **`calculation` flag:** Controls the workflow type. Must match the active namelists and the restart file state. Switching between `scf`, `relax`, and `vc-relax` mid-run requires understanding which output files (`.save/`, `CHARGE-DENSITY.dat`, `wfc*.dat`) are reusable.
- **`outdir` and `prefix`:** `outdir` is the scratch directory (should point to fast local storage on HPC); `prefix` labels all output files. Using the same `prefix` in different jobs in the same `outdir` will corrupt calculations silently.
- **Restart (`startingwfc`, `startingpot`):** QE can restart SCF from saved wavefunctions (`startingwfc = 'file'`) or charge density (`startingpot = 'file'`). This requires the `.save/` directory from a prior run. Restarting across different k-meshes or cutoffs will produce wrong results.
- **Parallelization:** QE parallelizes over k-points (`-npool`), plane waves (`-ntg`, task groups), bands (`-nband`), and images (for NEB/phonon). The most impactful flag for typical solid-state calculations is `-npool` (number of k-point pools), which must divide the number of irreducible k-points exactly.

## Key Workflows

### Workflow 1: SCF calculation — input anatomy and setup

The `pw.x` SCF input file has a fixed structure. All namelists must appear in order; all cards (ATOMIC_SPECIES, ATOMIC_POSITIONS, K_POINTS, CELL_PARAMETERS) are mandatory for `ibrav=0`.

```fortran
! scf.in — annotated pw.x input for a simple cubic perovskite (SrTiO3 as example)
! Units: energy in Ry, length in bohr unless otherwise stated in cards.

&CONTROL
  calculation   = 'scf'       ! scf | nscf | bands | relax | vc-relax | md
  prefix        = 'STO'       ! Label for all output files in outdir
  outdir        = './scratch'  ! Scratch directory; use fast local storage on HPC
  pseudo_dir    = './PP'       ! Path to pseudopotential files
  verbosity     = 'high'      ! 'low' suppresses useful convergence diagnostics
  etot_conv_thr = 1.0d-5      ! Total energy conv. thr for relax (Ry); not used for scf
  forc_conv_thr = 1.0d-4      ! Force conv. thr for relax (Ry/bohr)
  tprnfor       = .true.      ! Print forces to output
  tstress       = .true.      ! Print stress tensor (needed for vc-relax and ML data)
/

&SYSTEM
  ibrav         = 0           ! Always use ibrav=0 with explicit CELL_PARAMETERS
  nat           = 5           ! Number of atoms in the unit cell
  ntyp          = 3           ! Number of distinct atomic species
  ecutwfc       = 60          ! Wavefunction cutoff (Ry); must be converged
  ecutrho       = 480         ! Charge density cutoff (Ry); for USPP: 8x ecutwfc minimum
                              ! For NC pseudopotentials: 4x ecutwfc (can omit)
  occupations   = 'smearing'
  smearing      = 'mp'        ! Methfessel-Paxton for metals; 'gaussian' for unknowns
  degauss       = 0.005       ! In Ry (not eV!); 0.005 Ry ~ 0.07 eV
  ! For insulators after convergence: occupations='fixed' or smearing='tetrahedra_opt'
  ! with a dense k-mesh.
/

&ELECTRONS
  conv_thr      = 1.0d-9      ! SCF convergence threshold (Ry); tighten for forces/phonons
  mixing_beta   = 0.5         ! Charge mixing fraction; reduce to 0.1–0.2 for hard cases
  electron_maxstep = 200      ! Max SCF iterations; default 100 often insufficient
  ! diagonalization = 'david'  ! Default; use 'cg' for very large systems
/

! --- Cards (order matters) ---

ATOMIC_SPECIES
Sr  87.62   Sr.pbesol-spn-kjpaw_psl.1.0.0.UPF
Ti  47.867  Ti.pbesol-spn-kjpaw_psl.1.0.0.UPF
O   15.999  O.pbesol-n-kjpaw_psl.1.0.0.UPF

ATOMIC_POSITIONS {angstrom}
Sr  0.000000  0.000000  0.000000
Ti  1.952500  1.952500  1.952500
O   1.952500  1.952500  0.000000
O   1.952500  0.000000  1.952500
O   0.000000  1.952500  1.952500

K_POINTS {automatic}
6 6 6  0 0 0    ! Gamma-centered; test convergence (see dft-convergence skill)

CELL_PARAMETERS {angstrom}
3.905000  0.000000  0.000000
0.000000  3.905000  0.000000
0.000000  0.000000  3.905000
```

Parsing output with ASE:

```python
from ase.io import read
from pathlib import Path

# ASE can read pw.x output files directly
atoms = read("scf.out", format="espresso-out")
print(f"Energy: {atoms.get_potential_energy():.6f} eV")
print(f"Forces max: {atoms.get_forces().max():.4f} eV/Å")

# Or parse the XML data file for more complete information
from ase.calculators.espresso import EspressoProfile
# See ASE docs for calculator-based workflows

# pymatgen can also parse QE output:
from pymatgen.io.espresso.outputs import PWOutput
pw_out = PWOutput("scf.out")
print(f"Final energy: {pw_out.final_energy} Ry")
```

### Workflow 2: Structural relaxation (relax and vc-relax)

Use `relax` to relax atomic positions at fixed cell. Use `vc-relax` to relax both positions and the cell (variable-cell relaxation). Always run a final `scf` after `vc-relax` — the last ionic step may not be fully SCF-converged.

```fortran
! relax.in — ionic relaxation at fixed cell
&CONTROL
  calculation   = 'relax'
  prefix        = 'relax_run'
  outdir        = './scratch'
  pseudo_dir    = './PP'
  tprnfor       = .true.
  tstress       = .true.
  etot_conv_thr = 1.0d-5      ! Ry; energy convergence between ionic steps
  forc_conv_thr = 1.0d-4      ! Ry/bohr ≈ 0.005 eV/Å; tighten for phonons
/

&SYSTEM
  ibrav         = 0
  nat           = 5
  ntyp          = 3
  ecutwfc       = 60
  ecutrho       = 480
  occupations   = 'smearing'
  smearing      = 'mp'
  degauss       = 0.005
/

&ELECTRONS
  conv_thr      = 1.0d-8      ! Tighter than single-point; forces need converged SCF
  mixing_beta   = 0.5
/

&IONS
  ion_dynamics  = 'bfgs'      ! BFGS quasi-Newton; robust for most cases
  ! ion_dynamics = 'damp'     ! Alternative for very soft modes or near-transition states
/

! For vc-relax, add:
! &CELL
!   cell_dynamics  = 'bfgs'   ! Variable-cell BFGS
!   press_conv_thr = 0.5      ! kbar; pressure convergence for vc-relax
!   cell_dofree    = 'all'    ! Relax all cell degrees of freedom
!   ! cell_dofree = 'ibrav'   ! Maintain ibrav symmetry constraints
! /
```

Post-relaxation final SCF (always required after vc-relax):

```python
from pathlib import Path
import subprocess

def run_final_scf(relax_prefix: str, scf_prefix: str, outdir: str, pseudo_dir: str) -> None:
    """
    After vc-relax, run a final scf on the relaxed structure.
    The relaxed cell and positions are in the last CELL_PARAMETERS and
    ATOMIC_POSITIONS blocks of the relax output file.
    Parse them and write a new scf.in.
    """
    from ase.io import read, write

    # ASE can read the final relaxed structure from the QE output
    relaxed = read(f"{relax_prefix}.out", format="espresso-out", index=-1)

    # Write a new SCF input based on the relaxed structure
    input_data = {
        "control": {
            "calculation": "scf",
            "prefix": scf_prefix,
            "outdir": outdir,
            "pseudo_dir": pseudo_dir,
            "tprnfor": True,
            "tstress": True,
        },
        "system": {
            "ecutwfc": 60,
            "ecutrho": 480,
            "occupations": "smearing",
            "smearing": "mp",
            "degauss": 0.005,
        },
        "electrons": {
            "conv_thr": 1.0e-9,
            "mixing_beta": 0.5,
        },
    }
    pseudopotentials = {"Sr": "Sr.pbesol-spn-kjpaw_psl.1.0.0.UPF",
                        "Ti": "Ti.pbesol-spn-kjpaw_psl.1.0.0.UPF",
                        "O":  "O.pbesol-n-kjpaw_psl.1.0.0.UPF"}
    from ase.io.espresso import write_espresso_in
    with open("final_scf.in", "w") as f:
        write_espresso_in(f, relaxed, input_data=input_data,
                          pseudopotentials=pseudopotentials,
                          kpts=(8, 8, 8), koffset=(0, 0, 0))
```

**Convergence checks after relaxation:**

```bash
# Check that forces are below the threshold
grep "Total force" relax.out | tail -5

# Check that pressure is below press_conv_thr (for vc-relax)
grep "P=" relax.out | tail -5

# Check that the SCF converged at every ionic step (not just the last)
grep "convergence NOT achieved" relax.out
# Any match here means an ionic step used unconverged forces — the trajectory is unreliable.
```

### Workflow 3: Non-SCF calculation and band structure

The standard band structure workflow in QE is two-step: (1) SCF on a uniform k-mesh to generate the charge density; (2) nscf on the band path using the frozen charge density.

```fortran
! Step 1: SCF (standard, saves charge density to outdir/prefix.save/)
! calculation = 'scf' — as in Workflow 1

! Step 2: bands calculation along high-symmetry path
! bands.in
&CONTROL
  calculation   = 'bands'
  prefix        = 'STO'       ! Must match the SCF prefix and outdir
  outdir        = './scratch'
  pseudo_dir    = './PP'
/
&SYSTEM
  ibrav         = 0
  nat           = 5
  ntyp          = 3
  ecutwfc       = 60
  ecutrho       = 480
  occupations   = 'smearing'
  smearing      = 'mp'
  degauss       = 0.005
  nbnd          = 30          ! Number of bands; must cover all states of interest
/
&ELECTRONS
  conv_thr      = 1.0d-9
  diago_full_acc = .true.     ! Full diagonalization for accurate band energies
/

K_POINTS {crystal_b}         ! High-symmetry path in crystal coordinates
5
  0.000  0.000  0.000  30    ! Gamma, 30 points to next
  0.500  0.000  0.000  30    ! X
  0.500  0.500  0.000  30    ! M
  0.000  0.000  0.000  30    ! Gamma
  0.500  0.500  0.500  1     ! R (endpoint)
```

Post-processing with `bands.x` and plotting:

```python
from pathlib import Path
import subprocess

# Run bands.x to extract band data into a plottable format
bands_pp_input = """\
&BANDS
  prefix    = 'STO'
  outdir    = './scratch'
  filband   = 'STO.bands.dat'
  lsym      = .true.
/
"""
Path("bands_pp.in").write_text(bands_pp_input)
subprocess.run(["bands.x", "-in", "bands_pp.in"], check=True)

# Parse and plot with pymatgen or manually
import numpy as np
import matplotlib.pyplot as plt

# STO.bands.dat.gnu contains k-index and energy columns
data = np.loadtxt("STO.bands.dat.gnu")
# First column: k-path coordinate; remaining columns: band energies in eV
# (bands.x converts from Ry to eV in the .gnu file)
k = data[:, 0]
# Each band is a separate block separated by blank lines; parse accordingly.

# Alternatively, use pymatgen's QE output parser:
from pymatgen.io.espresso.outputs import PWOutput, BandStructure
# See pymatgen docs for full band structure parsing workflow.
```

### Workflow 4: DOS and projected DOS

```fortran
! Step 1: Dense nscf on uniform k-mesh (NOT the band-path k-points)
! nscf.in
&CONTROL
  calculation   = 'nscf'
  prefix        = 'STO'
  outdir        = './scratch'
  pseudo_dir    = './PP'
/
&SYSTEM
  ibrav = 0
  nat = 5
  ntyp = 3
  ecutwfc = 60
  ecutrho = 480
  occupations = 'tetrahedra_opt'  ! Best for DOS; requires uniform MP k-mesh
  nbnd = 40
/
&ELECTRONS
  conv_thr      = 1.0d-9
  diago_full_acc = .true.
/

K_POINTS {automatic}
16 16 16  0 0 0   ! 2-4x denser than SCF mesh for accurate DOS
```

```fortran
! Step 2: dos.x for total DOS
! dos.in
&DOS
  prefix    = 'STO'
  outdir    = './scratch'
  fildos    = 'STO.dos'
  Emin      = -10.0    ! eV relative to Fermi level
  Emax      =  10.0
  DeltaE    =  0.01    ! Energy resolution in eV
/
```

```fortran
! Step 3: projwfc.x for projected DOS (PDOS)
! projwfc.in
&PROJWFC
  prefix      = 'STO'
  outdir      = './scratch'
  filpdos     = 'STO.pdos'
  Emin        = -10.0
  Emax        =  10.0
  DeltaE      =  0.01
  degauss     = 0.005  ! Ry; Gaussian broadening for PDOS (separate from SCF degauss)
/
```

### Workflow 5: Parallelization for HPC

QE's performance is strongly dependent on choosing the right parallelization flags. The primary flag is `-npool` (number of k-point pools).

```bash
#!/bin/bash
# Example SLURM submission script for QE on HPC
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=32      # MPI ranks per node
#SBATCH --cpus-per-task=1         # OpenMP threads per rank (usually 1 for pw.x)
#SBATCH --time=12:00:00

module load quantum-espresso/7.3

NPOOL=8       # Must divide the number of irreducible k-points exactly.
              # For a 6x6x6 Gamma-centered mesh: nkpts_irred depends on symmetry;
              # for a low-symmetry system ~216, good choices: 4, 8, 12.
NTASK=$((SLURM_NTASKS))

mpirun -np $NTASK pw.x -npool $NPOOL -in scf.in > scf.out

# Parallelization hierarchy (from most to least impactful for typical pw.x runs):
# 1. -npool N    : k-point pools. Each pool handles N/nkpts k-points independently.
#                  Most effective when nkpts >> npool. Near-linear scaling with npool.
# 2. -ntg N      : Task groups for 3D FFTs. Reduces FFT communication; useful for
#                  many-node runs where FFT latency dominates.
# 3. -nband N    : Band parallelization. Rarely beneficial for standard DFT; useful
#                  for hybrid functionals or EXX.
# 4. OpenMP      : Set OMP_NUM_THREADS > 1 for thread-level parallelism within each
#                  MPI rank. Effective on architectures with many cores per socket.
#                  Test empirically; often not beneficial for pw.x. [EXPERT REVIEW NEEDED]

# Rule of thumb: npool = number of nodes is a good starting point.
# Verify: grep "Parallel routines" scf.out for timing breakdown.
```

Checking parallelization efficiency:

```bash
# Timing summary is at the end of pw.x output:
grep -A 20 "PWSCF        :" scf.out | tail -25

# If "fft" dominates: increase -ntg
# If "wfcollect" or "reduce" dominates: too many MPI ranks for system size
# If wall time >> CPU time / NTASK: poor load balancing; check npool divides nkpts

# Check nkpts irreducible in output:
grep "number of k points" scf.out

# Check total memory per process:
grep "per process" scf.out
# If memory per process is too large, increase npool (distributes k-points and
# reduces memory per pool).
```

### Workflow 6: Pseudopotential selection and validation

Choosing the right pseudopotential family is an explicit responsibility in QE that VASP handles internally via bundled PAW datasets.

```python
# Pseudopotential selection guide

# SSSP (Standard Solid-State Pseudopotentials)
# https://www.materialscloud.org/discover/sssp
# - Two tiers: 'efficiency' and 'precision'
# - Efficiency: lower ecutwfc, good for structural properties and forces
# - Precision: higher ecutwfc, suitable for phonons, EOS, and benchmarking
# - Always cite the specific SSSP version (currently v1.3 for PBE, v1.2 for PBEsol)
# - Mix of PAW (from PSLibrary) and USPP; ecutrho requirements vary per element

SSSP_EFFICIENCY_CUTOFFS = {
    # Element: (ecutwfc_Ry, ecutrho_Ry) — from SSSP v1.3 PBE
    # These are MINIMUM values; always test convergence above these.
    "H":  (30,  120),
    "C":  (40,  160),
    "N":  (40,  160),
    "O":  (40,  160),  # USPP; ecutrho = 4x for NC, 8-12x for USPP
    "Fe": (90,  720),  # Hard element; USPP, high ecutrho required
    "Ti": (40,  200),
    "Sr": (30,  150),
    # [EXPERT REVIEW NEEDED: verify these against current SSSP v1.3 table]
}

# PseudoDojo (https://www.pseudo-dojo.org)
# - Norm-conserving, scalar-relativistic and fully-relativistic (SOC)
# - Required for: spin-orbit coupling, GW, TDDFT, electron-phonon with SOC
# - ecutrho = 4 × ecutwfc (automatic for NC)
# - Provides delta-factor and GBRV benchmarks per pseudopotential
# - Use 'standard' accuracy for production; 'stringent' for high-accuracy benchmarks

def select_pseudopotential(element: str, functional: str = "PBE",
                            soc: bool = False, application: str = "structure") -> str:
    """
    Return a recommended pseudopotential choice given element and application.
    This is a decision guide, not an automated selector — always verify against
    the current SSSP/PseudoDojo website and benchmark tables.
    """
    if soc:
        return f"Use PseudoDojo fully-relativistic NC PP for {element} ({functional})"
    if application in ("phonon", "dfpt", "gw", "tddft"):
        return f"Use PseudoDojo NC PP for {element} ({functional}); ecutrho = 4*ecutwfc"
    if application in ("structure", "forces", "relax", "mlp"):
        return f"Use SSSP efficiency PP for {element} ({functional}); check ecutrho from SSSP table"
    return f"Use SSSP precision PP for {element} ({functional}) for high-accuracy benchmarks"
```

## Common Failure Modes

| Failure | Symptom | Likely Cause | Diagnostic | Fix |
|---------|---------|--------------|------------|-----|
| `degauss` unit confusion | Enormous smearing; DOS completely blurred; metals show insulating gap; energy wrong by large amount | `degauss` set in eV instead of Ry | Check: 0.1 eV = 0.00735 Ry; if degauss > 0.05, suspect unit error | Set `degauss` in Ry; recall 1 Ry = 13.606 eV |
| `ecutrho` too small for USPP | Forces wrong by > 0.1 eV/Å despite SCF convergence; energy does not converge with ecutwfc scan; density oscillations | `ecutrho` not set or set to 4× ecutwfc for an ultrasoft pseudopotential | Check PP header for recommended ecutrho; compare to input; plot charge density for oscillations | Set `ecutrho` ≥ 8× ecutwfc for USPP; rerun convergence scan |
| Mixed pseudopotential families | Energy difference between structures nonsensical; formation energies off by eV | Different PP family (NC vs USPP) or different functional used for different structures in the same dataset | `grep UPF *.in` to compare PP filenames across inputs; check UPF header for `generated using` functional | Use identical PP files and functional for all calculations in a thermodynamic cycle |
| Wrong `prefix` or `outdir` collision | QE reads stale wavefunctions or charge density from a prior run; SCF converges in 1 step with wrong energy | Another calculation used the same `prefix` in the same `outdir` | Check modification time of `outdir/prefix.save/` vs. run start time | Use unique `prefix` per calculation; or set `startingwfc = 'atomic'` and `startingpot = 'atomic'` to restart from scratch |
| `convergence NOT achieved` during relax | Ionic step uses unconverged forces; structural trajectory unreliable; final structure wrong | `conv_thr` too loose or `electron_maxstep` too small for difficult electronic structure | `grep "convergence NOT achieved" *.out`; count occurrences | Tighten `conv_thr` to 1e-8; increase `electron_maxstep` to 300; reduce `mixing_beta` to 0.2 |
| nscf/bands calculation ignores new k-path | Band structure is flat or missing; calculation uses SCF k-points instead of band path | `startingwfc` or `startingpot` incompatible with new k-mesh; QE falls back to SCF k-points | Check `K_POINTS` block in `bands.in`; grep "k-points" in output | Ensure `outdir/prefix.save/` exists from a prior SCF; use `startingwfc = 'atomic+random'` if restarting is problematic |
| `ibrav > 0` lattice vector error | Wrong lattice (sheared or wrong angles); silently produces wrong symmetry and k-path | `ibrav > 0` with non-standard angle convention or wrong `celldm` order | Compare lattice output in `scf.out` to intended structure; visualize with VESTA or ASE | Always use `ibrav = 0` with explicit `CELL_PARAMETERS {angstrom}` |
| `-npool` does not divide nkpts_irred | QE crashes with `npool not divisible` error or hangs | Number of k-point pools does not evenly divide the number of irreducible k-points | `grep "number of k points" scf.out` before setting npool | Choose npool to divide nkpts_irred; verify with a short test run |

## Best Practices

- **Always use `ibrav = 0` with explicit `CELL_PARAMETERS {angstrom}`.** The `ibrav > 0` convention has subtle angle definitions that differ from standard crystallographic conventions and have caused reproducibility failures in the community.
- **Always specify `ecutrho` explicitly for USPP and PAW pseudopotentials.** Relying on QE's default (4× ecutwfc) is only safe for norm-conserving pseudopotentials.
- **Pin the SSSP or PseudoDojo version in your workflow.** Record the exact PP filenames, checksums, and the library version. Pseudopotential files have changed between SSSP releases in ways that affect energies.
- **Use unique `prefix` values per calculation** and organize `outdir` by calculation type (e.g., `./scratch/scf/`, `./scratch/relax/`). A collision in `prefix` silently reuses stale wavefunctions.
- **Always run a final `scf` after `vc-relax`** before using the energy for thermodynamics. The last ionic step in a vc-relax is not always fully SCF-converged.
- **Check for `convergence NOT achieved` in every relaxation output.** Any ionic step that used unconverged forces produces an unreliable structural update.
- **For HPC runs, test parallelization scaling** on a small system before submitting large production jobs. The optimal npool depends on the k-mesh, system size, and node interconnect.
- **Set `tprnfor = .true.` and `tstress = .true.` in all production runs** — forces and stress are cheap to compute and essential for validating convergence, generating ML training data, and checking for Pulay stress.
- **For ML potential training data generation, use `wf_collect = .false.`** (default in QE 7+) to avoid writing large wavefunction files to disk for every single-point calculation.
- **Do not mix QE and VASP energies in the same thermodynamic cycle** without a careful cross-validation study. Systematic offsets arise from different pseudopotential types, core-electron definitions, and numerical implementations.

## Integration with Other Skills

- **[dft-convergence](../dft-convergence/SKILL.md):** The cutoff convergence protocol (ecutwfc, ecutrho, k-mesh, degauss) described in that skill applies directly to QE. Workflows 1, 3, and 4 of the convergence skill include QE-specific syntax. Always establish converged settings before production.
- **[vasp-workflow](../vasp-workflow/SKILL.md):** VASP and QE are complementary codes. When cross-validating or comparing results, both must use the same functional, equivalent pseudopotentials, and consistent convergence settings. Energy offsets between the two codes should be characterized before mixing datasets.
- **[dft-phonons](../dft-phonons/SKILL.md):** QE's PHonon module (`ph.x`) is one of the most mature DFPT phonon implementations. That skill covers the ph.x input structure, q-point grids, and postprocessing with `matdyn.x`. The SCF convergence settings established here (particularly `conv_thr` and `ecutwfc`) feed directly into the phonon calculation accuracy.
- **[high-throughput-dft](../high-throughput-dft/SKILL.md):** AiiDA has first-class QE plugins (`aiida-quantumespresso`). High-throughput QE workflows typically use the SSSP pseudopotential library and automated convergence protocols. This skill's input anatomy and failure modes inform the exception handling logic in those workflows.
- **[dft-dataset-generation](../dft-dataset-generation/SKILL.md):** When QE is the DFT oracle for ML potential training, the consistent settings (ecutwfc, ecutrho, PP family, smearing) established here become the fixed protocol for the entire dataset. Inconsistent settings corrupt force labels.
- **[mlp-validation](../../ml-interatomic-potentials/mlp-validation/SKILL.md):** PP family choice and cutoff consistency are first-order checks in MLP validation. Force errors that appear to be MLP failures often trace back to inconsistent or undertested DFT reference settings.
- **[ase-framework](../../scientific-data/ase-framework/SKILL.md):** ASE's `Espresso` calculator wraps `pw.x` and can read/write QE input and output files. This enables Python-driven workflows, high-throughput structure enumeration, and post-processing without manual file editing.
- **[pymatgen-analysis](../../scientific-data/pymatgen-analysis/SKILL.md):** pymatgen's `pymatgen.io.espresso` module parses QE output files and can generate inputs from `Structure` objects, enabling database-driven high-throughput workflows.

## Key References

1. Giannozzi, P., et al. (2009). *QUANTUM ESPRESSO: a modular and open-source software project for quantum simulations of materials.* Journal of Physics: Condensed Matter, 21(39), 395502. https://doi.org/10.1088/0953-8984/21/39/395502
2. Giannozzi, P., et al. (2017). *Advanced capabilities for materials modelling with Quantum ESPRESSO.* Journal of Physics: Condensed Matter, 29(46), 465901. https://doi.org/10.1088/1361-648X/aa8f79
3. Prandini, G., et al. (2018). *Precision and efficiency in solid-state pseudopotential calculations.* npj Computational Materials, 4, 72. https://doi.org/10.1038/s41524-018-0127-2
4. van Setten, M. J., et al. (2018). *The PseudoDojo: Training and grading a 85 element optimized norm-conserving pseudopotential table.* Computer Physics Communications, 226, 39–54. https://doi.org/10.1016/j.cpc.2018.01.012
5. Lejaeghere, K., et al. (2016). *Reproducibility in density functional theory calculations of solids.* Science, 351(6280), aad3000. https://doi.org/10.1126/science.aad3000
