---
name: gromacs-workflow
domain: atomistic-md
maturity: draft
tools:
  - GROMACS
  - MDAnalysis
  - ASE
  - pandas
  - matplotlib
tested_versions:
  GROMACS: "2024.2"
  MDAnalysis: "2.7.0"
  ASE: "3.23.0"
requires_gpu: false
estimated_runtime: minutes-to-weeks
---

# GROMACS Workflow

## Description

This skill covers end-to-end molecular dynamics simulations using GROMACS: topology construction and the `.top`/`.itp` hierarchy, `.mdp` parameter anatomy for energy minimization, NVT, NPT, and production runs, the `grompp → mdrun → analysis` pipeline, cutoff and PME setup, thermostat and barostat selection, constraint algorithms (LINCS, SETTLE), force-field compatibility (AMBER, CHARMM, OPLS-AA, GROMOS, Martini), HPC execution with MPI/OpenMP/GPU offload, and restart continuation. GROMACS is the standard tool for biomolecular and soft-matter simulations and is the reference code for AMBER, CHARMM36, and OPLS-AA force fields. Invoke this skill when setting up, running, diagnosing, or extending any GROMACS simulation.

## Domain Context

GROMACS integrates Newton's equations of motion using a leap-frog Verlet integrator (the `md` integrator) or stochastic dynamics (`sd`). Its distinctive architecture separates the **run-input file** (`.tpr`, compiled from topology + mdp + coordinates by `grompp`) from the **raw inputs** (`.top`, `.mdp`, `.gro`), which means all simulation parameters are locked into the binary `.tpr` before `mdrun` starts. This compile-then-run model makes GROMACS simulations highly reproducible but requires re-running `grompp` whenever any parameter changes.

Several domain assumptions govern GROMACS applicability:

- **Fixed topology:** The bond list, angle list, dihedral list, and improper list are defined at the start of the simulation in the `.top` file and cannot change during the run. GROMACS is not a reactive MD code. Bond-breaking, proton transfer, and chemical reactions require reactive force fields (ReaxFF via PLUMED or an external code) or QM/MM.
- **Classical nuclei:** GROMACS integrates classical equations of motion. Zero-point energy and nuclear tunneling are ignored. For proton dynamics at low temperature, path-integral extensions (GROMACS + i-PI) are required.
- **Long-range electrostatics are always present in biomolecular systems.** Biomolecular and drug-like systems have partial charges on every atom. PME (Particle Mesh Ewald) is mandatory; a plain cutoff produces artifacts in electrostatic energy and forces that invalidate free energy calculations, solvation energies, and structural properties of charged systems.
- **Force-field and water-model pairing is non-negotiable.** AMBER ff14SB was parameterized with TIP3P; CHARMM36m with mTIP3P (a modified TIP3P with different LJ parameters). Using TIP4P with ff14SB or TIP3P with CHARMM36m introduces systematic density and structural errors that cannot be corrected by re-equilibration.
- **GROMACS units:** nm, kJ/mol, ps, K, bar. All `.mdp` distance parameters (rcoulomb, rvdw, rlist, box vectors) are in **nanometres**, not Ångströms. This is the single most frequent source of silently wrong input from users familiar with LAMMPS or AMBER.

## When to Use This Skill

- Biomolecular simulations: proteins, nucleic acids, lipid bilayers, carbohydrates, protein–ligand complexes in explicit solvent.
- Organic molecular liquids, polymer melts, and soft-matter systems where AMBER, CHARMM, OPLS-AA, or GROMOS force fields have validated parameters.
- Coarse-grained simulations using the Martini force field (Martini 2 or Martini 3) for membrane proteins, lipid self-assembly, or polymer systems at mesoscale.
- Free energy perturbation and alchemical calculations (GROMACS has the most mature free-energy infrastructure among open-source MD codes).
- Enhanced sampling with PLUMED: metadynamics, umbrella sampling, replica exchange MD.
- Large-scale biomolecular simulations (10⁴–10⁷ atoms) where GROMACS GPU offload and PME scaling are required for throughput.

## When Not to Use This Skill

- **Materials science simulations of metals, ceramics, or semiconductors.** EAM, Tersoff, Stillinger-Weber, and COMB are not natively supported in GROMACS. Use LAMMPS for materials force fields. GROMACS can use these systems via PLUMED or external FF interfaces, but the workflow is non-standard.
- **Reactive MD (bond-breaking).** Fixed topology precludes bond formation and breaking. Use LAMMPS with ReaxFF, or ML potentials in an appropriate code.
- **When a custom ML interatomic potential (MACE, NequIP, DeePMD) is the force field.** GROMACS has no native MLP pair-style interface. Use LAMMPS (see `mlp-to-lammps`) or ASE for MLP-driven MD.
- **Very large inorganic systems (> 10⁶ atoms) where PME is a bottleneck.** LAMMPS scales better for non-biomolecular systems because it can use truncated potentials without long-range electrostatics.
- **When only single-point energies and forces are needed.** Use ASE with the relevant calculator; GROMACS requires a full `.tpr` compilation and `mdrun` invocation even for a single energy evaluation.

## Core Concepts

- **`.top` (topology file):** The top-level text file that defines the entire system: which force-field parameter files to include, which molecule types exist, and the number of each molecule type in the simulation box. The `[ system ]` section gives a human-readable name; `[ molecules ]` lists molecule name + count in the order they appear in the coordinate file. The atom count in `[ molecules ]` must exactly match the coordinate file.
- **`.itp` (include topology file):** Modular topology fragment included into `.top` via `#include`. Contains one or more `[ moleculetype ]` definitions with `[ atoms ]`, `[ bonds ]`, `[ angles ]`, `[ dihedrals ]`, `[ impropers ]`, and `[ pairs ]` sections. Force-field libraries are distributed as collections of `.itp` files (e.g., `amber14sb.ff/aminoacids.itp`, `charmm36-jul2022.ff/proteins.itp`).
- **`.tpr` (run input file):** Binary file produced by `grompp`. Contains the complete simulation state: topology, mdp parameters, and starting coordinates. Once compiled, the `.tpr` cannot be edited; any change requires re-running `grompp`. The `.tpr` is the canonical input to `mdrun` and ensures reproducibility — the same `.tpr` always produces the same trajectory.
- **`.mdp` (molecular dynamics parameter file):** Plain-text file specifying all simulation parameters: integrator, timestep, output frequencies, cutoffs, thermostat, barostat, constraints, and PME accuracy. One `.mdp` per stage (EM → NVT → NPT → production). Template `.mdp` files from CHARMM-GUI or the GROMACS manual are the recommended starting points.
- **`.gro` (coordinate file):** GROMACS format for atomic positions (and optionally velocities), one atom per line, with box vectors at the end. Coordinates are in nanometres. Can be read by ASE and MDAnalysis. Not suitable for long trajectories (one frame only); use `.xtc` or `.trr` for trajectories.
- **`.xtc` (compressed trajectory):** Single-precision lossy-compressed trajectory. Stores positions only; forces and velocities are not included. Standard format for long production runs. Readable by MDAnalysis, VMD, PyMOL, and OVITO.
- **`.trr` (full-precision trajectory):** Full-precision trajectory storing positions, velocities, and forces at each saved frame. Required for: velocity-dependent observables (VACF, kinetic energy per molecule), free energy calculations, and restart checkpoints. Significantly larger than `.xtc`.
- **`.edr` (energy file):** Binary file containing all thermodynamic observables (temperature, pressure, potential energy, kinetic energy, density, volume, box dimensions, and force-field energy terms) at every `nstenergy` steps. Parsed with `gmx energy` or the `panedr` Python library. This is the primary source for thermodynamic averages; do not use the `.log` file for analysis (it has lower precision).
- **`.cpt` (checkpoint file):** Binary restart file written every `nstcheckpoint` steps (default: every hour of wall time). Contains full simulation state: positions, velocities, random number generator state, and simulation step counter. Use with `gmx mdrun -cpi prefix.cpt` to continue a run without loss of statistics.
- **`grompp`:** Preprocessor that compiles `.mdp` + `.top` + `.gro` → `.tpr`. Performs extensive topology and parameter validation before the run starts. `grompp` warnings should be treated as errors — they indicate physical inconsistencies that will silently corrupt the simulation. Use `-maxwarn 0` only when you understand every warning and have confirmed it is safe.
- **`mdrun`:** The simulation engine. Reads a `.tpr` and produces `.xtc` / `.trr`, `.edr`, `.log`, `.cpt`, and final `.gro`. The most performance-critical step; uses MPI, OpenMP, and GPU offload.
- **LINCS (Linear Constraint Solver):** GROMACS default constraint algorithm for bond-length constraints. Applied to bonds involving hydrogen (h-bonds setting) or all bonds (all-bonds). Implemented via a matrix factorization that is parallel-efficient and numerically stable. `lincs-order = 4` is the default; increase to 6–8 for stiff bonds or large timesteps. LINCS replaces SHAKE for most GROMACS applications.
- **SETTLE:** Analytical constraint algorithm for three-site rigid water models (TIP3P, SPC, SPC/E). Faster and more accurate than LINCS for three-point water. Applied automatically to all water molecules when `constraints = h-bonds` or higher and a rigid water model is used.
- **V-rescale thermostat (Bussi–Donadio–Parrinello):** Stochastic velocity rescaling that generates the canonical (NVT) ensemble correctly. Temperature fluctuations follow the Maxwell-Boltzmann distribution. Recommended for production NVT. Replaces Berendsen for all production applications. `tau-t = 0.1 ps` is a typical coupling constant. [EXPERT REVIEW NEEDED]
- **Parrinello-Rahman barostat:** Correct NPT ensemble; generates proper pressure fluctuations and compressibility. Requires pre-equilibrated density (start from Berendsen NPT if the system is far from equilibrium — P-R oscillates dangerously from cold starts). `tau-p = 2.0 ps` is typical for condensed-phase systems. [EXPERT REVIEW NEEDED]
- **Verlet cutoff scheme:** GROMACS 2018+ exclusively uses the Verlet cutoff scheme with a buffer region determined by the `verlet-buffer-tolerance` parameter. The older `group` scheme was removed in GROMACS 2021. The Verlet scheme guarantees energy conservation and is required for GPU offload.

## Key Workflows

### Workflow 1: Topology construction — the `.top`/`.itp` hierarchy

The topology is the most error-prone part of a GROMACS setup. Every atom in the coordinate file must have a matching entry in `[ molecules ]`; every molecule type must be defined in an included `.itp`; every atom type must appear in the force-field nonbonded parameter database.

```
# ── Annotated .top file for a protein-in-water system ────────────────────────
# File: topol.top
# System: Hen lysozyme (HEWL) in TIP3P water, AMBER ff14SB force field

; Force field includes (from amber14sb.ff/ directory)
; Order matters: nonbonded parameters must be included before bonded parameters.
#include "amber14sb.ff/forcefield.itp"     ; Defines [ defaults ]: fudgeLJ, fudgeQQ, gen-pairs
                                            ; Includes ffnonbonded.itp: atom types, LJ params
                                            ; Includes ffbonded.itp: bond, angle, dihedral params

; Protein molecule type (generated by pdb2gmx or CHARMM-GUI)
#include "protein.itp"                     ; Contains [ moleculetype ] Protein_chain_A
                                            ;   [ atoms ], [ bonds ], [ angles ], [ dihedrals ],
                                            ;   [ impropers ], [ pairs ]

; Position restraints for the protein (applied during NVT/NPT equilibration)
; Guarded by ifdef — applied only when -DPOSRES is passed to grompp
#ifdef POSRES
#include "posre.itp"                       ; [ position_restraints ] with k = 1000 kJ/mol/nm²
#endif

; Water model — must match the force field
#include "amber14sb.ff/tip3p.itp"          ; TIP3P for AMBER ff14SB
; DO NOT use: tip4p.itp, spc.itp, or tips3p.itp with AMBER ff14SB
; CHARMM36m uses: charmm36-jul2022.ff/tip3p.itp (modified mTIP3P)

; Ions (Na+, Cl- to neutralize and add salt)
#include "amber14sb.ff/ions.itp"           ; Defines ion atom types

[ system ]
; Human-readable system name (does not affect simulation)
Hen Egg White Lysozyme in TIP3P water

[ molecules ]
; Molecule name  Count
; Order must exactly match the coordinate file (topol.gro / system.gro)
; Count must be exact — even one off causes grompp to reject the topology
Protein_chain_A  1
SOL              9173    ; TIP3P water molecules
NA               8       ; Sodium ions
CL               0       ; Chloride ions (adjust for desired salt concentration)
```

```python
"""
Topology construction with pdb2gmx (GROMACS built-in) or CHARMM-GUI.

pdb2gmx is the fastest path for standard proteins with standard residues.
CHARMM-GUI is the most reliable path for:
  - Lipid bilayers (CHARMM-GUI Membrane Builder)
  - Protein-ligand complexes (CHARMM-GUI Ligand Reader)
  - Glycoproteins (CHARMM-GUI Glycan Reader)
  - Coarse-grained Martini systems (CHARMM-GUI Martini Maker)

pdb2gmx workflow:
"""
import subprocess
from pathlib import Path


def build_topology_pdb2gmx(
    input_pdb: str,
    force_field: str = "amber14sb",    # or "charmm36m", "oplsaa", "gromos54a7"
    water_model: str = "tip3p",        # must match force field
    ignore_hydrogens: bool = False,
    histidine_state: str = None,       # None = interactive; "1" = all HID; see below
) -> dict:
    """
    Run gmx pdb2gmx to generate topology and processed coordinate files.

    pdb2gmx reads the PDB, assigns residue-based force-field parameters,
    adds missing hydrogens, and writes topol.top, topol.gro, and posre.itp.

    histidine_state: GROMACS asks for each His protonation state interactively.
    For batch runs:
      - Pass answers via stdin: echo "1 1 1 ..." | gmx pdb2gmx ...
      - Or use -his flag with a file specifying each His state:
        0 = HID (ε-protonated), 1 = HIE (δ-protonated), 2 = HIP (doubly protonated)
    Protonation state assignment is the single most impactful setup decision
    for proteins — wrong His state changes hydrogen bond networks and active sites.

    Water model / force-field compatibility:
      amber14sb   → tip3p (mTIP3P in GROMACS amber14sb.ff is standard TIP3P)
      charmm36m   → tip3p (mTIP3P — different LJ parameters from standard TIP3P)
      oplsaa      → tip4p (or spce)
      gromos54a7  → spce [EXPERT REVIEW NEEDED]
    """
    cmd = [
        "gmx", "pdb2gmx",
        "-f", input_pdb,
        "-o", "processed.gro",    # output coordinates
        "-p", "topol.top",        # output topology
        "-i", "posre.itp",        # position restraint file
        "-ff", force_field,
        "-water", water_model,
    ]
    if ignore_hydrogens:
        cmd.append("-ignh")       # Ignore existing H atoms; re-add based on force field

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"pdb2gmx FAILED:\n{result.stderr[-3000:]}")
        return {"success": False, "stderr": result.stderr}

    # Parse output statistics
    lines = result.stdout + result.stderr
    print(result.stdout[-2000:])
    return {"success": True, "output_gro": "processed.gro",
            "topology": "topol.top", "posre": "posre.itp"}


def add_solvent_and_ions(
    gro_file: str = "processed.gro",
    top_file: str = "topol.top",
    box_type: str = "dodecahedron",   # dodecahedron (most efficient for globular proteins)
    box_size: float = 1.0,            # nm — distance from protein to box edge
    concentration_M: float = 0.15,    # mol/L NaCl
    positive_ion: str = "NA",
    negative_ion: str = "CL",
) -> None:
    """
    Solvate and add ions to a protein system.

    Dodecahedron (rhombic dodecahedron) reduces the number of water molecules
    by ~29% compared to a cubic box for globular proteins — the protein-to-box
    volume ratio is better. Use a cubic box for membrane systems.

    box_size = 1.0 nm: minimum protein-image distance = 2 × 1.0 = 2.0 nm.
    For PME, the protein should not interact with its periodic image.
    With rcoulomb = 1.2 nm, a 2.0 nm minimum distance is sufficient but tight.
    Use 1.2–1.5 nm for highly charged proteins. [EXPERT REVIEW NEEDED]
    """
    # Step 1: define simulation box
    subprocess.run([
        "gmx", "editconf",
        "-f", gro_file, "-o", "boxed.gro",
        "-bt", box_type,
        "-d", str(box_size),
    ], check=True)

    # Step 2: fill with water
    subprocess.run([
        "gmx", "solvate",
        "-cp", "boxed.gro", "-cs", "spc216.gro",  # spc216.gro: pre-equilibrated water box
        "-o", "solvated.gro",
        "-p", top_file,                             # Updates [ molecules ] SOL count
    ], check=True)

    # Step 3: add ions (replace water molecules with Na+/Cl-)
    # First: compile a temporary .tpr for genion (genion needs a .tpr, not raw topology)
    subprocess.run([
        "gmx", "grompp",
        "-f", "ions.mdp",        # Minimal mdp: just integrator = steep, nsteps = 0
        "-c", "solvated.gro",
        "-p", top_file,
        "-o", "ions.tpr",
        "-maxwarn", "1",         # Allow the "no coordinates written" warning
    ], check=True)

    # Add ions to neutralize and reach target concentration
    p = subprocess.run([
        "gmx", "genion",
        "-s", "ions.tpr",
        "-o", "system.gro",
        "-p", top_file,
        "-pname", positive_ion,
        "-nname", negative_ion,
        "-neutral",              # Neutralize charge
        "-conc", str(concentration_M),
    ], input="SOL\n",            # Replace SOL (water) molecules with ions
    capture_output=True, text=True)

    print(p.stdout[-1000:])
    if p.returncode != 0:
        raise RuntimeError(f"genion failed:\n{p.stderr}")

    print(f"System built: system.gro + updated {top_file}")
    print("Check: cat topol.top | grep -A3 '\\[ molecules \\]'")
    print("Verify: NA and CL counts match expected charge neutralization")
```

### Workflow 2: `.mdp` parameter anatomy — four-stage protocol

A standard GROMACS simulation uses four separate `.mdp` files: energy minimization (EM), NVT equilibration, NPT equilibration, and production. Each stage has distinct parameter requirements.

```ini
; ── em.mdp: Energy minimization ──────────────────────────────────────────────
; Goal: remove steric clashes and overlaps from the initial structure
; before any dynamics. Forces > 1000 kJ/mol/nm after minimization indicate
; a structural problem that must be resolved before MD.

integrator  = steep         ; Steepest descent (most robust; use l-bfgs for finer convergence)
emtol       = 1000.0        ; kJ/mol/nm — stop when max force < this
emstep      = 0.01          ; Initial step size (nm); reduced automatically if energy rises
nsteps      = 100000        ; Maximum steps; increase if emtol not reached

; Output
nstxout     = 0
nstenergy   = 500
nstlog      = 500

; Cutoffs (consistent with production)
cutoff-scheme   = Verlet    ; Mandatory for GROMACS 2021+
nstlist         = 10
ns-type         = grid
rcoulomb        = 1.2       ; nm — Coulomb cutoff (long-range via PME)
rvdw            = 1.2       ; nm — vdW cutoff
rlist           = 1.2       ; nm — neighbor list cutoff (set = rcoulomb for Verlet)

; Long-range electrostatics
coulombtype     = PME
pme-order       = 4         ; Cubic interpolation (standard)
fourierspacing  = 0.16      ; nm — PME grid spacing; controls accuracy vs. speed

; vdW treatment
vdwtype         = cutoff
vdw-modifier    = force-switch   ; Smooth force truncation from rvdw-switch to rvdw
rvdw-switch     = 1.0            ; nm — begin switching at 1.0 nm

; No thermostat or barostat during EM
pcoupl  = no
tcoupl  = no

; Constraints: none during EM (constraints can prevent convergence)
constraints     = none
```

```ini
; ── nvt.mdp: NVT equilibration ───────────────────────────────────────────────
; Goal: equilibrate kinetic energy at target temperature before pressure coupling.
; Duration: 100–500 ps for most systems; longer for polymers or membranes.
; Do NOT use the final coordinates/velocities of this stage for production averages.

integrator  = md            ; Leap-frog Verlet
dt          = 0.002         ; ps = 2 fs (with h-bond constraints)
nsteps      = 50000         ; 100 ps (increase for slow-relaxing systems)

; Output (light during equilibration — do not write full trajectory)
nstxout             = 0     ; Do not write .trr positions (saves disk)
nstvout             = 0
nstfout             = 0
nstenergy           = 500   ; Write energies every 500 steps (1 ps)
nstlog              = 500
nstxout-compressed  = 1000  ; Write .xtc every 2 ps for monitoring

; Cutoffs (same as EM and production — must be consistent)
cutoff-scheme       = Verlet
nstlist             = 10
rcoulomb            = 1.2
rvdw                = 1.2
coulombtype         = PME
pme-order           = 4
fourierspacing      = 0.16
vdwtype             = cutoff
vdw-modifier        = force-switch
rvdw-switch         = 1.0

; Constraints — required for 2 fs timestep with H atoms
constraints         = h-bonds      ; Constrain all X-H bonds
constraint-algorithm = lincs
lincs-iter          = 1            ; 1 corrective iteration (default)
lincs-order         = 4            ; Expansion order (increase to 6 for stiff bonds)
; SETTLE is applied automatically to 3-site rigid water models.

; Temperature coupling — V-rescale (canonical ensemble)
tcoupl  = V-rescale
tc-grps = Protein  Non-Protein     ; Separate coupling groups for protein and solvent
tau-t   = 0.1      0.1             ; ps; thermostat time constant
ref-t   = 300.0    300.0           ; K
; WARNING: coupling groups must cover all atoms.
; Missing atoms → "atom X not covered by a temperature coupling group"
; For systems without protein: use "System" as the single group.

; No pressure coupling during NVT
pcoupl  = no

; Velocity generation (from Maxwell-Boltzmann at ref-t)
gen-vel     = yes
gen-temp    = 300.0
gen-seed    = 42            ; Fixed seed for reproducibility
                            ; Change seed for independent runs (different kinetics)

; Position restraints on protein heavy atoms (applied via -DPOSRES in grompp)
; Prevents protein from moving while solvent equilibrates around it.
; define = -DPOSRES
```

```ini
; ── npt.mdp: NPT equilibration ───────────────────────────────────────────────
; Goal: equilibrate density and pressure after NVT.
; Start from NVT checkpoint (-t nvt.cpt) to preserve velocities.
; Duration: 200 ps–2 ns; longer for lipid bilayers (5–20 ns). [EXPERT REVIEW NEEDED]
; NEVER use Parrinello-Rahman from a cold start — use Berendsen first.

integrator  = md
dt          = 0.002
nsteps      = 100000        ; 200 ps

; [same cutoff, constraint, and temperature coupling sections as NVT]
; Temperature coupling must match NVT exactly (groups, tau-t, ref-t)
tcoupl  = V-rescale
tc-grps = Protein  Non-Protein
tau-t   = 0.1      0.1
ref-t   = 300.0    300.0

; Pressure coupling
pcoupl          = Parrinello-Rahman
; NOTE: For a cold-start NPT (system not yet density-equilibrated),
; use pcoupl = Berendsen for the first 200 ps, then switch to Parrinello-Rahman.
; Berendsen converges density fast but gives wrong NPT ensemble (wrong fluctuations).
; Parrinello-Rahman is correct but oscillates violently from a high-stress start.
pcoupltype      = isotropic         ; isotropic: all three box dims coupled equally
                                     ; semiisotropic: xy coupled, z independent (membranes)
                                     ; anisotropic: all six components independent (crystals)
tau-p           = 2.0               ; ps; barostat time constant [EXPERT REVIEW NEEDED]
ref-p           = 1.013             ; bar (1 atm)
compressibility = 4.5e-5            ; bar⁻¹; isothermal compressibility of water at 300 K
                                     ; Set to ~0 for solids to prevent cell deformation
                                     ; Membranes: use semiisotropic with different compressibility

; Continue from NVT checkpoint (velocities from NVT, no new velocity generation)
gen-vel         = no
continuation    = yes               ; Tells grompp this is a continuation run
```

```ini
; ── production.mdp: Production NVT or NPT ────────────────────────────────────
; Goal: collect statistics for observables after confirmed equilibration.
; Always run gmx check -f npt.edr before starting production.

integrator  = md
dt          = 0.002
nsteps      = 500000000     ; 1 µs (adjust to target property timescale)

; Output: write more frequently than equilibration
nstxout             = 0
nstvout             = 0
nstfout             = 0
nstenergy           = 500        ; Energies every 1 ps
nstlog              = 5000       ; Log every 10 ps
nstxout-compressed  = 5000       ; .xtc every 10 ps (2 GB/µs for ~50,000 atoms)
; For velocity-dependent observables (VACF, kinetic energy per molecule):
; nstvout = 5000 and format = trr

; [same cutoff sections as NVT/NPT]
cutoff-scheme   = Verlet
nstlist         = 10
rcoulomb        = 1.2
rvdw            = 1.2
coulombtype     = PME
pme-order       = 4
fourierspacing  = 0.16
vdwtype         = cutoff
vdw-modifier    = force-switch
rvdw-switch     = 1.0

; [same constraints as NVT/NPT]
constraints             = h-bonds
constraint-algorithm    = lincs
lincs-iter              = 1
lincs-order             = 4

; Temperature coupling — Nose-Hoover preferred for production over V-rescale
; when computing heat capacity, velocity distributions, or kinetic quantities.
; V-rescale is fine for structural and thermodynamic averages. [EXPERT REVIEW NEEDED]
tcoupl  = Nose-Hoover
tc-grps = Protein  Non-Protein
tau-t   = 1.0      1.0           ; ps; longer than V-rescale — slower response
ref-t   = 300.0    300.0

; Pressure coupling — Parrinello-Rahman for correct NPT ensemble
pcoupl          = Parrinello-Rahman
pcoupltype      = isotropic
tau-p           = 2.0
ref-p           = 1.013
compressibility = 4.5e-5

gen-vel         = no
continuation    = yes
```

### Workflow 3: `grompp → mdrun` pipeline

The standard four-stage pipeline: each stage produces a checkpoint and coordinate file that feeds the next stage. Always check `grompp` output before submitting `mdrun` to a cluster.

```bash
#!/bin/bash
# ── Four-stage GROMACS pipeline ───────────────────────────────────────────────
# System: ~50,000 atom protein-in-water, AMBER ff14SB, TIP3P
# Environment: GROMACS 2024.2, 1 GPU (NVIDIA A100), 16 CPU threads

set -euo pipefail   # Exit on error; report undefined variables

GMX="gmx"           # Replace with "gmx_mpi" for MPI builds
MAXWARN=0           # Treat all grompp warnings as errors; investigate before increasing

# ── Stage 0: Energy minimization ─────────────────────────────────────────────
echo "=== Stage 0: Energy Minimization ==="
${GMX} grompp \
    -f em.mdp \
    -c system.gro \
    -p topol.top \
    -o em.tpr \
    -maxwarn ${MAXWARN}
# grompp checks: topology atom count vs. coordinate file; undefined atom types;
# missing bonded parameters; inconsistent force-field includes.
# Every WARNING from grompp must be understood before proceeding.

${GMX} mdrun -v -deffnm em -ntmpi 1 -ntomp 16

# Check minimization converged:
echo "--- EM convergence check ---"
${GMX} energy -f em.edr -o em_potential.xvg << EOF
Potential
0
EOF
# Potential energy should be negative and within 10% of expected cohesive energy.
# Max force should be < emtol (1000 kJ/mol/nm); check with:
grep "Maximum force" em.log | tail -1

# ── Stage 1: NVT equilibration ────────────────────────────────────────────────
echo "=== Stage 1: NVT Equilibration ==="
${GMX} grompp \
    -f nvt.mdp \
    -c em.gro \
    -r em.gro \               # -r: reference coordinates for position restraints
    -p topol.top \
    -o nvt.tpr \
    -maxwarn ${MAXWARN} \
    -define "-DPOSRES"        # Activates position restraints (#ifdef POSRES in .top)

${GMX} mdrun -v -deffnm nvt -ntmpi 1 -ntomp 16 -gpu_id 0 \
    -nb gpu -pme gpu -bonded gpu

# Check temperature equilibration:
${GMX} energy -f nvt.edr -o nvt_temperature.xvg << EOF
Temperature
0
EOF
# Temperature should plateau at 300 K within 20–30 ps.
# Std dev for 50,000 atoms should be ~2–3 K (equipartition).

# ── Stage 2: NPT equilibration ────────────────────────────────────────────────
echo "=== Stage 2: NPT Equilibration ==="
${GMX} grompp \
    -f npt.mdp \
    -c nvt.gro \              # Starting coordinates from NVT
    -r nvt.gro \              # Reference for position restraints
    -t nvt.cpt \              # -t: checkpoint → preserves velocities from NVT
    -p topol.top \
    -o npt.tpr \
    -maxwarn ${MAXWARN} \
    -define "-DPOSRES"

${GMX} mdrun -v -deffnm npt -ntmpi 1 -ntomp 16 -gpu_id 0 \
    -nb gpu -pme gpu -bonded gpu

# Check density equilibration:
${GMX} energy -f npt.edr -o npt_density.xvg << EOF
Density
0
EOF
# Density should converge within 1% of experimental value.
# Water at 300 K, 1 bar: ~997 kg/m³ ≈ 0.997 g/cm³.

# ── Stage 3: Production ───────────────────────────────────────────────────────
echo "=== Stage 3: Production ==="
# Remove position restraints for production
${GMX} grompp \
    -f production.mdp \
    -c npt.gro \
    -t npt.cpt \              # Velocities and box from NPT
    -p topol.top \
    -o production.tpr \
    -maxwarn ${MAXWARN}
# Note: no -define "-DPOSRES" → position restraints disabled

${GMX} mdrun -v -deffnm production \
    -ntmpi 1 -ntomp 16 -gpu_id 0 \
    -nb gpu -pme gpu -bonded gpu \
    -maxh 23.5                # Terminate cleanly after 23.5 h (for 24 h HPC job)

echo "Production complete. Output: production.xtc, production.edr, production.log"
```

### Workflow 4: PME setup, cutoffs, constraints, and neighbor lists

PME configuration is the most common source of energy conservation failures. Parameters must be consistent across all stages.

```ini
; ── PME and cutoff parameter reference ───────────────────────────────────────
;
; GROMACS Verlet scheme: rlist is set automatically from verlet-buffer-tolerance.
; Explicitly setting rlist = rcoulomb is equivalent to tolerance = 0 (no buffer);
; use verlet-buffer-tolerance instead for energy-conserving production runs.
;
; verlet-buffer-tolerance = 0.005    ; kJ/mol/ps per atom (default)
;                                    ; GROMACS computes rlist automatically to satisfy this.
;                                    ; For NVE energy conservation: use 0.0005 or lower.

cutoff-scheme   = Verlet

; Long-range Coulomb
coulombtype     = PME
rcoulomb        = 1.2       ; nm; real-space cutoff for PME
pme-order       = 4         ; Cubic spline interpolation (4 = standard; 6 = higher accuracy)
fourierspacing  = 0.12      ; nm; FFT grid spacing. Smaller → more accurate but slower.
                             ; 0.16 nm: standard for biomolecular MD
                             ; 0.12 nm: for free energy calculations or PME energy accuracy
ewald-rtol      = 1e-5      ; Relative RMS error in real-space Ewald sum (default: 1e-5)

; Short-range vdW
rvdw            = 1.2       ; nm; must equal rcoulomb for Verlet buffer calculation
vdwtype         = cutoff
vdw-modifier    = force-switch   ; CHARMM36 convention: force switched from rvdw-switch to rvdw
rvdw-switch     = 1.0            ; nm; begin switching at 1.0 nm
; Alternative for AMBER: vdw-modifier = Potential-shift (simpler, slightly different energetics)

; Neighbor list
nstlist         = 10        ; Rebuild neighbor list every 10 steps (20 fs at 2 fs timestep)
                             ; GROMACS updates the buffer automatically for Verlet; do not
                             ; set nstlist too small (< 5) — excessive rebuilding kills performance.
                             ; Set nstlist = 1 only for debugging (forces consistent).

; ── Constraint algorithm selection ───────────────────────────────────────────
;
; constraints = none         → No constraints; dt ≤ 0.5 fs for X-H bonds
; constraints = h-bonds      → Constrain all X-H bonds (recommended default); dt ≤ 2 fs
; constraints = all-bonds    → Constrain all bonds; dt ≤ 4 fs; removes all stretching modes
; constraints = h-angles     → Constrain X-H bonds + angles; dt ≤ 4 fs [EXPERT REVIEW NEEDED]
; constraints = all-angles   → All bonds + angles; very aggressive; rarely needed
;
; For TIP3P/SPC water: SETTLE is applied automatically (overrides LINCS for water).
; For TIP4P/TIP5P: SETTLE is applied to the rigid geometry defined in the water .itp.
;
constraints             = h-bonds
constraint-algorithm    = lincs
lincs-iter              = 1         ; Corrective iterations (1 = default; 2 for stiff systems)
lincs-order             = 4         ; Expansion order; increase to 6 for dt = 4 fs
lincs-warnangle         = 30        ; Degrees; warn if a constrained bond rotates > 30° per step
                                     ; LINCS warnings during production indicate the timestep
                                     ; is too large or the force field has a problem.
```

```python
"""
Checking PME consistency across stages and diagnosing PME errors.

The most common PME error: rcoulomb and rvdw are inconsistent between
stages (one stage uses 1.0 nm, another uses 1.2 nm). grompp catches this
for re-runs but not between independent stages.

A second common error: fourierspacing in the mdp does not match
the box dimensions, causing gmx grompp to print:
  "WARNING: The grid size for PME is too small to be efficient"
The FFT grid dimensions should be factorizable into 2³×3²×5 for efficiency.
"""

import subprocess
import re


def check_pme_grid(box_nm: tuple, fourierspacing: float = 0.16) -> dict:
    """
    Compute PME FFT grid dimensions for a given box and check efficiency.

    GROMACS rounds the FFT grid up to the nearest "FFT5" number
    (factorizable as 2^a × 3^b × 5^c). If the computed grid does not
    satisfy this, GROMACS will round up, reducing effective resolution.

    box_nm: (lx, ly, lz) in nm.
    Returns grid dimensions and warnings.
    """
    def next_fft5(n: int) -> int:
        """Find smallest FFT5 number >= n."""
        x = n
        while True:
            m = x
            for p in (2, 3, 5):
                while m % p == 0:
                    m //= p
            if m == 1:
                return x
            x += 1

    grid = []
    warnings = []
    for i, (dim, label) in enumerate(zip(box_nm, ("x", "y", "z"))):
        n_raw = int(dim / fourierspacing) + 1
        n_fft5 = next_fft5(n_raw)
        grid.append(n_fft5)
        actual_spacing = dim / n_fft5
        if actual_spacing > fourierspacing * 1.15:
            warnings.append(
                f"PME grid {label}: requested spacing {fourierspacing:.3f} nm → "
                f"actual {actual_spacing:.3f} nm ({n_fft5} grid points). "
                "Grid is coarser than requested; reduce fourierspacing or box size."
            )

    print(f"PME grid: {grid[0]} × {grid[1]} × {grid[2]}")
    for w in warnings:
        print(f"WARNING: {w}")
    if not warnings:
        print(f"PME grid: consistent with fourierspacing = {fourierspacing} nm")

    return {"grid": grid, "warnings": warnings}


def verify_cutoff_consistency(mdp_files: list[str]) -> list[str]:
    """
    Parse multiple mdp files and verify rcoulomb, rvdw, and coulombtype
    are identical across all stages.

    Returns list of inconsistency warnings.
    """
    params = {}
    keys_to_check = ["rcoulomb", "rvdw", "coulombtype", "pme-order",
                     "fourierspacing", "vdwtype", "vdw-modifier", "constraints"]
    issues = []

    for mdp_file in mdp_files:
        file_params = {}
        with open(mdp_file) as f:
            for line in f:
                stripped = line.strip()
                if stripped.startswith(";") or "=" not in stripped:
                    continue
                key, _, val = stripped.partition("=")
                key = key.strip().lower()
                val = val.split(";")[0].strip()
                if key in keys_to_check:
                    file_params[key] = val
        params[mdp_file] = file_params

    # Cross-check
    reference_file = mdp_files[0]
    for key in keys_to_check:
        ref_val = params[reference_file].get(key)
        if ref_val is None:
            continue
        for mdp_file in mdp_files[1:]:
            val = params[mdp_file].get(key)
            if val is not None and val != ref_val:
                issues.append(
                    f"INCONSISTENCY: {key} differs between {reference_file} "
                    f"({ref_val}) and {mdp_file} ({val})"
                )

    if not issues:
        print(f"Cutoff consistency check PASSED across {len(mdp_files)} mdp files")
    for issue in issues:
        print(f"WARNING: {issue}")

    return issues
```

### Workflow 5: Thermostat and barostat selection

The coupling algorithm determines the statistical ensemble. Choosing the wrong one invalidates thermodynamic fluctuation-based properties and can cause non-physical trajectories.

```ini
; ── Thermostat reference ─────────────────────────────────────────────────────
;
; V-rescale (Bussi-Donadio-Parrinello 2007):
;   + Correct canonical ensemble (correct temperature fluctuations)
;   + Fast equilibration; numerically stable
;   + Recommended for both equilibration and production NVT
;   - Stochastic: trajectories are not time-reversible
;   - Slightly different dynamical properties than deterministic thermostats
;   tau-t = 0.1–1.0 ps; tc-grps must cover all atoms
;
; Nose-Hoover:
;   + Deterministic; time-reversible
;   + Correct canonical ensemble for ergodic systems
;   - Slow equilibration (the extended variable has inertia)
;   - Unsafe for non-ergodic systems (use with NHCH chains for complex systems)
;   - Can cause temperature oscillations with tau-t too short
;   tau-t = 0.5–2.0 ps; nhchainlength = 10 (GROMACS default) [EXPERT REVIEW NEEDED]
;   Preferred when: VACF or velocity distribution must be unperturbed by noise.
;
; Berendsen (NEVER for production):
;   + Fast equilibration (exponential decay to target T)
;   - NOT canonical ensemble: temperature fluctuations suppressed
;   - Heat capacity, free energy, and kinetic energy distributions are wrong
;   Use only for: initial geometry relaxation from a very bad starting structure
;
; sd (stochastic dynamics / Langevin):
;   Integrator, not just a thermostat: replaces md + tcoupl=V-rescale.
;   Good for: implicit solvent, membrane simulations where viscous damping
;   is physically motivated, equilibration of polymer melts.
;   Not recommended for computing transport properties.

; ── Barostat reference ───────────────────────────────────────────────────────
;
; Parrinello-Rahman:
;   + Correct NPT ensemble (correct pressure fluctuations)
;   + Compressibility, volume fluctuations, enthalpy are correct
;   - Oscillates from cold starts; use Berendsen first for density pre-equilibration
;   - Requires tau-p ≥ 2 ps to avoid volume oscillations
;   tau-p = 2.0–5.0 ps [EXPERT REVIEW NEEDED]; compressibility = 4.5e-5 bar⁻¹ (water)
;
; C-rescale (GROMACS 2021+):
;   + Stochastic analogue of Berendsen: fast convergence + correct NPT ensemble
;   + Recommended for equilibration AND production (best of both worlds)
;   + Especially good for heterogeneous systems and membranes [EXPERT REVIEW NEEDED]
;   tau-p = 0.5–2.0 ps
;
; Berendsen (NEVER for production):
;   + Fast density equilibration
;   - Incorrect NPT ensemble (wrong compressibility, wrong volume fluctuations)
;   - Free energy calculations, compressibility estimates are invalid under Berendsen

; ── Membrane-specific coupling ────────────────────────────────────────────────
; Membranes require semiisotropic pressure coupling:
; - Lateral (xy): coupled together at ref-p (1.013 bar); surface tension preserved
; - Normal (z): independent at ref-p
pcoupltype      = semiisotropic
ref-p           = 1.013  1.013      ; bar; [lateral, normal] or [1, 1] bar
compressibility = 4.5e-5  4.5e-5    ; bar⁻¹ for lipid membranes [EXPERT REVIEW NEEDED]
; Some membrane simulations use a surface tension ensemble:
; ref-p = 0  1.013 with compressibility = 4.5e-5  4.5e-5
; where 0 bar lateral + normal coupling maintains a target area per lipid. [EXPERT REVIEW NEEDED]
```

### Workflow 6: HPC execution — MPI, OpenMP, and GPU offload

GROMACS performance depends critically on matching the parallelization strategy to the available hardware and system size. Misconfigured parallelization can reduce throughput by 10×.

```bash
#!/bin/bash
# ── HPC GROMACS performance guide ────────────────────────────────────────────
#
# GROMACS parallel execution model:
#   - ntmpi:  number of thread-MPI ranks (processes within one node)
#             each rank handles a spatial domain of atoms
#   - ntomp:  OpenMP threads per rank (shared-memory parallelism within one rank)
#   - gpu_id: which GPU(s) to use (0-indexed, comma-separated for multiple GPUs)
#   - nb gpu: nonbonded (short-range) forces computed on GPU
#   - pme gpu: PME reciprocal-space computed on GPU (GROMACS 2018+)
#   - bonded gpu: bonded (harmonic, dihedral) forces on GPU (GROMACS 2020+)
#
# Guiding rules:
#   1. Use ntmpi = n_GPUs for GPU-dominated runs (1 rank per GPU)
#   2. Use ntomp = (total_cpu_threads / ntmpi) to fill all CPU cores
#   3. Domain decomposition: ntmpi domains × ntomp threads/domain
#   4. Each domain must contain at least ~500 atoms for good load balance [EXPERT REVIEW NEEDED]
#   5. For PME on GPU: ntmpi ≥ 2 is required (separate PME rank needs one rank)
#
# Single node, 1 GPU, 16 CPU cores (most common HPC setup):
GMX_1GPU="gmx mdrun -v -deffnm production -ntmpi 1 -ntomp 16 -gpu_id 0 -nb gpu -pme gpu -bonded gpu"

# Single node, 4 GPUs, 64 CPU cores:
GMX_4GPU="gmx mdrun -v -deffnm production -ntmpi 4 -ntomp 16 -gpu_id 0,1,2,3 -nb gpu -pme gpu -bonded gpu"

# Multi-node with MPI (requires gmx_mpi build):
# mpirun -n 64 gmx_mpi mdrun -v -deffnm production -ntomp 4 -nb gpu -pme gpu -bonded gpu -maxh 23.5

# Performance tuning workflow:
# Step 1: Run gmx tune_pme to find optimal PME fraction and grid size
gmx tune_pme -v -mdp production.mdp -f production.tpr \
    -ntmpi 1 -ntomp 16 -gpu_id 0 \
    -maxh 0.5      # Run test for 0.5 hours

# tune_pme tries different -npme (number of dedicated PME ranks) values
# and reports ns/day for each. Use the recommended -npme value in production.

# Step 2: Run with optimized settings
# gmx mdrun -v -deffnm production -ntmpi 1 -ntomp 16 -gpu_id 0 \
#     -nb gpu -pme gpu -bonded gpu -npme 0 \
#     -maxh 23.5

# Performance expectations (rough, system-dependent): [EXPERT REVIEW NEEDED]
# 50,000 atoms, 1 A100 GPU:     ~500–1000 ns/day
# 200,000 atoms, 4 A100 GPUs:   ~200–500 ns/day
# 1,000,000 atoms, 8 A100 GPUs: ~50–200 ns/day

# ── Domain decomposition and load balancing ───────────────────────────────────
# GROMACS automatically determines domain decomposition from ntmpi.
# Manual override: -dd 2 2 4 (2×2×4 grid for 16 ranks)
# Dynamic load balancing: enabled by default (-dlb auto)
# Disable for debugging: -dlb no

# ── Continuation after job failure or time limit ─────────────────────────────
# GROMACS writes checkpoint (.cpt) every hour by default.
# Continue with -cpi:
gmx mdrun -v -deffnm production -cpi production.cpt \
    -ntmpi 1 -ntomp 16 -gpu_id 0 -nb gpu -pme gpu -bonded gpu \
    -maxh 23.5 -append     # -append: extend existing trajectory files
```

### Workflow 7: Analysis — energy, coordinates, and structural observables

All GROMACS analysis starts with coordinate pre-processing (`gmx trjconv`) to handle PBC and centering, then extraction of specific observables.

```bash
#!/bin/bash
# ── Standard GROMACS analysis pipeline ────────────────────────────────────────

# ── 1. Inspect trajectory integrity ──────────────────────────────────────────
gmx check -f production.xtc 2>&1 | head -20
# Reports: number of frames, total time, frame interval, box dimensions.
# Warning: "Inconsistent box" → PBC cell changed unexpectedly.

# ── 2. Extract thermodynamic observables from .edr ───────────────────────────
# Energy terms available:
gmx energy -f production.edr -o /dev/null << EOF
0
EOF
# (entering 0 lists all available terms)

# Extract key observables:
for term in Temperature Pressure Density "Potential Energy" "Kinetic En." "Total Energy" Volume; do
    gmx energy -f production.edr -o "${term// /_}.xvg" << EOF
$term
0
EOF
done

# ── 3. PBC correction and centering ─────────────────────────────────────────
# Trajectory from mdrun has atoms that may be split across PBC boundaries.
# trjconv re-centers and makes molecules whole before any structural analysis.

# Step 3a: Make molecules whole (fix split molecules across PBC)
gmx trjconv \
    -f production.xtc -s production.tpr \
    -o whole.xtc \
    -pbc mol -center << EOF
Protein
System
EOF
# First selection: group to center on (Protein)
# Second selection: group to output (System = all atoms)

# Step 3b: Remove residual center-of-mass drift (optional, for diffusion analysis)
gmx trjconv \
    -f whole.xtc -s production.tpr \
    -o nojump_nofit.xtc \
    -pbc nojump << EOF
System
EOF
# nojump: corrects atoms that jump across the periodic boundary between frames
# Required for: MSD, diffusion coefficients, unwrapped trajectories

# Step 3c: Fit to reference structure (for RMSD, RMSF)
gmx trjconv \
    -f whole.xtc -s production.tpr \
    -o fitted.xtc \
    -fit rot+trans << EOF
Backbone
Protein
EOF
# Fit rotation+translation to Backbone atoms; output Protein coordinates

# ── 4. Structural observables ─────────────────────────────────────────────────
# RMSD of protein backbone
gmx rms -f fitted.xtc -s production.tpr -o rmsd_backbone.xvg << EOF
Backbone
Backbone
EOF

# RMSF per residue
gmx rmsf -f fitted.xtc -s production.tpr -o rmsf_residue.xvg -res << EOF
C-alpha
EOF

# RDF: protein–water oxygen
gmx rdf -f whole.xtc -s production.tpr -o rdf_prot_water.xvg \
    -ref "Protein" -sel "name OW" -bin 0.005 -rmax 2.0

# Hydrogen bonds: protein–protein
gmx hbond -f whole.xtc -s production.tpr \
    -num hbond_protein_count.xvg << EOF
Protein
Protein
EOF

# SASA (solvent-accessible surface area)
gmx sasa -f whole.xtc -s production.tpr -o sasa_protein.xvg << EOF
Protein
EOF

# ── 5. Index groups ───────────────────────────────────────────────────────────
# Custom index groups for non-standard selections
gmx make_ndx -f production.tpr -o custom.ndx << EOF
a CA & r 1-50    ; C-alpha atoms of residues 1-50
name 20 N_terminus_CA
q
EOF
# Index groups are referenced by name in all gmx analysis commands via -n custom.ndx

# ── 6. Python post-processing with panedr (recommended over gmx energy) ──────
python3 - << 'PYEOF'
import panedr
import pandas as pd
import numpy as np

# Load entire .edr as a DataFrame (faster and more flexible than gmx energy)
df = panedr.edr_to_df("production.edr")
print(f"Available terms: {list(df.columns[:15])}")
print(f"Frames: {len(df)}, Time range: {df['Time'].min():.1f}–{df['Time'].max():.1f} ps")

# Production averages (discard first 20% as additional safety margin)
n_skip = len(df) // 5
prod = df.iloc[n_skip:]

for term in ["Temperature", "Pressure", "Density", "Potential Energy", "Total Energy"]:
    if term in prod.columns:
        val = prod[term]
        print(f"  {term:<25}: {val.mean():.4f} ± {val.std():.4f} "
              f"(block SE ≈ {val.std() / np.sqrt(len(val) / 500):.4f})")
PYEOF
```

### Workflow 8: Restart, continuation, and reproducibility

```bash
#!/bin/bash
# ── Checkpoint-based continuation ─────────────────────────────────────────────
# GROMACS writes .cpt every nstcheckpoint steps (wall time) AND at job end.
# The .cpt contains: positions, velocities, random number state, step counter.

# Continue from last checkpoint (append to existing trajectory):
gmx mdrun -v -deffnm production -cpi production.cpt \
    -ntmpi 1 -ntomp 16 -gpu_id 0 -nb gpu -pme gpu \
    -maxh 23.5 -append

# Continue with new output prefix (cleaner for job array restarts):
gmx mdrun -v \
    -s production.tpr \       # Same .tpr as original run
    -cpi production.cpt \
    -deffnm production_part2 \
    -ntmpi 1 -ntomp 16 \
    -noappend               # Write new trajectory files

# Concatenate trajectory parts after all jobs complete:
gmx trjcat \
    -f production.xtc production_part2.xtc \
    -o production_full.xtc \
    -settime                # Adjust time stamps to be continuous

# ── Extending a run beyond nsteps ─────────────────────────────────────────────
# Method 1: Convert checkpoint to extended .tpr
gmx convert-tpr \
    -s production.tpr \
    -extend 1000000 \       # Additional 1 ns (in ps)
    -o production_extended.tpr

gmx mdrun -v -deffnm production_ext \
    -s production_extended.tpr \
    -cpi production.cpt \
    -ntmpi 1 -ntomp 16 -nb gpu -pme gpu -append

# ── Reproducibility record ─────────────────────────────────────────────────────
# For each production run, record:
gmx -version 2>&1 | grep "GROMACS version"         # GROMACS version
md5sum *.itp topol.top *.mdp                        # Topology and parameter hashes
sha256sum production.tpr                             # Binary run input (fully determines simulation)
# The .tpr hash is the single most compact reproducibility record:
# the same .tpr + same GROMACS version + same hardware architecture
# produces the same trajectory.
```

## Common Failure Modes

| Failure | Symptom | Likely Cause | Diagnostic | Fix |
|---------|---------|--------------|------------|-----|
| Topology atom count mismatch | `grompp` error: "number of coordinates in coordinate file does not match topology" | `[ molecules ]` count in `.top` does not match number of atoms in `.gro`; often caused by manually editing `.gro` or `.top` without updating the other | `grep -c "^" processed.gro` (subtract 3 for header + box line = N atoms); compare to sum of molecule counts × atoms per molecule in `.top` | Re-run `pdb2gmx` + `solvate` + `genion` from scratch; never manually edit atom counts in topology or coordinate file |
| Net charge not zero | `grompp` warning: "System has a net charge of +8e"; PME with a non-neutral cell introduces a uniform background charge that biases electrostatic energies | Missing counterions; or genion used without `-neutral`; or wrong charge on a non-standard residue | `gmx grompp` prints the net charge; cross-check with `grep "charge" processed.gro` or sum charges in `.itp` `[ atoms ]` section | Add counterions with `gmx genion -neutral -conc 0.15`; verify GAFF/CGENFF charges for non-standard residues sum to the expected integer |
| LINCS warning / constraint failure | `LINCS WARNING: Step X, 3 atoms involved in constraint angle larger than 30 degrees` in `.log`; trajectory shows atoms with very high velocities | Timestep too large for the constraint solver; very large forces on constrained bonds from bad initial geometry; wrong constraint topology | Check `grep LINCS production.log | wc -l`; if > 0 per 10,000 steps, reduce timestep; grep for "Dangerous Build" in log | Minimize with `constraints = none` first; reduce dt from 2 fs to 1 fs; increase `lincs-order` from 4 to 6; verify that all bond types in `.itp` are correct |
| Energy explosion in first 10 ps | Total energy diverges; temperature spikes to > 10,000 K; `gmx mdrun` exits with "Reason: Too many LINCS warnings" | Overlapping atoms in initial structure; starting MD directly from PDB without EM; position restraints not applied during NVT so solvent atoms overlap protein | Check `grep "Maximum force" em.log`; if max force > 10,000 kJ/mol/nm, minimization has not converged | Re-run EM with `emtol = 100` and `nsteps = 500000`; use `-define "-DPOSRES"` in NVT grompp; inspect the structure with OVITO or VMD for obvious overlaps |
| Wrong water model paired with force field | Density wrong by > 1%; protein–water interface structure differs from literature; solvation free energies off | TIP4P used with AMBER ff14SB (should use TIP3P), or standard TIP3P used with CHARMM36m (should use mTIP3P with different LJ parameters) | Compare simulated density to reference: TIP3P at 300 K ≈ 0.982 g/cm³; mTIP3P ≈ 1.002 g/cm³; TIP4P/2005 ≈ 0.997 g/cm³ | Match force field to water model exactly; amber14sb.ff ships `tip3p.itp` (TIP3P); charmm36-jul2022.ff ships `tip3p.itp` (mTIP3P) — these are different files with the same name |
| Inconsistent force-field includes | `grompp` error: "Residue 'LIG' not found in residue topology database"; or wrong LJ parameters for a ligand | Custom ligand `.itp` not included before the residue is referenced; or CGENFF `.itp` for a ligand is included after the force-field `.itp` that defines conflicting atom types | `grep "#include" topol.top`; ensure custom ligand `.itp` appears after force-field includes but before the `[ system ]` section | Add `#include "ligand.itp"` in the correct order; verify atom type names in ligand `.itp` do not clash with the force-field atom type database |
| PME/cutoff inconsistency between stages | `grompp` warning: "Inconsistent van der Waals treatment"; structural observables differ between continuation and new runs | `rcoulomb`, `rvdw`, `vdwtype`, or `vdw-modifier` differs between `.mdp` files for different stages | Run `verify_cutoff_consistency([em.mdp, nvt.mdp, npt.mdp, production.mdp])` (Workflow 4) | Standardize cutoff parameters across all `.mdp` files using a shared template; `rcoulomb = rvdw = 1.2 nm` is required for many GROMACS builds |
| Parrinello-Rahman box oscillation | Box vectors oscillate with period ≈ tau-p; density oscillates sinusoidally; NPT barostat does not settle | Parrinello-Rahman started from a structure far from equilibrium density; or tau-p too short | Plot `gmx energy -f npt.edr` density vs. time; if oscillating with period 2–4 ps, P-R is destabilized | Pre-equilibrate with `pcoupl = Berendsen` for 500–1000 ps; switch to Parrinello-Rahman only when density is within 0.5% of target; use `tau-p ≥ 2.0 ps` |
| Production started before equilibration | RMSD drifts during production; block means trend monotonically; results not reproducible between seeds; published observables differ from literature | NVT/NPT stages too short; position restraints removed before density converged; visual inspection of `.log` substituted for quantitative equilibration criteria | Parse `.edr` with `panedr`; compute block means of T, P, density, E; if block means drift monotonically over the first half of production, equilibration is incomplete | Extend equilibration (double NPT time); verify density vs. time and RMSD vs. time converge before declaring production ready; see `md-equilibration` for formal criteria |
| Incorrect `special_bonds` / 1-4 scaling | Protein secondary structure wrong; torsion profiles shift from QM reference; conformational free energies inconsistent with AMBER benchmark | `gen-pairs = yes` with OPLS-AA force field (requires GROMACS to generate 1-4 pairs) but `fudgeLJ`/`fudgeQQ` set to AMBER values; or `gen-pairs = no` when the force field requires it | Check `[ defaults ]` section in the force-field `.itp` for `gen-pairs`, `fudgeLJ`, and `fudgeQQ`; compare to the GROMACS force-field documentation | Use force-field files unmodified from the CHARMM-GUI output or the GROMACS force-field distribution; never manually change `fudgeLJ`/`fudgeQQ` in the `.top` |
| GPU offload performance regression | `mdrun` runs slower with GPU than without; `gmx mdrun -v` shows low GPU utilization | System too small for GPU offload (< 10,000 atoms); `-bonded gpu` with an older GPU without FP64 atomic operations; PME on GPU requires `ntmpi ≥ 2` but only 1 rank is used | Run `gmx mdrun` without `-nb gpu -pme gpu` and compare ns/day; check `gmx mdrun` log for "PP-PME load imbalance" | For systems < 10,000 atoms: use CPU-only (`-nb cpu -pme cpu`); for PME on GPU: ensure `ntmpi ≥ 2`; run `gmx tune_pme` to find optimal configuration |

## Best Practices

- **Always use `gmx grompp -maxwarn 0` for production runs.** Every warning from `grompp` indicates a physical inconsistency. Investigate and fix each warning before submitting to a cluster. Increasing `-maxwarn` to suppress warnings without understanding them is the most common cause of silently wrong GROMACS simulations.
- **Use CHARMM-GUI for all non-trivial system construction.** Manually building `.top` files for membrane proteins, protein-ligand complexes, or glycosylated systems is error-prone. CHARMM-GUI validates topology against the force-field database and provides consistent `.mdp` templates, index groups, and step-by-step scripts.
- **Verify atom count consistency at every step.** After `pdb2gmx`, `solvate`, and `genion`, inspect the `[ molecules ]` section and compare the total atom count to the `.gro` file line count minus 3. One mismatch → `grompp` will reject the topology.
- **Pre-equilibrate with Berendsen barostat before switching to Parrinello-Rahman.** Start every NPT simulation with `pcoupl = Berendsen` for 200–500 ps, then switch to Parrinello-Rahman. Parrinello-Rahman applied from a high-stress starting geometry oscillates uncontrollably and may crash the simulation. C-rescale (GROMACS 2021+) is an excellent alternative that converges quickly while producing the correct NPT ensemble.
- **Record the `.tpr` SHA-256 hash for every production run.** The `.tpr` encodes the complete simulation specification (topology + parameters + starting coordinates) in a single binary file. The hash fully identifies the simulation for reproducibility purposes. Store it in the run directory README alongside the GROMACS version.
- **Run `panedr` analysis rather than `gmx energy` for thermodynamic averages.** `panedr` reads `.edr` files directly into a pandas DataFrame, enabling block averaging, autocorrelation time estimation, and stationarity checks that the `gmx energy` `.xvg` output requires manual processing for. Install with `pip install panedr`.
- **Use `gmx trjconv -pbc mol -center` before any structural analysis.** Trajectories from `mdrun` have molecules split across PBC boundaries. All RDF, RMSD, and SASA calculations require whole molecules. Apply the correction pipeline (Workflow 7) before any analysis and save the processed trajectory separately.
- **Match `nstlist` and `rcoulomb` to the Verlet buffer.** Setting `nstlist = 10` with `rcoulomb = 1.2 nm` gives a neighbor-list buffer of ~0.05 nm. If performance is critical, increase `nstlist` to 20–40 and let `verlet-buffer-tolerance` set the buffer automatically. Do not set `nstlist = 1` in production — it is 5–10× slower with no accuracy benefit.
- **For HPC runs, always set `-maxh` to 1–2 hours below the job wall time.** `mdrun -maxh 23.5` for a 24-hour job ensures GROMACS writes a final checkpoint before the job is killed by the scheduler. Without `-maxh`, a job killed at the wall time may leave a corrupt or absent `.cpt`, forcing the entire run to restart.
- **Keep a clean copy of the pre-production `.tpr`.** The `.tpr` used for `gmx trjconv`, `gmx rdf`, `gmx rms`, and all other analysis commands must correspond to the production trajectory. If the `.tpr` is regenerated with different parameters, topology-based selections (atom type, residue, molecule) may produce wrong results. Never overwrite `production.tpr` after a production run begins.

## Integration with Other Skills

- **[md-equilibration](../md-equilibration/SKILL.md):** This skill provides GROMACS-specific mdp syntax for equilibration (V-rescale, Parrinello-Rahman, density checks with `gmx energy`). That skill provides the physical criteria and diagnostic workflows (block averaging, stationarity tests, flying ice cube detection) that determine when the GROMACS equilibration stages have succeeded. Both skills must be used together.
- **[md-analysis](../md-analysis/SKILL.md):** Workflow 7 here covers GROMACS-native analysis tools (`gmx rdf`, `gmx msd`, `gmx hbond`, `gmx trjconv`). That skill covers MDAnalysis- and OVITO-based analysis that applies to GROMACS trajectories loaded via the `.tpr` + `.xtc` interface. The PBC pre-processing from Workflow 7 here is a prerequisite for the analysis in that skill.
- **[md-force-fields](../md-force-fields/SKILL.md):** That skill covers the physical basis, parameterization domain, and transferability of AMBER, CHARMM, OPLS-AA, and GROMOS force fields. This skill provides the GROMACS-specific deployment: topology hierarchy, `.itp` structure, `[ defaults ]` section, and `special_bonds` settings that implement those force fields correctly in GROMACS.
- **[enhanced-sampling](../enhanced-sampling/SKILL.md):** GROMACS is the reference code for PLUMED-driven enhanced sampling (metadynamics, umbrella sampling, replica exchange MD). The production `.tpr` produced in Workflow 3 here is the starting point for PLUMED-patched `mdrun` in that skill. The equilibration protocol here is a prerequisite before enhanced sampling begins.
- **[free-energy-md](../free-energy-md/SKILL.md):** GROMACS has the most mature free energy perturbation (FEP) and alchemical transformation infrastructure among open-source MD codes. That skill covers the `lambda` coupling parameter mdp settings, the `dhdl.xvg` output, and MBAR/BAR analysis. This skill's equilibration and topology construction are prerequisites for any GROMACS FEP calculation.
- **[coarse-grained-md](../coarse-grained-md/SKILL.md):** The Martini coarse-grained force field (Martini 2 and Martini 3) runs natively in GROMACS using the same `.top`/`.itp`/`.mdp` infrastructure covered here, with modified cutoffs (rcoulomb = 1.1 nm, rvdw = 1.1 nm for Martini), energy-shift vdW modifier, and LINCS-free integration. That skill covers Martini-specific parameterization and CHARMM-GUI Martini Maker setup.
- **[reproducibility-workflow](../../scientific-data/reproducibility-workflow/SKILL.md):** The `.tpr` SHA-256 hash, GROMACS version, force-field `.itp` hashes, and `.mdp` files together form the minimal reproducibility record for a GROMACS simulation. That skill covers DVC-based data provenance, MLflow run tracking, and the `README` structure for archiving simulation inputs alongside results.
- **[scientific-visualization](../../scientific-data/scientific-visualization/SKILL.md):** GROMACS trajectories (`.xtc` + `.tpr`) are loaded by VMD, PyMOL, OVITO, and MDAnalysis for visualization. That skill covers rendering proteins with secondary structure coloring, lipid bilayer visualization, and per-atom property mapping from GROMACS analysis outputs (RMSF, SASA, order parameters).

## Key References

1. Abraham, M. J., et al. (2015). *GROMACS: High performance molecular simulations through multi-level parallelism from laptops to supercomputers.* SoftwareX, 1–2, 19–25. https://doi.org/10.1016/j.softx.2015.06.001
2. Bussi, G., Donadio, D., & Parrinello, M. (2007). *Canonical sampling through velocity rescaling.* Journal of Chemical Physics, 126(1), 014101. https://doi.org/10.1063/1.2408420
3. Parrinello, M. & Rahman, A. (1981). *Polymorphic transitions in single crystals: A new molecular dynamics method.* Journal of Applied Physics, 52(12), 7182–7190. https://doi.org/10.1063/1.328693
4. Hess, B., et al. (1997). *LINCS: A linear constraint solver for molecular simulations.* Journal of Computational Chemistry, 18(12), 1463–1472. https://doi.org/10.1002/(SICI)1096-987X(199709)18:12<1463::AID-JCC4>3.0.CO;2-H
5. Huang, J., et al. (2017). *CHARMM36m: An improved force field for folded and intrinsically disordered proteins.* Nature Methods, 14(1), 71–73. https://doi.org/10.1038/nmeth.4067
6. Maier, J. A., et al. (2015). *ff14SB: Improving the accuracy of protein side chain and backbone parameters from ff99SB.* Journal of Chemical Theory and Computation, 11(8), 3696–3713. https://doi.org/10.1021/acs.jctc.5b00255
7. Darden, T., York, D., & Pedersen, L. (1993). *Particle mesh Ewald: An N log(N) method for Ewald sums in large systems.* Journal of Chemical Physics, 98(12), 10089–10092. https://doi.org/10.1063/1.464397
