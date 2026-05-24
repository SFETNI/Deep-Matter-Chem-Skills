---
name: lammps-workflow
domain: atomistic-md
maturity: draft
tools:
  - LAMMPS
  - ASE
  - pymatgen
  - MDAnalysis
  - OVITO
tested_versions:
  LAMMPS: "2Aug2023"
  ASE: "3.23.0"
  MDAnalysis: "2.7.0"
  OVITO: "3.10.6"
requires_gpu: false
estimated_runtime: minutes-to-weeks
---

# LAMMPS Workflow

## Description

This skill covers end-to-end molecular dynamics simulations using LAMMPS (Large-scale Atomic/Massively Parallel Simulator): input script anatomy, force-field selection and deployment, ensemble control, equilibration protocol, production runs, restart logic, and parallel execution. LAMMPS is the dominant open-source MD code for materials science applications and the standard deployment target for ML interatomic potentials. Invoke this skill when setting up, running, diagnosing, or extending a LAMMPS simulation of any material class from metals to polymers to reactive systems.

## Domain Context

LAMMPS integrates Newton's equations of motion for a set of interacting particles under a chosen force field. The equations are solved numerically using a symplectic integrator (Velocity Verlet by default), which conserves a shadow Hamiltonian close to the true Hamiltonian and preserves time-reversibility. Key physical approximations:

- **Force field validity domain:** Every pair_style encodes specific physical assumptions. EAM is validated for FCC metals near equilibrium; it fails for surfaces, defects with large distortions, or alloys outside the training composition range. Tersoff is validated for covalent semiconductors; it fails for amorphous structures far from the fitting database. ReaxFF can model bond breaking/forming but has known failures for unusual coordination environments. The force field, not the integrator, is the dominant source of error.
- **Classical vs. quantum nuclei:** LAMMPS integrates classical equations of motion. Zero-point energy and nuclear quantum effects are neglected. This is a good approximation for heavy atoms (≥ C) above ~200 K, but fails for light atoms (H, He, Li) at low temperature and for tunneling-dominated processes. [EXPERT REVIEW NEEDED for precise crossover conditions]
- **Boundary conditions:** LAMMPS supports periodic (p), shrink-wrapped (s), and fixed (f) boundaries. Periodic boundaries impose artificial periodicity; the simulation box must be large enough that a particle does not interact with its own image through the cutoff.
- **Timestep and numerical stability:** The timestep must resolve the fastest vibrational frequency in the system. A rule of thumb is dt ≤ period / 20. For stiff bonds (O-H, C-H) in molecular force fields, this requires dt ≤ 0.5 fs or constrained bonds (SHAKE/RATTLE). For metals with EAM, 1–2 fs is typically safe. For ReaxFF with short-range repulsion, 0.1–0.5 fs is required. Exceeding the stability limit produces exponentially growing kinetic energy — the "energy explosion" failure mode.
- **Ergodicity and sampling:** A single MD trajectory samples the microcanonical or canonical ensemble only if it is ergodic on the timescale of the run. Rare events (phase transitions, defect migration, folding) may not be sampled. Enhanced sampling methods are required for free energy calculations.

LAMMPS uses a variety of internal unit systems. **Choosing the wrong `units` command is one of the most common and damaging silent errors in LAMMPS** — energies, forces, distances, and timesteps are all unit-dependent, and LAMMPS does not warn if the units are physically unreasonable for the force field being used.

## When to Use This Skill

- Classical MD simulations of metals, alloys, ceramics, semiconductors, polymers, or biomolecular systems.
- Deploying a trained ML interatomic potential (MACE, NequIP, ACE, DeePMD) for large-scale MD — LAMMPS has mature interfaces for all major MLP frameworks.
- Computing dynamical properties: diffusion coefficients, thermal conductivity, viscosity, phonon spectra from velocity autocorrelation.
- Mechanical property calculations: elastic constants, yield stress, crack propagation, dislocation dynamics.
- Generating thermally distorted structures for ML potential training datasets (active learning loops).
- Large-scale simulations (10³–10⁸ atoms) where DFT is computationally infeasible.

## When Not to Use This Skill

- **When the required property depends on quantum electronic structure** — use DFT (VASP, QE) or tight-binding for band gaps, charge transfer, optical properties, or electronically driven phase transitions.
- **When bond breaking/forming is needed but ReaxFF has not been validated for the system** — incorrect reactive potentials produce qualitatively wrong chemistry. Validate against DFT reaction pathways before use.
- **For small systems (< 50 atoms) at low temperature** where nuclear quantum effects dominate — use path integral MD (PIMD) or DFT-MD instead.
- **For biomolecular free energy calculations requiring strict ensemble control** — GROMACS or AMBER have more mature free-energy perturbation and enhanced sampling infrastructure.
- **When the timescale of interest (µs–ms) exceeds what plain MD can access** — use enhanced sampling (metadynamics, replica exchange, forward flux sampling) or coarse-grained models.

## Core Concepts

- **`units`:** Defines the unit system for all subsequent inputs and outputs. Must be set before all other commands. Common choices: `metal` (Å, eV, ps — standard for metallic systems and MLPs); `real` (Å, kcal/mol, fs — standard for molecular force fields); `lj` (dimensionless — for model systems); `si` (rarely used in practice). See the full table below. **The timestep, cutoffs, temperatures, and pressures are all in units-specific values.**
- **`atom_style`:** Determines what per-atom data is stored. `atomic` for simple metals/ceramics (position, velocity, force, type); `charge` adds per-atom charge (required for Coulombic interactions, ReaxFF); `molecular` adds bond/angle/dihedral topology; `full` adds charge + topology (required for CHARMM/AMBER force fields). Changing `atom_style` after the data file has been read is not possible — it must match the data file format.
- **`pair_style` / `pair_coeff`:** Selects the force field functional form and supplies parameters. `pair_style eam/alloy` + `pair_coeff * * potential.eam.alloy El1 El2` is the standard pattern for EAM. The element order in `pair_coeff` must match the atom type order in the data file or `create_atoms` commands.
- **`neighbor` / `neigh_modify`:** Controls the Verlet neighbor list. `neighbor 2.0 bin` sets a skin distance of 2.0 Å beyond the cutoff. `neigh_modify every 1 delay 0 check yes` rebuilds the list every step if atoms have moved more than skin/2. Too-large skin wastes memory; too-small skin misses interactions.
- **`fix nve/nvt/npt`:** The integrator fix controls the ensemble. `fix ID all nve` is the microcanonical ensemble (constant N, V, E). `fix ID all nvt temp Tstart Tstop Tdamp` adds Nosé-Hoover thermostat. `fix ID all npt temp Tstart Tstop Tdamp iso Pstart Pstop Pdamp` adds Parrinello-Rahman-style barostat. `Tdamp` and `Pdamp` are the thermostat/barostat relaxation times **in time units** — in `metal` units, Tdamp=100 means 100 ps, which is almost always wrong; 0.1 (100 fs) is a common starting point.
- **`thermo` / `thermo_style`:** Controls frequency and format of thermodynamic output. `thermo_style custom step temp press vol etotal ke pe` gives a human-readable log. The default style (`one`) prints minimal information; always override it for production runs.
- **`dump`:** Writes atomic configurations to a trajectory file. `dump ID all atom 1000 traj.lammpstrj` writes every 1000 steps. `dump ID all custom 1000 traj.dump id type x y z vx vy vz` is more portable and directly readable by OVITO and MDAnalysis.
- **`compute`:** Defines derived quantities that are evaluated on the fly. `compute pe all pe/atom` calculates per-atom potential energy. `compute stress all stress/atom NULL` calculates the per-atom virial stress tensor. `compute msd all msd com yes` tracks mean square displacement for diffusion.
- **Restart files:** `restart N prefix` writes binary restart files every N steps. Restarts preserve the full simulation state (positions, velocities, box, fixes). Use `read_restart` to continue. ASCII data files from `write_data` do not preserve velocities or fix state.
- **`run_style`:** Controls the time integration algorithm. Default `verlet` (Velocity Verlet) is appropriate for most cases. `respa` (rRESPA) allows different timesteps for different interaction ranges — useful for molecular systems with stiff bonds and soft long-range interactions.

## Key Workflows

### Workflow 1: Input script anatomy — annotated example for a metallic system

A complete, annotated LAMMPS input script for NVT MD of a BCC metal (W as example) using an EAM potential.

```lammps
# ============================================================
# NVT MD of W using EAM — annotated example
# LAMMPS units: metal (Å, eV, ps, K, bar, g/mol)
# ============================================================

# --- Initialization ---
units           metal           # Å, eV, ps, K, bar — required for EAM/MLP potentials
dimension       3
boundary        p p p           # Periodic in all three directions
atom_style      atomic          # No charges or topology needed for EAM

# --- Structure ---
# Option A: read from a data file (preferred for non-trivial geometries)
read_data       W_bcc.lammps

# Option B: create a perfect BCC lattice (MINIMAL EXAMPLE — not for production defect studies)
# lattice         bcc 3.165         # W lattice parameter in Å
# region          box block 0 10 0 10 0 10  # 10x10x10 unit cells
# create_box      1 box
# create_atoms    1 box
# mass            1 183.84          # g/mol

# --- Force field ---
pair_style      eam/alloy
pair_coeff      * * Ackland_W.eam.alloy W
# The element list (W) must match the atom type order in the data file.
# Source: Ackland & Thetford (1987) EAM for W; always cite the potential source.

# --- Neighbor list ---
neighbor        2.0 bin         # Skin distance: 2.0 Å beyond pair cutoff
neigh_modify    every 1 delay 0 check yes  # Rebuild if any atom moved > skin/2

# --- Initial velocity (if not reading from restart) ---
velocity        all create 300.0 42 mom yes rot yes dist gaussian
# Seed 42: fix for reproducibility; change for independent runs.
# mom yes: remove net linear momentum; rot yes: remove net angular momentum.

# --- Energy minimization before MD (always recommended) ---
# Removes overlaps or large forces from initial structure.
fix             relax all box/relax iso 0.0 vmax 0.001
min_style       cg
minimize        1e-6 1e-8 10000 100000
unfix           relax
# After minimization, reset velocities — minimization destroys kinetic energy.
velocity        all create 300.0 42 mom yes rot yes dist gaussian

# --- Equilibration: NVT with Nosé-Hoover thermostat ---
fix             equil all nvt temp 300.0 300.0 0.1
# Tdamp = 0.1 ps = 100 fs in metal units. Rule of thumb: ~100x timestep.
# Too small Tdamp: unphysical temperature oscillations (overdamped).
# Too large Tdamp: slow equilibration; canonical ensemble poorly maintained.

thermo          100             # Print thermodynamics every 100 steps
thermo_style    custom step temp press vol etotal ke pe lx ly lz
thermo_modify   norm no flush yes

timestep        0.002           # 2 fs in metal units (ps); safe for W with EAM
                                # Metals with EAM: 1–2 fs typical
                                # Covalent (Si Tersoff): 0.5–1 fs
                                # Molecular (CHARMM/AMBER): 0.5–1 fs (0.5 with H)
                                # ReaxFF: 0.1–0.5 fs [EXPERT REVIEW NEEDED]

# Write trajectory during equilibration for monitoring
dump            equil_dump all custom 500 equil.dump id type x y z
dump_modify     equil_dump sort id

run             50000           # 50,000 × 2 fs = 100 ps equilibration
unfix           equil
undump          equil_dump

# --- Check equilibration before production ---
# Inspect: T, P, E all stable; no drift in etotal; RDF matches expected structure.
# If etotal drifts: timestep too large or force field error.
# If T oscillates wildly: Tdamp too small.

# --- Production: NVT ---
fix             prod all nvt temp 300.0 300.0 0.1
reset_timestep  0               # Reset step counter for clean production output

dump            prod_dump all custom 1000 prod.dump id type x y z vx vy vz
dump_modify     prod_dump sort id

# Per-atom potential energy and stress for analysis
compute         pe_atom all pe/atom
compute         stress_atom all stress/atom NULL  # NULL = include all contributions

thermo          500
thermo_style    custom step temp press vol etotal ke pe
run             500000          # 500,000 × 2 fs = 1 ns production
unfix           prod

write_data      final_structure.lammps
write_restart   final.restart
```

### Workflow 2: Data file structure and creation

The LAMMPS data file is the primary input format for non-trivial structures. Format errors are a leading cause of silent simulation failures.

```
# W_bcc.lammps — LAMMPS data file for a 5x5x5 BCC W supercell
# Header section (must appear before any section keywords)
LAMMPS data file via write_data

250 atoms          # Total number of atoms
1 atom types       # Number of distinct atom types

0.000000 15.825000 xlo xhi    # Box bounds in Å (5 × 3.165)
0.000000 15.825000 ylo yhi
0.000000 15.825000 zlo zhi
# For triclinic boxes:
# xlo xhi ylo yhi zlo zhi xy xz yz

Masses

1 183.84           # Type 1 = W, mass in g/mol (required for metal units)

Atoms  # atomic

# atom-ID  atom-type  x(Å)  y(Å)  z(Å)
1  1  0.000000  0.000000  0.000000
2  1  1.582500  1.582500  1.582500
# ... (remaining atoms)
```

Generating data files with ASE:

```python
from ase.build import bulk
from ase.io import write
from pathlib import Path

# Create a 5x5x5 BCC W supercell
W = bulk("W", "bcc", a=3.165) * (5, 5, 5)

# Write LAMMPS data file (ASE handles the format automatically)
write("W_bcc.lammps", W, format="lammps-data", units="metal", atom_style="atomic")

# For molecular systems (CHARMM/AMBER-style):
# atom_style="full" adds atom-ID, mol-ID, type, charge, x, y, z
# Topology sections (Bonds, Angles, Dihedrals) must be present and consistent.

# Verify the written file:
from ase.io import read
W_check = read("W_bcc.lammps", format="lammps-data", units="metal", style="atomic")
print(f"Atoms: {len(W_check)}, Cell: {W_check.cell.lengths()}")
```

For molecular systems, the data file must include topology sections:

```
# Molecular data file structure (atom_style full):

N atoms
M bonds
P angles
Q dihedrals

S atom types
T bond types
U angle types
V dihedral types

# Box bounds ...

Masses
# type  mass

Pair Coeffs
# type  epsilon  sigma   (if written inline; alternatively use pair_coeff in script)

Bond Coeffs
# type  K  r0

Atoms  # full
# atom-ID  mol-ID  atom-type  charge  x  y  z

Bonds
# bond-ID  bond-type  atom-I  atom-J

Angles
# angle-ID  angle-type  atom-I  atom-J  atom-K
```

### Workflow 3: NPT simulation and barostat selection

Variable pressure (NPT) simulations require careful barostat setup. The choice of coupling (isotropic vs. anisotropic) must match the system symmetry.

```lammps
# NPT for a cubic metal: isotropic pressure coupling
fix   npt_run all npt &
      temp 300.0 300.0 0.1 &
      iso 0.0 0.0 1.0
# Pdamp = 1.0 ps in metal units. Rule of thumb: ~10x Tdamp. [EXPERT REVIEW NEEDED]
# iso: couple all three box dimensions equally — correct for cubic systems.
# Do NOT use iso for hexagonal, tetragonal, or orthorhombic systems under uniaxial stress.

# NPT with anisotropic coupling (e.g., for orthorhombic cell or uniaxial compression):
fix   npt_aniso all npt &
      temp 300.0 300.0 0.1 &
      aniso 0.0 0.0 1.0
# aniso: each box dimension relaxes independently; off-diagonal shear strains fixed at zero.

# NPT with full triclinic coupling (allows shear):
fix   npt_tri all npt &
      temp 300.0 300.0 0.1 &
      tri 0.0 0.0 1.0
# tri: allows all 6 box parameters to relax; correct for low-symmetry systems.
# Required for: martensitic transformations, shear-induced phase transitions,
# anisotropic elastic constant calculations.

# Thermostat choices:
# Nosé-Hoover (nvt/npt): canonical ensemble; good for production. Requires Tdamp tuning.
# Langevin (fix langevin + fix nve): stochastic; good for equilibration and
#   systems where energy injection (e.g., from bond reactions) must be dissipated.
#   Not the canonical ensemble unless combined with correct algorithm. [EXPERT REVIEW NEEDED]
# Berendsen (fix temp/berendsen): fast equilibration only; does NOT generate
#   the canonical ensemble; do not use for production sampling.

# Barostat choices:
# Nosé-Hoover (via npt): generates NPT ensemble correctly for most applications.
# Berendsen (fix press/berendsen): fast pressure relaxation; not the correct NPT ensemble.
#   Acceptable for initial equilibration, not for production thermodynamics.
```

### Workflow 4: Equilibration protocol

A systematic equilibration protocol reduces the risk of starting production from an unequilibrated state.

```lammps
# Step 1: Minimize at T=0 to remove overlaps
fix             min all box/relax iso 0.0
minimize        1e-8 1e-10 50000 500000
unfix           min

# Step 2: Heat from 0 K to target T in NVT with ramp
velocity        all create 1.0 12345 mom yes rot yes
fix             heat all nvt temp 1.0 ${TARGET_T} 0.1
run             20000           # 40 ps ramp in metal units
unfix           heat

# Step 3: NVT equilibration at target T
fix             equil_nvt all nvt temp ${TARGET_T} ${TARGET_T} 0.1
run             50000           # 100 ps NVT
unfix           equil_nvt

# Step 4: Switch to NPT for pressure equilibration
fix             equil_npt all npt temp ${TARGET_T} ${TARGET_T} 0.1 iso 0.0 0.0 1.0
run             100000          # 200 ps NPT
unfix           equil_npt

# Step 5: Check equilibration quality before production
# - Temperature: mean within 1% of target; std within 5% [EXPERT REVIEW NEEDED]
# - Pressure: mean within 100 bar of target for condensed matter
# - Energy: no long-time drift; running average stabilized
# - RDF/structure factor: matches known structure or previous literature
# - MSD: linear in t for liquids (confirms transport); plateau for solids
```

Python post-processing for equilibration diagnostics:

```python
import numpy as np
from pathlib import Path

def parse_thermo_log(log_file: str, keywords: list[str]) -> dict[str, np.ndarray]:
    """Parse LAMMPS log file thermo output into arrays."""
    data = {k: [] for k in keywords}
    in_run = False
    header = []
    with open(log_file) as f:
        for line in f:
            if line.startswith("Step"):
                header = line.split()
                in_run = True
                continue
            if in_run:
                parts = line.split()
                if len(parts) == len(header):
                    try:
                        vals = list(map(float, parts))
                        for k in keywords:
                            if k in header:
                                data[k].append(vals[header.index(k)])
                    except ValueError:
                        in_run = False
    return {k: np.array(v) for k, v in data.items()}

log = parse_thermo_log("log.lammps", ["Step", "Temp", "Press", "Etotal"])

# Check temperature stability
T = log["Temp"]
print(f"T mean: {T.mean():.1f} K, std: {T.std():.1f} K, CV: {T.std()/T.mean()*100:.1f}%")

# Check for energy drift
E = log["Etotal"]
# Linear fit to detect drift
steps = log["Step"]
slope, _ = np.polyfit(steps, E, 1)
print(f"Energy drift: {slope * 1000:.4f} meV/step — should be < 0.001 meV/step for stable NVE")
```

### Workflow 5: Force-field and pair-style selection

```lammps
# --- EAM / EAM alloy (metals) ---
pair_style      eam/alloy
pair_coeff      * * Cu_Ag_Zhou04.eam.alloy Cu Ag
# Element list must match atom type order in data file.
# Common sources: NIST IPRP, Interatomic Potentials Repository
# EAM is valid for: FCC/BCC metals near equilibrium, point defects, grain boundaries.
# EAM fails for: covalent/ionic bonding, surfaces with strong charge redistribution.

# --- MEAM (Modified EAM, for alloys with angular dependence) ---
pair_style      meam
pair_coeff      * * library.meam Fe Cu Cu_Fe.meam Fe Cu
# MEAM captures more directionality than EAM; validated for Fe-Cu, Ni-Al systems.

# --- Tersoff (covalent semiconductors: Si, Ge, C, SiC, GaN) ---
pair_style      tersoff
pair_coeff      * * SiC.tersoff Si C
# Tersoff parameterized for covalent bonds. Fails for:
# - metallic Si under high pressure
# - surfaces with reconstruction outside training set
# - amorphous structures far from crystal

# --- Stillinger-Weber (Si, Ge, water models) ---
pair_style      sw
pair_coeff      * * Si.sw Si

# --- ReaxFF (reactive MD: combustion, oxidation, fracture, catalysis) ---
pair_style      reaxff NULL
pair_coeff      * * ffield.reax C H O N
# NULL: no charge equilibration outside QEq (handled by fix qeq/reaxff)
fix             qeq all qeq/reaxff 1 0.0 10.0 1e-6 reaxff
# ReaxFF timestep must be 0.1–0.5 fs. [EXPERT REVIEW NEEDED]
# ReaxFF is very sensitive to the ffield.reax parameter file version.
# NEVER mix parameter files from different publications.

# --- Lennard-Jones (noble gases, simple fluids, colloidal particles) ---
pair_style      lj/cut 10.0            # Cutoff in Å (or sigma for lj units)
pair_coeff      1 1 0.0104 3.405       # epsilon(eV), sigma(Å) for Ar
pair_modify     tail yes               # Long-range tail correction for energy/pressure

# --- CHARMM-style molecular FF (proteins, lipids, polymers) ---
pair_style      lj/charmm/coul/long 10.0 12.0   # Inner and outer cutoffs
kspace_style    pppm 1.0e-4            # Long-range electrostatics via PPPM
pair_coeff      * * ...                # From CHARMM force field parameter file
# atom_style full required; data file must contain Bonds, Angles, Dihedrals sections.
bond_style      harmonic
angle_style     charmm
dihedral_style  charmm

# --- ML Interatomic Potentials (MACE, NequIP, DeePMD) ---
# MACE (via pair_style mace, requires LAMMPS-MACE interface)
pair_style      mace no_domain_decomposition
pair_coeff      * * MACE_model.model W
# no_domain_decomposition: required for equivariant message-passing;
# limits parallel scaling but is mandatory for correctness.

# NequIP
pair_style      nequip
pair_coeff      * * deployed_model.pth W

# DeePMD-kit
pair_style      deepmd frozen_model.pb
pair_coeff      * *
```

### Workflow 6: Restart and reproducible continuation

```lammps
# Writing restart files during a run:
restart         100000 run.restart.1 run.restart.2
# Alternates between two files to protect against corruption during write.
# Binary format; not human-readable.

# Continuing from a restart:
read_restart    run.restart.1
# After read_restart:
# - Box, positions, velocities, atom types, and masses are restored.
# - Fixes, computes, dumps must be RE-DEFINED — they are NOT stored in binary restart.
# - pair_style and pair_coeff must be re-specified.

# Minimal continuation script:
read_restart    run.restart.1
pair_style      eam/alloy
pair_coeff      * * W.eam.alloy W
fix             cont all nvt temp 300.0 300.0 0.1
thermo          500
thermo_style    custom step temp press vol etotal
dump            cont_dump all custom 1000 continued.dump id type x y z
timestep        0.002
run             500000

# For reproducibility, record the RNG seed used for velocity initialization.
# velocity create uses a fixed seed; restart does not reset this seed.
# To produce byte-for-byte identical trajectories: use the same seed AND
# the same number of MPI ranks AND the same domain decomposition.
# Changing nproc changes the floating-point evaluation order and produces
# numerically different (but statistically equivalent) trajectories.
```

## Common Failure Modes

| Failure | Symptom | Likely Cause | Diagnostic | Fix |
|---------|---------|--------------|------------|-----|
| Wrong `units` for force field | Energy explosion within first few steps; kinetic energy diverges; unrealistic temperature | `units real` used with a potential parameterized for `units metal` (or vice versa); timestep, cutoffs, or masses in wrong unit system | Check: EAM/MLP potentials expect `units metal`; CHARMM/AMBER expect `units real`. Compare total energy per atom vs. expected cohesive energy | Match `units` to the force field documentation; verify cohesive energy is physically reasonable before MD |
| Tdamp / Pdamp in wrong units | Temperature oscillates wildly (Tdamp too small) or thermostat has no effect (Tdamp too large) | `Tdamp` given in fs instead of ps for `metal` units (factor of 1000 error) | Check: in `metal` units, Tdamp=0.1 means 100 fs, Tdamp=100 means 100 ps. Typical good value: 0.1. If T fluctuates ±50% of target, Tdamp is too small | In `metal` units: Tdamp ~ 0.05–0.2 ps (50–200 fs); in `real` units: Tdamp ~ 50–200 fs |
| Atom type / element mismatch in pair_coeff | Wrong forces or crash with `ERROR: Invalid atom type in pair_coeff`; silently incorrect energies if species swapped | Element list in `pair_coeff * * potential.eam Element1 Element2` does not match atom type ordering in the data file | Print `mass` output at start of log; verify atom type 1 → Element1, type 2 → Element2, etc. | Reorder elements in `pair_coeff` to match data file type ordering; regenerate data file if needed |
| Missing `fix qeq` for ReaxFF | `ERROR: Fix qeq/reaxff required` or charge not equilibrated; wildly wrong energies | `pair_style reaxff` requires a charge equilibration fix (`fix qeq/reaxff`) that is separate from the pair_style command | Check if `fix qeq/reaxff` is in the input script | Add `fix qeq_fix all qeq/reaxff 1 0.0 10.0 1e-6 reaxff` immediately after `pair_coeff` for ReaxFF |
| Timestep too large | `ERROR: Lost atoms` or energy explosion; kinetic energy grows exponentially; bonds stretch unphysically | dt exceeds the stability limit of the Velocity Verlet integrator for the stiffest mode in the system | Print max force every step (`thermo_style custom ... fmax`); if forces > 100 eV/Å, reduce timestep or re-minimize | Reduce timestep by 2–5×; re-minimize structure before MD; check for overlapping atoms in input |
| Neighbor list skin too small | Periodic energy discontinuities; `WARNING: N atoms missing in fix` or neighbor list overflow | Atoms displaced more than skin/2 between neighbor list builds; common in high-T MD or fast-moving systems | Check `neigh_modify` settings; add `neigh_modify check yes`; monitor `neighs/atom` in thermo | Increase neighbor skin (`neighbor 3.0 bin`); reduce `neigh_modify delay` |
| Restart file incompatibility | `ERROR: Cannot open restart file` or wrong number of atoms after restart | Restart written by a different LAMMPS version or different number of atom types | Check LAMMPS version match; verify `atom_style` matches the original run | Use `write_data` (ASCII, more portable) for long-term archival; always re-specify `pair_style`, fixes, and dumps after `read_restart` |
| MLP `no_domain_decomposition` omitted | Wrong forces for equivariant MLPs (MACE, NequIP); silently incorrect energies in parallel | Equivariant message-passing potentials require atoms within the cutoff to be on the same MPI rank; domain decomposition splits this | Check MACE/NequIP documentation for required LAMMPS flags; compare single-core vs. multi-core energy | Add `no_domain_decomposition` to `pair_style mace` or `pair_style nequip`; this limits parallel scaling |

## Best Practices

- **Always check the cohesive energy per atom after reading the data file and before any dynamics.** Run `run 0` with `thermo_style custom etotal pe ke temp press` and compare the potential energy per atom to the known value (e.g., W cohesive energy ≈ −8.9 eV/atom with EAM). A wrong value by more than 10% indicates a unit error, type mismatch, or wrong potential file.
- **Minimize before MD** whenever the input structure is not from a prior MD run (e.g., from DFT relaxation, a crystal builder, or a database). Even small overlaps produce enormous forces that destabilize the integrator.
- **Set `reset_timestep 0` at the start of production** to make step-based output counts independent of the equilibration length.
- **Use two alternating restart files** (`restart N file.1 file.2`) to protect against corruption during a write. A single restart file written during a crash may be unreadable.
- **Record the LAMMPS version, pair potential file, and input script in a run directory README.** LAMMPS potential files have been updated between releases with the same filename; results are not reproducible without the exact file.
- **For parallel runs, verify that the number of MPI ranks does not exceed the number of atoms divided by ~10.** Under-populated domains produce load imbalance and can crash with `Lost atoms`.
- **For ML potentials with `no_domain_decomposition`,** scaling is limited to the number of atoms per GPU or to a single node. Profile before requesting many nodes.
- **Check structural integrity during and after a run** with OVITO or MDAnalysis: radial distribution function (RDF) should show expected peaks; common neighbor analysis (CNA) should confirm the crystal structure is preserved (for solid-state runs). Amorphization, phase transformation, or surface reconstruction may indicate a physical result or a force-field failure — examine both.
- **Use `compute msd all msd com yes` and plot MSD vs. time** for liquids and diffusion studies. For solids, MSD should plateau (mean ≈ 3kT/mω²); a continuously growing MSD in a solid indicates melting or a timestep instability.
- **For thermodynamic integration or free energy calculations, never use the Berendsen thermostat or barostat in production.** They do not generate the correct ensemble. Use Nosé-Hoover (nvt/npt) or Langevin + NVE.

## Integration with Other Skills

- **[md-equilibration](../md-equilibration/SKILL.md):** This skill covers equilibration at the level of input-script mechanics. That skill goes deeper on detecting equilibration, choosing equilibration timescales for different systems, and diagnosing slow relaxation modes — it is the natural follow-on for production-ready simulation setup.
- **[md-force-fields](../md-force-fields/SKILL.md):** Force-field selection and validation is treated briefly here. That skill covers the physical basis, parameterization, transferability limits, and validation protocols for EAM, MEAM, Tersoff, ReaxFF, CHARMM/AMBER, and coarse-grained force fields in depth.
- **[md-analysis](../md-analysis/SKILL.md):** Post-processing LAMMPS trajectories with MDAnalysis, OVITO, and Python (RDF, MSD, VDOS, stress analysis, CNA, Voronoi). The dump formats established here feed directly into that skill's analysis workflows.
- **[reactive-md](../reactive-md/SKILL.md):** ReaxFF and COMB potentials for bond-breaking/forming simulations. Covers parameter file selection, charge equilibration, reaction tracking, and validation against DFT reaction pathways.
- **[mlp-to-lammps](../../ml-interatomic-potentials/mlp-to-lammps/SKILL.md):** Deploying trained MACE, NequIP, ACE, or DeePMD models in LAMMPS. Covers compilation flags, `pair_style` invocation, parallelization constraints, and validation of MLP forces against DFT.
- **[active-learning-mlp](../../ml-interatomic-potentials/active-learning-mlp/SKILL.md):** In active learning loops, LAMMPS runs ML potentials to explore configuration space; structures with high uncertainty are queued for DFT single-point calculations. This skill's restart and trajectory-writing mechanics are used to extract candidate structures from LAMMPS trajectories.
- **[vasp-workflow](../../electronic-structure/vasp-workflow/SKILL.md):** The DFT-to-MD handoff: VASP computes energies and forces on structures that LAMMPS explores; LAMMPS uses the resulting ML potential for large-scale dynamics. The energy and force units must be reconciled when comparing VASP (eV, eV/Å) to LAMMPS `metal` units (identical convention) or `real` units (kcal/mol, kcal/mol/Å).

## Key References

1. Thompson, A. P., et al. (2022). *LAMMPS — A flexible simulation tool for particle-based materials modeling at the atomic, meso, and continuum scales.* Computer Physics Communications, 271, 108171. https://doi.org/10.1016/j.cpc.2021.108171
2. Plimpton, S. (1995). *Fast parallel algorithms for short-range molecular dynamics.* Journal of Computational Physics, 117(1), 1–19. https://doi.org/10.1006/jcph.1995.1039
3. Frenkel, D. & Smit, B. (2002). *Understanding Molecular Simulation: From Algorithms to Applications.* 2nd ed. Academic Press. (Chapters 4–6 cover integrators, thermostats, barostats.)
4. Bitzek, E., et al. (2006). *Structural relaxation made simple.* Physical Review Letters, 97(17), 170201. https://doi.org/10.1103/PhysRevLett.97.170201
5. Martínez, L., et al. (2009). *PACKMOL: A package for building initial configurations for molecular dynamics simulations.* Journal of Computational Chemistry, 30(13), 2157–2164. https://doi.org/10.1002/jcc.21224
