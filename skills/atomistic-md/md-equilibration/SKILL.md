---
name: md-equilibration
domain: atomistic-md
maturity: draft
tools:
  - LAMMPS
  - GROMACS
  - ASE
  - MDAnalysis
tested_versions:
  LAMMPS: "2Aug2023"
  GROMACS: "2024.2"
  ASE: "3.23.0"
  MDAnalysis: "2.7.0"
requires_gpu: false
estimated_runtime: minutes-to-hours
---

# MD Equilibration

## Description

This skill covers the systematic process of bringing an atomistic simulation from its initial (often artificial) configuration to a state representative of the target thermodynamic ensemble before collecting production statistics. It addresses ensemble selection, thermostat and barostat time constants, timestep stability, velocity initialization, and quantitative criteria for detecting whether temperature, pressure, density, energy, and structural observables have converged. Invoke this skill before starting any production MD run, whenever changing thermodynamic conditions, and whenever diagnosing anomalous simulation behavior that may originate from incomplete equilibration.

## Domain Context

An MD simulation starts from an initial configuration that is almost never a genuine sample from the target ensemble. Crystal structures from databases have zero-temperature geometry; liquid models begin from arbitrary packings; protein structures come from X-ray refinement at cryogenic conditions. The equilibration phase drives the system from this artificial starting point to a state where the time average of any observable is independent of when in the trajectory it is measured.

Three distinct processes happen during equilibration, each on its own timescale:

1. **Energy redistribution (thermalization):** Kinetic energy assigned to all degrees of freedom reaches the target temperature. Timescale: tens to hundreds of picoseconds for most systems.
2. **Pressure and density relaxation:** The simulation box adjusts to the target pressure. Timescale: typically 10–100× longer than thermalization for liquids; can be nanoseconds for dense polymers or glasses. [EXPERT REVIEW NEEDED]
3. **Structural relaxation:** Local and global order parameters (RDF peaks, dihedral distributions, bond orientational order) reach their equilibrium values. This is the slowest and most system-dependent step. For crystalline systems, it may complete in picoseconds. For viscous liquids, glasses, or polymer melts, it may require microseconds of simulation time.

The thermostat and barostat do not just maintain temperature and pressure — they also define the statistical ensemble being sampled. Using the wrong algorithm (e.g., Berendsen) produces correct-looking temperatures and pressures while sampling a distribution that is not the canonical ensemble, which invalidates statistical mechanics observables like heat capacity, compressibility, and free energy differences.

The **flying ice cube effect** is the most dangerous and least intuitive equilibration failure: a stochastic thermostat slowly transfers kinetic energy from internal (relative) motion into net translational momentum of the whole system. Atoms appear cold relative to the center of mass, while the center of mass drifts at high velocity. The total kinetic energy is correct but the internal temperature is wrong. This is caused by insufficient or incorrect momentum removal.

## When to Use This Skill

- Before every production MD run, regardless of system type or force field.
- When restarting a simulation after modifying the force field, temperature, pressure, or composition.
- When diagnosing anomalous behavior in a production run (energy drift, temperature drift, pressure oscillations, structural disorder).
- When setting up equilibration protocols for a new material class where timescales are unknown.
- When generating thermally distorted structures for ML potential training datasets — the distribution of sampled configurations depends on the quality of equilibration.

## When Not to Use This Skill

- **Geometry optimization (0 K energy minimization):** Minimization is not equilibration and should precede it, not replace it.
- **Path integral MD:** Nuclear quantum effects require specialized initialization (ring polymer normal modes) that goes beyond standard velocity assignment.
- **Non-equilibrium MD (shock, NEMD transport):** These methods explicitly break equilibrium; standard equilibration criteria do not apply.
- **Enhanced sampling methods (metadynamics, replica exchange):** These have their own initialization protocols that differ from plain MD equilibration.

## Core Concepts

- **NVE (microcanonical):** No thermostat or barostat. Total energy is conserved. Useful as a diagnostic: a well-equilibrated system in NVE should show zero long-term energy drift. Not typically used for production of condensed-phase thermodynamics.
- **NVT (canonical):** Fixed N, V, T. Temperature is maintained by a thermostat. The correct ensemble for systems where volume fluctuations are irrelevant or where comparing to fixed-cell DFT.
- **NPT (isothermal-isobaric):** Fixed N, P, T. Both temperature and pressure are maintained. Correct for liquid-state thermodynamics, density calculations, and most experimental conditions. Requires both a thermostat and a barostat.
- **NPH (isoenthalpic-isobaric):** Fixed N, P, H. Rarely used; relevant for shock and adiabatic compression studies.
- **Nosé-Hoover thermostat:** Deterministic, time-reversible, conserves an extended Hamiltonian. Generates the canonical ensemble for ergodic systems. Requires a chain of oscillators (NHCH) for non-ergodic systems. Sensitive to the time constant `Tdamp` — too small causes temperature oscillations; too large gives slow thermalization. Standard choice for production NVT.
- **Langevin thermostat:** Stochastic friction + random noise. Canonical ensemble if implemented correctly. Faster thermalization than Nosé-Hoover but introduces artificial dynamics that may not be physical. Good for equilibration; use with care for dynamical properties (diffusion, viscosity). Characterized by damping coefficient γ (ps⁻¹) or its inverse, the friction time.
- **Stochastic velocity rescaling (SVR / V-rescale, Bussi-Donadio-Parrinello 2007):** Stochastic thermostat that rescales velocities while preserving the correct kinetic energy distribution. Generates the canonical ensemble; correct temperature fluctuations; no flying ice cube effect. Recommended for GROMACS production runs.
- **Berendsen thermostat:** Exponential velocity rescaling toward target T. Fast equilibration, but does NOT generate the canonical ensemble — temperature fluctuations are suppressed, violating the equipartition theorem. Heat capacity and other fluctuation-based properties computed under Berendsen will be wrong. **Use only for initial equilibration, never for production statistics.**
- **Berendsen barostat:** Analogous to the thermostat: fast pressure relaxation but incorrect ensemble. Compressibility and pressure fluctuations are wrong. **Use only for density pre-equilibration, never for free energy or compressibility calculations.**
- **Parrinello-Rahman barostat (GROMACS) / MTTK (LAMMPS npt):** Correct NPT ensemble. Should be used after Berendsen pre-equilibration, not for cold starts — it can oscillate dangerously if the initial pressure is far from the target.
- **Time constant (`Tdamp`, `tau_t`, `tau_p`):** The relaxation time of the thermostat or barostat. In LAMMPS `metal` units, `Tdamp` is in ps; in GROMACS, `tau_t` is in ps. Rule of thumb: `Tdamp ≈ 100× timestep` for Nosé-Hoover; `tau_p ≈ 10× tau_t` for Parrinello-Rahman. [EXPERT REVIEW NEEDED for system-class-specific values]
- **Maxwell-Boltzmann distribution:** At thermal equilibrium, atomic velocities follow the Maxwell-Boltzmann distribution. Checking that the velocity distribution has converged to MB is a sensitive test of thermalization. Deviations indicate the flying ice cube effect, an energy source (bond reactions, incorrect force field), or insufficient sampling.
- **Energy drift in NVE:** The standard quantitative test for timestep correctness and force field self-consistency. A well-behaved MD integrator conserves total energy to within a small fraction (< 0.01% over 1 ns is typical for Velocity Verlet at a safe timestep). [EXPERT REVIEW NEEDED]

## Key Workflows

### Workflow 1: Standard multi-stage equilibration protocol (LAMMPS)

A systematic four-stage protocol applicable to most solid and liquid systems. Each stage has explicit convergence criteria before proceeding.

```lammps
# ============================================================
# Four-stage equilibration protocol (LAMMPS, metal units)
# System: generic condensed-phase material
# ============================================================

variable  TARGET_T    equal  300.0     # K
variable  TARGET_P    equal  0.0       # bar (0 = ambient for condensed matter)
variable  DT          equal  0.002     # ps = 2 fs; reduce for stiff bonds or ReaxFF
variable  TDAMP       equal  0.1       # ps; ~100x dt for Nose-Hoover [EXPERT REVIEW NEEDED]
variable  PDAMP       equal  1.0       # ps; ~1000x dt for barostat [EXPERT REVIEW NEEDED]

timestep  ${DT}

thermo         100
thermo_style   custom step temp press vol etotal ke pe lx ly lz
thermo_modify  flush yes norm no

# -------------------------------------------------------
# Stage 1: Energy minimization
# Goal: remove overlaps and large forces before any dynamics.
# Criterion: max force < forc_conv (set per application).
# -------------------------------------------------------
print "=== Stage 1: Minimization ==="
min_style      cg
minimize       1e-8 1e-10 100000 1000000
# After minimization, ALL kinetic energy is gone. Must reinitialize velocities.

# -------------------------------------------------------
# Stage 2: NVT heating from 1 K to TARGET_T
# Goal: introduce kinetic energy smoothly, avoiding force spikes.
# Use velocity rescaling (Berendsen) here — fast, safe for heating.
# -------------------------------------------------------
print "=== Stage 2: NVT heating ==="
velocity       all create 1.0 87241 mom yes rot yes dist gaussian
# mom yes: zero net linear momentum at initialization
# rot yes: zero net angular momentum (important for non-periodic clusters)

fix            f_heat all nvt temp 1.0 ${TARGET_T} ${TDAMP}
run            20000     # 40 ps ramp; increase for stiff or large systems
unfix          f_heat

# -------------------------------------------------------
# Stage 3: NVT equilibration at target temperature
# Goal: stabilize temperature before coupling barostat.
# Starting NPT from a cold, high-pressure crystal can cause
# Parrinello-Rahman oscillations that destabilize the run.
# -------------------------------------------------------
print "=== Stage 3: NVT equilibration ==="
fix            f_nvt all nvt temp ${TARGET_T} ${TARGET_T} ${TDAMP}
run            50000     # 100 ps; check temperature convergence in log
unfix          f_nvt

# Convergence check: temperature mean should be within 2% of TARGET_T.
# If not, extend this stage or reduce TDAMP.

# -------------------------------------------------------
# Stage 4: NPT equilibration — density/pressure relaxation
# Goal: allow box to relax to equilibrium density at TARGET_P.
# -------------------------------------------------------
print "=== Stage 4: NPT equilibration ==="
fix            f_npt all npt &
               temp ${TARGET_T} ${TARGET_T} ${TDAMP} &
               iso ${TARGET_P} ${TARGET_P} ${PDAMP}
# For non-cubic systems: use 'aniso' or 'tri' coupling (see lammps-workflow skill).

run            200000    # 400 ps; liquids may need 1–2 ns
unfix          f_npt

# Write equilibrated structure for production:
write_data     equilibrated.lammps
write_restart  equilibrated.restart

# -------------------------------------------------------
# Transition to production: reset step counter, new dumps.
# NEVER include equilibration frames in production averages.
# -------------------------------------------------------
reset_timestep 0
```

### Workflow 2: Quantitative equilibration diagnostics

Never rely on visual inspection of the log alone. Use quantitative criteria on multiple observables.

```python
import numpy as np
from pathlib import Path
import warnings


def parse_lammps_log(log_file: str) -> dict[str, np.ndarray]:
    """Parse all thermo columns from a LAMMPS log file into numpy arrays."""
    data: dict[str, list] = {}
    header: list[str] = []
    in_run = False
    with open(log_file) as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith("Step"):
                header = stripped.split()
                for k in header:
                    if k not in data:
                        data[k] = []
                in_run = True
                continue
            if in_run:
                parts = stripped.split()
                if len(parts) == len(header):
                    try:
                        vals = list(map(float, parts))
                        for k, v in zip(header, vals):
                            data[k].append(v)
                    except ValueError:
                        in_run = False
    return {k: np.array(v) for k, v in data.items() if v}


def check_temperature(T: np.ndarray, target: float,
                      tol_mean: float = 0.02, tol_std_ratio: float = 0.10) -> dict:
    """
    Check temperature equilibration.
    tol_mean: fractional tolerance on mean (0.02 = 2%)
    tol_std_ratio: std/mean tolerance (0.10 = 10%)
    """
    # Use the last 50% of the trajectory to exclude transient
    tail = T[len(T) // 2:]
    mean, std = tail.mean(), tail.std()
    rel_err = abs(mean - target) / target
    cv = std / mean
    return {
        "mean": mean, "std": std,
        "relative_error": rel_err,
        "cv": cv,
        "converged": rel_err < tol_mean and cv < tol_std_ratio,
        "note": f"T = {mean:.1f} ± {std:.1f} K (target {target:.1f} K)",
    }


def check_energy_drift(E: np.ndarray, steps: np.ndarray,
                       max_drift_meV_per_atom_per_ns: float = 1.0,
                       natoms: int = 1,
                       dt_ps: float = 0.002) -> dict:
    """
    Check for linear energy drift (NVE stability test or production consistency).
    max_drift_meV_per_atom_per_ns: threshold in meV/atom/ns. [EXPERT REVIEW NEEDED]
    """
    tail_idx = len(E) // 2
    E_tail = E[tail_idx:]
    s_tail = steps[tail_idx:]
    slope_eV_per_step, _ = np.polyfit(s_tail, E_tail, 1)
    time_ns = (s_tail[-1] - s_tail[0]) * dt_ps / 1000.0
    drift_meV_atom_ns = abs(slope_eV_per_step) * (s_tail[-1] - s_tail[0]) / natoms * 1000 / time_ns
    return {
        "slope_eV_per_step": slope_eV_per_step,
        "drift_meV_atom_ns": drift_meV_atom_ns,
        "converged": drift_meV_atom_ns < max_drift_meV_per_atom_per_ns,
        "note": f"Energy drift: {drift_meV_atom_ns:.3f} meV/atom/ns",
    }


def check_pressure(P: np.ndarray, target: float,
                   tol_bar: float = 500.0) -> dict:
    """
    Pressure convergence is much looser than temperature — large fluctuations
    are physical in NPT (especially for small systems).
    tol_bar: tolerance on the mean pressure deviation.
    """
    tail = P[len(P) // 2:]
    mean, std = tail.mean(), tail.std()
    # std(P) ~ NkT/V * compressibility — can be thousands of bar for small boxes
    return {
        "mean": mean, "std": std,
        "converged": abs(mean - target) < tol_bar,
        "note": f"P = {mean:.0f} ± {std:.0f} bar (target {target:.0f} bar); "
                f"large std is physical for small NPT boxes",
    }


def detect_flying_ice_cube(vx: np.ndarray, vy: np.ndarray, vz: np.ndarray,
                            masses: np.ndarray) -> dict:
    """
    Flying ice cube: net CoM momentum builds up while internal T drops.
    Requires per-atom velocities from a dump file.
    vx, vy, vz: (N,) arrays; masses: (N,) array.
    """
    total_mass = masses.sum()
    vcm = np.array([
        (masses * vx).sum() / total_mass,
        (masses * vy).sum() / total_mass,
        (masses * vz).sum() / total_mass,
    ])
    # Internal velocities (relative to CoM)
    vx_int = vx - vcm[0]
    vy_int = vy - vcm[1]
    vz_int = vz - vcm[2]
    # T_internal from equipartition
    kb_eV = 8.617e-5   # eV/K
    ke_int = 0.5 * (masses * (vx_int**2 + vy_int**2 + vz_int**2)).sum()
    T_internal = 2 * ke_int / (3 * len(masses) * kb_eV)
    v_cm_magnitude = np.linalg.norm(vcm)
    return {
        "v_cm": v_cm_magnitude,
        "T_internal": T_internal,
        "flying_ice_cube_suspected": v_cm_magnitude > 0.01,  # Å/ps threshold [EXPERT REVIEW NEEDED]
        "note": f"|v_CoM| = {v_cm_magnitude:.4f} Å/ps, T_internal = {T_internal:.1f} K",
    }


# Usage:
log = parse_lammps_log("log.lammps")
natoms = 250   # replace with actual atom count
dt = 0.002     # ps

t_check = check_temperature(log["Temp"], target=300.0)
e_check = check_energy_drift(log["Etotal"], log["Step"], natoms=natoms, dt_ps=dt)
p_check = check_pressure(log["Press"], target=0.0)

print(t_check["note"], "—", "OK" if t_check["converged"] else "NOT converged")
print(e_check["note"], "—", "OK" if e_check["converged"] else "DRIFT DETECTED")
print(p_check["note"], "—", "OK" if p_check["converged"] else "NOT converged")
```

### Workflow 3: GROMACS equilibration protocol (NVT → NPT → production)

GROMACS uses separate `.mdp` (molecular dynamics parameters) files for each stage. The handoff between stages uses the `-t` flag to pass the checkpoint (`.cpt`) file, preserving velocities.

```ini
; nvt.mdp — NVT equilibration for a molecular/biomolecular system
; GROMACS 2024+ syntax

integrator    = md             ; Leap-frog Verlet integrator
dt            = 0.002          ; ps; 2 fs for bonded systems with constraints
nsteps        = 50000          ; 50,000 × 2 fs = 100 ps

nstxout       = 0              ; Do not write coordinates to .trr during equil
nstvout       = 0
nstfout       = 0
nstenergy     = 500            ; Write energies to .edr every 500 steps
nstlog        = 500            ; Write to .log every 500 steps
nstxout-compressed = 1000      ; Write .xtc trajectory every 1000 steps

; Constraints — required for 2 fs timestep with H atoms
constraints         = h-bonds      ; Constrain X-H bonds (LINCS)
constraint-algorithm = lincs
lincs-iter          = 1
lincs-order         = 4

; Neighbor list
cutoff-scheme = Verlet
nstlist       = 10              ; Rebuild neighbor list every 10 steps
ns-type       = grid
rcoulomb      = 1.2             ; nm
rvdw          = 1.2             ; nm

; Electrostatics
coulombtype   = PME             ; Particle Mesh Ewald for long-range electrostatics
pme-order     = 4
fourierspacing = 0.16           ; nm

; vdW
vdwtype       = cutoff
vdw-modifier  = force-switch
rvdw-switch   = 1.0             ; nm

; Temperature coupling
tcoupl        = V-rescale       ; Stochastic velocity rescaling (Bussi et al. 2007)
                                 ; Correct canonical ensemble; recommended over Berendsen
tc-grps       = Protein  Non-Protein
tau-t         = 0.1     0.1     ; ps; coupling time constant for each group
ref-t         = 300.0   300.0   ; K

; Pressure coupling — OFF during NVT
pcoupl        = no

; Velocity generation
gen-vel       = yes             ; Generate velocities from Maxwell-Boltzmann distribution
gen-temp      = 300.0
gen-seed      = 42              ; Fixed seed for reproducibility
```

```ini
; npt.mdp — NPT equilibration (run after NVT, using -t nvt.cpt)
; Only changes relative to nvt.mdp shown:

integrator    = md
dt            = 0.002
nsteps        = 100000          ; 200 ps

; Temperature coupling — same as NVT
tcoupl        = V-rescale
tc-grps       = Protein  Non-Protein
tau-t         = 0.1     0.1
ref-t         = 300.0   300.0

; Pressure coupling — Parrinello-Rahman for production-quality NPT
; NOTE: Do NOT use Parrinello-Rahman for cold-start NPT — it oscillates.
; Use Berendsen first (in a preliminary 100 ps run), then switch to P-R.
pcoupl        = Parrinello-Rahman
pcoupltype    = isotropic       ; isotropic for liquids; 'semiisotropic' for membranes
tau-p         = 2.0             ; ps; barostat time constant
ref-p         = 1.013           ; bar (1 atm)
compressibility = 4.5e-5        ; bar⁻¹; isothermal compressibility of water at 300 K
                                 ; Set to ~0 for solids to prevent cell deformation

; Velocity generation — OFF for NPT continuation (velocities come from -t flag)
gen-vel       = no
continuation  = yes
```

GROMACS stage execution:

```bash
#!/bin/bash
# GROMACS equilibration pipeline

# 0. Energy minimization
gmx grompp -f em.mdp -c system.gro -p topol.top -o em.tpr
gmx mdrun -v -deffnm em

# Check minimization converged:
# grep "Potential Energy" em.log | tail -1
# grep "Maximum force" em.log | tail -1

# 1. NVT equilibration
gmx grompp -f nvt.mdp -c em.gro -r em.gro -p topol.top -o nvt.tpr
gmx mdrun -deffnm nvt

# Check temperature equilibrated:
gmx energy -f nvt.edr -o nvt_temperature.xvg << EOF
Temperature
0
EOF
# Inspect nvt_temperature.xvg: should plateau at 300 K within first 20-30 ps.

# 2. NPT equilibration (continue from NVT velocities and positions)
gmx grompp -f npt.mdp -c nvt.gro -r nvt.gro -t nvt.cpt -p topol.top -o npt.tpr
gmx mdrun -deffnm npt

# Check density equilibrated:
gmx energy -f npt.edr -o npt_density.xvg << EOF
Density
0
EOF
# Density should plateau and match experimental value within ~1%. [EXPERT REVIEW NEEDED]

# 3. Verify before production:
gmx check -f npt.edr   # Summary statistics for all energy terms
```

### Workflow 4: System-class-specific equilibration considerations

Different system types require different timescales and precautions.

```python
# Equilibration timescale guide by system type:
#
# SOLID / CRYSTAL (metals, oxides, semiconductors)
# --------------------------------------------------
# Typical equilibration time: 50–200 ps (NVT) + 100–500 ps (NPT)
# Special considerations:
#   - Minimization is critical; lattice distortions from idealized geometry must relax.
#   - NPT coupling: use 'aniso' or 'tri' for non-cubic crystals; 'iso' for cubic.
#   - Volume convergence: monitor lattice parameters a, b, c individually.
#   - Thermal expansion is small; density converges quickly.
# Convergence criterion: a,b,c stable to < 0.1%; P mean within 100 bar of target.
#
# LIQUID (water, organic solvents, ionic liquids, liquid metals)
# ---------------------------------------------------------------
# Typical equilibration time: 200 ps (NVT) + 1–5 ns (NPT)
# Special considerations:
#   - Density equilibration is SLOW for viscous liquids — extend NPT.
#   - Start NPT with Berendsen barostat for initial density adjustment,
#     then switch to Parrinello-Rahman / Nose-Hoover for production.
#   - Ionic liquids and viscous organics may require 5–20 ns NPT. [EXPERT REVIEW NEEDED]
#   - Check RDF against experimental neutron/X-ray data where available.
# Convergence criterion: density within 0.5% of target; RDF peaks stable.
#
# INTERFACE / SLAB (solid-liquid, bilayer, surface)
# -------------------------------------------------
# Typical equilibration time: 1–10 ns
# Special considerations:
#   - Use 'semiisotropic' or 'aniso' pressure coupling — only the z-dimension
#     (surface normal) should relax; x-y locked to bulk lattice.
#   - For solid-liquid interfaces: equilibrate liquid layer separately first.
#   - Monitor interfacial density profile (mass density vs. z) for convergence.
#   - Flying ice cube effect more likely at interfaces — check CoM drift carefully.
#
# POLYMER / BIOMOLECULE
# ----------------------
# Typical equilibration time: 10–100 ns or longer for large systems
# Special considerations:
#   - Initial structure from experiment (X-ray, cryo-EM) may be far from force-field
#     equilibrium — use position restraints (GROMACS: posre.itp) during early NPT.
#   - Gradually release restraints: full restraints → backbone only → no restraints.
#   - Monitor RMSD from initial structure and Rg (radius of gyration).
#   - For proteins: backbone RMSD plateau < 2 Å from crystal structure usually
#     indicates equilibrium in the force field. [EXPERT REVIEW NEEDED]
#
# REACTIVE MD (ReaxFF, COMB)
# ---------------------------
# Typical equilibration time: same order as non-reactive, but with dt = 0.1–0.5 fs
# Special considerations:
#   - ReaxFF generates its own internal heat from bond formation/breaking.
#   - Langevin thermostat preferred over Nose-Hoover for reactive MD — it dissipates
#     local heat release more effectively.
#   - Monitor bond order distribution and species counts during equilibration.
#   - Never use NPT during reactive equilibration in rapidly reacting systems —
#     sudden density changes from reactions can crash the barostat.
#
# ML POTENTIAL MD
# ----------------
# Typical equilibration time: same as classical MD for the same system class.
# Special considerations:
#   - MLP forces at configurations far from the training set may be unphysical.
#   - Pre-equilibrate with a cheaper classical potential (EAM, LJ) to reach a
#     reasonable geometry, then switch to MLP.
#   - Monitor for force spikes (max force > 50 eV/Å is suspicious) during MLP equil.
#   - Energy drift in NVE is a sensitive test of MLP accuracy vs. DFT reference.
#   - For MACE/NequIP with no_domain_decomposition: equilibration is single-core or
#     single-node; plan for longer wall time.

# NVE stability test for timestep validation:
NVE_TEST_LAMMPS = """
# Run 10,000 steps NVE from equilibrated structure; check energy conservation
fix           nve_test all nve
thermo        100
thermo_style  custom step etotal ke pe temp
run           10000
unfix         nve_test
# Plot etotal vs step; slope should be < 1e-5 eV/step for a 2 fs timestep.
# If drift > 0.1% of total energy over 10,000 steps: reduce timestep.
"""
```

### Workflow 5: Detecting and fixing the flying ice cube effect

```python
# The flying ice cube: net CoM momentum accumulates → internal T drops.
# Mechanism: stochastic noise (Langevin, random velocity rescaling) adds
# momentum to each atom, and rounding errors in the summation do not exactly
# cancel over many steps. Without periodic momentum zeroing, the CoM drifts.

# LAMMPS fix to prevent the flying ice cube:
FIX_MOMENTUM = """
# Remove linear and angular momentum every 100 steps:
fix           zero_mom all momentum 100 linear 1 1 1 angular
# linear 1 1 1: remove x, y, z linear momentum
# angular: remove angular momentum (important for non-periodic systems or droplets)
# For fully periodic systems, angular momentum removal is optional but harmless.
"""

# Diagnostic: compare total T to internal T
# (requires per-atom velocities from a dump file)

import numpy as np
from MDAnalysis import Universe

def diagnose_flying_ice_cube(dump_file: str, topology: str = None) -> None:
    u = Universe(dump_file)
    for ts in u.trajectory[-10:]:   # Check last 10 frames
        masses = u.atoms.masses
        vels = u.atoms.velocities   # Å/ps
        total_mass = masses.sum()
        # Center-of-mass velocity
        vcm = (masses[:, None] * vels).sum(axis=0) / total_mass
        # Internal velocities
        vels_int = vels - vcm[None, :]
        # Internal kinetic energy (eV, using mass in g/mol, vel in Å/ps)
        # KE = 0.5 * m [g/mol] * v^2 [Å²/ps²] * (1/10363 [eV·ps²·mol/(g·Å²)])
        conv = 1.0 / 10363.0   # g·Å²/ps²/mol to eV
        ke_int = 0.5 * conv * (masses * (vels_int**2).sum(axis=1)).sum()
        ndof = 3 * len(masses) - 3   # subtract CoM dof
        T_internal = 2 * ke_int / (ndof * 8.617e-5)
        print(f"Frame {ts.frame}: |v_CoM|={np.linalg.norm(vcm):.4f} Å/ps, "
              f"T_internal={T_internal:.1f} K")

# Symptoms of flying ice cube:
# 1. Reported temperature normal, but system appears "frozen" structurally
# 2. MSD of all atoms grows linearly but equals center-of-mass drift
# 3. VDOS (vibrational DOS) shows a sharp peak at 0 frequency (translational mode)
# 4. Temperature drops systematically over long runs with Langevin thermostat
```

## Common Failure Modes

| Failure | Symptom | Likely Cause | Diagnostic | Fix |
|---------|---------|--------------|------------|-----|
| Flying ice cube | Reported T normal; system drifts as a rigid body; MSD grows without structural dynamics; velocity distribution has a sharp translational peak | Stochastic thermostat (Langevin, velocity rescaling) accumulates net CoM momentum over many steps | Compute |v_CoM| from per-atom velocities; should be < 0.01 Å/ps for typical solids | Add `fix momentum 100 linear 1 1 1` (LAMMPS) or set `comm-mode = Linear` (GROMACS); re-equilibrate |
| Production started too early | Observables (density, energy, RDF) show slow drift in production; averages depend on starting point; simulation not reproducible from different seeds | Equilibration truncated before structural or density convergence | Plot running average of density and energy vs. time; if they are still drifting at the end of equilibration, extend it | Double equilibration time; use block averaging to detect non-stationarity |
| Berendsen barostat in production | Density correct but pressure fluctuations too small; heat capacity, compressibility, free energy wrong | Berendsen does not generate the correct NPT ensemble; suppresses thermodynamic fluctuations | Compare pressure std to expected kT·κ/V value; if std(P) ≈ 0, Berendsen is active | Switch to Parrinello-Rahman (GROMACS) or Nosé-Hoover NPT (LAMMPS) after initial density equilibration |
| Parrinello-Rahman oscillations on cold start | Simulation crashes or box oscillates wildly within first 10–20 ps of NPT | Starting NPT from a cold, high-stress structure with Parrinello-Rahman — P-R assumes near-equilibrium | Check initial pressure (often >>10,000 bar for an unequilibrated solid); check box volume oscillations | Pre-equilibrate density with Berendsen for 200–500 ps; switch to Parrinello-Rahman after density stabilizes |
| Thermostat masking force-field errors | Simulation "runs" but structural properties (bond lengths, RDF) are wrong; atoms move more than physically expected | A strongly coupled thermostat (Berendsen, very small Tdamp/tau_t) injects or removes energy fast enough to compensate for wrong forces | Run a short NVE test from equilibrated structure; energy should be conserved. If energy explodes in NVE but is stable in NVT, the force field has errors | Diagnose force field: check pair_coeff units, element-type mapping, cutoff. Do not use thermostat coupling to paper over bad forces |
| Non-equilibrated density in liquid | Density slowly drifts during production; thermodynamic properties shift; trajectories non-stationary | NPT equilibration time too short for viscous liquid or dense polymer | Monitor running average of density; run autocorrelation of volume to estimate relaxation time | Extend NPT equilibration by 5–10× the density autocorrelation time; for polymers, may need 10–100 ns [EXPERT REVIEW NEEDED] |
| Velocity distribution not Maxwell-Boltzmann | T plateau at correct value but higher moments (kurtosis) wrong; VDOS shows unphysical features; equipartition violated | Constraints over-removing degrees of freedom; Nosé-Hoover chain too short for non-ergodic system; insufficient equilibration time | Compute velocity histogram and compare to MB; check with `compute temp/partial` for individual DOF contributions | Use SVR (V-rescale) thermostat for MD with many constraints; increase NHCH chain length; extend equilibration |
| Wrong ensemble for property of interest | Computed heat capacity, compressibility, or free energy does not match experiment; error does not decrease with run length | NVT used where NPT is required (volume fluctuations needed); or Berendsen used instead of canonical thermostat | Check the ensemble requirement of the property; verify thermostat/barostat type in input | Use the physically correct ensemble for the target property; see Core Concepts |

## Best Practices

- **Minimize before every MD run,** even for structures taken from a prior MD trajectory. Restart files preserve positions faithfully, but accumulated floating-point differences between runs can leave atoms in slightly unfavorable configurations.
- **Always zero linear (and, for non-periodic systems, angular) momentum after velocity initialization.** In LAMMPS: `velocity all create T seed mom yes rot yes`. In GROMACS: `comm-mode = Linear`. Re-zero every 100–1000 steps with `fix momentum` (LAMMPS) to prevent slow drift.
- **Run the NVE stability test before any production run.** Integrate 10,000–50,000 steps from the equilibrated structure with NVE. Total energy drift > 0.01% of the total over that period indicates a timestep that is too large, a problematic force field, or a bug.
- **Use block averages to test for stationarity.** Divide the equilibration trajectory into 4–8 blocks and compute the mean of T, P, density, and energy in each block. If block means trend monotonically, equilibration is incomplete.
- **Never use the Berendsen thermostat or barostat in production** for any property that depends on fluctuations. This includes heat capacity, isothermal compressibility, Kirkwood-Buff integrals, free energy perturbation, and any entropy-based quantity.
- **For interfaces and slabs, use decoupled pressure coupling.** Only the box dimension perpendicular to the interface should relax; in-plane dimensions should either be fixed or coupled to a separate reference pressure. GROMACS: `pcoupltype = semiisotropic`; LAMMPS: `fix npt ... z P_target P_target Pdamp`.
- **Increase equilibration time 5–10× for viscous or slowly-relaxing systems.** Ionic liquids, polymer melts, glasses, and proteins near phase transitions can take nanoseconds to microseconds to equilibrate density and structural order. Inadequate equilibration is the leading cause of irreproducible MD results in the literature. [EXPERT REVIEW NEEDED]
- **Document the equilibration protocol in every Methods section.** Specify the thermostat and barostat type and time constants, the total equilibration time per stage, and the convergence criteria used.
- **For ML potential MD, pre-equilibrate with a cheap classical potential** to bring the system into the training-set manifold before switching to the MLP. Starting MLP MD from a far-from-equilibrium structure risks force spikes from extrapolation.

## Integration with Other Skills

- **[lammps-workflow](../lammps-workflow/SKILL.md):** The input-script mechanics for equilibration (fix nvt, fix npt, velocity, restart) are covered there. This skill provides the physical criteria, timescale guidance, and diagnostic tools that determine when those mechanics have succeeded.
- **[gromacs-workflow](../gromacs-workflow/SKILL.md):** GROMACS-specific equilibration uses mdp files (nvt.mdp → npt.mdp), checkpoint files (`-t`), and `gmx energy` for diagnostics. This skill's protocols translate directly to GROMACS mdp syntax with the parameter names noted in Workflow 3.
- **[md-force-fields](../md-force-fields/SKILL.md):** Force field choice sets the equilibration timescale. Stiff bonds (CHARMM/AMBER) require dt = 0.5–1 fs; reactive potentials (ReaxFF) require dt = 0.1–0.5 fs. An apparently failed equilibration often traces back to a force field error detectable via the NVE test described here.
- **[md-analysis](../md-analysis/SKILL.md):** The quantitative equilibration diagnostics (RDF convergence, density autocorrelation, RMSD, velocity distribution) are computed using the analysis tools covered there. This skill specifies what to compute; that skill covers how.
- **[enhanced-sampling](../enhanced-sampling/SKILL.md):** When structural equilibration timescales exceed what plain MD can access (slow conformational transitions, nucleation, diffusion), enhanced sampling methods (metadynamics, replica exchange, parallel tempering) are needed. This skill's equilibration criteria apply to each replica or walker individually.
- **[reactive-md](../reactive-md/SKILL.md):** ReaxFF equilibration has the additional requirement that bond-order distributions and species inventories have stabilized, not just T, P, and density. This skill's generic diagnostics apply; that skill covers reactive-specific equilibration checks.
- **[mlp-validation](../../ml-interatomic-potentials/mlp-validation/SKILL.md):** NVE energy drift from an ML-potential MD run is a primary MLP validation metric. A validated MLP should show energy conservation comparable to the DFT reference. Anomalous drift indicates MLP extrapolation or force errors.
- **[mlp-to-lammps](../../ml-interatomic-potentials/mlp-to-lammps/SKILL.md):** MLP deployment in LAMMPS introduces constraints (no_domain_decomposition, single-node scaling) that affect equilibration wall time. This skill's multi-stage protocol applies; that skill covers the deployment mechanics.

## Key References

1. Basconi, J. E. & Shirts, M. R. (2013). *Effects of temperature control algorithms on transport properties and kinetics in molecular dynamics simulations.* Journal of Chemical Theory and Computation, 9(7), 2887–2899. https://doi.org/10.1021/ct400109a
2. Bussi, G., Donadio, D., & Parrinello, M. (2007). *Canonical sampling through velocity rescaling.* Journal of Chemical Physics, 126(1), 014101. https://doi.org/10.1063/1.2408420
3. Harvey, S. C., Tan, R. K.-Z., & Cheatham, T. E. (1998). *The flying ice cube: Velocity rescaling in molecular dynamics leads to violation of energy equipartition.* Journal of Computational Chemistry, 19(7), 726–740. https://doi.org/10.1002/(SICI)1096-987X(199805)19:7<726::AID-JCC4>3.0.CO;2-S
4. Parrinello, M. & Rahman, A. (1981). *Polymorphic transitions in single crystals: A new molecular dynamics method.* Journal of Applied Physics, 52(12), 7182–7190. https://doi.org/10.1063/1.328693
5. Shirts, M. R. & Chodera, J. D. (2008). *Statistically optimal analysis of samples from multiple equilibrium states.* Journal of Chemical Physics, 129(12), 124105. https://doi.org/10.1063/1.2978177
