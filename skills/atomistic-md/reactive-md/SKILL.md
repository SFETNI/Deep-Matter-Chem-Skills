---
name: reactive-md
domain: atomistic-md
maturity: draft
tools:
  - LAMMPS
  - ASE
  - OpenBabel
  - OVITO
  - MDAnalysis
tested_versions:
  LAMMPS: "2Aug2023"
  ASE: "3.23.0"
  OpenBabel: "3.1.1"
  OVITO: "3.10.6"
  MDAnalysis: "2.7.0"
requires_gpu: false
estimated_runtime: hours-to-days
---

# Reactive Molecular Dynamics

## Description

This skill covers reactive molecular dynamics (reactive MD) using LAMMPS with the ReaxFF and COMB force fields: force-field selection and parameter provenance, charge equilibration (QEq) setup and diagnostics, timestep and thermostat selection for reactive systems, thermal protocols for reaction sampling, reaction event detection and postprocessing, validation of reactive trajectories against DFT, and coupling reactive MD to active-learning ML potential workflows. Invoke this skill when bond-breaking or bond-forming chemistry must be modeled explicitly — combustion, oxidation, pyrolysis, stress corrosion cracking, catalyst surface chemistry, or shock-induced reactions — and a fixed-topology force field is physically inappropriate.

## Domain Context

Reactive MD differs from classical fixed-topology MD in one fundamental respect: **the connectivity of the bonding network is not fixed at the start of the simulation but emerges from the interatomic potential as atoms approach or recede**. This requires a potential energy function that varies continuously from a bonded to a non-bonded description as a function of local geometry.

**ReaxFF** (reactive force field) achieves this through a bond-order formalism. The key physical approximations are:

- **Bond order:** Each pair interaction carries a continuous bond order that depends on the interatomic distance and the local coordination environment. The bond order BO_ij decays smoothly from 1 (full bond) to 0 (no bond) as the interatomic distance increases. Bond order is not a quantum-mechanical quantity — it is a scalar empirical function parameterized to reproduce DFT bond-dissociation curves and reaction barriers.
- **Charge equilibration (QEq):** At each timestep, atomic partial charges are self-consistently minimized by the electronegativity equalization principle (Rappé & Goddard, 1991). Every atom's charge responds to its instantaneous bonding environment. QEq adds a per-timestep iterative solver that dominates the computational cost of ReaxFF and can fail to converge for systems with large charge separation.
- **Coupled energy terms:** Bond energy, overcoordination penalty, undercoordination energy, valence angle energy, torsion energy, and van der Waals and Coulomb interactions are all functions of bond order. These terms are globally coupled through a single parameter file (`ffield.reax`). **Partial substitution of parameters from different publications into one parameter file is physically invalid** — all ReaxFF parameters for all elements in the system must come from the same source and fitting procedure.
- **Transferability limits:** A ReaxFF parameterization is fitted to a specific set of elements and a specific set of reference reactions, geometries, and bond energies. The parameterization domain is not universal. Using ReaxFF outside the element set or chemistry it was fitted for — even for elements present in the parameter file — is extrapolation with unbounded errors.

**COMB (Charge-Optimized Many-Body)** is a related but distinct reactive potential. COMB3 is specifically designed for metal/oxide interfaces (Cu/ZnO, Pt/TiO₂, Ni/Al₂O₃) and uses a modified charge equilibration scheme with environment-dependent electronegativity. COMB3 covers different chemistry than ReaxFF and the two should not be conflated.

**ML interatomic potentials** (MACE, NequIP, DeePMD) can also model bond breaking and forming if the training data covers reactive events, but they are not reactive force fields in the ReaxFF sense — they are universal function approximators trained on DFT data. Their advantages over ReaxFF are higher accuracy and broader compositional flexibility; their disadvantage is the need for an explicit DFT dataset that covers reactive configurations.

**The central challenge of reactive MD** is not setting up the simulation but validating that the observed chemistry is physical and not an artifact of the parameterization. ReaxFF trajectories can produce spurious reactions at unrealistically low temperatures, miss important reaction pathways, or generate transition states at wrong barriers, all without any runtime error or warning.

## When to Use This Skill

- Modeling combustion, pyrolysis, or thermal decomposition where C-H, C-C, O-H, or N-O bonds must break and form, and a validated ReaxFF parameter set (e.g., Chenoweth 2008 for CHO, Liu 2018 for CHON) exists for the system.
- Studying oxidation kinetics of metals and alloys (Fe, Ni, Al) where O₂ dissociation and oxide film formation must be captured dynamically.
- Stress corrosion cracking and hydrogen embrittlement where bond breaking is mechanically induced and must be coupled to chemistry.
- Catalyst surface reactions (C-H activation, ammonia synthesis intermediates, CO₂ reduction) on metal surfaces for which ReaxFF has been validated.
- Generating reactive configurations (bond-breaking snapshots, transition-state-adjacent structures) for active-learning DFT labeling when a quick exploration of chemical space is needed before investing in an MLP.
- Mechanochemistry: reactions induced by mechanical stress (shear, compression) in solid-state materials where fixed-topology force fields cannot describe the bond rupture.

## When Not to Use This Skill

- **When a validated ReaxFF parameter set does not exist for the exact element set and chemistry of interest.** An unvalidated reactive potential is more dangerous than no reactive potential — it will produce reactions, but they will be wrong. Use DFT-MD (VASP AIMD) or a trained ML potential instead.
- **When only non-reactive processes are of interest** — diffusion, thermal expansion, phonon spectra, elastic properties. Use EAM, MEAM, Tersoff, or an MLP with lower computational cost and better-validated physical properties.
- **For quantitative reaction barriers, rate constants, or free energy profiles.** ReaxFF can identify that a reaction occurs, but the barriers are typically 20–50% off from CCSD(T) or even PBE-DFT benchmarks. For kinetics, use climbing-image NEB with DFT or DFT-MD with metadynamics.
- **When nuclear quantum effects govern the reaction** — H-atom tunneling at low temperature, proton transfer in hydrogen-bonded systems below 200 K. Classical ReaxFF dynamics neglects zero-point energy and tunneling entirely.
- **For large periodic systems (> 10,000 atoms at long timescales)** where the per-timestep QEq cost makes ReaxFF prohibitive. An ML potential trained on reactive DFT data scales better and will be more accurate. [EXPERT REVIEW NEEDED for exact system-size crossover]
- **When using GROMACS.** ReaxFF is only implemented in LAMMPS (and AMBER's REAX module for a limited element set). GROMACS does not support reactive force fields.

## Core Concepts

- **Bond order (BO):** A continuous, dimensionless scalar in [0, 1] that characterizes the bond between each atom pair as a function of distance and environment. In ReaxFF, BO_ij = BO_ij^σ + BO_ij^π + BO_ij^ππ for σ, π, and ππ bond contributions. A bond is conventionally considered present when BO_ij > 0.3 (the precise threshold is application-specific and **must be reported alongside any bond-order-based analysis**). Bond order in ReaxFF is NOT equivalent to quantum-mechanical bond order (Wiberg, Mayer) or to formal chemical bonding — it is an empirical fitting parameter whose physical meaning is defined by the ReaxFF functional form, not by quantum chemistry.
- **ffield.reax parameter file:** A single monolithic ASCII file containing all ReaxFF parameters for the element set: general bond parameters, element-specific parameters, bond parameters for each pair, off-diagonal van der Waals parameters, valence angle parameters, torsion parameters, and hydrogen bond parameters. All parameters in the file are globally coupled through the bond-order functional form. **Never modify individual sections of an ffield.reax file** by substituting parameters from a different publication — this produces a physically inconsistent potential whose behavior cannot be predicted or validated.
- **Charge equilibration (QEq):** At each timestep, ReaxFF solves a system of linear equations to find the partial charges {q_i} that minimize the electrostatic energy functional, subject to the constraint that the total charge is conserved. In LAMMPS this is implemented as `fix qeq/reaxff`, which is **mandatory and separate from the `pair_style reaxff` command**. QEq convergence depends on the tolerance (typically 1e-6), the cutoff radius (typically 10 Å, matching the ReaxFF non-bond cutoff), and the dielectric constant. QEq failure manifests as wildly oscillating charges, non-convergence warnings, or incorrect reaction products.
- **`pair_style reaxff NULL` / `pair_style reax/c NULL`:** LAMMPS has two implementations of ReaxFF: the older `reax/c` (C implementation, being deprecated) and the newer `reaxff` (Fortran-based, recommended from LAMMPS 2021 onward). The `NULL` keyword disables the legacy safezone/mincap settings (handled automatically in modern LAMMPS). `atom_style charge` is required — ReaxFF cannot run with `atom_style atomic`.
- **Units for ReaxFF:** ReaxFF is always run with `units real` in LAMMPS (energy in kcal/mol, distance in Å, time in fs). The `units metal` convention used for EAM and MLP potentials is **incompatible with ReaxFF parameter files**, which are explicitly parameterized in real units. Using `units metal` with ReaxFF silently rescales the timestep and all energy/force outputs.
- **Timestep for reactive MD:** ReaxFF requires much smaller timesteps than fixed-topology force fields because (a) bond-order changes are rapid at the timescale of bond breaking, (b) QEq charge updates at every step must track the bond order accurately, and (c) the high-frequency C-H and O-H stretch modes, while not constrained, must be resolved. Typical safe range: **0.1–0.5 fs** for systems containing H atoms; up to 1.0 fs for systems without H if validated by NVE energy conservation. [EXPERT REVIEW NEEDED for system-class-specific limits]
- **Thermostat choice for reactive MD:** Reactions are local exothermic or endothermic events that inject or remove heat on short timescales. The Langevin thermostat (stochastic friction + noise) dissipates local heat more effectively than the Nosé-Hoover thermostat, which couples the entire system to a single thermal reservoir and can exhibit slow heat transport in the presence of localized reaction events. Berendsen thermostat is fast for equilibration but does not generate the canonical ensemble. **For reactive production runs, use Langevin (LAMMPS: `fix langevin` + `fix nve`) or Nosé-Hoover (LAMMPS: `fix nvt`), with explicit NVE validation before NPT.** Never attempt NPT during rapid reactions — sudden density changes from exothermic reactions can crash the barostat.
- **Energy drift masking:** The thermostat in reactive MD continuously injects or removes energy. This makes thermostat-driven (NVT or NPT) energy drift diagnostics unreliable — the thermostat masks force-field errors. The correct diagnostic is to run a short NVE segment from an equilibrated structure, with the same timestep, and measure total energy conservation. If total energy drifts in NVE, the timestep is too large, the QEq is not converged, or the ffield.reax is incompatible with the system.
- **Reaction event detection via `fix reaxff/species`:** LAMMPS provides a built-in analysis fix that identifies molecular species by clustering atoms connected through bonds above a threshold bond order. It writes a species inventory at regular intervals: the number of each molecule type (identified by molecular formula) vs. time. This is the primary in-situ reaction monitoring tool. Post-simulation species analysis can be done with OpenBabel (converts LAMMPS atom-level data to molecular structures) or ChemTraYzer (automated reaction network extraction).
- **Bond-order cutoff for species identification:** The bond-order threshold used by `fix reaxff/species` to define a bond (default: 0.3 for most elements, 0.55 for C-C bonds in some parameterizations) determines which atoms are considered bonded. **This cutoff is an analysis parameter, not a physical constant.** Changing the cutoff by 0.1 can merge or split species in the output. Always report the cutoff used, verify it produces chemically sensible molecules for the known starting materials, and be aware that transient species (bonds with BO close to the threshold) will appear and disappear in the inventory due to thermal fluctuations rather than real reactions.
- **High-temperature artifacts:** At temperatures > 3000–4000 K (common in combustion and shock simulations), ReaxFF can produce chemistry that is unphysical for the force field's parameterization domain: highly exotic radical species, triple-bonded metallic species, or atom types in coordination environments not present in the fitting database. These artifacts produce real trajectory data but represent force-field extrapolation, not physical chemistry. [EXPERT REVIEW NEEDED for specific temperature limits per parameter set]
- **Parameter provenance and version control:** Multiple incompatible ffield.reax files exist for the same element set, published by different groups or revised over time. Files may share names but contain different parameter values. Always record the SHA-256 hash and the primary citation for every ffield.reax file. The LAMMPS source repository ships a set of example parameter files that may differ from those in the primary publications — prefer files obtained directly from the original authors or from verified repositories (e.g., the ReaxFF parameter database at reaxff.nl or the Open Knowledgebase of Interatomic Models, OpenKIM).
- **COMB3 vs. ReaxFF:** COMB3 (pair_style comb3 in LAMMPS) is a charge-equilibrated many-body potential parameterized specifically for metal/oxide interfaces. It uses a different charge equilibration scheme (`fix qeq/comb`) and different parameter files. COMB3 is not a replacement for ReaxFF — it covers a different compositional space with different validation history. Do not use COMB3 parameters with the ReaxFF pair_style or vice versa.

## Key Workflows

### Workflow 1: LAMMPS ReaxFF input anatomy — combustion simulation (C/H/O)

A complete annotated LAMMPS input for NVT reactive MD of a hydrocarbon/oxygen mixture at 2500 K using the Chenoweth 2008 CHO parameter set.

```lammps
# ============================================================
# NVT reactive MD of decane/O2 combustion at 2500 K
# ReaxFF: Chenoweth, van Duin, Goddard (2008), J. Phys. Chem. A
# LAMMPS units: real (kcal/mol, Å, fs, K, atm)
# ============================================================

# ─── Initialization ──────────────────────────────────────────────────────────
units           real            # REQUIRED for ReaxFF; do NOT use units metal
dimension       3
boundary        p p p
atom_style      charge          # REQUIRED: ReaxFF stores per-atom charges

# ─── Structure ───────────────────────────────────────────────────────────────
read_data       decane_O2_mixture.lammps
# Data file must include:
#   - Charge column (atom_style charge)
#   - Correct atom type ordering matching pair_coeff element list
#   - Initial charges can be 0.0; QEq will self-consistently update them

# ─── Force field ─────────────────────────────────────────────────────────────
pair_style      reaxff NULL
# NULL: no legacy safezone/mincap (required for LAMMPS 2021+)
# For older LAMMPS (pre-2020): pair_style reax/c NULL
# WARNING: reax/c and reaxff use different internal data structures;
#          never mix pair_style with parameters from the other implementation.

pair_coeff      * * ffield.reax.CHO C H O
# Element list MUST match atom type ordering in the data file.
# Type 1 → C, Type 2 → H, Type 3 → O.
# The ffield.reax.CHO file is from Chenoweth et al. (2008), J. Phys. Chem. A, 112, 1040.
# SHA-256 of ffield.reax.CHO: [record this before use]
# NEVER use a ffield.reax file from a different publication for CHO chemistry
# without re-validating the reaction barriers and species heats of formation.

# ─── Charge equilibration (MANDATORY) ─────────────────────────────────────────
fix             qeq all qeq/reaxff 1 0.0 10.0 1.0e-6 reaxff
# Syntax: fix ID group-ID qeq/reaxff Nevery cutlo cuthi tolerance keyword
#   Nevery = 1: equilibrate charges at every timestep (required for reactive MD)
#   cutlo = 0.0 Å: lower cutoff (start of charge interaction)
#   cuthi = 10.0 Å: upper cutoff (must match ReaxFF non-bond cutoff in ffield)
#   tolerance = 1e-6: convergence criterion for the iterative charge solver
#   reaxff: use the ReaxFF electronegativity/hardness parameters from ffield
#
# QEq convergence failure symptoms:
#   - LAMMPS WARNING: QEq loop did not converge in N iterations
#   - Wildly oscillating per-atom charges (print fix ave/atom or dump charge)
#   - Runaway energies or forces despite correct pair_style setup
#
# QEq failure causes:
#   - cuthi does not match the ReaxFF non-bond cutoff in ffield.reax
#   - The system has a large charge separation (ionic solid, large charge gradient)
#   - Atoms are too close (overlapping); minimize before running QEq
#   - Wrong element ordering in pair_coeff (causes wrong electronegativity parameters)

# ─── Neighbor list ────────────────────────────────────────────────────────────
neighbor        2.0 bin
neigh_modify    every 1 delay 0 check no
# ReaxFF requires the neighbor list to be rebuilt every step (check no = always rebuild).
# Skin distance 2.0 Å is standard; do not reduce below 1.5 Å for reactive systems
# where new bonds can form rapidly.

# ─── Timestep ─────────────────────────────────────────────────────────────────
timestep        0.25
# 0.25 fs in real units (fs). Safe for C/H/O with H atoms present.
# Maximum recommended: 0.5 fs (with H atoms); 1.0 fs (heavy atom only systems).
# [EXPERT REVIEW NEEDED for exact limits per parameter set]
# If NVE energy drifts > 0.01 kcal/mol/atom over 10,000 steps, reduce timestep.
# If simulation crashes with "Lost atoms" or NaN energies, halve the timestep first.

# ─── Equilibration: NVT heating from 300 K to 2500 K ─────────────────────────
velocity        all create 300.0 54321 mom yes rot yes dist gaussian
# Initialize velocities at 300 K before heating ramp; QEq will adjust charges.

fix             heat all nvt temp 300.0 2500.0 25.0
# Nosé-Hoover NVT with Tdamp = 25 fs in real units (25 × timestep = 100 steps).
# Tdamp for ReaxFF: ~50–100 × timestep. Too small → temperature oscillations.
# Too large → thermostat cannot dissipate reaction heat → runaway temperature.
# [EXPERT REVIEW NEEDED]

thermo          100
thermo_style    custom step temp press vol etotal ke pe
thermo_modify   flush yes norm no

dump            heat_dump all custom 500 heat.dump id type q x y z
# Include charge (q) in dump: essential for monitoring QEq convergence.
# Charge should be physically reasonable: C ≈ -0.2 to +0.2 e in hydrocarbons;
# O ≈ -0.5 to -0.8 e in C-O bonds; H ≈ +0.1 to +0.3 e.

run             40000       # 10 ps heating from 300 K to 2500 K
unfix           heat
undump          heat_dump

# ─── Check before production: short NVE test ─────────────────────────────────
# (see Workflow 5 for full NVE protocol)
# Minimally: confirm temperature is near 2500 K; check per-atom charges are physical;
# check no "Lost atoms" errors; inspect species output for chemically reasonable
# starting composition (mostly decane + O2 at this point).

# ─── Production: NVT at 2500 K ────────────────────────────────────────────────
fix             prod all nvt temp 2500.0 2500.0 25.0
reset_timestep  0

# Species tracking (primary reactive MD observable)
fix             species all reaxff/species 100 1 100 species.out element C H O
# Syntax: fix ID group reaxff/species Nevery Nrepeat Nfreq outfile keyword values
#   Nevery = 100: sample bond orders every 100 steps (every 25 fs)
#   Nrepeat = 1: use 1 sample per output (instantaneous snapshot)
#   Nfreq = 100: write output every 100 steps
#   element keyword: map atom types to element symbols in output
# Bond order threshold: default 0.3 for all bonds.
# For high-temperature combustion, some groups use 0.4 to suppress transient species.
# ALWAYS report the threshold used alongside any species analysis.

dump            prod_dump all custom 1000 prod.dump id type q x y z
dump_modify     prod_dump sort id

thermo          500
thermo_style    custom step temp press vol etotal ke pe
run             400000      # 100 ps production at 2500 K
unfix           prod
unfix           species

write_data      final.lammps
write_restart   final.restart
```

### Workflow 2: Charge equilibration diagnostics and QEq parameter selection

QEq convergence is the most common silent failure mode in ReaxFF simulations. These diagnostics detect and fix QEq problems before they corrupt a production run.

```python
import numpy as np
from pathlib import Path


def parse_reaxff_charge_dump(
    dump_file: str,
    n_atoms: int,
    element_map: dict,   # {atom_type_int: "El"}, e.g. {1: "C", 2: "H", 3: "O"}
) -> dict:
    """
    Parse per-atom charges from a LAMMPS dump file written with:
    dump ID all custom N file id type q x y z

    Returns a dict with:
      - frames: list of per-atom charge arrays (n_atoms,)
      - per_element_mean_q: dict of element → mean charge over all frames
      - per_element_std_q: dict of element → std of charge over all frames

    Physical charge ranges for common ReaxFF systems:
      C in hydrocarbons: -0.3 to +0.3 e
      C in CO2: +0.6 to +0.9 e
      H bonded to C: +0.05 to +0.15 e
      H bonded to O (OH): +0.30 to +0.45 e
      O (bridging): -0.4 to -0.6 e
      O (terminal, in O2): -0.1 to +0.1 e
      Fe (metal): 0 to +2 e depending on oxidation state
    [EXPERT REVIEW NEEDED: ranges depend on parameterization]
    """
    frames = []
    types = []
    in_atoms = False
    current_frame = {}

    with open(dump_file) as f:
        for line in f:
            line = line.strip()
            if line.startswith("ITEM: ATOMS"):
                in_atoms = True
                col_names = line.split()[2:]  # headers after "ITEM: ATOMS"
                try:
                    id_col = col_names.index("id")
                    type_col = col_names.index("type")
                    q_col = col_names.index("q")
                except ValueError:
                    raise ValueError(
                        "Dump file must include 'id type q' columns. "
                        "Verify: dump ID all custom N file id type q x y z"
                    )
                current_charges = np.zeros(n_atoms)
                current_types = np.zeros(n_atoms, dtype=int)
                atom_count = 0
                continue
            if in_atoms and line and not line.startswith("ITEM:"):
                parts = line.split()
                atom_id = int(parts[id_col]) - 1   # 0-indexed
                atom_type = int(parts[type_col])
                charge = float(parts[q_col])
                current_charges[atom_id] = charge
                current_types[atom_id] = atom_type
                atom_count += 1
                if atom_count == n_atoms:
                    frames.append(current_charges.copy())
                    if not types:
                        types = current_types.copy()
                    in_atoms = False
            elif line.startswith("ITEM:"):
                in_atoms = False

    frames = np.array(frames)   # (n_frames, n_atoms)

    # Per-element charge statistics
    per_element_mean = {}
    per_element_std = {}
    for atype, el in element_map.items():
        mask = (types == atype)
        if mask.sum() == 0:
            continue
        q_el = frames[:, mask]    # (n_frames, n_atoms_of_element)
        per_element_mean[el] = q_el.mean()
        per_element_std[el] = q_el.std()

    print(f"Parsed {len(frames)} frames, {n_atoms} atoms")
    print(f"\nPer-element charge summary (mean ± std over all frames):")
    for el, mean_q in sorted(per_element_mean.items()):
        print(f"  {el:>4}: {mean_q:>+8.4f} ± {per_element_std[el]:.4f} e")

    # Check charge neutrality
    total_q_per_frame = frames.sum(axis=1)
    if abs(total_q_per_frame.mean()) > 0.01:
        print(f"\nWARNING: Total charge is not zero: {total_q_per_frame.mean():.4f} ± "
              f"{total_q_per_frame.std():.4f} e. "
              "Check element mapping in pair_coeff and initial charge assignments.")
    else:
        print(f"\nTotal charge: {total_q_per_frame.mean():.6f} ± "
              f"{total_q_per_frame.std():.6f} e (charge neutral ✓)")

    return {"frames": frames, "types": types,
            "per_element_mean_q": per_element_mean,
            "per_element_std_q": per_element_std}


def diagnose_qeq_convergence(
    log_file: str,
    max_iter_expected: int = 100,
) -> dict:
    """
    Parse LAMMPS log for QEq non-convergence warnings.

    LAMMPS prints: "WARNING: Fix qeq/reaxff group ... did not converge in N iterations"
    when the iterative QEq solver reaches max_iter without meeting the tolerance.

    Returns a dict with the number of non-convergence events and the steps
    at which they occurred.
    """
    non_convergence_steps = []
    with open(log_file) as f:
        step = None
        for line in f:
            if line.strip().startswith("Step") and "Temp" in line:
                step = None   # Reset; step number read from thermo output below
            if "WARNING: Fix qeq" in line and "not converge" in line:
                non_convergence_steps.append(step)

    if non_convergence_steps:
        print(f"QEq non-convergence events: {len(non_convergence_steps)}")
        print(f"  First occurrence: step {non_convergence_steps[0]}")
        print(f"\nDiagnosis guide:")
        print("  1. If sparse (< 0.1% of steps): thermal fluctuation near a reaction event.")
        print("     Likely benign; monitor whether total energy drifts concurrently.")
        print("  2. If frequent or at step 0: wrong element mapping or wrong cuthi.")
        print("     Check that cuthi in fix qeq matches the ReaxFF non-bond cutoff.")
        print("  3. If after a sudden density change: NPT barostat may have compressed the")
        print("     box too rapidly. Switch to NVT and re-equilibrate.")
        print("  4. If persistent: try increasing max_iter (LAMMPS fix qeq maxiter keyword)")
        print("     or tightening geometry (energy minimize before reactive MD).")
    else:
        print("No QEq non-convergence warnings detected.")

    return {"n_non_convergence": len(non_convergence_steps),
            "steps": non_convergence_steps}


def check_qeq_cutoff_consistency(ffield_file: str) -> float:
    """
    Extract the non-bond cutoff from a ReaxFF ffield.reax file and confirm
    it matches the cuthi parameter in fix qeq/reaxff.

    The ReaxFF non-bond cutoff is stored in the ffield.reax general parameters
    section (line 2 of the file, column varies by version).

    WARNING: ffield.reax format is not standardized across versions.
    This function handles the Chenoweth/Goddard format; other formats
    may differ. Always verify manually. [EXPERT REVIEW NEEDED]
    """
    with open(ffield_file) as f:
        lines = f.readlines()

    # In Chenoweth-format ffield.reax, the general parameters header is:
    # Line 1: comment
    # Line 2: number of general parameters (e.g., "39  ! Number of general parameters")
    # Lines 3..: one parameter per line with value and description
    # The non-bond van der Waals cutoff is typically parameter 12 or 13.
    # Parse lines until we find a line mentioning "vdWaals cutoff" or "Outer cutoff"
    cutoff = None
    for i, line in enumerate(lines):
        lower = line.lower()
        if "cutoff" in lower and "vdwaals" in lower or "outer cutoff" in lower:
            parts = line.split()
            try:
                cutoff = float(parts[0])
                print(f"ReaxFF non-bond cutoff from ffield: {cutoff:.2f} Å  (line {i+1})")
                print(f"  Verify: fix qeq/reaxff cuthi parameter must equal {cutoff:.1f}")
                print(f"  Default in LAMMPS examples is 10.0 Å; ffield may differ.")
            except (ValueError, IndexError):
                pass
            break

    if cutoff is None:
        print("Could not auto-detect non-bond cutoff from ffield.reax.")
        print("Parse manually: look for 'outer cutoff' or 'vdWaals cutoff' in the file.")
        print("Set fix qeq/reaxff cuthi to match this value exactly.")

    return cutoff
```

### Workflow 3: Thermal protocol for reactive simulations

Reactive MD requires a staged thermal protocol that is more conservative than classical MD: reactions occurring during heating are physically real but may obscure the equilibrium chemistry at the target temperature.

```lammps
# ============================================================
# Staged thermal protocol for reactive MD
# System: iron oxide formation (Fe + O2), 4000 atoms
# ReaxFF: Aryanpour, van Duin, Kubicki (2010), J. Phys. Chem. A
# units: real
# ============================================================

# ─── Stage 0: Structure preparation ─────────────────────────────────────────
# Fe surface + O2 molecules above it
read_data       Fe_O2_interface.lammps

pair_style      reaxff NULL
pair_coeff      * * ffield.reax.Fe_O Fe O
fix             qeq all qeq/reaxff 1 0.0 10.0 1e-6 reaxff
neighbor        2.0 bin
neigh_modify    every 1 delay 0 check no
timestep        0.25

# Energy minimization BEFORE any MD
# This is mandatory: even small atomic overlaps produce enormous ReaxFF forces
# that prevent QEq convergence and can crash the simulation.
fix             relax all box/relax iso 0.0 vmax 0.001
min_style       cg
minimize        1e-6 1e-8 10000 100000
unfix           relax
# After minimization: check per-atom charges are physical (Fe ≈ 0 to +1.5 e, O ≈ -0.2 e)

# ─── Stage 1: NVT at 300 K — thermalization ────────────────────────────────
velocity        all create 300.0 78901 mom yes rot yes dist gaussian
fix             nvt_300 all nvt temp 300.0 300.0 25.0
thermo          200
thermo_style    custom step temp press vol etotal ke pe
fix             species_300 all reaxff/species 500 1 500 species_300K.out element Fe O

dump            dump_300 all custom 1000 nvt_300.dump id type q x y z
dump_modify     dump_300 sort id

run             20000       # 5 ps at 300 K
# Expected: species inventory shows mostly Fe atoms and O2 molecules.
# If Fe2O3 or FeO appears at 300 K: either there is a structure overlap,
# the ffield has unphysical O2 binding, or the initial geometry is wrong.
unfix           nvt_300
unfix           species_300
undump          dump_300

# ─── Stage 1b: NVE stability check before any heating ──────────────────────
# Run 5000 steps NVE from the 300 K equilibrated structure.
# Total energy should be conserved; any drift > 0.01 kcal/mol/atom/ns = red flag.
unfix           qeq
fix             qeq all qeq/reaxff 1 0.0 10.0 1e-6 reaxff  # re-apply with same settings
fix             nve_test all nve
thermo          100
thermo_style    custom step etotal ke pe temp
run             5000
unfix           nve_test
# If |Δ(etotal)| / natoms > 0.05 kcal/mol over 5000 × 0.25 fs = 1.25 ps:
# reduce timestep to 0.1 fs and repeat.

# ─── Stage 2: NVT heating ramp 300 K → 1000 K ────────────────────────────
# Ramp is slow enough that chemistry proceeds quasi-statically.
# For oxidation studies: early O2 dissociation on Fe surface may begin here.
fix             heat all nvt temp 300.0 1000.0 25.0
fix             sp_heat all reaxff/species 200 1 200 species_heat.out element Fe O
dump            dump_heat all custom 500 heat.dump id type q x y z
dump_modify     dump_heat sort id

run             80000       # 20 ps heating (300 → 1000 K)
unfix           heat
unfix           sp_heat
undump          dump_heat

# ─── Stage 3: NVT equilibration at 1000 K ─────────────────────────────────
fix             nvt_1000 all nvt temp 1000.0 1000.0 25.0
fix             sp_1000 all reaxff/species 100 1 100 species_1000K.out element Fe O

dump            dump_1000 all custom 1000 nvt_1000.dump id type q x y z
dump_modify     dump_1000 sort id

run             200000      # 50 ps equilibration at 1000 K
unfix           nvt_1000
unfix           sp_1000
undump          dump_1000

# ─── Stage 4: NVE production for reaction sampling ─────────────────────────
# NVE is preferred for unbiased reaction kinetics: the thermostat in NVT can
# artificially accelerate or suppress reactions by removing/injecting heat at
# the site of a reaction event. NVE provides unbiased microcanonical dynamics.
# For long production (> 100 ps): use NVT to prevent long-term T drift.
# [EXPERT REVIEW NEEDED for which ensemble is appropriate for rate extraction]
reset_timestep  0
fix             prod_nve all nve
fix             sp_prod all reaxff/species 50 1 50 species_prod.out element Fe O

dump            prod_dump all custom 500 prod.dump id type q x y z
dump_modify     prod_dump sort id

thermo          500
thermo_style    custom step temp press vol etotal ke pe
run             400000      # 100 ps production at 1000 K (NVE)

unfix           prod_nve
unfix           sp_prod
write_data      final_oxidized.lammps
write_restart   final_oxidized.restart
```

### Workflow 4: Reaction event detection and species postprocessing

`fix reaxff/species` provides a time-resolved species inventory. Post-processing extracts reaction events, rates, and the reaction network from this inventory.

```python
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
import re


def parse_reaxff_species_output(species_file: str) -> pd.DataFrame:
    """
    Parse the output of 'fix reaxff/species' from LAMMPS.

    Output format (species.out):
      # Timestep  No_Moles  No_Specs  C1H4O1  C2H6  H2O  O2  ...
      100         250       4         10       5     30   20  ...
      200         250       3         ...

    Returns a DataFrame with columns: Timestep + one column per species formula.

    Notes:
    - Species formulas are sorted by (element counts) and may differ between
      timesteps if new species appear. This function handles variable columns.
    - Transient species with count = 1 at a single timestep are often thermal
      fluctuations near the bond-order threshold, not real reaction events.
      Apply a persistence filter (species must appear for > N consecutive
      timesteps) before interpreting as chemistry.
    """
    records = []
    all_species = set()

    with open(species_file) as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("# Timestep"):
            # Header line: extract species names (column 4 onward)
            header = line.lstrip("# ").split()
            species_names = header[3:]   # after Timestep, No_Moles, No_Specs
            all_species.update(species_names)
            i += 1
            continue
        if line.startswith("#") or not line:
            i += 1
            continue
        # Data line
        parts = line.split()
        if len(parts) >= 3:
            step = int(parts[0])
            no_moles = int(parts[1])
            no_specs = int(parts[2])
            counts = [int(p) for p in parts[3:3+len(species_names)]]
            record = {"Timestep": step, "No_Moles": no_moles, "No_Specs": no_specs}
            for sp, cnt in zip(species_names, counts):
                record[sp] = cnt
            records.append(record)
        i += 1

    df = pd.DataFrame(records).fillna(0)
    for col in df.columns[3:]:
        df[col] = df[col].astype(int)

    print(f"Species output: {len(df)} timesteps, {len(df.columns)-3} unique species")
    print(f"Timestep range: {df['Timestep'].min()} – {df['Timestep'].max()}")
    print(f"\nSpecies present (max count over trajectory):")
    for col in sorted(df.columns[3:]):
        max_c = df[col].max()
        if max_c > 0:
            print(f"  {col:>15}: max = {max_c:>6}")

    return df


def extract_reaction_events(
    df: pd.DataFrame,
    min_delta: int = 2,
    persistence_steps: int = 3,
    reactant_species: list = None,
    product_species: list = None,
) -> list:
    """
    Identify reaction events from a species time series.

    A reaction event is defined as a change in species count that:
    - Persists for at least persistence_steps consecutive timesteps
    - Has a magnitude |Δcount| ≥ min_delta

    min_delta = 2 avoids flagging transient single-molecule bond fluctuations
    at the QEq threshold. [EXPERT REVIEW NEEDED]

    reactant_species: list of species formulas to monitor as reactants.
    product_species: list of species formulas to monitor as products.
    If None, all species with count changes are returned.

    Returns a list of dicts:
        {"timestep": T, "species": "H2O", "delta": +3,
         "direction": "appearance" or "disappearance",
         "persistent": True}
    """
    events = []
    species_cols = df.columns[3:].tolist()
    if reactant_species or product_species:
        target = set(reactant_species or []) | set(product_species or [])
        species_cols = [c for c in species_cols if c in target]

    for sp in species_cols:
        counts = df[sp].values
        steps = df["Timestep"].values
        diffs = np.diff(counts)

        for i, delta in enumerate(diffs):
            if abs(delta) < min_delta:
                continue
            # Check persistence: the new count level must last persistence_steps steps
            if i + persistence_steps >= len(counts):
                persist = True   # Near end of trajectory
            else:
                # All subsequent persistence_steps counts should remain near the new value
                new_level = counts[i + 1]
                persist = all(
                    abs(counts[i + 1 + j] - new_level) <= min_delta
                    for j in range(persistence_steps)
                )
            events.append({
                "timestep": int(steps[i + 1]),
                "species": sp,
                "delta": int(delta),
                "direction": "appearance" if delta > 0 else "disappearance",
                "persistent": persist,
            })

    # Sort by timestep
    events.sort(key=lambda e: e["timestep"])

    persistent_events = [e for e in events if e["persistent"]]
    print(f"\nReaction events detected: {len(events)} total, "
          f"{len(persistent_events)} persistent (duration ≥ {persistence_steps} steps)")

    if persistent_events:
        print("\nFirst 10 persistent events:")
        for e in persistent_events[:10]:
            arrow = "→ appeared" if e["delta"] > 0 else "→ disappeared"
            print(f"  Step {e['timestep']:>8d}: {e['species']:>15} {arrow}  "
                  f"Δcount = {e['delta']:>+4d}")

    return events


def compute_species_rates(
    df: pd.DataFrame,
    timestep_fs: float = 0.25,
    dump_frequency: int = 100,
    temperature_K: float = 1000.0,
) -> dict:
    """
    Compute approximate formation and consumption rates for each species.

    Rate is computed as the time-averaged first difference of the species count,
    normalized by the system volume.

    WARNING: These rates are descriptive summary statistics from the simulation,
    NOT validated kinetic rate constants. ReaxFF reaction barriers are
    typically 20–50% in error relative to high-level quantum chemistry.
    Do not use ReaxFF rates for quantitative kinetic modeling without
    validation against DFT NEB barriers or experimental Arrhenius data.
    [EXPERT REVIEW NEEDED]
    """
    dt_ps = timestep_fs * dump_frequency / 1000.0   # ps per output frame
    species_cols = df.columns[3:].tolist()
    rates = {}

    for sp in species_cols:
        counts = df[sp].values.astype(float)
        diffs = np.diff(counts) / dt_ps   # molecules/ps
        positive = diffs[diffs > 0]
        negative = diffs[diffs < 0]
        rates[sp] = {
            "mean_formation_rate_per_ps": float(positive.mean()) if len(positive) > 0 else 0.0,
            "mean_consumption_rate_per_ps": float(-negative.mean()) if len(negative) > 0 else 0.0,
            "net_change": int(counts[-1] - counts[0]),
        }

    print(f"\nSpecies rate summary (T = {temperature_K:.0f} K):")
    print(f"{'Species':>15}  {'Formation (/ps)':>18}  {'Consumption (/ps)':>19}  "
          f"{'Net Δ':>8}")
    print("─" * 65)
    for sp, r in sorted(rates.items(), key=lambda x: -abs(x[1]["net_change"])):
        f_rate = r["mean_formation_rate_per_ps"]
        c_rate = r["mean_consumption_rate_per_ps"]
        net = r["net_change"]
        if f_rate > 0 or c_rate > 0 or net != 0:
            print(f"  {sp:>15}  {f_rate:>18.3f}  {c_rate:>19.3f}  {net:>8d}")

    return rates


def extract_structures_for_dft(
    dump_file: str,
    species_df: pd.DataFrame,
    target_species: str,
    n_structures: int = 10,
    bond_order_threshold: float = 0.3,
    timestep_fs: float = 0.25,
    dump_every: int = 1000,
) -> list:
    """
    Extract atomic configurations from a reactive MD trajectory at timesteps
    where a target species first appears, for DFT single-point labeling.

    This is the primary coupling point between reactive MD and active-learning
    MLP workflows: ReaxFF identifies reactive configurations; DFT labels them;
    the MLP learns from the labeled structures.

    Returns a list of ASE Atoms objects.
    """
    from ase.io import read

    # Identify timesteps where target species count changes from 0 to > 0
    counts = species_df[target_species].values if target_species in species_df else None
    if counts is None:
        print(f"Species '{target_species}' not found in species output.")
        return []

    steps = species_df["Timestep"].values
    appearance_steps = steps[
        (np.diff(np.concatenate([[0], counts])) > 0) & (counts > 0)
    ]

    print(f"\n{target_species} first appears at {len(appearance_steps)} timestep windows.")
    if len(appearance_steps) == 0:
        print(f"  {target_species} never appears in the trajectory.")
        return []

    # Read trajectory frames at or near those timesteps
    # LAMMPS dump frame index = (timestep - first_step) / dump_every
    all_frames = read(dump_file, index=":", format="lammps-dump-text")
    frame_steps = [f.info.get("timestep", i * dump_every) for i, f in enumerate(all_frames)]
    frame_step_arr = np.array(frame_steps)

    selected = []
    for target_step in appearance_steps[:n_structures]:
        closest_idx = np.argmin(np.abs(frame_step_arr - target_step))
        selected.append(all_frames[closest_idx])
        print(f"  Step {target_step}: selected frame at step "
              f"{frame_step_arr[closest_idx]} (frame idx {closest_idx})")

    print(f"\nExtracted {len(selected)} structures for DFT labeling.")
    print("Next: run DFT single-point calculations (see vasp-workflow, dft-dataset-generation)")
    print("to obtain energies, forces, and stresses for active-learning MLP training.")

    return selected
```

### Workflow 5: NVE drift check and thermostat masking detection for reactive MD

Energy conservation in NVE is the definitive test that the timestep is appropriate and QEq is converging. In reactive MD, this check is more complex than for classical MD because reactions are exothermic or endothermic events that naturally perturb the energy.

```python
import numpy as np
from pathlib import Path


def parse_lammps_log_reactive(log_file: str) -> dict:
    """
    Parse LAMMPS thermo output from a reactive MD log.
    Extracts Step, Temp, Press, Etotal, KE, PE.
    Handles multiple run blocks from a staged protocol.
    """
    data = {}
    header = []
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


def nve_energy_conservation_reactive(
    log_file: str,
    natoms: int,
    timestep_fs: float = 0.25,
    n_steps_nve: int = 5000,
    max_drift_kcal_mol_atom_ns: float = 1.0,
) -> dict:
    """
    Assess NVE energy conservation for a reactive MD system.

    max_drift_kcal_mol_atom_ns: [EXPERT REVIEW NEEDED]
      Classical EAM/MLP: < 0.1 kcal/mol/atom/ns
      ReaxFF: < 1.0 kcal/mol/atom/ns (larger tolerance because QEq introduces
               numerical noise at each step that exceeds classical integrator error)

    If drift exceeds the threshold:
      1. Halve the timestep and re-run the NVE test
      2. Tighten QEq tolerance from 1e-6 to 1e-8 and re-run
      3. If drift persists: the ffield.reax may be incompatible with the system
         or there are structures with unphysical coordination at this temperature.
    """
    log = parse_lammps_log_reactive(log_file)

    if "Etotal" not in log:
        raise ValueError("Etotal not found in log. "
                         "Ensure thermo_style includes 'etotal'.")

    # Use only the NVE segment (last n_steps_nve data points if mixed with NVT)
    E = log["Etotal"][-n_steps_nve:]
    T = log["Temp"][-n_steps_nve:] if "Temp" in log else None
    steps = log["Step"][-n_steps_nve:] if "Step" in log else np.arange(len(E))

    # Compute drift
    time_ns = (steps[-1] - steps[0]) * timestep_fs / 1e6   # fs × step → ns
    slope_kcal_per_step, intercept = np.polyfit(steps, E, 1)
    drift_per_ns = abs(slope_kcal_per_step) * (steps[-1] - steps[0]) / natoms / time_ns

    # Energy spread (should be small in NVE if temperature is fixed by thermostat before)
    E_spread_kcal = (E.max() - E.min()) / natoms * 1000   # cal/mol/atom
    E_std_kcal = E.std() / natoms * 1000

    print(f"NVE energy conservation ({n_steps_nve} steps × {timestep_fs:.2f} fs):")
    print(f"  Total time:    {time_ns*1000:.2f} ps")
    print(f"  E drift:       {drift_per_ns:.4f} kcal/mol/atom/ns")
    print(f"  E range:       {E_spread_kcal:.4f} cal/mol/atom")
    print(f"  E std:         {E_std_kcal:.4f} cal/mol/atom")
    if T is not None:
        print(f"  T range:       {T.min():.1f} – {T.max():.1f} K")
        print(f"  T drift in NVE: {T[-1]-T[0]:+.1f} K "
              f"(should be < ±5% of mean T; larger = timestep too large or reaction heat)")

    ok = drift_per_ns < max_drift_kcal_mol_atom_ns
    if ok:
        print(f"  ✓ PASS: drift < {max_drift_kcal_mol_atom_ns:.1f} kcal/mol/atom/ns")
    else:
        print(f"  ✗ FAIL: drift {drift_per_ns:.3f} > {max_drift_kcal_mol_atom_ns:.1f} "
              f"kcal/mol/atom/ns")
        print("  Actions:")
        print("    1. Reduce timestep by 2× and repeat NVE test")
        print("    2. Tighten QEq tolerance: fix qeq/reaxff ... 1e-8 reaxff")
        print("    3. Re-minimize initial structure before MD")
        print("    4. Check ffield.reax provenance and element mapping in pair_coeff")

    return {
        "drift_kcal_mol_atom_ns": float(drift_per_ns),
        "E_std_cal_mol_atom": float(E_std_kcal),
        "pass": ok,
        "n_steps": n_steps_nve,
    }


def detect_thermostat_masking(
    log_file: str,
    natoms: int,
    timestep_fs: float = 0.25,
) -> dict:
    """
    Detect thermostat masking: a pathological state where the thermostat
    continuously injects or removes energy to compensate for force-field errors,
    producing a stable-looking trajectory that is physically wrong.

    Signature of thermostat masking in reactive MD:
    - NVT temperature is stable and at target
    - BUT short NVE segment from equilibrated structure shows large drift
    - AND the thermostat energy input (= Etotal - KE - PE = thermostat coupling work)
      is large in magnitude compared to PE changes from chemistry

    This function estimates the thermostat coupling work from the log:
    E_thermo_work ≈ Etotal - (KE + PE)  [should be ~ 0 for good MD]

    In LAMMPS, Etotal = KE + PE in NVE; in NVT, Etotal includes the
    Nosé-Hoover extended variable energy. This diagnostic only applies
    to NVT runs where the thermostat energy is NOT included in Etotal.
    [EXPERT REVIEW NEEDED: LAMMPS NVT etotal definition varies by version]
    """
    log = parse_lammps_log_reactive(log_file)

    required = ["Etotal", "KE", "PE"]
    if not all(k in log for k in required):
        print("Log missing KE or PE column. Cannot compute thermostat masking diagnostic.")
        print("Add 'ke pe' to thermo_style.")
        return {}

    E = log["Etotal"]
    KE = log["KE"]
    PE = log["PE"]

    # In NVE: Etotal = KE + PE (exactly). Residual = 0 by definition.
    # In NVT (Nosé-Hoover): Etotal is usually KE+PE (LAMMPS convention),
    # and the Nosé-Hoover variable is NOT included in Etotal.
    # The thermostat injects energy: ΔE_thermo = ΔEtotal = Δ(KE+PE) over time.
    # If Etotal drifts significantly while in NVT: thermostat is masking a problem.

    # Rolling mean of Etotal drift
    window = min(500, len(E) // 4)
    if window < 10:
        print("Trajectory too short for thermostat masking diagnostic.")
        return {}

    E_rolling = np.convolve(E, np.ones(window)/window, mode="valid")
    E_drift_per_atom = (E_rolling[-1] - E_rolling[0]) / natoms

    # KE fluctuations (proxy for how hard thermostat is working)
    KE_fluct = KE.std() / (1.5 * natoms * 8.617e-5 * log["Temp"].mean()
                             if "Temp" in log else 1.0)

    print(f"Thermostat masking diagnostic:")
    print(f"  Etotal drift (NVT): {E_drift_per_atom:.4f} kcal/mol/atom")
    print(f"  KE relative fluctuation: {KE_fluct:.4f}")

    masked = abs(E_drift_per_atom) > 0.5   # kcal/mol/atom over full run [EXPERT REVIEW NEEDED]
    if masked:
        print("  WARNING: Etotal drifts significantly in NVT — thermostat is compensating.")
        print("  Run a short NVE segment (nve_energy_conservation_reactive) to confirm.")
        print("  If NVE drift is large: force field error, too-large timestep, or QEq failure.")
        print("  If NVE drift is small: the NVT drift reflects real chemical energy changes.")
    else:
        print("  ✓ No thermostat masking detected.")

    return {"etotal_drift_kcal_mol_atom": float(E_drift_per_atom),
            "thermostat_masking_suspected": masked}
```

### Workflow 6: Validating reactive trajectories against DFT and coupling to active-learning MLP

ReaxFF trajectories must be validated before any chemical conclusion is drawn. The primary validation targets are reaction barriers, species heats of formation, and structural properties of products.

```python
import numpy as np
from ase.io import read, write
from pathlib import Path


def select_structures_for_dft_validation(
    dump_file: str,
    species_df,
    n_before_reaction: int = 5,
    n_after_reaction: int = 5,
    reaction_timestep: int = None,
    dump_every: int = 500,
    output_file: str = "validation_set.xyz",
) -> list:
    """
    Collect structures from before, during, and after a reaction event for
    DFT single-point validation.

    Strategy:
    - Before reaction: structures in the reactant basin (helps validate
      reactant geometry and charge distribution)
    - At reaction: structures near the bond-breaking/forming event (validates
      the transition state region — most critical and most likely to be wrong)
    - After reaction: structures in the product basin (validates product
      stability and geometry)

    These structures can be used for:
    1. DFT single-point calculations to compare ReaxFF energies and forces
    2. NEB calculations to validate reaction barriers
    3. Initial training data for an active-learning MLP covering reactive events

    The DFT single-points are run at the ReaxFF geometry, NOT re-relaxed with DFT.
    This isolates the force-field error from geometry differences.
    """
    frames = read(dump_file, index=":", format="lammps-dump-text")
    frame_steps = [f.info.get("timestep", i * dump_every) for i, f in enumerate(frames)]
    frame_steps = np.array(frame_steps)

    if reaction_timestep is None:
        # Use midpoint of trajectory as default
        reaction_timestep = int(frame_steps[len(frame_steps) // 2])
        print(f"No reaction_timestep specified; using midpoint step {reaction_timestep}.")

    # Find frame index closest to reaction event
    react_idx = int(np.argmin(np.abs(frame_steps - reaction_timestep)))

    # Select frames before and after
    before_indices = range(max(0, react_idx - n_before_reaction), react_idx)
    after_indices = range(react_idx, min(len(frames), react_idx + n_after_reaction + 1))

    selected_frames = [frames[i] for i in before_indices] + [frames[i] for i in after_indices]

    # Tag frames with config_type for DFT labeling and MLP training
    for i, atoms in enumerate(selected_frames):
        phase = "reactant" if i < n_before_reaction else "product"
        if i == n_before_reaction:
            phase = "transition_region"
        atoms.info["config_type"] = f"reaxff_{phase}"
        atoms.info["source"] = f"ReaxFF_step_{frame_steps[list(before_indices) + list(after_indices)][i]}"
        # Remove ReaxFF-specific per-atom data not needed for DFT
        if "charge" in atoms.arrays:
            del atoms.arrays["charge"]

    write(output_file, selected_frames, format="extxyz")
    print(f"Written {len(selected_frames)} structures for DFT validation to {output_file}")
    print(f"  {n_before_reaction} reactant frames + 1 transition-region + "
          f"{n_after_reaction} product frames")
    print("\nNext steps:")
    print("  1. Run DFT single-points with VASP (ISTART=0, NSW=0, IBRION=-1, ISIF=2).")
    print("     See vasp-workflow skill for INCAR/KPOINTS setup.")
    print("  2. Compare DFT energies/forces to ReaxFF values at same geometries.")
    print("  3. If DFT-ReaxFF force MAE < 200 meV/Å: ReaxFF is reasonable for this chemistry.")
    print("  4. If force MAE > 500 meV/Å: ReaxFF reaction path is unreliable;")
    print("     consider training a MACE model on the reactive DFT data.")
    print("  See active-learning-mlp skill for the full active-learning workflow.")

    return selected_frames


def compare_reaxff_to_dft(
    reaxff_dump: str,
    dft_extxyz: str,
    timestep_match: str = "source",
    energy_key: str = "energy",
    forces_key: str = "forces",
) -> dict:
    """
    Compare ReaxFF energies and forces to DFT single-points on the same geometries.

    reaxff_dump: LAMMPS dump file (for reading ReaxFF forces at the geometry)
    dft_extxyz: extxyz file with DFT energy/forces, with 'source' key matching
                the config_type tags written by select_structures_for_dft_validation

    Returns MAE and RMSE for energies and forces, with per-config-type breakdown.
    """
    from ase.calculators.singlepoint import SinglePointCalculator

    dft_frames = read(dft_extxyz, index=":", format="extxyz")
    reaxff_frames = read(reaxff_dump, index=":", format="lammps-dump-text")

    # Match by source tag (set by select_structures_for_dft_validation)
    reaxff_by_source = {f.info.get("source", str(i)): f
                        for i, f in enumerate(reaxff_frames)}

    e_errors = []
    f_maes = []
    config_errors = {}

    for dft_atoms in dft_frames:
        source = dft_atoms.info.get("source", None)
        ctype = dft_atoms.info.get("config_type", "unknown")
        if source not in reaxff_by_source:
            continue

        reaxff_atoms = reaxff_by_source[source]
        natoms = len(dft_atoms)

        dft_e = dft_atoms.info.get(energy_key, None)
        dft_f = dft_atoms.arrays.get(forces_key, None)

        if dft_e is None or dft_f is None:
            continue

        # ReaxFF energy and forces: read from the dump (requires 'fx fy fz' in dump)
        reaxff_e = reaxff_atoms.get_potential_energy() if hasattr(reaxff_atoms, "calc") else None
        reaxff_f = reaxff_atoms.arrays.get("forces", None)

        if reaxff_f is not None and dft_f is not None:
            f_mae = np.abs(reaxff_f - dft_f).mean() * 1000  # meV/Å
            f_maes.append(f_mae)
            if ctype not in config_errors:
                config_errors[ctype] = []
            config_errors[ctype].append(f_mae)

    if not f_maes:
        print("No matching frames found for comparison. Check 'source' key consistency.")
        return {}

    f_mae_global = float(np.mean(f_maes))
    f_rmse_global = float(np.sqrt(np.mean(np.array(f_maes)**2)))

    print(f"\nReaxFF vs. DFT comparison ({len(f_maes)} matched structures):")
    print(f"  Global force MAE:  {f_mae_global:.1f} meV/Å")
    print(f"  Global force RMSE: {f_rmse_global:.1f} meV/Å")

    print(f"\nPer-config-type force MAE:")
    for ctype, errs in config_errors.items():
        print(f"  {ctype:<30}: {np.mean(errs):>7.1f} meV/Å (N={len(errs)})")

    # Interpretation
    if f_mae_global < 200:
        print("\n✓ ReaxFF force MAE < 200 meV/Å — qualitatively reasonable for "
              "this chemistry. Proceed with reactive MD production, but validate "
              "reaction barriers separately with DFT NEB.")
    elif f_mae_global < 500:
        print("\n⚠  Force MAE 200–500 meV/Å — ReaxFF gives qualitative chemistry "
              "but quantitative barriers and rates are unreliable. Consider training "
              "a MACE model on DFT data from these configurations.")
    else:
        print("\n✗ Force MAE > 500 meV/Å — ReaxFF is outside its valid domain for "
              "this chemistry. Do not use for production. Train a reactive MLP using "
              "active-learning-mlp with these DFT-labeled structures as seed data.")

    return {"force_mae_meV_A": f_mae_global, "force_rmse_meV_A": f_rmse_global,
            "per_config_type": {k: float(np.mean(v)) for k, v in config_errors.items()}}
```

## Common Failure Modes

| Failure | Symptom | Likely Cause | Diagnostic | Fix |
|---------|---------|--------------|------------|-----|
| Wrong `units` for ReaxFF | Energy explosion within first 10 steps; unrealistic temperature; total energy off by factor ~4.184 (kcal/mol vs. kJ/mol) or ~23 (kcal/mol vs. eV) | `units metal` used with ReaxFF, which is parameterized in `units real`; timestep is then in ps instead of fs, making effective timestep 1000× too large | Verify: `units real` must appear before `pair_style reaxff`; check log for step 0 PE — for CHO at ~1 g/cm³, expected PE ~ −150 kcal/mol per molecule; if it is −3500 kcal/mol, units are wrong | Change to `units real`; ReaxFF parameter files always use kcal/mol and Å conventions; never use `units metal` with ReaxFF |
| Missing `fix qeq/reaxff` | LAMMPS ERROR: `QEq not defined` or charges remain at 0.0 throughout trajectory; bonding energies are unphysical; species output shows only original element atoms | `pair_style reaxff` does not perform charge equilibration internally; `fix qeq/reaxff` is a separate mandatory fix that must appear after `pair_coeff` | Check input script for `fix qeq/reaxff`; verify LAMMPS log shows `Setting up fix qeq/reaxff`; print per-atom charges from a dump frame — all zeros = QEq is absent | Add `fix qeq_fix all qeq/reaxff 1 0.0 10.0 1e-6 reaxff` immediately after `pair_coeff` |
| QEq cutoff mismatch | QEq non-convergence warnings at most timesteps; charges oscillate wildly between frames; species inventory shows exotic high-charge species | `cuthi` in `fix qeq/reaxff` does not match the non-bond (vdW + Coulomb) cutoff in ffield.reax; QEq computes charges over a different range than the force field | Extract non-bond cutoff from ffield.reax (see Workflow 2); compare to cuthi in `fix qeq/reaxff`; if they differ by > 0.5 Å, this is the cause | Set `fix qeq/reaxff 1 0.0 CUTOFF 1e-6 reaxff` where CUTOFF matches the ffield.reax non-bond cutoff (usually 10.0 Å) |
| Mixed ffield.reax parameter files | Simulation runs but produces reactions at wrong temperatures; species heats of formation off by > 50 kcal/mol; CO2 decomposition at 500 K where it should be stable | Two ffield.reax files from different publications have been merged, or individual element sections have been substituted across files; all ReaxFF parameters are globally coupled | Compare SHA-256 hash of ffield.reax to the primary publication's supplementary; recompute heat of formation for a known molecule (CH4, H2O) and compare to reference; if off by > 30 kcal/mol, file is corrupted | Obtain a fresh ffield.reax from the primary source (original author, OpenKIM, or ReaxFF database); never modify individual parameter sections |
| Timestep too large for reactive MD | Energy explosion or NaN within 1000 steps; alternatively, NVE energy drifts > 5 kcal/mol/atom/ns; high-frequency oscillations in T | dt = 0.5–2 fs used for a system with H atoms; H-O and C-H bond vibrations (~3500 cm⁻¹) require sub-femtosecond resolution in reactive MD; QEq charge updates cannot track bond-order changes faster than dt | Run NVE test (Workflow 5); if drift > 1 kcal/mol/atom/ns at dt = 0.5 fs, reduce to 0.25 fs and repeat | Reduce timestep to 0.1–0.25 fs for H-containing reactive systems; validate with NVE conservation before production; for heavy-atom-only systems, 0.5 fs may be acceptable |
| Overinterpreting bond orders as chemical bonds | Reaction counts in species output include single-timestep appearance of exotic species (CH5, O3, HO3) at low temperature; number of "reactions" is 10× larger than expected | Bond-order threshold (default 0.3) is close to the thermal fluctuation level at the simulation temperature; atoms near the threshold constantly cross it due to vibration without completing a real chemical bond | Apply a persistence filter: species must appear for ≥ 3 consecutive output steps to be counted as real; compare species inventory at BO threshold 0.3 vs. 0.4 — if species counts differ by > 50%, threshold is in the noise | Increase bond-order threshold to 0.4–0.5 for the transient bond type; apply persistence filter (Workflow 4); never report a reaction event from a single-timestep species appearance |
| Thermostat masking force-field errors | NVT simulation appears stable (T stable, no explosion) but product distribution is physically unreasonable; reactions occur at temperatures 500 K below expected activation | The thermostat (especially Berendsen) injects energy at reaction sites faster than the reaction barrier can be assessed; Nosé-Hoover can also mask slow energy injection from bad forces | Run NVE segment from equilibrated NVT structure (Workflow 5); if NVE energy drifts > 2 kcal/mol/atom/ns, thermostat was compensating for force errors | Diagnose and fix force-field problem (timestep, QEq, ffield provenance); switch from Berendsen to Nosé-Hoover or Langevin; validate NVE conservation before concluding that observed reactions are physical |
| Spurious chemistry from unminimized initial geometry | Reactions occur in the first 100 fs of production, producing species not expected at the initial temperature; bond-order analysis shows atoms with unphysical coordination | Initial geometry has atomic overlaps (distances < 1.0 Å) or unreasonably close contacts that produce enormous ReaxFF forces, which immediately cause unphysical bond breaking before thermal equilibration | Minimize energy before any dynamics: `minimize 1e-8 1e-10 10000 100000`; check that minimum energy has converged; verify per-atom charges are physically reasonable after minimization | Always minimize before reactive MD; for molecular systems, also use PACKMOL with minimum distance > 2.0 Å between molecules; inspect the first 1 ps of trajectory for instant anomalous reactions |
| Using ReaxFF outside its parameterization domain | Simulation produces reactions or structural configurations not present in the original validation set (e.g., metallic carbon clusters, Fe-O coordination > 6, exotic radical chains); forces are computed but physically wrong | ReaxFF parameters for element X were fitted to a specific set of chemical environments; applying them to different environments is extrapolation; the energy surface is still continuous but physically incorrect | Compare observed species and coordination numbers in the trajectory to those in the original paper's fitting database; if coordination or stoichiometry is outside the fitting range, the potential is extrapolating | Restrict simulations to chemistry within the parameterization domain; for chemistry outside the domain, train an MLP or use a different ReaxFF parameterization fitted to that chemistry |
| Invalid parameter file encoding | LAMMPS crashes with `ERROR: Cannot open ReaxFF parameter file` or `ERROR: Illegal fix qeq command` | ffield.reax file has Windows line endings (CRLF) on a Linux cluster; or file was corrupted during transfer; or comment character in wrong column | Run `file ffield.reax`; if it shows "CRLF line terminators", convert with `dos2unix ffield.reax`; compare file size to original | Convert line endings: `dos2unix ffield.reax`; verify SHA-256 matches original |
| Production run without NVE drift check | Reactive MD completes without error, but postprocessing reveals wrong product yields, unphysical temperature dependence, or non-reproducible results between independent runs | No NVE validation was performed; the simulation may have had a timestep too large, poor QEq convergence, or thermostat masking that were never detected | Run NVE stability check (Workflow 5) retroactively; compare independent runs with different random seeds — if species yields differ by > 30%, the simulation is not converged | Re-run with NVE validation before production; if NVE fails, diagnose and fix before committing to production; report NVE drift alongside all reactive MD results |

## Best Practices

- **Validate every ffield.reax parameter file before production.** At minimum, compute the heat of formation or bond dissociation energy for one or two key molecules in your system (e.g., for CHO: the C-H bond in methane, the O-H bond in water, the C=O bond in CO₂) and compare to the values quoted in the original publication. A > 10 kcal/mol discrepancy indicates a file version mismatch.
- **Always record the SHA-256 hash and primary DOI for every ffield.reax file used.** `sha256sum ffield.reax > ffield.reax.sha256`. These files are not versioned by filename; the hash is the only reliable identifier.
- **Run NVE energy conservation before every production run.** This is the only reliable test for timestep adequacy and QEq convergence in reactive MD. A drift > 1 kcal/mol/atom/ns in 5000 steps × 0.25 fs = 1.25 ps is a contraindication for production.
- **Use a bond-order threshold that matches the chemistry being tracked.** The default 0.3 threshold in `fix reaxff/species` may be too low for high-temperature simulations. Test threshold sensitivity: compare species inventories at 0.3, 0.4, and 0.5, and choose the value that gives chemically sensible starting-material identification at t=0. Report this threshold in all publications.
- **Apply a persistence filter to reaction events.** Species appearing for only 1–2 output frames at the bond-order detection threshold are thermal fluctuations, not reactions. Require ≥ 3 consecutive appearances before counting a species as formed.
- **Never use NPT during a rapidly reacting system.** Exothermic reactions locally increase temperature and density; the barostat responds by expanding the box, which can couple mechanically to subsequent reactions in physically unrealistic ways. Use NVT for equilibration and NVE or NVT for production of reactive trajectories.
- **Use Langevin thermostat for reactive production when local heat dissipation matters.** Langevin applies per-atom damping, which dissipates local reaction heat more physically than the global Nosé-Hoover thermostat. For combustion and oxidation simulations where exothermic events are localized, Langevin better approximates heat conduction to a thermal reservoir. [EXPERT REVIEW NEEDED]
- **Validate against DFT before interpreting any chemistry.** Compare ReaxFF energies and forces on 10–20 reactive configurations to DFT single-points (see Workflow 6). A force MAE > 500 meV/Å between ReaxFF and DFT at the reactive geometries indicates the force field is wrong for this chemistry, regardless of how plausible the trajectory looks.
- **Do not use ReaxFF rates for quantitative kinetics without additional validation.** ReaxFF reaction barriers are typically 20–50% in error relative to high-level quantum chemistry. Use DFT NEB or DFT-MD metadynamics for quantitative barriers. ReaxFF can correctly identify which reactions occur and their relative rates at a given temperature, but not their absolute magnitudes.
- **For active learning: use ReaxFF as a cheap explorer, not a source of truth.** Run ReaxFF MD to identify structurally diverse reactive configurations, then label those configurations with DFT and train a MACE or NequIP model. The MLP will be more accurate than ReaxFF while also being transferable. This workflow costs far less DFT time than blind DFT-MD.
- **Check that COMB and ReaxFF are not confused.** Both use charge equilibration and are deployed in LAMMPS with separate `fix qeq` commands, but they use different `pair_style` commands, different parameter file formats, and different `fix qeq` variants (`fix qeq/reaxff` vs. `fix qeq/comb`). Mixing them produces LAMMPS errors or, worse, silently wrong simulations.

## Integration with Other Skills

- **[lammps-workflow](../lammps-workflow/SKILL.md):** The LAMMPS input script mechanics — `units`, `atom_style`, `pair_style`, neighbor lists, `fix nvt/nve`, dump, restart, and parallel execution — are covered in depth in that skill. This skill covers only the reactive-specific commands (`pair_style reaxff`, `fix qeq/reaxff`, `fix reaxff/species`) and the reactive-specific validation protocol. Use both skills together for a production-ready reactive MD setup.
- **[md-force-fields](../md-force-fields/SKILL.md):** That skill covers the physical basis and selection logic for all major force-field functional forms including ReaxFF and COMB. This skill provides the reactive-specific deployment and validation details. The force-field selection decision (whether to use ReaxFF, COMB, or an ML potential for a given system) is made in that skill; the reactive MD protocol once that decision is made is this skill.
- **[md-equilibration](../md-equilibration/SKILL.md):** Equilibration for reactive MD requires the same NVE stability test, thermostat selection, and convergence criteria as classical MD, plus the additional reactive-specific checks on bond-order distribution and species inventory. That skill's NVE energy drift check is the primary diagnostic for reactive MD stability; the reactive-specific interpretation of drift (reaction heat vs. force-field error) is covered here.
- **[md-analysis](../md-analysis/SKILL.md):** RDF, coordination numbers, MSD, and temperature/pressure time-series analysis from that skill apply to the non-reactive epochs of a reactive trajectory. Bond-order and species-count analysis covered here is the additional reactive-specific layer. That skill's MDAnalysis and OVITO workflows apply to reactive MD trajectories after species identification.
- **[gromacs-workflow](../gromacs-workflow/SKILL.md):** GROMACS does not support ReaxFF or any reactive force field. Reactive MD must be performed in LAMMPS. If the broader simulation project uses GROMACS for non-reactive subsystems (e.g., a protein solvated in water with a reactive metal cofactor), the reactive region must be handled in LAMMPS with a separate simulation or via a QM/MM interface. This skill does not cover QM/MM coupling.
- **[active-learning-mlp](../../ml-interatomic-potentials/active-learning-mlp/SKILL.md):** The primary downstream use of reactive MD in modern computational chemistry is as a cheap explorer for active-learning MLP training. Workflow 4 and 6 here implement the coupling: ReaxFF identifies reactive configurations; DFT labels them; the active-learning loop trains an MLP. That skill covers the full active-learning protocol from uncertainty estimation to DFT query strategy to training iteration.
- **[mlp-validation](../../ml-interatomic-potentials/mlp-validation/SKILL.md):** When a reactive MLP is trained on DFT data from reactive configurations identified by ReaxFF, its validation must include the NVE stability test at the target temperature, RDF comparison to DFT-MD, and force MAE decomposed by config_type (reactant, product, transition region). That skill's validation protocols apply directly; this skill's reactive-specific trajectory contexts define which config_types are most important.
- **[mlp-to-lammps](../../ml-interatomic-potentials/mlp-to-lammps/SKILL.md):** A reactive MLP trained via active learning can replace ReaxFF in LAMMPS using `pair_style mace` or `pair_style nequip`. The element ordering, `no_domain_decomposition`, and unit conventions for MLP deployment in LAMMPS are covered in that skill. The transition from ReaxFF to MLP production runs follows directly: validate ASE vs. LAMMPS consistency, then substitute `pair_style mace` for `pair_style reaxff` with appropriate timestep (typically 2 fs for MLP vs. 0.25 fs for ReaxFF).
- **[vasp-workflow](../../electronic-structure/vasp-workflow/SKILL.md):** DFT single-point calculations for ReaxFF-vs-DFT validation (Workflow 6) and for active-learning labeling are run with VASP. The INCAR settings for single-point force calculations (ISTART=0, NSW=0, IBRION=-1, ISIF=2, LWAVE=.FALSE.) and the stress sign convention (VASP outputs compression-positive; ASE expects tensile-positive) from that skill must be applied correctly when building the validation dataset.
- **[dft-convergence](../../electronic-structure/dft-convergence/SKILL.md):** DFT single-points used for ReaxFF validation or MLP training on reactive structures must be computed at converged settings (ENCUT, k-points, PREC). Unconverged DFT produces noisy energies and forces that inflate apparent ReaxFF errors and degrade MLP training. That skill provides the convergence protocol; this skill provides the structures to be evaluated.
- **[dft-dataset-generation](../../electronic-structure/dft-dataset-generation/SKILL.md):** When reactive MD configurations are identified for DFT labeling, the DFT dataset construction workflow from that skill (extxyz format, config_type tagging, energy/force/stress consistency checks, and handling of spin-polarized reactive systems) applies directly. The structures extracted in Workflow 4 and 6 here feed into that skill's pipeline.
- **[reproducibility-workflow](../../scientific-data/reproducibility-workflow/SKILL.md):** Reactive MD trajectories require careful provenance tracking: ffield.reax SHA-256, LAMMPS version, QEq settings, bond-order threshold used for species analysis, and NVE drift measured before production. That skill's DVC-based data versioning and provenance record protocol should be applied to every reactive MD project. Trajectories from different ffield.reax versions are not reproducibly comparable without this record.

## Key References

1. van Duin, A. C. T., et al. (2001). *ReaxFF: A reactive force field for hydrocarbons.* Journal of Physical Chemistry A, 105(41), 9396–9409. https://doi.org/10.1021/jp004368u
2. Chenoweth, K., van Duin, A. C. T., & Goddard, W. A. (2008). *ReaxFF reactive force field for molecular dynamics simulations of hydrocarbon oxidation.* Journal of Physical Chemistry A, 112(5), 1040–1053. https://doi.org/10.1021/jp709896w
3. Senftle, T. P., et al. (2016). *The ReaxFF reactive force-field: Development, applications and future directions.* npj Computational Materials, 2, 15011. https://doi.org/10.1038/npjcompumats.2015.11
4. Aktulga, H. M., et al. (2012). *Parallel reactive molecular dynamics: Numerical methods and algorithmic techniques.* Parallel Computing, 38(4–5), 245–259. https://doi.org/10.1016/j.parco.2011.08.005
5. Rappé, A. K., & Goddard, W. A. (1991). *Charge equilibration for molecular dynamics simulations.* Journal of Physical Chemistry, 95(8), 3358–3363. https://doi.org/10.1021/j100161a070
6. Liang, T., et al. (2012). *Classical atomistic simulations of surfaces and heterogeneous interfaces with the charge-optimized many body (COMB) potentials.* Materials Science and Engineering: R: Reports, 74(9), 255–279. https://doi.org/10.1016/j.mser.2013.07.001
7. Aryanpour, M., van Duin, A. C. T., & Kubicki, J. D. (2010). *Development of a reactive force field for iron–oxyhydroxide systems.* Journal of Physical Chemistry A, 114(21), 6298–6307. https://doi.org/10.1021/jp101332k
8. Rahaman, O., et al. (2011). *Development of a ReaxFF reactive force field for glycine and application to solvent effect and tautomerization.* Journal of Physical Chemistry B, 115(2), 249–261. https://doi.org/10.1021/jp108642r
9. Islam, M. M., et al. (2015). *ReaxFF molecular dynamics simulations on lithiated sulfur cathode materials.* Physical Chemistry Chemical Physics, 17(5), 3383–3393. https://doi.org/10.1039/C4CP04532G
10. Thompson, A. P., et al. (2022). *LAMMPS — A flexible simulation tool for particle-based materials modeling at the atomic, meso, and continuum scales.* Computer Physics Communications, 271, 108171. https://doi.org/10.1016/j.cpc.2021.108171
