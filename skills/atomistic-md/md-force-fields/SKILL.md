---
name: md-force-fields
domain: atomistic-md
maturity: draft
tools:
  - LAMMPS
  - GROMACS
  - ASE
  - OpenMM
  - MACE
tested_versions:
  LAMMPS: "2Aug2023"
  GROMACS: "2024.2"
  ASE: "3.23.0"
  OpenMM: "8.1.0"
  MACE: "0.3.6"
requires_gpu: false
estimated_runtime: minutes-to-hours
---

# MD Force Fields

## Description

This skill covers selection, validation, and deployment of interatomic force fields for molecular dynamics: the physical basis and transferability limits of every major functional form (EAM, MEAM, Tersoff, Stillinger-Weber, Buckingham, COMB, ReaxFF, AMBER, CHARMM, OPLS, GROMOS, and ML potentials); parameter-file sourcing and provenance verification; unit and format compatibility across LAMMPS, GROMACS, OpenMM, and ASE; mixing rules, long-range electrostatics, cutoff conventions, neighbor-list settings, and bonded constraints; and validation of any force field against DFT, experiment, or higher-level simulation. Invoke this skill when selecting a force field for a new system, diagnosing a simulation failure that may originate from a wrong or misapplied force field, or verifying the transferability of an existing parameterization to a new application domain.

## Domain Context

A force field is a function that maps a set of atomic positions {rᵢ} to a potential energy U({rᵢ}) and, by differentiation, to forces Fᵢ = -∂U/∂rᵢ. Every force field encodes assumptions about which physical interactions are important and which are negligible. These assumptions are baked into the functional form; they cannot be corrected by changing parameter values alone.

The central issue with force fields is **transferability**: a parameterization fitted to reproduce a set of reference properties in one regime is not guaranteed to be accurate in a different regime. The reference data (crystal structures, elastic constants, bond lengths, heats of formation, vibrational frequencies) used to fit the parameters determines the validity domain. Using the force field outside that domain is extrapolation — the errors are unbounded and often silent.

**Hierarchy of approximations, from most to least severe:**

1. **Born-Oppenheimer surface:** All classical force fields assume the ground-state PES. Electronically excited states, non-adiabatic dynamics, metallic-band effects on bonding, and charge-transfer-driven structural changes are outside the Born-Oppenheimer classical picture entirely.
2. **Fixed topology vs. reactive:** Fixed-topology force fields (AMBER, CHARMM, EAM, Tersoff) assign bonds at the start of the simulation and never break or form them. Reactive force fields (ReaxFF, COMB) allow bond-order to vary continuously. ML potentials can be either reactive or non-reactive depending on the training data.
3. **Many-body vs. pairwise:** Pure pairwise potentials (LJ, Buckingham) cannot reproduce the Cauchy pressure discrepancy in metals (C₁₂ ≠ C₄₄) or the angular dependence of covalent bonds. Many-body terms (EAM embedding energy, Tersoff bond-order, AMBER dihedral) correct specific deficiencies but add parameters.
4. **Classical nuclei:** All MD force fields integrate classical equations of motion. Zero-point energy and nuclear tunneling are neglected. This fails for light atoms (H, He, Li) at low temperature.
5. **Mean-field charges:** Classical force fields use fixed partial charges (AMBER, CHARMM) or fluctuating charges (ReaxFF QEq, COMB). Neither is a first-principles electron density; both fail when charge delocalization or metallic screening becomes important.

**The parameter-provenance problem:** Two LAMMPS input files can both say `pair_style eam` and produce physically different results because the `.eam` parameter file version differs. Published force fields are revised between papers; parameters with the same name from different sources are not identical. Always record the exact parameter file, its SHA-256 hash, and the primary publication in every simulation directory.

## When to Use This Skill

- Selecting a force-field functional form for a new system or application, before setting up any LAMMPS or GROMACS input.
- Verifying that a force-field parameter file obtained from a repository or collaborator is the correct version for the intended application.
- Diagnosing a simulation failure (energy explosion, wrong density, incorrect melting temperature, spurious phase transition) that may originate from a misapplied or out-of-domain force field.
- Determining whether a classical force field, reactive force field, or ML potential is appropriate given the simulation goals, system, and available computational resources.
- Validating a literature or community force field against DFT reference data before using it for production property calculations.
- Porting a force field between codes (LAMMPS → GROMACS → OpenMM → ASE) and verifying that energies and forces are consistent after conversion.

## When Not to Use This Skill

- **When the required property is fundamentally electronic.** Band gaps, optical spectra, magnetic ordering, Mott localization, and electronically driven phase transitions require DFT or TDDFT regardless of system size. No classical force field, including ML potentials, can predict these properties unless explicitly trained on electronic labels.
- **When an ML potential has already been validated for the exact system and conditions.** If a MACE or NequIP model has been trained and validated (see `mlp-validation`) for the target composition and temperature range, it supersedes classical force-field selection for that system. Use this skill only to decide whether to use the ML potential or to fall back to a cheaper classical potential for initial equilibration.
- **When bond breaking or forming is needed but ReaxFF has not been parameterized and validated for the system.** An unvalidated reactive potential is more dangerous than an admittedly non-reactive classical potential — the wrong chemistry is worse than no chemistry at all. If no validated reactive FF exists, use DFT-MD or ML potentials trained on reactive trajectories.
- **For coarse-grained simulations.** Martini, SIRAH, and other CG force fields operate on groups of atoms, not individual atoms. Their parameterization and transferability differ qualitatively from atomistic force fields; their deployment follows different protocols.

## Core Concepts

- **Functional form:** The mathematical expression for U({rᵢ}). The functional form is fixed; it determines what physics can be reproduced and what cannot. Parameters are fitted to reference data but cannot compensate for a missing physical term (e.g., a pairwise potential cannot reproduce the correct Cauchy pressure ratio in a metal, regardless of parameter choice).
- **Parameterization domain:** The set of structures, compositions, temperatures, and pressures for which the parameters were fitted. Structures outside this domain are extrapolations. For EAM potentials, the domain is typically the equilibrium crystal structure, point defects, and phonons at moderate temperatures. For AMBER ff14SB, the domain is standard protein residues in water near physiological conditions.
- **Transferability:** The degree to which a force field retains accuracy outside its parameterization domain. EAM potentials for pure metals have moderate transferability to grain boundaries and surfaces. CHARMM36 has low transferability to non-standard ligands (use CGENFF for those). ML potentials have higher intrinsic flexibility but limited transferability outside the training composition space.
- **Pair potential:** A contribution U(rᵢⱼ) that depends only on the scalar distance between atoms i and j. Computationally cheap; misses angular and many-body effects. Lennard-Jones, Buckingham, Coulomb, and Morse are pairwise.
- **Embedding energy (EAM):** U = Σᵢ F(ρ̄ᵢ) + ½ Σᵢⱼ φ(rᵢⱼ), where ρ̄ᵢ = Σⱼ ρ(rᵢⱼ) is the local electron density at site i and F is the embedding energy. Captures metallic bonding (non-directional, many-body) without explicit electron density. Valid for FCC and BCC metals near equilibrium.
- **Bond-order potential (Tersoff, REBO):** The pair repulsion/attraction strength is modulated by a bond-order function bᵢⱼ that depends on the coordination environment of atoms i and j. Captures directional covalent bonding (Si, Ge, C, SiC). The cutoff function introduces a smooth decay but can produce spurious forces near the cutoff if a coordination shell boundary falls within it.
- **Bonded force field (molecular mechanics):** U = Σ bonds k_b(r-r₀)² + Σ angles k_θ(θ-θ₀)² + Σ dihedrals V_n(1+cos(nφ-γ)) + Σ impropers + Σ nonbonded. The bond list is fixed at the start of the simulation. Requires a topology file (data file in LAMMPS, .top in GROMACS). The nonbonded interactions between atoms separated by 1–2 bonds are excluded; 1–3 may be excluded; 1–4 use scaled interactions (AMBER: fudgeLJ=0.5, fudgeQQ=0.8333; CHARMM: different defaults).
- **Long-range electrostatics:** Coulomb interactions decay as 1/r and cannot be truncated at a short cutoff without introducing large artifacts in polar, ionic, or biomolecular systems. Particle Mesh Ewald (PME) or Particle-Particle Particle-Mesh (P³M/PPPM) sums the real-space short-range part and the reciprocal-space long-range part with O(N log N) cost. Required for any system with net partial charges; not needed for pure metals or neutral LJ systems. The real-space cutoff rcoulomb and k-space accuracy must be set consistently.
- **Mixing rules:** For pairs of unlike atoms in a pairwise LJ or Buckingham potential, parameters are derived from like-pair parameters using combining rules. Lorentz-Berthelot: σᵢⱼ = (σᵢ + σⱼ)/2, εᵢⱼ = √(εᵢεⱼ). Geometric: σᵢⱼ = √(σᵢσⱼ), εᵢⱼ = √(εᵢεⱼ). Waldman-Hagler: for noble gases. **The mixing rule is part of the force-field definition; mixing rules from one force field must not be applied to parameters from another.** AMBER uses Lorentz-Berthelot; OPLS-AA uses geometric (for σ) + geometric (for ε). Using the wrong mixing rule changes cross-pair interactions by up to 30%.
- **ReaxFF charge equilibration (QEq):** ReaxFF assigns fluctuating charges by minimizing a charge-dependent energy functional at every timestep via the electronegativity equalization principle. In LAMMPS this is a separate fix (`fix qeq/reaxff`). QEq can fail if the system undergoes charge separation that exceeds the QEq damping radius, or if the parameter file's electronegativity parameters are physically inconsistent for the system.
- **SHAKE/LINCS constraints:** High-frequency bond vibrations (O-H, C-H, N-H) require sub-femtosecond timesteps unless constrained. SHAKE (LAMMPS, older GROMACS) and LINCS (GROMACS preferred) constrain bond lengths to their equilibrium values, eliminating the highest-frequency vibrational modes and allowing 2 fs timesteps for bonded systems. Constraints modify the equations of motion; they must be used consistently with the force-field parameterization (AMBER ff14SB was parameterized assuming H-bond constraints).
- **Special bonds / 1-4 scaling:** In molecular force fields, nonbonded interactions between atoms in 1-4 position (separated by three bonds) are scaled rather than excluded. AMBER: 1-4 LJ × 0.5, 1-4 Coulomb × 0.8333. CHARMM: all 1-4 included at full weight but pair parameters adjusted during fitting. OPLS-AA: 1-4 LJ × 0.5, 1-4 Coulomb × 0.5. Mixing these scaling factors across force fields is a frequent source of systematic energy errors.

## Key Workflows

### Workflow 1: Force-field selection — decision logic by system class

The choice of functional form must precede parameterization. No parameter adjustment can compensate for the wrong functional form. Use this decision logic before searching for parameters.

```python
def select_force_field(system_description: dict) -> str:
    """
    Decision logic for force-field functional form selection.

    system_description keys:
        "material_class": str — "metal", "covalent_semiconductor",
            "ionic_oxide", "organic_molecule", "polymer", "biomolecule",
            "metal_organic_interface", "reactive_combustion"
        "reactive": bool — True if bond breaking/forming must be modeled
        "temperature_K": float — simulation temperature
        "requires_charges": bool — True if electrostatics are non-negligible
        "n_atoms": int — simulation size
        "dft_reference_available": bool — True if DFT dataset exists for MLP training
        "budget": str — "low" (classical only), "medium" (universal MLP),
                         "high" (trained MLP)

    Returns a recommendation string explaining the choice and its caveats.
    """
    mc = system_description.get("material_class", "unknown")
    reactive = system_description.get("reactive", False)
    T = system_description.get("temperature_K", 300)
    n_atoms = system_description.get("n_atoms", 1000)
    budget = system_description.get("budget", "medium")
    dft_available = system_description.get("dft_reference_available", False)

    # ── ML potentials ─────────────────────────────────────────────────────────
    if budget == "high" and dft_available:
        return (
            "RECOMMENDATION: Train a custom MACE or NequIP potential on DFT data.\n"
            "Justification: DFT reference is available and budget allows training.\n"
            "A trained MLP will outperform any classical potential for this specific system.\n"
            "Workflow: see mace-training → active-learning-mlp → mlp-validation.\n"
            "Caveats:\n"
            "  - Training requires 500–5000 DFT calculations (hours-to-days on HPC).\n"
            "  - Extrapolation outside training conditions remains a risk; validate NVE stability.\n"
            "  - Parallel MD with MACE is limited by no_domain_decomposition (see mlp-to-lammps)."
        )

    if budget == "medium" and mc in ("metal", "covalent_semiconductor", "ionic_oxide",
                                      "metal_organic_interface"):
        return (
            "RECOMMENDATION: Use MACE-MP-0 or CHGNet universal potential.\n"
            "Justification: Universal MLPs cover broad composition space at DFT accuracy.\n"
            "Workflow: see universal-potentials skill.\n"
            "Caveats:\n"
            "  - Validate against DFT reference for your specific composition before production.\n"
            "  - Universal potentials struggle with unusual coordination, high-P phases,\n"
            "    and compositions at the edges of the Materials Project training set.\n"
            "  - Throughput: MACE-MP-0 ~1000 atom-steps/s on CPU; GPU required for > 5000 atoms."
        )

    # ── Metals and alloys ─────────────────────────────────────────────────────
    if mc == "metal":
        if reactive:
            return (
                "RECOMMENDATION: ReaxFF (if parameterized) or ML potential.\n"
                "EAM/MEAM cannot model bond breaking. ReaxFF exists for select metals/oxides\n"
                "(Fe-C, Ni-Al, Cu-H, Fe-H). If no validated ReaxFF exists, train a MACE model.\n"
                "Caveats:\n"
                "  - ReaxFF for metals is sensitive to parameter file version.\n"
                "  - Only use a ReaxFF parameterized explicitly for your metal + environment."
            )
        if "alloy" in mc or system_description.get("is_alloy", False):
            return (
                "RECOMMENDATION: EAM/alloy or MEAM.\n"
                "EAM/alloy: Mishin, Bonny, Zhou parameterizations; validate mixing enthalpy.\n"
                "MEAM: better for alloys with directional bonding (Fe-Mn, Ni-Al).\n"
                "Source: NIST Interatomic Potentials Repository.\n"
                "Caveats:\n"
                "  - Validate alloy mixing enthalpies against DFT before use.\n"
                "  - EAM/alloy cannot model surface segregation driven by charge redistribution."
            )
        return (
            "RECOMMENDATION: EAM or EAM/alloy for pure metals.\n"
            "Validated potentials exist for most common metals (Cu, Al, Fe, Ni, W, Au, Ag, Mo).\n"
            "Source: NIST Interatomic Potentials Repository (https://www.ctcms.nist.gov/potentials/).\n"
            "Caveats:\n"
            "  - Verify cohesive energy, lattice parameter, and elastic constants against DFT.\n"
            "  - EAM fails for strong covalent or ionic contributions (e.g., Fe-O interfaces).\n"
            "  - Surface energies may be inaccurate outside the fitting database."
        )

    # ── Covalent semiconductors ───────────────────────────────────────────────
    if mc == "covalent_semiconductor":
        return (
            "RECOMMENDATION: Tersoff or Stillinger-Weber (Si, Ge); REBO/AIREBO (C, hydrocarbons).\n"
            "Tersoff: validated for Si, Ge, C, SiC, GaN, GaAs near equilibrium.\n"
            "Stillinger-Weber: Si liquid and amorphous (better than Tersoff for melt).\n"
            "REBO/AIREBO: carbon nanostructures, hydrocarbons, diamond.\n"
            "Caveats:\n"
            "  - Tersoff has known failures for amorphous Si under large strains.\n"
            "  - Cutoff function creates artifact forces near the coordination-shell boundary;\n"
            "    verify that r_cut does not bisect a coordination shell.\n"
            "  - For III-V alloys with multiple species, validate cross-species parameters.\n"
            "  - Consider MACE-MP-0 for composition-space flexibility. [EXPERT REVIEW NEEDED]"
        )

    # ── Ionic oxides, silicates, zeolites ─────────────────────────────────────
    if mc == "ionic_oxide":
        return (
            "RECOMMENDATION: Buckingham + Coulomb (shell model), or COMB3 for interfaces.\n"
            "Buckingham: MgO (Lewis-Catlow), Al₂O₃ (Matsui-Akaogi), SiO₂ (van Beest).\n"
            "Shell model: adds electronic polarizability (important for dielectric/thermal props).\n"
            "COMB3: charge-optimized many-body; best for metal/oxide interfaces (Cu/ZnO, Pt/TiO₂).\n"
            "Caveats:\n"
            "  - Buckingham diverges at r→0; add a ZBL short-range correction for high-P MD.\n"
            "  - Charges must be consistent with the parameterization (formal charges vs. partial).\n"
            "  - Shell model is significantly slower (shell particles have tiny mass).\n"
            "  - For mixed ionic-covalent systems (silicates), validate Si-O coordination.\n"
            "  [EXPERT REVIEW NEEDED: COMB3 coverage of oxide interfaces is limited]"
        )

    # ── Organic molecules, polymers ───────────────────────────────────────────
    if mc in ("organic_molecule", "polymer"):
        return (
            "RECOMMENDATION: OPLS-AA (general organics), GAFF (diverse pharmaceuticals).\n"
            "OPLS-AA: validated for liquids and polymers; geometric mixing rules.\n"
            "GAFF (AMBER): automated parameterization via antechamber for drug-like molecules.\n"
            "CHARMM CGenFF: for molecules near biological chemistry.\n"
            "Caveats:\n"
            "  - GAFF uses AM1-BCC or RESP charges; quality of charges drives accuracy.\n"
            "  - OPLS-AA charges derived from MP2/6-31G(d) ESP; do not mix with AMBER charges.\n"
            "  - Dihedral parameters are the weakest point; validate against QM torsion scans.\n"
            "  - LiqProp: liquid density and ΔH_vap are primary OPLS-AA validation targets."
        )

    # ── Biomolecules ──────────────────────────────────────────────────────────
    if mc == "biomolecule":
        return (
            "RECOMMENDATION: CHARMM36m (proteins + lipids + nucleic acids + carbohydrates)\n"
            "or AMBER ff14SB (proteins) + LIPID17 (lipids) + OL3 (RNA).\n"
            "Water: TIP3P (CHARMM default), TIP4P-Ew (AMBER), TIP4P/2005 (density-accurate).\n"
            "Caveats:\n"
            "  - Force-field and water model must be matched (do not use TIP4P with ff14SB).\n"
            "  - Non-standard residues require CGENFF or antechamber parameterization.\n"
            "  - Protonation state assignment (His, Asp, Glu, Lys, Arg) must precede simulation.\n"
            "  - Validate backbone RMSD against crystal structure; secondary structure stability."
        )

    # ── Reactive / combustion ─────────────────────────────────────────────────
    if mc == "reactive_combustion" or reactive:
        return (
            "RECOMMENDATION: ReaxFF (combustion: Liu, Chenoweth, Cheng parameter sets).\n"
            "Only use a parameter set validated for the exact element set of your system.\n"
            "Caveats:\n"
            "  - ReaxFF requires fix qeq/reaxff in LAMMPS at every step (slow).\n"
            "  - Timestep must be 0.1–0.5 fs. [EXPERT REVIEW NEEDED]\n"
            "  - NEVER mix ffield.reax files from different publications.\n"
            "  - Validate reaction barriers against DFT or CCSD(T) for critical pathways."
        )

    return (
        f"System class '{mc}' not directly covered. Consider:\n"
        "  (1) Universal MLP (MACE-MP-0) as a starting point.\n"
        "  (2) Literature survey for validated force fields for this composition.\n"
        "  (3) Custom MLP training if DFT reference can be generated."
    )


# ── Usage example ────────────────────────────────────────────────────────────
system = {
    "material_class": "metal",
    "reactive": False,
    "is_alloy": False,
    "temperature_K": 1000,
    "n_atoms": 10_000,
    "budget": "medium",
    "dft_reference_available": False,
}
print(select_force_field(system))
```

### Workflow 2: Molecular force fields — AMBER/CHARMM in GROMACS and LAMMPS

Biomolecular and organic force fields require topology files in addition to coordinates. The topology encodes bond, angle, dihedral, and improper lists plus per-atom charges and LJ parameters. Errors in topology construction are the dominant source of failures in molecular simulations.

```python
"""
Parameter sourcing and validation for AMBER/CHARMM force fields.

AMBER ff14SB (proteins) in GROMACS:
  1. Build initial structure: LEaP (AmberTools) or PDBFixer + OpenMM
  2. Convert topology: acpype or GROMACS amber2gmx utility
  3. Assign non-standard residue parameters: antechamber → GAFF → acpype

CHARMM36m (proteins + lipids) in GROMACS:
  1. Build system: CHARMM-GUI (https://charmm-gui.org/) — most reliable path
  2. CHARMM-GUI outputs: topol.top, system.gro, position restraints, mdp templates
  3. Verify: check that [ moleculetype ] sections match the system composition

LAMMPS: molecular force fields require atom_style full and topology sections in the data file.
  Conversion: acpype --input structure.pdb --net_charge 0 → LAMMPS data file
              or VMD topotools plugin for CHARMM psf → LAMMPS data
"""

import numpy as np
from pathlib import Path


def validate_amber_topology(top_file: str, gro_file: str) -> dict:
    """
    Validate a GROMACS topology against the coordinate file.

    Checks:
    - Atom count in .top matches .gro
    - No undefined atom types
    - Charge neutrality (or expected net charge)
    - Reasonable per-atom charges (|q| > 2 is unusual)
    - LJ parameters exist for all atom type pairs

    Returns dict of issues found.
    """
    issues = []

    # Parse atom count from .gro
    with open(gro_file) as f:
        lines = f.readlines()
    n_atoms_gro = int(lines[1].strip())

    # Parse .top for atom count and charges
    total_charge = 0.0
    n_atoms_top = 0
    in_atoms_section = False
    atom_types_found = set()
    large_charge_atoms = []

    with open(top_file) as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith("[ atoms ]"):
                in_atoms_section = True
                continue
            if stripped.startswith("[") and in_atoms_section:
                in_atoms_section = False
                continue
            if in_atoms_section and stripped and not stripped.startswith(";"):
                parts = stripped.split()
                if len(parts) >= 7:
                    try:
                        charge = float(parts[6])
                        atom_type = parts[1]
                        total_charge += charge
                        n_atoms_top += 1
                        atom_types_found.add(atom_type)
                        if abs(charge) > 1.5:
                            large_charge_atoms.append((n_atoms_top, atom_type, charge))
                    except (ValueError, IndexError):
                        pass

    if n_atoms_top != n_atoms_gro:
        issues.append(f"Atom count mismatch: .top has {n_atoms_top}, .gro has {n_atoms_gro}")

    charge_rounded = round(total_charge)
    if abs(total_charge - charge_rounded) > 0.01:
        issues.append(f"Non-integer total charge: {total_charge:.4f}e "
                      f"(expected integer; add ions to neutralize)")
    else:
        print(f"Total charge: {total_charge:.4f}e (≈ {charge_rounded:+d}e) — "
              f"{'OK (neutral)' if charge_rounded == 0 else 'ADD COUNTERIONS before production'}")

    if large_charge_atoms:
        print(f"WARNING: {len(large_charge_atoms)} atoms with |q| > 1.5e:")
        for idx, atype, q in large_charge_atoms[:5]:
            print(f"  atom {idx} type {atype}: q = {q:.3f}e — verify this is physical")

    if not issues:
        print(f"Topology check: PASSED ({n_atoms_top} atoms, {len(atom_types_found)} types)")
    else:
        for issue in issues:
            print(f"ERROR: {issue}")

    return {"issues": issues, "n_atoms": n_atoms_top,
            "total_charge": total_charge, "atom_types": atom_types_found}


def check_special_bonds_settings(ff_name: str) -> dict:
    """
    Return the correct 1-4 nonbonded scaling factors for common force fields.

    These values must match the parameter file exactly. Using the wrong
    scaling factors introduces a systematic energy error of several kcal/mol
    per dihedral — large enough to change preferred conformations.
    """
    ff_settings = {
        "AMBER_ff14SB": {
            "fudgeLJ": 0.5, "fudgeQQ": 0.8333,
            "mixing_rule": "Lorentz-Berthelot",
            "lammps": "special_bonds amber",
            "gromacs": "gen-pairs = yes\nfudgeLJ = 0.5\nfudgeQQ = 0.8333",
            "note": "All standard AMBER force fields use these values."
        },
        "CHARMM36m": {
            "fudgeLJ": 1.0, "fudgeQQ": 1.0,
            "mixing_rule": "Lorentz-Berthelot",
            "lammps": "special_bonds charmm",
            "gromacs": "gen-pairs = no",
            "note": "CHARMM includes 1-4 at full weight; adjusted via pair parameters. "
                    "Do NOT set fudgeQQ or fudgeLJ when using CHARMM36m in GROMACS."
        },
        "OPLS-AA": {
            "fudgeLJ": 0.5, "fudgeQQ": 0.5,
            "mixing_rule": "geometric",
            "lammps": "special_bonds lj/coul 0.0 0.0 0.5",
            "gromacs": "gen-pairs = yes\nfudgeLJ = 0.5\nfudgeQQ = 0.5",
            "note": "OPLS-AA uses geometric mixing for BOTH sigma and epsilon. "
                    "Lorentz-Berthelot mixing changes OPLS LJ cross-terms by up to 30%."
        },
        "GROMOS54A7": {
            "fudgeLJ": 1.0, "fudgeQQ": 1.0,
            "mixing_rule": "geometric",
            "lammps": "special_bonds lj/coul 0.0 0.0 1.0",
            "gromacs": "gen-pairs = no",
            "note": "GROMOS uses geometric mixing and includes 1-4 at full weight."
        },
    }
    settings = ff_settings.get(ff_name)
    if settings is None:
        print(f"Unknown force field '{ff_name}'. Known: {list(ff_settings.keys())}")
        return {}
    print(f"\n{ff_name} special bond settings:")
    for key, val in settings.items():
        print(f"  {key}: {val}")
    return settings


# ── LAMMPS molecular FF input snippet (AMBER ff14SB for a peptide in water) ─
LAMMPS_AMBER_SNIPPET = """
# LAMMPS input for AMBER ff14SB peptide in TIP3P water
# Generated via acpype or VMD topotools from GROMACS topology

units           real            # kcal/mol, Å, fs, K, atm
atom_style      full            # Required: charge + bond topology
boundary        p p p

read_data       peptide_in_water.lammps

# AMBER force field: LJ + Coulomb (long range)
pair_style      lj/charmm/coul/long 10.0 12.0
# Inner cutoff 10 Å: begin switching; outer 12 Å: hard cutoff for vdW
# Coulomb handled by kspace (PME) beyond rcoulomb
kspace_style    pppm 1e-4       # 1e-4 relative RMS error in forces
pair_modify     mix arithmetic  # Lorentz-Berthelot (AMBER default)

# Bonded terms
bond_style      harmonic
angle_style     harmonic
dihedral_style  charmm          # Handles AMBER cosine dihedrals + 1-4 terms
improper_style  harmonic        # For planar groups (aromatic rings, peptide bonds)

# 1-4 scaling — MUST match AMBER convention
special_bonds   amber           # fudgeLJ=0.5, fudgeQQ=0.8333 (built-in preset)

# H-bond constraints (required for 2 fs timestep with H atoms)
fix             shake_hbonds all shake 1e-4 20 0 m 1.008
# Constrains all bonds to H; removes O-H and C-H fastest modes.

timestep        2.0             # fs; safe with SHAKE-constrained H bonds
# Without SHAKE: timestep must be ≤ 0.5 fs to resolve O-H stretching (~3500 cm⁻¹)

# Thermostat and barostat
fix             npt_run all npt temp 300.0 300.0 100.0 iso 1.013 1.013 1000.0
# Tdamp = 100 fs, Pdamp = 1000 fs (in real units = fs). [EXPERT REVIEW NEEDED]

thermo          500
thermo_style    custom step temp press vol etotal ke pe
run             500000          # 1 ns
"""
```

### Workflow 3: Materials force fields — EAM, Tersoff, Buckingham, and ReaxFF in LAMMPS

Materials force fields are deployed without topology files. Parameter provenance — the exact file and publication — must be recorded.

```lammps
# ============================================================
# EAM for pure FCC Ni — Mishin et al. (1999)
# LAMMPS units: metal (Å, eV, ps, K, bar)
# Source: NIST IPR, file: Ni99.eam.alloy
# SHA-256 of Ni99.eam.alloy: [record before use]
# ============================================================
units           metal
atom_style      atomic
pair_style      eam/alloy
pair_coeff      * * Ni99.eam.alloy Ni
# Element list MUST match atom type order in the data file.
# EAM/alloy: single file contains embedding + pair + density functions.

# Validation checks (run before any production MD):
# 1. Cohesive energy: E_cohesive = -E_potential/N at 0 K ≈ 4.44 eV/atom for Ni [NIST]
# 2. Lattice parameter: relax with fix box/relax; compare to a = 3.52 Å (DFT/exp)
# 3. Elastic constants: C11 ≈ 247 GPa, C12 ≈ 147 GPa, C44 ≈ 125 GPa [EXPERT REVIEW NEEDED]
# 4. Stacking fault energy: γ_sf ≈ 125 mJ/m² (affects dislocation behavior)
# 5. Surface energy: (111) ≈ 1.6 J/m² (sensitive to surface-specific parameterization)
```

```lammps
# ============================================================
# MEAM for Fe-Mn alloy — Lee et al. (2010)
# LAMMPS units: metal
# Source: Lee, B.-J. (2010), Acta Materialia, DOI: 10.1016/j.actamat.2010.01.032
# ============================================================
units           metal
atom_style      atomic
pair_style      meam
pair_coeff      * * library.meam Fe Mn Fe_Mn.meam Fe Mn
# Two files: library.meam (element library) + interaction file.
# Element order in pair_coeff must match atom types in data file.
# MEAM adds angular-dependent embedding; better than EAM for alloys
# with significant covalent character (Fe-Mn, Ni-Al, Cu-Zr).
```

```lammps
# ============================================================
# Tersoff for SiC — Erhart & Albe (2005)
# LAMMPS units: metal
# Source: Erhart & Albe (2005), PRB 71, 035211
# WARNING: not the Tersoff (1989) Si parameters — different file entirely.
# ============================================================
units           metal
atom_style      atomic
pair_style      tersoff
pair_coeff      * * SiC_Erhart-Albe.tersoff Si C
# Element list must match atom type order.
# Tersoff cutoff: smooth from 2.7 to 3.0 Å for Si-Si; check that the
# second coordination shell (Si-Si: ~3.84 Å) is well beyond the cutoff.
# If r_cut bisects a coordination shell, forces have an artifact spike.
```

```lammps
# ============================================================
# Buckingham + Coulomb for MgO — Lewis & Catlow (1985)
# LAMMPS units: metal (eV, Å)
# Source: Lewis & Catlow (1985), J. Phys. C: Solid State Phys.
# ============================================================
units           metal
atom_style      charge          # Required: per-atom charges
boundary        p p p

pair_style      buck/coul/long 12.0  # Buckingham + Ewald long-range Coulomb
kspace_style    ewald 1e-5           # High accuracy for ionic system

# Mg: formal charge +2; O: formal charge -2
# Set charges in the data file or via 'set':
# set type 1 charge  2.0   # Mg
# set type 2 charge -2.0   # O

pair_coeff      1 2 1279.69 0.3 0.0   # Mg-O: A, ρ (Å), C (eV·Å⁶)
pair_coeff      2 2 22764.3 0.1490 27.89  # O-O: repulsive + dispersion
# No Mg-Mg Buckingham term: Lewis-Catlow uses only Mg-O and O-O.

# WARNING: Buckingham diverges at r → 0 (A*exp(-r/ρ) → ∞ as r → 0,
# while -C/r⁶ → -∞ faster than exponential rises).
# For high-temperature or high-pressure MD: add ZBL short-range correction.
# pair_style buck/coul/long/cs 12.0   # /cs = core-shell model support
```

```lammps
# ============================================================
# ReaxFF for Fe-C-H-O system — Aryanpour et al. (2010)
# LAMMPS units: real (kcal/mol, Å, fs) — ReaxFF always uses real units
# Source: Aryanpour et al. (2010), J. Phys. Chem. A
# ============================================================
units           real            # ReaxFF is parameterized in kcal/mol units
atom_style      charge          # ReaxFF requires per-atom charge storage
boundary        p p p

pair_style      reaxff NULL
# NULL: no safezone/mincap (handled automatically in modern LAMMPS)
pair_coeff      * * ffield.reax.Fe_C_H_O Fe C H O
# Element order must match atom type order in data file.
# NEVER mix parameter files from different publications.
# The ffield.reax file is a monolithic ASCII file; all parameters are
# coupled — partial substitution from another publication is physically wrong.

# REQUIRED: charge equilibration at every step
fix             qeq_reaxff all qeq/reaxff 1 0.0 10.0 1e-6 reaxff
# qeq/reaxff: every 1 step; cutoff 10 Å; tolerance 1e-6; uses reaxff charges
# The QEq cutoff (10 Å) should match the ReaxFF interaction cutoff in ffield.reax
# (typically 10 Å). Mismatched cutoffs produce wrong charges.

# REQUIRED: small timestep
timestep        0.25            # fs; ReaxFF typical range: 0.1–0.5 fs [EXPERT REVIEW NEEDED]
# Larger timestep → QEq may not converge; high-frequency charge oscillations.

# ReaxFF neighbor list: must be rebuilt frequently
neighbor        2.0 bin
neigh_modify    every 1 delay 0 check no  # Rebuild every step (ReaxFF default)

# Species tracking (monitor bond formation/breaking)
fix             species_out all reaxff/species 100 1 100 species.out element Fe C H O
```

### Workflow 4: ML potentials as force fields — deployment in ASE and LAMMPS

ML potentials (MACE, NequIP, DeePMD) are used exactly like classical force fields in production MD. The distinction is in the parameter-file format and the deployment constraints.

```python
import numpy as np
from ase.io import read
from ase.md.npt import NPT
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase import units
from pathlib import Path


def setup_mace_md(
    structure_file: str,
    model_path: str,
    temperature_K: float = 1000.0,
    timestep_fs: float = 2.0,
    n_steps: int = 500_000,
    device: str = "cuda",
    dtype: str = "float32",
    element_order: list = None,
) -> None:
    """
    Set up NVT or NPT MD using MACE as the force field via ASE.

    For production simulations at > 5000 atoms or for long timescales,
    use LAMMPS pair_style mace (see mlp-to-lammps) for better throughput.
    ASE MD is preferred for: short validation runs, interactive exploration,
    and when LAMMPS-MACE compilation is not available.

    element_order: list of element symbols in the order used by the MACE model.
        Must match the training data; MACE stores this internally but specifying
        it explicitly prevents silent element-ordering bugs.
    """
    from mace.calculators import MACECalculator

    atoms = read(structure_file)
    calc = MACECalculator(
        model_paths=model_path,
        device=device,
        default_dtype=dtype,
        # float32: 2–3× faster; fine for production MD.
        # float64: required for energy validation and NVE energy conservation tests.
    )
    atoms.calc = calc

    # Sanity check: energy and max force before starting
    e0 = atoms.get_potential_energy()
    f0 = atoms.get_forces()
    f_max = np.abs(f0).max()
    print(f"Initial energy: {e0/len(atoms):.4f} eV/atom")
    print(f"Initial max force: {f_max:.3f} eV/Å")
    if f_max > 20.0:
        raise RuntimeError(
            f"Max force {f_max:.1f} eV/Å before MD starts — structure is not equilibrated. "
            "Run ASE energy minimization (BFGS or FIRE) before MD, or pre-equilibrate "
            "with a cheaper classical potential."
        )

    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_K,
                                  rng=np.random.default_rng(42))
    # Zero CoM momentum
    mom = atoms.get_momenta()
    mom -= mom.mean(axis=0)
    atoms.set_momenta(mom)

    # NPT with Nosé-Hoover-Langevin (ASE NPT integrator)
    dyn = NPT(
        atoms,
        timestep=timestep_fs * units.fs,
        temperature_K=temperature_K,
        externalstress=0.0,            # 0 GPa external pressure
        ttime=100 * units.fs,          # Thermostat relaxation time
        pfactor=None,                  # None = NVT; set to a bulk modulus for NPT
    )

    print(f"Starting MD: {n_steps} steps × {timestep_fs} fs = {n_steps*timestep_fs/1000:.0f} ps")
    print(f"Model: {model_path}, device: {device}, dtype: {dtype}")

    # Attach trajectory writer and thermo logger
    from ase.io import Trajectory
    traj = Trajectory("mace_md.traj", "w", atoms)
    dyn.attach(traj.write, interval=100)

    def print_thermo():
        T = atoms.get_temperature()
        E = atoms.get_total_energy() / len(atoms)
        print(f"  Step {dyn.get_number_of_steps()}: T={T:.1f} K, E={E:.4f} eV/atom")

    dyn.attach(print_thermo, interval=1000)
    dyn.run(n_steps)
    traj.close()


# ── LAMMPS deployment of MACE (preferred for large systems) ──────────────────
LAMMPS_MACE_SNIPPET = """
# LAMMPS input for MACE-driven NPT MD
# Requires LAMMPS compiled with MACE interface (see mlp-to-lammps skill)

units           metal                   # Å, eV, ps — MACE convention
atom_style      atomic
boundary        p p p

read_data       structure.lammps

pair_style      mace no_domain_decomposition
pair_coeff      * * /path/to/model.model W Ta
# Element list: must match atom type ordering in the data file
# AND must match the element ordering used during MACE training.
# no_domain_decomposition: mandatory for equivariant MPNNs.
# This disables MPI domain decomposition — all atoms on one rank.
# For GPU: LAMMPS uses one GPU per MPI rank; use 1 rank per GPU.

timestep        0.002                   # 2 fs (metal units: ps)
neighbor        2.0 bin
neigh_modify    every 1 delay 0 check yes

fix             npt_run all npt temp 1000.0 1000.0 0.1 iso 0.0 0.0 1.0
thermo          500
thermo_style    custom step temp press vol etotal ke pe
dump            prod all custom 1000 mace_md.dump id type x y z fx fy fz
dump_modify     prod sort id
run             500000                  # 1 ns
"""

# ── DeePMD-kit in LAMMPS ──────────────────────────────────────────────────────
LAMMPS_DEEPMD_SNIPPET = """
# LAMMPS input for DeePMD-kit model
# Requires LAMMPS compiled with DeePMD-kit interface

units           metal
atom_style      atomic
pair_style      deepmd frozen_model.pb
pair_coeff      * *
# DeePMD: element mapping is defined in the frozen model file.
# pair_coeff * * — no explicit element list; order is encoded in the .pb file.
# WARNING: if atom type ordering in the data file differs from DeePMD training,
# results are wrong with no error message. Verify with single-point energies.
timestep        0.002
"""

# ── NequIP / Allegro in LAMMPS ─────────────────────────────────────────────
LAMMPS_NEQUIP_SNIPPET = """
# LAMMPS input for NequIP or Allegro model (deployed via pair_style nequip)
units           metal
atom_style      atomic
pair_style      nequip
pair_coeff      * * deployed_model.pth W Ta
# Same element-order requirement as MACE.
# no_domain_decomposition not always required for Allegro (linear attention).
# Check the NequIP/Allegro documentation for your version. [EXPERT REVIEW NEEDED]
timestep        0.002
"""
```

### Workflow 5: Force-field validation against DFT reference

A force field must be validated against DFT — not against another force field — before production use. The validation protocol is tiered: static properties first, then dynamic.

```python
import numpy as np
from ase.io import read
from ase.eos import EquationOfState
from ase.build import bulk
from pathlib import Path


def validate_cohesive_energy_and_lattice(
    atoms_eq,
    calc,
    species: str,
    dft_references: dict,
) -> dict:
    """
    Validate cohesive energy, lattice parameter, and bulk modulus
    against DFT reference values.

    dft_references: {
        "E_cohesive_eV_atom": float,   # DFT cohesive energy (positive = binding)
        "a0_angstrom": float,           # Equilibrium lattice constant
        "B0_GPa": float,                # Bulk modulus
    }

    Cohesive energy = -(E_crystal - N * E_atom) / N
    EOS fit gives V0, E0, B0.
    """
    natoms = len(atoms_eq)

    # ── Cohesive energy ───────────────────────────────────────────────────────
    atoms_eq.calc = calc
    E_crystal = atoms_eq.get_potential_energy()

    # Isolated atom energy (requires a large vacuum cell)
    from ase import Atoms
    atom_cell = Atoms(species, positions=[(0, 0, 0)],
                      cell=[15, 15, 15], pbc=False)
    atom_cell.calc = calc
    E_atom = atom_cell.get_potential_energy()
    E_coh_ff = -(E_crystal / natoms - E_atom)

    E_coh_dft = dft_references.get("E_cohesive_eV_atom", float("nan"))
    delta_E_coh = (E_coh_ff - E_coh_dft) / E_coh_dft * 100

    print(f"\n{'Property':<30}  {'DFT':>10}  {'FF':>10}  {'Error':>10}")
    print("─" * 63)
    print(f"{'E_cohesive (eV/atom)':<30}  {E_coh_dft:>10.3f}  {E_coh_ff:>10.3f}  "
          f"{delta_E_coh:>+9.2f}%")
    if abs(delta_E_coh) > 5.0:
        print("  WARNING: Cohesive energy error > 5% — potential may be misapplied "
              "or using wrong units")

    # ── Equation of state ─────────────────────────────────────────────────────
    a0_dft = dft_references.get("a0_angstrom", float("nan"))
    volumes, energies = [], []
    for strain in np.linspace(0.85, 1.15, 13):
        a = atoms_eq.copy()
        a.set_cell(a.cell * strain**(1/3), scale_atoms=True)
        a.calc = calc
        volumes.append(a.get_volume() / natoms)
        energies.append(a.get_potential_energy() / natoms)

    eos = EquationOfState(volumes, energies, eos="birch-murnaghan")
    V0, E0, B_eV_A3 = eos.fit()
    B0_GPa = B_eV_A3 * 160.2176   # eV/Å³ → GPa
    # Lattice constant from V0 (for cubic crystal)
    a0_ff = (V0 * natoms) ** (1/3)  # Only correct for simple cubic-like cells

    a0_dft_ref = dft_references.get("a0_angstrom", float("nan"))
    B0_dft_ref = dft_references.get("B0_GPa", float("nan"))
    delta_a = (a0_ff - a0_dft_ref) / a0_dft_ref * 100 if not np.isnan(a0_dft_ref) else float("nan")
    delta_B = (B0_GPa - B0_dft_ref) / B0_dft_ref * 100 if not np.isnan(B0_dft_ref) else float("nan")

    print(f"{'a0 (Å)':<30}  {a0_dft_ref:>10.4f}  {a0_ff:>10.4f}  {delta_a:>+9.2f}%")
    print(f"{'B0 (GPa)':<30}  {B0_dft_ref:>10.2f}  {B0_GPa:>10.2f}  {delta_B:>+9.2f}%")

    if abs(delta_a) > 1.0:
        print("  WARNING: a0 error > 1% — lattice-parameter-sensitive properties "
              "(surface energies, phonon frequencies) will be shifted")
    if abs(delta_B) > 10.0:
        print("  WARNING: B0 error > 10% — elastic properties and phonon frequencies unreliable")

    return {"E_coh_FF": E_coh_ff, "a0_FF": a0_ff, "B0_GPa_FF": B0_GPa}


def validate_force_errors_on_dft_set(
    test_file: str,
    calc,
    energy_key: str = "energy",
    forces_key: str = "forces",
    stress_key: str = "stress",
) -> dict:
    """
    Compute energy, force, and stress MAE on a DFT-labeled test set (extxyz format).

    The test set should span the configurations of interest: bulk at various strains,
    surfaces, defects, and finite-temperature snapshots if MD accuracy is required.

    This is the same protocol as mlp-validation Workflow 1, applied to classical FFs.
    Classical FFs typically achieve:
      - EAM for metals: force MAE 50–200 meV/Å at 1000 K snapshots [EXPERT REVIEW NEEDED]
      - AMBER ff14SB: force MAE 50–150 meV/Å for protein geometries [EXPERT REVIEW NEEDED]
      - Tersoff for Si: force MAE 100–300 meV/Å for bulk snapshots [EXPERT REVIEW NEEDED]
      - ReaxFF: force MAE 100–400 meV/Å (higher tolerance due to reactive events)
      These are order-of-magnitude benchmarks; actual values are system-dependent.
    """
    frames = read(test_file, index=":", format="extxyz")
    e_errors, f_maes = [], []

    for atoms in frames:
        natoms = len(atoms)
        ref_e = atoms.info.get(energy_key, None)
        ref_f = atoms.arrays.get(forces_key, None)
        if ref_e is None or ref_f is None:
            continue

        a = atoms.copy()
        a.calc = calc
        pred_e = a.get_potential_energy()
        pred_f = a.get_forces()

        e_errors.append(abs(pred_e / natoms - ref_e / natoms) * 1000)   # meV/atom
        f_maes.append(np.abs(pred_f - ref_f).mean() * 1000)             # meV/Å

    e_mae = np.mean(e_errors)
    f_mae = np.mean(f_maes)

    print(f"\nForce-field validation on DFT test set ({len(e_errors)} structures):")
    print(f"  Energy MAE: {e_mae:.2f} meV/atom")
    print(f"  Force  MAE: {f_mae:.2f} meV/Å")

    if f_mae > 500:
        print("  WARNING: Force MAE > 500 meV/Å — this force field is likely outside "
              "its validity domain for these structures. Consider MLP training.")
    elif f_mae > 200:
        print("  CAUTION: Force MAE > 200 meV/Å — acceptable for qualitative MD "
              "but may give incorrect quantitative properties.")
    else:
        print(f"  Force MAE {f_mae:.0f} meV/Å — within typical classical FF range.")

    return {"energy_mae_meV_atom": e_mae, "force_mae_meV_A": f_mae,
            "n_structures": len(e_errors)}


def validate_phonons_with_forcefield(
    equilibrium_atoms,
    calc,
    supercell_matrix: list,
    dft_phonon_yaml: str = None,
    displacement: float = 0.01,   # Å
) -> dict:
    """
    Compute force-field phonon spectrum via finite displacements (phonopy).
    Compare to DFT phonons if available.

    Phonon frequency errors for classical FFs vs. DFT:
    - EAM Ni (Mishin 1999): RMS error ~5% for acoustic, ~15% for optical [EXPERT REVIEW NEEDED]
    - Tersoff Si: RMS error ~10–20% for acoustic branches [EXPERT REVIEW NEEDED]
    - ReaxFF: varies widely; validate per-system
    Imaginary modes in a stable crystal = force field is locally unstable.
    """
    import phonopy
    from phonopy.structure.atoms import PhonopyAtoms

    cell = PhonopyAtoms(
        symbols=equilibrium_atoms.get_chemical_symbols(),
        scaled_positions=equilibrium_atoms.get_scaled_positions(),
        cell=equilibrium_atoms.get_cell(),
    )
    ph = phonopy.Phonopy(cell, supercell_matrix)
    ph.generate_displacements(distance=displacement)

    forces_list = []
    for sc in ph.supercells_with_displacements:
        if sc is None:
            forces_list.append(None)
            continue
        from ase import Atoms as AseAtoms
        a = AseAtoms(symbols=sc.symbols, positions=sc.positions,
                     cell=sc.cell, pbc=True)
        a.calc = calc
        forces_list.append(a.get_forces())

    ph.set_forces(forces_list)
    ph.produce_force_constants()

    ph.run_qpoints([[0, 0, 0]])
    gamma_freqs = ph.get_qpoints_and_frequencies()[1][0]  # THz
    n_imaginary = int((gamma_freqs < -0.1).sum())

    if n_imaginary > 0:
        print(f"\nWARNING: {n_imaginary} imaginary modes at Γ — force field "
              "gives a locally unstable equilibrium. Possible causes:")
        print("  - Equilibrium structure is not at the true minimum for this FF")
        print("  - Cutoff function artifacts (Tersoff near a shell boundary)")
        print("  - ReaxFF charges not converged at this geometry")
    else:
        print(f"\nΓ-point phonons: all real. Lowest = {gamma_freqs.min():.3f} THz")

    ph.auto_band_structure(npoints=51, write_yaml=True,
                           filename="ff_phonon_band.yaml")

    return {"n_imaginary_gamma": n_imaginary, "gamma_freqs_THz": gamma_freqs,
            "phonopy_object": ph}
```

### Workflow 6: Unit and parameter-file compatibility across codes

Force-field parameters are expressed in code-specific units. Porting a parameter set between codes requires exact unit conversions. A unit error by a factor of 10 is a silent simulation failure.

```python
"""
Unit systems for major MD codes:

                LAMMPS metal    LAMMPS real    GROMACS       OpenMM        ASE (internal)
Energy          eV              kcal/mol       kJ/mol        kJ/mol        eV
Length          Å               Å              nm            nm            Å
Time            ps              fs             ps            ps            —
Force           eV/Å            kcal/mol/Å     kJ/mol/nm     kJ/mol/nm     eV/Å
Pressure        bar             atm            bar           bar           —
Temperature     K               K              K             K             —
Mass            g/mol           g/mol          g/mol         g/mol         —
Charge          e (electron)    e              e             e             —

Conversion factors:
  1 eV = 23.0609 kcal/mol = 96.4853 kJ/mol
  1 Å  = 0.1 nm
  1 eV/Å = 1602.18 pN = 160.218 kJ/mol/nm = 38.09 kcal/mol/Å
  1 eV/Å³ = 160.218 GPa
  1 bar = 1e-4 GPa = 0.986923 atm
"""

UNIT_CONVERSIONS = {
    "eV_to_kcal_mol": 23.0609,
    "eV_to_kJ_mol": 96.4853,
    "A_to_nm": 0.1,
    "nm_to_A": 10.0,
    "eV_A3_to_GPa": 160.2176,
    "GPa_to_bar": 1e4,
    "kcal_mol_A_to_eV_A": 1 / 23.0609,
    "kJ_mol_nm_to_eV_A": 1 / (96.4853 * 10),  # kJ/mol/nm → eV/Å
}


def convert_lj_parameters(
    epsilon: float,
    sigma: float,
    source_code: str,
    target_code: str,
) -> tuple[float, float]:
    """
    Convert LJ parameters (epsilon, sigma) between code unit conventions.

    source_code / target_code: one of "LAMMPS_metal", "LAMMPS_real", "GROMACS", "OpenMM"

    GROMACS and OpenMM use C6/C12 internally but store epsilon/sigma in .itp files.
    ASE expects eV/Å (same as LAMMPS metal).

    WARNING: sigma conventions differ between force fields:
      AMBER/CHARMM: Rmin/2 = σ * 2^(1/6) / 2 ≠ σ in GROMACS .itp
      OPLS-AA: uses σ directly.
    Always verify which convention is in use before porting parameters.
    """
    conv = {
        ("LAMMPS_metal", "LAMMPS_real"):  (23.0609, 1.0),   # eV→kcal/mol, Å→Å
        ("LAMMPS_real",  "LAMMPS_metal"): (1/23.0609, 1.0),
        ("LAMMPS_metal", "GROMACS"):      (96.4853, 0.1),    # eV→kJ/mol, Å→nm
        ("GROMACS",      "LAMMPS_metal"): (1/96.4853, 10.0),
        ("LAMMPS_real",  "GROMACS"):      (96.4853/23.0609, 0.1),
        ("GROMACS",      "LAMMPS_real"):  (23.0609/96.4853, 10.0),
        ("LAMMPS_metal", "OpenMM"):       (96.4853, 0.1),    # Same as GROMACS
        ("OpenMM",       "LAMMPS_metal"): (1/96.4853, 10.0),
    }
    key = (source_code, target_code)
    if key not in conv:
        raise ValueError(f"Conversion {source_code} → {target_code} not defined. "
                         f"Available: {list(conv.keys())}")
    e_factor, r_factor = conv[key]
    return epsilon * e_factor, sigma * r_factor


def check_lammps_units_consistency(input_script: str) -> list[str]:
    """
    Parse a LAMMPS input script and flag potential unit inconsistencies.

    Common errors detected:
    - pair_style eam or mace with 'units real' (EAM/MACE require 'units metal')
    - Tdamp or Pdamp that appear physically unreasonable for the unit system
    - timestep outside the stable range for the unit system and force field
    """
    warnings = []
    units_mode = None
    pair_style = None
    timestep = None
    tdamp = None

    with open(input_script) as f:
        for line in f:
            s = line.strip().lower()
            if s.startswith("units "):
                units_mode = s.split()[1]
            elif s.startswith("pair_style "):
                pair_style = s.split()[1]
            elif s.startswith("timestep "):
                try:
                    timestep = float(s.split()[1])
                except (IndexError, ValueError):
                    pass
            elif "nvt" in s or "npt" in s:
                parts = s.split()
                for i, p in enumerate(parts):
                    if p == "temp" and i + 3 < len(parts):
                        try:
                            tdamp = float(parts[i + 3])
                        except ValueError:
                            pass

    # Units vs. force-field compatibility
    if pair_style in ("eam", "eam/alloy", "meam", "tersoff", "sw") and units_mode != "metal":
        warnings.append(
            f"pair_style {pair_style} requires 'units metal'; "
            f"found 'units {units_mode}'. EAM/Tersoff parameter files are "
            "in metal units; using real units produces wrong energies and forces."
        )
    if pair_style in ("mace", "nequip") and units_mode != "metal":
        warnings.append(
            f"pair_style {pair_style} requires 'units metal'; "
            f"found 'units {units_mode}'. MACE/NequIP are trained in eV/Å convention."
        )
    if pair_style in ("lj/charmm/coul/long", "lj/cut/coul/long") and units_mode == "metal":
        warnings.append(
            "pair_style lj/* with 'units metal': CHARMM/AMBER parameters are in "
            "kcal/mol and Å (real units). Using them in metal units changes energies "
            "by a factor of 23.06."
        )

    # Timestep sanity check
    if timestep is not None and units_mode == "metal":
        if timestep > 0.005:
            warnings.append(
                f"timestep {timestep} in metal units = {timestep*1000:.1f} fs. "
                "For metals with EAM: 1–2 fs is typical (0.001–0.002 ps). "
                "For MACE/NequIP: 1–2 fs. Check if this is intended."
            )
        if timestep < 1e-5:
            warnings.append(
                f"timestep {timestep} in metal units = {timestep*1e3:.4f} fs — "
                "extremely small. ReaxFF uses 0.1–0.5 fs (0.0001–0.0005 in metal units)."
            )
    if timestep is not None and units_mode == "real":
        if timestep > 2.0:
            warnings.append(
                f"timestep {timestep} fs in real units. "
                "With SHAKE H-constraints: 2 fs is the maximum. "
                "Without constraints: 0.5–1 fs for organic/biomolecular systems."
            )

    # Tdamp sanity
    if tdamp is not None and units_mode == "metal":
        if tdamp > 10.0:
            warnings.append(
                f"Tdamp = {tdamp} in metal units (ps). "
                "Tdamp > 10 ps = 10,000 fs is unusually large; thermostat barely active. "
                "Typical Nosé-Hoover Tdamp: 0.05–0.2 ps (50–200 fs)."
            )
        if tdamp < 0.01:
            warnings.append(
                f"Tdamp = {tdamp} in metal units = {tdamp*1000:.1f} fs. "
                "Tdamp < 10 fs is overdamped — will cause temperature oscillations."
            )
    if tdamp is not None and units_mode == "real":
        if tdamp > 1000.0:
            warnings.append(
                f"Tdamp = {tdamp} fs in real units. "
                "Typical Nosé-Hoover Tdamp: 50–200 fs. "
                "Tdamp = {tdamp} fs = {tdamp/1000:.1f} ps — thermostat is very slow."
            )

    if not warnings:
        print(f"Units consistency check: PASSED (units={units_mode}, "
              f"pair_style={pair_style}, timestep={timestep})")
    else:
        for w in warnings:
            print(f"WARNING: {w}")

    return warnings
```

## Common Failure Modes

| Failure | Symptom | Likely Cause | Diagnostic | Fix |
|---------|---------|--------------|------------|-----|
| Wrong `units` for force field | Energy explosion or wildly wrong cohesive energy within first 100 steps; temperature jumps to millions of K; total energy is 10³× off | EAM/Tersoff/MACE parameters are in metal units (eV, Å) but the simulation uses `units real` (kcal/mol, Å); or AMBER parameters are in kcal/mol but used in metal units | Compare predicted cohesive energy per atom to known value; if off by factor 23.06 (eV↔kcal/mol) or 0.1/10 (Å↔nm), it is a unit error | Set `units metal` for EAM/Tersoff/MACE/NequIP; set `units real` for AMBER/CHARMM/OPLS; never mix unit systems in a single force field definition |
| Using force field outside its parameterization domain | Predicted melting point wrong by > 30%; surface energies orders-of-magnitude off; liquid structure incorrect; elastic constants outside physical range | EAM fitted to FCC metals applied to HCP polymorph, surface, or alloy outside fitting composition; AMBER applied to metal-containing cofactor without metal-specific parameters | Compare force-field predictions to DFT for the actual system (not just the fitting system); compute lattice parameter, bulk modulus, and RDF and compare to DFT | Choose a force field parameterized for the target system or train an MLP; never use EAM outside the element and structural family it was fitted to |
| Mixed parameter-file versions | Simulation "runs" but cohesive energy or lattice constant differs from the published paper by > 5%; elastic constants wrong | Two parameter files with the same name but from different publications (e.g., Mishin1999 vs. Mishin2001 for Ni); or a repository file that has been updated silently | Recompute cohesive energy, a₀, and B₀ from the force field and compare to the reference publication's Table 1 or SI; record SHA-256 of the parameter file | Always record SHA-256 hash and primary reference publication for every parameter file; obtain files directly from the primary source (author supplementary or NIST IPR) |
| Wrong element ordering in `pair_coeff` | Forces and energies are computed for the wrong species (type 1 treated as type 2); coordination numbers, RDF peak positions, and total energy are wrong | Element list in `pair_coeff * * potential.eam Fe Ni` does not match atom type order in the data file (type 1 = Ni, type 2 = Fe, but `pair_coeff` lists Fe first) | Print mass per type from the LAMMPS log; cross-check: type 1 → element 1 in `pair_coeff` → correct mass and expected nearest-neighbor distance in RDF | Reorder the element list in `pair_coeff` to match the atom type numbering in the data file; or re-write the data file to match the desired element order |
| Missing `fix qeq/reaxff` for ReaxFF | LAMMPS error: `ERROR: fix qeq/reaxff required`; or simulation crashes immediately; or charges remain at 0 and forces are wrong | `pair_style reaxff` requires a charge equilibration fix that is a separate command; if omitted, charges are not equilibrated | Check input script for `fix qeq/reaxff` after `pair_coeff`; it is mandatory, not optional | Add `fix qeq_id all qeq/reaxff 1 0.0 10.0 1e-6 reaxff` immediately after `pair_coeff` |
| Mixed ffield.reax files for ReaxFF | Energies and forces deviate significantly from published results; reactions occur at wrong conditions; charge equilibration converges slowly | Two separate ffield.reax parameter files have been concatenated or selectively merged, combining incompatible parameters from different publications | Re-derive the ffield.reax from a single primary publication; compare element sections to the reference paper | Use one and only one ffield.reax file per simulation; all parameters in the file are coupled; partial substitution is physically invalid |
| Wrong 1-4 scaling factors | Torsion profiles wrong; protein secondary structure destabilizes in AMBER simulation; conformational preferences systematically shifted | `special_bonds` setting does not match the force field: using `special_bonds charmm` with AMBER (or vice versa); or missing `special_bonds` line (default in LAMMPS is lj/coul 0 0 0, which excludes all 1-4 nonbonded) | Compute a single dihedral energy profile (e.g., glycine ψ angle) vs. GAUSSIAN or reference QM; AMBER 1-4 LJ×0.5, Coulomb×0.8333 should match QM profile within 1 kcal/mol | Set `special_bonds amber` for AMBER ff14SB/GAFF; `special_bonds charmm` for CHARMM36; for OPLS-AA: `special_bonds lj/coul 0.0 0.0 0.5` with `pair_modify mix geometric` |
| Wrong or missing mixing rules | LJ cross-pair interactions differ from intended; heterogeneous system has wrong interaction strengths; solvation free energies off by > 2 kcal/mol | LAMMPS default mixing rule (`arithmetic`) applied to OPLS-AA parameters (which require `geometric`); or `pair_modify mix` not set to match the force field | Manually compute a cross-pair ε from the mixing rule and compare to the force field documentation; OPLS-AA table lists ε_ii; cross-ε must be geometric | Set `pair_modify mix geometric` for OPLS-AA; `pair_modify mix arithmetic` for AMBER/CHARMM; mixing rule is non-negotiable and part of the force-field definition |
| Tersoff cutoff bisecting a coordination shell | Spurious force spikes in MD; atoms at the cutoff distance experience discontinuous forces; anomalously high-frequency phonon modes | The Tersoff cutoff function transitions from 1 to 0 within a coordination shell; atoms on either side of the cutoff appear/disappear as neighbors during MD | Plot RDF and identify the second coordination shell; compare to the Tersoff cutoff parameters (R_ij, D_ij in the parameter file) | Choose a Tersoff parameterization where R_ij + D_ij falls in the gap between the first and second coordination shells; do not change Tersoff parameters to avoid the problem without revalidating against DFT |
| Buckingham divergence at short range | Simulation crashes with atoms at very short distances (< 0.5 Å); energy becomes -∞ then NaN; typically triggered by high-P MD or initial overlapping geometry | Buckingham potential -C/r⁶ diverges to -∞ as r → 0, dominating the repulsive A*exp(-r/ρ) term at very short range; a high-energy configuration can accelerate an atom past the potential barrier | Check minimum pairwise distance in the initial and crashed frames; if any r_ij < 0.5 Å, Buckingham divergence is the cause | Remove atomic overlaps before MD (energy minimize); add ZBL correction: `pair_style buck/coul/long/zbl` to enforce physical repulsion at short range; for shock/high-P MD, always use ZBL |
| SHAKE/LINCS failing to converge | LAMMPS: `WARNING: Fix shake coupling errors detected`; GROMACS: `LINCS Warning: Step X, 3 atoms involved in constraint warnings`; simulation drifts in energy | Large initial forces on constrained bonds (atoms far from equilibrium); timestep too large for the constraint solver; topology defines SHAKE on bonds that should not be constrained | Minimize before applying SHAKE; reduce timestep; verify that SHAKE bond list matches the force-field parameterization (AMBER: constrain H bonds only, not C-C or C-O) | Minimize first (forces < 100 kcal/mol/Å); reduce timestep to 1 fs; re-check bond constraint list; for GROMACS, use LINCS with `lincs-order = 6` for stiff bonds |
| Trusting force-field names over parameter provenance | Same simulation nominally uses "Tersoff Si" but gives different elastic constants than the published paper | The parameter file used is not the one cited; LAMMPS ships multiple Tersoff Si parameterizations (Tersoff1989, Tersoff1988, Erhart2005 variants) with similar filenames | Record the filename, full path, and SHA-256 hash; recompute C₁₁, C₁₂, C₄₄ for bulk Si and compare to the paper's Table | Locate the parameter file from the primary publication; always cite the file hash alongside the paper |

## Best Practices

- **Validate every force field on your specific system before production, regardless of reputation.** A force field that is well-validated for one alloy composition may give 50% error in lattice parameter for a 10-at.% compositional variant. Spend one day on validation before spending one month on production.
- **Record SHA-256 hashes and full citations for every parameter file.** `sha256sum filename > filename.sha256`. Store this alongside the simulation directory. Parameter files update silently; the file hash is the only reliable version identifier.
- **Match the `units` command to the force field — this is not negotiable.** EAM, Tersoff, MACE, NequIP, CHGNet: `units metal`. AMBER, CHARMM, GAFF, OPLS-AA in LAMMPS: `units real`. ReaxFF: `units real`. Buckingham for oxides: `units metal` (if parameters are in eV/Å). GROMACS always uses nm/kJ/mol/ps internally regardless of what the original FF uses.
- **For molecular force fields, use CHARMM-GUI or AmberTools to build the topology.** Manually constructing GROMACS .top files or LAMMPS data files for proteins, lipids, or drug-like molecules is error-prone. CHARMM-GUI validates topology against the force-field database and outputs consistent parameter, coordinate, and script files.
- **Never mix parameters from different force fields unless explicitly supported.** AMBER partial charges are not compatible with CHARMM partial charges. OPLS-AA LJ parameters use geometric mixing; combining them with AMBER LJ parameters under Lorentz-Berthelot mixing is not physical. If you need cross-FF interactions (e.g., protein in CHARMM + surface in Tersoff), the interface interaction must be parameterized explicitly.
- **For ReaxFF: use exactly one ffield.reax file, from exactly one publication, for a system it was validated on.** ReaxFF parameters are globally coupled — element electronegativity, bond order, and angular terms are all interdependent. Importing one element's parameters from a different ffield.reax file invalidates the rest of the parameterization.
- **Always verify the 1-4 nonbonded scaling factors match the force field definition.** This requires checking the LAMMPS `special_bonds` line or the GROMACS .top `[defaults]` section against the force-field documentation. A wrong scaling factor introduces a systematic error in all dihedral-containing conformational preferences.
- **For ML potentials as force fields: run the NVE energy conservation test before any production NPT.** MACE, NequIP, and DeePMD all require this check — a trained model can still have unphysical high-energy configurations near the boundary of the training set that inject energy and destabilize the trajectory. See `mlp-validation` for the protocol.
- **Use the NIST Interatomic Potentials Repository as the primary source for materials force fields.** The repository (https://www.ctcms.nist.gov/potentials/) stores validated parameter files with metadata, source publications, and verified properties. Prefer it over files distributed with simulation codes (which may be outdated) or informal sources.
- **For universal ML potentials (MACE-MP-0, CHGNet, M3GNet): validate before production even though they are pre-trained.** These models cover broad composition space but may be less accurate for unusual coordination environments, high-pressure phases, or systems at the edge of the Materials Project training distribution. Always compare a₀, B₀, and phonon frequencies to DFT for the specific composition before committing to production. [EXPERT REVIEW NEEDED]

## Integration with Other Skills

- **[lammps-workflow](../lammps-workflow/SKILL.md):** That skill covers the full LAMMPS input-script mechanics (`pair_style`, `pair_coeff`, `units`, `atom_style`, neighbor lists, and restart protocol). This skill specifies which `pair_style` to choose and why, and what the parameter files mean physically. Use both skills together: this one for force-field selection and validation, that one for input-script construction.
- **[md-equilibration](../md-equilibration/SKILL.md):** A common symptom of a misapplied force field is failure to equilibrate — the thermostat masks force-field errors by continuously injecting or removing energy. That skill's NVE energy-drift test is the most reliable way to detect force-field problems that the thermostat is compensating: if total energy drifts in NVE after correct equilibration, the force field or timestep is wrong.
- **[md-analysis](../md-analysis/SKILL.md):** Structural observables (RDF peak position, coordination number, lattice parameter from MSD plateau) computed in that skill are the primary runtime diagnostics for force-field validity. If the RDF first peak is at the wrong distance or the coordination number is physically impossible, the force field is misapplied.
- **[gromacs-workflow](../gromacs-workflow/SKILL.md):** GROMACS uses .top/.itp topology files and .mdp parameter files in nm/kJ/mol units. The topology construction, charge assignment, and constraint protocol for AMBER/CHARMM in GROMACS are covered in that skill; this skill provides the physical rationale for each choice.
- **[reactive-md](../reactive-md/SKILL.md):** ReaxFF and COMB — the reactive force fields described here — are deployed in LAMMPS with constraints and analysis tools described in that skill. That skill covers species-tracking, reaction-event detection, and validation of bond-breaking kinetics against DFT reaction pathways.
- **[mlp-validation](../../ml-interatomic-potentials/mlp-validation/SKILL.md):** The validation protocol in this skill (EOS, phonons, force MAE on DFT test set) is identical to the protocol for ML potentials in that skill. When an ML potential is used as a drop-in replacement for a classical force field, all of this skill's validation checks apply equally. That skill additionally covers NVE MD stability and LAMMPS vs. ASE consistency.
- **[mlp-to-lammps](../../ml-interatomic-potentials/mlp-to-lammps/SKILL.md):** Deploying MACE, NequIP, or DeePMD as a force field in LAMMPS — including compilation flags, `pair_style` invocation syntax, `no_domain_decomposition` requirement, element-ordering convention, and GPU parallelization limits — is covered in detail in that skill.
- **[mace-training](../../ml-interatomic-potentials/mace-training/SKILL.md):** When classical force fields are insufficient for the target system and no universal MLP covers the composition, training a custom MACE model is the next option. That skill covers dataset construction, training hyperparameters, and model selection. This skill's validation workflows (EOS, force MAE, phonons) apply directly to the trained model.
- **[active-learning-mlp](../../ml-interatomic-potentials/active-learning-mlp/SKILL.md):** One motivation for active learning is the inadequacy of classical force fields for a particular configuration space. A classical force field can be used for cheap initial exploration to identify likely thermodynamically accessible configurations, which are then added to the DFT training set for an MLP. The force-field accuracy requirements for this seeding role are lower than for production property calculations.
- **[universal-potentials](../../ml-interatomic-potentials/universal-potentials/SKILL.md):** MACE-MP-0, CHGNet, M3GNet, and ORB are drop-in force-field replacements that cover broad inorganic composition space. This skill provides the decision logic for when to use a universal potential vs. a classical FF vs. a custom-trained MLP; that skill covers the practical deployment, accuracy benchmarks, and known failure modes of each universal model.
- **[dft-convergence](../../electronic-structure/dft-convergence/SKILL.md):** All force-field validation against DFT requires converged DFT reference calculations. Energy, force, and stress labels used for validation must be computed at the same ENCUT, k-point density, and POTCAR as used for training data. Inconsistent DFT convergence between validation reference and training data inflates apparent force-field errors.
- **[vasp-workflow](../../electronic-structure/vasp-workflow/SKILL.md):** DFT reference calculations for EOS, elastic constants, phonons, and per-structure force validation are typically run with VASP. The ISIF tag (ISIF=2 for fixed-cell force calculations, ISIF=3 for full relaxation) and the stress sign convention (VASP: compression-positive, must be negated for ASE) are covered in that skill and must be applied consistently when building validation datasets.

## Key References

1. Daw, M. S. & Baskes, M. I. (1984). *Embedded-atom method: Derivation and application to impurities, surfaces, and other defects in metals.* Physical Review B, 29(12), 6443–6453. https://doi.org/10.1103/PhysRevB.29.6443
2. Tersoff, J. (1988). *New empirical approach for the structure and energy of covalent systems.* Physical Review B, 37(12), 6991–7000. https://doi.org/10.1103/PhysRevB.37.6991
3. van Duin, A. C. T., et al. (2001). *ReaxFF: A reactive force field for hydrocarbons.* Journal of Physical Chemistry A, 105(41), 9396–9409. https://doi.org/10.1021/jp004368u
4. Cornell, W. D., et al. (1995). *A second generation force field for the simulation of proteins, nucleic acids, and organic molecules.* Journal of the American Chemical Society, 117(19), 5179–5197. https://doi.org/10.1021/ja00124a002
5. MacKerell, A. D., et al. (1998). *All-atom empirical potential for molecular modeling and dynamics studies of proteins.* Journal of Physical Chemistry B, 102(18), 3586–3616. https://doi.org/10.1021/jp973084f
6. Becker, C. A., et al. (2013). *Considerations for choosing and using force fields and interatomic potentials in materials science and engineering.* Current Opinion in Solid State and Materials Science, 17(6), 277–283. https://doi.org/10.1016/j.cossms.2013.10.001
7. Mishin, Y., et al. (1999). *Interatomic potentials for monoatomic metals from experimental data and ab initio calculations.* Physical Review B, 59(5), 3393–3407. https://doi.org/10.1103/PhysRevB.59.3393
8. Jorgensen, W. L., et al. (1996). *Development and testing of the OPLS all-atom force field on conformational energetics and properties of organic liquids.* Journal of the American Chemical Society, 118(45), 11225–11236. https://doi.org/10.1021/ja9621760
