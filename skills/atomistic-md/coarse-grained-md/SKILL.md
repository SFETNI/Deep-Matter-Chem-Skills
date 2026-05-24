---
name: coarse-grained-md
domain: atomistic-md
maturity: draft
tools:
  - GROMACS
  - LAMMPS
  - MDAnalysis
  - numpy
  - pandas
  - matplotlib
tested_versions:
  GROMACS: "2024.2"
  LAMMPS: "2Aug2023"
  MDAnalysis: "2.7.0"
  martinize2: "0.9.7"
requires_gpu: false
estimated_runtime: hours-to-days
---

# Coarse-Grained MD

## Description

This skill covers coarse-grained (CG) molecular dynamics: mapping atomistic structures to CG bead representations, parameterizing CG force fields (Martini 3, iterative Boltzmann inversion, force matching), running CG simulations in GROMACS and LAMMPS, backmapping CG configurations to atomistic resolution, and validating CG models against atomistic and experimental reference data. CG-MD replaces groups of heavy atoms with single interaction sites, reducing degrees of freedom by 2–10× and enabling simulations on spatial scales of 10–1000 nm and temporal scales of microseconds to milliseconds. Invoke this skill when the relevant length or time scale exceeds what all-atom MD can access within the computational budget, or when high-throughput screening of molecular assembly, polymer morphology, lipid bilayer properties, or protein–membrane interactions is required.

## Domain Context

Coarse-graining is not simply a faster simulation; it is a change of the physical model. A CG bead represents 2–10 heavy atoms and the associated hydrogen atoms as a single spherical interaction site with effective mass, position, and interaction parameters. The reduction in degrees of freedom comes at a cost:

- **Loss of chemical detail:** CG representations cannot distinguish stereoisomers, tautomers, or conformations at the bond-angle level. The CG bead type is the resolution limit of the chemical identity.
- **Modified dynamics:** CG potentials are free-energy surfaces, not potential energy surfaces. The smoother landscape increases conformational sampling but accelerates dynamics beyond the true atomistic rates. The CG time scale is related to the atomistic time scale by a system-dependent, temperature-dependent, and observable-dependent factor (typically 4–8× for Martini water). [EXPERT REVIEW NEEDED]
- **Non-uniqueness of the mapping:** Many different atom-to-bead assignments are consistent with the same molecular structure. Different mappings produce different CG potentials, different dynamics, and potentially different thermodynamic observables. There is no uniquely correct CG representation.
- **Transferability:** CG potentials fitted to reproduce one thermodynamic state (temperature, pressure, composition) may not transfer to another. Martini 3 is parameterized for biomolecules near 310 K, 1 bar, with explicit CG water. Using it at 500 K or with implicit solvation is extrapolation.
- **Bottom-up vs. top-down:** Bottom-up CG potentials (IBI, force matching, relative entropy) are derived systematically from all-atom reference simulations. They reproduce the atomistic structural and thermodynamic reference data at the parameterization state point but may have limited transferability. Top-down CG potentials (Martini) are parameterized to reproduce experimental thermodynamic data (partitioning, phase behavior, structure) and are more transferable across compositions but less precise for a specific system.

The **two-level simulation strategy** is the most productive use of CG-MD: use CG-MD to generate equilibrated configurations and explore large-scale morphology, then backmap to all-atom resolution and refine with short AA-MD runs before computing detailed structural or thermodynamic observables.

## When to Use This Skill

- Lipid bilayer composition and phase behavior studies (100–10,000 lipid systems) where microsecond equilibration is required and Martini 3 parameters exist for all components.
- Polymer morphology (block copolymer phase separation, polymer crystallization, entanglement networks) where the relevant length scale (Rg, lamellar period, mesh size) greatly exceeds the monomer size.
- Protein–membrane interactions, including insertion, aggregation, and lipid sorting, where both protein and membrane must be modeled at relevant scales simultaneously.
- Self-assembly of amphiphiles, surfactants, peptides, or supramolecular structures in the 10–500 nm range.
- Initial system assembly and equilibration before back-mapping to all-atom for free-energy calculations (providing an equilibrated starting structure is faster in CG).
- High-throughput screening of molecular architectures, compositions, or sequence variants where the ranking of multiple candidates by structural or thermodynamic proxy properties is more important than high-accuracy absolute values.

## When Not to Use This Skill

- **When atomistic chemical detail is required.** Proton transfer, tautomerism, stereochemistry, specific hydrogen-bonding networks, or rare functional-group interactions are erased by CG mapping. For enzymatic mechanisms or ligand binding affinity at > 1 kJ/mol accuracy, all-atom MD is required.
- **When quantitative kinetics are needed.** CG dynamics are faster than real dynamics by a factor that is system- and observable-dependent and cannot be determined a priori for a new system. Rate constants, diffusion coefficients, and relaxation times from CG trajectories are qualitatively indicative but not quantitatively predictive unless the time-scaling factor has been validated for the specific observable. [EXPERT REVIEW NEEDED]
- **When Martini parameters do not exist for a key component.** Running Martini with a hand-assigned bead type for an unparameterized molecule is not equivalent to a validated Martini simulation. Use bottom-up CG or all-atom MD instead.
- **When the system is below the CG resolution.** For systems < 5 nm (small molecules in solution, peptide monomers, single-strand DNA) CG-MD offers little advantage over all-atom MD and introduces unnecessary approximation.
- **When the free-energy surface must be reproduced quantitatively at the all-atom level.** CG potentials are effective potentials that reproduce structural distributions but not necessarily the free-energy landscape with sub-kT accuracy. For ABFE and RBFE calculations, all-atom FEP is required.

## Core Concepts

- **CG bead / interaction site:** A single particle representing 2–10 heavy atoms (and associated H). Characterized by mass (sum of mapped atoms), position (center of mass or geometric center of mapped atoms), and interaction type (bead type in Martini, or fitted LJ parameters in bottom-up CG). The bead position is not uniquely defined — center-of-mass and geometric-center mappings give different CG configurations.
- **Mapping operator:** The function M(r_AA) = r_CG that takes all-atom positions to CG positions. For center-of-mass mapping: r_CG = Σᵢ mᵢ rᵢ / Σᵢ mᵢ over atoms i in the bead. The mapping operator determines which atomistic configurations are indistinguishable at the CG level.
- **Martini force field:** A top-down CG force field parameterized to reproduce experimental free energies of transfer (oil/water partitioning, solvation free energies) and phase behavior. Martini 3 (2021) uses four main bead types (Tiny, Small, Regular, Big) with Q (charged), P (polar), N (intermediate), and C (apolar) subtypes, giving 18 interaction classes. Martini beads interact via Lennard-Jones (nonbonded) and harmonic bonds/angles/dihedrals (bonded). Martini 3 is the current recommended version for proteins, lipids, carbohydrates, and small molecules. Martini 2 is legacy but still widely used for membrane simulations.
- **Iterative Boltzmann inversion (IBI):** A bottom-up method that iteratively adjusts a tabulated pair potential V(r) to reproduce a target pair distribution function g_target(r) from all-atom MD. Starting from V₀(r) = −k_B T ln g_target(r), the potential is updated: V_{n+1}(r) = V_n(r) + k_B T ln(g_n(r)/g_target(r)). Converges when g_CG(r) ≈ g_target(r) to within tolerance. IBI reproduces the structural reference exactly at the parameterization state point but may be unphysical at different temperatures or concentrations (state-point transferability is poor).
- **Force matching (multiscale coarse-graining, MS-CG):** Minimizes the mean-square force residual between CG forces and the all-atom forces projected onto the CG mapping: min_{V_CG} Σ_t Σ_I |F_I^CG - Σᵢ∈I fᵢ^AA|². The CG forces are obtained by summing and projecting all-atom forces within each bead. Force matching systematically satisfies the variational principle for CG force fields and is exact in the limit of an infinitely flexible basis. Computationally expensive: requires forces from the all-atom reference at many configurations.
- **Relative entropy minimization:** Minimizes the Kullback-Leibler divergence between the all-atom Boltzmann distribution and the CG distribution: S_rel = ⟨ln(P_AA/P_CG)⟩_AA. This is equivalent to maximum likelihood estimation of the CG potential. For pairwise CG potentials, it gives the same result as IBI in the infinite-data limit; in practice it is more numerically stable and generalizes to non-pairwise CG force fields.
- **Elastic network model (ENM):** A simplified protein CG model where Cα atoms are connected by harmonic springs between all atom pairs within a cutoff (~1 nm). ENMs reproduce the low-frequency vibrational modes of a protein (normal modes, B-factors) with zero MD cost. Used for conformational analysis and allostery, not for dynamics.
- **Structure-based model (Gō model):** A CG protein model where the native contacts (atom pairs within a distance cutoff in the reference structure) are assigned attractive Lennard-Jones-like interactions, and non-native contacts are purely repulsive. Gō models reproduce folding thermodynamics and kinetics for small single-domain proteins that fold via two-state mechanisms. Not appropriate for proteins with frustrated landscapes, intrinsically disordered regions, or oligomerization.
- **Backmapping:** Reconstruction of an atomistic structure from a CG configuration. Methods: (1) geometric placement of all atoms based on fragment libraries + energy minimization (backward.py, BACKWARD protocol); (2) atom placement using inverse kinematics or machine-learned mappings. Backmapping introduces steric clashes that must be resolved by short energy minimization and AA-MD equilibration (typically 1–10 ns) before any AA-level analysis.
- **Effective temperature / CG time factor:** CG potentials are smoother than all-atom PES, so CG dynamics are faster. The effective speed-up factor f_t is estimated by comparing CG and AA lateral diffusion coefficients (D_CG / D_AA). For Martini 3 water at 310 K, f_t ≈ 4 for lateral diffusion of water. [EXPERT REVIEW NEEDED: f_t is not universal; it depends on the observable, temperature, and system composition.] Never report CG simulation time directly as real time without stating f_t and the observable used to calibrate it.
- **Martinize2:** The standard tool for generating Martini 3 CG topologies from all-atom PDB/GRO structures. Applies the elastic network (ElNeDyn) for protein secondary structure preservation. Outputs GROMACS-compatible `.top`, `.itp`, and `.gro` files.

## Key Workflows

### Workflow 1: Mapping atomistic protein + lipid bilayer to Martini 3

Generate a Martini 3 CG topology for a membrane protein system using martinize2 and insane (Martini membrane builder), then assemble the full CG system.

```bash
#!/bin/bash
# ── Martini 3 system construction ─────────────────────────────────────────────
# System: GPCR-like membrane protein in a POPC:POPE 3:1 bilayer with CG water
# Tools: martinize2, insane (Martini membrane assembler), GROMACS

set -euo pipefail

# ── Step 1: Generate CG protein topology with martinize2 ─────────────────────
# Input: AA protein structure (all-atom PDB from MD equilibration or X-ray)
# Output: cg_protein.gro + cg_protein.itp + molecule-specific .itp files

# Install: pip install martinize2
martinize2 \
    -f aa_protein_equil.pdb \       # All-atom input (must be properly equilibrated)
    -o cg_protein.gro \              # CG output coordinates
    -x cg_protein.pdb \              # CG PDB for visualization
    -p cg_protein.itp \              # Main protein ITP (molecule type + bonded)
    -ff martini3001 \                # Martini 3.0.0.1 (current stable)
    -nt \                            # N-terminus is not charged (adjust per system)
    -ct \                            # C-terminus is charged (adjust per system)
    -scfix \                         # Apply side chain corrections for Martini 3
    -cys auto \                      # Auto-detect disulfide bonds from Sγ–Sγ distance
    -elastic \                       # Apply ElNeDyn elastic network for protein rigidity
    -ef 500 \                        # Elastic network force constant (kJ/mol/nm²) [EXPERT REVIEW NEEDED]
    -el 0.5 \                        # Elastic network cutoff: 0.5 nm (secondary structure preservation)
    -eu 0.9 \                        # Elastic network upper cutoff: 0.9 nm (long-range elasticity)
    -ea 0 \                          # Elastic network: include all secondary structures
    -ep 0                            # Elastic network: include all atom pairs within cutoff

# Check output: expected bead count ~ N_residues × 1.5 (varies by residue type)
echo "CG protein beads: $(wc -l < cg_protein.gro)"
# Verify elastic network applied: grep "[ elastic_bonds ]" cg_protein.itp | wc -l

# ── Step 2: Build CG membrane + water box with insane ────────────────────────
# insane: Martini membrane builder for lipid bilayers with embedded proteins
# Install: pip install insane  (or download from cgmartini.nl)

# First: center the CG protein at z=0 (membrane midplane)
python3 - << 'PYEOF'
import MDAnalysis as mda
import numpy as np

u = mda.Universe("cg_protein.gro")
protein = u.select_atoms("all")
centroid = protein.center_of_geometry()
protein.translate(-centroid)   # center at origin
protein.write("cg_protein_centered.gro")
print(f"Protein centered. z-extent: {protein.positions[:,2].min():.1f}–{protein.positions[:,2].max():.1f} Å")
PYEOF

# Build the full membrane system
python3 -m insane \
    -f cg_protein_centered.gro \   # Protein to embed in the membrane
    -o cg_system.gro \              # Output system coordinates
    -p cg_system.top \              # Output topology (molecule counts)
    -pbc rectangular \              # Box type
    -x 14 -y 14 -z 12 \            # Box dimensions (nm); must contain protein + membrane
    -l POPC:3 -l POPE:1 \           # Lipid composition (upper and lower leaflets combined)
    -sol W \                        # Martini 3 standard water (W bead = 4 real water molecules)
    -salt 0.15 \                    # 0.15 M NaCl (added as NA and CL beads)
    -charge auto \                  # Automatically neutralize system charge
    -center                         # Center the protein in the membrane plane

echo "System built: cg_system.gro + cg_system.top"

# ── Step 3: Assemble the full topology ────────────────────────────────────────
# Edit cg_system.top to include the Martini 3 force field and protein ITP:
cat > topol.top << 'TOPEOF'
; Martini 3 topology for GPCR in POPC:POPE bilayer
#include "martini_v3.0.0.itp"         ; Martini 3 main force-field parameters
#include "martini_v3.0.0_ions_v1.itp" ; Ion parameters (NA, CL)
#include "martini_v3.0.0_solvents_v1.itp" ; Water (W, PW) parameters
#include "martini_v3.0.0_lipids_v1.itp"   ; Lipid parameters (POPC, POPE, ...)
#include "cg_protein.itp"                  ; Protein CG topology (from martinize2)

; System definition (copy from cg_system.top output by insane)
[ system ]
GPCR in POPC:POPE 3:1 bilayer, Martini 3

[ molecules ]
; The molecule counts are written by insane into cg_system.top
; Paste those [ molecules ] entries here, ensuring the protein is listed first.
TOPEOF

echo "Topology assembled: topol.top"
```

### Workflow 2: GROMACS Martini 3 mdp settings

Martini 3 requires different mdp parameters from standard biomolecular AMBER/CHARMM simulations. The key differences: no PME (reaction-field electrostatics), different cutoffs, and a longer timestep.

```ini
; ── martini3_production.mdp — Martini 3 production run ──────────────────────
; Official Martini 3 mdp parameters (cgmartini.nl, 2021)
; KEY DIFFERENCES from all-atom AMBER/CHARMM:
;   - No PME: Martini uses reaction-field electrostatics (εr = 15 for lipid membranes)
;   - Longer cutoffs: rcoulomb = rvdw = 1.1 nm (NOT 1.2 nm as in AMBER/CHARMM)
;   - No constraints: Martini bonds are stiff harmonics, not constrained
;   - Larger timestep: 20 fs (10× larger than AA-MD typical 2 fs)
;   - Different thermostat coupling: V-rescale with separate groups

integrator              = md
dt                      = 0.020          ; ps = 20 fs [EXPERT REVIEW NEEDED: reduce to 10 fs
                                          ; if simulation is unstable or using Gō models]
nsteps                  = 50000000       ; 1 µs (50M × 20 fs)

; ── Output frequencies ────────────────────────────────────────────────────────
nstxout                 = 0
nstvout                 = 0
nstfout                 = 0
nstlog                  = 50000          ; Log every 1 ns
nstenergy               = 5000           ; Energies every 100 ps
nstxout-compressed      = 5000           ; .xtc every 100 ps (CG trajectories are smaller)

; ── Nonbonded interactions — CRITICAL: different from all-atom ───────────────
cutoff-scheme           = Verlet
nstlist                 = 20             ; Longer neighbor list update (20 steps vs 10 in AA)

; LJ cutoff
rcoulomb                = 1.1            ; nm — NOT 1.2 nm; Martini is parameterized at 1.1 nm
rvdw                    = 1.1            ; nm — must equal rcoulomb for Verlet scheme
vdwtype                 = cutoff
vdw-modifier            = Potential-shift  ; Shift LJ potential smoothly to 0 at rvdw
                                            ; NOTE: force-switch is NOT recommended for Martini 3

; Electrostatics — reaction-field (NOT PME)
coulombtype             = reaction-field
epsilon-rf              = 15             ; Dielectric constant of reaction field
                                          ; εr = 15 for lipid membranes and protein systems
                                          ; εr = infinity (0 in GROMACS) for explicit CG water runs
                                          ; [EXPERT REVIEW NEEDED: verify εr for your system]
epsilon-r               = 15             ; Coulomb dielectric constant

; ── Bonded interactions ────────────────────────────────────────────────────────
; Martini bonds are stiff harmonics — NO CONSTRAINTS (unlike AA where h-bonds are constrained)
constraints             = none
constraint-algorithm    = lincs          ; not used but set for completeness

; ── Thermostat — separate groups essential for membrane systems ───────────────
tcoupl                  = V-rescale
tc-grps                 = Protein  Membrane  W_and_ions
; Separate coupling groups: Protein + Membrane + solvent
; CRITICAL: if Protein couples with W, kinetic energy can flow between
; the frozen membrane and the fast-equilibrating water, causing instability.
tau-t                   = 1.0     1.0      1.0  ; ps (longer tau than AA: 0.1 → 1.0)
ref-t                   = 310.0   310.0    310.0 ; K (physiological)

; ── Barostat — semiisotropic for membrane systems ────────────────────────────
pcoupl                  = Parrinello-Rahman
pcoupltype              = semiisotropic   ; xy coupled, z independent
tau-p                   = 12.0            ; ps (longer than AA: 2.0 → 12.0) [EXPERT REVIEW NEEDED]
ref-p                   = 1.013  1.013    ; bar [lateral, normal]
compressibility         = 3.0e-4  3.0e-4  ; bar⁻¹; CG water is more compressible than real water
                                           ; [EXPERT REVIEW NEEDED: value varies with CG water model]

; ── Velocity initialization ───────────────────────────────────────────────────
gen-vel                 = no             ; Continuation from equilibration
continuation            = yes

; GROMACS notes for Martini:
; 1. Do NOT use PME with Martini — reaction-field is the Martini parameterization assumption.
; 2. The Verlet list with nstlist=20 is correct for CG (fewer degrees of freedom → slower phase-space exploration).
; 3. The elastic network in the protein ITP (from martinize2 -elastic) is ESSENTIAL:
;    without it, alpha-helices and beta-sheets unfold on microsecond timescales in Martini 3.
```

```bash
#!/bin/bash
# ── Martini 3 equilibration + production pipeline ─────────────────────────────

set -euo pipefail
GMX="gmx"

# ── Energy minimization ───────────────────────────────────────────────────────
# Martini systems often have bad initial geometries from insane; minimize first.
${GMX} grompp -f em_martini.mdp -c cg_system.gro -p topol.top -o em.tpr -maxwarn 1
# -maxwarn 1: insane-generated topologies often have 1 benign warning about
# the order of molecule types in [ molecules ] vs. coordinates.
${GMX} mdrun -v -deffnm em -ntmpi 1 -ntomp 8

grep "Maximum force" em.log | tail -1
# Martini EM converges at higher forces than AA; accept < 10,000 kJ/mol/nm.
# If EM does not converge: check for overlapping lipid/protein beads (reduce dt to 0.001 fs for EM).

# ── NVT equilibration (5 ns with position restraints on protein) ─────────────
${GMX} grompp -f nvt_martini.mdp -c em.gro -p topol.top -o nvt.tpr \
    -define "-DPOSRES_PROT" -maxwarn 1
${GMX} mdrun -v -deffnm nvt -ntmpi 1 -ntomp 8

# ── NPT equilibration (50–200 ns, release restraints gradually) ──────────────
# Key: start with Berendsen for fast density convergence, then switch to P-R.
${GMX} grompp -f npt_berendsen.mdp -c nvt.gro -t nvt.cpt -p topol.top \
    -o npt_ber.tpr -maxwarn 1
${GMX} mdrun -v -deffnm npt_ber -ntmpi 1 -ntomp 8

${GMX} grompp -f npt_pr.mdp -c npt_ber.gro -t npt_ber.cpt -p topol.top \
    -o npt_pr.tpr -maxwarn 1
${GMX} mdrun -v -deffnm npt_pr -ntmpi 1 -ntomp 8

# Check membrane area per lipid (convergence criterion for membrane equilibration):
${GMX} energy -f npt_pr.edr -o box_xy.xvg << 'EOF'
Box-X
0
EOF
python3 -c "
import numpy as np
data = np.loadtxt('box_xy.xvg', comments=['#','@'])
t, Lx = data[:,0], data[:,1]  # nm
n_lipids = 512  # adjust to your system
APL = Lx**2 / (n_lipids / 2)  # nm² per lipid (both leaflets)
# Wait until APL is stable; typical POPC at 310 K: ~0.64–0.68 nm² [EXPERT REVIEW NEEDED]
print(f'Mean APL (last 50 ns): {APL[-500:].mean():.3f} ± {APL[-500:].std():.3f} nm²')
"

# ── Production ────────────────────────────────────────────────────────────────
${GMX} grompp -f martini3_production.mdp -c npt_pr.gro -t npt_pr.cpt \
    -p topol.top -o production.tpr -maxwarn 1
${GMX} mdrun -v -deffnm production -ntmpi 1 -ntomp 16 -maxh 23.5
```

### Workflow 3: Bonded parameterization via iterative Boltzmann inversion

IBI derives CG bonded potentials (bond, angle, dihedral) from all-atom reference distributions. Produces tabulated potentials that match target distributions exactly at the parameterization state point.

```python
"""
IBI for bonded CG potential derivation.

Workflow:
1. Run AA-MD simulation (100–500 ns) and extract bond/angle distributions.
2. Start CG with V₀(r) = −kT ln g(r) as the initial potential.
3. Iteratively correct: V_{n+1} = V_n + kT ln(g_n/g_target).
4. Converge when max|g_CG − g_target| < tolerance (typically 0.01).

This workflow demonstrates bonded IBI for a single bond type.
Nonbonded IBI uses pair RDFs g(r) and tabulated pair potentials.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import subprocess
from pathlib import Path


def compute_distribution_from_traj(
    traj_file: str,
    topo_file: str,
    selection: str,
    dist_type: str = "bond",       # "bond", "angle", or "dihedral"
    n_bins: int = 200,
    r_min: float = None,
    r_max: float = None,
) -> tuple:
    """
    Extract bond/angle/dihedral distribution from trajectory using MDAnalysis.

    Returns (bin_centers, normalized_distribution).
    """
    import MDAnalysis as mda
    from MDAnalysis.analysis import distances

    u = mda.Universe(topo_file, traj_file)
    atoms = u.select_atoms(selection)

    if len(atoms) not in (2, 3, 4):
        raise ValueError(f"Selection '{selection}' returned {len(atoms)} atoms; "
                         "need 2 (bond), 3 (angle), or 4 (dihedral)")

    values = []
    for ts in u.trajectory:
        pos = atoms.positions   # Å

        if dist_type == "bond":
            d = np.linalg.norm(pos[1] - pos[0]) / 10.0  # Å → nm
            values.append(d)
        elif dist_type == "angle":
            v1 = pos[0] - pos[1]
            v2 = pos[2] - pos[1]
            cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            theta = np.degrees(np.arccos(np.clip(cos_theta, -1, 1)))
            values.append(theta)
        elif dist_type == "dihedral":
            b1 = pos[1] - pos[0]
            b2 = pos[2] - pos[1]
            b3 = pos[3] - pos[2]
            n1 = np.cross(b1, b2)
            n2 = np.cross(b2, b3)
            phi = np.degrees(np.arctan2(
                np.dot(np.cross(n1, n2), b2 / np.linalg.norm(b2)),
                np.dot(n1, n2)
            ))
            values.append(phi)

    values = np.array(values)
    if r_min is None:
        r_min = values.min()
    if r_max is None:
        r_max = values.max()

    hist, edges = np.histogram(values, bins=n_bins, range=(r_min, r_max), density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    # Smooth to reduce noise in the target distribution
    hist = gaussian_filter1d(hist, sigma=2)
    hist = np.maximum(hist, 1e-10)   # avoid log(0)
    return centers, hist / hist.max()  # normalize to peak = 1


def ibi_iteration(
    target_centers: np.ndarray,    # bin centers from AA reference
    target_dist: np.ndarray,       # AA target distribution (normalized)
    cg_dist: np.ndarray,           # CG distribution from current iteration
    current_potential: np.ndarray, # current tabulated potential (kJ/mol)
    kT: float = 2.479,             # kJ/mol at 298 K
    alpha: float = 1.0,            # update step size (reduce to 0.5 if oscillating)
    smooth_sigma: float = 1.5,     # smoothing width for the potential update
) -> np.ndarray:
    """
    One iteration of IBI: V_{n+1} = V_n + alpha * kT * ln(g_n / g_target).

    The log ratio is regularized to avoid divergence where g_target or g_CG → 0.
    Smoothing is applied to the correction to prevent high-frequency noise.
    """
    eps = 1e-8
    correction = alpha * kT * np.log((cg_dist + eps) / (target_dist + eps))
    # Smooth the correction to avoid oscillatory potentials
    correction = gaussian_filter1d(correction, sigma=smooth_sigma)
    new_potential = current_potential + correction
    return new_potential


def run_ibi(
    aa_traj_file: str,
    aa_topo_file: str,
    selection: str,
    dist_type: str = "bond",
    n_iterations: int = 50,
    convergence_tol: float = 0.01,
    kT: float = 2.479,              # kJ/mol at 298 K
) -> dict:
    """
    Full IBI loop for one CG bonded interaction type.

    In practice: each iteration requires running a CG simulation (not shown here)
    to obtain g_n(r). The workflow below sketches the structure; integrate with
    your CG simulation runner.
    """
    print(f"IBI: computing AA target distribution for '{selection}' ({dist_type})")
    r, g_target = compute_distribution_from_traj(
        aa_traj_file, aa_topo_file, selection, dist_type
    )

    # Initial potential: V₀ = −kT ln g_target (the direct Boltzmann inversion)
    V = -kT * np.log(g_target + 1e-10)
    V -= V.min()   # set minimum to 0 (reference)

    history = []
    for iteration in range(n_iterations):
        # ── In practice: write V to a GROMACS tabulated potential file,
        #    run a CG simulation, extract g_CG from the CG trajectory ──────────
        # Here we symbolically represent this step:
        # g_CG = run_cg_sim_and_extract_distribution(r, V, ...)

        # Placeholder for demonstration; replace with actual CG run:
        g_CG = g_target.copy()   # REPLACE with actual CG distribution

        max_error = np.max(np.abs(g_CG - g_target))
        history.append(max_error)
        print(f"  Iteration {iteration+1:3d}: max|g_CG − g_target| = {max_error:.4f}")

        if max_error < convergence_tol:
            print(f"  CONVERGED after {iteration+1} iterations")
            break

        V = ibi_iteration(r, g_target, g_CG, V, kT=kT)

    # Write final tabulated potential (GROMACS format)
    output = np.column_stack([r, V, -np.gradient(V, r)])   # r, V, F=-dV/dr
    np.savetxt("table_ibi.xvg", output,
               header="r (nm)    V (kJ/mol)    F (kJ/mol/nm)")
    print("Written: table_ibi.xvg (use with tablep in GROMACS for bonded terms)")

    return {"r": r, "V_final": V, "g_target": g_target, "convergence": history}
```

### Workflow 4: Backmapping CG to all-atom resolution

Backmapping reconstructs atomistic detail from a CG configuration. Always required before any all-atom analysis, free-energy calculation, or docking study using CG-equilibrated structures.

```python
"""
Backmapping workflow using the BACKWARD protocol (backward.py).

BACKWARD places all-atom fragments based on the positions of their parent
CG beads, using a library of all-atom fragment templates. After placement,
steric clashes are resolved by energy minimization + short MD equilibration.

Reference: Wassenaar et al., JCTC 2014 (https://doi.org/10.1021/ct400617g)
Install: pip install backward  (or clone github.com/Tsjerk/Backward)
"""

import subprocess
import shutil
from pathlib import Path


def backmap_martini_to_atomistic(
    cg_gro: str,               # CG structure to backmap
    cg_top: str,               # CG topology
    output_prefix: str = "backmapped",
    force_field: str = "charmm36m",   # All-atom FF for the backmapped system
    water_model: str = "tip3p",        # Must match force field
    n_em_steps: int = 5000,
    n_md_equil_ns: float = 1.0,       # Short AA equilibration after backmapping
    gromacs_gmx: str = "gmx",
) -> str:
    """
    Backmap a Martini CG structure to all-atom resolution.

    Steps:
    1. backward.py: place all-atom fragments at CG bead positions
    2. Energy minimization: resolve steric clashes from fragment placement
    3. Short NVT + NPT equilibration: relax local geometry before analysis

    Returns path to the backmapped equilibrated GRO file.

    CRITICAL: The backmapped structure has large steric clashes immediately
    after step 1. Skipping steps 2 and 3 produces an unphysical structure
    with forces > 10^6 kJ/mol/nm that will crash any subsequent simulation.
    """
    Path("backmap_work").mkdir(exist_ok=True)

    # Step 1: backward.py — fragment placement
    result = subprocess.run([
        "python", "-m", "backward",
        "-f", cg_gro,
        "-p", cg_top,
        "-o", "backmap_work/placed.gro",
        "-kick", "0.05",    # Random displacement to break degeneracy (nm)
        "-ff", force_field,
        "-water", water_model,
    ], capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"backward.py failed:\n{result.stderr[-2000:]}")
    print("Fragment placement complete. Checking for clashes...")

    # Quick clash check: minimum pairwise distance
    import numpy as np
    gro_data = []
    with open("backmap_work/placed.gro") as f:
        lines = f.readlines()
    for line in lines[2:-1]:   # skip header and box line
        try:
            x = float(line[20:28])
            y = float(line[28:36])
            z = float(line[36:44])
            gro_data.append([x, y, z])
        except ValueError:
            pass
    pos = np.array(gro_data)
    if len(pos) > 1:
        diffs = pos[np.newaxis,:,:] - pos[:,np.newaxis,:]
        dists = np.linalg.norm(diffs, axis=-1)
        np.fill_diagonal(dists, np.inf)
        min_dist = dists.min()
        print(f"Min pairwise distance after placement: {min_dist*10:.2f} Å "
              f"({'OK' if min_dist > 0.1 else 'WARNING: severe clash'})")

    # Step 2: Energy minimization
    subprocess.run([
        gromacs_gmx, "grompp",
        "-f", "templates/em_backmap.mdp",   # EM with constraints=none, emtol=100
        "-c", "backmap_work/placed.gro",
        "-p", "aa_topol.top",               # All-atom topology (generated separately)
        "-o", "backmap_work/em.tpr",
        "-maxwarn", "2",   # Expect 1-2 warnings from the placed structure
    ], check=True)

    subprocess.run([
        gromacs_gmx, "mdrun", "-v",
        "-deffnm", "backmap_work/em",
        "-ntmpi", "1", "-ntomp", "8",
    ], check=True)

    max_force_line = subprocess.run(
        ["grep", "Maximum force", "backmap_work/em.log"],
        capture_output=True, text=True
    ).stdout.strip().split("\n")[-1]
    print(f"EM result: {max_force_line}")

    # Step 3: Short NVT + NPT equilibration
    for stage in ["nvt", "npt"]:
        subprocess.run([
            gromacs_gmx, "grompp",
            "-f", f"templates/{stage}_backmap.mdp",
            "-c", f"backmap_work/em.gro" if stage == "nvt" else "backmap_work/nvt.gro",
            "-t", f"backmap_work/{'em' if stage == 'nvt' else 'nvt'}.cpt",
            "-p", "aa_topol.top",
            "-o", f"backmap_work/{stage}.tpr",
            "-maxwarn", "0",
        ], check=True)

        subprocess.run([
            gromacs_gmx, "mdrun", "-v",
            "-deffnm", f"backmap_work/{stage}",
            "-ntmpi", "1", "-ntomp", "8",
        ], check=True)

    output_file = f"{output_prefix}_equilibrated.gro"
    shutil.copy("backmap_work/npt.gro", output_file)
    print(f"Backmapping complete: {output_file}")
    print("IMPORTANT: Run at least 10–50 ns of AA-MD before any quantitative analysis.")
    return output_file
```

### Workflow 5: Validation against all-atom MD and experiment

CG models must be validated structurally and thermodynamically before being used for production. Validation compares CG outputs to all-atom MD (structural) and to experiment (thermodynamic).

```python
"""
Quantitative validation of a CG membrane simulation against:
1. AA-MD RDF / structural observables (structural accuracy)
2. Experimental area per lipid, bilayer thickness, lateral diffusion (thermodynamic accuracy)
3. Self-consistency: reproducibility of observables across independent CG runs

All validation should be performed at the parameterization state point (T, P, composition)
before using the CG model at different conditions.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import MDAnalysis as mda
from MDAnalysis.analysis import rdf as mdardf


def validate_cg_membrane(
    cg_traj: str,
    cg_top: str,
    aa_traj: str,
    aa_top: str,
    n_lipids: int,
    temp_K: float = 310.0,
    exp_apl_nm2: float = 0.64,       # Experimental APL for POPC at 310 K (nm²) [EXPERT REVIEW NEEDED]
    exp_thickness_nm: float = 3.8,    # Experimental bilayer thickness for POPC (nm) [EXPERT REVIEW NEEDED]
    exp_dlat_cm2_s: float = 7.5e-8,  # Experimental lateral diffusion coefficient [EXPERT REVIEW NEEDED]
) -> dict:
    """
    Validate a Martini 3 POPC bilayer simulation against AA-MD and experiment.
    """
    results = {}

    # ── 1. Area per lipid ─────────────────────────────────────────────────────
    u_cg = mda.Universe(cg_top, cg_traj)
    box_x = []
    for ts in u_cg.trajectory[-500:]:  # last 500 frames
        box_x.append(ts.dimensions[0])  # Å
    apl = np.array(box_x)**2 / (n_lipids / 2) / 100.0  # Å² → nm²
    apl_mean = apl.mean()
    apl_std  = apl.std()

    print(f"Area per lipid: {apl_mean:.3f} ± {apl_std:.3f} nm²")
    print(f"  Experimental ({temp_K:.0f} K): {exp_apl_nm2:.3f} nm²")
    print(f"  Deviation: {abs(apl_mean - exp_apl_nm2)*100/exp_apl_nm2:.1f}%")
    results["apl_nm2"] = apl_mean
    results["apl_std"] = apl_std

    # ── 2. Bilayer thickness (P–P distance) ───────────────────────────────────
    # For Martini POPC: PO4 bead = phosphate group of upper/lower leaflet
    upper_P = u_cg.select_atoms("resname POPC and name PO4 and prop z > 0")
    lower_P = u_cg.select_atoms("resname POPC and name PO4 and prop z < 0")

    thickness_list = []
    for ts in u_cg.trajectory[-500:]:
        z_upper = upper_P.positions[:, 2].mean()
        z_lower = lower_P.positions[:, 2].mean()
        thickness_list.append(abs(z_upper - z_lower) / 10.0)  # Å → nm

    thickness = np.array(thickness_list)
    print(f"\nBilayer thickness (PO4-PO4): {thickness.mean():.3f} ± {thickness.std():.3f} nm")
    print(f"  Experimental: {exp_thickness_nm:.3f} nm")
    print(f"  Deviation: {abs(thickness.mean() - exp_thickness_nm)*100/exp_thickness_nm:.1f}%")
    results["thickness_nm"] = thickness.mean()

    # ── 3. Lateral diffusion coefficient (MSD) ────────────────────────────────
    from MDAnalysis.analysis.msd import EinsteinMSD
    lipids = u_cg.select_atoms("resname POPC and name PO4")
    msd_analysis = EinsteinMSD(lipids, select="all", msd_type="xy",
                                fft=True)   # lateral MSD only
    msd_analysis.run(start=-min(1000, len(u_cg.trajectory)//2))

    # Fit linear region of MSD (t = 50–500 ns)
    times = msd_analysis.times / 1000.0   # ps → ns
    msd = msd_analysis.results.timeseries / 100.0  # Å² → nm²

    # Find linear region: skip first 10% and last 10%
    i_start = len(times) // 10
    i_end = len(times) - len(times) // 10
    coeffs = np.polyfit(times[i_start:i_end], msd[i_start:i_end], 1)
    D_cg = coeffs[0] / 4.0 * 1e-14  # nm²/ns → cm²/s (2d: MSD=4Dt)

    # Apply Martini time-scale factor (4x for lateral diffusion) [EXPERT REVIEW NEEDED]
    D_real = D_cg / 4.0
    print(f"\nLateral diffusion coefficient:")
    print(f"  CG (raw):   {D_cg:.2e} cm²/s")
    print(f"  CG (×1/4):  {D_real:.2e} cm²/s (Martini time-scale correction)")
    print(f"  Experimental: {exp_dlat_cm2_s:.2e} cm²/s")
    print(f"  [NOTE: Time-scale factor of 4 is an approximation; "
          "validate for your specific lipid composition and temperature]")
    results["D_lateral_cm2_s"] = D_real

    # ── 4. Structural: CG vs AA-MD RDF comparison ────────────────────────────
    # Compare PO4-PO4 (upper leaflet) RDF between CG and AA P-P RDF
    u_aa = mda.Universe(aa_top, aa_traj)
    aa_P = u_aa.select_atoms("resname POPC and name P")

    for label, u, sel in [("CG", u_cg, "resname POPC and name PO4"),
                           ("AA", u_aa, "resname POPC and name P")]:
        atoms = u.select_atoms(sel)
        rdf_analysis = mdardf.InterRDF(atoms, atoms, nbins=200,
                                       range=(0, 2.0), exclusion_block=(1, 1))
        rdf_analysis.run(start=max(0, len(u.trajectory)-500))
        r = rdf_analysis.results.bins / 10.0  # Å → nm
        g = rdf_analysis.results.rdf
        results[f"rdf_{label}_r"] = r
        results[f"rdf_{label}_g"] = g

    # Compute RDF agreement: integral of |g_CG - g_AA| over r < 1.5 nm
    r_aa = results["rdf_AA_r"]
    g_aa = results["rdf_AA_g"]
    r_cg = results["rdf_CG_r"]
    g_cg = np.interp(r_aa, r_cg, results["rdf_CG_g"])
    rdf_mae = np.trapz(np.abs(g_cg - g_aa), r_aa)
    print(f"\nRDF agreement (∫|g_CG - g_AA|dr, 0-1.5 nm): {rdf_mae:.3f}")
    print(f"  < 0.1: good structural agreement")
    print(f"  > 0.3: poor agreement; reconsider bead mapping or force field")

    # Summary
    print(f"\n{'='*50}")
    print(f"Validation summary:")
    print(f"  APL:        {apl_mean:.3f} nm² (exp: {exp_apl_nm2:.3f})")
    print(f"  Thickness:  {thickness.mean():.3f} nm (exp: {exp_thickness_nm:.3f})")
    print(f"  D_lat:      {D_real:.2e} cm²/s (exp: {exp_dlat_cm2_s:.2e})")
    print(f"  RDF MAE:    {rdf_mae:.3f}")
    print(f"{'='*50}")
    return results
```

### Workflow 6: Bottom-up CG of a polymer with LAMMPS (Kremer-Grest model)

Kremer-Grest is the standard bottom-up CG model for polymer dynamics. Each monomer is represented as one bead with WCA (truncated-shifted LJ) repulsion and FENE bonds.

```lammps
# ── Kremer-Grest polymer melt — LAMMPS ────────────────────────────────────────
# System: 200 chains of 100 monomers (N=100); 20,000 beads total
# CG level: 1 bead per monomer (backbone + side group → 1 CG site)
# Application: polymer entanglement, diffusion, viscoelasticity
# Reference: Kremer & Grest, JCP 92, 5057 (1990)

units       lj             # Lennard-Jones reduced units (m=1, σ=1, ε=1, τ=1)
                            # Mapping to real units: σ_LJ ≈ 0.47 nm (C-C bead) [EXPERT REVIEW NEEDED]
                            # ε_LJ ≈ 1.0 kJ/mol; τ_LJ ≈ 1.5 ps [EXPERT REVIEW NEEDED]
dimension   3
boundary    p p p

atom_style  molecular       # tracks bonds and angles (needed for chain connectivity)
read_data   polymer_200x100.lammps   # generated by fmatch or from a polymerizer script

# ── Nonbonded: WCA (truncated and shifted LJ at r_c = 2^{1/6}σ) ──────────────
# WCA is purely repulsive: the attractive tail of LJ is removed.
# This models excluded-volume interactions only (athermal solvent / melt conditions).
# To add cohesion (e.g., for phase separation): use full LJ with rc = 2.5σ.
pair_style   lj/cut 2.5     # rc = 2.5σ for full LJ (use 1.1225σ for WCA)
pair_coeff   * * 1.0 1.0    # ε=1, σ=1 for all bead pairs
pair_modify  shift yes      # Shift potential to 0 at cutoff (required for WCA)
# For WCA: pair_style lj/cut 1.1225; pair_coeff * * 1.0 1.0; pair_modify shift yes

# ── Bonded: FENE (finitely extensible nonlinear elastic) + WCA ────────────────
# FENE bond: U_FENE = -0.5 K R₀² ln(1 - (r/R₀)²) + LJ_WCA(r)
# K = 30ε/σ², R₀ = 1.5σ (standard Kremer-Grest parameters)
bond_style   fene
bond_coeff   1 30.0 1.5 1.0 1.0   # K R₀ ε σ
# FENE breaks if r > R₀; this is a physical bond-breaking criterion.
# Monitor for FENE failures in the log (indicate timestep too large or bad initial structure).

special_bonds lj 0 1 1     # 1-2 excluded, 1-3 and 1-4 at full strength
                             # NOTE: For FENE bonds, GROMACS convention differs; check docs.

# ── Timestep and integrator ────────────────────────────────────────────────────
timestep    0.005           # 0.005 τ_LJ ≈ 7.5 fs; standard for Kremer-Grest
                             # Reduce to 0.002 τ for dense melts above T_g [EXPERT REVIEW NEEDED]

# ── Langevin thermostat ────────────────────────────────────────────────────────
fix         thermo all langevin 1.0 1.0 1.0 12345
# Temperature = 1.0 ε/k_B (above Tg for all common bead-spring models)
# Friction coefficient = 1.0 τ⁻¹ (damping time = 1 τ_LJ)
# For viscosity/diffusion: use NVE + compute pressure tensor (fix nve, not Langevin)

fix         nve all nve
thermo      1000
thermo_style custom step temp press etotal epair ebond

# ── Output ────────────────────────────────────────────────────────────────────
dump        1 all custom 1000 polymer_traj.lammpsdump id mol type x y z
dump_modify 1 sort id

run         10000000        # 10M steps = 50,000 τ ≈ 75 ns real time (at σ≈0.47nm mapping)
                             # Entanglement time for N=100: ~1000 τ (well within run) [EXPERT REVIEW NEEDED]
```

### Workflow 7: Enhanced sampling with PLUMED on a CG system

CG-MD + PLUMED enables enhanced sampling of slow processes (membrane fusion, protein insertion, vesicle formation) that are slow even at CG resolution.

```bash
# ── plumed_cg_metad.dat — metadynamics on a CG membrane system ───────────────
# System: lipid bilayer, tracking the z-insertion depth of a CG peptide
# CV: z-distance between peptide center-of-mass and membrane center
# Note: CG PLUMED integration uses the same plumed.dat format; only the
# atom indices and CV definitions change from the all-atom case.

UNITS ENERGY=kj/mol LENGTH=nm TIME=ps

# CG peptide center-of-mass (all peptide backbone and sidechain beads)
PEP_COM: COM ATOMS=1-45 WEIGHTS=MASS

# Membrane center (PO4 beads of all POPC lipids, both leaflets)
MEM_COM: COM ATOMS=50-5100:10 WEIGHTS=MASS    # PO4 bead index for each POPC
# ADJUST the atom indices to match your topology; use gmx dump -s prod.tpr | grep PO4

# Z-insertion CV: z-component only
Z_INSERT: DISTANCE ATOMS=MEM_COM,PEP_COM COMPONENTS
Z_CV: COMBINE ARG=Z_INSERT.z COEFFICIENTS=1 PERIODIC=NO

# Well-tempered metadynamics on peptide insertion depth
METAD ...
    LABEL=metad
    ARG=Z_CV
    SIGMA=0.15           # nm; ~0.5 × equilibrium CV fluctuation
    HEIGHT=0.5           # kJ/mol; lower than AA case (CG landscape is smoother)
    BIASFACTOR=10        # gamma; appropriate for barriers < 40 kJ/mol at CG resolution
    PACE=200             # deposit hill every 200 CG steps (200 × 20 fs = 4 ps)
    GRID_MIN=-4.0        # nm; minimum z range
    GRID_MAX=4.0         # nm; maximum z range
    GRID_BIN=400
    CALC_RCT
    TEMP=310
... METAD

PRINT ARG=Z_CV,metad.bias FILE=colvar_cg_metad.dat STRIDE=50
```

```python
"""
Time-scale interpretation of CG enhanced sampling results.

CRITICAL: The free-energy surface from CG-PLUMED is a CG FES, not the AA FES.
- The barrier heights may differ from the all-atom FES by a factor related to
  the smoothing of the CG potential.
- The CG FES can be used qualitatively (which state is stable, which is metastable)
  but NOT quantitatively for kT-level accuracy of ΔG.
- For quantitative binding/insertion free energies: backmap to AA and run AA FEP.
"""

import numpy as np
import matplotlib.pyplot as plt


def interpret_cg_fes(
    colvar_file: str,
    bias_col: int = 2,
    cv_col: int = 1,
    temp_K: float = 310.0,
    bias_factor: float = 10.0,
    time_scale_factor: float = 4.0,   # CG time → real time [EXPERT REVIEW NEEDED]
    n_skip: int = 100,
) -> None:
    """
    Compute CG FES from WTM COLVAR and annotate with uncertainty caveats.

    time_scale_factor: how many times faster CG dynamics are than AA.
    For lateral diffusion in Martini 3 membranes: ~4x.
    For peptide insertion in a membrane: NOT well established. [EXPERT REVIEW NEEDED]
    """
    kT = 8.314e-3 * temp_K
    data = np.loadtxt(colvar_file, comments="#")[n_skip:]
    time_cg_ps = data[:, 0]
    cv = data[:, cv_col]
    bias = data[:, bias_col]

    # WTM reweighting
    log_w = bias / kT
    log_w -= log_w.max()
    weights = np.exp(log_w)

    hist, edges = np.histogram(cv, bins=100, weights=weights, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    fes = -kT * np.log(hist + 1e-40)
    fes -= fes.min()

    real_time_us = time_cg_ps[-1] / time_scale_factor / 1e6
    print(f"CG simulation length: {time_cg_ps[-1]/1e6:.1f} µs (CG time)")
    print(f"Equivalent real time: {real_time_us:.2f} µs (time factor: {time_scale_factor}x)")
    print(f"[NOTE: time-scale factor is uncertain for this observable]")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(centers, fes, lw=2)
    ax.set_xlabel("Insertion depth Z (nm)")
    ax.set_ylabel("Free energy (kJ/mol)")
    ax.set_title("CG membrane insertion FES\n"
                 "[CAUTION: CG FES; barriers are qualitative only]")
    ax.fill_between(centers, fes - kT, fes + kT, alpha=0.2,
                    label=f"±kT = ±{kT:.1f} kJ/mol uncertainty")
    ax.legend()
    plt.tight_layout()
    plt.savefig("cg_insertion_fes.png", dpi=150)
    print("Saved: cg_insertion_fes.png")
    print("\nFor quantitative ΔG: backmap representative CG structures and")
    print("run AA-MD + FEP/umbrella sampling to recover atomistic accuracy.")
```

## Common Failure Modes

| Failure | Symptom | Likely Cause | Diagnostic | Fix |
|---------|---------|--------------|------------|-----|
| Invalid bead mapping | CG trajectory shows unphysical geometry; bonds stretch or compress irregularly; simulations crash | Atom-to-bead assignment places bonded atoms in different beads without a proper CG bond; or center-of-mass positions lead to inconsistent bond topology | Print bead connectivity and verify each CG bond corresponds to a chemical bond or a small structural unit; visualize one molecule in VMD/OVITO at AA and CG levels simultaneously | Redesign the mapping: bonded beads should map to connected chemical fragments; always verify the mapping with a single-molecule test before a full system |
| Wrong bead type assignment in Martini | Membrane too fluid (APL too large) or too rigid (APL too small); peptide/protein mispartitioned between water and membrane | Incorrect polarity or charge assignment for a bead type; e.g., using C1 (apolar) for a hydroxyl group that should be P2 (polar) | Compare bead-type partitioning to the Martini 3 decision tree; compute octanol/water partition coefficients from short simulations and compare to experiment | Reassign bead types following the Martini 3 parameterization guide; validate with small-molecule partitioning before membrane insertion studies |
| Missing elastic network for protein | Alpha-helices unfold within 100 ns of CG simulation; secondary structure collapses; protein becomes a random coil | Elastic network (ElNeDyn) was not applied during martinize2; or elastic network cutoff is too short | Compute secondary structure assignment (DSSP) on the CG trajectory via `gmx do_dssp -sc`; if helix content drops by > 10% within 100 ns, the elastic network is absent or too weak | Re-run martinize2 with `-elastic -ef 500 -el 0.5 -eu 0.9`; verify `[ elastic_bonds ]` section in the protein ITP; increase `ef` if secondary structure is still lost |
| PME used instead of reaction-field | Simulation is significantly slower than expected; GROMACS prints "PME" in the log; electrostatic behavior of Martini system is wrong | `coulombtype = PME` set in the Martini mdp instead of `reaction-field` | Check `grep coulombtype production.mdp`; if PME, the Martini parameterization assumption is violated | Replace with `coulombtype = reaction-field` and `epsilon-rf = 15`; also set `epsilon-r = 15`; rerun from NPT equilibration |
| Poor bonded parameterization from IBI | Bond lengths or angles in CG-MD differ from AA reference distributions by > 5–10%; protein or lipid acyl chain length is wrong | IBI converged to wrong minimum; initial guess V₀ was far from the true potential; tabulated potential has noise from limited sampling | Compare bond/angle distributions between CG-MD (after IBI) and AA reference using MDAnalysis; plot overlaid histograms | Run more AA-MD frames to improve the reference distribution; apply more aggressive smoothing to the IBI correction; reduce the step size alpha from 1.0 to 0.5 |
| Incorrect time-scale interpretation | Reported CG diffusion coefficient of lipids is 4× too fast vs. experiment; or a CG folding rate is interpreted as the real folding rate | No time-scale correction applied; CG time is naively equated with real time | Compare CG lateral diffusion coefficient to experimental value; ratio gives f_t for that observable at that condition | Always apply the observable-specific time-scale factor and report both CG time and corrected real time; caveat that f_t is uncertain unless explicitly calibrated for the observable |
| Backmapping without equilibration | Backmapped structure has bond lengths 10× too large; energy minimization crashes; LINCS warnings in every step | AA topology does not match the backmapped coordinate file; or MM relaxation step was skipped after backward.py | Run `gmx grompp` on the backmapped structure and check for topology/coordinate mismatches; plot minimum pairwise distance histogram immediately after placement | Always run energy minimization then 1–10 ns NVT + NPT equilibration after backmapping before any quantitative AA analysis |
| Non-transferable IBI potential | CG simulation at 350 K (outside parameterization T) gives wrong structural properties; a change in lipid composition changes the CG results unexpectedly | IBI and force-matching potentials are state-point specific; thermodynamic consistency is not guaranteed outside the parameterization state | Compare RDFs at the new state point to AA reference at the same conditions; if they diverge, the potential is not transferable | Use top-down CG (Martini 3) for transferability across compositions and temperatures; or re-parameterize IBI at each state point of interest |
| Over-smoothed free-energy surface | CG FES shows barriers that are 2–5× smaller than the all-atom FES; transition rates are orders of magnitude too fast | CG potentials are effective free-energy surfaces; barriers correspond to free-energy barriers at the CG level, not atomistic level; degree of smoothing depends on the number of integrated-out degrees of freedom | Backmap two CG configurations (one in each basin) and compute the AA FES by PMF; compare barrier heights | Use CG FES for qualitative pathway identification only; report it as CG FES and note the expected smoothing; for quantitative barriers, use AA enhanced sampling |
| Treating CG dynamics as atomistic | Publication reports `10 µs CG simulation time` and interprets CG MSD, VACF, and relaxation times as real; no time-scale correction | Misunderstanding of CG dynamics; CG friction is not physical; correlation functions have different meaning in CG and AA | Check whether any time-scale correction is applied; verify by comparing CG diffusion coefficient to experimental value | Apply time-scale correction factor (system-specific); clearly distinguish CG time from real time in all plots and text; never report CG VACF as a physical velocity autocorrelation |

## Best Practices

- **Always validate against at least one experimental observable before any production analysis.** Area per lipid, bilayer thickness, or small-molecule partitioning are the standard benchmarks for membrane Martini simulations. A model that does not reproduce the reference APL within 5% is not validated.
- **Always use ElNeDyn elastic network for Martini 3 protein simulations.** Without the elastic network, protein secondary structure collapses within 100–500 ns in Martini 3. Always run martinize2 with `-elastic` and verify the protein DSSP secondary structure at the start of production.
- **Use CHARMM-GUI Martini Maker for complex membrane systems.** Manual insane setup is error-prone for asymmetric bilayers, cholesterol-containing membranes, and multi-component lipid mixtures. CHARMM-GUI generates consistent topology, coordinate, and mdp files and validates the lipid composition.
- **Never use PME electrostatics with Martini force fields.** Martini is parameterized with reaction-field electrostatics (ε_rf = 15). Using PME changes the effective dielectric environment and produces wrong partitioning, membrane thickness, and protein–membrane interactions. This is the single most common misconfiguration in Martini simulations.
- **Always apply and report a time-scale correction for any dynamical observable.** CG dynamics are faster than real dynamics by a system-dependent factor. Calibrate f_t by comparing the CG lateral diffusion coefficient to the experimental value for the dominant lipid species. Report CG time and corrected real time separately, and note the uncertainty in the correction.
- **Equilibrate each system component sequentially, not simultaneously.** For membrane systems: minimize → NVT with protein restrained → NPT (Berendsen, 50–100 ns to relax membrane area) → NPT (Parrinello-Rahman, 50–100 ns) → production. Skipping Berendsen NPT before Parrinello-Rahman leads to box oscillation crashes.
- **Backmap to all-atom before any high-accuracy measurement.** CG-MD gives the correct qualitative structure and thermodynamics, but free energies at the kT level, specific hydrogen bonds, and rare conformations require all-atom resolution. Use CG-MD to equilibrate, then backmap and collect AA-MD statistics.
- **For IBI-derived potentials: validate transferability explicitly.** Run the parameterized CG model at one or two state points different from the parameterization point (different T, different composition). If structural or thermodynamic properties diverge from AA reference at those conditions, document the validity domain of the potential.
- **Use GROMACS 2021+ for Martini 3.** Older GROMACS versions (< 2021) have known issues with the Verlet cutoff scheme and the Martini 3 LJ table. The C-rescale barostat (GROMACS 2021+) is particularly well-suited for Martini membrane simulations. [EXPERT REVIEW NEEDED]

## Integration with Other Skills

- **[gromacs-workflow](../gromacs-workflow/SKILL.md):** GROMACS provides the primary simulation engine for Martini CG-MD. The `.top`/`.itp` hierarchy, `grompp → mdrun` pipeline, and checkpoint-based restart from that skill apply directly to Martini simulations, with the mdp differences noted in Workflow 2 here.
- **[lammps-workflow](../lammps-workflow/SKILL.md):** LAMMPS is the standard engine for bottom-up CG models of polymers and materials (Kremer-Grest, IBI-derived CG potentials, tabulated pair potentials). The bead-spring polymer models and tabulated pair-style commands are native to LAMMPS.
- **[md-force-fields](../md-force-fields/SKILL.md):** Understanding the atomistic force field (CHARMM36m, AMBER ff14SB, OPLS-AA) is a prerequisite for any bottom-up CG parameterization — the CG potential must be derived from AA reference data, and the quality of the CG model is bounded by the quality of the AA force field.
- **[md-equilibration](../md-equilibration/SKILL.md):** CG systems require their own equilibration protocol: the faster dynamics and different timescales mean that equilibration criteria differ from all-atom MD. The quantitative stationarity tests from that skill (block averaging, rolling mean, stationarity tests) apply to CG thermodynamic observables but require longer windows.
- **[md-analysis](../md-analysis/SKILL.md):** MDAnalysis loads GROMACS Martini trajectories (`.tpr` + `.xtc`) identically to all-atom trajectories. The RDF, coordination number, MSD, and order parameter analysis workflows apply directly. Atom selections must use CG bead names (PO4, NC3, C1A, ...) rather than all-atom atom names.
- **[enhanced-sampling](../enhanced-sampling/SKILL.md):** PLUMED interfaces with GROMACS and LAMMPS for CG enhanced sampling (metadynamics, umbrella sampling) identically to the all-atom case. CG-MD enhanced sampling is particularly powerful for membrane protein insertion, lipid sorting, and vesicle formation where the relevant CVs are large-scale structural coordinates accessible only at CG resolution.
- **[free-energy-md](../free-energy-md/SKILL.md):** CG free-energy calculations (alchemical FEP with Martini, PMF from umbrella sampling) provide qualitative free-energy estimates at large scales, useful for ligand partitioning into membranes and peptide–lipid interactions. Quantitative free-energy differences (< 1 kJ/mol accuracy) require backmapping to AA and running AA FEP.
- **[scientific-visualization](../../scientific-data/scientific-visualization/SKILL.md):** Martini trajectories are visualized with VMD (Martini coloring scripts), OVITO (bead type coloring), and nglview (Jupyter). CG bead type color tables differ from all-atom element coloring. That skill covers rendering membrane systems, lipid tail order maps, and protein insertion movies from CG trajectories.
- **[reproducibility-workflow](../../scientific-data/reproducibility-workflow/SKILL.md):** CG simulations require archiving the martinize2 version, the elastic network parameters, the Martini force-field version (.itp files), insane parameters, and the backmapping protocol alongside the trajectory. All of these are non-obvious dependencies that determine the result. That skill covers DVC-based provenance for large trajectory datasets.

## Key References

1. Marrink, S. J., Risselada, H. J., Yefimov, S., Tieleman, D. P., & de Vries, A. H. (2007). *The MARTINI force field: Coarse grained model for biomolecular simulations.* Journal of Physical Chemistry B, 111(27), 7812–7824. https://doi.org/10.1021/jp071097f
2. Souza, P. C. T., Alessandri, R., Barnoud, J., Thallmair, S., Faustino, I., Grünewald, F., et al. (2021). *Martini 3: A general purpose force field for coarse-grained molecular dynamics.* Nature Methods, 18(4), 382–388. https://doi.org/10.1038/s41592-021-01098-3
3. Reith, D., Pütz, M., & Müller-Plathe, F. (2003). *Deriving effective mesoscale potentials from atomistic simulations.* Journal of Computational Chemistry, 24(13), 1624–1636. https://doi.org/10.1002/jcc.10307
4. Izvekov, S. & Voth, G. A. (2005). *A multiscale coarse-graining method for biomolecular systems.* Journal of Physical Chemistry B, 109(7), 2469–2473. https://doi.org/10.1021/jp044629q
5. Wassenaar, T. A., Ingólfsson, H. I., Böckmann, R. A., Tieleman, D. P., & Marrink, S. J. (2015). *Computational lipidomics with insane: A versatile tool for generating custom membranes for molecular simulations.* Journal of Chemical Theory and Computation, 11(5), 2144–2155. https://doi.org/10.1021/acs.jctc.5b00209
6. Kmiecik, S., Gront, D., Kolinski, M., Wieteska, L., Dawid, A. E., & Kolinski, A. (2016). *Coarse-grained protein models and their applications.* Chemical Reviews, 116(14), 7898–7936. https://doi.org/10.1021/acs.chemrev.6b00163
7. Noid, W. G. (2013). *Perspective: Coarse-grained models for biomolecular systems.* Journal of Chemical Physics, 139(9), 090901. https://doi.org/10.1063/1.4818908
8. Ingólfsson, H. I., Melo, M. N., van Eerden, F. J., Arnarez, C., Lopez, C. A., Wassenaar, T. A., et al. (2014). *Lipid organization of the plasma membrane.* Journal of the American Chemical Society, 136(41), 14554–14559. https://doi.org/10.1021/ja507832e
