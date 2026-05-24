---
name: dft-defects
domain: electronic-structure
maturity: draft
tools:
  - VASP
  - Quantum ESPRESSO
  - pymatgen
  - shakenbreak
  - doped
  - ASE
  - numpy
  - matplotlib
tested_versions:
  VASP: "6.4.2"
  Quantum ESPRESSO: "7.3"
  pymatgen: "2024.11.13"
  shakenbreak: "0.3.5"
  doped: "2.5.0"
  ASE: "3.23.0"
requires_gpu: false
estimated_runtime: hours-to-days
---

# DFT Point Defect Calculations

## Description

This skill covers the first-principles modeling of point defects in crystalline materials: vacancy, interstitial, antisite, substitutional, and complex defect construction; charge-state calculations; defect formation energy formalism with chemical potential constraints; finite-size corrections (Freysoldt-Neugebauer-Van de Walle, Kumagai-Oba, Makov-Payne); Fermi-level-dependent thermodynamics; transition levels; band-edge alignment; and defect concentration estimates. It uses VASP as the primary DFT backend with pymatgen (including the `doped` and `shakenbreak` packages) for automated defect generation, correction calculation, and thermodynamic analysis. Invoke this skill when computing defect formation energies, charge transition levels, defect concentrations, or carrier compensation in semiconductors and insulators.

## Domain Context

Point defects are local disruptions of the perfect crystal lattice that control a wide range of functional properties: carrier concentration in semiconductors, ionic conductivity in solid electrolytes, color centers in optical materials, magnetic impurities, and catalyst poison tolerance. Even at thermodynamic equilibrium, defect concentrations of 10¹⁰–10¹⁸ cm⁻³ are common and technologically significant.

**The fundamental challenge** of DFT defect modeling is that defects break translational symmetry. A defect must be embedded in a finite supercell large enough that the defect's strain and electrostatic fields decay before the cell boundary. For charged defects, the periodic boundary conditions introduce spurious long-range interactions between charge images that must be corrected analytically. For semiconductors with DFT band-gap underestimation, formation energies of charged defects are biased by the incorrect VBM and CBM positions.

**Defect formation energy** is the central quantity:

```
E_f[X^q] = E_tot[X^q] - E_tot[bulk] - Σ_i n_i μ_i + q(E_VBM + E_F) + E_corr[q]
```

where:
- `E_tot[X^q]` is the DFT total energy of the defect supercell in charge state q
- `E_tot[bulk]` is the DFT total energy of the perfect bulk supercell (identical size)
- `n_i` is the number of atoms of species i added (+) or removed (−) to form the defect
- `μ_i` is the chemical potential of species i (eV; relative to its DFT elemental reference)
- `q` is the formal charge state (integer; q > 0 = electron removed)
- `E_VBM` is the DFT valence-band maximum energy of the bulk (in the same potential reference frame)
- `E_F` is the Fermi level measured from the VBM (0 ≤ E_F ≤ E_gap)
- `E_corr[q]` is the finite-size correction for charged defects (zero for q = 0)

**Chemical potential constraints:** μ_i is not a free parameter — it is constrained by the thermodynamic stability of the host phase and its competing phases. By convention, μ_i = 0 corresponds to the DFT energy of the elemental standard state. A positive μ_i would mean atoms are more stable in the defect host than in the element, which is unphysical. The bounds are:

- Elemental limit: μ_i ≤ 0 (atom cannot be more stable than its elemental standard state)
- Host phase stability: Σ_i n_i^{host} × μ_i = ΔH_f(host) (chemical potentials consistent with host formation)
- Competing phase exclusion: Σ_i n_i^{comp} × μ_i < ΔH_f(comp) for all competing phases (host must be stable against decomposition)

**Charge transition level ε(q/q')** is the Fermi level at which two charge states have equal formation energy:

```
ε(q/q') = [E_f[X^q, E_F=0] − E_f[X^q', E_F=0]] / (q' − q)
```

Levels in the gap (0 < ε < E_gap) are thermodynamic transition levels. Levels above CBM or below VBM are resonant and the charge state is unstable in equilibrium.

**Image-charge corrections for charged defects:** A charged defect supercell interacts with its periodic images via a spurious electrostatic energy that scales as q²/εL, where L is the supercell linear dimension and ε is the dielectric constant. Three correction schemes are in common use:

- *Makov-Payne (MP):* Analytical correction assuming a model charge distribution in a homogeneous dielectric. Simple but inaccurate for non-cubic supercells and anisotropic dielectrics.
- *Freysoldt-Neugebauer-Van de Walle (FNV):* Uses the actual DFT electrostatic potential in the defect and bulk supercells. Aligns the far-field potentials (far from the defect) and subtracts the model Gaussian charge correction. More accurate than MP.
- *Kumagai-Oba (KO):* Extends FNV to anisotropic dielectric tensors and non-orthorhombic supercells. Required for materials with highly anisotropic dielectric constants (e.g., layered materials, hexagonal systems).

All three require the static dielectric tensor ε (ionic + electronic contributions), which must be computed separately via DFPT (IBRION=8 or LEPSILON=True in VASP).

**Band-gap underestimation:** GGA/PBE underestimates band gaps by 30–50% for many semiconductors (e.g., GaAs: DFT ≈ 0.2 eV, experiment 1.42 eV). For charge-state q, each electron added to the CBM or removed from the VBM carries an energy error. Band-gap correction schemes (Lany-Zunger, finite-size extrapolation, HSE06 benchmark, scissor correction) must be applied when comparing charged defect levels against experiment. [EXPERT REVIEW NEEDED: which band-gap correction is appropriate depends strongly on the material system and the charge state involved; there is no universally accepted single approach]

**Shallow vs. deep defects:**
- *Shallow defects:* Hydrogenic donors or acceptors with transition levels close to CBM or VBM (< kT from band edge at 300 K). Can be modeled by effective-mass theory; DFT gives the correct qualitative picture but may not capture effective-mass envelope accurately.
- *Deep defects:* Transition levels well inside the gap (> 0.1 eV from band edges). DFT captures these well qualitatively; quantitative accuracy requires HSE06 or GW, particularly for the gap correction.

## When to Use This Skill

- Computing defect formation energies and thermodynamic transition levels for point defects in semiconductors and insulators.
- Determining equilibrium carrier concentrations (electrons, holes) as a function of Fermi level and synthesis conditions.
- Identifying dominant compensating defects (e.g., which vacancy compensates intentional n-type doping).
- Predicting whether a defect acts as a trap, recombination center, or dopant.
- Generating DFT reference data for defect-containing structures for ML potential training.
- Cross-referencing with photoluminescence (PL), EPR, deep-level transient spectroscopy (DLTS), or Hall-effect measurements.
- Screening defect tolerance in candidate photovoltaic, thermoelectric, or solid-electrolyte materials.

## When Not to Use This Skill

- **Extended defects (dislocations, grain boundaries, stacking faults)** — these require larger simulation cells and different analysis methods; point-defect formalism does not apply.
- **Metallic hosts** — defect formation energy formalism is designed for semiconductors and insulators. In metals, there is no band gap and no well-defined Fermi-level thermodynamics; use simpler vacancy formation energy calculations instead.
- **Highly disordered or amorphous materials** — point defect concepts assume a well-defined reference crystal; ensemble approaches over many configurations are needed for disordered hosts.
- **Strongly correlated hosts (Mott insulators) without DFT+U or hybrid** — PBE defect levels in NiO, CoO, VO₂, and similar materials are qualitatively wrong without a Hubbard correction or hybrid functional.
- **Defect concentrations at very high temperature** — harmonic approximation for defect entropy breaks down; anharmonic vibrational entropy corrections become important above ~0.5 T_melt. [EXPERT REVIEW NEEDED]
- **Surface or interface defects** — these combine defect and surface physics; use this skill alongside the DFT Surfaces skill; the correction schemes differ significantly.

## Core Concepts

### Defect Types

- **Vacancy (V_X):** One atom of species X removed from a lattice site. Notation: V_Ga, V_O. Most common intrinsic defect.
- **Interstitial (X_i):** Extra atom inserted at a non-lattice position. Notation: O_i, H_i. High-energy in dense-packed structures; common in open frameworks and along diffusion channels.
- **Antisite (X_Y):** Atom of species X on a site normally occupied by species Y. Notation: As_Ga (As on Ga site in GaAs). Common in compound semiconductors and oxides.
- **Substitutional (X_Y):** Impurity atom X on a host lattice site Y. Notation: N_O (N on O site), Si_Ga. The primary form of intentional doping.
- **Complex defect:** Two or more point defects in close proximity, bound by elastic or electrostatic attraction. Notation: (V_Ga–V_As), (V_O–Ti_Li). Can have lower formation energy than the sum of isolated defects and drastically alter transport and optical properties.

### Supercell Size and Finite-Size Effects

For a charged defect in a supercell of linear dimension L:
- The dominant finite-size error scales as q²/(εL) for the electrostatic energy.
- Elastic relaxation errors scale as ~1/L³.
- Convergence criterion: surface energy of the relaxed defect changes by < 0.05 eV when doubling supercell volume. [EXPERT REVIEW NEEDED: the 0.05 eV threshold is a practical rule; some properties require tighter convergence]

Typical supercell sizes: 2×2×2 (64 atoms for 4-atom cell), 3×3×3 (108–216 atoms), or 4×4×4 (256–512 atoms). Larger supercells are preferred for:
- High charge states (|q| ≥ 2)
- Materials with low dielectric constants (ε < 5)
- Soft lattices with long-range elastic relaxation

For non-cubic systems, use a supercell that is approximately cubic (aspect ratio < 2 in all directions) to minimize anisotropic image interactions.

### Potential Alignment

Before computing E_f, the electrostatic potential of the defect supercell must be aligned with the perfect bulk supercell. This removes the arbitrary reference of the electrostatic zero. The standard procedure (FNV alignment):

1. Compute the planar-averaged electrostatic potential V_defect(r) and V_bulk(r)
2. In a region far from the defect (where the defect perturbation has decayed), compute ΔV = ⟨V_defect(r)⟩_far − ⟨V_bulk(r)⟩_far
3. Apply the alignment: shift E_VBM by ΔV

The far-field region should be identified by where |V_defect(r) − V_bulk(r) − q × V_model(r)| is minimized. The `doped` package automates this using the pymatgen `DefectEntry` infrastructure.

### Dielectric Tensor

All image-charge corrections require the total (ionic + electronic) static dielectric tensor ε_total = ε_ionic + ε_electronic. For isotropic materials, a scalar ε suffices. For anisotropic materials (hexagonal, tetragonal, orthorhombic), the full 3×3 tensor is needed.

In VASP: run a separate DFPT calculation on the perfect bulk unit cell with LEPSILON=True (electronic + ionic contributions) or IBRION=8 (ionic contribution from finite differences). The dielectric tensor appears in OUTCAR as "MACROSCOPIC STATIC DIELECTRIC TENSOR IONIC CONTRIBUTION" and "MACROSCOPIC STATIC DIELECTRIC TENSOR (including local field effects in DFT)".

### Spin Polarization and Localized States

Many defects carry unpaired electrons and require spin-polarized calculations (ISPIN=2). Missing spin polarization can:
- Artificially stabilize incorrect ground states (e.g., V_O in TiO₂ has a spin-1 ground state in some charge states)
- Incorrectly predict whether a defect is in the gap or resonant with the bands
- Give wrong total energies by several tenths of an eV

Always initialize with non-zero MAGMOM for transition-metal-containing hosts and for all defect charge states that could be magnetic. Run with LORBIT=11 to extract atom-projected DOS and verify localization of the defect state.

### Symmetry Breaking and Metastable Configurations

DFT geometry optimization can get trapped in a local minimum if the initial defect geometry has artificially high symmetry. For example, a vacancy in a Jahn-Teller-active system (e.g., V_N in GaN in certain charge states) must break symmetry to find the true ground state. The `shakenbreak` package addresses this by generating a set of symmetry-broken starting geometries (via atomic displacements) and running all of them; the lowest-energy relaxed structure is the ground state.

Failure to break symmetry can produce formation energies that are 0.1–0.5 eV too high and incorrect transition-level positions. This is one of the most commonly overlooked errors in defect calculations. [EXPERT REVIEW NEEDED: the magnitude of symmetry-breaking errors is highly material-specific; for some systems it can exceed 1 eV]

## Key Workflows

### Workflow 1: Defect Supercell Generation with `doped`

```python
# defect_generation.py
# Generate all symmetry-inequivalent point defects using the doped package.
# doped wraps pymatgen DefectGenerator and adds automation for VASP inputs.

from pathlib import Path
from pymatgen.core import Structure
from doped.generation import DefectGenerator
from doped.vasp import DefectsSet

# --- Load relaxed bulk unit cell (ISIF=3 converged) ---
bulk = Structure.from_file("POSCAR_bulk")
print(f"Bulk: {bulk.formula}, spacegroup: {bulk.get_space_group_info()[0]}")

# --- Generate all inequivalent defects ---
defect_gen = DefectGenerator(
    structure=bulk,
    generate_antisite=True,   # include all antisite defects
    generate_interstitial=True,
    interstitial_gen_kwargs={"min_dist": 0.9},   # Å; avoid too-close insertions
    extrinsic={},             # add {"N": ["O"]} for N substituting O, etc.
)

defects = defect_gen.generate_defects()
print(f"\nGenerated {len(defects)} symmetry-inequivalent defects:")
for d in defects:
    print(f"  {d.name}: {[cs for cs in d.charge_states]}")

# --- Set up VASP inputs for all defects ---
# INCAR tailored for defect supercells (semiconductors/insulators)
custom_incar = {
    "ENCUT": 550,           # eV; match bulk calculation; higher for accurate forces
    "EDIFF": 1e-6,          # for ionic relaxation; tighten to 1e-8 for final single-point
    "EDIFFG": -0.02,        # eV/Å force convergence threshold
    "NSW": 200,
    "IBRION": 2,            # CG; switch to 1 (RMM-DIIS) once close to minimum
    "ISIF": 2,              # relax ions only; cell fixed (defect must use bulk cell)
    "ISMEAR": 0,            # Gaussian; required for semiconductors/insulators
    "SIGMA": 0.05,          # eV; small; check that entropy contribution < 1 meV/atom
    "ALGO": "Normal",       # Davidson; more reliable than Fast for defect supercells
    "LREAL": "Auto",        # acceptable for supercells > 50 atoms; use False for final SP
    "ISPIN": 2,             # always spin-polarized for defect calculations
    "LORBIT": 11,           # atom-projected DOS and PDOS
    "LWAVE": True,          # save WAVECAR for post-processing
    "LCHARG": True,
    "NPAR": 4,
    "KPAR": 1,              # typically Gamma-only for large supercells; set KPAR=2 if >1 k-point
    "NELM": 200,            # allow more SCF steps; defect supercells can be slow to converge
}

# Supercell matrix: 2x2x2 for 8-atom cell → 64-atom supercell; adjust per system
# Target: supercell linear dimension > 12 Å and ideally > 15 Å
defects_set = DefectsSet(
    defects=defects,
    user_incar_settings=custom_incar,
    supercell_gen_kwargs={
        "min_length": 12,       # minimum cell dimension in Å
        "force_diagonal": False, # allow non-diagonal supercell if it gives better shape
    },
    charge_state_gen_kwargs={
        "charge_interval": 1,   # generate q = q_min, q_min+1, ..., q_max
    },
)

output_dir = Path("defect_calcs")
defects_set.write_files(output_dir)
print(f"\nVASP input directories written to {output_dir}/")
print("Each subdirectory contains INCAR, POSCAR, POTCAR, KPOINTS for one (defect, charge) pair.")
```

```bash
# KPOINTS for defect supercells (usually Gamma-only for large cells > 100 atoms):
cat > KPOINTS << 'EOF'
Gamma-only
0
Gamma
 1 1 1
 0 0 0
EOF
# For smaller supercells (< 64 atoms) or materials with flat bands, use denser mesh:
# 2x2x2 or 4x4x4 Gamma-centered, but always verify k-convergence on defect energy.
```

### Workflow 2: Symmetry Breaking with `shakenbreak`

This workflow is required for any defect that could undergo Jahn-Teller distortion or where the high-symmetry configuration may be a saddle point rather than a minimum.

```python
# shakenbreak_workflow.py
# Generate rattled/distorted starting geometries to escape symmetry-constrained local minima.
# Must be run BEFORE or instead of the doped workflow for systems with potential JT activity.

from pathlib import Path
from pymatgen.core import Structure
import shakenbreak
from shakenbreak.input import Distortions

bulk = Structure.from_file("POSCAR_bulk")

# Create the defect — here a Ga vacancy in GaAs as a realistic example
# In practice, use DefectGenerator from doped and pass defects directly
from pymatgen.analysis.defects.generators import VacancyGenerator
vac_gen = VacancyGenerator()
vacancies = list(vac_gen.get_defects(bulk))
print(f"Vacancy types: {[v.name for v in vacancies]}")

# Apply SnB distortions
distortions = Distortions(
    defects=vacancies,
    oxidation_states={el.symbol: ox for el, ox in bulk.composition.oxi_state_guesses()[0].items()},
)

# Generate distorted supercells for each charge state
distorted_defects = distortions.apply_distortions(
    distortion_parameters={
        "distortion_factor": 0.4,  # bond length perturbation fraction; 0.4 is SnB default
        "num_distortions": 10,     # number of random rattling configurations to try
        "mc_rattle": True,         # Monte Carlo rattling of all atoms
        "rattle_std": 0.1,         # Å standard deviation for atomic displacement in rattling
    }
)

output_dir = Path("shakenbreak_calcs")
distortions.write_vasp_files(
    output_path=str(output_dir),
    incar_updates={
        "ENCUT": 550,
        "EDIFF": 1e-6,
        "EDIFFG": -0.02,
        "IBRION": 2,
        "ISPIN": 2,
        "NSW": 300,
    },
)
print(f"\nDistorted structures written to {output_dir}/")
print(f"After all VASP runs complete, analyze with shakenbreak.analysis:")
print(f"  python -c \"import shakenbreak; shakenbreak.analysis.plot_energies('{output_dir}')\"")
```

```python
# After VASP calculations are complete, identify ground state:
from shakenbreak import analysis

results = analysis.get_energies(path=str(output_dir))
ground_state = analysis.get_ground_state_structure(path=str(output_dir))

for defect_name, charge_results in results.items():
    for q, energies in charge_results.items():
        e_unpert = energies.get("Unperturbed")
        e_min = min(energies.values())
        e_saved = e_unpert - e_min if e_unpert is not None else None
        print(f"  {defect_name} {q:+d}: best = {e_min:.4f} eV, "
              f"unperturbed = {e_unpert:.4f if e_unpert else 'N/A'} eV, "
              f"saved = {e_saved:.3f if e_saved else 'N/A'} eV")
```

### Workflow 3: Dielectric Tensor Calculation (VASP DFPT)

The dielectric tensor is required for all image-charge corrections. Run this once on the bulk unit cell.

```python
# dielectric_calc.py
# INCAR for VASP DFPT to get ionic + electronic dielectric tensor.
# Run on the RELAXED bulk unit cell (not supercell).

INCAR_DFPT = {
    "ENCUT": 550,
    "EDIFF": 1e-8,          # very tight SCF required for DFPT response
    "PREC": "Accurate",
    "NSW": 0,
    "IBRION": 8,            # DFPT ionic contribution to dielectric tensor
    "ISMEAR": 0,
    "SIGMA": 0.01,
    "ALGO": "Normal",
    "LREAL": False,
    "LEPSILON": True,       # compute electronic dielectric tensor and Born charges
    "NPAR": 1,              # LEPSILON requires NPAR=1
    "ISPIN": 1,             # spin-unpolarized for dielectric (use 2 if host is magnetic)
}

# After VASP runs, parse dielectric tensor
from pymatgen.io.vasp import Outcar
import numpy as np

outcar = Outcar("OUTCAR_dfpt")
eps_electronic = np.array(outcar.dielectric_tensor)   # 3x3, electronic contribution
# Ionic contribution from OUTCAR (search for "MACROSCOPIC STATIC DIELECTRIC TENSOR IONIC")
# pymatgen parses both automatically in recent versions

print("Electronic dielectric tensor:")
print(eps_electronic)
print(f"Mean ε_electronic: {np.trace(eps_electronic)/3:.3f}")

# For isotropic materials: ε_total ≈ trace(ε_electronic + ε_ionic) / 3
# For correction codes, pass the full 3x3 tensors to doped/pydefect
```

### Workflow 4: Defect Formation Energy and Transition Levels

```python
# defect_formation_energy.py
# Compute defect formation energies and thermodynamic transition levels
# using the doped package post-processing infrastructure.

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from doped.analysis import DefectThermodynamics
from doped.chemical_potentials import ChemicalPotentialDiagram
from pymatgen.core import Composition

# --- Step 1: Chemical potential limits ---
# For GaAs with intrinsic defects (V_Ga, V_As, Ga_As, As_Ga, Ga_i, As_i)
# Chemical potential space: μ_Ga + μ_As = ΔH_f(GaAs)
# Elemental limits: μ_Ga ≤ 0, μ_As ≤ 0

DH_GaAs = -0.74   # eV per formula unit (GGA value; exptl = -0.74 eV)

# Competing phases (from Materials Project or own DFT):
# Ga2As3: ΔH_f = -0.61 eV/f.u. → 2μ_Ga + 3μ_As < 2*(-0.61)/5 * 5 = -1.22 eV? 
# (simplified; use ChemicalPotentialDiagram for rigorous treatment)

# Ga-rich limit: μ_Ga → 0, μ_As = ΔH_f(GaAs) = -0.74 eV
# As-rich limit: μ_As → 0, μ_Ga = ΔH_f(GaAs) = -0.74 eV
chem_pots = {
    "Ga-rich": {"Ga": 0.0, "As": DH_GaAs},
    "As-rich": {"Ga": DH_GaAs, "As": 0.0},
}

# --- Step 2: Parse VASP results with doped ---
# doped reads all completed defect VASP directories and builds DefectEntry objects
thermo = DefectThermodynamics.from_directory(
    defect_dir="defect_calcs/",
    bulk_dir="bulk/",
    dielectric=np.array([[12.4, 0, 0], [0, 12.4, 0], [0, 0, 12.4]]),   # GaAs ε_r
    bulk_band_gap=1.424,   # experimental or HSE gap in eV; used for Fermi-level range
    correction_kwargs={"correction_algorithm": "Freysoldt"},   # FNV correction
)

print(f"Loaded {len(thermo.defect_entries)} defect entries")

# --- Step 3: Formation energy at chemical potential limits ---
ef_range = np.linspace(0, thermo.band_gap, 200)

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
for ax, (chem_label, mu_dict) in zip(axes, chem_pots.items()):
    thermo.plot(
        chempots=mu_dict,
        ef_range=ef_range,
        ax=ax,
        auto_labels=True,     # label each line with defect name
        colormap="tab20",
    )
    ax.set_title(f"Formation energy: {chem_label} limit")
    ax.set_xlabel("Fermi level (eV)")
    ax.set_ylabel("Formation energy (eV)")
    ax.axhline(0, color="gray", lw=0.5, ls=":")

plt.tight_layout()
plt.savefig("formation_energy.png", dpi=150, bbox_inches="tight")

# --- Step 4: Transition levels ---
transition_levels = thermo.get_transition_levels()
print("\nCharge transition levels:")
for defect_name, levels in transition_levels.items():
    for (q1, q2), ef_level in levels.items():
        label = f"ε({q1:+d}/{q2:+d})"
        in_gap = 0 < ef_level < thermo.band_gap
        print(f"  {defect_name:20s} {label}: {ef_level:.3f} eV "
              f"({'in gap' if in_gap else 'RESONANT — outside gap'})")

# --- Step 5: Equilibrium Fermi level and carrier concentration ---
T_K = 900   # synthesis temperature (K); adjust per material
print(f"\nEquilibrium analysis at T = {T_K} K:")
for chem_label, mu_dict in chem_pots.items():
    result = thermo.get_equilibrium_fermi_level(
        chempots=mu_dict,
        temperature=T_K,
        bulk_dos=None,   # pass pymatgen DOS object for more accurate carrier stats
    )
    print(f"  {chem_label}: E_F = {result['fermi_level']:.3f} eV, "
          f"n = {result.get('n', 'N/A'):.2e} cm⁻³, "
          f"p = {result.get('p', 'N/A'):.2e} cm⁻³")
```

### Workflow 5: Manual Formation Energy Calculation (Without Automation Packages)

For materials or charge states not handled by doped, or for cross-checking automated results.

```python
# manual_formation_energy.py
# Reference implementation of the defect formation energy formula.
# Uses VASP OUTCAR parsing and manual FNV alignment.

import numpy as np
from pathlib import Path

def parse_vasp_energy(outcar_path):
    """Parse final total energy from OUTCAR."""
    energy = None
    with open(outcar_path) as f:
        for line in f:
            if "TOTEN" in line and "eV" in line:
                try:
                    energy = float(line.split()[-2])
                except (ValueError, IndexError):
                    pass
    if energy is None:
        raise RuntimeError(f"Could not parse energy from {outcar_path}")
    return energy

def parse_vbm_from_outcar(outcar_path):
    """Parse VBM from VASP OUTCAR (for insulators with ISMEAR=0)."""
    vbm = None
    with open(outcar_path) as f:
        for line in f:
            if "E-fermi" in line:
                try:
                    vbm = float(line.split()[2])   # Fermi level ~ VBM for insulator
                except (ValueError, IndexError):
                    pass
    return vbm

def makov_payne_correction(q, epsilon, L_ang, madelung=2.8373):
    """
    Makov-Payne first-order correction for cubic supercell (eV).
    q: charge state (integer)
    epsilon: static dielectric constant (scalar)
    L_ang: linear supercell dimension (Å)
    madelung: Madelung constant for FCC (2.8373) or SC (2.8373 approx)
    """
    if q == 0:
        return 0.0
    eV_bohr_to_eV_ang = 1.0 / 0.529177
    L_bohr = L_ang / 0.529177
    correction = -(madelung * q**2) / (2.0 * epsilon * L_bohr) * eV_bohr_to_eV_ang
    return correction

# --- System: V_O (oxygen vacancy) in MgO in charge states 0, +1, +2 ---
# Inputs (all from VASP runs):
E_BULK    = -37.21   # eV; perfect bulk 64-atom MgO supercell
E_VO_0    = -35.44   # eV; V_O^0 supercell
E_VO_1    = -34.12   # eV; V_O^{+1} supercell (one electron removed)
E_VO_2    = -32.71   # eV; V_O^{+2} supercell (two electrons removed)
E_VBM     = -2.14    # eV; VBM energy from perfect bulk OUTCAR (E-fermi for insulator)
E_O_REF   = -4.93    # eV; ½ E(O2_GGA) per O atom [apply correction as needed]
EPS_MgO   = 9.8      # total dielectric constant of MgO
L_SUP     = 8.43     # Å; supercell linear dimension (4.215 Å * 2)

# Potential alignment (manual FNV far-field alignment)
# In production, compute ΔV from LOCPOT difference (far from defect)
DV_0  = 0.0     # eV; alignment for neutral (zero by definition)
DV_1  = 0.043   # eV; far-field LOCPOT shift for +1 supercell
DV_2  = 0.089   # eV; far-field LOCPOT shift for +2 supercell

def formation_energy(e_defect, e_bulk, n_removed_O, mu_O, q, e_vbm, ef, dv, epsilon, L):
    """
    Compute defect formation energy with Makov-Payne correction and potential alignment.
    n_removed_O: number of O atoms removed (positive for vacancy)
    """
    e_f = e_defect - e_bulk + n_removed_O * mu_O  # stoichiometry term
    e_f += q * (e_vbm + dv + ef)                   # charge term (aligned VBM)
    e_corr = makov_payne_correction(q, epsilon, L)
    e_f += e_corr
    return e_f, e_corr

# Scan Fermi level (0 to ~8 eV = MgO band gap)
EG_MgO = 7.83   # eV (experimental; GGA gives ~4.5 eV — significant gap error)
ef_values = np.linspace(0, EG_MgO, 200)
mu_O = E_O_REF   # O-rich limit: μ_O at ½ E(O2); O-poor: μ_O = ΔH_f(MgO)/1

print("V_O formation energies at E_F = 0 (VBM):")
for q, e_def, dv in [(0, E_VO_0, DV_0), (1, E_VO_1, DV_1), (2, E_VO_2, DV_2)]:
    ef_0, corr = formation_energy(e_def, E_BULK, 1, mu_O, q, E_VBM, 0.0, dv, EPS_MgO, L_SUP)
    print(f"  V_O^{q:+d}: E_f = {ef_0:.3f} eV, MP correction = {corr:.3f} eV")

# Transition levels
def transition_level(e_f_q1, e_f_q2, q1, q2):
    """Fermi level where E_f[q1] = E_f[q2]."""
    return (e_f_q1 - e_f_q2) / (q2 - q1)

ef_10_0, _ = formation_energy(E_VO_1, E_BULK, 1, mu_O, 1, E_VBM, 0, DV_1, EPS_MgO, L_SUP)
ef_10_2, _ = formation_energy(E_VO_2, E_BULK, 1, mu_O, 2, E_VBM, 0, DV_2, EPS_MgO, L_SUP)
ef_10_0_q0, _ = formation_energy(E_VO_0, E_BULK, 1, mu_O, 0, E_VBM, 0, DV_0, EPS_MgO, L_SUP)

eps_2_0 = transition_level(ef_10_2, ef_10_0_q0, 2, 0)
eps_2_1 = transition_level(ef_10_2, ef_10_0, 2, 1)
print(f"\nV_O transition levels (MP-corrected, O-rich, GGA VBM):")
print(f"  ε(+2/0):  {eps_2_0:.3f} eV from VBM")
print(f"  ε(+2/+1): {eps_2_1:.3f} eV from VBM")
```

### Workflow 6: Defect Concentration Estimates

```python
# defect_concentration.py
# Estimate equilibrium defect concentrations using formation energies.
# Assumes harmonic approximation (no vibrational entropy correction).
# [EXPERT REVIEW NEEDED: vibrational entropy contributions can shift concentrations
# by orders of magnitude for defects with large entropy differences between charge states]

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import k as kB_SI, N_A

kB = kB_SI / 1.602176634e-19   # Boltzmann constant in eV/K

def defect_concentration(e_formation_eV, T_K, N_sites_per_cm3, deg=1):
    """
    Equilibrium point defect concentration (cm⁻³) at temperature T_K.
    e_formation_eV: formation energy in eV (at the given Fermi level)
    N_sites_per_cm3: number density of available sites (from bulk lattice constant)
    deg: degeneracy factor (spin, symmetry; typically 1–4)
    """
    return N_sites_per_cm3 * deg * np.exp(-e_formation_eV / (kB * T_K))

# MgO V_O example: lattice constant a=4.215 Å → N_sites = 1/a³ × 4 (FCC)
a_MgO = 4.215e-8   # cm
V_cell = a_MgO**3
N_O_per_cm3 = 4 / (2 * V_cell)   # 4 O per cubic cell (FCC MgO: 4 formula units)
print(f"O site density: {N_O_per_cm3:.3e} cm⁻³")

# Formation energies at E_F = mid-gap (E_F = EG/2 ~ 3.9 eV for MgO GGA)
# (read from formation energy calculation above)
e_f_values = {
    "V_O^0":   2.8,    # eV at E_F = mid-gap, O-rich
    "V_O^{+1}": 2.1,
    "V_O^{+2}": 1.5,
}

T_values = np.arange(300, 2000, 50)   # K
print("\nEquilibrium [V_O] at O-rich limit, E_F = mid-gap:")
print(f"{'T (K)':<8} {'[V_O^0]':<15} {'[V_O^+1]':<15} {'[V_O^+2]':<15}")
for T in [600, 900, 1200, 1500, 1800]:
    row = f"{T:<8}"
    for label, ef in e_f_values.items():
        c = defect_concentration(ef, T, N_O_per_cm3, deg=1)
        row += f"{c:.2e}        "
    print(row)

# Plot as function of temperature
fig, ax = plt.subplots(figsize=(7, 5))
colors = {"V_O^0": "steelblue", "V_O^{+1}": "darkorange", "V_O^{+2}": "firebrick"}
for label, ef in e_f_values.items():
    conc = [defect_concentration(ef, T, N_O_per_cm3) for T in T_values]
    ax.semilogy(T_values, conc, label=label, color=colors[label], lw=1.8)

ax.set_xlabel("Temperature (K)")
ax.set_ylabel("Defect concentration (cm⁻³)")
ax.set_title("V_O equilibrium concentration in MgO (O-rich, E_F = mid-gap)")
ax.legend()
ax.set_xlim(300, 2000)
ax.axhline(1e10, ls=":", color="gray", lw=0.8, label="1e10 cm⁻³ threshold")
plt.savefig("defect_concentration.png", dpi=150, bbox_inches="tight")
```

### Workflow 7: Band-Gap Correction and VBM Alignment

When comparing DFT defect levels to experiment, the band-gap underestimation must be addressed. This workflow applies the Lany-Zunger scissor correction as the simplest approach.

```python
# bandgap_correction.py
# Apply Lany-Zunger scissor correction to charged defect formation energies.
# Shifts CBM upward to match experimental (or HSE06) band gap.
# More rigorous: run HSE06 single-points on converged PBE geometries.

import numpy as np

def lany_zunger_correction(e_f_gga, q, delta_e_gap, vbm_shift=0.0):
    """
    Apply Lany-Zunger correction to GGA defect formation energy.
    e_f_gga: GGA formation energy at a given E_F in eV
    q: charge state (integer; negative = electron donor)
    delta_e_gap: E_gap(hybrid/exp) - E_gap(GGA) in eV; always positive
    vbm_shift: fraction of delta_e_gap attributed to VBM shift (0 to 1);
               for pure band-gap opening, vbm_shift = 0 (CBM moves up only);
               Lany-Zunger uses vbm_shift = 0 for donor defects, 1 for acceptors
               [EXPERT REVIEW NEEDED: the partitioning is system-dependent]

    For donor (positive q): CBM is relevant → correction = -q * (1 - vbm_shift) * delta_E_gap
    For acceptor (negative q): VBM is relevant → correction = -q * vbm_shift * delta_E_gap
    """
    if q > 0:  # donor: valence band filled, charge comes from elevated CBM
        correction = -q * (1.0 - vbm_shift) * delta_e_gap
    elif q < 0:  # acceptor: charge comes from lowered VBM
        correction = -q * vbm_shift * delta_e_gap
    else:
        correction = 0.0
    return e_f_gga + correction

# Example: V_O in MgO
# GGA band gap = 4.5 eV; experimental = 7.83 eV; HSE06 = 7.6 eV
delta_eg_MgO = 7.83 - 4.5   # = 3.33 eV gap correction

print("V_O in MgO — Lany-Zunger correction (at E_F = mid-gap):")
for q, ef_gga in [(0, 2.8), (1, 2.1), (2, 1.5)]:
    ef_corrected = lany_zunger_correction(ef_gga, q, delta_eg_MgO, vbm_shift=0.0)
    print(f"  V_O^{q:+d}: GGA = {ef_gga:.2f} eV → corrected = {ef_corrected:.2f} eV "
          f"(Δ = {ef_corrected - ef_gga:+.2f} eV)")

print("\nNote: More rigorous approach = HSE06 single-point on PBE-relaxed geometry.")
print("Use doped's HSE correction infrastructure or compute transition levels directly at HSE06.")
```

### Workflow 8: Defect-Containing Structures for MLP Training Datasets

```python
# defect_dataset_generation.py
# Generate a diverse set of defect-containing structures for MLP training.
# Covers: perfect bulk, vacancy, interstitial, antisite at different charge states,
# partially relaxed structures along the relaxation path.

import numpy as np
from pathlib import Path
from ase.io import read, write
from ase.io.trajectory import Trajectory
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

output_dir = Path("mlp_defect_dataset")
output_dir.mkdir(exist_ok=True)
all_structures = []

adaptor = AseAtomsAdaptor()

def add_to_dataset(atoms, config_type, energy, forces, stress=None):
    """Store an ASE atoms object with DFT labels in extxyz format."""
    atoms.info["energy"] = energy
    atoms.info["config_type"] = config_type
    atoms.arrays["forces"] = np.array(forces)
    if stress is not None:
        atoms.info["stress"] = np.array(stress)
    all_structures.append(atoms)

# --- 1. Perfect bulk reference structures (at different volumes for EOS) ---
for v_scale in [0.97, 0.99, 1.00, 1.01, 1.03]:
    bulk = read("bulk/POSCAR", format="vasp")
    cell = bulk.get_cell()
    bulk.set_cell(cell * v_scale**(1/3), scale_atoms=True)
    # (in production, run DFT SP and parse energy/forces from OUTCAR)
    add_to_dataset(bulk, "bulk_eos", energy=None, forces=None)  # placeholder

# --- 2. Relaxed defect supercells (one per defect type and charge state) ---
defect_dirs = sorted(Path("defect_calcs").glob("*/"))
for d in defect_dirs:
    contcar = d / "CONTCAR"
    outcar  = d / "OUTCAR"
    if not (contcar.exists() and outcar.exists()):
        continue
    atoms = read(str(contcar), format="vasp")
    # Parse energy and forces from OUTCAR
    energy = None
    forces = []
    with open(outcar) as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if "TOTEN" in line and "eV" in line:
            try:
                energy = float(line.split()[-2])
            except (ValueError, IndexError):
                pass
        if "TOTAL-FORCE" in line:
            j = i + 2
            while j < len(lines) and lines[j].strip() and not lines[j].startswith(" -"):
                try:
                    parts = list(map(float, lines[j].split()))
                    if len(parts) >= 6:
                        forces.append(parts[3:6])
                    j += 1
                except ValueError:
                    break
    if energy and forces:
        config_type = f"defect_{d.name}"
        add_to_dataset(atoms, config_type, energy, np.array(forces))

# --- 3. Partially relaxed structures (snapshots along relaxation path) ---
for d in defect_dirs:
    traj_file = d / "vasprun.xml"
    if not traj_file.exists():
        continue
    from pymatgen.io.vasp import Vasprun
    try:
        vr = Vasprun(str(traj_file), parse_potcar_file=False)
        for step in vr.ionic_steps[::3]:   # every 3rd ionic step
            struct = step["structure"]
            atoms = adaptor.get_atoms(struct)
            f = np.array(step["forces"])
            e = step["e_wo_entrp"]
            add_to_dataset(atoms, f"defect_relax_{d.name}", e, f)
    except Exception as ex:
        print(f"  Could not parse {traj_file}: {ex}")

# --- 4. Write combined extxyz dataset ---
dataset_path = output_dir / "defects.extxyz"
write(str(dataset_path), all_structures, format="extxyz")
print(f"\nWrote {len(all_structures)} structures to {dataset_path}")
print("Config types present:", set(a.info.get("config_type", "unknown") for a in all_structures))
```

## Common Failure Modes

| Failure Mode | Symptoms | Likely Cause | Diagnostic | Fix |
|---|---|---|---|---|
| Supercell too small | Formation energy changes by > 0.1 eV when doubling supercell size; correction term is > 30% of total formation energy | Image-charge interactions dominate; strain not decayed | Run two supercell sizes; check that E_corr < 0.3 × E_f | Use larger supercell: min 128 atoms for |q|=1 in low-ε material |
| Wrong charge correction scheme | Formation energies inconsistent between q and q+1 at a transition level; kink in E_f vs E_F plot | MP correction for non-cubic supercell; FNV misalignment; wrong dielectric | Compare MP vs FNV corrections; plot potential alignment curves | Use FNV (Freysoldt) with full LOCPOT alignment; use KO for anisotropic ε |
| Unconverged potential alignment | FNV ΔV not flat in the far-field region; potential continues to oscillate far from defect | Defect charge delocalizes to supercell boundary; supercell too small for alignment region | Plot V_defect - V_bulk - q×V_model vs position; check flatness > 4 Å from defect | Increase supercell; check that LOCPOT was written (LVTOT=True) |
| Missing spin polarization | Formation energy wrong by 0.1–1 eV; DOS shows metallic character for insulating defect | ISPIN=1; no MAGMOM initialization; wrong spin state for complex defect | Re-run with ISPIN=2, MAGMOM initialized; check NUPDOWN for fixed-spin test | Always use ISPIN=2 for all defect charge states; initialize MAGMOM per species |
| Artificial symmetry (no SnB) | Ground state energy 0.1–0.5 eV too high; incorrect transition level positions; DFT finds saddle point | High-symmetry initial geometry prevents Jahn-Teller distortion | Compare energy with shakenbreak distorted runs; check if lowest-symmetry config is lower | Run shakenbreak for all defects; never use only the symmetric initial geometry |
| ISIF=3 applied to defect supercell | Cell volume changes upon relaxation; defect concentration depends on cell size; non-comparable to bulk | Variable-cell relaxation used; DFT minimizes artificial stress | Check ISIF in INCAR; cell should match bulk | Set ISIF=2 for all defect supercell calculations |
| Wrong chemical potential reference | Formation energies off by 0.5–2 eV; wrong stability window | μ_i referenced to wrong DFT total energy; competing phase not included | Cross-check μ_i values against ΔH_f of host and all competing phases | Use consistent DFT functional + POTCAR across all reference calculations; include all competing phases in μ space |
| GGA band-gap error uncorrected | Transition levels 0.5–2 eV off from experiment; levels appear above CBM that should be in gap | Band gap underestimated; VBM/CBM shifted | Compare DFT gap to experimental; apply Lany-Zunger or HSE06 correction | Apply scissor correction; run HSE06 single-point; always report DFT gap alongside results |
| Defect state resonant with bands | Charged defect has electrons/holes delocalized into band; formation energy unphysical | Shallow defect or wrong charge state in a too-small gap | Check PDOS of defect supercell; look for localized state in gap | Use HSE06 if bands are not correctly positioned; check charge state stability |
| k-mesh not converged | Energy oscillates with k-point sampling; adsorption-like energy noise | Too few k-points for metallic-like defect state | Convergence test: Γ-only vs 2×2×2 vs 4×4×4; check for metallic PDOS at defect | Use Γ-only only for large supercells (> 100 atoms) confirmed insulating |
| Comparing non-equivalent supercells | Formation energies inconsistent between defect and bulk | Different supercell sizes, shapes, or POTCAR versions used | Cross-check atom count, cell vectors, POTCAR SHA | Always use identical supercell (same transformation matrix from same bulk cell) |
| Missing competing phases | Chemical potential region appears wider than physical | Not all stable phases in the phase diagram included | Check Materials Project or ICSD for known binary/ternary phases in the composition space | Run DFT for all relevant competing phases; use ChemicalPotentialDiagram from pymatgen/doped |
| LREAL=Auto for final energies | Systematic force errors ~1 meV/Å; affects transition level positions slightly | Real-space projection for large supercells adds basis error | Check LREAL in INCAR; compare energy from LREAL=Auto vs LREAL=False single-point | Use LREAL=False for all final single-point energy calculations |

## Best Practices

**Supercell construction:**
- Always use the same bulk unit cell (same POTCAR, same lattice constant) for all defect calculations and the reference bulk calculation. A single inconsistent POTCAR version can introduce an error of several tenths of an eV.
- Construct supercells by a transformation matrix applied to the primitive cell, not by manually scaling; this ensures the defect-image distance is isotropic.
- For charged defects (|q| ≥ 2) in low-dielectric materials (ε < 10), use at least a 3×3×3 supercell of the conventional cell.
- Always compute the perfect bulk supercell using exactly the same ENCUT, KPOINTS, and POTCAR as the defect supercells.

**INCAR for defect supercells:**
- ISMEAR=0 with SIGMA=0.05 for all semiconductors and insulators. Never use ISMEAR=1 or ISMEAR=2 for semiconductors — incorrect occupation of gap states.
- ISPIN=2 with MAGMOM initialized to non-zero values for all atoms (even non-magnetic elements if the defect introduces a local moment).
- ALGO=Normal (Davidson); ALGO=Fast can miss some spin-state crossings.
- ISIF=2: relax atoms only; never relax the cell for defect calculations.
- NELM=200 or higher; defect supercells can require many SCF iterations.
- LVTOT=True for all charged defect calculations (required for FNV potential alignment).

**Correction and alignment:**
- Never report charged defect formation energies without an image-charge correction. Even the Makov-Payne correction is better than no correction.
- Always plot the LOCPOT alignment curves to verify that the far-field region is flat. A non-flat alignment indicates either a too-small supercell or a delocalized defect state.
- Use the full dielectric tensor (not a scalar average) for KO correction in anisotropic materials.
- Report which correction scheme was used and its magnitude relative to the total formation energy.

**Chemical potentials:**
- Always compute ΔH_f for the host and all competing phases with the same POTCAR and functional as the defect calculations. Mixing DFT+U energies with PBE energies for chemical potentials introduces systematic errors.
- For oxides with GGA+U: use VASP-recommended U values consistently for both host and competing phases, or apply the Materials Project mixing scheme.
- Always show the chemical potential stability diagram alongside formation energy diagrams; it defines the accessible synthesis window.

**Symmetry breaking:**
- For all defects in materials with partially filled d or f shells (transition metals, rare earths) and all vacancies in compound semiconductors, run shakenbreak before relying on any single starting geometry.
- Keep all distorted geometry outputs, even those that did not find the global minimum; they document the energy landscape.

**Reporting:**
- State: supercell size (atoms), correction scheme, dielectric tensor used, chemical potential limits, DFT functional, band gap (DFT and experimental/reference), whether band-gap correction was applied and how.
- Show formation energy diagrams for at least two chemical potential limits (element-rich and element-poor).
- Always plot potential alignment curves in a supplement or verification section.

## Integration with Other Skills

**Upstream (required before defect calculations):**

- [VASP Workflow](../vasp-workflow/SKILL.md) — bulk relaxation (ISIF=3) provides the reference lattice constant, POTCAR, and bulk energy. All VASP job submission, INCAR patterns, and convergence checking protocols apply to defect supercell calculations.
- [DFT Convergence](../dft-convergence/SKILL.md) — ENCUT and k-mesh convergence from the bulk skill carry over directly. EDIFF=1e-6 for relaxation, 1e-8 for single-points. The force threshold EDIFFG=-0.02 eV/Å is standard for defect relaxation.
- [Quantum ESPRESSO Workflow](../quantum-espresso/SKILL.md) — QE can be used as an alternative DFT backend for defect calculations; `ph.x` with `epsil=.true.` provides the dielectric tensor. Post-processing corrections via `pydeft` or equivalent tools.
- [Pymatgen Analysis](../../scientific-data/pymatgen-analysis/SKILL.md) — `DefectGenerator`, `ChemicalPotentialDiagram`, `DefectThermodynamics` (via `doped`), structure manipulation, OUTCAR parsing, and POSCAR generation for all defect structures. Phase diagram construction and competing-phase enumeration.
- [ASE Framework](../../scientific-data/ase-framework/SKILL.md) — structure manipulation, extxyz dataset generation for MLP training (Workflow 8), and MLP calculator evaluation on defect structures.

**Closely related DFT skills:**

- [DFT Phonons](../dft-phonons/SKILL.md) — defect-induced local vibrational modes (LVM) require phonon calculations on defect supercells. The same supercell and INCAR discipline (LREAL=False, EDIFF=1e-8) apply. LVM frequencies are used to identify defect species in HREELS and IR.
- [DFT Surfaces](../dft-surfaces/SKILL.md) — surface defects (vacancies, substitutions at surface sites) combine surface slab methodology with point defect formalism. Charge corrections are modified for slab geometry; see the Kumagai-Oba slab correction extension.
- [DFT Dataset Generation](../dft-dataset-generation/SKILL.md) — defect supercells and their relaxation trajectories are valuable training structures for MLPs. Workflow 8 in this skill generates an extxyz dataset compatible with DFT dataset generation pipelines.
- [High-Throughput DFT](../high-throughput-dft/SKILL.md) — automated defect workflows at scale (screening defects across many host materials) use FireWorks/AiiDA task management with the doped/pydefect infrastructure. [FORWARD REFERENCE — skill not yet created]

**Downstream consumers:**

- [MLP Validation](../../ml-interatomic-potentials/mlp-validation/SKILL.md) — defect formation energies from MLP vs DFT are a key validation metric for potentials targeting ionic conductors, radiation damage, or defect-sensitive functional materials.
- [Scientific Visualization](../../scientific-data/scientific-visualization/SKILL.md) — formation energy diagrams, transition level plots, chemical potential stability diagrams, and Fermi-level dependent concentration plots are the standard outputs of defect thermodynamics analysis.
- [Reproducibility Workflow](../../scientific-data/reproducibility-workflow/SKILL.md) — defect calculations involve many coupled files (bulk POSCAR, supercell POSCARs per defect/charge, dielectric OUTCAR, LOCPOT files for alignment, chemical potential inputs). All must be archived with clear provenance chains.
- [Materials Databases Access](../../scientific-data/materials-databases-access/SKILL.md) — competing phase energies, dielectric constants, and reference defect data can be cross-checked against the Materials Project, AFLOW, or Alexandria databases; chemical potential limits can be constructed from database entries when local DFT calculations are unavailable. [FORWARD REFERENCE — skill may be in progress]

## Key References

1. Freysoldt, C., Neugebauer, J. & Van de Walle, C. G. (2009). "Fully ab initio finite-size corrections for charged-defect supercell calculations." *Physical Review Letters* 102, 016402. The FNV correction scheme; must be cited when using FNV corrections.
2. Kumagai, Y. & Oba, F. (2014). "Electrostatics-based finite-size corrections for first-principles point defect calculations." *Physical Review B* 89, 195205. The Kumagai-Oba correction for anisotropic dielectrics; required for layered or hexagonal materials.
3. Van de Walle, C. G. & Neugebauer, J. (2004). "First-principles calculations for defects and impurities: Applications to III-nitrides." *Journal of Applied Physics* 95, 3851. Comprehensive review of defect formation energy formalism; the primary reference for the formalism.
4. Lany, S. & Zunger, A. (2008). "Assessment of correction methods for the band-gap problem and for finite-size effects in supercell defect calculations: Case studies for ZnO and GaAs." *Physical Review B* 78, 235104. The Lany-Zunger band-gap and finite-size correction scheme.
5. Mosquera-Lois, I., Kavanagh, S. R., Walsh, A. & Scanlon, D. O. (2023). "Identifying the ground state structures of point defects in solids." *npj Computational Materials* 9, 25. The SnB (shakenbreak) methodology for symmetry breaking; cite when using shakenbreak.
6. Kavanagh, S. R. (2024). "doped: Python toolkit for robust and reproducible charged defect supercell calculations." *Journal of Open Source Software* 9, 6433. The doped package; must be cited when using doped for automated defect workflows.
7. Makov, G. & Payne, M. C. (1995). "Periodic boundary conditions in ab initio calculations." *Physical Review B* 51, 4014. The Makov-Payne first-order electrostatic correction.
8. Freysoldt, C., Grabowski, B., Hickel, T., Neugebauer, J., Kresse, G., Janotti, A. & Van de Walle, C. G. (2014). "First-principles calculations for point defects in solids." *Reviews of Modern Physics* 86, 253. Comprehensive review of the field; the standard reference for the entire defect calculation workflow.

[EXPERT REVIEW NEEDED: The FNV vs Kumagai-Oba correction choice for moderately anisotropic materials (e.g., tetragonal perovskites with 10–20% anisotropy) is not always clear-cut. Verify the dielectric anisotropy threshold above which KO is necessary. Also verify that the ISPIN=2 requirement applies uniformly to all closed-shell defects (e.g., V_O^{+2} in closed-shell MgO) or only to open-shell charge states.]
