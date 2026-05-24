---
name: dft-surfaces
domain: electronic-structure
maturity: draft
tools:
  - VASP
  - Quantum ESPRESSO
  - ASE
  - pymatgen
  - numpy
  - matplotlib
tested_versions:
  VASP: "6.4.2"
  Quantum ESPRESSO: "7.3"
  ASE: "3.23.0"
  pymatgen: "2024.11.13"
requires_gpu: false
estimated_runtime: hours-to-days
---

# DFT Surface Calculations

## Description

This skill covers the construction and DFT calculation of surface slab models: Miller-index surface generation, termination selection, slab and vacuum thickness convergence, dipole corrections, surface energy and adsorption energy calculations, work function extraction, and ab initio surface phase diagrams. It integrates VASP and Quantum ESPRESSO as DFT backends with ASE and pymatgen for structure handling and analysis. Invoke this skill when studying adsorption, surface stability, catalytic reactions, surface reconstruction, work functions, or when generating surface structures for ML interatomic potential training.

## Domain Context

A surface is modeled by the **slab approximation**: a finite-thickness periodic layer of solid separated from its periodic images by a vacuum gap. Both the solid and vacuum are periodic in the plane of the surface (xy), and the slab stack is repeated along the surface normal (z). The key approximation is that the slab is thick enough to recover bulk-like behavior in its interior and that the vacuum is wide enough to suppress interaction between periodic images.

**Miller indices and surface orientation:** A surface is specified by a Miller index (hkl), which defines the plane perpendicular to the [hkl] direction of the bulk conventional cell. The surface unit cell may be primitive (1×1) or reconstructed (e.g., Si(001) 2×1, Pt(111) √3×√3-R30°). High-index surfaces (stepped, kinked) have lower symmetry but are more reactive.

**Surface termination:** Many materials have multiple inequivalent terminations at the same (hkl). For example, Al₂O₃(0001) can terminate at Al or O; TiO₂(110) can be stoichiometric or reduced. Termination determines surface energy, adsorption site geometry, and stability under reaction conditions. Each termination must be modeled separately.

**Symmetric vs. asymmetric slabs:**
- *Asymmetric slab*: top and bottom surfaces are different (e.g., top = Al-terminated, bottom = O-terminated). Has a net dipole moment in the z direction for polar surfaces. Requires dipole correction. Cannot extract individual surface energies from a single calculation without a reference.
- *Symmetric slab*: top and bottom surfaces are identical; obtained by inversion symmetry or by cleaving at a mirror plane. No net dipole. Surface energy is unambiguous. Harder to construct for some terminations.

**Polar surfaces:** Surfaces of polar materials (e.g., ZnO(0001), MgO(111), perovskite (001) with different AO/BO₂ terminations) carry a macroscopic electric field that diverges with slab thickness unless the surface is reconstructed, passivated, or a compensating charge is introduced. Polar slabs require special treatment: reconstruction, surface hydroxylation, artificial compensating charge, or very careful reference-state matching. [EXPERT REVIEW NEEDED: the stability criterion for polar surfaces via Tasker classification is well-established but the practical DFT treatment of type-III polar slabs remains code- and system-dependent]

**Dipole correction:** For asymmetric slabs or surfaces with adsorbates on one side only, there is a net dipole in the z direction. This creates a spurious electric field across the vacuum that shifts the electrostatic potential on each side by a different amount, introducing an artificial Δ to the work function and adsorption energies. In VASP, LDIPOL=True + IDIPOL=3 applies an electrostatic correction. In QE, `assume_isolated = 'esm'` or `tefield=.true., dipfield=.true.` serve similar roles.

**Adsorption sites:** Standard adsorption sites on a surface are labeled by the local geometry: top (atop), bridge (between two atoms), hollow (above a surface vacancy — hcp or fcc hollow on FCC(111)). For stepped surfaces, additional sites exist at step edges, kinks, and terraces. Lateral interactions between adsorbates at finite coverage require either larger surface unit cells or explicit correction schemes (cluster expansion, lateral interaction models).

**Slab relaxation:** Only the surface layers (typically top 1–3) are allowed to relax; the bottom layers are frozen to mimic bulk. The number of relaxed layers is a convergence parameter. Using too few frozen layers causes the slab to distort and introduces artificial strain. Using too many relaxed layers increases cost without benefit once the relaxed depth saturates.

## When to Use This Skill

- Computing surface energies and identifying the thermodynamically stable termination of a material.
- Calculating adsorption energies, binding geometries, and adsorption site preferences of molecules or atoms on a surface.
- Determining the work function of a clean or adsorbate-covered surface.
- Building surface phase diagrams (surface coverage as a function of temperature and gas-phase chemical potential).
- Generating training structures for ML interatomic potentials targeting surface reactions or heterogeneous catalysis.
- Studying surface reconstruction, relaxation, and rumpling under ultra-high-vacuum or reaction conditions.
- Cross-referencing with STM, LEED, HREELS, or TPD experimental data on surface structure and adsorption.

## When Not to Use This Skill

- **Amorphous surfaces or grain boundaries** — these are not well-described by a single periodic slab; use large supercells with special quasi-random structures or interface models instead.
- **Liquid–solid or electrochemical interfaces** — the electrolyte environment requires explicit solvent molecules, implicit solvent models (VASPsol, SCCS), or grand-canonical DFT; this skill covers vacuum surfaces only.
- **Surfaces at very high temperature** — thermal disorder, reconstruction dynamics, and anharmonic effects require ab initio MD or machine-learning MD on slab models, not static DFT alone.
- **Catalytic reaction rates** — transition state searches (NEB, dimer) and microkinetic modeling are beyond this skill's scope; use after adsorption energies are established.
- **Very large unit cells (> 200 atoms)** — cost scales linearly with slab atoms times k-points; consider ML potentials for structural sampling and DFT only for single-point corrections.
- **Surfaces of strongly correlated oxides without DFT+U or hybrid correction** — PBE surface energies and adsorption energies for NiO, CoO, Fe₂O₃ surfaces are qualitatively wrong without a Hubbard correction.

## Core Concepts

### Slab Geometry Parameters

- **Slab thickness (N_layers):** Number of atomic layers of the bulk material in the slab. Must be converged by incrementally adding layers until surface energy and adsorption energies change by < 0.01 J/m² and < 0.05 eV, respectively. Typical converged values: 4–6 layers for simple metals, 6–10 layers for oxides. [EXPERT REVIEW NEEDED: convergence criteria are material-class dependent; oxides and polar surfaces often require 10+ layers]
- **Vacuum thickness (d_vac):** Separation between the top of the slab and the bottom of the next periodic image. Must be converged until the electrostatic potential at the vacuum midpoint is flat (< 1 meV variation) and work function converges to < 0.01 eV. Typical: 12–20 Å. Metallic surfaces with long-range tails in the charge density may need 20+ Å.
- **Surface unit cell:** The 2D periodic cell of the surface. Affects coverage (1×1 = one adsorbate per surface site). Larger cells (2×2, √3×√3) model lower coverages and allow for lateral interaction between adsorbates. The cell must be commensurate with the adsorbate overlayer symmetry.
- **k-point mesh for slabs:** Slabs are 2D-periodic (or 3D with thick vacuum). The k-mesh in z (Γ-point in the vacuum direction) is fixed to 1. The xy k-mesh must be at least as dense as the converged bulk mesh projected onto the surface BZ. For metals, a dense 2D mesh (12×12 or denser for small cells) is needed. For insulators/semiconductors, a coarser mesh (6×6) may suffice.

### Surface Energy

The surface energy γ measures the energy cost of creating a surface:

For a **symmetric slab** (both faces identical):
```
γ = (E_slab − N × E_bulk_per_formula) / (2 × A)
```
where A is the surface area of one face, N is the number of formula units, and E_bulk_per_formula is the DFT total energy per formula unit of the bulk. The factor of 2 accounts for two equivalent surfaces.

For an **asymmetric slab** with two different terminations, a system of two equations with different slab thicknesses is needed to decouple γ_top and γ_bottom. Alternatively, use a "reference slab" approach where one surface energy is known from a symmetric calculation.

Units: J/m² (multiply eV/Å² × 16.022 to convert). Typical values: metals 1–3 J/m², oxides 0.5–2.5 J/m².

### Adsorption Energy

```
E_ads = E_{slab+ads} − E_{clean_slab} − E_{adsorbate_ref}
```

where E_{adsorbate_ref} is the total DFT energy of the adsorbate in a well-defined reference state. Sign convention: negative E_ads = exothermic/favorable adsorption (most common convention; always state the sign convention explicitly).

**Reference state choices:**
- Molecular reference: gas-phase molecule (e.g., CO gas, H₂O gas). Corrected for zero-point energy if comparing with experimental ΔH.
- Atomic reference: atomic gas (useful for surface chemistry stoichiometry; not directly comparable to experiment without reaction path).
- Elemental reference: standard states (H₂ gas, O₂ gas, N₂ gas). Used in surface phase diagrams.

**O₂ reference caveat:** PBE severely overbinds O₂ (by ~0.5–1.0 eV per O₂). Use a corrected O₂ reference: E(O) = ½ × (E_H₂O − E_H₂) + ΔG°(H₂O→H₂+½O₂) at standard conditions, or apply the Wang-Ceder-Dabo correction (+1.36 eV per O₂ for GGA). [EXPERT REVIEW NEEDED: the optimal O₂ correction depends on the functional and application; always cite the correction used]

### Dipole Correction

Applied when the slab lacks inversion symmetry about the z midplane. In VASP: add LDIPOL=True and IDIPOL=3. This adds a compensating dipole plate in the vacuum and removes the spurious electric field. The correction shifts the electrostatic potential on each side of the vacuum by an equal and opposite amount. After correction, the work function from each side can be extracted independently.

### Work Function

```
Φ = E_vac − E_F
```

where E_vac is the electrostatic potential in the vacuum midpoint (flat region of the potential) and E_F is the DFT Fermi energy. Both are available in VASP from the LOCPOT file. The work function depends on:
- Surface orientation (e.g., Cu(100) ≠ Cu(111))
- Adsorbate coverage (dipole layer from adsorbates shifts Φ)
- Slab thickness and vacuum thickness convergence

For asymmetric slabs with dipole correction, two work functions are obtained — one per surface. Without dipole correction, only an average (and meaningless for asymmetric slabs) is obtainable.

### Ab Initio Surface Phase Diagram

The surface stability under reactive gas conditions (T, p) is captured by the surface Gibbs free energy:

```
γ(T, p) = (1/2A) × [G_slab(T,p) − Σ_i N_i × μ_i(T,p)]
```

In the DFT approximation (neglecting vibrational entropy of the slab): G_slab ≈ E_DFT_slab. The chemical potentials μ_i(T,p) of gas-phase species (O₂, H₂, CO) are computed from standard thermodynamic relations:

```
μ_O(T,p) = ½ × E_O₂ + ½ × ΔμO(T, p°) + ½ × kT ln(p/p°)
```

where ΔμO(T, p°) is the temperature-dependent chemical potential at standard pressure from NIST-JANAF or GPAW thermochemistry. By computing γ for multiple surface compositions (clean, O-covered, OH-covered, etc.) and plotting vs. μ_O, the most stable surface at each condition is determined.

## Key Workflows

### Workflow 1: Slab Construction and Geometry Setup

```python
# slab_construction.py
# Build a surface slab from a bulk structure, check geometry, and prepare for DFT.
# Uses ASE and pymatgen; VASP output.

from pathlib import Path
import numpy as np
from ase.io import read, write
from ase.build import surface as ase_surface
from ase.constraints import FixAtoms
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core.surface import SlabGenerator

BULK_POSCAR = "POSCAR_bulk"       # fully relaxed, ISIF=3 converged bulk
MILLER = (1, 1, 1)                 # target surface
N_LAYERS = 6                       # slab thickness (to be converged)
VACUUM = 15.0                      # vacuum in Å (to be converged)
N_FIXED = 3                        # bottom layers frozen as bulk
MIN_SLAB_SIZE = 12.0               # minimum slab thickness in Å

# --- Approach A: pymatgen SlabGenerator (recommended for oxides, non-cubic systems) ---
bulk = Structure.from_file(BULK_POSCAR)

gen = SlabGenerator(
    initial_structure=bulk,
    miller_index=MILLER,
    min_slab_size=MIN_SLAB_SIZE,   # in Å; pymatgen adds layers until this is met
    min_vacuum_size=VACUUM,
    center_slab=True,              # center slab in the z cell; simplifies potential plots
    in_unit_planes=False,
    primitive=True,                # use primitive surface cell (smallest area)
    reorient_lattice=True,         # ensure c-axis is along surface normal
    lll_reduce=False,
)

slabs = gen.get_slabs(
    symmetrize=True,               # only return slabs with inversion-symmetric terminations
    tol=0.1,
    bonds=None,
    max_broken_bonds=0,
    repair=False,
)

print(f"Found {len(slabs)} symmetric slab(s) for {MILLER} surface")
for i, slab in enumerate(slabs):
    print(f"  Slab {i}: {len(slab)} atoms, "
          f"thickness={slab.lattice.c - VACUUM:.2f} Å, "
          f"area={slab.surface_area:.2f} Å²")

# Select the lowest-energy-candidate slab (fewest broken bonds)
slab_pmg = slabs[0]

# Convert to ASE for constraint setup and VASP output
adaptor = AseAtomsAdaptor()
slab_ase = adaptor.get_atoms(slab_pmg)

# --- Constraint: fix bottom N_FIXED layers ---
def fix_bottom_layers(atoms, n_fixed_layers, tol=0.5):
    """Fix atoms in the bottom n_fixed_layers (lowest z coordinates)."""
    z_coords = atoms.get_positions()[:, 2]
    z_sorted = np.sort(np.unique(np.round(z_coords, 1)))
    # Identify layer boundaries (group atoms within tol Å)
    layers = []
    current_layer = [z_sorted[0]]
    for z in z_sorted[1:]:
        if z - current_layer[-1] < tol:
            current_layer.append(z)
        else:
            layers.append(np.mean(current_layer))
            current_layer = [z]
    layers.append(np.mean(current_layer))

    if n_fixed_layers >= len(layers):
        raise ValueError(f"Cannot fix {n_fixed_layers} layers: slab only has {len(layers)}")

    z_threshold = layers[n_fixed_layers - 1] + tol
    fixed_indices = [i for i, pos in enumerate(atoms.get_positions()) if pos[2] <= z_threshold]
    print(f"  Total layers: {len(layers)}, fixing {n_fixed_layers} bottom layers "
          f"({len(fixed_indices)} atoms fixed)")
    atoms.set_constraint(FixAtoms(indices=fixed_indices))
    return atoms

slab_ase = fix_bottom_layers(slab_ase, n_fixed_layers=N_FIXED)

# --- Geometry checks ---
cell = slab_ase.get_cell()
pos = slab_ase.get_positions()
z_range = pos[:, 2].max() - pos[:, 2].min()
actual_vacuum = cell[2, 2] - z_range
print(f"\nSlab geometry:")
print(f"  Atoms: {len(slab_ase)}")
print(f"  Cell: a={cell[0,0]:.3f}, b={cell[1,1]:.3f}, c={cell[2,2]:.3f} Å")
print(f"  Slab thickness: {z_range:.2f} Å")
print(f"  Vacuum: {actual_vacuum:.2f} Å")
print(f"  Surface area: {np.linalg.norm(np.cross(cell[0], cell[1])):.2f} Å²")

if actual_vacuum < 12.0:
    print("WARNING: Vacuum < 12 Å; may be insufficient for most materials.")
if z_range < 6.0:
    print("WARNING: Slab < 6 Å; likely too thin for surface energy convergence.")

# Write POSCAR
write("POSCAR_slab", slab_ase, format="vasp", vasp5=True, direct=False)
print(f"\nPOSCAR_slab written.")
```

```python
# --- Approach B: ASE surface builder (simpler, for FCC/BCC metals) ---
# MINIMAL EXAMPLE — use pymatgen for production with complex oxides/terminations
from ase.build import fcc111, fcc100, bcc110
from ase.constraints import FixAtoms
from ase.io import write

# Pt(111) slab, 6 layers, 15 Å vacuum
slab = fcc111("Pt", size=(1, 1, 6), vacuum=7.5, a=3.924)  # vacuum=7.5 on each side = 15 Å total
# Fix bottom 3 layers
z = slab.get_positions()[:, 2]
z_min = z.min()
layer_spacing = (z.max() - z_min) / (6 - 1)
fixed = [i for i, pos in enumerate(slab.get_positions()) if pos[2] < z_min + 2.5 * layer_spacing]
slab.set_constraint(FixAtoms(indices=fixed))
write("POSCAR_Pt111", slab, format="vasp", vasp5=True)
print(f"Pt(111): {len(slab)} atoms, {len(fixed)} fixed")
```

### Workflow 2: INCAR Setup for Surface Calculations (VASP)

Surface calculations have different requirements from bulk calculations. Key differences: the k-mesh has only 1 point in z, dipole correction is often needed, and slab geometry requires selective dynamics (FixAtoms → POSCAR SD flags).

```python
# surface_incar.py
# INCAR settings for different stages of a surface DFT calculation

# --- Stage 1: Slab relaxation (ionic positions, fixed cell) ---
INCAR_RELAX = {
    "ENCUT": 520,           # eV; at least 1.3x ENMAX from POTCAR; increase for surface energies
    "EDIFF": 1e-6,          # eV; for relaxation; tighten to 1e-8 for single-point/adsorption
    "EDIFFG": -0.02,        # eV/Å; force convergence for surface relaxation
    "PREC": "Accurate",
    "NSW": 200,
    "IBRION": 2,            # CG (conjugate gradient); or IBRION=1 (RMM-DIIS) near minimum
    "POTIM": 0.3,           # step size for IBRION=2; reduce to 0.1 for soft surfaces
    "ISIF": 2,              # relax ions only; NEVER use ISIF=3 for slabs (cell should be fixed)
    "ISMEAR": 1,            # Methfessel-Paxton for metals; use 0 (Gaussian) for semiconductors/insulators
    "SIGMA": 0.1,           # eV; 0.1 for metals, 0.05 for semiconductors, 0.01 for insulators
    "ALGO": "Fast",         # for ionic relaxations
    "LREAL": "Auto",        # acceptable for large supercells; use False for force accuracy
    "LDIPOL": True,         # dipole correction along z
    "IDIPOL": 3,            # correct in z (surface normal)
    "LVTOT": True,          # write LOCPOT for work function extraction
    "LVHAR": True,          # write Hartree potential separately (cleaner for work function)
    "LWAVE": True,          # save wavecar for subsequent single-point
    "LCHARG": True,
    "NPAR": 4,
    "KPAR": 2,              # parallel over k-points; set KPAR × NPAR ≤ total cores
}

# --- Stage 2: Single-point calculation for accurate energy/forces ---
INCAR_SP = {
    **INCAR_RELAX,
    "NSW": 0,
    "IBRION": -1,
    "EDIFF": 1e-8,          # tight for surface energies and adsorption energies
    "ALGO": "Normal",
    "LREAL": False,         # exact force calculation
    "LWAVE": False,
    "LCHARG": False,
}

# --- KPOINTS for a Pt(111) 1x1 slab (surface area ~6.8 Å²) ---
# Rule: k-point density ≥ bulk mesh projected to 2D BZ
# For Pt bulk, a converged 12x12x12 mesh on 3.924 Å → ~30 k/Å in each direction
# For surface (in-plane only), need 12x12x1 minimum; increase for small cells
KPOINTS_SLAB = """Automatic mesh
0
Gamma
 12 12 1
 0 0 0
"""
# Note: always use Gamma-centered mesh for slabs; Monkhorst-Pack with an even mesh
# can shift the k-point away from Gamma, missing symmetry at high-symmetry surface points

def write_incar(path, settings):
    with open(path, "w") as f:
        for k, v in settings.items():
            if isinstance(v, bool):
                f.write(f"  {k} = {'T' if v else 'F'}\n")
            elif isinstance(v, float) and abs(v) < 1e-3 and v != 0:
                f.write(f"  {k} = {v:.1e}\n")
            else:
                f.write(f"  {k} = {v}\n")

import os
os.makedirs("slab_relax", exist_ok=True)
write_incar("slab_relax/INCAR", INCAR_RELAX)
with open("slab_relax/KPOINTS", "w") as f:
    f.write(KPOINTS_SLAB)
print("INCAR and KPOINTS written for slab relaxation.")
```

### Workflow 3: Slab Thickness and Vacuum Convergence

Always run explicit convergence tests. The converged slab from this workflow is the reference for all subsequent adsorption calculations.

```python
# slab_convergence.py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from ase.io import read
from ase.io.vasp import read_vasp_out
from pymatgen.core import Structure
from pymatgen.core.surface import SlabGenerator

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

def compute_surface_energy(e_slab, n_formula, e_bulk_per_formula, area_ang2):
    """
    Surface energy in J/m² for a symmetric slab.
    area_ang2: surface area of ONE face in Å²
    """
    eV_per_Ang2_to_J_per_m2 = 16.022
    gamma = (e_slab - n_formula * e_bulk_per_formula) / (2 * area_ang2)
    return gamma * eV_per_Ang2_to_J_per_m2

# --- Assumed directory structure: ---
# convergence/thickness_N/OUTCAR   (N = number of layers)
# convergence/vacuum_V/OUTCAR      (V = vacuum in Å)
# bulk/OUTCAR                      (bulk reference)

BULK_DIR = Path("bulk")
E_BULK = parse_vasp_energy(BULK_DIR / "OUTCAR")
BULK_FORMULA_UNITS = 4   # atoms in bulk cell / atoms per formula unit; adjust per system
E_BULK_PER_FU = E_BULK / BULK_FORMULA_UNITS

# Thickness convergence
layer_counts = [4, 6, 8, 10, 12]
surface_energies_thickness = []
for n in layer_counts:
    outcar = Path(f"convergence/thickness_{n}/OUTCAR")
    if not outcar.exists():
        print(f"  Missing: {outcar} — skipping")
        continue
    e_slab = parse_vasp_energy(outcar)
    atoms = read(f"convergence/thickness_{n}/CONTCAR")
    area = np.linalg.norm(np.cross(atoms.get_cell()[0], atoms.get_cell()[1]))
    n_fu = len([a for a in atoms if a.symbol == "Pt"]) // 1   # atoms per formula unit = 1 for Pt
    gamma = compute_surface_energy(e_slab, n_fu, E_BULK_PER_FU, area)
    surface_energies_thickness.append((n, gamma))
    print(f"  N={n:2d} layers: γ = {gamma:.4f} J/m²")

if len(surface_energies_thickness) >= 2:
    gammas = [g for _, g in surface_energies_thickness]
    converged = all(abs(gammas[i] - gammas[-1]) < 0.01 for i in range(-3, 0) if i < len(gammas))
    print(f"\nThickness convergence: {'CONVERGED' if converged else 'NOT CONVERGED'} "
          f"(threshold 0.01 J/m²)")

# Vacuum convergence (fixed slab thickness, varying vacuum)
vacuum_vals = [10, 12, 15, 18, 20, 25]
work_functions = []
for vac in vacuum_vals:
    outcar = Path(f"convergence/vacuum_{vac}/OUTCAR")
    locpot = Path(f"convergence/vacuum_{vac}/LOCPOT")
    if not outcar.exists():
        continue
    e_slab = parse_vasp_energy(outcar)
    # Work function from LOCPOT (proxy for vacuum convergence)
    # See Workflow 5 for full work function extraction
    work_functions.append((vac, None))   # placeholder; use extract_work_function() below
    print(f"  Vacuum={vac} Å: E_slab = {e_slab:.6f} eV")

# Plot convergence
if surface_energies_thickness:
    layers, gammas = zip(*surface_energies_thickness)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(layers, gammas, "o-", color="steelblue")
    ax.axhline(gammas[-1], ls="--", color="gray", lw=0.8)
    ax.fill_between(ax.get_xlim(), gammas[-1] - 0.01, gammas[-1] + 0.01,
                    alpha=0.15, color="gray", label="±0.01 J/m²")
    ax.set_xlabel("Number of layers")
    ax.set_ylabel("Surface energy (J/m²)")
    ax.set_title("Slab thickness convergence")
    ax.legend()
    plt.savefig("thickness_convergence.png", dpi=150, bbox_inches="tight")
```

### Workflow 4: Adsorption Energy Calculations

```python
# adsorption_energy.py
# Compute adsorption energies for multiple sites and adsorbates.
# Requires: clean slab OUTCAR, adsorbate+slab OUTCARs, gas-phase reference OUTCARs.

import numpy as np
from pathlib import Path
from ase.io import read, write
from ase.build import add_adsorbate
from ase.constraints import FixAtoms

# --- Reference energies (DFT total energies, eV) ---
# Gas-phase molecules computed in large box (15 Å cubic, Gamma-point, spin-polarized for O2/NO)
E_H2    = -6.7712    # H₂ (2 H atoms)
E_H2O   = -14.2282   # H₂O
E_CO    = -14.7754   # CO
E_O2    = -9.8627    # O₂ raw PBE value (overbinds by ~0.5-1 eV per O₂)

# Corrected O reference (Wang-Ceder-Dabo correction for PBE)
# E_O_corrected = 0.5 * (E_H2O - E_H2) + 2.46/2   # via water formation enthalpy
# Better: use the consistent reference scheme from the Materials Project
E_O_ref = 0.5 * (E_H2O - E_H2) + 1.229   # via H₂O→H₂+½O₂ at SHE

def adsorption_energy(e_slab_ads, e_clean_slab, e_gas_ref, stoich=1.0):
    """
    E_ads = E_{slab+ads} - E_clean - stoich * E_gas_ref
    Negative = exothermic (favorable).
    stoich: number of adsorbate molecules (1 for single molecule, 0.5 for dissociative adsorption)
    """
    return e_slab_ads - e_clean_slab - stoich * e_gas_ref

def parse_outcar_energy(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"OUTCAR not found: {path}")
    e = None
    with open(path) as f:
        for line in f:
            if "TOTEN" in line and "eV" in line:
                try:
                    e = float(line.split()[-2])
                except (ValueError, IndexError):
                    pass
    if e is None:
        raise RuntimeError(f"Energy not found in {path}")
    return e

# --- Clean slab reference ---
E_SLAB = parse_outcar_energy("slab_clean/OUTCAR")

# --- Adsorption sites on Pt(111): top, bridge, fcc-hollow, hcp-hollow ---
sites = {
    "top":        ("top",     0.0, 2.0),   # (site_type, x_offset, height)
    "bridge":     ("bridge",  0.0, 1.8),
    "fcc_hollow": ("fcc",     0.0, 1.5),
    "hcp_hollow": ("hcp",     0.0, 1.5),
}

# --- CO adsorption energies ---
results_CO = {}
for site_name, (site_type, x_off, height) in sites.items():
    outcar = Path(f"CO/{site_name}/OUTCAR")
    if not outcar.exists():
        print(f"  Skipping {site_name} (no OUTCAR)")
        continue
    e_slab_co = parse_outcar_energy(outcar)
    e_ads = adsorption_energy(e_slab_co, E_SLAB, E_CO)
    results_CO[site_name] = e_ads
    print(f"  CO/{site_name}: E_ads = {e_ads:.3f} eV")

if results_CO:
    best_site = min(results_CO, key=results_CO.get)
    print(f"\n  Most favorable CO site: {best_site} ({results_CO[best_site]:.3f} eV)")
    # Sanity check: CO on Pt(111) should be ~-1.4 eV (GGA/PBE overestimates magnitude slightly)
    if results_CO[best_site] < -2.5 or results_CO[best_site] > 0.0:
        print("  WARNING: Adsorption energy outside expected range for CO/Pt(111). "
              "Check reference states and slab relaxation.")

# --- Dissociative O adsorption (O₂ → 2O*) ---
outcar_2O = Path("O_dissociative/OUTCAR")
if outcar_2O.exists():
    e_slab_2O = parse_outcar_energy(outcar_2O)
    # Dissociative: E_ads = E(2O*) - E(clean) - E(O2)
    e_ads_dissoc = adsorption_energy(e_slab_2O, E_SLAB, E_O2, stoich=1.0)
    # Per O atom:
    e_ads_per_O = e_ads_dissoc / 2.0
    print(f"\n  Dissociative O₂: E_ads = {e_ads_dissoc:.3f} eV total, {e_ads_per_O:.3f} eV/O")

# --- Coverage effect: compare 1/4 ML vs 1 ML ---
coverage_results = {}
for cov_label in ["1x1", "2x2", "3x3"]:
    outcar = Path(f"CO_coverage/{cov_label}/OUTCAR")
    if not outcar.exists():
        continue
    e = parse_outcar_energy(outcar)
    # Number of CO molecules = 1 for 1x1, 1 for 2x2 (one per unit cell), etc.
    # Surface area scales: A(NxN) = N² × A(1x1)
    coverage_results[cov_label] = adsorption_energy(e, E_SLAB, E_CO)

if len(coverage_results) > 1:
    print("\n  Coverage-dependent adsorption energy (CO):")
    for label, e in coverage_results.items():
        print(f"    {label}: {e:.3f} eV")
    # Lateral interactions: E_lat = E_ads(1x1) - E_ads(2x2)
    if "1x1" in coverage_results and "2x2" in coverage_results:
        e_lat = coverage_results["1x1"] - coverage_results["2x2"]
        print(f"  Lateral interaction (1ML vs 1/4ML): {e_lat:+.3f} eV "
              f"({'repulsive' if e_lat > 0 else 'attractive'})")
```

### Workflow 5: Work Function Extraction from LOCPOT

```python
# work_function.py
# Extract the work function and vacuum level from VASP LOCPOT.
# Requires LVTOT=True or LVHAR=True in INCAR.

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def extract_work_function(locpot_path, outcar_path):
    """
    Parse LOCPOT and OUTCAR to extract the work function.
    Works for both clean surfaces and adsorbate-covered surfaces.
    Returns: (work_function_eV, e_fermi_eV, v_vac_eV)
    """
    # Parse Fermi energy from OUTCAR
    e_fermi = None
    with open(outcar_path) as f:
        for line in f:
            if "E-fermi" in line:
                try:
                    e_fermi = float(line.split()[2])
                except (ValueError, IndexError):
                    pass
    if e_fermi is None:
        raise RuntimeError("Could not parse Fermi energy from OUTCAR")

    # Parse LOCPOT — planar-averaged electrostatic potential
    with open(locpot_path) as f:
        lines = f.readlines()

    # Find cell and grid dimensions
    scale = float(lines[1].strip())
    a1 = np.array(list(map(float, lines[2].split()))) * scale
    a2 = np.array(list(map(float, lines[3].split()))) * scale
    a3 = np.array(list(map(float, lines[4].split()))) * scale
    c_length = np.linalg.norm(a3)   # cell length along z

    # Number of atoms — skip species/count lines
    species_line = lines[5].strip().split()
    counts_line  = lines[6].strip().split()
    try:
        n_atoms = sum(int(c) for c in counts_line)
        header_offset = 9 + n_atoms + 1   # skip atoms block + "Direct" + grid dims line
    except ValueError:
        # VASP5 format: species names on line 5
        n_atoms = sum(int(c) for c in lines[6].strip().split())
        header_offset = 10 + n_atoms + 1

    # Parse grid dimensions
    grid_line = lines[header_offset].strip().split()
    nx, ny, nz = int(grid_line[0]), int(grid_line[1]), int(grid_line[2])

    # Parse potential data
    data = []
    for line in lines[header_offset + 1:]:
        data.extend(map(float, line.split()))
        if len(data) >= nx * ny * nz:
            break
    pot = np.array(data[:nx * ny * nz]).reshape((nx, ny, nz), order='F')

    # Planar average along z
    v_planar = pot.mean(axis=(0, 1))   # average over xy plane for each z slice
    z_coords = np.linspace(0, c_length, nz, endpoint=False)

    # Find vacuum level: flat region far from slab
    # The vacuum is the region with minimal |dV/dz| and maximum V value
    dv = np.abs(np.gradient(v_planar, z_coords))
    # Identify flat region (gradient < 0.01 eV/Å)
    flat_mask = dv < 0.01
    if flat_mask.sum() < 5:
        print("WARNING: No clearly flat vacuum region found. Increase vacuum or check dipole correction.")
        v_vac = v_planar.max()
    else:
        v_vac = v_planar[flat_mask].mean()

    work_function = v_vac - e_fermi
    print(f"Fermi energy:    {e_fermi:.4f} eV")
    print(f"Vacuum level:    {v_vac:.4f} eV")
    print(f"Work function:   {work_function:.4f} eV")

    # Plot electrostatic potential
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(z_coords, v_planar, "b-", lw=1.2)
    ax.axhline(e_fermi, color="red", ls="--", lw=1, label=f"E_F = {e_fermi:.2f} eV")
    ax.axhline(v_vac, color="green", ls="--", lw=1, label=f"V_vac = {v_vac:.2f} eV")
    ax.set_xlabel("z (Å)")
    ax.set_ylabel("Electrostatic potential (eV)")
    ax.set_title(f"Work function: {work_function:.3f} eV")
    ax.legend()
    plt.savefig("work_function.png", dpi=150, bbox_inches="tight")

    return work_function, e_fermi, v_vac

# Usage
wf, ef, vvac = extract_work_function("LOCPOT", "OUTCAR")
print(f"\nWork function: {wf:.3f} eV")
# Reference: Pt(111) experimental WF ≈ 5.65 eV; GGA/PBE gives ~5.6-5.7 eV
```

### Workflow 6: Surface Phase Diagram (Ab Initio Thermodynamics)

```python
# surface_phase_diagram.py
# Compute and plot a surface phase diagram as a function of chemical potential (μ_O).
# Covers RuO2(110) as a realistic example (cf. Reuter & Scheffler 2002).

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# --- DFT energies (eV) ---
E_SLAB_CLEAN  = -312.45    # RuO2(110) clean surface, 2x1 cell, symmetric slab, DFT total
E_RUO2_BULK   = -78.11     # bulk RuO2 (per formula unit)
E_O2_GAS      = -9.86      # O2 gas (GGA; correct for overbinding in practice)
AREA_ANG2     = 12.32 * 6.43   # surface area in Å² (2x1 cell)
AREA_M2       = AREA_ANG2 * 1e-20

# Surface compositions to compare:
# 1. Clean surface
# 2. O-rich (+1 bridging O per formula unit above stoichiometric)
# 3. O-poor (oxygen vacancy, -1 O per formula unit)
compositions = {
    "clean":   {"E_slab": E_SLAB_CLEAN,    "delta_N_O": 0,    "delta_N_Ru": 0, "label": "Clean"},
    "O_rich":  {"E_slab": -318.32,          "delta_N_O": 2,    "delta_N_Ru": 0, "label": "+2O (O-rich)"},
    "O_vac":   {"E_slab": -308.91,          "delta_N_O": -1,   "delta_N_Ru": 0, "label": "-1O (vacancy)"},
    "Oad_cus": {"E_slab": -315.67,          "delta_N_O": 1,    "delta_N_Ru": 0, "label": "+1O (CUS)"},
}

N_FORMULA_UNITS = 4   # per surface cell, per face

def surface_gibbs(e_slab, delta_N_O, mu_O, e_bulk_per_fu, n_fu, area_m2):
    """
    Surface Gibbs free energy in J/m² (DFT approximation, no slab vibrational entropy).
    delta_N_O: excess O atoms relative to stoichiometric slab (positive = O-rich)
    mu_O: O chemical potential in eV (referenced to ½ E_O2)
    """
    # DFT approximation: G_slab ≈ E_slab
    gamma = (e_slab - n_fu * e_bulk_per_fu - delta_N_O * mu_O) / (2 * area_m2)
    eV_to_J = 1.602176634e-19
    return gamma * eV_to_J

# Chemical potential range: from ½ E_O2 (O-rich) to ½ E_O2 - ~3 eV (O-poor)
# μ_O relative to ½ E_O2: range [-3, 0] eV
mu_O_values = np.linspace(-3.0, 0.0, 200)
E_O2_half = 0.5 * E_O2_GAS

gamma_dict = {}
for name, comp in compositions.items():
    gammas = [
        surface_gibbs(
            comp["E_slab"],
            comp["delta_N_O"],
            E_O2_half + dmu,    # μ_O = ½E_O2 + Δμ_O
            E_RUO2_BULK,
            N_FORMULA_UNITS,
            AREA_M2,
        )
        for dmu in mu_O_values
    ]
    gamma_dict[name] = np.array(gammas)

# Plot
fig, ax = plt.subplots(figsize=(7, 5))
colors = {"clean": "steelblue", "O_rich": "firebrick", "O_vac": "darkorange", "Oad_cus": "green"}
for name, gammas in gamma_dict.items():
    ax.plot(mu_O_values, gammas, color=colors.get(name, "gray"),
            lw=1.8, label=compositions[name]["label"])

# Mark the most stable phase at each μ_O
gamma_matrix = np.array(list(gamma_dict.values()))
most_stable = gamma_matrix.argmin(axis=0)
phase_names = list(gamma_dict.keys())
# Shade regions by most stable phase
from matplotlib.patches import Patch
for i, name in enumerate(phase_names):
    mask = most_stable == i
    if mask.any():
        ax.fill_between(mu_O_values, -5, 20, where=mask, alpha=0.08, color=colors.get(name, "gray"))

ax.set_xlabel("ΔμO (eV)  [O-poor ← → O-rich]")
ax.set_ylabel("Surface Gibbs free energy (J/m²)")
ax.set_title("RuO₂(110) surface phase diagram (DFT approximation)")
ax.set_ylim(-2, 15)
ax.axvline(-1.8, color="gray", ls=":", lw=1, label="UHV at 600 K")
ax.legend(loc="upper left", fontsize=9)
plt.savefig("surface_phase_diagram.png", dpi=150, bbox_inches="tight")

# Map μ_O to temperature and pressure (ideal gas O2)
# μ_O(T,p) = ½ E_O2 + ΔμO(T,p°) + ½ kT ln(p/p°)
# ΔμO(T,p°) from JANAF tables (pre-tabulated):
from scipy.constants import k as kB, N_A
kB_eV = kB / 1.602176634e-19

def dmu_O_from_conditions(T_K, p_Pa, p0_Pa=101325.0):
    """
    Approximate ΔμO from temperature and partial pressure of O2.
    Uses tabulated JANAF values at 300 K as baseline; interpolate for other T.
    [EXPERT REVIEW NEEDED: use NIST-JANAF tables for accurate thermochemistry]
    """
    # Rough tabulation (eV): ΔμO at standard pressure for O2
    T_ref   = np.array([200, 300, 400, 600, 800, 1000, 1200])
    dmu_ref = np.array([-0.08, -0.20, -0.34, -0.60, -0.88, -1.17, -1.47])
    dmu_0   = np.interp(T_K, T_ref, dmu_ref)
    # Pressure correction
    dmu_p = 0.5 * kB_eV * T_K * np.log(p_Pa / p0_Pa)
    return dmu_0 + dmu_p

# Example: at 600 K, 1e-6 bar O2 partial pressure
T, p = 600, 1e-6 * 1e5  # Pa
dmu = dmu_O_from_conditions(T, p)
print(f"\nAt T={T} K, p_O2={p:.0e} Pa: ΔμO = {dmu:.3f} eV")
stable_idx = np.argmin([gamma_dict[n][np.argmin(abs(mu_O_values - dmu))]
                         for n in phase_names])
print(f"Most stable surface: {phase_names[stable_idx]}")
```

### Workflow 7: Surface Adsorbate Structure Preparation (ASE + pymatgen)

```python
# adsorbate_setup.py
# Systematically generate all inequivalent adsorption configurations on a surface.
# Uses pymatgen AdsorbateSiteFinder for symmetry-inequivalent site enumeration.

import numpy as np
from pathlib import Path
from pymatgen.core import Structure, Molecule
from pymatgen.analysis.adsorption import AdsorbateSiteFinder
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.vasp import Poscar
from ase.io import write
from ase.constraints import FixAtoms

SLAB_FILE = "POSCAR_slab_relaxed"   # relaxed clean slab (pymatgen Structure)
ADSORBATE = "CO"                     # adsorbate species
HEIGHT = 1.8                         # initial placement height above topmost surface atom (Å)
N_FIXED_LAYERS = 3                   # bottom layers frozen

slab = Structure.from_file(SLAB_FILE)
asf = AdsorbateSiteFinder(slab, selective_dynamics=True)

# Get all symmetry-inequivalent adsorption sites
sites = asf.find_adsorption_sites(
    distance=HEIGHT,
    symm_reduce=0.01,    # symmetry tolerance; 0.01 Å for high-symmetry surfaces
    near_reduce=0.01,
    positions=["ontop", "bridge", "hollow"],
)

print(f"Found {sum(len(v) for v in sites.values())} adsorption sites:")
for site_type, site_list in sites.items():
    print(f"  {site_type}: {len(site_list)} site(s)")

# Build CO molecule (C down, O up — most common binding geometry on metals)
co = Molecule(["C", "O"], [[0, 0, 0], [0, 0, 1.15]])

out_dir = Path("adsorbate_structures")
out_dir.mkdir(exist_ok=True)

adaptor = AseAtomsAdaptor()
config_idx = 0
for site_type, site_list in sites.items():
    for i, site_coords in enumerate(site_list):
        # Place adsorbate
        ads_structs = asf.generate_adsorption_structures(
            co,
            repeat=None,
            min_lw=5.0,     # minimum cell width; expand supercell if needed
            find_args={"positions": [site_type], "symm_reduce": 0.01},
        )
        # Generate for specific site only (one at a time)
        for j, ads in enumerate(ads_structs[:1]):   # take first (closest match)
            ase_atoms = adaptor.get_atoms(ads)
            # Apply fixed constraints
            z = ase_atoms.get_positions()[:, 2]
            z_layers = np.sort(np.unique(np.round(z, 1)))
            if len(z_layers) >= N_FIXED_LAYERS:
                z_thresh = z_layers[N_FIXED_LAYERS - 1] + 0.5
                fixed_idx = [k for k, p in enumerate(ase_atoms.get_positions()) if p[2] <= z_thresh]
                ase_atoms.set_constraint(FixAtoms(indices=fixed_idx))

            poscar_path = out_dir / f"POSCAR_{site_type}_{i:02d}"
            write(str(poscar_path), ase_atoms, format="vasp", vasp5=True, direct=False)
            config_idx += 1

print(f"\n{config_idx} POSCAR files written to {out_dir}/")
print("Submit each as a separate VASP relaxation (Stage 1 INCAR).")
print("After relaxation, run single-point (Stage 2 INCAR) for accurate energies.")
```

### Workflow 8: MLP Surface Validation

ML interatomic potentials trained on bulk data often fail on surfaces. This workflow tests MLP surface predictions against DFT references.

```python
# mlp_surface_validation.py
# Compare MLP vs DFT: surface energies, adsorption energies, relaxed geometries.

import numpy as np
from pathlib import Path
from ase.io import read
from ase.optimize import LBFGS
from ase.constraints import FixAtoms

def relax_slab_with_mlp(slab_poscar, calc, fmax=0.02, max_steps=300):
    """Relax slab with an ASE MLP calculator; return relaxed atoms and energy."""
    atoms = read(slab_poscar, format="vasp")
    atoms.calc = calc
    # Preserve existing constraints (FixAtoms from POSCAR SD flags or prior setup)
    if not atoms.constraints:
        # Default: fix bottom half
        z = atoms.get_positions()[:, 2]
        z_mid = (z.max() + z.min()) / 2
        atoms.set_constraint(FixAtoms(indices=[i for i, p in enumerate(atoms.get_positions())
                                                if p[2] < z_mid]))
    opt = LBFGS(atoms, logfile=None)
    opt.run(fmax=fmax, steps=max_steps)
    if not opt.converged():
        print(f"  WARNING: MLP relaxation not converged after {max_steps} steps")
    return atoms, atoms.get_potential_energy()

def surface_energy_from_ase(atoms_slab, e_slab, e_bulk_per_atom):
    """Surface energy from ASE atoms object."""
    n_atoms = len(atoms_slab)
    cell = atoms_slab.get_cell()
    area = np.linalg.norm(np.cross(cell[0], cell[1]))
    gamma = (e_slab - n_atoms * e_bulk_per_atom) / (2 * area)
    return gamma * 16.022   # eV/Å² → J/m²

def compare_relaxed_geometry(atoms_dft, atoms_mlp, label=""):
    """Compare interlayer spacing and surface rumpling between DFT and MLP."""
    z_dft = np.sort(atoms_dft.get_positions()[:, 2])
    z_mlp = np.sort(atoms_mlp.get_positions()[:, 2])
    if len(z_dft) != len(z_mlp):
        print(f"  {label}: atom count mismatch ({len(z_dft)} vs {len(z_mlp)})")
        return
    # Top-layer z spread (rumpling proxy)
    rumple_dft = z_dft[-len(z_dft)//6:].max() - z_dft[-len(z_dft)//6:].min()
    rumple_mlp = z_mlp[-len(z_mlp)//6:].max() - z_mlp[-len(z_mlp)//6:].min()
    rmsd = np.sqrt(np.mean((z_dft - z_mlp)**2))
    print(f"  {label}: z-RMSD={rmsd:.3f} Å, rumpling DFT={rumple_dft:.3f} Å, MLP={rumple_mlp:.3f} Å")

# --- Load MLP calculator ---
try:
    from mace.calculators import MACECalculator
    calc = MACECalculator(model_paths=["path/to/mace.model"], device="cpu")
except ImportError:
    print("MACE not available; substitute with CHGNet, SevenNet, or ALIGNN")
    calc = None

if calc is not None:
    # 1. Surface energy comparison
    atoms_mlp, e_mlp = relax_slab_with_mlp("POSCAR_slab", calc)
    # DFT reference energy parsed from OUTCAR
    e_dft = -312.45   # replace with parse_outcar_energy("slab_clean/OUTCAR")
    e_bulk_per_atom_dft  = -5.820   # DFT bulk energy per atom
    e_bulk_per_atom_mlp  = atoms_mlp.calc.get_potential_energy() / len(atoms_mlp)   # from bulk run

    gamma_dft = (e_dft - len(atoms_mlp) * e_bulk_per_atom_dft) / (2 * np.linalg.norm(
        np.cross(atoms_mlp.get_cell()[0], atoms_mlp.get_cell()[1]))) * 16.022
    gamma_mlp = surface_energy_from_ase(atoms_mlp, e_mlp, e_bulk_per_atom_mlp)
    print(f"Surface energy: DFT={gamma_dft:.3f} J/m², MLP={gamma_mlp:.3f} J/m², "
          f"diff={abs(gamma_dft-gamma_mlp):.3f} J/m²")
    if abs(gamma_dft - gamma_mlp) > 0.2:
        print("  WARNING: Surface energy discrepancy > 0.2 J/m² — MLP may not capture surface physics")

    # 2. Adsorption energy comparison
    atoms_dft_co = read("CO/top/CONTCAR", format="vasp")
    atoms_mlp_co, e_mlp_co = relax_slab_with_mlp("CO/top/POSCAR", calc)
    e_co_gas_mlp = -14.78   # MLP CO reference (from a CO gas-phase ASE calc with same MLP)
    e_ads_dft = -0.87       # from DFT adsorption_energy() above
    e_ads_mlp = e_mlp_co - e_mlp - e_co_gas_mlp
    print(f"\nCO top-site adsorption: DFT={e_ads_dft:.3f} eV, MLP={e_ads_mlp:.3f} eV, "
          f"diff={e_ads_dft-e_ads_mlp:+.3f} eV")

    # 3. Geometry comparison
    atoms_dft_slab = read("slab_clean/CONTCAR", format="vasp")
    compare_relaxed_geometry(atoms_dft_slab, atoms_mlp, label="clean slab geometry")
```

## Common Failure Modes

| Failure Mode | Symptoms | Likely Cause | Diagnostic | Fix |
|---|---|---|---|---|
| Slab too thin | Surface energy changes by > 0.05 J/m² when adding layers; adsorption energies not converged | Too few layers; surface perturbation reaches bottom | Run thickness convergence (Workflow 3); check if bottom layers still show surface-like distortion | Add 2 layers at a time until convergence < 0.01 J/m² |
| Insufficient vacuum | Electrostatic potential not flat in vacuum; work function shows thickness dependence; forces on topmost atom are non-zero | Periodic images interact through vacuum | Check LOCPOT planarly averaged potential for flatness; must be flat over ≥ 5 Å | Increase vacuum to 15–20 Å; use dipole correction |
| Missing dipole correction | Work function from each face differs by wrong amount; adsorption on one face shifts energies on the other; net dipole in OUTCAR | Asymmetric slab has net dipole in z; spurious field crosses vacuum | Check DIPOL tag and NELM output for net cell dipole; check LVHAR potential asymmetry | Add LDIPOL=True, IDIPOL=3 |
| Wrong adsorbate reference state | CO adsorption energy off by 0.5–1 eV vs literature | O₂ overbinding not corrected; wrong spin state for gas molecules; incorrect box size for gas reference | Recalculate gas reference with identical ENCUT, spin-polarized for triplet O₂ and doublet NO; check vacuum box ≥ 12 Å | Apply Wang-Ceder-Dabo correction for O; use same ENCUT and PREC as slab |
| ISIF=3 applied to slab | Cell volume and shape change; slab contracts along z; unphysical pressure | Variable-cell relaxation not appropriate for slabs | Check ISIF in INCAR; look for large cell changes in OUTCAR | Set ISIF=2 for all slab calculations; only use ISIF=3 for bulk |
| Unconverged k-mesh | Surface energy and adsorption energy oscillate; Γ-centered mesh with even N gives non-Γ k-points | k-mesh too coarse for metallic surface; or Monkhorst-Pack without Γ centering | Run k-mesh convergence (8×8×1, 12×12×1, 16×16×1) and check convergence | Use Γ-centered mesh; increase density until ΔE_ads < 0.01 eV |
| Non-equivalent slab surfaces compared | Surface energies or adsorption energies inconsistent across cells | Different terminations or different relaxation states in slab A vs slab B | Visualize both surfaces; check adsorbate coverage, surface composition, and relaxation depth | Use identical slab setups (same N_layers, N_fixed, vacuum, termination) |
| Polar surface without correction | Surface energy diverges with slab thickness; VASP may have SCF convergence issues | Type-III polar slab (Tasker type 3) with uncompensated surface charge | Check if surface is Tasker type 3; plot E vs thickness — linear growth signals divergence | Symmetrize, passivate, or apply fractional-charge correction; see Tasker (1979) |
| Artificial reconstruction in small cell | Adsorbate distorts lattice by more than ±5% in-plane; surface cell changes shape | Adsorbate-induced strain confined to small cell; image interactions | Check cell dimensions vs bulk; compute strain tensor | Use a larger surface supercell (2×2 or 3×3) |
| Fixed too few bottom layers | Bottom atoms relax significantly; slab behaves like a molecule | Only 1–2 bottom layers frozen; thin slab floats and re-relaxes bulk | Check force on supposedly fixed atoms in OUTCAR; monitor z-displacement of bottom layer | Fix at least N_fixed = round(N_layers/2) layers; minimum 2 |
| LREAL=Auto for forces | Force errors ~1–5 meV/Å; adsorption geometry slightly off | Real-space projection for large supercells introduces force noise | Compare forces from LREAL=Auto vs LREAL=False single-point | Use LREAL=False for final single-point calculations of adsorption energies |
| Spin state not converged | Magnetic surface (NiO, Fe₂O₃, FexOy) gives wrong energy; DOS looks metallic | Initial magnetic moments wrong; ISPIN=1 for magnetic material | Check MAGMOM in INCAR; check NUPDOWN; run with different MAGMOM initializations | Set ISPIN=2, MAGMOM per atom, LORBIT=11; use DFT+U for correlated oxides |
| Site selection bias | Claiming one site is most stable from only one calculation | Only top site computed; bridge or hollow may be more favorable | Compute all symmetry-inequivalent sites from AdsorbateSiteFinder | Always screen all inequivalent sites; report lowest-energy site |
| MLP surface energy divergence from DFT | MLP γ differs by > 0.3 J/m² from DFT; surface structure qualitatively wrong | MLP trained only on bulk; surface environments extrapolate | Check MLP uncertainty at surface atoms; compare force RMSE on surface structures | Add surface slab structures to training set; use active learning on surface |

## Best Practices

**Structure setup:**
- Always start from a fully relaxed bulk structure (ISIF=3 convergence to EDIFFG=-1e-3 or tighter) before constructing slabs. A strained bulk geometry propagates errors through the entire surface calculation.
- Use `pymatgen.core.surface.SlabGenerator` with `symmetrize=True` when possible — symmetric slabs with no net dipole give unambiguous surface energies.
- For non-symmetric terminations, always construct a complementary slab with the inverse termination to decouple γ_top and γ_bottom.
- Check that the surface area (in-plane cell) is large enough to avoid lateral adsorbate-adsorbate interactions. Rule of thumb: adsorbate–adsorbate distance > 6 Å for non-covalent species; for CO/NO/OH, use at least 2×2 cells.

**Convergence:**
- Always run both thickness and vacuum convergence before reporting surface energies or adsorption energies. Treat these as mandatory, not optional.
- k-point density in the surface plane must match the bulk-equivalent density; the bulk k-mesh projects to a 2D mesh for the surface cell. A 12×12×12 bulk mesh on a 1×1 cell → 12×12×1 for the 1×1 slab.
- For metals, use Methfessel-Paxton smearing (ISMEAR=1, SIGMA=0.1). For semiconductors and insulators, use Gaussian (ISMEAR=0, SIGMA=0.05). Always extrapolate to σ→0 if using large smearing.

**DFT settings for surfaces:**
- ISIF=2 (relax ions only, fixed cell). Never use ISIF=3 for slabs.
- LDIPOL=True, IDIPOL=3 for all asymmetric slabs and any calculation with adsorbates on one side only.
- LVTOT=True or LVHAR=True to enable work function extraction from LOCPOT.
- Use LREAL=Auto for relaxation of large supercells (> ~80 atoms), but always do a final single-point with LREAL=False for accurate energies and forces.
- Set NSW=0 for all single-point energy calculations.

**Reference states:**
- Always document which reference states were used for adsorption energies (molecular, atomic, or elemental gas phase; corrected or uncorrected for O₂ overbinding).
- For oxygen-containing species: use the water-formation route for O reference or explicitly state the O₂ correction applied.
- Report adsorption energies with the sign convention defined (negative = exothermic is standard in heterogeneous catalysis, but some communities use the reverse).
- Compare against multiple experimental or high-level theoretical benchmarks before reporting new adsorption energies as reference values.

**MLP validation on surfaces:**
- Never deploy an MLP for surface chemistry based on bulk validation alone. Surface environments differ fundamentally from bulk.
- Always compute surface energy and at least one adsorption energy with both DFT and MLP for each new surface.
- Add surface training structures (clean slab, adsorbate-covered, defect-containing) to the training set if surface errors exceed 0.15 J/m² or 0.1 eV respectively.

**Reproducibility:**
- Archive: bulk POSCAR, slab POSCAR (all terminations), INCAR for each stage, KPOINTS, POTCAR SHA, phonopy.yaml if used, BORN file if polar. The slab construction script and all parameters (Miller index, N_layers, N_fixed, vacuum, displacement, coverage) must be recorded.

## Integration with Other Skills

**Upstream (required before surface calculations):**

- [VASP Workflow](../vasp-workflow/SKILL.md) — bulk relaxation (ISIF=3) provides the equilibrium lattice constant and POTCAR for the surface. All VASP INCAR patterns, job submission, and convergence checking protocols from the VASP workflow skill apply to slab calculations.
- [DFT Convergence](../dft-convergence/SKILL.md) — ENCUT, k-mesh, and EDIFF convergence for bulk must be established before constructing slabs. Phonon-grade convergence (ENCUT≥550, EDIFF=1e-8, LREAL=False) is required for accurate surface energies and adsorption energies.
- [Quantum ESPRESSO Workflow](../quantum-espresso/SKILL.md) — QE `pw.x` is an alternative DFT backend for surface calculations; pseudopotential choice and convergence settings follow the QE skill. `assume_isolated` and `tefield` settings replace VASP LDIPOL for dipole corrections.
- [ASE Framework](../../scientific-data/ase-framework/SKILL.md) — slab construction (Workflow 1, 7), adsorbate placement, constraint setup, and file conversion all use ASE. The `FixAtoms` constraint and `AdsorbateSiteFinder` are the core ASE/pymatgen tools for surface structure preparation.
- [Pymatgen Analysis](../../scientific-data/pymatgen-analysis/SKILL.md) — `SlabGenerator`, `AdsorbateSiteFinder`, and `Structure.from_file` for slab and adsorbate configuration management. Cross-referencing against the Materials Project surface energy database for validation. [FORWARD REFERENCE — skill may be in progress]

**Downstream consumers (surface outputs feed into):**

- [DFT Phonons](../dft-phonons/SKILL.md) — surface phonons and adsorbate vibrational modes (e.g., C-O stretch frequency as a Brønsted-Evans-Polanyi proxy) are computed using the phonon skill on slab supercells. Imaginary surface modes distinguish reconstruction from slab thickness artifacts.
- [DFT Defects](../dft-defects/SKILL.md) — surface vacancies (O vacancies in oxide surfaces, surface cation vacancies) are a subclass of defects requiring the defect skill's charge correction and formation energy framework. [FORWARD REFERENCE — skill not yet created]
- [DFT Dataset Generation](../dft-dataset-generation/SKILL.md) — surface slab structures, adsorption configurations, and displaced supercells for surface phonons form key components of DFT training datasets for MLP surface chemistry.
- [MLP Validation](../../ml-interatomic-potentials/mlp-validation/SKILL.md) — surface energy and adsorption energy comparison (Workflow 8) is a required validation step for MLPs targeting surface reactions. The same metrics (RMSE, MAE, sign of adsorption energy) used for bulk validation apply here but with surface-specific reference states.
- [MACE Training](../../ml-interatomic-potentials/mace-training/SKILL.md) — surface slab single-points and displaced supercells are added to MACE training datasets when surface energy or adsorption energy errors exceed tolerance. Workflow 8 identifies which surface environments require additional training data.
- [Active Learning MLP](../../ml-interatomic-potentials/active-learning-mlp/SKILL.md) — active learning on surface trajectories (MD on slabs, NEB path sampling) identifies surface configurations underrepresented in the training set. High-uncertainty surface configurations from MLP ensemble disagreement are queued for DFT labeling.

**Visualization and reproducibility:**

- [Scientific Visualization](../../scientific-data/scientific-visualization/SKILL.md) — electrostatic potential plots (work function, dipole correction), surface phase diagrams, adsorption energy heatmaps by site, and comparison plots between DFT and MLP surface quantities.
- [Reproducibility Workflow](../../scientific-data/reproducibility-workflow/SKILL.md) — surface calculations involve many coupled files (bulk POSCAR → slab POSCAR → adsorbate POSCARs → OUTCARs → LOCPOT) that must be versioned and archived with clear provenance from bulk through adsorption.

## Key References

1. Reuter, K. & Scheffler, M. (2002). "Composition, structure, and stability of RuO₂(110) as a function of oxygen pressure." *Physical Review B* 65, 035406. The seminal ab initio thermodynamics surface phase diagram paper; must be cited when using Workflow 6.
2. Tasker, P. W. (1979). "The stability of ionic crystal surfaces." *Journal of Physics C: Solid State Physics* 12, 4977–4984. Classification of polar vs. non-polar surfaces (Tasker types I–III); foundational for polar surface treatment.
3. Bader, R. F. W. (1990). *Atoms in Molecules: A Quantum Theory*. Oxford University Press. Charge partitioning for surface adsorption charge transfer analysis; widely used alongside Bader charge code.
4. Methfessel, M. & Paxton, A. T. (1989). "High-precision sampling for Brillouin-zone integration in metals." *Physical Review B* 40, 3616. Methfessel-Paxton smearing for metals; the standard for metallic surface calculations.
5. Norskov, J. K., Bligaard, T., Logadottir, A., Bahn, S., Hansen, L. B., Bollinger, M., Bengaard, H., Hammer, B., Sljivancanin, Z., Mavrikakis, M., Xu, Y., Dahl, S. & Jacobsen, C. J. H. (2002). "Universality in heterogeneous catalysis." *Journal of Catalysis* 209, 275–278. Adsorption energy descriptor framework; motivates why accurate surface adsorption energies matter.
6. Wang, L., Maxisch, T. & Ceder, G. (2006). "Oxidation energies of transition metal oxides within the GGA+U framework." *Physical Review B* 73, 195107. The O₂ overbinding correction reference; cite when applying the Wang-Ceder-Dabo correction.
7. Henkelmann, G., Arnaldsson, A. & Jonsson, H. (2006). "A fast and robust algorithm for Bader decomposition of charge density." *Computational Materials Science* 36, 354–360. Bader charge analysis code and method.
8. Setyawan, W. & Curtarolo, S. (2010). "High-throughput electronic band structure calculations: Challenges and tools." *Computational Materials Science* 49, 299–312. BZ path conventions used for surface phonon band structures.

[EXPERT REVIEW NEEDED: Verify that the dipole correction implementation in VASP (LDIPOL/IDIPOL=3) is equivalent to the ESM method in Quantum ESPRESSO for common benchmarks. Confirm that the O₂ correction scheme chosen (+1.36 eV vs. water-formation-route) is appropriate for PBE and not PBEsol. Review whether MLP surface validation thresholds (0.2 J/m² for surface energy, 0.1 eV for adsorption energy) are appropriate for all material classes or should be tightened for catalysis applications.]
