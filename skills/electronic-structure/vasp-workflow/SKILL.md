---
name: vasp-workflow
domain: electronic-structure
maturity: draft
tools:
  - VASP
  - ASE
  - pymatgen
  - vaspkit
tested_versions:
  VASP: "6.3.2"
  ASE: "3.23.0"
  pymatgen: "2024.11.13"
  vaspkit: "1.3.5"
requires_gpu: false
estimated_runtime: hours-to-days
---

# VASP Workflow

## Description

This skill covers end-to-end DFT calculations using VASP 6, from input file construction through convergence testing to production runs. It is the entry point for electronic structure work and the primary source of training data for ML interatomic potentials. Invoke this skill when an agent needs to set up, run, analyze, or debug a VASP calculation, or when generating DFT reference data for downstream tasks.

## Domain Context

VASP solves the Kohn-Sham equations of density functional theory on a plane-wave basis set with PAW (Projector Augmented Wave) pseudopotentials. The key approximation is the exchange-correlation functional: it captures electron-electron interactions beyond the Hartree level, but no functional is exact for all systems.

Core assumptions that must be understood before using this workflow:

- **Plane-wave basis convergence:** The basis set size is controlled by ENCUT (cutoff energy in eV). Increasing ENCUT systematically improves completeness until results converge. The PAW potential files specify a recommended ENCUT; production runs should test convergence above this value.
- **Brillouin zone sampling:** The k-point mesh controls sampling of reciprocal space. Metals require denser meshes than insulators due to Fermi surface features. Convergence is material-dependent.
- **Pseudopotential choice:** VASP ships multiple PAW datasets per element (e.g., `Fe`, `Fe_pv`, `Fe_sv`). The choice affects which electrons are treated as valence, with direct consequences for accuracy and cost.
- **Exchange-correlation functional:** PBE systematically underestimates band gaps and over-delocalizes d/f electrons. PBEsol corrects surface energies. SCAN is more accurate for many systems but harder to converge. HSE06 opens band gaps but is far more expensive. DFT+U is empirical but fast.
- **Spin polarization:** Magnetic systems require ISPIN=2 and careful MAGMOM initialization. Omitting spin polarization in Fe, Co, Ni, or Mn-containing systems will produce wrong energies.
- **Periodic boundary conditions:** VASP assumes 3D periodicity. Surfaces, defects, and molecules require appropriate vacuum spacing and dipole corrections.

The Born-Oppenheimer approximation is assumed throughout: nuclear motion is treated separately from electronic structure.

## When to Use This Skill

- Ground-state energy, forces, and stress calculations for periodic solids, surfaces, or large molecules in a periodic box.
- Equation-of-state fitting, structural relaxation, or transition state searches on crystalline systems.
- Generating DFT reference data (energies, forces, stress tensors) for ML potential training.
- Band structure, density of states, and electronic structure analysis.
- Phonon calculations (using finite displacements or DFPT) for thermodynamic properties.
- High-throughput DFT workflows using pymatgen or AiiDA.

## When Not to Use This Skill

- **Molecular systems in vacuum** where ORCA, Gaussian, or PySCF are more appropriate and cheaper — VASP forces periodic boundary conditions.
- **Strongly correlated systems** (e.g., Mott insulators like NiO, VO2 near transition) where PBE and DFT+U give qualitatively wrong results and DMFT or multi-reference methods are needed.
- **Excited states and optical spectra** beyond the independent-particle level — use BSE (via VASP's BSE module or ABINIT) or TDDFT for this.
- **Very large systems (> 500–1000 atoms)** where linear-scaling DFT codes (ONETEP, OpenMX) or ML potentials are more tractable.
- **High-accuracy thermochemistry** requiring sub-kJ/mol precision — plane-wave DFT at GGA level has ~5–20 kJ/mol errors on reaction energies.
- **Systems where van der Waals interactions are dominant** without adding dispersion correction — bare PBE misses London dispersion entirely.

## Core Concepts

- **PAW method:** Reconstructs all-electron wavefunctions near nuclei from smooth pseudo-wavefunctions. Cheaper than norm-conserving pseudopotentials for the same accuracy. POTCAR files encode the PAW dataset.
- **ENCUT:** Plane-wave energy cutoff in eV. Must be converged for each property of interest. Energy differences converge faster than absolute energies.
- **k-point mesh:** Monkhorst-Pack grid for Brillouin zone integration. Convergence requires testing; metals need finer meshes than insulators. Gamma-only is appropriate for molecules in large boxes or for preliminary testing only.
- **Smearing:** Electronic temperature applied to occupation numbers to aid SCF convergence in metals. Methfessel-Paxton (ISMEAR=1 or 2) for metals; Gaussian (ISMEAR=0) for semiconductors and insulators; tetrahedron (ISMEAR=-5) for accurate DOS and final total energy of non-metals.
- **SCF convergence (EDIFF):** Convergence threshold for the self-consistent field loop on the electronic energy. Typical range: 1e-5 to 1e-6 eV.
- **Ionic convergence (EDIFFG):** Convergence criterion for ionic relaxation. Negative value means forces in eV/Å; positive means energy change in eV. Force convergence (negative EDIFFG) is generally more reliable.
- **IBRION:** Selects the ionic relaxation algorithm. IBRION=2 (conjugate gradient) for most cases; IBRION=1 (RMM-DIIS) for near-minimum structures; IBRION=-1 for single-point.
- **DFT+U (LDAUU/LDAUJ):** Adds a Hubbard U correction to correct d/f electron self-interaction error in transition metal oxides. The U value is empirical and must be validated against experiment or higher-level theory.
- **NELM/NELMIN:** Maximum and minimum number of SCF iterations. Default NELM=60 is often insufficient for difficult systems.
- **ALGO:** SCF algorithm. ALGO=Normal (Davidson) is robust; ALGO=Fast (RMM-DIIS) is faster but can diverge; ALGO=All is most stable for difficult convergence.

## Key Workflows

### Workflow 1: ENCUT and k-point convergence protocol

Establish material-specific convergence before production runs. This must be done for each new material class — convergence parameters from one system cannot be reliably transferred to another.

```python
from pymatgen.core import Structure
from pymatgen.io.vasp.inputs import Incar, Kpoints, Poscar, Potcar
from pymatgen.io.vasp.sets import MPRelaxSet
from pathlib import Path
import subprocess

# Load structure (e.g., from CIF or Materials Project)
structure = Structure.from_file("POSCAR")

# --- ENCUT convergence scan ---
encut_values = [400, 450, 500, 550, 600]  # eV; start above POTCAR ENMAX, scan upward

for encut in encut_values:
    calc_dir = Path(f"encut_{encut}")
    calc_dir.mkdir(exist_ok=True)

    vis = MPRelaxSet(structure, user_incar_settings={
        "ENCUT": encut,
        "EDIFF": 1e-6,
        "NSW": 0,          # single-point, no ionic relaxation
        "ISMEAR": 0,       # Gaussian smearing; use -5 for insulators with tetrahedron
        "SIGMA": 0.05,
        "IBRION": -1,
        "LWAVE": False,
        "LCHARG": False,
        "PREC": "Accurate",
        "ALGO": "Normal",
    })
    vis.write_input(str(calc_dir))

# Convergence criterion: total energy change < 1 meV/atom between successive ENCUT values.
# For forces and phonons, tighten to < 0.1 meV/atom [EXPERT REVIEW NEEDED for specific systems].

# --- k-point convergence scan ---
# Use a fixed ENCUT from the converged step above.
kpoint_grids = [(4,4,4), (6,6,6), (8,8,8), (10,10,10), (12,12,12)]

for grid in kpoint_grids:
    label = "x".join(map(str, grid))
    calc_dir = Path(f"kpts_{label}")
    calc_dir.mkdir(exist_ok=True)

    kpoints = Kpoints.gamma_automatic(grid, (0, 0, 0))
    # Write inputs manually or via MPRelaxSet with kpoints override
    kpoints.write_file(str(calc_dir / "KPOINTS"))
```

Convergence analysis:

```python
from pymatgen.io.vasp.outputs import Vasprun
from pathlib import Path

results = {}
for path in sorted(Path(".").glob("encut_*/vasprun.xml")):
    try:
        v = Vasprun(str(path))
        encut = v.incar["ENCUT"]
        natoms = len(v.final_structure)
        energy_per_atom = v.final_energy / natoms
        results[encut] = energy_per_atom
        print(f"ENCUT={encut:5d} eV  E/atom={energy_per_atom:.6f} eV")
    except Exception as e:
        print(f"Failed to parse {path}: {e}")

# Check: is |E(ENCUT_n) - E(ENCUT_n-1)| < 1 meV/atom?
encuts = sorted(results)
for i in range(1, len(encuts)):
    diff_meV = abs(results[encuts[i]] - results[encuts[i-1]]) * 1000
    print(f"  ENCUT {encuts[i-1]} -> {encuts[i]}: ΔE = {diff_meV:.3f} meV/atom")
```

### Workflow 2: Relaxation followed by static calculation

The standard two-step workflow for accurate energies: relax the geometry with looser convergence, then compute a static single-point with tight settings on the relaxed structure.

```python
from pymatgen.core import Structure
from pymatgen.io.vasp.sets import MPRelaxSet, MPStaticSet
from pathlib import Path

structure = Structure.from_file("POSCAR")

# Step 1: Ionic + cell relaxation
relax_dir = Path("01_relax")
relax_dir.mkdir(exist_ok=True)

relax_vis = MPRelaxSet(structure, user_incar_settings={
    "ENCUT": 520,         # Example: converged value for this material class
    "EDIFF": 1e-5,        # Electronic convergence; tighten to 1e-6 for forces
    "EDIFFG": -0.02,      # Force convergence criterion in eV/Å
    "NSW": 200,           # Max ionic steps
    "IBRION": 2,          # Conjugate gradient
    "ISIF": 3,            # Relax all: ions + cell shape + cell volume
    "ISMEAR": 0,
    "SIGMA": 0.05,
    "ALGO": "Normal",
    "LWAVE": True,        # Keep WAVECAR for next step restart
    "LCHARG": True,
    "PREC": "Accurate",
    "NELM": 100,
    "NELMIN": 4,
    # For magnetic systems, set ISPIN=2 and MAGMOM here
})
relax_vis.write_input(str(relax_dir))

# Step 2: Static single-point on relaxed structure
# Run this after Step 1 completes successfully.
# Copy WAVECAR from Step 1 to speed up SCF convergence.

relaxed_structure = Structure.from_file("01_relax/CONTCAR")
static_dir = Path("02_static")
static_dir.mkdir(exist_ok=True)

static_vis = MPStaticSet(
    relaxed_structure,
    prev_calc_dir=str(relax_dir),  # Picks up CHGCAR/WAVECAR
    user_incar_settings={
        "ENCUT": 520,
        "EDIFF": 1e-6,
        "NSW": 0,
        "IBRION": -1,
        "ISMEAR": -5,     # Tetrahedron for accurate final energy (insulators/semiconductors)
        # Use ISMEAR=0, SIGMA=0.10 for metals; ISMEAR=-5 is incorrect for metals
        "LWAVE": False,
        "LCHARG": True,
        "PREC": "Accurate",
        "NELM": 200,
        "ALGO": "Normal",
    },
)
static_vis.write_input(str(static_dir))
```

**Post-run checks (always perform):**

```python
from pymatgen.io.vasp.outputs import Vasprun, Outcar

# Check electronic convergence
v = Vasprun("02_static/vasprun.xml")
print("Converged:", v.converged)
print("Converged electronic:", v.converged_electronic)
print("Converged ionic:", v.converged_ionic)

outcar = Outcar("02_static/OUTCAR")
# Check that no warnings appear in OUTCAR:
# "Sub-Space-Matrix is not hermitian" -> ALGO issue
# "WARNING: Sub-Space-Matrix is not hermitian in DAV" -> increase NELMIN
# "NELM was reached" -> SCF did not converge, result is INVALID
```

### Workflow 3: Functional selection and DFT+U

Choosing the right functional for the system and property.

```python
# Functional guide (not exhaustive):
#
# PBE (GGA):
#   INCAR: default (no METAGGA tag needed)
#   Use for: structural properties, force fields, general screening
#   Avoid for: band gaps, magnetic moments in TM oxides, adsorption energies
#
# PBEsol:
#   INCAR: GGA = PS
#   Use for: surfaces, adsorbates, systems where lattice constants matter more than cohesive energies
#   [EXPERT REVIEW NEEDED: PBEsol vs PBE performance crossover for bulk systems]
#
# SCAN (meta-GGA):
#   INCAR: METAGGA = SCAN; ALGO = All; LASPH = True; NELM = 200
#   Use for: better lattice constants + formation energies than PBE without HSE cost
#   Avoid for: SCF convergence problems; SCAN is more sensitive to starting density
#   Note: Always use ALGO=All with SCAN; ALGO=Fast frequently diverges
#
# HSE06 (screened hybrid):
#   INCAR: LHFCALC = True; HFSCREEN = 0.2; AEXX = 0.25; ALGO = All
#   Use for: band gaps, optical properties, magnetic exchange
#   Cost: ~10-50x more expensive than PBE for the same system
#   Note: Cannot use ISMEAR=-5 with HSE06; use ISMEAR=0
#
# DFT+U (Dudarev scheme):
#   INCAR: LDAU = True; LDATYPE = 2; LDAUL = [2]; LDAUU = [U]; LDAUJ = [0]
#   Typical U values (in eV): Fe2O3 ~4-5, NiO ~6, CoO ~3-5, MnO ~3-4
#   [EXPERT REVIEW NEEDED: U values are system-dependent; always validate against
#    experimental formation energies or higher-level theory]

# DFT+U example for Fe3O4 (magnetite)
incar_dft_u = {
    "ENCUT": 520,
    "EDIFF": 1e-5,
    "EDIFFG": -0.02,
    "NSW": 200,
    "IBRION": 2,
    "ISIF": 3,
    "ISMEAR": 0,
    "SIGMA": 0.05,
    "ALGO": "Normal",
    "ISPIN": 2,
    "MAGMOM": "56*5 24*-5",   # Example: ferrimagnetic initialization; adjust for your structure
    "LDAU": True,
    "LDATYPE": 2,              # Dudarev scheme (only U-J matters)
    "LDAUL": "2 -1",           # l quantum number: 2=d for Fe, -1=none for O
    "LDAUU": "4.0 0.0",        # U value in eV
    "LDAUJ": "0.0 0.0",        # J=0 for Dudarev
    "LMAXMIX": 4,              # Required for d orbitals; use 6 for f
    "NELM": 200,
    "NELMIN": 6,
}
```

### Workflow 4: Generating DFT training data for ML potentials

When VASP is used as the oracle for ML interatomic potential training.

```python
from pymatgen.core import Structure
from pymatgen.io.vasp.sets import MPStaticSet
from pathlib import Path
import json

# For ML potential training, you need:
# - Energies (total energy, eV)
# - Forces (eV/Å on every atom)
# - Stress tensors (kBar or eV/Å³)
# These must be computed at the same level of theory and with the same POTCAR set.

def write_mlp_incar(structure: Structure, calc_dir: Path) -> None:
    """Write INCAR for ML potential training data generation."""
    vis = MPStaticSet(structure, user_incar_settings={
        "ENCUT": 520,
        "EDIFF": 1e-6,          # Tighter than usual — forces must be accurate
        "NSW": 0,
        "IBRION": -1,
        "ISMEAR": 0,
        "SIGMA": 0.10,
        "ALGO": "Normal",
        "LWAVE": False,
        "LCHARG": False,
        "PREC": "Accurate",
        "NELM": 200,
        "ISIF": 2,              # Compute stress tensor (ISIF >= 2)
        "ISTART": 0,
        "ICHARG": 2,
    })
    vis.write_input(str(calc_dir))


def extract_training_data(calc_dir: Path) -> dict:
    """Extract energy, forces, stress from a completed VASP run."""
    from pymatgen.io.vasp.outputs import Vasprun
    v = Vasprun(str(calc_dir / "vasprun.xml"), parse_potcar_file=False)
    if not v.converged_electronic:
        raise RuntimeError(f"SCF not converged in {calc_dir}")
    structure = v.final_structure
    return {
        "energy": v.final_energy,
        "forces": v.ionic_steps[-1]["forces"],
        "stress": v.ionic_steps[-1]["stress"],   # kBar; convert to eV/Å³ if needed
        "natoms": len(structure),
        "lattice": structure.lattice.matrix.tolist(),
        "species": [str(s) for s in structure.species],
        "positions": structure.cart_coords.tolist(),
    }
```

## Common Failure Modes

| Failure | Symptom | Likely Cause | Diagnostic | Fix |
|---------|---------|--------------|------------|-----|
| SCF not converged | `NELM was reached` in OUTCAR; `converged_electronic=False` in vasprun.xml | Bad starting charge density, ill-conditioned Hamiltonian, wrong ALGO | Check OUTCAR for oscillating free energy; plot F vs iteration | Switch to `ALGO=All`; add `AMIX=0.2 BMIX=0.0001`; use `ISTART=0 ICHARG=2` for fresh start |
| Wrong magnetic ground state | Energy higher than expected; magnetic moment collapses or wrong sign | Poor MAGMOM initialization or missing ISPIN=2 | Check `magnetization` in OUTCAR; compare total energy across different initializations | Test multiple MAGMOM initializations; use collinear AFM, FM, and ferrimagnetic starting points |
| Pulay stress artifact | Relaxation converges but volume is wrong; residual stress after relaxation | ENCUT too close to POTCAR ENMAX causes basis-set incomplete relaxation | Check final stress tensor in OUTCAR; compare lattice parameters to experiment | Increase ENCUT by 30% above POTCAR ENMAX; re-relax |
| POSCAR/POTCAR element mismatch | VASP crash with `POTCAR` error or wrong atom count | Elements in POSCAR must exactly match POTCAR order | Check first line of POSCAR vs POTCAR header | Ensure element order is identical in POSCAR and POTCAR |
| Incorrect ISMEAR for metal | Wrong total energy; free energy differs significantly from E0 | Using ISMEAR=-5 (tetrahedron) for a metal with a Fermi surface | Check `ISMEAR` in INCAR; check `NKPTS` | Use `ISMEAR=1` (Methfessel-Paxton) with `SIGMA=0.1–0.2` for metals |
| Dipole divergence in slab calculation | Electrostatic potential diverges; oscillating energy; asymmetric slab | Polar slab or charged slab without dipole correction | Check LOCPOT with vaspkit; plot planar average of electrostatic potential | Add `LDIPOL=True; IDIPOL=3; DIPOL=0.5 0.5 0.5`; increase vacuum to >15 Å |
| Spin contamination in DFT+U | Magnetic moments converge to a mixed or unphysical state | Multiple local minima on the DFT+U energy landscape | Check `Mag` column in OUTCAR; test multiple MAGMOM starting points | Try different MAGMOM initializations; use `LAMBDA` smearing; run with U=0 first then ramp U |
| KPOINTS and symmetry mismatch | VASP warning about broken symmetry; calculation slow | Non-Gamma-centered mesh on hexagonal systems | Check `KPOINTS` file; inspect symmetry in vasprun.xml | Use Gamma-centered mesh (`KGAMMA=.TRUE.` or `G` in KPOINTS) for hexagonal, trigonal, or FCC systems |

## Best Practices

- **Always test ENCUT and k-point convergence** for each new material class. Never assume that a parameter set converged for one material transfers to another.
- **Check OUTCAR after every run** for `WARNING` lines before trusting results. Warnings about non-hermitian matrices, NELM reached, or ZBRENT failures indicate invalid output.
- **Use ISIF=3 only when the unit cell shape and volume should relax.** Use ISIF=2 for fixed-cell relaxation and ISIF=0 for single-point.
- **Do not mix POTCAR vintages.** Comparing energies calculated with different POTCAR releases (e.g., 52 vs. 54) introduces systematic errors. Record POTCAR metadata in your workflow.
- **Initialize magnetic moments explicitly** for all systems containing Fe, Co, Ni, Mn, Cr, or rare-earth elements. Letting VASP default to zero spin will often find a wrong non-magnetic solution.
- **Use PREC=Accurate** for production runs. PREC=Normal reduces FFT grid size and can introduce errors of a few meV/atom.
- **Record the full INCAR, KPOINTS, and POTCAR TITEL lines** for reproducibility. These are the minimum necessary to reproduce a VASP calculation.
- **For ML potential training datasets,** compute all structures at identical settings (ENCUT, POTCAR, ISMEAR, SIGMA) so energies are on a consistent reference surface. Mixing settings introduces unphysical noise.
- **Restart SCF from WAVECAR** (ISTART=1) when running sequential calculations on structurally similar configurations — this saves 50–80% of SCF iterations.
- **Validate against known experimental properties** (lattice parameter, bulk modulus, band gap) before using a new functional/POTCAR combination for production.

## Integration with Other Skills

- **[dft-convergence](../dft-convergence/SKILL.md):** Prerequisite skill. ENCUT and k-point convergence testing should be performed using that skill's systematic protocol before this skill's production workflows.
- **[mace-training](../../ml-interatomic-potentials/mace-training/SKILL.md):** VASP is the DFT oracle that generates training data for MACE and other MLPs. The `Workflow 4` above produces the data that the MACE training skill consumes. Consistent ENCUT, POTCAR, and ISMEAR settings across the dataset are mandatory.
- **[active-learning-mlp](../../ml-interatomic-potentials/active-learning-mlp/SKILL.md):** Active learning loops query VASP for single-point energies and forces on structures selected by an uncertainty criterion. VASP convergence settings must be tight enough to avoid corrupting the training signal.
- **[materials-featurization](../../scientific-data/materials-featurization/SKILL.md):** DFT-computed properties (band gap, formation energy, elastic moduli) are target variables for materials ML models. The accuracy of those models is bounded by the accuracy of the DFT reference.
- **[reproducibility-workflow](../../scientific-data/reproducibility-workflow/SKILL.md):** VASP calculations require versioned POTCAR files, logged INCAR settings, and tracked vasprun.xml outputs to be reproducible. Use that skill's framework to archive calculation provenance.

## Key References

1. Kresse, G. & Furthmüller, J. (1996). *Efficient iterative schemes for ab initio total-energy calculations using a plane-wave basis set.* Physical Review B, 54(16), 11169. https://doi.org/10.1103/PhysRevB.54.11169
2. Kresse, G. & Joubert, D. (1999). *From ultrasoft pseudopotentials to the projector augmented-wave method.* Physical Review B, 59(3), 1758. https://doi.org/10.1103/PhysRevB.59.1758
3. Perdew, J. P., Burke, K., & Ernzerhof, M. (1996). *Generalized gradient approximation made simple.* Physical Review Letters, 77(18), 3865. https://doi.org/10.1103/PhysRevLett.77.3865
4. Dudarev, S. L., et al. (1998). *Electron-energy-loss spectra and the structural stability of nickel oxide: An LSDA+U study.* Physical Review B, 57(3), 1505. https://doi.org/10.1103/PhysRevB.57.1505
5. Ong, S. P., et al. (2013). *Python Materials Genomics (pymatgen): A robust, open-source python library for materials analysis.* Computational Materials Science, 68, 314–319. https://doi.org/10.1016/j.commatsci.2012.10.028
6. VASP Wiki — INCAR Tags. https://www.vasp.at/wiki/index.php/Category:INCAR_tag (accessed 2024)
