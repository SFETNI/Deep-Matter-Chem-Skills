---
name: dft-phonons
domain: electronic-structure
maturity: draft
tools:
  - VASP
  - Quantum ESPRESSO
  - Phonopy
  - ASE
  - pymatgen
  - numpy
  - matplotlib
tested_versions:
  VASP: "6.4.2"
  Quantum ESPRESSO: "7.3"
  Phonopy: "2.26.6"
  ASE: "3.23.0"
  pymatgen: "2024.11.13"
requires_gpu: false
estimated_runtime: hours-to-days
---

# DFT Phonon Calculations

## Description

This skill covers lattice dynamics from first principles: computing phonon dispersion curves, phonon density of states (PDOS), and thermal properties using the finite-displacement method (Phonopy + VASP/QE) and density functional perturbation theory (DFPT via Quantum ESPRESSO `ph.x` or VASP IBRION=7/8). It covers the full workflow from supercell construction and displaced-supercell DFT calculations through force constant extraction, acoustic sum rule enforcement, non-analytical corrections for polar materials, and phonon band structure post-processing. Invoke this skill when computing harmonic phonon properties, diagnosing dynamical instabilities, validating ML interatomic potentials against DFT phonons, or generating phonon reference data for materials databases.

## Domain Context

Phonons are the quantum of lattice vibration. They are essential for predicting thermodynamic stability, heat capacity, thermal conductivity, electron-phonon coupling, and phase transitions. Two formally equivalent but practically distinct methods are used in DFT practice:

**Finite-displacement (supercell) method:** Construct symmetry-inequivalent displaced supercells → run DFT single-point calculations → extract forces → build real-space force constants → Fourier transform to get the dynamical matrix D(q) at arbitrary q-points → diagonalize to get phonon frequencies ω(q). Implemented by Phonopy, which is DFT-code-agnostic.

**DFPT (Density Functional Perturbation Theory):** Compute the response of the charge density to an atomic displacement as a perturbation at a given wavevector q directly, without supercells. Gives the dynamical matrix at specified q-points directly from a single calculation. Native to Quantum ESPRESSO (`ph.x`) and also available in VASP (IBRION=7/8). More expensive per q-point than finite displacement for a single structure but avoids the supercell approximation entirely.

**Dynamical matrix:** D_αβ(q) = (1/√(m_α m_β)) Σ_R Φ_αβ(0,R) e^{iq·R}, where Φ_αβ are the interatomic force constants (second derivatives of the total energy). Eigenvalues of D(q) give ω²(q,s) for each phonon branch s. Negative eigenvalues (imaginary frequencies) signal dynamical instability at that q-point.

**Long-range interactions in polar materials:** In ionic materials, the dynamical matrix has a long-range Coulomb contribution that causes a discontinuity at Γ between the longitudinal optical (LO) and transverse optical (TO) modes. This LO-TO splitting must be handled via the non-analytical correction (NAC), which requires the Born effective charge tensors Z* and the dielectric constant tensor ε∞. Both are computed via DFPT. Omitting NAC in polar materials produces qualitatively wrong phonon dispersions near Γ.

**Acoustic sum rule (ASR):** Translational invariance of the total energy requires that acoustic modes approach ω→0 as q→0. Numerically, ASR violations (acoustic modes at Γ with |ω| > ~1 cm⁻¹) indicate: insufficient force convergence (EDIFF too loose), real-space projection errors (LREAL=True in VASP), or broken crystal symmetry in the supercell setup. Phonopy and QE both offer ASR correction flags that impose the rule algebraically post-hoc; these should be used only when violations are small (< 5 cm⁻¹) and treated as a convergence diagnostic when violations are large.

**Harmonic approximation:** All methods here assume the potential energy surface is quadratic in atomic displacements. This breaks down at high temperature (strong anharmonicity), near phase transitions, and for systems with large zero-point motion (hydrogen-rich materials, light atoms). Thermal properties derived from harmonic phonons (free energy, entropy, Cv) are only reliable at T ≪ Tmelt for stiff materials, or require anharmonic corrections (quasi-harmonic approximation, SSCHA, TDEP) beyond this skill's scope. [EXPERT REVIEW NEEDED: temperature thresholds for harmonic approximation failure vary by material class]

## When to Use This Skill

- Checking thermodynamic stability: imaginary acoustic modes at zone boundary = ground state is not the true minimum.
- Computing zero-point energy corrections to total energies.
- Predicting heat capacity Cv(T) and vibrational entropy for phase diagram construction.
- Validating that a relaxed structure is a true energy minimum (no imaginary modes at Γ or high-symmetry points).
- Generating phonon fingerprints for validating ML interatomic potentials against DFT.
- Computing Born effective charges and dielectric tensors for polar materials.
- Studying soft modes and structural phase transitions (e.g., perovskite tilting instabilities).
- Fitting interatomic force constants for thermal transport (κ) codes (ShengBTE, phono3py).
- Cross-referencing phonon dispersions against inelastic neutron scattering (INS) or Raman/IR data.

## When Not to Use This Skill

- **Strongly anharmonic systems at high T** (e.g., BCC metals near melting, plastic crystals, H-rich hydrides): harmonic phonons are qualitatively wrong; use TDEP or SSCHA instead.
- **Phonon linewidths or lifetime broadening** (e.g., for thermal conductivity beyond RTA): requires third-order force constants; use phono3py skill.
- **Electron-phonon coupling** for superconductivity or resistivity: requires specialized DFPT (EPW code or QE `ph.x` + `elph`).
- **Very large supercells (> 300 atoms) with Phonopy**: finite-displacement DFT cost scales as N_displacements × cost_per_supercell; DFPT at a few q-points may be cheaper.
- **Amorphous or disordered materials**: phonon band structure is ill-defined; use phonon DOS from MD instead.
- **Molecular systems in gas phase**: use vibrational normal modes from a molecular Hessian, not periodic phonons.

## Core Concepts

### Force Constants and the Supercell Approach

The interatomic force constant (IFC) Φ_αβ(i,j) is the second derivative of the total energy with respect to displacements of atoms i and j in Cartesian directions α,β. In the finite-displacement method, it is approximated by:

Φ_αβ(i,j) ≈ -∂F_α(i)/∂u_β(j) ≈ -(F_α[+u] - F_α[-u]) / (2u)

where F_α[±u] is the force on atom i when atom j is displaced by ±u in direction β. Central differences (±displacement) roughly doubles the DFT cost but cancels odd-order anharmonic terms, giving more accurate force constants.

Phonopy generates the minimal set of symmetry-inequivalent displaced supercells automatically, runs DFT on each, collects forces, and constructs the full IFC matrix. The number of displaced supercells is typically 1–6 for high-symmetry crystals and 10–50+ for low-symmetry structures.

### Supercell Size and Force Constant Cutoff

IFCs decay with interatomic distance, but the decay rate depends on the bonding character. Metals have slowly decaying IFCs (Friedel oscillations). Covalently bonded materials (Si, diamond) decay faster. Ionic materials have long-range Coulomb tails that must be handled separately via NAC.

The supercell must be large enough that the periodic image of the displaced atom is beyond the IFC cutoff. Rule of thumb: supercell dimensions ≥ 15 Å in each direction for insulators, ≥ 20 Å for metals. Always verify by comparing phonon dispersions from different supercell sizes. A practical test: if doubling the supercell changes any phonon frequency by > 1 cm⁻¹, the smaller supercell is not converged. [EXPERT REVIEW NEEDED: the 1 cm⁻¹ threshold is material-class dependent; soft phonon modes near instabilities require tighter convergence]

### Displacement Amplitude

The displacement amplitude u must satisfy two competing requirements:
1. **Large enough to exceed DFT numerical noise:** Forces must be computed accurately. With EDIFF=1e-8 in VASP, the force noise floor is ~1e-5 eV/Å. To get ~1% accuracy in the force, u ≥ 0.01 Å.
2. **Small enough to remain in the harmonic regime:** Anharmonic contributions to the force scale as u². For most crystalline solids at equilibrium, u = 0.01–0.03 Å is safe.

Soft materials (zeolites, MOFs, soft phonon branches) may need u = 0.01 Å to stay harmonic. Hard materials (diamond, SiC) tolerate u = 0.03–0.05 Å. The Phonopy default (0.01 Å) is conservative and appropriate for most cases.

[EXPERT REVIEW NEEDED: for materials with very soft phonon branches near instabilities, even 0.01 Å may not stay harmonic; amplitude convergence testing is recommended]

### Acoustic Sum Rule (ASR)

At Γ (q=0), the three acoustic modes must have ω=0. Violation magnitude:
- < 1 cm⁻¹: acceptable numerical noise; apply ASR correction algebraically
- 1–5 cm⁻¹: borderline; check LREAL setting, k-mesh, and EDIFF
- > 5 cm⁻¹: problematic; indicates under-converged forces or broken supercell symmetry; must diagnose before trusting optical modes

Phonopy applies ASR via `phonopy --fc-symmetry` (symmetrizes force constants) or through the FORCE_CONSTANTS file post-processing. QE applies ASR via `asr='crystal'` in the `matdyn.x` input.

### Non-Analytical Correction (NAC) for Polar Materials

In polar materials (most oxides, halides, III-V semiconductors), the dynamical matrix at q→0 has a singular non-analytical term:

D_NAC(q→0) ∝ (q · Z*_α)(q · Z*_β) / (q · ε∞ · q)

This term is direction-dependent: the LO-TO splitting depends on the direction of q as q→0. The magnitude of LO-TO splitting is proportional to (Z*)² / ε∞. For BaTiO3, LO-TO splitting at Γ can exceed 100 cm⁻¹.

NAC is absent if:
- Z* and ε∞ are not provided (will get wrong phonons near Γ)
- The material is centrosymmetric but piezoelectric (NAC applies even without a permanent dipole)
- The material is a metal (no NAC; dielectric screening is complete)

For Phonopy + VASP: run a separate DFPT calculation with LEPSILON=True to get OUTCAR with Born charges and dielectric tensor, then parse with `phonopy-vasp-born`.

For QE: `ph.x` computes both phonons and Born charges/dielectric tensor in one run when `epsil=.true.` is set.

### Phonon Band Structure and DOS

**Band structure:** Evaluate phonon frequencies along a high-symmetry path in the Brillouin zone. The path is material-specific and should follow the standard reciprocal space conventions (Setyawan-Curtarolo). Phonopy `band.conf` or pymatgen `HighSymmKpath` generates the path.

**Phonon DOS (PDOS):** Integrate phonon frequencies over the full BZ. Resolution depends on the q-mesh density. For smooth PDOS, a 20×20×20 q-mesh is often needed; convergence must be checked. Projected PDOS (element-resolved) identifies which atoms contribute to which frequency regions and is essential for understanding thermal properties.

**Thermal properties:** From the harmonic PDOS, compute:
- Helmholtz free energy F(T) = E_0 + kT ∫g(ω)ln(2sinh(ℏω/2kT))dω
- Entropy S(T), heat capacity Cv(T)
Valid in the harmonic approximation (see caveats above). Phonopy `thermal_properties.yaml` gives these directly.

## Key Workflows

### Workflow 1: VASP + Phonopy Finite-Displacement Workflow

This is the most common workflow for VASP users. Phonopy generates displaced supercells, VASP computes forces on each, Phonopy assembles the force constants.

```python
# phonopy_vasp_workflow.py
# Prerequisites: phonopy, pymatgen, numpy, matplotlib
# Structure: well-relaxed POSCAR (ISIF=3 with EDIFFG=-1e-3 or tighter)

import subprocess
import shutil
from pathlib import Path
import numpy as np
from phonopy import Phonopy
from phonopy.interface.vasp import read_vasp
from phonopy.file_IO import parse_FORCE_SETS
import yaml

WORKDIR = Path("phonon_calc")
POSCAR = Path("POSCAR")          # fully relaxed unit cell
POTCAR = Path("POTCAR")
KPOINTS_SC = Path("KPOINTS_sc")  # k-mesh for supercell (typically 2x2x2 or denser)

# --- Step 0: Phonon-appropriate INCAR ---
INCAR_PHONON = {
    "ENCUT": 550,           # at least 1.3x ENMAX from POTCAR; higher is safer for phonons
    "EDIFF": 1e-8,          # tight SCF; critical for accurate forces
    "PREC": "Accurate",     # disables VASP precision compromises
    "NSW": 0,               # single-point (no ionic relaxation on displaced cell)
    "IBRION": -1,           # single-point mode
    "ISMEAR": 0,            # Gaussian smearing; use -5 (tetrahedron) only for insulators
    "SIGMA": 0.01,          # small smearing; large sigma corrupts forces in metals
    "ALGO": "Normal",       # Davidson; "Fast" can cause SCF issues in large supercells
    "LREAL": False,         # CRITICAL: real-space projection introduces force errors ~1e-4 eV/Å
    "ADDGRID": True,        # finer augmentation grid; reduces charge density noise
    "LWAVE": False,
    "LCHARG": False,        # no need to write charge density for force calculations
    "NPAR": 4,              # adjust to available cores; NPAR=sqrt(num_cores) is a starting point
}

def write_incar(path, settings):
    with open(path / "INCAR", "w") as f:
        for k, v in settings.items():
            if isinstance(v, bool):
                f.write(f"  {k} = {'T' if v else 'F'}\n")
            elif isinstance(v, float) and v < 0.001:
                f.write(f"  {k} = {v:.1e}\n")
            else:
                f.write(f"  {k} = {v}\n")

# --- Step 1: Create supercell and generate displacements ---
phonon = Phonopy(
    read_vasp(str(POSCAR)),
    supercell_matrix=[[3, 0, 0], [0, 3, 0], [0, 0, 3]],  # 3x3x3 supercell; adjust to get >15 Å
    primitive_matrix="auto",    # Phonopy determines primitive cell automatically
)
phonon.generate_displacements(distance=0.01)   # 0.01 Å displacement; conservative default

print(f"Supercell: {phonon.supercell.get_number_of_atoms()} atoms")
print(f"Number of displaced supercells: {len(phonon.displacements)}")

# Write displaced supercells as POSCARs
from phonopy.interface.vasp import write_vasp
supercells = phonon.supercells_with_displacements
for i, sc in enumerate(supercells):
    sc_dir = WORKDIR / f"disp-{i+1:04d}"
    sc_dir.mkdir(parents=True, exist_ok=True)
    write_vasp(str(sc_dir / "POSCAR"), sc)
    write_incar(sc_dir, INCAR_PHONON)
    shutil.copy(POTCAR, sc_dir / "POTCAR")
    shutil.copy(KPOINTS_SC, sc_dir / "KPOINTS")

# Save phonopy.yaml for reconstruction after DFT runs
phonon.save("phonopy_disp.yaml")

print(f"Displaced supercells written to {WORKDIR}/disp-XXXX/")
print("Submit all disp-XXXX/ directories as VASP single-point calculations.")
```

```bash
# --- Step 2: Submit VASP jobs (HPC example with SLURM) ---
for dir in phonon_calc/disp-*/; do
    cd "$dir"
    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=phonon_disp
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=32
#SBATCH --time=04:00:00
module load vasp/6.4.2
mpirun vasp_std > vasp.out 2>&1
EOF
    cd -
done
```

```python
# --- Step 3: Collect forces and build force constants ---
from phonopy import load as phonopy_load
from phonopy.interface.vasp import get_vasp_structure_and_forces

phonon = phonopy_load("phonopy_disp.yaml")
disp_dirs = sorted(WORKDIR.glob("disp-????"))

# Verify all calculations completed
failed = []
forces_list = []
for d in disp_dirs:
    outcar = d / "OUTCAR"
    if not outcar.exists():
        failed.append(d)
        continue
    # Check VASP finished successfully
    with open(outcar) as f:
        content = f.read()
    if "reached required accuracy" not in content and "General timing" not in content:
        failed.append(d)
        continue
    # Parse forces from OUTCAR
    _, forces = get_vasp_structure_and_forces(str(outcar))
    forces_list.append(forces)

if failed:
    raise RuntimeError(f"VASP incomplete for: {[str(d) for d in failed]}")

phonon.forces = forces_list
phonon.produce_force_constants()

# Check acoustic sum rule violation
phonon.symmetrize_force_constants()  # applies crystal symmetry
freq_at_gamma = phonon.get_frequencies([0, 0, 0])
acoustic_freqs = sorted(np.abs(freq_at_gamma))[:3]
print(f"Acoustic modes at Γ (cm⁻¹): {[f'{f:.3f}' for f in acoustic_freqs]}")
if max(acoustic_freqs) > 5.0:
    print("WARNING: Large ASR violation. Check LREAL=False, EDIFF=1e-8, supercell symmetry.")
elif max(acoustic_freqs) > 1.0:
    print("WARNING: Moderate ASR violation. Consider tighter convergence.")

phonon.save("phonopy.yaml")  # save force constants for reuse
```

```python
# --- Step 4: Phonon band structure and DOS ---
import matplotlib.pyplot as plt
from phonopy.phonon.band_structure import get_band_qpoints_and_path_connections

# High-symmetry path (Setyawan-Curtarolo conventions via seekpath)
# For FCC: Γ-X-U|K-Γ-L-W-X
# For BCC: Γ-H-N-Γ-P-H|P-N
# Phonopy can determine the path automatically from spacegroup:
phonon.auto_band_structure(
    npoints=101,
    with_eigenvectors=True,
    is_band_connection=True,   # connect bands by eigenvector continuity
    plot=False,
)
phonon.plot_band_structure().savefig("phonon_bands.png", dpi=150, bbox_inches="tight")

# Phonon DOS
phonon.auto_total_dos(
    mesh=50,            # 50x50x50 q-mesh; increase for publication quality
    freq_pitch=0.1,     # frequency resolution in cm⁻¹
    use_tetrahedron_method=True,
    write_dos=True,     # writes total_dos.dat
)
phonon.plot_total_dos().savefig("phonon_dos.png", dpi=150, bbox_inches="tight")

# Thermal properties (harmonic approximation — see caveats in Domain Context)
phonon.run_thermal_properties(
    t_step=10,
    t_max=1200,
    t_min=0,
)
phonon.write_yaml_thermal_properties()  # thermal_properties.yaml

# Parse and plot thermal properties
with open("thermal_properties.yaml") as f:
    tp = yaml.safe_load(f)
temps = [p["temperature"] for p in tp["thermal_properties"]]
free_energy = [p["free_energy"] for p in tp["thermal_properties"]]   # kJ/mol
entropy = [p["entropy"] for p in tp["thermal_properties"]]           # J/mol/K
heat_cap = [p["heat_capacity"] for p in tp["thermal_properties"]]    # J/mol/K

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].plot(temps, free_energy); axes[0].set_xlabel("T (K)"); axes[0].set_ylabel("F (kJ/mol)")
axes[1].plot(temps, entropy); axes[1].set_xlabel("T (K)"); axes[1].set_ylabel("S (J/mol/K)")
axes[2].plot(temps, heat_cap); axes[2].set_xlabel("T (K)"); axes[2].set_ylabel("Cv (J/mol/K)")
axes[2].axhline(3 * len(phonon.primitive.masses) * 8.314, ls="--", c="gray", label="Dulong-Petit")
plt.tight_layout()
plt.savefig("thermal_properties.png", dpi=150, bbox_inches="tight")
```

### Workflow 2: Born Effective Charges and NAC for Polar Materials (VASP)

For polar (non-centrosymmetric or ionic) materials, always compute Born effective charges and include NAC in phonon post-processing. Omitting NAC produces wrong LO-TO splitting near Γ.

```python
# born_charges_vasp.py — DFPT run for Born charges and dielectric tensor
# Run this SEPARATELY from the finite-displacement calculations (it's a unit cell calculation)

INCAR_DFPT = {
    "ENCUT": 550,
    "EDIFF": 1e-8,
    "PREC": "Accurate",
    "NSW": 0,
    "IBRION": -1,
    "ISMEAR": 0,
    "SIGMA": 0.01,
    "ALGO": "Normal",
    "LREAL": False,
    "LEPSILON": True,       # compute Born effective charges and dielectric tensor via DFPT
    "LPEAD": False,         # use DFPT (not finite-field); recommended for consistency
    "NPAR": 1,              # LEPSILON REQUIRES NPAR=1 (or remove NPAR); will fail otherwise
}

# After VASP finishes, parse Born charges from OUTCAR:
from pymatgen.io.vasp import Outcar

outcar = Outcar("OUTCAR_dfpt")
born_charges = outcar.born          # list of 3x3 tensors, one per atom
dielectric = outcar.dielectric_tensor  # 3x3 tensor (electronic contribution)

print(f"Born effective charges shape: {len(born_charges)} atoms x 3x3")
print(f"Dielectric tensor:\n{np.array(dielectric)}")

# Write BORN file for Phonopy
# Phonopy BORN file format: first line = dielectric, then one line per atom = Born charges
def write_phonopy_born(born_charges, dielectric, primitive, path="BORN"):
    with open(path, "w") as f:
        f.write("# BORN charges and dielectric tensor\n")
        eps = np.array(dielectric)
        f.write(f"  {eps[0,0]:.8f} {eps[0,1]:.8f} {eps[0,2]:.8f} "
                f"{eps[1,0]:.8f} {eps[1,1]:.8f} {eps[1,2]:.8f} "
                f"{eps[2,0]:.8f} {eps[2,1]:.8f} {eps[2,2]:.8f}\n")
        for z in born_charges:
            z = np.array(z)
            f.write(f"  {z[0,0]:.8f} {z[0,1]:.8f} {z[0,2]:.8f} "
                    f"{z[1,0]:.8f} {z[1,1]:.8f} {z[1,2]:.8f} "
                    f"{z[2,0]:.8f} {z[2,1]:.8f} {z[2,2]:.8f}\n")
    print(f"Written {path}")

# Use primitive atoms only (Phonopy requires primitive-cell Born charges)
from phonopy.interface.vasp import read_vasp
prim = read_vasp("POSCAR")  # must be the primitive cell used in Phonopy
write_phonopy_born(born_charges[:prim.get_number_of_atoms()], dielectric, prim)

# Load phonon with NAC
from phonopy import load as phonopy_load
phonon = phonopy_load("phonopy.yaml", born_filename="BORN")

# Verify NAC is active
print(f"NAC loaded: {phonon.nac_params is not None}")

# Check LO-TO splitting at Γ
# Approach q from Γ along Γ→X direction
freq_to = phonon.get_frequencies_with_eigenvectors([1e-5, 0, 0])[0]
freq_lo = phonon.get_frequencies_with_eigenvectors([0, 0, 1e-5])[0]
print(f"Frequencies along Γ→X (TO): {[f'{f:.1f}' for f in sorted(freq_to, reverse=True)[:6]]} cm⁻¹")
print(f"Frequencies along Γ→Z (LO): {[f'{f:.1f}' for f in sorted(freq_lo, reverse=True)[:6]]} cm⁻¹")
```

### Workflow 3: DFPT Phonons with Quantum ESPRESSO

QE's `ph.x` computes the dynamical matrix directly via linear response, without supercells. This is the preferred method when Γ-point phonons, Born charges, and dielectric tensor are all needed in a single calculation.

```fortran
! --- scf.in: converged SCF calculation (must be tight for phonons) ---
&CONTROL
  calculation = 'scf'
  outdir      = './tmp'
  prefix      = 'MgO'
  pseudo_dir  = './pseudo'
/
&SYSTEM
  ibrav       = 0
  nat         = 2
  ntyp        = 2
  ecutwfc     = 80          ! Ry; use SSSP efficiency minimum + 20%
  ecutrho     = 640         ! Ry; for USPP: 8x ecutwfc; for NC: 4x ecutwfc
  occupations = 'smearing'
  smearing    = 'mv'        ! Methfessel-Paxton for metals; 'gaussian' for insulators
  degauss     = 0.01        ! Ry; ~0.14 eV; small for accurate forces
/
&ELECTRONS
  conv_thr    = 1.0d-12     ! tight: 100x tighter than default 1e-10; required for DFPT
  mixing_beta = 0.4
  electron_maxstep = 200
/
ATOMIC_SPECIES
  Mg  24.305  Mg.pbesol-spn-kjpaw_psl.1.0.0.UPF
  O   15.999  O.pbesol-n-kjpaw_psl.1.0.0.UPF
ATOMIC_POSITIONS {crystal}
  Mg  0.00  0.00  0.00
  O   0.50  0.50  0.50
K_POINTS {automatic}
  8 8 8 0 0 0                ! Γ-centered; verify k-mesh convergence
CELL_PARAMETERS {angstrom}
  0.000  2.106  2.106
  2.106  0.000  2.106
  2.106  2.106  0.000
```

```fortran
! --- ph.in: DFPT phonon calculation ---
&INPUTPH
  outdir       = './tmp'
  prefix       = 'MgO'
  fildyn       = 'MgO.dyn'  ! output dynamical matrix file prefix
  ldisp        = .true.     ! compute on a q-mesh (for full BZ coverage)
  nq1          = 4          ! q-mesh: 4x4x4 (Phonopy-equivalent to a 4x4x4 supercell)
  nq2          = 4
  nq3          = 4
  epsil        = .true.     ! compute Born effective charges and dielectric tensor
  tr2_ph       = 1.0d-16   ! convergence threshold for DFPT (tight; default 1e-12 may not be enough)
  reduce_io    = .true.
  recover      = .false.
/
```

```bash
# --- Run QE phonon workflow ---
mpirun -np 64 pw.x -npool 4 < scf.in > scf.out
# Check SCF converged: "convergence has been achieved" must appear in scf.out

mpirun -np 64 ph.x -npool 4 < ph.in > ph.out
# This is expensive; wall time scales as N_qpoints x N_modes x SCF_cost
# With ldisp=.true. and nq=4x4x4, expect 4x4x4/nsymm unique q-points

# Collect dynamical matrices and apply ASR
cat > q2r.in << EOF
&INPUT
  fildyn   = 'MgO.dyn'
  zasr     = 'crystal'   ! apply acoustic sum rule ('crystal' for periodic; 'simple' also available)
  flfrc    = 'MgO.fc'    ! output force constants file
/
EOF
mpirun -np 1 q2r.x < q2r.in > q2r.out

# Compute phonon dispersion along high-symmetry path
cat > matdyn_bands.in << EOF
&INPUT
  asr      = 'crystal'
  flfrc    = 'MgO.fc'
  flfrq    = 'MgO_bands.freq'
  q_in_band_form = .true.
  nk1      = 10       ! points per segment
  flvec    = 'MgO_bands.eig'
/
6                          ! number of special points
  0.000  0.000  0.000  20  ! Γ
  0.500  0.000  0.500  20  ! X
  0.500  0.250  0.750  20  ! W
  0.375  0.375  0.750  20  ! K
  0.000  0.000  0.000  20  ! Γ
  0.500  0.500  0.500  1   ! L
EOF
mpirun -np 1 matdyn.x < matdyn_bands.in > matdyn_bands.out

# Compute phonon DOS
cat > matdyn_dos.in << EOF
&INPUT
  asr    = 'crystal'
  flfrc  = 'MgO.fc'
  flfrq  = 'MgO_dos.freq'
  fldos  = 'MgO.dos'
  dos    = .true.
  nk1    = 20  nk2 = 20  nk3 = 20  ! fine mesh for DOS
  ndos   = 500
  DeltaE = 1.0   ! cm⁻¹ per bin
/
EOF
mpirun -np 1 matdyn.x < matdyn_dos.in > matdyn_dos.out
```

```python
# --- Parse and plot QE phonon bands ---
import numpy as np
import matplotlib.pyplot as plt

def parse_qe_freq(freq_file):
    """Parse matdyn.x frequency output file."""
    qpoints, freqs = [], []
    with open(freq_file) as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("q"):
            i += 1
            q = list(map(float, lines[i].split()))
            qpoints.append(q)
            i += 1
            freq_line = []
            while i < len(lines) and not lines[i].strip().startswith("q"):
                freq_line.extend(map(float, lines[i].split()))
                i += 1
            freqs.append(freq_line)
        else:
            i += 1
    return np.array(qpoints), np.array(freqs)

qpts, freqs_cm = parse_qe_freq("MgO_bands.freq")
# freqs in cm⁻¹ (matdyn.x default output)

fig, ax = plt.subplots(figsize=(6, 5))
x = np.arange(len(qpts))
for branch in range(freqs_cm.shape[1]):
    ax.plot(x, freqs_cm[:, branch], "b-", lw=0.8)
ax.axhline(0, color="gray", lw=0.5, ls="--")
ax.set_ylabel("Frequency (cm⁻¹)")
ax.set_xlabel("Wave vector")
# Mark high-symmetry points (set manually from q-points file)
hs_labels = ["Γ", "X", "W", "K", "Γ", "L"]
# hs_positions = [index of Γ, X, W, K, Γ, L in qpts]
plt.savefig("MgO_phonon_bands.png", dpi=150, bbox_inches="tight")
```

### Workflow 4: Supercell Size Convergence Test

Never trust phonon results from a single supercell size. Always check convergence.

```python
# supercell_convergence.py
import numpy as np
from phonopy import Phonopy
from phonopy.interface.vasp import read_vasp

POSCAR = "POSCAR"
FORCES_SETS = {
    "2x2x2": "FORCE_SETS_2x2x2",
    "3x3x3": "FORCE_SETS_3x3x3",
    "4x4x4": "FORCE_SETS_4x4x4",
}
SUPERCELL_MATS = {
    "2x2x2": [[2, 0, 0], [0, 2, 0], [0, 0, 2]],
    "3x3x3": [[3, 0, 0], [0, 3, 0], [0, 0, 3]],
    "4x4x4": [[4, 0, 0], [0, 4, 0], [0, 0, 4]],
}

# Test q-points: Γ and zone boundary
TEST_QPOINTS = {
    "Gamma": [0.0, 0.0, 0.0],
    "X":     [0.5, 0.0, 0.5],   # adjust for specific crystal
    "L":     [0.5, 0.5, 0.5],
}

results = {}
for size, sc_mat in SUPERCELL_MATS.items():
    phonon = Phonopy(read_vasp(POSCAR), supercell_matrix=sc_mat, primitive_matrix="auto")
    from phonopy.file_IO import parse_FORCE_SETS
    with open(FORCES_SETS[size]) as f:
        phonon.forces = parse_FORCE_SETS(phonon.supercell, f)
    phonon.produce_force_constants()
    phonon.symmetrize_force_constants()

    results[size] = {}
    for qname, q in TEST_QPOINTS.items():
        freq = phonon.get_frequencies(q)
        results[size][qname] = freq

print("Supercell convergence (frequencies in cm⁻¹):")
print(f"{'q-point':<10} {'Mode':<6} " + " ".join(f"{s:>12}" for s in SUPERCELL_MATS))
for qname in TEST_QPOINTS:
    n_modes = len(results[list(SUPERCELL_MATS.keys())[0]][qname])
    for mode_idx in range(n_modes):
        freqs = [results[s][qname][mode_idx] for s in SUPERCELL_MATS]
        max_diff = max(freqs) - min(freqs)
        flag = " <-- NOT CONVERGED" if max_diff > 1.0 else ""
        print(f"{qname:<10} {mode_idx:<6} " +
              " ".join(f"{f:>12.2f}" for f in freqs) + flag)
```

### Workflow 5: Imaginary Mode Diagnosis

Imaginary phonon frequencies (negative ω² → reported as negative values by Phonopy and QE) can have physical or numerical origin. Distinguishing them is critical before interpreting results.

```python
# diagnose_imaginary_modes.py
from phonopy import Phonopy
from phonopy.interface.vasp import read_vasp
import numpy as np

def diagnose_imaginary_modes(phonon: Phonopy, q_mesh=(20, 20, 20)):
    """
    Diagnose imaginary modes: find q-points with imaginary frequencies,
    report their q-coordinates, severity, and suggest likely cause.
    """
    phonon.run_mesh(q_mesh, with_eigenvectors=True)
    mesh_dict = phonon.get_mesh_dict()
    frequencies = mesh_dict["frequencies"]   # shape: (N_q, N_modes), cm⁻¹
    qpoints = mesh_dict["qpoints"]           # shape: (N_q, 3)

    imaginary = frequencies < -0.5   # threshold: > 0.5 cm⁻¹ below zero
    n_imaginary_q = np.any(imaginary, axis=1).sum()
    total_q = len(qpoints)

    print(f"\n=== Imaginary Mode Diagnosis ===")
    print(f"Q-mesh: {q_mesh}, total q-points: {total_q}")
    print(f"Q-points with imaginary modes: {n_imaginary_q} / {total_q} ({100*n_imaginary_q/total_q:.1f}%)")

    # Classify by q-point location
    gamma_mask = np.all(np.abs(qpoints) < 0.01, axis=1)
    zone_center_imaginary = np.any(imaginary[gamma_mask], axis=1) if gamma_mask.any() else [False]
    zone_boundary_imaginary = np.any(imaginary[~gamma_mask], axis=1).sum() if (~gamma_mask).any() else 0

    print(f"\nGamma-point imaginary modes: {np.any(zone_center_imaginary)}")
    print(f"Zone-boundary imaginary q-points: {zone_boundary_imaginary}")

    print("\n--- Diagnostic Guide ---")
    if np.any(zone_center_imaginary):
        freqs_at_gamma = frequencies[gamma_mask][0] if gamma_mask.any() else []
        optical_imaginary = [f for f in freqs_at_gamma if f < -0.5]
        acoustic_imaginary = sorted([f for f in freqs_at_gamma if abs(f) < 5.0], key=abs)[:3]
        print(f"  Imaginary optical modes at Γ: {[f'{f:.1f}' for f in optical_imaginary]} cm⁻¹")
        print(f"  Acoustic modes at Γ (ASR check): {[f'{abs(f):.2f}' for f in acoustic_imaginary[:3]]} cm⁻¹")
        if optical_imaginary:
            print("  LIKELY CAUSES (Γ imaginary optical):")
            print("    1. Structure not fully relaxed — check EDIFFG and restart relax with tighter threshold")
            print("    2. Pulay stress — relax at higher ENCUT (try current+20%) or ISIF=3 then recheck")
            print("    3. Polar material missing NAC — add BORN file and recompute")
            print("    4. Genuine ferroelectric/structural instability — check literature for known soft modes")
        if max(abs(f) for f in acoustic_imaginary[:3]) > 5.0:
            print("  LARGE ASR VIOLATION — check LREAL=False, EDIFF=1e-8, supercell symmetry")

    if zone_boundary_imaginary > 0:
        worst_q_idx = np.argmin(frequencies[:, 0])
        worst_q = qpoints[worst_q_idx]
        worst_freq = frequencies[worst_q_idx, 0]
        print(f"\n  Zone-boundary imaginary modes — worst: q={worst_q}, ω={worst_freq:.1f} cm⁻¹")
        print("  LIKELY CAUSES (zone-boundary imaginary):")
        print("    1. Supercell too small — IFCs cut off before they decay; increase supercell")
        print("    2. Genuine structural instability (perovskite octahedral tilt, CDW, etc.)")
        print("    3. Metallic smearing too large (SIGMA > 0.05 eV) — reduce and recheck")
        print("    4. Displacement too large — anharmonic contamination; try u=0.01 Å")
        print("    5. ENCUT too low — Pulay stress in forces; increase to 550+ eV")

    return {
        "n_imaginary_q": n_imaginary_q,
        "zone_center_imaginary": bool(np.any(zone_center_imaginary)),
        "zone_boundary_imaginary": int(zone_boundary_imaginary),
    }

# Usage
phonon = Phonopy(read_vasp("POSCAR"), supercell_matrix=[[3,0,0],[0,3,0],[0,0,3]])
# ... (load forces and produce_force_constants) ...
result = diagnose_imaginary_modes(phonon, q_mesh=(20, 20, 20))
```

### Workflow 6: MLP Phonon Validation Against DFT

Phonon dispersions are among the most sensitive tests of an ML interatomic potential. A potential that perfectly reproduces energies and forces on the training set may still produce qualitatively wrong phonon dispersions if the curvature of the PES is not captured accurately.

```python
# mlp_phonon_validation.py
# Compares phonon dispersions from DFT (VASP+Phonopy) and an MLP (e.g., MACE)
# Assumes: DFT phonon force constants are in phonopy_dft.yaml
#          MLP is loaded via ASE calculator

import numpy as np
import matplotlib.pyplot as plt
from phonopy import load as phonopy_load, Phonopy
from phonopy.interface.vasp import read_vasp
from ase.io import read as ase_read

def compute_mlp_phonons(ase_calc, poscar_path, supercell_matrix, displacement=0.01):
    """Compute phonons using an ASE calculator (MLP or DFT via ASE interface)."""
    from phonopy.interface.calculator import get_default_physical_units
    from phonopy import Phonopy
    from phonopy.interface.vasp import read_vasp
    import phonopy.interface.calculator as calc_interface

    phonon = Phonopy(
        read_vasp(poscar_path),
        supercell_matrix=supercell_matrix,
        primitive_matrix="auto",
    )
    phonon.generate_displacements(distance=displacement)
    supercells = phonon.supercells_with_displacements

    forces_list = []
    for sc in supercells:
        from ase import Atoms
        atoms = Atoms(
            symbols=sc.get_chemical_symbols(),
            positions=sc.get_positions(),
            cell=sc.get_cell(),
            pbc=True,
        )
        atoms.calc = ase_calc
        forces_list.append(atoms.get_forces())

    phonon.forces = forces_list
    phonon.produce_force_constants()
    phonon.symmetrize_force_constants()
    return phonon

def compare_phonons(phonon_dft, phonon_mlp, n_points=101):
    """Compare phonon dispersions and compute frequency errors."""
    # Generate band path (same for both)
    phonon_dft.auto_band_structure(npoints=n_points)
    phonon_mlp.auto_band_structure(npoints=n_points)

    bs_dft = phonon_dft.get_band_structure_dict()
    bs_mlp = phonon_mlp.get_band_structure_dict()

    freqs_dft = np.concatenate(bs_dft["frequencies"], axis=0)  # (N_q, N_modes)
    freqs_mlp = np.concatenate(bs_mlp["frequencies"], axis=0)

    diff = freqs_mlp - freqs_dft
    rmse = np.sqrt(np.mean(diff**2))
    max_err = np.max(np.abs(diff))
    mean_abs_err = np.mean(np.abs(diff))

    print(f"\n=== MLP vs DFT Phonon Comparison ===")
    print(f"RMSE: {rmse:.1f} cm⁻¹")
    print(f"MAE:  {mean_abs_err:.1f} cm⁻¹")
    print(f"Max error: {max_err:.1f} cm⁻¹")

    # Check for qualitative failure: imaginary modes in MLP but not DFT (or vice versa)
    dft_imaginary = (freqs_dft < -0.5).any()
    mlp_imaginary = (freqs_mlp < -0.5).any()
    if mlp_imaginary and not dft_imaginary:
        print("CRITICAL: MLP predicts imaginary modes but DFT does not — MLP fails for this structure")
        print("  Imaginary MLP modes at:", np.where(freqs_mlp < -0.5))
    elif dft_imaginary and not mlp_imaginary:
        print("WARNING: DFT shows instability but MLP does not — MLP may overestimate stability")

    # Frequency thresholds (rule of thumb; [EXPERT REVIEW NEEDED])
    if rmse < 10:
        print("Quality: EXCELLENT — MLP closely matches DFT phonons")
    elif rmse < 30:
        print("Quality: GOOD — minor deviations, may be acceptable for thermal properties")
    elif rmse < 60:
        print("Quality: MODERATE — thermal property errors possible; recheck high-freq modes")
    else:
        print("Quality: POOR — MLP phonons substantially wrong; retrain or add phonon data")

    # Plot comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    distances_dft = np.concatenate(bs_dft["distances"], axis=0)
    distances_mlp = np.concatenate(bs_mlp["distances"], axis=0)

    ax = axes[0]
    for mode in range(freqs_dft.shape[1]):
        ax.plot(distances_dft, freqs_dft[:, mode], "b-", lw=0.8, alpha=0.7, label="DFT" if mode == 0 else "")
        ax.plot(distances_mlp, freqs_mlp[:, mode], "r--", lw=0.8, alpha=0.7, label="MLP" if mode == 0 else "")
    ax.axhline(0, color="gray", lw=0.5, ls=":")
    ax.set_ylabel("Frequency (cm⁻¹)")
    ax.set_xlabel("Wave vector")
    ax.legend(loc="upper right")
    ax.set_title("Phonon Dispersion: DFT vs MLP")

    ax = axes[1]
    ax.hist(diff.flatten(), bins=50, color="steelblue", edgecolor="none", alpha=0.7)
    ax.axvline(0, color="k", lw=1)
    ax.set_xlabel("Frequency error: MLP − DFT (cm⁻¹)")
    ax.set_ylabel("Count")
    ax.set_title(f"Error Distribution (RMSE={rmse:.1f} cm⁻¹)")

    plt.tight_layout()
    plt.savefig("mlp_phonon_comparison.png", dpi=150, bbox_inches="tight")
    return {"rmse_cm": rmse, "mae_cm": mean_abs_err, "max_err_cm": max_err,
            "mlp_imaginary": mlp_imaginary, "dft_imaginary": dft_imaginary}

# Example usage with MACE calculator
from mace.calculators import MACECalculator
calc = MACECalculator(model_paths=["path/to/mace_model.model"], device="cuda")

poscar = "POSCAR_relaxed"
sc_mat = [[3, 0, 0], [0, 3, 0], [0, 0, 3]]

phonon_dft = phonopy_load("phonopy_dft.yaml")
phonon_mlp = compute_mlp_phonons(calc, poscar, sc_mat)

metrics = compare_phonons(phonon_dft, phonon_mlp)
```

### Workflow 7: Phonon Convergence Checklist

```python
# phonon_convergence_check.py
# Quick sanity check before reporting phonon results

def run_phonon_convergence_checks(phonon, outcar_path=None, n_atoms_primitive=None):
    """
    Run a series of convergence checks on a Phonopy object after force constant production.
    Returns a report dict with pass/fail flags.
    """
    import numpy as np
    report = {}

    # 1. Acoustic sum rule at Gamma
    freq_gamma = phonon.get_frequencies([0, 0, 0])
    acoustic = sorted(np.abs(freq_gamma))[:3]
    asr_max = max(acoustic)
    report["asr_max_cm"] = asr_max
    report["asr_pass"] = asr_max < 1.0

    # 2. Number of acoustic branches = 3 (or correct for 2D/1D)
    n_modes = len(freq_gamma)
    n_expected_acoustic = 3
    report["n_modes"] = n_modes
    report["n_modes_correct"] = (n_modes == 3 * (n_atoms_primitive or phonon.primitive.get_number_of_atoms()))

    # 3. Check for imaginary optical modes at Gamma
    optical_freqs = sorted(freq_gamma)[3:]   # skip 3 acoustic
    imaginary_optical = [f for f in optical_freqs if f < -0.5]
    report["imaginary_optical_at_gamma"] = len(imaginary_optical) > 0
    report["imaginary_optical_freqs"] = imaginary_optical

    # 4. Force constant symmetry (off-diagonal balance)
    fc = phonon.force_constants   # shape: (N_sc_atoms, N_sc_atoms, 3, 3)
    if fc is not None:
        max_fc_asym = np.max(np.abs(fc - np.transpose(fc, (1, 0, 3, 2))))
        report["fc_symmetry_violation"] = float(max_fc_asym)
        report["fc_symmetry_pass"] = max_fc_asym < 0.01   # eV/Å² units

    # 5. Parse OUTCAR for LREAL and EDIFF (if path provided)
    if outcar_path:
        with open(outcar_path) as f:
            content = f.read()
        report["lreal_false"] = "LREAL =  F" in content or "LREAL= FALSE" in content.upper()
        import re
        ediff_match = re.search(r"EDIFF\s*=\s*([0-9.eE+-]+)", content)
        if ediff_match:
            ediff = float(ediff_match.group(1))
            report["ediff"] = ediff
            report["ediff_tight"] = ediff <= 1e-7

    # Print summary
    print("\n=== Phonon Convergence Report ===")
    print(f"  ASR (max acoustic at Γ): {asr_max:.3f} cm⁻¹ — {'PASS' if report['asr_pass'] else 'FAIL'}")
    print(f"  Imaginary optical at Γ: {imaginary_optical} — {'FAIL' if imaginary_optical else 'PASS'}")
    if "lreal_false" in report:
        print(f"  LREAL=False: {report['lreal_false']} — {'PASS' if report['lreal_false'] else 'FAIL (may corrupt forces)'}")
    if "ediff_tight" in report:
        print(f"  EDIFF={report['ediff']:.1e}: {'PASS' if report['ediff_tight'] else 'FAIL (use 1e-8)'}")

    return report
```

## Common Failure Modes

| Failure Mode | Symptoms | Diagnosis | Fix |
|---|---|---|---|
| LREAL=True in VASP | ASR violations 2–20 cm⁻¹; acoustic modes non-zero at Γ; seemingly random optical mode errors | Check INCAR for LREAL=T or LREAL=Auto | Set LREAL=False in all displaced-supercell INCARs |
| EDIFF too loose | ASR violations > 5 cm⁻¹; forces noisy at 1e-5 eV/Å level; low-frequency modes worst affected | Check EDIFF in INCAR; compare forces between ±displacement | Set EDIFF=1e-8 for phonon calculations |
| Supercell too small | Zone-boundary modes poorly converged; artificial dispersion; modes differ between supercell sizes | Compare 2×2×2 vs 3×3×3 phonons; check IFC magnitude at boundary | Increase supercell until phonon frequencies converge to < 1 cm⁻¹ |
| Missing NAC in polar material | LO-TO splitting absent at Γ; phonon bands from all directions merge at Γ instead of splitting | Check if material is polar/ionic; compare TO/LO frequencies | Compute Born charges (LEPSILON=True or ph.x epsil=.true.) and add BORN file |
| Displacement too large | Anharmonic contamination; phonon frequencies depend on displacement amplitude | Rerun with u=0.005, 0.01, 0.02 Å and compare | Use u=0.01 Å as default; reduce to 0.005 Å for soft materials |
| ENCUT too low (Pulay stress) | Systematic frequency errors; forces artificially large; errors scale with volume distortion | Test with ENCUT 20% higher; check pressure in OUTCAR | Increase ENCUT to at least 550 eV or 1.3× ENMAX from POTCAR |
| SIGMA too large (metals) | Optical mode frequencies wrong by 10–30%; temperature-dependent fictitious contributions to forces | Check SIGMA value; compare with SIGMA=0.01 | Use SIGMA≤0.05 eV; ISMEAR=0 for insulators, ISMEAR=1 for metals |
| Structure not fully relaxed | Imaginary modes at Γ (optical); non-zero forces in OUTCAR before displacement | Check residual forces in relaxed POSCAR (EDIFFG convergence) | Tighten relaxation to EDIFFG=-1e-4 or -5e-4; use ISIF=3 |
| Wrong supercell symmetry | Missing/duplicate displacements; wrong number of force sets | Count displaced supercells vs Phonopy expected; check Phonopy output for symmetry detection | Ensure primitive cell is correct; use POSCAR with standard orientation |
| ibrav > 0 in QE | Silent unit cell errors; wrong k-path; non-standard lattice vectors | Use ibrav=0 with explicit CELL_PARAMETERS | Switch to ibrav=0 for all phonon calculations |
| degauss in Ry vs eV confusion (QE) | Smearing 13.6× too large; metallic occupation errors; forces wrong | Check degauss value; 0.01 Ry = 0.136 eV | Remember: QE uses Ry; 0.01 Ry ≈ 0.14 eV; 0.01 eV ≈ 0.00073 Ry |
| NPAR ≠ 1 for LEPSILON in VASP | VASP fails with error about parallelization incompatibility | Check VASP error output | Set NPAR=1 (or remove NPAR) for LEPSILON=True runs |
| Soft mode confused with numerical noise | Apparent imaginary mode at < 5i cm⁻¹ interpreted as instability | Check consistency across supercell sizes; compare with DFPT | Below ~5 cm⁻¹ imaginary is often numerical; > 20 cm⁻¹ is likely physical |
| MLP phonons with imaginary Γ modes | Stable crystal shows imaginary acoustic or optical modes with MLP | Compare DFT phonons; check training data coverage | Add harmonic force constant data or displaced supercell structures to training set |
| Unconverged DFPT (QE ph.x) | ph.x runs many iterations without convergence; warnings about convergence in ph.out | Check tr2_ph; tighten conv_thr in SCF | Tighten conv_thr to 1e-12 in scf.in; tighten tr2_ph to 1e-16; ensure SCF is fully converged first |

## Best Practices

**Before starting:**
- Always fully relax the structure to EDIFFG=-1e-4 eV/Å or tighter with ISIF=3 (cell shape and volume) before computing phonons. Do not compute phonons at a compressed or artificially strained geometry — use the equilibrium volume.
- Verify the relaxed structure has the expected symmetry (`spglib` or `phonopy --symmetry`). Accidental symmetry breaking in the POSCAR will generate wrong displacements.
- Run a preliminary SCF with EDIFF=1e-8 and LREAL=False on the undisplaced supercell and confirm forces < 5×10⁻³ eV/Å. If residual forces are larger, the relaxation is not converged enough for phonon calculations.

**Supercell construction:**
- Use the smallest supercell where all periodic image distances are > 15 Å (insulators) or > 20 Å (metals). Phonopy's `phonopy --dim` suggestion is a starting point, not a guarantee.
- Always run convergence tests with at least two supercell sizes before reporting results.
- For non-cubic systems, choose supercell dimensions to keep all three lengths comparable and all > 15 Å.

**INCAR for displaced supercells (VASP):**
- LREAL=False is non-negotiable for phonon force calculations.
- ADDGRID=True reduces charge density interpolation error.
- ALGO=Normal (Davidson); avoid ALGO=Fast or All for large supercells.
- LWAVE=False, LCHARG=False to reduce I/O — these are single-point calculations.
- NSW=0, IBRION=-1 — do not relax the displaced supercells.

**K-mesh for supercells:**
- The supercell k-mesh is coarser than the primitive cell k-mesh by the supercell multiplicity. A 3×3×3 supercell needs a k-mesh ~1/3 as dense per dimension as the converged primitive cell mesh.
- For insulators: Γ-point only (1×1×1) is often sufficient for large (> 200 atom) supercells.
- For metals: use at least 2×2×2 with Methfessel-Paxton smearing. Verify k-mesh convergence of forces (not just energies).

**Polar materials:**
- Always check if your material is polar before interpreting phonon results near Γ.
- Run a separate LEPSILON=True calculation (VASP) or include epsil=.true. in ph.in (QE).
- Verify Born charge sum rule: Σ_i Z*_i = 0 (for neutral primitive cell). Large violations indicate a poorly converged Born charge calculation.
- Acoustic sum rule and NAC are applied independently; apply both.

**MLP validation:**
- Always compare MLP phonons to DFT phonons on the same structure and with the same phonopy settings (same supercell, same displacement amplitude, same primitive cell).
- A RMSE < 10 cm⁻¹ across all branches is an excellent MLP. RMSE > 50 cm⁻¹ indicates the MLP needs more training data or retraining.
- Imaginary modes in MLP but not DFT is a hard failure and must be fixed before using the MLP for any dynamical property.

**Reporting:**
- Always report supercell size, displacement amplitude, ENCUT, EDIFF, k-mesh, and whether NAC was applied.
- Report ASR violation magnitude (max acoustic frequency at Γ before ASR correction).
- Phonon band structure plots should include a horizontal line at ω=0 to make any imaginary modes visible.

## Integration with Other Skills

**Upstream dependencies (must be completed before phonon calculations):**

- [VASP Workflow](../vasp-workflow/SKILL.md) — structural relaxation (ISIF=3 to equilibrium volume), SCF convergence, POTCAR selection, and job submission patterns. Phonon calculations require a fully relaxed structure from the VASP workflow.
- [DFT Convergence](../dft-convergence/SKILL.md) — ENCUT, k-mesh, and EDIFF convergence protocols. Phonon-specific convergence parameters (ENCUT≥550, EDIFF=1e-8, k-mesh 2× denser than energy-converged) are defined there.
- [Quantum ESPRESSO Workflow](../quantum-espresso/SKILL.md) — QE-specific convergence, pseudopotential selection, and `ph.x` DFPT workflow. The DFPT workflow in this skill (Workflow 3) supplements the QE skill.

**Downstream consumers (phonon outputs feed into):**

- [MLP Validation](../../ml-interatomic-potentials/mlp-validation/SKILL.md) — phonon dispersions are a primary validation metric for ML potentials. The `compare_phonons()` function in Workflow 6 produces the key metrics referenced in the MLP validation skill.
- [MACE Training](../../ml-interatomic-potentials/mace-training/SKILL.md) — phonon-based active learning: displaced supercells with large force errors can be added to the training set to improve MLP phonon accuracy. [FORWARD REFERENCE — skill not yet created]
- [DFT Dataset Generation](../dft-dataset-generation/SKILL.md) — phonon displaced supercells are a standard component of DFT training datasets for MLPs. This skill provides the force calculation protocol for those structures.

**Related domain skills:**

- [DFT Surfaces](../dft-surfaces/SKILL.md) — surface phonons require slab supercells; imaginary surface acoustic modes distinguish surface reconstruction from slab thickness artifacts. [FORWARD REFERENCE — skill not yet created]
- [DFT Defects](../dft-defects/SKILL.md) — defect-induced local vibrational modes (LVM) require phonon calculations on defect supercells; convergence requires very large supercells. [FORWARD REFERENCE — skill not yet created]
- [Reproducibility Workflow](../../scientific-data/reproducibility-workflow/SKILL.md) — phonon calculations involve many input files (INCAR, POSCAR, POTCAR, KPOINTS, phonopy.yaml, BORN) that must be versioned and archived together for reproducibility.

**Data and visualization:**

- [Scientific Visualization](../../scientific-data/scientific-visualization/SKILL.md) — phonon band structure and DOS plotting conventions, color schemes for projected PDOS, and comparison figures for MLP vs DFT phonon validation. [FORWARD REFERENCE — skill not yet created]
- [Pymatgen Analysis](../../scientific-data/pymatgen-analysis/SKILL.md) — structure handling, high-symmetry k-path generation (`HighSymmKpath`), and cross-referencing computed phonons against Materials Project phonon database. [FORWARD REFERENCE — skill not yet created]
- [ASE Framework](../../scientific-data/ase-framework/SKILL.md) — ASE calculators for MLP phonon calculations (Workflow 6); building supercell structures programmatically; converting between ASE and Phonopy structure formats. [FORWARD REFERENCE — skill not yet created]

## Key References

1. **Phonopy documentation:** https://phonopy.github.io/phonopy/ — comprehensive reference for finite-displacement methodology, BORN file format, band structure path generation, and thermal properties.
2. **Togo & Tanaka (2015):** "First principles phonon calculations in materials science." *Scripta Materialia* 108, 1–5. The primary Phonopy reference; must be cited when using Phonopy.
3. **Baroni et al. (2001):** "Phonons and related crystal properties from density-functional perturbation theory." *Rev. Mod. Phys.* 73, 515. Foundational DFPT theory reference.
4. **Wang et al. (2010):** "Mixed-space approach for calculation of vibration-induced dipole-dipole interactions." *Phys. Rev. B* 85, 224303. NAC implementation used in Phonopy.
5. **Setyawan & Curtarolo (2010):** "High-throughput electronic band structure calculations: Challenges and tools." *Comput. Mater. Sci.* 49, 299–312. Standard Brillouin zone path conventions.
6. **SSSP library:** https://www.materialscloud.org/discover/sssp — SSSP efficiency and precision pseudopotential libraries for QE; required to cite when using SSSP pseudopotentials.
7. **PseudoDojo:** http://www.pseudo-dojo.org/ — systematic norm-conserving pseudopotentials; required for spin-orbit coupling calculations.
8. **Quantum ESPRESSO:** Giannozzi et al. (2009, 2017, 2020) — three QE papers covering the codebase evolution; all should be cited for ph.x DFPT calculations.

[EXPERT REVIEW NEEDED: Verify that the displacement amplitude guidance (0.01–0.03 Å range) applies to all mentioned material classes, particularly hydrides and van der Waals materials where harmonic regime may be narrower. Confirm whether 1 cm⁻¹ ASR threshold is appropriate for very soft phonon branches. Review RMSE thresholds for MLP phonon quality assessment — these are based on community practice and may need adjustment for specific applications.]
