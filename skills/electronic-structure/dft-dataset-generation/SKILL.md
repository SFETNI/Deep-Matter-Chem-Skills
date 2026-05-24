---
name: dft-dataset-generation
domain: electronic-structure
maturity: draft
tools:
  - VASP
  - Quantum ESPRESSO
  - ASE
  - pymatgen
  - phonopy
tested_versions:
  VASP: "6.3.2"
  "Quantum ESPRESSO": "7.2"
  ASE: "3.23.0"
  pymatgen: "2024.1.26"
  phonopy: "2.21.0"
requires_gpu: false
estimated_runtime: days-to-weeks
---

# DFT Dataset Generation

## Description

This skill covers the systematic generation of DFT-labeled datasets for training machine-learned interatomic potentials: configuration-space coverage strategy, single-point labeling with VASP and Quantum ESPRESSO, label consistency enforcement (energies, forces, stresses, units, PBC), isolated atom reference energy protocols, dataset assembly in extxyz format, provenance tracking, outlier detection, and failed-calculation triage. Invoke this skill when designing the initial training corpus for an MLP, when extending an existing dataset to new thermodynamic conditions or structure families, or when auditing a dataset inherited from a prior project.

## Domain Context

A machine-learned interatomic potential is only accurate within the region of configuration space spanned by its training data. The central challenge of dataset generation is maximizing that coverage per DFT calculation. DFT calculations of bulk crystals are cheap relative to surface or defect calculations; a naive dataset dominated by near-equilibrium bulk structures will produce a model that is excellent for lattice dynamics but fails immediately for grain boundaries, liquid phases, or adsorption. Conversely, randomly generated structures — even if chemically valid — sample high-energy regions of configuration space that the target MD trajectory will never visit, wasting the DFT budget without improving model quality.

**Coverage vs. cost tradeoff.** Each DFT single-point evaluation costs O(N³) in SCF iterations (where N is the number of electrons) and is entirely determined by the atomic positions, cell, and DFT settings. The dataset designer's task is to identify which structures are both (a) physically accessible in the target application and (b) maximally informative given the structures already in the dataset. Active learning is the principled solution to (b); systematic structure generation (rattling, straining, AIMD sampling, defect enumeration) is the practical solution to (a) at the start of a campaign before an MLP exists.

**Label consistency is non-negotiable.** A training dataset is a set of (structure, energy, forces, stress) tuples where all four quantities were computed at identical DFT settings: same code, same functional, same pseudopotentials/PAW potentials, same energy cutoff, same k-point density, same smearing scheme, same convergence thresholds. Any deviation introduces a systematic offset in the reference energy surface. A single calculation at a tighter ENCUT or a different POTCAR version mixed into an otherwise consistent dataset can degrade model accuracy more than removing 100 correct structures. **This constraint must be established at the start of the campaign and enforced for every calculation thereafter.**

**Stress labels are required for NPT MD.** Many practitioners omit stress labels to avoid the additional DFT cost (VASP: `ISIF=2`; QE: `tstress=.true.`). A model trained without stresses will have no learned information about the volumetric response of the PES and cannot be used for pressure-controlled or NPT simulations. If the target application includes NPT MD or elastic property prediction, stress labels are mandatory from the beginning.

**Isolated atom energies define the zero of the formation energy.** DFT total energies are extensive and include core-electron contributions. Before training, each structure energy is shifted by the sum of isolated atom reference energies (E0) for all atoms in the structure. The E0 values must be computed at the same level of DFT as the rest of the dataset. Errors in E0 produce systematic per-species energy offsets that manifest as wrong formation energies and unstable training.

## When to Use This Skill

- Designing the initial DFT training corpus for a new MLP from scratch.
- Extending an existing MLP training set to a new temperature range, new phase (liquid, amorphous, surface, defect), or new composition that the current model cannot represent.
- Running the DFT oracle in an active learning loop (see `active-learning-mlp`) — the oracle call policy and convergence settings described here apply directly.
- Auditing a dataset inherited from a prior project or collaboration: verifying label consistency, detecting duplicates, checking for missing stress labels, and confirming provenance.
- Generating reference data for validating MLP predictions against DFT (see `mlp-validation`): the test set used for property validation must be generated at identical DFT settings as the training data.

## When Not to Use This Skill

- **When a validated universal potential already covers the system** — generating new DFT data is unnecessary if MACE-MP-0, CHGNet, or M3GNet meets your accuracy requirements on the target system. Benchmark first.
- **For excited-state, non-Born-Oppenheimer, or time-dependent properties** — DFT ground-state total energies are the correct reference for ground-state MLP training only. TDDFT, BSE, or GW data require a different dataset design entirely.
- **For properties requiring explicit treatment of nuclear quantum effects** — DFT static forces are classical; path-integral MD requires a different sampling strategy on top of the same PES.
- **When only a handful (< 50) of structures are needed** — for trivially small datasets, the overhead of a systematic campaign is unnecessary; manually curated structures and direct DFT runs are sufficient.

## Core Concepts

- **Configuration-space coverage:** The set of atomic geometries from which DFT labels are collected. Good coverage means the training set densely samples all regions of configuration space that will be visited during the target application (MD at a given T and P, surface chemistry, defect migration). Poor coverage — even with low test error on the held-out set — guarantees MLP failure for configurations outside the training distribution.
- **Structure families (config_type):** A labeling scheme that groups structures by their physical origin: bulk equilibrium, bulk rattled, bulk strained, surface slab, adsorption, vacancy, interstitial, liquid, amorphous, NEB image, AIMD snapshot, etc. The `config_type` tag is stored in the extxyz `atoms.info` dict and is used for stratified splitting, per-family error analysis, and loss weighting during training.
- **Rattle-and-strain (static generation):** The simplest dataset seed: isotropically and anisotropically strain the equilibrium cell, then apply random displacements (rattle) to each atom. Captures thermal-like displacements and the equation-of-state. Fast to generate; does not capture collective effects or phase transitions. Used as the starting set before any MLP exists.
- **AIMD snapshots:** Frames extracted from ab initio molecular dynamics (typically VASP AIMD or CP2K) at target temperatures. Cover thermally accessible configurations including anharmonic motion, pre-melting disorder, and surface reconstruction. Expensive to generate but produce correlated structures — must be downsampled (every 10–50 steps) to avoid redundancy in the training set.
- **Farthest-point sampling (FPS):** Greedy diversity selection in a structural descriptor space (SOAP, ACE, or MACE latent features) that ensures no two selected structures are more similar than necessary. Reduces the DFT budget by filtering redundant structures before submission.
- **Extxyz format:** ASE's flat-file format for atomistic ML datasets. Each frame contains: line 1 = atom count; line 2 = key-value metadata (`Lattice="..."`, `Properties=...`, `energy=...`, `stress=...`, `pbc=...`); lines 3..N+2 = per-atom species and positions. Forces stored as a `Properties` column. All physical quantities in ASE units: eV, Å, eV/Å (forces), eV/Å³ (stress).
- **Label consistency hash:** A SHA-256 hash of the canonical DFT settings dictionary (INCAR parameters, POTCAR species+version, k-point scheme, code version). Stored in the dataset manifest for every calculation. A mismatch between any two calculations' hashes means they cannot be mixed in the same training set without recomputing one of them.
- **Train / validation / test split:** The test set is a held-out set of DFT-labeled structures that is **never** shown to the model during training, hyperparameter search, or active learning query selection. It must be assembled before any model is trained and contain representative examples of all config_types. Contamination of the test set invalidates all accuracy estimates.
- **Duplicate filtering:** Structures that are identical under translation, rotation, or small numerical perturbation. Duplicates add no information but inflate dataset size, increase training cost, and bias the effective sampling distribution. Detected by pairwise SOAP distances or fingerprint comparison.
- **Provenance record:** A machine-readable record of every DFT calculation: input file hash, code version, pseudopotential filenames and checksums, wall time, HPC job ID, convergence status, and resulting labels. Required for dataset auditing and reproducing results.

## Key Workflows

### Workflow 1: Configuration-space coverage strategy

Systematic generation of candidate structures before DFT labeling. The goal is to cover the physically relevant configuration space at minimal DFT cost by generating diverse, non-redundant structures.

```python
import numpy as np
from ase import Atoms
from ase.build import bulk, surface, make_supercell
from ase.io import write, read
from ase.constraints import FixAtoms
from pathlib import Path
from collections import defaultdict

RNG = np.random.default_rng(seed=42)


# ─── 1. Equilibrium + volumetric strain (EOS coverage) ────────────────────────
def eos_structures(
    base: Atoms,
    vol_range: float = 0.12,
    n_isotropic: int = 13,
    n_anisotropic: int = 10,
    seed: int = 0,
) -> list[Atoms]:
    """
    Isotropic and anisotropic cell deformations.
    Isotropic: probe equation of state.
    Anisotropic: probe elastic constants and shear response.
    """
    rng = np.random.default_rng(seed)
    structs = []

    # Isotropic volume scaling
    for s in np.linspace(1 - vol_range, 1 + vol_range, n_isotropic):
        a = base.copy()
        a.set_cell(a.cell * s**(1/3), scale_atoms=True)
        a.info["config_type"] = "bulk_strain_iso"
        structs.append(a)

    # Anisotropic (random strain tensor, small magnitude)
    for _ in range(n_anisotropic):
        strain = rng.uniform(-0.06, 0.06, (3, 3))
        strain = 0.5 * (strain + strain.T)   # Symmetrize
        np.fill_diagonal(strain, rng.uniform(-0.04, 0.04, 3))
        a = base.copy()
        a.set_cell(a.cell @ (np.eye(3) + strain), scale_atoms=True)
        a.info["config_type"] = "bulk_strain_aniso"
        structs.append(a)

    return structs


# ─── 2. Rattled structures (thermal displacements) ────────────────────────────
def rattled_structures(
    base: Atoms,
    sigmas: list[float] = (0.04, 0.08, 0.12, 0.18, 0.25),
    n_per_sigma: int = 20,
    seed: int = 0,
) -> list[Atoms]:
    """
    Displace atoms from equilibrium to mimic thermal motion.
    sigmas: displacement std dev in Å. 0.04–0.12 Å ≈ 300–1200 K for metals.
    Larger sigma → higher effective temperature; validate forces before including.
    """
    rng = np.random.default_rng(seed)
    structs = []
    for sigma in sigmas:
        for i in range(n_per_sigma):
            a = base.copy()
            displacements = rng.normal(0, sigma, (len(a), 3))
            a.positions += displacements
            a.info["config_type"] = f"bulk_rattle_{sigma:.2f}"
            a.info["rattle_sigma"] = float(sigma)
            structs.append(a)
    return structs


# ─── 3. Surface slabs ─────────────────────────────────────────────────────────
def surface_slab_structures(
    bulk_atoms: Atoms,
    miller_indices: list[tuple],
    layer_counts: list[int],
    vacuum_A: float = 15.0,
    n_rattled_per_slab: int = 10,
    rattle_sigma: float = 0.08,
) -> list[Atoms]:
    """
    Generate surface slabs with rattled top layers.
    Vacuum is included: these are isolated slabs, not repeated surfaces.
    Set PBC=[True, True, False] for the slab direction — important for VASP DIPOL correction.
    [EXPERT REVIEW NEEDED: vacuum, layer count, and whether to use dipole correction
     depend on the surface and property of interest]
    """
    structs = []
    for miller in miller_indices:
        for n_layers in layer_counts:
            slab = surface(bulk_atoms, miller, layers=n_layers, vacuum=vacuum_A / 2)
            slab.center(vacuum=vacuum_A / 2, axis=2)
            slab.info["config_type"] = f"surface_{''.join(map(str, miller))}"
            slab.info["n_layers"] = n_layers
            structs.append(slab)

            # Rattled surface structures (top 2 layers free, bottom fixed)
            n_top = len(slab) // n_layers * 2   # rough estimate of top-2-layer count
            bottom_atoms = [i for i in range(len(slab) - n_top)]
            for j in range(n_rattled_per_slab):
                a = slab.copy()
                displacements = np.zeros((len(a), 3))
                displacements[n_top:] = RNG.normal(0, rattle_sigma, (len(a) - n_top, 3))
                a.positions += displacements
                a.info["config_type"] = f"surface_{''.join(map(str, miller))}_rattle"
                structs.append(a)

    return structs


# ─── 4. Point defects (vacancy, interstitial, substitutional) ────────────────
def defect_structures(
    supercell: Atoms,
    n_rattled_per_defect: int = 15,
    rattle_sigma: float = 0.10,
) -> list[Atoms]:
    """
    Generate vacancy, interstitial, and substitutional defect supercells.
    Supercell should be large enough to avoid defect-defect interactions
    across periodic boundaries: ≥ 3×3×3 for BCC metals (≥ 54 atoms).
    [EXPERT REVIEW NEEDED: required supercell size is system-dependent]
    """
    structs = []
    syms = supercell.get_chemical_symbols()
    unique_species = list(set(syms))

    # Vacancies: remove one atom of each species
    for i, sym in enumerate(syms[:min(len(syms), 3)]):  # sample 3 sites per species
        vacancy = supercell.copy()
        del vacancy[i]
        vacancy.info["config_type"] = f"vacancy_{sym}"
        vacancy.info["defect_type"] = "vacancy"
        vacancy.info["removed_species"] = sym
        structs.append(vacancy)

        # Rattled vacancy supercells
        for j in range(n_rattled_per_defect):
            a = vacancy.copy()
            a.positions += RNG.normal(0, rattle_sigma, (len(a), 3))
            a.info["config_type"] = f"vacancy_{sym}_rattle"
            structs.append(a)

    return structs


# ─── 5. Diversity filtering with SOAP descriptors ────────────────────────────
def fps_filter(
    candidates: list[Atoms],
    n_select: int,
    species: list[str],
    r_cut: float = 5.0,
) -> list[Atoms]:
    """
    Select n_select diverse structures from candidates via farthest-point sampling
    in SOAP descriptor space. Removes redundant structures before DFT submission.
    Requires dscribe and sklearn.
    """
    if len(candidates) <= n_select:
        return candidates

    from dscribe.descriptors import SOAP
    from sklearn.preprocessing import normalize

    soap = SOAP(species=species, r_cut=r_cut, n_max=8, l_max=6,
                average="inner", sparse=False)
    descs = soap.create(candidates)
    descs = normalize(descs, norm="l2")

    selected_idx = [0]
    min_dists = np.full(len(candidates), np.inf)

    for _ in range(n_select - 1):
        last_desc = descs[selected_idx[-1]]
        d = np.linalg.norm(descs - last_desc, axis=1)
        min_dists = np.minimum(min_dists, d)
        min_dists[selected_idx] = -1.0
        selected_idx.append(int(np.argmax(min_dists)))

    return [candidates[i] for i in selected_idx]


# ─── 6. Assemble and write candidate set ─────────────────────────────────────
def build_candidate_set(
    equilibrium_structures: list[Atoms],
    species: list[str],
    output_file: str = "dft_candidates.xyz",
    fps_budget: int = 500,
) -> int:
    """
    Full coverage strategy: EOS + rattle + surface + defect → FPS filter → extxyz.
    Returns total number of structures selected for DFT labeling.
    """
    all_candidates = []
    for base in equilibrium_structures:
        all_candidates += eos_structures(base)
        sc = make_supercell(base, [[3, 0, 0], [0, 3, 0], [0, 0, 3]])
        all_candidates += rattled_structures(sc)

    all_candidates = fps_filter(all_candidates, fps_budget, species)

    # Tag with provenance before writing
    for i, a in enumerate(all_candidates):
        a.info["candidate_index"] = i
        a.info["al_iteration"] = 0
        a.info.setdefault("config_type", "bulk")

    write(output_file, all_candidates, format="extxyz")
    print(f"Candidate set: {len(all_candidates)} structures → {output_file}")

    # Report config_type distribution
    counts = defaultdict(int)
    for a in all_candidates:
        counts[a.info.get("config_type", "unknown")] += 1
    for ctype, n in sorted(counts.items()):
        print(f"  {ctype:<35}: {n:>4}")

    return len(all_candidates)
```

### Workflow 2: VASP single-point labeling pipeline

VASP is the most common DFT oracle for MLP training data. The settings below are designed for consistent, converged labels across a heterogeneous structural dataset.

```python
from pathlib import Path
from pymatgen.io.vasp.sets import MPStaticSet
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.vasp.inputs import Incar, Kpoints
from ase.io import read
import json
import hashlib


# ─── Canonical DFT settings for MLP training data ────────────────────────────
# FREEZE AT CAMPAIGN START. Any change invalidates all prior calculations.
MLP_VASP_INCAR = {
    "ENCUT":   520,      # eV; must be converged for all species in the dataset
    "EDIFF":   1e-6,     # eV; tighter than default for force accuracy
    "NSW":     0,        # Static calculation (no ionic relaxation)
    "IBRION": -1,
    "ISMEAR":  0,        # Gaussian smearing: safe for all gap sizes
    "SIGMA":   0.10,     # eV; small enough to approximate 0 K
    "PREC":    "Accurate",
    "ALGO":    "Normal",
    "LWAVE":   False,    # Do not write WAVECAR — saves enormous disk space
    "LCHARG":  False,    # Do not write CHGCAR
    "ISIF":    2,        # Compute stress tensor; do NOT relax cell (NSW=0 anyway)
    "LREAL":   "Auto",
    "NELM":    200,
    "NELMIN":  4,
    # Spin: set ISPIN=2 for magnetic systems; MAGMOM must be set per species
    # For non-magnetic systems: ISPIN=1 (default, do not set explicitly)
}

# k-point density (Å⁻¹). Convergence requires ~0.25 Å⁻¹ for bulk metals;
# can be relaxed to ~0.30 Å⁻¹ for large supercells. [EXPERT REVIEW NEEDED]
KPOINT_DENSITY_REC = 0.25   # Å⁻¹ in reciprocal space

# Canonical settings fingerprint — store in every manifest
def settings_hash(incar: dict, kpoint_density: float) -> str:
    d = {"incar": {k: str(v) for k, v in sorted(incar.items())},
         "kpoint_density": kpoint_density}
    return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:16]

SETTINGS_HASH = settings_hash(MLP_VASP_INCAR, KPOINT_DENSITY_REC)
print(f"DFT settings hash (first 16 chars): {SETTINGS_HASH}")


def prepare_vasp_batch(
    candidate_file: str,
    output_base: Path,
    al_iteration: int,
    potcar_functional: str = "PBE",
    incar_overrides: dict = None,
) -> Path:
    """
    Write VASP input directories for each structure in the candidate file.
    Returns path to a batch manifest JSON.

    incar_overrides: settings to add or override for specific structure types
        (e.g., ISPIN=2, MAGMOM for magnetic structures).
    """
    output_base.mkdir(parents=True, exist_ok=True)
    frames = read(candidate_file, index=":", format="extxyz")
    adaptor = AseAtomsAdaptor()
    manifest = {
        "al_iteration": al_iteration,
        "n_structures": len(frames),
        "settings_hash": SETTINGS_HASH,
        "incar": MLP_VASP_INCAR,
        "kpoint_density": KPOINT_DENSITY_REC,
        "potcar_functional": potcar_functional,
        "structures": [],
    }

    for i, atoms in enumerate(frames):
        calc_dir = output_base / f"iter{al_iteration:03d}_struct{i:04d}"
        calc_dir.mkdir(exist_ok=True)

        structure = adaptor.get_structure(atoms)
        incar_settings = dict(MLP_VASP_INCAR)
        if incar_overrides:
            incar_settings.update(incar_overrides)

        # Auto k-points based on reciprocal lattice and density target
        vis = MPStaticSet(
            structure,
            user_incar_settings=incar_settings,
            user_kpoints_settings={
                "reciprocal_density": int(1 / KPOINT_DENSITY_REC**3),
            },
            potcar_functional=potcar_functional,
        )
        vis.write_input(str(calc_dir))

        # Write provenance sidecar
        sidecar = {
            "candidate_index": atoms.info.get("candidate_index", i),
            "config_type": atoms.info.get("config_type", "unknown"),
            "al_iteration": al_iteration,
            "settings_hash": SETTINGS_HASH,
            "natoms": len(atoms),
        }
        (calc_dir / "provenance.json").write_text(json.dumps(sidecar, indent=2))

        manifest["structures"].append({
            "dir": calc_dir.name,
            "config_type": sidecar["config_type"],
            "natoms": len(atoms),
        })

    manifest_file = output_base / f"iter{al_iteration:03d}_manifest.json"
    manifest_file.write_text(json.dumps(manifest, indent=2))
    print(f"Prepared {len(frames)} VASP calculations in {output_base}")
    return manifest_file


def collect_vasp_labels(
    batch_dir: Path,
    al_iteration: int,
    check_settings_hash: str = SETTINGS_HASH,
) -> list:
    """
    Parse completed VASP calculations and return labeled ASE Atoms in extxyz format.
    Filters unconverged SCF and wrong-settings calculations.
    """
    from pymatgen.io.vasp.outputs import Vasprun
    import numpy as np

    KBAR_TO_EV_A3 = 1.0 / 160.21766208

    labeled, n_skipped = [], 0
    for calc_dir in sorted(batch_dir.glob("iter*_struct*")):
        vr_path = calc_dir / "vasprun.xml"
        if not vr_path.exists():
            n_skipped += 1
            continue

        # Verify settings hash from provenance sidecar
        prov_file = calc_dir / "provenance.json"
        if prov_file.exists():
            prov = json.loads(prov_file.read_text())
            if prov.get("settings_hash") != check_settings_hash:
                print(f"SKIP {calc_dir.name}: settings hash mismatch — "
                      f"expected {check_settings_hash}, got {prov.get('settings_hash')}")
                n_skipped += 1
                continue

        try:
            v = Vasprun(str(vr_path), parse_potcar_file=False)
        except Exception as e:
            print(f"SKIP {calc_dir.name}: parse error: {e}")
            n_skipped += 1
            continue

        if not v.converged_electronic:
            print(f"SKIP {calc_dir.name}: SCF not converged (NELM reached or EDIFF not met)")
            n_skipped += 1
            continue

        structure = v.final_structure
        last_step = v.ionic_steps[-1]

        from ase import Atoms as AseAtoms
        atoms = AseAtoms(
            symbols=[str(s) for s in structure.species],
            positions=structure.cart_coords,
            cell=structure.lattice.matrix,
            pbc=True,
        )
        atoms.info["energy"] = v.final_energy    # eV
        atoms.arrays["forces"] = np.array(last_step["forces"])   # eV/Å

        # Stress: VASP kBar compression-positive → ASE eV/Å³ tensile-positive
        stress_kbar = np.array(last_step["stress"])
        stress_voigt = np.array([
            stress_kbar[0,0], stress_kbar[1,1], stress_kbar[2,2],
            stress_kbar[1,2], stress_kbar[0,2], stress_kbar[0,1],
        ])
        atoms.info["stress"] = -stress_voigt * KBAR_TO_EV_A3   # eV/Å³

        # Carry over provenance metadata
        if prov_file.exists():
            prov = json.loads(prov_file.read_text())
            atoms.info["config_type"] = prov.get("config_type", "unknown")
            atoms.info["al_iteration"] = al_iteration
            atoms.info["settings_hash"] = check_settings_hash

        labeled.append(atoms)

    print(f"Collected {len(labeled)} labeled structures ({n_skipped} skipped)")
    return labeled
```

### Workflow 3: Quantum ESPRESSO single-point labeling pipeline

```python
from pathlib import Path
from ase.io import read, write
from ase.calculators.espresso import Espresso, EspressoProfile
import numpy as np
import json


# ─── Canonical QE settings for MLP training data ────────────────────────────
# ecutwfc in Ry; typical PBE PAW values. Adjust per pseudopotential.
# [EXPERT REVIEW NEEDED: cutoffs are pseudopotential-dependent; use SSSP or
#  PseudoDojo values for each species in the target system]
MLP_QE_SETTINGS = {
    "calculation": "scf",
    "restart_mode": "from_scratch",
    "tprnfor": True,        # Compute forces (required)
    "tstress": True,        # Compute stress tensor (required for NPT training)
    "ecutwfc": 80,          # Ry; SSSP Precision value for most elements
    "ecutrho": 640,         # Ry; 8× ecutwfc for ultrasoft/PAW; 4× for norm-conserving
    "conv_thr": 1e-10,      # Ry; tighter than default for force accuracy
    "mixing_beta": 0.4,
    "electron_maxstep": 200,
    "occupations": "smearing",
    "smearing": "gauss",
    "degauss": 0.01,        # Ry (≈ 0.14 eV); NOT eV — common unit error
    "vdw_corr": "none",     # Set to "dft-d3" or "grimme-d3" if dispersion needed
}

# CRITICAL: degauss is in Ry, NOT eV. 0.01 Ry ≈ 0.136 eV.
# A common error is copying a VASP SIGMA value (eV) directly as degauss (Ry),
# which gives 13.6× too much smearing.

QE_PSEUDOPOTENTIAL_MAP = {
    # Format: "Element": "pseudopotential_filename"
    # Use SSSP Efficiency or PseudoDojo for ML training data.
    # Record exact filenames — they encode the version and type.
    "W":  "W.pbe-spfn-kjpaw_psl.1.0.0.UPF",
    "Ta": "Ta.pbe-spfn-kjpaw_psl.1.0.0.UPF",
    # Add all species in the training set.
}


def prepare_qe_batch(
    candidate_file: str,
    output_base: Path,
    al_iteration: int,
    pseudo_dir: str = "/path/to/pseudopotentials",
    k_spacing: float = 0.25,   # Å⁻¹; reciprocal-space k-point density
) -> Path:
    """
    Write Quantum ESPRESSO input files for each candidate structure.
    k_spacing: target k-point density (Å⁻¹). Auto-meshes using ASE.
    """
    output_base.mkdir(parents=True, exist_ok=True)
    frames = read(candidate_file, index=":", format="extxyz")
    manifest = {"al_iteration": al_iteration, "structures": []}

    for i, atoms in enumerate(frames):
        calc_dir = output_base / f"iter{al_iteration:03d}_struct{i:04d}"
        calc_dir.mkdir(exist_ok=True)

        # Auto k-point mesh from cell dimensions
        from ase.calculators.espresso import EspressoProfile
        from ase.dft.kpoints import monkhorst_pack
        reciprocal_lengths = np.linalg.norm(
            np.linalg.inv(atoms.cell).T, axis=1
        )  # Å⁻¹
        kgrid = [max(1, int(np.ceil(rl / k_spacing)))
                 for rl in reciprocal_lengths]

        input_data = dict(MLP_QE_SETTINGS)
        # For isolated molecules: use gamma-only, no smearing
        if not all(atoms.pbc):
            kgrid = [1, 1, 1]
            input_data["occupations"] = "fixed"
            del input_data["smearing"]
            del input_data["degauss"]

        calc = Espresso(
            profile=EspressoProfile(command="pw.x -in PREFIX.pwi > PREFIX.pwo"),
            directory=str(calc_dir),
            pseudopotentials=QE_PSEUDOPOTENTIAL_MAP,
            pseudo_dir=pseudo_dir,
            kpts=kgrid,
            input_data=input_data,
        )
        calc.write_input(atoms)

        # Provenance sidecar
        prov = {
            "config_type": atoms.info.get("config_type", "unknown"),
            "al_iteration": al_iteration,
            "natoms": len(atoms),
            "kgrid": kgrid,
            "ecutwfc_Ry": input_data["ecutwfc"],
            "ecutrho_Ry": input_data["ecutrho"],
            "degauss_Ry": input_data.get("degauss"),
            "pseudopotentials": QE_PSEUDOPOTENTIAL_MAP,
        }
        (calc_dir / "provenance.json").write_text(json.dumps(prov, indent=2))
        manifest["structures"].append({"dir": calc_dir.name,
                                       "config_type": prov["config_type"]})

    manifest_file = output_base / f"iter{al_iteration:03d}_manifest.json"
    manifest_file.write_text(json.dumps(manifest, indent=2))
    print(f"Prepared {len(frames)} QE calculations in {output_base}")
    return manifest_file


def collect_qe_labels(batch_dir: Path, al_iteration: int) -> list:
    """
    Parse completed QE pw.x output files and return labeled ASE Atoms.
    QE output units: energy in Ry, forces in Ry/bohr, stress in kbar.
    ASE's read("PREFIX.pwo", format="espresso-out") handles unit conversion.
    """
    labeled, n_skipped = [], 0

    for calc_dir in sorted(batch_dir.glob("iter*_struct*")):
        # ASE QE parser looks for PREFIX.pwo
        pwo_files = list(calc_dir.glob("*.pwo"))
        if not pwo_files:
            n_skipped += 1
            continue

        try:
            # ASE automatically converts Ry→eV, bohr→Å, and Ry/bohr→eV/Å
            atoms = read(str(pwo_files[0]), format="espresso-out")
        except Exception as e:
            print(f"SKIP {calc_dir.name}: {e}")
            n_skipped += 1
            continue

        # Check convergence: ASE QE parser raises StopIteration if not converged,
        # but double-check by inspecting the output for "convergence NOT achieved"
        output_text = pwo_files[0].read_text()
        if "convergence NOT achieved" in output_text:
            print(f"SKIP {calc_dir.name}: QE SCF not converged")
            n_skipped += 1
            continue

        # Verify stress was computed
        if atoms.info.get("stress") is None and "tstress" in str(output_text):
            print(f"WARNING {calc_dir.name}: stress not found despite tstress=.true.")

        # ASE QE reader stores energy as atoms.get_potential_energy() (eV)
        # and forces in atoms.arrays["forces"] (eV/Å)
        # Stress is in atoms.info["stress"] (eV/Å³, Voigt, tensile positive)
        # — ASE handles the sign and unit conversion from QE automatically.

        prov_file = calc_dir / "provenance.json"
        if prov_file.exists():
            prov = json.loads(prov_file.read_text())
            atoms.info["config_type"] = prov.get("config_type", "unknown")
            atoms.info["al_iteration"] = al_iteration

        labeled.append(atoms)

    print(f"Collected {len(labeled)} labeled QE structures ({n_skipped} skipped)")
    return labeled
```

### Workflow 4: Dataset assembly, quality checks, and outlier detection

```python
import numpy as np
from ase.io import read, write
from collections import defaultdict
from pathlib import Path


def check_label_completeness(frames: list, require_stress: bool = True) -> list[str]:
    """
    Verify that every frame has the required labels in the correct units.
    Returns list of warning/error strings.
    """
    issues = []
    for i, atoms in enumerate(frames):
        ctype = atoms.info.get("config_type", f"frame_{i}")

        # Energy
        e = atoms.info.get("energy")
        if e is None:
            issues.append(f"[ERROR] Frame {i} ({ctype}): missing 'energy' label")
        else:
            epa = e / len(atoms)
            # eV/atom sanity range: covers most inorganic/organic systems
            if not (-30.0 < epa < 10.0):
                issues.append(f"[WARNING] Frame {i} ({ctype}): unusual energy/atom "
                              f"{epa:.3f} eV/atom — check units (Ry? Ha?) or structure")

        # Forces
        forces = atoms.arrays.get("forces")
        if forces is None:
            issues.append(f"[ERROR] Frame {i} ({ctype}): missing 'forces' array")
        else:
            fmax = np.abs(forces).max()
            if fmax > 50.0:
                issues.append(f"[WARNING] Frame {i} ({ctype}): max force {fmax:.1f} eV/Å "
                              f"— verify structure is physical and SCF converged")
            if fmax < 1e-8 and len(atoms) > 1:
                issues.append(f"[WARNING] Frame {i} ({ctype}): forces are all zero — "
                              f"possibly a failed or incompletely parsed output")

        # Stress
        stress = atoms.info.get("stress")
        if require_stress:
            if stress is None:
                issues.append(f"[ERROR] Frame {i} ({ctype}): missing 'stress' label "
                              f"— required for NPT MD; set ISIF=2 (VASP) or tstress=.true. (QE)")
            else:
                stress_arr = np.asarray(stress)
                if stress_arr.shape != (6,):
                    issues.append(f"[ERROR] Frame {i} ({ctype}): stress shape {stress_arr.shape}, "
                                  f"expected (6,) Voigt")

        # PBC
        if not all(atoms.pbc) and atoms.info.get("config_type", "").startswith("bulk"):
            issues.append(f"[WARNING] Frame {i} ({ctype}): PBC is not all-True for bulk structure")

        # Minimum interatomic distance
        dists = atoms.get_all_distances(mic=True)
        np.fill_diagonal(dists, np.inf)
        dmin = dists.min()
        if dmin < 0.8:  # Å; extremely short bond — unphysical geometry
            issues.append(f"[ERROR] Frame {i} ({ctype}): minimum interatomic distance "
                          f"{dmin:.3f} Å — atom overlap; exclude from training")

    if not issues:
        print(f"Label check PASSED for {len(frames)} frames")
    else:
        n_err = sum(1 for s in issues if s.startswith("[ERROR]"))
        n_warn = sum(1 for s in issues if s.startswith("[WARNING]"))
        print(f"Label check: {n_err} errors, {n_warn} warnings in {len(frames)} frames")
        for issue in issues[:30]:  # Cap output
            print(f"  {issue}")

    return issues


def filter_duplicates(
    frames: list,
    species: list[str],
    distance_threshold: float = 0.005,   # Euclidean SOAP distance; adjust empirically
) -> tuple[list, list[int]]:
    """
    Remove near-duplicate structures using SOAP descriptor pairwise distances.
    Returns (deduplicated_frames, removed_indices).

    distance_threshold: structures with SOAP distance < threshold are considered
    duplicates. 0.005 is tight (removes very similar structures); 0.01 is more
    aggressive. [EXPERT REVIEW NEEDED: threshold is descriptor- and system-dependent]
    """
    from dscribe.descriptors import SOAP
    from sklearn.preprocessing import normalize

    soap = SOAP(species=species, r_cut=5.0, n_max=8, l_max=6,
                average="inner", sparse=False)
    descs = normalize(soap.create(frames), norm="l2")

    keep = [True] * len(frames)
    for i in range(len(frames)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(frames)):
            if not keep[j]:
                continue
            if np.linalg.norm(descs[i] - descs[j]) < distance_threshold:
                keep[j] = False   # Remove the later frame (keep the first occurrence)

    deduplicated = [f for f, k in zip(frames, keep) if k]
    removed = [i for i, k in enumerate(keep) if not k]
    print(f"Duplicate filtering: {len(frames)} → {len(deduplicated)} "
          f"({len(removed)} removed, threshold={distance_threshold})")
    return deduplicated, removed


def dataset_statistics(frames: list) -> None:
    """Print summary statistics for a labeled dataset."""
    by_type = defaultdict(list)
    for f in frames:
        by_type[f.info.get("config_type", "unknown")].append(f)

    print(f"\nDataset summary: {len(frames)} total frames")
    print(f"{'Config type':<30}  {'N':>5}  {'N_atoms range':>15}  "
          f"{'E/atom range (eV)':>20}")
    print("─" * 75)
    for ctype, fs in sorted(by_type.items()):
        natoms_range = f"{min(len(f) for f in fs)}–{max(len(f) for f in fs)}"
        energies = [f.info["energy"] / len(f) for f in fs
                    if f.info.get("energy") is not None]
        if energies:
            e_range = f"{min(energies):.2f} – {max(energies):.2f}"
        else:
            e_range = "N/A"
        print(f"{ctype:<30}  {len(fs):>5}  {natoms_range:>15}  {e_range:>20}")
```

### Workflow 5: Isolated atom reference energy protocol

```python
from pathlib import Path
from ase.build import molecule
from ase.io import write


# ─── Isolated atom energies: the zero of the MLP energy scale ─────────────────
# E0 must be computed at IDENTICAL DFT settings as the rest of the training set.
# Use a large vacuum box (≥ 15 Å per side) to eliminate self-image interactions.
# For magnetic atoms (Fe, Ni, Co, Cr, Mn), use spin-polarized DFT (ISPIN=2).
# The spin state must be the physical ground state, not the default.
# [EXPERT REVIEW NEEDED: appropriate spin states and MAGMOM values are species-dependent]

ISOLATED_ATOM_VASP_INCAR = {
    "ENCUT":   520,
    "EDIFF":   1e-7,     # Tighter than bulk: single atom, cheap
    "NSW":     0,
    "IBRION": -1,
    "ISMEAR":  0,
    "SIGMA":   0.01,     # Very small smearing: isolated atom has no band structure
    "PREC":    "Accurate",
    "ALGO":    "Normal",
    "LWAVE":   False,
    "LCHARG":  False,
    "ISIF":    0,        # No stress for isolated atom
    "NELM":    200,
    # ISPIN=2 and MAGMOM must be set for open-shell atoms:
    # "ISPIN": 2,
    # "MAGMOM": "1 3.0",   # 1 atom, initial magnetic moment 3.0 μB (e.g. for Fe)
    # "NUPDOWN": -1,       # Let VASP determine electron number per spin channel
}

ATOM_BOX_SIZE = 15.0   # Å; cubic box for isolated atom calculation


def build_isolated_atom_inputs(
    species: list[str],
    output_base: Path,
    magnetic_moments: dict = None,  # e.g. {"Fe": 4.0, "Ni": 2.0}
) -> None:
    """
    Write VASP input files for isolated atom reference energy calculations.
    magnetic_moments: initial MAGMOM for spin-polarized atoms.
    For non-magnetic species, ISPIN=1 (not set) is correct.
    """
    output_base.mkdir(parents=True, exist_ok=True)
    for elem in species:
        from ase import Atoms
        atom = Atoms(
            symbols=[elem],
            positions=[[ATOM_BOX_SIZE/2, ATOM_BOX_SIZE/2, ATOM_BOX_SIZE/2]],
            cell=[ATOM_BOX_SIZE, ATOM_BOX_SIZE, ATOM_BOX_SIZE],
            pbc=True,    # VASP requires PBC even for isolated atoms
        )

        calc_dir = output_base / f"atom_{elem}"
        calc_dir.mkdir(exist_ok=True)
        write(str(calc_dir / "POSCAR"), atom, format="vasp", direct=False)

        incar = dict(ISOLATED_ATOM_VASP_INCAR)
        if magnetic_moments and elem in magnetic_moments:
            incar["ISPIN"] = 2
            incar["MAGMOM"] = f"1 {magnetic_moments[elem]:.1f}"

        # Write INCAR manually (simple enough not to need MPStaticSet)
        incar_lines = [f"{k} = {v}" for k, v in sorted(incar.items())]
        (calc_dir / "INCAR").write_text("\n".join(incar_lines) + "\n")
        # KPOINTS: Gamma-only for isolated atom (no k-point sampling needed)
        (calc_dir / "KPOINTS").write_text(
            "Isolated atom\n0\nGamma\n1 1 1\n0 0 0\n"
        )
        # POTCAR: must match the species and functional of the training dataset
        print(f"  {elem}: write POTCAR manually (cp POTCAR_{elem} {calc_dir}/POTCAR)")


def collect_isolated_atom_energies(atom_calc_base: Path) -> dict:
    """
    Parse completed isolated atom calculations and return E0 dict.
    Returns {element: E0_eV} for use as MACE --E0s argument.
    """
    from pymatgen.io.vasp.outputs import Vasprun
    E0 = {}
    for calc_dir in sorted(atom_calc_base.glob("atom_*")):
        elem = calc_dir.name.replace("atom_", "")
        vr_path = calc_dir / "vasprun.xml"
        if not vr_path.exists():
            print(f"WARNING: atom_{elem}: no vasprun.xml")
            continue
        try:
            v = Vasprun(str(vr_path), parse_potcar_file=False)
            if not v.converged_electronic:
                print(f"WARNING: atom_{elem}: not converged")
                continue
            E0[elem] = v.final_energy
            print(f"  E0[{elem}] = {v.final_energy:.6f} eV")
        except Exception as e:
            print(f"WARNING: atom_{elem}: {e}")

    return E0


def format_mace_E0s(E0: dict) -> str:
    """Format E0 dict as MACE --E0s argument string."""
    items = ", ".join(f'"{k}": {v:.6f}' for k, v in sorted(E0.items()))
    return "{" + items + "}"
    # Usage: mace_run_train ... --E0s='{"W": -12.345678, "Ta": -11.234567}'
```

### Workflow 6: Provenance tracking and dataset manifest

```python
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from ase.io import read, write


class DatasetManifest:
    """
    Tracks provenance, settings consistency, and iteration history for an MLP
    training dataset. The manifest is a machine-readable audit trail stored as
    JSON alongside the extxyz dataset file.
    """

    def __init__(self, manifest_file: str):
        self.manifest_file = Path(manifest_file)
        if self.manifest_file.exists():
            self.data = json.loads(self.manifest_file.read_text())
        else:
            self.data = {
                "created": datetime.utcnow().isoformat(),
                "dft_settings_hash": None,
                "dft_code": None,
                "functional": None,
                "iterations": [],
                "total_structures": 0,
                "config_type_counts": {},
                "pseudopotential_checksums": {},
            }

    def register_settings(
        self,
        dft_code: str,
        functional: str,
        incar_or_input_dict: dict,
        pseudopotential_files: list[str],
    ) -> str:
        """
        Register the canonical DFT settings at the start of a campaign.
        Computes and stores a settings hash; raises if called a second time
        with different settings (enforces consistency).
        """
        # Compute hash of DFT inputs
        settings_str = json.dumps(incar_or_input_dict, sort_keys=True)
        h = hashlib.sha256(settings_str.encode()).hexdigest()[:16]

        if self.data["dft_settings_hash"] is None:
            self.data["dft_settings_hash"] = h
            self.data["dft_code"] = dft_code
            self.data["functional"] = functional
            print(f"DFT settings registered: hash={h}, code={dft_code}, functional={functional}")
        elif self.data["dft_settings_hash"] != h:
            raise ValueError(
                f"DFT settings hash mismatch! Existing: {self.data['dft_settings_hash']}, "
                f"New: {h}. Mixing different DFT settings in one dataset is not allowed."
            )

        # Record pseudopotential checksums
        for pp_file in pseudopotential_files:
            pp_path = Path(pp_file)
            if pp_path.exists():
                with open(pp_path, "rb") as f:
                    pp_hash = hashlib.sha256(f.read()).hexdigest()[:16]
                self.data["pseudopotential_checksums"][pp_path.name] = pp_hash

        self._save()
        return h

    def add_iteration(
        self,
        frames: list,
        al_iteration: int,
        notes: str = "",
    ) -> None:
        """
        Record a new set of labeled structures in the manifest.
        Validates settings hash on each frame before adding.
        """
        registered_hash = self.data["dft_settings_hash"]
        for frame in frames:
            frame_hash = frame.info.get("settings_hash")
            if frame_hash and frame_hash != registered_hash:
                raise ValueError(
                    f"Frame settings_hash {frame_hash} != registered {registered_hash} — "
                    f"do not mix DFT settings within one dataset"
                )

        # Count config types
        from collections import Counter
        ctype_counts = Counter(f.info.get("config_type", "unknown") for f in frames)
        for ctype, count in ctype_counts.items():
            self.data["config_type_counts"][ctype] = (
                self.data["config_type_counts"].get(ctype, 0) + count
            )

        self.data["iterations"].append({
            "iteration": al_iteration,
            "n_new": len(frames),
            "timestamp": datetime.utcnow().isoformat(),
            "notes": notes,
            "config_type_counts": dict(ctype_counts),
        })
        self.data["total_structures"] += len(frames)
        self._save()
        print(f"Manifest updated: iteration {al_iteration}, +{len(frames)} structures "
              f"(total={self.data['total_structures']})")

    def report(self) -> None:
        """Print a human-readable summary of the dataset manifest."""
        print(f"\nDataset manifest: {self.manifest_file}")
        print(f"  DFT code:      {self.data['dft_code']}")
        print(f"  Functional:    {self.data['functional']}")
        print(f"  Settings hash: {self.data['dft_settings_hash']}")
        print(f"  Total:         {self.data['total_structures']} structures")
        print(f"  Iterations:    {len(self.data['iterations'])}")
        print(f"\n  Config type distribution:")
        for ctype, count in sorted(self.data["config_type_counts"].items()):
            print(f"    {ctype:<35}: {count:>5}")

    def _save(self):
        self.manifest_file.write_text(json.dumps(self.data, indent=2))
```

## Common Failure Modes

| Failure | Symptom | Likely Cause | Diagnostic | Fix |
|---------|---------|--------------|------------|-----|
| Inconsistent DFT settings across iterations | Model accuracy degrades after adding new data; training loss increases when new structures are included; systematic energy offsets between dataset iterations | ENCUT, k-point density, POTCAR version, smearing scheme, or functional changed between iterations; VASP version upgrade changed default behavior | Compare INCAR files across all iterations; check POTCAR file checksums; run settings hash comparison on all provenance.json sidecars | Recompute affected structures at the canonical settings; use `DatasetManifest.register_settings` to enforce hash-check before any calculation is admitted to the dataset |
| Unconverged SCF labels silently included | Force MAE plateau despite increasing dataset size; outlier forces appear in training set; specific config_types drive disproportionate training loss | SCF convergence check not applied during label collection; VASP ran to NELM without reaching EDIFF; QE printed "convergence NOT achieved" but result was collected | Filter all calculations with `v.converged_electronic = False` (VASP) or grep output for "convergence NOT achieved" (QE); plot force distribution by config_type — unconverged calculations produce outlier forces | Always filter unconverged calculations in `collect_vasp_labels` / `collect_qe_labels`; increase NELM to 200–300 for difficult structures; check for partially-filled bands requiring ISMEAR=−5 |
| Missing stress labels — NPT MD fails | MLP trained without stresses predicts correct forces and energies but NPT MD density drifts; elastic constants are undefined; EOS fitting gives wrong bulk modulus | ISIF < 2 in VASP INCAR (stress not computed); tstress=.false. or omitted in QE; stress labels were stripped during dataset assembly | Check extxyz frames: `atoms.info.get("stress")` should return a length-6 array; run `grep stress *.xyz | head` | Set ISIF=2 (VASP) or tstress=.true. (QE) in the canonical INCAR from day one; if missing, recompute affected structures — there is no cheaper workaround |
| Broken periodic boundary conditions | Model fails for surfaces or interfaces; stress predictions are nonsensical for slab structures; atoms leave the box in MD | PBC set to False for slab structures that should be periodic in 2D; or PBC set to True for isolated molecules; or slab structures written without 3D periodicity but VASP run with 3D PBC | Check `atoms.pbc` for every frame; slab structures should have pbc=[True, True, False] in the non-periodic direction; run `grep pbc *.xyz` | Set PBC correctly in `ase write extxyz`; for VASP slabs use IDIPOL/LDIPOL if needed; ensure slab vacuum is large enough that the False-pbc direction is truly non-periodic (≥ 15 Å vacuum) |
| Duplicate structures inflate dataset — no accuracy gain | Dataset grows but model accuracy stops improving; training is slow and expensive; SOAP-PCA shows tight clusters | Trajectory subsampled too densely (every step instead of every 10–50 steps); rattle with the same seed repeated; same equilibrium structure added at multiple iterations | Compute pairwise SOAP distances for new additions; if median distance < 0.005, duplicates are present; plot SOAP-PCA colored by config_type | Apply FPS filter before each DFT batch (Workflow 1); increase AIMD subsampling interval; check seeds for rattled-structure generation are unique per iteration |
| Train/test leakage from trajectory-based splitting | Test force MAE appears very low (< 10 meV/Å) but model fails during production MD; active learning shows overconfidence | Random train/test split applied to frames from the same AIMD trajectory; adjacent frames are nearly identical; information about test frames is effectively available in training | Compute SOAP distance between each test frame and its nearest training neighbor; if > 50% have distance < 0.005, the split is contaminated | Split by trajectory origin, not by frame index; use `split_by_config_type` (see mace-training Workflow 2); group all frames from one trajectory and assign the entire group to one split |
| Overrepresentation of near-equilibrium structures | Model accurate for bulk equilibrium but fails for surfaces, high-T, defects; active learning immediately targets surface and defect configurations | Initial dataset design focused on rattled bulk only; surfaces and defects never sampled before active learning begins; learning curve for minority config_types is flat | Report per-config_type counts in the manifest; if any target config_type has < 5% of training structures, it is underrepresented | Explicitly generate surface, defect, and liquid structures in the initial campaign (Workflow 1); apply config_type_weights in MACE training loss to up-weight rare but critical environments |
| Unphysical high-energy geometries in training set | Outlier forces (> 50 eV/Å) in the training set; MLP learns to predict unphysical forces for compressed or overlapping geometries; NVE test fails immediately at high T | Large-sigma rattled structures or strained cells place atoms unrealistically close (d < 1 Å); DFT calculated forces for these structures but they are not physically accessible in MD | Check `atoms.get_all_distances(mic=True).min()` for every frame; plot force magnitude distribution and flag frames with |F| > 30 eV/Å; visualize outlier structures | Exclude frames with any interatomic distance < 0.8 Å; reduce maximum rattle sigma or maximum volumetric strain; or add short-range ZBL correction to the MLP for high-energy repulsion |

## Best Practices

- **Freeze the DFT settings before the first calculation and document them in a canonical `dft_settings.json`.** Any change invalidates all prior calculations for training purposes. Record the VASP version, POTCAR ENMAX and version, and QE ecutwfc alongside the INCAR/input.dat parameters.
- **Always compute stress tensors (ISIF=2 in VASP; tstress=.true. in QE) from the first iteration.** Going back to recompute stresses for hundreds of existing structures is expensive and error-prone. Even if NPT MD is not planned initially, stress labels enable elastic constant validation and make the dataset forward-compatible with NPT applications.
- **Stratify the candidate set by config_type from the start.** A dataset is only as good as the diversity of its structure families. Explicitly include: at least 3 surface terminations, at least 2 defect types, liquid/amorphous snapshots at the target temperature, and structures at ≥ 5 volumetric strains. A bulk-only dataset will always fail for non-bulk applications.
- **Apply FPS filtering before every DFT batch.** Generating 1,000 rattled structures and submitting all 1,000 to DFT is wasteful — 900 may be redundant. FPS in SOAP space typically reduces a 1,000-frame rattle set to 200–300 non-redundant candidates with negligible loss of coverage.
- **Filter unconverged SCF calculations before adding any frame to the dataset.** One unconverged frame can produce forces 10× larger than typical, shifting the loss function and corrupting gradients for hundreds of surrounding updates. Set NELM=200, EDIFF=1e-6, and filter strictly.
- **Use a dedicated held-out test set that is assembled before any model is trained.** Reserve 5–10% of every config_type as a clean test set. Never allow active learning queries to target test-set structures. Record the test-set file path and checksum in the manifest.
- **Record pseudopotential checksums at dataset creation time.** VASP POTCAR files are updated occasionally without changing the filename. A silent POTCAR swap mid-campaign can introduce a systematic energy offset of 0.01–0.1 eV/atom. Store the SHA-256 of each POTCAR in the manifest.
- **Document and report the dataset composition in publications.** Specify: total structure count, config_type distribution, DFT functional, basis set (ENCUT or ecutwfc), k-point density, smearing, and code version. This is the minimum for reproducibility of MLP training results.

## Integration with Other Skills

- **[dft-convergence](../dft-convergence/SKILL.md):** The canonical DFT settings frozen at campaign start must be established using that skill's convergence protocol for all species in the target chemical system. Never choose ENCUT or k-point density without a convergence test — and never change them once established.
- **[vasp-workflow](../vasp-workflow/SKILL.md):** Workflow 2 uses VASP for single-point labeling. The INCAR settings, POTCAR selection, and vasprun.xml parsing patterns from that skill apply directly. The ISIF=2 and PREC=Accurate settings there are required for accurate stress labels.
- **[quantum-espresso](../quantum-espresso/SKILL.md):** Workflow 3 uses QE for single-point labeling. The pseudopotential selection (SSSP vs. PseudoDojo), the critical degauss-in-Ry distinction, and ibrav=0 requirement from that skill all apply to dataset generation. ASE's QE interface handles unit conversion from Ry/bohr to eV/Å automatically but must be verified.
- **[mace-training](../../ml-interatomic-potentials/mace-training/SKILL.md):** This skill produces the extxyz files that feed directly into MACE training. The train/val/test splitting, isolated atom E0 protocol, and stress sign convention in Workflow 1 of that skill are downstream consumers of the dataset generated here.
- **[active-learning-mlp](../../ml-interatomic-potentials/active-learning-mlp/SKILL.md):** The DFT oracle policy (Workflow 4 in that skill) is a specialization of the labeling pipeline described here. The settings hash enforcement in `ActiveLearningDataset` and `DatasetManifest` here must be consistent — use the same hash at both levels.
- **[mlp-validation](../../ml-interatomic-potentials/mlp-validation/SKILL.md):** The held-out test set assembled during this skill's dataset generation phase is the primary input to that skill's static error analysis. The DFT settings used for validation reference calculations must be identical to the training data settings.
- **[mlp-to-lammps](../../ml-interatomic-potentials/mlp-to-lammps/SKILL.md):** When a LAMMPS production run reveals that the MLP fails at new conditions, the remedy is to generate new DFT data at those conditions using this skill's pipeline and retrain. The element ordering used in the LAMMPS data file must match the species ordering established in this skill's extxyz dataset.
- **[lammps-workflow](../../atomistic-md/lammps-workflow/SKILL.md):** LAMMPS AIMD snapshots (used as cheap DFT oracle candidates) are generated using the ML potential in LAMMPS. The input script mechanics and dump settings from that skill control the trajectory sampling that feeds this skill's FPS pipeline.
- **[md-equilibration](../../atomistic-md/md-equilibration/SKILL.md):** AIMD trajectories used for dataset generation must be properly equilibrated before subsampling. The equilibration diagnostics from that skill (temperature stability, density convergence, flying-ice-cube check) apply to any AIMD or MLMD run used as a source of training structures.
- **[reproducibility-workflow](../../scientific-data/reproducibility-workflow/SKILL.md):** The `DatasetManifest` class in Workflow 6 is a domain-specific implementation of the general provenance tracking described in that skill. For datasets intended for publication or long-term reuse, the full provenance schema described there should be adopted.
- **[materials-featurization](../../scientific-data/materials-featurization/SKILL.md):** FPS filtering in Workflow 1 uses SOAP descriptors to measure structural similarity. The descriptor parameterization (r_cut, n_max, l_max, species) described in that skill applies directly. The choice of descriptor affects which structures are considered duplicates and which are deemed informative.

## Key References

1. Deringer, V. L., et al. (2021). *Gaussian process regression for materials and molecules.* Chemical Reviews, 121(16), 10073–10141. https://doi.org/10.1021/acs.chemrev.1c00022
2. Unke, O. T., et al. (2021). *Machine learning force fields.* Chemical Reviews, 121(16), 10142–10186. https://doi.org/10.1021/acs.chemrev.0c01111
3. Cheng, B., et al. (2020). *Mapping materials and molecules.* Accounts of Chemical Research, 53(9), 1981–1991. https://doi.org/10.1021/acs.accounts.0c00403
4. Pickard, C. J., & Needs, R. J. (2011). *Ab initio random structure searching.* Journal of Physics: Condensed Matter, 23(5), 053201. https://doi.org/10.1088/0953-8984/23/5/053201
5. Musil, F., et al. (2021). *Physics-inspired structural representations for molecules and materials.* Chemical Reviews, 121(16), 9759–9815. https://doi.org/10.1021/acs.chemrev.1c00021
