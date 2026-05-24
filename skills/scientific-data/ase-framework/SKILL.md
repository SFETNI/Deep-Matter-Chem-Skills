---
name: ase-framework
domain: scientific-data
maturity: draft
tools:
  - ASE
  - pymatgen
  - Phonopy
  - LAMMPS
  - VASP
  - Quantum ESPRESSO
tested_versions:
  ASE: "3.23.0"
  pymatgen: "2024.1.26"
  Phonopy: "2.23.0"
  LAMMPS: "2Aug2023"
  VASP: "6.4.2"
  Quantum ESPRESSO: "7.2"
requires_gpu: false
estimated_runtime: minutes-to-hours
---

# ASE Framework

## Description

This skill covers the Atomic Simulation Environment (ASE) as a workflow hub for atomistic simulation: `Atoms` objects, calculators, constraints, optimizers, trajectories, file IO, structure manipulation, dataset generation, and conversion between DFT, MD, ML potentials, Phonopy, pymatgen, LAMMPS, and visualization tools. Invoke this skill when writing Python workflows that create, transform, calculate, relax, convert, or curate atomistic structures for computational materials science and chemistry.

## Domain Context

ASE is a Python framework built around a small set of abstractions: an `Atoms` object stores symbols, positions, cell, periodic boundary conditions, arrays, constraints, and metadata; a calculator attaches an energy/force/stress backend; optimizers and molecular dynamics drivers move atoms using calculator results; IO functions convert between simulation formats. This makes ASE useful as glue code across VASP, Quantum ESPRESSO, CP2K, LAMMPS, EMT, MACE, CHGNet, and many other engines.

The convenience comes with responsibility. ASE does not know whether a converted structure is scientifically valid, whether a calculator working directory was reused incorrectly, whether a CIF lost partial occupancies, or whether a LAMMPS data file preserved element ordering. Many ASE failures are silent: cell and PBC metadata are dropped by XYZ, constraints are not supported by a target format, or an extxyz file stores forces in the wrong key. The workflow must check these invariants explicitly.

ASE sits at the boundary between representation and calculation. The same atomic configuration can be represented as a molecule with no periodic cell, a slab with `pbc=[True, True, False]`, a fully periodic crystal, or a large supercell for phonons. The calculator interprets these choices physically: a DFT calculator with an unintended vacuum direction or missing PBC will compute a different problem, not just a differently formatted input.

For machine-learning potential work, ASE is often the canonical bridge between DFT outputs and training datasets. It reads VASP/QE/LAMMPS/XYZ trajectories, stores energies and forces in extxyz, applies constraints or filters, and evaluates MACE or CHGNet calculators during validation. This makes atom ordering, unit conventions, metadata, and provenance checks non-negotiable.

## When to Use This Skill

- Building scripted workflows that generate, relax, strain, rattle, or convert atomistic structures.
- Creating supercells, slabs, adsorbate configurations, NEB images, phonon displacement cells, or active-learning candidate structures.
- Reading and writing POSCAR, CIF, XYZ, extxyz, LAMMPS data, ASE trajectory, and calculator-specific inputs.
- Attaching calculators to `Atoms` objects for geometry optimization, single-point evaluation, force comparison, or trajectory generation.
- Converting structures and trajectories between ASE, pymatgen, Phonopy, LAMMPS, MACE, and visualization workflows.
- Preparing DFT or MLP datasets where energy, force, stress, config type, and provenance metadata must survive IO round trips.

## When Not to Use This Skill

- **As a replacement for code-specific convergence knowledge.** ASE can launch VASP or QE, but convergence settings, pseudopotential choice, k-point density, smearing, and force thresholds still belong to the DFT workflow skills.
- **When GUI inspection is the only goal.** Use visualization tools directly for one-off inspection; use ASE when a reproducible structure transformation or conversion is needed.
- **For unsupported chemistry encoded in lossy formats.** CIF disorder, partial occupancies, molecular bond orders, charges, and spin states may not survive conversion to `Atoms` without careful handling.
- **For high-throughput production without directory isolation.** ASE calculators write files to working directories. Reusing directories across structures can corrupt results unless each calculation is isolated and provenance-tracked.
- **When atom identity must persist but the format reorders atoms.** Some readers/writers sort atoms, expand symmetry, or drop tags. Do not use such a conversion path unless atom ordering is verified.

## Core Concepts

- **`Atoms` object:** ASE's central container for atomic symbols, Cartesian positions in Angstrom, cell vectors, PBC flags, per-atom arrays, constraints, tags, momenta, charges, and an optional calculator.
- **Calculator:** Backend object that computes energy, forces, stress, charges, or other properties for an `Atoms` object. Examples include `Vasp`, `Espresso`, `CP2K`, `EMT`, MACE calculators, CHGNet wrappers, and socket/file-based external calculators.
- **Cell and PBC:** `atoms.cell` and `atoms.pbc` define the simulation box and periodicity. These must be correct before neighbor lists, surface generation, DFT inputs, LAMMPS data files, or phonon supercells are created.
- **Constraints:** Objects such as `FixAtoms`, `FixBondLength`, and `FixCartesian` restrict degrees of freedom during optimization or MD. Constraints are not always preserved by file formats or respected by every calculator.
- **Optimizers:** Algorithms such as BFGS, LBFGS, FIRE, and MDMin update positions using calculator forces. Convergence depends on force thresholds, constraints, calculator noise, and whether cell degrees of freedom are included.
- **Filters:** ASE wrappers such as `UnitCellFilter` and `ExpCellFilter` expose cell degrees of freedom to optimizers. They are required for variable-cell relaxations, but can produce misleading results if stress units or external pressure are mishandled.
- **Trajectory:** ASE `.traj` files store sequences of `Atoms` objects with arrays and metadata. extxyz is more portable for MLP datasets, but the exact energy/force/stress keys must match downstream training tools.
- **Calculator state:** A calculator may cache results and file paths. Reusing a calculator or working directory across unrelated structures without resetting can return stale outputs or overwrite input files.
- **Atom ordering:** The order of atoms in an `Atoms` object is part of the data. Forces, constraints, species mappings, LAMMPS type IDs, and MLP labels all assume consistent ordering.
- **Round-trip test:** Read a file, write it to another format, read it back, and compare symbols, positions, cell, PBC, constraints, and metadata. This is the fastest way to catch silent conversion errors.

## Key Workflows

### Workflow 1: Safe structure IO and round-trip validation

Use explicit readers, preserve cell/PBC metadata, and validate critical invariants after every conversion. This is especially important before converting DFT outputs into extxyz datasets or LAMMPS input files.

```python
from pathlib import Path
import json
import numpy as np
from ase.io import read, write


def atoms_fingerprint(atoms) -> dict:
    """Return a compact representation of metadata that must survive conversion."""
    return {
        "n_atoms": len(atoms),
        "symbols": atoms.get_chemical_symbols(),
        "cell_A": np.asarray(atoms.cell).round(10).tolist(),
        "pbc": [bool(x) for x in atoms.pbc],
        "has_energy": "energy" in atoms.info,
        "has_forces": "forces" in atoms.arrays,
        "has_stress": "stress" in atoms.info or "stress" in atoms.arrays,
        "arrays": sorted(atoms.arrays.keys()),
        "info_keys": sorted(atoms.info.keys()),
    }


def validate_round_trip(
    input_file: str,
    output_file: str,
    input_format: str | None = None,
    output_format: str | None = None,
    atol_pos_A: float = 1e-8,
) -> None:
    """
    Convert a structure or trajectory and verify that key information survived.

    For production conversion, prefer explicit format names over extension guessing.
    XYZ is intentionally lossy for periodic systems unless using extxyz.
    """
    atoms = read(input_file, format=input_format)
    if isinstance(atoms, list):
        raise ValueError("Expected one structure; use validate_trajectory_round_trip for trajectories")

    before = atoms_fingerprint(atoms)
    write(output_file, atoms, format=output_format)
    reread = read(output_file, format=output_format)
    after = atoms_fingerprint(reread)

    if before["n_atoms"] != after["n_atoms"]:
        raise ValueError(f"Atom count changed: {before['n_atoms']} -> {after['n_atoms']}")
    if before["symbols"] != after["symbols"]:
        raise ValueError("Atom symbols or ordering changed during conversion")
    if not np.allclose(atoms.positions, reread.positions, atol=atol_pos_A):
        raise ValueError("Positions changed beyond tolerance during conversion")
    if not np.allclose(np.asarray(atoms.cell), np.asarray(reread.cell), atol=atol_pos_A):
        raise ValueError("Cell changed during conversion")
    if before["pbc"] != after["pbc"]:
        raise ValueError(f"PBC changed: {before['pbc']} -> {after['pbc']}")

    metadata_file = Path(output_file).with_suffix(Path(output_file).suffix + ".metadata.json")
    metadata_file.write_text(json.dumps({
        "input_file": input_file,
        "output_file": output_file,
        "before": before,
        "after": after,
    }, indent=2))
    print(f"Round-trip validation passed: {input_file} -> {output_file}")


def write_extxyz_dataset(
    frames: list,
    output_file: str,
    required_arrays: tuple[str, ...] = ("forces",),
    required_info: tuple[str, ...] = ("energy", "config_type"),
) -> None:
    """
    Write an MLP-ready extxyz dataset after checking labels and metadata.

    MACE and related tools expect consistent energy and force keys. Stress
    conventions vary by code and must be documented before training.
    """
    if not frames:
        raise ValueError("No frames to write")

    species_order = frames[0].get_chemical_symbols()
    for i, atoms in enumerate(frames):
        for key in required_arrays:
            if key not in atoms.arrays:
                raise ValueError(f"Frame {i} missing arrays['{key}']")
        for key in required_info:
            if key not in atoms.info:
                raise ValueError(f"Frame {i} missing info['{key}']")
        if len(atoms) != len(atoms.arrays["forces"]):
            raise ValueError(f"Frame {i} force array length does not match atom count")
        if atoms.pbc.any() and atoms.cell.volume <= 0:
            raise ValueError(f"Frame {i} has periodic PBC but invalid cell")
        if atoms.get_chemical_symbols() != species_order:
            print(f"WARNING frame {i}: atom ordering/species sequence differs from frame 0")

    write(output_file, frames, format="extxyz")
    print(f"Wrote {len(frames)} frames to {output_file}")
```

### Workflow 2: Calculator isolation and geometry optimization

Attach calculators in isolated directories, run force convergence checks, and avoid stale calculator state. The example uses EMT for demonstration of ASE mechanics; replace it with VASP, QE, CP2K, MACE, or CHGNet calculators in production.

```python
from pathlib import Path
import shutil
import numpy as np
from ase.build import bulk
from ase.calculators.emt import EMT
from ase.optimize import BFGS, FIRE
from ase.constraints import FixAtoms
from ase.io import write


def attach_calculator(atoms, calc_kind: str, workdir: str, **kwargs):
    """
    Attach a calculator with an explicit working directory.

    For file-writing DFT calculators, workdir must be unique per structure.
    This function uses EMT for local smoke tests and documents where
    production calculators should be configured.
    """
    Path(workdir).mkdir(parents=True, exist_ok=True)

    if calc_kind == "EMT":
        atoms.calc = EMT(**kwargs)
    elif calc_kind == "VASP":
        from ase.calculators.vasp import Vasp
        atoms.calc = Vasp(directory=workdir, **kwargs)
    elif calc_kind == "QE":
        from ase.calculators.espresso import Espresso
        atoms.calc = Espresso(directory=workdir, **kwargs)
    elif calc_kind == "CP2K":
        from ase.calculators.cp2k import CP2K
        atoms.calc = CP2K(directory=workdir, **kwargs)
    else:
        raise ValueError(f"Unknown calculator kind: {calc_kind}")
    return atoms


def relax_structure(
    atoms,
    output_prefix: str,
    calc_kind: str = "EMT",
    fmax_eV_A: float = 0.03,
    max_steps: int = 300,
    optimizer: str = "FIRE",
    fixed_indices: list[int] | None = None,
    calculator_kwargs: dict | None = None,
):
    """
    Relax atomic positions with optional fixed atoms.

    For DFT relaxations, fmax should match the target property. Phonons and
    elastic constants usually require tighter force convergence than screening
    relaxations. [EXPERT REVIEW NEEDED]
    """
    calculator_kwargs = calculator_kwargs or {}
    prefix = Path(output_prefix)
    workdir = prefix.with_suffix("").as_posix() + "_calc"
    traj_file = prefix.with_suffix(".traj")
    log_file = prefix.with_suffix(".opt.log")
    final_file = prefix.with_suffix(".final.extxyz")

    if fixed_indices:
        atoms.set_constraint(FixAtoms(indices=fixed_indices))

    attach_calculator(atoms, calc_kind, workdir, **calculator_kwargs)
    opt_cls = FIRE if optimizer.upper() == "FIRE" else BFGS
    opt = opt_cls(atoms, trajectory=str(traj_file), logfile=str(log_file))
    opt.run(fmax=fmax_eV_A, steps=max_steps)

    forces = atoms.get_forces()
    max_force = float(np.abs(forces).max())
    atoms.info["relax_fmax_eV_A"] = max_force
    atoms.info["relax_converged"] = bool(max_force <= fmax_eV_A)
    atoms.info["calculator_kind"] = calc_kind
    write(final_file, atoms, format="extxyz")

    print(f"Final max |F| = {max_force:.4f} eV/Angstrom")
    if max_force > fmax_eV_A:
        print("WARNING: relaxation did not reach requested fmax")
    return atoms


def calculator_reuse_smoke_test() -> None:
    """
    Minimal test showing why calculators should not be shared casually.
    Production workflows should create one calculator/workdir per calculation.
    """
    atoms1 = bulk("Cu", "fcc", a=3.6)
    atoms2 = bulk("Al", "fcc", a=4.05)
    atoms1.calc = EMT()
    e1 = atoms1.get_potential_energy()

    # Reusing EMT is harmless here, but file-writing calculators are different.
    atoms2.calc = atoms1.calc
    e2 = atoms2.get_potential_energy()
    print(f"EMT smoke test energies: Cu={e1:.6f} eV, Al={e2:.6f} eV")
```

### Workflow 3: Structure manipulation for supercells, slabs, adsorbates, strain, and NEB

ASE's builders are reliable for common structures, but every generated object should be inspected for PBC, vacuum, atom ordering, and constraints before calculation.

```python
from pathlib import Path
import numpy as np
from ase.build import bulk, surface, add_adsorbate, make_supercell
from ase.constraints import FixAtoms
from ase.io import write
from ase.mep import NEB


def build_surface_adsorbate_model(
    bulk_symbol: str = "Cu",
    miller: tuple[int, int, int] = (1, 1, 1),
    layers: int = 5,
    vacuum_A: float = 15.0,
    adsorbate: str = "O",
    height_A: float = 1.8,
    site: str = "ontop",
    freeze_bottom_layers: int = 2,
):
    """
    Build a slab + adsorbate model with bottom-layer constraints.

    The chosen adsorption site and height are starting guesses, not validated
    chemistry. Check relaxed geometry and alternative sites. [EXPERT REVIEW NEEDED]
    """
    slab = surface(bulk(bulk_symbol, "fcc", a=3.61), miller, layers)
    slab.center(vacuum=vacuum_A, axis=2)
    slab.pbc = [True, True, False]
    add_adsorbate(slab, adsorbate, height=height_A, position=site)

    z = slab.positions[:, 2]
    unique_z = np.unique(np.round(z, 3))
    bottom_cut = sorted(unique_z)[min(freeze_bottom_layers - 1, len(unique_z) - 1)]
    fixed = [i for i, zi in enumerate(z) if zi <= bottom_cut + 1e-3]
    slab.set_constraint(FixAtoms(indices=fixed))

    slab.info.update({
        "model_type": "surface_adsorbate",
        "miller": miller,
        "layers": layers,
        "vacuum_A": vacuum_A,
        "adsorbate": adsorbate,
        "fixed_atoms": len(fixed),
    })
    return slab


def generate_strained_structures(
    atoms,
    strains: list[float],
    mode: str = "isotropic",
) -> list:
    """
    Generate strained structures for EOS, elastic, or stress-response datasets.

    mode='isotropic' scales all cell vectors; mode='uniaxial-z' scales c only.
    Atomic fractional coordinates are preserved.
    """
    frames = []
    for eps in strains:
        a = atoms.copy()
        cell = a.cell.array.copy()
        if mode == "isotropic":
            new_cell = cell * (1.0 + eps)
        elif mode == "uniaxial-z":
            new_cell = cell.copy()
            new_cell[2] *= (1.0 + eps)
        else:
            raise ValueError(f"Unknown strain mode: {mode}")
        a.set_cell(new_cell, scale_atoms=True)
        a.info["strain"] = eps
        a.info["strain_mode"] = mode
        frames.append(a)
    return frames


def prepare_neb_images(
    initial,
    final,
    n_images: int,
    output_dir: str,
) -> list:
    """
    Prepare linearly interpolated NEB images.

    Atom order must be identical in initial and final structures. For reactions
    involving diffusion or exchange, manually map atoms before interpolation.
    """
    if initial.get_chemical_symbols() != final.get_chemical_symbols():
        raise ValueError("Initial/final atom symbols or ordering differ")
    if not np.allclose(initial.cell.array, final.cell.array, atol=1e-6):
        raise ValueError("Initial/final cells differ; use a consistent cell for NEB")

    images = [initial.copy()]
    images += [initial.copy() for _ in range(n_images)]
    images += [final.copy()]
    neb = NEB(images)
    neb.interpolate(method="idpp")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for i, img in enumerate(images):
        img.info["neb_image"] = i
        write(out / f"image_{i:02d}.extxyz", img, format="extxyz")
    return images
```

### Workflow 4: Trajectory handling and DFT/MLP dataset generation

ASE trajectories and extxyz files are common interchange formats for active learning, MACE training, and MLP validation. The important step is to preserve labels and provenance, not merely coordinates.

```python
from pathlib import Path
import hashlib
import json
import numpy as np
from ase.io import read, write


def sha256_file(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def collect_labeled_frames(
    input_patterns: list[str],
    output_extxyz: str,
    config_type: str,
    stride: int = 1,
    max_force_eV_A: float | None = None,
) -> list:
    """
    Collect frames from DFT or MD outputs into an extxyz dataset.

    input_patterns should be explicit file paths from a manifest in production.
    Globs are acceptable only for exploratory collection because their order can
    change as new calculations are added.
    """
    frames = []
    manifest = []
    for path in input_patterns:
        loaded = read(path, index=f"::{stride}")
        if not isinstance(loaded, list):
            loaded = [loaded]
        for local_i, atoms in enumerate(loaded):
            atoms.info.setdefault("config_type", config_type)
            atoms.info["source_file"] = path
            atoms.info["source_frame"] = local_i * stride

            if max_force_eV_A is not None and "forces" in atoms.arrays:
                if np.abs(atoms.arrays["forces"]).max() > max_force_eV_A:
                    continue
            frames.append(atoms)
        manifest.append({"path": path, "sha256": sha256_file(path), "n_read": len(loaded)})

    write_extxyz_dataset(frames, output_extxyz)
    Path(output_extxyz + ".manifest.json").write_text(json.dumps({
        "output": output_extxyz,
        "output_sha256": sha256_file(output_extxyz),
        "config_type": config_type,
        "stride": stride,
        "n_frames": len(frames),
        "inputs": manifest,
    }, indent=2))
    return frames


def split_extxyz_by_config_type(
    input_extxyz: str,
    output_dir: str,
    train_fraction: float = 0.8,
    val_fraction: float = 0.1,
    seed: int = 42,
) -> dict:
    """
    Deterministically split an extxyz dataset within each config_type group.

    This is a mechanical split helper. For final model assessment, use
    composition/prototype/descriptor-aware splits where appropriate.
    """
    frames = read(input_extxyz, index=":")
    rng = np.random.default_rng(seed)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    groups: dict[str, list] = {}
    for atoms in frames:
        groups.setdefault(atoms.info.get("config_type", "unknown"), []).append(atoms)

    splits = {"train": [], "val": [], "test": []}
    for ctype, group in groups.items():
        order = rng.permutation(len(group))
        n = len(group)
        n_train = int(round(train_fraction * n))
        n_val = int(round(val_fraction * n))
        for idx in order[:n_train]:
            splits["train"].append(group[int(idx)])
        for idx in order[n_train:n_train + n_val]:
            splits["val"].append(group[int(idx)])
        for idx in order[n_train + n_val:]:
            splits["test"].append(group[int(idx)])
        print(f"{ctype}: {n_train} train, {n_val} val, {n - n_train - n_val} test")

    outputs = {}
    for split, split_frames in splits.items():
        path = out / f"{split}.xyz"
        write(path, split_frames, format="extxyz")
        outputs[split] = {"path": str(path), "n_frames": len(split_frames), "sha256": sha256_file(path)}

    (out / "split_manifest.json").write_text(json.dumps({
        "input": input_extxyz,
        "input_sha256": sha256_file(input_extxyz),
        "seed": seed,
        "train_fraction": train_fraction,
        "val_fraction": val_fraction,
        "outputs": outputs,
    }, indent=2))
    return outputs
```

### Workflow 5: Interoperability with pymatgen, Phonopy, and LAMMPS

Use ASE as a bridge, but verify the assumptions that each package makes about cells, species, units, and ordering.

```python
import numpy as np
from ase.io import read, write
from pymatgen.io.ase import AseAtomsAdaptor


def ase_to_pymatgen_structure(atoms):
    """
    Convert ASE Atoms to pymatgen Structure or Molecule depending on PBC.

    pymatgen Structure requires a valid periodic cell. Molecules should not be
    forced into periodic structures unless a real simulation cell is intended.
    """
    adaptor = AseAtomsAdaptor()
    if atoms.pbc.any():
        if atoms.cell.volume <= 0:
            raise ValueError("Periodic ASE Atoms has invalid cell")
        return adaptor.get_structure(atoms)
    return adaptor.get_molecule(atoms)


def write_lammps_data_with_species_check(
    atoms,
    output_data: str,
    species_order: list[str],
    atom_style: str = "atomic",
) -> None:
    """
    Write a LAMMPS data file and record the intended species/type mapping.

    LAMMPS pair_coeff commands must use the same species_order. A mismatch here
    silently applies the wrong potential to the wrong element.
    """
    symbols = set(atoms.get_chemical_symbols())
    missing = symbols - set(species_order)
    if missing:
        raise ValueError(f"species_order missing symbols: {sorted(missing)}")
    atoms.info["lammps_species_order"] = " ".join(species_order)
    write(output_data, atoms, format="lammps-data", atom_style=atom_style, specorder=species_order)
    print(f"Wrote {output_data} with type order: {' '.join(species_order)}")


def prepare_phonopy_supercell_inputs(
    primitive_file: str,
    supercell_matrix: list[list[int]],
    output_prefix: str,
) -> None:
    """
    Convert an ASE-readable primitive cell into Phonopy displacement inputs.

    This function assumes Phonopy is installed. Supercell size must be converged
    for the material and property; small cells can create false imaginary modes.
    """
    from phonopy import Phonopy
    from phonopy.structure.atoms import PhonopyAtoms
    from ase.io import write

    atoms = read(primitive_file)
    ph_atoms = PhonopyAtoms(
        symbols=atoms.get_chemical_symbols(),
        cell=atoms.cell.array,
        scaled_positions=atoms.get_scaled_positions(),
    )
    phonon = Phonopy(ph_atoms, supercell_matrix=np.array(supercell_matrix))
    phonon.generate_displacements(distance=0.01)

    for i, scell in enumerate(phonon.supercells_with_displacements):
        ase_scell = scell.to_ase_atoms()
        ase_scell.info["phonopy_displacement_index"] = i
        write(f"{output_prefix}_{i:03d}.extxyz", ase_scell, format="extxyz")
    print(f"Wrote {len(phonon.supercells_with_displacements)} displaced supercells")
```

## Common Failure Modes

| Failure | Symptom | Likely Cause | Diagnostic | Fix |
|---------|---------|--------------|------------|-----|
| Lost cell/PBC metadata | Periodic crystal becomes a molecule; neighbor lists, RDFs, DFT inputs, or SOAP descriptors are wrong | Structure was written through plain XYZ or another lossy format | Print `atoms.cell`, `atoms.pbc`, and volume after reading; run a round-trip test | Use CIF, POSCAR, ASE `.traj`, or extxyz for periodic systems; assert PBC and cell before calculation |
| Wrong units after conversion | Forces, stress, timestep, or cell values differ by factors of 10, 1000, or 160.2 | Mixed Angstrom/nm, eV/Ry/Hartree, bar/GPa/eV/A^3, or LAMMPS real/metal units | Check one known bond length, energy, and force manually after read/write | Convert explicitly and encode units in key names; keep original code metadata in `atoms.info` |
| Calculator state reuse | Energies or forces repeat from a previous structure; files are overwritten; calculation reads stale outputs | Same calculator object or working directory reused across structures | Inspect calculator directory timestamps and input files; disable cache and rerun one structure | Create a fresh calculator and unique directory per calculation; clean or archive directories between runs |
| Overwritten working directories | High-throughput run leaves only the last structure's input/output files | All ASE calculators wrote into the same `directory` | Check for repeated `directory=` values in workflow logs | Use stable per-structure directories, e.g. hash or index based names; record them in a manifest |
| Broken constraints | Slab bottom layer moves; NEB endpoints relax; intended fixed Cartesian direction is unconstrained | Constraints not attached, not supported by the calculator, or lost in file conversion | Print `atoms.constraints`; displace constrained atoms and verify optimizer response | Attach constraints after reading; store constraint definitions in metadata; verify before optimization |
| Inconsistent atom ordering | Forces no longer align with atoms; LAMMPS species mapping wrong; NEB interpolation swaps atoms | Reader/writer sorted atoms or initial/final structures differ in order | Compare symbols and nearest-neighbor mapping before and after conversion | Preserve ordering; write mapping files; use explicit `specorder` for LAMMPS; manually map NEB endpoints |
| Non-deterministic IO ordering | Dataset split or manifest changes between runs with same inputs | Glob order, filesystem order, or dictionary iteration used implicitly | Compare manifests from two runs; check whether inputs are sorted | Sort all file lists explicitly; record input paths and hashes |
| Silent format conversion error | File writes successfully but downstream code reads wrong charges, masses, tags, or cell | Target format cannot represent all ASE metadata | Round-trip through the exact target reader and compare metadata | Use a richer format or write sidecar JSON for unsupported fields |
| Stress convention mismatch | Variable-cell relaxation moves in the wrong direction or MLP stress validation appears biased | Sign or unit convention differs between VASP, ASE, extxyz, and ML training code | Compare stress for one strained structure against finite differences | Document stress convention; convert before writing extxyz; test with a known strained cell |
| Bad supercell or surface construction | Artificial interactions across vacuum; phonon soft modes; slab dipole artifacts | Too little vacuum, wrong PBC, or unconverged supercell size | Visualize the structure; measure minimum image distances and vacuum thickness | Increase vacuum/supercell; set PBC intentionally; converge size before production |

## Best Practices

- **Assert cell and PBC before every calculation.** Do not rely on file extension or reader defaults. A single missing `pbc=True` can change the physical problem.
- **Use extxyz or ASE `.traj` for labeled datasets.** Plain XYZ is usually too lossy for MLP workflows because it drops cell, PBC, energy, force, stress, and custom metadata unless extended conventions are used.
- **Create one calculator working directory per structure.** This prevents stale files and makes failed calculations auditable.
- **Record species order whenever writing LAMMPS data or MLP inputs.** The element list is part of the model input contract.
- **Run round-trip tests on every new conversion route.** A conversion is not safe just because the writer succeeds.
- **Keep constraints close to the workflow code.** File formats often do not preserve constraints, so attach and verify them immediately before optimization.
- **Do not silently drop failed frames when generating datasets.** If frames are filtered by force threshold, convergence status, or missing labels, write a manifest explaining what was removed.
- **Use deterministic file ordering and random seeds.** Dataset generation and splitting should produce identical manifests on rerun.
- **Check calculator outputs before trusting optimizer convergence.** An optimizer can converge to noisy, unconverged, or stale forces if the calculator is misconfigured.
- **Store provenance in `atoms.info` and sidecar manifests.** Use both: `atoms.info` travels with extxyz frames, while sidecars preserve run-level hashes and settings.

## Integration with Other Skills

- **[vasp-workflow](../../electronic-structure/vasp-workflow/SKILL.md):** ASE can prepare POSCAR files, launch VASP calculators, parse outputs, and convert relaxed structures into extxyz datasets. VASP-specific convergence and POTCAR settings remain governed by that skill.
- **[quantum-espresso](../../electronic-structure/quantum-espresso/SKILL.md):** ASE's `Espresso` calculator and IO tools can generate QE inputs and parse results. QE-specific units, pseudopotential families, and smearing choices must follow that workflow.
- **[dft-convergence](../../electronic-structure/dft-convergence/SKILL.md):** ASE-generated structures and calculator inputs should use convergence parameters established by that skill before production data generation.
- **[dft-dataset-generation](../../electronic-structure/dft-dataset-generation/SKILL.md):** ASE is the practical layer for assembling labeled frames, preserving energies/forces/stresses, and writing extxyz datasets for MLP training.
- **[dft-phonons](../../electronic-structure/dft-phonons/SKILL.md):** ASE can generate supercells, interface with Phonopy, and write displaced structures for DFT force calculations. This is an intentional forward reference until that skill is added.
- **[lammps-workflow](../../atomistic-md/lammps-workflow/SKILL.md):** ASE writes LAMMPS data files and reads LAMMPS trajectories, but atom type ordering and unit style must be checked against the LAMMPS input script.
- **[mace-training](../../ml-interatomic-potentials/mace-training/SKILL.md):** MACE training datasets are commonly stored as extxyz files generated or validated with ASE. Energy, force, stress, and config-type keys must match the MACE command line.
- **[mlp-validation](../../ml-interatomic-potentials/mlp-validation/SKILL.md):** ASE calculators are used to evaluate MLP forces, compare against DFT labels, relax structures, and run short NVE checks before deployment.
- **[scientific-visualization](../scientific-visualization/SKILL.md):** ASE provides quick conversion and structure preparation for OVITO, VESTA, nglview, VMD, and matplotlib-based reporting.
- **[materials-featurization](../materials-featurization/SKILL.md):** ASE `Atoms` objects are direct inputs to SOAP, DScribe, and trajectory featurization pipelines. PBC and cell metadata must be correct before descriptors are computed.
- **[reproducibility-workflow](../reproducibility-workflow/SKILL.md):** ASE workflows should write manifests, hashes, seeds, calculator settings, and environment metadata so structure generation and conversion can be reproduced.
- **[pymatgen-analysis](../pymatgen-analysis/SKILL.md):** pymatgen complements ASE for symmetry analysis, phase diagrams, and electronic-structure postprocessing. This is an intentional forward reference until that skill is added.

## Key References

1. Larsen, A. H., et al. (2017). *The atomic simulation environment - a Python library for working with atoms.* Journal of Physics: Condensed Matter, 29(27), 273002. https://doi.org/10.1088/1361-648X/aa680e
2. Hjorth Larsen, A., et al. (2017). *The atomic simulation environment - a Python library for working with atoms.* https://wiki.fysik.dtu.dk/ase/
3. Ong, S. P., et al. (2013). *Python Materials Genomics (pymatgen): A robust, open-source python library for materials analysis.* Computational Materials Science, 68, 314-319. https://doi.org/10.1016/j.commatsci.2012.10.028
4. Togo, A., & Tanaka, I. (2015). *First principles phonon calculations in materials science.* Scripta Materialia, 108, 1-5. https://doi.org/10.1016/j.scriptamat.2015.07.021
5. Thompson, A. P., et al. (2022). *LAMMPS - a flexible simulation tool for particle-based materials modeling at the atomic, meso, and continuum scales.* Computer Physics Communications, 271, 108171. https://doi.org/10.1016/j.cpc.2021.108171
