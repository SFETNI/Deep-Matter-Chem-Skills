---
name: pymatgen-analysis
domain: scientific-data
maturity: draft
tools:
  - pymatgen
  - monty
  - spglib
  - numpy
  - pandas
  - matplotlib
tested_versions:
  pymatgen: "2024.11.13"
  monty: "2024.7.30"
  spglib: "2.5.0"
  numpy: "1.26.4"
  pandas: "2.2.0"
  matplotlib: "3.8.0"
requires_gpu: false
estimated_runtime: minutes-to-hours
---

# Pymatgen Analysis

## Description

This skill covers pymatgen as a materials analysis and structure-processing framework: `Structure`, `Molecule`, `Lattice`, `Composition`, `Element`, `Species`, and `Site` objects; CIF/POSCAR/VASP output parsing; symmetry analysis; cell standardization; supercell, slab, defect, and substitution preparation; phase diagrams and formation energies; and interoperability with ASE, matminer, MP API, Phonopy, and visualization workflows. Invoke this skill when a workflow needs robust materials-domain parsing, symmetry-aware structure manipulation, VASP input/output analysis, or thermodynamic construction from computed energies.

## Domain Context

pymatgen is a domain library for representing and manipulating materials data. Its core abstractions are chemically aware: a `Composition` knows oxidation states and reduced formulas; a `Structure` stores periodic sites in fractional coordinates with a `Lattice`; a `Species` can encode formal charge and spin; a `Site` can carry properties such as magnetic moment, selective dynamics, labels, or disorder. This is different from a generic coordinate container: pymatgen operations often use crystallographic assumptions, oxidation-state heuristics, and symmetry reductions.

The main strength of pymatgen is that it connects structure representation to analysis. The same `Structure` object can be standardized by spglib, converted to a VASP POSCAR, transformed into a slab, decorated with oxidation states, passed to matminer, converted to ASE, inserted into a phase diagram, or compared against entries from a database. This makes pymatgen a natural support layer for DFT workflows, dataset generation, visualization, and reproducibility.

The main risk is silent structure change. Symmetry analysis can reduce a distorted structure to a higher-symmetry prototype if tolerances are too loose. CIF parsing can expand partial occupancies or disorder in ways that are inappropriate for DFT. Conventional-cell choices differ between crystallographic, Setyawan-Curtarolo, and code-specific conventions. Site properties can be dropped during conversion. A structure that looks "standardized" may no longer be the structure actually computed.

Thermodynamic analysis has a separate reference-state risk. Formation energies, convex hulls, and Pourbaix diagrams are only meaningful when energies share compatible functionals, corrections, reference states, and units. Mixing uncorrected local DFT energies with database-corrected entries can produce phase diagrams that look polished but are physically inconsistent.

## When to Use This Skill

- Reading, writing, validating, and standardizing crystal structures from CIF, POSCAR, vasprun.xml, OUTCAR, CHGCAR, LOCPOT, JSON, or database records.
- Performing symmetry analysis, primitive/conventional cell conversion, equivalent-site identification, and high-symmetry path generation.
- Preparing supercells, substitutions, ordered approximants, slabs, interfaces, and defect precursor structures before DFT calculation.
- Parsing VASP outputs for energies, forces, stresses, band structures, DOS, charge densities, dielectric tensors, Born charges, and metadata.
- Constructing phase diagrams, convex hulls, formation energies, chemical-potential limits, and Pourbaix diagrams from compatible entries.
- Interoperating between pymatgen, ASE, matminer, Phonopy, MP API, and visualization workflows.
- Auditing inherited structures or datasets for oxidation-state ambiguity, disorder, wrong cell convention, lost site properties, or incompatible energy references.

## When Not to Use This Skill

- **As a substitute for physical convergence testing.** pymatgen can parse and organize results, but it does not validate ENCUT, k-points, smearing, force thresholds, or pseudopotential choices.
- **When disordered materials require statistical treatment.** A single ordered approximation generated from partial occupancies is not a thermodynamic treatment of disorder.
- **When molecular topology or bond order is primary.** pymatgen can represent molecules, but RDKit/OpenBabel are more appropriate for cheminformatics where connectivity, stereochemistry, and bond order drive the analysis.
- **When direct trajectory analysis is needed.** pymatgen can read structures and some outputs, but MD trajectory observables such as RDF, MSD, and VACF belong in trajectory-analysis tools.
- **For phase diagrams built from mixed references.** Do not combine database-corrected entries, raw local calculations, different functionals, and inconsistent Hubbard U settings without an explicit correction scheme.

## Core Concepts

- **`Structure`:** Periodic crystal object containing a `Lattice` and a list of periodic sites. Positions are stored naturally in fractional coordinates; Cartesian coordinates are derived from the lattice.
- **`Molecule`:** Non-periodic collection of sites. It has no lattice and should not be forced into a `Structure` unless a simulation cell is physically intended.
- **`Lattice`:** Three cell vectors plus derived lengths, angles, reciprocal lattice, volume, and coordinate transforms. Incorrect lattice metadata corrupts all periodic analysis.
- **`Composition`:** Chemical formula object that supports reduced formulas, fractional compositions, anonymized formulas, oxidation-state guesses, and elemental-property queries.
- **`Element` and `Species`:** `Element` stores element-level properties; `Species` can include oxidation state and spin. Featurization and charge-balance checks often depend on using the right one.
- **`Site` and `PeriodicSite`:** Atomic positions plus species and optional properties. Site properties such as `magmom`, `selective_dynamics`, `charge`, and labels can be lost during conversion if not checked.
- **Symmetry tolerance:** `symprec` and `angle_tolerance` determine whether spglib treats distorted sites as symmetry-equivalent. Loose tolerances can over-reduce structures; tight tolerances can miss intended symmetry.
- **Primitive vs. conventional cell:** Primitive cells minimize atom count; conventional cells expose crystallographic symmetry. The right choice depends on the downstream task.
- **Structure transformation:** A deterministic operation on a structure, such as substitution, supercell generation, oxidation-state decoration, slab generation, or disorder ordering. Transformations should be recorded as provenance.
- **Computed entry:** A composition plus energy and correction metadata used for phase diagrams. Formation energies are meaningful only when entries share compatible references.

## Key Workflows

### Workflow 1: Safe structure parsing, standardization, and round-trip checks

Use explicit parsing and verify that standardization did not silently change the physical system. This is the first step before DFT input generation, featurization, slab construction, or database comparison.

```python
from pathlib import Path
import json
import numpy as np
from pymatgen.core import Structure, Molecule
from pymatgen.io.vasp import Poscar
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer


def structure_fingerprint(structure: Structure) -> dict:
    """Capture invariants that should be checked before and after transformations."""
    return {
        "formula": structure.composition.reduced_formula,
        "n_sites": len(structure),
        "volume_A3": float(structure.volume),
        "lattice_abc": [float(x) for x in structure.lattice.abc],
        "lattice_angles": [float(x) for x in structure.lattice.angles],
        "species": [str(site.specie) for site in structure],
        "site_property_keys": sorted(structure.site_properties.keys()),
        "charge": structure.charge,
    }


def load_and_standardize_structure(
    input_file: str,
    output_dir: str,
    symprec: float = 1e-3,
    angle_tolerance: float = 5.0,
) -> dict:
    """
    Read a structure, write primitive and conventional standardized cells,
    and record what changed.

    symprec is material- and workflow-dependent. Distorted finite-temperature
    or defect structures should generally not be standardized with loose
    tolerances. [EXPERT REVIEW NEEDED]
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    structure = Structure.from_file(input_file)
    before = structure_fingerprint(structure)

    sga = SpacegroupAnalyzer(
        structure,
        symprec=symprec,
        angle_tolerance=angle_tolerance,
    )
    primitive = sga.get_primitive_standard_structure()
    conventional = sga.get_conventional_standard_structure()

    primitive.to(filename=str(out / "primitive.cif"))
    conventional.to(filename=str(out / "conventional.cif"))
    Poscar(primitive).write_file(out / "POSCAR.primitive")
    Poscar(conventional).write_file(out / "POSCAR.conventional")

    report = {
        "input_file": input_file,
        "symprec": symprec,
        "angle_tolerance": angle_tolerance,
        "space_group_symbol": sga.get_space_group_symbol(),
        "space_group_number": sga.get_space_group_number(),
        "before": before,
        "primitive": structure_fingerprint(primitive),
        "conventional": structure_fingerprint(conventional),
    }
    (out / "standardization_report.json").write_text(json.dumps(report, indent=2))

    if before["formula"] != report["primitive"]["formula"]:
        print("WARNING: formula changed during primitive standardization")
    if before["site_property_keys"] and not report["primitive"]["site_property_keys"]:
        print("WARNING: site properties were not preserved in standardized structure")
    print(f"Space group: {report['space_group_symbol']} ({report['space_group_number']})")
    print(f"Original sites: {before['n_sites']}; primitive sites: {len(primitive)}")
    return report


def compare_round_trip_cif_poscar(structure: Structure, output_prefix: str) -> None:
    """Write/read through CIF and POSCAR and check structural drift."""
    prefix = Path(output_prefix)
    cif = prefix.with_suffix(".cif")
    poscar = prefix.with_suffix(".vasp")
    structure.to(filename=str(cif))
    Poscar(structure).write_file(poscar)

    from_cif = Structure.from_file(cif)
    from_poscar = Poscar.from_file(poscar).structure

    for label, reread in [("cif", from_cif), ("poscar", from_poscar)]:
        if structure.composition != reread.composition:
            raise ValueError(f"{label} round trip changed composition")
        if not np.allclose(structure.lattice.matrix, reread.lattice.matrix, atol=1e-6):
            raise ValueError(f"{label} round trip changed lattice")
        if sorted(structure.site_properties) != sorted(reread.site_properties):
            print(f"WARNING: {label} round trip changed site properties")
```

### Workflow 2: VASP output parsing and dataset extraction

pymatgen's VASP parsers are robust, but a dataset workflow must explicitly choose which calculation state to trust and preserve settings provenance.

```python
from pathlib import Path
import json
import pandas as pd
from pymatgen.io.vasp.outputs import Vasprun, Outcar, Chgcar, Locpot
from pymatgen.io.vasp.inputs import Incar, Kpoints, Poscar


def parse_vasp_run_summary(
    calc_dir: str,
    parse_dos: bool = False,
    parse_eigen: bool = False,
) -> dict:
    """
    Parse a completed VASP directory into a compact machine-readable summary.

    The final energy is only meaningful if the run converged electronically
    and, for relaxations, ionically. Always check both flags.
    """
    d = Path(calc_dir)
    vasprun = Vasprun(
        d / "vasprun.xml",
        parse_dos=parse_dos,
        parse_eigen=parse_eigen,
        exception_on_bad_xml=False,
    )
    outcar = Outcar(d / "OUTCAR") if (d / "OUTCAR").exists() else None
    incar = Incar.from_file(d / "INCAR") if (d / "INCAR").exists() else {}
    kpoints = Kpoints.from_file(d / "KPOINTS") if (d / "KPOINTS").exists() else None
    poscar = Poscar.from_file(d / "POSCAR") if (d / "POSCAR").exists() else None

    final_structure = vasprun.final_structure
    summary = {
        "calc_dir": str(d),
        "formula": final_structure.composition.reduced_formula,
        "n_sites": len(final_structure),
        "final_energy_eV": float(vasprun.final_energy),
        "final_energy_per_atom_eV": float(vasprun.final_energy / len(final_structure)),
        "converged": bool(vasprun.converged),
        "converged_electronic": bool(vasprun.converged_electronic),
        "converged_ionic": bool(vasprun.converged_ionic),
        "functional": str(vasprun.run_type),
        "potcar_symbols": list(getattr(vasprun, "potcar_symbols", [])),
        "incar": dict(incar),
        "kpoints_style": str(kpoints.style) if kpoints else None,
        "kpoints": kpoints.kpts if kpoints else None,
        "input_formula": poscar.structure.composition.reduced_formula if poscar else None,
    }
    if outcar and getattr(outcar, "magnetization", None):
        summary["magnetization"] = outcar.magnetization
    if not summary["converged"]:
        print(f"WARNING: {calc_dir} did not fully converge")

    (d / "pymatgen_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary


def collect_vasp_summaries(calc_dirs: list[str], output_csv: str) -> pd.DataFrame:
    """Collect VASP summaries into a table for screening or dataset manifests."""
    rows = []
    for calc_dir in calc_dirs:
        try:
            rows.append(parse_vasp_run_summary(calc_dir))
        except Exception as exc:
            rows.append({"calc_dir": calc_dir, "error": repr(exc), "converged": False})
    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"Wrote {len(df)} VASP summaries to {output_csv}")
    return df


def parse_volumetric_outputs(calc_dir: str) -> dict:
    """
    Parse CHGCAR and LOCPOT metadata without exporting huge grids.

    Use this to verify that charge density or electrostatic potential outputs
    match the structure before visualization or Bader/planar-average analysis.
    """
    d = Path(calc_dir)
    result = {}
    if (d / "CHGCAR").exists():
        chgcar = Chgcar.from_file(d / "CHGCAR")
        result["chgcar_formula"] = chgcar.structure.composition.reduced_formula
        result["chgcar_grid"] = list(chgcar.data["total"].shape)
    if (d / "LOCPOT").exists():
        locpot = Locpot.from_file(d / "LOCPOT")
        result["locpot_formula"] = locpot.structure.composition.reduced_formula
        result["locpot_grid"] = list(locpot.data["total"].shape)
    return result
```

### Workflow 3: Symmetry-aware supercells, substitutions, and defect precursor structures

Use pymatgen transformations to generate controlled structure variants, then record the transformation sequence. This does not replace defect thermodynamics; it prepares candidate structures for downstream DFT workflows.

```python
from pathlib import Path
import json
from pymatgen.core import Structure
from pymatgen.transformations.standard_transformations import (
    SupercellTransformation,
    SubstitutionTransformation,
    OrderDisorderedStructureTransformation,
)
from pymatgen.analysis.structure_matcher import StructureMatcher


def make_ordered_substitution_supercell(
    input_structure: str,
    output_dir: str,
    supercell_matrix: list[list[int]],
    substitution: dict,
) -> Structure:
    """
    Create a substituted supercell and write provenance.

    substitution example:
        {"Ti": "Zr"} replaces Ti with Zr.
        {"Li+": {"Na+": 0.5, "Vacancy": 0.5}} creates disorder that must be ordered.

    Ordered approximants from disordered structures are not unique and require
    convergence over configurations for thermodynamic claims. [EXPERT REVIEW NEEDED]
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    s = Structure.from_file(input_structure)

    t_super = SupercellTransformation(supercell_matrix)
    s_super = t_super.apply_transformation(s)

    t_sub = SubstitutionTransformation(substitution)
    s_sub = t_sub.apply_transformation(s_super)

    if not s_sub.is_ordered:
        orderer = OrderDisorderedStructureTransformation()
        ordered = orderer.apply_transformation(s_sub, return_ranked_list=5)
        s_final = ordered[0]["structure"]
        (out / "ordering_candidates.json").write_text(json.dumps(
            [{"energy": float(x.get("energy", 0.0)), "n_sites": len(x["structure"])}
             for x in ordered],
            indent=2,
        ))
    else:
        s_final = s_sub

    s_final.to(filename=str(out / "substituted_structure.cif"))
    s_final.to(filename=str(out / "POSCAR"))
    (out / "transformation_provenance.json").write_text(json.dumps({
        "input_structure": input_structure,
        "supercell_matrix": supercell_matrix,
        "substitution": substitution,
        "input_formula": s.composition.reduced_formula,
        "output_formula": s_final.composition.reduced_formula,
        "n_input_sites": len(s),
        "n_output_sites": len(s_final),
        "is_ordered": s_final.is_ordered,
    }, indent=2, default=str))
    return s_final


def remove_symmetry_duplicate_structures(
    structures: list[Structure],
    ltol: float = 0.2,
    stol: float = 0.3,
    angle_tol: float = 5.0,
) -> list[Structure]:
    """Deduplicate generated structures with StructureMatcher."""
    matcher = StructureMatcher(ltol=ltol, stol=stol, angle_tol=angle_tol)
    unique = []
    for s in structures:
        if not any(matcher.fit(s, u) for u in unique):
            unique.append(s)
    print(f"Deduplicated {len(structures)} structures -> {len(unique)} unique")
    return unique
```

### Workflow 4: Surface and slab construction at the structure-preparation level

Generate slabs with explicit orientation, vacuum, thickness, and metadata. Adsorption energetics, dipole corrections, and convergence are handled by surface DFT workflows; this workflow prepares reproducible input structures.

```python
from pathlib import Path
import json
from pymatgen.core import Structure, Molecule
from pymatgen.core.surface import SlabGenerator, generate_all_slabs
from pymatgen.io.vasp import Poscar


def generate_low_index_slabs(
    bulk_structure_file: str,
    output_dir: str,
    max_index: int = 1,
    min_slab_size_A: float = 12.0,
    min_vacuum_size_A: float = 15.0,
    center_slab: bool = True,
) -> list:
    """
    Generate low-index slabs and write POSCARs.

    Slab energies require convergence over slab thickness, vacuum, termination,
    k-mesh, dipole correction, and relaxation protocol. This function only
    prepares candidate geometries.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    bulk = Structure.from_file(bulk_structure_file)
    slabs = generate_all_slabs(
        bulk,
        max_index=max_index,
        min_slab_size=min_slab_size_A,
        min_vacuum_size=min_vacuum_size_A,
        center_slab=center_slab,
        in_unit_planes=True,
    )

    manifest = []
    for i, slab in enumerate(slabs):
        miller = tuple(int(x) for x in slab.miller_index)
        label = f"slab_{i:03d}_{miller[0]}{miller[1]}{miller[2]}"
        slab_dir = out / label
        slab_dir.mkdir(exist_ok=True)
        Poscar(slab).write_file(slab_dir / "POSCAR")
        slab.to(filename=str(slab_dir / "slab.cif"))
        manifest.append({
            "label": label,
            "miller_index": miller,
            "n_sites": len(slab),
            "formula": slab.composition.reduced_formula,
            "surface_area_A2": float(slab.surface_area),
            "oriented_unit_cell_sites": len(slab.oriented_unit_cell),
        })

    (out / "slab_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Generated {len(slabs)} slabs in {output_dir}")
    return slabs


def add_molecular_adsorbate_to_slab(
    slab_file: str,
    adsorbate_xyz: str,
    output_poscar: str,
    height_A: float,
    lateral_frac_coords: tuple[float, float],
) -> Structure:
    """
    Place a molecule above a slab using fractional lateral coordinates.

    This is a deterministic starting-geometry generator, not a site-search
    algorithm. Multiple adsorption sites and orientations must be sampled.
    [EXPERT REVIEW NEEDED]
    """
    slab = Structure.from_file(slab_file)
    mol = Molecule.from_file(adsorbate_xyz)
    cart_xy = slab.lattice.get_cartesian_coords([lateral_frac_coords[0], lateral_frac_coords[1], 0])[:2]
    z_top = max(site.coords[2] for site in slab)
    shift = [cart_xy[0] - mol.center_of_mass[0], cart_xy[1] - mol.center_of_mass[1],
             z_top + height_A - min(site.coords[2] for site in mol)]

    combined = slab.copy()
    for site in mol:
        combined.append(site.species, site.coords + shift, coords_are_cartesian=True)
    Poscar(combined).write_file(output_poscar)
    return combined
```

### Workflow 5: Phase diagrams, formation energies, and Pourbaix inputs

Build thermodynamic analyses only from compatible entries. The code below supports local DFT entries; database entries require correction compatibility checks before mixing.

```python
from pathlib import Path
import json
import pandas as pd
from pymatgen.core import Composition
from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram, PDPlotter


def make_entries_from_energy_table(
    energy_csv: str,
    energy_col: str = "energy_eV",
    formula_col: str = "formula",
    name_col: str | None = "label",
) -> list[PDEntry]:
    """
    Build PDEntry objects from a table of total energies.

    All energies must come from the same functional, pseudopotential family,
    Hubbard U/correction scheme, and reference convention.
    """
    df = pd.read_csv(energy_csv)
    entries = []
    for _, row in df.iterrows():
        comp = Composition(row[formula_col])
        name = row[name_col] if name_col and name_col in row else comp.reduced_formula
        entries.append(PDEntry(comp, float(row[energy_col]), name=name))
    return entries


def analyze_phase_stability(
    energy_csv: str,
    output_dir: str,
) -> pd.DataFrame:
    """
    Construct a convex hull and write energy-above-hull results.

    Do not mix Materials Project corrected entries with raw local DFT entries
    unless using a compatible correction scheme. [EXPERT REVIEW NEEDED]
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    entries = make_entries_from_energy_table(energy_csv)
    pdia = PhaseDiagram(entries)

    rows = []
    for entry in entries:
        rows.append({
            "name": entry.name,
            "formula": entry.composition.reduced_formula,
            "energy_eV": float(entry.energy),
            "energy_per_atom_eV": float(entry.energy_per_atom),
            "e_above_hull_eV_atom": float(pdia.get_e_above_hull(entry)),
            "formation_energy_per_atom_eV": float(pdia.get_form_energy_per_atom(entry)),
            "is_stable": bool(pdia.get_e_above_hull(entry) < 1e-8),
        })

    result = pd.DataFrame(rows).sort_values("e_above_hull_eV_atom")
    result.to_csv(out / "phase_stability.csv", index=False)

    plotter = PDPlotter(pdia, show_unstable=0.1)
    fig = plotter.get_plot()
    try:
        fig.write_html(str(out / "phase_diagram.html"))
    except AttributeError:
        fig.savefig(out / "phase_diagram.png", dpi=200, bbox_inches="tight")

    (out / "phase_diagram_metadata.json").write_text(json.dumps({
        "input_csv": energy_csv,
        "n_entries": len(entries),
        "elements": [str(el) for el in pdia.elements],
        "warning": "Entries assumed to share one compatible DFT reference scheme.",
    }, indent=2))
    return result
```

### Workflow 6: Interoperability with ASE, matminer, MP API, and Phonopy

Use explicit conversion functions and preserve metadata that downstream tools depend on.

```python
import json
from pathlib import Path
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor


def pymatgen_to_ase_with_site_properties(structure: Structure):
    """
    Convert pymatgen Structure to ASE Atoms and copy simple site properties.

    Complex or nested site properties may not survive conversion and should be
    written to a sidecar JSON.
    """
    adaptor = AseAtomsAdaptor()
    atoms = adaptor.get_atoms(structure)
    atoms.info["pymatgen_formula"] = structure.composition.reduced_formula
    atoms.info["pymatgen_charge"] = structure.charge
    for key, values in structure.site_properties.items():
        if len(values) == len(structure) and all(isinstance(v, (int, float, bool, str)) for v in values):
            atoms.arrays[key] = __import__("numpy").array(values)
    return atoms


def prepare_matminer_dataframe(structure_files: list[str], output_json: str):
    """
    Build a JSON-serializable structure table for matminer workflows.
    """
    rows = []
    for path in structure_files:
        s = Structure.from_file(path)
        rows.append({
            "source_file": path,
            "formula": s.composition.reduced_formula,
            "n_sites": len(s),
            "density": float(s.density),
            "structure": s.as_dict(),
        })
    Path(output_json).write_text(json.dumps(rows, indent=2))
    return rows


def phonopy_atoms_from_pymatgen(structure: Structure):
    """Convert pymatgen Structure to PhonopyAtoms without losing fractional coordinates."""
    from phonopy.structure.atoms import PhonopyAtoms
    return PhonopyAtoms(
        symbols=[str(site.specie) for site in structure],
        cell=structure.lattice.matrix,
        scaled_positions=structure.frac_coords,
    )
```

## Common Failure Modes

| Failure | Symptom | Likely Cause | Diagnostic | Fix |
|---------|---------|--------------|------------|-----|
| Oxidation-state ambiguity | `add_oxidation_state_by_guess` assigns implausible charges or fails; matminer oxidation features become NaN | Mixed-valence, metallic, disordered, or charge-imbalanced composition | Print guessed oxidation states and charge balance; compare to known chemistry | Use explicit oxidation states only when justified; avoid oxidation-dependent features for ambiguous systems |
| Wrong conventional cell | Slab Miller indices, k-path labels, or reported lattice constants do not match literature convention | Used primitive cell, non-standard conventional cell, or wrong symmetry setting | Run `SpacegroupAnalyzer(...).get_conventional_standard_structure()` and compare cell vectors | Choose and document one convention; regenerate downstream slabs/k-paths from that cell |
| Symmetry over-reduction | Distortion, Jahn-Teller mode, defect relaxation, or finite-temperature snapshot is snapped back to higher symmetry | `symprec` or angle tolerance too loose | Run symmetry analysis at several `symprec` values; compare RMS displacement before/after | Use tighter tolerance or avoid standardization for distorted structures |
| Disordered structure mishandled as ordered | CIF parses but DFT input has fractional occupancies, mixed species, or impossible site count | Partial occupancy or disorder not ordered before calculation | Check `structure.is_ordered`; inspect species occupancies at each site | Generate ordered approximants; enumerate configurations; record ordering assumptions |
| Partial occupancies dropped | Composition changes after CIF to POSCAR conversion | Target format cannot encode partial occupancy | Compare composition before/after conversion and inspect disordered sites | Do not write disordered structures directly to POSCAR; order or reject them first |
| Lost site properties | Magnetic moments, selective dynamics, charges, labels, or tags disappear after conversion | Format or adaptor does not preserve site properties | Compare `structure.site_properties` before/after round trip | Write sidecar JSON; reattach properties after conversion; verify before calculation |
| Wrong energy reference | Phase diagram shows stable phases that contradict known chemistry | Mixed raw/corrected energies, different functionals, different elemental references, or incompatible Hubbard U settings | Print entry metadata by source and correction scheme | Use one compatible entry set; apply documented corrections; do not mix references silently |
| Incompatible units | Formation energies, voltages, or surface energies off by factors of atom count, area, or charge | Total energy vs per-atom energy confusion; Angstrom vs cm; eV vs kJ/mol | Recompute one value by hand from raw entries | Encode units in column names and metadata; convert once in a tested preprocessing step |
| Silent structure changes during format conversion | Atom ordering, lattice, species, or charge state differs after read/write | CIF/POSCAR/ASE adaptor limitations or symmetry expansion | Run round-trip fingerprint comparison | Use richer formats or sidecars; assert invariants before downstream use |
| VASP parser trusts incomplete output | Energy table includes unconverged calculations | `vasprun.xml` exists but electronic or ionic convergence failed | Check `vasprun.converged`, `converged_electronic`, and `converged_ionic` | Exclude unconverged runs; record failures in manifest |

## Best Practices

- **Record every transformation.** Structure standardization, supercell generation, substitution, ordering, and slab construction should write a manifest that captures input, parameters, and output formula.
- **Use conservative symmetry tolerances for distorted structures.** Defects, surfaces, finite-temperature snapshots, and soft-mode structures should often be analyzed with lower symmetry than ideal crystals.
- **Check `structure.is_ordered` before writing DFT inputs.** POSCAR cannot faithfully represent partial occupancies; ordered approximants must be generated intentionally.
- **Preserve site properties explicitly.** Magnetic moments, selective dynamics, labels, and charges should be validated after conversion and stored in sidecar metadata when needed.
- **Separate raw parsing from scientific filtering.** Parse all VASP outputs into summaries, then filter converged and compatible runs in a transparent table.
- **Never mix thermodynamic references silently.** Convex hulls and Pourbaix diagrams need consistent functionals, corrections, and elemental references.
- **Use `StructureMatcher` for duplicate detection after transformations.** It catches symmetry-equivalent variants that simple formula or coordinate comparisons miss.
- **Write both CIF and POSCAR when preparing structures for collaborators.** CIF is readable and preserves crystallographic context; POSCAR is the calculation input.
- **Validate round trips through the exact formats used downstream.** A conversion path is safe only if the downstream reader reconstructs the intended structure.
- **Keep database provenance.** MP IDs, correction schemes, API versions, and query filters must be stored with any imported entry or structure.

## Integration with Other Skills

- **[ase-framework](../ase-framework/SKILL.md):** ASE complements pymatgen for calculator execution, trajectory handling, extxyz datasets, and LAMMPS IO. Use explicit adaptors and verify site properties across conversions.
- **[materials-featurization](../materials-featurization/SKILL.md):** pymatgen `Structure` and `Composition` objects are primary inputs to matminer featurizers. Oxidation-state and structure-standardization choices directly affect feature values.
- **[scientific-visualization](../scientific-visualization/SKILL.md):** pymatgen parsers provide band structures, DOS objects, phase diagrams, charge-density grids, and standardized structures for plotting and rendering.
- **[reproducibility-workflow](../reproducibility-workflow/SKILL.md):** pymatgen transformations, parsed summaries, phase-diagram entries, and database queries should be stored with hashes, settings, and provenance manifests.
- **[vasp-workflow](../../electronic-structure/vasp-workflow/SKILL.md):** pymatgen generates VASP inputs and parses `vasprun.xml`, `OUTCAR`, `CHGCAR`, `LOCPOT`, and POSCAR files. VASP-specific convergence and physical settings remain governed by that skill.
- **[quantum-espresso](../../electronic-structure/quantum-espresso/SKILL.md):** pymatgen can assist structure conversion and some QE parsing, while QE-specific pseudopotential and input syntax remain in that workflow.
- **[dft-convergence](../../electronic-structure/dft-convergence/SKILL.md):** Converged DFT settings should be attached to parsed entries and structure datasets so downstream phase diagrams or featurization do not mix incompatible calculations.
- **[dft-phonons](../../electronic-structure/dft-phonons/SKILL.md):** pymatgen supplies standardized structures, symmetry analysis, and high-symmetry paths that feed phonon workflows.
- **[dft-surfaces](../../electronic-structure/dft-surfaces/SKILL.md):** Slab generation, Miller-index handling, and surface structure preparation are natural pymatgen tasks. This is an intentional forward reference until that skill is added.
- **[dft-defects](../../electronic-structure/dft-defects/SKILL.md):** Defect precursor structures, substitutions, supercells, charge-state metadata, and structure matching rely on pymatgen. This is an intentional forward reference until that skill is added.
- **[high-throughput-dft](../../electronic-structure/high-throughput-dft/SKILL.md):** pymatgen is often the structure and entry layer underneath high-throughput DFT pipelines. This is an intentional forward reference until that skill is added.
- **[materials-databases-access](../materials-databases-access/SKILL.md):** MP API and other materials database queries produce pymatgen structures and entries. This is an intentional forward reference until that skill is added.

## Key References

1. Ong, S. P., et al. (2013). *Python Materials Genomics (pymatgen): A robust, open-source python library for materials analysis.* Computational Materials Science, 68, 314-319. https://doi.org/10.1016/j.commatsci.2012.10.028
2. Jain, A., et al. (2013). *Commentary: The Materials Project: A materials genome approach to accelerating materials innovation.* APL Materials, 1, 011002. https://doi.org/10.1063/1.4812323
3. Setyawan, W., & Curtarolo, S. (2010). *High-throughput electronic band structure calculations: Challenges and tools.* Computational Materials Science, 49, 299-312. https://doi.org/10.1016/j.commatsci.2010.05.010
4. Togo, A., & Tanaka, I. (2018). *Spglib: a software library for crystal symmetry search.* https://spglib.github.io/spglib/
5. Ong, S. P., Wang, L., Kang, B., & Ceder, G. (2008). *Li-Fe-P-O2 phase diagram from first principles calculations.* Chemistry of Materials, 20, 1798-1807. https://doi.org/10.1021/cm702327g
