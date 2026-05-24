---
name: high-throughput-dft
domain: electronic-structure
maturity: draft
tools:
  - atomate2
  - jobflow
  - FireWorks
  - custodian
  - VASP
  - pymatgen
  - MongoDB
  - ASE
tested_versions:
  atomate2: "0.0.14"
  jobflow: "0.1.17"
  FireWorks: "2.0.3"
  custodian: "2024.3.12"
  VASP: "6.4.2"
  pymatgen: "2024.11.13"
requires_gpu: false
estimated_runtime: hours-to-weeks
---

# High-Throughput DFT Workflows

## Description

This skill covers automated, large-scale DFT screening workflows: workflow graph construction and provenance tracking with atomate2/jobflow or AiiDA, input-set standardization, error handling with custodian, HPC job submission and restart logic, MongoDB-backed results storage, deduplication, and quality control. It connects high-throughput DFT to materials databases, ML training datasets, active learning loops, and surrogate screening campaigns. Invoke this skill when running hundreds to hundreds of thousands of DFT calculations systematically, automating a multi-step DFT pipeline (relax → static → DOS/bands → phonon → defect), or coupling DFT calculations to ML potential training or property screening.

## Domain Context

High-throughput DFT (HT-DFT) is DFT run at scale: the same code, functional, and input protocol applied uniformly to a large structure library so that computed properties are mutually comparable. The value is not any single calculation but the internally consistent dataset that enables thermodynamic screening, property prediction, ML training, and active learning. This internal consistency is fragile — a single inconsistent INCAR, a different POTCAR version, or a missing convergence check can silently corrupt the comparative validity of hundreds of results.

**Workflow orchestration:** A DFT workflow is a directed acyclic graph (DAG) of calculations where each node is a DFT job and each edge is a data dependency. Orchestration tools (atomate2 + jobflow, FireWorks, AiiDA) manage job submission, output parsing, provenance recording, error detection, and chaining. The key distinction is between *workflow definition* (the graph and input sets) and *workflow execution* (the compute infrastructure that runs jobs). Separating these allows the same workflow graph to run on different HPC systems.

**Input-set standardization:** All DFT calculations in a screening campaign must use identical or explicitly documented input sets. The pymatgen `InputSet` machinery (MPRelaxSet, MPStaticSet, MPScanRelaxSet, MPHSEBSSet, etc.) encodes Materials Project–compatible calculation parameters, including POTCAR selection, INCAR defaults, k-mesh generation, and structure transformation. Using these sets ensures that computed energies are compatible with MP compatibility corrections and the MP thermodynamic database.

**Custodian error handling:** DFT calculations fail in stereotyped ways: BRMIX errors, ZPOTRF errors, EDDDAV convergence failures, electronic SCF divergence, wall-time expiry, memory errors. Custodian wraps each DFT job with error-detection handlers that parse stderr and OUTCAR, identify the failure mode, apply a documented fix (e.g., increase NCORE, lower POTIM, restart from WAVECAR), and resubmit. Every applied fix is logged with reason and parameters, forming part of the provenance record.

**Provenance and reproducibility:** An HT-DFT result is only as trustworthy as its provenance chain. At minimum, each result record must store: structure (with standardized cell), code version, POTCAR SHA or title line, functional and correction scheme, INCAR verbatim, k-mesh, convergence flags, custodian fixes applied, and a unique job UUID. Without this, a dataset cannot be reproduced, audited, or correctly cited.

**Convergence tiers:** HT-DFT workflows typically use two or three convergence tiers:
- *Screening tier:* Coarser k-mesh, lower ENCUT (e.g., 1.3× ENMAX), EDIFF=1e-5, EDIFFG=-0.05. Fast; suitable for relative ranking, phase stability screening, and candidate filtering. Results are NOT comparable to experimental values and should not be published as standalone numbers.
- *Standard tier:* MP-standard settings (ENCUT≥520 eV, EDIFF=1e-6, k-mesh ≥ 1000 k·atoms). Suitable for formation energies, band gaps, and inclusion in databases. The Materials Project uses this tier.
- *High-accuracy tier:* ENCUT≥600 eV, EDIFF=1e-8, EDIFFG=-0.01, dense k-mesh, LREAL=False. Required for phonons, defect energies, surface energies, and training data for high-accuracy MLPs.

The most common catastrophic mistake in HT-DFT is treating screening-tier results as publication-quality data. [EXPERT REVIEW NEEDED: the exact ENCUT and k-mesh thresholds that define "screening" vs "standard" are institution-dependent; always document which tier was used]

**MongoDB + maggma:** atomate2 stores results in MongoDB documents via the `maggma` store abstraction. Each job produces a document with fields for structure, energy, forces, stresses, magnetic moments, band gap, correction, and full provenance. The `emmet` package defines the document schema used by Materials Project. Direct MongoDB queries allow post-processing without re-running calculations.

## When to Use This Skill

- Systematically computing formation energies, band gaps, elastic moduli, or other properties across a library of hundreds to hundreds of thousands of structures.
- Building internally consistent DFT datasets for ML potential training, surrogate model construction, or property prediction.
- Running multi-step DFT pipelines (relax → static → DOS → phonon) automatically with dependency tracking.
- Coupling DFT calculations to an active learning loop where new candidate structures are generated iteratively.
- Setting up a group-internal materials database with consistent provenance for reproducible research.
- Benchmarking universal MLPs or surrogate models against systematically computed DFT references on a held-out structure library.

## When Not to Use This Skill

- **Single-material deep study:** If the goal is one accurate number on one material (band gap, surface energy, defect level), use the individual DFT workflow skills directly. HT-DFT infrastructure adds overhead that is not justified for fewer than ~20 structures.
- **Mixed-DFT-code campaigns without explicit compatibility checking:** Mixing VASP results with QE results, or PBE with SCAN, without a validated correction scheme produces a dataset that appears large but is internally inconsistent.
- **When the compute budget is extremely limited:** HT-DFT pipelines are designed to handle thousands of jobs; setting up the infrastructure is not cost-effective for fewer than ~50 calculations.
- **When provenance recording is optional:** If results will be published, presented, or used to train a model, provenance cannot be optional. If the team is unwilling to record and store full job metadata, HT-DFT at scale will produce untraceable results.
- **When no error handling strategy is planned:** Unhandled DFT failures at scale silently corrupt datasets (missing structures, wrong convergence). Custodian or equivalent is non-optional for production campaigns.

## Core Concepts

### Workflow Graph and Job Chaining

A workflow in atomate2/jobflow is a Python graph of `Job` objects connected by outputs-as-inputs. Each `Job` wraps a function (a `Maker`'s `make` method) that returns a `Response` containing the job output and optionally a set of child jobs to add dynamically. The key data flow:

```
structure → RelaxMaker → relaxed_structure, energy
relaxed_structure → StaticMaker → total_energy, forces, stresses
relaxed_structure → NonSCFMaker → band_structure, DOS
```

Dynamic workflows (jobs created at runtime based on prior outputs) are represented by `Response(addition=...)` — the child job is created inside the parent's function after parsing DFT output.

**FireWorks vs jobflow:** FireWorks uses a central MongoDB-backed LaunchPad to queue and track `Firework` tasks; suitable for large clusters with many simultaneous jobs. jobflow is a newer abstraction that defines workflows as Python objects without a central server; it can execute via `jobflow-remote` (SLURM-native) or convert to FireWorks for submission. atomate2 uses jobflow as its workflow definition layer.

**AiiDA** provides similar functionality with stronger provenance guarantees (immutable data nodes, verifiable data graph) but requires more infrastructure setup. Better suited for long-term archives and cross-institutional provenance than rapid internal screening campaigns.

### Input-Set Standardization

pymatgen `VaspInputSet` classes generate INCAR, KPOINTS, POSCAR, and POTCAR for a given structure and workflow step:

- `MPRelaxSet`: standard GGA ionic relaxation (ISIF=3 for bulk)
- `MPStaticSet`: post-relaxation single-point with denser k-mesh
- `MPNonSCFSet`: band structure or DOS on pre-computed charge density
- `MPScanRelaxSet`: SCAN meta-GGA for improved accuracy on oxides
- `MPHSEBSSet`: HSE06 band structure (expensive; use selectively)
- `MPDefectSet` (via doped/pydefect): charged defect supercells

All sets pull POTCAR from a local POTCAR library referenced by `VASP_PSP_DIR`. The POTCAR selection per element (e.g., Fe vs Fe_pv vs Fe_sv) follows the Materials Project recommended mapping and is critical for energy compatibility.

### Custodian Error Handlers

Custodian runs a DFT code as a subprocess, monitors output, detects known errors, and applies fixes. Key handlers for VASP:

| Handler | Triggers on | Fix applied |
|---|---|---|
| `VaspErrorHandler` | BRMIX, ZBRENT, ZPOTRF, EDDDDAV | Lower POTIM, set IBRION=1, increase NELM |
| `NonConvergingErrorHandler` | Electronic SCF not converging after NELM steps | Switch ALGO, adjust AMIX/BMIX |
| `WalltimeHandler` | Job approaching wall time | Save WAVECAR, set ISTART=1 for restart |
| `MeshSymmetryErrorHandler` | k-mesh incompatible with symmetry | Adjust k-mesh to satisfy symmetry |
| `MaxForceErrorHandler` | Forces not converging after NSW steps | Increase NSW, reduce EDIFFG slightly |
| `MemoryHandler` | OOM kill signal | Reduce NCORE, increase memory request |

Each applied fix is logged with the error string, handler name, and parameters changed. This log becomes part of the job's provenance record.

### Database Schema and Provenance

atomate2 stores results as MongoDB documents (via `JobStore`) with a stable schema:

```json
{
  "uuid": "3f8a...",
  "output": {
    "structure": {...},
    "energy": -42.31,
    "energy_per_atom": -5.29,
    "forces": [...],
    "stress": [...],
    "bandgap": 1.23,
    "is_gap_direct": false
  },
  "metadata": {
    "vasp_version": "6.4.2",
    "potcar_spec": [{"titel": "PAW_PBE Fe_pv 06Sep2000", "hash": "..."}],
    "incar": {...},
    "kpoints": {...},
    "custodian": [{"corrections": [], "job": {...}}]
  },
  "completed_at": "2024-11-15T14:32:11",
  "state": "completed"
}
```

### Structure Matching and Deduplication

Structure matching identifies whether two structures represent the same material (accounting for cell choice, atom ordering, and tolerance). pymatgen `StructureMatcher` uses a symmetry-aware algorithm. False negatives (missed duplicates) corrupt comparative datasets; false positives (over-aggressive deduplication) remove valid polymorphs.

### Convergence Quality Flags

Every completed result should carry a convergence quality flag:
- `forces_converged`: max force < EDIFFG threshold
- `energy_converged`: energy change < EDIFF × N_atoms in last two SCF steps
- `electronic_steps`: number of electronic steps (flag if > 0.8 × NELM)
- `ionic_steps`: number of ionic steps (flag if = NSW exactly, meaning relaxation may not have finished)
- `custodian_fixes`: list of fixes applied (flag if non-empty for high-accuracy results)

## Key Workflows

### Workflow 1: atomate2 Relax → Static → Band Structure Pipeline

```python
# ht_dft_pipeline.py
# Defines and submits a multi-step DFT workflow using atomate2 and jobflow.
# Requires: MongoDB server, FireWorks launchpad or jobflow-remote, VASP installed,
#           POTCAR library at $VASP_PSP_DIR, pymatgen config at ~/.config/pymatgen/config.yaml

from jobflow import Flow
from atomate2.vasp.jobs.core import RelaxMaker, StaticMaker, NonSCFMaker
from atomate2.vasp.powerups import add_metadata_to_flow, add_wavecar_to_flow
from pymatgen.core import Structure
from pymatgen.io.vasp.sets import MPRelaxSet, MPStaticSet, MPNonSCFSet

# --- Load structures from file list or database query ---
structure_files = [
    "candidates/TiO2_rutile.cif",
    "candidates/TiO2_anatase.cif",
    "candidates/VO2_monoclinic.cif",
]
structures = [Structure.from_file(f) for f in structure_files]

# --- Define workflow makers with explicit input-set overrides ---
relax_maker = RelaxMaker(
    input_set_generator=MPRelaxSet,
    run_vasp_kwargs={"vasp_cmd": "mpirun -np 64 vasp_std"},
    task_document_kwargs={"store_volumetric_data": []},   # skip CHGCAR/LOCPOT for screening
)

static_maker = StaticMaker(
    input_set_generator=MPStaticSet,
    run_vasp_kwargs={"vasp_cmd": "mpirun -np 64 vasp_std"},
)

nonscf_maker = NonSCFMaker(
    input_set_generator=MPNonSCFSet,
    mode="line",    # band structure along high-symmetry path
    run_vasp_kwargs={"vasp_cmd": "mpirun -np 32 vasp_std"},
)

# --- Build flows for all structures ---
all_flows = []
for struct in structures:
    formula = struct.composition.reduced_formula

    # Step 1: Relax
    relax_job = relax_maker.make(struct)
    relax_job.name = f"{formula}_relax"

    # Step 2: Static on relaxed structure (output of relax feeds in)
    static_job = static_maker.make(relax_job.output.structure)
    static_job.name = f"{formula}_static"

    # Step 3: Band structure on static charge density
    nonscf_job = nonscf_maker.make(
        structure=static_job.output.structure,
        prev_dir=static_job.output.dir_name,
    )
    nonscf_job.name = f"{formula}_bands"

    # Assemble flow (preserves dependency graph)
    flow = Flow([relax_job, static_job, nonscf_job], name=f"{formula}_workflow")
    flow = add_metadata_to_flow(flow, {"source": "candidate_screening", "batch": "2024-11"})
    all_flows.append(flow)

# --- Submit via FireWorks ---
from fireworks import LaunchPad
from jobflow.managers.fireworks import flow_to_workflow

lpad = LaunchPad.from_file("my_launchpad.yaml")
for flow in all_flows:
    wf = flow_to_workflow(flow)
    lpad.add_wf(wf)
    print(f"Submitted: {wf.name} ({wf.fw_ids})")

print(f"\nTotal workflows submitted: {len(all_flows)}")
print("Monitor with: lpad get_wflows -s RUNNING")
```

### Workflow 2: Input-Set Standardization and POTCAR Validation

Inconsistent POTCARs are the most common silent source of non-comparable energies. Always validate before submitting.

```python
# potcar_validation.py
# Validate that all structures in a batch use the expected POTCAR versions
# and that the element-to-POTCAR mapping is consistent.

import hashlib
import json
from pathlib import Path
from pymatgen.core import Structure
from pymatgen.io.vasp.sets import MPRelaxSet
from pymatgen.io.vasp.inputs import Potcar

# Expected POTCAR title lines (from Materials Project protocol, PBE 54 dataset)
# These are the canonical PAW_PBE titles for MP-compatible calculations.
MP_POTCAR_TITLES = {
    "Ti": "PAW_PBE Ti_pv 07Sep2000",
    "O":  "PAW_PBE O 08Apr2002",
    "V":  "PAW_PBE V_pv 07Sep2000",
    "Fe": "PAW_PBE Fe_pv 06Sep2000",
    "Ni": "PAW_PBE Ni_pv 06Sep2000",
    "Ga": "PAW_PBE Ga_d 06Oct2001",
    "As": "PAW_PBE As 06Sep2000",
    # ... extend for all elements in campaign
}

def validate_potcar(potcar_path, expected_titles):
    """Check that each species in a POTCAR matches the expected title line."""
    potcar = Potcar.from_file(potcar_path)
    errors = []
    for single in potcar:
        element = single.element
        expected = expected_titles.get(element)
        if expected is None:
            errors.append(f"  {element}: no expected title defined")
        elif single.TITEL.strip() != expected:
            errors.append(f"  {element}: got '{single.TITEL.strip()}' expected '{expected}'")
    return errors

def check_batch_consistency(job_dirs):
    """
    Check POTCAR consistency across all job directories in a batch.
    Flag any jobs where POTCARs differ from the batch majority.
    """
    title_sets = {}
    for d in job_dirs:
        potcar = d / "POTCAR"
        if not potcar.exists():
            continue
        p = Potcar.from_file(potcar)
        key = tuple(sorted(s.TITEL.strip() for s in p))
        title_sets[d.name] = key

    # Check all have same set of elements
    unique_sets = set(title_sets.values())
    if len(unique_sets) > 1:
        print(f"WARNING: {len(unique_sets)} different POTCAR configurations in batch:")
        for s in unique_sets:
            jobs_with_this = [k for k, v in title_sets.items() if v == s]
            print(f"  {s} → {len(jobs_with_this)} job(s)")
        return False
    print(f"POTCAR consistency: OK ({len(title_sets)} jobs, 1 unique POTCAR set)")
    return True

# Example usage
structure_dirs = sorted(Path("batch_01").glob("*/"))
check_batch_consistency(structure_dirs)

# Validate against MP expected titles for new submission
test_struct = Structure.from_file("candidates/Fe2O3.cif")
input_set = MPRelaxSet(test_struct)
# Write to temp dir and validate
import tempfile, os
with tempfile.TemporaryDirectory() as tmpdir:
    input_set.write_input(tmpdir)
    errors = validate_potcar(Path(tmpdir) / "POTCAR", MP_POTCAR_TITLES)
    if errors:
        print(f"POTCAR validation FAILED for {test_struct.formula}:")
        for e in errors:
            print(e)
    else:
        print(f"POTCAR validation PASSED for {test_struct.formula}")
```

### Workflow 3: Custodian Error Handling and Job Restart

```python
# custodian_runner.py
# Wrap a VASP job with custodian error handling and wall-time-aware restart.
# This is the pattern used inside atomate2 automatically; shown here for custom jobs.

from custodian import Custodian
from custodian.vasp.jobs import VaspJob
from custodian.vasp.handlers import (
    VaspErrorHandler,
    NonConvergingErrorHandler,
    WalltimeHandler,
    MeshSymmetryErrorHandler,
    MaxForceErrorHandler,
    PositiveEnergyErrorHandler,
    StdErrHandler,
)
from custodian.vasp.validators import VasprunXMLValidator, VaspFilesValidator
from pathlib import Path
import json

VASP_CMD = ["mpirun", "-np", "64", "vasp_std"]
WALL_TIME = 86400   # seconds (24 hours); set to SLURM wall time minus 10 minutes

handlers = [
    VaspErrorHandler(output_filename="vasp.out"),    # BRMIX, ZPOTRF, EDDAV, etc.
    NonConvergingErrorHandler(output_filename="vasp.out", nionic_steps=10),
    WalltimeHandler(wall_time=WALL_TIME, buffer_time=600),   # saves WAVECAR 10 min before wall
    MeshSymmetryErrorHandler(output_filename="vasp.out"),
    MaxForceErrorHandler(output_filename="vasp.out"),
    PositiveEnergyErrorHandler(),                    # catches positive total energies (bad geometry)
    StdErrHandler(output_filename="std_err.txt"),
]

validators = [
    VasprunXMLValidator(),    # checks that vasprun.xml is complete and parseable
    VaspFilesValidator(),     # checks that OUTCAR, CONTCAR, OSZICAR are present
]

jobs = [
    VaspJob(
        vasp_cmd=VASP_CMD,
        output_file="vasp.out",
        stderr_file="std_err.txt",
        suffix=".relax1",       # rename output files after completion
        settings_override=None,
        copy_magmom=True,       # copy MAGMOM from OUTCAR to INCAR of restart
        auto_npar=True,         # set NPAR based on available cores
        auto_gamma=True,        # auto-detect if k-mesh is Gamma-only
    )
]

# If restarting from WAVECAR (wall-time handler fired):
# Add a second VaspJob with ISTART=1
if Path("WAVECAR").exists() and Path("WAVECAR").stat().st_size > 1000:
    jobs.append(
        VaspJob(
            vasp_cmd=VASP_CMD,
            output_file="vasp.out",
            final=True,
            suffix=".relax2",
            settings_override=[{"dict": "INCAR", "action": {"_set": {"ISTART": 1}}}],
        )
    )

c = Custodian(
    handlers=handlers,
    jobs=jobs,
    validators=validators,
    max_errors=5,
    max_errors_per_job=3,
    scratch_dir=None,
    gzip_output=False,
)

# Run the job; custodian logs all corrections to custodian.json
corrections = c.run()

# Inspect corrections applied
with open("custodian.json") as f:
    cust_log = json.load(f)
print(f"Corrections applied: {len(cust_log[0].get('corrections', []))}")
for corr in cust_log[0].get("corrections", []):
    print(f"  {corr['handler']}: {corr['correction']}")
```

### Workflow 4: Database Ingestion and Results Querying

```python
# results_database.py
# Store and query HT-DFT results in MongoDB via maggma (atomate2's store abstraction).

from maggma.stores import MongoStore, MemoryStore
from maggma.builders import MapBuilder
from jobflow.managers.local import run_locally
from emmet.core.tasks import TaskDoc
from pymatgen.core import Structure
import numpy as np
import pandas as pd
from datetime import datetime

# --- Connect to results store ---
results_store = MongoStore(
    database="htdft_results",
    collection_name="tasks",
    host="localhost",
    port=27017,
    key="uuid",
    last_updated_field="completed_at",
)
results_store.connect()

# --- Query all completed relaxations ---
completed = list(results_store.query(
    criteria={"state": "completed", "name": {"$regex": ".*_relax$"}},
    properties=["uuid", "output.structure", "output.energy_per_atom",
                "output.bandgap", "metadata.vasp_version",
                "metadata.potcar_spec", "completed_at"],
))
print(f"Completed relaxations: {len(completed)}")

# --- Build results DataFrame ---
records = []
for doc in completed:
    out = doc.get("output", {})
    meta = doc.get("metadata", {})
    struct_dict = out.get("structure")
    if struct_dict is None:
        continue
    struct = Structure.from_dict(struct_dict)
    records.append({
        "uuid":            doc["uuid"],
        "formula":         struct.composition.reduced_formula,
        "n_atoms":         len(struct),
        "volume_per_atom": struct.volume / len(struct),
        "energy_per_atom": out.get("energy_per_atom"),
        "bandgap":         out.get("bandgap"),
        "vasp_version":    meta.get("vasp_version"),
        "completed_at":    doc.get("completed_at"),
        "potcar_hash":     meta.get("potcar_spec", [{}])[0].get("hash", "unknown"),
    })

df = pd.DataFrame(records)
print(f"\nDataFrame shape: {df.shape}")
print(df.describe())

# --- Quality control checks ---
def qc_check(df):
    issues = []
    # 1. Energy outliers: flag if > 3 σ from median
    e = df["energy_per_atom"].dropna()
    median, std = e.median(), e.std()
    outliers = df[np.abs(df["energy_per_atom"] - median) > 4 * std]
    if len(outliers) > 0:
        issues.append(f"Energy outliers (>4σ): {list(outliers['formula'])}")

    # 2. VASP version consistency
    versions = df["vasp_version"].dropna().unique()
    if len(versions) > 1:
        issues.append(f"Mixed VASP versions: {list(versions)}")

    # 3. POTCAR hash consistency
    hashes = df["potcar_hash"].dropna().unique()
    if len(hashes) > 1:
        issues.append(f"Mixed POTCAR hashes: {list(hashes)}")

    # 4. Suspiciously large volumes
    v = df["volume_per_atom"].dropna()
    if (v > 50).any():
        issues.append(f"Large volumes (>50 Å³/atom): {list(df.loc[v > 50, 'formula'])}")

    # 5. Negative band gaps (numerical artifact)
    if "bandgap" in df.columns:
        neg_bg = df[df["bandgap"].fillna(0) < -0.1]
        if len(neg_bg) > 0:
            issues.append(f"Negative band gaps: {list(neg_bg['formula'])}")

    if not issues:
        print("QC: all checks passed")
    else:
        print(f"QC: {len(issues)} issue(s) found:")
        for issue in issues:
            print(f"  - {issue}")
    return issues

qc_issues = qc_check(df)
```

### Workflow 5: Structure Matching and Deduplication

```python
# deduplication.py
# Deduplicate a set of structures before submission or after results ingestion.
# Critical for preventing train/test leakage and ensuring clean comparative datasets.

from pymatgen.analysis.structure_matcher import StructureMatcher, ElementComparator
from pymatgen.core import Structure
from collections import defaultdict
from pathlib import Path
import json
import numpy as np

def build_structure_groups(structure_dict, ltol=0.2, stol=0.3, angle_tol=5.0,
                           allow_subset=False, primitive_cell=True):
    """
    Group structures that are equivalent under StructureMatcher.
    Returns: list of groups, each group is a list of (key, Structure) pairs.

    ltol: fractional length tolerance
    stol: site tolerance (fraction of average nearest-neighbor distance)
    angle_tol: angle tolerance in degrees
    [EXPERT REVIEW NEEDED: tolerance values affect false-positive/negative rate;
     tighter = more duplicates missed; looser = distinct polymorphs merged]
    """
    sm = StructureMatcher(
        ltol=ltol,
        stol=stol,
        angle_tol=angle_tol,
        primitive_cell=primitive_cell,
        allow_subset=allow_subset,
        comparator=ElementComparator(),
    )

    # Sort by formula for faster grouping (only compare same-formula structures)
    by_formula = defaultdict(list)
    for key, struct in structure_dict.items():
        by_formula[struct.composition.reduced_formula].append((key, struct))

    groups = []
    processed = set()
    for formula, entries in by_formula.items():
        ungrouped = [(k, s) for k, s in entries if k not in processed]
        while ungrouped:
            key0, struct0 = ungrouped[0]
            group = [(key0, struct0)]
            processed.add(key0)
            still_ungrouped = []
            for key1, struct1 in ungrouped[1:]:
                if sm.fit(struct0, struct1):
                    group.append((key1, struct1))
                    processed.add(key1)
                else:
                    still_ungrouped.append((key1, struct1))
            groups.append(group)
            ungrouped = still_ungrouped

    return groups

# Load candidate structures
structure_files = list(Path("candidates").glob("*.cif")) + list(Path("candidates").glob("*.vasp"))
structures = {}
for f in structure_files:
    try:
        s = Structure.from_file(str(f))
        structures[f.stem] = s
    except Exception as e:
        print(f"  Could not read {f}: {e}")

print(f"Loaded {len(structures)} candidate structures")

groups = build_structure_groups(structures)
n_unique = len(groups)
n_duplicates = len(structures) - n_unique

print(f"Unique structures: {n_unique}")
print(f"Duplicates removed: {n_duplicates}")

# Report groups with more than one member
print("\nDuplicate groups (>1 member):")
for group in groups:
    if len(group) > 1:
        formula = group[0][1].composition.reduced_formula
        print(f"  {formula}: {[k for k, _ in group]}")

# Select canonical representative from each group (lowest energy or alphabetical)
unique_structures = {}
for group in groups:
    canonical_key = group[0][0]   # first member as canonical
    unique_structures[canonical_key] = group[0][1]

print(f"\nCanonical set: {len(unique_structures)} unique structures for submission")
```

### Workflow 6: Convergence Tier Policy and Quality Flags

```python
# convergence_policy.py
# Check that completed VASP jobs meet the convergence criteria for their tier,
# and flag results that failed to converge within tolerance.

import json
from pathlib import Path
from pymatgen.io.vasp import Outcar, Vasprun
import numpy as np

CONVERGENCE_TIERS = {
    "screening": {
        "ediff":    1e-5,
        "ediffg":   -0.05,   # eV/Å
        "encut_min": 400,    # eV
        "max_ionic_steps": 100,
        "description": "Screening quality: sufficient for relative ranking only",
    },
    "standard": {
        "ediff":    1e-6,
        "ediffg":   -0.02,
        "encut_min": 520,
        "max_ionic_steps": 200,
        "description": "Standard MP-compatible quality: suitable for database inclusion",
    },
    "high_accuracy": {
        "ediff":    1e-8,
        "ediffg":   -0.01,
        "encut_min": 600,
        "max_ionic_steps": 300,
        "description": "High accuracy: required for phonons, defects, MLPs",
    },
}

def check_vasp_convergence(job_dir, tier="standard"):
    """
    Check whether a completed VASP job meets the convergence requirements for a given tier.
    Returns a dict with pass/fail flags and diagnostic info.
    """
    job_dir = Path(job_dir)
    policy = CONVERGENCE_TIERS[tier]
    result = {"tier": tier, "job_dir": str(job_dir), "issues": []}

    # --- Parse OUTCAR ---
    outcar_path = job_dir / "OUTCAR"
    if not outcar_path.exists():
        result["issues"].append("OUTCAR missing")
        result["passed"] = False
        return result

    outcar = Outcar(str(outcar_path))

    # 1. Electronic convergence
    if outcar.converged_electronic is False:
        result["issues"].append("Electronic SCF not converged")

    # 2. Ionic convergence (check max force)
    if hasattr(outcar, "final_force_on_atoms") and outcar.final_force_on_atoms is not None:
        max_force = np.max(np.linalg.norm(outcar.final_force_on_atoms, axis=1))
        if max_force > abs(policy["ediffg"]):
            result["issues"].append(
                f"Max force {max_force:.4f} eV/Å > threshold {abs(policy['ediffg'])} eV/Å"
            )
        result["max_force_eV_ang"] = float(max_force)

    # 3. Number of ionic steps
    ionic_steps = outcar.run_stats.get("Elapsed time (sec)", None)
    # Use vasprun for ionic step count
    vasprun_path = job_dir / "vasprun.xml"
    if vasprun_path.exists():
        try:
            vr = Vasprun(str(vasprun_path), parse_potcar_file=False, exception_on_bad_xml=False)
            n_ionic = len(vr.ionic_steps)
            result["ionic_steps"] = n_ionic
            if n_ionic >= policy["max_ionic_steps"]:
                result["issues"].append(
                    f"Ionic steps ({n_ionic}) hit maximum ({policy['max_ionic_steps']}); "
                    "relaxation may not be complete"
                )
            # ENCUT check
            encut = vr.parameters.get("ENCUT", 0)
            if encut < policy["encut_min"]:
                result["issues"].append(
                    f"ENCUT={encut} < minimum {policy['encut_min']} for {tier} tier"
                )
            result["encut"] = encut
        except Exception as e:
            result["issues"].append(f"vasprun.xml parse error: {e}")

    # 4. Custodian corrections
    cust_path = job_dir / "custodian.json"
    if cust_path.exists():
        with open(cust_path) as f:
            cust_log = json.load(f)
        n_corrections = sum(len(run.get("corrections", [])) for run in cust_log)
        result["custodian_corrections"] = n_corrections
        if n_corrections > 0 and tier == "high_accuracy":
            result["issues"].append(
                f"Custodian applied {n_corrections} correction(s); "
                "flag for manual review at high-accuracy tier"
            )

    result["passed"] = len(result["issues"]) == 0
    return result

# Batch check all completed jobs
job_dirs = sorted(Path("batch_01").glob("*/"))
tier = "standard"
n_passed, n_failed = 0, 0
for d in job_dirs:
    if not (d / "OUTCAR").exists():
        continue
    check = check_vasp_convergence(d, tier=tier)
    status = "PASS" if check["passed"] else "FAIL"
    if not check["passed"]:
        n_failed += 1
        print(f"  {d.name}: {status} — {'; '.join(check['issues'])}")
    else:
        n_passed += 1

print(f"\n{tier.upper()} tier: {n_passed} passed, {n_failed} failed out of {n_passed+n_failed}")
```

### Workflow 7: Coupling HT-DFT to Active Learning and ML Screening

```python
# ht_dft_active_learning.py
# Interface between HT-DFT results and an active learning loop.
# After each HT-DFT batch, retrain the surrogate, identify high-uncertainty
# candidates, and queue them for the next DFT batch.

import numpy as np
import pandas as pd
from pathlib import Path

def extract_training_data_from_db(results_store, min_quality="standard"):
    """
    Pull completed DFT results from MongoDB, filter by quality tier,
    and return (structures, targets) ready for ML training.
    """
    from pymatgen.core import Structure
    quality_map = {"screening": 0, "standard": 1, "high_accuracy": 2}
    min_q = quality_map[min_quality]

    docs = list(results_store.query(
        criteria={"state": "completed", "metadata.tier_quality": {"$gte": min_q}},
        properties=["output.structure", "output.energy_per_atom", "output.bandgap",
                    "metadata.formula", "uuid"],
    ))

    structures, energies, band_gaps, ids = [], [], [], []
    for doc in docs:
        out = doc.get("output", {})
        e = out.get("energy_per_atom")
        bg = out.get("bandgap")
        struct_dict = out.get("structure")
        if struct_dict is None or e is None:
            continue
        structures.append(Structure.from_dict(struct_dict))
        energies.append(e)
        band_gaps.append(bg)
        ids.append(doc["uuid"])

    print(f"Extracted {len(structures)} structures from DB (tier >= {min_quality})")
    return structures, np.array(energies), np.array(band_gaps), ids

def rank_candidates_by_uncertainty(candidate_structures, surrogate_model, top_k=50):
    """
    Use surrogate model (e.g., GP or ensemble) uncertainty to rank candidates.
    Returns indices of top_k highest-uncertainty candidates.
    """
    # Featurize candidates
    from matminer.featurizers.composition import ElementProperty
    from matminer.featurizers.site import CrystalNNFingerprint
    import pandas as pd

    # Featurization (composition-level for speed; site-level for accuracy)
    featurizer = ElementProperty.from_preset("magpie")
    features = []
    for s in candidate_structures:
        try:
            feat = featurizer.featurize(s.composition)
            features.append(feat)
        except Exception:
            features.append([np.nan] * featurizer.feature_labels().__len__())

    X = np.array(features)
    valid = ~np.isnan(X).any(axis=1)
    X_valid = X[valid]

    # Get uncertainty from surrogate
    _, sigma = surrogate_model.predict(X_valid, return_std=True)

    # Map back to original indices
    uncertainty = np.zeros(len(candidate_structures))
    uncertainty[valid] = sigma

    # Rank by descending uncertainty (acquisition function: max uncertainty)
    ranked_indices = np.argsort(-uncertainty)
    selected = ranked_indices[:top_k]

    print(f"Selected top {top_k} candidates by uncertainty:")
    print(f"  Max uncertainty: {uncertainty[selected[0]]:.4f}")
    print(f"  Min uncertainty at cutoff: {uncertainty[selected[-1]]:.4f}")
    return selected, uncertainty

def submit_next_batch(candidate_structures, selected_indices, batch_label):
    """Submit selected candidates as the next DFT batch."""
    selected = [candidate_structures[i] for i in selected_indices]
    print(f"Preparing batch '{batch_label}': {len(selected)} structures")

    from pymatgen.io.vasp.sets import MPRelaxSet
    batch_dir = Path(f"batches/{batch_label}")
    for i, struct in enumerate(selected):
        formula = struct.composition.reduced_formula
        job_dir = batch_dir / f"{i:04d}_{formula}"
        job_dir.mkdir(parents=True, exist_ok=True)
        input_set = MPRelaxSet(struct)
        input_set.write_input(str(job_dir))

    print(f"Input files written to {batch_dir}/")
    print("Submit with: rlaunch -c my_fworker.yaml rapidfire --m_fw 100")
    return list(selected)

# --- Active learning iteration skeleton ---
# (Assumes surrogate model is already trained on a seed dataset)
# In practice, this runs as a loop: DFT → train → screen → DFT → ...

# 1. Load completed results
# structures_done, energies_done, _, ids_done = extract_training_data_from_db(results_store)

# 2. Retrain surrogate on updated dataset
# surrogate = train_gp(structures_done, energies_done)   # see gaussian-process-materials skill

# 3. Screen candidate pool for high-uncertainty structures
# candidates_all = load_candidate_pool("candidate_pool.json")
# selected_idx, uncertainty = rank_candidates_by_uncertainty(candidates_all, surrogate, top_k=50)

# 4. Submit next batch
# submit_next_batch(candidates_all, selected_idx, batch_label="al_iteration_03")

print("Active learning coupling: workflow skeleton complete.")
print("See gaussian-process-materials and active-learning-mlp skills for surrogate details.")
```

### Workflow 8: HPC Submission and SLURM Integration

```bash
#!/bin/bash
# slurm_rlaunch.sh — SLURM job script for FireWorks rlaunch (rocket launch)
# Each SLURM array job picks up one Firework from the LaunchPad.
# Array size controls parallelism; max_fws controls jobs per launch.

#SBATCH --job-name=htdft_rlaunch
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=64
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --time=24:00:00
#SBATCH --partition=compute
#SBATCH --array=1-200%50     # 200 total jobs, max 50 simultaneous
#SBATCH --output=logs/rlaunch_%A_%a.out
#SBATCH --error=logs/rlaunch_%A_%a.err

# Environment
module purge
module load vasp/6.4.2 intel/2023 impi/2023
conda activate htdft_env

export VASP_PSP_DIR=/data/potcar_lib
export PMG_VASP_PSP_DIR=$VASP_PSP_DIR

# Wall time in seconds: leave 10 min buffer for WalltimeHandler to save WAVECAR
# SLURM_TIMELIMIT is in MM:SS or HH:MM:SS format; parse carefully
WALL_TIME_SECS=$(( 24 * 3600 - 600 ))

# Run one Firework per array task
rlaunch -c /home/user/htdft_project/fw_config/ \
    --launchpad_file my_launchpad.yaml \
    --fworker_file my_fworker.yaml \
    singleshot \
    2>&1 | tee logs/fw_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log

# Alternative: rapidfire (multiple FWs per job, until time runs out)
# rlaunch -c /home/user/htdft_project/fw_config/ \
#     rapidfire --m_fw 5 --nlaunches 5 --sleep 30
```

```yaml
# my_fworker.yaml — FireWorks FWorker configuration
name: hpc_compute_nodes
category: ""
query: {}
env:
  vasp_cmd: mpirun -np 128 vasp_std
  gamma_vasp_cmd: mpirun -np 128 vasp_gam
  scratch_dir: /scratch/$USER/fw_scratch
  small_vasp_cmd: mpirun -np 32 vasp_std
```

```python
# monitor_campaign.py — Quick dashboard for running HT-DFT campaign
from fireworks import LaunchPad
from collections import Counter
import pandas as pd
from datetime import datetime, timedelta

lpad = LaunchPad.from_file("my_launchpad.yaml")

# Count FW states
states = Counter()
for fw in lpad.get_fw_ids():
    fw_doc = lpad.get_fw_by_id(fw)
    states[fw_doc.state] += 1

total = sum(states.values())
print(f"Campaign status ({datetime.now().strftime('%Y-%m-%d %H:%M')}):")
for state, count in sorted(states.items()):
    pct = 100 * count / total
    print(f"  {state:15s}: {count:5d} ({pct:.1f}%)")
print(f"  {'TOTAL':15s}: {total:5d}")

# Flag stuck RUNNING jobs (running > 26 hours = likely wall-time failure not caught)
running_fw_ids = lpad.get_fw_ids(state="RUNNING")
for fw_id in running_fw_ids:
    fw = lpad.get_fw_by_id(fw_id)
    last_update = fw.updated_on
    age = datetime.utcnow() - last_update
    if age > timedelta(hours=26):
        print(f"  WARNING: FW {fw_id} ({fw.name}) has been RUNNING for {age}; may be stuck")
```

## Common Failure Modes

| Failure Mode | Symptoms | Likely Cause | Diagnostic | Fix |
|---|---|---|---|---|
| Silent job failure | FIZZLED Firework; no custodian.json; OUTCAR missing or empty | VASP crashed before custodian could detect the error; OOM kill; filesystem issue | Check SLURM log: `sacct -j <jobid>`; check `std_err.txt` | Add `StdErrHandler`; increase memory request; check scratch disk quota |
| Inconsistent INCAR across batch | Energies non-comparable; phase diagram artifacts; suspicious outliers | Different template INCARs used; manual edits; different `InputSet` versions | Diff INCARs across job directories; check `pymatgen` version in environment | Pin `pymatgen` version; generate all INCARs from a single `InputSet` class at submission time |
| Mixed POTCAR versions | Systematic energy offsets by 0.1–1 eV; inconsistent formation energies | Different POTCAR library paths or dates used across nodes | Check POTCAR title lines (`grep TITEL */POTCAR`); compare hashes | Fix POTCAR library path; validate with `potcar_validation.py`; rerun affected jobs |
| Non-comparable energies across functional | Formation energies wrong sign or magnitude; phase diagram dominated by functional error | PBE and SCAN or PBE+U results mixed without correction | Check `GGA` tag in INCAR; confirm all jobs use same functional | Use `MaterialsProject2020Compatibility` correction scheme; never mix raw energies across functionals |
| Duplicate structures in dataset | Model overfits; benchmark scores inflated; misleading screening ranking | CIF import, database duplicates, or symmetry-equivalent supercells not deduplicated | Run `StructureMatcher` on entire dataset; check for formula duplicates with different IDs | Deduplicate before training and before train/test split using `build_structure_groups()` |
| Bad error handler causing infinite loop | Job re-queued repeatedly; FIZZLED after max_errors; same error each relaunch | Error handler applies a fix that doesn't resolve the underlying error | Check `custodian.json` for repeated corrections of same type | Increase `max_errors_per_job`; investigate root cause; add a more specific handler |
| Unconverged relaxations accepted | Forces > threshold; CONTCAR geometry incorrect; downstream static wrong | NSW limit hit; EDIFFG too loose for material; custodian did not catch | Check `ionic_steps == NSW` in convergence check; inspect max force in OUTCAR | Increase NSW; tighten EDIFFG; rerun failed relaxations with extended settings |
| Missing provenance | Results cannot be reproduced; journal requires full methodology | No INCAR/KPOINTS/POTCAR archived; UUID not stored; database query not cached | Try to reconstruct inputs from results — if impossible, provenance is lost | Archive inputs alongside results; store custodian.json with every job; record query parameters |
| Treating screening-tier as publication quality | Energies off by 0.1–0.5 eV per atom vs high-accuracy; conclusions do not replicate | Coarse k-mesh, low ENCUT, or loose EDIFF used for full campaign | Check tier label in DB; compare ENCUT and k-mesh to MP standard | Add `tier_quality` metadata field; never remove the tier label from any result |
| Corrupted MongoDB after failed write | Missing documents; partial document fields; duplicate UUIDs | Job wrote partial output before crash; two jobs wrote same UUID | Query for documents with missing required fields; check UUID uniqueness | Use `upsert=True` with UUID as key; run database integrity checks after each batch |
| Provenance chain broken by re-relaxation | It is unclear which relaxed structure produced a static or phonon result | Re-relaxation output stored under same job name; UUID not propagated | Check `prev_dir` fields in job inputs; trace UUID chain | Always propagate parent UUID to child jobs; store `parent_uuid` in every job document |

## Best Practices

**Workflow design:**
- Define every workflow step in code, not in manually edited files. A workflow whose inputs were edited by hand cannot be reproduced.
- Use `InputSet` classes exclusively for INCAR, KPOINTS, and POTCAR generation. Never manually edit individual INCAR files in a production campaign.
- Fix the `pymatgen`, `atomate2`, `custodian`, and `jobflow` versions at the start of a campaign and do not upgrade mid-campaign. Log the conda environment as part of the campaign's provenance record.
- Assign a UUID to every job at submission time (not at completion); use it as the database primary key throughout.

**Error handling:**
- Always wrap VASP with custodian in production. An uncaught VASP error silently leaves no result, which looks identical to a job that was never submitted.
- Set `max_errors=5` and `max_errors_per_job=3`; jobs that exceed this are FIZZLED and available for manual inspection. Do not set `max_errors` to an arbitrarily large number to force completion.
- Include `WalltimeHandler` for all VASP jobs on HPC. Without it, wall-time kills leave no WAVECAR and the job must restart from scratch.
- Review FIZZLED jobs before requeuing; some failure modes (wrong structure, bad initial geometry) will recur regardless of the fix applied.

**Convergence and quality:**
- Label every result with its convergence tier (`screening`, `standard`, `high_accuracy`) in the database and in any derived dataset.
- Never mix results from different tiers in the same phase diagram, ML training set, or comparative analysis without an explicit compatibility correction.
- Run `convergence_policy.py` (Workflow 6) on every completed batch before downstream use. Flag jobs where ionic steps == NSW or max force > threshold.

**Deduplication and compatibility:**
- Deduplicate candidate structures before submission (Workflow 5) to avoid wasted compute.
- Deduplicate again after ingestion, using completed-structure geometries (post-relaxation), which may have symmetrized or changed space group.
- For thermodynamic analysis, apply `MaterialsProject2020Compatibility` to all energies to make them compatible with the MP database. Do not mix corrected and uncorrected energies.

**HPC and scaling:**
- Use SLURM array jobs with a capped concurrency (`--array=1-N%M`) to avoid overwhelming the scheduler or exceeding fair-share allocations.
- Set scratch directories per job on fast local storage; do not write large binary files (WAVECAR, CHGCAR) to network filesystems during the run.
- Archive completed results (OUTCAR, vasprun.xml, custodian.json, CONTCAR) to long-term storage immediately after parsing. Do not rely on scratch disk for permanence.

## Integration with Other Skills

**Upstream DFT workflow skills (each workflow type is a branch of the HT-DFT graph):**

- [VASP Workflow](../vasp-workflow/SKILL.md) — the relaxation, static, and band structure jobs at each graph node use the same INCAR conventions and POTCAR selection protocol as the single-material VASP workflow skill. MPRelaxSet and MPStaticSet are the standard input sets for both.
- [Quantum ESPRESSO Workflow](../quantum-espresso/SKILL.md) — QE-based HT-DFT workflows use AiiDA (via `aiida-quantumespresso`) instead of atomate2; the same convergence-tier and provenance principles apply.
- [DFT Convergence](../dft-convergence/SKILL.md) — convergence tier definitions (ENCUT, k-mesh, EDIFF thresholds) are derived from the convergence skill. Phonon and defect branches use high-accuracy tier settings from that skill.
- [DFT Phonons](../dft-phonons/SKILL.md) — phonon workflow branches in HT-DFT use atomate2's `PhononMaker` (wrapping Phonopy + VASP). Force constant calculations follow the same supercell and INCAR discipline.
- [DFT Surfaces](../dft-surfaces/SKILL.md) — surface energy and adsorption energy branches use slab geometry conventions and dipole correction settings from the surfaces skill. Adsorption screening across many adsorbate/site combinations is a natural HT-DFT use case.
- [DFT Defects](../dft-defects/SKILL.md) — defect formation energy screening across host materials or dopant species uses doped's `DefectsSet` embedded in a jobflow `Flow`. Image-charge corrections are applied in post-processing via the same doped/pydefect infrastructure.
- [DFT Dataset Generation](../dft-dataset-generation/SKILL.md) — HT-DFT is the primary engine for generating large DFT reference datasets. This skill provides the orchestration layer; the dataset generation skill covers the structure sampling, labeling format, and train/test split.

**Data and analysis infrastructure:**

- [Pymatgen Analysis](../../scientific-data/pymatgen-analysis/SKILL.md) — `StructureMatcher`, `InputSet` classes, `ComputedEntry`, `PhaseDiagram`, and `ChemicalPotentialDiagram` are used throughout HT-DFT for structure matching, input generation, and thermodynamic analysis. atomate2 is built on pymatgen.
- [ASE Framework](../../scientific-data/ase-framework/SKILL.md) — structure import/export, extxyz dataset generation, and MLP evaluation on HT-DFT structures all use ASE as the interoperability layer.
- [Materials Databases Access](../../scientific-data/materials-databases-access/SKILL.md) — HT-DFT campaigns often start with structures retrieved from the Materials Project, AFLOW, or OQMD. Compatible energy references and correction schemes from this skill ensure that local HT-DFT results can be compared against database values.
- [Materials Featurization](../../scientific-data/materials-featurization/SKILL.md) — after HT-DFT, computed properties are featurized for surrogate model training. This skill provides the bridge from DFT results to ML-ready feature vectors.
- [Reproducibility Workflow](../../scientific-data/reproducibility-workflow/SKILL.md) — HT-DFT campaigns are the highest-stakes context for reproducibility. Campaign inputs (InputSet parameters, POTCAR library, queue configuration, environment), results (MongoDB snapshot), and analysis scripts must all be version-controlled and archived together.

**ML and screening downstream:**

- [Active Learning MLP](../../ml-interatomic-potentials/active-learning-mlp/SKILL.md) — Workflow 7 implements the interface between HT-DFT results and an active learning loop. Completed DFT results update the surrogate, which selects the next DFT batch.
- [Gaussian Process Materials](../../surrogate-active-learning/gaussian-process-materials/SKILL.md) — GP surrogates are a common choice for property screening with uncertainty quantification; the HT-DFT dataset is the training set and the surrogate ranks uncomputed candidates. [FORWARD REFERENCE — skill may be in progress]
- [Bayesian Optimization](../../surrogate-active-learning/bayesian-optimization/SKILL.md) — Bayesian optimization over material space uses HT-DFT results as the objective function evaluator; the surrogate (GP or neural network) and acquisition function determine the next batch. [FORWARD REFERENCE — skill not yet created]

## Key References

1. Jain, A., Ong, S. P., Hautier, G., Chen, W., Richards, W. D., Dacek, S., Cholia, S., Gunter, D., Skinner, D., Ceder, G. & Persson, K. A. (2013). "Commentary: The Materials Project: A materials genome approach to accelerating materials innovation." *APL Materials* 1, 011002. The Materials Project HT-DFT database and infrastructure reference.
2. Mathew, K., Montoya, J. H., Faghaninia, A., Dwaraknath, S., Aykol, M., Tang, H., Chu, I., Smidt, T., Bocklund, B., Horton, M., Dagdelen, J., Wood, B., Liu, Z.-K., Neaton, J., Ong, S. P., Persson, K. & Jain, A. (2017). "Atomate: A high-level interface to generate, execute, and analyze computational materials science workflows." *Computational Materials Science* 139, 140–152. atomate (v1) workflow infrastructure reference.
3. Ganose, A. M., Sai Gautam, G., Sherrill, C. D. & Jain, A. (2024). "Atomate2: A high-throughput materials science workflow management framework." *Journal of Open Source Software* (in press/preprint). atomate2 reference; cite when using atomate2.
4. Ong, S. P., Richards, W. D., Jain, A., Hautier, G., Kocher, M., Cholia, S., Gunter, D., Chevrier, V. L., Persson, K. A. & Ceder, G. (2013). "Python Materials Genomics (pymatgen): A robust, open-source python library for materials analysis." *Computational Materials Science* 68, 314–319. pymatgen reference.
5. Rosen, A. S., Gallant, M., George, J., Riebesell, J., Sasaki, H., Shen, J.-X., Sherrill, C. D. & Ganose, A. M. (2024). "jobflow: Computational workflows made simple." *Journal of Open Source Software* 9, 5995. jobflow reference.
6. Boes, J. R., Groenenboom, M. C., Keith, J. A. & Kitchin, J. R. (2016). "Neural network and ReaxFF comparison for Au properties." *International Journal of Quantum Chemistry* 116, 979–987. Example of custodian-style error handling in automated DFT.
7. Curtarolo, S., Setyawan, W., Hart, G. L. W., Jahnatek, M., Chepulskii, R. V., Taylor, R. H., Wang, S., Xue, J., Yang, K., Levy, O., Mehl, M. J., Stokes, H. T., Demchenko, D. O. & Morgan, D. (2012). "AFLOW: An automatic framework for high-throughput materials discovery." *Computational Materials Science* 58, 218–226. AFLOW HT-DFT framework; alternative to atomate2.
8. Pizzi, G., Cepellotti, A., Sabatini, R., Marzari, N. & Kozinsky, B. (2016). "AiiDA: Automated interactive infrastructure and database for computational science." *Computational Materials Science* 111, 218–230. AiiDA provenance-first HT-DFT infrastructure; preferred for cross-institutional reproducibility.

[EXPERT REVIEW NEEDED: Verify that atomate2 version 0.0.14 is still current; the package is under active development and API changes frequently. Confirm whether `MaterialsProject2020Compatibility` should be applied to SCAN results or only to GGA/GGA+U results. Verify the custodian handler list covers all common VASP 6.x failure modes, particularly new error strings introduced in VASP 6.3+.]
