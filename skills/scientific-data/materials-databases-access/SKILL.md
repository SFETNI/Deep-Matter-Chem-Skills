---
name: materials-databases-access
domain: scientific-data
maturity: draft
tools:
  - mp-api
  - pymatgen
  - requests
  - pandas
  - monty
  - tqdm
tested_versions:
  mp-api: "0.41.2"
  pymatgen: "2024.11.13"
  requests: "2.31.0"
  pandas: "2.2.0"
  monty: "2024.7.30"
  tqdm: "4.66.4"
requires_gpu: false
estimated_runtime: minutes-to-hours
---

# Materials Databases Access

## Description

This skill covers reproducible access to materials databases for computational materials science: Materials Project, AFLOW, OQMD, NOMAD, COD, and ICSD-style crystallographic sources; API authentication, pagination, rate limits, caching, snapshot provenance, structure matching, duplicate detection, energy compatibility, and dataset preparation for screening or ML. Invoke this skill when retrieving structures, entries, energies, band gaps, elastic data, phonons, tasks, or provenance metadata from external materials databases, or when auditing whether database-derived datasets are compatible enough for phase stability, ML training, or screening.

## Domain Context

Materials databases are not neutral collections of facts. Each database encodes choices about DFT code, exchange-correlation functional, pseudopotentials, Hubbard U corrections, structural relaxation protocol, convergence thresholds, duplicate filtering, and post-processing. A formation energy from Materials Project, AFLOW, OQMD, or NOMAD may refer to a different reference state, correction scheme, or calculation generation. Combining them without compatibility checks creates a polished but physically inconsistent dataset.

The most common use cases are structure retrieval, property screening, phase stability analysis, and ML dataset construction. These are different tasks. Structure retrieval can tolerate mixed provenance if each structure is re-relaxed locally. Phase diagrams require energies on one compatible reference scale. ML datasets require clear target definitions, duplicate handling, train/test leakage control, and metadata that makes future re-querying possible.

Database APIs are also moving targets. Documents are added, deprecated, reprocessed, and corrected. The same query issued months apart can return different records. A reproducible workflow records query parameters, API versions, database release or build information where available, result identifiers, task identifiers, timestamps, and raw response caches. Without this, an ML benchmark or screening campaign cannot be reconstructed.

Crystallographic databases such as COD and ICSD-style sources carry additional complications: disorder, partial occupancies, experimental temperature/pressure, measurement uncertainty, missing hydrogens, non-standard settings, and structure duplicates. These sources are excellent for candidate generation and comparison to experiment, but DFT-ready structures usually require cleaning, ordering, oxidation-state checks, and standardization before calculation.

## When to Use This Skill

- Querying Materials Project, AFLOW, OQMD, NOMAD, COD, or ICSD-style sources for structures, energies, band gaps, elastic tensors, phonons, entries, tasks, or metadata.
- Building a reproducible local cache of database records for screening, ML featurization, or high-throughput DFT input generation.
- Joining records across databases using structure matching, formula matching, database IDs, or provenance metadata.
- Constructing phase diagrams, convex hulls, and formation-energy tables from compatible database entries.
- Auditing a database-derived dataset for duplicates, outdated snapshots, missing metadata, incompatible units, or mixed energy references.
- Preparing database-derived train/validation/test sets for materials ML or surrogate modeling.

## When Not to Use This Skill

- **When the intended result requires a single internally consistent DFT reference but records come from multiple databases.** Recompute locally or use one compatibility scheme rather than mixing raw energies.
- **When experimental disorder is central to the material.** COD/ICSD-style structures with partial occupancies or disorder require explicit ordering or statistical modeling before DFT or ML use.
- **When API results are used without local caching.** A query that is not cached with IDs and metadata cannot be reproduced if the database changes.
- **When structure duplicates would leak into train/test splits.** Database records often contain polymorph duplicates, near-duplicates, and reprocessed tasks; deduplicate before splitting.
- **When task-level provenance is missing.** Material-level summaries can hide which calculation produced each property. For critical workflows, retrieve task IDs, calculation settings, and source metadata.

## Core Concepts

- **Material document:** A database-level summary for a material, often aggregating many calculations or experimental records. It may contain canonical structure, formula, energy, band gap, elasticity, phonons, and provenance pointers.
- **Task document:** A calculation-level record: one DFT run, relaxation, static calculation, band structure, phonon calculation, or workflow step. Task-level data are required when auditing settings and provenance.
- **Computed entry:** A composition-energy object with correction metadata used to build phase diagrams. Entries are only comparable if they share a correction and reference scheme.
- **Compatibility scheme:** A transformation applied to raw DFT energies to make them internally comparable, such as Materials Project compatibility corrections for GGA/GGA+U mixing.
- **Database snapshot:** The effective state of a database at query time: API version, release version where available, query date, filters, fields, and returned IDs. Store it with every derived dataset.
- **Pagination:** APIs return large result sets in pages or chunks. Failure to iterate through all pages silently truncates datasets.
- **Rate limit:** API-imposed limit on request frequency or total requests. Respecting limits requires batching, backoff, and caching.
- **Structure matching:** Symmetry- and tolerance-aware comparison of structures to detect duplicates or map entries across databases. Formula equality alone is not enough.
- **Provenance metadata:** IDs, task IDs, code versions, functional, pseudopotentials, correction scheme, source URL, query filters, and timestamps that explain where a record came from.
- **Target definition:** The precise property being modeled or screened, including units and provenance. For example, "MP corrected formation energy per atom" is not the same as "raw VASP total energy per atom."

## Key Workflows

### Workflow 1: Materials Project query with caching, pagination, and provenance

Use the Materials Project API through `mp-api`, request only required fields, cache raw results, and store the query metadata. API keys should come from environment variables or local config, never from committed source files.

```python
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os
from typing import Any

from monty.json import MontyEncoder
from mp_api.client import MPRester


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def query_materials_project_summary(
    output_dir: str,
    elements: list[str],
    fields: list[str] | None = None,
    theoretical: bool | None = None,
    band_gap_min_eV: float | None = None,
    chunk_size: int = 1000,
) -> list[dict[str, Any]]:
    """
    Query MP summary documents and cache raw monty-serialized records.

    Example use: retrieve oxide structures and summary properties for screening.
    Store both records and query metadata so the dataset can be reconstructed.
    """
    fields = fields or [
        "material_id",
        "formula_pretty",
        "structure",
        "energy_above_hull",
        "formation_energy_per_atom",
        "band_gap",
        "is_stable",
        "theoretical",
        "calc_types",
        "origins",
        "last_updated",
    ]

    api_key = os.environ.get("MP_API_KEY")
    if not api_key:
        raise RuntimeError("Set MP_API_KEY in the environment; do not commit API keys")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    query = {
        "elements": elements,
        "fields": fields,
        "theoretical": theoretical,
        "band_gap": (band_gap_min_eV, None) if band_gap_min_eV is not None else None,
    }
    query = {k: v for k, v in query.items() if v is not None}
    query_hash = sha256_text(json.dumps(query, sort_keys=True, default=str))[:16]

    with MPRester(api_key) as mpr:
        docs = mpr.materials.summary.search(
            elements=elements,
            fields=fields,
            theoretical=theoretical,
            band_gap=(band_gap_min_eV, None) if band_gap_min_eV is not None else None,
            chunk_size=chunk_size,
        )
        records = [doc.model_dump() if hasattr(doc, "model_dump") else doc.as_dict() for doc in docs]

    raw_file = out / f"mp_summary_{query_hash}.json"
    raw_file.write_text(json.dumps(records, cls=MontyEncoder, indent=2))

    metadata = {
        "database": "Materials Project",
        "api_package": "mp-api",
        "query_time_utc": datetime.now(timezone.utc).isoformat(),
        "query": query,
        "query_hash": query_hash,
        "n_records": len(records),
        "raw_file": str(raw_file),
        "raw_file_sha256": hashlib.sha256(raw_file.read_bytes()).hexdigest(),
        "notes": [
            "Material summary documents may aggregate multiple task documents.",
            "For critical provenance, retrieve task-level records using calc_types/origins.",
        ],
    }
    (out / f"mp_summary_{query_hash}.metadata.json").write_text(json.dumps(metadata, indent=2, default=str))
    print(f"Cached {len(records)} MP records at {raw_file}")
    return records
```

### Workflow 2: Task and provenance audit for selected materials

Material-level summaries are convenient, but property provenance often lives at the task level. Use task IDs and origins to verify which calculations generated a band gap, structure, elastic tensor, or phonon object.

```python
from pathlib import Path
from datetime import datetime, timezone
import json
import os

from monty.json import MontyEncoder
from mp_api.client import MPRester


def audit_mp_material_tasks(
    material_ids: list[str],
    output_file: str,
    task_fields: list[str] | None = None,
) -> list[dict]:
    """
    Retrieve task provenance for MP materials.

    Use this before combining properties where task type matters, e.g. mixing
    relaxation energies, static energies, band gaps, and elastic tensors.
    """
    task_fields = task_fields or [
        "task_id",
        "material_id",
        "task_type",
        "run_type",
        "orig_inputs",
        "output",
        "last_updated",
        "state",
    ]
    api_key = os.environ.get("MP_API_KEY")
    if not api_key:
        raise RuntimeError("Set MP_API_KEY in the environment")

    records = []
    with MPRester(api_key) as mpr:
        for mid in material_ids:
            tasks = mpr.materials.tasks.search(
                material_ids=[mid],
                fields=task_fields,
            )
            for task in tasks:
                d = task.model_dump() if hasattr(task, "model_dump") else task.as_dict()
                records.append(d)

    out = Path(output_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "query_time_utc": datetime.now(timezone.utc).isoformat(),
        "material_ids": material_ids,
        "task_fields": task_fields,
        "n_task_records": len(records),
        "records": records,
    }, cls=MontyEncoder, indent=2))
    print(f"Wrote {len(records)} task records to {output_file}")
    return records


def summarize_task_types(task_audit_json: str) -> dict:
    """Summarize task types and run types from a cached task audit."""
    data = json.loads(Path(task_audit_json).read_text())
    counts: dict[str, int] = {}
    run_types: dict[str, int] = {}
    for record in data["records"]:
        counts[record.get("task_type", "unknown")] = counts.get(record.get("task_type", "unknown"), 0) + 1
        run_types[record.get("run_type", "unknown")] = run_types.get(record.get("run_type", "unknown"), 0) + 1
    print("Task types:", counts)
    print("Run types:", run_types)
    return {"task_types": counts, "run_types": run_types}
```

### Workflow 3: Cross-database retrieval with pagination and local cache

For AFLOW, OQMD, NOMAD, and COD-style endpoints, use explicit pagination and write raw responses before normalization. Endpoint details change, so keep this pattern and adapt parameters to the current API documentation.

```python
from pathlib import Path
from datetime import datetime, timezone
import json
import time
import requests


def paginated_get_json(
    base_url: str,
    params: dict,
    output_dir: str,
    page_param: str = "page",
    per_page_param: str = "per_page",
    per_page: int = 100,
    max_pages: int | None = None,
    sleep_s: float = 0.25,
    headers: dict | None = None,
) -> list:
    """
    Generic paginated JSON retriever for materials APIs.

    Adapt page_param/per_page_param and termination logic to each database.
    Always inspect the first response manually before trusting the loop.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    all_records = []
    page = 1

    while True:
        request_params = dict(params)
        request_params[page_param] = page
        request_params[per_page_param] = per_page
        response = requests.get(base_url, params=request_params, headers=headers, timeout=60)
        response.raise_for_status()
        payload = response.json()

        # Common API shapes: list records directly, or dict with data/results.
        if isinstance(payload, list):
            records = payload
        else:
            records = payload.get("data", payload.get("results", []))

        page_file = out / f"page_{page:05d}.json"
        page_file.write_text(json.dumps(payload, indent=2, default=str))
        all_records.extend(records)

        print(f"Page {page}: {len(records)} records")
        if not records:
            break
        if max_pages is not None and page >= max_pages:
            break
        if len(records) < per_page:
            break
        page += 1
        time.sleep(sleep_s)

    metadata = {
        "base_url": base_url,
        "params": params,
        "page_param": page_param,
        "per_page_param": per_page_param,
        "per_page": per_page,
        "n_pages": page,
        "n_records": len(all_records),
        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "warning": "Verify pagination semantics against the database documentation.",
    }
    (out / "retrieval_metadata.json").write_text(json.dumps(metadata, indent=2, default=str))
    (out / "records_flat.json").write_text(json.dumps(all_records, indent=2, default=str))
    return all_records


def normalize_common_database_fields(records: list[dict], source: str) -> list[dict]:
    """
    Normalize only stable fields; keep raw records for source-specific details.

    This deliberately avoids pretending all databases have equivalent metadata.
    """
    normalized = []
    for i, rec in enumerate(records):
        normalized.append({
            "source": source,
            "source_index": i,
            "source_id": rec.get("id") or rec.get("auid") or rec.get("entry_id") or rec.get("material_id"),
            "formula": rec.get("formula") or rec.get("compound") or rec.get("pretty_formula"),
            "energy_per_atom_eV": rec.get("energy_per_atom") or rec.get("enthalpy_atom"),
            "band_gap_eV": rec.get("band_gap") or rec.get("Egap"),
            "structure_available": any(k in rec for k in ("structure", "cif", "crystal", "geometry")),
        })
    return normalized
```

### Workflow 4: Duplicate detection and cross-database joining

Join database records conservatively. Formula matching finds candidates; structure matching decides whether two records likely describe the same structure.

```python
from collections import defaultdict
import pandas as pd
from pymatgen.core import Structure
from pymatgen.analysis.structure_matcher import StructureMatcher


def load_structure_from_record(record: dict) -> Structure | None:
    """
    Convert a normalized/raw record to a pymatgen Structure when possible.

    Implement source-specific adapters outside this helper and keep failures
    explicit rather than silently dropping records.
    """
    if "structure" in record and isinstance(record["structure"], dict):
        return Structure.from_dict(record["structure"])
    if "cif" in record and record["cif"]:
        from pymatgen.io.cif import CifParser
        parser = CifParser.from_str(record["cif"])
        return parser.parse_structures(primitive=False)[0]
    return None


def find_duplicate_structures(
    records: list[dict],
    ltol: float = 0.2,
    stol: float = 0.3,
    angle_tol: float = 5.0,
) -> list[list[int]]:
    """
    Group likely duplicate structures using formula bins plus StructureMatcher.

    Tolerances should be tightened for relaxed DFT structures and loosened
    carefully for experimental structures with uncertainty. [EXPERT REVIEW NEEDED]
    """
    matcher = StructureMatcher(ltol=ltol, stol=stol, angle_tol=angle_tol)
    by_formula: dict[str, list[tuple[int, Structure]]] = defaultdict(list)

    for i, rec in enumerate(records):
        try:
            s = load_structure_from_record(rec)
        except Exception as exc:
            print(f"WARNING record {i}: cannot parse structure: {exc}")
            continue
        if s is None:
            continue
        by_formula[s.composition.reduced_formula].append((i, s))

    duplicate_groups = []
    for formula, items in by_formula.items():
        assigned = set()
        for i, s in items:
            if i in assigned:
                continue
            group = [i]
            assigned.add(i)
            for j, t in items:
                if j not in assigned and matcher.fit(s, t):
                    group.append(j)
                    assigned.add(j)
            if len(group) > 1:
                duplicate_groups.append(group)
                print(f"{formula}: duplicate group {group}")
    return duplicate_groups


def join_property_tables_on_structure(
    left_records: list[dict],
    right_records: list[dict],
) -> pd.DataFrame:
    """
    Join two database result sets with conservative structure matching.
    """
    combined = left_records + right_records
    groups = find_duplicate_structures(combined)
    rows = []
    n_left = len(left_records)
    for group in groups:
        left = [idx for idx in group if idx < n_left]
        right = [idx - n_left for idx in group if idx >= n_left]
        if left and right:
            rows.append({
                "left_indices": left,
                "right_indices": right,
                "left_ids": [left_records[i].get("source_id") for i in left],
                "right_ids": [right_records[i].get("source_id") for i in right],
            })
    return pd.DataFrame(rows)
```

### Workflow 5: Energy compatibility and phase-stability workflow

Use a single compatibility scheme for phase stability. If entries come from Materials Project, use MP-compatible entries directly; if entries come from local DFT or another database, do not mix them with MP corrected entries unless a documented correction scheme is applied.

```python
from pathlib import Path
import json
import pandas as pd
from pymatgen.analysis.phase_diagram import PhaseDiagram, PDEntry
from pymatgen.core import Composition


def phase_diagram_from_compatible_table(
    entries_csv: str,
    output_dir: str,
    energy_col: str = "energy_eV",
    formula_col: str = "formula",
    correction_scheme: str = "single-source-raw",
) -> pd.DataFrame:
    """
    Build a convex hull from one compatible entry table.

    Required columns:
        formula, energy_eV, source_id, source_database
    Optional:
        correction_eV, calculation_type, functional
    """
    df = pd.read_csv(entries_csv)
    required = {formula_col, energy_col, "source_id", "source_database"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    sources = sorted(df["source_database"].dropna().unique())
    if len(sources) > 1 and correction_scheme == "single-source-raw":
        raise ValueError(
            "Multiple source databases found with no compatibility scheme. "
            "Do not build a hull from mixed raw energies."
        )

    entries = []
    for _, row in df.iterrows():
        energy = float(row[energy_col]) + float(row.get("correction_eV", 0.0))
        entries.append(PDEntry(Composition(row[formula_col]), energy, name=str(row["source_id"])))

    pdia = PhaseDiagram(entries)
    rows = []
    for entry in entries:
        rows.append({
            "source_id": entry.name,
            "formula": entry.composition.reduced_formula,
            "energy_per_atom_eV": float(entry.energy_per_atom),
            "formation_energy_per_atom_eV": float(pdia.get_form_energy_per_atom(entry)),
            "energy_above_hull_eV_atom": float(pdia.get_e_above_hull(entry)),
            "is_stable": bool(pdia.get_e_above_hull(entry) < 1e-8),
        })

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(rows).sort_values("energy_above_hull_eV_atom")
    result.to_csv(out / "phase_stability.csv", index=False)
    (out / "phase_stability_metadata.json").write_text(json.dumps({
        "entries_csv": entries_csv,
        "n_entries": len(entries),
        "source_databases": sources,
        "correction_scheme": correction_scheme,
        "warning": "Hull is valid only for entries sharing a compatible energy reference.",
    }, indent=2))
    return result
```

### Workflow 6: Database-derived ML dataset preparation

Prepare a screening or ML dataset with explicit targets, duplicate handling, provenance, and split safeguards. The target definition must be specific enough that future queries can reproduce it.

```python
from pathlib import Path
import json
import pandas as pd


def prepare_database_ml_table(
    normalized_records: list[dict],
    output_csv: str,
    target_col: str,
    required_metadata: list[str] | None = None,
) -> pd.DataFrame:
    """
    Convert normalized records into an ML-ready table with metadata checks.

    This function does not featurize structures. It prepares a clean target
    table for use by materials-featurization and downstream ML workflows.
    """
    required_metadata = required_metadata or [
        "source",
        "source_id",
        "formula",
        "structure_available",
    ]
    df = pd.DataFrame(normalized_records)

    missing = [c for c in required_metadata + [target_col] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    before = len(df)
    df = df.dropna(subset=[target_col, "formula", "source_id"])
    after_drop = len(df)
    df = df.drop_duplicates(subset=["source", "source_id"])
    after_dedup_id = len(df)

    # Do not drop structural duplicates here unless structures are available;
    # use StructureMatcher workflow first for structure-aware deduplication.
    df["target_definition"] = target_col
    df.to_csv(output_csv, index=False)

    metadata = {
        "output_csv": output_csv,
        "target_col": target_col,
        "n_input": before,
        "n_after_dropna": after_drop,
        "n_after_id_dedup": after_dedup_id,
        "required_metadata": required_metadata,
        "warnings": [
            "Run structure-aware duplicate detection before train/test splitting.",
            "Do not mix target definitions across databases without compatibility checks.",
        ],
    }
    Path(output_csv).with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2))
    return df


def audit_ml_table_for_leakage_risk(csv_file: str, target_col: str) -> dict:
    """Flag common database-derived ML leakage and quality risks."""
    df = pd.read_csv(csv_file)
    risks = []
    if df["formula"].duplicated().mean() > 0.05:
        risks.append("Many duplicate formulas; use composition- or structure-aware splitting")
    if "energy_above_hull" in df.columns and target_col in {"is_stable", "stability_label"}:
        risks.append("energy_above_hull directly determines stability target; do not use it as an input feature")
    if "source" in df.columns and df["source"].nunique() > 1:
        risks.append("Multiple database sources; check target compatibility and source leakage")
    if df[target_col].isna().any():
        risks.append("Target contains NaN values")
    report = {
        "csv_file": csv_file,
        "target_col": target_col,
        "n_rows": len(df),
        "n_sources": int(df["source"].nunique()) if "source" in df.columns else None,
        "risks": risks,
    }
    print(json.dumps(report, indent=2))
    return report
```

## Common Failure Modes

| Failure | Symptom | Likely Cause | Diagnostic | Fix |
|---------|---------|--------------|------------|-----|
| Mixed incompatible energy references | Convex hull has surprising stable phases; formation energies disagree with database values | Raw entries from different databases, functionals, U settings, or correction schemes were combined | Print source database, functional, correction, and run type per entry | Use one compatible entry set or a documented correction scheme; recompute locally if needed |
| Duplicate structures in ML dataset | Test performance is unrealistically high; train/test records share near-identical structures | Multiple database IDs represent the same or symmetry-equivalent structure | Run `StructureMatcher` duplicate grouping before splitting | Deduplicate or split by structure group/prototype/composition family |
| Outdated database snapshot | Results cannot be reproduced from the same query later | Query date, API version, and returned IDs were not stored | Compare current query IDs to cached IDs | Cache raw responses and metadata; cite database release/build where available |
| Missing metadata | Records have structures but no task IDs, correction scheme, or provenance | Summary endpoint requested too few fields or source lacks provenance | Audit required fields before dataset export | Re-query task/provenance endpoints; exclude records with critical missing metadata |
| Wrong task selection | Band gap, energy, or structure comes from an unintended task type | Material summary aggregates multiple tasks; chosen property origin not checked | Inspect task types, origins, and task IDs | Select task-level records explicitly for critical properties |
| Unstable entries treated as stable | Screening list includes high-energy metastable phases as stable candidates | Used `is_stable` from one database incorrectly, or ignored `energy_above_hull` units | Check energy above hull threshold and source definition | Define stability threshold explicitly, e.g. `E_hull < 0.025 eV/atom` [EXPERT REVIEW NEEDED] |
| API pagination loss | Dataset size is smaller than expected; alphabetically late compositions missing | Only first API page was read | Compare reported total count to cached record count; inspect page files | Implement pagination loop and stop only on documented termination condition |
| Silent unit mismatch | Elastic moduli, energies, volumes, or band gaps differ by powers of ten | Database fields use different units or per-cell vs per-atom conventions | Recompute one value manually from raw fields | Normalize units into explicit column names before joining |
| Poor provenance tracking | Cannot identify which query produced a model training set | Only processed CSV was saved, not raw records or query metadata | Search for raw cache and metadata files | Store raw JSON, normalized table, query metadata, IDs, hashes, and script commit |
| COD/ICSD disorder mishandled | DFT input has partial occupancies or impossible stoichiometry | Experimental disordered CIF was converted directly to POSCAR | Check `structure.is_ordered` and species occupancies | Generate ordered approximants or exclude disordered entries from DFT-ready datasets |

## Best Practices

- **Cache raw API responses before normalization.** The raw response is the only faithful record of what the database returned.
- **Record query metadata with every dataset.** Include query filters, fields, API package version, database release if available, timestamp, result count, IDs, and file hashes.
- **Use task-level provenance for critical properties.** Material summaries are convenient; task records explain how properties were computed.
- **Never build phase diagrams from mixed raw energies.** Energy compatibility is a scientific requirement, not a formatting issue.
- **Deduplicate before splitting ML datasets.** Use structure matching and formula/prototype grouping to avoid train/test leakage.
- **Keep source-specific fields until late in the pipeline.** Premature normalization can erase the metadata needed to debug incompatibilities.
- **Respect API rate limits.** Use batching, backoff, and local caches; do not repeatedly query the same large dataset.
- **Treat experimental crystallographic structures as candidates, not DFT-ready inputs.** Check disorder, occupancies, missing species, cell conventions, and oxidation states.
- **Use explicit target definitions.** Record whether the target is raw total energy, corrected formation energy, MP band gap, task-level GGA gap, experimental gap, or another quantity.
- **Version the preprocessing script.** Database retrieval, filtering, deduplication, and target assembly should be reproducible from a clean checkout.

## Integration with Other Skills

- **[pymatgen-analysis](../pymatgen-analysis/SKILL.md):** Database records are usually converted into pymatgen structures, entries, phase diagrams, and structure-matching workflows before analysis.
- **[materials-featurization](../materials-featurization/SKILL.md):** Database-derived structures and compositions feed featurization pipelines; duplicate detection and target provenance must be settled before feature generation.
- **[reproducibility-workflow](../reproducibility-workflow/SKILL.md):** Query caches, raw responses, normalized tables, hashes, and retrieval scripts must be versioned like any other computational artifact.
- **[scientific-visualization](../scientific-visualization/SKILL.md):** Screening results, convex hulls, property distributions, and database coverage maps need clear visual encodings and provenance-aware captions.
- **[vasp-workflow](../../electronic-structure/vasp-workflow/SKILL.md):** Database structures often become VASP inputs for re-relaxation or validation; local VASP settings should replace database assumptions when recomputing properties.
- **[dft-convergence](../../electronic-structure/dft-convergence/SKILL.md):** Any local recomputation of database-derived structures must use converged settings appropriate to the chemistry and target property.
- **[dft-defects](../../electronic-structure/dft-defects/SKILL.md):** Defect workflows often start from database host structures but require careful supercell, charge-state, and chemical-potential provenance. This is an intentional forward reference until that skill is added.
- **[high-throughput-dft](../../electronic-structure/high-throughput-dft/SKILL.md):** Database-derived screening lists are common inputs to high-throughput DFT campaigns. This is an intentional forward reference until that skill is added.
- **[gaussian-process-materials](../../surrogate-active-learning/gaussian-process-materials/SKILL.md):** Database-derived feature tables can seed Gaussian-process surrogate models for screening and active learning. This is an intentional forward reference until that skill is added.
- **[bayesian-optimization](../../surrogate-active-learning/bayesian-optimization/SKILL.md):** Database priors and initial observations can initialize Bayesian optimization campaigns. This is an intentional forward reference until that skill is added.
- **[small-data-ml](../../scientific-ml/small-data-ml/SKILL.md):** Small curated database extracts require leakage-aware splits, uncertainty estimates, and provenance-rich target definitions. This is an intentional forward reference until that skill is added.

## Key References

1. Jain, A., et al. (2013). *Commentary: The Materials Project: A materials genome approach to accelerating materials innovation.* APL Materials, 1, 011002. https://doi.org/10.1063/1.4812323
2. Curtarolo, S., et al. (2012). *AFLOW: An automatic framework for high-throughput materials discovery.* Computational Materials Science, 58, 218-226. https://doi.org/10.1016/j.commatsci.2012.02.005
3. Kirklin, S., et al. (2015). *The Open Quantum Materials Database (OQMD): assessing the accuracy of DFT formation energies.* npj Computational Materials, 1, 15010. https://doi.org/10.1038/npjcompumats.2015.10
4. Draxl, C., & Scheffler, M. (2019). *NOMAD: The FAIR concept for big data-driven materials science.* MRS Bulletin, 43, 676-682. https://doi.org/10.1557/mrs.2018.208
5. Gražulis, S., et al. (2012). *Crystallography Open Database (COD): an open-access collection of crystal structures and platform for world-wide collaboration.* Nucleic Acids Research, 40, D420-D427. https://doi.org/10.1093/nar/gkr900
