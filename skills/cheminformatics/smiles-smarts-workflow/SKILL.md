---
name: smiles-smarts-workflow
domain: cheminformatics
maturity: draft
tools:
  - RDKit
  - openbabel
  - pandas
  - numpy
tested_versions:
  RDKit: "2023.9.4"
  openbabel: "3.1.1"
  pandas: "2.2.0"
  numpy: "1.26.4"
requires_gpu: false
estimated_runtime: seconds-to-minutes
---

# SMILES and SMARTS Workflow

## Description

This skill covers molecular line notations for cheminformatics and molecular machine learning: parsing and validating SMILES; canonical, non-canonical, and isomeric SMILES; stereochemistry handling; aromaticity perception; RDKit sanitization; molecule standardization, salt/fragment handling, neutralization, and tautomer policies; SMARTS substructure queries and recursive SMARTS; reaction SMARTS with atom mapping; duplicate detection; and interoperable export with provenance. Invoke this skill before ingesting molecules into a database or ML pipeline, when building or debugging substructure filters, or when defining the standardization policy that determines molecular identity for a project.

## Domain Context

A SMILES string is a serialization of a molecular graph, not a chemical measurement. It encodes atoms, bonds, connectivity, formal charges, isotopes, and — if written as isomeric SMILES — stereochemistry. It does not encode a conformer ensemble, a protonation microstate at a given pH, a solvent, a counterion's crystallographic role, or a measurement context. The first scientific decision when handling line notations is therefore *what molecular identity means for the task*: whether a salt and its parent, two tautomers, two protonation states, or two resonance forms should be treated as the same entity.

Representation identity is not the same as structure identity. Two different SMILES strings can encode the same molecule, and string equality between SMILES is neither necessary nor sufficient for chemical identity. Canonicalization produces a single deterministic SMILES for a graph *within one toolkit and one version*, which makes canonical SMILES useful as a deduplication key inside a controlled pipeline. It is not a portable universal identifier: canonical SMILES from RDKit and from Open Babel need not agree, and canonical algorithms can change across releases. For cross-toolkit or archival identity, standardized InChI/InChIKey is a safer key, though InChI has its own normalization rules for tautomers, charges, and stereochemistry. [EXPERT REVIEW NEEDED]

Aromaticity and valence are model-dependent. RDKit applies its own aromaticity perception and sanitization (valence checks, Kekulization, ring perception, implicit-hydrogen assignment) that can differ from Daylight, Open Babel, or OpenSMILES aromaticity models. A structure that parses cleanly in one toolkit may fail sanitization in another, and an aromatic-lowercase SMILES written under one model may not round-trip under another. Sanitization failures are data-quality signals, not nuisances to be silenced. [EXPERT REVIEW NEEDED]

SMARTS is a distinct query language layered on the same graph model. It expresses atom and bond query primitives, logical operators, and recursive environments, and it is only as reliable as its author's control over aromaticity, hydrogen counts, charge, and ring membership. A functional-group SMARTS that looks correct on a handful of examples is routinely too broad (matching protonated or tautomeric variants unintentionally) or too narrow (missing aromatic or charged forms). Reaction SMARTS adds atom mapping, which lets a transformation carry atoms from reactant to product; corrupted or inconsistent maps silently produce wrong products. Because standardization policy, aromaticity model, and toolkit version all change results, this skill treats those choices as first-class, versioned, and recorded rather than implicit.

## When to Use This Skill

- Parsing and validating SMILES, InChI, or SDF-derived molecules before featurization, database ingestion, or model training.
- Generating canonical or isomeric SMILES as deduplication or storage keys while preserving stereochemistry.
- Defining and applying an explicit standardization policy for salts, fragments, charges, protonation, and tautomers.
- Writing, testing, and debugging SMARTS queries for functional-group detection, structural alerts, or dataset filtering.
- Applying reaction SMARTS with atom mapping to enumerate or transform molecules and validating the products.
- Detecting duplicate molecules and representation inconsistencies before creating train/test splits.
- Exporting interoperable structures (SMILES, SDF, InChI) with a recorded standardization and toolkit provenance trail.

## When Not to Use This Skill

- **When 3D geometry or conformers define the property.** Line notations carry no coordinates; use conformer generation and 3D descriptors instead.
- **When the target depends on a periodic structure.** Crystals, surfaces, and lattice salts are not adequately captured by a single-molecule graph.
- **When the species is not a well-defined molecular graph.** Polymers with distributional composition, undefined R-groups, coordination complexes with ambiguous bonding, and unresolved mixtures need curation or a different representation first.
- **When canonical SMILES is being used as a cross-toolkit or archival identifier.** Use standardized InChIKey for that role and record the generating toolkit and version.
- **When standardization would destroy the signal.** For pH-dependent, redox, or counterion-sensitive properties, aggressive neutralization or salt stripping can remove the very feature being modeled.

## Core Concepts

- **SMILES:** Line notation for a molecular graph. Atoms are element symbols (bracketed when charge, isotope, explicit H count, or non-default valence is needed); bonds are `-`, `=`, `#`, `:`, and default single/aromatic; branches use parentheses; ring closures use matching digits or `%nn`.
- **Aromatic notation:** Lowercase atoms (`c`, `n`, `o`) denote aromatic atoms under the toolkit's aromaticity model. The same ring may be written Kekulized (alternating single/double) or aromatic; round-tripping depends on the model.
- **Charges and isotopes:** Formal charge and isotope are specified inside brackets, e.g. `[NH4+]`, `[13CH4]`. They change molecular identity and mass and must survive standardization intentionally, not accidentally.
- **Atom mapping:** Colon-tagged indices, e.g. `[CH3:1]`, associate atoms across a reaction SMARTS or a mapped transformation. Maps are metadata on the graph, not part of chemical identity.
- **Canonical SMILES:** A deterministic SMILES for a graph produced by a toolkit's canonical ranking. Stable within one toolkit/version; not guaranteed portable across toolkits or releases.
- **Isomeric SMILES:** SMILES that includes stereochemistry (`@`/`@@` tetrahedral parity, `/` and `\` double-bond geometry) and isotopes. Non-isomeric SMILES discards stereochemistry.
- **Stereochemistry:** Tetrahedral chirality at stereocenters and E/Z (cis/trans) configuration at double bonds. Unspecified stereochemistry is a distinct state from specified — it must be tracked, not silently resolved.
- **Sanitization:** RDKit's post-parse consistency pass: valence checks, Kekulization, aromaticity perception, ring finding, conjugation, and implicit-H assignment. Failure indicates an invalid or unsupported graph.
- **SMARTS:** Substructure query language extending SMILES with atom/bond query primitives (`[#6]`, `[R]`, `[!c]`, `[$(...)]`) and logical operators (`,` OR, `;` AND, `!` NOT).
- **Recursive SMARTS:** `$(...)` environments that constrain an atom by its surroundings without those surroundings being part of the returned match.
- **Reaction SMARTS:** `reactants>>products` templates with atom maps that define how atoms transform. Applied with a reaction engine to enumerate products.
- **Standardization:** Project-defined normalization of a molecule into a modeling form: fragment/salt handling, charge normalization, reionization, neutralization, tautomer canonicalization, and stereochemistry policy.
- **Structure vs representation identity:** Whether two records are the same molecule (graph/InChIKey identity) versus whether two strings are byte-equal (representation identity). Deduplication must use the former.

## Key Workflows

### Workflow 1: Parse and validate a molecular dataset

Parse with sanitization but retain failures in an audit table. Never silently drop rows — parse failure rates by source are a data-quality signal.

```python
from dataclasses import dataclass, field
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger

# Capture, do not suppress, RDKit warnings during auditing.
RDLogger.EnableLog("rdApp.*")


@dataclass
class ParseResult:
    source_id: str
    raw: str
    status: str
    canonical_smiles: str | None = None
    inchikey: str | None = None
    problems: list[str] = field(default_factory=list)


def parse_and_check(source_id: str, smiles: str) -> ParseResult:
    if smiles is None or not str(smiles).strip():
        return ParseResult(source_id, str(smiles), "empty")
    text = str(smiles).strip()

    # Parse without sanitizing first so we can inspect the raw graph and
    # run DetectChemistryProblems for a structured diagnostic.
    mol = Chem.MolFromSmiles(text, sanitize=False)
    if mol is None:
        return ParseResult(source_id, text, "unparseable", problems=["MolFromSmiles returned None"])

    problems = [f"{p.GetType()}: {p.Message()}" for p in Chem.DetectChemistryProblems(mol)]
    try:
        Chem.SanitizeMol(mol)
    except Exception as exc:  # sanitization can raise on invalid valence, Kekulization, etc.
        return ParseResult(source_id, text, "sanitize_failed", problems=problems + [f"{type(exc).__name__}: {exc}"])

    return ParseResult(
        source_id,
        text,
        "ok",
        canonical_smiles=Chem.MolToSmiles(mol, isomericSmiles=True),
        inchikey=Chem.MolToInchiKey(mol),
        problems=problems,  # e.g. non-fatal atom-valence notes still recorded
    )


def parse_dataset(df: pd.DataFrame, id_col: str, smiles_col: str, output_csv: str) -> pd.DataFrame:
    results = [parse_and_check(str(r[id_col]), r[smiles_col]) for _, r in df.iterrows()]
    out = pd.DataFrame([{
        "source_id": r.source_id,
        "raw": r.raw,
        "status": r.status,
        "canonical_smiles": r.canonical_smiles,
        "inchikey": r.inchikey,
        "problems": " | ".join(r.problems),
    } for r in results])
    out.to_csv(output_csv, index=False)
    print(out["status"].value_counts(dropna=False))
    return out
```

### Workflow 2: Generate canonical and isomeric SMILES while preserving stereochemistry

Distinguish the storage key (isomeric canonical SMILES) from a stereo-agnostic grouping key (non-isomeric), and count unspecified stereocenters so ambiguity is visible.

```python
from rdkit import Chem
from rdkit.Chem import FindMolChiralCenters


def smiles_identity_record(smiles: str) -> dict:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"status": "parse_failed"}

    # Unassigned stereocenters are a distinct state from assigned ones.
    assigned = FindMolChiralCenters(mol, useLegacyImplementation=False, includeUnassigned=False)
    unassigned = FindMolChiralCenters(mol, useLegacyImplementation=False, includeUnassigned=True)
    n_unspecified = len(unassigned) - len(assigned)

    # Count double bonds with unspecified E/Z geometry.
    n_stereo_double_unspec = sum(
        1 for b in mol.GetBonds()
        if b.GetStereo() == Chem.BondStereo.STEREONONE
        and b.GetBondType() == Chem.BondType.DOUBLE
        and not b.GetIsAromatic()
        and b.GetBeginAtom().GetDegree() > 1 and b.GetEndAtom().GetDegree() > 1
    )

    return {
        "status": "ok",
        # Isomeric canonical SMILES: use as the storage/dedup key when stereo matters.
        "isomeric_canonical": Chem.MolToSmiles(mol, isomericSmiles=True),
        # Non-isomeric canonical SMILES: use only to group stereoisomer families.
        "flat_canonical": Chem.MolToSmiles(mol, isomericSmiles=False),
        "inchikey": Chem.MolToInchiKey(mol),
        "n_assigned_stereocenters": len(assigned),
        "n_unspecified_stereocenters": n_unspecified,
        "n_unspecified_double_bonds": n_stereo_double_unspec,
    }
```

### Workflow 3: Standardize salts, charges, fragments, and tautomers with an explicit policy

Standardization is a modeling choice, not a universal cleanup. Encode the policy explicitly, record every operation applied, and keep the pre-standardization identifier as provenance.

```python
from dataclasses import dataclass
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize


@dataclass
class StandardizationPolicy:
    keep_largest_fragment: bool = True   # strip counterions/solvents; wrong for salt-sensitive targets
    normalize: bool = True               # canonicalize functional-group representations
    reionize: bool = True                # move charges to correct atoms
    neutralize: bool = False             # remove formal charges; can destroy pH-dependent signal
    canonical_tautomer: bool = False     # pick one tautomer; can merge distinct species


def standardize(mol: Chem.Mol, policy: StandardizationPolicy) -> tuple[Chem.Mol | None, list[str]]:
    applied: list[str] = []
    try:
        out = Chem.Mol(mol)
        if policy.keep_largest_fragment:
            out = rdMolStandardize.LargestFragmentChooser().choose(out)
            applied.append("largest_fragment")
        if policy.normalize:
            out = rdMolStandardize.Normalizer().normalize(out)
            applied.append("normalize")
        if policy.reionize:
            out = rdMolStandardize.Reionizer().reionize(out)
            applied.append("reionize")
        if policy.neutralize:
            out = rdMolStandardize.Uncharger().uncharge(out)
            applied.append("neutralize")
        if policy.canonical_tautomer:
            # Tautomer canonicalization is version- and scoring-dependent; it can
            # merge species that are genuinely distinct for the target. [EXPERT REVIEW NEEDED]
            out = rdMolStandardize.TautomerEnumerator().Canonicalize(out)
            applied.append("canonical_tautomer")
        Chem.SanitizeMol(out)
        return out, applied
    except Exception as exc:
        return None, applied + [f"failed:{type(exc).__name__}"]


def standardize_smiles(smiles: str, policy: StandardizationPolicy) -> dict:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"status": "parse_failed", "policy": policy}
    std, applied = standardize(mol, policy)
    if std is None:
        return {"status": "standardize_failed", "operations": applied}
    return {
        "status": "ok",
        "input_smiles": smiles,
        "standard_smiles": Chem.MolToSmiles(std, isomericSmiles=True),
        "standard_inchikey": Chem.MolToInchiKey(std),
        "operations": applied,
    }
```

### Workflow 4: Build and test SMARTS queries for functional-group detection

A SMARTS pattern is only trustworthy once it has been tested against molecules that *should* and *should not* match, including charged, aromatic, and tautomeric variants.

```python
from rdkit import Chem


def compile_smarts(smarts: str) -> Chem.Mol:
    query = Chem.MolFromSmarts(smarts)
    if query is None:
        raise ValueError(f"Invalid SMARTS: {smarts}")
    return query


def test_smarts(smarts: str, should_match: list[str], should_not_match: list[str]) -> dict:
    """Validate a SMARTS query against labeled positive and negative examples."""
    query = compile_smarts(smarts)
    failures = []

    for smi in should_match:
        mol = Chem.MolFromSmiles(smi)
        if mol is None or not mol.HasSubstructMatch(query):
            failures.append(("false_negative", smi))
    for smi in should_not_match:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None and mol.HasSubstructMatch(query):
            failures.append(("false_positive", smi))

    return {"smarts": smarts, "n_failures": len(failures), "failures": failures}


# Example: a carboxylic acid that must match both the protonated and deprotonated forms
# but must NOT match esters. Recursive SMARTS constrains the carbon environment.
CARBOXYLIC_ACID = "[CX3](=O)[OX2H1,OX1-]"

report = test_smarts(
    CARBOXYLIC_ACID,
    should_match=["CC(=O)O", "CC(=O)[O-]", "OC(=O)c1ccccc1"],
    should_not_match=["CC(=O)OC", "CCO", "CC(=O)N"],
)
# Inspect report["failures"] before trusting the pattern on a full dataset.
```

### Workflow 5: Apply reaction SMARTS with atom mapping and product validation

Reaction SMARTS can generate chemically invalid products. Always sanitize each product, check that atom maps are preserved, and deduplicate by InChIKey.

```python
from rdkit import Chem
from rdkit.Chem import AllChem


def run_reaction(reaction_smarts: str, reactant_smiles: list[str]) -> dict:
    rxn = AllChem.ReactionFromSmarts(reaction_smarts)
    if rxn is None:
        raise ValueError(f"Invalid reaction SMARTS: {reaction_smarts}")
    rxn.Initialize()

    reactants = tuple(Chem.MolFromSmiles(s) for s in reactant_smiles)
    if any(m is None for m in reactants):
        raise ValueError("One or more reactants failed to parse")

    valid, invalid = {}, []
    for product_set in rxn.RunReactants(reactants):
        for product in product_set:
            try:
                Chem.SanitizeMol(product)  # generated products are frequently invalid
            except Exception as exc:
                invalid.append((Chem.MolToSmiles(product), f"{type(exc).__name__}: {exc}"))
                continue
            key = Chem.MolToInchiKey(product)
            valid.setdefault(key, Chem.MolToSmiles(product, isomericSmiles=True))

    return {
        "reaction": reaction_smarts,
        "n_valid_unique": len(valid),
        "products": sorted(valid.values()),
        "n_invalid": len(invalid),
        "invalid": invalid,
    }


# Example: ester hydrolysis with explicit atom maps carrying atoms to products.
ESTER_HYDROLYSIS = "[C:1](=[O:2])[O:3][C:4]>>[C:1](=[O:2])[OH:3].[OH][C:4]"
result = run_reaction(ESTER_HYDROLYSIS, ["CC(=O)OCC"])
```

### Workflow 6: Detect duplicates and representation inconsistencies before ML

Deduplicate by a standardized identity key, not by raw SMILES. Report cases where the same molecule appears under different raw strings so representation inconsistencies are visible.

```python
import pandas as pd
from rdkit import Chem


def identity_key(smiles: str, use_stereo: bool = True) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    inchikey = Chem.MolToInchiKey(mol)
    # Full InChIKey distinguishes stereochemistry; the first block (14 chars)
    # groups stereoisomers and some tautomers together.
    return inchikey if use_stereo else inchikey.split("-")[0]


def audit_duplicates(df: pd.DataFrame, smiles_col: str) -> pd.DataFrame:
    work = df.copy()
    work["identity_key"] = work[smiles_col].map(lambda s: identity_key(s, use_stereo=True))
    work["raw_canonical"] = work[smiles_col].map(
        lambda s: Chem.MolToSmiles(Chem.MolFromSmiles(s)) if Chem.MolFromSmiles(s) else None
    )

    groups = work.dropna(subset=["identity_key"]).groupby("identity_key")
    dup_rows, inconsistent = 0, 0
    for _, g in groups:
        if len(g) > 1:
            dup_rows += len(g) - 1
            if g["raw_canonical"].nunique() > 1:
                inconsistent += 1  # same molecule, different raw representations

    print(f"Duplicate rows by InChIKey: {dup_rows}")
    print(f"Identity groups with inconsistent raw SMILES: {inconsistent}")
    return work
```

### Workflow 7: Export interoperable structures and retain provenance

When exchanging with Open Babel or other toolkits, re-parse and re-sanitize on ingest rather than trusting the source aromaticity model. Record the toolkit and version that produced every artifact.

```python
import json
from pathlib import Path
from rdkit import Chem
import rdkit


def export_with_provenance(records: list[dict], out_sdf: str, out_meta: str, policy_name: str) -> None:
    """records: dicts with 'source_id' and 'standard_smiles' from Workflow 3."""
    writer = Chem.SDWriter(out_sdf)
    n_written = 0
    for rec in records:
        mol = Chem.MolFromSmiles(rec["standard_smiles"])
        if mol is None:
            continue
        mol.SetProp("_Name", str(rec["source_id"]))
        mol.SetProp("standard_inchikey", Chem.MolToInchiKey(mol))
        writer.write(mol)
        n_written += 1
    writer.close()

    meta = {
        "n_written": n_written,
        "standardization_policy": policy_name,
        "rdkit_version": rdkit.__version__,
        "identity_key": "InChIKey (standardized)",
        "notes": [
            "Canonical SMILES here are RDKit-canonical and not portable across toolkits.",
            "On ingest from another toolkit, re-parse and re-sanitize before comparison.",
        ],
    }
    Path(out_meta).write_text(json.dumps(meta, indent=2))
    print(f"Wrote {n_written} molecules to {out_sdf}")
```

For Open Babel interoperability, run conversions through the CLI or `pybel` and then re-ingest into RDKit:

```bash
# Convert an SDF to canonical SMILES with Open Babel, then re-canonicalize in RDKit on ingest.
obabel input.sdf -osmi --canonical > openbabel_canonical.smi
```

Open Babel canonical SMILES and RDKit canonical SMILES are produced by different algorithms and aromaticity models and should not be compared as strings. [EXPERT REVIEW NEEDED]

## Common Failure Modes

| Failure | Symptom | Likely Cause | Diagnostic | Fix |
|---------|---------|--------------|------------|-----|
| Silently dropped stereochemistry | Enantiomers/diastereomers collapse to one identity | Non-isomeric SMILES export or a conversion that stripped chiral tags | Compare isomeric vs flat canonical SMILES; count assigned/unassigned centers | Use `isomericSmiles=True`; preserve `@`/`@@` and `/`,`\`; curate ambiguous centers |
| Canonical SMILES treated as universal ID | Cross-source dedup misses or over-merges | RDKit-canonical string compared against another toolkit/version | Regenerate keys with one toolkit; compare to InChIKey | Use standardized InChIKey as the portable identity key |
| Inconsistent aromaticity across toolkits | Structure valid in one tool, fails in another | Different aromaticity/Kekulization models | Re-parse and re-sanitize on ingest; inspect ring perception | Standardize to one model; round-trip test aromatic rings |
| Invalid valence | Sanitization raises AtomValenceException | Malformed input, wrong charge, or bad bond order | `Chem.DetectChemistryProblems` on the unsanitized mol | Fix source record; add explicit H/charge; exclude with report |
| Sanitization failure hidden | Rows vanish before modeling | Exceptions swallowed and mols dropped silently | Parse with `sanitize=False`, then sanitize in a try/except | Keep a failure audit table with the exception per row |
| Incorrect protonation state | Same compound appears with several charges | Mixed pH assumptions or source conventions | Group by parent InChIKey; inspect formal-charge spread | Define an explicit protonation policy; record it in metadata |
| Uncontrolled tautomer canonicalization | Distinct species merge or a non-representative tautomer is chosen | Tautomer canonicalizer applied blindly | Compare pre/post tautomer InChIKeys; review merged groups | Apply tautomer policy deliberately; version the enumerator settings |
| Removed meaningful counterions | Salt-dependent property loses signal | `LargestFragmentChooser` stripped an essential ion | Compare fragment count and formal charge before/after | Keep salt form for salt-sensitive targets; store parent separately |
| Overaggressive neutralization | pH-dependent signal disappears | `Uncharger` applied to a charged-form-relevant dataset | Compare charge distribution before/after | Disable neutralization for redox/pH targets |
| SMARTS too broad | Query matches unintended tautomers/charge states | Missing charge, H-count, or ring constraints | Run positive/negative labeled test set | Add `[CX3]`, `[OX2H1]`, `[R]`, or recursive `$(...)` constraints |
| SMARTS too narrow | Real matches missed | Aromatic/charged/tautomer forms not covered | Test against known-positive variants | Use OR primitives (`,`) for aromatic and ionic forms |
| Atom-map corruption | Reaction produces wrong or unmapped products | Duplicated/inconsistent maps in reaction SMARTS | Inspect map numbers on reactant and product templates | Repair maps so each appears once per side and consistently |
| Mixed toolkit outputs | Duplicates undetected across sources | Canonical strings from different toolkits combined | Recompute all identity keys with one toolkit/version | Normalize every source through one parser before comparison |
| String equality as identity | Duplicates missed or false uniques | Raw SMILES compared byte-for-byte | Compare against InChIKey-based grouping | Deduplicate on standardized identity key, not raw text |

## Best Practices

- **Freeze the standardization policy before splitting or modeling.** Salt handling, charge, protonation, tautomer, and stereochemistry decisions define the dataset and must be recorded.
- **Keep parse and sanitization failures in an audit table.** Silent row drops break label alignment and hide chemistry the parser cannot handle.
- **Separate the storage key from the grouping key.** Use isomeric canonical SMILES or full InChIKey for identity; use non-isomeric keys only to group stereoisomer families intentionally.
- **Do not treat canonical SMILES as portable.** It is stable only within one toolkit and version; use standardized InChIKey for cross-source or archival identity.
- **Test every SMARTS against positive and negative examples.** Include charged, aromatic, and tautomeric variants before running a query on real data.
- **Sanitize every reaction product.** Reaction SMARTS routinely yields invalid graphs; validate and deduplicate products before use.
- **Re-parse on ingest from other toolkits.** Do not trust the source aromaticity model; re-sanitize in one canonical model.
- **Record toolkit versions with artifacts.** RDKit version, standardization policy, and identity-key definition are part of the data provenance.
- **Treat unspecified stereochemistry as data, not noise.** Count and report it rather than letting a canonicalizer resolve it arbitrarily.

## Integration with Other Skills

- **[molecular-featurization](../molecular-featurization/SKILL.md):** Parsed, sanitized, standardized molecules with a fixed stereochemistry and protonation policy are the required input to fingerprint and descriptor generation.
- **[molecular-datasets](../molecular-datasets/SKILL.md):** Dataset curation consumes the identity keys, standardization policy, and duplicate audits defined here to build leakage-aware splits and provenance manifests.
- **[molecular-property-gnn](../../geometric-deep-learning/molecular-property-gnn/SKILL.md):** Molecular graph neural networks depend on correct atom/bond parsing, preserved stereochemistry, and consistent aromaticity perception from this skill.
- **[computational-chemistry-workflow](../../computational-chemistry/computational-chemistry-workflow/SKILL.md):** Standardized SMILES with an explicit protonation and charge state define the starting molecular identity before geometry generation and quantum-chemistry calculations.
- **[small-data-ml](../../scientific-ml/small-data-ml/SKILL.md):** Duplicate detection and representation-consistency checks prevent leakage that would otherwise inflate performance in the small-data regime typical of molecular datasets.
- **[scientific-visualization](../../scientific-data/scientific-visualization/SKILL.md):** SMARTS match highlighting, standardization diffs, and duplicate-group inspection benefit from reproducible molecule depiction and figure conventions.
- **[reproducibility-workflow](../../scientific-data/reproducibility-workflow/SKILL.md):** Standardization policies, SMARTS libraries, toolkit versions, and identity-key definitions are provenance artifacts that must be versioned so curation is auditable.

## Key References

1. Weininger, D. (1988). *SMILES, a chemical language and information system. 1. Introduction to methodology and encoding rules.* Journal of Chemical Information and Computer Sciences, 28, 31-36. https://doi.org/10.1021/ci00057a005
2. Weininger, D., Weininger, A., & Weininger, J. L. (1989). *SMILES. 2. Algorithm for generation of unique SMILES notation.* Journal of Chemical Information and Computer Sciences, 29, 97-101. https://doi.org/10.1021/ci00062a008
3. O'Boyle, N. M. (2012). *Towards a Universal SMILES representation - A standard method to generate canonical SMILES based on the InChI.* Journal of Cheminformatics, 4, 22. https://doi.org/10.1186/1758-2946-4-22
4. O'Boyle, N. M., Banck, M., James, C. A., Morley, C., Vandermeersch, T., & Hutchison, G. R. (2011). *Open Babel: An open chemical toolbox.* Journal of Cheminformatics, 3, 33. https://doi.org/10.1186/1758-2946-3-33
5. Heller, S. R., McNaught, A., Pletnev, I., Stein, S., & Tchekhovskoi, D. (2015). *InChI, the IUPAC International Chemical Identifier.* Journal of Cheminformatics, 7, 23. https://doi.org/10.1186/s13321-015-0068-4
6. Landrum, G. (2024). *RDKit: Open-source cheminformatics software.* https://www.rdkit.org/
7. Daylight Chemical Information Systems. *SMARTS - A Language for Describing Molecular Patterns.* https://www.daylight.com/dayhtml/doc/theory/theory.smarts.html
