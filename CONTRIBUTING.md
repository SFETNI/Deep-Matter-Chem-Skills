# Contributing to deep-matter-chem-skills

Thank you for contributing. This project aims for research-realistic, failure-aware skill content — not shallow tutorials or API walkthroughs.

---

## Before you start

Read these two documents first:

- [docs/design-principles.md](docs/design-principles.md) — what a skill must be and what it must not be
- [docs/maturity-model.md](docs/maturity-model.md) — the five maturity tiers and what each requires

The skill catalog is at [docs/skills-catalog.md](docs/skills-catalog.md). Check it before proposing a new skill to avoid duplication.

---

## Types of contribution

### New skills

Skills not yet in the repository. New contributions must target at minimum `draft` maturity. See the [Draft requirements](#draft-requirements) section below.

### Skill improvements

Corrections to numerical claims, convergence thresholds, tool defaults, failure mode descriptions, or best-practice recommendations in existing skills. Improvements do not need to change the maturity tier unless all remaining `EXPERT REVIEW NEEDED` markers are resolved.

### Reference and DOI corrections

If a reference title, DOI, author list, year, or journal does not match the actual publication, open an issue or submit a correction PR. Reference integrity is part of the review surface. See [Reference integrity](#reference-integrity).

### Expert review

If you have domain expertise in one of the covered pillars, review a draft skill and either confirm its claims or flag incorrect values. Replace `[EXPERT REVIEW NEEDED]` markers with either verified values or documented corrections. A skill with zero remaining markers and at least one domain-expert review is a candidate for promotion to `validated`.

### Roadmap suggestions

If you have a skill idea that fits the project's scope but does not yet exist, open an issue. Label it with the pillar name and a short description of the proposed skill.

---

## Creating a new skill

### Step 1 — Choose a pillar and name

Skills live under `skills/<pillar>/<skill-name>/SKILL.md`. The pillar must match one of the directories in `skills/`. Skill names use lowercase kebab-case. Use the most specific pillar that fits; if no pillar fits, propose a new one in an issue before proceeding.

### Step 2 — Copy the template

```bash
cp scripts/skill_template.md skills/<pillar>/<skill-name>/SKILL.md
```

### Step 3 — Fill in the metadata block

The YAML front matter at the top of every `SKILL.md` is required. All fields must be present:

```yaml
---
name: skill-name           # kebab-case, matches directory name
domain: pillar-name        # must match the parent pillar directory
maturity: draft            # planned | stub | draft | validated | flagship
tools:
  - PrimaryTool
  - SupportingTool
tested_versions:
  PrimaryTool: "x.y.z"    # exact version string; use "unknown" only if untested
requires_gpu: false        # true or false
estimated_runtime: minutes-to-hours
---
```

### Step 4 — Fill in all required sections

Every `draft` skill must have all ten section headings present with content. Missing or empty sections will cause validation to fail.

| Section | Requirement |
|---------|-------------|
| `## Description` | 1–3 sentences: what the skill does, which tools it uses, when to invoke it |
| `## Domain Context` | Physical, chemical, or mathematical assumptions; validity domain |
| `## When to Use This Skill` | Specific system types, problem regimes, expected outputs |
| `## When Not to Use This Skill` | Cases where the method is inappropriate, misleading, or overkill |
| `## Core Concepts` | Non-optional concepts a practitioner must understand |
| `## Key Workflows` | At least one workflow with research-realistic code |
| `## Common Failure Modes` | Table with ≥ 3 rows: Failure, Symptom, Likely Cause, Diagnostic, Fix |
| `## Best Practices` | Non-obvious practical rules that prevent silent errors |
| `## Integration with Other Skills` | At least one cross-link to an existing skill |
| `## Key References` | At least 3 primary sources with DOIs or URLs |

### Step 5 — Mark uncertain values

Any numerical value, parameter recommendation, or domain claim that you are not confident about must be tagged:

```
[EXPERT REVIEW NEEDED: brief description of what needs review]
```

Do not leave unmarked uncertain values. It is better to flag uncertainty explicitly than to present an unverified claim as fact.

### Step 6 — Add cross-references correctly

All integration links must use relative paths and point to a `SKILL.md` that exists in the repository at the time of the PR:

```markdown
- **[skill-name](../../pillar/skill-name/SKILL.md):** Reason for the connection.
```

Do not link to skills that do not yet exist. Mention planned future skills in prose instead. The cross-reference checker (`scripts/check_crossrefs.py`) will reject broken links at validation time.

### Step 7 — Validate

```bash
# Validate the new skill
python scripts/validate_skills.py skills/<pillar>/<skill-name>/SKILL.md

# Regenerate the catalog
python scripts/generate_catalog.py

# Verify no cross-reference links are broken
python scripts/check_crossrefs.py

# Validate all skills to confirm nothing was broken
for f in $(find skills -name "SKILL.md" | sort); do
    python scripts/validate_skills.py "$f" || exit 1
done
```

All four commands must pass with no errors before opening a PR.

### Step 8 — Open a pull request

Stage only the new skill folder and the regenerated catalog:

```bash
git add skills/<pillar>/<skill-name>/SKILL.md
git add docs/skills-catalog.md
git commit -m "Add <skill-name> skill"
```

Do not stage unrelated files, cache artifacts, or IDE configuration.

---

## Draft requirements

A skill at `draft` maturity must have:

- All ten required sections present and non-empty
- At least one `## Key Workflows` entry with a code block
- A failure modes table with at least 3 rows
- At least 3 references with DOIs or URLs
- At least 1 cross-link to an existing skill
- Zero validation errors from `scripts/validate_skills.py`
- Zero broken cross-references from `scripts/check_crossrefs.py`

`draft` does not mean scientifically validated. It means structurally complete and internally consistent. Uncertain claims must be marked `[EXPERT REVIEW NEEDED]`.

---

## Improving an existing skill

To correct a numerical value, threshold, or description:

1. Make the targeted change only — do not rewrite unrelated sections.
2. If you are resolving a `[EXPERT REVIEW NEEDED]` marker, replace the marker with the correct value and add a brief inline note or reference explaining the source.
3. Re-run validation and the cross-reference checker.
4. If the change resolves all remaining `[EXPERT REVIEW NEEDED]` markers in the skill, note this in your PR description — the skill may be a candidate for promotion to `validated`.

Do not silently remove `[EXPERT REVIEW NEEDED]` markers without providing the reviewed value.

---

## Reference integrity

References in skills are part of the scientific review surface. Before submitting or reviewing a skill:

- Verify that each DOI resolves to the correct paper.
- Verify that the title, author list, year, and journal name match the actual publication.
- Verify that the claim supported by the reference is actually made in that paper.

If you find a broken DOI, incorrect title, wrong year, or misattributed claim, open an issue with:

- The file path (`skills/<pillar>/<skill-name>/SKILL.md`)
- The exact reference line number
- The corrected citation

A broken-reference issue template will be added to `.github/ISSUE_TEMPLATE/` in the next release phase.

---

## Maturity promotion

To propose promoting a skill from `draft` to `validated`:

1. Resolve all `[EXPERT REVIEW NEEDED]` markers with verified values.
2. Ensure the failure modes table has at least 5 rows with credible, specific content.
3. Ensure at least one workflow has been tested against the declared tool versions.
4. Verify all references and cross-links.
5. Open a PR changing `maturity: draft` to `maturity: validated` and describe the review performed.

Promotion is never automatic. A maintainer must confirm the review before merging.

See [docs/maturity-model.md](docs/maturity-model.md) for the full promotion criteria at each tier.

---

## What not to submit

- Shallow tutorials or getting-started guides that do not encode domain knowledge
- Skills with toy parameters in code examples (e.g., 2×2×2 k-mesh for a metal, timestep of 10 fs for an all-atom protein)
- Skills with fewer than 3 references
- Skills that link to non-existent skills in the repository
- Broad reformatting or style-cleanup PRs that mix content changes with formatting changes
- Skill content copied from documentation websites without attribution or domain commentary

---

## Style and tone

- Write for a postdoc or research engineer who knows the domain but may not know the specific tool.
- Explain *why* a parameter matters, not just *what* to set it to.
- Use precise units and conditions: "60 Ry for norm-conserving pseudopotentials in PBE" rather than "use a high cutoff".
- Keep prose tight. Long paragraphs without structure are harder to use as agent context.
- Do not add emojis, marketing language, or hype.

---

## Running the full quality check

Before any PR, confirm that the repository is in a consistent state:

```bash
python scripts/generate_catalog.py
python scripts/check_crossrefs.py
for f in $(find skills -name "SKILL.md" | sort); do
    python scripts/validate_skills.py "$f" || exit 1
done
git status
```

Expected: catalog regenerates cleanly, zero broken cross-references, all skills pass, working tree clean except for your intended changes.

---

## Questions and discussions

Open a GitHub issue for questions, scope discussions, or pillar proposals. Issue templates for common contribution types (new skill, reference correction, expert review request) are being prepared as part of the release-readiness phase.
