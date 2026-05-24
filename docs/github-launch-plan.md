# GitHub Launch Plan

## Purpose

This document defines the launch plan for the first public milestone of `deep-matter-chem-skills`. It records what is ready, what still needs review, how the community can contribute, and what comes next.

---

## Current milestone

- **52 draft skills** across 15 active pillars
- Structurally validated — all skills pass `scripts/validate_skills.py`
- Zero broken internal cross-references — confirmed by `scripts/check_crossrefs.py`
- Public release candidate, not final scientific validation
- Expert and domain review still needed for most skills; `[EXPERT REVIEW NEEDED]` markers are present throughout

---

## What is ready

| Item | Status |
|------|--------|
| Skill template (`scripts/skill_template.md`) | Ready |
| Schema validator (`scripts/validate_skills.py`) | Ready |
| Cross-reference checker (`scripts/check_crossrefs.py`) | Ready |
| Catalog generator (`scripts/generate_catalog.py`) | Ready |
| Generated skill catalog (`docs/skills-catalog.md`) | Ready — 52 skills indexed |
| Maturity model (`docs/maturity-model.md`) | Ready |
| Design principles (`docs/design-principles.md`) | Ready |
| `README.md` | Ready — honest milestone wording |
| `CONTRIBUTING.md` | Ready — full contribution workflow documented |
| GitHub issue templates (5 templates + config) | Ready |
| Pull request template | Ready |
| Release checklist (`docs/release-checklist.md`) | Ready |
| Multi-pillar technical coverage (15 active pillars) | Ready |

---

## What still needs review

These items do not block the initial push but should be tracked as open work:

- **Scientific correctness** — numerical values, convergence thresholds, tool defaults, and best-practice recommendations in draft skills have not been independently reviewed
- **`[EXPERT REVIEW NEEDED]` markers** — present throughout; highest density in `atomistic-md` (coarse-grained-md: 16, gromacs-workflow: 15) and `ml-interatomic-potentials` (universal-potentials: 10)
- **Reference integrity** — DOI, title, author list, year, and journal name consistency has not been systematically audited across all 52 skills
- **Code examples and parameter defaults** — research-realistic but not all tested against declared tool versions
- **Maturity labels** — all 52 skills are currently `draft`; no skill has been promoted to `validated` or `flagship`
- **Pillar taxonomy and roadmap wording** — two pillars (`generative-matter`, `reaction-thermo`) have 0 or 1 skills; the broader ~110-skill roadmap is not yet documented in the repository
- **Flagship skill quality** — no skill has been designated as a `flagship` reference; the highest-quality skills (e.g., `vasp-workflow`, `gromacs-workflow`, `high-throughput-dft`) are candidates but have not been reviewed at that standard

---

## Community contribution model

Contributions are welcome in the following modes:

| Mode | Description |
|------|-------------|
| **New skill proposals** | Propose and implement skills not yet in the repository; use the new-skill-proposal issue template |
| **Skill improvements** | Correct numerical values, parameter defaults, code examples, or domain claims; use the skill-content-error template |
| **Reference corrections** | Fix DOI mismatches, wrong titles, incorrect author lists, or unsupported citations; use the broken-reference template |
| **Domain expert review** | Confirm or correct claims in a draft skill; resolve `[EXPERT REVIEW NEEDED]` markers; use the expert-review-request template |
| **Validation and tooling** | Improve `validate_skills.py`, `check_crossrefs.py`, or `generate_catalog.py`; report false positives via the schema-validation-failure template |
| **Roadmap suggestions** | Propose new pillars, skill names, or taxonomy changes via a GitHub issue |

**Cross-reference policy:** Skills that are planned but not yet implemented must be mentioned in prose. Do not add `[text](path/SKILL.md)` links until the target file exists — `check_crossrefs.py` will reject broken links.

---

## Recommended announcement text

> `deep-matter-chem-skills` is now available on GitHub.
>
> The repository contains 52 draft skills for AI coding agents working in computational chemistry, materials modeling, atomistic simulation, ML interatomic potentials, scientific ML, continuum modeling, thermodynamics, sustainability, electrochemistry, and related domains. All skills pass structural validation and internal cross-references are clean.
>
> Skills are at `draft` maturity. They are structurally complete and internally consistent, but have not been systematically expert-reviewed. `[EXPERT REVIEW NEEDED]` markers identify specific claims that require domain confirmation.
>
> Expert review, reference corrections, skill improvements, and new skill contributions are welcome. See `CONTRIBUTING.md` for how to contribute.

---

## Post-launch priorities

1. **Audit references in flagship-candidate skills** — systematically verify DOIs and titles for the highest-traffic skills before promoting any to `validated`
2. **Triage `[EXPERT REVIEW NEEDED]` markers** — categorize by urgency (blocking promotion vs. informational) and open tracking issues per pillar
3. **Recruit domain reviewers** — identify and contact domain experts for each active pillar; the `atomistic-md` and `ml-interatomic-potentials` pillars are highest priority given marker density
4. **Expand roadmap toward ~110 skills** — document the full planned taxonomy in a `ROADMAP.md` file; prioritize pillars with strong existing cross-links
5. **Add CI for validators** — add a GitHub Actions workflow that runs `validate_skills.py`, `check_crossrefs.py`, and `generate_catalog.py` on every PR; block merges if any check fails
6. **Decide maturity promotion criteria** — confirm whether the maturity model in `docs/maturity-model.md` is sufficient or needs a formal review protocol
7. **Automated reference checks** — evaluate tools (doi.org API, CrossRef API) for batch DOI verification; feasibility depends on rate limits and skill count
8. **Skill development in balanced waves** — prioritize completing pillars with 1–2 skills before adding more in already-dense pillars

---

## Roadmap toward ~110 skills

The current 52 skills provide substantial coverage in electronic structure, atomistic MD, and ML potentials. The main gaps toward ~110 skills are:

| Direction | Likely additions |
|-----------|-----------------|
| Computational chemistry | Multireference methods, reaction path workflows, solvation models, NMR/spectroscopy |
| Cheminformatics | Reaction informatics, SMILES/SMARTS workflows, retrosynthesis tools |
| Continuum mechanics | Plasticity, fatigue, creep, fracture mechanics, homogenization |
| Reaction thermodynamics and kinetics | Cantera workflows, microkinetic modeling, TST/RRKM |
| Electrochemistry and catalysis | Electrocatalysis descriptors, Butler-Volmer, explicit solvent DFT |
| Process modeling and sustainability | LCA integration, TEA workflows, process simulation |
| Generative matter | Molecule generation, inverse design, property-conditioned generation |
| Scientific ML and operator learning | DeepONet, Fourier neural operators, graph-based surrogates |
| Experimental-data integration | EXAFS, XRD, PDF, experimental featurization pipelines |
| Materials informatics | High-throughput screening workflows, feature importance, Pareto fronts |

---

## Launch checklist summary

Before pushing to GitHub, work through:

- `docs/release-checklist.md` — full pre-tag and post-tag checklist
- `CONTRIBUTING.md` — confirm contribution workflow is complete
- `.github/pull_request_template.md` — confirm PR template renders on GitHub
- `.github/ISSUE_TEMPLATE/` — confirm issue templates render in the GitHub issue chooser
- `.github/ISSUE_TEMPLATE/config.yml` — confirm contact links use the actual GitHub repository URL and `main` branch

---

## Known non-blockers

These are acceptable at the time of the initial public release:

- All 52 skills remain at `draft` maturity — this is accurate and is communicated honestly in `README.md`
- Many numerical claims and parameter recommendations require domain expert confirmation — `[EXPERT REVIEW NEEDED]` markers make this explicit
- Future reference drift or newly reported DOI issues should be handled through the broken-reference issue template
- Roadmap skills are intentionally not implemented — only skills with a committed `SKILL.md` are linked internally
