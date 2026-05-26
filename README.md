# Deep Matter Chem Skills

A production-oriented skill library for AI coding agents working in computational chemistry, materials modeling, atomistic simulation, scientific machine learning, continuum modeling, thermodynamics, sustainability, electrochemistry, and related domains.

Each skill encodes domain knowledge that a coding agent needs to operate more reliably in a technical workflow: what physical assumptions a method requires, when to use it, when not to use it, how to diagnose common failures, and how it connects to adjacent workflows.

This is a **research workflow support library**. It is designed to support workflow design, code generation, debugging, reproducibility checks, and technical review. It is **not** an autonomous scientific decision layer and does not replace domain expertise, convergence testing, experimental validation, or final scientific judgment.

---

## Current Status — v0.2 Reference-Audited Production-Support Release 

**52 production-usable support skills** across 15 active pillars.

Current repository checks:

- **52 skills** indexed in the generated catalog.
- **All skills pass structural validation** with `scripts/validate_skills.py`.
- **Zero broken internal cross-references** with `scripts/check_crossrefs.py`.
- **Reference and DOI audit completed** — all ~230 references checked across 52 skills; 8 wrong DOIs corrected, 4 metadata fields corrected.
- **GitHub issue templates and PR checklist are available** for community review and contribution.
- **Longer-term roadmap:** approximately 100+ skills across additional technical domains.

### Two distinct quality tiers

| Term | Meaning |
|---|---|
| **production-usable support skill** | Structurally complete, reference-audited, cross-reference clean, and usable with Claude Code, Codex, or manual workflows. Internal maturity: `draft`. |
| **validated skill** | Independently reviewed by a domain expert for scientific content and promoted beyond `draft`. No skills have reached this tier yet — community review is open. |

### What "production-usable support skill" means in practice

A skill is considered production-usable when it is structurally complete, internally consistent, reference-audited, cross-linked to existing relevant skills, and ready to load into Claude Code, Codex, or another coding-agent workflow.

This does **not** mean every technical claim has completed independent expert review. Internal maturity labels and `[EXPERT REVIEW NEEDED]` markers are used to make review status explicit.

In practical terms:

- Skills are ready to support workflow generation, code review, diagnostics, and reproducibility scaffolding.
- Skills provide structured domain guidance, not final scientific authority.
- Technical claims involving thresholds, software-version behavior, parameter choices, or contested best practices may still require specialist review.
- Skills do not replace expert judgment, convergence testing, experimental validation, or final scientific decisions.
- Contributors are encouraged to report reference issues, correct parameter guidance, and request domain review.

---

## Who This Is For

This repository is intended for:

- **Computational chemists and materials scientists** using AI coding agents for DFT, MD, thermodynamics, or electrochemistry workflows.
- **Atomistic simulation researchers** working with LAMMPS, GROMACS, VASP, Quantum ESPRESSO, CP2K, AiiDA, or related tools.
- **Scientific ML researchers** building ML interatomic potentials, PINNs, surrogate models, or active-learning pipelines.
- **Data scientists** working with materials databases, featurization pipelines, or reproducibility workflows.
- **Domain reviewers** who can confirm, improve, or correct technical claims.
- **Contributors** who want to add new skills, improve existing ones, verify references, or expand the roadmap.

---

## Quickstart

```bash
git clone https://github.com/SFETNI/Deep-Matter-Chem-Skills.git
cd Deep-Matter-Chem-Skills

# Regenerate the skill catalog
python scripts/generate_catalog.py

# Check all internal cross-references
python scripts/check_crossrefs.py

# Validate a single skill
python scripts/validate_skills.py skills/electronic-structure/vasp-workflow/SKILL.md

# Validate all skills
for f in $(find skills -name "SKILL.md" | sort); do
    python scripts/validate_skills.py "$f" || exit 1
done
```

---

## How to Use These Skills with AI Coding Agents

Skills are plain Markdown files. Use them by loading the relevant `SKILL.md` files into the context of Claude Code, Codex, or another coding agent before asking the agent to generate, review, or debug workflow code.

### Claude Code

Open Claude Code in the repository root and ask it to read the relevant skills before working.

Example prompt:

```text
Read skills/electronic-structure/vasp-workflow/SKILL.md and any linked prerequisite skills.
Use them as support material for workflow design and code generation.
Do not treat them as final authority.
Preserve [EXPERT REVIEW NEEDED] markers and flag assumptions that require domain review.
```

For repository edits, ask Claude Code to run the validation scripts before committing:

```text
After editing, run generate_catalog.py, check_crossrefs.py, and validate all changed skills.
Report files changed, validation status, catalog count, and cross-reference status.
```

### Codex or Other Coding Agents

Open the repository in the agent workspace and explicitly point the agent to the relevant skills.

Example prompt:

```text
Before writing code, inspect:
- skills/ml-interatomic-potentials/mace-training/SKILL.md
- skills/ml-interatomic-potentials/mlp-validation/SKILL.md
- skills/scientific-data/reproducibility-workflow/SKILL.md
Use these skills as implementation guidance.
Report any assumptions, uncertain claims, and required validation steps.
```

For changes to the repository, require the agent to report:

* Changed files.
* Validation commands run.
* Catalog count.
* Cross-reference status.
* Any new `[EXPERT REVIEW NEEDED]` markers.
* Any references added or corrected.

### Manual Use Without an Agent

1. Open [docs/skills-catalog.md](docs/skills-catalog.md).
2. Identify the skills relevant to your workflow.
3. Read the relevant `SKILL.md` files.
4. Use the workflows, failure modes, best practices, and references as structured checklists.
5. Check assumptions and parameter choices against your system, software version, and domain requirements.

---

![How It Works](docs/assets/workflow-diagram.png)

The skills support reasoning, implementation, review, and diagnostics. They do not replace expert judgment, physical validation, convergence testing, or experimental confirmation.

---

## Pillars Currently Covered

| Pillar                       | Skills | Domain coverage                                                                                                    |
| ---------------------------- | -----: | ------------------------------------------------------------------------------------------------------------------ |
| `electronic-structure`       |     11 | VASP, Quantum ESPRESSO, CP2K, AiiDA, DFT convergence, phonons, surfaces, defects, high-throughput DFT, EOS fitting |
| `atomistic-md`               |      9 | GROMACS, LAMMPS, force fields, equilibration, free energy, enhanced sampling, coarse-grained MD, reactive MD       |
| `ml-interatomic-potentials`  |      8 | MACE, NequIP, universal potentials, fine-tuning, active learning, validation, uncertainty, LAMMPS deployment       |
| `scientific-data`            |      6 | ASE, pymatgen, materials databases, featurization, visualization, reproducibility                                  |
| `geometric-deep-learning`    |      3 | Crystal GNNs, molecular GNNs, MatGL/M3GNet                                                                        |
| `surrogate-active-learning`  |      3 | Gaussian processes, Bayesian optimization, surrogate validation                                                    |
| `scientific-ml`              |      2 | PINNs, small-data ML                                                                                               |
| `electrochemistry-corrosion` |      2 | Battery materials workflows, corrosion and electrochemistry                                                        |
| `cheminformatics`            |      2 | Molecular featurization, molecular datasets                                                                        |
| `thermodynamics-calphad`     |      1 | CALPHAD workflows                                                                                                  |
| `sustainability-lca`         |      1 | Materials life-cycle assessment                                                                                    |
| `phase-field-microstructure` |      1 | Cahn-Hilliard and Allen-Cahn workflows with FiPy                                                                   |
| `continuum-modeling`         |      1 | FEM multiphysics                                                                                                   |
| `computational-chemistry`    |      1 | Molecular quantum chemistry workflows with ORCA, xTB, TDDFT                                                       |
| `generative-matter`          |      1 | Crystal diffusion generative models, including MatterGen and DiffCSP                                               |

---

## Skill Catalog

The full catalog is available at [docs/skills-catalog.md](docs/skills-catalog.md).

The catalog is generated automatically and lists every skill with its pillar, maturity, tools, GPU requirement, and estimated runtime.

Regenerate after adding or modifying skills:

```bash
python scripts/generate_catalog.py
```

---

## Maturity Model

Skills use five internal maturity labels:

```text
planned → stub → draft → validated → flagship
```

The current release uses `draft` as the entry-level production-support tier. In this repository, `draft` means:

* the skill is structurally complete;
* required sections and metadata are present;
* workflows, failure modes, and references are included;
* internal links resolve;
* the skill remains open to independent expert review.

`draft` does **not** mean the skill is unusable. It means the skill has not yet been promoted through documented external review.

See [docs/maturity-model.md](docs/maturity-model.md) for promotion criteria and tier definitions.

---

## Review Status

The repository is intentionally transparent about review status.

| Review surface            | Current status                        |
| ------------------------- | ------------------------------------- |
| Structural validation     | Passing for all 52 skills             |
| Internal cross-references | Zero broken skill links               |
| Catalog generation        | 52 skills indexed                     |
| Expert review             | Ongoing; community review welcome     |
| Reference and DOI audit   | Completed for v0.2; corrections still welcome         |
| Maturity promotions       | No skills promoted beyond `draft` yet |

Many skills intentionally contain `[EXPERT REVIEW NEEDED]` markers. These markers identify parameter choices, software-version details, reference-sensitive claims, and domain-specific recommendations that benefit from additional expert review.

The project owner has reviewed selected skills in areas aligned with his expertise, including materials modeling, atomistic simulation, phase-field methods, FEM, CALPHAD, electrochemistry/corrosion, sustainability, and scientific ML workflows. Other skills remain open for specialist review.

---

## Quality Checks

Three scripts cover the main verification surface:

| Script                        | What it checks                                                    |
| ----------------------------- | ----------------------------------------------------------------- |
| `scripts/validate_skills.py`  | Schema, required sections, heading names, and front-matter fields |
| `scripts/generate_catalog.py` | Renders all skills to `docs/skills-catalog.md`                    |
| `scripts/check_crossrefs.py`  | Confirms all internal `SKILL.md` links resolve to existing files  |

Recommended checks before committing:

```bash
python scripts/generate_catalog.py
python scripts/check_crossrefs.py
for f in $(find skills -name "SKILL.md" | sort); do
    python scripts/validate_skills.py "$f" || exit 1
done
```

---

## Reference Integrity

Citation and DOI metadata are part of the review surface.

If a reference title, DOI, author list, year, journal, or landing page does not match what is claimed in a skill, report it using the GitHub issue template for broken references:

[Report a reference issue](https://github.com/SFETNI/Deep-Matter-Chem-Skills/issues/new/choose)

Useful reports include:

* skill path;
* exact reference text;
* DOI or URL currently listed;
* expected paper title or authors;
* actual page opened;
* whether the reference supports a technical claim.

---

## Roadmap

The current release contains 52 skills. The longer-term roadmap targets approximately 100+ skills across additional computational and materials-science domains.

Priority expansion areas include:

| Direction                            | Likely additions                                                                        |
| ------------------------------------ | --------------------------------------------------------------------------------------- |
| Computational chemistry              | Multireference methods, reaction path workflows, solvation models, NMR and spectroscopy |
| Cheminformatics                      | Reaction informatics, SMILES/SMARTS workflows, retrosynthesis tools                     |
| Reaction thermodynamics and kinetics | Cantera workflows, microkinetic modeling, TST/RRKM                                      |
| Electrochemistry and catalysis       | Electrocatalysis descriptors, Butler-Volmer workflows, explicit-solvent DFT             |
| Continuum mechanics                  | Plasticity, fatigue, creep, fracture mechanics, homogenization                          |
| Process modeling and sustainability  | LCA integration, TEA workflows, process simulation                                      |
| Generative matter                    | Molecule generation, inverse design, property-conditioned generation                    |
| Scientific ML and operator learning  | DeepONet, Fourier neural operators, graph-based surrogates                              |
| Experimental-data integration        | EXAFS, XRD, PDF, experimental featurization pipelines                                  |
| Materials informatics                | High-throughput screening, feature importance, Pareto-front analysis                    |

Planned roadmap areas may be mentioned in prose before they exist as skills. Internal `SKILL.md` links should point only to skills that already exist in the repository.

---

## Contributing

Contributions are welcome.

Useful contributions include:

* new skills in covered or uncovered pillars;
* improvements to workflows, examples, or failure-mode tables;
* corrections to numerical claims, convergence thresholds, or tool defaults;
* DOI and reference title corrections;
* expert reviews of skills containing `[EXPERT REVIEW NEEDED]` markers;
* roadmap suggestions and domain coverage proposals;
* validation, catalog, or tooling improvements.

GitHub issue templates and a pull request template are available. See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution rules, skill authoring requirements, and the review process.

---

## Release and Launch Notes

Release preparation materials are available in:

* [docs/release-checklist.md](docs/release-checklist.md)
* [docs/github-launch-plan.md](docs/github-launch-plan.md)

These documents describe the release checks, launch workflow, and post-launch priorities.

---

## License

See [LICENSE](LICENSE).
