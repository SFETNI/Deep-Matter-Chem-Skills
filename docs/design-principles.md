# Design Principles

These principles govern what a skill in `deep-matter-chem-skills` should and should not be.

---

## 1. Domain-grounded

Every skill must explain the physical, chemical, or mathematical assumptions that the method requires. An AI agent using the skill should understand *why* the method works, not just *how* to call the API.

**What this means in practice:** A DFT skill must explain the exchange-correlation approximation. An MD skill must explain the force field validity domain. A surrogate skill must explain the prior assumptions of the model.

---

## 2. Failure-aware

Plausible-looking but physically invalid results are the primary risk in computational science. Every skill must document failure modes: what goes wrong, what it looks like in the output, what causes it, and how to diagnose it.

**What this means in practice:** A converged-looking energy is not proof of a correct result. Skills must document the silent failures that experienced practitioners know to check.

---

## 3. Research-realistic

Code examples and parameter choices must reflect actual research use, not tutorials. Toy examples are permitted only when they are explicitly labeled as minimal illustrations.

**What this means in practice:** ENCUT = 300 eV is not acceptable as a default without domain justification. Use realistic ranges and note material-class dependence.

---

## 4. Agent-usable

A skill must give an AI coding agent enough context to decide: when to use this method, when not to, how to invoke it correctly, and how to recognize when something went wrong.

**What this means in practice:** Include the "When Not to Use This Skill" section. Annotate when human judgment is required. State which outputs need to be checked.

---

## 5. Cross-linked

Computational science is multiscale and multi-tool. A DFT calculation feeds phonon calculations, ML potential training datasets, and thermodynamic integrations. Skills should document these connections explicitly.

**What this means in practice:** Each skill should name the skills upstream and downstream of it in realistic workflows.

---

## 6. Version-aware

Tool APIs change. A workflow that worked in VASP 5.4 may silently produce different results in VASP 6.x. A skill must record the versions it was tested against and flag known incompatibilities.

**What this means in practice:** Fill `tested_versions` in the metadata. Note VASP, ASE, pymatgen, or LAMMPS version-specific behaviors where relevant.

---

## 7. Honest about limits

Every method has a validity domain. Skills must state when the method is inappropriate, when it will produce unreliable results, and when a different approach should be used instead.

**What this means in practice:** Include the "When Not to Use This Skill" section. Do not oversell coverage. Mark uncertain values with `[EXPERT REVIEW NEEDED]` until they are confirmed.

---

## Anti-patterns

- **Toy parameters** — Using ENCUT = 200 eV, k-point mesh = 1x1x1, or EDIFF = 1e-4 in examples without explicit labeling.
- **Silent failures** — Documenting only the happy path without failure diagnostics.
- **False universality** — Claiming a parameter value is always correct across material classes.
- **Shallow API snippets** — Code that shows how to call a function but not what the result means or when to trust it.
- **Missing limits** — Omitting the "When Not to Use This Skill" section.
