# Cross-Skill Map

This document maps the connections between skills across pillars. It is generated manually during the authoring process and regenerated automatically by `scripts/generate_catalog.py` (planned feature).

---

## How to read this map

Each entry shows a source skill and the downstream or laterally connected skills it feeds into. The connection type explains why the link matters in a realistic research workflow.

---

## Current connections

### electronic-structure/vasp-workflow

| Connected skill | Connection type | Why it matters |
|---|---|---|
| `electronic-structure/dft-convergence` | prerequisite | ENCUT and k-point convergence must be established before production runs |
| `ml-interatomic-potentials/mace-training` | feeds into | VASP DFT energies and forces form the training dataset |
| `ml-interatomic-potentials/active-learning-mlp` | feeds into | DFT single-points are the oracle for active learning loops |
| `atomistic-md/lammps-workflow` | feeds into | Validated ML potentials trained from VASP data are deployed in LAMMPS |
| `scientific-data/materials-featurization` | feeds into | DFT-computed properties are the target variables for materials ML |
| `scientific-data/reproducibility-workflow` | cross-cuts | DFT calculations require careful versioning of POTCAR, INCAR, and pseudopotentials |

---

## Planned connections (to be documented as skills are drafted)

- `atomistic-md/lammps-workflow` → `atomistic-md/md-equilibration` → `atomistic-md/free-energy-methods`
- `ml-interatomic-potentials/mace-training` → `ml-interatomic-potentials/mlp-validation` → `atomistic-md/lammps-workflow`
- `ml-interatomic-potentials/active-learning-mlp` → `surrogate-active-learning/bayesian-optimization`
- `geometric-deep-learning/e3nn-basics` → `ml-interatomic-potentials/mace-training`
- `scientific-ml/deepxde-framework` → `phase-field/moose-phase-field`
- `reaction-thermo/calphad-workflow` → `phase-field/allen-cahn-workflow`
- `cheminformatics/rdkit-workflow` → `generative-matter/molecular-diffusion`
- `surrogate-active-learning/bayesian-optimization` → `scientific-data/reproducibility-workflow`

---

## Regeneration

This file will be auto-generated from `Integration with Other Skills` sections in each SKILL.md once `generate_catalog.py` implements that feature. Until then, it is maintained manually.
