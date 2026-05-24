---
name: nequip-training
domain: ml-interatomic-potentials
maturity: draft
tools:
  - NequIP
  - Allegro
  - e3nn
  - ASE
  - PyTorch
  - LAMMPS
  - numpy
  - matplotlib
tested_versions:
  NequIP: "0.7.0"
  Allegro: "0.3.0"
  e3nn: "0.5.1"
  ASE: "3.23.0"
  PyTorch: "2.2.0"
requires_gpu: true
estimated_runtime: hours-to-days
---

# NequIP Training

## Description

This skill covers training, validating, and deploying NequIP-style E(3)-equivariant neural network interatomic potentials from DFT reference data. Invoke it when building a system-specific equivariant potential, comparing NequIP against MACE, DeePMD, CHGNet, M3GNet, or universal potentials, diagnosing unstable MD despite low validation errors, or deciding whether to fine-tune an existing model versus training from scratch.

## Domain Context

NequIP is a local, E(3)-equivariant graph neural network potential. It represents atoms as nodes, neighbor pairs within a finite cutoff as edges, and internal features as irreducible representations (irreps) of the rotation group. The total energy is invariant to translation, rotation, and atom indexing; forces transform equivariantly because they are gradients of the energy with respect to positions.

The main physical approximation is locality: each atomic energy contribution depends on the environment within a cutoff radius and a finite number of message-passing layers. This is appropriate for many condensed-phase, molecular, and materials systems when the DFT training set covers the relevant local environments. It is incomplete for long-range electrostatics, charged defects, polar surfaces, dispersion-dominated systems, and electronic phenomena that are not encoded in the structure or training labels unless the workflow includes special treatment. [EXPERT REVIEW NEEDED]

Equivariance gives NequIP strong data efficiency because the model does not need to learn rotational symmetry from augmented data. This is especially useful for small to medium system-specific datasets with accurate forces. Data efficiency does not remove the need for broad configuration coverage: a model trained only on near-equilibrium structures can still fail catastrophically in MD, surfaces, defects, high-temperature events, or compression/expansion regimes.

NequIP is usually trained on DFT energies and forces, and sometimes stress/virial information where the software version and deployment path support it. Stress support, LAMMPS deployment, Allegro integration, and configuration keys are version-dependent. Mark every production workflow with the exact NequIP, e3nn, PyTorch, ASE, CUDA, and deployment plugin versions. [EXPERT REVIEW NEEDED]

## When to Use This Skill

- Use when training a system-specific equivariant MLP for a defined chemical space, phase, temperature range, and DFT reference level.
- Use when force accuracy and data efficiency are more important than maximum speed or broad pretrained coverage.
- Use when comparing an equivariant local model against MACE, DeePMD, CHGNet, M3GNet, or universal-potential baselines.
- Use when an active learning loop needs committee models, force disagreement, or extrapolation diagnostics for DFT query selection.
- Use when preparing an ASE-deployable model for geometry optimization, phonons, equation-of-state scans, or MD stability tests.
- Use when LAMMPS deployment is required and the target NequIP/Allegro deployment path has been tested on the exact platform. [EXPERT REVIEW NEEDED]

## When Not to Use This Skill

- **When a validated universal potential already meets the target accuracy.** Benchmark CHGNet, M3GNet, MACE-MP, or other universal models before spending DFT budget on a custom NequIP model.
- **When the dataset is tiny and narrow.** Fewer than a few hundred diverse DFT structures can produce attractive validation errors but unstable trajectories unless the deployment domain is extremely limited. [EXPERT REVIEW NEEDED]
- **When the important physics is long-ranged.** Ionic crystals, charged defects, polar surfaces, electrochemical interfaces, and molecular crystals may require explicit electrostatics, larger cells, or alternative models.
- **When the training labels are inconsistent.** Mixed functionals, pseudopotentials, cutoffs, smearing settings, energy references, or stress conventions will be learned as noise or bias.
- **When the deployment target is unsupported.** A trained model is not useful for production MD if it cannot be exported, loaded, or reproduced in the intended ASE or LAMMPS environment.
- **When extrapolation is the main task.** NequIP can interpolate accurately within a covered local-environment distribution but should not be trusted on new chemistry, coordination, or high-energy collisions without active learning and validation.

## Core Concepts

- **E(3)-equivariance:** Model features transform predictably under rotations, translations, and inversion/reflection operations. Scalar energies are invariant; vector forces are equivariant.
- **Irreducible representations (irreps):** Feature channels labeled by angular momentum and parity, such as scalar `0e`, vector `1o`, or tensor-like higher-order features. Wider and higher-order irreps increase expressiveness and cost.
- **Radial basis:** Expansion of pair distances into smooth basis functions used in edge embeddings. The number and cutoff behavior affect resolution of short- and medium-range interactions.
- **Cutoff radius (`r_max`):** Maximum neighbor distance included in the graph. Too small a cutoff excludes relevant interactions; too large a cutoff increases cost and may overfit noisy long-range correlations.
- **Message-passing depth:** Number of interaction layers. More layers enlarge the effective receptive field and body-order complexity, but increase memory, training time, and overfitting risk.
- **Species embeddings:** Learned representations for each element. Unsupported elements cannot be inferred unless included during training or supported by a pretrained checkpoint.
- **Energy reference:** Total DFT energies are extensive. Per-species offsets or isolated-atom energies should be handled consistently so the model learns relative configuration energies rather than arbitrary element baselines.
- **Force-dominated training:** Forces provide many labels per structure and usually dominate MLP training. Loss weights must still preserve correct energy differences if phase stability, formation energies, or free energies matter.
- **Stress and virial labels:** Cell-response targets used for NPT MD, elastic properties, and EOS validation. Support and conventions vary by dataset converter, NequIP version, and deployment route. [EXPERT REVIEW NEEDED]
- **Committee uncertainty:** Multiple models trained on different seeds or data subsets can estimate epistemic disagreement. Committee spread is a heuristic and must be calibrated against held-out DFT errors.

## Key Workflows

### Workflow 1: Prepare a DFT dataset for NequIP

Dataset preparation is the highest-risk step. Every frame must use consistent species, units, atom ordering, energy references, forces, cell, periodic boundary conditions, and optional stress/virial conventions.

```python
from pathlib import Path
import numpy as np
from ase.io import read, write

raw = sorted(Path("dft_relaxations").glob("*/vasprun.xml"))
frames = []

for vasprun in raw:
    atoms = read(vasprun, format="vasp-xml", index=-1)

    if "energy" not in atoms.info:
        raise ValueError(f"Missing energy in {vasprun}")
    if atoms.get_forces().shape != (len(atoms), 3):
        raise ValueError(f"Bad force shape in {vasprun}")

    atoms.info["source"] = str(vasprun.parent)
    atoms.info["dft_code"] = "VASP"
    atoms.info["xc"] = "PBE"
    atoms.info["encut_ev"] = 520
    atoms.info["kpoint_density"] = ">= 1000 per reciprocal atom"
    frames.append(atoms)

write("nequip_dataset.extxyz", frames)
print(f"Wrote {len(frames)} frames")
```

Before training, audit:

- Energy units are eV and forces are eV/A.
- Cell and periodic boundary flags are present for every periodic frame.
- Isolated molecules and periodic solids are not mixed unless the model is intended to handle both.
- Stress or virial fields use the convention expected by the training code.
- Failed, unconverged, or duplicate DFT calculations are excluded or labeled intentionally.
- Train/validation/test splits are stored as explicit file lists, not only random seeds.

### Workflow 2: Split by configuration domain

Random splits often overstate performance because nearby MD frames, relaxation steps, polymorphs, or strained variants leak across train and test sets.

Use split strategies that reflect the claim:

- **Trajectory blocked split:** Hold out full MD time blocks rather than adjacent frames.
- **Composition split:** Hold out selected compositions or chemical systems.
- **Structure-family split:** Hold out phases, defects, surfaces, or polymorph families.
- **Temperature/pressure split:** Hold out thermodynamic conditions to test transfer.
- **Active-learning iteration split:** Keep the final DFT query batch as an external test set.

Example audit:

```python
from ase.io import read
from collections import Counter

frames = read("nequip_dataset.extxyz", ":")
labels = [atoms.info.get("split_group", "unknown") for atoms in frames]
print(Counter(labels))

for split_name in ["train", "val", "test"]:
    subset = [a for a in frames if a.info.get("split") == split_name]
    formulas = Counter(a.get_chemical_formula() for a in subset)
    print(split_name, len(subset), formulas.most_common(10))
```

### Workflow 3: Configure and train a NequIP model

NequIP configuration is usually YAML-based. Exact keys can change between versions; start from a version-matched example configuration and edit deliberately. [EXPERT REVIEW NEEDED]

Research-realistic configuration choices:

```yaml
run_name: nequip_pbe_bulk_defects_v01
seed: 12345
dataset_file_name: nequip_dataset.extxyz
chemical_symbols:
  - Li
  - P
  - S
r_max: 5.0
num_layers: 4
l_max: 2
num_features: 64
parity: true
batch_size: 4
max_epochs: 1000
learning_rate: 0.005
loss_coeffs:
  total_energy:
    - 1.0
    - PerAtomMSELoss
  forces:
    - 100.0
    - PerSpeciesL1Loss
validation_dataset: val
```

Training checks:

- Start with a small dry run to catch parser, species, unit, and device errors.
- Monitor train and validation energy and force losses separately.
- Save checkpoints, configuration, split files, command line, and package versions.
- Compare multiple seeds before trusting a single low-error checkpoint.
- Increase capacity only after ruling out dataset and split problems.
- Do not tune hyperparameters on the final test set.

### Workflow 4: Validate static errors and physical properties

Static metrics are necessary but not sufficient. Validate NequIP models across targets and physical tests.

Minimum validation:

- Energy MAE/RMSE per atom.
- Force MAE/RMSE per Cartesian component and per species.
- Stress/virial error if the model will be used for cell relaxation or NPT MD.
- Residual plots versus energy, force magnitude, volume, composition, and temperature.
- Outlier inspection by structure source and local environment.

Physical validation:

- DFT versus NequIP equation of state for representative phases.
- Relaxed lattice parameters and internal coordinates.
- Defect, surface, or adsorption energies if those are deployment targets.
- Phonons or finite-displacement force consistency for near-equilibrium solids.
- Short NVE trajectories for energy conservation.
- NVT/NPT stability under intended temperatures and pressures.

Low force RMSE does not guarantee stable MD. A model can fit common near-equilibrium forces while producing unphysical forces for rare short bonds, high-energy collisions, or undercoordinated environments.

### Workflow 5: Deploy to ASE or LAMMPS

ASE deployment is often the first reproducibility test because it exercises the exported model outside the training script.

```python
from ase.io import read
from ase.optimize import FIRE

# API names and deployed model formats are version-dependent.
# Verify against the installed NequIP documentation before production use.
from nequip.ase import NequIPCalculator  # [EXPERT REVIEW NEEDED]

atoms = read("candidate_structure.extxyz")
atoms.calc = NequIPCalculator.from_deployed_model(
    model_path="deployed_nequip_model.pth",
    device="cuda",
)

opt = FIRE(atoms, logfile="ase_nequip_relax.log")
opt.run(fmax=0.03, steps=300)
```

Deployment checklist:

- Compare ASE forces against NequIP validation/inference forces on the same frame.
- If exporting to LAMMPS, compare ASE and LAMMPS energies/forces for identical structures before MD.
- Confirm supported elements and neighbor cutoff in the deployed model.
- Verify precision, device, and unit conventions.
- Store the exact deployed model artifact, not only the training checkpoint.

LAMMPS deployment for NequIP/Allegro has changed across releases and may require specific plugin builds, TorchScript/AOTI formats, or package versions. Treat unsupported deployment paths as blockers rather than forcing a conversion. [EXPERT REVIEW NEEDED]

### Workflow 6: Fine-tuning versus training from scratch

Choose fine-tuning when a checkpoint already covers the broad chemistry and only needs adaptation to a narrower DFT setup, surface class, defect class, pressure range, or functional. Choose training from scratch when species, bonding, target convention, or local environments are substantially different.

Fine-tuning safeguards:

- Preserve an unfine-tuned baseline for comparison.
- Use a leakage-resistant external test set.
- Lower learning rates and monitor forgetting on the original domain if relevant.
- Recompute top candidates or unstable trajectories with DFT.
- Record which layers were frozen, which were retrained, and how species embeddings were handled.

Training from scratch is safer when the pretrained model has unsupported elements, incompatible references, unclear training data, or systematically wrong relaxation behavior for the target domain.

## Common Failure Modes

| Failure | Symptom | Likely Cause | Diagnostic | Fix |
|---------|---------|--------------|------------|-----|
| Unit mismatch | Force errors off by factors of 10 or 100; MD explodes immediately | Mixed eV/A, Ha/Bohr, Ry/Bohr, kBar, GPa, or eV/A^3 | Check units on a hand-converted frame | Convert once with tested code and store units in metadata |
| Inconsistent energy references | Energy loss plateaus high; formation trends are biased | Mixed DFT levels, missing offsets, or incompatible isolated atom references | Plot residual by composition | Use one DFT reference convention and explicit per-species offsets |
| Too-small cutoff | Good local force fit but wrong elastic/EOS/defect behavior | Missing second-neighbor or medium-range physics | Sweep `r_max` and compare EOS/phonons | Increase cutoff or add long-range treatment |
| Insufficient angular features | Model misses directional bonding or polymorph ordering | Low `l_max`, too few features, or shallow message passing | Compare errors by coordination motif | Increase equivariant capacity after checking data coverage |
| Overfitting small datasets | Train loss falls while validation and MD stability degrade | Too much capacity or correlated split | Learning curves and blocked splits | Reduce model size, add data, or use active learning |
| Poor high-energy behavior | Stable near equilibrium but fails at high T or compression | Training lacks repulsive and distorted structures | Inspect short-bond force outliers | Add high-energy DFT data with careful weighting |
| Data leakage | Excellent test metrics but poor new-candidate performance | Duplicate structures or adjacent MD frames split across sets | Structure/trajectory/source overlap audit | Split by source, trajectory block, composition, or structure family |
| Unstable MD despite low force RMSE | Energy drift, atom overlap, phase collapse | Rare extrapolative configurations not represented in validation metrics | NVE/NVT tests and outlier frame extraction | Add active-learning queries from failing trajectories |
| Unsupported deployment path | Model trains but cannot run in LAMMPS or production ASE | Version mismatch or unsupported export route | Load exported model in target engine | Pin versions, use supported plugin, or change deployment target |
| Stress convention error | Cell relaxations go the wrong way or NPT density is wrong | Sign or unit mismatch between DFT, ASE, and training | Compare stress on a strained reference cell | Convert stress/virial explicitly and validate against DFT |

## Best Practices

- Keep the DFT reference level fixed across the entire dataset.
- Store split files, not just split fractions or random seeds.
- Validate on chemically meaningful holdouts and not only random frame splits.
- Use multiple seeds or committees when assessing uncertainty and active learning.
- Benchmark against MACE, DeePMD, CHGNet, M3GNet, and universal potentials on the same held-out structures.
- Confirm ASE and LAMMPS deployment consistency before production MD.
- Include high-energy, strained, undercoordinated, and thermalized structures when those regimes matter.
- Use NVE drift, RDFs, phonons, EOS, and relaxation tests to supplement static force errors.
- Treat model uncertainty as a workflow property, not a single number emitted by one checkpoint.
- Version-lock NequIP, e3nn, PyTorch, CUDA, ASE, and deployment plugins.

## Integration with Other Skills

- **[mace-training](../mace-training/SKILL.md):** Compare equivariant training choices, extxyz dataset handling, loss weighting, and deployment tradeoffs.
- **[mlp-validation](../mlp-validation/SKILL.md):** Provides static, physical-property, MD-stability, and ASE/LAMMPS consistency validation protocols.
- **[active-learning-mlp](../active-learning-mlp/SKILL.md):** Supplies query-selection loops for improving NequIP coverage where MD or screening discovers extrapolation.
- **[uncertainty-mlp](../uncertainty-mlp/SKILL.md):** Covers committee disagreement, calibration, and uncertainty-aware deployment decisions.
- **[fine-tuning-mlp](../fine-tuning-mlp/SKILL.md):** Helps decide when to adapt pretrained models versus train NequIP-style models from scratch.
- **[mlp-to-lammps](../mlp-to-lammps/SKILL.md):** Connects deployed NequIP or Allegro models to LAMMPS validation and production constraints.
- **[universal-potentials](../universal-potentials/SKILL.md):** Provides pretrained-model baselines and cautions before spending DFT budget on custom training.
- **[dft-dataset-generation](../../electronic-structure/dft-dataset-generation/SKILL.md):** Defines how to generate consistent energies, forces, stresses, and metadata for training.
- **[dft-convergence](../../electronic-structure/dft-convergence/SKILL.md):** Ensures the DFT oracle is converged enough for MLP labels.
- **[lammps-workflow](../../atomistic-md/lammps-workflow/SKILL.md):** Supports production MD setup once the deployed model has been validated.
- **[ase-framework](../../scientific-data/ase-framework/SKILL.md):** Handles extxyz IO, calculators, optimizers, trajectory manipulation, and deployment smoke tests.
- **[scientific-visualization](../../scientific-data/scientific-visualization/SKILL.md):** Used for parity plots, residual maps, learning curves, force histograms, and MD diagnostics.
- **[reproducibility-workflow](../../scientific-data/reproducibility-workflow/SKILL.md):** Tracks configurations, model artifacts, environments, split files, and provenance.

## Key References

1. Batzner, S. et al. (2022). *E(3)-equivariant graph neural networks for data-efficient and accurate interatomic potentials*. Nature Communications. https://doi.org/10.1038/s41467-022-29939-5
2. Musaelian, A. et al. (2023). *Learning local equivariant representations for large-scale atomistic dynamics*. Nature Communications. https://doi.org/10.1038/s41467-023-36329-y
3. Geiger, M. and Smidt, T. (2022). *e3nn: Euclidean neural networks*. arXiv. https://arxiv.org/abs/2207.09453
4. Drautz, R. (2019). *Atomic cluster expansion for accurate and transferable interatomic potentials*. Physical Review B. https://doi.org/10.1103/PhysRevB.99.014104
5. Smith, J. S. et al. (2017). *ANI-1: an extensible neural network potential with DFT accuracy at force field computational cost*. Chemical Science. https://doi.org/10.1039/C6SC05720A
