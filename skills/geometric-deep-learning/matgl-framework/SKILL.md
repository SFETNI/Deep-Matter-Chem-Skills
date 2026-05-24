---
name: matgl-framework
domain: geometric-deep-learning
maturity: draft
tools:
  - matgl
  - pymatgen
  - DGL
  - PyTorch
  - ASE
  - numpy
  - pandas
  - matplotlib
tested_versions:
  matgl: "1.1.3"
  pymatgen: "2024.11.13"
  dgl: "2.1.0"
  pytorch: "2.2.0"
  ase: "3.23.0"
requires_gpu: true
estimated_runtime: minutes-to-days
---

# MatGL Framework

## Description

Use this skill when building, applying, validating, or fine-tuning MatGL models for materials property prediction, learned interatomic potentials, and graph-based materials workflows. MatGL is most useful when a workflow needs pretrained graph neural network models, pymatgen-compatible structure handling, periodic crystal graph construction, or a bridge between materials databases, DFT datasets, and learned potential deployment.

MatGL should not be treated as a black-box replacement for DFT. Pretrained models can be extremely useful for screening, relaxation preconditioning, or baseline predictions, but their reliability depends on the training domain, target definition, element coverage, structure type, and validation protocol.

## Domain Context

MatGL sits in the materials graph-learning ecosystem alongside M3GNet-style universal potentials, MEGNet-style property models, CHGNet-style charge-informed workflows, and newer pretrained materials models. In practice, it often appears in workflows that start from `pymatgen` `Structure` objects, convert periodic structures into graph representations, apply a pretrained model, and then validate predictions against DFT or experiment.

Useful MatGL tasks include:

- Predicting crystal properties from periodic structures.
- Loading pretrained models for formation energy, band gap, or potential-energy-surface inference where available.
- Relaxing structures using learned energy/force/stress models.
- Fine-tuning pretrained models on project-specific DFT datasets.
- Preparing graph-learning baselines for high-throughput screening.
- Comparing universal-potential relaxations against VASP, Quantum ESPRESSO, or CP2K reference calculations.

## When to Use This Skill

- Use when applying MatGL pretrained models to crystal property prediction, screening, relaxation preconditioning, or potential-energy-surface inference.
- Use when converting pymatgen structures into periodic graph-learning inputs with explicit control over cell representation, cutoffs, neighbor lists, and metadata.
- Use when fine-tuning MatGL or M3GNet-style models on project-specific DFT datasets with energy, force, stress, or scalar property labels.
- Use when comparing MatGL predictions against DFT, high-throughput DFT, universal potentials, or other graph neural network baselines.

## When Not to Use This Skill

MatGL is not automatically appropriate when:

- The chemistry, charge state, defect type, surface, molecule, or pressure/temperature regime is outside the model training distribution.
- The target requires strict DFT-level accuracy for final publication claims.
- The dataset is too small to fine-tune without leakage-resistant validation.
- The structure contains unsupported elements, unusual oxidation states, partial occupancies, disorder, molecular crystals, charged defects, or nonperiodic species that the model was not designed to handle.
- Uncertainty estimates are required but the deployed model does not provide calibrated uncertainty.

## Core Concepts

### Graph Construction From Structures

MatGL workflows usually begin with periodic structures from pymatgen or ASE. Crystal graphs encode atoms as nodes and neighbor relationships as edges, usually with distances, periodic image offsets, and sometimes angular or three-body features depending on the model architecture.

Check these details before trusting predictions:

- Whether the input cell, species, occupancies, oxidation states, and site properties survived file conversion.
- Whether the graph cutoff is appropriate for the model family and target.
- Whether the neighbor list includes the intended periodic images.
- Whether disordered or partially occupied structures were filtered, ordered, or otherwise transformed.
- Whether the structure was standardized to a primitive, conventional, or model-specific cell consistently across the dataset.

Small changes in graph construction can change predictions. For benchmarking, record the MatGL version, model name, graph converter settings, cutoff, and preprocessing code.

### Pretrained Models

Pretrained MatGL models can provide fast screening estimates or initial relaxations. They are most defensible when the query structures resemble the training data and the output is used as a candidate-ranking or preconditioning signal.

Treat model metadata as part of the scientific input:

- Model identifier and checkpoint version.
- Training dataset, if documented.
- Element coverage.
- Target definition and units.
- Whether energies are absolute, per-atom, formation-like, corrected, or referenced to a particular convention.
- Whether forces and stresses were included in training.
- Known benchmark set and reported errors.

[EXPERT REVIEW NEEDED] Public model names, default checkpoints, and loader APIs can change across MatGL releases. Pin versions and save the resolved model metadata with every production run.

### M3GNet, CHGNet, and Related Workflows

M3GNet-style models commonly use graph representations to predict energy, forces, stresses, and sometimes related materials properties for periodic structures. CHGNet-style workflows also use graph neural networks for materials, with charge-informed representations and pretrained potentials in adjacent tooling. These models are related in purpose but not interchangeable: training data, target definitions, graph construction, element coverage, and stress conventions can differ.

When comparing MatGL, CHGNet-style, MACE-style, and classical potentials:

- Compare on the same structures and target definitions.
- Use the same DFT reference settings when possible.
- Separate energy, force, stress, relaxation, and downstream-property metrics.
- Inspect failure cases by chemistry, coordination, volume, and structural motif.
- Avoid calling a model "universal" without stating the validation domain.

## Key Workflows

### Workflow 1: Pretrained Inference

Use pretrained inference for rapid triage, sanity checks, or baseline predictions.

1. Load structures with pymatgen and validate species, lattice, occupancies, and units.
2. Load the intended MatGL pretrained model and record the exact model identifier.
3. Run predictions on a small known reference set before processing a full dataset.
4. Check output units and target definitions.
5. Compare a sample of predictions against DFT, experiment, or trusted literature values.
6. Save inputs, model metadata, predictions, and environment information.

Minimal example pattern:

```python
import matgl
from pymatgen.core import Structure

structure = Structure.from_file("POSCAR")
model = matgl.load_model("M3GNet-MP-2021.2.8-PES")  # verify name for your MatGL version

prediction = model.predict_structure(structure)
print(prediction)
```

Before adapting this pattern, verify the current MatGL API and whether the loaded model expects a structure, graph, state attributes, or an ASE calculator wrapper. [EXPERT REVIEW NEEDED]

### Workflow 2: Structure Relaxation With Learned Potentials

Use MatGL relaxations when screening many candidates, pre-relaxing structures before DFT, or comparing learned-potential geometries against DFT references.

Recommended protocol:

1. Start from well-defined structures with valid lattice vectors and periodic boundary conditions.
2. Filter unsupported elements, partial occupancies, and obviously invalid geometries.
3. Run a short test relaxation on representative structures.
4. Track energy, force, stress, volume, and symmetry changes.
5. Compare final structures against DFT relaxations for a chemically diverse subset.
6. Re-run suspicious relaxations with stricter settings or DFT.

Important diagnostics:

- Large volume collapse or expansion.
- Abrupt coordination changes.
- Unphysical short bonds after relaxation.
- Stress sign or unit inconsistencies.
- Relaxed structures that leave the intended phase, composition, or coordination environment.
- Systematic bias by chemistry or oxidation state.

For publication-quality workflows, learned-potential relaxations should be labeled as such and validated against the DFT setup used for downstream conclusions.

### Workflow 3: Custom Dataset Preparation

MatGL fine-tuning and training require stricter dataset hygiene than one-off inference.

Each record should preserve:

- Structure in a reproducible format.
- Composition, chemical system, formula, and source identifier.
- Energy target with units and reference convention.
- Forces with units and atom ordering.
- Stress tensor with units and sign convention.
- Property labels, task definitions, and uncertainty if available.
- DFT code, functional, pseudopotentials, cutoffs, k-point density, smearing, convergence thresholds, and relaxation state.
- Provenance fields linking back to database entries, workflow IDs, or calculation directories.

Before training:

- Deduplicate structures with a structure-matching method appropriate for the chemistry.
- Split by chemical system, structure family, time, or source when testing extrapolation.
- Keep polymorphs distinct when their target values differ.
- Remove failed or unconverged DFT calculations unless explicitly modeling failure states.
- Check that train, validation, and test sets do not share near-duplicate structures.

Random splits are often too optimistic for materials datasets because related compositions, prototypes, or relaxation trajectories can leak across splits.

### Workflow 4: Fine-Tuning and Transfer Learning

Use fine-tuning when a pretrained model is close to the target domain but systematically biased for a project-specific chemistry, functional, pressure range, defect class, or property definition.

Fine-tuning checklist:

- Start with a documented pretrained checkpoint.
- Freeze or unfreeze model components deliberately.
- Use a validation split that reflects the intended deployment domain.
- Tune learning rates conservatively; small datasets overfit quickly.
- Monitor energy, force, stress, and property losses separately.
- Compare against a non-fine-tuned baseline and a simple descriptor-based baseline.
- Retain the original test set if claiming transfer improvement.
- Save all training configuration, data split files, checkpoints, and environment metadata.

When fine-tuning on forces and stresses, confirm that every structure has consistent atom ordering, cell units, force units, and stress conventions. A single silent conversion error can dominate training behavior.

### Workflow 5: Validation

Validate MatGL workflows at the level of the scientific claim.

For property prediction:

- Report MAE/RMSE and residual distributions by chemistry, structure family, and target range.
- Include parity plots, calibration checks if uncertainty is available, and high-error outlier inspection.
- Compare against baseline descriptors, simple regressors, and other pretrained models where appropriate.

For learned potentials:

- Validate energies, forces, and stresses separately.
- Compare relaxed structures to DFT relaxations.
- Test equation-of-state curves, defect formation trends, surface energies, phonon stability, or MD stability when relevant.
- Use short NVE or thermostat-sensitivity checks before trusting MD trajectories.

For high-throughput screening:

- Recompute top candidates and uncertain regions with DFT.
- Track which predictions came from pretrained inference, fine-tuned models, and DFT.
- Avoid mixing corrected DFT energies and raw model predictions without explicit reference alignment.

## Common Failure Modes

| Failure mode | Symptom | Mechanical check |
| --- | --- | --- |
| Pretrained model outside domain | Confident-looking predictions for unsupported chemistry | Check element coverage, training data, and nearest known compositions |
| Unit mismatch | Energies, forces, or stresses off by large factors | Record and test all unit conversions on reference examples |
| Stress convention mismatch | Relaxation expands when it should compress, or vice versa | Confirm tensor order, sign convention, and pressure units |
| Broken periodic graph construction | Missing neighbors, isolated atoms, or unrealistic coordination | Inspect neighbor counts and periodic image offsets |
| Duplicate leakage | Excellent test metrics but poor external performance | Run structure matching and split by chemistry/prototype |
| Overfit fine-tuning | Validation improves briefly, then degrades or fails out-of-domain | Use small learning rates, early stopping, and external tests |
| Unsupported elements | Loader, graph builder, or embedding lookup fails | Pre-filter compositions and log excluded structures |
| Treating predictions as DFT-quality | Screening claims do not survive recomputation | Recompute representative and top-ranked candidates with DFT |
| Missing model metadata | Results cannot be reproduced later | Save model ID, checkpoint, package versions, and code commit |
| Wrong target interpretation | Formation energy, total energy, and property labels mixed | Audit target definitions before training or ranking |

## Best Practices

- Pin MatGL, pymatgen, DGL, PyTorch, and CUDA versions for production runs.
- Store train/validation/test split files as explicit artifacts.
- Use pymatgen or ASE readers consistently; do not mix silently converted structures without checking.
- Validate on chemically meaningful held-out sets, not only random splits.
- Log excluded structures and unsupported elements.
- Keep DFT reference settings attached to every training and validation record.
- Use scientific visualization to inspect residuals, outliers, relaxation paths, and chemical coverage.
- Use high-throughput DFT to confirm top candidates, boundary cases, and model disagreement regions.
- Treat pretrained uncertainty as unavailable unless the model and workflow explicitly provide calibrated uncertainty.

## Integration with Other Skills

- **[crystal-property-gnn](../crystal-property-gnn/SKILL.md):** Crystal graph construction, periodic materials targets, and dataset splitting.
- **[molecular-property-gnn](../molecular-property-gnn/SKILL.md):** Related graph-learning concepts for molecular property prediction.
- **[universal-potentials](../../ml-interatomic-potentials/universal-potentials/SKILL.md):** Deployment and validation expectations for pretrained interatomic potentials.
- **[fine-tuning-mlp](../../ml-interatomic-potentials/fine-tuning-mlp/SKILL.md):** Model adaptation strategies for domain-specific datasets.
- **[mlp-validation](../../ml-interatomic-potentials/mlp-validation/SKILL.md):** Energy, force, stress, relaxation, and MD validation protocols.
- **[mace-training](../../ml-interatomic-potentials/mace-training/SKILL.md):** Comparison point for equivariant learned potentials and dataset preparation.
- **[active-learning-mlp](../../ml-interatomic-potentials/active-learning-mlp/SKILL.md):** Selecting DFT queries to improve model coverage.
- **[materials-featurization](../../scientific-data/materials-featurization/SKILL.md):** Descriptor baselines and feature-audit practices.
- **[pymatgen-analysis](../../scientific-data/pymatgen-analysis/SKILL.md):** Structure handling, transformations, parsing, and structure matching.
- **[materials-databases-access](../../scientific-data/materials-databases-access/SKILL.md):** Database-derived structures, labels, and provenance.
- **[high-throughput-dft](../../electronic-structure/high-throughput-dft/SKILL.md):** DFT recomputation for screening validation.
- **[dft-convergence](../../electronic-structure/dft-convergence/SKILL.md):** Reference-quality DFT settings for validation datasets.
- **[scientific-visualization](../../scientific-data/scientific-visualization/SKILL.md):** Parity plots, residuals, coverage maps, and relaxation diagnostics.
- **[reproducibility-workflow](../../scientific-data/reproducibility-workflow/SKILL.md):** Environment pinning, provenance tracking, and artifact management.

## Key References

1. Chen, C. and Ong, S. P. "A universal graph deep learning interatomic potential for the periodic table." *Nature Computational Science* 2, 718-728 (2022).
2. Chen, C. et al. "Graph Networks as a Universal Machine Learning Framework for Molecules and Crystals." *Chemistry of Materials* 31, 3564-3572 (2019).
3. Deng, B. et al. "CHGNet as a pretrained universal neural network potential for charge-informed atomistic modelling." *Nature Machine Intelligence* 5, 1031-1041 (2023).
4. Xie, T. and Grossman, J. C. "Crystal Graph Convolutional Neural Networks for an Accurate and Interpretable Prediction of Material Properties." *Physical Review Letters* 120, 145301 (2018).
5. Dunn, A. et al. "Benchmarking materials property prediction methods: the Matbench test set and Automatminer reference algorithm." *npj Computational Materials* 6, 138 (2020).
