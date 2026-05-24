---
name: calphad-workflow
domain: thermodynamics-calphad
maturity: draft
tools:
  - pycalphad
  - Thermo-Calc
  - ESPEI
  - pandas
  - numpy
  - matplotlib
tested_versions:
  pycalphad: "0.10.5"
  ESPEI: "0.8.7"
  pandas: "2.2.0"
  numpy: "1.26.4"
  matplotlib: "3.8.0"
requires_gpu: false
estimated_runtime: minutes-to-days
---

# CALPHAD Workflow

## Description

This skill covers CALPHAD workflows for computational thermodynamics and materials design: thermodynamic database selection, equilibrium and metastable calculations, binary and ternary phase diagrams, Scheil solidification, chemical potentials, activities, driving forces, parameter assessment, uncertainty, and reproducible coupling to DFT, experiments, materials databases, precipitation, solidification, diffusion, and phase-field models. Invoke this skill when an agent needs phase stability or thermodynamic inputs from assessed Gibbs-energy models rather than from a single DFT calculation or empirical rule.

## Domain Context

CALPHAD, short for CALculation of PHAse Diagrams, represents the Gibbs energy of each phase as a function of temperature, pressure, composition, and internal sublattice degrees of freedom. Equilibria are obtained by minimizing the total Gibbs energy subject to mass balance and phase constraints. A CALPHAD database is not just a table of phase boundaries; it is a set of model functions, parameters, reference states, phase descriptions, and assessed experimental or first-principles data.

The validity of a CALPHAD result is therefore the validity of the database assessment. Results are strongest inside the assessed composition, temperature, pressure, and phase-space domain. Extrapolated ternaries, quaternaries, high-temperature liquids, metastable phases, magnetic transitions, order-disorder models, and non-stoichiometric compounds can be useful but must be treated as database-dependent predictions. [EXPERT REVIEW NEEDED]

CALPHAD complements DFT rather than replacing it. DFT gives 0 K or finite-temperature free-energy contributions for specific configurations; CALPHAD integrates experimental phase equilibria, calorimetry, activities, DFT formation energies, magnetic terms, solution models, and compound-energy formalism into thermodynamic descriptions usable across composition and temperature. Combining them requires consistent reference states, phases, and provenance.

Thermodynamic outputs from CALPHAD often feed other modeling workflows: precipitation driving forces, local equilibrium in diffusion simulations, liquidus/solidus surfaces for casting, phase fractions for alloy design, chemical potentials for defect calculations, and free-energy densities for phase-field simulations. Those downstream workflows inherit every database assumption.

## When to Use This Skill

- Use when calculating binary, ternary, or multicomponent phase equilibria from an assessed thermodynamic database.
- Use when generating isothermal sections, isopleths, liquidus/solidus projections, phase-fraction maps, or metastable diagrams.
- Use when extracting chemical potentials, activities, driving forces, or Gibbs energies for downstream precipitation, diffusion, solidification, or phase-field models.
- Use when comparing DFT formation energies, experimental phase equilibria, and database predictions.
- Use when selecting or auditing thermodynamic databases for a target alloy or materials system.
- Use when assessing uncertainty or sensitivity in database parameters, phase descriptions, or extrapolated higher-order systems.

## When Not to Use This Skill

- **When no assessed database covers the system.** A calculation from an unrelated database can look precise while being physically meaningless.
- **When the phase of interest is missing from the database.** Suppressing or omitting a real phase changes equilibria, driving forces, and predicted stability.
- **When atomic-scale local structure is the target.** CALPHAD gives thermodynamic state functions and phase amounts, not atomistic mechanisms or interface structures.
- **When the property depends on kinetics alone.** Equilibrium phase fractions do not predict nucleation barriers, diffusion paths, microstructure length scales, or transformation rates without additional models.
- **When database mixing is required but reference states are incompatible.** Parameters from different databases cannot usually be combined mechanically.
- **When using extrapolated high-order systems for final claims without validation.** Ternary and quaternary predictions must be checked against experiments, DFT, or higher-quality assessments. [EXPERT REVIEW NEEDED]

## Core Concepts

- **Gibbs energy model:** Mathematical expression for a phase Gibbs energy as a function of temperature, composition, and internal degrees of freedom.
- **Phase description:** Database definition of a phase, including constituents, sublattices, model type, parameters, and validity range.
- **Sublattice model:** Representation of ordered phases or solution phases with multiple crystallographic or chemical sublattices, e.g. `(Al,Ni):(Va)` or more complex site models.
- **Solution model:** Excess Gibbs-energy description for non-ideal mixing, often using Redlich-Kister or related polynomial forms.
- **Compound-energy formalism (CEF):** Framework for phases with multiple sublattices and endmembers whose site fractions are optimized during equilibrium.
- **Endmember:** A limiting configuration of a sublattice model with a defined Gibbs energy contribution.
- **Reference state:** The standard state used for each element and phase. Wrong reference states shift formation energies, chemical potentials, and driving forces.
- **Equilibrium calculation:** Gibbs-energy minimization over allowed phases subject to mass balance and state variables.
- **Metastable equilibrium:** Equilibrium calculated with selected stable phases suppressed or with constrained phase sets to inspect driving forces or transformation paths.
- **Activity and chemical potential:** Thermodynamic quantities describing effective composition and partial Gibbs energy, used for diffusion, defect equilibria, and reactions.
- **Scheil solidification:** Non-equilibrium solidification approximation with complete liquid mixing and no solid diffusion. Useful for casting trends but not a universal microsegregation model.
- **Parameter assessment:** Optimization of model parameters against experimental, first-principles, or database-derived data with uncertainty and model-form assumptions.

## Key Workflows

### Workflow 1: Select and audit a thermodynamic database

Database choice is the most important scientific decision in a CALPHAD workflow. Record it as input data, not as a software detail.

Checklist:

- Confirm the database covers every element, phase, and temperature/composition range needed.
- Read database documentation for assessed binaries, ternaries, extrapolation policy, magnetic models, and order-disorder descriptions.
- Check whether the target system is assessed directly or only extrapolated.
- Identify missing phases, metastable phases, liquids, gas phases, or compound descriptions.
- Record database name, vendor/source, version, release date, license, checksum if available, and local path.
- Compare at least one known experimental or literature phase diagram before production calculations.

Example provenance record:

```yaml
calphad_database:
  file: TCNI12.TDB
  source: Thermo-Calc nickel alloy database
  version: "12"
  checksum_sha256: "recorded-from-local-file"
  system: Ni-Al-Cr
  assessed_domain: "verify in database documentation"
  phases_enabled:
    - LIQUID
    - FCC_A1
    - BCC_A2
    - B2_BCC
    - GAMMA_PRIME
  notes:
    - "Do not combine with parameters from another database."
    - "[EXPERT REVIEW NEEDED] Confirm target alloy range is assessed."
```

### Workflow 2: Compute an equilibrium phase diagram with pycalphad

Use pycalphad for scriptable, reproducible equilibrium calculations when the database is available in a compatible TDB format.

```python
from pathlib import Path

import matplotlib.pyplot as plt
from pycalphad import Database, equilibrium, variables as v
from pycalphad.plot.eqplot import eqplot

db_path = Path("databases/Ni-Al-Cr_validated.tdb")
dbf = Database(str(db_path))

components = ["NI", "AL", "CR", "VA"]
phases = ["LIQUID", "FCC_A1", "BCC_A2", "B2_BCC"]

conditions = {
    v.T: (800, 1800, 10),
    v.P: 101325,
    v.X("AL"): (0.0, 0.30, 0.005),
    v.X("CR"): 0.10,
}

eq = equilibrium(dbf, components, phases, conditions)

fig = plt.figure(figsize=(7, 5))
ax = fig.add_subplot()
eqplot(eq, ax=ax)
ax.set_title("Ni-Al-Cr isopleth at X(Cr)=0.10")
fig.tight_layout()
fig.savefig("ni_al_cr_isopleth.png", dpi=300)
```

Before reporting:

- Verify phases are intentionally included or suppressed.
- Confirm compositions are mole fractions unless otherwise documented.
- Check that the database uses the same element symbols and vacancy convention expected by the code.
- Repeat key calculations with finer grids or point equilibria near invariant reactions.
- Save the exact phase list and conditions used to produce the figure.

### Workflow 3: Calculate metastable equilibria, driving forces, and chemical potentials

Metastable calculations are useful for nucleation, precipitation, defect, and transformation analyses. They are also easy to misuse because suppressing a stable phase changes the reference equilibrium.

Research workflow:

1. Compute the full equilibrium with all relevant stable phases enabled.
2. Identify the phase or phases whose metastable driving force is needed.
3. Recompute with selected phases suppressed or with only selected phases enabled.
4. Extract chemical potentials, activities, phase fractions, or driving forces.
5. Document the suppressed phase set and why the metastable state is physically meaningful.

Typical outputs:

- Chemical potentials for elements or species.
- Activities relative to database reference states.
- Gibbs energy difference between parent and product phases.
- Precipitation driving force as a function of temperature and composition.
- Phase amount sensitivity to a suppressed phase.

Do not compare driving forces from two databases unless reference states and phase descriptions are compatible.

### Workflow 4: Scheil solidification and liquidus/solidus analysis

Scheil calculations approximate non-equilibrium solidification with complete liquid mixing and no diffusion in the solid. They are useful for alloy screening, segregation trends, liquidus/solidus estimates, and identifying possible terminal phases.

Use Scheil results as process-sensitive indicators, not final microstructure predictions:

- Back diffusion, dendrite arm spacing, convection, nucleation undercooling, and phase coarsening are not included in the basic Scheil approximation.
- The result depends strongly on liquid and solid phase descriptions.
- Small terminal liquid fractions can produce noisy or unrealistic low-temperature phases.
- Experimental solidification paths may require kinetic or process models beyond CALPHAD.

For publication or design decisions, compare Scheil predictions against equilibrium solidification, experimental thermal analysis, or process simulation when available. [EXPERT REVIEW NEEDED]

### Workflow 5: Couple CALPHAD with DFT, experiments, and databases

CALPHAD assessment and validation often combine:

- Experimental phase boundaries, invariant reactions, calorimetry, activities, and heat capacities.
- DFT formation energies for stable and metastable compounds.
- DFT phonons or thermodynamic integration for finite-temperature free-energy contributions.
- Materials database structures and energies after compatibility checks.
- Bayesian or Gaussian-process surrogates for parameter sensitivity and uncertainty analysis.

DFT coupling checklist:

- Match reference states between DFT formation energies and CALPHAD endmembers.
- Use consistent magnetic, ordering, and structural states.
- Include uncertainty or quality flags for DFT convergence and finite-temperature corrections.
- Avoid overfitting sparse DFT data outside experimental constraints.
- Keep raw data, processed data, weights, parameter files, and scripts versioned.

### Workflow 6: Validate parameter assessments and database predictions

Validation should be external to the data used for assessment whenever possible.

Validation targets:

- Binary and ternary phase boundaries.
- Invariant reaction temperatures and compositions.
- Liquidus/solidus curves.
- Activities and chemical potentials.
- Enthalpies of mixing and formation.
- Phase fractions from heat treatment experiments.
- DFT formation energies of compounds not used in fitting.

Uncertainty practices:

- Report which parameters were varied and which were fixed.
- Use parameter covariance, Bayesian posterior samples, bootstrap fits, or sensitivity scans where available.
- Propagate parameter samples to phase boundaries, phase fractions, and driving forces.
- Identify regions where phase equilibria change discontinuously across plausible parameter sets.

Parameter uncertainty is often strongly model-form dependent. A narrow parameter posterior does not prove the phase model is correct. [EXPERT REVIEW NEEDED]

## Common Failure Modes

| Failure | Symptom | Likely Cause | Diagnostic | Fix |
|---------|---------|--------------|------------|-----|
| Database outside assessed range | Smooth but wrong phase diagram | Extrapolated composition or temperature domain | Check database documentation and assessed systems | Restrict domain or validate against data |
| Incompatible databases mixed | Discontinuous or nonsensical equilibria | Different reference states and phase models | Compare element references and phase definitions | Use one database or reassess parameters consistently |
| Missing metastable phase | Driving forces look too large or phase path is wrong | Phase absent or suppressed unintentionally | List enabled and disabled phases | Add/enable phase or document suppression |
| Extrapolated ternary trusted | Ternary predictions fail experiments | Binary-only or sparse ternary assessment | Check assessed ternaries and residuals | Treat as hypothesis and validate |
| Wrong reference state | Chemical potentials or formation energies shifted | Element or phase reference mismatch | Reproduce pure-element and binary limits | Convert references or rederive inputs |
| Missing magnetic model | Transition temperatures and phase boundaries biased | Magnetic contribution absent or invalid | Inspect database model terms | Use database with magnetic assessment |
| Order-disorder mishandled | Ordered phase fraction missing or duplicated | Wrong phase selection or sublattice interpretation | Compare ordered/disordered phase descriptions | Use documented phase set and constraints |
| Poor grid resolution | Missed invariant reactions or narrow phase fields | Coarse temperature/composition grid | Refine around boundaries and run point equilibria | Use adaptive refinement or targeted points |
| No provenance | Phase diagram cannot be reproduced | Missing database version, phase list, or conditions | Audit figure metadata | Store database/version/conditions with outputs |
| Activities misreported | Diffusion or defect coupling gives wrong trends | Wrong standard state or component convention | Compare limiting behavior and documentation | State reference state and units explicitly |

## Best Practices

- Treat the thermodynamic database as primary input data and version it like code.
- Never publish a phase diagram without database name, version, phase list, components, conditions, and software version.
- Validate the database on a simpler subsystem before using high-order extrapolations.
- Save both the plotting script and the raw equilibrium result object or table.
- Keep metastable calculations separate from full-equilibrium calculations in filenames and figures.
- Use point equilibria to verify important features from gridded diagrams.
- Record suppressed phases, fixed activities, reference states, and composition basis.
- Cross-check thermodynamic predictions with DFT or experiments for new compositions.
- Propagate parameter uncertainty when phase-boundary location or driving force magnitude controls a conclusion.
- Mention useful future couplings, such as phase-field or precipitation modeling, in prose until the corresponding repository skills exist.

## Integration with Other Skills

- **[materials-databases-access](../../scientific-data/materials-databases-access/SKILL.md):** Supplies database-derived structures, energies, provenance, and compatibility checks before thermodynamic comparison or assessment.
- **[pymatgen-analysis](../../scientific-data/pymatgen-analysis/SKILL.md):** Supports composition handling, phase diagram concepts, structure parsing, and formation-energy analysis around CALPHAD workflows.
- **[materials-featurization](../../scientific-data/materials-featurization/SKILL.md):** Provides descriptors for thermodynamic surrogate models and composition-space analysis.
- **[high-throughput-dft](../../electronic-structure/high-throughput-dft/SKILL.md):** Generates DFT data that can validate or inform CALPHAD assessments.
- **[dft-convergence](../../electronic-structure/dft-convergence/SKILL.md):** Ensures DFT formation energies and finite-temperature inputs are converged enough for thermodynamic use.
- **[scientific-visualization](../../scientific-data/scientific-visualization/SKILL.md):** Produces phase diagrams, uncertainty bands, residual plots, and publication-ready thermodynamic figures.
- **[reproducibility-workflow](../../scientific-data/reproducibility-workflow/SKILL.md):** Tracks database versions, scripts, results, checksums, and figure provenance.
- **[gaussian-process-materials](../../surrogate-active-learning/gaussian-process-materials/SKILL.md):** Models thermodynamic residuals, parameter sensitivities, or surrogate response surfaces for composition screening.
- **[bayesian-optimization](../../surrogate-active-learning/bayesian-optimization/SKILL.md):** Guides alloy composition search using CALPHAD outputs and uncertainty-aware acquisition.
- **[surrogate-validation](../../surrogate-active-learning/surrogate-validation/SKILL.md):** Validates surrogate models trained on CALPHAD-derived phase stability or driving-force targets.
- **[small-data-ml](../../scientific-ml/small-data-ml/SKILL.md):** Provides validation strategies for sparse experimental and thermodynamic datasets.

## Key References

1. Saunders, N. and Miodownik, A. P. (1998). *CALPHAD: Calculation of Phase Diagrams: A Comprehensive Guide*. Pergamon.
2. Lukas, H. L., Fries, S. G., and Sundman, B. (2007). *Computational Thermodynamics: The Calphad Method*. Cambridge University Press.
3. Andersson, J. O. et al. (2002). *Thermo-Calc and DICTRA, computational tools for materials science*. Calphad. https://doi.org/10.1016/S0364-5916(02)00037-8
4. Otis, R. and Liu, Z.-K. (2017). *pycalphad: CALPHAD-based computational thermodynamics in Python*. Journal of Open Research Software. https://doi.org/10.5334/jors.140
5. Bocklund, B. et al. (2019). *ESPEI for efficient thermodynamic database development, modification, and uncertainty quantification: application to Cu-Mg*. MRS Communications. https://doi.org/10.1557/mrc.2019.59
