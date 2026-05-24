---
name: corrosion-electrochemistry
domain: electrochemistry-corrosion
maturity: draft
tools:
  - pycalphad
  - FEniCSx
  - MOOSE
  - pymatgen
  - ASE
  - pandas
  - numpy
  - matplotlib
tested_versions:
  pycalphad: "0.10.5"
  FEniCSx: "0.7.3"
  MOOSE: "2024.11.11"
  pymatgen: "2024.11.13"
  ASE: "3.23.0"
  pandas: "2.2.0"
  numpy: "1.26.4"
requires_gpu: false
estimated_runtime: hours-to-weeks
---

# Corrosion Electrochemistry

## Description

This skill covers corrosion and electrochemistry workflows for materials degradation and surface reactions: electrode potentials, reference electrodes, polarization curves, impedance, passivation, pitting, galvanic coupling, Pourbaix diagrams, transport modeling, surface chemistry, parameter calibration, uncertainty, and validation against experiments, thermodynamics, DFT, and continuum models. Invoke this skill when an agent needs to reason about electrochemical stability, corrosion kinetics, solution chemistry, surface reactions, or degradation risk in materials environments.

## Domain Context

Corrosion is an electrochemical degradation process controlled by thermodynamics, kinetics, mass transport, surface chemistry, microstructure, and environment. A metal or semiconductor surface may be thermodynamically unstable but kinetically protected by a passive film, or thermodynamically passive but vulnerable to pitting, crevice chemistry, galvanic coupling, stress corrosion cracking, or local breakdown. No single diagram or polarization curve captures the whole risk.

Electrode potentials are defined relative to reference electrodes. Converting between SHE, RHE, Ag/AgCl, SCE, Hg/HgO, Li/Li+, and other references requires temperature, electrolyte, activity, and reference-electrode details. Wrong conversions can shift conclusions by hundreds of millivolts. pH, chloride activity, dissolved oxygen, temperature, flow, alloy composition, surface preparation, oxide history, and stress state can all control the measured response. [EXPERT REVIEW NEEDED]

Thermodynamic tools such as Pourbaix diagrams indicate equilibrium stability as a function of potential and pH under specified activities. They do not prove corrosion resistance, passivation quality, pit nucleation probability, or current density. Kinetic workflows use polarization curves, Tafel analysis, Butler-Volmer models, exchange current densities, impedance spectra, and transport models to estimate rates and mechanisms. The thermodynamic and kinetic pictures must be interpreted together.

Modeling corrosion often couples atomic-scale surface chemistry, CALPHAD or aqueous thermodynamics, continuum transport, FEM multiphysics, phase-field damage or pitting models, and experimental calibration. Outputs should be tied to the environment being modeled: electrolyte composition, pH, temperature, gas saturation, flow, applied potential, reference electrode, exposure time, and surface condition.

## When to Use This Skill

- Use when interpreting corrosion experiments, polarization curves, open-circuit potentials, Tafel slopes, impedance spectra, or passivation behavior.
- Use when building thermodynamic or kinetic models of dissolution, oxide formation, adsorption, pitting, crevice corrosion, galvanic corrosion, or stress corrosion cracking.
- Use when coupling DFT surface energetics, CALPHAD thermodynamics, solution chemistry, and continuum transport to corrosion risk.
- Use when calibrating exchange current densities, transfer coefficients, diffusion coefficients, passivation parameters, or surface reaction rates.
- Use when comparing candidate alloys, coatings, catalysts, or surface treatments under a specified electrochemical environment.
- Use when connecting degradation risk to sustainability, lifetime, materials selection, or accelerated testing.

## When Not to Use This Skill

- **When the electrolyte and reference electrode are unspecified.** Potential, pH, and chemistry define the problem.
- **When Pourbaix diagrams are used as rate predictions.** They show equilibrium stability, not corrosion current, pit probability, or film protectiveness.
- **When surface preparation differs across experiments.** Polishing, oxide history, heat treatment, roughness, and contamination can dominate results.
- **When transport boundary layers are unresolved.** Mass-transport-limited corrosion or electrochemistry cannot be interpreted from kinetics alone.
- **When Tafel fits are applied outside a valid potential window.** Mixed control, passivation, ohmic drop, or transport limitation can invalidate slopes.
- **When stress, microstructure, or local chemistry controls degradation but is omitted.** Stress corrosion cracking, pitting, and crevice corrosion are local multiphysics phenomena.

## Core Concepts

- **Electrode potential:** Electrical potential of an electrode relative to a reference electrode. Always state the reference scale and conversion.
- **Reference electrode:** Experimental reference such as SHE, RHE, Ag/AgCl, SCE, Hg/HgO, or Li/Li+. Conversions are electrolyte- and temperature-dependent.
- **Overpotential:** Difference between applied potential and equilibrium potential for a reaction.
- **Nernst equation:** Relates equilibrium potential to activities, temperature, and reaction stoichiometry.
- **Butler-Volmer kinetics:** Relationship between current density and overpotential for charge-transfer reactions.
- **Exchange current density:** Kinetic parameter measuring the rate of anodic and cathodic reactions at equilibrium.
- **Tafel analysis:** Linearized high-overpotential regime used to estimate kinetic slopes and corrosion current under strict assumptions.
- **Open-circuit potential (OCP):** Mixed potential at zero net external current, often time-dependent during film growth or surface evolution.
- **Polarization curve:** Current-potential response that can reveal active dissolution, passivation, transpassivity, pitting, and transport limitation.
- **Electrochemical impedance spectroscopy (EIS):** Frequency-dependent response used to probe charge transfer, films, transport, and capacitance with model-dependent interpretation.
- **Passivation:** Formation of a protective film that suppresses dissolution. Film stability, defect chemistry, and local breakdown are environment-dependent.
- **Pourbaix diagram:** Potential-pH equilibrium diagram for aqueous species, solids, and ions at specified activities.
- **Mass transport:** Diffusion, migration, and convection of ions and reactants in electrolyte or porous media.

## Key Workflows

### Workflow 1: Define the electrochemical environment and potential scale

Every corrosion workflow should begin by defining the environment precisely.

Environment checklist:

- Alloy, coating, surface orientation, microstructure, heat treatment, and surface preparation.
- Electrolyte composition, pH, chloride concentration, buffer, dissolved oxygen, additives, and impurities.
- Temperature, pressure, flow, gas atmosphere, and exposure time.
- Reference electrode and conversion to SHE or RHE if needed.
- Applied potential, scan rate, hold time, or current control.
- Geometry, exposed area, crevice configuration, and counter electrode.
- Replicates, stabilization criteria, and surface characterization before and after testing.

Example metadata record:

```yaml
experiment:
  material: "316L stainless steel"
  surface_preparation: "1200 grit polish, ethanol rinse, air exposure 24 h"
  electrolyte: "0.6 M NaCl, naturally aerated"
  pH: 7.0
  temperature_C: 25
  reference_electrode: "Ag/AgCl saturated KCl"
  reported_potential_scale: "V vs Ag/AgCl"
  conversion_to_SHE_V: "[EXPERT REVIEW NEEDED] verify for electrolyte and temperature"
  exposed_area_cm2: 1.0
  scan_rate_mV_s: 0.167
```

### Workflow 2: Interpret polarization curves and Tafel regions

Polarization curves must be interpreted with mechanism and experimental context, not just fitted automatically.

Analysis steps:

1. Inspect OCP stabilization before scan.
2. Correct or estimate uncompensated resistance where needed.
3. Identify active, passive, transpassive, pitting, and transport-limited regions.
4. Fit Tafel slopes only in potential windows that are kinetically controlled.
5. Report reference electrode, scan rate, area normalization, and electrolyte.
6. Compare repeated scans and post-test surfaces.

Python sketch for transparent fitting:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("polarization_curve.csv")
E = df["potential_V_vs_SHE"].to_numpy()
j = np.abs(df["current_density_A_cm2"].to_numpy())

window = (E > -0.25) & (E < -0.12) & (j > 0)
slope, intercept = np.polyfit(E[window], np.log10(j[window]), 1)

fig, ax = plt.subplots()
ax.semilogy(E, j, label="measured")
ax.semilogy(E[window], 10 ** (slope * E[window] + intercept), label="fit window")
ax.set_xlabel("Potential (V vs SHE)")
ax.set_ylabel("|j| (A/cm2)")
ax.legend()
fig.savefig("tafel_fit_with_window.png", dpi=300)
```

This fit is only meaningful if the selected region is under activation control and not dominated by passivation, transport limitation, ohmic drop, or mixed reactions.

### Workflow 3: Build Pourbaix and thermodynamic stability workflows

Pourbaix diagrams are useful for hypothesis generation and thermodynamic context.

Workflow:

- Define elements, aqueous species, solid phases, activities, temperature, and pressure.
- State whether species are from a database, literature, CALPHAD-like assessment, DFT, or manual thermodynamic data.
- Convert all energies and standard states consistently.
- Generate potential-pH regions and mark experimental conditions.
- Treat missing species, complexation, and activity effects as uncertainty.
- Compare predicted stable phases with observed corrosion products and surface analysis.

Do not use a Pourbaix diagram outside its assumptions. Chloride complexation, carbonate, borate, ammonia, high ionic strength, non-equilibrium films, metastable pits, or concentrated electrolytes can invalidate simple diagrams. [EXPERT REVIEW NEEDED]

### Workflow 4: Couple surface chemistry, DFT, and corrosion kinetics

DFT can inform corrosion models through surface energies, adsorption energies, oxide formation energies, vacancy formation, dissolution steps, and reaction barriers. These quantities are sensitive to surface orientation, coverage, solvation, potential, pH, and functional choice.

Useful DFT-derived inputs:

- Adsorption energies for O, OH, H, Cl, water, or inhibitor species.
- Surface Pourbaix or ab initio thermodynamics trends.
- Defect formation energies in passive films.
- Reaction barriers for elementary surface steps.
- Work functions and potential alignment estimates.
- Relative stability of oxide terminations or adsorbate coverages.

DFT values should be validated or bounded with experiment when used in rate models. Constant-potential methods, solvation, electric fields, and explicit electrolyte can matter for electrochemical surfaces. [EXPERT REVIEW NEEDED]

### Workflow 5: Model continuum transport and multiphysics corrosion

Transport and local chemistry often control corrosion rate and localization.

Model choices:

- Diffusion-only models for dilute species under weak electric fields.
- Migration-diffusion models for charged species under electric potential gradients.
- Poisson-Nernst-Planck models when charge separation matters.
- Electroneutral Nernst-Planck models for many concentrated electrolyte corrosion problems.
- Convection-diffusion when flow or rotating disk experiments matter.
- Coupled mechanics or phase-field models for cracking, pitting, or moving boundaries.

Continuum checklist:

- Define species, charges, reactions, electroneutrality or Poisson equation, and boundary fluxes.
- Calibrate diffusion coefficients, reaction rates, and film transport parameters.
- Resolve boundary layers, crevice gaps, pit geometry, and moving interfaces.
- Validate against concentration profiles, limiting currents, pit growth rates, or impedance features.
- Run mesh and timestep convergence for local current density and species gradients.

### Workflow 6: Interpret EIS with model discipline

EIS can separate time scales but is highly model-dependent.

Good practice:

- Verify linearity, stability, and causality conditions.
- Inspect Bode and Nyquist plots before fitting.
- Use equivalent circuits only when elements map to plausible physics.
- Compare multiple candidate models and report parameter confidence.
- Validate fitted capacitance, film resistance, charge-transfer resistance, and diffusion terms against independent measurements.
- Avoid overfitting with many circuit elements.

Distributed elements, porous films, rough surfaces, and nonstationary corrosion can make simple equivalent circuits misleading. [EXPERT REVIEW NEEDED]

### Workflow 7: Validate and quantify uncertainty

Validation should combine electrochemical, chemical, and surface evidence.

Validation targets:

- Repeatability of OCP, polarization, and impedance measurements.
- Surface characterization: XPS, Raman, SEM/EDS, AFM, TEM, XRD, or profilometry.
- Solution chemistry: ICP-OES/MS dissolution, pH changes, chloride depletion, dissolved oxygen.
- Corrosion rate comparison: mass loss, current integration, pit depth, or profilometry.
- Thermodynamic consistency with Pourbaix or CALPHAD predictions.
- Transport model validation against limiting currents or concentration profiles.

Uncertainty sources:

- Reference electrode conversion.
- Surface area and roughness.
- Solution composition and pH drift.
- Temperature and oxygen content.
- Replicate variability.
- Parameter covariance in fitted kinetic models.
- Missing species or phases in thermodynamic models.

## Common Failure Modes

| Failure | Symptom | Likely Cause | Diagnostic | Fix |
|---------|---------|--------------|------------|-----|
| Thermodynamics confused with kinetics | Pourbaix predicts stability but corrosion occurs | Passive film or local breakdown not represented | Compare current density and surface products | Add kinetic/passivation evidence |
| Wrong reference conversion | Potential windows shifted | Incorrect reference electrode offset | Recompute versus SHE/RHE with conditions | Record reference and conversion explicitly |
| Missing solution chemistry | Model misses pitting or precipitation | Chloride, oxygen, buffer, complexation, or pH omitted | Species and activity audit | Include relevant aqueous chemistry |
| pH/chloride/temperature ignored | Trends fail across environments | Environment-sensitive kinetics | Sensitivity experiments or model sweep | Report environment-specific validity |
| Tafel overfitting | Corrosion current changes with fit window | Mixed control, passivation, or ohmic drop | Fit-window sensitivity | Use mechanistic window or avoid Tafel claim |
| Pourbaix outside assumptions | Stability map contradicts experiment | Activity, complexation, or metastability ignored | Check species and activities | Use environment-specific thermodynamics |
| Under-resolved boundary layer | Limiting current or pit growth wrong | Mesh too coarse near interface | Mesh/time convergence | Refine boundary layer or use analytical mass transfer |
| Inconsistent surface preparation | Replicates disagree strongly | Oxide, roughness, contamination, or stress differs | Surface characterization before/after | Standardize preparation and report it |
| EIS over-parameterization | Excellent fit with nonphysical parameters | Too many circuit elements | Confidence intervals and model comparison | Use simpler physics-based model |
| Galvanic coupling omitted | Local corrosion appears unexpectedly | Mixed materials or electrical contact | Couple potential/current balance | Include galvanic geometry and area ratio |

## Best Practices

- Always report potential scale, reference electrode, temperature, pH, electrolyte composition, and surface preparation.
- Separate thermodynamic stability, kinetic current density, transport limitation, and surface-film evidence.
- Use Pourbaix diagrams as context, not proof of corrosion resistance.
- Fit Tafel slopes only in justified kinetic regions and report fit windows.
- Validate transport models with mesh and timestep convergence.
- Use independent surface and solution characterization to interpret electrochemical data.
- Track all parameter sources, units, and uncertainty.
- Run replicate experiments or model scenarios before ranking materials.
- Couple corrosion performance to sustainability or lifetime claims only when the service environment is defined.
- Store raw curves, fitting scripts, metadata, and processed figures for reproducibility.

## Integration with Other Skills

- **[dft-surfaces](../../electronic-structure/dft-surfaces/SKILL.md):** Provides surface energies, adsorption models, work functions, and ab initio thermodynamics for corrosion surface chemistry.
- **[dft-defects](../../electronic-structure/dft-defects/SKILL.md):** Supports defect energetics in oxides, passive films, and corrosion products.
- **[calphad-workflow](../../thermodynamics-calphad/calphad-workflow/SKILL.md):** Supplies thermodynamic stability, phase equilibria, chemical potentials, and alloy phase context.
- **[fem-multiphysics](../../continuum-modeling/fem-multiphysics/SKILL.md):** Supports transport, electrochemical, mechanical, and coupled corrosion simulations.
- **[phase-field-workflow](../../phase-field-microstructure/phase-field-workflow/SKILL.md):** Connects to pitting, phase transformation, moving-boundary, and damage-evolution models.
- **[materials-lca](../../sustainability-lca/materials-lca/SKILL.md):** Connects corrosion resistance and lifetime to sustainability and materials selection.
- **[materials-databases-access](../../scientific-data/materials-databases-access/SKILL.md):** Supplies composition, structure, property, and provenance data for corrosion screening.
- **[materials-featurization](../../scientific-data/materials-featurization/SKILL.md):** Provides descriptors for corrosion surrogate models and alloy screening.
- **[scientific-visualization](../../scientific-data/scientific-visualization/SKILL.md):** Supports polarization curves, EIS plots, Pourbaix maps, transport fields, and uncertainty figures.
- **[reproducibility-workflow](../../scientific-data/reproducibility-workflow/SKILL.md):** Tracks experimental metadata, raw curves, fitting scripts, model parameters, and provenance.
- **[gaussian-process-materials](../../surrogate-active-learning/gaussian-process-materials/SKILL.md):** Builds uncertainty-aware corrosion-property surrogates from sparse data.
- **[bayesian-optimization](../../surrogate-active-learning/bayesian-optimization/SKILL.md):** Supports alloy, coating, or inhibitor optimization under corrosion constraints.
- **[surrogate-validation](../../surrogate-active-learning/surrogate-validation/SKILL.md):** Validates corrosion surrogates under extrapolation and environment shift.
- **[small-data-ml](../../scientific-ml/small-data-ml/SKILL.md):** Supports leakage-safe modeling with sparse corrosion datasets and replicate variability.

## Key References

1. Bard, A. J. and Faulkner, L. R. (2001). *Electrochemical Methods: Fundamentals and Applications*. Wiley.
2. Pourbaix, M. (1974). *Atlas of Electrochemical Equilibria in Aqueous Solutions*. NACE.
3. Jones, D. A. (1996). *Principles and Prevention of Corrosion*. Prentice Hall.
4. Macdonald, D. D. (1992). *The point defect model for the passive state*. Journal of the Electrochemical Society. https://doi.org/10.1149/1.2069096
5. Frankel, G. S. (1998). *Pitting corrosion of metals: a review of the critical factors*. Journal of the Electrochemical Society. https://doi.org/10.1149/1.1838615
