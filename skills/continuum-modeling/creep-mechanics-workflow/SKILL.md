---
name: creep-mechanics-workflow
domain: continuum-modeling
maturity: draft
tools:
  - Python
  - NumPy
  - SciPy
  - pandas
  - matplotlib
  - Abaqus
  - ANSYS Mechanical
  - CalculiX
  - MFront
tested_versions:
  Python: "unknown"
  NumPy: "unknown"
  SciPy: "unknown"
  pandas: "unknown"
  matplotlib: "unknown"
  Abaqus: "unknown"
  ANSYS Mechanical: "unknown"
  CalculiX: "unknown"
  MFront: "unknown"
requires_gpu: false
estimated_runtime: hours-to-months
---

# Creep Mechanics Workflow

## Description

Select, calibrate, implement, and validate creep deformation and rupture models for materials and components under sustained thermomechanical loading. The workflow separates rate prediction, transient deformation, stress redistribution, damage evolution, and rupture so that a secondary-creep fit is not mistaken for a full-life model.

This skill supports scientific and engineering analysis. It does not replace high-temperature design codes, qualified material databases, inspection evidence, or safety review.

## Domain Context

Creep is time-dependent inelastic deformation under sustained stress, commonly accelerated by temperature. Whether creep is engineering-relevant depends on material class, microstructure, stress, temperature, time, environment, and loading history; no universal temperature or homologous-temperature threshold defines its onset. [EXPERT REVIEW NEEDED]

A conventional uniaxial creep curve may contain:

- **Primary creep:** decreasing strain rate as hardening or internal resistance develops.
- **Secondary or minimum-rate region:** approximately steady or minimum strain rate over part of the test; a truly constant regime may be absent.
- **Tertiary creep:** accelerating strain from damage, necking, cavitation, microstructural evolution, or increasing true stress, ending in rupture.

These stages are curve descriptions rather than unique mechanisms. Diffusion-controlled creep, dislocation glide/climb, grain-boundary sliding, precipitate evolution, recovery, cavitation, oxidation, and phase changes can overlap. Mechanism maps guide hypothesis formation but do not replace alloy- and state-specific evidence.

Homologous temperature uses absolute temperature, normally `T/T_m`, but the relevant melting or transformation temperature can be ambiguous for multiphase alloys, ceramics, polymers, and evolving materials. [EXPERT REVIEW NEEDED]

Creep analysis has several distinct outputs:

- deformation or clearance at a service time;
- stress relaxation under constrained strain;
- redistributed multiaxial stress and strain fields;
- onset and evolution of creep damage;
- time to a specified strain, crack, loss of section, or rupture event;
- probability or lower-bound life under material and loading scatter.

## When to Use This Skill

- Fit steady or minimum creep rates across stress and temperature.
- Calibrate primary-secondary-tertiary or internal-variable constitutive models.
- Simulate creep deformation, relaxation, redistribution, and damage in finite elements.
- Assess rupture correlations and long-time extrapolation with uncertainty.
- Compare competing mechanism-based, empirical, and continuum-damage models.
- Evaluate multiaxial components, weld regions, anisotropy, gradients, or evolving material states.
- Design validation tests that separate interpolation from service-life extrapolation.

## When Not to Use This Skill

- Do not apply a steady-state power law as a full-life model when primary strain, tertiary damage, necking, or rupture matters.
- Do not transfer parameters between heat treatments, product forms, weld zones, irradiation states, environments, or alloy batches without evidence.
- Do not fit Arrhenius behavior across a phase or mechanism transition with one activation energy.
- Do not use uniaxial rupture correlations as a multiaxial damage law without validation.
- Do not extrapolate laboratory tests by orders of magnitude without model-form comparison and uncertainty bounds.
- Do not interpret a converged FE time integration as proof that the constitutive or rupture mechanism is valid.

## Core Concepts

### Stress and temperature dependence

A common steady-rate representation is

```text
epsilon_dot = A * (sigma / sigma_ref)^n * exp(-Q / (R * T))
```

where `T` is absolute temperature, `Q` and `R` use consistent energy-per-mole units, and `sigma_ref` makes the power-law argument dimensionless. The fitted `A`, 
`, and `Q` are effective parameters over the calibration domain. A change in slope with stress or temperature may indicate mechanism transition, threshold stress, microstructure evolution, or data limitations rather than one global exponent.

Diffusion-controlled models often predict lower stress exponents and explicit grain-size dependence, while dislocation-controlled power-law creep can show stronger stress dependence. Exact exponents and activation energies are not mechanism identifiers by themselves. [EXPERT REVIEW NEEDED]

### Transient formulations

Time-hardening models express creep rate as a function of stress, temperature, and elapsed time; strain-hardening models use accumulated creep strain as an internal history measure. Their predictions differ under variable load, unloading, and stress redistribution. Internal-variable models can represent hardening, recovery, back stress, damage, precipitate state, or other evolving resistance, but additional parameters increase identifiability demands.

Stress relaxation is a coupled boundary-value problem: under fixed total strain, elastic stress falls as creep strain accumulates. Substituting the initial stress into a constant-stress creep equation is generally inconsistent.

### Multiaxial creep and redistribution

Many isotropic implementations drive deviatoric creep with an equivalent stress and assume nearly incompressible creep flow. Hydrostatic stress can still strongly influence cavity growth, grain-boundary damage, and rupture. Choice of von Mises, maximum principal, Huddleston, stress triaxiality, or another rupture/damage measure is model- and material-dependent. [EXPERT REVIEW NEEDED]

Components redistribute load as high-stress regions creep. The life-controlling location can therefore move, and initial elastic peaks may relax while other regions accumulate strain or damage. Weldments and anisotropic products require region- or orientation-specific behavior where evidence supports it.

### Damage and rupture

Continuum damage models introduce internal variables that reduce resistance or accelerate strain rate. Damage is model-defined and usually not directly observable; calibrate it against multiple observables such as full creep curves, reduction of area, cavity density, stiffness, and rupture time. Local softening can cause mesh dependence unless the model or failure criterion is regularized or otherwise objective. [EXPERT REVIEW NEEDED]

A Larson-Miller-type parameter is commonly written

```text
P_LM = T * (C + log10(t_r))
```

but the logarithm base, time unit, fitted constant, temperature unit, stress relation, and calibration window must be stated. `C` is not a universal constant. [EXPERT REVIEW NEEDED]

A generalized Monkman-Grant relation connects minimum creep rate and rupture time,

```text
(epsilon_dot_min)^m * t_r = C_MG
```

as an empirical correlation. It does not represent tertiary damage or guarantee transfer across stress, temperature, microstructure, or ductility regimes. [EXPERT REVIEW NEEDED]

Creep-fatigue interaction requires cyclic plasticity, dwell-dependent creep/relaxation, oxidation or environment where relevant, and an interaction rule validated for the intended waveform. Summing independent fatigue and creep fractions is not a universal physical law. [EXPERT REVIEW NEEDED]

### Numerical integration and identifiability

Implicit FE integration generally solves local constitutive residuals with Newton iterations and a consistent or numerically suitable tangent. Large time increments can skip rapid primary response, tertiary acceleration, or load redistribution even when global equilibrium converges. Converge deformation, stress, damage, and predicted life with respect to both mesh and time increments.

Creep parameters are often strongly correlated. Sparse tests over a narrow stress-temperature window may identify a curve but not `A`, 
`, `Q`, hardening, and damage parameters separately. Use grouped validation, profile likelihood, bootstrap, Bayesian inference, or comparable uncertainty analysis before long-time extrapolation.
## Key Workflows

### Workflow 1: Fit steady or minimum creep parameters

Require specimen-level stress, absolute temperature, minimum or steady-rate definition, time unit, strain measure, material state, environment, and test provenance. Fit in log-rate space only after deciding whether the residual model is appropriate; log transformation changes the error assumptions.

```python
import numpy as np
from scipy.optimize import least_squares

R_GAS = 8.314462618  # J mol^-1 K^-1


def fit_norton_arrhenius(
    stress_pa, temperature_K, rate_per_s, sigma_ref_pa, initial, bounds
):
    """Fit log(A), n, and Q[J/mol] to positive finite creep-rate data."""
    stress = np.asarray(stress_pa, dtype=float)
    temp = np.asarray(temperature_K, dtype=float)
    rate = np.asarray(rate_per_s, dtype=float)
    if stress.shape != temp.shape or stress.shape != rate.shape or stress.ndim != 1:
        raise ValueError("Stress, temperature, and rate must be equal-length vectors")
    if len(rate) < 4 or not np.isfinite(np.column_stack([stress, temp, rate])).all():
        raise ValueError("Provide at least four finite observations")
    if (
        (stress <= 0).any()
        or (temp <= 0).any()
        or (rate <= 0).any()
        or not np.isfinite(sigma_ref_pa)
        or sigma_ref_pa <= 0
    ):
        raise ValueError("Stress, absolute temperature, rate, and reference stress must be positive")

    initial = np.asarray(initial, dtype=float)
    lower, upper = (np.asarray(bound, dtype=float) for bound in bounds)
    if initial.shape != (3,) or lower.shape != (3,) or upper.shape != (3,):
        raise ValueError("initial and both bounds must contain log(A), n, and Q")
    if not np.isfinite(np.concatenate([initial, lower, upper])).all():
        raise ValueError("Initial values and bounds must be finite")
    if not ((lower < initial).all() and (initial < upper).all()):
        raise ValueError("Initial values must lie strictly inside ordered bounds")

    def residual(theta):
        log_A, n, Q_J_per_mol = theta
        predicted_log_rate = (
            log_A + n * np.log(stress / sigma_ref_pa) - Q_J_per_mol / (R_GAS * temp)
        )
        return predicted_log_rate - np.log(rate)

    result = least_squares(residual, x0=initial, bounds=(lower, upper))
    if not result.success:
        raise RuntimeError(result.message)
    log_A, n, Q = result.x
    A = np.exp(log_A)
    if not np.isfinite([A, n, Q]).all() or A <= 0 or n <= 0 or Q <= 0:
        raise RuntimeError("Fit returned nonphysical or nonfinite parameters")
    return {
        "A_per_s": float(A),
        "n": float(n),
        "Q_J_per_mol": float(Q),
        "result": result,
    }
```
Set bounds from defensible material knowledge and the experimental domain; no universal bounds apply. [EXPERT REVIEW NEEDED] Inspect residuals against stress, temperature, time, batch, and specimen. Bootstrap by specimen or batch, not by individual rows from the same test. Enforce no mechanism interpretation from `n` or `Q` without independent evidence.

### Workflow 2: Calibrate temperature-dependent transient creep

1. Convert every temperature to kelvin and every time to one declared unit.
2. Align zero strain/time, machine compliance, thermal strain, and load-ramp treatment.
3. Fit full curves or carefully selected features across multiple stresses and temperatures.
4. Compare time-hardening, strain-hardening, theta-projection, internal-variable, or mechanistic candidates using held-out histories.
5. Test load changes and relaxation histories because constant-load curves may not identify path dependence.
6. Check whether fitted activation behavior changes across phase, precipitation, oxidation, or mechanism transitions.

Do not concatenate curves with different rupture definitions, extensometer conventions, or heat treatments merely to increase sample count.

### Workflow 3: Simulate creep deformation and stress relaxation in FEM

1. Establish the thermal and mechanical initial state, including residual stress and preload.
2. Verify the constitutive update with a single integration-point test under constant stress, fixed strain, load change, and temperature change.
3. Compare FE uniaxial output against the calibration curves using the same true/engineering stress and strain conventions.
4. Converge time increments during primary response, stress redistribution, and tertiary acceleration.
5. Converge the mesh on displacement, stress, accumulated creep strain, damage, and life-controlling location.
6. Check reactions, energy, local integration status, cutbacks, and state-variable bounds.
7. Validate component deformation or relaxation against independent measurements where available.

Use [FEM multiphysics](../fem-multiphysics/SKILL.md) for boundary conditions, solver tolerances, coupling, and mesh/time-step convergence.

### Workflow 4: Assess creep rupture or time to a limit

Define the event: specified creep strain, onset of tertiary acceleration, crack indication, loss of load, or complete rupture. Fit Larson-Miller, Monkman-Grant, continuum-damage, or another model only over a documented stress-temperature-time window. Retain runouts and interrupted tests with correct censoring. Compare predictions with direct long-duration tests withheld from calibration, and report both interpolation and extrapolation distances.

For component life, integrate the constitutive/damage history under redistributed stress rather than applying a uniaxial rupture time to the initial peak stress. Apply any design-code reduction factors and allowable definitions exactly as specified by the governing code.

### Workflow 5: Analyze multiaxial component creep

Partition base material, weld metal, heat-affected zones, coatings, and anisotropic regions when evidence supports different behavior. Simulate sustained loads and temperature gradients through the service sequence. Track equivalent and principal stresses, triaxiality, creep strain, damage, constraint, and location migration. Compare candidate multiaxial rupture measures against notched-bar, tube, cross-weld, or other relevant validation tests before relying on component predictions.

### Workflow 6: Quantify parameter uncertainty and extrapolation risk

Create grouped calibration and validation partitions by heat, specimen, stress level, and temperature. Sample parameter uncertainty and model-form alternatives, then propagate them through service histories. Plot prediction intervals against time and explicitly mark the convex hull of calibration conditions. Perform leave-one-stress and leave-one-temperature-out tests; a random row split is not a meaningful extrapolation test for correlated creep curves.

Use [surrogate validation](../../surrogate-active-learning/surrogate-validation/SKILL.md) for grouped splits and calibration diagnostics. Long-life tails can be dominated by model discrepancy rather than fitted-parameter variance; report both where possible. [EXPERT REVIEW NEEDED]

### Workflow 7: Compare competing creep models

Use a common data contract, objective function, weighting policy, censoring treatment, and validation set. Compare at minimum:

- minimum-rate prediction across stress and temperature;
- full strain-time curves and relaxation histories;
- rupture or limit-event time;
- parameter stability under data removal;
- extrapolation behavior and physical admissibility;
- FE robustness, time-step sensitivity, and computational cost.

Prefer the simplest model that reproduces the observables and paths needed for the decision. A lower calibration error does not justify an unidentifiable or physically implausible model.
## Common Failure Modes

| Failure | Symptom | Likely Cause | Diagnostic | Fix |
|---|---|---|---|---|
| Power law extrapolated across unsupported stresses | Large life/rate errors at stress extremes | Mechanism or threshold behavior changed | Plot local slopes and leave-one-stress-out predictions | Restrict domain or use a supported transition model |
| Inappropriate temperature transfer | Arrhenius fit fails at new temperature | Phase, diffusion path, or microstructure changed | Residuals and microscopy versus temperature | Partition regimes or adopt state-aware behavior |
| Secondary creep treated as full lifetime | Deformation fits but rupture is missed | Primary and tertiary stages omitted | Compare full curves and rupture times | Add transient and damage/rupture modeling |
| Tertiary damage ignored | Nonconservative remaining life | Cavitation, necking, or softening absent | Inspect accelerating rate, area loss, and damage evidence | Calibrate a validated full-life model |
| Sparse long-duration data overfit | Unstable extrapolation | Too many correlated parameters | Profiles, bootstrap, and held-out long tests | Reduce model complexity and widen uncertainty |
| Activation energy units wrong | Implausible rate changes | kJ/mol mixed with J/mol or wrong gas constant | Dimensional audit and manual exponent check | Convert `Q` and `R` consistently |
| Celsius used in Arrhenius term | Catastrophic temperature error | Absolute temperature not used | Assert kelvin and inspect value range | Convert to kelvin at ingestion |
| Time-unit mismatch | Rate/life error by factors of 3600 or more | Hours and seconds mixed | Recompute one observation manually | Canonicalize time and record source units |
| Stress units hidden in coefficient | Parameters cannot transfer between scripts | Dimensional power law used without reference stress | Inspect equation and units of `A` | Use dimensionless stress ratio or explicit units |
| Engineering and true quantities mixed | Artificial tertiary acceleration or rate bias | Area change and strain definitions inconsistent | Reprocess raw force/extension consistently | Declare and transform measures appropriately |
| Poor parameter identifiability | Large parameter changes with similar curves | Narrow stress-temperature-path coverage | Correlation matrix, profiles, and data-removal tests | Add discriminating histories or simplify model |
| Minimum rate selected inconsistently | Scatter tied to analyst choice | Smoothing/window policy varies | Repeat automated and blinded extraction | Define and version the extraction method |
| Stress redistribution neglected | Wrong hot spot and life | Initial elastic field treated as permanent | Compare transient FE stress maps | Integrate creep under evolving stress |
| Constant-stress law used for relaxation | Stress decay is inconsistent | Fixed total strain not enforced | Single-element fixed-strain benchmark | Solve coupled elastic-creep evolution |
| Excessive FE time increments | Missed primary or tertiary response | Adaptive stepping governed only by equilibrium | Time-step convergence of state variables | Limit increments by constitutive evolution |
| Constitutive integration errors | Negative damage, oscillations, or cutbacks | Local Newton/tangent/state update is faulty | Material-point tests and residual histories | Correct integration, tangent, and substepping |
| Damage localization is mesh dependent | Rupture time changes with refinement | Local softening lacks objectivity | Mesh study of dissipation and life | Regularize or use an objective failure measure |
| Uniaxial rupture law used multiaxially | Wrong failure location | Hydrostatic/maximum stress effects omitted | Compare notched or multiaxial validation tests | Calibrate a suitable multiaxial measure |
| Parameters transferred across alloy states | Strong batch or heat-treatment bias | Microstructure differs | Stratify residuals by material state | Calibrate state-specific or hierarchical models |
| Rupture correlation used outside its window | Implausible long-time life | Larson-Miller or other relation extrapolated | Mark calibration hull and withheld tests | Restrict use and propagate model discrepancy |
| Creep-fatigue fractions treated as universal | Dwell/waveform ranking fails | Interaction and oxidation omitted | Compare representative dwell-cycle tests | Use a validated interaction framework |
| Fit quality confused with mechanism validity | Accurate interpolation, wrong response to path changes | Empirical flexibility hides wrong physics | Validate load changes, temperatures, and microstructure | Separate predictive validation from mechanism claims |

## Best Practices

1. Define the requested observable and terminal event before selecting a creep equation.
2. Preserve raw force, extension, temperature, time, atmosphere, and specimen geometry.
3. Use kelvin, explicit stress units, explicit time units, and a dimensionless stress ratio in fitted equations.
4. Retain full creep curves instead of storing only minimum rate and rupture time.
5. Partition specimens by heat, product form, weld region, orientation, and environment.
6. Calibrate constant-load, load-change, and relaxation behavior when service paths require them.
7. Verify the local constitutive update before running a component model.
8. Converge both mesh and time increment on the final deformation and life decisions.
9. Track evolving true stress and geometry when deformation is no longer small.
10. Separate rate, deformation, damage, and rupture validation.
11. Quantify parameter correlation, censored observations, and model discrepancy.
12. Compare multiple extrapolation models and preserve direct long-duration validation data.
13. Treat grain size, precipitate state, oxidation, and phase evolution as state variables or domain limits when they change during service. [EXPERT REVIEW NEEDED]
14. Preserve software version, material subroutine, state-variable definitions, solver tolerances, and restart behavior.
15. Require engineering review when results support allowable stress, remaining life, inspection, or safety decisions.

## Integration with Other Skills

- **[fem-multiphysics](../fem-multiphysics/SKILL.md):** Implement transient thermal-mechanical fields, nonlinear solvers, and verified boundary conditions.
- **[crystal-plasticity-workflow](../crystal-plasticity-workflow/SKILL.md):** Represent anisotropic slip, grain interactions, and microstructure-sensitive creep where data support the added resolution.
- **[fatigue-analysis-workflow](../fatigue-analysis-workflow/SKILL.md):** Couple dwell, relaxation, and cyclic response in creep-fatigue assessments without using universal damage fractions.
- **[phase-field-workflow](../../phase-field-microstructure/phase-field-workflow/SKILL.md):** Model coupled microstructure or damage evolution only with calibrated thermodynamics, kinetics, and length scales.
- **[reproducibility-workflow](../../scientific-data/reproducibility-workflow/SKILL.md):** Preserve test provenance, fitting code, material models, and FE state definitions.
- **[scientific-visualization](../../scientific-data/scientific-visualization/SKILL.md):** Plot full curves, rate minima, residuals, redistribution, damage, censoring, and extrapolation domains.
- **[small-data-ml](../../scientific-ml/small-data-ml/SKILL.md):** Fit sparse creep datasets with grouped validation and physically informed constraints.
- **[surrogate-validation](../../surrogate-active-learning/surrogate-validation/SKILL.md):** Test uncertainty calibration and distribution shift across stress, temperature, and material state.

## Key References

1. Norton, F. H. (1929). *The Creep of Steel at High Temperatures*. McGraw-Hill. https://books.google.com/books?id=vswxAAAAMAAJ
2. Frost, H. J., & Ashby, M. F. (1982). *Deformation-Mechanism Maps: The Plasticity and Creep of Metals and Ceramics*. Pergamon Press. https://engineering.dartmouth.edu/defmech/
3. Riedel, H. (1987). *Fracture at High Temperatures*. Springer. https://doi.org/10.1007/978-3-642-82961-1
4. Larson, F. R., & Miller, J. (1952). A time-temperature relationship for rupture and creep stresses. *Transactions of the ASME*, 74(5), 765-771. https://doi.org/10.1115/1.4015909
5. Monkman, F. C., & Grant, N. J. (1956). An empirical relationship between rupture life and minimum creep rate in creep-rupture tests. *Proceedings of ASTM*, 56, 593-620. No DOI located in the bibliographic sources used for verification.
6. Hayhurst, D. R. (1972). Creep rupture under multi-axial states of stress. *Journal of the Mechanics and Physics of Solids*, 20(6), 381-382. https://doi.org/10.1016/0022-5096(72)90015-4
7. Dyson, B. (2000). Use of CDM in materials modeling and component creep life prediction. *Journal of Pressure Vessel Technology*, 122(3), 281-296. https://doi.org/10.1115/1.556185
8. Rabotnov, Y. N. (1969). *Creep Problems in Structural Members*. North-Holland. No DOI located in the bibliographic sources used for verification.
9. ASTM International. (2024). *ASTM E139-24: Standard Test Methods for Conducting Creep, Creep-Rupture, and Stress-Rupture Tests of Metallic Materials*. https://doi.org/10.1520/E0139-24
10. International Organization for Standardization. (2023). *ISO 204:2023 Metallic materials - Uniaxial creep testing in tension - Method of test* (4th ed.). https://www.iso.org/standard/82162.html