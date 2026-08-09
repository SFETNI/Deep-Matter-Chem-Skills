---
name: fatigue-analysis-workflow
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
  - DAMASK
  - MFront
tested_versions:
  Python: "unknown"
  NumPy: "unknown"
  SciPy: "unknown"
  pandas: "unknown"
  matplotlib: "unknown"
  rainflow: "unknown"
  Abaqus: "unknown"
  ANSYS Mechanical: "unknown"
  CalculiX: "unknown"
  DAMASK: "unknown"
  MFront: "unknown"
requires_gpu: false
estimated_runtime: minutes-to-weeks
---

## Description

Build, calibrate, and validate fatigue assessments for metallic materials and components. Select among stress-life, strain-life, local-strain, multiaxial critical-plane, defect-sensitive, crystal-plasticity, and fracture-mechanics methods according to the physical stage being predicted and the available evidence.

This skill treats fatigue initiation, microstructurally short-crack behavior, long-crack propagation, and total life as distinct modeling targets. It supports scientific and engineering analysis; it does not replace applicable design codes, certified material allowables, inspection planning, or safety review.

## Domain Context

Fatigue is progressive damage under repeated or fluctuating loading. A useful workflow begins by defining the prediction target:

- **Initiation life**: cycles to a specified detectable crack, stiffness change, or other operational criterion. The definition must accompany every reported value.
- **Crack nucleation**: formation of a crack embryo at a persistent slip band, inclusion, pore, surface feature, interface, or other microstructural site. This is not automatically equivalent to an engineering initiation-life measurement.
- **Short-crack life**: growth while crack dimensions are comparable to microstructural or plastic-zone length scales. Long-crack threshold and closure assumptions may not apply.
- **Long-crack life**: propagation from a characterized crack under conditions where continuum fracture-mechanics parameters are valid.
- **Total life**: an explicitly stated combination of initiation and propagation models, with a consistent handoff crack size and no double counting.

The nominal HCF, LCF, VHCF, and oligocyclic labels are useful descriptors, not universal mechanism boundaries. Cycle-count thresholds vary with material, environment, geometry, temperature, loading mode, and the life definition. [EXPERT REVIEW NEEDED]

Thermomechanical fatigue (TMF) additionally couples cyclic strain and temperature histories. In-phase and out-of-phase labels do not by themselves define the relevant oxidation, creep, phase-transformation, or constitutive response.

## When to Use This Skill

Use this skill when:

- selecting a defensible fatigue model for coupon, subcomponent, or component data;
- reducing constant-amplitude or variable-amplitude histories into auditable cycle measures;
- fitting S-N, strain-life, crack-growth, defect-sensitive, or statistical fatigue models;
- transferring calibrated material behavior to local finite-element fields;
- ranking fatigue-critical regions using multiaxial criteria or fatigue indicator parameters (FIPs);
- assessing how pores, inclusions, surface finish, residual stress, texture, or grain structure change risk;
- designing verification experiments, sensitivity studies, and uncertainty propagation;
- comparing conventional continuum and crystal-plasticity predictions without conflating numerical convergence with physical validation.

## When Not to Use This Skill

Do not use this skill as the sole basis when:

- a regulated design requires a specific code, certified database, safety factor, or damage-tolerance procedure;
- creep rupture, monotonic overload, brittle cleavage, corrosion cracking, fretting, irradiation, or wear dominates and is not represented;
- the load history, temperature, environment, surface state, residual stress, or material condition is unknown;
- a crack is outside the domain where the chosen stress-intensity, J-integral, or small-scale-yielding assumptions hold;
- a fitted relation is being extrapolated beyond its stress, strain, life, crack-size, frequency, temperature, or environment range;
- a visually plausible finite-element or CPFE field is being treated as validated life prediction without independent fatigue evidence.

## Core Concepts

### Method selection

| Prediction target and evidence | Preferred starting point | Required boundary statement |
|---|---|---|
| Nominally elastic, constant-amplitude coupon or component loading | Stress-life (S-N/Basquin) | Stress definition, load ratio, runout treatment, surface and environment |
| Local cyclic plasticity or low-cycle response | Strain-life (Coffin-Manson-Basquin) | Stabilized cyclic response, strain measure, mean stress, failure criterion |
| Elastic global model with a localized notch | Local-strain method with cyclic Ramberg-Osgood and notch correction | Applicability of Neuber or alternative correction and multiaxiality |
| Nonproportional or multiaxial local histories | Critical-plane or invariant method | Plane search, stress/strain tensors, criterion calibration domain |
| Measured variable-amplitude history | Rainflow counting plus an explicitly justified damage rule | Filtering, residual cycles, sequence effects, mean-stress treatment |
| Characterized long crack | Fracture-mechanics crack-growth model | Geometry factor, crack closure/ratio, threshold, validity limits |
| Microstructure-sensitive initiation ranking | CPFE plus calibrated FIP | RVE representativeness, averaging rule, life mapping, uncertainty |
| Pore- or inclusion-controlled fatigue | Defect-sensitive model or explicit defect simulation | Defect metric, location, morphology, residual stress, population statistics |
| TMF | Temperature-dependent constitutive and fatigue model | Phase relation, dwell, oxidation/creep interactions, thermal gradients |

### Stress-life and mean stress

For a Basquin convention written in reversals,

```text
sigma_a = sigma_f_prime * (2 N_f)^b
```

record whether `N_f` means cycles or reversals, whether stress is nominal or local, and how runouts enter the fit. Goodman, Gerber, Soderberg, and Walker relations encode different assumptions; none is a universal mean-stress correction. SWT is instead a fatigue damage parameter based on maximum normal stress and normal strain amplitude. Select and validate the correction for the material, life regime, load ratio, and stress/strain formulation. [EXPERT REVIEW NEEDED]

### Strain-life and local notch response

A common uniaxial strain-life form is

```text
epsilon_a = (sigma_f_prime / E) * (2 N_f)^b
          + epsilon_f_prime * (2 N_f)^c
```

Use stabilized cyclic properties where the material response stabilizes. If cyclic softening, hardening, ratcheting, mean-stress relaxation, or non-Masing behavior matters, retain the cycle evolution or use a suitable constitutive model. Cyclic Ramberg-Osgood parameters are not interchangeable with monotonic parameters.

Neuber's rule is an approximate local elastic-plastic correction, commonly applied to notch problems under constrained conditions. Its accuracy depends on geometry, loading, material behavior, and implementation; compare it with an elastic-plastic finite-element solution for consequential decisions. [EXPERT REVIEW NEEDED]

### Multiaxial fatigue

The resolved shear stress on a slip system is

```text
tau = s . sigma . n
```

where `s` is the slip direction and 
` the plane normal expressed in the same frame as the Cauchy stress tensor. The Schmid factor is a uniaxial special case, not a complete cyclic multiaxial fatigue model.

Fatemi-Socie combines cyclic shear with the maximum normal stress on the candidate plane. Smith-Watson-Topper combines maximum normal stress and normal strain amplitude. Brown-Miller combines shear strain amplitude with normal strain. Criterion choice, plane-search resolution, nonproportional hardening, and the definition of amplitudes materially affect results. [EXPERT REVIEW NEEDED]

### Variable amplitude and cumulative damage

Rainflow counting converts a scalar history into cycle ranges and means. Document turning-point extraction, filtering, binning, treatment of half cycles, and the scalar channel used for multiaxial loading. Palmgren-Miner summation,

```text
D = sum(n_i / N_i)
```

is a linear accounting model. It does not inherently represent sequence, overload retardation, load interaction, evolving residual stress, or a fatigue limit. Treat `D = 1` as a calibrated convention rather than a physical law. [EXPERT REVIEW NEEDED]

### Crack growth and total life

For long-crack growth, a Paris-region model is

```text
da/dN = C * (Delta K)^m
```

Use it only over the fitted range. Near-threshold behavior, crack closure, load ratio, overloads, environment, corrosion, temperature, and rapid fracture require appropriate extensions and validity checks. Define `Delta K`, effective driving force if used, geometry factors, crack-front representation, and termination criterion.

Do not start a Paris-law integration at an arbitrary microscopic flaw and call the result total life. The transition from nucleation or short-crack behavior to long-crack behavior must use an explicit handoff size and compatible life definitions.

### Microstructure, defects, and statistics

CPFE resolves orientation-dependent slip and can generate FIPs based on cyclic plastic strain, plastic work, resolved shear, or related measures. FIP averaging volume, hot-spot rule, boundary treatment, RVE ensemble, and empirical mapping to life are model choices requiring calibration and sensitivity analysis. [EXPERT REVIEW NEEDED]

Murakami-style `sqrt(area)` relations are empirical defect-sensitive frameworks whose constants and applicability depend on material class, hardness, defect location, load ratio, and environment. Do not transfer a surface-defect fit to internal defects or a new alloy without validation. [EXPERT REVIEW NEEDED]

Fatigue scatter often requires censored-data treatment for runouts, specimen-level covariates, batch effects, and probabilistic reporting. A deterministic best-fit curve is not a lower-bound design curve.
### Thermomechanical fatigue

TMF analysis must retain synchronized temperature and mechanical histories, thermal expansion, temperature-dependent cyclic constitutive response, and the selected life metric. Oxidation, creep-fatigue interaction, phase change, coatings, and dwell effects may dominate in different regimes. Their omission or coupling requires system-specific justification. [EXPERT REVIEW NEEDED]

## Key Workflows

### Workflow 1: Define the fatigue data contract

Create a machine-readable record before fitting any model:

```yaml
material:
  alloy: "Ti-6Al-4V"
  processing: "laser powder bed fusion, stress relieved"
  orientation: "build direction parallel to loading axis"
  surface: "as-built"
loading:
  control: "strain"
  waveform: "triangular"
  ratio_definition: "R_epsilon = epsilon_min / epsilon_max"
  temperature_K: 293
  environment: "laboratory air"
life_definition:
  event: "25 percent peak-load drop"
  unit: "cycles"
data:
  runouts_are_censored: true
  specimen_geometry: "documented separately"
  raw_history_retained: true
```

Add calibration/test partition, specimen identifiers, uncertainty, machine compliance correction, and links to raw data. Reject rows whose units, life criterion, or loading ratio cannot be recovered.

### Workflow 2: Fit and audit a stress-life model

Use a log-likelihood or regression appropriate to the variance model and include runouts as right-censored observations when possible. A minimal calculation from already calibrated Basquin parameters is:

```python
import numpy as np


def basquin_cycles(sigma_a, sigma_f_prime, b, calibration_bounds):
    """Return cycles to the calibrated event; Basquin form uses reversals."""
    sigma_a = np.asarray(sigma_a, dtype=float)
    if np.any(sigma_a <= 0) or sigma_f_prime <= 0 or b >= 0:
        raise ValueError("Require positive stresses and b < 0")

    cycles = 0.5 * (sigma_a / sigma_f_prime) ** (1.0 / b)
    lo, hi = calibration_bounds
    if np.any((sigma_a < lo) | (sigma_a > hi)):
        raise ValueError("Stress amplitude is outside the calibration range")
    return cycles
```

Plot observations, runouts, fitted median, uncertainty or tolerance bounds, and residuals against stress, batch, surface state, and load ratio. Do not infer an endurance limit merely because the test stopped.

### Workflow 3: Solve a strain-life relation with an explicit bracket

```python
import numpy as np
from scipy.optimize import brentq


def strain_life_cycles(
    epsilon_a, E, sigma_f_prime, b, epsilon_f_prime, c, reversal_bounds
):
    """Solve Coffin-Manson-Basquin for cycles to the calibrated event."""
    if min(epsilon_a, E, sigma_f_prime, epsilon_f_prime) <= 0:
        raise ValueError("Positive strain and material parameters are required")
    if b >= 0 or c >= 0:
        raise ValueError("This convention requires b < 0 and c < 0")

    def residual(log_reversals):
        reversals = np.exp(log_reversals)
        predicted = (
            sigma_f_prime / E * reversals**b
            + epsilon_f_prime * reversals**c
        )
        return predicted - epsilon_a

    lo, hi = reversal_bounds
    if not (0 < lo < hi):
        raise ValueError("Provide positive ordered reversal bounds")
    reversals = np.exp(brentq(residual, np.log(lo), np.log(hi)))
    return 0.5 * reversals
```

Validate the root bracket and parameter covariance. Compare predictions against held-out hysteresis loops and life data, not only the fitted strain-life points. For notches, compare local-strain approximations with elastic-plastic finite-element fields at representative amplitudes.

### Workflow 4: Reduce a variable-amplitude history without hiding assumptions

1. Preserve the original timestamped signal and units.
2. Remove offsets or noise only with a documented physical rationale.
3. Extract reversals and apply a specified rainflow implementation.
4. Retain range, mean, count, and any dwell or temperature labels.
5. Apply mean-stress and life models inside their calibration domains.
6. Compare linear damage with block or spectrum tests that expose sequence effects.

```python
import numpy as np
import pandas as pd


def miner_damage(counted_cycles, life_function):
    """counted_cycles columns: range, mean, count."""
    required = {"range", "mean", "count"}
    if not required.issubset(counted_cycles.columns):
        raise ValueError(f"Missing columns: {required - set(counted_cycles.columns)}")
    if counted_cycles.empty:
        raise ValueError("At least one counted cycle is required")

    table = counted_cycles.copy()
    numeric = table[["range", "mean", "count"]].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("Cycle ranges, means, and counts must be finite")
    if (numeric["range"] <= 0).any() or (numeric["count"] <= 0).any():
        raise ValueError("Cycle ranges and counts must be positive")
    table[["range", "mean", "count"]] = numeric

    table["amplitude"] = 0.5 * table["range"]
    life = np.asarray(life_function(table["amplitude"], table["mean"]), dtype=float)
    if life.shape != (len(table),) or not np.isfinite(life).all() or (life <= 0).any():
        raise ValueError("The life model must return one positive finite life per row")
    table["predicted_life"] = life
    table["damage"] = table["count"] / table["predicted_life"]
    return float(table["damage"].sum()), table
```

Record the rainflow package and version. Validate its output against a published or hand-checked history before production use.

### Workflow 5: Transfer loads through an elastic-plastic finite-element model

1. Define geometry, contact, preload, residual stress, temperature, and load sequence.
2. Calibrate a cyclic constitutive model against hysteresis loops covering relevant amplitudes and temperatures.
3. Refine the mesh at notches while tracking local stress, strain, plastic strain, and the fatigue metric.
4. Converge both spatial mesh and load/time increments; check equilibrium and constitutive integration.
5. Extract tensor histories in a declared coordinate frame, not only extrema.
6. Compute the fatigue metric outside or inside the solver with a version-controlled implementation.
7. Validate local fields against strain gauges, digital image correlation, diffraction, or another independent measurement where available.

Use [FEM multiphysics](../fem-multiphysics/SKILL.md) for weak-form, solver, boundary-condition, and mesh-convergence practice.

### Workflow 6: Evaluate multiaxial critical planes

Transform the full stress and strain histories to candidate material planes. Refine the angular grid until the maximum damage parameter and critical-plane orientation are stable. Preserve shear sign conventions and distinguish tensor shear from engineering shear strain. For nonproportional loading, use a constitutive model capable of representing the observed additional hardening before applying Fatemi-Socie, SWT, Brown-Miller, or another calibrated criterion.

Validate against held-out proportional and nonproportional tests. A criterion that ranks one loading path correctly may still fail under a different phase angle or mean stress.

### Workflow 7: Build a CPFE-to-FIP workflow

1. Construct an ensemble of RVEs with measured or justified grain size, morphology, texture, phases, and boundary conditions.
2. Verify orientation conventions, slip systems, constitutive parameters, and crystal/laboratory frame transformations.
3. Converge mesh, cycle discretization, number of simulated cycles, and RVE ensemble size.
4. Compare aggregate stress-strain loops and texture-sensitive response against experiment.
5. Compute multiple physically motivated FIPs over declared grains, bands, or averaging volumes.
6. Calibrate the FIP-to-initiation relation on separate specimens and quantify uncertainty.
7. Test whether hot-spot location and life ranking persist across statistically equivalent RVEs.

Use [crystal plasticity workflow](../crystal-plasticity-workflow/SKILL.md) for kinematics, slip, hardening, RVE construction, and texture validation.

### Workflow 8: Assess defect-sensitive and statistical fatigue

Characterize defects in three dimensions when morphology or connectivity matters. Separate surface, near-surface, and internal populations; retain inspection resolution and missed-detection limits. Compare empirical `sqrt(area)` predictions, explicit local-field simulations, and available defect-containing specimen tests. Fit population and life models with specimen or volume effects, censored runouts, and batch structure. Report prediction intervals and sensitivity to the largest-defect assumption rather than a single deterministic life.
## Common Failure Modes

| Failure mode | Why it fails | Detection and correction |
|---|---|---|
| Mixing cycles and reversals | Introduces factor-of-two errors in fitted and predicted life | State the convention in equations, data, code, and plots; test a known point |
| Undefined initiation criterion | Makes life values incomparable | Record the event, detection method, and threshold for every dataset |
| Calling nucleation, initiation, and total life equivalent | Combines physically different stages | Define each stage and the handoff crack size |
| Applying an HCF/LCF/VHCF cycle boundary universally | Mechanisms do not change at universal cycle counts | Classify by observed response and document system-dependent labels |
| Fitting runouts as failures | Biases long-life behavior | Use censored-data methods or report the limitation |
| Inferring an endurance limit from test termination | A runout is not proof of infinite life | Retain censoring and test beyond the proposed knee when justified |
| Using nominal stress at a notch | Misses local concentration and plasticity | Use a validated notch correction or local finite-element solution |
| Using monotonic Ramberg-Osgood parameters cyclically | Misrepresents stabilized hysteresis | Fit cyclic stress-strain data and inspect loop evolution |
| Applying Neuber without checking its domain | Local approximation may fail under multiaxiality or extensive plasticity | Compare with elastic-plastic FE at representative cases |
| Ignoring mean stress or mean-strain relaxation | Distorts fatigue driving force | Measure or model relaxation and validate the selected correction |
| Treating Goodman, Gerber, or SWT as universal | Mean-stress models have different assumptions | Select by material and regime; compare held-out load ratios |
| Using a scalar rainflow history for multiaxial loading without justification | Loses path and phase information | Use a validated multiaxial counting/critical-plane procedure |
| Filtering out damaging peaks | Alters counted ranges and sequence | Archive raw histories and quantify filter sensitivity |
| Dropping rainflow half cycles silently | Changes spectrum damage | State residual-cycle policy and test the implementation |
| Assuming Miner damage is sequence-independent physics | Linear summation omits interaction and retardation | Validate with representative spectrum or block tests |
| Extrapolating an S-N curve below its evidence | Produces unsupported long-life estimates | Enforce calibration bounds and show extrapolated regions |
| Applying Paris law outside the Paris regime | Misses threshold and rapid-growth behavior | Use regime-appropriate growth laws and validity checks |
| Ignoring load ratio, closure, or overloads in crack growth | Misstates effective driving force | Calibrate with relevant histories and closure treatment |
| Starting long-crack integration at a microscopic flaw | Violates continuum and long-crack assumptions | Model or bound short-crack behavior and define a handoff size |
| Double counting initiation and propagation | Inflates total life | Align stage definitions and count each interval once |
| Using a 2D defect metric for a connected 3D defect without qualification | Misrepresents morphology and local fields | Compare sections with tomography or explicit simulations |
| Transferring a `sqrt(area)` fit across defect locations or alloys | Empirical constants and mechanisms differ | Recalibrate by material, surface/internal class, and environment |
| Using one RVE as a statistical prediction | Confounds one realization with population behavior | Run statistically equivalent ensembles and report spread |
| Selecting the maximum element-level FIP | Creates mesh-sensitive hot spots | Define a physical averaging rule and converge it |
| Calibrating and validating a FIP on the same specimens | Hides mapping error and overfit | Reserve independent specimens or microstructures |
| Incorrect orientation or tensor-shear convention | Rotates slip systems or changes amplitudes | Unit-test transformations and tensor/engineering shear factors |
| Unconverged FE mesh or load increments | Produces numerical hot spots or missed cyclic response | Converge fields and final fatigue metric, not only displacement |
| Treating solver convergence as physical validation | A stable numerical answer can represent the wrong model | Compare independent fields, loops, sites, and lives with data |
| Omitting surface state, residual stress, or environment | Removes major fatigue drivers | Match specimen/component state and propagate uncertainty |
| Combining incompatible datasets | Mixes life definitions, units, ratios, and processing states | Audit metadata and fit hierarchical or separate models |
| Ignoring TMF phase, dwell, and thermal gradients | Misses creep, oxidation, and constraint interactions | Retain synchronized histories and validate representative cycles |
| Reporting only median life | Conceals fatigue scatter and reliability | Report data, censoring, uncertainty, and lower-tail metrics as appropriate |

## Best Practices

1. Define the modeled life stage and experimental event before selecting an equation.
2. Preserve raw load, strain, temperature, and environment histories with units and timestamps.
3. Keep nominal, structural, notch, and local stresses distinct.
4. Write every amplitude, range, ratio, tensor-shear, and cycle/reversal convention explicitly.
5. Partition data by specimen or batch before fitting transforms, descriptors, or models.
6. Retain runouts as censored observations and show them on plots.
7. Fit only within experimentally supported loading, temperature, environment, and microstructure domains.
8. Compare against simple baselines before adding high-dimensional FE, CPFE, or surrogate models.
9. Converge the final fatigue quantity with mesh, increment, plane grid, RVE size, and cycle resolution.
10. Validate cyclic constitutive behavior before validating a derived fatigue indicator.
11. Separate parameter calibration, numerical verification, physical validation, and uncertainty assessment.
12. Use held-out load ratios, spectra, microstructures, or geometries to expose distribution shift.
13. Report sensitivity to notch treatment, mean-stress model, damage rule, defect metric, and FIP averaging.
14. Preserve software versions, solver inputs, random seeds, scripts, and intermediate counted cycles.
15. Treat commercial-solver fatigue modules as version-sensitive implementations; verify conventions, defaults, supported criteria, and field extraction against documentation and benchmark cases. [EXPERT REVIEW NEEDED]

## Integration with Other Skills

- Use [FEM multiphysics](../fem-multiphysics/SKILL.md) to construct and verify local continuum fields.
- Use [crystal plasticity workflow](../crystal-plasticity-workflow/SKILL.md) for microstructure-resolved cyclic constitutive response.
- Use [phase-field workflow](../../phase-field-microstructure/phase-field-workflow/SKILL.md) when fatigue damage is coupled to explicit microstructure evolution, without assuming that a qualitative field is a calibrated crack model.
- Use [reproducibility workflow](../../scientific-data/reproducibility-workflow/SKILL.md) for provenance, environments, seeds, and executable records.
- Use [scientific visualization](../../scientific-data/scientific-visualization/SKILL.md) for hysteresis loops, S-N data, residuals, crack-growth curves, FIP maps, and uncertainty displays.
- Use [small-data ML](../../scientific-ml/small-data-ml/SKILL.md) for regularized models when fatigue datasets are sparse and structured.
- Use [Gaussian process materials](../../surrogate-active-learning/gaussian-process-materials/SKILL.md) for calibrated interpolation and uncertainty-aware surrogate studies.
- Use [Bayesian optimization](../../surrogate-active-learning/bayesian-optimization/SKILL.md) only after defining physically admissible design spaces and validation gates.
- Use [surrogate validation](../../surrogate-active-learning/surrogate-validation/SKILL.md) to test grouped splits, extrapolation, calibration, and hidden duplicates.
## Key References

1. Suresh, S. (1998). *Fatigue of Materials* (2nd ed.). Cambridge University Press. https://doi.org/10.1017/CBO9780511806575
2. Schijve, J. (2009). *Fatigue of Structures and Materials* (2nd ed.). Springer. https://doi.org/10.1007/978-1-4020-6808-9
3. Basquin, O. H. (1910). The exponential law of endurance tests. *Proceedings of the American Society for Testing and Materials*, 10, 625-630. No DOI listed in the bibliographic source used for verification.
4. Coffin, L. F., Jr. (1954). A study of the effects of cyclic thermal stresses on a ductile metal. *Transactions of the ASME*, 76, 931-950. No DOI listed in the bibliographic source used for verification.
5. Manson, S. S. (1953). *Behavior of Materials Under Conditions of Thermal Stress* (NACA-TN-2933). National Advisory Committee for Aeronautics. https://ntrs.nasa.gov/citations/19930083626
6. Miner, M. A. (1945). Cumulative damage in fatigue. *Journal of Applied Mechanics*, 12(3), A159-A164. https://doi.org/10.1115/1.4009458
7. Fatemi, A., & Socie, D. F. (1988). A critical plane approach to multiaxial fatigue damage including out-of-phase loading. *Fatigue & Fracture of Engineering Materials & Structures*, 11(3), 149-165. https://doi.org/10.1111/j.1460-2695.1988.tb01169.x
8. Paris, P., & Erdogan, F. (1963). A critical analysis of crack propagation laws. *Journal of Basic Engineering*, 85(4), 528-533. https://doi.org/10.1115/1.3656900
9. McDowell, D. L., & Dunne, F. P. E. (2010). Microstructure-sensitive computational modeling of fatigue crack formation. *International Journal of Fatigue*, 32(9), 1521-1542. https://doi.org/10.1016/j.ijfatigue.2010.01.003
10. Murakami, Y. (2019). *Metal Fatigue: Effects of Small Defects and Nonmetallic Inclusions* (2nd ed.). Elsevier. https://shop.elsevier.com/books/metal-fatigue-effects-of-small-defects-and-nonmetallic-inclusions/murakami/978-0-12-813876-2
11. ASTM International. (2021). *ASTM E466-21: Standard Practice for Conducting Force Controlled Constant Amplitude Axial Fatigue Tests of Metallic Materials*. https://doi.org/10.1520/E0466-21
12. ASTM International. (2021). *ASTM E606/E606M-21: Standard Test Method for Strain-Controlled Fatigue Testing*. https://doi.org/10.1520/E0606_E0606M-21
13. ASTM International. (2023). *ASTM E1049-85(2023): Standard Practices for Cycle Counting in Fatigue Analysis*. https://doi.org/10.1520/E1049-85R23
14. ASTM International. (2024). *ASTM E647-24: Standard Test Method for Measurement of Fatigue Crack Growth Rates*. https://doi.org/10.1520/E0647-24
15. ASTM International. (2025). *ASTM E2368-25: Standard Practice for Strain Controlled Thermomechanical Fatigue Testing*. https://doi.org/10.1520/E2368-25
16. International Organization for Standardization. (2012). *ISO 12107:2012 Metallic materials - Fatigue testing - Statistical planning and analysis of data* (2nd ed.). https://www.iso.org/standard/50242.html