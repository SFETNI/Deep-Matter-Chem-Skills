---
name: fracture-mechanics-workflow
domain: continuum-modeling
maturity: draft
tools:
  - Python
  - NumPy
  - pandas
  - matplotlib
  - Abaqus
  - ANSYS Mechanical
  - CalculiX
  - MOOSE
tested_versions:
  Python: "unknown"
  NumPy: "unknown"
  pandas: "unknown"
  matplotlib: "unknown"
  Abaqus: "unknown"
  ANSYS Mechanical: "unknown"
  CalculiX: "unknown"
  MOOSE: "unknown"
requires_gpu: false
estimated_runtime: hours-to-weeks
---

# Fracture Mechanics Workflow

## Description

Select, construct, and validate fracture-mechanics assessments for cracked materials and components. The workflow distinguishes strength, damage, fracture initiation, stable crack propagation, and unstable fracture, then connects analytical LEFM checks with finite-element, elastic-plastic, cohesive-zone, phase-field, and XFEM models.

This skill supports research and engineering analysis. It does not replace applicable fitness-for-service codes, certified toughness data, inspection requirements, or safety review.

## Domain Context

Fracture mechanics represents the effect of a crack or crack-like defect through crack-tip fields, energy measures, or regularized separation models. A model must state what event it predicts:

- **Strength or collapse:** loss of load capacity without necessarily resolving a crack-growth process.
- **Damage initiation:** onset of a constitutive degradation variable, interface separation, void process, or other model-specific event.
- **Fracture initiation:** a measured or defined onset of crack extension under a specified test method.
- **Stable propagation:** resistance may evolve with crack extension and geometry or constraint.
- **Instability:** applied driving force exceeds the structure's resistance under the governing control mode.

Brittle fracture can often be represented by a relatively small fracture-process zone and an energy or toughness criterion. Ductile fracture generally involves substantial plasticity, void nucleation/growth/coalescence, tearing resistance, and constraint dependence. These labels are mechanism descriptions, not universal thresholds based only on elongation or temperature. [EXPERT REVIEW NEEDED]

Linear elastic fracture mechanics (LEFM) requires linear-elastic response outside a crack-tip process zone that is small relative to crack size, remaining ligament, thickness, and other geometric dimensions. Elastic-plastic fracture mechanics (EPFM) uses measures such as `J` or CTOD when plasticity is not negligible but the selected parameter still characterizes the crack-tip state within its validity domain.

Plane stress and plane strain are idealizations. Thin free surfaces tend toward lower constraint; sufficiently thick interior regions can approach plane strain. A two-dimensional label does not establish the actual three-dimensional constraint state.

## When to Use This Skill

- Assess a detected or postulated flaw under static or quasistatic loading.
- Select between LEFM, EPFM, cohesive-zone, phase-field, XFEM, or explicit-remeshing approaches.
- Calculate or extract `K_I`, `K_II`, `K_III`, energy release rate, `J`, CTOD, or resistance curves.
- Study crack-path sensitivity, mixed-mode loading, residual stress, or geometry effects.
- Calibrate fracture models from fatigue-precracked specimens and validate them on separate configurations.
- Quantify sensitivity to crack size, mesh, constitutive law, temperature, environment, and toughness scatter.

## When Not to Use This Skill

- Do not use a crack-tip parameter when no scale separation exists between the process zone and the structure.
- Do not treat a blunt notch as a sharp crack without a justified equivalence or explicit notch analysis.
- Do not apply quasistatic toughness to impact, dynamic fracture, creep fracture, corrosion-assisted cracking, or fatigue growth without the required rate and mechanism models.
- Do not infer a material property from the peak stress in a singular finite-element field.
- Do not transfer toughness across thickness, orientation, temperature, environment, loading rate, weld region, or microstructure without validation.
- Do not use a numerically converged crack path as evidence that the physical fracture mechanism is correct.

## Core Concepts

### LEFM parameters and constraint

For an isotropic Mode-I configuration,

```text
K_I = Y * sigma * sqrt(pi * a)
```

where `Y` is defined for the exact geometry, crack-length convention, loading, and boundary conditions. `K_I`, `K_II`, and `K_III` characterize opening, in-plane shear, and antiplane shear modes only when the corresponding asymptotic representation is valid.

For isotropic linear elasticity under pure Mode I,

```text
G = K_I^2 / E_prime
E_prime = E                  # plane stress
E_prime = E / (1 - nu^2)    # plane strain
```

Mixed-mode energy relations require the correct mode decomposition and constitutive assumptions. Fracture toughness reported as `K_Q`, `J_Q`, or another provisional value is not automatically a valid material toughness; apply the size, crack-front, loading, and data-quality requirements of the governing test standard.

Small-scale-yielding estimates based on `(K_I / sigma_y)^2` are screening measures whose coefficient and interpretation depend on stress state and convention. Confirm validity using geometry-specific standard requirements and nonlinear analysis for consequential decisions. [EXPERT REVIEW NEEDED]

### Crack-tip fields and geometry

The elastic crack-tip stress field is singular as distance to the mathematical tip approaches zero. Element-level peak stress therefore increases with refinement and is not a material property. Use domain or interaction integrals, displacement correlation, virtual crack extension, or another verified extraction method.

Finite-width corrections, free surfaces, curvature, crack-front shape, load introduction, and finite ligament can invalidate an infinite-body geometry factor. State whether `a` is edge-crack length, half-length, depth, surface length, or another dimension.

### Elastic-plastic fracture

The `J`-integral is path independent for the conditions in its derivation, including appropriate two-dimensional elastic or deformation-theory behavior. Incremental plasticity with unloading, nonproportional loading, body forces, inertia, thermal gradients, residual stress, or crack-face tractions can make naive contour interpretations unreliable. Use solver-specific domain-integral formulations and demonstrate contour convergence. [EXPERT REVIEW NEEDED]

Under valid small-scale-yielding conditions, `J` connects to `K`; under larger-scale yielding, `J` can characterize HRR-type fields for suitable power-law hardening behavior. CTOD definitions and conversions are convention- and geometry-dependent; do not convert between `J`, CTOD, and `K` with an undocumented coefficient. [EXPERT REVIEW NEEDED]

### Propagation and regularized fracture

A propagation law requires a direction criterion and an evolution rule. A toughness initiation condition alone does not define stable growth, and a traction-separation law does not become a material law merely because it reproduces one load-displacement curve.

- **Cohesive-zone models (CZM):** prescribe interface traction versus separation, usually through strength, fracture energy, shape, and mixed-mode evolution. Parameters can be non-identifiable and mesh-sensitive when the cohesive zone is unresolved.
- **Phase-field fracture:** regularizes a crack over a length scale and couples fracture energy to a degradation and irreversibility formulation. Length scale, mesh, split of tensile/compressive energy, residual stiffness, and history treatment affect apparent strength and crack path. [EXPERT REVIEW NEEDED]
- **XFEM:** enriches the approximation so a crack can cut elements without conforming remeshing. Enrichment, level-set updates, integration, stabilization, branching, and propagation criteria remain implementation-sensitive. [EXPERT REVIEW NEEDED]

Mixed-mode initiation and growth criteria are material-, interface-, and loading-path dependent. Calibrate mode interaction with data that span the intended mode mix rather than selecting a criterion by software availability. [EXPERT REVIEW NEEDED]
## Key Workflows

### Workflow 1: Screen a flaw with LEFM

Require a documented stress distribution, flaw geometry and orientation, crack-length convention, geometry factor source, elastic constants, yield strength, and toughness value with its validity designation.

```python
import numpy as np


def mode_i_lefm(sigma_pa, crack_m, geometry_factor, toughness_pa_sqrt_m, E_pa, nu, state):
    """Calculate K, G, and toughness margin using one consistent SI unit system."""
    values = np.asarray([sigma_pa, crack_m, geometry_factor, toughness_pa_sqrt_m, E_pa, nu])
    if not np.isfinite(values).all():
        raise ValueError("All inputs must be finite")
    if min(sigma_pa, crack_m, geometry_factor, toughness_pa_sqrt_m, E_pa) <= 0:
        raise ValueError("Stress, crack size, factor, toughness, and modulus must be positive")
    if not (-1.0 < nu < 0.5):
        raise ValueError("Poisson ratio is outside the isotropic stability range")

    if state == "plane_stress":
        E_prime = E_pa
    elif state == "plane_strain":
        E_prime = E_pa / (1.0 - nu**2)
    else:
        raise ValueError("state must be 'plane_stress' or 'plane_strain'")

    K = geometry_factor * sigma_pa * np.sqrt(np.pi * crack_m)
    G = K**2 / E_prime
    return {"K_Pa_sqrt_m": K, "G_J_per_m2": G, "K_margin": toughness_pa_sqrt_m / K}
```

Do not use a margin alone. Check net-section collapse, small-scale yielding, crack-front constraint, residual stress, uncertainty, and the provenance of `Y` and toughness. Repeat with defensible lower-bound toughness and upper-bound flaw/load combinations.

### Workflow 2: Extract K or J from finite elements

1. Reproduce a published benchmark for the selected element family and extraction method.
2. Represent the crack front, local coordinate system, and crack-face boundary conditions explicitly.
3. Refine around the front using focused rings, quarter-point elements where applicable, or a verified enriched formulation.
4. Request multiple contours or domains along several crack-front positions.
5. Plot `K_I`, `K_II`, `K_III`, or `J` against contour number and position; reject contours contaminated by the tip discretization, boundaries, contact, loads, or material discontinuities.
6. Compare with an analytical geometry factor in the linear-elastic limit.
7. Check energy balance, reaction forces, symmetry, and sign conventions.

Residual stress requires an extraction formulation that handles initial stress consistently. A naive superposition or contour integral may not preserve the assumptions used by the solver. [EXPERT REVIEW NEEDED]

### Workflow 3: Perform an elastic-plastic fracture assessment

Calibrate a true-stress/true-strain constitutive law over the strain and temperature range sampled near the crack. Determine whether `J`-dominance and specimen/component constraint support the intended transfer. Extract load, displacement, crack-mouth opening displacement, `J`, and CTOD; compare several contours and mesh refinements. For stable tearing, use a resistance curve rather than a single initiation point, and update crack size consistently with the chosen growth procedure.

Keep plastic collapse and fracture criteria separate. If a failure-assessment diagram or engineering assessment code is required, implement the specified definitions and partial factors rather than inventing an equivalent rule.

### Workflow 4: Calibrate a cohesive-zone simulation

1. Choose an interface location only when a potential path is known or physically justified.
2. Select a traction-separation shape, mixed-mode measure, damage initiation rule, and evolution law.
3. Calibrate fracture energy using independent fracture data and constrain cohesive strength using local or micromechanical evidence where possible.
4. Resolve the cohesive zone with a mesh-sensitivity study; avoid fitting strength solely to compensate for a coarse mesh.
5. Check artificial compliance before damage and dissipation during growth.
6. Validate load-displacement response, initiation load, crack extension, and path on a separate geometry or mode mix.

Cohesive strength and separation parameters are generally not unique when only global load-displacement data are fitted. Report parameter correlations and alternate fits. [EXPERT REVIEW NEEDED]

### Workflow 5: Run a phase-field fracture model

Define the fracture functional, degradation function, tension-compression split, irreversibility treatment, kinetic or quasistatic evolution, regularization length, residual stiffness, and boundary conditions. Calibrate fracture energy to an appropriate test, then study `h/l` resolution, length-scale sensitivity, load increments, staggered versus monolithic solution, and alternate crack paths. Compare dissipated fracture energy and structural response against a sharp-crack benchmark where available.

Use [phase-field workflow](../../phase-field-microstructure/phase-field-workflow/SKILL.md) for discretization and convergence discipline. Do not interpret the diffuse damage width as a measured process-zone width unless that link was independently established.

### Workflow 6: Resolve mixed-mode fracture

Calculate mode components in a declared local crack-front basis and verify signs with a manufactured displacement field. Sweep mode mix using specimens or loads that span the intended application. Calibrate initiation and propagation direction separately when the model distinguishes them. Compare maximum circumferential stress, maximum energy-release-rate, cohesive interaction, or another candidate criterion against held-out crack angles and loads; do not assume agreement under Mode I establishes mixed-mode validity.

### Workflow 7: Quantify numerical and model sensitivity

Construct a design table varying mesh size, crack-front discretization, contour set, crack shape and size, constitutive law, boundary distance, residual stress, toughness, CZM/phase-field regularization, and load increment. Track the requested engineering result as well as local fields and energy balance. Classify each variation as numerical, geometric, constitutive, or material uncertainty so numerical refinement is not used to conceal model-form spread.

### Workflow 8: Validate against fracture observations

Use fatigue-precracked, dimensionally qualified specimens where the target standard requires them. Separate calibration and validation specimens. Compare force-displacement or moment-rotation histories, crack-mouth opening, initiation event, resistance curve, crack-front tunneling, fracture surface, and final path. Include thickness, orientation, temperature, environment, loading rate, precracking, side grooves, and crack measurement method in the provenance record. Report invalid or provisional tests rather than silently pooling them with valid toughness values.

For statistical characterization, define the target population and distinguish invalid tests from legitimately censored observations. Report sample size, estimator, distributional assumptions, confidence intervals, and any tolerance or lower-bound construction. Use condition- and mechanism-appropriate standards for highly scattered cleavage or transition behavior rather than assuming normality. [EXPERT REVIEW NEEDED]
## Common Failure Modes

| Failure | Symptom | Likely Cause | Diagnostic | Fix |
|---|---|---|---|---|
| LEFM used outside small-scale yielding | K-based result conflicts with nonlinear response | Process zone is not small relative to geometry | Compare plastic-zone screening and elastic-plastic FE fields | Use EPFM or another mechanism-appropriate model |
| Plane stress/plane strain confusion | Toughness or G differs by undocumented factors | Wrong constraint idealization or `E_prime` | Compare thickness response and 3D crack-front fields | State constraint and use qualified data/model |
| Mixed units | Orders-of-magnitude error | MPa, Pa, mm, and m combined | Dimensional audit and benchmark calculation | Convert once to a declared unit system |
| Incorrect crack-length definition | Wrong K despite correct arithmetic | Edge depth, half-length, or total length confused | Draw and label the geometry used by `Y` | Match the source convention exactly |
| Incorrect geometry factor | Nonconservative flaw assessment | Infinite-body or wrong specimen solution used | Compare source dimensions and `a/W` range | Use a validated geometry-specific solution |
| Finite-width effects omitted | K error grows with crack size | Boundary/ligament influence ignored | Sweep `a/W` against analytical or FE benchmark | Apply the correct finite-geometry formulation |
| Peak crack-tip stress reported as a property | Stress rises under refinement | Singular elastic field sampled directly | Plot peak stress versus mesh size | Extract K/J/CTOD or a regularized observable |
| Singular mesh interpreted physically | Artificial local yielding or damage initiation | Tip element field used outside its purpose | Compare field away from first element ring | Use verified extraction and process-zone modeling |
| Poor contour-integral convergence | K or J varies across contours | Tip mesh, loads, boundaries, contact, or inhomogeneity contaminate domains | Plot all contour values by crack-front position | Refine/reposition domains and verify formulation |
| Insufficient crack-tip refinement | No stable integral or cohesive response | Tip/process zone unresolved | Mesh-convergence study of final fracture metric | Refine using a geometry- and model-based criterion |
| Wrong crack-front basis | Swapped or signed mode components | Local directions inconsistent | Apply manufactured Mode-I/II/III fields | Unit-test transformations and signs |
| J path independence assumed under unloading | Contours disagree in cyclic or residual-stress case | Derivation assumptions are violated | Compare loading paths and solver formulation | Use an appropriate domain integral or alternate measure |
| Plastic collapse confused with fracture | Predicted fracture after section collapse | Only toughness criterion checked | Compare reference/collapse load and ligament plasticity | Perform coupled collapse and fracture assessment |
| Provisional toughness treated as valid | Geometry transfer fails | Standard validity requirements not met | Audit test report and qualification flags | Retain `K_Q`/`J_Q` designation or retest |
| Cohesive parameters treated as unique | Different parameter sets fit one curve | Global calibration is non-identifiable | Profile likelihood or alternate calibrated sets | Add local data and report parameter uncertainty |
| Cohesive zone unresolved | Mesh-dependent peak load/path | Too few elements span active zone | Refine along interface and track dissipation | Resolve the zone or use objective regularization |
| Non-objective softening | Dissipation changes with element size | Local damage law lacks regularization | Compare energy under mesh refinement | Use crack-band, nonlocal, cohesive, or phase-field regularization |
| Phase-field length dependence hidden | Strength and path shift with `l` | Regularization parameter treated as numerical only | Sweep `l` and `h/l` separately | Calibrate/justify length and report sensitivity |
| Residual stress superposed incorrectly | Driving force violates equilibrium | Initial stress and redistribution mishandled | Equilibrate initial state and compare reactions | Use solver-supported residual-stress treatment |
| Same data used for calibration and validation | Excellent fit but poor transfer | No independent validation | Trace specimen IDs and data partitions | Reserve independent geometry/mode/condition data |
| Toughness transferred across conditions | Unexpected nonconservative behavior | Constraint, temperature, rate, environment, or microstructure changed | Compare provenance and fracture mechanism | Recalibrate or bound the relevant condition |
| Numerical convergence called physical validity | Stable but wrong crack response | Model form or mechanism untested | Compare independent observables and specimens | Separate verification from validation |

## Best Practices

1. Define the fracture event, crack geometry, loading mode, and material condition before choosing a parameter.
2. Preserve provisional and valid toughness designations from the governing standard.
3. Use fatigue-precracked specimens when a sharp naturalistic crack is required.
4. Track `K`, `J`, CTOD, crack extension, force-displacement response, and energy balance together where applicable.
5. Validate geometry factors and integral extraction against analytical benchmarks before component use.
6. Converge the final fracture decision, not the singular peak stress.
7. Separate mesh sensitivity from cohesive or phase-field regularization sensitivity.
8. Calibrate on one configuration and validate across thickness, geometry, mode mix, or condition.
9. Quantify flaw-size, load, residual-stress, toughness, and model-form uncertainty.
10. Preserve solver version, element/integration choices, crack-front basis, contour definitions, and postprocessing code.
11. Treat environmental, temperature, and rate effects as coupled material-state changes, not scalar correction factors. [EXPERT REVIEW NEEDED]
12. Require engineering review when the result supports structural integrity, inspection intervals, or safety decisions.

## Integration with Other Skills

- **[fem-multiphysics](../fem-multiphysics/SKILL.md):** Build verified weak forms, meshes, nonlinear solvers, and coupled fields for fracture analyses.
- **[crystal-plasticity-workflow](../crystal-plasticity-workflow/SKILL.md):** Resolve orientation-dependent crack-tip plasticity and microstructure-sensitive initiation hypotheses.
- **[fatigue-analysis-workflow](../fatigue-analysis-workflow/SKILL.md):** Connect static fracture resistance to fatigue crack-growth and damage-tolerance workflows without conflating initiation stages.
- **[phase-field-workflow](../../phase-field-microstructure/phase-field-workflow/SKILL.md):** Implement regularized crack evolution with mesh, increment, and energy checks.
- **[reproducibility-workflow](../../scientific-data/reproducibility-workflow/SKILL.md):** Preserve specimen provenance, solver inputs, extraction scripts, and parameter versions.
- **[scientific-visualization](../../scientific-data/scientific-visualization/SKILL.md):** Plot contour convergence, resistance curves, crack fronts, energy histories, and uncertainty without singular-field artifacts.
- **[small-data-ml](../../scientific-ml/small-data-ml/SKILL.md):** Model sparse toughness datasets while retaining group structure and censoring.
- **[surrogate-validation](../../surrogate-active-learning/surrogate-validation/SKILL.md):** Validate reduced-order fracture predictions under geometry and material distribution shift.

## Key References

1. Anderson, T. L. (2017). *Fracture Mechanics: Fundamentals and Applications* (4th ed.). CRC Press. https://www.routledge.com/Fracture-Mechanics-Fundamentals-and-Applications-Fourth-Edition/Anderson/p/book/9781498728133
2. Rice, J. R. (1968). A path independent integral and the approximate analysis of strain concentration by notches and cracks. *Journal of Applied Mechanics*, 35(2), 379-386. https://doi.org/10.1115/1.3601206
3. Hutchinson, J. W. (1968). Singular behaviour at the end of a tensile crack in a hardening material. *Journal of the Mechanics and Physics of Solids*, 16(1), 13-31. https://doi.org/10.1016/0022-5096(68)90014-8
4. Rice, J. R., & Rosengren, G. F. (1968). Plane strain deformation near a crack tip in a power-law hardening material. *Journal of the Mechanics and Physics of Solids*, 16(1), 1-12. https://doi.org/10.1016/0022-5096(68)90013-6
5. Barenblatt, G. I. (1962). The mathematical theory of equilibrium cracks in brittle fracture. *Advances in Applied Mechanics*, 7, 55-129. https://doi.org/10.1016/S0065-2156(08)70121-2
6. Xu, X.-P., & Needleman, A. (1994). Numerical simulations of fast crack growth in brittle solids. *Journal of the Mechanics and Physics of Solids*, 42(9), 1397-1434. https://doi.org/10.1016/0022-5096(94)90003-5
7. Bourdin, B., Francfort, G. A., & Marigo, J.-J. (2000). Numerical experiments in revisited brittle fracture. *Journal of the Mechanics and Physics of Solids*, 48(4), 797-826. https://doi.org/10.1016/S0022-5096(99)00028-9
8. Moes, N., Dolbow, J., & Belytschko, T. (1999). A finite element method for crack growth without remeshing. *International Journal for Numerical Methods in Engineering*, 46(1), 131-150. https://doi.org/10.1002/(SICI)1097-0207(19990910)46:1%3C131::AID-NME726%3E3.0.CO;2-J
9. ASTM International. (2024). *ASTM E399-24: Standard Test Method for Linear-Elastic Plane-Strain Fracture Toughness of Metallic Materials*. https://doi.org/10.1520/E0399-24
10. ASTM International. (2025). *ASTM E1820-25: Standard Test Method for Measurement of Fracture Toughness*. https://doi.org/10.1520/E1820-25
11. International Organization for Standardization. (2021). *ISO 12135:2021 Metallic materials - Unified method of test for the determination of quasistatic fracture toughness* (3rd ed.). https://www.iso.org/standard/78208.html