---
name: crystal-plasticity-workflow
domain: continuum-modeling
maturity: draft
tools:
  - DAMASK
  - MFront
  - Abaqus
  - MOOSE
  - gmsh
  - orix
  - numpy
  - scipy
  - matplotlib
tested_versions:
  DAMASK: "3.0.0"
  MFront: "4.2.0"
  orix: "0.12.1"
  numpy: "1.26.4"
  scipy: "1.12.0"
  matplotlib: "3.8.0"
requires_gpu: false
estimated_runtime: hours-to-weeks
---

# Crystal Plasticity Workflow

## Description

This skill covers crystal plasticity and crystal plasticity finite element (CPFE) modeling of metals: orientation representations and conventions, slip-system definitions for FCC/BCC/HCP, resolved shear stress and Schmid factors, the multiplicative decomposition `F = Fe Fp`, rate-dependent and rate-independent flow rules, hardening laws, single-crystal and polycrystal (RVE) simulation, texture evolution, parameter calibration and identifiability, and validation against stress-strain curves and crystallographic observables. Invoke this skill when anisotropy, crystallographic slip, texture evolution, or grain-scale heterogeneity control the mechanical response and an isotropic or phenomenological anisotropic plasticity model is insufficient.

## Domain Context

Crystal plasticity resolves plastic deformation onto crystallographic slip systems rather than treating the material as an isotropic continuum. It sits above dislocation dynamics and below macroscopic phenomenological plasticity: it does not resolve individual dislocations, but it carries crystallographic orientation, slip-system geometry, and their evolution, which lets it predict anisotropy, lattice rotation, and texture development that isotropic J2 plasticity cannot. Choosing the right model tier matters — isotropic plasticity for texture-free bulk response, phenomenological anisotropic (e.g. Hill/Barlat) yield surfaces for formability without explicit crystallography, crystal plasticity when slip-system activity and orientation evolution are the target, and CPFE when grain morphology and intergranular stress fields must be spatially resolved.

The kinematic core is the multiplicative decomposition of the deformation gradient `F = Fe Fp`, where `Fp` accumulates plastic shear on slip systems and `Fe` carries elastic stretch and lattice rotation. The plastic velocity gradient `Lp = Fp_dot Fp^{-1}` is a sum over slip systems of shear rate times the Schmid tensor (slip direction dyad slip-plane normal). The resolved shear stress on each system is the projection of the Mandel/second Piola-Kirchhoff stress onto its Schmid tensor, and slip is driven by that resolved stress relative to a slip resistance. Rate-dependent (viscoplastic power-law) formulations regularize slip-system selection and improve convergence at the cost of a rate-sensitivity parameter; rate-independent formulations require an active-set or singular-value treatment of the ambiguous slip-selection problem. [EXPERT REVIEW NEEDED]

Orientation is where quiet errors originate. Euler angles (commonly Bunge ZXZ), rotation matrices, and quaternions each encode crystal-to-sample orientation, but conventions differ across codes and EBSD vendors: active vs passive rotations, crystal-to-sample vs sample-to-crystal direction, degrees vs radians, and the reference frame attached to HCP. A misapplied convention produces a simulation that runs, converges, and is wrong — predicting the wrong texture and the wrong anisotropy. Slip-system definitions also require care: direction and plane indices must be normalized correctly, and for HCP the four-index Miller-Bravais notation and the c/a ratio determine which slip and twinning families are geometrically consistent.

CPFE couples this constitutive model to a finite-element solution over a representative volume element (RVE) or explicit microstructure. Results inherit every FEM concern — mesh sensitivity, boundary conditions, time increment, and nonlinear convergence — plus crystal-plasticity-specific ones: RVE size relative to the number of grains needed for statistical representativeness, orientation assignment, and grain morphology. Parameters (slip resistances, hardening coefficients, rate sensitivity) are calibrated against experiments, but calibration is frequently ill-posed: many parameter sets fit a single macroscopic stress-strain curve equally well. Fitting only the macroscopic curve, without texture or per-grain data, leaves parameters non-identifiable and transfers poorly across alloys, temperatures, and loading paths. Numerical convergence of the solver is never, on its own, evidence of physical validity. [EXPERT REVIEW NEEDED]

## When to Use This Skill

- Use when crystallographic anisotropy, slip-system activity, or texture evolution control the mechanical response.
- Use when predicting lattice rotation, texture development, or orientation-dependent yield and hardening.
- Use for single-crystal deformation studies, Schmid-factor analysis, and slip-system activity prediction.
- Use for polycrystal RVE simulations where grain-scale stress/strain heterogeneity and intergranular fields matter.
- Use when calibrating slip resistances and hardening parameters against combined macroscopic and crystallographic data.
- Use when comparing rate-dependent and rate-independent formulations, or evaluating FCC/BCC/HCP slip and twinning families.
- Use when providing microstructure-informed anisotropic response to a larger continuum, phase-field, or surrogate workflow.

## When Not to Use This Skill

- **When the response is effectively isotropic and texture-free.** Isotropic J2 plasticity is simpler, cheaper, and adequate.
- **When only a macroscopic yield surface is needed.** A phenomenological anisotropic model (Hill, Barlat) may suffice without crystallographic detail.
- **When atomistic mechanisms are the target.** Dislocation nucleation, cross-slip energetics, and core structure require discrete dislocation dynamics or atomistics, not crystal plasticity.
- **When no orientation or texture data are available and none can be assumed.** Crystal plasticity without credible orientation input reduces to an expensive, poorly constrained anisotropic model.
- **When parameters cannot be calibrated or identified.** Guessed slip resistances and hardening coefficients make quantitative predictions fragile.
- **When only the macroscopic stress-strain curve will be used to validate.** That data alone does not identify crystal-plasticity parameters; do not overclaim grain-scale predictions from it.

## Core Concepts

- **Model tier:** Isotropic plasticity, phenomenological anisotropic plasticity, crystal plasticity (material-point), and CPFE (spatially resolved). Each adds capability and cost; choose the lowest tier that captures the controlling physics.
- **Orientation:** Rotation relating crystal and sample frames, represented as Euler angles (e.g. Bunge ZXZ), a rotation matrix, or a quaternion. Convention (active/passive, direction, degrees/radians) must be fixed and documented.
- **Slip system:** A slip direction (Burgers direction) and slip-plane normal, e.g. FCC `{111}<110>`, BCC `{110}<111>` (and `{112}`, `{123}`), HCP basal/prismatic/pyramidal families. Directions and normals must be unit-normalized in the crystal basis.
- **Schmid tensor:** The dyad of unit slip direction and unit slip-plane normal for a system. Its symmetric part projects stress to resolved shear stress; the full tensor enters the plastic velocity gradient.
- **Resolved shear stress (RSS):** Projection of stress onto a slip system's Schmid tensor; slip is driven by RSS relative to slip resistance.
- **Schmid factor:** For uniaxial loading, `m = cos(phi) cos(lambda)`, the geometric factor relating applied stress to RSS; the highest-Schmid-factor system tends to activate first in a single crystal.
- **Critical resolved shear stress (CRSS):** Slip resistance at which a system becomes active; its evolution under hardening is central to the model.
- **`F = Fe Fp`:** Multiplicative split of the deformation gradient into elastic (lattice stretch and rotation) and plastic (crystallographic shear) parts.
- **Plastic velocity gradient `Lp`:** `Lp = sum_alpha gamma_dot^alpha (s^alpha tensor m^alpha)`, the sum of slip-system shear rates times Schmid tensors.
- **Flow rule:** Relation between RSS and shear rate. Rate-dependent power law `gamma_dot = gamma_dot_0 (|tau|/g)^n sign(tau)` regularizes selection; rate-independent forms need active-set handling.
- **Hardening law:** Evolution of slip resistance, e.g. Voce saturation, power-law, or Kocks-Mecking dislocation-density forms, with self and latent hardening via a hardening matrix.
- **Texture:** Statistical distribution of crystal orientations, visualized with pole figures, inverse pole figures, or an orientation distribution function; it evolves by lattice rotation during deformation.
- **Representative volume element (RVE):** A microstructure sample with enough grains and appropriate boundary conditions to yield statistically representative homogenized response.
- **Identifiability:** Whether the available calibration data constrain the model parameters uniquely; macroscopic curves alone often do not.

## Key Workflows

### Workflow 1: FCC single-crystal loading and Schmid-factor analysis

Before running a full simulation, compute Schmid factors from geometry to predict which slip systems should activate. This is a cheap physical check that catches orientation-convention errors early.

```python
import numpy as np

# FCC slip systems: {111}<110>. Directions and normals in the cubic crystal basis.
# Each entry: (slip plane normal, slip direction). Non-normalized integer indices.
FCC_SLIP_SYSTEMS = [
    ((1, 1, 1), (0, 1, -1)), ((1, 1, 1), (1, 0, -1)), ((1, 1, 1), (1, -1, 0)),
    ((1, -1, 1), (0, 1, 1)), ((1, -1, 1), (1, 0, -1)), ((1, -1, 1), (1, 1, 0)),
    ((-1, 1, 1), (0, 1, -1)), ((-1, 1, 1), (1, 0, 1)), ((-1, 1, 1), (1, 1, 0)),
    ((1, 1, -1), (0, 1, 1)), ((1, 1, -1), (1, 0, 1)), ((1, 1, -1), (1, -1, 0)),
]


def schmid_factors(load_direction_crystal, slip_systems=FCC_SLIP_SYSTEMS):
    """Schmid factor m = cos(phi) cos(lambda) for uniaxial load, per slip system.

    load_direction_crystal must be expressed in the CRYSTAL frame. If you have a
    load direction in the SAMPLE frame and an orientation, rotate it first and be
    explicit about active vs passive convention. [EXPERT REVIEW NEEDED]
    """
    b = np.asarray(load_direction_crystal, dtype=float)
    b /= np.linalg.norm(b)
    out = []
    for plane, direction in slip_systems:
        n = np.asarray(plane, dtype=float);  n /= np.linalg.norm(n)
        s = np.asarray(direction, dtype=float); s /= np.linalg.norm(s)
        if abs(np.dot(n, s)) > 1e-8:
            raise ValueError(f"Slip direction not in slip plane: {plane}, {direction}")
        m = float(np.dot(b, n) * np.dot(b, s))  # cos(phi)*cos(lambda)
        out.append({"plane": plane, "direction": direction, "schmid_factor": abs(m)})
    return sorted(out, key=lambda d: d["schmid_factor"], reverse=True)


# Single crystal loaded along [100]: the classic result is m_max = 0.408 on {111}<110>.
top = schmid_factors((1, 0, 0))[0]
assert abs(top["schmid_factor"] - 0.4082) < 1e-3, "Check slip-system table / normalization"
```

### Workflow 2: Polycrystalline RVE construction with assigned orientations

Build a synthetic RVE, assign orientations sampled from a measured or model texture, and confirm the grain count is large enough for statistical representativeness before committing to a production mesh.

```python
import numpy as np


def sample_orientations_euler(n_grains, rng_seed=0):
    """Uniformly random Bunge (ZXZ) Euler angles in radians. Replace with measured
    EBSD orientations or a fitted ODF for a real texture. Uniform sampling requires
    cos(Phi) uniform in [-1, 1], not Phi uniform, to avoid pole clustering."""
    rng = np.random.default_rng(rng_seed)
    phi1 = rng.uniform(0.0, 2 * np.pi, n_grains)
    Phi = np.arccos(rng.uniform(-1.0, 1.0, n_grains))
    phi2 = rng.uniform(0.0, 2 * np.pi, n_grains)
    return np.column_stack([phi1, Phi, phi2])


def rve_representativeness_check(n_grains, elements_per_grain=8):
    """Rough guidance only. Statistical representativeness depends on property,
    anisotropy, and boundary conditions; verify by convergence over RVE realizations."""
    notes = []
    if n_grains < 100:
        notes.append("Fewer than ~100 grains: homogenized response may be realization-dependent.")
    if elements_per_grain < 8:
        notes.append("Fewer than ~8 elements/grain: intragranular fields likely under-resolved.")
    return {
        "n_grains": n_grains,
        "approx_elements": n_grains * elements_per_grain,
        "warnings": notes or ["Passes rough heuristics; still verify by RVE-size convergence."],
    }


euler = sample_orientations_euler(200)
print(rve_representativeness_check(200))
```

For explicit microstructures, synthetic RVEs are commonly generated with DREAM.3D (grain morphology and texture statistics) or Voronoi tessellation, then meshed and passed to the CP solver. Record the seed, grain-size distribution, and texture source with the RVE. [EXPERT REVIEW NEEDED]

### Workflow 3: Rate-dependent CP simulation for an FCC metal (DAMASK)

DAMASK is a spectral/FE crystal plasticity solver that separates material, load, and geometry definitions. The material configuration below is a schematic of a phenopowerlaw FCC phase; exact keys and schema are version-specific, so pin the version and validate against the DAMASK documentation. [EXPERT REVIEW NEEDED]

```yaml
# material.yaml (schematic, DAMASK 3.x phenopowerlaw). Verify keys against the
# installed DAMASK version; do not assume other CP codes accept the same schema.
phase:
  Aluminum:
    lattice: cF                      # FCC
    mechanical:
      elastic:
        type: Hooke
        C_11: 106.75e9               # Pa; single-crystal elastic constants
        C_12: 60.41e9
        C_44: 28.34e9
      plastic:
        type: phenopowerlaw
        N_sl: [12]                   # {111}<110> family
        xi_0_sl: [31e6]              # initial slip resistance (CRSS), Pa
        xi_inf_sl: [63e6]            # saturation slip resistance, Pa
        h_0_sl-sl: [75e6]            # reference hardening, Pa
        h_sl-sl: [1, 1, 1.4, 1.4, 1.4, 1.4, 1.4]  # self/latent hardening matrix
        dot_gamma_0_sl: 0.001        # reference shear rate, 1/s
        n_sl: 20                     # rate sensitivity exponent (1/m)
        a_sl: 2.25                   # hardening exponent
```

```bash
# Run a uniaxial tension load case on a synthetic polycrystal grid.
DAMASK_grid --load tensionX.yaml --geom polycrystal_200grains.vti --material material.yaml
```

Report which slip systems carried significant accumulated shear, the homogenized stress-strain response, and the evolved texture — not just that the run converged.

### Workflow 4: Calibration of slip resistance and hardening parameters

Calibrate against combined macroscopic and crystallographic targets, and explicitly test identifiability. A fit to the macroscopic curve alone is not a validated parameter set.

```python
import numpy as np
from scipy.optimize import least_squares


def residuals(params, exp_strain, exp_stress, run_cp_model, weights=None):
    """params: [xi_0, xi_inf, h_0, n] for a phenomenological FCC model.
    run_cp_model(params, exp_strain) -> simulated stress on the same strain grid.
    Include crystallographic residuals (texture index, pole-figure intensity) when
    available; fitting stress alone leaves parameters non-identifiable."""
    sim_stress = run_cp_model(params, exp_strain)
    r = (sim_stress - exp_stress)
    if weights is not None:
        r = r * weights
    return r


def calibrate(run_cp_model, exp_strain, exp_stress, p0, bounds):
    result = least_squares(
        residuals, x0=p0, bounds=bounds,
        args=(exp_strain, exp_stress, run_cp_model),
        method="trf", diff_step=1e-2,
    )
    # Approximate parameter covariance from the Jacobian at the solution.
    J = result.jac
    dof = max(1, len(exp_stress) - len(p0))
    sigma2 = 2 * result.cost / dof
    try:
        cov = sigma2 * np.linalg.inv(J.T @ J)
        stderr = np.sqrt(np.diag(cov))
        # Large correlations / near-singular J.T@J => non-identifiable parameters.
        corr = cov / np.outer(stderr, stderr)
    except np.linalg.LinAlgError:
        stderr, corr = None, None  # singular: parameters not identifiable from this data
    return {"params": result.x, "success": result.success,
            "stderr": stderr, "correlation": corr}
```

Gaussian-process surrogates and Bayesian optimization are appropriate when each CP evaluation is expensive; use them to explore parameter space and to expose parameter correlations, not to hide non-convergence.

### Workflow 5: Texture evolution and pole-figure comparison

Extract the evolving orientation distribution and compare simulated pole figures against measured EBSD/XRD texture. Texture agreement is a stronger validation than the stress-strain curve because it constrains slip-system activity.

```python
import numpy as np
from orix.quaternion import Orientation
from orix.crystal_map import Phase
from orix.vector import Miller


def load_orientations(euler_bunge_rad, symmetry="m-3m"):
    """Build an orix Orientation set from Bunge (ZXZ) Euler angles in radians.
    Confirm the code you exported from uses the SAME Euler convention. [EXPERT REVIEW NEEDED]"""
    phase = Phase(point_group=symmetry)
    ori = Orientation.from_euler(euler_bunge_rad, symmetry=phase.point_group)
    return ori, phase


def texture_index(euler_bunge_rad):
    """Cheap scalar proxy for texture strength: mean pairwise misorientation.
    A random texture -> large mean misorientation; a sharp texture -> small.
    Use a full ODF (e.g. MTEX) for quantitative texture indices."""
    ori, _ = load_orientations(euler_bunge_rad)
    return float(ori.angle_with_outer(ori).mean())


def plot_pole_figure(euler_bunge_rad, hkl=(1, 1, 1), symmetry="m-3m", out="pf_111.png"):
    ori, phase = load_orientations(euler_bunge_rad, symmetry)
    poles = Miller(hkl=[hkl], phase=phase)
    directions = ori * poles
    fig = directions.scatter(hemisphere="upper", return_figure=True, s=4, alpha=0.4)
    fig.savefig(out, dpi=300)
```

### Workflow 6: Mesh and increment convergence study

CPFE results must be shown independent of mesh resolution, time increment, and (for polycrystals) RVE realization. Strain localization is especially mesh-sensitive and is a common source of spurious conclusions.

```python
import numpy as np


def convergence_report(observable_by_setting, setting_name):
    """observable_by_setting: dict {setting_value: observable_scalar}, e.g.
    {elements_per_grain: yield_stress} or {n_increments: final_texture_index}."""
    settings = sorted(observable_by_setting)
    values = np.array([observable_by_setting[s] for s in settings], dtype=float)
    rel_change = np.abs(np.diff(values)) / np.abs(values[:-1])
    converged = bool(rel_change[-1] < 0.01) if len(rel_change) else False
    return {
        "setting": setting_name,
        "settings": settings,
        "values": values.tolist(),
        "last_relative_change": float(rel_change[-1]) if len(rel_change) else None,
        "converged_below_1pct": converged,
        "note": "Check localization/texture separately; a converged mean can hide "
                "mesh-dependent strain localization.",
    }


# Example: refine elements per grain and check homogenized yield stress stability.
yield_by_mesh = {4: 121.0e6, 8: 118.5e6, 16: 117.9e6, 32: 117.8e6}
print(convergence_report(yield_by_mesh, "elements_per_grain"))
```

### Workflow 7: Validation against macroscopic and crystallographic observables

Layer validation from geometry checks to macroscopic data to crystallographic data. A model that matches the stress-strain curve but predicts the wrong texture has the wrong slip-system activity.

Validation checklist:

- **Geometry:** Schmid factors for known single-crystal orientations reproduce textbook values; slip directions lie in slip planes; orientation round-trips through the code's Euler convention.
- **Elastic anisotropy:** Single-crystal elastic response matches the input stiffness tensor for oriented loading.
- **Macroscopic:** Simulated stress-strain curves match experiments across at least two loading paths or orientations, not one.
- **Crystallographic:** Simulated texture evolution (pole figures, ODF, texture index) matches EBSD/XRD; active slip systems are consistent with observed slip traces.
- **Per-grain (when available):** Grain-resolved lattice rotations or residual stresses match in-situ diffraction or HR-EBSD.
- **Robustness:** Results are independent of mesh, time increment, and RVE realization, with uncertainty reported on calibrated parameters.

Higher-level phenomena — fatigue-indicator parameters, creep, fracture initiation, and homogenization to a macroscale constitutive law — build on a validated crystal plasticity model but are separate modeling steps. Treat them as downstream work and do not report them as validated outputs of this workflow until dedicated skills and validation exist.

## Common Failure Modes

| Failure | Symptom | Likely Cause | Diagnostic | Fix |
|---------|---------|--------------|------------|-----|
| Orientation-convention mismatch | Wrong texture and anisotropy despite clean convergence | Active/passive or crystal-to-sample vs sample-to-crystal mismatch between EBSD and solver | Round-trip a known orientation; check Schmid factors vs textbook | Fix and document the convention; verify against a single-crystal benchmark |
| Degrees/radians confusion | Nonsensical orientations or immediate divergence | Euler angles passed in the wrong unit | Print min/max of Euler inputs; expect radians in most APIs | Convert explicitly and assert expected ranges |
| Incorrect slip-system normalization | Wrong Schmid factors and slip activity | Direction/normal not unit-normalized or wrong basis | Assert `dot(n, s) == 0` and unit norms | Normalize in the crystal basis before building Schmid tensors |
| Wrong lattice basis (HCP) | Missing or spurious slip/twin families | Miller-Bravais indices or c/a ratio mishandled | Recompute families with correct c/a; compare to literature families | Use the proper HCP basis and c/a; validate family list |
| Incorrect tensor transformation | Anisotropy inconsistent with input stiffness | Stiffness or Schmid tensor rotated with the wrong rotation | Check oriented single-crystal elastic response | Use consistent rotation for stress, strain, and stiffness |
| Insufficient RVE size | Homogenized response changes between realizations | Too few grains for the property/anisotropy | Repeat with independent seeds; check spread | Increase grain count; average over realizations |
| Unrealistic boundary conditions | Artificial stiffening or surface artifacts | BCs inconsistent with intended loading/periodicity | Compare periodic vs traction BCs on the same RVE | Use periodic BCs for bulk response; justify chosen BCs |
| Mesh-dependent localization | Shear bands sharpen with refinement without converging | Rate-independent softening or no length scale | Refine mesh and track band width and peak strain | Add rate sensitivity or a regularizing length scale |
| Time increment too large | Non-convergence or inaccurate slip integration | Increment large vs slip-rate/hardening time scale | Increment refinement study on stress and texture | Reduce/adapt increment; check integration scheme |
| Non-identifiable calibration | Many parameter sets fit equally well | Only macroscopic stress-strain data used | Inspect Jacobian conditioning and parameter correlations | Add texture/per-grain targets; report parameter uncertainty |
| Fitting only macroscopic data | Good curve fit, wrong texture and slip activity | Calibration ignores crystallographic observables | Compare simulated vs measured pole figures | Include texture/EBSD residuals in calibration |
| Ignoring texture validation | Confident predictions with wrong micromechanics | Validation stopped at the stress-strain curve | Overlay simulated and measured texture | Require texture agreement before trusting grain-scale outputs |
| Overinterpreting phenomenological parameters | Physical meaning claimed for fit constants | Phenomenological CRSS/hardening read as dislocation physics | Test transferability across paths/temperatures | Treat parameters as effective; use physics-based models for mechanism claims |
| Convergence mistaken for validity | Polished fields reported as validated results | Solver residual met but no physical validation | Check against benchmarks and experiments | Separate numerical verification from physical validation |

## Best Practices

- Choose the lowest model tier that captures the controlling physics; use crystal plasticity only when crystallography or texture actually matter.
- Fix and document the orientation convention (active/passive, direction, Euler variant, degrees/radians) and verify it with a single-crystal Schmid-factor benchmark.
- Normalize slip directions and plane normals in the correct crystal basis; assert orthogonality before building Schmid tensors.
- Verify RVE representativeness by convergence over grain count and independent realizations, not a single microstructure.
- Use periodic boundary conditions for bulk homogenized response unless the physical problem demands otherwise, and justify the choice.
- Run mesh, time-increment, and RVE-realization convergence on the reported observables, and inspect strain localization separately from mean quantities.
- Calibrate against combined macroscopic and crystallographic data; report parameter uncertainty and correlations, and test transferability before extrapolating.
- Treat phenomenological CRSS and hardening coefficients as effective parameters, not direct dislocation physics, unless a physics-based model is used.
- Validate texture evolution against EBSD/XRD; do not claim grain-scale predictions from macroscopic curves alone.
- Record elastic constants, slip families, flow-rule and hardening parameters, orientations, BCs, mesh, increments, and solver settings with every run.
- Mention downstream fatigue, creep, fracture, and homogenization workflows in prose only until dedicated repository skills and validation exist.

## Integration with Other Skills

- **[fem-multiphysics](../fem-multiphysics/SKILL.md):** CPFE is a specialized nonlinear finite-element problem; the weak-form construction, mesh auditing, solver configuration, and convergence practices from FEM multiphysics apply directly.
- **[phase-field-workflow](../../phase-field-microstructure/phase-field-workflow/SKILL.md):** Microstructures and orientation fields from phase-field simulations can seed crystal plasticity RVEs, and crystal plasticity supplies mechanical response for coupled chemo-mechanical models.
- **[calphad-workflow](../../thermodynamics-calphad/calphad-workflow/SKILL.md):** Phase fractions, stability, and temperature-dependent properties from CALPHAD inform which phases and slip/twin systems are active in multiphase alloys.
- **[scientific-visualization](../../scientific-data/scientific-visualization/SKILL.md):** Pole figures, inverse pole figures, stress/strain fields, and convergence plots require reproducible, publication-quality figure conventions.
- **[reproducibility-workflow](../../scientific-data/reproducibility-workflow/SKILL.md):** Orientations, material parameters, RVE seeds, meshes, solver settings, and postprocessing scripts are provenance artifacts that must be versioned.
- **[gaussian-process-materials](../../surrogate-active-learning/gaussian-process-materials/SKILL.md):** Gaussian-process surrogates approximate expensive CPFE responses for parameter studies and calibration.
- **[bayesian-optimization](../../surrogate-active-learning/bayesian-optimization/SKILL.md):** Bayesian optimization guides parameter calibration and microstructure/design searches over CPFE inputs.
- **[surrogate-validation](../../surrogate-active-learning/surrogate-validation/SKILL.md):** Surrogates trained on CPFE outputs need held-out validation, applicability-domain checks, and residual diagnostics before use.
- **[small-data-ml](../../scientific-ml/small-data-ml/SKILL.md):** Calibration and validation datasets for crystal plasticity are typically small; nested CV, strong baselines, and uncertainty quantification apply.

## Key References

1. Roters, F., Eisenlohr, P., Hantcherli, L., Tjahjanto, D. D., Bieler, T. R., & Raabe, D. (2010). *Overview of constitutive laws, kinematics, homogenization and multiscale methods in crystal plasticity finite-element modeling: Theory, experiments, applications.* Acta Materialia, 58, 1152-1211. https://doi.org/10.1016/j.actamat.2009.10.058
2. Roters, F., et al. (2019). *DAMASK - The Dusseldorf Advanced Material Simulation Kit for modeling multi-physics crystal plasticity, thermal, and damage phenomena from the single crystal up to the component scale.* Computational Materials Science, 158, 420-478. https://doi.org/10.1016/j.commatsci.2018.04.030
3. Asaro, R. J. (1983). *Micromechanics of crystals and polycrystals.* Advances in Applied Mechanics, 23, 1-115. https://doi.org/10.1016/S0065-2156(08)70242-4
4. Kalidindi, S. R., Bronkhorst, C. A., & Anand, L. (1992). *Crystallographic texture evolution in bulk deformation processing of FCC metals.* Journal of the Mechanics and Physics of Solids, 40, 537-569. https://doi.org/10.1016/0022-5096(92)80003-9
5. Kocks, U. F., & Mecking, H. (2003). *Physics and phenomenology of strain hardening: the FCC case.* Progress in Materials Science, 48, 171-273. https://doi.org/10.1016/S0079-6425(02)00003-8
6. Bachmann, F., Hielscher, R., & Schaeben, H. (2010). *Texture analysis with MTEX - free and open source software toolbox.* Solid State Phenomena, 160, 63-68. https://doi.org/10.4028/www.scientific.net/SSP.160.63
