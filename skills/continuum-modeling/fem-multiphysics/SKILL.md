---
name: fem-multiphysics
domain: continuum-modeling
maturity: draft
tools:
  - FEniCSx
  - FEniCS
  - MOOSE
  - deal.II
  - COMSOL
  - gmsh
  - ParaView
  - numpy
  - matplotlib
tested_versions:
  FEniCSx: "0.7.3"
  FEniCS: "2019.1.0"
  MOOSE: "2024.11.11"
  gmsh: "4.12.2"
  numpy: "1.26.4"
  matplotlib: "3.8.0"
requires_gpu: false
estimated_runtime: hours-to-weeks
---

# FEM Multiphysics

## Description

This skill covers finite element modeling for materials, chemistry, mechanics, transport, and coupled multiphysics problems: weak-form construction, boundary and initial conditions, mesh generation, element choice, solver selection, parameter calibration, convergence studies, validation, uncertainty analysis, and reproducible implementation in FEniCS/FEniCSx, MOOSE, deal.II, COMSOL-style workflows, or similar tools. Invoke this skill when a continuum model requires spatially resolved fields in complex geometries or coupled PDEs that cannot be handled reliably by simple lumped models or closed-form approximations.

## Domain Context

Finite element modeling approximates continuum field equations by converting strong-form PDEs into weak or variational forms on a mesh. Unknown fields such as displacement, concentration, temperature, electric potential, phase fraction, or pressure are represented by basis functions over elements. Boundary conditions, source terms, material laws, and coupling terms determine whether the simulation represents the physical problem or only a numerically polished artifact.

FEM is most powerful when geometry, heterogeneous materials, boundary conditions, or coupled physics matter. It is not automatically predictive: the constitutive model, parameters, mesh, time integration, nonlinear solver tolerances, and validation data define the credibility of the result. Visually smooth stress, temperature, or concentration fields can be wrong if the boundary conditions, units, material model, or mesh convergence are wrong.

For materials and chemistry, FEM often connects thermodynamics, kinetics, mechanics, and transport: diffusion with stress coupling, heat transfer during processing, electrochemical transport in porous electrodes, fracture and plasticity, elastic coherency around precipitates, and multiphysics phase-field models. Coupling can be monolithic, with all fields solved in one nonlinear system, or staggered, with physics blocks solved sequentially. The right choice depends on stiffness, coupling strength, stability, implementation complexity, and available solvers. [EXPERT REVIEW NEEDED]

FEM results should be treated as model-dependent predictions. Calibration against experiments, DFT/MD-derived parameters, CALPHAD thermodynamics, phase-field simulations, or literature benchmarks must be documented with units and uncertainty. Extrapolating calibrated material parameters across temperature, strain rate, composition, state of charge, or microstructure can be invalid. [EXPERT REVIEW NEEDED]

## When to Use This Skill

- Use when geometry, boundary conditions, material heterogeneity, or coupled fields control the response.
- Use when solving elasticity, plasticity, diffusion, heat transfer, electrochemistry, phase-field mechanics, fracture, or coupled transport-mechanics problems.
- Use when analytical solutions are unavailable but benchmark or convergence validation is possible.
- Use when mesh-based local fields, fluxes, stresses, strains, reaction rates, or interface quantities are needed.
- Use when calibrating continuum material parameters from experiments, DFT/MD calculations, phase-field outputs, CALPHAD thermodynamics, or literature.
- Use when comparing monolithic and staggered multiphysics coupling strategies for stability, accuracy, or computational cost.

## When Not to Use This Skill

- **When the governing physics is not known.** FEM discretizes equations; it does not invent a valid constitutive law.
- **When atomic-scale mechanisms are the target.** FEM cannot directly resolve atomistic diffusion paths, dislocation cores, bond breaking, or electronic structure.
- **When a lumped model or analytical solution is sufficient.** FEM can obscure simple physics and add unnecessary numerical error.
- **When material parameters are guessed.** Uncalibrated elasticity, mobility, diffusivity, reaction kinetics, or plasticity parameters make quantitative claims fragile.
- **When validation is only visual.** Smooth contours are not evidence that stresses, fluxes, or reaction rates are correct.
- **When the mesh cannot resolve the relevant length scale.** Boundary layers, cracks, interfaces, diffusion fronts, and contact zones need appropriate resolution or specialized methods.

## Core Concepts

- **Strong form:** PDE and boundary conditions written in differential form, such as balance of linear momentum or mass conservation.
- **Weak form:** Integral form obtained by multiplying the strong form by test functions and integrating by parts. This is the usual FEM implementation target.
- **Trial and test functions:** Approximation space for unknown fields and weighting functions used in the variational formulation.
- **Essential boundary condition:** Dirichlet-type constraint imposed directly on field values, such as displacement, temperature, concentration, or potential.
- **Natural boundary condition:** Neumann/traction/flux-type condition that appears through weak-form boundary terms.
- **Element type:** Shape and interpolation family, such as triangles, quadrilaterals, tetrahedra, hexahedra, Lagrange elements, mixed elements, or discontinuous Galerkin elements.
- **Interpolation order:** Polynomial degree of basis functions. Higher order can improve accuracy but may increase cost and require suitable quadrature.
- **Locking:** Artificial stiffness from poor element choice or formulation, common in nearly incompressible elasticity, bending-dominated problems, and constrained systems.
- **Monolithic coupling:** Solving all coupled fields simultaneously in one residual/Jacobian system.
- **Staggered coupling:** Solving coupled physics sequentially, often easier to implement but potentially less stable for strong coupling.
- **Nonlinear solve:** Newton or quasi-Newton iteration over nonlinear residuals, requiring tolerances, line search, and robust initial guesses.
- **Mesh convergence:** Demonstrating reported observables do not change materially with further mesh refinement.

## Key Workflows

### Workflow 1: Translate physics into a weak form

Start from conservation laws, constitutive assumptions, and boundary conditions before opening a solver GUI or writing code.

Model specification checklist:

- Define fields: displacement, concentration, temperature, electric potential, phase field, pressure, or internal variables.
- Write governing equations in dimensional units.
- Identify source terms, body forces, reaction terms, fluxes, and coupling terms.
- State constitutive laws, parameter sources, and validity ranges.
- Define initial conditions for transient fields.
- Define essential and natural boundary conditions separately.
- Derive or verify the weak form and sign conventions.
- Identify output observables and validation data before tuning parameters.

Example record:

```yaml
model:
  problem: "Li concentration and stress in a spherical electrode particle"
  fields:
    c_li:
      equation: diffusion_with_stress_coupling
      units: mol/m^3
    u:
      equation: linear_elasticity
      units: m
  boundary_conditions:
    concentration:
      surface_flux: "applied current density converted to mol/m^2/s"
      center: symmetry
    displacement:
      center: symmetry
      surface: traction_free
  validation_targets:
    - analytical_diffusion_limit
    - surface_concentration_vs_time
    - peak_tensile_stress
```

### Workflow 2: Build and audit the mesh

Mesh decisions are scientific decisions when gradients, singularities, interfaces, or localized sources matter.

Mesh workflow:

1. Simplify geometry only after identifying which features affect the observable.
2. Choose dimensionality: 1D, axisymmetric, 2D plane strain/stress, or full 3D.
3. Select element type and interpolation order appropriate for the PDE and constraints.
4. Refine near boundaries, interfaces, cracks, reaction zones, or high-gradient regions.
5. Check element quality: aspect ratio, skewness, Jacobian, and curved-boundary representation.
6. Run mesh convergence on physical observables, not just residual norms.
7. Store mesh generation scripts or CAD-to-mesh settings.

For coupled problems, convergence must be checked for each relevant field. A mesh adequate for temperature may be inadequate for stress concentration or electrochemical reaction fronts.

### Workflow 3: Configure transient and nonlinear solvers

Transient multiphysics simulations require both physical and numerical time scales.

Solver checklist:

- Choose explicit, implicit, or semi-implicit time integration based on stiffness.
- Use adaptive timesteps when nonlinear events, boundary changes, or phase transitions occur.
- Monitor conservation of mass, energy balance, or mechanical equilibrium where applicable.
- Set nonlinear Newton tolerances in terms of physically meaningful residuals.
- Use direct solvers for smaller robust problems and iterative solvers with preconditioners for large systems.
- Record linear solver, preconditioner, nonlinear tolerances, timestep policy, and restart behavior.

Common solver choices:

- Direct sparse solvers for small to medium linear systems.
- Krylov methods such as CG, GMRES, or BiCGStab for large systems.
- Algebraic multigrid for elliptic diffusion, elasticity, and pressure-like blocks when appropriate.
- Block preconditioners for mixed or coupled multiphysics systems.
- Newton with line search or damping for strongly nonlinear constitutive laws.

Solver convergence is not physical validation. A tightly converged wrong model is still wrong.

### Workflow 4: Implement a reproducible FEniCSx-style model

FEniCSx and related weak-form frameworks are useful for transparent custom PDE models. Exact APIs change across versions, so treat code examples as patterns and pin software versions. [EXPERT REVIEW NEEDED]

```python
from mpi4py import MPI
import ufl
from dolfinx import fem, mesh
from petsc4py import PETSc

domain = mesh.create_unit_square(MPI.COMM_WORLD, 80, 80)
V = fem.functionspace(domain, ("Lagrange", 1))

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

D = fem.Constant(domain, PETSc.ScalarType(1.0e-14))
source = fem.Constant(domain, PETSc.ScalarType(0.0))

a = D * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
L = source * v * ufl.dx

# Boundary conditions and solver setup omitted here because they are problem-specific.
# Production code must define units, dimensional scales, and physical boundary data.
```

Before production use:

- Replace nondimensional constants with unit-checked physical parameters.
- Add boundary conditions explicitly and test them on a known solution.
- Verify sign conventions for fluxes and tractions.
- Export fields and derived quantities with physical units.
- Save mesh, code, input parameters, and solver logs.

### Workflow 5: Calibrate material parameters

Material parameters link the FEM model to chemistry and materials science.

Parameter sources:

- Experiments: elastic constants, yield curves, fracture toughness, diffusivity, thermal conductivity, reaction rates, conductivity, or swelling.
- DFT/MD: elastic constants, diffusion barriers, defect formation volumes, interfacial energies, or local transport trends.
- CALPHAD and thermodynamic databases: chemical potentials, phase equilibria, driving forces, activities, and temperature-dependent free energies.
- Phase-field simulations: effective interface kinetics, microstructure-informed material response, or coarsening trends.
- Literature: benchmark values and constitutive model forms.

Calibration checks:

- Keep training/calibration data separate from validation data.
- Preserve units, temperature, composition, strain rate, and microstructure metadata.
- Fit parameters to observables that the model can actually represent.
- Use uncertainty bounds instead of a single best-fit value when parameters are weakly identified.
- Test extrapolation across temperature, composition, or loading path before using the model outside calibration conditions.

### Workflow 6: Validate and quantify uncertainty

Validation should be layered from numerical verification to physical evidence.

Verification:

- Manufactured solutions or analytical limits.
- Patch tests for elasticity.
- Mesh refinement and timestep refinement.
- Conservation checks for mass, charge, or energy.
- Solver tolerance sensitivity.

Validation:

- Benchmark problems from the FEM or materials literature.
- Experimental displacement, strain, temperature, concentration, current, or phase-front data.
- Comparison with DFT, MD, CALPHAD, phase-field, or high-fidelity simulations where appropriate.
- Independent datasets not used for parameter calibration.

Uncertainty and sensitivity:

- Vary uncertain material parameters, boundary conditions, geometry, and initial conditions.
- Use local sensitivity, Sobol indices, Gaussian-process surrogates, or Bayesian calibration when simulations are expensive.
- Report which outputs are robust and which depend on poorly constrained inputs.

## Common Failure Modes

| Failure | Symptom | Likely Cause | Diagnostic | Fix |
|---------|---------|--------------|------------|-----|
| Wrong boundary conditions | Plausible fields but wrong flux, stress, or symmetry | Dirichlet/Neumann condition misapplied | Compare to analytical limit and inspect boundary residuals | Re-derive BCs and test minimal benchmark |
| Mesh-dependent results | Observable changes with refinement | Under-resolved gradients or singularities | Mesh convergence study | Refine mesh or improve formulation |
| Locking | Artificially stiff response | Poor element choice for incompressibility, bending, or constraints | Compare element families and analytical tests | Use mixed/reduced/selective integration formulation |
| Unstable time integration | Oscillations or divergence | Timestep too large or explicit method on stiff system | Timestep refinement and stability checks | Use implicit/adaptive scheme |
| Bad nonlinear convergence | Newton stalls or converges to wrong branch | Poor initial guess, discontinuous law, or bad scaling | Residual history and line-search diagnostics | Improve continuation, scaling, damping, or Jacobian |
| Inconsistent units | Results off by orders of magnitude | Mixed SI, eV, nm, MPa, or molar units | Dimensional audit of every parameter | Use unit tests and documented conversions |
| Overconstrained system | Singular matrix or artificial stresses | Conflicting Dirichlet conditions or removed rigid modes | Reaction-force and constraint audit | Apply minimal constraints and symmetry correctly |
| Unphysical material parameters | Fit succeeds but predictions fail | Parameters compensate for missing physics | Validate outside calibration set | Use physically bounded calibration |
| Visually plausible but unvalidated fields | Nice contours with wrong quantitative outputs | No benchmark or convergence evidence | Extract observables and compare to data | Run verification and validation suite |
| Poor preconditioning | Simulation too slow or memory-heavy | Solver/preconditioner mismatch | Solver iteration and residual logs | Use block preconditioners or better scaling |

## Best Practices

- Write the governing equations and weak form before implementing them.
- Track every parameter with units, source, temperature/composition range, and uncertainty.
- Separate numerical verification from physical validation.
- Use mesh and timestep convergence on reported observables, not screenshots.
- Validate boundary conditions on simplified problems.
- Store mesh generation, solver settings, code version, input files, and postprocessing scripts.
- Treat monolithic and staggered coupling choices as numerical modeling assumptions.
- Calibrate parameters only against observables represented by the model.
- Report solver tolerances and convergence behavior for nonlinear multiphysics simulations.
- Mention useful future PINN or specialized electrochemistry skills in prose until corresponding repository skills exist.

## Integration with Other Skills

- **[phase-field-workflow](../../phase-field-microstructure/phase-field-workflow/SKILL.md):** Provides diffuse-interface microstructure models that often require FEM mechanics, elasticity, diffusion, or heat-transfer coupling.
- **[calphad-workflow](../../thermodynamics-calphad/calphad-workflow/SKILL.md):** Supplies thermodynamic driving forces, chemical potentials, and free-energy inputs for continuum simulations.
- **[scientific-visualization](../../scientific-data/scientific-visualization/SKILL.md):** Supports field visualization, convergence plots, residual histories, and publication-quality multiphysics figures.
- **[reproducibility-workflow](../../scientific-data/reproducibility-workflow/SKILL.md):** Tracks meshes, solver settings, parameter files, environment versions, and result provenance.
- **[materials-featurization](../../scientific-data/materials-featurization/SKILL.md):** Provides descriptors for surrogate modeling of material response or parameter studies.
- **[materials-databases-access](../../scientific-data/materials-databases-access/SKILL.md):** Supplies materials data and provenance for parameter selection and validation.
- **[gaussian-process-materials](../../surrogate-active-learning/gaussian-process-materials/SKILL.md):** Builds surrogates for expensive FEM parameter sweeps or calibration studies.
- **[bayesian-optimization](../../surrogate-active-learning/bayesian-optimization/SKILL.md):** Guides design or calibration searches over FEM inputs and operating conditions.
- **[surrogate-validation](../../surrogate-active-learning/surrogate-validation/SKILL.md):** Validates surrogate models trained on FEM simulation outputs.
- **[small-data-ml](../../scientific-ml/small-data-ml/SKILL.md):** Supports sparse experimental calibration and uncertainty-aware validation.

## Key References

1. Zienkiewicz, O. C., Taylor, R. L., and Zhu, J. Z. (2013). *The Finite Element Method: Its Basis and Fundamentals*. Butterworth-Heinemann.
2. Hughes, T. J. R. (2000). *The Finite Element Method: Linear Static and Dynamic Finite Element Analysis*. Dover.
3. Belytschko, T., Liu, W. K., Moran, B., and Elkhodary, K. (2014). *Nonlinear Finite Elements for Continua and Structures*. Wiley.
4. Logg, A., Mardal, K.-A., and Wells, G. N. (2012). *Automated Solution of Differential Equations by the Finite Element Method*. Springer. https://doi.org/10.1007/978-3-642-23099-8
5. Permann, C. J. et al. (2020). *MOOSE: Enabling massively parallel multiphysics simulation*. SoftwareX. https://doi.org/10.1016/j.softx.2020.100430
