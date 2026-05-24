---
name: pinns-scientific-ml
domain: scientific-ml
maturity: draft
tools:
  - PyTorch
  - JAX
  - DeepXDE
  - NVIDIA Modulus
  - TensorFlow
  - numpy
  - scipy
  - matplotlib
tested_versions:
  PyTorch: "2.2.0"
  JAX: "0.4.25"
  DeepXDE: "1.10.2"
  NVIDIA Modulus: "23.09"
  TensorFlow: "2.15.0"
  numpy: "1.26.4"
requires_gpu: true
estimated_runtime: hours-to-days
---

# PINNs for Scientific Machine Learning

## Description

This skill covers physics-informed neural networks (PINNs) for scientific and engineering modeling: strong-form and weak-form physics losses, PDE residuals, boundary and initial condition enforcement, inverse problems, parameter identification, surrogate modeling for PDE systems, optimizer choices, validation, uncertainty analysis, and comparison against FEM, finite difference, spectral methods, and Gaussian-process surrogates. Invoke this skill when a neural model must satisfy known governing equations while fitting sparse data or approximating a parametric PDE solution.

## Domain Context

PINNs approximate unknown fields with neural networks and train them by minimizing a combination of data mismatch and physics residual losses. The physics loss is usually computed by automatic differentiation of the network output with respect to space, time, and parameters. This makes PINNs attractive for inverse problems, sparse-data settings, moving between data and governing equations, and building differentiable surrogates for PDE systems.

PINNs are not a drop-in replacement for finite element, finite difference, finite volume, or spectral solvers. Classical solvers usually remain more reliable for well-posed forward PDEs with known geometry, boundary conditions, and material parameters. PINNs can be useful when data are sparse, parameters are unknown, the solution needs to be differentiable with respect to inputs, or the model must combine noisy measurements and governing equations. They can fail silently through loss imbalance, spectral bias, poor boundary enforcement, non-identifiability, bad nondimensionalization, or weak validation. [EXPERT REVIEW NEEDED]

The physical interpretation of a PINN depends on the exact loss. A network that minimizes a PDE residual at collocation points is not guaranteed to satisfy conservation, boundary conditions, shocks, discontinuities, or sharp interfaces unless the formulation and sampling strategy enforce them. Weak-form or variational PINNs can improve treatment of integral constraints and lower derivative order, but introduce quadrature, test-function, and domain-integration choices. [EXPERT REVIEW NEEDED]

Scientific PINN workflows overlap with continuum modeling, phase-field modeling, heat and mass transfer, mechanics, diffusion, electrochemistry, and surrogate modeling. A credible workflow must state the governing equations, nondimensional variables, loss terms, collocation strategy, boundary treatment, optimizer settings, validation benchmarks, and uncertainty treatment.

## When to Use This Skill

- Use when fitting sparse scientific data while enforcing known PDEs, boundary conditions, or conservation constraints.
- Use when solving inverse problems such as diffusivity, mobility, elastic modulus, reaction rate, source term, or boundary-flux identification.
- Use when building differentiable surrogates for parametric PDE systems over geometry, material parameters, operating conditions, or time.
- Use when automatic differentiation of solution fields is useful for optimization, sensitivity analysis, or parameter inference.
- Use when classical solvers are available as validation references and can generate benchmark or training data.
- Use when coupling neural surrogates to FEM, phase-field, diffusion, heat-transfer, mechanics, or electrochemistry workflows in a carefully validated setting.

## When Not to Use This Skill

- **When a standard PDE solver is fast, stable, and validated.** FEM, finite difference, finite volume, or spectral methods are usually better for routine forward solves.
- **When boundary and initial conditions are uncertain or wrong.** PINNs can fit a physically inconsistent problem with visually plausible fields.
- **When the solution has shocks, discontinuities, sharp interfaces, or multiscale boundary layers without special treatment.** Standard PINNs often struggle with high-frequency or localized features.
- **When parameters are non-identifiable.** Inverse PINNs cannot recover unique material parameters if different parameter sets produce indistinguishable observations.
- **When nondimensionalization and loss scaling are ignored.** Large residual terms can dominate training and hide failure in other constraints.
- **When validation is only visual.** A smooth neural field is not proof of PDE satisfaction, conservation, or predictive accuracy.

## Core Concepts

- **Physics-informed loss:** Objective term that penalizes violation of governing equations, boundary conditions, initial conditions, conservation laws, or constitutive constraints.
- **Strong-form residual:** Pointwise PDE residual evaluated by automatic differentiation at collocation points.
- **Weak-form residual:** Integral residual against test functions, often reducing derivative order and improving conservation-style constraints.
- **Collocation points:** Spatial, temporal, or parameter-space samples where PDE residuals are evaluated.
- **Boundary-condition loss:** Penalty enforcing Dirichlet, Neumann, Robin, periodic, symmetry, flux, or traction conditions.
- **Initial-condition loss:** Penalty enforcing the starting field for transient problems.
- **Data loss:** Mismatch against measurements, simulations, or synthetic labels.
- **Hard boundary condition:** Network architecture or transformation enforces a boundary condition exactly by construction.
- **Soft boundary condition:** Boundary condition appears as a weighted penalty term in the loss.
- **Loss scaling:** Relative weighting or normalization of residual, data, boundary, and initial losses.
- **Spectral bias:** Neural networks tend to learn low-frequency components earlier than high-frequency or sharp localized features.
- **Identifiability:** Whether unknown parameters can be uniquely inferred from the available equations, observations, and boundary conditions.
- **Automatic differentiation:** Differentiation of network outputs with respect to inputs or parameters; high-order derivatives can be unstable or expensive.

## Key Workflows

### Workflow 1: Formulate the PINN problem before training

Write the mathematical problem first. A PINN should not begin as a network architecture search.

Specification checklist:

- Define independent variables: space, time, material parameters, operating conditions, or geometry parameters.
- Define dependent fields: concentration, temperature, displacement, pressure, electric potential, order parameter, or other state variables.
- Write governing PDEs and constitutive laws in dimensional form.
- Nondimensionalize variables and equations or explicitly justify dimensional training.
- Define boundary conditions, initial conditions, source terms, and measurement locations.
- Decide strong-form versus weak-form losses.
- Identify unknown parameters for inverse problems.
- Define validation observables before tuning the loss.

Example record:

```yaml
pinn_problem:
  system: "diffusion in an electrode particle with unknown diffusivity"
  inputs:
    - r_normalized
    - t_normalized
  outputs:
    - c_normalized
  physics:
    pde: "spherical diffusion equation"
    boundary_conditions:
      center: symmetry
      surface: measured_flux
    initial_condition: uniform_concentration
  inverse_parameters:
    - log_diffusivity
  validation_targets:
    - concentration_profiles_from_FEM
    - surface_concentration_time_series
    - mass_conservation_error
```

### Workflow 2: Build physics, boundary, initial, and data losses

A useful PINN loss separates each scientific constraint so failures are visible.

Loss components:

- PDE residual loss over interior collocation points.
- Boundary loss over boundary collocation points.
- Initial-condition loss over initial-time samples.
- Data loss over measurements or high-fidelity simulations.
- Regularization or prior loss for inverse parameters.
- Conservation or integral-constraint loss when local residuals are insufficient.

Minimal pattern:

```python
import torch

def gradients(y, x):
    return torch.autograd.grad(
        y, x, grad_outputs=torch.ones_like(y),
        create_graph=True, retain_graph=True
    )[0]

def diffusion_residual(model, x_t, diffusivity):
    x_t.requires_grad_(True)
    c = model(x_t)
    grad_c = gradients(c, x_t)
    c_x = grad_c[:, 0:1]
    c_t = grad_c[:, 1:2]
    c_xx = gradients(c_x, x_t)[:, 0:1]
    return c_t - diffusivity * c_xx

residual = diffusion_residual(model, interior_points, diffusivity)
loss_pde = torch.mean(residual**2)
```

This is a one-dimensional strong-form sketch. Production models must handle dimensional scaling, geometry, boundary conditions, higher dimensions, and validation explicitly.

### Workflow 3: Sample collocation points and scale losses

Collocation sampling determines where the physics is enforced. Uniform random sampling can miss boundaries, interfaces, early-time transients, singularities, or steep gradients.

Sampling strategies:

- Uniform interior sampling for smooth baseline problems.
- Boundary-enriched sampling for flux, traction, or interface-sensitive problems.
- Time-biased sampling near initial transients.
- Residual-based adaptive sampling where PDE residual remains high.
- Latin hypercube or Sobol sampling for parametric PDE surrogates.
- Separate held-out collocation points for residual validation.

Loss-scaling practices:

- Nondimensionalize equations so residual magnitudes are comparable.
- Track each loss term separately.
- Use adaptive loss weights or curriculum training cautiously.
- Verify that reducing one loss term does not degrade boundary or data constraints.
- Report final PDE, boundary, initial, and data errors separately.

Bad loss scaling is one of the most common reasons a PINN appears to train while violating the physics that matters.

### Workflow 4: Train with staged optimizers and curriculum

PINN training is often more fragile than standard supervised training because the loss landscape mixes PDE residuals, data, and boundary constraints.

Common training pattern:

1. Pretrain on boundary and initial conditions if they are difficult to satisfy.
2. Train with Adam or another stochastic optimizer for broad exploration.
3. Switch to L-BFGS or a quasi-Newton method for deterministic refinement.
4. Increase collocation density or residual-adaptive samples after the coarse solution stabilizes.
5. Monitor each loss term, validation error, and physical conservation metric.
6. Save checkpoints and random seeds because training variance can be large.

Optimizer notes:

- Adam is robust for noisy minibatches but may stall at residual floors.
- L-BFGS often improves smooth full-batch PINNs but can be memory-heavy.
- Learning-rate schedules help when PDE and data losses decrease on different timescales.
- Curriculum training can help hard problems, but it must not hide final-domain failure. [EXPERT REVIEW NEEDED]

### Workflow 5: Use PINNs for inverse problems and parameter identification

Inverse PINNs infer unknown parameters jointly with solution fields. They are most defensible when the unknown parameters are identifiable from the measurements and constraints.

Inverse-problem checklist:

- Parameterize unknowns with physical bounds, e.g. log diffusivity or positive mobility.
- Use measurements that are sensitive to the unknown parameter.
- Check identifiability with synthetic data before fitting experiments.
- Include measurement noise and uncertainty if available.
- Compare inferred parameters against independent experiments, literature, DFT/MD, or FEM calibration.
- Report parameter confidence intervals or posterior-like uncertainty when possible.

Non-identifiability can look like successful training: the network fits data while multiple parameter values satisfy the observed fields.

### Workflow 6: Validate against classical solvers and experiments

PINNs need validation at the level of the scientific claim.

Validation hierarchy:

- Analytical solution for simplified limits.
- FEM, finite difference, finite volume, or spectral reference solution on the same PDE.
- Residual evaluation on held-out collocation points.
- Boundary and initial condition errors measured separately.
- Conservation checks for mass, charge, energy, or momentum.
- Experimental measurements not used in training.
- Sensitivity to collocation sampling, loss weights, network size, random seed, and nondimensionalization.

For surrogate models, validate across the parameter domain, not only at interpolation points. For inverse problems, validate the inferred parameter by predicting an independent experiment or condition.

### Workflow 7: Quantify uncertainty and failure modes

PINN uncertainty is not automatically captured by the trained network. Use explicit methods when uncertainty matters.

Options:

- Deep ensembles over random seeds, architectures, and collocation samples.
- Bayesian neural networks or variational approximations where feasible.
- Monte Carlo over uncertain boundary conditions, material parameters, or observations.
- Gaussian-process surrogates over PINN residuals or outputs for small parameter studies.
- Calibration against held-out simulation or experimental data.

Uncertainty estimates must include model-form uncertainty when the PDE, boundary conditions, or constitutive law are approximate. [EXPERT REVIEW NEEDED]

## Common Failure Modes

| Failure | Symptom | Likely Cause | Diagnostic | Fix |
|---------|---------|--------------|------------|-----|
| Loss imbalance | PDE loss small but BC/data loss poor, or vice versa | Residual terms have incompatible magnitudes | Plot every loss component separately | Nondimensionalize and tune/adapt weights |
| Spectral bias | Smooth solution misses sharp fronts or high-frequency modes | Network learns low-frequency features first | Compare against reference gradients | Use adaptive sampling, architecture changes, or classical solver |
| Poor BC enforcement | Boundary values or fluxes visibly wrong | Soft BC weight too low or boundary sampling sparse | Compute boundary error separately | Hard constraints or boundary-enriched sampling |
| Non-identifiability | Many parameter values fit the data | Measurements not sensitive to parameter | Synthetic identifiability study | Add measurements or constrain parameters |
| Unstable higher derivatives | Noisy residuals or exploding gradients | High-order autodiff through unsuitable network | Inspect derivative distributions | Use smooth activations, weak form, scaling |
| Bad nondimensionalization | Training stalls or one term dominates | Variables differ by many orders of magnitude | Dimensional audit and loss magnitudes | Rescale variables and equations |
| Extrapolation failure | Good training-domain fit, bad new condition | PINN surrogate outside sampled domain | Held-out parameter-domain tests | Expand training domain or use classical solves |
| Visually plausible wrong solution | Nice fields but conservation or PDE residual fails | Validation relies on plots only | Held-out residual and conservation checks | Add quantitative validation suite |
| Optimizer stagnation | Loss plateaus above acceptable residual | Poor initialization, scaling, or stiffness | Run seed and optimizer comparisons | Staged Adam/L-BFGS, curriculum, resampling |
| Hidden data leakage | Surrogate metrics too optimistic | Same simulation condition appears in train/test | Group split by parameter or experiment | Use leakage-safe split design |

## Best Practices

- Start with a classical solver or analytical benchmark before training a PINN.
- Nondimensionalize variables and equations wherever possible.
- Track PDE, boundary, initial, data, and conservation losses separately.
- Use held-out collocation points and held-out physical conditions.
- Compare against FEM, finite difference, spectral, or experimental references for the intended claim.
- Prefer hard boundary conditions when they are simple and exact.
- Run multiple seeds and report variability.
- Save collocation samples, loss weights, architecture, optimizer schedule, random seeds, and software versions.
- Treat inverse parameters as scientific results requiring independent validation.
- Mention useful future domain skills, such as electrochemistry-specific PINNs or operator-learning workflows, in prose until repository skills exist.

## Integration with Other Skills

- **[fem-multiphysics](../../continuum-modeling/fem-multiphysics/SKILL.md):** Provides classical PDE formulations, weak forms, solver baselines, and validation references for PINNs.
- **[phase-field-workflow](../../phase-field-microstructure/phase-field-workflow/SKILL.md):** Supplies phase-field PDEs, microstructure observables, and benchmark problems for physics-informed surrogates.
- **[scientific-visualization](../../scientific-data/scientific-visualization/SKILL.md):** Supports field plots, residual diagnostics, loss curves, parity plots, and error maps.
- **[reproducibility-workflow](../../scientific-data/reproducibility-workflow/SKILL.md):** Tracks code, data, collocation points, random seeds, checkpoints, and software environments.
- **[small-data-ml](../small-data-ml/SKILL.md):** Provides small-data validation practices for sparse experimental or simulation observations.
- **[gaussian-process-materials](../../surrogate-active-learning/gaussian-process-materials/SKILL.md):** Provides Gaussian-process baselines and uncertainty tools for small PDE-surrogate datasets.
- **[bayesian-optimization](../../surrogate-active-learning/bayesian-optimization/SKILL.md):** Guides design or calibration searches using PINN surrogates or inverse-problem objectives.
- **[surrogate-validation](../../surrogate-active-learning/surrogate-validation/SKILL.md):** Validates PINN surrogate predictions under interpolation, extrapolation, and distribution shift.
- **[materials-featurization](../../scientific-data/materials-featurization/SKILL.md):** Supplies descriptors for parametric material-property inputs in scientific ML workflows.
- **[high-throughput-dft](../../electronic-structure/high-throughput-dft/SKILL.md):** Generates first-principles data that can inform continuum parameters or validation targets.

## Key References

1. Raissi, M., Perdikaris, P., and Karniadakis, G. E. (2019). *Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations*. Journal of Computational Physics. https://doi.org/10.1016/j.jcp.2018.10.045
2. Karniadakis, G. E. et al. (2021). *Physics-informed machine learning*. Nature Reviews Physics. https://doi.org/10.1038/s42254-021-00314-5
3. Cuomo, S. et al. (2022). *Scientific machine learning through physics-informed neural networks: Where we are and what's next*. Journal of Scientific Computing. https://doi.org/10.1007/s10915-022-01939-z
4. Wang, S., Yu, X., and Perdikaris, P. (2022). *When and why PINNs fail to train: A neural tangent kernel perspective*. Journal of Computational Physics. https://doi.org/10.1016/j.jcp.2021.110768
5. Kharazmi, E., Zhang, Z., and Karniadakis, G. E. (2019). *Variational physics-informed neural networks for solving partial differential equations*. arXiv. https://arxiv.org/abs/1912.00873
