---
name: phase-field-workflow
domain: phase-field-microstructure
maturity: draft
tools:
  - MOOSE
  - FiPy
  - PRISMS-PF
  - pycalphad
  - numpy
  - scipy
  - matplotlib
  - ParaView
tested_versions:
  MOOSE: "2024.11.11"
  FiPy: "3.4.5"
  pycalphad: "0.10.5"
  numpy: "1.26.4"
  scipy: "1.12.0"
  matplotlib: "3.8.0"
requires_gpu: false
estimated_runtime: hours-to-weeks
---

# Phase-Field Workflow

## Description

This skill covers phase-field modeling for microstructure evolution: choosing order parameters, constructing free-energy functionals, solving Allen-Cahn and Cahn-Hilliard equations, calibrating thermodynamic and kinetic parameters, verifying numerical convergence, and validating simulations of spinodal decomposition, coarsening, grain growth, precipitation, solidification, and phase separation. Invoke this skill when a microstructure evolution problem requires mesoscale fields and thermodynamic driving forces rather than atomistic trajectories or purely empirical kinetics.

## Domain Context

Phase-field models describe evolving microstructures with continuous fields rather than explicitly tracking sharp interfaces. A field can represent composition, phase fraction, crystallographic variant, grain orientation, order-disorder state, or another coarse-grained state variable. Interfaces emerge from gradient-energy penalties and diffuse over a finite width, which makes topology changes such as coalescence, splitting, and impingement easier to simulate than with sharp-interface methods.

The central approximation is coarse graining. Phase-field models do not resolve atomic jumps, explicit dislocations, molecular mechanisms, or interface structures unless those effects are embedded into free energies, mobilities, boundary conditions, or coupled fields. Results are quantitative only when thermodynamic functions, gradient coefficients, mobilities, elastic constants, anisotropies, noise terms, boundary conditions, and numerical discretization are all calibrated and validated for the material and regime. [EXPERT REVIEW NEEDED]

The physical meaning of the fields matters. Conserved fields such as composition obey mass conservation and are commonly evolved with Cahn-Hilliard-type equations. Non-conserved fields such as structural order parameters, phase indicators, and orientation variables often follow Allen-Cahn-type kinetics. Coupled models combine both, for example precipitation with conserved solute composition and non-conserved precipitate order parameters.

Phase-field workflows sit between thermodynamics, kinetics, mechanics, and numerical PDE solving. CALPHAD databases can provide free-energy surfaces and chemical potentials; DFT or MD can estimate interfacial energies, elastic constants, and defect energetics; experiments calibrate mobilities and validate morphologies; surrogate models can explore parameter uncertainty. Every downstream conclusion inherits assumptions from these inputs.

## When to Use This Skill

- Use when studying mesoscale microstructure evolution where interface motion, phase morphology, and thermodynamic driving forces matter.
- Use when sharp-interface tracking would be fragile because phases split, merge, nucleate, coarsen, or form complex morphologies.
- Use when modeling spinodal decomposition, phase separation, grain growth, solidification, precipitation, coarsening, or coupled chemo-mechanical evolution.
- Use when CALPHAD, DFT, MD, experiment, or literature provide thermodynamic and kinetic parameters that can be mapped into a continuum model.
- Use when comparing morphology, length scale, phase fraction, interface velocity, or coarsening exponent against experiments or analytical limits.
- Use when sensitivity analysis is needed to understand how uncertain mobility, interfacial energy, anisotropy, or boundary conditions affect microstructure evolution.

## When Not to Use This Skill

- **When atomistic mechanisms are the target.** Phase-field models do not directly resolve atomic diffusion paths, bond breaking, dislocation cores, or interface reconstructions.
- **When no physically meaningful field variables can be defined.** A model with arbitrary order parameters may produce attractive images but little predictive value.
- **When parameters are unavailable and cannot be calibrated.** Interface width, gradient coefficients, mobilities, and free energies cannot be guessed for quantitative claims.
- **When the diffuse interface is comparable to the microstructure length scale.** Artificial interface width can distort nucleation barriers, kinetics, and phase morphology.
- **When kinetics are dominated by rare events.** Classical nucleation, defect-mediated transformations, or reaction-limited processes may need stochastic, atomistic, or kinetic models.
- **When validation is only visual.** Qualitative morphology agreement is not enough for quantitative predictions of time, length scale, phase fraction, or driving force.

## Core Concepts

- **Order parameter:** Continuous field describing a phase, variant, orientation, degree of order, or structural state. It may be conserved or non-conserved.
- **Conserved field:** Field whose integral is conserved, such as composition or concentration. Evolution usually follows a Cahn-Hilliard equation.
- **Non-conserved field:** Field without global conservation, such as a phase indicator or grain orientation variable. Evolution commonly follows an Allen-Cahn equation.
- **Free-energy functional:** Spatial integral of chemical, elastic, gradient, interfacial, and coupling terms. Its variational derivatives drive evolution.
- **Chemical free energy:** Local thermodynamic contribution, often fit from CALPHAD, polynomial approximations, Landau expansions, or tabulated functions.
- **Gradient-energy term:** Penalty on spatial gradients that regularizes diffuse interfaces and controls interfacial energy together with barrier height and interface width.
- **Interface width:** Numerical and physical length over which an interface changes smoothly. It must be resolved by the mesh and must not dominate the physics.
- **Mobility:** Kinetic coefficient linking thermodynamic driving force to field evolution. It may depend on phase, composition, temperature, orientation, or stress.
- **Anisotropy:** Direction dependence in interfacial energy, mobility, elasticity, or diffusion that can control dendrites, facets, and texture.
- **Boundary condition:** Mathematical constraint at simulation boundaries, such as periodic, no-flux, fixed composition, fixed temperature, or mechanical displacement.
- **Non-dimensionalization:** Scaling of space, time, energy, and fields. Mistakes here can silently change kinetics by orders of magnitude.
- **Numerical convergence:** Mesh, timestep, domain size, solver tolerance, and interface-resolution independence for the quantities being reported.

## Key Workflows

### Workflow 1: Define the physical model and fields

Start from the transformation mechanism, not from the solver. A phase-field model should state what fields exist, what each field means, which fields are conserved, and which thermodynamic potential drives them.

Model specification checklist:

- Define composition variables and order parameters.
- State whether each field is conserved or non-conserved.
- Identify phases, variants, grains, or precipitates represented by each field.
- Write the free-energy functional, including chemical, gradient, elastic, interfacial, thermal, or coupling terms.
- State the governing equations and boundary conditions.
- Define initial conditions and any stochastic noise terms.
- List all material parameters with units, sources, and uncertainty.
- Identify what observable will be compared with experiment, DFT, MD, analytical theory, or literature.

Example model record:

```yaml
model:
  system: "Ni-Al precipitation at aging temperature"
  fields:
    c_al:
      type: conserved
      equation: Cahn-Hilliard
      units: mole_fraction
    eta_gamma_prime:
      type: non_conserved
      equation: Allen-Cahn
      meaning: ordered precipitate phase indicator
  free_energy:
    chemical: "CALPHAD-derived gamma/gamma_prime free-energy difference"
    gradient: "kappa_eta and kappa_c calibrated to interface energy"
    elastic: "coherent misfit strain with cubic elastic constants"
  boundary_conditions:
    composition: periodic
    order_parameter: zero_flux
  validation_targets:
    - precipitate_radius_distribution
    - phase_fraction
    - coarsening_exponent
```

### Workflow 2: Construct and calibrate the free-energy functional

The free-energy functional controls the thermodynamic driving force. For quantitative work, its terms must be connected to assessed thermodynamics, first-principles calculations, experiments, or documented literature values.

Common sources:

- CALPHAD databases for chemical free energies, phase equilibria, chemical potentials, and driving forces.
- DFT for formation energies, elastic constants, interface energies, defect energies, and metastable phases.
- MD for interfacial mobility, anisotropy, or transport trends where the potential is validated.
- Experiments for phase boundaries, precipitate fractions, coarsening rates, interface velocities, and grain-growth kinetics.
- Literature for benchmark parameter sets, analytical limits, and validation problems.

Parameter mapping checks:

- Match composition variables and reference states between CALPHAD, DFT, and the phase-field model.
- Convert energy units consistently, e.g. J/mol, J/m^3, eV/atom, or nondimensional free energy.
- Ensure gradient coefficients and barrier terms produce the intended interfacial energy and interface width.
- Use thermodynamically consistent derivatives if tabulating free energy or chemical potential.
- Avoid fitting mobility to compensate for incorrect thermodynamics.

### Workflow 3: Choose discretization, mesh, and time integration

Discretization determines whether the simulation solves the intended PDE or a grid artifact.

Common numerical approaches:

- **Finite difference:** Straightforward for structured grids, periodic domains, and spectral-like stencil implementations.
- **Finite element:** Flexible for complex geometry, mechanics coupling, adaptive meshes, and weak-form solvers.
- **Spectral methods:** Efficient for periodic domains and semi-implicit treatment of linear operators.
- **Semi-implicit schemes:** Often used for Cahn-Hilliard equations to handle stiff gradient terms with larger timesteps.

Convergence protocol:

1. Resolve every diffuse interface with enough grid points for the chosen scheme.
2. Run a mesh refinement study for phase fraction, interface velocity, characteristic length scale, or free-energy decay.
3. Run a timestep refinement study for the same observables.
4. Check solver tolerance sensitivity.
5. Confirm domain size is large enough to avoid artificial periodic self-interaction.
6. Verify mass conservation for conserved fields.
7. Confirm total free energy decreases for dissipative closed systems unless external driving is present.

### Workflow 4: Simulate microstructure evolution and extract observables

Do not validate a phase-field model only by looking at snapshots. Extract physical observables from the fields.

Useful observables:

- Phase fraction versus time.
- Characteristic domain size or precipitate radius.
- Particle size distribution and number density.
- Interface velocity or front position.
- Coarsening exponent.
- Structure factor or correlation length.
- Grain-size distribution and misorientation metrics.
- Solute profiles and chemical-potential gradients.
- Free-energy components over time.
- Elastic strain energy and stress fields for coupled mechanics.

Example postprocessing sketch:

```python
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

frames = sorted(Path("results").glob("eta_*.npy"))
times = []
phase_fraction = []
length_scale = []

for frame in frames:
    eta = np.load(frame)
    t = float(frame.stem.split("_")[-1])
    mask = eta > 0.5
    labels, n = ndimage.label(mask)
    sizes = np.bincount(labels.ravel())[1:]
    radii = np.sqrt(sizes / np.pi)

    times.append(t)
    phase_fraction.append(mask.mean())
    length_scale.append(np.mean(radii) if len(radii) else 0.0)

fig, ax = plt.subplots()
ax.plot(times, phase_fraction, label="phase fraction")
ax.set_xlabel("time")
ax.set_ylabel("area fraction")
ax.legend()
fig.savefig("phase_fraction_vs_time.png", dpi=300)
```

This example assumes a 2D thresholded scalar order parameter. Production analysis should use the correct dimensionality, grid spacing, physical units, connectivity, and field definitions.

### Workflow 5: Validate against analytical limits, experiments, and higher-fidelity simulations

Validation should match the intended claim.

Analytical and numerical benchmarks:

- Cahn-Hilliard spinodal wavelength in the linear regime.
- Planar interface profile and interfacial energy.
- Allen-Cahn interface velocity under known driving force.
- Lifshitz-Slyozov-Wagner coarsening trends where assumptions apply.
- Grain-growth scaling laws for idealized isotropic cases.
- Manufactured solutions or solver regression tests.

Materials validation:

- Phase fraction versus heat-treatment time and temperature.
- Precipitate size distribution or coarsening exponent.
- Dendrite arm spacing, solidification path, or segregation profile.
- Grain growth kinetics and texture evolution.
- Interface velocity or transformation front position.
- Composition profiles from microscopy or atom probe tomography.
- DFT, MD, or CALPHAD checks for thermodynamic and kinetic inputs.

Quantitative validation should include uncertainty in parameters, initial conditions, segmentation, grid spacing, and experimental measurements. [EXPERT REVIEW NEEDED]

### Workflow 6: Sensitivity and uncertainty analysis

Phase-field models often have uncertain thermodynamic and kinetic inputs. Sensitivity analysis prevents overinterpreting a single parameter set.

Recommended workflow:

1. Define uncertain parameters: mobility, gradient coefficients, interface energy, free-energy coefficients, elastic constants, nucleation rate, boundary conditions, or initial noise amplitude.
2. Choose observables: length scale, phase fraction, interface velocity, coarsening exponent, or morphology descriptor.
3. Sample parameters using Latin hypercube, Sobol sequences, Gaussian processes, or Bayesian optimization where appropriate.
4. Run reduced-domain or lower-resolution simulations for screening.
5. Validate sensitivity trends on production resolution.
6. Report which conclusions are robust and which depend on parameter choices.

Surrogate models are useful for exploring expensive phase-field parameter spaces, but they must be validated against held-out simulations and should not hide numerical non-convergence.

## Common Failure Modes

| Failure | Symptom | Likely Cause | Diagnostic | Fix |
|---------|---------|--------------|------------|-----|
| Unphysical interface width | Interface dominates particles or phase fraction | Artificial width too large for length scale | Compare interface width to feature size | Recalibrate width or enlarge domain/resolution |
| Grid-dependent kinetics | Coarsening rate changes with mesh | Interface under-resolved or discretization artifacts | Mesh refinement study | Refine mesh and use consistent interface resolution |
| Unstable time stepping | Oscillations, blow-up, negative concentration | Timestep too large for stiffness | Timestep refinement and CFL-like checks | Use smaller timestep or semi-implicit scheme |
| Wrong boundary conditions | Artificial alignment, mass loss, or boundary wetting | Periodic/no-flux/fixed boundary mismatch | Track flux and inspect boundary fields | Set physical BCs and document them |
| Inconsistent thermodynamics | Equilibrium phase fraction contradicts CALPHAD | Free-energy fit or reference mismatch | Compare homogeneous minimization to CALPHAD | Refit free energy and units |
| Mobility overfitting | One experiment fits but transfer fails | Mobility absorbs wrong thermodynamics or missing physics | Validate across temperature/composition | Separate thermodynamic and kinetic calibration |
| Non-dimensionalization mistake | Timescale or length scale off by orders of magnitude | Incorrect scaling of energy, mobility, or gradients | Reconstruct dimensional units end-to-end | Use unit tests and documented scales |
| Qualitative morphology overclaimed | Pretty snapshots but no predictive metric | No quantitative validation observable | Extract phase fraction, length, velocity, distributions | Validate against measured or benchmark observables |
| Missing elastic coupling | Wrong precipitate shape or variant selection | Misfit strain or anisotropic elasticity omitted | Compare with elastic energy scale | Add mechanics coupling if physically required |
| Poor initial conditions | Nucleation density or morphology unrealistic | Arbitrary random noise or seed placement | Sensitivity to seed/noise amplitude | Calibrate initial state or nucleation model |

## Best Practices

- Start with the simplest physically meaningful model and add couplings only when validation demands them.
- Store the free-energy functional, parameters, units, nondimensionalization, mesh, timestep, boundary conditions, and solver tolerances with every run.
- Validate the thermodynamic model independently before evolving microstructure.
- Run mesh, timestep, domain-size, and solver-tolerance convergence studies for reported observables.
- Track conserved quantities and free-energy monotonicity where applicable.
- Use quantitative observables instead of relying on visual similarity.
- Keep calibration data separate from validation data.
- Propagate uncertainty in thermodynamics, mobility, initial conditions, and numerical settings.
- Use visualization for diagnosis, but export raw field data and postprocessing scripts for reproducibility.
- Mention useful future couplings, such as finite-element mechanics or dedicated precipitation modeling skills, in prose until corresponding repository skills exist.

## Integration with Other Skills

- **[calphad-workflow](../../thermodynamics-calphad/calphad-workflow/SKILL.md):** Provides thermodynamic free energies, phase equilibria, chemical potentials, and driving forces for phase-field models.
- **[scientific-visualization](../../scientific-data/scientific-visualization/SKILL.md):** Supports microstructure snapshots, time-series observables, morphology metrics, and publication-quality figures.
- **[reproducibility-workflow](../../scientific-data/reproducibility-workflow/SKILL.md):** Tracks parameter files, solver settings, meshes, field outputs, postprocessing scripts, and provenance.
- **[materials-databases-access](../../scientific-data/materials-databases-access/SKILL.md):** Supplies structures, thermodynamic data, and metadata that may inform parameter selection or validation.
- **[materials-featurization](../../scientific-data/materials-featurization/SKILL.md):** Provides composition and structure descriptors for surrogate modeling or parameter-space analysis.
- **[high-throughput-dft](../../electronic-structure/high-throughput-dft/SKILL.md):** Generates DFT inputs for free-energy, elastic, and interfacial parameter calibration.
- **[dft-convergence](../../electronic-structure/dft-convergence/SKILL.md):** Ensures DFT-derived phase-field parameters are numerically reliable.
- **[gaussian-process-materials](../../surrogate-active-learning/gaussian-process-materials/SKILL.md):** Builds surrogates for expensive phase-field simulations or uncertain parameter responses.
- **[bayesian-optimization](../../surrogate-active-learning/bayesian-optimization/SKILL.md):** Guides calibration or design searches over phase-field input parameters.
- **[surrogate-validation](../../surrogate-active-learning/surrogate-validation/SKILL.md):** Validates surrogate models trained on phase-field outputs.
- **[small-data-ml](../../scientific-ml/small-data-ml/SKILL.md):** Supports analysis when only sparse experimental or high-fidelity simulation data are available.

## Key References

1. Chen, L.-Q. (2002). *Phase-field models for microstructure evolution*. Annual Review of Materials Research. https://doi.org/10.1146/annurev.matsci.32.112001.132041
2. Boettinger, W. J., Warren, J. A., Beckermann, C., and Karma, A. (2002). *Phase-field simulation of solidification*. Annual Review of Materials Research. https://doi.org/10.1146/annurev.matsci.32.101901.155803
3. Steinbach, I. (2009). *Phase-field models in materials science*. Modelling and Simulation in Materials Science and Engineering. https://doi.org/10.1088/0965-0393/17/7/073001
4. Provatas, N. and Elder, K. (2010). *Phase-Field Methods in Materials Science and Engineering*. Wiley-VCH.
5. Moelans, N., Blanpain, B., and Wollants, P. (2008). *An introduction to phase-field modeling of microstructure evolution*. Calphad. https://doi.org/10.1016/j.calphad.2007.11.003
