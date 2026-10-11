---
name: pinns-mpf-workflow
domain: phase-field-microstructure
maturity: draft
tools:
  - PINNs-MPF
  - TensorFlow
  - SciPy
  - numpy
  - matplotlib
tested_versions:
  # Documented, not execution-verified here: requirements-lock.txt of the PINNs-MPF
  # repository pins tensorflow 2.16.2, tensorflow-probability 0.24.0, numpy 1.26.4,
  # scipy 1.17.1 (Python 3.11 conda env). The TensorFlow residual sketch below was
  # not run in this repository.
  # Execution-verified: the numpy finite-difference reference solver, the sampler and
  # the pyramid schedule (Workflows 2, 4, 6) ran with these versions (Python 3.14).
  numpy: "2.5.2"
  scipy: "1.18.0"
requires_gpu: false
estimated_runtime: hours-to-days
---

# PINNs-MPF Workflow

## Description

This skill covers the PINNs-MPF approach: coordinate-based physics-informed neural networks that solve multiphase-field (MPF) interface dynamics. The skill includes:

- the pairwise MPF equation and its nondimensionalization;
- extended domain decomposition over space, time, and phases;
- the master/worker network organization, pyramidal training, and transfer of learning;
- interface-focused adaptive collocation (AMFO);
- the Adam/L-BFGS "wheel of optimizers";
- driven-interface, curvature-shrinkage, and triple-junction benchmarks;
- independent comparison against a conventional finite-difference solver.

Invoke this skill when setting up, reproducing, or auditing PINNs-MPF-style simulations, or when deciding whether a coordinate-network PINN is appropriate for a multiphase interface problem.

## Domain Context

PINNs-MPF (Elfetni & Darvishi Kamachali, *Engineering Analysis with Boundary Elements* 176, 2025, peer reviewed) solves the isotropic pairwise multiphase-field equation with networks mapping (x, y, t) to phase fractions. The published formulation is:

  ∂φ_α/∂t = (μσ/N) Σ_β [ ∇²φ_α − ∇²φ_β + (π²/2η²)(φ_α − φ_β) ] + (μ/N) Σ_β ΔG(φ_α, φ_β)

- **Symbols:** η is the interface width, μ the interface mobility, σ the interface energy, and N the number of locally present phases. This is the Steinbach/OpenPhase-type obstacle-potential MPF family.
- **Coefficient convention (check this before implementing).** The obstacle coefficient is printed as π²/(2η²) in Eq. 1 above. The paper's expanded form uses I_α = ∇²φ_α + (π²/η²)φ_α (Eq. 15), i.e. σ(I_α − I_β), and its two-phase form (Eq. 17) is consistent with Eq. 15. The dimensionless Eqs. 22 and 24 scale differently again.
  - The reference solver in Workflow 6 uses the Eq. 15/17 convention. That convention makes the sine profile of width η stationary, and it reproduced both v = μσκ and v = μΔg in this skill's tests.
  - Its driving term is normalized so the N = 2 limit equals Eq. 17.
  - [EXPERT REVIEW NEEDED: reconcile the obstacle and driving-force coefficients of Eqs. 1, 15, 17, 22 and 24 with the code that produced the published results]
- **Nondimensional time scale:** T = η²/(μσ).
- **Two-phase limit:** the driven case reduces to a scalar equation with driving term proportional to π√(φ(1−φ))/η. Its sharp-interface limits are v = μΔg for a flat driven front and v = μσκ for curvature-driven motion.

**How PINNs-MPF organizes the solve (published version):**

- **Discrete time intervals.** Time is advanced in fixed intervals; training moves to the next interval when the loss falls below a problem-specific threshold. Each interval's prediction becomes the next initial condition, so errors can accumulate over time marching.
- **Extended domain decomposition.** Space is split into batches (boxes). There is one small network per batch *and per phase*. "Vertical" communication shares Laplacian/interaction terms across phases, and a boundary loss couples neighbouring spatial batches.
- **Master/worker roles.** A non-training MASTER-PINN prepares data, assigns WORKER-PINNs, manages time stepping, and orchestrates transfer of learning. Workers run on separate threads.
- **Pyramidal training.** Training starts at the finest subdivision. The best-informed network (most interfacial points) seeds coarser levels with N_coarser = (√N − 2)². The paper reports this was decisive for the triple junction (convergence within about 850 epochs versus stagnation without it).
- **Constraints.** The published version has a sigmoid output for 0 ≤ φ ≤ 1, a *soft* Σφ = 1 loss, and a dynamic renormalization of predictions.

The current public repository has evolved from the published paper. Its README and technical report describe:

- softmax outputs (Σφ = 1 by construction);
- fewer workers for the triple junction (2 × 2);
- TensorFlow-Probability L-BFGS;
- GPU float64;
- its own explicit finite-difference reference solver instead of OpenPhase.

Pyramidal training is exploratory there. Document which version produced any result you cite, and never mix paper numbers with repository numbers.

**Scientific scope:**

- **Covered:** 1D–2D, isotropic interface energy and mobility, a handful of grains (single interface, shrinking grain with or without driving force, a four-grain triple-junction configuration), on 64 × 64-scale domains in the paper.
- **Not covered:** anisotropy, solute or conserved fields, elasticity, 3D, and large grain ensembles.
- **Cost:** the paper states that PINN training is slower than the classical solver for these forward problems. Its value is methodological (mesh-free, differentiable, decomposable), not speed. [EXPERT REVIEW NEEDED: whether any regime exists where PINNs-MPF is cost-competitive with an optimized MPF solver has not been established]

PINNs-MPF is distinct from PINN-Phase (the 2026 preprint): PINN-Phase is a recurrent neural time integrator on a grid with hard admissibility. See the comparison in [pinn-phase-neural-integrator](../pinn-phase-neural-integrator/SKILL.md). Results from the two publications must not be pooled.

## When to Use This Skill

- Use when reproducing or extending the PINNs-MPF benchmarks: 1D traveling interface, driven grain shrinkage, curvature-driven shrinkage, triple junction.
- Use when a multiphase interface problem needs a mesh-free, differentiable solution, e.g. as a stepping stone to inverse identification of mobility or driving force.
- Use when designing domain decomposition, interface-focused sampling, or loss balancing for multiphase PINNs.
- Use when auditing a multiphase PINN result for constraint violations, interface-width artifacts, or time-marching error accumulation.
- Use when teaching or benchmarking how PINN training convergence differs from physical accuracy on problems with analytical sharp-interface limits.

## When Not to Use This Skill

- **For routine forward MPF simulations.** An explicit or semi-implicit MPF solver (e.g. OpenPhase-type finite differences) is faster and better understood. The paper itself reports the PINN is slower.
- **For large polycrystals, 3D microstructures, or long coarsening times.** These are beyond the published benchmarks. Decomposition and per-phase networks scale poorly with phase count.
- **For anisotropic, solute-coupled, elastic, or conserved (Cahn–Hilliard) dynamics.** These are not part of the PINNs-MPF formulation.
- **When interface width is under-resolved.** If η spans only a few grid spacings, both the PINN and the reference suffer finite-width errors that can be mistaken for physics.
- **When transfer to a new configuration is assumed.** Transfer of learning was demonstrated within benchmark families (e.g. across driving forces), not across arbitrary microstructures.

## Core Concepts

- **Pairwise MPF rate:** The sum over phase pairs of curvature (Laplacian difference plus obstacle term) and driving-force contributions, normalized by the local phase count N.
- **Sharp-interface anchors:** A flat driven front gives v = μΔg; curvature flow gives v = μσκ, so a 2D circle obeys R² = R0² − 2μσt. These are the quantitative checks for any MPF solver, PINN or classical.
- **Interface-width resolution:** The ratio η/dx (reference) and collocation density per η (PINN) controls finite-width error. In this skill's reference solver, the R²-slope ratio was 0.887, 0.973, and 0.986 at η = 4, 7, and 9 dx.
- **Coordinate network per batch and phase:** PINN_i(x, y, t) → φ_α in one spatial box. Many small networks replace one large one; the paper found larger widths (6 × 256 and above) failed to converge.
- **Boundary (coupling) loss:** A penalty on mismatch between neighbouring batches along shared edges. Its weight and activation schedule control seam artifacts.
- **Vertical communication:** Exchange of per-phase Laplacian/interaction terms between the networks of different phases in the same batch, needed because each phase's PDE depends on all others.
- **Pyramidal training and transfer of learning:** Warm-starting coarser decompositions, later time intervals, and stalled workers from converged networks.
- **AMFO (adaptive mesh-free optimization):** IC points concentrated at interfaces, collocation points within about η/2 of them (N_f = C_f · N_ini, with C_f ≈ 20–40 in the paper), resampled after each L-BFGS stage.
- **Wheel of optimizers:** Adam every epoch, with periodic L-BFGS refinement.
- **Time-interval threshold:** The loss value that triggers advancing to the next interval. It is problem-specific and set by trial in the paper.

## Key Workflows

### Workflow 1: Formulate and nondimensionalize the MPF problem

**Inputs:** phases and initial geometry, interface energy σ, mobility μ, driving forces ΔG_αβ, interface width η, domain and boundary conditions.
**Assumptions:** isotropic σ and μ; obstacle-type MPF; periodic or symmetric boundaries.
**Output:** a dimensionless problem record and the analytical targets used for acceptance.

```yaml
problem:
  name: curvature_shrinkage_2d
  phases: 2
  equation: "pairwise MPF (PINNs-MPF Eq. 1), isotropic"
  scales:
    length: eta            # x* = x / eta
    time: eta**2 / (mu*sigma)
  dimensionless_inputs:
    R0_over_eta: 6.7
    domain_over_eta: 21.3
    dG_eta_over_sigma: 0.0   # driving-force number; 0 for pure curvature flow
  boundary: periodic
  analytical_target: "R^2 = R0^2 - 2*mu*sigma*t (2D sharp-interface limit)"
  acceptance:
    slope_ratio_vs_reference: [0.97, 1.03]     # vs converged FD reference, not vs theory
    unit_sum_max: 1.0e-4
    bounds_violation_max: 1.0e-4
```

**Silent failure targeted:** comparing a PINN against the sharp-interface law at coarse η/dx. The reference solver itself deviates by around 3–11% (see Core Concepts), so acceptance must be relative to a converged reference with the same η.

### Workflow 2: Build the physics residual and interface-focused point sets

**Inputs:** the problem record, the IC field for the current time interval.
**Outputs:** IC points (interface-weighted plus some bulk), collocation points near interfaces, and the residual loss.

The sampler below is an executed numpy implementation of the AMFO *idea*: interface-weighted IC points, plus collocation inside a ±η/2 band, wrapped periodically. It does not reproduce PINNs-MPF Eqs. 9–10 exactly.

```python
# Execution-verified (numpy 2.5.2): for a 2D circular grain all collocation points
# land within the interface band.
import numpy as np


def interface_mask(phi, tol=1e-3):
    return phi.max(axis=0) < 1.0 - tol          # no phase is numerically pure


def sample_ic_and_collocation(phi, dx, eta, n_bulk=64, chi=40.0, c_f=20, rng=None):
    rng = rng or np.random.default_rng(0)
    ny, nx = phi.shape[1:]
    m = interface_mask(phi)
    iy, ix = np.nonzero(m)
    n_int = int(np.ceil(chi * m.sum() * dx * dx / eta**2)) if len(iy) else 0
    pick = rng.choice(len(iy), size=min(n_int, len(iy)), replace=False) if n_int else np.array([], int)
    by, bx = np.nonzero(~m)
    bpick = rng.choice(len(by), size=min(n_bulk, len(by)), replace=False)
    ic = np.concatenate([np.c_[ix[pick], iy[pick]], np.c_[bx[bpick], by[bpick]]]) * dx + dx / 2
    base = ic[: len(pick)].repeat(c_f, axis=0)                 # N_f = c_f * N_ini(interface)
    r = (eta / 2) * np.sqrt(rng.uniform(size=len(base)))
    th = rng.uniform(0, 2 * np.pi, size=len(base))
    col = (base + np.c_[r * np.cos(th), r * np.sin(th)]) % np.array([nx * dx, ny * dx])
    return ic, col
```

The residual for one phase network uses autodiff Laplacians of *all* phase networks in the same batch. This is the vertical communication.

```python
# ILLUSTRATIVE, NOT EXECUTED: TensorFlow sketch of the pairwise MPF residual, written in
# the Eq. 15/17 convention so that it matches the executed numpy reference (Workflow 6).
# Assumes nets[a] maps (x, y, t) -> phi_a for one batch.
import numpy as np
import tensorflow as tf


def mpf_residual(nets, xyt, mu, sigma, eta, dG):
    with tf.GradientTape(persistent=True) as t2:
        t2.watch(xyt)
        with tf.GradientTape(persistent=True) as t1:
            t1.watch(xyt)
            phis = [net(xyt) for net in nets]
        grads = [t1.gradient(p, xyt) for p in phis]
    lap = [t2.gradient(g[:, 0:1], xyt)[:, 0:1] + t2.gradient(g[:, 1:2], xyt)[:, 1:2] for g in grads]
    n = len(nets)
    I = [lap[a] + (np.pi**2 / eta**2) * phis[a] for a in range(n)]      # Eq. 15
    res = []
    for a in range(n):
        rate = 0.0
        for b in range(n):
            if a != b:
                rate += mu * sigma * (I[a] - I[b])
                # factor 2 so that n = 2 reproduces the two-phase Eq. 17 (v = mu * dG)
                rate += 2.0 * mu * dG[a][b] * (np.pi / eta) * tf.sqrt(tf.nn.relu(phis[a] * phis[b]))
        res.append(grads[a][:, 2:3] - rate / n)   # N = n here; the paper uses locally present phases
    return res   # loss_pde = mean over phases of mean(res_a**2)
```

**Checks:**

- Track the PDE, IC, boundary-coupling, and Σφ losses separately. The paper's loss also includes an optional denoising term.
- Verify the driving-force coefficient against the two-phase limit (v = μΔg) before trusting multiphase runs.
- Using a global N where the paper uses the local phase count changes the dynamics in multi-junction regions. [EXPERT REVIEW NEEDED: impact of global versus local N on triple-junction kinetics]

### Workflow 3: Interface benchmark (driven front and curvature-driven shrinkage)

**Purpose:** establish quantitative accuracy on problems with analytical limits before attempting junctions.

1. **Driven front:** 1D (or a thin 2D strip) with a single interface and constant ΔG. Extract the front position as the integral of φ, fit the velocity, and compare with μΔg *and* with the converged FD reference (Workflow 6).
2. **Periodicity trap:** a periodic strip containing one phase band has *two* fronts. The integral of φ then grows at 2v. In this skill's reference runs this produced an apparent factor of about 1.9; per front, v/(μΔg) was about 0.91–0.94 at η = 6 dx. Use non-periodic boundaries or count fronts.
3. **Curvature shrinkage:** a circular grain with ΔG = 0. Fit R² = A/π against t over the regime where R ≫ η. Compare the slope with −2μσ and with the reference.
4. **Driven shrinkage:** combine both. The paper's Eq. 25 gives the normal velocity v_n = μ(σκ + Δg). For a circular grain the radius obeys dR/dt = −μσ/R ± μΔg, depending on which phase the driving force favours. A grain of radius R* = σ/|Δg| is stationary when Δg favours growth, so pick R0 well away from R* to get a clean benchmark. [EXPERT REVIEW NEEDED: sign of Δg relative to the shrinking phase in the PINNs-MPF benchmark definitions]
5. **Repeat per architecture.** The paper reports MAE/MSE against theory for driven shrinkage improving markedly from 6 × 32 to 6 × 128 hidden layers per network (four networks). Do not assume an architecture transfers between benchmarks.

**Silent failure:** the PDE loss can reach the interval threshold while the front velocity is wrong, for example when collocation does not cover the moving interface after resampling. Always check the velocity, not only the loss.

### Workflow 4: Triple-junction evolution with domain decomposition and pyramidal training

**Inputs:** a four-grain configuration with junctions (the paper uses 64 × 64, six junctions), the batch layout, and thresholds.
**Outputs:** junction angles over time, grain areas, constraint residuals, and per-batch loss histories.

```python
# Execution-verified (numpy 2.5.2): 64 -> [64, 36, 16, 4], consistent with the paper's
# "64 networks initialised, 16 selected" triple-junction setup.
import numpy as np


def pyramid_levels(n_batches_finest: int) -> list[int]:
    """PINNs-MPF Eq. 8: N_coarser = (sqrt(N) - 2)^2."""
    levels = [n_batches_finest]
    while True:
        s = int(round(np.sqrt(levels[-1])))
        if s * s != levels[-1]:
            raise ValueError("finest level must be a perfect square")
        nxt = (s - 2) ** 2
        if nxt < 1:
            return levels
        levels.append(nxt)
```

Procedure:

1. Decompose space into batches with a shared boundary band. Each batch holds one network per phase present there.
2. Train the finest level first. Within each block, start with the network holding the most interfacial points. Warm-start the next level from it.
3. Couple batches through the boundary loss. Watch for seam artifacts: kinks in iso-contours along batch edges, or non-zero boundary loss after the interval converges.
4. Advance time intervals at the threshold. The paper used 6 × 10⁻⁴ for the triple junction and reports Σφ loss around 10⁻¹² and boundary loss in the 10⁻⁵ range at convergence.
5. Measure junction angles. For equal isotropic energies they should approach 120°. The paper reports this qualitatively; quantify it with tangent fits, and report resolution.

**Limitations to state:** thresholds were set by trial; larger network widths failed; sharp (90°) initial corners are hard early in training; seam coupling is an approximation that can bias junction motion near batch edges.

### Workflow 5: Training convergence versus physical accuracy

A falling loss is not evidence of correct interface motion. For every benchmark record:

- **Per-term loss history:** PDE, IC, boundary coupling, Σφ, and (if used) denoising, per batch and per time interval. A dominant PDE loss can hide IC drift. The paper reports PDE loss about 10× the IC loss at convergence for curvature shrinkage.
- **Physical observables against time:** front position or velocity, radius-squared slope, grain areas, junction angles.
- **Constraint residuals:** max |Σφ − 1| and bound violations on a dense evaluation grid, not only at collocation points.
- **Accumulation diagnostics:** error against interval index. The paper flags accumulation across time marching as a risk.
- **Loss scaling:** nondimensionalize with T = η²/(μσ) and length η. Report any λ weights. The paper keeps all λ = 1 by default and notes that only the boundary-coupling and Σφ weights are safely tunable.
- **Precision and runtime:** state float32 or float64, CPU or GPU, wall-clock time per interval, and total time. The published runs used a 32-core CPU workstation; the current repository runs on GPU in float64.
- **Reproducibility:** record seeds, the lock file, thresholds, and the code version (paper-era legacy code versus the maintained package).

### Workflow 6: Independent comparison against a conventional MPF solver

**Inputs:** the same equation, parameters, η, IC, and boundary conditions as the PINN.
**Outputs:** reference fields at the PINN's output times, a convergence record, and comparison metrics.
**Assumption:** the reference solves the same continuum equation. Differences between autodiff and grid Laplacians shift interface profiles; the paper attributes interface-width differences against OpenPhase to this.

```python
# Execution-verified (numpy 2.5.2): an explicit, periodic, 5-point-Laplacian solver of the
# pairwise MPF equation with local N. Circle test at 128^2, R0 = 40, dt = 0.1, mu = sigma = 1:
# R^2-slope ratio to -2*mu*sigma = 0.887 / 0.973 / 0.986 for eta = 4 / 7 / 9 dx;
# unit-sum error ~1e-16 after clip-and-renormalize.
import numpy as np


def lap(f, dx):
    return (np.roll(f, 1, -1) + np.roll(f, -1, -1) + np.roll(f, 1, -2) + np.roll(f, -1, -2) - 4 * f) / dx**2


def mpf_rate(phi, mu, sigma, eta, dx, dG=None):
    n = phi.shape[0]
    L = lap(phi, dx)
    I = L + (np.pi**2 / eta**2) * phi
    active = (phi > 0) | (np.abs(L) > 0)
    N = np.maximum(active.sum(0), 1)                    # locally present phases
    rate = np.zeros_like(phi)
    for a in range(n):
        for b in range(n):
            if a == b:
                continue
            term = mu * sigma * (I[a] - I[b]) / 2.0
            if dG is not None:
                term = term + mu * (np.pi / eta) * np.sqrt(np.clip(phi[a] * phi[b], 0, None)) * dG[a, b]
            rate[a] += np.where(active[a] & active[b], term, 0.0)
    return 2.0 * rate / N


def step(phi, dt, **kw):
    phi = np.clip(phi + dt * mpf_rate(phi, **kw), 0.0, 1.0)
    return phi / phi.sum(0, keepdims=True)
# Explicit stability: keep dt well below dx^2 / (4 mu sigma); dt = 0.1 dx^2/(mu sigma) was stable here.
```

Comparison protocol:

1. Converge the reference in η/dx and dt for the reported observables.
2. Evaluate the PINN on the reference grid at matching times.
3. Score with the metrics in [neural-microstructure-validation](../neural-microstructure-validation/SKILL.md): field error, interface distance, area kinetics, junction count and angles, and admissibility.
4. Report runtime for both on stated hardware, with PINN training time included.

## Common Failure Modes

| Failure | Symptom | Likely Cause | Diagnostic | Fix |
|---------|---------|--------------|------------|-----|
| Loss converged, interface wrong | Interval threshold met but front velocity or R² slope off | Collocation misses moving interface; PDE loss dominated by bulk | Compare velocity/slope with reference per interval | Resample around the predicted interface each L-BFGS stage; tighten threshold |
| Seam artifacts between batches | Kinks or offsets in contours along batch edges | Weak or late boundary-coupling loss | Boundary-loss history; contour overlay on batch grid | Increase coupling weight; widen overlap band; warm-start neighbours |
| Single network fails in 2D | Loss stagnates; unstable fields | Problem too stiff for one coordinate network | Compare one network with decomposed training | Use per-batch/per-phase networks as in the paper |
| Oversized networks do not converge | 6×256 or wider networks stall | Optimization difficulty, not capacity | Architecture sweep (paper Table 5 trend) | Prefer many moderate networks (e.g. 6×128) |
| Triple junction stalls | Loss above threshold for thousands of epochs | No pyramidal warm start; poor initialization | Loss history with and without pyramid | Pyramidal training from the finest level |
| Unit-sum or bound drift | Σφ ≠ 1 between collocation points | Soft Σφ loss only sampled at points | Dense-grid admissibility residuals | Renormalize predictions; or softmax outputs (current repo variant) |
| Under-resolved interface width | Reference and PINN both deviate from sharp-interface law | η too small relative to dx or point density | η/dx sweep on the reference | Increase η/dx; compare PINN to reference at the same η |
| Periodic two-front miscount | Driven velocity about 2× the expected value | Periodic strip contains two interfaces | Count interfaces in the profile | Use non-periodic BCs or divide by the number of fronts |
| Time-marching accumulation | Error grows with interval index | Each interval inherits the previous prediction | Error versus interval index against reference | Stricter thresholds; periodic reference re-anchoring for diagnosis only |
| Paper/repository results mixed | Inconsistent numbers or settings in a report | Two code generations with different constraints and solvers | Record code version per result | Cite paper results and repository results separately |

## Best Practices

- Start every study with the two-phase driven front and the curvature circle, against both the sharp-interface law and a converged reference at the same η.
- Nondimensionalize with length η and time η²/(μσ). Report the dimensionless driving-force number.
- Compare against a reference solving the same equation, not only against theory. Finite-width error in the reference can exceed the PINN's.
- Track every loss term per batch and per interval, together with physical observables.
- Evaluate admissibility on a dense grid, not only at collocation points.
- Use pyramidal warm starts for junction problems. Record the selected networks at each level.
- Record which code generation (published legacy code or maintained package), precision, hardware, thresholds, and seeds produced each result.
- State scope honestly: isotropic, 2D, few grains. Do not extrapolate to polycrystals, 3D, or anisotropic physics without new evidence.

## Integration with Other Skills

- **[phase-field-workflow](../phase-field-workflow/SKILL.md):** Provides the general phase-field formulation, free-energy calibration, and convergence practice behind the MPF equations solved here.
- **[neural-microstructure-validation](../neural-microstructure-validation/SKILL.md):** Defines the model-independent metrics and benchmark protocol used to score PINNs-MPF results against references.
- **[pinn-phase-neural-integrator](../pinn-phase-neural-integrator/SKILL.md):** A distinct, grid-based recurrent approach to the same class of curvature-driven problems; consult it for the side-by-side comparison of formulations.
- **[pinns-scientific-ml](../../scientific-ml/pinns-scientific-ml/SKILL.md):** General PINN loss construction, optimizer staging, and failure analysis that this skill specializes to multiphase interfaces.
- **[fem-multiphysics](../../continuum-modeling/fem-multiphysics/SKILL.md):** Relevant when MPF problems are coupled to mechanics or transport and a mesh-based reference is needed.
- **[reproducibility-workflow](../../scientific-data/reproducibility-workflow/SKILL.md):** Lock files, seeds, thresholds, and checkpoints per interval must be versioned for PINN results to be auditable.
- **[scientific-visualization](../../scientific-data/scientific-visualization/SKILL.md):** Loss-term histories, contour overlays on batch layouts, and observable-versus-time plots.

## Key References

1. Elfetni, S., & Darvishi Kamachali, R. (2025). *PINNs-MPF: A Physics-Informed Neural Network framework for Multi-Phase-Field simulation of interface dynamics.* Engineering Analysis with Boundary Elements, 176, 106200. https://doi.org/10.1016/j.enganabound.2025.106200
2. Elfetni, S., & Darvishi Kamachali, R. (2025). *PINNs-MPF: An Efficient Physics-Informed Machine Learning-based Solver for Multi-Phase-Field Simulations using Tensorflow.* Software Impacts, 24, 100753. https://doi.org/10.1016/j.simpa.2025.100753
3. Elfetni, S. *PINNs-MPF source code repository.* GitHub. https://github.com/SFETNI/PINNs_MPF--a-Physics-Informed-Neural-Network-for-Multi-Phase-Field-problems
4. Steinbach, I., & Pezzolla, F. (1999). *A generalized field method for multiphase transformations using interface fields.* Physica D: Nonlinear Phenomena, 134, 385-393. https://doi.org/10.1016/S0167-2789(99)00129-3
5. Steinbach, I. (2009). *Phase-field models in materials science.* Modelling and Simulation in Materials Science and Engineering, 17, 073001. https://doi.org/10.1088/0965-0393/17/7/073001
6. Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). *Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations.* Journal of Computational Physics, 378, 686-707. https://doi.org/10.1016/j.jcp.2018.10.045
