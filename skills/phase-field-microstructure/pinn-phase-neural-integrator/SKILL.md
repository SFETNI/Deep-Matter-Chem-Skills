---
name: pinn-phase-neural-integrator
domain: phase-field-microstructure
maturity: draft
tools:
  - PINN-Phase
  - PyTorch
  - numpy
  - scipy
tested_versions:
  # Documented, not execution-verified here: the PINN-Phase repository (package
  # pinn-phase 0.1.0, Python >=3.11,<3.12) pins torch 2.6.0, numpy 2.2.6, scipy 1.16.2
  # in CI. No PINN-Phase checkpoint or PyTorch code was run in this repository.
  # Execution-verified: the numpy operator, admissibility map, residual and
  # equivariance tests in Workflows 1-3 ran with these versions (Python 3.14).
  numpy: "2.5.2"
  scipy: "1.18.0"
requires_gpu: true
estimated_runtime: hours-to-days
---

# PINN-Phase Neural Time Integration

## Description

This skill covers PINN-Phase-style neural time integration of curvature-driven multiphase-field evolution. A recurrent field model advances a discrete phase-field state autonomously from an initial condition. It is trained only on the physical residual of the discrete operator evaluated on its own predicted trajectory, with hard admissibility (bounds and unit sum) enforced in the state update.

The skill covers:

- the discrete operator and the bounded increment map;
- label-free trajectory training with truncated BPTT;
- permutation-equivariant grain handling;
- long-horizon rollout and error accumulation;
- extinction and topology changes;
- the difference between horizon extrapolation and initial-condition generalization;
- frozen-checkpoint prospective evaluation in 2D and 3D;
- fair runtime reporting and checkpoint provenance.

Invoke this skill when building, reproducing, or auditing such an integrator. It is based on a 2026 arXiv preprint.

## Domain Context

**Evidence status.** PINN-Phase (Elfetni, Thinius, Seeberger & Tyrikos-Ergas, arXiv:2608.29413, v2 October 2026) is a preprint and has not been peer reviewed. All numbers below are attributed to it and are scoped to its benchmark families. The repository README's citation block omits the second author (S. Thinius); cite the arXiv author list.

**Equation.** The preprint integrates a projected multi-channel Allen–Cahn operator. This is the equal-parameter, isotropic member of the constrained multiphase Allen–Cahn family:

  R(φ) = μσ · P_N[ ∇²φ − W′(φ)/η² ],  W(φ) = φ²(1 − φ)² applied componentwise.

- P_N subtracts the phase mean, so the rate sums to zero.
- The Laplacian is the second-order central-difference stencil with periodic indexing, so validation is against direct integration of *the same discrete operator*.
- This is a different discrete model from the Steinbach-type obstacle-potential MPF used by PINNs-MPF. The two are not interchangeable references.
- The preprint's scalar benchmarks use a separate double-obstacle-type Allen–Cahn form with a bound-aware rate map, which is not the N = 1 limit of the multiphase operator.

**Neural formulation:**

- **State map.** The network maps the current *predicted* state to a bounded increment, Δφ_t = Δφ_max · P_N[tanh(z_t)], then applies φ_{t+1} = Π(φ_t + Δφ_t).
- **Admissibility map.** Π clips to [0, 1], renormalizes to unit sum, and falls back to 1/N if the sum collapses. It is explicitly *not* a Euclidean simplex projection.
- **Response.** z combines a site-local MLP and a ConvLSTM branch through a trainable gate.
- **Training target.** Training compares Δφ_t with q_t = Δt_m · R(φ_t) evaluated on the model's own field, over horizon H, with truncated backpropagation through time (TBPTT).
- **Label-free training.** On the explicit-MPF path, no reference state after t0 enters the loss, early stopping, or checkpoint selection. The repository notes one exception: the scalar 64³ checkpoint was selected using post-t0 reference terms.
- **Permutation equivariance.** An equivariant variant (shared channel maps, Deep-Sets-style aggregation, circular padding, no phase-index or absolute-coordinate inputs) is exact under phase relabeling and integer periodic translation. It was used for the N = 25 and N = 64 2D work and the prospective 2D cohort. The earlier hybrid (junction, 3D cubes, native 128³) is not equivariant.

**What the preprint reports (scoped, as stated there):**

- **2D, N = 25:**
  - Trained on one initial condition with H = 4,096 and rolled to 12,000 steps.
  - Label disagreement was about 1% versus about 27–51% for the persistence baseline.
  - Survivor sets were exact, but extinctions were premature (median −53 and −85 steps), so the pre-registered timing tolerance was not met.
- **2D prospective cohort** (8 unseen plus 2 stress initial conditions, frozen checkpoint, scored once): 7/8 unseen and 2/2 stress cases passed every criterion. The Wilson 95% interval for 7/8 is about 0.53–0.98.
- **3D:**
  - Designed-extinction cubes (N = 8 at 64³, N = 16 at 96³) reproduced the survivor sets.
  - The native 128³, N = 8 model met all six strict criteria on only 2 of 6 prospectively fixed unseen cases. The repository states that the cohort rule was not met.
- **Runtime:** the preprint reports runtime as indicative and explicitly does not claim a matched end-to-end solver speed-up.

**Scope boundary:**

- **Evaluated:** isotropic, equal-parameter, curvature-driven, periodic multiphase Allen–Cahn evolution, at specific phase counts and resolutions.
- **Not evaluated:**
  - anisotropic energies, heterogeneous mobilities, and bulk driving forces;
  - conserved (Cahn–Hilliard), solute, or elastic coupling;
  - precipitation, solidification, and industrial microstructures;
  - transfer across phase count, resolution, or physics.
- Claims outside this envelope need independent evidence. [EXPERT REVIEW NEEDED: whether the label-free trajectory-residual strategy remains stable for stiffer or driven operators is untested]

## When to Use This Skill

- Use when building an autoregressive neural integrator for a discrete phase-field operator that is known exactly and can be evaluated on predicted states.
- Use when training must be label-free (no reference trajectories) and admissibility must hold by construction during long rollouts.
- Use when grain identities are exchangeable and permutation equivariance should be built into the architecture.
- Use when designing prospective, frozen-checkpoint evaluations of rollout stability, extinction events, and initial-condition generalization.
- Use when auditing a neural-integrator claim to separate horizon extrapolation, development evidence, and prospective generalization.

## When Not to Use This Skill

- **When the governing operator is unknown or must be learned from data.** The training signal here *is* the known discrete operator.
- **For physics outside the evaluated family.** This includes anisotropy, driving forces, conserved dynamics, solute, elasticity, precipitation, solidification, and real alloy processing. Nothing in the preprint supports transfer there.
- **When a fast classical solver for the same operator suffices.** An explicit stencil solver is simple and exact by definition, and no matched speed-up has been established.
- **When a single rollout is all that is needed.** Training cost (hours on data-centre GPUs in the preprint) amortizes only over many rollouts.
- **When results must transfer across N, resolution, or boundary conditions without retraining.** This is not demonstrated.

## Core Concepts

- **Discrete operator R(φ):** A projected multi-channel Allen–Cahn rate on a periodic finite-difference grid. The neural model approximates the *integrator* of this operator, not an unknown PDE.
- **Bounded zero-sum increment:** Δφ = Δφ_max · P_N[tanh(z)] sums to zero exactly. Each component is bounded by 2(N − 1)/N · Δφ_max rather than by Δφ_max, because of the projection. In this skill's tests, max |Δφ| was 0.111 for Δφ_max = 0.0667 at N = 6.
- **Admissibility map Π:** Clip, renormalize, and fall back to 1/N. It guarantees bounds and unit sum after every step. The fallback can manufacture a uniform "all-phases" cell, so count fallback events.
- **Self-consistent (label-free) residual:** A loss comparing the predicted increment with Δt_m · R(φ_t) evaluated on the predicted state. It penalizes local inconsistency with the operator, not global trajectory error. Small per-step residuals can still accumulate into wrong event timing.
- **TBPTT window and horizon H:** Gradients flow through a window of steps; the training horizon H is the total rollout length used for training. Behaviour beyond H is extrapolation in time.
- **Persistence baseline:** The t0 state held fixed. Gain over persistence is the meaningful skill measure.
- **Permutation equivariance:** Relabeling the input phases relabels the output identically. This is required for exchangeable grains and testable as a property (Workflow 3).
- **Evidence tiers:** Training initial condition; past-horizon rollout on it; held-out-but-inspected (development); prospective unseen with frozen checkpoint; stress cases. Never pooled.
- **Checkpoint provenance:** Weight hash, lineage (parent checkpoint, continuation steps), and which cohort was seen when. The preprint distributes replay weights (`.npz`) with lineage JSON. Training configurations and parent checkpoints are not distributed, so training reproduction is not possible from the archive.

## Key Workflows

### Workflow 1: Implement and verify the discrete operator and admissibility maps

**Inputs:** grid, η (in grid spacings), μσ, time step Δt, number of phases N.
**Outputs:** a verified reference stepper, which is also the training target and the evaluation reference.
**Checks:** unit sum at machine precision, bounds respected, and calibrated effective kinetics.

```python
# Execution-verified (numpy 2.5.2). 2D two-grain circle, 128^2, eta = 3 dx, mu*sigma = 1,
# dt = 0.04: fitted area slope -6.299 -> effective M*sigma = 1.0025; unit-sum error
# ~1e-16 after 20,000 steps.
import numpy as np


def lap_periodic(phi, dx):
    out = -2 * (phi.ndim - 1) * phi
    for ax in range(1, phi.ndim):
        out = out + np.roll(phi, 1, ax) + np.roll(phi, -1, ax)
    return out / dx**2


def project_zero_sum(v):
    return v - v.mean(axis=0, keepdims=True)


def mac_rate(phi, mu_sigma, eta, dx):
    """mu*sigma * P_N[lap(phi) - W'(phi)/eta^2], W = phi^2 (1-phi)^2 componentwise."""
    dW = 2 * phi * (1 - phi) * (1 - 2 * phi)
    return mu_sigma * project_zero_sum(lap_periodic(phi, dx) - dW / eta**2)


def admissibility_map(phi, eps=1e-12):
    """Clip to [0,1], renormalize, fall back to 1/N. Not a Euclidean simplex projection."""
    c = np.clip(phi, 0.0, 1.0)
    s = c.sum(axis=0, keepdims=True)
    return np.where(s > eps, c / np.maximum(s, eps), 1.0 / phi.shape[0])


def bounded_increment(z, dphi_max):
    """Zero-sum increment; per-component bound is 2(N-1)/N * dphi_max, not dphi_max."""
    return dphi_max * project_zero_sum(np.tanh(z))


def reference_step(phi, dt, mu_sigma, eta, dx):
    return admissibility_map(phi + dt * mac_rate(phi, mu_sigma, eta, dx))
```

**Calibration:**

- Fit the area decay of an isolated circular grain (2D: dA/dt ≈ −2π(Mσ)_eff) and of a sphere in 3D.
- Report (Mσ)_eff for the chosen η/dx and Δt. Analytical laws should be used with this calibrated constant, never with the nominal μσ.
- Explicit stability requires Δt well below dx²/(2d · μσ) (d = dimension), with the reaction stiffness ~1/η² also considered. Verify stability empirically for the chosen parameters.

### Workflow 2: Label-free trajectory training

**Inputs:** training initial condition(s), the operator from Workflow 1, horizon H, TBPTT window, and Δφ_max.
**Assumption:** the operator evaluated on the model's predicted states is a sufficient training signal.
**Output:** a checkpoint whose selection used no post-t0 reference data, verifiable from the logs.

```python
# Execution-verified (numpy 2.5.2) as a loss function: zero for an exact stepper, 0.0025
# for a 5%-biased stepper on the same trajectory.
import numpy as np


def self_consistent_residual(phi_traj, dphi_traj, dt_m, mu_sigma, eta, dx,
                             s_abs=0.05, eps_r=1e-8):
    """Relative + absolute squared error between predicted increments and the operator
    evaluated on the predicted states; first step weighted x2. This follows the form
    described in the PINN-Phase preprint; check the paper for its exact normalization."""
    total, weight = 0.0, 0.0
    for t, (phi_t, p_t) in enumerate(zip(phi_traj, dphi_traj)):
        q_t = dt_m * mac_rate(phi_t, mu_sigma, eta, dx)
        err = np.sum((p_t - q_t) ** 2)
        w = 2.0 if t == 0 else 1.0
        total += w * (err / (np.sum(q_t ** 2) + eps_r) + s_abs * err / p_t.size)
        weight += w
    return float(total / weight)
```

```python
# ILLUSTRATIVE, NOT EXECUTED: PyTorch training-loop skeleton with TBPTT.
# model(phi, hidden) -> (z, hidden); rate(phi) is a torch port of mac_rate.
import torch


def train_epoch(model, phi0, rate, horizon, tbptt, dt_m, dphi_max, opt):
    phi, hidden = phi0.clone(), None
    for start in range(0, horizon, tbptt):
        loss = 0.0
        for k in range(tbptt):
            z, hidden = model(phi, hidden)
            dphi = dphi_max * (torch.tanh(z) - torch.tanh(z).mean(0, keepdim=True))
            q = dt_m * rate(phi)                      # operator on the PREDICTED state
            w = 2.0 if start == 0 and k == 0 else 1.0
            loss = loss + w * ((dphi - q) ** 2).sum() / ((q ** 2).sum() + 1e-8)
            phi = torch.clamp(phi + dphi, 0, 1)
            phi = phi / phi.sum(0, keepdim=True).clamp_min(1e-12)
        opt.zero_grad(); loss.backward(); opt.step()
        phi = phi.detach()                            # truncate BPTT between windows
        hidden = tuple(h.detach() for h in hidden) if hidden is not None else None
```

Training diagnostics:

- Log the residual per step index within the horizon. Rising late-horizon residuals predict rollout failure.
- Log the gate value between the local and ConvLSTM branches, the magnitude of z, and the count of admissibility-map fallback cells.
- Horizon curricula (e.g. 512 → 2,048 steps, as used for one 3D case in the preprint) can stabilize training. Report them, because they change what "training horizon" means.
- The preprint's junction study found a non-monotone relation between horizon length and long-time error. Do not assume a longer H is always better without testing.

### Workflow 3: Verify equivariance and admissibility as properties

Run these property tests on every checkpoint before any benchmark. They are model-agnostic.

```python
# Execution-verified (numpy 2.5.2): the reference step gives permutation error ~1e-16 and
# translation error 0; a step with a phase-index-dependent bias gives a permutation error
# of 3.9e-3 and is detected.
import numpy as np


def check_equivariance(step_fn, phi, rng, shift=(3, -5)):
    perm = rng.permutation(phi.shape[0])
    axes = tuple(range(1, phi.ndim))
    return {
        "perm_max_abs": float(np.abs(step_fn(phi[perm]) - step_fn(phi)[perm]).max()),
        "trans_max_abs": float(np.abs(step_fn(np.roll(phi, shift, axis=axes))
                                      - np.roll(step_fn(phi), shift, axis=axes)).max()),
    }
```

- A model claimed to be equivariant should give errors at float precision. Larger values indicate phase-index inputs, coordinate features, non-circular padding, or order-dependent aggregation.
- A non-equivariant model is still usable, but labels must then be matched before scoring, and generalization to relabeled microstructures is not guaranteed.
- Translation equivariance holds only for integer shifts on periodic grids.

### Workflow 4: Autonomous rollout, error accumulation, and topology events

**Inputs:** a frozen checkpoint (hash recorded), initial conditions, and the reference stepper.
**Outputs:** time series of label disagreement, gain over persistence, survivor sets, extinction times and order, grain areas, and constraint residuals.

Procedure:

1. Roll the model autonomously from t0. Do not inject reference states; that would be teacher forcing, not autonomous rollout.
2. Roll the reference stepper from the same t0 with the same Δt, saving at the same cadence. Save more frequently near expected extinctions, because the preprint's timing resolution was limited to the save interval (50–200 steps).
3. Score with [neural-microstructure-validation](../neural-microstructure-validation/SKILL.md): matched label disagreement, persistence baseline, survivor F1, signed extinction errors, spurious grains, and admissibility.
4. Plot error against step on a log time axis. Classify error growth as linear, super-linear (instability), or event-driven (jumps at extinctions).
5. Report beyond-horizon results separately from within-horizon results. Rolling the training initial condition from 4,096 to 12,000 steps tests temporal extrapolation, not generalization.

The characteristic failure is event timing. A model can reproduce the survivor set while extinctions are systematically early or late, which mean field errors will not show.

### Workflow 5: Prospective initial-condition generalization (2D and 3D)

1. Freeze the checkpoint (hash), metrics, thresholds, evaluation steps, and case list in a benchmark specification *before* generating or viewing the held-out reference trajectories.
2. Generate unseen initial conditions from the same family as training (same N, resolution, grain-size range), plus explicitly labelled stress cases (dense or sparse grains).
3. Score each case once. Report k/n with a Wilson interval per cohort. Keep development, prospective, and stress cohorts separate.
4. If a model is continued or adapted after scoring (e.g. longer-horizon continuation, case-specific physics-only adaptation), label the resulting numbers as sensitivity or development evidence. The preprint does this for its N = 64 continuation and its case-adaptation study.
5. In 3D, report grid size, N, η, and memory. Some 3D variants in the preprint lose phase purity over long evolution, so track the fraction of cells without a dominant phase.

### Workflow 6: Runtime measurement and fair reference-solver comparison

- Measure rollout wall-clock time with warm-up excluded, at fixed precision, on named hardware (including vGPU slices). Report peak memory.
- Run the reference stencil solver for the same operator, grid, and number of steps on comparable hardware (GPU stencil versus GPU network). Report both.
- Include training cost as measured GPU-hours. A projected training time must be labelled as projected; the preprint labels one such figure as a projection.
- State the break-even number of rollouts. Without a matched comparison, report "indicative runtime", as the preprint does, and do not report a speed-up.

### Workflow 7: Comparing PINN-Phase with PINNs-MPF

| Aspect | PINNs-MPF (EABE 2025, peer reviewed) | PINN-Phase (arXiv 2026, preprint) |
|---|---|---|
| Discrete model | Steinbach-type obstacle-potential pairwise MPF | Projected multi-channel Allen–Cahn, double-well W = φ²(1−φ)² |
| Network role | Coordinate networks (x, y, t) → φ per batch and per phase | One shared recurrent field model mapping state → increment |
| Temporal advancement | Discrete time intervals trained to a loss threshold; prediction becomes next IC | Autonomous step-by-step rollout trained over horizon H with TBPTT |
| Constraint enforcement | Sigmoid bounds, soft Σφ loss, dynamic renormalization (paper); softmax in current repo | Hard: zero-sum bounded increment plus clip–renormalize map every step |
| Coupling | Spatial batches with boundary loss; master/worker; pyramidal training | Single field model; optional graph conditioner (2D, non-equivariant variant) |
| Training signal | PDE residual by autodiff at collocation points, plus IC/BC losses | Operator residual on the model's own predicted trajectory; no post-t0 labels |
| Validation | Sharp-interface theory and OpenPhase (paper); repo FD solver | Direct integration of the same discrete operator; prospective frozen cohorts |
| Scale demonstrated | 1D–2D, up to a four-grain junction on 64²-scale domains | 2D up to N = 64; 3D up to 128³ with N = 8 |
| Transfer assumption | Within benchmark families (e.g. across Δg values) | Unseen initial conditions in the same family; not across N, resolution, or physics |
| Stated main limitation | Problem-specific thresholds; slower than classical solver | Extinction timing; only 2/6 prospective strict passes at native 128³ |

The preprint describes the two as complementary scaling strategies and gives no numerical ranking. Do not rank them using numbers from different operators and benchmarks.

## Common Failure Modes

| Failure | Symptom | Likely Cause | Diagnostic | Fix |
|---------|---------|--------------|------------|-----|
| Premature or late extinctions | Survivor set correct but event times systematically shifted | Per-step residual small but biased near vanishing grains | Signed extinction errors per grain; dense saving near events | Train on more extinction events; longer horizon; report timing as a limitation |
| Horizon extrapolation sold as generalization | Strong results only on the training IC beyond H | Evidence tiers conflated | Audit which IC each number comes from | Report beyond-horizon and unseen-IC results separately |
| Development cases counted as prospective | Pass rate overstated | Held-out case inspected during design | Specification timestamps and case history | Re-label; collect new prospective cases |
| Hidden equivariance break | Different results after relabeling phases | Phase-index inputs, coordinate features, non-circular padding | Permutation/translation property test | Remove index features; shared channel maps; circular padding |
| Increment bound misunderstood | Per-step change exceeds Δφ_max | Zero-sum projection after tanh | Max |Δφ| per step | Use 2(N−1)/N · Δφ_max as the bound or rescale |
| Fallback-cell artifacts | Patches of uniform 1/N phase fractions | Admissibility map collapses clipped sums | Count fallback cells per step | Smaller increments; investigate local instability |
| Phase-purity loss in long 3D rollouts | Growing fraction of cells without dominant phase | Accumulated diffuse smearing | Fraction of cells with max φ < threshold | Stricter map, retraining with longer horizon; state as limitation |
| Reference-label leakage | "Label-free" claim but selection used reference trajectories | Checkpoint chosen by post-t0 reference error | Audit selection logs | Select checkpoints by residual only, or disclose |
| Unmatched speed-up | Large acceleration claimed | Different hardware/precision; training excluded | Hardware and cost audit | Matched stencil baseline; report indicative runtime |
| Transfer beyond tested family | Failure on anisotropic, driven, or different-N problems | No evidence outside the evaluated envelope | Explicit envelope check before deployment | Retrain and re-validate per family |

## Best Practices

- Treat the discrete operator as the ground truth of the study. Verify it, calibrate its effective kinetics, and use it both as training signal and reference.
- Enforce admissibility in the state update and still measure it every step. Count fallback events.
- Property-test equivariance and translation invariance for every checkpoint before benchmarking.
- Keep training label-free if claiming it, and keep the checkpoint-selection logs that prove it.
- Freeze checkpoint hash, cases, metrics, and thresholds before prospective scoring. Score once and report per cohort with Wilson intervals.
- Separate training-IC, beyond-horizon, development, prospective, and stress evidence in every table.
- Save densely near topological events so extinction timing is not limited by the output cadence.
- Report runtime with hardware, precision, warm-up, and training cost. Avoid "speed-up" without a matched baseline.
- State the evaluated envelope (operator, N, resolution, dimensionality, isotropy) wherever results are cited.

## Integration with Other Skills

- **[neural-microstructure-validation](../neural-microstructure-validation/SKILL.md):** Supplies the grain matching, event metrics, persistence baseline, cohort statistics, and cost-accounting protocol used to evaluate rollouts.
- **[pinns-mpf-workflow](../pinns-mpf-workflow/SKILL.md):** The coordinate-network alternative for multiphase interface dynamics; see Workflow 7 for the formulation comparison.
- **[phase-field-workflow](../phase-field-workflow/SKILL.md):** Background on phase-field operators, interface resolution, and convergence of reference simulations.
- **[pinns-scientific-ml](../../scientific-ml/pinns-scientific-ml/SKILL.md):** General physics-informed training practice (loss balancing, optimizer staging) underlying the residual-trained integrator.
- **[surrogate-validation](../../surrogate-active-learning/surrogate-validation/SKILL.md):** Leakage-aware cohort design and uncertainty reporting for learned surrogates.
- **[reproducibility-workflow](../../scientific-data/reproducibility-workflow/SKILL.md):** Checkpoint hashes, lineage records, benchmark specifications, and environment pins.

## Key References

1. Elfetni, S., Thinius, S., Seeberger, P. H., & Tyrikos-Ergas, T. (2026). *PINN-Phase: A physics-informed neural network for curvature-driven multiphase-field evolution.* arXiv:2608.29413 (preprint, v2). https://doi.org/10.48550/arXiv.2608.29413
2. Elfetni, S., Thinius, S., Seeberger, P. H., & Tyrikos-Ergas, T. *PINN-Phase source code and replay checkpoints.* GitHub. https://github.com/SFETNI/PINN-Phase
3. Elfetni, S., & Darvishi Kamachali, R. (2025). *PINNs-MPF: A Physics-Informed Neural Network framework for Multi-Phase-Field simulation of interface dynamics.* Engineering Analysis with Boundary Elements, 176, 106200. https://doi.org/10.1016/j.enganabound.2025.106200
4. Shi, X., Chen, Z., Wang, H., Yeung, D.-Y., Wong, W.-K., & Woo, W.-C. (2015). *Convolutional LSTM network: A machine learning approach for precipitation nowcasting.* Advances in Neural Information Processing Systems 28. https://arxiv.org/abs/1506.04214
5. Zaheer, M., Kottur, S., Ravanbakhsh, S., Poczos, B., Salakhutdinov, R., & Smola, A. (2017). *Deep Sets.* Advances in Neural Information Processing Systems 30. https://arxiv.org/abs/1703.06114
6. Mullins, W. W. (1956). *Two-dimensional motion of idealized grain boundaries.* Journal of Applied Physics, 27, 900-904. https://doi.org/10.1063/1.1722511
