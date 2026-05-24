---
name: bayesian-optimization
domain: surrogate-active-learning
maturity: draft
tools:
  - BoTorch
  - Ax
  - GPyTorch
  - scikit-learn
  - numpy
  - pandas
  - matplotlib
tested_versions:
  BoTorch: "0.10.0"
  Ax: "0.3.7"
  GPyTorch: "1.11"
  scikit-learn: "1.4.0"
  numpy: "1.26.4"
requires_gpu: false
estimated_runtime: minutes-to-hours
---

# Bayesian Optimization for Materials and Chemistry

## Description

This skill covers Bayesian optimization (BO) for materials discovery and chemistry design: surrogate model construction, acquisition function selection and optimization, batch and constrained BO, multi-objective Pareto-front search, multi-fidelity optimization, and integration with high-throughput DFT, experimental workflows, and active learning loops. It uses BoTorch and Ax as the primary frameworks. Invoke this skill when sequentially selecting experiments or calculations to optimize an expensive objective — band gap, formation energy, synthesis yield, reaction rate, or any property accessible via DFT, simulation, or laboratory measurement — especially when the budget is small (tens to low hundreds of evaluations) and uncertainty-guided exploration matters.

## Domain Context

Bayesian optimization is a sequential decision-making strategy for optimizing an expensive black-box function f(x) using as few evaluations as possible. It maintains a probabilistic surrogate model p(f | data) — typically a Gaussian process — and selects the next evaluation point by maximizing an acquisition function α(x) that balances exploration (high uncertainty) and exploitation (high predicted value). After each evaluation, the surrogate is updated and the process repeats.

**Why BO for materials:** Materials property spaces are high-dimensional, discontinuous, combinatorially large, and expensive to evaluate. A DFT calculation costs minutes to days; a synthesis-and-characterization cycle costs weeks. Random or grid search wastes budget on obviously poor candidates. BO uses all past evaluations to build a model of the objective landscape and targets the most promising regions. In practice, BO routinely outperforms random search on materials benchmarks at budgets of 20–200 evaluations, though the advantage shrinks as budget grows. [EXPERT REVIEW NEEDED: the relative advantage of BO over random search is problem-specific and sometimes smaller than reported in clean benchmarks; noisy experimental settings often reduce the margin]

**The surrogate–acquisition loop:**
1. Fit a surrogate model to all evaluated (x, f(x)) pairs.
2. Optimize the acquisition function over the search space to find x_next.
3. Evaluate f(x_next) (DFT, experiment, simulation).
4. Add (x_next, f(x_next)) to the dataset. Go to 1.

The surrogate provides calibrated uncertainty; the acquisition function converts predicted mean and uncertainty into a score for each candidate; the optimizer (L-BFGS-B for continuous, enumeration for discrete) finds the highest-scoring candidate. The bottleneck is usually step 3 (the expensive evaluation), not steps 1–2.

**Exploration vs. exploitation:** Acquisition functions parameterize the trade-off. Expected Improvement (EI) tends toward exploitation near the current best; Upper Confidence Bound (UCB) adds κ × σ(x) as an explicit exploration bonus; Thompson Sampling (TS) explores by sampling from the posterior and optimizing the sample. No single strategy dominates across all problem types. [EXPERT REVIEW NEEDED: the optimal exploration-exploitation balance depends on the noise level, budget, and objective landscape topology; there is no universally best acquisition function]

**Search space design:** The search space is the most consequential design choice. A too-narrow space excludes the optimum; a too-wide space dilutes budget in irrelevant regions; a poorly featurized space makes the surrogate useless. For materials, search spaces include:
- *Continuous:* alloy compositions (x in [0,1]^N with Σx_i=1), synthesis temperatures, dopant concentrations, DFT parameters.
- *Discrete/Categorical:* element choices, crystal structure prototypes, functional groups, ligands.
- *Structural:* crystal geometries described by unit cell parameters + Wyckoff positions; requires a representation that maps structure → fixed-length descriptor before the GP.
- *Mixed:* categorical element selection + continuous composition + discrete stoichiometry. BoTorch handles mixed spaces via `ParameterConstraint` and one-hot encoding of categoricals.

**Regret:** Cumulative regret measures total suboptimality over the budget. Simple regret is the gap between the best found and the true optimum at the final iteration. These are meaningful only relative to a baseline (random search, grid search, or expert heuristic). Always report a baseline alongside BO results.

**Stopping criteria:** BO should stop when: (a) the acquisition value of the best unexplored candidate is below a meaningful threshold, (b) the surrogate posterior has converged (low uncertainty everywhere in the feasible region), or (c) the budget is exhausted. In practice, budget exhaustion is the usual stopping condition. Do not stop early based on subjective "convergence" without a quantitative criterion.

## When to Use This Skill

- Optimizing a materials property with a small evaluation budget (< ~500 evaluations) where each evaluation is expensive (DFT, experiment, or simulation).
- Sequentially selecting the next structure, composition, or synthesis condition to evaluate — especially when uncertainty guidance matters.
- Multi-objective optimization where multiple properties must be simultaneously optimized and a Pareto front is the goal.
- Multi-fidelity optimization where cheap low-quality evaluations (screening DFT, rapid experiment) can inform where to spend high-fidelity budget.
- Constrained optimization where the search space has physical, chemical, or experimental feasibility constraints.
- Benchmarking BO against random or heuristic search on a retrospective materials dataset.

## When Not to Use This Skill

- **Very large budgets (> ~1000 evaluations):** BO overhead per step grows; simpler methods (neural network surrogates, evolutionary algorithms, grid search) become competitive or superior.
- **When the objective is cheap to evaluate:** If a single evaluation costs < 1 second, random or grid search with thousands of samples outperforms BO on most problems.
- **When the surrogate is fundamentally mis-specified:** A GP with the wrong kernel or descriptor will give overconfident or uninformative uncertainty. BO with a bad surrogate can be worse than random search.
- **When the search space is not well-defined:** BO requires an explicit parameterization of the search space before starting. If candidate generation is the bottleneck, address that first.
- **When constraints are so tight that most of the space is infeasible:** Constrained BO degrades when the feasible fraction is < ~1%. Use constraint-aware initialization (rejection sampling, feasibility surrogates) or problem reformulation.
- **When the objective is deterministic and smooth and the space is low-dimensional (< 5):** Classic gradient-based optimization or quasi-Newton methods may be faster and more reliable.

## Core Concepts

### Acquisition Functions

**Expected Improvement (EI):**
```
EI(x) = E[max(f(x) - f*, 0)] = (μ(x) - f*) Φ(Z) + σ(x) φ(Z)
Z = (μ(x) - f*) / σ(x)
```
where f* is the current best observed value, μ(x) and σ(x) are the GP posterior mean and standard deviation, and Φ, φ are the standard normal CDF and PDF. EI is zero for x where μ(x) < f* and σ(x) ≈ 0. It naturally balances exploration and exploitation without a tuning parameter.

**Probability of Improvement (PI):**
```
PI(x) = Φ((μ(x) - f* - ξ) / σ(x))
```
where ξ > 0 is a jitter that penalizes points too close to the current best. Simpler than EI but tends toward exploitation; ξ must be tuned.

**Upper Confidence Bound (UCB):**
```
UCB(x) = μ(x) + κ σ(x)
```
κ controls exploration. Large κ = explore; small κ = exploit. Principled choices of κ based on theoretical regret bounds exist but rarely match the best empirical value. [EXPERT REVIEW NEEDED: the theoretically optimal κ schedule is rarely used in practice; κ is typically treated as a hyperparameter tuned by cross-validation or prior experience]

**Thompson Sampling (TS):** Draw a sample function from the GP posterior and maximize it. Naturally explores the posterior landscape. Scales to batch settings without q-EI Monte Carlo overhead. Tends to be robust to mis-specified exploration parameters.

**Entropy Search (ES) / Max-value Entropy Search (MES):** Select x to maximally reduce uncertainty about the location or value of the optimum. More principled than EI in high-noise settings; computationally heavier. MES is often the best choice for noisy objectives. [EXPERT REVIEW NEEDED]

**Knowledge Gradient (KG):** Select x to maximize the expected improvement in the optimal prediction after observing f(x). Accounts for learning value, not just direct improvement. Expensive to compute but well-suited to multi-fidelity and look-ahead settings.

### Batch Bayesian Optimization

When evaluations can be parallelized (multiple DFT jobs simultaneously, multiple synthesis runs in parallel), select a batch of q points jointly. The joint acquisition is:

**q-EI (Monte Carlo EI):**
```
qEI(X_batch) = E[max_{i=1..q} max(f(x_i) - f*, 0)]
```
Optimized via stochastic gradient descent using the reparameterization trick (BoTorch's `qExpectedImprovement`). The key requirement is that points in the batch are jointly optimal, not greedily selected.

**Greedy batch with hallucination:** Select x_1 by EI; add it to the training data with its predicted mean as a "hallucinated" label; refit the GP; select x_2 by EI; repeat. Simple but biased toward exploitation. Useful when q-EI optimization is too slow.

**GIBBON (General Information-Based Batch Bayesian Optimization):** Information-theoretic batch selection that penalizes redundant points without retraining. Practical for large batches (q > 10).

### Multi-Objective Bayesian Optimization

When optimizing multiple objectives simultaneously (e.g., maximizing band gap and minimizing formation energy), the result is a Pareto front — the set of non-dominated solutions where no objective can be improved without worsening another.

**Hypervolume improvement:** The dominated hypervolume (HV) measures the volume of objective space dominated by the Pareto front relative to a reference point. HV is the primary quality metric for multi-objective BO. The acquisition function `qExpectedHypervolumeImprovement` (qEHVI) in BoTorch selects batches that maximize expected HV gain.

**Reference point selection:** The HV reference point must lie below all feasible objective values. A poor reference point makes HV uninformative. Common choices: 10% worse than the worst observed value in each objective, or the nadir point of the current Pareto front. [EXPERT REVIEW NEEDED]

**Pareto dominance:** Point a dominates point b if a is at least as good as b in all objectives and strictly better in at least one. All non-dominated points form the Pareto front. Report the full Pareto front, not a single "best" point.

### Multi-Fidelity Optimization

Different levels of fidelity correspond to different approximation qualities: screening DFT (cheap, noisy) vs. converged DFT (expensive, accurate); rapid experiment vs. full characterization. Multi-fidelity BO allocates budget across fidelities to maximize information gain per unit cost.

**Linear model of coregionalization (LMC) / MTGP:** Multi-task GP that models correlation across fidelities. Learns how well cheap evaluations predict expensive ones. BoTorch's `SingleTaskMultiFidelityGP` implements this with a cost-aware acquisition (`qMultiFidelityMaxValueEntropySearch`).

**Cost-aware acquisition:** `UtilityEI = EI(x, fidelity) / cost(fidelity)`. Prefer cheap evaluations at high-uncertainty points; expensive evaluations near the apparent optimum. The cost ratio between fidelities must be estimated or measured.

### Constrained Bayesian Optimization

Constraints are of two types:
- *Known constraints:* Physical (composition sums to 1, concentration ≥ 0), crystallographic (Wyckoff position parity), or experimental (temperature ≤ 1200°C). Enforce directly in the search space parameterization.
- *Unknown constraints:* Feasibility learned from data (synthesis fails for certain compositions, DFT calculation diverges, structure is mechanically unstable). Model feasibility as a separate binary classification GP and use a constrained acquisition: `EIC(x) = EI(x) × P(feasible | x)`.

BoTorch's `ConstrainedExpectedImprovement` and `qNoisyExpectedImprovement` with outcome constraints handle both cases.

### Noisy Objectives

When f(x) is observed with noise (experimental measurement error, DFT with loose convergence, stochastic simulation), the GP noise variance must be modeled explicitly:

- **Homoscedastic noise:** All observations have the same noise level σ²_n. Estimated as a GP hyperparameter.
- **Heteroscedastic noise:** Each observation has its own noise level (e.g., from DFT convergence quality or experimental error bars). Use a heteroscedastic GP or pass per-observation noise to BoTorch's `FixedNoiseGP`.
- **Noisy EI (NEI):** qNoisyExpectedImprovement accounts for observation noise in the best-so-far by integrating over uncertainty in f*. Use NEI instead of EI whenever objectives are noisy. [EXPERT REVIEW NEEDED: EI vs NEI matters most when noise is comparable to the improvement magnitude; for very low noise they are nearly equivalent]

## Key Workflows

### Workflow 1: Single-Objective BO with BoTorch (Continuous Space)

```python
# single_objective_bo.py
# Bayesian optimization of a materials property over continuous composition space.
# Example: optimize band gap as a function of x in Ti_{1-x}Sn_xO2 composition.
# Objective: maximize band gap (maximize f = -E_gap for minimization convention in BoTorch)

import torch
import numpy as np
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import qExpectedImprovement, qNoisyExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.utils.transforms import normalize, unnormalize

torch.manual_seed(42)
dtype = torch.double

# --- Search space: x in [0, 1] (fraction of Sn replacing Ti) ---
BOUNDS = torch.tensor([[0.0], [1.0]], dtype=dtype)   # shape (2, d)
D = 1   # search space dimensionality

def evaluate_objective(x_tensor):
    """
    Placeholder for the expensive black-box objective.
    In production: submit a VASP/QE calculation and parse the band gap.
    Here: synthetic function for illustration.
    Returns: (n, 1) tensor of objective values (higher = better).
    """
    x = x_tensor.numpy().flatten()
    # Synthetic: band gap (eV) for Ti_{1-x}Sn_xO2 — replace with real DFT/experiment
    # True optimum near x=0.3 in this toy model
    gap = 3.0 - 2.5 * (x - 0.3)**2 + 0.05 * np.random.randn(len(x))
    return torch.tensor(gap, dtype=dtype).unsqueeze(-1)

# --- Initial design: Latin hypercube or random ---
N_INIT = 5
N_ITER = 20
BATCH_SIZE = 1   # sequential; set > 1 for parallel batch BO

# Generate initial evaluations
X_init = torch.rand(N_INIT, D, dtype=dtype)
Y_init = evaluate_objective(X_init)

print(f"Initial design: {N_INIT} points, best = {Y_init.max().item():.4f}")

X_all = X_init.clone()
Y_all = Y_init.clone()

# --- BO loop ---
for iteration in range(N_ITER):
    # Fit GP on normalized inputs, standardized outputs
    X_norm = normalize(X_all, bounds=BOUNDS)
    # SingleTaskGP assumes homoscedastic noise; use FixedNoiseGP if noise is known
    gp = SingleTaskGP(X_norm, Y_all)
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)

    # Acquisition function: qNEI handles noisy objectives
    best_f = Y_all.max()
    acq = qNoisyExpectedImprovement(
        model=gp,
        X_baseline=X_norm,
        sampler=None,           # use default SobolQMCNormalSampler
        prune_baseline=True,    # remove dominated baseline points
    )

    # Optimize acquisition over normalized space
    candidates_norm, acq_value = optimize_acqf(
        acq_function=acq,
        bounds=normalize(BOUNDS, bounds=BOUNDS),   # [0,1]^d after normalization
        q=BATCH_SIZE,
        num_restarts=10,
        raw_samples=256,
        options={"batch_limit": 5, "maxiter": 200},
    )

    # Unnormalize candidate and evaluate
    x_next = unnormalize(candidates_norm, bounds=BOUNDS)
    y_next = evaluate_objective(x_next)

    X_all = torch.cat([X_all, x_next])
    Y_all = torch.cat([Y_all, y_next])

    print(f"  Iter {iteration+1:3d}: x={x_next.item():.4f}, f={y_next.item():.4f}, "
          f"best={Y_all.max().item():.4f}, acq={acq_value.item():.4e}")

best_idx = Y_all.argmax()
print(f"\nBest found: x={X_all[best_idx].item():.4f}, f={Y_all[best_idx].item():.4f}")
print(f"Evaluated {len(X_all)} points total ({N_INIT} init + {N_ITER} BO)")
```

### Workflow 2: Batch BO for Parallel HT-DFT Campaigns

When DFT jobs run in parallel, select a batch of q candidates per iteration instead of one.

```python
# batch_bo.py
# Batch Bayesian optimization for parallel DFT campaigns.
# Each iteration submits BATCH_SIZE calculations simultaneously.

import torch
import numpy as np
import pandas as pd
from botorch.models import SingleTaskGP, FixedNoiseGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition.logei import qLogExpectedImprovement, qLogNoisyExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.utils.transforms import normalize, unnormalize
from gpytorch.mlls import ExactMarginalLogLikelihood

dtype = torch.double
BATCH_SIZE = 8    # number of parallel DFT jobs per iteration
N_INIT = 16
N_ITER = 10
D = 5             # descriptor dimension (e.g., 5 Magpie features after PCA)

# Bounds per feature dimension
BOUNDS = torch.zeros(2, D, dtype=dtype)
BOUNDS[0] = -3.0   # lower bounds
BOUNDS[1] =  3.0   # upper bounds

def submit_dft_batch(X_candidates, iteration):
    """
    In production: generate structures from descriptors, write VASP inputs,
    submit SLURM jobs, wait for completion, parse results.
    Returns: (n_candidates, 1) tensor of DFT-computed properties.
    """
    # Placeholder: noisy quadratic (replace with real DFT call)
    x = X_candidates.numpy()
    y = -np.sum(x**2, axis=1, keepdims=True) + 0.1 * np.random.randn(len(x), 1)
    return torch.tensor(y, dtype=dtype)

def fit_and_select_batch(X_obs, Y_obs, bounds, batch_size, noise_var=None):
    """
    Fit a GP and select next batch via qLogNEI (log-space NEI for numerical stability).
    """
    X_norm = normalize(X_obs, bounds=bounds)
    bounds_norm = torch.stack([torch.zeros(bounds.shape[1]),
                                torch.ones(bounds.shape[1])])

    if noise_var is not None:
        # FixedNoiseGP for known per-observation noise
        Yvar = torch.full_like(Y_obs, noise_var)
        gp = FixedNoiseGP(X_norm, Y_obs, Yvar)
    else:
        gp = SingleTaskGP(X_norm, Y_obs)

    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)

    acq = qLogNoisyExpectedImprovement(
        model=gp,
        X_baseline=X_norm,
        prune_baseline=True,
        cache_root=True,       # cache the root decomposition for efficiency
    )

    candidates_norm, _ = optimize_acqf(
        acq_function=acq,
        bounds=bounds_norm,
        q=batch_size,
        num_restarts=20,
        raw_samples=512,
        sequential=False,      # jointly optimize entire batch (not greedy)
        options={"batch_limit": 5, "maxiter": 300},
    )
    return unnormalize(candidates_norm, bounds=bounds)

# Initial Latin hypercube design
from torch.quasirandom import SobolEngine
sobol = SobolEngine(dimension=D, scramble=True, seed=0)
X_init_unit = sobol.draw(N_INIT).to(dtype)
X_all = X_init_unit * (BOUNDS[1] - BOUNDS[0]) + BOUNDS[0]
Y_all = submit_dft_batch(X_all, iteration=0)
print(f"Init: {N_INIT} points, best Y = {Y_all.max().item():.4f}")

# BO iterations
for it in range(N_ITER):
    X_batch = fit_and_select_batch(X_all, Y_all, BOUNDS, BATCH_SIZE)
    Y_batch = submit_dft_batch(X_batch, iteration=it+1)
    X_all = torch.cat([X_all, X_batch])
    Y_all = torch.cat([Y_all, Y_batch])
    n_total = len(X_all)
    print(f"  Iter {it+1:2d}: submitted {BATCH_SIZE} jobs, "
          f"best so far = {Y_all.max().item():.4f} "
          f"(n_total={n_total})")

print(f"\nFinal best: {Y_all.max().item():.4f} after {len(X_all)} evaluations")
```

### Workflow 3: Multi-Objective BO with qEHVI (Pareto Front)

```python
# multi_objective_bo.py
# Multi-objective BO: simultaneously maximize band gap and minimize formation energy.
# Uses qExpectedHypervolumeImprovement (qEHVI) from BoTorch.

import torch
import numpy as np
import matplotlib.pyplot as plt
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition.multi_objective.logei import qLogExpectedHypervolumeImprovement
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.utils.multi_objective.hypervolume import Hypervolume
from botorch.optim import optimize_acqf
from botorch.utils.transforms import normalize, unnormalize
from gpytorch.mlls import ExactMarginalLogLikelihood

dtype = torch.double
NUM_OBJECTIVES = 2   # [band_gap (maximize), -formation_energy (maximize, i.e., minimize E_f)]
D = 4                # descriptor dimension
BATCH_SIZE = 4
N_INIT = 12
N_ITER = 15

# Reference point: must lie below all feasible objective values.
# Set to slightly below worst expected: [0.0 eV gap, -(-3.0) eV formation energy]
REF_POINT = torch.tensor([-0.5, -3.5], dtype=dtype)
BOUNDS = torch.stack([torch.full((D,), -2.0), torch.full((D,), 2.0)], dim=0).to(dtype)

def evaluate_objectives(X):
    """
    Returns: (n, 2) tensor of [band_gap, neg_formation_energy] — both to maximize.
    In production: two DFT properties parsed from VASP outputs.
    """
    x = X.numpy()
    # Synthetic bi-objective: replace with real DFT/experiment
    bg = 1.5 * np.exp(-0.3 * np.sum(x**2, axis=1)) + 0.5 * x[:, 0] + 0.05 * np.random.randn(len(x))
    ef = -0.8 * np.sum(x**2, axis=1) + 0.3 * x[:, 1] + 0.05 * np.random.randn(len(x))
    return torch.tensor(np.stack([bg, ef], axis=1), dtype=dtype)

# Initial design
from torch.quasirandom import SobolEngine
sobol = SobolEngine(D, scramble=True, seed=1)
X_all = sobol.draw(N_INIT).to(dtype) * 4.0 - 2.0
Y_all = evaluate_objectives(X_all)

def compute_hypervolume(Y, ref_point):
    """Compute dominated hypervolume of the current Pareto front."""
    pareto_mask = is_non_dominated(Y)
    pareto_Y = Y[pareto_mask]
    if len(pareto_Y) == 0:
        return 0.0
    hv = Hypervolume(ref_point=ref_point)
    return hv.compute(pareto_Y)

hv_history = [compute_hypervolume(Y_all, REF_POINT)]
print(f"Init: HV = {hv_history[-1]:.4f}, Pareto front size = {is_non_dominated(Y_all).sum().item()}")

for it in range(N_ITER):
    X_norm = normalize(X_all, bounds=BOUNDS)
    bounds_norm = torch.zeros(2, D, dtype=dtype); bounds_norm[1] = 1.0

    # Fit independent GPs for each objective (correlated GPs via MTGP are more accurate
    # but harder to fit; independent GPs are standard for 2–3 objectives)
    gp = SingleTaskGP(X_norm, Y_all, outcome_transform=Standardize(m=NUM_OBJECTIVES))
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)

    # Compute current Pareto front for partitioning
    with torch.no_grad():
        pareto_mask = is_non_dominated(Y_all)
        ref_point_standardized = (REF_POINT - Y_all.mean(0)) / Y_all.std(0)

    acq = qLogExpectedHypervolumeImprovement(
        model=gp,
        ref_point=REF_POINT,
        partitioning=None,    # BoTorch auto-computes box decomposition
        Y_pmean=Y_all,        # provide observed Y for NEHVI variant
    )

    candidates_norm, _ = optimize_acqf(
        acq_function=acq,
        bounds=bounds_norm,
        q=BATCH_SIZE,
        num_restarts=10,
        raw_samples=256,
        sequential=True,    # sequential greedy for multi-objective (faster than joint)
        options={"batch_limit": 5, "maxiter": 200},
    )
    X_batch = unnormalize(candidates_norm, bounds=BOUNDS)
    Y_batch = evaluate_objectives(X_batch)
    X_all = torch.cat([X_all, X_batch])
    Y_all = torch.cat([Y_all, Y_batch])

    hv = compute_hypervolume(Y_all, REF_POINT)
    hv_history.append(hv)
    pareto_size = is_non_dominated(Y_all).sum().item()
    print(f"  Iter {it+1:2d}: HV = {hv:.4f}, Pareto size = {pareto_size}")

# Plot Pareto front
pareto_mask = is_non_dominated(Y_all)
Y_np = Y_all.numpy()
fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(Y_np[~pareto_mask, 0], Y_np[~pareto_mask, 1], c="lightgray", s=30, label="Dominated")
ax.scatter(Y_np[pareto_mask, 0], Y_np[pareto_mask, 1], c="steelblue", s=60, label="Pareto front")
# Sort Pareto front for line plot
pf = Y_np[pareto_mask]
pf = pf[np.argsort(pf[:, 0])]
ax.plot(pf[:, 0], pf[:, 1], "b--", lw=0.8)
ax.set_xlabel("Band gap (eV)")
ax.set_ylabel("–Formation energy (eV/atom)")
ax.set_title(f"Pareto front after {len(X_all)} evaluations")
ax.legend()
plt.savefig("pareto_front.png", dpi=150, bbox_inches="tight")

# HV convergence
fig2, ax2 = plt.subplots(figsize=(6, 4))
ax2.plot(range(len(hv_history)), hv_history, "o-", color="steelblue")
ax2.set_xlabel("BO iteration"); ax2.set_ylabel("Hypervolume")
ax2.set_title("Multi-objective BO convergence")
plt.savefig("hv_convergence.png", dpi=150, bbox_inches="tight")
```

### Workflow 4: Constrained BO with Feasibility Surrogate

```python
# constrained_bo.py
# Constrained BO where some compositions are infeasible (synthesis fails,
# DFT does not converge, structure unstable).
# Models feasibility as a separate binary GP classifier.

import torch
import numpy as np
from botorch.models import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.acquisition import ConstrainedExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.utils.transforms import normalize, unnormalize
from gpytorch.mlls import SumMarginalLogLikelihood

dtype = torch.double
D = 3
BOUNDS = torch.tensor([[0.0] * D, [1.0] * D], dtype=dtype)

def evaluate_constrained(X):
    """
    Returns: (n, 2) tensor of [objective, constraint].
    Constraint: c(x) <= 0 means feasible.
    Objective: larger is better (only evaluated when feasible).
    In production: objective = DFT band gap; constraint = synthesis feasibility check.
    """
    x = X.numpy()
    obj = -np.sum((x - 0.5)**2, axis=1, keepdims=True)  # maximum at x=0.5
    # Constraint: infeasible if x[0] + x[1] > 1.2 (e.g., charge balance violation)
    c = (x[:, 0:1] + x[:, 1:2] - 1.2)   # c > 0 = infeasible
    return (torch.tensor(obj, dtype=dtype),
            torch.tensor(c, dtype=dtype))

# Initial design: random, some may be infeasible
N_INIT = 10
X_all = torch.rand(N_INIT, D, dtype=dtype)
Y_obj_list, Y_con_list = [], []
for i in range(N_INIT):
    obj, con = evaluate_constrained(X_all[i:i+1])
    Y_obj_list.append(obj)
    Y_con_list.append(con)

Y_obj = torch.cat(Y_obj_list)
Y_con = torch.cat(Y_con_list)

feasible = (Y_con <= 0).squeeze()
print(f"Init: {N_INIT} points, {feasible.sum().item()} feasible, "
      f"best feasible = {Y_obj[feasible].max().item():.4f if feasible.any() else 'N/A'}")

N_ITER = 20
for it in range(N_ITER):
    X_norm = normalize(X_all, bounds=BOUNDS)
    bounds_norm = torch.zeros(2, D, dtype=dtype); bounds_norm[1] = 1.0

    # Fit objective GP and constraint GP as a ModelList
    gp_obj = SingleTaskGP(X_norm, Y_obj)
    gp_con = SingleTaskGP(X_norm, Y_con)
    model = ModelListGP(gp_obj, gp_con)
    mll = SumMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)

    # ConstrainedExpectedImprovement: EI × P(constraint satisfied)
    best_f = Y_obj[feasible].max() if feasible.any() else Y_obj.max()
    acq = ConstrainedExpectedImprovement(
        model=model,
        best_f=best_f,
        objective_index=0,
        constraints={1: (None, 0.0)},   # constraint output index 1, upper bound 0.0
    )

    candidate_norm, _ = optimize_acqf(
        acq_function=acq,
        bounds=bounds_norm,
        q=1,
        num_restarts=10,
        raw_samples=256,
    )
    x_next = unnormalize(candidate_norm, bounds=BOUNDS)
    obj_next, con_next = evaluate_constrained(x_next)

    X_all = torch.cat([X_all, x_next])
    Y_obj = torch.cat([Y_obj, obj_next])
    Y_con = torch.cat([Y_con, con_next])
    feasible = (Y_con <= 0).squeeze()

    status = "feasible" if (con_next <= 0).all() else "INFEASIBLE"
    best_now = Y_obj[feasible].max().item() if feasible.any() else float('nan')
    print(f"  Iter {it+1:2d}: {status}, obj={obj_next.item():.4f}, "
          f"con={con_next.item():.4f}, best_feasible={best_now:.4f}")
```

### Workflow 5: Multi-Fidelity BO (Screening DFT → Converged DFT)

```python
# multi_fidelity_bo.py
# Multi-fidelity BO: cheap low-fidelity (screening DFT) + expensive high-fidelity.
# Uses BoTorch's SingleTaskMultiFidelityGP with cost-aware acquisition.

import torch
import numpy as np
from botorch.models.gp_regression_fidelity import SingleTaskMultiFidelityGP
from botorch.models.transforms.outcome import Standardize
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition.cost_aware import InverseCostWeightedUtility
from botorch.acquisition.fixed_feature import FixedFeatureAcquisitionFunction
from botorch.acquisition.knowledge_gradient import qMultiFidelityMaxValueEntropySearch
from botorch.optim import optimize_acqf, optimize_acqf_mixed
from botorch.utils.transforms import normalize, unnormalize
from gpytorch.mlls import ExactMarginalLogLikelihood

dtype = torch.double
D_PROBLEM = 3       # actual search space dimension
# The last column is the fidelity indicator: 0.0 = low fidelity, 1.0 = high fidelity
D_TOTAL = D_PROBLEM + 1
BOUNDS = torch.tensor([[0.0] * D_PROBLEM + [0.0],
                        [1.0] * D_PROBLEM + [1.0]], dtype=dtype)

COST_LOW  = 1.0     # relative cost: 1 unit (screening DFT, ~minutes)
COST_HIGH = 20.0    # relative cost: 20x (converged DFT, ~hours)
FIDELITIES = [0.0, 1.0]   # two fidelity levels; extend to [0.0, 0.5, 1.0] for three

def evaluate_at_fidelity(X_with_fidelity):
    """
    X_with_fidelity: (n, D_TOTAL) tensor, last column = fidelity.
    Low fidelity: coarse DFT (ENCUT=300, 1x1x1 k-mesh).
    High fidelity: converged DFT (ENCUT=550, 6x6x6 k-mesh).
    """
    x = X_with_fidelity[:, :D_PROBLEM].numpy()
    fidelity = X_with_fidelity[:, -1].numpy()
    # Synthetic: high-fidelity = true value; low-fidelity = true + bias + noise
    y_true = -np.sum((x - 0.5)**2, axis=1)
    y_low  = y_true + 0.3 * np.random.randn(len(x))  # biased cheap proxy
    y_high = y_true + 0.05 * np.random.randn(len(x)) # accurate expensive result
    y = np.where(fidelity >= 0.9, y_high, y_low)
    return torch.tensor(y, dtype=dtype).unsqueeze(-1)

# Initial design: mostly low fidelity with a few high-fidelity points
N_INIT_LOW  = 12
N_INIT_HIGH = 4
X_low  = torch.cat([torch.rand(N_INIT_LOW, D_PROBLEM, dtype=dtype),
                     torch.zeros(N_INIT_LOW, 1, dtype=dtype)], dim=1)
X_high = torch.cat([torch.rand(N_INIT_HIGH, D_PROBLEM, dtype=dtype),
                     torch.ones(N_INIT_HIGH, 1, dtype=dtype)], dim=1)
X_all = torch.cat([X_low, X_high])
Y_all = evaluate_at_fidelity(X_all)

print(f"Init: {N_INIT_LOW} low + {N_INIT_HIGH} high fidelity points")
print(f"Total cost so far: {N_INIT_LOW*COST_LOW + N_INIT_HIGH*COST_HIGH:.0f} units")

BUDGET = 200.0   # total cost budget in relative units
cost_spent = N_INIT_LOW * COST_LOW + N_INIT_HIGH * COST_HIGH

N_ITER = 15
for it in range(N_ITER):
    if cost_spent >= BUDGET:
        print("Budget exhausted.")
        break

    X_norm = normalize(X_all, bounds=BOUNDS)
    bounds_norm = torch.zeros(2, D_TOTAL, dtype=dtype); bounds_norm[1] = 1.0

    # Multi-fidelity GP (linear downsampling kernel on fidelity dimension)
    gp = SingleTaskMultiFidelityGP(
        X_norm, Y_all,
        data_fidelities=[D_TOTAL - 1],   # last column is fidelity
        outcome_transform=Standardize(m=1),
    )
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)

    # Choose fidelity: if cost budget remaining is tight, use low fidelity;
    # otherwise use knowledge gradient to select fidelity jointly
    remaining = BUDGET - cost_spent
    if remaining < COST_HIGH * 2:
        chosen_fidelity = 0.0   # switch to cheap evaluations only
    else:
        chosen_fidelity = 1.0   # exploit with high-fidelity near convergence

    # Fixed-fidelity EI acquisition
    acq_base = qMultiFidelityMaxValueEntropySearch(
        model=gp,
        target_fidelities={D_TOTAL - 1: 1.0},   # optimize for high-fidelity target
        num_fantasies=64,
    )
    # Fix fidelity column to chosen_fidelity
    acq_fixed = FixedFeatureAcquisitionFunction(
        acq_function=acq_base,
        d=D_TOTAL,
        columns=[D_TOTAL - 1],
        values=[chosen_fidelity],
    )
    candidate_norm, _ = optimize_acqf(
        acq_function=acq_fixed,
        bounds=bounds_norm[:, :D_PROBLEM],   # optimize only over problem dims
        q=1,
        num_restarts=10,
        raw_samples=256,
    )
    x_full_norm = torch.cat([candidate_norm,
                              torch.tensor([[chosen_fidelity]], dtype=dtype)], dim=1)
    x_next = unnormalize(x_full_norm, bounds=BOUNDS)
    y_next = evaluate_at_fidelity(x_next)

    X_all = torch.cat([X_all, x_next])
    Y_all = torch.cat([Y_all, y_next])

    step_cost = COST_HIGH if chosen_fidelity >= 0.9 else COST_LOW
    cost_spent += step_cost

    # Only high-fidelity results count for the "best found"
    hf_mask = X_all[:, -1] >= 0.9
    best_hf = Y_all[hf_mask].max().item() if hf_mask.any() else float('nan')
    print(f"  Iter {it+1:2d}: fidelity={chosen_fidelity:.0f}, y={y_next.item():.4f}, "
          f"cost_spent={cost_spent:.0f}/{BUDGET:.0f}, best_HF={best_hf:.4f}")
```

### Workflow 6: BO over Discrete/Categorical Space (Element Selection)

```python
# discrete_bo.py
# BO over a discrete chemical space: selecting dopant element for a perovskite.
# Search space: B-site cation in ABO3 chosen from a list of 20 transition metals.
# Uses Ax's SearchSpace with ChoiceParameter for categorical variables.

from ax.service.ax_client import AxClient, ObjectiveProperties
from ax.modelbridge.registry import Models
import numpy as np

# Candidate B-site elements (reduced set for illustration)
B_SITE_ELEMENTS = [
    "Ti", "Zr", "Hf", "V", "Nb", "Ta", "Cr", "Mo", "W",
    "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "In", "Sn", "Pb", "Bi"
]

def run_dft_for_element(element):
    """
    Placeholder: compute band gap of ABO3 perovskite with given B-site element via VASP.
    Returns band gap in eV (higher = more useful for photovoltaics in this example).
    """
    # Synthetic values — replace with real DFT workflow
    np.random.seed(hash(element) % 2**32)
    known_gaps = {"Ti": 3.0, "Fe": 2.1, "Co": 1.8, "Mn": 1.5, "Ni": 3.4,
                  "Nb": 3.2, "V": 2.8, "Cr": 1.2, "Cu": 2.0, "Zn": 3.5}
    return known_gaps.get(element, np.random.uniform(0.5, 4.0)) + 0.1 * np.random.randn()

# --- Set up Ax client ---
ax_client = AxClient()
ax_client.create_experiment(
    name="perovskite_B_site_optimization",
    parameters=[
        {
            "name": "B_site",
            "type": "choice",
            "values": B_SITE_ELEMENTS,
            "is_ordered": False,
            "sort_values": False,
        }
    ],
    objectives={"band_gap": ObjectiveProperties(minimize=False)},   # maximize band gap
    parameter_constraints=[],
    tracking_metric_names=[],
)

# --- BO loop ---
N_TRIALS = 25

for trial_idx in range(N_TRIALS):
    # First few trials: random/Sobol initialization; subsequent: BO with GP
    parameters, trial_index = ax_client.get_next_trial()
    element = parameters["B_site"]

    # Run DFT
    band_gap = run_dft_for_element(element)

    ax_client.complete_trial(
        trial_index=trial_index,
        raw_data={"band_gap": (band_gap, 0.05)},   # (mean, SEM)
    )
    print(f"  Trial {trial_idx+1:3d}: B={element:4s}, Eg={band_gap:.3f} eV")

# --- Report results ---
best_params, values = ax_client.get_best_parameters()
print(f"\nBest B-site element: {best_params['B_site']}")
print(f"Best band gap: {values[0]['band_gap']:.3f} eV")

# Get full trial data sorted by performance
trials_df = ax_client.get_trials_data_frame()
print("\nTop 5 candidates:")
print(trials_df.sort_values("band_gap", ascending=False).head(5)[["B_site", "band_gap"]])
```

### Workflow 7: BO Baseline Comparison and Stopping Analysis

```python
# bo_benchmarking.py
# Compare BO against random search on a retrospective dataset.
# Critical: always report a random search baseline alongside BO results.

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

def retrospective_bo_benchmark(X_all, Y_all, n_init, n_iter, batch_size,
                                n_seeds=20, method="random"):
    """
    Retrospective BO benchmark on a pre-computed dataset (X_all, Y_all).
    Simulates a sequential selection process using the dataset as the oracle.
    Returns: array of shape (n_seeds, n_iter+1) with best-found Y at each iteration.
    """
    n_total = len(Y_all)
    results = np.zeros((n_seeds, n_iter + 1))

    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        # Initial random selection
        init_idx = rng.choice(n_total, n_init, replace=False)
        selected = list(init_idx)
        best = Y_all[selected].max()
        results[seed, 0] = best

        for it in range(n_iter):
            unselected = list(set(range(n_total)) - set(selected))
            if not unselected:
                break

            if method == "random":
                # Random selection baseline
                next_idx = rng.choice(unselected, min(batch_size, len(unselected)),
                                      replace=False)
            elif method == "greedy_max":
                # Greedy: always pick the highest-predicted (cheating oracle baseline)
                next_idx = np.argsort(Y_all[unselected])[-batch_size:]
                next_idx = np.array(unselected)[next_idx]
            # Real BO would use GP + acquisition here; shown as placeholder
            # In production: fit GP on X_all[selected], Y_all[selected]
            #                then score X_all[unselected] by acquisition function
            else:
                raise ValueError(f"Unknown method: {method}")

            selected.extend(next_idx)
            best = Y_all[selected].max()
            results[seed, it + 1] = best

    return results

# Example: load a pre-computed materials dataset
# X_all = featurized structures (n_materials, n_features)
# Y_all = computed property (n_materials,)
# Replace with real dataset loading:
rng = np.random.default_rng(0)
n_materials = 500
X_fake = rng.standard_normal((n_materials, 10))
Y_fake = -np.sum(X_fake[:, :3]**2, axis=1) + rng.normal(0, 0.1, n_materials)
Y_fake = (Y_fake - Y_fake.mean()) / Y_fake.std()

N_INIT, N_ITER, BATCH = 10, 20, 5

random_results = retrospective_bo_benchmark(X_fake, Y_fake, N_INIT, N_ITER, BATCH,
                                              n_seeds=30, method="random")
greedy_results = retrospective_bo_benchmark(X_fake, Y_fake, N_INIT, N_ITER, BATCH,
                                              n_seeds=30, method="greedy_max")

# Plot with confidence intervals
iters = np.arange(N_ITER + 1)
fig, ax = plt.subplots(figsize=(7, 4))
for results, label, color in [
    (random_results, "Random search", "steelblue"),
    (greedy_results, "Oracle greedy (upper bound)", "firebrick"),
]:
    mean = results.mean(axis=0)
    sem = results.std(axis=0) / np.sqrt(len(results))
    ax.plot(iters, mean, color=color, lw=1.8, label=label)
    ax.fill_between(iters, mean - sem, mean + sem, alpha=0.2, color=color)

ax.set_xlabel("BO iteration")
ax.set_ylabel("Best property found (σ)")
ax.set_title("BO benchmarking on retrospective dataset")
ax.legend()
plt.savefig("bo_benchmark.png", dpi=150, bbox_inches="tight")

# Statistical test at final iteration
u_stat, p_val = mannwhitneyu(greedy_results[:, -1], random_results[:, -1],
                              alternative="greater")
print(f"\nFinal iteration — greedy vs random: Mann-Whitney U={u_stat:.0f}, p={p_val:.4f}")
print(f"Random search best (mean ± std): {random_results[:, -1].mean():.3f} ± {random_results[:, -1].std():.3f}")
print(f"Always report random search as baseline. BO gains are meaningful only relative to it.")
```

## Common Failure Modes

| Failure Mode | Symptoms | Likely Cause | Diagnostic | Fix |
|---|---|---|---|---|
| Overconfident surrogate | BO selects candidates in the same region repeatedly; GP posterior variance collapses far from data | Kernel length scale too short; feature space collapses in high dimensions | Plot GP posterior variance over the search space; check calibration on held-out points | Recalibrate GP; try ARD kernel; reduce feature dimensionality; add jitter to diagonal |
| Bad search-space definition | BO finds candidates outside the physically meaningful range; many infeasible evaluations | Bounds set too wide; composition constraints not enforced; categorical choices include impossible elements | Enumerate a sample of candidates and manually inspect feasibility; compute feasibility fraction | Tighten bounds; add linear constraints; use feasibility surrogate; validate search space before submitting |
| Acquisition over-optimization | BO selects the same point or a tight cluster every iteration | Acquisition function has a narrow global maximum; numerical precision issues in L-BFGS-B | Check diversity of selected candidates; increase `num_restarts` and `raw_samples`; use multi-start | Increase `num_restarts` (≥20); use Sobol-based initialization for L-BFGS-B; switch to TS |
| Data leakage | BO benchmark appears much better than random; inflated gains | Test points in initial design; train/test split applied after BO; dataset not deduplicated | Check if initial design or BO selections appear in the test set | Deduplicate; create test split before any BO; use held-out structures never seen during optimization |
| Invalid candidates | DFT calculation fails or returns nonsensical energy for BO-selected candidate | BO ignores physical validity; selected composition is charge-unbalanced or mechanically unstable | Check failure rate of DFT jobs selected by BO vs random; inspect selected compositions | Add known-constraint enforcement; add feasibility surrogate trained on DFT success/failure outcomes |
| Optimizing noisy artifacts | BO converges to a candidate with an anomalously low (or high) DFT energy that is a numerical artifact | Noisy DFT convergence; custodian applied many corrections; basis set incompleteness | Compare single-point energies of "best" candidates at higher ENCUT and tighter EDIFF | Flag high-uncertainty DFT outputs; rerun suspicious best candidates at high-accuracy tier |
| Ignoring constraints | Selected candidates violate composition, stability, or experimental limits | Constraints were not encoded in the search space or acquisition | Check constraint satisfaction rate; plot selected candidates against feasibility boundary | Use `ConstrainedExpectedImprovement`; encode constraints as linear parameter constraints in BoTorch |
| Reporting only best-found without baseline | Claimed improvement over random is not statistically supported | Only the BO result is reported; no comparison to random search or heuristic | Re-run random search with same budget and n_seeds ≥ 20; compute p-value | Always include random search baseline; report mean ± std over seeds; use Mann-Whitney U test |
| Surrogate not re-fit after each batch | GP becomes stale; uncertainty not updated; acquisition degrades | Re-fitting skipped for speed; model hyperparameters not optimized each iteration | Check if GP hyperparameters change between iterations; compare predictions to held-out data | Re-fit GP after every completed batch; use `fit_gpytorch_mll` with optimizer each iteration |
| Wrong reference point in multi-objective BO | Hypervolume metric is dominated by one objective; Pareto front metrics inconsistent | Reference point chosen inside the feasible region or not updated as new data arrives | Verify reference point is below all observed objective values; plot dominated region | Set reference point ≤ worst feasible observation per objective; update conservatively if observations improve |

## Best Practices

**Search space design:**
- Define bounds, constraints, and feasibility checks before the first iteration. A search space that must be changed mid-campaign invalidates comparisons across iterations.
- For compositional spaces, always enforce the simplex constraint (Σx_i = 1) explicitly, not post-hoc. BoTorch's `LinearConstraint` handles this natively.
- For element-selection problems, use Ax's `ChoiceParameter` with appropriate one-hot or ordinal encoding; do not arbitrarily impose ordinal order on categorical choices.
- Start with the minimum number of features needed (3–10). Adding more features increases the curse of dimensionality without necessarily improving the surrogate.

**Surrogate fitting and calibration:**
- Always check GP calibration on a held-out subset before trusting the uncertainty for acquisition. A coverage probability plot (fraction of test points inside predicted 90% interval vs. 90%) is the minimum calibration check.
- Re-fit GP hyperparameters (length scales, output scale, noise) after every batch. Stale hyperparameters lead to poor uncertainty estimates.
- For objectives with known noise levels (DFT convergence error, experimental uncertainty), use `FixedNoiseGP` with per-observation noise rather than estimating noise from data.

**Acquisition function selection:**
- Use `qLogNoisyExpectedImprovement` (qLogNEI) as the default for single-objective BO with noise. It is numerically more stable than qEI and handles the noisy best-so-far correctly.
- For multi-objective problems, use `qLogExpectedHypervolumeImprovement` with `sequential=True` for batches; joint optimization is only needed for very small batches.
- For pure exploration (building a training dataset without a specific target), use posterior variance maximization or Thompson Sampling rather than EI.
- For noisy, low-budget settings where calibration is uncertain, consider Thompson Sampling as a robust fallback.

**Initialization:**
- Use Sobol sequences (quasi-random) rather than uniform random for the initial design; they cover the space more uniformly with the same number of points.
- Initial design size: at least max(5, 2×D) points for a D-dimensional space. For multi-objective problems: at least max(10, 4×D).
- Never use a biased initialization (e.g., only known-good compositions) without explicitly accounting for it in the surrogate and baseline comparison.

**Baselines and reporting:**
- Always run ≥ 10 random search seeds with the same budget and report mean ± std. BO that does not consistently beat random search is not doing useful work.
- Report: search space definition, initial design size, surrogate type and kernel, acquisition function, batch size, total evaluations, and how the best candidate was selected (not just identified by BO but independently validated).
- For materials discovery claims, validate the BO-identified optimum with a higher-accuracy DFT calculation (high-accuracy tier) or synthesis experiment before publishing.

**Multi-fidelity:**
- Establish the correlation between fidelities empirically on a calibration set before committing to a multi-fidelity protocol. If low-fidelity predictions are uncorrelated with high-fidelity (ρ < 0.3), multi-fidelity BO provides no benefit. [EXPERT REVIEW NEEDED]
- Track total cost (not total evaluations) as the x-axis in multi-fidelity convergence plots.

## Integration with Other Skills

**Surrogate and modeling:**

- [Gaussian Process Materials](../gaussian-process-materials/SKILL.md) — the GP surrogate at the heart of BO is built and calibrated using the GP skill. Kernel selection, descriptor choice, length scale fitting, and uncertainty calibration all follow that skill's protocols. BoTorch uses GPyTorch-based GPs that are directly compatible.
- [Surrogate Validation](../surrogate-validation/SKILL.md) — before trusting BO acquisition decisions, the surrogate must pass cross-validation, calibration, and generalization checks defined in the surrogate validation skill. A BO loop built on an uncalibrated surrogate is unreliable. [FORWARD REFERENCE — skill may be in progress]
- [Small-Data ML](../../scientific-ml/small-data-ml/SKILL.md) — BO operates in the small-data regime (tens to hundreds of evaluations); this skill covers regularization, cross-validation, and model selection strategies that apply both inside the surrogate and when comparing surrogate families. [FORWARD REFERENCE — skill not yet created]

**Data and featurization:**

- [Materials Featurization](../../scientific-data/materials-featurization/SKILL.md) — descriptors (Magpie, SOAP, MBTR, graph features) that map structures or compositions to fixed-length vectors are required before a GP or neural surrogate can operate over the chemical space. The featurization skill provides the descriptor pipelines used in Workflows 1–5.
- [Materials Databases Access](../../scientific-data/materials-databases-access/SKILL.md) — the initial candidate pool for BO screening is often drawn from Materials Project, AFLOW, or OQMD. Compatibility checking and structure deduplication from the databases skill are required before building the search space.
- [Scientific Visualization](../../scientific-data/scientific-visualization/SKILL.md) — Pareto front plots, acquisition function landscape visualization, GP posterior plots, and BO convergence curves are the standard outputs of BO campaigns; this skill provides the plotting conventions and multi-panel figure standards.
- [Reproducibility Workflow](../../scientific-data/reproducibility-workflow/SKILL.md) — BO campaigns involve many coupled objects: the dataset, the surrogate checkpoint, the acquisition log, the evaluation oracle (DFT inputs and outputs), and the final candidate. All must be versioned and archived together for the campaign to be reproducible.

**Evaluation oracle:**

- [High-Throughput DFT](../../electronic-structure/high-throughput-dft/SKILL.md) — in materials BO, the evaluation oracle is usually HT-DFT. The batch selection from BO maps directly to job submission in atomate2/FireWorks. The HT-DFT skill provides job orchestration; this skill provides the acquisition logic that determines what to submit next.
- [DFT Convergence](../../electronic-structure/dft-convergence/SKILL.md) — the DFT convergence tier used for BO evaluations must be documented. Mixing convergence tiers (screening vs. high-accuracy) within the BO dataset corrupts the surrogate.

**Active learning:**

- [Active Learning MLP](../../ml-interatomic-potentials/active-learning-mlp/SKILL.md) — BO is a special case of active learning where the query criterion is an acquisition function targeting a single optimum or Pareto front, rather than uncertainty reduction across the full space. This skill and the active-learning-mlp skill share the surrogate → acquisition → oracle → update loop structure.

## Key References

1. Shahriari, B., Swersky, K., Wang, Z., Adams, R. P. & de Freitas, N. (2016). "Taking the human out of the loop: A review of Bayesian optimization." *Proceedings of the IEEE* 104, 148–175. Comprehensive review of acquisition functions, theoretical guarantees, and applications.
2. Balandat, M., Karrer, B., Jiang, D. R., Daulton, S., Letham, B., Wilson, A. G. & Bakshy, E. (2020). "BoTorch: A framework for efficient Monte-Carlo Bayesian optimization." *Advances in Neural Information Processing Systems* 33. The primary BoTorch reference; cite when using BoTorch.
3. Daulton, S., Balandat, M. & Bakshy, E. (2020). "Differentiable expected hypervolume improvement for parallel multi-objective Bayesian optimization." *Advances in Neural Information Processing Systems* 33. qEHVI reference for multi-objective BO.
4. Hernández-Lobato, J. M., Hoffman, M. W. & Ghahramani, Z. (2014). "Predictive entropy search for efficient global optimization of black-box functions." *Advances in Neural Information Processing Systems* 27. Entropy Search acquisition function.
5. Snoek, J., Larochelle, H. & Adams, R. P. (2012). "Practical Bayesian optimization of machine learning algorithms." *Advances in Neural Information Processing Systems* 25. Foundational practical BO paper with GP-based acquisition and hyperparameter optimization.
6. Griffiths, R.-R. & Hernández-Lobato, J. M. (2020). "Constrained Bayesian optimization for automatic chemical design using variational autoencoders." *Chemical Science* 11, 577–586. Constrained BO with learned latent spaces for molecular design.
7. Häse, F., Roch, L. M. & Aspuru-Guzik, A. (2019). "Phoenics: A Bayesian optimizer for chemistry." *ACS Central Science* 5, 1231–1241. BO applied to chemistry optimization campaigns; covers periodic and categorical spaces.
8. Forrester, A. I. J., Sóbester, A. & Keane, A. J. (2008). *Engineering Design via Surrogate Modelling: A Practical Guide*. Wiley. Textbook-level reference for surrogate-assisted optimization including multi-fidelity methods.

[EXPERT REVIEW NEEDED: The optimal correlation threshold (ρ < 0.3) for deciding when multi-fidelity BO is not beneficial is a practical rule of thumb; verify against recent multi-fidelity BO benchmarks. Confirm whether qLogNEI is always preferred over qNEI in BoTorch 0.10+ or whether there are regression cases. Review whether the Mann-Whitney U test is the most appropriate statistical test for comparing BO vs. random search regret distributions, as these may not be independent across seeds.]
