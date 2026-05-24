---
name: small-data-ml
domain: scientific-ml
maturity: draft
tools:
  - scikit-learn
  - pandas
  - numpy
  - matplotlib
  - scipy
  - matminer
  - shap
tested_versions:
  scikit-learn: "1.4.0"
  pandas: "2.2.0"
  numpy: "1.26.4"
  matplotlib: "3.8.0"
  scipy: "1.12.0"
  matminer: "0.9.2"
requires_gpu: false
estimated_runtime: minutes-to-hours
---

# Small-Data Machine Learning for Materials and Chemistry

## Description

This skill covers machine learning on small scientific datasets (roughly 20–2000 samples) in materials science and chemistry: leakage-safe splitting strategies for correlated structures and compositions, baseline model selection, descriptor screening, uncertainty quantification, learning curves, model selection under limited data, transfer learning from pretrained representations, and rigorous reporting of performance including confidence intervals and negative results. Invoke this skill when building a property predictor, surrogate model, or screening model from a dataset small enough that every methodological choice — splitting strategy, baseline, hyperparameter tuning procedure — materially affects the reported metrics.

## Domain Context

Most published materials ML models are trained on fewer than a few thousand samples. This is not a weakness — DFT, synthesis, and characterization are expensive — but it creates methodological traps that are easy to miss and hard to detect from the outside. High R² on a random 80/20 split of a 200-sample dataset is not evidence of a useful model; it may reflect correlation leakage, a narrow target range, or an over-tuned hyperparameter.

**Why scientific small-data is different from standard ML small-data:**

In general ML, small-data problems usually assume i.i.d. samples. Materials datasets violate this in structured ways:
- *Compositional correlation:* Two alloys that differ by 1 at% are nearly identical in descriptor space and probably in target space. Random splitting separates near-duplicates across train and test, making the model look more accurate than it generalizes.
- *Structural prototype correlation:* All rocksalt oxides share the same topology; a model trained on MgO, NiO, CoO will trivially interpolate to FeO even under random splitting.
- *Database-sourced duplicates:* Materials Project, AFLOW, and OQMD share many structures computed by different methods; deduplication before splitting is mandatory.
- *Experimental campaign structure:* Samples in a synthesis campaign often share substrate, precursor, or equipment — a group that must be held out together, not split randomly.
- *Active-learning iteration structure:* Samples from later iterations are in regions near those from earlier iterations. Splitting across iterations pretends the model was trained on future knowledge.

**Signal-to-noise ratio:** DFT targets are exact given the settings (no measurement noise), but carry systematic errors (functional, ENCUT, k-mesh). Experimental targets carry both measurement noise and sample-to-sample variability. A model whose MAE is less than the experimental measurement uncertainty is not doing better than a mean predictor — it is fitting measurement noise.

**Extrapolation vs interpolation:** Small datasets typically do not cover the full composition, structure, or synthesis-condition space. A model that interpolates within the training hull may look excellent but extrapolate catastrophically. The deployment regime (same chemistry? new chemistry family? new element? different processing?) must be stated and tested, not assumed.

**Baseline models are not optional:** Ridge regression, random forests, and k-nearest-neighbor with good descriptors often perform comparably to neural networks on small materials datasets. If a complex model does not beat simple baselines by a statistically meaningful margin, the complexity adds no value. [EXPERT REVIEW NEEDED: threshold beyond which neural networks consistently outperform GPs/RF on materials small-data is dataset-specific; 500–2000 samples is a common empirical transition zone but it varies with descriptor quality]

## When to Use This Skill

- Building a property surrogate or screening model from a dataset of 20–2000 structures, compositions, or experimental conditions.
- Auditing a published or inherited materials ML model for methodological soundness before extending or citing it.
- Choosing a splitting strategy for a dataset with known correlation structure (same prototype, same campaign, same active-learning iteration).
- Diagnosing whether a high R² or low MAE reflects genuine generalization or a data artifact.
- Setting up nested cross-validation for rigorous model selection when a hold-out test set is not available.
- Determining whether the current dataset is large enough to support a reliable model, or whether more data are needed.

## When Not to Use This Skill

- **Large datasets (> ~5000 samples):** With enough data, random splits approximate i.i.d., learning curves saturate, and neural networks become competitive. Use standard ML validation practices.
- **When the target is a trajectory or time series:** Correlation structure is temporal; use specialized time-series CV (forward chaining), not group-based spatial splitting.
- **When data quality has not been audited:** Validation techniques produce misleading metrics on datasets with unit errors, unconverged DFT, mixed functionals, or undisclosed duplicates. Audit provenance before modeling.
- **When the dataset has fewer than ~20 samples:** Below ~20 samples, any train/test split is statistically unreliable. Use physical reasoning, Bayesian priors, or transfer learning exclusively; do not report ML performance metrics.
- **When the scientific claim requires extrapolation to a new chemical space and no held-out examples from that space exist:** Validation on interpolating test sets cannot support extrapolation claims. Collect or use existing data from the target space.

## Core Concepts

### Dataset Size and Regime Classification

| Samples | Regime | Recommended approach |
|---|---|---|
| < 20 | Very small | Physical reasoning, GP with informative priors, or transfer learning only; no standard CV |
| 20–100 | Small | LOOCV or nested k-fold; GP or kernel ridge; simple baselines; no neural nets |
| 100–500 | Moderate-small | Nested k-fold or LOGO-CV; GP, RF, gradient boosting; learning curves required |
| 500–2000 | Moderate | Standard nested CV; RF, gradient boosting, GPs; neural surrogates can be tested |
| > 2000 | Grows toward standard ML; use the appropriate ML skill | |

These boundaries are rough. Dataset effective size also depends on target variance, descriptor quality, and correlation structure. [EXPERT REVIEW NEEDED]

### Correlation Leakage in Materials Splits

The most consequential methodological error in materials ML is splitting a correlated dataset as if it were i.i.d. Types of leakage:

- **Prototype leakage:** Structures with the same Wyckoff topology but different lattice constants share 90%+ of structural descriptors. Splitting by structure gives good metrics but predicts nothing about new topologies.
- **Composition leakage:** Near-identical alloy compositions (e.g., Fe60Ni40 vs Fe61Ni39) are nearly identical in descriptor space. Random splits routinely place them on opposite sides.
- **Campaign leakage:** In an experimental screening campaign, all samples from one run share substrate, atmosphere, and operator. These must be held out as a unit.
- **Temporal leakage:** In active learning, models selected in iteration N have already seen iteration N−1 compositions. Cross-validation that ignores iteration order is biased.

A minimum safe split must ensure that train and test are separated by a meaningful chemical or structural distance — typically a distinct prototype family, a distinct composition cluster, or a distinct experimental campaign.

### Splitting Strategies

**Random split:** Only valid when the dataset has been deduplicated, prototype-checked, and confirmed to be compositionally diverse. Report the random seed and the fraction; run multiple seeds and report variance.

**Stratified split:** Ensures each target-value stratum is proportionally represented in train and test. Useful for imbalanced property distributions but does not address compositional or structural correlation.

**Leave-one-group-out (LOGO-CV):** Each group (prototype family, element period, experimental batch, active-learning iteration) is held out once as the test set, and the model is trained on all others. Gives a pessimistic but honest estimate of generalization to a new group. The number of folds equals the number of groups.

**Leave-one-composition-out (LOCO-CV):** Each distinct composition is the test fold. For alloy series, this estimates performance on a nearby but unseen composition — still correlated but less so than random.

**Leave-one-material-family-out (LOMF-CV):** Each material family (e.g., spinel oxides, perovskites, FCC metals) is held out. Most honest for gauging extrapolation to new chemistry classes.

**Nested cross-validation:** Outer loop (k folds) estimates generalization; inner loop (k' folds) tunes hyperparameters and selects features. Prevents model-selection bias. The outer test fold is never used for any model decision. Essential when dataset is too small for a fixed hold-out test set.

### Baseline Models for Small Scientific Data

Always compute these before investing in a complex model:

| Model | When to use | Key hyperparameter |
|---|---|---|
| Mean predictor | Absolute baseline; defines R² = 0 | — |
| Linear ridge regression | Correlated descriptors; interpretable coefficients | α (regularization) |
| Lasso / elastic net | Sparse descriptor selection | α, l1_ratio |
| Kernel ridge regression (KRR) | Nonlinear, few samples, known kernel | α, γ (RBF) |
| Gaussian process (GPR) | Uncertainty required; calibrated intervals | Kernel, noise |
| Random forest | Mixed descriptor types; robustness to scale | n_estimators, max_depth |
| Gradient boosting (XGBoost/LightGBM) | Tabular data; handles mixed types | n_estimators, learning_rate |
| k-nearest neighbor (KNN) | No training; explainable predictions | k |

A complex model (neural network, graph network, transformer) that does not beat gradient boosting by a statistically significant margin on a small materials dataset should not be presented as the recommended approach.

### Uncertainty Quantification

**GP posterior variance:** Exact under the model; calibrate against held-out data (reliability diagram). Do not trust GP uncertainty in regions far from training data without empirical calibration.

**Ensemble variance:** Train multiple models (different random seeds, bootstrapped training sets); variance of predictions estimates epistemic uncertainty. Tends to underestimate uncertainty in deep extrapolation.

**Conformal prediction:** Provides distribution-free prediction intervals with exact marginal coverage guarantees, given exchangeable samples. Does not require a probabilistic model. For small datasets, split-conformal or cross-conformal methods. Note: coverage is marginal (average over test distribution), not conditional on input.

**Calibration check:** For 90% prediction intervals, check that ~90% of test points fall inside. Plot calibration curve: nominal coverage vs. empirical coverage. An overconfident model plots below the diagonal; underconfident plots above.

### Learning Curves and Data Sufficiency

A learning curve plots validation error vs. training set size (using subsets of training data, not the test set). Diagnostic patterns:

- **Both train and validation error high and similar:** Bias-dominated; descriptor or model is inadequate. More data will not help.
- **Large gap between train and validation error:** Variance-dominated; model overfits. More data will close the gap; alternatively, regularize or simplify.
- **Validation error has plateaued:** The model has extracted most available signal; the remaining error is irreducible noise or bias. More data may not help.
- **Validation error still decreasing at maximum training size:** More data would help; request additional calculations or experiments.

For DFT datasets, a learning curve reaching a plateau at MAE ≈ DFT numerical uncertainty (1–5 meV/atom for energies) is a strong indicator of sufficiency.

### Descriptor Selection and Scaling

Descriptor selection in small-data settings requires care to avoid leakage:

- **Always fit the scaler, PCA, or feature selector using only training data.** Apply the fitted transformer to validation and test sets. Fitting on all data leaks test-set statistics.
- **Recursive feature elimination (RFE), LASSO, or mutual information** for selection. Use nested CV to select features; the outer fold must not see the inner selection process.
- **SHAP values** for post-hoc attribution after model is selected. Do not use SHAP for feature selection inside CV loops — it is expensive and can overfit.
- **Correlation filtering:** Remove highly correlated descriptors (|ρ| > 0.95) to improve conditioning and interpretability. Do this before the split, but based only on descriptor-to-descriptor correlation (no target leakage).

## Key Workflows

### Workflow 1: Leakage-Safe Splitting with Group and Prototype Awareness

```python
# safe_splitting.py
# Build a leakage-safe train/test split for a materials dataset with
# known structure-prototype groups. Uses LOGO-CV as the outer split.

import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_absolute_error, r2_score
from pymatgen.analysis.structure_matcher import StructureMatcher

# --- Load dataset (realistic structure: CSV with structure IDs and features) ---
# Columns: formula, prototype_group, target_property, feat_0, feat_1, ...
df = pd.read_csv("dataset.csv")
print(f"Dataset: {len(df)} samples, {df['prototype_group'].nunique()} unique prototype groups")
print(f"Target range: {df['target_property'].min():.3f} – {df['target_property'].max():.3f}")
print(f"Target std: {df['target_property'].std():.3f}")

feat_cols = [c for c in df.columns if c.startswith("feat_")]
X = df[feat_cols].values
y = df["target_property"].values
groups = df["prototype_group"].values   # LOGO-CV groups

# --- Sanity check: warn if any group has very few samples ---
group_counts = pd.Series(groups).value_counts()
small_groups = group_counts[group_counts < 3]
if len(small_groups) > 0:
    print(f"WARNING: {len(small_groups)} groups with < 3 samples: {list(small_groups.index)}")
    print("  Consider merging small groups or using LOOCV instead.")

# --- Mean-predictor baseline (R² = 0 by definition, sets MAE floor) ---
y_mean = np.full_like(y, y.mean())
print(f"\nMean-predictor MAE: {mean_absolute_error(y, y_mean):.4f}")
print(f"Mean-predictor R²:  {r2_score(y, y_mean):.4f}")   # should be 0.0

# --- Leave-One-Group-Out CV with Kernel Ridge Regression ---
logo = LeaveOneGroupOut()

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model", KernelRidge(alpha=0.1, kernel="rbf", gamma=0.1)),
])

# Nested CV: outer = LOGO, inner = 5-fold for hyperparameter search
from sklearn.model_selection import GridSearchCV

param_grid = {
    "model__alpha": [0.01, 0.1, 1.0, 10.0],
    "model__gamma": [0.01, 0.1, 1.0],
}

outer_maes, outer_r2s = [], []
outer_predictions = np.full_like(y, np.nan)

for train_idx, test_idx in logo.split(X, y, groups):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    groups_train = groups[train_idx]

    # Inner CV for hyperparameter selection (on training data only)
    inner_cv = LeaveOneGroupOut()
    gs = GridSearchCV(
        pipe,
        param_grid,
        cv=inner_cv.split(X_train, y_train, groups_train),
        scoring="neg_mean_absolute_error",
        refit=True,
        n_jobs=-1,
    )
    gs.fit(X_train, y_train)

    y_pred = gs.predict(X_test)
    outer_predictions[test_idx] = y_pred
    outer_maes.append(mean_absolute_error(y_test, y_pred))
    outer_r2s.append(r2_score(y_test, y_pred))

print(f"\nLOGO nested CV results (KRR):")
print(f"  MAE:  {np.mean(outer_maes):.4f} ± {np.std(outer_maes):.4f}")
print(f"  R²:   {np.mean(outer_r2s):.4f} ± {np.std(outer_r2s):.4f}")
print(f"  Worst group MAE: {max(outer_maes):.4f}")
print(f"  Best  group MAE: {min(outer_maes):.4f}")
print(f"  Note: variance across groups reveals which chemistries extrapolate poorly.")

# --- Final model trained on full dataset (only after CV is complete) ---
# Use best hyperparams from inner CV on full data; this model is NOT validated by the above
final_gs = GridSearchCV(pipe, param_grid, cv=5, scoring="neg_mean_absolute_error")
final_gs.fit(X, y)
print(f"\nFinal model best params: {final_gs.best_params_}")
print(f"Training MAE (overestimate of performance): {-final_gs.best_score_:.4f}")
```

### Workflow 2: Baseline Model Comparison

```python
# baseline_comparison.py
# Compare multiple baseline models under the same LOGO-CV protocol.
# Never tune baselines on the test fold.

import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.metrics import mean_absolute_error, r2_score
from scipy import stats

# (Assume X, y, groups already loaded as in Workflow 1)
# For demonstration: synthetic 150-sample dataset
rng = np.random.default_rng(42)
n, d = 150, 20
X = rng.standard_normal((n, d))
y = X[:, 0] * 2.0 - X[:, 1] + 0.5 * rng.standard_normal(n)
groups = np.repeat(np.arange(15), 10)   # 15 groups of 10 each

MODELS = {
    "Mean predictor": Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsRegressor(n_neighbors=len(X), weights="uniform")),
        # Approximates mean predictor; replace with DummyRegressor(strategy='mean')
    ]),
    "Ridge (α=1)": Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=1.0)),
    ]),
    "Lasso (α=0.01)": Pipeline([
        ("scaler", StandardScaler()),
        ("model", Lasso(alpha=0.01, max_iter=5000)),
    ]),
    "Kernel Ridge (RBF)": Pipeline([
        ("scaler", StandardScaler()),
        ("model", KernelRidge(alpha=0.1, kernel="rbf", gamma=0.1)),
    ]),
    "Random Forest": Pipeline([
        ("scaler", StandardScaler()),   # RF doesn't need scaling but pipeline is uniform
        ("model", RandomForestRegressor(n_estimators=100, max_depth=5, random_state=0)),
    ]),
    "Gradient Boosting": Pipeline([
        ("scaler", StandardScaler()),
        ("model", GradientBoostingRegressor(n_estimators=100, max_depth=3,
                                             learning_rate=0.1, random_state=0)),
    ]),
    "GP (RBF + noise)": Pipeline([
        ("scaler", StandardScaler()),
        ("model", GaussianProcessRegressor(
            kernel=ConstantKernel(1.0) * RBF(1.0) + WhiteKernel(0.1),
            n_restarts_optimizer=3, alpha=1e-6,
        )),
    ]),
}

logo = LeaveOneGroupOut()
results = {}

for model_name, pipe in MODELS.items():
    maes, r2s = [], []
    for train_idx, test_idx in logo.split(X, y, groups):
        pipe.fit(X[train_idx], y[train_idx])
        y_pred = pipe.predict(X[test_idx])
        maes.append(mean_absolute_error(y[test_idx], y_pred))
        r2s.append(r2_score(y[test_idx], y_pred))
    results[model_name] = {
        "mae_mean": np.mean(maes),
        "mae_std":  np.std(maes),
        "r2_mean":  np.mean(r2s),
        "r2_std":   np.std(r2s),
        "maes":     maes,
    }

# Sort by MAE and print comparison table
print(f"\n{'Model':<25} {'MAE mean':>10} {'MAE std':>10} {'R² mean':>10}")
print("-" * 60)
for name, res in sorted(results.items(), key=lambda x: x[1]["mae_mean"]):
    print(f"{name:<25} {res['mae_mean']:>10.4f} {res['mae_std']:>10.4f} {res['r2_mean']:>10.4f}")

# Statistical significance: Wilcoxon signed-rank test vs best baseline
best_name = min(results, key=lambda x: results[x]["mae_mean"])
best_maes = results[best_name]["maes"]
print(f"\nStatistical comparison vs best ({best_name}):")
for name, res in results.items():
    if name == best_name:
        continue
    stat, p = stats.wilcoxon(res["maes"], best_maes, alternative="greater")
    sig = "p < 0.05 *" if p < 0.05 else "n.s."
    print(f"  {name:<25} vs {best_name}: p={p:.4f}  {sig}")
print("Note: p-value from Wilcoxon across LOGO folds; only valid if ≥6 groups.")
```

### Workflow 3: Learning Curves and Data Sufficiency

```python
# learning_curves.py
# Generate learning curves to diagnose bias vs variance and data sufficiency.
# Critical: use the same LOGO-CV protocol as the main validation.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

# (Assume X, y, groups from Workflow 1)
rng = np.random.default_rng(42)
n, d = 300, 20
X = rng.standard_normal((n, d))
y = X[:, 0] * 2.0 - X[:, 1] + 0.5 * rng.standard_normal(n)
groups = np.repeat(np.arange(20), 15)

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model", GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=0)),
])
logo = LeaveOneGroupOut()

# Training set sizes to evaluate (as fractions of available training data)
train_fracs = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
train_sizes_abs, train_maes, val_maes = [], [], []
N_SEEDS = 5   # repeat subsampling for stability

for frac in train_fracs:
    fold_train_maes = []
    fold_val_maes_frac = []
    for train_idx, test_idx in logo.split(X, y, groups):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_te, y_te = X[test_idx], y[test_idx]

        seed_val_maes = []
        seed_train_maes = []
        for seed in range(N_SEEDS):
            rng_sub = np.random.default_rng(seed)
            n_sub = max(5, int(frac * len(X_tr)))
            sub_idx = rng_sub.choice(len(X_tr), n_sub, replace=False)
            pipe.fit(X_tr[sub_idx], y_tr[sub_idx])
            seed_val_maes.append(mean_absolute_error(y_te, pipe.predict(X_te)))
            seed_train_maes.append(mean_absolute_error(y_tr[sub_idx], pipe.predict(X_tr[sub_idx])))
        fold_val_maes_frac.append(np.mean(seed_val_maes))
        fold_train_maes.append(np.mean(seed_train_maes))

    train_sizes_abs.append(int(frac * len(X) * (1 - 1 / logo.get_n_splits(X, y, groups))))
    train_maes.append(np.mean(fold_train_maes))
    val_maes.append(np.mean(fold_val_maes_frac))
    print(f"  Frac={frac:.1f}: n_train≈{train_sizes_abs[-1]}, "
          f"train_MAE={train_maes[-1]:.4f}, val_MAE={val_maes[-1]:.4f}")

# Diagnose
gap = np.array(val_maes) - np.array(train_maes)
print(f"\nLearning curve diagnosis:")
if gap[-1] > 2 * train_maes[-1]:
    print("  HIGH VARIANCE: large train/val gap at full data. Consider regularization.")
elif val_maes[-1] / val_maes[0] > 0.9:
    print("  HIGH BIAS: validation error barely decreases with more data. "
          "Improve descriptor or model complexity.")
elif abs(val_maes[-1] - val_maes[-2]) / val_maes[-1] < 0.02:
    print("  SATURATED: adding more data gives < 2% improvement. Current data may be sufficient.")
else:
    print("  STILL IMPROVING: more data would help; consider acquiring more.")

# Plot
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(train_sizes_abs, train_maes, "o-", color="steelblue", label="Train MAE")
ax.plot(train_sizes_abs, val_maes, "s--", color="firebrick", label="Validation MAE (LOGO)")
ax.fill_between(train_sizes_abs, val_maes, train_maes, alpha=0.08, color="gray",
                label="Bias-variance gap")
ax.set_xlabel("Training set size")
ax.set_ylabel("MAE")
ax.set_title("Learning curve")
ax.legend()
plt.savefig("learning_curve.png", dpi=150, bbox_inches="tight")
```

### Workflow 4: Uncertainty Calibration and Conformal Prediction

```python
# uncertainty_calibration.py
# Calibrate uncertainty estimates from a GP or ensemble model.
# Produce a reliability diagram and conformal prediction intervals.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel

# (Assume X, y, groups from prior workflows)
rng = np.random.default_rng(0)
n, d = 200, 10
X = rng.standard_normal((n, d))
y = np.sin(X[:, 0]) + 0.3 * rng.standard_normal(n)
groups = np.repeat(np.arange(20), 10)

gp_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("gp", GaussianProcessRegressor(
        kernel=Matern(nu=2.5) + WhiteKernel(noise_level=0.1),
        n_restarts_optimizer=3,
        alpha=1e-6,
    )),
])

logo = LeaveOneGroupOut()
all_y_true, all_y_pred, all_y_std = [], [], []

for train_idx, test_idx in logo.split(X, y, groups):
    gp_pipe.fit(X[train_idx], y[train_idx])
    # Access the GP step directly for uncertainty
    X_test_scaled = gp_pipe.named_steps["scaler"].transform(X[test_idx])
    y_pred, y_std = gp_pipe.named_steps["gp"].predict(X_test_scaled, return_std=True)
    all_y_true.extend(y[test_idx])
    all_y_pred.extend(y_pred)
    all_y_std.extend(y_std)

y_true = np.array(all_y_true)
y_pred = np.array(all_y_pred)
y_std  = np.array(all_y_std)
residuals = y_true - y_pred
z_scores = residuals / y_std   # standardized residuals; should be ~N(0,1) if well-calibrated

# --- Reliability diagram ---
nominal_coverages = np.arange(0.05, 1.0, 0.05)
empirical_coverages = []
for p in nominal_coverages:
    z_crit = np.percentile(np.abs(np.random.standard_normal(100000)), 100 * p)
    empirical = np.mean(np.abs(z_scores) <= z_crit)
    empirical_coverages.append(empirical)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

ax = axes[0]
ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
ax.plot(nominal_coverages, empirical_coverages, "o-", color="steelblue", label="GP")
overconf = np.mean(np.array(empirical_coverages) < np.array(nominal_coverages))
ax.set_xlabel("Nominal coverage")
ax.set_ylabel("Empirical coverage")
ax.set_title(f"Calibration ({overconf:.0%} of intervals overconfident)")
ax.legend()

# --- Standardized residual distribution ---
ax = axes[1]
ax.hist(z_scores, bins=30, density=True, color="steelblue", alpha=0.7, label="Standardized residuals")
xg = np.linspace(-4, 4, 200)
ax.plot(xg, np.exp(-0.5 * xg**2) / np.sqrt(2 * np.pi), "r-", lw=1.5, label="N(0,1)")
ax.set_xlabel("Standardized residual (z-score)")
ax.set_ylabel("Density")
ax.set_title(f"GP uncertainty calibration\n(mean |z| = {np.mean(np.abs(z_scores)):.2f}; ideal = 0.80)")
ax.legend()
plt.tight_layout()
plt.savefig("calibration.png", dpi=150, bbox_inches="tight")

# --- Split-conformal prediction intervals ---
# Non-conformity scores on calibration fold, then apply to test fold
# Simple split-conformal: split training into proper-train and calibration
n_cal = len(y_true) // 5
cal_idx = np.arange(n_cal)
proper_idx = np.arange(n_cal, len(y_true))

# Re-fit on proper-train, compute nonconformity on calibration
nonconf_scores = np.abs(residuals[cal_idx])   # |y_true - y_pred| on calibration set

for alpha in [0.1, 0.2]:
    q_hat = np.quantile(nonconf_scores, 1 - alpha)
    # Conformal interval: [y_pred - q_hat, y_pred + q_hat]
    in_interval = np.abs(residuals[proper_idx]) <= q_hat
    print(f"Conformal interval (α={alpha}): q_hat={q_hat:.4f}, "
          f"coverage={in_interval.mean():.3f} (target={1-alpha:.2f})")
print("Conformal intervals have guaranteed marginal coverage; GP intervals require calibration.")
```

### Workflow 5: Descriptor Selection with Leakage Protection

```python
# descriptor_selection.py
# Feature selection in a small-data regime, fully contained within CV folds.
# Never fit the selector on full data before splitting.

import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV, Ridge
from sklearn.feature_selection import SelectFromModel, mutual_info_regression
from sklearn.ensemble import RandomForestRegressor
import shap

# (Assume X_raw = (n, n_feat) with many descriptors, some correlated/irrelevant)
rng = np.random.default_rng(1)
n, n_feat = 180, 50
X_raw = rng.standard_normal((n, n_feat))
# True model: only features 0, 3, 7 matter
y = X_raw[:, 0] * 2.0 + X_raw[:, 3] * -1.5 + X_raw[:, 7] * 0.8 + 0.3 * rng.standard_normal(n)
groups = np.repeat(np.arange(12), 15)

# --- Step 1: Remove highly correlated descriptors (descriptor-to-descriptor only, no target) ---
corr = np.corrcoef(X_raw.T)
to_drop = set()
for i in range(n_feat):
    for j in range(i + 1, n_feat):
        if abs(corr[i, j]) > 0.95 and j not in to_drop:
            to_drop.add(j)
X = np.delete(X_raw, list(to_drop), axis=1)
print(f"Dropped {len(to_drop)} highly correlated features; remaining: {X.shape[1]}")

# --- Step 2: LOGO-CV with in-fold Lasso feature selection ---
logo = LeaveOneGroupOut()
selected_feature_counts = np.zeros(X.shape[1])
fold_maes = []

for train_idx, test_idx in logo.split(X, y, groups):
    X_tr, y_tr = X[train_idx], y[train_idx]
    X_te, y_te = X[test_idx], y[test_idx]

    # Fit scaler and Lasso selector on training fold only
    scaler = StandardScaler().fit(X_tr)
    X_tr_s = scaler.transform(X_tr)
    X_te_s = scaler.transform(X_te)

    # Lasso for feature selection (alpha tuned by inner 5-fold on training)
    lasso = LassoCV(cv=5, max_iter=10000, random_state=0).fit(X_tr_s, y_tr)
    selector = SelectFromModel(lasso, prefit=True, threshold="median")
    X_tr_sel = selector.transform(X_tr_s)
    X_te_sel = selector.transform(X_te_s)

    # Track which features are selected
    selected_feature_counts[selector.get_support()] += 1

    # Final model on selected features
    ridge = Ridge(alpha=1.0).fit(X_tr_sel, y_tr)
    fold_maes.append(np.mean(np.abs(y_te - ridge.predict(X_te_sel))))

print(f"\nLOGO-CV with in-fold Lasso selection: MAE = {np.mean(fold_maes):.4f} ± {np.std(fold_maes):.4f}")

# Report most consistently selected features
n_folds = logo.get_n_splits(X, y, groups)
freq = selected_feature_counts / n_folds
top_features = np.argsort(-freq)[:10]
print("\nTop selected features (selection frequency across folds):")
for i in top_features:
    if freq[i] > 0:
        print(f"  Feature {i:3d}: selected in {freq[i]*100:.0f}% of folds")

# --- Step 3: SHAP post-hoc attribution (on final model, NOT for selection) ---
# Train final model on ALL data (for attribution only, not for performance reporting)
scaler_all = StandardScaler().fit(X)
lasso_all = LassoCV(cv=5, max_iter=10000, random_state=0).fit(scaler_all.transform(X), y)
selector_all = SelectFromModel(lasso_all, prefit=True, threshold="median")
X_sel_all = selector_all.transform(scaler_all.transform(X))
rf = RandomForestRegressor(n_estimators=200, random_state=0).fit(X_sel_all, y)

explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_sel_all)
print(f"\nSHAP mean |shap| for selected features: {np.abs(shap_values).mean(axis=0)}")
print("NOTE: SHAP values are attribution for the FINAL model, not for model selection.")
print("      Never use SHAP to select features inside a CV loop for small datasets.")
```

### Workflow 6: Transfer Learning from Pretrained Representations

```python
# transfer_learning.py
# Use pretrained materials representations (e.g., MatVec, M3GNet embeddings,
# CHGNet site features) as frozen descriptors for small-data regression.
# This avoids training a large model on small data.

import numpy as np
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_absolute_error
from pymatgen.core import Structure

# --- Option A: Use universal MLP energy/force embeddings as descriptors ---
# CHGNet, MACE, SevenNet can produce latent-space embeddings for structures.
# These embeddings encode the local chemical environment and transfer well
# to downstream regression tasks.

def get_chgnet_embeddings(structures):
    """Extract site-averaged CHGNet embeddings for each structure."""
    try:
        from chgnet.model import CHGNet
        model = CHGNet.load()
        embeddings = []
        for struct in structures:
            graph = model.graph_converter(struct)
            pred = model.predict_structure(struct, return_site_energies=False,
                                            return_atom_feas=True)
            # pred["atom_fea"] has shape (n_atoms, embed_dim); mean-pool over atoms
            atom_fea = pred["atom_fea"]
            embeddings.append(atom_fea.mean(axis=0))
        return np.array(embeddings)
    except ImportError:
        print("CHGNet not installed; using random embeddings as placeholder")
        return np.random.standard_normal((len(structures), 64))

# --- Option B: Use composition-based descriptors from matminer (zero training needed) ---
def get_matminer_descriptors(df, structure_col="structure", composition_col="composition"):
    """
    Fast composition-level descriptor (Magpie) as a baseline for transfer.
    No model is trained; this is a fixed representation.
    """
    from matminer.featurizers.composition import ElementProperty
    from matminer.featurizers.base import MultipleFeaturizer
    featurizer = ElementProperty.from_preset("magpie")
    features = []
    for _, row in df.iterrows():
        try:
            feat = featurizer.featurize(row[composition_col])
            features.append(feat)
        except Exception:
            features.append([np.nan] * len(featurizer.feature_labels()))
    return np.array(features)

# Synthetic dataset demonstration
n_structures = 80   # small dataset
rng = np.random.default_rng(99)
X_transfer = rng.standard_normal((n_structures, 64))   # pretrained embeddings (placeholder)
y_target = X_transfer[:, 0] * 1.5 - X_transfer[:, 5] + 0.2 * rng.standard_normal(n_structures)
groups = np.repeat(np.arange(8), 10)

# Two models: Magpie (no pretraining) vs CHGNet embeddings (pretrained)
X_magpie = rng.standard_normal((n_structures, 130))   # Magpie-scale features (130 descriptors)

logo = LeaveOneGroupOut()
results = {}
for name, X_feat in [("Magpie (no pretrain)", X_magpie), ("CHGNet embeddings", X_transfer)]:
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", KernelRidge(alpha=0.1, kernel="rbf")),
    ])
    maes = []
    for tr, te in logo.split(X_feat, y_target, groups):
        pipe.fit(X_feat[tr], y_target[tr])
        maes.append(mean_absolute_error(y_target[te], pipe.predict(X_feat[te])))
    results[name] = (np.mean(maes), np.std(maes))
    print(f"  {name:<30}: MAE = {np.mean(maes):.4f} ± {np.std(maes):.4f}")

print("\nNotes on transfer learning for small materials datasets:")
print("  - Pretrained embeddings (CHGNet, MACE, M3GNet) often outperform Magpie on < 200 samples")
print("  - Always compare to Magpie baseline; if gap < 1σ, pretrain adds no value for this target")
print("  - Fine-tuning the full pretrained model on < 100 samples is almost always counterproductive")
print("  - Report whether embeddings were frozen or fine-tuned, and at which layer")
```

### Workflow 7: Rigorous Reporting — Confidence Intervals and Negative Results

```python
# rigorous_reporting.py
# Compute bootstrap confidence intervals for metrics, compare models properly,
# and detect common inflation artifacts before publishing.

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from scipy import stats

def bootstrap_ci(y_true, y_pred, metric_fn=mean_absolute_error,
                  n_bootstrap=2000, ci=0.95, seed=0):
    """
    Bootstrap confidence interval for a scalar metric.
    All bootstrapping is on the held-out predictions, not on the model.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    boot_stats = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot_stats.append(metric_fn(y_true[idx], y_pred[idx]))
    lower = np.percentile(boot_stats, 100 * (1 - ci) / 2)
    upper = np.percentile(boot_stats, 100 * (1 + ci) / 2)
    return np.mean(boot_stats), lower, upper

def check_inflation_artifacts(y_true, y_pred, groups=None):
    """
    Detect common sources of inflated R² and misleadingly low MAE.
    """
    issues = []

    # 1. Narrow target range: R² can be high even with large absolute errors
    target_range = y_true.max() - y_true.min()
    target_std = y_true.std()
    if target_std < 0.5:   # property-dependent threshold; adjust per domain
        issues.append(
            f"NARROW TARGET RANGE: std={target_std:.3f}, range={target_range:.3f}. "
            "R² near 1.0 does not imply practical utility."
        )

    # 2. R² > 0.99 on a small dataset without leave-out groups: suspect leakage
    r2 = r2_score(y_true, y_pred)
    n = len(y_true)
    if r2 > 0.99 and n < 300:
        issues.append(
            f"SUSPICIOUSLY HIGH R²={r2:.4f} on n={n}. "
            "Verify no leakage through group-aware CV."
        )

    # 3. MAE ≈ 0 on DFT data: may indicate train-test overlap
    mae = mean_absolute_error(y_true, y_pred)
    if mae < 1e-4 and n < 500:
        issues.append(
            f"NEAR-ZERO MAE={mae:.2e}. "
            "Check for train-test data leakage or duplicate structures."
        )

    # 4. Mean-predictor comparison
    mae_mean = mean_absolute_error(y_true, np.full_like(y_true, y_true.mean()))
    skill_score = 1 - mae / mae_mean
    if skill_score < 0.1:
        issues.append(
            f"MODEL BARELY BEATS MEAN PREDICTOR: skill score={skill_score:.3f}. "
            "The model captures only {skill_score*100:.0f}% of reducible error."
        )

    if not issues:
        print("Inflation checks: no obvious artifacts detected.")
    else:
        print(f"Inflation checks: {len(issues)} potential issue(s):")
        for iss in issues:
            print(f"  WARNING: {iss}")

    return issues

# Example usage
rng = np.random.default_rng(42)
n = 150
y_true = rng.standard_normal(n) * 0.8 + 3.0   # narrow range
y_pred = y_true + 0.1 * rng.standard_normal(n)  # almost perfect (leakage simulation)

print("--- Artifact checks ---")
check_inflation_artifacts(y_true, y_pred)

mae_est, mae_lo, mae_hi = bootstrap_ci(y_true, y_pred, mean_absolute_error)
r2_est, r2_lo, r2_hi   = bootstrap_ci(y_true, y_pred, r2_score)

print(f"\n--- Metrics with 95% bootstrap CI ---")
print(f"  MAE: {mae_est:.4f}  [{mae_lo:.4f}, {mae_hi:.4f}]")
print(f"  R²:  {r2_est:.4f}  [{r2_lo:.4f}, {r2_hi:.4f}]")

# Report template
print("\n--- Suggested reporting statement ---")
print(f"  Using nested LOGO-CV with {n} samples across N groups,")
print(f"  the model achieved MAE = {mae_est:.3f} (95% CI: {mae_lo:.3f}–{mae_hi:.3f}),")
print(f"  compared to a mean-predictor baseline MAE of {mean_absolute_error(y_true, np.full_like(y_true, y_true.mean())):.3f}.")
print(f"  This corresponds to a skill score of {1 - mae_est/mean_absolute_error(y_true, np.full_like(y_true, y_true.mean())):.2f}.")
print(f"  Random search baseline and learning curve are provided in the supplement.")
```

## Common Failure Modes

| Failure Mode | Symptoms | Likely Cause | Diagnostic | Fix |
|---|---|---|---|---|
| Random split over correlated structures | R² near 1.0 on test; model fails on new chemistry | Near-duplicate structures in train and test; prototype or composition correlation | Switch to LOGO-CV; check R² drop; plot pairwise descriptor distances between train/test | Use LOGO-CV or LOCO-CV; deduplicate before splitting |
| Hidden duplicates | Test MAE near zero; model perfectly "predicts" test structures | Same structure appears under different IDs or slightly different cells | Run `StructureMatcher` on full dataset before splitting | Deduplicate with `pymatgen.analysis.structure_matcher` |
| Descriptor leakage | Validation R² inflated; model uses information unavailable at prediction time | Scaler, PCA, or feature selector fit on all data before splitting | Check: is any transformation fitted before the split? | Wrap all transformations in a `Pipeline`; fit only inside training fold |
| Target leakage | Near-perfect validation; feature correlated with target for non-causal reasons | A feature encodes the target (e.g., using experimental band gap to predict DFT band gap) | Correlate each feature with target; check feature definitions | Remove leaking features; audit feature definitions against the prediction task |
| Over-tuned hyperparameters | Validation MAE low; independent test MAE much higher | Hyperparameters tuned on the validation fold; same fold used for selection and reporting | Check: was the test fold touched during tuning? | Nested CV; treat test fold as final and untouched until reporting |
| High R² from narrow target range | R² = 0.97, but model is barely better than mean predictor | Target has low variance; R² amplifies small improvements in low-variance settings | Check skill score vs mean predictor; inspect target distribution | Report MAE, skill score, and target std alongside R²; do not rely on R² alone |
| Extrapolation without acknowledgment | Model applied to new chemistry family with confident predictions; physical nonsense results | No held-out samples from the target chemistry; validation only covers interpolation | Compute descriptor-domain coverage; check distances of new points from training hull | Flag all out-of-domain predictions; use GP or conformal intervals that widen in extrapolation |
| Confusing statistical accuracy with physical validity | Model passes CV metrics but predicts negative band gaps, wrong phase stabilities, or violated sum rules | CV metrics only measure statistical fit, not physical correctness | Add physical sanity checks: bounds, signs, limiting cases, known constraints | Implement domain-specific physical validation checks alongside statistical metrics |
| Cherry-picked metric or split | Paper reports best metric across many seeds, splits, or metrics tried | Implicit multiple comparisons; only the best result reported | Count how many splits, seeds, and metrics were tried before reporting | Pre-register the evaluation protocol; report mean ± std across seeds; correct for multiple comparisons |
| Learning curve not computed | Data sufficiency not established; possible to collect more data cheaply | Effort focused on model tuning rather than data quality assessment | Generate learning curve; check if error has plateaued | Run learning curve on training subsets; if not saturated, collect more data before tuning model |

## Best Practices

**Data before modeling:**
- Deduplicate structures using `StructureMatcher` before any split. A dataset with hidden duplicates will give inflated metrics regardless of the model.
- Audit target provenance: mixed DFT functionals, different convergence settings, or mixed experimental conditions produce a multi-modal target distribution that single-model regression cannot handle.
- Check target distribution: is it unimodal? Are there outliers? What is the physical range? Log-transform if the target spans orders of magnitude (e.g., ionic conductivity, catalytic rate).

**Splitting:**
- Default to group-aware splitting (LOGO-CV or LOCO-CV) for any materials dataset. Random splits are valid only when the dataset is verified to be compositionally and structurally diverse with no cross-split correlations.
- Assign groups before looking at the target values. Do not choose groups post-hoc to produce a desired result.
- The test fold must not be used for any model decision. Nested CV enforces this automatically.

**Baselines:**
- Always report mean predictor, ridge regression, and random forest alongside any complex model. The complex model must beat all three by a statistically significant margin (Wilcoxon, n_folds ≥ 6) to be considered an advance.
- Run multiple random seeds for any random component (forest, bootstrap, random initialization) and report mean ± std.

**Metrics:**
- Report MAE (or RMSE) alongside R². R² alone is uninterpretable without knowing the target variance.
- Report skill score: 1 − MAE / MAE_mean_predictor. Values < 0.2 indicate the model is barely useful.
- Report 95% bootstrap CIs for all metrics. A difference of MAE ± 0.05 vs 0.04 is not meaningful without intervals.
- For uncertainty-aware models (GP, ensemble), report calibration coverage (e.g., "90% of test points fall within the 90% predictive interval").

**Reporting negative results:**
- If a complex model does not beat a simple baseline, report it. Negative results prevent repeated effort in the community.
- If the learning curve shows that the model is bias-limited (more data will not help), report the descriptor or model limitation.
- If the model fails physical sanity checks despite acceptable MAE, report both the metric and the failure. Statistical accuracy and physical validity are independent.

## Integration with Other Skills

**Featurization and data sources:**

- [Materials Featurization](../../scientific-data/materials-featurization/SKILL.md) — descriptor pipelines (Magpie, SOAP, MBTR, graph fingerprints) that feed the feature matrix for all models in this skill. Feature selection, correlation filtering, and scaling must be applied after splitting, not before.
- [Materials Databases Access](../../scientific-data/materials-databases-access/SKILL.md) — small datasets are often derived from Materials Project, AFLOW, or OQMD queries. Deduplication, compatibility checking, and provenance metadata from that skill are prerequisites for safe splitting.

**Modeling and uncertainty:**

- [Gaussian Process Materials](../../surrogate-active-learning/gaussian-process-materials/SKILL.md) — GP regression is the preferred baseline model for small materials datasets where calibrated uncertainty is needed. Kernel selection, noise modeling, and calibration workflows from the GP skill are used directly in Workflows 4 and 6.
- [Bayesian Optimization](../../surrogate-active-learning/bayesian-optimization/SKILL.md) — BO uses the small-data surrogate as its model; the validation protocol here determines whether the surrogate's uncertainty is trustworthy enough to use as an acquisition signal. Always validate the surrogate before deploying it in a BO loop.
- [Surrogate Validation](../../surrogate-active-learning/surrogate-validation/SKILL.md) — all validation concepts in this skill (LOGO-CV, calibration, learning curves, baseline comparison) directly feed into the surrogate validation skill. The surrogate validation skill applies these in the specific context of screening and optimization loops.
- [Uncertainty MLP](../../ml-interatomic-potentials/uncertainty-mlp/SKILL.md) — ensemble-based and evidential uncertainty quantification for neural MLP surrogates follow the same calibration and reliability-diagram protocol introduced here for GPs and simple ensembles.

**Data generation and active learning:**

- [High-Throughput DFT](../../electronic-structure/high-throughput-dft/SKILL.md) — small materials datasets often grow through iterative HT-DFT campaigns. The learning curve workflow here determines when to stop collecting data; the convergence tier policy there determines data quality.
- [Active Learning MLP](../../ml-interatomic-potentials/active-learning-mlp/SKILL.md) — active learning exploits the surrogate model built under this skill's protocol. Iteration-aware splits (never mixing training and future query iterations in validation) are essential for honest evaluation of closed-loop campaigns.

**Visualization and reproducibility:**

- [Scientific Visualization](../../scientific-data/scientific-visualization/SKILL.md) — parity plots, residual plots, calibration curves, learning curves, and descriptor coverage maps are the standard outputs of small-data model evaluation; this skill provides the plot conventions and multi-panel figure standards.
- [Reproducibility Workflow](../../scientific-data/reproducibility-workflow/SKILL.md) — small-data ML studies require archiving the exact dataset (with IDs and provenance), the train/val/test split indices or group assignments, the fitted scaler and selector, and the CV results. Without this, results cannot be reproduced or audited.

## Key References

1. Meredig, B., Agrawal, A., Kirklin, S., Saal, J. E., Doak, J. W., Thompson, A., Zhang, K., Choudhary, A. & Wolverton, C. (2014). "Combinatorial screening for new materials in unconstrained composition space with machine learning." *Physical Review B* 89, 094104. Early materials ML benchmark illustrating composition-based descriptors and dataset construction.
2. Ramprasad, R., Batra, R., Pilania, G., Mannodi-Kanakkithodi, A. & Kim, C. (2017). "Machine learning in materials informatics: Recent applications and prospects." *npj Computational Materials* 3, 54. Review of descriptor families, model selection, and validation practices in materials ML.
3. Schutt, K. T., Arbabzadah, F., Chmiela, S., Muller, K. R. & Tkatchenko, A. (2017). "Quantum-chemical insights from deep tensor neural networks." *Nature Communications* 8, 13890. Establishes neural network baselines; useful for comparing simple vs complex model performance on small molecular datasets.
4. Lones, M. A. (2021). "How to avoid machine learning pitfalls: A guide for academic researchers." *arXiv:2108.02497*. Accessible guide to leakage, overfitting, cherry-picking, and reporting bias applicable to any scientific ML setting.
5. Cawley, G. C. & Talbot, N. L. C. (2010). "On over-fitting in model selection and subsequent selection bias in performance evaluation." *Journal of Machine Learning Research* 11, 2079–2107. Foundational paper on nested cross-validation and model-selection bias; required reading for small-data ML.
6. Angelopoulos, A. N. & Bates, S. (2023). "Conformal prediction: A gentle introduction." *Foundations and Trends in Machine Learning* 16, 494–591. Tutorial on conformal prediction for distribution-free coverage-guaranteed intervals.
7. Vovk, V., Gammerman, A. & Shafer, G. (2005). *Algorithmic Learning in a Random World*. Springer. Foundational text on conformal prediction and transductive methods for small datasets.
8. Barber, R. F., Candes, E. J., Ramdas, A. & Tibshirani, R. J. (2023). "Conformal prediction beyond exchangeability." *Annals of Statistics* 51, 816–845. Extension of conformal prediction to correlated (non-exchangeable) scientific datasets.

[EXPERT REVIEW NEEDED: The dataset size thresholds in the regime table (20/100/500/2000) are based on practitioner experience and published benchmarks in materials ML; they are not universal and depend heavily on descriptor quality, target complexity, and the degree of correlation structure. Verify whether the Wilcoxon test requires ≥6 folds specifically or whether alternatives (paired t-test, sign test) are preferable for very few groups. Confirm whether CHGNet, MACE, and M3GNet embeddings as frozen descriptors consistently outperform Magpie on small materials datasets across multiple benchmarks, as the evidence is still accumulating.]
