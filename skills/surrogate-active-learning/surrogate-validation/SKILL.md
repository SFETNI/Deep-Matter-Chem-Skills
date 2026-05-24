---
name: surrogate-validation
domain: surrogate-active-learning
maturity: draft
tools:
  - scikit-learn
  - pandas
  - numpy
  - matplotlib
  - scipy
  - pymatgen
tested_versions:
  scikit-learn: "1.4.0"
  pandas: "2.2.0"
  numpy: "1.26.4"
  matplotlib: "3.8.0"
  scipy: "1.12.0"
  pymatgen: "2024.11.13"
requires_gpu: false
estimated_runtime: minutes-to-hours
---

# Surrogate Model Validation

## Description

This skill covers validation of surrogate models for materials and chemistry: leakage-safe train/validation/test splits, nested cross-validation, leave-one-group-out validation, uncertainty calibration, parity and residual diagnostics, interpolation versus extrapolation checks, descriptor-domain coverage, learning curves, baseline and ablation studies, distribution-shift testing, and validation of Bayesian optimization or active learning loops. Invoke this skill before trusting a surrogate model for screening, optimization, scientific claims, or automated data acquisition.

## Domain Context

A surrogate model is only useful if its validation protocol matches the way it will be used. A random split over a materials database may look strong because near-duplicate crystal prototypes, adjacent compositions, or repeated DFT tasks appear in both train and test sets. The same model may fail when asked to extrapolate to a new chemistry family, defect type, surface termination, or higher-fidelity calculation. Validation is therefore a statement about a deployment scenario, not a single score.

Materials and chemistry datasets are especially vulnerable to correlation leakage. Structures with the same prototype but different lattice constants can share almost identical descriptors. Consecutive active-learning iterations can contain many near-duplicate candidates. Database-derived records can duplicate the same material under several IDs. Random train/test splits treat these as independent, inflating R2 and deflating MAE.

Uncertainty validation is separate from point-error validation. A model can have acceptable MAE but be overconfident in extrapolation regions, making it dangerous for Bayesian optimization or active learning. Conversely, a conservative model may have wide intervals that cover most held-out labels but are too broad to guide decisions. Reliability diagrams, calibration curves, conformal intervals, and domain-shift tests are needed before using uncertainty as a query signal.

Physical sanity checks complement statistical metrics. Surrogate predictions should respect known signs, bounds, monotonicities, stability relations, and unit conventions where those constraints apply. A model predicting negative elastic moduli for stable crystals, impossible formation energies, or a band gap for a metal-like structure may pass aggregate test metrics but fail scientific usefulness.

## When to Use This Skill

- Validating a surrogate model before using it for screening, Bayesian optimization, active learning, or publication-quality claims.
- Designing train/validation/test splits for small, correlated materials datasets.
- Auditing uncertainty calibration for Gaussian processes, ensembles, conformal predictors, or probabilistic neural networks.
- Comparing descriptor sets, kernels, baselines, or model classes through ablations and nested cross-validation.
- Testing whether a surrogate is interpolating inside descriptor-domain coverage or extrapolating to new chemistry or structure classes.
- Validating closed-loop optimization or active-learning campaigns using retrospective simulations and negative-result tracking.

## When Not to Use This Skill

- **As a replacement for physical reference calculations.** Validation diagnoses a surrogate; it does not create new ground truth.
- **When the target definition is inconsistent.** If labels mix incompatible DFT settings, experimental conditions, or correction schemes, validate the data provenance first.
- **When only training error is available.** Training metrics are not validation. Hold out data before model selection or use nested cross-validation.
- **When the deployment domain is unknown.** Validation needs a concrete target domain: same chemistry interpolation, new composition extrapolation, new structure class, or closed-loop acquisition.
- **When negative results are excluded.** Screening campaigns that only validate on promising candidates overestimate performance on the full candidate pool.

## Core Concepts

- **Data leakage:** Any information from validation/test data that influences preprocessing, feature selection, hyperparameters, model choice, or training.
- **Correlation leakage:** A materials-specific leakage mode where near-duplicate structures, same-prototype compounds, or trajectory-neighbor frames appear in both train and test sets.
- **Nested cross-validation:** Outer loop estimates generalization; inner loop tunes hyperparameters and selects models. Prevents model-selection bias in small datasets.
- **Leave-one-group-out validation:** Train on all groups except one and test on the held-out group. Groups may be composition families, crystal prototypes, database sources, active-learning iterations, or structure clusters.
- **Reliability diagram:** Plot of predicted confidence or interval width versus observed error or coverage. Used to diagnose overconfidence or underconfidence.
- **Interpolation vs. extrapolation:** A prediction is interpolative if the point lies inside the training descriptor-domain coverage; extrapolative if it is far from all training points or outside the chemistry/structure groups represented in training.
- **Descriptor-domain coverage:** Quantitative description of where training data live in feature space, often using nearest-neighbor distances, convex hull membership, PCA/UMAP maps, or density estimates.
- **Learning curve:** Error versus training-set size. It reveals whether more data are likely to improve the model or whether descriptor/model bias dominates.
- **Baseline model:** Simple reference such as mean predictor, composition-only random forest, linear ridge regression, k-nearest neighbors, or literature heuristic. A surrogate must beat relevant baselines.
- **Physical sanity check:** Domain-specific test of predictions against known constraints, bounds, signs, units, or limiting cases.

## Key Workflows

### Workflow 1: Leakage-safe split design for small scientific datasets

Start validation by defining the split. Random splits are useful only for interpolation within the same distribution; they are not a default for materials discovery claims.

```python
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold, LeaveOneGroupOut
from sklearn.metrics import mean_absolute_error, r2_score


def assign_validation_groups(
    df: pd.DataFrame,
    formula_col: str = "formula",
    prototype_col: str | None = "prototype",
    source_col: str | None = "source",
) -> pd.DataFrame:
    """
    Add conservative validation groups for materials datasets.

    Priority:
      1. explicit prototype if available
      2. reduced formula family if prototype is absent
      3. source database if validating source transfer
    """
    out = df.copy()
    if prototype_col and prototype_col in out.columns:
        out["validation_group"] = out[prototype_col].fillna("unknown_prototype").astype(str)
    elif formula_col in out.columns:
        # Crude family grouping: sorted element symbols, e.g. Li-Fe-O.
        from pymatgen.core import Composition
        def family(formula):
            try:
                comp = Composition(formula)
                return "-".join(sorted(str(el) for el in comp.elements))
            except Exception:
                return "unknown_formula"
        out["validation_group"] = out[formula_col].map(family)
    elif source_col and source_col in out.columns:
        out["validation_group"] = out[source_col].astype(str)
    else:
        raise ValueError("Need prototype, formula, or source column to create groups")
    return out


def evaluate_grouped_cv(
    model_pipeline,
    df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    group_column: str = "validation_group",
    output_csv: str = "grouped_cv_predictions.csv",
    max_splits: int = 8,
) -> pd.DataFrame:
    """
    Evaluate a model with group-aware CV.

    model_pipeline must include preprocessing so each fold fits scalers,
    imputers, and feature selectors on training data only.
    """
    df = df.dropna(subset=feature_columns + [target_column, group_column]).copy()
    groups = df[group_column].to_numpy()
    n_groups = len(np.unique(groups))
    if n_groups < 2:
        raise ValueError("Need at least two validation groups")

    splitter = GroupKFold(n_splits=min(max_splits, n_groups))
    rows = []
    for fold, (train_idx, test_idx) in enumerate(splitter.split(df[feature_columns], df[target_column], groups), 1):
        train = df.iloc[train_idx]
        test = df.iloc[test_idx]
        model_pipeline.fit(train[feature_columns], train[target_column])
        pred = model_pipeline.predict(test[feature_columns])
        for idx, y, yhat, group in zip(test.index, test[target_column], pred, test[group_column]):
            rows.append({
                "fold": fold,
                "row_index": int(idx),
                "validation_group": group,
                "y_true": float(y),
                "y_pred": float(yhat),
                "abs_error": float(abs(yhat - y)),
            })

    pred_df = pd.DataFrame(rows)
    pred_df.to_csv(output_csv, index=False)
    print(f"Grouped CV MAE: {mean_absolute_error(pred_df['y_true'], pred_df['y_pred']):.4f}")
    print(f"Grouped CV R2:  {r2_score(pred_df['y_true'], pred_df['y_pred']):.4f}")
    return pred_df
```

### Workflow 2: Nested cross-validation for model and descriptor selection

Use nested CV when model class, descriptors, kernels, or hyperparameters are being chosen. Reporting the best inner-CV score as the final score is optimistic.

```python
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error


def nested_group_cv(
    estimator,
    param_grid: dict,
    df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    group_column: str,
    output_json: str,
    outer_splits: int = 5,
    inner_splits: int = 3,
) -> dict:
    """
    Nested group CV for small materials datasets.

    estimator should be a Pipeline that includes preprocessing.
    """
    df = df.dropna(subset=feature_columns + [target_column, group_column]).copy()
    X = df[feature_columns]
    y = df[target_column].to_numpy()
    groups = df[group_column].to_numpy()
    n_groups = len(np.unique(groups))
    outer = GroupKFold(n_splits=min(outer_splits, n_groups))

    fold_reports = []
    all_true, all_pred = [], []
    for fold, (train_idx, test_idx) in enumerate(outer.split(X, y, groups), 1):
        inner_groups = groups[train_idx]
        inner = GroupKFold(n_splits=min(inner_splits, len(np.unique(inner_groups))))
        search = GridSearchCV(
            estimator,
            param_grid,
            scoring="neg_mean_absolute_error",
            cv=inner,
            n_jobs=1,
            refit=True,
        )
        search.fit(X.iloc[train_idx], y[train_idx], groups=inner_groups)
        pred = search.predict(X.iloc[test_idx])
        mae = mean_absolute_error(y[test_idx], pred)
        rmse = mean_squared_error(y[test_idx], pred, squared=False)
        all_true.extend(y[test_idx])
        all_pred.extend(pred)
        fold_reports.append({
            "fold": fold,
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "heldout_groups": sorted(set(groups[test_idx])),
            "best_params": search.best_params_,
            "inner_best_mae": float(-search.best_score_),
            "outer_mae": float(mae),
            "outer_rmse": float(rmse),
        })

    report = {
        "outer_mae_mean": float(np.mean([r["outer_mae"] for r in fold_reports])),
        "outer_mae_std": float(np.std([r["outer_mae"] for r in fold_reports])),
        "pooled_mae": float(mean_absolute_error(all_true, all_pred)),
        "folds": fold_reports,
    }
    Path(output_json).write_text(json.dumps(report, indent=2, default=str))
    return report
```

### Workflow 3: Uncertainty calibration and reliability diagrams

For probabilistic surrogates, validate intervals. A calibrated 90% interval should contain about 90% of held-out labels under the validation distribution.

```python
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def calibration_table(
    predictions_csv: str,
    output_prefix: str,
    y_true_col: str = "y_true",
    y_pred_col: str = "y_pred",
    y_std_col: str = "y_std",
) -> pd.DataFrame:
    """
    Compute coverage of Gaussian predictive intervals.

    predictions_csv should contain out-of-fold predictions only. In-sample
    posterior uncertainty is not a valid calibration estimate.
    """
    df = pd.read_csv(predictions_csv).dropna(subset=[y_true_col, y_pred_col, y_std_col])
    err = np.abs(df[y_pred_col].to_numpy() - df[y_true_col].to_numpy())
    sigma = np.maximum(df[y_std_col].to_numpy(), 1e-12)

    z_values = np.array([0.5, 0.674, 1.0, 1.282, 1.645, 1.96, 2.576])
    nominal = np.array([0.383, 0.50, 0.683, 0.80, 0.90, 0.95, 0.99])
    observed = np.array([(err <= z * sigma).mean() for z in z_values])
    table = pd.DataFrame({
        "z": z_values,
        "nominal_coverage": nominal,
        "observed_coverage": observed,
        "coverage_error": observed - nominal,
    })
    table.to_csv(f"{output_prefix}_calibration.csv", index=False)

    fig, ax = plt.subplots(figsize=(3.2, 3.2))
    ax.plot(nominal, observed, "o-", color="#1f77b4")
    ax.plot([0, 1], [0, 1], "--", color="0.4", lw=0.8)
    ax.set_xlabel("Nominal coverage")
    ax.set_ylabel("Observed coverage")
    ax.set_title("Reliability diagram")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(f"{output_prefix}_reliability.png", dpi=200)
    plt.close(fig)

    undercoverage = table.loc[table["nominal_coverage"] == 0.90, "observed_coverage"].iloc[0] < 0.85
    if undercoverage:
        print("WARNING: 90% intervals under-cover held-out labels; uncertainty is overconfident")
    return table


def conformal_interval_width(
    calibration_csv: str,
    alpha: float = 0.10,
    y_true_col: str = "y_true",
    y_pred_col: str = "y_pred",
    y_std_col: str = "y_std",
) -> float:
    """
    Compute a scalar conformal multiplier for normalized residuals.

    Assumes calibration points are exchangeable with future predictions.
    This assumption is often violated under active learning distribution shift.
    [EXPERT REVIEW NEEDED]
    """
    df = pd.read_csv(calibration_csv).dropna(subset=[y_true_col, y_pred_col, y_std_col])
    scores = np.abs(df[y_true_col] - df[y_pred_col]) / np.maximum(df[y_std_col], 1e-12)
    n = len(scores)
    q_idx = int(np.ceil((n + 1) * (1 - alpha))) - 1
    q_idx = min(max(q_idx, 0), n - 1)
    multiplier = float(np.sort(scores)[q_idx])
    print(f"Conformal multiplier for {(1-alpha):.0%} intervals: {multiplier:.3f}")
    return multiplier
```

### Workflow 4: Parity, residual, and leverage diagnostics

Aggregate metrics hide systematic errors. Always pair MAE/RMSE with residual plots, error distributions, group breakdowns, and leverage or distance-to-training diagnostics.

```python
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_surrogate_diagnostics(
    predictions_csv: str,
    output_file: str,
    group_col: str | None = "validation_group",
    distance_col: str | None = "nearest_train_distance",
) -> None:
    """
    Make a four-panel validation figure:
    parity, residual vs predicted, error distribution, and error vs distance.
    """
    df = pd.read_csv(predictions_csv)
    df["residual"] = df["y_pred"] - df["y_true"]
    df["abs_error"] = df["residual"].abs()

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.6))
    lo = float(np.nanmin([df["y_true"].min(), df["y_pred"].min()]))
    hi = float(np.nanmax([df["y_true"].max(), df["y_pred"].max()]))

    ax = axes[0, 0]
    ax.scatter(df["y_true"], df["y_pred"], s=16, alpha=0.7)
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)
    ax.set_xlabel("Reference")
    ax.set_ylabel("Prediction")
    ax.set_title("Parity")
    ax.set_aspect("equal", adjustable="box")

    ax = axes[0, 1]
    ax.scatter(df["y_pred"], df["residual"], s=16, alpha=0.7)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("Prediction")
    ax.set_ylabel("Prediction - reference")
    ax.set_title("Residuals")

    ax = axes[1, 0]
    ax.hist(df["residual"], bins=30, color="#4c78a8", alpha=0.85)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("Residual")
    ax.set_ylabel("Count")
    ax.set_title("Error distribution")

    ax = axes[1, 1]
    if distance_col and distance_col in df.columns:
        ax.scatter(df[distance_col], df["abs_error"], s=16, alpha=0.7)
        ax.set_xlabel("Nearest train distance")
        ax.set_ylabel("Absolute error")
        ax.set_title("Extrapolation diagnostic")
    elif group_col and group_col in df.columns:
        grouped = df.groupby(group_col)["abs_error"].mean().sort_values(ascending=False).head(12)
        ax.bar(range(len(grouped)), grouped.values)
        ax.set_xticks(range(len(grouped)))
        ax.set_xticklabels(grouped.index.astype(str), rotation=45, ha="right")
        ax.set_ylabel("Mean absolute error")
        ax.set_title("Worst groups")
    else:
        ax.text(0.5, 0.5, "No group or distance column", ha="center", va="center")
        ax.set_axis_off()

    fig.tight_layout()
    fig.savefig(output_file, dpi=200)
    plt.close(fig)
```

### Workflow 5: Descriptor-domain coverage and extrapolation labeling

A prediction may be statistically accurate in interpolation but unreliable in extrapolation. Label held-out points by distance to training data and report errors separately.

```python
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def add_descriptor_coverage_columns(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    feature_columns: list[str],
    output_csv: str,
    extrapolation_quantile: float = 0.95,
) -> pd.DataFrame:
    """
    Add nearest-neighbor distance and interpolation/extrapolation labels.

    The extrapolation threshold is estimated from leave-one-out training
    distances. This is descriptor-dependent and should be reported.
    """
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[feature_columns])
    X_eval = scaler.transform(eval_df[feature_columns])

    nn_train = NearestNeighbors(n_neighbors=2).fit(X_train)
    train_dists, _ = nn_train.kneighbors(X_train)
    loo_dists = train_dists[:, 1]
    threshold = float(np.quantile(loo_dists, extrapolation_quantile))

    nn = NearestNeighbors(n_neighbors=1).fit(X_train)
    eval_dists, eval_idx = nn.kneighbors(X_eval)
    out = eval_df.copy()
    out["nearest_train_distance"] = eval_dists[:, 0]
    out["nearest_train_index"] = eval_idx[:, 0]
    out["coverage_class"] = np.where(
        out["nearest_train_distance"] <= threshold,
        "interpolation",
        "extrapolation",
    )
    out.to_csv(output_csv, index=False)
    print(f"Coverage threshold ({extrapolation_quantile:.0%} train LOO distance): {threshold:.4f}")
    print(out["coverage_class"].value_counts())
    return out


def summarize_error_by_coverage(predictions_csv: str) -> pd.DataFrame:
    """Report error separately for interpolation and extrapolation points."""
    df = pd.read_csv(predictions_csv)
    if "coverage_class" not in df.columns:
        raise ValueError("Missing coverage_class column")
    df["abs_error"] = (df["y_pred"] - df["y_true"]).abs()
    summary = df.groupby("coverage_class").agg(
        n=("abs_error", "size"),
        mae=("abs_error", "mean"),
        p90_abs_error=("abs_error", lambda x: x.quantile(0.90)),
        max_abs_error=("abs_error", "max"),
    ).reset_index()
    print(summary)
    return summary
```

### Workflow 6: Learning curves, baselines, and ablation studies

A surrogate should beat simple baselines and show whether more data or better descriptors are needed. Learning curves should be generated with the same grouped split logic used for final validation.

```python
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error


def baseline_models() -> dict:
    return {
        "mean": DummyRegressor(strategy="mean"),
        "median": DummyRegressor(strategy="median"),
        "ridge": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("ridge", Ridge(alpha=1.0)),
        ]),
        "random_forest": RandomForestRegressor(
            n_estimators=300,
            random_state=42,
            min_samples_leaf=2,
        ),
    }


def compare_against_baselines(
    model_pipeline,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    output_json: str,
) -> dict:
    """Compare the candidate surrogate against simple baselines."""
    models = baseline_models()
    models["candidate"] = model_pipeline
    results = {}
    for name, model in models.items():
        m = clone(model)
        m.fit(train_df[feature_columns], train_df[target_column])
        pred = m.predict(test_df[feature_columns])
        results[name] = {
            "mae": float(mean_absolute_error(test_df[target_column], pred)),
            "n_train": int(len(train_df)),
            "n_test": int(len(test_df)),
        }
    Path(output_json).write_text(json.dumps(results, indent=2))
    return results


def learning_curve_by_group_split(
    model_pipeline,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    subset_sizes: list[int],
    output_csv: str,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Train on increasing subsets and evaluate on a fixed external test set.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for n in subset_sizes:
        if n > len(train_df):
            continue
        idx = rng.choice(train_df.index.to_numpy(), size=n, replace=False)
        sub = train_df.loc[idx]
        model = clone(model_pipeline)
        model.fit(sub[feature_columns], sub[target_column])
        pred = model.predict(test_df[feature_columns])
        rows.append({
            "n_train": int(n),
            "mae": float(mean_absolute_error(test_df[target_column], pred)),
        })
    out = pd.DataFrame(rows)
    out.to_csv(output_csv, index=False)
    return out
```

### Workflow 7: Validation for active-learning and Bayesian-optimization loops

Closed-loop models must be evaluated on the full candidate pool and on negative results, not only on candidates selected by the acquisition function.

```python
import json
from pathlib import Path
import pandas as pd


def audit_closed_loop_validation(
    campaign_log_csv: str,
    output_json: str,
    iteration_col: str = "iteration",
    selected_col: str = "selected_for_labeling",
    measured_col: str = "label_available",
    acquisition_col: str = "acquisition_value",
) -> dict:
    """
    Audit whether a BO/AL campaign retained enough unselected and negative
    data for honest validation.

    Each row should represent a candidate considered in an iteration.
    """
    df = pd.read_csv(campaign_log_csv)
    required = {iteration_col, selected_col, measured_col, acquisition_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    rows = []
    warnings = []
    for it, group in df.groupby(iteration_col):
        n_total = len(group)
        n_selected = int(group[selected_col].sum())
        n_measured = int(group[measured_col].sum())
        if n_selected == n_total:
            warnings.append(f"Iteration {it}: all candidates selected; no unselected control pool")
        if n_measured < n_selected:
            warnings.append(f"Iteration {it}: selected candidates missing labels")
        rows.append({
            "iteration": it,
            "n_candidates": n_total,
            "n_selected": n_selected,
            "n_measured": n_measured,
            "selected_fraction": n_selected / max(n_total, 1),
        })

    report = {
        "iterations": rows,
        "warnings": warnings,
        "validation_notes": [
            "Retain the full candidate pool and acquisition values for retrospective validation.",
            "Track failed and negative candidates; excluding them biases model assessment.",
        ],
    }
    Path(output_json).write_text(json.dumps(report, indent=2, default=str))
    return report
```

## Common Failure Modes

| Failure | Symptom | Likely Cause | Diagnostic | Fix |
|---------|---------|--------------|------------|-----|
| Data leakage | Near-perfect CV score but poor performance on new chemistry | Preprocessing, feature selection, duplicate structures, or hyperparameter tuning used test information | Re-run with nested grouped CV and preprocessing inside the pipeline | Rebuild split and pipeline; report only outer-fold metrics |
| Random splits across correlated structures | Random-test MAE is low but leave-one-family-out MAE is high | Same prototype, trajectory neighbors, or database duplicates appear in train and test | Compare random CV to GroupKFold by prototype/formula/source | Use group splits matching deployment domain |
| Overconfident uncertainty | 90% intervals cover only 50-70% of held-out points | Miscalibrated probabilistic model or distribution shift | Reliability diagram and conformal multiplier | Calibrate intervals, revise kernel/model, or restrict domain |
| Hidden duplicates | Learning curve saturates early and test set has tiny nearest-neighbor distances | Duplicate structures or records not removed before split | StructureMatcher or descriptor nearest-neighbor audit | Deduplicate or group duplicates before splitting |
| Poor extrapolation | Interpolation points look good but extrapolation points have large residuals | Training coverage does not include held-out chemistry or descriptor region | Error by `coverage_class` and nearest-train distance | Report extrapolation limits; acquire labels in missing regions |
| Metric cherry-picking | Only R2 is reported; large absolute errors hidden | Chosen metric flatters target distribution | Report MAE, RMSE, residuals, per-group errors, and worst cases | Predefine metrics and include negative results |
| Validation on screened candidates only | Model appears accurate on selected high-confidence candidates but fails on full pool | Evaluation excludes rejected or low-scoring candidates | Compare selected vs unselected candidate distributions | Validate on full candidate pool or external benchmark |
| Ignoring negative results | Failed calculations or poor candidates are absent from validation set | Filtering after seeing labels or calculation outcomes | Audit campaign manifest for dropped candidates | Track every candidate, including failed and unfavorable ones |
| Physical constraint violations | Predictions imply impossible values or signs despite good MAE | Model lacks physics constraints or training coverage | Count negative moduli, invalid stability labels, or impossible units | Add sanity filters, constrained models, or targeted training data |
| Baseline not beaten | Complex surrogate only marginally better than mean/ridge/random forest | Descriptor/model mismatch or insufficient data | Run baseline comparison with same split | Simplify model or improve descriptors/data before deployment |

## Best Practices

- **Define the deployment scenario before splitting.** Interpolation within known chemistries and extrapolation to a new family require different validation splits.
- **Use nested CV for model selection.** Hyperparameter and descriptor choices must happen inside the inner loop.
- **Always include a simple baseline.** A complex surrogate that does not beat ridge, random forest, or mean predictor is not ready.
- **Report uncertainty calibration.** If uncertainty guides acquisition, reliability diagrams and coverage statistics are mandatory.
- **Track descriptor-domain coverage.** Report nearest-neighbor distances or group membership for held-out predictions.
- **Preserve the full candidate pool in closed-loop campaigns.** Optimization logs must include unselected, failed, and negative candidates.
- **Break down errors by group.** Composition family, structure prototype, database source, fidelity, active-learning iteration, and coverage class often reveal failures hidden by global MAE.
- **Use physical sanity checks as validation tests.** Count and inspect violations rather than treating them as cosmetic outliers.
- **Do not validate only after filtering by model confidence.** Confidence-based filtering can hide the regions where the model is least reliable.
- **Version validation artifacts.** Splits, predictions, metrics, figures, scripts, and model checksums belong in the reproducibility record.

## Integration with Other Skills

- **[gaussian-process-materials](../gaussian-process-materials/SKILL.md):** GP models require the calibration, coverage, split, and acquisition-validation checks described here before posterior uncertainty is used for decisions.
- **[bayesian-optimization](../bayesian-optimization/SKILL.md):** BO campaigns need retrospective validation of acquisition behavior, candidate-pool coverage, and negative-result tracking. This is an intentional forward reference until that skill is added.
- **[small-data-ml](../../scientific-ml/small-data-ml/SKILL.md):** Small-data ML practices such as nested CV, regularization, grouped splitting, and uncertainty reporting are shared with this validation workflow. This is an intentional forward reference until that skill is added.
- **[materials-featurization](../../scientific-data/materials-featurization/SKILL.md):** Feature leakage, descriptor choice, NaN handling, and descriptor-domain distances are upstream validation dependencies.
- **[materials-databases-access](../../scientific-data/materials-databases-access/SKILL.md):** Database-derived datasets require duplicate detection, target provenance, and source-aware splits before surrogate validation.
- **[scientific-visualization](../../scientific-data/scientific-visualization/SKILL.md):** Parity plots, residual distributions, reliability diagrams, learning curves, and coverage maps should follow reproducible plotting conventions from that skill.
- **[reproducibility-workflow](../../scientific-data/reproducibility-workflow/SKILL.md):** Validation splits, predictions, figures, metrics, model hashes, and candidate logs should be versioned as reproducibility artifacts.
- **[high-throughput-dft](../../electronic-structure/high-throughput-dft/SKILL.md):** Surrogate validation decides whether a model is reliable enough to prioritize or triage high-throughput DFT queues.
- **[mlp-validation](../../ml-interatomic-potentials/mlp-validation/SKILL.md):** The same principles apply to MLPs: held-out integrity, per-environment errors, physical sanity checks, and deployment-domain validation.
- **[uncertainty-mlp](../../ml-interatomic-potentials/uncertainty-mlp/SKILL.md):** Committee uncertainty for MLPs requires calibration and reliability checks analogous to surrogate uncertainty validation.
- **[active-learning-mlp](../../ml-interatomic-potentials/active-learning-mlp/SKILL.md):** Active-learning loops must retain candidate pools, uncertainty scores, selected structures, failed labels, and validation sets to avoid self-confirming progress metrics.

## Key References

1. Arlot, S., & Celisse, A. (2010). *A survey of cross-validation procedures for model selection.* Statistics Surveys, 4, 40-79. https://doi.org/10.1214/09-SS054
2. Varma, S., & Simon, R. (2006). *Bias in error estimation when using cross-validation for model selection.* BMC Bioinformatics, 7, 91. https://doi.org/10.1186/1471-2105-7-91
3. Dunn, A., et al. (2020). *Benchmarking materials property prediction methods: the Matbench test set and Automatminer reference algorithm.* npj Computational Materials, 6, 138. https://doi.org/10.1038/s41524-020-00406-3
4. Musil, F., et al. (2021). *Physics-inspired structural representations for molecules and materials.* Chemical Reviews, 121, 9759-9815. https://doi.org/10.1021/acs.chemrev.1c00021
5. Angelopoulos, A. N., & Bates, S. (2023). *Conformal prediction: A gentle introduction.* Foundations and Trends in Machine Learning, 16, 494-591. https://doi.org/10.1561/2200000101
