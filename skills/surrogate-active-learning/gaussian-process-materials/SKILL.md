---
name: gaussian-process-materials
domain: surrogate-active-learning
maturity: draft
tools:
  - scikit-learn
  - GPyTorch
  - BoTorch
  - GPflow
  - pymatgen
  - matminer
tested_versions:
  scikit-learn: "1.4.0"
  GPyTorch: "1.11"
  BoTorch: "0.10.0"
  GPflow: "2.9.1"
  pymatgen: "2024.11.13"
  matminer: "0.9.2"
requires_gpu: false
estimated_runtime: minutes-to-hours
---

# Gaussian Process Materials Models

## Description

This skill covers Gaussian process regression (GPR) for materials and chemistry datasets: descriptor selection, feature scaling, kernel design, small-data modeling, calibrated uncertainty, heteroscedastic noise, cross-validation, multi-output and multi-fidelity extensions, and integration with Bayesian optimization and active learning. Invoke this skill when building a data-efficient surrogate model for DFT, experimental, or database-derived properties where uncertainty estimates and query decisions matter as much as point predictions.

## Domain Context

Gaussian processes are probabilistic nonparametric models. A GP defines a distribution over functions; after observing data, it returns both a posterior mean and a posterior variance for each new input. This makes GPs attractive in computational materials science, where datasets are often small because DFT or experiment is expensive, and where the next calculation should be chosen by expected information gain or improvement rather than by random search.

The GP is only as meaningful as the representation and kernel. A crystal composition encoded by Magpie features, a SOAP vector, a graph kernel, and a hand-picked descriptor set each define a different notion of similarity. The kernel then states how property covariance decays with that similarity. A smooth RBF kernel assumes nearby descriptor vectors have smoothly varying properties; a Matern kernel permits rougher functions; additive kernels assume separable contributions from feature groups; product kernels encode interactions. These are scientific assumptions, not just hyperparameters.

GP uncertainty is epistemic under the model, not physical uncertainty by default. Posterior standard deviation grows far from the training data only if the kernel and feature scaling recognize that distance. In high-dimensional descriptor spaces, distances can concentrate, kernels can become nearly constant, and the GP can become overconfident or numerically unstable. Calibration against held-out data is required before using posterior uncertainty as an acquisition signal.

GPs are most useful for tens to a few thousands of training points. Exact GP inference scales as O(N^3) in time and O(N^2) in memory; sparse/variational GPs and structured kernels can extend this, but the simplicity and reliability of exact GPs is lost. For very large materials databases, GPs are usually used after downselection, featurization, or active learning rather than as a global model over millions of records.

## When to Use This Skill

- Modeling DFT or experimental properties with small datasets where uncertainty-aware prediction is needed.
- Driving Bayesian optimization or active learning for alloy composition, dopant selection, adsorption energy screening, synthesis condition search, or surrogate DFT campaigns.
- Comparing candidate descriptor families by cross-validated error and uncertainty calibration.
- Building calibrated surrogates for expensive labels such as formation energy, defect formation energy, adsorption energy, elastic constants, band gaps, or phonon-derived properties.
- Combining measurements or calculations with known uncertainty through per-point noise terms.
- Prototyping interpretable surrogate models before training larger neural networks.

## When Not to Use This Skill

- **For very large datasets without approximation.** Exact GPs become impractical beyond a few thousand points. Use sparse GPs, random feature approximations, kernel ridge regression, gradient boosting, or neural networks instead.
- **When descriptors are unscaled or poorly defined.** GP length scales are meaningless if feature units and scales are inconsistent.
- **When the target has discontinuities not captured by the representation.** Phase changes, spin-state switches, and structural relaxations can make a smooth kernel misleading unless the descriptor encodes the discontinuity.
- **When uncertainty will not be calibrated.** Posterior variance from a mis-specified kernel is not a trustworthy acquisition signal.
- **For graph-size-varying chemistry without a suitable kernel or descriptor.** Molecules and crystals need graph, fingerprint, or local-environment representations before a GP can operate.

## Core Concepts

- **Gaussian process prior:** A distribution over functions specified by a mean function and covariance kernel. The kernel determines how outputs co-vary as inputs change.
- **Kernel:** A similarity function such as RBF, Matern, periodic, linear, additive, product, SOAP kernel, graph kernel, or descriptor dot-product kernel. Kernel choice encodes assumptions about smoothness, periodicity, and feature interactions.
- **Length scale:** Kernel parameter controlling how rapidly predictions change with feature distance. Short length scales can overfit; long length scales can underfit and become overconfident.
- **Noise term:** Observation noise added to the diagonal of the covariance matrix. It can represent DFT convergence noise, experimental measurement error, or unresolved variability.
- **Heteroscedastic noise:** Different observations have different noise levels. This is common when mixing DFT settings, experimental sources, or task types; it must be modeled or avoided.
- **Posterior uncertainty:** Model uncertainty conditional on kernel, features, and noise assumptions. It is not automatically physical uncertainty.
- **Calibration:** Checking whether predicted uncertainty intervals contain the held-out labels at the expected frequency.
- **Acquisition function:** Rule for selecting the next point to evaluate, such as expected improvement, probability of improvement, upper confidence bound, posterior variance, or information gain.
- **Multi-fidelity model:** A surrogate that combines cheap low-fidelity labels with expensive high-fidelity labels, such as PBE and HSE band gaps or loose and tight DFT calculations.
- **Descriptor collapse:** In high-dimensional or poorly scaled feature spaces, points can appear similarly distant from each other, causing the kernel matrix to become ill-conditioned or uninformative.

## Key Workflows

### Workflow 1: Small-data GPR with descriptors, scaling, and calibration

Start with exact GPR and careful validation. This workflow is appropriate for tens to a few thousand structures with composition or structure descriptors from matminer, DScribe, or pymatgen-derived tables.

```python
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, Matern, WhiteKernel
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error


def build_gpr_pipeline(
    feature_columns: list[str],
    kernel_type: str = "matern",
    noise_level: float = 1e-5,
) -> Pipeline:
    """
    Build a leakage-safe GPR pipeline.

    Scaling and imputation are inside the pipeline so cross-validation fits them
    on training folds only. This is mandatory for honest small-data evaluation.
    """
    if kernel_type == "rbf":
        base = RBF(length_scale=np.ones(len(feature_columns)),
                   length_scale_bounds=(1e-2, 1e3))
    elif kernel_type == "matern":
        base = Matern(length_scale=np.ones(len(feature_columns)),
                      length_scale_bounds=(1e-2, 1e3),
                      nu=2.5)
    else:
        raise ValueError(f"Unknown kernel_type: {kernel_type}")

    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * base + WhiteKernel(
        noise_level=noise_level,
        noise_level_bounds=(1e-8, 1e0),
    )

    pre = ColumnTransformer([
        ("features", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), feature_columns)
    ])

    gpr = GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,
        n_restarts_optimizer=8,
        random_state=42,
    )
    return Pipeline([("preprocess", pre), ("gpr", gpr)])


def cross_validate_gpr(
    data_csv: str,
    feature_columns: list[str],
    target_column: str,
    output_dir: str,
    n_splits: int = 5,
) -> pd.DataFrame:
    """Run K-fold CV and save predictions with uncertainty for calibration."""
    df = pd.read_csv(data_csv)
    missing = [c for c in feature_columns + [target_column] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    rows = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(df), start=1):
        train = df.iloc[train_idx]
        test = df.iloc[test_idx]
        pipe = build_gpr_pipeline(feature_columns, kernel_type="matern")
        pipe.fit(train[feature_columns], train[target_column])
        pred, std = pipe.predict(test[feature_columns], return_std=True)
        for row_i, mu, sigma, y in zip(test.index, pred, std, test[target_column]):
            rows.append({
                "fold": fold,
                "row_index": int(row_i),
                "y_true": float(y),
                "y_pred": float(mu),
                "y_std": float(sigma),
                "abs_error": float(abs(mu - y)),
            })

    pred_df = pd.DataFrame(rows)
    mae = mean_absolute_error(pred_df["y_true"], pred_df["y_pred"])
    rmse = mean_squared_error(pred_df["y_true"], pred_df["y_pred"], squared=False)
    coverage_1sigma = (pred_df["abs_error"] <= pred_df["y_std"]).mean()
    coverage_2sigma = (pred_df["abs_error"] <= 2 * pred_df["y_std"]).mean()

    pred_df.to_csv(out / "gpr_cv_predictions.csv", index=False)
    (out / "gpr_cv_metrics.json").write_text(json.dumps({
        "data_csv": data_csv,
        "target_column": target_column,
        "feature_columns": feature_columns,
        "n_splits": n_splits,
        "mae": mae,
        "rmse": rmse,
        "coverage_1sigma": float(coverage_1sigma),
        "coverage_2sigma": float(coverage_2sigma),
        "expected_gaussian_1sigma": 0.6827,
        "expected_gaussian_2sigma": 0.9545,
    }, indent=2))
    print(f"CV MAE={mae:.4f}, RMSE={rmse:.4f}")
    print(f"Coverage: 1 sigma={coverage_1sigma:.1%}, 2 sigma={coverage_2sigma:.1%}")
    return pred_df
```

### Workflow 2: Kernel comparison and descriptor-family selection

Compare kernels under the same split and preprocessing. Kernel selection should be judged by both error and calibration, not by log marginal likelihood alone.

```python
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic, DotProduct, WhiteKernel, ConstantKernel
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error


def candidate_kernels(n_features: int) -> dict:
    """Return practical kernels for materials descriptor tables."""
    return {
        "rbf": ConstantKernel(1.0) * RBF(np.ones(n_features)) + WhiteKernel(1e-5),
        "matern_25": ConstantKernel(1.0) * Matern(np.ones(n_features), nu=2.5) + WhiteKernel(1e-5),
        "rough_matern_15": ConstantKernel(1.0) * Matern(np.ones(n_features), nu=1.5) + WhiteKernel(1e-5),
        "rational_quadratic": ConstantKernel(1.0) * RationalQuadratic(alpha=1.0, length_scale=1.0) + WhiteKernel(1e-5),
        "linear_plus_rbf": DotProduct() + ConstantKernel(1.0) * RBF(np.ones(n_features)) + WhiteKernel(1e-5),
    }


def compare_gpr_kernels(
    df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    output_json: str,
    n_splits: int = 5,
) -> list[dict]:
    """
    Compare kernels with identical folds.

    Use this before committing to Bayesian optimization, because a bad kernel
    produces misleading acquisition decisions even if the point error is modest.
    """
    X = df[feature_columns]
    y = df[target_column].to_numpy()
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=7)
    results = []

    for name, kernel in candidate_kernels(len(feature_columns)).items():
        fold_mae, fold_coverage = [], []
        for train_idx, test_idx in kf.split(X):
            pipe = Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("gpr", GaussianProcessRegressor(
                    kernel=kernel,
                    normalize_y=True,
                    n_restarts_optimizer=5,
                    random_state=7,
                )),
            ])
            pipe.fit(X.iloc[train_idx], y[train_idx])
            pred, std = pipe.predict(X.iloc[test_idx], return_std=True)
            abs_err = np.abs(pred - y[test_idx])
            fold_mae.append(mean_absolute_error(y[test_idx], pred))
            fold_coverage.append((abs_err <= 2 * std).mean())

        results.append({
            "kernel": name,
            "mae_mean": float(np.mean(fold_mae)),
            "mae_std": float(np.std(fold_mae)),
            "coverage_2sigma_mean": float(np.mean(fold_coverage)),
            "coverage_2sigma_target": 0.9545,
        })

    results = sorted(results, key=lambda r: (r["mae_mean"], abs(r["coverage_2sigma_mean"] - 0.9545)))
    Path(output_json).write_text(json.dumps(results, indent=2))
    return results
```

### Workflow 3: Heteroscedastic noise from DFT or experimental uncertainty

When each label has a known uncertainty, pass per-point noise through `alpha` in scikit-learn or use likelihood models in GPyTorch/GPflow. This is essential when mixing converged DFT with experimental labels or lower-fidelity calculations.

```python
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern
from sklearn.preprocessing import StandardScaler


def fit_heteroscedastic_gpr(
    data_csv: str,
    feature_columns: list[str],
    target_column: str,
    noise_std_column: str,
) -> dict:
    """
    Fit a GPR with per-observation noise variance.

    noise_std_column is in the same units as the target, e.g. eV/atom.
    For DFT labels, estimate this from convergence tests; for experiments,
    use reported measurement uncertainty where available.
    """
    df = pd.read_csv(data_csv).dropna(subset=feature_columns + [target_column, noise_std_column])
    scaler = StandardScaler()
    X = scaler.fit_transform(df[feature_columns])
    y = df[target_column].to_numpy()
    noise_var = np.square(df[noise_std_column].to_numpy())

    if np.any(noise_var <= 0):
        raise ValueError("Noise variances must be positive")

    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
        length_scale=np.ones(X.shape[1]),
        length_scale_bounds=(1e-2, 1e3),
        nu=2.5,
    )
    model = GaussianProcessRegressor(
        kernel=kernel,
        alpha=noise_var,
        normalize_y=True,
        n_restarts_optimizer=8,
        random_state=42,
    )
    model.fit(X, y)

    return {
        "model": model,
        "scaler": scaler,
        "n_train": len(df),
        "noise_std_min": float(np.sqrt(noise_var).min()),
        "noise_std_max": float(np.sqrt(noise_var).max()),
        "kernel": str(model.kernel_),
    }
```

### Workflow 4: Active learning acquisition from posterior mean and uncertainty

Use acquisition functions to select candidate structures or compositions for the next DFT or experiment. The acquisition function must match the campaign objective: minimize a property, maximize it, improve stability, or reduce uncertainty.

```python
import numpy as np
import pandas as pd
from scipy.stats import norm


def expected_improvement_minimize(
    mu: np.ndarray,
    sigma: np.ndarray,
    best_y: float,
    xi: float = 0.01,
) -> np.ndarray:
    """
    Expected improvement for minimization.

    xi controls exploration. Larger xi favors uncertain regions.
    """
    sigma = np.maximum(sigma, 1e-12)
    improvement = best_y - mu - xi
    z = improvement / sigma
    return improvement * norm.cdf(z) + sigma * norm.pdf(z)


def upper_confidence_bound_minimize(
    mu: np.ndarray,
    sigma: np.ndarray,
    beta: float = 2.0,
) -> np.ndarray:
    """Lower confidence bound for minimization; smaller is better."""
    return mu - beta * sigma


def rank_candidates_for_dft(
    trained_pipeline,
    candidate_csv: str,
    feature_columns: list[str],
    output_csv: str,
    objective: str = "minimize",
    n_select: int = 50,
) -> pd.DataFrame:
    """
    Rank candidate materials by GP acquisition score.

    Candidates should already be deduplicated and inside the intended chemistry
    domain. Acquisition cannot repair bad candidate generation.
    """
    candidates = pd.read_csv(candidate_csv)
    mu, sigma = trained_pipeline.predict(candidates[feature_columns], return_std=True)
    candidates["gpr_mean"] = mu
    candidates["gpr_std"] = sigma

    if objective == "minimize":
        best_y = candidates["gpr_mean"].min()
        candidates["expected_improvement"] = expected_improvement_minimize(mu, sigma, best_y)
        candidates["lcb"] = upper_confidence_bound_minimize(mu, sigma)
        ranked = candidates.sort_values(["expected_improvement", "gpr_std"], ascending=False)
    elif objective == "explore":
        ranked = candidates.sort_values("gpr_std", ascending=False)
    else:
        raise ValueError(f"Unknown objective: {objective}")

    selected = ranked.head(n_select).copy()
    selected.to_csv(output_csv, index=False)
    print(f"Selected {len(selected)} candidates for DFT or experiment")
    return selected
```

### Workflow 5: GPyTorch model for larger descriptor tables

Use GPyTorch when scikit-learn becomes too slow or when GPU acceleration, custom likelihoods, or variational approximations are needed. Start with exact GPs before moving to sparse variational models.

```python
import torch
import gpytorch


class ExactMaterialsGP(gpytorch.models.ExactGP):
    """Exact GP for scaled descriptor tensors."""

    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=train_x.shape[-1])
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def train_gpytorch_exact(
    X_train,
    y_train,
    n_iter: int = 300,
    lr: float = 0.05,
    device: str = "cpu",
):
    """
    Train an exact GPyTorch GP.

    X_train should already be imputed and standardized. y_train should be
    standardized or at least centered for stable optimization.
    """
    train_x = torch.as_tensor(X_train, dtype=torch.float32, device=device)
    train_y = torch.as_tensor(y_train, dtype=torch.float32, device=device)
    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
    model = ExactMaterialsGP(train_x, train_y, likelihood).to(device)

    model.train()
    likelihood.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in range(n_iter):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()
        if (i + 1) % 50 == 0:
            print(f"Iter {i+1}: loss={loss.item():.3f}")

    return model, likelihood
```

### Workflow 6: Multi-fidelity and multi-output planning

Many materials campaigns have cheap approximate labels and expensive accurate labels. Multi-fidelity GPs can exploit correlations, but they require careful bookkeeping and validation.

```python
import pandas as pd


def audit_multifidelity_table(
    csv_file: str,
    structure_id_col: str,
    fidelity_col: str,
    target_col: str,
) -> dict:
    """
    Audit a table before multi-fidelity GP modeling.

    Example fidelities: PBE, PBE+U, HSE, experiment, loose_DFT, tight_DFT.
    Multi-fidelity modeling assumes a learnable relation between fidelities;
    this must be checked empirically.
    """
    df = pd.read_csv(csv_file)
    pivot = df.pivot_table(
        index=structure_id_col,
        columns=fidelity_col,
        values=target_col,
        aggfunc="mean",
    )
    overlap = pivot.notna().sum(axis=1)
    pair_counts = pivot.notna().T @ pivot.notna()
    report = {
        "n_rows": int(len(df)),
        "n_structures": int(pivot.shape[0]),
        "fidelities": list(pivot.columns),
        "n_with_multiple_fidelities": int((overlap >= 2).sum()),
        "pair_overlap_counts": pair_counts.astype(int).to_dict(),
    }
    print(report)
    if report["n_with_multiple_fidelities"] < 10:
        print("WARNING: too little overlap to learn fidelity correlations reliably")
    return report
```

## Common Failure Modes

| Failure | Symptom | Likely Cause | Diagnostic | Fix |
|---------|---------|--------------|------------|-----|
| Unscaled features | Optimized length scales differ by orders of magnitude; one feature dominates predictions | Descriptor columns have incompatible units and ranges | Inspect scaler means/stds and fitted ARD length scales | Put imputation and `StandardScaler` inside the CV pipeline |
| Overconfident extrapolation | Posterior std is low for candidates far outside training chemistry | Kernel length scale too long, descriptor collapse, or candidate features inside scaled numeric range but chemically novel | Compare nearest-neighbor descriptor distance and calibration residuals | Add OOD detector, tighter domain filter, or recalibrate on chemically held-out splits |
| Bad kernel choice | Good training fit but poor CV error or poor uncertainty coverage | Smoothness or additivity assumptions do not match target property | Compare kernels under identical folds and inspect residuals by chemistry | Use Matern/rough kernels, additive/product kernels, or a better descriptor |
| Numerical instability in covariance matrix | Cholesky errors, optimizer warnings, wildly varying predictions | Duplicate points, tiny noise level, ill-conditioned kernel matrix | Check duplicate descriptor rows and kernel condition number | Add jitter/noise, remove duplicates, scale features, or merge repeated labels |
| Data leakage | CV MAE is unrealistically low and fails on new chemistry | Scaling, imputation, feature selection, or duplicate structures leak across folds | Audit pipeline; run composition/prototype/structure-grouped CV | Put preprocessing inside pipeline and split by chemistry/structure groups |
| High-dimensional descriptor collapse | All pairwise distances become similar; kernel matrix nearly constant | Too many redundant descriptors relative to data size | Plot pairwise distance histogram and PCA variance | Reduce descriptors, use feature selection inside CV, or use domain descriptors |
| Posterior uncertainty treated as physical uncertainty | Reported GP std is interpreted as experimental or DFT uncertainty | Model uncertainty was not calibrated and excludes label noise/model bias | Check interval coverage on held-out data | Report calibrated predictive intervals and distinguish epistemic/model uncertainty from measurement noise |
| Heteroscedastic noise ignored | Model overfits noisy experiments and underweights precise DFT labels incorrectly | One global white-noise term used for mixed uncertainty sources | Plot residuals by source and known uncertainty | Use per-point `alpha`, source-specific noise, or separate models |
| Acquisition stuck near known optimum | Bayesian optimization repeatedly selects near-duplicates | Expected improvement over-exploits and diversity filter absent | Measure descriptor distance among selected candidates | Add diversity filtering, batch acquisition, or UCB/posterior-variance exploration |
| Wrong target definition | Model mixes band gaps, formation energies, or corrected/raw energies under one column | Database joins or preprocessing collapsed incompatible targets | Inspect target provenance by source/task/correction | Split targets by definition and record provenance metadata |

## Best Practices

- **Start with exact scikit-learn GPR for small data.** It is easier to inspect, cross-validate, and debug than variational models.
- **Put preprocessing inside the model pipeline.** Imputation, scaling, feature selection, and dimensionality reduction must be fit only on training folds.
- **Use chemistry-aware validation splits.** Random splits overestimate performance when near-duplicate structures or composition families appear in both train and test.
- **Judge uncertainty by calibration, not appearance.** Report one- and two-sigma coverage, reliability curves, or conformal intervals.
- **Keep descriptor metadata with the model.** Feature names, scaling parameters, SOAP/matminer settings, and target definition are part of the model.
- **Use per-point noise when label uncertainty varies.** Do not force experimental and DFT labels into a single homoscedastic noise term.
- **Inspect fitted length scales.** Very small length scales indicate overfitting; very large length scales indicate underfitting or irrelevant features.
- **Deduplicate before fitting.** Duplicate feature rows with conflicting labels should be averaged with noise metadata or investigated, not silently fit.
- **Treat acquisition as a policy, not a result.** The selected candidates inherit all assumptions of the GP, descriptor, domain filter, and acquisition function.
- **Log every active-learning iteration.** Store model state, training IDs, candidate pool, acquisition values, selected candidates, and labels returned by DFT or experiment.

## Integration with Other Skills

- **[materials-featurization](../../scientific-data/materials-featurization/SKILL.md):** GP quality depends on descriptor choice, scaling, leakage control, and feature metadata from that skill.
- **[materials-databases-access](../../scientific-data/materials-databases-access/SKILL.md):** Database-derived tables provide common small-data and screening inputs for GP surrogates, but target compatibility and duplicate handling must be resolved first.
- **[reproducibility-workflow](../../scientific-data/reproducibility-workflow/SKILL.md):** GP models, feature scalers, training IDs, acquisition results, seeds, and candidate pools should be versioned as campaign artifacts.
- **[active-learning-mlp](../../ml-interatomic-potentials/active-learning-mlp/SKILL.md):** GP acquisition logic is conceptually parallel to MLP active learning; both require calibrated uncertainty and diversity filtering.
- **[uncertainty-mlp](../../ml-interatomic-potentials/uncertainty-mlp/SKILL.md):** GP posterior uncertainty provides a probabilistic comparison point for committee uncertainty, but both require calibration before use.
- **[bayesian-optimization](../bayesian-optimization/SKILL.md):** Bayesian optimization formalizes acquisition functions, batch selection, constraints, and stopping criteria for GP-guided search. This is an intentional forward reference until that skill is added.
- **[surrogate-validation](../surrogate-validation/SKILL.md):** Surrogate validation covers calibration curves, domain shift tests, and residual diagnostics for models like the GPs described here. This is an intentional forward reference until that skill is added.
- **[small-data-ml](../../scientific-ml/small-data-ml/SKILL.md):** Small-data practices such as nested CV, grouped splits, regularization, and uncertainty reporting apply directly to GP materials models. This is an intentional forward reference until that skill is added.
- **[high-throughput-dft](../../electronic-structure/high-throughput-dft/SKILL.md):** GP surrogates can prioritize high-throughput DFT queues and decide which candidates to recompute at higher fidelity. This is an intentional forward reference until that skill is added.
- **[dft-convergence](../../electronic-structure/dft-convergence/SKILL.md):** DFT label noise and mixed convergence settings directly affect GP noise terms and target compatibility.
- **[scientific-visualization](../../scientific-data/scientific-visualization/SKILL.md):** GP parity plots, calibration diagrams, acquisition maps, and uncertainty heatmaps should follow the visualization conventions in that skill.

## Key References

1. Rasmussen, C. E., & Williams, C. K. I. (2006). *Gaussian Processes for Machine Learning.* MIT Press. https://gaussianprocess.org/gpml/
2. Bartok, A. P., Kondor, R., & Csanyi, G. (2013). *On representing chemical environments.* Physical Review B, 87, 184115. https://doi.org/10.1103/PhysRevB.87.184115
3. Seko, A., Maekawa, T., Tsuda, K., & Tanaka, I. (2014). *Machine learning with systematic density-functional theory calculations: Application to melting temperatures of single- and binary-component solids.* Physical Review B, 89, 054303. https://doi.org/10.1103/PhysRevB.89.054303
4. Tran, K., & Ulissi, Z. W. (2018). *Active learning across intermetallics to guide discovery of electrocatalysts for CO2 reduction and H2 evolution.* Nature Catalysis, 1, 696-703. https://doi.org/10.1038/s41929-018-0142-1
5. Balandat, M., et al. (2020). *BoTorch: A framework for efficient Monte-Carlo Bayesian optimization.* Advances in Neural Information Processing Systems, 33, 21524-21538. https://arxiv.org/abs/1910.06403
