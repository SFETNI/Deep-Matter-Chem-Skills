---
name: neural-microstructure-validation
domain: phase-field-microstructure
maturity: draft
tools:
  - numpy
  - scipy
  - scikit-image
  - pandas
  - matplotlib
tested_versions:
  # Execution-verified: the metric code in Workflows 2-5 was run on synthetic
  # shrinking-circle and Voronoi label fields with these versions (Python 3.14).
  numpy: "2.5.2"
  scipy: "1.18.0"
  scikit-image: "0.26.0"
requires_gpu: false
estimated_runtime: minutes-to-hours
---

# Neural Microstructure Validation

## Description

This skill is a model-independent protocol for checking learned microstructure evolution against physical reference simulations. It applies to PINNs, neural time integrators, neural operators, and other surrogates of phase-field or grain-growth dynamics. It covers how to pre-register a benchmark, admissibility residuals, permutation-invariant grain matching, interface and kinetics errors, extinction and survivor identity, topology, energy-decay checks, refinement and seed statistics, and fair cost accounting. Invoke it before reporting any accuracy, generalization, or speed-up claim for a neural microstructure model. Its numpy/scipy/scikit-image metric library was executed on synthetic test fields.

## Domain Context

A neural microstructure model is only as credible as the comparison used to judge it. Phase-field microstructures are high-dimensional, multiphase, and topologically active: grains shrink, vanish, and change neighbours. A single field-averaged error can therefore hide the failures that matter. A model can reach a small mean absolute error while it:

- delays or advances every extinction;
- retains a grain the reference eliminated;
- nucleates a spurious grain;
- drifts off the simplex of admissible phase fractions.

Validation must therefore combine three kinds of check:

1. **Field-level errors.**
2. **Grain-level and event-level metrics:** areas, survivor sets, extinction timing.
3. **Constraint and physics checks:** bounds, unit sum, monotone interfacial energy where the dynamics are a gradient flow.

Three distinctions recur:

- **Reference fidelity versus physical fidelity.** Most neural studies compare against direct integration of the *same* discrete operator. That establishes numerical agreement with that discretization, not agreement with a real material. Material claims additionally need calibrated parameters and experimental or higher-fidelity validation. Treat these as separate evidence tiers.
- **Training-horizon extrapolation versus initial-condition generalization.** Rolling a model past its training horizon on the training microstructure tests temporal stability. Running it on an unseen initial microstructure tests generalization. These are different claims. Neither implies transfer to a different phase count, resolution, mobility law, or physics.
- **Development versus prospective evidence.** A case that was inspected while choosing architectures, thresholds, or checkpoints is development evidence, even if it was never trained on. Prospective evidence requires these to be frozen before scoring, and each case scored once:
  - cases;
  - reference data;
  - metrics;
  - acceptance thresholds;
  - checkpoint (by hash).

For isotropic curvature-driven grain growth, sharp-interface theory gives useful analytical anchors:

- **Isolated grain:** an isolated circular grain in 2D obeys R² = R0² − 2Mσt. Its area decreases linearly and it vanishes at t_ext = R0²/(2Mσ). In 3D a sphere obeys R² = R0² − 4Mσt.
- **2D von Neumann–Mullins law:** dA/dt = (πMσ/3)(n − 6) for a grain with n neighbours, assuming equal boundary energies and 120° triple junctions.
- **3D:** the MacPherson–Srolovitz relation generalizes this law.

Mapping phase-field parameters to an effective sharp-interface mobility M and energy σ depends on the model and calibration. Diffuse-interface width, grid resolution, and finite-time-step effects shift the effective constants. [EXPERT REVIEW NEEDED: effective Mσ must be calibrated for the specific phase-field formulation before analytical laws are used as quantitative acceptance criteria]

## When to Use This Skill

- Use when evaluating any neural surrogate of multiphase-field or grain-growth evolution against a reference solver: PINN, recurrent integrator, neural operator, or autoregressive CNN.
- Use when designing held-out or prospective benchmark cohorts and fixing acceptance criteria before scoring.
- Use when a claim involves grain extinction, survivor sets, topology changes, triple-junction geometry, or coarsening kinetics, not only field error.
- Use when comparing two neural models, or a neural model against a persistence baseline, on equal footing.
- Use when reporting runtime or speed-up for a learned model relative to a classical solver.
- Use when auditing a published neural phase-field result before relying on it.

## When Not to Use This Skill

- **When the target is material prediction rather than solver agreement.** This protocol measures agreement with a reference simulation. Material validation also needs calibrated thermodynamics and kinetics plus experimental microstructure data (EBSD, tomography, coarsening kinetics).
- **When the reference itself is unconverged.** If the reference solver is not mesh- and time-step converged for the reported observables, agreement with it is not meaningful.
- **For conserved-order-parameter dynamics without adaptation.** Spinodal decomposition and Cahn–Hilliard coarsening need conserved-mass checks and structure-factor metrics. Grain-identity matching may not apply.
- **For a single qualitative snapshot.** Visual similarity of one frame is not validation; use the time-series and event metrics here.
- **When only one benchmark case exists.** Report it as a case study, not as a pass rate or generalization result.

## Core Concepts

- **Benchmark specification:** A frozen, versioned record fixing cases, reference data, metrics, thresholds, evaluation times, and checkpoint hash before any held-out scoring.
- **Persistence baseline:** The trivial predictor that keeps the t0 field unchanged. Gain over persistence, 1 − D_model/D_persistence, shows whether the model captures the evolution or just stays near the initial state. Early-time errors are uninformative without it.
- **Admissibility:** Phase fractions must satisfy 0 ≤ φᵢ ≤ 1 and Σᵢ φᵢ = 1 pointwise. Violations are measured, not assumed away, even when the architecture enforces them, because post-processing and float precision matter.
- **Grain label map:** The argmax over phase fields gives a partition into grains. Label identities in a model output are arbitrary unless the model tracks them, so grain-level comparison requires permutation-invariant matching.
- **Hungarian matching:** An optimal one-to-one assignment between reference and predicted grains, here maximizing intersection-over-union (IoU). Unmatched reference grains are *missed*; unmatched predicted grains are *spurious*.
- **Survivor set and survivor F1:** The set of grains present at an evaluation time, and the F1 score between reference and predicted survivor sets after matching.
- **Extinction event:** The first time a grain's area falls below a sub-resolution floor with no later reappearance. Its timing resolution is limited by the snapshot cadence unless fitted.
- **Interface distance:** The symmetric mean and Hausdorff distance between reference and predicted boundary sets, in physical units. Its lower bound is set by the grid spacing.
- **Junction geometry:** The count and location of triple (or higher) junctions, and their dihedral angles. For equal isotropic energies the reference angle is 120°. Measured diffuse-field angles depend on resolution.
- **Interfacial-energy proxy:** For isotropic curvature flow, total boundary length (2D) or area (3D) must decrease monotonically. A rise signals non-physical dynamics, independent of the model.
- **Evidence tier:** Development, held-out-but-inspected, and prospective. Report them separately and never pool them.

## Key Workflows

### Workflow 1: Pre-register the benchmark and acceptance criteria

**Inputs:** reference solver configuration, candidate test microstructures, model checkpoint.
**Assumptions:** the reference is converged for the observables below (see Workflow 6).
**Output:** a frozen, hashed specification that later scoring scripts read and do not modify.

```yaml
# benchmark_spec.yaml — fix BEFORE scoring held-out cases; record its SHA-256 in the report.
benchmark: n16_isotropic_grain_growth_2d
reference:
  operator: "projected multiphase Allen-Cahn, isotropic, equal mobility/energy"
  solver: "explicit FD, periodic, dx=1, dt=0.04"     # must match the claim being made
  convergence_record: refs/convergence_study.json    # Workflow 6 output
checkpoint:
  path: checkpoints/model.npz
  sha256: "<fill before scoring>"
cases:
  development: [ic_000]                 # used during design; never counted as prospective
  prospective: [ic_101, ic_102, ic_103, ic_104, ic_105, ic_106, ic_107, ic_108]
  stress: [ic_dense, ic_sparse]
evaluation_steps: [1000, 4000, 8000, 12000]
metrics_and_thresholds:                 # study-specific; justify each value in the report
  label_disagreement_max: 0.05
  gain_over_persistence_min: 0.80       # evaluated only where persistence disagreement > 0.10
  survivor_f1_min: 0.95
  extinction_timing_abs_max_steps: 100
  unit_sum_abs_max: 1.0e-5
  spurious_grains_max: 0
pass_rule: "a case passes only if every metric passes at every evaluation step"
cohort_reporting: "k/n with Wilson 95% interval; development and prospective never pooled"
```

**Validation criterion:** the specification hash in the report must match the file used by the scoring script. Any change after scoring has begun turns the affected cases into development evidence.

### Workflow 2: Admissibility residuals at every saved step

**Inputs:** a phase-fraction array `phi` of shape (n_phases, ny, nx) or (n_phases, nz, ny, nx).
**Outputs:** worst-case and mean violations of bounds and unit sum, tracked through time.
**Silent failure targeted:** architectures that enforce admissibility (softmax, zero-sum increments, clip-and-renormalize maps) can still violate it after float32 accumulation or post-processing. A model without enforcement can drift slowly over long rollouts.

```python
# Execution-verified on synthetic fields (numpy 2.5.2). Works for 2D or 3D arrays.
import numpy as np


def admissibility_residuals(phi: np.ndarray) -> dict:
    """phi: (n_phases, *spatial) phase fractions at one time step."""
    s = phi.sum(axis=0)
    return {
        "unit_sum_max": float(np.abs(s - 1.0).max()),
        "unit_sum_mean": float(np.abs(s - 1.0).mean()),
        "below_zero_max": float(np.clip(-phi, 0.0, None).max()),
        "above_one_max": float(np.clip(phi - 1.0, 0.0, None).max()),
        "frac_cells_violating_1e-3": float(
            ((np.abs(s - 1.0) > 1e-3) | (phi < -1e-3).any(0) | (phi > 1 + 1e-3).any(0)).mean()
        ),
    }


def labels_from_phi(phi: np.ndarray) -> np.ndarray:
    """Grain partition by argmax. Ties and near-uniform cells (e.g. phi = 1/N fallback)
    are silently assigned to the lowest index; count such cells separately."""
    return np.argmax(phi, axis=0).astype(np.int64)
```

**Diagnostic:** plot `unit_sum_max` against step on a log scale. Growth over the rollout indicates accumulating projection error, even if every single step looks acceptable.

### Workflow 3: Permutation-invariant grain matching and label disagreement

**Inputs:** reference and predicted label maps on the same grid.
**Outputs:** matched grain pairs with IoU; missed and spurious grains; aligned label disagreement; disagreement of the persistence baseline.
**Assumption:** grains are compared at identical times and resolution. Resample one field first if not, and record the resampling.

```python
# Execution-verified on synthetic Voronoi fields with permuted label ids (scipy 1.18.0).
import numpy as np
from scipy.optimize import linear_sum_assignment


def iou_matrix(lab_ref: np.ndarray, lab_pred: np.ndarray):
    ref_ids, pred_ids = np.unique(lab_ref), np.unique(lab_pred)
    r = np.searchsorted(ref_ids, lab_ref.ravel())
    p = np.searchsorted(pred_ids, lab_pred.ravel())
    inter = np.bincount(r * len(pred_ids) + p,
                        minlength=len(ref_ids) * len(pred_ids)).reshape(len(ref_ids), len(pred_ids))
    union = inter.sum(1, keepdims=True) + inter.sum(0, keepdims=True) - inter
    return inter / np.maximum(union, 1), ref_ids, pred_ids


def match_grains(lab_ref, lab_pred, min_iou: float = 0.5) -> dict:
    iou, ref_ids, pred_ids = iou_matrix(lab_ref, lab_pred)
    rows, cols = linear_sum_assignment(-iou)          # Hungarian assignment, maximize IoU
    pairs = [(int(ref_ids[i]), int(pred_ids[j]), float(iou[i, j]))
             for i, j in zip(rows, cols) if iou[i, j] >= min_iou]
    return {
        "pairs": pairs,
        "mean_matched_iou": float(np.mean([q[2] for q in pairs])) if pairs else 0.0,
        "missed_ref": sorted(set(map(int, ref_ids)) - {q[0] for q in pairs}),
        "spurious_pred": sorted(set(map(int, pred_ids)) - {q[1] for q in pairs}),
    }


def aligned_disagreement(lab_ref, lab_pred, pairs) -> float:
    """Cell fraction whose predicted label (mapped to reference ids) is wrong.
    Unmatched predicted ids count as wrong."""
    mapping = {p: r for r, p, _ in pairs}
    mapped = np.vectorize(lambda v: mapping.get(int(v), -1))(lab_pred)
    return float((mapped != lab_ref).mean())


def gain_over_persistence(d_model: float, d_persistence: float) -> float:
    return float(1.0 - d_model / d_persistence) if d_persistence > 0 else float("nan")


# Persistence baseline: score the t0 reference labels against the reference at time t.
# Identity mapping is valid because both come from the reference solver.
# d_pers = aligned_disagreement(ref_t, ref_t0, [(g, g, 1.0) for g in np.unique(ref_t0)])
```

**Diagnostics:**

- If `missed_ref` and `spurious_pred` are both non-empty and the grains are small, check whether a small grain has simply been displaced. IoU matching is harsh for grains a few cells across. Report the area distribution of unmatched grains.
- A grain that is split into disconnected fragments (`scipy.ndimage.label` on one label returns more than one component) is a topology error, even if IoU is high.

### Workflow 4: Interface geometry, kinetics, and extinction timing

**Inputs:** time series of label maps from the reference and the matched model, plus grid spacing `dx`.
**Outputs:** interface distance against time; per-grain area curves; fitted kinetic constants; extinction times and their errors.

```python
# Execution-verified: an analytical shrinking circle gives fitted M*sigma = 0.9994 (true 1.0)
# and extinction times exact to 0.1 time units (scikit-image 0.26.0).
import numpy as np
from scipy import ndimage
from skimage.segmentation import find_boundaries


def interface_distance(lab_ref, lab_pred, dx: float) -> dict:
    """Symmetric boundary distances in physical units; resolution floor ~ dx."""
    b_ref = find_boundaries(lab_ref, mode="inner")
    b_pred = find_boundaries(lab_pred, mode="inner")
    if not b_ref.any() or not b_pred.any():
        return {"mean_sym": np.nan, "p95_sym": np.nan, "hausdorff": np.nan, "dx": dx}
    to_ref = ndimage.distance_transform_edt(~b_ref) * dx
    to_pred = ndimage.distance_transform_edt(~b_pred) * dx
    a, b = to_ref[b_pred], to_pred[b_ref]
    return {"mean_sym": float(0.5 * (a.mean() + b.mean())),
            "p95_sym": float(max(np.percentile(a, 95), np.percentile(b, 95))),
            "hausdorff": float(max(a.max(), b.max())), "dx": dx}


def grain_areas(lab, dx: float) -> dict:
    ids, counts = np.unique(lab, return_counts=True)
    return {int(i): float(c * dx ** lab.ndim) for i, c in zip(ids, counts)}


def extinction_time_fit(times, areas, area_floor: float, n_fit: int = 5) -> float:
    """Fit A(t) linearly over the last n_fit samples above area_floor and extrapolate
    to zero. In 2D isotropic curvature flow A(t) is ~linear (exact for an isolated
    circle; von Neumann-Mullins at fixed n). Taking the first saved frame below the
    floor instead biases t_ext late by up to one snapshot interval.
    3D: fit A^(3/2)-equivalent forms or volume-based laws instead."""
    times, areas = np.asarray(times, float), np.asarray(areas, float)
    alive = np.nonzero(areas >= area_floor)[0]
    if len(alive) == len(areas):
        return float("nan")                    # grain survives the evaluated horizon
    k = alive[-n_fit:]
    if len(k) < 2:
        return float(times[alive[-1] + 1]) if len(alive) else float(times[0])
    slope, intercept = np.polyfit(times[k], areas[k], 1)
    return float(-intercept / slope) if slope < 0 else float("nan")
```

Kinetic checks:

- **Isolated grain:** fit dA/dt and compare with −2πMσ (2D), using the reference's calibrated Mσ. Report the slope ratio of model to reference rather than an absolute value.
- **Polycrystal:** regress dA/dt per grain against neighbour count n. The reference should give a slope near πMσ/3 with the intercept near n = 6. The model must reproduce the reference's slope, not the textbook value, because both share the same discretization artifacts. [EXPERT REVIEW NEEDED: neighbour counting near periodic boundaries and for grains with transient neighbours needs a convention]
- **Extinction:** report signed timing errors per grain (premature negative, late positive), the extinction *order*, and the survivor set. Systematic premature extinction is a known failure pattern of neural integrators and does not show up in mean field error.

### Workflow 5: Topology, spurious nucleation, and energy decay

**Inputs:** label maps through time.
**Outputs:** junction counts, fragment counts, spurious-grain events, and the monotonicity of the interfacial-energy proxy.

```python
# Execution-verified on synthetic Voronoi fields.
import numpy as np
from scipy import ndimage


def junction_points(lab2d) -> np.ndarray:
    """2D: mask of 2x2 windows containing >= 3 distinct labels."""
    w = np.stack([lab2d[:-1, :-1], lab2d[:-1, 1:], lab2d[1:, :-1], lab2d[1:, 1:]])
    srt = np.sort(w, axis=0)
    return (1 + (np.diff(srt, axis=0) != 0).sum(axis=0)) >= 3


def count_junctions(lab2d) -> int:
    """Connected clusters of junction windows (one cluster ~ one junction)."""
    return int(ndimage.label(junction_points(lab2d))[1])


def fragments_per_label(lab) -> dict:
    return {int(g): int(ndimage.label(lab == g)[1]) for g in np.unique(lab)}


def total_boundary_measure(lab, dx: float) -> float:
    """Grid-face count of label changes * dx^(ndim-1). Biased for curved boundaries
    (lattice metric) but consistent between reference and model on the same grid."""
    faces = sum(int((np.take(lab, range(1, lab.shape[a]), a) !=
                     np.take(lab, range(0, lab.shape[a] - 1), a)).sum()) for a in range(lab.ndim))
    return float(faces * dx ** (lab.ndim - 1))


def energy_monotonicity(series: np.ndarray, rel_tol: float = 1e-3) -> dict:
    """Isotropic curvature flow is a gradient flow of boundary measure: it should not
    increase. Report every step where it rises by more than rel_tol."""
    rises = np.nonzero(np.diff(series) > rel_tol * series[:-1])[0]
    return {"n_rises": int(len(rises)), "first_rise_index": int(rises[0]) if len(rises) else None}
```

Interpretation:

- **Spurious nucleation:** a predicted grain absent at t0, or a re-appearing extinct grain. Such an event is a hard failure for a curvature-driven model, because the dynamics contain no nucleation term.
- **Junction angles:** measure them from tangents fitted along each boundary at a distance of several interface widths from the junction. Report the resolution used. Deviations of a few degrees can come from the diffuse interface alone. [EXPERT REVIEW NEEDED: tangent-fitting distance and its effect on angle bias]
- **Exact discrete free energy:** where the reference defines a discrete free-energy functional (gradient plus multiwell or obstacle terms), evaluate it for both fields. Require the model's evolution to be non-increasing when the reference's is.
- **Conserved quantities:** curvature-driven multiphase Allen–Cahn does not conserve individual phase volumes; only the local unit sum is constrained. Conserved-dynamics models need per-phase mass checks instead.

### Workflow 6: Reference convergence, statistics, and fair cost accounting

**Reference convergence:**

- Before scoring, refine the reference grid (e.g. 2× in each direction, with interface width held fixed in physical units) and the time step (e.g. ½).
- Confirm that extinction times, survivor sets, and area-slope ratios change by less than the acceptance thresholds.
- A model "error" smaller than the reference's own discretization error is not meaningful.

**Statistics:**

- Report per-case results, never only averages.
- For cohort pass rates, report k/n with a Wilson score interval. With n = 8 and k = 7, the 95% interval is about 0.53–0.98, which is far from a demonstration of reliability.
- Where training is stochastic, repeat it over several seeds. Report the spread of every metric, and run the full matrix of seeds × microstructure realizations when resources allow.
- A single seed per benchmark should be stated as a limitation.

```python
# Execution-verified (numpy 2.5.2): 22 predicted vs 21 reference survivors -> F1 = 0.9767;
# Wilson 95% interval for 7/8 = (0.529, 0.978).
import numpy as np


def survivor_f1(ref_survivors, pred_survivors) -> float:
    r, p = set(ref_survivors), set(pred_survivors)
    tp = len(r & p)
    return 0.0 if tp == 0 else float(2 * tp / (len(r) + len(p)))


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    ph, den = k / n, 1 + z * z / n
    centre = (ph + z * z / (2 * n)) / den
    half = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return (float(centre - half), float(centre + half))
```

**Fair cost accounting** (record each item):

1. Hardware for each side (GPU model or vGPU slice, CPU cores, precision).
2. Rollout wall-clock time excluding warm-up and checkpoint loading. Report those separately.
3. Training cost in GPU-hours, *measured* or clearly labeled *projected*.
4. The reference solver run at the accuracy level the model actually achieves, on comparable hardware. A GPU network compared with a single-threaded CPU stencil code is not a fair speed-up.
5. The break-even number of rollouts at which training cost is amortized.

If these are not matched, report "indicative runtime" and not "speed-up".

## Common Failure Modes

| Failure | Symptom | Likely Cause | Diagnostic | Fix |
|---------|---------|--------------|------------|-----|
| Early-time error mistaken for skill | Low disagreement at early steps for any model | Microstructure has barely evolved; the persistence baseline is equally good | Compute persistence disagreement and gain over persistence | Evaluate where persistence disagreement is large; report gain, not raw error |
| Label permutation counted as error | Near-100% disagreement despite visually correct fields | Arbitrary grain ids in model output | Hungarian IoU matching before scoring | Always match grains before label metrics |
| Premature or late extinctions hidden by field MAE | Mean error small; survivor set or extinction order wrong | Small-grain kinetics poorly learned; error concentrated near vanishing grains | Signed per-grain extinction errors and survivor F1 | Make event metrics acceptance criteria; examine small-grain regime |
| Snapshot-cadence bias in timing | Extinction systematically "late" by up to one save interval | First-below-threshold frame used as event time | Compare bracketing against fitted A(t) extrapolation | Fit area decay or save more frequently near events |
| Spurious grain nucleation | New label appears or extinct grain returns | Admissibility map, fallback to uniform state, or rollout instability | Track label set and fragments per grain through time | Treat as hard failure; inspect update/projection map |
| Admissibility drift | Unit-sum error grows over rollout | float32 accumulation or non-exact projection | Plot max unit-sum residual per step on log axis | Enforce in state update; evaluate in float64 |
| Unconverged reference | Model "error" changes when reference is refined | Reference resolution/time step too coarse | Reference refinement study on the same observables | Converge reference before scoring |
| Development cases reported as prospective | Generalization overstated | Case inspected during design or threshold choice | Audit case history and spec timestamps | Re-label as development; collect new prospective cases |
| Pooled cohorts | Pass rate looks higher than any single cohort | Training, development, and prospective cases mixed | Report per-cohort k/n | Never pool evidence tiers |
| Unfair speed-up | Large claimed acceleration | Different hardware, accuracy level, or excluded training cost | Audit hardware, precision, warm-up, training hours | Match accuracy and hardware or report as indicative runtime |
| Energy rise in curvature flow | Boundary measure increases over steps | Non-physical update or oscillation | Monotonicity check on boundary measure/free energy | Investigate time stepping or rate map; reject rollout |

## Best Practices

- Freeze the benchmark specification, thresholds, and checkpoint hash before scoring held-out cases. Changing them afterwards demotes those cases to development evidence.
- Always report a persistence baseline. Errors measured before the microstructure has evolved say nothing about the model.
- Match grains with Hungarian IoU before any label-based metric. Report missed and spurious grains explicitly.
- Make event metrics (survivor set, extinction order, signed timing errors) first-class acceptance criteria next to field error.
- Measure admissibility at every saved step, even for architectures that enforce it by construction.
- Converge the reference solver first. Agreement with an unconverged reference is not evidence.
- Separate the evidence tiers (training case, past-horizon rollout, development, prospective unseen, stress) and never pool them.
- Use interval estimates for cohort pass rates. Small cohorts cannot establish reliability.
- State the scope explicitly: phase count, resolution, isotropy, mobility law, and dimensionality. No transfer outside the tested families is implied.
- Report runtime as indicative unless hardware, precision, accuracy level, and training cost are matched and measured.

## Integration with Other Skills

- **[phase-field-workflow](../phase-field-workflow/SKILL.md):** Supplies the reference phase-field models, convergence practice, and microstructure observables that this protocol compares against.
- **[pinns-mpf-workflow](../pinns-mpf-workflow/SKILL.md):** Coordinate-network PINN solutions of multiphase-field problems should be scored with the grain, interface, and admissibility metrics defined here.
- **[pinn-phase-neural-integrator](../pinn-phase-neural-integrator/SKILL.md):** Neural time integrators with long autonomous rollouts need the prospective-cohort, extinction-timing, and persistence-baseline protocol in this skill.
- **[pinns-scientific-ml](../../scientific-ml/pinns-scientific-ml/SKILL.md):** Provides general PINN validation practice (held-out residuals, loss-component tracking); this skill adds microstructure-specific event and topology metrics.
- **[surrogate-validation](../../surrogate-active-learning/surrogate-validation/SKILL.md):** Leakage-aware splitting and uncertainty reporting apply when microstructure realizations are grouped into training and test cohorts.
- **[scientific-visualization](../../scientific-data/scientific-visualization/SKILL.md):** Label-map overlays, error maps, area-versus-time curves, and extinction timelines should follow reproducible figure conventions.
- **[reproducibility-workflow](../../scientific-data/reproducibility-workflow/SKILL.md):** Benchmark specifications, checkpoint hashes, reference fields, and scoring scripts are provenance artifacts that must be versioned.

## Key References

1. Mullins, W. W. (1956). *Two-dimensional motion of idealized grain boundaries.* Journal of Applied Physics, 27, 900-904. https://doi.org/10.1063/1.1722511
2. MacPherson, R. D., & Srolovitz, D. J. (2007). *The von Neumann relation generalized to coarsening of three-dimensional microstructures.* Nature, 446, 1053-1055. https://doi.org/10.1038/nature05745
3. Steinbach, I., & Pezzolla, F. (1999). *A generalized field method for multiphase transformations using interface fields.* Physica D: Nonlinear Phenomena, 134, 385-393. https://doi.org/10.1016/S0167-2789(99)00129-3
4. Moelans, N., Blanpain, B., & Wollants, P. (2008). *Quantitative analysis of grain boundary properties in a generalized phase field model for grain growth in anisotropic systems.* Physical Review B, 78, 024113. https://doi.org/10.1103/PhysRevB.78.024113
5. Krill III, C. E., & Chen, L.-Q. (2002). *Computer simulation of 3-D grain growth using a phase-field model.* Acta Materialia, 50, 3059-3075. https://doi.org/10.1016/S1359-6454(02)00084-8
6. Hillert, M. (1965). *On the theory of normal and abnormal grain growth.* Acta Metallurgica, 13, 227-238. https://doi.org/10.1016/0001-6160(65)90200-2
7. Kuhn, H. W. (1955). *The Hungarian method for the assignment problem.* Naval Research Logistics Quarterly, 2, 83-97. https://doi.org/10.1002/nav.3800020109
8. Elfetni, S., Thinius, S., Seeberger, P. H., & Tyrikos-Ergas, T. (2026). *PINN-Phase: A physics-informed neural network for curvature-driven multiphase-field evolution.* arXiv preprint (not peer reviewed). https://arxiv.org/abs/2608.29413
