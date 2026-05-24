---
name: scientific-visualization
domain: scientific-data
maturity: draft
tools:
  - matplotlib
  - OVITO
  - ASE
  - pymatgen
  - VESTA
  - nglview
  - VMD
tested_versions:
  matplotlib: "3.8.0"
  OVITO: "3.10.6"
  ASE: "3.23.0"
  pymatgen: "2024.1.26"
  VESTA: "3.5.8"
  nglview: "3.0.8"
  VMD: "1.9.4"
requires_gpu: false
estimated_runtime: minutes-to-hours
---

# Scientific Visualization

## Description

This skill covers publication-quality visualization for computational materials science, chemistry, atomistic simulation, electronic structure, and scientific ML: plotting scalar observables with matplotlib, rendering atomic structures and trajectories with OVITO, ASE, pymatgen, VESTA, nglview, and VMD, and building reproducible figure pipelines for papers, talks, reports, and model diagnostics. Invoke this skill when converting validated simulation or ML results into figures, movies, notebooks, or report artifacts, or when auditing whether a visualization could mislead through units, color scales, cherry-picked frames, or non-reproducible preprocessing.

## Domain Context

Scientific visualization is not decoration; it is a measurement interface. A plot or rendering encodes choices about units, normalization, sampling, smoothing, color scale, camera angle, periodic wrapping, and what data were excluded. In computational materials science, those choices can change the scientific conclusion as strongly as a simulation parameter.

Atomistic visualizations depend on the representation of coordinates and topology. A crystal structure can be shown as a conventional cell, primitive cell, supercell, slab, molecule, interface, or trajectory frame. Wrapped coordinates may make a diffusing ion appear stationary; unwrapped coordinates may make a compact periodic crystal appear split across cell boundaries. A surface slab must show vacuum and exposed facets clearly enough that viewers can distinguish surface atoms from bulk-like atoms. A defect rendering must identify the reference lattice, otherwise a colored per-atom scalar field can be mistaken for a defect classification.

Scalar plots have their own physical constraints. RDFs should approach 1 at large distance for homogeneous liquids; MSDs should be plotted on log-log and linear axes when diffusion regimes matter; temperature and pressure traces require equilibration trimming before averages are annotated; parity plots for ML potentials should preserve equal aspect ratio so slope errors are visible. Band structures and DOS require a consistent energy zero, usually the Fermi level or valence-band maximum. Phonon dispersions require negative frequencies to be displayed rather than hidden, because imaginary modes are often the result.

Visualization workflows must also be reproducible. A figure script should read explicit input files, record versions, set deterministic style choices, save vector formats when appropriate, and export the data behind each plotted curve. Interactive notebooks are useful for exploration, but production figures should be regenerated from a script or pipeline step with no manual GUI edits unless the manual step is explicitly documented.

## When to Use This Skill

- Preparing publication-quality figures from MD, DFT, active learning, or MLP validation results.
- Rendering atomic structures, surfaces, defects, interfaces, amorphous systems, and trajectory movies for inspection or presentation.
- Plotting RDFs, MSDs, VACFs, density profiles, order parameters, energy drift, stress, temperature, and pressure traces after trajectory analysis.
- Visualizing electronic-structure outputs: band structures, DOS, PDOS, phonon dispersions, phonon DOS, charge density, ELF, and phase diagrams.
- Creating ML potential diagnostics: parity plots, residual distributions, force-error histograms, per-species errors, learning curves, uncertainty calibration plots, and outlier maps.
- Auditing existing figures for misleading axes, color maps, units, smoothing, aspect ratios, non-equilibrated data, or cherry-picked frames.

## When Not to Use This Skill

- **As a substitute for analysis.** A visually plausible trajectory, RDF, or parity plot does not validate a simulation. Use the relevant analysis and validation skills first, then visualize the resulting quantities.
- **To hide failed calculations.** Negative phonon branches, energy drift, unconverged pressure traces, or high-uncertainty active-learning frames should be shown and diagnosed, not clipped away.
- **When the plotted data are not provenance-tracked.** Figures generated from ad hoc notebooks, implicit glob patterns, or manually renamed files cannot be reproduced reliably.
- **For quantitative conclusions from a single rendered frame.** Static images are useful for communication, but structural claims require measurements over frames, configurations, or replicas.
- **When interactive GUI state is the only record.** OVITO, VESTA, and VMD are excellent inspection tools, but publication figures should save the scene file, camera state, modifier stack, and data pipeline.

## Core Concepts

- **Data-to-ink mapping:** Every color, marker, line, axis limit, and glyph must correspond to a documented variable or visual encoding. Decorative encodings are risky in scientific plots because readers infer meaning from them.
- **Coordinate representation:** Wrapped, unwrapped, centered, conventional-cell, primitive-cell, and supercell representations are not interchangeable. The representation used for visualization must match the scientific point of the figure.
- **Aspect ratio:** Equal-axis parity plots reveal systematic slope and offset errors; distorted axes can hide bias. Atomic renderings should preserve geometry unless a deliberately exaggerated dimension is labeled.
- **Color scale:** Sequential maps for scalar magnitude, diverging maps for signed quantities, categorical palettes for species or phases. Never use a diverging map for a strictly positive error unless the center has physical meaning.
- **Normalization:** DOS, RDF, histograms, phonon DOS, and error distributions can be normalized by count, atom, volume, bin width, or probability density. The caption and axis label must state the convention.
- **Equilibration trimming:** Production plots should exclude equilibration unless the figure is explicitly about equilibration. Time-series panels should mark the discarded region when showing the full run.
- **Uncertainty visualization:** Error bars, confidence bands, ensemble spread, block-averaged standard errors, and calibration curves communicate different uncertainties. Use the one that matches the statistical object.
- **Vector vs. raster export:** Line plots, text, and schematics should usually be SVG or PDF; dense heatmaps and rendered structures are often PNG or TIFF. Keep the data and script alongside the exported image.
- **Reproducible figure script:** A script that can regenerate a figure from declared inputs, software versions, style settings, and output paths. Notebook cells can be exploratory, but final figures should be scriptable.

## Key Workflows

### Workflow 1: Publication-quality matplotlib figures from simulation observables

Build a reusable plotting layer for MD observables and DFT/ML scalar outputs. The example assumes that analysis has already produced CSV files with explicit units.

```python
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


PAPER_STYLE = {
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.linewidth": 0.8,
}


def setup_matplotlib_style() -> None:
    """Set deterministic style defaults for publication figures."""
    plt.rcParams.update(PAPER_STYLE)
    plt.rcParams["svg.fonttype"] = "none"  # Keep text editable in vector editors
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42


def plot_md_timeseries_panel(
    thermo_csv: str,
    output_prefix: str,
    equilibration_time_ps: float,
) -> None:
    """
    Plot temperature, pressure, and total energy traces from a production MD run.

    thermo_csv columns:
        time_ps, temperature_K, pressure_GPa, total_energy_eV
    The equilibration region is shaded rather than silently removed.
    """
    setup_matplotlib_style()
    df = pd.read_csv(thermo_csv)
    required = {"time_ps", "temperature_K", "pressure_GPa", "total_energy_eV"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {thermo_csv}: {sorted(missing)}")

    prod = df[df["time_ps"] >= equilibration_time_ps]
    if len(prod) < 10:
        raise ValueError("Too few production points after equilibration trimming")

    fig, axes = plt.subplots(3, 1, figsize=(3.35, 4.2), sharex=True)
    series = [
        ("temperature_K", "Temperature (K)", "#1f77b4"),
        ("pressure_GPa", "Pressure (GPa)", "#2ca02c"),
        ("total_energy_eV", "Total energy (eV)", "#d62728"),
    ]

    for ax, (col, ylabel, color) in zip(axes, series):
        ax.plot(df["time_ps"], df[col], lw=0.7, color=color)
        ax.axvspan(df["time_ps"].min(), equilibration_time_ps,
                   color="0.85", alpha=0.8, lw=0)
        mean = prod[col].mean()
        ax.axhline(mean, color="black", lw=0.7, ls="--")
        ax.set_ylabel(ylabel)
        ax.margins(x=0)

    axes[-1].set_xlabel("Time (ps)")
    fig.align_ylabels(axes)

    out = Path(output_prefix)
    fig.savefig(out.with_suffix(".pdf"))
    fig.savefig(out.with_suffix(".png"))
    plt.close(fig)

    metadata = {
        "input": thermo_csv,
        "equilibration_time_ps": equilibration_time_ps,
        "n_total": int(len(df)),
        "n_production": int(len(prod)),
        "outputs": [str(out.with_suffix(".pdf")), str(out.with_suffix(".png"))],
    }
    out.with_suffix(".figure.json").write_text(json.dumps(metadata, indent=2))


def plot_rdf_with_coordination_window(
    rdf_csv: str,
    output_file: str,
    r_min_A: float,
    r_max_A: float,
) -> None:
    """
    Plot an RDF and highlight the integration window used for coordination number.

    rdf_csv columns:
        r_A, g_r
    """
    setup_matplotlib_style()
    df = pd.read_csv(rdf_csv)
    fig, ax = plt.subplots(figsize=(3.35, 2.25))
    ax.plot(df["r_A"], df["g_r"], color="#1f77b4", lw=1.1)
    ax.axhline(1.0, color="0.4", ls=":", lw=0.8)
    ax.axvspan(r_min_A, r_max_A, color="#ffbf00", alpha=0.25, lw=0)
    ax.set_xlabel("r (Angstrom)")
    ax.set_ylabel("g(r)")
    ax.set_xlim(0, df["r_A"].max())
    ax.set_ylim(bottom=0)
    fig.savefig(output_file)
    plt.close(fig)
```

### Workflow 2: OVITO rendering for structures, defects, and trajectories

Use OVITO's Python API to create reproducible scene files and rendered images. This is preferred over manual GUI screenshots for publication figures because modifier settings and camera state are explicit.

```python
from pathlib import Path
from ovito.io import import_file
from ovito.modifiers import (
    CommonNeighborAnalysisModifier,
    CoordinationAnalysisModifier,
    ExpressionSelectionModifier,
)
from ovito.vis import TachyonRenderer, Viewport


def render_defect_snapshot(
    trajectory_file: str,
    output_png: str,
    frame: int = -1,
    defect_property: str = "Centrosymmetry",
    threshold: float = 4.0,
) -> None:
    """
    Render a defect-rich atomistic snapshot with a reproducible OVITO pipeline.

    trajectory_file may be a LAMMPS dump, extxyz, or another OVITO-supported file.
    The threshold is system- and temperature-dependent; inspect reference perfect
    crystals before treating selected atoms as defects. [EXPERT REVIEW NEEDED]
    """
    pipeline = import_file(trajectory_file, multiple_frames=True)

    # For crystalline metals, CNA gives an interpretable background phase label.
    pipeline.modifiers.append(CommonNeighborAnalysisModifier())

    # Select atoms above a scalar threshold if the property exists in the file.
    pipeline.modifiers.append(ExpressionSelectionModifier(
        expression=f"{defect_property} > {threshold}"
    ))

    data = pipeline.compute(frame)
    n_selected = int(data.attributes.get("ExpressionSelection.count", 0))
    print(f"Selected {n_selected} atoms with {defect_property} > {threshold}")

    pipeline.add_to_scene()
    vp = Viewport(type=Viewport.Type.Perspective)
    vp.camera_pos = (80, -100, 70)
    vp.camera_dir = (-0.55, 0.70, -0.45)
    vp.fov = 0.55
    vp.zoom_all(size=(1800, 1300))

    renderer = TachyonRenderer(shadows=True, ambient_occlusion=True)
    vp.render_image(
        filename=output_png,
        size=(1800, 1300),
        frame=frame,
        renderer=renderer,
    )

    scene_file = Path(output_png).with_suffix(".ovito")
    pipeline.save(str(scene_file))
    print(f"Rendered {output_png}; saved OVITO pipeline {scene_file}")


def export_rdf_from_ovito(
    trajectory_file: str,
    output_txt: str,
    cutoff_A: float = 8.0,
    n_bins: int = 200,
) -> None:
    """Use OVITO's CoordinationAnalysisModifier as an independent RDF cross-check."""
    pipeline = import_file(trajectory_file, multiple_frames=True)
    pipeline.modifiers.append(CoordinationAnalysisModifier(
        cutoff=cutoff_A,
        number_of_bins=n_bins,
        partial=True,
    ))
    data = pipeline.compute()
    rdf = data.tables["coordination-rdf"]
    rdf.xy().tofile(output_txt, sep=" ")
```

### Workflow 3: Electronic-structure figures with consistent energy references

Band structures, DOS, PDOS, phonon dispersions, and phase diagrams must carry enough metadata that another person can reproduce the energy zero, path, labels, and units.

```python
from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pymatgen.electronic_structure.plotter import BSPlotter, DosPlotter
from pymatgen.io.vasp.outputs import Vasprun, BSVasprun


def plot_vasp_band_dos(
    band_vasprun_xml: str,
    dos_vasprun_xml: str,
    output_prefix: str,
    energy_window_eV: tuple[float, float] = (-6, 4),
) -> None:
    """
    Plot VASP band structure and DOS with the Fermi level as zero.

    The line-mode band calculation and DOS calculation must use compatible
    functional, POTCAR, smearing, and spin settings. Mixing inconsistent runs
    can create artificial offsets or missing states.
    """
    bs_run = BSVasprun(band_vasprun_xml, parse_projected_eigen=False)
    dos_run = Vasprun(dos_vasprun_xml, parse_dos=True)

    bs = bs_run.get_band_structure(line_mode=True, efermi=dos_run.efermi)
    dos = dos_run.complete_dos

    fig, axes = plt.subplots(1, 2, figsize=(6.7, 3.0),
                             gridspec_kw={"width_ratios": [2.2, 1.0]})

    bs_plotter = BSPlotter(bs)
    bs_data = bs_plotter.bs_plot_data()
    for branch in bs_data["distances"]:
        for band in bs_data["energy"]["1"]:
            axes[0].plot(branch, band[:len(branch)], color="black", lw=0.6)
    axes[0].axhline(0, color="0.5", lw=0.8, ls="--")
    axes[0].set_ylim(*energy_window_eV)
    axes[0].set_ylabel("Energy - E_F (eV)")
    axes[0].set_xlabel("Wave vector")

    dos_plotter = DosPlotter(zero_at_efermi=True)
    dos_plotter.add_dos("Total DOS", dos)
    dos_data = dos_plotter.get_dos_dict()["Total DOS"]
    axes[1].plot(dos_data["densities"]["1"], dos_data["energies"], color="#1f77b4", lw=0.9)
    axes[1].axhline(0, color="0.5", lw=0.8, ls="--")
    axes[1].set_ylim(*energy_window_eV)
    axes[1].set_xlabel("DOS")
    axes[1].set_yticklabels([])

    fig.tight_layout()
    out = Path(output_prefix)
    fig.savefig(out.with_suffix(".pdf"))
    fig.savefig(out.with_suffix(".png"), dpi=300)
    plt.close(fig)

    metadata = {
        "band_vasprun_xml": band_vasprun_xml,
        "dos_vasprun_xml": dos_vasprun_xml,
        "energy_zero": "Fermi level from DOS vasprun.xml",
        "efermi_eV": float(dos_run.efermi),
        "energy_window_eV": list(energy_window_eV),
    }
    out.with_suffix(".figure.json").write_text(json.dumps(metadata, indent=2))
```

### Workflow 4: MLP validation and active-learning diagnostics

ML potential figures should reveal systematic errors, not only aggregate MAE. Always combine parity plots with residual distributions and per-species or per-configuration breakdowns.

```python
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_force_parity_and_residuals(
    csv_file: str,
    output_file: str,
) -> None:
    """
    Plot force parity and residual diagnostics for an MLP validation set.

    csv_file columns:
        species, config_type, f_ref_eV_A, f_pred_eV_A
    One row per Cartesian force component is acceptable and recommended.
    """
    df = pd.read_csv(csv_file)
    required = {"species", "config_type", "f_ref_eV_A", "f_pred_eV_A"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    df["residual_meV_A"] = 1000 * (df["f_pred_eV_A"] - df["f_ref_eV_A"])
    mae = df["residual_meV_A"].abs().mean()
    p95 = df["residual_meV_A"].abs().quantile(0.95)

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4))

    lim = np.nanmax(np.abs(df[["f_ref_eV_A", "f_pred_eV_A"]].to_numpy()))
    axes[0].scatter(df["f_ref_eV_A"], df["f_pred_eV_A"],
                    s=3, alpha=0.25, color="#1f77b4", rasterized=True)
    axes[0].plot([-lim, lim], [-lim, lim], color="black", lw=0.8)
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].set_xlim(-lim, lim)
    axes[0].set_ylim(-lim, lim)
    axes[0].set_xlabel("DFT force (eV/Angstrom)")
    axes[0].set_ylabel("MLP force (eV/Angstrom)")

    axes[1].hist(df["residual_meV_A"], bins=100, color="#4c78a8", alpha=0.85)
    axes[1].axvline(0, color="black", lw=0.8)
    axes[1].set_xlabel("Residual (meV/Angstrom)")
    axes[1].set_ylabel("Count")
    axes[1].set_title(f"MAE={mae:.1f}, P95={p95:.1f}")

    species_mae = df.groupby("species")["residual_meV_A"].apply(lambda x: x.abs().mean())
    axes[2].bar(species_mae.index.astype(str), species_mae.values, color="#59a14f")
    axes[2].set_ylabel("Force MAE (meV/Angstrom)")
    axes[2].set_xlabel("Species")
    axes[2].tick_params(axis="x", rotation=45)

    fig.tight_layout()
    fig.savefig(output_file)
    plt.close(fig)


def plot_active_learning_history(
    history_csv: str,
    output_file: str,
) -> None:
    """
    Plot active-learning query counts, validation error, and uncertainty coverage.

    history_csv columns:
        iteration, n_train, n_queried, force_mae_meV_A,
        max_committee_std_meV_A, extrapolation_fraction
    """
    df = pd.read_csv(history_csv)
    fig, axes = plt.subplots(3, 1, figsize=(3.35, 4.4), sharex=True)

    axes[0].plot(df["iteration"], df["n_train"], "o-", color="#1f77b4")
    axes[0].bar(df["iteration"], df["n_queried"], alpha=0.25, color="#1f77b4")
    axes[0].set_ylabel("Structures")

    axes[1].plot(df["iteration"], df["force_mae_meV_A"], "o-", color="#d62728")
    axes[1].set_ylabel("Force MAE\n(meV/Angstrom)")

    axes[2].plot(df["iteration"], 100 * df["extrapolation_fraction"],
                 "o-", color="#9467bd")
    axes[2].set_ylabel("Extrapolated\nframes (%)")
    axes[2].set_xlabel("Active-learning iteration")

    fig.tight_layout()
    fig.savefig(output_file)
    plt.close(fig)
```

### Workflow 5: Structure inspection with ASE, pymatgen, VESTA, nglview, and VMD

Use lightweight Python renderers for quick checks and specialized viewers for domain-specific inspection. Export structures in explicit formats so GUI operations can be repeated.

```python
from pathlib import Path
from ase.io import read, write
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor


def prepare_structure_visualization_bundle(
    structure_file: str,
    output_dir: str,
    make_supercell: tuple[int, int, int] = (2, 2, 2),
) -> dict:
    """
    Prepare a reproducible visualization bundle for a crystal, slab, or interface.

    Outputs:
        structure.cif       for VESTA and pymatgen
        structure.xyz       for VMD/nglview quick inspection
        supercell.cif       for periodic context
        visualization.json  records conversion choices
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    atoms = read(structure_file)
    structure = AseAtomsAdaptor.get_structure(atoms)

    cif_path = out / "structure.cif"
    xyz_path = out / "structure.xyz"
    supercell_path = out / "supercell.cif"

    structure.to(filename=str(cif_path))
    write(xyz_path, atoms)

    supercell = structure.copy()
    supercell.make_supercell(make_supercell)
    supercell.to(filename=str(supercell_path))

    metadata = {
        "input": structure_file,
        "n_sites": len(structure),
        "formula": structure.composition.reduced_formula,
        "cell_lengths_A": list(map(float, structure.lattice.abc)),
        "cell_angles_deg": list(map(float, structure.lattice.angles)),
        "supercell": list(make_supercell),
        "notes": [
            "Use structure.cif for VESTA publication rendering.",
            "Use supercell.cif when local coordination or defects need periodic context.",
            "Use structure.xyz for VMD/nglview quick inspection; XYZ does not preserve full periodic metadata.",
        ],
    }
    (out / "visualization.json").write_text(__import__("json").dumps(metadata, indent=2))
    return metadata
```

## Common Failure Modes

| Failure | Symptom | Likely Cause | Diagnostic | Fix |
|---------|---------|--------------|------------|-----|
| Cherry-picked trajectory frame | A figure shows an impressive event that is absent from most of the trajectory; average observables do not support the visual claim | One frame selected manually from a long run without reporting selection criteria | Compare the rendered frame index to the time series and event log; sample multiple frames before and after the event | State frame time and selection criterion; include event frequency or representative-frame sampling |
| Wrapped trajectory artifact | Molecule, defect, or diffusing ion appears broken or stationary at a periodic boundary | Wrapped coordinates used for displacement-sensitive visualization | Render both wrapped and unwrapped coordinates; inspect image flags or trajectory unwrap metadata | Center and unwrap trajectories for movies; use wrapped coordinates only for unit-cell structural views |
| Misleading color scale | Charge density, ELF, error map, or per-atom scalar appears to support a contrast that disappears with fixed limits | Per-panel autoscaling or non-physical color-map center | Print min/max per panel; compare with a globally fixed color scale | Use shared limits across comparable panels; use diverging maps only for signed quantities with meaningful zero |
| Inconsistent units | Energy drift, force residual, pressure, or DOS axis differs by factors of 10, 1000, or volume | Mixed LAMMPS real/metal units, eV vs meV, Angstrom vs nm, or DOS per cell vs per atom | Check metadata and labels against raw output units; reproduce one value manually | Convert in a preprocessing step and encode units in column names and axis labels |
| Unequilibrated data plotted as production | Temperature, volume, or order parameter trend dominates the plotted average | Equilibration frames included in production plot | Shade discarded region and recompute averages after trimming | Mark equilibration explicitly; use only production frames for reported means |
| Parity plot hides bias | MLP predictions look excellent but residuals show slope or offset error | Unequal x/y aspect ratio, overplotting, or missing identity line | Force equal aspect ratio; bin residuals by target value and species | Use equal axes, identity line, residual histogram, and per-species panels |
| Oversmoothed distributions | RDF peaks, residual tails, or phase fractions look cleaner than the underlying data | Kernel bandwidth or rolling-window size too large | Plot raw histogram or unsmoothed trace underneath | Report smoothing window; choose physically justified bin width or bandwidth |
| Non-reproducible GUI figure | A paper figure cannot be regenerated after the OVITO/VESTA/VMD session is closed | Manual camera, color, and modifier settings were never saved | Search for saved scene files or scripts; compare exported image timestamp with data provenance | Save OVITO pipelines, VESTA state files when possible, camera settings, and figure scripts alongside outputs |
| Hidden failed DFT modes | Phonon plot has no imaginary frequencies because negative branches were clipped | Axis limits or preprocessing removed negative frequencies | Inspect raw phonon output and minimum frequency before plotting | Show imaginary branches explicitly; annotate unstable modes instead of omitting them |

## Best Practices

- **Treat figure generation as part of the computational workflow.** Store plotting scripts, figure metadata, and input data in the same provenance system as simulations and models.
- **Encode units in column names before plotting.** A DataFrame column named `force_mae_meV_A` is harder to misuse than a generic `force_error`.
- **Use equal aspect ratio for parity plots and geometric comparisons.** Distorted axes are one of the easiest ways to hide systematic model bias.
- **Show equilibration and discarded regions when plotting time series.** This makes the analysis decision visible and prevents accidental averaging over transients.
- **Prefer perceptually uniform color maps for scalar fields.** Use `viridis`, `magma`, `cividis`, or carefully chosen diverging maps; avoid rainbow maps for quantitative scalar interpretation.
- **Export both vector and raster formats for line figures.** PDF/SVG are appropriate for manuscript assembly; PNG is useful for quick review and web reports.
- **Save the data behind every figure.** A figure should never be the only copy of binned RDF values, residual histograms, or phase fractions.
- **For structure renderings, include enough context.** Defects, surfaces, and interfaces need neighboring atoms, cell outlines, or reference markers so the viewer can interpret the local environment.
- **Audit color scales across panels.** Side-by-side comparisons require shared limits unless the caption explicitly says otherwise.
- **Avoid plotting every point when it destroys the message.** For million-point force parity plots, rasterized scatter or density maps are preferable, but residual tails and outliers must still be visible.

## Integration with Other Skills

- **[md-analysis](../../atomistic-md/md-analysis/SKILL.md):** Provides the RDF, MSD, VACF, thermodynamic traces, order parameters, and uncertainty estimates that this skill turns into figures. Visualization should not recompute these quantities differently from the analysis workflow.
- **[lammps-workflow](../../atomistic-md/lammps-workflow/SKILL.md):** LAMMPS dump fields, `thermo_style` output, unit style, and image flags determine whether trajectory renderings and time-series plots have the data needed for correct visualization.
- **[gromacs-workflow](../../atomistic-md/gromacs-workflow/SKILL.md):** GROMACS `.xtc`, `.trr`, `.edr`, and `.gro` outputs feed trajectory movies and thermodynamic plots. This skill supplies plotting conventions for outputs parsed from GROMACS tools or MDAnalysis.
- **[md-equilibration](../../atomistic-md/md-equilibration/SKILL.md):** Equilibration diagnostics define which frames are safe to show as production data. Figures should mark or exclude unequilibrated regions using the criteria from that skill.
- **[mlp-validation](../../ml-interatomic-potentials/mlp-validation/SKILL.md):** MLP validation produces parity data, residuals, NVE drift traces, phonon comparisons, and EOS curves. This skill provides the figure patterns that make those diagnostics interpretable.
- **[active-learning-mlp](../../ml-interatomic-potentials/active-learning-mlp/SKILL.md):** Active-learning campaigns require visualizing acquisition history, committee disagreement, descriptor coverage, and selected DFT queries. The plots here are reporting layers over that campaign state.
- **[materials-featurization](../materials-featurization/SKILL.md):** Descriptor and latent-space visualizations depend on the featurizer settings, train/test split, and leakage checks established in that skill.
- **[reproducibility-workflow](../reproducibility-workflow/SKILL.md):** Figure scripts, exported images, scene files, and source data should be versioned and hashed as reproducible artifacts.
- **[pymatgen-analysis](../pymatgen-analysis/SKILL.md):** pymatgen-derived band structures, DOS objects, phase diagrams, and structure transformations are common inputs to electronic-structure figures. This is an intentional forward reference until that skill is added.
- **[ase-framework](../ase-framework/SKILL.md):** ASE provides structure conversion, quick trajectory inspection, and integration with notebooks. This is an intentional forward reference until that skill is added.
- **[dft-phonons](../../electronic-structure/dft-phonons/SKILL.md):** Phonon dispersions and phonon DOS require plotting conventions that expose imaginary modes, q-path labels, and acoustic sum-rule choices. This is an intentional forward reference until that skill is added.
- **[dft-surfaces](../../electronic-structure/dft-surfaces/SKILL.md):** Surface and adsorption figures require slab orientation, vacuum, coverage, and active-site annotations. This is an intentional forward reference until that skill is added.
- **[dft-defects](../../electronic-structure/dft-defects/SKILL.md):** Defect visualizations require charge state, supercell context, local relaxation, and reference-lattice comparison. This is an intentional forward reference until that skill is added.

## Key References

1. Hunter, J. D. (2007). *Matplotlib: A 2D graphics environment.* Computing in Science & Engineering, 9(3), 90-95. https://doi.org/10.1109/MCSE.2007.55
2. Stukowski, A. (2010). *Visualization and analysis of atomistic simulation data with OVITO - the Open Visualization Tool.* Modelling and Simulation in Materials Science and Engineering, 18(1), 015012. https://doi.org/10.1088/0965-0393/18/1/015012
3. Momma, K., & Izumi, F. (2011). *VESTA 3 for three-dimensional visualization of crystal, volumetric and morphology data.* Journal of Applied Crystallography, 44, 1272-1276. https://doi.org/10.1107/S0021889811038970
4. Humphrey, W., Dalke, A., & Schulten, K. (1996). *VMD: Visual molecular dynamics.* Journal of Molecular Graphics, 14(1), 33-38. https://doi.org/10.1016/0263-7855(96)00018-5
5. Ong, S. P., et al. (2013). *Python Materials Genomics (pymatgen): A robust, open-source python library for materials analysis.* Computational Materials Science, 68, 314-319. https://doi.org/10.1016/j.commatsci.2012.10.028
