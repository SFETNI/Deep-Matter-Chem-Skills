---
name: enhanced-sampling
domain: atomistic-md
maturity: draft
tools:
  - PLUMED
  - GROMACS
  - LAMMPS
  - MDAnalysis
  - numpy
  - matplotlib
tested_versions:
  PLUMED: "2.9.2"
  GROMACS: "2024.2"
  LAMMPS: "2Aug2023"
  MDAnalysis: "2.7.0"
requires_gpu: false
estimated_runtime: hours-to-weeks
---

# Enhanced Sampling

## Description

This skill covers enhanced sampling methods for atomistic MD simulations where rare events, large free-energy barriers, or slow conformational transitions prevent standard MD from converging on accessible timescales. It addresses collective variable (CV) selection and validation; umbrella sampling with WHAM/MBAR reconstruction; well-tempered metadynamics; adaptive biasing force; steered MD; temperature replica exchange; accelerated MD; free-energy surface convergence diagnostics; and PLUMED integration with GROMACS, LAMMPS, and ML interatomic potentials. Invoke this skill whenever a process of interest has an estimated barrier ≳ 10 k_B T or an estimated timescale that cannot be sampled directly within the available simulation budget.

## Domain Context

Standard MD is limited to continuous trajectories. For a system at 300 K, barrier-crossing events with ΔG‡ ≳ 10 k_B T (≳ 25 kJ/mol) occur with rates of order k ~ ν exp(−ΔG‡/k_B T) ≈ 10¹³ × e⁻¹⁰ s⁻¹ ≈ 10⁻⁸ s⁻¹ — a mean first-passage time of ~100 ns. Protein folding, crystal nucleation, ion permeation through channels, and solid-state diffusion all fall into this regime.

Enhanced sampling methods modify the system to accelerate barrier crossing and then reconstruct the unbiased free energy surface (FES) from the biased simulation. The modifications take four forms:

1. **Bias potentials on collective variables (CVs):** A time-varying or history-dependent potential V_bias(ξ) is added along one or more CVs ξ(r). The CV must capture the reaction coordinate — it must separate the metastable states and project out the slow degrees of freedom. A poor CV does not accelerate the relevant transition even if the method runs correctly.
2. **Hamiltonian or temperature ladders (replica exchange):** Multiple copies of the system run in parallel at different temperatures or with different Hamiltonians. Exchanges between replicas are accepted with the Metropolis criterion, enabling high-temperature configurations to escape local minima and donate configurations to lower-temperature replicas.
3. **Time-independent bias (flat histogram):** The WAN to-WHAM or MBAR approach reconstructs the FES from multiple biased simulations (umbrella sampling windows) by reweighting each frame by the inverse of its bias weight.
4. **Velocity-boosted or modified-potential approaches (accelerated MD):** A dihedral or total boost potential is added to reduce barriers globally rather than along a specific CV.

Several non-negotiable physical constraints govern applicability:

- **CV completeness:** The CV must have low friction along all relevant slow modes. Hidden slow modes — degrees of freedom orthogonal to the CV that also relax slowly — cause hysteresis and non-reproducible FES estimates even when the CV-projected FES looks converged.
- **Reweightability:** The bias must be expressible as a function of the current configuration, not dependent on the trajectory history, for exact reweighting. Metadynamics with a Gaussian hill width smaller than the thermal fluctuation of the CV is not accurately reweightable frame-by-frame; use the time-integrated bias for reweighting.
- **Ergodicity on the biased surface:** Enhanced sampling does not solve the ergodicity problem; it shifts it. A metadynamics run that floods a two-minima system will still fail if a third metastable state exists along an orthogonal CV that is not biased.
- **Timescale separation:** Replica exchange requires the fast motions (solvent, bond vibrations) to decorrelate within each replica's simulation between swaps. The exchange interval must be long enough that each replica is ergodic on its FES segment between swaps. [EXPERT REVIEW NEEDED]

## When to Use This Skill

- When the process of interest (folding, unbinding, nucleation, diffusion, isomerization, phase transition) is not observed in a plain MD run of the affordable length.
- When a free energy difference or free energy profile along a reaction coordinate is needed quantitatively.
- When screening multiple candidate reaction pathways and the correct pathway is unknown a priori.
- When using ML potentials whose speed advantage over DFT allows access to nanosecond trajectories, but the barrier is still too large for plain MLP-MD to converge.
- When sampling thermally activated defect migration, adsorption/desorption, or catalytic transitions where DFT-level accuracy is needed but timescales preclude AIMD.

## When Not to Use This Skill

- **When the process is not actually slow.** If the transition is observed multiple times in a 10 ns plain MD run, enhanced sampling adds complexity and potential bias without benefit. Verify the timescale problem before choosing a method.
- **When a good collective variable is unknown and cannot be constructed.** Automated CV discovery (TICA, sketch-map, deep TDA) is a prerequisite, not part of the sampling itself. Metadynamics with a wrong CV wastes compute and produces a meaningless FES.
- **For equilibrium structural observables (RDF, density, bond angles) that are not barrier-limited.** Enhanced sampling changes the statistical weights of sampled configurations; using it to compute structural averages requires exact reweighting and introduces additional error versus plain MD.
- **For non-equilibrium or kinetics studies requiring real-time dynamics.** Metadynamics and umbrella sampling destroy the real-time kinetics; use transition path sampling or milestoning if rate constants are the target quantity. [EXPERT REVIEW NEEDED]
- **For systems where the bias violates the relevant physics.** Biasing a bond length CV across a bond-breaking event with a classical force field is physically meaningless (the force field has no dissociation energy surface). Use ReaxFF or QM/MM for reactive transitions.

## Core Concepts

- **Collective variable (CV) / reaction coordinate:** A low-dimensional function ξ(r) of all atomic positions that distinguishes the relevant metastable states. Must be differentiable with respect to atomic positions (required for force projection). Common choices: distances, angles, torsions, RMSD to a reference, radius of gyration, coordination numbers, path CVs, TICA or machine-learned CVs. The quality of the CV determines the quality of the entire enhanced sampling calculation.
- **Free energy surface (FES):** F(ξ) = −k_B T ln P(ξ) + C, where P(ξ) is the probability density of CV values under the unbiased Boltzmann distribution. The FES encodes all thermodynamic information about the process: barrier heights, free energy differences between states, and relative stability of intermediates.
- **Metadynamics:** History-dependent bias V_meta(ξ, t) = Σ_k h exp(−(ξ − ξ_k)² / (2σ²)) accumulated by periodically depositing Gaussian hills at the current CV value. The bias eventually compensates the FES (V_meta → −F(ξ) at convergence). Raw metadynamics oscillates at convergence; well-tempered metadynamics corrects this.
- **Well-tempered metadynamics (WTM):** Hills are deposited with a height that decreases exponentially as the bias accumulates: h(t) = h₀ exp(−V_bias(ξ, t) / k_B ΔT), where ΔT is the bias temperature and γ = (T + ΔT)/T is the bias factor. WTM converges to F(ξ) × (T/(T+ΔT)) — a rescaled FES — rather than oscillating. The FES is recovered as F(ξ) = −(1 + T/ΔT) × V_bias(ξ, t→∞) / γ. Standard choice: γ = 10–15 for barriers up to 50–80 kJ/mol at 300 K. [EXPERT REVIEW NEEDED]
- **Umbrella sampling (US):** A set of independent simulations, each with a harmonic restraint V_US = (k/2)(ξ − ξ₀)² centered at a different point ξ₀ along the reaction coordinate. The windows together cover the entire pathway from reactant to product. The unbiased FES is reconstructed from all windows simultaneously by WHAM or MBAR. Requires overlapping window distributions; gaps in coverage produce discontinuities in the reconstructed FES.
- **WHAM (Weighted Histogram Analysis Method):** Iteratively solves a set of self-consistency equations to find the optimal combination of biased histograms that reconstructs P(ξ). Requires overlapping histograms and converges numerically. Superseded by MBAR for non-histogram-based reconstructions but still standard for simple 1D/2D umbrella sampling.
- **MBAR (Multistate Bennett Acceptance Ratio):** Statistically optimal estimator for free energy differences from multiple simulations with different biasing potentials. Does not require binning (no histogram); uses all frames with their exact bias weights. Implemented in PyMBAR (or FastMBAR). More accurate than WHAM for the same data, especially at low frame counts.
- **Adaptive biasing force (ABF):** At each CV value, the thermodynamic integration integrand ∂F/∂ξ is estimated from the instantaneous generalized force and accumulated; the accumulated mean force is applied as a counter-bias. Adaptively flattens the FES along the CV without prior knowledge of barrier shape. Implemented in COLVARS (NAMD, LAMMPS, GROMACS) and as `abf` in PLUMED.
- **Steered MD (SMD):** A moving harmonic spring pulls the CV from ξ_start to ξ_end at constant velocity. The work W performed by the spring along the trajectory is measured. The Jarzynski equality ⟨e^{−W/k_B T}⟩ = e^{−ΔF/k_B T} allows recovery of equilibrium ΔF from nonequilibrium pulling; requires many pulling trajectories (typically 50–500) for statistical convergence, and the end-state ΔF is more accurate than the full FES. SMD also generates physically reasonable initial structures for umbrella sampling windows.
- **Temperature replica exchange MD (T-REMD):** N replicas run simultaneously at temperatures T₁ < T₂ < … < T_N with exchange attempts between adjacent replicas at intervals τ_swap. Exchange accepted with Metropolis probability P_swap = min(1, exp((β_i − β_j)(E_i − E_j))). At high T, barriers are suppressed; the configurational diversity propagates back to low T. Requires ~20–30% acceptance rate between adjacent replicas; temperature spacing is typically 10–30 K. [EXPERT REVIEW NEEDED]
- **Metadynamics convergence (hills flooding / block analysis):** The bias is considered converged when the time-averaged FES computed from the last N_block frames no longer changes as a function of when in the trajectory N_block begins. A practical convergence test: divide the total simulation into blocks; compute the FES from each block independently; compare FES estimates. Large block-to-block variation indicates the bias is still evolving.
- **Reweighting:** Any biased MD trajectory can in principle be reweighted to recover unbiased observables: ⟨O⟩_unbias = ⟨O × e^{+V_bias/k_B T}⟩_bias / ⟨e^{+V_bias/k_B T}⟩_bias. In practice, this requires V_bias to be a known, time-independent potential. For WTM where the bias evolves with time, use the instantaneous bias V_bias(ξ, t) evaluated at each frame.
- **Transition-state recrossing and transmission coefficient:** Even a perfect FES does not give the rate constant without the transmission coefficient κ, which accounts for trajectory recrossings of the dividing surface. κ < 1 for systems with dynamical friction; it is not accessible from enhanced sampling alone.

## Key Workflows

### Workflow 1: CV definition and validation with PLUMED

Define CVs in PLUMED and test that they correctly discriminate the relevant states before committing to a long enhanced sampling run.

```bash
# ── PLUMED input file: plumed_cv_test.dat ─────────────────────────────────────
# System: alanine dipeptide isomerization in explicit TIP3P water
# CVs: phi and psi backbone torsion angles (standard test for peptide conformations)
# Purpose: validate CV discrimination before metadynamics

# -- Torsion angle CVs ---------------------------------------------------
PHI: TORSION ATOMS=5,7,9,15
PSI: TORSION ATOMS=7,9,15,17
# Atom indices: use GROMACS 1-based numbering (atom 5 = C in ACE)
# PLUMED uses 1-based indexing by default.
# CRITICAL: verify atom assignments with:
#   gmx dump -s system.tpr | grep "atom types" | head -40
# or load in VMD and check atom serial numbers.

# -- Distance CV (e.g., protein–ligand COM distance) ----------------------
# LIG_COM: COM ATOMS=1235-1280    # ligand heavy atoms
# PROT_COM: COM ATOMS=1-1234
# DIST_PL: DISTANCE ATOMS=PROT_COM,LIG_COM

# -- Coordination number CV (ion through channel) -------------------------
# Useful when the true RC is the number of water molecules coordinating an ion.
# COORD_ION: COORDINATION GROUPA=ION_IDX GROUPB=WATER_O_IDX R_0=0.315 NN=6 MM=12
# R_0: reference distance in nm; NN/MM: switching function exponents

# -- RMSD CV (folding) ----------------------------------------------------
# RMSD: RMSD REFERENCE=folded.pdb TYPE=OPTIMAL ATOMS=1-100
# REFERENCE: PDB file with REMARKS PLUMED DOMAIN specifying which atoms are active.

# -- Output: monitor CVs without any bias first ---------------------------
PRINT ARG=PHI,PSI FILE=colvar_test.dat STRIDE=100
```

```python
"""
Post-process the unbiased CV trajectory to verify state discrimination.

Before running any enhanced sampling, run 10–20 ns of plain MD and verify:
1. The CV discriminates the relevant states (bimodal or multimodal distribution).
2. The CV fluctuates significantly at equilibrium (σ_CV > hill width σ_meta).
3. No obvious state interconversion is seen (confirms the system IS slow).
4. CV time series is stationary (no drift from initial structure bias).

If the CV shows only one mode: either the system is in one state, or the CV
does not distinguish states. Compute P(CV) from the plain MD run.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def validate_cv_from_plain_md(colvar_file: str, cv_names: list,
                               n_skip_frames: int = 100) -> dict:
    """
    Load PLUMED COLVAR output and check that CVs show meaningful distributions.

    cv_names: list of column names matching PLUMED PRINT ARG= order.
    n_skip_frames: discard initial equilibration frames.
    """
    data = np.loadtxt(colvar_file, comments="#")
    time = data[n_skip_frames:, 0]
    cvs = data[n_skip_frames:, 1:1+len(cv_names)]

    results = {}
    fig, axes = plt.subplots(2, len(cv_names), figsize=(5*len(cv_names), 8))

    for i, name in enumerate(cv_names):
        cv = cvs[:, i]

        # Time series
        axes[0, i].plot(time / 1000, cv, lw=0.5, alpha=0.7)
        axes[0, i].set_xlabel("Time (ns)")
        axes[0, i].set_ylabel(name)
        axes[0, i].set_title(f"{name} — time series")

        # Histogram (FES estimate from plain MD)
        hist, bins = np.histogram(cv, bins=100, density=True)
        centers = 0.5 * (bins[:-1] + bins[1:])
        fes = -8.314e-3 * 300 * np.log(hist + 1e-20)  # kJ/mol, 300 K
        fes -= fes.min()  # set minimum to 0

        axes[1, i].plot(centers, fes, lw=2)
        axes[1, i].set_xlabel(name)
        axes[1, i].set_ylabel("Free energy (kJ/mol)")
        axes[1, i].set_title(f"{name} — FES from plain MD")
        axes[1, i].set_ylim([0, 40])

        results[name] = {
            "mean": cv.mean(),
            "std": cv.std(),
            "min": cv.min(),
            "max": cv.max(),
            "n_frames": len(cv),
        }
        print(f"  {name}: mean={cv.mean():.3f}, std={cv.std():.3f}, "
              f"range=[{cv.min():.3f}, {cv.max():.3f}]")
        if cv.std() < 0.05:
            print(f"  WARNING: {name} has very small fluctuations (std={cv.std():.3f}). "
                  "Gaussian hill width must be << std; check that the CV is physical.")

    plt.tight_layout()
    plt.savefig("cv_validation.png", dpi=150)
    print("Saved: cv_validation.png")
    return results
```

### Workflow 2: Well-tempered metadynamics with PLUMED + GROMACS

Production well-tempered metadynamics along one or two CVs. Includes PLUMED input, GROMACS run command, and convergence monitoring.

```bash
# ── plumed_wtmetad.dat — well-tempered metadynamics on phi/psi ───────────────
# System: alanine dipeptide, AMBER ff14SB, TIP3P, 300 K NPT
# Bias factor: 15 (suitable for barriers up to ~75 kJ/mol at 300 K)
# Hill height: 1.0 kJ/mol (start moderate; lower if CV fluctuates slowly)
# Hill width: sigma derived from equilibrium CV standard deviation (see Workflow 1)
# Hill deposition: every 500 steps (1 ps at 2 fs timestep)

UNITS ENERGY=kj/mol LENGTH=nm TIME=ps

# CV definitions (same as Workflow 1)
PHI: TORSION ATOMS=5,7,9,15
PSI: TORSION ATOMS=7,9,15,17

# Well-tempered metadynamics
METAD ...
    LABEL=metad
    ARG=PHI,PSI
    SIGMA=0.35,0.35        # hill width in rad; set to ~0.5 × CV_std from plain MD
                           # Too small: hills overfill the local well before sampling neighboring states
                           # Too large: poor resolution; FES smeared out
    HEIGHT=1.0             # initial hill height (kJ/mol)
    BIASFACTOR=15          # gamma = (T + delta_T) / T; delta_T = T * (gamma-1)
                           # At 300 K, bias temperature = 300 * 14 = 4200 K
                           # Effective temperature for the CV: T_eff = 300 * 15 = 4500 K
    PACE=500               # deposit a hill every 500 MD steps
    GRID_MIN=-pi,-pi       # grid for storing the bias (faster than list-based HILLS)
    GRID_MAX=pi,pi
    GRID_BIN=200,200       # 200x200 grid => Δξ = 2π/200 ≈ 0.031 rad per bin
    CALC_RCT               # compute reweighting correction c(t) for unbiased averages
    TEMP=300               # system temperature (K); required for WTM
... METAD

# Monitor CV and bias in real time
PRINT ARG=PHI,PSI,metad.bias FILE=colvar_wtmetad.dat STRIDE=100
PRINT ARG=metad.rct FILE=rct.dat STRIDE=1000    # reweighting correction c(t)

# Save the accumulated hills (for restart and analysis)
# HILLS file is written automatically by METAD if GRID_FILE is not set.
```

```bash
# ── GROMACS + PLUMED execution ────────────────────────────────────────────────
# GROMACS must be compiled (or patched) with PLUMED support.
# Patch procedure:
#   plumed patch -p        # interactive; select gromacs-2024.2
#   cmake ... ; make -j16  # rebuild GROMACS with PLUMED linkage
# Or use the plumed-gromacs conda package.

# Run WTM simulation (5 µs target; start with 500 ns to check convergence):
gmx mdrun \
    -v \
    -deffnm production_wtmetad \
    -plumed plumed_wtmetad.dat \
    -ntmpi 1 -ntomp 16 -gpu_id 0 \
    -nb gpu -pme gpu \
    -maxh 23.5

# If the run must be restarted (PLUMED checkpoint):
# PLUMED writes its state to plumed.dat.bck; use RESTART in plumed input
# or pass -restart to mdrun -cpi production_wtmetad.cpt

# ── Monitor convergence in real time ─────────────────────────────────────────
# Check FES evolution every ~100 ns:
plumed sum_hills \
    --hills HILLS \
    --outfile fes_100ns.dat \
    --stride 100000    # 100 ns worth of hills (pace=500 steps × 500 deposits × 0.002 ps/step)

plumed sum_hills \
    --hills HILLS \
    --outfile fes_200ns.dat \
    --stride 200000

# Compare fes_100ns.dat vs fes_200ns.dat: maximal difference should decrease over time.
# If it grows: the CV is not converging; a hidden slow mode is re-routing the system.
```

```python
"""
FES reconstruction from HILLS file and convergence block analysis.
"""
import numpy as np
import subprocess
from pathlib import Path


def compute_fes_blocks(hills_file: str, n_blocks: int = 5,
                       total_hills: int = None, temp_K: float = 300.0) -> dict:
    """
    Divide the HILLS file into n_blocks equal time segments.
    Compute FES from each segment and report block-to-block variance.

    A converged WTM simulation has FES_block[i] ≈ FES_block[j] for all i,j.
    Large block variance indicates unconverged bias.
    """
    if total_hills is None:
        # Count total hills deposited
        with open(hills_file) as f:
            total_hills = sum(1 for line in f if not line.startswith("#"))
    print(f"Total hills: {total_hills}")

    fes_blocks = []
    for block_idx in range(n_blocks):
        start = int(block_idx * total_hills / n_blocks)
        stop  = int((block_idx + 1) * total_hills / n_blocks)
        out_file = f"fes_block{block_idx}.dat"

        # Use the HILLS from (start, stop] — plumed sum_hills can select by --mintime/--maxtime
        # Here we approximate by using stride:
        result = subprocess.run([
            "plumed", "sum_hills",
            "--hills", hills_file,
            "--outfile", out_file,
            "--stride", str(stop - start),
            "--kt", str(8.314e-3 * temp_K),  # kJ/mol
        ], capture_output=True, text=True)

        if result.returncode != 0:
            print(f"Block {block_idx} FES failed: {result.stderr[-500:]}")
            continue

        fes_data = np.loadtxt(out_file, comments="#")
        fes_blocks.append(fes_data[:, -1])  # last column = FES in kJ/mol
        print(f"  Block {block_idx}: FES range {fes_data[:,-1].min():.1f}–"
              f"{fes_data[:,-1].max():.1f} kJ/mol")

    if len(fes_blocks) >= 2:
        fes_array = np.array(fes_blocks)
        block_std = np.std(fes_array, axis=0)
        max_spread = block_std.max()
        mean_spread = block_std.mean()
        print(f"\nFES block analysis:")
        print(f"  Max block std (worst point): {max_spread:.2f} kJ/mol")
        print(f"  Mean block std: {mean_spread:.2f} kJ/mol")
        if max_spread < 2.0:
            print("  CONVERGED: max block std < 2 kJ/mol")
        else:
            print("  NOT CONVERGED: continue simulation or check CV choice")

    return {"n_blocks": n_blocks, "fes_blocks": fes_blocks}
```

### Workflow 3: Umbrella sampling setup and MBAR reconstruction

Systematic umbrella sampling along a 1D reaction coordinate (e.g., pore radius, ion–channel distance, protein–ligand COM distance). Includes window setup via steered MD, parallel window execution, and PyMBAR reconstruction.

```python
"""
Step 1: Generate starting structures for umbrella windows via steered MD.
Pull the CV from ξ_start to ξ_end; extract snapshots at each target ξ.
"""

import numpy as np
from pathlib import Path
import subprocess


def generate_umbrella_windows(
    xi_start: float,          # nm (e.g., 0.2 nm = ion inside channel)
    xi_end: float,            # nm (e.g., 2.5 nm = ion in bulk)
    n_windows: int = 25,
    spring_const: float = 4000.0,   # kJ/mol/nm² (high k for controlled pulling)
    pull_rate: float = 0.001,       # nm/ps (slow pulling → quasi-static)
    tpr_file: str = "production.tpr",
    output_dir: str = "us_windows",
) -> list:
    """
    Generate umbrella sampling windows via steered MD (GROMACS pull code).

    The pull section in the mdp instructs GROMACS to pull the CV at constant
    velocity; frames at each target ξ₀ are extracted for window equilibration.

    Returns list of (xi_center, gro_file) tuples for window setup.
    """
    Path(output_dir).mkdir(exist_ok=True)
    xi_centers = np.linspace(xi_start, xi_end, n_windows)

    # Write mdp with pull code (GROMACS 2020+):
    pull_mdp = f"""
integrator      = md
dt              = 0.002
nsteps          = {int((xi_end - xi_start) / pull_rate / 0.002 + 100000)}

; Pull code for steered MD
pull            = yes
pull-ngroups    = 2
pull-ncoords    = 1
pull-group1-name = Ion          ; Reference: the ion to pull
pull-group2-name = Channel      ; Fixed group (COM of channel heavy atoms)
pull-coord1-type = umbrella     ; or linear-force-no-virial for SMD
pull-coord1-geometry = distance
pull-coord1-groups = 1 2
pull-coord1-dim = N N Y         ; pull only along z-axis
pull-coord1-start = yes         ; measure initial distance, set as reference
pull-coord1-rate  = {pull_rate} ; nm/ps
pull-coord1-k     = {spring_const}

; Output for pull force analysis
pull-nstxout = 100
pull-nstfout = 100
"""
    with open(f"{output_dir}/pull.mdp", "w") as f:
        f.write(pull_mdp)

    subprocess.run([
        "gmx", "grompp",
        "-f", f"{output_dir}/pull.mdp",
        "-c", "npt.gro", "-p", "topol.top",
        "-o", f"{output_dir}/pull.tpr",
    ], check=True)

    subprocess.run([
        "gmx", "mdrun", "-v",
        "-deffnm", f"{output_dir}/pull",
        "-ntmpi", "1", "-ntomp", "16",
    ], check=True)

    # Extract one frame per window from the pull trajectory
    window_files = []
    for i, xi in enumerate(xi_centers):
        # Compute approximate time in ps when the CV reaches xi
        t_target = (xi - xi_start) / pull_rate  # ps
        gro_out = f"{output_dir}/window_{i:03d}.gro"

        subprocess.run([
            "gmx", "trjconv",
            "-f", f"{output_dir}/pull.xtc",
            "-s", f"{output_dir}/pull.tpr",
            "-o", gro_out,
            "-dump", str(t_target),  # extract frame nearest to t_target
        ], input="System\n", text=True, capture_output=True)

        window_files.append((xi, gro_out))
        print(f"  Window {i:03d}: xi={xi:.3f} nm → {gro_out}")

    return window_files
```

```python
"""
Step 2: MBAR reconstruction from umbrella sampling windows.

Each window runs for 10–20 ns with a harmonic restraint centered at ξ₀.
PyMBAR reconstructs the unbiased FES from all windows simultaneously.
"""

import numpy as np
import pymbar


def reconstruct_fes_mbar(
    colvar_files: list,       # PLUMED COLVAR output; one file per window
    xi_centers: np.ndarray,   # window centers (nm)
    spring_k: float = 3000.0, # kJ/mol/nm²; must match mdp/plumed umbrella k
    temp_K: float = 300.0,
    xi_bins: int = 200,
    n_skip_frames: int = 500, # discard first N frames per window (equilibration)
) -> dict:
    """
    MBAR reconstruction of 1D umbrella sampling FES.

    Requirements:
    - Sufficient window overlap: neighboring window distributions must overlap
      in the tails (P_window_i(ξ) × P_window_j(ξ) > 0 for adjacent i,j).
      Check with: plot P(ξ) for each window on the same axes; no gaps allowed.
    - Adequate sampling per window: autocorrelation time << window length.
    - Known spring constant: must match what PLUMED or GROMACS pull code used.
    """
    kT = 8.314e-3 * temp_K  # kJ/mol

    # Load CV data from each window
    all_cv = []
    window_lengths = []
    for cfile in colvar_files:
        data = np.loadtxt(cfile, comments="#")
        cv = data[n_skip_frames:, 1]  # second column = CV value
        all_cv.append(cv)
        window_lengths.append(len(cv))

    N_total = sum(window_lengths)
    n_windows = len(xi_centers)

    # Build MBAR u_kn matrix (reduced potential for each frame under each window's bias)
    u_kn = np.zeros((n_windows, N_total))
    cv_all = np.concatenate(all_cv)

    for k, xi_k in enumerate(xi_centers):
        # Reduced potential for window k at all frames: u_k(ξ) = (k/2)(ξ-ξ₀)² / kT
        u_kn[k, :] = (spring_k / (2 * kT)) * (cv_all - xi_k) ** 2

    N_k = np.array(window_lengths, dtype=int)

    # Run MBAR
    mbar = pymbar.MBAR(u_kn, N_k, verbose=True)

    # Compute FES on a fine grid
    xi_grid = np.linspace(cv_all.min(), cv_all.max(), xi_bins)
    u_grid = np.zeros((n_windows, xi_bins))
    for k, xi_k in enumerate(xi_centers):
        u_grid[k, :] = (spring_k / (2 * kT)) * (xi_grid - xi_k) ** 2

    fes_result = mbar.compute_free_energy_differences(u_grid, return_dict=True)

    fes = fes_result["Delta_f"] * kT  # kJ/mol
    fes_err = fes_result.get("dDelta_f", np.zeros_like(fes)) * kT

    # Shift minimum to 0
    fes -= fes.min()

    # Check window overlap
    print("\nWindow overlap check:")
    for i in range(n_windows - 1):
        overlap = np.sum((all_cv[i] > xi_centers[i+1] - 0.3 * np.std(all_cv[i+1])) |
                         (all_cv[i+1] < xi_centers[i] + 0.3 * np.std(all_cv[i])))
        print(f"  Windows {i}–{i+1}: {overlap} overlapping frames "
              f"(need > 50 for reliable MBAR)")

    return {"xi": xi_grid, "fes_kJ_mol": fes, "fes_err": fes_err,
            "mbar": mbar, "kT": kT}
```

### Workflow 4: Adaptive biasing force (ABF) with PLUMED

ABF does not require prior knowledge of barrier shape. Suitable for systems where the CV range is known but the barrier profile is not.

```bash
# ── plumed_abf.dat — adaptive biasing force along a 1D CV ───────────────────
# System: Na+ permeation through a model ion channel, CHARMM36m + TIP3P
# CV: z-position of Na+ relative to channel center-of-mass
# ABF flattens the free energy surface along this CV adaptively.

UNITS ENERGY=kj/mol LENGTH=nm TIME=ps

# Channel center-of-mass
CHANNEL_COM: CENTER ATOMS=1-2400 WEIGHTS=MASS

# Na+ ion
ION: GROUP ATOMS=2401

# CV: z-distance between Na+ and channel COM
Z_ION: DISTANCE ATOMS=CHANNEL_COM,ION COMPONENTS
Z_CV: COMBINE ARG=Z_ION.z COEFFICIENTS=1 PERIODIC=NO

# Extended ABF (eABF) — more stable than pure ABF for stiff CVs
EXTENDED_LAGRANGIAN ...
    LABEL=eabf
    ARG=Z_CV
    KAPPA=4000.0          # kJ/mol/nm²; coupling constant for extended DOF
    TAU=0.5               # ps; fictitious mass / relaxation time
    FRICTION=40.0         # ps⁻¹; Langevin thermostat for extended DOF
    TEMP=300
... EXTENDED_LAGRANGIAN

# ABF bias on the extended DOF
abf: ABF ...
    ARG=eabf.Z_CV-0       # biased DOF is the extended coordinate
    MIN=-2.0              # nm; start of biased region
    MAX=2.0               # nm; end of biased region
    BIN_SIZE=0.05         # nm; bin size for force accumulation
    FULL_SAMPLES=200      # do not apply bias until 200 samples in a bin
    WRITE_BIAS=yes
    WRITE_PMFFILE=pmf_abf.dat
    WRITE_COUNT=yes
    PACE=500              # write output every 500 steps
... abf

PRINT ARG=Z_CV,abf.bias FILE=colvar_abf.dat STRIDE=100
```

```python
"""
ABF convergence check: the PMF should stop changing once ≥ 200 samples
per bin have been accumulated everywhere along the CV.
"""
import numpy as np
import matplotlib.pyplot as plt


def check_abf_convergence(pmf_file: str, count_file: str,
                           min_samples: int = 200) -> bool:
    """
    Check ABF PMF convergence from PLUMED ABF output.

    pmf_file: PLUMED WRITE_PMFFILE output (cols: xi, pmf, pmf_err or single col)
    count_file: PLUMED WRITE_COUNT output (cols: xi, count)
    """
    pmf = np.loadtxt(pmf_file, comments="#")
    count = np.loadtxt(count_file, comments="#")

    xi = pmf[:, 0]
    fes = pmf[:, 1]
    n = count[:, 1]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))

    ax1.plot(xi, fes - fes.min(), lw=2)
    ax1.set_xlabel("CV (nm)")
    ax1.set_ylabel("PMF (kJ/mol)")
    ax1.set_title("ABF free energy profile")

    ax2.bar(xi, n, width=(xi[1] - xi[0]) * 0.9,
            color=np.where(n >= min_samples, "green", "red"))
    ax2.axhline(min_samples, color="k", ls="--", label=f"Min samples = {min_samples}")
    ax2.set_xlabel("CV (nm)")
    ax2.set_ylabel("Sample count")
    ax2.set_title("Sample count per bin (red = under-sampled)")
    ax2.legend()

    plt.tight_layout()
    plt.savefig("abf_convergence.png", dpi=150)

    n_under = np.sum(n < min_samples)
    total_bins = len(n)
    print(f"ABF: {n_under}/{total_bins} bins under-sampled (< {min_samples} samples)")
    print(f"Min samples in any bin: {n.min():.0f}")
    converged = n_under == 0
    print(f"Status: {'CONVERGED' if converged else 'NOT CONVERGED — continue simulation'}")
    return converged
```

### Workflow 5: Temperature replica exchange MD (T-REMD) with GROMACS

T-REMD for systems where the relevant transitions are not easily described by a single CV. Particularly useful for peptide folding and amyloid aggregation.

```python
"""
T-REMD temperature spacing calculation.
Target 20–25% acceptance rate between adjacent replicas.
The acceptance rate depends on the overlap of potential energy distributions
between adjacent temperatures.
"""
import numpy as np
from scipy.special import erf


def compute_tremd_temperatures(
    T_min: float,         # K (target low temperature, e.g., 300 K)
    T_max: float,         # K (target high temperature, e.g., 450 K)
    n_atoms: int,         # number of atoms in the system
    target_acceptance: float = 0.23,  # typical optimal value [EXPERT REVIEW NEEDED]
    n_dof: int = None,    # degrees of freedom; if None = 3*n_atoms - 6
) -> np.ndarray:
    """
    Estimate temperature ladder for T-REMD using the equation from
    Patriksson & van der Spoel (2008): acceptance rate ≈ erfc(sqrt(ΔE²/(4kT²Nf/2)))
    where Nf = degrees of freedom.

    For most biomolecular systems in explicit solvent:
    ΔT/T ≈ sqrt(2*f_target / (n_dof * k_B)) where f_target = erf^-1(target_acceptance)
    Practical rule: ΔT/T ~ 0.02–0.05 for typical biomolecular sizes. [EXPERT REVIEW NEEDED]
    """
    if n_dof is None:
        n_dof = 3 * n_atoms - 6  # classical

    kB = 8.314e-3  # kJ/mol/K

    # Geometric spacing: T_{i+1} = T_i * r (constant ratio)
    # Solve: accept_rate ~ erfc(sqrt(n_dof / 2) * |1/T_i - 1/T_{i+1}|
    #                                              * kB / sqrt(2))
    # For simplicity, estimate number of replicas from target ΔT/T:
    delta_T_over_T = np.sqrt(2.0 / n_dof) * 0.5  # rough estimate [EXPERT REVIEW NEEDED]
    n_replicas_est = int(np.log(T_max / T_min) / np.log(1 + delta_T_over_T)) + 2
    T_ladder = np.geomspace(T_min, T_max, n_replicas_est)

    print(f"T-REMD temperature ladder ({n_replicas_est} replicas):")
    for i, T in enumerate(T_ladder):
        print(f"  Replica {i:3d}: {T:.1f} K")
    print(f"  ΔT range: {T_ladder[1]-T_ladder[0]:.1f}–"
          f"{T_ladder[-1]-T_ladder[-2]:.1f} K")
    print(f"  NOTE: Run a short test (5–10 ns/replica) and check actual "
          f"acceptance rates with: grep 'Repl ex' md.log | awk '{{...}}'")
    return T_ladder
```

```bash
#!/bin/bash
# ── T-REMD execution with GROMACS ─────────────────────────────────────────────
# Assumes N_REPLICAS separate .tpr files: replica_000.tpr, replica_001.tpr, ...
# Each .tpr compiled with the same topology + coordinates but different ref-t.

N_REPLICAS=16
SWAP_INTERVAL=1000   # steps between exchange attempts; must give ~0.5–2 ps between swaps

# Generate per-replica .tpr files (different ref-t only):
for i in $(seq 0 $((N_REPLICAS - 1))); do
    TEMP=$(awk "BEGIN {printf \"%.1f\", 300 * exp(${i} * log(450.0/300.0) / ($N_REPLICAS - 1))}")
    sed "s/ref-t.*/ref-t = ${TEMP} ${TEMP}/" production.mdp > replica_${i}.mdp
    gmx grompp -f replica_${i}.mdp -c npt.gro -t npt.cpt -p topol.top \
               -o replica_${i}.tpr -maxwarn 0
done

# Run T-REMD with gmx mdrun -multidir:
mkdir -p $(seq -f "replica_%03g" 0 $((N_REPLICAS - 1)))
for i in $(seq 0 $((N_REPLICAS - 1))); do
    cp replica_${i}.tpr replica_$(printf '%03d' ${i})/topol.tpr
done

mpirun -n ${N_REPLICAS} gmx_mpi mdrun \
    -multidir replica_*/  \
    -replex ${SWAP_INTERVAL} \
    -nex 100              \   # number of exchange attempts per replex interval
    -deffnm production    \
    -ntomp 8              \
    -maxh 23.5

# ── Check acceptance rates from log ────────────────────────────────────────────
# Target: 20–30% acceptance rate between adjacent replicas
grep "Repl ex" replica_000/production.log | tail -20
# Output format: "Repl ex  0-1  A  1-2  A  2-3  D ..." (A=accepted, D=declined)

# Count acceptance rate per pair:
grep "Repl ex" replica_000/production.log | \
    awk '{for(i=2;i<=NF;i++) if($i=="A"||$i=="D") counts[$(i-1)]+=(($i=="A")?1:0); total[$(i-1)]++}
         END {for(p in counts) printf "Pair %s: %.1f%%\n", p, 100*counts[p]/total[p]}' | sort
# If acceptance < 15%: reduce T spacing (add more replicas between adjacent T pair)
# If acceptance > 35%: increase T spacing (can run fewer replicas, same total compute)
```

### Workflow 6: Enhanced sampling with ML interatomic potentials

ML potentials (MACE, NequIP) dramatically reduce the cost per MD step, enabling enhanced sampling at DFT accuracy on barriers that were previously inaccessible.

```python
"""
Well-tempered metadynamics with MACE via ASE + PLUMED-Python interface.

ASE provides the MACE calculator; PLUMED's Python interface (plumed-python)
applies the metadynamics bias at each MD step. The PLUMED Python library
is independent of the GROMACS/LAMMPS patching and works with any ASE calculator.
"""

import numpy as np
from ase.io import read
from ase.md.langevin import Langevin
from ase import units
from mace.calculators import MACECalculator
import plumed


def run_wtmetad_mace(
    structure_file: str,     # initial structure (e.g., POSCAR, CIF, xyz)
    mace_model_path: str,    # path to MACE .model file
    plumed_input_file: str,  # path to PLUMED input (.dat)
    temperature_K: float = 600.0,
    timestep_fs: float = 1.0,    # reduce for light atoms or reactive systems
    n_steps: int = 2_000_000,    # 2 ns at 1 fs timestep
    friction_ps: float = 10.0,   # Langevin friction coefficient (ps⁻¹)
    device: str = "cuda",        # "cuda" or "cpu"
) -> None:
    """
    Run well-tempered metadynamics with MACE potential via ASE + PLUMED-Python.

    PLUMED-Python receives atomic positions and cell at each step, computes
    CVs and bias forces, and returns biased forces to the ASE MD engine.

    Prerequisites:
    - pip install mace-torch plumed
    - PLUMED compiled as a shared library: plumed/lib/libplumed.so
    - Environment: PLUMED_KERNEL=/path/to/libplumedKernel.so
    """
    atoms = read(structure_file)
    atoms.pbc = True

    # MACE calculator (equivariant MLP)
    calc = MACECalculator(model_paths=mace_model_path, device=device,
                           default_dtype="float64")
    atoms.calc = calc

    # PLUMED wrapper: intercepts forces and adds bias at each step
    n_atoms = len(atoms)
    p = plumed.Plumed()
    p.cmd("setMDEngine", "python")
    p.cmd("setTimestep", timestep_fs / 1000.0)     # fs → ps
    p.cmd("setKbT", 8.314e-3 * temperature_K)       # kJ/mol
    p.cmd("setNatoms", n_atoms)
    p.cmd("setLogFile", "plumed_mace.log")
    p.cmd("init")

    with open(plumed_input_file) as f:
        plumed_script = f.read()
    p.cmd("readInputLines", plumed_script)

    # Wrap ASE Langevin with PLUMED bias injection
    class PlumedLangevin(Langevin):
        def __init__(self, *args, plumed_obj=None, **kwargs):
            super().__init__(*args, **kwargs)
            self.plumed = plumed_obj
            self._step = 0

        def run(self, steps):
            for _ in range(steps):
                self.step()
                self._step += 1

        def step(self):
            pos = self.atoms.get_positions()   # Å → PLUMED expects nm
            cell = self.atoms.get_cell()[:]

            # Set PLUMED inputs
            self.plumed.cmd("setStep", self._step)
            self.plumed.cmd("setPositions", pos * 0.1)          # Å → nm
            self.plumed.cmd("setBox", cell * 0.1)               # Å → nm
            self.plumed.cmd("setMasses", self.atoms.get_masses())

            forces = self.atoms.get_forces().copy()             # eV/Å
            bias_forces = np.zeros_like(forces)
            self.plumed.cmd("setForces", bias_forces)
            self.plumed.cmd("setVirial", np.zeros((3, 3)))
            self.plumed.cmd("calc")

            # bias_forces now contains PLUMED bias (kJ/mol/nm → eV/Å)
            forces += bias_forces * (1 / 96.485) * 10  # kJ/mol/nm → eV/Å
            self.atoms.calc.results["forces"] = forces

            super().step()

    md = PlumedLangevin(
        atoms,
        timestep=timestep_fs * units.fs,
        temperature_K=temperature_K,
        friction=friction_ps / (1000.0 * units.fs),  # ps⁻¹ → fs⁻¹
    )
    md.run(n_steps)
    p.cmd("runFinalJobs")
    print("WTM-MACE run complete.")
```

### Workflow 7: FES convergence diagnostics and reweighting

Diagnosing convergence is the most commonly skipped step. A non-converged FES looks like a converged one to visual inspection.

```python
"""
Quantitative FES convergence check: block analysis + reweighting for WTM.

The standard test: compute the FES from successive time windows (blocks) of
the simulation. If the FES has converged, FES estimates from all blocks should
agree to within thermal noise (< 1–2 kJ/mol for most processes).
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter


def wt_metad_fes_from_colvar(
    colvar_file: str,      # PLUMED PRINT ARG=PHI,PSI,metad.bias output
    bias_col: int = 3,     # column index of the bias V_meta(ξ, t) (0-indexed)
    cv_col: int = 1,       # column index of the CV
    temp_K: float = 300.0,
    bias_factor: float = 15.0,    # gamma; must match PLUMED BIASFACTOR
    n_blocks: int = 5,
    xi_bins: int = 100,
    n_skip: int = 200,
) -> dict:
    """
    Reconstruct unbiased FES from WTM COLVAR using direct reweighting.

    Reweighting formula (Tiana & Camilloni 2019 / standard):
      w(t) = exp(+V_bias(ξ(t), t) / k_B T) × exp(−c(t) / k_B T)
    where c(t) is the reweighting correction (metad.rct in PLUMED).

    If rct is not available, use the instantaneous bias as-is (approximate).
    The bias-corrected probability: P_unbias(ξ) ∝ Σ_t δ(ξ − ξ(t)) × w(t).
    """
    kT = 8.314e-3 * temp_K

    data = np.loadtxt(colvar_file, comments="#")
    data = data[n_skip:]

    time = data[:, 0]
    cv   = data[:, cv_col]
    bias = data[:, bias_col]  # V_meta(ξ(t), t) in kJ/mol

    # Reweighting weights (exp(+V_bias/kT))
    log_w = bias / kT
    log_w -= log_w.max()  # numerical stability
    weights = np.exp(log_w)

    # FES from full trajectory (reweighted)
    hist_w, edges = np.histogram(cv, bins=xi_bins, weights=weights, density=True)
    hist_uw, _   = np.histogram(cv, bins=xi_bins, density=True)  # unweighted for comparison
    centers = 0.5 * (edges[:-1] + edges[1:])

    fes_rw = -kT * np.log(hist_w + 1e-40)
    fes_rw -= fes_rw.min()
    fes_uw = -kT * np.log(hist_uw + 1e-40)
    fes_uw -= fes_uw.min()

    # Block analysis for convergence
    block_fes = []
    block_labels = []
    n_per_block = len(data) // n_blocks
    for b in range(n_blocks):
        block_cv = cv[b*n_per_block:(b+1)*n_per_block]
        block_w  = weights[b*n_per_block:(b+1)*n_per_block]
        h, _ = np.histogram(block_cv, bins=edges, weights=block_w, density=True)
        fes_b = -kT * np.log(h + 1e-40)
        fes_b -= np.nanmin(fes_b)
        block_fes.append(fes_b)
        block_labels.append(f"Block {b+1}")

    # Block variance
    fes_array = np.array(block_fes)
    block_std  = np.nanstd(fes_array, axis=0)
    max_spread = np.nanmax(block_std)
    print(f"\nFES convergence (block analysis, {n_blocks} blocks):")
    print(f"  Max block std: {max_spread:.2f} kJ/mol")
    print(f"  Mean block std: {np.nanmean(block_std):.2f} kJ/mol")
    print(f"  Status: {'CONVERGED' if max_spread < 2.0 else 'NOT CONVERGED'}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for b_fes, label in zip(block_fes, block_labels):
        axes[0].plot(centers, b_fes, lw=1.5, alpha=0.8, label=label)
    axes[0].plot(centers, fes_rw, "k-", lw=2.5, label="Full (reweighted)")
    axes[0].set_xlabel("CV")
    axes[0].set_ylabel("Free energy (kJ/mol)")
    axes[0].set_title("Block analysis — WTM FES")
    axes[0].legend(fontsize=8)
    axes[0].set_ylim([0, 60])

    axes[1].plot(centers, block_std, "r-", lw=2)
    axes[1].axhline(2.0, color="k", ls="--", label="2 kJ/mol threshold")
    axes[1].set_xlabel("CV")
    axes[1].set_ylabel("Block std (kJ/mol)")
    axes[1].set_title("FES uncertainty (block std)")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("wtmetad_convergence.png", dpi=150)
    print("Saved: wtmetad_convergence.png")

    return {
        "cv": centers,
        "fes_reweighted": fes_rw,
        "fes_unweighted": fes_uw,
        "block_std": block_std,
        "max_block_std_kJ_mol": max_spread,
    }
```

## Common Failure Modes

| Failure | Symptom | Likely Cause | Diagnostic | Fix |
|---------|---------|--------------|------------|-----|
| Bad collective variable | FES reconstructed repeatedly looks same but barrier is wrong vs experiment; different CV choices give inconsistent ΔG | CV does not project out the true slow mode; hidden slow degrees of freedom not captured | Compute TICA / TIC projections from short unbiased MD; check if bimodal structure appears only along a combination of variables | Add orthogonal CVs; use path CVs or machine-learned CVs (TICA); or switch to replica exchange which does not require CV knowledge |
| Hidden slow mode (hysteresis) | FES changes between early and late parts of simulation in a direction-dependent way; restarting from different initial states gives different FES | A slow orthogonal degree of freedom is not captured by the CV; the system passes through different hidden-variable states in different simulation segments | Plot FES vs. time blocks — if blocks differ systematically not just noisily, and different restarts disagree, suspect a hidden mode | Identify the hidden mode by projecting trajectory onto additional CVs; add it to the bias; or use higher-dimensional MBAR with the orthogonal variable as a second CV |
| Insufficient window overlap (US) | WHAM/MBAR diverges; FES has discontinuities or inflection points at window boundaries; reconstructed FES changes with WHAM bin size | Umbrella windows are spaced too far apart; insufficient sampling in a high-barrier window causes gaps in the histogram | Plot P(ξ) for each window on the same axes; gaps in coverage → add intermediate windows between the sparse pair | Run intermediate umbrella windows at ξ between the gap pair; increase spring constant to narrow individual distributions; increase simulation length per window |
| Metadynamics overfilling | FES flips sign (barrier becomes a well); the system appears trapped in a previously well-sampled state | Hill deposition rate too fast relative to diffusion time of CV in the well; or SIGMA too small creating sharp hills that overshoot | Track V_bias(ξ, t) over time; if V_bias exceeds the expected barrier height for the current state, hills are being deposited too fast | Reduce PACE (deposit hills less frequently); reduce HEIGHT; switch to WTM if using standard metadynamics; increase SIGMA to lower resolution but slower overfilling |
| Non-reweightable bias | Reweighted FES differs drastically from FES obtained by another method (MBAR vs direct); reweighted observables have enormous variance | Hill width too small → large point-to-point V_bias fluctuations; or the simulation is so short that the accumulated bias is still far from convergence | Compute effective sample size: Neff = (Σ w)² / Σ w² where w = exp(+V/kT); if Neff/N < 0.1, reweighting is unreliable | Increase SIGMA; run longer to flatten the FES before reweighting; use CALC_RCT and reweight using the c(t) correction |
| T-REMD: low acceptance rate | Acceptance rate < 10% between adjacent replicas; replicas do not mix; low-T replica does not escape local minima | Temperature spacing too large; system too large (energy gap between replicas too wide) | Compute `grep "Repl ex" md.log | awk` acceptance fraction per pair | Reduce temperature spacing (add more replicas); scale temperature range more conservatively; for very large systems consider Hamiltonian REMD instead of T-REMD |
| T-REMD: wrong temperature for analysis | Production observables extracted from a fixed-replica trajectory include frames at different temperatures; computed averages wrong | Replica trajectories mix replicas at all temperatures; confusing replica trajectory with temperature trajectory | In GROMACS T-REMD, `mdrun -multidir` writes one trajectory per temperature slot (demixed); verify using `demux.pl` script from GROMACS tools | Use `demux.pl` to convert replica trajectories to constant-temperature trajectories before analysis; never analyze the raw per-replica trajectories directly |
| Biased initial state | FES has an artifact at the starting CV value; the FES minimum appears at the initial configuration not at the equilibrium structure | Simulation started from a non-equilibrium structure (e.g., crystal structure with constraints just released) before the enhanced sampling begins | Plot CV time series; if the CV drifts monotonically for the first 10–20% of the simulation from the initial value, the system was not equilibrated first | Always equilibrate with plain MD (10–100 ns) before starting enhanced sampling; verify that the CV is fluctuating around a stable value before starting metadynamics |
| Thermostat artifacts in T-REMD | Kinetic energy appears correct but potential energy distribution does not match the target T | Incompatible thermostat (Berendsen suppresses potential energy fluctuations, breaking the Boltzmann criterion for T-REMD swaps) | Check thermostat type; Berendsen gives wrong energy distribution → swap acceptance rates are wrong even if they appear ~20% | Use V-rescale or Nosé-Hoover for all T-REMD replicas; never use Berendsen in T-REMD |
| Reporting FES without uncertainty | Published FES claims 2 kJ/mol barrier with no confidence interval; subsequent simulations yield 8 kJ/mol | FES from a single run with no block analysis; convergence not verified | Block analysis (Workflow 7) quantifies uncertainty; if max block std > 2 kJ/mol, the FES is not converged | Always report block analysis std as the uncertainty on the FES; never report a single-run FES without convergence verification |

## Best Practices

- **Always test the CV before committing to enhanced sampling.** Run 5–20 ns of plain MD first, compute P(ξ), and verify that the target states correspond to distinct modes in P(ξ). If they do not, the CV cannot distinguish the states and enhanced sampling will not help.
- **Set SIGMA to 0.3–0.5 × the equilibrium CV standard deviation.** Too small: hills overfill the well before the system diffuses across it. Too large: poor FES resolution. Estimate σ_CV from the plain MD test run.
- **Use well-tempered metadynamics (WTM) for all production runs; never use untempered metadynamics.** Untempered metadynamics accumulates bias indefinitely, eventually destabilizing all states. WTM converges to a finite bias and is directly reweightable.
- **Always run block analysis before reporting a FES.** Divide the trajectory into 5–10 blocks and compute the FES from each block. The maximum block standard deviation is the uncertainty on the FES. Report it alongside every FES figure. A max std of > 2 kJ/mol indicates an unconverged simulation.
- **For umbrella sampling, verify window overlap before running MBAR.** Plot the P(ξ) histogram for each window on a single axis. No gaps are allowed between adjacent windows for MBAR to converge. If gaps exist, add intermediate windows before analysis, not after.
- **For T-REMD: check acceptance rates after the first 1 ns and before committing to a full run.** A short 5 ns test run with all replicas gives acceptance rate statistics sufficient to decide whether the temperature spacing is correct. Fixing spacing is free; wasting 100 ns per replica on a bad ladder is expensive.
- **Equilibrate with plain MD before starting any enhanced sampling.** Starting metadynamics or umbrella sampling from a non-equilibrated structure introduces systematic bias at the initial CV position. The enhanced sampling method will then focus on exploring away from an unphysical starting point rather than the true free energy landscape.
- **For MLP-accelerated enhanced sampling: regularly check committee uncertainty.** MACE and NequIP can produce confident but wrong forces outside the training distribution. If the trajectory visits configurations with high committee force variance (σ_F > 0.1–0.5 eV/Å) during enhanced sampling, those frames are unreliable and may produce artifactual FES features. [EXPERT REVIEW NEEDED]
- **Do not mix reweighting from different methods without verification.** Mixing WTM-reweighted FES with direct US/MBAR FES requires thermodynamic consistency checks. Use a single method end-to-end, or verify that the two FES agree within uncertainty at an overlapping region.
- **For 2D FES: use METAD with a grid (GRID_MIN/MAX/BIN) rather than list-based HILLS.** The grid-based accumulation is an order of magnitude faster for 2D CVs and avoids the O(N) cost per step of summing all deposited hills.

## Integration with Other Skills

- **[gromacs-workflow](../gromacs-workflow/SKILL.md):** GROMACS provides the MD engine that PLUMED patches into. The production `.tpr` from that skill is the starting point for all PLUMED-driven enhanced sampling runs. The equilibration protocol from that skill is a prerequisite before metadynamics or umbrella sampling begins.
- **[lammps-workflow](../lammps-workflow/SKILL.md):** PLUMED interfaces with LAMMPS via the `fix plumed` command, enabling umbrella sampling and metadynamics in LAMMPS simulations of metals, ceramics, and other materials-science systems where GROMACS is not appropriate.
- **[md-equilibration](../md-equilibration/SKILL.md):** Enhanced sampling must always be preceded by confirmed equilibration. The convergence criteria from that skill (block averaging of T, P, density; stationarity tests) must be satisfied before any biased simulation begins.
- **[md-analysis](../md-analysis/SKILL.md):** Post-processing enhanced sampling trajectories requires the PBC unwrapping, trajectory loading, and block averaging methods from that skill, especially for extracting unbiased structural observables from reweighted trajectories.
- **[md-force-fields](../md-force-fields/SKILL.md):** The choice of force field determines the accuracy of the enhanced sampling FES. The validation criteria and transferability limits from that skill apply equally to enhanced sampling contexts; a FES computed with a poorly parameterized force field is not physically meaningful.
- **[free-energy-md](../free-energy-md/SKILL.md) [FORWARD REFERENCE — skill not yet created]:** Alchemical free energy (FEP) methods compute free energy differences along a coupling parameter λ rather than a physical CV. Umbrella sampling along a physical CV and FEP along λ are complementary and can be combined for protein–ligand binding free energies.
- **[coarse-grained-md](../coarse-grained-md/SKILL.md) [FORWARD REFERENCE — skill not yet created]:** Enhanced sampling with Martini or other coarse-grained force fields is an important computational strategy: the faster dynamics and lower barrier heights in CG models provide a rougher but much cheaper FES that can guide the design of all-atom enhanced sampling protocols.
- **[reactive-md](../reactive-md/SKILL.md):** ReaxFF or QM/MM trajectories can be coupled to PLUMED for enhanced sampling of chemical reactions. The CV must be chosen to describe the bond-forming/breaking event; the reweighting is identical but force field setup differs from classical MD.
- **[mlp-validation](../../ml-interatomic-potentials/mlp-validation/SKILL.md):** Before running MLP-accelerated enhanced sampling, validate the MLP on structures that span the expected CV range. Committee uncertainty along the sampled CV trajectory must be checked; frames with high uncertainty must be flagged or excluded from FES reconstruction.
- **[mlp-to-lammps](../../ml-interatomic-potentials/mlp-to-lammps/SKILL.md):** LAMMPS is the most common engine for MLP-driven MD in materials contexts. PLUMED + LAMMPS + MACE/NequIP via `pair_style mace` enables enhanced sampling with MLP accuracy at LAMMPS throughput, which is the standard approach for materials-science rare events.
- **[scientific-visualization](../../scientific-data/scientific-visualization/SKILL.md):** FES visualization (2D heatmaps, projected FES along CVs, convergence plots), trajectory coloring by CV value, and replica exchange acceptance animations are all common outputs of enhanced sampling that require the plotting patterns from that skill.
- **[reproducibility-workflow](../../scientific-data/reproducibility-workflow/SKILL.md):** Enhanced sampling results depend critically on the PLUMED input file version, hill deposition parameters, and accumulated HILLS file. All three must be archived alongside the trajectory for any published FES. That skill covers DVC-based provenance for large simulation outputs.

## Key References

1. Laio, A. & Parrinello, M. (2002). *Escaping free-energy minima.* Proceedings of the National Academy of Sciences, 99(20), 12562–12566. https://doi.org/10.1073/pnas.202427399
2. Barducci, A., Bussi, G., & Parrinello, M. (2008). *Well-tempered metadynamics: A smoothly converging and tunable free-energy method.* Physical Review Letters, 100(2), 020603. https://doi.org/10.1103/PhysRevLett.100.020603
3. Tribello, G. A., Bonomi, M., Branduardi, D., Camilloni, C., & Bussi, G. (2014). *PLUMED 2: New feathers for an old bird.* Computer Physics Communications, 185(2), 604–613. https://doi.org/10.1016/j.cpc.2013.09.018
4. Shirts, M. R. & Chodera, J. D. (2008). *Statistically optimal analysis of samples from multiple equilibrium states.* Journal of Chemical Physics, 129(12), 124105. https://doi.org/10.1063/1.2978177
5. Frenkel, D. & Smit, B. (2002). *Understanding Molecular Simulation: From Algorithms to Applications* (2nd ed.). Academic Press. ISBN 978-0-12-267351-1.
6. Valsson, O., Tiwary, P., & Parrinello, M. (2016). *Enhancing important fluctuations: Rare events and metadynamics from a conceptual viewpoint.* Annual Review of Physical Chemistry, 67, 159–184. https://doi.org/10.1146/annurev-physchem-040215-112229
7. Henin, J., Lelievre, T., Shirts, M. R., Valsson, O., & Delemotte, L. (2022). *Enhanced sampling methods for molecular dynamics simulations.* Living Journal of Computational Molecular Science, 4(1), 1583. https://doi.org/10.33011/livecoms.4.1.1583
8. Sugita, Y. & Okamoto, Y. (1999). *Replica-exchange molecular dynamics method for protein folding.* Chemical Physics Letters, 314(1–2), 141–151. https://doi.org/10.1016/S0009-2614(99)01123-9
