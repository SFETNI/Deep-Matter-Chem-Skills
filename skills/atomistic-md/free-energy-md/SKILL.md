---
name: free-energy-md
domain: atomistic-md
maturity: draft
tools:
  - GROMACS
  - PLUMED
  - pymbar
  - alchemlyb
  - numpy
  - pandas
  - matplotlib
tested_versions:
  GROMACS: "2024.2"
  pymbar: "4.0.1"
  alchemlyb: "2.1.0"
  PLUMED: "2.9.2"
  pandas: "2.2.0"
requires_gpu: false
estimated_runtime: hours-to-days
---

# Free-Energy MD

## Description

This skill covers quantitative free-energy calculations from molecular dynamics: alchemical methods (free energy perturbation, thermodynamic integration, BAR, MBAR), potential of mean force from umbrella sampling, absolute and relative binding/hydration/solvation free energies, thermodynamic cycles, lambda schedules and soft-core potentials, overlap diagnostics, standard-state corrections, and statistical uncertainty analysis using pymbar and alchemlyb. GROMACS provides the simulation engine with the `lambda` free-energy infrastructure; PLUMED provides CV-based PMF calculations; pymbar/alchemlyb provide statistically optimal analysis. Invoke this skill when a quantitative ΔG, ΔΔG, or PMF is the target observable and cannot be obtained from plain MD averages.

## Domain Context

Free-energy calculations in MD operate on two distinct paradigms:

**Alchemical methods** compute ΔG by coupling a Hamiltonian H(λ) that interpolates between two physical end-states (λ = 0 and λ = 1) through unphysical intermediate states. The Zwanzig relation gives ΔG = −k_B T ln⟨exp(−ΔU/k_B T)⟩₀, which converges only when the configuration spaces of the two states overlap sufficiently. BAR and MBAR exploit bidirectional sampling across multiple lambda windows to give statistically optimal estimates. No physical trajectory connects the two end-states; the path is a computational construction, but the endpoint free-energy difference is exact in the limit of convergence.

**PMF methods** compute F(ξ) = −k_B T ln P(ξ) directly along a physical collective variable (CV) by enhanced sampling. The PMF is a projection of the true free energy landscape onto one or two coordinates; its accuracy depends on the quality of the CV as a reaction coordinate.

Several physical constraints govern applicability:

- **Phase-space overlap requirement:** For FEP (Zwanzig), the configurations sampled at λ = 0 must include configurations representative of λ = 1. A metric: if exp(−ΔU/k_B T) is dominated by a handful of frames (effective sample size N_eff ≪ N), the estimate is unreliable. BAR/MBAR relax this by pooling data from all intermediate windows.
- **Soft-core potentials at lambda endpoints:** At λ → 0 or λ → 1, a fully decoupled particle has a core that overlaps with solvent, creating numerical singularities in dU/dλ and instability in the integrator. Soft-core (Beutler) potentials replace the Lennard-Jones core with a regularized form that remains finite at λ = 0. All production alchemical FEP requires soft-core LJ.
- **Standard-state correction for binding:** Absolute binding free energies ΔG_bind are referenced to a standard state of 1 M (1 mol/L). Restraints that confine the ligand during alchemical annihilation impose a loss of translational/rotational entropy that must be corrected analytically (Boresch correction). Omitting this correction can introduce errors of 10–30 kJ/mol. [EXPERT REVIEW NEEDED]
- **Thermodynamic cycle closure:** Free energies are state functions. The sum of ΔG around a closed thermodynamic cycle must equal zero. A non-zero cycle closure error ≥ 1 kJ/mol indicates either a numerical problem (insufficient sampling in a window) or a force-field inconsistency. Use cycle closure as a convergence diagnostic.
- **Force-field accuracy is the dominant error source.** A perfectly converged FEP calculation with a poorly parameterized force field gives a precisely wrong answer. Experimental validation of the method on a known binding affinity series is required before applying it to novel systems.

## When to Use This Skill

- Hydration free energies: ΔG_hyd = ΔG(liquid) − ΔG(gas), the fundamental benchmark for force-field parameterization.
- Absolute binding free energies (ABFE) of protein–ligand complexes where a quantitative Kd estimate is needed.
- Relative binding free energies (RBFE): comparing two ligands that share a common scaffold, where the alchemical perturbation is small.
- Solvation and adsorption free energies for ions, small molecules, or polymers in solvents or at interfaces.
- Conformational free energies: the ΔG between two structural states of a peptide or polymer when a PMF along a structural CV is desired.
- Pore or channel permeation free energies (PMF of ion along channel z-axis) where no good force-field parameterization of the transition is available and enhanced sampling is used instead.

## When Not to Use This Skill

- **When the process involves bond breaking or formation.** Alchemical FEP is defined for fixed-topology Hamiltonians. Reactive transitions require QM/MM or ReaxFF and different free-energy methods (finite-temperature string, metadynamics with bond-order CV).
- **When a quick estimate suffices.** MM/GBSA or MM/PBSA post-processing of an existing trajectory gives rough ΔG estimates (±5–15 kJ/mol) in minutes. Alchemical FEP is 100–1000× more expensive but gives uncertainties of 1–3 kJ/mol if converged. Use the cheaper method first for screening.
- **When the force field is not validated for the system of interest.** An un-validated force field for a ligand functional group (rare tautomer, exotic heteroatom, unusual formal charge) makes any FEP result unreliable regardless of statistical convergence.
- **When the timescale problem is not solved by the lambda schedule.** If the protein undergoes a slow conformational change that couples to ligand binding, alchemical FEP will not converge unless the protein conformational change is also sampled at every lambda window. This is the most common cause of hysteresis in ABFE calculations.
- **When enhanced sampling has not been applied to a high-barrier PMF.** Plain MD umbrella sampling fails if the unbiased trajectories within each window do not visit the window center. Verify that the equilibrium fluctuation of the CV exceeds the window spacing before trusting a WHAM/MBAR reconstruction.

## Core Concepts

- **Free energy perturbation (FEP / Zwanzig equation):** ΔG(0→1) = −k_B T ln⟨exp(−(U_1 − U_0)/k_B T)⟩₀. Converges only when the phase-space overlap between state 0 and state 1 is high. Direct application is limited to very small perturbations (ΔU ≲ 3 k_B T for most frames). Used in practice as the foundation for lambda-window methods.
- **Thermodynamic integration (TI):** ΔG(0→1) = ∫₀¹ ⟨∂H/∂λ⟩_λ dλ. The derivative ⟨∂H/∂λ⟩_λ is ensemble-averaged at each lambda window; the integral is evaluated numerically (trapezoidal or Gaussian quadrature). TI requires ⟨∂H/∂λ⟩_λ to be smooth in λ; discontinuities at λ endpoints indicate soft-core or sampling problems.
- **Bennett Acceptance Ratio (BAR):** Solves the implicit equation ΔG = k_B T ln[⟨f(U₀ − U₁ − C)⟩₁ / ⟨f(U₁ − U₀ + C)⟩₀] + C, where f is the Fermi function. BAR uses samples from both states simultaneously and is the minimum-variance estimator for two-state free energy differences. Implemented in pymbar.
- **Multistate BAR (MBAR):** Generalizes BAR to N states simultaneously. Uses all samples from all lambda windows to estimate all pairwise free energy differences and their covariance. Statistically optimal; equivalent to BAR for two states but significantly more accurate when N ≥ 3 states and their overlaps are heterogeneous. Implemented in pymbar; integrated into alchemlyb.
- **Lambda coupling parameter (λ):** Parameterizes the Hamiltonian between two end-states. For a disappearing ligand: λ = 0 (ligand fully interacting) → λ = 1 (ligand fully decoupled). Separate lambda schedules for electrostatics (q-lambda) and LJ (vdw-lambda) improve numerical stability: decouple charges first (λ_q: 0→1), then LJ (λ_vdw: 0→1). GROMACS uses separate lambda vectors for each energy term.
- **Soft-core LJ (Beutler potential):** V_sc(r, λ) = 4 ε λ^n [(α_LJ σ^6 (1−λ)^m + r^6)^{−2} − (α_LJ σ^6 (1−λ)^m + r^6)^{−1}]. The α_LJ parameter (typically 0.5) and softcore power (sc-power = 1 in GROMACS) regularize the λ → 0 singularity. Without soft-core, ∂H/∂λ diverges as λ → 0 for an annihilating atom and the simulation crashes.
- **dhdl.xvg (GROMACS free-energy output):** At each lambda window, GROMACS writes ∂H/∂λ values (for TI) and H(λ) − H(λ_current) for all other lambda states (for BAR/MBAR) to `dhdl.xvg` at every `nstdhdl` step. alchemlyb reads all `dhdl.xvg` files and extracts the data in a consistent pandas DataFrame format.
- **Phase-space overlap matrix:** An N_windows × N_windows matrix O_ij = ⟨exp(−(U_j − U_i)/2k_B T)⟩_i² / (⟨exp(−(U_j − U_i)/k_B T)⟩_i × ⟨1⟩_j). Diagonal entries are 1; off-diagonal entries near 1 indicate good overlap; entries ≈ 0 indicate that windows i and j have no configurational overlap and the free energy estimate between them is unreliable. Computed by pymbar.MBAR.compute_overlap_matrix().
- **Boresch absolute binding free energy and standard-state correction:** When a ligand is alchemically annihilated in the complex, its translational and rotational entropy is restricted by harmonic restraints (3 distance + 3 angle + 3 dihedral). The correction for removing these restraints and converting to 1 M standard state is: ΔG_std = −k_B T ln[8 π² C° r₀² sin(θ_A0) sin(θ_B0) / (K_r K_{θA} K_{θB} K_{φA} K_{φB} K_{φC})^{1/2} (2πk_BT)^3]. The analytical expression requires the equilibrium restraint geometry and spring constants. [EXPERT REVIEW NEEDED: numerical value depends sensitively on restraint parameters]
- **Thermodynamic cycle:** For RBFE (relative binding FE), compute ΔΔG_bind = ΔG_complex(A→B) − ΔG_solvent(A→B) where A and B are two ligands. Both legs are alchemical perturbations. The relative result ΔΔG_bind does not require absolute free energies; only the difference between the two alchemical legs. Cycle closure: ΔG_complex + ΔG_bind(B) − ΔG_solvent − ΔG_bind(A) = 0.
- **Hysteresis:** The forward ΔG(0→1) and reverse ΔG(1→0) disagree when computed independently from Zwanzig. Hysteresis magnitude |ΔG_forward + ΔG_reverse| > 1 kJ/mol indicates insufficient phase-space overlap or a slow conformational change coupled to the alchemical path. BAR/MBAR reduce hysteresis by optimally combining forward and reverse samples, but cannot eliminate it from insufficient sampling.
- **Effective sample size (N_eff):** N_eff = (Σ w_i)² / Σ w_i² where w_i = exp(−U/k_B T). For a good FEP estimate: N_eff/N > 0.1. If N_eff/N < 0.01, the estimate is unreliable regardless of N; more intermediate lambda windows are required.

## Key Workflows

### Workflow 1: GROMACS lambda schedule and soft-core setup for hydration FEP

Computing the hydration free energy of ethanol: ΔG_hyd = ΔG_vac(couple→decouple) − ΔG_wat(couple→decouple). Full lambda schedule with separate electrostatics and LJ decoupling, soft-core LJ, and dhdl output.

```ini
; ── fep_lambda.mdp — lambda window for alchemical decoupling ─────────────────
; System: ethanol in TIP3P water (hydration FEP) or in vacuum (gas-phase FEP)
; Lambda vector: separate q-lambda (charges) and vdw-lambda (LJ) schedules
; Convention: lambda=0 → fully coupled; lambda=1 → fully decoupled
;
; Standard schedule (20 windows):
;   q-lambda:   0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 [11 windows]
;   vdw-lambda: 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 [fixed at 0 while charges decouple]
; then:
;   q-lambda:   1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 [9 windows, fixed at 1]
;   vdw-lambda: 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.9 1.0 [LJ decouple, non-uniform spacing]
; Decouple charges FIRST to avoid the Born-charging artifact from a neutral particle
; with remaining LJ core. [EXPERT REVIEW NEEDED: optimal lambda spacing is system-dependent]

; ── Core free-energy parameters ──────────────────────────────────────────────
free-energy              = yes
couple-moltype           = ETH          ; name of the molecule type to decouple
couple-lambda0           = vdw-q        ; at lambda=0: both LJ and charges ON
couple-lambda1           = none         ; at lambda=1: both LJ and charges OFF (gas phase)
couple-intramol          = no           ; do NOT remove intramolecular interactions of ETH
                                         ; (set yes only for absolute solvation FE)
; NOTE: couple-lambda1 = q-only would decouple only charges (test partial decoupling)

; ── Lambda state selection ────────────────────────────────────────────────────
; This window is lambda state index 5 out of the total schedule
init-lambda-state        = 5

; Full lambda schedule (must be consistent across ALL windows):
; Order must match what alchemlyb expects; GROMACS reads these as the lambda vectors.
vdw-lambdas              = 0.00 0.00 0.00 0.00 0.00 0.00 0.05 0.10 0.20 0.30 0.40 0.50 0.60 0.70 0.80 0.90 1.00
coul-lambdas             = 0.00 0.25 0.50 0.75 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00
; 17 windows total: 5 charge windows + 12 LJ windows (last charge window overlaps LJ=0)
; Finer spacing near lambda=0 for LJ (0.05, 0.10, 0.20...) avoids the divergence in dH/dlambda

; ── Soft-core LJ (REQUIRED for lambda endpoints) ─────────────────────────────
sc-alpha                 = 0.5          ; Beutler soft-core alpha; 0.5 is standard
sc-power                 = 1            ; soft-core power n; 1 is standard for GROMACS
sc-sigma                 = 0.3          ; nm; minimum sigma for atoms without LJ parameters
sc-coul                  = no           ; soft-core electrostatics (not usually needed)

; ── Free-energy output ───────────────────────────────────────────────────────
nstdhdl                  = 100          ; write dH/dlambda every 100 steps (200 fs)
                                         ; Lower = more data but larger dhdl.xvg files
                                         ; Lower autocorrelation concern: use 50 steps for small molecules
dhdl-print-energy        = total        ; write total energy U(λ_k) for all states (for MBAR)
                                         ; "total" = potential + kinetic; "potential" = potential only
                                         ; MBAR needs energy differences: potential is sufficient
separate-dhdl-file       = yes          ; write dhdl to separate dhdl.xvg (not mixed into energy)
dhdl-derivatives         = yes          ; write ∂H/∂λ for each lambda type (for TI)

; ── Standard MD parameters (consistent with equilibration) ───────────────────
integrator               = md
dt                       = 0.002        ; ps
nsteps                   = 5000000      ; 10 ns per window (increase for large molecules)
cutoff-scheme            = Verlet
coulombtype              = PME
rcoulomb                 = 1.2
rvdw                     = 1.2
pme-order                = 4
fourierspacing           = 0.12         ; smaller grid for FEP accuracy
tcoupl                   = V-rescale
tc-grps                  = System
tau-t                    = 0.1
ref-t                    = 298.15
pcoupl                   = Parrinello-Rahman
tau-p                    = 2.0
ref-p                    = 1.013
compressibility          = 4.5e-5
constraints              = h-bonds
gen-vel                  = no
continuation             = yes
```

```bash
#!/bin/bash
# ── Hydration FEP workflow: run all lambda windows ────────────────────────────
# System: ethanol in water + ethanol in vacuum (two separate FEP calculations)
# 17 windows each; equilibrate each window for 2 ns, collect 10 ns production.
#
# Directory structure:
#   fep_water/lambda_00/ ... fep_water/lambda_16/
#   fep_vacuum/lambda_00/ ... fep_vacuum/lambda_16/

set -euo pipefail

N_LAMBDA=17
BASE_DIRS=("fep_water" "fep_vacuum")

for BASE in "${BASE_DIRS[@]}"; do
    mkdir -p "${BASE}"
    for i in $(seq 0 $((N_LAMBDA - 1))); do
        LAMBDA_DIR="${BASE}/lambda_$(printf '%02d' ${i})"
        mkdir -p "${LAMBDA_DIR}"

        # Write mdp with init-lambda-state = i (modify template)
        sed "s/init-lambda-state.*/init-lambda-state = ${i}/" \
            templates/fep_lambda.mdp > "${LAMBDA_DIR}/fep.mdp"

        # Equilibrate: 2 ns with the same lambda (continue from bulk NPT equilibration)
        gmx grompp \
            -f "${LAMBDA_DIR}/fep.mdp" \
            -c "initial_configs/${BASE}_equil.gro" \
            -t "initial_configs/${BASE}_equil.cpt" \
            -p "${BASE}/topol.top" \
            -o "${LAMBDA_DIR}/equil.tpr" \
            -maxwarn 0

        gmx mdrun -v -deffnm "${LAMBDA_DIR}/equil" \
            -s "${LAMBDA_DIR}/equil.tpr" \
            -ntmpi 1 -ntomp 8 -maxh 4.0

        # Production: 10 ns, collecting dhdl.xvg
        sed "s/nsteps.*/nsteps = 5000000/" "${LAMBDA_DIR}/fep.mdp" | \
            sed "s/gen-vel.*/gen-vel = no/" > "${LAMBDA_DIR}/prod.mdp"

        gmx grompp \
            -f "${LAMBDA_DIR}/prod.mdp" \
            -c "${LAMBDA_DIR}/equil.gro" \
            -t "${LAMBDA_DIR}/equil.cpt" \
            -p "${BASE}/topol.top" \
            -o "${LAMBDA_DIR}/prod.tpr" \
            -maxwarn 0

        gmx mdrun -v -deffnm "${LAMBDA_DIR}/prod" \
            -s "${LAMBDA_DIR}/prod.tpr" \
            -ntmpi 1 -ntomp 8 -maxh 12.0

        echo "Window ${i} (${BASE}): DONE"
    done
done

# Collect dhdl files:
find fep_water -name "prod.xvg" | sort > fep_water_dhdl_files.txt
find fep_vacuum -name "prod.xvg" | sort > fep_vacuum_dhdl_files.txt

echo "All windows complete. Run MBAR analysis (Workflow 2)."
```

### Workflow 2: MBAR analysis with alchemlyb

Parse GROMACS `dhdl.xvg` files with alchemlyb, run MBAR, compute ΔG with uncertainty, and validate with overlap matrix.

```python
"""
MBAR free-energy analysis for a completed hydration FEP calculation.
Uses alchemlyb (GROMACS parser + MBAR estimator) and pymbar (statistics).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from alchemlyb.parsing.gmx import extract_u_nk, extract_dHdl
from alchemlyb.estimators import MBAR, TI, BAR
from alchemlyb.preprocessing import (
    subsampling,
    statistical_inefficiency,
    decorrelate_u_nk,
)
from alchemlyb.convergence import forward_backward_convergence
from alchemlyb.visualisation import plot_mbar_overlap_matrix


def run_mbar_hydration(
    dhdl_dir_water: str,
    dhdl_dir_vacuum: str,
    temp_K: float = 298.15,
    n_discard_frac: float = 0.2,    # discard first 20% as additional equilibration
) -> dict:
    """
    Compute ΔG_hyd = ΔG_water - ΔG_vacuum from two sets of GROMACS dhdl.xvg files.

    ΔG_hyd < 0: hydration is favorable (typical for polar molecules).
    The sign convention here: lambda=0 → coupled, lambda=1 → decoupled.
    ΔG_decouple(water) = G(decoupled in water) - G(coupled in water)
    ΔG_decouple(vacuum) = G(decoupled in vacuum) - G(coupled in vacuum)
    ΔG_hyd = ΔG_decouple(vacuum) - ΔG_decouple(water)  [solvation = gas→liquid]
    (Note: some conventions invert the sign; always state the direction explicitly.)
    """
    results = {}

    for phase, dhdl_dir in [("water", dhdl_dir_water), ("vacuum", dhdl_dir_vacuum)]:
        dhdl_files = sorted(Path(dhdl_dir).glob("**/*.xvg"))
        if not dhdl_files:
            raise FileNotFoundError(f"No dhdl.xvg files found in {dhdl_dir}")
        print(f"\n{phase.upper()}: {len(dhdl_files)} lambda windows found")

        # ── Parse u_nk (energy differences for MBAR) ─────────────────────────
        u_nk_list = []
        for f in dhdl_files:
            try:
                u_nk = extract_u_nk(str(f), T=temp_K)
                u_nk_list.append(u_nk)
            except Exception as e:
                print(f"  WARNING: Failed to parse {f}: {e}")
                continue

        if not u_nk_list:
            raise RuntimeError(f"No valid dhdl files parsed for {phase}")

        u_nk_all = pd.concat(u_nk_list)

        # ── Discard equilibration ─────────────────────────────────────────────
        n_total = u_nk_all.groupby(level="time").ngroups
        t_min = u_nk_all.index.get_level_values("time").min()
        t_max = u_nk_all.index.get_level_values("time").max()
        t_discard = t_min + n_discard_frac * (t_max - t_min)
        u_nk_trimmed = u_nk_all[u_nk_all.index.get_level_values("time") > t_discard]
        print(f"  Trimmed: {t_discard:.0f}–{t_max:.0f} ps used")

        # ── Statistical inefficiency subsampling ──────────────────────────────
        # Decorrelate each lambda window independently to remove autocorrelation
        u_nk_sub = decorrelate_u_nk(u_nk_trimmed, method="dhdl_all", remove_burnin=False)
        n_frames_per_window = u_nk_sub.groupby(level="fep-lambda").size()
        print(f"  Frames per window after decorrelation:\n{n_frames_per_window.to_string()}")

        # Warn if any window has < 200 effectively independent frames
        low_windows = n_frames_per_window[n_frames_per_window < 200]
        if not low_windows.empty:
            print(f"  WARNING: Low N_eff in windows: {list(low_windows.index)}")
            print("  Consider extending simulation or reducing nstdhdl.")

        # ── MBAR estimation ───────────────────────────────────────────────────
        mbar = MBAR(method="robust")
        mbar.fit(u_nk_sub)

        delta_f = mbar.delta_f_.iloc[0, -1]   # ΔG: 0→last lambda (kJ/mol)
        delta_f_err = mbar.d_delta_f_.iloc[0, -1]
        print(f"  MBAR ΔG_{phase} = {delta_f:.3f} ± {delta_f_err:.3f} kJ/mol")

        # ── BAR and TI for cross-validation ──────────────────────────────────
        dhdl_list = [extract_dHdl(str(f), T=temp_K) for f in dhdl_files]
        dhdl_all = pd.concat(dhdl_list)
        dhdl_trimmed = dhdl_all[dhdl_all.index.get_level_values("time") > t_discard]

        bar = BAR()
        bar.fit(u_nk_sub)
        ti = TI()
        ti.fit(dhdl_trimmed)

        dg_bar = bar.delta_f_.iloc[0, -1]
        dg_ti  = ti.delta_f_.iloc[0, -1]
        print(f"  BAR  ΔG_{phase} = {dg_bar:.3f} kJ/mol")
        print(f"  TI   ΔG_{phase} = {dg_ti:.3f} kJ/mol")
        print(f"  MBAR–BAR discrepancy: {abs(delta_f - dg_bar):.2f} kJ/mol "
              f"({'OK' if abs(delta_f - dg_bar) < 0.5 else 'WARNING: > 0.5 kJ/mol'})")

        results[phase] = {
            "mbar": mbar,
            "delta_f_kJ": delta_f,
            "delta_f_err_kJ": delta_f_err,
            "u_nk": u_nk_sub,
        }

    # ── Hydration free energy ─────────────────────────────────────────────────
    dg_wat = results["water"]["delta_f_kJ"]
    dg_vac = results["vacuum"]["delta_f_kJ"]
    err_wat = results["water"]["delta_f_err_kJ"]
    err_vac = results["vacuum"]["delta_f_err_kJ"]

    dg_hyd = dg_vac - dg_wat   # solvation: favorable when negative
    err_hyd = np.sqrt(err_wat**2 + err_vac**2)

    print(f"\n{'='*60}")
    print(f"ΔG_hydration = {dg_hyd:.2f} ± {err_hyd:.2f} kJ/mol")
    print(f"  (ethanol experimental: −20.1 kJ/mol at 298 K)")
    print(f"{'='*60}")

    # ── Overlap matrix visualization ──────────────────────────────────────────
    for phase in ["water", "vacuum"]:
        mbar_obj = results[phase]["mbar"]
        try:
            ax = plot_mbar_overlap_matrix(mbar_obj.overlap_matrix)
            ax.figure.suptitle(f"MBAR overlap matrix — {phase}")
            ax.figure.savefig(f"overlap_matrix_{phase}.png", dpi=150)
            print(f"Saved: overlap_matrix_{phase}.png")
        except Exception as e:
            print(f"Overlap plot failed for {phase}: {e}")

    return results, dg_hyd, err_hyd
```

### Workflow 3: Absolute binding free energy with Boresch restraints

Computing an ABFE for a protein–ligand complex. Requires two alchemical legs (complex + solvent) and an analytical standard-state correction for the translational/rotational restraints imposed during the complex leg.

```python
"""
Absolute binding free energy (ABFE) using the double-decoupling method.

Protocol:
  Leg 1 — solvent: decouple ligand from water (ΔG_solvent)
  Leg 2 — complex: decouple ligand from complex (ΔG_complex) with Boresch restraints
  ΔG_bind = ΔG_complex − ΔG_solvent + ΔG_restraint_correction

The restraint correction accounts for the volume restriction imposed by the
harmonic restraints on the ligand pose during the complex leg.
"""

import numpy as np
from scipy import constants


def boresch_standard_state_correction(
    r0: float,           # nm; equilibrium distance (protein anchor–ligand anchor)
    theta_A0: float,     # rad; equilibrium angle A (protein–protein–ligand)
    theta_B0: float,     # rad; equilibrium angle B (protein–ligand–ligand)
    K_r: float,          # kJ/mol/nm²; spring constant for distance r
    K_thetaA: float,     # kJ/mol/rad²; spring constant for angle theta_A
    K_thetaB: float,     # kJ/mol/rad²; spring constant for angle theta_B
    K_phiA: float,       # kJ/mol/rad²; spring constant for dihedral phi_A
    K_phiB: float,       # kJ/mol/rad²; spring constant for dihedral phi_B
    K_phiC: float,       # kJ/mol/rad²; spring constant for dihedral phi_C
    temp_K: float = 298.15,
    C0_M: float = 1.0,   # M; standard concentration (1 M = 1 mol/L)
) -> float:
    """
    Analytical standard-state correction for Boresch restraints.

    Equation from Boresch et al. (2003), J. Phys. Chem. B:
    ΔG_corr = −k_B T ln [8π²C° r₀² sin(θ_A0) sin(θ_B0)
                          / ((K_r K_θA K_θB K_φA K_φB K_φC)^{1/2} (2πk_BT)³)]

    Returns ΔG_corr in kJ/mol.
    This is the free energy cost of APPLYING the restraints
    (i.e., ΔG_off → ΔG_restrained ligand at 1M).
    It must be ADDED to ΔG_complex when computing ΔG_bind.

    CRITICAL: The correction is sensitive to the choice of r₀, θ₀, and K.
    Use the average values from a restrained equilibration trajectory
    (not the crystal structure values directly). [EXPERT REVIEW NEEDED]

    Typical correction magnitudes: +10 to +30 kJ/mol (destabilizing, as expected:
    restraints reduce the ligand's translational/rotational freedom in the complex).
    """
    kT = 8.314e-3 * temp_K   # kJ/mol
    C0_nm3 = C0_M * 1e-3 * constants.N_A * 1e-27  # 1 M in nm^{-3} = 0.6022 nm^{-3}

    numerator = (8.0 * np.pi**2 * C0_nm3
                 * r0**2
                 * np.sin(theta_A0)
                 * np.sin(theta_B0))

    denom = (np.sqrt(K_r * K_thetaA * K_thetaB * K_phiA * K_phiB * K_phiC)
             * (2.0 * np.pi * kT) ** 3)

    dg_corr = -kT * np.log(numerator / denom)
    print(f"Boresch correction: ΔG_corr = {dg_corr:.3f} kJ/mol")
    print(f"  r₀={r0:.3f} nm, θ_A0={np.degrees(theta_A0):.1f}°, θ_B0={np.degrees(theta_B0):.1f}°")
    print(f"  K_r={K_r:.0f} kJ/mol/nm², K_θA={K_thetaA:.0f}, K_θB={K_thetaB:.0f} kJ/mol/rad²")
    return dg_corr


def compute_abfe(
    dg_complex_kJ: float,    # ΔG from complex alchemical leg (couple→decouple)
    dg_complex_err: float,
    dg_solvent_kJ: float,    # ΔG from solvent alchemical leg (couple→decouple)
    dg_solvent_err: float,
    dg_restraint_corr: float,  # Boresch correction (positive = destabilizing in complex)
    dg_restraint_err: float = 0.5,  # Approximate error on the restraint correction [EXPERT REVIEW NEEDED]
) -> dict:
    """
    Combine complex and solvent legs with Boresch correction.

    ΔG_bind = ΔG_complex_decouple + ΔG_restraint_corr − ΔG_solvent_decouple
            = −ΔG_couple_in_complex − ΔG_restraint_corr + ΔG_couple_in_solvent
    (Signs depend on coupling direction convention; verify for your setup.)

    Convention used here:
      ΔG_complex_decouple > 0 (costs energy to remove ligand from protein)
      ΔG_solvent_decouple > 0 (costs energy to remove ligand from water)
      ΔG_restraint_corr > 0 (restraints reduce freedom → unfavorable correction)
      ΔG_bind = ΔG_restraint_corr − ΔG_complex_decouple + ΔG_solvent_decouple

      Negative ΔG_bind = favorable binding.
    """
    dg_bind = dg_restraint_corr - dg_complex_kJ + dg_solvent_kJ
    err_bind = np.sqrt(dg_complex_err**2 + dg_solvent_err**2 + dg_restraint_err**2)

    print(f"\nABFE summary:")
    print(f"  ΔG_complex (decouple)  = {dg_complex_kJ:+.2f} ± {dg_complex_err:.2f} kJ/mol")
    print(f"  ΔG_solvent (decouple)  = {dg_solvent_kJ:+.2f} ± {dg_solvent_err:.2f} kJ/mol")
    print(f"  ΔG_restraint_corr      = {dg_restraint_corr:+.2f} ± {dg_restraint_err:.2f} kJ/mol")
    print(f"  ΔG_bind                = {dg_bind:+.2f} ± {err_bind:.2f} kJ/mol")
    print(f"  K_d estimate           = {np.exp(dg_bind / (8.314e-3 * 298.15)) * 1e3:.2e} µM")
    return {"dg_bind_kJ": dg_bind, "err_kJ": err_bind}
```

### Workflow 4: Thermodynamic integration with convergence check

TI via numerical integration of ⟨∂H/∂λ⟩_λ across the lambda schedule. Includes the TI integrand plot and Gaussian quadrature comparison.

```python
"""
Thermodynamic integration from GROMACS dhdl.xvg files.
TI is a useful cross-check for MBAR: they should agree to within 0.5 kJ/mol
for a well-converged calculation with a smooth lambda schedule.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import integrate
from alchemlyb.parsing.gmx import extract_dHdl
from alchemlyb.estimators import TI
from alchemlyb.preprocessing import decorrelate_dhdl
from pathlib import Path


def ti_from_dhdl(
    dhdl_dir: str,
    temp_K: float = 298.15,
    n_discard_frac: float = 0.2,
    plot_integrand: bool = True,
) -> dict:
    """
    Compute TI free energy from ∂H/∂λ time series.

    Returns ΔG_TI and the TI integrand ⟨∂H/∂λ⟩_λ for diagnostic inspection.

    The TI integrand should be:
    - Smooth in λ (no sharp features except near λ=0 for LJ with soft-core)
    - Have a monotonic trend or a single maximum/minimum (double peak indicates
      a hidden slow mode or phase transition along the lambda path)
    - Converge within the production window (check by computing TI from the
      first and second halves of each window)
    """
    dhdl_files = sorted(Path(dhdl_dir).glob("**/*.xvg"))
    dhdl_list = []
    for f in dhdl_files:
        try:
            dhdl = extract_dHdl(str(f), T=temp_K)
            dhdl_list.append(dhdl)
        except Exception as e:
            print(f"WARNING: {f}: {e}")
    dhdl_all = pd.concat(dhdl_list)

    # Discard equilibration
    t_max = dhdl_all.index.get_level_values("time").max()
    t_min = dhdl_all.index.get_level_values("time").min()
    t_discard = t_min + n_discard_frac * (t_max - t_min)
    dhdl_trimmed = dhdl_all[dhdl_all.index.get_level_values("time") > t_discard]

    # Decorrelate
    dhdl_dec = decorrelate_dhdl(dhdl_trimmed, remove_burnin=False)

    # TI estimator
    ti = TI()
    ti.fit(dhdl_dec)
    dg_ti = ti.delta_f_.iloc[0, -1]
    dg_ti_err = ti.d_delta_f_.iloc[0, -1]

    # Extract integrand for plotting
    integrand = ti.dhdl  # ⟨∂H/∂λ⟩ per lambda state

    if plot_integrand:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Check if there are separate vdw and coul lambda components
        lambda_cols = [c for c in integrand.columns if "lambda" in c.lower() or
                       c in ("vdw", "coul", "fep")]
        if not lambda_cols:
            lambda_cols = list(integrand.columns)

        for col in lambda_cols:
            lam_vals = integrand.index.get_level_values(col)
            dhdl_mean = integrand.groupby(level=col)[col].mean() if col in integrand else None

        # Plot all columns of integrand
        for col in integrand.columns:
            vals = integrand[col]
            x = np.arange(len(vals))
            axes[0].plot(x, vals, "o-", label=col, markersize=4)
        axes[0].set_xlabel("Lambda state index")
        axes[0].set_ylabel("⟨∂H/∂λ⟩ (kJ/mol)")
        axes[0].set_title(f"TI integrand — ΔG = {dg_ti:.2f} ± {dg_ti_err:.2f} kJ/mol")
        axes[0].legend()
        axes[0].axhline(0, color="k", lw=0.5)

        # Convergence: forward vs backward half
        n_per_lambda = dhdl_dec.groupby(
            level=[c for c in dhdl_dec.index.names if "lambda" in c.lower()]
        ).size()
        print(f"Frames per lambda (after decorrelation):\n{n_per_lambda.to_string()}")
        low = n_per_lambda[n_per_lambda < 100]
        if not low.empty:
            print(f"WARNING: Low N_eff at lambdas: {list(low.index)}")

        axes[1].text(0.5, 0.5,
                     f"TI  ΔG = {dg_ti:.2f} ± {dg_ti_err:.2f} kJ/mol",
                     ha="center", va="center", fontsize=14,
                     transform=axes[1].transAxes)
        axes[1].axis("off")

        plt.tight_layout()
        plt.savefig("ti_integrand.png", dpi=150)
        print("Saved: ti_integrand.png")

    print(f"\nTI ΔG = {dg_ti:.3f} ± {dg_ti_err:.3f} kJ/mol")
    return {"dg_ti": dg_ti, "err_ti": dg_ti_err, "integrand": integrand}
```

### Workflow 5: Phase-space overlap diagnostics

The overlap matrix is the primary diagnostic for whether BAR/MBAR has sufficient data to give a reliable estimate. Always inspect it before reporting ΔG.

```python
"""
Overlap diagnostics for alchemical FEP calculations.

Three levels of overlap checking:
  1. MBAR overlap matrix: visual inspection of O_ij (should be > 0.03 between adjacent windows)
  2. Per-window energy distribution: distributions from adjacent windows should visually overlap
  3. Zwanzig forward vs reverse: |ΔG_forward + ΔG_reverse| < 1 kJ/mol for a well-overlapping pair
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from alchemlyb.parsing.gmx import extract_u_nk
from alchemlyb.estimators import MBAR
from alchemlyb.preprocessing import decorrelate_u_nk


def check_phase_space_overlap(
    u_nk: pd.DataFrame,
    temp_K: float = 298.15,
    min_overlap: float = 0.03,    # minimum acceptable O_ij for adjacent windows
    plot: bool = True,
) -> dict:
    """
    Compute and visualize the MBAR overlap matrix and per-window energy distributions.

    Good overlap: O_{i,i+1} > 0.03 for all adjacent pairs.
    Poor overlap: O_{i,i+1} < 0.03 → insert intermediate lambda windows.
    """
    kT = 8.314e-3 * temp_K

    # Fit MBAR (already decorrelated u_nk assumed)
    mbar = MBAR(method="robust")
    mbar.fit(u_nk)

    O = mbar.overlap_matrix
    n_states = O.shape[0]
    lambda_states = u_nk.index.get_level_values(-1).unique().sort_values()

    # ── Check adjacent overlap ────────────────────────────────────────────────
    adj_overlaps = [O[i, i+1] for i in range(n_states - 1)]
    poor_pairs = [(i, i+1, o) for i, o in enumerate(adj_overlaps) if o < min_overlap]

    print(f"MBAR overlap matrix ({n_states}x{n_states}):")
    print(f"  Min adjacent overlap: {min(adj_overlaps):.4f}")
    print(f"  Mean adjacent overlap: {np.mean(adj_overlaps):.4f}")
    if poor_pairs:
        print(f"  POOR OVERLAP (< {min_overlap}) at pairs:")
        for i, j, o in poor_pairs:
            print(f"    Windows {i}–{j} (λ={lambda_states[i]:.3f}–{lambda_states[j]:.3f}): O={o:.4f}")
        print("  ACTION: Add intermediate lambda windows between flagged pairs.")
    else:
        print(f"  All adjacent overlaps > {min_overlap}: ADEQUATE for MBAR")

    # ── Effective sample size per window ──────────────────────────────────────
    # N_eff for the MBAR free-energy estimate at each state:
    # Use the diagonal of the inverse of the MBAR precision matrix as a proxy
    weights = mbar._mbar.W_nk  # N_frames × N_states weight matrix
    for k, lam in enumerate(lambda_states):
        w_k = weights[:, k]
        n_eff = 1.0 / np.sum(w_k**2)
        n_total = len(w_k)
        print(f"  λ={lam:.4f}: N_eff={n_eff:.0f}/{n_total} ({100*n_eff/n_total:.0f}%)"
              + (" ← LOW" if n_eff / n_total < 0.1 else ""))

    if plot:
        # ── Overlap matrix heatmap ────────────────────────────────────────────
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        im = axes[0].imshow(O, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        axes[0].set_xlabel("Lambda state j")
        axes[0].set_ylabel("Lambda state i")
        axes[0].set_title("MBAR phase-space overlap matrix")
        plt.colorbar(im, ax=axes[0])

        # Annotate with numbers for small matrices
        if n_states <= 20:
            for i in range(n_states):
                for j in range(n_states):
                    axes[0].text(j, i, f"{O[i,j]:.2f}", ha="center", va="center",
                                 fontsize=6, color="red" if O[i,j] < min_overlap else "black")

        # ── Per-window energy distributions ───────────────────────────────────
        # For each window, plot the distribution of U(λ_k) - U(λ_i) for k = i±1
        all_lambda = u_nk.index.get_level_values(-1).unique().sort_values()
        colors = plt.cm.viridis(np.linspace(0, 1, min(len(all_lambda), 10)))

        for idx in range(min(len(all_lambda), 10)):
            lam = all_lambda[idx]
            mask = u_nk.index.get_level_values(-1) == lam
            data = u_nk[mask]

            # Energy difference to adjacent state
            if idx + 1 < len(all_lambda):
                lam_next = all_lambda[idx + 1]
                if lam_next in data.columns:
                    delta_u = data[lam_next] - data[lam]
                    axes[1].hist(delta_u.values / kT, bins=50, density=True,
                                 alpha=0.4, color=colors[idx],
                                 label=f"λ={lam:.3f}→{lam_next:.3f}")

        axes[1].set_xlabel("ΔU/kT (energy difference to next window)")
        axes[1].set_ylabel("Density")
        axes[1].set_title("Energy distributions: adjacent windows\n(distributions must overlap)")
        axes[1].legend(fontsize=7, ncol=2)
        axes[1].axvline(0, color="k", lw=0.5)

        plt.tight_layout()
        plt.savefig("overlap_diagnostics.png", dpi=150)
        print("Saved: overlap_diagnostics.png")

    return {
        "overlap_matrix": O,
        "adj_overlaps": adj_overlaps,
        "poor_pairs": poor_pairs,
        "mbar": mbar,
    }


def zwanzig_hysteresis_check(
    u_nk: pd.DataFrame,
    temp_K: float = 298.15,
    max_acceptable_hysteresis: float = 2.0,  # kJ/mol
) -> dict:
    """
    Compute the FEP hysteresis: |ΔG_forward(Zwanzig) + ΔG_reverse(Zwanzig)|.

    A large hysteresis indicates poor phase-space overlap between end states.
    BAR/MBAR will still give a better estimate than either Zwanzig direction,
    but hysteresis > 5 kJ/mol indicates the calculation is unreliable overall.
    """
    kT = 8.314e-3 * temp_K
    lambdas = u_nk.index.get_level_values(-1).unique().sort_values()

    lam0, lam1 = lambdas[0], lambdas[-1]

    # Forward: Zwanzig from state 0
    u0 = u_nk[u_nk.index.get_level_values(-1) == lam0]
    delta_u_01 = u0[lam1] - u0[lam0] if lam1 in u0.columns else None

    # Reverse: Zwanzig from state 1 (perturbation in opposite direction)
    u1 = u_nk[u_nk.index.get_level_values(-1) == lam1]
    delta_u_10 = u1[lam0] - u1[lam1] if lam0 in u1.columns else None

    results = {}
    if delta_u_01 is not None and delta_u_10 is not None:
        dg_fwd = -kT * np.log(np.mean(np.exp(-delta_u_01.values / kT)))
        dg_rev = -kT * np.log(np.mean(np.exp(-delta_u_10.values / kT)))

        hysteresis = abs(dg_fwd + dg_rev)   # should be ~0 if converged
        print(f"\nZwanzig forward ΔG (0→1):  {dg_fwd:+.2f} kJ/mol")
        print(f"Zwanzig reverse ΔG (1→0): {dg_rev:+.2f} kJ/mol")
        print(f"Hysteresis |ΔG_fwd + ΔG_rev|: {hysteresis:.2f} kJ/mol "
              f"({'OK' if hysteresis < max_acceptable_hysteresis else 'LARGE — check overlap'})")

        results = {"dg_fwd": dg_fwd, "dg_rev": dg_rev, "hysteresis": hysteresis}
    else:
        print("WARNING: Both end-state lambda values not found in u_nk columns; "
              "verify that dhdl-print-energy = total in the mdp.")

    return results
```

### Workflow 6: Convergence diagnostics — forward/backward block analysis

The standard convergence test: compute cumulative ΔG from the beginning and from the end of each window's trajectory. A converged calculation should show both curves plateau to the same value.

```python
"""
alchemlyb forward-backward convergence analysis.

Runs MBAR on the first t%, 2t%, ..., 100% of the trajectory and on the
last t%, 2t%, ..., 100%, and overlays both curves. Convergence is indicated
when both curves plateau and converge to the same ΔG value.
"""

import numpy as np
import matplotlib.pyplot as plt
from alchemlyb.convergence import forward_backward_convergence
from alchemlyb.estimators import MBAR, BAR
from alchemlyb.visualisation import plot_convergence
import pandas as pd


def fep_convergence_analysis(
    u_nk: pd.DataFrame,
    temp_K: float = 298.15,
    num_points: int = 10,    # number of time fractions to test
    estimator_cls = MBAR,    # or BAR
    plot: bool = True,
) -> dict:
    """
    Forward-backward convergence test for MBAR.

    Returns convergence_result dict with forward and backward ΔG vs time fraction.
    The max spread between forward and backward estimates is reported as
    an additional uncertainty estimate beyond the statistical error.
    """
    convergence = forward_backward_convergence(
        [u_nk],
        estimator=estimator_cls.__name__.lower(),
        num_points=num_points,
    )

    if plot:
        ax = plot_convergence(convergence)
        ax.set_title(f"Forward-backward convergence — {estimator_cls.__name__}")
        ax.figure.savefig("fep_convergence.png", dpi=150, bbox_inches="tight")
        print("Saved: fep_convergence.png")

    # Extract final forward and backward estimates
    fwd_vals = convergence["Forward"]["values"]
    bwd_vals = convergence["Backward"]["values"]
    fwd_final = fwd_vals.iloc[-1]
    bwd_final = bwd_vals.iloc[-1]

    spread = abs(fwd_final - bwd_final)
    print(f"\nConvergence analysis:")
    print(f"  Forward (full data):  ΔG = {fwd_final:.3f} kJ/mol")
    print(f"  Backward (full data): ΔG = {bwd_final:.3f} kJ/mol")
    print(f"  Forward–backward spread: {spread:.3f} kJ/mol")

    if spread < 0.5:
        print("  CONVERGED (spread < 0.5 kJ/mol)")
    elif spread < 2.0:
        print("  MARGINAL — consider extending simulation")
    else:
        print("  NOT CONVERGED — extend simulation or add lambda windows")

    # Stationarity check: is the last 25% of each window consistent with
    # the MBAR estimate from the full trajectory?
    return {
        "forward": fwd_vals.values.tolist(),
        "backward": bwd_vals.values.tolist(),
        "forward_backward_spread_kJ": spread,
        "converged": spread < 0.5,
    }


def block_averaging_fep(
    u_nk: pd.DataFrame,
    n_blocks: int = 5,
    estimator_cls = MBAR,
    temp_K: float = 298.15,
) -> dict:
    """
    Estimate statistical uncertainty via block averaging of the full u_nk dataset.

    Divide the trajectory (across all lambda windows simultaneously) into
    n_blocks equal time segments. Run MBAR on each block. Report:
    - Block mean ± block std (independent of the MBAR internal error estimate)
    - Whether the MBAR internal error underestimates the true statistical error.
    """
    all_times = u_nk.index.get_level_values("time").unique().sort_values()
    n_times = len(all_times)
    block_size = n_times // n_blocks

    block_dg = []
    for b in range(n_blocks):
        t_start = all_times[b * block_size]
        t_end = all_times[min((b + 1) * block_size - 1, n_times - 1)]
        mask = ((u_nk.index.get_level_values("time") >= t_start) &
                (u_nk.index.get_level_values("time") <= t_end))
        block_data = u_nk[mask]

        try:
            est = estimator_cls(method="robust")
            est.fit(block_data)
            dg = est.delta_f_.iloc[0, -1]
            block_dg.append(dg)
            print(f"  Block {b+1}: ΔG = {dg:.3f} kJ/mol")
        except Exception as e:
            print(f"  Block {b+1}: FAILED ({e})")

    if len(block_dg) >= 2:
        block_mean = np.mean(block_dg)
        block_sem  = np.std(block_dg) / np.sqrt(len(block_dg))
        print(f"\nBlock averaging: ΔG = {block_mean:.3f} ± {block_sem:.3f} kJ/mol "
              f"(from {len(block_dg)} blocks)")
        return {"block_mean": block_mean, "block_sem": block_sem, "block_values": block_dg}
    return {"block_values": block_dg}
```

### Workflow 7: Relative binding free energy via thermodynamic cycle

RBFE is the standard workflow for lead optimization in drug discovery. Two ligands A and B are connected by an alchemical perturbation in both the protein complex and in water.

```python
"""
Relative binding free energy (RBFE) via thermodynamic double-perturbation cycle.

ΔΔG_bind(A→B) = ΔG_complex(A→B) − ΔG_solvent(A→B)

Convention (double-decoupling / alchemical mutation):
  λ=0: ligand A; λ=1: ligand B
  ΔG_complex: morph A→B in the protein binding site
  ΔG_solvent: morph A→B in water
  ΔΔG_bind = ΔG_complex − ΔG_solvent

  Negative ΔΔG_bind: B binds more tightly than A.

Thermodynamic cycle closure:
  ΔG_bind(A) + ΔG_complex(A→B) − ΔG_bind(B) − ΔG_solvent(A→B) = 0
  Cycle closure error should be < 1 kJ/mol for a self-consistent calculation.
"""

import numpy as np
import pandas as pd
from alchemlyb.parsing.gmx import extract_u_nk
from alchemlyb.estimators import MBAR
from alchemlyb.preprocessing import decorrelate_u_nk
from pathlib import Path


def compute_rbfe(
    complex_dhdl_dir: str,   # lambda windows for A→B morphing in complex
    solvent_dhdl_dir: str,   # lambda windows for A→B morphing in water
    temp_K: float = 298.15,
    n_discard_frac: float = 0.2,
    known_dg_bind_A: float = None,   # kJ/mol; from experiment or ABFE for cycle check
    known_dg_bind_B: float = None,
) -> dict:
    """
    Run MBAR on both complex and solvent legs of an RBFE calculation.
    """
    results = {}

    for phase, dhdl_dir in [("complex", complex_dhdl_dir), ("solvent", solvent_dhdl_dir)]:
        dhdl_files = sorted(Path(dhdl_dir).glob("**/*.xvg"))
        print(f"\n{phase.upper()}: {len(dhdl_files)} windows")

        u_nk_list = [extract_u_nk(str(f), T=temp_K) for f in dhdl_files]
        u_nk_all = pd.concat(u_nk_list)

        t_max = u_nk_all.index.get_level_values("time").max()
        t_min = u_nk_all.index.get_level_values("time").min()
        t_discard = t_min + n_discard_frac * (t_max - t_min)
        u_nk_trimmed = u_nk_all[u_nk_all.index.get_level_values("time") > t_discard]
        u_nk_dec = decorrelate_u_nk(u_nk_trimmed, remove_burnin=False)

        mbar = MBAR(method="robust")
        mbar.fit(u_nk_dec)
        dg = mbar.delta_f_.iloc[0, -1]
        dg_err = mbar.d_delta_f_.iloc[0, -1]
        print(f"  MBAR ΔG_{phase}(A→B) = {dg:.3f} ± {dg_err:.3f} kJ/mol")
        results[phase] = {"dg": dg, "err": dg_err}

    # ── RBFE result ───────────────────────────────────────────────────────────
    ddg = results["complex"]["dg"] - results["solvent"]["dg"]
    ddg_err = np.sqrt(results["complex"]["err"]**2 + results["solvent"]["err"]**2)

    print(f"\nRBFE result:")
    print(f"  ΔG_complex(A→B) = {results['complex']['dg']:+.2f} ± {results['complex']['err']:.2f} kJ/mol")
    print(f"  ΔG_solvent(A→B) = {results['solvent']['dg']:+.2f} ± {results['solvent']['err']:.2f} kJ/mol")
    print(f"  ΔΔG_bind(A→B)   = {ddg:+.2f} ± {ddg_err:.2f} kJ/mol")
    print(f"  {'B binds STRONGER than A' if ddg < 0 else 'A binds STRONGER than B'}")

    # ── Thermodynamic cycle closure ───────────────────────────────────────────
    if known_dg_bind_A is not None and known_dg_bind_B is not None:
        closure = known_dg_bind_A + results["complex"]["dg"] - \
                  known_dg_bind_B - results["solvent"]["dg"]
        print(f"\n  Cycle closure error: {closure:.2f} kJ/mol "
              f"({'OK' if abs(closure) < 1.0 else 'FAIL — check sampling'})")

    return {"ddg_kJ": ddg, "ddg_err_kJ": ddg_err, **results}
```

## Common Failure Modes

| Failure | Symptom | Likely Cause | Diagnostic | Fix |
|---------|---------|--------------|------------|-----|
| Poor phase-space overlap | MBAR overlap matrix O_{i,i+1} < 0.03; ΔG estimate changes drastically with added windows; large forward-backward hysteresis | Lambda windows too sparse; the energy gap between adjacent states is too large | `plot_mbar_overlap_matrix(mbar.overlap_matrix)`; if any adjacent O < 0.03, overlap is insufficient | Add intermediate lambda windows between the under-overlapping pair; halve the lambda spacing in the affected region |
| Soft-core not enabled at LJ endpoints | `gmx mdrun` crashes with energy explosion at λ=0 or λ=1 LJ window; dH/dλ diverges at the endpoints | LJ potential not soft-cored at the vanishing end; the atom overlaps with solvent during LJ annihilation | Check `sc-alpha = 0` or `free-energy = no` in mdp; grep for "LINCS WARNING" or "NaN in forces" in log | Set `sc-alpha = 0.5`, `sc-power = 1`, `sc-sigma = 0.3` in mdp; verify `couple-lambda0/1` settings |
| Insufficient equilibration per window | ΔG from the first half of the trajectory differs from the second half by > 1 kJ/mol; forward-backward convergence curves do not plateau | Lambda window started from an unequilibrated structure; slow protein conformational change not sampled at a specific window | Forward-backward convergence plot shows no plateau; block std > 1 kJ/mol | Extend equilibration per window (add 2–5 ns equilibration before production); verify by checking that equilibration is complete in each window independently |
| Hidden slow mode / hysteresis | ΔΔG from RBFE differs by > 2 kJ/mol between independent repeats; block averaging gives large std; protein Cα RMSD drifts differently in complex vs solvent legs | A protein loop or water network coupled to ligand binding relaxes slowly; it samples different conformations in different lambda windows or runs | Compute protein RMSD and ligand RMSD per lambda window; if RMSD differs by > 0.2 nm between early and late windows, a slow mode is active | Apply enhanced sampling (metadynamics on the relevant protein mode) at each lambda window; or use replica exchange combined with lambda sampling (HREX) |
| Standard-state correction omitted in ABFE | ABFE ΔG_bind disagrees with experiment by 10–30 kJ/mol; agreement improves only for relative differences (RBFE) | Boresch correction not applied or applied with wrong restraint geometry/spring constants | Recompute the Boresch correction using the actual restraint parameters from the mdp and the average geometry from the restrained trajectory | Apply analytical Boresch correction (Workflow 3); verify that restraint geometry r₀, θ_A0, θ_B0 was measured from the restrained trajectory, not from the crystal structure |
| Unstable lambda endpoint (λ=0 or λ=1) | Simulation crashes or has high forces at the very first or last lambda window; dH/dλ vs time shows large spikes | Fully decoupled atom has no forces from the environment; Langevin thermostat then drives it to the boundary or another atom's core | Check the minimum-distance analysis for the decoupled molecule across the trajectory; if it approaches another atom within 0.1 nm, the endpoint is unstable | Add a flat-bottom distance restraint on the decoupled molecule relative to the binding site or box center; or use positional restraints on the decoupled molecule for endpoint windows only |
| Incorrect cycle closure | ΔG_A + ΔG_AB − ΔG_B − ΔG_BA ≠ 0 by > 1 kJ/mol; different RBFE cycles give inconsistent results | Insufficient sampling in one or more legs; MBAR convergence failure at a specific lambda pair | Compute per-leg block averaging uncertainty; the leg with the largest block std is the most likely culprit | Extend the unconverged leg; check overlap matrix for gaps; consider adding more lambda windows to the problematic leg |
| Reporting ΔG without uncertainty | Published ΔG = −35.2 kJ/mol with no confidence interval; another group's repeat gives −28.9 kJ/mol | Forward-backward convergence not checked; no block analysis performed; single run reported | Block averaging (Workflow 6) and forward-backward convergence (Workflow 6) are mandatory before reporting | Always report: MBAR internal error, block averaging std, forward-backward spread; if the three estimates disagree by > 1 kJ/mol, the calculation is not converged |
| Wrong lambda coupling for intramolecular FEP | Intramolecular energetics of the decoupled molecule change with lambda when they should not; ΔG has unexpected sign | `couple-intramol = yes` removes intramolecular interactions of the ligand, introducing a gas-phase reference instead of the vacuum reference | Check `couple-intramol` in mdp; if ΔG_hyd is expected for the molecule in its own force-field environment, use `couple-intramol = no` | Set `couple-intramol = no` for standard hydration FEP; use `yes` only when computing the absolute chemical potential (Kirkwood charging) |
| alchemlyb parsing failure | `extract_u_nk` returns empty DataFrame; MBAR has no data for one or more windows | `dhdl-print-energy = no` in mdp; `separate-dhdl-file = no`; output file truncated by HPC wall time | Check that `dhdl.xvg` exists in each lambda directory and is not empty; `wc -l dhdl.xvg` should match `nsteps / nstdhdl` | Set `dhdl-print-energy = total` and `separate-dhdl-file = yes` in all window mdp files; check HPC output for premature termination |

## Best Practices

- **Always decouple electrostatics before LJ.** Remove partial charges first (q-lambda: 0→1 with vdw-lambda = 0), then remove LJ (vdw-lambda: 0→1 with q-lambda = 1). Decoupling LJ first from a charged particle involves a Born-charging artifact where a neutral hard sphere in water has an anomalously high free energy of hydration.
- **Use MBAR over TI or BAR for all production calculations.** MBAR is statistically optimal and uses all data from all windows simultaneously. TI is a useful cross-check; agreement between MBAR and TI to within 0.5 kJ/mol is a necessary (not sufficient) convergence indicator.
- **Always inspect the MBAR overlap matrix before reporting ΔG.** A single adjacent window pair with O < 0.03 will make the entire MBAR estimate unreliable. This check takes 10 seconds and prevents reporting a wrong number with false confidence.
- **Use the forward-backward convergence plot as the primary convergence metric.** The MBAR internal error underestimates the true statistical error when autocorrelation is high. The forward-backward spread and block averaging std provide independent estimates of uncertainty that should be reported alongside the MBAR error.
- **Equilibrate each lambda window independently before production.** Starting every window from the same structure (e.g., the NPT equilibrated complex at λ=0) and collecting production immediately leads to slow modes being unsampled in windows far from λ=0. Each window needs 1–5 ns of equilibration in the lambda subspace. The `init-lambda-state` mdp parameter selects the equilibration lambda.
- **For ABFE: always compute and report the Boresch correction.** The standard-state correction for translational and rotational restraints is always significant (typically +10 to +30 kJ/mol) and is exact analytically given the restraint parameters. Never report an ABFE without it, and always state the restraint parameters and correction value in the methods.
- **Run at least three independent repeats for any ABFE or RBFE result.** The spread between independent runs (different initial velocities, different initial solvent configurations) is a practical estimate of the systematic convergence error. Publication-quality results require N ≥ 3 repeats. [EXPERT REVIEW NEEDED]
- **Use `nstdhdl = 50–100` for small molecules, `nstdhdl = 100–250` for protein-ligand.** Too large: autocorrelation time not captured by the dhdl output, leading to underestimated effective sample size. Too small: file size becomes the bottleneck (a 100 ns run at nstdhdl=10 produces ~100 GB of dhdl.xvg for 20 lambda windows). Balance accuracy and storage.
- **For TI: use finer lambda spacing near λ=0 for LJ annihilation.** The TI integrand ⟨∂H/∂λ⟩_λ diverges as λ→0 for LJ interactions even with soft-core (it remains finite but rises steeply). Gaussian quadrature with optimal nodes is more accurate than uniform spacing for the LJ leg; use points at λ = 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 rather than uniform 0.1 increments.

## Integration with Other Skills

- **[enhanced-sampling](../enhanced-sampling/SKILL.md):** Umbrella sampling and metadynamics-based PMF calculations overlap directly with free-energy MD. That skill covers PLUMED CV setup, WTM, and PMF reweighting. This skill covers alchemical FEP/MBAR analysis. For systems where both a physical CV (for PMF) and an alchemical path (for ΔG_bind) are relevant, the two approaches provide complementary information.
- **[gromacs-workflow](../gromacs-workflow/SKILL.md):** GROMACS provides the simulation engine and the `free-energy` mdp section for all alchemical calculations. The `.top`, `.mdp`, `grompp → mdrun` pipeline, and PME setup from that skill are direct prerequisites for any GROMACS-based FEP calculation.
- **[lammps-workflow](../lammps-workflow/SKILL.md):** LAMMPS implements alchemical FEP via the `fix adapt` and `compute fep` commands. For materials-science systems (ion insertion, solvation in non-aqueous solvents, surface adsorption) where GROMACS force fields are not available, LAMMPS is the standard FEP engine.
- **[md-equilibration](../md-equilibration/SKILL.md):** Each lambda window must be independently equilibrated before production. The equilibration criteria from that skill (block averaging of temperature, pressure, and structural observables; stationarity tests) apply per window, not just for the λ=0 starting state.
- **[md-analysis](../md-analysis/SKILL.md):** Block averaging, autocorrelation analysis, and effective sample size estimation from that skill are prerequisites for interpreting dhdl.xvg time series. Trajectory-level structural checks (ligand RMSD, protein RMSD, water coordination number vs. λ) verify that each window is sampling the correct state.
- **[md-force-fields](../md-force-fields/SKILL.md):** Force-field accuracy is the dominant error source in free-energy calculations. GAFF2 and CGenFF parameterization for ligands, water model selection, and ionic parameter quality all directly affect the accuracy of ΔG_hyd and ΔG_bind estimates. That skill covers validation benchmarks and known limitations.
- **[coarse-grained-md](../coarse-grained-md/SKILL.md) [FORWARD REFERENCE — skill not yet created]:** Martini CG free-energy calculations are an important screening tool for membrane permeability and lipid–ligand interactions. The alchemical methods described here apply to CG force fields with modified lambda schedules appropriate for CG bead resolution.
- **[mlp-validation](../../ml-interatomic-potentials/mlp-validation/SKILL.md):** Free-energy calculations with ML interatomic potentials (MACE, NequIP) require that the MLP is validated across the full alchemical path from λ=0 to λ=1. Committee uncertainty must be checked at each lambda window; windows where the MLP is extrapolating produce unreliable dhdl values.
- **[scientific-visualization](../../scientific-data/scientific-visualization/SKILL.md):** MBAR overlap matrix heatmaps, TI integrand plots, forward-backward convergence plots, PMF profiles with uncertainty envelopes, and thermodynamic cycle diagrams are all standard FEP output figures requiring the matplotlib patterns from that skill.
- **[reproducibility-workflow](../../scientific-data/reproducibility-workflow/SKILL.md):** Free-energy calculations require archiving all lambda-window mdp files, topology, dhdl.xvg files, and the alchemlyb analysis script. The GROMACS `.tpr` for each window is the canonical reproducibility record. That skill covers DVC-based provenance and the run-metadata structure for large multi-window datasets.

## Key References

1. Zwanzig, R. W. (1954). *High-temperature equation of state by a perturbation method. I. Nonpolar gases.* Journal of Chemical Physics, 22(8), 1420–1426. https://doi.org/10.1063/1.1740409
2. Bennett, C. H. (1976). *Efficient estimation of free energy differences from Monte Carlo data.* Journal of Computational Physics, 22(2), 245–268. https://doi.org/10.1016/0021-9991(76)90078-4
3. Shirts, M. R. & Chodera, J. D. (2008). *Statistically optimal analysis of samples from multiple equilibrium states.* Journal of Chemical Physics, 129(12), 124105. https://doi.org/10.1063/1.2978177
4. Boresch, S., Tettinger, F., Leitgeb, M., & Karplus, M. (2003). *Absolute binding free energies: A quantitative approach for their calculation.* Journal of Physical Chemistry B, 107(35), 9535–9551. https://doi.org/10.1021/jp0217839
5. Klimovich, P. V., Shirts, M. R., & Mobley, D. L. (2015). *Guidelines for the analysis of free energy calculations.* Journal of Computer-Aided Molecular Design, 29(5), 397–411. https://doi.org/10.1007/s10822-015-9840-9
6. Cournia, Z., Allen, B., & Sherman, W. (2017). *Relative binding free energy calculations in drug discovery: Recent advances and practical considerations.* Journal of Chemical Information and Modeling, 57(12), 2911–2937. https://doi.org/10.1021/acs.jcim.7b00564
7. Pohorille, A., Jarzynski, C., & Chipot, C. (2010). *Good practices in free-energy calculations.* Journal of Physical Chemistry B, 114(32), 10235–10253. https://doi.org/10.1021/jp102971x
8. Gapsys, V., Pérez-Benito, L., Aldeghi, M., Seeliger, D., van Vlijmen, H., Tresadern, G., & de Groot, B. L. (2020). *Large scale relative protein–ligand binding affinities using non-equilibrium alchemy.* Chemical Science, 11(4), 1140–1152. https://doi.org/10.1039/C9SC03754C
