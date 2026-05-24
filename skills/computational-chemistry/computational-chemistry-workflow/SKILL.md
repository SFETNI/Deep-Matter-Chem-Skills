---
name: computational-chemistry-workflow
domain: computational-chemistry
maturity: draft
tools:
  - ORCA 5.0.4
  - xTB 6.6.1
  - RDKit 2023.9.4
  - numpy 1.26.4
  - matplotlib 3.8.0
  - pandas 2.2.0
tested_versions:
  ORCA: "5.0.4"
  xTB: "6.6.1"
  RDKit: "2023.9.4"
  numpy: "1.26.4"
  matplotlib: "3.8.0"
  pandas: "2.2.0"
requires_gpu: false
estimated_runtime: seconds-to-days
---

# Computational Chemistry Workflow

## Description

This skill covers end-to-end quantum chemistry workflows for molecules, reactions, and molecular materials: structure preparation and standardization; geometry optimization; frequency analysis and thermochemistry; transition-state search and reaction-profile construction; method and basis-set selection and benchmarking; solvation modeling; open-shell and multireference diagnostics; excited-state calculations; and coupling to molecular featurization, property prediction, and surrogate-model pipelines. Primary backends are ORCA for DFT and post-HF calculations and xTB for fast semiempirical pre-screening and conformer ranking. Invoke this skill when an agent needs to compute molecular geometries, energies, reaction barriers, thermochemical corrections, spectra, or molecular properties from first principles.

## Domain Context

Quantum chemistry describes molecular electronic structure by solving approximations to the electronic Schrödinger equation. The accuracy of any prediction depends on three choices that must be made consistently: the **electronic structure method** (how electron correlation is treated), the **basis set** (how atomic orbitals are represented), and the **molecular model** (charge, multiplicity, conformer, solvent environment). Getting any one wrong can produce qualitatively incorrect results regardless of compute time spent.

**The method–accuracy–cost ladder** (approximate, molecule-dependent):

| Tier | Representative methods | Formal cost scaling | Typical use |
|------|----------------------|--------------------|-|
| Force-field / classical | MMFF94, UFF, GAFF | O(N) | Conformer search, crude geometry seed |
| Semiempirical tight-binding | GFN2-xTB, GFN1-xTB, PM7 | O(N²–N³) | Large-scale conformer ranking, pre-screening |
| Pure DFT / HF | PBE, BLYP, HF | O(N³–N⁴) | Fast DFT for large systems; limited accuracy |
| Hybrid DFT | B3LYP, PBE0, M06-2X, ωB97X-D | O(N³–N⁴) | Mainstream molecular energetics and geometries |
| Double-hybrid DFT | B2PLYP, DLPNO-B2PLYP, ωB97M(2) | O(N⁵) | High-accuracy energetics; challenging cases |
| MP2 / SCS-MP2 | MP2, RI-MP2, SCS-MP2 | O(N⁵) | Post-HF baseline; poor for MR or open-shell |
| Coupled cluster | CCSD, CCSD(T), DLPNO-CCSD(T) | O(N⁶–N⁷) | Benchmark quality; DLPNO makes it feasible |
| Composite methods | G4, W1, CBS-QB3, ONIOM | Problem-dependent | Thermochemical accuracy (ΔH < 1 kcal/mol) |

Basis-set completeness is equally important as method choice. A hybrid DFT calculation with a minimal basis (STO-3G, 3-21G) can be worse than a well-parameterized semiempirical method. The Ahlrichs def2 series (def2-SVP → def2-TZVP → def2-QZVP) and Dunning cc-pV(D/T/Q)Z series provide systematic improvability. Diffuse functions (def2-TZVPD, aug-cc-pVTZ) are critical for anions, excited states, weak interactions, and polarizabilities. Basis-set superposition error (BSSE) inflates binding energies in non-covalent complexes by 0.5–5 kcal/mol with double-ζ or triple-ζ bases.

DFT dispersion corrections (D3BJ, D4) are not optional for conformational energies, non-covalent interactions, adsorption, or molecular crystals. [EXPERT REVIEW NEEDED: the best dispersion correction and functional combination is system-dependent; D3BJ and D4 perform differently across interaction types and functional families.]

Thermochemistry requires Gibbs free energies, not electronic energies. At 298.15 K and 1 atm, the Gibbs free energy correction (G − E_elec) is typically −10 to −30 kcal/mol from ZPE, thermal enthalpy, and entropy. Comparing electronic energies directly to experimental ΔG values introduces errors of 5–20 kcal/mol from entropy alone.

## When to Use This Skill

- Computing molecular geometries, bond lengths, conformer preferences, and tautomer stabilities from first principles.
- Calculating reaction energies (ΔE, ΔH, ΔG), activation barriers, and reaction free-energy profiles.
- Benchmarking DFT methods, basis sets, or solvation models against higher-level reference data or experiment.
- Computing vibrational frequencies, IR/Raman spectra, thermochemical functions (ZPE, Cp, S, G), or zero-point corrections.
- Locating transition states, verifying them via frequency analysis, and confirming reactant–product connectivity.
- Computing excitation energies, oscillator strengths, and singlet–triplet gaps via TDDFT for UV-Vis or photochemistry.
- Generating reference data (energies, forces, dipoles) for training molecular property surrogates, GNNs, or machine learning potentials.
- Screening molecular candidates by computed reactivity, selectivity, or thermodynamic descriptors before synthesis.

## When Not to Use This Skill

- **When a periodic crystal structure is required.** Molecular cluster models cannot replace periodic DFT (plane-wave codes) for bulk solids, surfaces, or extended materials where packing controls properties.
- **When the molecule is too large for the method.** A 500-atom system with CCSD(T) is not feasible; use xTB or QM/MM and document the approximation explicitly.
- **When multireference character is present and a single-reference method is chosen.** Transition-metal compounds, open-shell singlets, biradicals, and bond-breaking reactions often require CASSCF, NEVPT2, or MRCISD. Proceeding with DFT or MP2 without first diagnosing MR character produces unreliable energies.
- **When charge and spin multiplicity are ambiguous.** If the formal charge, protonation state, or spin state is not established, all downstream results inherit that ambiguity.
- **When only a qualitative trend is needed.** Molecular fingerprints, force-field descriptors, or semiempirical methods are often sufficient for large-scale screening; investing in full DFT for every candidate wastes compute and produces false precision if the model or target is noisy.

## Core Concepts

### Charge, Multiplicity, and Electronic State

Every quantum chemistry calculation requires an unambiguous specification of formal charge Q and spin multiplicity M = 2S+1. Common mistakes:

- Forgetting to protonate a basic nitrogen before gas-phase optimization (neutral form may not match the solution-phase species of interest)
- Using M = 1 (singlet) for a radical (M = 2) or biradical (M = 1 broken-symmetry, or M = 3 triplet)
- Failing to account for overall charge across all ligands in a metal complex

```python
from rdkit import Chem

mol = Chem.MolFromSmiles("[NH4+]")        # ammonium: charge = +1
formal_charge = Chem.GetFormalCharge(mol) # +1
radical_electrons = sum(
    a.GetNumRadicalElectrons() for a in mol.GetAtoms()
)
# multiplicity = 1 if radical_electrons == 0 else (radical_electrons + 1)
print(f"Charge: {formal_charge}, radical electrons: {radical_electrons}")
```

For transition-metal complexes and ambiguous spin states, run single-point calculations at multiple multiplicities at a cheap level (GFN2-xTB or DFT/def2-SVP) before committing to a full optimization.

### Geometry Optimization and Convergence

Geometry optimization minimizes the total energy with respect to nuclear coordinates. Convergence is assessed on **gradient and step size, not energy alone**. ORCA default convergence (`OPT`) uses gradient RMS < 1×10⁻⁴ Eh/Bohr; tighter thresholds (`TIGHTOPT`, `VERYTIGHTOPT`) are required for thermochemistry and frequency calculations.

A well-optimized geometry must satisfy:
1. All gradient components converged to the specified threshold
2. No imaginary frequencies — confirmed by a subsequent frequency calculation
3. Consistent connectivity with the target structure (check for bond formation, H-migration, ring opening)

### Frequency Analysis and Thermochemistry

A frequency calculation at the optimized geometry serves three purposes:

1. **Verification**: zero imaginary frequencies confirms a true minimum; exactly one imaginary frequency at a stationary point confirms a transition state
2. **Zero-point energy (ZPE)**: E_ZPE = ½Σ_i ħω_i; typically +1 to +5 kcal/mol for organic molecules
3. **Thermochemistry**: H(T) and G(T) at temperature T from the harmonic oscillator–rigid rotor–ideal gas partition function (RRHO)

Standard-state correction: gas-phase QC at 1 atm gives G°(gas). For solution-phase comparison to 1 M reference state, the Ben-Naim correction applies:

```
ΔG°(1 atm → 1 M) ≈ +1.89 kcal/mol at 298.15 K
```

[EXPERT REVIEW NEEDED: the direction and magnitude of the standard-state correction depends on the reference state convention; different programs and textbooks use incompatible conventions — verify before comparing to experiment.]

### Basis-Set Selection and BSSE

```python
def recommend_basis(property_type: str, has_anion: bool, needs_benchmark: bool) -> str:
    """Minimal decision tree for basis-set selection in organic/main-group chemistry."""
    if needs_benchmark:
        return "def2-QZVP or cc-pV(T→Q)Z CBS extrapolation"
    if has_anion or property_type in ("EA", "polarizability", "excited_state", "Rydberg"):
        return "def2-TZVPD or aug-cc-pVTZ"
    if property_type == "geometry_only":
        return "def2-SVP (geometry seed); upgrade to def2-TZVP for SP"
    if property_type in ("barrier", "reaction_energy", "thermochemistry"):
        return "def2-TZVP minimum; def2-QZVP for final benchmark"
    return "def2-TZVP"  # safe default for most organic chemistry
```

**BSSE counterpoise correction** for a non-covalent dimer A·B:

```
ΔE_CP = E(AB, basis_AB) − E(A, basis_AB) − E(B, basis_AB)
```

where E(A, basis_AB) uses the full dimer basis with ghost atoms at B's positions. Omitting CP overestimates binding by 0.5–5 kcal/mol with triple-ζ or smaller bases. [EXPERT REVIEW NEEDED: the magnitude of BSSE depends strongly on the basis set and the type of interaction; for large basis sets (def2-QZVP), BSSE is often negligible.]

### Solvation Models

| Model | Codes | Strengths | Limitations |
|-------|-------|-----------|-------------|
| COSMO | ORCA, Turbomole | Fast; simple dielectric | Crude cavity; less accurate for ions |
| SMD | Gaussian, ORCA | Parameterized for ΔG_solv; good for neutral organics | Sensitive to charge; may need explicit first shell |
| CPCM | ORCA, Gaussian | Fast; suitable for geometry | Less accurate than SMD for thermochemistry |
| ALPB (GFN-xTB) | xTB | Very fast semiempirical solvation | Only at the GFN level; not for DFT |

For ions, zwitterions, and molecules that form strong hydrogen bonds with solvent, implicit models can fail by several kcal/mol. Augmenting with 1–4 explicit first-shell solvent molecules inside the QM region, followed by re-optimization, often improves accuracy. [EXPERT REVIEW NEEDED: the number of explicit solvent molecules required is system- and property-dependent; there is no universal protocol.]

### Open-Shell Systems, Spin Contamination, and Multireference Diagnostics

For open-shell singlets and diradicals, use **broken-symmetry DFT (BS-DFT)**:

1. Converge a high-spin (e.g., triplet) solution.
2. Mix the HOMO and LUMO to seed the broken-symmetry singlet guess.
3. Optimize; check ⟨S²⟩: expected 0.0 for pure singlet, 2.0 for pure triplet. A BS-singlet ⟨S²⟩ ≈ 1.0 indicates ~50/50 mixing.
4. Apply the Yamaguchi correction if needed:

```
E(singlet) ≈ E_BS − [⟨S²⟩_BS / (⟨S²⟩_HS − ⟨S²⟩_BS)] × (E_HS − E_BS)
```

[EXPERT REVIEW NEEDED: the Yamaguchi correction is an approximation that depends on the degree of spin contamination and nature of the diradical; it is not universally reliable.]

**Multireference diagnostics before using single-reference methods:**

- T1 diagnostic from CCSD: T1 > 0.02 for organics or > 0.05 for transition-metal compounds suggests significant MR character [EXPERT REVIEW NEEDED: thresholds are approximate and system-dependent]
- ⟨S²⟩ deviation from the expected value in DFT or HF
- HOMO–LUMO gap: gaps < 0.5 eV often indicate near-degeneracy

### TDDFT and Excited-State Workflows

TDDFT computes vertical excitation energies within the linear-response formalism. Practical guidance:

- Use range-separated hybrids (CAM-B3LYP, ωB97X-D, LC-ωPBE) for charge-transfer excitations, Rydberg states, and long-range transitions; global hybrids severely underestimate CT excitation energies.
- Use a triple-ζ basis with diffuse functions (def2-TZVPD) for excited states above ~6 eV or Rydberg character.
- TDDFT typically underestimates singlet–triplet gaps by 0.1–0.5 eV; oscillator strengths for dark states can be unreliable. [EXPERT REVIEW NEEDED: TDDFT fails for states with strong static correlation, double excitations, and near-conical intersections; use EOM-CCSD or MRCISD for benchmarks of challenging cases.]
- For excited-state geometry optimization, verify state character and absence of state crossings at the Franck–Condon geometry before running `OPT` in the excited state.

## Key Workflows

### Workflow 1: Structure Preparation and Conformer Pre-Screening

Before any DFT calculation, generate and rank conformers with GFN2-xTB to avoid optimizing a high-energy conformer as the representative structure.

```python
from rdkit import Chem
from rdkit.Chem import AllChem
import subprocess
from pathlib import Path

def prepare_conformers(smiles: str, n_conf: int = 200,
                       energy_window_kcal: float = 5.0) -> tuple[Chem.Mol, list[int]]:
    """Generate ETKDG conformers, rank by MMFF, filter by energy window."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    mol = Chem.AddHs(mol)

    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    params.numThreads = 0
    AllChem.EmbedMultipleConfs(mol, numConfs=n_conf, params=params)

    ff_result = AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=500)
    energies = [r[1] for r in ff_result]   # kcal/mol
    min_e = min(energies)

    kept = [(i, e) for i, e in enumerate(energies) if e - min_e <= energy_window_kcal]
    kept.sort(key=lambda x: x[1])
    print(f"Kept {len(kept)}/{n_conf} conformers within {energy_window_kcal} kcal/mol")
    return mol, [i for i, _ in kept]

# Example: flexible drug-like molecule (naproxen)
smiles = "CC(C1=CC2=CC=CC=C2C=C1)C(=O)O"
mol, conf_ids = prepare_conformers(smiles, n_conf=200, energy_window_kcal=5.0)

# Write lowest conformer to XYZ, then refine with xTB
# xtb command:
#   xtb conformer_0.xyz --opt normal --gfn 2 --alpb methanol -P 4 > xtb_opt.out
# After xTB, select the lowest-energy xTB conformer as DFT geometry seed.
```

For molecules with more than ~8 rotatable bonds, use CREST (Conformer-Rotamer Ensemble Sampling Tool) for exhaustive conformer sampling. CREST wraps GFN2-xTB with iMTD-GC metadynamics to explore rugged conformational landscapes efficiently. [EXPERT REVIEW NEEDED: CREST ensemble completeness depends on wall time, temperature, and force-constant settings; validate ensemble size for flexible macrocycles or peptides.]

### Workflow 2: DFT Geometry Optimization and Frequency Verification with ORCA

```
# ORCA input: geometry optimization + frequency for an organic molecule
# File: opt_freq.inp

! r2SCAN-3c OPT FREQ TIGHTOPT

%maxcore 4000
%pal nprocs 8 end

%freq
  Temp 298.15
  Pressure 1.0
end

* xyzfile 0 1 conformer_best.xyz
```

Post-processing: parse the output and verify convergence.

```python
import re
from pathlib import Path

def parse_orca_thermo(outfile: str) -> dict:
    """Extract electronic energy, ZPE, H, G, and imaginary-frequency count."""
    text = Path(outfile).read_text()
    results: dict = {}

    for line in text.splitlines():
        if "FINAL SINGLE POINT ENERGY" in line:
            results["E_elec"] = float(line.split()[-1])   # Hartree

    for pattern, key in [
        (r"Zero point energy\s+\.\.\.\s+([\-\d\.]+)", "ZPE"),
        (r"Total enthalpy\s+\.\.\.\s+([\-\d\.]+)", "H"),
        (r"Final Gibbs free energy\s+\.\.\.\s+([\-\d\.]+)", "G"),
    ]:
        m = re.search(pattern, text)
        if m:
            results[key] = float(m.group(1))   # Hartree

    imag = [float(x) for x in re.findall(r"(-\d+\.\d+)\s+cm\*\*-1", text)]
    results["n_imaginary"] = len(imag)
    results["imaginary_freqs_cm"] = imag
    return results

thermo = parse_orca_thermo("opt_freq.out")
if thermo["n_imaginary"] > 0:
    print(f"WARNING: {thermo['n_imaginary']} imaginary frequencies "
          f"{thermo['imaginary_freqs_cm']} cm⁻¹ — NOT a true minimum.")
    print("Re-optimize: displace along imaginary mode, or check for symmetry constraint.")
else:
    hartree_to_kcal = 627.5095
    zpe_kcal = thermo.get("ZPE", 0.0) * hartree_to_kcal
    print(f"Minimum confirmed.  G = {thermo['G']:.6f} Eh")
    print(f"ZPE correction:     {zpe_kcal:.2f} kcal/mol")
```

**r2SCAN-3c** (meta-GGA + D4 dispersion + geometric counterpoise) is a strong default for geometry and thermochemistry at low cost. For accurate barriers and reaction energetics of challenging systems, upgrade to ωB97X-D/def2-TZVP or run a DLPNO-CCSD(T)/def2-TZVP single-point on the DFT geometry. [EXPERT REVIEW NEEDED: "best default functional" is an evolving consensus; validate against relevant benchmark sets for the target chemistry before committing to a production workflow.]

### Workflow 3: Transition-State Search and Reaction Profile

Recommended sequence for computing an activation barrier from reactant to product.

```
# Step 1: relaxed PES scan along the forming/breaking bond
# File: pes_scan.inp  (ORCA)

! r2SCAN-3c OPT

%maxcore 4000
%pal nprocs 8 end

%geom
  Scan
    B 0 5 = 1.40, 3.00, 17    # scan atom-0–atom-5 bond from 1.4 to 3.0 Å in 17 steps
  end
end

* xyzfile 0 1 reactant.xyz
```

```
# Step 2: refine TS from PES maximum
# File: ts_opt.inp  (ORCA)

! ωB97X-D def2-TZVP OPTTS FREQ TIGHTOPT

%maxcore 4000
%pal nprocs 8 end

%geom
  Calc_Hess true    # exact Hessian at start — important for OPTTS reliability
  MaxIter 150
end

* xyzfile 0 1 ts_guess.xyz
```

```python
# Step 3: compute barriers and validate TS

HARTREE_TO_KCAL = 627.5095

def reaction_profile(E_dict: dict, G_dict: dict) -> None:
    """Print forward barrier and reaction energy from parsed ORCA outputs."""
    delta_E_fwd  = (E_dict["ts"]      - E_dict["reactant"]) * HARTREE_TO_KCAL
    delta_G_fwd  = (G_dict["ts"]      - G_dict["reactant"]) * HARTREE_TO_KCAL
    delta_E_rxn  = (E_dict["product"] - E_dict["reactant"]) * HARTREE_TO_KCAL
    delta_G_rxn  = (G_dict["product"] - G_dict["reactant"]) * HARTREE_TO_KCAL
    print(f"ΔE‡ = {delta_E_fwd:.1f}  kcal/mol  (electronic)")
    print(f"ΔG‡ = {delta_G_fwd:.1f}  kcal/mol  (free energy, 298 K)")
    print(f"ΔE  = {delta_E_rxn:.1f}  kcal/mol")
    print(f"ΔG  = {delta_G_rxn:.1f}  kcal/mol")

# After TS confirmed (one imaginary freq with correct displacement vector):
# Run IRC in both directions to verify reactant–product connectivity:
#   Add to ORCA TS input:  ! IRC
# Inspect IRC path to confirm the TS connects the intended reactant and product.
```

**Entropy and free-energy corrections matter most for:** bimolecular steps (large negative ΔS‡ ≈ −30 to −50 cal/mol/K), ring-forming reactions, and reactions involving flexible substrates. For such cases, ΔE‡ and ΔG‡ can differ by 5–15 kcal/mol at 298 K.

### Workflow 4: Method Benchmarking Before Screening

Before screening a molecular series, validate the method on a representative subset with known experimental or high-level reference data.

```python
import numpy as np
import pandas as pd

# Reference activation barriers (kcal/mol) from DLPNO-CCSD(T)/CBS calculations
# for 10 hydrogen-atom transfer reactions (subset of BH76)
ref_barriers = [4.2, 8.7, 12.3, 15.1, 2.9, 18.4, 6.6, 11.0, 9.3, 20.5]

# DFT results at three levels on the same geometries
dft_results = {
    "B3LYP-D3BJ/def2-SVP":         [4.9, 9.5, 13.1, 14.2, 2.2, 17.8, 7.0, 11.9, 9.8, 20.1],
    "M06-2X/def2-TZVP":             [4.4, 9.0, 12.7, 15.3, 2.8, 18.2, 6.8, 11.2, 9.5, 20.6],
    "ωB97X-D/def2-TZVP":            [4.3, 8.9, 12.5, 15.0, 2.9, 18.3, 6.7, 11.1, 9.4, 20.4],
}

ref = np.array(ref_barriers)
rows = []
for method, values in dft_results.items():
    err = np.array(values) - ref
    rows.append({
        "method": method,
        "MAE":  round(float(np.mean(np.abs(err))), 2),
        "RMSE": round(float(np.sqrt(np.mean(err**2))), 2),
        "MaxE": round(float(np.max(np.abs(err))), 2),
    })

df = pd.DataFrame(rows).set_index("method")
print(df)

# Decision: choose the cheapest method with MAE < 0.5 kcal/mol for production
# Key reporting requirements:
#   - Which benchmark set was used and at what reference level
#   - Whether benchmark molecules are chemically representative of the screening set
#   - Separate statistics for barriers and reaction energies
```

### Workflow 5: Simulated UV-Vis Spectrum from TDDFT

```
# ORCA TDDFT for a push-pull organic dye
# File: tddft_dye.inp

! CAM-B3LYP def2-TZVPD TDDFT

%maxcore 4000
%pal nprocs 8 end

%tddft
  NRoots 15
  MaxDim 5000
  ETol 1e-6
end

* xyzfile 0 1 dye_gs_opt.xyz
```

```python
import re, numpy as np, matplotlib.pyplot as plt

def parse_tddft_transitions(outfile: str) -> list[dict]:
    """Extract excitation energies (eV) and oscillator strengths from ORCA output."""
    text = open(outfile).read()
    pattern = r"STATE\s+\d+:\s+E=\s+[\d\.]+\s+au\s+([\d\.]+)\s+eV.*?\n.*?f=\s*([\d\.]+)"
    return [{"E_eV": float(m.group(1)),
             "wavelength_nm": 1240.0 / float(m.group(1)),
             "f": float(m.group(2))}
            for m in re.finditer(pattern, text, re.DOTALL)]

def broadened_spectrum(transitions: list[dict], sigma_eV: float = 0.15) -> tuple:
    e_grid = np.linspace(1.5, 6.5, 1000)
    spectrum = sum(
        t["f"] * np.exp(-0.5 * ((e_grid - t["E_eV"]) / sigma_eV)**2)
        for t in transitions
    )
    return e_grid, spectrum

transitions = parse_tddft_transitions("tddft_dye.out")
e_grid, spectrum = broadened_spectrum(transitions)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=False)
ax1.plot(1240.0 / e_grid, spectrum, color="navy")
ax1.set_xlabel("Wavelength (nm)")
ax1.set_ylabel("Intensity (arb.)")
ax1.set_title(f"Simulated UV-Vis (CAM-B3LYP/def2-TZVPD, Gaussian broadening σ=0.15 eV)")

wl = [t["wavelength_nm"] for t in transitions]
f  = [t["f"]             for t in transitions]
ax2.stem(wl, f, linefmt="b-", markerfmt="bo", basefmt="k-")
ax2.set_xlabel("Wavelength (nm)")
ax2.set_ylabel("Oscillator strength f")
fig.tight_layout()
fig.savefig("uv_vis_simulated.png", dpi=300)

strongest = max(transitions, key=lambda t: t["f"])
print(f"Strongest transition: λ = {strongest['wavelength_nm']:.1f} nm, "
      f"E = {strongest['E_eV']:.2f} eV, f = {strongest['f']:.4f}")
```

## Common Failure Modes

| Failure Mode | Symptom | Cause | Diagnostic | Fix |
|---|---|---|---|---|
| Wrong charge or multiplicity | Wildly wrong energies; SCF fails to converge; bond orders inconsistent with chemistry | Charge not tracking protonation state; multiplicity guessed rather than determined from radical electron count | Check `Chem.GetFormalCharge(mol)`; verify with GFN2-xTB at multiple multiplicities | Enumerate charge/multiplicity combinations at xTB level first; select lowest energy; document choice explicitly |
| Missing conformer search | Computed property inconsistent with experiment; wrong lowest-energy conformer used | Single conformer generated by 2D→3D builder; global minimum missed | Compare ETKDG and xTB-optimized geometry RMSD; run CREST for flexible molecules | Run ETKDG + MMFF pre-filter + GFN2-xTB ranking before any DFT optimization |
| Imaginary frequencies in minimum | Frequency output shows one or more negative wavenumbers | Optimization stalled at saddle point; too-loose convergence; flat potential surface | Inspect displacement vector of imaginary mode; check gradient convergence in output | Re-optimize with `TIGHTOPT`; displace geometry along imaginary mode and re-optimize |
| False transition state | TS has exactly one imaginary frequency but IRC goes to wrong products | TS guess geometry wrong; optimization converged on an unintended stationary point | Run IRC in both directions; visualize normal-mode displacement | Restart TS search from a better guess built from the PES scan; confirm normal-mode vector connects intended reactant–product |
| DFT functional overclaiming | Barriers differ by 3–10 kcal/mol across functionals; kcal/mol accuracy claimed without benchmarking | B3LYP used for radicals, CT states, or metallocatalysis without validation; dispersion omitted | Run benchmark on representative reactions vs DLPNO-CCSD(T) or experiment | Use dispersion-corrected functional (ωB97X-D, r2SCAN-3c); benchmark first; report MAE and RMSE |
| Basis-set incompleteness | Energies not converged; BSSE inflates non-covalent binding | Double-ζ used for production energetics; triple-ζ skipped to save cost | SP at def2-SVP → def2-TZVP → def2-QZVP; check energy convergence < 0.1 kcal/mol | Use def2-TZVP minimum for single-points; add diffuse functions for anions and excited states |
| Spin contamination | ⟨S²⟩ significantly above expected (e.g., 0.85–1.2 for a doublet vs expected 0.75) | Broken-symmetry ground state; near-degenerate orbitals; HF exchange overestimates spin polarization | Print ⟨S²⟩ from ORCA output; compare UHF and UKS values | Switch to range-separated hybrid or lower HF-exchange functional for the radical; use ROHF/RODFT if applicable |
| Unrecognized multireference character | Large T1 diagnostic; ⟨S²⟩ deviations; CCSD(T) and MP2 disagree by > 2 kcal/mol | TM compound, singlet diradical, bond-breaking, or strained system treated with SR methods | T1 > 0.02 (organics) or > 0.05 (TM); compare DLPNO-CCSD(T) and DFT results | Use CASSCF/NEVPT2 or DLPNO-NEVPT2; flag results as uncertain; report T1 diagnostic |
| Solvent-model mismatch | Computed pKa or solvation ΔG off by 2–10 kcal/mol | Wrong dielectric constant; implicit model used for ions or strongly H-bonding species | Compare with and without explicit first-shell; check ε parameterization in the code | Use SMD with correct solvent parameters; add explicit molecules for ions; apply standard-state correction |
| Comparing electronic energies to experimental free energies | Barriers or ΔG values systematically off by 5–20 kcal/mol even with a good method | ΔE_elec compared directly to ΔG_exp; ZPE, thermal, entropy corrections omitted | Check whether RRHO thermochemical corrections were applied; inspect ΔE vs ΔG separately | Always report and compare Gibbs free energies for solution-phase experiments; apply standard-state correction |

## Best Practices

- **Establish charge and spin before optimization.** A single-point at GFN2-xTB level across plausible spin states costs seconds and prevents hours of wasted DFT time on the wrong electronic state.
- **Use a two-step protocol for molecular series:** xTB geometry + conformer ranking → DFT single-point or short optimization. Reserve full DFT optimization + frequency for final candidates only.
- **Never report an optimized structure without a subsequent frequency calculation.** A converged geometry optimizer can stall at a flat region that is not a true minimum; frequency analysis is the only rigorous check.
- **Benchmark before screening.** Run the target method on 5–20 molecules with known reference data. Report MAE and RMSE; a method with RMSE of 5 kcal/mol is not useful for reaction selectivity prediction, regardless of cost.
- **Apply dispersion corrections universally for non-covalent and condensed-phase properties.** D3BJ or D4 adds negligible cost; omitting it for conformational energies, host–guest binding, or adsorption is not justified.
- **Document the full computational protocol in every output file and in the repository metadata.** Method, basis set, solvation model, software version, keywords, convergence thresholds, and date. Small changes in defaults between software versions can shift energies by 0.1–0.5 kcal/mol. [EXPERT REVIEW NEEDED: version-to-version reproducibility is not guaranteed even within ORCA 5.x.]
- **For reference dataset generation:** use a consistent geometry optimization level, single-point level, thermochemical corrections, and solvation treatment across all molecules. Mixed-level datasets produce inconsistent training signals for downstream ML models.
- **Report uncertainty alongside computed properties.** For barriers in well-behaved organic reactions, ±1–2 kcal/mol from method uncertainty is a realistic minimal estimate; larger uncertainties apply for radical, metal-catalyzed, or multireference systems.

## Integration with Other Skills

- **[molecular-featurization](../../cheminformatics/molecular-featurization/SKILL.md):** Computed properties (partial charges, dipole moments, conformer geometries, bond-order matrices) augment fingerprints and physicochemical descriptors for molecular ML pipelines.
- **[molecular-datasets](../../cheminformatics/molecular-datasets/SKILL.md):** QC-computed labels (HOMO–LUMO gaps, solvation energies, reaction energies) populate or augment molecular datasets; the same curation and provenance standards apply.
- **[molecular-property-gnn](../../geometric-deep-learning/molecular-property-gnn/SKILL.md):** High-accuracy QC labels (barrier heights, excitation energies) provide better training targets for molecular GNNs than experimental proxy labels, especially for rare reaction classes.
- **[small-data-ml](../../scientific-ml/small-data-ml/SKILL.md):** Small QC datasets (20–200 molecules with CCSD(T) barriers or solvation energies) are prime candidates for GP regression, kernel ridge regression, or delta-learning models.
- **[surrogate-validation](../../surrogate-active-learning/surrogate-validation/SKILL.md):** Validates surrogates trained on QC data under scaffold or chemotype distribution shift; especially important when training and deployment sets come from different reaction classes.
- **[gaussian-process-materials](../../surrogate-active-learning/gaussian-process-materials/SKILL.md):** GP models built on molecular descriptors + QC labels provide uncertainty-quantified predictions for active-learning-driven QC screening campaigns.
- **[bayesian-optimization](../../surrogate-active-learning/bayesian-optimization/SKILL.md):** Bayesian optimization over molecular structure space guided by QC-computed objectives (selectivity, barrier height, Stokes shift) supports computational reaction discovery or photocatalyst design.
- **[scientific-visualization](../../scientific-data/scientific-visualization/SKILL.md):** Reaction coordinate diagrams, NEB or IRC paths, simulated UV-Vis spectra, PES scans, and molecular orbital visualizations require structured output parsing and matplotlib pipelines.
- **[reproducibility-workflow](../../scientific-data/reproducibility-workflow/SKILL.md):** Version-controlled input files, output parsing scripts, environment lock files (ORCA version, xTB version, conda env), and raw output files are essential for reproducible QC workflows.
- **[materials-lca](../../sustainability-lca/materials-lca/SKILL.md):** Computed reaction energetics, atom economies, and synthetic route feasibility feed into green chemistry assessments and life-cycle analysis for molecular process design.

## Key References

1. Jensen, F. (2017). *Introduction to Computational Chemistry* (3rd ed.). Wiley. — Standard textbook covering HF, DFT, basis sets, geometry optimization, thermochemistry, and transition-state theory.

2. Cramer, C. J. (2004). *Essentials of Computational Chemistry: Theories and Models* (2nd ed.). Wiley. — Covers semiempirical, DFT, post-HF, and reaction-rate methods with worked examples.

3. Grimme, S., Hansen, A., Brandenburg, J. G. & Bannwarth, C. (2016). Dispersion-corrected mean-field electronic structure methods. *Chemical Reviews*, 116(9), 5105–5154. https://doi.org/10.1021/acs.chemrev.5b00533 — Comprehensive review of D3, D4, and NL dispersion corrections and their domain of applicability.

4. Bannwarth, C., Caldeweyher, E., Ehlert, S., Hansen, A., Pracht, P., Seibert, J., Spicher, S. & Grimme, S. (2021). Extended tight-binding quantum chemistry methods. *WIREs Computational Molecular Science*, 11(2), e1493. https://doi.org/10.1002/wcms.1493 — xTB/GFN methods reference for semiempirical pre-screening and conformer sampling.

5. Neese, F. (2022). Software update: The ORCA program system — Version 5.0. *WIREs Computational Molecular Science*, 12(5), e1606. https://doi.org/10.1002/wcms.1606 — ORCA 5 reference; cite when reporting ORCA calculations.

6. Goerigk, L., Hansen, A., Bauer, C., Ehrlich, S., Najibi, A. & Grimme, S. (2017). A look at the density functional theory zoo with the advanced GMTKN55 database for general main group thermochemistry, kinetics and noncovalent interactions. *Physical Chemistry Chemical Physics*, 19(48), 32184–32215. https://doi.org/10.1039/C7CP04913G — GMTKN55 benchmark: the standard reference for validating DFT methods across thermochemistry, kinetics, and non-covalent interactions.
