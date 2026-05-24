---
name: battery-materials-workflow
domain: electrochemistry-corrosion
maturity: draft
tools:
  - VASP 6.4.2
  - pymatgen 2024.11.13
  - ASE 3.23.0
  - mp-api 0.41.2
  - numpy 1.26.4
  - matplotlib 3.8.0
  - pandas 2.2.0
tested_versions:
  VASP: "6.4.2"
  pymatgen: "2024.11.13"
  ASE: "3.23.0"
  mp-api: "0.41.2"
  numpy: "1.26.4"
  matplotlib: "3.8.0"
  pandas: "2.2.0"
requires_gpu: false
estimated_runtime: hours-to-weeks
---

# Battery Materials Workflow

## Description

This skill covers computational and data-driven workflows for battery materials discovery and characterization: voltage profiles, redox couples, intercalation thermodynamics, phase stability, ion migration barriers, defect chemistry, ionic conductivity, electrochemical stability windows, surface and interface reactions, SEI/CEI formation concepts, high-throughput screening, surrogate modeling, and coupling to experiments. Primary backends are VASP (DFT) and pymatgen for structure manipulation, phase diagram construction, convex hull analysis, and migration pathway automation. Invoke this skill when an agent needs to reason computationally about cathodes, anodes, solid electrolytes, liquid electrolytes, interfaces, coatings, or degradation mechanisms in rechargeable battery systems.

## Domain Context

A rechargeable battery converts chemical energy to electrical energy reversibly through coupled electrochemical reactions at two electrodes separated by an electrolyte. Performance is governed by thermodynamic, kinetic, mechanical, and interfacial properties across length scales from Ångstroms (defect cores, interfacial reactions) to micrometers (electrode particles, separator, pores) to centimeters (cell architecture, thermal gradients).

**Voltage** is set by the difference in lithium (or sodium, magnesium, zinc, …) chemical potential between cathode and anode:

```
V = -(μ_Li(cathode) - μ_Li(anode)) / F
```

In DFT, the average intercalation voltage over a composition range x₁ → x₂ in Li_x·HostO₂ is:

```
V_avg = -[E(Li_{x2}HostO₂) - E(Li_{x1}HostO₂) - (x2-x1)·E(Li_bcc)] / ((x2-x1)·F)
```

where E(Li_bcc) is the DFT total energy per atom of body-centered-cubic lithium metal. The sign convention requires care: a more negative total energy of the lithiated cathode raises the voltage. Wrong assignment of the lithium reference state is the most common numerical error in voltage calculations.

**Energy density** is limited by voltage × capacity (mAh g⁻¹); practical limits are dominated by cycle stability, safety, rate capability, and cost. Computational workflows address thermodynamics (what phases are stable at what compositions and temperatures?), kinetics (how fast do ions diffuse?), and interfacial chemistry (how does the electrolyte decompose and what passivating layer forms?).

Key material classes and representative systems:

| Class | Examples | Key computational targets |
|-------|----------|--------------------------|
| Layered oxide cathodes | LCO (LiCoO₂), NMC, NCA, Li-rich | Voltage, delithiation phase stability, O₂ release, Mn/Ni/Co redox, structural distortions |
| Olivine cathodes | LFP (LiFePO₄), NaFePO₄ | Fe²⁺/Fe³⁺ voltage, two-phase reaction, Li diffusion along [010] channel |
| Spinel cathodes | LMO (LiMn₂O₄), LNMO | Jahn-Teller distortion at Mn³⁺, voltage steps, Li ordering |
| Conversion anodes | SnO₂, Fe₃O₄, CuO | Multi-step reduction, large volume change, reversibility |
| Graphite anodes | Stage I–IV intercalation | Li staging, formation energy, diffusion, SEI chemistry |
| Li/Na metal anodes | Dendrite nucleation, stripping | Surface energy, Li adsorption, SEI stability |
| Solid electrolytes | LLZO, LGPS, NASICON, LIPON | Li-ion conductivity, activation energy, electrochemical window, grain boundary |
| Liquid electrolytes | LiPF₆-EC/DMC, NaPF₆-EC/DMC | Oxidation/reduction potential, solvation, SEI precursors |
| Interfaces/coatings | LCO|LGPS, graphite|EC, Al₂O₃ coating | Reaction thermodynamics, stability window, passivation |

Computational battery research depends critically on correct chemical potential references, magnetic ground states, Hubbard U corrections for transition-metal oxides, and careful treatment of charged supercells for defect and ion-migration calculations. Missing any of these can shift predicted voltages by hundreds of millivolts or migration barriers by 0.2–0.5 eV.

## When to Use This Skill

- Use when computing intercalation voltages, voltage profiles, or open-circuit voltage curves from DFT total energies.
- Use when assessing phase stability of electrode or electrolyte materials via convex hull analysis.
- Use when calculating ion migration barriers (Li, Na, Mg, …) in cathodes, anodes, or solid electrolytes.
- Use when modeling defect chemistry (vacancies, interstitials, antisites) in solid electrolytes or electrode materials for ionic conductivity.
- Use when computing electrochemical stability windows for electrolytes or coating materials.
- Use when screening materials databases for candidate cathodes, anodes, or solid electrolytes by computational descriptors.
- Use when coupling battery thermodynamics to CALPHAD, phase-field, FEM, or corrosion/electrochemistry workflows.
- Use when calibrating or validating MLP potentials for battery materials MD simulations.
- Use when connecting battery performance to sustainability, life-cycle analysis, or materials selection.

## When Not to Use This Skill

- **When the lithium/sodium/magnesium chemical potential reference is undefined.** Every voltage and phase stability calculation depends on a well-defined reference state; do not proceed without establishing it.
- **When spin states and magnetic ordering are unknown for transition-metal compounds.** GGA+U calculations for Mn, Fe, Co, Ni oxides require correct magnetic ordering; wrong spin states shift energies by 0.5–2 eV/f.u.
- **When Hubbard U values have not been validated for the target system.** Transferring U values across oxidation states, coordination environments, or chemistries without validation is a known source of large voltage errors.
- **When the target property is a transport coefficient at experimental temperature** (e.g., μS/cm ionic conductivity at 25°C): dilute-defect 0 K migration barriers do not directly predict room-temperature conductivity without accounting for correlation effects, defect concentrations, and grain boundaries.
- **When interface chemistry is the critical unknown but only bulk calculations are done.** Electrochemical stability windows computed from bulk phases miss interface reconstruction, space-charge layers, strain, and mixed ionic-electronic conduction.
- **When atomistic detail below the nanometer scale is the goal** (e.g., SEI monolayer reaction mechanisms under potential): use explicit electrochemical DFT with solvation or constant-charge methods instead.
- **When the material has not been structurally characterized.** Using unvalidated or database-retrieved crystal structures without provenance checks leads to incorrect ground-state energies and phase stability predictions.

## Core Concepts

### Intercalation Voltage and Redox Couples

The theoretical intercalation voltage between compositions x₁ and x₂ (x₂ > x₁) in Li_x·HostO₂ is computed from DFT total energies:

```python
import numpy as np

# VASP total energies (eV per formula unit)
E_host_x2 = -45.321   # Li_{x2}HostO2
E_host_x1 = -38.744   # Li_{x1}HostO2
E_Li_metal = -1.896   # BCC Li metal per atom

x1, x2 = 0.0, 1.0     # Li stoichiometry range
dx = x2 - x1

# Average voltage (V vs Li/Li+)
V_avg = -(E_host_x2 - E_host_x1 - dx * E_Li_metal) / (dx * 1.0)
# 1.0 is the electron count per Li (for monovalent Li)
print(f"Average voltage: {V_avg:.3f} V vs Li/Li+")
```

For a step-like voltage profile from multiple intermediate compositions, build the convex hull and compute pairwise voltages between stable endpoints on the hull. [EXPERT REVIEW NEEDED: GGA/PBE+U systematic errors for transition-metal oxides can shift computed voltages vs experiment by 0.2–0.5 V; mixing energetics (GGA+U for oxide phases vs GGA for metallic Li) requires a correction scheme.]

### Convex Hull and Phase Stability

The convex hull of formation energies ΔH_f(x) at 0 K identifies thermodynamically stable compositions. Phases above the hull are metastable by the distance to the hull (e_hull, meV/atom). Materials Project uses e_hull < 25 meV/atom as a stability threshold for synthesizability under certain conditions.

```python
from pymatgen.analysis.phase_diagram import PhaseDiagram, PDPlotter
from pymatgen.core.composition import Composition
from pymatgen.entries.computed_entries import ComputedEntry

# Build entries: each entry has a composition and DFT energy
entries = [
    ComputedEntry("Li", -1.896),          # BCC Li metal
    ComputedEntry("CoO2", -10.542),        # delithiated end
    ComputedEntry("LiCoO2", -14.231),      # fully lithiated
    ComputedEntry("Li2O", -14.278),        # competing phase
    ComputedEntry("Co3O4", -34.182),       # competing phase
]

pd = PhaseDiagram(entries)
for e in entries:
    e_hull = pd.get_e_above_hull(e)
    print(f"{e.composition.reduced_formula}: {e_hull*1000:.1f} meV/atom above hull")
```

For cathode screening, construct the pseudo-binary Li₀HostO₂ — Li₁HostO₂ hull, identify stable intermediate phases, and compute step voltages. Non-convex hull points correspond to two-phase coexistence regimes visible in the experimental voltage profile as flat plateaus.

### Chemical Potential of Li (and Na, Mg)

The chemical potential of the mobile ion must be consistent across all phases in a calculation. Standard choices:

- **Li/Li⁺ (metallic Li reference):** μ_Li = E_DFT(BCC Li per atom). Gives V vs Li/Li⁺.
- **Na/Na⁺:** μ_Na = E_DFT(BCC Na per atom). Check structure (BCC vs 9R-Na) at the DFT level.
- **Mg/Mg²⁺:** μ_Mg = E_DFT(HCP Mg per atom). Note: Mg²⁺ carries 2 electrons, so voltage is halved per formula unit relative to Li.

Wrong reference (e.g., using μ_Li = 0 eV, a molecule, or an incorrect polymorph) is the most frequent source of voltage errors.

### Electrochemical Stability Window

The electrochemical stability window of an electrolyte or coating material is the range of potentials within which it neither oxidizes nor reduces. Computed from DFT as:

```
Reduction potential (V vs Li/Li+) = -(E_reduced - E_neutral - Σ μ_i) / (n_e · F)
Oxidation potential (V vs Li/Li+) = -(E_oxidized - E_neutral - Σ μ_i) / (n_e · F)
```

In practice, use the grand potential phase diagram (vary μ_Li while fixing all other chemical potentials) to find which competing phases become stable at each potential. This is implemented in pymatgen's `GrandPotentialPhaseDiagram`. [EXPERT REVIEW NEEDED: computed stability windows are thermodynamic limits; kinetic passivation can extend practical windows significantly beyond thermodynamic predictions, e.g., LLZO is thermodynamically unstable against Li metal but passivated in practice.]

### Ion Migration Barriers

Ion migration barriers in solid electrolytes and electrode materials are computed using the Nudged Elastic Band (NEB) method, which finds the minimum-energy path (MEP) between two stable configurations. Key parameters:

- **Supercell size:** ≥ 3×3×3 primitive cell repetitions to avoid image-charge interactions between the migrating ion and its periodic images; validate with supercell scaling.
- **Images:** 5–9 intermediate NEB images; use climbing-image NEB (CI-NEB) for accurate barrier tops.
- **Charge state:** Migrating ion in the correct charge state; for Li⁺ migration use +1 charge (remove one background electron), or keep neutral cell if the host is metallic.
- **Relaxation:** Fully relax the endpoint structures before constructing the NEB path.

Typical activation energy (Ea) targets for practical room-temperature ionic conductors: < 0.3 eV for superionic materials (LGPS, LLZO), 0.4–0.7 eV for moderate conductors, > 0.8 eV implies poor room-temperature transport.

The Arrhenius conductivity model:

```
σ(T) = (σ₀ / T) · exp(-Ea / kT)
```

is often used to connect computed Ea to experimental conductivity data. [EXPERT REVIEW NEEDED: single-particle NEB Ea underestimates or overestimates effective conductivity when correlation effects, defect concentrations, multiple pathways, or grain boundaries dominate; MD-derived conductivity from MLMD is often more representative at finite temperature.]

### Defect Chemistry and Ionic Conductivity

Ionic conductivity depends on both migration barriers (kinetics) and defect concentration (thermodynamics). For a Schottky defect pair in a rock-salt-type oxide:

```
n_vacancy ∝ exp(-ΔH_Schottky / (2kT))
σ ∝ n_vacancy · μ_ion ∝ exp(-(ΔH_Schottky/2 + Ea_hop) / kT)
```

DFT defect calculations (vacancy formation energy, interstitial formation energy, Frenkel energy) feed this model. Chemical potential conditions (Li-rich vs Li-poor, oxygen partial pressure) must be specified consistently with synthesis conditions.

### SEI and CEI Formation

The solid electrolyte interphase (SEI) on the anode and cathode electrolyte interphase (CEI) on the cathode are metastable passivation layers formed by electrolyte decomposition at the first cycle. Workflow-level modeling:

1. **Stability window check:** Compare the electrode operating potential against the electrolyte oxidation/reduction potential computed from the grand potential phase diagram.
2. **Thermodynamic reaction products:** Use the interface stability analysis (e.g., pymatgen `InterfaceReactivity`) to predict which phases form when electrode and electrolyte are in contact.
3. **DFT surface/interface models:** Construct slab models of the electrode surface and adsorb solvent molecules to compute adsorption energies, decomposition barriers, and initial SEI precursors. This is a dft-surfaces workflow.
4. **Kinetic passivation:** SEI growth self-limits when the electronic conductivity drops below a critical value, blocking further electrolyte reduction. Full modeling requires coupled ionic-electronic transport.

[EXPERT REVIEW NEEDED: SEI composition, thickness, morphology, and evolution depend on many poorly controlled variables — electrolyte additives, current density, temperature, formation protocol, electrode surface chemistry — that are not captured by thermodynamic stability calculations alone.]

## Key Workflows

### Workflow 1: Average Intercalation Voltage and Voltage Profile for LFP

LiFePO₄ (LFP, olivine) undergoes a two-phase reaction between LiFePO₄ and FePO₄ at ~3.45 V vs Li/Li⁺. Fe²⁺/Fe³⁺ is the redox couple. This calculation requires PBE+U with U_eff(Fe) ≈ 4.3 eV and correct antiferromagnetic ordering.

```python
from pymatgen.core import Structure, Element
from pymatgen.entries.computed_entries import ComputedEntry
from pymatgen.analysis.phase_diagram import PhaseDiagram
import numpy as np

# DFT+U total energies (eV per formula unit, VASP INCAR: LDAUU = 4.3 for Fe)
# Correct magnetic ordering: antiferromagnetic for LFP/FP
E_LFP = -58.273   # LiFePO4 per f.u., AFM ground state
E_FP  = -46.981   # FePO4 per f.u., AFM ground state
E_Li  = -1.896    # BCC Li metal per atom

# Average voltage
V_avg = -(E_FP - E_LFP + E_Li) / (1.0 * 1.0)
# Numerator: removing 1 Li from LFP; denominator: 1 Li × 1 electron
print(f"Average LFP voltage: {V_avg:.3f} V vs Li/Li+")
# Expected: ~3.4–3.5 V (experiment: 3.45 V)

# Voltage profile: loop over intermediate compositions if available
compositions = [0.0, 0.25, 0.5, 0.75, 1.0]   # x in Li_x FePO4
# E_list[i] must be DFT energies at each x (from separate VASP calculations)
# E_list = [E_FP, E_Li0.25FP, E_Li0.5FP, E_Li0.75FP, E_LFP]
#
# Voltage between consecutive stable compositions:
# V(x1->x2) = -(E(x2) - E(x1) - (x2-x1)*E_Li) / (x2-x1)
```

Validation targets: experiment gives 3.45 V plateau, flat two-phase profile, 170 mAh/g capacity.

### Workflow 2: Convex Hull and Phase Stability Screening for Cathode Candidates

This workflow screens a set of Li-TM-O₂ compositions for thermodynamic stability using pymatgen and the Materials Project database.

```python
from mp_api.client import MPRester
from pymatgen.analysis.phase_diagram import PhaseDiagram, PDEntry
from pymatgen.core.composition import Composition
import pandas as pd

API_KEY = "YOUR_MP_API_KEY"   # store in environment variable, not in script

with MPRester(API_KEY) as mpr:
    # Retrieve all computed entries in the Li-Mn-O system
    entries = mpr.get_entries_in_chemsys(["Li", "Mn", "O"])

pd = PhaseDiagram(entries)
results = []

# Target compositions to check: Li_{x}MnO2 family
for x in [0.0, 0.25, 0.5, 0.75, 1.0]:
    comp = Composition(f"Li{x}MnO2") if x > 0 else Composition("MnO2")
    # Find the stable phases at this composition from the hull
    decomp, e_hull = pd.get_decomp_and_e_above_hull(
        PDEntry(comp, energy=0)   # placeholder; we ask for the hull energy
    )
    results.append({"x": x, "e_hull_meV_at": e_hull * 1000})

df = pd.DataFrame(results)
print(df)
# Filter candidates with e_hull < 50 meV/atom for further DFT relaxation
candidates = df[df["e_hull_meV_at"] < 50]
```

[EXPERT REVIEW NEEDED: Materials Project GGA/GGA+U mixing corrections are applied to entries, but the mixing scheme may not be consistent for all transition-metal systems; always check the correction scheme used for your specific TM-O compound family.]

### Workflow 3: Ion Migration Barrier Estimation and Arrhenius Conductivity

This workflow illustrates setting up NEB endpoints for Li migration in a spinel-type structure and connecting the DFT barrier to an experimental conductivity target.

```python
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.io.vasp.sets import MITNEBSet
import numpy as np

# Load fully relaxed endpoint structures
s_start = Structure.from_file("POSCAR_Li_at_16c")  # Li in octahedral 16c site
s_end   = Structure.from_file("POSCAR_Li_at_8a")   # Li in tetrahedral 8a site

# Generate NEB images (linear interpolation as starting guess)
n_images = 7
images = s_start.interpolate(s_end, n_images + 2, interpolate_lattices=False)
# images[0] = start, images[-1] = end, images[1:-1] = NEB images

# Write VASP input files
neb_set = MITNEBSet(images)
neb_set.write_input("neb_run")

# --- Post-processing: extract barrier from NEB output ---
# (Run VASP CI-NEB, then read OUTCAR files for each image)

def arrhenius_conductivity(E_a_eV, sigma0, T_K):
    """σ(T) = σ₀/T · exp(-Ea/kT)"""
    kB = 8.617e-5  # eV/K
    return sigma0 / T_K * np.exp(-E_a_eV / (kB * T_K))

# Example: LGPS-type solid electrolyte, Ea from NEB
E_a = 0.22   # eV, computed CI-NEB barrier for Li along c-channel
sigma0 = 1e7  # S·K/cm (prefactor; fit from experimental Arrhenius plot)
T = 298        # K

sigma_RT = arrhenius_conductivity(E_a, sigma0, T)
print(f"Predicted σ(298K) ≈ {sigma_RT:.2e} S/cm")
# LGPS experiment: ~12 mS/cm at RT
# Discrepancy > 2× signals that single-barrier NEB is insufficient;
# use MLMD + Green-Kubo or mean square displacement for quantitative conductivity.
```

For quantitative RT conductivity predictions, machine-learning MD (MACE, NEQUIP) run at 300–1200 K with the Green-Kubo formalism or via mean square displacement analysis is significantly more reliable than NEB Arrhenius extrapolation. [EXPERT REVIEW NEEDED]

### Workflow 4: Electrochemical Stability Window via Grand Potential Phase Diagram

```python
from mp_api.client import MPRester
from pymatgen.analysis.phase_diagram import GrandPotentialPhaseDiagram
from pymatgen.core.composition import Element
import numpy as np

API_KEY = "YOUR_MP_API_KEY"

with MPRester(API_KEY) as mpr:
    # Retrieve entries for the Li-P-S system (for LGPS/Li3PS4 type SE)
    entries = mpr.get_entries_in_chemsys(["Li", "P", "S"])

# Vary μ_Li from -4.0 to 0.0 eV (relative to Li metal at 0 V vs Li/Li+)
# potential vs Li/Li+ = -μ_Li/F (F=1 for per-electron quantities in eV/e)
mu_Li_values = np.linspace(-4.0, 0.0, 81)

stability_range = []
for mu_Li in mu_Li_values:
    gpd = GrandPotentialPhaseDiagram(entries, {Element("Li"): mu_Li})
    # Check if a target electrolyte phase is stable
    # (this requires retrieving the specific entry and checking hull distance)
    # --- placeholder for phase stability check ---
    stability_range.append(mu_Li)

print("Scan μ_Li range to identify reduction and oxidation onset potentials")
print("Reduction (Li plating): μ_Li → 0 V (Li metal deposition, ~0 V vs Li/Li+)")
print("Oxidation: phase changes at positive μ_Li offset signal electrolyte decomposition")
# Expected for LGPS: thermodynamic window ~1.7–2.1 V vs Li/Li+;
# practical window extended to ~0–5 V by kinetic passivation [EXPERT REVIEW NEEDED]
```

### Workflow 5: High-Throughput Cathode Screening with Computational Descriptors

```python
from mp_api.client import MPRester
import pandas as pd

API_KEY = "YOUR_MP_API_KEY"

with MPRester(API_KEY) as mpr:
    # Query all Li-TM-O ternaries with e_above_hull < 50 meV/atom
    docs = mpr.summary.search(
        elements=["Li", "O"],
        num_elements=(3, 4),
        energy_above_hull=(0, 0.05),   # eV/atom
        fields=["material_id", "formula_pretty", "energy_above_hull",
                "band_gap", "volume", "nsites", "theoretical"]
    )

df = pd.DataFrame([{
    "mp_id": d.material_id,
    "formula": d.formula_pretty,
    "e_hull": d.energy_above_hull * 1000,   # meV/atom
    "band_gap": d.band_gap,
    "volume": d.volume,
    "theoretical": d.theoretical,
} for d in docs])

# Filter: insulators (band_gap > 1 eV) and experimental phases preferred
candidates = df[(df["band_gap"] > 1.0)].sort_values("e_hull")
print(f"{len(candidates)} candidate Li-TM-O cathode materials")
print(candidates.head(20))

# Next steps for shortlisted candidates:
# 1. Retrieve structure and check provenance (experimental vs computed)
# 2. Check for correct magnetic ordering
# 3. Run voltage calculation (Workflow 1 protocol)
# 4. Run NEB migration barrier screening (Workflow 3 protocol)
# 5. Run electrochemical stability window check (Workflow 4)
# 6. Flag materials with large volume change (> 5%) on delithiation
```

**Provenance check (mandatory):** Always verify whether a retrieved structure is experimental (from ICSD) or purely predicted. Structures tagged `theoretical=True` in Materials Project have not been experimentally confirmed and may represent metastable or hypothetical phases.

### Workflow 6: Degradation Descriptor Calculation — Volume Change on Cycling

Volume change on full delithiation is a key mechanical degradation descriptor. Materials with ΔV/V > ~10% are prone to cracking, particle pulverization, and capacity fade.

```python
from mp_api.client import MPRester

API_KEY = "YOUR_MP_API_KEY"

def get_volume_change(mp_id_lithiated, mp_id_delithiated, api_key):
    with MPRester(api_key) as mpr:
        s_lit   = mpr.get_structure_by_material_id(mp_id_lithiated)
        s_delit = mpr.get_structure_by_material_id(mp_id_delithiated)
    vol_change = (s_delit.volume - s_lit.volume) / s_lit.volume * 100
    return vol_change

# LCO: LiCoO2 (mp-24850) → CoO2
# LFP: LiFePO4 (mp-19017) → FePO4 (mp-25064)
dv_lfp = get_volume_change("mp-19017", "mp-25064", API_KEY)
print(f"LFP volume change on delithiation: {dv_lfp:.1f}%")
# Experiment: LFP ~6.8%, LCO ~2% (c-axis contract), NMC ~5-8%, Si anode ~300% [EXPERT REVIEW NEEDED]

# Key degradation mechanisms to evaluate:
degradation_checklist = {
    "volume_change_pct": "< 5% preferred; > 10% high cracking risk",
    "oxygen_release": "Li-rich NMC: activated above ~4.4 V; check using O p-band center descriptor",
    "TM_dissolution": "Mn dissolution in spinel at elevated T; Ni dissolution in NMC",
    "Li_plating_risk": "Anode potential < 0 V vs Li/Li+ under fast charge",
    "SEI_stability":   "Electrolyte reduction product thermodynamics vs potential",
    "dendrite_risk":   "Li metal: current density vs sand time estimate",
}
for k, v in degradation_checklist.items():
    print(f"  {k}: {v}")
```

## Common Failure Modes

| Failure Mode | Symptom | Cause | Diagnostic | Fix |
|--------------|---------|-------|------------|-----|
| Wrong Li/Na/Mg chemical potential reference | Voltage shifted by 0.5–3 V vs experiment; incorrect phase stability | Used μ_Li = 0, molecular reference, wrong polymorph, or mixed GGA/GGA+U without correction | Recompute E(Li_bcc), E(Na_bcc), E(Mg_hcp) at the same DFT level; cross-check against known calibration material (e.g., LFP ~3.45 V vs Li/Li+) | Use BCC Li metal at the same VASP INCAR settings; apply GGA+U mixing correction for oxide phases |
| Unstable charged supercells | NEB or defect calculation diverges or gives nonsensical energy; Fermi level outside gap | Charged supercell with net charge not handled correctly; background charge not balanced; NELECT wrong | Check NELECT in INCAR; verify total charge; inspect LOCPOT for potential divergence | Set NELECT = NELECT_neutral ± charge; apply image-charge correction (FNV/KO); validate against neutral reference |
| Confusing thermodynamic and kinetic stability | Claims that a material is "stable" from hull distance but it decomposes in practice; or dismisses a functional material as unstable | e_above_hull is 0 K DFT; kinetic trapping, synthesis conditions, and surface energy are ignored | Compare e_hull against synthesis reports; check finite-T entropy contributions; review synthesis history | Treat e_hull < 25–50 meV/atom as "potentially synthesizable," not guaranteed; acknowledge metastability |
| Overinterpreting dilute-defect NEB barriers | Predicted Arrhenius conductivity off by orders of magnitude from experiment | Single-vacancy NEB ignores correlation effects, multi-ion hops, interstitial mechanisms, amorphous grain boundaries, defect-defect interactions | Compare against MLMD Green-Kubo conductivity; check concentration dependence of barrier | Use MLMD-derived conductivity for quantitative predictions; treat NEB Ea as screening filter only |
| Missing magnetic states in TM oxides | Voltage errors 0.2–1 V; wrong phase stability; incorrect ground-state structure | Default ferromagnetic or non-magnetic INCAR settings; insufficient magnetic initialization; local minima trapping | Initialize multiple spin configurations; check OSZICAR for magnetization; compare total energies across spin states | Run ferromagnetic, antiferromagnetic, and ferrimagnetic initializations; use MAGMOM from literature or ICSD; confirm ground state |
| Ignoring finite-temperature phase behavior | Phase diagram shows stability at 0 K, but phase is absent at synthesis temperature | Vibrational entropy, configurational disorder, or magnetic transitions flip stability at elevated T | Compare with CALPHAD assessment; estimate vibrational entropy contribution via phonon calculation | Use CALPHAD for temperature-dependent phase stability; apply phonon-informed free energy corrections for critical phases |
| Comparing computed voltage to experiment without corrections | DFT voltage consistently low by 0.3–0.5 V for Co/Ni oxides | GGA/PBE+U systematic error; inconsistent reference state; missing electrostatic corrections | Plot DFT vs experiment for a calibration set of known cathodes; apply fitted correction | Apply empirical DFT voltage correction (e.g., +0.35 V for GGA+U Co-oxide [EXPERT REVIEW NEEDED]); use HSE06 for quantitative accuracy at higher cost |
| Using database structures without provenance checks | Ground-state energy wrong; incorrect structure symmetry; wrong magnetic ordering; phases missing from competition | Retrieved structure is theoretical (not experimentally confirmed), or uses different functional/U settings than calculation | Check `theoretical` flag in Materials Project; verify space group, lattice parameters, and composition against ICSD or literature | Retrieve structure provenance; re-relax with consistent INCAR before using in voltage or stability calculations |

## Best Practices

- **Establish the reference state first.** Before any voltage or stability calculation, confirm the DFT energy of the mobile-ion metal (Li_bcc, Na_bcc, Mg_hcp) at the same INCAR settings (ENCUT, KPOINTS, functional, U correction, magnetism) as all other calculations.
- **Use consistent DFT settings across all phases in a phase diagram.** Mixing GGA energies for metallic phases with GGA+U energies for oxide phases without a correction scheme introduces systematic errors of 0.2–0.5 eV/f.u.
- **Initialize multiple magnetic states for every TM compound.** Antiferromagnetic ordering is often the ground state for Fe²⁺, Mn³⁺, Ni²⁺ oxides; ferromagnetic initialization misses it.
- **Validate against a known calibration material before screening.** Run LFP (3.45 V) or LCO (3.9 V) before computing novel cathodes; if calibration is off by > 0.1 V, fix the setup before proceeding.
- **Report the stability window as thermodynamic, not kinetic.** Always note that electrochemical stability windows from grand potential phase diagrams are thermodynamic lower bounds; kinetic passivation can extend practical windows.
- **Separate thermodynamic and kinetic stability arguments.** Phase is above the hull → metastable, not necessarily non-functional; migration barrier is low → fast, but actual conductivity requires defect concentrations and correlation factors.
- **Validate NEB migration barriers with convergence tests.** Check image count (5 vs 7 vs 9), supercell size (2×2×2 vs 3×3×3), and k-point density on the barrier value before reporting.
- **Track chemical potential conditions for defect calculations.** Li-rich vs Li-poor growth conditions dramatically change vacancy formation energies and thus defect concentrations.
- **Verify database structure provenance.** Flag theoretical structures explicitly; re-relax with consistent settings before use in thermodynamic calculations.
- **Version-control all INCAR, KPOINTS, and POTCAR metadata.** Small changes in ENCUT, SIGMA, or ISMEAR can shift total energies by tens of meV; commit parameter files alongside calculated energies.

## Integration with Other Skills

- **[corrosion-electrochemistry](../corrosion-electrochemistry/SKILL.md):** Shares electrode potential conventions, Pourbaix analysis, and electrochemical stability concepts; battery degradation at the electrode-electrolyte interface is electrochemical corrosion.
- **[dft-defects](../../electronic-structure/dft-defects/SKILL.md):** Provides vacancy, interstitial, and antisite formation energy formalism for solid electrolytes, cathodes, and anodes; chemical potential framework is shared.
- **[dft-surfaces](../../electronic-structure/dft-surfaces/SKILL.md):** Supports electrode surface and interface models, SEI/CEI precursor adsorption, surface Pourbaix stability, and coating adhesion.
- **[dft-convergence](../../electronic-structure/dft-convergence/SKILL.md):** Ensures ENCUT, KPOINTS, and SIGMA convergence for accurate total energies before voltage or phase stability calculations.
- **[high-throughput-dft](../../electronic-structure/high-throughput-dft/SKILL.md):** Automates multi-composition intercalation calculations, NEB screening, and stability windows across cathode families.
- **[dft-dataset-generation](../../electronic-structure/dft-dataset-generation/SKILL.md):** Generates curated DFT datasets (structures, energies, forces) for MLP training on battery materials.
- **[calphad-workflow](../../thermodynamics-calphad/calphad-workflow/SKILL.md):** Provides temperature-dependent phase stability, chemical potentials, and Gibbs energies for electrolyte and electrode systems beyond 0 K DFT.
- **[phase-field-workflow](../../phase-field-microstructure/phase-field-workflow/SKILL.md):** Connects battery thermodynamics to mesoscale phase separation, intercalation-induced phase transformation, and electrode microstructure evolution.
- **[fem-multiphysics](../../continuum-modeling/fem-multiphysics/SKILL.md):** Supports electrode-level ionic and electronic transport, stress evolution from volume change, thermal management, and dendrite nucleation models.
- **[materials-databases-access](../../scientific-data/materials-databases-access/SKILL.md):** Retrieves crystal structures, computed energies, stability data, experimental properties, and provenance from Materials Project, ICSD, and battery-specific databases.
- **[pymatgen-analysis](../../scientific-data/pymatgen-analysis/SKILL.md):** Core library for structure manipulation, convex hull construction, phase diagram building, NEB path generation, and electrochemical stability window analysis.
- **[materials-featurization](../../scientific-data/materials-featurization/SKILL.md):** Extracts structural and chemical descriptors for surrogate modeling of voltage, capacity, migration barriers, and degradation rates.
- **[mlp-validation](../../ml-interatomic-potentials/mlp-validation/SKILL.md):** Validates MACE, NequIP, or CHGNET potentials trained on battery material DFT data before use in MD conductivity or volume-change calculations.
- **[active-learning-mlp](../../ml-interatomic-potentials/active-learning-mlp/SKILL.md):** Iteratively refines MLP training data for battery materials by selecting configurations with high uncertainty from MD trajectories.
- **[scientific-visualization](../../scientific-data/scientific-visualization/SKILL.md):** Produces voltage profiles, convex hull diagrams, NEB barrier plots, Arrhenius conductivity fits, electrochemical stability windows, and phase diagrams.
- **[reproducibility-workflow](../../scientific-data/reproducibility-workflow/SKILL.md):** Tracks DFT parameter provenance, database query versions, NEB job metadata, and computed energy tables for reproducible battery materials results.
- **[gaussian-process-materials](../../surrogate-active-learning/gaussian-process-materials/SKILL.md):** Builds uncertainty-aware surrogates for voltage, migration barrier, or degradation rate as a function of composition or dopant.
- **[bayesian-optimization](../../surrogate-active-learning/bayesian-optimization/SKILL.md):** Efficiently optimizes cathode composition, dopant level, or solid electrolyte stoichiometry against target voltage, conductivity, or stability metrics.
- **[surrogate-validation](../../surrogate-active-learning/surrogate-validation/SKILL.md):** Validates composition-to-property surrogates under compositional extrapolation and distribution shift for battery screening campaigns.
- **[small-data-ml](../../scientific-ml/small-data-ml/SKILL.md):** Supports leakage-safe regression or classification for battery property datasets that are small, heterogeneous, or sourced from multiple labs.
- **[materials-lca](../../sustainability-lca/materials-lca/SKILL.md):** Connects computed battery material performance (energy density, cycle life, efficiency) to life-cycle environmental impact, critical material use, and end-of-life recovery.

## Key References

1. Aydinol, M. K., Kohan, A. F., Ceder, G., Cho, K. & Joannopoulos, J. (1997). Ab initio study of lithium intercalation in metal oxides and metal dichalcogenides. *Physical Review B*, 56(3), 1354–1365. https://doi.org/10.1103/PhysRevB.56.1354 — Foundational paper establishing DFT voltage calculation methodology.

2. Ong, S. P., Wang, L., Kang, B. & Ceder, G. (2008). Li-Fe-P-O₂ phase diagram from first principles calculations: Implications for the electrochemical properties of LiFePO₄. *Chemistry of Materials*, 20(5), 1798–1807. https://doi.org/10.1021/cm702327g — Phase diagram and stability framework using pymatgen convex hull.

3. Hautier, G., Ong, S. P., Jain, A., Moore, C. J. & Ceder, G. (2012). Accuracy of density functional theory in predicting formation energies of ternary oxides from binary oxides and its implication on phase stability. *Physical Review B*, 85(15), 155208. https://doi.org/10.1103/PhysRevB.85.155208 — GGA/GGA+U mixing scheme and formation energy accuracy analysis.

4. Mo, Y., Ong, S. P. & Ceder, G. (2012). First principles study of the Li₁₀GeP₂S₁₂ lithium super ionic conductor material. *Chemistry of Materials*, 24(1), 15–17. https://doi.org/10.1021/cm203303y — NEB migration barriers and ionic conductivity prediction for LGPS solid electrolyte.

5. Zhu, Y., He, X. & Mo, Y. (2015). Origin of outstanding stability in the lithium solid electrolyte materials: Insights from thermodynamic analyses based on first-principles calculations. *ACS Applied Materials & Interfaces*, 7(42), 23685–23693. https://doi.org/10.1021/acsami.5b07517 — Electrochemical stability window methodology via grand potential phase diagrams.

6. Bai, P., Li, J., Brushett, F. R. & Bazant, M. Z. (2016). Transition of lithium growth mechanisms in liquid electrolytes. *Energy & Environmental Science*, 9(10), 3221–3229. https://doi.org/10.1039/C6EE01674J — Multiscale framework connecting DFT surface energetics to dendrite formation.
