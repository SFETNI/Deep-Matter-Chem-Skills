---
name: crystal-diffusion
domain: generative-matter
maturity: draft
tools:
  - MatterGen
  - DiffCSP
  - pymatgen
  - ASE
  - torch
tested_versions:
  MatterGen: "1.0.0"
  DiffCSP: "0.1.0"
  pymatgen: "2024.11.13"
  torch: "2.2.0"
requires_gpu: true
estimated_runtime: "minutes-to-days (generation: minutes; training: hours–days)"
---

# Crystal Diffusion

## Description

This skill covers de novo crystal structure generation using score-based diffusion and flow-matching generative models, including MatterGen (Microsoft Research), DiffCSP (Jiao et al.), and CDVAE (Xie et al.). These models learn the distribution of stable inorganic crystals from crystallographic databases and sample novel structures either unconditionally or conditioned on chemical composition, space group, or target properties. Invoke this skill when generating crystal structure candidates for materials discovery, when proposing hypothetical structures for a given formula, or when augmenting structural training sets with plausible synthetic crystals.

## Domain Context

Crystal structure generation differs fundamentally from molecular generation because of three structural features that must be respected throughout the diffusion process:

- **Periodic boundary conditions:** Atom positions are fractional coordinates on a 3D torus [0, 1)³. A diffusion process operating on fractional coordinates must handle wraparound: coordinates 0.02 and 0.98 are separated by only 0.04 in fractional units, not 0.96. Naive Euclidean diffusion in fractional space destroys this periodicity.
- **Lattice degrees of freedom:** A crystal is fully specified by the 3×3 lattice matrix **L** (or equivalently the six scalars a, b, c, α, β, γ) together with the fractional coordinates and atom types. The lattice determines the physical scale and shape of the unit cell. Diffusion models must jointly denoise both the lattice and the atomic positions, and the parameterization of **L** affects equivariance properties.
- **Discrete atom types:** Element identities are discrete categorical variables. Most models represent them as continuous embeddings during diffusion and decode to discrete types at generation time, but the discretization step introduces additional approximation.

Physical concepts that determine model validity:

- **SE(3)-equivariance:** Crystal properties (energy, forces) are invariant under rotations and translations of the crystal. The generative model's score network or velocity field must be equivariant under these operations. Models based on MACE, NequIP, or e3nn backbones provide exact equivariance; models based on standard graph convolutions with distance-based features provide approximate invariance. [EXPERT REVIEW NEEDED: degree to which equivariance matters for generation quality vs. only for property prediction]
- **Score matching and denoising diffusion:** Score-based models learn the score function ∇_x log p_t(x) at each noise level t. At generation time, Langevin dynamics or an ODE/SDE solver follows the learned score backwards from noise to the data distribution. The score must be learned on the correct metric for each degree of freedom (Euclidean for fractional coordinates, SO(3)-like for the lattice, categorical for atom types).
- **Flow matching:** Learns a time-dependent velocity field v(x, t) that transports a simple prior (Gaussian noise) to the data distribution via an ODE. Flow matching on Riemannian manifolds (RFM) handles the non-Euclidean geometry of the lattice matrix and fractional coordinate torus more naturally than score matching. FlowMM and related models use this approach.
- **Thermodynamic plausibility vs. stability:** A generated structure that passes geometric validity checks (no overlapping atoms, reasonable bond lengths, charge balance) is not guaranteed to be thermodynamically stable. Stability requires DFT total-energy evaluation and comparison to competing phases on the convex hull. Pre-screening with a universal MLP (MACE-MP-0, CHGNet, M3GNet) is fast enough to apply to all generated structures before committing DFT compute.
- **Composition space:** Some models (CDVAE, DiffCSP) are conditioned on a fixed composition (the number and type of each element) provided by the user. Others (MatterGen) can generate composition jointly or condition on element-level properties. Composition-conditioned generation is more targeted for materials discovery but requires specifying a plausible composition.
- **Crystal validity metrics:** The community evaluates generative models using: (1) *validity* — fraction of generated structures passing geometric checks; (2) *uniqueness* — fraction that are structurally distinct from each other; (3) *novelty* — fraction not present in the training database; (4) *FCD (Fréchet Crystal Distance)* — distributional similarity between generated and reference structures in an embedding space; (5) *match rate* — fraction of generated structures that match DFT-relaxed experimental structures (for CSP benchmarks).

## When to Use This Skill

- Generating novel inorganic crystal structure candidates for multi-objective materials discovery when the chemical composition or element range is known but the crystal structure is not.
- Crystal structure prediction (CSP) for a specific composition when no prior structure is known and traditional ab initio random structure searching would be too slow for rapid screening.
- Augmenting ML training datasets with plausible hypothetical structures to improve coverage of chemical space, especially for rare element combinations underrepresented in experimental databases.
- Conditional generation — proposing structures compatible with a target space group, composition, or property constraint (band gap, stability) when MatterGen's conditioning capabilities are available.
- Exploring structural diversity around a known stable phase to generate nearby polymorphs or defect-free supercell variants.

## When Not to Use This Skill

- **High-accuracy CSP for a specific target compound** where the correct structure is the primary deliverable — evolutionary algorithms (USPEX, CALYPSO) and ab initio random structure searching (AIRSS) are more rigorous for single-composition CSP and do not require large training databases.
- **Molecular crystals (organic, pharmaceutical)** — crystal diffusion models are trained predominantly on inorganic materials databases (ICSD, Materials Project); molecular crystal packing requires models that account for intramolecular conformations (e.g., DiffCrys, Uni-Mol) and is outside the scope of these models.
- **Protein or biomolecular structure prediction** — use AlphaFold3, RoseTTAFold, or ESMFold.
- **Small organic molecule generation** — use molecular-scale diffusion models (EDM, DiffSBDD, GEOM-DDPM).
- **When a DFT-validated structure is required immediately without additional DFT relaxation** — generated structures are hypothetical and must be relaxed and energy-evaluated. Do not use raw generated coordinates as inputs to property calculations without prior relaxation.
- **Systems requiring explicit treatment of disorder, vacancies, or partial occupancies** — current crystal diffusion models generate ordered periodic structures with integer occupancies. Disordered or defect structures require post-processing or dedicated models.

## Core Concepts

- **Fractional coordinate torus:** Atomic positions (x, y, z) ∈ [0, 1)³ are defined modulo lattice translations. Distance between two fractional positions must account for the minimum image convention. Models that diffuse in fractional space must apply a wrapped Gaussian or von Mises distribution rather than an unconstrained Gaussian.
- **Lattice matrix parameterization:** The lattice matrix **L** = [**a**, **b**, **c**] ∈ ℝ³ˣ³ defines the unit cell. Some models diffuse the lattice directly in matrix space (CDVAE, DiffCSP); others use a lower-triangular or Cholesky parameterization to enforce positive definiteness of the metric tensor. The parameterization choice affects whether the diffusion respects physical symmetries.
- **Composition conditioning:** In composition-conditioned models, the number and type of atoms per unit cell (the "composition vector") is fixed and provided as input. The model generates only the positions and lattice, not the element identities. This is appropriate for CSP but not for open-ended discovery.
- **DDPM noise schedule:** Denoising Diffusion Probabilistic Models add Gaussian noise over T steps. Crystal diffusion models apply: a wrapped Gaussian to fractional coordinates, an isotropic Gaussian scaled by the lattice to Cartesian positions, and a continuous relaxation of categorical noise to atom types. The noise schedule (linear, cosine, or EDM) must be chosen per degree of freedom.
- **Classifier-free guidance (CFG):** MatterGen and similar models support conditional generation via CFG. A conditioning signal (e.g., target band gap, space group, composition) is provided during training with dropout; at generation time, the score is interpolated between conditional and unconditional directions. CFG strength (guidance scale) controls how strongly the generated distribution is steered toward the conditioning target.
- **Structure matching and deduplication:** Generated structures must be deduplicated using structure-matching criteria (e.g., pymatgen's `StructureMatcher`, `StructureComparator`) before DFT. Two structures are considered identical if they are related by a combination of translations, rotations, and site permutations within a tolerance.
- **FCD (Fréchet Crystal Distance):** Analogous to FID in image generation. A crystal encoder (e.g., trained on Materials Project property prediction) maps structures to a latent vector; FCD measures the Fréchet distance between the generated and reference structure distributions in that latent space. Lower FCD indicates that generated structures are more similar in distribution to real crystals.
- **Convex hull stability:** A structure is thermodynamically stable if its DFT formation energy per atom is at or below the convex hull of all competing phases in the same chemical system. The energy above the convex hull (E_hull) quantifies metastability; structures with E_hull < 0.05–0.1 eV/atom are considered practically accessible.

## Key Workflows

### Workflow 1: Crystal generation with MatterGen

MatterGen is a flow-matching generative model trained on the Materials Project. It generates crystal structures by reversing a diffusion process over the joint space of lattice parameters, fractional coordinates, and atom types.

```python
# Requirements: pip install mattergen
# MatterGen checkpoint: download from https://github.com/microsoft/mattergen (HuggingFace Hub)
# GPU strongly recommended; generation of 100 structures takes ~2-5 min on A100

import torch
from pathlib import Path
from mattergen.generator import MatterGenGenerator
from mattergen.evaluation.utils import get_structure_list
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

# Load the pretrained model
checkpoint_path = Path("checkpoints/mattergen_base")  # download from HuggingFace
generator = MatterGenGenerator.from_pretrained(str(checkpoint_path))
generator.eval()

# --- Unconditioned generation ---
# Generate 100 crystals with default settings
# Default: T=1000 diffusion steps, guidance_scale=1.0, batch_size=64
results = generator.generate(
    num_samples=100,
    batch_size=64,          # Reduce to 16 if GPU memory < 16 GB
    num_steps=1000,         # Diffusion steps; 200 for fast draft, 1000 for quality
    temperature=1.0,        # Sampling temperature; < 1.0 focuses on high-likelihood structures
)

# results is a list of pymatgen Structure objects (or ASE Atoms via AseAtomsAdaptor)
structures = results.structures
print(f"Generated {len(structures)} structures")

# --- Composition-conditioned generation ---
# Condition on Li-Mn-O compositions (e.g., for battery material discovery)
from mattergen.conditioning import CompositionConditioning

condition = CompositionConditioning(
    compositions=["Li1Mn1O2", "Li2MnO3", "LiMn2O4"],  # one sample per composition
    guidance_scale=3.0,  # Higher guidance = more faithful to condition; reduces diversity
)

cond_results = generator.generate(
    num_samples=len(condition.compositions),
    batch_size=8,
    num_steps=1000,
    conditioning=condition,
)

# --- Space-group-conditioned generation ---
from mattergen.conditioning import SpaceGroupConditioning

sg_condition = SpaceGroupConditioning(
    space_groups=[225, 225, 225, 194, 194],  # Fm-3m, Fm-3m, ..., P6_3/mmc
    guidance_scale=2.0,
)

sg_results = generator.generate(
    num_samples=5,
    batch_size=5,
    num_steps=1000,
    conditioning=sg_condition,
)

# Save as CIF files for downstream analysis
output_dir = Path("generated_structures")
output_dir.mkdir(exist_ok=True)
for i, struct in enumerate(structures):
    struct.to(filename=str(output_dir / f"gen_{i:04d}.cif"))
print(f"Saved {len(structures)} structures to {output_dir}")
```

### Workflow 2: Crystal structure prediction with DiffCSP

DiffCSP uses joint equivariant diffusion over fractional coordinates and the lattice matrix, conditioned on a fixed composition. It is well-suited for composition-driven crystal structure prediction.

```python
# Requirements: pip install diffcsp
# DiffCSP checkpoint: download from https://github.com/jiaor17/DiffCSP
# Original paper: Jiao et al., NeurIPS 2023

import torch
import numpy as np
from pathlib import Path
from diffcsp.pl_modules.diffusion import CSPDiffusion
from diffcsp.common.data_utils import (
    lattice_params_to_matrix_torch,
    get_pbc_distances,
)
from pymatgen.core import Structure, Lattice

# Load pretrained DiffCSP model (trained on MP-20 subset of Materials Project)
checkpoint = torch.load("diffcsp_mp20.ckpt", map_location="cuda")
model = CSPDiffusion.load_from_checkpoint("diffcsp_mp20.ckpt")
model.eval()
model = model.cuda()

# Define target compositions for crystal structure prediction
# Each composition is specified as element symbols + atom counts per unit cell
compositions = [
    {"elements": ["Li", "Fe", "P", "O"], "num_atoms": [4, 4, 4, 16]},  # LiFePO4 (olivine)
    {"elements": ["Ba", "Ti", "O"],      "num_atoms": [1, 1, 3]},       # BaTiO3 (perovskite)
    {"elements": ["Mg", "O"],            "num_atoms": [4, 4]},           # MgO (rocksalt)
]

def run_diffcsp_prediction(model, composition: dict, num_samples: int = 20,
                           num_steps: int = 1000) -> list[Structure]:
    """
    Run DiffCSP for a single composition, returning a list of candidate structures.
    """
    elements = composition["elements"]
    num_atoms = composition["num_atoms"]
    total_atoms = sum(num_atoms)

    # Build atom-type tensor (Z) from composition
    atom_types = []
    for elem, count in zip(elements, num_atoms):
        from pymatgen.core import Element
        atom_types.extend([Element(elem).Z] * count)
    atom_types = torch.tensor(atom_types, dtype=torch.long).cuda()

    # Run reverse diffusion from Gaussian noise
    # DiffCSP jointly diffuses fractional coordinates (wrapped Gaussian) and
    # the lattice matrix (scaled by volume prior)
    with torch.no_grad():
        outputs = model.sample(
            atom_types=atom_types.unsqueeze(0).expand(num_samples, -1),
            num_atoms=torch.tensor([total_atoms] * num_samples).cuda(),
            num_steps=num_steps,
            step_lr=1e-4,           # Langevin step size; reduce if structures collapse
        )

    # Parse outputs into pymatgen Structure objects
    structures = []
    for i in range(num_samples):
        frac_coords = outputs["frac_coords"][i].cpu().numpy()  # (N, 3)
        lattice_mat = outputs["lattice"][i].cpu().numpy()       # (3, 3)
        types = outputs["atom_types"][i].cpu().numpy()          # (N,) atomic numbers

        from pymatgen.core import Element
        species = [Element.from_Z(int(z)).symbol for z in types]
        try:
            struct = Structure(
                lattice=Lattice(lattice_mat),
                species=species,
                coords=frac_coords,
                coords_are_cartesian=False,
            )
            structures.append(struct)
        except Exception:
            pass  # Skip malformed structures

    return structures


all_candidates = {}
for comp in compositions:
    formula = "".join(f"{el}{n}" for el, n in zip(comp["elements"], comp["num_atoms"]))
    print(f"Predicting structure for {formula}...")
    candidates = run_diffcsp_prediction(model, comp, num_samples=20)
    all_candidates[formula] = candidates
    print(f"  Generated {len(candidates)} candidates")

# Save candidates
for formula, candidates in all_candidates.items():
    out_dir = Path(f"csp_{formula}")
    out_dir.mkdir(exist_ok=True)
    for i, struct in enumerate(candidates):
        struct.to(filename=str(out_dir / f"candidate_{i:03d}.cif"))
```

### Workflow 3: Validity screening and evaluation of generated structures

Generated structures must be screened for geometric validity and then ranked by energy before DFT. This workflow applies a universal MLP for fast pre-relaxation, computes standard generation metrics, and identifies unique structures.

```python
import numpy as np
from pathlib import Path
from pymatgen.core import Structure
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.io.ase import AseAtomsAdaptor

# --- Step 1: Load generated structures ---
generated_dir = Path("generated_structures")
structures = []
for cif_path in sorted(generated_dir.glob("*.cif")):
    try:
        struct = Structure.from_file(str(cif_path))
        structures.append(struct)
    except Exception as e:
        print(f"Failed to read {cif_path.name}: {e}")

print(f"Loaded {len(structures)} structures")

# --- Step 2: Geometric validity check ---
def is_geometrically_valid(struct: Structure, min_distance_ang: float = 0.5) -> bool:
    """
    Check that no two atoms are closer than min_distance_ang.
    A threshold of 0.5 Å catches severely overlapping atoms without rejecting
    compressed but physically plausible structures.
    """
    from pymatgen.analysis.structure_analyzer import VoronoiConnectivity
    try:
        all_distances = struct.distance_matrix
        # Zero out diagonal (self-distances)
        np.fill_diagonal(all_distances, np.inf)
        min_dist = np.min(all_distances)
        return float(min_dist) >= min_distance_ang
    except Exception:
        return False


valid_structures = [s for s in structures if is_geometrically_valid(s)]
validity_rate = len(valid_structures) / len(structures) if structures else 0
print(f"Validity rate: {len(valid_structures)}/{len(structures)} = {validity_rate:.2%}")

# --- Step 3: Uniqueness (deduplication) ---
matcher = StructureMatcher(
    ltol=0.2,          # Fractional tolerance on lattice parameters
    stol=0.3,          # Site tolerance as fraction of average free length per atom
    angle_tol=5.0,     # Angle tolerance in degrees
    primitive_cell=True,  # Compare primitive cells to catch supercell duplicates
    scale=True,         # Normalize volumes before comparison
)

unique_structures = []
for struct in valid_structures:
    is_duplicate = any(matcher.fit(struct, u) for u in unique_structures)
    if not is_duplicate:
        unique_structures.append(struct)

uniqueness_rate = len(unique_structures) / len(valid_structures) if valid_structures else 0
print(f"Uniqueness rate: {len(unique_structures)}/{len(valid_structures)} = {uniqueness_rate:.2%}")

# --- Step 4: Pre-relax with universal MLP for fast energy ranking ---
# Uses MACE-MP-0 (universal potential); install with: pip install mace-torch
try:
    from mace.calculators import mace_mp
    from ase.optimize import BFGS

    calc = mace_mp(model="medium", dispersion=False, default_dtype="float32", device="cuda")
    adaptor = AseAtomsAdaptor()

    mlp_energies = {}
    relaxed_structures = {}

    for i, struct in enumerate(unique_structures):
        atoms = adaptor.get_atoms(struct)
        atoms.calc = calc
        try:
            optimizer = BFGS(atoms, logfile=None)
            optimizer.run(fmax=0.1, steps=200)  # Loose convergence for screening
            e_per_atom = atoms.get_potential_energy() / len(atoms)
            mlp_energies[i] = e_per_atom
            # Convert back to pymatgen Structure
            relaxed_structures[i] = adaptor.get_structure(atoms)
        except Exception as e:
            print(f"  Structure {i}: MLP relaxation failed ({e})")
            mlp_energies[i] = np.inf

    # Rank by MLP energy per atom
    ranked = sorted(mlp_energies.items(), key=lambda x: x[1])
    print("\nTop 10 structures by MLP energy (eV/atom):")
    for rank, (idx, e) in enumerate(ranked[:10]):
        print(f"  Rank {rank+1}: structure {idx:3d}  E/atom = {e:.4f} eV")

    # Save top-ranked unique structures for DFT
    dft_input_dir = Path("top_candidates_for_dft")
    dft_input_dir.mkdir(exist_ok=True)
    for rank, (idx, e) in enumerate(ranked[:20]):
        if idx in relaxed_structures:
            relaxed_structures[idx].to(
                filename=str(dft_input_dir / f"rank{rank+1:02d}_idx{idx:03d}.cif")
            )
    print(f"\nSaved top 20 candidates to {dft_input_dir}")

except ImportError:
    print("mace-torch not installed; skipping MLP pre-relaxation")
    print("Install with: pip install mace-torch")

# --- Step 5: Summary metrics ---
print("\n--- Generation Summary ---")
print(f"Total generated:   {len(structures)}")
print(f"Geometrically valid: {len(valid_structures)} ({validity_rate:.1%})")
print(f"Unique valid:        {len(unique_structures)} ({uniqueness_rate:.1%} of valid)")
print(f"Yield (unique valid / total): {len(unique_structures)/len(structures):.1%}")
```

### Workflow 4: Novelty check against a reference database

Novelty measures what fraction of generated structures are not present in the training database (e.g., Materials Project). This is distinct from uniqueness (which only compares generated structures to each other).

```python
from pymatgen.core import Structure
from pymatgen.analysis.structure_matcher import StructureMatcher
from mp_api.client import MPRester  # pip install mp-api
import os

def check_novelty_against_mp(
    structures: list[Structure],
    api_key: str | None = None,
    chemsys_filter: str | None = None,
) -> dict:
    """
    Check how many structures in `structures` match known Materials Project entries.
    Returns dict with novelty rate and lists of novel vs. known structures.

    Args:
        structures: List of candidate pymatgen Structures.
        api_key: Materials Project API key (defaults to MP_API_KEY env var).
        chemsys_filter: Restrict MP query to a specific chemical system, e.g. "Li-Fe-O".
                        If None, queries all systems matching each structure's elements.
    """
    api_key = api_key or os.environ.get("MP_API_KEY")
    matcher = StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5.0,
                               primitive_cell=True, scale=True)
    novel = []
    known = []

    with MPRester(api_key) as mpr:
        for i, struct in enumerate(structures):
            # Build chemical system string for this structure
            elements = sorted({str(s.element) for s in struct.species})
            chemsys = "-".join(elements)

            # Query MP for all structures with this chemical system
            try:
                mp_structures = mpr.get_structures(chemsys)
            except Exception:
                mp_structures = []

            is_novel = not any(matcher.fit(struct, ref) for ref in mp_structures)
            if is_novel:
                novel.append((i, struct))
            else:
                known.append((i, struct))

    novelty_rate = len(novel) / len(structures) if structures else 0
    return {
        "novelty_rate": novelty_rate,
        "novel_count": len(novel),
        "known_count": len(known),
        "novel_structures": [s for _, s in novel],
        "known_structures": [s for _, s in known],
    }


# Example usage (requires valid MP_API_KEY):
# result = check_novelty_against_mp(unique_structures[:50], chemsys_filter="Li-Mn-O")
# print(f"Novelty rate: {result['novelty_rate']:.1%}")
```

## Common Failure Modes

| Failure | Symptom | Likely Cause | Diagnostic | Fix |
|---------|---------|--------------|------------|-----|
| High rate of overlapping atoms | Validity < 50%; many structures fail the minimum-distance check | Diffusion step size too large for fractional coordinate denoising; model trained on different distribution than target chemistry | Print minimum pairwise distances for failed structures; check if failure rate is composition-dependent | Reduce sampling temperature to < 1.0; increase num_steps from 200 to 1000; verify checkpoint was trained on similar chemical space |
| Generated structures all nearly identical | Uniqueness < 20%; structure-matcher deduplication removes most samples | Mode collapse during training; guidance scale too high during conditional generation | Compute pairwise RMSD and plot distribution; should be broad | Increase sampling temperature; reduce guidance_scale for conditioned generation; use more diverse initial noise seeds |
| MLP relaxation explodes or produces unphysical densities | Atoms fly apart during BFGS; density < 0.5 g/cm³ or > 20 g/cm³ | Raw generated coordinates have severe force discontinuities; MLP calculator extrapolates outside training domain | Log energy before and after first BFGS step; check if energy increases | Apply stronger fmax threshold for early stopping; pre-screen with a simple Lennard-Jones repulsion before MACE; use tighter validity threshold (min_distance ≥ 1.0 Å) |
| All generated structures match training database (novelty ≈ 0) | 100% of structures found in MP reference query | Model has memorized training data; distribution collapse | Reduce sampling temperature to increase sharpness; separately verify that matcher tolerance is not too loose (tighten stol from 0.3 to 0.2) | If using a small model or checkpoint trained for too many epochs, use a checkpoint from an earlier training epoch; increase model capacity |
| Conditioning fails: generated structures ignore space group or composition constraint | Generated structures have wrong space group or wrong formula | Guidance scale too low; model not well trained on this conditioning type | Check distribution of space groups or compositions in generated batch; compute fraction that match the requested condition | Increase guidance_scale (try 3.0–6.0); verify conditioning input format matches model expectations exactly |
| DFT relaxation starting from generated structure diverges | DFT forces > 100 eV/Å at step 0; electronic structure does not converge; negative frequencies everywhere | Generated structure is too far from any local minimum on the DFT PES | Inspect structure in VESTA/OVITO before DFT; compute MLP energy and forces first | Always pre-relax with a universal MLP (MACE-MP-0 or CHGNet) to fmax < 0.5 eV/Å before handing to DFT; never use raw generated coordinates as DFT input |

## Best Practices

- **Always pre-relax with a universal MLP before DFT.** Generated structures contain coordinate noise from the reverse diffusion process. MACE-MP-0, CHGNet, or M3GNet can relax 1000 structures in the time it takes DFT to relax one. Only submit the top-ranked, pre-relaxed structures to DFT.
- **Validate with multiple metrics.** Report validity, uniqueness, and novelty together. A model with 95% validity but 20% novelty is likely memorizing training data. A model with 95% novelty but 30% validity is generating mostly unphysical structures. Target all three metrics simultaneously.
- **Use composition-conditioned generation for targeted discovery.** Unconditioned generation produces random compositions; if you are exploring a specific chemical space (e.g., Li-Mn-O for battery cathodes), provide composition constraints to focus compute on relevant structures.
- **Match the checkpoint's training distribution.** MatterGen's base checkpoint is trained on inorganic oxides and intermetallics from the Materials Project. Generating structures with unusual element combinations (actinides, rare earths, mixed-valence sulfides) may produce lower-quality outputs if these are rare in the training data.
- **Use primitive cell comparison in StructureMatcher.** Setting `primitive_cell=True` ensures that supercell duplicates are detected. Without this, a 1×1×1 and 2×1×1 version of the same structure are counted as unique.
- **Log generation metadata.** Record the model checkpoint path, guidance scale, number of diffusion steps, sampling temperature, and random seed with every batch of generated structures. Without this, results cannot be reproduced or compared across experiments.
- **For property-conditioned generation, verify the conditioning head is calibrated.** CFG guidance steers the generation toward high-probability regions under the conditioning signal. If the property predictor head is poorly calibrated for your target property range, the guidance will be misleading. Validate the conditioning model separately on held-out structures before relying on it.

## Integration with Other Skills

- **[universal-potentials](../../ml-interatomic-potentials/universal-potentials/SKILL.md):** MACE-MP-0, CHGNet, and M3GNet provide fast pre-relaxation and energy ranking of generated structures before DFT, enabling orders-of-magnitude reduction in DFT cost by filtering implausible candidates early.
- **[crystal-property-gnn](../../geometric-deep-learning/crystal-property-gnn/SKILL.md):** Graph neural network property predictors (MEGNet, ALIGNN, CGCNN) can be used as fast energy and property surrogates to rank generated structures, or as the conditioning target for property-conditioned generation via classifier-free guidance.
- **[high-throughput-dft](../../electronic-structure/high-throughput-dft/SKILL.md):** Top-ranked generated structures from this workflow feed directly into high-throughput DFT pipelines for energy evaluation, stability screening, and convex hull placement. The handoff point is a directory of pre-relaxed CIF files.
- **[dft-dataset-generation](../../electronic-structure/dft-dataset-generation/SKILL.md):** Generated novel structures can augment DFT training datasets by providing structurally diverse configurations in underrepresented regions of chemical space, improving MLP coverage without requiring experimental discovery of new compounds.
- **[pymatgen-analysis](../../scientific-data/pymatgen-analysis/SKILL.md):** `StructureMatcher`, `SpacegroupAnalyzer`, and convex hull analysis via `PhaseDiagram` are the primary tools for validating and characterizing generated structures. pymatgen's API is the standard interface between crystal diffusion models and downstream analysis.
- **[materials-databases-access](../../scientific-data/materials-databases-access/SKILL.md):** The Materials Project API provides reference structures for novelty checking and convex hull stability assessment. mp-api queries against the training database are the standard way to detect training-set memorization.
- **[active-learning-mlp](../../ml-interatomic-potentials/active-learning-mlp/SKILL.md):** Generated structures with diverse and novel compositions can seed active learning loops — novel hypothetical structures are labeled by DFT and incorporated into iterative MLP training, bridging generative exploration and potential refinement.
- **[scientific-visualization](../../scientific-data/scientific-visualization/SKILL.md):** VESTA is the standard tool for visualizing CIF structures from generated batches. OVITO handles trajectory visualization for pre-relaxation sequences. Property distributions of generated structures (density, lattice parameter histograms, space group distribution) are plotted with matplotlib.

## Key References

1. Zeni, C., Pinsler, R., Zügner, D., et al. (2025). *MatterGen: a generative model for inorganic materials design.* Nature, 639, 1018–1026. https://doi.org/10.1038/s41586-025-08628-5
2. Jiao, R., Huang, W., Lin, P., et al. (2024). *Crystal Structure Prediction by Joint Equivariant Diffusion.* Advances in Neural Information Processing Systems 36 (NeurIPS 2023). https://arxiv.org/abs/2309.04475
3. Xie, T., Fu, X., Ganea, O.-E., Barzilay, R., & Jaakkola, T. (2022). *Crystal Diffusion Variational Autoencoder for Periodic Material Generation.* ICLR 2022. https://arxiv.org/abs/2110.06197
4. Miller, B. K., Smidt, T. E., & Geiger, M. (2024). *FlowMM: Generating Materials with Riemannian Flow Matching.* Proceedings of the 41st International Conference on Machine Learning (ICML 2024). https://arxiv.org/abs/2406.04713
5. Pakornchote, T., Avery, P., Ektarawong, A., et al. (2024). *DiffCSP++: A Unified Framework for Crystal Structure Prediction via Equivariant Score-based Generative Models.* arXiv:2409.08954. https://arxiv.org/abs/2409.08954
6. Merchant, A., Batzner, S., Schoenholz, S. S., et al. (2023). *Scaling deep learning for materials discovery.* Nature, 624, 80–85. https://doi.org/10.1038/s41586-023-06735-9
