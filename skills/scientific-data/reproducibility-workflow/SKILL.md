---
name: reproducibility-workflow
domain: scientific-data
maturity: draft
tools:
  - Git
  - DVC
  - MLflow
  - ASE
  - pymatgen
tested_versions:
  Git: "2.43.0"
  DVC: "3.40.0"
  MLflow: "2.11.0"
  ASE: "3.23.0"
  pymatgen: "2024.1.26"
requires_gpu: false
estimated_runtime: minutes
---

# Reproducibility Workflow

## Description

This skill covers the systematic provenance tracking, data versioning, and environment capture required to make computational materials science campaigns reproducible: DFT dataset provenance, MLP training experiment tracking, active learning iteration versioning, environment snapshots, model registry management, and data integrity verification. Invoke this skill when designing a new simulation or ML campaign from scratch, when auditing an inherited dataset or model, or when preparing data and results for publication or long-term archiving.

## Domain Context

Reproducibility in computational materials science and scientific ML has two distinct failure modes that are easy to confuse. The first is **bit-exact reproducibility**: given identical inputs, code, and hardware, the same numbers appear. The second is **scientific reproducibility**: an independent researcher using the same protocol and nominally identical inputs arrives at the same scientific conclusions. The first is often achievable for DFT and classical MD; it becomes harder for GPU-accelerated ML training where floating-point nondeterminism is inherent. The second requires that the protocol itself — inputs, settings, preprocessing steps, and analysis code — is fully recorded.

In computational materials science, the most common reproducibility failures are not deliberate falsification but silent data corruption: a POTCAR file quietly updated mid-campaign, a preprocessing script run without recording its parameters, a train/test split generated with no fixed random seed, a model checkpoint overwritten by a re-run. Each of these produces results that look correct locally but cannot be reproduced six months later, or produce subtly different results when the dataset is extended.

**The provenance chain** in a typical MLP campaign spans at least four levels:

1. **DFT calculations:** Code version, pseudopotentials (checksums), functional, ENCUT/ecutwfc, k-mesh density, smearing, convergence thresholds, cell, PBC, MAGMOM. A change at this level invalidates all downstream training.
2. **Dataset assembly:** Which raw calculations were included, which were filtered (and why), the extxyz conversion script, the train/val/test split seed, and the resulting dataset file checksums.
3. **Model training:** Architecture hyperparameters, loss weights, random seed, training data file checksum, optimizer settings, number of epochs, the exact training command, and the validation metrics at the best checkpoint.
4. **Deployment and production:** Model file checksum, LAMMPS version and plugin hash, simulation input script, initial structure, random seed for velocity initialization, ensemble and thermostat settings.

**FAIR data principles** (Findable, Accessible, Interoperable, Reusable) are the upstream standard for materials data. Reproducibility is a prerequisite for FAIR reusability: data that cannot be reproduced cannot be reused. DFT dataset provenance metadata (functional, pseudopotentials, k-mesh) maps directly to the "Reusable" and "Interoperable" FAIR criteria and must be machine-readable, not buried in a paper's supplementary PDF.

**The tension between cost and completeness.** Full provenance tracking has overhead: every DFT run writes a sidecar JSON; every preprocessing step logs its inputs and outputs; every training run is tracked in MLflow. This overhead is negligible compared to DFT wall time and MLP training GPU-hours, but it requires upfront infrastructure setup. The alternative — tracking provenance manually or reconstructing it post-hoc — is expensive and error-prone at campaign scale (hundreds of DFT calculations, multiple AL iterations, dozens of training runs).

## When to Use This Skill

- Setting up the infrastructure for a new MLP training campaign before any DFT calculations are submitted.
- When a dataset is being shared with collaborators or submitted to a data repository (NOMAD, Materials Cloud, Zenodo).
- When preparing the methods section of a paper that reports MLP training results — the provenance metadata is the source of truth for what was actually computed.
- When inheriting a dataset or model from a prior project and needing to audit whether it is internally consistent and correctly annotated.
- When an active learning campaign runs for multiple iterations and the connection between model version N and the DFT batch that produced it must be unambiguous.
- When debugging an unexpected discrepancy between two runs that should have been identical.

## When Not to Use This Skill

- **For purely exploratory single-run calculations** where no downstream use is planned — full provenance overhead is not justified for a one-off convergence test that will be discarded.
- **As a substitute for proper experimental design** — tracking provenance of a poorly designed dataset does not fix the dataset.
- **When the overhead of Git-LFS or DVC exceeds the benefit** — for very small datasets (< 100 structures) stored in a private directory, a well-organized folder with checksum files and a README may be sufficient.
- **For real-time or latency-sensitive workflows** — provenance metadata writes add I/O overhead; they are appropriate for batch HPC workflows, not for low-latency inference services.

## Core Concepts

- **Git commit hash:** A 40-character SHA-1 identifier that uniquely and immutably identifies the state of the codebase at a given point. Every production run script, analysis notebook, and LAMMPS input file should be version-controlled; the Git commit hash at run time should be recorded in the run's metadata.
- **DVC (Data Version Control):** A Git-compatible version control system for large binary files (datasets, model weights) that Git was not designed to handle. DVC stores a small `.dvc` pointer file in Git while the actual data resides in a remote storage (S3, GCS, SSH). DVC enables dataset versioning with the same semantics as code versioning.
- **MLflow:** An experiment tracking platform. MLflow records hyperparameters (`mlflow.log_param`), metrics (`mlflow.log_metric`), artifacts (model files, figures, configs), and environment information for each training run. Provides a UI for comparing runs and a model registry for versioning production models.
- **SHA-256 file hash:** A 256-bit content hash that uniquely identifies a file's contents regardless of filename, path, or metadata. Used to detect silent file corruption, verify POTCAR identity, and confirm that the model file deployed in LAMMPS is the same one reported in the training log. `hashlib.sha256(open(f,"rb").read()).hexdigest()`.
- **Random seed:** An integer used to initialize a pseudorandom number generator. Fixing the seed in Python (`random.seed`, `np.random.seed`, `torch.manual_seed`), in LAMMPS velocity initialization (`velocity ... create ... seed`), and in train/test splitting ensures that the same sequence of random decisions is made in every run. Without a fixed seed, two runs on the same data can produce different train/test splits, different model initializations, and different MD initial conditions.
- **Environment snapshot:** A machine-readable record of all software versions used in a computation: Python package versions (`conda env export`, `pip freeze`), compiler versions (`gcc --version`, `mpicc --version`), CUDA version, LAMMPS version, and any HPC modules loaded. Required for reproducing results on a different machine or after a system upgrade.
- **Immutable dataset:** A dataset file (extxyz, HDF5, or directory) that is never modified in place. Once a dataset version is created, it receives a checksum and is treated as read-only. Any modification (filtering, adding structures) creates a new version with a new checksum. The link between dataset version and model version must be explicit.
- **Provenance sidecar:** A small JSON file written alongside each computational artifact (DFT output directory, model checkpoint, extxyz file) recording the inputs, settings, code version, and checksums that produced it. Sidecars are co-located with their artifacts and survive directory moves better than a centralized database.
- **Model registry:** A catalog of trained model versions with associated metadata: training data checksum, validation metrics, deployment status (staging / production / archived), and creation timestamp. MLflow provides a built-in registry; a simple JSON file is an acceptable minimum.
- **Idempotent pipeline:** A data processing pipeline where running the same step twice with the same inputs produces exactly the same output. Idempotent pipelines are a prerequisite for reproducibility. Steps that depend on system time, network calls, or uncontrolled random state are not idempotent.

## Key Workflows

### Workflow 1: Git + DVC repository setup for a simulation campaign

Initialize a version-controlled repository for a new MLP training campaign. The setup below assumes a standard campaign structure: raw DFT calculations → assembled dataset → trained model → validation results.

```bash
#!/bin/bash
# ─── Step 1: Initialize Git repository ─────────────────────────────────────
git init mlp_campaign_WTa
cd mlp_campaign_WTa

# ─── Step 2: Standard .gitignore for computational materials science ─────────
cat > .gitignore << 'EOF'
# DFT raw outputs — tracked by DVC, not Git
raw_dft/*/WAVECAR
raw_dft/*/CHGCAR
raw_dft/*/CHG
raw_dft/*/PROCAR
raw_dft/*/*.pwo.bak

# Model weights — tracked by DVC, not Git
models/*.model
models/*.pt
models/*.pb

# Conda and Python caches
__pycache__/
*.pyc
*.pyo
.ipynb_checkpoints/
*.egg-info/

# Large dataset files — tracked by DVC
datasets/*.xyz
datasets/*.h5

# LAMMPS output files
*.lammpstrj
*.dump
restart.*
log.lammps
EOF

# ─── Step 3: Initialize DVC ──────────────────────────────────────────────────
pip install dvc dvc-s3   # or dvc-ssh, dvc-gdrive depending on remote
dvc init

# ─── Step 4: Configure DVC remote (example: SSH cluster scratch) ─────────────
dvc remote add -d cluster ssh://hpc.example.ac.uk/scratch/group/mlp_campaign_WTa/dvc_store
# For S3: dvc remote add -d s3remote s3://your-bucket/mlp_campaign_WTa/dvc

# ─── Step 5: Create campaign directory structure ─────────────────────────────
mkdir -p raw_dft/{iter000,iter001,iter002}
mkdir -p datasets
mkdir -p models/{staging,production,archived}
mkdir -p validation/{eos,phonons,elastic,md_stability}
mkdir -p scripts
mkdir -p logs/mlflow

git add .gitignore .dvc/ .dvcignore
git commit -m "Initialize campaign repository with Git + DVC"
```

Tracking the canonical DFT settings and extxyz datasets:

```bash
# After assembling the first dataset:
dvc add datasets/iter000_train.xyz datasets/iter000_val.xyz datasets/iter000_test.xyz

# DVC creates .dvc pointer files (small text files tracked by Git)
git add datasets/iter000_train.xyz.dvc datasets/iter000_val.xyz.dvc datasets/iter000_test.xyz.dvc
git commit -m "Add iteration 0 dataset (N structures)"

# Push large files to remote storage
dvc push

# Later: a collaborator or CI job can reproduce the exact dataset version
git clone <repo>
dvc pull  # Downloads exact dataset files matching the Git commit
```

```python
# scripts/record_campaign_settings.py
# Run once at campaign start; commit the output alongside the .gitignore.
import json
import hashlib
import subprocess
from datetime import datetime
from pathlib import Path


def git_commit_hash() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True
    )
    return result.stdout.strip()


def sha256_file(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def record_campaign_settings(
    dft_code: str,
    dft_settings: dict,
    potcar_paths: list[str],
    output_file: str = "campaign_settings.json",
) -> None:
    """
    Write a campaign-level provenance record at the start of a DFT campaign.
    Commit this file to Git alongside the first batch of DFT inputs.
    """
    settings_str = json.dumps(dft_settings, sort_keys=True)
    settings_hash = hashlib.sha256(settings_str.encode()).hexdigest()[:16]

    potcar_checksums = {}
    for pp in potcar_paths:
        pp_path = Path(pp)
        if pp_path.exists():
            potcar_checksums[pp_path.name] = sha256_file(pp)
        else:
            potcar_checksums[pp_path.name] = "FILE NOT FOUND"

    record = {
        "created": datetime.utcnow().isoformat() + "Z",
        "git_commit": git_commit_hash(),
        "dft_code": dft_code,
        "dft_settings": dft_settings,
        "settings_hash": settings_hash,
        "potcar_checksums": potcar_checksums,
        "note": "FROZEN: do not modify DFT settings after this file is committed.",
    }

    Path(output_file).write_text(json.dumps(record, indent=2))
    print(f"Campaign settings written to {output_file}")
    print(f"Settings hash: {settings_hash}")
    print(f"Git commit: {record['git_commit']}")
    print("Commit this file before running any DFT calculations.")
```

### Workflow 2: DFT provenance capture per calculation

Extend the per-calculation provenance sidecar beyond what `DatasetManifest` tracks (see `dft-dataset-generation` Workflow 6), capturing everything needed to re-run or audit a single DFT calculation.

```python
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path


def sha256_file(path) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def record_dft_provenance(
    calc_dir: Path,
    config_type: str,
    al_iteration: int,
    dft_code: str,
    settings_hash: str,
    hpc_job_id: str = None,
) -> None:
    """
    Write a provenance JSON sidecar to a completed DFT calculation directory.
    Called after VASP/QE terminates, before the output is parsed.
    """
    # Capture VASP/QE code version from output
    code_version = "unknown"
    if dft_code == "VASP":
        outcar = calc_dir / "OUTCAR"
        if outcar.exists():
            for line in outcar.read_text().splitlines():
                if "vasp." in line.lower() and "build" in line.lower():
                    code_version = line.strip()
                    break
    elif dft_code == "QE":
        pwo_files = list(calc_dir.glob("*.pwo"))
        if pwo_files:
            for line in pwo_files[0].read_text().splitlines()[:20]:
                if "Program PWSCF" in line:
                    code_version = line.strip()
                    break

    # Hash all input files
    input_checksums = {}
    input_extensions = {"VASP": ["INCAR", "POSCAR", "KPOINTS"],
                       "QE": ["*.pwi"]}
    for pattern in input_extensions.get(dft_code, []):
        for f in list(calc_dir.glob(pattern)) + [calc_dir / pattern]:
            if Path(f).exists():
                input_checksums[Path(f).name] = sha256_file(f)

    # Hash output files
    output_checksums = {}
    output_patterns = {"VASP": ["vasprun.xml", "OUTCAR"],
                       "QE": ["*.pwo"]}
    for pattern in output_patterns.get(dft_code, []):
        for f in calc_dir.glob(pattern):
            output_checksums[f.name] = sha256_file(f)

    provenance = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "calc_dir": str(calc_dir),
        "config_type": config_type,
        "al_iteration": al_iteration,
        "dft_code": dft_code,
        "code_version": code_version,
        "settings_hash": settings_hash,
        "hpc_job_id": hpc_job_id,
        "input_checksums": input_checksums,
        "output_checksums": output_checksums,
    }

    sidecar_path = calc_dir / "provenance.json"
    sidecar_path.write_text(json.dumps(provenance, indent=2))


def verify_dft_provenance(calc_dir: Path, expected_settings_hash: str) -> bool:
    """
    Verify that a completed DFT calculation matches the expected settings hash.
    Returns True if consistent, False if the hash has changed.
    Prints a diagnostic message on failure.
    """
    sidecar = calc_dir / "provenance.json"
    if not sidecar.exists():
        print(f"WARNING {calc_dir.name}: no provenance.json — run record_dft_provenance")
        return False

    prov = json.loads(sidecar.read_text())
    actual_hash = prov.get("settings_hash")

    if actual_hash != expected_settings_hash:
        print(f"FAIL {calc_dir.name}: settings hash mismatch.")
        print(f"  Expected: {expected_settings_hash}")
        print(f"  Actual:   {actual_hash}")
        print(f"  This calculation was run with different DFT settings.")
        print(f"  Do NOT include it in the training dataset.")
        return False

    # Verify output file checksums have not changed since provenance was written
    for fname, recorded_hash in prov.get("output_checksums", {}).items():
        fpath = calc_dir / fname
        if fpath.exists():
            current_hash = sha256_file(fpath)
            if current_hash != recorded_hash:
                print(f"WARNING {calc_dir.name}/{fname}: file modified after provenance was recorded")
                return False

    return True
```

### Workflow 3: MLP training experiment tracking with MLflow

Track every MACE training run as an MLflow experiment. This enables comparison across runs, model registry management, and reproducibility of the best model.

```python
import mlflow
import mlflow.pytorch
import hashlib
import json
import subprocess
from pathlib import Path


def sha256_file(path) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def git_commit_hash() -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"],
                            capture_output=True, text=True)
    return result.stdout.strip()


def run_mace_with_tracking(
    train_file: str,
    val_file: str,
    test_file: str,
    mace_kwargs: dict,
    experiment_name: str = "mace_training",
    mlflow_tracking_uri: str = "logs/mlflow",
) -> str:
    """
    Launch a MACE training run tracked by MLflow.
    Returns the MLflow run_id for the completed run.
    """
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        print(f"MLflow run ID: {run_id}")

        # ─── Log dataset provenance ───────────────────────────────────────────
        for label, fpath in [("train", train_file), ("val", val_file), ("test", test_file)]:
            mlflow.log_param(f"data_{label}_file", fpath)
            mlflow.log_param(f"data_{label}_sha256", sha256_file(fpath))

        # ─── Log code version ─────────────────────────────────────────────────
        mlflow.log_param("git_commit", git_commit_hash())
        try:
            import mace
            mlflow.log_param("mace_version", mace.__version__)
        except AttributeError:
            pass

        try:
            import torch
            mlflow.log_param("torch_version", torch.__version__)
            mlflow.log_param("cuda_available", torch.cuda.is_available())
            if torch.cuda.is_available():
                mlflow.log_param("cuda_version", torch.version.cuda)
                mlflow.log_param("gpu_name", torch.cuda.get_device_name(0))
        except ImportError:
            pass

        # ─── Log all hyperparameters ──────────────────────────────────────────
        for k, v in mace_kwargs.items():
            mlflow.log_param(k, v)

        # ─── Build and run MACE training command ──────────────────────────────
        import subprocess
        cmd = ["mace_run_train",
               f"--train_file={train_file}",
               f"--valid_file={val_file}",
               f"--name={mace_kwargs.get('name', 'mace_model')}",
               ]
        for k, v in mace_kwargs.items():
            if k != "name":
                cmd.append(f"--{k}={v}")

        # Write training command to artifact for exact reproducibility
        cmd_file = Path(f"run_{run_id}_command.sh")
        cmd_file.write_text("#!/bin/bash\n" + " \\\n    ".join(cmd) + "\n")
        mlflow.log_artifact(str(cmd_file))

        # ─── Parse training log and log metrics per epoch ────────────────────
        log_file = Path(f"{mace_kwargs.get('name', 'mace_model')}.log")
        if log_file.exists():
            with open(log_file) as f:
                for line in f:
                    try:
                        d = json.loads(line)
                        epoch = d.get("epoch", 0)
                        if "val_forces_mae" in d:
                            mlflow.log_metric("val_force_mae_meV_A",
                                              d["val_forces_mae"] * 1000, step=epoch)
                        if "val_energy_mae" in d:
                            mlflow.log_metric("val_energy_mae_meV_atom",
                                              d["val_energy_mae"] * 1000, step=epoch)
                        if "val_stress_mae" in d:
                            mlflow.log_metric("val_stress_mae_GPa",
                                              d["val_stress_mae"] * 160.218, step=epoch)
                    except (json.JSONDecodeError, KeyError):
                        pass

        # ─── Archive model checkpoint ─────────────────────────────────────────
        model_name = mace_kwargs.get("name", "mace_model")
        model_file = Path(f"{model_name}_stagetwo.model")
        if model_file.exists():
            model_hash = sha256_file(model_file)
            mlflow.log_param("model_sha256", model_hash)
            mlflow.log_artifact(str(model_file))
            print(f"Model archived: {model_file} (sha256={model_hash[:16]}...)")

    return run_id


def register_model(
    run_id: str,
    model_artifact_path: str,
    model_name: str,
    stage: str = "Staging",
    mlflow_tracking_uri: str = "logs/mlflow",
) -> None:
    """
    Register a trained model in the MLflow model registry.
    stage: 'Staging' (not yet validated) or 'Production' (passed mlp-validation).
    """
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    model_uri = f"runs:/{run_id}/{model_artifact_path}"
    result = mlflow.register_model(model_uri, model_name)
    client = mlflow.MlflowClient()
    client.transition_model_version_stage(
        name=model_name,
        version=result.version,
        stage=stage,
    )
    print(f"Registered model '{model_name}' version {result.version} → {stage}")
    print(f"MLflow run: {run_id}")
```

### Workflow 4: Environment capture and snapshot

A computation cannot be reproduced without knowing the exact software environment. Record Python packages, compilers, CUDA, MPI, and HPC modules at run time.

```python
import subprocess
import json
import sys
import platform
from datetime import datetime
from pathlib import Path


def capture_environment(output_file: str = "environment_snapshot.json") -> dict:
    """
    Capture all software versions relevant to a computational materials science run.
    Write to a JSON sidecar. Call once at the start of a campaign and once before
    each production LAMMPS run.
    """
    env = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "platform": platform.platform(),
        "python_version": sys.version,
    }

    # ─── Python packages ──────────────────────────────────────────────────────
    pip_result = subprocess.run(["pip", "list", "--format=json"],
                                capture_output=True, text=True)
    if pip_result.returncode == 0:
        env["pip_packages"] = json.loads(pip_result.stdout)

    # ─── Conda environment ────────────────────────────────────────────────────
    conda_result = subprocess.run(
        ["conda", "env", "export", "--no-builds"],
        capture_output=True, text=True,
    )
    if conda_result.returncode == 0:
        env["conda_env_yaml"] = conda_result.stdout
        # Also write a standalone environment.yml
        Path("environment.yml").write_text(conda_result.stdout)

    # ─── CUDA and GPU ─────────────────────────────────────────────────────────
    nvcc_result = subprocess.run(["nvcc", "--version"], capture_output=True, text=True)
    env["cuda_version"] = nvcc_result.stdout.strip() if nvcc_result.returncode == 0 else None

    nvidia_result = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                                    "--format=csv,noheader"],
                                   capture_output=True, text=True)
    env["gpu_info"] = nvidia_result.stdout.strip() if nvidia_result.returncode == 0 else None

    # ─── Compiler versions ────────────────────────────────────────────────────
    for compiler in ["gcc", "g++", "mpicc", "mpicxx", "icc"]:
        result = subprocess.run([compiler, "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            env[f"{compiler}_version"] = result.stdout.splitlines()[0]

    # ─── LAMMPS version ───────────────────────────────────────────────────────
    lmp_result = subprocess.run(["lmp", "-help"], capture_output=True, text=True)
    if lmp_result.returncode == 0:
        for line in lmp_result.stdout.splitlines()[:5]:
            if "LAMMPS" in line and ("version" in line.lower() or "/" in line):
                env["lammps_version"] = line.strip()
                break

    # ─── HPC module list (SLURM/Lmod environments) ────────────────────────────
    module_result = subprocess.run(["module", "list"], capture_output=True, text=True)
    if module_result.returncode == 0:
        env["hpc_modules"] = module_result.stderr  # module list writes to stderr

    Path(output_file).write_text(json.dumps(env, indent=2))
    print(f"Environment snapshot written to {output_file}")
    return env


def write_requirements_file(output_file: str = "requirements.txt") -> None:
    """Freeze current pip environment to requirements.txt with pinned versions."""
    result = subprocess.run(["pip", "freeze"], capture_output=True, text=True)
    if result.returncode == 0:
        Path(output_file).write_text(result.stdout)
        print(f"Requirements written to {output_file} ({len(result.stdout.splitlines())} packages)")
    else:
        print(f"WARNING: pip freeze failed: {result.stderr}")


def compare_environments(env_file_a: str, env_file_b: str) -> list[str]:
    """
    Compare two environment snapshots and report differences.
    Useful for diagnosing why results differ between two machines or time points.
    """
    a = json.loads(Path(env_file_a).read_text())
    b = json.loads(Path(env_file_b).read_text())

    diffs = []

    # Compare key versions
    for key in ["python_version", "cuda_version", "lammps_version"]:
        va, vb = a.get(key), b.get(key)
        if va != vb:
            diffs.append(f"{key}: '{va}' vs '{vb}'")

    # Compare Python packages
    if "pip_packages" in a and "pip_packages" in b:
        pa = {p["name"]: p["version"] for p in a["pip_packages"]}
        pb = {p["name"]: p["version"] for p in b["pip_packages"]}
        all_pkgs = set(pa) | set(pb)
        for pkg in sorted(all_pkgs):
            va, vb = pa.get(pkg, "NOT INSTALLED"), pb.get(pkg, "NOT INSTALLED")
            if va != vb:
                diffs.append(f"  {pkg}: {va} vs {vb}")

    if diffs:
        print(f"Environment differences ({env_file_a} vs {env_file_b}):")
        for d in diffs[:30]:
            print(f"  {d}")
    else:
        print("Environments are equivalent (checked Python + packages + CUDA + LAMMPS)")

    return diffs
```

### Workflow 5: Dataset integrity verification and hash registry

Build and verify a hash registry for all datasets, model files, and critical input files. Detect silent corruption or accidental modification.

```python
import hashlib
import json
from pathlib import Path
from datetime import datetime


def sha256_file(path) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def sha256_extxyz(path: str) -> dict:
    """
    Compute content hash of an extxyz file plus structural statistics.
    Structural statistics catch corruption that preserves the byte count
    (e.g., frames reordered or a frame silently replaced).
    """
    from ase.io import read
    import numpy as np

    file_hash = sha256_file(path)
    try:
        frames = read(path, index=":", format="extxyz")
        energies = [f.info.get("energy", 0.0) for f in frames]
        forces_all = [f.arrays.get("forces") for f in frames if f.arrays.get("forces") is not None]
        force_means = [float(np.abs(ff).mean()) for ff in forces_all]

        stats = {
            "n_frames": len(frames),
            "n_atoms_total": sum(len(f) for f in frames),
            "energy_sum": float(sum(e for e in energies if e is not None)),
            "force_mean_mean": float(np.mean(force_means)) if force_means else None,
        }
    except Exception as e:
        stats = {"parse_error": str(e)}

    return {"file_sha256": file_hash, "stats": stats}


class HashRegistry:
    """
    Maintains a registry of file checksums for datasets, models, and inputs.
    Write to disk and commit to Git alongside DVC pointer files.
    """

    def __init__(self, registry_file: str = "hash_registry.json"):
        self.registry_file = Path(registry_file)
        if self.registry_file.exists():
            self.data = json.loads(self.registry_file.read_text())
        else:
            self.data = {
                "created": datetime.utcnow().isoformat() + "Z",
                "entries": {},
            }

    def register(self, label: str, path: str, description: str = "") -> str:
        """Add or update a file in the registry. Returns its SHA-256 hash."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Cannot register non-existent file: {path}")

        h = sha256_file(path)
        size = p.stat().st_size

        if label in self.data["entries"]:
            old = self.data["entries"][label]
            if old["sha256"] != h:
                print(f"WARNING: {label} hash changed!")
                print(f"  Old: {old['sha256'][:16]}... (registered {old['timestamp']})")
                print(f"  New: {h[:16]}... (current)")
                print(f"  If this is intentional (new dataset version), this is expected.")
                print(f"  If unintentional (file was accidentally modified), INVESTIGATE.")

        entry = {
            "path": str(p.resolve()),
            "sha256": h,
            "size_bytes": size,
            "description": description,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
        # For extxyz files, add structural statistics for content verification
        if path.endswith(".xyz") or path.endswith(".extxyz"):
            entry["extxyz_stats"] = sha256_extxyz(path)

        self.data["entries"][label] = entry
        self._save()
        print(f"Registered: {label} → sha256={h[:16]}... ({size:,} bytes)")
        return h

    def verify(self, label: str) -> bool:
        """Verify that a registered file has not changed since registration."""
        if label not in self.data["entries"]:
            print(f"ERROR: '{label}' not found in registry")
            return False

        entry = self.data["entries"][label]
        p = Path(entry["path"])

        if not p.exists():
            print(f"FAIL {label}: file not found at {entry['path']}")
            return False

        current_hash = sha256_file(entry["path"])
        if current_hash != entry["sha256"]:
            print(f"FAIL {label}: hash mismatch")
            print(f"  Registered: {entry['sha256'][:32]}...")
            print(f"  Current:    {current_hash[:32]}...")
            return False

        print(f"PASS {label}: hash verified ({entry['sha256'][:16]}...)")
        return True

    def verify_all(self) -> dict[str, bool]:
        """Verify all registered files. Returns {label: pass/fail}."""
        results = {}
        for label in self.data["entries"]:
            results[label] = self.verify(label)
        n_pass = sum(results.values())
        print(f"\nVerification: {n_pass}/{len(results)} passed")
        return results

    def report(self) -> None:
        """Print a summary of all registered files."""
        print(f"\nHash registry: {self.registry_file}")
        print(f"  Created: {self.data['created']}")
        print(f"  Entries: {len(self.data['entries'])}")
        print(f"\n  {'Label':<35}  {'SHA-256 (first 16)':>20}  {'Size':>12}")
        print("  " + "─" * 72)
        for label, entry in sorted(self.data["entries"].items()):
            size_str = f"{entry['size_bytes']:,}"
            print(f"  {label:<35}  {entry['sha256'][:16]:>20}  {size_str:>12}")

    def _save(self):
        self.registry_file.write_text(json.dumps(self.data, indent=2))
```

### Workflow 6: Active learning campaign provenance chain

An active learning campaign spans multiple iterations, each adding DFT structures and retraining the model. Maintaining an unambiguous connection between model version N and the DFT batch that produced it is the central provenance challenge.

```python
import hashlib
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path


@dataclass
class ALIterationRecord:
    """Immutable record for one active learning iteration."""
    iteration: int
    timestamp: str
    # Dataset state entering this iteration
    train_file: str
    train_file_sha256: str
    val_file: str
    val_file_sha256: str
    test_file: str
    test_file_sha256: str
    n_train: int
    # Active learning query
    query_strategy: str            # e.g. "committee_force_std"
    sigma_low: float               # lower uncertainty threshold (eV/Å)
    sigma_high: float              # extrapolation threshold (eV/Å)
    n_candidates_explored: int
    n_uncertain_flagged: int       # σ_F > σ_low
    n_extrapolation_flagged: int   # σ_F > σ_high
    n_selected_for_dft: int
    # DFT oracle
    dft_settings_hash: str
    dft_code: str
    n_submitted: int
    n_converged: int
    n_failed: int
    # Retrained model
    model_name: str
    model_file: str
    model_sha256: str
    mlflow_run_id: str
    val_force_mae_meV_A: float
    val_energy_mae_meV_atom: float
    # Stopping criterion check
    extrapolation_fraction_pct: float  # % of exploration frames with σ_F > σ_high
    converged: bool
    notes: str = ""


class ALCampaignLog:
    """
    Tracks the full provenance chain of an active learning MLP campaign.
    One record per iteration; each record is immutable once written.
    """

    def __init__(self, log_file: str = "al_campaign_log.json"):
        self.log_file = Path(log_file)
        if self.log_file.exists():
            data = json.loads(self.log_file.read_text())
            self.iterations: list[ALIterationRecord] = [
                ALIterationRecord(**d) for d in data["iterations"]
            ]
        else:
            self.iterations = []

    def record_iteration(self, record: ALIterationRecord) -> None:
        """Append a completed iteration to the campaign log."""
        # Check for duplicate iteration numbers
        existing_iters = {r.iteration for r in self.iterations}
        if record.iteration in existing_iters:
            raise ValueError(
                f"Iteration {record.iteration} already exists in the log. "
                f"Each iteration must have a unique number."
            )
        self.iterations.append(record)
        self._save()
        print(f"AL log: recorded iteration {record.iteration}")
        print(f"  Training set: {record.n_train} structures")
        print(f"  Model: {record.model_file} (sha256={record.model_sha256[:16]}...)")
        print(f"  Val force MAE: {record.val_force_mae_meV_A:.1f} meV/Å")
        if record.converged:
            print(f"  *** CONVERGED *** (extrapolation fraction {record.extrapolation_fraction_pct:.2f}%)")

    def summary(self) -> None:
        """Print convergence history across all iterations."""
        print(f"\nAL Campaign Log: {self.log_file}")
        print(f"  {'Iter':>4}  {'N_train':>8}  {'Val F MAE (meV/Å)':>20}  "
              f"{'Extrap %':>10}  {'N_DFT_new':>10}")
        print("  " + "─" * 60)
        for r in self.iterations:
            extrap_str = f"{r.extrapolation_fraction_pct:.2f}%"
            conv_str = " ← CONVERGED" if r.converged else ""
            print(f"  {r.iteration:>4}  {r.n_train:>8}  "
                  f"{r.val_force_mae_meV_A:>20.1f}  "
                  f"{extrap_str:>10}  {r.n_converged:>10}{conv_str}")

    def trace_model_to_data(self, model_sha256_prefix: str) -> ALIterationRecord | None:
        """
        Given a model SHA-256 prefix, find the iteration that produced it.
        Use this to trace a production model back to its training data.
        """
        for r in self.iterations:
            if r.model_sha256.startswith(model_sha256_prefix):
                print(f"Model {model_sha256_prefix}... produced in iteration {r.iteration}")
                print(f"  Training data: {r.train_file} (sha256={r.train_file_sha256[:16]}...)")
                print(f"  DFT settings hash: {r.dft_settings_hash}")
                print(f"  MLflow run: {r.mlflow_run_id}")
                return r
        print(f"Model {model_sha256_prefix}... not found in campaign log")
        return None

    def _save(self):
        data = {"iterations": [asdict(r) for r in self.iterations]}
        self.log_file.write_text(json.dumps(data, indent=2))
```

## Common Failure Modes

| Failure | Symptom | Likely Cause | Diagnostic | Fix |
|---------|---------|--------------|------------|-----|
| Overwritten dataset | Training from "the latest version" of train.xyz produces different model each time; no record of which structures were in the dataset at a given model checkpoint | extxyz files written in-place without versioning; DVC not used for large dataset files; `write(output_file, frames)` overwrites prior iteration | Check `git log -- datasets/train.xyz`; if there is only one commit, the file was never tracked with history; run `dvc diff` | Use DVC to track all extxyz dataset files; name files with iteration suffix (`train_iter002.xyz`) rather than generic names; never use `write` to overwrite a file that is referenced by an existing model checkpoint |
| Untracked preprocessing | Results not reproducible from raw DFT outputs despite having all raw files; a filtering step (FPS, convergence check, unit conversion) was applied but not recorded | Preprocessing script run interactively in a notebook or terminal without being committed to Git; script modified between runs; `--seed` not set for FPS | Check `git log scripts/`; if the preprocessing script has uncommitted changes or is absent, the exact transformation is unknown; verify extxyz dataset frame count matches what the script would produce from the raw inputs | Commit all preprocessing scripts before running them; record the script's Git commit hash and all CLI arguments in the dataset sidecar; use DVC pipeline stages (`dvc.yaml`) to declare preprocessing as a tracked step |
| Mixed code versions across AL iterations | Training loss increases when adding DFT data from a later iteration; systematic energy offset between iteration 0 and iteration 3 structures | VASP was upgraded mid-campaign (different default POTCAR behavior or FFT grid precision); QE pseudopotential downloaded with a different version; Python environment updated mid-campaign | Compare VASP version string from OUTCAR across all iterations; compare POTCAR SHA-256 checksums across iteration directories; run `compare_environments` on snapshots from each iteration | Record code version and POTCAR checksums at every DFT batch (Workflow 2); enforce via `DatasetManifest.register_settings` hash check; if version mismatch found, recompute the affected structures |
| Missing pseudopotential hashes | POTCAR or UPF file silently updated; energy reference surface shifts by 0.01–0.1 eV/atom; apparent improvement in validation metrics that is actually a data artifact | POTCAR files are in a shared directory that was updated by a sysadmin; same filename but different content; no checksum was recorded at campaign start | Compute SHA-256 of every POTCAR or UPF file in the dataset directory; compare to the checksum stored in `campaign_settings.json`; if they differ, the reference energy surface is inconsistent | Record pseudopotential checksums in `campaign_settings.json` at campaign start (Workflow 1); verify checksums before each DFT batch submission; if a PP file was updated, recompute all structures that used the old version |
| Nondeterministic train/test split | Two colleagues rerunning the same split script on the same dataset produce different train/val/test assignments; test error is unreliable across re-runs | `random.shuffle` called without `random.seed(N)`; NumPy RNG initialized from system entropy; Pandas `DataFrame.sample` without `random_state` | Check the splitting script for any `random.seed` call; if absent, the split is nondeterministic; compare frame indices across two re-runs of the script | Add `random.seed(42)` (or equivalent `np.random.default_rng(42)`) at the top of all splitting scripts; log the seed in the dataset manifest; include the split as a DVC pipeline stage with the seed as a parameter |
| Broken relative paths after cluster migration | DVC pull or MLflow model load fails after moving the project to a new HPC cluster or storage path; model files not found despite being registered | DVC `.dvc` files or MLflow artifact URIs contain absolute paths hardcoded to the old file system; LAMMPS input scripts have `pair_coeff * * /old/path/model.pt` | Run `grep -r "/old/path" . --include="*.dvc" --include="*.json" --include="*.lmp"` to find hardcoded paths; inspect `.dvc/config` for absolute remote paths | Use relative paths in DVC pipeline stages and LAMMPS scripts; configure DVC remote using environment variables or per-user local config (`dvc remote add -d ... --local`); test the full pipeline on a fresh clone before campaign handoff |
| Stale model checkpoints — loss of the best model | After re-running training with updated data, the original production model file is gone; only the most recent training run is available | `mace_run_train` writes to a fixed output filename and overwrites previous runs if `--name` is the same; no versioned archive was made before re-running | Check MLflow artifacts and model registry; if no artifact was logged, the file is gone unless a backup exists elsewhere | Register every candidate model in MLflow before re-running training; use DVC to track model files alongside datasets (`dvc add models/model_v1.model`); never run training with the same `--name` as an existing production model without archiving first |
| Irreproducible figures | A figure in a paper or report cannot be exactly reproduced from the raw data; re-running the plotting script produces different axis ranges, colors, or statistical summaries | Plotting script reads data via glob pattern that changes as new runs are added; matplotlib random state not fixed for stochastic methods; data transformations embedded in the plotting script are not separately versioned | Run the plotting script in a clean environment with only the specified input files; compare output figures pixel-by-pixel using `imageio.v3.imread` and `np.allclose` | Pin all input files explicitly (no glob) in figure scripts; seed all random state; version the plotting script alongside the data it visualizes using DVC; log figures as MLflow artifacts immediately after generation |

## Best Practices

- **Set up Git + DVC before submitting the first DFT calculation.** Retroactively adding versioning to an existing dataset is possible but lossy — the history of how the dataset evolved is gone. The setup overhead is less than one DFT calculation's wall time.
- **Treat the DFT settings file (`campaign_settings.json`) as immutable once committed.** Print it, sign it, and enforce the hash check programmatically in every label collection script. A settings mismatch detected later requires recomputing the inconsistent structures, which is expensive.
- **Use explicit seeds everywhere: Python RNG, NumPy RNG, PyTorch RNG, and LAMMPS velocity initialization.** A missing seed is invisible — results appear correct until a collaborator or reviewer cannot reproduce them. Document all seeds in the campaign log.
- **Archive every candidate model file before re-running training** with the same model name. Storage is cheap; lost model weights require days of GPU time to recover. Use `dvc add` or MLflow artifact logging to create versioned archives.
- **Write provenance sidecars immediately after each computational step**, not after the campaign is complete. Retroactive provenance reconstruction from file timestamps and OUTCAR headers is unreliable and cannot recover information that was never recorded.
- **Track environment snapshots at campaign start and at every major software upgrade.** A `requirements.txt` or `environment.yml` is not sufficient if compiled binaries (LAMMPS, VASP, QE) or CUDA drivers are involved. Record full `pip list`, `conda env export`, and key binary versions.
- **Use content hashes (SHA-256), not timestamps, to verify file identity.** File timestamps are easily corrupted by `touch`, rsync, or cluster backup systems. A SHA-256 hash is unforgeable without modifying the file content.
- **Document the complete training command** (not just hyperparameters) as an artifact. `mace_run_train` has many default values that change between versions; recording only the non-default parameters in a README is insufficient for bit-exact reproduction.

## Integration with Other Skills

- **[dft-dataset-generation](../../electronic-structure/dft-dataset-generation/SKILL.md):** The `DatasetManifest` class in that skill's Workflow 6 is the domain-specific provenance layer for DFT datasets. This skill's `HashRegistry` and `record_campaign_settings` are the campaign-level layer above it. Both must be used together for complete DFT dataset provenance.
- **[dft-convergence](../../electronic-structure/dft-convergence/SKILL.md):** The ENCUT and k-point density established by that skill's convergence protocol must be frozen at campaign start and recorded in `campaign_settings.json`. Any convergence re-test that changes the settings hash invalidates all prior data.
- **[vasp-workflow](../../electronic-structure/vasp-workflow/SKILL.md):** VASP calculations require versioned POTCAR files (SHA-256), logged INCAR settings, and tracked vasprun.xml outputs. The DFT provenance capture in Workflow 2 above implements the recording layer for VASP. The POTCAR TITEL line and checksum from that skill's best practices are required fields in `campaign_settings.json`.
- **[quantum-espresso](../../electronic-structure/quantum-espresso/SKILL.md):** QE calculations require pseudopotential filename and checksum provenance (SSSP or PseudoDojo, specific version). The `degauss`-in-Ry distinction documented there is a canonical example of a silent settings error that only provenance tracking reveals.
- **[lammps-workflow](../../atomistic-md/lammps-workflow/SKILL.md):** LAMMPS production runs must record their input script Git commit hash, model file SHA-256, initial structure file, random seed for velocity initialization, and the LAMMPS version. The environment capture in Workflow 4 covers LAMMPS binary version; the input script belongs in Git.
- **[md-equilibration](../../atomistic-md/md-equilibration/SKILL.md):** Equilibration runs are often repeated when a new MLP model is trained. Without versioned model files and fixed equilibration seeds, two "identical" equilibration runs with different model checkpoints will produce different structures. Always checkpoint the model SHA-256 alongside the equilibration output.
- **[mace-training](../../ml-interatomic-potentials/mace-training/SKILL.md):** Workflow 3 above wraps `mace_run_train` with MLflow tracking. The `--seed` argument, the training data file checksum, the training command, and the model file checksum are the minimum required to reproduce a MACE training result. The best-practice note in that skill to record the MACE version and model checksum is implemented here via `run_mace_with_tracking`.
- **[active-learning-mlp](../../ml-interatomic-potentials/active-learning-mlp/SKILL.md):** The `ALCampaignLog` in Workflow 6 above is the provenance layer for the active learning loop from that skill. Each AL iteration record connects the committee uncertainty thresholds, the selected DFT structures, the DFT settings hash, and the resulting model version into a single auditable chain.
- **[mlp-validation](../../ml-interatomic-potentials/mlp-validation/SKILL.md):** Validation results (EOS errors, phonon errors, NVE drift) must be logged as MLflow metrics alongside the model they characterize. A model version can only be moved from `Staging` to `Production` in the MLflow model registry after all validation checks from that skill pass. The validation metrics are stored as MLflow run metadata and linked to the model checkpoint.
- **[mlp-to-lammps](../../ml-interatomic-potentials/mlp-to-lammps/SKILL.md):** Every LAMMPS deployment of an MLP should verify that the deployed model file's SHA-256 matches the registered production model's checksum in the MLflow model registry. The `validate_lammps_forces` function in that skill should record the model hash alongside the force comparison result.
- **[materials-featurization](../materials-featurization/SKILL.md):** Structural descriptors (SOAP, ACE, MACE latent features) used for diversity filtering and duplicate detection must be versioned alongside the dataset. The SOAP hyperparameters (r_cut, n_max, l_max, species) are part of the preprocessing provenance and must be recorded in the dataset sidecar.

## Key References

1. Wilkinson, M. D., et al. (2016). *The FAIR Guiding Principles for scientific data management and stewardship.* Scientific Data, 3, 160018. https://doi.org/10.1038/sdata.2016.18
2. Scheffler, M., et al. (2022). *FAIR data enabling new horizons for materials research.* Nature, 604, 635–642. https://doi.org/10.1038/s41586-022-04501-x
3. Huber, S. P., et al. (2021). *AiiDA 1.0: A scalable computational infrastructure for automated reproducible workflows and data provenance in computational science.* Scientific Data, 7, 300. https://doi.org/10.1038/s41597-020-00638-4
4. Pineau, J., et al. (2021). *Improving reproducibility in machine learning research (a report from the NeurIPS 2019 reproducibility program).* Journal of Machine Learning Research, 22(164), 1–20. https://jmlr.org/papers/v22/20-1364.html
5. Draxl, C., & Scheffler, M. (2019). *NOMAD: The FAIR concept for big data-driven materials science.* MRS Bulletin, 43(9), 676–682. https://doi.org/10.1557/mrs.2018.208
