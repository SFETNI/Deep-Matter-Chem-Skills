---
name: aiida-workflow
domain: electronic-structure
maturity: draft
tools:
  - AiiDA
  - aiida-quantumespresso
  - aiida-cp2k
  - Quantum ESPRESSO
  - pymatgen
  - ASE
tested_versions:
  AiiDA: "2.5.0"
  aiida-quantumespresso: "4.4.0"
  aiida-cp2k: "2.1.0"
  Quantum ESPRESSO: "7.3"
  pymatgen: "2024.11.13"
  ASE: "3.23.0"
requires_gpu: false
estimated_runtime: "hours-to-weeks (profile setup: hours; individual DFT jobs: minutes-to-days)"
---

# AiiDA Workflow

## Description

AiiDA (Automated Interactive Infrastructure and Database for Computational Science) is a Python-based, provenance-first workflow engine for computational materials science. It manages the complete lifecycle of calculations: input generation, remote HPC job submission via SSH, output retrieval and parsing, automatic error handling and restart, and immutable provenance recording in a directed acyclic graph (DAG) of data and process nodes. Invoke this skill when designing automated, reproducible DFT or AIMD workflows for long-term campaigns, when cross-institutional provenance is required, when workflows must survive HPC failures and restart automatically, or when producing data to be archived to external repositories (NOMAD, Materials Cloud) with machine-readable, verifiable calculation histories.

## Domain Context

AiiDA's central design decision is that **all data and all processes are nodes in a provenance graph**. Every calculation input, every output, and every transformation is stored in a relational database (PostgreSQL or SQLite) as an immutable node. Directed links between nodes record who created what: a `CalcJobNode` that consumed a `StructureData` input carries an `INPUT_CALC` link recording that dependency. The resulting graph is the provenance record. It is impossible to run a calculation and lose its inputs, because all inputs must be stored as nodes before the calculation starts.

This architecture differs fundamentally from atomate2/jobflow and from ad hoc SLURM scripts:

- **atomate2/jobflow:** Stores results as MongoDB documents; provenance is recorded at the job level, but the full input–output DAG is not rigidly enforced. Setup is faster. Preferred for rapid internal campaigns where throughput matters more than deep provenance or cross-institutional sharing.
- **Ad hoc SLURM scripts:** No provenance. Results live wherever files land on disk. Restart after failure requires manual intervention. Querying results across a campaign requires writing custom parsers.
- **AiiDA:** The graph is the primary artifact. Every calculation is traceable to its input structure, pseudopotentials, k-mesh, cutoff, and code version. The archive (`.aiida` format) is self-contained and importable on any AiiDA installation. The overhead is real: configuring profiles, computers, codes, and the daemon takes several hours for a new HPC resource.

**AiiDA Architecture:**

An AiiDA installation has three components:

1. **Storage backend:** PostgreSQL (for production) or SQLite (for local testing) stores node metadata; a Disk-Objectstore repository stores binary objects (input files, output files, large arrays). Together these are the AiiDA database.
2. **Daemon:** Persistent worker processes that pick up submitted calculations, submit them to the remote HPC scheduler, poll for completion, retrieve outputs, and advance WorkChain steps. AiiDA 2.x supports a lightweight daemon without RabbitMQ for single-user setups; multi-user production setups may use RabbitMQ as a message broker. [EXPERT REVIEW NEEDED: RabbitMQ requirement changed across AiiDA 2.x minor versions]
3. **Profile:** A named configuration binding a user to a specific storage backend and daemon. All `verdi` commands and `aiida.orm` API calls operate on the active profile. Multiple profiles can coexist on one machine for different projects.

**Provenance Graph Nodes and Links:**

AiiDA nodes fall into two categories:

- **Data nodes:** Immutable records of computational inputs and outputs. Key subtypes: `StructureData` (periodic crystal structure), `Dict` (JSON-serializable parameter dictionary), `KpointsData` (k-mesh or high-symmetry path), `UpfData` (pseudopotential file), `FolderData` (directory tree), `ArrayData` (numpy arrays), `TrajectoryData` (MD or relaxation trajectory), `BandsData` (band structure), and scalars `Int`/`Float`/`Bool`/`Str`.
- **Process nodes:** Records of computations. Subtypes: `CalcJobNode` (one external-code execution, e.g. `pw.x`), `WorkChainNode` (multi-step workflow), `CalcFunctionNode` (pure Python function decorated with `@calcfunction`), `WorkFunctionNode` (pure Python function calling CalcJobs).

Links between nodes record the data flow:
- `INPUT_CALC`: Data → CalcJob (this data was an input to this calculation)
- `CREATE`: CalcJob → Data (this calculation created this data)
- `INPUT_WORK`: Data → WorkChain
- `RETURN`: WorkChain → Data (this workflow returned this data as an output)
- `CALL_CALC`: WorkChain → CalcJob (this workflow called this calculation)
- `CALL_WORK`: WorkChain → WorkChain (nested workflows)

**CalcJob vs. WorkChain vs. CalcFunction:**

- **CalcJob:** Wraps a single external code execution. Defines how to write input files (`prepare_for_submission`), what resources to request, and how to parse outputs. Instantiated via a plugin (e.g., `aiida-quantumespresso`'s `PwCalculation`). No automatic error handling.
- **WorkChain:** A multi-step workflow containing CalcJobs, logic, and error handlers. Uses `outline()` to declare the step sequence and `spec.exit_code()` to define failure conditions. Checkpoints after each step, enabling automatic restart after daemon restarts or HPC failures. Always prefer WorkChain over raw CalcJob for production.
- **CalcFunction / WorkFunction:** Python-decorated functions (`@calcfunction`, `@workfunction`) that create provenance nodes for pure Python transformations. Use for data manipulation steps (structure symmetrization, featurization, unit conversion) that must appear in the provenance graph.

**ProcessBuilder:**

Every AiiDA process has a `ProcessBuilder` returned by `Process.get_builder()`. The builder provides attribute-style access to all input ports with type checking. `builder.structure = structure_node` stores the node reference and validates the type. The builder is inspectable — printing it shows all ports and their current values — and is the canonical documentation of a calculation setup.

**Exit Codes:**

WorkChains communicate success or failure via integer exit codes. Exit code 0 = success; non-zero = error or warning. By convention: 1–99 warnings, 100–199 input errors, 200–299 infrastructure errors (SSH timeout, no retrieved files), 300–399 solver errors (unconverged SCF, forces not converged). `PwBaseWorkChain` from `aiida-quantumespresso` implements automatic recovery for many common QE failure patterns before issuing a non-zero code.

## When to Use This Skill

- Long-term, cross-institutional, or publication-grade HT-DFT campaigns where every result must be auditable with full input–output linkage and verifiable provenance.
- Workflows that must survive HPC failures, wall-time kills, or network interruptions: AiiDA's WorkChain checkpointing and daemon restart logic handle these automatically.
- Multi-code campaigns (QE + CP2K + VASP) requiring a unified provenance graph across codes.
- When results must be shared via NOMAD, Materials Cloud, or AiiDA Archive — the `.aiida` archive format is self-contained and importable on any AiiDA installation.
- Phonon calculations using `aiida-quantumespresso`'s `PhononWorkChain` (dozens of DFPT or displaced-supercell CalcJobs benefit from automatic management and provenance).
- When a code-agnostic interface is needed: `aiida-common-workflows` provides `CommonRelaxWorkChain` for structure relaxation with QE, VASP, CP2K, or GPAW under a single unified API.

## When Not to Use This Skill

- **Short campaigns (fewer than ~50 calculations) on a single system:** Profile, computer, code, and daemon setup overhead is not recovered. Use ASE calculators with a SLURM array script instead.
- **When turnaround time is the primary constraint:** AiiDA's daemon polling interval and node-creation overhead add latency compared to direct SLURM submission. For urgent exploratory calculations, submit directly.
- **When the team cannot maintain a daemon and PostgreSQL server:** AiiDA requires persistent processes. HPC environments that disallow long-running processes on login nodes need a gateway machine or container setup. SQLite mode exists but does not support the daemon and serializes all work.
- **When atomate2 + MongoDB is already the group standard:** Migrating an atomate2 campaign midway to AiiDA is expensive. Use atomate2 for code that already uses it; use AiiDA for new campaigns where provenance depth justifies the setup.
- **Workflows requiring sub-minute task granularity:** The daemon polling interval (default every few seconds) and per-node database writes add overhead disproportionate to tasks completing in under a minute.

## Core Concepts

- **Profile:** A named AiiDA installation instance binding a PostgreSQL database, Disk-Objectstore repository, and daemon. Switch with `verdi profile setdefault <name>` or the `AIIDA_PROFILE` environment variable. AiiDA 2.x supports `core.psql_dos` (PostgreSQL, production) and `core.sqlite_dos` (SQLite, local testing).
- **Daemon:** Worker processes that execute the submission–poll–retrieve loop. Must be running for submitted calculations to advance. `verdi daemon start 2` launches 2 workers; `verdi daemon status` shows their state. WorkChain checkpoints survive daemon restarts; they do not survive PostgreSQL data loss.
- **verdi CLI:** The AiiDA command-line interface. Key commands: `verdi setup`/`verdi quicksetup` (profile creation), `verdi computer setup` + `verdi computer configure` (HPC resource registration), `verdi code create core.code.installed` (code registration), `verdi daemon start|stop|status|restart`, `verdi process list` (active and recent processes), `verdi node show <UUID|PK>` (node inspection), `verdi calcjob gotocomputer <PK>` (SSH to job directory), `verdi archive export` (provenance export).
- **Computer:** An AiiDA object representing a remote HPC resource. Stores hostname, SSH username, transport plugin, scheduler type, work directory, and default MPI command. Configure once per HPC system; test with `verdi computer test <name>`. Transport plugins: `core.local` (for local execution), `core.ssh` (for remote HPC via SSH).
- **Code:** An AiiDA object representing an executable installed on a Computer. Stores the absolute path to the binary, the AiiDA plugin entry point (e.g., `quantumespresso.pw`), and prepend/append shell text for module loading. A separate Code node is required for each executable–computer combination.
- **StructureData:** AiiDA's crystal structure data node. Stores unit cell vectors and atomic sites. Converts via `structure.get_ase()` (returns ASE `Atoms`) and `structure.get_pymatgen()` (returns pymatgen `Structure`). Constructed from `StructureData(ase=atoms)` or `StructureData(pymatgen=struct)`. Immutable once stored — modifying a structure requires creating a new node.
- **Dict:** AiiDA's dictionary node for parameters. Used for all DFT input parameters (QE namelists, CP2K input sections, VASP INCAR-equivalent dicts). Must be JSON-serializable. Nested dicts are allowed.
- **KpointsData:** Stores a k-mesh (Monkhorst-Pack), explicit k-point list with weights, or high-symmetry path. `kpoints.set_kpoints_mesh([6, 6, 6])` sets a 6×6×6 grid centered at Γ. `kpoints.set_kpoints_mesh_from_density(0.1)` generates a mesh automatically from a reciprocal-space density in Å⁻¹.
- **ProcessBuilder:** Interface for setting up a CalcJob or WorkChain. `Process.get_builder()` returns it; inputs are set as attributes with type validation. `submit(builder)` submits asynchronously to the daemon; `run(builder)` executes synchronously (blocking, no daemon needed, for testing).
- **WorkChain outline:** The step sequence, defined in `cls.define(spec)` via `spec.outline(cls.step1, if_(cls.condition)(cls.step2), cls.step3)`. Each step is a checkpoint — the WorkChain persists its context before each step, enabling daemon-restart recovery. `self.ctx` is the persistent context dictionary.
- **ExitCode:** An integer exit condition attached to a WorkChain class via `spec.exit_code(code, 'LABEL', message='...')`. Returned from a step as `return self.exit_codes.LABEL` to terminate the WorkChain with that status. Calling code queries `wc.exit_status` and `wc.exit_message`.
- **QueryBuilder:** AiiDA's graph query engine. Supports filtering by node type, attributes, extras, and link labels; joining across node types; and projecting specific fields. More powerful than MongoDB for graph traversal; comparable to SQL for scalar fields.
- **AiiDA Archive:** A portable `.aiida` zip file that contains all nodes and links reachable from a set of root nodes (traversing the provenance graph automatically). Created with `verdi archive export`; imported with `verdi archive import`. Used for sharing data with collaborators and uploading to NOMAD or Materials Cloud.

## Key Workflows

### Workflow 1: Profile setup, computer registration, code configuration, and daemon startup

Setting up AiiDA for a new project requires creating a profile, registering the remote HPC computer, configuring SSH transport, installing pseudopotentials, registering the DFT code, and starting the daemon. This is a one-time setup per project–system combination.

```bash
# ─── Step 1: Install AiiDA and plugins ──────────────────────────────────────
pip install "aiida-core[atomic_tools]"   # includes ASE and pymatgen interop
pip install aiida-quantumespresso         # PwCalculation, PwBaseWorkChain, etc.
pip install aiida-pseudo                  # pseudopotential family management
# pip install aiida-cp2k                 # optional; for CP2K workflows
# pip install aiida-vasp                 # optional; for VASP workflows [EXPERT REVIEW NEEDED: aiida-vasp AiiDA 2.x compatibility]

# ─── Step 2: Create a profile ───────────────────────────────────────────────
# Option A: quicksetup (interactive, creates PostgreSQL database automatically)
verdi quicksetup

# Option B: 'presto' for quick local SQLite testing (no daemon; no PostgreSQL)
verdi presto   # creates a default profile with SQLite backend; good for testing

# Option C: non-interactive setup (for CI or scripted deployment)
verdi setup \
  --non-interactive \
  --profile my_project \
  --email user@institution.edu \
  --first-name User \
  --last-name Name \
  --institution "My University" \
  --db-backend core.psql_dos \
  --db-host localhost \
  --db-port 5432 \
  --db-name aiida_my_project \
  --db-username aiida_user \
  --db-password "<DB_PASSWORD>" \
  --repository /data/aiida/my_project/repository

verdi profile setdefault my_project

# ─── Step 3: Register the remote HPC computer ───────────────────────────────
# Write a YAML config to avoid the interactive wizard for reproducible setup:
cat > hpc_cluster.yml << 'EOF'
---
label: myHPC
hostname: hpc.example.ac.uk
description: "Main HPC cluster"
transport: core.ssh
scheduler: core.slurm
work_dir: /scratch/{username}/aiida_workdir/
shebang: "#!/bin/bash"
mpirun_command: "srun -n {tot_num_mpiprocs}"
default_mpiprocs_per_machine: 32
prepend_text: |
  module purge
  module load QuantumESPRESSO/7.3-intel-2022a
append_text: ""
EOF
verdi computer setup --config hpc_cluster.yml

# Configure SSH transport (stores credentials in AiiDA's keyring, not in plain text)
verdi computer configure core.ssh myHPC
# Interactive prompts:
#   username:        your_hpc_login
#   key_filename:    /home/user/.ssh/id_rsa_hpc   (leave blank for SSH agent)
#   port:            22
#   proxy_jump:      login.internal.edu:22         (if a jump host is required)
#   timeout:         60

# Verify the computer connection (SSH + scheduler commands + work directory)
verdi computer test myHPC
# Expected:
#   * Testing SSH connection     ... OK
#   * Testing default shell      ... OK
#   * Testing scheduler commands ... OK
#   * Testing working directory  ... OK

# ─── Step 4: Install SSSP pseudopotential families ──────────────────────────
# SSSP is the recommended choice for QE; both tiers should be installed.
aiida-pseudo install sssp -v 1.3 -x PBE -p efficiency   # fast, suitable for structure/forces
aiida-pseudo install sssp -v 1.3 -x PBE -p precision    # higher cutoffs, for phonons/EOS
# Verify
aiida-pseudo list
# Expected: SsspFamily<SSSP/1.3/PBE/efficiency>, SsspFamily<SSSP/1.3/PBE/precision>

# ─── Step 5: Register the QE pw.x code ──────────────────────────────────────
verdi code create core.code.installed \
  --computer=myHPC \
  --label=pw-7.3 \
  --description="QE pw.x 7.3 on myHPC" \
  --default-calc-job-plugin=quantumespresso.pw \
  --filepath-executable=/apps/QuantumESPRESSO/7.3/bin/pw.x

# Register ph.x for DFPT phonons
verdi code create core.code.installed \
  --computer=myHPC \
  --label=ph-7.3 \
  --description="QE ph.x 7.3 on myHPC" \
  --default-calc-job-plugin=quantumespresso.ph \
  --filepath-executable=/apps/QuantumESPRESSO/7.3/bin/ph.x

verdi code list   # should show pw-7.3@myHPC and ph-7.3@myHPC

# ─── Step 6: Start the daemon ───────────────────────────────────────────────
verdi daemon start 2     # 2 worker processes; increase for large campaigns
verdi daemon status      # both workers should show as RUNNING
verdi process list       # shows submitted processes (empty initially)
```

### Workflow 2: Single QE SCF calculation as a CalcJob (ProcessBuilder)

A `PwCalculation` CalcJob wraps a single `pw.x` execution. All inputs are passed as AiiDA data nodes via a `ProcessBuilder`. This workflow shows the minimal complete setup; in production always wrap in a WorkChain (Workflow 3).

```python
# submit_pw_scf.py
# Submit a single pw.x SCF calculation.
# Requires: active AiiDA profile, myHPC computer, pw-7.3 code, SSSP installed.

import aiida
aiida.load_profile()  # loads the default profile; specify 'my_project' explicitly if needed

from aiida.engine import submit
from aiida.orm import Dict, KpointsData, StructureData, load_code
from aiida.plugins import CalculationFactory
from aiida_pseudo.groups.family import SsspFamily
from ase.build import bulk

# ─── Structure ───────────────────────────────────────────────────────────────
ase_si = bulk("Si", "diamond", a=5.43)
structure = StructureData(ase=ase_si)
structure.label = "Si_diamond_bulk"
structure.description = "Si diamond cubic, a=5.43 Å"
# structure.store() is called automatically by submit(); explicit store gives the PK sooner

# ─── pw.x input parameters (nested namelists as Python dict) ─────────────────
parameters = Dict({
    "CONTROL": {
        "calculation": "scf",
        "restart_mode": "from_scratch",
        "tprnfor": True,    # print forces
        "tstress": True,    # print stress tensor (needed for ML training data)
        "etot_conv_thr": 1.0e-6,   # Ry; energy convergence for geometry steps
    },
    "SYSTEM": {
        "ecutwfc": 50,             # Ry; wavefunction cutoff — converge explicitly
        "ecutrho": 200,            # Ry; 4x ecutwfc for NC pseudopotentials
        "occupations": "smearing",
        "smearing": "mp",          # Methfessel-Paxton; use 'cold' for insulators
        "degauss": 0.01,           # Ry; ~0.14 eV; set to 0.001-0.005 for insulators
    },
    "ELECTRONS": {
        "conv_thr": 1.0e-9,        # Ry; SCF convergence; tighten to 1e-10 for forces
        "mixing_beta": 0.7,
        "electron_maxstep": 100,
    },
})

# ─── k-points ────────────────────────────────────────────────────────────────
kpoints = KpointsData()
kpoints.set_kpoints_mesh([8, 8, 8], offset=[0, 0, 0])   # 8×8×8 MP grid, Γ-centered

# ─── Pseudopotentials from installed SSSP family ─────────────────────────────
sssp_family = SsspFamily.get("SSSP/1.3/PBE/efficiency")
pseudos = sssp_family.get_pseudos(structure=structure)   # returns {element: UpfData}

# ─── Build and submit ────────────────────────────────────────────────────────
PwCalculation = CalculationFactory("quantumespresso.pw")
builder = PwCalculation.get_builder()

builder.code        = load_code("pw-7.3@myHPC")
builder.structure   = structure
builder.parameters  = parameters
builder.kpoints     = kpoints
builder.pseudos     = pseudos

# SLURM resource requests
builder.metadata.options.resources = {
    "num_machines": 1,
    "num_mpiprocs_per_machine": 32,
}
builder.metadata.options.max_wallclock_seconds = 3600
builder.metadata.options.account     = "myallocation"
builder.metadata.options.queue_name  = "regular"
builder.metadata.options.scheduler_stderr = "_scheduler-stderr.txt"
builder.metadata.options.scheduler_stdout = "_scheduler-stdout.txt"

builder.metadata.label       = "Si_scf_sssp_efficiency_8x8x8"
builder.metadata.description = "Si SCF, SSSP efficiency, 8×8×8 k-mesh, ecutwfc=50 Ry"

calcjob = submit(builder)
print(f"Submitted PwCalculation PK={calcjob.pk}  UUID={calcjob.uuid}")
print(f"Monitor: verdi process show {calcjob.pk}")
print(f"Logs:    verdi calcjob logs {calcjob.pk}")

# ─── After completion (do not poll in production; use WorkChain callbacks) ───
from aiida.engine import wait_for
calcjob = wait_for(calcjob)   # blocking; only for interactive scripts

if calcjob.is_finished_ok:
    out = calcjob.outputs.output_parameters.get_dict()
    print(f"Total energy:    {out['energy']:.6f} eV")
    print(f"Volume:          {out.get('volume', 'N/A')} Å³")
    print(f"Band gap:        {out.get('bandgap', 'N/A')} eV")
else:
    print(f"FAILED: exit_status={calcjob.exit_status}, message={calcjob.exit_message}")
    print(f"  Go to remote dir: verdi calcjob gotocomputer {calcjob.pk}")
```

### Workflow 3: PwBaseWorkChain — automatic error handling and restart

`PwBaseWorkChain` wraps `PwCalculation` with automatic error detection and recovery. It handles common QE failures (wall-time expiry, electronic convergence failures, CRASH, bad parallelisation) by adjusting settings and restarting. Always use `PwBaseWorkChain` in production instead of raw `PwCalculation`.

```python
# submit_pw_relax_workchain.py
# Submit a PwBaseWorkChain for a geometry relaxation with automatic error handling.

import aiida
aiida.load_profile()

from aiida.engine import submit
from aiida.orm import Dict, Float, Int, Bool, KpointsData, StructureData, load_code
from aiida_quantumespresso.workflows.pw.base import PwBaseWorkChain
from aiida_pseudo.groups.family import SsspFamily
from pymatgen.core import Structure

pmg_struct = Structure.from_file("LiFePO4.cif")
structure = StructureData(pymatgen=pmg_struct)

pw_code    = load_code("pw-7.3@myHPC")
sssp_fam   = SsspFamily.get("SSSP/1.3/PBE/efficiency")
pseudos    = sssp_fam.get_pseudos(structure=structure)

parameters = Dict({
    "CONTROL": {
        "calculation": "vc-relax",   # variable-cell relaxation (positions + cell)
        "etot_conv_thr": 1.0e-6,
        "forc_conv_thr": 1.0e-4,     # a.u.; ~0.005 eV/Å
        "tprnfor": True,
        "tstress": True,
        "nstep": 300,
    },
    "SYSTEM": {
        "ecutwfc": 60,
        "ecutrho": 480,
        "occupations": "smearing",
        "smearing": "cold",
        "degauss": 0.002,    # Ry; narrow smearing for low-conductivity oxide
        "nspin": 2,
        "starting_magnetization(1)": 0.5,   # Fe starting moment; index 1 = first species
    },
    "ELECTRONS": {
        "conv_thr": 1.0e-9,
        "mixing_beta": 0.3,        # reduced mixing for magnetic systems
        "electron_maxstep": 200,
        "mixing_mode": "local-TF",
    },
    "IONS": {"ion_dynamics": "bfgs"},
    "CELL": {"cell_dynamics": "bfgs", "press_conv_thr": 0.5},
})

kpoints = KpointsData()
kpoints.set_kpoints_mesh([4, 4, 4])

builder = PwBaseWorkChain.get_builder()
builder.pw.code        = pw_code
builder.pw.structure   = structure
builder.pw.parameters  = parameters
builder.pw.kpoints     = kpoints
builder.pw.pseudos     = pseudos

# PwBaseWorkChain-specific settings
builder.max_iterations = Int(5)      # maximum restart attempts before issuing non-zero exit
builder.clean_workdir  = Bool(True)  # delete remote scratch after successful retrieval

# Resources applied to every PwCalculation spawned by this WorkChain
builder.pw.metadata.options.resources = {
    "num_machines": 2,
    "num_mpiprocs_per_machine": 32,
}
builder.pw.metadata.options.max_wallclock_seconds = 7200
builder.pw.metadata.options.account    = "myallocation"
builder.pw.metadata.options.queue_name = "regular"

builder.metadata.label       = "LiFePO4_vc-relax_PwBase"
builder.metadata.description = "LiFePO4 vc-relax, spin-polarized PBE, SSSP efficiency"

wc = submit(builder)
print(f"Submitted PwBaseWorkChain PK={wc.pk}  UUID={wc.uuid}")
print(f"Live output: verdi process report {wc.pk}")
print(f"Full graph:  verdi node graph generate {wc.pk}")
```

Inspecting WorkChain results after completion:

```python
from aiida.orm import load_node

wc = load_node(pk=<PK>)

print(f"State:        {wc.process_state.value}")
print(f"Exit status:  {wc.exit_status}")
print(f"Exit message: {wc.exit_message}")

if wc.is_finished_ok:
    out = wc.outputs.output_parameters.get_dict()
    n   = out["number_of_atoms"]
    print(f"Energy/atom:  {out['energy'] / n:.6f} eV/atom")
    print(f"Volume:       {out.get('volume', 'N/A')} Å³")
    # Relaxed structure (for vc-relax or relax)
    if hasattr(wc.outputs, "output_structure"):
        relaxed = wc.outputs.output_structure.get_pymatgen()
        print(f"Relaxed a/b/c: {relaxed.lattice.abc}")
else:
    # Trace the failing CalcJob inside the WorkChain
    for called in wc.called_descendants:
        if called.exit_status and called.exit_status != 0:
            print(f"  Failed node: PK={called.pk}, type={called.node_type}, "
                  f"exit={called.exit_status}: {called.exit_message}")
    print(f"  Remote files: verdi calcjob gotocomputer <failed_calcjob_PK>")
    print(f"  Full logs:    verdi process report {wc.pk}")
```

### Workflow 4: k-point convergence study using AiiDA and QueryBuilder

This workflow submits a series of SCF calculations at increasing k-mesh densities, tags them with extras for easy retrieval, and queries the provenance graph to build a convergence table after completion.

```python
# kpoints_convergence.py
# k-point convergence series for Cu FCC using PwBaseWorkChain + QueryBuilder.

import aiida
aiida.load_profile()

from aiida.engine import submit
from aiida.orm import Dict, Int, Bool, KpointsData, StructureData, load_code
from aiida_quantumespresso.workflows.pw.base import PwBaseWorkChain
from aiida_pseudo.groups.family import SsspFamily
from ase.build import bulk

pw_code   = load_code("pw-7.3@myHPC")
sssp_fam  = SsspFamily.get("SSSP/1.3/PBE/efficiency")

# Store structure once; all CalcJobs reference the same node → shared provenance link
structure = StructureData(ase=bulk("Cu", "fcc", a=3.63))
structure.label = "Cu_fcc_a363"
structure.store()
print(f"StructureData PK={structure.pk}  UUID={structure.uuid}")

pseudos = sssp_fam.get_pseudos(structure=structure)

base_parameters = Dict({
    "CONTROL": {"calculation": "scf", "tprnfor": True, "tstress": True},
    "SYSTEM": {
        "ecutwfc": 40,   # fixed cutoff; only k-mesh varies in this series
        "ecutrho": 320,
        "occupations": "smearing",
        "smearing": "mp",
        "degauss": 0.01,
    },
    "ELECTRONS": {"conv_thr": 1.0e-10, "mixing_beta": 0.7},
})

SERIES_TAG = "Cu_kpts_convergence_v1"
kmesh_list = [2, 4, 6, 8, 10, 12, 14, 16]
submitted_pks = []

for n in kmesh_list:
    kpoints = KpointsData()
    kpoints.set_kpoints_mesh([n, n, n])

    builder = PwBaseWorkChain.get_builder()
    builder.pw.code        = pw_code
    builder.pw.structure   = structure   # same stored node every time
    builder.pw.parameters  = base_parameters
    builder.pw.kpoints     = kpoints
    builder.pw.pseudos     = pseudos
    builder.max_iterations = Int(3)
    builder.clean_workdir  = Bool(True)
    builder.pw.metadata.options.resources = {
        "num_machines": 1, "num_mpiprocs_per_machine": 32
    }
    builder.pw.metadata.options.max_wallclock_seconds = 1800
    builder.metadata.label = f"Cu_scf_{n}x{n}x{n}"

    wc = submit(builder)
    wc.set_extra("kmesh_n",  n)
    wc.set_extra("series",   SERIES_TAG)
    submitted_pks.append(wc.pk)
    print(f"Submitted {n}×{n}×{n}: PK={wc.pk}")

print(f"\n{len(submitted_pks)} WorkChains submitted.")
print(f"Monitor: verdi process list -a -G {SERIES_TAG}")

# ─── Query results after completion ─────────────────────────────────────────
# (run separately after all WorkChains finish)
from aiida.orm import QueryBuilder, WorkChainNode, Dict as ADict
import numpy as np

qb = QueryBuilder()
qb.append(
    WorkChainNode,
    filters={
        "attributes.process_label": "PwBaseWorkChain",
        "extras.series": SERIES_TAG,
        "attributes.exit_status": 0,
    },
    project=["pk", "extras.kmesh_n"],
    tag="wc",
)
qb.append(
    ADict,
    with_incoming="wc",
    edge_filters={"label": "output_parameters"},
    project=["attributes.energy", "attributes.number_of_atoms"],
)

rows = sorted([(r[1], r[2], r[3]) for r in qb.all()], key=lambda x: x[0])

print(f"\n{'k-mesh':>10}  {'E_total (eV)':>14}  {'ΔE (meV/atom)':>16}")
for i, (n, energy, n_atoms) in enumerate(rows):
    if i == 0 or rows[i-1][1] is None:
        de_str = "---"
    else:
        de = (energy - rows[i-1][1]) / n_atoms * 1000
        de_str = f"{de:+.3f}"
    print(f"{n:>3}×{n:>3}×{n:>3}  {energy:>14.6f}  {de_str:>16}")

# Accept the smallest k-mesh where |ΔE/atom| < 1 meV/atom.
```

### Workflow 5: QueryBuilder — extracting results and exporting provenance

QueryBuilder is AiiDA's graph traversal and query engine. It replaces MongoDB queries and direct database introspection for extracting completed-calculation results, quality-checking a campaign, and preparing data for export.

```python
# query_and_export.py
import aiida
aiida.load_profile()

from aiida.orm import QueryBuilder, WorkChainNode, Dict, StructureData, CalcJobNode
from aiida.orm import load_node, Group
import pandas as pd
import numpy as np

# ─── Example 1: Completed relaxations — energy and volume table ──────────────
qb = QueryBuilder()
qb.append(
    WorkChainNode,
    filters={
        "attributes.process_label": "PwBaseWorkChain",
        "attributes.process_state": "finished",
        "attributes.exit_status": 0,
    },
    project=["pk", "uuid", "label", "ctime"],
    tag="wc",
)
qb.append(
    Dict,
    with_incoming="wc",
    edge_filters={"label": "output_parameters"},
    project=["attributes.energy", "attributes.number_of_atoms", "attributes.volume"],
)
qb.order_by({"wc": {"ctime": "asc"}})

rows = []
for pk, uuid, label, ctime, energy, n_atoms, volume in qb.all():
    if energy is None or n_atoms is None:
        continue
    rows.append({
        "pk": pk, "uuid": uuid, "label": label,
        "energy_eV": energy,
        "energy_per_atom_eV": energy / n_atoms,
        "volume_A3_per_atom": (volume / n_atoms) if volume else None,
        "n_atoms": n_atoms,
    })

df = pd.DataFrame(rows)
print(f"Completed PwBaseWorkChains: {len(df)}")
print(df[["label", "energy_per_atom_eV", "n_atoms"]].to_string())

# ─── Example 2: Trace a structure to all calculations that used it ───────────
target_pk = 123   # StructureData PK
qb2 = QueryBuilder()
qb2.append(StructureData, filters={"pk": target_pk}, tag="struct")
qb2.append(
    CalcJobNode,
    with_incoming="struct",
    project=["pk", "uuid", "label", "attributes.process_state", "attributes.exit_status"],
)
calcs = qb2.all()
print(f"\nCalcJobs using StructureData PK={target_pk}: {len(calcs)}")
for c in calcs:
    print(f"  PK={c[0]}, label={c[2]}, state={c[3]}, exit={c[4]}")

# ─── Example 3: Campaign quality check — identify failed WorkChains ──────────
qb3 = QueryBuilder()
qb3.append(
    WorkChainNode,
    filters={
        "attributes.process_state": "finished",
        "attributes.exit_status": {"!=": 0},
        "extras.campaign": "oxide_screening_2024_PBE",
    },
    project=["pk", "extras.formula", "attributes.exit_status", "attributes.exit_message"],
)
qb3.order_by({"WorkChainNode": {"ctime": "desc"}})

failed = qb3.all()
print(f"\nFailed WorkChains in oxide campaign: {len(failed)}")
for pk, formula, code, msg in failed:
    print(f"  {formula:25s}  exit={code}  msg={msg}")
    print(f"    verdi process report {pk}")

# ─── Example 4: Export a campaign group to AiiDA Archive ──────────────────
# Archive includes all nodes reachable in the provenance graph from root nodes.
# Suitable for NOMAD upload, Materials Cloud sharing, or long-term archiving.
print("\nExport commands:")
print("  verdi archive export -O oxide_campaign_v1.aiida -G 'oxide_screening_2024_PBE'")
print("  # Import on another machine:")
print("  verdi archive import oxide_campaign_v1.aiida")
print("  # Upload to Materials Cloud via mc-tools or the web interface")

# ─── Example 5: Extract DFT data to extxyz for ML potential training ─────────
# Each completed CalcJob with forces becomes one frame in the training set.
from aiida.orm import CalcJobNode as CJN
import numpy as np
from ase.io import write

qb5 = QueryBuilder()
qb5.append(
    CJN,
    filters={
        "attributes.process_label": "PwCalculation",
        "attributes.process_state": "finished",
        "attributes.exit_status": 0,
        "extras.campaign": "oxide_screening_2024_PBE",
    },
    project=["pk", "uuid"],
    tag="cj",
)
qb5.append(
    Dict,
    with_incoming="cj",
    edge_filters={"label": "output_parameters"},
    project=["attributes.energy", "attributes.forces"],
    tag="out",
)
qb5.append(
    StructureData,
    with_outgoing="cj",
    edge_filters={"label": "structure"},
    project=["*"],
    tag="struct",
)

frames = []
for pk, uuid, energy, forces, struct_node in qb5.all():
    if energy is None or forces is None:
        continue
    atoms = struct_node.get_ase()
    atoms.info["energy"]   = energy
    atoms.info["aiida_pk"] = pk
    atoms.info["aiida_uuid"] = str(uuid)
    atoms.arrays["forces"] = np.array(forces)
    frames.append(atoms)

write("oxide_campaign_training.xyz", frames, format="extxyz")
print(f"\nExported {len(frames)} frames to oxide_campaign_training.xyz")
```

### Workflow 6: High-throughput relaxation campaign

A production HT-DFT campaign submits hundreds of structures as `PwRelaxWorkChain` jobs, organises them in an AiiDA Group for bulk monitoring and export, and restarts idempotently (skipping already-submitted structures on re-run).

```python
# ht_campaign_submit.py
import aiida
aiida.load_profile()

from pathlib import Path
from aiida.engine import submit
from aiida.orm import Dict, Float, Int, Bool, KpointsData, StructureData, load_code, Group
from aiida_quantumespresso.workflows.pw.relax import PwRelaxWorkChain
from aiida_pseudo.groups.family import SsspFamily
from pymatgen.core import Structure

CAMPAIGN_LABEL = "oxide_screening_2024_PBE"
PW_CODE_LABEL  = "pw-7.3@myHPC"
SSSP_LABEL     = "SSSP/1.3/PBE/efficiency"
STRUCTS_DIR    = Path("structures/oxide_candidates/")
KPOINTS_DIST   = 0.15    # Å⁻¹; automatic mesh from reciprocal-space density

pw_code     = load_code(PW_CODE_LABEL)
sssp_family = SsspFamily.get(SSSP_LABEL)

# Create or reuse campaign group (idempotent: safe to re-run if campaign is interrupted)
campaign_group, created = Group.objects.get_or_create(label=CAMPAIGN_LABEL)
print(f"{'Created' if created else 'Loaded'} group '{CAMPAIGN_LABEL}' "
      f"({len(campaign_group.nodes)} nodes)")

# Shared parameters (commit to campaign_settings.json alongside this script)
base_params = Dict({
    "CONTROL": {
        "calculation": "vc-relax",
        "etot_conv_thr": 1.0e-6,
        "forc_conv_thr": 1.0e-4,
        "tprnfor": True, "tstress": True,
    },
    "SYSTEM": {
        "ecutwfc": 60, "ecutrho": 480,
        "occupations": "smearing", "smearing": "cold", "degauss": 0.002,
    },
    "ELECTRONS": {"conv_thr": 1.0e-10, "mixing_beta": 0.4},
    "IONS":      {"ion_dynamics": "bfgs"},
    "CELL":      {"cell_dynamics": "bfgs", "press_conv_thr": 0.5},
})

submitted, skipped, failed = [], [], []

for cif_path in sorted(STRUCTS_DIR.glob("*.cif")):
    formula = cif_path.stem

    # Idempotency check: skip if this formula is already a WorkChainNode in the group
    already_submitted = any(
        n.node_type == "process.workflow.workchain.WorkChainNode."
        and n.label.startswith(formula)
        for n in campaign_group.nodes
    )
    if already_submitted:
        skipped.append(formula)
        continue

    try:
        pmg_struct = Structure.from_file(str(cif_path))
    except Exception as e:
        print(f"  SKIP {formula}: read error ({e})")
        failed.append(formula)
        continue

    # Pseudopotential availability check before submitting
    elements = {str(s.specie) for s in pmg_struct.sites}
    try:
        pseudos = sssp_family.get_pseudos(elements=elements)
    except Exception as e:
        print(f"  SKIP {formula}: missing PP for elements {elements} ({e})")
        failed.append(formula)
        continue

    structure = StructureData(pymatgen=pmg_struct)
    structure.label = formula
    structure.store()

    # Automatic k-mesh from density
    kpoints = KpointsData()
    kpoints.set_cell_from_structure(structure)
    kpoints.set_kpoints_mesh_from_density(KPOINTS_DIST)

    builder = PwRelaxWorkChain.get_builder()
    builder.structure            = structure
    builder.base.pw.code        = pw_code
    builder.base.pw.pseudos     = pseudos
    builder.base.pw.parameters  = base_params
    builder.base.pw.kpoints     = kpoints
    builder.base.max_iterations = Int(5)
    builder.base.clean_workdir  = Bool(True)
    builder.base.pw.metadata.options.resources = {
        "num_machines": 1, "num_mpiprocs_per_machine": 32
    }
    builder.base.pw.metadata.options.max_wallclock_seconds = 7200
    builder.base.pw.metadata.options.account    = "myallocation"
    builder.base.pw.metadata.options.queue_name = "regular"
    builder.metadata.label       = f"{formula}_vc-relax"
    builder.metadata.description = (
        f"{formula} PBE vc-relax, SSSP efficiency, kpts_dist={KPOINTS_DIST}"
    )

    wc = submit(builder)
    wc.set_extra("formula",  formula)
    wc.set_extra("campaign", CAMPAIGN_LABEL)
    campaign_group.add_nodes([structure, wc])
    submitted.append((formula, wc.pk))
    print(f"  Submitted {formula}: PK={wc.pk}")

print(f"\nSubmitted: {len(submitted)} | Skipped: {len(skipped)} | Failed: {len(failed)}")
print(f"Monitor:  verdi process list --all --group '{CAMPAIGN_LABEL}'")
print(f"Progress: verdi process list --all --group '{CAMPAIGN_LABEL}' --raw | sort | uniq -c")
```

Campaign monitoring and harvest:

```bash
# Progress overview (count by state)
verdi process list --all --group 'oxide_screening_2024_PBE' --raw \
  | awk '{print $NF}' | sort | uniq -c

# Inspect a specific failed WorkChain
verdi process report <PK>

# Kill a stuck job (e.g., caught in an infinite restart loop)
verdi process kill <PK>

# Export the completed campaign for archiving and sharing
verdi archive export \
  --output oxide_screening_2024_PBE.aiida \
  --group 'oxide_screening_2024_PBE'

# Import on a collaborator's machine or for NOMAD upload
verdi archive import oxide_screening_2024_PBE.aiida

# Clean up stale nodes and reclaim disk (run periodically; safe)
verdi storage maintain
```

## Common Failure Modes

| Failure | Symptom | Likely Cause | Diagnostic | Fix |
|---------|---------|--------------|------------|-----|
| Broken daemon | All submitted WorkChains stay `CREATED` or `WAITING` indefinitely; `verdi process list` shows no `RUNNING` entries | Daemon process killed (HPC login-node reboot, OOM); RabbitMQ broker stopped; daemon was never started | `verdi daemon status` — if workers are `STOPPED` or absent; check `~/.aiida/daemon/log/*.log` for crash trace | `verdi daemon restart`; if that hangs: `verdi daemon stop --no-wait && verdi daemon start 2`; if PostgreSQL is unreachable, restore from backup |
| Wrong code or computer configuration | CalcJob fails immediately with `SUBMIT_FAILED`; no SLURM job appears; SSH errors in `_scheduler-stderr.txt` | Wrong executable path in Code node; SSH key not loaded; HPC hostname changed; SLURM account expired or partition renamed | `verdi computer test <computer>` — each step (SSH, shell, scheduler, workdir) reports pass/fail independently; `verdi calcjob logs <PK>` for the full scheduler error | Re-run `verdi computer configure core.ssh <computer>` if SSH changed; create a new Code node with the updated executable path (old Code node is immutable) |
| Missing pseudopotential family | `sssp_family.get_pseudos()` raises `NotExistent` or `KeyError`; submission script crashes before submitting | SSSP or PseudoDojo family not installed on this profile; wrong family label; element not covered by the installed family tier | `aiida-pseudo list` to see installed families and their element coverage | `aiida-pseudo install sssp -v 1.3 -x PBE -p efficiency`; verify element coverage with `aiida-pseudo show <family>` before batch submission |
| Non-reproducible local files | Exported AiiDA archive fails to reproduce a calculation; a pseudopotential or input file is missing from the archive | `UpfData` or `FolderData` node was created by referencing a local file path rather than storing file contents inside AiiDA's Disk-Objectstore repository | `verdi node show <UpfData_PK>` — inspect whether `repository_content` is populated; a path-referenced node will have empty repository content | Always use `aiida-pseudo` to install PP families (files are stored inside the repository, not by path); for custom files use `FolderData.put_object_from_path(path)` |
| Unmanaged external data | Campaign cannot be reproduced after the input CIF files are deleted; AiiDA archive does not contain the original structures | `StructureData` nodes were created from in-memory objects but `.store()` was not called before submission, or structures were constructed from file paths not tracked in the provenance graph | Check stored node count: `verdi node list -t StructureData`; inspect whether node UUIDs match what is in the exported archive | Always call `structure.store()` before submission or rely on `submit(builder)` to store automatically; verify that every input node appears in the exported archive with `verdi archive inspect campaign.aiida` |
| Database bloat | AiiDA repository grows to hundreds of GB; `verdi process list` queries slow; disk quota exceeded | CalcJobs retrieving large files (WAVECAR, CHGCAR, density cubes) into the local AiiDA repository; `clean_workdir=False` leaving full output trees in both remote scratch and local repo; many abandoned WAITING nodes | `verdi storage info` for node counts and repository size; `du -sh ~/.aiida/repository/` | Set `clean_workdir=True` in production; configure `retrieve_list` in the CalcJob plugin to retrieve only essential output files; run `verdi storage maintain` periodically; delete orphaned draft nodes with `verdi node delete` |
| Plugin version drift | Old CalcJobs cannot be loaded after an `aiida-core` or plugin upgrade; WorkChain class names changed; `verdi process list` shows `EXCEPTED` nodes with `ImportError` | `aiida-quantumespresso` or `aiida-core` upgraded without running database migrations; process entry points renamed across major versions | `verdi storage migrate` to check whether migrations are pending; inspect `verdi node show <PK>` for `process_label` vs. current entry point; check plugin CHANGELOG for renamed classes | Pin `aiida-core` and plugin versions in `requirements.txt`; run `verdi storage migrate` on a database copy before upgrading in production; do not upgrade plugin versions mid-campaign |
| Scheduler mismatch | CalcJob fails with `SUBMIT_FAILED` or `UNEXPECTED_SCHEDULER_OUTPUT`; job never appears in `squeue`/`qstat` | Scheduler plugin mismatch (SLURM configured but HPC upgraded to Slurm 23+ with changed output format); `mpirun_command` template wrong for the actual MPI launcher; wrong partition or account name | `verdi computer test <computer>` — the scheduler step will fail with the actual error; inspect `_scheduler-stderr.txt` in the CalcJob remote directory via `verdi calcjob gotocomputer` | Update `mpirun_command` via `verdi computer configure`; verify account and partition with `sacct -u <user>` and `sinfo` on the HPC; update scheduler plugin if the HPC changed schedulers |
| Hidden scientific assumptions in opaque WorkChains | Results are not reproducible from the published methods; k-mesh, cutoff, or smearing values cannot be found in the provenance graph; apparent protocol compliance that relies on plugin defaults that changed across versions | `ecutwfc`, `kpoints`, or `degauss` were embedded as default values inside a WorkChain class rather than passed as explicit nodes; plugin defaults changed between `aiida-quantumespresso` 4.3 and 4.4 | `verdi node show <CalcJobNode_PK>` → inspect `inputs.parameters` Dict; if `ecutwfc` is absent from the stored Dict, it was hardcoded | Always pass `ecutwfc`, `ecutrho`, `degauss`, and `kpoints` as explicit nodes to every WorkChain, never relying on plugin defaults; record the `aiida-quantumespresso` version alongside each campaign using `verdi status` output |

## Best Practices

- **Store all inputs as AiiDA nodes before submission.** `submit(builder)` stores unstored nodes automatically, but calling `.store()` explicitly gives you the PK/UUID immediately for cross-referencing with external systems.
- **Use Groups to organise every campaign.** Create one Group per campaign (`Group.objects.get_or_create(label=campaign_label)`) and add every submitted process and its input structure. This enables `verdi process list --group`, bulk QueryBuilder filtering by group, and clean bulk export. Never rely solely on label-prefix matching.
- **Always use WorkChains, not raw CalcJobs, in production.** `PwBaseWorkChain`, `Cp2kBaseWorkChain`, and their equivalents implement error handling, restart logic, and exit codes that raw CalcJobs lack. A raw CalcJob that hits a wall-time limit leaves a `FAILED` node with no attempt at recovery.
- **Pin `aiida-core` and plugin versions per project.** Database schemas and plugin APIs change across minor versions. Record the pinned versions in `requirements.txt`, commit it alongside the campaign setup scripts, and run `verdi storage migrate` on a database copy before any upgrade.
- **Set `clean_workdir=True` in production campaigns.** AiiDA stores the files it needs in its repository automatically during retrieval. The remote HPC scratch directory can and should be deleted after that. Without this, scratch quotas fill up silently over long campaigns.
- **Tag every submitted process with `extras`.** `wc.set_extra("formula", formula)` and `wc.set_extra("campaign", campaign_label)` enable QueryBuilder filtering and `verdi process list` grouping. Label strings alone are fragile; extras are queryable fields.
- **Use `aiida-pseudo` for all pseudopotential management.** Pseudopotentials installed via `aiida-pseudo install sssp` are stored as `UpfData` nodes in the AiiDA repository with checksums, are included in exported archives, and can be imported on other machines with matching UUIDs. Manually placed UPF files referenced by local path break portability and reproducibility.
- **Export provenance archives early and regularly.** `verdi archive export -O campaign.aiida -G <group>` creates a portable, importable snapshot. Export after each milestone (cutoff convergence, relaxations complete, phonons complete). A database corruption between exports loses only the most recent work. Archives are immutable; they cannot be accidentally edited.
- **Validate computer configuration after any network or HPC change.** `verdi computer test <computer>` verifies SSH connectivity, shell availability, scheduler command parsing, and working directory access. Run it whenever HPC hostnames, SSH keys, module paths, or scheduler configurations change. A failing computer test explains most `SUBMIT_FAILED` errors.

## Integration with Other Skills

- **[high-throughput-dft](../high-throughput-dft/SKILL.md):** AiiDA is an alternative to atomate2/jobflow for HT-DFT orchestration, with stronger provenance guarantees and better cross-institutional portability. The convergence tier taxonomy (screening vs. standard vs. high-accuracy), quality flag methodology, and deduplication strategy from that skill apply equally to AiiDA-managed campaigns. AiiDA and atomate2 can coexist in the same research group: use atomate2 for rapid internal campaigns, AiiDA for publication-grade or cross-institutional work.
- **[quantum-espresso](../quantum-espresso/SKILL.md):** `aiida-quantumespresso` wraps QE's `pw.x`, `ph.x`, `nscf.x`, and `pp.x` as AiiDA CalcJobs. The `ecutwfc`, `ecutrho`, `degauss`, `ibrav=0`, and pseudopotential selection rules from the QE skill are the domain knowledge that must be encoded in the `Dict` parameters passed to `PwCalculation` and `PwBaseWorkChain`. The SSSP pseudopotential family installed via `aiida-pseudo` corresponds directly to the SSSP library described in that skill.
- **[cp2k-workflow](../cp2k-workflow/SKILL.md):** `aiida-cp2k` provides `Cp2kCalculation` and `Cp2kBaseWorkChain` for running CP2K DFT and AIMD workflows under AiiDA provenance. CP2K input files are constructed as nested Python dicts (converted to `&SECTION` format by the plugin). CP2K is the preferred AiiDA backend for large-scale AIMD and condensed-phase simulations. The CUTOFF, REL_CUTOFF, and OT/diagonalization settings from the CP2K skill must be encoded as explicit Dict nodes, not embedded as plugin defaults.
- **[vasp-workflow](../vasp-workflow/SKILL.md):** `aiida-vasp` provides VASP workflow support under AiiDA. VASP's proprietary license means POTCAR files must be installed manually on each Computer and are not redistributable via `aiida-pseudo`. pymatgen `VaspInputSet` classes can generate INCAR and KPOINTS, which are then wrapped as Dict nodes before submission. [EXPERT REVIEW NEEDED: verify aiida-vasp compatibility with AiiDA 2.5 and VASP 6.4]
- **[dft-convergence](../dft-convergence/SKILL.md):** The cutoff scan, k-mesh convergence, and smearing checks from the convergence skill are the domain foundation for the AiiDA parameter sweeps in Workflow 4. AiiDA preserves the full convergence series as a set of linked WorkChainNodes sharing a common StructureData input node, making the chosen production parameters auditable in the provenance graph without requiring a separate convergence log.
- **[dft-dataset-generation](../dft-dataset-generation/SKILL.md):** AiiDA provides a provenance-tracked alternative to ad hoc SLURM arrays for DFT training data generation. Workflow 5 above extracts energy, forces, and stresses from `PwCalculation` output nodes and writes them to extxyz with the CalcJob UUID as a per-frame identifier, satisfying the dataset-level provenance requirements described in that skill.
- **[materials-databases-access](../../scientific-data/materials-databases-access/SKILL.md):** Starting structures for AiiDA campaigns are typically retrieved from the Materials Project via `mp-api` or from AFLOW, COD, or ICSD. The retrieved pymatgen `Structure` objects are immediately converted to `StructureData` nodes and stored before submission, establishing a direct provenance link from the external database record to the AiiDA calculation. The ICSD and COD identifiers can be stored as node extras for traceability.
- **[pymatgen-analysis](../../scientific-data/pymatgen-analysis/SKILL.md):** `StructureData.get_pymatgen()` and `StructureData(pymatgen=...)` are the primary interfaces between AiiDA's provenance graph and pymatgen's analysis tools. Phase diagram construction via `PhaseDiagram`, structure symmetrization via `SpacegroupAnalyzer`, and convex hull stability analysis all happen on pymatgen objects extracted from AiiDA via QueryBuilder.
- **[ase-framework](../../scientific-data/ase-framework/SKILL.md):** `StructureData.get_ase()` and `StructureData(ase=atoms)` are the low-overhead bridge for structure manipulation and extxyz export. The ASE interface is preferred over pymatgen for bulk trajectory export from AiiDA (Workflow 5) due to lower conversion overhead for large frame counts.
- **[reproducibility-workflow](../../scientific-data/reproducibility-workflow/SKILL.md):** AiiDA's provenance graph is the strongest available reproducibility tool for DFT campaigns — the graph itself is the provenance record. The complementary layer from that skill (environment snapshots, pseudopotential checksums, DVC for exported archives) applies to the AiiDA setup: pin and record `aiida-core` and plugin versions in `requirements.txt`; checksum AiiDA archive files with SHA-256 before depositing to NOMAD; version-control `verdi computer` and `verdi code` configuration YAML files alongside campaign setup scripts.
- **[scientific-visualization](../../scientific-data/scientific-visualization/SKILL.md):** `BandsData` nodes from `PwBandsWorkChain` are visualized with pymatgen's `BSPlotter` or via `verdi data bands export --format gnuplot`. Phonon dispersion from `PhononWorkChain` is exported as a `BandsData` node and plotted with phonopy. Relaxation trajectories stored as `TrajectoryData` nodes are exported to extxyz and visualized with OVITO.
- **[mlp-validation](../../ml-interatomic-potentials/mlp-validation/SKILL.md):** DFT reference data in AiiDA (energies, forces, stresses from `PwCalculation` or `Cp2kCalculation` output nodes) can be exported as extxyz test sets for MLP validation (Workflow 5). The CalcJob UUID provides a unique, traceable identifier for each frame, enabling exact traceability from an MLP validation error back to the specific DFT calculation and its full provenance chain.

## Key References

1. Pizzi, G., Cepellotti, A., Sabatini, R., Marzari, N. & Kozinsky, B. (2016). *AiiDA: Automated interactive infrastructure and database for computational science.* Computational Materials Science, 111, 218–230. https://doi.org/10.1016/j.commatsci.2015.09.013
2. Huber, S. P., Zoupanos, S., Uhrin, M., Talirz, L., Kahle, L., Häuselmann, R., Gresch, D., Müller, T., Yakutovich, A. V., Andersen, C. W., Ramirez, F. F., Adorf, C. S., Gargiulo, F., Kumbhar, S., Passaro, E., Johnston, C., Merkys, A., Cepellotti, A., Mounet, N., Marzari, N., Kozinsky, B. & Pizzi, G. (2020). *AiiDA 1.0, a scalable computational infrastructure for automated reproducible workflows and data provenance in computational science.* Scientific Data, 7, 300. https://doi.org/10.1038/s41597-020-00638-4
3. Uhrin, M., Huber, S. P., Yu, J., Marzari, N. & Pizzi, G. (2021). *Workflows in AiiDA: Engineering a high-throughput, event-based engine for robust and modular computational workflows.* Computational Materials Science, 187, 110086. https://doi.org/10.1016/j.commatsci.2020.110086
4. Huber, S. P., Bosoni, E., Bercx, M., Bröder, J., Degomme, A., Dikan, V., Eimre, K., Flament, M., García, A., Grisafi, A., Guda, S., Hourahine, B., Huran, A. W., Illas, F., Keller, M., Laskowski, R., Lin, L., Mercado, R., Mostofi, A. A., Petretto, G., Ratcliff, L. E., Rignanese, G.-M., Schütt, O., Timrov, I., Torrent, M., Van Hove, M., Wirtz, L., Zhang, C., Normand, J., Marzari, N., Goedecker, S., Needs, R. J., Marzari, N. & Pizzi, G. (2021). *Common workflows for computing material properties using different quantum engines.* npj Computational Materials, 7, 291. https://doi.org/10.1038/s41524-021-00594-6
5. Bosoni, E., Beal, L., Bercx, M., Blöchl, P., Bühlmann, M., Cappi, F., Castelli, I., Cepellotti, A., Cerqueira, T., Corsetti, F., & Pizzi, G. (2024). *How to verify the precision of density-functional-theory implementations via reproducible and universal workflows.* Nature Reviews Physics, 6, 45–58. https://doi.org/10.1038/s42254-023-00655-3
6. Talirz, L., Kumbhar, S., Passaro, E., Yakutovich, A. V., Granata, V., Gargiulo, F., Borelli, M., Uhrin, M., Huber, S. P., Zoupanos, S., Adorf, C. S., Andersen, C. W., Schütt, O., Pignedoli, C. A., Passerone, D., VandeVondele, J., Schulthess, T. C., Smit, B., Pizzi, G. & Marzari, N. (2020). *Materials Cloud, a platform for open computational science.* Scientific Data, 7, 299. https://doi.org/10.1038/s41597-020-00637-5
