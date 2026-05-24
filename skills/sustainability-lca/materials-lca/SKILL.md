---
name: materials-lca
domain: sustainability-lca
maturity: draft
tools:
  - openLCA
  - Brightway
  - ecoinvent
  - GREET
  - pandas
  - numpy
  - matplotlib
tested_versions:
  openLCA: "2.0"
  Brightway: "2.5"
  ecoinvent: "3.10"
  GREET: "2023"
  pandas: "2.2.0"
  numpy: "1.26.4"
requires_gpu: false
estimated_runtime: hours-to-weeks
---

# Materials Life-Cycle Assessment

## Description

This skill covers life-cycle assessment (LCA) for materials, chemicals, and manufacturing routes: goal and scope definition, functional units, system boundaries, inventory construction, allocation, impact assessment, uncertainty, scenario analysis, database provenance, prospective LCA, and sustainability-aware materials design. Invoke this skill when comparing environmental burdens of materials or processes, screening candidate routes, or coupling sustainability metrics to materials databases, high-throughput screening, small-data ML, Bayesian optimization, or thermodynamic design workflows.

## Domain Context

Life-cycle assessment estimates environmental impacts associated with a product, process, material, or service across a defined life cycle. It is not a single number. An LCA result depends on the goal and scope, functional unit, system boundary, allocation choices, life-cycle inventory, background database, impact assessment method, geography, electricity mix, technology maturity, and assumptions about use phase, end of life, recycling, and co-products.

For materials and chemicals, the functional unit is often the most important modeling decision. "1 kg of material" is rarely enough when materials differ in lifetime, strength, conductivity, catalytic activity, purity, embodied function, or replacement rate. A battery cathode, catalyst, membrane, structural alloy, polymer, solvent, or precursor should be compared on a unit of service whenever possible. The wrong functional unit can reverse conclusions.

LCA is appropriate for structured environmental comparison, hotspot analysis, prospective technology assessment, and sustainability-aware screening. It is not a substitute for techno-economic analysis, process safety analysis, toxicity testing, policy modeling, or full social/environmental justice assessment, although it may provide inputs to those workflows. Impact categories are model-dependent and region-dependent; global warming potential is only one dimension of sustainability. [EXPERT REVIEW NEEDED]

Emerging materials add extra uncertainty. Lab-scale inventories may not represent industrial operation, yields may improve with scale, solvents and precursors may change, recycling may not exist yet, and future electricity mixes may be different. Prospective LCA should therefore report scenarios and uncertainty rather than a single polished number. [EXPERT REVIEW NEEDED]

## When to Use This Skill

- Use when comparing material choices, synthesis routes, chemical processes, or manufacturing pathways with a defined functional unit.
- Use when identifying environmental hotspots in energy, solvents, precursors, purification, heat, electricity, transport, or end-of-life steps.
- Use when building cradle-to-gate, cradle-to-grave, or gate-to-gate inventories for materials design.
- Use when coupling sustainability metrics to high-throughput screening, Bayesian optimization, surrogate models, or multi-objective materials selection.
- Use when performing prospective LCA for emerging materials or technologies with scenario and uncertainty analysis.
- Use when auditing whether sustainability claims are traceable to inventories, assumptions, database versions, and impact methods.

## When Not to Use This Skill

- **When the functional unit is undefined.** Environmental comparison without a service basis is often meaningless.
- **When process data are too speculative for quantitative claims.** Use qualitative hotspot analysis or scenario ranges instead of precise rankings.
- **When comparing lab-scale and industrial routes directly.** Lab yields, solvent use, purification, and energy demand often differ by orders of magnitude.
- **When only one impact category is considered.** Lower GWP can coincide with higher toxicity, water use, land use, or resource depletion.
- **When system boundaries differ across alternatives.** Cradle-to-gate results cannot be compared directly to cradle-to-grave results.
- **When regional conditions dominate but are ignored.** Electricity mix, water stress, transport, waste handling, and regulation are location-dependent. [EXPERT REVIEW NEEDED]

## Core Concepts

- **Goal and scope:** Formal statement of why the LCA is being performed, who will use it, what decisions it supports, and what system is studied.
- **Functional unit:** Quantified service basis for comparison, such as "1 kWh delivered over battery lifetime" or "1 m2 membrane treating 1 m3 water."
- **System boundary:** Processes included in the model, such as cradle-to-gate, cradle-to-grave, gate-to-gate, recycling, use phase, or infrastructure.
- **Life-cycle inventory (LCI):** Quantified inputs and outputs: materials, energy, emissions, water, wastes, transport, yields, and co-products.
- **Life-cycle impact assessment (LCIA):** Conversion of inventory flows into impact indicators such as GWP, water use, toxicity, or resource depletion.
- **Allocation:** Partitioning burdens among co-products, using mass, energy, economic value, system expansion, substitution, or another documented rule.
- **Foreground system:** Processes modeled specifically for the studied material or technology.
- **Background system:** Generic upstream and downstream processes from databases such as ecoinvent, GaBi/Sphera, GREET, or openLCA-compatible libraries.
- **Process-based LCA:** Bottom-up inventory of unit processes and supply chains.
- **Input-output LCA:** Economy-wide sector-level model based on monetary flows and environmental extensions.
- **Hybrid LCA:** Combination of process detail and input-output coverage.
- **Prospective LCA:** LCA for future or emerging technologies using scale-up, scenario, and learning assumptions.

## Key Workflows

### Workflow 1: Define goal, scope, functional unit, and boundaries

Do not collect data before defining the comparison.

Scope checklist:

- State the decision context and intended audience.
- Define the alternatives being compared.
- Choose a functional unit tied to material performance, not just mass.
- Choose system boundary: cradle-to-gate, cradle-to-grave, gate-to-gate, or another justified boundary.
- State geography, time horizon, technology readiness level, and scale.
- Select impact categories and LCIA method.
- Define allocation rules for co-products and recycling.
- Identify cut-off criteria and excluded processes.
- Plan uncertainty, sensitivity, and scenario analyses.

Example scope record:

```yaml
lca_scope:
  goal: "compare cathode synthesis routes for screening"
  functional_unit: "1 kWh nominal cathode active material capacity"
  boundary: "cradle-to-gate"
  geography: "EU electricity mix baseline, China sensitivity case"
  time_horizon: "near-term industrial production"
  impact_categories:
    - global_warming_potential_100yr
    - cumulative_energy_demand
    - water_use
    - mineral_resource_depletion
  allocation:
    co_products: "economic allocation, sensitivity with system expansion"
  excluded:
    - "laboratory analytical equipment"
    - "building infrastructure"
```

### Workflow 2: Build a life-cycle inventory with provenance

Inventory construction is where most hidden assumptions enter an LCA.

Inventory sources:

- Measured process data: mass balances, energy use, yields, emissions, water use, waste treatment.
- Literature inventories: synthesis recipes, process models, industrial reports, patents, or supplementary information.
- Background databases: ecoinvent, GaBi/Sphera, GREET, or openLCA-compatible datasets.
- Materials databases and high-throughput screening outputs for composition, performance, and candidate selection.
- Engineering estimates for scale-up, heat duties, solvent recovery, and process yields.

Inventory table fields:

```text
process_step, flow_name, amount, unit, direction, data_source,
geography, year, uncertainty, allocation_rule, database_process_id,
comment
```

Quality checks:

- Mass and elemental balances close within stated tolerance.
- Energy units are consistent and separated into heat, electricity, and fuel where possible.
- Solvent recovery and recycling assumptions are explicit.
- Regional electricity and heat sources match the scenario.
- Cut-off flows are documented rather than silently omitted.
- Database process IDs and versions are recorded.

### Workflow 3: Choose allocation and co-product handling

Many chemical and materials routes generate co-products, by-products, recycled streams, or shared infrastructure. Allocation choices can dominate results.

Common approaches:

- **Mass allocation:** Burdens split by mass of outputs. Simple but inappropriate when low-mass high-value products drive production.
- **Economic allocation:** Burdens split by market value. Sensitive to price volatility and regional markets.
- **Energy allocation:** Useful for fuels or energy products.
- **System expansion/substitution:** Credits a co-product for displacing another product. Requires careful market assumptions.
- **Cut-off recycling:** Recycled material carries no upstream burden or receives no credit depending on convention.

Always report allocation method and run sensitivity analysis when co-products materially affect conclusions. Allocation is policy- and standard-dependent. [EXPERT REVIEW NEEDED]

### Workflow 4: Calculate impact categories and compare alternatives

Impact assessment should cover the categories relevant to the decision.

Useful categories for materials and chemicals:

- Global warming potential (GWP).
- Cumulative energy demand.
- Water consumption and water scarcity.
- Human toxicity and ecotoxicity.
- Particulate matter formation.
- Acidification and eutrophication.
- Mineral resource depletion and critical material use.
- Land use.
- Circularity or recycled-content metrics.

Comparison checklist:

- Normalize results to the same functional unit.
- Keep system boundaries identical across alternatives.
- Report contribution analysis by process step.
- Show uncertainty intervals or scenario ranges.
- Avoid declaring a route "sustainable" from one impact metric.
- Check tradeoffs between climate, toxicity, water, land, and resource indicators.

### Workflow 5: Prospective LCA for emerging materials

Prospective LCA is essential for early-stage materials but must be explicit about uncertainty.

Scenario dimensions:

- Lab scale, pilot scale, and industrial scale.
- Current and future electricity mixes.
- Yield improvements and solvent recovery.
- Alternative precursors or synthesis routes.
- Recycling, reuse, or circularity assumptions.
- Manufacturing location and transport distances.
- Learning curves or process intensification.

Do not compare a hand-measured lab inventory directly to mature industrial production without scale-up scenarios. Instead, report current-lab, plausible-pilot, and target-industrial cases. [EXPERT REVIEW NEEDED]

### Workflow 6: Couple LCA to screening, ML, and optimization

Sustainability-aware materials design combines environmental metrics with performance and cost.

Integration patterns:

- Join LCA indicators to materials database records or screening candidates.
- Use descriptors and small-data ML to estimate impacts when inventories are sparse.
- Use Bayesian optimization for multi-objective design over performance, cost, and impacts.
- Use surrogate validation to audit extrapolation and uncertainty.
- Combine CALPHAD or high-throughput DFT outputs with functional-unit performance models.

Multi-objective outputs should show Pareto fronts and tradeoffs, not only a single weighted score. Weighting impact categories is value-laden and should be documented. [EXPERT REVIEW NEEDED]

### Workflow 7: Uncertainty, sensitivity, and scenario analysis

LCA results without uncertainty are brittle.

Recommended analyses:

- Parameter uncertainty: yields, energy use, emissions factors, transport distances, allocation factors.
- Database uncertainty: alternative background processes, database versions, and missing data.
- Scenario uncertainty: region, electricity mix, scale-up, end-of-life, future technology state.
- Sensitivity analysis: one-at-a-time, Sobol, Morris, or model-based sensitivity.
- Monte Carlo: inventory and characterization-factor uncertainty where data support it.

Report which conclusions are robust and which depend on assumptions. If rankings flip under reasonable assumptions, present the result as conditional.

## Common Failure Modes

| Failure | Symptom | Likely Cause | Diagnostic | Fix |
|---------|---------|--------------|------------|-----|
| Wrong functional unit | Ranking changes when performance is included | Comparing mass instead of service | Recompute per unit function | Define function-based unit |
| Hidden allocation assumptions | Co-product route looks artificially favorable | Undocumented burden splitting or credits | Inspect co-product processes | Report and sensitivity-test allocation |
| Inconsistent boundaries | One route omits use phase or end-of-life | Cradle-to-gate compared to cradle-to-grave | Boundary audit | Harmonize system boundaries |
| Outdated inventory data | Results reflect obsolete energy or process mix | Old database or literature source | Check year/version/geography | Update or scenario-test data |
| Regional electricity mismatch | Electrified route over- or under-estimated | Wrong grid mix for production location | Contribution analysis by electricity | Use regional and future grid scenarios |
| Double counting | Recycling or co-product credits counted twice | Hybrid data or system expansion overlap | Flow and credit audit | Remove duplicated process/credit |
| Ignoring uncertainty | Overconfident ranking | Single deterministic inventory | Sensitivity/Monte Carlo | Report ranges and rank robustness |
| Lab-scale overclaiming | Emerging route appears much worse or better | Lab recipe not scaled | Compare lab/pilot/industrial scenarios | Use prospective LCA scenarios |
| Single-metric overclaiming | "Green" claim conflicts with toxicity or resource results | GWP-only framing | Multi-category impact check | Report tradeoffs and limitations |
| Poor provenance | Result cannot be reproduced | Missing database, process IDs, or assumptions | Metadata audit | Store inventory and database versions |

## Best Practices

- Define the functional unit before collecting inventory data.
- Keep foreground inventory data separate from background database processes.
- Record database names, versions, process IDs, geography, year, and impact method.
- Harmonize boundaries before comparing alternatives.
- Report contribution analysis and impact tradeoffs, not only totals.
- Use uncertainty and scenario analysis for every decision-relevant conclusion.
- Treat recycling, allocation, and avoided burden assumptions as sensitivity cases.
- Avoid comparing lab-scale recipes directly to industrial processes.
- Couple environmental metrics to material performance when defining the service.
- Store raw inventories, transformation scripts, figures, and assumptions in a reproducible workflow.

## Integration with Other Skills

- **[materials-databases-access](../../scientific-data/materials-databases-access/SKILL.md):** Supplies materials records, provenance, and database joins for sustainability-aware screening.
- **[materials-featurization](../../scientific-data/materials-featurization/SKILL.md):** Provides descriptors for impact prediction, route clustering, or screening models.
- **[scientific-visualization](../../scientific-data/scientific-visualization/SKILL.md):** Supports contribution charts, uncertainty intervals, Pareto fronts, and impact tradeoff figures.
- **[reproducibility-workflow](../../scientific-data/reproducibility-workflow/SKILL.md):** Tracks inventory files, database versions, scripts, assumptions, and result provenance.
- **[small-data-ml](../../scientific-ml/small-data-ml/SKILL.md):** Provides modeling and validation practices for sparse LCA datasets.
- **[gaussian-process-materials](../../surrogate-active-learning/gaussian-process-materials/SKILL.md):** Builds uncertainty-aware impact or performance-impact surrogates.
- **[bayesian-optimization](../../surrogate-active-learning/bayesian-optimization/SKILL.md):** Supports multi-objective search over performance, cost, and environmental metrics.
- **[surrogate-validation](../../surrogate-active-learning/surrogate-validation/SKILL.md):** Validates sustainability surrogate models under extrapolation and distribution shift.
- **[high-throughput-dft](../../electronic-structure/high-throughput-dft/SKILL.md):** Provides candidate properties and performance data for functional-unit-normalized screening.
- **[calphad-workflow](../../thermodynamics-calphad/calphad-workflow/SKILL.md):** Supplies phase stability and processing context for materials selection and route assessment.

## Key References

1. ISO 14040:2006. *Environmental management - Life cycle assessment - Principles and framework*. International Organization for Standardization.
2. ISO 14044:2006. *Environmental management - Life cycle assessment - Requirements and guidelines*. International Organization for Standardization.
3. Guinée, J. B. et al. (2002). *Handbook on Life Cycle Assessment: Operational Guide to the ISO Standards*. Kluwer Academic Publishers. https://doi.org/10.1007/0-306-48055-7
4. Hellweg, S. and Mila i Canals, L. (2014). *Emerging approaches, challenges and opportunities in life cycle assessment*. Science. https://doi.org/10.1126/science.1248361
5. Wender, B. A. et al. (2014). *Illustrating anticipatory life cycle assessment for emerging photovoltaic technologies*. Environmental Science and Technology. https://doi.org/10.1021/es5016923
