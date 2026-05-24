# Skill Maturity Model

Every skill in `deep-matter-chem-skills` carries a maturity label. This model defines what each level requires and when a skill is appropriate for agent use.

---

## Levels

### `planned`

The skill exists in the roadmap taxonomy but has no content.

| Requirement | Status |
|---|---|
| Metadata block | Not required |
| Description | Not required |
| Domain context | Not required |
| Workflows | Not required |
| Failure modes | Not required |
| References | Not required |
| Code examples tested | Not required |
| Cross-skill links | Not required |
| Expert review | Not required |

**Agent use:** Do not use. Listed only in ROADMAP.md and the catalog for planning.

---

### `stub`

Metadata and description exist. The skill is recognizable but not yet usable.

| Requirement | Status |
|---|---|
| Metadata block (all required fields) | Required |
| Description | Required |
| Domain context | Required (brief) |
| When Not to Use This Skill | Required (brief) |
| Workflows | At least a placeholder |
| Failure modes table | Not required |
| References | At least 1 |
| Code examples tested | Not required |
| Cross-skill links | Not required |
| `[EXPERT REVIEW NEEDED]` markers allowed | Yes |
| Expert review | Not required |

**Agent use:** Not recommended. The skill may mislead.

---

### `draft`

All main sections exist. The skill is usable with caution. Some values or claims may still carry `[EXPERT REVIEW NEEDED]` markers.

| Requirement | Status |
|---|---|
| Metadata block (all required fields) | Required |
| Description | Required |
| Domain context | Required |
| When Not to Use This Skill | Required |
| Core Concepts | Required |
| At least 1 workflow | Required |
| Failure modes table (≥ 3 rows) | Required |
| Best Practices | Required |
| Integration with Other Skills | Required (at least 1 link) |
| References | At least 3 |
| Code examples tested | Encouraged but not required |
| `[EXPERT REVIEW NEEDED]` markers allowed | Yes |
| Expert review | Not required |

**Agent use:** Usable with caution. Flag `[EXPERT REVIEW NEEDED]` values to the user.

---

### `validated`

Workflows reviewed, references checked, failure modes credible, and no `[EXPERT REVIEW NEEDED]` markers remain.

| Requirement | Status |
|---|---|
| All `draft` requirements | Required |
| At least 1 workflow with tested code | Required |
| Failure modes table (≥ 5 rows) | Required |
| References (≥ 3, checked) | Required |
| Cross-skill links (verified) | Required |
| `[EXPERT REVIEW NEEDED]` markers | Not allowed |
| Expert review | Recommended |

**Agent use:** Recommended for production use.

---

### `flagship`

High-quality reference skill. Serves as a project exemplar.

| Requirement | Status |
|---|---|
| All `validated` requirements | Required |
| At least 2 distinct workflows | Required |
| Failure modes table (≥ 5 rows, all credible) | Required |
| All code examples tested against declared versions | Required |
| Cross-skill links (≥ 2, verified) | Required |
| References (≥ 5, primary sources) | Required |
| Domain context reviewed by domain expert | Required |
| `[EXPERT REVIEW NEEDED]` markers | Not allowed |
| Expert review | Required |

**Agent use:** Showcase quality. Recommended as the first skill to consult in its domain.

---

## Promotion criteria

A skill is promoted when all requirements for the target level are met and a pull request is reviewed and merged. Automatic promotion is never granted — a human must confirm the review.

## Demotion

If a skill's tools release breaking changes or if failure modes are found to be misleading, the maintainer should demote the skill to `draft` and open an issue for remediation.
