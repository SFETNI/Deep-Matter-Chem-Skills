"""Validate a SKILL.md file against the deep-matter-chem-skills schema."""

import re
import sys
from pathlib import Path

VALID_MATURITY = {"planned", "stub", "draft", "validated", "flagship"}

REQUIRED_METADATA = [
    "name",
    "domain",
    "maturity",
    "tools",
    "tested_versions",
    "requires_gpu",
    "estimated_runtime",
]

REQUIRED_SECTIONS = [
    "Description",
    "Domain Context",
    "When to Use This Skill",
    "When Not to Use This Skill",
    "Core Concepts",
    "Key Workflows",
    "Common Failure Modes",
    "Best Practices",
    "Integration with Other Skills",
    "Key References",
]

FLAGSHIP_MIN_WORKFLOWS = 2
DRAFT_MIN_WORKFLOWS = 1
FAILURE_MODE_MIN_ROWS = {
    "flagship": 5,
    "validated": 5,
    "draft": 3,
    "stub": 0,
    "planned": 0,
}
REFERENCE_MIN = {
    "flagship": 5,
    "validated": 3,
    "draft": 3,
    "stub": 1,
    "planned": 0,
}


def extract_yaml_block(text: str) -> tuple[str, bool]:
    """Return (yaml_text, found). Expects --- delimited front matter."""
    if not text.startswith("---"):
        return "", False
    end = text.find("\n---", 3)
    if end == -1:
        return "", False
    return text[3:end].strip(), True


def parse_yaml_simple(yaml_text: str) -> dict:
    """Minimal YAML parser for scalar and list values only."""
    result: dict = {}
    lines = yaml_text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip() or line.strip().startswith("#"):
            i += 1
            continue
        if ":" in line and not line.startswith(" ") and not line.startswith("-"):
            key, _, rest = line.partition(":")
            key = key.strip()
            rest = rest.strip()
            if rest:
                result[key] = rest
            else:
                # Possibly a list or nested block follows
                items = []
                i += 1
                while i < len(lines) and (lines[i].startswith("  ") or lines[i].startswith("\t")):
                    item = lines[i].strip()
                    if item.startswith("- "):
                        items.append(item[2:].strip())
                    elif ":" in item:
                        # nested key-value (e.g. tested_versions)
                        nk, _, nv = item.partition(":")
                        if isinstance(result.get(key), dict):
                            result[key][nk.strip()] = nv.strip().strip('"')
                        else:
                            result[key] = {nk.strip(): nv.strip().strip('"')}
                    i += 1
                if items:
                    result[key] = items
                continue
        i += 1
    return result


def count_workflows(text: str) -> int:
    return len(re.findall(r"^###\s+Workflow\s+\d+", text, re.MULTILINE))


def count_failure_rows(text: str) -> int:
    """Count data rows in the Common Failure Modes table."""
    in_failure_section = False
    row_count = 0
    header_passed = False
    for line in text.splitlines():
        if re.match(r"^##\s+Common Failure Modes", line):
            in_failure_section = True
            header_passed = False
            continue
        if in_failure_section:
            if line.startswith("##"):
                break
            if re.match(r"^\|[-| ]+\|", line):
                header_passed = True
                continue
            if header_passed and line.startswith("|") and line.strip() != "|":
                row_count += 1
    return row_count


def count_references(text: str) -> int:
    in_ref_section = False
    count = 0
    for line in text.splitlines():
        if re.match(r"^##\s+Key References", line):
            in_ref_section = True
            continue
        if in_ref_section:
            if line.startswith("##"):
                break
            if re.match(r"^\d+\.", line.strip()):
                count += 1
    return count


def has_expert_review_markers(text: str) -> bool:
    return "[EXPERT REVIEW NEEDED]" in text


def check_skill(path: Path) -> list[str]:
    errors: list[str] = []
    text = path.read_text(encoding="utf-8")

    # --- YAML block ---
    yaml_text, found = extract_yaml_block(text)
    if not found:
        errors.append("MISSING: YAML front matter block (--- delimited)")
        return errors  # Cannot continue without metadata

    meta = parse_yaml_simple(yaml_text)

    for field in REQUIRED_METADATA:
        if field not in meta:
            errors.append(f"MISSING metadata field: '{field}'")

    maturity = meta.get("maturity", "")
    if maturity and maturity not in VALID_MATURITY:
        errors.append(f"INVALID maturity value: '{maturity}' (must be one of {sorted(VALID_MATURITY)})")

    # --- Required sections ---
    for section in REQUIRED_SECTIONS:
        pattern = re.compile(r"^##\s+" + re.escape(section), re.MULTILINE)
        if not pattern.search(text):
            errors.append(f"MISSING section: '## {section}'")

    # --- Workflow count ---
    workflow_count = count_workflows(text)
    if maturity == "flagship" and workflow_count < FLAGSHIP_MIN_WORKFLOWS:
        errors.append(
            f"INSUFFICIENT workflows: flagship requires {FLAGSHIP_MIN_WORKFLOWS}, found {workflow_count}"
        )
    elif maturity in ("draft", "validated") and workflow_count < DRAFT_MIN_WORKFLOWS:
        errors.append(
            f"INSUFFICIENT workflows: {maturity} requires {DRAFT_MIN_WORKFLOWS}, found {workflow_count}"
        )

    # --- Failure mode rows ---
    min_rows = FAILURE_MODE_MIN_ROWS.get(maturity, 0)
    if min_rows > 0:
        row_count = count_failure_rows(text)
        if row_count < min_rows:
            errors.append(
                f"INSUFFICIENT failure modes: {maturity} requires {min_rows} rows, found {row_count}"
            )

    # --- Reference count ---
    min_refs = REFERENCE_MIN.get(maturity, 0)
    if min_refs > 0:
        ref_count = count_references(text)
        if ref_count < min_refs:
            errors.append(
                f"INSUFFICIENT references: {maturity} requires {min_refs}, found {ref_count}"
            )

    # --- Expert review markers in validated/flagship ---
    if maturity in ("validated", "flagship") and has_expert_review_markers(text):
        errors.append(
            "INVALID: '[EXPERT REVIEW NEEDED]' markers are not allowed in validated or flagship skills"
        )

    return errors


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: validate_skills.py <path-to-SKILL.md>")
        sys.exit(2)

    path = Path(sys.argv[1])
    if not path.exists():
        print(f"ERROR: File not found: {path}")
        sys.exit(2)

    errors = check_skill(path)

    if errors:
        print(f"VALIDATION FAILED: {path}")
        for err in errors:
            print(f"  - {err}")
        sys.exit(1)
    else:
        print(f"VALIDATION PASSED: {path}")
        sys.exit(0)


if __name__ == "__main__":
    main()
