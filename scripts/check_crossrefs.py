"""Check that local cross-references in SKILL.md files resolve to existing skills."""

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SKILLS_DIR = REPO_ROOT / "skills"


def find_local_skill_links(text: str, source_file: Path) -> list[tuple[str, Path]]:
    """Return (link_text, resolved_path) for all markdown links pointing to SKILL.md files."""
    results = []
    for match in re.finditer(r"\[([^\]]+)\]\(([^)]+)\)", text):
        label = match.group(1)
        target = match.group(2)
        if "SKILL.md" not in target or target.startswith("http"):
            continue
        resolved = (source_file.parent / target).resolve()
        results.append((label, resolved))
    return results


def main() -> None:
    skill_files = list(SKILLS_DIR.rglob("SKILL.md"))
    if not skill_files:
        print("No SKILL.md files found.")
        sys.exit(0)

    broken: list[tuple[Path, str, Path]] = []

    for skill_file in sorted(skill_files):
        text = skill_file.read_text(encoding="utf-8")
        links = find_local_skill_links(text, skill_file)
        for label, resolved in links:
            if not resolved.exists():
                broken.append((skill_file, label, resolved))

    if broken:
        print(f"BROKEN cross-references ({len(broken)} found):")
        for source, label, target in broken:
            rel_source = source.relative_to(REPO_ROOT)
            rel_target = target.relative_to(REPO_ROOT) if target.is_relative_to(REPO_ROOT) else target
            print(f"  [{label}] in {rel_source}")
            print(f"    -> {rel_target} (does not exist)")
        sys.exit(1)
    else:
        print(f"All cross-references OK ({len(skill_files)} skill(s) checked)")
        sys.exit(0)


if __name__ == "__main__":
    main()
