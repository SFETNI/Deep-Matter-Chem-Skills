# Release Checklist

## Purpose

Use this checklist before tagging or publishing a GitHub milestone release of `deep-matter-chem-skills`. Work through the sections in order. Do not tag until all "Must fix" items are checked.

---

## Current milestone — v0.2 reference-audited production-support release

- **52 production-usable support skills** across 15 active pillars
- Structurally validated — all skills pass `scripts/validate_skills.py`
- Zero broken internal cross-references — confirmed by `scripts/check_crossrefs.py`
- **Reference and DOI audit completed** — ~230 references checked; 8 wrong DOIs corrected, 4 metadata fields corrected across 7 skills
- Skills are production-usable support material, not scientifically validated — `draft` maturity means structurally complete and reference-audited, not domain-expert reviewed
- No skills have been promoted beyond `draft`; independent domain review is the path to `validated`

---

## Must fix before GitHub

All items in this section must be checked before the repository is made public or a release tag is pushed.

### Tooling and schema

- [ ] All skill validators pass: `python scripts/validate_skills.py path/to/SKILL.md` for every skill
- [ ] Catalog generation passes without errors: `python scripts/generate_catalog.py`
- [ ] Cross-reference check passes: `python scripts/check_crossrefs.py`
- [ ] Zero merge-conflict markers in any tracked file (`<<<<<<`, `=======`, `>>>>>>>` at line start)

### Repository hygiene

- [ ] No tracked local settings, cache files, worktree metadata, or temporary files
  - Check: `.claude/settings.local.json` is in `.gitignore` and not staged
  - Check: `.claude/worktrees/` is in `.gitignore` and not staged
  - Check: `__pycache__/`, `.ipynb_checkpoints/`, `*.pyc` are not tracked
- [ ] Working tree is clean: `git status` reports nothing to commit

### Documentation

- [ ] `README.md` reflects the current skill count and milestone status honestly
- [ ] `README.md` does not claim skills are scientifically validated
- [ ] `CONTRIBUTING.md` is complete and links to existing docs
- [ ] `docs/maturity-model.md` exists and is consistent with validator behavior
- [ ] `docs/skills-catalog.md` was regenerated and is current

### GitHub templates

- [ ] Issue templates exist in `.github/ISSUE_TEMPLATE/`
  - [ ] `bug-report.yml`
  - [ ] `broken-reference.yml`
  - [ ] `new-skill-proposal.yml`
  - [ ] `expert-review-request.yml`
  - [ ] `schema-validation-failure.yml`
  - [ ] `config.yml`
- [ ] PR template exists: `.github/pull_request_template.md`
- [x] Issue-template contact links use the actual GitHub repository URL and `main` branch

### Content integrity

- [ ] No broken internal skill links (covered by `check_crossrefs.py`, but confirm manually for any skill added after last run)
- [ ] Catalog renders all non-empty pillars correctly — spot-check `docs/skills-catalog.md`
- [x] No known wrong DOI or reference issues have been left unaddressed — full audit completed for v0.2

---

## Should fix before public announcement

Items here do not block the initial push but should be resolved before actively promoting the repository.

### Reference and content quality

- [x] Reference audit — full audit of all ~230 references across all 52 skills completed for v0.2
- [ ] `[EXPERT REVIEW NEEDED]` marker triage: decide which markers are blocking and which can wait
  - High priority: `atomistic-md` pillar (16 markers in coarse-grained-md, 15 in gromacs-workflow, etc.)
  - Medium priority: `ml-interatomic-potentials` pillar (10 markers in universal-potentials)
  - See `scripts/check_crossrefs.py` output and grep for counts: `grep -r "EXPERT REVIEW NEEDED" skills/ | wc -l`

### Maturity and roadmap

- [ ] Maturity label review — confirm no skill was assigned `draft` when it should be `stub` or vice versa
- [ ] Roadmap clarity — update or create `ROADMAP.md` if needed to document the ~110-skill longer-term target
- [ ] Identify and document domain-review priorities in an issue or `docs/` file

### Release logistics

- [ ] Prepare initial GitHub release notes summarising the milestone (52 skills, pillar coverage, known gaps)
- [x] Issue-template contact links use actual GitHub repository URLs
- [ ] Review `README.md` quickstart commands against a fresh clone to confirm they work end-to-end

---

## Can fix after release

Items here are improvements that do not need to block either the initial push or public announcement.

- [x] Broader reference audit — systematic DOI verification across all 52 skills (completed v0.2)
- [ ] More domain-expert reviews and maturity promotions from `draft` to `validated`
- [ ] Additional diagrams (workflow illustrations, pillar dependency maps)
- [ ] Additional roadmap skills toward the ~110-skill target
- [ ] Automated DOI and reference-link checking (CI step or pre-commit hook)
- [ ] More contributor-facing examples in `CONTRIBUTING.md`
- [ ] Deeper taxonomy cleanup — review whether any skills belong in a different pillar
- [ ] `reaction-thermo` pillar: currently empty; add first skill or remove from `PILLARS` list
- [x] No stale Git worktrees are registered in the clean checkout

---

## Final release commands

Run these in order immediately before tagging. All four must pass cleanly.

```bash
python scripts/generate_catalog.py
python scripts/check_crossrefs.py
for f in $(find skills -name "SKILL.md" | sort); do python scripts/validate_skills.py "$f" || exit 1; done
git status
```

If all pass and working tree is clean, tag the release:

```bash
git tag v0.2-52-skills-zero-crossrefs
```

---

## Pre-tag verification

- [ ] Skill count confirmed: `find skills -name "SKILL.md" | wc -l` matches expected count
- [ ] Catalog count matches skill count: `python scripts/generate_catalog.py` reports the same number
- [ ] Cross-reference checker reports zero broken links
- [ ] Full validation passes with no errors
- [ ] Working tree is clean: `git status` reports nothing to commit
- [ ] Milestone tag name selected and agreed
- [ ] Release notes drafted and ready to paste into GitHub

---

## Post-tag / GitHub steps

- [ ] Create the GitHub repository if it does not exist, or confirm the remote is set correctly
- [ ] Confirm remote URL: `git remote -v`
- [ ] Push the branch: `git push -u origin main`
- [ ] Push the tag: `git push origin v0.2-reference-audited`
- [ ] Verify issue templates render in the GitHub "New issue" chooser
- [ ] Verify README Mermaid flowchart renders on the GitHub repository landing page
- [ ] Open one test issue using each template and confirm fields render correctly; close the test issues
- [ ] Publish release notes on the GitHub Releases page for the tag
- [ ] Invite contributors — link from relevant communities, mailing lists, or Slack channels as appropriate
