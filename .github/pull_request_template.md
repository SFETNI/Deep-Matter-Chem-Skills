# Pull Request

## Summary

Briefly describe what this PR changes.

## Type of change

- [ ] New skill
- [ ] Skill improvement
- [ ] Documentation
- [ ] Reference correction
- [ ] Tooling/script
- [ ] Catalog or cross-reference maintenance
- [ ] Other

## Changed files

List the main files changed by this PR.

## Skill changes

If this PR adds or changes a skill, provide:

- Skill path:
- Pillar/domain:
- Maturity level:
- New or changed workflows:
- New or changed failure modes:
- New or changed references:
- New `[EXPERT REVIEW NEEDED]` markers:

## Reference integrity

- [ ] References were checked for title/DOI/author/year consistency where possible
- [ ] Any DOI/title mismatch is documented
- [ ] Any questionable citation is marked for review
- [ ] Reference corrections are explained in the PR body

## Validation checklist

- [ ] I ran `python scripts/validate_skills.py path/to/SKILL.md` for each changed skill
- [ ] I ran `python scripts/generate_catalog.py`
- [ ] I ran `python scripts/check_crossrefs.py`
- [ ] I ran full repo validation if appropriate:
      `for f in $(find skills -name "SKILL.md" | sort); do python scripts/validate_skills.py "$f" || exit 1; done`
- [ ] No broken internal skill links were introduced
- [ ] Catalog was regenerated if skill metadata changed
- [ ] No local settings, cache files, temporary files, or worktree metadata are committed

## Maturity impact

- [ ] No maturity change
- [ ] Draft skill added
- [ ] Maturity promotion requested
- [ ] Maturity demotion or correction requested
- [ ] Domain review required before maturity change

## Cross-reference policy

- [ ] This PR links only to existing skills
- [ ] Future roadmap skills are mentioned in prose without hyperlinks
- [ ] Any new cross-links were checked manually

## Notes for reviewers

Mention domain-specific concerns, assumptions, unresolved expert-review items, or anything reviewers should inspect closely.
