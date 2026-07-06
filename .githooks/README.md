# Git hooks

Repo-managed git hooks. Enable them once per clone:

```bash
git config core.hooksPath .githooks
```

## `pre-push`

Runs the deterministic agent-skill lint
(`python -m dev_tools.agent.skills.checks --skills-root skills`) and blocks the
push if it finds anything, so the agent skills checked into GitHub stay clean.
It covers `skills/` and the eval suites under `dev_tools/agent/skill_evals/`.

Prerequisite: the `python3` on your `PATH` must have PyYAML installed (it is part
of the nvflare dev environment; otherwise `pip install pyyaml`). The hook probes
for it and fails with an actionable message if it is missing.

The same lint also runs in `./runtest.sh -s` and in the pre-merge CI unit tests
(`tests/unit_test/tool/agent_skill_checks/seed_skills_test.py`), so this hook is
a fast local pre-push gate rather than the only enforcement.

Emergency bypass: `git push --no-verify`.
