# Agent-skill lint

Do not configure Git to execute hooks from a tracked worktree directory.
Tracked hook code changes when branches are checked out and would then run with
the developer's credentials.

Run the deterministic agent-skill lint directly when changing skills or evals:

```bash
python -m dev_tools.agent.skills.checks --skills-root skills
```

The lint also runs in `./runtest.sh -s` and in the pre-merge CI unit tests
(`tests/unit_test/tool/agent_skill_checks/seed_skills_test.py`). Keep those
trusted style/CI gates as the enforcement points.
