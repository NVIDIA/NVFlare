# NVFlare Agent Notes

- To trigger CI/CD from a PR review thread, post a single-line comment exactly: `/build`.
- Prefer `rg` and `rg --files` for fast codebase search.
- Keep edits scoped to the task; do not modify unrelated files in a dirty worktree.
- Start with targeted tests for changed files, then run broader checks as needed.
- Before pushing or opening a PR, run the project style check (`./runtest.sh -s`, or the closest scoped equivalent when justified). If it cannot be run, state that clearly before pushing.
- Read `CLAUDE.md` for shared repo guidance such as project overview, architecture notes, commands, package layout, and style/testing conventions. Keep `AGENTS.md` limited to agent-specific addenda.

## Public Pull Requests

- When creating a PR that should be included in the 2.9 release, assign the `2.9` milestone.
- Keep public GitHub PRs, issues, and comments self-contained. Do not include private Jira URLs or ticket identifiers; summarize the relevant requirements and context instead.
- For internal traceability, link from the Jira ticket to the public GitHub artifact, not from the public artifact to Jira.

## Main Branch Versioning

- Treat `main` as the development branch for the next NVFlare release.
- On `main`, example `requirements.txt` files may intentionally pin the first upcoming NVFlare version that supports a new feature, even if that package is not published on PyPI yet.
- Do not change those pins back to the latest stable release just to make `pip install -r requirements.txt` succeed.
- The requirements pin is sufficient. Do not add temporary main-branch or install-from-source caveats solely because the pinned version is not published yet.
