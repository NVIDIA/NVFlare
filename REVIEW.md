# Code Review Guidelines

Read [CLAUDE.md](CLAUDE.md) and applicable directory-level instructions for project context,
and [AGENTS.md](AGENTS.md) for agent rules. PR requirements and human approvals follow
[CONTRIBUTING.md](CONTRIBUTING.md#the-code-reviewing-process).

## Review bar

- Read the diff against its base revision, the stated intent, and affected callers and tests.
  For bug fixes, check that the change addresses the root cause across affected paths.
- Report concrete defects introduced or exposed by the change. Verify each candidate finding
  against surrounding code in a separate pass; require a trigger, consequence, and file/line
  evidence. Names alone are not evidence. Separate unrelated pre-existing bugs from PR findings.
- Check relevant `CLAUDE.md` conventions and whether the change makes existing guidance stale.
- Check the linked GitHub issue and CI status per `CONTRIBUTING.md`; disclose unavailable context.
- Scale validation to impact: documentation/link checks for prose, targeted behavior tests for
  code, and deeper failure-path and security checks for distributed execution or sensitive changes.
  Auth, provisioning, wire/persisted formats, and CI/release changes warrant human code-owner scrutiny.
  AI review does not replace required human approvals.
- Before handing off an authored change, run relevant checks and review the diff. Record actual
  commands, results, and validation gaps in the PR template. Reviewers with execution tools should
  use focused tests or reproductions; otherwise state the limits of static review.

## What failure looks like in NVFlare

| Area | Look for |
| --- | --- |
| Federated execution | Tasks hang, results are lost or counted twice, or rounds diverge after late replies, disconnects, timeouts, or aborts. |
| Data and aggregation | `Shareable`, DXO, or `FLContext` contracts break; return codes or metadata are lost; weights or model differences produce incorrect updates. |
| Concurrency and cleanup | Lock ordering deadlocks, shared state races, or threads/processes/resources survive failure or cancellation. |
| Security and privacy | Auth checks are bypassed, untrusted data reaches unsafe deserialization/path/command handling, or private data leaks through logs or results. |
| APIs and configuration | Public APIs, CLI options, job configs, or saved state break; config routes tasks or loads components incorrectly; optional dependencies become mandatory. |
| Docs and examples | Instructions contradict behavior, commands fail, or examples configure an invalid job. On `main`, upcoming-release pins can be intentional; follow `AGENTS.md`. |

## Noise and skipped paths

- Style-only nit budget: zero. Leave formatting/import findings already enforced by CI to those
  checks. Avoid speculative refactors and missing-test comments without a concrete behavioral risk.
- Skip routine line-by-line review of `**/*_pb2.py` and `**/*_pb2_grpc.py`; review the `.proto`
  source, regeneration consistency, runtime compatibility, and unexpected manual changes instead.
- Skip local output in `build/`, `dist/`, `docs/_build/`, `**/__pycache__/`, and `.venv/`.
  Investigate if a change unexpectedly tracks or ships these artifacts.
- Recognize intentional invalid inputs in negative tests; verify the expected rejection behavior.

## Report

Start the summary with a finding count, for example `2 findings` or `0 findings — nothing to fix`.
Then state scope, checks/results, and remaining uncertainty. A clean review is not proof of correctness.

Order findings by severity: **P0** catastrophic and unconditional; **P1** serious failure in a
supported scenario, fix before merge; **P2** concrete defect with narrower impact; **P3** minor defect.
If the review tool uses its own severity labels, preserve that format and apply the same impact bar.

Each finding needs a short title, a precise file/line location, and the trigger, faulty behavior,
and consequence. Consolidate repeated instances by root cause and identify other affected locations.
Keep unresolved questions separate from verified findings.

## Re-review and learning

- Verify earlier findings are fixed or have a reasoned resolution; do not repeat resolved findings.
  Focus subsequent passes on new changes and their interactions with affected code.
- After three review/fix rounds, summarize remaining blockers and ask the author to reconsider
  the approach; avoid continuing an automatic fix/review loop. Keep unresolved defects visible.
- Turn recurring feedback into a focused instruction or automated check. Use confirmed incidents,
  reverts, and clean changes to assess reviewer accuracy and noise when evolving these rules.

Adapted for NVFlare from [Anthropic's review workflow](https://claude.dev/blog/solving-code-review-at-anthropic/)
and [review customization guidance](https://code.claude.com/docs/en/code-review-customize).
