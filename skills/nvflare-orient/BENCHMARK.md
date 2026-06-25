# Benchmark Summary

Status: draft/internal pending Milestone 11 runtime evaluation.

Skill version: 0.1.0
FLARE version: 2.8.0 minimum

## Initial Checks

| Check | Status | Notes |
| --- | --- | --- |
| Positive trigger | Draft | `orient-ambiguous-project` defines the initial routing prompt. |
| Adjacent negative trigger | Draft | PyTorch conversion routes to `nvflare-convert-pytorch`. |
| Diagnosis handoff trigger | Draft | Failed or suspicious FLARE jobs route to `nvflare-diagnose-job`. |
| Global negative trigger | Draft | Non-FLARE web-app prompt routes to no skill. |
| Mandatory behavior | Draft | Behavior IDs cover inspect-first, read-only routing, single lead skill, and diagnosis handoff. |
| Prohibited behavior | Draft | Behavior IDs prohibit file edits, production actions, and implementing diagnosis during routing. |
| Process evaluation | Draft | Metrics cover first-pass route correctness, turns to route, unwanted actions, and handoff clarity. |

## Known Gaps

- Runtime agent-performance scoring has not been run yet.
- Runtime process scoring has not been run yet.
- Orientation routing will need new adjacent negatives as more workflow skills
  are added.
