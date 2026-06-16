# Benchmark Summary

Status: draft/internal pending Milestone 11 runtime evaluation.

Skill version: 0.1.0
FLARE version: 2.8.0 minimum

## Initial Checks

| Check | Status | Notes |
| --- | --- | --- |
| Positive POC/production trigger | Draft | `diagnose-poc-component-not-authorized` covers failed POC job diagnosis. |
| Positive simulation trigger | Draft | `diagnose-simulation-import-error` covers local `python job.py` failure diagnosis. |
| Partial evidence trigger | Draft | `diagnose-partial-log-visibility` covers missing or denied site logs. |
| Retryable transfer trigger | Draft | `diagnose-transfer-progress-timeout` covers peer timeout with active transfer progress. |
| Adjacent negative trigger | Draft | PyTorch conversion routes to `nvflare-convert-pytorch`. |
| Job lifecycle negative trigger | Draft | Healthy job monitoring routes away from diagnosis. |
| Global negative trigger | Draft | Non-FLARE web-app prompt routes to no skill. |
| Mandatory behavior | Draft | Behavior IDs cover runtime mode selection, mode-specific evidence, bounded logs, pattern matching, partial evidence, retryable transfer timeouts, and recovery reporting. |
| Prohibited behavior | Draft | Behavior IDs prohibit mutation, private-key reads, unbounded logs, job CLI use for simulation-only failures, and confident causes with missing evidence. |
| Process evaluation | Draft | Metrics cover first-pass supported diagnosis, turns to supported diagnosis, bounded evidence collection, correction count, and unwanted actions. |

## Known Gaps

- Runtime agent-performance scoring has not been run yet.
- Runtime process scoring has not been run yet.
- The pattern catalog is intentionally compact and should grow from support and
  Auto-FL feedback.
- No helper scripts are included in the first release; evidence collection is
  skill-guided through existing CLI and local artifact inspection.
