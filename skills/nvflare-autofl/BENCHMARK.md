# Benchmark Summary

Status: draft/internal pending runtime evaluation.

Skill version: 0.1.0
FLARE version: 2.8.0 minimum

## Initial Checks

| Check | Status | Notes |
| --- | --- | --- |
| Positive trigger | Draft | `autofl-optimize-existing-job` defines the primary Auto-FL prompt. |
| Adjacent negative trigger | Draft | PyTorch conversion routes to `nvflare-convert-pytorch`. |
| Diagnosis negative trigger | Draft | Failed-job diagnosis routes to `nvflare-diagnose-job`. |
| Global negative trigger | Draft | Non-FLARE prompts route to no skill. |
| Mandatory behavior | Draft | Behavior IDs cover deterministic import, campaign summary, bounded edits, and existing FLARE execution. |
| Prohibited behavior | Draft | Behavior IDs prohibit bypassing policy, editing outside allowed paths, and treating `autofl.yaml` as an exported job. |
| Process evaluation | Draft | Metrics cover import-first behavior, contract preservation, score extraction, and unwanted production actions. |

## Known Gaps

- Runtime agent-performance scoring has not been run yet.
- Adjacent negatives should be expanded after more NVFLARE workflow skills land.
- Production execution behavior needs site-policy fixture coverage before the skill graduates from draft status.
