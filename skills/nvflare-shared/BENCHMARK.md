# Skill Benchmark: nvflare-shared

> ⚠️ **Overall verdict: INCOMPLETE — Required evidence is missing**

One or more required evaluation tiers did not complete, so this benchmark is not publication-complete.

## Evaluation Metadata

- Skill: `nvflare-shared`
- Evaluation date: 2026-09-04
- Evaluator version: `1.5.4`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`), Codex (`openai/openai/gpt-5.5`)
- Tasks: 3 evaluation tasks (3 positive)
- Dataset digest: `sha256:0267fb75cdd17b87a2fc8e8a481b55058ed90097b92f97f5a9c8dba18724d58e` (skill-evaluator-dataset-snapshot/1)
- Attempts per task: 1
- Environment: `k8s-sandbox`
- Tier 2 evidence: required for publication
- Tier 3 evidence: required for publication

Each task attempt ran in its own isolated sandbox pod.

## What This Report Answers

The three-tier evaluation checks whether the skill:

- is safe to use;
- produces correct answers;
- is discovered and activated when needed;
- helps the agent complete the user's goal and expected workflow; and
- avoids wasted skill and tool usage.

## Results at a Glance

| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 83.4% — baseline ran, but no comparable score was available; uplift unavailable | 82.2% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 83.3% → 100.0% (+16.7 points) | 66.7% → 100.0% (+33.3 points) |
| Correctness | 100.0% → 100.0% (±0.0 points) | 93.3% → 86.7% (-6.6 points) |
| Discoverability | 71.7% — baseline ran, but no comparable score was available; uplift unavailable | 66.7% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 60.8% → 66.7% (+5.9 points) | 70.0% → 70.0% (±0.0 points) |
| Efficiency | 78.8% — baseline ran, but no comparable score was available; uplift unavailable | 87.4% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 1,660,838 | 1,755,501 | -94,663 | -5.39% | skill 3/3; base 3/3 |
| claude-code | shared-global-negative-web-app | 1,196,294 | 1,093,366 | +102,928 | +9.41% | skill 1/1; base 1/1 |
| claude-code | shared-negative-direct-conversion | 125,826 | 441,516 | -315,690 | -71.50% | skill 1/1; base 1/1 |
| claude-code | shared-recover-incomplete-installation | 338,718 | 220,619 | +118,099 | +53.53% | skill 1/1; base 1/1 |
| codex | All cases | 561,433 | 1,901,568 | -1,340,135 | -70.48% | skill 3/3; base 3/3 |
| codex | shared-global-negative-web-app | 273,946 | 203,207 | +70,739 | +34.81% | skill 1/1; base 1/1 |
| codex | shared-negative-direct-conversion | 158,117 | 1,569,165 | -1,411,048 | -89.92% | skill 1/1; base 1/1 |
| codex | shared-recover-incomplete-installation | 129,370 | 129,196 | +174 | +0.13% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 2,222,271 | 3,657,069 | -1,434,798 | -39.23% | skill 6/6; base 6/6 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. N/A means the relevant trajectory counters were not available; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED** | 1 validator(s); 0 finding(s) |
| Tier 2 | Semantic deduplication | **NOT RUN** | No result was recorded |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 3 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- Schema & Repository Governance: Found skill manifest: SKILL.md
- AGENT_EVAL: Tier 3 evaluation complete: verdict PASS; best agent claude-code

</details>

## Scoring Methodology

<details>
<summary>Show dimension definitions, source signals, and thresholds</summary>

| Dimension | Question | Scored signals |
|---|---|---|
| Security | Is it safe to use? | `security` (100%) |
| Correctness | Is the answer correct? | `accuracy` (100%) |
| Discoverability | Was the right skill loaded when needed? | `skill_execution` (100%) |
| Effectiveness | Did the skill help complete the task? | `goal_accuracy` (50%) + `behavior_check` (50%) |
| Efficiency | Did it avoid wasted tool calls and token usage? | `skill_efficiency` (50%) + `token_efficiency` (50%) |

- Dimension bands: PASS at 50% or above; NEUTRAL from 40% to below 50%; FAIL below 40%.
- Overall Tier 3 lift: PASS at +5 points or more; FAIL at -10 points or less; values between those bands are NEUTRAL.
- Overall verdict: PASS only when every configured dimension passes for at least one supported agent. Lift is reported as diagnostic evidence and does not override this gate.
- The 50% attempt pass threshold is a separate per-task gate; it is not the dimension pass threshold.
- Effectiveness is the equal-weight mean of goal completion (`goal_accuracy`) and expected workflow adherence (`behavior_check`).
- Efficiency is 50% tool-call productivity (the backward-compatible `skill_efficiency` wire id) and 50% `token_efficiency`. Positive-case skill routing is scored under Discoverability, not Efficiency; a negative case without a routing target is N/A. N/A sources are omitted, remaining weights are renormalized, and the dimension is marked partial.

Signals present in this run:

- `security` (Security): unsafe operations, secret leakage, and unauthorized access.
- `skill_execution` (Skill Execution): whether the expected skill was selected, decoys were avoided, and the workflow executed.
- `skill_efficiency` (Tool Productivity): tool-call productivity (legacy wire id; routing is scored under Discoverability).
- `accuracy` (Accuracy): final-answer correctness against the reference answer.
- `goal_accuracy` (Goal Accuracy): whether the user's goal was achieved.
- `behavior_check` (Behavior Check): whether the expected workflow behavior was followed.
- `token_efficiency` (Token Efficiency): actual uncached prompt plus completion usage (50% of Efficiency).

</details>

## Freshness

Regenerate this benchmark when the skill, evaluation dataset, target agent/model, evaluator version, environment, or scoring policy changes.
