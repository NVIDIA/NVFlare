# Skill Benchmark: nvflare-autofl

> ⚠️ **Overall verdict: INCOMPLETE — Required evidence is missing**

One or more required evaluation tiers did not complete, so this benchmark is not publication-complete.

## Evaluation Metadata

- Skill: `nvflare-autofl`
- Evaluation date: 2026-09-10
- Evaluator version: `1.5.6`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`), Codex (`openai/openai/gpt-5.5`)
- Tasks: 7 evaluation tasks (7 positive)
- Dataset digest: `sha256:47bb5f4f9de5f1a56ecb009c45deee42349c7c12a196a16f05072e36e82b36ec` (skill-evaluator-dataset-snapshot/1)
- Attempts per task: 3
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
| Overall | Not available | 74.2% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | Not available | 81.8% → 100.0% (+18.2 points) |
| Correctness | Not available | 67.3% → 72.5% (+5.2 points) |
| Discoverability | Not available | 63.8% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | Not available | 32.6% → 44.0% (+11.4 points) |
| Efficiency | Not available | 90.7% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 2,787,776 | 12,920,494 | N/A | N/A | skill 7/7; base 12/21 |
| claude-code | autofl-diversify-after-family-repeat | 199,145 | 237,320 | -38,175 | -16.09% | skill 1/1; base 1/1 |
| claude-code | autofl-global-negative-web-app | 1,332,620 | 1,103,231 | +229,389 | +20.79% | skill 1/1; base 1/1 |
| claude-code | autofl-literature-batch-before-tuning | 365,684 | 367,017 | N/A | N/A | skill 1/1; base 2/2 |
| claude-code | autofl-natural-phrasing-low-accuracy | 176,241 | 670,093 | N/A | N/A | skill 1/1; base 3/3 |
| claude-code | autofl-negative-diagnose-job | 287,173 | 186,391 | +100,782 | +54.07% | skill 1/1; base 1/1 |
| claude-code | autofl-negative-pytorch-conversion | 216,377 | 152,619 | +63,758 | +41.78% | skill 1/1; base 1/1 |
| claude-code | autofl-optimize-existing-job | 210,536 | 10,203,823 | N/A | N/A | skill 1/1; base 3/3 |
| codex | All cases | 954,809 | 2,244,651 | N/A | N/A | skill 8/8; base 11/11 |
| codex | autofl-diversify-after-family-repeat | 89,523 | 71,042 | +18,481 | +26.01% | skill 1/1; base 1/1 |
| codex | autofl-global-negative-web-app | 250,708 | 258,849 | -8,141 | -3.15% | skill 1/1; base 1/1 |
| codex | autofl-literature-batch-before-tuning | 85,720 | 43,608 | +42,112 | +96.57% | skill 1/1; base 1/1 |
| codex | autofl-natural-phrasing-low-accuracy | 115,796 | 314,398 | N/A | N/A | skill 1/1; base 3/3 |
| codex | autofl-negative-diagnose-job | 248,899 | 115,188 | N/A | N/A | skill 2/2; base 1/1 |
| codex | autofl-negative-pytorch-conversion | 100,970 | 69,411 | +31,559 | +45.47% | skill 1/1; base 1/1 |
| codex | autofl-optimize-existing-job | 63,193 | 1,372,155 | N/A | N/A | skill 1/1; base 3/3 |
| ALL AGENTS | Dataset aggregate | 3,742,585 | 15,165,145 | N/A | N/A | skill 15/15; base 23/32 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. N/A means the relevant trajectory counters were not available; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED** | 1 validator(s); 0 finding(s) |
| Tier 2 | Semantic deduplication | **NOT RUN** | No result was recorded |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 7 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- Schema & Repository Governance: Found skill manifest: SKILL.md
- AGENT_EVAL: Tier 3 evaluation complete: verdict PASS; best agent codex

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
