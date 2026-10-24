# Skill Benchmark: nvflare-autofl-report

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `nvflare-autofl-report`
- Evaluation date: 2026-09-15
- Evaluator version: `1.5.6`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`), Codex (`openai/openai/gpt-5.5`)
- Tasks: 4 evaluation tasks (4 positive)
- Dataset digest: `sha256:c671db2ce45dc278f5a5da89828e94521cd5fc52f73156f263302bb4d43c6e43` (skill-evaluator-dataset-snapshot/1)
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
| Overall | 88.2% — baseline ran, but no comparable score was available; uplift unavailable | 62.3% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 88.9% → 100.0% (+11.1 points) | 70.0% → 85.7% (+15.7 points) |
| Correctness | 22.2% → 100.0% (+77.8 points) | 20.0% → 45.7% (+25.7 points) |
| Discoverability | 83.3% — baseline ran, but no comparable score was available; uplift unavailable | 56.4% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 31.1% → 66.3% (+35.2 points) | 34.0% → 28.6% (-5.4 points) |
| Efficiency | 91.2% — baseline ran, but no comparable score was available; uplift unavailable | 94.9% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 983,436 | 1,816,856 | N/A | N/A | skill 4/4; base 9/9 |
| claude-code | autofl-report-active-negative | 271,228 | 381,644 | N/A | N/A | skill 1/1; base 2/2 |
| claude-code | autofl-report-global-negative | 119,491 | 119,024 | +467 | +0.39% | skill 1/1; base 1/1 |
| claude-code | autofl-report-optimize-negative | 323,592 | 638,690 | N/A | N/A | skill 1/1; base 3/3 |
| claude-code | autofl-report-stopped-campaign | 269,125 | 677,498 | N/A | N/A | skill 1/1; base 3/3 |
| codex | All cases | 662,338 | 1,814,050 | N/A | N/A | skill 7/7; base 10/10 |
| codex | autofl-report-active-negative | 205,649 | 372,043 | N/A | N/A | skill 1/1; base 3/3 |
| codex | autofl-report-global-negative | 40,809 | 40,769 | +40 | +0.10% | skill 1/1; base 1/1 |
| codex | autofl-report-optimize-negative | 254,585 | 1,156,474 | -901,889 | -77.99% | skill 3/3; base 3/3 |
| codex | autofl-report-stopped-campaign | 161,295 | 244,764 | N/A | N/A | skill 2/2; base 3/3 |
| ALL AGENTS | Dataset aggregate | 1,645,774 | 3,630,906 | N/A | N/A | skill 11/11; base 19/19 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. N/A means the relevant trajectory counters were not available; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 11 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 4 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **MEDIUM** SCHEMA/body_recommended_section: Missing recommended section: '## Instructions' (`skills/nvflare-autofl-report/SKILL.md`)
- **MEDIUM** SCHEMA/body_recommended_section: Missing recommended section: '## Examples' (`skills/nvflare-autofl-report/SKILL.md`)
- **MEDIUM** SECURITY/Skill Enumeration (AS3): Agent Snooping: skills/nvflare-autofl-report/SKILL.md (`BENCHMARK.md:72`)
- **MEDIUM** SECURITY/Skill Enumeration (AS3): Agent Snooping: skills/nvflare-autofl-report/SKILL.md (`BENCHMARK.md:73`)
- **MEDIUM** SECURITY/Unknown (LP3): MCP Least Privilege: The skill declares no explicit tool scope (no 'permissions' or 'allowed-tools' field in its metadata), yet the skill's w (`SKILL.md:1`)
- 6 additional finding(s) are available in the full evaluation artifacts.

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
