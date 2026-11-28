# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Section headers and intros for optional prompt parts (only included when content is non-empty).
SECTION_REFERENCE_CASES = """## Successful Examples (Reference Cases)
Below are some examples of similar tasks that were successfully completed.
Please use these as references to guide your thinking and approach to the current task:

"""
SECTION_PAST_SUCCESSES = """## Your Own Past Successes (Execution Patterns)
Here are examples of successful execution processes you've previously used on similar tasks.
Pay special attention to the step-by-step procedures and strategies, especially when encountering obstacles:

"""
SECTION_KEY_INSIGHTS = """## Key Insights from Related Tasks
The following are insights gathered during the execution of similar tasks. You may refer to them during your task execution to improve problem-solving accuracy.

"""
SECTION_HOW_TO_USE_SEARCH = """## How to Use Your Intermediate Findings
- Use observations to update your current hypothesis (what to try next), not to assume the whole world is static.
- Prefer relational cues and reusable patterns (e.g., “target objects often appear on shelves/basins”, “open before placing into a closed receptacle”) over memorizing exact coordinates or one-off locations.
- If an action returns “Nothing happens.”, do not repeat it;
- When you identify a plausible next step, execute it and re-check the new observation to stay aligned with the current episode.

"""
SECTION_HOW_TO_USE_FEVER = """## How to Use Your Intermediate Findings
- Use retrieved memories as evidence-search guidance, not as label priors.
- Prefer a focused Search[...] from the claim, then Lookup[...] key entities or relations in the returned evidence.
- At the final decision, compare the claim against the current evidence: use Finish[SUPPORTS] only for direct entailment, Finish[REFUTES] only for a clear contradiction, and Finish[NOT ENOUGH INFO] when the evidence is missing or ambiguous.
- If a lookup returns No Results, reformulate the evidence query instead of repeating the same lookup.

"""
SECTION_HOW_TO_USE_PDDL = """## How to Use Your Intermediate Findings
- Treat retrieved memories as planning hints over predicates, operators, and goal progress.
- Prefer currently valid actions that move unsatisfied goal literals closer to completion.
- Do not copy object names, operator arguments, or actions that are not available in the current problem.
- If an action fails or stalls, choose a different valid operator rather than repeating the same step.

"""
SECTION_HOW_TO_USE_MTMIND2WEB = """## How to Use Your Intermediate Findings
- Treat retrieved trajectories as reusable interaction patterns on websites (search → filter → sort → open detail → add to cart / book / submit), not as exact HTML.
- Prefer reusing successful action order and control types (textbox vs button vs select) over copying unrelated site-specific wording.
- When the task gives numbered candidate actions, you must only pick indices from that list; do not invent new element labels.
- If memory and the current candidate list conflict, follow the current candidate list and output format rules.

"""
SECTION_YOUR_TURN = """## Your Turn: Take Action!
Use the above examples and insights as a foundation, and now work on the following task:
{task_description}
"""
SEP = "\n---\n\n"

task_format: str = """
### Task description:
{task_description}

### Key steps:
{key_steps}

### Detailed trajectory:
{trajectory}
"""

temp = """NOTE: You must use the command `think: <your thoughts here>` if you want to think!!!
    - Right output: think: To solve the task, ...
    - Wrong output: To solve the task, ... """


def _strip_textworld_banner(text: str) -> str:
    """
    Remove TextWorld/ALFRED welcome banner lines from prompts.
    This is a display/input cleanup only; it should not affect memory storage.
    """
    if not text:
        return text
    lines = str(text).splitlines()
    banned = {"-= Welcome to TextWorld, ALFRED! =-"}

    cleaned: list[str] = []
    last_nonempty: str | None = None
    for ln in lines:
        s = ln.strip()
        if s in banned:
            continue
        # Drop repeated non-empty lines even if separated only by blank lines.
        if s and last_nonempty is not None and s == last_nonempty:
            continue
        cleaned.append(ln)
        if s:
            last_nonempty = s

    # Keep original line breaks as much as possible.
    return "\n".join(cleaned).strip("\n")


def format_task_prompt_with_insights(
    few_shots: list[str],
    memory_few_shots: list[str],
    insights: list[str],
    task_description: str,
    *,
    intermediate_findings_env: str | None = None,
) -> str:
    """Build task prompt; include each of the three reference sections only when it has content."""
    parts: list[str] = []
    env_name = (intermediate_findings_env or "").strip().lower()

    if few_shots:
        cleaned_few_shots = [_strip_textworld_banner(s) for s in few_shots if str(s).strip()]
        # Keep reference few-shots small and stable (match g-memory default style).
        cleaned_few_shots = cleaned_few_shots[:1]
        few_shots_text = "\n\n".join([f"Task {i+1}:\n{shot}" for i, shot in enumerate(cleaned_few_shots)])
        parts.append(SECTION_REFERENCE_CASES + few_shots_text.strip())

    if memory_few_shots:
        cleaned_memory_shots = [_strip_textworld_banner(s) for s in memory_few_shots if str(s).strip()]
        memory_text = "\n\n".join([f"Task {i+1}:\n{shot}" for i, shot in enumerate(cleaned_memory_shots)])
        parts.append(SECTION_PAST_SUCCESSES + memory_text.strip())

    if insights:
        # Drop LocalGraphHint from injected insights (keep other signals like [GGCommon], [MergedPath], etc.)
        filtered = [
            r
            for r in insights
            if r is not None and str(r).strip() and "[LocalGraphHint]" not in str(r) and "LocalGraphHint" not in str(r)
        ]
        insights_text = "\n".join([f"{i}. {r}" for i, r in enumerate(filtered, 1)])
        parts.append(SECTION_KEY_INSIGHTS + insights_text.strip())

    if parts:
        body = SEP.join(parts) + "\n\n"
    else:
        body = ""

    if env_name == "mtmind2web":
        how_section = SECTION_HOW_TO_USE_MTMIND2WEB
    elif env_name == "fever":
        how_section = SECTION_HOW_TO_USE_FEVER
    elif env_name.startswith("pddl"):
        how_section = SECTION_HOW_TO_USE_PDDL
    else:
        how_section = SECTION_HOW_TO_USE_SEARCH
    user_prompt = (
        body + how_section + SECTION_YOUR_TURN.format(task_description=_strip_textworld_banner(task_description))
    )
    return user_prompt


def format_task_context(task_description: str, task_traj: str, key_steps: str = None, source: str | None = None) -> str:
    source_line = f"Source: {source}\n" if source else ""

    td = _strip_textworld_banner(source_line + (task_description or ""))
    ks = _strip_textworld_banner(key_steps or "")
    traj = _strip_textworld_banner(task_traj or "")

    parts: list[str] = []
    parts.append("### Task description:   \n" + td.strip())
    if ks.strip():
        parts.append("### Key steps:\n" + ks.strip())
    if traj.strip():
        parts.append("### Detailed trajectory:\n" + traj.strip())
    return "\n\n".join(parts).strip()


def format_memory_augmentation_parts(
    memory_few_shots: list[str] | None,
    insights: list[str] | None,
) -> dict[str, str]:
    """
    Return structured prompt parts for audit / offline analysis.

    This is used by AutoGen's memory augmentation logger to dump what was injected into
    the solver prompt (successful reference cases + insights). It intentionally avoids
    including the full task_description to keep logs focused and smaller.
    """
    mem = [str(s) for s in (memory_few_shots or []) if s is not None and str(s).strip()]
    ins = [str(s) for s in (insights or []) if s is not None and str(s).strip()]

    mem_text = "\n\n".join([_strip_textworld_banner(s) for s in mem]).strip()
    ins_text = "\n".join([_strip_textworld_banner(s) for s in ins]).strip()

    return {
        "memory_few_shots": mem_text,
        "insights": ins_text,
    }


def format_prompt_payload(
    payload: dict[str, list[str]] | None,
    task_description: str,
    *,
    intermediate_findings_env: str | None = None,
) -> str:
    payload = payload or {}
    reference_cases = [str(item) for item in payload.get("reference_cases", []) if str(item).strip()]
    execution_patterns = [str(item) for item in payload.get("execution_patterns", []) if str(item).strip()]
    insights = [str(item) for item in payload.get("insights", []) if str(item).strip()]
    planner_notes = [str(item) for item in payload.get("planner_notes", []) if str(item).strip()]
    action_constraints = [str(item) for item in payload.get("action_constraints", []) if str(item).strip()]
    repair_hints = [str(item) for item in payload.get("repair_hints", []) if str(item).strip()]

    combined_insights = insights + planner_notes + action_constraints + repair_hints
    return format_task_prompt_with_insights(
        few_shots=reference_cases,
        memory_few_shots=execution_patterns,
        insights=combined_insights,
        task_description=task_description,
        intermediate_findings_env=intermediate_findings_env,
    )
