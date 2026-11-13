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

from __future__ import annotations

import re
from typing import Any


class BasePromptStyle:
    """Dataset-facing language and transfer boundary for MemCo prompt injection."""

    name = "alfworld"

    def phase_items(self, renderer: Any, *, query: Any, target: str, tool: str, destination: str) -> list[str]:
        progress = str(getattr(query, "progress_state", "") or "")
        held = int(getattr(query, "held_relevant_count", 0) or 0)
        visible = bool(getattr(query, "goal_object_matches_visible", False))
        is_two_object = bool(renderer._memco_is_two_object_task(query))
        items: list[str] = []
        if held > 0:
            if tool and progress in {"carry_target", "process_target"}:
                items.append(
                    f"held target_object={target}; next phase is process with tool={tool} "
                    f"if not already processed, otherwise deliver to destination={destination}."
                )
            else:
                items.append(
                    f"held target_object={target}; next phase is delivery to destination={destination}, not more source search."
                )
        elif visible:
            items.append(f"target_object={target} is visible; take the matching target before broad search.")
        elif progress.startswith("search"):
            items.append(
                f"search target_object={target}; if graph priority gives an admissible queue, execute the next queued action."
            )
        if is_two_object:
            items.append(
                "two-object policy: finish one target's full required workflow first, including any required processing, "
                "delivery, and put action; only then start another target."
            )
        return items

    def keep_global_text(self, renderer: Any, *, text: str, norm_text: str, task_family: str) -> bool:
        return True

    def keep_local_artifact_text(
        self,
        renderer: Any,
        *,
        domain: str,
        task_family: str,
        text: str,
        norm_text: str,
    ) -> bool:
        return True

    def format_memory_line(self, renderer: Any, *, query: Any, slot: str, text: str) -> str:
        return text

    def phase_label(self, renderer: Any, *, query: Any, macro: str) -> str:
        labels = {
            "process_held_target": "process held target",
            "deliver_held_target": "deliver held target",
            "acquire_visible_target": "acquire visible target",
            "search_additional_target": "search additional target",
            "search_target": "search target",
            "continue_task": "continue task",
        }
        return labels.get(macro, macro.replace("_", " "))

    def state_summary(
        self,
        renderer: Any,
        *,
        query: Any,
        target: str,
        tool: str,
        destination: str,
        held: list[str],
        visible: list[str],
        exhausted: list[str],
    ) -> str:
        parts = [
            f"target={target or 'unknown'}",
            f"tool={tool or 'none'}",
            f"destination={destination or 'unknown'}",
        ]
        if renderer._memco_is_two_object_task(query):
            parts.append("two_object=finish one full required workflow before starting another")
        if held:
            parts.append("held=" + ", ".join(renderer._memco_base(x) for x in held[:2] if str(x).strip()))
        elif visible:
            visible_bases = [renderer._memco_base(x) for x in visible[:3] if str(x).strip()]
            if visible_bases:
                parts.append("visible=" + ", ".join(visible_bases))
        if exhausted:
            counts = renderer._memco_exhausted_base_counts(exhausted)
            if counts:
                top = ", ".join(
                    f"{base}x{count}" for base, count in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:2]
                )
                parts.append("searched=" + top)
        return "; ".join(part for part in parts if part)

    def next_priority_line(
        self,
        renderer: Any,
        *,
        query: Any,
        priority_items: list[str],
        admissible: list[str],
    ) -> str:
        goal_roles = getattr(query, "goal_roles", {}) or {}
        target = renderer._memco_base(str(goal_roles.get("object", "") or ""))
        tool = renderer._memco_base(str(goal_roles.get("tool", "") or ""))
        destination = renderer._memco_base(str(goal_roles.get("destination", "") or ""))
        progress = str(getattr(query, "progress_state", "") or "")
        held_count = int(getattr(query, "held_relevant_count", 0) or 0)
        visible_match = bool(getattr(query, "goal_object_matches_visible", False))
        is_two_object = bool(renderer._memco_is_two_object_task(query))

        if held_count > 0:
            actions: list[str] = []
            if tool and progress in {"carry_target", "process_target"}:
                actions = renderer._memco_tool_priority_actions(target=target, tool=tool, admissible=admissible)
            if not actions and destination:
                actions = renderer._memco_destination_priority_actions(
                    target=target, destination=destination, admissible=admissible
                )
            if actions:
                return actions[0]
            if is_two_object:
                return "continue the held target's required workflow now; do not search or take another target until it is processed if needed and put at destination."
            return "use the admissible process or delivery action for the held target; ignore source search."

        if visible_match:
            actions = renderer._memco_take_priority_actions(target=target, admissible=admissible)
            if actions:
                return actions[0]
            if is_two_object:
                return "take one visible target, then finish its required processing and delivery before searching another."
            return "take the visible matching target before broad search."

        for item in priority_items[:2]:
            text = str(item or "").strip()
            if text:
                return text
        if progress.startswith("search"):
            if is_two_object:
                return "start one target's full required workflow; after taking it, finish any required processing and delivery before searching another."
            return "search for the target using current admissible actions; avoid repeating exhausted source types."
        return "follow the current observation and admissible actions."


class ALFWorldPromptStyle(BasePromptStyle):
    name = "alfworld"


class PDDLPromptStyle(BasePromptStyle):
    name = "pddl"

    def phase_items(self, renderer: Any, *, query: Any, target: str, tool: str, destination: str) -> list[str]:
        progress = str(getattr(query, "progress_state", "") or "")
        return [
            f"PDDL graph stage={progress or 'unknown'}; prefer current valid actions that previous local/global traces linked to goal-literal progress."
        ]

    def phase_label(self, renderer: Any, *, query: Any, macro: str) -> str:
        progress = str(getattr(query, "progress_state", "") or "")
        labels = {
            "need_plan": "choose valid operator",
            "goal_progress": "advance goal literals",
            "invalid_action": "recover from invalid operator",
            "done": "finished",
        }
        return labels.get(progress, progress.replace("_", " ") or "solve planning goal")

    def state_summary(
        self,
        renderer: Any,
        *,
        query: Any,
        target: str,
        tool: str,
        destination: str,
        held: list[str],
        visible: list[str],
        exhausted: list[str],
    ) -> str:
        goal = renderer._memco_clean(str(getattr(query, "goal", "") or "")).strip()
        progress = str(getattr(query, "progress_state", "") or "unknown")
        belief = getattr(query, "belief", {}) or {}
        goal_literals = [str(item) for item in (belief.get("goal_literals") or []) if str(item).strip()]
        current_literals = {renderer._memco_norm(str(item)) for item in (belief.get("current_literals") or [])}
        unsatisfied = [literal for literal in goal_literals if renderer._memco_norm(literal) not in current_literals]
        parts = [f"goal={renderer._memco_shorten(goal, 120) or 'unknown'}", f"stage={progress}"]
        if unsatisfied:
            parts.append("unsatisfied=" + "; ".join(renderer._memco_shorten(lit, 50) for lit in unsatisfied[:2]))
        if visible:
            parts.append("state=" + "; ".join(renderer._memco_base(x) for x in visible[:3] if str(x).strip()))
        if exhausted:
            parts.append("tried=" + ", ".join(renderer._memco_base(x) for x in exhausted[:3] if str(x).strip()))
        return "; ".join(parts)

    def next_priority_line(
        self,
        renderer: Any,
        *,
        query: Any,
        priority_items: list[str],
        admissible: list[str],
    ) -> str:
        progress = str(getattr(query, "progress_state", "") or "")
        setup_action = ""
        for item in priority_items[:3]:
            text = str(item or "").strip()
            mapped = renderer._memco_first_admissible_action_in_text(text, admissible)
            if (
                mapped
                and not renderer._memco_pddl_is_meta_action(mapped)
                and renderer._memco_pddl_action_advances_unsatisfied_goal(query, mapped)
            ):
                return f"execute current valid operator `{mapped}` only if it advances an unsatisfied goal literal."
            if (
                mapped
                and progress == "search_preconditions"
                and not renderer._memco_pddl_is_meta_action(mapped)
                and renderer._memco_pddl_action_is_safe_setup(query, mapped)
            ):
                setup_action = setup_action or mapped
        hint = renderer._memco_pddl_current_action_hint(query, admissible)
        if hint:
            return f"prefer current valid operator `{hint}` because it directly achieves an unsatisfied goal literal; do not invent actions."
        if setup_action:
            if bool(getattr(renderer, "_memco_is_gpt4omini_model", lambda: False)()):
                return "choose a currently valid non-meta operator from the admissible list that advances unsatisfied goal literals; do not spend turns on meta-actions unless no task operator is valid."
            return f"execute memory-grounded setup operator `{setup_action}` only if it prepares a remaining goal precondition; do not invent actions."
        if admissible:
            return "choose a currently valid non-meta operator that advances unsatisfied goal literals; do not invent actions or loop on meta-actions."
        return "continue planning from current predicates and goal literals."


class FeverPromptStyle(BasePromptStyle):
    name = "fever"

    def phase_items(self, renderer: Any, *, query: Any, target: str, tool: str, destination: str) -> list[str]:
        progress = str(getattr(query, "progress_state", "") or "")
        ctx = _fever_claim_context(query)
        if progress == "ready_finish":
            return [
                f"FEVER label decision stage ({ctx['claim_type']}): compare only the current factual evidence against this claim; action names, prior searches, and tool misses are not evidence."
            ]
        if progress in {"search_failed", "invalid_action"}:
            variants = ctx.get("query_variants", []) or []
            route = (
                "; ".join(f"Search[{variant}]" for variant in variants[:3])
                if variants
                else "a relation/type query from the current claim"
            )
            return [
                "FEVER failed-search recovery: one tool miss is not evidence for NOT ENOUGH INFO; "
                f"try an untried current-claim query first ({route})."
            ]
        return [
            f"FEVER graph stage={progress or 'unknown'}, claim_type={ctx['claim_type']}; use memory only as a query/evidence gate, not as a label or stop-rule prior."
        ]

    def keep_global_text(self, renderer: Any, *, text: str, norm_text: str, task_family: str) -> bool:
        return _is_fever_transferable_hint(norm_text)

    def keep_local_artifact_text(
        self,
        renderer: Any,
        *,
        domain: str,
        task_family: str,
        text: str,
        norm_text: str,
    ) -> bool:
        return _is_fever_transferable_hint(norm_text)

    def phase_label(self, renderer: Any, *, query: Any, macro: str) -> str:
        progress = str(getattr(query, "progress_state", "") or "")
        labels = {
            "need_search": "choose focused evidence search",
            "need_lookup_or_finish": "inspect evidence or decide label",
            "ready_finish": "decide FEVER label from evidence",
            "done": "finished",
            "search_failed": "recover from failed evidence search",
            "invalid_action": "recover from invalid FEVER action",
        }
        return labels.get(progress, progress.replace("_", " ") or "verify claim")

    def state_summary(
        self,
        renderer: Any,
        *,
        query: Any,
        target: str,
        tool: str,
        destination: str,
        held: list[str],
        visible: list[str],
        exhausted: list[str],
    ) -> str:
        goal = renderer._memco_clean(str(getattr(query, "goal", "") or "")).removeprefix("Verify claim:").strip()
        progress = str(getattr(query, "progress_state", "") or "unknown")
        ctx = _fever_claim_context(query)
        evidence = [renderer._memco_base(x) for x in visible[:2] if str(x).strip()]
        parts = [f"claim={renderer._memco_shorten(goal, 110) or 'unknown'}", f"stage={progress}"]
        parts.append(f"claim_type={ctx['claim_type']}")
        if ctx["entity"]:
            parts.append(f"primary_entity={ctx['entity']}")
        if ctx.get("secondary_entity"):
            parts.append(f"relation_entity={ctx['secondary_entity']}")
        if ctx["lookup_keywords"]:
            parts.append("lookup_keywords=" + ", ".join(ctx["lookup_keywords"][:3]))
        if ctx.get("query_variants"):
            parts.append("query_variants=" + " | ".join(ctx["query_variants"][:3]))
        if evidence:
            parts.append("evidence=" + "; ".join(evidence))
        if exhausted:
            parts.append("searched=" + ", ".join(renderer._memco_base(x) for x in exhausted[:2] if str(x).strip()))
        return "; ".join(parts)

    def next_priority_line(
        self,
        renderer: Any,
        *,
        query: Any,
        priority_items: list[str],
        admissible: list[str],
    ) -> str:
        progress = str(getattr(query, "progress_state", "") or "")
        for item in priority_items[:2]:
            text = str(item or "").strip()
            if text:
                return text
        ctx = _fever_claim_context(query)
        lookup_hint = _fever_lookup_hint(renderer, query)
        search_hint = _fever_search_hint(renderer, query, admissible)
        if progress == "need_search":
            if search_hint:
                return f"claim-type workflow={ctx['claim_type']}; start with {search_hint}; prefer relation/entity2 query over primary-only search when available."
            return f"claim-type workflow={ctx['claim_type']}; choose a focused relation/type Search[...] from the current claim before any label."
        if progress == "need_lookup_or_finish":
            if lookup_hint:
                return f"claim-type workflow={ctx['claim_type']}; try Lookup[{lookup_hint}] only if the current page is the right subject; otherwise use a relation query."
            return "Finish only when the current page is the right subject and the relation slot is checked; otherwise search the relation/entity2 query."
        if progress == "ready_finish":
            return (
                "decide the FEVER label from the current evidence: Finish[SUPPORTS] for direct entailment, "
                "Finish[REFUTES] for a clear contradiction, otherwise Finish[NOT ENOUGH INFO]. "
                "For coarse temporal wording, do not REFUTE unless the evidence clearly contradicts the claimed period."
            )
        if progress in {"search_failed", "invalid_action"}:
            if search_hint:
                return (
                    f"the last search/page miss is tool status, not factual evidence; do not Finish[NOT ENOUGH INFO] yet; "
                    f"retry untried current-claim query {search_hint}."
                )
            return (
                "the last search/page miss is tool status, not factual evidence; do not Finish[NOT ENOUGH INFO] yet; "
                "try entity+relation/entity2, entity+type suffix, or full-claim keywords first."
            )
        return "verify the claim using Search, Lookup, then an evidence-grounded Finish label."


class BFCLPromptStyle(BasePromptStyle):
    name = "bfcl_mt"

    def phase_items(self, renderer: Any, *, query: Any, target: str, tool: str, destination: str) -> list[str]:
        progress = str(getattr(query, "progress_state", "") or "")
        ctx = _bfcl_context(query)
        if progress == "tool_result_observed":
            return [
                "BFCL stage=tool_result_observed; inspect the latest tool output for the current user turn, then use FinishTurn[] when the turn is answered instead of repeating the same call."
            ]
        if progress in {"invalid_action", "tool_error"}:
            return [
                "BFCL repair stage; rewrite the next response as valid BFCL Python or Exec[...] using only the current tool definitions and current error."
            ]
        if progress == "checker_failed":
            return [
                "BFCL checker failed; compare the whole turn sequence against the user requests before adding more calls."
            ]
        apis = ", ".join(ctx["apis"][:3]) if ctx["apis"] else "current APIs"
        items = [
            f"BFCL stage=need_tool_call; choose a minimal current-turn tool call from {apis}, or FinishTurn[] only after the turn is satisfied."
        ]
        if ctx["authenticated"] is True and _bfcl_turn_mentions_auth(ctx["current_turn"]):
            items.append(
                "BFCL state says the current API session is already authenticated; do not call login/auth just because the user mentioned login, unless a tool error proves auth is missing."
            )
        return items

    def keep_global_text(self, renderer: Any, *, text: str, norm_text: str, task_family: str) -> bool:
        return "bfcl" in norm_text or "tool" in norm_text or "finishturn" in norm_text

    def keep_local_artifact_text(
        self,
        renderer: Any,
        *,
        domain: str,
        task_family: str,
        text: str,
        norm_text: str,
    ) -> bool:
        return domain.startswith("bfcl") or task_family.startswith("bfcl") or "bfcl" in norm_text

    def phase_label(self, renderer: Any, *, query: Any, macro: str) -> str:
        progress = str(getattr(query, "progress_state", "") or "")
        labels = {
            "need_tool_call": "choose BFCL tool call",
            "tool_result_observed": "decide next BFCL turn step",
            "invalid_action": "repair BFCL action format",
            "tool_error": "repair BFCL tool arguments",
            "checker_failed": "repair BFCL trajectory",
            "done": "finished",
        }
        return labels.get(progress, progress.replace("_", " ") or "BFCL tool use")

    def state_summary(
        self,
        renderer: Any,
        *,
        query: Any,
        target: str,
        tool: str,
        destination: str,
        held: list[str],
        visible: list[str],
        exhausted: list[str],
    ) -> str:
        ctx = _bfcl_context(query)
        progress = str(getattr(query, "progress_state", "") or "unknown")
        parts = [f"stage={progress}"]
        if ctx["apis"]:
            parts.append("apis=" + ", ".join(ctx["apis"][:4]))
        if ctx["last_action"]:
            parts.append(f"last_action={renderer._memco_shorten(ctx['last_action'], 70)}")
        if ctx["authenticated"] is not None:
            parts.append(f"authenticated={str(ctx['authenticated']).lower()}")
        if ctx["balance"] is not None:
            parts.append(f"balance={ctx['balance']}")
        if ctx["current_turn"]:
            parts.append(f"turn={renderer._memco_shorten(ctx['current_turn'], 100)}")
        if ctx["repeat_count"] >= 2:
            parts.append(f"repeat_count={ctx['repeat_count']}")
        if ctx["last_observation"]:
            parts.append(f"last_output={renderer._memco_shorten(ctx['last_observation'], 90)}")
        return "; ".join(parts)

    def next_priority_line(
        self,
        renderer: Any,
        *,
        query: Any,
        priority_items: list[str],
        admissible: list[str],
    ) -> str:
        progress = str(getattr(query, "progress_state", "") or "")
        ctx = _bfcl_context(query)
        for item in priority_items[:2]:
            text = str(item or "").strip()
            if text:
                return text
        if progress == "tool_result_observed":
            if ctx["repeat_count"] >= 2:
                return "stop repeating the last tool call; emit FinishTurn[] if the latest output answers the user turn, otherwise call a different needed tool."
            return "if the latest tool output satisfies the current user turn, emit FinishTurn[]; otherwise make one different necessary tool call."
        if progress in {"invalid_action", "tool_error"}:
            return "repair with one valid BFCL Python call or Exec[...] grounded in the current tool schema; do not reuse arguments that caused the error."
        if progress == "checker_failed":
            return "start the next attempt by matching each user turn to the required tool calls, then FinishTurn[] after each satisfied turn."
        if ctx["authenticated"] is True and _bfcl_turn_mentions_auth(ctx["current_turn"]):
            return "the session is already authenticated; skip login/auth unless required by an error, and execute the actual requested operation using current tool outputs."
        return "make the minimal valid BFCL tool call for the current user turn; avoid thoughts or copied examples."


def _fever_search_hint(renderer: Any, query: Any, admissible: list[str]) -> str:
    ctx = _fever_claim_context(query)
    variants = [str(x).strip() for x in (ctx.get("query_variants", []) or []) if str(x).strip()]
    for variant in variants:
        variant_norm = _fever_norm(variant)
        for action in admissible:
            if str(action).lower().startswith("search[") and variant_norm and variant_norm in _fever_norm(str(action)):
                return str(action)
        return f"Search[{_fever_display(variant)}]"
    anchor = _fever_entity_text(str(ctx.get("entity", "") or ""))
    if not anchor:
        return ""
    anchor_norm = _fever_norm(anchor)
    for action in admissible:
        if str(action).lower().startswith("search[") and anchor_norm and anchor_norm in _fever_norm(str(action)):
            return str(action)
    return f"Search[{_fever_display(anchor)}]"


def _fever_lookup_hint(renderer: Any, query: Any) -> str:
    ctx = _fever_claim_context(query)
    for hint in ctx.get("lookup_keywords", []) or []:
        cleaned = _fever_entity_text(str(hint or ""))
        if cleaned:
            return cleaned
    goal = renderer._memco_clean(str(getattr(query, "goal", "") or "")).removeprefix("Verify claim:").strip()
    if not goal:
        return ""
    goal_roles = getattr(query, "goal_roles", {}) or {}
    anchor = _fever_entity_text(str(goal_roles.get("object", "") or ""))
    text = _fever_norm(goal)
    if anchor:
        text = re.sub(rf"\b{re.escape(_fever_norm(anchor))}\b", " ", text)
    text = re.sub(
        r"\b(?:claim|has|have|had|is|are|was|were|a|an|the|to|of|in|on|by|for|with|and|or|no|not)\b",
        " ",
        text,
    )
    words = [w for w in text.split() if len(w) > 2]
    return " ".join(words[:5]).strip()


def _fever_claim_context(query: Any) -> dict[str, Any]:
    belief = getattr(query, "belief", {}) or {}
    goal_roles = getattr(query, "goal_roles", {}) or {}
    claim_type = (
        str(belief.get("claim_type") or goal_roles.get("claim_type") or "general_fact").strip() or "general_fact"
    )
    entity = str(belief.get("primary_entity") or goal_roles.get("object") or "").strip()
    secondary = str(belief.get("secondary_entity") or goal_roles.get("secondary") or "").strip()
    raw_variants = belief.get("query_variants") or dynamic_query_variants(query) or []
    if isinstance(raw_variants, str):
        raw_variants = [raw_variants]
    raw_keywords = belief.get("relation_keywords") or goal_roles.get("relation") or []
    if isinstance(raw_keywords, str):
        raw_keywords = [raw_keywords]
    keywords = [
        _fever_entity_text(str(keyword or "")) for keyword in raw_keywords if _fever_entity_text(str(keyword or ""))
    ]
    return {
        "claim_type": claim_type,
        "entity": _fever_entity_text(entity),
        "secondary_entity": _fever_entity_text(secondary),
        "lookup_keywords": keywords,
        "query_variants": [
            _fever_entity_text(str(item or "")) for item in raw_variants if _fever_entity_text(str(item or ""))
        ][:5],
    }


def dynamic_query_variants(query: Any) -> list[str]:
    dynamic = getattr(query, "dynamic_context", {}) or {}
    values = dynamic.get("query_variants", []) if isinstance(dynamic, dict) else []
    if isinstance(values, str):
        return [values]
    return [str(item) for item in (values or []) if str(item).strip()]


def _fever_entity_text(value: str) -> str:
    text = re.sub(r"[_\s]+", " ", str(value or "")).strip()
    return re.sub(r"\s+", " ", text)


def _fever_norm(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()


def _fever_display(value: str) -> str:
    text = _fever_entity_text(value)
    if not text:
        return ""
    # Reconstruct a readable entity from normalized graph roles without
    # collapsing multi-word titles such as "Off the Wall".
    small = {"a", "an", "and", "for", "in", "of", "on", "the", "to"}
    parts: list[str] = []
    for idx, part in enumerate(text.split()):
        low = part.lower()
        if len(low) == 1:
            parts.append(low.upper())
        elif idx > 0 and low in small:
            parts.append(low)
        else:
            parts.append(low[:1].upper() + low[1:])
    return " ".join(parts)


def _is_fever_label_only_hint(norm_text: str) -> bool:
    text = re.sub(r"[^a-z0-9]+", " ", str(norm_text or "").lower()).strip()
    if "fever" not in text:
        return False
    padded = f" {text} "
    has_finish = " finish supports " in padded or " finish refutes " in padded or " finish not enough info " in padded
    has_evidence_action = (
        " via search " in padded
        or " via lookup " in padded
        or " action search " in padded
        or " action lookup " in padded
        or " prefer search " in padded
        or " prefer lookup " in padded
    )
    return bool(has_finish and not has_evidence_action)


def _is_fever_transferable_hint(norm_text: str) -> bool:
    text = re.sub(r"[^a-z0-9]+", " ", str(norm_text or "").lower()).strip()
    if "fever" not in text:
        # FEVER bundle items can contain entity- or label-specific action
        # examples such as "try finish(label=refutes)" or
        # "try search(query=off_the_wall)".  Those are local priors, not
        # transferable evidence discipline, so only explicit FEVER artifacts
        # are allowed through this style boundary.
        return False
    if _is_fever_label_only_hint(text):
        return False
    padded = f" {text} "
    if any(
        marker in padded
        for marker in (
            " fever evidence workflow ",
            " fever claim workflow ",
            " global fever workflow ",
            " local fever workflow ",
            " fever lookup workflow ",
            " fever content search route ",
            " fever recovery workflow ",
            " fever stop rule ",
            " fever failure avoidance ",
            " evidence strategy ",
            " lookup strategy ",
            " no results recovery ",
            " premature finish ",
        )
    ):
        return True
    if " no results " in padded or " failure " in padded or " avoid " in padded or " reformulat" in text:
        return True
    if " current admissible grounding " in padded and (" search " in padded or " lookup " in padded):
        return True
    # Generic phase-only chunks like "need_lookup_or_finish -> done" add almost no
    # evidence value, and old entity-specific Search[...] examples bias the claim.
    return False


def _bfcl_context(query: Any) -> dict[str, Any]:
    belief = getattr(query, "belief", {}) or {}
    dynamic = getattr(query, "dynamic_context", {}) or {}
    apis = belief.get("involved_classes") or dynamic.get("involved_classes") or []
    if isinstance(apis, str):
        apis = [apis]
    api_list = [str(api).strip() for api in apis if str(api).strip()]
    session = belief.get("session_state") or dynamic.get("session_state") or {}
    if not isinstance(session, dict):
        session = {}
    return {
        "apis": api_list,
        "current_turn": str(belief.get("current_user_turn") or "").strip(),
        "authenticated": _bfcl_first_session_value(session, ".authenticated"),
        "balance": _bfcl_first_session_value(session, ".balance"),
        "last_action": str(belief.get("last_action") or dynamic.get("last_action") or "").strip(),
        "repeat_count": int(belief.get("last_action_repeat_count") or dynamic.get("last_action_repeat_count") or 0),
        "last_observation": str(belief.get("last_observation") or "").strip(),
    }


def _bfcl_first_session_value(session: dict[str, Any], suffix: str) -> Any:
    for key, value in session.items():
        if str(key).endswith(suffix):
            return value
    return None


def _bfcl_turn_mentions_auth(text: str) -> bool:
    low = str(text or "").lower()
    return any(token in low for token in ("login", "log in", "authenticate", "password", "username", "usermame"))


def prompt_style_for_env(env_name: str) -> BasePromptStyle:
    env = str(env_name or "").strip().lower()
    if env.startswith("bfcl") or env.startswith("gorilla"):
        return BFCLPromptStyle()
    if env.startswith("fever"):
        return FeverPromptStyle()
    if env.startswith("pddl"):
        return PDDLPromptStyle()
    return ALFWorldPromptStyle()


def prompt_style_for_query(query: Any, task_family: str = "") -> BasePromptStyle:
    scene_id = str(getattr(query, "scene_id", "") or "").strip().lower()
    family = str(task_family or getattr(query, "task_family", "") or "").strip().lower()
    if scene_id.startswith("bfcl") or family.startswith("bfcl"):
        return BFCLPromptStyle()
    if scene_id.startswith("fever:") or family.startswith("fever"):
        return FeverPromptStyle()
    if scene_id.startswith("pddl:") or family.startswith("pddl"):
        return PDDLPromptStyle()
    return ALFWorldPromptStyle()
