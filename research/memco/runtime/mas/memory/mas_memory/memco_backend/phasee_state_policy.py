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
from dataclasses import dataclass, field
from typing import Callable

from .graph_types import CandidateType, MemoryQuery, SupportBundle


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip())


def _normalize_token(text: str) -> str:
    lowered = _normalize_space(text).lower().replace(" ", "_")
    return re.sub(r"_[0-9]+$", "", lowered)


def _base_token(text: str) -> str:
    token = _normalize_token(text)
    for prefix in ("a_", "an_", "the_", "some_"):
        if token.startswith(prefix):
            token = token[len(prefix) :]
    return token


_CONTAINER_TYPES = {"drawer", "cabinet", "safe", "fridge", "microwave"}
_SUPPORT_TYPES = {
    "desk",
    "dresser",
    "shelf",
    "sidetable",
    "coffeetable",
    "diningtable",
    "bed",
    "sofa",
    "armchair",
    "tvstand",
    "countertop",
    "garbagecan",
}


@dataclass(slots=True)
class StatePolicy:
    phase: str
    target: str
    destination: str | None
    current_location: str
    visible_objects: list[str] = field(default_factory=list)
    held_objects: list[str] = field(default_factory=list)
    force_action: str | None = None
    suggested_actions: list[str] = field(default_factory=list)
    blocked_actions: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    state_facts: list[str] = field(default_factory=list)
    action_scores: dict[str, float] = field(default_factory=dict)
    process_verb: str | None = None
    process_tool: str | None = None
    target_processed: bool = False
    reason: str = ""


def _process_requirement(
    task_family: str,
    goal_roles: dict[str, str] | None = None,
) -> tuple[str | None, str | None]:
    if task_family == "pick_clean_then_place_in_recep":
        return "clean", "sinkbasin"
    if task_family == "pick_heat_then_place_in_recep":
        return "heat", "microwave"
    if task_family == "pick_cool_then_place_in_recep":
        return "cool", "fridge"
    if task_family == "look_at_obj_in_light":
        tool = _base_token(str((goal_roles or {}).get("tool", "")))
        return "use", tool or "desklamp"
    return None, None


def _has_processed_target(
    session,
    target: str,
    process_verb: str | None,
    process_tool: str | None = None,
) -> bool:
    if not process_verb:
        return False
    for step in reversed(getattr(session, "_steps", []) or []):
        if getattr(step.feedback, "failure_label", None):
            continue
        action = getattr(step, "action", None)
        if action is None or action.verb != process_verb:
            continue
        if process_verb == "use":
            tool = _base_token(str(action.slots.get("tool", "")))
            return bool(process_tool and tool == process_tool)
        if _object_is_target(str(action.slots.get("object", "")), target):
            return True
    return False


def infer_phase(query: MemoryQuery, session=None) -> str:
    process_verb, process_tool = _process_requirement(str(query.task_family or ""), query.goal_roles)
    target = _base_token(str(query.goal_roles.get("object", "")))
    if query.required_count <= 1 and process_verb:
        if query.held_relevant_count > 0:
            return (
                "deliver_target"
                if _has_processed_target(session, target, process_verb, process_tool)
                else "process_target"
            )
        return "search_target"
    if query.required_count <= 1:
        return "deliver_target" if query.held_relevant_count > 0 else "search_target"
    if query.placed_relevant_count <= 0 and query.held_relevant_count <= 0:
        return "search_first_target"
    if query.placed_relevant_count <= 0 and query.held_relevant_count > 0:
        return "deliver_first_target"
    if query.remaining_relevant_count > 0 and query.held_relevant_count <= 0:
        return "search_second_target"
    if query.remaining_relevant_count > 0 and query.held_relevant_count > 0:
        return "deliver_second_target"
    return "finalize"


def _canonical_map(session, admissible_actions: list[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for action in admissible_actions:
        canonical = session.adapter.canonicalize_action(action).canonical_str.lower()
        mapping[canonical] = action
    return mapping


def _object_is_target(name: str, target: str) -> bool:
    return bool(target) and _base_token(name) == target


def _location_role(name: str) -> str:
    token = _base_token(name)
    if token in _CONTAINER_TYPES:
        return "container"
    if token in _SUPPORT_TYPES:
        return "support_surface"
    return token


def _location_family(name: str) -> str:
    return _base_token(name)


def _preferred_location_families(
    destination: str,
) -> tuple[str, ...]:
    # Keep search priors generic and scene-structural (no task-family/object hardcoding).
    base = (
        "coffeetable",
        "sidetable",
        "sofa",
        "armchair",
        "shelf",
        "dresser",
        "tvstand",
        "countertop",
        "diningtable",
        "cabinet",
        "drawer",
    )
    ordered = list(base)
    if destination:
        ordered.append(destination)
    return tuple(dict.fromkeys(ordered))


def _action_location(adapter, action_text: str) -> str:
    action = adapter.canonicalize_action(action_text)
    if action.verb == "go":
        return str(action.slots.get("target", "")).lower()
    if action.verb == "open":
        return str(action.slots.get("container", "")).lower()
    if action.verb == "examine":
        return str(action.slots.get("target", "")).lower()
    return ""


def _same_location(left: str, right: str) -> bool:
    return bool(left and right) and (_base_token(left) == _base_token(right) or left.lower() == right.lower())


def _phase_is_search(phase: str) -> bool:
    return str(phase or "").startswith("search")


def _surface_or_none(canonical_map: dict[str, str], action_text: str) -> str | None:
    if not action_text:
        return None
    if action_text in canonical_map.values():
        return action_text
    normalized = action_text.lower().strip()
    if normalized in canonical_map:
        return canonical_map[normalized]
    return None


def _project_bundle_actions(session, admissible_actions: list[str], actions: list[str]) -> list[str]:
    canonical_map = _canonical_map(session, admissible_actions)
    projected: list[str] = []
    for action in actions or []:
        surface = _surface_or_none(canonical_map, action)
        if surface:
            projected.append(surface)
            continue
        raw = _normalize_space(action).lower()
        for admissible in admissible_actions:
            if raw == _normalize_space(admissible).lower():
                projected.append(admissible)
                break
    return list(dict.fromkeys(projected))


def derive_state_policy(
    query: MemoryQuery,
    session,
    admissible_actions: list[str],
    recent_actions: list[str] | None = None,
) -> StatePolicy:
    state = session.current_state
    target = _base_token(str(query.goal_roles.get("object", "")))
    destination = _base_token(str(query.goal_roles.get("destination", "")))
    visible_objects = list(state.visible_objects or ())
    held_objects = list(state.held_objects or ())
    location = str(state.location or "unknown")
    process_verb, process_tool = _process_requirement(str(query.task_family or ""), query.goal_roles)
    target_processed = _has_processed_target(session, target, process_verb, process_tool)
    phase = infer_phase(query, session=session)
    policy = StatePolicy(
        phase=phase,
        target=target,
        destination=destination or None,
        current_location=location,
        visible_objects=visible_objects,
        held_objects=held_objects,
        process_verb=process_verb,
        process_tool=process_tool,
        target_processed=target_processed,
    )

    recent_actions = list(recent_actions or [])
    adapter = session.adapter
    exhausted_locations = {
        str(value).lower()
        for value in (
            (query.dynamic_context.get("exhausted_locations", []) if query.dynamic_context else [])
            or state.searched_locations
            or ()
        )
        if value
    }
    locked_instances = {
        str(value).lower()
        for value in (query.dynamic_context.get("locked_instances", []) if query.dynamic_context else [])
        if value
    }
    preferred_families = _preferred_location_families(destination)
    container_attempts = 0
    if query.dynamic_context:
        counts = query.dynamic_context.get("search_attempt_counts", {}) or {}
        if isinstance(counts, dict):
            for key, value in counts.items():
                if _location_role(str(key)) == "container":
                    try:
                        container_attempts += int(value)
                    except Exception:
                        continue

    if "help" in admissible_actions:
        policy.blocked_actions.append("help")
    if "inventory" in admissible_actions:
        policy.blocked_actions.append("inventory")

    if query.goal_object_matches_visible:
        for action_text in admissible_actions:
            action = adapter.canonicalize_action(action_text)
            if action.verb == "take" and _object_is_target(str(action.slots.get("object", "")), target):
                policy.force_action = action_text
                policy.suggested_actions = [action_text]
                policy.reason = "target_visible_take_immediately"
                policy.state_facts.append(f"target {target} is visible at {location}")
                return policy

    if process_verb and query.held_relevant_count > 0 and not target_processed:
        policy.warnings.append(f"complete_{process_verb}_before_delivery")
        policy.state_facts.append(f"target {target} still needs {process_verb}")
        for action_text in admissible_actions:
            action = adapter.canonicalize_action(action_text)
            if action.verb == process_verb and (
                (
                    process_verb == "use"
                    and process_tool
                    and _base_token(str(action.slots.get("tool", ""))) == process_tool
                )
                or _object_is_target(str(action.slots.get("object", "")), target)
            ):
                policy.force_action = action_text
                policy.suggested_actions = [action_text]
                policy.reason = f"holding_target_{process_verb}_immediately"
                return policy
        for action_text in admissible_actions:
            action = adapter.canonicalize_action(action_text)
            if (
                action.verb == "go"
                and process_tool
                and _base_token(str(action.slots.get("target", ""))) == process_tool
            ):
                policy.suggested_actions.append(action_text)
                policy.action_scores[action_text] = max(policy.action_scores.get(action_text, 0.0), 9.0)
            elif (
                action.verb == "move"
                and _object_is_target(str(action.slots.get("object", "")), target)
                and destination
                and _base_token(str(action.slots.get("destination", ""))) == destination
            ):
                policy.blocked_actions.append(action_text)
                policy.state_facts.append(f"do not deliver {target} before {process_verb}")

    if query.held_relevant_count > 0:
        for action_text in admissible_actions:
            action = adapter.canonicalize_action(action_text)
            if (
                action.verb == "move"
                and _object_is_target(str(action.slots.get("object", "")), target)
                and destination
                and _base_token(str(action.slots.get("destination", ""))) == destination
            ):
                policy.force_action = action_text
                policy.suggested_actions = [action_text]
                policy.reason = "holding_target_place_immediately"
                policy.state_facts.append(f"holding target {target}")
                return policy
        for action_text in admissible_actions:
            action = adapter.canonicalize_action(action_text)
            if action.verb == "go" and destination and _base_token(str(action.slots.get("target", ""))) == destination:
                policy.suggested_actions.append(action_text)
                policy.action_scores[action_text] = max(policy.action_scores.get(action_text, 0.0), 8.0)
                policy.state_facts.append(f"holding target {target}; destination is {destination}")

    visible_bases = {_base_token(item) for item in visible_objects}
    for action_text in admissible_actions:
        action = adapter.canonicalize_action(action_text)
        if action.verb == "take":
            action_object_raw = str(action.slots.get("object", "")).lower()
            action_object = _base_token(action_object_raw)
            if action_object_raw and action_object_raw in locked_instances:
                policy.blocked_actions.append(action_text)
                policy.state_facts.append(f"instance {action_object_raw} already delivered; lock repick")
                continue
            if target and action_object and action_object in visible_bases and action_object != target:
                policy.blocked_actions.append(action_text)
                policy.state_facts.append(f"visible object {action_object} is not target {target}")
        elif action.verb == "move":
            action_object_raw = str(action.slots.get("object", "")).lower()
            if action_object_raw and action_object_raw in locked_instances:
                policy.blocked_actions.append(action_text)
                policy.state_facts.append(f"instance {action_object_raw} already delivered; lock remanipulation")

    if query.held_relevant_count <= 0 and destination:
        for action_text in admissible_actions:
            action = adapter.canonicalize_action(action_text)
            if action.verb == "go" and _base_token(str(action.slots.get("target", ""))) == destination:
                policy.action_scores[action_text] = policy.action_scores.get(action_text, 0.0) - 2.0
            elif action.verb == "examine" and _base_token(str(action.slots.get("target", ""))) == destination:
                policy.action_scores[action_text] = policy.action_scores.get(action_text, 0.0) - 2.0
        policy.warnings.append(f"do_not_prioritize_destination_before_holding_{target or 'target'}")

    current_location = str(state.location or "").lower()
    if current_location and current_location in exhausted_locations and not query.goal_object_matches_visible:
        policy.state_facts.append(f"{current_location} already checked without target")
        for action_text in admissible_actions:
            action = adapter.canonicalize_action(action_text)
            if action.verb == "look":
                policy.blocked_actions.append(action_text)
            elif action.verb == "examine" and _same_location(str(action.slots.get("target", "")), current_location):
                policy.blocked_actions.append(action_text)
            elif action.verb == "open" and _same_location(str(action.slots.get("container", "")), current_location):
                policy.blocked_actions.append(action_text)

    if len(recent_actions) >= 2 and recent_actions[-1] == recent_actions[-2]:
        repeated = recent_actions[-1]
        if repeated in admissible_actions:
            policy.blocked_actions.append(repeated)
            policy.warnings.append(f"avoid_repeating_{repeated}")

    for action_text in admissible_actions:
        if action_text in policy.blocked_actions:
            continue
        action = adapter.canonicalize_action(action_text)
        score = policy.action_scores.get(action_text, 0.0)
        pending_high_priority = any(
            adapter.canonicalize_action(candidate).verb == "go"
            and _location_family(str(adapter.canonicalize_action(candidate).slots.get("target", "")))
            in preferred_families
            and str(adapter.canonicalize_action(candidate).slots.get("target", "")).lower() not in exhausted_locations
            for candidate in admissible_actions
        )
        if action.verb == "go":
            target_loc = str(action.slots.get("target", "")).lower()
            if target_loc:
                family = _location_family(target_loc)
                if target_loc in exhausted_locations:
                    score -= 4.0
                else:
                    score += 4.0
                if family in preferred_families:
                    score += 3.0
                elif _location_role(target_loc) == "container":
                    score -= 1.0
                    if pending_high_priority and container_attempts >= 3:
                        score -= 4.0
        elif action.verb == "open":
            container = str(action.slots.get("container", "")).lower()
            if container:
                if container in exhausted_locations:
                    score -= 3.0
                else:
                    score += 6.0
                if pending_high_priority and container_attempts >= 3:
                    score -= 4.0
        elif action.verb == "examine":
            target_loc = str(action.slots.get("target", "")).lower()
            role = _location_role(target_loc)
            family = _location_family(target_loc)
            if role in {"container", "support_surface"}:
                if target_loc in exhausted_locations:
                    score -= 3.5
                else:
                    score += 3.0
                if family in preferred_families:
                    score += 1.5
                elif role == "container" and pending_high_priority and container_attempts >= 3:
                    score -= 3.0
            elif _object_is_target(target_loc, target):
                score += 12.0
            else:
                score -= 1.5
        elif action.verb == "look":
            score -= 2.0
        elif action.verb == "close":
            score -= 4.0
        elif action.verb == "move" and query.held_relevant_count <= 0:
            score -= 5.0
        policy.action_scores[action_text] = score

    ranked = [
        (action, policy.action_scores.get(action, 0.0))
        for action in admissible_actions
        if action not in policy.blocked_actions
    ]
    ranked.sort(key=lambda item: item[1], reverse=True)
    if not _phase_is_search(policy.phase):
        for action, score in ranked[:3]:
            if score > -2.5 and action not in policy.suggested_actions:
                policy.suggested_actions.append(action)

    if not policy.reason:
        policy.reason = "phase_guided_search_or_progress"
    policy.blocked_actions = list(dict.fromkeys(policy.blocked_actions))
    policy.suggested_actions = list(dict.fromkeys(policy.suggested_actions))
    policy.warnings = list(dict.fromkeys(policy.warnings))
    policy.state_facts = list(dict.fromkeys(policy.state_facts))
    return policy


def fuse_state_memory(
    state_policy: StatePolicy,
    bundle: SupportBundle | None,
    admissible_actions: list[str],
    session,
) -> dict[str, object]:
    evidence: list[str] = []
    failure_reflections: list[str] = []
    if bundle is not None:
        routed_fused_items = list(getattr(bundle, "fused_support_items", []) or [])
        if routed_fused_items:
            for item in routed_fused_items[:5]:
                summary = str(getattr(item, "summary", "")).strip()
                if not summary:
                    continue
                if item.candidate_type == CandidateType.FAILURE or item.branch_tag in {
                    "failure_branch",
                    "repair_branch",
                }:
                    failure_reflections.append(summary)
                else:
                    evidence.append(summary)
        for item in list(getattr(bundle, "relation_items", []) or [])[:2]:
            summary = str(getattr(item, "summary", "")).strip()
            if summary:
                if item.candidate_type == CandidateType.FAILURE or item.branch_tag in {
                    "failure_branch",
                    "repair_branch",
                }:
                    failure_reflections.append(summary)
                else:
                    evidence.append(summary)
        for item in list(getattr(bundle, "fact_items", []) or [])[:2]:
            summary = str(getattr(item, "summary", "")).strip()
            if summary:
                evidence.append(summary)
        for item in list(getattr(bundle, "workflow_items", []) or [])[:1]:
            summary = str(getattr(item, "summary", "")).strip()
            if summary:
                if item.branch_tag in {"failure_branch", "repair_branch"}:
                    failure_reflections.append(summary)
                else:
                    evidence.append(summary)
        for item in list(getattr(bundle, "plan_items", []) or [])[:1]:
            summary = str(getattr(item, "summary", "")).strip()
            if summary:
                evidence.append(summary)
        for item in list(getattr(bundle, "reflection_items", []) or [])[:1]:
            summary = str(getattr(item, "summary", "")).strip()
            if summary:
                failure_reflections.append(summary)

    if state_policy.force_action:
        return {
            "force_action": state_policy.force_action,
            "blocked_actions": [action for action in admissible_actions if action != state_policy.force_action],
            "suggested_actions": [state_policy.force_action],
            "warnings": list(state_policy.warnings),
            "evidence": list(dict.fromkeys(evidence))[:5],
            "failure_reflections": list(dict.fromkeys(failure_reflections))[:3],
            "state_facts": list(state_policy.state_facts),
        }

    final_blocked = list(dict.fromkeys(list(state_policy.blocked_actions)))
    final_suggested = list(dict.fromkeys(list(state_policy.suggested_actions)))
    if _phase_is_search(state_policy.phase):
        final_suggested = []
    warnings = list(dict.fromkeys(list(state_policy.warnings)))
    return {
        "force_action": None,
        "blocked_actions": final_blocked,
        "suggested_actions": final_suggested[:5],
        "warnings": warnings[:6],
        "evidence": list(dict.fromkeys(evidence))[:5],
        "failure_reflections": list(dict.fromkeys(failure_reflections))[:3],
        "state_facts": list(state_policy.state_facts)[:6],
    }


def _recent_repeat_penalty(action: str, recent_actions: list[str]) -> float:
    if not recent_actions:
        return 0.0
    recent = recent_actions[-3:]
    return -10.0 if action in recent else 0.0


def score_action(
    action: str,
    state_policy: StatePolicy,
    bundle: SupportBundle | None,
    session,
    query: MemoryQuery,
    fused_support: dict[str, object],
) -> float:
    if action == state_policy.force_action:
        return 100.0
    score = float(state_policy.action_scores.get(action, 0.0))
    if action in state_policy.blocked_actions or action in list(fused_support.get("blocked_actions", []) or []):
        score -= 100.0
    if action in state_policy.suggested_actions and not _phase_is_search(state_policy.phase):
        score += 20.0
    recent_actions = [
        step.action.surface_form for step in getattr(session, "_steps", [])[-3:] if step.action.surface_form
    ]
    score += _recent_repeat_penalty(action, recent_actions)
    return score


def _fuzzy_repair(
    raw_response: str,
    admissible_actions: list[str],
    session,
) -> str | None:
    if not raw_response:
        return None
    lines = [line.strip() for line in str(raw_response).splitlines() if line.strip()]
    candidate_text = lines[-1] if lines else str(raw_response).strip()
    if ":" in candidate_text:
        candidate_text = candidate_text.split(":", 1)[-1].strip()
    parsed_candidate = session.adapter.canonicalize_action(candidate_text)
    if not parsed_candidate.verb or parsed_candidate.verb == "other":
        return None
    for action_text in admissible_actions:
        action = session.adapter.canonicalize_action(action_text)
        if action.verb != parsed_candidate.verb:
            continue
        if action.slots == parsed_candidate.slots:
            return action_text
        matched_slots = True
        for key, value in parsed_candidate.slots.items():
            if not value:
                continue
            if _base_token(str(action.slots.get(key, ""))) != _base_token(value):
                matched_slots = False
                break
        if matched_slots:
            return action_text
    return None


def safe_extract_or_repair(
    raw_response: str,
    admissible_actions: list[str],
    state_policy: StatePolicy,
    fused_support: dict[str, object],
    session,
    parser: Callable[[str, list[str]], str | None],
) -> tuple[str, str, str | None]:
    parsed = parser(raw_response, admissible_actions)
    force_action = fused_support.get("force_action")
    blocked_actions = set(fused_support.get("blocked_actions", []) or [])
    suggested_actions = [
        action for action in fused_support.get("suggested_actions", []) or [] if action in admissible_actions
    ]

    if force_action in admissible_actions:
        if parsed == force_action:
            return force_action, "parsed", parsed
        return force_action, "state_force", parsed

    if parsed and parsed in admissible_actions and parsed not in blocked_actions:
        return parsed, "parsed", parsed

    repaired = _fuzzy_repair(raw_response, admissible_actions, session)
    if repaired and repaired not in blocked_actions:
        return repaired, "fuzzy_repair", parsed

    if not _phase_is_search(state_policy.phase):
        for action in suggested_actions:
            if action not in blocked_actions:
                return action, "state_suggested_fallback", parsed

    valid = [action for action in admissible_actions if action not in blocked_actions]
    if valid:
        best = max(valid, key=lambda action: state_policy.action_scores.get(action, 0.0))
        return best, "safe_rank_fallback", parsed

    return admissible_actions[0], "last_resort_admissible", parsed


def render_state_critical_rules(
    query: MemoryQuery,
    state_policy: StatePolicy,
    fused_support: dict[str, object],
    max_items: int = 5,
) -> str:
    goal_anchor_label = "Destination"
    goal_anchor_value = state_policy.destination or "goal_destination"
    if not state_policy.destination and state_policy.process_verb == "use" and state_policy.process_tool:
        goal_anchor_label = "Goal tool"
        goal_anchor_value = state_policy.process_tool
    lines = [
        f"Target object: {state_policy.target or 'target_object'}",
        f"{goal_anchor_label}: {goal_anchor_value}",
        f"Current phase: {state_policy.phase}",
        f"Held relevant count: {query.held_relevant_count}",
        f"Placed relevant count: {query.placed_relevant_count}",
    ]
    if state_policy.process_verb:
        lines.append(f"Task requirement: {state_policy.process_verb} the target before final delivery.")
        if state_policy.process_tool:
            lines.append(f"Processing tool: {state_policy.process_tool}")
    lines.extend(
        [
            "",
            "State-derived hard rules:",
            "- If target is visible, take it immediately.",
            "- Do not prioritize destination before holding target.",
            "- Do not take visible non-target objects.",
            "- Do not repeat actions on checked locations.",
            "- If holding target, prioritize progress-consistent actions.",
            "- Only choose one admissible action.",
        ]
    )
    blocked = list(fused_support.get("blocked_actions", []) or [])
    if state_policy.process_verb and not state_policy.target_processed:
        lines.append("")
        lines.append("Task-specific processing rules:")
        lines.append(f"- Do not move the target to the final destination before it is {state_policy.process_verb}ed.")
        if query.held_relevant_count > 0 and state_policy.process_tool:
            lines.append(
                f"- While holding the target, prioritize reaching {state_policy.process_tool} to {state_policy.process_verb} it."
            )
    if state_policy.force_action:
        lines.append("")
        lines.append("Immediate state-required action:")
        lines.append(f"- {state_policy.force_action}")
    if blocked:
        lines.append("")
        lines.append("State-blocked actions:")
        for action in blocked[:max_items]:
            lines.append(f"- {action}")
    state_facts = list(fused_support.get("state_facts", []) or [])
    if state_facts:
        lines.append("")
        lines.append("State facts:")
        for item in state_facts[:max_items]:
            lines.append(f"- {item}")
    return "\n".join(lines)
