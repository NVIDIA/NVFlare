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

import os
import re
from collections import defaultdict

from .graph_types import (
    ArtifactKind,
    CandidateType,
    CanonicalAction,
    EdgeType,
    GlobalGraphMemory,
    LocalGraphMemory,
    MemoryArtifact,
    MemoryQuery,
    MemoryRule,
    NodeType,
    PromotionCandidate,
    RuleType,
    SupportBundle,
    SupportItem,
)
from .routing_graph import HeuristicSupportRouter, StudentSupportRouter, route_bundle


def _paper_eq22_27_all_ones() -> bool:
    return str(os.getenv("NV_MEMCO_PAPER_EQ22_27_ALL_ONES", "")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _tokenize(text: str) -> set[str]:
    return {tok for tok in re.findall(r"[a-z0-9_]+", text.lower()) if tok}


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _candidate_tokens(candidate: PromotionCandidate) -> set[str]:
    structure_blob = " ".join(f"{k} {v}" for k, v in candidate.structure.items())
    return _tokenize(candidate.summary + " " + structure_blob)


def _action_bonus(candidate: PromotionCandidate, actions: tuple[CanonicalAction, ...]) -> float:
    if not actions:
        return 0.0
    if candidate.candidate_type == CandidateType.WORKFLOW:
        return 1.0 if _paper_eq22_27_all_ones() else 0.05
    action_strings = {action.canonical_str for action in actions}
    action_verbs = {action.verb for action in actions}
    structure = candidate.structure
    bonus = 0.0
    exact_action_bonus = 1.0 if _paper_eq22_27_all_ones() else 0.2
    structured_value_bonus = 1.0 if _paper_eq22_27_all_ones() else 0.15
    verb_bonus = 1.0 if _paper_eq22_27_all_ones() else 0.08
    if structure.get("action") in action_strings:
        bonus += exact_action_bonus
    if structure.get("repair_action") in action_strings:
        bonus += exact_action_bonus
    for value in structure.values():
        if isinstance(value, str):
            if value in action_strings:
                bonus += structured_value_bonus
            if any(verb in value for verb in action_verbs):
                bonus += verb_bonus
    return bonus


def _graph_evidence_bonus(candidate: PromotionCandidate) -> float:
    return (1.0 if _paper_eq22_27_all_ones() else 0.45) * candidate.prior_score


def _candidate_penalty(candidate: PromotionCandidate) -> float:
    support = candidate.positive + candidate.negative
    if support == 0:
        return 0.0
    negative_rate = candidate.negative / support
    stall_rate = candidate.stalled / support
    if _paper_eq22_27_all_ones():
        return 1.0 * negative_rate + 1.0 * stall_rate
    return 0.25 * negative_rate + 0.35 * stall_rate


def _branch_tag(
    candidate_type: CandidateType,
    *,
    pattern_kind: str = "",
    positive: int = 0,
    negative: int = 0,
    stalled: int = 0,
) -> str:
    support = positive + negative
    negative_rate = negative / support if support else 0.0
    stall_rate = stalled / support if support else 0.0

    if candidate_type == CandidateType.REPAIR:
        return "repair_branch"
    if candidate_type == CandidateType.FAILURE or pattern_kind == "anti_pattern":
        return "failure_branch"
    if pattern_kind == "repair":
        return "repair_branch"
    if negative_rate >= 0.75 or stall_rate >= 0.6:
        return "failure_branch"
    return "success_branch"


def _local_branch_tag(
    candidate_type: CandidateType,
    *,
    pattern_kind: str = "",
    positive: int = 0,
    negative: int = 0,
    stalled: int = 0,
) -> str:
    support = positive + negative
    negative_rate = negative / support if support else 0.0
    stall_rate = stalled / support if support else 0.0

    if candidate_type == CandidateType.FAILURE or pattern_kind == "anti_pattern":
        return "failure_branch"
    if candidate_type == CandidateType.REPAIR or pattern_kind == "repair":
        return "repair_branch"
    # Keep local retrieval close to the previously stable behavior:
    # only mark an item as a failure branch when the evidence is overwhelmingly bad.
    if negative_rate >= 0.9 or stall_rate >= 0.8:
        return "failure_branch"
    return "success_branch"


def _query_tokens(query: MemoryQuery) -> set[str]:
    base = _tokenize(query.goal)
    if query.current_stage:
        base |= _tokenize(query.current_stage)
    if query.location:
        base |= _tokenize(query.location)
    for keyword in query.keywords:
        base |= _tokenize(keyword)
    for action in query.admissible_actions:
        base |= _tokenize(action.canonical_str)
    if query.failure_label:
        base |= _tokenize(query.failure_label)
    return base


def _query_task_family(query: MemoryQuery) -> str:
    if query.task_family:
        return query.task_family
    goal = query.goal.lower()
    if "under the" in goal or "with the" in goal:
        return "look_at_obj_in_light"
    if goal.startswith("put two "):
        return "pick_two_obj_and_place"
    return "pick_and_place_simple"


def _is_multi_object_query(query: MemoryQuery) -> bool:
    if query.required_count > 1:
        return True
    if query.progress_hint.startswith("multi_object"):
        return True
    family = _query_task_family(query)
    goal = query.goal.lower()
    return "two_obj" in family or "two object" in goal or "two objects" in goal


def _keyword_suffixes(query: MemoryQuery, prefix: str) -> set[str]:
    values: set[str] = set()
    for keyword in query.keywords:
        if keyword.startswith(prefix):
            values.add(keyword.split("=", 1)[-1].split("+=", 1)[-1])
    return values


def _canonical_strings(actions: tuple[CanonicalAction, ...]) -> set[str]:
    return {action.canonical_str for action in actions}


def _canonical_verbs(actions: tuple[CanonicalAction, ...]) -> set[str]:
    return {action.verb for action in actions}


def _goal_role_bonus(query: MemoryQuery, text: str, *, task_family: str = "", pattern_kind: str = "") -> float:
    lowered = text.lower()
    bonus = 0.0
    object_role = str(query.goal_roles.get("object", "")).lower()
    destination_role = str(query.goal_roles.get("destination", "")).lower()
    if object_role and object_role in lowered:
        bonus += 0.12
    if destination_role and destination_role in lowered:
        bonus += 0.12
    if _is_multi_object_query(query) and task_family == "pick_two_obj_and_place":
        bonus += 0.08
    if _is_multi_object_query(query) and pattern_kind == "closure" and query.remaining_relevant_count > 0:
        bonus -= 0.08
    return bonus


def _payload_action_str(payload: dict) -> str:
    verb = payload.get("verb", "")
    slots = payload.get("slots", {}) or {}
    if not verb:
        return ""
    if not slots:
        return verb
    parts = ",".join(f"{k}={slots[k]}" for k in sorted(slots))
    return f"{verb}({parts})"


def _action_text_bonus(action_text: str, actions: tuple[CanonicalAction, ...]) -> float:
    if not action_text or not actions:
        return 0.0
    action_strings = _canonical_strings(actions)
    action_verbs = _canonical_verbs(actions)
    bonus = 0.0
    if action_text in action_strings:
        bonus += 0.25
    if any(action_text.startswith(f"{verb}(") or action_text == verb for verb in action_verbs):
        bonus += 0.12
    return bonus


def _goal_tokens(query: MemoryQuery) -> tuple[str, str]:
    object_role = str(query.goal_roles.get("object", "")).lower().strip()
    destination_role = str(query.goal_roles.get("destination", "")).lower().strip()
    return object_role, destination_role


def _goal_tool_token(query: MemoryQuery) -> str:
    return str(query.goal_roles.get("tool", "")).lower().strip()


def _search_attempt_count(query: MemoryQuery, ref: str) -> int:
    counts = query.dynamic_context.get("search_attempt_counts", {}) if query.dynamic_context else {}
    if not isinstance(counts, dict):
        return 0
    return int(counts.get(ref, counts.get(_base_token(ref), 0)) or 0)


def _goal_relevance_from_text(query: MemoryQuery, text: str) -> float:
    lowered = text.lower()
    tokens = _tokenize(lowered)
    object_role, destination_role = _goal_tokens(query)
    score = 0.0
    if object_role and object_role in tokens:
        score += 0.38
        if "take(" in lowered:
            score += 0.16
    elif object_role and "take(" in lowered:
        score -= 0.14
    if destination_role and destination_role in tokens:
        score += 0.28
        if "move(" in lowered or "go(" in lowered:
            score += 0.12
    return score


def _goal_relevance_from_state(query: MemoryQuery, node_payload: dict) -> float:
    object_role, destination_role = _goal_tokens(query)
    score = 0.0
    visible_tokens = _tokenize(" ".join(node_payload.get("visible_objects", []) or []))
    held_tokens = _tokenize(" ".join(node_payload.get("held_objects", []) or []))
    location_tokens = _tokenize(str(node_payload.get("location") or ""))
    if object_role and object_role in visible_tokens:
        score += 0.34
    if object_role and object_role in held_tokens:
        score += 0.42
    if destination_role and destination_role in location_tokens:
        score += 0.24
    return score


def _task_relevance_score(
    query: MemoryQuery,
    *,
    text: str = "",
    task_family: str = "",
    pattern_kind: str = "",
    workflow_stage: str = "",
) -> float:
    lowered = text.lower()
    query_family = _query_task_family(query)
    score = 0.0

    if task_family:
        if task_family == query_family:
            score += 0.28
        else:
            score -= 0.06

    stage = workflow_stage or query.current_stage or ""
    if stage and stage in lowered:
        score += 0.1

    progress_state = query.progress_state or ""
    if progress_state:
        if progress_state in lowered:
            score += 0.14
        if progress_state.startswith("search"):
            if any(marker in lowered for marker in ("locate_target", "inspect_container", "acquire_target")):
                score += 0.08
        if progress_state == "search_second":
            if any(marker in lowered for marker in ("acquire_target", "inspect_container")):
                score += 0.08
        if progress_state in {"carry_target", "finalize"}:
            if any(marker in lowered for marker in ("closure", "move(", "place_target", "transport_target")):
                score += 0.1

    if query.failure_label:
        if "after " in lowered or pattern_kind in {"repair", "failure", "anti_pattern"}:
            score += 0.1
    elif pattern_kind in {"repair", "failure", "anti_pattern"}:
        score -= 0.06

    return score


def _relevance_weights(query: MemoryQuery) -> tuple[float, float, float]:
    if _paper_eq22_27_all_ones():
        return 1.0, 1.0, 1.0
    if query.progress_state in {"carry_target", "finalize"}:
        return 0.26, 0.24, 0.5
    return 0.34, 0.26, 0.4


def _combine_relevance(
    query: MemoryQuery, state_relevance: float, task_relevance: float, goal_relevance: float
) -> float:
    state_weight, task_weight, goal_weight = _relevance_weights(query)
    return state_weight * state_relevance + task_weight * task_relevance + goal_weight * goal_relevance


def _facet_priority(query: MemoryQuery) -> list[str]:
    if query.progress_state in {"carry_target", "finalize"}:
        return ["goal", "task", "state"]
    return ["goal", "state", "task"]


def _facet_value(item: SupportItem, facet: str) -> float:
    if facet == "state":
        return item.state_relevance
    if facet == "task":
        return item.task_relevance
    return item.goal_relevance


def _joint_select(query: MemoryQuery, items: list[SupportItem], limit: int) -> list[SupportItem]:
    if limit <= 0:
        return []
    merged: dict[str, SupportItem] = {}
    for item in items:
        existing = merged.get(item.candidate_id)
        if existing is None or item.score > existing.score:
            merged[item.candidate_id] = item
    ranked = sorted(merged.values(), key=lambda item: item.score, reverse=True)
    if len(ranked) <= limit:
        return ranked

    selected: list[SupportItem] = []
    seen: set[str] = set()
    for facet in _facet_priority(query):
        candidates = [item for item in ranked if item.candidate_id not in seen and _facet_value(item, facet) > 0.05]
        if not candidates:
            continue
        best = max(candidates, key=lambda item: (_facet_value(item, facet), item.score))
        selected.append(best)
        seen.add(best.candidate_id)
        if len(selected) >= limit:
            return selected

    for item in ranked:
        if item.candidate_id in seen:
            continue
        selected.append(item)
        seen.add(item.candidate_id)
        if len(selected) >= limit:
            break
    return selected


def _base_token(value: str) -> str:
    lowered = re.sub(r"_[0-9]+$", "", value.lower().strip())
    for prefix in ("a_", "an_", "the_", "some_"):
        if lowered.startswith(prefix):
            lowered = lowered[len(prefix) :]
    return lowered


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
}


def _parse_action_text(action_text: str) -> tuple[str, dict[str, str]]:
    if "(" not in action_text or not action_text.endswith(")"):
        return action_text.strip(), {}
    verb, rest = action_text.split("(", 1)
    slots: dict[str, str] = {}
    for piece in rest[:-1].split(","):
        if "=" in piece:
            key, value = piece.split("=", 1)
            slots[key.strip()] = value.strip()
    return verb.strip(), slots


def _slot_pattern_matches(query: MemoryQuery, slot_name: str, expected: str, actual: str) -> bool:
    if not expected:
        return True
    expected = expected.lower().strip()
    actual = actual.lower().strip()
    object_role = _base_token(str(query.goal_roles.get("object", "")))
    destination_role = _base_token(str(query.goal_roles.get("destination", "")))
    tool_role = _base_token(str(query.goal_roles.get("tool", "")))
    actual_base = _base_token(actual)
    if expected in {"target_object", "goal_object"}:
        return bool(object_role) and actual_base == object_role
    if expected == "goal_destination":
        return bool(destination_role) and actual_base == destination_role
    if expected == "tool_object":
        return bool(tool_role) and actual_base == tool_role
    if expected == "container":
        return actual_base in _CONTAINER_TYPES
    if expected == "support_surface":
        return actual_base in _SUPPORT_TYPES
    if expected in {"target", "destination", "source"}:
        return True
    return _base_token(expected) == actual_base


def _action_matches_pattern(query: MemoryQuery, action: CanonicalAction, pattern: str) -> bool:
    verb, slots = _parse_action_text(pattern)
    if action.verb != verb:
        return False
    for slot_name, expected in slots.items():
        actual = action.slots.get(slot_name, "")
        if not _slot_pattern_matches(query, slot_name, expected, actual):
            return False
    return True


def _project_rule_actions(query: MemoryQuery, pattern: str) -> list[str]:
    lowered = pattern.lower()
    if any(token in lowered for token in ("=support_surface", "=container")):
        return []
    matches = [
        action.canonical_str for action in query.admissible_actions if _action_matches_pattern(query, action, pattern)
    ]
    return list(dict.fromkeys(matches))


def _project_wrong_target_actions(query: MemoryQuery) -> list[str]:
    target_object, _destination = _goal_tokens(query)
    if not target_object:
        return []
    blocked: list[str] = []
    for action in query.admissible_actions:
        if action.verb == "take":
            action_object = _base_token(str(action.slots.get("object", "")))
            if action_object and action_object != target_object:
                blocked.append(action.canonical_str)
        elif action.verb == "move":
            action_object = _base_token(str(action.slots.get("object", "")))
            if action_object and action_object != target_object:
                blocked.append(action.canonical_str)
        elif action.verb == "examine":
            target = _base_token(str(action.slots.get("target", "")))
            if not target:
                continue
            if target in _CONTAINER_TYPES or target in _SUPPORT_TYPES:
                continue
            if target != target_object:
                blocked.append(action.canonical_str)
    return list(dict.fromkeys(blocked))


def _is_destination_action(query: MemoryQuery, action_text: str) -> bool:
    destination_role = _base_token(str(query.goal_roles.get("destination", "")))
    if not destination_role:
        return False
    verb, slots = _parse_action_text(action_text)
    if verb == "go":
        return _base_token(str(slots.get("target", ""))) == destination_role
    if verb == "examine":
        return _base_token(str(slots.get("target", ""))) == destination_role
    if verb == "move":
        return _base_token(str(slots.get("destination", ""))) == destination_role
    return False


def _is_goal_delivery_action(query: MemoryQuery, action_text: str) -> bool:
    """Whether an action is the current goal-object delivery action.

    Blocked/repair artifacts are learned from previous failures and can be
    useful during search, but they must not mask the exact terminal delivery
    action when the current state already satisfies the delivery preconditions.
    """
    object_role = _base_token(str(query.goal_roles.get("object", "")))
    destination_role = _base_token(str(query.goal_roles.get("destination", "")))
    if not object_role or not destination_role:
        return False
    verb, slots = _parse_action_text(action_text)
    if verb != "move":
        return False
    action_object = _base_token(str(slots.get("object", "")))
    action_destination = _base_token(str(slots.get("destination", "")))
    return action_object == object_role and action_destination == destination_role


def _delivery_ready(query: MemoryQuery) -> bool:
    if query.held_relevant_count <= 0:
        return False
    destination_role = _base_token(str(query.goal_roles.get("destination", "")))
    if not destination_role:
        return False
    location = _base_token(str(query.location or ""))
    return bool(query.destination_reached or (location and location == destination_role))


def _goal_delivery_actions(query: MemoryQuery) -> list[str]:
    if not _delivery_ready(query):
        return []
    return [
        action.canonical_str
        for action in query.admissible_actions
        if _is_goal_delivery_action(query, action.canonical_str)
    ]


def _filter_blocked_actions(query: MemoryQuery, actions: list[str]) -> list[str]:
    delivery_actions = set(_goal_delivery_actions(query))
    if not delivery_actions:
        return list(dict.fromkeys(actions))
    return list(dict.fromkeys(action for action in actions if action not in delivery_actions))


def _filter_suggested_actions(query: MemoryQuery, actions: list[str]) -> list[str]:
    filtered: list[str] = []
    search_stage = query.held_relevant_count <= 0 and query.progress_state not in {"carry_target", "finalize"}
    for action in actions:
        if search_stage and _is_destination_action(query, action):
            continue
        filtered.append(action)
    return list(dict.fromkeys(filtered))


def _rule_candidate_type(rule_type: RuleType) -> CandidateType:
    if rule_type == RuleType.WORKFLOW or rule_type == RuleType.CLOSURE:
        return CandidateType.WORKFLOW
    if rule_type == RuleType.REPAIR:
        return CandidateType.REPAIR
    if rule_type == RuleType.BLOCKED:
        return CandidateType.FAILURE
    return CandidateType.PRECONDITION


def _rule_branch_tag(rule_type: RuleType) -> str:
    if rule_type == RuleType.REPAIR:
        return "repair_branch"
    if rule_type == RuleType.BLOCKED:
        return "failure_branch"
    return "success_branch"


def _rule_specific_action_bonus(query: MemoryQuery, rule: MemoryRule) -> float:
    effect = rule.effect
    bonus = 0.0
    all_ones = _paper_eq22_27_all_ones()
    for key in ("action", "prefer_action", "block_action"):
        action_text = str(effect.get(key, "")).strip()
        if not action_text:
            continue
        projected = _project_rule_actions(query, action_text)
        if projected:
            if len(projected) <= 2:
                bonus += 1.0 if all_ones else (0.28 if key != "block_action" else 0.15)
            elif len(projected) <= 4:
                bonus += 1.0 if all_ones else (0.08 if key != "block_action" else 0.04)
            else:
                bonus -= 1.0 if all_ones else 0.14
        elif key == "prefer_action":
            bonus -= 1.0 if all_ones else 0.08
    return bonus


def _rule_action_patterns(rule: MemoryRule) -> tuple[str, ...]:
    patterns: list[str] = []
    for key in ("prefer_action", "block_action", "action"):
        action_text = str(rule.effect.get(key, "")).strip()
        if action_text:
            patterns.append(action_text)
    return tuple(dict.fromkeys(patterns))


def _artifact_action_patterns(artifact: MemoryArtifact) -> tuple[str, ...]:
    payload = artifact.payload
    patterns: list[str] = []
    for key in ("action_patterns", "repair_patterns", "avoid_patterns"):
        values = payload.get(key, [])
        if isinstance(values, str):
            values = [values]
        for value in values or []:
            text = str(value).strip()
            if text:
                patterns.append(text)
    if not patterns:
        patterns.extend(re.findall(r"[a-z_]+\([a-z0-9_,= ]+\)", artifact.summary.lower()))
    return tuple(dict.fromkeys(patterns))


def _artifact_task_family(artifact: MemoryArtifact) -> str:
    return str(artifact.anchor.get("task_family", ""))


def _artifact_pattern_kind(artifact: MemoryArtifact) -> str:
    payload = artifact.payload
    if artifact.kind == ArtifactKind.REFLECTION:
        return "reflection"
    if artifact.kind == ArtifactKind.RULE:
        return str(payload.get("rule_type", "rule"))
    return str(payload.get("pattern_kind") or payload.get("source") or "prototype")


def _artifact_candidate_type(artifact: MemoryArtifact) -> CandidateType:
    if artifact.kind == ArtifactKind.REFLECTION:
        return CandidateType.REPAIR
    if artifact.kind == ArtifactKind.RULE:
        rule_type = str(artifact.payload.get("rule_type", ""))
        if rule_type == RuleType.BLOCKED.value:
            return CandidateType.FAILURE
        if rule_type == RuleType.REPAIR.value:
            return CandidateType.REPAIR
        if rule_type in {RuleType.WORKFLOW.value, RuleType.CLOSURE.value}:
            return CandidateType.WORKFLOW
        return CandidateType.PRECONDITION
    pattern_kind = str(artifact.payload.get("pattern_kind", ""))
    if pattern_kind in {"workflow", "closure"}:
        return CandidateType.WORKFLOW
    if pattern_kind in {"failure", "anti_pattern"}:
        return CandidateType.FAILURE
    if pattern_kind == "repair":
        return CandidateType.REPAIR
    return CandidateType.PRECONDITION


def _artifact_branch_tag(artifact: MemoryArtifact) -> str:
    if artifact.kind == ArtifactKind.REFLECTION:
        return "repair_branch"
    if artifact.kind == ArtifactKind.RULE and str(artifact.payload.get("rule_type", "")) == RuleType.BLOCKED.value:
        return "failure_branch"
    return "success_branch"


def _location_role(value: str) -> str:
    base = _base_token(value)
    if base in _CONTAINER_TYPES:
        return "container"
    if base in _SUPPORT_TYPES:
        return "support_surface"
    return base


def _graph_transition_alignment(query: MemoryQuery, action_payload: dict, next_state_payload: dict) -> float:
    verb = str(action_payload.get("verb", "")).lower().strip()
    slots = dict(action_payload.get("slots", {}) or {})
    object_role, destination_role = _goal_tokens(query)
    score = 0.0

    if verb == "take":
        action_object = _base_token(str(slots.get("object", "")))
        if object_role and action_object == object_role:
            score += 0.68
        elif object_role and action_object:
            score -= 0.85
    elif verb == "move":
        action_object = _base_token(str(slots.get("object", "")))
        action_destination = _base_token(str(slots.get("destination", "")))
        if object_role and action_object == object_role and destination_role and action_destination == destination_role:
            score += 0.72
        elif action_destination and destination_role and action_destination != destination_role:
            score -= 0.82
    elif verb == "go":
        target = _base_token(str(slots.get("target", "")))
        if destination_role and target == destination_role and query.held_relevant_count <= 0:
            score -= 0.72
        elif target:
            if target in _CONTAINER_TYPES or target in _SUPPORT_TYPES:
                score += 0.1
            attempts = _search_attempt_count(query, target)
            if attempts > 1:
                score -= 0.36
            elif attempts == 0:
                score += 0.08
    elif verb == "open":
        container = _base_token(str(slots.get("container", "")))
        attempts = _search_attempt_count(query, container)
        score += 0.18 if attempts == 0 else -0.12
    elif verb == "examine":
        target = _base_token(str(slots.get("target", "")))
        attempts = _search_attempt_count(query, target)
        if object_role and target == object_role:
            score += 0.48
        elif target in _CONTAINER_TYPES or target in _SUPPORT_TYPES:
            score += 0.16 if attempts == 0 else -0.14
    elif verb == "look":
        score += 0.08 if not query.goal_object_matches_visible else -0.1

    next_goal = _goal_relevance_from_state(query, next_state_payload)
    next_task = _task_relevance_score(
        query,
        workflow_stage=str(next_state_payload.get("workflow_stage") or ""),
        text=str(next_state_payload.get("workflow_stage") or ""),
    )
    score += 0.32 * next_goal + 0.14 * next_task
    return score


def _artifact_relevance(
    query: MemoryQuery,
    artifact: MemoryArtifact,
    graph_context: dict | None = None,
) -> tuple[float, float, float, float]:
    anchor = artifact.anchor
    payload = artifact.payload
    text_parts = [
        artifact.summary,
        " ".join(f"{key} {value}" for key, value in anchor.items()),
        " ".join(f"{key} {value}" for key, value in payload.items()),
    ]
    blob = " ".join(text_parts)
    state_relevance = 0.0
    location_role = str(anchor.get("location_role", "") or payload.get("location_role", ""))
    if location_role and query.location:
        query_location_base = _base_token(query.location)
        if location_role == query_location_base:
            state_relevance += 0.24
        elif location_role == "container" and query_location_base in _CONTAINER_TYPES:
            state_relevance += 0.14
        elif location_role == "support_surface" and query_location_base in _SUPPORT_TYPES:
            state_relevance += 0.14
    progress_state = str(anchor.get("progress_state", ""))
    if progress_state and progress_state == query.progress_state:
        state_relevance += 0.2
    elif progress_state and query.progress_state:
        belief = query.belief.get("progress_hypothesis", {}) if query.belief else {}
        state_relevance += 0.1 * float(belief.get(progress_state, 0.0))
    if "holding_target" in anchor:
        if bool(anchor.get("holding_target")) == bool(query.held_relevant_count > 0):
            state_relevance += 0.16
        else:
            state_relevance -= 0.08
    if "remaining" in anchor and int(anchor.get("remaining", -1)) == query.remaining_relevant_count:
        state_relevance += 0.12

    task_family = _artifact_task_family(artifact)
    pattern_kind = _artifact_pattern_kind(artifact)
    task_relevance = _task_relevance_score(
        query,
        text=blob,
        task_family=task_family,
        pattern_kind=pattern_kind,
        workflow_stage=progress_state,
    )
    is_fever_query = str(query.task_family or "").startswith("fever")
    is_fever_artifact = str(anchor.get("domain", "") or payload.get("domain", "")) == "fever" or task_family.startswith(
        "fever"
    )
    if is_fever_query and is_fever_artifact:
        fever_pattern = str(payload.get("fever_pattern", "") or "")
        if task_family and task_family == _query_task_family(query):
            task_relevance += 0.35
        elif task_family:
            task_relevance -= 0.35
        if progress_state and progress_state == query.progress_state:
            state_relevance += 0.25
        if "evidence_sufficiency_stop" in fever_pattern:
            if query.progress_state == "need_lookup_or_finish":
                task_relevance += 0.35
            else:
                task_relevance -= 0.25
        if "content_search_route" in fever_pattern:
            if query.progress_state == "need_search":
                task_relevance += 0.45
                state_relevance += 0.20
            elif query.progress_state == "need_lookup_or_finish":
                task_relevance += 0.20
            else:
                task_relevance -= 0.15
        if "no_results_recovery" in fever_pattern:
            if query.progress_state in {"search_failed", "invalid_action"} or query.failure_label:
                task_relevance += 0.35
            else:
                task_relevance -= 0.25
        if "premature_finish_failure" in fever_pattern:
            if query.progress_state in {"search_failed", "invalid_action"} or query.failure_label:
                task_relevance += 0.25
            else:
                task_relevance -= 0.35
    goal_relevance = _goal_relevance_from_text(query, blob)
    goal_relevance += _goal_role_bonus(query, blob, task_family=task_family, pattern_kind=pattern_kind)

    score = _combine_relevance(query, state_relevance, task_relevance, goal_relevance)
    if artifact.kind == ArtifactKind.PROTOTYPE:
        score += 0.32 * artifact.stats.confidence + 0.1 * artifact.stats.utility_avg
    elif artifact.kind == ArtifactKind.REFLECTION:
        score += 0.28 * artifact.stats.failure_rate + 0.12 * artifact.stats.utility_avg
        if not query.failure_label and query.progress_state.startswith("search"):
            score += 0.06
    else:
        score += 0.26 * artifact.stats.confidence + 0.08 * artifact.stats.utility_avg
    action_patterns = _artifact_action_patterns(artifact)
    if action_patterns:
        projected_count = sum(len(_project_rule_actions(query, pattern)) for pattern in action_patterns)
        if projected_count == 0 and artifact.kind != ArtifactKind.REFLECTION:
            score -= 0.14
        elif projected_count > 6:
            score -= 0.12
    if artifact.kind == ArtifactKind.PROTOTYPE and pattern_kind == "closure" and query.remaining_relevant_count > 0:
        score -= 0.75
    if artifact.kind == ArtifactKind.PROTOTYPE and pattern_kind == "plan":
        if _query_task_family(query) == task_family:
            score += 0.18
        if query.progress_state.startswith("search"):
            score += 0.1
    if artifact.kind == ArtifactKind.PROTOTYPE and pattern_kind == "scene_relation":
        relation_kind = str(payload.get("relation_kind", ""))
        source_base = str(payload.get("source_base", ""))
        if goal_relevance >= 0.16:
            score += 0.24
        if relation_kind == "object_location_prior":
            score += 0.26 if query.progress_state.startswith("search") else 0.12
        elif relation_kind == "searched_empty":
            score -= 0.06 if query.progress_state.startswith("search") else 0.12
        if source_base:
            attempts = _search_attempt_count(query, source_base)
            if relation_kind == "searched_empty":
                score += 0.08 if attempts > 0 else -0.04
            elif relation_kind == "object_location_prior":
                score -= 0.12 if attempts > 1 else 0.04
    if artifact.kind == ArtifactKind.PROTOTYPE and pattern_kind == "source_type_prior":
        source_base = str(payload.get("source_base", ""))
        goal_object = str(payload.get("goal_object", ""))
        query_object = str(query.goal_roles.get("object", ""))
        if query.progress_state.startswith("search"):
            score += 0.22
        if goal_object and query_object and goal_object == query_object:
            score += 0.18
        elif goal_object and query_object:
            score -= 0.08
        if source_base:
            attempts = _search_attempt_count(query, source_base)
            if artifact.stats.confidence >= 0.5:
                score -= min(0.18, attempts * 0.06)
            else:
                score -= 0.04
    if (
        artifact.kind == ArtifactKind.REFLECTION
        and str(anchor.get("failure_signature", "")) == "premature_destination_focus"
    ):
        if query.held_relevant_count <= 0 and query.progress_state not in {"carry_target", "finalize"}:
            score += 0.16
    if graph_context:
        linked_refs = set(graph_context.get("linked_refs", set()))
        anchor_state_ids = set(graph_context.get("anchor_state_ids", set()))
        neighbor_state_ids = set(graph_context.get("neighbor_state_ids", set()))
        graph_bonus = 0.0
        for ref in payload.get("graph_refs", []) or []:
            if ref in linked_refs:
                graph_bonus += 0.08
        if payload.get("state_ref") in anchor_state_ids:
            graph_bonus += 0.18
        if payload.get("next_state_ref") in neighbor_state_ids:
            graph_bonus += 0.14
        progress_state = str(anchor.get("progress_state", ""))
        stage_votes = graph_context.get("stage_votes", {})
        if progress_state and progress_state in stage_votes:
            graph_bonus += min(0.12, 0.06 * float(stage_votes[progress_state]))
        role_focus = str(anchor.get("role_focus", "") or payload.get("role_focus", ""))
        role_votes = graph_context.get("role_votes", {})
        if role_focus and role_focus in role_votes:
            graph_bonus += min(0.1, 0.05 * float(role_votes[role_focus]))
        score += min(graph_bonus, 0.34)
    return state_relevance, task_relevance, goal_relevance, score


def _artifact_support_item(
    query: MemoryQuery,
    artifact: MemoryArtifact,
    source: str,
    graph_context: dict | None = None,
) -> SupportItem:
    state_relevance, task_relevance, goal_relevance, score = _artifact_relevance(query, artifact, graph_context)
    dynamic = {
        "artifact_kind": artifact.kind.value,
        "anchor": dict(artifact.anchor),
        "payload": dict(artifact.payload),
    }
    return SupportItem(
        source=source,
        candidate_id=artifact.artifact_id,
        candidate_type=_artifact_candidate_type(artifact),
        summary=artifact.summary,
        score=score,
        task_family=_artifact_task_family(artifact),
        pattern_kind=_artifact_pattern_kind(artifact),
        branch_tag=_artifact_branch_tag(artifact),
        positive=artifact.stats.success,
        negative=artifact.stats.failure,
        stalled=artifact.stats.stalled,
        state_relevance=state_relevance,
        task_relevance=task_relevance,
        goal_relevance=goal_relevance,
        action_patterns=_artifact_action_patterns(artifact),
        dynamic=dynamic,
    )


def _rule_relevance(
    query: MemoryQuery,
    rule: MemoryRule,
    graph_context: dict | None = None,
) -> tuple[float, float, float, float]:
    state_relevance = 0.0
    if rule.condition.get("location_role") and query.location:
        query_location_base = _base_token(query.location)
        if rule.condition.get("location_role") == query_location_base:
            state_relevance += 0.25
        elif rule.condition.get("location_role") in {"container", "support_surface"}:
            if rule.condition.get("location_role") == "container" and query_location_base in _CONTAINER_TYPES:
                state_relevance += 0.16
            if rule.condition.get("location_role") == "support_surface" and query_location_base in _SUPPORT_TYPES:
                state_relevance += 0.16
    if "holding_target" in rule.condition:
        if bool(rule.condition["holding_target"]) == bool(query.held_relevant_count > 0):
            state_relevance += 0.18
        else:
            state_relevance -= 0.12
    if "remaining" in rule.condition:
        remaining = int(rule.condition["remaining"])
        if remaining == query.remaining_relevant_count:
            state_relevance += 0.14

    task_relevance = 0.0
    if rule.task_family:
        if rule.task_family == _query_task_family(query):
            task_relevance += 0.5
        else:
            task_relevance -= 0.22
    if rule.goal_arity == query.required_count:
        task_relevance += 0.16
    else:
        task_relevance -= 0.1
    if rule.progress_state:
        if rule.progress_state == query.progress_state:
            task_relevance += 0.4
        elif (
            rule.rule_type in {RuleType.WORKFLOW, RuleType.CLOSURE}
            and query.progress_state.startswith("search")
            and rule.progress_state.startswith("search")
        ):
            task_relevance += 0.1
        else:
            task_relevance -= 0.14
    if query.failure_label:
        condition_failure = str(rule.condition.get("failure_label", ""))
        if condition_failure == query.failure_label:
            task_relevance += 0.3
        elif rule.rule_type in {RuleType.REPAIR, RuleType.BLOCKED}:
            task_relevance -= 0.12
    elif rule.rule_type in {RuleType.REPAIR, RuleType.BLOCKED} and rule.condition.get("failure_label"):
        task_relevance -= 0.08

    goal_blob = " ".join(
        [
            rule.summary,
            " ".join(str(value) for value in rule.goal_roles.values()),
            " ".join(str(value) for value in rule.condition.values()),
            " ".join(str(value) for value in rule.effect.values()),
        ]
    )
    goal_relevance = _goal_relevance_from_text(query, goal_blob)
    object_role, destination_role = _goal_tokens(query)
    if object_role and str(rule.goal_roles.get("object", "")).lower() == object_role:
        goal_relevance += 0.18
    if destination_role and str(rule.goal_roles.get("destination", "")).lower() == destination_role:
        goal_relevance += 0.16

    score = _combine_relevance(query, state_relevance, task_relevance, goal_relevance)
    if _paper_eq22_27_all_ones():
        score += 1.0 * rule.stats.confidence + 1.0 * min(rule.support / 1.0, 1.0) + 1.0 * min(rule.coverage / 1.0, 1.0)
    else:
        score += (
            0.38 * rule.stats.confidence + 0.18 * min(rule.support / 4.0, 1.0) + 0.12 * min(rule.coverage / 2.0, 1.0)
        )
    score += _rule_specific_action_bonus(query, rule)
    if _paper_eq22_27_all_ones():
        score -= 1.0 * rule.specificity + 1.0 * rule.conflict
    else:
        score -= 0.28 * rule.specificity + 0.24 * rule.conflict
    if rule.rule_type == RuleType.CLOSURE and query.remaining_relevant_count > 0:
        score -= 1.0 if _paper_eq22_27_all_ones() else 0.75
    if rule.rule_type == RuleType.BLOCKED and not query.failure_label and query.progress_state.startswith("search"):
        score += 0.08
    action_patterns = _rule_action_patterns(rule)
    if action_patterns:
        projected_count = sum(len(_project_rule_actions(query, pattern)) for pattern in action_patterns)
        if projected_count == 0 and rule.rule_type != RuleType.BLOCKED:
            score -= 1.0 if _paper_eq22_27_all_ones() else 0.2
        elif projected_count > 4:
            score -= 0.18
    if rule.effect.get("action_role") == "target_object" and not str(query.goal_roles.get("object", "")):
        score -= 0.12
    if rule.effect.get("action_role") == "goal_destination" and not str(query.goal_roles.get("destination", "")):
        score -= 0.12
    if (
        rule.effect.get("action_role") == "goal_destination"
        and query.held_relevant_count <= 0
        and query.progress_state not in {"carry_target", "finalize"}
    ):
        score -= 0.8
    if rule.effect.get("action_role") == "non_target_object" and str(query.goal_roles.get("object", "")):
        score += 0.18
    if graph_context:
        role_votes = graph_context.get("role_votes", {})
        stage_votes = graph_context.get("stage_votes", {})
        location_role = str(rule.condition.get("location_role", ""))
        if location_role and location_role in role_votes:
            score += min(0.12, 0.06 * float(role_votes[location_role]))
        if rule.progress_state and rule.progress_state in stage_votes:
            score += min(0.12, 0.06 * float(stage_votes[rule.progress_state]))
    return state_relevance, task_relevance, goal_relevance, score


def _rule_support_item(
    query: MemoryQuery,
    rule: MemoryRule,
    source: str,
    graph_context: dict | None = None,
) -> SupportItem:
    state_relevance, task_relevance, goal_relevance, score = _rule_relevance(query, rule, graph_context)
    return SupportItem(
        source=source,
        candidate_id=rule.rule_id,
        candidate_type=_rule_candidate_type(rule.rule_type),
        summary=rule.summary,
        score=score,
        task_family=rule.task_family,
        pattern_kind=rule.rule_type.value,
        branch_tag=_rule_branch_tag(rule.rule_type),
        positive=rule.stats.success,
        negative=rule.stats.failure,
        stalled=rule.stats.stalled,
        state_relevance=state_relevance,
        task_relevance=task_relevance,
        goal_relevance=goal_relevance,
        action_patterns=_rule_action_patterns(rule),
    )


def _rule_bundle(
    query: MemoryQuery,
    *,
    facts: list[SupportItem],
    local_rules: list[SupportItem],
    global_rules: list[SupportItem],
) -> SupportBundle:
    bundle = SupportBundle(
        query=f"query:{query.goal}",
        goal_object=str(query.goal_roles.get("object", "")),
        goal_destination=str(query.goal_roles.get("destination", "")),
        goal_tool=str(query.goal_roles.get("tool", "")),
        progress_state=str(query.progress_state or ""),
    )
    bundle.local_graph_items = _dedupe_support_items(list(facts), len(facts))
    bundle.local_promoted_items = _dedupe_support_items(list(local_rules), len(local_rules))
    bundle.global_promoted_items = _dedupe_support_items(list(global_rules), len(global_rules))
    bundle.fact_items = _dedupe_support_items(facts, 2)
    bundle.local_items = _dedupe_support_items(
        list(bundle.fact_items) + list(local_rules), len(bundle.fact_items) + len(local_rules)
    )
    bundle.global_items = _dedupe_support_items(list(global_rules), len(global_rules))
    bundle.workflow_items = _dedupe_support_items(
        [item for item in local_rules + global_rules if item.pattern_kind == RuleType.WORKFLOW.value], 2
    )
    bundle.precondition_items = _dedupe_support_items(
        [item for item in local_rules + global_rules if item.pattern_kind == RuleType.PRECONDITION.value], 2
    )
    bundle.repair_items = _dedupe_support_items(
        [item for item in local_rules + global_rules if item.pattern_kind == RuleType.REPAIR.value], 2
    )
    bundle.closure_items = _dedupe_support_items(
        [item for item in local_rules + global_rules if item.pattern_kind == RuleType.CLOSURE.value], 2
    )

    bundle.routing_decisions = {
        "fact": "local" if bundle.fact_items else "none",
        "workflow": (
            "mixed" if (bundle.workflow_items and global_rules) else ("local" if bundle.workflow_items else "none")
        ),
        "precondition": (
            "mixed"
            if (bundle.precondition_items and global_rules)
            else ("local" if bundle.precondition_items else "none")
        ),
        "repair": "mixed" if (bundle.repair_items and global_rules) else ("local" if bundle.repair_items else "none"),
        "closure": (
            "mixed" if (bundle.closure_items and global_rules) else ("local" if bundle.closure_items else "none")
        ),
        "blocked": (
            "local"
            if any(item.pattern_kind == RuleType.BLOCKED.value for item in (local_rules + global_rules))
            else "none"
        ),
    }
    bundle.routing_weights = {
        slot: {"local": 1.0 if value != "none" else 0.0, "global": 0.0, "none": 0.0 if value != "none" else 1.0}
        for slot, value in bundle.routing_decisions.items()
    }

    workflow_hints: list[str] = []
    for item in bundle.workflow_items:
        workflow_hints.append(item.summary.replace("Workflow: ", "").replace("Workflow pattern: ", ""))
    bundle.workflow_hints = workflow_hints[:3]
    bundle.task_need_analysis = (
        ["Need transferable stage guidance and admissible-action constraints for the current step."]
        if str(query.progress_state or "").startswith("search")
        else ["Need process/delivery guidance grounded in current admissible actions."]
    )
    bundle.local_graph_contribution = list(bundle.fact_items[:2])
    bundle.local_promoted_contribution = list((bundle.workflow_items + bundle.precondition_items)[:2])
    bundle.global_promoted_contribution = list((bundle.workflow_items + bundle.precondition_items)[2:4])
    bundle.fused_support_items = _dedupe_support_items(
        list(bundle.local_graph_contribution)
        + list(bundle.local_promoted_contribution)
        + list(bundle.global_promoted_contribution),
        6,
    )

    # Resolve blocked / suggested actions from rules against the current admissible set.
    blocked_actions: list[str] = []
    suggested_actions: list[str] = []
    for source_items in (local_rules, global_rules):
        for item in source_items:
            if item.pattern_kind == RuleType.BLOCKED.value:
                for pattern in item.action_patterns:
                    blocked_actions.extend(_project_rule_actions(query, pattern))
            elif item.pattern_kind in {RuleType.PRECONDITION.value, RuleType.REPAIR.value, RuleType.CLOSURE.value}:
                for pattern in item.action_patterns:
                    suggested_actions.extend(_project_rule_actions(query, pattern))

    bundle.blocked_actions = list(dict.fromkeys(blocked_actions))
    bundle.suggested_actions = _filter_suggested_actions(
        query,
        list(dict.fromkeys(suggested_actions)),
    )
    return bundle


def _artifact_bundle(
    query: MemoryQuery,
    *,
    facts: list[SupportItem],
    graph_items: list[SupportItem],
    local_artifacts: list[SupportItem],
    global_artifacts: list[SupportItem],
    local_rules: list[SupportItem],
    global_rules: list[SupportItem],
) -> SupportBundle:
    search_like = str(query.progress_state or "").startswith("search")
    process_like = str(query.progress_state or "") == "process_target"
    deliver_like = str(query.progress_state or "") in {"carry_target", "deliver_target", "finalize"}

    def _search_phase_misaligned(item: SupportItem) -> bool:
        if not search_like:
            return False
        summary = str(item.summary or "").lower()
        return any(
            marker in summary
            for marker in (
                "holding relevant",
                "holding target",
                "carry_target",
                "carry target",
            )
        )

    def _search_relation_priority(item: SupportItem) -> float:
        relation_kind = str(item.dynamic.get("relation_kind", ""))
        score = float(item.score) + 0.35 * float(item.goal_relevance)
        if relation_kind == "object_location_prior":
            score += 1.4
        elif relation_kind == "searched_empty":
            score += 1.0
        if _search_phase_misaligned(item):
            score -= 4.0
        return score

    def _needs_for_phase() -> list[str]:
        if search_like:
            return [
                "Need concrete scene grounding for where to continue search.",
                "Need domain-specific search prior from local experience.",
                "Need transferable task-family guidance only when it sharpens search.",
            ]
        if process_like:
            return [
                "Need processing prerequisite and tool guidance.",
                "Need local domain detail about how processing is usually grounded.",
                "Need transferable process workflow as backup.",
            ]
        if deliver_like:
            return [
                "Need delivery-progress guidance that stays consistent with goal completion.",
                "Need local grounding for final destination handling.",
                "Need transferable completion workflow only as support.",
            ]
        return [
            "Need support that best advances the current task phase.",
            "Prefer scene-grounded local evidence plus transferable guidance when they agree.",
        ]

    def _phase_route_score(item: SupportItem, route: str) -> float:
        summary = str(item.summary or "").lower()
        score = float(item.score) + 0.45 * float(item.goal_relevance) + 0.18 * float(item.task_relevance)
        if search_like:
            if route == "local_graph":
                score += 0.95
                if item.pattern_kind in {"fact", "scene_relation", "graph_transition", "graph_anchor"}:
                    score += 0.5
            elif route == "local_promoted":
                if item.pattern_kind in {
                    "scene_relation",
                    "plan",
                    RuleType.PRECONDITION.value,
                    RuleType.WORKFLOW.value,
                    "workflow",
                }:
                    score += 0.35
            elif route == "global_promoted":
                if item.pattern_kind in {"plan", RuleType.PRECONDITION.value, RuleType.WORKFLOW.value, "workflow"}:
                    score += 0.22
            if _search_phase_misaligned(item):
                score -= 4.0
            if "holding relevant" in summary:
                score -= 4.0
            if "graph anchor:" in summary and item.goal_relevance < 0.3:
                score -= 1.5
        elif process_like:
            if route == "local_promoted" and item.pattern_kind in {
                RuleType.PRECONDITION.value,
                "plan",
                "workflow",
                RuleType.WORKFLOW.value,
            }:
                score += 0.7
            if route == "global_promoted" and item.pattern_kind in {
                RuleType.PRECONDITION.value,
                "plan",
                "workflow",
                RuleType.WORKFLOW.value,
            }:
                score += 0.45
            if route == "local_graph" and item.pattern_kind in {"graph_transition", "fact"}:
                score += 0.2
        elif deliver_like:
            if route == "local_promoted" and item.pattern_kind in {"closure", RuleType.CLOSURE.value, "plan"}:
                score += 0.6
            if route == "global_promoted" and item.pattern_kind in {
                "closure",
                RuleType.CLOSURE.value,
                "plan",
                RuleType.WORKFLOW.value,
                "workflow",
            }:
                score += 0.45
            if route == "local_graph" and item.pattern_kind in {"graph_transition", "fact"}:
                score += 0.2
        return score

    def _pick_route_contribution(items: list[SupportItem], route: str, limit: int = 2) -> list[SupportItem]:
        ranked = sorted(items, key=lambda item: _phase_route_score(item, route), reverse=True)
        filtered: list[SupportItem] = []
        for item in ranked:
            if search_like and route != "local_graph" and item.pattern_kind == "fact":
                continue
            if search_like and route == "global_promoted" and item.pattern_kind == "scene_relation":
                pass
            filtered.append(item)
            if len(filtered) >= limit:
                break
        return _dedupe_support_items(filtered, limit)

    bundle = SupportBundle(
        query=f"query:{query.goal}",
        goal_object=str(query.goal_roles.get("object", "")),
        goal_destination=str(query.goal_roles.get("destination", "")),
        goal_tool=str(query.goal_roles.get("tool", "")),
        progress_state=str(query.progress_state or ""),
    )
    anchor_items = [
        item
        for item in graph_items
        if item.pattern_kind == "graph_anchor" and (not search_like or item.goal_relevance >= 0.22)
    ]
    transition_items = [item for item in graph_items if item.pattern_kind == "graph_transition"]
    search_facts = list(facts)
    if search_like:
        search_facts = [
            item
            for item in search_facts
            if (item.goal_relevance >= 0.18 or "relevant visible" in item.summary.lower())
            and not _search_phase_misaligned(item)
        ]
    bundle.fact_items = _dedupe_support_items(list(search_facts) + anchor_items, 3)
    bundle.local_graph_items = _dedupe_support_items(
        list(bundle.fact_items) + transition_items, len(bundle.fact_items) + len(transition_items)
    )
    bundle.local_promoted_items = _dedupe_support_items(
        list(local_artifacts) + list(local_rules), len(local_artifacts) + len(local_rules)
    )
    bundle.global_promoted_items = _dedupe_support_items(
        list(global_artifacts) + list(global_rules), len(global_artifacts) + len(global_rules)
    )
    local_pool = list(bundle.local_graph_items) + list(bundle.local_promoted_items)
    global_pool = list(bundle.global_promoted_items)
    bundle.local_items = _dedupe_support_items(local_pool, len(local_pool))
    bundle.global_items = _dedupe_support_items(global_pool, len(global_pool))

    combined = transition_items + local_artifacts + global_artifacts + local_rules + global_rules
    query_family = _query_task_family(query)
    global_task_plan_candidates = [
        item
        for item in global_artifacts + global_rules
        if item.pattern_kind == "plan"
        and (not query_family or not item.task_family or item.task_family == query_family)
    ]
    if not global_task_plan_candidates:
        global_task_plan_candidates = [
            item
            for item in global_artifacts + global_rules
            if item.pattern_kind in {"closure", RuleType.CLOSURE.value, RuleType.WORKFLOW.value, "workflow"}
            and (not query_family or not item.task_family or item.task_family == query_family)
        ]
    bundle.global_task_plan_items = _dedupe_support_items(
        sorted(
            global_task_plan_candidates, key=lambda item: (item.score, item.positive, item.goal_relevance), reverse=True
        ),
        2,
    )
    global_task_plan_ids = {item.candidate_id for item in bundle.global_task_plan_items}
    relation_candidates = [
        item for item in combined if item.pattern_kind == "scene_relation" and not _search_phase_misaligned(item)
    ]
    if search_like:
        relation_candidates = sorted(relation_candidates, key=_search_relation_priority, reverse=True)
    bundle.relation_items = _dedupe_support_items(relation_candidates, 2)
    bundle.plan_items = _dedupe_support_items(
        [item for item in combined if item.pattern_kind == "plan" and item.candidate_id not in global_task_plan_ids],
        1 if search_like else 2,
    )
    bundle.workflow_items = _dedupe_support_items(
        [
            item
            for item in combined
            if item.pattern_kind in {RuleType.WORKFLOW.value, "workflow", "transition", "candidate", "episode_graph"}
            or item.dynamic.get("artifact_kind") == ArtifactKind.PROTOTYPE.value
        ],
        3,
    )
    bundle.workflow_items = [
        item
        for item in bundle.workflow_items
        if item.pattern_kind not in {"scene_relation", "plan"} and (not search_like or item.goal_relevance >= 0.14)
    ]
    bundle.precondition_items = _dedupe_support_items(
        [
            item
            for item in combined
            if item.candidate_type == CandidateType.PRECONDITION or item.pattern_kind == RuleType.PRECONDITION.value
        ],
        3,
    )
    bundle.precondition_items = [
        item for item in bundle.precondition_items if item.pattern_kind not in {"scene_relation", "plan"}
    ]
    bundle.repair_items = _dedupe_support_items(
        [
            item
            for item in combined
            if item.pattern_kind == RuleType.REPAIR.value
            or item.dynamic.get("artifact_kind") == ArtifactKind.REFLECTION.value
        ],
        3,
    )
    bundle.reflection_items = _dedupe_support_items(
        [item for item in combined if item.dynamic.get("artifact_kind") == ArtifactKind.REFLECTION.value],
        3,
    )
    bundle.closure_items = _dedupe_support_items(
        [item for item in combined if item.pattern_kind == RuleType.CLOSURE.value or item.pattern_kind == "closure"],
        2,
    )

    def _route_strength(items: list[SupportItem], route: str) -> float:
        if not items:
            return 0.0
        scores = [_phase_route_score(item, route) for item in items]
        top = max(scores)
        mean_top = sum(sorted(scores, reverse=True)[:2]) / min(len(scores), 2)
        return max(0.0, 0.7 * top + 0.3 * mean_top)

    def _source_weights(
        *,
        local_items: list[SupportItem],
        global_items: list[SupportItem],
        slot: str,
    ) -> dict[str, float]:
        local_strength = _route_strength(local_items, "local_graph")
        local_strength = max(local_strength, _route_strength(local_items, "local_promoted"))
        global_strength = _route_strength(global_items, "global_promoted")

        local_prior = 1.0
        global_prior = 0.55
        if slot in {"fact", "relation", "repair", "blocked"}:
            global_prior = 0.15
        elif slot == "task_plan":
            local_prior = 0.35
            global_prior = 0.95
        elif slot in {"workflow", "precondition", "closure"}:
            global_prior = 0.45

        if search_like:
            if slot != "task_plan":
                global_prior *= 0.45
            local_prior *= 1.15
        elif process_like and slot in {"task_plan", "workflow", "precondition"}:
            global_prior *= 1.15
        elif deliver_like and slot in {"task_plan", "workflow", "closure"}:
            global_prior *= 1.1

        local_raw = local_strength * local_prior
        global_raw = global_strength * global_prior
        total = local_raw + global_raw
        if total <= 1e-8:
            return {"local": 0.0, "global": 0.0, "none": 1.0}
        return {
            "local": round(local_raw / total, 3),
            "global": round(global_raw / total, 3),
            "none": 0.0,
        }

    route_slots = {
        "fact": (bundle.fact_items, []),
        "relation": (bundle.relation_items, []),
        "task_plan": ([], bundle.global_task_plan_items),
        "plan": (bundle.plan_items, []),
        "workflow": (
            [item for item in bundle.workflow_items if not str(item.source).startswith("global")],
            [item for item in bundle.workflow_items if str(item.source).startswith("global")],
        ),
        "precondition": (
            [item for item in bundle.precondition_items if not str(item.source).startswith("global")],
            [item for item in bundle.precondition_items if str(item.source).startswith("global")],
        ),
        "repair": (
            [item for item in bundle.repair_items if not str(item.source).startswith("global")],
            [item for item in bundle.repair_items if str(item.source).startswith("global")],
        ),
        "closure": (
            [item for item in bundle.closure_items if not str(item.source).startswith("global")],
            [item for item in bundle.closure_items if str(item.source).startswith("global")],
        ),
        "blocked": (
            [
                item
                for item in combined
                if item.pattern_kind == RuleType.BLOCKED.value and not str(item.source).startswith("global")
            ],
            [
                item
                for item in combined
                if item.pattern_kind == RuleType.BLOCKED.value and str(item.source).startswith("global")
            ],
        ),
    }
    bundle.routing_weights = {
        slot: _source_weights(local_items=local_items, global_items=global_items, slot=slot)
        for slot, (local_items, global_items) in route_slots.items()
    }
    bundle.routing_decisions = {
        slot: (
            "none"
            if weights.get("none", 0.0) >= 1.0
            else (
                "mixed"
                if weights.get("local", 0.0) > 0.0 and weights.get("global", 0.0) > 0.0
                else "global" if weights.get("global", 0.0) > 0.0 else "local"
            )
        )
        for slot, weights in bundle.routing_weights.items()
    }

    workflow_hints: list[str] = []
    for item in bundle.plan_items + bundle.workflow_items:
        workflow_hints.append(item.summary.replace("Prototype: ", "").replace("Workflow: ", ""))
    bundle.workflow_hints = workflow_hints[:4]

    blocked_votes: dict[str, float] = {}
    suggested_votes: dict[str, float] = {}
    for item in combined:
        artifact_kind = item.dynamic.get("artifact_kind")
        if item.pattern_kind == RuleType.BLOCKED.value:
            for pattern in item.action_patterns:
                for action in _project_rule_actions(query, pattern):
                    blocked_votes[action] = blocked_votes.get(action, 0.0) + max(item.score, 0.2)
            continue
        if artifact_kind == ArtifactKind.REFLECTION.value:
            payload = dict(item.dynamic.get("payload", {}))
            diagnosis = str(payload.get("diagnosis", ""))
            if diagnosis in {"wrong_target_object", "premature_destination_focus", "premature_non_goal_place"}:
                for pattern in payload.get("avoid_patterns", []) or []:
                    for action in _project_rule_actions(query, pattern):
                        blocked_votes[action] = blocked_votes.get(action, 0.0) + max(item.score, 0.2)
                if diagnosis == "wrong_target_object":
                    for action in _project_wrong_target_actions(query):
                        blocked_votes[action] = blocked_votes.get(action, 0.0) + max(item.score * 0.95, 0.18)
            for pattern in payload.get("repair_patterns", []) or []:
                if (
                    query.held_relevant_count <= 0
                    and query.progress_state not in {"carry_target", "finalize"}
                    and "goal_destination" in str(pattern)
                ):
                    continue
                for action in _project_rule_actions(query, pattern):
                    suggested_votes[action] = suggested_votes.get(action, 0.0) + max(item.score * 0.9, 0.1)
            continue
        if item.pattern_kind in {RuleType.PRECONDITION.value, RuleType.REPAIR.value, RuleType.CLOSURE.value}:
            for pattern in item.action_patterns:
                for action in _project_rule_actions(query, pattern):
                    suggested_votes[action] = suggested_votes.get(action, 0.0) + max(item.score, 0.1)
            continue
        if artifact_kind == ArtifactKind.PROTOTYPE.value:
            for pattern in item.action_patterns:
                for action in _project_rule_actions(query, pattern):
                    suggested_votes[action] = suggested_votes.get(action, 0.0) + max(item.score * 0.75, 0.08)

    ranked_blocked_actions = [
        action for action, _ in sorted(blocked_votes.items(), key=lambda item: item[1], reverse=True)[:8]
    ]
    bundle.blocked_actions = _filter_blocked_actions(query, ranked_blocked_actions)
    blocked_for_suggestions = set(bundle.blocked_actions)
    delivery_suggestions = _goal_delivery_actions(query)
    bundle.suggested_actions = _filter_suggested_actions(
        query,
        delivery_suggestions
        + [
            action
            for action, _ in sorted(suggested_votes.items(), key=lambda item: item[1], reverse=True)[:8]
            if action not in blocked_for_suggestions
        ],
    )
    warnings: list[str] = []
    if any(item.dynamic.get("artifact_kind") == ArtifactKind.REFLECTION.value for item in combined):
        for item in combined:
            if item.dynamic.get("artifact_kind") != ArtifactKind.REFLECTION.value:
                continue
            summary = str(item.summary or "").strip()
            if summary:
                warnings.append(summary)
    bundle.warnings = list(dict.fromkeys(warnings))[:4]
    bundle.task_need_analysis = _needs_for_phase()
    bundle.local_graph_contribution = _pick_route_contribution(bundle.local_graph_items, "local_graph", limit=2)
    bundle.local_promoted_contribution = _pick_route_contribution(
        bundle.local_promoted_items, "local_promoted", limit=2
    )
    bundle.global_promoted_contribution = _pick_route_contribution(
        bundle.global_promoted_items, "global_promoted", limit=2
    )
    bundle.fused_support_items = _dedupe_support_items(
        list(bundle.local_graph_contribution)
        + list(bundle.local_promoted_contribution)
        + list(bundle.global_promoted_contribution),
        6,
    )
    return bundle


def _state_anchor_score(query: MemoryQuery, node_payload: dict) -> float:
    score = 0.0
    query_tokens = _query_tokens(query)
    state_blob = " ".join(
        [
            str(node_payload.get("location") or ""),
            " ".join(node_payload.get("visible_objects", []) or []),
            " ".join(node_payload.get("held_objects", []) or []),
            " ".join(node_payload.get("open_containers", []) or []),
            str(node_payload.get("workflow_stage") or ""),
        ]
    )
    score += _jaccard(query_tokens, _tokenize(state_blob))

    location = node_payload.get("location")
    if query.location and location == query.location:
        score += 0.45

    node_stage = node_payload.get("workflow_stage")
    if query.current_stage and node_stage == query.current_stage:
        score += 0.35

    visible = set(node_payload.get("visible_objects", []) or [])
    held = set(node_payload.get("held_objects", []) or [])
    score += 0.18 * len(visible & _keyword_suffixes(query, "visible+="))
    score += 0.22 * len(held & _keyword_suffixes(query, "holding+="))
    if (
        query.current_stage
        and node_payload.get("workflow_stage")
        and node_payload.get("workflow_stage") != query.current_stage
    ):
        score -= 0.08
    return score


def _candidate_rank(
    query: MemoryQuery,
    candidates: list[PromotionCandidate] | tuple[PromotionCandidate, ...] | any,
    source: str,
) -> list[SupportItem]:
    ranked: list[SupportItem] = []
    for candidate in candidates:
        lexical = _jaccard(_query_tokens(query), _candidate_tokens(candidate))
        stage_bonus = 0.12 if query.current_stage and query.current_stage in candidate.summary else 0.0
        scene_bonus = 0.08 if query.scene_id and query.scene_id in candidate.source_scenes else 0.0
        action_bonus = _action_bonus(candidate, query.admissible_actions)
        pattern_kind = str(candidate.structure.get("pattern_kind", ""))
        task_family = str(candidate.structure.get("task_family", ""))
        candidate_text = candidate.summary + " " + " ".join(str(value) for value in candidate.structure.values())
        state_relevance = lexical + stage_bonus + 0.4 * scene_bonus
        task_relevance = _task_relevance_score(
            query,
            text=candidate_text,
            task_family=task_family,
            pattern_kind=pattern_kind,
        )
        goal_relevance = _goal_relevance_from_text(query, candidate_text)
        goal_relevance += _goal_role_bonus(
            query,
            candidate_text,
            task_family=task_family,
            pattern_kind=pattern_kind,
        )
        score = action_bonus + _graph_evidence_bonus(candidate) - _candidate_penalty(candidate)
        score += _combine_relevance(query, state_relevance, task_relevance, goal_relevance)
        if candidate.candidate_type == CandidateType.REPAIR and not query.failure_label:
            score -= 1.0 if _paper_eq22_27_all_ones() else 1.1
        if candidate.candidate_type == CandidateType.FAILURE and not query.failure_label:
            score -= 1.0 if _paper_eq22_27_all_ones() else 0.65
        if pattern_kind == "closure" and query.remaining_relevant_count > 0:
            score -= 1.0 if _paper_eq22_27_all_ones() else 0.9
        ranked.append(
            SupportItem(
                source=source,
                candidate_id=candidate.candidate_id,
                candidate_type=candidate.candidate_type,
                summary=candidate.summary,
                score=score,
                task_family=task_family,
                pattern_kind=pattern_kind,
                branch_tag=_branch_tag(
                    candidate.candidate_type,
                    pattern_kind=str(candidate.structure.get("pattern_kind", "")),
                    positive=candidate.positive,
                    negative=candidate.negative,
                    stalled=candidate.stalled,
                ),
                positive=candidate.positive,
                negative=candidate.negative,
                stalled=candidate.stalled,
                state_relevance=state_relevance,
                task_relevance=task_relevance,
                goal_relevance=goal_relevance,
            )
        )
    ranked.sort(key=lambda item: item.score, reverse=True)
    return ranked


def _merge_ranked_items(items: list[SupportItem], limit: int) -> list[SupportItem]:
    merged: dict[str, SupportItem] = {}
    for item in items:
        existing = merged.get(item.candidate_id)
        if existing is None or item.score > existing.score:
            merged[item.candidate_id] = item
    return sorted(merged.values(), key=lambda item: item.score, reverse=True)[:limit]


def _dedupe_support_items(items: list[SupportItem], limit: int) -> list[SupportItem]:
    deduped: list[SupportItem] = []
    seen: set[tuple[str, tuple[str, ...]]] = set()
    for item in items:
        key = (item.summary, tuple(item.action_patterns))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
        if len(deduped) >= limit:
            break
    return deduped


class LightweightRetriever:
    def __init__(
        self,
        local_top_k: int = 3,
        global_top_k: int = 3,
        router: HeuristicSupportRouter | StudentSupportRouter | None = None,
    ) -> None:
        self.local_top_k = local_top_k
        self.global_top_k = global_top_k
        self.router = router

    def retrieve(
        self,
        query: MemoryQuery,
        local_memory: LocalGraphMemory,
        global_memory: GlobalGraphMemory,
    ) -> SupportBundle:
        if local_memory.artifacts_by_id or global_memory.artifacts_by_id:
            facts = self._ground_facts(query, local_memory)
            graph_context = self._anchor_subgraph_context(query, local_memory)
            local_artifacts = self._rank_artifacts(
                query,
                local_memory.artifacts_by_id.values(),
                source="local_artifact",
                graph_context=graph_context,
            )
            global_artifacts = self._rank_artifacts(
                query,
                global_memory.artifacts_by_id.values(),
                source="global_artifact",
                graph_context=graph_context,
            )
            local_rules = self._rank_rules(
                query,
                local_memory.rules_by_id.values(),
                source="local_rule",
                graph_context=graph_context,
            )
            global_rules = self._rank_rules(
                query,
                global_memory.rules_by_id.values(),
                source="global_rule",
                graph_context=graph_context,
            )
            return _artifact_bundle(
                query,
                facts=_joint_select(query, facts, 2),
                graph_items=_joint_select(query, graph_context.get("items", []), 3),
                local_artifacts=_joint_select(query, local_artifacts, self.local_top_k),
                global_artifacts=_joint_select(query, global_artifacts, self.global_top_k),
                local_rules=_joint_select(query, local_rules, max(1, self.local_top_k - 1)),
                global_rules=_joint_select(query, global_rules, max(1, self.global_top_k - 1)),
            )
        if local_memory.rules_by_id or global_memory.rules_by_id:
            facts = self._ground_facts(query, local_memory)
            local_rules = self._rank_rules(query, local_memory.rules_by_id.values(), source="local_rule")
            global_rules = self._rank_rules(query, global_memory.rules_by_id.values(), source="global_rule")
            bundle = _rule_bundle(
                query,
                facts=_joint_select(query, facts, 2),
                local_rules=_joint_select(query, local_rules, self.local_top_k),
                global_rules=_joint_select(query, global_rules, self.global_top_k),
            )
            return bundle
        query_text = f"{query.goal} | {query.current_stage or 'unknown'}"
        bundle = SupportBundle(query=query_text)

        local_ranked = self._rank_local(query, local_memory)
        global_ranked = self._rank_global(query, global_memory)

        bundle.local_items = _joint_select(query, local_ranked, self.local_top_k)
        bundle.global_items = _joint_select(query, global_ranked, self.global_top_k)
        route_bundle(query, bundle, router=self.router)
        return bundle

    def _ground_facts(self, query: MemoryQuery, local_memory: LocalGraphMemory) -> list[SupportItem]:
        if not local_memory.nodes_by_signature:
            return []
        facts: list[SupportItem] = []
        object_role, destination_role = _goal_tokens(query)
        tool_role = _goal_tool_token(query)
        for node in local_memory.nodes_by_signature.values():
            if node.node_type != NodeType.STATE:
                continue
            state_relevance = _state_anchor_score(query, node.payload)
            if state_relevance <= 0:
                continue
            task_relevance = _task_relevance_score(
                query,
                workflow_stage=str(node.payload.get("workflow_stage") or ""),
            )
            goal_relevance = _goal_relevance_from_state(query, node.payload)
            location = str(node.payload.get("location") or "")
            visible = list(node.payload.get("visible_objects", []) or [])
            held = list(node.payload.get("held_objects", []) or [])
            visible_focus = [
                item
                for item in visible
                if any(token and token in str(item).lower() for token in (object_role, destination_role, tool_role))
            ]
            held_focus = [
                item
                for item in held
                if any(token and token in str(item).lower() for token in (object_role, destination_role, tool_role))
            ]
            parts: list[str] = []
            location_tokens = _tokenize(location)
            location_is_goal_aligned = any(
                token and token in location_tokens for token in (destination_role, tool_role)
            )
            if location and (goal_relevance >= 0.18 or visible_focus or held_focus or location_is_goal_aligned):
                parts.append(f"near {location}")
            if visible_focus:
                parts.append("relevant visible " + ", ".join(visible_focus[:4]))
            if held_focus:
                parts.append("holding relevant " + ", ".join(held_focus[:2]))
            if not parts:
                continue
            facts.append(
                SupportItem(
                    source="local_graph",
                    candidate_id=f"fact:{node.node_id}",
                    candidate_type=CandidateType.PRECONDITION,
                    summary="Local state fact: " + "; ".join(parts) + ".",
                    score=_combine_relevance(query, state_relevance + 0.12, task_relevance, goal_relevance),
                    task_family="",
                    pattern_kind="fact",
                    branch_tag="success_branch",
                    positive=max(node.stats.positive, 0),
                    negative=max(node.stats.negative, 0),
                    stalled=max(node.stats.stalled, 0),
                    state_relevance=state_relevance,
                    task_relevance=task_relevance,
                    goal_relevance=goal_relevance,
                )
            )
        return sorted(facts, key=lambda item: item.score, reverse=True)

    def _anchor_subgraph_context(self, query: MemoryQuery, local_memory: LocalGraphMemory) -> dict:
        node_by_id = {node.node_id: node for node in local_memory.nodes_by_signature.values()}
        if not node_by_id:
            return {
                "items": [],
                "anchor_state_ids": set(),
                "neighbor_state_ids": set(),
                "linked_refs": set(),
                "role_votes": {},
                "stage_votes": {},
            }

        outgoing: dict[str, list] = defaultdict(list)
        for edge in local_memory.edges_by_signature.values():
            outgoing[edge.src].append(edge)

        anchors: list[tuple[GraphNode, float, float, float, float]] = []
        for node in node_by_id.values():
            if node.node_type != NodeType.STATE:
                continue
            state_relevance = _state_anchor_score(query, node.payload)
            if state_relevance <= 0:
                continue
            task_relevance = _task_relevance_score(
                query,
                text=str(node.payload.get("workflow_stage") or ""),
                workflow_stage=str(node.payload.get("workflow_stage") or ""),
            )
            goal_relevance = _goal_relevance_from_state(query, node.payload)
            anchor_score = _combine_relevance(query, state_relevance, task_relevance, goal_relevance)
            if anchor_score > 0.05:
                anchors.append((node, state_relevance, task_relevance, goal_relevance, anchor_score))
        anchors.sort(key=lambda item: item[4], reverse=True)
        anchors = anchors[:3]

        items: list[SupportItem] = []
        anchor_state_ids: set[str] = set()
        neighbor_state_ids: set[str] = set()
        linked_refs: set[str] = set()
        role_votes: dict[str, float] = defaultdict(float)
        stage_votes: dict[str, float] = defaultdict(float)
        transition_votes: dict[tuple[str, str], float] = defaultdict(float)

        for state_node, state_rel, task_rel, goal_rel, anchor_score in anchors:
            anchor_state_ids.add(state_node.node_id)
            linked_refs.add(state_node.node_id)
            stage = str(state_node.payload.get("workflow_stage") or "")
            if stage:
                stage_votes[stage] += max(anchor_score, 0.1)
            location = str(state_node.payload.get("location") or "")
            location_role = _location_role(location) if location else ""
            if location_role:
                role_votes[location_role] += max(anchor_score, 0.1)
            anchor_parts: list[str] = []
            if location:
                anchor_parts.append(f"center on {location}")
            if stage:
                anchor_parts.append(f"during {stage}")
            if anchor_parts and (goal_rel >= 0.16 or query.progress_state in {"process_target", "deliver_target"}):
                items.append(
                    SupportItem(
                        source="local_graph",
                        candidate_id=f"graph_anchor:{state_node.node_id}",
                        candidate_type=CandidateType.WORKFLOW,
                        summary="Graph anchor: similar subgraphs " + " ".join(anchor_parts) + ".",
                        score=_combine_relevance(query, state_rel + 0.08, task_rel + 0.06, goal_rel),
                        pattern_kind="graph_anchor",
                        branch_tag="success_branch",
                        positive=max(state_node.stats.positive, 0),
                        negative=max(state_node.stats.negative, 0),
                        stalled=max(state_node.stats.stalled, 0),
                        state_relevance=state_rel,
                        task_relevance=task_rel,
                        goal_relevance=goal_rel,
                        dynamic={"node_id": state_node.node_id, "location_role": location_role},
                    )
                )

            for edge in outgoing.get(state_node.node_id, []):
                if edge.edge_type != EdgeType.TEMPORAL:
                    continue
                action_node = node_by_id.get(edge.dst)
                if action_node is None or action_node.node_type != NodeType.ACTION:
                    continue
                linked_refs.add(edge.signature)
                for next_edge in outgoing.get(action_node.node_id, []):
                    if next_edge.edge_type != EdgeType.TEMPORAL:
                        continue
                    next_state = node_by_id.get(next_edge.dst)
                    if next_state is None or next_state.node_type != NodeType.STATE:
                        continue
                    action_alignment = _graph_transition_alignment(query, action_node.payload, next_state.payload)
                    if action_alignment <= 0.02:
                        continue
                    linked_refs.add(next_edge.signature)
                    neighbor_state_ids.add(next_state.node_id)
                    linked_refs.add(next_state.node_id)
                    next_location = str(next_state.payload.get("location") or "")
                    next_role = _location_role(next_location) if next_location else ""
                    next_stage = str(next_state.payload.get("workflow_stage") or "")
                    next_goal = _goal_relevance_from_state(query, next_state.payload)
                    next_task = _task_relevance_score(
                        query,
                        workflow_stage=next_stage,
                        text=next_stage,
                    )
                    transition_score = (
                        anchor_score
                        + action_alignment
                        + 0.14 * next_edge.stats.confidence
                        + 0.08 * next_task
                        + 0.08 * next_goal
                    )
                    if next_stage and next_stage == query.progress_state:
                        transition_score += 0.06
                    if next_role:
                        role_votes[next_role] += max(transition_score, 0.08)
                    if next_stage:
                        stage_votes[next_stage] += max(transition_score, 0.08)
                    if next_role or next_stage:
                        key = (next_role or _location_role(next_location), next_stage or stage)
                        transition_votes[key] += transition_score

        ranked_transitions = sorted(transition_votes.items(), key=lambda item: item[1], reverse=True)[:2]
        for idx, ((next_role, next_stage), vote) in enumerate(ranked_transitions):
            fragments: list[str] = []
            if next_role:
                if next_role in {"container", "support_surface"}:
                    fragments.append(f"through {next_role} locations")
                else:
                    fragments.append(f"toward {next_role}")
            if next_stage:
                fragments.append(f"while staying near {next_stage}")
            if not fragments:
                continue
            items.append(
                SupportItem(
                    source="local_graph",
                    candidate_id=f"graph_transition:{idx}:{next_role}:{next_stage}",
                    candidate_type=CandidateType.WORKFLOW,
                    summary="Graph transition: similar subgraphs often continue " + " ".join(fragments) + ".",
                    score=max(vote, 0.1),
                    pattern_kind="graph_transition",
                    branch_tag="success_branch",
                    state_relevance=0.18,
                    task_relevance=0.16 if next_stage else 0.08,
                    goal_relevance=0.12 if next_role in {"container", "support_surface"} else 0.04,
                    dynamic={"location_role": next_role, "workflow_stage": next_stage},
                )
            )

        return {
            "items": sorted(items, key=lambda item: item.score, reverse=True),
            "anchor_state_ids": anchor_state_ids,
            "neighbor_state_ids": neighbor_state_ids,
            "linked_refs": linked_refs,
            "role_votes": dict(role_votes),
            "stage_votes": dict(stage_votes),
        }

    def _rank_rules(
        self, query: MemoryQuery, rules, source: str, graph_context: dict | None = None
    ) -> list[SupportItem]:
        ranked: list[SupportItem] = []
        for rule in rules:
            item = _rule_support_item(query, rule, source, graph_context)
            if item.score <= 0.0:
                continue
            if item.action_patterns:
                projected = sum(len(_project_rule_actions(query, pattern)) for pattern in item.action_patterns)
                if projected > 5:
                    continue
                if (
                    query.held_relevant_count <= 0
                    and query.progress_state not in {"carry_target", "finalize"}
                    and any("goal_destination" in pattern for pattern in item.action_patterns)
                    and item.pattern_kind != RuleType.BLOCKED.value
                ):
                    continue
            if (
                item.pattern_kind == RuleType.PRECONDITION.value
                and item.goal_relevance <= 0.02
                and item.task_relevance <= 0.22
            ):
                continue
            ranked.append(item)
        ranked.sort(key=lambda item: item.score, reverse=True)
        return ranked

    def _rank_artifacts(
        self, query: MemoryQuery, artifacts, source: str, graph_context: dict | None = None
    ) -> list[SupportItem]:
        ranked: list[SupportItem] = []
        for artifact in artifacts:
            item = _artifact_support_item(query, artifact, source, graph_context)
            if item.score <= 0.0:
                continue
            if item.pattern_kind == "closure" and query.remaining_relevant_count > 0:
                continue
            if item.dynamic.get("artifact_kind") == ArtifactKind.REFLECTION.value and not query.failure_label:
                diagnosis = str(item.dynamic.get("payload", {}).get("diagnosis", ""))
                counts = query.dynamic_context.get("search_attempt_counts", {}) if query.dynamic_context else {}
                max_attempts = (
                    max((int(value) for value in counts.values()), default=0) if isinstance(counts, dict) else 0
                )
                if diagnosis == "search_stall" and max_attempts < 2:
                    continue
                if diagnosis not in {"wrong_target_object", "premature_destination_focus", "search_stall"}:
                    item.score -= 0.08
            ranked.append(item)
        ranked.sort(key=lambda item: item.score, reverse=True)
        return ranked

    def _rank_local(
        self,
        query: MemoryQuery,
        local_memory: LocalGraphMemory,
    ) -> list[SupportItem]:
        if not local_memory.nodes_by_signature or not local_memory.edges_by_signature:
            return _candidate_rank(query, local_memory.candidates.values(), source="local")

        node_by_id = {node.node_id: node for node in local_memory.nodes_by_signature.values()}
        outgoing: dict[str, list] = defaultdict(list)
        for edge in local_memory.edges_by_signature.values():
            outgoing[edge.src].append(edge)

        anchors = []
        for node in node_by_id.values():
            if node.node_type != NodeType.STATE:
                continue
            state_relevance = _state_anchor_score(query, node.payload)
            task_relevance = _task_relevance_score(
                query,
                text=str(node.payload.get("workflow_stage") or ""),
                workflow_stage=str(node.payload.get("workflow_stage") or ""),
            )
            goal_relevance = _goal_relevance_from_state(query, node.payload)
            anchor_score = _combine_relevance(query, state_relevance, task_relevance, goal_relevance)
            if anchor_score > 0:
                anchors.append((node, state_relevance, task_relevance, goal_relevance, anchor_score))
        anchors.sort(key=lambda item: item[4], reverse=True)
        anchors = anchors[:3]

        ranked: list[SupportItem] = []
        object_role, destination_role = _goal_tokens(query)
        tool_role = _goal_tool_token(query)
        for state_node, anchor_state, anchor_task, anchor_goal, anchor_score in anchors:
            location = state_node.payload.get("location")
            visible = list(state_node.payload.get("visible_objects", []) or [])
            held = list(state_node.payload.get("held_objects", []) or [])
            visible_focus = [
                item
                for item in visible
                if any(token and token in str(item).lower() for token in (object_role, destination_role, tool_role))
            ]
            held_focus = [
                item
                for item in held
                if any(token and token in str(item).lower() for token in (object_role, destination_role, tool_role))
            ]
            fact_parts: list[str] = []
            location_tokens = _tokenize(str(location or ""))
            location_is_goal_aligned = any(
                token and token in location_tokens for token in (destination_role, tool_role)
            )
            if location and (anchor_goal >= 0.18 or visible_focus or held_focus or location_is_goal_aligned):
                fact_parts.append(f"similar states are near {location}")
            if visible_focus:
                fact_parts.append("relevant visible " + ", ".join(visible_focus[:4]))
            if held_focus:
                fact_parts.append("holding relevant " + ", ".join(held_focus[:2]))
            if fact_parts:
                fact_goal = anchor_goal + _goal_relevance_from_text(query, " ".join(fact_parts))
                fact_task = anchor_task
                fact_state = anchor_state + 0.12
                fact_score = _combine_relevance(query, fact_state, fact_task, fact_goal)
                if _is_multi_object_query(query) and (held or query.current_stage == "acquire_target"):
                    fact_score -= 0.08
                ranked.append(
                    SupportItem(
                        source="local_graph",
                        candidate_id=f"local_fact:{state_node.node_id}",
                        candidate_type=CandidateType.PRECONDITION,
                        summary="Local state fact: " + "; ".join(fact_parts) + ".",
                        score=fact_score,
                        task_family="",
                        pattern_kind="fact",
                        branch_tag="success_branch",
                        positive=max(state_node.stats.positive, 0),
                        negative=max(state_node.stats.negative, 0),
                        stalled=max(state_node.stats.stalled, 0),
                        state_relevance=fact_state,
                        task_relevance=fact_task,
                        goal_relevance=fact_goal,
                    )
                )

            for edge in outgoing.get(state_node.node_id, []):
                if edge.edge_type != EdgeType.TEMPORAL:
                    continue
                action_node = node_by_id.get(edge.dst)
                if action_node is None or action_node.node_type != NodeType.ACTION:
                    continue
                action_text = _payload_action_str(action_node.payload)
                if not action_text:
                    continue

                negative_rate = edge.stats.negative / edge.stats.support if edge.stats.support else 0.0
                action_state = anchor_state
                action_task = anchor_task + _task_relevance_score(query, text=action_text, pattern_kind="action")
                action_goal = 0.35 * anchor_goal + _goal_relevance_from_text(query, action_text)
                action_score = (
                    _combine_relevance(query, action_state, action_task, action_goal)
                    + _action_text_bonus(action_text, query.admissible_actions)
                    + 0.18 * edge.stats.confidence
                    - 0.22 * edge.stats.stall_rate
                    - 0.15 * negative_rate
                )
                if _is_multi_object_query(query):
                    goal_object = str(query.goal_roles.get("object", ""))
                    goal_destination = str(query.goal_roles.get("destination", ""))
                    if query.current_stage == "acquire_target":
                        action_score += (
                            0.1 if any(marker in action_text for marker in ("go(", "move(", "take(")) else 0.0
                        )
                    if held:
                        action_score -= (
                            0.06 if any(marker in action_text for marker in ("look", "examine", "open(")) else 0.0
                        )
                    if goal_object and action_text.startswith("take("):
                        action_score += 0.35 if goal_object in action_text else -0.08
                    if (
                        query.remaining_relevant_count > 0
                        and query.held_relevant_count > 0
                        and action_text.startswith("move(")
                    ):
                        if goal_destination and f"destination={goal_destination}" in action_text:
                            action_score += 0.2
                        else:
                            action_score -= 1.0
                    if (
                        query.remaining_relevant_count == 0
                        and query.held_relevant_count > 0
                        and action_text.startswith("move(")
                    ):
                        if goal_destination and f"destination={goal_destination}" in action_text:
                            action_score += 0.65
                        else:
                            action_score -= 0.45
                ranked.append(
                    SupportItem(
                        source="local_graph",
                        candidate_id=f"local_action:{state_node.node_id}:{action_text}",
                        candidate_type=CandidateType.PRECONDITION,
                        summary=f"In similar states, try {action_text}.",
                        score=action_score,
                        task_family="",
                        pattern_kind="action",
                        branch_tag=_local_branch_tag(
                            CandidateType.PRECONDITION,
                            pattern_kind="action",
                            positive=max(edge.stats.positive, 0),
                            negative=max(edge.stats.negative, 0),
                            stalled=max(edge.stats.stalled, 0),
                        ),
                        positive=max(edge.stats.positive, 0),
                        negative=max(edge.stats.negative, 0),
                        stalled=max(edge.stats.stalled, 0),
                        state_relevance=action_state,
                        task_relevance=action_task,
                        goal_relevance=action_goal,
                    )
                )

                for action_edge in outgoing.get(action_node.node_id, []):
                    dst_node = node_by_id.get(action_edge.dst)
                    if action_edge.edge_type == EdgeType.FAILS_UNDER and dst_node is not None:
                        failure_text = f"{action_text} fails under {dst_node.signature}."
                        failure_state = anchor_state
                        failure_task = anchor_task + _task_relevance_score(
                            query,
                            text=failure_text,
                            pattern_kind="failure",
                        )
                        failure_goal = 0.25 * anchor_goal + _goal_relevance_from_text(query, failure_text)
                        ranked.append(
                            SupportItem(
                                source="local_graph",
                                candidate_id=f"local_failure:{action_node.node_id}:{dst_node.signature}",
                                candidate_type=CandidateType.FAILURE,
                                summary=failure_text,
                                score=_combine_relevance(query, failure_state, failure_task, failure_goal)
                                + 0.16
                                + 0.15 * action_edge.stats.confidence
                                + 0.1 * min(action_edge.stats.support / 2.0, 1.0),
                                task_family="",
                                pattern_kind="anti_pattern" if "stall" in dst_node.signature else "failure",
                                branch_tag="failure_branch",
                                positive=max(action_edge.stats.positive, 0),
                                negative=max(action_edge.stats.negative, 0),
                                stalled=max(action_edge.stats.stalled, 0),
                                state_relevance=failure_state,
                                task_relevance=failure_task,
                                goal_relevance=failure_goal,
                            )
                        )
                    elif action_edge.edge_type == EdgeType.ADVANCES_TO and dst_node is not None:
                        current = query.current_stage or "current"
                        stage_penalty = (
                            0.18 if query.current_stage and dst_node.signature == query.current_stage else 0.0
                        )
                        workflow_text = f"Workflow pattern: {current} -> {dst_node.signature}."
                        workflow_state = 0.75 * anchor_state
                        workflow_task = anchor_task + _task_relevance_score(
                            query,
                            text=workflow_text,
                            pattern_kind="workflow",
                            workflow_stage=dst_node.signature,
                        )
                        workflow_goal = 0.2 * anchor_goal + _goal_relevance_from_text(query, workflow_text)
                        workflow_score = _combine_relevance(query, workflow_state, workflow_task, workflow_goal)
                        workflow_score += (
                            0.2
                            + 0.12 * action_edge.stats.confidence
                            - stage_penalty
                            - 0.2 * action_edge.stats.stall_rate
                        )
                        if _is_multi_object_query(query):
                            if query.current_stage == "acquire_target":
                                workflow_score += 0.18
                            if held:
                                workflow_score += 0.12
                        ranked.append(
                            SupportItem(
                                source="local_graph",
                                candidate_id=f"local_workflow:{action_node.node_id}:{dst_node.signature}",
                                candidate_type=CandidateType.WORKFLOW,
                                summary=workflow_text,
                                score=workflow_score,
                                task_family="",
                                pattern_kind="workflow",
                                branch_tag=_local_branch_tag(
                                    CandidateType.WORKFLOW,
                                    pattern_kind="workflow",
                                    positive=max(action_edge.stats.positive, 0),
                                    negative=max(action_edge.stats.negative, 0),
                                    stalled=max(action_edge.stats.stalled, 0),
                                ),
                                positive=max(action_edge.stats.positive, 0),
                                negative=max(action_edge.stats.negative, 0),
                                stalled=max(action_edge.stats.stalled, 0),
                                state_relevance=workflow_state,
                                task_relevance=workflow_task,
                                goal_relevance=workflow_goal,
                            )
                        )

        supplemental = [
            candidate
            for candidate in local_memory.candidates.values()
            if candidate.candidate_type in {CandidateType.REPAIR, CandidateType.FAILURE, CandidateType.WORKFLOW}
        ]
        ranked.extend(_candidate_rank(query, supplemental, source="local_pattern"))
        return _merge_ranked_items(ranked, self.local_top_k + 6)

    def _rank_global(self, query: MemoryQuery, global_memory: GlobalGraphMemory) -> list[SupportItem]:
        qtokens = _query_tokens(query)
        task_family = _query_task_family(query)
        visible = _keyword_suffixes(query, "visible+=")
        holding = _keyword_suffixes(query, "holding+=")
        verbs = _canonical_verbs(query.admissible_actions)
        ranked: list[SupportItem] = []
        for candidate in global_memory.candidates.values():
            lexical = _jaccard(qtokens, _candidate_tokens(candidate))
            score = (
                _graph_evidence_bonus(candidate)
                + _action_bonus(candidate, query.admissible_actions)
                - _candidate_penalty(candidate)
            )
            candidate_family = str(candidate.structure.get("task_family", ""))
            pattern_kind = str(candidate.structure.get("pattern_kind", ""))
            candidate_text = candidate.summary + " " + " ".join(str(value) for value in candidate.structure.values())
            state_relevance = 0.35 * lexical
            task_relevance = _task_relevance_score(
                query,
                text=candidate_text,
                task_family=candidate_family,
                pattern_kind=pattern_kind,
            )
            goal_relevance = _goal_relevance_from_text(query, candidate_text)

            if candidate_family:
                if candidate_family == task_family:
                    task_relevance += 0.18
                else:
                    task_relevance -= 0.08

            goal_relevance += _goal_role_bonus(
                query,
                candidate_text,
                task_family=candidate_family,
                pattern_kind=pattern_kind,
            )
            if pattern_kind == "closure" and query.remaining_relevant_count > 0:
                score -= 1.0

            if candidate.candidate_type == CandidateType.WORKFLOW:
                if pattern_kind == "closure":
                    if "use" in verbs or holding or visible:
                        task_relevance += 0.2
                    elif query.current_stage == "inspect_container":
                        task_relevance += 0.1
                    else:
                        task_relevance -= 0.05
                elif query.failure_label:
                    task_relevance -= 0.08
                elif not query.current_stage:
                    task_relevance += 0.08
                else:
                    workflow = tuple(candidate.structure.get("workflow", ()) or ())
                    if query.current_stage in workflow or query.current_stage in candidate.summary:
                        task_relevance += 0.16
                    elif holding or ("use" in verbs):
                        task_relevance -= 0.18
            elif candidate.candidate_type == CandidateType.PRECONDITION:
                if _action_bonus(candidate, query.admissible_actions) > 0:
                    score += 0.18
                if query.location and query.location in candidate.summary:
                    state_relevance += 0.12
                if holding and "holding" in candidate.summary.lower():
                    state_relevance += 0.18
                if visible and any(marker in candidate.summary.lower() for marker in ("drawer", "shelf", "container")):
                    state_relevance -= 0.2
            elif candidate.candidate_type == CandidateType.FAILURE:
                if pattern_kind == "anti_pattern":
                    if query.current_stage == "inspect_container" or "look" in verbs:
                        task_relevance += 0.1
                    else:
                        task_relevance -= 0.06
                if query.failure_label and candidate.structure.get("failure_label") == query.failure_label:
                    task_relevance += 0.2
                elif not query.failure_label:
                    task_relevance -= 0.1
            elif candidate.candidate_type == CandidateType.REPAIR:
                if query.failure_label and candidate.structure.get("failure_label") == query.failure_label:
                    task_relevance += 0.24
                else:
                    task_relevance -= 0.08

            score += _combine_relevance(query, state_relevance, task_relevance, goal_relevance)

            ranked.append(
                SupportItem(
                    source="global_motif",
                    candidate_id=candidate.candidate_id,
                    candidate_type=candidate.candidate_type,
                    summary=candidate.summary,
                    score=score,
                    task_family=candidate_family,
                    pattern_kind=pattern_kind,
                    branch_tag=_branch_tag(
                        candidate.candidate_type,
                        pattern_kind=pattern_kind,
                        positive=candidate.positive,
                        negative=candidate.negative,
                        stalled=candidate.stalled,
                    ),
                    positive=candidate.positive,
                    negative=candidate.negative,
                    stalled=candidate.stalled,
                    state_relevance=state_relevance,
                    task_relevance=task_relevance,
                    goal_relevance=goal_relevance,
                )
            )
        ranked.sort(key=lambda item: item.score, reverse=True)
        return ranked


class QueryBasedRetriever:
    def __init__(self, top_k: int = 5, router: HeuristicSupportRouter | StudentSupportRouter | None = None) -> None:
        self.top_k = top_k
        self.router = router

    def retrieve(
        self,
        query: MemoryQuery,
        local_memory: LocalGraphMemory,
        global_memory: GlobalGraphMemory,
    ) -> SupportBundle:
        if str(query.task_family or "").startswith("fever") and (
            local_memory.artifacts_by_id or global_memory.artifacts_by_id
        ):
            return LightweightRetriever(local_top_k=self.top_k, global_top_k=self.top_k, router=self.router).retrieve(
                query,
                local_memory,
                global_memory,
            )
        if local_memory.rules_by_id or global_memory.rules_by_id:
            return LightweightRetriever(local_top_k=self.top_k, global_top_k=self.top_k, router=self.router).retrieve(
                query,
                local_memory,
                global_memory,
            )
        bundle = SupportBundle(
            query=f"query:{query.goal}",
            goal_object=str(query.goal_roles.get("object", "")),
            goal_destination=str(query.goal_roles.get("destination", "")),
            goal_tool=str(query.goal_roles.get("tool", "")),
            progress_state=str(query.progress_state or ""),
        )
        local_items = self._typed_local_rank(query, local_memory)
        global_items = self._typed_global_rank(query, global_memory)
        bundle.local_items = _joint_select(query, local_items, self.top_k)
        bundle.global_items = _joint_select(query, global_items, self.top_k)
        route_bundle(query, bundle, router=self.router)
        return bundle

    def _typed_local_rank(self, query: MemoryQuery, local_memory: LocalGraphMemory) -> list[SupportItem]:
        base = LightweightRetriever(local_top_k=self.top_k, global_top_k=self.top_k)._rank_local(query, local_memory)
        if not query.desired_types:
            return base
        reranked = []
        for item in base:
            score = item.score + (0.35 if item.candidate_type in query.desired_types else -0.1)
            reranked.append(
                SupportItem(
                    item.source,
                    item.candidate_id,
                    item.candidate_type,
                    item.summary,
                    score,
                    task_family=item.task_family,
                    pattern_kind=item.pattern_kind,
                    branch_tag=item.branch_tag,
                    positive=item.positive,
                    negative=item.negative,
                    stalled=item.stalled,
                    state_relevance=item.state_relevance,
                    task_relevance=item.task_relevance,
                    goal_relevance=item.goal_relevance,
                )
            )
        reranked.sort(key=lambda item: item.score, reverse=True)
        return reranked

    def _typed_global_rank(self, query: MemoryQuery, global_memory: GlobalGraphMemory) -> list[SupportItem]:
        base = LightweightRetriever(local_top_k=self.top_k, global_top_k=self.top_k)._rank_global(query, global_memory)
        if not query.desired_types:
            return base
        reranked = []
        for item in base:
            score = item.score + (0.35 if item.candidate_type in query.desired_types else -0.1)
            reranked.append(
                SupportItem(
                    item.source,
                    item.candidate_id,
                    item.candidate_type,
                    item.summary,
                    score,
                    task_family=item.task_family,
                    pattern_kind=item.pattern_kind,
                    branch_tag=item.branch_tag,
                    positive=item.positive,
                    negative=item.negative,
                    stalled=item.stalled,
                    state_relevance=item.state_relevance,
                    task_relevance=item.task_relevance,
                    goal_relevance=item.goal_relevance,
                )
            )
        reranked.sort(key=lambda item: item.score, reverse=True)
        return reranked
