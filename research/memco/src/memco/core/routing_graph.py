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

import json
import math
import re
from dataclasses import dataclass

from .graph_types import CandidateType, CanonicalAction, MemoryQuery, SupportBundle, SupportItem

SLOTS = ("fact", "workflow", "precondition", "repair", "closure", "blocked")


def _query_family(query: MemoryQuery) -> str:
    goal = query.goal.lower()
    if "under the" in goal or "with the" in goal:
        return "look_at_obj_in_light"
    return re.sub(r"[^a-z0-9_]+", "_", goal).strip("_") or "unknown_task"


def _is_multi_object_query(query: MemoryQuery) -> bool:
    if query.required_count > 1:
        return True
    if query.progress_hint.startswith("multi_object"):
        return True
    family = _query_family(query)
    goal = query.goal.lower()
    return "two_obj" in family or "two object" in goal or "two objects" in goal


def _keyword_suffixes(query: MemoryQuery, prefix: str) -> set[str]:
    values: set[str] = set()
    for keyword in query.keywords:
        if keyword.startswith(prefix):
            values.add(keyword.split("=", 1)[-1].split("+=", 1)[-1])
    return values


def _canonical_verbs(actions: tuple[CanonicalAction, ...]) -> set[str]:
    return {action.verb for action in actions}


def _canonical_strings(actions: tuple[CanonicalAction, ...]) -> set[str]:
    return {action.canonical_str for action in actions}


def _action_text_bonus(action_text: str, actions: tuple[CanonicalAction, ...]) -> float:
    if not action_text or not actions:
        return 0.0
    action_text = action_text.lower()
    action_strings = _canonical_strings(actions)
    action_verbs = _canonical_verbs(actions)
    bonus = 0.0
    if action_text in action_strings:
        bonus += 1.0
    if any(action_text.startswith(f"{verb}(") or action_text == verb for verb in action_verbs):
        bonus += 0.45
    return bonus


def _summary_action_alignment(summary: str, actions: tuple[CanonicalAction, ...]) -> float:
    candidates = re.findall(r"[a-z_]+\([a-z0-9_,= ]+\)|[a-z_]+", summary.lower())
    if not candidates:
        return 0.0
    return max((_action_text_bonus(candidate.strip(), actions) for candidate in candidates), default=0.0)


def _summary_grounding(summary: str, query: MemoryQuery) -> float:
    lowered = summary.lower()
    grounding = 0.0
    if query.location and query.location.lower() in lowered:
        grounding += 0.45
    visible = _keyword_suffixes(query, "visible+=")
    holding = _keyword_suffixes(query, "holding+=")
    for token in visible:
        if token.lower() in lowered:
            grounding += 0.18
    for token in holding:
        if token.lower() in lowered:
            grounding += 0.24
    if query.current_stage and query.current_stage.lower() in lowered:
        grounding += 0.16
    goal_tokens = set(re.findall(r"[a-z0-9_]+", query.goal.lower()))
    summary_tokens = set(re.findall(r"[a-z0-9_]+", lowered))
    if goal_tokens and summary_tokens:
        grounding += 0.12 * (len(goal_tokens & summary_tokens) / len(goal_tokens | summary_tokens))
    return grounding


def _evidence_quality(item: SupportItem) -> float:
    support = item.positive + item.negative
    if support <= 0:
        return 0.0
    confidence = item.positive / support
    negative_rate = item.negative / support
    stall_rate = item.stalled / support
    return 0.45 * confidence - 0.28 * negative_rate - 0.35 * stall_rate + 0.08 * min(support / 4.0, 1.0)


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


def _family_alignment(query: MemoryQuery, item: SupportItem) -> float:
    if not item.task_family:
        return 0.0
    if item.task_family == _query_family(query):
        return 0.18
    query_tokens = set(re.findall(r"[a-z0-9_]+", _query_family(query)))
    item_tokens = set(re.findall(r"[a-z0-9_]+", item.task_family))
    if not query_tokens or not item_tokens:
        return -0.04
    overlap = len(query_tokens & item_tokens) / len(query_tokens | item_tokens)
    return 0.08 * overlap - 0.06 * (1.0 - overlap)


def _entry_state_match(query: MemoryQuery, item: SupportItem) -> float:
    summary = item.summary.lower()
    score = _summary_grounding(item.summary, query)
    verbs = _canonical_verbs(query.admissible_actions)
    visible = _keyword_suffixes(query, "visible+=")
    holding = _keyword_suffixes(query, "holding+=")
    if visible and any(token.lower() in summary for token in visible):
        score += 0.16
    if holding and any(token.lower() in summary for token in holding):
        score += 0.2
    if "use" in verbs and "use" in summary:
        score += 0.14
    return score


def _search_markers(summary: str) -> int:
    lowered = summary.lower()
    return sum(marker in lowered for marker in ("locate_target", "inspect_container", "drawer", "shelf", "open("))


def _risk_score(query: MemoryQuery, slot: str, item: SupportItem) -> float:
    risk = 0.0
    lowered = item.summary.lower()
    if item.pattern_kind == "anti_pattern":
        risk += 0.04
    if item.branch_tag == "failure_branch":
        risk += 0.03 if item.source.startswith("local") else 0.08
    support = item.positive + item.negative
    if support > 0:
        evidence_scale = 0.45 if item.source.startswith("local") else 0.7
        risk += evidence_scale * 0.14 * (item.negative / support)
        risk += evidence_scale * 0.18 * (item.stalled / support)
    if _late_stage(query):
        risk += 0.08 * _search_markers(lowered)
    if slot == "workflow" and item.pattern_kind == "closure" and not _late_stage(query):
        risk += 0.08
    if slot == "closure" and item.pattern_kind not in {"closure", "workflow"}:
        risk += 0.08
    return risk


def _pattern_stage_fit(query: MemoryQuery, slot: str, item: SupportItem) -> float:
    score = 0.0
    if slot == "fact" and item.pattern_kind == "fact":
        score += 0.35
    if slot == "workflow":
        if item.pattern_kind == "workflow":
            score += 0.14
        if item.pattern_kind == "closure":
            score += 0.08 if _late_stage(query) else -0.04
    if slot == "closure":
        if item.pattern_kind == "closure":
            score += 0.18 if _late_stage(query) else -0.1
        elif item.pattern_kind == "workflow":
            score += 0.04 if _late_stage(query) else -0.06
    if slot == "repair":
        if item.candidate_type == CandidateType.REPAIR:
            score += 0.16
        if item.pattern_kind == "anti_pattern":
            score -= 0.04
    if slot == "blocked" and item.pattern_kind == "anti_pattern":
        score += 0.14
    if slot == "precondition" and item.candidate_type == CandidateType.PRECONDITION:
        score += 0.1
    return score


def _success_value(query: MemoryQuery, slot: str, item: SupportItem) -> float:
    action_align = _summary_action_alignment(item.summary, query.admissible_actions)
    value = (
        0.62 * item.score
        + 0.22 * action_align
        + 0.18 * _entry_state_match(query, item)
        + 0.12 * _family_alignment(query, item)
        + 0.12 * _evidence_quality(item)
        + _pattern_stage_fit(query, slot, item)
    )
    if item.branch_tag == "success_branch":
        value += 0.03
    elif item.branch_tag == "failure_branch":
        value -= 0.01 if item.source.startswith("local") else 0.03
    elif item.branch_tag == "repair_branch":
        if item.source.startswith("local"):
            value += 0.0 if slot != "repair" else 0.03
        else:
            value -= 0.02 if slot != "repair" else 0.03
    return value


def _failure_risk(query: MemoryQuery, slot: str, item: SupportItem) -> float:
    risk = _risk_score(query, slot, item)
    if item.branch_tag == "failure_branch":
        risk += 0.02 if item.source.startswith("local") else 0.08
    if item.branch_tag == "repair_branch" and slot != "repair":
        risk += 0.01 if item.source.startswith("local") else 0.04
    return risk


def _repair_value(query: MemoryQuery, slot: str, item: SupportItem) -> float:
    if item.branch_tag != "repair_branch":
        return 0.0
    if not _has_failure(query):
        return -0.01 if item.source.startswith("local") else -0.04
    base = (
        0.1
        + 0.14 * _entry_state_match(query, item)
        + 0.12 * _summary_action_alignment(item.summary, query.admissible_actions)
    )
    if slot == "repair":
        base += 0.06
    return base


def _item_utility(query: MemoryQuery, slot: str, item: SupportItem) -> float:
    utility = _success_value(query, slot, item) - _failure_risk(query, slot, item) + _repair_value(query, slot, item)
    if slot == "repair" and not _has_failure(query):
        utility -= 0.45
    return utility


def _item_compatibility(base: SupportItem, other: SupportItem) -> float:
    bonus = 0.0
    if base.task_family and other.task_family:
        if base.task_family == other.task_family:
            bonus += 0.22
        else:
            bonus -= 0.35
    if base.pattern_kind and other.pattern_kind and base.pattern_kind == other.pattern_kind:
        bonus += 0.08
    if base.source != other.source:
        bonus += 0.04
    return bonus


def _late_stage(query: MemoryQuery) -> bool:
    if query.progress_state == "finalize":
        return True
    if _multi_object_mid_phase(query):
        return False
    holding = bool(_keyword_suffixes(query, "holding+="))
    visible = bool(_keyword_suffixes(query, "visible+="))
    verbs = _canonical_verbs(query.admissible_actions)
    return holding or ("use" in verbs and visible)


def _multi_object_mid_phase(query: MemoryQuery) -> bool:
    if query.required_count > 1:
        return query.remaining_relevant_count > 0 and query.held_relevant_count > 0
    return query.progress_hint == "multi_object_mid" or (
        _is_multi_object_query(query)
        and (bool(_keyword_suffixes(query, "holding+=")) or query.current_stage == "acquire_target")
    )


def _has_failure(query: MemoryQuery) -> bool:
    return bool(query.failure_label)


def _search_penalty(summary: str, query: MemoryQuery) -> float:
    lowered = summary.lower()
    if not _late_stage(query):
        return 0.0
    penalty = 0.0
    for marker in ("locate_target", "inspect_container", "drawer", "shelf", "open("):
        if marker in lowered:
            penalty += 0.28
    return penalty


def _stale_penalty(summary: str, query: MemoryQuery) -> float:
    lowered = summary.lower()
    penalty = _search_penalty(summary, query)
    if not _has_failure(query) and lowered.startswith("after "):
        penalty += 0.35
    return penalty


def _hard_gate_item(query: MemoryQuery, slot: str, item: SupportItem) -> bool:
    if slot != "repair" and item.branch_tag == "repair_branch":
        return False
    if slot == "repair" and not _has_failure(query):
        return False
    if slot == "closure":
        if not _late_stage(query):
            return False
        if item.pattern_kind not in {"closure", "workflow"}:
            return False

    return True


def _suppress_local_action_precondition(query: MemoryQuery, item: SupportItem) -> bool:
    if item.source != "local_graph":
        return False
    if not item.candidate_id.startswith("local_action:"):
        return False
    lowered = item.summary.lower()
    if not lowered.startswith("in similar states, try "):
        return False
    if _late_stage(query):
        return False
    # Early-stage search is where these weak local action hints most often
    # derail the agent into generic container/support scans.
    noisy_markers = ("go(", "open(", "examine(", "look", "inventory")
    return any(marker in lowered for marker in noisy_markers)


def _slot_candidates(
    bundle: SupportBundle, slot: str, source: str, query: MemoryQuery | None = None
) -> list[SupportItem]:
    items = bundle.local_items if source == "local" else bundle.global_items
    if slot == "fact":
        if source == "local":
            selected = [
                item for item in items if item.source == "local_graph" and item.candidate_id.startswith("local_fact:")
            ]
        else:
            selected = []
    elif slot == "workflow":
        selected = [item for item in items if item.candidate_type == CandidateType.WORKFLOW]
    elif slot == "precondition":
        selected = [
            item
            for item in items
            if item.candidate_type == CandidateType.PRECONDITION and not item.candidate_id.startswith("local_fact:")
        ]
    elif slot == "repair":
        selected = [item for item in items if item.candidate_type == CandidateType.REPAIR]
    elif slot == "closure":
        selected = [
            item
            for item in items
            if item.candidate_type in {CandidateType.PRECONDITION, CandidateType.WORKFLOW}
            and not any(
                marker in item.summary.lower()
                for marker in ("locate_target", "inspect_container", "drawer", "shelf", "open(")
            )
        ]
    elif slot == "blocked":
        selected = [item for item in items if item.candidate_type == CandidateType.FAILURE]
    else:
        selected = []

    if query is not None:
        selected = [item for item in selected if _hard_gate_item(query, slot, item)]
        if slot == "precondition":
            selected = [item for item in selected if not _suppress_local_action_precondition(query, item)]
    return selected


def _slot_source_score(query: MemoryQuery, slot: str, source: str, items: list[SupportItem]) -> float:
    if not items:
        return -1e9
    top = items[:2]
    base = sum(item.score for item in top) / len(top)
    action = max((_summary_action_alignment(item.summary, query.admissible_actions) for item in top), default=0.0)
    grounding = sum(_summary_grounding(item.summary, query) for item in top) / len(top)
    stale = sum(_stale_penalty(item.summary, query) for item in top) / len(top)
    score = 0.72 * base + 0.16 * action + 0.12 * grounding - 0.12 * stale
    if source == "global":
        score += 0.02 * sum(_family_alignment(query, item) for item in top) / len(top)
    return score


def extract_router_features(query: MemoryQuery, bundle: SupportBundle) -> dict[str, float]:
    features: dict[str, float] = {
        "has_failure": float(_has_failure(query)),
        "late_stage": float(_late_stage(query)),
        "has_holding": float(bool(_keyword_suffixes(query, "holding+="))),
        "multi_object_mid": float(_multi_object_mid_phase(query)),
        "has_visible": float(bool(_keyword_suffixes(query, "visible+="))),
        "admissible_count": float(len(query.admissible_actions)),
    }
    for slot in SLOTS:
        for source in ("local", "global"):
            items = _slot_candidates(bundle, slot, source, query)
            key_prefix = f"{source}_{slot}"
            features[f"{key_prefix}_count"] = float(len(items))
            features[f"{key_prefix}_best"] = items[0].score if items else 0.0
            features[f"{key_prefix}_success_count"] = float(
                sum(1 for item in items if item.branch_tag == "success_branch")
            )
            features[f"{key_prefix}_failure_count"] = float(
                sum(1 for item in items if item.branch_tag == "failure_branch")
            )
            features[f"{key_prefix}_repair_count"] = float(
                sum(1 for item in items if item.branch_tag == "repair_branch")
            )
            features[f"{key_prefix}_action"] = max(
                (_summary_action_alignment(item.summary, query.admissible_actions) for item in items[:2]), default=0.0
            )
            features[f"{key_prefix}_grounding"] = (
                sum(_summary_grounding(item.summary, query) for item in items[:2]) / max(len(items[:2]), 1)
                if items
                else 0.0
            )
            features[f"{key_prefix}_stale"] = (
                sum(_stale_penalty(item.summary, query) for item in items[:2]) / max(len(items[:2]), 1)
                if items
                else 0.0
            )
    return features


@dataclass(slots=True)
class RouteSelection:
    source_by_slot: dict[str, str]
    weights_by_slot: dict[str, dict[str, float]]


def _softmax_scores(local_score: float, global_score: float, none_score: float) -> dict[str, float]:
    scores = {
        "local": local_score,
        "global": global_score,
        "none": none_score,
    }
    finite_scores = {key: value for key, value in scores.items() if value > -1e8}
    if not finite_scores:
        return {"local": 0.0, "global": 0.0, "none": 1.0}
    max_score = max(finite_scores.values())
    exp_scores = {key: math.exp(value - max_score) for key, value in finite_scores.items()}
    total = sum(exp_scores.values())
    weights = {key: exp_scores.get(key, 0.0) / total for key in ("local", "global", "none")}
    return weights


def _weights_to_label(weights: dict[str, float]) -> str:
    local_weight = weights.get("local", 0.0)
    global_weight = weights.get("global", 0.0)
    none_weight = weights.get("none", 0.0)
    if none_weight >= max(local_weight, global_weight):
        return "none"
    if local_weight >= 0.25 and global_weight >= 0.25:
        return "mixed"
    return "local" if local_weight >= global_weight else "global"


class HeuristicSupportRouter:
    def route(self, query: MemoryQuery, bundle: SupportBundle) -> RouteSelection:
        source_by_slot: dict[str, str] = {}
        weights_by_slot: dict[str, dict[str, float]] = {}
        for slot in SLOTS:
            if slot == "repair" and not _has_failure(query):
                source_by_slot[slot] = "none"
                weights_by_slot[slot] = {"local": 0.0, "global": 0.0, "none": 1.0}
                continue
            if slot == "closure" and not _late_stage(query):
                source_by_slot[slot] = "none"
                weights_by_slot[slot] = {"local": 0.0, "global": 0.0, "none": 1.0}
                continue
            local_items = _slot_candidates(bundle, slot, "local", query)
            global_items = _slot_candidates(bundle, slot, "global", query)
            local_score = _slot_source_score(query, slot, "local", local_items)
            global_score = _slot_source_score(query, slot, "global", global_items)
            if local_items and global_items and slot in {"workflow", "precondition", "fact"}:
                global_score -= 0.04
            best = max(local_score, global_score)
            none_score = 0.16 - max(best, 0.0)
            if slot == "fact" and global_items:
                global_score = max(global_score - 0.6, -1e9)
            weights = _softmax_scores(local_score, global_score, none_score)
            weights_by_slot[slot] = weights
            source_by_slot[slot] = _weights_to_label(weights)
        return RouteSelection(source_by_slot=source_by_slot, weights_by_slot=weights_by_slot)


class StudentSupportRouter:
    def __init__(self, examples: list[dict], feature_names: list[str], k: int = 15) -> None:
        self.examples = examples
        self.feature_names = feature_names
        self.k = k

    @classmethod
    def load(cls, path: str) -> "StudentSupportRouter":
        payload = json.load(open(path, "r", encoding="utf-8"))
        return cls(examples=payload["examples"], feature_names=payload["feature_names"], k=int(payload.get("k", 15)))

    def route(self, query: MemoryQuery, bundle: SupportBundle) -> RouteSelection:
        feature_dict = extract_router_features(query, bundle)
        neighbors: list[tuple[float, dict]] = []
        for example in self.examples:
            dist = 0.0
            example_features = example.get("features", {})
            for name in self.feature_names:
                diff = feature_dict.get(name, 0.0) - float(example_features.get(name, 0.0))
                dist += diff * diff
            neighbors.append((dist, example))
        neighbors.sort(key=lambda item: item[0])
        selected = neighbors[: max(1, self.k)]

        source_by_slot: dict[str, str] = {}
        weights_by_slot: dict[str, dict[str, float]] = {}
        for slot in SLOTS:
            accum = {"local": 0.0, "global": 0.0, "none": 0.0}
            total = 0.0
            for dist, example in selected:
                weight = 1.0 / (dist + 1e-6)
                slot_weights = (example.get("weights") or {}).get(slot, {})
                accum["local"] += weight * float(slot_weights.get("local", 0.0))
                accum["global"] += weight * float(slot_weights.get("global", 0.0))
                accum["none"] += weight * float(slot_weights.get("none", 0.0))
                total += weight
            if total > 0:
                accum = {key: value / total for key, value in accum.items()}
            else:
                accum = {"local": 0.0, "global": 0.0, "none": 1.0}
            weights_by_slot[slot] = accum
            source_by_slot[slot] = _weights_to_label(accum)
        return RouteSelection(source_by_slot=source_by_slot, weights_by_slot=weights_by_slot)


def _select_mixed_items(
    query: MemoryQuery,
    slot: str,
    local_items: list[SupportItem],
    global_items: list[SupportItem],
    weights: dict[str, float],
    *,
    limit: int = 2,
) -> list[SupportItem]:
    scored: list[tuple[float, SupportItem]] = []
    local_weight = weights.get("local", 0.0)
    global_weight = weights.get("global", 0.0)
    if local_weight >= 0.12:
        for item in local_items[:2]:
            scored.append((0.55 * local_weight + _item_utility(query, slot, item), item))
    if global_weight >= 0.12:
        for item in global_items[:2]:
            scored.append((0.55 * global_weight + _item_utility(query, slot, item), item))
    if slot == "repair":
        scored = [
            (score, item)
            for score, item in scored
            if item.branch_tag == "repair_branch" or item.candidate_type == CandidateType.REPAIR
        ]
    if slot == "closure":
        scored = [(score, item) for score, item in scored if item.pattern_kind in {"closure", "workflow"}]
    scored.sort(key=lambda pair: pair[0], reverse=True)
    selected: list[SupportItem] = []
    seen: set[str] = set()
    seen_summaries: set[str] = set()
    while scored and len(selected) < limit:
        current_score, item = scored.pop(0)
        if item.candidate_id in seen or item.summary in seen_summaries:
            continue
        seen.add(item.candidate_id)
        seen_summaries.add(item.summary)
        selected.append(item)
        rescored: list[tuple[float, SupportItem]] = []
        for score, candidate in scored:
            if candidate.candidate_id in seen or candidate.summary in seen_summaries:
                continue
            rescored.append((score + _item_compatibility(item, candidate), candidate))
        rescored.sort(key=lambda pair: pair[0], reverse=True)
        scored = rescored
    return selected


def route_bundle(
    query: MemoryQuery, bundle: SupportBundle, router: HeuristicSupportRouter | StudentSupportRouter | None = None
) -> None:
    router = router or HeuristicSupportRouter()
    selection = router.route(query, bundle)

    if not _has_failure(query):
        selection.source_by_slot["repair"] = "none"
        selection.weights_by_slot["repair"] = {"local": 0.0, "global": 0.0, "none": 1.0}
    if not _late_stage(query):
        selection.source_by_slot["closure"] = "none"
        selection.weights_by_slot["closure"] = {"local": 0.0, "global": 0.0, "none": 1.0}

    local_fact = _slot_candidates(bundle, "fact", "local", query)
    global_fact = _slot_candidates(bundle, "fact", "global", query)
    local_workflow = _slot_candidates(bundle, "workflow", "local", query)
    global_workflow = _slot_candidates(bundle, "workflow", "global", query)
    local_precondition = _slot_candidates(bundle, "precondition", "local", query)
    global_precondition = _slot_candidates(bundle, "precondition", "global", query)
    local_repair = _slot_candidates(bundle, "repair", "local", query)
    global_repair = _slot_candidates(bundle, "repair", "global", query)
    local_closure = _slot_candidates(bundle, "closure", "local", query)
    global_closure = _slot_candidates(bundle, "closure", "global", query)
    local_blocked = _slot_candidates(bundle, "blocked", "local", query)
    global_blocked = _slot_candidates(bundle, "blocked", "global", query)

    bundle.fact_items = _select_mixed_items(
        query, "fact", local_fact, global_fact, selection.weights_by_slot.get("fact", {})
    )
    bundle.workflow_items = _select_mixed_items(
        query, "workflow", local_workflow, global_workflow, selection.weights_by_slot.get("workflow", {})
    )
    used_ids = {item.candidate_id for item in bundle.fact_items + bundle.workflow_items}
    bundle.precondition_items = [
        item
        for item in _select_mixed_items(
            query,
            "precondition",
            local_precondition,
            global_precondition,
            selection.weights_by_slot.get("precondition", {}),
        )
        if item.candidate_id not in used_ids
    ]
    used_ids |= {item.candidate_id for item in bundle.precondition_items}
    bundle.repair_items = [
        item
        for item in _select_mixed_items(
            query, "repair", local_repair, global_repair, selection.weights_by_slot.get("repair", {})
        )
        if item.candidate_id not in used_ids
    ]
    used_ids |= {item.candidate_id for item in bundle.repair_items}
    bundle.closure_items = [
        item
        for item in _select_mixed_items(
            query, "closure", local_closure, global_closure, selection.weights_by_slot.get("closure", {})
        )
        if item.candidate_id not in used_ids
    ]
    blocked_items = _select_mixed_items(
        query, "blocked", local_blocked, global_blocked, selection.weights_by_slot.get("blocked", {})
    )
    bundle.routing_weights = selection.weights_by_slot
    bundle.routing_decisions = {
        "fact": selection.source_by_slot["fact"] if bundle.fact_items else "none",
        "workflow": selection.source_by_slot["workflow"] if bundle.workflow_items else "none",
        "precondition": selection.source_by_slot["precondition"] if bundle.precondition_items else "none",
        "repair": selection.source_by_slot["repair"] if bundle.repair_items else "none",
        "closure": selection.source_by_slot["closure"] if bundle.closure_items else "none",
        "blocked": selection.source_by_slot["blocked"] if blocked_items else "none",
    }

    blocked: list[str] = []
    suggested: list[str] = []
    workflows: list[str] = []
    routed_items = (
        bundle.fact_items
        + bundle.workflow_items
        + bundle.precondition_items
        + bundle.repair_items
        + bundle.closure_items
        + blocked_items
    )
    for item in routed_items:
        lowered = item.summary.lower()
        if "fails under" in lowered:
            action_name = lowered.split(" fails under", 1)[0].replace(".", "").strip()
            if _action_text_bonus(action_name, query.admissible_actions) > 0:
                blocked.append(action_name)
        if "enables" in lowered:
            match = re.search(r"enables ([a-z0-9_(),= ]+)", lowered)
            if match:
                action_name = match.group(1).replace(".", "").strip()
                if _action_text_bonus(action_name, query.admissible_actions) > 0:
                    suggested.append(action_name)
        if "try " in lowered:
            match = re.search(r"try ([a-z0-9_(),= ]+)", lowered)
            if match:
                action_name = match.group(1).replace(".", "").strip()
                if _action_text_bonus(action_name, query.admissible_actions) > 0:
                    suggested.append(action_name)
        if item.candidate_type == CandidateType.WORKFLOW:
            workflows.append(item.summary.replace("Workflow pattern: ", ""))
    bundle.blocked_actions = list(dict.fromkeys(blocked))
    bundle.suggested_actions = list(dict.fromkeys(suggested))
    bundle.workflow_hints = list(dict.fromkeys(workflows))
    bundle.local_graph_items = _dedupe_support_items(list(bundle.fact_items), len(bundle.fact_items))
    bundle.local_promoted_items = _dedupe_support_items(
        list(bundle.workflow_items)
        + list(bundle.precondition_items)
        + list(bundle.repair_items)
        + list(bundle.closure_items),
        len(bundle.workflow_items)
        + len(bundle.precondition_items)
        + len(bundle.repair_items)
        + len(bundle.closure_items),
    )
    bundle.global_promoted_items = _dedupe_support_items(list(bundle.global_items), len(bundle.global_items))
    progress_state = str(query.progress_state or "")
    if progress_state.startswith("search"):
        bundle.task_need_analysis = [
            "Need concrete local grounding for search.",
            "Need local/global promoted guidance only when it sharpens the search path.",
        ]
    elif progress_state == "process_target":
        bundle.task_need_analysis = [
            "Need process prerequisites and tool guidance grounded in current state.",
        ]
    else:
        bundle.task_need_analysis = [
            "Need delivery-consistent support that advances goal completion.",
        ]
    bundle.local_graph_contribution = list(bundle.local_graph_items[:2])
    bundle.local_promoted_contribution = list(bundle.local_promoted_items[:2])
    bundle.global_promoted_contribution = list(bundle.global_promoted_items[:2])
    bundle.fused_support_items = _dedupe_support_items(
        list(bundle.local_graph_contribution)
        + list(bundle.local_promoted_contribution)
        + list(bundle.global_promoted_contribution),
        6,
    )
