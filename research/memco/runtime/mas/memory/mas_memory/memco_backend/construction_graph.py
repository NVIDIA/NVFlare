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
from dataclasses import asdict, dataclass
from typing import Any

from .alfworld_adapter import ALFWorldAdapter
from .graph_types import (
    ArtifactKind,
    CandidateType,
    EdgeType,
    EpisodeGraph,
    EpisodeRecord,
    GlobalGraphMemory,
    GraphEdge,
    GraphNode,
    LocalGraphMemory,
    MemoryArtifact,
    MemoryRule,
    NodeType,
    PromotionCandidate,
    RuleType,
    Stats,
)
from .promotion import (
    artifact_is_structurally_valid,
    artifact_wilson_evidence,
    candidate_is_structurally_valid,
    candidate_wilson_evidence,
    normalize_promotion_policy,
    rule_is_structurally_valid,
    rule_wilson_evidence,
)


def _merge_wilson_episode_evidence(
    target: dict[str, dict[str, int]],
    source: dict[str, dict[str, int]],
) -> None:
    """Merge the side-car episode ledger without changing legacy statistics."""

    for episode_id, verdict in source.items():
        merged = target.setdefault(
            episode_id,
            {"supporting": 0, "contradicting": 0, "stalled": 0},
        )
        for key in ("supporting", "contradicting", "stalled"):
            merged[key] += int((verdict or {}).get(key, 0))


@dataclass(slots=True)
class _ReplayBelief:
    progress: dict[str, object]
    next_progress: dict[str, object]
    search_ref: str | None
    revisit_count: int
    searched_locations: set[str]
    search_attempt_counts: dict[str, int]
    placed_relevant_count_est: int


def _memco_fever_policy_enabled() -> bool:
    return str(os.environ.get("NV_MEMCO_FEVER_POLICY", "")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
        "adaptive_cache",
    }


def _record_graph_stats(stats: Stats, *, positive: bool, stalled: bool = False) -> None:
    stats.record(positive=positive, stalled=stalled)


def _episode_success(episode: EpisodeRecord) -> bool:
    status = str(episode.metadata.get("status", "")).lower()
    if status in {"success", "succeeded"}:
        return True
    if status in {"fail", "failed"}:
        return False
    return float(episode.metadata.get("final_score", 0.0)) > 0


def _episode_domain(episode: EpisodeRecord) -> str:
    domain = str(episode.metadata.get("memco_domain", "") or "").strip().lower()
    if domain:
        return domain
    if episode.scene_id.startswith("pddl:"):
        return "pddl"
    if episode.scene_id.startswith("fever:"):
        return "fever"
    return "alfworld"


def _task_family(episode: EpisodeRecord) -> str:
    metadata_family = str(episode.metadata.get("task_family", "") or "").strip()
    if metadata_family:
        return metadata_family
    task_id = episode.task_id or ""
    if "-" in task_id:
        return task_id.split("-", 1)[0]
    goal = episode.goal.lower()
    if "under the" in goal or "with the" in goal:
        return "look_at_obj_in_light"
    return re.sub(r"[^a-z0-9_]+", "_", goal).strip("_") or "unknown_task"


def _task_roles(episode: EpisodeRecord) -> dict[str, str]:
    adapter = ALFWorldAdapter(scene_label=episode.scene_id or "alfworld_scene")
    slots = adapter.goal_slots(episode.goal)
    roles: dict[str, str] = {}
    target = slots.get("object", "")
    destination = slots.get("destination", "")
    tool = slots.get("tool", "")
    if target:
        roles[target] = "target_object"
    if destination:
        roles[destination] = "goal_destination"
    if tool:
        roles[tool] = "tool_object"
    return roles


def _goal_arity(episode: EpisodeRecord) -> int:
    goal = episode.goal.lower()
    if goal.startswith("put two "):
        return 2
    return 1


def _normalize_entity(value: str, roles: dict[str, str]) -> str:
    lowered = value.lower()
    base = re.sub(r"_\d+$", "", lowered)
    for key, role in roles.items():
        if key == base or key in base:
            return role
    if base in {
        "drawer",
        "cabinet",
        "safe",
        "fridge",
        "microwave",
        "garbagecan",
        "sinkbasin",
        "bathtubbasin",
        "toilet",
    }:
        return "container"
    if base in {
        "desk",
        "dresser",
        "shelf",
        "sidetable",
        "coffeetable",
        "diningtable",
        "bed",
        "sofa",
        "armchair",
        "countertop",
        "tvstand",
        "stoveburner",
    }:
        return "support_surface"
    return base


def _abstract_action_text(action, roles: dict[str, str]) -> str:
    if not action.slots:
        return action.verb
    normalized = {key: _normalize_entity(value, roles) for key, value in action.slots.items()}
    parts = ",".join(f"{key}={normalized[key]}" for key in sorted(normalized))
    return f"{action.verb}({parts})"


def _location_signature(location: str | None, roles: dict[str, str]) -> str:
    if not location:
        return ""
    return _normalize_entity(location, roles)


def _goal_target_base(goal_roles: dict[str, str]) -> str:
    return re.sub(r"_\d+$", "", str(goal_roles.get("object", "")).lower())


def _is_target_object(action, target_base: str) -> bool:
    if not target_base:
        return False
    raw_object = str(action.slots.get("object", "")).lower()
    raw_base = re.sub(r"_\d+$", "", raw_object)
    return bool(raw_base) and raw_base == target_base


def _replay_episode_beliefs(episode: EpisodeRecord) -> list[_ReplayBelief]:
    adapter = ALFWorldAdapter(scene_label=episode.scene_id or "alfworld_scene")
    goal_roles = adapter.goal_slots(episode.goal)
    goal_destination = str(goal_roles.get("destination", ""))
    target_base = _goal_target_base(goal_roles)
    required_count = adapter.required_goal_count(episode.goal)

    placed_relevant_count_est = 0
    searched_locations: set[str] = set()
    search_attempt_counts: dict[str, int] = defaultdict(int)
    beliefs: list[_ReplayBelief] = []

    for step in episode.steps:
        progress = adapter.summarize_goal_progress(
            step.state,
            episode.goal,
            belief={
                "placed_relevant_count_est": placed_relevant_count_est,
            },
        )
        search_ref = adapter.search_reference_for_action(step.action, step.state)
        revisit_count = search_attempt_counts.get(search_ref, 0) if search_ref else 0
        beliefs.append(
            _ReplayBelief(
                progress=progress,
                next_progress={},
                search_ref=search_ref,
                revisit_count=revisit_count,
                searched_locations=set(searched_locations),
                search_attempt_counts=dict(search_attempt_counts),
                placed_relevant_count_est=placed_relevant_count_est,
            )
        )

        if step.feedback.success and search_ref:
            searched_locations.add(search_ref)
            search_attempt_counts[search_ref] += 1

        if step.feedback.success and step.action.verb == "take" and _is_target_object(step.action, target_base):
            source = str(step.action.slots.get("source", ""))
            if goal_destination and source == goal_destination:
                placed_relevant_count_est = max(placed_relevant_count_est - 1, 0)
        if step.feedback.success and step.action.verb == "move" and _is_target_object(step.action, target_base):
            destination = str(step.action.slots.get("destination", ""))
            if goal_destination and destination == goal_destination:
                placed_relevant_count_est = min(required_count, placed_relevant_count_est + 1)

        beliefs[-1].next_progress = adapter.summarize_goal_progress(
            step.next_state,
            episode.goal,
            belief={
                "placed_relevant_count_est": placed_relevant_count_est,
            },
        )

    return beliefs


def _step_advances_goal(
    step,
    progress: dict[str, object],
    next_progress: dict[str, object],
    *,
    revisit_count: int,
) -> bool:
    if step.feedback.failure_label:
        return False
    if int(next_progress.get("held_relevant_count", 0)) > int(progress.get("held_relevant_count", 0)):
        return True
    if int(next_progress.get("placed_relevant_count", 0)) > int(progress.get("placed_relevant_count", 0)):
        return True
    if bool(next_progress.get("goal_object_matches_visible")) and not bool(progress.get("goal_object_matches_visible")):
        return True
    if str(next_progress.get("progress_state", "")) != str(progress.get("progress_state", "")):
        return True
    if step.action.verb == "look" and any(delta.startswith("visible+=") for delta in step.feedback.state_delta):
        return True
    if step.action.verb in {"open", "examine"} and any(
        delta.startswith("opened=") or delta.startswith("visible+=") for delta in step.feedback.state_delta
    ):
        return True
    if step.action.verb == "go" and any(delta.startswith("location=") for delta in step.feedback.state_delta):
        return revisit_count == 0
    return False


def _goal_aligned_action(
    action,
    *,
    progress_state: str,
    roles: dict[str, str],
) -> bool:
    verb = action.verb
    normalized_slots = {key: _normalize_entity(value, roles) for key, value in action.slots.items()}
    if verb == "take":
        return normalized_slots.get("object") == "target_object"
    if verb == "move":
        if normalized_slots.get("object") != "target_object":
            return False
        destination = normalized_slots.get("destination", "")
        return destination == "goal_destination"
    if verb == "go":
        target = normalized_slots.get("target", "")
        if progress_state in {"carry_target", "finalize"}:
            return target == "goal_destination"
        return target in {"container", "support_surface", "tool_object"}
    if verb == "open":
        return normalized_slots.get("container") == "container"
    if verb == "examine":
        target = normalized_slots.get("target", "")
        if progress_state in {"carry_target", "finalize"}:
            return target in {"goal_destination", "target_object"}
        return target in {"container", "support_surface", "target_object", "tool_object"}
    if verb == "look":
        return True
    return False


def _induce_plan_template(
    episode: EpisodeRecord,
    *,
    beliefs: list[_ReplayBelief],
    roles: dict[str, str],
    goal_roles: dict[str, str],
    goal_arity: int,
) -> tuple[str, list[str], list[str]] | None:
    if not episode.steps:
        return None

    destination = str(goal_roles.get("destination", "")).lower()
    destination_base = re.sub(r"_\d+$", "", destination)
    target = str(goal_roles.get("object", "")).lower()
    family = ALFWorldAdapter(scene_label=episode.scene_id or "alfworld_scene").infer_task_family(episode.goal)
    process_verb_by_family = {
        "pick_heat_then_place_in_recep": "heat",
        "pick_cool_then_place_in_recep": "cool",
        "pick_clean_then_place_in_recep": "clean",
    }
    process_verb = process_verb_by_family.get(family)

    first_take = None
    first_process = None
    process_tool_role = ""
    first_move = None
    source_role = ""
    for idx, step in enumerate(episode.steps):
        abstract_action = _abstract_action_text(step.action, roles)
        if (
            first_take is None
            and step.feedback.success
            and step.action.verb == "take"
            and _is_target_object(step.action, target)
        ):
            source_role = _normalize_entity(str(step.action.slots.get("source", "")), roles)
            first_take = abstract_action
        if (
            process_verb
            and first_process is None
            and first_take is not None
            and step.feedback.success
            and step.action.verb == process_verb
            and _is_target_object(step.action, target)
        ):
            process_tool_role = _normalize_entity(str(step.action.slots.get("tool", "")), roles)
            first_process = abstract_action
        if (
            first_move is None
            and step.feedback.success
            and step.action.verb == "move"
            and _is_target_object(step.action, target)
            and destination
        ):
            move_destination = str(step.action.slots.get("destination", "")).lower()
            move_destination_base = re.sub(r"_\d+$", "", move_destination)
            move_destination_role = _normalize_entity(move_destination, roles)
            if (
                move_destination == destination
                or move_destination_base == destination_base
                or move_destination_role == "goal_destination"
            ):
                first_move = abstract_action
        if first_take and first_move and (not process_verb or first_process):
            break

    if not first_take or not first_move:
        return None
    if process_verb and not first_process:
        return None

    if source_role not in {"support_surface", "container"}:
        source_role = "support_surface"
    if process_tool_role not in {"support_surface", "container", "tool_object"}:
        process_tool_role = "container"

    search_step = f"go(target={source_role})"
    delivery_step = "go(target=goal_destination)"
    plan_steps = [search_step, first_take]
    if first_process:
        plan_steps.extend([f"go(target={process_tool_role})", first_process])
    plan_steps.extend([delivery_step, first_move])
    action_patterns = list(plan_steps)

    if goal_arity > 1:
        plan_steps.append("repeat acquire and deliver for remaining target_object")

    summary = "Plan: " + " -> ".join(plan_steps)
    return summary, plan_steps, action_patterns


def _relation_action_patterns(source_instance: str, source_base: str, source_role: str) -> list[str]:
    patterns: list[str] = []
    if source_instance:
        patterns.append(f"go(target={source_instance})")
        if source_role == "container":
            patterns.append(f"open(container={source_instance})")
            patterns.append(f"examine(target={source_instance})")
        else:
            patterns.append(f"examine(target={source_instance})")
    if source_base and source_base != source_instance:
        patterns.append(f"go(target={source_base})")
        if source_role == "container":
            patterns.append(f"open(container={source_base})")
            patterns.append(f"examine(target={source_base})")
    if source_role:
        patterns.append(f"go(target={source_role})")
        if source_role == "container":
            patterns.append("open(container=container)")
        else:
            patterns.append(f"examine(target={source_role})")
    return list(dict.fromkeys(patterns))


def _condition_signature(payload: dict) -> str:
    pieces: list[str] = []
    for key in sorted(payload):
        value = payload[key]
        if isinstance(value, dict):
            inner = ",".join(f"{inner_key}={value[inner_key]}" for inner_key in sorted(value))
            pieces.append(f"{key}={{" + inner + "}}")
        elif isinstance(value, (list, tuple)):
            pieces.append(f"{key}=[{','.join(str(item) for item in value)}]")
        else:
            pieces.append(f"{key}={value}")
    return "|".join(pieces)


def _abstract_goal_roles(goal_roles: dict[str, str]) -> dict[str, str]:
    roles: dict[str, str] = {}
    if goal_roles.get("object"):
        roles["object"] = "target_object"
    if goal_roles.get("destination"):
        roles["destination"] = "goal_destination"
    if goal_roles.get("tool"):
        roles["tool"] = "tool_object"
    return roles


def _abstract_global_value(value: Any) -> Any:
    if isinstance(value, dict):
        return _abstract_global_mapping(value)
    if isinstance(value, list):
        return [_abstract_global_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_abstract_global_value(item) for item in value)
    return value


def _abstract_global_mapping(payload: dict) -> dict:
    """Drop instance/layout anchors before promoting local evidence globally."""
    drop_keys = {
        "goal_signature",
        "layout_id",
        "source_instance",
        "source_base",
        "graph_refs",
        "state_ref",
        "next_state_ref",
        "target_object",
    }
    abstracted: dict[str, Any] = {}
    for key, value in payload.items():
        if key in drop_keys:
            continue
        abstracted[key] = _abstract_global_value(value)
    return abstracted


def _global_rule_id(rule: MemoryRule, condition: dict, effect: dict) -> str:
    return (
        f"global:{rule.rule_type.value}:{rule.task_family}:{rule.goal_arity}:"
        f"{rule.progress_state}:{_condition_signature(condition)}:"
        f"{_condition_signature(effect)}"
    )


def _global_artifact_id(artifact: MemoryArtifact, anchor: dict, payload: dict) -> str:
    return f"global:{artifact.kind.value}:" f"{_condition_signature(anchor)}:" f"{_condition_signature(payload)}"


def _rule_specificity(rule: MemoryRule) -> float:
    if str(os.getenv("NV_MEMCO_PAPER_EQ22_27_ALL_ONES", "")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        condition_blob = " ".join(str(value) for value in rule.condition.values())
        instance_count = len(re.findall(r"\b[a-z]+_[0-9]+\b", condition_blob.lower()))
        object_destination = int(bool(rule.goal_roles.get("object") and rule.goal_roles.get("destination")))
        return min(1.0, 1.0 * instance_count + 1.0 * object_destination)
    score = 0.0
    condition_blob = " ".join(str(value) for value in rule.condition.values())
    specific_tokens = re.findall(r"\b[a-z]+_[0-9]+\b", condition_blob.lower())
    if specific_tokens:
        score += min(0.25, 0.08 * len(specific_tokens))
    if rule.goal_roles.get("object") and rule.goal_roles.get("destination"):
        score += 0.04
    return min(score, 0.35)


def _artifact_specificity(artifact: MemoryArtifact) -> float:
    score = 0.0
    anchor_blob = " ".join(str(value) for value in artifact.anchor.values())
    payload_blob = " ".join(str(value) for value in artifact.payload.values())
    specific_tokens = re.findall(r"\b[a-z]+_[0-9]+\b", f"{anchor_blob} {payload_blob}".lower())
    if specific_tokens:
        score += min(0.25, 0.08 * len(specific_tokens))
    if artifact.anchor.get("goal_signature") and artifact.anchor.get("progress_state"):
        score += 0.03
    return min(score, 0.35)


def _is_low_information_global_rule(rule: MemoryRule) -> bool:
    """Filter rules that are true but too generic to transfer usefully.

    For ALFWorld, rules such as "within search, open advances the search" are
    often correct but not discriminative. They crowd out the more useful global
    memories: workflow skeletons, source-type priors, and concrete phase
    transitions. Keep this filter conservative so local memory remains intact.
    """
    summary = str(rule.summary or "").lower()
    progress = str(rule.progress_state or "").lower()
    effect = rule.effect or {}
    prefer_action = str(effect.get("prefer_action", "") or "").lower()
    via = str(effect.get("via", "") or "").lower()
    action_role = str(effect.get("action_role", "") or "").lower()
    task_family = str(rule.task_family or "").lower()

    if task_family.startswith("pddl") and (
        "check valid actions" in summary
        or prefer_action.startswith(("check_valid_actions", "check valid actions"))
        or via in {"check_valid_actions", "check valid actions", "check valid action"}
    ):
        return True

    if "within search" in summary and via in {"open", "go", "look", "examine"}:
        return True
    if progress.startswith("search") and via in {"open", "go", "look", "examine"}:
        return True
    if progress.startswith("search") and prefer_action.startswith(("open(", "go(", "look(", "examine(")):
        return True
    if progress.startswith("search") and action_role in {"container", "support_surface", "unknown"}:
        return True
    return False


def _is_low_information_global_artifact(artifact: MemoryArtifact) -> bool:
    summary = str(artifact.summary or "").lower()
    payload = artifact.payload or {}
    pattern_kind = str(payload.get("pattern_kind", "") or artifact.anchor.get("pattern_kind", ""))
    source_base = str(payload.get("source_base", "") or "")
    source_role = str(payload.get("source_role", "") or "")
    task_family = str(artifact.anchor.get("task_family", "") or payload.get("task_family", "") or "").lower()

    if task_family.startswith("pddl") and pattern_kind in {"workflow", "precondition"}:
        if "check valid actions" in summary or "check_valid_actions" in summary:
            return True

    if pattern_kind == "source_type_prior":
        return not bool(source_base or source_role)
    if "within search" in summary and "open" in summary and "advances" in summary:
        return True
    if summary.strip() in {
        "prototype: workflow: within search, open supports progress.",
        "prototype: within search, open supports progress.",
    }:
        return True
    return False


def _candidate_action_patterns(candidate: PromotionCandidate) -> tuple[str, ...]:
    patterns: list[str] = []
    for key in ("action", "repair_action"):
        value = str(candidate.structure.get(key, "")).strip()
        if value:
            patterns.append(value)
    patterns.extend(re.findall(r"[a-z_]+\([a-z0-9_,= ]+\)", candidate.summary.lower()))
    return tuple(dict.fromkeys(patterns))


def _candidate_support(candidate: PromotionCandidate) -> int:
    return candidate.positive + candidate.negative


def _keep_candidate(candidate: PromotionCandidate) -> bool:
    pattern_kind = str(candidate.structure.get("pattern_kind", ""))
    support = _candidate_support(candidate)
    confidence = candidate.confidence

    if candidate.candidate_type == CandidateType.PRECONDITION:
        # Keep only clearly helpful preconditions; generic negative rules such
        # as the old holding_before_move candidate are more harmful than useful.
        return support >= 2 and candidate.positive >= 2 and confidence >= 0.6

    if candidate.candidate_type == CandidateType.WORKFLOW:
        if pattern_kind == "closure":
            return candidate.positive >= 1
        # Generic search workflows need noticeably more positive than negative
        # evidence to stay in memory.
        return candidate.positive >= 2 and candidate.positive >= candidate.negative

    if candidate.candidate_type == CandidateType.REPAIR:
        return candidate.positive >= 1

    if candidate.candidate_type == CandidateType.FAILURE:
        if pattern_kind == "anti_pattern":
            # Keep only stable, repeatable anti-patterns as reusable failure
            # memory. One-off stalls add noise without giving a reliable blocker.
            return candidate.stalled >= 2
        # Raw failure candidates are useful as evidence during rule/artifact
        # induction, but they are too noisy to stay in the top-level candidate
        # pool. We rely on blocked/repair rules to turn failures into positive
        # guidance instead.
        return False

    return True


def _rule_action_patterns(rule: MemoryRule) -> tuple[str, ...]:
    patterns: list[str] = []
    for key in ("prefer_action", "block_action", "action"):
        value = str(rule.effect.get(key, "")).strip()
        if value:
            patterns.append(value)
    return tuple(dict.fromkeys(patterns))


def _safe_pattern_id(value: str) -> str:
    return re.sub(r"[^a-z0-9_=-]+", "_", str(value or "").lower()).strip("_")[:160] or "unknown"


def _generic_action_patterns(step) -> list[str]:
    patterns = [step.action.canonical_str]
    surface = str(getattr(step.action, "surface_form", "") or "").strip()
    if surface:
        patterns.append(surface)
    return list(dict.fromkeys(patterns))


def _generic_action_text(step, *, prefer_surface: bool = False) -> str:
    surface = str(getattr(step.action, "surface_form", "") or "").strip()
    if prefer_surface and surface:
        return surface
    return step.action.canonical_str


_FEVER_LABELS = {
    "supports",
    "refutes",
    "not enough info",
    "not_enough_info",
}


def _fever_claim_text(episode: EpisodeRecord) -> str:
    return re.sub(r"^\s*verify claim:\s*", "", str(episode.goal or ""), flags=re.IGNORECASE).strip()


def _fever_action_arg(step) -> str:
    surface = str(getattr(step.action, "surface_form", "") or "").strip()
    match = re.match(r"^\s*[A-Za-z_]+\[(.*)\]\s*$", surface)
    if match:
        return match.group(1).strip()
    for key in ("query", "keyword", "label"):
        value = str(getattr(step.action, "slots", {}).get(key, "") or "").strip()
        if value:
            return value.replace("_", " ")
    return ""


def _fever_is_finish(step) -> bool:
    return str(getattr(step.action, "verb", "") or "").lower() == "finish"


def _fever_is_search(step) -> bool:
    return str(getattr(step.action, "verb", "") or "").lower() == "search"


def _fever_is_lookup(step) -> bool:
    return str(getattr(step.action, "verb", "") or "").lower() == "lookup"


def _fever_no_results(step) -> bool:
    text = " ".join(
        str(value or "")
        for value in (
            getattr(step.state, "raw_observation", ""),
            getattr(step.next_state, "raw_observation", ""),
            getattr(step.feedback, "failure_label", ""),
        )
    ).lower()
    return any(marker in text for marker in ("no results", "cannot find", "not found", "lookup_without_page"))


def _fever_observation_text(step) -> str:
    return str(
        getattr(getattr(step, "next_state", None), "raw_observation", "")
        or getattr(getattr(step, "state", None), "raw_observation", "")
        or ""
    )


def _fever_clean_evidence_snippet(text: str, *, limit: int = 260) -> str:
    cleaned = re.sub(r"\s+", " ", str(text or "").strip())
    if not cleaned:
        return ""
    lowered = cleaned.lower()
    if any(marker in lowered for marker in ("invalid action", "wrong answer", "game over")):
        return ""
    cleaned = re.sub(r"\b(?:SUPPORTS|REFUTES|NOT ENOUGH INFO|NOT_ENOUGH_INFO)\b", "", cleaned, flags=re.I)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .;")
    if len(cleaned) > limit:
        cleaned = cleaned[:limit].rsplit(" ", 1)[0].strip(" .;,") + "..."
    return cleaned


def _fever_memory_tokens(text: str) -> list[str]:
    stop = {
        "claim",
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "of",
        "in",
        "on",
        "and",
        "or",
        "to",
        "by",
        "for",
        "with",
        "has",
        "have",
        "had",
        "not",
        "from",
        "that",
        "this",
    }
    tokens = re.findall(r"[a-z0-9]+", str(text or "").lower())
    return [token for token in tokens if len(token) > 2 and token not in stop]


def _fever_evidence_terms(*, claim: str, evidence: str, search_arg: str = "", lookup_arg: str = "") -> list[str]:
    evidence_norm = set(_fever_memory_tokens(evidence))
    candidates = _fever_memory_tokens(" ".join([claim, search_arg, lookup_arg]))
    terms: list[str] = []
    for token in candidates:
        if token in evidence_norm and token not in terms:
            terms.append(token)
        if len(terms) >= 12:
            break
    if len(terms) < 2:
        for token in _fever_memory_tokens(search_arg):
            if token not in terms:
                terms.append(token)
            if len(terms) >= 4:
                break
    return terms


def _fever_relation_keyword_hint(claim: str, *, search_arg: str = "", lookup_arg: str = "") -> str:
    """Return a short relation/attribute phrase type without turning labels into priors."""
    if lookup_arg:
        normalized_lookup = re.sub(r"[_\s]+", " ", lookup_arg).strip().lower()
        if normalized_lookup and normalized_lookup not in _FEVER_LABELS:
            return "claim_relation_keyword"
    text = re.sub(r"[^a-z0-9\s]+", " ", str(claim or "").lower())
    if search_arg:
        search_tokens = set(re.findall(r"[a-z0-9]+", search_arg.lower()))
    else:
        search_tokens = set()
    stop = {
        "claim",
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "of",
        "in",
        "on",
        "and",
        "or",
        "to",
        "by",
        "for",
        "with",
        "has",
        "have",
        "had",
        "not",
        "no",
    }
    words = [word for word in text.split() if len(word) > 2 and word not in stop and word not in search_tokens]
    return "claim_relation_keyword" if words else "claim_keyword"


def _fever_claim_profile(claim: str) -> dict[str, Any]:
    text = str(claim or "").strip()
    norm = re.sub(r"[^a-z0-9\s]+", " ", text.lower())
    primary = _fever_claim_anchor(text)
    secondary = _fever_secondary_entity_hint(text, primary)
    claim_type = "general_fact"
    lookup_keywords: list[str] = []
    search_role = "primary entity"

    if any(marker in norm for marker in (" soundtrack", " track", " song ", " album", " single")):
        claim_type = "album_track_membership"
        search_role = "song or album title"
        lookup_keywords = ["soundtrack", "track listing", "album", "song"]
    elif any(
        marker in norm
        for marker in (" cast", " stars ", " starring", " starred", " appeared in", " actor", " actress", " played ")
    ):
        claim_type = "cast_membership"
        search_role = "film, show, or person title"
        lookup_keywords = ["cast", "starring", "appeared", "played by"]
    elif any(
        marker in norm
        for marker in (" directed", " director", " created", " creator", " producer", " wrote", " written")
    ):
        claim_type = "creative_role"
        search_role = "work or creator title"
        lookup_keywords = ["director", "creator", "producer", "writer"]
    elif any(marker in norm for marker in (" released", " release date", " premiered", " publication", " published")):
        claim_type = "release_date"
        search_role = "claimed work or artist title"
        lookup_keywords = ["release date", "released", "premiered", "published"]
    elif any(marker in norm for marker in (" genre", " style", " type of music", " rock", " pop", " hip hop", " jazz")):
        claim_type = "genre"
        search_role = "artist, work, or album title"
        lookup_keywords = ["genre", "style", "music genre"]
    elif any(marker in norm for marker in (" award", " won", " nominated", " grammy", " oscar", " emmy")):
        claim_type = "award"
        search_role = "person or work title"
        lookup_keywords = ["awards", "nominations", "won"]
    elif any(
        marker in norm
        for marker in (" singer", " musician", " band", " guitarist", " vocalist", " rapper", " composer")
    ):
        claim_type = "music_role"
        search_role = "artist or band name"
        lookup_keywords = ["occupation", "members", "associated acts", "music career"]
    elif any(marker in norm for marker in (" born", " nationality", " country", " city", " located", " from ")):
        claim_type = "nationality_location"
        search_role = "person or place name"
        lookup_keywords = ["born", "nationality", "origin", "location"]
    elif any(marker in norm for marker in (" occupation", " profession", " known for", " worked as")):
        claim_type = "profession"
        search_role = "person name"
        lookup_keywords = ["occupation", "known for", "career"]
    elif any(marker in norm for marker in (" spouse", " married", " member of", " part of", " belongs to")):
        claim_type = "identity_relation"
        search_role = "person or group name"
        lookup_keywords = ["relationship", "member", "spouse"]

    if not lookup_keywords:
        relation = _fever_relation_keyword_hint(claim, search_arg=primary)
        lookup_keywords = [relation.replace("_", " ")]
    query_variants = _fever_query_variants(
        text,
        primary_entity=primary,
        secondary_entity=secondary,
        claim_type=claim_type,
        lookup_keywords=lookup_keywords,
    )
    return {
        "claim_type": claim_type,
        "primary_entity": primary,
        "secondary_entity": secondary,
        "search_role": search_role,
        "lookup_keywords": lookup_keywords[:5],
        "query_variants": query_variants[:6],
    }


def _fever_query_variants(
    claim: str,
    *,
    primary_entity: str,
    secondary_entity: str = "",
    claim_type: str = "general_fact",
    lookup_keywords: list[str] | None = None,
) -> list[str]:
    primary = re.sub(r"[_\s]+", " ", str(primary_entity or "")).strip()
    secondary = re.sub(r"[_\s]+", " ", str(secondary_entity or "")).strip()
    keywords = [re.sub(r"[_\s]+", " ", str(item or "")).strip() for item in (lookup_keywords or [])]
    claim_norm = re.sub(r"[^a-z0-9\s]+", " ", str(claim or "").lower())
    variants: list[str] = []

    def add(value: str) -> None:
        cleaned = re.sub(r"\s+", " ", str(value or "")).strip(" .,:;\"'")
        key = re.sub(r"[^a-z0-9]+", "_", cleaned.lower()).strip("_")
        if cleaned and key not in {re.sub(r"[^a-z0-9]+", "_", item.lower()).strip("_") for item in variants}:
            variants.append(cleaned)

    if primary and secondary:
        add(f"{primary} {secondary}")
    if primary and "film career" in claim_norm:
        add(f"{primary} filmography")
        add(f"{primary} film career")
    if primary and claim_type in {"creative_role", "cast_membership", "identity_relation"} and keywords:
        add(f"{primary} {secondary or keywords[0]}")
    if primary and claim_type == "album_track_membership":
        add(f"{primary} album")
        add(f"{primary} track listing")
    if primary and claim_type in {"cast_membership", "creative_role", "general_fact"}:
        if any(word in claim_norm for word in (" film", " movie", " drama", " directed", " stars", " starred")):
            add(f"{primary} film")
            add(f"{primary} movie")
        if any(word in claim_norm for word in (" tv ", " television", " series", " show")):
            add(f"{primary} TV series")
    if primary and keywords:
        add(f"{primary} {keywords[0]}")
    if primary:
        add(primary)
    return variants


def _fever_secondary_entity_hint(claim: str, primary_entity: str = "") -> str:
    candidates = re.findall(
        r"\b(?:[A-Z][A-Za-z0-9]*|[A-Z]\.)(?:\s+(?:(?:the|of|and|in|on|for|to|a|an)\b|[A-Z][A-Za-z0-9]*|[A-Z]\.)){0,5}",
        str(claim or ""),
    )
    primary_norm = re.sub(r"[^a-z0-9_]+", "_", str(primary_entity or "").lower()).strip("_")
    for candidate in candidates[1:]:
        normalized = re.sub(r"[^a-z0-9_]+", "_", candidate.strip().lower()).strip("_")
        if normalized and normalized != primary_norm:
            return normalized
    return ""


def _fever_claim_anchor(claim: str) -> str:
    text = str(claim or "").strip()
    called = re.search(r"\bthere\s+(?:is|are|was|were)\b.*?\bcalled\s+(.+?)(?:[.;]|$)", text, flags=re.I)
    if called:
        candidate = called.group(1).strip(" .,:;\"'")
        if candidate:
            return re.sub(r"[^a-z0-9_]+", "_", candidate.strip().lower()).strip("_")
    predicate = re.search(
        r"\b(?:is|are|was|were|has|have|had|does|do|did|worked|appeared|released|formed|created|directed|stars|starred|starring|features|featured|includes|included)\b",
        text,
        flags=re.I,
    )
    if predicate:
        subject = text[: predicate.start()].strip(" .,:;\"'")
        subject = re.sub(r"\b(?:only|also|just)\s*$", "", subject, flags=re.I).strip(" .,:;\"'")
        if 1 <= len(re.findall(r"[A-Za-z0-9]+", subject)) <= 8:
            return re.sub(r"[^a-z0-9_]+", "_", subject.strip().lower()).strip("_")
    candidates = re.findall(
        r"\b(?:[A-Z][A-Za-z0-9]*|[A-Z]\.)(?:\s+(?:(?:the|of|and|in|on|for|to|a|an)\b|[A-Z][A-Za-z0-9]*|[A-Z]\.)){0,5}",
        text,
    )
    if candidates:
        return re.sub(r"[^a-z0-9_]+", "_", candidates[0].strip().lower()).strip("_")
    return ""


class EpisodeGraphBuilder:
    def build(self, episode: EpisodeRecord) -> EpisodeGraph:
        episode_id = f"{episode.agent_id}:{episode.task_id}:{len(episode.steps)}"
        graph = EpisodeGraph(
            episode_id=episode_id,
            agent_id=episode.agent_id,
            scene_id=episode.scene_id,
            task_id=episode.task_id,
            goal=episode.goal,
            metadata=dict(episode.metadata),
        )

        for step in episode.steps:
            state_sig = f"state:{step.state.signature}"
            next_state_sig = f"state:{step.next_state.signature}"
            action_sig = f"action:{step.action.canonical_str}"

            graph.nodes.setdefault(
                state_sig,
                GraphNode(state_sig, NodeType.STATE, step.state.signature, asdict(step.state)),
            )
            graph.nodes.setdefault(
                action_sig,
                GraphNode(action_sig, NodeType.ACTION, step.action.canonical_str, asdict(step.action)),
            )
            graph.nodes.setdefault(
                next_state_sig,
                GraphNode(next_state_sig, NodeType.STATE, step.next_state.signature, asdict(step.next_state)),
            )

            graph.edges.append(
                GraphEdge(
                    src=state_sig,
                    dst=action_sig,
                    edge_type=EdgeType.TEMPORAL,
                    signature=f"{state_sig}|temporal|{action_sig}",
                    payload={"step_idx": step.step_idx},
                    stats=Stats(support=1, positive=1),
                )
            )
            graph.edges.append(
                GraphEdge(
                    src=action_sig,
                    dst=next_state_sig,
                    edge_type=EdgeType.TEMPORAL,
                    signature=f"{action_sig}|temporal|{next_state_sig}",
                    payload={"step_idx": step.step_idx},
                    stats=Stats(support=1, positive=1),
                )
            )

            if step.feedback.state_delta:
                delta_sig = f"delta:{step.step_idx}:{'|'.join(step.feedback.state_delta)}"
                graph.nodes.setdefault(
                    delta_sig,
                    GraphNode(
                        delta_sig,
                        NodeType.STATE,
                        delta_sig,
                        {"delta": list(step.feedback.state_delta)},
                    ),
                )
                graph.edges.append(
                    GraphEdge(
                        src=action_sig,
                        dst=delta_sig,
                        edge_type=EdgeType.CAUSES,
                        signature=f"{action_sig}|causes|{delta_sig}",
                        payload={"delta": list(step.feedback.state_delta)},
                        stats=Stats(
                            support=1, positive=int(step.feedback.success), negative=int(not step.feedback.success)
                        ),
                    )
                )

            if step.subgoal:
                subgoal_sig = f"subgoal:{step.subgoal}"
                graph.nodes.setdefault(
                    subgoal_sig,
                    GraphNode(subgoal_sig, NodeType.SUBGOAL, step.subgoal, {"name": step.subgoal}),
                )
                graph.edges.append(
                    GraphEdge(
                        src=action_sig,
                        dst=subgoal_sig,
                        edge_type=EdgeType.ADVANCES_TO,
                        signature=f"{action_sig}|advances_to|{subgoal_sig}",
                        payload={"step_idx": step.step_idx},
                        stats=Stats(
                            support=1, positive=int(step.feedback.success), negative=int(not step.feedback.success)
                        ),
                    )
                )

            if step.feedback.failure_label:
                failure_sig = f"failure:{step.feedback.failure_label}"
                graph.nodes.setdefault(
                    failure_sig,
                    GraphNode(
                        failure_sig,
                        NodeType.FAILURE,
                        step.feedback.failure_label,
                        {"label": step.feedback.failure_label},
                    ),
                )
                graph.edges.append(
                    GraphEdge(
                        src=action_sig,
                        dst=failure_sig,
                        edge_type=EdgeType.FAILS_UNDER,
                        signature=f"{action_sig}|fails_under|{failure_sig}",
                        payload={"step_idx": step.step_idx},
                        stats=Stats(support=1, negative=1),
                    )
                )

        return graph


class LocalGraphMaintainer:
    def __init__(self, *, record_wilson_evidence: bool = False) -> None:
        self.record_wilson_evidence = bool(record_wilson_evidence)

    def update(self, memory: LocalGraphMemory, episode_graph: EpisodeGraph, episode: EpisodeRecord) -> LocalGraphMemory:
        if episode_graph.episode_id not in memory.episode_ids:
            memory.episode_ids.append(episode_graph.episode_id)
        episode_success = _episode_success(episode)

        for node in episode_graph.nodes.values():
            local_node = memory.nodes_by_signature.get(node.signature)
            if local_node is None:
                memory.nodes_by_signature[node.signature] = GraphNode(
                    node_id=node.node_id,
                    node_type=node.node_type,
                    signature=node.signature,
                    payload=dict(node.payload),
                )
                local_node = memory.nodes_by_signature[node.signature]
            _record_graph_stats(local_node.stats, positive=episode_success)

        stalled_steps = {
            step.step_idx
            for step in episode.steps
            if not step.feedback.failure_label and not step.feedback.state_delta and not step.feedback.done
        }
        for edge in episode_graph.edges:
            local_edge = memory.edges_by_signature.get(edge.signature)
            if local_edge is None:
                memory.edges_by_signature[edge.signature] = GraphEdge(
                    src=edge.src,
                    dst=edge.dst,
                    edge_type=edge.edge_type,
                    signature=edge.signature,
                    payload=dict(edge.payload),
                )
                local_edge = memory.edges_by_signature[edge.signature]
            step_idx = int(edge.payload.get("step_idx", -1))
            stalled = step_idx in stalled_steps
            positive = episode_success and edge.edge_type != EdgeType.FAILS_UNDER
            _record_graph_stats(local_edge.stats, positive=positive, stalled=stalled)

        candidates = self._induce_candidates(episode_graph, episode, episode_success=episode_success)
        if not self.record_wilson_evidence:
            for candidate in candidates:
                candidate.wilson_episode_evidence.clear()
        for candidate in candidates:
            existing = memory.candidates.get(candidate.candidate_id)
            if existing is None:
                memory.candidates[candidate.candidate_id] = candidate
            else:
                existing.source_episode_ids |= candidate.source_episode_ids
                existing.source_scenes |= candidate.source_scenes
                existing.positive += candidate.positive
                existing.negative += candidate.negative
                existing.stalled += candidate.stalled
                existing.utility += candidate.utility
                _merge_wilson_episode_evidence(
                    existing.wilson_episode_evidence,
                    candidate.wilson_episode_evidence,
                )

        rules = self._induce_rules(episode_graph, episode, episode_success=episode_success)
        if not self.record_wilson_evidence:
            for rule in rules:
                rule.wilson_episode_evidence.clear()
        for rule in rules:
            existing_rule = memory.rules_by_id.get(rule.rule_id)
            if existing_rule is None:
                rule.specificity = _rule_specificity(rule)
                memory.rules_by_id[rule.rule_id] = rule
            else:
                existing_rule.source_episode_ids |= rule.source_episode_ids
                existing_rule.source_scenes |= rule.source_scenes
                existing_rule.stats.support += rule.stats.support
                existing_rule.stats.success += rule.stats.success
                existing_rule.stats.failure += rule.stats.failure
                existing_rule.stats.stalled += rule.stats.stalled
                existing_rule.stats.utility += rule.stats.utility
                existing_rule.stats.transfer_success += rule.stats.transfer_success
                existing_rule.stats.transfer_trials += rule.stats.transfer_trials
                _merge_wilson_episode_evidence(
                    existing_rule.wilson_episode_evidence,
                    rule.wilson_episode_evidence,
                )
                existing_rule.specificity = max(existing_rule.specificity, _rule_specificity(existing_rule))
        artifacts = self._induce_artifacts(
            episode_graph,
            episode,
            candidates=candidates,
            rules=rules,
            episode_success=episode_success,
        )
        if not self.record_wilson_evidence:
            for artifact in artifacts:
                artifact.wilson_episode_evidence.clear()
        for artifact in artifacts:
            existing_artifact = memory.artifacts_by_id.get(artifact.artifact_id)
            if existing_artifact is None:
                artifact.specificity = _artifact_specificity(artifact)
                memory.artifacts_by_id[artifact.artifact_id] = artifact
                continue
            existing_artifact.source_episode_ids |= artifact.source_episode_ids
            existing_artifact.source_scenes |= artifact.source_scenes
            existing_artifact.stats.support += artifact.stats.support
            existing_artifact.stats.success += artifact.stats.success
            existing_artifact.stats.failure += artifact.stats.failure
            existing_artifact.stats.stalled += artifact.stats.stalled
            existing_artifact.stats.utility += artifact.stats.utility
            existing_artifact.stats.transfer_success += artifact.stats.transfer_success
            existing_artifact.stats.transfer_trials += artifact.stats.transfer_trials
            _merge_wilson_episode_evidence(
                existing_artifact.wilson_episode_evidence,
                artifact.wilson_episode_evidence,
            )
            existing_artifact.specificity = max(existing_artifact.specificity, _artifact_specificity(existing_artifact))
        return memory

    def refine_memory(self, memory: LocalGraphMemory) -> LocalGraphMemory:
        refined_candidates: dict[str, PromotionCandidate] = {}
        for candidate_id, candidate in memory.candidates.items():
            if _keep_candidate(candidate):
                refined_candidates[candidate_id] = candidate
        memory.candidates = refined_candidates
        # additional refinement for artifacts: unify object_location_prior entries and remove low-support noise
        import os

        try:
            memco_min_support = int(os.getenv("NV_MEMCO_MIN_ARTIFACT_SUPPORT", "2"))
        except Exception:
            memco_min_support = 2
        try:
            memco_min_confidence = float(os.getenv("NV_MEMCO_MIN_ARTIFACT_CONFIDENCE", "0.4"))
        except Exception:
            memco_min_confidence = 0.4
        unified: dict = {}
        to_remove: set[str] = set()
        for aid, artifact in list(memory.artifacts_by_id.items()):
            try:
                payload = artifact.payload or {}
                pattern_kind = str(payload.get("pattern_kind", "") or "")
                relation_kind = str(payload.get("relation_kind", "") or "")
                object_role = str(payload.get("object_role", "") or "")
                # unify only object_location_prior; skip others
                if pattern_kind == "scene_relation" and relation_kind == "object_location_prior":
                    base = str(payload.get("source_base", "") or "")
                    inst = str(payload.get("source_instance", "") or "")
                    # filter low-support noisy artifacts
                    support = artifact.stats.support
                    conf = artifact.stats.confidence
                    if support < memco_min_support and conf < memco_min_confidence:
                        to_remove.add(aid)
                        continue
                    # Local graph should preserve exact source instances when
                    # available. Cross-scene abstraction happens later during
                    # global promotion through source_type_prior artifacts.
                    key = (pattern_kind, relation_kind, object_role, base, inst or base)
                    existing = unified.get(key)
                    if existing is None:
                        unified[key] = artifact
                    else:
                        # prefer artifact with higher support and confidence
                        if (artifact.stats.support > existing.stats.support) or (
                            artifact.stats.support == existing.stats.support
                            and artifact.stats.confidence > existing.stats.confidence
                        ):
                            to_remove.add(existing.artifact_id)
                            unified[key] = artifact
                        else:
                            to_remove.add(aid)
                else:
                    # optionally filter general low-support/low-confidence artifacts
                    support = artifact.stats.support
                    conf = artifact.stats.confidence
                    if support < memco_min_support and conf < memco_min_confidence:
                        to_remove.add(aid)
            except Exception:
                continue
        for aid in to_remove:
            memory.artifacts_by_id.pop(aid, None)
        return memory

    def _induce_candidates(
        self,
        episode_graph: EpisodeGraph,
        episode: EpisodeRecord,
        *,
        episode_success: bool,
    ) -> list[PromotionCandidate]:
        domain = _episode_domain(episode)
        if domain == "fever":
            return []
        if domain != "alfworld":
            return self._induce_generic_candidates(episode_graph, episode, episode_success=episode_success)
        candidates: list[PromotionCandidate] = []
        family = _task_family(episode)
        roles = _task_roles(episode)
        subgoals = [step.subgoal for step in episode.steps if step.subgoal]
        stalled_steps = {
            step.step_idx
            for step in episode.steps
            if not step.feedback.failure_label and not step.feedback.state_delta and not step.feedback.done
        }
        if len(subgoals) >= 2:
            workflow = tuple(dict.fromkeys(subgoals))
            workflow_id = f"workflow:{family}:" + "->".join(workflow)
            candidate = PromotionCandidate(
                candidate_id=workflow_id,
                candidate_type=CandidateType.WORKFLOW,
                summary=f"Workflow pattern: {' -> '.join(workflow)}",
                structure={
                    "workflow": workflow,
                    "goal": episode.goal,
                    "task_family": family,
                    "pattern_kind": "workflow",
                },
            )
            candidate.observe(
                scene_id=episode.scene_id,
                episode_id=episode_graph.episode_id,
                positive=episode_success,
                utility_delta=0.1 if episode_success else 0.02,
            )
            candidates.append(candidate)

        for step in episode.steps:
            if step.action.verb == "open" and step.state.location == step.action.slots.get("container"):
                cid = f"precondition:{family}:at_target_before_{step.action.verb}"
                candidate = PromotionCandidate(
                    candidate_id=cid,
                    candidate_type=CandidateType.PRECONDITION,
                    summary="Being at container enables open(container).",
                    structure={
                        "precondition": {"location_matches_container": True},
                        "action": "open(container)",
                        "task_family": family,
                        "pattern_kind": "precondition",
                    },
                )
                candidate.observe(
                    scene_id=episode.scene_id,
                    episode_id=episode_graph.episode_id,
                    positive=step.feedback.success,
                    stalled=step.step_idx in stalled_steps,
                    utility_delta=0.05 if step.feedback.success else 0.0,
                )
                candidates.append(candidate)

            goal_roles = ALFWorldAdapter(scene_label=episode.scene_id or "alfworld_scene").goal_slots(episode.goal)
            target_object = str(goal_roles.get("object", ""))
            goal_destination = str(goal_roles.get("destination", ""))
            moved_object = str(step.action.slots.get("object", ""))
            move_destination = str(step.action.slots.get("destination", ""))
            target_match = bool(target_object and moved_object and re.sub(r"_\\d+$", "", moved_object) == target_object)
            if (
                step.action.verb == "move"
                and step.action.slots.get("object") in step.state.held_objects
                and step.feedback.success
                and target_match
                and goal_destination
                and move_destination == goal_destination
            ):
                cid = f"precondition:holding_target_before_goal_move:{family}"
                candidate = PromotionCandidate(
                    candidate_id=cid,
                    candidate_type=CandidateType.PRECONDITION,
                    summary="Holding the target object enables move(target_object, goal_destination).",
                    structure={
                        "precondition": {"holding_target_object": True},
                        "action": "move(target_object,goal_destination)",
                        "task_family": family,
                        "pattern_kind": "precondition",
                    },
                )
                candidate.observe(
                    scene_id=episode.scene_id,
                    episode_id=episode_graph.episode_id,
                    positive=True,
                    stalled=step.step_idx in stalled_steps,
                    utility_delta=0.08,
                )
                candidates.append(candidate)

            if step.feedback.failure_label:
                abstract_action = _abstract_action_text(step.action, roles)
                cid = f"failure:{family}:{step.action.verb}:{step.feedback.failure_label}"
                candidate = PromotionCandidate(
                    candidate_id=cid,
                    candidate_type=CandidateType.FAILURE,
                    summary=f"{abstract_action} fails under {step.feedback.failure_label}.",
                    structure={
                        "action": abstract_action,
                        "failure_label": step.feedback.failure_label,
                        "task_family": family,
                        "pattern_kind": "failure",
                    },
                )
                candidate.observe(
                    scene_id=episode.scene_id,
                    episode_id=episode_graph.episode_id,
                    positive=True,
                    utility_delta=0.03,
                )
                candidates.append(candidate)

        for prev_step, next_step in zip(episode.steps, episode.steps[1:]):
            if prev_step.feedback.failure_label and next_step.feedback.success:
                abstract_repair = _abstract_action_text(next_step.action, roles)
                if family.startswith("pddl") and abstract_repair in {
                    "check_valid_actions",
                    "check valid actions",
                    "check valid action",
                    "look",
                    "look around",
                }:
                    continue
                cid = f"repair:{family}:{prev_step.feedback.failure_label}:{next_step.action.verb}"
                candidate = PromotionCandidate(
                    candidate_id=cid,
                    candidate_type=CandidateType.REPAIR,
                    summary=f"After {prev_step.feedback.failure_label}, try {abstract_repair}.",
                    structure={
                        "failure_label": prev_step.feedback.failure_label,
                        "repair_action": abstract_repair,
                        "task_family": family,
                        "pattern_kind": "repair",
                    },
                )
                candidate.observe(
                    scene_id=episode.scene_id,
                    episode_id=episode_graph.episode_id,
                    positive=True,
                    utility_delta=0.06,
                )
                candidates.append(candidate)

        if episode_success and len(episode.steps) >= 2:
            suffix = episode.steps[-min(3, len(episode.steps)) :]
            closure_actions = tuple(_abstract_action_text(step.action, roles) for step in suffix)
            executable_verbs = {"go", "open", "close", "take", "put", "move", "clean", "cool", "heat", "use", "examine"}
            executable_closure_actions = tuple(
                action
                for action in closure_actions
                if action
                and not action.lower().startswith("think:")
                and action.split("(", 1)[0].strip().lower() in executable_verbs
            )
            if len(executable_closure_actions) >= 2:
                cid = f"closure:{family}:{'->'.join(executable_closure_actions)}"
                candidate = PromotionCandidate(
                    candidate_id=cid,
                    candidate_type=CandidateType.WORKFLOW,
                    summary=f"Closure pattern: {' -> '.join(executable_closure_actions)}",
                    structure={
                        "workflow": executable_closure_actions,
                        "task_family": family,
                        "pattern_kind": "closure",
                    },
                )
                candidate.observe(
                    scene_id=episode.scene_id,
                    episode_id=episode_graph.episode_id,
                    positive=True,
                    utility_delta=0.18,
                )
                candidates.append(candidate)

        if stalled_steps:
            stalled_actions = tuple(
                _abstract_action_text(step.action, roles) for step in episode.steps if step.step_idx in stalled_steps
            )
            if stalled_actions:
                cid = f"anti:{family}:{'->'.join(stalled_actions[:3])}"
                candidate = PromotionCandidate(
                    candidate_id=cid,
                    candidate_type=CandidateType.FAILURE,
                    summary=f"Anti-pattern: repeated {' -> '.join(stalled_actions[:3])} leads to stall.",
                    structure={
                        "anti_pattern": stalled_actions[:3],
                        "task_family": family,
                        "pattern_kind": "anti_pattern",
                    },
                )
                candidate.observe(
                    scene_id=episode.scene_id,
                    episode_id=episode_graph.episode_id,
                    positive=False,
                    stalled=True,
                    utility_delta=0.0,
                )
                candidates.append(candidate)

        return candidates

    def _induce_generic_candidates(
        self,
        episode_graph: EpisodeGraph,
        episode: EpisodeRecord,
        *,
        episode_success: bool,
    ) -> list[PromotionCandidate]:
        candidates: list[PromotionCandidate] = []
        family = _task_family(episode)
        domain = _episode_domain(episode)
        stalled_steps = {
            step.step_idx
            for step in episode.steps
            if not step.feedback.failure_label and not step.feedback.state_delta and not step.feedback.done
        }
        stages = [str(step.subgoal or step.next_state.workflow_stage or "") for step in episode.steps]
        workflow = tuple(dict.fromkeys(stage for stage in stages if stage))
        if len(workflow) >= 2:
            candidate = PromotionCandidate(
                candidate_id=f"workflow:{domain}:{family}:{'->'.join(workflow)}",
                candidate_type=CandidateType.WORKFLOW,
                summary=f"Workflow pattern ({domain}): {' -> '.join(workflow)}",
                structure={
                    "workflow": workflow,
                    "task_family": family,
                    "domain": domain,
                    "pattern_kind": "workflow",
                },
            )
            candidate.observe(
                scene_id=episode.scene_id,
                episode_id=episode_graph.episode_id,
                positive=episode_success,
                utility_delta=0.12 if episode_success else 0.03,
            )
            candidates.append(candidate)

        for step in episode.steps:
            action = _generic_action_text(step)
            surface_action = _generic_action_text(step, prefer_surface=True)
            progress_state = str(step.state.workflow_stage or "unknown")
            next_state = str(step.next_state.workflow_stage or progress_state)
            if step.feedback.success and not step.feedback.failure_label:
                cid = f"precondition:{domain}:{family}:{progress_state}:{_safe_pattern_id(action)}"
                candidate = PromotionCandidate(
                    candidate_id=cid,
                    candidate_type=CandidateType.PRECONDITION,
                    summary=f"Within {progress_state}, action {surface_action} can advance {domain} progress.",
                    structure={
                        "action": action,
                        "surface_action": surface_action,
                        "from": progress_state,
                        "to": next_state,
                        "task_family": family,
                        "domain": domain,
                        "pattern_kind": "precondition",
                    },
                )
                candidate.observe(
                    scene_id=episode.scene_id,
                    episode_id=episode_graph.episode_id,
                    positive=True,
                    stalled=step.step_idx in stalled_steps,
                    utility_delta=0.06,
                )
                candidates.append(candidate)

            if step.feedback.failure_label:
                cid = f"failure:{domain}:{family}:{_safe_pattern_id(action)}:{step.feedback.failure_label}"
                candidate = PromotionCandidate(
                    candidate_id=cid,
                    candidate_type=CandidateType.FAILURE,
                    summary=f"Failure pattern ({domain}): avoid {surface_action} under {step.feedback.failure_label}.",
                    structure={
                        "action": action,
                        "surface_action": surface_action,
                        "failure_label": step.feedback.failure_label,
                        "task_family": family,
                        "domain": domain,
                        "pattern_kind": "failure",
                    },
                )
                candidate.observe(
                    scene_id=episode.scene_id,
                    episode_id=episode_graph.episode_id,
                    positive=True,
                    stalled=step.step_idx in stalled_steps,
                    utility_delta=0.04,
                )
                candidates.append(candidate)

        for prev_step, next_step in zip(episode.steps, episode.steps[1:]):
            if prev_step.feedback.failure_label and next_step.feedback.success:
                repair = _generic_action_text(next_step)
                surface_repair = _generic_action_text(next_step, prefer_surface=True)
                if family.startswith("pddl") and repair in {
                    "check_valid_actions",
                    "check valid actions",
                    "check valid action",
                    "look",
                    "look around",
                }:
                    continue
                candidate = PromotionCandidate(
                    candidate_id=f"repair:{domain}:{family}:{prev_step.feedback.failure_label}:{_safe_pattern_id(repair)}",
                    candidate_type=CandidateType.REPAIR,
                    summary=f"Repair pattern ({domain}): after {prev_step.feedback.failure_label}, try {surface_repair}.",
                    structure={
                        "failure_label": prev_step.feedback.failure_label,
                        "repair_action": repair,
                        "surface_repair_action": surface_repair,
                        "task_family": family,
                        "domain": domain,
                        "pattern_kind": "repair",
                    },
                )
                candidate.observe(
                    scene_id=episode.scene_id,
                    episode_id=episode_graph.episode_id,
                    positive=True,
                    utility_delta=0.08,
                )
                candidates.append(candidate)

        if episode_success and len(episode.steps) >= 2:
            suffix = [step for step in episode.steps[-min(4, len(episode.steps)) :] if step.feedback.success]
            closure = tuple(_generic_action_text(step) for step in suffix)
            surface_closure = tuple(_generic_action_text(step, prefer_surface=True) for step in suffix)
            if len(closure) >= 2:
                candidate = PromotionCandidate(
                    candidate_id=f"closure:{domain}:{family}:{'->'.join(_safe_pattern_id(x) for x in closure)}",
                    candidate_type=CandidateType.WORKFLOW,
                    summary=f"Closure pattern ({domain}): {' -> '.join(surface_closure)}",
                    structure={
                        "workflow": closure,
                        "surface_workflow": surface_closure,
                        "task_family": family,
                        "domain": domain,
                        "pattern_kind": "closure",
                    },
                )
                candidate.observe(
                    scene_id=episode.scene_id,
                    episode_id=episode_graph.episode_id,
                    positive=True,
                    utility_delta=0.16,
                )
                candidates.append(candidate)
        return candidates

    def _induce_rules(
        self,
        episode_graph: EpisodeGraph,
        episode: EpisodeRecord,
        *,
        episode_success: bool,
    ) -> list[MemoryRule]:
        domain = _episode_domain(episode)
        if domain == "fever":
            return []
        if domain != "alfworld":
            return self._induce_generic_rules(episode_graph, episode, episode_success=episode_success)
        adapter = ALFWorldAdapter(scene_label=episode.scene_id)
        family = adapter.infer_task_family(episode.goal)
        goal_roles = adapter.goal_slots(episode.goal)
        goal_arity = adapter.required_goal_count(episode.goal)
        roles = _task_roles(episode)
        beliefs = _replay_episode_beliefs(episode)
        rules: dict[str, MemoryRule] = {}

        def upsert_rule(
            *,
            rule_type: RuleType,
            summary: str,
            progress_state: str,
            condition: dict,
            effect: dict,
            success: bool,
            stalled: bool = False,
            utility_delta: float = 0.0,
        ) -> None:
            rid = f"{rule_type.value}:{family}:{progress_state}:{_condition_signature(condition)}:{_condition_signature(effect)}"
            rule = rules.get(rid)
            if rule is None:
                rule = MemoryRule(
                    rule_id=rid,
                    rule_type=rule_type,
                    summary=summary,
                    task_family=family,
                    goal_arity=goal_arity,
                    progress_state=progress_state,
                    goal_roles=dict(goal_roles),
                    condition=dict(condition),
                    effect=dict(effect),
                )
                rules[rid] = rule
            rule.observe(
                scene_id=episode.scene_id,
                episode_id=episode_graph.episode_id,
                success=success,
                stalled=stalled,
                utility_delta=utility_delta,
            )

        stalled_steps = {
            step.step_idx
            for step in episode.steps
            if not step.feedback.failure_label and not step.feedback.state_delta and not step.feedback.done
        }

        for idx, step in enumerate(episode.steps):
            belief = beliefs[idx]
            progress = belief.progress
            next_progress = belief.next_progress
            progress_state = str(progress["progress_state"])
            next_progress_state = str(next_progress["progress_state"])
            abstract_action = _abstract_action_text(step.action, roles)
            action_role = _normalize_entity(
                step.action.slots.get("object", "")
                or step.action.slots.get("target", "")
                or step.action.slots.get("destination", "")
                or step.action.slots.get("container", ""),
                roles,
            )
            target_object = str(goal_roles.get("object", ""))
            target_base = re.sub(r"_\d+$", "", target_object)
            raw_object_slot = str(step.action.slots.get("object", ""))
            raw_object_base = re.sub(r"_\d+$", "", raw_object_slot.lower())
            action_condition = {
                "action_family": step.action.family.value,
                "location_role": _location_signature(step.state.location, roles),
                "holding_target": progress["held_relevant_count"] > 0,
                "goal_visible": progress["goal_object_matches_visible"],
                "action_role": action_role,
            }
            action_effect = {"prefer_action": abstract_action, "action_role": action_role}
            aligned_action = _goal_aligned_action(step.action, progress_state=progress_state, roles=roles)
            goal_advancing_step = _step_advances_goal(
                step,
                progress,
                next_progress,
                revisit_count=belief.revisit_count,
            )

            if step.action.verb == "take" and target_base and raw_object_base and raw_object_base != target_base:
                upsert_rule(
                    rule_type=RuleType.BLOCKED,
                    summary=f"Blocked: during {progress_state}, avoid {abstract_action} when searching for {target_base}.",
                    progress_state=progress_state,
                    condition={
                        "holding_target": progress["held_relevant_count"] > 0,
                        "target_object": target_base,
                    },
                    effect={"block_action": abstract_action, "action_role": "non_target_object"},
                    success=False,
                    stalled=step.step_idx in stalled_steps,
                    utility_delta=0.0,
                )

            if step.feedback.success and aligned_action:
                allow_precondition = True
                if step.action.verb == "go" and progress_state not in {"carry_target", "finalize"}:
                    allow_precondition = False
                if step.action.verb in {"go", "look"} and not goal_advancing_step:
                    allow_precondition = False
                if step.action.verb in {"open", "examine"} and belief.revisit_count > 1 and not goal_advancing_step:
                    allow_precondition = False

                if allow_precondition:
                    upsert_rule(
                        rule_type=RuleType.PRECONDITION,
                        summary=f"Precondition: when {progress_state}, prefer {abstract_action}.",
                        progress_state=progress_state,
                        condition=action_condition,
                        effect=action_effect,
                        success=True,
                        stalled=step.step_idx in stalled_steps,
                        utility_delta=0.08,
                    )

                if (next_progress_state != progress_state or goal_advancing_step) and not (
                    step.action.verb == "go" and progress_state not in {"carry_target", "finalize"}
                ):
                    upsert_rule(
                        rule_type=RuleType.WORKFLOW,
                        summary=(
                            f"Workflow: {progress_state} -> {next_progress_state} via {step.action.verb}."
                            if next_progress_state != progress_state
                            else f"Workflow: within {progress_state}, {step.action.verb} advances the search."
                        ),
                        progress_state=progress_state,
                        condition={
                            "from": progress_state,
                            "holding_target": progress["held_relevant_count"] > 0,
                            "remaining": progress["remaining_relevant_count"],
                        },
                        effect={
                            "to": next_progress_state,
                            "via": step.action.verb,
                            "prefer_action": abstract_action,
                        },
                        success=True,
                        utility_delta=0.12 if next_progress_state != progress_state else 0.06,
                    )

                if (
                    goal_arity == 1
                    and progress_state in {"carry_target", "finalize"}
                    and step.action.verb == "move"
                    and goal_roles.get("destination")
                    and step.action.slots.get("destination") == goal_roles.get("destination")
                ):
                    upsert_rule(
                        rule_type=RuleType.CLOSURE,
                        summary=f"Closure: when carrying the target, prefer {abstract_action}.",
                        progress_state=progress_state,
                        condition={
                            "holding_target": True,
                            "remaining": progress["remaining_relevant_count"],
                        },
                        effect={"prefer_action": abstract_action, "action_role": "goal_destination"},
                        success=True,
                        utility_delta=0.18,
                    )
            else:
                upsert_rule(
                    rule_type=RuleType.BLOCKED,
                    summary=f"Blocked: avoid {abstract_action} under {step.feedback.failure_label or 'current context'}.",
                    progress_state=progress_state,
                    condition={
                        "failure_label": step.feedback.failure_label or "",
                        "holding_target": progress["held_relevant_count"] > 0,
                    },
                    effect={"block_action": abstract_action, "action_role": action_role},
                    success=False,
                    stalled=step.step_idx in stalled_steps,
                    utility_delta=0.0,
                )

            if (
                goal_arity > 1
                and progress_state == "search_second"
                and progress["held_relevant_count"] > 0
                and step.action.verb == "move"
                and goal_roles.get("destination")
                and step.action.slots.get("destination") != goal_roles.get("destination")
            ):
                upsert_rule(
                    rule_type=RuleType.BLOCKED,
                    summary="Blocked: while searching for the remaining target, do not place the held target on a non-goal destination.",
                    progress_state=progress_state,
                    condition={
                        "holding_target": True,
                        "remaining": progress["remaining_relevant_count"],
                    },
                    effect={"block_action": abstract_action, "action_role": "non_goal_destination"},
                    success=False,
                    utility_delta=0.0,
                )

            if idx > 0:
                prev_step = episode.steps[idx - 1]
                prev_progress = beliefs[idx - 1].progress
                if prev_step.feedback.failure_label and step.feedback.success and goal_advancing_step:
                    upsert_rule(
                        rule_type=RuleType.REPAIR,
                        summary=f"Repair: after {prev_step.feedback.failure_label}, try {abstract_action}.",
                        progress_state=str(prev_progress["progress_state"]),
                        condition={
                            "failure_label": prev_step.feedback.failure_label,
                            "holding_target": prev_progress["held_relevant_count"] > 0,
                        },
                        effect={"prefer_action": abstract_action, "action_role": action_role},
                        success=True,
                        utility_delta=0.1,
                    )

        return list(rules.values())

    def _induce_generic_rules(
        self,
        episode_graph: EpisodeGraph,
        episode: EpisodeRecord,
        *,
        episode_success: bool,
    ) -> list[MemoryRule]:
        family = _task_family(episode)
        domain = _episode_domain(episode)
        goal_arity = 1
        rules: dict[str, MemoryRule] = {}
        stalled_steps = {
            step.step_idx
            for step in episode.steps
            if not step.feedback.failure_label and not step.feedback.state_delta and not step.feedback.done
        }

        def upsert_rule(
            *,
            rule_type: RuleType,
            summary: str,
            progress_state: str,
            condition: dict,
            effect: dict,
            success: bool,
            stalled: bool = False,
            utility_delta: float = 0.0,
        ) -> None:
            rid = f"{rule_type.value}:{domain}:{family}:{progress_state}:{_condition_signature(condition)}:{_condition_signature(effect)}"
            rule = rules.get(rid)
            if rule is None:
                rule = MemoryRule(
                    rule_id=rid,
                    rule_type=rule_type,
                    summary=summary,
                    task_family=family,
                    goal_arity=goal_arity,
                    progress_state=progress_state,
                    goal_roles={"object": domain},
                    condition=dict(condition),
                    effect=dict(effect),
                )
                rules[rid] = rule
            rule.observe(
                scene_id=episode.scene_id,
                episode_id=episode_graph.episode_id,
                success=success,
                stalled=stalled,
                utility_delta=utility_delta,
            )

        for idx, step in enumerate(episode.steps):
            progress_state = str(step.state.workflow_stage or "unknown")
            next_progress_state = str(step.next_state.workflow_stage or progress_state)
            action = _generic_action_text(step)
            surface_action = _generic_action_text(step, prefer_surface=True)
            condition = {
                "domain": domain,
                "action_family": step.action.family.value,
                "progress_state": progress_state,
                "verb": step.action.verb,
            }
            if step.feedback.success and not step.feedback.failure_label:
                upsert_rule(
                    rule_type=RuleType.PRECONDITION,
                    summary=f"Precondition ({domain}): during {progress_state}, prefer {surface_action}.",
                    progress_state=progress_state,
                    condition=condition,
                    effect={
                        "prefer_action": action,
                        "surface_action": surface_action,
                        "to": next_progress_state,
                    },
                    success=True,
                    stalled=step.step_idx in stalled_steps,
                    utility_delta=0.07,
                )
                if next_progress_state != progress_state or step.feedback.done or step.feedback.state_delta:
                    upsert_rule(
                        rule_type=RuleType.WORKFLOW,
                        summary=f"Workflow ({domain}): {progress_state} -> {next_progress_state} via {surface_action}.",
                        progress_state=progress_state,
                        condition={"domain": domain, "from": progress_state, "verb": step.action.verb},
                        effect={
                            "to": next_progress_state,
                            "via": step.action.verb,
                            "prefer_action": action,
                            "surface_action": surface_action,
                        },
                        success=True,
                        utility_delta=0.12,
                    )
                if step.feedback.done:
                    upsert_rule(
                        rule_type=RuleType.CLOSURE,
                        summary=f"Closure ({domain}): {surface_action} finishes the current trajectory.",
                        progress_state=progress_state,
                        condition={"domain": domain, "from": progress_state},
                        effect={"prefer_action": action, "surface_action": surface_action, "to": "done"},
                        success=True,
                        utility_delta=0.16,
                    )
            else:
                failure = step.feedback.failure_label or "no_progress"
                upsert_rule(
                    rule_type=RuleType.BLOCKED,
                    summary=f"Blocked ({domain}): avoid {surface_action} under {failure}.",
                    progress_state=progress_state,
                    condition={"domain": domain, "failure_label": failure, "from": progress_state},
                    effect={"block_action": action, "surface_action": surface_action},
                    success=False,
                    stalled=step.step_idx in stalled_steps,
                )

            if idx > 0:
                prev_step = episode.steps[idx - 1]
                if prev_step.feedback.failure_label and step.feedback.success:
                    upsert_rule(
                        rule_type=RuleType.REPAIR,
                        summary=f"Repair ({domain}): after {prev_step.feedback.failure_label}, try {surface_action}.",
                        progress_state=str(prev_step.state.workflow_stage or "unknown"),
                        condition={"domain": domain, "failure_label": prev_step.feedback.failure_label},
                        effect={"prefer_action": action, "surface_action": surface_action},
                        success=True,
                        utility_delta=0.1,
                    )
        return list(rules.values())

    def _induce_artifacts(
        self,
        episode_graph: EpisodeGraph,
        episode: EpisodeRecord,
        *,
        candidates: list[PromotionCandidate],
        rules: list[MemoryRule],
        episode_success: bool,
    ) -> list[MemoryArtifact]:
        if _episode_domain(episode) != "alfworld":
            return self._induce_generic_artifacts(
                episode_graph,
                episode,
                candidates=candidates,
                rules=rules,
                episode_success=episode_success,
            )
        adapter = ALFWorldAdapter(scene_label=episode.scene_id)
        family = adapter.infer_task_family(episode.goal)
        goal_roles = adapter.goal_slots(episode.goal)
        goal_arity = adapter.required_goal_count(episode.goal)
        roles = _task_roles(episode)
        beliefs = _replay_episode_beliefs(episode)
        artifacts: dict[str, MemoryArtifact] = {}

        node_ids_by_signature = {node.signature: node.node_id for node in episode_graph.nodes.values()}
        edge_refs = {edge.signature for edge in episode_graph.edges}
        stalled_steps = {
            step.step_idx
            for step in episode.steps
            if not step.feedback.failure_label and not step.feedback.state_delta and not step.feedback.done
        }

        def upsert_artifact(
            *,
            kind: ArtifactKind,
            summary: str,
            anchor: dict,
            payload: dict,
            success: bool,
            stalled: bool = False,
            utility_delta: float = 0.0,
        ) -> None:
            aid = f"{kind.value}:{_condition_signature(anchor)}:{_condition_signature(payload)}"
            artifact = artifacts.get(aid)
            if artifact is None:
                artifact = MemoryArtifact(
                    artifact_id=aid,
                    kind=kind,
                    summary=summary,
                    anchor=dict(anchor),
                    payload=dict(payload),
                )
                artifacts[aid] = artifact
            artifact.observe(
                scene_id=episode.scene_id,
                episode_id=episode_graph.episode_id,
                success=success,
                stalled=stalled,
                utility_delta=utility_delta,
            )

        for candidate in candidates:
            if candidate.positive < max(1, candidate.negative):
                continue
            if candidate.candidate_type == CandidateType.PRECONDITION:
                continue
            pattern_kind = str(candidate.structure.get("pattern_kind", candidate.candidate_type.value))
            if candidate.candidate_type == CandidateType.FAILURE and pattern_kind != "anti_pattern":
                continue
            action_patterns = list(_candidate_action_patterns(candidate))
            upsert_artifact(
                kind=ArtifactKind.PROTOTYPE,
                summary=f"Prototype: {candidate.summary}",
                anchor={
                    "task_family": family,
                    "goal_arity": goal_arity,
                    "pattern_kind": pattern_kind,
                    "goal_signature": f"{goal_roles.get('object', '')}->{goal_roles.get('destination', '')}",
                },
                payload={
                    "source": "candidate",
                    "pattern_kind": pattern_kind,
                    "structure": dict(candidate.structure),
                    "action_patterns": action_patterns,
                },
                success=True,
                utility_delta=max(candidate.utility, 0.02),
            )

        for rule in rules:
            upsert_artifact(
                kind=ArtifactKind.RULE,
                summary=rule.summary,
                anchor={
                    "task_family": rule.task_family,
                    "goal_arity": rule.goal_arity,
                    "progress_state": rule.progress_state,
                    "goal_signature": f"{rule.goal_roles.get('object', '')}->{rule.goal_roles.get('destination', '')}",
                    "rule_type": rule.rule_type.value,
                },
                payload={
                    "source": "rule",
                    "rule_type": rule.rule_type.value,
                    "condition": dict(rule.condition),
                    "effect": dict(rule.effect),
                    "action_patterns": list(_rule_action_patterns(rule)),
                },
                success=rule.stats.success >= rule.stats.failure,
                stalled=rule.stats.stalled > 0,
                utility_delta=max(rule.stats.utility_avg, 0.0),
            )

        for idx, step in enumerate(episode.steps):
            belief = beliefs[idx]
            progress = belief.progress
            next_progress = belief.next_progress
            progress_state = str(progress["progress_state"])
            next_progress_state = str(next_progress["progress_state"])
            abstract_action = _abstract_action_text(step.action, roles)
            action_role = _normalize_entity(
                step.action.slots.get("object", "")
                or step.action.slots.get("target", "")
                or step.action.slots.get("destination", "")
                or step.action.slots.get("container", ""),
                roles,
            )
            graph_refs = [
                signature
                for signature in (
                    f"state:{step.state.signature}|temporal|action:{step.action.canonical_str}",
                    f"action:{step.action.canonical_str}|temporal|state:{step.next_state.signature}",
                )
                if signature in edge_refs
            ]
            aligned_action = _goal_aligned_action(step.action, progress_state=progress_state, roles=roles)
            goal_advancing_step = _step_advances_goal(
                step,
                progress,
                next_progress,
                revisit_count=belief.revisit_count,
            )

            if (
                step.feedback.success
                and aligned_action
                and (next_progress_state != progress_state or goal_advancing_step)
                and not (step.action.verb == "go" and progress_state not in {"carry_target", "finalize"})
            ):
                upsert_artifact(
                    kind=ArtifactKind.PROTOTYPE,
                    summary=(
                        f"Prototype: {progress_state} -> {next_progress_state} via {step.action.verb}."
                        if next_progress_state != progress_state
                        else f"Prototype: within {progress_state}, {step.action.verb} supports progress."
                    ),
                    anchor={
                        "task_family": family,
                        "goal_arity": goal_arity,
                        "progress_state": progress_state,
                        "goal_signature": f"{goal_roles.get('object', '')}->{goal_roles.get('destination', '')}",
                        "action_family": step.action.family.value,
                        "role_focus": action_role,
                    },
                    payload={
                        "source": "episode_graph",
                        "transition": {"from": progress_state, "to": next_progress_state, "via": step.action.verb},
                        "action_patterns": [abstract_action],
                        "graph_refs": graph_refs,
                        "state_ref": node_ids_by_signature.get(step.state.signature, ""),
                        "next_state_ref": node_ids_by_signature.get(step.next_state.signature, ""),
                    },
                    success=True,
                    stalled=step.step_idx in stalled_steps,
                    utility_delta=0.08 if next_progress_state == progress_state else 0.14,
                )

            diagnosis = ""
            if step.feedback.failure_label:
                diagnosis = step.feedback.failure_label
            elif step.step_idx in stalled_steps and (
                belief.revisit_count > 0 or step.action.verb in {"look", "examine", "open"}
            ):
                diagnosis = "search_stall"
            else:
                target_object = str(goal_roles.get("object", ""))
                target_base = re.sub(r"_\d+$", "", target_object)
                raw_object = str(step.action.slots.get("object", ""))
                raw_object_base = re.sub(r"_\d+$", "", raw_object.lower())
                if step.action.verb == "take" and target_base and raw_object_base and raw_object_base != target_base:
                    diagnosis = "wrong_target_object"
                elif (
                    step.action.verb == "go"
                    and step.action.slots.get("target") == goal_roles.get("destination")
                    and progress_state not in {"carry_target", "finalize"}
                    and progress["held_relevant_count"] <= 0
                ):
                    diagnosis = "premature_destination_focus"
                elif (
                    goal_arity > 1
                    and progress_state == "search_second"
                    and progress["held_relevant_count"] > 0
                    and step.action.verb == "move"
                    and goal_roles.get("destination")
                    and step.action.slots.get("destination") != goal_roles.get("destination")
                ):
                    diagnosis = "premature_non_goal_place"

            if not diagnosis:
                continue

            repair_patterns: list[str] = []
            for future_offset, future_step in enumerate(episode.steps[idx + 1 : idx + 5], start=idx + 1):
                future_belief = beliefs[future_offset]
                if future_step.feedback.success and _step_advances_goal(
                    future_step,
                    future_belief.progress,
                    future_belief.next_progress,
                    revisit_count=future_belief.revisit_count,
                ):
                    repair_patterns.append(_abstract_action_text(future_step.action, roles))
                    break
            if not repair_patterns:
                continue

            upsert_artifact(
                kind=ArtifactKind.REFLECTION,
                summary=f"Reflection: during {progress_state}, {diagnosis.replace('_', ' ')} tends to derail progress.",
                anchor={
                    "task_family": family,
                    "goal_arity": goal_arity,
                    "progress_state": progress_state,
                    "goal_signature": f"{goal_roles.get('object', '')}->{goal_roles.get('destination', '')}",
                    "failure_signature": diagnosis,
                },
                payload={
                    "source": "episode_graph",
                    "diagnosis": diagnosis,
                    "trigger_action": abstract_action,
                    "avoid_patterns": [abstract_action],
                    "repair_patterns": repair_patterns,
                    "graph_refs": graph_refs,
                    "holding_target": bool(progress["held_relevant_count"] > 0),
                },
                success=False,
                stalled=step.step_idx in stalled_steps,
                utility_delta=0.12 if repair_patterns else 0.04,
            )

        if episode_success:
            plan_template = _induce_plan_template(
                episode,
                beliefs=beliefs,
                roles=roles,
                goal_roles=goal_roles,
                goal_arity=goal_arity,
            )
            if plan_template is not None:
                summary, plan_steps, action_patterns = plan_template
                upsert_artifact(
                    kind=ArtifactKind.PROTOTYPE,
                    summary=summary,
                    anchor={
                        "task_family": family,
                        "goal_arity": goal_arity,
                        "goal_signature": f"{goal_roles.get('object', '')}->{goal_roles.get('destination', '')}",
                        "artifact_role": "plan",
                    },
                    payload={
                        "source": "plan_template",
                        "pattern_kind": "plan",
                        "plan_steps": plan_steps,
                        "action_patterns": action_patterns,
                    },
                    success=True,
                    utility_delta=0.16,
                )

            layout_id = str(episode.metadata.get("layout_id", ""))
            target_base = re.sub(r"_\d+$", "", str(goal_roles.get("object", "")).lower())
            for idx, step in enumerate(episode.steps):
                belief = beliefs[idx]
                next_progress = belief.next_progress
                progress = belief.progress
                if (
                    step.feedback.success
                    and step.action.verb == "take"
                    and target_base
                    and _is_target_object(step.action, target_base)
                ):
                    source_raw = str(step.action.slots.get("source", "")).lower()
                    source_instance = source_raw
                    source_base = re.sub(r"_\d+$", "", source_instance)
                    source_role = _normalize_entity(source_raw, roles)
                    upsert_artifact(
                        kind=ArtifactKind.PROTOTYPE,
                        summary=f"Scene relation: in this layout, target_object was found at {source_instance} ({source_role}).",
                        anchor={
                            "task_family": family,
                            "goal_arity": goal_arity,
                            "goal_signature": f"{goal_roles.get('object', '')}->{goal_roles.get('destination', '')}",
                            "artifact_role": "scene_relation",
                            "layout_id": layout_id,
                        },
                        payload={
                            "source": "scene_relation",
                            "pattern_kind": "scene_relation",
                            "relation_kind": "object_location_prior",
                            "object_role": "target_object",
                            "source_role": source_role,
                            "source_base": source_base,
                            "source_instance": source_instance,
                            "evidence_type": "take_success",
                            "action_patterns": _relation_action_patterns(source_instance, source_base, source_role),
                        },
                        success=True,
                        utility_delta=0.14,
                    )
                    if source_base and not source_base.isdigit():
                        upsert_artifact(
                            kind=ArtifactKind.PROTOTYPE,
                            summary=(
                                "Source type prior: for this task family and target_object, "
                                f"source type {source_base} has helped find the target."
                            ),
                            anchor={
                                "task_family": family,
                                "goal_arity": goal_arity,
                                "goal_signature": f"{goal_roles.get('object', '')}->{goal_roles.get('destination', '')}",
                                "artifact_role": "source_type_prior",
                            },
                            payload={
                                "source": "source_type_prior",
                                "pattern_kind": "source_type_prior",
                                "relation_kind": "source_type_prior",
                                "object_role": "target_object",
                                "source_role": source_role,
                                "source_base": source_base,
                                "goal_object": target_base,
                                "action_patterns": _relation_action_patterns("", source_base, source_role),
                            },
                            success=True,
                            utility_delta=0.12,
                        )

                search_ref = str(belief.search_ref or "").lower()
                if (
                    step.feedback.success
                    and search_ref
                    and step.action.verb in {"open", "examine", "look"}
                    and not bool(next_progress.get("goal_object_matches_visible"))
                    and int(progress.get("held_relevant_count", 0)) <= 0
                ):
                    source_instance = search_ref
                    source_base = re.sub(r"_\d+$", "", source_instance)
                    source_role = _normalize_entity(search_ref, roles)
                    upsert_artifact(
                        kind=ArtifactKind.PROTOTYPE,
                        summary=f"Scene relation: in this layout, target_object was not found after searching {source_instance}.",
                        anchor={
                            "task_family": family,
                            "goal_arity": goal_arity,
                            "goal_signature": f"{goal_roles.get('object', '')}->{goal_roles.get('destination', '')}",
                            "artifact_role": "scene_relation",
                            "layout_id": layout_id,
                        },
                        payload={
                            "source": "scene_relation",
                            "pattern_kind": "scene_relation",
                            "relation_kind": "searched_empty",
                            "object_role": "target_object",
                            "source_role": source_role,
                            "source_base": source_base,
                            "source_instance": source_instance,
                            "evidence_type": "searched_empty",
                            "action_patterns": _relation_action_patterns(source_instance, source_base, source_role),
                        },
                        success=True,
                        utility_delta=0.08,
                    )
                    if source_base and not source_base.isdigit():
                        upsert_artifact(
                            kind=ArtifactKind.PROTOTYPE,
                            summary=(
                                "Source type prior: for this task family and target_object, "
                                f"source type {source_base} has been searched without finding the target."
                            ),
                            anchor={
                                "task_family": family,
                                "goal_arity": goal_arity,
                                "goal_signature": f"{goal_roles.get('object', '')}->{goal_roles.get('destination', '')}",
                                "artifact_role": "source_type_prior",
                            },
                            payload={
                                "source": "source_type_prior",
                                "pattern_kind": "source_type_prior",
                                "relation_kind": "source_type_prior",
                                "object_role": "target_object",
                                "source_role": source_role,
                                "source_base": source_base,
                                "goal_object": target_base,
                                "action_patterns": _relation_action_patterns("", source_base, source_role),
                            },
                            success=False,
                            utility_delta=0.02,
                        )

        return list(artifacts.values())

    def _induce_generic_artifacts(
        self,
        episode_graph: EpisodeGraph,
        episode: EpisodeRecord,
        *,
        candidates: list[PromotionCandidate],
        rules: list[MemoryRule],
        episode_success: bool,
    ) -> list[MemoryArtifact]:
        family = _task_family(episode)
        domain = _episode_domain(episode)
        if domain == "fever":
            return self._induce_fever_artifacts(
                episode_graph,
                episode,
                episode_success=episode_success,
            )
        artifacts: dict[str, MemoryArtifact] = {}
        node_ids_by_signature = {node.signature: node.node_id for node in episode_graph.nodes.values()}
        edge_refs = {edge.signature for edge in episode_graph.edges}
        stalled_steps = {
            step.step_idx
            for step in episode.steps
            if not step.feedback.failure_label and not step.feedback.state_delta and not step.feedback.done
        }

        def upsert_artifact(
            *,
            kind: ArtifactKind,
            summary: str,
            anchor: dict,
            payload: dict,
            success: bool,
            stalled: bool = False,
            utility_delta: float = 0.0,
        ) -> None:
            aid = f"{kind.value}:{_condition_signature(anchor)}:{_condition_signature(payload)}"
            artifact = artifacts.get(aid)
            if artifact is None:
                artifact = MemoryArtifact(
                    artifact_id=aid,
                    kind=kind,
                    summary=summary,
                    anchor=dict(anchor),
                    payload=dict(payload),
                )
                artifacts[aid] = artifact
            artifact.observe(
                scene_id=episode.scene_id,
                episode_id=episode_graph.episode_id,
                success=success,
                stalled=stalled,
                utility_delta=utility_delta,
            )

        for candidate in candidates:
            pattern_kind = str(candidate.structure.get("pattern_kind", candidate.candidate_type.value))
            action_patterns = list(_candidate_action_patterns(candidate))
            for key in ("surface_action", "surface_repair_action"):
                value = str(candidate.structure.get(key, "") or "").strip()
                if value:
                    action_patterns.append(value)
            for value in candidate.structure.get("surface_workflow", ()) or ():
                if str(value).strip():
                    action_patterns.append(str(value).strip())
            upsert_artifact(
                kind=ArtifactKind.PROTOTYPE,
                summary=f"Prototype ({domain}): {candidate.summary}",
                anchor={
                    "task_family": family,
                    "goal_arity": 1,
                    "pattern_kind": pattern_kind,
                    "domain": domain,
                },
                payload={
                    "source": "generic_candidate",
                    "pattern_kind": pattern_kind,
                    "structure": dict(candidate.structure),
                    "action_patterns": list(dict.fromkeys(action_patterns)),
                },
                success=candidate.positive >= max(1, candidate.negative),
                stalled=candidate.stalled > 0,
                utility_delta=max(candidate.utility, 0.02),
            )

        for rule in rules:
            patterns = list(_rule_action_patterns(rule))
            surface = str(rule.effect.get("surface_action", "") or "").strip()
            if surface:
                patterns.append(surface)
            upsert_artifact(
                kind=ArtifactKind.RULE,
                summary=rule.summary,
                anchor={
                    "task_family": rule.task_family,
                    "goal_arity": rule.goal_arity,
                    "progress_state": rule.progress_state,
                    "rule_type": rule.rule_type.value,
                    "domain": domain,
                },
                payload={
                    "source": "generic_rule",
                    "pattern_kind": rule.rule_type.value,
                    "rule_type": rule.rule_type.value,
                    "condition": dict(rule.condition),
                    "effect": dict(rule.effect),
                    "action_patterns": list(dict.fromkeys(patterns)),
                },
                success=rule.stats.success >= rule.stats.failure,
                stalled=rule.stats.stalled > 0,
                utility_delta=max(rule.stats.utility_avg, 0.0),
            )

        for idx, step in enumerate(episode.steps):
            progress_state = str(step.state.workflow_stage or "unknown")
            next_progress_state = str(step.next_state.workflow_stage or progress_state)
            action = _generic_action_text(step)
            surface_action = _generic_action_text(step, prefer_surface=True)
            graph_refs = [
                signature
                for signature in (
                    f"state:{step.state.signature}|temporal|action:{step.action.canonical_str}",
                    f"action:{step.action.canonical_str}|temporal|state:{step.next_state.signature}",
                )
                if signature in edge_refs
            ]
            if step.feedback.success and not step.feedback.failure_label:
                upsert_artifact(
                    kind=ArtifactKind.PROTOTYPE,
                    summary=f"Workflow ({domain}): {progress_state} -> {next_progress_state} via {surface_action}.",
                    anchor={
                        "task_family": family,
                        "goal_arity": 1,
                        "progress_state": progress_state,
                        "artifact_role": "workflow",
                        "domain": domain,
                    },
                    payload={
                        "source": "generic_episode_graph",
                        "pattern_kind": "workflow",
                        "transition": {"from": progress_state, "to": next_progress_state, "via": step.action.verb},
                        "action_patterns": list(dict.fromkeys([action, surface_action])),
                        "graph_refs": graph_refs,
                        "state_ref": node_ids_by_signature.get(step.state.signature, ""),
                        "next_state_ref": node_ids_by_signature.get(step.next_state.signature, ""),
                    },
                    success=True,
                    stalled=step.step_idx in stalled_steps,
                    utility_delta=0.10 if next_progress_state == progress_state else 0.14,
                )

            diagnosis = step.feedback.failure_label or ("no_progress" if step.step_idx in stalled_steps else "")
            if not diagnosis:
                continue
            repair_patterns: list[str] = []
            for future_step in episode.steps[idx + 1 : idx + 5]:
                if future_step.feedback.success:
                    repair_patterns.extend(_generic_action_patterns(future_step))
                    break
            upsert_artifact(
                kind=ArtifactKind.REFLECTION,
                summary=f"Reflection ({domain}): during {progress_state}, {diagnosis.replace('_', ' ')} can derail progress.",
                anchor={
                    "task_family": family,
                    "goal_arity": 1,
                    "progress_state": progress_state,
                    "failure_signature": diagnosis,
                    "domain": domain,
                },
                payload={
                    "source": "generic_episode_graph",
                    "pattern_kind": "failure",
                    "diagnosis": diagnosis,
                    "trigger_action": action,
                    "avoid_patterns": list(dict.fromkeys([action, surface_action])),
                    "repair_patterns": repair_patterns,
                    "graph_refs": graph_refs,
                },
                success=False,
                stalled=step.step_idx in stalled_steps,
                utility_delta=0.10 if repair_patterns else 0.04,
            )
        return list(artifacts.values())

    def _induce_fever_artifacts_legacy(
        self,
        episode_graph: EpisodeGraph,
        episode: EpisodeRecord,
        *,
        episode_success: bool,
    ) -> list[MemoryArtifact]:
        """GM3-compatible FEVER memories: coarse search/lookup workflow only."""
        domain = "fever"
        family = "fever:claim_verification"
        claim = _fever_claim_text(episode)
        artifacts: dict[str, MemoryArtifact] = {}

        def upsert_artifact(
            *,
            kind: ArtifactKind,
            summary: str,
            anchor: dict,
            payload: dict,
            success: bool,
            stalled: bool = False,
            utility_delta: float = 0.0,
        ) -> None:
            aid = f"{kind.value}:{_condition_signature(anchor)}:{_condition_signature(payload)}"
            artifact = artifacts.get(aid)
            if artifact is None:
                artifact = MemoryArtifact(
                    artifact_id=aid,
                    kind=kind,
                    summary=summary,
                    anchor=dict(anchor),
                    payload=dict(payload),
                )
                artifacts[aid] = artifact
            artifact.observe(
                scene_id=episode.scene_id,
                episode_id=episode_graph.episode_id,
                success=success,
                stalled=stalled,
                utility_delta=utility_delta,
            )

        search_steps = [step for step in episode.steps if _fever_is_search(step)]
        lookup_steps = [step for step in episode.steps if _fever_is_lookup(step)]
        finish_steps = [step for step in episode.steps if _fever_is_finish(step)]
        first_search_arg = _fever_action_arg(search_steps[0]) if search_steps else ""
        first_lookup_arg = _fever_action_arg(lookup_steps[0]) if lookup_steps else ""
        relation_hint = _fever_relation_keyword_hint(
            claim,
            search_arg=first_search_arg,
            lookup_arg=first_lookup_arg,
        )
        has_successful_search = any(step.feedback.success and not step.feedback.failure_label for step in search_steps)
        has_successful_lookup = any(step.feedback.success and not step.feedback.failure_label for step in lookup_steps)

        if search_steps or episode_success:
            upsert_artifact(
                kind=ArtifactKind.PROTOTYPE,
                summary=(
                    "FEVER evidence workflow: start with Search on the claim's primary entity "
                    "before any Finish label."
                ),
                anchor={
                    "task_family": family,
                    "goal_arity": 1,
                    "progress_state": "need_search",
                    "artifact_role": "fever_evidence_search",
                    "domain": domain,
                },
                payload={
                    "source": "fever_episode_graph",
                    "pattern_kind": "workflow",
                    "fever_pattern": "evidence_search",
                    "claim_role": "primary_entity",
                    "action_patterns": [
                        "Search[claim primary entity]",
                        "search(query=claim_primary_entity)",
                    ],
                },
                success=bool(episode_success or has_successful_search),
                utility_delta=0.20 if episode_success else 0.08,
            )

        if lookup_steps:
            upsert_artifact(
                kind=ArtifactKind.PROTOTYPE,
                summary=(
                    "FEVER lookup workflow: after Search retrieves a page, use Lookup on a "
                    "relation or attribute keyword from the claim before Finish."
                ),
                anchor={
                    "task_family": family,
                    "goal_arity": 1,
                    "progress_state": "need_lookup_or_finish",
                    "artifact_role": "fever_lookup_strategy",
                    "domain": domain,
                },
                payload={
                    "source": "fever_episode_graph",
                    "pattern_kind": "workflow",
                    "fever_pattern": "lookup_relation_keyword",
                    "keyword_role": relation_hint,
                    "action_patterns": [
                        "Lookup[claim relation keyword]",
                        "lookup(keyword=claim_relation_keyword)",
                    ],
                },
                success=bool(episode_success or has_successful_lookup),
                utility_delta=0.22 if episode_success else 0.09,
            )

        if any(_fever_no_results(step) for step in episode.steps):
            upsert_artifact(
                kind=ArtifactKind.PROTOTYPE,
                summary=(
                    "FEVER recovery workflow: if Search or Lookup returns No Results, reformulate "
                    "to a shorter entity/relation keyword before giving a final label."
                ),
                anchor={
                    "task_family": family,
                    "goal_arity": 1,
                    "progress_state": "search_failed",
                    "artifact_role": "fever_no_results_recovery",
                    "domain": domain,
                },
                payload={
                    "source": "fever_episode_graph",
                    "pattern_kind": "workflow",
                    "fever_pattern": "no_results_recovery",
                    "repair_patterns": [
                        "Search[shorter claim entity]",
                        "Lookup[shorter claim relation keyword]",
                    ],
                },
                success=True,
                utility_delta=0.14,
            )

        first_finish_idx = finish_steps[0].step_idx if finish_steps else None
        first_search_idx = search_steps[0].step_idx if search_steps else None
        first_lookup_idx = lookup_steps[0].step_idx if lookup_steps else None
        premature_finish = (
            first_finish_idx is not None
            and (first_search_idx is None or first_finish_idx < first_search_idx or first_lookup_idx is None)
            and not episode_success
        )
        wrong_finish = any(step.feedback.failure_label == "wrong_finish_label" for step in finish_steps)
        if premature_finish or wrong_finish:
            upsert_artifact(
                kind=ArtifactKind.PROTOTYPE,
                summary=(
                    "FEVER failure avoidance: avoid Finish[...] before Search/Lookup evidence "
                    "settles the claim; early label guesses often fail."
                ),
                anchor={
                    "task_family": family,
                    "goal_arity": 1,
                    "progress_state": "ready_finish",
                    "artifact_role": "fever_premature_finish_failure",
                    "domain": domain,
                },
                payload={
                    "source": "fever_episode_graph",
                    "pattern_kind": "workflow",
                    "fever_pattern": "premature_finish_failure",
                    "avoid_patterns": ["Finish[unsupported label]"],
                    "repair_patterns": [
                        "Search[claim primary entity]",
                        "Lookup[claim relation keyword]",
                    ],
                },
                success=True,
                utility_delta=0.12,
            )

        return list(artifacts.values())

    def _induce_fever_artifacts(
        self,
        episode_graph: EpisodeGraph,
        episode: EpisodeRecord,
        *,
        episode_success: bool,
    ) -> list[MemoryArtifact]:
        """Build FEVER memories around evidence acquisition, not answer labels.

        FEVER transfer is different from ALFWorld object grounding: concrete
        entities and final labels rarely transfer, but claim-type evidence
        workflows do.  These artifacts avoid old labels/entities while keeping
        the reusable search/lookup strategy specific enough to route.
        """
        if _memco_fever_policy_enabled():
            return self._induce_fever_artifacts_legacy(
                episode_graph,
                episode,
                episode_success=episode_success,
            )
        domain = "fever"
        claim = _fever_claim_text(episode)
        claim_profile = _fever_claim_profile(claim)
        claim_type = str(claim_profile.get("claim_type", "") or "general_fact")
        lookup_keywords = [
            str(keyword).strip() for keyword in (claim_profile.get("lookup_keywords", []) or []) if str(keyword).strip()
        ][:5]
        query_variants = [
            str(query).strip() for query in (claim_profile.get("query_variants", []) or []) if str(query).strip()
        ][:6]
        family = f"fever:{claim_type}"
        generic_family = "fever:claim_verification"
        artifacts: dict[str, MemoryArtifact] = {}

        def upsert_artifact(
            *,
            kind: ArtifactKind,
            summary: str,
            anchor: dict,
            payload: dict,
            success: bool,
            stalled: bool = False,
            utility_delta: float = 0.0,
        ) -> None:
            aid = f"{kind.value}:{_condition_signature(anchor)}:{_condition_signature(payload)}"
            artifact = artifacts.get(aid)
            if artifact is None:
                artifact = MemoryArtifact(
                    artifact_id=aid,
                    kind=kind,
                    summary=summary,
                    anchor=dict(anchor),
                    payload=dict(payload),
                )
                artifacts[aid] = artifact
            artifact.observe(
                scene_id=episode.scene_id,
                episode_id=episode_graph.episode_id,
                success=success,
                stalled=stalled,
                utility_delta=utility_delta,
            )

        search_steps = [step for step in episode.steps if _fever_is_search(step)]
        lookup_steps = [step for step in episode.steps if _fever_is_lookup(step)]
        finish_steps = [step for step in episode.steps if _fever_is_finish(step)]
        if episode_success and search_steps and finish_steps:
            search_arg = _fever_action_arg(search_steps[0])
            lookup_arg = _fever_action_arg(lookup_steps[0]) if lookup_steps else ""
            observed_lookup = _fever_relation_keyword_hint(claim, search_arg=search_arg, lookup_arg=lookup_arg)
            route_shape = "search_lookup_finish" if lookup_steps else "search_finish"
            upsert_artifact(
                kind=ArtifactKind.PROTOTYPE,
                summary=(
                    f"FEVER content search route ({claim_type}): successful verification starts by searching "
                    f"the claim's {claim_profile.get('search_role', 'primary entity')}; "
                    + (
                        "then lookup a claim-relation keyword before finishing from evidence."
                        if lookup_steps
                        else "finish directly if the searched page already settles the claim."
                    )
                ),
                anchor={
                    "task_family": family,
                    "goal_arity": 1,
                    "progress_state": "need_search",
                    "artifact_role": "fever_content_search_route",
                    "domain": domain,
                    "claim_type": claim_type,
                },
                payload={
                    "source": "fever_episode_graph",
                    "pattern_kind": "workflow",
                    "fever_pattern": "content_search_route",
                    "claim_type": claim_type,
                    "search_role": claim_profile.get("search_role", "primary entity"),
                    "lookup_keywords": lookup_keywords,
                    "query_variants": query_variants,
                    "relation_entity": claim_profile.get("secondary_entity", ""),
                    "observed_lookup_role": observed_lookup,
                    "route_shape": route_shape,
                    "action_patterns": [
                        "Search[current claim relation/type query]",
                        "Lookup[current claim relation keyword]",
                    ],
                },
                success=True,
                utility_delta=0.16 if lookup_steps else 0.12,
            )
            evidence_step = lookup_steps[-1] if lookup_steps else search_steps[-1]
            evidence_snippet = _fever_clean_evidence_snippet(_fever_observation_text(evidence_step))
            evidence_terms = _fever_evidence_terms(
                claim=claim,
                evidence=evidence_snippet,
                search_arg=search_arg,
                lookup_arg=lookup_arg,
            )
            if evidence_snippet and len(evidence_terms) >= 2:
                source_title = search_arg or str(claim_profile.get("primary_entity", "") or "")
                upsert_artifact(
                    kind=ArtifactKind.PROTOTYPE,
                    summary=(
                        f"FEVER evidence memory ({claim_type}): a prior successful verification used source "
                        f"`{source_title}` with evidence terms {', '.join(evidence_terms[:6])}. "
                        "Treat this as candidate evidence only; verify against the current claim and do not copy any old label."
                    ),
                    anchor={
                        "task_family": family,
                        "goal_arity": 1,
                        "progress_state": "evidence_available",
                        "artifact_role": "fever_claim_evidence",
                        "domain": domain,
                        "claim_type": claim_type,
                    },
                    payload={
                        "source": "fever_episode_graph",
                        "pattern_kind": "claim_evidence",
                        "fever_pattern": "claim_evidence",
                        "claim_type": claim_type,
                        "source_title": source_title,
                        "lookup_keyword": lookup_arg,
                        "evidence_terms": evidence_terms[:12],
                        "claim_terms": _fever_memory_tokens(claim)[:16],
                        "evidence_snippet": evidence_snippet,
                        "confidence_source": "successful_episode_observation",
                        "conflict_policy": "discard_if_current_claim_or_page_conflicts; never use as label prior",
                        "label_guard": "no_prior_label_stored",
                    },
                    success=True,
                    utility_delta=0.20,
                )
        if episode_success and search_steps and finish_steps and not lookup_steps:
            upsert_artifact(
                kind=ArtifactKind.PROTOTYPE,
                summary=(
                    f"FEVER stop rule ({claim_type}): when the searched page already settles the "
                    "claim, finish from evidence instead of forcing an extra Lookup."
                ),
                anchor={
                    "task_family": family,
                    "goal_arity": 1,
                    "progress_state": "need_lookup_or_finish",
                    "artifact_role": "fever_evidence_sufficiency_stop",
                    "domain": domain,
                    "claim_type": claim_type,
                },
                payload={
                    "source": "fever_episode_graph",
                    "pattern_kind": "workflow",
                    "fever_pattern": "evidence_sufficiency_stop",
                    "claim_type": claim_type,
                    "lookup_keywords": lookup_keywords,
                    "stop_rule": "finish only when the correct subject page directly supports or refutes; otherwise run a relation query before NEI",
                    "avoid_patterns": ["Lookup[generic relation keyword]"],
                },
                success=True,
                utility_delta=0.18,
            )
            upsert_artifact(
                kind=ArtifactKind.PROTOTYPE,
                summary=(
                    "FEVER stop rule (claim_verification): when the searched page already settles "
                    "the current claim, finish from evidence instead of forcing an extra Lookup."
                ),
                anchor={
                    "task_family": generic_family,
                    "goal_arity": 1,
                    "progress_state": "need_lookup_or_finish",
                    "artifact_role": "fever_generic_evidence_sufficiency_stop",
                    "domain": domain,
                    "claim_type": "claim_verification",
                },
                payload={
                    "source": "fever_episode_graph",
                    "pattern_kind": "workflow",
                    "fever_pattern": "generic_evidence_sufficiency_stop",
                    "claim_type": "claim_verification",
                    "stop_rule": "finish only when the correct subject page directly supports or refutes; otherwise run a relation query before NEI",
                    "avoid_patterns": ["Lookup[generic relation keyword]"],
                },
                success=True,
                utility_delta=0.12,
            )

        if episode_success and any(_fever_no_results(step) for step in episode.steps):
            upsert_artifact(
                kind=ArtifactKind.PROTOTYPE,
                summary=(
                    f"FEVER recovery workflow ({claim_type}): if Search or Lookup returns No Results, "
                    "reformulate to a shorter current entity/relation keyword before giving a final label."
                ),
                anchor={
                    "task_family": family,
                    "goal_arity": 1,
                    "progress_state": "search_failed",
                    "artifact_role": "fever_no_results_recovery",
                    "domain": domain,
                    "claim_type": claim_type,
                },
                payload={
                    "source": "fever_episode_graph",
                    "pattern_kind": "workflow",
                    "fever_pattern": "claim_type_no_results_recovery",
                    "claim_type": claim_type,
                    "lookup_keywords": lookup_keywords,
                    "repair_patterns": [
                        "Search[shorter current claim entity]",
                        "Lookup[shorter current claim relation keyword]",
                    ],
                },
                success=True,
                utility_delta=0.14,
            )
            upsert_artifact(
                kind=ArtifactKind.PROTOTYPE,
                summary=(
                    "FEVER recovery workflow (claim_verification): if Search or Lookup returns No Results, "
                    "reformulate to a shorter current entity/relation keyword before giving a final label."
                ),
                anchor={
                    "task_family": generic_family,
                    "goal_arity": 1,
                    "progress_state": "search_failed",
                    "artifact_role": "fever_generic_no_results_recovery",
                    "domain": domain,
                    "claim_type": "claim_verification",
                },
                payload={
                    "source": "fever_episode_graph",
                    "pattern_kind": "workflow",
                    "fever_pattern": "generic_no_results_recovery",
                    "claim_type": "claim_verification",
                    "repair_patterns": [
                        "Search[shorter current claim entity]",
                        "Lookup[shorter current claim relation keyword]",
                    ],
                },
                success=True,
                utility_delta=0.10,
            )

        first_finish_idx = finish_steps[0].step_idx if finish_steps else None
        first_search_idx = search_steps[0].step_idx if search_steps else None
        first_lookup_idx = lookup_steps[0].step_idx if lookup_steps else None
        premature_finish = (
            first_finish_idx is not None
            and (first_search_idx is None or first_finish_idx < first_search_idx or first_lookup_idx is None)
            and not episode_success
        )
        wrong_finish = any(step.feedback.failure_label == "wrong_finish_label" for step in finish_steps)
        if premature_finish or wrong_finish:
            upsert_artifact(
                kind=ArtifactKind.PROTOTYPE,
                summary=(
                    f"FEVER failure avoidance ({claim_type}): avoid Finish[...] before Search/Lookup "
                    "evidence settles the current claim; early label guesses often fail."
                ),
                anchor={
                    "task_family": family,
                    "goal_arity": 1,
                    "progress_state": "ready_finish",
                    "artifact_role": "fever_premature_finish_failure",
                    "domain": domain,
                    "claim_type": claim_type,
                },
                payload={
                    "source": "fever_episode_graph",
                    "pattern_kind": "failure",
                    "fever_pattern": "claim_type_premature_finish_failure",
                    "claim_type": claim_type,
                    "lookup_keywords": lookup_keywords,
                    "avoid_patterns": ["Finish[unsupported label]"],
                    "repair_patterns": [
                        "Search[current claim primary entity]",
                        "Lookup[current claim relation keyword]",
                    ],
                },
                success=False,
                utility_delta=0.12,
            )

        return list(artifacts.values())


class GlobalPromoter:
    def __init__(
        self,
        score_threshold: float = 0.35,
        *,
        policy: str = "legacy",
        wilson_alpha: float = 0.05,
        wilson_threshold: float = 0.5,
        wilson_min_coverage: int | None = None,
    ) -> None:
        self.score_threshold = score_threshold
        self.policy = normalize_promotion_policy(policy)
        if not 0.0 < wilson_alpha < 0.5:
            raise ValueError("wilson_alpha must lie strictly between 0 and 0.5")
        if not 0.0 <= wilson_threshold <= 1.0:
            raise ValueError("wilson_threshold must lie between 0 and 1")
        self.wilson_alpha = float(wilson_alpha)
        self.wilson_threshold = float(wilson_threshold)
        # Retain the keyword for compatibility with older launchers, but source
        # coverage is now descriptive evidence rather than a promotion gate.
        self.wilson_min_coverage = None
        self.last_wilson_memory: GlobalGraphMemory | None = None
        self.last_promotion_report: dict[str, Any] = {}

    def _specificity_penalty(self, candidate: PromotionCandidate) -> float:
        lowered = candidate.summary.lower()
        specific_tokens = re.findall(r"\b[a-z]+_\d+\b", lowered)
        if not specific_tokens:
            return 0.0
        return min(0.24, 0.08 * len(specific_tokens))

    def _transferability_score(self, candidate: PromotionCandidate) -> float:
        support = candidate.positive + candidate.negative
        if support <= 0:
            return -1.0
        confidence = candidate.positive / support
        negative_rate = candidate.negative / support
        stall_rate = candidate.stalled / support
        scene_term = min(candidate.coverage / 3.0, 1.0)
        utility_term = candidate.utility / support
        family_term = 0.15 if candidate.structure.get("task_family") else 0.0
        abstraction_term = 0.1 if self._specificity_penalty(candidate) == 0.0 else -self._specificity_penalty(candidate)
        return (
            0.65 * confidence
            + 0.45 * scene_term
            + 0.25 * utility_term
            + family_term
            + abstraction_term
            - 0.45 * negative_rate
            - 0.55 * stall_rate
        )

    def _min_scene_coverage(self, candidate: PromotionCandidate) -> int:
        pattern_kind = str(candidate.structure.get("pattern_kind", ""))
        support = candidate.positive + candidate.negative
        abstract_enough = self._specificity_penalty(candidate) == 0.0
        if candidate.candidate_type == CandidateType.WORKFLOW and pattern_kind in {"closure", "workflow"}:
            if abstract_enough and candidate.positive >= 4 and support >= 4 and candidate.negative == 0:
                return 1
        return 2

    def _rule_min_scene_coverage(self, rule: MemoryRule) -> int:
        if rule.coverage >= 2:
            return 2
        if (
            rule.rule_type in {RuleType.PRECONDITION, RuleType.WORKFLOW, RuleType.CLOSURE}
            and rule.support >= 4
            and rule.stats.success >= 3
            and rule.stats.confidence >= 0.65
            and rule.stats.stalled <= max(1, rule.support // 3)
            and rule.specificity <= 0.12
        ):
            return 1
        return 2

    def _artifact_min_scene_coverage(self, artifact: MemoryArtifact) -> int:
        if artifact.coverage >= 2:
            return 2
        pattern_kind = str(artifact.payload.get("pattern_kind", "") or artifact.anchor.get("pattern_kind", ""))
        artifact_role = str(artifact.anchor.get("artifact_role", ""))
        if pattern_kind == "source_type_prior" or artifact_role == "source_type_prior":
            if (
                artifact.support >= 2
                and artifact.stats.success >= 1
                and artifact.stats.confidence >= 0.45
                and artifact.stats.stalled <= max(1, artifact.support // 3)
            ):
                return 1
        if (
            artifact.kind in {ArtifactKind.PROTOTYPE, ArtifactKind.RULE}
            and pattern_kind != "scene_relation"
            and artifact_role != "scene_relation"
            and artifact.support >= 4
            and artifact.stats.success >= 3
            and artifact.stats.confidence >= 0.65
            and artifact.stats.stalled <= max(1, artifact.support // 3)
            and artifact.specificity <= 0.12
        ):
            return 1
        return 2

    def _global_rule_copy(self, rule: MemoryRule) -> MemoryRule:
        condition = _abstract_global_mapping(rule.condition)
        effect = _abstract_global_mapping(rule.effect)
        return MemoryRule(
            rule_id=_global_rule_id(rule, condition, effect),
            rule_type=rule.rule_type,
            summary=rule.summary,
            task_family=rule.task_family,
            goal_arity=rule.goal_arity,
            progress_state=rule.progress_state,
            goal_roles=_abstract_goal_roles(rule.goal_roles),
            condition=condition,
            effect=effect,
        )

    def _global_artifact_copy(self, artifact: MemoryArtifact) -> MemoryArtifact | None:
        raw_pattern_kind = str(artifact.payload.get("pattern_kind", "") or artifact.anchor.get("pattern_kind", ""))
        raw_artifact_role = str(artifact.anchor.get("artifact_role", ""))
        fever_pattern = str(artifact.payload.get("fever_pattern", "") or "")
        if artifact.kind == ArtifactKind.REFLECTION:
            return None
        if raw_pattern_kind == "scene_relation" or raw_artifact_role == "scene_relation":
            return None
        if raw_pattern_kind == "source_type_prior" or raw_artifact_role == "source_type_prior":
            anchor = {
                "task_family": artifact.anchor.get("task_family", ""),
                "goal_arity": artifact.anchor.get("goal_arity", 1),
                "artifact_role": "source_type_prior",
            }
            payload = {
                "source": "source_type_prior",
                "pattern_kind": "source_type_prior",
                "relation_kind": "source_type_prior",
                "object_role": artifact.payload.get("object_role", "target_object"),
                "source_role": artifact.payload.get("source_role", ""),
                "source_base": artifact.payload.get("source_base", ""),
                "goal_object": artifact.payload.get("goal_object", ""),
                "action_patterns": artifact.payload.get("action_patterns", []),
            }
        elif fever_pattern == "content_search_route" or raw_artifact_role == "fever_content_search_route":
            # FEVER route memories should transfer as claim-type/search-role
            # procedures. Drop per-episode query variants so support can
            # aggregate into a real global route instead of many singleton
            # artifacts.
            claim_type = str(
                artifact.payload.get("claim_type") or artifact.anchor.get("claim_type") or "claim_verification"
            )
            anchor = {
                "task_family": artifact.anchor.get("task_family", "fever:claim_verification"),
                "goal_arity": artifact.anchor.get("goal_arity", 1),
                "progress_state": "need_search",
                "artifact_role": "fever_content_search_route",
                "domain": "fever",
                "claim_type": claim_type,
            }
            payload = {
                "source": "fever_episode_graph",
                "pattern_kind": "workflow",
                "fever_pattern": "content_search_route",
                "claim_type": claim_type,
                "search_role": artifact.payload.get("search_role", "primary entity"),
                "route_shape": artifact.payload.get("route_shape", "search_finish"),
                "action_patterns": [
                    "Search[current claim focused query]",
                    "Lookup[current claim relation keyword]",
                ],
            }
        elif fever_pattern in {"claim_type_no_results_recovery", "generic_no_results_recovery"}:
            claim_type = str(
                artifact.payload.get("claim_type") or artifact.anchor.get("claim_type") or "claim_verification"
            )
            generic = fever_pattern == "generic_no_results_recovery"
            anchor = {
                "task_family": (
                    "fever:claim_verification"
                    if generic
                    else artifact.anchor.get("task_family", "fever:claim_verification")
                ),
                "goal_arity": artifact.anchor.get("goal_arity", 1),
                "progress_state": "search_failed",
                "artifact_role": "fever_generic_no_results_recovery" if generic else "fever_no_results_recovery",
                "domain": "fever",
                "claim_type": "claim_verification" if generic else claim_type,
            }
            payload = {
                "source": "fever_episode_graph",
                "pattern_kind": "workflow",
                "fever_pattern": fever_pattern,
                "claim_type": "claim_verification" if generic else claim_type,
                "repair_patterns": [
                    "Search[focused current claim query]",
                    "Lookup[current claim relation keyword]",
                ],
            }
        else:
            anchor = _abstract_global_mapping(artifact.anchor)
            payload = _abstract_global_mapping(artifact.payload)
        return MemoryArtifact(
            artifact_id=_global_artifact_id(artifact, anchor, payload),
            kind=artifact.kind,
            summary=artifact.summary,
            anchor=anchor,
            payload=payload,
            level="global",
        )

    def _promote_legacy(
        self,
        global_memory: GlobalGraphMemory,
        local_memories: list[LocalGraphMemory],
        batch_name: str,
    ) -> GlobalGraphMemory:
        aggregated: dict[str, PromotionCandidate] = {}
        for local in local_memories:
            for candidate in local.candidates.values():
                merged = aggregated.get(candidate.candidate_id)
                if merged is None:
                    merged = PromotionCandidate(
                        candidate_id=candidate.candidate_id,
                        candidate_type=candidate.candidate_type,
                        summary=candidate.summary,
                        structure=dict(candidate.structure),
                    )
                    aggregated[candidate.candidate_id] = merged
                merged.source_episode_ids |= candidate.source_episode_ids
                merged.source_scenes |= candidate.source_scenes
                merged.positive += candidate.positive
                merged.negative += candidate.negative
                merged.stalled += candidate.stalled
                merged.utility += candidate.utility
                _merge_wilson_episode_evidence(
                    merged.wilson_episode_evidence,
                    candidate.wilson_episode_evidence,
                )

        for candidate in aggregated.values():
            pattern_kind = str(candidate.structure.get("pattern_kind", ""))
            transferability = self._transferability_score(candidate)
            adjusted_score = 0.55 * candidate.prior_score + 0.9 * transferability
            min_scene_coverage = self._min_scene_coverage(candidate)
            support = candidate.positive + candidate.negative
            confidence = candidate.confidence

            if candidate.candidate_type == CandidateType.PRECONDITION:
                if support < 2 or candidate.positive < 2 or confidence < 0.6:
                    continue
            elif candidate.candidate_type == CandidateType.WORKFLOW and pattern_kind == "workflow":
                if candidate.positive < 2 or candidate.positive < candidate.negative:
                    continue
            elif candidate.candidate_type == CandidateType.WORKFLOW and pattern_kind == "closure":
                if candidate.positive < 2:
                    continue
            elif candidate.candidate_type == CandidateType.FAILURE and pattern_kind == "anti_pattern":
                if candidate.stalled < 2:
                    continue
            if adjusted_score >= self.score_threshold and candidate.coverage >= min_scene_coverage:
                global_memory.candidates[candidate.candidate_id] = candidate

        aggregated_rules: dict[str, MemoryRule] = {}
        for local in local_memories:
            for rule in local.rules_by_id.values():
                if rule.rule_type in {RuleType.BLOCKED, RuleType.REPAIR}:
                    # These are valuable locally but too easy to over-apply across
                    # unseen tasks without the original PhaseE action reranker.
                    continue
                abstract_rule = self._global_rule_copy(rule)
                merged = aggregated_rules.get(abstract_rule.rule_id)
                if merged is None:
                    merged = abstract_rule
                    aggregated_rules[abstract_rule.rule_id] = merged
                merged.source_episode_ids |= rule.source_episode_ids
                merged.source_scenes |= rule.source_scenes
                merged.stats.support += rule.stats.support
                merged.stats.success += rule.stats.success
                merged.stats.failure += rule.stats.failure
                merged.stats.stalled += rule.stats.stalled
                merged.stats.utility += rule.stats.utility
                merged.stats.transfer_success += rule.stats.transfer_success
                merged.stats.transfer_trials += rule.stats.transfer_trials
                _merge_wilson_episode_evidence(
                    merged.wilson_episode_evidence,
                    rule.wilson_episode_evidence,
                )
                merged.specificity = max(merged.specificity, _rule_specificity(merged))
                merged.conflict = max(merged.conflict, rule.conflict)

        for rule in aggregated_rules.values():
            if _is_low_information_global_rule(rule):
                continue
            min_scene_coverage = self._rule_min_scene_coverage(rule)
            if (
                rule.support >= 3
                and rule.stats.success >= 2
                and rule.stats.confidence >= 0.6
                and rule.coverage >= min_scene_coverage
                and rule.promotion_score >= self.score_threshold
            ):
                global_memory.rules_by_id[rule.rule_id] = rule

        aggregated_artifacts: dict[str, MemoryArtifact] = {}
        for local in local_memories:
            for artifact in local.artifacts_by_id.values():
                abstract_artifact = self._global_artifact_copy(artifact)
                if abstract_artifact is None:
                    continue
                merged = aggregated_artifacts.get(abstract_artifact.artifact_id)
                if merged is None:
                    merged = abstract_artifact
                    aggregated_artifacts[abstract_artifact.artifact_id] = merged
                merged.source_episode_ids |= artifact.source_episode_ids
                merged.source_scenes |= artifact.source_scenes
                merged.stats.support += artifact.stats.support
                merged.stats.success += artifact.stats.success
                merged.stats.failure += artifact.stats.failure
                merged.stats.stalled += artifact.stats.stalled
                merged.stats.utility += artifact.stats.utility
                merged.stats.transfer_success += artifact.stats.transfer_success
                merged.stats.transfer_trials += artifact.stats.transfer_trials
                _merge_wilson_episode_evidence(
                    merged.wilson_episode_evidence,
                    artifact.wilson_episode_evidence,
                )
                merged.specificity = max(merged.specificity, _artifact_specificity(merged))
                merged.conflict = max(merged.conflict, artifact.conflict)

        for artifact in aggregated_artifacts.values():
            if _is_low_information_global_artifact(artifact):
                continue
            min_scene_coverage = self._artifact_min_scene_coverage(artifact)
            pattern_kind = str(artifact.payload.get("pattern_kind", "") or artifact.anchor.get("pattern_kind", ""))
            if pattern_kind == "source_type_prior":
                min_support = 2
                min_success = 1
                min_confidence = 0.45
            else:
                min_support = 3
                min_success = 2
                min_confidence = 0.6
            if (
                artifact.support >= min_support
                and artifact.stats.success >= min_success
                and artifact.stats.confidence >= min_confidence
                and artifact.coverage >= min_scene_coverage
                and artifact.promotion_score >= self.score_threshold
            ):
                global_memory.artifacts_by_id[artifact.artifact_id] = artifact

        if batch_name and batch_name not in global_memory.promoted_batches:
            global_memory.promoted_batches.append(batch_name)
        return global_memory

    def _aggregate_wilson_candidates(
        self,
        local_memories: list[LocalGraphMemory],
    ) -> dict[str, PromotionCandidate]:
        aggregated: dict[str, PromotionCandidate] = {}
        for local in local_memories:
            for candidate in local.candidates.values():
                merged = aggregated.get(candidate.candidate_id)
                if merged is None:
                    merged = PromotionCandidate(
                        candidate_id=candidate.candidate_id,
                        candidate_type=candidate.candidate_type,
                        summary=candidate.summary,
                        structure=dict(candidate.structure),
                    )
                    aggregated[candidate.candidate_id] = merged
                merged.source_episode_ids |= candidate.source_episode_ids
                merged.source_scenes |= candidate.source_scenes
                merged.positive += candidate.positive
                merged.negative += candidate.negative
                merged.stalled += candidate.stalled
                merged.utility += candidate.utility
                _merge_wilson_episode_evidence(
                    merged.wilson_episode_evidence,
                    candidate.wilson_episode_evidence,
                )
        return aggregated

    def _aggregate_wilson_rules(
        self,
        local_memories: list[LocalGraphMemory],
    ) -> dict[str, MemoryRule]:
        aggregated: dict[str, MemoryRule] = {}
        for local in local_memories:
            for rule in local.rules_by_id.values():
                abstract_rule = self._global_rule_copy(rule)
                merged = aggregated.get(abstract_rule.rule_id)
                if merged is None:
                    merged = abstract_rule
                    aggregated[abstract_rule.rule_id] = merged
                merged.source_episode_ids |= rule.source_episode_ids
                merged.source_scenes |= rule.source_scenes
                merged.stats.support += rule.stats.support
                merged.stats.success += rule.stats.success
                merged.stats.failure += rule.stats.failure
                merged.stats.stalled += rule.stats.stalled
                merged.stats.utility += rule.stats.utility
                merged.stats.transfer_success += rule.stats.transfer_success
                merged.stats.transfer_trials += rule.stats.transfer_trials
                _merge_wilson_episode_evidence(
                    merged.wilson_episode_evidence,
                    rule.wilson_episode_evidence,
                )
                merged.specificity = max(merged.specificity, _rule_specificity(merged))
                merged.conflict = max(merged.conflict, rule.conflict)
        return aggregated

    def _aggregate_wilson_artifacts(
        self,
        local_memories: list[LocalGraphMemory],
    ) -> dict[str, MemoryArtifact]:
        aggregated: dict[str, MemoryArtifact] = {}
        for local in local_memories:
            for artifact in local.artifacts_by_id.values():
                abstract_artifact = self._global_artifact_copy(artifact)
                if abstract_artifact is None:
                    continue
                merged = aggregated.get(abstract_artifact.artifact_id)
                if merged is None:
                    merged = abstract_artifact
                    aggregated[abstract_artifact.artifact_id] = merged
                merged.source_episode_ids |= artifact.source_episode_ids
                merged.source_scenes |= artifact.source_scenes
                merged.stats.support += artifact.stats.support
                merged.stats.success += artifact.stats.success
                merged.stats.failure += artifact.stats.failure
                merged.stats.stalled += artifact.stats.stalled
                merged.stats.utility += artifact.stats.utility
                merged.stats.transfer_success += artifact.stats.transfer_success
                merged.stats.transfer_trials += artifact.stats.transfer_trials
                _merge_wilson_episode_evidence(
                    merged.wilson_episode_evidence,
                    artifact.wilson_episode_evidence,
                )
                merged.specificity = max(merged.specificity, _artifact_specificity(merged))
                merged.conflict = max(merged.conflict, artifact.conflict)
        return aggregated

    def _wilson_decision(
        self,
        *,
        record_kind: str,
        record_id: str,
        evidence,
        structurally_valid: bool,
        low_information: bool = False,
    ) -> tuple[bool, dict[str, Any]]:
        score = evidence.score(alpha=self.wilson_alpha)
        reasons: list[str] = []
        if not structurally_valid:
            reasons.append("invalid_structure")
        if low_information:
            reasons.append("low_information")
        if score < self.wilson_threshold:
            reasons.append("wilson_below_threshold")
        accepted = not reasons
        decision: dict[str, Any] = {
            "record_kind": record_kind,
            "record_id": record_id,
            **evidence.to_dict(alpha=self.wilson_alpha),
            "structurally_valid": structurally_valid,
            "low_information": low_information,
            "accepted": accepted,
            "reasons": reasons,
        }
        return accepted, decision

    def _promote_wilson(
        self,
        global_memory: GlobalGraphMemory,
        local_memories: list[LocalGraphMemory],
        batch_name: str,
    ) -> GlobalGraphMemory:
        # Wilson promotion defines a committed snapshot from the supplied local
        # evidence.  Rebuild it instead of retaining records accepted by an
        # earlier, weaker evidence state.
        global_memory = GlobalGraphMemory()
        decisions: dict[str, list[dict[str, Any]]] = {
            "candidates": [],
            "rules": [],
            "artifacts": [],
        }

        for candidate in self._aggregate_wilson_candidates(local_memories).values():
            accepted, decision = self._wilson_decision(
                record_kind="candidate",
                record_id=candidate.candidate_id,
                evidence=candidate_wilson_evidence(candidate),
                structurally_valid=candidate_is_structurally_valid(candidate),
            )
            decisions["candidates"].append(decision)
            if accepted:
                global_memory.candidates[candidate.candidate_id] = candidate

        for rule in self._aggregate_wilson_rules(local_memories).values():
            accepted, decision = self._wilson_decision(
                record_kind="rule",
                record_id=rule.rule_id,
                evidence=rule_wilson_evidence(rule),
                structurally_valid=rule_is_structurally_valid(rule),
                low_information=_is_low_information_global_rule(rule),
            )
            decisions["rules"].append(decision)
            if accepted:
                global_memory.rules_by_id[rule.rule_id] = rule

        for artifact in self._aggregate_wilson_artifacts(local_memories).values():
            accepted, decision = self._wilson_decision(
                record_kind="artifact",
                record_id=artifact.artifact_id,
                evidence=artifact_wilson_evidence(artifact),
                structurally_valid=artifact_is_structurally_valid(artifact),
                low_information=_is_low_information_global_artifact(artifact),
            )
            decisions["artifacts"].append(decision)
            if accepted:
                global_memory.artifacts_by_id[artifact.artifact_id] = artifact

        if batch_name and batch_name not in global_memory.promoted_batches:
            global_memory.promoted_batches.append(batch_name)
        self.last_wilson_memory = global_memory
        self.last_promotion_report = {
            "policy": self.policy,
            "active_policy": "wilson" if self.policy == "wilson" else "legacy",
            "evidence_source": "candidate_relative_episode_verdicts",
            "limitations": [
                "Repeated matches of the same record within one source episode are collapsed to one verdict; old snapshots without an episode ledger use their occurrence counts as a compatibility fallback.",
                "The verdicts describe record-relative observed outcomes and do not by themselves establish a causal transfer effect.",
            ],
            "wilson": {
                "alpha": self.wilson_alpha,
                "threshold": self.wilson_threshold,
                "interval": "one_sided_lower_bound",
                "source_coverage": "diagnostic_only",
            },
            "retrieval_experiment": {
                "paper_eq22_27_all_ones": str(os.getenv("NV_MEMCO_PAPER_EQ22_27_ALL_ONES", "")).strip().lower()
                in {"1", "true", "yes", "on"},
                "paper_a3_2_activation_all_ones": str(os.getenv("NV_MEMCO_PAPER_A3_2_ACTIVATION_ALL_ONES", ""))
                .strip()
                .lower()
                in {"1", "true", "yes", "on"},
                "manifest": (
                    "configs/wilson_qwen32b_alfworld_eq22_27_a3_2_activation_allones.json"
                    if str(os.getenv("NV_MEMCO_PAPER_A3_2_ACTIVATION_ALL_ONES", "")).strip().lower()
                    in {"1", "true", "yes", "on"}
                    else "configs/wilson_qwen32b_alfworld_eq22_27_allones.json"
                ),
            },
            "selected": {
                "candidate_ids": sorted(global_memory.candidates),
                "rule_ids": sorted(global_memory.rules_by_id),
                "artifact_ids": sorted(global_memory.artifacts_by_id),
            },
            "decisions": decisions,
        }
        return global_memory

    def promote(
        self,
        global_memory: GlobalGraphMemory,
        local_memories: list[LocalGraphMemory],
        batch_name: str,
    ) -> GlobalGraphMemory:
        if self.policy == "legacy":
            self.last_wilson_memory = None
            self.last_promotion_report = {}
            return self._promote_legacy(global_memory, local_memories, batch_name)

        if self.policy == "wilson":
            return self._promote_wilson(global_memory, local_memories, batch_name)

        # Shadow mode computes Wilson decisions from an empty output memory so
        # legacy records cannot leak into the comparison.  The returned object
        # is still produced exclusively by the unchanged legacy path.
        shadow_memory = self._promote_wilson(
            GlobalGraphMemory(),
            local_memories,
            batch_name,
        )
        report = dict(self.last_promotion_report)
        legacy_memory = self._promote_legacy(global_memory, local_memories, batch_name)
        report["comparison"] = {
            "legacy": {
                "candidate_ids": sorted(legacy_memory.candidates),
                "rule_ids": sorted(legacy_memory.rules_by_id),
                "artifact_ids": sorted(legacy_memory.artifacts_by_id),
            },
            "wilson": {
                "candidate_ids": sorted(shadow_memory.candidates),
                "rule_ids": sorted(shadow_memory.rules_by_id),
                "artifact_ids": sorted(shadow_memory.artifacts_by_id),
            },
        }
        self.last_wilson_memory = shadow_memory
        self.last_promotion_report = report
        return legacy_memory
