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
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Domain(str, Enum):
    ALFWORLD = "alfworld"
    PDDL = "pddl"
    FEVER = "fever"
    WEB = "web"


class NodeType(str, Enum):
    STATE = "state"
    ACTION = "action"
    SUBGOAL = "subgoal"
    FAILURE = "failure"


class EdgeType(str, Enum):
    TEMPORAL = "temporal"
    CAUSES = "causes"
    FAILS_UNDER = "fails_under"
    ADVANCES_TO = "advances_to"


class CandidateType(str, Enum):
    PRECONDITION = "precondition_rule"
    FAILURE = "failure_pattern"
    WORKFLOW = "workflow_chunk"
    REPAIR = "repair_pattern"


class RuleType(str, Enum):
    PRECONDITION = "precondition"
    BLOCKED = "blocked"
    REPAIR = "repair"
    WORKFLOW = "workflow"
    CLOSURE = "closure"


class ArtifactKind(str, Enum):
    PROTOTYPE = "prototype"
    RULE = "rule"
    REFLECTION = "reflection"


class ActionFamily(str, Enum):
    NAVIGATE = "navigate"
    INSPECT = "inspect"
    MANIPULATE = "manipulate"
    CONTROL = "control"
    SEARCH = "search"
    OTHER = "other"


@dataclass(slots=True)
class Stats:
    support: int = 0
    positive: int = 0
    negative: int = 0
    stalled: int = 0

    def record(self, *, positive: bool, stalled: bool = False) -> None:
        self.support += 1
        if positive:
            self.positive += 1
        else:
            self.negative += 1
        if stalled:
            self.stalled += 1

    @property
    def confidence(self) -> float:
        total = self.positive + self.negative
        return self.positive / total if total else 0.0

    @property
    def stall_rate(self) -> float:
        return self.stalled / self.support if self.support else 0.0


@dataclass(slots=True)
class CanonicalAction:
    domain: Domain
    family: ActionFamily
    verb: str
    slots: dict[str, str] = field(default_factory=dict)
    surface_form: str | None = None

    @property
    def canonical_str(self) -> str:
        if not self.slots:
            return self.verb
        parts = ",".join(f"{k}={self.slots[k]}" for k in sorted(self.slots))
        return f"{self.verb}({parts})"


@dataclass(slots=True)
class StateSummary:
    domain: Domain
    scene_id: str
    location: str | None = None
    visible_objects: tuple[str, ...] = ()
    open_containers: tuple[str, ...] = ()
    closed_containers: tuple[str, ...] = ()
    held_objects: tuple[str, ...] = ()
    searched_locations: tuple[str, ...] = ()
    admissible_verbs: tuple[str, ...] = ()
    workflow_stage: str | None = None
    raw_observation: str | None = None

    @property
    def signature(self) -> str:
        return "|".join(
            [
                self.scene_id,
                self.location or "unknown",
                self.workflow_stage or "unknown",
                ",".join(self.held_objects),
                ",".join(self.open_containers),
                ",".join(sorted(self.visible_objects[:6])),
            ]
        )


@dataclass(slots=True)
class StepFeedback:
    success: bool
    score: float = 0.0
    done: bool = False
    failure_label: str | None = None
    state_delta: tuple[str, ...] = ()


@dataclass(slots=True)
class EpisodeStep:
    step_idx: int
    state: StateSummary
    action: CanonicalAction
    next_state: StateSummary
    feedback: StepFeedback
    subgoal: str | None = None


@dataclass(slots=True)
class EpisodeRecord:
    agent_id: str
    scene_id: str
    task_id: str
    goal: str
    steps: list[EpisodeStep] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class GraphNode:
    node_id: str
    node_type: NodeType
    signature: str
    payload: dict[str, Any]
    stats: Stats = field(default_factory=Stats)


@dataclass(slots=True)
class GraphEdge:
    src: str
    dst: str
    edge_type: EdgeType
    signature: str
    payload: dict[str, Any]
    stats: Stats = field(default_factory=Stats)


@dataclass(slots=True)
class EpisodeGraph:
    episode_id: str
    agent_id: str
    scene_id: str
    task_id: str
    goal: str
    nodes: dict[str, GraphNode] = field(default_factory=dict)
    edges: list[GraphEdge] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PromotionCandidate:
    candidate_id: str
    candidate_type: CandidateType
    summary: str
    structure: dict[str, Any]
    source_episode_ids: set[str] = field(default_factory=set)
    source_scenes: set[str] = field(default_factory=set)
    positive: int = 0
    negative: int = 0
    stalled: int = 0
    utility: float = 0.0
    # Wilson uses one candidate-relative verdict per episode.  The aggregate
    # counters above remain unchanged for the legacy scorer and reporting.
    wilson_episode_evidence: dict[str, dict[str, int]] = field(default_factory=dict)

    def observe(
        self,
        *,
        scene_id: str,
        episode_id: str,
        positive: bool,
        stalled: bool = False,
        utility_delta: float = 0.0,
    ) -> None:
        self.source_episode_ids.add(episode_id)
        self.source_scenes.add(scene_id)
        if positive:
            self.positive += 1
        else:
            self.negative += 1
        if stalled:
            self.stalled += 1
        self.utility += utility_delta
        evidence_id = f"{scene_id}\x1f{episode_id}"
        verdict = self.wilson_episode_evidence.setdefault(
            evidence_id,
            {"supporting": 0, "contradicting": 0, "stalled": 0},
        )
        verdict["supporting" if positive else "contradicting"] += 1
        if stalled:
            verdict["stalled"] += 1

    @property
    def confidence(self) -> float:
        total = self.positive + self.negative
        return self.positive / total if total else 0.0

    @property
    def coverage(self) -> int:
        return len(self.source_scenes)

    @property
    def prior_score(self) -> float:
        support = self.positive + self.negative
        if support == 0:
            return 0.0
        all_ones = str(os.getenv("NV_MEMCO_PAPER_EQ22_27_ALL_ONES", "")).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        support_term = min(support / (1.0 if all_ones else 4.0), 1.0)
        coverage_term = min(self.coverage / (1.0 if all_ones else 2.0), 1.0)
        utility_term = self.utility / support
        stall_penalty = self.stalled / support
        if all_ones:
            return (
                1.0 * self.confidence
                + 1.0 * support_term
                + 1.0 * coverage_term
                + 1.0 * utility_term
                - 1.0 * stall_penalty
            )
        return (
            1.2 * self.confidence + 0.6 * support_term + 0.4 * coverage_term + 0.4 * utility_term - 0.5 * stall_penalty
        )


@dataclass(slots=True)
class RuleStats:
    support: int = 0
    success: int = 0
    failure: int = 0
    stalled: int = 0
    utility: float = 0.0
    transfer_success: int = 0
    transfer_trials: int = 0

    def observe(
        self,
        *,
        success: bool,
        stalled: bool = False,
        utility_delta: float = 0.0,
        transfer_success: bool = False,
        transfer_trial: bool = False,
    ) -> None:
        self.support += 1
        if success:
            self.success += 1
        else:
            self.failure += 1
        if stalled:
            self.stalled += 1
        self.utility += utility_delta
        if transfer_trial:
            self.transfer_trials += 1
            if transfer_success:
                self.transfer_success += 1

    @property
    def confidence(self) -> float:
        total = self.success + self.failure
        return self.success / total if total else 0.0

    @property
    def transfer_rate(self) -> float:
        return self.transfer_success / self.transfer_trials if self.transfer_trials else 0.0

    @property
    def utility_avg(self) -> float:
        return self.utility / self.support if self.support else 0.0


@dataclass(slots=True)
class ArtifactStats:
    support: int = 0
    success: int = 0
    failure: int = 0
    stalled: int = 0
    utility: float = 0.0
    transfer_success: int = 0
    transfer_trials: int = 0

    def observe(
        self,
        *,
        success: bool,
        stalled: bool = False,
        utility_delta: float = 0.0,
        transfer_success: bool = False,
        transfer_trial: bool = False,
    ) -> None:
        self.support += 1
        if success:
            self.success += 1
        else:
            self.failure += 1
        if stalled:
            self.stalled += 1
        self.utility += utility_delta
        if transfer_trial:
            self.transfer_trials += 1
            if transfer_success:
                self.transfer_success += 1

    @property
    def confidence(self) -> float:
        total = self.success + self.failure
        return self.success / total if total else 0.0

    @property
    def failure_rate(self) -> float:
        total = self.success + self.failure
        return self.failure / total if total else 0.0

    @property
    def transfer_rate(self) -> float:
        return self.transfer_success / self.transfer_trials if self.transfer_trials else 0.0

    @property
    def utility_avg(self) -> float:
        return self.utility / self.support if self.support else 0.0


@dataclass(slots=True)
class MemoryRule:
    rule_id: str
    rule_type: RuleType
    summary: str
    task_family: str = ""
    goal_arity: int = 1
    progress_state: str = ""
    goal_roles: dict[str, str] = field(default_factory=dict)
    condition: dict[str, Any] = field(default_factory=dict)
    effect: dict[str, Any] = field(default_factory=dict)
    source_episode_ids: set[str] = field(default_factory=set)
    source_scenes: set[str] = field(default_factory=set)
    stats: RuleStats = field(default_factory=RuleStats)
    specificity: float = 0.0
    conflict: float = 0.0
    wilson_episode_evidence: dict[str, dict[str, int]] = field(default_factory=dict)

    def observe(
        self,
        *,
        scene_id: str,
        episode_id: str,
        success: bool,
        stalled: bool = False,
        utility_delta: float = 0.0,
        transfer_success: bool = False,
        transfer_trial: bool = False,
    ) -> None:
        self.source_episode_ids.add(episode_id)
        self.source_scenes.add(scene_id)
        self.stats.observe(
            success=success,
            stalled=stalled,
            utility_delta=utility_delta,
            transfer_success=transfer_success,
            transfer_trial=transfer_trial,
        )
        evidence_id = f"{scene_id}\x1f{episode_id}"
        verdict = self.wilson_episode_evidence.setdefault(
            evidence_id,
            {"supporting": 0, "contradicting": 0, "stalled": 0},
        )
        verdict["supporting" if success else "contradicting"] += 1
        if stalled:
            verdict["stalled"] += 1

    @property
    def coverage(self) -> int:
        return len(self.source_scenes)

    @property
    def support(self) -> int:
        return self.stats.support

    @property
    def promotion_score(self) -> float:
        support_term = min(self.support / 4.0, 1.0)
        coverage_term = min(self.coverage / 2.0, 1.0)
        return (
            1.1 * self.stats.confidence
            + 0.45 * support_term
            + 0.35 * coverage_term
            + 0.4 * self.stats.utility_avg
            + 0.25 * self.stats.transfer_rate
            - 0.35 * self.specificity
            - 0.3 * self.conflict
            - 0.45 * (self.stats.stalled / self.support if self.support else 0.0)
        )


@dataclass(slots=True)
class MemoryArtifact:
    artifact_id: str
    kind: ArtifactKind
    summary: str
    anchor: dict[str, Any] = field(default_factory=dict)
    payload: dict[str, Any] = field(default_factory=dict)
    level: str = "local"
    source_episode_ids: set[str] = field(default_factory=set)
    source_scenes: set[str] = field(default_factory=set)
    stats: ArtifactStats = field(default_factory=ArtifactStats)
    specificity: float = 0.0
    conflict: float = 0.0
    wilson_episode_evidence: dict[str, dict[str, int]] = field(default_factory=dict)

    def observe(
        self,
        *,
        scene_id: str,
        episode_id: str,
        success: bool,
        stalled: bool = False,
        utility_delta: float = 0.0,
        transfer_success: bool = False,
        transfer_trial: bool = False,
    ) -> None:
        self.source_episode_ids.add(episode_id)
        self.source_scenes.add(scene_id)
        self.stats.observe(
            success=success,
            stalled=stalled,
            utility_delta=utility_delta,
            transfer_success=transfer_success,
            transfer_trial=transfer_trial,
        )
        evidence_id = f"{scene_id}\x1f{episode_id}"
        verdict = self.wilson_episode_evidence.setdefault(
            evidence_id,
            {"supporting": 0, "contradicting": 0, "stalled": 0},
        )
        verdict["supporting" if success else "contradicting"] += 1
        if stalled:
            verdict["stalled"] += 1

    @property
    def coverage(self) -> int:
        return len(self.source_scenes)

    @property
    def support(self) -> int:
        return self.stats.support

    @property
    def promotion_score(self) -> float:
        support_term = min(self.support / 4.0, 1.0)
        coverage_term = min(self.coverage / 2.0, 1.0)
        if self.kind == ArtifactKind.REFLECTION:
            evidence_term = self.stats.failure_rate
            return (
                0.95 * evidence_term
                + 0.35 * support_term
                + 0.25 * coverage_term
                + 0.3 * self.stats.utility_avg
                + 0.2 * self.stats.transfer_rate
                - 0.25 * self.specificity
                - 0.2 * self.conflict
                - 0.2 * (self.stats.stalled / self.support if self.support else 0.0)
            )
        return (
            1.0 * self.stats.confidence
            + 0.4 * support_term
            + 0.3 * coverage_term
            + 0.35 * self.stats.utility_avg
            + 0.2 * self.stats.transfer_rate
            - 0.3 * self.specificity
            - 0.25 * self.conflict
            - 0.3 * (self.stats.stalled / self.support if self.support else 0.0)
        )


@dataclass(slots=True)
class LocalGraphMemory:
    agent_id: str
    nodes_by_signature: dict[str, GraphNode] = field(default_factory=dict)
    edges_by_signature: dict[str, GraphEdge] = field(default_factory=dict)
    candidates: dict[str, PromotionCandidate] = field(default_factory=dict)
    rules_by_id: dict[str, MemoryRule] = field(default_factory=dict)
    artifacts_by_id: dict[str, MemoryArtifact] = field(default_factory=dict)
    episode_ids: list[str] = field(default_factory=list)


@dataclass(slots=True)
class GlobalGraphMemory:
    candidates: dict[str, PromotionCandidate] = field(default_factory=dict)
    rules_by_id: dict[str, MemoryRule] = field(default_factory=dict)
    artifacts_by_id: dict[str, MemoryArtifact] = field(default_factory=dict)
    promoted_batches: list[str] = field(default_factory=list)


@dataclass(slots=True)
class MemoryQuery:
    goal: str
    scene_id: str | None = None
    current_stage: str | None = None
    location: str | None = None
    progress_hint: str = ""
    progress_state: str = ""
    task_family: str = ""
    goal_roles: dict[str, str] = field(default_factory=dict)
    required_count: int = 1
    held_relevant_count: int = 0
    placed_relevant_count: int = 0
    remaining_relevant_count: int = 0
    destination_reached: bool = False
    goal_object_matches_visible: bool = False
    admissible_actions: tuple[CanonicalAction, ...] = ()
    desired_types: tuple[CandidateType, ...] = ()
    failure_label: str | None = None
    keywords: tuple[str, ...] = ()
    belief: dict[str, Any] = field(default_factory=dict)
    dynamic_context: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SupportItem:
    source: str
    candidate_id: str
    candidate_type: CandidateType
    summary: str
    score: float
    task_family: str = ""
    pattern_kind: str = ""
    branch_tag: str = ""
    positive: int = 0
    negative: int = 0
    stalled: int = 0
    state_relevance: float = 0.0
    task_relevance: float = 0.0
    goal_relevance: float = 0.0
    action_patterns: tuple[str, ...] = ()
    dynamic: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SupportBundle:
    query: str
    goal_object: str = ""
    goal_destination: str = ""
    goal_tool: str = ""
    progress_state: str = ""
    local_graph_items: list[SupportItem] = field(default_factory=list)
    local_promoted_items: list[SupportItem] = field(default_factory=list)
    global_promoted_items: list[SupportItem] = field(default_factory=list)
    local_items: list[SupportItem] = field(default_factory=list)
    global_items: list[SupportItem] = field(default_factory=list)
    fact_items: list[SupportItem] = field(default_factory=list)
    relation_items: list[SupportItem] = field(default_factory=list)
    plan_items: list[SupportItem] = field(default_factory=list)
    global_task_plan_items: list[SupportItem] = field(default_factory=list)
    workflow_items: list[SupportItem] = field(default_factory=list)
    precondition_items: list[SupportItem] = field(default_factory=list)
    repair_items: list[SupportItem] = field(default_factory=list)
    closure_items: list[SupportItem] = field(default_factory=list)
    reflection_items: list[SupportItem] = field(default_factory=list)
    routing_decisions: dict[str, str] = field(default_factory=dict)
    routing_weights: dict[str, dict[str, float]] = field(default_factory=dict)
    blocked_actions: list[str] = field(default_factory=list)
    suggested_actions: list[str] = field(default_factory=list)
    workflow_hints: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    task_need_analysis: list[str] = field(default_factory=list)
    local_graph_contribution: list[SupportItem] = field(default_factory=list)
    local_promoted_contribution: list[SupportItem] = field(default_factory=list)
    global_promoted_contribution: list[SupportItem] = field(default_factory=list)
    fused_support_items: list[SupportItem] = field(default_factory=list)

    def to_text(self, max_items: int = 5) -> str:
        def _clean(summary: str) -> str:
            cleaned = str(summary or "").strip()
            prefixes = (
                "Scene relation: ",
                "Plan: ",
                "Workflow pattern: ",
                "Workflow: ",
                "Reflection: ",
                "Closure: ",
                "Local state fact: ",
                "Prototype: ",
            )
            for prefix in prefixes:
                if cleaned.startswith(prefix):
                    cleaned = cleaned[len(prefix) :].strip()
            return cleaned

        lines: list[str] = []
        if self.fact_items:
            lines.append("Facts:")
            for item in self.fact_items[:1]:
                lines.append(f"- {_clean(item.summary)}")
        if self.relation_items:
            lines.append("Search Priors:")
            for item in self.relation_items[:2]:
                lines.append(f"- {_clean(item.summary)}")
        if self.plan_items:
            lines.append("Plan:")
            for item in self.plan_items[:1]:
                plan_steps = list(item.dynamic.get("plan_steps", []) or [])
                if self.progress_state == "search" and len(plan_steps) >= 2:
                    lines.append(f"- Search subplan: {plan_steps[0]} -> {plan_steps[1]}")
                    if len(plan_steps) >= 4:
                        lines.append(f"- Delivery subplan after target is found: {plan_steps[2]} -> {plan_steps[3]}")
                else:
                    lines.append(f"- {_clean(item.summary)}")
        if self.workflow_items:
            lines.append("Workflow:")
            for item in self.workflow_items[:1]:
                lines.append(f"- {_clean(item.summary)}")
        renderable_preconditions = [
            item for item in self.precondition_items if not item.summary.lower().startswith("in similar states, try ")
        ]
        if renderable_preconditions:
            lines.append("Preconditions:")
            for item in renderable_preconditions[:1]:
                lines.append(f"- {_clean(item.summary)}")
        if self.repair_items:
            lines.append("Repair:")
            for item in self.repair_items[:1]:
                lines.append(f"- {_clean(item.summary)}")
        if self.reflection_items:
            lines.append("Reflection:")
            for item in self.reflection_items[:1]:
                lines.append(f"- {_clean(item.summary)}")
        if self.closure_items:
            lines.append("Closure:")
            for item in self.closure_items[:1]:
                lines.append(f"- {_clean(item.summary)}")
        if self.blocked_actions:
            lines.append(f"Blocked Actions: {', '.join(self.blocked_actions[:max_items])}")
        if self.workflow_hints:
            lines.append(f"Workflow Hints: {', '.join(self.workflow_hints[:max_items])}")
        if self.task_need_analysis:
            lines.append(f"Task Needs: {', '.join(self.task_need_analysis[:max_items])}")
        if not lines:
            return ""
        return "\n".join(lines)

    def to_planner_text(
        self,
        max_items: int = 5,
        *,
        current_location: str = "",
        exhausted_locations: tuple[str, ...] | list[str] = (),
        visible_objects: tuple[str, ...] | list[str] = (),
        held_objects: tuple[str, ...] | list[str] = (),
        task_family: str = "",
    ) -> str:
        def _clean(summary: str) -> str:
            cleaned = str(summary or "").strip()
            prefixes = (
                "Scene relation: ",
                "Plan: ",
                "Workflow pattern: ",
                "Workflow: ",
                "Reflection: ",
                "Closure: ",
                "Local state fact: ",
                "Prototype: ",
                "Precondition: ",
                "Blocked: ",
            )
            for prefix in prefixes:
                if cleaned.startswith(prefix):
                    cleaned = cleaned[len(prefix) :].strip()
            return cleaned

        def _normalize(text: str) -> str:
            lowered = str(text or "").lower().replace("_", " ").replace("-", " ")
            parts = [part for part in lowered.split() if part not in {"a", "an", "the", "some"}]
            return " ".join(parts)

        goal_terms = tuple(
            term
            for term in (
                _normalize(self.goal_object),
                _normalize(self.goal_destination),
                _normalize(self.goal_tool),
            )
            if term
        )
        search_like = str(self.progress_state or "").startswith("search")
        current_location_term = _normalize(current_location)
        exhausted_terms = tuple(_normalize(item) for item in exhausted_locations if item)
        visible_terms = tuple(_normalize(item) for item in visible_objects if item)
        held_terms = tuple(_normalize(item) for item in held_objects if item)
        task_family_term = _normalize(task_family)
        goal_object_term = _normalize(self.goal_object)
        secondary_goal_terms = tuple(
            term
            for term in (
                _normalize(self.goal_destination),
                _normalize(self.goal_tool),
            )
            if term
        )

        def _mentions_goal(text: str) -> bool:
            normalized = _normalize(text)
            return any(term in normalized for term in goal_terms)

        def _mentions_goal_object(text: str) -> bool:
            normalized = _normalize(text)
            return bool(goal_object_term) and goal_object_term in normalized

        def _mentions_secondary_goal(text: str) -> bool:
            normalized = _normalize(text)
            return any(term in normalized for term in secondary_goal_terms)

        def _is_low_precision(text: str) -> bool:
            normalized = _normalize(text)
            if normalized.startswith("near "):
                return True
            if "graph anchor:" in normalized or "similar subgraphs center on" in normalized:
                return True
            generic_patterns = (
                "within search, open supports progress",
                "within search, examine supports progress",
                "examine advances the search",
                "wrong target object tends to derail progress",
            )
            return any(pattern in normalized for pattern in generic_patterns)

        def _is_template_like(text: str) -> bool:
            normalized = _normalize(text)
            template_markers = (
                "target object",
                "goal destination",
                "support surface",
                "container",
                "search -> carry target",
                "go(target=",
                "take(object=target_object",
                "move(destination=goal_destination",
            )
            return any(marker in normalized for marker in template_markers)

        def _has_negative_search_signal(text: str) -> bool:
            normalized = _normalize(text)
            negative_patterns = (
                "not found",
                "without target",
                "did not reveal",
                "no target",
                "after searching",
                "already checked",
            )
            return any(pattern in normalized for pattern in negative_patterns)

        def _has_positive_located_claim(text: str) -> bool:
            normalized = _normalize(text)
            positive_patterns = (
                "visible",
                "found at",
                "was found at",
                "located at",
                "contains",
                "similar states are near",
            )
            return any(pattern in normalized for pattern in positive_patterns)

        def _is_post_acquisition_evidence(text: str) -> bool:
            normalized = _normalize(text)
            post_patterns = (
                "holding relevant",
                "holding target",
                "pick up the target",
                "pick up target",
                "carry target",
            )
            return any(pattern in normalized for pattern in post_patterns)

        def _conflicts_with_state(text: str) -> bool:
            normalized = _normalize(text)
            if not normalized:
                return False
            if _has_negative_search_signal(normalized):
                return False
            if _mentions_goal_object(normalized) and _has_positive_located_claim(normalized):
                if any(term and term in normalized for term in exhausted_terms):
                    return True
                if current_location_term and current_location_term in normalized:
                    goal_currently_visible = any(
                        goal_object_term and goal_object_term in item for item in visible_terms
                    )
                    goal_currently_held = any(goal_object_term and goal_object_term in item for item in held_terms)
                    if not goal_currently_visible and not goal_currently_held:
                        return True
            return False

        def _is_conflicting_positive_evidence(text: str) -> bool:
            normalized = _normalize(text)
            if not normalized:
                return False
            return (
                _mentions_goal_object(normalized)
                and _has_positive_located_claim(normalized)
                and _conflicts_with_state(normalized)
            )

        def _score_text(text: str, *, channel: str) -> float:
            if not text:
                return -999.0
            normalized = _normalize(text)
            score = 0.0
            mentions_goal_object = _mentions_goal_object(normalized)
            mentions_secondary = _mentions_secondary_goal(normalized)
            mentions_goal = mentions_goal_object or mentions_secondary
            if mentions_goal_object:
                score += 4.0
            if mentions_secondary:
                score += 2.5
            if mentions_goal_object and mentions_secondary:
                score += 1.5
            if _has_negative_search_signal(normalized):
                score += 2.0
            if _is_low_precision(normalized):
                score -= 3.0
            if _is_template_like(normalized):
                score -= 1.6
                if search_like:
                    score -= 1.0
            if search_like and _is_post_acquisition_evidence(normalized):
                score -= 4.0
            if (
                search_like
                and mentions_goal_object
                and _has_positive_located_claim(normalized)
                and not _is_post_acquisition_evidence(normalized)
            ):
                score += 1.25
            if _conflicts_with_state(normalized):
                score -= 8.0
            if search_like and not mentions_goal:
                score -= 2.5
            if channel == "warning":
                score += 0.5
            elif channel == "experience":
                score += 0.25
                if search_like and not mentions_goal and "workflow" not in normalized:
                    score -= 1.0
            elif channel == "evidence":
                score += 0.75
            if task_family_term and task_family_term == "look at obj in light":
                if "use" in normalized:
                    score += 1.5
                if _normalize(self.goal_tool) and _normalize(self.goal_tool) in normalized:
                    score += 1.0
            return score

        def _rank(items: list[tuple[str, str]]) -> list[tuple[float, int, str, str]]:
            scored: list[tuple[float, int, str, str]] = []
            for idx, (text, channel) in enumerate(items):
                if not text:
                    continue
                score = _score_text(text, channel=channel)
                scored.append((score, -idx, text, channel))
            scored.sort(reverse=True)
            return scored

        def _select(
            scored: list[tuple[float, int, str, str]],
            *,
            limit: int,
            minimum_score: float,
        ) -> list[str]:
            picked: list[str] = []
            for score, _neg_idx, text, _channel in scored:
                if score < minimum_score:
                    continue
                if text not in picked:
                    picked.append(text)
                if len(picked) >= limit:
                    break
            return picked

        def _fallback_select(
            scored: list[tuple[float, int, str, str]],
            *,
            limit: int,
            conflict_floor: float = -6.0,
            allow_conflicting_positive: bool = True,
        ) -> list[str]:
            picked: list[str] = []
            for score, _neg_idx, text, _channel in scored:
                if score <= conflict_floor:
                    continue
                if not allow_conflicting_positive and _is_conflicting_positive_evidence(text):
                    continue
                if text not in picked:
                    picked.append(text)
                if len(picked) >= limit:
                    break
            return picked

        def _strength_label(top_score: float, kept_count: int) -> str:
            if kept_count <= 0:
                return "none"
            if top_score >= 5.0:
                return "strong"
            if top_score >= 2.5:
                return "moderate"
            return "weak"

        def _is_failure_derived(item: SupportItem) -> bool:
            if item.candidate_type in {CandidateType.FAILURE, CandidateType.REPAIR}:
                return True
            if item.branch_tag in {"failure_branch", "repair_branch"}:
                return True
            if item.dynamic.get("artifact_kind") == ArtifactKind.REFLECTION.value:
                return True
            if item.pattern_kind in {"reflection", "repair", "failure", "anti_pattern"}:
                return True
            if str(item.dynamic.get("relation_kind", "")) == "searched_empty":
                return True
            return False

        success_evidence_candidates: list[tuple[str, str]] = []
        failure_reflection_candidates: list[tuple[str, str]] = []

        route_group_specs = [
            ("grounding", list(self.local_graph_contribution[:2])),
            ("domain", list(self.local_promoted_contribution[:2])),
            ("transfer", list(self.global_promoted_contribution[:2])),
        ]
        route_sections: list[tuple[str, list[str]]] = []
        route_score_values: list[float] = []

        for route_name, route_items in route_group_specs:
            picked: list[str] = []
            route_candidates = [(_clean(item.summary), "evidence") for item in route_items if _clean(item.summary)]
            scored_route = _rank(route_candidates)
            route_score_values.extend(score for score, _idx, text, _channel in scored_route if text)
            if route_name == "grounding":
                picked = _select(scored_route, limit=2, minimum_score=1.0)
                if not picked and scored_route:
                    picked = _fallback_select(
                        scored_route,
                        limit=1,
                        allow_conflicting_positive=not search_like,
                    )
            elif route_name == "domain":
                picked = _select(scored_route, limit=2, minimum_score=0.75)
                if not picked and scored_route:
                    picked = _fallback_select(scored_route, limit=1)
            else:
                picked = _select(scored_route, limit=2, minimum_score=0.75)
                if not picked and scored_route:
                    picked = _fallback_select(scored_route, limit=1)
            if picked:
                route_sections.append((route_name, picked))

        for item in self.relation_items[:2]:
            text = _clean(item.summary)
            if search_like and _is_post_acquisition_evidence(text):
                continue
            if _is_failure_derived(item):
                failure_reflection_candidates.append((text, "warning"))
            else:
                success_evidence_candidates.append((text, "evidence"))
        for item in self.fact_items[:2]:
            text = _clean(item.summary)
            if search_like and _is_post_acquisition_evidence(text):
                continue
            if _is_failure_derived(item):
                failure_reflection_candidates.append((text, "warning"))
            else:
                success_evidence_candidates.append((text, "evidence"))
        for item in self.precondition_items[:1]:
            text = _clean(item.summary)
            if search_like and _is_post_acquisition_evidence(text):
                continue
            if _is_failure_derived(item):
                failure_reflection_candidates.append((text, "warning"))
            else:
                success_evidence_candidates.append((text, "experience"))
        for item in self.reflection_items[:1]:
            text = _clean(item.summary)
            failure_reflection_candidates.append((text, "warning"))
        for item in self.repair_items[:1]:
            text = _clean(item.summary)
            failure_reflection_candidates.append((text, "warning"))

        success_experience_candidates: list[tuple[str, str]] = []
        if self.plan_items:
            for item in self.plan_items[:1]:
                if text := _clean(item.summary):
                    if search_like and _is_post_acquisition_evidence(text):
                        continue
                    if _is_failure_derived(item):
                        failure_reflection_candidates.append((text, "warning"))
                    else:
                        success_experience_candidates.append((text, "experience"))
        if self.workflow_items:
            for item in self.workflow_items[:1]:
                if text := _clean(item.summary):
                    if search_like and _is_post_acquisition_evidence(text):
                        continue
                    if _is_failure_derived(item):
                        failure_reflection_candidates.append((text, "warning"))
                    else:
                        success_experience_candidates.append((text, "experience"))
        if not success_experience_candidates and self.workflow_hints:
            for item in self.workflow_hints[:max_items]:
                if text := str(item).strip():
                    success_experience_candidates.append((text, "experience"))

        explicit_warning_candidates = [
            (text, "warning") for item in self.warnings[:max_items] if (text := str(item).strip())
        ]

        scored_success_evidence = _rank(success_evidence_candidates)
        scored_success_experience = _rank(success_experience_candidates)
        scored_failure_reflections = _rank(failure_reflection_candidates + explicit_warning_candidates)

        success_evidence = _select(scored_success_evidence, limit=max_items, minimum_score=1.5)
        success_experience = _select(scored_success_experience, limit=max_items, minimum_score=0.75)
        failure_reflections = _select(scored_failure_reflections, limit=1, minimum_score=0.8)

        if not success_experience and search_like:
            fallback_workflow = (
                _select(scored_success_experience, limit=1, minimum_score=-0.25) if not success_evidence else []
            )
            for text in fallback_workflow:
                if text not in success_experience:
                    success_experience.append(text)

        if not success_evidence and scored_success_evidence:
            success_evidence = _fallback_select(
                scored_success_evidence,
                limit=min(2, max_items),
                allow_conflicting_positive=not search_like,
            )
        if not success_experience and scored_success_experience:
            success_experience = _fallback_select(scored_success_experience, limit=1)
        if not failure_reflections and scored_failure_reflections:
            failure_reflections = _fallback_select(scored_failure_reflections, limit=1)

        if not failure_reflections and not success_experience and not success_evidence:
            return ""

        kept_scores = [
            score
            for score, _idx, text, _channel in scored_success_evidence + scored_success_experience
            if text in set(success_evidence + success_experience)
        ]
        if kept_scores:
            strength = _strength_label(max(kept_scores), len(kept_scores))
        elif route_score_values and route_sections:
            strength = _strength_label(max(route_score_values), len(route_sections))
        elif failure_reflections:
            strength = "weak"
        else:
            strength = "none"

        lines: list[str] = []
        lines.append("Memory evidence is supplementary context only.")
        lines.append("Do not treat memory as an action recommendation.")
        lines.append("Successful experience is the main reference; failure-derived memory is caution only.")
        lines.append(f"Memory evidence strength: {strength}.")
        if strength == "weak":
            lines.append("Use memory only as a weak tie-break unless it matches the current state very closely.")
        elif strength == "moderate":
            lines.append("Use memory as supporting evidence when it is consistent with the current state.")
        else:
            lines.append("Use memory carefully, but current state still overrides any conflict.")
        if success_experience:
            lines.append("Successful prior experience:")
            for item in success_experience:
                lines.append(f"- {item}")
        if success_evidence:
            lines.append("Successful evidence:")
            for item in success_evidence:
                lines.append(f"- {item}")
        route_titles = {
            "grounding": "Local graph grounding:",
            "domain": "Local domain guidance:",
            "transfer": "Global transferable guidance:",
        }
        for route_name, route_lines in route_sections:
            lines.append(route_titles.get(route_name, "Memory support:"))
            for item in route_lines:
                lines.append(f"- {item}")
        if failure_reflections:
            lines.append("Failure reflections (caution, not direct instruction):")
            for item in failure_reflections:
                lines.append(f"- {item}")
        return "\n".join(lines)
