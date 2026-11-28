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

from collections import defaultdict

from .graph_types import (
    ArtifactKind,
    CandidateType,
    EdgeType,
    GlobalGraphMemory,
    LocalGraphMemory,
    MemoryQuery,
    NodeType,
    SupportBundle,
    SupportItem,
)
from .phasee_routing import route_bundle
from .retrieval_graph import (
    _candidate_rank,
    _dedupe_support_items,
    _goal_relevance_from_state,
    _query_task_family,
    _state_anchor_score,
    _task_relevance_score,
)


def _late_stage(query: MemoryQuery) -> bool:
    return (query.held_relevant_count > 0 and query.remaining_relevant_count <= 0) or query.progress_state in {
        "carry_target",
        "finalize",
    }


def _summary_mentions_goal(item: SupportItem, query: MemoryQuery) -> bool:
    lowered = item.summary.lower()
    target = str(query.goal_roles.get("object", "")).lower()
    destination = str(query.goal_roles.get("destination", "")).lower()
    return bool((target and target in lowered) or (destination and destination in lowered))


def _fact_items(query: MemoryQuery, local_memory: LocalGraphMemory, limit: int = 2) -> list[SupportItem]:
    facts: list[SupportItem] = []
    target = str(query.goal_roles.get("object", "")).lower()
    destination = str(query.goal_roles.get("destination", "")).lower()
    for node in local_memory.nodes_by_signature.values():
        if node.node_type != NodeType.STATE:
            continue
        state_relevance = _state_anchor_score(query, node.payload)
        if state_relevance <= 0.0:
            continue
        location = node.payload.get("location")
        visible = list(node.payload.get("visible_objects", []) or [])
        held = list(node.payload.get("held_objects", []) or [])
        parts: list[str] = []
        if location:
            parts.append(f"similar states are near {location}")
        visible_focus = [
            item
            for item in visible
            if (target and target in str(item).lower()) or (destination and destination in str(item).lower())
        ]
        held_focus = [
            item
            for item in held
            if (target and target in str(item).lower()) or (destination and destination in str(item).lower())
        ]
        if visible_focus:
            parts.append("relevant visible " + ", ".join(visible_focus[:3]))
        if held_focus:
            parts.append("holding relevant " + ", ".join(held_focus[:2]))
        if not parts:
            continue
        goal_relevance = _goal_relevance_from_state(query, node.payload)
        task_relevance = _task_relevance_score(
            query,
            workflow_stage=str(node.payload.get("workflow_stage") or ""),
        )
        score = 0.65 * state_relevance + 0.2 * task_relevance + 0.15 * goal_relevance
        facts.append(
            SupportItem(
                source="local_graph",
                candidate_id=f"local_fact:{node.node_id}",
                candidate_type=CandidateType.PRECONDITION,
                summary="Local state fact: " + "; ".join(parts) + ".",
                score=score,
                pattern_kind="fact",
                positive=max(node.stats.positive, 0),
                negative=max(node.stats.negative, 0),
                stalled=max(node.stats.stalled, 0),
                state_relevance=state_relevance,
                task_relevance=task_relevance,
                goal_relevance=goal_relevance,
            )
        )
    facts.sort(key=lambda item: item.score, reverse=True)
    return _dedupe_support_items(facts, limit)


def _plan_items(query: MemoryQuery, memory, source: str, limit: int = 1) -> list[SupportItem]:
    ranked: list[SupportItem] = []
    family = _query_task_family(query)
    for artifact in memory.artifacts_by_id.values():
        if artifact.kind != ArtifactKind.PROTOTYPE:
            continue
        if str(artifact.payload.get("pattern_kind", "")) != "plan":
            continue
        anchor_family = str(artifact.anchor.get("task_family", ""))
        if anchor_family and anchor_family != family:
            continue
        goal_arity = int(artifact.anchor.get("goal_arity", 1) or 1)
        if goal_arity != int(query.required_count or 1):
            continue
        summary = str(artifact.summary).strip()
        if not summary:
            continue
        task_relevance = 0.55 + (0.15 if anchor_family == family else 0.0)
        goal_relevance = 0.25
        if _summary_mentions_goal(
            SupportItem(
                source=source,
                candidate_id=artifact.artifact_id,
                candidate_type=CandidateType.WORKFLOW,
                summary=summary,
                score=0.0,
            ),
            query,
        ):
            goal_relevance += 0.15
        score = 0.55 * artifact.stats.confidence + 0.25 * task_relevance + 0.20 * goal_relevance
        if source == "local_plan":
            score += 0.06
        ranked.append(
            SupportItem(
                source=source,
                candidate_id=artifact.artifact_id,
                candidate_type=CandidateType.WORKFLOW,
                summary=summary,
                score=score,
                task_family=anchor_family,
                pattern_kind="plan",
                positive=max(artifact.stats.success, 0),
                negative=max(artifact.stats.failure, 0),
                stalled=max(artifact.stats.stalled, 0),
                task_relevance=task_relevance,
                goal_relevance=goal_relevance,
                action_patterns=tuple(str(item) for item in artifact.payload.get("action_patterns", []) or ()),
                dynamic={"plan_steps": list(artifact.payload.get("plan_steps", []) or ())},
            )
        )
    ranked.sort(key=lambda item: item.score, reverse=True)
    return _dedupe_support_items(ranked, limit)


def _scene_relation_items(query: MemoryQuery, memory, source: str, limit: int = 2) -> list[SupportItem]:
    ranked: list[SupportItem] = []
    family = _query_task_family(query)
    for artifact in memory.artifacts_by_id.values():
        if artifact.kind != ArtifactKind.PROTOTYPE:
            continue
        if str(artifact.payload.get("pattern_kind", "")) != "scene_relation":
            continue
        anchor_family = str(artifact.anchor.get("task_family", ""))
        if anchor_family and anchor_family != family:
            continue
        if str(artifact.payload.get("object_role", "")) != "target_object":
            continue
        layout_id = str(artifact.anchor.get("layout_id", ""))
        if layout_id and query.scene_id and layout_id != query.scene_id:
            continue
        relation_kind = str(artifact.payload.get("relation_kind", ""))
        summary = str(artifact.summary).strip()
        if not summary:
            continue
        score = 0.5 * artifact.stats.confidence + 0.2 * min(artifact.support / 3.0, 1.0) + 0.15
        if layout_id and query.scene_id and layout_id == query.scene_id:
            score += 0.35
        if relation_kind == "object_location_prior":
            score += 0.18
        elif relation_kind == "searched_empty":
            score += 0.08
        source_instance = str(artifact.payload.get("source_instance", ""))
        source_base = str(artifact.payload.get("source_base", ""))
        if source_instance:
            score += 0.12
        if source_base in {"tvstand", "dresser", "sofa"}:
            score += 0.10
        ranked.append(
            SupportItem(
                source=source,
                candidate_id=artifact.artifact_id,
                candidate_type=(
                    CandidateType.PRECONDITION if relation_kind == "object_location_prior" else CandidateType.FAILURE
                ),
                summary=summary,
                score=score,
                task_family=anchor_family,
                pattern_kind="scene_relation",
                positive=max(artifact.stats.success, 0),
                negative=max(artifact.stats.failure, 0),
                stalled=max(artifact.stats.stalled, 0),
                state_relevance=0.0,
                task_relevance=0.45,
                goal_relevance=0.45,
                action_patterns=tuple(str(item) for item in artifact.payload.get("action_patterns", []) or ()),
                dynamic={
                    "relation_kind": relation_kind,
                    "source_role": str(artifact.payload.get("source_role", "")),
                    "source_base": str(artifact.payload.get("source_base", "")),
                    "source_instance": str(artifact.payload.get("source_instance", "")),
                    "layout_id": layout_id,
                },
            )
        )
    ranked.sort(key=lambda item: item.score, reverse=True)
    return _dedupe_support_items(ranked, limit)


def _local_workflow_from_graph(query: MemoryQuery, local_memory: LocalGraphMemory, limit: int = 2) -> list[SupportItem]:
    if not local_memory.nodes_by_signature or not local_memory.edges_by_signature:
        return []
    node_by_id = {node.node_id: node for node in local_memory.nodes_by_signature.values()}
    outgoing: dict[str, list] = defaultdict(list)
    for edge in local_memory.edges_by_signature.values():
        outgoing[edge.src].append(edge)

    ranked: list[SupportItem] = []
    for node in node_by_id.values():
        if node.node_type != NodeType.STATE:
            continue
        anchor = _state_anchor_score(query, node.payload)
        if anchor <= 0.0:
            continue
        for edge in outgoing.get(node.node_id, []):
            if edge.edge_type != EdgeType.TEMPORAL:
                continue
            action_node = node_by_id.get(edge.dst)
            if action_node is None or action_node.node_type != NodeType.ACTION:
                continue
            for next_edge in outgoing.get(action_node.node_id, []):
                if next_edge.edge_type != EdgeType.ADVANCES_TO:
                    continue
                dst_node = node_by_id.get(next_edge.dst)
                if dst_node is None:
                    continue
                next_stage = dst_node.signature
                current = query.current_stage or "current"
                summary = f"Workflow pattern: {current} -> {next_stage}."
                task_relevance = _task_relevance_score(
                    query,
                    text=summary,
                    pattern_kind="workflow",
                    workflow_stage=next_stage,
                )
                score = 0.7 * anchor + 0.3 * task_relevance + 0.15 * next_edge.stats.confidence
                ranked.append(
                    SupportItem(
                        source="local_graph",
                        candidate_id=f"local_workflow:{action_node.node_id}:{next_stage}",
                        candidate_type=CandidateType.WORKFLOW,
                        summary=summary,
                        score=score,
                        pattern_kind="workflow",
                        positive=max(next_edge.stats.positive, 0),
                        negative=max(next_edge.stats.negative, 0),
                        stalled=max(next_edge.stats.stalled, 0),
                        state_relevance=anchor,
                        task_relevance=task_relevance,
                    )
                )
    ranked.sort(key=lambda item: item.score, reverse=True)
    return _dedupe_support_items(ranked, limit)


class PhaseECompatRetriever:
    """
    Candidate-centric retrieval layer that keeps the current memory file format
    but uses a much lighter, phaseE-like decision surface:
    - local/global candidates are the primary evidence
    - local graph only contributes coarse facts and workflow continuation
    - no local action suggestions are surfaced into the prompt
    """

    def __init__(self, local_top_k: int = 4, global_top_k: int = 3) -> None:
        self.local_top_k = local_top_k
        self.global_top_k = global_top_k

    def retrieve(
        self,
        query: MemoryQuery,
        local_memory: LocalGraphMemory,
        global_memory: GlobalGraphMemory,
    ) -> SupportBundle:
        bundle = SupportBundle(query=f"query:{query.goal}")
        bundle.goal_object = str(query.goal_roles.get("object", ""))
        bundle.goal_destination = str(query.goal_roles.get("destination", ""))
        bundle.goal_tool = str(query.goal_roles.get("tool", ""))
        bundle.progress_state = str(query.progress_state or "")

        local_candidates = _candidate_rank(query, local_memory.candidates.values(), source="local_pattern")
        global_candidates = _candidate_rank(query, global_memory.candidates.values(), source="global_motif")
        local_graph_workflows = _local_workflow_from_graph(query, local_memory, limit=2)
        local_facts = _fact_items(query, local_memory, limit=2)
        local_plans = _plan_items(query, local_memory, source="local_plan", limit=1)
        global_plans = _plan_items(query, global_memory, source="global_plan", limit=1)
        local_relations = _scene_relation_items(query, local_memory, source="local_relation", limit=2)
        global_relations = _scene_relation_items(query, global_memory, source="global_relation", limit=1)

        # Keep local memory primary. Global is supplementary and slightly more
        # conservative unless it matches the task family.
        family = _query_task_family(query)
        for item in local_candidates:
            if item.pattern_kind == "failure":
                item.score -= 0.1
            if item.pattern_kind == "closure" and not _late_stage(query):
                item.score -= 0.12
            if item.candidate_type == CandidateType.PRECONDITION:
                item.score -= 0.18
                if "at_target_before_open" in item.summary.lower():
                    item.score -= 0.22
                if _summary_mentions_goal(item, query):
                    item.score += 0.08
        for item in global_candidates:
            item.score -= 0.08
            if item.task_family and item.task_family != family:
                item.score -= 0.12
            if item.pattern_kind == "closure" and not _late_stage(query):
                item.score -= 0.18
            if item.candidate_type == CandidateType.PRECONDITION:
                item.score -= 0.24
                if "at_target_before_open" in item.summary.lower():
                    item.score -= 0.28
                if _summary_mentions_goal(item, query):
                    item.score += 0.08

        local_pool = sorted(
            local_candidates + local_graph_workflows + local_facts + local_plans + local_relations,
            key=lambda item: item.score,
            reverse=True,
        )
        global_pool = sorted(
            global_candidates + global_plans + global_relations, key=lambda item: item.score, reverse=True
        )

        bundle.local_items = _dedupe_support_items(local_pool, self.local_top_k + 4)
        bundle.global_items = _dedupe_support_items(global_pool, self.global_top_k + 2)
        route_bundle(query, bundle)
        return bundle
