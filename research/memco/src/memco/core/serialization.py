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

from .graph_types import (
    ArtifactKind,
    ArtifactStats,
    CandidateType,
    EdgeType,
    GlobalGraphMemory,
    GraphEdge,
    GraphNode,
    LocalGraphMemory,
    MemoryArtifact,
    MemoryRule,
    NodeType,
    PromotionCandidate,
    RuleStats,
    RuleType,
)


def candidate_from_dict(payload: dict) -> PromotionCandidate:
    candidate = PromotionCandidate(
        candidate_id=payload["candidate_id"],
        candidate_type=CandidateType(payload["candidate_type"]),
        summary=payload["summary"],
        structure=payload.get("structure", {}),
    )
    candidate.source_episode_ids = set(payload.get("source_episode_ids", []))
    candidate.source_scenes = set(payload.get("source_scenes", []))
    candidate.positive = int(payload.get("positive", 0))
    candidate.negative = int(payload.get("negative", 0))
    candidate.stalled = int(payload.get("stalled", 0))
    candidate.utility = float(payload.get("utility", 0.0))
    candidate.wilson_episode_evidence = {
        str(episode_id): {
            "supporting": int((verdict or {}).get("supporting", 0)),
            "contradicting": int((verdict or {}).get("contradicting", 0)),
            "stalled": int((verdict or {}).get("stalled", 0)),
        }
        for episode_id, verdict in (payload.get("wilson_episode_evidence", {}) or {}).items()
    }
    return candidate


def rule_from_dict(payload: dict) -> MemoryRule:
    rule = MemoryRule(
        rule_id=payload["rule_id"],
        rule_type=RuleType(payload["rule_type"]),
        summary=payload["summary"],
        task_family=str(payload.get("task_family", "")),
        goal_arity=int(payload.get("goal_arity", 1)),
        progress_state=str(payload.get("progress_state", "")),
        goal_roles=dict(payload.get("goal_roles", {}) or {}),
        condition=dict(payload.get("condition", {}) or {}),
        effect=dict(payload.get("effect", {}) or {}),
    )
    rule.source_episode_ids = set(payload.get("source_episode_ids", []))
    rule.source_scenes = set(payload.get("source_scenes", []))
    rule.specificity = float(payload.get("specificity", 0.0))
    rule.conflict = float(payload.get("conflict", 0.0))
    rule.wilson_episode_evidence = {
        str(episode_id): {
            "supporting": int((verdict or {}).get("supporting", 0)),
            "contradicting": int((verdict or {}).get("contradicting", 0)),
            "stalled": int((verdict or {}).get("stalled", 0)),
        }
        for episode_id, verdict in (payload.get("wilson_episode_evidence", {}) or {}).items()
    }
    stats_blob = payload.get("stats", {})
    rule.stats = RuleStats(
        support=int(stats_blob.get("support", 0)),
        success=int(stats_blob.get("success", 0)),
        failure=int(stats_blob.get("failure", 0)),
        stalled=int(stats_blob.get("stalled", 0)),
        utility=float(stats_blob.get("utility", 0.0)),
        transfer_success=int(stats_blob.get("transfer_success", 0)),
        transfer_trials=int(stats_blob.get("transfer_trials", 0)),
    )
    return rule


def artifact_from_dict(payload: dict) -> MemoryArtifact:
    artifact = MemoryArtifact(
        artifact_id=payload["artifact_id"],
        kind=ArtifactKind(payload["kind"]),
        summary=payload["summary"],
        anchor=dict(payload.get("anchor", {}) or {}),
        payload=dict(payload.get("payload", {}) or {}),
        level=str(payload.get("level", "local")),
    )
    artifact.source_episode_ids = set(payload.get("source_episode_ids", []))
    artifact.source_scenes = set(payload.get("source_scenes", []))
    artifact.specificity = float(payload.get("specificity", 0.0))
    artifact.conflict = float(payload.get("conflict", 0.0))
    artifact.wilson_episode_evidence = {
        str(episode_id): {
            "supporting": int((verdict or {}).get("supporting", 0)),
            "contradicting": int((verdict or {}).get("contradicting", 0)),
            "stalled": int((verdict or {}).get("stalled", 0)),
        }
        for episode_id, verdict in (payload.get("wilson_episode_evidence", {}) or {}).items()
    }
    stats_blob = payload.get("stats", {})
    artifact.stats = ArtifactStats(
        support=int(stats_blob.get("support", 0)),
        success=int(stats_blob.get("success", 0)),
        failure=int(stats_blob.get("failure", 0)),
        stalled=int(stats_blob.get("stalled", 0)),
        utility=float(stats_blob.get("utility", 0.0)),
        transfer_success=int(stats_blob.get("transfer_success", 0)),
        transfer_trials=int(stats_blob.get("transfer_trials", 0)),
    )
    return artifact


def load_local_memory(path: str) -> LocalGraphMemory:
    with open(path, "r", encoding="utf-8") as reader:
        payload = json.load(reader)
    memory = LocalGraphMemory(agent_id=payload["agent_id"])
    memory.episode_ids = list(payload.get("episode_ids", []))
    for node_blob in payload.get("nodes", []):
        node = GraphNode(
            node_id=node_blob["node_id"],
            node_type=NodeType(node_blob["node_type"]),
            signature=node_blob["signature"],
            payload=dict(node_blob.get("payload", {})),
        )
        stats_blob = node_blob.get("stats", {})
        node.stats.support = int(stats_blob.get("support", 0))
        node.stats.positive = int(stats_blob.get("positive", 0))
        node.stats.negative = int(stats_blob.get("negative", 0))
        node.stats.stalled = int(stats_blob.get("stalled", 0))
        memory.nodes_by_signature[node.signature] = node
    for edge_blob in payload.get("edges", []):
        edge = GraphEdge(
            src=edge_blob["src"],
            dst=edge_blob["dst"],
            edge_type=EdgeType(edge_blob["edge_type"]),
            signature=edge_blob["signature"],
            payload=dict(edge_blob.get("payload", {})),
        )
        stats_blob = edge_blob.get("stats", {})
        edge.stats.support = int(stats_blob.get("support", 0))
        edge.stats.positive = int(stats_blob.get("positive", 0))
        edge.stats.negative = int(stats_blob.get("negative", 0))
        edge.stats.stalled = int(stats_blob.get("stalled", 0))
        memory.edges_by_signature[edge.signature] = edge
    for candidate_blob in payload.get("candidates", []):
        candidate = candidate_from_dict(candidate_blob)
        memory.candidates[candidate.candidate_id] = candidate
    for rule_blob in payload.get("rules", []):
        rule = rule_from_dict(rule_blob)
        memory.rules_by_id[rule.rule_id] = rule
    for artifact_blob in payload.get("artifacts", []):
        artifact = artifact_from_dict(artifact_blob)
        memory.artifacts_by_id[artifact.artifact_id] = artifact
    return memory


def load_global_memory(path: str) -> GlobalGraphMemory:
    with open(path, "r", encoding="utf-8") as reader:
        payload = json.load(reader)
    memory = GlobalGraphMemory()
    memory.promoted_batches = list(payload.get("promoted_batches", []))
    for candidate_blob in payload.get("candidates", []):
        candidate = candidate_from_dict(candidate_blob)
        memory.candidates[candidate.candidate_id] = candidate
    for rule_blob in payload.get("rules", []):
        rule = rule_from_dict(rule_blob)
        memory.rules_by_id[rule.rule_id] = rule
    for artifact_blob in payload.get("artifacts", []):
        artifact = artifact_from_dict(artifact_blob)
        memory.artifacts_by_id[artifact.artifact_id] = artifact
    return memory


def empty_local(agent_id: str) -> LocalGraphMemory:
    return LocalGraphMemory(agent_id=agent_id)


def empty_global() -> GlobalGraphMemory:
    return GlobalGraphMemory()


def _candidate_to_dict(candidate: PromotionCandidate) -> dict:
    return {
        "candidate_id": candidate.candidate_id,
        "candidate_type": candidate.candidate_type.value,
        "summary": candidate.summary,
        "structure": dict(candidate.structure),
        "source_episode_ids": sorted(candidate.source_episode_ids),
        "source_scenes": sorted(candidate.source_scenes),
        "positive": candidate.positive,
        "negative": candidate.negative,
        "stalled": candidate.stalled,
        "utility": candidate.utility,
        **({"wilson_episode_evidence": candidate.wilson_episode_evidence} if candidate.wilson_episode_evidence else {}),
        "confidence": candidate.confidence,
        "coverage": candidate.coverage,
        "prior_score": candidate.prior_score,
    }


def _rule_to_dict(rule: MemoryRule) -> dict:
    return {
        "rule_id": rule.rule_id,
        "rule_type": rule.rule_type.value,
        "summary": rule.summary,
        "task_family": rule.task_family,
        "goal_arity": rule.goal_arity,
        "progress_state": rule.progress_state,
        "goal_roles": dict(rule.goal_roles),
        "condition": dict(rule.condition),
        "effect": dict(rule.effect),
        "source_episode_ids": sorted(rule.source_episode_ids),
        "source_scenes": sorted(rule.source_scenes),
        "specificity": rule.specificity,
        "conflict": rule.conflict,
        **({"wilson_episode_evidence": rule.wilson_episode_evidence} if rule.wilson_episode_evidence else {}),
        "stats": {
            "support": rule.stats.support,
            "success": rule.stats.success,
            "failure": rule.stats.failure,
            "stalled": rule.stats.stalled,
            "utility": rule.stats.utility,
            "transfer_success": rule.stats.transfer_success,
            "transfer_trials": rule.stats.transfer_trials,
            "confidence": rule.stats.confidence,
            "transfer_rate": rule.stats.transfer_rate,
            "utility_avg": rule.stats.utility_avg,
            "promotion_score": rule.promotion_score,
        },
    }


def _artifact_to_dict(artifact: MemoryArtifact) -> dict:
    return {
        "artifact_id": artifact.artifact_id,
        "kind": artifact.kind.value,
        "summary": artifact.summary,
        "anchor": dict(artifact.anchor),
        "payload": dict(artifact.payload),
        "level": artifact.level,
        "source_episode_ids": sorted(artifact.source_episode_ids),
        "source_scenes": sorted(artifact.source_scenes),
        "specificity": artifact.specificity,
        "conflict": artifact.conflict,
        **({"wilson_episode_evidence": artifact.wilson_episode_evidence} if artifact.wilson_episode_evidence else {}),
        "stats": {
            "support": artifact.stats.support,
            "success": artifact.stats.success,
            "failure": artifact.stats.failure,
            "stalled": artifact.stats.stalled,
            "utility": artifact.stats.utility,
            "transfer_success": artifact.stats.transfer_success,
            "transfer_trials": artifact.stats.transfer_trials,
            "confidence": artifact.stats.confidence,
            "failure_rate": artifact.stats.failure_rate,
            "transfer_rate": artifact.stats.transfer_rate,
            "utility_avg": artifact.stats.utility_avg,
            "promotion_score": artifact.promotion_score,
        },
    }


def local_memory_from_dict(payload: dict) -> LocalGraphMemory:
    """Deserialize local memory without requiring an intermediate file."""
    memory = LocalGraphMemory(agent_id=payload["agent_id"])
    memory.episode_ids = list(payload.get("episode_ids", []))
    for candidate_payload in payload.get("candidates", []):
        candidate = candidate_from_dict(candidate_payload)
        memory.candidates[candidate.candidate_id] = candidate
    for rule_payload in payload.get("rules", []):
        rule = rule_from_dict(rule_payload)
        memory.rules_by_id[rule.rule_id] = rule
    for artifact_payload in payload.get("artifacts", []):
        artifact = artifact_from_dict(artifact_payload)
        memory.artifacts_by_id[artifact.artifact_id] = artifact
    return memory


def global_memory_to_dict(memory: GlobalGraphMemory) -> dict:
    """Serialize global memory into an NVFlare-safe primitive payload."""
    return {
        "promoted_batches": list(memory.promoted_batches),
        "candidate_count": len(memory.candidates),
        "rule_count": len(memory.rules_by_id),
        "artifact_count": len(memory.artifacts_by_id),
        "candidates": [_candidate_to_dict(item) for item in memory.candidates.values()],
        "rules": [_rule_to_dict(item) for item in memory.rules_by_id.values()],
        "artifacts": [_artifact_to_dict(item) for item in memory.artifacts_by_id.values()],
    }
