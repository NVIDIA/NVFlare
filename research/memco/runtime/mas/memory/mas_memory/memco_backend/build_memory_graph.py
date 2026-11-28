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

import argparse
import json
from pathlib import Path

from .alfworld_adapter import ALFWorldAdapter
from .construction_graph import EpisodeGraphBuilder, GlobalPromoter, LocalGraphMaintainer
from .graph_types import (
    GlobalGraphMemory,
    GraphEdge,
    GraphNode,
    LocalGraphMemory,
    MemoryArtifact,
    MemoryRule,
    PromotionCandidate,
)
from .promotion import PROMOTION_POLICIES


def _candidate_to_dict(candidate: PromotionCandidate) -> dict:
    return {
        "candidate_id": candidate.candidate_id,
        "candidate_type": candidate.candidate_type.value,
        "summary": candidate.summary,
        "structure": candidate.structure,
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


def _node_to_dict(node: GraphNode) -> dict:
    return {
        "node_id": node.node_id,
        "node_type": node.node_type.value,
        "signature": node.signature,
        "payload": node.payload,
        "stats": {
            "support": node.stats.support,
            "positive": node.stats.positive,
            "negative": node.stats.negative,
            "stalled": node.stats.stalled,
            "confidence": node.stats.confidence,
        },
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


def _edge_to_dict(edge: GraphEdge) -> dict:
    return {
        "src": edge.src,
        "dst": edge.dst,
        "edge_type": edge.edge_type.value,
        "signature": edge.signature,
        "payload": edge.payload,
        "stats": {
            "support": edge.stats.support,
            "positive": edge.stats.positive,
            "negative": edge.stats.negative,
            "stalled": edge.stats.stalled,
            "confidence": edge.stats.confidence,
        },
    }


def _local_to_dict(memory: LocalGraphMemory, include_topology: bool = False) -> dict:
    payload = {
        "agent_id": memory.agent_id,
        "episode_ids": memory.episode_ids,
        "node_count": len(memory.nodes_by_signature),
        "edge_count": len(memory.edges_by_signature),
        "candidates": [_candidate_to_dict(candidate) for candidate in memory.candidates.values()],
        "rules": [_rule_to_dict(rule) for rule in memory.rules_by_id.values()],
        "artifacts": [_artifact_to_dict(artifact) for artifact in memory.artifacts_by_id.values()],
    }
    payload["nodes"] = [_node_to_dict(node) for node in memory.nodes_by_signature.values()]
    payload["edges"] = [_edge_to_dict(edge) for edge in memory.edges_by_signature.values()]
    return payload


def _global_to_dict(memory: GlobalGraphMemory) -> dict:
    return {
        "promoted_batches": memory.promoted_batches,
        "candidate_count": len(memory.candidates),
        "rule_count": len(memory.rules_by_id),
        "artifact_count": len(memory.artifacts_by_id),
        "candidates": [_candidate_to_dict(candidate) for candidate in memory.candidates.values()],
        "rules": [_rule_to_dict(rule) for rule in memory.rules_by_id.values()],
        "artifacts": [_artifact_to_dict(artifact) for artifact in memory.artifacts_by_id.values()],
    }


def _load_protocol(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def build_from_protocol(
    protocol: dict,
    promotion_threshold: float,
    *,
    promoter: GlobalPromoter | None = None,
) -> tuple[dict[str, LocalGraphMemory], GlobalGraphMemory]:
    scenes = protocol["scenes"]
    agents = protocol["agents"]

    builder = EpisodeGraphBuilder()
    promoter = promoter or GlobalPromoter(score_threshold=promotion_threshold)
    maintainer = LocalGraphMaintainer(record_wilson_evidence=promoter.policy in {"shadow", "wilson"})

    local_memories: dict[str, LocalGraphMemory] = {}
    promotion_memories: list[LocalGraphMemory] = []

    for scene in scenes:
        adapter = ALFWorldAdapter(scene_label=scene)
        agent_id = f"agent_{scene}"
        local_memory = LocalGraphMemory(agent_id=agent_id)
        train_histories = list(agents[scene].get("train_histories", []))

        for history_path in train_histories:
            episode = adapter.episode_from_history(history_path, agent_id=agent_id)
            episode_graph = builder.build(episode)
            maintainer.update(local_memory, episode_graph, episode)
        maintainer.refine_memory(local_memory)
        local_memories[scene] = local_memory

        promo_memory = LocalGraphMemory(agent_id=f"{agent_id}_promotion")
        # Global promotion is defined over the full train history pool.
        # We intentionally do not use `promotion_histories` here; those are a
        # legacy curated subset, while transferable global prototypes should be
        # induced from all available train histories.
        for history_path in train_histories:
            episode = adapter.episode_from_history(history_path, agent_id=promo_memory.agent_id)
            episode_graph = builder.build(episode)
            maintainer.update(promo_memory, episode_graph, episode)
        maintainer.refine_memory(promo_memory)
        promotion_memories.append(promo_memory)

    global_memory = promoter.promote(
        GlobalGraphMemory(),
        promotion_memories,
        batch_name="protocol_build",
    )
    return local_memories, global_memory


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build local/global graph memory from train histories and save artifacts."
    )
    parser.add_argument("--protocol", required=True, help="Protocol manifest from graph_memory.eval build-protocol")
    parser.add_argument("--promotion-threshold", type=float, default=0.35)
    parser.add_argument("--promotion-policy", choices=PROMOTION_POLICIES, default="legacy")
    parser.add_argument("--wilson-alpha", type=float, default=0.05)
    parser.add_argument("--wilson-threshold", type=float, default=0.5)
    parser.add_argument(
        "--wilson-min-coverage",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--include-topology",
        action="store_true",
        help="Include full local nodes/edges in local_<scene>.json for visualization.",
    )
    args = parser.parse_args()

    protocol = _load_protocol(args.protocol)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    promoter = GlobalPromoter(
        score_threshold=args.promotion_threshold,
        policy=args.promotion_policy,
        wilson_alpha=args.wilson_alpha,
        wilson_threshold=args.wilson_threshold,
    )
    local_memories, global_memory = build_from_protocol(
        protocol,
        args.promotion_threshold,
        promoter=promoter,
    )
    agents = protocol.get("agents", {})
    global_promotion_source_by_scene = {scene: len(agent.get("train_histories", [])) for scene, agent in agents.items()}
    global_promotion_source_episodes = sum(global_promotion_source_by_scene.values())

    for scene, memory in local_memories.items():
        with open(output_dir / f"local_{scene}.json", "w", encoding="utf-8") as handle:
            json.dump(_local_to_dict(memory, include_topology=args.include_topology), handle, indent=2)

    with open(output_dir / "global_memory.json", "w", encoding="utf-8") as handle:
        json.dump(_global_to_dict(global_memory), handle, indent=2)

    if promoter.last_promotion_report:
        with open(output_dir / "promotion_wilson.json", "w", encoding="utf-8") as handle:
            json.dump(promoter.last_promotion_report, handle, indent=2)
    if args.promotion_policy == "shadow" and promoter.last_wilson_memory is not None:
        with open(output_dir / "global_memory_wilson.json", "w", encoding="utf-8") as handle:
            json.dump(_global_to_dict(promoter.last_wilson_memory), handle, indent=2)

    summary = {
        "protocol": args.protocol,
        "promotion_threshold": args.promotion_threshold,
        "locals": {
            scene: {
                "episodes": len(memory.episode_ids),
                "candidates": len(memory.candidates),
                "rules": len(memory.rules_by_id),
                "artifacts": len(memory.artifacts_by_id),
                "nodes": len(memory.nodes_by_signature),
                "edges": len(memory.edges_by_signature),
            }
            for scene, memory in local_memories.items()
        },
        "global": {
            "candidate_count": len(global_memory.candidates),
            "rule_count": len(global_memory.rules_by_id),
            "artifact_count": len(global_memory.artifacts_by_id),
            "promoted_batches": global_memory.promoted_batches,
            "promotion_source_episodes": global_promotion_source_episodes,
            "promotion_source_by_scene": global_promotion_source_by_scene,
        },
    }
    if args.promotion_policy != "legacy":
        summary["promotion_policy"] = args.promotion_policy
        summary["wilson"] = {
            "alpha": args.wilson_alpha,
            "threshold": args.wilson_threshold,
            "source_coverage": "diagnostic_only",
        }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
