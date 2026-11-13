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

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Callable

from .graph_types import Domain, MemoryQuery, StateSummary, SupportBundle
from .phasee_state_policy import derive_state_policy, fuse_state_memory, score_action


@dataclass(slots=True)
class GraphPolicyDecision:
    action: str
    decision_source: str
    parsed_action: str | None = None
    selected_score: float = 0.0
    parsed_score: float | None = None
    top_scores: list[tuple[str, float]] = field(default_factory=list)


def _history_observation(env_ref: Any) -> str:
    history = list(getattr(env_ref, "current_history", []) or [])
    if history:
        return str(history[-1].get("Observation") or "")
    return str(getattr(env_ref, "initial_observation", "") or "")


def _session_from_env(
    *,
    adapter: Any,
    query: MemoryQuery,
    env_ref: Any,
    admissible_actions: list[str],
) -> Any:
    context = query.dynamic_context or {}
    current_state = StateSummary(
        domain=Domain.ALFWORLD,
        scene_id=str(query.scene_id or "alfworld:unknown"),
        location=query.location,
        visible_objects=tuple(context.get("visible_objects", ()) or ()),
        held_objects=tuple(context.get("held_objects", ()) or ()),
        searched_locations=tuple(context.get("exhausted_locations", ()) or ()),
        workflow_stage=query.current_stage,
        raw_observation=_history_observation(env_ref),
    )

    steps = []
    for row in list(getattr(env_ref, "current_history", []) or []):
        action_text = str(row.get("Action") or "").strip()
        if not action_text:
            continue
        obs = str(row.get("Observation") or "")
        failure_label = "no_effect" if "Nothing happens" in obs else None
        steps.append(
            SimpleNamespace(
                action=adapter.canonicalize_action(action_text),
                feedback=SimpleNamespace(failure_label=failure_label),
            )
        )

    return SimpleNamespace(
        adapter=adapter,
        current_state=current_state,
        _steps=steps,
        admissible_actions=admissible_actions,
    )


def select_graph_policy_action(
    *,
    adapter: Any,
    query: MemoryQuery,
    bundle: SupportBundle,
    env_ref: Any,
    raw_response: str,
    processed_action: str,
    admissible_actions: list[str],
    parser: Callable[[str, list[str]], str | None],
    recent_actions: list[str] | None = None,
    seed_bonus: float = 3.0,
    keep_parsed_margin: float = 1.0,
) -> GraphPolicyDecision:
    """Choose an exact admissible action using MemCo state policy + graph support.

    This is the nvdamas-facing analogue of the original MemCo rerank
    path: the model's output is treated as a proposal, but the final decision is
    made over the current admissible action list using state policy, fused graph
    support, and a small seed bonus for the parsed proposal.
    """
    if not admissible_actions:
        return GraphPolicyDecision(
            action=processed_action,
            decision_source="no_admissible",
        )

    session = _session_from_env(
        adapter=adapter,
        query=query,
        env_ref=env_ref,
        admissible_actions=admissible_actions,
    )
    policy = derive_state_policy(
        query=query,
        session=session,
        admissible_actions=admissible_actions,
        recent_actions=list(recent_actions or [])[-5:],
    )
    fused = fuse_state_memory(
        state_policy=policy,
        bundle=bundle,
        admissible_actions=admissible_actions,
        session=session,
    )

    parsed = parser(raw_response, admissible_actions)
    force_action = fused.get("force_action")
    if force_action in admissible_actions:
        return GraphPolicyDecision(
            action=str(force_action),
            decision_source="state_force",
            parsed_action=parsed,
            selected_score=100.0,
            parsed_score=100.0 if parsed == force_action else None,
            top_scores=[(str(force_action), 100.0)],
        )

    blocked = set(fused.get("blocked_actions", []) or [])
    scored: list[tuple[str, float]] = []
    for action in admissible_actions:
        score = score_action(
            action,
            state_policy=policy,
            bundle=bundle,
            session=session,
            query=query,
            fused_support=fused,
        )
        if parsed and action == parsed:
            score += seed_bonus
        scored.append((action, float(score)))

    valid_scored = [(action, score) for action, score in scored if action not in blocked]
    if not valid_scored:
        valid_scored = scored
    valid_scored.sort(key=lambda item: item[1], reverse=True)

    best_action, best_score = valid_scored[0]
    parsed_score = None
    if parsed:
        for action, score in scored:
            if action == parsed:
                parsed_score = score
                break
        if parsed in admissible_actions and parsed not in blocked and parsed_score is not None:
            if best_score <= parsed_score + keep_parsed_margin:
                return GraphPolicyDecision(
                    action=parsed,
                    decision_source="parsed_with_seed",
                    parsed_action=parsed,
                    selected_score=float(parsed_score),
                    parsed_score=float(parsed_score),
                    top_scores=valid_scored[:5],
                )

    return GraphPolicyDecision(
        action=best_action,
        decision_source="graph_policy_rerank",
        parsed_action=parsed,
        selected_score=float(best_score),
        parsed_score=None if parsed_score is None else float(parsed_score),
        top_scores=valid_scored[:5],
    )
