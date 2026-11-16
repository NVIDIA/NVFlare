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

from .graph_types import CandidateType, MemoryQuery, SupportBundle, SupportItem


def _route_stage(query: MemoryQuery) -> str:
    if query.failure_label:
        return "failure"
    if (
        query.held_relevant_count > 0
        or query.progress_state in {"carry_target", "finalize"}
        or (query.held_relevant_count > 0 and query.remaining_relevant_count <= 0)
    ):
        return "carry"
    if int(query.required_count or 1) > 1 and query.placed_relevant_count > 0 and query.remaining_relevant_count > 0:
        return "search_next"
    return "search"


def _late_stage(query: MemoryQuery) -> bool:
    return (query.held_relevant_count > 0 and query.remaining_relevant_count <= 0) or query.progress_state in {
        "carry_target",
        "finalize",
    }


def _dedupe(items: list[SupportItem], limit: int) -> list[SupportItem]:
    out: list[SupportItem] = []
    seen: set[str] = set()
    seen_summary: set[str] = set()
    for item in sorted(items, key=lambda it: it.score, reverse=True):
        if item.candidate_id in seen or item.summary in seen_summary:
            continue
        seen.add(item.candidate_id)
        seen_summary.add(item.summary)
        out.append(item)
        if len(out) >= limit:
            break
    return out


def _pick_workflow(query: MemoryQuery, items: list[SupportItem], limit: int = 2) -> list[SupportItem]:
    selected = [
        item
        for item in items
        if item.candidate_type == CandidateType.WORKFLOW and item.pattern_kind in {"workflow", "closure", ""}
    ]
    if not _late_stage(query):
        selected = [item for item in selected if item.pattern_kind != "closure"]
    return _dedupe(selected, limit)


def _pick_plans(items: list[SupportItem], limit: int = 1) -> list[SupportItem]:
    selected = [item for item in items if item.pattern_kind == "plan"]
    return _dedupe(selected, limit)


def _pick_relations(items: list[SupportItem], limit: int = 2) -> list[SupportItem]:
    selected = [item for item in items if item.pattern_kind == "scene_relation"]
    return _dedupe(selected, limit)


def _goal_focused_relations(query: MemoryQuery, items: list[SupportItem], limit: int = 1) -> list[SupportItem]:
    target = str(query.goal_roles.get("object", "")).lower()
    destination = str(query.goal_roles.get("destination", "")).lower()
    selected = []
    for item in items:
        if item.pattern_kind != "scene_relation":
            continue
        lowered = item.summary.lower()
        if (target and target in lowered) or (destination and destination in lowered):
            selected.append(item)
    return _dedupe(selected, limit)


def _pick_preconditions(items: list[SupportItem], limit: int = 1) -> list[SupportItem]:
    selected = [
        item
        for item in items
        if item.candidate_type == CandidateType.PRECONDITION
        and item.pattern_kind != "action"
        and not item.summary.lower().startswith("in similar states, try ")
        and "at_target_before_open" not in item.summary.lower()
    ]
    return _dedupe(selected, limit)


def _pick_repairs(query: MemoryQuery, items: list[SupportItem], limit: int = 1) -> list[SupportItem]:
    if not query.failure_label:
        return []
    selected = [item for item in items if item.candidate_type == CandidateType.REPAIR]
    return _dedupe(selected, limit)


def _pick_closure(query: MemoryQuery, items: list[SupportItem], limit: int = 1) -> list[SupportItem]:
    if not _late_stage(query):
        return []
    selected = [item for item in items if item.pattern_kind == "closure"]
    return _dedupe(selected, limit)


def route_bundle(query: MemoryQuery, bundle: SupportBundle) -> None:
    # Strong local-first behavior, close to phaseD/E:
    # short support, stage-specific slots, local relations first.
    local = bundle.local_items
    global_ = bundle.global_items
    stage = _route_stage(query)

    search_stage = stage in {"search", "search_next"}

    local_relation = _goal_focused_relations(query, local, limit=1 if search_stage else 0)
    if not local_relation:
        local_relation = _pick_relations(local, limit=1 if search_stage else 0)
    global_relation = _goal_focused_relations(query, global_, limit=1 if search_stage and not local_relation else 0)
    if not global_relation and not local_relation:
        global_relation = _pick_relations(global_, limit=1 if search_stage else 0)
    bundle.relation_items = _dedupe(local_relation + global_relation, 1 if search_stage else 0)

    local_plan = _pick_plans(local, limit=1)
    global_plan = _pick_plans(global_, limit=1 if not local_plan else 0)
    bundle.plan_items = _dedupe(local_plan + global_plan, 1 if stage != "failure" else 0)

    local_workflow = _pick_workflow(query, local, limit=1 if search_stage and not bundle.plan_items else 0)
    global_workflow = _pick_workflow(
        query, global_, limit=1 if search_stage and not local_workflow and not bundle.plan_items else 0
    )
    bundle.workflow_items = _dedupe(
        local_workflow + global_workflow, 1 if search_stage and not bundle.plan_items else 0
    )

    # For search we still want one lightweight location prior when no stronger
    # scene relation is available, but facts should stay concise and goal-aware.
    bundle.fact_items = _dedupe(
        [item for item in local if item.pattern_kind == "fact"],
        1 if search_stage and not bundle.relation_items else (0 if search_stage else 1),
    )

    if search_stage or stage == "failure":
        local_pre = []
        global_pre = []
    else:
        local_pre = _pick_preconditions(local, limit=1 if stage == "carry" else 0)
        global_pre = _pick_preconditions(global_, limit=1 if stage == "carry" and not local_pre else 0)
    bundle.precondition_items = _dedupe(local_pre + global_pre, 1 if stage == "carry" else 0)

    bundle.repair_items = _dedupe(
        _pick_repairs(query, local, 1 if stage == "failure" else 0)
        + _pick_repairs(query, global_, 1 if stage == "failure" else 0),
        1 if stage == "failure" else 0,
    )
    bundle.closure_items = _dedupe(
        _pick_closure(query, local, 1 if stage == "carry" else 0)
        + _pick_closure(query, global_, 1 if stage == "carry" else 0),
        1 if stage == "carry" else 0,
    )

    blocked = [item for item in local + global_ if item.candidate_type == CandidateType.FAILURE]
    bundle.blocked_actions = []
    for item in blocked[:2]:
        lowered = item.summary.lower()
        if "fails under" in lowered:
            bundle.blocked_actions.append(lowered.split(" fails under", 1)[0].replace(".", "").strip())
    bundle.blocked_actions = list(dict.fromkeys(bundle.blocked_actions))

    bundle.suggested_actions = []
    ordered_hint_items = []
    if search_stage:
        ordered_hint_items.extend(bundle.plan_items)
        ordered_hint_items.extend(bundle.workflow_items)
    elif stage == "carry":
        ordered_hint_items.extend(bundle.closure_items)
        ordered_hint_items.extend(bundle.plan_items)
    else:
        ordered_hint_items.extend(bundle.repair_items)
        ordered_hint_items.extend(bundle.plan_items)
    bundle.workflow_hints = list(
        dict.fromkeys(
            item.summary.replace("Workflow pattern: ", "").replace("Closure pattern: ", "").replace("Plan: ", "")
            for item in ordered_hint_items
        )
    )[:2]

    bundle.routing_decisions = {
        "stage": stage,
        "fact": "local" if bundle.fact_items else "none",
        "relation": "mixed" if (local_relation and global_relation) else ("local" if bundle.relation_items else "none"),
        "plan": "mixed" if (local_plan and global_plan) else ("local" if bundle.plan_items else "none"),
        "workflow": "mixed" if (local_workflow and global_workflow) else ("local" if bundle.workflow_items else "none"),
        "precondition": "mixed" if (local_pre and global_pre) else ("local" if bundle.precondition_items else "none"),
        "repair": "local" if bundle.repair_items else "none",
        "closure": (
            "mixed"
            if (bundle.closure_items and any(item in global_ for item in bundle.closure_items))
            else ("local" if bundle.closure_items else "none")
        ),
        "blocked": "local" if bundle.blocked_actions else "none",
    }
    bundle.routing_weights = {
        slot: {
            "local": 1.0 if decision in {"local", "mixed"} else 0.0,
            "global": 0.35 if decision == "mixed" else 0.0,
            "none": 1.0 if decision == "none" else 0.0,
        }
        for slot, decision in bundle.routing_decisions.items()
    }
