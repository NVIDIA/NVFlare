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
import re
from typing import Any

from .graph_types import ActionFamily, CanonicalAction, Domain, EpisodeRecord, EpisodeStep, StateSummary, StepFeedback
from .scene_filter import extract_scene_num_from_game_path, scene_type_from_scene_num


def _normalize_token(text: str) -> str:
    return re.sub(r"\s+", "_", text.strip().lower())


def _denormalize_token(text: str) -> str:
    return text.strip().lower().replace("_", " ")


def _split_seen_items(text: str) -> tuple[str, ...]:
    if "you see" not in text.lower():
        return ()
    seen = text.lower().split("you see", 1)[-1]
    seen = seen.replace(".", " ").replace(" and ", ", ")
    items = []
    for part in seen.split(","):
        item = _normalize_token(part)
        if item and item not in {"nothing", "nothing_", "nothing_here"}:
            items.append(item)
    return tuple(dict.fromkeys(items))


def _extract_location(text: str) -> str | None:
    lowered = text.lower()
    patterns = [
        r"you arrive at ([^.]+?)[\.,]",
        r"you are at ([^.]+?)[\.,]",
        r"on the ([^.]+?)[\.,]",
        r"the ([^.]+?) is open",
        r"the ([^.]+?) is closed",
    ]
    for pattern in patterns:
        match = re.search(pattern, lowered)
        if match:
            return _normalize_token(match.group(1))
    if "middle of a room" in lowered:
        return "room_center"
    return None


def _extract_container_states(text: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    lowered = text.lower()
    open_matches = tuple(_normalize_token(m) for m in re.findall(r"the ([^.]+?) is open", lowered))
    closed_matches = tuple(_normalize_token(m) for m in re.findall(r"the ([^.]+?) is closed", lowered))
    return tuple(dict.fromkeys(open_matches)), tuple(dict.fromkeys(closed_matches))


def _parse_task_descriptor(descriptor: str) -> dict[str, str]:
    token = descriptor.strip().split("::", 1)[0].strip()
    parts = token.split("-")
    if len(parts) < 4:
        return {}
    family = parts[0]
    raw_object = parts[1]
    raw_aux = parts[2]
    raw_destination = parts[3]
    slots: dict[str, str] = {}
    if raw_object and raw_object.lower() != "none":
        slots["object"] = _normalize_token(raw_object)
    if raw_destination and raw_destination.lower() != "none":
        if family == "look_at_obj_in_light":
            slots["tool"] = _normalize_token(raw_destination)
        else:
            slots["destination"] = _normalize_token(raw_destination)
    if raw_aux and raw_aux.lower() != "none":
        slots["aux"] = _normalize_token(raw_aux)
    slots["family"] = family
    return slots


def _canonical_goal_from_descriptor(descriptor: str) -> str:
    parsed = _parse_task_descriptor(descriptor)
    family = parsed.get("family", "")
    obj = _denormalize_token(parsed.get("object", "object"))
    dest = _denormalize_token(parsed.get("destination", "destination"))
    tool = _denormalize_token(parsed.get("tool", "tool"))
    if family == "pick_two_obj_and_place":
        return f"find two {obj} and put them in {dest}."
    if family == "pick_heat_then_place_in_recep":
        return f"heat some {obj} and put it in {dest}."
    if family == "pick_cool_then_place_in_recep":
        return f"cool some {obj} and put it in {dest}."
    if family == "pick_clean_then_place_in_recep":
        return f"clean some {obj} and put it in {dest}."
    if family == "look_at_obj_in_light":
        return f"look at {obj} under the {tool}."
    return f"put some {obj} on {dest}."


def _parse_goal_slots(goal: str) -> dict[str, str]:
    lowered = goal.lower().strip().strip(".")
    if not lowered:
        return {}
    slots: dict[str, str] = {}
    if "-" in lowered and " " not in lowered.split("-", 1)[0]:
        descriptor_slots = _parse_task_descriptor(lowered)
        for key in ("object", "destination", "tool"):
            if key in descriptor_slots:
                slots[key] = descriptor_slots[key]
        if slots:
            return slots

    match = re.search(
        r"(?:heat|cool|clean) some ([a-z0-9 ]+) and put it (?:in|on) ([a-z0-9 ]+)",
        lowered,
    )
    if match:
        slots["object"] = _normalize_token(match.group(1))
        slots["destination"] = _normalize_token(match.group(2))
        return slots

    match = re.search(r"put a (?:hot|cool|clean) ([a-z0-9 ]+?) (?:in|on) ([a-z0-9 ]+)", lowered)
    if match:
        slots["object"] = _normalize_token(match.group(1))
        slots["destination"] = _normalize_token(match.group(2))
        return slots

    match = re.search(r"(?:find|put) two ([a-z0-9 ]+?)(?: and put them)? (?:in|on) ([a-z0-9 ]+)", lowered)
    if match:
        slots["object"] = _normalize_token(match.group(1))
        slots["destination"] = _normalize_token(match.group(2))
        return slots

    goal_patterns = [
        r"put some ([a-z0-9 ]+) on ([a-z0-9 ]+)",
        r"put a ([a-z0-9 ]+) on ([a-z0-9 ]+)",
        r"put some ([a-z0-9 ]+) in ([a-z0-9 ]+)",
        r"put a ([a-z0-9 ]+) in ([a-z0-9 ]+)",
        r"put it in ([a-z0-9 ]+)",
    ]
    for pattern in goal_patterns:
        match = re.search(pattern, lowered)
        if match:
            groups = match.groups()
            if len(groups) == 2:
                slots["object"] = _normalize_token(groups[0])
                slots["destination"] = _normalize_token(groups[1])
                return slots

    match = re.search(r"(?:look at|examine(?: the)?) ([a-z0-9 ]+) (?:under the|with the) ([a-z0-9 ]+)", lowered)
    if match:
        slots["object"] = _normalize_token(match.group(1))
        slots["tool"] = _normalize_token(match.group(2))
    return slots


def _strip_article_prefix(token: str) -> str:
    for prefix in ("a_", "an_", "the_", "some_"):
        if token.startswith(prefix):
            return token[len(prefix) :]
    return token


def _base_object_token(token: str) -> str:
    normalized = _normalize_token(token)
    normalized = _strip_article_prefix(normalized)
    normalized = re.sub(r"_[0-9]+$", "", normalized)
    return normalized


_CONTAINER_TYPES = {"drawer", "cabinet", "safe", "fridge", "microwave"}


def _location_role(value: str) -> str:
    token = _base_object_token(value)
    if token in _CONTAINER_TYPES:
        return "container"
    return "support_surface"


class ALFWorldAdapter:
    def __init__(self, scene_label: str = "alfworld_scene") -> None:
        self.scene_label = scene_label

    def canonicalize_action(self, raw_action: str) -> CanonicalAction:
        raw = raw_action.strip().lower()
        patterns = [
            (r"go to (?P<target>.+)", ActionFamily.NAVIGATE, "go"),
            (r"open (?P<container>.+)", ActionFamily.CONTROL, "open"),
            (r"close (?P<container>.+)", ActionFamily.CONTROL, "close"),
            (r"examine (?P<target>.+)", ActionFamily.INSPECT, "examine"),
            (r"look", ActionFamily.SEARCH, "look"),
            (r"inventory", ActionFamily.OTHER, "inventory"),
            (r"take (?P<object>.+) from (?P<source>.+)", ActionFamily.MANIPULATE, "take"),
            (r"move (?P<object>.+) to (?P<destination>.+)", ActionFamily.MANIPULATE, "move"),
            (r"clean (?P<object>.+) with (?P<tool>.+)", ActionFamily.MANIPULATE, "clean"),
            (r"cool (?P<object>.+) with (?P<tool>.+)", ActionFamily.MANIPULATE, "cool"),
            (r"heat (?P<object>.+) with (?P<tool>.+)", ActionFamily.MANIPULATE, "heat"),
            (r"use (?P<tool>.+)", ActionFamily.CONTROL, "use"),
        ]
        for pattern, family, verb in patterns:
            match = re.fullmatch(pattern, raw)
            if match:
                slots = {k: _normalize_token(v) for k, v in match.groupdict().items()}
                return CanonicalAction(Domain.ALFWORLD, family, verb, slots, surface_form=raw_action)
        return CanonicalAction(Domain.ALFWORLD, ActionFamily.OTHER, raw.split(" ", 1)[0], surface_form=raw_action)

    def _scene_components(
        self, game_file: str | None = None, history_path: str | None = None
    ) -> tuple[str, int | None]:
        target = game_file or history_path or self.scene_label
        scene_num = extract_scene_num_from_game_path(target) if game_file else None
        if scene_num is None:
            layout_match = re.search(r"-([0-9]+)/trial_|-([0-9]+)$", target)
            if layout_match:
                raw = next(group for group in layout_match.groups() if group)
                scene_num = int(raw)
        scene_type = scene_type_from_scene_num(scene_num)
        scope = scene_type or self.scene_label
        return scope, scene_num

    def derive_scene_id(self, game_file: str | None = None, history_path: str | None = None) -> str:
        scope, _ = self._scene_components(game_file, history_path)
        return f"alfworld:{scope}"

    def derive_layout_id(self, game_file: str | None = None, history_path: str | None = None) -> str:
        scope, scene_num = self._scene_components(game_file, history_path)
        return f"alfworld:{scope}:{scene_num if scene_num is not None else 'unknown'}"

    def goal_slots(self, goal: str) -> dict[str, str]:
        return _parse_goal_slots(goal)

    def infer_task_family(self, goal: str) -> str:
        lowered = goal.lower().strip().rstrip(".")
        descriptor = _parse_task_descriptor(lowered)
        if descriptor.get("family"):
            return str(descriptor["family"])
        padded = f" {lowered} "
        if " under the " in padded or " with the " in padded:
            return "look_at_obj_in_light"
        if lowered.startswith("heat some ") or lowered.startswith("put a hot "):
            return "pick_heat_then_place_in_recep"
        if lowered.startswith("cool some ") or lowered.startswith("put a cool "):
            return "pick_cool_then_place_in_recep"
        if lowered.startswith("clean some ") or lowered.startswith("put a clean "):
            return "pick_clean_then_place_in_recep"
        if lowered.startswith("find two ") or lowered.startswith("put two "):
            return "pick_two_obj_and_place"
        if lowered.startswith("put two "):
            return "pick_two_obj_and_place"
        return "pick_and_place_simple"

    def required_goal_count(self, goal: str) -> int:
        lowered = goal.lower().strip(".")
        descriptor = _parse_task_descriptor(lowered)
        if descriptor.get("family") == "pick_two_obj_and_place":
            return 2
        if re.search(r"\bput two\b", lowered) or re.search(r"\bfind two\b", lowered) or "pick_two_obj" in lowered:
            return 2
        return 1

    def count_relevant_objects(self, objects: tuple[str, ...], target: str | None) -> int:
        if not target:
            return 0
        target_base = _base_object_token(target)
        return sum(1 for obj in objects if _base_object_token(obj) == target_base)

    def summarize_goal_progress(
        self,
        state: StateSummary,
        goal: str,
        *,
        belief: dict[str, Any] | None = None,
    ) -> dict[str, object]:
        goal_roles = self.goal_slots(goal)
        target = goal_roles.get("object")
        destination = goal_roles.get("destination")
        required_count = self.required_goal_count(goal)
        held_relevant_count = self.count_relevant_objects(state.held_objects, target)
        visible_relevant_count = self.count_relevant_objects(state.visible_objects, target)
        destination_reached = bool(destination and state.location == destination)
        fallback_placed_count = visible_relevant_count if destination_reached and held_relevant_count == 0 else 0
        if belief is not None:
            placed_estimate = int(belief.get("placed_relevant_count_est", fallback_placed_count))
            delivered_instances = belief.get("delivered_instances", []) or []
            delivered_estimate = len({str(item) for item in delivered_instances if item})
            placed_relevant_count = max(placed_estimate, delivered_estimate, 0)
        else:
            placed_relevant_count = fallback_placed_count
        placed_relevant_count = min(required_count, placed_relevant_count)
        remaining_relevant_count = max(required_count - held_relevant_count - placed_relevant_count, 0)

        if required_count > 1:
            if remaining_relevant_count <= 0 and held_relevant_count > 0:
                progress_state = "finalize"
            elif held_relevant_count > 0 and remaining_relevant_count > 0:
                progress_state = "search_second"
            elif placed_relevant_count > 0 and remaining_relevant_count > 0:
                progress_state = "search_second"
            else:
                progress_state = "search_first"
        else:
            if held_relevant_count > 0:
                progress_state = "finalize" if destination_reached else "carry_target"
            else:
                progress_state = "search"

        return {
            "goal_roles": goal_roles,
            "required_count": required_count,
            "held_relevant_count": held_relevant_count,
            "placed_relevant_count": placed_relevant_count,
            "remaining_relevant_count": remaining_relevant_count,
            "destination_reached": destination_reached,
            "goal_object_matches_visible": visible_relevant_count > 0,
            "progress_state": progress_state,
        }

    def search_reference_for_action(self, action: CanonicalAction, state: StateSummary) -> str | None:
        if action.verb == "look":
            return state.location
        if action.verb == "open":
            return action.slots.get("container")
        if action.verb == "examine":
            return action.slots.get("target")
        if action.verb == "use":
            return action.slots.get("tool")
        return None

    def detect_failure(
        self, observation: str, action: CanonicalAction, admissible: list[str] | None = None
    ) -> str | None:
        lowered = observation.lower()
        success_markers = (
            "you pick up",
            "you open",
            "you close",
            "you move",
            "you put",
            "you place",
            "you arrive at",
            "you clean",
            "you cool",
            "you heat",
            "you turn on",
        )
        if any(marker in lowered for marker in success_markers):
            return None
        if any(token in lowered for token in ["you can't", "cannot", "failed", "nothing happens"]):
            if "closed" in lowered:
                return "container_closed"
            if "don't see" in lowered or "cannot find" in lowered or "can't see" in lowered:
                return "target_not_found"
            if "not holding" in lowered:
                return "missing_possession"
            return "action_invalid"
        if action.family == ActionFamily.MANIPULATE and admissible is not None:
            normalized_admissible = {cmd.strip().lower() for cmd in admissible if cmd}
            if (
                action.surface_form
                and action.surface_form.strip().lower() not in normalized_admissible
                and "pick up" not in lowered
            ):
                return "action_not_admissible"
        return None

    def infer_subgoal(self, goal: str, state: StateSummary, action: CanonicalAction) -> str:
        goal_slots = _parse_goal_slots(goal)
        family = self.infer_task_family(goal)
        target = goal_slots.get("object")
        destination = goal_slots.get("destination")
        tool = goal_slots.get("tool")
        if family == "look_at_obj_in_light":
            if action.verb == "use":
                return "inspect_under_light"
            if target and target in state.held_objects:
                return "use_light"
            if target and any(target in obj for obj in state.visible_objects):
                return "acquire_target"
            return "locate_target"
        if target and target in state.held_objects and destination and state.location == destination:
            return "place_target"
        if target and target in state.held_objects:
            return "transport_target"
        if target and any(target in obj for obj in state.visible_objects):
            return "acquire_target"
        if action.verb in {"open", "examine"}:
            return "inspect_container"
        return "locate_target"

    def build_state(
        self,
        scene_id: str,
        observation: str,
        admissible_commands: list[str],
        held_objects: set[str],
        searched_locations: set[str],
        goal: str,
        last_action: CanonicalAction | None = None,
    ) -> StateSummary:
        location = _extract_location(observation)
        open_containers, closed_containers = _extract_container_states(observation)
        visible_objects = _split_seen_items(observation)
        admissible_verbs = tuple(
            dict.fromkeys(_normalize_token(cmd.split(" ", 1)[0]) for cmd in admissible_commands if cmd)
        )
        workflow_stage = None
        if last_action is not None:
            workflow_stage = self.infer_subgoal(
                goal,
                StateSummary(
                    domain=Domain.ALFWORLD,
                    scene_id=scene_id,
                    location=location,
                    visible_objects=visible_objects,
                    open_containers=open_containers,
                    closed_containers=closed_containers,
                    held_objects=tuple(sorted(held_objects)),
                    searched_locations=tuple(sorted(searched_locations)),
                    admissible_verbs=admissible_verbs,
                    raw_observation=observation,
                ),
                last_action,
            )
        return StateSummary(
            domain=Domain.ALFWORLD,
            scene_id=scene_id,
            location=location,
            visible_objects=visible_objects,
            open_containers=open_containers,
            closed_containers=closed_containers,
            held_objects=tuple(sorted(held_objects)),
            searched_locations=tuple(sorted(searched_locations)),
            admissible_verbs=admissible_verbs,
            workflow_stage=workflow_stage,
            raw_observation=observation,
        )

    def load_history(self, path: str) -> dict:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def episode_from_history(self, path: str, agent_id: str) -> EpisodeRecord:
        payload = self.load_history(path)
        history = payload["history"]
        scene_id = self.derive_scene_id(payload.get("game_file"), path)
        layout_id = self.derive_layout_id(payload.get("game_file"), path)
        game_name = payload.get("game_name", "task")
        task_id = f"{game_name}::{payload.get('game_index', 'run')}"
        raw_goal = str(payload.get("game_task", "") or "").strip()
        goal = raw_goal or _canonical_goal_from_descriptor(game_name)

        initial_admissible = history[0].get("Admissible Commands", [])
        held_objects: set[str] = set()
        searched_locations: set[str] = set()
        exhausted_locations: set[str] = set()
        delivered_instances: set[str] = set()
        initial_state = self.build_state(
            scene_id,
            history[0]["Observation"],
            initial_admissible,
            held_objects,
            exhausted_locations,
            goal,
        )

        steps: list[EpisodeStep] = []
        prev_state = initial_state
        previous_score = 0.0
        previous_admissible = list(initial_admissible)

        for idx, record in enumerate(history[1:], start=1):
            action = self.canonicalize_action(record["Action"])
            observation = record["Observation"]
            failure_label = self.detect_failure(observation, action, previous_admissible)
            success = failure_label is None

            if success and action.verb == "take" and "pick up" in observation.lower():
                held_objects.add(action.slots.get("object", "unknown"))
                obj = str(action.slots.get("object", "") or "")
                src = str(action.slots.get("source", "") or "")
                goal_slots = self.goal_slots(goal)
                target_obj = str(goal_slots.get("object", ""))
                goal_destination = str(goal_slots.get("destination", ""))
                if obj and target_obj and obj.startswith(target_obj) and goal_destination and src == goal_destination:
                    delivered_instances.discard(obj)
            if success and action.verb == "move" and "move" in observation.lower():
                held_objects.discard(action.slots.get("object", "unknown"))
                obj = str(action.slots.get("object", "") or "")
                dst = str(action.slots.get("destination", "") or "")
                goal_slots = self.goal_slots(goal)
                target_obj = str(goal_slots.get("object", ""))
                goal_destination = str(goal_slots.get("destination", ""))
                if obj and target_obj and obj.startswith(target_obj) and goal_destination and dst == goal_destination:
                    delivered_instances.add(obj)

            next_admissible = record.get("Admissible Commands", [])
            next_state = self.build_state(
                scene_id,
                observation,
                next_admissible,
                held_objects,
                exhausted_locations,
                goal,
                last_action=action,
            )
            if next_state.location is None:
                next_state.location = prev_state.location
            if success:
                target_object = str(self.goal_slots(goal).get("object", ""))
                target_base = _base_object_token(target_object)
                visible_target = any(
                    _base_object_token(item) == target_base for item in (next_state.visible_objects or ())
                )
                search_ref = self.search_reference_for_action(action, prev_state)
                if action.verb == "go" and next_state.location and _location_role(next_state.location) != "container":
                    search_ref = next_state.location
                if search_ref and not visible_target:
                    normalized_ref = str(search_ref).lower()
                    searched_locations.add(normalized_ref)
                    exhausted_locations.add(normalized_ref)
            delta = self._compute_state_delta(prev_state, next_state)
            score = float(record.get("Score", previous_score))
            feedback = StepFeedback(
                success=success,
                score=score,
                done=bool(record.get("Done", False)),
                failure_label=failure_label,
                state_delta=delta,
            )
            steps.append(
                EpisodeStep(
                    step_idx=idx,
                    state=prev_state,
                    action=action,
                    next_state=next_state,
                    feedback=feedback,
                    subgoal=next_state.workflow_stage,
                )
            )
            prev_state = next_state
            previous_score = score
            previous_admissible = list(next_admissible)

        return EpisodeRecord(
            agent_id=agent_id,
            scene_id=scene_id,
            task_id=task_id,
            goal=goal,
            steps=steps,
            metadata={
                "history_path": path,
                "game_file": payload.get("game_file"),
                "layout_id": layout_id,
                "status": payload.get("status"),
                "final_score": payload.get("final_score", 0.0),
                "delivered_instances_est": sorted(delivered_instances),
            },
        )

    @staticmethod
    def _compute_state_delta(prev_state: StateSummary, next_state: StateSummary) -> tuple[str, ...]:
        delta: list[str] = []
        if prev_state.location != next_state.location and next_state.location:
            delta.append(f"location={next_state.location}")
        prev_visible = set(prev_state.visible_objects)
        next_visible = set(next_state.visible_objects)
        for item in sorted(next_visible - prev_visible):
            delta.append(f"visible+={item}")
        for item in sorted(next_state.open_containers):
            if item not in prev_state.open_containers:
                delta.append(f"opened={item}")
        for item in sorted(set(next_state.held_objects) - set(prev_state.held_objects)):
            delta.append(f"holding+={item}")
        for item in sorted(set(prev_state.held_objects) - set(next_state.held_objects)):
            delta.append(f"holding-={item}")
        return tuple(delta)
