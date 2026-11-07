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
from pathlib import Path
from typing import Any

from .graph_types import ActionFamily, CanonicalAction, Domain, EpisodeRecord, EpisodeStep, StateSummary, StepFeedback


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", str(text or "").strip().lower()).strip("_")


def _short_text(text: str, *, limit: int = 180) -> str:
    text = re.sub(r"\s+", " ", str(text or "").strip())
    return text[:limit]


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _literal_texts(value: Any) -> list[str]:
    items: list[str] = []
    for item in _as_list(value):
        text = str(item or "").strip()
        if text:
            items.append(text)
    return items


class PDDLAdapter:
    """GraphMemory adapter for language-facing PDDL tasks.

    It intentionally keeps the schema domain-generic: actions become canonical
    verb/argument records, states expose compact fact tokens, and progress is
    measured by satisfied goal literals rather than ALFWorld object slots.
    """

    domain_name = "pddl"

    def canonicalize_action(self, raw_action: str) -> CanonicalAction:
        surface = str(raw_action or "").strip()
        raw = surface.lower().strip().strip(".")
        if "check" in raw and "valid" in raw:
            return CanonicalAction(Domain.PDDL, ActionFamily.SEARCH, "check_valid_actions", surface_form=surface)
        if "look around" in raw or raw == "look":
            return CanonicalAction(Domain.PDDL, ActionFamily.INSPECT, "look_around", surface_form=surface)
        tokens = re.findall(r"[a-z0-9_-]+", raw)
        if not tokens:
            return CanonicalAction(Domain.PDDL, ActionFamily.OTHER, "unknown", surface_form=surface)
        verb = tokens[0]
        slots = {f"arg{i}": _normalize(token) for i, token in enumerate(tokens[1:], start=1)}
        family = ActionFamily.CONTROL if verb in {"check", "look"} else ActionFamily.MANIPULATE
        return CanonicalAction(Domain.PDDL, family, verb, slots, surface_form=surface)

    def derive_scene_id(self, game_name: str | None = None, history_path: str | None = None) -> str:
        if game_name:
            return f"pddl:{_normalize(game_name)}"
        if history_path:
            name = Path(history_path).stem
            match = re.search(r"history_([^_]+)_", name)
            if match:
                return f"pddl:{_normalize(match.group(1))}"
        return "pddl:unknown"

    def derive_layout_id(
        self, game_name: str | None = None, problem_index: Any = None, history_path: str | None = None
    ) -> str:
        scene_id = self.derive_scene_id(game_name, history_path)
        if problem_index is None:
            return f"{scene_id}:unknown"
        return f"{scene_id}:{problem_index}"

    def infer_task_family(self, goal: str, game_name: str | None = None) -> str:
        if game_name:
            return f"pddl:{_normalize(game_name)}"
        goal_token = _normalize(goal)[:60]
        return f"pddl:{goal_token or 'task'}"

    def goal_slots(self, goal: str) -> dict[str, str]:
        conditions = self._goal_conditions(goal)
        first = _normalize(conditions[0]) if conditions else ""
        return {"object": first} if first else {}

    def required_goal_count(self, goal: str, goal_literals: list[str] | None = None) -> int:
        return max(len(goal_literals or self._goal_conditions(goal)), 1)

    def build_state(
        self,
        *,
        scene_id: str,
        observation: str,
        admissible_commands: list[str] | tuple[str, ...] = (),
        goal: str = "",
        goal_literals: list[str] | None = None,
        current_literals: list[str] | None = None,
    ) -> StateSummary:
        visible = tuple(
            dict.fromkeys(
                _normalize(x) for x in (current_literals or self._extract_fact_phrases(observation)) if _normalize(x)
            )
        )[:20]
        verbs = tuple(
            dict.fromkeys(self.canonicalize_action(cmd).verb for cmd in admissible_commands if str(cmd).strip())
        )
        progress = self._progress(
            goal=goal, goal_literals=goal_literals or [], current_literals=current_literals or [], step_count=0
        )
        return StateSummary(
            domain=Domain.PDDL,
            scene_id=scene_id,
            visible_objects=visible,
            admissible_verbs=verbs,
            workflow_stage=str(progress["progress_state"]),
            raw_observation=_short_text(observation, limit=500),
        )

    def build_query(self, *, env_ref: Any, task_config: dict, query_task: str, MemoryQuery: Any, CandidateType: Any):
        game_name = str(getattr(env_ref, "game_name", "") or task_config.get("game_name", "") or "").strip()
        goal = str(getattr(env_ref, "goal", "") or getattr(env_ref, "goal_instruction", "") or query_task or "").strip()
        if not goal:
            return None
        scene_id = self.derive_scene_id(game_name)
        goal_literals = _literal_texts(
            getattr(env_ref, "goal_literals_text", None) or getattr(env_ref, "goal_literals", None)
        )
        current_literals = _literal_texts(getattr(env_ref, "current_literals_text", None))
        admissible = [str(cmd) for cmd in (getattr(env_ref, "last_admissible_commands", []) or []) if str(cmd).strip()]
        observation = str(
            getattr(env_ref, "states", [""])[-1]
            if getattr(env_ref, "states", None)
            else getattr(env_ref, "init_obs", "") or ""
        )
        state = self.build_state(
            scene_id=scene_id,
            observation=observation,
            admissible_commands=admissible,
            goal=goal,
            goal_literals=goal_literals,
            current_literals=current_literals,
        )
        progress = self._progress(
            goal=goal,
            goal_literals=goal_literals,
            current_literals=current_literals,
            step_count=int(getattr(env_ref, "steps", 0) or 0),
            done=bool(getattr(env_ref, "done", False)),
        )
        current_norm = {_normalize(item) for item in current_literals}
        unsatisfied_goal_literals = [literal for literal in goal_literals if _normalize(literal) not in current_norm]
        desired_types = [CandidateType.PRECONDITION, CandidateType.WORKFLOW]
        if str((getattr(env_ref, "infos", {}) or {}).get("action_is_valid", "")) == "False":
            desired_types.append(CandidateType.FAILURE)
        keywords = tuple(
            dict.fromkeys(
                [
                    f"domain={game_name}",
                    f"progress={progress['progress_state']}",
                    *(_normalize(x) for x in goal_literals[:8]),
                    *(_normalize(x) for x in current_literals[:12]),
                ]
            )
        )
        return MemoryQuery(
            goal=goal,
            scene_id=scene_id,
            current_stage=state.workflow_stage,
            progress_state=str(progress["progress_state"]),
            task_family=self.infer_task_family(goal, game_name),
            goal_roles={"object": _normalize(game_name or "pddl_task")},
            required_count=int(progress["required_count"]),
            placed_relevant_count=int(progress["placed_relevant_count"]),
            remaining_relevant_count=int(progress["remaining_relevant_count"]),
            destination_reached=bool(progress["remaining_relevant_count"] <= 0),
            goal_object_matches_visible=bool(progress["placed_relevant_count"] > 0),
            admissible_actions=tuple(self.canonicalize_action(cmd) for cmd in admissible),
            desired_types=tuple(desired_types),
            failure_label=str((getattr(env_ref, "infos", {}) or {}).get("last_failure_label", "") or "") or None,
            keywords=tuple(str(k) for k in keywords if str(k).strip()),
            belief={
                "goal_literals": goal_literals,
                "current_literals": current_literals,
                "unsatisfied_goal_literals": unsatisfied_goal_literals,
                "game_name": game_name,
                "problem_index": getattr(env_ref, "problem_index", None),
            },
            dynamic_context={
                "visible_objects": list(state.visible_objects),
                "unsatisfied_goal_literals": unsatisfied_goal_literals,
                "layout_id": self.derive_layout_id(game_name, getattr(env_ref, "problem_index", None)),
                "task_config_env_name": str(task_config.get("env_name", "")),
                "memco_domain": self.domain_name,
            },
        )

    def episode_from_history(self, history_path: str, agent_id: str) -> EpisodeRecord:
        with open(history_path, "r", encoding="utf-8") as reader:
            payload = json.load(reader)
        goal = str(payload.get("game_task") or payload.get("goal") or "")
        game_name = str(payload.get("game_name") or "unknown")
        problem_index = payload.get("game_index", "unknown")
        scene_id = self.derive_scene_id(game_name, history_path)
        task_id = f"{game_name}-{problem_index}"
        history = [row for row in payload.get("history", []) if isinstance(row, dict)]
        episode = EpisodeRecord(
            agent_id=agent_id,
            scene_id=scene_id,
            task_id=task_id,
            goal=goal,
            metadata={
                "status": payload.get("status", ""),
                "final_score": float(payload.get("final_score", 0.0) or 0.0),
                "layout_id": self.derive_layout_id(game_name, problem_index, history_path),
                "memco_domain": self.domain_name,
                "task_family": self.infer_task_family(goal, game_name),
            },
        )
        goal_literals = _literal_texts(payload.get("goal_literals"))
        previous = history[0] if history else {}
        prev_state = self._state_from_record(previous, scene_id=scene_id, goal=goal, goal_literals=goal_literals)
        for idx, record in enumerate(history[1:], start=1):
            action_text = str(record.get("Action") or "").strip()
            if not action_text:
                continue
            next_state = self._state_from_record(record, scene_id=scene_id, goal=goal, goal_literals=goal_literals)
            failure_label = self._failure_label(record)
            success = failure_label is None and float(record.get("Reward", record.get("Score", 0.0)) or 0.0) >= 0
            delta = self._state_delta(prev_state, next_state)
            episode.steps.append(
                EpisodeStep(
                    step_idx=len(episode.steps),
                    state=prev_state,
                    action=self.canonicalize_action(action_text),
                    next_state=next_state,
                    feedback=StepFeedback(
                        success=bool(success),
                        score=float(record.get("Score", 0.0) or 0.0),
                        done=bool(record.get("Done", False)),
                        failure_label=failure_label,
                        state_delta=delta,
                    ),
                    subgoal=str(next_state.workflow_stage or ""),
                )
            )
            prev_state = next_state
        return episode

    def _state_from_record(
        self, record: dict[str, Any], *, scene_id: str, goal: str, goal_literals: list[str]
    ) -> StateSummary:
        current_literals = _literal_texts(record.get("Current Literals") or record.get("State Literals"))
        observation = str(record.get("Observation") or "")
        progress = self._progress(
            goal=goal,
            goal_literals=goal_literals or _literal_texts(record.get("Goal Literals")),
            current_literals=current_literals,
            step_count=int(record.get("Step", 0) or 0),
            done=bool(record.get("Done", False)),
        )
        return StateSummary(
            domain=Domain.PDDL,
            scene_id=scene_id,
            visible_objects=tuple(
                dict.fromkeys(
                    _normalize(x)
                    for x in (current_literals or self._extract_fact_phrases(observation))
                    if _normalize(x)
                )
            )[:20],
            admissible_verbs=tuple(
                dict.fromkeys(
                    self.canonicalize_action(cmd).verb for cmd in _literal_texts(record.get("Admissible Commands"))
                )
            ),
            workflow_stage=str(progress["progress_state"]),
            raw_observation=_short_text(observation, limit=500),
        )

    def _progress(
        self,
        *,
        goal: str,
        goal_literals: list[str],
        current_literals: list[str],
        step_count: int,
        done: bool = False,
    ) -> dict[str, Any]:
        goals = goal_literals or self._goal_conditions(goal)
        current_norm = {_normalize(x) for x in current_literals}
        satisfied = sum(1 for item in goals if _normalize(item) in current_norm)
        required = max(len(goals), 1)
        remaining = max(required - satisfied, 0)
        if done or remaining <= 0:
            progress_state = "goal_satisfied"
        elif satisfied <= 0 and step_count <= 0:
            progress_state = "initial_planning"
        elif satisfied <= 0:
            progress_state = "search_preconditions"
        else:
            progress_state = "advance_goal_literals"
        return {
            "required_count": required,
            "placed_relevant_count": satisfied,
            "remaining_relevant_count": remaining,
            "progress_state": progress_state,
        }

    @staticmethod
    def _goal_conditions(goal: str) -> list[str]:
        text = str(goal or "")
        if ":" in text:
            text = text.split(":", 1)[-1]
        return [piece.strip(" .") for piece in re.split(r",|;", text) if piece.strip(" .")]

    @staticmethod
    def _extract_fact_phrases(observation: str) -> list[str]:
        text = str(observation or "")
        return [piece.strip(" .") for piece in re.split(r"\.\s+|,\s+", text) if piece.strip(" .")][:20]

    @staticmethod
    def _failure_label(record: dict[str, Any]) -> str | None:
        obs = str(record.get("Observation") or "").lower()
        valid = record.get("Valid", record.get("Action Valid", None))
        if valid is False:
            return "invalid_action"
        if "not valid" in obs or "takes no effect" in obs:
            return "invalid_action"
        return None

    @staticmethod
    def _state_delta(prev_state: StateSummary, next_state: StateSummary) -> tuple[str, ...]:
        prev = set(prev_state.visible_objects)
        nxt = set(next_state.visible_objects)
        added = sorted(nxt - prev)[:6]
        removed = sorted(prev - nxt)[:6]
        deltas = [f"fact+={item}" for item in added] + [f"fact-={item}" for item in removed]
        if prev_state.workflow_stage != next_state.workflow_stage:
            deltas.append(f"stage={next_state.workflow_stage}")
        return tuple(deltas)
