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

import json
import os
import re
from dataclasses import dataclass
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any, Union

REPO_ROOT = Path(__file__).resolve().parents[2]
# Keep ALFWorld runtime assets inside the repo so imports do not depend on ~/.cache.
os.environ.setdefault("ALFWORLD_DATA", str(REPO_ROOT / "data" / "alfworld" / "_runtime_cache"))

import alfworld


def _load_alfred_tw_env() -> type:
    alfworld_root = Path(alfworld.__file__).resolve().parent
    module_path = alfworld_root / "agents" / "environment" / "alfred_tw_env.py"
    spec = spec_from_file_location("nvdamas_alfred_tw_env", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load AlfredTWEnv from {module_path}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.AlfredTWEnv


AlfredTWEnv = _load_alfred_tw_env()

from .base_env import BaseEnv, BaseRecorder

prefixes = {  # tasks: task_type
    "pick_and_place": "put",
    "pick_clean_then_place": "clean",
    "pick_heat_then_place": "heat",
    "pick_cool_then_place": "cool",
    "look_at_obj": "examine",
    "pick_two_obj": "puttwo",
}


def get_env_name_from_gamefile(gamefile: str) -> Union[str, None]:

    for k in prefixes.keys():
        if k in gamefile:
            return k
    return None


def _resolve_gamefile_with_external_root(gamefile: str, env_config: dict[str, Any]) -> str:
    value = str(gamefile or "").replace("\\", "/").strip()
    root = str((env_config or {}).get("external_game_root") or "").strip().rstrip("/")
    marker = "json_2.1.1/"
    if root and marker in value:
        suffix = value.split(marker, 1)[1]
        return str(Path(root) / suffix)
    return value


class AlfworldEnv(BaseEnv):
    def __init__(self, env_config: dict[str, Any], max_trials: int = 50):
        self.env_config = env_config
        env_type = self.env_config["env"]["type"]
        if env_type != "AlfredTWEnv":
            raise ValueError(
                f"Unsupported ALFWorld env type: {env_type}. " "This project currently supports text-only AlfredTWEnv."
            )
        self.main_env = AlfredTWEnv(self.env_config, train_eval=self.env_config["split"])

        self.max_trials: int = max_trials
        self.gamefile: str | None = None
        self.env_name: str | None = None
        self.goal_instruction: str = ""
        self.initial_observation: str = ""
        self.current_history: list[dict[str, Any]] = []
        self.last_admissible_commands: list[str] = []
        # Delay env reset until a concrete task/gamefile is selected in set_env().
        self.env = None

    def set_env(self, configs: dict) -> tuple[str, str]:
        raw_gamefile = configs["env_kwargs"]["gamefile"]
        self.gamefile = _resolve_gamefile_with_external_root(raw_gamefile, self.env_config)
        self.env_name: str = configs["env_name"]
        self.main_env.game_files = [self.gamefile]

        self.reset()
        task = configs.get("task")
        if not task or "Your task is to:" not in task:
            goal_instruction = configs.get("goal_instruction")
            if not goal_instruction:
                raise ValueError("Missing `task` or `goal_instruction` for ALFWorld task config.")
            task = f"{self.initial_observation}\n\n{goal_instruction}".strip()
        self.goal_instruction = self._extract_goal_instruction(task)
        return self._parse_task_main(task), self._parse_task_description(task)

    def reset(self):

        self.done = False
        self.won = False
        self.env = self.main_env.init_env(batch_size=1)
        observation, info = self.env.reset()
        self.initial_observation = self._normalize_observation(observation[0]) if observation else ""
        self.last_admissible_commands = self._extract_admissible_commands(info)
        self.current_history = []
        if self.initial_observation:
            self.current_history.append(
                {
                    "Observation": self.initial_observation,
                    "Admissible Commands": list(self.last_admissible_commands),
                }
            )

    def step(self, action: str) -> tuple[str, float, bool]:

        action = self.process_action(action)
        action = self._adapt_action_to_admissible(action)
        is_think_action = action.lower().startswith("think:")
        try:
            observation, reward, done, info = self.env.step([action])
        except Exception:
            if not is_think_action:
                raise
            observation = ["OK."]
            reward = [-1.0]
            done = [bool(self.done)]
            info = {"admissible_commands": [list(self.last_admissible_commands)], "won": [False]}
        observation = self._normalize_observation(observation[0])

        self.done = done[0]
        # TextWorld/ALFWorld exposes a win signal via `info['won']`.
        # `done` can be True when the max-step budget is exhausted, which is not success.
        try:
            self.won = bool(info["won"][0])
        except Exception:
            self.won = False

        if is_think_action:
            observation = "OK."
            processed_reward = -1
            self.won = False
        elif observation == "Nothing happens.":
            processed_reward = -1
            self.won = False
        else:
            processed_reward = 1 if self.won else 0
        try:
            raw_score = float(reward[0] if isinstance(reward, (list, tuple)) else reward)
        except Exception:
            raw_score = float(processed_reward)
        self.last_admissible_commands = self._extract_admissible_commands(info)
        self.current_history.append(
            {
                "Action": action,
                "Observation": observation,
                "Admissible Commands": list(self.last_admissible_commands),
                "Score": raw_score,
                "Done": bool(self.done),
            }
        )

        return observation, processed_reward, self.done

    def feedback(self) -> tuple[float, bool, str]:
        # Only count true wins as success.
        success = getattr(self, "won", False)
        reward = 1.0 if success else 0.0
        message = "You successfully finished this task!" if success else "You failed the task."

        return reward, success, message

    @staticmethod
    def process_action(action: str) -> str:
        action = AlfworldEnv._extract_action(action)
        # The solver may output numbered/bulleted steps like "1. take X from Y"
        # or "- think: ...". Strip those prefixes so TextWorld can parse the command.
        action = re.sub(r"^\s*\d+\.\s*", "", action)
        action = re.sub(r"^\s*[-*]\s*", "", action)
        action = action.replace(">", "").replace("OK.", "").replace("OK", "").strip()
        action = action.rstrip(".。").strip()
        if action.lower().startswith("think:"):
            action = AlfworldEnv._normalize_think_action(action)

        return action

    @staticmethod
    def _extract_action(text: str) -> str:
        """Extract the first valid ALFWorld command from verbose/Qwen-style output."""
        text = str(text or "").strip()
        thought = AlfworldEnv._extract_thought(text)
        # Qwen3 may emit native reasoning tags before the actual answer.
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL).strip()
        text = re.sub(r"</?think>", "", text, flags=re.IGNORECASE).strip()

        commands = (
            "take ",
            "go to ",
            "open ",
            "put ",
            "move ",
            "clean ",
            "heat ",
            "cool ",
            "use ",
            "think:",
        )
        candidates: list[str] = []
        for raw_line in text.splitlines():
            line = raw_line.strip().lstrip(">").strip()
            line = re.sub(r"^\s*\d+\.\s*", "", line)
            line = re.sub(r"^\s*[-*]\s*", "", line)
            if not line:
                continue
            lowered = line.lower()
            if lowered.startswith(commands):
                candidates.append(line)

        # Prefer an executable environment action over an internal thought.
        for candidate in candidates:
            if not candidate.lower().startswith("think:"):
                return candidate
        if candidates:
            return candidates[0]

        # MCMA/ALFWorld trajectories support a legal ReAct-style thought action.
        # If Qwen only produced native thinking text and no executable command,
        # normalize it to `think:` instead of introducing a `look` action that
        # some validators/postprocessors do not expect.
        if thought:
            return AlfworldEnv._normalize_think_action(f"think: {thought}")
        return "think: I need to choose one valid next action."

    @staticmethod
    def _normalize_think_action(action: str, limit: int = 240) -> str:
        """Keep ReAct thoughts useful without letting small models bloat prompts."""
        text = str(action or "").strip()
        text = re.sub(r"^think\s*:\s*", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"\s+", " ", text)
        if not text:
            return "think: I need to choose one valid next action."

        pieces = [p.strip() for p in re.split(r"(?<=[.!?])\s+", text) if p.strip()]
        deduped: list[str] = []
        seen: set[str] = set()
        for piece in pieces:
            key = re.sub(r"\W+", " ", piece.lower()).strip()
            if key and key in seen:
                continue
            if key:
                seen.add(key)
            deduped.append(piece)
            if len(" ".join(deduped)) >= limit:
                break
        text = " ".join(deduped) if deduped else text
        if len(text) > limit:
            text = text[:limit].rsplit(" ", 1)[0].rstrip(".,;:")
        return f"think: {text}"

    def _adapt_action_to_admissible(self, action: str) -> str:
        """Map legacy ALFWorld placement syntax to the official move syntax.

        Official ALFWorld tw-pddl games expose placement commands such as
        `move bowl 1 to shelf 2`, while this project's prompts often produce
        `put bowl 1 in/on shelf 2`. Only rewrite when the target command is
        explicitly admissible in the current state.
        """
        action = str(action or "").strip()
        admissible = {cmd.lower(): cmd for cmd in self.last_admissible_commands}
        if action.lower() in admissible:
            return admissible[action.lower()]

        match = re.match(r"^put\s+(.+?)\s+(?:in/on|in|on)\s+(.+)$", action, flags=re.IGNORECASE)
        if not match:
            return action
        obj = re.sub(r"\s+", " ", match.group(1)).strip()
        dest = re.sub(r"\s+", " ", match.group(2)).strip()
        candidate = f"move {obj} to {dest}"
        return admissible.get(candidate.lower(), action)

    @staticmethod
    def _extract_thought(text: str, limit: int = 220) -> str:
        text = str(text or "").strip()
        match = re.search(r"<think>(.*?)(?:</think>|$)", text, flags=re.IGNORECASE | re.DOTALL)
        if match:
            thought = match.group(1)
        else:
            thought = text
        thought = re.sub(r"</?think>", "", thought, flags=re.IGNORECASE)
        thought = re.sub(r"\s+", " ", thought).strip(" >\t\r\n")
        if not thought or thought.lower() in {"think", "/think"}:
            return ""
        if len(thought) > limit:
            thought = thought[:limit].rsplit(" ", 1)[0].rstrip(".,;:")
        return thought

    @staticmethod
    def _normalize_observation(observation: str) -> str:
        if observation.startswith("You arrive at loc "):
            observation = observation[observation.find(". ") + 2 :]
        return observation

    @staticmethod
    def _extract_admissible_commands(info: Any) -> list[str]:
        if not isinstance(info, dict):
            return []
        commands = info.get("admissible_commands", [])
        if isinstance(commands, list) and commands:
            first = commands[0]
            if isinstance(first, list):
                return [str(item) for item in first]
            return [str(item) for item in commands]
        return []

    @staticmethod
    def _extract_goal_instruction(task: str) -> str:
        match = re.search(r"Your task is to:\s*(.+)", str(task or ""), re.DOTALL)
        if not match:
            return str(task or "").strip()
        return match.group(1).strip()

    def has_exportable_history(self) -> bool:
        return bool(self.gamefile and self.current_history)

    def export_memco_history(
        self,
        output_dir: str,
        *,
        model_id: str = "",
        status_override: str | None = None,
    ) -> str | None:
        if not self.has_exportable_history():
            return None
        path = Path(str(self.gamefile).replace("\\", "/"))
        parts = path.parts
        game_name = parts[-3] if len(parts) >= 3 else "task"
        game_index = parts[-2] if len(parts) >= 2 else "run"
        final_done = bool(self.current_history[-1].get("Done", False)) if self.current_history else False
        final_score = float(self.current_history[-1].get("Score", 0.0)) if self.current_history else 0.0
        status = status_override or ("success" if final_done else "fail")
        payload = {
            "last_updated": __import__("time").strftime("%Y%m%d_%H%M%S"),
            "game_file": str(self.gamefile),
            "game_name": game_name,
            "game_index": game_index,
            "game_task": self.goal_instruction,
            "status": status,
            "step_count": max(len(self.current_history) - 1, 0),
            "final_score": final_score,
            "history": list(self.current_history),
            "model_id": model_id,
        }
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"history_{game_name}_{game_index}_{status}.json"
        with out_path.open("w", encoding="utf-8") as writer:
            json.dump(payload, writer, ensure_ascii=False, indent=2)
        return str(out_path)

    def _parse_task_main(self, task: str):
        return self.env_name + "-" + re.search(r"Your task is to:\s*(.+)", task, re.DOTALL).group(1).strip()

    @staticmethod
    def _parse_task_description(task: str) -> str:
        return task.split("___")[0]


@dataclass
class AlfworldRecorder(BaseRecorder):

    def __post_init__(self):

        super().__post_init__()
        self.task = "alfworld"
        self.counts = [0] * 6
        self.results = [0] * 6

    def task_begin(self, task_id, task_config):
        super().task_begin(task_id, task_config)

        message: str = f"---------- Task: {task_id} ----------"
        self.log(message)

    def task_end(self, reward: float, done: bool):
        gamefile: str = self.current_task_config["env_kwargs"]["gamefile"]
        env_name = get_env_name_from_gamefile(gamefile)
        if env_name is None:
            raise ValueError("Format of the task config is wrong.")

        for i, (k, v) in enumerate(prefixes.items()):
            if env_name == k:
                self.results[i] += done
                self.counts[i] += 1
                break

        message = f"success: {done}, ave success: {sum(self.results) / sum(self.counts)}"
        self.log(message)
        self.log("rs: " + str(self.results))
        self.log("cnts: " + str(self.counts))
