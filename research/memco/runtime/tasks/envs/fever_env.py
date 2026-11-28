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
from typing import Any, Literal

from .base_env import BaseEnv, BaseRecorder
from .utils import LangChainWiki, match_exactly


class FeverEnv(BaseEnv):
    def __init__(self, env_config: dict[str, Any], max_trials: int = 7) -> None:

        self.env_config = env_config
        self.memco_domain = "fever"
        self._memco_fever_policy_enabled = str(os.environ.get("NV_MEMCO_FEVER_POLICY", "")).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
            "adaptive_cache",
        }
        cache_path = os.environ.get(
            "FEVER_WIKI_CACHE",
            os.path.join(os.getcwd(), ".cache", "fever_wiki_cache.json"),
        )
        self.explorer = LangChainWiki(cache_path=cache_path)
        self.max_trials: int = max_trials
        self.config: dict[str, Any] = {}
        self.claim: str = ""
        self.ab_domain: str = ""
        self.current_history: list[dict[str, Any]] = []
        self.last_admissible_commands: list[str] = []

        self.reset()

    def set_env(self, configs: dict) -> tuple[str, str]:
        if configs.get("answer") is None:
            raise ValueError("Please provide the answer for the question.")
        if configs.get("task") is None:
            raise ValueError("The configs dict should have the `task` attribute.")
        self.config = dict(configs or {})
        self.claim = str(self.config.get("task") or "")
        self.ab_domain = str(self.config.get("ab_domain", "") or "")

        task_value = self.claim
        task: str = f"Claim: {task_value}"
        self.current_task = task
        return task, task

    def reset(self) -> None:
        claim = str(getattr(self, "claim", "") or (getattr(self, "config", {}) or {}).get("task", "") or "")
        self.current_task = f"Claim: {claim}" if claim else None
        self.reward: float = 0
        self.steps: int = 0
        self.last_admissible_commands = self._default_admissible_commands(claim)
        self.current_history = [
            {
                "Step": 0,
                "Observation": f"Claim: {claim}" if claim else "",
                "Claim": claim,
                "Admissible Commands": list(self.last_admissible_commands),
                "Score": float(self.reward),
                "Reward": 0.0,
                "Done": False,
            }
        ]

    def step(self, action: str) -> tuple[str, float, bool]:

        action: str = (
            self._process_action_legacy(action) if self._memco_fever_policy_enabled else self.process_action(action)
        )
        self.steps += 1

        if self._parse_action_type(action) == "thought":
            observation = "OK."
            self._append_memco_history(
                action=action,
                observation=observation,
                reward=0,
                done=False,
                action_type="thought",
                argument="",
            )
            return observation, 0, False

        action_type, argument = self._parse_action(action)
        if action_type in {"Search", "Lookup"} and not self._memco_fever_policy_enabled:
            argument = self._normalize_query_argument(action_type, argument)
            action = f"{action_type}[{argument}]"

        if action_type == "Finish":
            if self.success_fn(argument):
                observation = "Answer is CORRECT"
                self.reward = 1
                self._append_memco_history(
                    action=action,
                    observation=observation,
                    reward=1,
                    done=True,
                    action_type=action_type,
                    argument=argument,
                )
                return observation, 1, True

            else:
                observation = "Answer is INCORRECT"
                self._append_memco_history(
                    action=action,
                    observation=observation,
                    reward=0,
                    done=True,
                    action_type=action_type,
                    argument=argument,
                )
                return observation, 0, True

        elif action_type == "Search":
            while True:
                try:
                    if self._memco_fever_policy_enabled:
                        observation = self.explorer.search(argument).strip("\n").strip()
                    else:
                        observation = self.explorer.search(argument, context=self.claim).strip("\n").strip()
                    self.summary = observation
                    break
                except Exception as e:
                    print(e)
                    observation = "Cannot find corresponding pages."
                    break
        elif action_type == "Lookup":
            try:
                observation = self.explorer.lookup(argument).strip("\n").strip()
            except ValueError:
                observation = "The last page Searched was not found, so you cannot Lookup a keyword in it. Please try one of the similar pages given."
        else:
            observation = "Invalid Action. Valid Actions are Lookup[<topic>] Search[<topic>] and Finish[<answer>]."

        if "Invalid Action" in observation:
            processed_reward = -1
        else:
            processed_reward = 0
        self._append_memco_history(
            action=action,
            observation=observation,
            reward=processed_reward,
            done=False,
            action_type=str(action_type or ""),
            argument=str(argument or ""),
        )
        return observation, processed_reward, False

    @staticmethod
    def _parse_action_type(action: str) -> Literal["action", "thought"]:
        if "thought" in action.lower():
            return "thought"
        else:
            return "action"

    @staticmethod
    def process_action(action: str) -> str:
        raw = str(action or "").strip().replace("<", "").replace(">", "")
        extracted = FeverEnv._extract_action_command(raw)
        if extracted:
            return extracted
        action = raw.split("\n")[0]
        action = action.replace(">", "").replace("OK.", "").replace("OK", "").strip()
        extracted = FeverEnv._extract_action_command(action)
        if extracted:
            return extracted

        if FeverEnv._parse_action_type(action) == "thought":
            return action
        if ":" in action:
            action = action.split(":")[1].strip()
            extracted = FeverEnv._extract_action_command(action)
            if extracted:
                return extracted

        return action

    @staticmethod
    def _process_action_legacy(action: str) -> str:
        action = str(action or "").strip().replace("<", "").split("\n")[0]
        action = action.replace(">", "").replace("OK.", "").replace("OK", "").strip()

        if FeverEnv._parse_action_type(action) == "thought":
            return action
        if ":" in action:
            action = action.split(":")[1].strip()

        return action

    @staticmethod
    def _extract_action_command(text: str) -> str:
        """Recover an executable command from Qwen-style verbose FEVER output."""
        raw = str(text or "").strip()
        match = re.search(r"\b(Search|Lookup|Finish)\[([^\]\n]+)\]", raw, flags=re.IGNORECASE)
        if match:
            action_type = match.group(1).lower()
            argument = match.group(2).strip()
            if action_type == "finish":
                normalized = re.sub(r"\s+", " ", argument.upper())
                if normalized in {"SUPPORTS", "REFUTES", "NOT ENOUGH INFO"}:
                    return f"Finish[{normalized}]"
            elif action_type == "search":
                return f"Search[{argument}]"
            elif action_type == "lookup":
                return f"Lookup[{argument}]"
        lowered = raw.lower()
        if lowered:
            if any(
                phrase in lowered
                for phrase in (
                    "not enough information",
                    "not enough info",
                    "cannot be verified or refuted",
                    "cannot verify or refute",
                    "cannot confirm",
                    "unable to determine from the available information",
                )
            ):
                return "Finish[NOT ENOUGH INFO]"

            if any(
                phrase in lowered
                for phrase in (
                    "claim is false",
                    "claim is refuted",
                    "therefore, the claim is refuted",
                    "this directly refutes the claim",
                    "this contradicts the claim",
                    "the claim is contradicted",
                    "the claim is not supported",
                )
            ):
                return "Finish[REFUTES]"

            if any(
                phrase in lowered
                for phrase in (
                    "claim is true",
                    "claim is supported",
                    "therefore, the claim is supported",
                    "this directly supports the claim",
                    "the evidence supports the claim",
                )
            ):
                return "Finish[SUPPORTS]"

            quoted_search_matches = list(
                re.finditer(
                    r'(?:search(?:ing)?(?:\s+for)?|search)\s+["“]([^"\n\]]+)["”]',
                    raw,
                    flags=re.IGNORECASE,
                )
            )
            if quoted_search_matches:
                query = quoted_search_matches[-1].group(1).strip(" \"'.,:;!?")
                if query:
                    return f"Search[{query}]"

            quoted_lookup_matches = list(
                re.finditer(
                    r'(?:look\s*up|lookup)(?:\s+for)?\s+["“]([^"\n\]]+)["”]',
                    raw,
                    flags=re.IGNORECASE,
                )
            )
            if quoted_lookup_matches:
                keyword = quoted_lookup_matches[-1].group(1).strip(" \"'.,:;!?")
                if keyword:
                    return f"Lookup[{keyword}]"

        return ""

    @staticmethod
    def _parse_action(string: str) -> tuple[str, str]:

        pattern = r"^(\w+)\[(.+)\]$"
        match = re.match(pattern, string)

        if match:
            action_type = match.group(1)
            argument = match.group(2)
            return action_type, argument
        else:
            return None, None

    def _normalize_query_argument(self, action_type: str, argument: str) -> str:
        text = str(argument or "").strip()
        if not text:
            return text

        text = re.sub(r"\s+", " ", text).strip(" \"'`")
        lowered = text.lower()
        claim_anchor = self._claim_anchor(str(getattr(self, "claim", "") or ""))

        generic_queries = {
            "result",
            "results",
            "strategy",
            "related terms",
            "terms related to the claim",
            "attempts did not yield results",
            "the broader franchise or related terms",
        }
        if lowered in generic_queries:
            return claim_anchor or text

        stop_phrases = (
            " to determine ",
            " and determine ",
            " to gather ",
            " and find ",
            " to find ",
            " to check ",
            " and check ",
            " to see if ",
            " and see if ",
            " to focus on ",
            " specifically ",
            " result indicates ",
            " results repeatedly mention ",
            " attempts did not yield results",
        )
        for marker in stop_phrases:
            idx = lowered.find(marker)
            if idx > 0:
                text = text[:idx].strip(" ,.:;!?")
                lowered = text.lower()
                break

        quoted = re.search(r'["“]([^"\n]+)["”]', text)
        if quoted:
            candidate = quoted.group(1).strip(" \"'.,:;!?")
            if candidate:
                return candidate

        if action_type == "Search":
            title_match = re.search(r"\b([A-Z][\w'&.-]*(?:\s+[A-Z][\w'&().-]*){0,6})\b", text)
            if title_match:
                candidate = title_match.group(1).strip(" \"'.,:;!?")
                if candidate and candidate.lower() not in {"search", "lookup"}:
                    return candidate

        if claim_anchor:
            anchor_lower = claim_anchor.lower()
            if anchor_lower in lowered or lowered in generic_queries or len(text.split()) > 6:
                return claim_anchor

        if len(text.split()) > 8:
            return " ".join(text.split()[:8]).strip(" ,.:;!?")
        return text

    def success_fn(self, agent_ans: str) -> bool:
        return match_exactly(agent_ans, self.config.get("answer"))

    def feedback(self) -> tuple[float, bool, str]:

        feedback: str = "You successfully finished this task." if self.reward == 1 else "You failed the task."
        done = self.reward == 1

        return self.reward, done, feedback

    @staticmethod
    def _default_admissible_commands(claim: str = "") -> list[str]:
        anchor = FeverEnv._claim_anchor(claim)
        commands = [
            f"Search[{anchor or '<topic>'}]",
            f"Lookup[{anchor or '<keyword>'}]",
            "Finish[SUPPORTS]",
            "Finish[REFUTES]",
            "Finish[NOT ENOUGH INFO]",
        ]
        return commands

    @staticmethod
    def _claim_anchor(claim: str) -> str:
        text = str(claim or "").strip()
        match = re.search(r"\b[A-Z][A-Za-z0-9]*(?:\s+[A-Z][A-Za-z0-9]*){0,4}", text)
        if match:
            return match.group(0)
        words = [w for w in re.findall(r"[A-Za-z0-9]+", text) if len(w) > 2]
        return words[0] if words else ""

    def _append_memco_history(
        self,
        *,
        action: str,
        observation: str,
        reward: float,
        done: bool,
        action_type: str,
        argument: str,
    ) -> None:
        claim = str(getattr(self, "claim", "") or (getattr(self, "config", {}) or {}).get("task", "") or "")
        self.last_admissible_commands = self._default_admissible_commands(claim)
        self.current_history.append(
            {
                "Step": int(self.steps),
                "Action": str(action or ""),
                "Action Type": str(action_type or ""),
                "Argument": str(argument or ""),
                "Observation": str(observation or ""),
                "Claim": claim,
                "Admissible Commands": list(self.last_admissible_commands),
                "Score": float(self.reward),
                "Reward": float(reward),
                "Done": bool(done),
            }
        )

    def has_exportable_history(self) -> bool:
        return bool(getattr(self, "current_history", None))

    def export_memco_history(
        self,
        output_dir: str,
        *,
        model_id: str = "",
        status_override: str | None = None,
    ) -> str | None:
        if not self.has_exportable_history():
            return None
        claim = str(getattr(self, "claim", "") or (getattr(self, "config", {}) or {}).get("task", "") or "")
        final_done = bool(self.current_history[-1].get("Done", False)) if self.current_history else False
        final_score = float(self.reward or 0.0)
        status = status_override or ("success" if final_done and final_score > 0 else "fail")
        domain = str(
            getattr(self, "ab_domain", "") or (getattr(self, "config", {}) or {}).get("ab_domain", "") or "default"
        )
        payload = {
            "last_updated": __import__("time").strftime("%Y%m%d_%H%M%S"),
            "game_file": "",
            "game_name": "fever",
            "game_index": str((getattr(self, "config", {}) or {}).get("task_id", "")),
            "game_task": f"Claim: {claim}",
            "claim": claim,
            "answer": str((getattr(self, "config", {}) or {}).get("answer", "")),
            "ab_domain": domain,
            "status": status,
            "step_count": max(len(self.current_history) - 1, 0),
            "final_score": final_score,
            "history": list(self.current_history),
            "model_id": model_id,
            "memco_domain": self.memco_domain,
            "task_family": f"fever:{domain}",
            "task_config": dict(getattr(self, "config", {}) or {}),
        }
        out_dir = os.path.abspath(output_dir)
        os.makedirs(out_dir, exist_ok=True)
        digest = abs(hash(claim)) % 10_000_000
        safe_domain = domain.replace("/", "_")
        out_path = os.path.join(out_dir, f"history_fever_{safe_domain}_{digest}_{status}.json")
        with open(out_path, "w", encoding="utf-8") as writer:
            json.dump(payload, writer, ensure_ascii=False, indent=2)
        return out_path


@dataclass
class FeverRecorder(BaseRecorder):
    def __post_init__(self):
        super().__post_init__()
        self.task = "hotpotqa"
        self.counts = 0
        self.dones = 0
        self.rewards = 0

    def task_begin(self, task_id: int, task_config: dict):
        super().task_begin(task_id, task_config)

        message: str = f"---------- Task: {task_id} ----------"
        self.log(message)

    def task_end(self, reward: float, done: bool):

        self.rewards += reward
        self.dones += done
        self.counts += 1

        message = f"reward: {reward}, ave reward: {self.rewards / self.counts}.\ndone: {done}, ave done: {self.dones / self.counts}"
        self.log(message)
