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

import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Optional, Protocol

from openai import OpenAI

from .llm_replay import LLMReplayTape
from .utils import load_config

# model configs
CONFIG: dict = load_config(str(Path(__file__).resolve().parents[1] / "configs" / "configs.yaml"))
LLM_CONFIG: dict = CONFIG.get("llm_config", {})
MAX_TOKEN = LLM_CONFIG.get("max_token", 512)
TEMPERATURE = LLM_CONFIG.get("temperature", 0.1)
NUM_COMPS = LLM_CONFIG.get("num_comps", 1)

URL = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")
KEY = os.environ.get("OPENAI_API_KEY", "")
print("# api url: ", URL)


completion_tokens, prompt_tokens = 0, 0


@dataclass(frozen=True)
class Message:
    role: Literal["system", "user", "assistant"]
    content: str


class LLMCallable(Protocol):

    def __call__(
        self,
        messages: List[Message],
        temperature: float = TEMPERATURE,
        max_tokens: int = MAX_TOKEN,
        stop_strs: Optional[List[str]] = None,
        num_comps: int = NUM_COMPS,
    ) -> str:
        pass


class LLM(ABC):

    def __init__(self, model_name: str):
        self.model_name: str = model_name

    @abstractmethod
    def __call__(
        self,
        messages: List[Message],
        temperature: float = TEMPERATURE,
        max_tokens: int = MAX_TOKEN,
        stop_strs: Optional[List[str]] = None,
        num_comps: int = NUM_COMPS,
    ) -> str:
        pass


class GPTChat(LLM):

    def __init__(self, model_name: str):
        super().__init__(model_name=model_name)
        replay_mode = str(os.environ.get("MEMCO_LLM_REPLAY_MODE", "") or "").strip().lower()
        replay_path = str(os.environ.get("MEMCO_LLM_REPLAY_PATH", "") or "").strip()
        if bool(replay_mode) != bool(replay_path):
            raise ValueError("MEMCO_LLM_REPLAY_MODE and MEMCO_LLM_REPLAY_PATH must be set together")
        self.replay_tape = LLMReplayTape(Path(replay_path), replay_mode) if replay_mode else None
        if replay_mode != "replay" and not KEY:
            raise ValueError("OPENAI_API_KEY is required for live or record mode")
        self.client = None if replay_mode == "replay" else OpenAI(base_url=URL, api_key=KEY)
        self._is_qwen = "qwen" in (model_name or "").lower()
        self._is_qwen4b = "qwen4b" in (model_name or "").lower()
        self._is_qwen32b = "qwen32b" in (model_name or "").lower()
        self._is_gpt4omini = "gpt-4o-mini" in (model_name or "").lower()

    def _use_memco_fever_policy(self) -> bool:
        return str(os.environ.get("NV_MEMCO_FEVER_POLICY", "")).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
            "adaptive_cache",
        }

    def _use_memco_pddl_policy(self) -> bool:
        return str(os.environ.get("NV_MEMCO_PDDL_POLICY", "")).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
            "action_guard",
        }

    def _dataset_policy(self) -> str:
        return (
            str(os.environ.get("NV_DATASET_FAMILY", "") or "").strip().lower()
            or str(os.environ.get("NV_MEMCO_DATASET_POLICY", "") or "").strip().lower()
        )

    def _use_fever_action_format(self) -> bool:
        return self._dataset_policy() == "fever"

    def _use_alfworld_action_format(self) -> bool:
        return self._dataset_policy() == "alfworld"

    def _use_alfworld_qwen32_action_guard(self) -> bool:
        return self._use_alfworld_action_format() and self._is_qwen32b

    def _use_memco_qwen_command_guard(self) -> bool:
        if self._use_memco_pddl_policy():
            return self._is_qwen4b
        return False

    @staticmethod
    def _is_alfworld_action_prompt(content: str) -> bool:
        lowered = str(content or "").lower()
        if "-= welcome to textworld, alfred! =-" in lowered:
            return True
        if "your task is to:" in lowered and (
            "admissible commands" in lowered
            or "what action would you like to do next" in lowered
            or "valid commands" in lowered
        ):
            return True
        return False

    @staticmethod
    def _is_action_only_env_prompt(content: str) -> bool:
        text = str(content or "")
        lowered = text.lower()
        markers = (
            "admissible commands",
            "check valid actions",
            "the goal is to satisfy the following conditions",
            "what action would you like to do next",
            "valid commands",
        )
        return any(marker in lowered for marker in markers)

    def _is_fever_action_prompt(self, content: str) -> bool:
        if not self._use_fever_action_format():
            return False
        text = str(content or "")
        lowered = text.lower()
        # Exclude extraction/analysis prompts that contain FEVER trajectories as examples
        # but are NOT asking for a FEVER action themselves.
        extraction_markers = (
            "you are an agent skilled at extracting key points",
            "strictly follow the original trajectory",
            "you are an analytical agent",  # detect_mistakes prompt
        )
        if any(m in lowered for m in extraction_markers):
            return False
        markers = (
            "## successful examples",
            "## key insights from related tasks",
            "## your turn: take action!",
            "claim:",
            "search[entity]",
            "lookup[keyword]",
            "finish[answer]",
            "valid actions are lookup",
            "use retrieved memories as evidence-search guidance",
            "prefer a focused search[...] from the claim",
        )
        bracket_actions = sum(token in text for token in ("Search[", "Lookup[", "Finish["))
        return sum(marker in lowered for marker in markers) >= 2 or bracket_actions >= 2

    @staticmethod
    def _looks_like_thought_answer(answer: str) -> bool:
        lowered = str(answer or "").strip().lower()
        return lowered.startswith("think:") or lowered.startswith("thought:") or lowered.startswith("i need to ")

    @staticmethod
    def _looks_like_prompt_echo(answer: str) -> bool:
        text = str(answer or "").strip()
        lowered = text.lower()
        markers = (
            "## your turn: take action!",
            "here is your initial observation:",
            "**here is your task:",
            "do not copy object names",
            "if an action fails or stalls",
            "use the above examples and insights as a foundation",
        )
        if any(marker in lowered for marker in markers):
            return True
        if "\n" in text and text.count("\n") >= 3:
            return True
        if text.startswith(">") and len(text) < 8:
            return True
        return False

    @staticmethod
    def _sanitize_action_only_answer(answer: str) -> str:
        text = str(answer or "").strip()
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            return text
        first = lines[0].lstrip(">").strip()
        if len(lines) == 1:
            return first
        lowered = first.lower()
        if lowered.startswith(("think:", "thought:")):
            return first
        if any(
            marker in lowered
            for marker in (
                "## your turn",
                "here is your initial observation",
                "here is your task",
                "do not copy object names",
                "use the above examples",
            )
        ):
            return text
        return first

    @staticmethod
    def _safe_action_only_fallback(prompt: str) -> str:
        lowered = str(prompt or "").lower()
        if "check valid actions" in lowered:
            return "check valid actions"
        return ""

    @staticmethod
    def _extract_first_fever_action(answer: str) -> str:
        text = str(answer or "")
        if not text.strip():
            return ""
        pattern = re.compile(
            r"\b(?:Search|Lookup|Finish)\[[^\]\n]+\]",
            flags=re.IGNORECASE,
        )
        match = pattern.search(text)
        if not match:
            return ""
        action = match.group(0).strip()
        if action.lower().startswith("finish["):
            inner = action[len("Finish[") : -1].strip()
            upper_inner = inner.upper()
            if upper_inner in {"SUPPORTS", "REFUTES", "NOT ENOUGH INFO"}:
                return f"Finish[{upper_inner}]"
        if action.lower().startswith("search["):
            return f"Search[{action[action.find('[')+1:-1].strip()}]"
        if action.lower().startswith("lookup["):
            return f"Lookup[{action[action.find('[')+1:-1].strip()}]"
        return action

    def _inject_action_only_guard(self, content: str, *, strict_retry: bool = False) -> str:
        content = str(content or "")
        if self._use_alfworld_qwen32_action_guard():
            if "/no_think" not in content:
                return "/no_think\nOutput exactly one valid command and nothing else.\n" + content
            return content
        if self._is_qwen4b and (self._use_alfworld_action_format() or self._is_alfworld_action_prompt(content)):
            if "/no_think" not in content:
                return "/no_think\nOutput exactly one valid ALFWorld command and nothing else.\n" + content
            return content
        if self._use_memco_qwen_command_guard():
            if "/no_think" not in content:
                return "/no_think\nOutput exactly one valid command and nothing else.\n" + content
            return content
        if self._is_fever_action_prompt(content):
            prefix = (
                "Output exactly one action and nothing else.\n"
                "Valid formats: Search[...], Lookup[...], Finish[SUPPORTS], Finish[REFUTES], Finish[NOT ENOUGH INFO].\n"
                "Do not output Thought, Observation, explanation, XML tags, function_calls, Markdown, or multiple actions.\n"
            )
            if strict_retry:
                prefix = (
                    "Return exactly one FEVER action only.\n"
                    "Your entire answer must be exactly one of: Search[...], Lookup[...], Finish[SUPPORTS], Finish[REFUTES], Finish[NOT ENOUGH INFO].\n"
                    "No prose, no Thought, no Action 1 label, no explanation, no XML, no extra lines.\n"
                )
            if self._is_qwen and "/no_think" not in content:
                prefix = "/no_think\n" + prefix
            if prefix not in content:
                return prefix + content
            return content
        if self._is_qwen:
            if "/no_think" not in content:
                return "/no_think\nOutput exactly one valid command and nothing else.\n" + content
            return content

        if self._is_gpt4omini:
            prefix = (
                "Return exactly one admissible command from the current command list.\n"
                "Do not output think:, reasoning, explanation, or multiple lines.\n"
            )
            if strict_retry:
                prefix = (
                    "Return exactly one admissible command from the current admissible command list.\n"
                    "Do not output think:, Thought:, analysis, explanation, Markdown, or any extra text.\n"
                    "Your entire answer must be the command only.\n"
                )
            if prefix not in content:
                return prefix + content
        return content

    def _prepare_messages(
        self, messages: List[Message], *, strict_retry: bool = False
    ) -> tuple[list[dict], bool, bool, str]:
        prepared = [{"role": msg.role, "content": msg.content} for msg in messages]
        for msg in reversed(prepared):
            if msg["role"] == "user":
                content = str(msg.get("content") or "")
                is_action_only = self._is_action_only_env_prompt(content)
                is_alfworld_qwen32_action = self._use_alfworld_qwen32_action_guard()
                is_alfworld_action = self._is_qwen4b and (
                    self._is_alfworld_action_prompt(content)
                    or (self._use_alfworld_action_format() and "your task is to:" in content.lower())
                )
                is_fever_action = self._is_fever_action_prompt(content)
                if is_alfworld_qwen32_action or is_alfworld_action:
                    is_action_only = True
                if self._use_memco_qwen_command_guard():
                    msg["content"] = self._inject_action_only_guard(content, strict_retry=strict_retry)
                elif is_action_only or is_fever_action:
                    msg["content"] = self._inject_action_only_guard(content, strict_retry=strict_retry)
                action_prompt = content
                break
        else:
            is_action_only = False
            is_fever_action = False
            action_prompt = ""
        return prepared, is_action_only, is_fever_action, action_prompt

    @staticmethod
    def _sanitize_stop_strs(stop_strs: Optional[List[str]]) -> Optional[List[str]]:
        if not stop_strs:
            return None
        cleaned: list[str] = []
        for item in stop_strs:
            value = str(item or "")
            if not value:
                continue
            if not value.strip():
                continue
            cleaned.append(value)
        return cleaned or None

    def __call__(
        self,
        messages: List[Message],
        temperature: float = TEMPERATURE,
        max_tokens: int = MAX_TOKEN,
        stop_strs: Optional[List[str]] = None,
        num_comps: int = NUM_COMPS,
    ) -> str:
        import time

        global prompt_tokens, completion_tokens

        stop_strs = self._sanitize_stop_strs(stop_strs)
        request_kwargs = {}
        if self._is_qwen:
            request_kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}

        max_retries = 5
        wait_time = 1
        none_count = 0  # consecutive None responses from silent rate-limiting (OpenRouter)
        max_none_count = 10
        attempt = 0

        while attempt < max_retries:
            try:
                prepared_messages, is_action_only, is_fever_action, action_prompt = self._prepare_messages(
                    messages,
                    strict_retry=(attempt > 0),
                )
                create_kwargs = dict(
                    model=self.model_name,
                    messages=prepared_messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    n=num_comps,
                    **request_kwargs,
                )
                if stop_strs is not None:
                    create_kwargs["stop"] = stop_strs
                if self.replay_tape and self.replay_tape.mode == "replay":
                    answer = self.replay_tape.replay(create_kwargs)
                else:
                    response = self.client.chat.completions.create(**create_kwargs)
                    answer = response.choices[0].message.content
                    prompt_tokens += response.usage.prompt_tokens
                    completion_tokens += response.usage.completion_tokens

                if answer is None:
                    # OpenRouter returns HTTP 200 with content=None when silently rate-limiting
                    # instead of a proper 429. Back off without burning the main retry budget.
                    none_count += 1
                    print(f"Error: LLM returned None ({none_count}/{max_none_count})")
                    time.sleep(wait_time)
                    wait_time = min(wait_time * 2, 60)
                    if none_count >= max_none_count:
                        attempt += 1  # give up waiting, burn one main retry
                    continue

                answer = answer.strip()
                if not answer:
                    # Avoid silent empty outputs causing infinite retries in downstream parsers.
                    raise RuntimeError("Empty LLM response")
                if self.replay_tape and self.replay_tape.mode == "record":
                    self.replay_tape.record(create_kwargs, answer)
                if is_fever_action:
                    extracted = self._extract_first_fever_action(answer)
                    if extracted:
                        return extracted
                if is_action_only and self._is_gpt4omini:
                    answer = self._sanitize_action_only_answer(answer)
                if is_action_only and self._is_gpt4omini and self._looks_like_thought_answer(answer):
                    fallback = self._safe_action_only_fallback(action_prompt)
                    return fallback or answer
                if is_action_only and self._is_gpt4omini and self._looks_like_prompt_echo(answer):
                    fallback = self._safe_action_only_fallback(action_prompt)
                    return fallback or answer
                return answer

            except Exception as e:
                if self.replay_tape and self.replay_tape.mode == "replay":
                    raise
                # Convert error to string as safely as possible (avoid nested Unicode errors)
                try:
                    error_message = str(e)
                except Exception:
                    error_message = ""

                # For rate limit or 429 errors, back off and retry
                if "rate limit" in error_message.lower() or "429" in error_message:
                    time.sleep(wait_time)
                    attempt += 1
                    continue

                if request_kwargs and any(
                    marker in error_message for marker in ("chat_template_kwargs", "enable_thinking", "extra_body")
                ):
                    request_kwargs = {}
                    time.sleep(wait_time)
                    attempt += 1
                    continue

                # For all other errors, print a short preview and retry a few times,
                # then raise to prevent downstream infinite loops.
                preview = (error_message or repr(e))[:500]
                print(f"[GPTChat] error: {type(e).__name__}: {preview}")
                time.sleep(wait_time)
                attempt += 1
                continue

        raise RuntimeError("GPTChat failed after retries")


def get_price():
    global completion_tokens, prompt_tokens
    return completion_tokens, prompt_tokens, completion_tokens * 60 / 1000000 + prompt_tokens * 30 / 1000000
