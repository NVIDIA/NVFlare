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

from typing import TypeVar

from mas.llm import Message
from mas.reasoning import ReasoningBase, ReasoningConfig

T = TypeVar("T")


class Agent:
    def __init__(
        self, name: str, role: str, system_instruction: str, reasoning_module: ReasoningBase, memory_module=None
    ):
        if reasoning_module is None:
            raise ValueError("The reasoning module should not be none.")

        # basic info
        self.name: str = name
        self.profile: str = role
        self.system_instruction: str = system_instruction

        # reasoning module
        self.reasoning: ReasoningBase = reasoning_module
        self.memory = memory_module

        self.total_system_instruction: str = self.system_instruction

    def add_task_instruction(self, task_instruction: str) -> str:
        self.total_system_instruction = self.system_instruction + "\n" + task_instruction
        return self.total_system_instruction

    def response(self, user_prompt: str, reason_config: ReasoningConfig) -> str:
        messages: list[Message] = [Message("system", self.total_system_instruction), Message("user", user_prompt)]
        return self.reasoning(messages, reason_config)


class Env:

    def __init__(self) -> None:
        pass

    def set_env(self, configs: dict) -> None:
        pass

    def reset(self) -> None:
        pass

    def step(self, action: str) -> None:
        pass
