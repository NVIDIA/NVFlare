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

from attr import dataclass
from mas.llm import LLMCallable, Message


@dataclass
class ReasoningConfig:
    temperature: float = None
    max_tokens: int = None
    stop_strs: list[str] = None
    num_comps: int = None


class ReasoningBase:
    def __init__(self, llm_model: LLMCallable):
        self.llm_model: LLMCallable = llm_model

    def __call__(self, prompts: list[Message], config: ReasoningConfig) -> str:
        raise NotImplementedError("This method should be implemented by subclasses.")


class ReasoningIO(ReasoningBase):
    """
    A subclass of ReasoningBase that defines how to perform a reasoning call
    just by using an underlying language model (LLM).
    """

    def __call__(self, prompts: list[Message], config: ReasoningConfig) -> str:

        reasoning_result = self.llm_model(
            prompts,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            stop_strs=config.stop_strs,
            num_comps=config.num_comps,
        )
        return reasoning_result
