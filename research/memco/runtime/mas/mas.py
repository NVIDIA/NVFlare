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

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Optional

from .agents import Agent, Env
from .memory import MASMemoryBase
from .reasoning import ReasoningBase


@dataclass
class MetaMAS:
    """Unified management of members in the MAS system to achieve overall scheduling of the MAS."""

    agents_team: Dict[str, Agent] = field(default_factory=dict)
    env: Optional[Env] = None
    meta_memory: Optional[MASMemoryBase] = None

    def hire(self, agents: Iterable[Agent]) -> None:
        for agent in agents:
            if agent.name not in self.agents_team:
                self.agents_team[agent.name] = agent
            else:
                print(f"Agent {agent.name} is already in the team.")

    def set_env(self, env: Env) -> None:
        self.env = env

    def get_agent(self, agent_name: str) -> Optional[Agent]:
        return self.agents_team.get(agent_name)

    @abstractmethod
    def build_system(self, reasoning: ReasoningBase, mas_memory: MASMemoryBase) -> Any:
        pass

    @abstractmethod
    def schedule(self, **args) -> Any:
        pass
