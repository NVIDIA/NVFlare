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


@dataclass
class MemCoEpisodeState:
    task_main: str
    task_description: str = ""
    recent_actions: list[str] = field(default_factory=list)
    recent_observations: list[str] = field(default_factory=list)
    step_count: int = 0


@dataclass
class MemCoPromptPayload:
    reference_cases: list[str] = field(default_factory=list)
    execution_patterns: list[str] = field(default_factory=list)
    insights: list[str] = field(default_factory=list)
    planner_notes: list[str] = field(default_factory=list)
    action_constraints: list[str] = field(default_factory=list)
    repair_hints: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, list[str]]:
        return {
            "reference_cases": list(self.reference_cases),
            "execution_patterns": list(self.execution_patterns),
            "insights": list(self.insights),
            "planner_notes": list(self.planner_notes),
            "action_constraints": list(self.action_constraints),
            "repair_hints": list(self.repair_hints),
        }
