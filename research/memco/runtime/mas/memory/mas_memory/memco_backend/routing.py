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

from ...common import MASMessage
from .overlay import MemCoOnlineEpisodeBuilder
from .types import MemCoPromptPayload


def _message_to_execution_pattern(message: MASMessage) -> str:
    description = str(message.task_description or "").strip()
    trajectory = str(message.task_trajectory or "").strip()
    blocks: list[str] = []
    if description:
        blocks.append("### Task description:\n" + description)
    if trajectory:
        blocks.append("### Detailed trajectory:\n" + trajectory)
    return "\n\n".join(blocks).strip()


def build_memco_prompt_payload(
    *,
    successful_messages: list[MASMessage],
    failed_messages: list[MASMessage],
    overlay_builder: MemCoOnlineEpisodeBuilder | None,
    stored_insights: list[str],
) -> MemCoPromptPayload:
    execution_patterns = [
        _message_to_execution_pattern(message)
        for message in successful_messages
        if str(message.task_trajectory or "").strip()
    ]
    planner_notes = overlay_builder.planner_notes() if overlay_builder is not None else []
    repair_hints: list[str] = []
    if failed_messages:
        repair_hints.append(
            "[MemCoRepair] Similar failed cases exist; avoid repeating actions that produced no progress."
        )
    return MemCoPromptPayload(
        reference_cases=[],
        execution_patterns=execution_patterns,
        insights=list(stored_insights),
        planner_notes=planner_notes,
        action_constraints=[],
        repair_hints=repair_hints,
    )
