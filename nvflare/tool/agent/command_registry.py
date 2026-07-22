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

"""Shared metadata for the agent-facing command surface."""

from __future__ import annotations

from copy import deepcopy

AGENT_COMMANDS = (
    {
        "name": "info",
        "command": "nvflare agent info",
        "status": "available",
        "mutating": False,
        "streaming": False,
    },
    {
        "name": "inspect",
        "command": "nvflare agent inspect",
        "status": "available",
        "mutating": False,
        "streaming": False,
    },
)


def agent_commands() -> list[dict]:
    return deepcopy(list(AGENT_COMMANDS))


def agent_command_registry() -> dict:
    return {"status": "ok", "commands": agent_commands()}
