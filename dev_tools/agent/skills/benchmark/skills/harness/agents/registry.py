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

"""Agent adapter registry for the benchmark harness."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from .base import AgentAdapter
from .config import ConfigurableAgentAdapter

BENCHMARK_ROOT = Path(__file__).resolve().parents[3]
AGENT_CONFIG_DIR = BENCHMARK_ROOT / "config" / "agents"
DEFAULT_BENCHMARK_AGENT = "codex"


@dataclass(frozen=True)
class AgentRegistration:
    name: str
    category: str
    config_name: str | None = None
    message: str = ""


AGENT_REGISTRY: dict[str, AgentRegistration] = {
    "codex": AgentRegistration("codex", "supported", "codex.yaml"),
    "claude": AgentRegistration("claude", "supported", "claude.yaml"),
    "hermes": AgentRegistration("hermes", "known_pending", None, "Hermes benchmark adapter is planned."),
    "openclaw": AgentRegistration("openclaw", "known_pending", None, "OpenClaw benchmark adapter is planned."),
}


def supported_agent_names() -> tuple[str, ...]:
    return tuple(name for name, registration in AGENT_REGISTRY.items() if registration.category == "supported")


def validate_benchmark_agent(agent: str) -> str:
    registration = AGENT_REGISTRY.get(agent)
    if registration is None:
        supported = ", ".join(supported_agent_names())
        known_pending = ", ".join(name for name, item in AGENT_REGISTRY.items() if item.category == "known_pending")
        raise ValueError(
            f"Unsupported BENCHMARK_AGENT={agent!r}. Supported benchmark agents: {supported}. "
            f"Known pending agents: {known_pending}."
        )
    if registration.category == "known_pending":
        raise ValueError(f"BENCHMARK_AGENT={agent!r} is known but not implemented. {registration.message}")
    if registration.category != "supported":
        raise ValueError(f"BENCHMARK_AGENT={agent!r} has invalid registry category {registration.category!r}")
    return agent


@lru_cache(maxsize=None)
def get_agent_adapter(agent: str) -> AgentAdapter:
    name = validate_benchmark_agent(agent)
    registration = AGENT_REGISTRY[name]
    if not registration.config_name:
        raise ValueError(f"BENCHMARK_AGENT={name!r} is missing an agent config file")
    return ConfigurableAgentAdapter(AGENT_CONFIG_DIR / registration.config_name)


def load_agent_adapter(agent: str) -> AgentAdapter:
    return get_agent_adapter(agent)


def clear_agent_adapter_cache() -> None:
    """Reset adapter instances cached from YAML configs.

    Tests that patch registry entries or config files should call this before
    reloading adapters.
    """

    get_agent_adapter.cache_clear()


def normalize_agent_event(agent: str, raw_line: str) -> dict | None:
    return get_agent_adapter(agent).normalize_event(raw_line)
