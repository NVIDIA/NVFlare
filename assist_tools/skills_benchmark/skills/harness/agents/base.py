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

"""Agent adapter contracts for the benchmark harness."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class DockerMount:
    host_path: Path
    container_path: str
    read_only: bool = True
    description: str = ""
    required: bool = False


@dataclass(frozen=True)
class AgentImageTargets:
    skills: str
    baseline: str
    report: str


@dataclass(frozen=True)
class AgentLaunchSpec:
    argv: list[str]
    cwd: Path
    prompt_file: Path
    prompt_input_mode: str
    stdout_events_dest: Path
    stderr_dest: Path
    final_message_dest: Path | None = None
    environment: dict[str, str] = field(default_factory=dict)
    login_shell: bool = False
    approval_flags: list[str] = field(default_factory=list)
    sandbox_flags: list[str] = field(default_factory=list)
    bypass_reason: str | None = None
    launch_timeout: int | None = None
    extra_artifact_paths: list[Path] = field(default_factory=list)


@dataclass(frozen=True)
class SkillExposureSpec:
    mechanism_type: str
    container_home: Path | None = None
    skill_root: Path | None = None
    source_paths: list[Path] = field(default_factory=list)
    setup_action: list[str] = field(default_factory=list)
    probe_action: list[str] = field(default_factory=list)
    disable_action: list[str] = field(default_factory=list)
    launch_args: list[str] = field(default_factory=list)
    environment: dict[str, str] = field(default_factory=dict)
    metadata_files: list[Path] = field(default_factory=list)
    expected_post_setup_state: str | None = None
    disable_packaged_source: bool = False


@dataclass(frozen=True)
class SkillExposureResult:
    status: str
    mechanism_type: str
    installed_paths: list[str] = field(default_factory=list)
    disabled_paths: list[str] = field(default_factory=list)
    launch_args: list[str] = field(default_factory=list)
    environment: dict[str, str] = field(default_factory=dict)
    probe_status: str | None = None
    probe_output_ref: str | None = None
    metadata_files: list[dict[str, str]] = field(default_factory=list)
    parser_warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class FinalMessageSource:
    source_type: str
    path: Path | None = None
    event_selector: dict[str, Any] | None = None
    tail_bytes: int | None = None
    parser: str | None = None
    parser_warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class AgentLaunchContext:
    workspace_dir: Path
    prompt_file: Path
    result_dir: Path
    events_dest: Path
    stderr_dest: Path
    final_message_dest: Path
    model: str
    model_was_explicit: bool
    timeout_seconds: int | None = None


@dataclass(frozen=True)
class SkillExposureContext:
    result_dir: Path
    container_home: Path
    mode: str
    skills_enabled: bool
    sdk_image_kind: str


class AgentAdapter(ABC):
    """Contract for agent-specific benchmark mechanics.

    The adapter starts and parses an agent surface. It must not construct,
    edit, append, summarize, or reinterpret benchmark prompt content.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def display_name(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def default_model(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def agent_home_env(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def container_home(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def model_from_env(self, env: Mapping[str, str]) -> str:
        raise NotImplementedError

    @abstractmethod
    def model_was_explicit(self, env: Mapping[str, str]) -> bool:
        raise NotImplementedError

    @abstractmethod
    def model_env_names(self) -> tuple[str, ...]:
        raise NotImplementedError

    @abstractmethod
    def build_args(self) -> dict[str, str]:
        raise NotImplementedError

    @abstractmethod
    def image_targets(self) -> AgentImageTargets:
        raise NotImplementedError

    @abstractmethod
    def auth_mounts(self, host_config) -> list[DockerMount]:
        raise NotImplementedError

    @abstractmethod
    def runtime_env(self, config) -> dict[str, str]:
        raise NotImplementedError

    @abstractmethod
    def launch_spec(self, config: AgentLaunchContext) -> AgentLaunchSpec:
        raise NotImplementedError

    @abstractmethod
    def skill_exposure(self, config: SkillExposureContext) -> SkillExposureSpec:
        raise NotImplementedError

    @abstractmethod
    def availability_probe(self) -> list[str]:
        raise NotImplementedError

    @abstractmethod
    def normalize_event(self, raw_line: str) -> dict[str, Any] | None:
        raise NotImplementedError

    @abstractmethod
    def parse_usage(self, events_path: Path) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def parse_activity(self, events_path: Path) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def final_message_source(self, result_dir: Path) -> FinalMessageSource:
        raise NotImplementedError

    @abstractmethod
    def metadata(self) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def exit_summary(self, exit_code: int, stderr_path: Path) -> dict[str, Any]:
        raise NotImplementedError

    def artifact_alias_prefixes(self) -> tuple[str, ...]:
        return ()

    def passthrough_env_names(self) -> tuple[str, ...]:
        return ()

    def host_home_from_env(self, env: Mapping[str, str]) -> Path:
        return Path.home() / f".{self.name}"

    def mount_auth_from_env(self, env: Mapping[str, str]) -> bool:
        return True


def normalize_agent_event(agent: str, raw_line: str) -> dict[str, Any] | None:
    """Compatibility wrapper around the selected adapter's event normalizer."""

    from .registry import get_agent_adapter

    return get_agent_adapter(agent).normalize_event(raw_line)


def validate_benchmark_agent(agent: str) -> str:
    """Return a supported benchmark agent name or raise a clear error."""

    from .registry import validate_benchmark_agent as _validate_benchmark_agent

    return _validate_benchmark_agent(agent)
