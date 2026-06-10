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

"""YAML-driven benchmark agent adapter implementation."""

from __future__ import annotations

import hashlib
import string
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

from .base import (
    AgentAdapter,
    AgentImageTargets,
    AgentLaunchContext,
    AgentLaunchSpec,
    DockerMount,
    FinalMessageSource,
    SkillExposureContext,
    SkillExposureSpec,
)
from .classifiers import classify_exit, validate_exit_config
from .parsers import (
    normalize_event_with_parser,
    parse_activity_from_events,
    parse_usage_from_events,
    validate_activity_parser,
    validate_event_parser,
    validate_final_message_config,
    validate_usage_parser,
)

PROMPT_TEXT_PLACEHOLDER = "prompt_text"
UNSPECIFIED_MODEL = "unspecified_default"
MODEL_ARGV_POSITION_BEFORE_STDIN_SENTINEL = "before_stdin_sentinel"
MODEL_ARGV_POSITION_APPEND = "append"
MODEL_ARGV_POSITIONS = {MODEL_ARGV_POSITION_BEFORE_STDIN_SENTINEL, MODEL_ARGV_POSITION_APPEND}
LAUNCH_CONDITIONS = {"model_was_explicit"}
SKILL_EXPOSURE_CONDITIONS: set[str] = set()
SUPPORTED_SKILL_EXPOSURE_MECHANISMS = {"launch_flag", "none", "preinstalled_home"}


def template_contains(value: Any, needle: str) -> bool:
    if isinstance(value, str):
        return needle in value
    if isinstance(value, list):
        return any(template_contains(item, needle) for item in value)
    if isinstance(value, dict):
        return any(template_contains(item, needle) for item in value.values())
    return False


def legacy_artifact_prefixes(data: Mapping[str, Any], config_path: Path) -> tuple[str, ...]:
    prefixes = []
    for item in data.get("legacy_artifact_prefixes") or ():
        prefix = str(item)
        if not prefix or Path(prefix).name != prefix:
            raise ValueError(f"{config_path}: legacy_artifact_prefixes entries must be bare file prefixes: {prefix}")
        prefixes.append(prefix)
    return tuple(prefixes)


def validate_when_conditions(values: Any, allowed: set[str], label: str, config_path: Path) -> None:
    if isinstance(values, list):
        for index, item in enumerate(values):
            validate_when_conditions(item, allowed, f"{label}[{index}]", config_path)
        return
    if not isinstance(values, dict):
        return
    condition = values.get("when")
    if condition is not None:
        condition_name = str(condition)
        if condition_name not in allowed:
            allowed_text = ", ".join(sorted(allowed)) if allowed else "none"
            raise ValueError(
                f"{config_path}: {label}.when references unknown condition {condition_name!r}; "
                f"allowed conditions: {allowed_text}"
            )
    validate_when_conditions(values.get("args") or [], allowed, f"{label}.args", config_path)


@dataclass(frozen=True)
class ParserConfig:
    parser: str
    fidelity: str | None = None
    is_cumulative: bool | None = None


@dataclass(frozen=True)
class AgentConfig:
    source_path: Path
    raw: dict[str, Any]
    source_sha256: str
    name: str
    display_name: str
    default_model: str
    model_env: str | None
    requires_explicit_model: bool
    agent_home_env: str
    container_home: str
    legacy_artifact_prefixes: tuple[str, ...]
    images: dict[str, str]
    auth: dict[str, Any]
    launch: dict[str, Any]
    skill_exposure: dict[str, Any]
    final_message: dict[str, Any]
    events: ParserConfig
    usage: ParserConfig
    activity: ParserConfig
    exit_config: dict[str, Any]
    exit_classifier: str
    availability_probe: list[str] = field(default_factory=list)

    @classmethod
    def load(cls, config_path: Path) -> "AgentConfig":
        source = config_path.read_text(encoding="utf-8")
        if f"{{{PROMPT_TEXT_PLACEHOLDER}}}" in source:
            raise ValueError(f"{config_path} must not use {{{PROMPT_TEXT_PLACEHOLDER}}}; use prompt_file delivery")
        data = yaml.safe_load(source) or {}
        if not isinstance(data, dict):
            raise ValueError(f"{config_path} must contain a YAML object")
        required = (
            "name",
            "display_name",
            "agent_home_env",
            "container_home",
            "launch",
            "skill_exposure",
            "final_message",
            "events",
            "usage",
            "activity",
            "exit",
        )
        missing = [name for name in required if name not in data]
        if missing:
            raise ValueError(f"{config_path} is missing required field(s): {', '.join(missing)}")

        launch = required_mapping(data, "launch", config_path)
        if "argv" not in launch or not isinstance(launch["argv"], list):
            raise ValueError(f"{config_path}: launch.argv must be a list")
        prompt_input_mode = launch.get("prompt_input_mode")
        if prompt_input_mode not in {"stdin", "file_arg"}:
            raise ValueError(f"{config_path}: launch.prompt_input_mode must be stdin or file_arg")
        if prompt_input_mode == "file_arg" and not template_contains(launch["argv"], "{prompt_file}"):
            raise ValueError(f"{config_path}: launch.argv must include {{prompt_file}} for file_arg prompt delivery")
        if launch.get("model_argv"):
            model_position = str(launch.get("model_argv_position") or "")
            if model_position not in MODEL_ARGV_POSITIONS:
                raise ValueError(
                    f"{config_path}: launch.model_argv_position must be one of: "
                    f"{', '.join(sorted(MODEL_ARGV_POSITIONS))}"
                )
            if model_position == MODEL_ARGV_POSITION_BEFORE_STDIN_SENTINEL and (
                prompt_input_mode != "stdin" or launch["argv"][-1:] != ["-"]
            ):
                raise ValueError(
                    f"{config_path}: launch.model_argv_position={MODEL_ARGV_POSITION_BEFORE_STDIN_SENTINEL} "
                    "requires stdin prompt delivery and launch.argv ending with '-'"
                )
            if model_position == MODEL_ARGV_POSITION_APPEND and prompt_input_mode == "stdin":
                raise ValueError(
                    f"{config_path}: launch.model_argv_position={MODEL_ARGV_POSITION_APPEND} is not valid with "
                    "stdin prompt delivery; put model args directly in argv or use before_stdin_sentinel"
                )
        validate_when_conditions(launch["argv"], LAUNCH_CONDITIONS, "launch.argv", config_path)
        validate_when_conditions(launch.get("model_argv") or [], LAUNCH_CONDITIONS, "launch.model_argv", config_path)
        skill_exposure = required_mapping(data, "skill_exposure", config_path)
        mechanism_type = str(skill_exposure.get("mechanism_type") or "none")
        if mechanism_type not in SUPPORTED_SKILL_EXPOSURE_MECHANISMS:
            raise ValueError(
                f"{config_path}: skill_exposure.mechanism_type must be one of: "
                f"{', '.join(sorted(SUPPORTED_SKILL_EXPOSURE_MECHANISMS))}"
            )
        for field_name in ("setup_action", "probe_action", "disable_action", "launch_args"):
            validate_when_conditions(
                skill_exposure.get(field_name) or [],
                SKILL_EXPOSURE_CONDITIONS,
                f"skill_exposure.{field_name}",
                config_path,
            )

        events = parser_config(data, "events", config_path)
        usage = parser_config(data, "usage", config_path)
        activity = parser_config(data, "activity", config_path)
        exit_config = required_mapping(data, "exit", config_path)
        exit_classifier = str(exit_config.get("classifier") or "")
        final_message = required_mapping(data, "final_message", config_path)
        final_message_source_type = str(final_message.get("source_type") or "file")
        final_message_parser = str(final_message["parser"]) if final_message.get("parser") else None
        validate_event_parser(events.parser)
        validate_usage_parser(usage.parser)
        validate_activity_parser(activity.parser)
        validate_exit_config(exit_config)
        validate_final_message_config(final_message_source_type, final_message_parser)

        return cls(
            source_path=config_path,
            raw=data,
            source_sha256=hashlib.sha256(source.encode("utf-8")).hexdigest(),
            name=str(data["name"]),
            display_name=str(data["display_name"]),
            default_model=str(data.get("default_model") or UNSPECIFIED_MODEL),
            model_env=str(data["model_env"]) if data.get("model_env") else None,
            requires_explicit_model=bool(data.get("requires_explicit_model", False)),
            agent_home_env=str(data["agent_home_env"]),
            container_home=str(data["container_home"]),
            legacy_artifact_prefixes=legacy_artifact_prefixes(data, config_path),
            images=dict(data.get("images") or {}),
            auth=auth_config(data),
            launch=launch,
            skill_exposure=skill_exposure,
            final_message=final_message,
            events=events,
            usage=usage,
            activity=activity,
            exit_config=exit_config,
            exit_classifier=exit_classifier,
            availability_probe=[str(item) for item in data.get("availability_probe") or []],
        )


def required_mapping(data: Mapping[str, Any], key: str, config_path: Path) -> dict[str, Any]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{config_path}: {key} must be a mapping")
    return value


def parser_config(data: Mapping[str, Any], key: str, config_path: Path) -> ParserConfig:
    raw = required_mapping(data, key, config_path)
    parser = raw.get("parser")
    if not parser:
        raise ValueError(f"{config_path}: {key}.parser is required")
    return ParserConfig(
        parser=str(parser),
        fidelity=str(raw["fidelity"]) if raw.get("fidelity") else None,
        is_cumulative=bool(raw["is_cumulative"]) if "is_cumulative" in raw else None,
    )


def template_fields(value: str) -> set[str]:
    fields = set()
    for _literal, field_name, _format_spec, _conversion in string.Formatter().parse(value):
        if field_name is None:
            continue
        if field_name == "":
            raise ValueError("Adapter templates must not use positional '{}' placeholders")
        if "." in field_name or "[" in field_name or "]" in field_name:
            raise ValueError("Adapter templates must not use attribute or index access")
        fields.add(field_name)
    return fields


def render_string(value: str, values: Mapping[str, Any]) -> str:
    fields = template_fields(value)
    unknown = fields.difference(values)
    if unknown:
        raise ValueError(f"Unknown adapter template placeholder(s): {', '.join(sorted(unknown))}")
    if PROMPT_TEXT_PLACEHOLDER in fields:
        raise ValueError("Adapter templates must not reference prompt_text")
    return value.format(**values)


def auth_config(data: Mapping[str, Any]) -> dict[str, Any]:
    auth = dict(data.get("auth") or {})
    if "files" not in auth and isinstance(data.get("auth_mounts"), list):
        auth["files"] = data["auth_mounts"]
    return auth


def render_list(values: list[Any], render_values: Mapping[str, Any]) -> list[str]:
    rendered = []
    for item in values:
        if isinstance(item, dict):
            condition = item.get("when")
            if condition and not render_values.get(str(condition)):
                continue
            rendered.extend(render_list(list(item.get("args") or []), render_values))
            continue
        if not isinstance(item, str):
            raise ValueError(f"Adapter argv/action entries must be strings; got {item!r}")
        rendered.append(render_string(item, render_values))
    return rendered


def maybe_render_path(value: Any, render_values: Mapping[str, Any]) -> Path | None:
    if not value:
        return None
    return Path(render_string(str(value), render_values))


class ConfigurableAgentAdapter(AgentAdapter):
    """Concrete adapter driven by a validated YAML config."""

    def __init__(self, config_path: Path) -> None:
        self._cfg = AgentConfig.load(config_path)

    @property
    def name(self) -> str:
        return self._cfg.name

    @property
    def display_name(self) -> str:
        return self._cfg.display_name

    @property
    def default_model(self) -> str:
        return self._cfg.default_model

    @property
    def agent_home_env(self) -> str:
        return self._cfg.agent_home_env

    @property
    def container_home(self) -> str:
        return self._cfg.container_home

    def model_from_env(self, env: Mapping[str, str]) -> str:
        explicit_model = env.get("BENCHMARK_AGENT_MODEL") or (
            env.get(self._cfg.model_env) if self._cfg.model_env else None
        )
        if explicit_model:
            return explicit_model
        if self._cfg.requires_explicit_model:
            accepted = ["BENCHMARK_AGENT_MODEL"]
            if self._cfg.model_env:
                accepted.append(self._cfg.model_env)
            raise ValueError(
                f"{self.display_name} requires an explicit benchmark model. " f"Set one of: {', '.join(accepted)}."
            )
        return self.default_model or UNSPECIFIED_MODEL

    def model_was_explicit(self, env: Mapping[str, str]) -> bool:
        return bool(env.get("BENCHMARK_AGENT_MODEL") or (env.get(self._cfg.model_env) if self._cfg.model_env else None))

    def model_env_names(self) -> tuple[str, ...]:
        return (self._cfg.model_env,) if self._cfg.model_env else ()

    def build_args(self) -> dict[str, str]:
        build = self._cfg.raw.get("build") or {}
        if not isinstance(build, dict):
            return {}
        args = {}
        for key, value in (build.get("args") or {}).items():
            if isinstance(value, dict):
                raise ValueError(f"{self._cfg.source_path}: build.args.{key} must be a scalar profile value")
            else:
                args[str(key)] = render_string(str(value), {"agent": self.name})
        return args

    def image_targets(self) -> AgentImageTargets:
        render_values = {"agent": self.name}
        skills = render_string(
            str(self._cfg.images.get("skills") or "agent-skills-benchmark:{agent}-skills"), render_values
        )
        baseline = render_string(
            str(self._cfg.images.get("baseline") or "agent-skills-benchmark:{agent}-baseline"), render_values
        )
        report = render_string(str(self._cfg.images.get("report") or skills), render_values)
        return AgentImageTargets(
            skills=skills,
            baseline=baseline,
            report=report,
        )

    def auth_mounts(self, host_config) -> list[DockerMount]:
        auth = self._cfg.auth
        if not auth:
            return []
        host_home = Path(getattr(host_config, "host_agent_home", Path.home()))
        mounts = []
        for item in auth.get("files") or []:
            if not isinstance(item, dict):
                continue
            source_name = str(item.get("source") or "")
            target_name = str(item.get("target") or source_name)
            if not source_name or not target_name:
                continue
            if Path(source_name).name != source_name:
                raise ValueError(f"{self._cfg.source_path}: auth file source must be a file name: {source_name}")
            if Path(target_name).name != target_name:
                raise ValueError(f"{self._cfg.source_path}: auth file target must be a file name: {target_name}")
            mounts.append(
                DockerMount(
                    host_path=host_home / source_name,
                    container_path=f"{self.container_home.rstrip('/')}/{target_name}",
                    read_only=bool(item.get("read_only", True)),
                    description=str(item.get("description") or f"{self.display_name} auth/config"),
                    required=bool(item.get("required", False)),
                )
            )
        return mounts

    def host_home_from_env(self, env: Mapping[str, str]) -> Path:
        auth = self._cfg.auth if self._cfg.auth else {}
        host_home_env = auth.get("host_home_env")
        if host_home_env and env.get(str(host_home_env)):
            return Path(str(env[str(host_home_env)])).expanduser()
        default_host_home = auth.get("default_host_home")
        if default_host_home:
            return Path(str(default_host_home)).expanduser()
        return Path.home() / f".{self.name}"

    def mount_auth_from_env(self, env: Mapping[str, str]) -> bool:
        auth = self._cfg.auth if self._cfg.auth else {}
        mount_env = auth.get("mount_env")
        if not mount_env or str(mount_env) not in env:
            return bool(auth.get("mount_by_default", True))
        value = env[str(mount_env)]
        if value not in {"true", "false"}:
            raise ValueError(f"{mount_env} must be true or false; got {value}")
        return value == "true"

    def runtime_env(self, config) -> dict[str, str]:
        env = {
            self.agent_home_env: self.container_home,
            "BENCHMARK_AGENT_HOME": self.container_home,
            "BENCHMARK_AGENT": self.name,
        }
        model_was_explicit = getattr(config, "model_was_explicit", None)
        if model_was_explicit is None:
            model_was_explicit = getattr(config, "agent_model_was_explicit", False)
        if self._cfg.requires_explicit_model and not model_was_explicit:
            accepted = ["BENCHMARK_AGENT_MODEL"]
            if self._cfg.model_env:
                accepted.append(self._cfg.model_env)
            raise ValueError(
                f"{self.display_name} requires an explicit benchmark model before runtime setup. "
                f"Set one of: {', '.join(accepted)}."
            )
        if model_was_explicit:
            env["BENCHMARK_AGENT_MODEL"] = getattr(config, "agent_model", self.default_model)
        for key, value in (self._cfg.raw.get("runtime_env") or {}).items():
            env[str(key)] = render_string(str(value), {"container_home": self.container_home, "agent": self.name})
        return env

    def passthrough_env_names(self) -> tuple[str, ...]:
        return tuple(str(item) for item in self._cfg.raw.get("passthrough_env") or ())

    def launch_spec(self, config: AgentLaunchContext) -> AgentLaunchSpec:
        if self._cfg.requires_explicit_model and not config.model_was_explicit:
            accepted = ["BENCHMARK_AGENT_MODEL"]
            if self._cfg.model_env:
                accepted.append(self._cfg.model_env)
            raise ValueError(
                f"{self.display_name} requires an explicit benchmark model before launch. "
                f"Set one of: {', '.join(accepted)}."
            )
        final_message_dest = config.final_message_dest
        render_values = {
            "agent": self.name,
            "model": config.model,
            "workspace_dir": str(config.workspace_dir),
            "prompt_file": str(config.prompt_file),
            "result_dir": str(config.result_dir),
            "events_dest": str(config.events_dest),
            "stderr_dest": str(config.stderr_dest),
            "final_message_dest": str(final_message_dest),
            "container_home": self.container_home,
            "model_was_explicit": config.model_was_explicit,
        }
        argv = render_list(list(self._cfg.launch["argv"]), render_values)
        if config.model_was_explicit and self._cfg.launch.get("model_argv"):
            model_argv = render_list(list(self._cfg.launch["model_argv"]), render_values)
            model_position = str(self._cfg.launch["model_argv_position"])
            if model_position == MODEL_ARGV_POSITION_BEFORE_STDIN_SENTINEL:
                argv = [*argv[:-1], *model_argv, argv[-1]]
            elif model_position == MODEL_ARGV_POSITION_APPEND:
                argv.extend(model_argv)
            else:
                raise ValueError(f"Unsupported launch.model_argv_position: {model_position}")
        env = {}
        for key, value in (self._cfg.launch.get("environment") or {}).items():
            env[str(key)] = render_string(str(value), render_values)
        env.update({self.agent_home_env: self.container_home})
        return AgentLaunchSpec(
            argv=argv,
            cwd=config.workspace_dir,
            prompt_file=config.prompt_file,
            prompt_input_mode=str(self._cfg.launch["prompt_input_mode"]),
            stdout_events_dest=config.events_dest,
            stderr_dest=config.stderr_dest,
            final_message_dest=final_message_dest,
            environment=env,
            login_shell=bool(self._cfg.launch.get("login_shell", False)),
            approval_flags=[str(item) for item in self._cfg.launch.get("approval_flags") or []],
            sandbox_flags=[str(item) for item in self._cfg.launch.get("sandbox_flags") or []],
            bypass_reason=str(self._cfg.launch.get("bypass_reason")) if self._cfg.launch.get("bypass_reason") else None,
            launch_timeout=config.timeout_seconds,
        )

    def skill_exposure(self, config: SkillExposureContext) -> SkillExposureSpec:
        render_values = {
            "agent": self.name,
            "container_home": str(config.container_home),
            "result_dir": str(config.result_dir),
            "skills_dir": str(config.container_home / "skills"),
        }
        raw = self._cfg.skill_exposure
        return SkillExposureSpec(
            mechanism_type=str(raw.get("mechanism_type") or "none"),
            container_home=config.container_home,
            skill_root=maybe_render_path(raw.get("skill_root"), render_values),
            source_paths=[Path(render_string(str(item), render_values)) for item in raw.get("source_paths") or []],
            setup_action=render_list(list(raw.get("setup_action") or []), render_values),
            probe_action=render_list(list(raw.get("probe_action") or []), render_values),
            disable_action=render_list(list(raw.get("disable_action") or []), render_values),
            launch_args=render_list(list(raw.get("launch_args") or []), render_values),
            environment={
                str(key): render_string(str(value), render_values)
                for key, value in (raw.get("environment") or {}).items()
            },
            metadata_files=[Path(render_string(str(item), render_values)) for item in raw.get("metadata_files") or []],
            expected_post_setup_state=(
                str(raw.get("expected_post_setup_state")) if raw.get("expected_post_setup_state") else None
            ),
            disable_packaged_source=bool(raw.get("disable_packaged_source", False)),
        )

    def availability_probe(self) -> list[str]:
        return list(self._cfg.availability_probe)

    def normalize_event(self, raw_line: str) -> dict[str, Any] | None:
        return normalize_event_with_parser(raw_line, self._cfg.events.parser)

    def parse_usage(self, events_path: Path) -> dict[str, Any]:
        usage = parse_usage_from_events(events_path, self._cfg.usage)
        if self._cfg.usage.fidelity:
            usage.setdefault("usage_fidelity", self._cfg.usage.fidelity)
        if self._cfg.usage.is_cumulative is not None:
            usage.setdefault("is_cumulative", self._cfg.usage.is_cumulative)
        return usage

    def parse_activity(self, events_path: Path) -> dict[str, Any]:
        return parse_activity_from_events(events_path, self._cfg.activity)

    def final_message_source(self, result_dir: Path) -> FinalMessageSource:
        render_values = {"result_dir": str(result_dir), "agent": self.name}
        source_type = str(self._cfg.final_message.get("source_type") or "file")
        return FinalMessageSource(
            source_type=source_type,
            path=maybe_render_path(self._cfg.final_message.get("path"), render_values),
            event_selector=(
                dict(self._cfg.final_message["event_selector"])
                if isinstance(self._cfg.final_message.get("event_selector"), dict)
                else None
            ),
            tail_bytes=(
                int(self._cfg.final_message["tail_bytes"])
                if self._cfg.final_message.get("tail_bytes") is not None
                else None
            ),
            parser=str(self._cfg.final_message["parser"]) if self._cfg.final_message.get("parser") else None,
            parser_warnings=[str(item) for item in self._cfg.final_message.get("parser_warnings") or []],
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "agent": self.name,
            "display_name": self.display_name,
            "config_path": str(self._cfg.source_path),
            "config_sha256": self._cfg.source_sha256,
            "adapter_type": "ConfigurableAgentAdapter",
        }

    def exit_summary(self, exit_code: int, stderr_path: Path) -> dict[str, Any]:
        return classify_exit(exit_code, stderr_path, self._cfg.exit_config)

    def artifact_alias_prefixes(self) -> tuple[str, ...]:
        return self._cfg.legacy_artifact_prefixes
