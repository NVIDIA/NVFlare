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

"""YAML-driven SDK adapter implementation."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from .base import SdkAdapter, SdkSkillsSetup, SdkSource, SdkWheelBuild, SdkWheelVariant


def required_mapping(data: Mapping[str, Any], key: str, config_path: Path) -> dict[str, Any]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{config_path}: {key} must be a mapping")
    return value


def string_tuple(value: Any, *, label: str, config_path: Path) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValueError(f"{config_path}: {label} must be a list")
    return tuple(str(item) for item in value)


def format_config_value(value: Any, values: Mapping[str, str], *, label: str, config_path: Path) -> str:
    try:
        return str(value).format(**values)
    except KeyError as exc:
        raise ValueError(f"{config_path}: unknown placeholder {{{exc.args[0]}}} in {label}") from exc
    except ValueError as exc:
        raise ValueError(f"{config_path}: invalid template syntax in {label}: {exc}") from exc


def validate_skills_setup(skills: Mapping[str, Any], config_path: Path) -> None:
    setup = required_mapping(skills, "setup", config_path)
    setup_type = str(setup.get("type") or "")
    if setup_type not in {"command", "copy", "none"}:
        raise ValueError(f"{config_path}: skills.setup.type must be command, copy, or none")
    if setup_type == "command" and not setup.get("install_command"):
        raise ValueError(f"{config_path}: skills.setup.install_command is required for skills.setup.type=command")
    if setup_type == "copy" and not setup.get("source_path"):
        raise ValueError(f"{config_path}: skills.setup.source_path is required for skills.setup.type=copy")


def validate_wheel_build(build: Mapping[str, Any], source_type: str, config_path: Path) -> None:
    build_type = str(build.get("type") or "")
    if build_type not in {"uv_wheel", "provided_wheels"}:
        raise ValueError(f"{config_path}: build.type must be uv_wheel or provided_wheels")
    if build_type == "uv_wheel" and source_type != "repo":
        raise ValueError(f"{config_path}: build.type=uv_wheel requires source.type=repo")
    if build_type == "provided_wheels" and source_type != "wheels":
        raise ValueError(f"{config_path}: build.type=provided_wheels requires source.type=wheels")


@dataclass(frozen=True)
class SdkConfig:
    source_path: Path
    raw: dict[str, Any]
    source_sha256: str
    name: str
    display_name: str
    package_name: str
    import_name: str
    source: dict[str, Any]
    build: dict[str, Any]
    docker: dict[str, Any]
    skills: dict[str, Any]

    @classmethod
    def load(cls, config_path: Path) -> "SdkConfig":
        source = config_path.read_text(encoding="utf-8")
        data = yaml.safe_load(source) or {}
        if not isinstance(data, dict):
            raise ValueError(f"{config_path} must contain a YAML object")
        required = ("name", "display_name", "package_name", "import_name", "source", "build", "docker", "skills")
        missing = [name for name in required if name not in data]
        if missing:
            raise ValueError(f"{config_path} is missing required field(s): {', '.join(missing)}")

        source_config = required_mapping(data, "source", config_path)
        source_type = str(source_config.get("type") or "")
        if source_type not in {"repo", "wheels"}:
            raise ValueError(f"{config_path}: source.type must be repo or wheels")
        if source_type == "repo":
            if not source_config.get("path"):
                raise ValueError(f"{config_path}: source.path is required for source.type=repo")
            markers = string_tuple(source_config.get("markers"), label="source.markers", config_path=config_path)
            if not markers:
                raise ValueError(f"{config_path}: source.markers must not be empty for source.type=repo")
        else:
            wheels = required_mapping(source_config, "wheels", config_path)
            for variant_name in ("skills", "baseline"):
                if not wheels.get(variant_name):
                    raise ValueError(f"{config_path}: source.wheels.{variant_name} is required for source.type=wheels")

        build = required_mapping(data, "build", config_path)
        validate_wheel_build(build, source_type, config_path)
        docker = required_mapping(data, "docker", config_path)
        skills = required_mapping(data, "skills", config_path)
        validate_skills_setup(skills, config_path)
        variants = required_mapping(build, "variants", config_path)
        for variant_name in ("skills", "baseline"):
            variant = required_mapping(variants, variant_name, config_path)
            wheel_globs = string_tuple(
                variant.get("wheel_globs"),
                label=f"build.variants.{variant_name}.wheel_globs",
                config_path=config_path,
            )
            if not wheel_globs:
                raise ValueError(f"{config_path}: build.variants.{variant_name}.wheel_globs must not be empty")

        return cls(
            source_path=config_path,
            raw=data,
            source_sha256=hashlib.sha256(source.encode("utf-8")).hexdigest(),
            name=str(data["name"]),
            display_name=str(data["display_name"]),
            package_name=str(data["package_name"]),
            import_name=str(data["import_name"]),
            source=source_config,
            build=build,
            docker=docker,
            skills=skills,
        )


class ConfigurableSdkAdapter(SdkAdapter):
    """Concrete SDK adapter driven by a validated YAML config."""

    def __init__(self, config_path: Path) -> None:
        self._cfg = SdkConfig.load(config_path)

    @property
    def name(self) -> str:
        return self._cfg.name

    @property
    def display_name(self) -> str:
        return self._cfg.display_name

    @property
    def package_name(self) -> str:
        return self._cfg.package_name

    @property
    def import_name(self) -> str:
        return self._cfg.import_name

    @property
    def build_env_name(self) -> str:
        return str(self._cfg.build.get("env_name") or "")

    def wheel_build(self) -> SdkWheelBuild:
        return SdkWheelBuild(build_type=str(self._cfg.build.get("type") or ""))

    def source(self, *, repo_root: Path, home: Path) -> SdkSource:
        values = {
            "repo_root": str(repo_root),
            "home": str(home),
        }
        source_type = str(self._cfg.source.get("type"))
        if source_type == "repo":
            repo_path = Path(
                format_config_value(
                    self._cfg.source["path"],
                    values,
                    label="source.path",
                    config_path=self._cfg.source_path,
                )
            ).expanduser()
            return SdkSource(
                source_type="repo",
                repo_path=repo_path,
                repo_markers=string_tuple(
                    self._cfg.source.get("markers"),
                    label="source.markers",
                    config_path=self._cfg.source_path,
                ),
            )
        wheels = required_mapping(self._cfg.source, "wheels", self._cfg.source_path)
        return SdkSource(
            source_type="wheels",
            wheel_paths={
                "skills": Path(
                    format_config_value(
                        wheels["skills"],
                        values,
                        label="source.wheels.skills",
                        config_path=self._cfg.source_path,
                    )
                ).expanduser(),
                "baseline": Path(
                    format_config_value(
                        wheels["baseline"],
                        values,
                        label="source.wheels.baseline",
                        config_path=self._cfg.source_path,
                    )
                ).expanduser(),
            },
        )

    def wheel_variant(self, name: str) -> SdkWheelVariant:
        variants = required_mapping(self._cfg.build, "variants", self._cfg.source_path)
        if name not in variants:
            raise ValueError(f"{self._cfg.source_path}: unknown SDK wheel variant {name!r}")
        raw = required_mapping(variants, name, self._cfg.source_path)
        wheel_globs = string_tuple(
            raw.get("wheel_globs"),
            label=f"build.variants.{name}.wheel_globs",
            config_path=self._cfg.source_path,
        )
        wheel_exclude_globs = string_tuple(
            raw.get("wheel_exclude_globs"),
            label=f"build.variants.{name}.wheel_exclude_globs",
            config_path=self._cfg.source_path,
        )
        return SdkWheelVariant(
            name=name,
            label=str(raw.get("label") or name),
            build_env_value=str(raw.get("build_env_value", "")),
            wheel_globs=wheel_globs,
            wheel_exclude_globs=wheel_exclude_globs,
        )

    def skills_setup(self, *, repo_root: Path, home: Path) -> SdkSkillsSetup:
        values = {
            "repo_root": str(repo_root),
            "home": str(home),
        }
        raw = required_mapping(self._cfg.skills, "setup", self._cfg.source_path)
        setup_type = str(raw.get("type") or "")
        source_path = raw.get("source_path")
        return SdkSkillsSetup(
            setup_type=setup_type,
            source_path=(
                Path(
                    format_config_value(
                        source_path,
                        values,
                        label="skills.setup.source_path",
                        config_path=self._cfg.source_path,
                    )
                ).expanduser()
                if source_path
                else None
            ),
            install_command=str(raw.get("install_command", "")),
            list_command=str(raw.get("list_command", "")),
            install_output=str(raw.get("install_output", "skills_build_install.json")),
            list_output=str(raw.get("list_output", "skills_list.json")),
            expected_source=str(raw.get("expected_source", "local_sdk_wheel")),
        )

    def docker_build_args(self) -> dict[str, str]:
        setup = self.skills_setup(repo_root=Path(), home=Path.home())
        return {
            "SDK_PACKAGE_NAME": self.package_name,
            "SDK_IMPORT_NAME": self.import_name,
            "SDK_VERSION_COMMAND": str(self._cfg.docker.get("version_command", "")),
            "SKILLS_SETUP_TYPE": setup.setup_type,
            "SKILLS_INSTALL_COMMAND": setup.install_command,
            "SKILLS_LIST_COMMAND": setup.list_command,
            "SKILLS_INSTALL_OUTPUT": setup.install_output,
            "SKILLS_LIST_OUTPUT": setup.list_output,
            "SKILLS_INSTALL_EXPECTED_SOURCE": setup.expected_source,
        }

    def metadata(self) -> dict[str, object]:
        return {
            "name": self.name,
            "display_name": self.display_name,
            "package_name": self.package_name,
            "import_name": self.import_name,
            "source_path": str(self._cfg.source_path),
            "source_sha256": self._cfg.source_sha256,
        }
