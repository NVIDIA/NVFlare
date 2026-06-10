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

"""SDK adapter contracts for benchmark Docker setup."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SdkWheelVariant:
    name: str
    label: str
    build_env_value: str
    wheel_globs: tuple[str, ...]
    wheel_exclude_globs: tuple[str, ...] = ()


@dataclass(frozen=True)
class SdkSource:
    source_type: str
    repo_path: Path | None = None
    repo_markers: tuple[str, ...] = ()
    wheel_paths: dict[str, Path] | None = None


@dataclass(frozen=True)
class SdkSkillsSetup:
    setup_type: str
    source_path: Path | None = None
    install_command: str = ""
    list_command: str = ""
    install_output: str = "skills_build_install.json"
    list_output: str = "skills_list.json"
    expected_source: str = "local_sdk_wheel"


@dataclass(frozen=True)
class SdkWheelBuild:
    build_type: str


class SdkAdapter(ABC):
    """Contract for SDK-specific Docker build mechanics.

    The SDK adapter owns source-package details: how to find the checkout, how
    to build/stage the skills and baseline wheels, and how to preinstall skills
    in the runtime image. Agent adapters still own agent CLI behavior.
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
    def package_name(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def import_name(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def build_env_name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def source(self, *, repo_root: Path, home: Path) -> SdkSource:
        raise NotImplementedError

    @abstractmethod
    def wheel_build(self) -> SdkWheelBuild:
        raise NotImplementedError

    @abstractmethod
    def wheel_variant(self, name: str) -> SdkWheelVariant:
        raise NotImplementedError

    @abstractmethod
    def skills_setup(self, *, repo_root: Path, home: Path) -> SdkSkillsSetup:
        raise NotImplementedError

    @abstractmethod
    def docker_build_args(self) -> dict[str, str]:
        raise NotImplementedError

    @abstractmethod
    def metadata(self) -> dict[str, object]:
        raise NotImplementedError
