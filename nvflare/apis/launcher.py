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

"""Shared launch-mode identity for job and task launchers."""

from enum import Enum


class LauncherMode(str, Enum):
    PROCESS = "process"
    DOCKER = "docker"
    K8S = "k8s"
    SLURM = "slurm"

    @classmethod
    def validate(cls, value, name="launcher mode") -> str:
        if isinstance(value, cls):
            return value.value
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be one of {[mode.value for mode in cls]} but got {value!r}")
        try:
            return cls(value).value
        except ValueError as e:
            raise ValueError(f"{name} must be one of {[mode.value for mode in cls]} but got {value!r}") from e
