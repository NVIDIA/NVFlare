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

"""Retain one execution-start capacity observation without resampling."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from .contract import normalize_measurement, normalize_scope_issues


class ResourceCapacityRecorder:
    """Keep the already numeric probe output and its measurement quality."""

    def __init__(self):
        self._capacity = {"status": "unavailable", "issues": ["observation_incomplete"]}
        self._observed = False

    def observe(self, capacity: Mapping[str, Any]) -> None:
        if self._observed:
            raise ValueError("capacity is observed once at measurement startup")
        if not isinstance(capacity, Mapping):
            raise TypeError("capacity must be an object")
        value = dict(capacity)
        if "status" not in value and not all(name in value for name in ("cpu", "memory", "gpu")):
            value.update(status="partial", issues=["observation_incomplete"])
        codes = value.pop("issues", [])
        checked = normalize_measurement("resource_capacity", value)
        tags = [{"metric": "resource_capacity", "code": code} for code in codes]
        normalize_scope_issues(tags, {"resource_capacity": checked})
        self._capacity = checked
        if codes:
            self._capacity["issues"] = deepcopy(codes)
        self._observed = True

    def mark_prior_observation_incomplete(self) -> None:
        self._capacity["status"] = (
            "partial" if any(name in self._capacity for name in ("cpu", "memory", "gpu")) else "unavailable"
        )
        self._capacity["issues"] = sorted(set(self._capacity.get("issues", [])) | {"observation_incomplete"})

    def finish(self) -> dict[str, Any]:
        return deepcopy(self._capacity)
