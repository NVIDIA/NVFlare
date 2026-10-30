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

"""Normalize one startup capacity observation; timing belongs to the participant."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from decimal import Decimal, localcontext
from typing import Any

from .contract import GPU_KINDS, MAX_GPU_GROUPS, normalize_measurement
from .probes._shared import _decimal, _decimal_text, _normalize_architecture, _normalize_model


class ResourceCapacityRecorder:
    """Retain one normalized startup snapshot without a separate duration."""

    def __init__(self):
        self._capacity = {"status": "unavailable", "issues": ["observation_incomplete"]}
        self._observed = False

    def observe(self, capacity: Mapping[str, Any]) -> None:
        if self._observed:
            raise ValueError("capacity is observed once at measurement startup")
        if not isinstance(capacity, Mapping) or set(capacity) - {"cpu", "memory", "gpu", "status", "issues"}:
            raise ValueError("capacity has unsupported fields")
        dimensions = {name: capacity[name] for name in ("cpu", "memory", "gpu") if name in capacity}
        normalized = self._public_capacity(self._normalize_capacity(dimensions))
        if "status" in capacity or "issues" in capacity:
            normalized.update({name: capacity[name] for name in ("status", "issues") if name in capacity})
        elif not all(name in normalized for name in ("cpu", "memory", "gpu")):
            normalized.update(status="partial", issues=["observation_incomplete"])
        self._capacity = normalize_measurement("resource_capacity", normalized)
        self._observed = True

    def mark_prior_observation_incomplete(self) -> None:
        self._capacity["status"] = (
            "partial" if any(name in self._capacity for name in ("cpu", "memory", "gpu")) else "unavailable"
        )
        self._capacity["issues"] = sorted(set(self._capacity.get("issues", [])) | {"observation_incomplete"})

    def finish(self) -> dict[str, Any]:
        return deepcopy(self._capacity)

    @staticmethod
    def _public_capacity(capacity: Mapping[str, Any]) -> dict[str, Any]:
        result = {}
        for resource, value in capacity.items():
            if resource == "memory":
                result[resource] = {"bytes": _decimal_text(value["bytes"])}
                continue
            field = "units" if resource == "cpu" else "count"
            consolidated = {}
            for group in value["groups"]:
                metadata = {key: item for key, item in group.items() if key != field and item is not None}
                key = tuple(sorted(metadata.items()))
                if key not in consolidated:
                    consolidated[key] = {**metadata, field: Decimal(0)}
                with localcontext() as context:
                    context.prec = 100
                    consolidated[key][field] += group[field]
            result[resource] = {
                "groups": [{**group, field: _decimal_text(group[field])} for _, group in sorted(consolidated.items())]
            }
        return result

    @staticmethod
    def _normalize_capacity(capacity: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(capacity, Mapping):
            raise TypeError("capacity must be a mapping")
        if set(capacity) - {"cpu", "memory", "gpu"}:
            raise ValueError("capacity has unsupported fields")
        result: dict[str, Any] = {}
        if capacity.get("cpu") is not None:
            cpu = capacity["cpu"]
            if not isinstance(cpu, Mapping):
                raise TypeError("cpu capacity must be an object")
            raw_groups = cpu.get("groups")
            if not isinstance(raw_groups, list) or not raw_groups:
                raise TypeError("cpu.groups must be a non-empty list")
            groups = []
            for index, raw in enumerate(raw_groups):
                if not isinstance(raw, Mapping):
                    raise TypeError(f"cpu.groups[{index}] must be an object")
                group = {
                    "units": _decimal(raw.get("units"), f"cpu.groups[{index}].units", positive=True),
                    "model": _normalize_model(raw.get("model")),
                    "architecture": _normalize_architecture(raw.get("architecture")),
                }
                groups.append(group)
            result["cpu"] = {"groups": groups}
        if capacity.get("memory") is not None:
            memory = capacity["memory"]
            if not isinstance(memory, Mapping):
                raise TypeError("memory capacity must be an object")
            result["memory"] = {"bytes": _decimal(memory.get("bytes"), "memory.bytes", positive=True)}
        if capacity.get("gpu") is not None:
            gpu = capacity["gpu"]
            if not isinstance(gpu, Mapping) or not isinstance(gpu.get("groups"), list):
                raise TypeError("gpu.groups must be a list")
            if len(gpu["groups"]) > MAX_GPU_GROUPS:
                raise ValueError("gpu.groups exceeds the supported device limit")
            groups = []
            total_count = Decimal(0)
            for index, raw in enumerate(gpu["groups"]):
                if not isinstance(raw, Mapping) or raw.get("kind") not in GPU_KINDS:
                    raise ValueError(f"gpu.groups[{index}] has an unsupported kind")
                count = _decimal(raw.get("count"), f"gpu.groups[{index}].count", positive=True)
                if count != count.to_integral_value():
                    raise ValueError(f"gpu.groups[{index}].count must be an integer")
                total_count += count
                group: dict[str, Any] = {
                    "kind": raw["kind"],
                    "count": count,
                }
                model = _normalize_model(raw.get("model"))
                if model:
                    group["model"] = model
                if raw.get("memory_bytes") is not None:
                    group["memory_bytes"] = _decimal_text(
                        _decimal(raw["memory_bytes"], f"gpu.groups[{index}].memory_bytes", positive=True)
                    )
                mig_profile = raw.get("mig_profile")
                if isinstance(mig_profile, str) and mig_profile:
                    if raw["kind"] != "mig_compute_instance":
                        raise ValueError(f"gpu.groups[{index}].mig_profile requires a confirmed MIG instance")
                    group["mig_profile"] = mig_profile
                groups.append(group)
            if total_count > MAX_GPU_GROUPS:
                raise ValueError("gpu.groups exceeds the supported device limit")
            result["gpu"] = {"groups": groups}
        return result
