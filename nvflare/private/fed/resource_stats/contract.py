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

"""Numeric formatting and validation at the GPU discovery helper boundary."""

import re
from collections.abc import Mapping
from decimal import ROUND_FLOOR, Decimal, localcontext

SCHEMA_VERSION = "1.0"
MAX_GPU_GROUPS = 4_096
GPU_KINDS = frozenset({"cuda_device_unclassified", "full_gpu", "mig_compute_instance"})

_U64_MAX = 2**64 - 1
_MAX_CPU_UNITS = Decimal("1048576")
_INTEGER_PATTERN = re.compile(r"^[0-9]{1,39}(?:\.0{1,9})?$")
_MODEL_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 ._()+/@-]{0,127}$")
_MIG_PROFILE_PATTERN = re.compile(r"^[1-9][0-9]*g\.[1-9][0-9]*gb(?:\+me)?$")
_FORBIDDEN_MODEL_PATTERNS = (
    re.compile(r"(?:GPU|MIG|MIG-GPU)-[0-9A-Fa-f-]{16,}", re.IGNORECASE),
    re.compile(r"(?:[0-9A-Fa-f]{4}:)?[0-9A-Fa-f]{2}:[0-9A-Fa-f]{2}\.[0-7]"),
    re.compile(r"\b(?:serial|uuid|cuda_visible_devices|pci_bus_id|hostname)\b", re.IGNORECASE),
)


class ContractError(ValueError):
    """A GPU helper result has invalid or private measurement fields."""


def normalize_quota_units(quota_us: int, period_us: int) -> str:
    """Return a positive CPU quota ratio, conservatively floored to nine digits."""

    for name, value in (("quota_us", quota_us), ("period_us", period_us)):
        if type(value) is not int or not 0 < value <= _U64_MAX:
            raise ValueError(f"{name} must be a positive unsigned 64-bit integer")
    with localcontext() as context:
        context.prec = 100
        ratio = (Decimal(quota_us) / Decimal(period_us)).quantize(Decimal("0.000000001"), rounding=ROUND_FLOOR)
    if not 0 < ratio <= _MAX_CPU_UNITS:
        raise ValueError("normalized CPU quota is outside the visible-unit bound or nine-decimal precision")
    return format(ratio, "f").rstrip("0").rstrip(".")


def _positive_integer(value, maximum, field):
    if not isinstance(value, str) or not _INTEGER_PATTERN.fullmatch(value):
        raise ContractError(f"{field} must be an integer-valued decimal string")
    number = int(value.partition(".")[0])
    if not 0 < number <= maximum:
        raise ContractError(f"{field} must be positive and no greater than {maximum}")
    return str(number)


def normalize_gpu_capacity(value: Mapping) -> dict:
    """Accept only bounded GPU counts and non-identifying metadata from the helper.

    Construct the supported shape directly so unknown/nested fields cannot reach
    a report. Count zero is an empty groups list; unavailable discovery is None
    and is handled by the helper owner before calling this function.
    """

    if not isinstance(value, Mapping) or set(value) != {"groups"}:
        raise ContractError("GPU capacity must contain only groups")
    groups = value["groups"]
    if not isinstance(groups, list) or len(groups) > MAX_GPU_GROUPS:
        raise ContractError(f"GPU capacity must contain 0..{MAX_GPU_GROUPS} groups")
    normalized = {}
    total = 0
    for group in groups:
        if not isinstance(group, Mapping) or not {"kind", "count"} <= group.keys():
            raise ContractError("GPU group must contain kind and count")
        if set(group) - {"kind", "count", "model", "memory_bytes", "mig_profile"}:
            raise ContractError("GPU group contains unsupported fields")
        kind = group["kind"]
        if not isinstance(kind, str) or kind not in GPU_KINDS:
            raise ContractError("GPU group has an unsupported kind")
        item = {
            "kind": kind,
            "count": _positive_integer(group["count"], MAX_GPU_GROUPS, "GPU count"),
        }
        total += int(item["count"])
        if total > MAX_GPU_GROUPS:
            raise ContractError(f"GPU total exceeds {MAX_GPU_GROUPS} CUDA devices")
        if "model" in group:
            model = group["model"]
            if (
                not isinstance(model, str)
                or not _MODEL_PATTERN.fullmatch(model)
                or any(pattern.search(model) for pattern in _FORBIDDEN_MODEL_PATTERNS)
            ):
                raise ContractError("GPU model must be a non-identifying ASCII label of at most 128 characters")
            item["model"] = model
        if "memory_bytes" in group:
            item["memory_bytes"] = _positive_integer(group["memory_bytes"], _U64_MAX, "GPU memory_bytes")
        if "mig_profile" in group:
            profile = group["mig_profile"]
            if (
                kind != "mig_compute_instance"
                or not isinstance(profile, str)
                or not _MIG_PROFILE_PATTERN.fullmatch(profile)
            ):
                raise ContractError("GPU mig_profile must describe a MIG compute-instance group")
            item["mig_profile"] = profile
        key = tuple(item.get(field, "") for field in ("kind", "model", "memory_bytes", "mig_profile"))
        if key in normalized:
            raise ContractError("GPU groups must be consolidated and unique by identifying fields")
        normalized[key] = item
    return {"groups": [normalized[key] for key in sorted(normalized)]}
