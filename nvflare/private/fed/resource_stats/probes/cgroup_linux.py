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

"""Best-effort Linux CPU and memory capacity from readable OS limits.

Use the smallest usable bound. A limit that cannot be checked adds one
incomplete-observation indication without discarding other readable bounds.
"""

from __future__ import annotations

import ctypes
import os
import platform
import re
from collections.abc import Mapping
from decimal import Decimal
from pathlib import Path
from typing import Any, Optional

from ..contract import normalize_quota_units
from ._shared import _decimal_text, _normalize_architecture, _normalize_model

_CGROUP_FILE_MISSING = object()
_CGROUP_FILE_UNREADABLE = object()
_V1_MEMORY_UNLIMITED_MIN = 1 << 60


def _read_text(path: Path) -> Optional[str]:
    try:
        return path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError):
        return None


def _read_cgroup_file(path: Path) -> str | object:
    """Read a cgroup file without conflating absence and unreadability."""

    try:
        return path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return _CGROUP_FILE_MISSING
    except (OSError, UnicodeError):
        return _CGROUP_FILE_UNREADABLE


def _incomplete(issues: Optional[set[str]]) -> None:
    if issues is not None:
        issues.add("observation_incomplete")


def _limit_text(path: Path, issues: Optional[set[str]], *, required: bool = False) -> Optional[str]:
    """Read one limit; an absent optional controller file is not an error."""

    value = _read_cgroup_file(path)
    if value is _CGROUP_FILE_UNREADABLE or (required and value is _CGROUP_FILE_MISSING):
        _incomplete(issues)
    return value if isinstance(value, str) else None


def _positive_sysconf(name: str) -> Optional[int]:
    try:
        value = os.sysconf(name)
    except (AttributeError, OSError, TypeError, ValueError):
        return None
    return value if type(value) is int and value > 0 else None


def _is_v1_memory_unlimited(limit: int, page_size: int) -> bool:
    """Recognize the page-aligned LONG_MAX sentinel emitted by cgroup v1."""

    if page_size <= 0:
        return False
    long_max = (1 << (ctypes.sizeof(ctypes.c_long) * 8 - 1)) - 1
    return limit == (long_max // page_size) * page_size


def _decode_mountinfo(value: str) -> str:
    return re.sub(r"\\([0-7]{3})", lambda match: chr(int(match.group(1), 8)), value)


def _mount_entries() -> Optional[list[dict[str, str]]]:
    text = _read_text(Path("/proc/self/mountinfo"))
    if text is None:
        return None
    entries = []
    for line in text.splitlines():
        before, separator, after = line.partition(" - ")
        if not separator:
            return None
        left = before.split()
        right = after.split()
        if len(left) >= 6 and len(right) >= 3:
            entries.append(
                {
                    "root": _decode_mountinfo(left[3]),
                    "mount_point": _decode_mountinfo(left[4]),
                    "fs_type": right[0],
                    "super_options": right[2],
                }
            )
        else:
            return None
    return entries


def _cgroup_records() -> Optional[list[tuple[list[str], str]]]:
    text = _read_text(Path("/proc/self/cgroup"))
    if text is None:
        return None
    records = []
    for line in text.splitlines():
        hierarchy, separator, remainder = line.partition(":")
        if not separator or not hierarchy.isdigit():
            return None
        controllers, separator, relative_path = remainder.partition(":")
        if separator:
            records.append((controllers.split(",") if controllers else [], relative_path))
        else:
            return None
    return records


def discover_cgroups() -> tuple:
    """Read membership and mounts once for one observation, never cache them."""

    if platform.system() != "Linux":
        return None, None
    return _cgroup_records(), _mount_entries()


def _path_under_mount(mount: Mapping[str, str], cgroup_path: str) -> Optional[Path]:
    root = mount["root"].rstrip("/") or "/"
    if (
        not Path(root).is_absolute()
        or not Path(mount["mount_point"]).is_absolute()
        or not Path(cgroup_path).is_absolute()
        or ".." in Path(cgroup_path).parts
        or ".." in Path(root).parts
    ):
        return None
    relative = cgroup_path
    if root != "/":
        if relative == root:
            relative = ""
        elif relative.startswith(root + "/"):
            relative = relative[len(root) :]
        else:
            return None
    return Path(mount["mount_point"]) / relative.lstrip("/")


def _cgroup_location(controller: Optional[str], discovery=None) -> tuple[Optional[Path], Optional[Path], str]:
    """Find a covering mount; distinguish absent controllers from unknown limits."""

    records, mounts = discover_cgroups() if discovery is None else discovery
    if records is None or mounts is None:
        return None, None, "unavailable"
    if controller is None:
        path = next((value for controllers, value in records if not controllers), None)
        applicable = [value for value in mounts if value["fs_type"] == "cgroup2"]
        version = "v2"
    else:
        path = next((value for controllers, value in records if controller in controllers), None)
        applicable = [
            value
            for value in mounts
            if value["fs_type"] == "cgroup" and controller in set(value["super_options"].split(","))
        ]
        version = "v1"
    if path is None:
        return None, None, "none"
    # Prefer the broadest visible root so its ancestor limits can be checked.
    for mount in sorted(applicable, key=lambda value: len(Path(value["root"]).parts)):
        directory = _path_under_mount(mount, path)
        if directory is not None:
            return Path(mount["mount_point"]), directory, version
    return None, None, "unavailable"


def _controller_location(controller: str, discovery) -> tuple[Optional[Path], Optional[Path], str]:
    """A controller assigned to v1 must not be inferred from the v2 mount."""

    location = _cgroup_location(controller, discovery)
    return _cgroup_location(None, discovery) if location[2] == "none" else location


def _ancestors(directory: Path, mount_point: Path) -> list[Path]:
    if directory != mount_point and mount_point not in directory.parents:
        return []
    result = []
    current = directory
    while True:
        result.append(current)
        if current == mount_point:
            return result
        if current.parent == current:
            return []
        current = current.parent


def _parse_cpu_list(value: Optional[str]) -> Optional[int]:
    if not value:
        return None
    intervals = []
    for token in value.split(","):
        match = re.fullmatch(r"\s*([0-9]+)(?:-([0-9]+))?\s*", token)
        if not match:
            return None
        start = int(match.group(1))
        end = int(match.group(2)) if match.group(2) else start
        if end < start:
            return None
        intervals.append((start, end))
    intervals.sort()
    merged: list[list[int]] = []
    for start, end in intervals:
        if not merged or start > merged[-1][1] + 1:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    count = sum(end - start + 1 for start, end in merged)
    return count if 0 < count <= 2**32 - 1 else None


def _cpu_models(affinity: Optional[set[int]]) -> dict[int, str]:
    """Read model descriptions once, omitting missing or ambiguous identities."""

    if not affinity:
        return {}
    text = _read_text(Path("/proc/cpuinfo"))
    if not text:
        return {}
    models_by_cpu: dict[int, set[str]] = {}
    for block in re.split(r"\n\s*\n", text):
        processor_id = None
        model = None
        for line in block.splitlines():
            key, separator, raw_value = line.partition(":")
            if not separator:
                continue
            key = key.strip().lower()
            raw_value = raw_value.strip()
            if key == "processor" and raw_value.isdigit():
                processor_id = int(raw_value)
            elif key in ("model name", "cpu model"):
                model = _normalize_model(raw_value)
        if processor_id is not None and model:
            models_by_cpu.setdefault(processor_id, set()).add(model)
    selected = {}
    for cpu_id in affinity:
        models = models_by_cpu.get(cpu_id)
        if not models or len(models) != 1:
            return {}
        selected[cpu_id] = next(iter(models))
    return selected


def _cpu_identity(affinity: Optional[set[int]], models: dict[int, str]) -> dict[str, str]:
    result = {}
    architecture = _normalize_architecture(platform.machine())
    if architecture:
        result["architecture"] = architecture
    selected = set(models.values())
    if len(selected) == 1:
        result["model"] = next(iter(selected))
    return result


def _heterogeneous_cpu_groups(affinity: Optional[set[int]], models: dict[int, str]) -> Optional[list[dict[str, Any]]]:
    """Group affinity-visible logical CPUs only when every model is known."""

    if not affinity or not models:
        return None
    architecture = _normalize_architecture(platform.machine())
    counts: dict[str, int] = {}
    for cpu_id in affinity:
        model = models[cpu_id]
        counts[model] = counts.get(model, 0) + 1
    return [
        {
            "units": str(counts[model]),
            "model": model,
            **({"architecture": architecture} if architecture else {}),
        }
        for model in sorted(counts)
    ]


def _controller_paths(controller: str, discovery, issues: Optional[set[str]]) -> tuple[list[Path], str]:
    mount, directory, version = _controller_location(controller, discovery)
    if version == "unavailable":
        _incomplete(issues)
    return (_ancestors(directory, mount) if directory is not None and mount is not None else []), version


def _cpuset_bound(path: Path, version: str, issues: Optional[set[str]]) -> Optional[Decimal]:
    filename = "cpuset.cpus.effective" if version == "v2" else "cpuset.effective_cpus"
    value = _limit_text(path / filename, issues)
    if value is None and version == "v1":
        value = _limit_text(path / "cpuset.cpus", issues, required=True)
    if value is None:
        return None
    count = _parse_cpu_list(value)
    if count is None:
        _incomplete(issues)
        return None
    return Decimal(count)


def _quota_bound(path: Path, version: str, issues: Optional[set[str]]) -> Optional[Decimal]:
    if version == "v2":
        value = _limit_text(path / "cpu.max", issues)
        if value is None:
            return None
        tokens = value.split()
        if len(tokens) != 2:
            _incomplete(issues)
            return None
        quota_text, period_text = tokens
    else:
        quota_text = _limit_text(path / "cpu.cfs_quota_us", issues, required=True)
        if quota_text is None or quota_text == "-1":
            return None
        period_text = _limit_text(path / "cpu.cfs_period_us", issues, required=True)
        if period_text is None:
            return None
    try:
        period = int(period_text)
        if period <= 0:
            raise ValueError("CPU quota period must be positive")
        if quota_text == "max" and version == "v2":
            return None
        return Decimal(normalize_quota_units(int(quota_text), period))
    except ValueError:
        _incomplete(issues)
        return None


def probe_cpu(_discovery=None, *, issues: Optional[set[str]] = None) -> Optional[dict[str, Any]]:
    """Return the smallest usable CPU bound, retaining a best-effort fallback."""

    if platform.system() != "Linux":
        return None
    discovery = discover_cgroups() if _discovery is None else _discovery
    try:
        affinity = set(os.sched_getaffinity(0))
        affinity_count = len(affinity) or None
    except (AttributeError, OSError, TypeError, ValueError):
        affinity = None
        affinity_count = None

    candidates: list[Decimal] = []
    if affinity_count:
        candidates.append(Decimal(affinity_count))
    else:
        _incomplete(issues)
        online = _positive_sysconf("SC_NPROCESSORS_ONLN")
        if online is None:
            try:
                online = os.cpu_count()
            except (OSError, TypeError, ValueError):
                online = None
        if type(online) is int and online > 0:
            candidates.append(Decimal(online))

    # Affinity already includes effective CPU-set restrictions. Only inspect
    # those files when affinity is unavailable; quota remains an independent
    # CPU-time bound even when the allowed logical CPUs are known.
    bounds = [("cpu", _quota_bound)]
    if affinity_count is None:
        bounds.insert(0, ("cpuset", _cpuset_bound))
    for controller, read_bound in bounds:
        paths, version = _controller_paths(controller, discovery, issues)
        for path in paths:
            bound = read_bound(path, version, issues)
            if bound is not None:
                candidates.append(bound)
    if not candidates:
        return None

    selected = min(candidates)
    group: dict[str, Any] = {"units": _decimal_text(selected)}
    try:
        models = _cpu_models(affinity)
        if affinity_count is not None and selected == Decimal(affinity_count):
            groups = _heterogeneous_cpu_groups(affinity, models)
            if groups:
                return {"groups": groups}
        group.update(_cpu_identity(affinity, models))
    except Exception:
        # Optional hardware descriptions must never erase the capacity.
        pass
    return {"groups": [group]}


def probe_memory(_discovery=None, *, issues: Optional[set[str]] = None) -> Optional[dict[str, str]]:
    """Return the smallest usable physical/cgroup memory bound."""

    if platform.system() != "Linux":
        return None
    discovery = discover_cgroups() if _discovery is None else _discovery
    candidates = []
    page_size = _positive_sysconf("SC_PAGE_SIZE")
    page_count = _positive_sysconf("SC_PHYS_PAGES")
    if page_size is not None and page_count is not None:
        candidates.append(page_size * page_count)
    else:
        _incomplete(issues)

    paths, version = _controller_paths("memory", discovery, issues)
    for path in paths:
        value = _limit_text(
            path / ("memory.max" if version == "v2" else "memory.limit_in_bytes"), issues, required=version == "v1"
        )
        if value is None or (version == "v2" and value == "max"):
            continue
        try:
            limit = int(value)
            if limit <= 0:
                raise ValueError("Memory limit must be positive")
        except ValueError:
            _incomplete(issues)
            continue
        if version == "v1":
            if _is_v1_memory_unlimited(limit, page_size or 0):
                continue
            if limit >= _V1_MEMORY_UNLIMITED_MIN:
                _incomplete(issues)
                continue
        candidates.append(limit)
    return {"bytes": str(min(candidates))} if candidates else None
