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

"""Checked read/write of the private, job-process-owned terminal handoff.

The child writes capacity, CPU work, file sizes, and traffic observations.
The parent validates each measurement while reading the handoff. Identity
comes from parent-owned lifecycle state, never from the child file.
Directory-relative operations protect against path redirection on supported
platforms. Other platforms reject pre-existing directory symlinks but do not
claim race-proof traversal.
"""

from __future__ import annotations

import json
import math
import os
import secrets
import stat
import tempfile
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any, Optional

from .measurements import canonical_json_bytes, ordered_issues

# The private handoff carries startup capacity and accumulated execution time.
INTERNAL_HANDOFF_VERSION = "4"
INTERNAL_HANDOFF_KIND = "nvflare.resource_stats.internal.terminal_handoff"
MAX_TERMINAL_HANDOFF_BYTES = 1024 * 1024

RESOURCE_STATS_DIR = "resource_stats"
STAGING_DIR = "staging"
TERMINAL_HANDOFF_FILE = "terminal_handoff.json"
LAUNCH_GENERATION_OPTION = "resource_launch_generation"
_DIRECTORY_OPEN_FLAGS = (
    os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
)
_SECURE_HANDOFF_DIR_FD = (
    hasattr(os, "O_NOFOLLOW")
    and bool(getattr(os, "O_DIRECTORY", 0))
    and os.open in os.supports_dir_fd
    and os.unlink in os.supports_dir_fd
    and os.rmdir in os.supports_dir_fd
)
_SECURE_HANDOFF_WRITE_DIR_FD = (
    _SECURE_HANDOFF_DIR_FD and os.mkdir in os.supports_dir_fd and os.rename in os.supports_dir_fd
)


class InvalidTerminalHandoff(ValueError):
    """Raised when the child-to-parent handoff is malformed or oversized."""


def launch_generation_from_options(options: list[str], *, option_name: str = LAUNCH_GENERATION_OPTION) -> str:
    """Capture the parent-owned token before generic option parsing changes its type."""

    prefix = f"{option_name}="
    for option in reversed(options):
        if option.startswith(prefix):
            return option[len(prefix) :]
    return ""


def _handoff_filename(generation: str | None) -> str:
    if generation is None:
        return TERMINAL_HANDOFF_FILE  # Unbound diagnostic helpers only.
    if (
        not isinstance(generation, str)
        or not 1 <= len(generation) <= 128
        or any(not (char.isascii() and (char.isalnum() or char in "_-")) for char in generation)
    ):
        raise InvalidTerminalHandoff("invalid launch generation")
    return f"terminal_handoff.{generation}.json"


def terminal_handoff_path(run_dir: str | Path, *, launch_generation: str | None = None) -> Path:
    return Path(run_dir) / RESOURCE_STATS_DIR / STAGING_DIR / _handoff_filename(launch_generation)


def _open_handoff_directory(
    run_dir: str | Path, *, include_staging: bool, create: bool = False
) -> tuple[int | None, Path]:
    """Open every parent component without following an untrusted symlink."""

    root = Path(run_dir)
    components = (RESOURCE_STATS_DIR, STAGING_DIR) if include_staging else (RESOURCE_STATS_DIR,)
    target = root.joinpath(*components)
    if create:
        root.mkdir(parents=True, exist_ok=True)
    if not _SECURE_HANDOFF_DIR_FD or (create and not _SECURE_HANDOFF_WRITE_DIR_FD):
        current = root
        for component in (None, *components):
            if component is not None:
                current /= component
                if create:
                    current.mkdir(exist_ok=True)
            mode = current.lstat().st_mode
            if not stat.S_ISDIR(mode) or stat.S_ISLNK(mode):
                raise OSError(f"resource handoff path is not a plain directory: {current}")
        return None, target

    opened = []
    try:
        current_fd = os.open(root, _DIRECTORY_OPEN_FLAGS)
        opened.append(current_fd)
        for component in components:
            if create:
                try:
                    os.mkdir(component, mode=0o700, dir_fd=current_fd)
                except FileExistsError:
                    pass
            next_fd = os.open(component, _DIRECTORY_OPEN_FLAGS, dir_fd=current_fd)
            opened.append(next_fd)
            current_fd = next_fd
        result_fd = opened.pop()
        return result_fd, target
    finally:
        for descriptor in reversed(opened):
            os.close(descriptor)


def _read_fields(value, required, optional=()):
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ValueError("measurement must be an object with string keys")
    required, optional = set(required), set(optional)
    if not required <= value.keys() or value.keys() - required - optional:
        raise ValueError("measurement has missing or unexpected fields")
    return value


def _read_number(value, *, integer=False, positive=False):
    if isinstance(value, bool) or not isinstance(value, int if integer else (int, float)):
        raise ValueError("measurement must be an integer" if integer else "measurement must be a number")
    if value < 0 or (positive and value == 0) or (isinstance(value, float) and not math.isfinite(value)):
        raise ValueError(
            "measurement must be finite and positive" if positive else "measurement must be finite and non-negative"
        )
    return value


def _read_status(value, allowed=("partial", "unavailable")):
    if not isinstance(value, Mapping):
        raise ValueError("measurement must be an object")
    status = value.get("status", "reported")
    if "status" in value and status not in allowed:
        raise ValueError("measurement has an invalid status")
    return status


def _read_capacity(value, *, elapsed=False):
    status = _read_status(value)
    dimensions = {"cpu", "memory", "gpu"}
    _read_fields(value, dimensions if status == "reported" else {"status"}, dimensions if status == "partial" else ())
    if status == "partial" and not dimensions & value.keys():
        raise ValueError("partial capacity needs an observed dimension")
    fields = (
        {"cpu": "unit_seconds", "memory": "byte_seconds", "gpu": "instance_seconds"}
        if elapsed
        else {"cpu": "units", "memory": "bytes", "gpu": "count"}
    )
    for dimension, field in fields.items():
        if dimension in value:
            item = _read_fields(value[dimension], {field})
            _read_number(
                item[field], integer=not elapsed and dimension != "cpu", positive=not elapsed and dimension == "cpu"
            )
    return deepcopy(value)


def _read_usage(value, fields, *, integer=True):
    status = _read_status(value)
    required = set(fields) if status != "unavailable" else set()
    if status != "reported":
        required.add("status")
    _read_fields(value, required)
    if status != "unavailable":
        for field in fields:
            _read_number(value[field], integer=integer)
    return deepcopy(value)


def _read_workspace(value):
    if _read_status(value, ("unavailable",)) == "reported":
        _read_fields(value, {"capacity_bytes"})
        _read_number(value["capacity_bytes"], integer=True, positive=True)
    else:
        _read_fields(value, {"status"})
    return deepcopy(value)


def _read_issues(issues, measurements):
    """Check incoming scope diagnostics and their relation to accepted values."""
    if not isinstance(issues, list):
        raise ValueError("scope issues must be a list")
    allowed_codes = {"observation_incomplete", "attribution_incomplete", "unsupported", "malformed_source"}
    for item in issues:
        _read_fields(item, {"metric", "code"})
        if not isinstance(item["metric"], str) or item["metric"] not in measurements:
            raise ValueError("scope issue names an unknown metric")
        if not isinstance(item["code"], str) or item["code"] not in allowed_codes:
            raise ValueError("scope issue has an unknown cause")
    result = ordered_issues(issues)
    for metric, value in measurements.items():
        codes = {item["code"] for item in result if item["metric"] == metric}
        status = value.get("status", "reported")
        if status == "reported":
            if codes:
                raise ValueError("complete measurements cannot have issues")
            continue
        allowed = {"observation_incomplete", "attribution_incomplete"} if status == "partial" else allowed_codes
        if metric == "message_traffic":
            allowed = (
                {"observation_incomplete"} if status == "partial" else {"observation_incomplete", "malformed_source"}
            )
        if not codes or codes - allowed:
            raise ValueError("measurement status needs supported causes")
    return result


def _read_handoff(value: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "internal_version",
        "kind",
        "resource_capacity",
        "resource_time",
        "resource_utilization",
        "workspace_filesystem",
        "issues",
    }
    if (
        not isinstance(value, Mapping)
        or not required <= set(value)
        or set(value) - required - {"duration_seconds", "launch_generation"}
    ):
        raise InvalidTerminalHandoff(f"terminal handoff fields must be {sorted(required)}")
    if value["internal_version"] != INTERNAL_HANDOFF_VERSION or value["kind"] != INTERNAL_HANDOFF_KIND:
        raise InvalidTerminalHandoff("terminal handoff has an unsupported version or kind")
    utilization = value["resource_utilization"]
    fields = {"cpu_consumed", "retained_content", "message_traffic"}
    if not isinstance(utilization, Mapping) or set(utilization) != fields:
        raise InvalidTerminalHandoff(
            "resource_utilization must contain CPU work, retained content, and message traffic"
        )
    measurements = {
        **{name: value[name] for name in ("resource_capacity", "resource_time", "workspace_filesystem")},
        **utilization,
    }
    by_metric = {name: [] for name in measurements}
    if not isinstance(value["issues"], list):
        raise InvalidTerminalHandoff("terminal handoff issues must be a list")
    for issue in value["issues"]:
        if (
            not isinstance(issue, Mapping)
            or not isinstance(issue.get("metric"), str)
            or issue["metric"] not in by_metric
        ):
            raise InvalidTerminalHandoff("terminal handoff issue must identify a known metric")
        by_metric[issue["metric"]].append(issue)
    issues = []
    for name, measurement in measurements.items():
        try:
            if name in {"resource_capacity", "resource_time"}:
                checked = _read_capacity(measurement, elapsed=name == "resource_time")
            elif name == "workspace_filesystem":
                checked = _read_workspace(measurement)
            elif name == "cpu_consumed":
                checked = _read_usage(measurement, ("cpu_seconds",), integer=False)
            else:
                quantity_fields = ("bytes",) if name == "retained_content" else ("sent_bytes", "received_bytes")
                checked = _read_usage(measurement, quantity_fields)
            tags = _read_issues(by_metric[name], {name: checked})
        except ValueError:
            checked = {"status": "unavailable"}
            tags = [{"metric": name, "code": "malformed_source"}]
        measurements[name] = checked
        issues.extend(tags)
    result = {
        "internal_version": INTERNAL_HANDOFF_VERSION,
        "kind": INTERNAL_HANDOFF_KIND,
        **{name: measurements[name] for name in ("resource_capacity", "resource_time", "workspace_filesystem")},
        "resource_utilization": {name: measurements[name] for name in sorted(fields)},
        "issues": ordered_issues(issues),
    }
    if "launch_generation" in value:
        _handoff_filename(value["launch_generation"])
        if value["launch_generation"] is None:
            raise InvalidTerminalHandoff("invalid launch generation")
        result["launch_generation"] = value["launch_generation"]
    if "duration_seconds" in value:
        try:
            result["duration_seconds"] = _read_number(value["duration_seconds"])
        except ValueError:
            pass  # Preserve untimed quantities; never invent an elapsed duration.
    return result


def write_terminal_handoff(
    run_dir: str | Path, handoff: Mapping[str, Any], *, launch_generation: str | None = None
) -> Path:
    """Atomically write the child handoff under its existing job workspace."""

    filename = _handoff_filename(launch_generation)
    if launch_generation is not None:
        if "launch_generation" in handoff and handoff["launch_generation"] != launch_generation:
            raise InvalidTerminalHandoff("conflicting launch generation")
        handoff = {**handoff, "launch_generation": launch_generation}
    data = canonical_json_bytes(handoff)
    if len(data) > MAX_TERMINAL_HANDOFF_BYTES:
        raise InvalidTerminalHandoff("terminal handoff exceeds its size limit")
    path = terminal_handoff_path(run_dir, launch_generation=launch_generation)
    directory_fd, directory_path = _open_handoff_directory(run_dir, include_staging=True, create=True)
    temp_path = None
    try:
        if directory_fd is None:
            # On platforms without directory-relative operations, reject
            # existing directory symlinks above, then retain best-effort atomic
            # publication. This fallback does not claim protection from races.
            stream = tempfile.NamedTemporaryFile(dir=directory_path, prefix=".terminal-", suffix=".tmp", delete=False)
            temp_path = Path(stream.name)
        else:
            for _ in range(8):
                temp_name = f".terminal-{secrets.token_hex(8)}.tmp"
                flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
                try:
                    fd = os.open(temp_name, flags, mode=0o600, dir_fd=directory_fd)
                except FileExistsError:
                    continue
                temp_path = temp_name
                try:
                    stream = os.fdopen(fd, "wb")
                except BaseException:
                    os.close(fd)
                    raise
                break
            else:
                raise OSError("could not create a private resource handoff temporary file")
        with stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        if directory_fd is None:
            os.replace(temp_path, path)
        else:
            os.replace(temp_path, filename, src_dir_fd=directory_fd, dst_dir_fd=directory_fd)
        return path
    finally:
        try:
            if temp_path is not None:
                try:
                    if directory_fd is None:
                        temp_path.unlink()
                    else:
                        os.unlink(temp_path, dir_fd=directory_fd)
                except FileNotFoundError:
                    pass
        finally:
            if directory_fd is not None:
                os.close(directory_fd)


def read_terminal_handoff(run_dir: str | Path, *, expected_generation: str | None = None) -> Optional[dict[str, Any]]:
    """Read only this launch's handoff; return ``None`` if absent or malformed."""

    try:
        filename = _handoff_filename(expected_generation)
    except InvalidTerminalHandoff:
        return None

    directory_fd = None
    fd = None
    try:
        directory_fd, directory_path = _open_handoff_directory(run_dir, include_staging=True)
        # The job process owns this self-report, so treat the path as untrusted:
        # do not follow any path-component symlink and do not let a FIFO/device
        # block the parent.
        flags = os.O_RDONLY
        flags |= getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NONBLOCK", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        if directory_fd is None:
            path = directory_path / filename
            if not hasattr(os, "O_NOFOLLOW") and path.is_symlink():
                return None
            fd = os.open(path, flags)
        else:
            fd = os.open(filename, flags, dir_fd=directory_fd)
        file_stat = os.fstat(fd)
        if not stat.S_ISREG(file_stat.st_mode) or file_stat.st_size > MAX_TERMINAL_HANDOFF_BYTES:
            return None
        with os.fdopen(fd, "rb") as stream:
            fd = None
            data = stream.read(MAX_TERMINAL_HANDOFF_BYTES + 1)
    except OSError:
        return None
    finally:
        if fd is not None:
            os.close(fd)
        if directory_fd is not None:
            os.close(directory_fd)
    if len(data) > MAX_TERMINAL_HANDOFF_BYTES:
        return None

    def reject_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise InvalidTerminalHandoff(f"duplicate terminal handoff key: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda value: (_ for _ in ()).throw(InvalidTerminalHandoff(value)),
        )
        checked = _read_handoff(value)
        if expected_generation is not None and checked.get("launch_generation") != expected_generation:
            return None
        return checked
    except (ValueError, RecursionError, TypeError):
        return None


def remove_terminal_handoff(run_dir: str | Path, *, expected_generation: str | None = None) -> None:
    """Remove only the owned generation; never consume a replacement's report."""

    try:
        filename = _handoff_filename(expected_generation)
    except InvalidTerminalHandoff:
        return

    directory_fd = None
    try:
        directory_fd, directory_path = _open_handoff_directory(run_dir, include_staging=True)
        if directory_fd is None:
            (directory_path / filename).unlink()
        else:
            os.unlink(filename, dir_fd=directory_fd)
    except FileNotFoundError:
        pass
    except OSError:
        # Unsafe job-owned paths are left untouched.  The server's
        # final inventory check will reject any remaining staging entry.
        return
    finally:
        if directory_fd is not None:
            os.close(directory_fd)

    stats_fd = None
    try:
        stats_fd, stats_path = _open_handoff_directory(run_dir, include_staging=False)
        if stats_fd is None:
            (stats_path / STAGING_DIR).rmdir()
        else:
            os.rmdir(STAGING_DIR, dir_fd=stats_fd)
    except OSError:
        pass
    finally:
        if stats_fd is not None:
            os.close(stats_fd)
