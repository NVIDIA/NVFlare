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

"""Public, static Agent Inspector V3 facade."""

from __future__ import annotations

import os
from pathlib import Path

from nvflare.tool.agent.inspection.data import inspect_data_target
from nvflare.tool.agent.inspection.existing_job import existing_job_state
from nvflare.tool.agent.inspection.files import scan_sources
from nvflare.tool.agent.inspection.project import LocalImportGraph, integration
from nvflare.tool.agent.inspection.result import source_result
from nvflare.tool.agent.inspection.source import ownership

DEFAULT_MAX_FILES = 250
DEFAULT_MAX_FILE_BYTES = 512 * 1024


def inspect_source(
    path: Path | str,
    *,
    redact: bool = True,
    max_files: int = DEFAULT_MAX_FILES,
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
) -> dict:
    """Inspect local Python source without importing or executing it."""
    target = _target(path)
    _validate_limits(max_files, max_file_bytes)
    if not (target.is_symlink() or target.is_file() or target.is_dir()):
        raise ValueError("source inspection path must be a file, directory, or symlink")
    scan = scan_sources(target, redact=redact, max_files=max_files, max_file_bytes=max_file_bytes)
    owner = ownership(scan)
    graph = LocalImportGraph(scan)
    converted = integration(scan, owner, graph)
    job_state = existing_job_state(scan, owner, graph)
    return source_result(
        scan,
        owner,
        converted,
        job_state,
        max_files=max_files,
        max_file_bytes=max_file_bytes,
    )


def inspect_data(
    path: Path | str,
    *,
    redact: bool = True,
    max_files: int = DEFAULT_MAX_FILES,
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
) -> dict:
    """Inspect a local data directory using bounded metadata reads."""
    del redact  # Data inspection never serializes values or discovered absolute paths.
    target = _target(path)
    _validate_limits(max_files, max_file_bytes)
    if target.is_symlink() or not target.is_dir():
        raise ValueError("data inspection path must be an existing non-symlink directory")
    return inspect_data_target(target, max_files=max_files, max_file_bytes=max_file_bytes)


def _target(path: Path | str) -> Path:
    target = Path(os.path.abspath(os.path.normpath(str(Path(path).expanduser()))))
    if not target.exists() and not target.is_symlink():
        raise FileNotFoundError(f"inspect path does not exist: {path}")
    return target


def _validate_limits(max_files: int, max_file_bytes: int) -> None:
    if max_files <= 0 or max_file_bytes <= 0:
        raise ValueError("inspection limits must be positive integers")
