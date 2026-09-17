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

"""Internal records for one bounded inspection pass."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

Dependency = tuple[str, tuple[str, ...]]
FactRecord = tuple[str, int, tuple[Dependency, ...]]


@dataclass
class FileFacts:
    path: str
    complete: bool = True
    is_job_py: bool = False
    owners: list[FactRecord] = field(default_factory=list)
    supporting: list[FactRecord] = field(default_factory=list)
    unresolved: list[FactRecord] = field(default_factory=list)
    local_imports: list[tuple[str | None, int, tuple[str, ...]]] = field(default_factory=list)
    nvflare_import: bool = False
    client_calls: list[tuple[str, int]] = field(default_factory=list)
    possible_client_calls: list[int] = field(default_factory=list)


@dataclass
class SourceScan:
    target: Path
    facts: dict[str, FileFacts] = field(default_factory=dict)
    complete: bool = True
    entries_visited: int = 0
    files_considered: int = 0
    files_read: int = 0
    findings: list[dict] = field(default_factory=list)
    files_seen: set[str] = field(default_factory=set)
