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

"""Private conversion guard for existing source and exported jobs."""

from __future__ import annotations

from pathlib import Path

from nvflare.tool.agent.inspection.project import LocalImportGraph, is_secondary, owner_scopes
from nvflare.tool.agent.inspection.types import SourceScan

CONFIG_NAMES = {"config_fed_client.json", "config_fed_server.json"}


def existing_job_state(scan: SourceScan, ownership: dict, graph: LocalImportGraph) -> str:
    marker_state = _exported_marker_state(scan)
    if marker_state == "credible":
        return marker_state

    candidates = {
        path
        for path, facts in scan.facts.items()
        if facts.is_job_py
        and any(
            scan.facts[reached].nvflare_import and reached not in graph.local_nvflare_shadow_files
            for reached in graph.closure(path)
        )
    }
    if scan.target.is_file():
        source_state = "credible" if "." in candidates else "none"
    else:
        source_state = _directory_source_state(scan, ownership, graph, candidates)
    if source_state == "credible":
        return source_state
    if marker_state == "possible" or source_state == "possible" or not scan.complete:
        return "possible"
    return "none"


def _directory_source_state(scan: SourceScan, ownership: dict, graph: LocalImportGraph, candidates: set[str]) -> str:
    if not candidates:
        return "none"
    scopes = owner_scopes(scan, ownership, graph)
    if scopes:
        owner_files = {
            path
            for path, facts in scan.facts.items()
            if any(framework == ownership["framework"] for framework, _, _ in facts.owners)
        }
        authoritative = {candidate for candidate in candidates if any(candidate in scope for scope in scopes)}
        for path in scan.facts:
            if _is_root(path) and not is_secondary(path) and graph.closure(path) & owner_files:
                authoritative.update(candidates & graph.closure(path))
        roots = {candidate for candidate in authoritative if _is_root(candidate)}
        if len(roots) == 1:
            return "credible"
        if len(roots) > 1 or len(authoritative) > 1:
            return "possible"
        if authoritative:
            return "credible"
        # Independent non-secondary jobs are real evidence but cannot override the selected owner.
        # Secondary fixtures remain ignored so they do not suppress an active conversion target.
        has_active_candidate = any(not is_secondary(candidate) for candidate in candidates)
        secondary_only_owner = bool(owner_files) and all(is_secondary(path) for path in owner_files)
        return "possible" if has_active_candidate or secondary_only_owner else "none"

    roots = {candidate for candidate in candidates if _is_root(candidate) and not is_secondary(candidate)}
    if len(roots) == 1:
        return "credible"
    if len(roots) > 1:
        return "possible"
    active = {candidate for candidate in candidates if not is_secondary(candidate)}
    if len(active) == 1:
        return "credible"
    launchers = {
        path: candidates & graph.closure(path)
        for path in scan.facts
        if _is_root(path) and not is_secondary(path) and candidates & graph.closure(path)
    }
    independent = {
        path: reached
        for path, reached in launchers.items()
        if not any(path in graph.closure(other) for other in launchers if other != path)
    }
    if len(independent) == 1:
        return "credible" if len(next(iter(independent.values()))) == 1 else "possible"
    if len(independent) > 1:
        return "possible"
    # Candidates exist, but no active owner or launcher can distinguish a secondary-only project
    # from a fixture. Preserve that uncertainty instead of reporting that no job evidence exists.
    return "possible"


def _exported_marker_state(scan: SourceScan) -> str:
    if not scan.target.is_dir():
        return "none"
    has_meta = "meta.json" in scan.files_seen
    has_config = any(_is_exact_config(path) for path in scan.files_seen)
    if has_meta and has_config:
        return "credible"
    return "possible" if has_meta or has_config else "none"


def _is_exact_config(path: str) -> bool:
    parts = Path(path).parts
    if Path(path).name not in CONFIG_NAMES:
        return False
    return len(parts) == 1 or (len(parts) == 3 and parts[1] == "config")


def _is_root(path: str) -> bool:
    return path != "." and Path(path).parent == Path(".")
