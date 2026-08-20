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

"""Pure routing and final V3 serialization."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from nvflare.tool.agent.inspection.types import SourceScan

EVIDENCE_LIMIT = 12
FRAMEWORK_SKILLS = {
    "huggingface": "nvflare-convert-huggingface",
    "lightning": "nvflare-convert-lightning",
    "pytorch": "nvflare-convert-pytorch",
}


def source_result(
    scan: SourceScan,
    ownership: dict,
    integration: dict,
    job_state: str,
    *,
    max_files: int,
    max_file_bytes: int,
) -> dict:
    selected_ownership = deepcopy(ownership)
    selected_integration = deepcopy(integration)
    selected_ownership["evidence"] = _evidence(selected_ownership["evidence"])
    selected_integration["evidence"] = _evidence(selected_integration["evidence"])
    skill, reason = route_source(job_state, integration, ownership)
    result = _common(
        "source",
        scan.target,
        max_files=max_files,
        max_file_bytes=max_file_bytes,
        complete=scan.complete,
        entries_visited=scan.entries_visited,
        files_considered=scan.files_considered,
        files_read=scan.files_read,
        findings=scan.findings,
        skill=skill,
        reason=reason,
    )
    result["ownership"] = selected_ownership
    result["integration"] = selected_integration
    return result


def data_result(
    target: Path,
    dataset: dict | None,
    audit: dict,
    *,
    max_files: int,
    max_file_bytes: int,
) -> dict:
    complete = not audit["truncated"]
    findings = [] if complete else [{"code": "DATA_SCAN_INCOMPLETE", "file": ".", "line": 0}]
    skill, reason = route_data(dataset)
    result = _common(
        "data",
        target,
        max_files=max_files,
        max_file_bytes=max_file_bytes,
        complete=complete,
        entries_visited=audit["entries_visited"],
        files_considered=audit["files_considered"],
        files_read=dataset.get("scan", {}).get("files_read", 0) if dataset else 0,
        findings=findings,
        skill=skill,
        reason=reason,
    )
    result["dataset"] = dataset
    return result


def route_source(job_state: str, integration: dict, ownership: dict) -> tuple[str | None, str]:
    if job_state == "credible":
        return None, "existing_job"
    if job_state == "possible":
        return "nvflare-orient", "possible_existing_job"
    if integration["state"] == "converted":
        return None, "already_integrated"
    if integration["state"] == "possible":
        return "nvflare-orient", "possible_integration"
    if ownership["state"] == "conflicting":
        return "nvflare-orient", "conflicting_owner"
    if ownership["state"] == "unresolved":
        return "nvflare-orient", "unresolved_owner"
    if ownership["state"] == "clear":
        return FRAMEWORK_SKILLS[ownership["framework"]], "clear_owner"
    return None, "no_route"


def route_data(dataset: dict | None) -> tuple[str | None, str]:
    if dataset and dataset.get("modality") in {"image", "tabular"}:
        return "nvflare-fed-stats", "dataset"
    if dataset:
        return "nvflare-orient", "ambiguous_dataset"
    return None, "no_route"


def _common(
    capability: str,
    target: Path,
    *,
    max_files: int,
    max_file_bytes: int,
    complete: bool,
    entries_visited: int,
    files_considered: int,
    files_read: int,
    findings: list[dict],
    skill: str | None,
    reason: str,
) -> dict:
    return {
        "schema_version": "3",
        "capability": capability,
        "path": str(target),
        "static_only": True,
        "limits": {"max_files": max_files, "max_file_bytes": max_file_bytes},
        "scan": {
            "complete": complete,
            "entries_visited": entries_visited,
            "files_considered": files_considered,
            "files_read": files_read,
            "findings": sorted(findings, key=_finding_key)[:EVIDENCE_LIMIT],
        },
        "routing": {"recommended_skill": skill, "reason": reason},
    }


def _evidence(items: list[dict]) -> list[dict]:
    return sorted(
        items,
        key=lambda item: (
            item["file"],
            item["line"],
            item["kind"],
            item.get("framework", ""),
        ),
    )[:EVIDENCE_LIMIT]


def _finding_key(item: dict) -> tuple:
    return item.get("file", ""), item.get("line", 0), item.get("code", "")
