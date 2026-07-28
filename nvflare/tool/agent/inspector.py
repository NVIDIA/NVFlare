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

"""Static read-only inspection for agent workflows."""

from pathlib import Path

import nvflare
from nvflare.tool.agent import dataset_inspect, frameworks
from nvflare.tool.agent.inspection.models import MAX_EVIDENCE_PER_BUCKET, InspectionFacts
from nvflare.tool.agent.inspection.project import (
    authoritative_source_job_candidate,
    build_local_import_graph,
    build_project_scope,
)
from nvflare.tool.agent.inspection.routing import FamilyResolver as _FamilyResolver
from nvflare.tool.agent.inspection.routing import (
    conversion_state,
    detect_primary_framework,
    order_frameworks_for_display,
    rank_frameworks,
)
from nvflare.tool.agent.inspection.routing import routing_decision as _routing_decision
from nvflare.tool.agent.inspection.routing import target_type as classify_target_type
from nvflare.tool.agent.inspection.scanner import DEFAULT_MAX_FILE_BYTES, DEFAULT_MAX_FILES, _display_path, scan_path
from nvflare.tool.agent.inspection.skill_discovery import discover_installed_skills


def inspect_path(
    path: Path | str,
    *,
    redact: bool = True,
    max_files: int = DEFAULT_MAX_FILES,
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
) -> dict:
    """Inspect a path without importing or executing user code."""
    facts = scan_path(path, redact=redact, max_files=max_files, max_file_bytes=max_file_bytes)
    target = facts.root
    import_graph = build_local_import_graph(facts)
    exported_job_info = _exported_job_info(facts)
    project_scope = build_project_scope(facts, import_graph)
    family_resolver = _FamilyResolver(facts, import_graph)
    ranked_frameworks = rank_frameworks(facts)
    detected_framework = detect_primary_framework(facts, ranked_frameworks, import_graph, family_resolver)
    ranked_frameworks = order_frameworks_for_display(ranked_frameworks, detected_framework)
    source_job = authoritative_source_job_candidate(facts, project_scope, import_graph)
    current_conversion_state = conversion_state(
        facts,
        detected_framework,
        exported_job_info,
        project_scope,
        source_job,
    )
    target_type = classify_target_type(target, facts, detected_framework, current_conversion_state)

    dataset = None
    utility_only_code = target_type == "training_repository" and detected_framework in frameworks.UTILITY_FRAMEWORKS
    if target.is_dir() and (target_type == "unknown_target" or utility_only_code):
        dataset = dataset_inspect.inspect_dataset(target, max_files=max_files, max_file_bytes=max_file_bytes)
        if dataset and dataset["modality"] in ("tabular", "image"):
            target_type = f"{dataset['modality']}_dataset"

    family_member_conflict = frameworks.has_active_family_member_conflict(
        facts.framework_evidence,
        family_resolver,
    )
    routing = _routing_decision(
        detected_framework,
        current_conversion_state,
        source_job,
        facts,
        dataset,
        family_member_conflict,
    )

    return {
        "schema_version": "1",
        "nvflare_version": nvflare.__version__,
        "path": str(target),
        "target_type": target_type,
        "static_only": True,
        "redaction": "on" if redact else "off",
        "limits": {
            "max_files": max_files,
            "max_file_bytes": max_file_bytes,
            "max_evidence_per_bucket": MAX_EVIDENCE_PER_BUCKET,
        },
        "classification_incomplete": facts.classification_incomplete,
        "framework_ownership": frameworks.ownership_summary(
            detected_framework,
            facts.framework_evidence,
            family_resolver,
            family_member_conflict=family_member_conflict,
        ),
        "scan": {
            "entries_visited": facts.entries_visited,
            "files_considered": facts.files_considered,
            "files_scanned": facts.files_scanned,
            "bytes_scanned": facts.bytes_scanned,
            "files_skipped_count": facts.files_skipped_count,
            "files_skipped_count_approximate": facts.file_limit_skip_accounting_truncated,
            "files_skipped_truncated": facts.file_limit_reached,
            "files_skipped_evidence_truncated": facts.files_skipped_count > len(facts.files_skipped),
            "files_skipped": list(facts.files_skipped),
        },
        "frameworks": ranked_frameworks,
        "entry_points": list(facts.entry_points[:MAX_EVIDENCE_PER_BUCKET]),
        "flare_integration": {
            "present": bool(facts.flare_imports or facts.flare_calls),
            "imports": list(facts.flare_imports[:MAX_EVIDENCE_PER_BUCKET]),
            "calls": sorted(facts.flare_calls),
        },
        "conversion_state": current_conversion_state,
        "job": {
            "job_py": facts.job_py,
            "sim_env_used": facts.sim_env_used,
            "export_support": facts.export_support,
            "exported_job_markers": list(facts.exported_job_markers[:MAX_EVIDENCE_PER_BUCKET]),
            "exported_job_candidates": exported_job_info["submit_ready_candidates"][:MAX_EVIDENCE_PER_BUCKET],
            "nested_candidates": exported_job_info["nested_candidates"][:MAX_EVIDENCE_PER_BUCKET],
        },
        "patterns": {
            "distributed": list(facts.distributed_patterns[:MAX_EVIDENCE_PER_BUCKET]),
            "dynamic": list(facts.dynamic_patterns[:MAX_EVIDENCE_PER_BUCKET]),
            "absolute_data_paths": list(facts.absolute_path_findings[:MAX_EVIDENCE_PER_BUCKET]),
        },
        "findings": list(facts.findings[:MAX_EVIDENCE_PER_BUCKET]),
        "dataset": dataset,
        "skill_selection": routing.skill_selection(),
        "recommended_next_commands": list(routing.recommended_next_commands),
        "installed_skills": discover_installed_skills(target),
    }


def _exported_job_info(facts: InspectionFacts) -> dict:
    root = facts.root if facts.root.is_dir() and not facts.root.is_symlink() else facts.root.parent
    markers_by_dir: dict[Path, set[str]] = {}
    for path in facts.exported_job_marker_paths:
        markers_by_dir.setdefault(path.parent, set()).add(path.name)

    valid_candidate_dirs = set()
    consumed_marker_dirs = set()
    for directory, names in markers_by_dir.items():
        if "meta.json" in names and names.intersection({"config_fed_server.json", "config_fed_client.json"}):
            valid_candidate_dirs.add(directory)

    meta_dirs = {directory for directory, names in markers_by_dir.items() if "meta.json" in names}
    config_paths = [
        path
        for path in facts.exported_job_marker_paths
        if path.name in {"config_fed_server.json", "config_fed_client.json"}
    ]
    for meta_dir in meta_dirs:
        for config_path in config_paths:
            if config_path.parent.name == "config" and config_path.parent.parent.parent == meta_dir:
                valid_candidate_dirs.add(meta_dir)
                consumed_marker_dirs.add(config_path.parent)

    submit_ready = sorted(
        _display_path(directory, root, facts.redact) for directory in valid_candidate_dirs if directory == root
    )
    nested = []
    for directory, names in sorted(
        markers_by_dir.items(),
        key=lambda item: _display_path(item[0], root, facts.redact),
    ):
        if directory in consumed_marker_dirs:
            continue
        rel_dir = _display_path(directory, root, facts.redact)
        if directory in valid_candidate_dirs:
            if directory != root:
                nested.append(
                    {
                        "path": rel_dir,
                        "markers": sorted(names),
                        "reason": "nested_exported_job_candidate",
                    }
                )
        else:
            nested.append(
                {
                    "path": rel_dir,
                    "markers": sorted(names),
                    "reason": "incomplete_exported_job_marker_set",
                }
            )
    return {"submit_ready_candidates": submit_ready, "nested_candidates": nested}
