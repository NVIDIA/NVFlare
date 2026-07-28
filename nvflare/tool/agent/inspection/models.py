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

"""Read-only facts and decisions shared by inspector implementation modules."""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Optional, Sequence

MAX_EVIDENCE_PER_BUCKET = 12
SECRET_NAME_PATTERN = re.compile(r"(api[_-]?key|secret|token|password|passwd|credential|access[_-]?key)", re.I)


def looks_like_absolute_path(value: str) -> bool:
    return value.startswith(("/", "~")) or bool(re.match(r"^[A-Za-z]:[\\/]", value))


def redact_literal(value: str, redact: bool) -> str:
    if not redact:
        return value
    if looks_like_absolute_path(value):
        return "<REDACTED_PATH>"
    if SECRET_NAME_PATTERN.search(value):
        return "<REDACTED>"
    return value


@dataclass(frozen=True)
class InspectionFacts:
    """Structurally immutable metadata collected by the source scanner.

    Evidence records remain dictionaries for output compatibility. Consumers
    must treat those nested records as read-only.
    """

    root: Path
    redact: bool
    entries_visited: int = 0
    files_considered: int = 0
    files_scanned: int = 0
    bytes_scanned: int = 0
    files_skipped_count: int = 0
    file_limit_reached: bool = False
    file_limit_accounted_skips: int = 0
    file_limit_skip_accounting_truncated: bool = False
    classification_incomplete: bool = False
    files_skipped: Sequence[dict] = ()
    findings: Sequence[dict] = ()
    framework_evidence: Mapping[str, Sequence[dict]] = field(default_factory=dict)
    flare_imports: Sequence[dict] = ()
    flare_calls: frozenset[str] = frozenset()
    flare_calls_by_file: Mapping[str, frozenset[str]] = field(default_factory=dict)
    integration_signals: Mapping[str, frozenset[str]] = field(default_factory=dict)
    integration_signal_files: frozenset[str] = frozenset()
    file_imports: Mapping[str, frozenset[str]] = field(default_factory=dict)
    entry_points: Sequence[dict] = ()
    job_py: Optional[str] = None
    job_py_paths: Sequence[str] = ()
    sim_env_used: bool = False
    sim_env_files: frozenset[str] = frozenset()
    export_support: bool = False
    export_support_files: frozenset[str] = frozenset()
    exported_job_markers: Sequence[str] = ()
    exported_job_marker_paths: Sequence[Path] = ()
    distributed_patterns: Sequence[dict] = ()
    dynamic_patterns: Sequence[dict] = ()
    absolute_path_findings: Sequence[dict] = ()
    class_body_ranges: Mapping[str, Sequence[tuple[int, int]]] = field(default_factory=dict)


@dataclass(frozen=True)
class RoutingDecision:
    detected_framework: Optional[str]
    conversion_state: str
    recommended_skills: tuple[str, ...]
    recommended_next_commands: tuple[str, ...]
    safety_findings: tuple[str, ...]

    def skill_selection(self) -> dict:
        return {
            "detected_framework": self.detected_framework,
            "conversion_state": self.conversion_state,
            "exported_job": self.conversion_state == "exported_job",
            "recommended_skills": list(self.recommended_skills),
            "safety_findings": list(self.safety_findings),
        }
