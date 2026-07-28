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

"""Framework ownership, conversion state, and next-action routing."""

import shlex
from pathlib import Path
from typing import Optional

from nvflare.tool.agent import frameworks
from nvflare.tool.agent.inspection.models import MAX_EVIDENCE_PER_BUCKET, InspectionFacts, RoutingDecision
from nvflare.tool.agent.inspection.project import (
    LocalImportGraph,
    ProjectScope,
    SourceJobCandidate,
    authorizes_flare_evidence,
    build_local_import_graph,
    evidence_tied_to_entry_context,
)


def rank_frameworks(facts: InspectionFacts) -> list[dict]:
    total = sum(len(evidence) for evidence in facts.framework_evidence.values())
    ranked = []
    for framework, evidence in facts.framework_evidence.items():
        count = len(evidence)
        confidence = min(0.99, 0.45 + (count / total) * 0.5) if total else 0.0
        ranked.append(
            {
                "name": framework,
                "confidence": round(confidence, 2),
                "evidence": list(evidence[:MAX_EVIDENCE_PER_BUCKET]),
                "contradicting_evidence": [],
            }
        )
    return sorted(ranked, key=lambda item: (-item["confidence"], item["name"]))


def detect_primary_framework(
    facts: InspectionFacts,
    ranked: list[dict],
    import_graph: LocalImportGraph,
    family_resolver: "FamilyResolver",
) -> Optional[str]:
    if not ranked:
        return None
    primary = _primary_by_confidence_and_entry_context(facts, ranked, import_graph)
    return frameworks.resolve_primary_framework(primary, facts.framework_evidence, family_resolver)


def order_frameworks_for_display(ranked: list[dict], detected_framework: Optional[str]) -> list[dict]:
    if not detected_framework:
        return ranked
    return sorted(ranked, key=lambda item: item["name"] != detected_framework)


def conversion_state(
    facts: InspectionFacts,
    detected_framework: Optional[str],
    exported_job_info: dict,
    project_scope: ProjectScope,
    source_job: Optional[SourceJobCandidate],
) -> str:
    if exported_job_info["submit_ready_candidates"]:
        return "exported_job"
    if project_scope.ambiguous:
        return "ambiguous"
    # job.py and SimEnv are common names outside NVFLARE. A source candidate is
    # authoritative only when its project component also contains NVFLARE
    # evidence, as established by project analysis.
    if source_job:
        return "flare_job"
    if _has_conversion_integration(facts, project_scope):
        return "client_api_converted"
    if all(_has_authoritative_flare_call(facts, project_scope, name) for name in ("flare.receive", "flare.send")):
        return "client_api_converted"
    if _has_authoritative_flare_call(facts, project_scope, "FLModel"):
        return "client_api_converted"
    if _has_authoritative_flare_evidence(facts, project_scope):
        return "partial_client_api"
    if detected_framework:
        return "not_converted"
    return "unknown"


def target_type(
    path: Path,
    facts: InspectionFacts,
    detected_framework: Optional[str],
    current_conversion_state: str,
) -> str:
    if path.is_symlink():
        return "unknown_target"
    if path.is_file():
        return "single_training_script" if path.suffix == ".py" else "unknown_target"
    if current_conversion_state == "exported_job":
        return "exported_submit_ready_flare_job"
    if current_conversion_state == "flare_job":
        return "flare_job_source"
    if current_conversion_state == "ambiguous":
        return "mixed_workspace"
    if detected_framework and current_conversion_state in {"partial_client_api", "client_api_converted"}:
        return "mixed_workspace"
    # A base framework plus a specialized family member is distinct from a
    # partially converted FLARE workspace.
    if frameworks.family_base_has_member(detected_framework, facts.framework_evidence):
        return "mixed_framework_workspace"
    if detected_framework:
        return "training_repository"
    return "unknown_target"


def routing_decision(
    detected_framework: Optional[str],
    current_conversion_state: str,
    source_job: Optional[SourceJobCandidate],
    facts: InspectionFacts,
    dataset: Optional[dict] = None,
    family_member_conflict: bool = False,
) -> RoutingDecision:
    recommended = []
    if current_conversion_state == "exported_job":
        pass
    elif current_conversion_state == "flare_job":
        recommended.append("nvflare-autofl")
    elif current_conversion_state == "ambiguous":
        recommended.append("nvflare-orient")
    elif dataset and dataset.get("modality") in ("tabular", "image"):
        recommended.append("nvflare-fed-stats")
    elif dataset and dataset.get("modality") == "mixed":
        recommended.append("nvflare-orient")
    elif current_conversion_state in {"not_converted", "partial_client_api"} and family_member_conflict:
        recommended.append("nvflare-orient")
    elif detected_framework and current_conversion_state in {"not_converted", "partial_client_api"}:
        evidence = facts.framework_evidence.get(detected_framework, [])
        skill = frameworks.recommended_skill_for(detected_framework, evidence)
        if skill:
            recommended.append(skill)
        else:
            fallback = frameworks.fallback_skill_for(detected_framework, evidence)
            if fallback:
                recommended.append(fallback)
    if (
        (facts.findings or facts.classification_incomplete)
        and "nvflare-fed-stats" not in recommended
        and "nvflare-orient" not in recommended
    ):
        recommended.append("nvflare-orient")

    commands = [f"Use the {skill} skill before editing." for skill in recommended]
    if current_conversion_state == "exported_job":
        commands.append("nvflare job submit <job-folder> --format json")
    elif source_job and source_job.export_supported:
        commands.append(f"python {shlex.quote(source_job.source_file)} --export --export-dir <job-dir>")

    return RoutingDecision(
        detected_framework=detected_framework,
        conversion_state=current_conversion_state,
        recommended_skills=tuple(recommended),
        recommended_next_commands=tuple(commands),
        safety_findings=tuple(finding["code"] for finding in facts.findings[:MAX_EVIDENCE_PER_BUCKET]),
    )


class FamilyResolver:
    """Framework-neutral evidence adapter used by family-member detectors."""

    def __init__(
        self,
        facts,
        import_graph: Optional[LocalImportGraph] = None,
    ):
        self._facts = facts
        self._import_graph = import_graph or build_local_import_graph(facts)

    def evidence(self, framework: str) -> list[dict]:
        return list(self._facts.framework_evidence.get(framework, ()))

    def active_evidence(self, framework: str) -> list[dict]:
        return [item for item in self.evidence(framework) if frameworks.is_active_evidence(framework, item)]

    def training_owner_evidence(self, framework: str) -> list[dict]:
        return [item for item in self.evidence(framework) if frameworks.is_training_owner_evidence(framework, item)]

    def candidate_evidence(self, framework: str) -> list[dict]:
        return [item for item in self.evidence(framework) if frameworks.is_candidate_evidence(framework, item)]

    def score(self, evidence: list[dict]) -> int:
        return evidence_score(evidence)

    def tied_to_entry_context(self, evidence: list[dict]) -> bool:
        return evidence_tied_to_entry_context(self._facts, evidence, self._import_graph)

    def has_inspected_file_or_entry_point(self) -> bool:
        return self._facts.root.is_file() or bool(self._facts.entry_points)

    @staticmethod
    def evidence_outside_files(evidence: list[dict], reference_evidence: list[dict]) -> list[dict]:
        reference_files = {item["file"] for item in reference_evidence}
        return [item for item in evidence if item["file"] not in reference_files]

    def evidence_outside_class_bodies(self, evidence: list[dict], class_evidence: list[dict]) -> list[dict]:
        ranges_by_file: dict[str, list[tuple[int, int]]] = {}
        for item in class_evidence:
            file_path = item["file"]
            def_line = item.get("line")
            for start, end in self._facts.class_body_ranges.get(file_path, []):
                if start == def_line:
                    ranges_by_file.setdefault(file_path, []).append((start, end))
        return [
            item for item in evidence if not _line_within_ranges(item.get("line"), ranges_by_file.get(item["file"]))
        ]


def evidence_score(evidence: list[dict]) -> int:
    weights = frameworks.evidence_weights()
    return sum(weights.get(item["kind"], 1) for item in evidence)


def _primary_by_confidence_and_entry_context(
    facts: InspectionFacts,
    ranked: list[dict],
    import_graph: LocalImportGraph,
) -> str:
    # Count-based confidence is blind to reachability: incidental imports in an
    # unused helper can outrank the framework an entry point executes. Prefer,
    # in order, entry-reachable active evidence, entry-reachable import/candidate
    # evidence, then the highest-ranked non-utility framework. Family promotion
    # refines that choice afterward. Do not collapse this to pure count or pure
    # entry context: both regress covered active-owner/import-only cases.
    for item in ranked:
        name = item["name"]
        if name in frameworks.UTILITY_FRAMEWORKS:
            continue
        active = [e for e in facts.framework_evidence.get(name, []) if frameworks.is_active_evidence(name, e)]
        if active and evidence_tied_to_entry_context(facts, active, import_graph):
            return name
    for item in ranked:
        name = item["name"]
        if name in frameworks.UTILITY_FRAMEWORKS:
            continue
        if evidence_tied_to_entry_context(facts, facts.framework_evidence.get(name, []), import_graph):
            return name
    for item in ranked:
        if item["name"] not in frameworks.UTILITY_FRAMEWORKS:
            return item["name"]
    return ranked[0]["name"]


def _has_authoritative_flare_evidence(facts: InspectionFacts, project_scope: ProjectScope) -> bool:
    evidence_files = {item["file"] for item in facts.flare_imports} | set(facts.flare_calls_by_file)
    return any(authorizes_flare_evidence(project_scope, path) for path in evidence_files)


def _has_authoritative_flare_call(
    facts: InspectionFacts,
    project_scope: ProjectScope,
    call_name: str,
) -> bool:
    return any(
        call_name in calls and authorizes_flare_evidence(project_scope, path)
        for path, calls in facts.flare_calls_by_file.items()
    )


def _has_conversion_integration(facts: InspectionFacts, project_scope: ProjectScope) -> bool:
    # Framework patch calls are definitive conversion signals even when static
    # analysis cannot see the wrapped trainer constructor or an explicit send.
    flare_import_files = {item["file"] for item in facts.flare_imports}
    return any(
        path in flare_import_files and authorizes_flare_evidence(project_scope, path)
        for path in facts.integration_signal_files
    )


def _line_within_ranges(line: Optional[int], ranges: Optional[list[tuple[int, int]]]) -> bool:
    if line is None or not ranges:
        return False
    return any(start <= line <= end for start, end in ranges)
