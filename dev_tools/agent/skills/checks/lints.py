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

"""Deterministic v1 admission lints for NVFLARE-owned agent skills.

DESIGN INVARIANT -- lint engine independence (do not revert):
This engine reads only the ``skills/`` runtime tree and the repo-local eval
suites under ``evals_root`` (``dev_tools/agent/skill_evals/``, one dir per skill
name). It must NOT read ``docs/design/*.md`` or rely on offline-only catalog
metadata. ``SKILL.md`` is a runtime artifact loaded by the agent; fields
validated here must be runtime or public skill metadata, not private lint
scratch data. ``evals_root`` is dev/QA tooling input, distinct from the
forbidden ``docs_root``.

Concretely:
- Group skills for ``skill-trigger-overlap-lint`` by deterministic skill-name
  families (see ``_trigger_overlap_group``), not by frontmatter ``category``
  values or by a product-catalog table parsed from design docs.
- ``category`` is valid public SKILL.md metadata for publishable skills; it is
  not a trigger-overlap grouping source or a docs-catalog sync key for this
  engine.
- Do not add a ``docs_root`` parameter or a ``--docs-root`` flag back to this
  engine, and do not re-introduce ``skill-catalog-category-lint`` /
  ``agent-doc-crosslink-lint`` here.
- Catalog/publication sync (skill listed in the human product catalog) is a
  docs concern: put it in a SEPARATE docs check, not in this skill engine.

Rationale and history: docs/design/skills_architecture.md "Lint engine
independence". A prior change coupled this engine to design docs and made
catalog synchronization part of the skill lint runner; that coupling was
reverted on purpose. Keep category validation local to SKILL.md frontmatter, and
keep docs/catalog synchronization in separate docs tooling.
"""

import json
import os
import re
import shlex
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

try:
    from .frontmatter import SKILL_FILE_NAME, parse_skill_frontmatter, should_skip_skill_dir, validate_skill_dir
except ImportError:
    from frontmatter import SKILL_FILE_NAME, parse_skill_frontmatter, should_skip_skill_dir, validate_skill_dir

V1_LINT_IDS = (
    "skill-frontmatter-lint",
    "skill-md-size-lint",
    "skill-trigger-lint",
    "skill-trigger-overlap-lint",
    "skill-global-negative-lint",
    "skill-policy-coverage-lint",
    "skill-process-metric-lint",
    "skill-command-drift-lint",
    "skill-helper-script-lint",
    "skill-fixture-lint",
    "skill-runtime-boundary-lint",
)

LINT_SKILL_FRONTMATTER = "skill-frontmatter-lint"
LINT_SKILL_MD_SIZE = "skill-md-size-lint"
LINT_SKILL_TRIGGER = "skill-trigger-lint"
LINT_SKILL_TRIGGER_OVERLAP = "skill-trigger-overlap-lint"
LINT_SKILL_GLOBAL_NEGATIVE = "skill-global-negative-lint"
LINT_SKILL_POLICY_COVERAGE = "skill-policy-coverage-lint"
LINT_SKILL_PROCESS_METRIC = "skill-process-metric-lint"
LINT_SKILL_COMMAND_DRIFT = "skill-command-drift-lint"
LINT_SKILL_HELPER_SCRIPT = "skill-helper-script-lint"
LINT_SKILL_FIXTURE = "skill-fixture-lint"
LINT_SKILL_RUNTIME_BOUNDARY = "skill-runtime-boundary-lint"

FINDING_ERROR = "error"
FINDING_WARNING = "warning"
FINDING_INFO = "info"
SKILL_MD_MAX_LINES = 200
SKILL_MD_ADVISORY_WORDS = 2000
MAX_SKILL_TEXT_FILE_BYTES = 512 * 1024
DEFAULT_MAX_TRIGGER_OVERLAP_SKILLS = 200

_PUBLIC_EXEMPT_STATUS = {"draft", "internal", "private"}
_SIZE_EXCEPTION_MARKERS = (
    "nvflare-lint: allow skill-md-size-lint",
    "skill-md-size-lint: approved-exception",
)
_TRIGGER_TERMS = (
    "trigger",
    "use when",
    "when to use",
    "use this skill",
    "do not use",
    "do-not-use",
    "boundary",
)
_BOUNDARY_TERMS = ("do not use", "do-not-use", "use boundary", "boundary", "negative")
_NORMATIVE_RE = re.compile(r"\b(must|must not|required|prohibited|approval)\b", re.IGNORECASE)
_BACKTICK_NVFLARE_RE = re.compile(r"`(nvflare(?:\s+[^`]+)?)`")
_SAFE_COMMAND_TOKEN_RE = re.compile(r"^(?:--?[A-Za-z0-9][\w-]*(?:=[^\s`;&|]+)?|[A-Za-z0-9_./:=+@%,-]+|<[^>\n]+>)$")
_SIGNIFICANT_TOKEN_RE = re.compile(r"[a-z][a-z0-9_-]{2,}")
_STOPWORDS = {
    "and",
    "for",
    "the",
    "this",
    "that",
    "with",
    "into",
    "from",
    "using",
    "when",
    "skill",
    "nvflare",
    "flare",
    "federated",
    "workflow",
}
_KNOWN_NVFLARE_ROOT_COMMANDS = {
    "agent",
    "authz-preview",
    "cert",
    "config",
    "dashboard",
    "deploy",
    "job",
    "package",
    "poc",
    "preflight-check",
    "provision",
    "recipe",
    "simulator",
    "study",
    "system",
}
_KNOWN_AGENT_COMMANDS = {"doctor", "info", "inspect", "skills"}
_KNOWN_AGENT_SKILLS_COMMANDS = {"install", "list"}
_PLANNED_AGENT_SKILLS_COMMANDS = set()
_KNOWN_AGENT_FLAGS = {
    "agent": {"--format", "--schema"},
    "agent doctor": {"--format", "--online", "--schema", "--startup-kit", "--project", "--org"},
    "agent info": {"--format", "--schema"},
    "agent inspect": {"--format", "--redact", "--schema"},
    "agent skills": {"--format", "--schema"},
    "agent skills install": {"--agent", "--dry-run", "--format", "--schema", "--skill", "--target"},
    "agent skills list": {"--agent", "--format", "--schema", "--target"},
}


@dataclass(frozen=True)
class LintFinding:
    id: str
    severity: str
    file: str
    message: str
    hint: str
    line: Optional[int] = None
    code: Optional[str] = None
    skill: Optional[str] = None
    global_finding: bool = False

    def as_dict(self) -> dict[str, Any]:
        data = {
            "id": self.id,
            "severity": self.severity,
            "file": self.file,
            "message": self.message,
            "hint": self.hint,
        }
        if self.line is not None:
            data["line"] = self.line
        if self.code is not None:
            data["code"] = self.code
        if self.skill is not None:
            data["skill"] = self.skill
        if self.global_finding:
            data["global"] = True
        return data


@dataclass
class SkillRecord:
    name: str
    skill_dir: Path
    skill_file: Path
    metadata: dict[str, Any]
    text: str
    body: str
    evals: list[dict[str, Any]]
    # Eval content lives outside the shipped skill tree (repo-only QA data).
    # evals_dir is the skill's eval suite root; evals_path is its evals.json.
    evals_dir: Path
    evals_path: Path
    evals_error: Optional[str]

    @property
    def public(self) -> bool:
        status = str(self.metadata.get("status", "public")).strip().lower()
        return status not in _PUBLIC_EXEMPT_STATUS


@dataclass
class LintContext:
    skills_root: Path
    evals_root: Path
    max_skill_md_lines: int
    records: list[SkillRecord]
    findings: list[LintFinding]
    skipped_checks: list[dict[str, str]]


def run_v1_lints(
    skills_root: Path | str = "skills",
    *,
    evals_root: Path | str | None = None,
    checks: Optional[Iterable[str]] = None,
    max_skill_md_lines: int = SKILL_MD_MAX_LINES,
) -> dict[str, Any]:
    """Run deterministic v1 admission lints and return structured findings."""
    result, _records = _run_v1_lints_with_records(
        skills_root,
        evals_root=evals_root,
        checks=checks,
        max_skill_md_lines=max_skill_md_lines,
    )
    return result


def _default_evals_root(skills_root: Path) -> Path:
    # Eval suites live in a repo-local dev-tools tree, one directory per skill
    # name, alongside the skills source root (not shipped in installed skills).
    return skills_root.resolve().parent / "dev_tools" / "agent" / "skill_evals"


def _run_v1_lints_with_records(
    skills_root: Path | str = "skills",
    *,
    evals_root: Path | str | None = None,
    checks: Optional[Iterable[str]] = None,
    max_skill_md_lines: int = SKILL_MD_MAX_LINES,
) -> tuple[dict[str, Any], list[SkillRecord]]:
    selected = tuple(checks or V1_LINT_IDS)
    unknown = sorted(set(selected).difference(V1_LINT_IDS))
    if unknown:
        raise ValueError(f"unknown agent skill lint check(s): {', '.join(unknown)}")

    root = Path(skills_root)
    evals_root_path = Path(evals_root) if evals_root is not None else _default_evals_root(root)
    findings: list[LintFinding] = []
    records = _load_skill_records(root, evals_root_path, findings)
    context = LintContext(
        skills_root=root,
        evals_root=evals_root_path,
        max_skill_md_lines=max_skill_md_lines,
        records=records,
        findings=findings,
        skipped_checks=[],
    )
    root_error_codes = {"skills-root-missing", "skills-root-not-directory"}
    if any(finding.global_finding and finding.code in root_error_codes for finding in findings):
        summary = _summary_from_lint_findings(context.findings, len(records))
        status = "failed" if summary["error_count"] else "ok"
        return {
            "schema_version": "1",
            "status": status,
            "passed": status == "ok",
            "skills_root": str(root),
            "checks": list(selected),
            "skipped_checks": context.skipped_checks,
            "summary": summary,
            "findings": [finding.as_dict() for finding in context.findings],
        }, records

    lint_functions = {
        "skill-frontmatter-lint": _lint_frontmatter,
        "skill-md-size-lint": _lint_md_size,
        "skill-trigger-lint": _lint_trigger,
        "skill-trigger-overlap-lint": _lint_trigger_overlap,
        "skill-global-negative-lint": _lint_global_negative,
        "skill-policy-coverage-lint": _lint_policy_coverage,
        "skill-process-metric-lint": _lint_process_metrics,
        "skill-command-drift-lint": _lint_command_drift,
        "skill-helper-script-lint": _lint_helper_scripts,
        "skill-fixture-lint": _lint_fixtures,
        "skill-runtime-boundary-lint": _lint_runtime_boundary,
    }

    for check in selected:
        lint_functions[check](context)

    summary = _summary_from_lint_findings(context.findings, len(records))
    status = "failed" if summary["error_count"] else "ok"
    return {
        "schema_version": "1",
        "status": status,
        "passed": status == "ok",
        "skills_root": str(root),
        "checks": list(selected),
        "skipped_checks": context.skipped_checks,
        "summary": summary,
        "findings": [finding.as_dict() for finding in context.findings],
    }, records


def validate_skills(
    skills_root: Path | str = "skills",
    *,
    evals_root: Path | str | None = None,
    skill_name: Optional[str] = None,
    max_skill_md_lines: int = SKILL_MD_MAX_LINES,
) -> dict[str, Any]:
    """Compatibility wrapper for callers that validate one skill source root."""
    result, records = _run_v1_lints_with_records(
        skills_root,
        evals_root=evals_root,
        max_skill_md_lines=max_skill_md_lines,
    )

    if skill_name is not None:
        result["requested_skill"] = skill_name
        result["findings"] = [
            finding for finding in result["findings"] if _finding_matches_requested_skill(finding, skill_name)
        ]
        result["summary"] = _summary_from_findings(result["findings"], _matching_skill_count(records, skill_name))
        result["status"] = "failed" if result["summary"]["error_count"] else "ok"
        result["passed"] = result["status"] == "ok"
    else:
        result["requested_skill"] = None
    return result


def _summary_from_findings(findings: list[dict[str, Any]], skill_count: int) -> dict[str, int]:
    severity_counts = Counter(finding.get("severity", FINDING_ERROR) for finding in findings)
    return _summary_from_counts(severity_counts, len(findings), skill_count)


def _finding_matches_requested_skill(finding: dict[str, Any], skill_name: str) -> bool:
    finding_skill = finding.get("skill")
    return finding_skill == skill_name or (finding_skill is None and finding.get("global") is True)


def _summary_from_lint_findings(findings: list[LintFinding], skill_count: int) -> dict[str, int]:
    severity_counts = Counter(finding.severity for finding in findings)
    return _summary_from_counts(severity_counts, len(findings), skill_count)


def _summary_from_counts(severity_counts: Counter, finding_count: int, skill_count: int) -> dict[str, int]:
    return {
        "skill_count": skill_count,
        "finding_count": finding_count,
        "error_count": severity_counts.get(FINDING_ERROR, 0),
        "warning_count": severity_counts.get(FINDING_WARNING, 0),
        "info_count": severity_counts.get(FINDING_INFO, 0),
    }


def _load_skill_records(skills_root: Path, evals_root: Path, findings: list[LintFinding]) -> list[SkillRecord]:
    if not skills_root.exists():
        findings.append(
            _finding(
                "skill-frontmatter-lint",
                FINDING_ERROR,
                skills_root,
                "skills root does not exist",
                "Pass --skills-root pointing at the repository skills/ directory.",
                code="skills-root-missing",
                global_finding=True,
            )
        )
        return []
    if not skills_root.is_dir():
        findings.append(
            _finding(
                "skill-frontmatter-lint",
                FINDING_ERROR,
                skills_root,
                "skills root is not a directory",
                "Pass --skills-root pointing at the repository skills/ directory.",
                code="skills-root-not-directory",
                global_finding=True,
            )
        )
        return []

    records = []
    for child in sorted(skills_root.iterdir(), key=lambda p: p.name):
        if should_skip_skill_dir(child):
            continue
        skill_file = child / SKILL_FILE_NAME
        text = _read_bounded_text(skill_file) if skill_file.is_file() else None
        metadata = _try_parse_frontmatter(skill_file) if text is not None else {}
        text = text or ""
        skill_name = str(metadata.get("name") or child.name)
        # Eval suites live outside the shipped skill tree, one dir per skill name.
        evals_dir = evals_root / skill_name
        evals_path = evals_dir / "evals.json"
        evals, evals_error = _load_evals(evals_path)
        records.append(
            SkillRecord(
                name=skill_name,
                skill_dir=child,
                skill_file=skill_file,
                metadata=metadata,
                text=text,
                body=_skill_body(text),
                evals=evals,
                evals_dir=evals_dir,
                evals_path=evals_path,
                evals_error=evals_error,
            )
        )
    return records


def _matching_skill_count(records: list[SkillRecord], skill_name: str) -> int:
    return sum(1 for record in records if record.name == skill_name)


def _lint_frontmatter(context: LintContext) -> None:
    for record in context.records:
        result = validate_skill_dir(record.skill_dir)
        for issue in result.issues:
            context.findings.append(
                _finding(
                    "skill-frontmatter-lint",
                    FINDING_ERROR,
                    Path(issue.path),
                    issue.message,
                    "Fix SKILL.md frontmatter before publishing this skill.",
                    code=issue.code,
                    skill=record.name,
                    line=_line_for_frontmatter_issue(record.skill_file, issue.code, issue.message),
                )
            )

        if record.public and _has_valid_name(record.metadata) and not record.name.startswith("nvflare-"):
            context.findings.append(
                _finding(
                    "skill-frontmatter-lint",
                    FINDING_ERROR,
                    record.skill_file,
                    f"public NVFLARE skill name '{record.name}' must start with 'nvflare-'",
                    "Rename the skill directory and frontmatter name, or mark the skill draft/internal.",
                    code="skill-name-prefix-required",
                    skill=record.name,
                    line=_line_for_field(record.skill_file, "name"),
                )
            )


def _has_valid_name(metadata: dict[str, Any]) -> bool:
    name = metadata.get("name")
    return isinstance(name, str) and bool(name.strip())


def _lint_md_size(context: LintContext) -> None:
    for record in context.records:
        if not record.skill_file.is_file():
            continue
        if _is_oversized_text_file(record.skill_file):
            if _has_bounded_size_exception(record.skill_file):
                continue
            context.findings.append(
                _finding(
                    "skill-md-size-lint",
                    FINDING_ERROR,
                    record.skill_file,
                    f"SKILL.md exceeds the readable size limit of {MAX_SKILL_TEXT_FILE_BYTES} bytes",
                    "Move detailed workflow notes into references/ or add an approved exception marker near the top.",
                    code="skill-md-too-large",
                    skill=record.name,
                    line=1,
                )
            )
            continue
        lines = record.text.splitlines()
        max_lines = context.max_skill_md_lines
        if len(lines) > max_lines and not _has_size_exception(record.text):
            context.findings.append(
                _finding(
                    "skill-md-size-lint",
                    FINDING_ERROR,
                    record.skill_file,
                    f"SKILL.md has {len(lines)} lines; v1 hard limit is {max_lines}",
                    "Move detailed workflow notes into references/ or add an approved exception marker.",
                    code="skill-md-too-large",
                    skill=record.name,
                    line=max_lines + 1,
                )
            )
        word_count = len(record.text.split())
        if word_count > SKILL_MD_ADVISORY_WORDS:
            context.findings.append(
                _finding(
                    "skill-md-size-lint",
                    FINDING_INFO,
                    record.skill_file,
                    f"SKILL.md has about {word_count} whitespace-delimited tokens",
                    "The roughly 2,000-token target is advisory; keep SKILL.md concise when practical.",
                    code="skill-md-token-advisory",
                    skill=record.name,
                )
            )


def _lint_trigger(context: LintContext) -> None:
    for record in _public_records(context.records):
        searchable = f"{record.metadata.get('description', '')}\n{record.body}".lower()
        if not any(term in searchable for term in _TRIGGER_TERMS):
            context.findings.append(
                _finding(
                    "skill-trigger-lint",
                    FINDING_ERROR,
                    record.skill_file,
                    "skill is missing trigger or use-boundary text",
                    "Add concise trigger guidance and negative boundary language to SKILL.md.",
                    code="skill-trigger-text-missing",
                    skill=record.name,
                )
            )

        if record.evals_error:
            _add_evals_error(context, "skill-trigger-lint", record)
            continue
        if not record.evals_path.is_file():
            context.findings.append(
                _finding(
                    "skill-trigger-lint",
                    FINDING_ERROR,
                    record.evals_path,
                    "evals/evals.json is required for public skill trigger checks",
                    "Add guide-compatible evals/evals.json with positive and adjacent negative trigger evals.",
                    code="skill-evals-missing",
                    skill=record.name,
                )
            )
            continue
        if not any(_is_positive_eval(item, record.name) for item in record.evals):
            context.findings.append(
                _finding(
                    "skill-trigger-lint",
                    FINDING_ERROR,
                    record.evals_path,
                    "missing positive trigger eval for this skill",
                    "Add an eval whose nvflare.expected_skill matches this skill.",
                    code="skill-positive-trigger-eval-missing",
                    skill=record.name,
                )
            )
        if not any(_is_adjacent_negative_eval(item, record.name) for item in record.evals):
            context.findings.append(
                _finding(
                    "skill-trigger-lint",
                    FINDING_ERROR,
                    record.evals_path,
                    "missing adjacent negative trigger eval for this skill",
                    "Add an eval whose nvflare.negative_for names this skill and expected_skill names the neighbor.",
                    code="skill-adjacent-negative-eval-missing",
                    skill=record.name,
                )
            )


def _lint_trigger_overlap(context: LintContext) -> None:
    grouped: dict[str, list[SkillRecord]] = defaultdict(list)
    for record in _public_records(context.records):
        group = _trigger_overlap_group(record.name)
        grouped[group].append(record)

    max_trigger_overlap_skills = _max_trigger_overlap_skills()
    for group, records in grouped.items():
        if len(records) > max_trigger_overlap_skills:
            _skip(
                context,
                "skill-trigger-overlap-lint",
                f"group {group!r} has {len(records)} skills; limit is {max_trigger_overlap_skills}",
            )
            continue
        token_cache = {record.name: _trigger_tokens(record) for record in records}
        for i, left in enumerate(records):
            for right in records[i + 1 :]:
                if not _records_overlap(left, right, token_cache):
                    continue
                if _has_boundary_text(left) and _has_boundary_text(right) and _has_adjacent_negative_pair(left, right):
                    continue
                context.findings.append(
                    _finding(
                        "skill-trigger-overlap-lint",
                        FINDING_ERROR,
                        left.skill_file,
                        f"same trigger-group skills '{left.name}' and '{right.name}' have overlapping trigger language",
                        "Add use/do-not-use boundaries and adjacent negative evals covering the overlap.",
                        code="skill-trigger-overlap",
                        skill=left.name,
                    )
                )


def _trigger_overlap_group(skill_name: str) -> str:
    normalized = skill_name.strip().lower()
    if normalized.startswith("nvflare-"):
        normalized = normalized[len("nvflare-") :]
    family = normalized.split("-", maxsplit=1)[0].strip()
    if family:
        return f"nvflare-{family}"
    return skill_name.strip().lower() or skill_name


def _max_trigger_overlap_skills() -> int:
    value = os.environ.get("NVFLARE_AGENT_MAX_TRIGGER_OVERLAP_SKILLS")
    if value is None or value == "":
        return DEFAULT_MAX_TRIGGER_OVERLAP_SKILLS
    try:
        parsed = int(value)
    except ValueError:
        return DEFAULT_MAX_TRIGGER_OVERLAP_SKILLS
    return parsed if parsed > 0 else DEFAULT_MAX_TRIGGER_OVERLAP_SKILLS


def _lint_global_negative(context: LintContext) -> None:
    for record in _public_records(context.records):
        if record.evals_error:
            _add_evals_error(context, "skill-global-negative-lint", record)
            continue
        if not record.evals_path.is_file():
            context.findings.append(
                _finding(
                    "skill-global-negative-lint",
                    FINDING_ERROR,
                    record.evals_path,
                    "evals/evals.json is required for global negative coverage",
                    "Add at least one eval representing an unrelated prompt that should trigger no FLARE skill.",
                    code="skill-evals-missing",
                    skill=record.name,
                )
            )
            continue
        if not any(_is_global_negative_eval(item) for item in record.evals):
            context.findings.append(
                _finding(
                    "skill-global-negative-lint",
                    FINDING_ERROR,
                    record.evals_path,
                    "missing global negative eval",
                    "Add an eval for an unrelated prompt with nvflare.expected_skill set to null or 'none'.",
                    code="skill-global-negative-eval-missing",
                    skill=record.name,
                )
            )


def _lint_policy_coverage(context: LintContext) -> None:
    for record in _public_records(context.records):
        has_behavior_ids = any(_behavior_id_count(item) for item in record.evals)
        has_helper_tests = _skill_has_helper_tests(record.skill_dir)
        has_checklist = _skill_text_contains(record.skill_dir, "checklist")
        if has_behavior_ids or has_helper_tests or has_checklist:
            continue

        for file_path, text in _iter_skill_text_files(record.skill_dir):
            for line_no, line in enumerate(text.splitlines(), start=1):
                if _NORMATIVE_RE.search(line):
                    context.findings.append(
                        _finding(
                            "skill-policy-coverage-lint",
                            FINDING_ERROR,
                            file_path,
                            "normative rule has no measurable behavior ID, helper test, or checklist coverage",
                            "Map required/prohibited behavior to evals/evals.json nvflare behavior IDs or tests.",
                            code="skill-policy-coverage-missing",
                            skill=record.name,
                            line=line_no,
                        )
                    )
                    break
            else:
                continue
            break


def _lint_process_metrics(context: LintContext) -> None:
    for record in _public_records(context.records):
        if record.evals_error:
            _add_evals_error(context, LINT_SKILL_PROCESS_METRIC, record)
            continue
        if not record.evals_path.is_file():
            context.findings.append(
                _finding(
                    LINT_SKILL_PROCESS_METRIC,
                    FINDING_ERROR,
                    record.evals_path,
                    "evals/evals.json is required for process-metric coverage",
                    "Add process metric contracts under nvflare.process_metrics.",
                    code="skill-evals-missing",
                    skill=record.name,
                )
            )
            continue

        process_metrics = []
        for item in record.evals:
            for metric in _process_metrics(item):
                process_metrics.append((item, metric))

        if not process_metrics:
            context.findings.append(
                _finding(
                    LINT_SKILL_PROCESS_METRIC,
                    FINDING_ERROR,
                    record.evals_path,
                    "missing process metric contracts for this public skill",
                    "Add nvflare.process_metrics entries for first-pass quality, correction count, unwanted actions, "
                    "validation evidence, or other skill-process outcomes.",
                    code="skill-process-metric-missing",
                    skill=record.name,
                )
            )
            continue

        for item, metric in process_metrics:
            if not isinstance(metric, dict):
                context.findings.append(
                    _finding(
                        LINT_SKILL_PROCESS_METRIC,
                        FINDING_ERROR,
                        record.evals_path,
                        f"eval '{item.get('id', '<missing>')}' process metric must be an object",
                        "Use objects with at least id and description fields.",
                        code="skill-process-metric-type",
                        skill=record.name,
                    )
                )
                continue
            metric_id = metric.get("id")
            description = metric.get("description")
            if not isinstance(metric_id, str) or not metric_id.strip():
                context.findings.append(
                    _finding(
                        LINT_SKILL_PROCESS_METRIC,
                        FINDING_ERROR,
                        record.evals_path,
                        f"eval '{item.get('id', '<missing>')}' process metric is missing id",
                        "Add a stable metric id such as turns_to_acceptable or user_correction_count.",
                        code="skill-process-metric-id-missing",
                        skill=record.name,
                    )
                )
            if not isinstance(description, str) or not description.strip():
                context.findings.append(
                    _finding(
                        LINT_SKILL_PROCESS_METRIC,
                        FINDING_ERROR,
                        record.evals_path,
                        f"eval '{item.get('id', '<missing>')}' process metric '{metric_id}' is missing description",
                        "Describe what the metric measures and what evidence records it.",
                        code="skill-process-metric-description-missing",
                        skill=record.name,
                    )
                )


def _lint_command_drift(context: LintContext) -> None:
    for record in _public_records(context.records):
        for file_path, text in _iter_skill_text_files(record.skill_dir, include_scripts=True):
            for line_no, command in _extract_nvflare_commands(text):
                message = _command_drift_message(command)
                if message is None:
                    continue
                context.findings.append(
                    _finding(
                        "skill-command-drift-lint",
                        FINDING_ERROR,
                        file_path,
                        message,
                        "Update the referenced nvflare command or the CLI command schema before publishing.",
                        code="skill-command-drift",
                        skill=record.name,
                        line=line_no,
                    )
                )


def _lint_helper_scripts(context: LintContext) -> None:
    for record in _public_records(context.records):
        scripts_dir = record.skill_dir / "scripts"
        if not scripts_dir.is_dir():
            continue
        script_files = list(_iter_files_no_follow(scripts_dir))
        if script_files and not _skill_has_helper_tests(record.skill_dir):
            context.findings.append(
                _finding(
                    "skill-helper-script-lint",
                    FINDING_ERROR,
                    scripts_dir,
                    "helper scripts are shipped without tests",
                    "Add tests under the skill or repository test tree for each shipped helper script.",
                    code="skill-helper-tests-missing",
                    skill=record.name,
                )
            )

        for script in script_files:
            try:
                if script.stat().st_size > MAX_SKILL_TEXT_FILE_BYTES:
                    continue
            except OSError:
                continue
            text = script.read_text(encoding="utf-8", errors="replace")
            lowered = text.lower()
            if "promoted_to:" in lowered or "_promoted_to:" in lowered:
                context.findings.append(
                    _finding(
                        "skill-helper-script-lint",
                        FINDING_ERROR,
                        script,
                        "helper script is marked as promoted to a product CLI command",
                        "Update SKILL.md to call the product CLI command instead of the promoted helper script.",
                        code="skill-helper-promoted",
                        skill=record.name,
                    )
                )
            declares_json_output = any(
                token in lowered for token in ("json output", "stdout json", "json stdout", "jsonl", "machine-readable")
            )
            if script.suffix == ".py" and declares_json_output and "json.dumps" not in text and "json.dump" not in text:
                context.findings.append(
                    _finding(
                        "skill-helper-script-lint",
                        FINDING_WARNING,
                        script,
                        "script mentions JSON but does not appear to write JSON with the json module",
                        "Ensure machine-readable stdout is valid JSON and diagnostics go to stderr.",
                        code="skill-helper-json-unclear",
                        skill=record.name,
                    )
                )


def _lint_fixtures(context: LintContext) -> None:
    for record in _public_records(context.records):
        if record.evals_error:
            _add_evals_error(context, "skill-fixture-lint", record)
            continue
        if not record.evals_path.is_file():
            continue

        for item in record.evals:
            files = item.get("files", [])
            if files is None:
                files = []
            if not isinstance(files, list):
                context.findings.append(
                    _finding(
                        "skill-fixture-lint",
                        FINDING_ERROR,
                        record.evals_path,
                        f"eval '{item.get('id', '<missing>')}' files field must be a list",
                        "Use guide-compatible files: [...] entries relative to the skill directory.",
                        code="skill-fixture-files-type",
                        skill=record.name,
                    )
                )
                continue

            if _eval_mentions_file_editing(item) and not files:
                context.findings.append(
                    _finding(
                        "skill-fixture-lint",
                        FINDING_ERROR,
                        record.evals_path,
                        f"eval '{item.get('id', '<missing>')}' describes file editing without input fixtures",
                        "Add deterministic input files under evals/files/ and reference them from the eval.",
                        code="skill-fixture-input-missing",
                        skill=record.name,
                    )
                )

            for rel_path in files:
                fixture_path = record.evals_dir / str(rel_path)
                resolved_fixture_path = fixture_path.resolve(strict=False)
                resolved_evals_dir = record.evals_dir.resolve()
                if not resolved_fixture_path.is_relative_to(resolved_evals_dir):
                    context.findings.append(
                        _finding(
                            "skill-fixture-lint",
                            FINDING_ERROR,
                            record.evals_path,
                            f"eval fixture path escapes eval suite directory: {rel_path}",
                            "Use fixture paths relative to the eval suite directory.",
                            code="skill-fixture-path-escape",
                            skill=record.name,
                        )
                    )
                    continue
                if not fixture_path.is_file():
                    context.findings.append(
                        _finding(
                            "skill-fixture-lint",
                            FINDING_ERROR,
                            record.evals_path,
                            f"eval fixture does not exist: {rel_path}",
                            "Place deterministic fixtures under evals/files/ and reference existing files.",
                            code="skill-fixture-file-missing",
                            skill=record.name,
                        )
                    )

        files_dir = record.evals_dir / "files"
        if _has_files(files_dir) and not _has_fixture_notes(record.evals_dir):
            context.findings.append(
                _finding(
                    "skill-fixture-lint",
                    FINDING_ERROR,
                    files_dir,
                    "eval fixtures are missing source/provenance notes",
                    "Add README.md, files/README.md, or files/SOURCE.md in the eval suite dir.",
                    code="skill-fixture-notes-missing",
                    skill=record.name,
                )
            )


_DESIGN_DOC_REF_RE = re.compile(r"docs[\\/]+design\b", re.IGNORECASE)
_EVALUATOR_HOOK_RE = re.compile(
    r"(?:"
    r"\bevals?/"
    r"|\bevals\.json\b"
    r"|\beval\s+cases?\b"
    r"|\beval\s+fixtures?\b"
    r"|\bevaluator\b"
    r"|\bbenchmark[ -]?harness\b"  # benchmark-harness instructions (not "benchmark dataset")
    r"|\beval\s+(?:suite|harness)\b"  # evaluator harness references
    r"|\bgrader\b"  # eval grader references
    r"|\beval\s*=\s*\w"  # eval-mode toggles such as eval=on
    r"|--eval\b"  # evaluator harness flags such as --eval / --eval-only
    r"|(?-i:\b[A-Z][A-Z0-9_]*_EVAL(?![A-Z])[A-Z0-9_]*\b)"  # env vars such as NVFLARE_SKILL_EVAL, not DEFAULT_EVALUATION
    r")",
    re.IGNORECASE,
)
# Content directories that release packaging strips or that are not shipped as
# runtime guidance. Mirrors nvflare release packaging exclusions without
# importing product code (keeps this lint engine self-contained over skills/).
_RUNTIME_BOUNDARY_EXCLUDED_DIRS = {"evals", "__pycache__"}
_RUNTIME_TEXT_SUFFIXES = {
    ".md",
    ".txt",
    ".rst",
    ".py",
    ".sh",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".cfg",
    ".ini",
    ".j2",
    ".jinja",
    ".jinja2",
    "",
}


def _lint_runtime_boundary(context: LintContext) -> None:
    """Packaged runtime skill content must stay inside the runtime boundary.

    Runtime content is everything a skill ships. It must not contain an
    ``evals/`` suite (grading-oracle data belongs in the repo-only eval root,
    not inside a shipped skill), must not reference ``docs/design/`` documents
    as operational guidance, and must not contain evaluator hooks or
    benchmark-harness-only instructions. The scan covers what packaging ships,
    so it iterates every skill record (public and non-public) and every shared
    reference directory, not only ``SKILL.md`` and ``.md`` references.
    """
    for record in context.records:
        if (record.skill_dir / "evals").is_dir():
            context.findings.append(
                _finding(
                    LINT_SKILL_RUNTIME_BOUNDARY,
                    FINDING_ERROR,
                    record.skill_dir / "evals",
                    "eval suite must not live inside a shipped skill directory",
                    "Move the eval suite to the eval root (dev_tools/agent/skill_evals/<skill>/); "
                    "grading-oracle data must not ship in installed skills.",
                    code="skill-runtime-eval-dir-in-skill",
                    skill=record.name,
                )
            )
        for file_path, text in _iter_packaged_runtime_files(record.skill_dir):
            _scan_runtime_boundary(context, file_path, text, skill=record.name)
    for file_path, text in _iter_shared_runtime_files(context.skills_root):
        _scan_runtime_boundary(context, file_path, text, skill=None)


def _iter_packaged_runtime_files(skill_dir: Path) -> Iterable[tuple[Path, str]]:
    """Yield decoded text files a skill ships as runtime content (minus evals/)."""
    if not skill_dir.is_dir():
        return
    for path in _iter_files_no_follow(skill_dir):
        if any(part in _RUNTIME_BOUNDARY_EXCLUDED_DIRS for part in path.relative_to(skill_dir).parts):
            continue
        content = _read_runtime_text_file(path)
        if content is not None:
            yield path, content


def _iter_shared_runtime_files(skills_root: Path) -> Iterable[tuple[Path, str]]:
    if not skills_root.is_dir():
        return
    for child in sorted(skills_root.iterdir(), key=lambda p: p.name):
        if not child.is_dir() or child.is_symlink() or not child.name.startswith("_"):
            continue
        yield from _iter_packaged_runtime_files(child)


def _read_runtime_text_file(path: Path) -> Optional[str]:
    # Runtime-scan-specific guards (skip symlinked files and non-text suffixes),
    # then defer the size-cap + bounded read to the shared reader.
    if path.is_symlink() or path.suffix.lower() not in _RUNTIME_TEXT_SUFFIXES:
        return None
    return _read_bounded_text(path)


def _scan_runtime_boundary(context: LintContext, file_path: Path, text: str, *, skill: Optional[str]) -> None:
    for line_no, line in enumerate(text.splitlines(), start=1):
        if _DESIGN_DOC_REF_RE.search(line):
            context.findings.append(
                _finding(
                    LINT_SKILL_RUNTIME_BOUNDARY,
                    FINDING_ERROR,
                    file_path,
                    "packaged runtime skill content references docs/design/ documents",
                    "Copy the runtime-relevant rule into SKILL.md/reference content or product docs; "
                    "design docs are authoring and review inputs.",
                    code="skill-runtime-design-doc-ref",
                    skill=skill,
                    line=line_no,
                    global_finding=skill is None,
                )
            )
        if _EVALUATOR_HOOK_RE.search(line):
            context.findings.append(
                _finding(
                    LINT_SKILL_RUNTIME_BOUNDARY,
                    FINDING_ERROR,
                    file_path,
                    "packaged runtime skill content contains evaluator or benchmark-harness instructions",
                    "Keep evaluator hooks and benchmark instructions in repo-only evals/ content, "
                    "not in SKILL.md, references/, scripts/, or shared references.",
                    code="skill-runtime-evaluator-hook",
                    skill=skill,
                    line=line_no,
                    global_finding=skill is None,
                )
            )


def _try_parse_frontmatter(skill_file: Path) -> dict[str, Any]:
    try:
        return parse_skill_frontmatter(skill_file)
    except Exception:
        return {}


def _skill_body(text: str) -> str:
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return text
    for index, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            return "\n".join(lines[index + 1 :])
    return text


def _load_evals(evals_path: Path) -> tuple[list[dict[str, Any]], Optional[str]]:
    if not evals_path.is_file():
        return [], None
    if _is_oversized_text_file(evals_path):
        return [], f"evals/evals.json exceeds size limit ({MAX_SKILL_TEXT_FILE_BYTES} bytes)"
    try:
        raw = json.loads(evals_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        return [], f"failed to parse evals/evals.json: {e}"
    if isinstance(raw, dict):
        items = raw.get("evals", [])
    elif isinstance(raw, list):
        items = raw
    else:
        return [], "evals/evals.json must be an object with an evals list or a list"
    if not isinstance(items, list):
        return [], "evals/evals.json field 'evals' must be a list"
    evals = [item for item in items if isinstance(item, dict)]
    if len(evals) != len(items):
        return evals, "each evals/evals.json entry must be an object"
    return evals, None


def _add_evals_error(context: LintContext, lint_id: str, record: SkillRecord) -> None:
    context.findings.append(
        _finding(
            lint_id,
            FINDING_ERROR,
            record.evals_path,
            record.evals_error or "evals/evals.json is invalid",
            "Use guide-compatible JSON with an evals list.",
            code="skill-evals-invalid",
            skill=record.name,
        )
    )


def _public_records(records: list[SkillRecord]) -> list[SkillRecord]:
    return [record for record in records if record.public]


def _is_positive_eval(item: dict[str, Any], skill_name: str) -> bool:
    nvflare = _nvflare_ext(item)
    if nvflare.get("expected_skill") == skill_name:
        return True
    tags = _eval_tags(item)
    return "positive" in tags or "positive-trigger" in tags


def _is_adjacent_negative_eval(item: dict[str, Any], skill_name: str) -> bool:
    nvflare = _nvflare_ext(item)
    if nvflare.get("negative_for") == skill_name:
        return True
    tags = _eval_tags(item)
    return "adjacent-negative" in tags or "adjacent_negative" in tags


def _is_global_negative_eval(item: dict[str, Any]) -> bool:
    nvflare = _nvflare_ext(item)
    expected_skill = nvflare.get("expected_skill")
    if expected_skill is None and "expected_skill" in nvflare:
        return True
    if isinstance(expected_skill, str) and expected_skill.lower() in {"none", "no-skill", "no_skill"}:
        return True
    if nvflare.get("negative_for") == "*":
        return True
    tags = _eval_tags(item)
    text = _eval_text(item).lower()
    return "global-negative" in tags or "global_negative" in tags or "trigger no flare skill" in text


def _eval_tags(item: dict[str, Any]) -> set[str]:
    values = item.get("tags", [])
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, list):
        return set()
    return {str(value).strip().lower() for value in values}


def _nvflare_ext(item: dict[str, Any]) -> dict[str, Any]:
    nvflare = item.get("nvflare", {})
    return nvflare if isinstance(nvflare, dict) else {}


def _behavior_id_count(item: dict[str, Any]) -> int:
    nvflare = _nvflare_ext(item)
    count = 0
    for key in ("mandatory_behavior", "optional_behavior", "prohibited_behavior"):
        values = nvflare.get(key, [])
        if isinstance(values, list):
            count += len(values)
    return count


def _process_metrics(item: dict[str, Any]) -> list[Any]:
    nvflare = _nvflare_ext(item)
    metrics = nvflare.get("process_metrics", [])
    return metrics if isinstance(metrics, list) else [metrics]


def _eval_text(item: dict[str, Any]) -> str:
    parts = [str(item.get("id", "")), str(item.get("prompt", "")), str(item.get("expected_output", ""))]
    assertions = item.get("assertions", [])
    if isinstance(assertions, list):
        parts.extend(str(assertion) for assertion in assertions)
    return "\n".join(parts)


def _records_overlap(left: SkillRecord, right: SkillRecord, token_cache: dict[str, set[str]] | None = None) -> bool:
    token_cache = token_cache or {}
    left_tokens = token_cache[left.name] if left.name in token_cache else _trigger_tokens(left)
    right_tokens = token_cache[right.name] if right.name in token_cache else _trigger_tokens(right)
    if not left_tokens or not right_tokens:
        return False
    shared = left_tokens.intersection(right_tokens)
    smaller = min(len(left_tokens), len(right_tokens))
    return len(shared) >= 4 and (len(shared) / smaller) >= 0.35


def _trigger_tokens(record: SkillRecord) -> set[str]:
    prompts = "\n".join(str(item.get("prompt", "")) for item in record.evals)
    text = f"{record.metadata.get('description', '')}\n{prompts}"
    return {
        token
        for token in _SIGNIFICANT_TOKEN_RE.findall(text.lower())
        if token not in _STOPWORDS and not token.startswith("nvflare")
    }


def _has_boundary_text(record: SkillRecord) -> bool:
    text = f"{record.metadata.get('description', '')}\n{record.body}".lower()
    return any(term in text for term in _BOUNDARY_TERMS)


def _has_adjacent_negative_pair(left: SkillRecord, right: SkillRecord) -> bool:
    return any(_negative_for_neighbor(item, left.name, right.name) for item in left.evals) or any(
        _negative_for_neighbor(item, right.name, left.name) for item in right.evals
    )


def _negative_for_neighbor(item: dict[str, Any], skill_name: str, neighbor_name: str) -> bool:
    nvflare = _nvflare_ext(item)
    return nvflare.get("negative_for") == skill_name and nvflare.get("expected_skill") == neighbor_name


def _strip_backticks(value: str) -> str:
    return value.strip().strip("`")


def _extract_nvflare_commands(text: str) -> list[tuple[int, str]]:
    commands = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        matches = list(_BACKTICK_NVFLARE_RE.finditer(line))
        if matches:
            commands.extend((line_no, match.group(1).strip()) for match in matches)
            continue
        index = line.find("nvflare ")
        if index == -1:
            continue
        commands.append((line_no, _trim_command(line[index:])))
    return commands


def _trim_command(text: str) -> str:
    text = text.strip()
    for separator in ("&&", "|", ";"):
        if separator in text:
            text = text.split(separator, 1)[0].strip()
    return text


def _command_drift_message(command: str, *, check_flags: bool = True, allow_planned: bool = False) -> Optional[str]:
    tokens = _command_tokens(command)
    if not tokens or tokens[0] != "nvflare":
        return None
    positional = [token for token in tokens[1:] if not token.startswith("-") and not _looks_like_value(token)]
    if not positional:
        return None
    root = positional[0]
    if root not in _KNOWN_NVFLARE_ROOT_COMMANDS:
        return f"unknown nvflare command root '{root}' in '{command}'"
    if root != "agent":
        return None

    if len(positional) >= 2 and positional[1] not in _KNOWN_AGENT_COMMANDS:
        return f"unknown nvflare agent command '{positional[1]}' in '{command}'"
    if len(positional) >= 3 and positional[1] == "skills":
        skills_command = positional[2]
        if skills_command not in _KNOWN_AGENT_SKILLS_COMMANDS:
            if not allow_planned or skills_command not in _PLANNED_AGENT_SKILLS_COMMANDS:
                return f"unknown nvflare agent skills command '{skills_command}' in '{command}'"

    command_key = " ".join(positional[:3] if len(positional) >= 3 and positional[1] == "skills" else positional[:2])
    allowed_flags = _KNOWN_AGENT_FLAGS.get(command_key, _KNOWN_AGENT_FLAGS.get(root, set()))
    if check_flags:
        for token in tokens:
            if token.startswith("--"):
                flag = token.split("=", 1)[0]
                if flag not in allowed_flags:
                    return f"unknown flag '{flag}' for 'nvflare {command_key}' in '{command}'"
    return None


def _command_tokens(command: str) -> list[str]:
    try:
        tokens = shlex.split(command)
    except ValueError:
        return []
    try:
        start = tokens.index("nvflare")
    except ValueError:
        return []
    command_tokens = []
    for token in tokens[start:]:
        if token in {"&&", "|", ";"}:
            break
        if not _safe_command_token(token):
            break
        command_tokens.append(token)
    return command_tokens


def _safe_command_token(token: str) -> bool:
    return bool(_SAFE_COMMAND_TOKEN_RE.match(token))


def _looks_like_value(token: str) -> bool:
    return token.startswith("<") or "/" in token or token in {"on", "off", "json", "jsonl", "human"}


def _skill_has_helper_tests(skill_dir: Path) -> bool:
    tests_dir = skill_dir / "tests"
    if tests_dir.is_dir() and any(True for _path in _iter_files_no_follow(tests_dir)):
        return True
    return any(path.name.endswith(("_test.py", ".test.py")) for path in _iter_files_no_follow(skill_dir))


def _skill_text_contains(skill_dir: Path, needle: str) -> bool:
    needle = needle.lower()
    return any(needle in text.lower() for _, text in _iter_skill_text_files(skill_dir))


def _iter_skill_text_files(skill_dir: Path, *, include_scripts: bool = False) -> Iterable[tuple[Path, str]]:
    candidates = [skill_dir / SKILL_FILE_NAME]
    references_dir = skill_dir / "references"
    if references_dir.is_dir():
        candidates.extend(
            path for path in _iter_files_no_follow(references_dir) if path.suffix.lower() in {".md", ".txt"}
        )
    if include_scripts:
        scripts_dir = skill_dir / "scripts"
        if scripts_dir.is_dir():
            candidates.extend(_iter_files_no_follow(scripts_dir))
    for path in candidates:
        if path.is_file():
            try:
                if path.stat().st_size > MAX_SKILL_TEXT_FILE_BYTES:
                    continue
            except OSError:
                continue
            yield path, path.read_text(encoding="utf-8", errors="replace")


def _eval_mentions_file_editing(item: dict[str, Any]) -> bool:
    text = _eval_text(item).lower()
    patterns = (
        r"\b(?:edit|modify|update|rewrite|write|create|generate|export)s?\s+(?:a\s+|an\s+|the\s+)?file\b",
        r"\b(?:edit|modify|update|rewrite|write|create|generate|export)s?\s+(?:source|code|artifact)s?\b",
        r"\b(?:file|artifact)s?\s+(?:is|are|must be|should be)?\s*(?:created|generated|written|exported|modified)\b",
        r"\boutput\s+(?:file|artifact|directory)\b",
    )
    return any(re.search(pattern, text) for pattern in patterns)


def _has_files(path: Path) -> bool:
    return path.is_dir() and any(True for _child in _iter_files_no_follow(path))


def _iter_files_no_follow(root: Path) -> Iterable[Path]:
    if root.is_symlink() or not root.is_dir():
        return
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = sorted(name for name in dirnames if not (Path(dirpath) / name).is_symlink())
        for filename in sorted(filenames):
            path = Path(dirpath) / filename
            if path.is_file():
                yield path


def _has_fixture_notes(evals_dir: Path) -> bool:
    note_paths = (
        evals_dir / "README.md",
        evals_dir / "files" / "README.md",
        evals_dir / "files" / "SOURCE.md",
    )
    return any(path.is_file() for path in note_paths)


def _read_bounded_text(path: Path) -> Optional[str]:
    if not path.is_file():
        return None
    if _is_oversized_text_file(path):
        return None
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def _is_oversized_text_file(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > MAX_SKILL_TEXT_FILE_BYTES
    except OSError:
        return False


def _has_bounded_size_exception(path: Path) -> bool:
    try:
        with path.open("rb") as stream:
            prefix = stream.read(16 * 1024)
    except OSError:
        return False
    return _has_size_exception(prefix.decode("utf-8", errors="replace"))


def _line_for_frontmatter_issue(skill_file: Path, code: str, message: str) -> Optional[int]:
    if code == "skill-frontmatter-field-required":
        match = re.search(r"field '([^']+)'", message)
        if match:
            return _line_for_field(skill_file, match.group(1))
    if code in {
        "skill-name-directory-mismatch",
        "skill-blast-radius-invalid",
        "skill-frontmatter-field-type",
        "skill-frontmatter-field-unsupported",
    }:
        for field in ("name", "blast_radius", "description", "min_flare_version", "category"):
            if field in message:
                return _line_for_field(skill_file, field)
    return 1 if skill_file.is_file() else None


def _line_for_field(skill_file: Path, field: str) -> Optional[int]:
    if not skill_file.is_file():
        return None
    if _is_oversized_text_file(skill_file):
        return 1
    prefix = f"{field}:"
    for line_no, line in enumerate(skill_file.read_text(encoding="utf-8-sig", errors="replace").splitlines(), start=1):
        if line.strip().startswith(prefix):
            return line_no
    return 1


def _has_size_exception(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in _SIZE_EXCEPTION_MARKERS)


def _skip(context: LintContext, check: str, reason: str) -> None:
    context.skipped_checks.append({"id": check, "reason": reason})


def _finding(
    lint_id: str,
    severity: str,
    path: Path,
    message: str,
    hint: str,
    *,
    line: Optional[int] = None,
    code: Optional[str] = None,
    skill: Optional[str] = None,
    global_finding: bool = False,
) -> LintFinding:
    return LintFinding(
        id=lint_id,
        severity=severity,
        file=str(path),
        line=line,
        message=message,
        hint=hint,
        code=code,
        skill=skill,
        global_finding=global_finding,
    )
