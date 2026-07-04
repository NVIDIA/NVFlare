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

import ast
import errno
import json
import os
import re
import shlex
import stat
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import cached_property, lru_cache
from pathlib import Path
from typing import Any, Iterable, Optional

try:
    from .frontmatter import (
        PUBLIC_EXEMPT_STATUS,
        SKILL_FILE_NAME,
        SkillValidationResult,
        parse_skill_frontmatter,
        should_skip_skill_dir,
        skill_metadata,
        validate_skill_dir,
    )
except ImportError:
    from frontmatter import (
        PUBLIC_EXEMPT_STATUS,
        SKILL_FILE_NAME,
        SkillValidationResult,
        parse_skill_frontmatter,
        should_skip_skill_dir,
        skill_metadata,
        validate_skill_dir,
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

_READ_MISSING = "missing"
_READ_SYMLINK = "symlink"
_READ_NOT_REGULAR = "not-regular"
_READ_TOO_LARGE = "too-large"
_READ_CHANGED = "changed-during-read"
_READ_UNREADABLE = "unreadable"

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
_NORMATIVE_RE = re.compile(
    r"\b(?:must(?:\s+not)?|never|required|prohibited|do\s+not)\b"
    r"|\b(?:ask|requires?|requiring)\b[^.;\n]{0,80}\bapproval\b"
    r"|\bwithout\b[^.;\n]{0,60}\bapproval\b"
    r"|\bfollow\b[^.;\n]{0,80}\bapproval\s+boundary\b",
    re.IGNORECASE,
)
_TABLE_NORMATIVE_RE = re.compile(r"\b(?:must(?:\s+not)?|never|prohibited|do\s+not)\b", re.IGNORECASE)
_POLICY_TOKEN_RE = re.compile(r"[a-z][a-z0-9]*")
_POLICY_NEGATIVE_MODAL_RE = re.compile(
    r"\b(?:must\s+not|must\s+avoid|never|prohibited)\b"
    r"|\bmust\s+keep\b[^.;\n]{0,80}\bread[- ]only\b"
    r"|\bmust\s+treat\b[^.;\n]{0,80}\bunverified\b"
    r"|\bdo\s+not\b",
    re.I,
)
_POLICY_REQUIRED_MODAL_RE = re.compile(r"\b(?:must|required)\b", re.I)
_POLICY_TEST_EVIDENCE_VARIABLE = "NVFLARE_POLICY_TEST_EVIDENCE"
_POLICY_POLARITIES = {"required", "prohibited"}
_POLICY_CONFLICTING_ACTIONS = (
    frozenset({"accept", "reject"}),
    frozenset({"allow", "deny"}),
    frozenset({"copy", "delet"}),
    frozenset({"download", "upload"}),
    frozenset({"enabl", "disabl"}),
    frozenset({"encrypt", "delet"}),
    frozenset({"includ", "exclud"}),
    frozenset({"install", "uninstall"}),
    frozenset({"preserv", "remov"}),
    frozenset({"read", "writ"}),
    frozenset({"start", "stop"}),
    frozenset({"submit", "cancel"}),
)
_POLICY_STOPWORDS = {
    "a",
    "an",
    "and",
    "approval",
    "as",
    "be",
    "before",
    "by",
    "do",
    "does",
    "for",
    "from",
    "in",
    "into",
    "is",
    "it",
    "keep",
    "must",
    "never",
    "not",
    "of",
    "or",
    "prohibited",
    "required",
    "rule",
    "that",
    "the",
    "their",
    "this",
    "to",
    "use",
    "when",
    "with",
}
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
_KNOWN_AGENT_FLAGS = {
    "agent": {"--format", "--schema"},
    "agent doctor": {"--format", "--kit-id", "--online", "--schema", "--startup-kit"},
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


@dataclass(frozen=True)
class _PolicyCoverageEvidence:
    text: str
    polarity: str
    source: str
    evidence_id: Optional[str] = None


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
        status = str(skill_metadata(self.metadata).get("status", "public")).strip().lower()
        return status not in PUBLIC_EXEMPT_STATUS

    @cached_property
    def validation(self) -> SkillValidationResult:
        # Computed lazily (once) so scoped runs that never consume it — e.g.
        # checks=[skill-md-size-lint] — keep the loader's bounded-read behavior
        # instead of validate_skill_dir's unbounded SKILL.md parse.
        return validate_skill_dir(self.skill_dir)

    @cached_property
    def has_helper_tests(self) -> bool:
        return _skill_has_helper_tests(self.skill_dir)


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
    root_error_codes = {"skills-root-missing", "skills-root-not-directory", "skills-root-symlink-not-allowed"}
    if not any(finding.global_finding and finding.code in root_error_codes for finding in findings):
        for check in selected:
            _LINT_FUNCTIONS[check](context)

    summary = _summary_from_severities((finding.severity for finding in context.findings), len(records))
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
        result["summary"] = _summary_from_severities(
            (finding.get("severity", FINDING_ERROR) for finding in result["findings"]),
            _matching_skill_count(records, skill_name),
        )
        result["status"] = "failed" if result["summary"]["error_count"] else "ok"
        result["passed"] = result["status"] == "ok"
    else:
        result["requested_skill"] = None
    return result


def _summary_from_severities(severities: Iterable[str], skill_count: int) -> dict[str, int]:
    severity_counts = Counter(severities)
    return {
        "skill_count": skill_count,
        "finding_count": sum(severity_counts.values()),
        "error_count": severity_counts.get(FINDING_ERROR, 0),
        "warning_count": severity_counts.get(FINDING_WARNING, 0),
        "info_count": severity_counts.get(FINDING_INFO, 0),
    }


def _finding_matches_requested_skill(finding: dict[str, Any], skill_name: str) -> bool:
    finding_skill = finding.get("skill")
    return finding_skill == skill_name or (finding_skill is None and finding.get("global") is True)


def _load_skill_records(skills_root: Path, evals_root: Path, findings: list[LintFinding]) -> list[SkillRecord]:
    if skills_root.is_symlink():
        findings.append(
            _finding(
                LINT_SKILL_FRONTMATTER,
                FINDING_ERROR,
                skills_root,
                "skills root must not be a symlink",
                "Pass --skills-root pointing directly at the repository skills/ directory.",
                code="skills-root-symlink-not-allowed",
                global_finding=True,
            )
        )
        return []
    if not skills_root.exists():
        findings.append(
            _finding(
                LINT_SKILL_FRONTMATTER,
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
                LINT_SKILL_FRONTMATTER,
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
        text = _read_bounded_text(skill_file)
        metadata = _try_parse_frontmatter(skill_file) if text is not None else {}
        text = text or ""
        skill_name = str(metadata.get("name") or child.name)
        # Eval suites live outside the shipped skill tree, one dir per skill name.
        evals_dir = evals_root / skill_name
        evals_path = evals_dir / "evals.json"
        if evals_dir.is_symlink():
            evals, evals_error = [], "eval suite directory must not be a symlink"
        else:
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
        for issue in record.validation.issues:
            context.findings.append(
                _finding(
                    LINT_SKILL_FRONTMATTER,
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
                    LINT_SKILL_FRONTMATTER,
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
        _text, read_error = _read_bounded_regular_text(record.skill_file)
        if read_error in {_READ_MISSING, _READ_UNREADABLE}:
            continue
        if read_error in {_READ_SYMLINK, _READ_NOT_REGULAR, _READ_CHANGED}:
            context.findings.append(
                _finding(
                    LINT_SKILL_MD_SIZE,
                    FINDING_ERROR,
                    record.skill_file,
                    "SKILL.md must be a stable regular file, not a symlink or special file",
                    "Replace SKILL.md with a regular file inside the skill directory.",
                    code="skill-md-unsafe-file",
                    skill=record.name,
                    line=1,
                )
            )
            continue
        if read_error == _READ_TOO_LARGE:
            if _has_bounded_size_exception(record.skill_file):
                continue
            context.findings.append(
                _finding(
                    LINT_SKILL_MD_SIZE,
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
                    LINT_SKILL_MD_SIZE,
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
                    LINT_SKILL_MD_SIZE,
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
                    LINT_SKILL_TRIGGER,
                    FINDING_ERROR,
                    record.skill_file,
                    "skill is missing trigger or use-boundary text",
                    "Add concise trigger guidance and negative boundary language to SKILL.md.",
                    code="skill-trigger-text-missing",
                    skill=record.name,
                )
            )

        if not _evals_available(
            context,
            LINT_SKILL_TRIGGER,
            record,
            "evals.json (under dev_tools/agent/skill_evals/<skill>/) is required for public skill trigger checks",
            "Add a guide-compatible evals.json under the eval root "
            "(dev_tools/agent/skill_evals/<skill>/) with positive and adjacent negative trigger evals.",
        ):
            continue
        if not any(_is_positive_eval(item, record.name) for item in record.evals):
            context.findings.append(
                _finding(
                    LINT_SKILL_TRIGGER,
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
                    LINT_SKILL_TRIGGER,
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
                LINT_SKILL_TRIGGER_OVERLAP,
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
                        LINT_SKILL_TRIGGER_OVERLAP,
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
        if not _evals_available(
            context,
            LINT_SKILL_GLOBAL_NEGATIVE,
            record,
            "evals.json (under dev_tools/agent/skill_evals/<skill>/) is required for global negative coverage",
            "Add at least one eval representing an unrelated prompt that should trigger no FLARE skill.",
        ):
            continue
        if not any(_is_global_negative_eval(item) for item in record.evals):
            context.findings.append(
                _finding(
                    LINT_SKILL_GLOBAL_NEGATIVE,
                    FINDING_ERROR,
                    record.evals_path,
                    "missing global negative eval",
                    "Add an eval for an unrelated prompt with nvflare.expected_skill set to null or 'none'.",
                    code="skill-global-negative-eval-missing",
                    skill=record.name,
                )
            )


def _lint_policy_coverage(context: LintContext) -> None:
    policy_records = _policy_records(context.records)
    evidence_by_record: dict[str, list[_PolicyCoverageEvidence]] = {}
    checklist_errors: dict[str, str] = {}
    repo_root = context.skills_root.resolve().parent

    for record in context.records:
        evidence = _behavior_coverage(record.evals)
        evidence.extend(_helper_test_coverage(record.skill_dir))
        checklist_coverage, checklist_error = _release_checklist_coverage(
            record.evals_dir / "release_checklist.json",
            record=record,
            repo_root=repo_root,
        )
        evidence.extend(checklist_coverage)
        evidence_by_record[record.name] = evidence
        if checklist_error:
            checklist_errors[record.name] = checklist_error

    reported_checklist_errors: set[str] = set()
    for record in policy_records:
        consumers = _policy_consumers(record, context.records)
        evidence_records = _unique_records([record, *consumers])
        coverage = [item for source in evidence_records for item in evidence_by_record.get(source.name, [])]

        for source in evidence_records:
            checklist_error = checklist_errors.get(source.name)
            if not checklist_error or source.name in reported_checklist_errors:
                continue
            reported_checklist_errors.add(source.name)
            context.findings.append(
                _finding(
                    LINT_SKILL_POLICY_COVERAGE,
                    FINDING_ERROR,
                    source.evals_dir / "release_checklist.json",
                    checklist_error,
                    "Use schema_version '1' and make every checklist item resolve to a concrete eval behavior or "
                    "an executable NVFLARE_POLICY_TEST_EVIDENCE marker.",
                    code="skill-policy-checklist-invalid",
                    skill=source.name,
                )
            )

        for file_path, line_no, rule in _normative_rules(record.skill_dir):
            if any(_policy_coverage_matches(rule, candidate) for candidate in coverage):
                continue
            context.findings.append(
                _finding(
                    LINT_SKILL_POLICY_COVERAGE,
                    FINDING_ERROR,
                    file_path,
                    f"normative rule has no matching measurable coverage: {_short_rule(rule)!r}",
                    "Add a polarity-matching mandatory/prohibited runtime behavior, an executable helper-test "
                    "evidence marker, or a checklist item that resolves to one of those concrete verifiers.",
                    code="skill-policy-coverage-missing",
                    skill=record.name,
                    line=line_no,
                )
            )


def _lint_process_metrics(context: LintContext) -> None:
    for record in _public_records(context.records):
        if not _evals_available(
            context,
            LINT_SKILL_PROCESS_METRIC,
            record,
            "evals.json (under dev_tools/agent/skill_evals/<skill>/) is required for process-metric coverage",
            "Add process metric contracts under nvflare.process_metrics.",
        ):
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
            for line_no, command, require_complete in _extract_nvflare_commands(text):
                message = _command_drift_message(command, require_complete=require_complete)
                if message is None:
                    continue
                context.findings.append(
                    _finding(
                        LINT_SKILL_COMMAND_DRIFT,
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
        if script_files and not record.has_helper_tests:
            context.findings.append(
                _finding(
                    LINT_SKILL_HELPER_SCRIPT,
                    FINDING_ERROR,
                    scripts_dir,
                    "helper scripts are shipped without tests",
                    "Add tests under the skill or repository test tree for each shipped helper script.",
                    code="skill-helper-tests-missing",
                    skill=record.name,
                )
            )

        for script in script_files:
            text = _read_bounded_text(script)
            if text is None:
                continue
            lowered = text.lower()
            if "promoted_to:" in lowered or "_promoted_to:" in lowered:
                context.findings.append(
                    _finding(
                        LINT_SKILL_HELPER_SCRIPT,
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
                        LINT_SKILL_HELPER_SCRIPT,
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
            _add_evals_error(context, LINT_SKILL_FIXTURE, record)
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
                        LINT_SKILL_FIXTURE,
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
                        LINT_SKILL_FIXTURE,
                        FINDING_ERROR,
                        record.evals_path,
                        f"eval '{item.get('id', '<missing>')}' describes file editing without input fixtures",
                        "Add deterministic input files under the eval root "
                        "(dev_tools/agent/skill_evals/<skill>/files/) and reference them from the eval.",
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
                            LINT_SKILL_FIXTURE,
                            FINDING_ERROR,
                            record.evals_path,
                            f"eval fixture path escapes eval suite directory: {rel_path}",
                            "Use fixture paths relative to the eval suite directory.",
                            code="skill-fixture-path-escape",
                            skill=record.name,
                        )
                    )
                    continue
                unsafe_fixture = _path_has_symlink_component(record.evals_dir, Path(str(rel_path)))
                if unsafe_fixture or not _is_regular_file(fixture_path):
                    code = (
                        "skill-fixture-file-unsafe"
                        if unsafe_fixture or fixture_path.exists() or fixture_path.is_symlink()
                        else "skill-fixture-file-missing"
                    )
                    message = (
                        f"eval fixture must be a regular non-symlink file: {rel_path}"
                        if code == "skill-fixture-file-unsafe"
                        else f"eval fixture does not exist: {rel_path}"
                    )
                    context.findings.append(
                        _finding(
                            LINT_SKILL_FIXTURE,
                            FINDING_ERROR,
                            record.evals_path,
                            message,
                            "Place deterministic fixtures under the eval root "
                            "(dev_tools/agent/skill_evals/<skill>/files/) and reference existing files.",
                            code=code,
                            skill=record.name,
                        )
                    )

        files_dir = record.evals_dir / "files"
        if _has_files(files_dir) and not _has_fixture_notes(record.evals_dir):
            context.findings.append(
                _finding(
                    LINT_SKILL_FIXTURE,
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
        for eval_dir in _iter_eval_dirs(record.skill_dir):
            context.findings.append(
                _finding(
                    LINT_SKILL_RUNTIME_BOUNDARY,
                    FINDING_ERROR,
                    eval_dir,
                    "eval suite must not live inside a shipped skill directory",
                    "Move the eval suite to the eval root (dev_tools/agent/skill_evals/<skill>/); "
                    "grading-oracle data must not ship in installed skills.",
                    code="skill-runtime-eval-dir-in-skill",
                    skill=record.name,
                )
            )
        for file_path, text in _iter_packaged_runtime_files(record.skill_dir):
            _scan_runtime_boundary(context, file_path, text, skill=record.name)


# Canonical lint registry: single source of truth for lint IDs, their run
# order, and their implementations. V1_LINT_IDS and _LINT_FUNCTIONS derive
# from it; do not maintain separate lists.
_LINT_REGISTRY = (
    (LINT_SKILL_FRONTMATTER, _lint_frontmatter),
    (LINT_SKILL_MD_SIZE, _lint_md_size),
    (LINT_SKILL_TRIGGER, _lint_trigger),
    (LINT_SKILL_TRIGGER_OVERLAP, _lint_trigger_overlap),
    (LINT_SKILL_GLOBAL_NEGATIVE, _lint_global_negative),
    (LINT_SKILL_POLICY_COVERAGE, _lint_policy_coverage),
    (LINT_SKILL_PROCESS_METRIC, _lint_process_metrics),
    (LINT_SKILL_COMMAND_DRIFT, _lint_command_drift),
    (LINT_SKILL_HELPER_SCRIPT, _lint_helper_scripts),
    (LINT_SKILL_FIXTURE, _lint_fixtures),
    (LINT_SKILL_RUNTIME_BOUNDARY, _lint_runtime_boundary),
)
V1_LINT_IDS = tuple(lint_id for lint_id, _ in _LINT_REGISTRY)
_LINT_FUNCTIONS = dict(_LINT_REGISTRY)


def _iter_eval_dirs(skill_dir: Path) -> Iterable[Path]:
    """Yield ``evals`` directories at any depth inside a skill.

    Packaging strips directories named ``evals`` at any depth, so nested eval
    content (e.g. ``references/evals/``) is silently omitted from bundles. The
    boundary lint must match that depth so authors are told to relocate it
    instead of shipping nothing. Once an excluded runtime directory is reported,
    its subtree is also outside the shipped boundary and does not need traversal.
    """
    if not skill_dir.is_dir():
        return
    excluded = _RUNTIME_BOUNDARY_EXCLUDED_DIRS - {"evals"}
    for current_dir, dir_names, _file_names in _walk_no_follow(skill_dir, excluded):
        if "evals" in dir_names:
            yield current_dir / "evals"
            dir_names.remove("evals")


def _iter_packaged_runtime_files(skill_dir: Path) -> Iterable[tuple[Path, str]]:
    """Yield decoded text files a skill ships as runtime content."""
    if not skill_dir.is_dir():
        return
    for path in _iter_files_no_follow(skill_dir, excluded_dir_names=_RUNTIME_BOUNDARY_EXCLUDED_DIRS):
        content = _read_runtime_text_file(path)
        if content is not None:
            yield path, content


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
    text, read_error = _read_bounded_regular_text(evals_path, encoding="utf-8", errors="strict")
    if read_error == _READ_MISSING:
        return [], None
    if read_error == _READ_TOO_LARGE:
        return [], f"evals.json exceeds size limit ({MAX_SKILL_TEXT_FILE_BYTES} bytes)"
    if read_error in {_READ_SYMLINK, _READ_NOT_REGULAR, _READ_CHANGED}:
        return [], "evals.json must be a stable regular file, not a symlink or special file"
    if read_error or text is None:
        return [], "evals.json could not be read as bounded UTF-8 text"
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as e:
        return [], f"failed to parse evals.json: {e}"
    if isinstance(raw, dict):
        items = raw.get("evals", [])
    elif isinstance(raw, list):
        items = raw
    else:
        return [], "evals.json must be an object with an evals list or a list"
    if not isinstance(items, list):
        return [], "evals.json field 'evals' must be a list"
    evals = [item for item in items if isinstance(item, dict)]
    if len(evals) != len(items):
        return evals, "each evals.json entry must be an object"
    return evals, None


def _add_evals_error(context: LintContext, lint_id: str, record: SkillRecord) -> None:
    context.findings.append(
        _finding(
            lint_id,
            FINDING_ERROR,
            record.evals_path,
            record.evals_error or "evals.json is invalid",
            "Use guide-compatible JSON with an evals list.",
            code="skill-evals-invalid",
            skill=record.name,
        )
    )


def _evals_available(
    context: LintContext, lint_id: str, record: SkillRecord, missing_message: str, missing_hint: str
) -> bool:
    """Report invalid or missing evals.json for one lint; True when evals are usable."""
    if record.evals_error:
        _add_evals_error(context, lint_id, record)
        return False
    if not record.evals_path.is_file():
        context.findings.append(
            _finding(
                lint_id,
                FINDING_ERROR,
                record.evals_path,
                missing_message,
                missing_hint,
                code="skill-evals-missing",
                skill=record.name,
            )
        )
        return False
    return True


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


def _policy_records(records: list[SkillRecord]) -> list[SkillRecord]:
    """Return public skills plus internal policy sources consumed by them."""

    public = _public_records(records)
    result = list(public)
    for record in records:
        if record.public:
            continue
        if any(_record_references_skill(consumer, record.name) for consumer in public):
            result.append(record)
    return result


def _policy_consumers(record: SkillRecord, records: list[SkillRecord]) -> list[SkillRecord]:
    if record.public:
        return []
    return [candidate for candidate in _public_records(records) if _record_references_skill(candidate, record.name)]


def _record_references_skill(record: SkillRecord, referenced_skill: str) -> bool:
    pattern = re.compile(rf"(?:^|[/\\]){re.escape(referenced_skill)}(?:[/\\]|$)")
    return any(pattern.search(text) for _path, text in _iter_skill_text_files(record.skill_dir))


def _unique_records(records: list[SkillRecord]) -> list[SkillRecord]:
    result = []
    seen = set()
    for record in records:
        if record.name not in seen:
            seen.add(record.name)
            result.append(record)
    return result


def _behavior_coverage(evals: list[dict[str, Any]]) -> list[_PolicyCoverageEvidence]:
    coverage: list[_PolicyCoverageEvidence] = []
    for item in evals:
        case_id = item.get("id")
        if not all(
            isinstance(value, str) and value.strip()
            for value in (case_id, item.get("prompt"), item.get("expected_output"))
        ):
            continue
        nvflare = _nvflare_ext(item)
        for key, polarity in (
            ("mandatory_behavior", "required"),
            ("prohibited_behavior", "prohibited"),
        ):
            values = nvflare.get(key, [])
            if not isinstance(values, list):
                continue
            for value in values:
                if not isinstance(value, dict):
                    continue
                behavior_id = value.get("id")
                description = value.get("description")
                if not all(isinstance(part, str) and part.strip() for part in (behavior_id, description)):
                    continue
                coverage.append(
                    _PolicyCoverageEvidence(
                        text=f"{behavior_id.strip()} {description.strip()}",
                        polarity=polarity,
                        source=f"eval:{case_id}:{key}:{behavior_id}",
                        evidence_id=behavior_id,
                    )
                )
        negative_for = nvflare.get("negative_for")
        if isinstance(negative_for, str) and negative_for.strip():
            coverage.append(
                _PolicyCoverageEvidence(
                    text=f"{case_id} {item['prompt']} {item['expected_output']}",
                    polarity="prohibited",
                    source=f"eval:{case_id}:negative_for:{negative_for}",
                    evidence_id=f"negative-for-{negative_for}",
                )
            )
    return coverage


def _helper_test_coverage(skill_dir: Path) -> list[_PolicyCoverageEvidence]:
    coverage: list[_PolicyCoverageEvidence] = []
    tests_dir = skill_dir / "tests"
    candidates = (
        [path for path in _iter_files_no_follow(tests_dir) if _is_pytest_module(path)] if tests_dir.is_dir() else []
    )
    candidates.extend(
        path for path in _iter_files_no_follow(skill_dir) if _is_pytest_module(path) and path not in candidates
    )
    for path in candidates:
        text = _read_bounded_text(path)
        if text is not None:
            coverage.extend(_policy_test_evidence(path, text))
    return coverage


def _policy_test_evidence(path: Path, text: str) -> list[_PolicyCoverageEvidence]:
    """Load explicit evidence markers that point at executable test functions.

    Arbitrary test source, comments, docstrings, and function names are not
    coverage. A module must declare ``NVFLARE_POLICY_TEST_EVIDENCE`` entries and
    each entry must resolve to a test function containing an assertion.
    """

    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return []

    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        value = node.value
        if (
            value is not None
            and any(isinstance(target, ast.Name) and target.id == "pytestmark" for target in targets)
            and _node_has_skip_marker(value)
        ):
            return []

    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_")
    }
    declarations = []
    for node in tree.body:
        value = None
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == _POLICY_TEST_EVIDENCE_VARIABLE for target in node.targets
        ):
            value = node.value
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == _POLICY_TEST_EVIDENCE_VARIABLE
        ):
            value = node.value
        if value is not None:
            try:
                declarations = ast.literal_eval(value)
            except (ValueError, TypeError, SyntaxError):
                return []
            break
    if not isinstance(declarations, list):
        return []

    result = []
    for entry in declarations:
        if not isinstance(entry, dict):
            continue
        evidence_id = entry.get("id")
        description = entry.get("description")
        polarity = entry.get("polarity")
        test_name = entry.get("test")
        if not all(isinstance(value, str) and value.strip() for value in (evidence_id, description, test_name)):
            continue
        if polarity not in _POLICY_POLARITIES:
            continue
        test_node = functions.get(test_name)
        if test_node is None or not _test_function_has_assertion(test_node):
            continue
        result.append(
            _PolicyCoverageEvidence(
                text=f"{evidence_id} {description} {test_name}",
                polarity=polarity,
                source=f"pytest:{path}:{test_name}",
                evidence_id=evidence_id,
            )
        )
    return result


def _test_function_has_assertion(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    if any(_node_has_skip_marker(decorator) for decorator in node.decorator_list):
        return False
    if any(isinstance(child, ast.Call) and _is_skip_function(child.func) for child in ast.walk(node)):
        return False

    for statement in node.body:
        if isinstance(statement, (ast.Return, ast.Raise)):
            return False
        if isinstance(statement, ast.Assert):
            return not (isinstance(statement.test, ast.Constant) and bool(statement.test.value))
        if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call):
            if _is_assertion_call(statement.value):
                return True
        if isinstance(statement, (ast.With, ast.AsyncWith)):
            if any(
                isinstance(item.context_expr, ast.Call) and _call_name(item.context_expr.func) == "raises"
                for item in statement.items
            ):
                return True
        if isinstance(statement, ast.If):
            constant = _constant_truth(statement.test)
            if constant is not None:
                branch = statement.body if constant else statement.orelse
                if _test_statements_have_direct_assertion(branch):
                    return True
    return False


def _test_statements_have_direct_assertion(statements: list[ast.stmt]) -> bool:
    wrapper = ast.FunctionDef(
        name="test_evidence",
        args=ast.arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]),
        body=statements,
        decorator_list=[],
    )
    return _test_function_has_assertion(wrapper)


def _node_has_skip_marker(node: ast.AST) -> bool:
    return any(
        (isinstance(child, ast.Call) and _is_skip_function(child.func))
        or (isinstance(child, ast.Attribute) and child.attr in {"skip", "skipif", "xfail"})
        for child in ast.walk(node)
    )


def _is_skip_function(function: ast.AST) -> bool:
    return _call_name(function) in {"skip", "skipif", "xfail"}


def _call_name(function: ast.AST) -> Optional[str]:
    if isinstance(function, ast.Name):
        return function.id
    if isinstance(function, ast.Attribute):
        return function.attr
    return None


def _is_assertion_call(call: ast.Call) -> bool:
    name = _call_name(call.func)
    if name == "raises":
        return True
    if not name or not name.startswith("assert"):
        return False
    if name == "assertTrue" and call.args and isinstance(call.args[0], ast.Constant):
        return not bool(call.args[0].value)
    if name == "assertFalse" and call.args and isinstance(call.args[0], ast.Constant):
        return bool(call.args[0].value)
    return True


def _is_pytest_module(path: Path) -> bool:
    return path.suffix == ".py" and (path.name.startswith("test_") or path.name.endswith("_test.py"))


def _release_checklist_coverage(
    path: Path,
    *,
    record: SkillRecord,
    repo_root: Path,
) -> tuple[list[_PolicyCoverageEvidence], Optional[str]]:
    text, read_error = _read_bounded_regular_text(path, encoding="utf-8", errors="strict")
    if read_error == _READ_MISSING:
        return [], None
    if read_error == _READ_TOO_LARGE:
        return [], f"release_checklist.json exceeds size limit ({MAX_SKILL_TEXT_FILE_BYTES} bytes)"
    if read_error is not None or text is None:
        return [], "release_checklist.json must be a bounded regular non-symlink UTF-8 file"
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as e:
        return [], f"failed to parse release_checklist.json: {e}"
    if not isinstance(raw, dict) or str(raw.get("schema_version")) != "1":
        return [], "release_checklist.json must be an object with schema_version '1'"
    items = raw.get("items")
    if not isinstance(items, list):
        return [], "release_checklist.json field 'items' must be a list"

    coverage: list[_PolicyCoverageEvidence] = []
    item_ids = set()
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            return [], f"release_checklist.json item {index} must be an object"
        values = [item.get("id"), item.get("description"), item.get("evidence_expected")]
        if not all(isinstance(value, str) and value.strip() for value in values):
            return [], (
                f"release_checklist.json item {index} must contain non-empty id, description, and evidence_expected"
            )
        item_id = item["id"].strip()
        if item_id in item_ids:
            return [], f"release_checklist.json contains duplicate item id {item_id!r}"
        item_ids.add(item_id)
        verification = item.get("verification")
        if not isinstance(verification, dict):
            return [], f"release_checklist.json item {index} must contain a verification object"
        evidence, error = _resolve_checklist_verification(
            item_id,
            verification,
            record=record,
            repo_root=repo_root,
        )
        if error:
            return [], f"release_checklist.json item {index} verification is invalid: {error}"
        coverage.append(evidence)
    return coverage, None


def _resolve_checklist_verification(
    checklist_id: str,
    verification: dict[str, Any],
    *,
    record: SkillRecord,
    repo_root: Path,
) -> tuple[Optional[_PolicyCoverageEvidence], Optional[str]]:
    kind = verification.get("kind")
    if kind == "eval_behavior":
        case_id = verification.get("case_id")
        category = verification.get("category")
        behavior_id = verification.get("behavior_id")
        if category not in {"mandatory_behavior", "prohibited_behavior"}:
            return None, "eval_behavior category must be mandatory_behavior or prohibited_behavior"
        if not all(isinstance(value, str) and value.strip() for value in (case_id, behavior_id)):
            return None, "eval_behavior requires non-empty case_id and behavior_id"
        candidates = [
            evidence
            for evidence in _behavior_coverage(record.evals)
            if evidence.source == f"eval:{case_id}:{category}:{behavior_id}"
        ]
        if len(candidates) != 1:
            return None, f"could not resolve exactly one {category} {behavior_id!r} in eval {case_id!r}"
        return candidates[0], None

    if kind == "pytest":
        relative_path = verification.get("path")
        evidence_id = verification.get("evidence_id")
        if not all(isinstance(value, str) and value.strip() for value in (relative_path, evidence_id)):
            return None, "pytest verification requires non-empty path and evidence_id"
        relative = Path(relative_path)
        if relative.is_absolute() or ".." in relative.parts:
            return None, "pytest path must stay relative to the repository root"
        if not relative.parts or relative.parts[0] != "tests" or not _is_pytest_module(relative):
            return None, "pytest path must name a collected test_*.py or *_test.py module under tests/"
        test_path = repo_root / relative
        if _path_has_symlink_component(repo_root, relative):
            return None, "pytest path must not contain symlinks"
        test_text, read_error = _read_bounded_regular_text(test_path, encoding="utf-8", errors="strict")
        if read_error is not None or test_text is None:
            return None, "pytest path must be a bounded regular UTF-8 file"
        candidates = [
            evidence for evidence in _policy_test_evidence(test_path, test_text) if evidence.evidence_id == evidence_id
        ]
        if len(candidates) != 1:
            return None, f"could not resolve exactly one executable pytest evidence id {evidence_id!r}"
        return candidates[0], None

    return None, f"unsupported verification kind {kind!r} for checklist item {checklist_id!r}"


def _normative_rules(skill_dir: Path) -> Iterable[tuple[Path, int, str]]:
    for file_path, text in _iter_skill_text_files(skill_dir):
        lines = text.splitlines()
        index = 0
        in_fence = False
        while index < len(lines):
            stripped = lines[index].strip()
            if stripped.startswith("```") or stripped.startswith("~~~"):
                in_fence = not in_fence
                index += 1
                continue
            if in_fence or not stripped or stripped.startswith("#"):
                index += 1
                continue

            line_no = index + 1
            table_line = stripped.startswith("|")
            block = [_markdown_table_text(stripped) if table_line else stripped]
            index += 1
            while not table_line and index < len(lines):
                continuation = lines[index].strip()
                if (
                    not continuation
                    or continuation.startswith(("#", "|", "```", "~~~"))
                    or re.match(r"^(?:[-*+]\s+|\d+[.)]\s+)", continuation)
                ):
                    break
                block.append(continuation)
                index += 1
            block_text = " ".join(block)
            search_text = _normative_search_text(block_text)
            if not (_TABLE_NORMATIVE_RE if table_line else _NORMATIVE_RE).search(search_text):
                continue
            for rule in _split_normative_clauses(block_text):
                if _is_policy_maintenance_directive(rule):
                    continue
                yield file_path, line_no, rule


def _markdown_table_text(line: str) -> str:
    cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
    if cells and all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells):
        return ""
    return " ".join(cell for cell in cells if cell)


def _is_policy_maintenance_directive(rule: str) -> bool:
    lowered = _normative_search_text(rule).lower()
    return bool(
        re.search(r"\bdo\s+not\s+(?:encode|maintain|restate)\b", lowered)
        and re.search(r"\b(?:this\s+reference|here)\b", lowered)
    )


def _split_normative_clauses(block: str) -> list[str]:
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.;])\s+", block)
        if _NORMATIVE_RE.search(_normative_search_text(sentence))
    ]


def _normative_search_text(text: str) -> str:
    """Remove quoted labels/code so policy words there do not create rules."""

    return re.sub(r"`[^`]*`|\"[^\"]*\"|'[^']*'", " ", text)


def _policy_coverage_matches(rule: str, candidate: _PolicyCoverageEvidence) -> bool:
    rule_polarity = _policy_rule_polarity(rule)
    if rule_polarity != "neutral" and rule_polarity != candidate.polarity:
        return False
    rule_tokens = _policy_tokens(rule)
    candidate_tokens = _policy_tokens(candidate.text)
    if not rule_tokens or not candidate_tokens:
        return False
    if _policy_actions_conflict(rule_tokens, candidate_tokens):
        return False
    shared = rule_tokens.intersection(candidate_tokens)
    required_shared = 1 if len(rule_tokens) == 1 else 2
    return len(shared) >= 3 or (
        len(shared) >= required_shared and len(shared) / min(len(rule_tokens), len(candidate_tokens)) >= 0.4
    )


def _policy_actions_conflict(rule_tokens: set[str], candidate_tokens: set[str]) -> bool:
    return any(
        bool(actions & rule_tokens)
        and bool(actions & candidate_tokens)
        and (actions & rule_tokens) != (actions & candidate_tokens)
        for actions in _POLICY_CONFLICTING_ACTIONS
    )


def _policy_rule_polarity(rule: str) -> str:
    negative = _POLICY_NEGATIVE_MODAL_RE.search(rule)
    required = _POLICY_REQUIRED_MODAL_RE.search(rule)
    if negative:
        return "prohibited"
    if required:
        return "required"
    # Approval-only clauses often describe a gate rather than one direction of
    # behavior. They still need concrete evidence, but either a required gate or
    # a prohibited unapproved action may implement it.
    return "neutral"


def _policy_tokens(text: str) -> set[str]:
    normalized = text.lower().replace("read-only", "read only no mutation").replace("_", " ").replace("-", " ")
    return {
        stemmed
        for token in _POLICY_TOKEN_RE.findall(normalized)
        if token not in _POLICY_STOPWORDS
        for stemmed in [_policy_stem(token)]
        if len(stemmed) >= 3
    }


def _policy_stem(token: str) -> str:
    if token == "uses":
        return "use"
    for suffix, replacement in (
        ("ation", ""),
        ("ments", ""),
        ("ment", ""),
        ("ing", ""),
        ("ied", "y"),
        ("ies", "y"),
        ("ed", ""),
        ("es", ""),
        ("s", ""),
    ):
        if token.endswith(suffix) and len(token) - len(suffix) >= 4:
            token = token[: -len(suffix)] + replacement
            break
    if token.endswith("e") and len(token) > 4:
        token = token[:-1]
    return token


def _short_rule(rule: str, max_length: int = 120) -> str:
    return rule if len(rule) <= max_length else rule[: max_length - 3].rstrip() + "..."


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


def _records_overlap(left: SkillRecord, right: SkillRecord, token_cache: dict[str, set[str]]) -> bool:
    left_tokens = token_cache[left.name]
    right_tokens = token_cache[right.name]
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


def _extract_nvflare_commands(text: str) -> list[tuple[int, str, bool]]:
    commands = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        matches = list(_BACKTICK_NVFLARE_RE.finditer(line))
        if matches:
            commands.extend(
                (line_no, match.group(1).strip(), _command_is_invocation(line, match.start(), match.end()))
                for match in matches
            )
            continue
        index = line.find("nvflare ")
        if index == -1:
            continue
        commands.append((line_no, _trim_command(line[index:]), _command_is_invocation(line, index, len(line))))
    return commands


def _command_is_invocation(line: str, command_start: int, command_end: int) -> bool:
    prefix = line[:command_start].strip().lower()
    if not prefix or re.fullmatch(r"(?:[-*+]\s*|\d+[.)]\s*)", prefix):
        suffix = line[command_end:].strip()
        return not re.search(r"[a-z]", suffix, re.IGNORECASE)
    return bool(
        re.search(
            r"\b(?:run|execute|invoke)(?:\s+(?:this|the|following|command|it)){0,4}\s*:?\s*$",
            prefix,
        )
    )


def _trim_command(text: str) -> str:
    text = text.strip()
    for separator in ("&&", "|", ";"):
        if separator in text:
            text = text.split(separator, 1)[0].strip()
    return text


def _command_drift_message(command: str, *, require_complete: bool = True) -> Optional[str]:
    tokens, token_error = _command_tokens(command)
    if not tokens or tokens[0] != "nvflare":
        return token_error
    try:
        parser = _nvflare_cli_parser_spec()
    except Exception as e:
        return f"could not load the NVFLARE CLI parser while validating '{command}': {type(e).__name__}: {e}"
    parser_error = _validate_command_against_parser(tokens, command, parser, require_complete=require_complete)
    return parser_error or token_error


_CLI_PARSER_SPEC_SCRIPT = r"""
import argparse
import json

from nvflare.cli import parse_args


def encode_nargs(value):
    if value is None or isinstance(value, int):
        return value
    return str(value)


def action_spec(action):
    choices = action.choices
    if choices is not None:
        try:
            choices = [str(value) for value in choices]
        except TypeError:
            choices = None
    return {
        "dest": action.dest,
        "nargs": encode_nargs(action.nargs),
        "required": bool(getattr(action, "required", False)),
        "choices": choices,
        "action_type": type(action).__name__,
        "value_type": action.type.__name__ if action.type in (int, float, str) else None,
    }


def parser_spec(parser):
    options = {}
    positionals = []
    subcommands = {}
    subcommands_required = False
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            subcommands_required = bool(getattr(action, "required", False))
            subcommands = {name: parser_spec(child) for name, child in action.choices.items()}
            continue
        spec = action_spec(action)
        if action.option_strings:
            for option in action.option_strings:
                options[option] = spec
        else:
            positionals.append(spec)
    mutex_groups = []
    for group in parser._mutually_exclusive_groups:
        destinations = [action.dest for action in group._group_actions]
        mutex_groups.append({"required": bool(group.required), "destinations": destinations})
    return {
        "options": options,
        "positionals": positionals,
        "subcommands": subcommands,
        "subcommands_required": subcommands_required,
        "mutex_groups": mutex_groups,
    }


root, _args, _subcommands = parse_args("nvflare")
print(json.dumps(parser_spec(root), sort_keys=True))
"""


@lru_cache(maxsize=1)
def _nvflare_cli_parser_spec() -> dict[str, Any]:
    """Serialize the real parser in an isolated process without touching ``sys.argv``.

    ``nvflare.cli.parse_args`` currently takes its argv only from process-global
    state. The child starts with an empty command line naturally, builds the real
    parser, and returns action metadata. No handler runs; the lint process never
    mutates its own argv or CLI parser globals.
    """

    completed = subprocess.run(
        [sys.executable, "-c", _CLI_PARSER_SPEC_SCRIPT],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or f"exit code {completed.returncode}"
        raise RuntimeError(f"NVFLARE CLI parser subprocess failed: {detail}")
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("NVFLARE CLI parser subprocess produced no schema")
    try:
        spec = json.loads(lines[-1])
    except json.JSONDecodeError as e:
        raise RuntimeError("NVFLARE CLI parser subprocess produced invalid JSON") from e
    if not isinstance(spec, dict):
        raise RuntimeError("NVFLARE CLI parser subprocess produced an invalid schema")
    return spec


def _validate_command_against_parser(
    tokens: list[str],
    command: str,
    root_parser: dict[str, Any],
    *,
    require_complete: bool,
) -> Optional[str]:
    parser = root_parser
    command_path: list[str] = []
    global_options = root_parser.get("options", {})
    contexts: list[tuple[tuple[str, ...], dict[str, Any]]] = [((), root_parser)]
    seen_options: dict[tuple[str, ...], set[str]] = defaultdict(set)
    positional_tokens: list[str] = []
    schema_mode = any(_split_option(token)[0] == "--schema" for token in tokens[1:] if token.startswith("-"))
    allow_incomplete = schema_mode or not require_complete
    index = 1

    while index < len(tokens):
        token = tokens[index]
        if token.startswith("-"):
            flag, inline_value = _split_option(token)
            local_options = parser.get("options", {})
            action = local_options.get(flag) or global_options.get(flag)
            if action is None:
                display_path = " ".join(["nvflare", *command_path])
                return f"unknown flag '{flag}' for '{display_path}' in '{command}'"
            if action.get("action_type") == "_HelpAction":
                return None
            owner = tuple(command_path) if flag in local_options else ()
            seen_options[owner].add(action.get("dest", flag))
            index, error = _consume_option_values(
                tokens,
                index,
                action,
                inline_value,
                parser,
                root_parser,
                command,
                flag,
            )
            if error:
                return error
            continue

        subcommands = parser.get("subcommands", {})
        if subcommands:
            positional_error = _validate_positionals(
                parser,
                positional_tokens,
                command_path,
                command,
                allow_incomplete,
            )
            if positional_error:
                return positional_error
            positional_tokens = []
            if token not in subcommands:
                if not command_path:
                    return f"unknown nvflare command root '{token}' in '{command}'"
                if command_path == ["agent"]:
                    return f"unknown nvflare agent command '{token}' in '{command}'"
                if command_path == ["agent", "skills"]:
                    return f"unknown nvflare agent skills command '{token}' in '{command}'"
                display_path = " ".join(["nvflare", *command_path])
                return f"unknown subcommand '{token}' for '{display_path}' in '{command}'"
            parser = subcommands[token]
            command_path.append(token)
            contexts.append((tuple(command_path), parser))
            index += 1
            continue

        positional_tokens.append(token)
        index += 1

    if parser.get("subcommands_required") and not allow_incomplete:
        display_path = " ".join(["nvflare", *command_path])
        return f"missing required subcommand for '{display_path}' in '{command}'"
    positional_error = _validate_positionals(
        parser,
        positional_tokens,
        command_path,
        command,
        allow_incomplete,
    )
    if positional_error:
        return positional_error
    return _validate_required_options(contexts, seen_options, command, allow_incomplete)


def _split_option(token: str) -> tuple[str, Optional[str]]:
    if token.startswith("--") and "=" in token:
        return tuple(token.split("=", 1))
    return token, None


def _consume_option_values(
    tokens: list[str],
    index: int,
    action: dict[str, Any],
    inline_value: Optional[str],
    parser: dict[str, Any],
    root_parser: dict[str, Any],
    command: str,
    flag: str,
) -> tuple[int, Optional[str]]:
    nargs = action.get("nargs")
    if nargs == 0:
        if inline_value is not None:
            return index + 1, f"flag '{flag}' does not accept a value in '{command}'"
        return index + 1, None

    values = [inline_value] if inline_value is not None else []
    cursor = index + 1
    if isinstance(nargs, int) or nargs in (None, 1):
        required = nargs if isinstance(nargs, int) else 1
        while len(values) < required and cursor < len(tokens):
            if _token_starts_option(tokens[cursor], parser, root_parser):
                break
            values.append(tokens[cursor])
            cursor += 1
        if len(values) != required:
            return cursor, f"flag '{flag}' requires {required} value(s) in '{command}'"
    elif nargs == "?":
        if not values and cursor < len(tokens) and not _token_starts_option(tokens[cursor], parser, root_parser):
            values.append(tokens[cursor])
            cursor += 1
    else:
        minimum = 1 if nargs == "+" else 0
        while cursor < len(tokens) and not _token_starts_option(tokens[cursor], parser, root_parser):
            values.append(tokens[cursor])
            cursor += 1
        if len(values) < minimum:
            return cursor, f"flag '{flag}' requires at least one value in '{command}'"

    choices = action.get("choices")
    value_type = action.get("value_type")
    invalid_type = next((value for value in values if not _value_matches_cli_type(value, value_type)), None)
    if invalid_type is not None:
        return cursor, f"invalid value {invalid_type!r} for flag '{flag}': expected {value_type} in '{command}'"
    if choices is not None:
        invalid = next((value for value in values if str(value) not in choices), None)
        if invalid is not None:
            return cursor, f"invalid value {invalid!r} for flag '{flag}' in '{command}'"
    return cursor, None


def _value_matches_cli_type(value: Any, value_type: Optional[str]) -> bool:
    if value_type in (None, "str"):
        return True
    converter = {"int": int, "float": float}.get(value_type)
    if converter is None:
        return True
    try:
        converter(value)
    except (TypeError, ValueError, OverflowError):
        return False
    return True


def _token_starts_option(token: str, parser: dict[str, Any], root_parser: dict[str, Any]) -> bool:
    if not token.startswith("-"):
        return False
    flag, _inline = _split_option(token)
    return flag in parser.get("options", {}) or flag in root_parser.get("options", {}) or token.startswith("--")


def _validate_positionals(
    parser: dict[str, Any],
    values: list[str],
    command_path: list[str],
    command: str,
    schema_mode: bool,
) -> Optional[str]:
    actions = parser.get("positionals", [])
    if schema_mode:
        actions = [dict(action, required=False, nargs="?") for action in actions]
    if _positionals_match(actions, values):
        return None

    display_path = " ".join(["nvflare", *command_path])
    minimum = sum(_positional_minimum(action) for action in actions)
    maximum_values = [_positional_maximum(action) for action in actions]
    maximum = None if any(value is None for value in maximum_values) else sum(maximum_values)
    if len(values) < minimum:
        missing = next(
            (action.get("dest", "value") for action in actions if _positional_minimum(action) > 0),
            "value",
        )
        return f"missing required positional argument '{missing}' for '{display_path}' in '{command}'"
    if maximum is not None and len(values) > maximum:
        return f"unexpected positional argument {values[maximum]!r} for '{display_path}' in '{command}'"
    return f"invalid positional arguments for '{display_path}' in '{command}'"


def _positionals_match(actions: list[dict[str, Any]], values: list[str]) -> bool:
    @lru_cache(maxsize=None)
    def matches(action_index: int, value_index: int) -> bool:
        if action_index == len(actions):
            return value_index == len(values)
        action = actions[action_index]
        minimum = _positional_minimum(action)
        maximum = _positional_maximum(action)
        available = len(values) - value_index
        upper = available if maximum is None else min(maximum, available)
        for count in range(minimum, upper + 1):
            selected = values[value_index : value_index + count]
            choices = action.get("choices")
            if choices is not None and any(value not in choices for value in selected):
                continue
            if matches(action_index + 1, value_index + count):
                return True
        return False

    return matches(0, 0)


def _positional_minimum(action: dict[str, Any]) -> int:
    nargs = action.get("nargs")
    if isinstance(nargs, int):
        return nargs
    if nargs in (None, 1):
        return 1
    return 1 if nargs == "+" else 0


def _positional_maximum(action: dict[str, Any]) -> Optional[int]:
    nargs = action.get("nargs")
    if isinstance(nargs, int):
        return nargs
    if nargs in (None, 1, "?"):
        return 1
    return None


def _validate_required_options(
    contexts: list[tuple[tuple[str, ...], dict[str, Any]]],
    seen_options: dict[tuple[str, ...], set[str]],
    command: str,
    schema_mode: bool,
) -> Optional[str]:
    for path, parser in contexts:
        seen = seen_options.get(path, set())
        if not schema_mode:
            required = {action.get("dest") for action in parser.get("options", {}).values() if action.get("required")}
            missing = sorted(value for value in required if value and value not in seen)
            if missing:
                display_path = " ".join(["nvflare", *path])
                return f"missing required option '{missing[0]}' for '{display_path}' in '{command}'"
        for group in parser.get("mutex_groups", []):
            selected = set(group.get("destinations", [])) & seen
            if len(selected) > 1:
                return f"mutually exclusive options {sorted(selected)} are combined in '{command}'"
            if group.get("required") and not selected and not schema_mode:
                display_path = " ".join(["nvflare", *path])
                return f"one mutually exclusive option is required for '{display_path}' in '{command}'"
    return None


def _command_tokens(command: str) -> tuple[list[str], Optional[str]]:
    try:
        tokens = shlex.split(command)
    except ValueError as e:
        return [], f"could not parse nvflare command {command!r}: {e}"
    try:
        start = tokens.index("nvflare")
    except ValueError:
        return [], None
    command_tokens = []
    for token in tokens[start:]:
        if token in {"&&", "|", ";"}:
            break
        if not _safe_command_token(token):
            return command_tokens, f"unsupported shell token {token!r} in nvflare command '{command}'"
        command_tokens.append(token)
    return command_tokens, None


def _safe_command_token(token: str) -> bool:
    # ``shlex`` tokenization does not expand shell variables/command
    # substitutions. Reject dollar syntax explicitly, including when attached
    # to an otherwise-valid ``--option=value`` token.
    return "$" not in token and bool(_SAFE_COMMAND_TOKEN_RE.match(token))


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
        text = _read_bounded_text(path)
        if text is not None:
            yield path, text


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


def _walk_no_follow(
    root: Path, excluded_dir_names: Iterable[str] = frozenset()
) -> Iterable[tuple[Path, list[str], list[str]]]:
    """Deterministic os.walk that never follows symlinks and prunes excluded dirs.

    Yields ``(current_dir_path, dir_names, file_names)`` with both name lists
    sorted; ``dir_names`` is the live pruned list, so callers may remove entries
    to stop descent into those directories.
    """
    excluded_dir_names = set(excluded_dir_names)
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        current_dir = Path(dirpath)
        dirnames[:] = sorted(
            name for name in dirnames if name not in excluded_dir_names and not (current_dir / name).is_symlink()
        )
        yield current_dir, dirnames, sorted(filenames)


def _iter_files_no_follow(root: Path, *, excluded_dir_names: Iterable[str] = ()) -> Iterable[Path]:
    if root.is_symlink() or not root.is_dir():
        return
    for current_dir, _dir_names, file_names in _walk_no_follow(root, excluded_dir_names):
        for filename in file_names:
            path = current_dir / filename
            if _is_regular_file(path):
                yield path


def _has_fixture_notes(evals_dir: Path) -> bool:
    note_paths = (
        evals_dir / "README.md",
        evals_dir / "files" / "README.md",
        evals_dir / "files" / "SOURCE.md",
    )
    return any(_is_regular_file(path) for path in note_paths)


def _path_has_symlink_component(base: Path, relative_path: Path) -> bool:
    current = base
    if current.is_symlink():
        return True
    for part in relative_path.parts:
        if part in {"", ".", ".."}:
            continue
        current = current / part
        if current.is_symlink():
            return True
    return False


def _is_regular_file(path: Path) -> bool:
    """Check the directory entry itself, never a symlink target."""

    try:
        return stat.S_ISREG(path.lstat().st_mode)
    except OSError:
        return False


def _read_bounded_regular_text(
    path: Path,
    *,
    max_bytes: int = MAX_SKILL_TEXT_FILE_BYTES,
    encoding: str = "utf-8",
    errors: str = "replace",
) -> tuple[Optional[str], Optional[str]]:
    data, error = _read_bounded_regular_bytes(path, max_bytes=max_bytes)
    if error is not None or data is None:
        return None, error
    try:
        return data.decode(encoding, errors=errors), None
    except UnicodeDecodeError:
        return None, _READ_UNREADABLE


def _read_bounded_regular_bytes(
    path: Path,
    *,
    max_bytes: int,
    allow_truncation: bool = False,
) -> tuple[Optional[bytes], Optional[str]]:
    """Read a stable regular file through one bounded, no-follow descriptor.

    ``O_NONBLOCK`` prevents a malicious FIFO from hanging before ``fstat`` can
    reject it. ``O_NOFOLLOW`` rejects a final symlink, and comparing the initial
    ``lstat`` identity with ``fstat`` closes the size-check/open replacement
    race. The descriptor read is capped even when a pseudo-file reports size 0.
    """

    try:
        before = path.lstat()
    except FileNotFoundError:
        return None, _READ_MISSING
    except OSError:
        return None, _READ_UNREADABLE
    if stat.S_ISLNK(before.st_mode):
        return None, _READ_SYMLINK
    if not stat.S_ISREG(before.st_mode):
        return None, _READ_NOT_REGULAR
    if before.st_size > max_bytes and not allow_truncation:
        return None, _READ_TOO_LARGE

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as e:
        if e.errno == errno.ENOENT:
            return None, _READ_MISSING
        if e.errno in {errno.ELOOP, getattr(errno, "EMLINK", errno.ELOOP)}:
            return None, _READ_SYMLINK
        return None, _READ_UNREADABLE

    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            return None, _READ_NOT_REGULAR
        if (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino):
            return None, _READ_CHANGED
        if opened.st_size > max_bytes and not allow_truncation:
            return None, _READ_TOO_LARGE

        read_limit = max_bytes if allow_truncation else max_bytes + 1
        chunks: list[bytes] = []
        bytes_read = 0
        while bytes_read < read_limit:
            chunk = os.read(descriptor, min(64 * 1024, read_limit - bytes_read))
            if not chunk:
                break
            chunks.append(chunk)
            bytes_read += len(chunk)
        data = b"".join(chunks)
        if not allow_truncation and len(data) > max_bytes:
            return None, _READ_TOO_LARGE
        return data, None
    except OSError:
        return None, _READ_UNREADABLE
    finally:
        os.close(descriptor)


def _read_bounded_text(path: Path) -> Optional[str]:
    text, error = _read_bounded_regular_text(path)
    return text if error is None else None


def _is_oversized_text_file(path: Path) -> bool:
    try:
        file_stat = path.lstat()
        return stat.S_ISREG(file_stat.st_mode) and file_stat.st_size > MAX_SKILL_TEXT_FILE_BYTES
    except OSError:
        return False


def _has_bounded_size_exception(path: Path) -> bool:
    prefix, error = _read_bounded_regular_bytes(path, max_bytes=16 * 1024, allow_truncation=True)
    if error is not None or prefix is None:
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
    return 1 if _is_regular_file(skill_file) else None


def _line_for_field(skill_file: Path, field: str) -> Optional[int]:
    text, error = _read_bounded_regular_text(skill_file, encoding="utf-8-sig", errors="replace")
    if error == _READ_MISSING:
        return None
    if error is not None or text is None:
        return 1
    prefix = f"{field}:"
    for line_no, line in enumerate(text.splitlines(), start=1):
        if line.strip().startswith(prefix):
            return line_no
    return 1


def _has_size_exception(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in _SIZE_EXCEPTION_MARKERS)


def _skip(context: LintContext, check: str, reason: str) -> None:
    context.skipped_checks.append({"id": check, "reason": reason})


def _finding(lint_id: str, severity: str, path: Path, message: str, hint: str, **kwargs: Any) -> LintFinding:
    """Build a LintFinding, converting the path to the string ``file`` field."""
    return LintFinding(id=lint_id, severity=severity, file=str(path), message=message, hint=hint, **kwargs)
