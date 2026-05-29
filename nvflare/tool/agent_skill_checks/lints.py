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

"""Deterministic v1 admission lints for NVFLARE-owned agent skills."""

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

from nvflare.tool.agent_skill_checks.frontmatter import SKILL_FILE_NAME, parse_skill_frontmatter, validate_skill_dir

V1_LINT_IDS = (
    "skill-frontmatter-lint",
    "skill-md-size-lint",
    "skill-trigger-lint",
    "skill-trigger-overlap-lint",
    "skill-catalog-category-lint",
    "skill-global-negative-lint",
    "skill-policy-coverage-lint",
    "skill-command-drift-lint",
    "skill-helper-script-lint",
    "skill-fixture-lint",
    "agent-doc-crosslink-lint",
)

LINT_SKILL_FRONTMATTER = "skill-frontmatter-lint"
LINT_SKILL_MD_SIZE = "skill-md-size-lint"
LINT_SKILL_TRIGGER = "skill-trigger-lint"
LINT_SKILL_TRIGGER_OVERLAP = "skill-trigger-overlap-lint"
LINT_SKILL_CATALOG_CATEGORY = "skill-catalog-category-lint"
LINT_SKILL_GLOBAL_NEGATIVE = "skill-global-negative-lint"
LINT_SKILL_POLICY_COVERAGE = "skill-policy-coverage-lint"
LINT_SKILL_COMMAND_DRIFT = "skill-command-drift-lint"
LINT_SKILL_HELPER_SCRIPT = "skill-helper-script-lint"
LINT_SKILL_FIXTURE = "skill-fixture-lint"
LINT_AGENT_DOC_CROSSLINK = "agent-doc-crosslink-lint"

FINDING_ERROR = "error"
FINDING_WARNING = "warning"
FINDING_INFO = "info"
SKILL_MD_MAX_LINES = 200
SKILL_MD_ADVISORY_WORDS = 2000

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
_MARKDOWN_LINK_RE = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
_BACKTICK_NVFLARE_RE = re.compile(r"`(nvflare(?:\s+[^`]+)?)`")
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
    "agent doctor": {"--format", "--online", "--schema", "--startup-kit", "--project", "--org"},
    "agent info": {"--format", "--schema"},
    "agent inspect": {"--format", "--redact", "--schema"},
    "agent skills": {"--format", "--schema"},
    "agent skills install": {"--agent", "--dry-run", "--format", "--schema", "--skill", "--target"},
    "agent skills list": {"--agent", "--format", "--schema", "--target"},
}
_DOC_FILES = (
    "agent_implementation_plan.md",
    "agent_integration.md",
    "agent_skill_authoring.md",
    "agent_skill_evaluation.md",
    "agent_skills_deferred_roadmap.md",
)


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
    evals_path: Path
    evals_error: Optional[str]

    @property
    def public(self) -> bool:
        status = str(self.metadata.get("status", "public")).strip().lower()
        return status not in _PUBLIC_EXEMPT_STATUS


@dataclass
class LintContext:
    skills_root: Path
    docs_root: Optional[Path]
    max_skill_md_lines: int
    records: list[SkillRecord]
    findings: list[LintFinding]
    skipped_checks: list[dict[str, str]]


def run_v1_lints(
    skills_root: Path | str = "skills",
    *,
    docs_root: Path | str | None = None,
    checks: Optional[Iterable[str]] = None,
    max_skill_md_lines: int = SKILL_MD_MAX_LINES,
) -> dict[str, Any]:
    """Run deterministic v1 admission lints and return structured findings."""
    selected = tuple(checks or V1_LINT_IDS)
    unknown = sorted(set(selected).difference(V1_LINT_IDS))
    if unknown:
        raise ValueError(f"unknown agent skill lint check(s): {', '.join(unknown)}")

    root = Path(skills_root)
    resolved_docs_root = _resolve_docs_root(docs_root)
    findings: list[LintFinding] = []
    records = _load_skill_records(root, findings)
    context = LintContext(
        skills_root=root,
        docs_root=resolved_docs_root,
        max_skill_md_lines=max_skill_md_lines,
        records=records,
        findings=findings,
        skipped_checks=[],
    )

    lint_functions = {
        "skill-frontmatter-lint": _lint_frontmatter,
        "skill-md-size-lint": _lint_md_size,
        "skill-trigger-lint": _lint_trigger,
        "skill-trigger-overlap-lint": _lint_trigger_overlap,
        "skill-catalog-category-lint": _lint_catalog_category,
        "skill-global-negative-lint": _lint_global_negative,
        "skill-policy-coverage-lint": _lint_policy_coverage,
        "skill-command-drift-lint": _lint_command_drift,
        "skill-helper-script-lint": _lint_helper_scripts,
        "skill-fixture-lint": _lint_fixtures,
        "agent-doc-crosslink-lint": _lint_doc_crosslinks,
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
        "docs_root": str(resolved_docs_root) if resolved_docs_root is not None else None,
        "checks": list(selected),
        "skipped_checks": context.skipped_checks,
        "summary": summary,
        "findings": [finding.as_dict() for finding in context.findings],
    }


def validate_skills(
    skills_root: Path | str = "skills",
    *,
    skill_name: Optional[str] = None,
    docs_root: Path | str | None = None,
    max_skill_md_lines: int = SKILL_MD_MAX_LINES,
) -> dict[str, Any]:
    """Compatibility wrapper for callers that validate one skill source root."""
    result = run_v1_lints(skills_root, docs_root=docs_root, max_skill_md_lines=max_skill_md_lines)

    if skill_name is not None:
        result["requested_skill"] = skill_name
        result["findings"] = [finding for finding in result["findings"] if finding.get("skill") in {None, skill_name}]
        result["summary"] = _summary_from_findings(
            result["findings"], _matching_skill_count(Path(skills_root), skill_name)
        )
        result["status"] = "failed" if result["summary"]["error_count"] else "ok"
        result["passed"] = result["status"] == "ok"
    else:
        result["requested_skill"] = None
    return result


def _summary_from_findings(findings: list[dict[str, Any]], skill_count: int) -> dict[str, int]:
    severity_counts = Counter(finding.get("severity", FINDING_ERROR) for finding in findings)
    return _summary_from_counts(severity_counts, len(findings), skill_count)


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
        "error": severity_counts.get(FINDING_ERROR, 0),
        "warning": severity_counts.get(FINDING_WARNING, 0),
        "info": severity_counts.get(FINDING_INFO, 0),
    }


def _load_skill_records(skills_root: Path, findings: list[LintFinding]) -> list[SkillRecord]:
    if not skills_root.exists():
        findings.append(
            _finding(
                "skill-frontmatter-lint",
                FINDING_ERROR,
                skills_root,
                "skills root does not exist",
                "Pass --skills-root pointing at the repository skills/ directory.",
                code="skills-root-missing",
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
            )
        )
        return []

    records = []
    for child in sorted(skills_root.iterdir(), key=lambda p: p.name):
        if child.name.startswith(".") or not child.is_dir():
            continue
        skill_file = child / SKILL_FILE_NAME
        text = skill_file.read_text(encoding="utf-8-sig") if skill_file.is_file() else ""
        metadata = _try_parse_frontmatter(skill_file) if skill_file.is_file() else {}
        evals_path = child / "evals" / "evals.json"
        evals, evals_error = _load_evals(evals_path)
        records.append(
            SkillRecord(
                name=str(metadata.get("name") or child.name),
                skill_dir=child,
                skill_file=skill_file,
                metadata=metadata,
                text=text,
                body=_skill_body(text),
                evals=evals,
                evals_path=evals_path,
                evals_error=evals_error,
            )
        )
    return records


def _matching_skill_count(skills_root: Path, skill_name: str) -> int:
    return sum(1 for record in _load_skill_records(skills_root, []) if record.name == skill_name)


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
    if context.docs_root is None or not context.docs_root.is_dir():
        _skip(context, "skill-trigger-overlap-lint", "docs root is not available")
        return

    category_map = _category_map(context)
    if not category_map:
        _skip(context, "skill-trigger-overlap-lint", "catalog categories are not available")
        return

    grouped: dict[str, list[SkillRecord]] = defaultdict(list)
    for record in _public_records(context.records):
        category = category_map.get(record.name)
        if category:
            grouped[category].append(record)

    for category, records in grouped.items():
        for i, left in enumerate(records):
            for right in records[i + 1 :]:
                if not _records_overlap(left, right):
                    continue
                if _has_boundary_text(left) and _has_boundary_text(right) and _has_adjacent_negative_pair(left, right):
                    continue
                context.findings.append(
                    _finding(
                        "skill-trigger-overlap-lint",
                        FINDING_ERROR,
                        left.skill_file,
                        f"same-category skills '{left.name}' and '{right.name}' have overlapping trigger language",
                        "Add use/do-not-use boundaries and adjacent negative evals covering the overlap.",
                        code="skill-trigger-overlap",
                        skill=left.name,
                    )
                )


def _lint_catalog_category(context: LintContext) -> None:
    docs_root = context.docs_root
    if docs_root is None or not docs_root.is_dir():
        _skip(context, "skill-catalog-category-lint", "docs root is not available")
        return

    product = _parse_product_catalog(docs_root / "agent_integration.md")
    conversion = _parse_conversion_table(docs_root / "agent_skill_authoring.md")
    if not product:
        context.findings.append(
            _finding(
                "skill-catalog-category-lint",
                FINDING_ERROR,
                docs_root / "agent_integration.md",
                "product skill catalog could not be parsed",
                "Keep the catalog markdown table headed by Category, Skill, Tier, and Purpose.",
                code="skill-catalog-unparseable",
            )
        )
        return

    for record in _public_records(context.records):
        if record.name not in product:
            context.findings.append(
                _finding(
                    "skill-catalog-category-lint",
                    FINDING_ERROR,
                    record.skill_file,
                    f"public skill '{record.name}' is missing from the product skill catalog",
                    "Add the skill to agent_integration.md or mark it draft/internal.",
                    code="skill-catalog-entry-missing",
                    skill=record.name,
                )
            )

    for skill, conversion_tier in conversion.items():
        product_row = product.get(skill)
        if product_row is None:
            context.findings.append(
                _finding(
                    "skill-catalog-category-lint",
                    FINDING_ERROR,
                    docs_root / "agent_skill_authoring.md",
                    f"conversion-family skill '{skill}' is missing from the product catalog",
                    "Keep conversion-family rows and the product catalog in sync.",
                    code="conversion-skill-catalog-missing",
                    skill=skill,
                )
            )
            continue
        if product_row["category"] != "Conversion":
            context.findings.append(
                _finding(
                    "skill-catalog-category-lint",
                    FINDING_ERROR,
                    docs_root / "agent_integration.md",
                    f"conversion-family skill '{skill}' has category '{product_row['category']}'",
                    "Conversion-family skills should use the Conversion category for overlap checks.",
                    code="conversion-skill-category-mismatch",
                    skill=skill,
                )
            )
        product_tier = _strip_backticks(product_row["tier"])
        conversion_tier = _strip_backticks(conversion_tier)
        if conversion_tier and product_tier and conversion_tier != product_tier:
            context.findings.append(
                _finding(
                    "skill-catalog-category-lint",
                    FINDING_ERROR,
                    docs_root / "agent_integration.md",
                    f"skill '{skill}' tier differs between catalog and conversion-family table",
                    "Use one tier consistently across the authoring and integration docs.",
                    code="skill-catalog-tier-mismatch",
                    skill=skill,
                )
            )


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
        script_files = [p for p in sorted(scripts_dir.rglob("*")) if p.is_file()]
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
            if "json" in lowered and "json.dumps" not in text and "json.dump" not in text:
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
                fixture_path = record.skill_dir / str(rel_path)
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

        files_dir = record.skill_dir / "evals" / "files"
        if _has_files(files_dir) and not _has_fixture_notes(record.skill_dir):
            context.findings.append(
                _finding(
                    "skill-fixture-lint",
                    FINDING_ERROR,
                    files_dir,
                    "eval fixtures are missing source/provenance notes",
                    "Add evals/README.md, evals/files/README.md, or evals/files/SOURCE.md.",
                    code="skill-fixture-notes-missing",
                    skill=record.name,
                )
            )


def _lint_doc_crosslinks(context: LintContext) -> None:
    docs_root = context.docs_root
    if docs_root is None or not docs_root.is_dir():
        _skip(context, "agent-doc-crosslink-lint", "docs root is not available")
        return

    anchors_by_file = {
        doc_path.name: _markdown_anchors(doc_path.read_text(encoding="utf-8", errors="replace"))
        for doc_path in _iter_existing_doc_files(docs_root)
    }
    for doc_path in _iter_existing_doc_files(docs_root):
        text = doc_path.read_text(encoding="utf-8", errors="replace")
        for line_no, href in _iter_markdown_links(text):
            if href.startswith(("http://", "https://", "mailto:")):
                continue
            target, _, anchor = href.partition("#")
            target_path = (doc_path.parent / target).resolve() if target else doc_path.resolve()
            if target and not target_path.exists():
                context.findings.append(
                    _finding(
                        "agent-doc-crosslink-lint",
                        FINDING_ERROR,
                        doc_path,
                        f"markdown link target does not exist: {href}",
                        "Update the link or restore the referenced design document.",
                        code="agent-doc-link-missing",
                        line=line_no,
                    )
                )
                continue
            if anchor:
                target_name = target_path.name if target else doc_path.name
                if _normalize_anchor(anchor) not in anchors_by_file.get(target_name, set()):
                    context.findings.append(
                        _finding(
                            "agent-doc-crosslink-lint",
                            FINDING_ERROR,
                            doc_path,
                            f"markdown link anchor does not exist: {href}",
                            "Update the link anchor to match the target heading.",
                            code="agent-doc-anchor-missing",
                            line=line_no,
                        )
                    )

    eval_doc = docs_root / "agent_skill_evaluation.md"
    if eval_doc.is_file():
        text = eval_doc.read_text(encoding="utf-8", errors="replace")
        for lint_id in V1_LINT_IDS:
            if f"`{lint_id}`" not in text:
                context.findings.append(
                    _finding(
                        "agent-doc-crosslink-lint",
                        FINDING_ERROR,
                        eval_doc,
                        f"lint id '{lint_id}' is missing from the canonical evaluation doc",
                        "Keep agent_skill_evaluation.md#v1-engineering-lints as the canonical lint table.",
                        code="agent-doc-lint-id-missing",
                    )
                )

    for doc_path in _iter_existing_doc_files(docs_root):
        if doc_path.name == "agent_skills_deferred_roadmap.md":
            continue
        text = doc_path.read_text(encoding="utf-8", errors="replace")
        for line_no, command in _extract_nvflare_commands(text):
            if not command.startswith("nvflare agent"):
                continue
            message = _command_drift_message(command, check_flags=False)
            if message is None:
                continue
            context.findings.append(
                _finding(
                    "agent-doc-crosslink-lint",
                    FINDING_ERROR,
                    doc_path,
                    f"agent command reference is stale: {message}",
                    "Update the design doc command reference or the agent CLI command surface.",
                    code="agent-doc-command-reference-stale",
                    line=line_no,
                )
            )


def _resolve_docs_root(docs_root: Path | str | None) -> Optional[Path]:
    if docs_root is not None:
        return Path(docs_root)
    candidate = Path.cwd() / "docs" / "design"
    return candidate if (candidate / "agent_skill_evaluation.md").is_file() else None


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


def _eval_text(item: dict[str, Any]) -> str:
    parts = [str(item.get("id", "")), str(item.get("prompt", "")), str(item.get("expected_output", ""))]
    assertions = item.get("assertions", [])
    if isinstance(assertions, list):
        parts.extend(str(assertion) for assertion in assertions)
    return "\n".join(parts)


def _records_overlap(left: SkillRecord, right: SkillRecord) -> bool:
    left_tokens = _trigger_tokens(left)
    right_tokens = _trigger_tokens(right)
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


def _category_map(context: LintContext) -> dict[str, str]:
    if context.docs_root is None or not context.docs_root.is_dir():
        return {}
    product = _parse_product_catalog(context.docs_root / "agent_integration.md")
    return {skill: row["category"] for skill, row in product.items()}


def _parse_product_catalog(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        return {}
    rows = {}
    in_table = False
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if stripped.startswith("| Category | Skill | Tier | Purpose |"):
            in_table = True
            continue
        if not in_table:
            continue
        if not stripped.startswith("|"):
            break
        if set(stripped.replace("|", "").replace("-", "").replace(" ", "")) == set():
            continue
        columns = [column.strip() for column in stripped.strip("|").split("|")]
        if len(columns) < 4 or columns[0] == "Category":
            continue
        skill = _strip_backticks(columns[1])
        if skill:
            rows[skill] = {"category": columns[0], "tier": columns[2]}
    return rows


def _parse_conversion_table(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    rows = {}
    in_table = False
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if stripped.startswith("| Code Family | Skill | Scope | Current Repo Evidence | Tier |"):
            in_table = True
            continue
        if not in_table:
            continue
        if not stripped.startswith("|"):
            break
        columns = [column.strip() for column in stripped.strip("|").split("|")]
        if len(columns) < 5 or columns[0] == "Code Family" or columns[0].startswith("---"):
            continue
        skill = _strip_backticks(columns[1])
        if skill:
            rows[skill] = columns[4]
    return rows


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


def _command_drift_message(command: str, *, check_flags: bool = True) -> Optional[str]:
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
    if len(positional) >= 3 and positional[1] == "skills" and positional[2] not in _KNOWN_AGENT_SKILLS_COMMANDS:
        return f"unknown nvflare agent skills command '{positional[2]}' in '{command}'"

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
    return re.findall(r"nvflare|--?[A-Za-z0-9][\w-]*(?:=[^\s`]+)?|<[^>]+>|[A-Za-z0-9][\w.-]*", command)


def _looks_like_value(token: str) -> bool:
    return token.startswith("<") or "/" in token or token in {"on", "off", "json", "jsonl", "human"}


def _skill_has_helper_tests(skill_dir: Path) -> bool:
    tests_dir = skill_dir / "tests"
    if tests_dir.is_dir() and any(path.is_file() for path in tests_dir.rglob("*")):
        return True
    return any(path.name.endswith(("_test.py", ".test.py")) for path in skill_dir.rglob("*") if path.is_file())


def _skill_text_contains(skill_dir: Path, needle: str) -> bool:
    needle = needle.lower()
    return any(needle in text.lower() for _, text in _iter_skill_text_files(skill_dir))


def _iter_skill_text_files(skill_dir: Path, *, include_scripts: bool = False) -> Iterable[tuple[Path, str]]:
    candidates = [skill_dir / SKILL_FILE_NAME]
    references_dir = skill_dir / "references"
    if references_dir.is_dir():
        candidates.extend(sorted(path for path in references_dir.rglob("*") if path.suffix.lower() in {".md", ".txt"}))
    if include_scripts:
        scripts_dir = skill_dir / "scripts"
        if scripts_dir.is_dir():
            candidates.extend(sorted(path for path in scripts_dir.rglob("*") if path.is_file()))
    for path in candidates:
        if path.is_file():
            yield path, path.read_text(encoding="utf-8", errors="replace")


def _eval_mentions_file_editing(item: dict[str, Any]) -> bool:
    text = _eval_text(item).lower()
    return any(term in text for term in ("edit", "generate", "create", "export", "artifact", "file"))


def _has_files(path: Path) -> bool:
    return path.is_dir() and any(child.is_file() for child in path.rglob("*"))


def _has_fixture_notes(skill_dir: Path) -> bool:
    note_paths = (
        skill_dir / "evals" / "README.md",
        skill_dir / "evals" / "files" / "README.md",
        skill_dir / "evals" / "files" / "SOURCE.md",
    )
    return any(path.is_file() for path in note_paths)


def _iter_existing_doc_files(docs_root: Path) -> Iterable[Path]:
    for name in _DOC_FILES:
        path = docs_root / name
        if path.is_file():
            yield path


def _iter_markdown_links(text: str) -> Iterable[tuple[int, str]]:
    for line_no, line in enumerate(text.splitlines(), start=1):
        for match in _MARKDOWN_LINK_RE.finditer(line):
            yield line_no, match.group(1).strip()


def _markdown_anchors(text: str) -> set[str]:
    anchors = set()
    for line in text.splitlines():
        if not line.startswith("#"):
            continue
        heading = line.lstrip("#").strip()
        if heading:
            anchors.add(_normalize_anchor(heading))
    return anchors


def _normalize_anchor(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"`([^`]+)`", r"\1", value)
    value = re.sub(r"[^a-z0-9 _-]", "", value)
    value = value.replace(" ", "-")
    value = re.sub(r"-+", "-", value)
    return value.strip("-")


def _line_for_frontmatter_issue(skill_file: Path, code: str, message: str) -> Optional[int]:
    if code == "skill-frontmatter-field-required":
        match = re.search(r"field '([^']+)'", message)
        if match:
            return _line_for_field(skill_file, match.group(1))
    if code in {"skill-name-directory-mismatch", "skill-blast-radius-invalid", "skill-frontmatter-field-type"}:
        for field in ("name", "blast_radius", "description", "min_flare_version"):
            if field in message:
                return _line_for_field(skill_file, field)
    return 1 if skill_file.is_file() else None


def _line_for_field(skill_file: Path, field: str) -> Optional[int]:
    if not skill_file.is_file():
        return None
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
    )
