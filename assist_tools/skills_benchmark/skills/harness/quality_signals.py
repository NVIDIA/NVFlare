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

"""Best-effort quality signals derived from job instructions and final output."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Mapping

FLOAT_PATTERN = r"(?<![A-Za-z0-9_])([0-9]+\.[0-9]+)(?![A-Za-z0-9_])"
GENERIC_VALIDATION_METRIC_PATTERN = (
    r"\b(?:"
    r"(?:(?:aggregated|aggregate|global|server)\s+)?(?:best\s+)?validation\s+(?:metric|[A-Za-z0-9_/-]+)"
    r"|(?:best\s+)?(?:aggregated|aggregate|global|server)\s+validation\s+(?:metric|[A-Za-z0-9_/-]+)"
    r")\b"
)
METRIC_ALIAS_PATTERNS = {
    "AUROC": r"\b(?:AUROC|AUC)\b|\b(?:valid|validation)[_-]?auroc\b",
    "accuracy": r"\baccuracy\b|\b(?:valid|validation)[_-]?accuracy\b|\bacc\b",
    "loss": r"\b(?:loss|valid[_-]?loss|validation[_-]?loss|train[_-]?loss)\b",
    "f1": r"\b(?:f1|f1[_-]?score|valid[_-]?f1|validation[_-]?f1)\b",
}


def canonical_metric_name(name: Any) -> str:
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", str(name or "").strip()).strip("_")
    aliases = {
        "auroc": "AUROC",
        "auc": "AUROC",
        "valid_auroc": "AUROC",
        "validation_auroc": "AUROC",
        "accuracy": "accuracy",
        "acc": "accuracy",
        "valid_accuracy": "accuracy",
        "validation_accuracy": "accuracy",
        "loss": "loss",
        "valid_loss": "loss",
        "validation_loss": "loss",
        "train_loss": "loss",
        "f1": "f1",
        "f1_score": "f1",
        "valid_f1": "f1",
        "validation_f1": "f1",
    }
    return aliases.get(normalized.lower(), normalized)


def first_float(pattern: str, text: str) -> float | None:
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if not match:
        return None
    try:
        return float(match.group(1))
    except (TypeError, ValueError):
        return None


def parse_float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def is_numeric_metric_value(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def format_metric_value(value: Any) -> str:
    if isinstance(value, float):
        return f" {value:.4f}."
    if isinstance(value, int) and not isinstance(value, bool):
        return f" {value}."
    return "."


def line_value_after_metric(line: str, match: re.Match[str]) -> float | None:
    tail = line[match.end() :]
    delimiter = re.search(r"[,;]", tail)
    if delimiter:
        tail = tail[: delimiter.start()]
    value_match = re.search(FLOAT_PATTERN, tail)
    return parse_float(value_match.group(1)) if value_match else None


def label_from_metric_line(line: str) -> str | None:
    match = re.match(r"\s*[-*]?\s*`?([^`:]+?)`?\s*:", line)
    if not match:
        return None
    label = match.group(1).strip("` ")
    return label or None


def metric_value_entry(value: float, label: str | None = None) -> dict[str, Any]:
    entry: dict[str, Any] = {"value": value}
    if label:
        entry["label"] = label
    return entry


def following_line_values(lines: list[str], start_index: int, metric_pattern: str, limit: int = 8) -> list[float]:
    entries, _consumed = following_line_value_entries(lines, start_index, metric_pattern, limit)
    return [entry["value"] for entry in entries]


def line_metric_entries(line: str, metric_pattern: str) -> list[dict[str, Any]]:
    label = label_from_metric_line(line)
    entries: list[dict[str, Any]] = []
    for match in re.finditer(metric_pattern, line, flags=re.IGNORECASE):
        value = line_value_after_metric(line, match)
        if value is not None:
            entries.append(metric_value_entry(value, label))
    return entries


def single_unlabeled_metric_entry(line: str) -> dict[str, Any] | None:
    label = label_from_metric_line(line)
    matches = list(re.finditer(FLOAT_PATTERN, line))
    if len(matches) != 1:
        return None
    value = parse_float(matches[0].group(1))
    return metric_value_entry(value, label) if value is not None else None


def following_line_value_entries(
    lines: list[str],
    start_index: int,
    metric_pattern: str,
    limit: int = 8,
) -> tuple[list[dict[str, Any]], set[int]]:
    entries: list[dict[str, Any]] = []
    consumed: set[int] = set()
    for index in range(start_index + 1, min(len(lines), start_index + 1 + limit)):
        line = lines[index]
        stripped = line.strip()
        if not stripped:
            if entries:
                break
            continue
        if entries and not stripped.startswith(("-", "*", "`")) and ":" not in stripped:
            break
        line_entries = line_metric_entries(line, metric_pattern)
        if not line_entries:
            entry = single_unlabeled_metric_entry(line)
            line_entries = [entry] if entry else []
        if line_entries:
            entries.extend(line_entries)
            consumed.add(index)
        if entries and not stripped.startswith(("-", "*", "`")):
            break
    return entries, consumed


def metric_values(metric_name: str, text: str) -> list[float]:
    return [entry["value"] for entry in metric_value_entries(metric_name, text)]


def metric_value_entries(metric_name: str, text: str) -> list[dict[str, Any]]:
    canonical = canonical_metric_name(metric_name)
    pattern = METRIC_ALIAS_PATTERNS.get(canonical)
    if not pattern:
        pattern = rf"\b{re.escape(canonical)}\b"
    lines = text.splitlines()
    entries: list[dict[str, Any]] = []
    consumed_lines: set[int] = set()
    for index, line in enumerate(lines):
        if index in consumed_lines:
            continue
        matches = list(re.finditer(pattern, line, flags=re.IGNORECASE))
        if not matches:
            continue
        line_entries = line_metric_entries(line, pattern)
        if line_entries:
            entries.extend(line_entries)
            continue
        following_entries, consumed = following_line_value_entries(lines, index, pattern)
        entries.extend(following_entries)
        consumed_lines.update(consumed)
    return entries


def metric_mentioned(metric_name: str, text: str) -> bool:
    canonical = canonical_metric_name(metric_name)
    pattern = METRIC_ALIAS_PATTERNS.get(canonical, rf"\b{re.escape(canonical)}\b")
    return re.search(pattern, text, flags=re.IGNORECASE) is not None


def is_site_label(label: Any) -> bool:
    return re.search(r"\bsite[-_ ]?\d+\b", str(label or ""), flags=re.IGNORECASE) is not None


def is_fl_summary_metric_label(label: Any) -> bool:
    text = str(label or "")
    return (
        not is_site_label(text) and re.search(GENERIC_VALIDATION_METRIC_PATTERN, text, flags=re.IGNORECASE) is not None
    )


def generic_validation_metric_entries(text: str) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for line in text.splitlines():
        label = label_from_metric_line(line)
        for match in re.finditer(GENERIC_VALIDATION_METRIC_PATTERN, line, flags=re.IGNORECASE):
            value = line_value_after_metric(line, match)
            if value is None:
                continue
            entries.append(metric_value_entry(value, label or match.group(0).strip()))
    return entries


def merge_metric_entries(*entry_groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[tuple[str, float]] = set()
    for entries in entry_groups:
        for entry in entries:
            value = entry.get("value")
            if not is_numeric_metric_value(value):
                continue
            key = (str(entry.get("label") or "").strip().lower(), float(value))
            if key in seen:
                continue
            seen.add(key)
            merged.append(entry)
    return merged


def fl_summary_metric_entry(entries: list[dict[str, Any]]) -> dict[str, Any] | None:
    for entry in reversed(entries):
        value = entry.get("value")
        if is_numeric_metric_value(value) and is_fl_summary_metric_label(entry.get("label")):
            return entry
    return None


def reported_metric_payload(name: str, entries: list[dict[str, Any]]) -> dict[str, Any]:
    values = [entry["value"] for entry in entries if is_numeric_metric_value(entry.get("value"))]
    labels = [entry.get("label") for entry in entries]
    site_entries = [
        entry for entry in entries if is_numeric_metric_value(entry.get("value")) and is_site_label(entry.get("label"))
    ]
    summary_entry = fl_summary_metric_entry(entries)
    has_single_value = len(values) == 1
    has_site_labels = bool(values) and len(site_entries) == len(values)
    if summary_entry:
        value = summary_entry["value"]
        value_scope = "fl_summary_metric"
    elif has_single_value:
        value = values[0]
        value_scope = "reported_scalar"
    elif has_site_labels:
        value = None
        value_scope = "site_values_only"
    elif values:
        value = None
        value_scope = "reported_values_only"
    else:
        value = None
        value_scope = "not_available"
    return {
        "name": canonical_metric_name(name),
        "value": value,
        "reported_values": values,
        "reported_value_labels": labels,
        "reported_value_entries": entries,
        "site_values": [entry["value"] for entry in site_entries],
        "site_value_labels": [entry.get("label") for entry in site_entries],
        "site_value_count": len(site_entries),
        "summary_value_label": summary_entry.get("label") if summary_entry else None,
        "value_scope": value_scope,
        "source": "agent_last_message",
    }


PRIMARY_METRIC_PATTERNS = (
    r"^\s*[-*]?\s*([A-Za-z][A-Za-z0-9_./ -]{0,40}?)\s+is\s+the\s+main\s+metric\b",
    r"\bmain\s+metric\s*(?:to\s+watch\s+)?(?:is|:)\s*([A-Za-z][A-Za-z0-9_./ -]{0,40})",
    r"\bprimary\s+(?:validation\s+)?metric\s*(?:is|:)\s*([A-Za-z][A-Za-z0-9_./ -]{0,40})",
    r"\btarget\s+(?:validation\s+)?metric\s*(?:is|:)\s*([A-Za-z][A-Za-z0-9_./ -]{0,40})",
)


def primary_metric_from_guidance(guidance_text: str) -> str | None:
    metric, _source = primary_metric_from_guidance_sources(
        [{"source_type": "job_guidance", "text": guidance_text}],
        guidance_text,
    )
    return metric


def primary_metric_from_text(guidance_text: str) -> str | None:
    for pattern in PRIMARY_METRIC_PATTERNS:
        match = re.search(pattern, guidance_text, flags=re.IGNORECASE | re.MULTILINE)
        if match:
            metric = re.split(r"[,.;\n]", match.group(1).strip(), maxsplit=1)[0].strip()
            metric = canonical_metric_name(metric)
            if metric:
                return metric
    return None


def primary_metric_from_readme(readme_text: str) -> str | None:
    return primary_metric_from_guidance(readme_text)


def guidance_source_payload(source: Any) -> tuple[str | None, list[dict[str, Any]]]:
    if source is None:
        return None, []
    if isinstance(source, Path):
        path = str(source)
        return path, [{"path": path, "source_type": "job_guidance"}]
    if isinstance(source, list):
        sources = []
        for item in source:
            if isinstance(item, dict):
                path = item.get("path")
                if path:
                    sources.append(
                        {
                            "path": str(path),
                            "source_type": str(item.get("source_type") or "job_guidance"),
                        }
                    )
            elif item:
                sources.append({"path": str(item), "source_type": "job_guidance"})
        primary = sources[0]["path"] if sources else None
        return primary, sources
    path = str(source)
    return path, [{"path": path, "source_type": "job_guidance"}]


def guidance_source_entries(source: Any, guidance_text: str) -> list[dict[str, Any]]:
    if isinstance(source, list):
        entries = []
        for item in source:
            if isinstance(item, dict):
                entries.append(
                    {
                        "path": str(item.get("path") or ""),
                        "source_type": str(item.get("source_type") or "job_guidance"),
                        "text": str(item.get("text") or ""),
                    }
                )
            elif item:
                entries.append({"path": str(item), "source_type": "job_guidance", "text": ""})
        if any(entry["text"] for entry in entries):
            return entries

    primary_source, sources = guidance_source_payload(source)
    if sources:
        return [
            {
                "path": str(sources[0].get("path") or primary_source or ""),
                "source_type": str(sources[0].get("source_type") or "job_guidance"),
                "text": guidance_text,
            }
        ]
    return [{"path": "", "source_type": "job_guidance", "text": guidance_text}]


def public_guidance_source(entry: dict[str, Any]) -> dict[str, Any]:
    result = {
        "source_type": str(entry.get("source_type") or "job_guidance"),
    }
    if entry.get("path"):
        result["path"] = str(entry["path"])
    return result


def primary_metric_from_guidance_sources(
    guidance_sources: list[dict[str, Any]], guidance_text: str
) -> tuple[str | None, dict[str, Any] | None]:
    entries = guidance_sources or [{"source_type": "job_guidance", "text": guidance_text}]
    for entry in entries:
        metric = primary_metric_from_text(str(entry.get("text") or ""))
        if metric:
            return metric, entry
    if not any(entry.get("text") for entry in entries):
        metric = primary_metric_from_text(guidance_text)
        if metric:
            return metric, entries[0] if entries else None
    return None, None


def reported_validation_metric(last_message: str, expected_metric: str | None) -> dict[str, Any]:
    detected = []
    generic_entries = generic_validation_metric_entries(last_message)
    for name in ("AUROC", "accuracy", "loss", "f1"):
        entries = metric_value_entries(name, last_message)
        if entries or metric_mentioned(name, last_message):
            detected.append(reported_metric_payload(name, entries))
    if generic_entries:
        detected.append(reported_metric_payload("validation metric", generic_entries))
    if expected_metric and metric_mentioned(expected_metric, last_message):
        entries = merge_metric_entries(metric_value_entries(expected_metric, last_message), generic_entries)
        return reported_metric_payload(expected_metric, entries)
    if detected:
        return detected[0]
    return {
        "name": None,
        "value": None,
        "reported_values": [],
        "reported_value_labels": [],
        "reported_value_entries": [],
        "site_values": [],
        "site_value_labels": [],
        "site_value_count": 0,
        "value_scope": "not_available",
        "source": "agent_last_message",
    }


def required_validation_metric_status(signal: Mapping[str, Any] | None) -> str:
    if not isinstance(signal, dict) or not signal.get("expected_primary_metric"):
        return "not_required"
    if signal.get("metric_value_available"):
        return "present"
    metric = signal.get("reported_validation_metric")
    if isinstance(metric, dict):
        value = metric.get("value")
        values = metric.get("reported_values")
        if is_numeric_metric_value(value):
            return "present"
        if isinstance(values, list) and any(is_numeric_metric_value(item) for item in values):
            return "present"
    return "missing"


def critical_quality_checks_failed(*sources: Mapping[str, Any] | None) -> bool:
    for source in sources:
        if not isinstance(source, dict):
            continue
        checks = source.get("quality_checks")
        if not isinstance(checks, list):
            continue
        for check in checks:
            if not isinstance(check, dict):
                continue
            severity = str(check.get("severity") or "").lower()
            status = str(check.get("status") or "").lower()
            if severity == "critical" and (check.get("passed") is False or status in {"fail", "failed", "error"}):
                return True
    return False


def metric_signal(guidance_source: Any, guidance_text: str, final_message: str) -> dict[str, Any]:
    guidance_entries = guidance_source_entries(guidance_source, guidance_text)
    expected, matched_source = primary_metric_from_guidance_sources(guidance_entries, guidance_text)
    reported = reported_validation_metric(final_message, expected)
    _primary_source, sources = guidance_source_payload(guidance_source)
    if not sources and guidance_entries:
        sources = [public_guidance_source(entry) for entry in guidance_entries if entry.get("path")]
    matched_public_source = public_guidance_source(matched_source) if matched_source else {}
    primary_source = matched_public_source.get("path") or (sources[0]["path"] if sources else None)
    signal: dict[str, Any] = {
        "source": primary_source,
        "matched_source": matched_public_source or None,
        "sources": sources,
        "source_type": "job_guidance",
        "expected_primary_metric": expected,
        "reported_validation_metric": reported,
        "available": bool(expected),
    }
    if not expected:
        signal["status"] = "not_available"
        return signal

    value = reported.get("value")
    reported_values = reported.get("reported_values")
    if not isinstance(reported_values, list):
        reported_values = []
    site_values = reported.get("site_values")
    if not isinstance(site_values, list):
        site_values = []
    has_value = is_numeric_metric_value(value)
    numeric_reported_values = [
        reported_value for reported_value in reported_values if is_numeric_metric_value(reported_value)
    ]
    has_reported_numeric = has_value or bool(numeric_reported_values)
    reported_name = canonical_metric_name(reported.get("name"))
    expected_name = canonical_metric_name(expected)
    names_match = bool(reported.get("name")) and reported_name == expected_name
    aligned = names_match and has_reported_numeric
    mismatch = bool(reported.get("name")) and not names_match
    if aligned:
        status = "pass"
        if has_value:
            evidence = (
                f"Job guidance declares {expected} as the primary metric, and the final response reported "
                f"{reported.get('name')} {value:.4f}."
            )
        elif site_values:
            evidence = (
                f"Job guidance declares {expected} as the primary metric, and the final response reported "
                f"{len(site_values)} site-level {reported.get('name')} values."
            )
        else:
            evidence = (
                f"Job guidance declares {expected} as the primary metric, and the final response reported "
                f"{len(numeric_reported_values)} {reported.get('name')} values."
            )
    elif mismatch:
        status = "fail"
        if has_value:
            evidence = (
                f"Job guidance declares {expected} as the primary metric, but the final response reported "
                f"{reported.get('name')}" + format_metric_value(value)
            )
        else:
            evidence = (
                f"Job guidance declares {expected} as the primary metric, but the final response reported "
                f"{reported.get('name')}."
            )
    elif reported.get("name"):
        status = "missing"
        if site_values:
            evidence = (
                f"Job guidance declares {expected} as the primary metric, and the final response reported "
                f"{len(site_values)} site-level {reported.get('name')} values but no single FL-level value."
            )
        elif reported_values:
            evidence = (
                f"Job guidance declares {expected} as the primary metric, and the final response reported "
                f"{len(reported_values)} {reported.get('name')} values but no single FL-level value."
            )
        else:
            evidence = (
                f"Job guidance declares {expected} as the primary metric, and the final response mentioned "
                f"{reported.get('name')} but did not report a numeric value."
            )
    else:
        status = "missing"
        evidence = f"Job guidance declares {expected} as the primary metric, but the final response did not report it."

    signal.update(
        {
            "status": status,
            "evidence": evidence,
            "metric_value_available": has_reported_numeric,
            "metric_scalar_available": has_value,
            "aligned_with_job_guidance": aligned,
            "aligned_with_readme": aligned,
            "mismatch": mismatch,
        }
    )
    return signal


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("readme", type=Path)
    parser.add_argument("final_message", type=Path)
    args = parser.parse_args()
    readme_text = args.readme.read_text(encoding="utf-8", errors="replace") if args.readme.is_file() else ""
    final_text = (
        args.final_message.read_text(encoding="utf-8", errors="replace") if args.final_message.is_file() else ""
    )
    print(json.dumps(metric_signal(args.readme if args.readme.is_file() else None, readme_text, final_text), indent=2))


if __name__ == "__main__":
    main()
