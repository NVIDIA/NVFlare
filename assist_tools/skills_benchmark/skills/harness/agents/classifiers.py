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

"""Exit and failure classifiers for benchmark agent adapters."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def stderr_excerpt(stderr_path: Path) -> str:
    stderr_text = ""
    try:
        stderr_text = stderr_path.read_text(encoding="utf-8", errors="replace")[:4000]
    except OSError:
        pass
    return stderr_text


def generic_cli_exit(exit_code: int, stderr_path: Path, classifier_id: str = "generic_cli") -> dict[str, Any]:
    stderr_text = stderr_excerpt(stderr_path)
    return {
        "classifier": classifier_id,
        "exit_code": exit_code,
        "passed": exit_code == 0,
        "failure_category": "agent_unknown_failure" if exit_code else None,
        "stderr_excerpt": stderr_text,
    }


EXIT_CLASSIFIERS = {"generic_cli", "stderr_patterns"}


def validate_exit_classifier(classifier_id: str) -> None:
    if classifier_id not in EXIT_CLASSIFIERS:
        raise ValueError(f"Unknown agent exit classifier: {classifier_id}")


def as_string_list(value: Any, field_path: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise ValueError(f"{field_path} must be a list of non-empty strings")
    return [str(item).lower() for item in value]


def as_exit_codes(value: Any, field_path: str) -> set[int]:
    if value is None:
        return set()
    if not isinstance(value, list) or any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        raise ValueError(f"{field_path} must be a list of integer exit codes")
    return {int(item) for item in value}


def validate_stderr_pattern_rules(config: dict[str, Any]) -> None:
    rules = config.get("rules") or []
    if not isinstance(rules, list):
        raise ValueError("exit.rules must be a list")
    for index, rule in enumerate(rules):
        if not isinstance(rule, dict):
            raise ValueError(f"exit.rules[{index}] must be a mapping")
        category = rule.get("category")
        if not isinstance(category, str) or not category:
            raise ValueError(f"exit.rules[{index}].category must be a non-empty string")
        any_patterns = as_string_list(rule.get("any"), f"exit.rules[{index}].any")
        all_patterns = as_string_list(rule.get("all"), f"exit.rules[{index}].all")
        exit_codes = as_exit_codes(rule.get("exit_codes"), f"exit.rules[{index}].exit_codes")
        if not any_patterns and not all_patterns and not exit_codes:
            raise ValueError(f"exit.rules[{index}] must define at least one of any, all, or exit_codes")


def validate_exit_config(config: dict[str, Any]) -> None:
    classifier_id = str(config.get("classifier") or "")
    validate_exit_classifier(classifier_id)
    if classifier_id == "stderr_patterns":
        validate_stderr_pattern_rules(config)


def stderr_rule_matches(rule: dict[str, Any], exit_code: int, stderr_lower: str) -> bool:
    if exit_code == 0:
        return False
    exit_codes = as_exit_codes(rule.get("exit_codes"), "exit.rules[].exit_codes")
    if exit_codes and exit_code not in exit_codes:
        return False
    all_patterns = as_string_list(rule.get("all"), "exit.rules[].all")
    if all_patterns and not all(pattern in stderr_lower for pattern in all_patterns):
        return False
    any_patterns = as_string_list(rule.get("any"), "exit.rules[].any")
    if any_patterns and not any(pattern in stderr_lower for pattern in any_patterns):
        return False
    return bool(exit_codes or all_patterns or any_patterns)


def stderr_pattern_exit(exit_code: int, stderr_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    summary = generic_cli_exit(exit_code, stderr_path, classifier_id="stderr_patterns")
    stderr_lower = str(summary.get("stderr_excerpt") or "").lower()
    for rule in config.get("rules") or []:
        if stderr_rule_matches(rule, exit_code, stderr_lower):
            summary["failure_category"] = str(rule["category"])
            break
    return summary


def classify_exit(exit_code: int, stderr_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    validate_exit_config(config)
    classifier_id = str(config.get("classifier") or "")
    if classifier_id == "generic_cli":
        return generic_cli_exit(exit_code, stderr_path)
    if classifier_id == "stderr_patterns":
        return stderr_pattern_exit(exit_code, stderr_path, config)
    raise ValueError(f"Unknown agent exit classifier: {classifier_id}")
