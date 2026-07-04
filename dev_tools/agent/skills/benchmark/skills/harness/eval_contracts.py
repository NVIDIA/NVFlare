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

"""Load eval declarations into immutable, run-scoped behavior contracts."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

MAX_EVAL_CONTRACT_BYTES = 512 * 1024
MAX_BEHAVIORS_PER_CATEGORY = 200
MAX_BEHAVIOR_ID_CHARS = 160
MAX_BEHAVIOR_DESCRIPTION_CHARS = 2000
MAX_EVAL_FILES = 100
MAX_EVAL_FILE_BYTES = 2 * 1024 * 1024
MAX_EVAL_TOTAL_BYTES = 25 * 1024 * 1024
BEHAVIOR_CATEGORIES = ("mandatory_behavior", "prohibited_behavior", "optional_behavior")


class EvalContractError(ValueError):
    """Raised when an eval declaration cannot become a safe run contract."""


def _read_bounded_json(path: Path) -> tuple[dict[str, Any], bytes]:
    if path.is_symlink() or not path.is_file():
        raise EvalContractError(f"eval declaration is not a regular file: {path}")
    try:
        size = path.stat().st_size
        if size > MAX_EVAL_CONTRACT_BYTES:
            raise EvalContractError(f"eval declaration exceeds {MAX_EVAL_CONTRACT_BYTES} bytes: {path}")
        raw = path.read_bytes()
        data = json.loads(raw)
    except EvalContractError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvalContractError(f"could not read eval declaration {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise EvalContractError(f"eval declaration must contain a JSON object: {path}")
    return data, raw


def _behavior_contract(entries: Any, label: str) -> dict[str, dict[str, str]]:
    if entries is None:
        return {}
    if not isinstance(entries, list):
        raise EvalContractError(f"{label} must be a list")
    if len(entries) > MAX_BEHAVIORS_PER_CATEGORY:
        raise EvalContractError(f"{label} exceeds {MAX_BEHAVIORS_PER_CATEGORY} entries")
    result: dict[str, dict[str, str]] = {}
    for index, entry in enumerate(entries):
        if isinstance(entry, str):
            behavior_id = entry.strip()
            description = ""
        elif isinstance(entry, Mapping):
            behavior_id = str(entry.get("id") or "").strip()
            description = str(entry.get("description") or "").strip()
        else:
            raise EvalContractError(f"{label}[{index}] must be a string or object")
        if not behavior_id or len(behavior_id) > MAX_BEHAVIOR_ID_CHARS:
            raise EvalContractError(f"{label}[{index}].id must be 1..{MAX_BEHAVIOR_ID_CHARS} characters")
        if behavior_id in result:
            raise EvalContractError(f"{label} contains duplicate id {behavior_id!r}")
        result[behavior_id] = {"description": description[:MAX_BEHAVIOR_DESCRIPTION_CHARS]}
    return result


def _eval_case_files(evals_path: Path, entries: Any, label: str) -> tuple[list[dict[str, Any]], Path | None]:
    if not isinstance(entries, list):
        raise EvalContractError(f"{label} must be a list")
    if not entries:
        return [], None
    if len(entries) > MAX_EVAL_FILES:
        raise EvalContractError(f"{label} exceeds {MAX_EVAL_FILES} entries")
    eval_root = evals_path.parent
    root_resolved = eval_root.resolve()
    files: list[dict[str, Any]] = []
    parents = set()
    total_bytes = 0
    for index, value in enumerate(entries):
        if not isinstance(value, str) or not value or "\\" in value:
            raise EvalContractError(f"{label}[{index}] must be a relative POSIX path")
        relative = Path(*value.split("/"))
        if relative.is_absolute() or ".." in relative.parts:
            raise EvalContractError(f"{label}[{index}] must stay beneath the eval suite")
        path = eval_root / relative
        try:
            resolved = path.resolve(strict=True)
            if not resolved.is_relative_to(root_resolved):
                raise EvalContractError(f"{label}[{index}] escapes the eval suite")
            current = eval_root
            for part in relative.parts:
                current = current / part
                if os.path.islink(current):
                    raise EvalContractError(f"{label}[{index}] contains a symlink component")
            size = resolved.stat().st_size
            if not resolved.is_file() or size > MAX_EVAL_FILE_BYTES:
                raise EvalContractError(
                    f"{label}[{index}] must be a regular file no larger than {MAX_EVAL_FILE_BYTES} bytes"
                )
            raw = resolved.read_bytes()
        except EvalContractError:
            raise
        except OSError as exc:
            raise EvalContractError(f"could not read {label}[{index}]: {exc}") from exc
        total_bytes += len(raw)
        if total_bytes > MAX_EVAL_TOTAL_BYTES:
            raise EvalContractError(f"{label} exceeds {MAX_EVAL_TOTAL_BYTES} total bytes")
        parents.add(relative.parent)
        files.append(
            {
                "path": relative.as_posix(),
                "fixture_path": relative.name,
                "sha256": hashlib.sha256(raw).hexdigest(),
                "size_bytes": len(raw),
            }
        )
    if len(parents) != 1:
        raise EvalContractError(f"{label} entries must share one fixture directory")
    fixture_dir = (eval_root / next(iter(parents))).resolve()
    return files, fixture_dir


def load_eval_contract(path: Path, *, eval_suite_skill: str, case_id: str) -> dict[str, Any]:
    """Resolve one external eval case into the compact contract mounted per run."""

    data, raw = _read_bounded_json(path)
    declared_skill = str(data.get("skill_name") or "").strip()
    if declared_skill != eval_suite_skill:
        raise EvalContractError(
            f"eval declaration skill_name={declared_skill!r} does not match requested suite {eval_suite_skill!r}"
        )
    evals = data.get("evals")
    if not isinstance(evals, list):
        raise EvalContractError("eval declaration evals must be a list")
    matches = [item for item in evals if isinstance(item, Mapping) and str(item.get("id") or "") == case_id]
    if len(matches) != 1:
        raise EvalContractError(f"expected exactly one eval case {case_id!r}; found {len(matches)}")
    eval_case = matches[0]
    nvflare = eval_case.get("nvflare")
    if not isinstance(nvflare, Mapping):
        raise EvalContractError(f"eval case {case_id!r} is missing its nvflare behavior contract")
    raw_expected_skill = nvflare.get("expected_skill", eval_suite_skill)
    if raw_expected_skill is None:
        expected_skill = None
    elif isinstance(raw_expected_skill, str) and raw_expected_skill.strip():
        expected_skill = raw_expected_skill.strip()
    else:
        raise EvalContractError(f"eval case {case_id!r} expected_skill must be a non-empty string or null")
    prompt = eval_case.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise EvalContractError(f"eval case {case_id!r} prompt must be a non-empty string")
    prompt_bytes = prompt.encode("utf-8")
    if len(prompt_bytes) > MAX_EVAL_FILE_BYTES:
        raise EvalContractError(f"eval case {case_id!r} prompt exceeds {MAX_EVAL_FILE_BYTES} bytes")
    files, fixture_dir = _eval_case_files(path, eval_case.get("files"), f"{case_id}.files")
    assertions = eval_case.get("assertions") or []
    if not isinstance(assertions, list) or not all(isinstance(item, str) for item in assertions):
        raise EvalContractError(f"eval case {case_id!r} assertions must be a list of strings")
    contract: dict[str, Any] = {
        "schema_version": "1",
        "eval_suite_skill": eval_suite_skill,
        "case_id": case_id,
        "expected_skill": expected_skill,
        "negative_for": str(nvflare.get("negative_for") or "")[:MAX_BEHAVIOR_DESCRIPTION_CHARS],
        "prompt": prompt,
        "prompt_sha256": hashlib.sha256(prompt_bytes).hexdigest(),
        "prompt_bytes": len(prompt_bytes),
        "expected_output": str(eval_case.get("expected_output") or "")[:MAX_BEHAVIOR_DESCRIPTION_CHARS],
        "assertions": [item[:MAX_BEHAVIOR_DESCRIPTION_CHARS] for item in assertions[:MAX_BEHAVIORS_PER_CATEGORY]],
        "files": files,
        "fixture_dir": str(fixture_dir) if fixture_dir is not None else None,
        "source_path": str(path),
        "source_sha256": hashlib.sha256(raw).hexdigest(),
    }
    for category in BEHAVIOR_CATEGORIES:
        contract[category] = _behavior_contract(nvflare.get(category), f"{case_id}.nvflare.{category}")
    return contract


def load_run_behavior_contract(path: Path) -> dict[str, Any]:
    """Load the already-normalized contract from its read-only run mount."""

    data, _raw = _read_bounded_json(path)
    if str(data.get("schema_version") or "") != "1":
        raise EvalContractError("run behavior contract schema_version must be '1'")
    if not isinstance(data.get("eval_suite_skill"), str) or not data["eval_suite_skill"].strip():
        raise EvalContractError("run behavior contract eval_suite_skill must be a non-empty string")
    if not isinstance(data.get("case_id"), str) or not data["case_id"].strip():
        raise EvalContractError("run behavior contract case_id must be a non-empty string")
    for category in BEHAVIOR_CATEGORIES:
        if not isinstance(data.get(category), dict):
            raise EvalContractError(f"run behavior contract {category} must be an object")
        if len(data[category]) > MAX_BEHAVIORS_PER_CATEGORY:
            raise EvalContractError(f"run behavior contract {category} exceeds {MAX_BEHAVIORS_PER_CATEGORY} entries")
    return data


def apply_behavior_contract(record: dict[str, Any], contract: Mapping[str, Any]) -> None:
    """Seed declared behaviors as missing until an authoritative checker supplies evidence."""

    eval_suite_skill = str(contract.get("eval_suite_skill") or "").strip()
    case_id = str(contract.get("case_id") or "").strip()
    if eval_suite_skill:
        record["eval_suite_skill"] = eval_suite_skill
    if case_id:
        record["case_id"] = case_id
    undeclared: dict[str, list[str]] = {}
    for category in BEHAVIOR_CATEGORIES:
        declared = contract.get(category)
        if not isinstance(declared, Mapping):
            continue
        reported = record.get(category)
        if isinstance(reported, dict):
            extras = sorted(str(key)[:MAX_BEHAVIOR_ID_CHARS] for key in reported if str(key) not in declared)[
                :MAX_BEHAVIORS_PER_CATEGORY
            ]
            if extras:
                undeclared[category] = extras
        behavior_map: dict[str, Any] = {}
        record[category] = behavior_map
        for behavior_id, metadata in declared.items():
            description = metadata.get("description") if isinstance(metadata, Mapping) else ""
            behavior_map[str(behavior_id)] = {
                "status": "missing",
                "evidence": "No authoritative harness evaluator produced evidence for this declared behavior.",
                "description": str(description or "")[:MAX_BEHAVIOR_DESCRIPTION_CHARS],
                "source": "eval_contract",
            }
    if undeclared:
        record["untrusted_undeclared_behavior"] = undeclared
    record["behavior_contract"] = {
        "schema_version": str(contract.get("schema_version") or "1"),
        "eval_suite_skill": eval_suite_skill,
        "case_id": case_id,
        "expected_skill": contract.get("expected_skill"),
        "negative_for": str(contract.get("negative_for") or ""),
        "prompt_sha256": str(contract.get("prompt_sha256") or ""),
        "files": list(contract.get("files") or []),
        "source_path": str(contract.get("source_path") or ""),
        "source_sha256": str(contract.get("source_sha256") or ""),
    }
