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

"""Selection and integrity checks for operator-prepared task references."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from fedready.agents.common import adapter_source_digest

_PACKAGED_TASK_EXAMPLE_DIR = Path(__file__).resolve().parent / "task_example"
_SOURCE_TASK_EXAMPLE_DIR = Path(__file__).resolve().parents[2] / "task_example"
_CONFIGURED_TASK_EXAMPLE_DIR = os.environ.get("FEDREADY_TASK_EXAMPLE_DIR")
TASK_EXAMPLE_DIR = (
    Path(_CONFIGURED_TASK_EXAMPLE_DIR).expanduser().resolve()
    if _CONFIGURED_TASK_EXAMPLE_DIR
    else (_PACKAGED_TASK_EXAMPLE_DIR if _PACKAGED_TASK_EXAMPLE_DIR.is_dir() else _SOURCE_TASK_EXAMPLE_DIR)
)


def adapter_preflight_task_example_path(*, record_type: str) -> Path | None:
    example = _select_task_example(record_type=record_type)
    return _task_example_asset_path(example, "image_path", "image")


def task_example_image_path() -> Path | None:
    return _task_example_asset_path(_select_task_example(), "image_path", "image")


def task_example_context() -> dict[str, Any]:
    example = _select_task_example()
    return {
        "schema_version": "fedready.task_example_context.v1",
        "root": "task_example",
        "storage": "canonical_final_expected_form",
        "examples": [_public_task_example(example)] if example is not None else [],
        "safe_to_share": True,
    }


def _load_task_example_manifest() -> dict[str, Any]:
    manifest_path = TASK_EXAMPLE_DIR / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return manifest if isinstance(manifest, dict) else {}


def _select_task_example(*, record_type: str | None = None) -> Mapping[str, Any] | None:
    """Select the operator-declared reference, never a keyword-scored substitute."""
    task_key = os.environ.get("FEDREADY_TASK_EXAMPLE_ID", "").strip()
    examples = _load_task_example_manifest().get("examples", [])
    if not isinstance(examples, list):
        return None
    matches = [item for item in examples if isinstance(item, Mapping) and item.get("task_key") == task_key]
    if not task_key or len(matches) != 1:
        return None
    example = matches[0]
    if record_type is not None and not _task_example_record_type_matches(
        str(example.get("record_type") or ""), record_type
    ):
        return None
    return example if _task_example_assets_valid(example) else None


def _public_task_example(example: Mapping[str, Any]) -> dict[str, Any]:
    form = example.get("final_expected_form")
    safe_form = dict(form) if isinstance(form, Mapping) else {}
    return {
        "task_key": example.get("task_key"),
        "task": example.get("task"),
        "record_type": example.get("record_type"),
        "description": example.get("description"),
        "final_expected_form": safe_form,
    }


def _task_example_assets_valid(example: Mapping[str, Any]) -> bool:
    form = example.get("final_expected_form")
    digests = example.get("asset_sha256")
    if not isinstance(form, Mapping) or not isinstance(digests, Mapping) or not digests:
        return False
    return all(
        isinstance(field, str) and field in form and _task_example_asset_path(example, field) is not None
        for field in digests
    )


def _task_example_asset_path(
    example: Mapping[str, Any] | None,
    field: str,
    alias: str | None = None,
) -> Path | None:
    if not isinstance(example, Mapping):
        return None
    form = example.get("final_expected_form")
    digests = example.get("asset_sha256")
    if not isinstance(form, Mapping) or not isinstance(digests, Mapping):
        return None
    value = form.get(field)
    digest = digests.get(field)
    if value is None and alias is not None:
        value = form.get(alias)
        digest = digests.get(alias)
    path = _task_example_form_path(value)
    if path is None or not isinstance(digest, str):
        return None
    expected = digest.strip().casefold()
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", expected):
        return None
    return path if adapter_source_digest(path) == expected else None


def _task_example_form_path(value: Any) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    path = (TASK_EXAMPLE_DIR / value).resolve()
    try:
        path.relative_to(TASK_EXAMPLE_DIR.resolve())
    except ValueError:
        return None
    return path if path.is_file() else None


def _task_example_record_type_matches(example_record_type: str, record_type: str) -> bool:
    example_type = str(example_record_type or "").casefold()
    target_type = str(record_type or "").casefold()
    if example_type == target_type:
        return True
    return example_type == "generated_contract" and target_type not in {"", "unknown", "segmentation", "classification"}
