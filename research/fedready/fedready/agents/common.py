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

"""Shared agent turns, safe payload helpers, and adapter request metadata."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fedready.data.contracts import CLASSIFICATION


@dataclass(frozen=True)
class AgentTurn:
    """Prompt and structured output for an agent step."""

    prompt: str
    output: dict[str, Any]
    guardrail: dict[str, Any] | None = None


def _copy_present_keys(source: dict[str, Any], keys: list[str]) -> dict[str, Any]:
    return {key: source[key] for key in keys if key in source}


def _raise_if_template_passthrough(*, action: str, output: dict[str, Any], template: dict[str, Any]) -> None:
    if output == template:
        raise RuntimeError(f"{action} backend returned output_template unchanged; refusing template fallback")


def _generated_materializer_agent_summary(materializer: Any) -> dict[str, Any] | None:
    if not isinstance(materializer, dict):
        return None
    summary: dict[str, Any] = {}
    for key in (
        "schema_version",
        "status",
        "record_type",
        "sample_manifest",
        "sample_manifest_format",
        "entry_script",
        "source_digest",
    ):
        value = materializer.get(key)
        if value is not None:
            summary[key] = value
    source_files = materializer.get("source_files")
    if isinstance(source_files, list):
        summary["source_file_count"] = len(source_files)
        summary["source_files_redacted"] = True
    return summary


def _summarize_generated_materializer_payload(payload: dict[str, Any]) -> dict[str, Any]:
    summarized = json.loads(json.dumps(payload, default=str))
    materializer_summary = _generated_materializer_agent_summary(summarized.get("generated_data_materializer"))
    if materializer_summary is not None:
        summarized["generated_data_materializer"] = materializer_summary
    policies = summarized.get("policies")
    if isinstance(policies, dict):
        summarized_policies: dict[str, Any] = {}
        for client_id, policy in policies.items():
            if isinstance(policy, dict):
                policy_copy = dict(policy)
                policy_summary = _generated_materializer_agent_summary(policy_copy.get("generated_data_materializer"))
                if policy_summary is not None:
                    policy_copy["generated_data_materializer"] = policy_summary
                summarized_policies[client_id] = policy_copy
            else:
                summarized_policies[client_id] = policy
        summarized["policies"] = summarized_policies
    return summarized


def _merge_output_template(output_template: dict[str, Any], output: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(output, dict):
        raise TypeError("Agent output must be a JSON object; refusing output-template fallback.")
    merged = dict(output_template)
    for key, value in output.items():
        template_value = output_template.get(key)
        if isinstance(template_value, dict) and isinstance(value, dict):
            merged[key] = _merge_output_template(template_value, value)
        else:
            merged[key] = value
    if isinstance(output_template.get("schema_version"), str):
        merged["schema_version"] = output_template["schema_version"]
    return merged


DEFAULT_LOCAL_ADAPTER_MAX_ATTEMPTS = 5


def adapter_source_digest(script_path: Path) -> str:
    return f"sha256:{hashlib.sha256(script_path.read_bytes()).hexdigest()}"


def _visual_qc_required_for_training_result(result: dict[str, Any]) -> bool:
    visual_qc = result.get("visual_qc") if isinstance(result.get("visual_qc"), dict) else {}
    if visual_qc.get("review_required") is False:
        return False
    if result.get("record_type") == CLASSIFICATION:
        return False
    extraction = result.get("extraction") if isinstance(result.get("extraction"), dict) else {}
    if extraction.get("classification_storage") is not None:
        return False
    if extraction.get("sample_manifest_format") == "monai_decathlon_datalist_json":
        return False
    if visual_qc.get("status") == "not_performed":
        reason = str(visual_qc.get("reason") or "").lower()
        if "not applicable" in reason or "not required" in reason or "classification" in reason:
            return False
    return True


def _default_visual_qc_rule() -> dict[str, Any]:
    return {
        "required": True,
        "sample_count": 3,
        "reviewer": "client_local_vlm",
        "purpose": "verify local image/label correspondence and task alignment before training use",
        "if_ambiguous": "sample_additional_cases_or_hold_client_for_review",
        "if_failed": "return qc failure and require local adapter correction",
    }
