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

"""Client-local representative-data preflight, visual inspection, and QC rendering."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from fedready.agents.client.adapter import (
    ADAPTER_PREFLIGHT_STATE_FILE,
    UNLIMITED_LOCAL_ADAPTER_RECORDS,
    AdapterRuntimeResult,
    ClientDataProvenanceSnapshot,
    execute_local_adapter,
    read_adapter_state,
    resolve_adapter_record_path,
    result_from_state,
    state_attests_pass,
    write_adapter_state,
)
from fedready.agents.client.local_vlm import is_local_http_url
from fedready.agents.common import adapter_source_digest
from fedready.agents.references import (
    _public_task_example,
    _select_task_example,
    _task_example_asset_path,
    _task_example_record_type_matches,
    adapter_preflight_task_example_path,
)
from fedready.data.contracts import generated_contract_record_type, manifest_record_type
from fedready.data.contracts.base import GENERATED_DATA_CONTRACT_SCHEMA_VERSION
from fedready.prompts import render_client_prompt
from PIL import Image

ADAPTER_PREFLIGHT_QC_STATE_FILE = "fedready_adapter_preflight_qc_state.json"
ADAPTER_PREFLIGHT_QC_REFERENCE_FILE = "adapter_preflight_reference.png"
DEFAULT_LOCAL_ADAPTER_PREFLIGHT_RECORDS = 8
ADAPTER_PREFLIGHT_QC_MAX_LONG_SIDE = 768
ADAPTER_PREFLIGHT_RAW_MAX_LONG_SIDE = 768
ADAPTER_PREFLIGHT_QC_REVIEW_COUNT = 1
ADAPTER_PREFLIGHT_VLM_CONSENSUS_COUNT = 2


def ensure_local_adapter_pipeline(
    *,
    workspace: Path,
    local_data_path: Path,
    task: str,
    max_records: int | None,
    min_records: int,
    timeout_seconds: int,
    local_vlm_model: str,
    local_vlm_base_url: str,
    local_vlm_api_key_env: str,
    local_vlm_max_tokens: int,
    query_image: Callable[..., tuple[str, dict[str, Any]]],
    reuse_existing_attestations: bool = False,
    provenance_snapshot: ClientDataProvenanceSnapshot | None = None,
    data_contract: Mapping[str, Any] | None = None,
) -> AdapterRuntimeResult:
    """Execute the same bounded preflight/QC pipeline for any backend.

    The client acceptance boundary leaves ``reuse_existing_attestations`` false
    because an agent with workspace write access must not be able to mint its
    own runtime verdict. Reuse is available only to trusted harness callers.
    """

    workspace = workspace.resolve()
    local_data_path = local_data_path.resolve()
    if provenance_snapshot is not None and not provenance_snapshot.complete:
        return AdapterRuntimeResult(
            stage="pre_agent_provenance",
            status="failed",
            diagnostic=provenance_snapshot.diagnostic,
        )
    bounded_max = max_records if max_records is not None else UNLIMITED_LOCAL_ADAPTER_RECORDS
    bounded_min = min(max(1, min_records), bounded_max)
    script_path = workspace / "adapter.py"
    if not script_path.is_file():
        return AdapterRuntimeResult(
            stage="representative_sample_preflight",
            status="failed",
            diagnostic="adapter.py does not exist in the adapter workspace",
        )
    source_digest = adapter_source_digest(script_path)
    preflight_state = read_adapter_state(workspace / ADAPTER_PREFLIGHT_STATE_FILE)
    if not reuse_existing_attestations or not state_attests_pass(
        preflight_state,
        source_digest=source_digest,
    ):
        preflight = execute_local_adapter(
            workspace=workspace,
            local_data_path=local_data_path,
            max_records=min(DEFAULT_LOCAL_ADAPTER_PREFLIGHT_RECORDS, bounded_max),
            min_records=min(bounded_min, DEFAULT_LOCAL_ADAPTER_PREFLIGHT_RECORDS),
            timeout_seconds=timeout_seconds,
            manifest_name="adapter_preflight_manifest.json",
            output_dir_name="preflight_generated_labels",
            state_file_name=ADAPTER_PREFLIGHT_STATE_FILE,
            stage="representative_sample_preflight",
            provenance_snapshot=provenance_snapshot,
            data_contract=data_contract,
        )
        if not preflight.passed:
            return preflight

    qc_state = read_adapter_state(workspace / ADAPTER_PREFLIGHT_QC_STATE_FILE)
    if not reuse_existing_attestations or not state_attests_pass(
        qc_state,
        source_digest=source_digest,
    ):
        qc = review_local_adapter_preflight(
            workspace=workspace,
            task=task,
            model=local_vlm_model,
            base_url=local_vlm_base_url,
            api_key_env=local_vlm_api_key_env,
            max_tokens=local_vlm_max_tokens,
            query_image=query_image,
            data_contract=data_contract,
        )
        if not qc.passed:
            return qc

    latest_qc_state = read_adapter_state(workspace / ADAPTER_PREFLIGHT_QC_STATE_FILE)
    return result_from_state(latest_qc_state, default_stage="representative_sample_visual_qc")


def _generated_contract_defers_adapter_visual_qc(
    data_contract: Mapping[str, Any] | None,
) -> bool:
    if not isinstance(data_contract, Mapping):
        return False
    record_type = generated_contract_record_type(data_contract)
    return data_contract.get("schema_version") == GENERATED_DATA_CONTRACT_SCHEMA_VERSION and record_type not in {
        "unknown",
        "segmentation",
        "classification",
    }


def review_local_adapter_preflight(
    *,
    workspace: Path,
    task: str,
    model: str,
    base_url: str,
    api_key_env: str,
    max_tokens: int,
    query_image: Callable[..., tuple[str, dict[str, Any]]],
    data_contract: Mapping[str, Any] | None = None,
) -> AdapterRuntimeResult:
    """Ask the local VLM to judge one adapter-selected representative record."""

    workspace = workspace.resolve()
    script_path = workspace / "adapter.py"
    source_digest = adapter_source_digest(script_path) if script_path.is_file() else None
    preflight_state = read_adapter_state(workspace / ADAPTER_PREFLIGHT_STATE_FILE)
    if source_digest is None or not state_attests_pass(preflight_state, source_digest=source_digest):
        result = AdapterRuntimeResult(
            stage="representative_sample_visual_qc",
            status="failed",
            diagnostic="client-local VLM review requires a passed executable preflight",
            adapter_sha256=source_digest,
            issue_codes=("preflight_not_passed",),
        )
        write_adapter_state(workspace / ADAPTER_PREFLIGHT_QC_STATE_FILE, result.as_state())
        return result
    if not is_local_http_url(base_url):
        result = AdapterRuntimeResult(
            stage="representative_raw_image_review",
            status="failed",
            diagnostic="client-local VLM is unavailable",
            adapter_sha256=source_digest,
            issue_codes=("local_vlm_unavailable",),
        )
        write_adapter_state(workspace / ADAPTER_PREFLIGHT_QC_STATE_FILE, result.as_state())
        return result

    try:
        manifest_path = workspace / "adapter_preflight_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        records = manifest.get("records")
        if not isinstance(records, list) or not records:
            raise ValueError("preflight manifest does not contain sample records")
        record_type = manifest_record_type(manifest, generated_contract=data_contract)
        selected = _adapter_preflight_qc_records(records, review_count=ADAPTER_PREFLIGHT_QC_REVIEW_COUNT)
        fallback_selected = _adapter_preflight_qc_records(records, review_count=2)[1] if len(records) > 1 else None

        for stale_path in workspace.glob("adapter_preflight_raw*.png"):
            stale_path.unlink(missing_ok=True)
        for stale_path in workspace.glob("adapter_preflight_qc*.png"):
            stale_path.unlink(missing_ok=True)
        (workspace / ADAPTER_PREFLIGHT_QC_REFERENCE_FILE).unlink(missing_ok=True)

        example_image = adapter_preflight_task_example_path(record_type=record_type)
        if example_image is None:
            result = AdapterRuntimeResult(
                stage="representative_raw_image_review",
                status="failed",
                diagnostic="canonical task_example raw reference is unavailable or failed integrity validation",
                adapter_sha256=source_digest,
                issue_codes=("task_example_reference_unavailable",),
                record_count=len(records),
            )
            state = result.as_state()
            state.update(
                {
                    "raw_reference_available": False,
                    "raw_reference_source": "task_example_final_expected_form",
                }
            )
            write_adapter_state(workspace / ADAPTER_PREFLIGHT_QC_STATE_FILE, state)
            return result
        raw_reference_path = workspace / "adapter_preflight_raw_reference.png"
        write_adapter_preflight_raw_image(
            image_path=example_image,
            output_path=raw_reference_path,
        )
        raw_reference_sha256 = adapter_source_digest(example_image)

        raw_reviews: list[dict[str, Any]] = []
        for review_index, record_index, record in selected:
            image_path = resolve_adapter_record_path(
                record.get("image_path") or record.get("image"),
                manifest_path.parent,
            )
            raw_path = workspace / f"adapter_preflight_raw_{review_index + 1:02d}.png"
            write_adapter_preflight_raw_image(
                image_path=image_path,
                output_path=raw_path,
            )
            query_kwargs = {
                "base_url": base_url,
                "model": model,
                "image_path": raw_path,
                "reference_image_path": raw_reference_path,
                "prompt": adapter_preflight_raw_vlm_prompt(task),
                "api_key_env": api_key_env,
                "max_tokens": max_tokens,
                "log_path": workspace / "local_vlm_adapter_preflight_raw_calls.jsonl",
            }
            fallback_query_kwargs = None
            fallback_record_index = None
            if fallback_selected is not None:
                _, fallback_record_index, fallback_record = fallback_selected
                fallback_image_path = resolve_adapter_record_path(
                    fallback_record.get("image_path") or fallback_record.get("image"),
                    manifest_path.parent,
                )
                fallback_raw_path = workspace / "adapter_preflight_raw_02.png"
                write_adapter_preflight_raw_image(
                    image_path=fallback_image_path,
                    output_path=fallback_raw_path,
                )
                fallback_query_kwargs = {**query_kwargs, "image_path": fallback_raw_path}
            review_status, review_issue_codes, verdicts = _query_adapter_vlm_consensus(
                query_image=query_image,
                parse_response=parse_adapter_preflight_raw_vlm_response,
                query_kwargs=query_kwargs,
                fallback_query_kwargs=fallback_query_kwargs,
                fallback_on_consistent_failure=True,
            )
            fallback_used = any(verdict["artifact_role"] == "fallback" for verdict in verdicts)
            raw_reviews.append(
                {
                    "review_index": review_index,
                    "record_index": record_index,
                    "fallback_record_index": fallback_record_index if fallback_used else None,
                    "status": review_status,
                    "issue_codes": review_issue_codes,
                    "vlm_call_count": len(verdicts),
                    "reviewed_sample_count": 2 if fallback_used else 1,
                    "verdicts": verdicts,
                }
            )

        raw_status = str(raw_reviews[0]["status"])
        raw_issue_codes = [str(code) for code in raw_reviews[0]["issue_codes"]]
        raw_passed = int(raw_status == "passed")
        raw_required = 1
        if raw_status != "passed":
            result = AdapterRuntimeResult(
                stage="representative_raw_image_review",
                status="failed",
                diagnostic="client-local VLM rejected the representative raw input: " + ", ".join(raw_issue_codes),
                adapter_sha256=source_digest,
                issue_codes=tuple(raw_issue_codes),
                record_count=len(records),
            )
            state = result.as_state()
            state.update(
                {
                    "raw_review_count": sum(int(review["reviewed_sample_count"]) for review in raw_reviews),
                    "raw_passed_review_count": raw_passed,
                    "raw_required_pass_count": raw_required,
                    "raw_reference_available": True,
                    "raw_reference_source": "task_example_final_expected_form",
                    "raw_reference_sha256": raw_reference_sha256,
                    "raw_reviews": raw_reviews,
                }
            )
            write_adapter_state(workspace / ADAPTER_PREFLIGHT_QC_STATE_FILE, state)
            return result

        common_state = {
            "raw_review_count": sum(int(review["reviewed_sample_count"]) for review in raw_reviews),
            "raw_passed_review_count": raw_passed,
            "raw_required_pass_count": raw_required,
            "raw_vlm_call_count": sum(int(review["vlm_call_count"]) for review in raw_reviews),
            "raw_vlm_consensus_required_count": ADAPTER_PREFLIGHT_VLM_CONSENSUS_COUNT,
            "raw_reference_available": True,
            "raw_reference_source": "task_example_final_expected_form",
            "raw_reference_sha256": raw_reference_sha256,
            "raw_reviews": raw_reviews,
        }
        if record_type == "classification":
            result = AdapterRuntimeResult(
                stage="representative_sample_visual_qc",
                status="passed",
                diagnostic="client-local VLM accepted the representative raw classification input",
                adapter_sha256=source_digest,
                issue_codes=("raw_image_review_passed",),
                record_count=len(records),
            )
            write_adapter_state(
                workspace / ADAPTER_PREFLIGHT_QC_STATE_FILE,
                {**result.as_state(), **common_state},
            )
            return result

        if _generated_contract_defers_adapter_visual_qc(data_contract):
            result = AdapterRuntimeResult(
                stage="representative_sample_visual_qc",
                status="passed",
                diagnostic=(
                    "client-local VLM accepted the representative raw input; "
                    "label-specific visual QC is deferred to materializer-owned artifacts"
                ),
                adapter_sha256=source_digest,
                issue_codes=(
                    "raw_image_review_passed",
                    "deferred_to_generated_materializer_visual_qc",
                ),
                record_count=len(records),
            )
            write_adapter_state(
                workspace / ADAPTER_PREFLIGHT_QC_STATE_FILE,
                {
                    **result.as_state(),
                    **common_state,
                    "visual_qc_owner": "generated_materializer",
                },
            )
            return result

        reference_qc_path = workspace / ADAPTER_PREFLIGHT_QC_REFERENCE_FILE
        reference_example = write_adapter_preflight_task_example_reference(
            record_type=record_type,
            output_path=reference_qc_path,
            data_contract=data_contract,
        )
        if reference_example is None:
            result = AdapterRuntimeResult(
                stage="representative_sample_visual_qc",
                status="failed",
                diagnostic="canonical task_example label reference is unavailable or failed integrity validation",
                adapter_sha256=source_digest,
                issue_codes=("task_example_reference_unavailable",),
                record_count=len(records),
            )
            write_adapter_state(
                workspace / ADAPTER_PREFLIGHT_QC_STATE_FILE,
                {
                    **result.as_state(),
                    **common_state,
                    "reference_source": "task_example_final_expected_form",
                },
            )
            return result

        reviews: list[dict[str, Any]] = []
        for review_index, record_index, record in selected:
            qc_path = _adapter_preflight_qc_path(workspace, review_index)
            write_adapter_preflight_qc_record(
                record=record,
                record_type=record_type,
                manifest_path=manifest_path,
                workspace=workspace,
                output_path=qc_path,
                data_contract=data_contract,
                record_index=record_index,
            )
            query_kwargs = {
                "base_url": base_url,
                "model": model,
                "image_path": qc_path,
                "reference_image_path": reference_qc_path,
                "prompt": adapter_preflight_vlm_prompt(task),
                "api_key_env": api_key_env,
                "max_tokens": max_tokens,
                "log_path": workspace / "local_vlm_adapter_preflight_calls.jsonl",
            }
            fallback_query_kwargs = None
            fallback_record_index = None
            fallback_qc_path = None
            if fallback_selected is not None:
                _, fallback_record_index, fallback_record = fallback_selected
                fallback_qc_path = _adapter_preflight_qc_path(workspace, 1)
                write_adapter_preflight_qc_record(
                    record=fallback_record,
                    record_type=record_type,
                    manifest_path=manifest_path,
                    workspace=workspace,
                    output_path=fallback_qc_path,
                    data_contract=data_contract,
                    record_index=fallback_record_index,
                )
                fallback_query_kwargs = {**query_kwargs, "image_path": fallback_qc_path}
            review_status, review_issue_codes, verdicts = _query_adapter_vlm_consensus(
                query_image=query_image,
                parse_response=parse_adapter_preflight_vlm_response,
                query_kwargs=query_kwargs,
                fallback_query_kwargs=fallback_query_kwargs,
                fallback_on_consistent_failure=True,
            )
            fallback_used = any(verdict["artifact_role"] == "fallback" for verdict in verdicts)
            reviews.append(
                {
                    "review_index": review_index,
                    "record_index": record_index,
                    "fallback_record_index": fallback_record_index if fallback_used else None,
                    "status": review_status,
                    "issue_codes": review_issue_codes,
                    "qc_image": qc_path.name,
                    "fallback_qc_image": (
                        fallback_qc_path.name if fallback_used and fallback_qc_path is not None else None
                    ),
                    "vlm_call_count": len(verdicts),
                    "reviewed_sample_count": 2 if fallback_used else 1,
                    "verdicts": verdicts,
                }
            )

        status = str(reviews[0]["status"])
        issue_codes = [str(code) for code in reviews[0]["issue_codes"]]
        passed_count = int(status == "passed")
        required_count = 1
        diagnostic = (
            "client-local VLM accepted the representative raw input and label rendering"
            if status == "passed"
            else "client-local VLM rejected the representative label rendering: " + ", ".join(issue_codes)
        )
        result = AdapterRuntimeResult(
            stage="representative_sample_visual_qc",
            status=status,
            diagnostic=diagnostic,
            adapter_sha256=source_digest,
            issue_codes=tuple(issue_codes),
            record_count=len(records),
        )
        state = result.as_state()
        state.update(
            {
                **common_state,
                "review_count": sum(int(review["reviewed_sample_count"]) for review in reviews),
                "passed_review_count": passed_count,
                "required_pass_count": required_count,
                "vlm_call_count": sum(int(review["vlm_call_count"]) for review in reviews),
                "vlm_consensus_required_count": ADAPTER_PREFLIGHT_VLM_CONSENSUS_COUNT,
                "reference_record_index": None,
                "reference_qc_image": reference_qc_path.name,
                "reference_source": "task_example_final_expected_form",
                "reference_example": reference_example,
                "reviews": reviews,
            }
        )
        write_adapter_state(workspace / ADAPTER_PREFLIGHT_QC_STATE_FILE, state)
        return result
    except ValueError as exc:
        result = AdapterRuntimeResult(
            stage="representative_sample_visual_qc",
            status="failed",
            diagnostic="client-local adapter review could not prepare its artifact: " + str(exc),
            adapter_sha256=source_digest,
            issue_codes=("visual_qc_renderer_unavailable",),
        )
    except Exception:
        result = AdapterRuntimeResult(
            stage="representative_sample_visual_qc",
            status="failed",
            diagnostic="client-local VLM review could not be completed",
            adapter_sha256=source_digest,
            issue_codes=("local_vlm_unavailable",),
        )
    write_adapter_state(workspace / ADAPTER_PREFLIGHT_QC_STATE_FILE, result.as_state())
    return result


def _adapter_preflight_qc_path(workspace: Path, review_index: int) -> Path:
    if review_index == 0:
        return workspace / "adapter_preflight_qc.png"
    return workspace / f"adapter_preflight_qc_{review_index + 1:02d}.png"


def write_adapter_preflight_task_example_reference(
    *,
    record_type: str,
    output_path: Path,
    data_contract: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    del data_contract
    example = _select_task_example(record_type=record_type)
    if example is None:
        return None
    image_path = _task_example_asset_path(example, "image_path", "image")
    if image_path is None:
        return None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    example_record_type = str(example.get("record_type") or record_type)
    if _task_example_record_type_matches(example_record_type, "segmentation"):
        mask_path = _task_example_asset_path(example, "mask_path", "mask")
        if mask_path is None:
            return None
        write_adapter_preflight_qc_image(
            image_path=image_path,
            mask_path=mask_path,
            output_path=output_path,
        )
    else:
        return None
    public = _public_task_example(example)
    public["rendered_reference_image"] = output_path.name
    return public


def write_adapter_preflight_qc_record(
    *,
    record: Mapping[str, Any],
    record_type: str,
    manifest_path: Path,
    workspace: Path,
    output_path: Path,
    data_contract: Mapping[str, Any] | None,
    record_index: int,
) -> None:
    image_path = resolve_adapter_record_path(record.get("image_path"), manifest_path.parent)
    if record_type == "segmentation":
        mask_path = resolve_adapter_record_path(record.get("mask_path"), manifest_path.parent)
        write_adapter_preflight_qc_image(
            image_path=image_path,
            mask_path=mask_path,
            output_path=output_path,
        )
    else:
        raise ValueError("active data contract does not have a built-in visual-QC renderer")


def _adapter_preflight_qc_records(
    records: Sequence[Any], *, review_count: int
) -> list[tuple[int, int, Mapping[str, Any]]]:
    if review_count < 1:
        raise ValueError("preflight QC review_count must be positive")
    if not records:
        raise ValueError("preflight manifest does not contain sample records")
    if len(records) >= review_count:
        if review_count == 1:
            indices = [0]
        else:
            indices = []
            last_index = len(records) - 1
            for index in range(review_count):
                candidate = round(index * last_index / (review_count - 1))
                if candidate not in indices:
                    indices.append(candidate)
            for candidate in range(len(records)):
                if len(indices) >= review_count:
                    break
                if candidate not in indices:
                    indices.append(candidate)
    else:
        indices = [index % len(records) for index in range(review_count)]

    selected: list[tuple[int, int, Mapping[str, Any]]] = []
    for review_index, record_index in enumerate(indices):
        record = records[record_index]
        if not isinstance(record, Mapping):
            raise ValueError(f"preflight manifest record {record_index} must be an object")
        selected.append((review_index, record_index, record))
    return selected


def write_adapter_preflight_raw_image(*, image_path: Path, output_path: Path) -> None:
    """Write an unannotated, bounded copy for the local guardrail agent."""

    with Image.open(image_path) as source:
        image = source.convert("RGB")
    longest_side = max(image.size)
    if longest_side > ADAPTER_PREFLIGHT_RAW_MAX_LONG_SIDE:
        scale = ADAPTER_PREFLIGHT_RAW_MAX_LONG_SIDE / longest_side
        image = image.resize(
            (
                max(1, round(image.width * scale)),
                max(1, round(image.height * scale)),
            ),
            Image.Resampling.LANCZOS,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, format="PNG")


def write_adapter_preflight_qc_image(*, image_path: Path, mask_path: Path, output_path: Path) -> None:
    with Image.open(image_path) as source:
        image = source.convert("RGBA")
    with Image.open(mask_path) as source_mask:
        mask = source_mask.convert("L")
    if image.size != mask.size:
        raise ValueError("preflight QC image and mask dimensions do not match")
    longest_side = max(image.size)
    if longest_side > ADAPTER_PREFLIGHT_QC_MAX_LONG_SIDE:
        scale = ADAPTER_PREFLIGHT_QC_MAX_LONG_SIDE / longest_side
        resized = (
            max(1, round(image.width * scale)),
            max(1, round(image.height * scale)),
        )
        image = image.resize(resized, Image.Resampling.LANCZOS)
        mask = mask.resize(resized, Image.Resampling.NEAREST)
    alpha = mask.point(lambda value: 150 if value > 0 else 0)
    red = Image.new("RGBA", image.size, (255, 0, 0, 0))
    red.putalpha(alpha)
    Image.alpha_composite(image, red).convert("RGB").save(output_path, format="PNG")


def adapter_preflight_vlm_prompt(task: str) -> str:
    return render_client_prompt("adapter_preflight_vlm", task=task)


def adapter_preflight_raw_vlm_prompt(task: str) -> str:
    return render_client_prompt("adapter_preflight_raw_vlm", task=task)


_ADAPTER_PREFLIGHT_RAW_VLM_ISSUE_CODES = frozenset(
    {
        "none",
        "unreadable",
        "annotation_burned_in",
        "task_mismatch",
        "unsuitable_raw_input",
    }
)


def _query_adapter_vlm_consensus(
    *,
    query_image: Callable[..., tuple[str, dict[str, Any]]],
    parse_response: Callable[[str], tuple[str, list[str]]],
    query_kwargs: Mapping[str, Any],
    fallback_query_kwargs: Mapping[str, Any] | None = None,
    fallback_on_consistent_failure: bool = False,
) -> tuple[str, list[str], list[dict[str, Any]]]:
    """Confirm one sample, with a bounded second-sample ambiguity check."""

    verdicts: list[dict[str, Any]] = []

    def review_artifact(active_query_kwargs: Mapping[str, Any], *, artifact_role: str) -> list[dict[str, Any]]:
        artifact_verdicts: list[dict[str, Any]] = []
        for _ in range(ADAPTER_PREFLIGHT_VLM_CONSENSUS_COUNT):
            response_text, _ = query_image(**dict(active_query_kwargs))
            status, issue_codes = parse_response(response_text)
            verdict = {
                "attempt": len(verdicts) + 1,
                "artifact_role": artifact_role,
                "status": status,
                "issue_codes": issue_codes,
            }
            verdicts.append(verdict)
            artifact_verdicts.append(verdict)
        return artifact_verdicts

    primary_verdicts = review_artifact(query_kwargs, artifact_role="primary")
    if all(verdict["status"] == "passed" for verdict in primary_verdicts):
        return "passed", ["none"], verdicts

    primary_conflicted = len({verdict["status"] for verdict in primary_verdicts}) > 1
    if fallback_query_kwargs is not None and (primary_conflicted or fallback_on_consistent_failure):
        fallback_verdicts = review_artifact(fallback_query_kwargs, artifact_role="fallback")
        if all(verdict["status"] == "passed" for verdict in fallback_verdicts):
            return "passed", ["none"], verdicts
        deciding_verdicts = fallback_verdicts
    else:
        deciding_verdicts = primary_verdicts

    failed_codes = [
        code
        for verdict in deciding_verdicts
        if verdict["status"] == "failed"
        for code in verdict["issue_codes"]
        if code != "none"
    ]
    return "failed", list(dict.fromkeys(failed_codes)) or ["unreadable"], verdicts


def parse_adapter_preflight_raw_vlm_response(
    response_text: str,
) -> tuple[str, list[str]]:
    return _parse_adapter_vlm_response(
        response_text,
        allowed_issue_codes=_ADAPTER_PREFLIGHT_RAW_VLM_ISSUE_CODES,
    )


_ADAPTER_PREFLIGHT_VLM_ISSUE_CODES = frozenset(
    {"none", "undercoverage", "spillover", "misalignment", "wrong_target", "unreadable"}
)


def parse_adapter_preflight_vlm_response(response_text: str) -> tuple[str, list[str]]:
    return _parse_adapter_vlm_response(
        response_text,
        allowed_issue_codes=_ADAPTER_PREFLIGHT_VLM_ISSUE_CODES,
    )


def _parse_adapter_vlm_response(
    response_text: str,
    *,
    allowed_issue_codes: frozenset[str],
) -> tuple[str, list[str]]:
    payload: Any = None
    try:
        payload = json.loads(response_text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*?\}", response_text, flags=re.DOTALL)
        if match:
            try:
                payload = json.loads(match.group(0))
            except json.JSONDecodeError:
                payload = None
    if not isinstance(payload, dict):
        return "failed", ["unreadable"]
    status = {
        "pass": "passed",
        "passed": "passed",
        "fail": "failed",
        "failed": "failed",
    }.get(str(payload.get("status") or "").strip().lower(), "failed")
    raw_codes = payload.get("issue_codes")
    codes = (
        [str(code).strip().lower() for code in raw_codes if str(code).strip().lower() in allowed_issue_codes]
        if isinstance(raw_codes, list)
        else []
    )
    codes = list(dict.fromkeys(codes))
    if status == "passed" and (not codes or codes == ["none"]):
        return "passed", ["none"]
    return "failed", [code for code in codes if code != "none"] or ["unreadable"]
