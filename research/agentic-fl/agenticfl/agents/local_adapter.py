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

"""Backend-neutral execution contract for client-local data adapters.

Agent backends may differ in how they generate ``adapter.py``, but the harness
owns the representative real-data preflight, local visual review, deferred data-dir
execution, and manifest validation that determine whether an implementation is accepted.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from agenticfl.data.contracts import classification as classification_contract
from agenticfl.data.contracts import (
    generated_contract_field_names,
    generated_contract_record_type,
    infer_record_type,
    manifest_record_type,
)
from agenticfl.data.contracts.base import GENERATED_DATA_CONTRACT_SCHEMA_VERSION
from agenticfl.prompts import render_client_prompt
from PIL import Image, UnidentifiedImageError

ADAPTER_BROKER_STATE_FILE = "agenticfl_adapter_broker_state.json"
ADAPTER_PREFLIGHT_STATE_FILE = "agenticfl_adapter_preflight_state.json"
ADAPTER_PREFLIGHT_QC_STATE_FILE = "agenticfl_adapter_preflight_qc_state.json"
ADAPTER_PREFLIGHT_ATTEMPT_STATE_FILE = "agenticfl_adapter_preflight_attempts.json"
ADAPTER_PREFLIGHT_QC_REFERENCE_FILE = "adapter_preflight_reference.png"
_PACKAGED_TASK_EXAMPLE_DIR = Path(__file__).resolve().parent / "task_example"
_SOURCE_TASK_EXAMPLE_DIR = Path(__file__).resolve().parents[2] / "task_example"
_CONFIGURED_TASK_EXAMPLE_DIR = os.environ.get("AGENTICFL_TASK_EXAMPLE_DIR")
TASK_EXAMPLE_DIR = (
    Path(_CONFIGURED_TASK_EXAMPLE_DIR).expanduser().resolve()
    if _CONFIGURED_TASK_EXAMPLE_DIR
    else (_PACKAGED_TASK_EXAMPLE_DIR if _PACKAGED_TASK_EXAMPLE_DIR.is_dir() else _SOURCE_TASK_EXAMPLE_DIR)
)
DEFAULT_LOCAL_ADAPTER_MAX_ATTEMPTS = 5
DEFAULT_LOCAL_ADAPTER_PREFLIGHT_RECORDS = 8
ADAPTER_PREFLIGHT_QC_MAX_LONG_SIDE = 768
ADAPTER_PREFLIGHT_RAW_MAX_LONG_SIDE = 768
ADAPTER_PREFLIGHT_QC_REVIEW_COUNT = 1
ADAPTER_PREFLIGHT_VLM_CONSENSUS_COUNT = 2
UNLIMITED_LOCAL_ADAPTER_RECORDS = 2_147_483_647


@dataclass(frozen=True)
class AdapterRuntimeResult:
    """Aggregate, path-redacted result from one harness-owned adapter stage."""

    stage: str
    status: str
    diagnostic: str
    return_code: int | None = None
    record_count: int = 0
    adapter_sha256: str | None = None
    issue_codes: tuple[str, ...] = ()

    @property
    def passed(self) -> bool:
        return self.status == "passed"

    @property
    def infrastructure_failure(self) -> bool:
        return "local_vlm_unavailable" in self.issue_codes

    def as_state(self) -> dict[str, Any]:
        state: dict[str, Any] = {
            "schema_version": "agenticfl.local_adapter_runtime_state.v1",
            "stage": self.stage,
            "status": self.status,
            "return_code": self.return_code,
            "record_count": self.record_count,
            "diagnostic": self.diagnostic,
        }
        if self.adapter_sha256 is not None:
            state["adapter_sha256"] = self.adapter_sha256
        if self.issue_codes:
            state["issue_codes"] = list(self.issue_codes)
        return state


@dataclass(frozen=True)
class ClientDataProvenanceSnapshot:
    """Private pre-agent identity map for files under one client data root."""

    root: Path
    files: Mapping[str, tuple[int, int, int, int, int]]
    complete: bool
    diagnostic: str


def capture_client_data_provenance(local_data_path: Path) -> ClientDataProvenanceSnapshot:
    """Record which client files existed before agent-authored code can run.

    The snapshot stays in the client executor process and is never included in
    an agent request. File identity metadata detects newly created or replaced
    label sources without hashing or exposing the raw client files.
    """

    root = local_data_path.resolve()
    identities: dict[str, tuple[int, int, int, int, int]] = {}
    try:
        if root.is_file():
            identities["."] = _client_file_identity(root)
        elif root.is_dir():
            for candidate in root.rglob("*"):
                resolved = candidate.resolve()
                try:
                    relative = resolved.relative_to(root).as_posix()
                except ValueError:
                    continue
                identities[relative] = _client_file_identity(resolved)
        else:
            return ClientDataProvenanceSnapshot(
                root=root,
                files={},
                complete=False,
                diagnostic="client-local data root was unavailable before agent execution",
            )
    except OSError:
        return ClientDataProvenanceSnapshot(
            root=root,
            files=identities,
            complete=False,
            diagnostic="client-local provenance snapshot could not be completed",
        )
    return ClientDataProvenanceSnapshot(
        root=root,
        files=identities,
        complete=True,
        diagnostic=f"captured identity metadata for {len(identities)} pre-existing client files",
    )


def execute_local_adapter(
    *,
    workspace: Path,
    local_data_path: Path,
    max_records: int,
    min_records: int,
    timeout_seconds: int,
    manifest_name: str,
    output_dir_name: str,
    state_file_name: str,
    stage: str,
    provenance_snapshot: ClientDataProvenanceSnapshot | None = None,
    data_contract: Mapping[str, Any] | None = None,
    allowed_generated_roots: Sequence[Path] | None = None,
) -> AdapterRuntimeResult:
    """Execute one adapter stage with the common CLI and validation contract."""

    workspace = workspace.resolve()
    local_data_path = local_data_path.resolve()
    script_path = workspace / "adapter.py"
    manifest_path = workspace / manifest_name
    output_dir = workspace / output_dir_name
    source_digest = adapter_source_digest(script_path) if script_path.is_file() else None

    def result(
        *,
        status: str,
        diagnostic: str,
        return_code: int | None = None,
        record_count: int = 0,
    ) -> AdapterRuntimeResult:
        value = AdapterRuntimeResult(
            stage=stage,
            status=status,
            diagnostic=diagnostic,
            return_code=return_code,
            record_count=record_count,
            adapter_sha256=source_digest,
        )
        write_adapter_state(workspace / state_file_name, value.as_state())
        return value

    if not script_path.is_file():
        return result(
            status="failed",
            diagnostic="adapter.py does not exist in the adapter workspace",
        )
    if not local_data_path.exists():
        return result(status="failed", diagnostic="bound client-local data root is unavailable")
    try:
        unsupported_imports = unsupported_adapter_imports(script_path)
    except (OSError, SyntaxError) as exc:
        return result(
            status="failed",
            diagnostic=sanitize_adapter_diagnostic(
                f"adapter.py could not be parsed: {exc}",
                private_roots=(local_data_path, workspace),
            ),
        )
    if unsupported_imports:
        return result(
            status="failed",
            diagnostic=(
                "adapter.py imports unsupported dependencies: "
                + ", ".join(unsupported_imports)
                + "; revise it to use only Python standard library, Pillow, and NumPy"
            ),
        )

    shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.unlink(missing_ok=True)
    command = [
        sys.executable,
        str(script_path),
        "--data-root",
        str(local_data_path),
        "--output-dir",
        str(output_dir),
        "--manifest-path",
        str(manifest_path),
        "--max-records",
        str(max_records),
    ]
    try:
        completed = subprocess.run(
            command,
            cwd=workspace,
            env=_adapter_subprocess_environment(workspace),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return result(
            status="failed",
            diagnostic=f"adapter execution exceeded {timeout_seconds} seconds",
        )
    diagnostic = sanitize_adapter_diagnostic(
        "\n".join(part for part in (completed.stdout, completed.stderr) if part),
        private_roots=(local_data_path, workspace),
    )
    if completed.returncode != 0:
        return result(
            status="failed",
            return_code=completed.returncode,
            diagnostic=diagnostic or "adapter.py exited with a nonzero status",
        )
    try:
        record_count = validate_adapter_manifest(
            manifest_path=manifest_path,
            local_data_path=local_data_path,
            workspace=workspace,
            max_records=max_records,
            provenance_snapshot=provenance_snapshot,
            data_contract=data_contract,
            allowed_generated_roots=allowed_generated_roots,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return result(
            status="failed",
            return_code=completed.returncode,
            diagnostic=sanitize_adapter_diagnostic(
                str(exc),
                private_roots=(local_data_path, workspace),
            ),
        )
    if record_count < min_records:
        return result(
            status="failed",
            return_code=completed.returncode,
            record_count=record_count,
            diagnostic=f"adapter manifest contains {record_count} records; at least {min_records} are required",
        )
    return result(
        status="passed",
        return_code=completed.returncode,
        record_count=record_count,
        diagnostic=diagnostic or f"{stage} adapter execution and manifest validation passed",
    )


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
        record_type = adapter_manifest_record_type(manifest, data_contract=data_contract)
        selected = _adapter_preflight_qc_records(records, review_count=ADAPTER_PREFLIGHT_QC_REVIEW_COUNT)
        fallback_selected = _adapter_preflight_qc_records(records, review_count=2)[1] if len(records) > 1 else None

        for stale_path in workspace.glob("adapter_preflight_raw*.png"):
            stale_path.unlink(missing_ok=True)
        for stale_path in workspace.glob("adapter_preflight_qc*.png"):
            stale_path.unlink(missing_ok=True)
        (workspace / ADAPTER_PREFLIGHT_QC_REFERENCE_FILE).unlink(missing_ok=True)

        example_image = adapter_preflight_task_example_path(task=task, record_type=record_type)
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
            task=task,
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


def adapter_preflight_task_example_path(*, task: str, record_type: str) -> Path | None:
    example = _select_task_example(task=task, record_type=record_type)
    return _task_example_asset_path(example, "image_path", "image")


def task_example_image_path_for_task(task: str | None = None) -> Path | None:
    """Return the best matching canonical task example image, if one exists."""

    example = _select_task_example_for_task(task=task)
    return _task_example_asset_path(example, "image_path", "image")


def task_example_image_label_paths_for_task(task: str | None = None) -> dict[str, Any] | None:
    """Return the matched canonical image and its validated label asset."""

    example = _select_task_example_for_task(task=task)
    image_path = _task_example_asset_path(example, "image_path", "image")
    if image_path is None or not isinstance(example, Mapping):
        return None
    for label_kind, field in (
        ("mask", "mask_path"),
        ("annotation", "annotation_path"),
        ("class_label", "label_path"),
    ):
        label_path = _task_example_asset_path(example, field)
        if label_path is not None:
            return {
                "task_key": str(example.get("task_key") or ""),
                "record_type": str(example.get("record_type") or ""),
                "image_path": image_path,
                "label_path": label_path,
                "label_kind": label_kind,
            }
    return None


def task_example_context_for_task(task: str | None = None) -> dict[str, Any]:
    manifest = _load_task_example_manifest()
    examples = manifest.get("examples") if isinstance(manifest, Mapping) else None
    safe_examples: list[dict[str, Any]] = []
    task_text = str(task or "").casefold()
    for example in examples if isinstance(examples, list) else []:
        if not isinstance(example, Mapping):
            continue
        if not _task_example_assets_valid(example):
            continue
        public = _public_task_example(example)
        if task_text:
            score = _task_example_match_score(
                task_text,
                " ".join(str(example.get(key) or "") for key in ("task_key", "task", "description")).casefold(),
            )
            if score <= 0:
                continue
            public["match_score"] = score
        safe_examples.append(public)
    if task_text:
        safe_examples.sort(key=lambda value: int(value.get("match_score") or 0), reverse=True)
    return {
        "schema_version": "agenticfl.task_example_context.v1",
        "root": "task_example",
        "storage": "canonical_final_expected_form",
        "examples": safe_examples,
        "safe_to_share": True,
    }


def write_adapter_preflight_task_example_reference(
    *,
    task: str,
    record_type: str,
    output_path: Path,
    data_contract: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    del data_contract
    example = _select_task_example(task=task, record_type=record_type)
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


def _load_task_example_manifest() -> dict[str, Any]:
    manifest_path = TASK_EXAMPLE_DIR / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return manifest if isinstance(manifest, dict) else {}


def _select_task_example_for_task(*, task: str | None) -> Mapping[str, Any] | None:
    manifest = _load_task_example_manifest()
    examples = manifest.get("examples") if isinstance(manifest, Mapping) else None
    task_text = str(task or "").casefold()
    best: tuple[int, Mapping[str, Any]] | None = None
    fallback: Mapping[str, Any] | None = None
    for example in examples if isinstance(examples, list) else []:
        if not isinstance(example, Mapping):
            continue
        if not _task_example_assets_valid(example):
            continue
        if fallback is None:
            fallback = example
        haystack = " ".join(str(example.get(key) or "") for key in ("task_key", "task", "description")).casefold()
        score = _task_example_match_score(task_text, haystack)
        if score <= 0:
            continue
        if best is None or score > best[0]:
            best = (score, example)
    if best is not None:
        return best[1]
    return fallback if not task_text else None


def _select_task_example(*, task: str, record_type: str) -> Mapping[str, Any] | None:
    manifest = _load_task_example_manifest()
    examples = manifest.get("examples") if isinstance(manifest, Mapping) else None
    task_text = str(task or "").casefold()
    best: tuple[int, Mapping[str, Any]] | None = None
    for example in examples if isinstance(examples, list) else []:
        if not isinstance(example, Mapping):
            continue
        if not _task_example_record_type_matches(str(example.get("record_type") or ""), record_type):
            continue
        if not _task_example_assets_valid(example):
            continue
        haystack = " ".join(str(example.get(key) or "") for key in ("task_key", "task", "description")).casefold()
        score = _task_example_match_score(task_text, haystack)
        if score <= 0:
            continue
        if best is None or score > best[0]:
            best = (score, example)
    return best[1] if best is not None else None


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


def _task_example_match_score(task_text: str, haystack: str) -> int:
    score = 0
    if "cup" in task_text and "cup" in haystack:
        score += 10
    if ("disc" in task_text or "disk" in task_text) and ("disc" in haystack or "disk" in haystack):
        score += 10
    if "segmentation" in task_text and "segmentation" in haystack:
        score += 2
    if "detection" in task_text and "detection" in haystack:
        score += 2
    if "box" in task_text and "box" in haystack:
        score += 2
    if "glaucoma" in task_text and "glaucoma" in haystack:
        score += 10
    if "classification" in task_text and "classification" in haystack:
        score += 2
    if "fundus" in task_text and "fundus" in haystack:
        score += 1
    return score


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


def adapter_source_digest(script_path: Path) -> str:
    return f"sha256:{hashlib.sha256(script_path.read_bytes()).hexdigest()}"


def read_adapter_state(path: Path) -> dict[str, Any] | None:
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return state if isinstance(state, dict) else None


def write_adapter_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def state_attests_pass(state: dict[str, Any] | None, *, source_digest: str) -> bool:
    return bool(
        isinstance(state, dict) and state.get("status") == "passed" and state.get("adapter_sha256") == source_digest
    )


def result_from_state(state: dict[str, Any], *, default_stage: str) -> AdapterRuntimeResult:
    issue_codes = state.get("issue_codes")
    return AdapterRuntimeResult(
        stage=str(state.get("stage") or default_stage),
        status=str(state.get("status") or "failed"),
        diagnostic=str(state.get("diagnostic") or "adapter runtime state has no diagnostic"),
        return_code=(state.get("return_code") if isinstance(state.get("return_code"), int) else None),
        record_count=(state.get("record_count") if isinstance(state.get("record_count"), int) else 0),
        adapter_sha256=(state.get("adapter_sha256") if isinstance(state.get("adapter_sha256"), str) else None),
        issue_codes=(tuple(str(value) for value in issue_codes) if isinstance(issue_codes, list) else ()),
    )


def is_local_http_url(value: str) -> bool:
    try:
        parsed = urlsplit(value)
    except ValueError:
        return False
    return parsed.scheme == "http" and parsed.hostname in {
        "127.0.0.1",
        "localhost",
        "::1",
    }


def unsupported_adapter_imports(script_path: Path) -> list[str]:
    tree = ast.parse(script_path.read_text(encoding="utf-8"), filename=str(script_path))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".", 1)[0])
    return sorted(
        module for module in imported if module not in sys.stdlib_module_names and module not in {"PIL", "numpy"}
    )


def validate_adapter_manifest(
    *,
    manifest_path: Path,
    local_data_path: Path,
    workspace: Path,
    max_records: int,
    provenance_snapshot: ClientDataProvenanceSnapshot | None = None,
    data_contract: Mapping[str, Any] | None = None,
    allowed_generated_roots: Sequence[Path] | None = None,
) -> int:
    if not manifest_path.is_file():
        raise ValueError(f"adapter.py did not create {manifest_path.name}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError(f"{manifest_path.name} must contain a JSON object")
    if manifest.get("schema_version") != "agenticfl.local_adapter_manifest.v1":
        raise ValueError("adapter manifest schema_version must be agenticfl.local_adapter_manifest.v1")
    records = manifest.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("adapter manifest must contain non-empty records")
    if len(records) > max_records:
        raise ValueError(f"adapter manifest contains {len(records)} records, exceeding bound max_records={max_records}")
    generated_roots = _normalized_generated_roots(allowed_generated_roots)
    record_type = adapter_manifest_record_type(manifest, data_contract=data_contract)
    if record_type == "classification":
        return validate_classification_adapter_manifest_records(
            records=records,
            manifest_path=manifest_path,
            local_data_path=local_data_path,
            workspace=workspace,
            provenance_snapshot=provenance_snapshot,
            allowed_generated_roots=generated_roots,
        )
    if record_type == "segmentation":
        return validate_segmentation_adapter_manifest_records(
            records=records,
            manifest_path=manifest_path,
            local_data_path=local_data_path,
            workspace=workspace,
            provenance_snapshot=provenance_snapshot,
            allowed_generated_roots=generated_roots,
        )
    if generated_contract_record_type(data_contract) != "unknown":
        return validate_generated_contract_adapter_manifest_records(
            records=records,
            manifest_path=manifest_path,
            local_data_path=local_data_path,
            workspace=workspace,
            provenance_snapshot=provenance_snapshot,
            data_contract=data_contract,
            allowed_generated_roots=generated_roots,
        )
    raise ValueError("adapter manifest record_type must match the active built-in or generated data contract")


def adapter_manifest_record_type(
    manifest: Mapping[str, Any],
    *,
    data_contract: Mapping[str, Any] | None = None,
) -> str:
    return manifest_record_type(manifest, generated_contract=data_contract)


def _normalized_generated_roots(roots: Sequence[Path] | None) -> tuple[Path, ...]:
    if not roots:
        return ()
    normalized: list[Path] = []
    for root in roots:
        try:
            normalized.append(Path(root).resolve())
        except OSError:
            continue
    return tuple(normalized)


def _path_is_relative_to_any(path: Path, roots: Sequence[Path]) -> bool:
    return any(path.is_relative_to(root) for root in roots)


def _adapter_record_type(record: Mapping[str, Any]) -> str:
    return infer_record_type(record)


def validate_segmentation_adapter_manifest_records(
    *,
    records: Sequence[Any],
    manifest_path: Path,
    local_data_path: Path,
    workspace: Path,
    provenance_snapshot: ClientDataProvenanceSnapshot | None,
    data_contract: Mapping[str, Any] | None = None,
    allowed_generated_roots: Sequence[Path] = (),
) -> int:
    seen_image_paths: set[Path] = set()
    seen_mask_paths: set[Path] = set()
    foreground_record_count = 0
    background_record_count = 0
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"adapter manifest record {index} must be an object")
        if _adapter_record_type(record) != "segmentation":
            raise ValueError(f"adapter manifest record {index} does not match record_type=segmentation")
        image_path = resolve_adapter_record_path(record.get("image_path"), manifest_path.parent)
        label_source_path = resolve_adapter_record_path(record.get("label_source_path"), manifest_path.parent)
        mask_path = resolve_adapter_record_path(record.get("mask_path"), manifest_path.parent)
        if image_path in seen_image_paths:
            raise ValueError(f"adapter manifest record {index} image path is reused")
        if mask_path in seen_mask_paths:
            raise ValueError(f"adapter manifest record {index} mask path is reused")
        seen_image_paths.add(image_path)
        seen_mask_paths.add(mask_path)
        if not image_path.is_file():
            raise ValueError(f"adapter manifest record {index} image is not readable")
        if not label_source_path.exists():
            raise ValueError(f"adapter manifest record {index} label source is not readable")
        if not mask_path.is_file():
            raise ValueError(f"adapter manifest record {index} mask is not readable")
        if (
            not image_path.is_relative_to(local_data_path)
            and not image_path.is_relative_to(workspace)
            and not _path_is_relative_to_any(image_path, allowed_generated_roots)
        ):
            raise ValueError(f"adapter manifest record {index} image is outside allowed client-local roots")
        label_source_is_local = (
            label_source_path.is_relative_to(local_data_path)
            if local_data_path.is_dir()
            else label_source_path == local_data_path
        )
        if not label_source_is_local:
            raise ValueError(f"adapter manifest record {index} label source is outside client-local data")
        if not mask_path.is_relative_to(workspace) and not _path_is_relative_to_any(mask_path, allowed_generated_roots):
            raise ValueError(
                f"adapter manifest record {index} mask is outside the adapter workspace or generated output root"
            )
        if label_source_path in {image_path, mask_path}:
            raise ValueError(f"adapter manifest record {index} label source must differ from image and generated mask")
        if provenance_snapshot is not None:
            _require_preexisting_client_file(
                label_source_path,
                snapshot=provenance_snapshot,
                record_index=index,
            )
        has_background, has_foreground = validate_adapter_image_mask(
            image_path=image_path,
            mask_path=mask_path,
            record_index=index,
        )
        background_record_count += int(has_background)
        foreground_record_count += int(has_foreground)
    if foreground_record_count == 0:
        raise ValueError("adapter manifest masks contain no nonzero task foreground")
    if background_record_count == 0:
        raise ValueError("adapter manifest masks contain no background encoded as 0")
    return len(records)


def validate_generated_contract_adapter_manifest_records(
    *,
    records: Sequence[Any],
    manifest_path: Path,
    local_data_path: Path,
    workspace: Path,
    provenance_snapshot: ClientDataProvenanceSnapshot | None,
    data_contract: Mapping[str, Any] | None = None,
    allowed_generated_roots: Sequence[Path] = (),
) -> int:
    if generated_contract_record_type(data_contract) == "unknown":
        raise ValueError("generated adapter records require an explicit server-generated data contract")
    field_names = generated_contract_field_names(data_contract)
    required_fields = field_names["required"]
    if not required_fields:
        raise ValueError("generated data contract must declare required adapter record fields")

    path_fields = _generated_contract_path_fields(required_fields)

    seen_image_paths: set[Path] = set()
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"adapter manifest record {index} must be an object")
        missing_fields = [
            field
            for field in sorted(required_fields)
            if field not in record
            or record.get(field) is None
            or record.get(field) == ""
            or (isinstance(record.get(field), list) and not record.get(field))
        ]
        if missing_fields:
            raise ValueError(
                f"adapter manifest record {index} missing generated-contract required fields: "
                + ", ".join(missing_fields)
            )

        resolved_paths: dict[str, Path] = {}
        for field in sorted(path_fields):
            value = record.get(field)
            path = resolve_adapter_record_path(value, manifest_path.parent)
            if not path.exists():
                raise ValueError(f"adapter manifest record {index} {field} is not readable")
            resolved_paths[field] = path

        image_path = resolved_paths.get("image_path") or resolved_paths.get("image")
        if image_path is not None:
            if image_path in seen_image_paths:
                raise ValueError(f"adapter manifest record {index} image path is reused")
            seen_image_paths.add(image_path)
            if not image_path.is_file():
                raise ValueError(f"adapter manifest record {index} image is not readable")
            if (
                not image_path.is_relative_to(local_data_path)
                and not image_path.is_relative_to(workspace)
                and not _path_is_relative_to_any(image_path, allowed_generated_roots)
            ):
                raise ValueError(f"adapter manifest record {index} image is outside allowed client-local roots")

        label_source_path = resolved_paths.get("label_source_path") or resolved_paths.get("label_source")
        if label_source_path is not None:
            label_source_is_local = (
                label_source_path.is_relative_to(local_data_path)
                if local_data_path.is_dir()
                else label_source_path == local_data_path
            )
            if not label_source_is_local:
                raise ValueError(f"adapter manifest record {index} label source is outside client-local data")
            if image_path is not None and label_source_path == image_path:
                raise ValueError(f"adapter manifest record {index} label source must differ from image")
            if provenance_snapshot is not None:
                _require_preexisting_client_file(
                    label_source_path,
                    snapshot=provenance_snapshot,
                    record_index=index,
                )

    return len(records)


def _generated_contract_path_fields(required_fields: set[str]) -> set[str]:
    return {
        field
        for field in required_fields
        if field.endswith("_path") or field in {"image", "image_path", "label_source", "label_source_path"}
    }


def validate_classification_adapter_manifest_records(
    *,
    records: Sequence[Any],
    manifest_path: Path,
    local_data_path: Path,
    workspace: Path,
    provenance_snapshot: ClientDataProvenanceSnapshot | None,
    allowed_generated_roots: Sequence[Path] = (),
) -> int:
    seen_image_paths: set[Path] = set()
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"adapter manifest record {index} must be an object")
        if _adapter_record_type(record) != "classification":
            raise ValueError(f"adapter manifest record {index} does not match record_type=classification")
        image_path = resolve_adapter_record_path(record.get("image_path"), manifest_path.parent)
        label_source_path = resolve_adapter_record_path(record.get("label_source_path"), manifest_path.parent)
        _normalize_adapter_classification_label(record.get("label"), record_index=index)
        if image_path in seen_image_paths:
            raise ValueError(f"adapter manifest record {index} image path is reused")
        seen_image_paths.add(image_path)
        if not image_path.is_file():
            raise ValueError(f"adapter manifest record {index} image is not readable")
        if not label_source_path.exists():
            raise ValueError(f"adapter manifest record {index} label source is not readable")
        if (
            not image_path.is_relative_to(local_data_path)
            and not image_path.is_relative_to(workspace)
            and not _path_is_relative_to_any(image_path, allowed_generated_roots)
        ):
            raise ValueError(f"adapter manifest record {index} image is outside allowed client-local roots")
        label_source_is_local = (
            label_source_path.is_relative_to(local_data_path)
            if local_data_path.is_dir()
            else label_source_path == local_data_path
        )
        if not label_source_is_local:
            raise ValueError(f"adapter manifest record {index} label source is outside client-local data")
        if label_source_path == image_path:
            raise ValueError(f"adapter manifest record {index} label source must differ from image")
        if provenance_snapshot is not None:
            _require_preexisting_client_file(
                label_source_path,
                snapshot=provenance_snapshot,
                record_index=index,
            )
        try:
            with Image.open(image_path) as image:
                image.verify()
        except (OSError, UnidentifiedImageError) as exc:
            raise ValueError(f"adapter manifest record {index} image is not image-readable") from exc
    return len(records)


def _normalize_adapter_classification_label(value: Any, *, record_index: int) -> int:
    label = classification_contract.label_value(value)
    if label is not None:
        return label
    raise ValueError(f"adapter manifest record {record_index} classification label must be a non-negative integer")


def validate_adapter_image_mask(*, image_path: Path, mask_path: Path, record_index: int) -> tuple[bool, bool]:
    try:
        with Image.open(image_path) as image:
            image.load()
            image_size = image.size
        with Image.open(mask_path) as mask:
            mask.load()
            if mask.format != "PNG":
                raise ValueError(f"adapter manifest record {record_index} mask is not PNG-readable")
            if len(mask.getbands()) != 1:
                raise ValueError(f"adapter manifest record {record_index} mask is not single-channel")
            mask_size = mask.size
            mask_values = mask.convert("L").getcolors(maxcolors=3)
    except (OSError, UnidentifiedImageError) as exc:
        raise ValueError(f"adapter manifest record {record_index} image or mask is not image-readable") from exc
    if image_size != mask_size:
        raise ValueError(
            f"adapter manifest record {record_index} image and mask dimensions differ: "
            f"image={image_size}, mask={mask_size}"
        )
    if mask_values is None or len(mask_values) > 2:
        raise ValueError(f"adapter manifest record {record_index} mask is not binary")
    intensities = {int(value) for _, value in mask_values}
    if len(intensities) == 2 and 0 not in intensities:
        raise ValueError(f"adapter manifest record {record_index} binary mask does not encode background as 0")
    return 0 in intensities, any(value > 0 for value in intensities)


def resolve_adapter_record_path(value: Any, base: Path) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            "adapter manifest records require non-empty image_path, label_source_path, and task-specific label fields"
        )
    path = Path(value).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def sanitize_adapter_diagnostic(text: str, *, private_roots: Sequence[Path]) -> str:
    sanitized = text
    for root in private_roots:
        sanitized = sanitized.replace(str(root), "[redacted-local-root]")
    sanitized = re.sub(r"(?<![A-Za-z0-9_])/\S+", "[redacted-local-path]", sanitized)
    sanitized = re.sub(
        r"(?<![A-Za-z0-9_.-])(?:\.\.?/)?(?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.-]+",
        "[redacted-relative-path]",
        sanitized,
    )

    def redact_filename(match: re.Match[str]) -> str:
        filename = match.group(0)
        return filename if filename in {"adapter.py", "adapter_manifest.json"} else "[redacted-filename]"

    def redact_identifier(match: re.Match[str]) -> str:
        identifier = match.group(0)
        return "[redacted-identifier]" if re.search(r"[_-]\d{3,}", identifier) else identifier

    sanitized = re.sub(
        r"(?<![A-Za-z0-9_.-])[A-Za-z0-9_.-]+\.(?:png|jpe?g|tiff?|bmp|gif|csv|tsv|txt|json|xml|mat|npy|npz|nii(?:\.gz)?|dcm|mha|mhd|nrrd|nhdr|h5|hdf5|svs|zip|tar|gz)(?![A-Za-z0-9_.-])",
        redact_filename,
        sanitized,
        flags=re.IGNORECASE,
    )
    sanitized = re.sub(r"[A-Za-z][A-Za-z0-9_-]*", redact_identifier, sanitized)
    return " ".join(sanitized.split())[:2000]


def _adapter_subprocess_environment(workspace: Path) -> dict[str, str]:
    """Build a minimal portable environment without parent credentials."""

    allowed = ("PATH", "LANG", "LC_ALL", "LC_CTYPE", "TZ", "LD_LIBRARY_PATH")
    environment = {key: value for key in allowed if isinstance((value := os.environ.get(key)), str) and value}
    environment.setdefault("PATH", os.defpath)
    home = workspace / ".adapter_home"
    temporary = workspace / ".adapter_tmp"
    home.mkdir(parents=True, exist_ok=True)
    temporary.mkdir(parents=True, exist_ok=True)
    environment.update(
        {
            "HOME": str(home),
            "TMPDIR": str(temporary),
            "TMP": str(temporary),
            "TEMP": str(temporary),
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
            "OMP_NUM_THREADS": "1",
        }
    )
    return environment


def _client_file_identity(path: Path) -> tuple[int, int, int, int, int]:
    stat = path.stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def _require_preexisting_client_file(
    path: Path,
    *,
    snapshot: ClientDataProvenanceSnapshot,
    record_index: int,
) -> None:
    if not snapshot.complete:
        raise ValueError("client-local provenance snapshot is incomplete")
    try:
        if snapshot.root.is_file():
            relative = "." if path == snapshot.root else None
        else:
            relative = path.relative_to(snapshot.root).as_posix()
    except ValueError:
        relative = None
    if relative is None or relative not in snapshot.files:
        raise ValueError(f"adapter manifest record {record_index} label source did not exist before agent execution")
    if _is_reserved_agenticfl_label_source(relative):
        raise ValueError(
            f"adapter manifest record {record_index} label source is an AgenticFL-generated artifact, not raw client evidence"
        )
    try:
        current = _client_file_identity(path)
    except OSError as exc:
        raise ValueError(f"adapter manifest record {record_index} label source is no longer readable") from exc
    if current != snapshot.files[relative]:
        raise ValueError(f"adapter manifest record {record_index} label source changed after agent execution began")


def _is_reserved_agenticfl_label_source(relative_path: str) -> bool:
    parts = Path(relative_path).parts
    return any(part.startswith(".agenticfl") for part in parts)


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
