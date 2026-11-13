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

"""Infrastructure checks for live FedReady workflows."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from fedready.agents import DEFAULT_LOCAL_VISION_MODEL, GuardrailAgent, GuardrailCheck, _query_local_vlm_image
from fedready.agents.bridge import LOCAL_VISION_API_BASE_URL, build_agent_backend
from fedready.agents.local_adapter import (
    is_local_http_url,
    task_example_context_for_task,
    task_example_image_path_for_task,
)
from fedready.utils.io import atomic_write_json
from fedready.utils.logging import timestamp_utc

RUNTIME_PREFLIGHT_SCHEMA = "fedready.runtime_preflight.v1"
DEFAULT_RUNTIME_PREFLIGHT_AGENT_TIMEOUT_SECONDS = 300.0
DEFAULT_RUNTIME_PREFLIGHT_MIN_FREE_GIB = 10.0
DEFAULT_RUNTIME_PREFLIGHT_MIN_FREE_INODES = 100_000


def run_live_runtime_preflight(
    *,
    agent_backend: str,
    task: str,
    project_root: str | Path,
    output_dir: str | Path,
    session_id: str | None = None,
    agent_timeout_seconds: float = DEFAULT_RUNTIME_PREFLIGHT_AGENT_TIMEOUT_SECONDS,
    local_vlm_base_url: str = LOCAL_VISION_API_BASE_URL,
    local_vlm_model: str = DEFAULT_LOCAL_VISION_MODEL,
    local_vlm_api_key_env: str = "FEDREADY_LOCAL_VISION_API_KEY",
    min_free_gib: float = DEFAULT_RUNTIME_PREFLIGHT_MIN_FREE_GIB,
    min_free_inodes: int = DEFAULT_RUNTIME_PREFLIGHT_MIN_FREE_INODES,
    guardrail_smoke: bool = True,
    visual_checks: bool = True,
) -> dict[str, Any]:
    """Fail before expensive work when required live infrastructure is unavailable."""

    backend_kind = str(agent_backend).strip().lower()
    if backend_kind != "codex":
        raise ValueError("runtime preflight requires agent_backend=codex")
    if not task.strip():
        raise ValueError("runtime preflight requires a non-empty task")
    if agent_timeout_seconds <= 0:
        raise ValueError("agent_timeout_seconds must be positive")
    if min_free_gib < 0 or min_free_inodes < 0:
        raise ValueError("storage thresholds must be non-negative")
    if visual_checks:
        if not local_vlm_model.strip():
            raise ValueError("local VLM model must be configured")
        if not is_local_http_url(local_vlm_base_url):
            raise ValueError("runtime preflight refuses a non-local VLM endpoint")

    root = Path(project_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"project root does not exist: {root}")
    preflight_session = session_id or _default_session_id()
    run_dir = Path(output_dir).expanduser().resolve() / "runtime_preflight" / preflight_session
    run_dir.mkdir(parents=True, exist_ok=True)

    storage = _check_storage(
        roots=(root, Path(tempfile.gettempdir())),
        min_free_gib=min_free_gib,
        min_free_inodes=min_free_inodes,
    )
    visual_skip_reason = "visual infrastructure checks are not required for this phase"
    task_example_report: dict[str, Any] = {
        "status": "skipped",
        "reason": visual_skip_reason,
    }
    local_vlm_report: dict[str, Any] = {
        "status": "skipped",
        "reason": visual_skip_reason,
    }
    if visual_checks:
        task_examples = task_example_context_for_task(task)
        examples = task_examples.get("examples") if isinstance(task_examples, dict) else None
        reference_image = task_example_image_path_for_task(task)
        if not isinstance(examples, list) or not examples or reference_image is None:
            raise RuntimeError("no digest-valid task_example reference is available for the requested task")

        vlm_response, vlm_call = _query_local_vlm_image(
            base_url=local_vlm_base_url,
            model=local_vlm_model,
            image_path=reference_image,
            reference_image_path=reference_image,
            prompt=(
                "Infrastructure readiness check only. Confirm that both canonical task-example images "
                "are readable. Do not assess dataset suitability. Return a short non-empty response."
            ),
            api_key_env=local_vlm_api_key_env,
            max_tokens=32,
            log_path=run_dir / "local_vlm_preflight_calls.jsonl",
        )
        if not vlm_response.strip() or vlm_call.get("status") != "completed":
            raise RuntimeError("local VLM preflight did not complete successfully")
        task_example_report = {
            "status": "ready",
            "matched_example_count": len(examples),
            "reference_image_digest_valid": True,
        }
        local_vlm_report = {
            "status": "ready",
            "model": local_vlm_model,
            "endpoint_local": True,
            "response_status_code": vlm_call.get("response_status_code"),
        }

    guardrail_report: dict[str, Any] = {
        "status": "skipped",
        "backend": backend_kind,
        "agent_reviewed": False,
        "normal_message_allowed": None,
        "reason": "guardrail smoke already completed for this experiment batch",
    }
    if guardrail_smoke:
        backend = build_agent_backend(
            kind=backend_kind,
            run_dir=run_dir,
            session_id=preflight_session,
            timeout_seconds=agent_timeout_seconds,
            poll_interval_seconds=1.0,
        )
        guardrail_decision = GuardrailAgent(
            party_role="server",
            party_id="server",
            agent_backend=backend,
            review_mode="agent",
        ).inspect(
            direction="outgoing",
            check=GuardrailCheck(
                role="server_agent",
                source="human_user",
                channel="server_console",
                action="SERVER.DEFINE_PROFILE_REQUEST",
                phase="data_profile",
                input_schema="TaskSpec@v1",
                output_schema="FedReadyClientInquiry@v1",
            ),
            payload={
                "schema_version": "fedready.client_inquiry.v1",
                "task": task,
                "intent": {
                    "routing_scope": "task_family_evidenced_safe_aggregate_metadata_only",
                    "task_description": task,
                },
                "message": {"instructions": ["Inspect safe aggregate metadata for explicit task-relevant evidence."]},
            },
            counterpart="smoke_test_client",
        )
        if not guardrail_decision.allowed or guardrail_decision.redaction_applied:
            raise RuntimeError("live guardrail smoke test blocked a normal profile message")
        guardrail_report = {
            "status": "ready",
            "backend": backend_kind,
            "agent_reviewed": True,
            "normal_message_allowed": True,
            "allow_list_rule": guardrail_decision.rule_id,
        }

    report = {
        "schema_version": RUNTIME_PREFLIGHT_SCHEMA,
        "status": "ready",
        "session_id": preflight_session,
        "agent_backend": backend_kind,
        "task": task,
        "checks": {
            "storage": storage,
            "task_example": task_example_report,
            "local_vlm": local_vlm_report,
            "live_guardrail": guardrail_report,
        },
        "completed_at": timestamp_utc(),
    }
    atomic_write_json(run_dir / "runtime_preflight.json", report)
    return report


def _check_storage(
    *,
    roots: tuple[Path, ...],
    min_free_gib: float,
    min_free_inodes: int,
) -> list[dict[str, Any]]:
    minimum_bytes = int(min_free_gib * 1024**3)
    checked_devices: set[int] = set()
    reports: list[dict[str, Any]] = []
    for root in roots:
        resolved = root.expanduser().resolve()
        stat = resolved.stat()
        if stat.st_dev in checked_devices:
            continue
        checked_devices.add(stat.st_dev)
        usage = shutil.disk_usage(resolved)
        free_inodes = int(os.statvfs(resolved).f_favail)
        if usage.free < minimum_bytes:
            raise RuntimeError(f"runtime preflight requires at least {min_free_gib:g} GiB free on {resolved}")
        if free_inodes < min_free_inodes:
            raise RuntimeError(f"runtime preflight requires at least {min_free_inodes} free inodes on {resolved}")
        reports.append({"root": str(resolved), "free_bytes": usage.free, "free_inodes": free_inodes})
    return reports


def _default_session_id() -> str:
    stamp = timestamp_utc().replace(":", "").replace(".", "_")
    return f"runtime_preflight_{stamp}"
