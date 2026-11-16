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

from pathlib import Path
from typing import Any

from fedready.agents.bridge import CodexWorkerBackend
from fedready.agents.guardrail import ACTION_CHECKS, GuardrailAgent
from fedready.agents.local_adapter import is_local_http_url, task_example_context, task_example_image_path
from fedready.agents.local_vlm import DEFAULT_LOCAL_VISION_MODEL, LOCAL_VISION_API_BASE_URL, _query_local_vlm_image
from fedready.utils.io import atomic_write_json
from fedready.utils.logging import timestamp_utc

RUNTIME_PREFLIGHT_SCHEMA = "fedready.runtime_preflight.v1"
DEFAULT_RUNTIME_PREFLIGHT_AGENT_TIMEOUT_SECONDS = 300.0


def run_live_runtime_preflight(
    *,
    task: str,
    project_root: str | Path,
    output_dir: str | Path,
    session_id: str | None = None,
    agent_timeout_seconds: float = DEFAULT_RUNTIME_PREFLIGHT_AGENT_TIMEOUT_SECONDS,
    local_vlm_base_url: str = LOCAL_VISION_API_BASE_URL,
    local_vlm_model: str = DEFAULT_LOCAL_VISION_MODEL,
    local_vlm_api_key_env: str = "FEDREADY_LOCAL_VISION_API_KEY",
) -> dict[str, Any]:
    """Fail before expensive work when required live infrastructure is unavailable."""

    if not task.strip():
        raise ValueError("runtime preflight requires a non-empty task")
    if agent_timeout_seconds <= 0:
        raise ValueError("agent_timeout_seconds must be positive")
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

    task_examples = task_example_context()
    examples = task_examples.get("examples") if isinstance(task_examples, dict) else None
    reference_image = task_example_image_path()
    if not isinstance(examples, list) or not examples or reference_image is None:
        raise RuntimeError("set FEDREADY_TASK_EXAMPLE_ID to a digest-valid prepared reference task_key")

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

    backend = CodexWorkerBackend.from_env(
        run_dir=run_dir,
        session_id=preflight_session,
        timeout_seconds=agent_timeout_seconds,
    )
    guardrail_decision = GuardrailAgent(
        party_role="server",
        party_id="server",
        agent_backend=backend,
    ).inspect(
        direction="outgoing",
        check=ACTION_CHECKS["SERVER.DEFINE_PROFILE_REQUEST"],
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
        "backend": "codex",
        "agent_reviewed": True,
        "normal_message_allowed": True,
        "allow_list_rule": guardrail_decision.rule_id,
    }

    report = {
        "schema_version": RUNTIME_PREFLIGHT_SCHEMA,
        "status": "ready",
        "session_id": preflight_session,
        "agent_backend": "codex",
        "task": task,
        "checks": {
            "task_example": task_example_report,
            "local_vlm": local_vlm_report,
            "live_guardrail": guardrail_report,
        },
        "completed_at": timestamp_utc(),
    }
    atomic_write_json(run_dir / "runtime_preflight.json", report)
    return report


def _default_session_id() -> str:
    stamp = timestamp_utc().replace(":", "").replace(".", "_")
    return f"runtime_preflight_{stamp}"
