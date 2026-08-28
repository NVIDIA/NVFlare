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

"""Append-only flow logging utilities for AgenticFL simulations."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

ACTION_ORDER = {
    "SERVER_AGENT_OUTPUT": 1,
    "SERVER_GUARDRAIL_CHECKED": 2,
    "SERVER_OUTGOING_GUARDRAIL_CHECKED": 2,
    "SERVER_INCOMING_GUARDRAIL_CHECKED": 2,
    "CLIENT_AGENT_OUTPUT_RECEIVED": 3,
    "CLIENT_EXTRACTION_OUTPUT_RECEIVED": 3,
    "CLIENT_REQUEST_RECEIVED": 1,
    "CLIENT_EXTRACTION_REQUEST_RECEIVED": 1,
    "CLIENT_GUARDRAIL_CHECKED": 2,
    "CLIENT_EXTRACTION_GUARDRAIL_CHECKED": 2,
    "CLIENT_OUTGOING_GUARDRAIL_CHECKED": 7,
    "CLIENT_EXTRACTION_OUTGOING_GUARDRAIL_CHECKED": 7,
    "CLIENT_LOCAL_ADAPTER_COMPOSED": 3,
    "CLIENT_LOCAL_ADAPTER_EXECUTED": 4,
    "CLIENT_LOCAL_EXECUTION_STARTED": 5,
    "CLIENT_EXTRACTION_EXECUTION_STARTED": 5,
    "CLIENT_ORIENTATION_QC_DECISION": 6,
    "CLIENT_LOCAL_EXECUTION_COMPLETED": 7,
    "CLIENT_EXTRACTION_EXECUTION_COMPLETED": 7,
    "CLIENT_AGENT_OUTPUT": 8,
    "CLIENT_EXTRACTION_OUTPUT": 8,
}


def canonical_json(value: Any) -> str:
    """Return deterministic JSON for hashing and log persistence."""

    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def payload_digest(value: Any) -> str:
    """Return a stable SHA-256 digest for a JSON-compatible payload."""

    return "sha256:" + hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def timestamp_utc() -> str:
    """Return an ISO-8601 UTC timestamp with microsecond precision."""

    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


class FlowLogger:
    """Small JSONL logger for server and client-local flow traces."""

    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir)
        (self.run_dir / "server" / "requests").mkdir(parents=True, exist_ok=True)
        (self.run_dir / "server" / "responses").mkdir(parents=True, exist_ok=True)
        (self.run_dir / "server" / "decisions").mkdir(parents=True, exist_ok=True)

    def write_server_event(
        self,
        *,
        session_id: str,
        phase_id: str,
        step_id: str,
        correlation_id: str,
        event_type: str,
        payload: dict[str, Any],
        schema_ref: str,
        status: str,
        next_step: dict[str, Any],
        parent_event_id: str | None = None,
        privacy_class: str = "aggregate_stats",
        visibility: str = "server",
        policy: dict[str, Any] | None = None,
        flare: dict[str, Any] | None = None,
        agent_prompt: str | None = None,
    ) -> dict[str, Any]:
        event = self._event(
            session_id=session_id,
            phase_id=phase_id,
            step_id=step_id,
            correlation_id=correlation_id,
            parent_event_id=parent_event_id,
            actor={"role": "server", "id": "server"},
            event_type=event_type,
            visibility=visibility,
            privacy_class=privacy_class,
            schema_ref=schema_ref,
            payload=payload,
            policy=policy,
            status=status,
            next_step=next_step,
            flare=flare,
            agent_prompt=agent_prompt,
        )
        self._append_round_action_event(self.run_dir / "server", event)
        return event

    def write_client_event(
        self,
        *,
        client_id: str,
        session_id: str,
        phase_id: str,
        step_id: str,
        correlation_id: str,
        event_type: str,
        payload: dict[str, Any],
        schema_ref: str,
        status: str,
        next_step: dict[str, Any],
        parent_event_id: str | None = None,
        privacy_class: str = "local_private",
        visibility: str = "client_local",
        policy: dict[str, Any] | None = None,
        flare: dict[str, Any] | None = None,
        agent_prompt: str | None = None,
    ) -> dict[str, Any]:
        site_dir = self.run_dir / "sites" / client_id
        site_dir.mkdir(parents=True, exist_ok=True)
        event = self._event(
            session_id=session_id,
            phase_id=phase_id,
            step_id=step_id,
            correlation_id=correlation_id,
            parent_event_id=parent_event_id,
            actor={"role": "client", "id": client_id},
            event_type=event_type,
            visibility=visibility,
            privacy_class=privacy_class,
            schema_ref=schema_ref,
            payload=payload,
            policy=policy,
            status=status,
            next_step=next_step,
            flare=flare,
            agent_prompt=agent_prompt,
        )
        self._append_round_action_event(site_dir, event)
        return event

    def write_artifact(self, relative_path: str | Path, payload: dict[str, Any]) -> Path:
        path = self.run_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return path

    def _event(
        self,
        *,
        session_id: str,
        phase_id: str,
        step_id: str,
        correlation_id: str,
        parent_event_id: str | None,
        actor: dict[str, str],
        event_type: str,
        visibility: str,
        privacy_class: str,
        schema_ref: str,
        payload: dict[str, Any],
        policy: dict[str, Any] | None,
        status: str,
        next_step: dict[str, Any],
        flare: dict[str, Any] | None,
        agent_prompt: str | None,
    ) -> dict[str, Any]:
        prompt_digest = payload_digest(agent_prompt) if agent_prompt is not None else None
        return {
            "schema_version": "agenticfl.log_event.v1",
            "event_id": str(uuid4()),
            "timestamp_utc": timestamp_utc(),
            "session_id": session_id,
            "phase_id": phase_id,
            "step_id": step_id,
            "correlation_id": correlation_id,
            "parent_event_id": parent_event_id,
            "actor": actor,
            "event_type": event_type,
            "visibility": visibility,
            "privacy_class": privacy_class,
            "flare": flare or {},
            "schema_ref": schema_ref,
            "payload_digest": payload_digest(payload),
            "agent_prompt_digest": prompt_digest,
            "agent_prompt": agent_prompt,
            "agent_output": payload,
            "policy": policy
            or {
                "guardrail_checked": True,
                "decision": "allowed",
                "allow_list_rule": "simulation.basic_poc.v1",
                "redactions": [],
            },
            "status": status,
            "next_step": next_step,
        }

    @staticmethod
    def _append_jsonl(path: Path, event: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(canonical_json(event) + "\n")

    def _append_round_action_event(self, root: Path, event: dict[str, Any]) -> None:
        round_dir = _round_dir_name(event)
        action_name = _action_file_name(event)
        self._append_jsonl(root / "rounds" / round_dir / "actions" / action_name, event)


def _round_dir_name(event: dict[str, Any]) -> str:
    phase = _slug(event.get("phase_id", "phase"))
    flare = event.get("flare") if isinstance(event.get("flare"), dict) else {}
    current_round = flare.get("current_round")
    if isinstance(current_round, int):
        return f"round_{current_round:02d}_{phase}"
    if isinstance(current_round, str) and current_round.isdigit():
        return f"round_{int(current_round):02d}_{phase}"
    inferred_round = {"profile": 1, "data_profile": 1, "harmonization": 2, "data_harmonization": 2}.get(phase)
    if inferred_round is not None:
        return f"round_{inferred_round:02d}_{phase}"
    return f"round_unknown_{phase}"


def _slug(value: Any) -> str:
    text = str(value or "unknown").lower()
    chars = [char if char.isalnum() else "_" for char in text]
    slug = "_".join(part for part in "".join(chars).split("_") if part)
    return slug or "unknown"


def _action_file_name(event: dict[str, Any]) -> str:
    event_type = str(event.get("event_type", "event"))
    return f"{ACTION_ORDER.get(event_type, 99):02d}_{_slug(event_type)}.jsonl"
