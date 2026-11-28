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

"""Allow-list rules, the guardrail agent and redaction helpers for FedReady messages.

Every agent action is checked against ``ALLOW_LIST_RULES`` before it runs, and every
cross-site payload is scanned and redacted by ``GuardrailAgent`` before it leaves a site.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from fedready.prompts import render_client_prompt


@dataclass(frozen=True)
class GuardrailCheck:
    """Minimal explicit allow-list request descriptor."""

    role: str
    source: str
    channel: str
    action: str
    phase: str
    input_schema: str
    output_schema: str

    def key(self) -> tuple[str, str, str, str, str, str, str]:
        return (
            self.role,
            self.source,
            self.channel,
            self.action,
            self.phase,
            self.input_schema,
            self.output_schema,
        )


@dataclass(frozen=True)
class GuardrailDecision:
    """Explicit allow-list decision suitable for audit logs."""

    allowed: bool
    rule_id: str
    reason_code: str | None
    reason: str
    redacted_fields: tuple[str, ...] = ()

    @property
    def redaction_applied(self) -> bool:
        return bool(self.redacted_fields)

    def effective_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not self.allowed:
            raise ValueError("cannot produce an effective payload for a denied guardrail decision")
        if not self.redacted_fields:
            return payload
        return _guardrail_apply_redactions(payload, self.redacted_fields)

    def as_policy(self) -> dict[str, Any]:
        return {
            "guardrail_checked": True,
            "decision": "redacted" if self.redaction_applied else "allowed" if self.allowed else "denied",
            "allow_list_rule": self.rule_id,
            "reason_code": self.reason_code,
            "reason": self.reason,
            "redactions": list(self.redacted_fields),
        }

    def as_payload(self) -> dict[str, Any]:
        return {
            "schema_version": "fedready.guardrail_decision.v1",
            "allowed": self.allowed,
            "allow_list_rule": self.rule_id,
            "reason_code": self.reason_code,
            "reason": self.reason,
            "redaction_applied": self.redaction_applied,
            "redacted_fields": list(self.redacted_fields),
        }


GUARDRAIL_DECISION_OUTPUT_SCHEMA = "fedready.guardrail_decision.v1"


class GuardrailError(RuntimeError):
    """Raised when a local agent request is outside the explicit allow list."""

    def __init__(self, decision: GuardrailDecision) -> None:
        super().__init__(decision.reason)
        self.decision = decision


ALLOW_LIST_RULES = {
    (
        "server_agent",
        "human_user",
        "server_console",
        "SERVER.DEFINE_PROFILE_REQUEST",
        "data_profile",
        "TaskSpec@v1",
        "FedReadyClientInquiry@v1",
    ): "server.profile_request.v1",
    (
        "server_agent",
        "client_agent_via_flare",
        "flare_task_result",
        "SERVER.AGGREGATE_CLIENT_PROFILES",
        "data_profile",
        "FedReadyClientResponse@v1",
        "FedReadyServerState@v1",
    ): "server.aggregate_client_profiles.v1",
    (
        "server_agent",
        "server_state",
        "flare_task",
        "SERVER.DISPATCH_EXTRACTION_POLICY",
        "data_harmonization",
        "FedReadyServerState@v1",
        "FedReadySiteExtractionPolicy@v1",
    ): "server.dispatch_extraction_policy.v1",
    (
        "server_agent",
        "client_agent_via_flare",
        "flare_task_result",
        "SERVER.AGGREGATE_EXTRACTION_RESULTS",
        "data_harmonization",
        "FedReadyClientExtractionResponse@v1",
        "FedReadyExtractionRoundSummary@v1",
    ): "server.aggregate_extraction_results.v1",
    (
        "server_agent",
        "server_state",
        "local_training_workspace",
        "SERVER.IMPLEMENT_TRAINING_CODE",
        "fl_training",
        "FedReadyExtractionRoundSummary@v1",
        "FedReadyTrainingCodeSpec@v1",
    ): "server.implement_training_code.v1",
    (
        "server_agent",
        "server_state",
        "local_materializer_workspace",
        "SERVER.IMPLEMENT_DATA_MATERIALIZER",
        "data_harmonization",
        "FedReadyGeneratedDataContract@v1",
        "FedReadyDataMaterializerSpec@v1",
    ): "server.implement_data_materializer.v1",
    (
        "client_agent",
        "server_agent_via_flare",
        "flare_task",
        "CLIENT.REPORT_DATA_PROFILE",
        "data_profile",
        "FedReadyClientInquiry@v1",
        "FedReadyClientResponse@v1",
    ): "client.task_inquiry.v1",
    (
        "client_agent",
        "server_agent_via_flare",
        "flare_task",
        "CLIENT.EXTRACT_DATA",
        "data_harmonization",
        "FedReadySiteExtractionPolicy@v1",
        "FedReadyClientExtractionResponse@v1",
    ): "client.extract_data.v1",
    (
        "client_agent",
        "client_local_extractor",
        "local_adapter_workspace",
        "CLIENT.IMPLEMENT_LOCAL_ADAPTER",
        "data_harmonization",
        "FedReadyLocalAdapterRequest@v1",
        "FedReadyLocalAdapterSpec@v1",
    ): "client.implement_local_adapter.v1",
    (
        "client_agent",
        "client_local_extractor",
        "local_visual_qc",
        "CLIENT.VISUAL_QC_EXTRACTION",
        "data_harmonization",
        "FedReadyExtractionQCBundle@v1",
        "FedReadyExtractionQCDecision@v1",
    ): "client.visual_qc_extraction.v1",
}


ACTION_CHECKS = {key[3]: GuardrailCheck(*key) for key in ALLOW_LIST_RULES}


def check_allow_list(check: GuardrailCheck) -> GuardrailDecision:
    rule_id = ALLOW_LIST_RULES.get(check.key())
    if rule_id is None:
        return GuardrailDecision(
            allowed=False,
            rule_id="none",
            reason_code="NOT_IN_ALLOW_LIST",
            reason=(
                "The requested action is not in the agent allow list for this "
                "source, channel, phase, input schema, and output schema."
            ),
        )
    return GuardrailDecision(
        allowed=True,
        rule_id=rule_id,
        reason_code=None,
        reason="Request matched the explicit FedReady allow list.",
    )


def require_allowed(check: GuardrailCheck) -> GuardrailDecision:
    decision = check_allow_list(check)
    if not decision.allowed:
        raise GuardrailError(decision)
    return decision


def _require_configured_agent_backend(agent_backend: Any) -> Any:
    if agent_backend is None:
        raise ValueError("A live Codex agent backend is required; deterministic fallback execution is disabled.")
    return agent_backend


class GuardrailAgent:
    """Agent-mediated FLARE I/O guard with local fail-closed audit signals."""

    def __init__(
        self,
        *,
        party_role: str,
        party_id: str,
        agent_backend: Any,
    ) -> None:
        self.party_role = party_role
        self.party_id = party_id
        self.agent_backend = _require_configured_agent_backend(agent_backend)

    def inspect(
        self,
        *,
        direction: str,
        check: GuardrailCheck,
        payload: dict[str, Any],
        counterpart: str | None = None,
    ) -> GuardrailDecision:
        shape_decision = check_allow_list(check)
        if not shape_decision.allowed:
            return shape_decision

        audit = _guardrail_local_audit(payload=payload, direction=direction)
        allow_outgoing_redaction = direction == "outgoing" and self.party_role == "client"
        baseline_decision = shape_decision
        output_template = baseline_decision.as_payload()
        outgoing_redaction_prompt = (
            render_client_prompt("guardrail_outgoing_redaction") if allow_outgoing_redaction else ""
        )
        prompt = render_client_prompt(
            "guardrail_inspect",
            party_role=self.party_role,
            party_id=self.party_id,
            direction=direction,
            action=check.action,
            outgoing_redaction_prompt=outgoing_redaction_prompt,
        )
        output = self.agent_backend.request(
            role=f"{self.party_role}_guardrail_agent",
            site_id=None if self.party_role == "server" else self.party_id,
            action=f"GUARDRAIL.{check.action}",
            phase=check.phase,
            input_schema=check.input_schema,
            output_schema="FedReadyGuardrailDecision@v1",
            prompt=prompt,
            context={
                "direction": direction,
                "party": {"role": self.party_role, "id": self.party_id},
                "counterpart": counterpart,
                "allow_list_decision": shape_decision.as_payload(),
                "local_audit": audit,
                "payload_for_review": _guardrail_review_payload(payload),
                "output_template": output_template,
                "output_template_instruction": render_client_prompt("guardrail_output_template_instruction"),
            },
        )
        agent_decision = _normalize_guardrail_agent_output(
            output,
            baseline_decision=baseline_decision,
            rule_id=shape_decision.rule_id,
        )
        if agent_decision.allowed and allow_outgoing_redaction and agent_decision.redacted_fields:
            return _guardrail_redaction_decision(
                payload=payload,
                redacted_fields=agent_decision.redacted_fields,
                rule_id=shape_decision.rule_id,
            )
        return agent_decision


SENSITIVE_FIELD_NAMES = {
    "local_data_path",
    "local_data_paths",
    "filename",
    "filenames",
    "file_path",
    "file_paths",
    "image_path",
    "image_paths",
    "mask_path",
    "mask_paths",
    "raw_image",
    "raw_images",
    "raw_mask",
    "raw_masks",
    "patient_id",
    "patient_ids",
    "sample_id",
    "sample_ids",
    "per_sample_record",
    "per_sample_records",
    "credential",
    "credentials",
    "api_key",
    "token",
    "secret",
}


SENSITIVE_REQUEST_TERMS = {
    "local_data_path",
    "local path",
    "local paths",
    "filename",
    "filenames",
    "raw image",
    "raw images",
    "raw mask",
    "raw masks",
    "patient id",
    "patient ids",
    "sample id",
    "sample ids",
    "per-sample",
    "private records",
    "other client",
    "other clients",
    "credential",
    "credentials",
    "api key",
    "secret",
    "/secret",
}


OVERREACHING_REQUEST_TERMS = {
    "ignore privacy",
    "ignore all privacy",
    "ignore all fedready privacy",
    "bypass privacy",
    "disable privacy",
    "override privacy",
    "ignore guardrail",
    "ignore guardrails",
    "bypass guardrail",
    "exfiltrate",
}


REQUEST_COMMAND_TERMS = {
    "return",
    "send",
    "provide",
    "give",
    "list",
    "dump",
    "print",
    "show",
    "reveal",
    "expose",
}


GUARDRAIL_SKIP_KEYS = {
    "redacted",
    "redactions",
    "server_response_redaction",
    "do_not_return_to_server",
    "forbidden_actions",
    "safe_to_share",
}


GUARDRAIL_ALLOWED_RUNTIME_PATH_SUFFIXES = {
    ("extraction_config", "output_root"),
}


def _guardrail_local_audit(*, payload: dict[str, Any], direction: str) -> dict[str, Any]:
    audit: dict[str, Any] = {
        "schema_version": "fedready.guardrail_local_audit.v1",
        "direction": direction,
        "must_deny": False,
        "reason_code": None,
        "unsafe_key_paths": [],
        "local_path_value_paths": [],
        "semantic_warning_paths": [],
        "semantic_warning_terms": [],
    }
    _guardrail_scan_value(payload, path=[], audit=audit)
    return audit


def _guardrail_scan_value(value: Any, *, path: list[str], audit: dict[str, Any]) -> None:
    if _guardrail_should_skip_path(path):
        return
    if isinstance(value, dict):
        for key, child in value.items():
            key_text = str(key)
            child_path = [*path, key_text]
            if key_text.lower() in SENSITIVE_FIELD_NAMES and not _guardrail_should_skip_path(child_path):
                audit["unsafe_key_paths"].append(".".join(child_path))
            _guardrail_scan_value(child, path=child_path, audit=audit)
        return
    if isinstance(value, list):
        for index, child in enumerate(value):
            _guardrail_scan_value(child, path=[*path, str(index)], audit=audit)
        return
    if isinstance(value, str):
        if _looks_like_local_path(value):
            audit["local_path_value_paths"].append(".".join(path))
        terms = _semantic_guardrail_warning_terms(value)
        if terms:
            audit["semantic_warning_paths"].append(".".join(path))
            for term in terms:
                if term not in audit["semantic_warning_terms"]:
                    audit["semantic_warning_terms"].append(term)


def _guardrail_should_skip_path(path: list[str]) -> bool:
    lowered = [part.lower() for part in path]
    if set(lowered) & GUARDRAIL_SKIP_KEYS:
        return True
    return any(tuple(lowered[-len(suffix) :]) == suffix for suffix in GUARDRAIL_ALLOWED_RUNTIME_PATH_SUFFIXES)


def _looks_like_local_path(value: str) -> bool:
    text = value.strip()
    if not text:
        return False
    without_urls = re.sub(r"\b[A-Za-z][A-Za-z0-9+.-]*://[^\s\"'<>]+", "", text)
    return any(
        re.search(pattern, without_urls)
        for pattern in (
            r"(?<![A-Za-z0-9_.-])/(?:[^\s\"'<>]+)",
            r"(?<![A-Za-z0-9_.-])~/(?:[^\s\"'<>]+)",
            r"(?<![A-Za-z0-9_.-])[A-Za-z]:[\\/](?:[^\s\"'<>]+)",
        )
    )


def _redact_local_paths_in_text(value: str) -> str:
    urls: dict[str, str] = {}

    def preserve_url(match: re.Match[str]) -> str:
        marker = f"__FEDREADY_URL_{len(urls)}__"
        urls[marker] = match.group(0)
        return marker

    redacted = re.sub(r"\b[A-Za-z][A-Za-z0-9+.-]*://[^\s\"'<>]+", preserve_url, value)
    for pattern in (
        r"(?<![A-Za-z0-9_.-])/(?:[^\s\"'<>]+)",
        r"(?<![A-Za-z0-9_.-])~/(?:[^\s\"'<>]+)",
        r"(?<![A-Za-z0-9_.-])[A-Za-z]:[\\/](?:[^\s\"'<>]+)",
    ):
        redacted = re.sub(pattern, "[redacted-local-path]", redacted)
    for marker, url in urls.items():
        redacted = redacted.replace(marker, url)
    return redacted


def _semantic_guardrail_warning_terms(value: str) -> list[str]:
    lowered = value.lower()
    terms = [term for term in sorted(OVERREACHING_REQUEST_TERMS) if term in lowered]
    sensitive_terms = [term for term in sorted(SENSITIVE_REQUEST_TERMS) if term in lowered]
    command_present = any(
        re.search(rf"(?<![\w-]){re.escape(term)}(?![\w-])", lowered) for term in REQUEST_COMMAND_TERMS
    )
    if sensitive_terms and command_present:
        terms.extend(term for term in sensitive_terms if term not in terms)
    return terms


def _guardrail_review_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return _guardrail_redact_value(payload, path=[])


def _guardrail_redact_value(value: Any, *, path: list[str]) -> Any:
    if isinstance(value, dict):
        if value.get("schema_version") == "fedready.generated_data_materializer.v1":
            return _generated_materializer_guardrail_payload(value)
        redacted: dict[str, Any] = {}
        for key, child in value.items():
            key_text = str(key)
            if key_text.lower() in SENSITIVE_FIELD_NAMES:
                redacted[key_text] = "[redacted-sensitive-field]"
            else:
                redacted[key_text] = _guardrail_redact_value(child, path=[*path, key_text])
        return redacted
    if isinstance(value, list):
        return [_guardrail_redact_value(item, path=[*path, str(index)]) for index, item in enumerate(value)]
    if isinstance(value, str) and _looks_like_local_path(value):
        return _redact_local_paths_in_text(value)
    return value


def _generated_materializer_guardrail_payload(materializer: dict[str, Any]) -> dict[str, Any]:
    """Preserve server-supplied executable source for client boundary review."""

    return json.loads(json.dumps(materializer, default=str))


def _guardrail_redaction_decision(
    *,
    payload: dict[str, Any],
    redacted_fields: tuple[str, ...],
    rule_id: str,
) -> GuardrailDecision:
    effective_redacted_fields = _guardrail_normalize_redaction_paths(payload, redacted_fields)
    if not effective_redacted_fields or any(
        not _guardrail_path_exists(payload, field.removeprefix("payload_for_review.")) for field in redacted_fields
    ):
        return GuardrailDecision(
            allowed=False,
            rule_id=rule_id,
            reason_code="INVALID_GUARDRAIL_REDACTION",
            reason="Guardrail requested redactions that do not exist in the outgoing payload.",
        )
    try:
        redacted_payload = _guardrail_apply_redactions(payload, effective_redacted_fields)
        if redacted_payload == payload:
            raise ValueError("redaction did not change the payload")
    except ValueError as exc:
        return GuardrailDecision(
            allowed=False,
            rule_id=rule_id,
            reason_code="INVALID_GUARDRAIL_REDACTION",
            reason=f"Guardrail could not apply the requested field redaction ({type(exc).__name__}).",
        )
    return GuardrailDecision(
        allowed=True,
        rule_id=rule_id,
        reason_code="REDACTED_SENSITIVE_OUTPUT",
        reason="Guardrail removed sensitive or unnecessary local metadata before FLARE transmission.",
        redacted_fields=effective_redacted_fields,
    )


def _guardrail_normalize_redaction_paths(payload: dict[str, Any], redacted_fields: tuple[str, ...]) -> tuple[str, ...]:
    """Keep only optional paths that exist in the payload actually sent over FLARE."""
    normalized: list[str] = []
    for field_path in redacted_fields:
        candidate = field_path.removeprefix("payload_for_review.")
        if candidate in normalized or not _guardrail_path_exists(payload, candidate):
            continue
        normalized.append(candidate)
    return tuple(normalized)


def _guardrail_path_exists(payload: dict[str, Any], field_path: str) -> bool:
    parts = field_path.split(".")
    if not parts or any(not part for part in parts):
        return False
    current: Any = payload
    for part in parts:
        if isinstance(current, dict) and part in current:
            current = current[part]
        elif isinstance(current, list) and part.isdigit() and int(part) < len(current):
            current = current[int(part)]
        else:
            return False
    return True


def _guardrail_apply_redactions(payload: dict[str, Any], redacted_fields: tuple[str, ...]) -> dict[str, Any]:
    redacted = json.loads(json.dumps(payload, default=str))
    protected_top_level = {
        "schema_version",
        "client_id",
        "data",
        "status",
        "counts",
        "verification",
        "visual_qc",
        "fit",
        "meta",
    }
    ordered_fields = sorted(
        dict.fromkeys(redacted_fields),
        key=lambda value: value.count("."),
        reverse=True,
    )
    for field_path in ordered_fields:
        parts = field_path.split(".")
        if not parts or any(not part for part in parts):
            raise ValueError(f"invalid redaction path {field_path!r}")
        if len(parts) == 1 and parts[0] in protected_top_level:
            raise ValueError(f"required top-level field cannot be redacted: {field_path}")
        parent: Any = redacted
        for part in parts[:-1]:
            if isinstance(parent, dict) and part in parent:
                parent = parent[part]
            elif isinstance(parent, list) and part.isdigit() and int(part) < len(parent):
                parent = parent[int(part)]
            else:
                raise ValueError(f"redaction path does not exist: {field_path}")
        leaf = parts[-1]
        if isinstance(parent, dict) and leaf in parent:
            del parent[leaf]
        elif isinstance(parent, list) and leaf.isdigit() and int(leaf) < len(parent):
            parent[int(leaf)] = "[redacted-by-guardrail]"
        else:
            raise ValueError(f"redaction path does not exist: {field_path}")
    if redacted.get("schema_version") != payload.get("schema_version"):
        raise ValueError("redaction changed schema_version")
    if "client_id" in payload and redacted.get("client_id") != payload.get("client_id"):
        raise ValueError("redaction changed client_id")
    return redacted


def _normalize_guardrail_agent_output(
    output: dict[str, Any],
    *,
    baseline_decision: GuardrailDecision,
    rule_id: str,
) -> GuardrailDecision:
    if not isinstance(output, dict):
        return GuardrailDecision(
            allowed=False,
            rule_id=rule_id,
            reason_code="INVALID_GUARDRAIL_OUTPUT",
            reason="Guardrail agent returned a non-object decision.",
        )
    normalized = dict(output)
    if normalized.get("schema_version") == "FedReadyGuardrailDecision@v1":
        normalized["schema_version"] = GUARDRAIL_DECISION_OUTPUT_SCHEMA
    if normalized.get("schema_version") != GUARDRAIL_DECISION_OUTPUT_SCHEMA:
        return GuardrailDecision(
            allowed=False,
            rule_id=rule_id,
            reason_code="INVALID_GUARDRAIL_OUTPUT",
            reason="Guardrail agent returned an invalid decision schema.",
        )
    allowed = normalized.get("allowed")
    if not isinstance(allowed, bool):
        return GuardrailDecision(
            allowed=False,
            rule_id=rule_id,
            reason_code="INVALID_GUARDRAIL_OUTPUT",
            reason="Guardrail agent decision missing boolean allowed field.",
        )
    reason_code = normalized.get("reason_code")
    if reason_code is not None and not isinstance(reason_code, str):
        reason_code = "INVALID_GUARDRAIL_REASON_CODE"
    reason = normalized.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        reason = baseline_decision.reason if allowed == baseline_decision.allowed else "Guardrail agent decision."
    raw_redacted_fields = normalized.get("redacted_fields", baseline_decision.redacted_fields)
    if not isinstance(raw_redacted_fields, (list, tuple)) or not all(
        isinstance(value, str) and value for value in raw_redacted_fields
    ):
        return GuardrailDecision(
            allowed=False,
            rule_id=rule_id,
            reason_code="INVALID_GUARDRAIL_REDACTION",
            reason="Guardrail agent returned invalid redacted_fields.",
        )
    return GuardrailDecision(
        allowed=allowed,
        rule_id=str(normalized.get("allow_list_rule") or normalized.get("rule_id") or rule_id),
        reason_code=reason_code,
        reason=reason,
        redacted_fields=tuple(dict.fromkeys(raw_redacted_fields)),
    )


def _short_error(exc: BaseException) -> str:
    message = _redact_local_paths_in_text(str(exc))
    if len(message) > 800:
        return message[:800] + "...<truncated>"
    return message
