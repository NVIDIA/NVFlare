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

"""NVFlare message helpers for AgenticFL jobs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable

AGENTICFL_MESSAGE_KEY = "agenticfl_message"
TASK_QUERY_TASK_NAME = "agenticfl_task_query"
SIMULATION_SCHEMA_VERSION = "agenticfl.nvflare_task_query_job.v1"
CLIENT_INQUIRY_SCHEMA = "AgenticFLClientInquiry@v1"
CLIENT_RESPONSE_SCHEMA = "AgenticFLClientResponse@v1"
SERVER_STATE_SCHEMA = "AgenticFLServerState@v1"
GUARDRAIL_SCHEMA = "AgenticFLGuardrailDecision@v1"
EXTRACTION_POLICY_SCHEMA = "AgenticFLSiteExtractionPolicy@v1"
EXTRACTION_RESPONSE_SCHEMA = "AgenticFLClientExtractionResponse@v1"
EXTRACTION_SUMMARY_SCHEMA = "AgenticFLExtractionRoundSummary@v1"
LOCAL_EXECUTION_SCHEMA = "AgenticFLLocalExecution@v1"

ROUND_PROFILE = 1
ROUND_EXTRACTION = 2

HEADER_SESSION_ID = "agenticfl.session_id"
HEADER_CORRELATION_ID = "agenticfl.correlation_id"
HEADER_SCHEMA_REF = "agenticfl.schema_ref"
HEADER_TASK_ID = "agenticfl.task_id"
HEADER_SITE_ID = "agenticfl.site_id"
HEADER_CURRENT_ROUND = "agenticfl.current_round"
HEADER_TOTAL_ROUNDS = "agenticfl.total_rounds"
HEADER_ROUND_ACTION = "agenticfl.round_action"


@dataclass(frozen=True)
class FlareMessage:
    """A typed message carried in a NVFlare Shareable."""

    session_id: str
    correlation_id: str
    schema_ref: str
    task_id: str
    payload: dict[str, Any]
    site_id: str | None = None


def to_shareable(message: FlareMessage) -> Shareable:
    shareable = Shareable({AGENTICFL_MESSAGE_KEY: message.payload})
    shareable.set_header(HEADER_SESSION_ID, message.session_id)
    shareable.set_header(HEADER_CORRELATION_ID, message.correlation_id)
    shareable.set_header(HEADER_SCHEMA_REF, message.schema_ref)
    shareable.set_header(HEADER_TASK_ID, message.task_id)
    if message.site_id is not None:
        shareable.set_header(HEADER_SITE_ID, message.site_id)
    return shareable


def from_shareable(shareable: Shareable) -> FlareMessage:
    payload = shareable.get(AGENTICFL_MESSAGE_KEY)
    if not isinstance(payload, dict):
        raise ValueError("Shareable does not contain an AgenticFL message payload")
    site_id = shareable.get_header(HEADER_SITE_ID)
    if site_id is not None:
        site_id = str(site_id).strip()
        if not site_id:
            raise ValueError(f"Shareable header {HEADER_SITE_ID} is empty")
    return FlareMessage(
        session_id=_required_text_header(shareable, HEADER_SESSION_ID),
        correlation_id=_required_text_header(shareable, HEADER_CORRELATION_ID),
        schema_ref=_required_text_header(shareable, HEADER_SCHEMA_REF),
        task_id=_required_text_header(shareable, HEADER_TASK_ID),
        site_id=site_id,
        payload=payload,
    )


def _required_text_header(shareable: Shareable, key: str) -> str:
    value = shareable.get_header(key)
    if value is None:
        raise ValueError(f"Shareable missing required header {key}")
    text = str(value).strip()
    if not text:
        raise ValueError(f"Shareable header {key} is empty")
    return text


def set_round_headers(shareable: Shareable, *, current_round: int, total_rounds: int, action: str) -> None:
    shareable.set_header(HEADER_CURRENT_ROUND, current_round)
    shareable.set_header(HEADER_TOTAL_ROUNDS, total_rounds)
    shareable.set_header(HEADER_ROUND_ACTION, action)


def round_props(current_round: int, total_rounds: int, action: str) -> dict[str, Any]:
    return {
        HEADER_CURRENT_ROUND: current_round,
        HEADER_TOTAL_ROUNDS: total_rounds,
        HEADER_ROUND_ACTION: action,
    }


def set_fl_ctx_round(fl_ctx: FLContext, current_round: int, total_rounds: int) -> None:
    fl_ctx.set_prop(HEADER_CURRENT_ROUND, current_round, private=False, sticky=False)
    fl_ctx.set_prop(HEADER_TOTAL_ROUNDS, total_rounds, private=False, sticky=False)


def current_round_from_shareable(shareable: Shareable) -> int:
    value = shareable.get_header(HEADER_CURRENT_ROUND)
    if value is None:
        raise ValueError(f"Shareable missing required header {HEADER_CURRENT_ROUND}")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Shareable header {HEADER_CURRENT_ROUND} is malformed") from exc


def total_rounds_from_shareable(shareable: Shareable, default: int) -> int:
    value = shareable.get_header(HEADER_TOTAL_ROUNDS, default)
    if value is None:
        raise ValueError(f"Shareable missing required header {HEADER_TOTAL_ROUNDS}")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Shareable header {HEADER_TOTAL_ROUNDS} is malformed") from exc
