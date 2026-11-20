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

"""Visual QC for extracted samples: output templates, local VLM review and sample voting."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from fedready.agents.client.local_vlm import (
    LOCAL_VISION_API_BASE_URL,
    _is_local_vlm_base_url,
    _local_vlm_model_name,
    _query_local_vlm_image,
)
from fedready.agents.guardrail import _short_error
from fedready.agents.references import task_example_image_path
from fedready.data.qc import VISUAL_QC_TRANSFORM_SET, VISUAL_QC_TRANSFORMS
from fedready.prompts import render_client_prompt

VISUAL_QC_ALLOWED_SELECTIONS = VISUAL_QC_TRANSFORM_SET | {"undecided"}
LOCAL_VLM_QC_SCHEMA = "fedready.local_vlm_visual_qc.v1"


def _adapter_preflight_visual_qc_decision(*, client_id: str, qc_context: dict[str, Any]) -> dict[str, Any]:
    """Reuse the digest-bound local guardrail verdict without a second VLM pass."""

    output = _live_visual_qc_template(client_id=client_id, qc_context=qc_context)
    sample_count = max(1, int(output.get("sample_count") or 0))
    output.update(
        {
            "status": "passed",
            "passed": True,
            "reviewed": True,
            "review_required": True,
            "visual_qc_owner": "adapter_preflight_local_guardrail",
            "reviewed_sample_count": 1,
            "selected_transform": "as_is",
            "selected_transform_counts": {"as_is": 1},
            "consensus_required_count": 1,
            "consensus_reached": True,
            "issues": [],
            "reason": "Digest-bound adapter preflight raw and label review passed locally.",
            "recommendation": "continue_with_extracted_records",
            "sample_count": sample_count,
        }
    )
    return output


def _normalize_visual_qc_output(output: dict[str, Any], qc_context: dict[str, Any] | None = None) -> dict[str, Any]:
    allowed_statuses = {"passed", "failed", "needs_more_samples", "not_performed", "needs_agent_visual_review"}
    normalized = dict(output)
    consensus = _visual_qc_consensus(normalized, qc_context)
    if consensus["expected_sample_count"] > 0:
        normalized["sample_count"] = consensus["expected_sample_count"]
        normalized["reviewed_sample_count"] = consensus["reviewed_sample_count"]
        normalized["consensus_required_count"] = consensus["required_count"]
        normalized["consensus_reached"] = consensus["reached"]
        normalized["selected_transform_counts"] = consensus["counts"]
    if consensus["expected_sample_count"] > 1:
        if consensus["reached"]:
            selected_transform = consensus["selected_transform"]
            normalized["selected_transform"] = selected_transform
            normalized["reviewed"] = True
            if selected_transform == "as_is":
                normalized["status"] = "passed"
                normalized["passed"] = True
                normalized["issues"] = []
                normalized["recommendation"] = "accept_extraction_for_training"
                return normalized
            normalized["status"] = "failed"
            normalized["passed"] = False
            _append_visual_qc_issue(normalized, f"visual QC selected non-as_is transform: {selected_transform}")
            normalized["recommendation"] = f"rerun extraction or adapter with selected_transform={selected_transform}"
            return normalized
        if consensus["reviewed_sample_count"] > 0 or _visual_qc_has_decision(normalized):
            normalized["status"] = "needs_more_samples"
            normalized["passed"] = False
            normalized["selected_transform"] = "undecided"
            _append_visual_qc_issue(
                normalized,
                "visual QC requires >=2/3 transform consensus before passing or selecting a repair transform",
            )
            normalized["recommendation"] = "review_more_visual_qc_samples_before_accepting_extraction"
            return normalized

    selected = normalized.get("selected_transform")
    status = normalized.get("status")
    if selected in {"hflip", "vflip", "rot180"}:
        normalized["status"] = "failed"
        normalized["passed"] = False
        _append_visual_qc_issue(normalized, f"visual QC selected non-as_is transform: {selected}")
        normalized.setdefault("recommendation", f"rerun extraction or adapter with selected_transform={selected}")
        return normalized
    if status not in allowed_statuses:
        if normalized.get("passed") is True and selected == "as_is":
            normalized["status"] = "passed"
        elif selected == "undecided" and normalized.get("reviewed") is True:
            normalized["status"] = "needs_more_samples"
        else:
            normalized["status"] = "needs_agent_visual_review"
    if normalized.get("status") == "passed":
        if selected == "as_is" and normalized.get("reviewed") is True:
            normalized["passed"] = True
        else:
            normalized["status"] = "needs_agent_visual_review"
            normalized["passed"] = False
    elif normalized.get("status") in {"failed", "needs_more_samples", "needs_agent_visual_review"}:
        normalized["passed"] = False
    return normalized


def _live_visual_qc_template(*, client_id: str, qc_context: dict[str, Any]) -> dict[str, Any]:
    artifacts = _visual_qc_artifacts(qc_context)
    sample_count = len(artifacts)
    available = bool(qc_context.get("available")) and sample_count > 0
    review_required = qc_context.get("review_required") is not False
    visual_qc_owner = str(qc_context.get("visual_qc_owner") or "client_local_visual_qc")
    return {
        "schema_version": "fedready.extraction_visual_qc_decision.v1",
        "client_id": client_id,
        "status": "needs_agent_visual_review" if available and review_required else "not_performed",
        "passed": False,
        "reviewed": False,
        "review_required": review_required,
        "visual_qc_owner": visual_qc_owner,
        "sample_count": sample_count,
        "reviewed_sample_count": 0,
        "selected_transform": "undecided",
        "selected_transform_counts": {},
        "consensus_required_count": max(1, (2 * sample_count + 2) // 3) if sample_count else 0,
        "consensus_reached": False,
        "issues": [] if available or not review_required else ["no_local_qc_artifacts"],
        "reason": (
            "Visual QC is not required by the active generated contract."
            if not review_required
            else (
                "Live visual QC requires an explicit local review decision."
                if available
                else "No local QC artifacts are available for visual review."
            )
        ),
        "recommendation": (
            "continue_without_visual_qc"
            if not review_required
            else ("review_local_qc_artifacts" if available else "do_not_use_for_training_until_qc_artifacts_exist")
        ),
        "safe_to_share": True,
    }


def _aggregate_visual_qc_sample_outputs(
    *,
    base_output: dict[str, Any],
    sample_outputs: list[dict[str, Any]],
    qc_context: dict[str, Any],
) -> dict[str, Any]:
    if base_output.get("deterministic_ground_truth") is True or any(
        sample.get("deterministic_ground_truth") is True for sample in sample_outputs
    ):
        raise RuntimeError("local visual QC received deterministic_ground_truth output")
    counts: dict[str, int] = {}
    reviewed_count = 0
    issues: list[str] = []
    local_vlm_reviews: list[dict[str, Any]] = []
    for sample_output in sample_outputs:
        selected = _visual_qc_sample_vote(sample_output)
        if selected in VISUAL_QC_ALLOWED_SELECTIONS:
            counts[str(selected)] = counts.get(str(selected), 0) + 1
        if sample_output.get("reviewed") is True:
            reviewed_count += 1
        local_vlm_review = sample_output.get("local_vlm_review")
        if isinstance(local_vlm_review, dict):
            local_vlm_reviews.append(local_vlm_review)
    output = dict(base_output)
    output.update(
        {
            "reviewed": reviewed_count > 0,
            "reviewed_sample_count": reviewed_count,
            "selected_transform_counts": counts,
            "issues": issues,
        }
    )
    if local_vlm_reviews:
        output["local_vlm_review"] = {
            "schema_version": LOCAL_VLM_QC_SCHEMA,
            "backend": "local_vlm",
            "status": (
                "completed"
                if all(review.get("status") == "completed" for review in local_vlm_reviews)
                else "partial_or_failed"
            ),
            "reviewed_sample_count": len(local_vlm_reviews),
            "sample_reviews": local_vlm_reviews,
            "calls_logged_client_local": True,
        }

    normalized = _normalize_visual_qc_output(output, qc_context=qc_context)
    if local_vlm_reviews:
        normalized["reason"] = (
            f"Local VLM reviewed {len(local_vlm_reviews)} samples; consensus selected "
            f"{normalized.get('selected_transform', 'undecided')}."
        )
    return normalized


def _visual_qc_sample_vote(sample_output: dict[str, Any]) -> str | None:
    selected = sample_output.get("selected_transform")
    if selected in VISUAL_QC_TRANSFORMS:
        return str(selected)
    counts = _extract_visual_qc_transform_counts(sample_output)
    decision_counts = {transform: count for transform, count in counts.items() if transform in VISUAL_QC_TRANSFORMS}
    if decision_counts:
        top_count = max(decision_counts.values())
        top_transforms = [transform for transform, count in decision_counts.items() if count == top_count]
        if len(top_transforms) == 1 and top_count > 0:
            return top_transforms[0]
    if selected == "undecided":
        return "undecided"
    return None


def _visual_qc_consensus(output: dict[str, Any], qc_context: dict[str, Any] | None) -> dict[str, Any]:
    counts = _extract_visual_qc_transform_counts(output)
    expected_sample_count = _expected_visual_qc_sample_count(output, qc_context, counts)
    selected = output.get("selected_transform")
    if not counts and expected_sample_count <= 1 and selected in VISUAL_QC_TRANSFORMS:
        counts = {str(selected): 1}
        expected_sample_count = max(expected_sample_count, 1)
    reviewed_sample_count = _visual_qc_reviewed_sample_count(output, counts)
    if counts:
        reviewed_sample_count = max(reviewed_sample_count, sum(counts.values()))
    required_count = _visual_qc_required_vote_count(expected_sample_count)
    decision_counts = {transform: count for transform, count in counts.items() if transform in VISUAL_QC_TRANSFORMS}
    selected_transform = "undecided"
    reached = False
    if decision_counts and required_count > 0:
        top_count = max(decision_counts.values())
        top_transforms = [transform for transform, count in decision_counts.items() if count == top_count]
        if len(top_transforms) == 1 and top_count >= required_count:
            selected_transform = top_transforms[0]
            reached = True
    return {
        "counts": {selection: counts.get(selection, 0) for selection in sorted(VISUAL_QC_ALLOWED_SELECTIONS)},
        "expected_sample_count": expected_sample_count,
        "reviewed_sample_count": reviewed_sample_count,
        "required_count": required_count,
        "reached": reached,
        "selected_transform": selected_transform,
    }


def _extract_visual_qc_transform_counts(output: dict[str, Any]) -> dict[str, int]:
    counts = output.get("selected_transform_counts")
    return _coerce_visual_qc_counts(counts) if isinstance(counts, dict) else {}


def _coerce_visual_qc_counts(value: dict[Any, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for key, count_value in value.items():
        transform = str(key)
        if transform not in VISUAL_QC_ALLOWED_SELECTIONS:
            continue
        count = _coerce_nonnegative_int(count_value)
        if count is not None and count > 0:
            counts[transform] = counts.get(transform, 0) + count
    return counts


def _expected_visual_qc_sample_count(
    output: dict[str, Any],
    qc_context: dict[str, Any] | None,
    counts: dict[str, int],
) -> int:
    candidates: list[int] = []
    for value in (output.get("sample_count"), output.get("reviewed_sample_count")):
        count = _coerce_nonnegative_int(value)
        if count is not None and count > 0:
            candidates.append(count)
    if isinstance(qc_context, dict):
        count = _coerce_nonnegative_int(qc_context.get("sample_count"))
        if count is not None and count > 0:
            candidates.append(count)
        artifacts = qc_context.get("artifacts")
        if isinstance(artifacts, list) and artifacts:
            candidates.append(len(artifacts))
    if counts:
        candidates.append(sum(counts.values()))
    return max(candidates) if candidates else 0


def _visual_qc_reviewed_sample_count(output: dict[str, Any], counts: dict[str, int]) -> int:
    count = _coerce_nonnegative_int(output.get("reviewed_sample_count"))
    return count if count is not None else sum(counts.values())


def _visual_qc_required_vote_count(sample_count: int) -> int:
    if sample_count <= 0:
        return 0
    return max(1, (2 * sample_count + 2) // 3)


def _visual_qc_has_decision(output: dict[str, Any]) -> bool:
    return (
        output.get("status") in {"passed", "failed"}
        or output.get("passed") is True
        or output.get("selected_transform") in VISUAL_QC_TRANSFORMS
    )


def _append_visual_qc_issue(output: dict[str, Any], message: str) -> None:
    issues = output.get("issues")
    if isinstance(issues, list):
        if message not in issues:
            issues.append(message)
    else:
        output["issues"] = [message]


def _coerce_nonnegative_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return max(0, value)
    if isinstance(value, float) and value.is_integer():
        return max(0, int(value))
    if isinstance(value, str):
        try:
            parsed = int(value)
        except ValueError:
            return None
        return max(0, parsed)
    return None


def _visual_qc_artifacts(qc_context: dict[str, Any]) -> list[dict[str, Any]]:
    artifacts = qc_context.get("artifacts") if isinstance(qc_context, dict) else None
    if not isinstance(artifacts, list):
        return []
    return [artifact for artifact in artifacts if isinstance(artifact, dict)]


def _request_local_vlm_visual_qc(
    *,
    client_id: str,
    policy: dict[str, Any],
    extraction_result: dict[str, Any],
    qc_context: dict[str, Any],
    output_template: dict[str, Any],
) -> dict[str, Any]:
    model = _local_vlm_model_name()
    base_url = os.environ.get("FEDREADY_VISION_AGENT_API_BASE_URL", LOCAL_VISION_API_BASE_URL).strip()
    api_key_env = os.environ.get("FEDREADY_VISION_AGENT_API_KEY_ENV", "FEDREADY_LOCAL_VISION_API_KEY").strip()
    max_tokens = _env_positive_int("FEDREADY_VISUAL_QC_VLM_MAX_TOKENS", 512)

    if not model:
        return _local_vlm_qc_failure(
            client_id=client_id,
            output_template=output_template,
            reason="FEDREADY_VISION_AGENT_MODEL is not configured for local visual QC",
            model=model,
            base_url=base_url,
        )
    if not _is_local_vlm_base_url(base_url):
        return _local_vlm_qc_failure(
            client_id=client_id,
            output_template=output_template,
            reason=f"Refusing to send visual QC image to non-local endpoint: {base_url}",
            model=model,
            base_url=base_url,
        )

    artifacts = _visual_qc_artifacts(qc_context)
    artifact = artifacts[0] if artifacts else {}
    image_path = _local_vlm_qc_image_path(artifact)
    if image_path is None:
        return _local_vlm_qc_failure(
            client_id=client_id,
            output_template=output_template,
            reason="local visual QC requires a readable candidate_sheet_path artifact",
            model=model,
            base_url=base_url,
        )
    reference_image_path = task_example_image_path()
    if reference_image_path is None:
        return _local_vlm_qc_failure(
            client_id=client_id,
            output_template=output_template,
            reason="canonical task_example reference is unavailable or failed integrity validation",
            model=model,
            base_url=base_url,
        )

    prompt = _local_vlm_visual_qc_prompt(
        policy=policy,
        extraction_result=extraction_result,
        qc_context=qc_context,
    )
    log_path = _local_vlm_qc_log_path(image_path)
    try:
        response_text, call_record = _query_local_vlm_image(
            base_url=base_url,
            model=model,
            image_path=image_path,
            prompt=prompt,
            api_key_env=api_key_env,
            max_tokens=max_tokens,
            log_path=log_path,
            reference_image_path=reference_image_path,
        )
        selected_transform, reason = _parse_local_vlm_visual_qc_response(response_text)
        return _local_vlm_qc_decision(
            client_id=client_id,
            output_template=output_template,
            selected_transform=selected_transform,
            reason=reason,
            model=model,
            base_url=base_url,
            call_record=call_record,
        )
    except Exception as exc:
        return _local_vlm_qc_failure(
            client_id=client_id,
            output_template=output_template,
            reason=f"local VLM visual QC call failed: {_short_error(exc)}",
            model=model,
            base_url=base_url,
        )


def _local_vlm_visual_qc_prompt(
    *,
    policy: dict[str, Any],
    extraction_result: dict[str, Any],
    qc_context: dict[str, Any],
) -> str:
    task = _safe_short_json(policy.get("task") or policy.get("task_description") or "the requested task")
    label_rule = _safe_short_json(policy.get("label_rule", {}), limit=1200)
    site_mapping = _safe_short_json(policy.get("site_label_mapping", {}), limit=1600)
    extraction_orientation = _safe_short_json(extraction_result.get("label_orientation", {}), limit=800)
    candidates = qc_context.get("transform_candidates", sorted(VISUAL_QC_TRANSFORMS))
    return render_client_prompt(
        "local_vlm_visual_qc",
        task=task,
        candidates=", ".join(str(item) for item in candidates),
        label_rule=label_rule,
        site_mapping=site_mapping,
        extraction_orientation=extraction_orientation,
    )


def _parse_local_vlm_visual_qc_response(response_text: str) -> tuple[str, str]:
    try:
        data = json.loads(response_text)
    except json.JSONDecodeError:
        return "undecided", "local VLM returned invalid decision JSON"
    if (
        not isinstance(data, dict)
        or not isinstance(data.get("selected_transform"), str)
        or data["selected_transform"] not in VISUAL_QC_ALLOWED_SELECTIONS
    ):
        return "undecided", "local VLM did not return an explicit supported transform"
    reason = data.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        return "undecided", "local VLM decision is missing its reason"
    return data["selected_transform"], reason.strip()


def _local_vlm_qc_decision(
    *,
    client_id: str,
    output_template: dict[str, Any],
    selected_transform: str,
    reason: str,
    model: str,
    base_url: str,
    call_record: dict[str, Any],
) -> dict[str, Any]:
    selected_transform = selected_transform if selected_transform in VISUAL_QC_ALLOWED_SELECTIONS else "undecided"
    reviewed = selected_transform in VISUAL_QC_ALLOWED_SELECTIONS
    passed = selected_transform == "as_is"
    if selected_transform == "undecided":
        status = "needs_more_samples"
        recommendation = "review_more_visual_qc_samples_before_accepting_extraction"
    elif passed:
        status = "passed"
        recommendation = "accept_extraction_for_training"
    else:
        status = "failed"
        recommendation = f"rerun extraction or adapter with selected_transform={selected_transform}"
    counts = {selection: 0 for selection in sorted(VISUAL_QC_ALLOWED_SELECTIONS)}
    counts[selected_transform] = 1
    output = dict(output_template)
    output.update(
        {
            "schema_version": "fedready.extraction_visual_qc_decision.v1",
            "client_id": client_id,
            "status": status,
            "passed": passed,
            "reviewed": reviewed,
            "selected_transform": selected_transform,
            "reviewed_sample_count": 1 if reviewed else 0,
            "selected_transform_counts": counts,
            "consensus_required_count": 1,
            "consensus_reached": selected_transform != "undecided",
            "recommendation": recommendation,
            "reason": reason,
            "local_vlm_review": _local_vlm_review_metadata(model=model, base_url=base_url, call_record=call_record),
        }
    )
    if selected_transform == "undecided":
        _append_visual_qc_issue(output, "local VLM could not select a transform from the visual QC sheet")
    elif selected_transform != "as_is":
        _append_visual_qc_issue(output, f"local VLM selected non_as_is transform: {selected_transform}")
    return output


def _local_vlm_qc_failure(
    *,
    client_id: str,
    output_template: dict[str, Any],
    reason: str,
    model: str,
    base_url: str,
) -> dict[str, Any]:
    output = dict(output_template)
    output.update(
        {
            "schema_version": "fedready.extraction_visual_qc_decision.v1",
            "client_id": client_id,
            "status": "failed",
            "passed": False,
            "reviewed": False,
            "selected_transform": "undecided",
            "reviewed_sample_count": 0,
            "selected_transform_counts": {selection: 0 for selection in sorted(VISUAL_QC_ALLOWED_SELECTIONS)},
            "consensus_required_count": 1,
            "consensus_reached": False,
            "recommendation": "fix_local_vlm_visual_qc_before_training",
            "reason": reason,
            "local_vlm_review": {
                "schema_version": LOCAL_VLM_QC_SCHEMA,
                "backend": "local_vlm",
                "status": "failed",
                "model": model or None,
                "base_url": base_url,
            },
        }
    )
    _append_visual_qc_issue(output, reason)
    return output


def _local_vlm_review_metadata(*, model: str, base_url: str, call_record: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": LOCAL_VLM_QC_SCHEMA,
        "backend": "local_vlm",
        "status": call_record.get("status"),
        "model": model,
        "base_url": base_url,
        "image_digest": call_record.get("image_digest"),
        "prompt_digest": call_record.get("prompt_digest"),
        "response_digest": call_record.get("response_digest"),
        "image_size_bytes": call_record.get("image_size_bytes"),
        "latency_seconds": call_record.get("latency_seconds"),
        "call_logged_client_local": True,
    }


def _local_vlm_qc_image_path(artifact: dict[str, Any]) -> Path | None:
    value = artifact.get("candidate_sheet_path")
    if not isinstance(value, str) or not value.strip():
        return None
    path = Path(value).expanduser()
    if not path.exists() or not path.is_file():
        return None
    return path.resolve()


def _local_vlm_qc_log_path(image_path: Path) -> Path:
    env_path = os.environ.get("FEDREADY_VISUAL_QC_LOG_PATH", "").strip()
    if env_path:
        return Path(env_path).expanduser().resolve()
    return image_path.parent / "local_vlm_qc_calls.jsonl"


def _safe_short_json(value: Any, *, limit: int = 1000) -> str:
    try:
        text = json.dumps(value, sort_keys=True)
    except TypeError:
        text = str(value)
    if len(text) > limit:
        return text[:limit] + "...<truncated>"
    return text


def _env_positive_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if not value:
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    return parsed if parsed > 0 else default
