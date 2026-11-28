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
execution, and manifest validation that determine whether an implementation is accepted."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import time
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from fedready.agents.guardrail import _short_error
from fedready.utils.logging import timestamp_utc
from PIL import Image


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


LOCAL_VLM_MAX_TRANSPORT_ATTEMPTS = 3
LOCAL_VLM_RETRY_DELAY_SECONDS = 1.0
LOCAL_VLM_TRANSPORT_MAX_LONG_SIDE = 768
LOCAL_VLM_TRANSPORT_JPEG_QUALITY = 90
LOCAL_VISION_API_BASE_URL = "http://127.0.0.1:8001/v1"
DEFAULT_LOCAL_VISION_MODEL = "Qwen/Qwen3-VL-8B-Instruct"


def _local_vlm_model_name() -> str:
    value = os.environ.get("FEDREADY_VISION_AGENT_MODEL")
    if value is None:
        return DEFAULT_LOCAL_VISION_MODEL
    return value.strip()


def _query_local_vlm_image(
    *,
    base_url: str,
    model: str,
    image_path: Path,
    prompt: str,
    api_key_env: str,
    max_tokens: int,
    log_path: Path,
    reference_image_path: Path | None = None,
) -> tuple[str, dict[str, Any]]:
    for attempt in range(1, LOCAL_VLM_MAX_TRANSPORT_ATTEMPTS + 1):
        try:
            return _query_local_vlm_image_once(
                base_url=base_url,
                model=model,
                image_path=image_path,
                prompt=prompt,
                api_key_env=api_key_env,
                max_tokens=max_tokens,
                log_path=log_path,
                reference_image_path=reference_image_path,
            )
        except Exception as exc:
            if attempt >= LOCAL_VLM_MAX_TRANSPORT_ATTEMPTS or not _is_retryable_local_vlm_error(exc):
                raise
            time.sleep(LOCAL_VLM_RETRY_DELAY_SECONDS * attempt)
    raise AssertionError("local VLM transport retry loop exited unexpectedly")


def _is_retryable_local_vlm_error(exc: Exception) -> bool:
    if isinstance(exc, HTTPError):
        return exc.code == 429 or 500 <= exc.code < 600
    return isinstance(exc, (URLError, TimeoutError, ConnectionError))


def _local_vlm_transport_image(image_path: Path, source_bytes: bytes) -> tuple[bytes, str]:
    try:
        with Image.open(BytesIO(source_bytes)) as source:
            image = source.convert("RGB")
        if max(image.size) > LOCAL_VLM_TRANSPORT_MAX_LONG_SIDE:
            image.thumbnail(
                (LOCAL_VLM_TRANSPORT_MAX_LONG_SIDE, LOCAL_VLM_TRANSPORT_MAX_LONG_SIDE),
                Image.Resampling.LANCZOS,
            )
        output = BytesIO()
        image.save(
            output,
            format="JPEG",
            quality=LOCAL_VLM_TRANSPORT_JPEG_QUALITY,
            optimize=True,
        )
        return output.getvalue(), "image/jpeg"
    except Exception:
        return source_bytes, _local_vlm_mime_type(image_path)


def _query_local_vlm_image_once(
    *,
    base_url: str,
    model: str,
    image_path: Path,
    prompt: str,
    api_key_env: str,
    max_tokens: int,
    log_path: Path,
    reference_image_path: Path | None = None,
) -> tuple[str, dict[str, Any]]:
    started_at = timestamp_utc()
    start_time = time.monotonic()
    source_image_bytes = image_path.read_bytes()
    image_bytes, mime_type = _local_vlm_transport_image(image_path, source_image_bytes)
    record: dict[str, Any] = {
        "schema_version": "fedready.local_vlm_call.v1",
        "started_at": started_at,
        "status": "started",
        "base_url": base_url,
        "model": model,
        "prompt": prompt,
        "prompt_digest": _digest_text(prompt),
        "prompt_length": len(prompt),
        "image_path": str(image_path),
        "image_digest": _digest_bytes(source_image_bytes),
        "image_size_bytes": len(source_image_bytes),
        "image_mime_type": _local_vlm_mime_type(image_path),
        "transport_image_digest": _digest_bytes(image_bytes),
        "transport_image_size_bytes": len(image_bytes),
        "transport_image_mime_type": mime_type,
        "transport_max_long_side": LOCAL_VLM_TRANSPORT_MAX_LONG_SIDE,
        "max_tokens": max_tokens,
        "log_path": str(log_path),
    }
    reference_transport: tuple[bytes, str] | None = None
    if reference_image_path is not None:
        source_reference_bytes = reference_image_path.read_bytes()
        reference_transport = _local_vlm_transport_image(reference_image_path, source_reference_bytes)
        record.update(
            {
                "reference_image_path": str(reference_image_path),
                "reference_image_digest": _digest_bytes(source_reference_bytes),
                "reference_image_size_bytes": len(source_reference_bytes),
                "reference_image_mime_type": _local_vlm_mime_type(reference_image_path),
                "transport_reference_image_digest": _digest_bytes(reference_transport[0]),
                "transport_reference_image_size_bytes": len(reference_transport[0]),
                "transport_reference_image_mime_type": reference_transport[1],
            }
        )
    try:
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        if reference_transport is not None:
            reference_bytes, reference_mime_type = reference_transport
            reference_encoded = base64.b64encode(reference_bytes).decode("ascii")
            content.extend(
                [
                    {"type": "text", "text": "Reference example for the requested task:"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{reference_mime_type};base64,{reference_encoded}"},
                    },
                    {"type": "text", "text": "Candidate preflight QC image to evaluate:"},
                ]
            )
        encoded = base64.b64encode(image_bytes).decode("ascii")
        content.append({"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{encoded}"}})
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": content,
                }
            ],
            "temperature": 0,
            "max_tokens": max_tokens,
            "stream": False,
        }
        headers = {"Content-Type": "application/json"}
        if api_key_env:
            api_key = os.environ.get(api_key_env)
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
        url = base_url.rstrip("/")
        if not url.endswith("/chat/completions"):
            url = f"{url}/chat/completions"
        request = Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
        with urlopen(request, timeout=300) as response:
            response_body = response.read().decode("utf-8")
            response_status = response.getcode()
        response_text = _extract_chat_completion_text(response_body)
        record.update(
            {
                "status": "completed",
                "completed_at": timestamp_utc(),
                "latency_seconds": round(time.monotonic() - start_time, 6),
                "response_status_code": response_status,
                "response_digest": _digest_text(response_body),
                "response_body": response_body,
            }
        )
        _safe_append_jsonl(log_path, record)
        return response_text, record
    except Exception as exc:
        record.update(
            {
                "status": "failed",
                "failed_at": timestamp_utc(),
                "latency_seconds": round(time.monotonic() - start_time, 6),
                "error": {"type": type(exc).__name__, "message": _short_error(exc)},
            }
        )
        _safe_append_jsonl(log_path, record)
        raise


def _extract_chat_completion_text(response_body: str) -> str:
    try:
        response_json = json.loads(response_body)
    except json.JSONDecodeError:
        return response_body
    choices = response_json.get("choices")
    if not isinstance(choices, list) or not choices:
        return response_body
    first = choices[0]
    if not isinstance(first, dict):
        return response_body
    message = first.get("message")
    if not isinstance(message, dict):
        return response_body
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        if parts:
            return "\n".join(parts)
    return response_body


def _is_local_vlm_base_url(base_url: str) -> bool:
    try:
        parsed = urlsplit(base_url)
    except ValueError:
        return False
    return parsed.scheme == "http" and parsed.hostname in {"127.0.0.1", "localhost", "::1"}


def _local_vlm_mime_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".png":
        return "image/png"
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    raise ValueError(f"Unsupported image suffix for local VLM visual QC: {suffix or '<none>'}")


def _digest_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _digest_text(text: str) -> str:
    return _digest_bytes(text.encode("utf-8"))


def _safe_append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
    except OSError:
        return
