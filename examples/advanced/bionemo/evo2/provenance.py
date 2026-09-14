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
"""Content identities for reproducible Evo2 training and evaluation inputs."""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping
from pathlib import Path

CONTINUATION_SIGNATURE_FORMAT_VERSION = 1
_CONTINUATION_SIGNATURE_KEYS = {"format_version", "payload", "sha256"}


def _canonical_json_bytes(value) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Value is not canonical JSON data: {exc}") from exc


def make_continuation_signature(payload: Mapping) -> dict:
    """Return a canonical, self-verifying continuation signature."""

    if not isinstance(payload, Mapping) or not payload:
        raise ValueError("Continuation signature payload must be a non-empty mapping.")
    canonical_payload = json.loads(_canonical_json_bytes(payload))
    return {
        "format_version": CONTINUATION_SIGNATURE_FORMAT_VERSION,
        "payload": canonical_payload,
        "sha256": hashlib.sha256(_canonical_json_bytes(canonical_payload)).hexdigest(),
    }


def validate_continuation_signature(signature: Mapping, *, context: str = "Continuation signature") -> dict:
    """Validate and normalize a continuation signature without trusting its digest."""

    if not isinstance(signature, Mapping) or set(signature) != _CONTINUATION_SIGNATURE_KEYS:
        received = set(signature) if isinstance(signature, Mapping) else type(signature).__name__
        raise ValueError(f"{context} has invalid keys: {received}.")
    if type(signature["format_version"]) is not int or (
        signature["format_version"] != CONTINUATION_SIGNATURE_FORMAT_VERSION
    ):
        raise ValueError(
            f"{context} has unsupported format_version={signature['format_version']!r}; "
            f"expected {CONTINUATION_SIGNATURE_FORMAT_VERSION}."
        )
    payload = signature["payload"]
    if not isinstance(payload, Mapping) or not payload:
        raise ValueError(f"{context} payload must be a non-empty mapping.")
    digest = signature["sha256"]
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ValueError(f"{context} has an invalid SHA-256 digest.")
    normalized = make_continuation_signature(payload)
    if normalized["sha256"] != digest:
        raise ValueError(f"{context} SHA-256 digest does not match its payload.")
    return normalized


def sha256_file(path: str | os.PathLike[str]) -> str:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"File not found: {resolved}")
    digest = hashlib.sha256()
    with resolved.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_identity(path: str | os.PathLike[str]) -> dict:
    resolved = Path(path).resolve()
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "bytes": resolved.stat().st_size,
    }


def sha256_directory(path: str | os.PathLike[str]) -> str:
    root = Path(path).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Directory not found: {root}")
    files = sorted(candidate for candidate in root.rglob("*") if candidate.is_file())
    if not files:
        raise ValueError(f"Directory contains no files: {root}")
    digest = hashlib.sha256()
    for file_path in files:
        relative_path = file_path.relative_to(root).as_posix().encode()
        content_length = file_path.stat().st_size
        digest.update(len(relative_path).to_bytes(8, byteorder="big"))
        digest.update(relative_path)
        digest.update(content_length.to_bytes(8, byteorder="big"))
        with file_path.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def directory_identity(path: str | os.PathLike[str]) -> dict:
    root = Path(path).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Directory not found: {root}")
    files = [candidate for candidate in root.rglob("*") if candidate.is_file()]
    return {
        "path": str(root),
        "sha256": sha256_directory(root),
        "files": len(files),
        "bytes": sum(candidate.stat().st_size for candidate in files),
    }


def jsonl_identity(
    path: str | os.PathLike[str], *, expected_rows: int | None = None, label: str = "JSONL input"
) -> dict:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} file not found: {resolved}")
    digest = hashlib.sha256()
    rows = 0
    with resolved.open("rb") as file:
        for line_number, line in enumerate(file, start=1):
            digest.update(line)
            if not line.strip():
                raise ValueError(f"{label} contains a blank row at {resolved}:{line_number}.")
            try:
                row = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError(f"{label} contains invalid JSON at {resolved}:{line_number}.") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{label} row {resolved}:{line_number} must contain a JSON object.")
            rows += 1
    if rows == 0:
        raise ValueError(f"{label} is empty: {resolved}")
    if expected_rows is not None and rows != expected_rows:
        raise ValueError(
            f"{label} row count does not match the manifest: expected {expected_rows}, observed {rows} at {resolved}."
        )
    return {
        "path": str(resolved),
        "sha256": digest.hexdigest(),
        "bytes": resolved.stat().st_size,
        "rows": rows,
    }


def _identity_payload(identity: dict, *, directory: bool, label: str) -> dict:
    if not isinstance(identity, dict):
        raise ValueError(f"{label} identity must be a dictionary.")
    required = ("sha256", "bytes", "files") if directory else ("sha256", "bytes")
    missing = [field for field in required if field not in identity]
    if missing:
        raise ValueError(f"{label} identity is missing fields: {missing}.")
    payload = {field: identity[field] for field in required}
    if not isinstance(payload["sha256"], str) or len(payload["sha256"]) != 64:
        raise ValueError(f"{label} identity has an invalid SHA-256 digest.")
    if type(payload["bytes"]) is not int or payload["bytes"] < 0:
        raise ValueError(f"{label} identity has an invalid byte count.")
    if directory and (type(payload["files"]) is not int or payload["files"] <= 0):
        raise ValueError(f"{label} identity has an invalid file count.")
    if "rows" in identity:
        if type(identity["rows"]) is not int or identity["rows"] <= 0:
            raise ValueError(f"{label} identity has an invalid row count.")
        payload["rows"] = identity["rows"]
    return payload


def resolve_initialization_metadata(checkpoint_metadata: dict) -> dict:
    """Return original initialization metadata from an initial or once-continued checkpoint."""

    if not isinstance(checkpoint_metadata, dict):
        raise ValueError("Trainable checkpoint metadata must be a dictionary.")
    if "initialization" not in checkpoint_metadata:
        return checkpoint_metadata
    initialization = checkpoint_metadata["initialization"]
    if not isinstance(initialization, dict):
        raise ValueError("Trainable checkpoint metadata field 'initialization' must be a dictionary.")
    if "initialization" in initialization:
        raise ValueError("Trainable checkpoint metadata contains more than one nested initialization layer.")
    return initialization


def validate_initialization_metadata(
    checkpoint_metadata: dict,
    *,
    backend: str,
    peft_mode: str,
    seed: int,
    seq_length: int,
    lora_dim: int,
    lora_alpha: int,
    lora_dropout: float,
    lora_target_modules: tuple[str, ...],
    base_checkpoint_identity: dict | None,
    classifier_file_identity: dict | None,
    exchange_dtype: str,
    initialization_data_identity: dict | None = None,
) -> dict:
    """Bind a trainable checkpoint to its model topology and source artifacts."""

    initialization = resolve_initialization_metadata(checkpoint_metadata)

    expected_settings = {
        "backend": backend,
        "exchange_dtype": exchange_dtype,
        "peft_mode": peft_mode,
        "seed": seed,
        "seq_length": seq_length,
        "lora_dim": lora_dim if peft_mode == "lora" else None,
        "lora_alpha": lora_alpha if peft_mode == "lora" else None,
        "lora_dropout": lora_dropout if peft_mode == "lora" else None,
        "lora_target_modules": list(lora_target_modules) if peft_mode == "lora" else [],
    }
    mismatches = {
        field: {"expected": expected, "observed": initialization.get(field)}
        for field, expected in expected_settings.items()
        if initialization.get(field) != expected
    }
    if mismatches:
        raise ValueError(f"Trainable checkpoint initialization settings do not match this run: {mismatches}.")

    training_inputs = initialization.get("training_inputs")
    if not isinstance(training_inputs, dict):
        raise ValueError("Trainable checkpoint metadata is missing dictionary field 'training_inputs'.")

    identity_pairs = (
        ("base_checkpoint", base_checkpoint_identity, True),
        ("classifier_file", classifier_file_identity, False),
        ("data_file", initialization_data_identity, False),
    )
    for name, expected_identity, is_directory in identity_pairs:
        observed_identity = training_inputs.get(name)
        if expected_identity is None:
            if backend == "bionemo" and name in ("base_checkpoint", "classifier_file"):
                raise ValueError(f"Current BioNeMo run is missing the {name} content identity.")
            if backend == "mock" and observed_identity is not None:
                raise ValueError(f"Mock initialization must not contain a {name} content identity.")
            continue
        expected_payload = _identity_payload(
            expected_identity,
            directory=is_directory,
            label=f"Current {name}",
        )
        observed_payload = _identity_payload(
            observed_identity,
            directory=is_directory,
            label=f"Checkpoint {name}",
        )
        if observed_payload != expected_payload:
            raise ValueError(
                f"Trainable checkpoint {name} content identity does not match this run: "
                f"expected {expected_payload}, observed {observed_payload}."
            )
    return initialization
