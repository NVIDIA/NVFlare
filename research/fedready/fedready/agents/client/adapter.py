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

"""Client-local adapter execution, provenance, state, and manifest validation."""

from __future__ import annotations

import ast
import json
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fedready.agents.common import adapter_source_digest
from fedready.data.contracts import classification as classification_contract
from fedready.data.contracts import (
    generated_contract_field_names,
    generated_contract_record_type,
    infer_record_type,
    manifest_record_type,
)
from PIL import Image, UnidentifiedImageError

ADAPTER_PREFLIGHT_STATE_FILE = "fedready_adapter_preflight_state.json"
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
            "schema_version": "fedready.local_adapter_runtime_state.v1",
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
    if manifest.get("schema_version") != "fedready.local_adapter_manifest.v1":
        raise ValueError("adapter manifest schema_version must be fedready.local_adapter_manifest.v1")
    records = manifest.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("adapter manifest must contain non-empty records")
    if len(records) > max_records:
        raise ValueError(f"adapter manifest contains {len(records)} records, exceeding bound max_records={max_records}")
    generated_roots = _normalized_generated_roots(allowed_generated_roots)
    record_type = manifest_record_type(manifest, generated_contract=data_contract)
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
        if infer_record_type(record) != "segmentation":
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
        if infer_record_type(record) != "classification":
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
    if _is_reserved_fedready_label_source(relative):
        raise ValueError(
            f"adapter manifest record {record_index} label source is a FedReady-generated artifact, not raw client evidence"
        )
    try:
        current = _client_file_identity(path)
    except OSError as exc:
        raise ValueError(f"adapter manifest record {record_index} label source is no longer readable") from exc
    if current != snapshot.files[relative]:
        raise ValueError(f"adapter manifest record {record_index} label source changed after agent execution began")


def _is_reserved_fedready_label_source(relative_path: str) -> bool:
    parts = Path(relative_path).parts
    return any(part.startswith(".fedready") for part in parts)
