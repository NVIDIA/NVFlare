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

"""Generated training-code validation and preflight-bound package integrity."""

from __future__ import annotations

import hashlib
import py_compile
from pathlib import Path
from typing import Any

from fedready.prompts import render_server_prompt

TRAINING_PACKAGE_NAME = "fedready_task_training"
TRAINING_ENTRY_SCRIPT = "train.py"

TRAINING_CODE_SPEC_SCHEMA_VERSION = "fedready.training_code_spec.v1"
LOCAL_TRAINING_SIMULATION_SCHEMA_VERSION = "fedready.local_training_simulation.v1"
MOCK_TRAINING_DATASET_SCHEMA_VERSION = "fedready.mock_training_dataset.v1"
TRAINING_PACKAGE_INTEGRITY_SCHEMA_VERSION = "fedready.training_package_integrity.v1"


def build_training_code_validation_feedback(
    *,
    spec: dict[str, Any],
    error: Exception,
    attempt: int,
    max_attempts: int,
    code_workspace: str | Path,
    require_local_simulation: bool = False,
) -> dict[str, Any]:
    """Build local validation feedback for an agent revision pass."""

    errors = _training_code_spec_validation_errors(
        spec,
        code_workspace=code_workspace,
        require_local_simulation=require_local_simulation,
    )
    if not errors:
        errors = [{"kind": type(error).__name__, "message": str(error)}]
    code_workspace_path = Path(code_workspace).resolve()
    expected_package_dir = code_workspace_path / TRAINING_PACKAGE_NAME
    return {
        "schema_version": "fedready.training_code_validation_feedback.v1",
        "status": "failed",
        "attempt": attempt,
        "max_attempts": max_attempts,
        "error_type": type(error).__name__,
        "error_message": str(error),
        "errors": errors,
        "code_workspace": str(code_workspace_path),
        "expected_package_dir": str(expected_package_dir),
        "expected_schema_version": TRAINING_CODE_SPEC_SCHEMA_VERSION,
        "expected_status": "implemented",
        "previous_output_summary": {
            "schema_version": (spec.get("schema_version") if isinstance(spec, dict) else None),
            "status": spec.get("status") if isinstance(spec, dict) else None,
            "package_dir": spec.get("package_dir") if isinstance(spec, dict) else None,
            "entry_script": (spec.get("entry_script") if isinstance(spec, dict) else None),
            "model_class_path": (spec.get("model_class_path") if isinstance(spec, dict) else None),
            "selected_reference_example_path": (
                _selected_reference_example_path(spec) if isinstance(spec, dict) else None
            ),
            "reference_api_evidence_count": (
                len(_reference_api_evidence(spec))
                if isinstance(spec, dict) and isinstance(_reference_api_evidence(spec), list)
                else 0
            ),
        },
        "required_fields_to_preserve": [
            "schema_version",
            "status",
            "package_dir",
            "entry_script",
            "model_class_path",
            "selected_reference_example_path",
            "reference_api_evidence",
            "mock_record_template",
        ],
        "agent_instruction": render_server_prompt("training_validation_agent_instruction"),
    }


def validate_training_code_spec(
    spec: dict[str, Any],
    *,
    code_workspace: str | Path | None = None,
    require_local_simulation: bool = True,
) -> dict[str, Any]:
    errors = _training_code_spec_validation_errors(
        spec,
        code_workspace=code_workspace,
        require_local_simulation=require_local_simulation,
    )
    if errors:
        message = "; ".join(str(error.get("message")) for error in errors)
        if any(str(error.get("kind", "")).endswith("not_found") for error in errors):
            raise FileNotFoundError(message)
        raise ValueError(message)
    validated = dict(spec)
    package_dir = _resolve_training_package_dir(str(spec["package_dir"]), code_workspace=code_workspace)
    validated["package_dir"] = str(package_dir.resolve())
    return validated


def _training_code_spec_validation_errors(
    spec: dict[str, Any],
    *,
    code_workspace: str | Path | None = None,
    require_local_simulation: bool = True,
) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    if not isinstance(spec, dict):
        return [
            {
                "kind": "invalid_spec",
                "message": "training code spec must be a JSON object",
            }
        ]
    if spec.get("deterministic_ground_truth") is True:
        errors.append(
            {
                "kind": "deterministic_training_spec",
                "message": "deterministic or reference training-code specs are unsupported",
            }
        )
    if spec.get("schema_version") != TRAINING_CODE_SPEC_SCHEMA_VERSION:
        errors.append(
            {
                "kind": "schema_mismatch",
                "message": f"expected {TRAINING_CODE_SPEC_SCHEMA_VERSION}",
                "actual": spec.get("schema_version"),
            }
        )
    if spec.get("status") != "implemented":
        errors.append(
            {
                "kind": "status_mismatch",
                "message": "training code spec status must be implemented",
                "actual": spec.get("status"),
            }
        )
    for key in ("package_dir", "entry_script", "model_class_path"):
        value = spec.get(key)
        if not isinstance(value, str) or not value:
            error = {
                "kind": "missing_required_key",
                "message": f"training code spec {key} must be a non-empty string",
                "key": key,
                "actual_type": type(value).__name__,
            }
            if isinstance(value, dict):
                error["actual_keys"] = sorted(str(item) for item in value)
            errors.append(error)
    if errors:
        return errors

    package_dir = _resolve_training_package_dir(str(spec["package_dir"]), code_workspace=code_workspace)
    if not package_dir.exists() or not package_dir.is_dir():
        return [
            {
                "kind": "package_dir_not_found",
                "message": f"training code package_dir not found: {package_dir}",
                "package_dir": str(spec["package_dir"]),
                "resolved_package_dir": str(package_dir),
            }
        ]

    if package_dir.name != TRAINING_PACKAGE_NAME or (
        code_workspace is not None and package_dir.parent != Path(code_workspace).resolve()
    ):
        errors.append(
            {
                "kind": "package_layout_invalid",
                "message": f"package_dir must be the {TRAINING_PACKAGE_NAME} directory under code_workspace",
            }
        )
    if spec["entry_script"] != TRAINING_ENTRY_SCRIPT:
        errors.append(
            {
                "kind": "entry_script_invalid",
                "message": f"entry_script must be {TRAINING_ENTRY_SCRIPT}, relative to package_dir",
            }
        )
    model_parts = spec["model_class_path"].split(".")
    if (
        len(model_parts) != 3
        or model_parts[:2] != [TRAINING_PACKAGE_NAME, "model"]
        or not model_parts[-1].isidentifier()
    ):
        errors.append(
            {
                "kind": "model_class_path_invalid",
                "message": f"model_class_path must be {TRAINING_PACKAGE_NAME}.model.<ModelClass>",
            }
        )
    for filename in ("__init__.py", TRAINING_ENTRY_SCRIPT, "model.py"):
        if not (package_dir / filename).is_file():
            errors.append(
                {
                    "kind": "required_file_not_found",
                    "message": f"generated training package is missing {filename}",
                }
            )

    model_args = spec.get("model_args", {})
    if not isinstance(model_args, dict):
        errors.append(
            {
                "kind": "model_args_type",
                "message": "training code spec model_args must be an object",
            }
        )
    mock_record_template = spec.get("mock_record_template")
    if mock_record_template is not None and not isinstance(mock_record_template, dict):
        errors.append(
            {
                "kind": "mock_record_template_type",
                "message": "training code spec mock_record_template must be an object when provided",
            }
        )

    py_files = sorted(package_dir.rglob("*.py"))
    if len(py_files) <= 50:
        for py_file in py_files:
            try:
                py_compile.compile(str(py_file), doraise=True)
            except py_compile.PyCompileError as exc:
                errors.append(
                    {
                        "kind": "python_compile_failed",
                        "message": f"generated Python file failed to compile: {py_file}",
                        "path": str(py_file),
                        "details": str(exc),
                    }
                )
    if require_local_simulation and not errors:
        errors.extend(_training_local_simulation_errors(spec))
        errors.extend(_training_package_integrity_errors(spec, package_dir=package_dir))
    return errors


def build_training_package_integrity(package_dir: str | Path) -> dict[str, Any]:
    """Return a deterministic content manifest for deployable training files."""

    root = Path(package_dir).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"training package directory is missing: {root}")
    files: list[dict[str, Any]] = []
    aggregate = hashlib.sha256()
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        relative = path.relative_to(root)
        if "__pycache__" in relative.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        content = path.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        relative_text = relative.as_posix()
        aggregate.update(relative_text.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(digest.encode("ascii"))
        aggregate.update(b"\0")
        files.append(
            {
                "path": relative_text,
                "sha256": f"sha256:{digest}",
                "size_bytes": len(content),
            }
        )
    if not files:
        raise ValueError(f"training package contains no deployable files: {root}")
    return {
        "schema_version": TRAINING_PACKAGE_INTEGRITY_SCHEMA_VERSION,
        "sha256": f"sha256:{aggregate.hexdigest()}",
        "file_count": len(files),
        "files": files,
    }


def _training_package_integrity_errors(spec: dict[str, Any], *, package_dir: Path) -> list[dict[str, Any]]:
    """Bind local preflight approval to the exact package being exported."""

    recorded = spec.get("package_integrity")
    current = build_training_package_integrity(package_dir)
    if not isinstance(recorded, dict):
        return [
            {
                "kind": "package_integrity_missing",
                "message": "training package must be revalidated by local SimEnv preflight before export",
            }
        ]
    if recorded != current:
        return [
            {
                "kind": "package_integrity_mismatch",
                "message": "training package changed after local SimEnv preflight; run preflight again before export",
                "preflight_sha256": recorded.get("sha256"),
                "current_sha256": current["sha256"],
            }
        ]
    return []


def _training_local_simulation_errors(spec: dict[str, Any]) -> list[dict[str, Any]]:
    """Require live agent specs to include a FedReady local SimEnv preflight report."""

    simulation = spec.get("local_simulation")
    if not isinstance(simulation, dict):
        return [
            {
                "kind": "local_simulation_missing",
                "message": (
                    "live training code spec must include local_simulation from FedReady local NVFlare preflight"
                ),
            }
        ]

    errors: list[dict[str, Any]] = []
    if simulation.get("schema_version") != LOCAL_TRAINING_SIMULATION_SCHEMA_VERSION:
        errors.append(
            {
                "kind": "local_simulation_schema_mismatch",
                "message": f"local_simulation.schema_version must be {LOCAL_TRAINING_SIMULATION_SCHEMA_VERSION}",
                "actual": simulation.get("schema_version"),
            }
        )
    if simulation.get("status") != "passed":
        errors.append(
            {
                "kind": "local_simulation_not_passed",
                "message": "FedReady local NVFlare SimEnv preflight must pass before training job export",
                "actual": simulation.get("status"),
            }
        )
    if simulation.get("client_deployment") is not False:
        errors.append(
            {
                "kind": "local_simulation_client_deployment_not_false",
                "message": "local_simulation.client_deployment must be false; pre-export simulation must not deploy to clients",
                "actual": simulation.get("client_deployment"),
            }
        )
    nonempty_metric_artifacts = simulation.get("nonempty_metric_artifacts")
    has_metric_artifact = (
        simulation.get("metric_artifact_available") is True
        and isinstance(nonempty_metric_artifacts, list)
        and any(isinstance(path, str) and path for path in nonempty_metric_artifacts)
    )
    if not has_metric_artifact:
        errors.append(
            {
                "kind": "local_simulation_metric_artifact_missing",
                "message": "local_simulation must include a non-empty metric artifact from the local NVFlare preflight",
                "metric_artifact_available": simulation.get("metric_artifact_available"),
                "nonempty_metric_artifacts": nonempty_metric_artifacts,
            }
        )
    dataset_contract = simulation.get("dataset_contract")
    if isinstance(dataset_contract, dict) and dataset_contract.get("schema_version") not in (
        None,
        MOCK_TRAINING_DATASET_SCHEMA_VERSION,
    ):
        errors.append(
            {
                "kind": "local_simulation_dataset_contract_schema_mismatch",
                "message": f"local_simulation.dataset_contract.schema_version must be {MOCK_TRAINING_DATASET_SCHEMA_VERSION}",
                "actual": dataset_contract.get("schema_version"),
            }
        )
    return errors


def _selected_reference_example_path(spec: dict[str, Any]) -> str | None:
    value = spec.get("selected_reference_example_path")
    return value.strip() if isinstance(value, str) and value.strip() else None


def _reference_api_evidence(spec: dict[str, Any]) -> Any:
    return spec.get("reference_api_evidence")


def _resolve_training_package_dir(package_dir: str, *, code_workspace: str | Path | None = None) -> Path:
    path = Path(package_dir).expanduser()
    if not path.is_absolute() and code_workspace is not None:
        path = Path(code_workspace) / path
    return path.resolve()


def _safe_code_spec(spec: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": spec.get("schema_version"),
        "status": spec.get("status"),
        "package_dir": spec.get("package_dir"),
        "entry_script": spec.get("entry_script"),
        "model_class_path": spec.get("model_class_path"),
        "model_args": spec.get("model_args", {}),
        "framework": spec.get("framework"),
        "base_reference": spec.get("base_reference"),
        "selected_reference_example_path": spec.get("selected_reference_example_path"),
        "reference_api_evidence": spec.get("reference_api_evidence"),
        "metric_artifacts": spec.get("metric_artifacts"),
        "local_simulation": spec.get("local_simulation"),
        "package_integrity": spec.get("package_integrity"),
        "safe_to_share": False,
        "local_paths_redacted_for_server_summary": True,
    }
