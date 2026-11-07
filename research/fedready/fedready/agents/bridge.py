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

"""Codex backend bridge for FedReady simulations."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from fedready.agents.local_adapter import TASK_EXAMPLE_DIR
from fedready.prompts import render_server_prompt
from fedready.utils.io import atomic_write_json
from fedready.utils.logging import payload_digest, timestamp_utc

FILE_AGENT_REQUEST_SCHEMA = "fedready.file_agent_request.v1"
FILE_AGENT_RESPONSE_SCHEMA = "fedready.file_agent_response.v1"
LOCAL_VISION_API_BASE_URL = "http://127.0.0.1:8001/v1"
# The research contribution deliberately exposes one live backend. Keeping the
# backend surface narrow makes agent behavior and failure accounting comparable.
DEFAULT_CODEX_NVFLARE_SKILL = "nvflare-orient"
DISABLED_CODEX_SKILL_VALUES = {"", "0", "false", "none", "off", "disabled"}
DEFAULT_CODEX_SANDBOX_RETRY_ATTEMPTS = 1


def _request_uses_nvflare_orientation(*, phase: str, action: str) -> bool:
    return phase == "fl_training" and action == "SERVER.IMPLEMENT_TRAINING_CODE"


def _skill_names_from_raw(raw: str) -> list[str]:
    names = [name.strip() for name in raw.split(",")]
    return [name for name in names if name and name.lower() not in DISABLED_CODEX_SKILL_VALUES]


def _codex_requested_skill_names() -> list[str]:
    raw = os.environ.get("FEDREADY_CODEX_NVFLARE_SKILL", DEFAULT_CODEX_NVFLARE_SKILL)
    return _skill_names_from_raw(raw)


def _codex_skill_path(name: str) -> Path:
    repository_root = Path(__file__).resolve().parents[4]
    return repository_root / "skills" / name / "SKILL.md"


def _load_codex_request_local_skill(name: str) -> tuple[str, str]:
    if name != DEFAULT_CODEX_NVFLARE_SKILL:
        raise RuntimeError(
            f"Unsupported Codex request-local NVFlare skill: {name}. "
            f"Supported value is {DEFAULT_CODEX_NVFLARE_SKILL!r}, or set FEDREADY_CODEX_NVFLARE_SKILL=none."
        )
    skill_path = _codex_skill_path(name)
    if skill_path.exists():
        return skill_path.read_text(encoding="utf-8"), str(skill_path)
    raise RuntimeError(
        f"Missing Codex NVFlare skill {name!r}: expected {skill_path}. No static prompt fallback is available. "
        "Restore the repository skill or set "
        "FEDREADY_CODEX_NVFLARE_SKILL=none to intentionally compare without it."
    )


def _package_codex_request_local_skills(request_dir: Path, *, phase: str, action: str) -> list[dict[str, str]]:
    if not _request_uses_nvflare_orientation(phase=phase, action=action):
        return []
    packaged: list[dict[str, str]] = []
    for name in _codex_requested_skill_names():
        content, source = _load_codex_request_local_skill(name)
        rel_path = Path("codex_worker") / "skills" / name / "SKILL.md"
        dest = request_dir / rel_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(content, encoding="utf-8")
        packaged.append(
            {
                "name": name,
                "status": "packaged_request_local",
                "loader": "fedready.codex.request_local_skill_file",
                "path": str(rel_path),
                "source": source,
                "content_digest": payload_digest(content),
            }
        )
    return packaged


def _codex_sandbox_retry_attempts() -> int:
    raw = os.environ.get("FEDREADY_CODEX_SANDBOX_RETRY_ATTEMPTS")
    if raw is None:
        return DEFAULT_CODEX_SANDBOX_RETRY_ATTEMPTS
    return max(int(raw), 0)


def _is_codex_sandbox_unavailable(text: str) -> bool:
    normalized = " ".join(text.casefold().split())
    return (
        "filesystem sandbox failed before command execution" in normalized
        or "filesystem sandbox is currently unavailable to shell commands" in normalized
        or ("bwrap:" in normalized and "operation not permitted" in normalized)
        or "failed rtm_newaddr" in normalized
    )


def _is_codex_startup_transient(text: str) -> bool:
    normalized = " ".join(text.casefold().split())
    return (
        "timed out waiting for cloud config bundle" in normalized
        or "cloud config bundle" in normalized
        and any(token in normalized for token in ("timeout", "timed out", "temporarily unavailable"))
    )


def _is_retryable_codex_worker_startup_failure(exc: Exception) -> bool:
    text = str(exc)
    return _is_codex_sandbox_unavailable(text) or _is_codex_startup_transient(text)


@dataclass(frozen=True)
class CodexWorkerBackend:
    """Run FedReady requests through workflow-owned Codex worker identities.

    The worker identity is persistent across requests, but each request is run by a
    scoped ``codex exec`` invocation rooted at the request directory. This avoids
    inheriting a supervising Codex conversation while keeping explicit server/client
    worker lifecycle state under ``agent_workers``.
    """

    run_dir: str | Path
    session_id: str
    model: str | None = None
    binary: str = "codex"
    timeout_seconds: float = 3600.0
    sandbox: str = "workspace-write"
    approval_policy: str = "never"
    ignore_rules: bool = True
    ephemeral: bool = False
    sandbox_retry_attempts: int = DEFAULT_CODEX_SANDBOX_RETRY_ATTEMPTS

    @classmethod
    def from_env(
        cls,
        *,
        run_dir: str | Path,
        session_id: str,
        timeout_seconds: float = 3600.0,
    ) -> "CodexWorkerBackend":
        return cls(
            run_dir=run_dir,
            session_id=session_id,
            model=_env_text("FEDREADY_CODEX_MODEL"),
            binary=os.environ.get("FEDREADY_CODEX_BINARY", "codex"),
            timeout_seconds=float(os.environ.get("FEDREADY_CODEX_TIMEOUT_SECONDS", str(timeout_seconds))),
            sandbox=os.environ.get("FEDREADY_CODEX_SANDBOX", "workspace-write"),
            approval_policy=os.environ.get("FEDREADY_CODEX_APPROVAL_POLICY", "never"),
            ignore_rules=_env_flag("FEDREADY_CODEX_IGNORE_RULES", default=True),
            ephemeral=_env_flag("FEDREADY_CODEX_EPHEMERAL", default=False),
            sandbox_retry_attempts=_codex_sandbox_retry_attempts(),
        )

    def request(
        self,
        *,
        role: str,
        action: str,
        phase: str,
        input_schema: str,
        output_schema: str,
        prompt: str,
        context: dict[str, Any],
        site_id: str | None = None,
        correlation_id: str | None = None,
        _sandbox_retry_attempt: int = 0,
        _retry_of_request_id: str | None = None,
    ) -> dict[str, Any]:
        retry_context = context
        execution_context = context
        request_id = _request_id(role=role, action=action, site_id=site_id)
        run_dir = Path(self.run_dir).expanduser().resolve()
        request_dir = run_dir / "agent_mailbox" / role / (site_id or "server") / request_id
        request_dir.mkdir(parents=True, exist_ok=False)
        worker_id = _codex_worker_id(role=role, site_id=site_id)
        worker_dir = run_dir / "agent_workers" / role / (site_id or "server")
        worker_dir.mkdir(parents=True, exist_ok=True)
        context = _package_codex_request_context(
            context=context,
            request_dir=request_dir,
            role=role,
            action=action,
        )
        agent_skills = _package_codex_request_local_skills(request_dir, phase=phase, action=action)

        request_payload = _request_payload(
            request_id=request_id,
            session_id=self.session_id,
            correlation_id=correlation_id,
            role=role,
            site_id=site_id,
            action=action,
            phase=phase,
            input_schema=input_schema,
            output_schema=output_schema,
            prompt=prompt,
            context=context,
            backend="codex",
            model=self.model,
            base_url=None,
        )
        if agent_skills:
            request_payload["agent_skills"] = agent_skills
        if _retry_of_request_id is not None:
            request_payload["retry"] = {
                "schema_version": "fedready.codex_sandbox_retry.v1",
                "attempt": _sandbox_retry_attempt + 1,
                "max_attempts": self.sandbox_retry_attempts + 1,
                "retry_of_request_id": _retry_of_request_id,
                "reason": "request-local filesystem sandbox was unavailable before command execution",
            }
        request_payload["worker"] = {
            "schema_version": "fedready.codex_worker_ref.v1",
            "worker_id": worker_id,
            "worker_dir": "[workflow-managed-worker-directory]",
            "state_file": "[workflow-managed-worker-state]",
            "lifecycle_owner": "fedready_nvflare_workflow",
            "lifecycle": "workflow_owned_active_inactive_worker",
            "request_scoped_process": True,
            "parent_conversation_context_inherited": False,
        }
        _write_json(request_dir / "request.json", request_payload)
        _write_json(request_dir / "context.json", context)
        (request_dir / "prompt.md").write_text(_render_prompt(request_payload, prompt), encoding="utf-8")

        codex_dir = request_dir / "codex_worker"
        codex_dir.mkdir(parents=True, exist_ok=True)
        response_path = request_dir / "response.json"
        task_path = codex_dir / "codex_task.md"
        stdout_path = codex_dir / "codex_events.jsonl"
        stderr_path = codex_dir / "codex_stderr.txt"
        last_message_path = codex_dir / "codex_last_message.txt"
        add_dirs = _codex_worker_add_dirs(context=execution_context, role=role, action=action)
        add_dir_labels = _codex_worker_add_dir_labels(context=context, role=role, action=action)
        task_text = _codex_worker_task_text(
            request_id=request_id,
            worker_id=worker_id,
            role=role,
            site_id=site_id,
            action=action,
            output_schema=output_schema,
            prompt=prompt,
            request_payload=request_payload,
            context=context,
            add_dir_labels=add_dir_labels,
            agent_skills=agent_skills,
        )
        task_path.write_text(task_text, encoding="utf-8")
        _write_json(
            request_dir / "status.json",
            {
                "schema_version": "fedready.file_agent_status.v1",
                "request_id": request_id,
                "status": "calling_codex_worker",
                "created_at": request_payload["created_at"],
                "request_dir": ".",
                "backend": "codex",
                "worker_id": worker_id,
                "worker_dir": "[workflow-managed-worker-directory]",
                "model": self.model,
                "codex_logs": {
                    "task": "codex_worker/codex_task.md",
                    "events": "codex_worker/codex_events.jsonl",
                    "stderr": "codex_worker/codex_stderr.txt",
                    "last_message": "codex_worker/codex_last_message.txt",
                },
                "agent_skills": agent_skills,
            },
        )
        _mark_codex_worker_state(
            worker_dir=worker_dir,
            worker_id=worker_id,
            role=role,
            site_id=site_id,
            lifecycle_status="active",
            request_id=request_id,
            request_dir=request_dir,
            request_status="running",
            model=self.model,
        )
        try:
            command = self._codex_command(
                request_dir=request_dir,
                last_message_path=last_message_path,
                add_dirs=add_dirs,
            )
            _write_json(
                codex_dir / "codex_command.json",
                {
                    "schema_version": "fedready.codex_worker_command.v1",
                    "request_id": request_id,
                    "worker_id": worker_id,
                    "command": _redact_codex_command(
                        command,
                        request_dir=request_dir,
                        last_message_path=last_message_path,
                        add_dirs=add_dirs,
                        add_dir_labels=add_dir_labels,
                    ),
                    "cwd": ".",
                    "additional_dirs": add_dir_labels,
                    "agent_skills": agent_skills,
                    "timeout_seconds": self.timeout_seconds,
                    "sandbox": self.sandbox,
                    "approval_policy": self.approval_policy,
                    "ignore_rules": self.ignore_rules,
                    "ephemeral": self.ephemeral,
                },
            )
            result = subprocess.run(
                command,
                input=task_text,
                text=True,
                capture_output=True,
                cwd=str(request_dir),
                timeout=self.timeout_seconds,
                check=False,
            )
            stdout_path.write_text(result.stdout or "", encoding="utf-8")
            stderr_path.write_text(result.stderr or "", encoding="utf-8")
            _write_json(
                codex_dir / "codex_result.json",
                {
                    "schema_version": "fedready.codex_worker_result.v1",
                    "request_id": request_id,
                    "worker_id": worker_id,
                    "returncode": result.returncode,
                    "stdout_digest": payload_digest(result.stdout or ""),
                    "stderr_digest": payload_digest(result.stderr or ""),
                    "last_message_exists": last_message_path.exists(),
                    "response_exists": response_path.exists(),
                },
            )
            if result.returncode != 0 and not response_path.exists():
                stderr_text = result.stderr or ""
                if _is_codex_sandbox_unavailable(stderr_text):
                    raise RuntimeError("Codex request-local filesystem sandbox failed before command execution")
                if _is_codex_startup_transient(stderr_text):
                    raise RuntimeError(
                        "Codex worker startup failed transiently before agent execution: "
                        f"{stderr_text.strip()[:300]}"
                    )
                raise RuntimeError(
                    f"Codex worker {worker_id} failed for {request_id} with exit code {result.returncode}. "
                    f"See {stderr_path}"
                )
            if not response_path.exists() and last_message_path.exists():
                response_from_message = _parse_agent_response(
                    last_message_path.read_text(encoding="utf-8"),
                    request_id=request_id,
                    repair_request_id=True,
                    repair_schema_version=True,
                    expected_output_schema=output_schema,
                )
                _write_json(response_path, response_from_message)
            response = _parse_agent_response(
                json.dumps(_read_json_object(response_path)),
                request_id=request_id,
                repair_request_id=False,
                repair_schema_version=False,
            )
            _apply_file_writes(response=response, context=execution_context, request_dir=request_dir)
            status = response.get("status")
            if status != "completed" and not _can_return_failed_output(
                action=action, response=response, output_schema=output_schema
            ):
                raise RuntimeError(f"Agent request {request_id} returned status {status}: {response.get('reason')}")
            output = response.get("output")
            if not isinstance(output, dict):
                raise ValueError(f"Agent response {request_id} missing object output")
            output = _resolve_local_output_paths(output, context=execution_context)
            response["output"] = output
            _write_json(response_path, response)
        except Exception as exc:
            retryable_sandbox_failure = _is_retryable_codex_worker_startup_failure(exc)
            should_retry_sandbox_failure = (
                retryable_sandbox_failure and _sandbox_retry_attempt < self.sandbox_retry_attempts
            )
            _write_json(
                request_dir / "status.json",
                {
                    "schema_version": "fedready.file_agent_status.v1",
                    "request_id": request_id,
                    "status": "failed",
                    "created_at": request_payload["created_at"],
                    "failed_at": timestamp_utc(),
                    "request_dir": str(request_dir),
                    "backend": "codex",
                    "worker_id": worker_id,
                    "worker_dir": str(worker_dir),
                    "error": _agent_error_summary(exc),
                },
            )
            _mark_codex_worker_state(
                worker_dir=worker_dir,
                worker_id=worker_id,
                role=role,
                site_id=site_id,
                lifecycle_status="inactive",
                request_id=request_id,
                request_dir=request_dir,
                request_status="failed",
                model=self.model,
                error=_agent_error_summary(exc),
            )
            if should_retry_sandbox_failure:
                return self.request(
                    role=role,
                    action=action,
                    phase=phase,
                    input_schema=input_schema,
                    output_schema=output_schema,
                    prompt=prompt,
                    context=retry_context,
                    site_id=site_id,
                    correlation_id=correlation_id,
                    _sandbox_retry_attempt=_sandbox_retry_attempt + 1,
                    _retry_of_request_id=request_id,
                )
            raise
        _write_json(
            request_dir / "status.json",
            {
                "schema_version": "fedready.file_agent_status.v1",
                "request_id": request_id,
                "status": "completed",
                "created_at": request_payload["created_at"],
                "completed_at": timestamp_utc(),
                "request_dir": str(request_dir),
                "response_digest": payload_digest(response),
                "backend": "codex",
                "worker_id": worker_id,
                "worker_dir": str(worker_dir),
            },
        )
        _mark_codex_worker_state(
            worker_dir=worker_dir,
            worker_id=worker_id,
            role=role,
            site_id=site_id,
            lifecycle_status="inactive",
            request_id=request_id,
            request_dir=request_dir,
            request_status="completed",
            model=self.model,
            response_digest=payload_digest(response),
        )
        return output

    def _codex_command(
        self,
        *,
        request_dir: Path,
        last_message_path: Path,
        add_dirs: list[Path],
    ) -> list[str]:
        command = [
            self.binary,
            "--ask-for-approval",
            self.approval_policy,
            "exec",
            "--skip-git-repo-check",
            "--cd",
            str(request_dir.expanduser().resolve()),
            "--sandbox",
            self.sandbox,
            "--output-last-message",
            str(last_message_path.expanduser().resolve()),
            "--json",
            "--color",
            "never",
        ]
        if self.ignore_rules:
            command.append("--ignore-rules")
        if self.ephemeral:
            command.append("--ephemeral")
        if self.model:
            command.extend(["--model", self.model])
        for add_dir in add_dirs:
            command.extend(["--add-dir", str(add_dir)])
        command.append("-")
        return command


def _codex_worker_id(*, role: str, site_id: str | None) -> str:
    return f"{_slug(role)}:{_slug(site_id or 'server')}"


def _package_codex_request_context(
    *,
    context: dict[str, Any],
    request_dir: Path,
    role: str,
    action: str,
) -> dict[str, Any]:
    packaged = json.loads(json.dumps(context))
    _ensure_request_local_task_example_snapshot(request_dir=request_dir, context=packaged)
    if context.get("code_workspace_allowed_for_agent_writes") is True:
        workspace_value = context.get("code_workspace")
        if isinstance(workspace_value, str) and workspace_value.strip():
            _ensure_request_local_directory_alias(
                request_dir / "code_workspace",
                Path(workspace_value).expanduser().resolve(),
            )
            packaged["code_workspace"] = "code_workspace"
            _replace_codex_output_paths_with_relative_names(packaged)
    if role == "client_agent" and action == "CLIENT.IMPLEMENT_LOCAL_ADAPTER":
        private_adapter_context = context.get("adapter_context")
        adapter_context = packaged.get("adapter_context")
        if isinstance(private_adapter_context, dict) and isinstance(adapter_context, dict):
            local_data_value = private_adapter_context.get("local_data_path")
            if isinstance(local_data_value, str) and local_data_value.strip():
                _ensure_request_local_directory_alias(
                    request_dir / "client_data",
                    Path(local_data_value).expanduser().resolve(),
                )
                adapter_context["local_data_path"] = "client_data"
            if adapter_context.get("adapter_workspace"):
                adapter_context["adapter_workspace"] = "code_workspace"
    if role != "client_agent" or action != "CLIENT.VISUAL_QC_EXTRACTION":
        return packaged
    qc_context = packaged.get("qc_context")
    if not isinstance(qc_context, dict):
        return packaged
    artifacts = qc_context.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        return packaged

    artifact_root = request_dir / "qc_artifacts"
    copied_files: list[str] = []
    warnings: list[str] = []
    for sample_index, artifact in enumerate(artifacts, start=1):
        if not isinstance(artifact, dict):
            continue
        sample_dir = artifact_root / f"sample_{sample_index:02d}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        for key, filename in (
            ("image_path", "image.png"),
            ("label_path", "label.png"),
            ("candidate_sheet_path", "candidate_transforms.png"),
        ):
            rel_path = _copy_codex_qc_artifact(artifact.get(key), sample_dir / filename, request_dir=request_dir)
            if rel_path is None:
                if artifact.get(key):
                    warnings.append(f"{key} was not readable for sample_{sample_index:02d}")
                continue
            artifact[key] = rel_path
            copied_files.append(rel_path)
        overlay_paths = artifact.get("candidate_overlay_paths")
        if isinstance(overlay_paths, dict):
            packaged_overlays = {}
            for transform, source_path in sorted(overlay_paths.items()):
                safe_transform = _slug(str(transform)) or "candidate"
                rel_path = _copy_codex_qc_artifact(
                    source_path,
                    sample_dir / f"overlay_{safe_transform}.png",
                    request_dir=request_dir,
                )
                if rel_path is None:
                    warnings.append(
                        f"candidate_overlay_paths.{transform} was not readable for sample_{sample_index:02d}"
                    )
                    continue
                packaged_overlays[str(transform)] = rel_path
                copied_files.append(rel_path)
            artifact["candidate_overlay_paths"] = packaged_overlays
        artifact["packaged_artifact_root"] = str(sample_dir.relative_to(request_dir))

    packaging = {
        "schema_version": "fedready.codex_request_artifact_package.v1",
        "strategy": "copied_request_local_qc_artifacts",
        "artifact_root": str(artifact_root.relative_to(request_dir)),
        "copied_file_count": len(copied_files),
        "original_local_paths_removed": True,
    }
    if warnings:
        packaging["warnings"] = warnings
    qc_context["artifact_paths_are_request_local"] = True
    qc_context["packaged_for_codex_worker"] = packaging
    return packaged


def _package_request_task_example_context(*, context: dict[str, Any], request_dir: Path) -> dict[str, Any]:
    packaged = json.loads(json.dumps(context))
    _ensure_request_local_task_example_snapshot(request_dir=request_dir, context=packaged)
    return packaged


def _ensure_request_local_directory_alias(alias: Path, target: Path) -> None:
    if alias.exists() or alias.is_symlink():
        return
    alias.symlink_to(target, target_is_directory=True)


def _ensure_request_local_task_example_snapshot(*, request_dir: Path, context: dict[str, Any]) -> None:
    task_examples = _context_task_examples(context)
    if task_examples is None or task_examples.get("safe_to_share") is not True:
        return
    if not TASK_EXAMPLE_DIR.is_dir():
        return
    destination = request_dir / "task_example"
    if destination.is_symlink():
        destination.unlink()
    if not destination.exists():
        shutil.copytree(TASK_EXAMPLE_DIR.resolve(), destination)
    task_examples["root"] = "task_example"
    task_examples["path_scope"] = "request_local_task_example_snapshot"


def _context_task_examples(context: dict[str, Any]) -> dict[str, Any] | None:
    adapter_context = context.get("adapter_context")
    if isinstance(adapter_context, dict) and isinstance(adapter_context.get("task_examples"), dict):
        return adapter_context["task_examples"]
    task_examples = context.get("task_examples")
    if isinstance(task_examples, dict):
        return task_examples
    return None


def _replace_codex_output_paths_with_relative_names(value: Any) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key in {
                "manifest_path",
                "adapter_manifest_path",
                "script_path",
            } and isinstance(child, str):
                value[key] = Path(child).name
            else:
                _replace_codex_output_paths_with_relative_names(child)
    elif isinstance(value, list):
        for child in value:
            _replace_codex_output_paths_with_relative_names(child)


def _copy_codex_qc_artifact(source_value: Any, destination: Path, *, request_dir: Path) -> str | None:
    if not isinstance(source_value, str) or not source_value.strip():
        return None
    source_path = Path(source_value).expanduser()
    if not source_path.is_absolute():
        source_path = source_path.resolve()
    if not source_path.is_file():
        return None
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, destination)
    return str(destination.relative_to(request_dir))


def _codex_worker_add_dirs(*, context: dict[str, Any], role: str, action: str) -> list[Path]:
    roots: list[Path] = []
    if context.get("code_workspace_allowed_for_agent_writes") is True:
        workspace = context.get("code_workspace")
        if isinstance(workspace, str) and workspace.strip():
            code_root = Path(workspace).expanduser().resolve()
            code_root.mkdir(parents=True, exist_ok=True)
            roots.append(code_root)
    adapter_context = context.get("adapter_context") if isinstance(context.get("adapter_context"), dict) else {}
    if role == "client_agent" and action == "CLIENT.IMPLEMENT_LOCAL_ADAPTER":
        local_data_path = adapter_context.get("local_data_path") if isinstance(adapter_context, dict) else None
        if isinstance(local_data_path, str) and local_data_path.strip():
            roots.append(Path(local_data_path).expanduser().resolve())
    unique: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        key = str(root)
        if key in seen:
            continue
        seen.add(key)
        unique.append(root)
    return unique


def _codex_worker_add_dir_labels(*, context: dict[str, Any], role: str, action: str) -> list[str]:
    labels: list[str] = []
    if context.get("code_workspace_allowed_for_agent_writes") is True:
        labels.append("code_workspace (request-local alias; writable)")
    if role == "client_agent" and action == "CLIENT.IMPLEMENT_LOCAL_ADAPTER":
        labels.append("client_data (request-local alias; client-local data only)")
    task_examples = _context_task_examples(context)
    if task_examples is not None and task_examples.get("safe_to_share") is True and TASK_EXAMPLE_DIR.is_dir():
        labels.append("task_example (request-local snapshot of canonical example files)")
    return labels


def _codex_worker_task_text(
    *,
    request_id: str,
    worker_id: str,
    role: str,
    site_id: str | None,
    action: str,
    output_schema: str,
    prompt: str,
    request_payload: dict[str, Any],
    context: dict[str, Any],
    add_dir_labels: list[str],
    agent_skills: list[dict[str, str]],
) -> str:
    add_dir_text = "\n".join(f"- `{label}`" for label in add_dir_labels) or "- none"
    skill_text = (
        "\n".join(f"- `{skill.get('name')}`: `{skill.get('path')}` ({skill.get('status')})" for skill in agent_skills)
        or "- none"
    )
    visual_qc_text = (
        render_server_prompt("codex_worker_visual_qc_note") if action == "CLIENT.VISUAL_QC_EXTRACTION" else ""
    )
    return render_server_prompt(
        "codex_worker_task_text",
        worker_id=worker_id,
        role=role,
        site_id=site_id,
        request_id=request_id,
        action=action,
        output_schema=output_schema,
        prompt=prompt,
        request_json=json.dumps(request_payload, indent=2, sort_keys=True),
        context_json=json.dumps(context, indent=2, sort_keys=True),
        add_dir_text=add_dir_text,
        skill_text=skill_text,
        visual_qc_text=visual_qc_text,
        file_agent_response_schema=FILE_AGENT_RESPONSE_SCHEMA,
    )


def _redact_codex_command(
    command: list[str],
    *,
    request_dir: Path,
    last_message_path: Path,
    add_dirs: list[Path],
    add_dir_labels: list[str],
) -> list[str]:
    replacements = {
        str(request_dir.expanduser().resolve()): ".",
        str(last_message_path.expanduser().resolve()): "codex_worker/codex_last_message.txt",
    }
    for index, path in enumerate(add_dirs):
        label = add_dir_labels[index] if index < len(add_dir_labels) else f"additional_dir_{index + 1}"
        replacements[str(path)] = label.split(" ", 1)[0]
    return [replacements.get(part, part) for part in command]


def _mark_codex_worker_state(
    *,
    worker_dir: Path,
    worker_id: str,
    role: str,
    site_id: str | None,
    lifecycle_status: str,
    request_id: str,
    request_dir: Path,
    request_status: str,
    model: str | None,
    error: dict[str, Any] | None = None,
    response_digest: str | None = None,
) -> None:
    state_path = worker_dir / "worker_state.json"
    try:
        state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    except json.JSONDecodeError:
        state = {}
    if not isinstance(state, dict):
        state = {}
    created_at = state.get("created_at") or timestamp_utc()
    request_count = int(state.get("request_count") or 0)
    if lifecycle_status == "active":
        request_count += 1
    event = {
        "schema_version": "fedready.codex_worker_event.v1",
        "timestamp_utc": timestamp_utc(),
        "worker_id": worker_id,
        "role": role,
        "site_id": site_id,
        "lifecycle_status": lifecycle_status,
        "request_id": request_id,
        "request_status": request_status,
        "request_dir": str(request_dir),
    }
    if error is not None:
        event["error"] = error
    if response_digest is not None:
        event["response_digest"] = response_digest
    _append_jsonl(worker_dir / "events.jsonl", event)
    state = {
        "schema_version": "fedready.codex_worker_state.v1",
        "worker_id": worker_id,
        "role": role,
        "site_id": site_id,
        "backend": "codex",
        "model": model,
        "created_at": created_at,
        "updated_at": event["timestamp_utc"],
        "lifecycle_owner": "fedready_nvflare_workflow",
        "lifecycle_status": lifecycle_status,
        "active": lifecycle_status == "active",
        "current_request_id": request_id if lifecycle_status == "active" else None,
        "last_request_id": request_id,
        "last_request_status": request_status,
        "last_request_dir": str(request_dir),
        "request_count": request_count,
        "parent_conversation_context_inherited": False,
        "filesystem_scope": "request_dir_plus_explicit_request_roots",
    }
    if error is not None:
        state["last_error"] = error
    if response_digest is not None:
        state["last_response_digest"] = response_digest
    _write_json(state_path, state)


def list_agent_requests(run_dir: str | Path, *, status: str | None = None) -> list[dict[str, Any]]:
    """Return agent mailbox requests under a run directory."""

    mailbox = Path(run_dir) / "agent_mailbox"
    if not mailbox.exists():
        return []
    requests = []
    for request_path in sorted(mailbox.glob("*/*/*/request.json")):
        try:
            request = json.loads(request_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if not isinstance(request, dict):
            continue
        request_dir = request_path.parent
        current_status = "completed" if (request_dir / "response.json").exists() else "pending"
        if status is not None and current_status != status:
            continue
        requests.append(
            {
                "request_id": request.get("request_id"),
                "role": request.get("role"),
                "site_id": request.get("site_id"),
                "action": request.get("action"),
                "phase": request.get("phase"),
                "status": current_status,
                "request_dir": str(request_dir),
                "prompt": str(request_dir / "prompt.md"),
                "context": str(request_dir / "context.json"),
                "response": str(request_dir / "response.json"),
            }
        )
    return requests


def build_agent_backend(
    *,
    kind: str,
    run_dir: str | Path,
    session_id: str,
    timeout_seconds: float = 3600.0,
    poll_interval_seconds: float = 2.0,
) -> CodexWorkerBackend:
    """Build the single supported live backend without alternate fallbacks."""

    normalized = kind.strip().lower()
    if normalized in {
        "codex",
        "codex-worker",
        "codex_worker",
        "codex-nvflare",
        "codex_nvflare",
    }:
        return CodexWorkerBackend.from_env(
            run_dir=run_dir,
            session_id=session_id,
            timeout_seconds=timeout_seconds,
        )
    raise ValueError("Unsupported agent_backend: this research workflow supports only 'codex'.")


def _request_id(*, role: str, action: str, site_id: str | None) -> str:
    parts = [role, site_id or "server", action, str(uuid4())]
    return "_".join(_slug(part) for part in parts if part)


def _slug(value: str) -> str:
    chars = [char.lower() if char.isalnum() else "_" for char in value]
    return "_".join(part for part in "".join(chars).split("_") if part) or "agent"


def _request_payload(
    *,
    request_id: str,
    session_id: str,
    correlation_id: str | None,
    role: str,
    site_id: str | None,
    action: str,
    phase: str,
    input_schema: str,
    output_schema: str,
    prompt: str,
    context: dict[str, Any],
    backend: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
) -> dict[str, Any]:
    payload = {
        "schema_version": FILE_AGENT_REQUEST_SCHEMA,
        "request_id": request_id,
        "session_id": session_id,
        "correlation_id": correlation_id,
        "created_at": timestamp_utc(),
        "role": role,
        "site_id": site_id,
        "action": action,
        "phase": phase,
        "input_schema": input_schema,
        "output_schema": output_schema,
        "prompt_file": "prompt.md",
        "context_file": "context.json",
        "response_file": "response.json",
        "prompt_digest": payload_digest(prompt),
        "context_digest": payload_digest(context),
        "response_contract": {
            "write_json_file": "response.json",
            "schema_version": FILE_AGENT_RESPONSE_SCHEMA,
            "required_top_level_keys": [
                "schema_version",
                "request_id",
                "status",
                "output",
            ],
            "status_values": ["completed", "denied", "failed"],
            "output_must_match": output_schema,
        },
        "privacy": {
            "request_is_local_to_simulation": True,
            "do_not_add_local_paths_or_raw_samples_to_response": True,
        },
    }
    if backend is not None:
        payload["backend"] = {"kind": backend, "model": model, "base_url": base_url}
    return payload


def _render_prompt(request: dict[str, Any], prompt: str) -> str:
    return render_server_prompt(
        "file_agent_request_prompt",
        request_id=request["request_id"],
        role=request["role"],
        site_id=request.get("site_id"),
        action=request["action"],
        phase=request["phase"],
        input_schema=request["input_schema"],
        output_schema=request["output_schema"],
        file_agent_response_schema=FILE_AGENT_RESPONSE_SCHEMA,
        prompt=prompt,
    )


def _redact_api_payload(payload: dict[str, Any]) -> dict[str, Any]:
    messages = []
    for message in payload.get("messages", []):
        if not isinstance(message, dict):
            continue
        messages.append(
            {
                "role": message.get("role"),
                "content": _redact_message_content(message.get("content")),
            }
        )
    return {**payload, "messages": messages}


def _redact_message_content(content: Any) -> Any:
    if isinstance(content, str):
        return {"type": "text", "content_digest": payload_digest(content)}
    if isinstance(content, list):
        redacted = []
        for block in content:
            if not isinstance(block, dict):
                redacted.append({"type": "unknown", "content_digest": payload_digest(block)})
                continue
            if block.get("type") == "image_url":
                image_url = block.get("image_url") if isinstance(block.get("image_url"), dict) else {}
                url = image_url.get("url") if isinstance(image_url, dict) else None
                redacted.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "[redacted-image-data-url]",
                            "url_digest": payload_digest(url),
                            "url_length": len(url) if isinstance(url, str) else None,
                        },
                    }
                )
            elif block.get("type") == "text":
                redacted.append(
                    {
                        "type": "text",
                        "content_digest": payload_digest(block.get("text")),
                    }
                )
            else:
                redacted.append({"type": block.get("type"), "content_digest": payload_digest(block)})
        return redacted
    return {"type": type(content).__name__, "content_digest": payload_digest(content)}


def _safe_remote_text_api_context(context: dict[str, Any]) -> dict[str, Any]:
    safe_context = _redact_remote_text_value(json.loads(json.dumps(context, default=str)))
    output_template = context.get("output_template")
    if isinstance(safe_context, dict) and isinstance(output_template, dict):
        safe_context["output_template"] = _safe_remote_output_template(
            json.loads(json.dumps(output_template, default=str))
        )
    if isinstance(safe_context, dict) and context.get("code_workspace_allowed_for_agent_writes") is True:
        safe_context["code_workspace"] = "[redacted-local-code-workspace]"
        safe_context["file_write_instruction"] = (
            "Return file writes as relative paths under code_workspace; do not use absolute local paths."
        )
    return safe_context if isinstance(safe_context, dict) else {}


def _safe_remote_output_template(value: Any, *, key: str | None = None) -> Any:
    if key is not None and _is_local_path_key(key):
        return None if value is None else "[redacted-local-artifact]"
    if key in {"reason", "redaction_reason"}:
        return None if value is None else "[model-supplied-safe-reason]"
    if isinstance(value, dict):
        return {
            str(child_key): _safe_remote_output_template(child, key=str(child_key))
            for child_key, child in value.items()
        }
    if isinstance(value, list):
        return [_safe_remote_output_template(item) for item in value]
    if isinstance(value, str) and value.startswith("/"):
        return "[redacted-local-path]"
    return value


_SAFE_TO_SHARE_FALSE_SUMMARY_KEYS = {
    "schema_version",
    "available",
    "client_id",
    "site_id",
    "source_label_type",
    "sample_count",
    "safe_to_share",
}


def _redact_remote_text_value(value: Any) -> Any:
    if isinstance(value, dict):
        if value.get("safe_to_share") is False:
            summary: dict[str, Any] = {
                "redacted": True,
                "safe_to_share": False,
                "redaction_reason": "client-local context removed from model-readable remote-text request",
            }
            for key in _SAFE_TO_SHARE_FALSE_SUMMARY_KEYS:
                if key in value and not _is_local_path_key(str(key)):
                    summary[key] = _redact_remote_text_value(value[key])
            return summary
        redacted: dict[str, Any] = {}
        for key, child in value.items():
            if _is_local_path_key(str(key)):
                redacted[str(key)] = "[redacted-local-artifact]"
            else:
                redacted[str(key)] = _redact_remote_text_value(child)
        return redacted
    if isinstance(value, list):
        return [_redact_remote_text_value(item) for item in value]
    if isinstance(value, str) and value.startswith("/"):
        return "[redacted-local-path]"
    return value


def _is_local_path_key(key: str) -> bool:
    lowered = key.lower()
    return (
        lowered.endswith("_path")
        or lowered.endswith("_paths")
        or lowered.endswith("_workspace")
        or lowered
        in {
            "image",
            "label",
            "overlay",
            "candidate_sheet",
            "candidate_overlays",
            "artifact_root",
            "code_workspace",
            "adapter_workspace",
            "local_data_path",
        }
    )


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_text(name: str) -> str | None:
    value = os.environ.get(name)
    if value is None:
        return None
    cleaned = value.strip()
    if not cleaned or cleaned.lower() in {"0", "false", "no", "none", "off"}:
        return None
    return cleaned


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuntimeError(f"Expected JSON file does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Expected valid JSON object at {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return value


def _parse_agent_response(
    content: str,
    *,
    request_id: str,
    repair_request_id: bool = False,
    repair_schema_version: bool = False,
    expected_output_schema: str | None = None,
) -> dict[str, Any]:
    response = _loads_json_object(content)
    if response.get("schema_version") != FILE_AGENT_RESPONSE_SCHEMA:
        if expected_output_schema and _schema_versions_equivalent(
            response.get("schema_version"),
            expected_output_schema,
        ):
            response = {
                "schema_version": FILE_AGENT_RESPONSE_SCHEMA,
                "request_id": response.get("request_id", request_id),
                "status": "completed",
                "output": response,
            }
        elif (
            repair_schema_version
            and response.get("status") in {"completed", "denied", "failed"}
            and isinstance(response.get("output"), dict)
        ):
            response["schema_version"] = FILE_AGENT_RESPONSE_SCHEMA
        else:
            raise ValueError(f"Agent response schema mismatch for {request_id}")
    if response.get("request_id") != request_id:
        if repair_request_id:
            response["request_id"] = request_id
        else:
            raise ValueError(f"Agent response request_id mismatch for {request_id}")
    if response.get("status") not in {"completed", "denied", "failed"}:
        raise ValueError(f"Agent response status mismatch for {request_id}")
    if not isinstance(response.get("output"), dict):
        raise ValueError(f"Agent response {request_id} missing object output")
    return response


def _can_return_failed_output(*, action: str, response: dict[str, Any], output_schema: str) -> bool:
    if response.get("status") != "failed":
        return False
    if _can_return_failed_visual_qc_output(action=action, response=response, output_schema=output_schema):
        return True
    if _can_return_failed_profile_output(action=action, response=response, output_schema=output_schema):
        return True
    return False


def _can_return_failed_visual_qc_output(*, action: str, response: dict[str, Any], output_schema: str) -> bool:
    if action != "CLIENT.VISUAL_QC_EXTRACTION":
        return False
    output = response.get("output")
    if not isinstance(output, dict):
        return False
    if not _schema_versions_equivalent(output.get("schema_version"), output_schema):
        return False
    return (
        output.get("status") in {"failed", "needs_more_samples", "not_performed"} and output.get("passed") is not True
    )


def _can_return_failed_profile_output(*, action: str, response: dict[str, Any], output_schema: str) -> bool:
    if action != "CLIENT.REPORT_DATA_PROFILE":
        return False
    output = response.get("output")
    if not isinstance(output, dict):
        return False
    if not _schema_versions_equivalent(output.get("schema_version"), output_schema):
        return False
    return output.get("data") == "not applicable"


def _schema_versions_equivalent(actual: Any, expected: str) -> bool:
    if actual == expected:
        return True
    aliases = {
        "FedReadyClientInquiry@v1": {"fedready.client_inquiry.v1"},
        "FedReadyClientResponse@v1": {"fedready.client_response.v1"},
        "FedReadyLocalAdapterSpec@v1": {"fedready.local_adapter_spec.v1"},
        "FedReadyExtractionQCDecision@v1": {
            "fedready.extraction_qc_decision.v1",
            "fedready.extraction_visual_qc_decision.v1",
        },
        "FedReadyTrainingCodeSpec@v1": {"fedready.training_code_spec.v1"},
        "FedReadyGuardrailDecision@v1": {"fedready.guardrail_decision.v1"},
    }
    return isinstance(actual, str) and actual in aliases.get(expected, set())


def _loads_json_object(content: str) -> dict[str, Any]:
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise
        value = json.loads(text[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("agent response content must be a JSON object")
    return value


def _resolve_local_output_paths(output: dict[str, Any], *, context: dict[str, Any]) -> dict[str, Any]:
    code_workspace = context.get("code_workspace")
    if not isinstance(code_workspace, str) or not code_workspace:
        return output
    root = Path(code_workspace).resolve()
    resolved = dict(output)
    for key in ("manifest_path", "adapter_manifest_path", "script_path"):
        value = resolved.get(key)
        if not isinstance(value, str) or not value.strip():
            continue
        candidate = Path(value).expanduser()
        if not candidate.is_absolute():
            if candidate.parts and candidate.parts[0] in {
                "code_workspace",
                "adapter_workspace",
            }:
                candidate = Path(*candidate.parts[1:])
            candidate = root / candidate
        candidate = candidate.resolve()
        if candidate == root or root in candidate.parents:
            resolved[key] = str(candidate)
    return resolved


def _apply_file_writes(*, response: dict[str, Any], context: dict[str, Any], request_dir: Path) -> None:
    files = response.get("files")
    if files is None:
        return
    if context.get("code_workspace_allowed_for_agent_writes") is not True:
        raise ValueError("agent response included file writes without code workspace permission")
    code_workspace = context.get("code_workspace")
    if not isinstance(code_workspace, str) or not code_workspace:
        raise ValueError("agent response included file writes but context.code_workspace is missing")
    root = Path(code_workspace).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if not isinstance(files, list):
        raise ValueError("agent response files must be a list")
    written: list[str] = []
    for entry in files:
        if not isinstance(entry, dict):
            raise ValueError("agent response file entry must be an object")
        rel_path = entry.get("path")
        content = entry.get("content")
        if not isinstance(rel_path, str) or not rel_path:
            raise ValueError("agent response file entry missing path")
        if not isinstance(content, str):
            raise ValueError(f"agent response file entry {rel_path!r} missing string content")
        target = Path(rel_path)
        if not target.is_absolute():
            target = root / target
        target = target.resolve()
        if target != root and root not in target.parents:
            raise ValueError(f"agent attempted to write outside code workspace: {rel_path}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        written.append(str(target))
    _write_json(request_dir / "agent_file_writes.json", {"files": written})


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_write_json(path, payload)


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _json_safe(value: Any) -> Any:
    return json.loads(json.dumps(value, default=_json_default))


def _json_default(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if hasattr(value, "dict"):
        return value.dict()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _agent_error_summary(exc: Exception) -> dict[str, Any]:
    message = str(exc)
    if len(message) > 2000:
        message = message[:2000] + "...<truncated>"
    return {"type": type(exc).__name__, "message": message}
