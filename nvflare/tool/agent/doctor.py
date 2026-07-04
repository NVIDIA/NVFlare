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

"""Read-only agent environment readiness checks."""

import importlib.util
import json
import math
import os
import stat
import tempfile
from datetime import datetime, timezone
from itertools import islice
from pathlib import Path

import nvflare
from nvflare.tool.agent.command_registry import agent_command_registry
from nvflare.tool.agent.skill_manager import find_skill_source

OPTIONAL_DEPENDENCIES = (
    "torch",
    "lightning",
    "pytorch_lightning",
    "tensorflow",
    "sklearn",
    "xgboost",
    "jax",
    "flwr",
)
MAX_STARTUP_CONFIG_BYTES = 1024 * 1024
MAX_STARTUP_KIT_ENTRIES = 64
MAX_DOCTOR_TEXT_LENGTH = 4096
MAX_ONLINE_STATUS_ITEMS = 1000
MAX_ONLINE_STATUS_FIELDS = 64
MAX_ONLINE_STATUS_NESTED_ITEMS = 128
MAX_ONLINE_STATUS_DEPTH = 4
MAX_ONLINE_STATUS_NODES = 10_000
MAX_ONLINE_STATUS_TEXT_LENGTH = 1024
MAX_ONLINE_STATUS_KEY_LENGTH = 128
MAX_ONLINE_TRUNCATION_DETAILS = 32


def doctor_environment(*, online: bool = False, args=None) -> dict:
    """Return a read-only readiness snapshot, optionally with a bounded online check.

    Conversion readiness is always reported through the command, dependency, and
    skill sections.  The deployment sections are retained for schema-v1 callers.
    """
    cli_config, cli_config_error = _load_doctor_cli_config()
    startup_kits = _startup_kit_summary(cli_config, cli_config_error)
    skills = _skills_summary()
    poc = _poc_summary(cli_config)
    data = {
        "schema_version": "1",
        "timestamp": _now_utc(),
        "nvflare": {"import_ok": True, "version": nvflare.__version__},
        "commands": _command_registry(),
        "startup_kits": startup_kits,
        "optional_dependencies": _optional_dependency_summary(),
        "skills": skills,
        "poc": poc,
        "online": {"enabled": False, "status": "not_requested"},
        "findings": [],
    }
    deployment_findings = [*startup_kits.get("findings", []), *poc.get("findings", [])]
    conversion_findings = list(skills.get("findings", []))
    data["findings"].extend(deployment_findings)
    data["findings"].extend(conversion_findings)

    if online:
        data["online"] = _online_summary(args, cli_config, cli_config_error)
        data["findings"].extend(data["online"].get("findings", []))

    # Preserve the schema-v1 deployment sections without making an offline,
    # conversion-scoped readiness check fail merely because POC or production
    # has not been configured.  An explicitly requested online check is status
    # relevant; offline deployment findings remain available through their
    # sections and deployment_status.
    online_findings = data["online"].get("findings", []) if online else []
    data["conversion_status"] = "attention" if _has_attention_findings(conversion_findings) else "ok"
    data["deployment_status"] = (
        "attention" if _has_attention_findings([*deployment_findings, *online_findings]) else "ok"
    )
    data["status_scope"] = "conversion_and_online" if online else "conversion"
    data["status"] = (
        "attention"
        if data["conversion_status"] == "attention" or (online and _has_attention_findings(online_findings))
        else "ok"
    )
    return data


def _command_registry() -> dict:
    return agent_command_registry()


def _load_doctor_cli_config():
    """Read the local registry once without HOCON includes or substitutions."""

    from nvflare.tool.kit.kit_config import StartupKitConfigError, get_cli_config_path

    config_path = get_cli_config_path()
    try:
        mode = config_path.lstat().st_mode
    except FileNotFoundError:
        return None, None
    except OSError:
        return None, StartupKitConfigError(f"cannot inspect {config_path}")

    try:
        if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
            raise ValueError("CLI config must be a regular file")
        _reject_symlink_components(config_path)
        raw = _read_bounded_regular_file(config_path, MAX_STARTUP_CONFIG_BYTES)
        text = raw.decode("utf-8")
        if _hocon_contains_include(text):
            raise ValueError("HOCON include directives are not allowed in agent doctor")
        from pyhocon import ConfigFactory as CF

        return CF.parse_string(text, basedir=None, resolve=False), None
    except Exception as e:
        return None, StartupKitConfigError(
            f"cannot safely parse {config_path}",
            hint="Remove include directives and keep the local CLI config below 1 MiB.",
        )


def _hocon_contains_include(text: str) -> bool:
    """Return whether an include keyword occurs outside strings/comments."""

    index = 0
    length = len(text)
    while index < length:
        if text.startswith("//", index) or text[index] == "#":
            newline = text.find("\n", index + 1)
            index = length if newline < 0 else newline + 1
            continue
        if text.startswith("/*", index):
            end = text.find("*/", index + 2)
            index = length if end < 0 else end + 2
            continue
        if text.startswith('"""', index):
            end = text.find('"""', index + 3)
            index = length if end < 0 else end + 3
            continue
        if text[index] == '"':
            index += 1
            while index < length:
                if text[index] == "\\":
                    index += 2
                elif text[index] == '"':
                    index += 1
                    break
                else:
                    index += 1
            continue
        if text[index].isalpha() or text[index] == "_":
            end = index + 1
            while end < length and (text[end].isalnum() or text[end] in "_-"):
                end += 1
            if text[index:end].lower() == "include":
                return True
            index = end
            continue
        index += 1
    return False


def _startup_kit_summary(config, config_error) -> dict:
    from nvflare.tool.kit.kit_config import get_active_startup_kit_id, get_cli_config_path, get_startup_kit_entries

    config_file = str(get_cli_config_path())
    if config_error:
        return {
            "config_file": config_file,
            "active_id": None,
            "entries": [],
            "findings": [
                _finding(
                    "STARTUP_KIT_CONFIG_INVALID",
                    "error",
                    str(config_error),
                    getattr(config_error, "hint", None),
                )
            ],
        }
    try:
        active = get_active_startup_kit_id(config) if config is not None else None
        entries = get_startup_kit_entries(config) if config is not None else {}
    except Exception as e:
        return {
            "config_file": config_file,
            "active_id": None,
            "entries": [],
            "findings": [_finding("STARTUP_KIT_CONFIG_INVALID", "error", str(e))],
        }

    findings = []
    if not active:
        findings.append(
            _finding(
                "STARTUP_KIT_NOT_CONFIGURED",
                "warning",
                "No active startup kit is configured.",
                "Run nvflare config list and nvflare config use <id>, or pass --startup-kit for online checks.",
            )
        )

    sorted_entries = sorted(entries.items())
    if len(sorted_entries) > MAX_STARTUP_KIT_ENTRIES:
        findings.append(
            _finding(
                "STARTUP_KIT_REGISTRY_TRUNCATED",
                "warning",
                f"Startup-kit registry exceeds {MAX_STARTUP_KIT_ENTRIES} entries; extra entries were not inspected.",
            )
        )
        sorted_entries = sorted_entries[:MAX_STARTUP_KIT_ENTRIES]

    entry_rows = []
    for kit_id, path in sorted_entries:
        status, normalized_path, metadata = _inspect_doctor_startup_kit(path)
        entry_rows.append(
            {
                "id": _bounded_text(kit_id),
                "path": _bounded_text(path),
                "normalized_path": normalized_path,
                "active": kit_id == active,
                "status": status,
                "metadata": metadata,
            }
        )
        for finding in metadata.get("findings", []):
            finding = dict(finding)
            finding["startup_kit_id"] = kit_id
            findings.append(finding)

    return {
        "config_file": config_file,
        "active_id": _bounded_text(active) if active else None,
        "entries": entry_rows,
        "findings": findings,
    }


def _inspect_doctor_startup_kit(path: str) -> tuple[str, str | None, dict]:
    metadata = {
        "identity": None,
        "cert_role": None,
        "role": None,
        "org": None,
        "project": None,
        "kind": "admin",
        "certificate": None,
        "findings": [],
    }
    try:
        if not isinstance(path, str) or not path or len(path) > MAX_DOCTOR_TEXT_LENGTH:
            raise ValueError("startup-kit path must be a bounded non-empty string")
        try:
            kit_path = _normalize_startup_kit_root(path)
        except FileNotFoundError:
            metadata["findings"].append(
                _finding(
                    "STARTUP_KIT_PATH_MISSING",
                    "warning",
                    f"Registered startup kit path does not exist: {_bounded_text(path)}",
                    "Remove the stale registration or restore the startup kit.",
                )
            )
            return "missing", None, metadata
        config = _load_status_admin_config(kit_path)
        admin = config["admin"]
        username = admin.get("username")
        if isinstance(username, str) and username:
            metadata["identity"] = _bounded_text(username)
        cert_value = admin.get("client_cert") or "client.crt"
        try:
            cert_path = _startup_member_path(kit_path / "startup", cert_value)
            cert_bytes = _read_bounded_regular_file(cert_path, MAX_STARTUP_CONFIG_BYTES)
            _add_certificate_metadata(metadata, cert_bytes, cert_path)
        except Exception as e:
            metadata["certificate"] = {"path": _bounded_text(cert_value), "status": "unreadable"}
            metadata["findings"].append(
                _finding(
                    "STARTUP_KIT_CERT_UNREADABLE",
                    "warning",
                    _bounded_text(f"Startup kit certificate could not be read safely: {type(e).__name__}: {e}"),
                    "Replace this startup kit if the certificate is missing or corrupted.",
                )
            )
        status = "invalid" if any(finding.get("severity") == "error" for finding in metadata["findings"]) else "ok"
        return status, str(kit_path), metadata
    except Exception as e:
        metadata["findings"].append(
            _finding(
                "STARTUP_KIT_INVALID",
                "warning",
                _bounded_text(f"Registered startup kit could not be inspected safely: {type(e).__name__}: {e}"),
                "Replace the startup kit or remove the stale registry entry.",
            )
        )
        return "invalid", None, metadata


def _add_certificate_metadata(metadata: dict, cert_bytes: bytes, cert_path: Path) -> None:
    from cryptography import x509
    from cryptography.hazmat.backends import default_backend
    from cryptography.x509.oid import NameOID

    certificate = x509.load_pem_x509_certificate(cert_bytes, default_backend())

    def subject_value(oid):
        values = certificate.subject.get_attributes_for_oid(oid)
        return _bounded_text(values[0].value) if values else None

    def issuer_value(oid):
        values = certificate.issuer.get_attributes_for_oid(oid)
        return _bounded_text(values[0].value) if values else None

    cert_role = subject_value(NameOID.UNSTRUCTURED_NAME)
    metadata["cert_role"] = cert_role
    metadata["role"] = cert_role
    metadata["org"] = subject_value(NameOID.ORGANIZATION_NAME)
    metadata["project"] = issuer_value(NameOID.COMMON_NAME)
    if not metadata.get("identity"):
        metadata["identity"] = subject_value(NameOID.COMMON_NAME)

    expires_at = getattr(certificate, "not_valid_after_utc", None)
    if expires_at is None:
        expires_at = certificate.not_valid_after.replace(tzinfo=timezone.utc)
    elif expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)
    expires_at = expires_at.astimezone(timezone.utc)
    now = datetime.now(timezone.utc)
    seconds_remaining = (expires_at - now).total_seconds()
    days_remaining = int(seconds_remaining // 86400)
    if seconds_remaining < 0:
        cert_status = "expired"
        metadata["findings"].append(
            _finding(
                "STARTUP_KIT_CERT_EXPIRED",
                "error",
                f"Startup kit certificate expired at {_format_utc(expires_at)}.",
                "Request or select a renewed startup kit.",
            )
        )
    elif days_remaining <= 30:
        cert_status = "expiring_soon"
        metadata["findings"].append(
            _finding(
                "STARTUP_KIT_CERT_EXPIRING_SOON",
                "warning",
                f"Startup kit certificate expires at {_format_utc(expires_at)}.",
                "Plan to renew or replace this startup kit before it expires.",
            )
        )
    else:
        cert_status = "ok"
    metadata["certificate"] = {
        "path": str(cert_path),
        "expires_at": _format_utc(expires_at),
        "days_remaining": days_remaining,
        "status": cert_status,
    }


def _format_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _optional_dependency_summary() -> list[dict]:
    return [{"name": name, "available": importlib.util.find_spec(name) is not None} for name in OPTIONAL_DEPENDENCIES]


def _skills_summary() -> dict:
    try:
        source = find_skill_source()
    except Exception as e:
        return {
            "status": "error",
            "source": None,
            "available_count": 0,
            "findings": [_finding("AGENT_SKILL_SOURCE_UNAVAILABLE", "error", str(e))],
        }

    manifest = source.manifest or {}
    return {
        "status": "ok",
        "source": {
            "type": source.source_type,
            "root": str(source.root),
            "manifest_schema_version": manifest.get("schema_version"),
            "nvflare_version": manifest.get("nvflare_version"),
        },
        "available_count": len(manifest.get("skills", [])),
        "available": manifest.get("skills", []),
        "findings": manifest.get("findings", []),
    }


def _poc_summary(config) -> dict:
    from nvflare.tool.poc.poc_commands import DEFAULT_WORKSPACE

    workspace = os.getenv("NVFLARE_POC_WORKSPACE") or _configured_poc_workspace(config) or DEFAULT_WORKSPACE
    if not isinstance(workspace, str) or not workspace or len(workspace) > MAX_DOCTOR_TEXT_LENGTH:
        workspace = DEFAULT_WORKSPACE
    workspace_path = Path(workspace).expanduser()
    findings = []
    status = "present" if _real_directory_exists(workspace_path) else "missing"
    if status == "missing":
        findings.append(
            _finding(
                "POC_WORKSPACE_MISSING",
                "info",
                f"POC workspace does not exist: {workspace}",
                "Run nvflare poc prepare when local POC execution is needed.",
            )
        )
    return {
        "workspace": str(workspace_path),
        "status": status,
        "startup_dir_exists": _real_directory_exists(workspace_path / "startup"),
        "local_dir_exists": _real_directory_exists(workspace_path / "local"),
        "findings": findings,
    }


def _configured_poc_workspace(config) -> str | None:
    if config is None:
        return None
    try:
        workspace = config.get("poc.workspace", None) or config.get("poc_workspace.path", None)
    except Exception:
        return None
    return workspace if isinstance(workspace, str) else None


def _online_summary(args, cli_config, cli_config_error) -> dict:
    from nvflare.fuel.flare_api.api_spec import AuthenticationError, NoConnection
    from nvflare.tool.cli_output import get_connect_timeout
    from nvflare.tool.kit.kit_config import StartupKitConfigError

    try:
        startup_kit = _resolve_doctor_startup_kit(args, cli_config, cli_config_error)
    except StartupKitConfigError as e:
        return {
            "enabled": True,
            "status": "skipped",
            "startup_kit": None,
            "recommended_ttl_seconds": 0,
            "findings": [_finding("STARTUP_KIT_NOT_READY", "warning", str(e), getattr(e, "hint", None))],
        }

    session = None
    result = None
    try:
        session = _new_doctor_status_session(startup_kit, timeout=get_connect_timeout())
        status_fields, truncation = _bounded_online_status(session.check_status("all", None))
        findings = []
        online_status = "ok"
        if truncation:
            online_status = "partial"
            findings.append(
                _finding(
                    "ONLINE_STATUS_TRUNCATED",
                    "warning",
                    "Online status exceeded report bounds and was truncated: " + ", ".join(truncation),
                )
            )
        result = {
            "enabled": True,
            "status": online_status,
            "startup_kit": startup_kit,
            **status_fields,
            "status_truncated": bool(truncation),
            "status_truncation": truncation,
            "recommended_ttl_seconds": 30,
            "findings": findings,
        }
    except AuthenticationError as e:
        result = _online_error("AUTHENTICATION_FAILED", "authentication_failed", str(e), startup_kit)
    except NoConnection as e:
        result = _online_error("CONNECTION_FAILED", "connection_failed", str(e), startup_kit)
    except TimeoutError as e:
        result = _online_error("ONLINE_CHECK_TIMEOUT", "timeout", str(e), startup_kit)
    except Exception as e:
        result = _online_error(f"ONLINE_CHECK_FAILED_{type(e).__name__.upper()}", "error", str(e), startup_kit)
    finally:
        if session is not None:
            cleanup_error = _close_doctor_session(session)
            if cleanup_error and result is not None:
                result["findings"].append(
                    _finding("ONLINE_SESSION_CLEANUP_FAILED", "warning", _bounded_text(cleanup_error))
                )
                if result.get("status") == "ok":
                    result["status"] = "partial"
    return result


def _resolve_doctor_startup_kit(args, config, config_error) -> dict:
    """Resolve a kit path structurally, before any fed_admin.json parsing."""

    from nvflare.tool.cli_arg_utils import get_arg_value
    from nvflare.tool.kit.kit_config import (
        NVFLARE_STARTUP_KIT_DIR,
        StartupKitConfigError,
        get_active_startup_kit_id,
        get_startup_kit_entries,
    )

    kit_id = get_arg_value(args, "kit_id")
    startup_kit = get_arg_value(args, "startup_kit")
    if kit_id and startup_kit:
        raise StartupKitConfigError("--kit-id and --startup-kit are mutually exclusive")
    if startup_kit:
        source, selected_id, path = "startup_kit", None, startup_kit
    elif kit_id:
        if config_error:
            raise config_error
        entries = get_startup_kit_entries(config) if config is not None else {}
        if kit_id not in entries:
            raise StartupKitConfigError(f"startup kit id '{_bounded_text(kit_id)}' is not registered")
        source, selected_id, path = "kit_id", kit_id, entries[kit_id]
    else:
        env_path = os.getenv(NVFLARE_STARTUP_KIT_DIR)
        if env_path and env_path.strip():
            source, selected_id, path = "env", None, env_path
        else:
            if config_error:
                raise config_error
            active = get_active_startup_kit_id(config) if config is not None else None
            entries = get_startup_kit_entries(config) if config is not None else {}
            if not active or active not in entries:
                raise StartupKitConfigError(
                    "no active startup kit is configured",
                    hint="Pass --kit-id or --startup-kit, or configure an active startup kit.",
                )
            source, selected_id, path = "active", active, entries[active]

    if not isinstance(path, str) or not path or len(path) > MAX_DOCTOR_TEXT_LENGTH:
        raise StartupKitConfigError("startup kit path must be a bounded non-empty string")
    normalized = _normalize_startup_kit_root(path)
    return {"source": source, "id": selected_id, "path": str(normalized)}


def _bounded_online_status(status) -> tuple[dict, list[str]]:
    if type(status) is not dict:
        raise ValueError("online status response must be a dictionary")

    truncation = []
    budget = {"nodes": MAX_ONLINE_STATUS_NODES}
    server_status = status.get("server_status")
    if server_status is not None and type(server_status) is not str:
        raise ValueError("online server_status must be a string or null")
    server_start_time = status.get("server_start_time")
    if server_start_time is not None and type(server_start_time) not in (int, float, str):
        raise ValueError("online server_start_time must be a scalar or null")

    result = {
        "server_status": _bounded_remote_value(server_status, "server_status", 0, budget, truncation),
        "server_start_time": _bounded_remote_value(server_start_time, "server_start_time", 0, budget, truncation),
    }
    for key in ("clients", "client_status", "jobs"):
        value = status.get(key, [])
        if type(value) is not list:
            raise ValueError(f"online {key} must be a list")
        if any(type(item) is not dict for item in value[: MAX_ONLINE_STATUS_ITEMS + 1]):
            raise ValueError(f"online {key} entries must be dictionaries")
        if len(value) > MAX_ONLINE_STATUS_ITEMS:
            truncation.append(f"{key}:items")
        result[key] = [
            _bounded_remote_value(item, f"{key}[{index}]", 1, budget, truncation)
            for index, item in enumerate(value[:MAX_ONLINE_STATUS_ITEMS])
        ]
    details = sorted(set(truncation))
    if len(details) > MAX_ONLINE_TRUNCATION_DETAILS:
        details = [*details[:MAX_ONLINE_TRUNCATION_DETAILS], "additional_truncation"]
    return result, details


def _bounded_remote_value(value, path: str, depth: int, budget: dict, truncation: list[str]):
    budget["nodes"] -= 1
    if budget["nodes"] < 0:
        if "node_budget" not in truncation:
            truncation.append("node_budget")
        return None
    if value is None or type(value) is bool:
        return value
    if type(value) is int:
        if value.bit_length() > 4096:
            raise ValueError(f"online status integer at {path} exceeds the supported range")
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"online status float at {path} must be finite")
        return value
    if type(value) is str:
        if len(value) > MAX_ONLINE_STATUS_TEXT_LENGTH:
            truncation.append(f"{path}:text")
            return value[:MAX_ONLINE_STATUS_TEXT_LENGTH]
        return value
    if depth >= MAX_ONLINE_STATUS_DEPTH:
        truncation.append(f"{path}:depth")
        return None
    if type(value) is list:
        if len(value) > MAX_ONLINE_STATUS_NESTED_ITEMS:
            truncation.append(f"{path}:items")
        return [
            _bounded_remote_value(item, f"{path}[{index}]", depth + 1, budget, truncation)
            for index, item in enumerate(value[:MAX_ONLINE_STATUS_NESTED_ITEMS])
        ]
    if type(value) is dict:
        if len(value) > MAX_ONLINE_STATUS_FIELDS:
            truncation.append(f"{path}:fields")
        result = {}
        for key, item in islice(value.items(), MAX_ONLINE_STATUS_FIELDS):
            if type(key) is not str or not key:
                raise ValueError(f"online status keys at {path} must be non-empty strings")
            safe_key = key[:MAX_ONLINE_STATUS_KEY_LENGTH]
            if safe_key != key:
                truncation.append(f"{path}:key")
            if safe_key in result:
                truncation.append(f"{path}:key_collision")
                continue
            result[safe_key] = _bounded_remote_value(item, f"{path}.{safe_key}", depth + 1, budget, truncation)
        return result
    raise ValueError(f"online status value at {path} has unsupported type {type(value).__name__}")


def _online_error(code: str, status: str, message: str, startup_kit: dict) -> dict:
    return {
        "enabled": True,
        "status": status,
        "startup_kit": startup_kit,
        "recommended_ttl_seconds": 0,
        "findings": [_finding(code, "warning", _bounded_text(message))],
    }


def format_doctor_human(data: dict) -> str:
    """Render a concise human-readable doctor summary."""
    lines = ["NVFLARE Agent Doctor", f"status: {data.get('status', 'unknown')}", ""]
    if "conversion_status" in data or "deployment_status" in data:
        lines.append(
            "readiness: "
            f"conversion {data.get('conversion_status', 'unknown')}; "
            f"deployment {data.get('deployment_status', 'unknown')}"
        )

    nvflare_info = data.get("nvflare") or {}
    nvflare_status = "import ok" if nvflare_info.get("import_ok") else "import failed"
    lines.append(f"nvflare: {nvflare_info.get('version', 'unknown')} ({nvflare_status})")

    commands = data.get("commands") or {}
    lines.append(f"commands: {commands.get('status', 'unknown')} ({len(commands.get('commands', []))} registered)")

    startup_kits = data.get("startup_kits") or {}
    startup_entries = startup_kits.get("entries", [])
    valid_startup_count = sum(1 for entry in startup_entries if entry.get("status") == "ok")
    active_id = startup_kits.get("active_id") or "none"
    lines.append(f"startup kits: {valid_startup_count}/{len(startup_entries)} valid (active: {active_id})")

    skills = data.get("skills") or {}
    source = skills.get("source")
    if source:
        lines.append(
            f"skills: {skills.get('available_count', 0)} available "
            f"({source.get('type', 'unknown')}: {source.get('root', 'unknown')})"
        )
    else:
        lines.append(f"skills: {skills.get('status', 'unknown')}")

    optional_dependencies = data.get("optional_dependencies", [])
    available_deps = [dep["name"] for dep in optional_dependencies if dep.get("available")]
    missing_deps = [dep["name"] for dep in optional_dependencies if not dep.get("available")]
    lines.append(f"optional dependencies: available {_join_names(available_deps)}; missing {_join_names(missing_deps)}")

    poc = data.get("poc") or {}
    lines.append(f"poc: {poc.get('status', 'unknown')} (workspace: {poc.get('workspace', 'unknown')})")

    online = data.get("online") or {}
    if online.get("enabled"):
        online_line = f"online: {online.get('status', 'unknown')}"
        first_online_finding = _first_finding(online.get("findings", []))
        if first_online_finding:
            online_line += f" ({first_online_finding})"
        lines.append(online_line)
    else:
        lines.append(f"online: {online.get('status', 'not_requested')}")

    findings = data.get("findings", [])
    if findings:
        lines.append("")
        lines.append(f"findings ({len(findings)}):")
        for finding in findings:
            lines.extend(_format_finding(finding))
    else:
        lines.append("")
        lines.append("findings: none")

    return "\n".join(lines)


def _join_names(names: list[str]) -> str:
    return ", ".join(names) if names else "none"


def _first_finding(findings: list[dict]) -> str | None:
    if not findings:
        return None
    finding = findings[0]
    return f"{finding.get('severity', 'info')} {finding.get('code', 'UNKNOWN')}"


def _format_finding(finding: dict) -> list[str]:
    severity = finding.get("severity", "info")
    code = finding.get("code", "UNKNOWN")
    message = str(finding.get("message", "")).splitlines() or [""]
    lines = [f"- {severity} {code}: {message[0]}"]
    lines.extend(f"  {line}" for line in message[1:])
    if finding.get("hint"):
        lines.append(f"  hint: {finding['hint']}")
    return lines


def _new_doctor_status_session(startup_kit: dict, *, timeout: float):
    """Create a status-only session without startup-kit custom code or file transfer."""

    from nvflare.apis.job_def import DEFAULT_STUDY
    from nvflare.fuel.flare_api.flare_api import Session
    from nvflare.fuel.hci.client.api import AdminAPI
    from nvflare.fuel.hci.client.api_spec import AdminConfigKey

    kit_path = _normalize_startup_kit_root(startup_kit["path"])
    startup_dir = kit_path / "startup"
    config = _load_status_admin_config(kit_path)
    admin = dict(config[AdminConfigKey.ADMIN])
    credential_dir = tempfile.TemporaryDirectory(prefix="nvflare-doctor-")
    credential_bytes = {}
    credential_names = {
        AdminConfigKey.CLIENT_KEY: "client.key",
        AdminConfigKey.CLIENT_CERT: "client.crt",
        AdminConfigKey.CA_CERT: "rootCA.pem",
    }
    try:
        for key, file_name in credential_names.items():
            value = admin.get(key)
            if not value:
                continue
            content = _read_startup_member(startup_dir, value)
            credential_bytes[key] = content
            copied_path = Path(credential_dir.name) / file_name
            _write_private_file(copied_path, content)
            admin[key] = str(copied_path)

        username = admin.get(AdminConfigKey.USERNAME)
        if not isinstance(username, str) or not username:
            username = _certificate_identity(credential_bytes.get(AdminConfigKey.CLIENT_CERT))
        if not username:
            raise ValueError("admin identity is missing from fed_admin.json and the client certificate")

        # Deliberately bypass Session.__init__: it loads custom handler classes
        # and creates transfer directories. TLS consumes private copies made
        # from the same bounded file reads, so later startup-kit replacement
        # cannot change the credentials actually used for this check.
        session = Session.__new__(Session)
        session.startup_path = str(kit_path)
        session.secure_mode = True
        session.username = username
        session._debug = False
        session._study = DEFAULT_STUDY
        session.upload_dir = None
        session.download_dir = None
        session._doctor_credential_dir = credential_dir
        session.api = AdminAPI(
            admin_config=admin,
            user_name=username,
            debug=False,
            cmd_modules=[],
            event_handlers=[],
            study=DEFAULT_STUDY,
        )
        original_close = session.close

        def close_status_session():
            try:
                original_close()
            finally:
                credential_dir.cleanup()

        session.close = close_status_session
        try:
            session.try_connect(timeout)
        except Exception:
            try:
                session.close()
            except Exception:
                pass
            credential_dir.cleanup()
            raise
        return session
    except Exception:
        credential_dir.cleanup()
        raise


def _close_doctor_session(session) -> str | None:
    errors = []
    try:
        session.close()
    except Exception as e:
        errors.append(f"session close failed: {type(e).__name__}: {e}")
    credential_dir = getattr(session, "_doctor_credential_dir", None)
    if credential_dir is not None:
        try:
            credential_dir.cleanup()
        except Exception as e:
            errors.append(f"credential cleanup failed: {type(e).__name__}: {e}")
    return "; ".join(errors) if errors else None


def _load_status_admin_config(kit_path: Path) -> dict:
    """Load and, when provisioned, verify fed_admin.json without executing config components."""

    startup_dir = kit_path / "startup"
    _reject_symlink_components(kit_path)
    _reject_symlink_components(startup_dir)
    config_path = startup_dir / "fed_admin.json"
    config_bytes = _read_bounded_regular_file(config_path, MAX_STARTUP_CONFIG_BYTES)

    signature_path = startup_dir / "signature.json"
    root_cert_path = startup_dir / "rootCA.pem"
    signature_exists = _regular_path_exists(signature_path)
    root_cert_exists = _regular_path_exists(root_cert_path)
    if signature_exists:
        if not root_cert_exists:
            raise ValueError("signed startup kit must contain rootCA.pem")
        _verify_startup_config_signature(
            config_bytes,
            _read_bounded_regular_file(signature_path, MAX_STARTUP_CONFIG_BYTES),
            _read_bounded_regular_file(root_cert_path, MAX_STARTUP_CONFIG_BYTES),
        )

    try:
        config = json.loads(config_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise ValueError("fed_admin.json must be bounded UTF-8 JSON") from e
    if not isinstance(config, dict) or not isinstance(config.get("admin"), dict):
        raise ValueError("fed_admin.json must contain an admin object")
    return config


def _verify_startup_config_signature(config_bytes: bytes, signature_bytes: bytes, root_cert_bytes: bytes) -> None:
    from base64 import b64decode

    from cryptography import x509
    from cryptography.hazmat.backends import default_backend
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import padding

    try:
        signatures = json.loads(signature_bytes.decode("utf-8"))
        encoded_signature = signatures["fed_admin.json"]
        signature = b64decode(encoded_signature.encode("utf-8"), validate=True)
        certificate = x509.load_pem_x509_certificate(root_cert_bytes, default_backend())
        certificate.public_key().verify(
            signature=signature,
            data=config_bytes,
            padding=padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
            algorithm=hashes.SHA256(),
        )
    except Exception as e:
        raise ValueError("fed_admin.json signature verification failed") from e


def _normalize_startup_kit_root(path: str | Path) -> Path:
    if not isinstance(path, (str, Path)) or not str(path):
        raise ValueError("startup-kit path must be a non-empty path")
    normalized = Path(os.path.abspath(os.path.normpath(str(Path(path).expanduser()))))
    if normalized.name == "startup":
        normalized = normalized.parent
    _reject_symlink_components(normalized)
    _require_real_directory(normalized)
    _require_real_directory(normalized / "startup")
    return normalized


def _startup_member_path(startup_dir: Path, value: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("startup-kit credential paths must be non-empty strings")
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = startup_dir / candidate
    normalized = Path(os.path.abspath(os.path.normpath(str(candidate))))
    startup_normalized = Path(os.path.abspath(os.path.normpath(str(startup_dir))))
    try:
        normalized.relative_to(startup_normalized)
    except ValueError as e:
        raise ValueError(f"startup-kit credential path escapes startup directory: {value}") from e
    _reject_symlink_components(normalized)
    return normalized


def _read_startup_member(startup_dir: Path, value: str) -> bytes:
    return _read_bounded_regular_file(_startup_member_path(startup_dir, value), MAX_STARTUP_CONFIG_BYTES)


def _write_private_file(path: Path, content: bytes) -> None:
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_BINARY", 0)
    )
    fd = os.open(path, flags, 0o600)
    try:
        if hasattr(os, "fchmod"):
            os.fchmod(fd, 0o600)
        view = memoryview(content)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise OSError("short write while copying startup-kit credential")
            view = view[written:]
        os.fsync(fd)
    finally:
        os.close(fd)


def _certificate_identity(cert_bytes: bytes | None) -> str | None:
    if not cert_bytes:
        return None
    try:
        from cryptography import x509
        from cryptography.hazmat.backends import default_backend
        from cryptography.x509.oid import NameOID

        certificate = x509.load_pem_x509_certificate(cert_bytes, default_backend())
        names = certificate.subject.get_attributes_for_oid(NameOID.COMMON_NAME)
        identity = names[0].value if names else None
        return identity if isinstance(identity, str) and identity else None
    except Exception:
        return None


def _regular_path_exists(path: Path) -> bool:
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError:
        return False
    if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
        raise ValueError(f"startup-kit path must be a regular file: {path}")
    return True


def _require_real_directory(path: Path) -> None:
    mode = path.lstat().st_mode
    if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
        raise ValueError(f"startup-kit path must be a real directory: {path}")


def _real_directory_exists(path: Path) -> bool:
    try:
        normalized = Path(os.path.abspath(os.path.normpath(str(path))))
        _reject_symlink_components(normalized)
        _require_real_directory(normalized)
        return True
    except (OSError, ValueError):
        return False


def _reject_symlink_components(path: Path) -> None:
    absolute = Path(os.path.abspath(os.path.normpath(str(path))))
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current = current / part
        mode = current.lstat().st_mode
        if stat.S_ISLNK(mode):
            raise ValueError(f"startup-kit path must not contain symlinks: {current}")


def _read_bounded_regular_file(path: Path, max_bytes: int) -> bytes:
    before = path.lstat()
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise ValueError(f"startup-kit path must be a regular file: {path}")
    if before.st_nlink != 1:
        raise ValueError(f"startup-kit path must not be hard-linked: {path}")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
        | getattr(os, "O_BINARY", 0)
    )
    fd = _open_regular_from_anchored_parents(path, flags)
    try:
        opened = os.fstat(fd)
        if not stat.S_ISREG(opened.st_mode) or (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise ValueError(f"startup-kit path must be a regular file: {path}")
        chunks = []
        total = 0
        while total <= max_bytes:
            chunk = os.read(fd, min(64 * 1024, max_bytes + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
        if total > max_bytes:
            raise ValueError(f"startup-kit file exceeds {max_bytes} bytes: {path}")
        current = path.lstat()
        if (
            stat.S_ISLNK(current.st_mode)
            or (current.st_dev, current.st_ino) != (opened.st_dev, opened.st_ino)
            or current.st_nlink != 1
            or current.st_size != opened.st_size
            or current.st_mtime_ns != opened.st_mtime_ns
        ):
            raise ValueError(f"startup-kit file changed while being read: {path}")
        return b"".join(chunks)
    finally:
        os.close(fd)


def _open_regular_from_anchored_parents(path: Path, flags: int) -> int:
    """Open an absolute file through no-follow directory descriptors when supported."""

    absolute = Path(os.path.abspath(os.path.normpath(str(path))))
    supports_dir_fd = os.open in getattr(os, "supports_dir_fd", set())
    directory_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_DIRECTORY", 0)
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    if not supports_dir_fd or not nofollow or not getattr(os, "O_DIRECTORY", 0):
        return os.open(absolute, flags)

    directory_fd = os.open(absolute.anchor, directory_flags | nofollow)
    try:
        for component in absolute.parts[1:-1]:
            next_fd = os.open(component, directory_flags | nofollow, dir_fd=directory_fd)
            try:
                if not stat.S_ISDIR(os.fstat(next_fd).st_mode):
                    raise ValueError(f"startup-kit path component must be a directory: {component}")
            except Exception:
                os.close(next_fd)
                raise
            os.close(directory_fd)
            directory_fd = next_fd
        return os.open(absolute.name, flags, dir_fd=directory_fd)
    finally:
        os.close(directory_fd)


def _finding(code: str, severity: str, message: str, hint: str = None) -> dict:
    result = {"code": code, "severity": severity, "message": message}
    if hint:
        result["hint"] = hint
    return result


def _has_attention_findings(findings: list[dict]) -> bool:
    return any(finding.get("severity") in {"warning", "error"} for finding in findings)


def _bounded_text(value) -> str:
    text = str(value)
    return text if len(text) <= MAX_DOCTOR_TEXT_LENGTH else text[:MAX_DOCTOR_TEXT_LENGTH]


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
