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
import os
from datetime import datetime, timezone
from pathlib import Path

import nvflare
from nvflare.tool.agent.command_registry import agent_command_registry
from nvflare.tool.agent.skill_manager import find_skill_source

OPTIONAL_DEPENDENCIES = ("torch", "tensorflow", "sklearn", "xgboost", "jax", "flwr")


def doctor_environment(*, online: bool = False, args=None) -> dict:
    """Return a read-only local readiness snapshot, optionally with a bounded online check."""
    data = {
        "schema_version": "1",
        "timestamp": _now_utc(),
        "nvflare": {"import_ok": True, "version": nvflare.__version__},
        "commands": _command_registry(),
        "startup_kits": _startup_kit_summary(),
        "optional_dependencies": _optional_dependency_summary(),
        "skills": _skills_summary(),
        "poc": _poc_summary(),
        "online": {"enabled": False, "status": "not_requested"},
        "findings": [],
    }
    data["findings"].extend(data["startup_kits"].get("findings", []))
    data["findings"].extend(data["skills"].get("findings", []))
    data["findings"].extend(data["poc"].get("findings", []))

    if online:
        data["online"] = _online_summary(args)
        data["findings"].extend(data["online"].get("findings", []))

    data["status"] = "attention" if data["findings"] else "ok"
    return data


def _command_registry() -> dict:
    return agent_command_registry()


def _startup_kit_summary() -> dict:
    from nvflare.tool.kit.kit_config import (
        StartupKitConfigError,
        get_active_startup_kit_id,
        get_cli_config_path,
        get_startup_kit_entries,
        get_startup_kit_status,
        load_cli_config,
    )

    config_file = str(get_cli_config_path())
    try:
        config = load_cli_config()
        active = get_active_startup_kit_id(config)
        entries = get_startup_kit_entries(config)
    except StartupKitConfigError as e:
        return {
            "config_file": config_file,
            "active_id": None,
            "entries": [],
            "findings": [_finding("STARTUP_KIT_CONFIG_INVALID", "error", str(e), getattr(e, "hint", None))],
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

    entry_rows = []
    for kit_id, path in sorted(entries.items()):
        status, normalized_path, metadata = get_startup_kit_status(path)
        entry_rows.append(
            {
                "id": kit_id,
                "path": path,
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
        "active_id": active,
        "entries": entry_rows,
        "findings": findings,
    }


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


def _poc_summary() -> dict:
    from nvflare.tool.poc.poc_commands import DEFAULT_WORKSPACE

    workspace = os.getenv("NVFLARE_POC_WORKSPACE") or _configured_poc_workspace() or DEFAULT_WORKSPACE
    workspace_path = Path(workspace).expanduser()
    findings = []
    status = "present" if workspace_path.exists() else "missing"
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
        "startup_dir_exists": (workspace_path / "startup").is_dir(),
        "local_dir_exists": (workspace_path / "local").is_dir(),
        "findings": findings,
    }


def _configured_poc_workspace() -> str | None:
    from nvflare.tool.kit.kit_config import get_cli_config_path

    try:
        from pyhocon import ConfigFactory as CF
    except ImportError:
        return None

    config_path = get_cli_config_path()
    if not config_path.is_file():
        return None
    try:
        config = CF.parse_file(str(config_path))
    except Exception:
        return None
    return config.get("poc.workspace", None) or config.get("poc_workspace.path", None)


def _online_summary(args) -> dict:
    from nvflare.fuel.flare_api.api_spec import AuthenticationError, NoConnection
    from nvflare.tool.cli_output import get_connect_timeout
    from nvflare.tool.cli_session import new_cli_session_for_args, resolve_startup_kit_info_for_args
    from nvflare.tool.kit.kit_config import StartupKitConfigError

    try:
        startup_kit = resolve_startup_kit_info_for_args(args)
    except StartupKitConfigError as e:
        return {
            "enabled": True,
            "status": "skipped",
            "startup_kit": None,
            "recommended_ttl_seconds": 0,
            "findings": [_finding("STARTUP_KIT_NOT_READY", "warning", str(e), getattr(e, "hint", None))],
        }

    read_only_finding = _online_read_only_preflight(startup_kit)
    if read_only_finding:
        return {
            "enabled": True,
            "status": "skipped",
            "startup_kit": startup_kit,
            "recommended_ttl_seconds": 0,
            "findings": [read_only_finding],
        }

    session = None
    try:
        session = new_cli_session_for_args(args=args, timeout=get_connect_timeout())
        status = session.check_status("all", None)
        return {
            "enabled": True,
            "status": "ok",
            "startup_kit": startup_kit,
            "server_status": status.get("server_status"),
            "server_start_time": status.get("server_start_time"),
            "clients": status.get("clients", []),
            "client_status": status.get("client_status", []),
            "jobs": status.get("jobs", []),
            "recommended_ttl_seconds": 30,
            "findings": [],
        }
    except AuthenticationError as e:
        return _online_error("AUTHENTICATION_FAILED", "authentication_failed", str(e), startup_kit)
    except NoConnection as e:
        return _online_error("CONNECTION_FAILED", "connection_failed", str(e), startup_kit)
    except TimeoutError as e:
        return _online_error("ONLINE_CHECK_TIMEOUT", "timeout", str(e), startup_kit)
    except Exception as e:
        return _online_error(f"ONLINE_CHECK_FAILED_{type(e).__name__.upper()}", "error", str(e), startup_kit)
    finally:
        if session is not None:
            session.close()


def _online_error(code: str, status: str, message: str, startup_kit: dict) -> dict:
    return {
        "enabled": True,
        "status": status,
        "startup_kit": startup_kit,
        "recommended_ttl_seconds": 0,
        "findings": [_finding(code, "warning", message)],
    }


def format_doctor_human(data: dict) -> str:
    """Render a concise human-readable doctor summary."""
    lines = ["NVFLARE Agent Doctor", f"status: {data.get('status', 'unknown')}", ""]

    nvflare_info = data.get("nvflare") or {}
    nvflare_status = "import ok" if nvflare_info.get("import_ok") else "import failed"
    lines.append(f"nvflare: {nvflare_info.get('version', 'unknown')} ({nvflare_status})")

    commands = data.get("commands") or {}
    lines.append(f"commands: {commands.get('status', 'unknown')} ({len(commands.get('commands', []))} registered)")

    startup_kits = data.get("startup_kits") or {}
    startup_entries = startup_kits.get("entries", [])
    valid_startup_count = sum(1 for entry in startup_entries if entry.get("status") == "valid")
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


def _online_read_only_preflight(startup_kit: dict) -> dict | None:
    """Skip online checks when the existing session API would create local directories."""
    kit_path = Path(startup_kit["path"])
    admin_config_path = kit_path / "startup" / "fed_admin.json"
    try:
        config = json.loads(admin_config_path.read_text(encoding="utf-8"))
    except Exception:
        return _finding(
            "ONLINE_CHECK_ADMIN_CONFIG_UNREADABLE",
            "warning",
            "Online check skipped because fed_admin.json could not be read before guarded session creation.",
        )

    admin = config.get("admin") if isinstance(config, dict) else None
    download_dir = admin.get("download_dir") if isinstance(admin, dict) else None
    if not download_dir:
        return _finding(
            "ONLINE_CHECK_REQUIRES_DOWNLOAD_DIR",
            "warning",
            "Online check skipped because the active admin config has no pre-existing download_dir.",
            "Create the startup kit transfer directory or use nvflare system status for the normal CLI path.",
        )
    if not isinstance(download_dir, str):
        return _finding(
            "ONLINE_CHECK_DOWNLOAD_DIR_INVALID",
            "warning",
            "Online check skipped because fed_admin.json download_dir is not a string.",
            "Set admin.download_dir to a pre-existing directory path or use nvflare system status for the normal CLI path.",
        )

    download_path = Path(download_dir)
    if not download_path.is_absolute():
        download_path = kit_path / download_path
    if not download_path.is_dir():
        return _finding(
            "ONLINE_CHECK_WOULD_CREATE_DOWNLOAD_DIR",
            "warning",
            "Online check skipped because the normal session path would create download_dir.",
            "Create the configured download_dir first or use nvflare system status for the normal CLI path.",
        )
    return None


def _finding(code: str, severity: str, message: str, hint: str = None) -> dict:
    result = {"code": code, "severity": severity, "message": message}
    if hint:
        result["hint"] = hint
    return result


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
