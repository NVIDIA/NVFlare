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
from datetime import datetime, timezone

import nvflare
from nvflare.tool.agent.command_registry import agent_command_registry
from nvflare.tool.agent.skill_manager import find_skill_source

OPTIONAL_DEPENDENCIES = ("torch", "tensorflow", "sklearn", "xgboost", "jax", "flwr")


def doctor_environment() -> dict:
    """Return a read-only local readiness snapshot for the conversion skills.

    Scope is conversion-only: NVFLARE import, the agent command surface, optional
    ML framework availability, and the installed skill bundle. Deployment/POC
    readiness (startup kits, POC workspace, live-server status) is intentionally
    out of scope and not checked here.
    """
    data = {
        "schema_version": "1",
        "timestamp": _now_utc(),
        "nvflare": {"import_ok": True, "version": nvflare.__version__},
        "commands": _command_registry(),
        "optional_dependencies": _optional_dependency_summary(),
        "skills": _skills_summary(),
        "findings": [],
    }
    data["findings"].extend(data["skills"].get("findings", []))
    data["status"] = "attention" if data["findings"] else "ok"
    return data


def _command_registry() -> dict:
    return agent_command_registry()


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


def format_doctor_human(data: dict) -> str:
    """Render a concise human-readable doctor summary."""
    lines = ["NVFLARE Agent Doctor", f"status: {data.get('status', 'unknown')}", ""]

    nvflare_info = data.get("nvflare") or {}
    nvflare_status = "import ok" if nvflare_info.get("import_ok") else "import failed"
    lines.append(f"nvflare: {nvflare_info.get('version', 'unknown')} ({nvflare_status})")

    commands = data.get("commands") or {}
    lines.append(f"commands: {commands.get('status', 'unknown')} ({len(commands.get('commands', []))} registered)")

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


def _format_finding(finding: dict) -> list[str]:
    severity = finding.get("severity", "info")
    code = finding.get("code", "UNKNOWN")
    message = str(finding.get("message", "")).splitlines() or [""]
    lines = [f"- {severity} {code}: {message[0]}"]
    lines.extend(f"  {line}" for line in message[1:])
    if finding.get("hint"):
        lines.append(f"  hint: {finding['hint']}")
    return lines


def _finding(code: str, severity: str, message: str, hint: str = None) -> dict:
    result = {"code": code, "severity": severity, "message": message}
    if hint:
        result["hint"] = hint
    return result


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
