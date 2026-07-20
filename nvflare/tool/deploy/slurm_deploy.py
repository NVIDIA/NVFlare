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

"""Slurm-specific implementation for ``nvflare deploy prepare``."""

from __future__ import annotations

import re
import shlex
import shutil
from pathlib import Path
from typing import Any

from nvflare.app_opt.job_launcher.slurm.config import (
    SCHEMA_VERSION,
    SLURM_COMPUTE_EXECUTABLES,
    SLURM_PARENT_EXECUTABLES,
    SlurmLauncherError,
    normalize_slurm_directives,
    normalize_slurm_executables,
    normalize_slurm_launcher_settings,
    normalize_slurm_workspace_path,
)
from nvflare.tool.deploy.deploy_common import (
    COMM_CONFIG_JSON,
    RESOURCES_JSON_DEFAULT,
    ROLE_SERVER,
    RUNTIME_SLURM,
    SLURM_CLIENT_LAUNCHER,
    SLURM_PARENT_SH,
    SLURM_SERVER_LAUNCHER,
    SLURM_STAGE_MANIFEST,
    SLURM_START_SH,
    SUB_START_SH,
    KitInfo,
    _ensure_study_runtime_template,
    _fail,
    _internal_resources,
    _load_or_default_comm_config,
    _mapping,
    _patch_resources,
    _paths_overlap,
    _relocate_server_storage_to_workspace,
    _remove_start_scripts,
    _required_str,
    _validate_allowed_keys,
    _write_json,
)

TEMPLATE_TOKEN_PATTERN = re.compile(r"@@NVFLARE_[A-Z0-9_]+@@")
TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"


def prepare(kit_info: KitInfo, final_output: Path, config: dict[str, Any]) -> dict[str, Any]:
    """Generate the Slurm launcher configuration and startup artifacts."""
    if (kit_info.kit_dir / "local" / SLURM_STAGE_MANIFEST).exists():
        _fail(
            "INVALID_KIT",
            "The input is already a prepared Slurm kit.",
            "Prepare from the original provisioned startup kit.",
        )
    _validate_role_config(config, kit_info.role)
    config["workspace_path"] = _validate_workspace_location(config["workspace_path"], kit_info.kit_dir, final_output)
    workspace_path = config["workspace_path"]
    job_launcher = config["job_launcher"]

    executables = _resolve_executables(job_launcher.get("executables") or {})
    launcher_path = SLURM_SERVER_LAUNCHER if kit_info.role == ROLE_SERVER else SLURM_CLIENT_LAUNCHER
    launcher_args = _normalize_job_launcher(job_launcher, workspace_path, executables)
    launcher_args["prepared_path"] = str(final_output)
    _patch_resources(kit_info.kit_dir, "slurm_launcher", launcher_path, launcher_args)
    _patch_comm_config(kit_info.kit_dir, port=launcher_args["internal_port"])
    _ensure_study_runtime_template(kit_info.kit_dir)
    if kit_info.role == ROLE_SERVER:
        _relocate_server_storage_to_workspace(kit_info.kit_dir, workspace_path)

    _write_json(
        kit_info.kit_dir / "local" / SLURM_STAGE_MANIFEST,
        {
            "schema_version": SCHEMA_VERSION,
            "site": kit_info.name,
            "prepared_path": str(final_output),
            "workspace_path": workspace_path,
        },
    )

    _remove_start_scripts(kit_info.kit_dir, {SUB_START_SH})
    _write_start_script(kit_info.kit_dir, workspace_path)

    if config.get("parent") is not None:
        parent = config["parent"]
        _write_parent_script(kit_info.kit_dir, final_output, parent)

    final_start_script = final_output / "startup" / SLURM_START_SH
    result = {
        "runtime": RUNTIME_SLURM,
        "role": kit_info.role,
        "name": kit_info.name,
        "workspace_path": workspace_path,
        "output": str(final_output),
        "next_step": "Stage the prepared kit with the stage_command before starting the parent.",
        "stage_command": f"nvflare deploy slurm stage {shlex.quote(str(final_output))}",
        "start_command": f"cd {shlex.quote(str(final_output))} && ./startup/{SLURM_START_SH}",
        "start_script": str(final_start_script),
        "resources": str(final_output / "local" / RESOURCES_JSON_DEFAULT),
    }
    if config.get("parent") is not None:
        result["submit_command"] = shlex.join(
            [
                executables["sbatch"],
                "--parsable",
                f"--output={final_output}/parent-slurm-%j.out",
                str(final_output / "startup" / SLURM_PARENT_SH),
            ]
        )
    return result


def _validate_workspace_location(workspace_text: str, kit: Path, output: Path) -> str:
    try:
        workspace_text = normalize_slurm_workspace_path(workspace_text)
    except SlurmLauncherError as ex:
        _fail("INVALID_CONFIG", f"Invalid slurm config.workspace_path: {ex}", "Fix the workspace path.")
    try:
        workspace = Path(workspace_text).resolve(strict=False)
    except (OSError, RuntimeError) as ex:
        _fail(
            "INVALID_CONFIG",
            f"Cannot resolve slurm config.workspace_path: {ex}",
            "Fix the workspace path and any symlink loop.",
        )
    for label, protected in (("input kit", kit), ("prepared output", output)):
        if _paths_overlap(workspace, protected.resolve(strict=False)):
            _fail(
                "INVALID_CONFIG",
                f"slurm config.workspace_path overlaps the {label}: {protected}",
                "Use an exclusive shared workspace outside both the provisioned kit and prepared output.",
            )
    return str(workspace)


def _resolve_executables(configured: dict[str, str | None]) -> dict[str, str | None]:
    try:
        configured = normalize_slurm_executables(configured, require_parent=False)
    except SlurmLauncherError as ex:
        _fail("INVALID_CONFIG", str(ex), "Fix job_launcher.executables.")
    resolved: dict[str, str | None] = {}
    for name in SLURM_PARENT_EXECUTABLES:
        candidate = configured.get(name) or shutil.which(name)
        if not candidate:
            _fail(
                "INVALID_CONFIG",
                f"Could not resolve required Slurm executable: {name}",
                f"Install {name}, add it to PATH, or configure job_launcher.executables.{name}.",
            )
        try:
            path = Path(candidate).resolve(strict=True)
        except (OSError, RuntimeError) as ex:
            _fail("INVALID_CONFIG", f"Cannot resolve Slurm executable {name}: {ex}", "Fix the executable path.")
        resolved[name] = str(path)

    for name in SLURM_COMPUTE_EXECUTABLES:
        resolved[name] = configured.get(name)
    return resolved


def validate_config(config: dict[str, Any]) -> None:
    """Validate the complete Slurm deployment configuration."""
    _validate_allowed_keys(config, {"runtime", "workspace_path", "parent", "job_launcher"}, "slurm config")
    _required_str(config, "workspace_path", "slurm config")
    workspace_path = config["workspace_path"]

    if "parent" in config and config["parent"] is not None:
        parent = _mapping(config["parent"], "parent")
        _validate_allowed_keys(parent, {"sbatch_directives", "environment_setup"}, "parent")
        try:
            normalize_slurm_directives(parent.get("sbatch_directives"), "parent.sbatch_directives")
        except SlurmLauncherError as ex:
            _fail("INVALID_CONFIG", str(ex), "Fix the parent Slurm directives.")
        _optional_shell_text(parent, "environment_setup", "parent")

    if "job_launcher" not in config:
        _fail("INVALID_CONFIG", "slurm config.job_launcher is required.", "Add the Slurm launcher settings.")
    job_launcher = _mapping(config["job_launcher"], "job_launcher")
    _validate_allowed_keys(
        job_launcher,
        {
            "sandbox",
            "image",
            "internal_port",
            "sbatch_directives",
            "setup",
            "python_path",
            "executables",
            "forward_env",
            "parent_host",
            "poll_interval",
            "pending_timeout",
        },
        "job_launcher",
    )

    _normalize_job_launcher(job_launcher, workspace_path)


def _validate_role_config(config: dict[str, Any], role: str) -> None:
    if role == ROLE_SERVER and config.get("parent") is not None:
        _fail(
            "INVALID_CONFIG",
            "Slurm server kits do not support parent configuration.",
            "Run the server parent on a stable login/service host and remove parent from slurm.yaml.",
        )


def _optional_shell_text(data: dict[str, Any], key: str, where: str) -> str | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        _fail("INVALID_CONFIG", f"{where}.{key} must be a string.", "Fix the runtime config.")
    if "\x00" in value:
        _fail("INVALID_CONFIG", f"{where}.{key} must not contain NUL characters.", "Fix the runtime config.")
    return value


def _normalize_job_launcher(
    job_launcher: dict[str, Any],
    workspace_path: str,
    executables: dict[str, str | None] | None = None,
) -> dict:
    require_parent_executables = executables is not None
    if executables is None:
        executables = job_launcher.get("executables") or {}
    try:
        return normalize_slurm_launcher_settings(
            workspace_path=workspace_path,
            sandbox=job_launcher.get("sandbox"),
            python_path=job_launcher.get("python_path"),
            executables=executables,
            image=job_launcher.get("image"),
            internal_port=job_launcher.get("internal_port", 8102),
            sbatch_directives=job_launcher.get("sbatch_directives"),
            setup=job_launcher.get("setup"),
            forward_env=job_launcher.get("forward_env"),
            parent_host=job_launcher.get("parent_host"),
            poll_interval=job_launcher.get("poll_interval", 10),
            pending_timeout=job_launcher.get("pending_timeout", 600),
            require_parent_executables=require_parent_executables,
            require_image_file=True,
        )
    except SlurmLauncherError as ex:
        _fail("INVALID_CONFIG", str(ex), "Fix slurm config.job_launcher.")


def _patch_comm_config(kit_dir: Path, port: int) -> None:
    comm_config_path = kit_dir / "local" / COMM_CONFIG_JSON
    comm_config = _load_or_default_comm_config(comm_config_path)
    internal = comm_config.setdefault("internal", {})
    if not isinstance(internal, dict):
        _fail("INVALID_KIT", "comm_config.json internal must be a mapping.", "Fix the startup kit comm config.")
    internal["scheme"] = "tcp"
    resources = _internal_resources(comm_config)
    resources.update(
        {
            "host": "0.0.0.0",
            "port": port,
            "connection_security": "clear",
        }
    )
    _write_json(comm_config_path, comm_config)


def _render_template(
    relative_path: str,
    destination: Path,
    replacements: dict[str, str],
    mode: int,
) -> Path:
    source = TEMPLATES_DIR / relative_path
    if not source.is_file():
        _fail("DEPLOY_PREPARE_FAILED", f"Deployment template is missing: {source}", "Reinstall NVFlare.")
    template = source.read_text(encoding="utf-8")
    if missing := set(TEMPLATE_TOKEN_PATTERN.findall(template)).difference(replacements):
        _fail(
            "DEPLOY_PREPARE_FAILED",
            f"Deployment template contains unresolved placeholder(s) {sorted(missing)}: {source.name}.",
            "Reinstall NVFlare with matching deploy templates.",
        )
    rendered = TEMPLATE_TOKEN_PATTERN.sub(lambda match: replacements.get(match.group(0), match.group(0)), template)
    destination.write_text(rendered, encoding="utf-8")
    destination.chmod(mode)
    return destination


def _write_start_script(kit_dir: Path, workspace_path: str) -> Path:
    path = kit_dir / "startup" / SLURM_START_SH
    return _render_template(
        "slurm/start_slurm.sh",
        path,
        {"@@NVFLARE_WORKSPACE_PATH@@": shlex.quote(workspace_path)},
        mode=0o700,
    )


def _write_parent_script(kit_dir: Path, final_output: Path, parent: dict[str, Any]) -> Path:
    directive_lines = "".join(
        f"#SBATCH --{name}={value}\n" for name, value in (parent.get("sbatch_directives") or {}).items()
    )
    environment_setup = parent.get("environment_setup") or ""
    if environment_setup and not environment_setup.endswith("\n"):
        environment_setup += "\n"
    path = kit_dir / "startup" / SLURM_PARENT_SH
    return _render_template(
        "slurm/parent.slurm",
        path,
        {
            "@@NVFLARE_SBATCH_DIRECTIVES@@": directive_lines,
            "@@NVFLARE_ENVIRONMENT_SETUP@@": environment_setup,
            "@@NVFLARE_PREPARED_ROOT@@": shlex.quote(str(final_output)),
        },
        mode=0o700,
    )
