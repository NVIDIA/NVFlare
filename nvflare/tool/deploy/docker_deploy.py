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

"""Docker-specific validation and preparation for ``nvflare deploy``."""

from __future__ import annotations

import re
import shlex
import stat
from pathlib import Path
from typing import Any

from nvflare.app_opt.job_launcher.study_runtime import RESERVED_DOCKER_KWARGS
from nvflare.tool.deploy.deploy_common import (
    COMM_CONFIG_JSON,
    DOCKER_CLIENT_LAUNCHER,
    DOCKER_SERVER_LAUNCHER,
    DOCKER_START_SH,
    FED_CLIENT_JSON,
    FED_SERVER_JSON,
    RESOURCES_JSON_DEFAULT,
    ROLE_SERVER,
    RUNTIME_DOCKER,
    WORKSPACE_MOUNT_PATH,
    KitInfo,
    _ensure_study_runtime_template,
    _fail,
    _internal_resources,
    _load_or_default_comm_config,
    _mapping,
    _optional_mapping,
    _optional_str,
    _patch_resources,
    _relocate_server_storage_to_workspace,
    _remove_start_scripts,
    _required_str,
    _validate_allowed_keys,
    _write_json,
)

DOCKER_RESERVED_KWARGS = RESERVED_DOCKER_KWARGS
DOCKER_START_TEMPLATE = Path(__file__).resolve().parent / "templates" / "docker" / "start_docker.sh"
DOCKER_TEMPLATE_TOKEN_PATTERN = re.compile(r"@@NVFLARE_[A-Z_]+@@")


def validate_config(config: dict[str, Any]) -> None:
    _validate_allowed_keys(config, {"runtime", "parent", "job_launcher"}, "docker config")
    parent = _mapping(config.get("parent"), "parent")
    job_launcher = _mapping(config.get("job_launcher", {}), "job_launcher")
    _validate_allowed_keys(parent, {"docker_image", "network"}, "parent")
    _validate_allowed_keys(
        job_launcher,
        {"default_python_path", "default_job_env", "default_job_container_kwargs"},
        "job_launcher",
    )
    _required_str(parent, "docker_image", "parent")
    if "network" in parent:
        _required_str(parent, "network", "parent")
    _optional_str(job_launcher, "default_python_path", "job_launcher")
    _optional_mapping(job_launcher, "default_job_env", "job_launcher")
    default_kwargs = _optional_mapping(job_launcher, "default_job_container_kwargs", "job_launcher")
    if default_kwargs:
        reserved = DOCKER_RESERVED_KWARGS & set(default_kwargs)
        if reserved:
            _fail(
                "INVALID_CONFIG",
                f"default_job_container_kwargs contains reserved keys: {sorted(reserved)}",
                "Remove keys controlled by DockerJobLauncher. For a site default job image, "
                "set studies.<study>.container.image in local/study_runtime.yaml.",
            )


def prepare(kit_info: KitInfo, final_output: Path, config: dict[str, Any]) -> dict[str, Any]:
    parent = config.get("parent") or {}
    job_launcher = config.get("job_launcher") or {}
    docker_image = parent["docker_image"]
    network = parent.get("network", "nvflare-network")

    launcher_path = DOCKER_SERVER_LAUNCHER if kit_info.role == ROLE_SERVER else DOCKER_CLIENT_LAUNCHER
    launcher_args = {
        "network": network,
        "default_python_path": job_launcher.get("default_python_path", "/usr/local/bin/python"),
        "default_job_container_kwargs": job_launcher.get("default_job_container_kwargs") or {},
        "default_job_env": job_launcher.get("default_job_env") or {},
    }
    _patch_resources(kit_info.kit_dir, "docker_launcher", launcher_path, launcher_args)
    if kit_info.role == ROLE_SERVER:
        _relocate_server_storage_to_workspace(kit_info.kit_dir, WORKSPACE_MOUNT_PATH)
    _patch_comm_config_for_docker(kit_info.kit_dir)
    _ensure_study_runtime_template(kit_info.kit_dir)
    _remove_start_scripts(kit_info.kit_dir, keep={DOCKER_START_SH})
    start_script = _write_docker_start_script(kit_info, docker_image=docker_image, network=network)

    final_start_script = final_output / "startup" / start_script.name
    return {
        "runtime": RUNTIME_DOCKER,
        "role": kit_info.role,
        "name": kit_info.name,
        "output": str(final_output),
        "start_command": f"cd {final_output} && ./startup/{DOCKER_START_SH}",
        "start_script": str(final_start_script),
        "resources": str(final_output / "local" / RESOURCES_JSON_DEFAULT),
    }


def _patch_comm_config_for_docker(kit_dir: Path) -> None:
    comm_config_path = kit_dir / "local" / COMM_CONFIG_JSON
    comm_config = _load_or_default_comm_config(comm_config_path)
    internal = comm_config.setdefault("internal", {})
    internal["scheme"] = "tcp"
    resources = _internal_resources(comm_config)
    resources["host"] = "0.0.0.0"
    resources.setdefault("connection_security", "clear")
    _write_json(comm_config_path, comm_config)


def _write_docker_start_script(kit_info: KitInfo, docker_image: str, network: str) -> Path:
    role_label = "server" if kit_info.role == ROLE_SERVER else "client"
    publish_port = ""
    network_alias = ""
    if kit_info.role == ROLE_SERVER:
        fed_learn_port = kit_info.fed_learn_port or 8002
        ports = [fed_learn_port]
        if kit_info.admin_port is not None and kit_info.admin_port != fed_learn_port:
            ports.append(kit_info.admin_port)
        publish_port = "".join(f"    -p {port}:{port} \\\n" for port in ports)
        network_alias = f"    --network-alias {shlex.quote(ROLE_SERVER)} \\\n"

    replacements = {
        "@@NVFLARE_ROLE_LABEL@@": role_label,
        "@@NVFLARE_SITE_NAME@@": kit_info.name,
        "@@NVFLARE_DOCKER_IMAGE@@": _bash_double_quote(docker_image),
        "@@NVFLARE_NETWORK_NAME@@": _bash_double_quote(network),
        "@@NVFLARE_CONTAINER_NAME@@": shlex.quote(kit_info.name),
        "@@NVFLARE_NETWORK_ALIAS@@": network_alias,
        "@@NVFLARE_WORKSPACE_MOUNT_PATH@@": WORKSPACE_MOUNT_PATH,
        "@@NVFLARE_PUBLISH_PORTS@@": publish_port,
        "@@NVFLARE_PARENT_COMMAND@@": _docker_parent_command(kit_info),
    }
    template = DOCKER_START_TEMPLATE.read_text(encoding="utf-8")
    script = DOCKER_TEMPLATE_TOKEN_PATTERN.sub(lambda match: replacements[match.group(0)], template)
    script_path = kit_info.kit_dir / "startup" / DOCKER_START_SH
    script_path.write_text(script, encoding="utf-8")
    script_path.chmod(script_path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return script_path


def _bash_double_quote(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"').replace("$", "\\$").replace("`", "\\`")
    return f'"{escaped}"'


def _docker_parent_command(kit_info: KitInfo) -> str:
    if kit_info.role == ROLE_SERVER:
        module = "nvflare.private.fed.app.server.server_train"
        startup_config = FED_SERVER_JSON
        set_args = ["secure_train=true", "config_folder=config", f"org={kit_info.org}"]
    else:
        module = "nvflare.private.fed.app.client.client_train"
        startup_config = FED_CLIENT_JSON
        set_args = ["secure_train=true", f"uid={kit_info.name}", f"org={kit_info.org}", "config_folder=config"]

    command = [
        "/usr/local/bin/python3",
        "-u",
        "-m",
        module,
        "-m",
        WORKSPACE_MOUNT_PATH,
        "-s",
        startup_config,
        "--set",
        *set_args,
    ]
    return " \\\n    ".join(shlex.quote(arg) for arg in command)
