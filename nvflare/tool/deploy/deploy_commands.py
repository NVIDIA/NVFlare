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

"""Implementation for ``nvflare deploy prepare``."""

from __future__ import annotations

import hashlib
import json
import re
import shlex
import shutil
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from cryptography import x509
from cryptography.hazmat.backends import default_backend
from cryptography.x509.oid import NameOID

from nvflare.tool.cli_output import output_error_message, output_ok

RUNTIME_DOCKER = "docker"
RUNTIME_K8S = "k8s"

ROLE_SERVER = "server"
ROLE_CLIENT = "client"
ROLE_ADMIN = "admin"

FED_SERVER_JSON = "fed_server.json"
FED_CLIENT_JSON = "fed_client.json"
FED_ADMIN_JSON = "fed_admin.json"

RESOURCES_JSON_DEFAULT = "resources.json.default"
RESOURCES_JSON = "resources.json"
COMM_CONFIG_JSON = "comm_config.json"
STUDY_DATA_YAML = "study_data.yaml"

HELM_CHART_DIR = "helm_chart"
START_SH = "start.sh"
SUB_START_SH = "sub_start.sh"
STOP_FL_SH = "stop_fl.sh"
LEGACY_DOCKER_SH = "docker.sh"
DOCKER_START_SH = "start_docker.sh"

WORKSPACE_MOUNT_PATH = "/var/tmp/nvflare/workspace"
WORKSPACE_VOLUME_NAME = "workspace"
HELM_TEMPLATES_DIR = Path(__file__).resolve().parent / "templates" / "helm"

PASSTHROUGH_RESOURCE_MANAGER = (
    "nvflare.app_common.resource_managers.passthrough_resource_manager.PassthroughResourceManager"
)
DOCKER_CLIENT_LAUNCHER = "nvflare.app_opt.job_launcher.docker_launcher.ClientDockerJobLauncher"
DOCKER_SERVER_LAUNCHER = "nvflare.app_opt.job_launcher.docker_launcher.ServerDockerJobLauncher"
K8S_CLIENT_LAUNCHER = "nvflare.app_opt.job_launcher.k8s_launcher.ClientK8sJobLauncher"
K8S_SERVER_LAUNCHER = "nvflare.app_opt.job_launcher.k8s_launcher.ServerK8sJobLauncher"

LAUNCHER_IDS = {"process_launcher", "docker_launcher", "k8s_launcher"}
RESOURCE_CONSUMER_IDS = {"resource_consumer"}
DOCKER_RESERVED_KWARGS = {
    "volumes",
    "mounts",
    "network",
    "environment",
    "command",
    "name",
    "detach",
    "user",
    "working_dir",
}

STUDY_DATA_TEMPLATE = """# Study data mapping used by Docker and Kubernetes job launchers.
# Example:
# default:
#   training:
#     source: /data/training    # Docker: host path; K8s: PVC claim name
#     mode: ro                  # ro or rw
{}
"""

K8S_NAME_PATTERN = re.compile(r"^[a-z]([-a-z0-9]*[a-z0-9])?$")
K8S_NAMESPACE_PATTERN = re.compile(r"^[a-z0-9]([-a-z0-9]*[a-z0-9])?$")
K8S_NAMESPACE_MAX_LENGTH = 63
K8S_SERVICE_NAME_MAX_LENGTH = 63
HELM_RELEASE_NAME_MAX_LENGTH = 53


@dataclass
class KitInfo:
    kit_dir: Path
    role: str
    name: str
    org: str
    fed_learn_port: int | None
    admin_port: int | None


def prepare_deployment(args) -> None:
    kit, output_arg, config_path = _resolve_prepare_inputs(args)

    config = _load_config(config_path)
    runtime = config["runtime"]
    _validate_runtime_config(runtime, config)
    output = _resolve_output_path(kit, output_arg, runtime)
    kit_info = _validate_kit(kit)
    if kit_info.role == ROLE_ADMIN:
        _fail(
            "UNSUPPORTED_KIT",
            "Admin startup kits are not supported by 'nvflare deploy prepare'.",
            "Use a server or client startup kit.",
        )
    if output == kit or _is_path_relative_to(kit, output):
        _fail(
            "INVALID_ARGS",
            "--output must not be the same as --kit or contain it.",
            "Choose a prepared-kit directory outside the kit, or use the default <kit>/prepared/<runtime>.",
        )
    if output.exists() and not output.is_dir():
        _fail("INVALID_ARGS", f"Output path exists and is not a directory: {output}", "Choose a directory path.")

    parent_dir = output.parent
    try:
        parent_dir.mkdir(parents=True, exist_ok=True)
    except Exception as ex:
        _fail("INVALID_ARGS", f"Cannot create output parent directory: {parent_dir}", f"Check the path: {ex}")

    temp_parent = None if _is_path_relative_to(output, kit) else str(parent_dir)
    temp_dir = Path(tempfile.mkdtemp(prefix=f".{output.name}.prepare-", dir=temp_parent))
    prepared_dir = temp_dir / output.name
    try:
        shutil.copytree(kit, prepared_dir, ignore=_ignore_output_path(kit, output))
        prepared_info = KitInfo(
            kit_dir=prepared_dir,
            role=kit_info.role,
            name=kit_info.name,
            org=kit_info.org,
            fed_learn_port=kit_info.fed_learn_port,
            admin_port=kit_info.admin_port,
        )
        if runtime == RUNTIME_DOCKER:
            result = _prepare_docker(prepared_info, output, config)
        elif runtime == RUNTIME_K8S:
            result = _prepare_k8s(prepared_info, output, config)
        else:
            _fail("INVALID_CONFIG", f"Unsupported runtime: {runtime}", "Use runtime: docker or runtime: k8s.")

        if output.exists():
            shutil.rmtree(output)
        shutil.move(str(prepared_dir), str(output))
    except SystemExit:
        raise
    except Exception as ex:
        _fail("DEPLOY_PREPARE_FAILED", f"Failed to prepare deployment: {ex}", "Check the kit and runtime config.")
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)

    output_ok(result)


def _resolve_prepare_inputs(args) -> tuple[Path, str | None, Path]:
    positional_kit = getattr(args, "kit", None)
    flag_kit = getattr(args, "kit_flag", None)
    if positional_kit and flag_kit:
        _fail("INVALID_ARGS", "Specify the startup kit only once.", "Use either positional kit or --kit.")
    kit_arg = positional_kit or flag_kit
    if not kit_arg:
        _fail("INVALID_ARGS", "Missing startup kit directory.", "Run nvflare deploy prepare <startup-kit-dir>.")

    kit = Path(kit_arg).expanduser().resolve()
    output_arg = getattr(args, "output", None)
    config_arg = getattr(args, "config", None)
    config_path = Path(config_arg).expanduser().resolve() if config_arg else kit / "config.yaml"
    return kit, output_arg, config_path


def _resolve_output_path(kit: Path, output_arg: str | None, runtime: str) -> Path:
    if output_arg:
        return Path(output_arg).expanduser().resolve()
    return kit / "prepared" / runtime


def _ignore_output_path(kit: Path, output: Path):
    if not _is_path_relative_to(output, kit):
        return None

    excluded = output.resolve()
    prepared_root = (kit / "prepared").resolve()

    def _ignore(dir_name: str, names: list[str]) -> list[str]:
        current = Path(dir_name).resolve()
        ignored = []
        for name in names:
            path = (current / name).resolve()
            if path == excluded or path == prepared_root:
                ignored.append(name)
        return ignored

    return _ignore


def _prepare_docker(kit_info: KitInfo, final_output: Path, config: dict[str, Any]) -> dict[str, Any]:
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
    _patch_comm_config_for_docker(kit_info.kit_dir)
    _ensure_study_data_template(kit_info.kit_dir)
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


def _prepare_k8s(kit_info: KitInfo, final_output: Path, config: dict[str, Any]) -> dict[str, Any]:
    namespace = config.get("namespace", "default")
    parent = config.get("parent") or {}
    job_launcher = config.get("job_launcher") or {}
    parent_port = parent.get("parent_port", 8102)
    workspace_mount_path = parent.get("workspace_mount_path", WORKSPACE_MOUNT_PATH)

    launcher_path = K8S_SERVER_LAUNCHER if kit_info.role == ROLE_SERVER else K8S_CLIENT_LAUNCHER
    launcher_args = {
        "config_file_path": job_launcher.get("config_file_path"),
        "study_data_pvc_file_path": f"{workspace_mount_path}/local/{STUDY_DATA_YAML}",
        "namespace": namespace,
        "default_python_path": job_launcher.get("default_python_path", "/usr/local/bin/python"),
    }
    if job_launcher.get("pending_timeout") is not None:
        launcher_args["pending_timeout"] = job_launcher["pending_timeout"]
    if job_launcher.get("job_pod_security_context"):
        launcher_args["security_context"] = job_launcher["job_pod_security_context"]

    _patch_resources(kit_info.kit_dir, "k8s_launcher", launcher_path, launcher_args)
    _patch_comm_config_for_k8s(kit_info.kit_dir, kit_info.role, kit_info.name, parent_port)
    _ensure_study_data_template(kit_info.kit_dir)
    if kit_info.role == ROLE_SERVER:
        _relocate_server_storage_to_workspace(kit_info.kit_dir, workspace_mount_path)
    _remove_start_scripts(kit_info.kit_dir, keep=set())
    _write_helm_chart(kit_info, config)

    release_name = _k8s_release_name(kit_info.name)
    final_chart_dir = final_output / HELM_CHART_DIR
    return {
        "runtime": RUNTIME_K8S,
        "role": kit_info.role,
        "name": kit_info.name,
        "namespace": namespace,
        "output": str(final_output),
        "helm_chart": str(final_chart_dir),
        "helm_command": f"helm upgrade --install {release_name} {final_chart_dir} --namespace {namespace}",
        "resources": str(final_output / "local" / RESOURCES_JSON_DEFAULT),
    }


def _load_config(config_path: Path) -> dict[str, Any]:
    if not config_path.is_file():
        _fail("CONFIG_NOT_FOUND", f"Config file not found: {config_path}", "Provide a YAML runtime config file.")
    try:
        with config_path.open("rt", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except Exception as ex:
        _fail("INVALID_CONFIG", f"Failed to parse config file: {ex}", "Ensure the file is valid YAML.")
    if not isinstance(config, dict):
        _fail("INVALID_CONFIG", "Runtime config must be a YAML mapping.", "Add runtime: docker or runtime: k8s.")
    runtime = config.get("runtime")
    if runtime not in {RUNTIME_DOCKER, RUNTIME_K8S}:
        _fail("INVALID_CONFIG", "Config must contain runtime: docker or runtime: k8s.", "Set a supported runtime.")
    return config


def _validate_runtime_config(runtime: str, config: dict[str, Any]) -> None:
    if runtime == RUNTIME_DOCKER:
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
                    "Remove keys controlled by DockerJobLauncher.",
                )
    elif runtime == RUNTIME_K8S:
        _validate_allowed_keys(config, {"runtime", "namespace", "parent", "job_launcher"}, "k8s config")
        parent = _mapping(config.get("parent"), "parent")
        job_launcher = _mapping(config.get("job_launcher", {}), "job_launcher")
        _validate_allowed_keys(
            parent,
            {
                "docker_image",
                "parent_port",
                "workspace_pvc",
                "workspace_mount_path",
                "resources",
                "pod_security_context",
            },
            "parent",
        )
        _validate_allowed_keys(
            job_launcher,
            {"config_file_path", "pending_timeout", "default_python_path", "job_pod_security_context"},
            "job_launcher",
        )
        _required_str(parent, "docker_image", "parent")
        if "namespace" in config:
            _validate_k8s_namespace(config, "namespace", "k8s config")
        _optional_int(parent, "parent_port", "parent")
        _optional_str(parent, "workspace_pvc", "parent")
        _optional_str(parent, "workspace_mount_path", "parent")
        _optional_mapping(parent, "resources", "parent")
        _optional_mapping(parent, "pod_security_context", "parent")
        _optional_str(job_launcher, "config_file_path", "job_launcher")
        _optional_str(job_launcher, "default_python_path", "job_launcher")
        _optional_int(job_launcher, "pending_timeout", "job_launcher")
        _optional_mapping(job_launcher, "job_pod_security_context", "job_launcher")


def _validate_kit(kit_dir: Path) -> KitInfo:
    if not kit_dir.is_dir():
        _fail("INVALID_KIT", f"Startup kit directory does not exist: {kit_dir}", "Provide an existing kit directory.")
    startup_dir = kit_dir / "startup"
    local_dir = kit_dir / "local"
    if not startup_dir.is_dir():
        _fail("INVALID_KIT", f"Missing startup directory: {startup_dir}", "Provide a valid startup kit.")
    if not local_dir.is_dir():
        _fail("INVALID_KIT", f"Missing local directory: {local_dir}", "Provide a valid startup kit.")

    role_files = {
        ROLE_SERVER: startup_dir / FED_SERVER_JSON,
        ROLE_CLIENT: startup_dir / FED_CLIENT_JSON,
        ROLE_ADMIN: startup_dir / FED_ADMIN_JSON,
    }
    roles = [role for role, path in role_files.items() if path.is_file()]
    if len(roles) != 1:
        _fail(
            "INVALID_KIT",
            f"Expected exactly one role file in startup/, found: {roles or 'none'}",
            "A kit should contain one of fed_server.json, fed_client.json, or fed_admin.json.",
        )
    role = roles[0]

    resources_path = local_dir / RESOURCES_JSON_DEFAULT
    _load_json_file(resources_path, "resources.json.default")
    role_config = _load_json_file(role_files[role], role_files[role].name)
    if role != ROLE_ADMIN:
        _validate_identity_files(startup_dir, role)

    name = _detect_name(kit_dir, role, role_config)
    org = _detect_org(startup_dir, role) if role != ROLE_ADMIN else name
    fed_learn_port, admin_port = _detect_ports(role, role_config)
    return KitInfo(kit_dir=kit_dir, role=role, name=name, org=org, fed_learn_port=fed_learn_port, admin_port=admin_port)


def _validate_identity_files(startup_dir: Path, role: str) -> None:
    required = ["rootCA.pem"]
    if role == ROLE_SERVER:
        required.extend(["server.crt", "server.key"])
    else:
        required.extend(["client.crt", "client.key"])
    missing = [name for name in required if not (startup_dir / name).is_file()]
    if missing:
        _fail(
            "INVALID_KIT",
            f"Missing identity material in {startup_dir}: {missing}",
            "Use a fully provisioned or packaged startup kit.",
        )


def _detect_name(kit_dir: Path, role: str, role_config: dict[str, Any]) -> str:
    if role == ROLE_CLIENT:
        client = role_config.get("client") or {}
        return str(client.get("fqsn") or kit_dir.name)
    if role == ROLE_SERVER:
        server = (role_config.get("servers") or [{}])[0]
        return str(server.get("identity") or server.get("admin_server") or kit_dir.name)
    admin = role_config.get("admin") or {}
    return str(admin.get("name") or kit_dir.name)


def _detect_org(startup_dir: Path, role: str) -> str:
    cert_name = "server.crt" if role == ROLE_SERVER else "client.crt"
    cert_path = startup_dir / cert_name
    try:
        cert = x509.load_pem_x509_certificate(cert_path.read_bytes(), default_backend())
        org_attrs = cert.subject.get_attributes_for_oid(NameOID.ORGANIZATION_NAME)
    except Exception as ex:
        _fail("INVALID_KIT", f"Failed to parse {cert_name}: {ex}", "Use a valid provisioned startup kit.")
    if not org_attrs or not org_attrs[0].value:
        _fail("INVALID_KIT", f"Missing organization in {cert_name}.", "Use a valid provisioned startup kit.")
    return org_attrs[0].value


def _detect_ports(role: str, role_config: dict[str, Any]) -> tuple[int | None, int | None]:
    if role != ROLE_SERVER:
        return None, None
    server = (role_config.get("servers") or [{}])[0]
    fed_port = _port_from_target((server.get("service") or {}).get("target"))
    admin_port = server.get("admin_port")
    if isinstance(admin_port, int):
        return fed_port, admin_port
    return fed_port, fed_port


def _port_from_target(target: str | None) -> int | None:
    if not target or ":" not in target:
        return None
    try:
        return int(str(target).rsplit(":", 1)[-1])
    except ValueError:
        return None


def _patch_resources(kit_dir: Path, launcher_id: str, launcher_path: str, launcher_args: dict[str, Any]) -> None:
    local_dir = kit_dir / "local"
    default_path = local_dir / RESOURCES_JSON_DEFAULT
    resources = _load_json_file(default_path, RESOURCES_JSON_DEFAULT)
    components = resources.setdefault("components", [])
    if not isinstance(components, list):
        _fail("INVALID_KIT", "resources.json.default components must be a list.", "Fix the startup kit resources file.")

    components[:] = [
        c for c in components if c.get("id") not in LAUNCHER_IDS and c.get("id") not in RESOURCE_CONSUMER_IDS
    ]
    components[:] = [c for c in components if c.get("id") != "resource_manager"]
    components.append(
        {
            "id": "resource_manager",
            "path": PASSTHROUGH_RESOURCE_MANAGER,
            "args": {},
        }
    )
    components.append({"id": launcher_id, "path": launcher_path, "args": launcher_args})
    _write_resources(local_dir, resources)


def _write_resources(local_dir: Path, resources: dict[str, Any]) -> None:
    payload = json.dumps(resources, indent=4)
    (local_dir / RESOURCES_JSON_DEFAULT).write_text(payload + "\n", encoding="utf-8")
    active_resources = local_dir / RESOURCES_JSON
    if active_resources.exists():
        active_resources.unlink()


def _patch_comm_config_for_docker(kit_dir: Path) -> None:
    comm_config_path = kit_dir / "local" / COMM_CONFIG_JSON
    comm_config = _load_or_default_comm_config(comm_config_path)
    internal = comm_config.setdefault("internal", {})
    internal["scheme"] = "tcp"
    resources = _internal_resources(comm_config)
    resources["host"] = "0.0.0.0"
    resources.setdefault("connection_security", "clear")
    _write_json(comm_config_path, comm_config)


def _patch_comm_config_for_k8s(kit_dir: Path, role: str, site_name: str, parent_port: int) -> None:
    comm_config_path = kit_dir / "local" / COMM_CONFIG_JSON
    comm_config = _load_or_default_comm_config(comm_config_path)
    resources = _internal_resources(comm_config)
    resources.update(
        {
            "host": _k8s_parent_service_name(role, site_name),
            "port": parent_port,
            "connection_security": "clear",
        }
    )
    internal = comm_config.setdefault("internal", {})
    internal["scheme"] = "tcp"
    _write_json(comm_config_path, comm_config)


def _load_or_default_comm_config(comm_config_path: Path) -> dict[str, Any]:
    if comm_config_path.is_file():
        return _load_json_file(comm_config_path, COMM_CONFIG_JSON)
    return {
        "allow_adhoc_conns": False,
        "backbone_conn_gen": 2,
        "internal": {"scheme": "tcp", "resources": {"connection_security": "clear"}},
    }


def _internal_resources(comm_config: dict[str, Any]) -> dict[str, Any]:
    internal = comm_config.setdefault("internal", {})
    resources = internal.setdefault("resources", {})
    if not isinstance(resources, dict):
        _fail(
            "INVALID_KIT",
            "comm_config.json internal.resources must be a mapping.",
            "Fix the startup kit comm config.",
        )
    return resources


def _ensure_study_data_template(kit_dir: Path) -> None:
    path = kit_dir / "local" / STUDY_DATA_YAML
    if not path.exists():
        path.write_text(STUDY_DATA_TEMPLATE, encoding="utf-8")


def _remove_start_scripts(kit_dir: Path, keep: set[str]) -> None:
    startup_dir = kit_dir / "startup"
    for filename in (START_SH, SUB_START_SH, STOP_FL_SH, LEGACY_DOCKER_SH, DOCKER_START_SH):
        if filename in keep:
            continue
        path = startup_dir / filename
        if path.exists():
            path.unlink()


def _k8s_parent_service_name(role: str, site_name: str) -> str:
    if role == ROLE_SERVER:
        return "nvflare-server"
    return _k8s_service_name(site_name)


def _k8s_service_name(site_name: str) -> str:
    return _stable_k8s_name(site_name, K8S_SERVICE_NAME_MAX_LENGTH)


def _stable_k8s_name(raw_name: str, max_length: int) -> str:
    if len(raw_name) <= max_length and K8S_NAME_PATTERN.match(raw_name):
        return raw_name

    normalized = re.sub(r"[^a-z0-9-]", "-", raw_name.lower())
    normalized = re.sub(r"-+", "-", normalized).strip("-")
    if not normalized:
        normalized = "site"
    if not normalized[0].isalpha():
        normalized = f"site-{normalized}"

    digest = hashlib.sha256(raw_name.encode("utf-8")).hexdigest()[:8]
    head_max = max_length - len(digest) - 1
    head = normalized[:head_max].rstrip("-") or "site"
    return f"{head}-{digest}"


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

    script = f"""#!/usr/bin/env bash
# Generated by NVFlare deploy prepare - Docker runtime {role_label} startup script.
# The parent {role_label} container mounts this prepared kit and launches job containers through Docker.
DIR="$( cd "$( dirname "${{BASH_SOURCE[0]}}" )" >/dev/null 2>&1 && pwd )"
HOST_WORKSPACE="$(cd "$DIR/.." && pwd)"
DOCKER_IMAGE=${{NVFL_P_IMAGE:-{_bash_double_quote(docker_image)}}}

if ! docker info > /dev/null 2>&1; then
    echo "ERROR: cannot connect to Docker daemon. Make sure your user has permission to run docker."
    echo "  Option 1 (docker group): sudo usermod -aG docker \\$USER  then log out and back in."
    echo "  Option 2 (rootless Docker): https://docs.docker.com/engine/security/rootless/"
    exit 1
fi

echo "Starting NVFlare {role_label} {kit_info.name} with image $DOCKER_IMAGE"
echo "Host workspace: $HOST_WORKSPACE"

NETWORK_NAME={_bash_double_quote(network)}
if ! docker network ls --filter name="$NETWORK_NAME" --format "{{{{.Name}}}}" | grep -wq "$NETWORK_NAME"; then
    docker network create "$NETWORK_NAME"
    echo "Created Docker network: $NETWORK_NAME"
fi

rm -f "$HOST_WORKSPACE/daemon_pid.fl"

SOCK_GID=$(stat -c '%g' /var/run/docker.sock 2>/dev/null || stat -f '%g' /var/run/docker.sock 2>/dev/null || echo "")
GROUP_ADD_ARG=""
if [ -n "$SOCK_GID" ] && [ "$SOCK_GID" != "0" ]; then
    GROUP_ADD_ARG="--group-add $SOCK_GID"
fi

docker run --name {shlex.quote(kit_info.name)} \\
    --user "$(id -u):$(id -g)" \\
    $GROUP_ADD_ARG \\
    --network "$NETWORK_NAME" \\
{network_alias}    -v "$HOST_WORKSPACE":{WORKSPACE_MOUNT_PATH} \\
    -v /var/run/docker.sock:/var/run/docker.sock \\
    -e NVFL_DOCKER_WORKSPACE="$HOST_WORKSPACE" \\
{publish_port}    --rm "$DOCKER_IMAGE" \\
    {_docker_parent_command(kit_info)}
"""
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


def _write_helm_chart(kit_info: KitInfo, config: dict[str, Any]) -> Path:
    parent = config.get("parent") or {}
    docker_image = parent["docker_image"]
    parent_port = parent.get("parent_port", 8102)
    workspace_pvc = parent.get("workspace_pvc", "nvflws")
    workspace_mount_path = parent.get("workspace_mount_path", WORKSPACE_MOUNT_PATH)
    chart_dir = kit_info.kit_dir / HELM_CHART_DIR
    if chart_dir.exists():
        shutil.rmtree(chart_dir)
    templates_dir = chart_dir / "templates"
    templates_dir.mkdir(parents=True)

    if kit_info.role == ROLE_SERVER:
        _write_server_helm_chart(
            kit_info,
            chart_dir,
            docker_image,
            parent_port,
            workspace_pvc,
            workspace_mount_path,
            parent,
        )
        _copy_helm_templates("server", templates_dir)
    else:
        _write_client_helm_chart(
            kit_info,
            chart_dir,
            docker_image,
            parent_port,
            workspace_pvc,
            workspace_mount_path,
            parent,
        )
        _copy_helm_templates("client", templates_dir)
    return chart_dir


def _write_server_helm_chart(
    kit_info: KitInfo,
    chart_dir: Path,
    docker_image: str,
    parent_port: int,
    workspace_pvc: str,
    workspace_mount_path: str,
    parent: dict[str, Any],
) -> None:
    repo, tag = _split_image(docker_image)
    _write_yaml(
        chart_dir / "Chart.yaml",
        {
            "apiVersion": "v2",
            "name": "nvflare-server",
            "description": f"NVFlare federated learning server for {kit_info.name}",
            "type": "application",
            "version": "0.1.0",
            "appVersion": tag or "latest",
            "keywords": ["nvflare", "federated-learning"],
            "maintainers": [],
        },
    )
    fed_learn_port = kit_info.fed_learn_port or 8002
    admin_port = kit_info.admin_port if kit_info.admin_port != fed_learn_port else None
    values = {
        "name": kit_info.name,
        "serviceName": _k8s_parent_service_name(kit_info.role, kit_info.name),
        "image": {"repository": repo, "tag": tag, "pullPolicy": "IfNotPresent"},
        "serviceAccount": {"create": True, "annotations": {}, "automountServiceAccountToken": True},
        "podAnnotations": {},
        "rbac": {"create": True},
        "persistence": {
            "workspace": {
                "claimName": workspace_pvc,
                "volumeName": WORKSPACE_VOLUME_NAME,
                "mountPath": workspace_mount_path,
            }
        },
        "fedLearnPort": fed_learn_port,
        "adminPort": admin_port,
        "parentPort": parent_port,
        "resources": parent.get("resources") or {"requests": {"cpu": "2", "memory": "8Gi"}},
        "securityContext": parent.get("pod_security_context") or {},
        "hostPortEnabled": False,
        "tcpConfigMapEnabled": False,
        "service": {"type": "ClusterIP", "loadBalancerIP": None, "annotations": {}},
        "command": ["/usr/local/bin/python3"],
        "args": [
            "-u",
            "-m",
            "nvflare.private.fed.app.server.server_train",
            "-m",
            workspace_mount_path,
            "-s",
            FED_SERVER_JSON,
            "--set",
            "secure_train=true",
            "config_folder=config",
            f"org={kit_info.org}",
        ],
    }
    _write_yaml(chart_dir / "values.yaml", values)


def _write_client_helm_chart(
    kit_info: KitInfo,
    chart_dir: Path,
    docker_image: str,
    parent_port: int,
    workspace_pvc: str,
    workspace_mount_path: str,
    parent: dict[str, Any],
) -> None:
    repo, tag = _split_image(docker_image)
    _write_yaml(
        chart_dir / "Chart.yaml",
        {
            "apiVersion": "v2",
            "name": "nvflare-client",
            "description": f"NVFlare federated learning client deployment and service for {kit_info.name}",
            "type": "application",
            "version": "0.1.0",
            "appVersion": tag or "latest",
            "keywords": ["nvflare", "federated-learning"],
            "maintainers": [],
        },
    )
    values = {
        "name": kit_info.name,
        "siteName": kit_info.name,
        "serviceName": _k8s_parent_service_name(kit_info.role, kit_info.name),
        "image": {"repository": repo, "tag": tag, "pullPolicy": "Always"},
        "serviceAccount": {"create": True, "annotations": {}, "automountServiceAccountToken": True},
        "podAnnotations": {},
        "rbac": {"create": True},
        "persistence": {
            "workspace": {
                "claimName": workspace_pvc,
                "volumeName": WORKSPACE_VOLUME_NAME,
                "mountPath": workspace_mount_path,
            }
        },
        "port": parent_port,
        "service": {"annotations": {}},
        "securityContext": parent.get("pod_security_context") or {},
        "resources": parent.get("resources") or {"requests": {"cpu": "2", "memory": "8Gi"}},
        "command": ["/usr/local/bin/python3"],
        "args": [
            "-u",
            "-m",
            "nvflare.private.fed.app.client.client_train",
            "-m",
            workspace_mount_path,
            "-s",
            FED_CLIENT_JSON,
            "--set",
            "secure_train=true",
            "config_folder=config",
            f"org={kit_info.org}",
        ],
    }
    _write_yaml(chart_dir / "values.yaml", values)


def _copy_helm_templates(role: str, templates_dir: Path) -> None:
    files = {
        "server": [
            ("_helpers.tpl", "_helpers.tpl"),
            ("deployment.yaml", "server-deployment.yaml"),
            ("service.yaml", "server-service.yaml"),
            ("tcp-services.yaml", "server-tcp-services.yaml"),
            ("serviceaccount.yaml", "serviceaccount.yaml"),
            ("role.yaml", "role.yaml"),
        ],
        "client": [
            ("_helpers.tpl", "_helpers.tpl"),
            ("deployment.yaml", "client-deployment.yaml"),
            ("service.yaml", "service.yaml"),
            ("serviceaccount.yaml", "serviceaccount.yaml"),
            ("role.yaml", "role.yaml"),
        ],
    }
    for src_name, dst_name in files[role]:
        shutil.copy(_helm_src(role, src_name), templates_dir / dst_name)


def _split_image(docker_image: str) -> tuple[str, str]:
    if ":" in docker_image:
        return docker_image.rsplit(":", 1)
    return docker_image, ""


def _helm_src(role: str, filename: str) -> Path:
    return HELM_TEMPLATES_DIR / role / filename


def _relocate_server_storage_to_workspace(kit_dir: Path, workspace_mount_path: str) -> None:
    local_dir = kit_dir / "local"
    resources = _load_json_file(local_dir / RESOURCES_JSON_DEFAULT, RESOURCES_JSON_DEFAULT)
    try:
        resources["snapshot_persistor"]["args"]["storage"]["args"][
            "root_dir"
        ] = f"{workspace_mount_path}/snapshot-storage"
    except KeyError:
        pass
    for component in resources.get("components", []):
        if component.get("id") == "job_manager":
            component.setdefault("args", {})["uri_root"] = f"{workspace_mount_path}/jobs-storage"
    _write_resources(local_dir, resources)


def _k8s_release_name(name: str) -> str:
    return _stable_k8s_name(name, HELM_RELEASE_NAME_MAX_LENGTH)


def _is_path_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail("INVALID_CONFIG", f"{name} must be a YAML mapping.", "Fix the runtime config.")
    return value


def _validate_allowed_keys(data: dict[str, Any], allowed: set[str], where: str) -> None:
    unknown = sorted(set(data) - allowed)
    if unknown:
        _fail("INVALID_CONFIG", f"Unknown keys in {where}: {unknown}", "Remove or rename unsupported keys.")


def _required_str(data: dict[str, Any], key: str, where: str) -> None:
    if not isinstance(data.get(key), str) or not data.get(key):
        _fail("INVALID_CONFIG", f"{where}.{key} must be a non-empty string.", "Fix the runtime config.")


def _validate_k8s_namespace(data: dict[str, Any], key: str, where: str) -> None:
    namespace = data.get(key)
    if not isinstance(namespace, str) or not namespace:
        _fail("INVALID_CONFIG", f"{where}.{key} must be a non-empty string.", "Fix the runtime config.")
        return
    if len(namespace) > K8S_NAMESPACE_MAX_LENGTH or not K8S_NAMESPACE_PATTERN.match(namespace):
        _fail(
            "INVALID_CONFIG",
            f"{where}.{key} must be a valid Kubernetes namespace (DNS-1123 label): {namespace!r}.",
            "Use lower case alphanumeric characters or '-', start and end with an alphanumeric character, "
            f"and keep length <= {K8S_NAMESPACE_MAX_LENGTH}.",
        )


def _optional_str(data: dict[str, Any], key: str, where: str) -> str | None:
    if key not in data or data[key] is None:
        return None
    if not isinstance(data[key], str) or not data[key]:
        _fail("INVALID_CONFIG", f"{where}.{key} must be a non-empty string.", "Fix the runtime config.")
    return data[key]


def _optional_mapping(data: dict[str, Any], key: str, where: str) -> dict[str, Any] | None:
    if key not in data or data[key] is None:
        return None
    if not isinstance(data[key], dict):
        _fail("INVALID_CONFIG", f"{where}.{key} must be a mapping.", "Fix the runtime config.")
    return data[key]


def _optional_int(data: dict[str, Any], key: str, where: str) -> int | None:
    if key not in data or data[key] is None:
        return None
    if not isinstance(data[key], int):
        _fail("INVALID_CONFIG", f"{where}.{key} must be an integer.", "Fix the runtime config.")
    return data[key]


def _load_json_file(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        _fail("INVALID_KIT", f"Missing {label}: {path}", "Provide a valid startup kit.")
    try:
        with path.open("rt", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as ex:
        _fail("INVALID_KIT", f"Failed to parse {label}: {ex}", "Fix the startup kit JSON.")
    if not isinstance(data, dict):
        _fail("INVALID_KIT", f"{label} must contain a JSON object.", "Fix the startup kit JSON.")
    return data


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=4) + "\n", encoding="utf-8")


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(data, default_flow_style=False, sort_keys=False), encoding="utf-8")


def _fail(error_code: str, message: str, hint: str = "") -> None:
    output_error_message(error_code, message, hint, None, exit_code=4)
