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

"""Helpers shared by the Docker and Kubernetes deploy backends."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cryptography import x509
from cryptography.hazmat.backends import default_backend
from cryptography.x509.oid import NameOID

from nvflare.tool.cli_output import output_error_message, print_human

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
STUDY_RUNTIME_YAML = "study_runtime.yaml"
START_SH = "start.sh"
SUB_START_SH = "sub_start.sh"
STOP_FL_SH = "stop_fl.sh"
LEGACY_DOCKER_SH = "docker.sh"
DOCKER_START_SH = "start_docker.sh"
WORKSPACE_MOUNT_PATH = "/var/tmp/nvflare/workspace"
PASSTHROUGH_RESOURCE_MANAGER = (
    "nvflare.app_common.resource_managers.passthrough_resource_manager.PassthroughResourceManager"
)
GPU_RESOURCE_MANAGER = "nvflare.app_common.resource_managers.gpu_resource_manager.GPUResourceManager"
DOCKER_CLIENT_LAUNCHER = "nvflare.app_opt.job_launcher.docker_launcher.ClientDockerJobLauncher"
DOCKER_SERVER_LAUNCHER = "nvflare.app_opt.job_launcher.docker_launcher.ServerDockerJobLauncher"
K8S_CLIENT_LAUNCHER = "nvflare.app_opt.job_launcher.k8s_launcher.ClientK8sJobLauncher"
K8S_SERVER_LAUNCHER = "nvflare.app_opt.job_launcher.k8s_launcher.ServerK8sJobLauncher"
PROCESS_CLIENT_LAUNCHER = "nvflare.app_common.job_launcher.client_process_launcher.ClientProcessJobLauncher"
PROCESS_SERVER_LAUNCHER = "nvflare.app_common.job_launcher.server_process_launcher.ServerProcessJobLauncher"
GPU_RESOURCE_CONSUMER = "nvflare.app_common.resource_consumers.gpu_resource_consumer.GPUResourceConsumer"
LAUNCHER_IDS = {"process_launcher", "docker_launcher", "k8s_launcher"}
BUILTIN_LAUNCHER_PATHS = {
    DOCKER_CLIENT_LAUNCHER,
    DOCKER_SERVER_LAUNCHER,
    K8S_CLIENT_LAUNCHER,
    K8S_SERVER_LAUNCHER,
    PROCESS_CLIENT_LAUNCHER,
    PROCESS_SERVER_LAUNCHER,
}
BUILTIN_RESOURCE_MANAGER_PATHS = {GPU_RESOURCE_MANAGER, PASSTHROUGH_RESOURCE_MANAGER}
BUILTIN_RESOURCE_CONSUMER_PATHS = {GPU_RESOURCE_CONSUMER}
RESOURCE_CONSUMER_IDS = {"resource_consumer"}
_STUDY_RUNTIME_TEMPLATE_PATH = Path(__file__).resolve().parent / "templates" / STUDY_RUNTIME_YAML


@dataclass
class KitInfo:
    kit_dir: Path
    role: str
    name: str
    org: str
    fed_learn_port: int | None
    admin_port: int | None


def validate_kit(kit_dir: Path) -> KitInfo:
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

    _warn_for_replaced_components(components, launcher_id, launcher_path)
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


def _warn_for_replaced_components(components: list[dict[str, Any]], launcher_id: str, launcher_path: str) -> None:
    for component in components:
        component_id = component.get("id")
        if component_id in RESOURCE_CONSUMER_IDS and _component_has_custom_config(
            component, BUILTIN_RESOURCE_CONSUMER_PATHS
        ):
            _warn(
                f"deploy prepare removes component '{component_id}' from resources.json.default; "
                "existing resource consumer configuration will not be used by the prepared runtime."
            )
        elif component_id == "resource_manager" and _component_has_custom_config(
            component, BUILTIN_RESOURCE_MANAGER_PATHS
        ):
            _warn(
                "deploy prepare replaces component 'resource_manager' with PassthroughResourceManager; "
                "existing resource manager path/args will not be used by the prepared runtime."
            )
        elif component_id in LAUNCHER_IDS and _launcher_replacement_discards_config(
            component, launcher_id, launcher_path
        ):
            _warn(
                f"deploy prepare replaces component '{component_id}' with generated '{launcher_id}' configuration; "
                "existing launcher path/args will not be used by the prepared runtime."
            )


def _component_has_custom_config(component: dict[str, Any], builtin_paths: set[str]) -> bool:
    args = component.get("args")
    # Empty args are the canonical "no arguments" shape in generated resources files.
    if args:
        return True
    path = component.get("path")
    return bool(path) and path not in builtin_paths


def _launcher_replacement_discards_config(component: dict[str, Any], launcher_id: str, launcher_path: str) -> bool:
    if component.get("args"):
        return True
    path = component.get("path")
    if not path:
        return False
    if component.get("id") == launcher_id and path != launcher_path:
        return True
    return path not in BUILTIN_LAUNCHER_PATHS


def _warn(message: str) -> None:
    print_human(f"Warning: {message}")


def _write_resources(local_dir: Path, resources: dict[str, Any]) -> None:
    payload = json.dumps(resources, indent=4)
    (local_dir / RESOURCES_JSON_DEFAULT).write_text(payload + "\n", encoding="utf-8")
    active_resources = local_dir / RESOURCES_JSON
    if active_resources.exists():
        active_resources.unlink()


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


def _ensure_study_runtime_template(kit_dir: Path) -> None:
    local_dir = kit_dir / "local"
    if (local_dir / STUDY_RUNTIME_YAML).exists():
        return
    if (local_dir / STUDY_DATA_YAML).exists():
        # legacy v1 kit: study_data.yaml and study_runtime.yaml must not coexist
        return
    (local_dir / STUDY_RUNTIME_YAML).write_bytes(_STUDY_RUNTIME_TEMPLATE_PATH.read_bytes())


def _remove_start_scripts(kit_dir: Path, keep: set[str]) -> None:
    startup_dir = kit_dir / "startup"
    for filename in (START_SH, SUB_START_SH, STOP_FL_SH, LEGACY_DOCKER_SH, DOCKER_START_SH):
        if filename in keep:
            continue
        path = startup_dir / filename
        if path.exists():
            path.unlink()


def _relocate_server_storage_to_workspace(kit_dir: Path, workspace_mount_path: str) -> None:
    local_dir = kit_dir / "local"
    resources = _load_json_file(local_dir / RESOURCES_JSON_DEFAULT, RESOURCES_JSON_DEFAULT)
    if "snapshot_persistor" in resources:
        try:
            resources["snapshot_persistor"]["args"]["storage"]["args"][
                "root_dir"
            ] = f"{workspace_mount_path}/snapshot-storage"
        except (KeyError, TypeError):
            _warn(
                "snapshot_persistor is present, but deploy prepare could not relocate snapshot storage to the "
                f"workspace at {workspace_mount_path}/snapshot-storage. Expected nested key: "
                "snapshot_persistor.args.storage.args.root_dir."
            )
    for component in resources.get("components", []):
        if component.get("id") == "job_manager":
            component.setdefault("args", {})["uri_root"] = f"{workspace_mount_path}/jobs-storage"
    _write_resources(local_dir, resources)


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


def _fail(error_code: str, message: str, hint: str = "") -> None:
    output_error_message(error_code, message, hint, None, exit_code=4)
