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

"""Kubernetes-specific validation and preparation for ``nvflare deploy``."""

from __future__ import annotations

import hashlib
import re
import shutil
from pathlib import Path
from typing import Any

import yaml

from nvflare.tool.deploy.deploy_common import (
    COMM_CONFIG_JSON,
    FED_CLIENT_JSON,
    FED_SERVER_JSON,
    K8S_CLIENT_LAUNCHER,
    K8S_SERVER_LAUNCHER,
    RESOURCES_JSON_DEFAULT,
    ROLE_SERVER,
    RUNTIME_K8S,
    STUDY_DATA_YAML,
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

HELM_CHART_DIR = "helm_chart"
WORKSPACE_VOLUME_NAME = "workspace"
K8S_PARENT_PYTHON_PATH = "/usr/local/bin/python3"
HELM_TEMPLATES_DIR = Path(__file__).resolve().parent / "templates" / "helm"
K8S_LAUNCHER_PATHS = {K8S_CLIENT_LAUNCHER, K8S_SERVER_LAUNCHER}
K8S_NAME_PATTERN = re.compile(r"^[a-z]([-a-z0-9]*[a-z0-9])?$")
K8S_NAMESPACE_PATTERN = re.compile(r"^[a-z0-9]([-a-z0-9]*[a-z0-9])?$")
K8S_SECRET_NAME_PATTERN = re.compile(r"^[a-z0-9]([-a-z0-9]*[a-z0-9])?(\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*$")
K8S_NAMESPACE_MAX_LENGTH = 63
K8S_SERVICE_NAME_MAX_LENGTH = 63
K8S_SECRET_NAME_MAX_LENGTH = 253
HELM_RELEASE_NAME_MAX_LENGTH = 53
DEFAULT_K8S_SERVER_SERVICE_NAME = "nvflare-server"


def validate_config(config: dict[str, Any]) -> None:
    _validate_allowed_keys(
        config,
        {"runtime", "namespace", "server_service_name", "parent", "job_launcher"},
        "k8s config",
    )
    parent = _mapping(config.get("parent"), "parent")
    job_launcher = _mapping(config.get("job_launcher", {}), "job_launcher")
    _validate_allowed_keys(
        parent,
        {
            "docker_image",
            "parent_port",
            "workspace_pvc",
            "workspace_mount_path",
            "python_path",
            "resources",
            "pod_security_context",
            "image_pull_secrets",
        },
        "parent",
    )
    _validate_allowed_keys(
        job_launcher,
        {
            "config_file_path",
            "pending_timeout",
            "default_python_path",
            "job_pod_security_context",
            "image_pull_secrets",
        },
        "job_launcher",
    )
    _required_str(parent, "docker_image", "parent")
    if "namespace" in config:
        _validate_k8s_namespace(config, "namespace", "k8s config")
    if "server_service_name" in config:
        _validate_k8s_service_name(config, "server_service_name", "k8s config")
    _optional_int(parent, "parent_port", "parent")
    _optional_str(parent, "workspace_pvc", "parent")
    _optional_str(parent, "workspace_mount_path", "parent")
    _optional_str(parent, "python_path", "parent")
    _optional_mapping(parent, "resources", "parent")
    _optional_mapping(parent, "pod_security_context", "parent")
    _optional_k8s_secret_name_list(parent, "image_pull_secrets", "parent image pull references")
    _optional_str(job_launcher, "config_file_path", "job_launcher")
    _optional_str(job_launcher, "default_python_path", "job_launcher")
    _optional_non_negative_int(job_launcher, "pending_timeout", "job_launcher")
    _optional_mapping(job_launcher, "job_pod_security_context", "job_launcher")
    _optional_k8s_secret_name_list(job_launcher, "image_pull_secrets", "job launcher image pull references")


def prepare(kit_info: KitInfo, final_output: Path, config: dict[str, Any]) -> dict[str, Any]:
    namespace = config.get("namespace", "default")
    parent = config.get("parent") or {}
    job_launcher = config.get("job_launcher") or {}
    parent_port = parent.get("parent_port", 8102)
    workspace_mount_path = parent.get("workspace_mount_path", WORKSPACE_MOUNT_PATH)
    server_service_name = config.get("server_service_name", DEFAULT_K8S_SERVER_SERVICE_NAME)

    launcher_path = K8S_SERVER_LAUNCHER if kit_info.role == ROLE_SERVER else K8S_CLIENT_LAUNCHER
    launcher_args = {
        "config_file_path": job_launcher.get("config_file_path"),
        "namespace": namespace,
        "default_python_path": job_launcher.get("default_python_path", K8S_PARENT_PYTHON_PATH),
        "workspace_mount_path": workspace_mount_path,
    }
    if (kit_info.kit_dir / "local" / STUDY_DATA_YAML).exists():
        # legacy v1 kit: keep the study data mounts working; new kits use study_runtime.yaml
        launcher_args["study_data_pvc_file_path"] = f"{workspace_mount_path}/local/{STUDY_DATA_YAML}"
    if "pending_timeout" in job_launcher:
        launcher_args["pending_timeout"] = job_launcher["pending_timeout"]
    if job_launcher.get("job_pod_security_context"):
        launcher_args["security_context"] = job_launcher["job_pod_security_context"]
    if job_launcher.get("image_pull_secrets") is not None:
        launcher_args["image_pull_secrets"] = job_launcher["image_pull_secrets"]

    _patch_resources(kit_info.kit_dir, "k8s_launcher", launcher_path, launcher_args)
    _patch_comm_config_for_k8s(kit_info.kit_dir, kit_info.role, kit_info.name, parent_port, server_service_name)
    _ensure_study_runtime_template(kit_info.kit_dir)
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


def _patch_comm_config_for_k8s(
    kit_dir: Path,
    role: str,
    site_name: str,
    parent_port: int,
    server_service_name: str = DEFAULT_K8S_SERVER_SERVICE_NAME,
) -> None:
    comm_config_path = kit_dir / "local" / COMM_CONFIG_JSON
    comm_config = _load_or_default_comm_config(comm_config_path)
    resources = _internal_resources(comm_config)
    resources.update(
        {
            "host": _k8s_parent_service_name(role, site_name, server_service_name),
            "port": parent_port,
            "connection_security": "clear",
        }
    )
    internal = comm_config.setdefault("internal", {})
    internal["scheme"] = "tcp"
    _write_json(comm_config_path, comm_config)


def _k8s_parent_service_name(
    role: str, site_name: str, server_service_name: str = DEFAULT_K8S_SERVER_SERVICE_NAME
) -> str:
    if role == ROLE_SERVER:
        return server_service_name
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


def _write_helm_chart(kit_info: KitInfo, config: dict[str, Any]) -> Path:
    parent = config.get("parent") or {}
    docker_image = parent["docker_image"]
    parent_port = parent.get("parent_port", 8102)
    workspace_pvc = parent.get("workspace_pvc", "nvflws")
    workspace_mount_path = parent.get("workspace_mount_path", WORKSPACE_MOUNT_PATH)
    parent_python_path = parent.get("python_path") or K8S_PARENT_PYTHON_PATH
    server_service_name = config.get("server_service_name", DEFAULT_K8S_SERVER_SERVICE_NAME)
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
            parent_python_path,
            parent,
            server_service_name,
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
            parent_python_path,
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
    parent_python_path: str,
    parent: dict[str, Any],
    server_service_name: str,
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
        "serviceName": server_service_name,
        "image": {"repository": repo, "tag": tag, "pullPolicy": "IfNotPresent"},
        "imagePullSecrets": _image_pull_secret_refs(parent),
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
        "workspaceConfig": {
            "namespace": None,
            "local": {"configMapName": None, "items": []},
            "startup": {"secretName": None, "items": []},
        },
        "fedLearnPort": fed_learn_port,
        "adminPort": admin_port,
        "parentPort": parent_port,
        "resources": parent.get("resources") or {"requests": {"cpu": "2", "memory": "8Gi"}},
        "securityContext": parent.get("pod_security_context") or {},
        "hostPortEnabled": False,
        "tcpConfigMapEnabled": False,
        "service": {"type": "ClusterIP", "loadBalancerIP": None, "annotations": {}},
        "command": [parent_python_path],
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
    parent_python_path: str,
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
        "imagePullSecrets": _image_pull_secret_refs(parent),
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
        "workspaceConfig": {
            "namespace": None,
            "local": {"configMapName": None, "items": []},
            "startup": {"secretName": None, "items": []},
        },
        "port": parent_port,
        "service": {"annotations": {}},
        "securityContext": parent.get("pod_security_context") or {},
        "resources": parent.get("resources") or {"requests": {"cpu": "2", "memory": "8Gi"}},
        "command": [parent_python_path],
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


def _image_pull_secret_refs(parent: dict[str, Any]) -> list[dict[str, str]]:
    return [{"name": name} for name in parent.get("image_pull_secrets") or []]


def _helm_src(role: str, filename: str) -> Path:
    return HELM_TEMPLATES_DIR / role / filename


def _k8s_release_name(name: str) -> str:
    return _stable_k8s_name(name, HELM_RELEASE_NAME_MAX_LENGTH)


def _validate_k8s_namespace(
    data: dict[str, Any],
    key: str,
    where: str,
    error_code: str = "INVALID_CONFIG",
    hint: str = "Fix the runtime config.",
) -> None:
    namespace = data.get(key)
    if not isinstance(namespace, str) or not namespace:
        _fail(error_code, f"{where}.{key} must be a non-empty string.", hint)
        return
    if len(namespace) > K8S_NAMESPACE_MAX_LENGTH or not K8S_NAMESPACE_PATTERN.fullmatch(namespace):
        _fail(
            error_code,
            f"{where}.{key} must be a valid Kubernetes namespace (DNS-1123 label): {namespace!r}.",
            "Use lower case alphanumeric characters or '-', start and end with an alphanumeric character, "
            f"and keep length <= {K8S_NAMESPACE_MAX_LENGTH}.",
        )


def _validate_k8s_service_name(data: dict[str, Any], key: str, where: str) -> None:
    service_name = data.get(key)
    if not isinstance(service_name, str) or not service_name:
        _fail("INVALID_CONFIG", f"{where}.{key} must be a non-empty string.", "Fix the runtime config.")
    if len(service_name) > K8S_SERVICE_NAME_MAX_LENGTH or not K8S_NAME_PATTERN.fullmatch(service_name):
        _fail(
            "INVALID_CONFIG",
            f"{where}.{key} must be a valid Kubernetes Service name (DNS-1035 label): {service_name!r}.",
            "Use lower case alphanumeric characters or '-', start with a letter, end with an alphanumeric character, "
            f"and keep length <= {K8S_SERVICE_NAME_MAX_LENGTH}.",
        )


def _validate_k8s_secret_name(
    name: str, label: str, error_code: str = "INVALID_CONFIG", hint: str = "Fix the runtime config."
) -> None:
    if (
        not isinstance(name, str)
        or not name
        or len(name) > K8S_SECRET_NAME_MAX_LENGTH
        or not K8S_SECRET_NAME_PATTERN.fullmatch(name)
    ):
        _fail(
            error_code,
            f"{label} must contain valid Kubernetes object names.",
            "Use lower case alphanumeric characters, '-', or '.', start and end with an alphanumeric character, "
            f"and keep length <= 253. {hint}",
        )


def _optional_k8s_secret_name_list(data: dict[str, Any], key: str, label: str) -> list[str] | None:
    if key not in data or data[key] is None:
        return None
    names = data[key]
    if not isinstance(names, list):
        _fail("INVALID_CONFIG", f"{label} must be a list of Kubernetes object names.", "Fix the runtime config.")
    for name in names:
        _validate_k8s_secret_name(name, label)
    return names


def _optional_int(data: dict[str, Any], key: str, where: str) -> int | None:
    if key not in data or data[key] is None:
        return None
    if not isinstance(data[key], int):
        _fail("INVALID_CONFIG", f"{where}.{key} must be an integer.", "Fix the runtime config.")
    return data[key]


def _optional_non_negative_int(data: dict[str, Any], key: str, where: str) -> int | None:
    value = _optional_int(data, key, where)
    if isinstance(value, bool):
        _fail("INVALID_CONFIG", f"{where}.{key} must be an integer.", "Fix the runtime config.")
    if value is not None and value < 0:
        _fail("INVALID_CONFIG", f"{where}.{key} must be a non-negative integer.", "Fix the runtime config.")
    return value


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(data, default_flow_style=False, sort_keys=False), encoding="utf-8")
