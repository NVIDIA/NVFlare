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

"""Kubernetes stage and unstage implementation for ``nvflare deploy``."""

from __future__ import annotations

import base64
import hashlib
import os
import re
import shlex
import subprocess
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

from nvflare.tool.cli_output import output_ok
from nvflare.tool.deploy.deploy_common import RESOURCES_JSON_DEFAULT, _fail, _load_json_file, _warn
from nvflare.tool.deploy.k8s_deploy import (
    HELM_CHART_DIR,
    K8S_LAUNCHER_PATHS,
    K8S_SERVICE_NAME_MAX_LENGTH,
    _k8s_release_name,
    _stable_k8s_name,
    _validate_k8s_namespace,
    _validate_k8s_secret_name,
    _write_yaml,
)

K8_STAGE_VALUES_KEY = "workspaceConfig"
K8_STAGE_NAMESPACE_KEY = "namespace"
K8_STAGE_LOCAL_KEY = "local"
K8_STAGE_STARTUP_KEY = "startup"
K8_STAGE_OBJECT_NAME_MAX_LENGTH = 253
K8_STAGE_OBJECT_SIZE_WARN_BYTES = 900 * 1024
KUBECTL_ENV_VAR = "KUBECTL"
K8_STAGE_ALLOWED_KUBECTL_NAMES = {"kubectl", "oc"}


def stage_k8_deployment(args) -> None:
    kit, namespace, local_configmap, startup_secret = _resolve_k8_stage_inputs(args)
    _validate_k8_stage_inputs(kit, namespace, local_configmap, startup_secret)
    kubectl = _resolve_kubectl(args)

    local_bundle = _build_volume_bundle(kit / "local")
    startup_bundle = _build_volume_bundle(kit / "startup")

    _warn_if_large_k8_object("ConfigMap", local_configmap, local_bundle["encoded_size"])
    _warn_if_large_k8_object("Secret", startup_secret, startup_bundle["encoded_size"])

    # Record every cleanup target before changing the cluster. If apply only
    # partially succeeds, ``unstage`` can still remove the exact objects.
    _patch_k8_stage_values(
        kit=kit,
        namespace=namespace,
        local_configmap=local_configmap,
        local_items=local_bundle["items"],
        startup_secret=startup_secret,
        startup_items=startup_bundle["items"],
    )
    _kubectl_apply(_configmap_manifest(local_configmap, namespace, local_bundle["data"]), kubectl)
    _kubectl_apply(_secret_manifest(startup_secret, namespace, startup_bundle["data"]), kubectl)
    helm_command = _k8_stage_helm_command(kit, namespace)
    cleanup_command = _k8_unstage_command(kit, kubectl)

    output_ok(
        {
            "namespace": namespace,
            "prepared_kit": str(kit),
            "local_configmap": local_configmap,
            "startup_secret": startup_secret,
            "local_files": len(local_bundle["items"]),
            "startup_files": len(startup_bundle["items"]),
            "helm_values": str(kit / HELM_CHART_DIR / "values.yaml"),
            "kubectl": kubectl,
            "next_step": "Start the server/client parent pod with the helm_command.",
            "helm_command": helm_command,
            "cleanup_step": "After Helm uninstall, remove the staged credentials with the cleanup_command.",
            "cleanup_command": cleanup_command,
        }
    )


def unstage_k8_deployment(args) -> None:
    kit, namespace, local_configmap, startup_secret = _resolve_k8_unstage_inputs(args)
    _validate_k8_unstage_inputs(kit, namespace, local_configmap, startup_secret)
    kubectl = _resolve_kubectl(args)

    # Delete the credential-bearing Secret first. Passing both exact
    # kind/name references in one command lets kubectl attempt both deletions.
    delete_cmd = [
        kubectl,
        "delete",
        f"secret/{startup_secret}",
        f"configmap/{local_configmap}",
        "--namespace",
        namespace,
        "--ignore-not-found=true",
    ]
    error_cmd = [
        kubectl,
        "delete",
        "secret/<staged-name>",
        f"configmap/{local_configmap}",
        "--namespace",
        namespace,
        "--ignore-not-found=true",
    ]
    _kubectl(delete_cmd, error_cmd=error_cmd)
    _clear_k8_stage_values(kit, namespace, local_configmap, startup_secret)

    output_ok(
        {
            "status": "unstaged",
            "namespace": namespace,
            "prepared_kit": str(kit),
            "local_configmap": local_configmap,
            "helm_values": str(kit / HELM_CHART_DIR / "values.yaml"),
            "kubectl": kubectl,
        }
    )


def _resolve_k8_stage_inputs(args) -> tuple[Path, str, str, str]:
    kit = _resolve_k8_kit(args, "stage")
    values = _load_k8_stage_values(kit)
    stage_state = _read_k8_stage_state(values)
    requested_namespace = getattr(args, "namespace", None)
    requested_local_configmap = getattr(args, "local_configmap", None)
    requested_startup_secret = getattr(args, "startup_secret", None)
    if (
        not requested_namespace
        and not stage_state.get(K8_STAGE_NAMESPACE_KEY)
        and (stage_state.get("local_configmap") or stage_state.get("startup_secret"))
    ):
        _fail(
            "INVALID_ARGS",
            "The existing staged resource names do not include their Kubernetes namespace.",
            "Pass the original stage namespace with --namespace before restaging this legacy prepared kit.",
        )
    _reject_k8_stage_binding_change("namespace", stage_state.get(K8_STAGE_NAMESPACE_KEY), requested_namespace)
    _reject_k8_stage_binding_change("local ConfigMap", stage_state.get("local_configmap"), requested_local_configmap)
    _reject_k8_stage_binding_change("startup Secret", stage_state.get("startup_secret"), requested_startup_secret)

    namespace = (
        requested_namespace or stage_state.get(K8_STAGE_NAMESPACE_KEY) or _read_k8_launcher_namespace(kit) or "default"
    )
    local_configmap = requested_local_configmap or stage_state.get("local_configmap")
    startup_secret = requested_startup_secret or stage_state.get("startup_secret")
    if not local_configmap or not startup_secret:
        default_local_configmap, default_startup_secret = _default_k8_stage_resource_names(kit, values)
        local_configmap = local_configmap or default_local_configmap
        startup_secret = startup_secret or default_startup_secret
    return kit, namespace, local_configmap, startup_secret


def _resolve_k8_unstage_inputs(args) -> tuple[Path, str, str, str]:
    kit = _resolve_k8_kit(args, "unstage")
    values = _load_k8_stage_values(kit)
    stage_state = _read_k8_stage_state(values)
    requested_namespace = getattr(args, "namespace", None)
    requested_local_configmap = getattr(args, "local_configmap", None)
    requested_startup_secret = getattr(args, "startup_secret", None)
    _reject_k8_unstage_target_mismatch("namespace", stage_state.get(K8_STAGE_NAMESPACE_KEY), requested_namespace)
    _reject_k8_unstage_target_mismatch("local ConfigMap", stage_state.get("local_configmap"), requested_local_configmap)
    _reject_k8_unstage_target_mismatch("startup Secret", stage_state.get("startup_secret"), requested_startup_secret)

    namespace = requested_namespace or stage_state.get(K8_STAGE_NAMESPACE_KEY)
    local_configmap = requested_local_configmap or stage_state.get("local_configmap")
    startup_secret = requested_startup_secret or stage_state.get("startup_secret")

    if not namespace:
        _fail(
            "INVALID_ARGS",
            "The staged Kubernetes namespace is not recorded in the prepared kit.",
            "Pass the namespace used by stage with --namespace.",
        )
    if not local_configmap or not startup_secret:
        _fail(
            "INVALID_ARGS",
            "The staged ConfigMap and Secret names are not recorded in the prepared kit.",
            "Pass the exact staged names with --local-configmap and --startup-secret.",
        )
    return kit, namespace, local_configmap, startup_secret


def _resolve_k8_kit(args, action: str) -> Path:
    positional_kit = getattr(args, "kit", None)
    flag_kit = getattr(args, "kit_flag", None)
    if positional_kit and flag_kit:
        _fail("INVALID_ARGS", "Specify the prepared startup kit only once.", "Use either positional kit or --kit.")
    kit_arg = positional_kit or flag_kit
    if not kit_arg:
        _fail(
            "INVALID_ARGS",
            "Missing prepared startup kit directory.",
            f"Run nvflare deploy k8 {action} <prepared-kit-dir>.",
        )
    return Path(kit_arg).expanduser().resolve()


def reject_replacing_staged_k8_output(output: Path) -> None:
    if not (output / HELM_CHART_DIR / "values.yaml").is_file():
        return

    values = _load_k8_stage_values(output)
    stage_state = _read_k8_stage_state(values)
    if stage_state.get("local_configmap") or stage_state.get("startup_secret"):
        _fail(
            "OUTPUT_STAGED",
            f"Prepared Kubernetes output still has staged resources: {output}",
            f"After Helm uninstall, run nvflare deploy k8 unstage {output} before replacing this output.",
        )


def _read_k8_stage_state(values: dict[str, Any]) -> dict[str, Any]:
    workspace_config = values.get(K8_STAGE_VALUES_KEY)
    if workspace_config is None:
        return {}
    if not isinstance(workspace_config, dict):
        _fail("INVALID_KIT", "helm_chart/values.yaml workspaceConfig must be a mapping.", "Fix the prepared chart.")

    local = workspace_config.get(K8_STAGE_LOCAL_KEY) or {}
    startup = workspace_config.get(K8_STAGE_STARTUP_KEY) or {}
    if not isinstance(local, dict) or not isinstance(startup, dict):
        _fail(
            "INVALID_KIT",
            "helm_chart/values.yaml workspaceConfig local/startup entries must be mappings.",
            "Fix the prepared chart.",
        )
    return {
        K8_STAGE_NAMESPACE_KEY: workspace_config.get(K8_STAGE_NAMESPACE_KEY),
        "local_configmap": local.get("configMapName"),
        "startup_secret": startup.get("secretName"),
    }


def _reject_k8_stage_binding_change(label: str, staged_value: Any, requested_value: Any) -> None:
    if staged_value and requested_value and staged_value != requested_value:
        _fail(
            "INVALID_ARGS",
            f"Prepared kit is already staged with a different {label}.",
            "Run nvflare deploy k8 unstage before changing staged resource names or namespace.",
        )


def _reject_k8_unstage_target_mismatch(label: str, staged_value: Any, requested_value: Any) -> None:
    if staged_value and requested_value and staged_value != requested_value:
        _fail(
            "INVALID_ARGS",
            f"Requested {label} does not match the value recorded by stage.",
            "Omit the override to use the target recorded by stage.",
        )


def _validate_k8_stage_inputs(kit: Path, namespace: str, local_configmap: str, startup_secret: str) -> None:
    if not kit.is_dir():
        _fail("INVALID_KIT", f"Prepared kit directory does not exist: {kit}", "Provide an existing prepared K8s kit.")
    for folder in ("local", "startup"):
        path = kit / folder
        if not path.is_dir():
            _fail(
                "INVALID_KIT",
                f"Missing prepared kit folder: {path}",
                "Run nvflare deploy prepare with runtime: k8s before staging.",
            )
        if path.is_symlink():
            _fail(
                "INVALID_KIT",
                f"Prepared kit folder must not be a symlink: {path}",
                "Use a prepared K8s kit whose local/ and startup/ folders are regular directories.",
            )
    if not _find_k8s_launcher(kit):
        _fail(
            "INVALID_KIT",
            f"Input folder was not generated for the Kubernetes runtime: {kit}",
            "Run nvflare deploy prepare with runtime: k8s and pass that prepared output directory to "
            "nvflare deploy k8 stage.",
        )
    if not (kit / HELM_CHART_DIR / "values.yaml").is_file():
        _fail(
            "INVALID_KIT",
            f"Missing Helm values file: {kit / HELM_CHART_DIR / 'values.yaml'}",
            "Run nvflare deploy prepare with runtime: k8s before staging.",
        )
    _validate_k8s_namespace(
        {"namespace": namespace},
        "namespace",
        "deploy k8 stage",
        error_code="INVALID_ARGS",
        hint="Use a valid --namespace value.",
    )
    _validate_k8s_secret_name(
        local_configmap,
        "local ConfigMap name",
        error_code="INVALID_ARGS",
        hint="Use a valid --local-configmap value.",
    )
    _validate_k8s_secret_name(
        startup_secret,
        "startup Secret name",
        error_code="INVALID_ARGS",
        hint="Use a valid --startup-secret value.",
    )


def _validate_k8_unstage_inputs(kit: Path, namespace: str, local_configmap: str, startup_secret: str) -> None:
    if not kit.is_dir():
        _fail("INVALID_KIT", f"Prepared kit directory does not exist: {kit}", "Provide an existing prepared K8s kit.")
    if not (kit / HELM_CHART_DIR / "values.yaml").is_file():
        _fail(
            "INVALID_KIT",
            f"Missing Helm values file: {kit / HELM_CHART_DIR / 'values.yaml'}",
            "Provide the prepared K8s kit used for staging.",
        )
    _validate_k8s_namespace(
        {"namespace": namespace},
        "namespace",
        "deploy k8 unstage",
        error_code="INVALID_ARGS",
        hint="Use the namespace passed to stage.",
    )
    _validate_k8s_secret_name(
        local_configmap,
        "local ConfigMap name",
        error_code="INVALID_ARGS",
        hint="Use the exact ConfigMap name passed to stage.",
    )
    _validate_k8s_secret_name(
        startup_secret,
        "startup Secret name",
        error_code="INVALID_ARGS",
        hint="Use the exact Secret name passed to stage.",
    )


def _load_k8_stage_values(kit: Path) -> dict[str, Any]:
    values_path = kit / HELM_CHART_DIR / "values.yaml"
    if not values_path.is_file():
        return {}
    try:
        with values_path.open("rt", encoding="utf-8") as f:
            values = yaml.safe_load(f)
    except Exception as ex:
        _fail("INVALID_KIT", f"Failed to parse Helm values: {ex}", "Fix helm_chart/values.yaml.")
    if not isinstance(values, dict):
        _fail("INVALID_KIT", "helm_chart/values.yaml must contain a YAML mapping.", "Fix the prepared Helm chart.")
    return values


def _read_k8_launcher_namespace(kit: Path) -> str | None:
    component = _find_k8s_launcher(kit)
    if not component:
        return None
    args = component.get("args") or {}
    namespace = args.get("namespace")
    return namespace if isinstance(namespace, str) and namespace else None


def _find_k8s_launcher(kit: Path) -> dict[str, Any] | None:
    resources_path = kit / "local" / RESOURCES_JSON_DEFAULT
    if not resources_path.is_file():
        return None
    resources = _load_json_file(resources_path, RESOURCES_JSON_DEFAULT)
    components = resources.get("components", [])
    if not isinstance(components, list):
        return None
    for component in components:
        if (
            isinstance(component, dict)
            and component.get("id") == "k8s_launcher"
            and component.get("path") in K8S_LAUNCHER_PATHS
        ):
            return component
    return None


def _default_k8_stage_resource_names(kit: Path, values: dict[str, Any]) -> tuple[str, str]:
    raw_name = values.get("siteName") or values.get("name") or kit.name
    safe_name = _stable_k8s_name(str(raw_name), K8S_SERVICE_NAME_MAX_LENGTH)
    return (
        _stable_k8s_name(f"nvflare-local-{safe_name}", K8_STAGE_OBJECT_NAME_MAX_LENGTH),
        _stable_k8s_name(f"nvflare-startup-{safe_name}", K8_STAGE_OBJECT_NAME_MAX_LENGTH),
    )


def _resolve_kubectl(args) -> str:
    kubectl = getattr(args, "kubectl", None) or os.environ.get(KUBECTL_ENV_VAR) or "kubectl"
    if not isinstance(kubectl, str) or not kubectl.strip():
        _fail("INVALID_ARGS", "Kubernetes CLI command must be a non-empty string.", "Set --kubectl or KUBECTL.")
    kubectl = kubectl.strip()
    if kubectl not in K8_STAGE_ALLOWED_KUBECTL_NAMES:
        _fail(
            "INVALID_ARGS",
            f"Kubernetes CLI command must be one of {sorted(K8_STAGE_ALLOWED_KUBECTL_NAMES)}: {kubectl!r}",
            "Set --kubectl or KUBECTL to kubectl or oc.",
        )
    return kubectl


def _k8_stage_helm_command(kit: Path, namespace: str) -> str:
    values = _load_k8_stage_values(kit)
    raw_name = values.get("siteName") or values.get("name") or kit.name
    release_name = _k8s_release_name(str(raw_name))
    chart_dir = kit / HELM_CHART_DIR
    return _format_command(["helm", "upgrade", "--install", release_name, str(chart_dir), "--namespace", namespace])


def _k8_unstage_command(kit: Path, kubectl: str) -> str:
    return _format_command(
        [
            "nvflare",
            "deploy",
            "k8s",
            "unstage",
            str(kit),
            "--kubectl",
            kubectl,
        ]
    )


def _build_volume_bundle(root: Path) -> dict[str, Any]:
    data = {}
    items = []
    encoded_size = 0
    root_resolved = root.resolve()
    files = sorted(path for path in root.rglob("*") if path.is_file() or path.is_symlink())
    if not files:
        _fail("INVALID_KIT", f"No files found in prepared kit folder: {root}", "Stage a non-empty prepared folder.")

    for index, path in enumerate(files):
        if path.is_symlink():
            _fail(
                "INVALID_KIT",
                f"Symlinks cannot be staged into Kubernetes volumes: {path}",
                "Replace it with a file.",
            )
        source_path = path.resolve()
        if not source_path.is_relative_to(root_resolved):
            _fail(
                "INVALID_KIT",
                f"Staged file resolves outside the prepared kit folder: {path}",
                "Use files contained under local/ or startup/.",
            )
        rel_path = path.relative_to(root).as_posix()
        _validate_k8_volume_item_path(rel_path, path)
        key = _k8_stage_file_key(index, rel_path)
        encoded = base64.b64encode(source_path.read_bytes()).decode("ascii")
        data[key] = encoded
        encoded_size += len(encoded)
        items.append({"key": key, "path": rel_path})
    return {"data": data, "items": items, "encoded_size": encoded_size}


def _validate_k8_volume_item_path(rel_path: str, source_path: Path) -> None:
    path = PurePosixPath(rel_path)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        _fail("INVALID_KIT", f"Unsafe staged file path: {source_path}", "Use files contained under local/ or startup/.")


def _k8_stage_file_key(index: int, rel_path: str) -> str:
    digest = hashlib.sha256(rel_path.encode("utf-8")).hexdigest()[:12]
    safe_name = re.sub(r"[^A-Za-z0-9_.-]", "_", PurePosixPath(rel_path).name) or "file"
    prefix = f"f{index:04d}-{digest}-"
    return f"{prefix}{safe_name}"[:253]


def _configmap_manifest(name: str, namespace: str, data: dict[str, str]) -> dict[str, Any]:
    return {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {"name": name, "namespace": namespace},
        "binaryData": data,
    }


def _secret_manifest(name: str, namespace: str, data: dict[str, str]) -> dict[str, Any]:
    return {
        "apiVersion": "v1",
        "kind": "Secret",
        "metadata": {"name": name, "namespace": namespace},
        "type": "Opaque",
        "data": data,
    }


def _patch_k8_stage_values(
    kit: Path,
    namespace: str,
    local_configmap: str,
    local_items: list[dict[str, str]],
    startup_secret: str,
    startup_items: list[dict[str, str]],
) -> None:
    values_path = kit / HELM_CHART_DIR / "values.yaml"
    values = _load_k8_stage_values(kit)
    workspace_config = values.get(K8_STAGE_VALUES_KEY)
    if workspace_config is None:
        workspace_config = {}
        values[K8_STAGE_VALUES_KEY] = workspace_config
    workspace_config[K8_STAGE_NAMESPACE_KEY] = namespace
    workspace_config[K8_STAGE_LOCAL_KEY] = {
        "configMapName": local_configmap,
        "items": local_items,
    }
    workspace_config[K8_STAGE_STARTUP_KEY] = {
        "secretName": startup_secret,
        "items": startup_items,
    }
    _write_yaml(values_path, values)


def _clear_k8_stage_values(kit: Path, namespace: str, local_configmap: str, startup_secret: str) -> None:
    values_path = kit / HELM_CHART_DIR / "values.yaml"
    values = _load_k8_stage_values(kit)
    workspace_config = values.get(K8_STAGE_VALUES_KEY)
    if not isinstance(workspace_config, dict):
        return

    changed = False
    local = workspace_config.get(K8_STAGE_LOCAL_KEY)
    if isinstance(local, dict) and local.get("configMapName") == local_configmap:
        workspace_config[K8_STAGE_LOCAL_KEY] = {"configMapName": None, "items": []}
        changed = True

    startup = workspace_config.get(K8_STAGE_STARTUP_KEY)
    if isinstance(startup, dict) and startup.get("secretName") == startup_secret:
        workspace_config[K8_STAGE_STARTUP_KEY] = {"secretName": None, "items": []}
        changed = True

    local = workspace_config.get(K8_STAGE_LOCAL_KEY) or {}
    startup = workspace_config.get(K8_STAGE_STARTUP_KEY) or {}
    if (
        isinstance(local, dict)
        and isinstance(startup, dict)
        and not local.get("configMapName")
        and not startup.get("secretName")
        and workspace_config.get(K8_STAGE_NAMESPACE_KEY) == namespace
    ):
        workspace_config[K8_STAGE_NAMESPACE_KEY] = None
        changed = True

    if changed:
        _write_yaml(values_path, values)


def _warn_if_large_k8_object(kind: str, name: str, encoded_size: int) -> None:
    if encoded_size > K8_STAGE_OBJECT_SIZE_WARN_BYTES:
        _warn(
            f"{kind} '{name}' encoded payload is about {encoded_size} bytes; "
            "Kubernetes objects have size limits, so large local/startup folders may fail to apply."
        )


def _kubectl_apply(manifest: dict[str, Any], kubectl: str) -> subprocess.CompletedProcess:
    payload = yaml.safe_dump(manifest, default_flow_style=False, sort_keys=False)
    if kubectl == "oc":
        return _kubectl(["oc", "apply", "-f", "-"], input_text=payload)
    return _kubectl(["kubectl", "apply", "-f", "-"], input_text=payload)


def _kubectl(
    cmd: list[str], input_text: str | None = None, error_cmd: list[str] | None = None
) -> subprocess.CompletedProcess:
    try:
        result = subprocess.run(
            cmd,
            input=input_text,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    except OSError:
        _fail(
            "KUBECTL_NOT_FOUND",
            f"Kubernetes CLI executable could not be started: {cmd[0]}",
            "Install kubectl or set --kubectl/KUBECTL to a compatible command such as oc.",
        )

    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        message = f"Kubernetes command failed: {_format_command(error_cmd if error_cmd is not None else cmd)}"
        if detail:
            message = f"{message}\n{detail}"
        _fail("KUBECTL_FAILED", message, "Check kubectl access, namespace, and resource quotas.")
    return result


def _format_command(cmd: list[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in cmd)
