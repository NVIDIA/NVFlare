# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
"""Stateless Kubernetes Pod construction and startup diagnostics shared by launchers."""

import copy

_PENDING_FAILURE_WAITING_REASONS = {
    "CreateContainerConfigError",
    "CreateContainerError",
    "ErrImagePull",
    "ErrImageNeverPull",
    "ImagePullBackOff",
    "InvalidImageName",
    "RunContainerError",
    "CrashLoopBackOff",
}


def _normalize_image_pull_secrets(image_pull_secrets) -> list[str]:
    if image_pull_secrets is None:
        return []
    if not isinstance(image_pull_secrets, list):
        raise ValueError("image_pull_secrets must be a list of Kubernetes Secret names")
    for name in image_pull_secrets:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("image_pull_secrets entries must be non-empty strings")
    return list(image_pull_secrets)


def _ensure_manifest_mapping(parent: dict, key: str, label: str) -> dict:
    value = parent.get(key)
    if value is None:
        value = {}
        parent[key] = value
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a dictionary.")
    return value


def _ensure_manifest_containers(spec: dict) -> list[dict]:
    containers = spec.get("containers")
    if containers is None:
        containers = [{}]
        spec["containers"] = containers
    if not isinstance(containers, list):
        raise ValueError("pod spec containers must be a list.")
    if not containers:
        containers.append({})
    for container in containers:
        if not isinstance(container, dict):
            raise ValueError("pod spec containers entries must be dictionaries.")
    return containers


def _prepare_pod_manifest_template(pod_manifest_template: dict) -> dict:
    if not isinstance(pod_manifest_template, dict):
        raise ValueError("pod manifest template must be a dictionary.")
    kind = pod_manifest_template.get("kind")
    if kind and kind != "Pod":
        raise ValueError("pod manifest template must define kind: Pod.")
    pod_manifest = copy.deepcopy(pod_manifest_template)
    pod_manifest.setdefault("apiVersion", "v1")
    pod_manifest["kind"] = "Pod"
    _ensure_manifest_mapping(pod_manifest, "metadata", "pod manifest metadata")
    spec = _ensure_manifest_mapping(pod_manifest, "spec", "pod manifest spec")
    _ensure_manifest_containers(spec)
    return pod_manifest


def _merge_named_items(template_items, job_items, label: str) -> list:
    if template_items is None:
        template_items = []
    if job_items is None:
        job_items = []
    if not isinstance(template_items, list) or not isinstance(job_items, list):
        raise ValueError(f"{label} must be a list.")

    job_items_by_name = {}
    unnamed_job_items = []
    for item in job_items:
        if not isinstance(item, dict):
            raise ValueError(f"{label} entries must be dictionaries.")
        name = item.get("name")
        if isinstance(name, str) and name:
            job_items_by_name[name] = item
        else:
            unnamed_job_items.append(item)

    result = []
    used_job_item_names = set()
    for item in template_items:
        if not isinstance(item, dict):
            raise ValueError(f"{label} entries must be dictionaries.")
        name = item.get("name")
        if name in job_items_by_name:
            result.append(copy.deepcopy(job_items_by_name[name]))
            used_job_item_names.add(name)
        else:
            result.append(copy.deepcopy(item))

    for name, item in job_items_by_name.items():
        if name not in used_job_item_names:
            result.append(copy.deepcopy(item))
    result.extend(copy.deepcopy(unnamed_job_items))
    return result


def _select_job_container(containers: list[dict], container_name: str, require_main_container: bool = False) -> dict:
    target_names = {name for name in (container_name, "nvflare_job") if isinstance(name, str) and name}
    for container in containers:
        if container.get("name") in target_names:
            return container
    if require_main_container and len(containers) > 1:
        raise ValueError(
            f"pod_template has multiple containers and none is named one of {sorted(target_names)}; "
            "mark the main container so study env/secret entries cannot land on a sidecar."
        )
    return containers[0]


def build_pod_manifest(job_config, python_path, pod_manifest_template=None, *, preserve_empty_env=False):
    """Build an independent Pod manifest without API calls or execution state."""
    job_config = copy.deepcopy(job_config)
    uses_template = pod_manifest_template is not None
    if uses_template:
        manifest = _prepare_pod_manifest_template(pod_manifest_template)
        spec = manifest["spec"]
        container = _select_job_container(
            spec["containers"], job_config.get("container_name"), job_config.get("require_main_container", False)
        )
    else:
        container = {"imagePullPolicy": "Always"}
        manifest = {
            "apiVersion": "v1",
            "kind": "Pod",
            "metadata": {},
            "spec": {"containers": [container], "volumes": [], "restartPolicy": "Never"},
        }
        spec = manifest["spec"]
    command = job_config.get("command")
    if not command:
        raise ValueError("job_config must contain a non-empty 'command' key")
    manifest["metadata"]["name"] = job_config.get("name")
    spec["restartPolicy"] = "Never"
    volumes = job_config.get("volume_list", [])
    spec["volumes"] = (
        _merge_named_items(spec.get("volumes"), volumes, "pod spec volumes") if uses_template else list(volumes)
    )
    image_pull_secrets = _normalize_image_pull_secrets(job_config.get("image_pull_secrets"))
    if image_pull_secrets:
        refs = [{"name": name} for name in image_pull_secrets]
        spec["imagePullSecrets"] = (
            _merge_named_items(spec.get("imagePullSecrets"), refs, "pod spec imagePullSecrets")
            if uses_template
            else refs
        )
    if job_config.get("security_context"):
        spec["securityContext"] = job_config["security_context"]
    image = job_config.get("image")
    if not image:
        raise ValueError("job_config must contain a non-empty 'image' key")
    container["image"] = image
    container["name"] = job_config.get("container_name", "nvflare_job")
    container["command"] = [python_path]
    module_args = []
    for key, value in (job_config.get("module_args") or {}).items():
        if value is not None:
            module_args.extend((key, str(value)))
    set_list = job_config.get("set_list")
    container["args"] = ["-u", "-m", command] + module_args + (["--set"] + set_list if set_list else [])
    mounts = list(job_config.get("volume_mount_list", []))
    if uses_template:
        container.setdefault("imagePullPolicy", "Always")
        container["volumeMounts"] = _merge_named_items(container.get("volumeMounts"), mounts, "container volumeMounts")
    else:
        container["volumeMounts"] = mounts
    if job_config.get("resources"):
        container["resources"] = job_config["resources"]
    env_items = [
        {"name": key, "value": str(value)}
        for key, value in job_config.get("env", {}).items()
        if preserve_empty_env or str(value)
    ]
    env_items.extend(
        {"name": ref["name"], "valueFrom": {"secretKeyRef": {"name": ref["source"], "key": ref["key"]}}}
        for ref in job_config.get("secret_env", [])
    )
    if env_items or preserve_empty_env:
        container["env"] = (
            _merge_named_items(container.get("env"), env_items, "container env") if uses_template else env_items
        )
    return manifest


def get_container_waiting_failure(status):
    """Describe a failed container startup without owning a job/task lifecycle."""
    for attribute in ("init_container_statuses", "container_statuses"):
        statuses = getattr(status, attribute, None)
        if not isinstance(statuses, (list, tuple)):
            continue
        for container_status in statuses:
            waiting = getattr(getattr(container_status, "state", None), "waiting", None)
            if not waiting:
                continue
            reason = getattr(waiting, "reason", None)
            if reason in _PENDING_FAILURE_WAITING_REASONS:
                message = getattr(waiting, "message", None)
                return " ".join(str(value) for value in (reason, message) if value) or "container is waiting"
    return ""
