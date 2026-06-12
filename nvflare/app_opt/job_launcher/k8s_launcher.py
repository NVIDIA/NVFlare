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
from __future__ import annotations

import base64
import copy
import hashlib
import logging
import os
import re
import time
from abc import abstractmethod
from datetime import datetime
from enum import Enum

import yaml

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import FLContextKey, JobConstants
from nvflare.apis.fl_context import FLContext
from nvflare.apis.job_def import JobMetaKey
from nvflare.apis.job_launcher_spec import JobHandleSpec, JobLauncherSpec, JobProcessArgs, JobReturnCode, add_launcher
from nvflare.app_opt.job_launcher.study_data import (
    load_study_data_file,
    resolve_study_dataset_mounts,
    should_mount_study_data,
)
from nvflare.app_opt.job_launcher.workspace_cell_transfer import (
    ENV_WORKSPACE_OWNER_FQCN,
    ENV_WORKSPACE_TRANSFER_TOKEN,
    WorkspaceTransferManager,
)
from nvflare.utils.job_launcher_utils import get_client_job_args, get_job_launcher_spec, get_server_job_args


class JobState(Enum):
    STARTING = "starting"
    RUNNING = "running"
    TERMINATED = "terminated"
    SUCCEEDED = "succeeded"
    UNKNOWN = "unknown"


class PodPhase(Enum):
    PENDING = "Pending"
    RUNNING = "Running"
    SUCCEEDED = "Succeeded"
    FAILED = "Failed"
    UNKNOWN = "Unknown"


class PendingPodAction(Enum):
    WAIT = "wait"
    WAIT_FOR_RESOURCES = "wait_for_resources"
    FAIL = "fail"


POD_STATE_MAPPING = {
    PodPhase.PENDING.value: JobState.STARTING,
    PodPhase.RUNNING.value: JobState.RUNNING,
    PodPhase.SUCCEEDED.value: JobState.SUCCEEDED,
    PodPhase.FAILED.value: JobState.TERMINATED,
    PodPhase.UNKNOWN.value: JobState.UNKNOWN,
}

JOB_RETURN_CODE_MAPPING = {
    JobState.SUCCEEDED: JobReturnCode.SUCCESS,
    JobState.STARTING: JobReturnCode.UNKNOWN,
    JobState.RUNNING: JobReturnCode.UNKNOWN,
    JobState.TERMINATED: JobReturnCode.ABORTED,
    JobState.UNKNOWN: JobReturnCode.UNKNOWN,
}

DEFAULT_CONTAINER_ARGS_MODULE_ARGS_DICT = {
    "-m": None,
    "-w": None,
    "-t": None,
    "-d": None,
    "-n": None,
    "-c": None,
    "-p": None,
    "-g": None,
    "-scheme": None,
    "-s": None,
}

DEFAULT_NAMESPACE = "default"
DEFAULT_PENDING_TIMEOUT = 120
DEFAULT_PYTHON_PATH = "/usr/local/bin/python"
POLL_INTERVAL = 1
SCHEDULED_EVENT_FAILURE_MAX_AGE = 60


WORKSPACE_MOUNT_PATH = "/var/tmp/nvflare/workspace"
DEFAULT_EPHEMERAL_STORAGE = "1Gi"

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
_PENDING_FAILURE_EVENT_REASONS = {
    "BackOff",
    "Failed",
    "FailedAttachVolume",
    "FailedCreatePodSandBox",
    "FailedMount",
    "FailedScheduling",
    "FailedSync",
    "InspectFailed",
    "InvalidImageName",
    "NetworkNotReady",
}
# Files actually read from startup/ by the job pod at runtime. Others in
# startup/ are dropped to shrink the Secret. local/ is bundled whole with each
# job workspace so job resource files and local custom code keep working.
_STARTUP_KEEP_SUFFIXES = (".crt", ".key", ".pem", ".json")


def _keep_startup_file(fname: str) -> bool:
    return fname.endswith(_STARTUP_KEEP_SUFFIXES)


def _normalize_image_pull_secrets(image_pull_secrets) -> list[str]:
    if image_pull_secrets is None:
        return []
    if not isinstance(image_pull_secrets, list):
        raise ValueError("image_pull_secrets must be a list of Kubernetes Secret names")
    for name in image_pull_secrets:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("image_pull_secrets entries must be non-empty strings")
    return list(image_pull_secrets)


def _normalize_pending_timeout(pending_timeout, field_name="pending_timeout"):
    if pending_timeout is None:
        return None
    if isinstance(pending_timeout, bool) or not isinstance(pending_timeout, (int, float)):
        raise ValueError(f"{field_name} must be a non-negative number of seconds or None")
    if pending_timeout < 0:
        raise ValueError(f"{field_name} must be a non-negative number of seconds or None")
    return pending_timeout


def _obj_text(*values) -> str:
    return " ".join(str(v) for v in values if v)


def _is_cpu_memory_gpu_shortage(message: str) -> bool:
    if not message:
        return False
    for resource_name in re.findall(r"\binsufficient\s+([a-z0-9./_-]+)", message, flags=re.IGNORECASE):
        resource_name = resource_name.lower().rstrip(".,;:")
        if resource_name in {"cpu", "memory"} or "gpu" in resource_name:
            return True
    return False


def _timestamp_to_seconds(value):
    if isinstance(value, datetime):
        return value.timestamp()
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.strip().replace("Z", "+00:00")).timestamp()
        except ValueError:
            return None
    return None


def _event_timestamp(event):
    series = getattr(event, "series", None)
    metadata = getattr(event, "metadata", None)
    for value in (
        getattr(event, "event_time", None),
        getattr(series, "last_observed_time", None),
        getattr(event, "last_timestamp", None),
        getattr(event, "first_timestamp", None),
        getattr(metadata, "creation_timestamp", None),
    ):
        seconds = _timestamp_to_seconds(value)
        if seconds is not None:
            return seconds
    return None


def _event_sort_key(event):
    event_time = _event_timestamp(event)
    return event_time if event_time is not None else 0


def _is_recent_event(event, now, max_age) -> bool:
    event_time = _event_timestamp(event)
    if event_time is None:
        return False
    return now - event_time <= max_age


def uuid4_to_rfc1123(uuid_str: str) -> str:
    name = uuid_str.lower()
    # Strip any chars that aren't alphanumeric or hyphen
    name = re.sub(r"[^a-z0-9-]", "", name)
    # Prefix with a letter if it starts with a digit
    if name and name[0].isdigit():
        name = "j" + name
    # Kubernetes label limit: 63 chars; strip trailing hyphens after truncation
    # (truncation can expose a hyphen that was interior before slicing)
    return name[:63].rstrip("-")


def site_name_to_rfc1123(site_name: str, max_length: int = 47) -> str:
    """Convert a site name into a stable RFC1123-safe label with a hash suffix."""

    digest = hashlib.sha256(site_name.encode("utf-8")).hexdigest()[:8]
    name = site_name.lower()
    name = re.sub(r"[^a-z0-9-]", "", name).strip("-")
    if not name:
        name = "site"
    if name[0].isdigit():
        name = "s" + name
    head_max = max_length - len(digest) - 1
    name = name[:head_max].rstrip("-") or "site"
    return f"{name}-{digest}"


def job_pod_name(job_id: str, site_name: str) -> str:
    """Build a site-scoped Kubernetes pod name for a FL job."""

    site_suffix = site_name_to_rfc1123(site_name, max_length=20)
    job_prefix_max = 63 - len(site_suffix) - 1
    job_prefix = job_id[:job_prefix_max].rstrip("-")
    return f"{job_prefix}-{site_suffix}"


def study_dataset_volume_name(study: str, dataset: str) -> str:
    return site_name_to_rfc1123(f"data-{study}-{dataset}", max_length=63)


def _load_yaml_file(file_path: str, label: str):
    try:
        with open(file_path, "rt") as f:
            return yaml.safe_load(f)
    except FileNotFoundError as e:
        raise ValueError(f"{label} file '{file_path}' was not found") from e
    except OSError as e:
        raise ValueError(f"Could not read {label} file '{file_path}': {e}") from e
    except yaml.YAMLError as e:
        raise ValueError(f"Could not parse {label} file '{file_path}': {e}") from e


def load_study_job_spec_file(file_path: str, logger: logging.Logger = None) -> dict:
    study_job_spec = _load_yaml_file(file_path, "study job spec")
    if study_job_spec is None:
        study_job_spec = {}
    if not isinstance(study_job_spec, dict):
        raise ValueError(f"file at study_job_spec_file_path '{file_path}' does not contain a dictionary.")
    if not study_job_spec and logger:
        logger.warning("study job spec file '%s' has no study entries; built-in pod manifests will be used", file_path)
    for study, pod_spec_file in study_job_spec.items():
        if not isinstance(study, str) or not study:
            raise ValueError(f"study name {study!r} in '{file_path}' must be a non-empty string.")
        if not isinstance(pod_spec_file, str) or not pod_spec_file:
            raise ValueError(
                f"study job spec entry for study '{study}' in '{file_path}' must be a non-empty pod YAML file path."
            )
    return study_job_spec


def resolve_study_job_spec_path(
    study_job_spec: dict, study: str, file_path: str, logger: logging.Logger = None
) -> str | None:
    if not study or not study_job_spec:
        return None
    pod_spec_file = study_job_spec.get(study)
    if pod_spec_file is None:
        if logger:
            logger.warning(
                "study job spec file '%s' has no entry for study '%s'; built-in pod manifest will be used",
                file_path,
                study,
            )
        return None
    if os.path.isabs(pod_spec_file):
        return pod_spec_file
    return os.path.join(os.path.dirname(file_path), pod_spec_file)


def load_pod_spec_file(file_path: str) -> dict:
    pod_spec = _load_yaml_file(file_path, "pod spec")
    if not isinstance(pod_spec, dict):
        raise ValueError(f"pod spec file '{file_path}' must contain a Kubernetes Pod dictionary.")
    kind = pod_spec.get("kind")
    if kind and kind != "Pod":
        raise ValueError(f"pod spec file '{file_path}' must define kind: Pod.")
    return pod_spec


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


def _select_job_container(containers: list[dict], container_name: str) -> dict:
    for container in containers:
        if container.get("name") in (container_name, "nvflare_job"):
            return container
    return containers[0]


class K8sJobHandle(JobHandleSpec):
    def __init__(
        self,
        job_id: str,
        api_instance,
        job_config: dict,
        namespace=DEFAULT_NAMESPACE,
        timeout=None,
        pending_timeout=DEFAULT_PENDING_TIMEOUT,
        python_path=DEFAULT_PYTHON_PATH,
        workspace_transfer: WorkspaceTransferManager = None,
        workspace_job_id: str = "",
        pod_name: str = None,
        pod_manifest_template: dict = None,
    ):
        super().__init__()
        self.job_id = job_id
        self.pod_name = pod_name if pod_name is not None else job_id
        self.timeout = timeout
        self.terminal_state = None
        self.terminal_return_code = None
        self.workspace_transfer = workspace_transfer
        self.workspace_job_id = workspace_job_id
        self.api_instance = api_instance
        self.namespace = namespace
        self.pending_timeout = _normalize_pending_timeout(pending_timeout)
        self.python_path = python_path
        self.uses_pod_manifest_template = pod_manifest_template is not None
        if self.uses_pod_manifest_template:
            self.pod_manifest = _prepare_pod_manifest_template(pod_manifest_template)
        else:
            self.pod_manifest = {
                "apiVersion": "v1",
                "kind": "Pod",
                "metadata": {"name": None},  # set by job_config['name']
                "spec": {
                    "containers": None,  # link to container_list
                    "volumes": None,  # link to volume_list
                    "restartPolicy": "Never",
                },
            }
        self.volume_list = []

        if self.uses_pod_manifest_template:
            spec = self.pod_manifest["spec"]
            self.container_list = spec["containers"]
            self.job_container = _select_job_container(self.container_list, job_config.get("container_name"))
        else:
            self.container_list = [
                {
                    "image": None,
                    "name": None,
                    "command": [python_path],
                    "args": None,  # args_list + args_dict + args_sets
                    "volumeMounts": None,  # volume_mount_list
                    "imagePullPolicy": "Always",
                }
            ]
            self.job_container = self.container_list[0]
        command = job_config.get("command")
        if not command:
            raise ValueError("job_config must contain a non-empty 'command' key")
        self.container_args_python_args_list = ["-u", "-m", command]
        self.container_volume_mount_list = []
        self._make_manifest(job_config)
        self._stuck_count = 0
        self._pending_since = None
        # Kept for diagnostics only; unit is seconds, not poll iterations like _stuck_count.
        self._pending_timeout_secs = self.pending_timeout
        self._last_event_query_failed = False
        self._pending_timer_paused_at = None
        self.logger = logging.getLogger(self.__class__.__name__)

    def _make_manifest(self, job_config):
        self.container_volume_mount_list.extend(job_config.get("volume_mount_list", []))
        set_list = job_config.get("set_list")
        if not set_list:
            self.container_args_module_args_sets = list()
        else:
            self.container_args_module_args_sets = ["--set"] + set_list
        if job_config.get("module_args") is None:
            self.container_args_module_args_dict = DEFAULT_CONTAINER_ARGS_MODULE_ARGS_DICT.copy()
        else:
            self.container_args_module_args_dict = job_config.get("module_args")
        self.container_args_module_args_dict_as_list = list()
        for k, v in self.container_args_module_args_dict.items():
            if v is None:
                continue
            self.container_args_module_args_dict_as_list.append(k)
            self.container_args_module_args_dict_as_list.append(str(v))
        job_volume_list = job_config.get("volume_list", [])
        metadata = _ensure_manifest_mapping(self.pod_manifest, "metadata", "pod manifest metadata")
        spec = _ensure_manifest_mapping(self.pod_manifest, "spec", "pod manifest spec")
        metadata["name"] = job_config.get("name")
        spec["restartPolicy"] = "Never"
        if self.uses_pod_manifest_template:
            spec["volumes"] = _merge_named_items(spec.get("volumes"), job_volume_list, "pod spec volumes")
        else:
            self.volume_list.extend(job_volume_list)
            spec["containers"] = self.container_list
            spec["volumes"] = self.volume_list
        image_pull_secrets = _normalize_image_pull_secrets(job_config.get("image_pull_secrets"))
        if image_pull_secrets:
            image_pull_secret_refs = [{"name": name} for name in image_pull_secrets]
            if self.uses_pod_manifest_template:
                spec["imagePullSecrets"] = _merge_named_items(
                    spec.get("imagePullSecrets"), image_pull_secret_refs, "pod spec imagePullSecrets"
                )
            else:
                spec["imagePullSecrets"] = image_pull_secret_refs
        security_context = job_config.get("security_context")
        if security_context:
            spec["securityContext"] = security_context

        image = job_config.get("image")
        if not image:
            raise ValueError("job_config must contain a non-empty 'image' key")
        container = self.job_container
        container["image"] = image
        container["name"] = job_config.get("container_name", "nvflare_job")
        container["command"] = [self.python_path]
        container["args"] = (
            self.container_args_python_args_list
            + self.container_args_module_args_dict_as_list
            + self.container_args_module_args_sets
        )
        if self.uses_pod_manifest_template:
            container.setdefault("imagePullPolicy", "Always")
            container["volumeMounts"] = _merge_named_items(
                container.get("volumeMounts"), self.container_volume_mount_list, "container volumeMounts"
            )
        else:
            container["volumeMounts"] = self.container_volume_mount_list
        # resources now always includes ephemeral-storage; GPU limits are merged
        # into the same dict only when requested for the job.
        if job_config.get("resources"):
            container["resources"] = job_config["resources"]
        env_vars = {k: v for k, v in job_config.get("env", {}).items() if str(v)}
        if env_vars:
            env_items = [{"name": k, "value": str(v)} for k, v in env_vars.items()]
            if self.uses_pod_manifest_template:
                container["env"] = _merge_named_items(container.get("env"), env_items, "container env")
            else:
                container["env"] = env_items

    def get_manifest(self):
        return copy.deepcopy(self.pod_manifest)

    def enter_states(self, job_states_to_enter: list):
        starting_time = time.time()
        if not isinstance(job_states_to_enter, (list, tuple)):
            job_states_to_enter = [job_states_to_enter]
        if not all([isinstance(js, JobState) for js in job_states_to_enter]):
            raise ValueError(f"expect job_states_to_enter with valid values, but get {job_states_to_enter}")
        while True:
            if self.terminal_state is not None:
                return False
            pod = self._query_pod()
            if self.terminal_state is not None:
                return False
            pod_phase = self._get_pod_phase(pod)
            now = time.time()
            if self._handle_starting_pod(pod, pod_phase, now=now):
                return False
            job_state = POD_STATE_MAPPING.get(pod_phase, JobState.UNKNOWN)
            if job_state in job_states_to_enter:
                return True
            elif pod_phase in [PodPhase.FAILED.value, PodPhase.SUCCEEDED.value]:  # terminal state
                self.terminal_state = POD_STATE_MAPPING.get(pod_phase, JobState.UNKNOWN)
                self._remove_workspace_job()
                return False
            elif self.timeout is not None and now - starting_time >= self.timeout:
                self._terminate_for_timeout(f"timed out waiting for pod to enter {job_states_to_enter}")
                return False
            time.sleep(POLL_INTERVAL)

    def _remove_workspace_job(self) -> None:
        if self.workspace_transfer and self.workspace_job_id:
            self.workspace_transfer.remove_job(self.workspace_job_id)
            self.workspace_job_id = ""

    def terminate(self):
        from kubernetes.client.rest import ApiException

        try:
            self.api_instance.delete_namespaced_pod(
                name=self.pod_name, namespace=self.namespace, grace_period_seconds=0
            )
            self.terminal_state = JobState.TERMINATED
        except ApiException as e:
            if getattr(e, "status", None) == 404:
                # Expected when terminate() runs as an idempotent cleanup after the
                # pod already exited gracefully (e.g. server abort path where the SJ
                # left on its own before the safety-net terminate fires). Not an
                # event of interest for operators monitoring logs.
                self.logger.debug(
                    f"job {self.job_id} pod {self.pod_name} not found during termination; assuming terminated"
                )
            else:
                self.logger.error(f"failed to terminate job {self.job_id} pod {self.pod_name}: {e}")
            self.terminal_state = JobState.TERMINATED
        except Exception as e:
            self.logger.error(f"unexpected error terminating job {self.job_id} pod {self.pod_name}: {e}")
            self.terminal_state = JobState.TERMINATED
        self._remove_workspace_job()
        return None

    def _terminate_for_timeout(self, reason: str):
        self._terminate_for_exception(reason)

    def _terminate_for_exception(self, reason: str):
        self.logger.warning(f"job {self.job_id} pod {self.pod_name}: {reason}")
        self.terminate()
        self.terminal_return_code = JobReturnCode.EXCEPTION

    def _get_return_code(self, job_state):
        if self.terminal_return_code is not None:
            return self.terminal_return_code
        return JOB_RETURN_CODE_MAPPING.get(job_state)

    def poll(self):
        if self.terminal_state is not None:
            return self._get_return_code(self.terminal_state)
        pod = self._query_pod()
        if self.terminal_state is not None:
            return self._get_return_code(self.terminal_state)
        pod_phase = self._get_pod_phase(pod)
        if self._handle_starting_pod(pod, pod_phase):
            return self._get_return_code(self.terminal_state)
        job_state = POD_STATE_MAPPING.get(pod_phase, JobState.UNKNOWN)
        if job_state in (JobState.SUCCEEDED, JobState.TERMINATED):
            self.terminal_state = job_state
            self._remove_workspace_job()
        return self._get_return_code(job_state)

    def _query_pod(self):
        from kubernetes.client.rest import ApiException

        try:
            return self.api_instance.read_namespaced_pod(name=self.pod_name, namespace=self.namespace)
        except ApiException as e:
            if getattr(e, "status", None) == 404:
                self.logger.info(
                    f"job {self.job_id} pod {self.pod_name} not found during querying; assuming terminated"
                )
                self.terminal_state = JobState.TERMINATED
                self._remove_workspace_job()
            else:
                self.logger.warning(f"failed to query pod for job {self.job_id} pod {self.pod_name}: {e}")
            return None
        except Exception as e:
            self.logger.warning(f"unexpected error querying pod for job {self.job_id} pod {self.pod_name}: {e}")
            return None

    def _query_phase(self):
        pod = self._query_pod()
        if pod is None and self.terminal_state is not None:
            return PodPhase.UNKNOWN.value
        return self._get_pod_phase(pod)

    def _get_pod_phase(self, pod):
        if pod is None:
            return None
        phase = getattr(getattr(pod, "status", None), "phase", None)
        if not phase:
            self.logger.warning(f"pod phase is missing for job {self.job_id} pod {self.pod_name}")
            return None
        return phase

    def _query_state(self):
        pod_phase = self._query_phase()
        return POD_STATE_MAPPING.get(pod_phase, JobState.UNKNOWN)

    def _stuck_in_pending(self, current_phase, now=None):
        if current_phase is None:
            return False
        if current_phase == PodPhase.PENDING.value:
            self._stuck_count += 1
            if self.pending_timeout is None:
                return False
            current_time = time.time() if now is None else now
            if self._pending_since is None:
                self._pending_since = current_time
                self._pending_timer_paused_at = None
            else:
                self._resume_pending_timer(current_time)
            if self.pending_timeout == 0:
                return True
            if current_time - self._pending_since >= self.pending_timeout:
                return True
        else:
            self._reset_pending_timer()
        return False

    def _handle_starting_pod(self, pod, pod_phase, now=None) -> bool:
        self._last_event_query_failed = False
        action, detail = self._classify_starting_pod(pod, pod_phase, now=now)
        if action == PendingPodAction.FAIL:
            self._terminate_for_exception(f"pod startup failure: {detail}")
            return True
        if action == PendingPodAction.WAIT_FOR_RESOURCES:
            if self._stuck_in_pending(pod_phase, now=now):
                self._terminate_for_timeout(f"timed out waiting for CPU/memory/GPU resources: {detail}")
                return True
            return False

        if pod_phase is None:
            self._pause_pending_timer(now)
            return False
        if pod_phase == PodPhase.PENDING.value and self._pending_since is not None and self._last_event_query_failed:
            if not self._pod_is_scheduled(getattr(pod, "status", None)):
                self._pause_pending_timer(now)
                return False
        self._reset_pending_timer()
        return False

    def _pause_pending_timer(self, now=None):
        if self._pending_since is None or self._pending_timer_paused_at is not None:
            return
        self._pending_timer_paused_at = time.time() if now is None else now

    def _resume_pending_timer(self, now=None):
        if self._pending_timer_paused_at is None:
            return
        current_time = time.time() if now is None else now
        paused_duration = max(0, current_time - self._pending_timer_paused_at)
        self._pending_since += paused_duration
        self._pending_timer_paused_at = None

    def _reset_pending_timer(self):
        self._stuck_count = 0
        self._pending_since = None
        self._pending_timer_paused_at = None

    def _classify_starting_pod(self, pod, pod_phase, now=None):
        if pod_phase == PodPhase.UNKNOWN.value:
            return PendingPodAction.FAIL, "pod phase is Unknown"
        if pod_phase != PodPhase.PENDING.value:
            return PendingPodAction.WAIT, ""

        status = getattr(pod, "status", None)
        if self._pod_is_scheduled(status):
            failure = self._get_container_waiting_failure(status)
            if failure:
                return PendingPodAction.FAIL, failure
            failure = self._get_event_failure(ignore_failed_scheduling=True, now=now)
            if failure:
                return PendingPodAction.FAIL, failure
            return PendingPodAction.WAIT, "pod is scheduled and still starting"

        action, detail = self._classify_unscheduled_pod(status)
        if action != PendingPodAction.WAIT:
            return action, detail

        event_action, event_detail = self._classify_unscheduled_events()
        if event_action != PendingPodAction.WAIT:
            return event_action, event_detail

        return PendingPodAction.WAIT, "pod is pending without a scheduler failure"

    def _pod_is_scheduled(self, status) -> bool:
        node_name = getattr(status, "node_name", None)
        if isinstance(node_name, str) and node_name:
            return True
        for condition in self._get_pod_conditions(status):
            if getattr(condition, "type", None) == "PodScheduled" and getattr(condition, "status", None) == "True":
                return True
        return False

    def _classify_unscheduled_pod(self, status):
        for condition in self._get_pod_conditions(status):
            if getattr(condition, "type", None) != "PodScheduled":
                continue
            condition_status = getattr(condition, "status", None)
            if condition_status != "False":
                continue
            reason = getattr(condition, "reason", None)
            message = getattr(condition, "message", None)
            detail = _obj_text(reason, message) or "pod is not scheduled"
            if _is_cpu_memory_gpu_shortage(detail):
                return PendingPodAction.WAIT_FOR_RESOURCES, detail
            if reason == "Unschedulable":
                return PendingPodAction.FAIL, detail
        return PendingPodAction.WAIT, ""

    def _classify_unscheduled_events(self):
        for event in sorted(self._query_pod_events(), key=_event_sort_key, reverse=True):
            reason = getattr(event, "reason", None)
            message = getattr(event, "message", None)
            event_type = getattr(event, "type", None)
            if event_type != "Warning":
                continue
            detail = _obj_text(reason, message) or "pod event reported startup issue"
            if _is_cpu_memory_gpu_shortage(detail):
                return PendingPodAction.WAIT_FOR_RESOURCES, detail
            if reason == "FailedScheduling":
                return PendingPodAction.FAIL, detail
            if reason in _PENDING_FAILURE_EVENT_REASONS:
                return PendingPodAction.FAIL, detail
        return PendingPodAction.WAIT, ""

    def _get_container_waiting_failure(self, status):
        for container_status in self._get_all_container_statuses(status):
            waiting = getattr(getattr(container_status, "state", None), "waiting", None)
            if not waiting:
                continue
            reason = getattr(waiting, "reason", None)
            message = getattr(waiting, "message", None)
            detail = _obj_text(reason, message) or "container is waiting"
            if reason in _PENDING_FAILURE_WAITING_REASONS:
                return detail
        return ""

    def _get_event_failure(self, ignore_failed_scheduling=False, now=None):
        now = time.time() if now is None else now
        for event in sorted(self._query_pod_events(), key=_event_sort_key, reverse=True):
            reason = getattr(event, "reason", None)
            if ignore_failed_scheduling and reason == "FailedScheduling":
                continue
            event_type = getattr(event, "type", None)
            if event_type != "Warning" or reason not in _PENDING_FAILURE_EVENT_REASONS:
                continue
            if not _is_recent_event(event, now, SCHEDULED_EVENT_FAILURE_MAX_AGE):
                continue
            message = getattr(event, "message", None)
            return _obj_text(reason, message) or "pod event reported startup issue"
        return ""

    def _query_pod_events(self):
        from kubernetes.client.rest import ApiException

        self._last_event_query_failed = False
        try:
            resp = self.api_instance.list_namespaced_event(
                namespace=self.namespace,
                field_selector=f"involvedObject.name={self.pod_name}",
            )
        except ApiException as e:
            self._last_event_query_failed = True
            self.logger.warning(f"failed to query events for job {self.job_id} pod {self.pod_name}: {e}")
            return []
        except Exception as e:
            self._last_event_query_failed = True
            self.logger.warning(f"unexpected error querying events for job {self.job_id} pod {self.pod_name}: {e}")
            return []
        items = getattr(resp, "items", None)
        return items if isinstance(items, (list, tuple)) else []

    def _get_pod_conditions(self, status):
        conditions = getattr(status, "conditions", None)
        return conditions if isinstance(conditions, (list, tuple)) else []

    def _get_all_container_statuses(self, status):
        result = []
        for attr_name in ("init_container_statuses", "container_statuses"):
            statuses = getattr(status, attr_name, None)
            if isinstance(statuses, (list, tuple)):
                result.extend(statuses)
        return result

    def wait(self):
        while True:
            if self.terminal_state is not None:
                return
            pod = self._query_pod()
            if self.terminal_state is not None:
                return
            pod_phase = self._get_pod_phase(pod)
            if self._handle_starting_pod(pod, pod_phase):
                return
            job_state = POD_STATE_MAPPING.get(pod_phase, JobState.UNKNOWN)
            if job_state in (JobState.SUCCEEDED, JobState.TERMINATED):
                self.terminal_state = job_state  # persist so poll() stays accurate
                self._remove_workspace_job()
                return
            time.sleep(POLL_INTERVAL)


class K8sJobLauncher(JobLauncherSpec):
    def __init__(
        self,
        config_file_path: str,
        study_data_pvc_file_path: str = None,
        timeout=None,
        namespace=DEFAULT_NAMESPACE,
        pending_timeout=DEFAULT_PENDING_TIMEOUT,
        python_path=None,
        security_context: dict = None,
        ephemeral_storage: str = DEFAULT_EPHEMERAL_STORAGE,
        default_python_path: str = None,
        workspace_mount_path: str = WORKSPACE_MOUNT_PATH,
        image_pull_secrets: list[str] = None,
        study_job_spec_file_path: str = None,
    ):
        super().__init__()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config_file_path = config_file_path
        if study_data_pvc_file_path is not None and (
            not isinstance(study_data_pvc_file_path, str) or not study_data_pvc_file_path
        ):
            raise ValueError("study_data_pvc_file_path must be a non-empty string or None")
        self.study_data_pvc_file_path = study_data_pvc_file_path
        if study_job_spec_file_path is not None and (
            not isinstance(study_job_spec_file_path, str) or not study_job_spec_file_path
        ):
            raise ValueError("study_job_spec_file_path must be a non-empty string or None")
        self.study_job_spec_file_path = study_job_spec_file_path
        self.timeout = timeout
        self.namespace = namespace
        self.pending_timeout = _normalize_pending_timeout(pending_timeout)
        self.default_python_path = default_python_path if default_python_path is not None else python_path
        if self.default_python_path is None:
            self.default_python_path = DEFAULT_PYTHON_PATH
        if not isinstance(self.default_python_path, str) or not self.default_python_path:
            raise ValueError("default_python_path must be a non-empty string")
        self.security_context = security_context
        if not isinstance(ephemeral_storage, str) or not ephemeral_storage:
            raise ValueError("ephemeral_storage must be a non-empty string")
        self.ephemeral_storage = ephemeral_storage
        if not isinstance(workspace_mount_path, str) or not workspace_mount_path:
            raise ValueError("workspace_mount_path must be a non-empty string")
        self.workspace_mount_path = workspace_mount_path
        self.image_pull_secrets = _normalize_image_pull_secrets(image_pull_secrets)
        self.study_data_pvc_dict = None
        self.study_job_spec_dict = None
        self.pod_manifest_template_dict = {}
        self.core_v1 = None

    def _get_pod_manifest_template(self, study: str):
        if not self.study_job_spec_file_path or not study:
            return None
        if self.study_job_spec_dict is None:
            self.study_job_spec_dict = load_study_job_spec_file(self.study_job_spec_file_path, logger=self.logger)
        pod_spec_file_path = resolve_study_job_spec_path(
            self.study_job_spec_dict, study, self.study_job_spec_file_path, logger=self.logger
        )
        if not pod_spec_file_path:
            return None
        if pod_spec_file_path not in self.pod_manifest_template_dict:
            self.pod_manifest_template_dict[pod_spec_file_path] = load_pod_spec_file(pod_spec_file_path)
        return self.pod_manifest_template_dict[pod_spec_file_path]

    def _ensure_startup_secret(self, site_name: str, startup_dir: str) -> str:
        """Create or update a k8s Secret containing the site startup kit.

        Returns the Secret name.
        """
        from kubernetes.client.rest import ApiException

        secret_name = f"nvflare-startup-{site_name_to_rfc1123(site_name)}"
        data = {}
        if os.path.isdir(startup_dir):
            for fname in os.listdir(startup_dir):
                if not _keep_startup_file(fname):
                    continue
                fpath = os.path.join(startup_dir, fname)
                if os.path.isfile(fpath):
                    with open(fpath, "rb") as f:
                        data[fname] = base64.b64encode(f.read()).decode()

        secret_body = {
            "apiVersion": "v1",
            "kind": "Secret",
            "metadata": {"name": secret_name, "namespace": self.namespace},
            "type": "Opaque",
            "data": data,
        }
        try:
            self.core_v1.create_namespaced_secret(namespace=self.namespace, body=secret_body)
            self.logger.debug("Created startup Secret %s", secret_name)
        except ApiException as e:
            if getattr(e, "status", None) == 409:
                self.core_v1.replace_namespaced_secret(name=secret_name, namespace=self.namespace, body=secret_body)
                self.logger.debug("Updated startup Secret %s", secret_name)
            else:
                raise
        return secret_name

    def _replace_pod_manifest_template_namespace(self, pod_manifest: dict) -> None:
        metadata = _ensure_manifest_mapping(pod_manifest, "metadata", "pod manifest metadata")
        if "namespace" not in metadata:
            return

        template_namespace = metadata["namespace"]
        if template_namespace == self.namespace:
            return

        metadata["namespace"] = self.namespace
        self.logger.warning(
            "job pod is launched in namespace '%s' instead of metadata.namespace '%s'",
            self.namespace,
            template_namespace,
        )

    def launch_job(self, job_meta: dict, fl_ctx: FLContext) -> JobHandleSpec:
        if self.core_v1 is None:
            from kubernetes import config
            from kubernetes.client import Configuration
            from kubernetes.client.api import core_v1_api

            try:
                if self.config_file_path:
                    config.load_kube_config(self.config_file_path)
                else:
                    config.load_incluster_config()
                c = Configuration().get_default_copy()
            except AttributeError:
                c = Configuration()
                c.assert_hostname = False
            Configuration.set_default(c)
            self.core_v1 = core_v1_api.CoreV1Api()
        site_name = fl_ctx.get_identity_name()
        raw_job_id = job_meta.get(JobConstants.JOB_ID)
        if not raw_job_id:
            raise RuntimeError(f"missing {JobConstants.JOB_ID} in job_meta")
        job_id = uuid4_to_rfc1123(raw_job_id)
        pod_name = job_pod_name(job_id, site_name)
        workspace_obj = fl_ctx.get_prop(FLContextKey.WORKSPACE_OBJECT)
        if workspace_obj is None:
            raise RuntimeError(f"missing {FLContextKey.WORKSPACE_OBJECT} in FLContext")
        app_custom_folder = workspace_obj.get_app_custom_dir(raw_job_id)
        args = fl_ctx.get_prop(FLContextKey.ARGS)
        if args is None:
            raise RuntimeError(f"missing {FLContextKey.ARGS} in FLContext")
        k8s_spec = get_job_launcher_spec(job_meta, site_name, "k8s")
        job_pending_timeout = k8s_spec["pending_timeout"] if "pending_timeout" in k8s_spec else self.pending_timeout
        try:
            job_pending_timeout = _normalize_pending_timeout(
                job_pending_timeout, f"launcher_spec['{site_name}']['k8s']['pending_timeout']"
            )
        except ValueError as e:
            raise RuntimeError(str(e)) from e
        job_image = k8s_spec.get("image")
        job_ephemeral_storage = k8s_spec.get("ephemeral_storage")
        if job_ephemeral_storage is None:
            job_ephemeral_storage = self.ephemeral_storage
        if not isinstance(job_ephemeral_storage, str) or not job_ephemeral_storage:
            raise RuntimeError(f"launcher_spec['{site_name}']['k8s']['ephemeral_storage'] must be a non-empty string")
        if not job_image:
            raise RuntimeError(
                f"K8sJobLauncher is configured for site '{site_name}' but no job image "
                f"was specified in meta.json for this site. "
                f"Set launcher_spec['{site_name}']['k8s']['image'] (preferred), "
                f"launcher_spec['default']['k8s']['image'] (shared default), "
                f"or resource_spec['{site_name}']['k8s']['image'] (legacy)."
            )
        study = job_meta.get(JobMetaKey.STUDY.value)
        pod_manifest_template = self._get_pod_manifest_template(study)
        data_mounts = []
        if should_mount_study_data(study) and self.study_data_pvc_file_path:
            if self.study_data_pvc_dict is None:
                self.study_data_pvc_dict = load_study_data_file(self.study_data_pvc_file_path, logger=self.logger)
            data_mounts = resolve_study_dataset_mounts(
                self.study_data_pvc_dict, study, self.study_data_pvc_file_path, logger=self.logger
            )
            if pod_manifest_template is not None and data_mounts:
                self.logger.warning(
                    "study_job_spec_file_path '%s' is used for study '%s'; matching entries from "
                    "study_data_pvc_file_path '%s' will be added as extra volume mounts",
                    self.study_job_spec_file_path,
                    study,
                    self.study_data_pvc_file_path,
                )
        site_resources = (job_meta.get(JobMetaKey.RESOURCE_SPEC.value) or {}).get(site_name) or {}
        flat_gpu_count = (
            0
            if any(k in site_resources for k in ("process", "docker", "k8s"))
            else site_resources.get("num_of_gpus", 0)
        )
        job_resource = k8s_spec["num_of_gpus"] if "num_of_gpus" in k8s_spec else flat_gpu_count
        job_args = fl_ctx.get_prop(FLContextKey.JOB_PROCESS_ARGS)
        if not job_args:
            raise RuntimeError(f"missing {FLContextKey.JOB_PROCESS_ARGS} in FLContext")

        exe_module_entry = job_args.get(JobProcessArgs.EXE_MODULE)
        if not exe_module_entry:
            raise RuntimeError(f"missing {JobProcessArgs.EXE_MODULE} in {FLContextKey.JOB_PROCESS_ARGS}")
        _, job_cmd = exe_module_entry

        workspace_root = args.workspace
        env = {}
        if app_custom_folder:
            workspace_root_abs = os.path.abspath(workspace_root)
            custom_folder_abs = os.path.abspath(app_custom_folder)
            if os.path.commonpath([workspace_root_abs, custom_folder_abs]) != workspace_root_abs:
                raise RuntimeError(f"custom folder {app_custom_folder} is not under workspace {workspace_root}")
            env["PYTHONPATH"] = os.path.join(
                self.workspace_mount_path, os.path.relpath(custom_folder_abs, workspace_root_abs)
            )

        startup_dir = workspace_obj.get_startup_kit_dir()
        engine = fl_ctx.get_engine()
        owner_cell = getattr(engine, "cell", None) if engine else None
        if owner_cell is None:
            raise RuntimeError("missing parent CellNet cell for workspace transfer")

        workspace_transfer = WorkspaceTransferManager.get_or_create(owner_cell)
        workspace_transfer_token = workspace_transfer.add_job(raw_job_id, workspace_root)
        try:
            startup_secret_name = self._ensure_startup_secret(site_name, startup_dir)

            env[ENV_WORKSPACE_OWNER_FQCN] = workspace_transfer.owner_fqcn
            env[ENV_WORKSPACE_TRANSFER_TOKEN] = workspace_transfer_token

            volume_list = [
                {"name": "workspace-job", "emptyDir": {"sizeLimit": job_ephemeral_storage}},
                {"name": "startup-kit", "secret": {"secretName": startup_secret_name}},
            ]
            volume_mount_list = [
                {"name": "workspace-job", "mountPath": self.workspace_mount_path},
                {
                    "name": "startup-kit",
                    "mountPath": os.path.join(self.workspace_mount_path, "startup"),
                    "readOnly": True,
                },
            ]
            for dataset_mount in data_mounts:
                volume_name = study_dataset_volume_name(dataset_mount.study, dataset_mount.dataset)
                volume_list.append({"name": volume_name, "persistentVolumeClaim": {"claimName": dataset_mount.source}})
                volume_mount_list.append(
                    {
                        "name": volume_name,
                        "mountPath": dataset_mount.mount_path,
                        "readOnly": dataset_mount.read_only,
                    }
                )

            job_config = {
                "name": pod_name,
                "image": job_image,
                "container_name": f"container-{job_id}",
                "command": job_cmd,
                "volume_mount_list": volume_mount_list,
                "volume_list": volume_list,
                "module_args": self.get_module_args(job_id, fl_ctx),
                "env": env,
            }
            if self.image_pull_secrets:
                job_config["image_pull_secrets"] = self.image_pull_secrets
            if args is not None and getattr(args, "set", None) is not None:
                job_config.update({"set_list": args.set})
            resources = {
                "requests": {"ephemeral-storage": job_ephemeral_storage},
                "limits": {"ephemeral-storage": job_ephemeral_storage},
            }
            for key in ("cpu", "memory"):
                limit_val = k8s_spec.get(key)
                # cpu_request / memory_request allow request < limit; when absent,
                # request mirrors the limit so admission webhooks that require
                # explicit cpu/memory requests (e.g. AKS deployment safeguards) pass.
                request_val = k8s_spec.get(f"{key}_request", limit_val)
                if limit_val:
                    resources["limits"][key] = limit_val
                if request_val:
                    resources["requests"][key] = request_val
            if job_resource:
                resources["limits"]["nvidia.com/gpu"] = job_resource
                resources["requests"]["nvidia.com/gpu"] = job_resource
            job_config["resources"] = resources
            if self.security_context:
                job_config["security_context"] = self.security_context
            python_path = k8s_spec.get("python_path", self.default_python_path)
            if not isinstance(python_path, str) or not python_path:
                raise RuntimeError(f"launcher_spec['{site_name}']['k8s']['python_path'] must be a non-empty string")
            job_handle = K8sJobHandle(
                job_id,
                self.core_v1,
                job_config,
                namespace=self.namespace,
                timeout=self.timeout,
                pending_timeout=job_pending_timeout,
                python_path=python_path,
                workspace_transfer=workspace_transfer,
                workspace_job_id=raw_job_id,
                pod_name=pod_name,
                pod_manifest_template=pod_manifest_template,
            )
            pod_manifest = job_handle.get_manifest()
            if pod_manifest_template is not None:
                self._replace_pod_manifest_template_namespace(pod_manifest)
            self.logger.debug(
                "launch job with k8s_launcher: pod_name=%s namespace=%s image=%s",
                pod_manifest["metadata"]["name"],
                self.namespace,
                job_image,
            )
            self.core_v1.create_namespaced_pod(body=pod_manifest, namespace=self.namespace)
        except Exception as e:
            workspace_transfer.remove_job(raw_job_id)
            if "job_handle" in locals():
                self.logger.error(f"failed to launch job {job_id}: {e}")
                job_handle.terminal_state = JobState.TERMINATED
                job_handle.terminal_return_code = JobReturnCode.EXCEPTION
                return job_handle
            raise
        try:
            entered_running = job_handle.enter_states([JobState.RUNNING])
        except BaseException:
            job_handle.terminate()
            raise
        if not entered_running:
            self.logger.warning(f"unable to enter running phase {job_id}")
        return job_handle

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.BEFORE_JOB_LAUNCH:
            add_launcher(self, fl_ctx)

    @abstractmethod
    def get_module_args(self, job_id, fl_ctx: FLContext):
        """To get the args to run the launcher

        Args:
            job_id: run job_id
            fl_ctx: FLContext

        Returns:

        """
        pass


def _job_args_dict(job_args: dict, arg_names: list) -> dict:
    result = {}
    for name in arg_names:
        e = job_args.get(name)
        if e is None:
            continue

        n, v = e
        result[n] = v
    return result


class ClientK8sJobLauncher(K8sJobLauncher):
    def get_module_args(self, _job_id, fl_ctx: FLContext):
        job_args = fl_ctx.get_prop(FLContextKey.JOB_PROCESS_ARGS)
        if not job_args:
            raise RuntimeError(f"missing {FLContextKey.JOB_PROCESS_ARGS} in FLContext")

        return _job_args_dict(job_args, get_client_job_args(False, False))


class ServerK8sJobLauncher(K8sJobLauncher):
    def get_module_args(self, _job_id, fl_ctx: FLContext):
        job_args = fl_ctx.get_prop(FLContextKey.JOB_PROCESS_ARGS)
        if not job_args:
            raise RuntimeError(f"missing {FLContextKey.JOB_PROCESS_ARGS} in FLContext")

        return _job_args_dict(job_args, get_server_job_args(False, False))
