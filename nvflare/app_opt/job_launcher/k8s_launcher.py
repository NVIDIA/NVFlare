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
import json
import logging
import os
import re
import threading
import time
import uuid
from abc import abstractmethod
from datetime import datetime
from enum import Enum

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import FLContextKey, JobConstants
from nvflare.apis.fl_context import FLContext
from nvflare.apis.job_def import JobMetaKey
from nvflare.apis.job_launcher_spec import (
    JobHandleSpec,
    JobLauncherSpec,
    JobProcessArgs,
    JobProcessEnv,
    JobReturnCode,
    add_launcher,
    publish_job_handle,
)
from nvflare.app_opt.job_launcher.k8s_pod import (
    build_pod_manifest,
    ensure_manifest_mapping,
    get_container_waiting_failure,
    normalize_image_pull_secrets,
    pod_shutdown_confirmed,
)
from nvflare.app_opt.job_launcher.study_data import (
    load_study_data_file,
    resolve_study_dataset_mounts,
    should_mount_study_data,
)
from nvflare.app_opt.job_launcher.study_runtime import (
    load_study_runtime_file,
    resolve_study_runtime,
    study_runtime_file_path,
)
from nvflare.app_opt.job_launcher.workspace_cell_transfer import (
    ENV_PERSISTENT_WORKSPACE,
    ENV_WORKSPACE_OWNER_FQCN,
    ENV_WORKSPACE_TRANSFER_TOKEN,
    WorkspaceTransferManager,
    workspace_download_complete,
)
from nvflare.fuel.common.exit_codes import ProcessExitCode
from nvflare.fuel.utils.config_service import ConfigService
from nvflare.job_config.task_execution import get_client_execution_lifetime, reject_job_gpu_reservation
from nvflare.private.fed.utils.job_cert_utils import job_startup_files, read_job_cert, require_job_cert
from nvflare.utils.job_launcher_utils import (
    get_cj_resource_spec,
    get_client_job_args,
    get_credential_env,
    get_job_launcher_spec,
    get_portable_resource_spec,
    get_server_job_args,
    has_cj_resource_spec,
    validate_cj_resource_spec,
)

ENV_CJ_POD_IP = "NVFLARE_K8S_CJ_POD_IP"

_STARTUP_FEDERATION_LABEL = "nvflare.nvidia.com/startup-federation"


def _startup_federation_identity(data):
    """A label-safe SHA-256 CA fingerprint, or payload identity for unsecured kits."""
    if data.get("rootCA.pem"):
        from cryptography import x509
        from cryptography.hazmat.primitives import hashes

        certificate = x509.load_pem_x509_certificate(base64.b64decode(data["rootCA.pem"], validate=True))
        digest = certificate.fingerprint(hashes.SHA256())
    else:
        digest = hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":")).encode()).digest()
    return base64.b32encode(digest).decode("ascii").rstrip("=").lower()


def validate_dns_name(name, value):
    if not isinstance(value, str) or len(value) > 63 or not re.fullmatch(r"[a-z0-9]([-a-z0-9]*[a-z0-9])?", value):
        raise ValueError(f"{name} must be a DNS label")
    return value


def validate_startup_secret_name(value):
    # Secret names use DNS subdomains (up to 253 characters), not the shorter
    # label limit used by namespace and service-account names.
    label = r"[a-z0-9](?:[-a-z0-9]*[a-z0-9])?"
    if not isinstance(value, str) or len(value) > 253 or not re.fullmatch(rf"{label}(?:\.{label})*", value):
        raise ValueError("startup_secret_name must be a DNS subdomain of at most 253 characters")
    return value


def job_workspace_subpath(job_id, site_name):
    """Separate each site's job workspace without embedding untrusted path components."""
    digest = hashlib.sha256(json.dumps([job_id, site_name]).encode()).hexdigest()
    return f"nvflare-tasks/{digest}"


def task_security_context(uid, gid):
    for name, value in (("run_as_user", uid), ("run_as_group", gid)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be an explicit positive numeric ID shared by CJ and workers")
    return {"runAsUser": uid, "runAsGroup": gid, "runAsNonRoot": True, "fsGroup": gid}


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

DEFAULT_NAMESPACE = "default"
DEFAULT_PENDING_TIMEOUT = 120
DEFAULT_PYTHON_PATH = "/usr/local/bin/python"
POLL_INTERVAL = 1
SCHEDULED_EVENT_FAILURE_MAX_AGE = 60


WORKSPACE_MOUNT_PATH = "/var/tmp/nvflare/workspace"
DEFAULT_EPHEMERAL_STORAGE = "1Gi"

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
# startup/ are dropped to shrink the Secret (job_startup_files() already withholds
# private keys). local/ is bundled whole with each job workspace so job resource
# files and local custom code keep working.
_STARTUP_KEEP_SUFFIXES = (".crt", ".pem", ".json")


def _keep_startup_file(fname: str) -> bool:
    return fname.endswith(_STARTUP_KEEP_SUFFIXES)


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


def is_cpu_memory_gpu_shortage(message: str) -> bool:
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


def study_secret_volume_name(study: str, name: str) -> str:
    return site_name_to_rfc1123(f"secret-{study}-{name}", max_length=63)


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
        credential_secret_name: str = None,
    ):
        super().__init__()
        self.job_id = job_id
        self.pod_name = pod_name if pod_name is not None else job_id
        self.timeout = timeout
        self.terminal_state = None
        self.terminal_return_code = None
        self.task_owner = None
        self.pod_uid = None
        self.workspace_reservation = None
        self._restart_ready = False
        self._termination_requested = False
        self.workspace_transfer = workspace_transfer
        self.workspace_job_id = workspace_job_id
        self._raw_job_id = workspace_job_id
        self.credential_secret_name = credential_secret_name
        self.api_instance = api_instance
        self.namespace = namespace
        self.pending_timeout = _normalize_pending_timeout(pending_timeout)
        self.python_path = python_path
        self.pod_manifest = build_pod_manifest(job_config, python_path, pod_manifest_template)
        self._job_container_name = job_config.get("container_name", "nvflare_job")
        self._stuck_count = 0
        self._pending_since = None
        # Kept for diagnostics only; unit is seconds, not poll iterations like _stuck_count.
        self._pending_timeout_secs = self.pending_timeout
        self._last_event_query_failed = False
        self._pending_timer_paused_at = None
        self.logger = logging.getLogger(self.__class__.__name__)

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
                self._record_terminal_state(pod, pod_phase)
                return False
            elif self.timeout is not None and now - starting_time >= self.timeout:
                self._terminate_for_timeout(f"timed out waiting for pod to enter {job_states_to_enter}")
                return False
            time.sleep(POLL_INTERVAL)

    def _release_job_resources(self) -> None:
        if self.task_owner is not None:
            self.task_owner.release()
        if self.workspace_transfer and self.workspace_job_id:
            self.workspace_transfer.remove_job(self.workspace_job_id)
            self.workspace_job_id = ""
        self._delete_credential_secret()

    def _delete_credential_secret(self) -> None:
        """Delete the per-job credential Secret once the job is terminal.

        Completed pods are not deleted, so ownerReference GC alone would leave the
        Secret alive after a successful job; GC remains the backstop for paths where
        the handle never observes a terminal state (e.g. parent crash).
        """
        if not self.credential_secret_name:
            return
        from kubernetes.client.rest import ApiException

        try:
            self.api_instance.delete_namespaced_secret(name=self.credential_secret_name, namespace=self.namespace)
        except ApiException as e:
            if getattr(e, "status", None) != 404:
                self.logger.warning("failed to delete credential Secret")
        except Exception:
            self.logger.warning("failed to delete credential Secret")
        self.credential_secret_name = None

    def terminate(self):
        self._termination_requested = True
        # Fence task admission before deleting CJ. CP retains worker cleanup.
        if self.task_owner is not None:
            self.task_owner.close()
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
        self._release_job_resources()
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

    def _record_terminal_state(self, pod, pod_phase):
        self.terminal_state = POD_STATE_MAPPING.get(pod_phase, JobState.UNKNOWN)
        if self.terminal_state == JobState.TERMINATED:
            job_container_name = self._job_container_name
            statuses = getattr(getattr(pod, "status", None), "container_statuses", None)
            if not isinstance(statuses, (list, tuple)):
                statuses = []
            for container_status in statuses:
                if getattr(container_status, "name", None) != job_container_name:
                    continue
                terminated = getattr(getattr(container_status, "state", None), "terminated", None)
                exit_code = getattr(terminated, "exit_code", None)
                if exit_code in (
                    ProcessExitCode.EXCEPTION,
                    ProcessExitCode.UNSAFE_COMPONENT,
                    ProcessExitCode.CONFIG_ERROR,
                ):
                    self.terminal_return_code = exit_code
                break
        self._release_job_resources()

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
            self._record_terminal_state(pod, pod_phase)
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
                self._release_job_resources()
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
            if is_cpu_memory_gpu_shortage(detail):
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
            if is_cpu_memory_gpu_shortage(detail):
                return PendingPodAction.WAIT_FOR_RESOURCES, detail
            if reason == "FailedScheduling":
                return PendingPodAction.FAIL, detail
            if reason in _PENDING_FAILURE_EVENT_REASONS:
                return PendingPodAction.FAIL, detail
        return PendingPodAction.WAIT, ""

    def _get_container_waiting_failure(self, status):
        return get_container_waiting_failure(status)

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
                self._record_terminal_state(pod, pod_phase)
                return
            time.sleep(POLL_INTERVAL)

    def can_restart(self):
        if (
            self._termination_requested
            or self.terminal_state != JobState.TERMINATED
            or not self.pod_uid
            or not self.workspace_reservation
        ):
            return False
        if self._restart_ready:
            return True
        directory, _, _ = self.workspace_reservation
        if not workspace_download_complete(directory, self._raw_job_id):
            return False
        # Reuse the task backend's cleanup bound; resident CJs get the same
        # default. A timeout never authorizes overlapping checkpoint writers.
        timeout = self.task_owner.backend.cleanup_timeout if self.task_owner is not None else 60.0
        deadline = time.monotonic() + timeout
        while not self._termination_requested:
            try:
                pod = self.api_instance.read_namespaced_pod(
                    name=self.pod_name, namespace=self.namespace, _request_timeout=10.0
                )
                if pod.metadata.uid != self.pod_uid or pod.status.reason in ("NodeLost", "Shutdown"):
                    return False
                if pod_shutdown_confirmed(pod) and (self.task_owner is None or self.task_owner.wait_for_settlement(0)):
                    self._restart_ready = True
                    return True
            except Exception as error:
                self.logger.warning("cannot confirm CJ shutdown for recovery: %s", type(error).__name__)
                return False
            if time.monotonic() >= deadline:
                self.logger.warning("CJ recovery cleanup timed out for job %s", self.job_id)
                return False
            time.sleep(POLL_INTERVAL)
        return False


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
        startup_secret_name: str = None,
    ):
        super().__init__()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config_file_path = config_file_path
        if study_data_pvc_file_path is not None and (
            not isinstance(study_data_pvc_file_path, str) or not study_data_pvc_file_path
        ):
            raise ValueError("study_data_pvc_file_path must be a non-empty string or None")
        self.study_data_pvc_file_path = study_data_pvc_file_path
        self.timeout = timeout
        self.namespace = namespace
        self.startup_secret_name = (
            None if startup_secret_name is None else validate_startup_secret_name(startup_secret_name)
        )
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
        self.image_pull_secrets = normalize_image_pull_secrets(image_pull_secrets)
        self.study_data_pvc_dict = None
        self.core_v1 = None

    def _resolve_study_runtime(self, workspace_root: str, study):
        runtime_file = study_runtime_file_path(workspace_root)
        if not os.path.exists(runtime_file):
            return None
        legacy_files = {os.path.join(workspace_root, "local", "study_data.yaml")}
        if self.study_data_pvc_file_path:
            legacy_files.add(self.study_data_pvc_file_path)
        conflicts = sorted(path for path in legacy_files if os.path.exists(path))
        if conflicts:
            raise RuntimeError(
                f"study runtime file '{runtime_file}' cannot be combined with the legacy study data "
                f"file(s) {conflicts}; migrate all studies to study_runtime.yaml and delete the v1 file."
            )
        runtime_map = load_study_runtime_file(runtime_file, launcher_mode="k8s", logger=self.logger)
        return resolve_study_runtime(runtime_map, study, runtime_file, logger=self.logger)

    def _create_or_replace_secret(self, secret_name: str, contents: dict, *, federation_identity: str = None) -> str:
        """Upsert a Secret, checking and version-fencing startup kit ownership."""
        from kubernetes.client.rest import ApiException

        secret_body = {
            "apiVersion": "v1",
            "kind": "Secret",
            "metadata": {"name": secret_name, "namespace": self.namespace},
            "type": "Opaque",
            **contents,
        }
        if federation_identity is not None:
            secret_body["metadata"]["labels"] = {_STARTUP_FEDERATION_LABEL: federation_identity}
        try:
            self.core_v1.create_namespaced_secret(namespace=self.namespace, body=secret_body)
            self.logger.debug("created Secret")
        except ApiException as e:
            if getattr(e, "status", None) != 409:
                raise
            if federation_identity is not None:
                self._replace_startup_secret(secret_body, federation_identity)
                return secret_name
            try:
                self.core_v1.replace_namespaced_secret(name=secret_name, namespace=self.namespace, body=secret_body)
                self.logger.debug("replaced Secret")
            except ApiException as e2:
                # 409 -> (GC deletes the old Secret) -> replace 404: recreate.
                if getattr(e2, "status", None) != 404:
                    raise
                secret_body["metadata"].pop("resourceVersion", None)
                self.core_v1.create_namespaced_secret(namespace=self.namespace, body=secret_body)
                self.logger.debug("recreated Secret after concurrent deletion")
        return secret_name

    def _replace_startup_secret(self, body, federation_identity):
        """Retry concurrent same-kit launches while rechecking ownership each time."""
        from kubernetes.client.rest import ApiException

        name = body["metadata"]["name"]
        for attempt in range(3):
            try:
                try:
                    existing = self.core_v1.read_namespaced_secret(name=name, namespace=self.namespace)
                except ApiException as error:
                    if getattr(error, "status", None) != 404:
                        raise
                    body["metadata"].pop("resourceVersion", None)
                    self.core_v1.create_namespaced_secret(namespace=self.namespace, body=body)
                    return
                # Legacy Secrets can be upgraded only after matching their CA
                # (or unsecured payload). Never trust a name match alone.
                label = (existing.metadata.labels or {}).get(_STARTUP_FEDERATION_LABEL)
                identity = label if label is not None else _startup_federation_identity(existing.data or {})
                if identity != federation_identity:
                    raise ValueError(
                        f"startup Secret {name!r} belongs to a different federation; "
                        "configure a unique startup_secret_name in this namespace"
                    )
                if label == federation_identity and existing.data == body["data"]:
                    return
                version = existing.metadata.resource_version
                if not isinstance(version, str) or not version:
                    raise ValueError("startup Secret ownership cannot be confirmed without resourceVersion")
                body["metadata"]["resourceVersion"] = version
                self.core_v1.replace_namespaced_secret(name=name, namespace=self.namespace, body=body)
                return
            except ApiException as error:
                if getattr(error, "status", None) not in (404, 409) or attempt == 2:
                    raise

    def _ensure_startup_secret(self, site_name: str, startup_dir: str) -> str:
        """Create or update a k8s Secret containing the site startup kit.

        Returns the Secret name.
        """
        data = {}
        if os.path.isdir(startup_dir):
            for fname in job_startup_files(startup_dir):
                if not _keep_startup_file(fname):
                    continue
                with open(os.path.join(startup_dir, fname), "rb") as f:
                    data[fname] = base64.b64encode(f.read()).decode()

        # Deployment staging owns nvflare-startup-<site> and projects encoded
        # keys into CP's full kit. Jobs need a separate public-only kit: never
        # replace CP's mounted Secret or copy its private keys into a job Pod.
        name = self.startup_secret_name or f"nvflare-job-startup-{site_name_to_rfc1123(site_name)}"
        return self._create_or_replace_secret(
            name, {"data": data}, federation_identity=_startup_federation_identity(data)
        )

    def _ensure_job_credential_secret(self, pod_name: str, credential_env: dict) -> str:
        """Create (or replace) the per-job Secret carrying bootstrap credentials.

        Lifecycle: the job handle deletes it when the job reaches a terminal state; a
        launch failure before pod creation deletes it immediately; the pod ownerReference
        set after pod creation makes Kubernetes GC the backstop if the parent dies first.
        """
        return self._create_or_replace_secret(f"nvflare-cred-{pod_name}", {"stringData": credential_env})

    def _delete_secret(self, secret_name: str) -> None:
        from kubernetes.client.rest import ApiException

        try:
            self.core_v1.delete_namespaced_secret(name=secret_name, namespace=self.namespace)
        except ApiException as e:
            if getattr(e, "status", None) != 404:
                self.logger.warning("failed to delete Secret")
        except Exception:
            self.logger.warning("failed to delete Secret")

    def _own_credential_secret(self, secret_name: str, pod_name: str, pod) -> None:
        """Patch the credential Secret with an ownerReference to its pod so K8s GC deletes it."""
        try:
            owner = {"apiVersion": "v1", "kind": "Pod", "name": pod_name, "uid": pod.metadata.uid}
            self.core_v1.patch_namespaced_secret(
                name=secret_name, namespace=self.namespace, body={"metadata": {"ownerReferences": [owner]}}
            )
        except Exception:
            self.logger.warning(
                "failed to set ownerReference on credential Secret; "
                "it will be deleted when the job handle reaches a terminal state"
            )

    def _replace_pod_manifest_template_namespace(self, pod_manifest: dict) -> None:
        metadata = ensure_manifest_mapping(pod_manifest, "metadata", "pod manifest metadata")
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
        execution_lifetime = self.get_execution_lifetime(job_meta, fl_ctx)
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
        previous = fl_ctx.get_prop(FLContextKey.JOB_RESTART_HANDLE)
        if previous is not None:
            if not isinstance(previous, K8sJobHandle) or previous.job_id != job_id or not previous.can_restart():
                raise ValueError("Kubernetes recovery requires the confirmed previous job handle")
            # Completed Pods may remain for diagnostics; each execution and its
            # credential Secret need a new name while the logical job stays put.
            pod_name = f"{pod_name[:50].rstrip('-')}-{uuid.uuid4().hex[:12]}"
        workspace_obj = fl_ctx.get_prop(FLContextKey.WORKSPACE_OBJECT)
        if workspace_obj is None:
            raise RuntimeError(f"missing {FLContextKey.WORKSPACE_OBJECT} in FLContext")
        app_custom_folder = workspace_obj.get_app_custom_dir(raw_job_id)
        args = fl_ctx.get_prop(FLContextKey.ARGS)
        if args is None:
            raise RuntimeError(f"missing {FLContextKey.ARGS} in FLContext")
        k8s_spec = get_job_launcher_spec(job_meta, site_name, "k8s")
        portable_spec = get_portable_resource_spec(job_meta, site_name)
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
        study = job_meta.get(JobMetaKey.STUDY.value)
        workspace_root = args.workspace
        study_runtime = self._resolve_study_runtime(workspace_root, study)
        if not job_image and study_runtime is not None:
            # job-supplied image wins; the study's container.image is the site default
            job_image = study_runtime.container_image
        if not job_image:
            raise RuntimeError(
                f"K8sJobLauncher is configured for site '{site_name}' but no job image "
                f"was specified in meta.json for this site. "
                f"Set launcher_spec['{site_name}']['k8s']['image'] (preferred), "
                f"launcher_spec['default']['k8s']['image'] (shared default), "
                f"resource_spec['{site_name}']['k8s']['image'] (legacy), "
                f"or studies.<study>.container.image in local/study_runtime.yaml (site default)."
            )
        if study_runtime is not None:
            pod_manifest_template = study_runtime.pod_template
            data_mounts = study_runtime.datasets
        else:
            pod_manifest_template = None
            data_mounts = []
            if should_mount_study_data(study) and self.study_data_pvc_file_path:
                if self.study_data_pvc_dict is None:
                    self.study_data_pvc_dict = load_study_data_file(self.study_data_pvc_file_path, logger=self.logger)
                data_mounts = resolve_study_dataset_mounts(
                    self.study_data_pvc_dict, study, self.study_data_pvc_file_path, logger=self.logger
                )
        native_resources = k8s_spec
        if execution_lifetime == "task":
            # Existing fields describe compute; none leak into the persistent CJ.
            native_resources = {}
            portable_spec = self.get_cj_resources(job_meta, site_name)
        job_resource = native_resources.get("num_of_gpus", portable_spec.get("num_of_gpus", 0))
        job_args = fl_ctx.get_prop(FLContextKey.JOB_PROCESS_ARGS)
        if not job_args:
            raise RuntimeError(f"missing {FLContextKey.JOB_PROCESS_ARGS} in FLContext")

        exe_module_entry = job_args.get(JobProcessArgs.EXE_MODULE)
        if not exe_module_entry:
            raise RuntimeError(f"missing {JobProcessArgs.EXE_MODULE} in {FLContextKey.JOB_PROCESS_ARGS}")
        _, job_cmd = exe_module_entry

        env = dict(study_runtime.env) if study_runtime is not None else {}
        if app_custom_folder:
            workspace_root_abs = os.path.abspath(workspace_root)
            custom_folder_abs = os.path.abspath(app_custom_folder)
            if os.path.commonpath([workspace_root_abs, custom_folder_abs]) != workspace_root_abs:
                raise RuntimeError(f"custom folder {app_custom_folder} is not under workspace {workspace_root}")
            env["PYTHONPATH"] = os.path.join(
                self.workspace_mount_path, os.path.relpath(custom_folder_abs, workspace_root_abs)
            )

        startup_dir = workspace_obj.get_startup_kit_dir()
        run_dir = workspace_obj.get_run_dir(raw_job_id)
        job_cert = read_job_cert(run_dir) if require_job_cert(fl_ctx, run_dir) else None
        engine = fl_ctx.get_engine()
        owner_cell = getattr(engine, "cell", None) if engine else None
        if owner_cell is None:
            raise RuntimeError("missing parent CellNet cell for workspace transfer")

        workspace_transfer = WorkspaceTransferManager.get_or_create(owner_cell)
        workspace_transfer_token = workspace_transfer.add_job(raw_job_id, workspace_root)
        credential_secret_name = None
        created_pod = None
        pod_create_attempted = False
        workspace_reservations = []
        try:
            startup_secret_name = self._ensure_startup_secret(site_name, startup_dir)

            # The transfer token rides the credential Secret too: a literal env value
            # would be readable in the pod object by anyone with pods/get.
            credential_env = get_credential_env(job_args)
            credential_env[ENV_WORKSPACE_TRANSFER_TOKEN] = workspace_transfer_token
            if job_cert is not None:
                # the pod's bootstrap cell needs the credential before the run dir is downloaded
                credential_env[JobProcessEnv.JOB_CERT] = job_cert[0].decode("ascii")
                credential_env[JobProcessEnv.JOB_KEY] = job_cert[1].decode("ascii")
            credential_secret_name = self._ensure_job_credential_secret(pod_name, credential_env)

            env[ENV_WORKSPACE_OWNER_FQCN] = workspace_transfer.owner_fqcn

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
            secret_mounts = study_runtime.secret_mounts if study_runtime is not None else []
            for secret_mount in secret_mounts:
                volume_name = study_secret_volume_name(secret_mount.study, secret_mount.name)
                secret_source = {"secretName": secret_mount.source}
                if secret_mount.items:
                    secret_source["items"] = [{"key": key, "path": path} for key, path in secret_mount.items]
                volume_list.append({"name": volume_name, "secret": secret_source})
                volume_mount_list.append({"name": volume_name, "mountPath": secret_mount.mount_path, "readOnly": True})

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
            secret_env_refs = [{"name": name, "source": credential_secret_name, "key": name} for name in credential_env]
            if study_runtime is not None:
                if study_runtime.secret_env:
                    secret_env_refs.extend(
                        {"name": ref.name, "source": ref.source, "key": ref.key} for ref in study_runtime.secret_env
                    )
                if pod_manifest_template is not None:
                    # Credential secretKeyRefs attach to every job, so a multi-container
                    # template must always name its main container.
                    job_config["require_main_container"] = True
            if secret_env_refs:
                job_config["secret_env"] = secret_env_refs
            if self.image_pull_secrets:
                job_config["image_pull_secrets"] = self.image_pull_secrets
            if args is not None and getattr(args, "set", None) is not None:
                job_config.update({"set_list": args.set})
            resources = {
                "requests": {"ephemeral-storage": job_ephemeral_storage},
                "limits": {"ephemeral-storage": job_ephemeral_storage},
            }
            for key in ("cpu", "memory"):
                limit_val = native_resources.get(key)
                # cpu_request / memory_request allow request < limit; when absent,
                # request mirrors the limit so admission webhooks that require
                # explicit cpu/memory requests (e.g. AKS deployment safeguards) pass.
                request_val = native_resources.get(f"{key}_request", limit_val)
                if limit_val:
                    resources["limits"][key] = limit_val
                if request_val:
                    resources["requests"][key] = request_val
            if "num_of_cpus" in portable_spec:
                cpu_quantity = str(portable_spec["num_of_cpus"])
                resources["limits"]["cpu"] = cpu_quantity
                resources["requests"]["cpu"] = cpu_quantity
            if "memory" in portable_spec:
                resources["limits"]["memory"] = portable_spec["memory"]
                resources["requests"]["memory"] = portable_spec["memory"]
            if job_resource:
                resources["limits"]["nvidia.com/gpu"] = str(job_resource)
                resources["requests"]["nvidia.com/gpu"] = str(job_resource)
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
                credential_secret_name=credential_secret_name,
            )
            pod_manifest = job_handle.get_manifest()
            if pod_manifest_template is not None:
                self._replace_pod_manifest_template_namespace(pod_manifest)
            job_handle.task_owner = self.prepare_job_manifest(
                pod_manifest,
                raw_job_id,
                site_name,
                workspace_obj,
                execution_lifetime=execution_lifetime,
                job_meta=job_meta,
                fl_ctx=fl_ctx,
                transfer_token=workspace_transfer_token,
                workspace_reservations=workspace_reservations,
                startup_secret_name=startup_secret_name,
                previous_handle=previous,
            )
            if workspace_reservations:
                job_handle.workspace_reservation = workspace_reservations[0]
            job_handle.pod_manifest = pod_manifest
            self.logger.debug(
                "launch job with k8s_launcher: pod_name=%s namespace=%s image=%s",
                pod_manifest["metadata"]["name"],
                self.namespace,
                job_image,
            )
            pod_create_attempted = True
            created_pod = self.core_v1.create_namespaced_pod(body=pod_manifest, namespace=self.namespace)
            job_handle.pod_uid = created_pod.metadata.uid
            if job_handle.task_owner is not None:
                job_handle.task_owner.attach_cj(pod_name, created_pod.metadata.uid)
            if credential_secret_name:
                self._own_credential_secret(credential_secret_name, pod_name, created_pod)
        except Exception as e:
            # A transport/5xx error can hide a successful create. Keep its
            # workspace fenced; a later empty read is not proof it never ran.
            rejected = getattr(e, "status", None) in {400, 401, 403, 404, 405, 413, 415, 422, 429}
            if previous is None and created_pod is None and (not pod_create_attempted or rejected):
                for directory, device, inode in workspace_reservations:
                    self._discard_empty_workspace(directory, device, inode)
            workspace_transfer.remove_job(raw_job_id)
            if credential_secret_name and created_pod is None:
                # No pod means no ownerReference, so GC would never collect the Secret.
                self._delete_secret(credential_secret_name)
            if "job_handle" in locals():
                if job_handle.task_owner is not None:
                    job_handle.task_owner.release()
                job_handle.workspace_job_id = ""
                self.logger.error(f"failed to launch job {job_id}: {e}")
                job_handle.terminal_state = JobState.TERMINATED
                job_handle.terminal_return_code = JobReturnCode.EXCEPTION
                return job_handle
            raise
        try:
            publish_job_handle(job_handle, fl_ctx)
            entered_running = job_handle.enter_states([JobState.RUNNING])
        except BaseException:
            job_handle.terminate()
            raise
        if not entered_running:
            self.logger.warning(f"unable to enter running phase {job_id}")
        return job_handle

    def _discard_empty_workspace(self, directory, device, inode):
        try:
            parent_fd = os.open(os.path.dirname(directory), os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            try:
                name = os.path.basename(directory)
                info = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
                if (info.st_dev, info.st_ino) != (device, inode):
                    return
                # rmdir refuses symlinks and nonempty workspaces. Never recurse.
                os.rmdir(name, dir_fd=parent_fd)
            finally:
                os.close(parent_fd)
        except OSError as error:
            self.logger.warning("retaining job workspace after rejected launch: %s", error)

    def get_execution_lifetime(self, job_meta, fl_ctx):
        return "job"

    def prepare_job_manifest(self, manifest, job_id, site_name, workspace, **kwargs):
        """Prepare a job Pod, optionally returning its CP-owned task service record."""

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
    """Launch resident or task CJs with an optional shared persistent jobs workspace."""

    def __init__(
        self,
        *args,
        job_workspace_pvc: str = None,
        job_workspace_pvc_root: str = None,
        run_as_user: int = None,
        run_as_group: int = None,
        cj_resources: dict = None,
        persist_resident_workspaces: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if cj_resources is not None:
            validate_cj_resource_spec({"@default": cj_resources})
        self.cj_resources = {"num_of_cpus": 0.25, "memory": "512Mi", "num_of_gpus": 0}
        self.cj_resources.update(cj_resources or {})
        if not isinstance(persist_resident_workspaces, bool):
            raise ValueError("persist_resident_workspaces must be a boolean")
        self.persist_resident_workspaces = persist_resident_workspaces
        self.job_workspace_pvc = None
        self.job_workspace_pvc_root = None
        self.job_security_context = None
        self._job_storage_lock = threading.Lock()
        self._job_storage_checked = False
        settings = (job_workspace_pvc, job_workspace_pvc_root, run_as_user, run_as_group)
        if all(value is None for value in settings):
            if persist_resident_workspaces:
                raise ValueError("persist_resident_workspaces requires the persistent jobs PVC profile")
            return
        if any(value is None for value in settings):
            raise ValueError("persistent job workspaces require PVC, CP mount and numeric UID/GID together")
        self.job_workspace_pvc = validate_dns_name("job_workspace_pvc", job_workspace_pvc)
        if not isinstance(job_workspace_pvc_root, str) or not os.path.isabs(job_workspace_pvc_root):
            raise ValueError("job_workspace_pvc_root must name the CP's explicit mount of the existing jobs PVC")
        self.job_workspace_pvc_root = os.path.realpath(job_workspace_pvc_root)
        self.job_security_context = task_security_context(run_as_user, run_as_group)

    def get_cj_resources(self, job_meta, site_name):
        return {**self.cj_resources, **get_cj_resource_spec(job_meta, site_name)}

    def get_execution_lifetime(self, job_meta, fl_ctx):
        if self.job_workspace_pvc is None:
            if has_cj_resource_spec(job_meta, fl_ctx.get_identity_name()):
                raise ValueError("task CJ resources require the persistent Kubernetes jobs profile")
            return "job"
        workspace = fl_ctx.get_prop(FLContextKey.WORKSPACE_OBJECT)
        if workspace is None:
            return "job"
        config = ConfigService.load_config_dict(
            workspace.get_client_app_config_file_path(job_meta[JobConstants.JOB_ID])
        )
        lifetime = get_client_execution_lifetime(config)
        if lifetime == "task":
            reject_job_gpu_reservation(job_meta, fl_ctx.get_identity_name())
        elif has_cj_resource_spec(job_meta, fl_ctx.get_identity_name()):
            raise ValueError("cj_resource_spec does not match the deployed client execution_lifetime")
        return lifetime

    def prepare_job_manifest(
        self,
        manifest,
        job_id,
        site_name,
        workspace,
        *,
        execution_lifetime="job",
        job_meta=None,
        fl_ctx=None,
        transfer_token=None,
        workspace_reservations=None,
        startup_secret_name=None,
        previous_handle=None,
    ):
        if self.job_workspace_pvc is None or (execution_lifetime == "job" and not self.persist_resident_workspaces):
            return

        spec = manifest["spec"]
        if (
            len(spec["containers"]) != 1
            or spec.get("initContainers")
            or spec.get("ephemeralContainers")
            or any(spec.get(name) for name in ("hostPID", "hostIPC", "hostNetwork"))
        ):
            raise ValueError("the persistent Kubernetes CJ profile requires one isolated container")
        volumes = [volume for volume in spec["volumes"] if volume["name"] == "workspace-job"]
        if len(volumes) != 1:
            raise ValueError("persistent CJ requires the resident workspace-job volume")
        volumes[0].clear()
        volumes[0].update(name="workspace-job", persistentVolumeClaim={"claimName": self.job_workspace_pvc})
        container = spec["containers"][0]
        mounts = [mount for mount in container["volumeMounts"] if mount["name"] == "workspace-job"]
        if len(mounts) != 1 or mounts[0]["mountPath"] != self.workspace_mount_path:
            raise ValueError("CJ workspace mount differs from the configured durable workspace")
        if execution_lifetime == "task":
            from nvflare.app_opt.task_launcher.k8s_launcher import K8sTaskLauncher
            from nvflare.app_opt.task_launcher.k8s_task_service import K8sTaskService
            from nvflare.private.fed.client.task_execution_policy import GPU_ENVIRONMENT_NAMES, TaskExecutionPolicy

            policy = TaskExecutionPolicy.from_workspace(workspace)
            if type(policy.launcher) is not K8sTaskLauncher:
                raise ValueError("Kubernetes task CJ requires the canonical site K8sTaskLauncher")
        if (
            any(
                volume.get("persistentVolumeClaim", {}).get("claimName") == self.job_workspace_pvc
                for volume in spec["volumes"]
                if volume is not volumes[0]
            )
            or len([mount for mount in container["volumeMounts"] if mount["name"] == "workspace-job"]) != 1
        ):
            raise ValueError("the jobs PVC may only be mounted through the assigned job subPath")
        mounts[0]["subPath"] = self._prepare_workspace_subpath(job_id, site_name, previous_handle)
        container["env"] = [item for item in container.get("env", []) if item["name"] != ENV_PERSISTENT_WORKSPACE]
        container["env"].append({"name": ENV_PERSISTENT_WORKSPACE, "value": "1"})
        if workspace_reservations is not None:
            directory = os.path.join(self.job_workspace_pvc_root, mounts[0]["subPath"])
            info = os.stat(directory, follow_symlinks=False)
            workspace_reservations.append((directory, info.st_dev, info.st_ino))
        container["securityContext"] = {
            **container.get("securityContext", {}),
            "privileged": False,
            "runAsUser": self.job_security_context["runAsUser"],
            "runAsGroup": self.job_security_context["runAsGroup"],
            "runAsNonRoot": True,
            "allowPrivilegeEscalation": False,
            "capabilities": {"drop": ["ALL"]},
        }
        spec["securityContext"] = {**spec.get("securityContext", {}), **self.job_security_context}
        spec["automountServiceAccountToken"] = False
        if execution_lifetime != "task":
            return
        # A worker Pod cannot reach CJ's loopback. Kubernetes, rather than a
        # job/template environment value, supplies the advertised CJ address.
        env = [item for item in container.get("env", []) if item["name"] != ENV_CJ_POD_IP]
        env.append(
            {
                "name": ENV_CJ_POD_IP,
                "valueFrom": {"fieldRef": {"apiVersion": "v1", "fieldPath": "status.podIP"}},
            }
        )
        container["env"] = env
        if not self.get_cj_resources(job_meta, site_name).get("num_of_gpus"):
            env = [item for item in container.get("env", []) if item["name"] not in GPU_ENVIRONMENT_NAMES]
            env.extend(
                {"name": name, "value": "none" if name == "NVIDIA_VISIBLE_DEVICES" else ""}
                for name in sorted(GPU_ENVIRONMENT_NAMES)
            )
            container["env"] = env
        spec["serviceAccountName"] = policy.launcher.worker_service_account
        backend = policy.launcher.create_backend(
            self.core_v1,
            os.path.join(self.job_workspace_pvc_root, mounts[0]["subPath"]),
            job_meta,
            job_id,
            site_name,
            startup_secret_name=startup_secret_name,
        )
        cell = fl_ctx.get_engine().cell
        return K8sTaskService.get_or_create(cell).add_job(backend, job_id, site_name, transfer_token)

    def _prepare_workspace_subpath(self, job_id, site_name, previous_handle=None):
        subpath = job_workspace_subpath(job_id, site_name)
        # CP explicitly mounts the same PVC. Create a fresh, owned job subtree
        # instead of relying on kubelet-created subPath directory permissions.
        # Only CP's confirmed previous handle can authorize CJ recovery. A
        # retained directory alone never authorizes adoption after CP restart.
        from nvflare.app_opt.task_launcher.k8s_launcher import check_posix_storage

        root = self.job_workspace_pvc_root
        if not os.path.isdir(root):
            raise ValueError("the jobs workspace PVC must already be mounted in CP")
        with self._job_storage_lock:
            if not self._job_storage_checked:
                check_posix_storage(root)
                self._job_storage_checked = True
        parent = os.path.join(root, "nvflare-tasks")
        os.makedirs(parent, mode=0o700, exist_ok=True)
        if os.path.islink(parent) or os.path.realpath(parent) != parent:
            raise ValueError("job workspace parent must be a real directory on the configured PVC")
        directory = os.path.join(root, subpath)
        if previous_handle is not None:
            info = os.stat(directory, follow_symlinks=False)
            if (
                previous_handle.workspace_reservation != (directory, info.st_dev, info.st_ino)
                or os.path.realpath(directory) != directory
                or not previous_handle.can_restart()
            ):
                raise ValueError("persistent workspace differs from the confirmed previous execution")
            return subpath
        try:
            os.mkdir(directory, mode=0o700)
        except FileExistsError as error:
            raise ValueError("job workspace already exists; retained job/site data cannot be reused") from error
        info = os.stat(directory)
        if info.st_uid != self.job_security_context["runAsUser"]:
            self._discard_empty_workspace(directory, info.st_dev, info.st_ino)
            raise ValueError("CP, CJ and workers must use the same nonroot numeric UID")
        return subpath

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
