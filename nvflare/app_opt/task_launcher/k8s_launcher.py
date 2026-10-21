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

"""Site-owned Kubernetes task execution with native CPU/GPU admission."""

import hashlib
import json
import math
import os
import tempfile
import threading
import time
import uuid

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.task_launcher_spec import (
    TaskExecutionPhase,
    TaskExecutionStatus,
    TaskHandleSpec,
    TaskLauncherError,
    TaskLaunchError,
    TaskLauncherSpec,
    TaskLaunchRequest,
    TaskResourceRequest,
    TaskSettlementError,
)
from nvflare.app_opt.job_launcher.k8s_launcher import (
    DEFAULT_NAMESPACE,
    WORKSPACE_MOUNT_PATH,
    _dns_name,
    _is_cpu_memory_gpu_shortage,
    reject_job_gpu_reservation,
    site_name_to_rfc1123,
    task_security_context,
    task_workspace_subpath,
)
from nvflare.app_opt.job_launcher.k8s_pod import build_pod_manifest, get_container_waiting_failure
from nvflare.private.fed.client.task_execution_policy import FORBIDDEN_ENVIRONMENT_NAMES, GPU_ENVIRONMENT_NAMES
from nvflare.private.fed.task_worker.protocol import read_bootstrap
from nvflare.utils.job_launcher_utils import (
    get_job_launcher_spec,
    get_portable_resource_spec,
    validate_portable_resource_conflicts,
)

WORKER_MODULE = "nvflare.private.fed.app.client.task_worker_process"
SETTLEMENT_FINALIZER = "nvflare.nvidia.com/task-settlement"
_OWNER_ANNOTATION = "nvflare.nvidia.com/launch-owner"
_IDENTITY_ANNOTATION = "nvflare.nvidia.com/task-identity"
_CONTAINER = "nvflare-task"
_LABEL_PREFIX = "nvflare.nvidia.com/"


def _seconds(name, value, *, positive=True):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
        or (positive and value == 0)
    ):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")
    return value


def _within(root, path):
    if (
        not isinstance(path, str)
        or not os.path.isabs(path)
        or os.path.commonpath([root, os.path.realpath(path)]) != root
    ):
        raise ValueError("task paths must be absolute paths within the configured durable workspace")
    return path


def check_posix_storage(root):
    """Probe local primitives, not cross-pod coherence or the backing PVC's durability."""
    with tempfile.TemporaryDirectory(prefix=".nvflare-storage-", dir=root) as temporary:
        source, linked, manifest = (os.path.join(temporary, name) for name in ("source", "linked", "manifest"))
        with open(source, "wb") as stream:
            stream.write(b"nvflare-posix-probe")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(source, linked)
        if os.stat(source).st_ino != os.stat(linked).st_ino:
            raise ValueError("durable storage must support real hard links")
        os.replace(linked, manifest)
        os.unlink(source)
        with open(manifest, "rb") as stream:
            if stream.read() != b"nvflare-posix-probe":
                raise ValueError("durable storage failed atomic manifest probe")
        directory_fd = os.open(temporary, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)


def _native_resources(resources):
    result = {"requests": {}, "limits": {}}
    for key, value in (
        ("cpu", str(resources.cpu_cores) if resources.cpu_cores is not None else None),
        ("memory", f"{resources.memory_mb}Mi" if resources.memory_mb is not None else None),
        ("nvidia.com/gpu", str(resources.gpu_count) if resources.gpu_count else None),
    ):
        if value is not None:
            result["requests"][key] = result["limits"][key] = value
    return result


def task_pod_resources(meta, site_name, defaults):
    """Resolve this site's existing job metadata into task-pod admission requests.

    This does not select a backend or reserve resources in CJ. Portable and
    native K8s precedence/conflicts follow the resident job metadata contract.
    An explicit zero GPU request overrides a GPU site default.
    """
    from kubernetes.utils.quantity import parse_quantity

    validate_portable_resource_conflicts(meta)
    portable = get_portable_resource_spec(meta, site_name)
    native = get_job_launcher_spec(meta, site_name, "k8s")
    result = _native_resources(defaults)
    for key, portable_key in (("cpu", "num_of_cpus"), ("memory", "memory")):
        if portable_key in portable:
            limit = request = str(portable[portable_key])
        elif key in native or f"{key}_request" in native:
            limit = native.get(key)
            request = native.get(f"{key}_request", limit)
        else:
            continue
        quantities = {}
        for kind, value in (("limits", limit), ("requests", request)):
            result[kind].pop(key, None)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float, str)):
                raise ValueError(f"task {key} {kind} must be a positive Kubernetes quantity")
            try:
                quantity = parse_quantity(value)
            except (ValueError, ArithmeticError) as error:
                raise ValueError(f"invalid task {key} {kind} quantity") from error
            if not quantity.is_finite() or quantity <= 0:
                raise ValueError(f"task {key} {kind} must be positive and finite")
            quantities[kind] = quantity
            result[kind][key] = str(value)
        if quantities.get("requests", 0) > quantities.get("limits", float("inf")):
            raise ValueError(f"task {key} request exceeds its limit")
    gpu_count = native.get("num_of_gpus", portable.get("num_of_gpus", defaults.gpu_count))
    TaskResourceRequest(gpu_count=gpu_count)
    for kind in result.values():
        kind.pop("nvidia.com/gpu", None)
        if gpu_count:
            kind["nvidia.com/gpu"] = str(gpu_count)
    return result


class _PodOwnershipError(RuntimeError):
    """An observed Pod no longer preserves the launcher's ownership fence."""


class K8sTaskLauncher(TaskLauncherSpec):
    """Site-owned image and existing POSIX PVC; one bare Pod per physical attempt.

    No Kubernetes resources are created until launch_task. No kubeconfig or
    parent environment is implicitly forwarded into workers. The CP, CJ and
    worker use Kubernetes; the common CJ request is staged metadata, then
    materialized into the site image using the owning CJ's resource binding.
    """

    launch_mode = "k8s"

    def __init__(
        self,
        namespace: str,
        image: str,
        workspace_root: str,
        workspace_pvc: str,
        worker_service_account: str,
        python_path: str = "/usr/local/bin/python",
        config_file_path: str = None,
        cleanup_timeout: float = 60,
        poll_interval: float = 0.5,
        pending_timeout: float = 120,
        request_timeout: float = 5,
        termination_grace_seconds: int = 30,
        run_as_user: int = 1000,
        run_as_group: int = 1000,
        cpu_cores: float = None,
        memory_mb: int = None,
        gpu_count: int = 0,
        environment: dict = None,
    ):
        super().__init__()
        self.namespace = _dns_name("namespace", namespace)
        self.workspace_pvc = _dns_name("workspace_pvc", workspace_pvc)
        self.worker_service_account = _dns_name("worker_service_account", worker_service_account)
        if not isinstance(image, str) or not image.strip():
            raise ValueError("image must be site configured")
        if not isinstance(python_path, str) or not os.path.isabs(python_path):
            raise ValueError("python_path must be an absolute image path")
        if not isinstance(workspace_root, str) or not os.path.isabs(workspace_root) or workspace_root == "/":
            raise ValueError("workspace_root must be an absolute dedicated directory")
        self.workspace_root = os.path.realpath(workspace_root)
        self.image = image
        self.python_path = python_path
        self.config_file_path = config_file_path
        self.cleanup_timeout = _seconds("cleanup_timeout", cleanup_timeout)
        self.poll_interval = _seconds("poll_interval", poll_interval)
        self.pending_timeout = _seconds("pending_timeout", pending_timeout)
        self.request_timeout = _seconds("request_timeout", request_timeout)
        self.security_context = task_security_context(run_as_user, run_as_group)
        self.resources = TaskResourceRequest(cpu_cores=cpu_cores, memory_mb=memory_mb, gpu_count=gpu_count)
        self.environment = dict(environment or {})
        # Reuse the request's strict environment validation for site policy.
        TaskLaunchRequest("policy", "policy", "policy", "policy", (python_path,), environment=self.environment)
        if set(self.environment) & FORBIDDEN_ENVIRONMENT_NAMES:
            raise ValueError("site worker environment cannot include credentials or manual GPU visibility")
        if isinstance(termination_grace_seconds, bool) or not isinstance(termination_grace_seconds, int):
            raise ValueError("termination_grace_seconds must be a positive integer")
        self.termination_grace_seconds = _seconds("termination_grace_seconds", termination_grace_seconds)
        self.core_v1 = None
        self._lock = threading.Lock()
        self._identities = set()
        self._job_binding = None
        self._storage_checked = False

    def validate_site_config(self, site_config):
        """Check the canonical CP/CJ/task profile without creating any resources."""
        expected = "nvflare.app_opt.job_launcher.k8s_launcher.ClientK8sJobLauncher"
        launchers = []
        for item in site_config.get("components", []):
            if not isinstance(item, dict):
                continue
            path = item.get("path", item.get("class_path", item.get("name", "")))
            if path == "ClientK8sJobLauncher":
                raise ValueError(f"Kubernetes CJ launcher is not registered by short name; specify path={expected!r}")
            if isinstance(path, str) and path.endswith("JobLauncher"):
                launchers.append((path, item))
        if len(launchers) != 1 or launchers[0][0] != expected:
            raise ValueError("Kubernetes task execution requires exactly one matching Kubernetes task CJ JobLauncher")
        if site_config.get("task_execution", {}).get("environment_variables"):
            raise ValueError("Kubernetes worker environment belongs to the site task launcher configuration")
        args = launchers[0][1].get("args", {})
        if not isinstance(args, dict):
            raise ValueError("Kubernetes task CJ launcher args must be a dict")
        if args.get("namespace", DEFAULT_NAMESPACE) != self.namespace:
            raise ValueError("Kubernetes CP/CJ/task namespaces must match")
        if args.get("task_workspace_pvc") != self.workspace_pvc:
            raise ValueError("Kubernetes CP/CJ/task workspace PVCs must match")
        mount = args.get("workspace_mount_path", WORKSPACE_MOUNT_PATH)
        if not isinstance(mount, str) or not os.path.isabs(mount) or os.path.realpath(mount) != self.workspace_root:
            raise ValueError("Kubernetes CJ/task workspace mount paths must match")
        cp_mount = args.get("task_workspace_pvc_root")
        if not isinstance(cp_mount, str) or not os.path.isabs(cp_mount) or os.path.realpath(cp_mount) == "/":
            raise ValueError("Kubernetes CP requires an explicit dedicated mount of the existing task PVC")
        if task_security_context(args.get("run_as_user"), args.get("run_as_group")) != self.security_context:
            raise ValueError("Kubernetes CP/CJ/task numeric UID/GID must match")
        cj_account = _dns_name("task_cj_service_account", args.get("task_cj_service_account"))
        if cj_account == self.worker_service_account:
            raise ValueError("Kubernetes CJ and task workers require distinct service accounts")

    def handle_event(self, event_type, fl_ctx):
        if event_type != EventType.START_RUN:
            return
        job_id, site_name = fl_ctx.get_job_id(), fl_ctx.get_identity_name()
        if not job_id or not site_name:
            raise ValueError("Kubernetes task launch requires the owning CJ's job/site identity")
        meta = fl_ctx.get_prop(FLContextKey.JOB_META, {}) or {}
        reject_job_gpu_reservation(meta, site_name)
        resources = task_pod_resources(meta, site_name, self.resources)
        binding = (job_id, site_name, resources)
        with self._lock:
            if self._job_binding is not None and self._job_binding != binding:
                raise ValueError("Kubernetes task launcher is already bound to a different CJ or resource request")
            self._job_binding = binding

    def _api(self):
        if self.core_v1 is None:
            from kubernetes import client, config

            configuration = client.Configuration()
            if self.config_file_path:
                config.load_kube_config(self.config_file_path, client_configuration=configuration)
            else:
                config.load_incluster_config(client_configuration=configuration)
            self.core_v1 = client.CoreV1Api(client.ApiClient(configuration))
        return self.core_v1

    def launch_task(self, request: TaskLaunchRequest):
        if not isinstance(request, TaskLaunchRequest):
            raise TypeError("request must be TaskLaunchRequest")
        if not (
            len(request.argv) == 7
            and request.argv[1:4] == ("-m", WORKER_MODULE, "--bootstrap")
            and request.argv[5] == "--parent_pid"
            and request.argv[6].isdigit()
        ):
            raise ValueError("Kubernetes requires the owning CJ's staged task-worker request")
        if request.environment or not request.resources.is_empty():
            raise ValueError("Kubernetes task environment and resources belong to site policy and the bound CJ")
        binding = self._job_binding
        if binding is None:
            raise ValueError("staged CJ task requests require the owning START_RUN resource binding")
        job_id, site_name, resources = binding
        if (request.job_id, request.site_name) != (job_id, site_name):
            raise ValueError("task request does not belong to the bound Kubernetes CJ")
        # The common command locates staged bootstrap. The Pod always uses the
        # site image/interpreter and native scope, without the CJ's local PID.
        environment = dict(self.environment)
        if "nvidia.com/gpu" not in resources["limits"]:
            # CPU tasks can use the same GPU-capable image. Do not inherit an
            # image's NVIDIA_VISIBLE_DEVICES=all on a GPU-runtime node.
            environment.update(
                {name: "none" if name == "NVIDIA_VISIBLE_DEVICES" else "" for name in GPU_ENVIRONMENT_NAMES}
            )
        path = _within(self.workspace_root, request.argv[4])
        _within(self.workspace_root, request.cwd)
        bootstrap = read_bootstrap(path)
        _within(self.workspace_root, bootstrap.artifact_root)
        if os.stat(self.workspace_root).st_dev != os.stat(bootstrap.artifact_root).st_dev:
            raise ValueError("task artifacts and checkpoints must share the durable workspace filesystem")
        if os.path.realpath(bootstrap.workspace_root) != self.workspace_root:
            raise ValueError("bootstrap workspace must match the explicit durable PVC mount")
        identity = bootstrap.identity
        if request.identity != (identity.job_id, identity.site_name, identity.task_id, identity.attempt_id):
            raise ValueError("request and bootstrap identities differ")
        with self._lock:
            if not self._storage_checked:
                check_posix_storage(self.workspace_root)
                self._storage_checked = True
            if request.identity in self._identities:
                raise TaskLauncherError("task attempt already owned; resubmit its saved result")
            self._identities.add(request.identity)
            api = self._api()
        owner = uuid.uuid4().hex
        digest = hashlib.sha256(json.dumps(request.identity).encode()).hexdigest()[:20]
        name = f"nvft-{digest}-{owner[:16]}"
        manifest = build_pod_manifest(
            {
                "name": name,
                "container_name": _CONTAINER,
                "image": self.image,
                "command": WORKER_MODULE,
                "module_args": {"--bootstrap": path, "--execution_scope": "native"},
                "resources": resources,
                "env": environment,
                "volume_list": [
                    {"name": "workspace", "persistentVolumeClaim": {"claimName": self.workspace_pvc}},
                    {
                        "name": "startup-kit",
                        "secret": {"secretName": f"nvflare-startup-{site_name_to_rfc1123(request.site_name)}"},
                    },
                ],
                "volume_mount_list": [
                    {
                        "name": "workspace",
                        "mountPath": self.workspace_root,
                        "subPath": task_workspace_subpath(request.job_id, request.site_name),
                    },
                    {
                        "name": "startup-kit",
                        "mountPath": os.path.join(self.workspace_root, "startup"),
                        "readOnly": True,
                    },
                    {
                        "name": "workspace",
                        "mountPath": os.path.join(self.workspace_root, "local"),
                        "readOnly": True,
                        "subPath": task_workspace_subpath(request.job_id, request.site_name) + "/local",
                    },
                ],
            },
            python_path=self.python_path,
            preserve_empty_env=True,
        )
        manifest["spec"]["containers"][0]["workingDir"] = request.cwd
        manifest["metadata"].update(
            namespace=self.namespace,
            finalizers=[SETTLEMENT_FINALIZER],
            labels={
                _LABEL_PREFIX + key: site_name_to_rfc1123(value, max_length=63)
                for key, value in zip(("job", "site", "task", "attempt"), request.identity)
            },
            annotations={_OWNER_ANNOTATION: owner, _IDENTITY_ANNOTATION: json.dumps(request.identity)},
        )
        manifest["spec"].update(
            automountServiceAccountToken=False,
            serviceAccountName=self.worker_service_account,
            terminationGracePeriodSeconds=self.termination_grace_seconds,
            enableServiceLinks=False,
            securityContext=self.security_context,
        )
        manifest["spec"]["containers"][0]["securityContext"] = {
            "allowPrivilegeEscalation": False,
            "capabilities": {"drop": ["ALL"]},
        }
        handle = K8sTaskHandle(self, request, name, owner)
        try:
            pod = api.create_namespaced_pod(
                namespace=self.namespace, body=manifest, _request_timeout=self.request_timeout
            )
            handle._identify(pod)
            if SETTLEMENT_FINALIZER not in (pod.metadata.finalizers or []):
                raise _PodOwnershipError("admission removed the task settlement finalizer")
        except BaseException as error:
            # A transport failure may mean the pod exists. Keep the name/nonce
            # and a cleanup-capable handle rather than retrying physical compute.
            if getattr(error, "status", None) in (400, 401, 403, 404, 405, 409, 422):
                handle._rejected = True
            handle._failure = f"pod launch failed: {type(error).__name__}"
            raise TaskLaunchError(handle._failure, handle) from error
        return handle


class K8sTaskHandle(TaskHandleSpec):
    def __init__(self, launcher, request, name, owner):
        self.launcher = launcher
        self._request = request
        self.pod_name = name
        self._owner = owner
        self._uid = None
        self._lock = threading.RLock()
        self._started = time.monotonic()
        self._cancel_requested = False
        self._delete_sent = False
        self._release_authorized = False
        self._rejected = False
        self._settled = False
        self._phase = TaskExecutionPhase.QUEUED
        self._exit_code = None
        self._failure = None
        self._observation_error = None
        self._log_tail = ""
        self.log_error = None

    @property
    def request(self):
        return self._request

    @property
    def execution_id(self):
        return f"k8s:{self.launcher.namespace}/{self.pod_name}@{self._uid or 'unresolved'}"

    def _identify(self, pod):
        metadata = pod.metadata
        if (
            metadata.name != self.pod_name
            or metadata.namespace != self.launcher.namespace
            or (metadata.annotations or {}).get(_OWNER_ANNOTATION) != self._owner
            or not metadata.uid
            or (self._uid is not None and metadata.uid != self._uid)
        ):
            raise _PodOwnershipError("pod ownership or UID differs from launched execution")
        self._uid = metadata.uid

    def _status(self):
        return TaskExecutionStatus(
            self._phase,
            exit_code=self._exit_code,
            cancel_requested=self._cancel_requested,
            settled=self._settled,
            failure_reason=self._failure or self._observation_error,
        )

    def _delete(self):
        if self._delete_sent:
            return
        self.launcher.core_v1.delete_namespaced_pod(
            name=self.pod_name,
            namespace=self.launcher.namespace,
            body={
                "preconditions": {"uid": self._uid},
                "gracePeriodSeconds": self.launcher.termination_grace_seconds,
            },
            _request_timeout=self.launcher.request_timeout,
        )
        self._delete_sent = True

    def _shutdown_confirmed(self, pod):
        if not pod.spec.node_name:
            # Read after DELETE: an unbound, deleting pod cannot be scheduled.
            return bool(pod.metadata.deletion_timestamp)
        if pod.status.phase not in ("Succeeded", "Failed") or pod.status.reason in ("NodeLost", "Shutdown"):
            return False
        sandbox_down = any(
            condition.type == "PodReadyToStartContainers" and condition.status == "False"
            for condition in (pod.status.conditions or [])
        )
        containers = [
            *(pod.spec.containers or []),
            *(pod.spec.init_containers or []),
            *(pod.spec.ephemeral_containers or []),
        ]
        statuses = [
            *(pod.status.container_statuses or []),
            *(pod.status.init_container_statuses or []),
            *(pod.status.ephemeral_container_statuses or []),
        ]
        terminated = {status.name for status in statuses if status.state and status.state.terminated is not None}
        return sandbox_down and bool(containers) and all(container.name in terminated for container in containers)

    def read_logs(self):
        """Bounded API log tail, cached before removal; worker file logs live on PVC."""
        with self._lock:
            if not self._settled:
                try:
                    pod = self.launcher.core_v1.read_namespaced_pod(
                        name=self.pod_name,
                        namespace=self.launcher.namespace,
                        _request_timeout=self.launcher.request_timeout,
                    )
                    self._identify(pod)
                    self._log_tail = self.launcher.core_v1.read_namespaced_pod_log(
                        name=self.pod_name,
                        namespace=self.launcher.namespace,
                        container=_CONTAINER,
                        tail_lines=1000,
                        limit_bytes=64 * 1024,
                        _request_timeout=self.launcher.request_timeout,
                    )
                except Exception as error:
                    self.log_error = type(error).__name__
            return self._log_tail

    def poll(self):
        with self._lock:
            if self._settled:
                return self._status()
            if self._rejected:
                self._settled = True
                self._phase = TaskExecutionPhase.TERMINAL
                return self._status()
            try:
                pod = self.launcher.core_v1.read_namespaced_pod(
                    name=self.pod_name,
                    namespace=self.launcher.namespace,
                    _request_timeout=self.launcher.request_timeout,
                )
                self._identify(pod)
                if SETTLEMENT_FINALIZER not in (pod.metadata.finalizers or []) and not self._release_authorized:
                    raise _PodOwnershipError("task settlement finalizer disappeared before shutdown confirmation")
                self._observation_error = None
                phase = pod.status.phase
                if phase in ("Succeeded", "Failed"):
                    self._phase = TaskExecutionPhase.TERMINAL
                    for status in pod.status.container_statuses or []:
                        if status.name == _CONTAINER and status.state and status.state.terminated:
                            self._exit_code = status.state.terminated.exit_code
                    if phase == "Failed":
                        self._failure = self._failure or pod.status.reason or "task pod failed"
                elif phase == "Running":
                    self._phase = TaskExecutionPhase.RUNNING
                elif phase == "Pending":
                    failure = get_container_waiting_failure(pod.status)
                    for condition in pod.status.conditions or []:
                        if condition.type == "PodScheduled" and condition.status == "False":
                            detail = f"{condition.reason or ''} {condition.message or ''}"
                            if condition.reason == "Unschedulable" and not _is_cpu_memory_gpu_shortage(detail):
                                failure = detail
                    if time.monotonic() - self._started >= self.launcher.pending_timeout:
                        failure = failure or "task pod pending timeout"
                    self._failure = self._failure or failure or None
                else:
                    self._phase = TaskExecutionPhase.TERMINAL
                    self._failure = self._failure or f"pod phase {phase!r} has no confirmed runtime state"
                if self._cancel_requested or self._failure or phase in ("Succeeded", "Failed"):
                    self._delete()
                if self._shutdown_confirmed(pod):
                    self._release_authorized = True
                    self.read_logs()
                    finalizers = pod.metadata.finalizers or []
                    if SETTLEMENT_FINALIZER in finalizers:
                        # Compare before removing only our finalizer. Never
                        # overwrite concurrent controller metadata or another UID.
                        self.launcher.core_v1.patch_namespaced_pod(
                            name=self.pod_name,
                            namespace=self.launcher.namespace,
                            body=[
                                {"op": "test", "path": "/metadata/uid", "value": self._uid},
                                {
                                    "op": "test",
                                    "path": "/metadata/resourceVersion",
                                    "value": pod.metadata.resource_version,
                                },
                                {"op": "test", "path": "/metadata/finalizers", "value": finalizers},
                                {
                                    "op": "replace",
                                    "path": "/metadata/finalizers",
                                    "value": [value for value in finalizers if value != SETTLEMENT_FINALIZER],
                                },
                            ],
                            _request_timeout=self.launcher.request_timeout,
                        )
                return self._status()
            except Exception as error:
                # Missing/replaced Pods must enter bounded settlement, but an
                # observation failure must not terminate healthy running work.
                if getattr(error, "status", None) == 404 and self._release_authorized:
                    self._settled = True
                    self._phase = TaskExecutionPhase.TERMINAL
                    self._observation_error = None
                elif getattr(error, "status", None) == 404:
                    self._phase = TaskExecutionPhase.TERMINAL
                    self._failure = self._failure or "pod disappeared without confirmed shutdown"
                elif isinstance(error, _PodOwnershipError):
                    self._phase = TaskExecutionPhase.TERMINAL
                    self._failure = self._failure or str(error)
                else:
                    # Observation errors cannot authorize settlement. Keep the
                    # error separate so a transient API outage can recover.
                    code = getattr(error, "status", None)
                    detail = f"HTTP {code}" if code is not None else type(error).__name__
                    self._observation_error = f"pod observation failed: {detail}"
                return self._status()

    def wait_for_settlement(self, timeout=None):
        if timeout is not None:
            _seconds("timeout", timeout, positive=False)
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            status = self.poll()
            if status.settled:
                return status
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError(f"pod settlement unconfirmed: {self.execution_id}")
            delay = self.launcher.poll_interval
            if deadline is not None:
                delay = min(delay, max(0, deadline - time.monotonic()))
            time.sleep(delay)

    def cancel(self):
        with self._lock:
            if self._settled:
                return self._status()
            self._cancel_requested = True
        try:
            return self.wait_for_settlement(self.launcher.cleanup_timeout)
        except TimeoutError as error:
            raise TaskSettlementError(str(error), self._status()) from error
