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

"""CP-only Docker authority, reusing the resident JobLauncher's daemon and mounts."""

import copy
import hashlib
import json
import logging
import os
import secrets
import threading
import time
from dataclasses import asdict, replace

from nvflare.apis.job_launcher_spec import JobReturnCode
from nvflare.apis.task_launcher_spec import (
    TaskExecutionPhase,
    TaskExecutionStatus,
    TaskLauncherError,
    TaskLaunchRequest,
    TaskResourceRequest,
    UnsupportedTaskResourceError,
)
from nvflare.app_opt.job_launcher import docker_launcher as backend
from nvflare.app_opt.task_launcher.docker_gpu import DockerGpuReservations, validate_partition
from nvflare.app_opt.task_launcher.docker_launcher import (
    CHANNEL,
    DEFAULT_MAX_RETAINED_ATTEMPTS,
    SESSION_ENV,
    _retained_limit,
    decode_control,
    encode_control,
)
from nvflare.fuel.common.exit_codes import ProcessExitCode
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey, ReturnCode
from nvflare.fuel.f3.cellnet.fqcn import FQCN
from nvflare.fuel.f3.message import Message
from nvflare.job_config.task_execution import reject_job_gpu_reservation, validate_task_resources
from nvflare.private.fed.client.task_execution_policy import FORBIDDEN_ENVIRONMENT_NAMES
from nvflare.private.fed.task_worker.protocol import TaskAttemptIdentity, _publish_exclusive, read_bootstrap
from nvflare.utils.job_launcher_utils import (
    get_compute_resource_manager_spec,
    get_job_launcher_spec,
    get_launcher_resource_spec,
    get_portable_resource_spec,
    portable_memory_to_mib,
)

OWNER_LABEL = "nvflare.task.owner"
CPU_ENVIRONMENT = {
    "CUDA_VISIBLE_DEVICES": "",
    "HIP_VISIBLE_DEVICES": "",
    "NVIDIA_VISIBLE_DEVICES": "none",
    "ROCR_VISIBLE_DEVICES": "",
}
DOCKER_AUTHORITY_ENVIRONMENT = {
    "DOCKER_HOST",
    "DOCKER_CONTEXT",
    "DOCKER_CERT_PATH",
    "DOCKER_TLS_VERIFY",
    "DOCKER_API_VERSION",
    "DOCKER_CONFIG",
}


class DockerOwnershipError(RuntimeError):
    """A native identity is no longer owned; automatic control stays fenced."""


def restrict_task_cj(container_kwargs, environment):
    """Keep job-supplied CJ code outside the CP's daemon and namespaces."""
    forbidden = (
        "devices",
        "device_cgroup_rules",
        "privileged",
        "volumes_from",
        "use_api_socket",
        "pid_mode",
        "userns_mode",
        "uts_mode",
        "network_mode",
        "cap_add",
        "group_add",
    )
    if any(container_kwargs.get(name) for name in forbidden):
        raise ValueError("Docker task CJ cannot inherit host devices, namespaces or daemon authority")
    for name in ("ipc_mode", "cgroupns"):
        if container_kwargs.get(name) not in (None, "private"):
            raise ValueError("Docker task CJ cannot inherit host or other-container namespaces")
    if container_kwargs.get("security_opt") not in (None, [], ["no-new-privileges:true"]):
        raise ValueError("Docker task CJ requires the fixed no-new-privileges security policy")
    if set(environment) & DOCKER_AUTHORITY_ENVIRONMENT:
        raise ValueError("Docker task CJ environment cannot contain Docker control credentials or endpoints")
    # Override image defaults as well as enforcing the site configuration.
    environment.update({name: "" for name in DOCKER_AUTHORITY_ENVIRONMENT})
    return {**container_kwargs, "cap_drop": ["ALL"], "security_opt": ["no-new-privileges:true"]}


def _worker_network(network):
    if (
        not isinstance(network, str)
        or not network
        or network in {"host", "none", "bridge"}
        or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-" for char in network)
    ):
        raise ValueError("Docker task workers require the CP-owned user-defined Docker network")
    return network


def validate_worker_config(config):
    if not isinstance(config, dict) or set(config) - {
        "image",
        "python_path",
        "environment",
        "nano_cpus",
        "mem_limit",
        "gpu_partition",
        "max_retained_attempts",
    }:
        raise ValueError("invalid Docker task_worker_config fields")
    config = dict(config)
    config.setdefault("python_path", backend.DockerJobLauncher.DEFAULT_PYTHON_PATH)
    config.setdefault("nano_cpus", 1_000_000_000)
    config.setdefault("mem_limit", 1024 * 1024 * 1024)
    config["max_retained_attempts"] = _retained_limit(
        config.get("max_retained_attempts", DEFAULT_MAX_RETAINED_ATTEMPTS)
    )
    for name in ("image", "python_path"):
        value = config.get(name)
        if not isinstance(value, str) or not value.strip() or "\x00" in value:
            raise ValueError(f"Docker task {name} must be a nonempty string without NUL")
    env = config.get("environment", {})
    TaskLaunchRequest("validate", "site", "task", "attempt", ("validate",), environment=env)
    if set(env) & (FORBIDDEN_ENVIRONMENT_NAMES | DOCKER_AUTHORITY_ENVIRONMENT | {SESSION_ENV}):
        raise ValueError("Docker task environment cannot contain credentials, bootstrap or GPU visibility")
    for name in ("nano_cpus", "mem_limit"):
        if name in config and (type(config[name]) is not int or config[name] <= 0):
            raise ValueError(f"Docker task {name} must be a positive integer")
    config["environment"] = {**env, **CPU_ENVIRONMENT, "HOME": "/tmp", "USER": "nvflare"}
    config["gpu_partition"] = validate_partition(config.get("gpu_partition"))
    return config


def resolve_task_resources(meta, site_name):
    """Resolve compute once through the common helpers for both CP and CJ."""
    validate_task_resources(meta, site_name, allow_gpu=True)
    reject_job_gpu_reservation(meta, site_name)
    requirements = get_compute_resource_manager_spec(meta, site_name)
    legacy = get_launcher_resource_spec(meta, site_name, "process")
    if (
        set(requirements) - {"num_of_gpus", "mem_per_gpu_in_GiB"}
        or requirements.get("mem_per_gpu_in_GiB", 0) != 0
        or legacy.get("mem_per_gpu_in_GiB", 0) != 0
    ):
        raise UnsupportedTaskResourceError("Docker task workers cannot enforce custom RM or per-GPU memory claims")
    portable = get_portable_resource_spec(meta, site_name)
    native = get_job_launcher_spec(meta, site_name, "docker")
    return TaskResourceRequest(
        cpu_cores=portable.get("num_of_cpus"),
        memory_mb=portable_memory_to_mib(portable["memory"]) if "memory" in portable else None,
        gpu_count=native.get(
            "num_of_gpus", portable.get("num_of_gpus", legacy.get("num_of_gpus", requirements.get("num_of_gpus", 0)))
        ),
    )


class _ContainerExecution:
    """A retained name/ID, with removal confirmation separate from leader exit."""

    def __init__(self, client, name, owner, log_path, gpu_reservations=None):
        self.client, self.name, self.owner, self.log_path = client, name, owner, log_path
        self.container_id = None
        self.removal_requested = False
        self.worker_creation_attempted = False
        self.worker_removed = False
        self.gpu_reservations = gpu_reservations
        self.launch_failed = False
        self.launching = False
        self.ownership_lost = False
        self._cancel_requested = threading.Event()
        self.status = TaskExecutionStatus(TaskExecutionPhase.QUEUED)
        self.lock = threading.RLock()

    @property
    def execution_id(self):
        return "docker:" + (self.container_id or self.name)

    def _get(self):
        if self.ownership_lost:
            raise DockerOwnershipError("Docker task ownership loss requires operator reconciliation")
        container = self.client.containers.get(self.container_id or self.name)
        if container.attrs.get("Config", {}).get("Labels", {}).get(OWNER_LABEL) != self.owner:
            self.ownership_lost = True
            raise DockerOwnershipError("Docker task ownership label does not match; refusing container control")
        if self.container_id is not None and container.id != self.container_id:
            self.ownership_lost = True
            raise DockerOwnershipError("Docker task physical identity changed")
        self.container_id = container.id
        return container

    def _failure(self, reason):
        with self.lock:
            self.status = replace(self.status, failure_reason=reason[:2048])
            return self.status

    def request_cancel(self):
        # No Docker call or execution lock: a blocked create/stop must not
        # prevent the trusted owner from closing admission.
        self._cancel_requested.set()

    def _api_failure(self, reason, error):
        # Backend exception text can include CP-only endpoint/configuration
        # details. Keep it in CP logs; CJ receives a bounded classification.
        # These exact framework errors contain fixed admission/ownership
        # explanations. Docker SDK and filesystem error text stays CP-only.
        if type(error) in (TaskLauncherError, DockerOwnershipError):
            message = str(error)[:2048]
        else:
            message = f"{reason} ({type(error).__name__})"
        if self.status.failure_reason != message:
            logging.getLogger(__name__).warning("%s: %s", reason, error)
        return self._failure(message)

    def _settle_resources(self):
        try:
            if self.gpu_reservations is not None:
                self.gpu_reservations.release()
            self.status = replace(self.status, settled=True)
        except Exception as error:
            self._api_failure("Docker GPU settlement failed", error)
        return self.status

    def poll(self):
        with self.lock:
            if self._cancel_requested.is_set():
                self.status = replace(self.status, cancel_requested=True)
            if self.launching:
                return self.status
            if self.status.settled:
                return self.status
            if self.worker_removed or not self.worker_creation_attempted:
                self.status = replace(self.status, phase=TaskExecutionPhase.TERMINAL)
                return self._settle_resources()
            try:
                container = self._get()
                state = container.attrs.get("State", {})
                if container.status in backend.TERMINAL_STATUSES:
                    code = state.get("ExitCode")
                    self.status = replace(
                        self.status,
                        phase=TaskExecutionPhase.TERMINAL,
                        exit_code=code if type(code) is int else None,
                    )
                    if container.status == backend.DockerStatus.DEAD or type(code) is not int or state.get("OOMKilled"):
                        self._failure("Docker task is dead, OOM-killed, or has no exit code")
                else:
                    phase = (
                        TaskExecutionPhase.QUEUED
                        if container.status == backend.DockerStatus.CREATED
                        else TaskExecutionPhase.RUNNING
                    )
                    self.status = replace(self.status, phase=phase)
                    if container.status == backend.DockerStatus.CREATED and self.status.cancel_requested:
                        # An acquired but never started container can be removed
                        # during cancellation, including a failed start call.
                        self.status = replace(self.status, phase=TaskExecutionPhase.TERMINAL)
            except backend.docker.errors.NotFound:
                if self.removal_requested and self.status.phase == TaskExecutionPhase.TERMINAL:
                    self.worker_removed = True
                    return self._settle_resources()
                else:
                    self.status = replace(self.status, phase=TaskExecutionPhase.TERMINAL)
                    self._failure("Docker task disappeared without confirmed owned removal")
            except DockerOwnershipError as error:
                self.status = replace(self.status, phase=TaskExecutionPhase.TERMINAL)
                self._failure(str(error))
            except Exception as error:
                self._api_failure("Docker task observation failed", error)
            return self.status

    def settle(self):
        with self.lock:
            status = self.poll()
            if status.settled or status.phase != TaskExecutionPhase.TERMINAL:
                return status
            try:
                container = self._get()
                # Recheck the daemon's current state. A cached PID-1 exit alone
                # cannot authorize removal/settlement of a restarted container.
                never_started = container.status == backend.DockerStatus.CREATED and self.status.cancel_requested
                if container.status not in backend.TERMINAL_STATUSES and not never_started:
                    return self._failure("Docker task changed state before settlement")
                if not self.removal_requested:
                    try:
                        data = container.logs(stdout=True, stderr=True, tail=1000)
                        _publish_exclusive(
                            os.path.dirname(self.log_path),
                            os.path.basename(self.log_path),
                            lambda stream: stream.write(data[-1024 * 1024 :]),
                        )
                    except Exception as error:
                        self._api_failure("Docker task log capture failed", error)
                self.removal_requested = True
                # Never remove bind sources or their payloads. Docker removal
                # owns the container's entire PID/cgroup namespace, not a PID.
                container.remove(force=True, v=True)
            except backend.docker.errors.NotFound:
                pass
            except Exception as error:
                self._api_failure("Docker task removal failed", error)
            return self.poll()

    def cancel(self):
        self.request_cancel()
        with self.lock:
            if self.status.settled:
                return self.status
            self.status = replace(self.status, cancel_requested=True)
            if self.launching:
                return self.status
            try:
                container = self._get()
                if container.status not in backend.TERMINAL_STATUSES | {backend.DockerStatus.CREATED}:
                    container.stop(timeout=0)
            except backend.docker.errors.NotFound:
                pass
            except Exception as error:
                self._api_failure("Docker task stop failed", error)
            return self.settle()


class _JobSession:
    def __init__(
        self,
        client,
        job_id,
        site_name,
        workspace,
        mounts,
        config,
        uid_gid,
        gpu_count=0,
        *,
        cpu_cores=None,
        memory_mb=None,
        network="nvflare-network",
        on_settled=None,
    ):
        self.client, self.job_id, self.site_name = client, job_id, site_name
        self.on_settled = on_settled
        self.workspace, self.mounts, self.config, self.uid_gid = workspace, mounts, config, uid_gid
        self.token = secrets.token_urlsafe(32)
        self.owner = secrets.token_hex(32)
        self.executions = {}
        self.closed = False
        self.gpu_count = gpu_count
        self.cpu_cores = cpu_cores
        self.memory_mb = memory_mb
        self.network = _worker_network(network)
        self.max_retained_attempts = config.get("max_retained_attempts", DEFAULT_MAX_RETAINED_ATTEMPTS)
        self.lock = threading.RLock()
        self.job_handle = None

    def launch(self, identity):
        with self.lock:
            if identity in self.executions:
                return self.executions[identity]
            if self.closed:
                raise RuntimeError("Docker task session is closed")
            # The caller can select an already staged attempt, never host paths,
            # commands, environment or Docker kwargs. All paths are CP-derived.
            TaskAttemptIdentity(*identity[:3], "unused", identity[3])
            self._check_retained_capacity()
            root = os.path.join(self.workspace, self.job_id, ".nvflare", "task-execution", "attempts")
            digest = hashlib.sha256(json.dumps([self.owner, *identity]).encode()).hexdigest()
            execution = _ContainerExecution(
                self.client,
                "nvflare-task-" + digest,
                digest,
                os.path.join(root, identity[3], "container.log"),
                DockerGpuReservations(
                    self.client, self.config["gpu_partition"], self.config["image"], self.uid_gid, digest, identity
                ),
            )
            # Remember cheap validation/busy rejections too. Replaying a rejected
            # identity must never acquire compute after its staged files change.
            execution.launch_failed = True
            execution.status = TaskExecutionStatus(
                TaskExecutionPhase.TERMINAL,
                settled=True,
                failure_reason="Docker task launch rejected before native creation",
            )
            self.executions[identity] = execution
            if any(not other.status.settled for other in self.executions.values() if other is not execution):
                raise RuntimeError("Docker task session still owns an unsettled execution")
            # Walk the lexical, CP-derived path with directory descriptors.
            # Canonicalizing a job-controlled parent would follow its symlinks
            # before validation, potentially into another job's workspace.
            path = os.path.join(root, identity[3], "bootstrap.json")
            bootstrap = read_bootstrap(path, resolve_parent=False)
            expected = bootstrap.identity
            if (expected.job_id, expected.site_name, expected.task_id, expected.attempt_id) != identity:
                raise ValueError("Docker task bootstrap identity does not match request")
            if bootstrap.workspace_root != self.workspace or bootstrap.artifact_root != root:
                raise ValueError("Docker task bootstrap uses unsupported workspace/artifact paths")
            # Fence before the mutating daemon call, including lost responses.
            execution.launch_failed = False
            execution.status = TaskExecutionStatus(TaskExecutionPhase.QUEUED)
            execution.launching = True
        return self._launch_worker(execution, identity, path)

    def _launch_worker(self, execution, identity, path):
        try:
            execution.gpu_reservations.acquire(self.gpu_count)
            execution.gpu_reservations.verify()
            devices = execution.gpu_reservations.devices
            environment = dict(self.config["environment"])
            gpu_kwargs = {}
            if devices:
                environment.update(
                    {
                        "NVIDIA_VISIBLE_DEVICES": ",".join(devices),
                        "CUDA_VISIBLE_DEVICES": ",".join(str(index) for index in range(len(devices))),
                        "NVIDIA_DRIVER_CAPABILITIES": "compute,utility",
                    }
                )
                gpu_kwargs["device_requests"] = [
                    {"Driver": "nvidia", "DeviceIDs": list(devices), "Capabilities": [["gpu"]]}
                ]
            _publish_exclusive(
                os.path.dirname(path),
                "resources.json",
                lambda stream: stream.write(
                    json.dumps(
                        {
                            "identity": list(identity),
                            "gpu_devices": list(devices),
                            "execution_name": execution.name,
                            "nano_cpus": self.config["nano_cpus"],
                            "mem_limit": self.config["mem_limit"],
                            "gpu_partition": self.config["gpu_partition"]["id"] if devices else None,
                            "gpu_reservations": [
                                {
                                    "device": reservation.device,
                                    "container_id": reservation.container_id,
                                    "name": reservation.name,
                                }
                                for reservation in execution.gpu_reservations.reservations
                            ],
                        },
                        sort_keys=True,
                    ).encode()
                ),
            )
            with execution.lock:
                if execution._cancel_requested.is_set():
                    return execution
                execution.worker_creation_attempted = True
            container = self.client.containers.create(
                self.config["image"],
                command=[
                    self.config["python_path"],
                    "-u",
                    "-m",
                    "nvflare.private.fed.app.client.task_worker_container",
                    "--bootstrap",
                    path,
                ],
                entrypoint=[],
                name=execution.name,
                labels={
                    OWNER_LABEL: execution.owner,
                    "nvflare.task.identity": json.dumps(identity),
                    "nvflare.task.resource": "worker",
                },
                environment=environment,
                mounts=self.mounts,
                user=self.uid_gid,
                working_dir=os.path.join(self.workspace, self.job_id),
                network=self.network,
                cap_drop=["ALL"],
                security_opt=["no-new-privileges:true"],
                detach=True,
                auto_remove=False,
                init=False,
                **{name: self.config[name] for name in ("nano_cpus", "mem_limit") if name in self.config},
                **gpu_kwargs,
            )
            execution.container_id = container.id
            if not execution._cancel_requested.is_set():
                container.start()
        except Exception as error:
            execution.launch_failed = True
            response_status = getattr(error, "status_code", None)
            if (
                execution.container_id is None
                and type(response_status) is int
                and 400 <= response_status < 500
                and response_status != 409
            ):
                # An explicit rejected create acquired no worker. Release
                # admitted devices; transport/5xx uncertainty still retains
                # ownership until an actual worker can be reconciled.
                execution.worker_creation_attempted = False
            execution._api_failure("Docker task create/start failed", error)
        finally:
            with execution.lock:
                execution.launching = False
            if execution.launch_failed or execution._cancel_requested.is_set():
                execution.cancel()
        return execution

    def cancel(self, identity):
        with self.lock:
            if identity not in self.executions:
                if self.closed:
                    raise RuntimeError("Docker task session is closed")
                # Cancellation can arrive before a timed-out launch RPC. Fence
                # its identity under the launch lock so a late callback cannot
                # acquire compute after cancellation was acknowledged.
                TaskAttemptIdentity(*identity[:3], "unused", identity[3])
                self._check_retained_capacity()
                execution = _ContainerExecution(self.client, "unlaunched-" + secrets.token_hex(16), self.owner, "")
                execution.status = TaskExecutionStatus(TaskExecutionPhase.TERMINAL, cancel_requested=True, settled=True)
                self.executions[identity] = execution
            execution = self.executions[identity]
        return execution

    def _check_retained_capacity(self):
        if len(self.executions) >= self.max_retained_attempts:
            self.request_close()
            raise RuntimeError("Docker task retained-attempt limit reached; admission is closed")

    def request_close(self):
        with self.lock:
            self.closed = True
            executions = list(self.executions.values())
        for execution in executions:
            execution.request_cancel()

    def close(self):
        self.request_close()
        with self.lock:
            executions = list(self.executions.values())
        unresolved = []
        for execution in executions:
            try:
                if execution.cancel().settled:
                    continue
            except Exception:
                pass
            unresolved.append(execution.execution_id)
        if unresolved:
            raise RuntimeError(f"Docker task settlement unconfirmed: {unresolved}")


class DockerTaskService:
    """One CP service; sessions are issued only by ClientDockerJobLauncher."""

    def __init__(self, cell, client, config, *, network="nvflare-network"):
        self.cell, self.client = cell, client
        self.network = _worker_network(network)
        self.config = validate_worker_config(config)
        self.sessions = {}
        self.closed = False
        self.lock = threading.RLock()
        cell.register_request_cb(channel=CHANNEL, topic="*", cb=self.dispatch)

    def add_job(self, job_id, site_name, workspace, mounts, uid_gid, job_meta=None):
        # Validate job path components before constructing a daemon-side bind.
        TaskAttemptIdentity(job_id, site_name, "validate", "validate", "validate")
        with self.lock:
            if self.closed:
                raise RuntimeError("Docker task service is closing")
            if job_id in self.sessions:
                raise RuntimeError("Docker task job session already exists")
            resources = resolve_task_resources(job_meta or {}, site_name)
            gpu_count = resources.gpu_count
            partition = self.config["gpu_partition"]
            if gpu_count and (partition is None or gpu_count > len(partition["devices"])):
                raise ValueError("Docker GPU task requires sufficient devices in an approved exclusive gpu_partition")
            config = copy.deepcopy(self.config)
            if resources.cpu_cores is not None:
                config["nano_cpus"] = int(resources.cpu_cores * 1_000_000_000)
            if resources.memory_mb is not None:
                config["mem_limit"] = resources.memory_mb * 1024 * 1024
            session = _JobSession(
                self.client,
                job_id,
                site_name,
                workspace,
                mounts,
                config,
                uid_gid,
                gpu_count,
                cpu_cores=resources.cpu_cores,
                memory_mb=resources.memory_mb,
                network=self.network,
                on_settled=self._remove_settled_job,
            )
            self.sessions[job_id] = session
            return session

    def _remove_settled_job(self, session):
        # Called only after the handle confirms CJ/worker removal and GPU
        # release. Old callbacks must never remove a newer logical session.
        with self.lock:
            if self.sessions.get(session.job_id) is session:
                del self.sessions[session.job_id]
            session.job_handle = None
            session.on_settled = None

    def close(self):
        with self.lock:
            self.closed = True
            sessions = list(self.sessions.values())
        for session in sessions:
            session.request_close()
            if session.job_handle is not None:
                session.job_handle.terminate()

    def dispatch(self, message):
        try:
            payload = decode_control(message.payload)
            required = {"session", "identity", "gpu_count"}
            if not required <= set(payload) or set(payload) - (required | {"cpu_cores", "memory_mb"}):
                raise ValueError("invalid Docker task request")
            identity = payload["identity"]
            if not isinstance(identity, list) or len(identity) != 4 or any(type(item) is not str for item in identity):
                raise ValueError("invalid Docker task identity")
            identity = tuple(identity)
            TaskAttemptIdentity(*identity[:3], "unused", identity[3])
            with self.lock:
                session = self.sessions[identity[0]]
            expected_origin = FQCN.join([self.cell.get_fqcn(), session.job_id])
            if (
                message.get_header(MessageHeaderKey.ORIGIN) != expected_origin
                or identity[1] != session.site_name
                or not isinstance(payload["session"], str)
                or not secrets.compare_digest(payload["session"], session.token)
                or type(payload["gpu_count"]) is not int
                or payload["gpu_count"] != session.gpu_count
                or payload.get("cpu_cores") != session.cpu_cores
                or payload.get("memory_mb") != session.memory_mb
            ):
                raise ValueError("Docker task session identity mismatch")
            TaskResourceRequest(payload.get("cpu_cores"), payload.get("memory_mb"), payload["gpu_count"])
            operation = message.get_header(MessageHeaderKey.TOPIC)
            if operation not in ("launch", "poll", "settle", "cancel"):
                raise ValueError("invalid Docker task operation")
            if operation == "launch":
                execution = session.launch(identity)
            elif operation == "cancel":
                execution = session.cancel(identity)
            else:
                with session.lock:
                    execution = session.executions[identity]
            status = execution.poll() if operation == "launch" else getattr(execution, operation)()
            if operation in ("cancel", "settle") and status.phase == TaskExecutionPhase.TERMINAL and not status.settled:
                # A later successful cleanup does not reopen task admission.
                with session.lock:
                    session.closed = True
            return Message(
                {MessageHeaderKey.RETURN_CODE: ReturnCode.OK},
                encode_control(
                    {
                        "execution_id": execution.execution_id,
                        # Cell peers need only wire values, not a registered enum
                        # decomposer or matching FOBS type whitelist.
                        "status": {**asdict(status), "phase": status.phase.value},
                        "launch_failed": execution.launch_failed,
                    }
                ),
            )
        except Exception:
            # Never echo session capabilities or arbitrary payload data.
            return Message({MessageHeaderKey.RETURN_CODE: ReturnCode.INVALID_REQUEST})


class TaskDockerJobHandle(backend.DockerJobHandle):
    """CP retains CJ and task cleanup until physical removal is confirmed.

    The monitor owns its session even after the CJ exits or the launch caller
    loses its reply. CP waiters retain job files/credentials until cleanup ends.
    A missing or replaced container is never proof of owned settlement.
    """

    def __init__(self, *args, task_session, cj_gpu_count=0, **kwargs):
        if type(cj_gpu_count) is not int or cj_gpu_count < 0:
            raise ValueError("Docker CJ GPU count must be a nonnegative integer")
        kwargs["watch_exit"] = False
        super().__init__(*args, **kwargs)
        self.task_session = task_session
        self._cj_gpu_count = cj_gpu_count
        self._cj_gpu_reservations = (
            DockerGpuReservations(
                self.docker_client,
                task_session.config["gpu_partition"],
                task_session.config["image"],
                task_session.uid_gid,
                task_session.owner,
                (task_session.job_id, task_session.site_name, "client-job"),
            )
            if cj_gpu_count
            else None
        )
        self._cj_creation_attempted = self.container_id is not None
        self._launch_complete = threading.Event()
        if self.container_id:
            self._launch_complete.set()
        self._creation_rejected = False
        self._terminate_requested = threading.Event()
        self._cleanup_done = threading.Event()
        self._wake = threading.Event()
        self._observed_state = None
        self._cj_return_code = None
        self._cj_removal_requested = False
        self._cj_removed = False
        self._cj_ownership_lost = False
        self._cleanup_lock = threading.Lock()
        self._last_cleanup_error = None
        task_session.job_handle = self
        if task_session.closed:
            self._terminate_requested.set()
        self._start_cleanup_monitor()

    def _start_cleanup_monitor(self):
        threading.Thread(
            target=self._monitor_cleanup, name=f"docker-task-owner-{self.container_name}", daemon=True
        ).start()

    @property
    def cj_gpu_devices(self):
        return self._cj_gpu_reservations.devices if self._cj_gpu_reservations is not None else ()

    def launch(self, create):
        try:
            if self._terminate_requested.is_set():
                self._creation_rejected = True
                return
            if self._cj_gpu_reservations is not None:
                self._cj_gpu_reservations.acquire(self._cj_gpu_count)
                self._cj_gpu_reservations.verify()
                if self._terminate_requested.is_set():
                    self._creation_rejected = True
                    return
            self._cj_creation_attempted = True
            container = create()
            self.container_id = container.id
            if not self._terminate_requested.is_set():
                container.start()
        except Exception as error:
            status = getattr(error, "status_code", None)
            self._creation_rejected = not self._cj_creation_attempted or (
                self.container_id is None and type(status) is int and 400 <= status < 500 and status != 409
            )
            self._terminate_requested.set()
            self.task_session.request_close()
            self.logger.warning("Docker task CJ create/start was not acknowledged; CP retains cleanup: %s", error)
        finally:
            self._launch_complete.set()
            self._wake.set()

    def _get_owned_cj(self):
        if self._cj_ownership_lost:
            raise DockerOwnershipError("Docker task CJ ownership loss requires operator reconciliation")
        container = self.docker_client.containers.get(self.container_id or self.container_name)
        labels = container.attrs.get("Config", {}).get("Labels", {})
        if labels.get(OWNER_LABEL) != self.task_session.owner:
            self._cj_ownership_lost = True
            self.task_session.request_close()
            raise DockerOwnershipError("Docker task CJ ownership changed; refusing container control")
        if self.container_id is not None and container.id != self.container_id:
            self._cj_ownership_lost = True
            self.task_session.request_close()
            raise DockerOwnershipError("Docker task CJ physical identity changed")
        self.container_id = container.id
        return container

    def _cleanup_once(self):
        with self._cleanup_lock:
            if self._terminate_requested.is_set() or self._cj_return_code is not None:
                self.task_session.request_close()
            container = None
            terminal = False
            cj_error = None
            if self._launch_complete.is_set():
                if self._creation_rejected:
                    self._cj_return_code = JobReturnCode.ABORTED
                    self._cj_removed = True
                if not self._cj_removed:
                    try:
                        container = self._get_owned_cj()
                        self._observed_state = container.status
                        never_started = (
                            container.status == backend.DockerStatus.CREATED and self._terminate_requested.is_set()
                        )
                        terminal = container.status in backend.TERMINAL_STATUSES or never_started
                        if not terminal:
                            # Stop the owned CJ independently of worker/GPU
                            # settlement. Their uncertainty must not prevent abort.
                            if self._terminate_requested.is_set():
                                container.stop(timeout=0)
                        else:
                            if self._terminate_requested.is_set():
                                self._cj_return_code = JobReturnCode.ABORTED
                            elif self._cj_return_code is None:
                                self._cj_return_code = self._resolve_terminal_return_code(container)
                                with self.task_session.lock:
                                    active = any(
                                        not execution.status.settled
                                        for execution in self.task_session.executions.values()
                                    )
                                if active:
                                    self._cj_return_code = ProcessExitCode.INFRASTRUCTURE_ERROR
                            self.task_session.request_close()
                    except backend.docker.errors.NotFound:
                        self.task_session.request_close()
                        if self._cj_removal_requested:
                            self._cj_removed = True
                        else:
                            cj_error = RuntimeError("Docker task CJ disappeared without confirmed owned removal")
                    except Exception as error:
                        cj_error = error
            # Also attempt worker cleanup when CJ observation/stop failed. Both
            # owners stay retained; neither failure authorizes resource release.
            worker_error = None
            if self.task_session.closed:
                try:
                    self.task_session.close()
                except Exception as error:
                    worker_error = error
            if cj_error is not None:
                raise cj_error
            if worker_error is not None:
                raise worker_error
            if not self._launch_complete.is_set():
                return
            if not self._cj_removed:
                if self._cj_gpu_reservations is not None:
                    self._cj_gpu_reservations.verify()
                if not terminal:
                    return
                self._cj_removal_requested = True
                container.remove(force=True, v=True)
                # Only a later owned-ID absence check confirms removal.
                return
            self.task_session.close()
            if self._cj_gpu_reservations is not None:
                self._cj_gpu_reservations.release()
            self._set_terminal_state(self._cj_return_code)
            if self.task_session.on_settled is not None:
                self.task_session.on_settled(self.task_session)
            self._cleanup_done.set()

    def _monitor_cleanup(self):
        while not self._cleanup_done.is_set():
            try:
                self._cleanup_once()
                self._last_cleanup_error = None
            except Exception as error:
                reason = str(error)
                if reason != self._last_cleanup_error:
                    self.logger.warning("CP retains Docker task job cleanup: %s", reason)
                    self._last_cleanup_error = reason
            self._wake.wait(0.25)
            self._wake.clear()

    def poll(self):
        return self._get_terminal_state() if self._cleanup_done.is_set() else JobReturnCode.UNKNOWN

    def can_restart(self):
        # Server redelivery reuses the same attempt ID. A new CJ cannot resume
        # that attempt's retained artifacts or replay fences without a durable
        # task recovery protocol, even after physical cleanup is confirmed.
        return False

    def wait(self):
        self._cleanup_done.wait()

    def terminate(self, heartbeat_cleanup=False):
        self._terminate_requested.set()
        self.task_session.request_close()
        self._wake.set()

    def enter_states(self, states_to_enter):
        states = states_to_enter if isinstance(states_to_enter, (tuple, list)) else [states_to_enter]
        deadline = None if self.timeout is None else time.monotonic() + self.timeout
        while not self._cleanup_done.is_set():
            if self._observed_state in states:
                return True
            if deadline is not None and time.monotonic() >= deadline:
                self.terminate()
                return False
            self._cleanup_done.wait(0.05)
        return False
