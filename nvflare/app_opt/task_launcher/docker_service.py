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

import hashlib
import json
import os
import secrets
import threading
from dataclasses import asdict, replace

from nvflare.apis.job_launcher_spec import JobReturnCode
from nvflare.apis.task_launcher_spec import TaskExecutionPhase, TaskExecutionStatus, TaskLaunchRequest
from nvflare.app_opt.job_launcher import docker_launcher as backend
from nvflare.app_opt.task_launcher.docker_gpu import DockerGpuReservations, validate_partition
from nvflare.app_opt.task_launcher.docker_launcher import CHANNEL, SESSION_ENV
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey, ReturnCode
from nvflare.fuel.f3.cellnet.fqcn import FQCN
from nvflare.fuel.f3.message import Message
from nvflare.private.fed.client.task_execution_policy import FORBIDDEN_ENVIRONMENT_NAMES
from nvflare.private.fed.task_worker.artifacts import FileTaskArtifactStore
from nvflare.private.fed.task_worker.protocol import (
    _MAX_BOOTSTRAP_BYTES,
    TaskAttemptIdentity,
    WorkerBootstrap,
    _load_json,
    _open_regular,
    _publish_exclusive,
)
from nvflare.utils.job_launcher_utils import get_job_launcher_spec, get_resource_manager_spec

OWNER_LABEL = "nvflare.task.owner"
CPU_ENVIRONMENT = {
    "CUDA_VISIBLE_DEVICES": "",
    "HIP_VISIBLE_DEVICES": "",
    "NVIDIA_VISIBLE_DEVICES": "none",
    "ROCR_VISIBLE_DEVICES": "",
}


def validate_worker_config(config):
    if not isinstance(config, dict) or set(config) - {
        "image",
        "python_path",
        "environment",
        "nano_cpus",
        "mem_limit",
        "gpu_partition",
    }:
        raise ValueError("invalid Docker task_worker_config fields")
    config = dict(config)
    config.setdefault("python_path", backend.DockerJobLauncher.DEFAULT_PYTHON_PATH)
    for name in ("image", "python_path"):
        value = config.get(name)
        if not isinstance(value, str) or not value.strip() or "\x00" in value:
            raise ValueError(f"Docker task {name} must be a nonempty string without NUL")
    env = config.get("environment", {})
    TaskLaunchRequest("validate", "site", "task", "attempt", ("validate",), environment=env)
    if set(env) & (FORBIDDEN_ENVIRONMENT_NAMES | {SESSION_ENV}):
        raise ValueError("Docker task environment cannot contain credentials, bootstrap or GPU visibility")
    for name in ("nano_cpus", "mem_limit"):
        if name in config and (type(config[name]) is not int or config[name] <= 0):
            raise ValueError(f"Docker task {name} must be a positive integer")
    config["environment"] = {**env, **CPU_ENVIRONMENT, "HOME": "/tmp", "USER": "nvflare"}
    config["gpu_partition"] = validate_partition(config.get("gpu_partition"))
    return config


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
        self.status = TaskExecutionStatus(TaskExecutionPhase.QUEUED)
        self.lock = threading.RLock()

    @property
    def execution_id(self):
        return "docker:" + (self.container_id or self.name)

    def _get(self):
        container = self.client.containers.get(self.container_id or self.name)
        if container.attrs.get("Config", {}).get("Labels", {}).get(OWNER_LABEL) != self.owner:
            raise RuntimeError("Docker task ownership label does not match; refusing container control")
        if self.container_id is not None and container.id != self.container_id:
            raise RuntimeError("Docker task physical identity changed")
        self.container_id = container.id
        return container

    def _failure(self, reason):
        self.status = replace(self.status, failure_reason=reason)
        return self.status

    def _settle_resources(self):
        try:
            if self.gpu_reservations is not None:
                self.gpu_reservations.release()
            self.status = replace(self.status, settled=True)
        except Exception as error:
            self._failure(f"Docker GPU settlement failed: {error}")
        return self.status

    def poll(self):
        with self.lock:
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
            except Exception as error:
                self._failure(f"Docker task observation failed: {error}")
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
                        self._failure(f"Docker task log capture failed: {error}")
                self.removal_requested = True
                # Never remove bind sources or their payloads. Docker removal
                # owns the container's entire PID/cgroup namespace, not a PID.
                container.remove(force=True, v=True)
            except backend.docker.errors.NotFound:
                pass
            except Exception as error:
                self._failure(f"Docker task removal failed: {error}")
            return self.poll()

    def cancel(self):
        with self.lock:
            if self.status.settled:
                return self.status
            self.status = replace(self.status, cancel_requested=True)
            try:
                container = self._get()
                if container.status not in backend.TERMINAL_STATUSES | {backend.DockerStatus.CREATED}:
                    container.stop(timeout=0)
            except backend.docker.errors.NotFound:
                pass
            except Exception as error:
                self._failure(f"Docker task stop failed: {error}")
            return self.settle()


class _JobSession:
    def __init__(self, client, job_id, site_name, workspace, mounts, config, uid_gid, gpu_count=0):
        self.client, self.job_id, self.site_name = client, job_id, site_name
        self.workspace, self.mounts, self.config, self.uid_gid = workspace, mounts, config, uid_gid
        self.token = secrets.token_urlsafe(32)
        self.owner = secrets.token_hex(32)
        self.executions = {}
        self.closed = False
        self.gpu_count = gpu_count
        self.lock = threading.RLock()

    def launch(self, identity):
        with self.lock:
            if identity in self.executions:
                return self.executions[identity]
            if self.closed:
                raise RuntimeError("Docker task session is closed")
            if any(not execution.status.settled for execution in self.executions.values()):
                raise RuntimeError("Docker task session still owns an unsettled execution")
            # The caller can select an already staged attempt, never host paths,
            # commands, environment or Docker kwargs. All paths are CP-derived.
            partial_identity = TaskAttemptIdentity(*identity[:3], "unused", identity[3])
            root = os.path.join(self.workspace, self.job_id, ".nvflare", "task-execution", "attempts")
            store = FileTaskArtifactStore(root)
            if store.root_dir != root:
                raise ValueError("Docker task artifact root must not contain symlinks")
            path = store.bootstrap_path(partial_identity)
            with _open_regular(path) as stream:
                encoded = stream.read(_MAX_BOOTSTRAP_BYTES + 1)
            if len(encoded) > _MAX_BOOTSTRAP_BYTES:
                raise ValueError("Docker task bootstrap is too large")
            bootstrap = WorkerBootstrap.from_dict(_load_json(encoded))
            expected = bootstrap.identity
            if (expected.job_id, expected.site_name, expected.task_id, expected.attempt_id) != identity:
                raise ValueError("Docker task bootstrap identity does not match request")
            if bootstrap.workspace_root != self.workspace or bootstrap.artifact_root != root:
                raise ValueError("Docker task bootstrap uses unsupported workspace/artifact paths")
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
            # Fence before the mutating daemon call, including lost responses.
            self.executions[identity] = execution
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
                    network_mode="none",
                    cap_drop=["ALL"],
                    security_opt=["no-new-privileges:true"],
                    detach=True,
                    auto_remove=False,
                    **{name: self.config[name] for name in ("nano_cpus", "mem_limit") if name in self.config},
                    **gpu_kwargs,
                )
                execution.container_id = container.id
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
                execution._failure(f"Docker task create/start failed: {error}")
                execution.cancel()
            return execution

    def cancel(self, identity):
        with self.lock:
            if identity not in self.executions:
                # Cancellation can arrive before a timed-out launch RPC. Fence
                # its identity under the launch lock so a late callback cannot
                # acquire compute after cancellation was acknowledged.
                TaskAttemptIdentity(*identity[:3], "unused", identity[3])
                execution = _ContainerExecution(self.client, "unlaunched-" + secrets.token_hex(16), self.owner, "")
                execution.status = TaskExecutionStatus(TaskExecutionPhase.TERMINAL, cancel_requested=True, settled=True)
                self.executions[identity] = execution
            execution = self.executions[identity]
        return execution

    def close(self):
        with self.lock:
            self.closed = True
            unresolved = [
                execution.execution_id for execution in self.executions.values() if not execution.cancel().settled
            ]
            if unresolved:
                raise RuntimeError(f"Docker task settlement unconfirmed: {unresolved}")


class DockerTaskService:
    """One CP service; sessions are issued only by ClientDockerJobLauncher."""

    def __init__(self, cell, client, config):
        self.cell, self.client = cell, client
        self.config = validate_worker_config(config)
        self.sessions = {}
        self.lock = threading.RLock()
        cell.register_request_cb(channel=CHANNEL, topic="*", cb=self.dispatch)

    def add_job(self, job_id, site_name, workspace, mounts, uid_gid, job_meta=None):
        # Validate job path components before constructing a daemon-side bind.
        TaskAttemptIdentity(job_id, site_name, "validate", "validate", "validate")
        with self.lock:
            if job_id in self.sessions:
                raise RuntimeError("Docker task job session already exists")
            spec = get_job_launcher_spec(job_meta or {}, site_name, "docker")
            if get_resource_manager_spec(job_meta or {}, site_name).get("num_of_gpus", 0):
                raise ValueError(
                    "Docker task GPU admission requires native launcher_spec; portable GPUs reserve the job"
                )
            gpu_count = spec.get("num_of_gpus", 0)
            if type(gpu_count) is not int or gpu_count < 0:
                raise ValueError("Docker task GPU count must be a nonnegative integer")
            partition = self.config["gpu_partition"]
            if gpu_count and (partition is None or gpu_count > len(partition["devices"])):
                raise ValueError("Docker GPU task requires sufficient devices in an approved exclusive gpu_partition")
            session = _JobSession(self.client, job_id, site_name, workspace, mounts, self.config, uid_gid, gpu_count)
            self.sessions[job_id] = session
            return session

    def dispatch(self, message):
        try:
            payload = message.payload
            if not isinstance(payload, dict) or set(payload) != {"session", "identity", "gpu_count"}:
                raise ValueError("invalid Docker task request")
            identity = payload["identity"]
            if not isinstance(identity, list) or len(identity) != 4 or any(type(item) is not str for item in identity):
                raise ValueError("invalid Docker task identity")
            identity = tuple(identity)
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
            ):
                raise ValueError("Docker task session identity mismatch")
            operation = message.get_header(MessageHeaderKey.TOPIC)
            if operation not in ("launch", "poll", "settle", "cancel"):
                raise ValueError("invalid Docker task operation")
            with session.lock:
                if operation == "launch":
                    execution = session.launch(identity)
                elif operation == "cancel":
                    execution = session.cancel(identity)
                else:
                    execution = session.executions[identity]
            status = execution.poll() if operation == "launch" else getattr(execution, operation)()
            if operation in ("cancel", "settle") and status.phase == TaskExecutionPhase.TERMINAL and not status.settled:
                # A later successful cleanup does not reopen task admission.
                with session.lock:
                    session.closed = True
            return Message(
                {MessageHeaderKey.RETURN_CODE: ReturnCode.OK},
                {
                    "execution_id": execution.execution_id,
                    # Cell peers need only wire values, not a registered enum
                    # decomposer or matching FOBS type whitelist.
                    "status": {**asdict(status), "phase": status.phase.value},
                    "launch_failed": execution.launch_failed,
                },
            )
        except Exception:
            # Never echo session capabilities or arbitrary payload data.
            return Message({MessageHeaderKey.RETURN_CODE: ReturnCode.INVALID_REQUEST})


class TaskDockerJobHandle(backend.DockerJobHandle):
    """Normal CP job teardown closes task admission and settles owned workers."""

    task_session = None

    def poll(self):
        code = super().poll()
        if code != JobReturnCode.UNKNOWN:
            self.task_session.close()
        return code

    def wait(self):
        super().wait()
        self.task_session.close()

    def terminate(self):
        try:
            self.task_session.close()
        finally:
            super().terminate()
