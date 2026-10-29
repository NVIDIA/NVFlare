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

"""Job-scoped Cell interface to CP-owned Kubernetes task execution.

Only inert JSON is accepted. The CP installs each job's trusted backend before
starting CJ; callers cannot supply a Pod manifest or choose runtime authority.
Cleanup stays in CP after a CJ exits, disconnects, or loses its launch reply.
"""

import json
import logging
import os
import secrets
import threading
import time
from dataclasses import dataclass, field

from nvflare.apis.task_launcher_spec import (
    TaskExecutionPhase,
    TaskExecutionStatus,
    TaskHandleSpec,
    TaskLauncherError,
    TaskLaunchError,
    TaskLaunchRequest,
    TaskSettlementError,
)
from nvflare.app_opt.job_launcher.workspace_cell_transfer import ENV_WORKSPACE_OWNER_FQCN, ENV_WORKSPACE_TRANSFER_TOKEN
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey, ReturnCode
from nvflare.fuel.f3.cellnet.fqcn import FQCN
from nvflare.fuel.f3.cellnet.utils import make_reply, new_cell_message

_CHANNEL = "k8s_task_control"
_TOPIC = "request"
_MAX_MESSAGE = 65536
_MAX_CANCEL_FENCES = 128
_CJ_OBSERVATION_INTERVAL = 10.0
logger = logging.getLogger(__name__)


def _status_dict(status):
    return dict(
        phase=status.phase.value,
        exit_code=status.exit_code,
        settled=status.settled,
        cancel_requested=status.cancel_requested,
        failure_reason=status.failure_reason,
        retryable=status.retryable,
    )


def _terminal(reason=None):
    return TaskExecutionStatus(TaskExecutionPhase.TERMINAL, settled=True, failure_reason=reason)


@dataclass
class _Attempt:
    request: TaskLaunchRequest
    handle: object = None
    cancel: bool = False
    status: TaskExecutionStatus = field(default_factory=lambda: TaskExecutionStatus(TaskExecutionPhase.QUEUED))
    execution_id: str = "k8s:pending"


class _Job:
    def __init__(self, backend, job_id, site_name, token, origin):
        self.backend, self.job_id, self.site_name = backend, job_id, site_name
        self.token, self.origin = token, origin
        self.lock = threading.Lock()
        self.wake = threading.Event()
        self.attempts = {}
        self.cancel_fences = 0
        self.next_cj_observation = 0.0
        self.closed = False
        self.released = False
        self.on_done = None
        self.cj_name = self.cj_uid = None

    def attach_cj(self, name, uid):
        if not isinstance(uid, str) or not uid:
            raise ValueError("CP task owner requires the created CJ UID")
        with self.lock:
            self.cj_name, self.cj_uid = name, uid
            self.next_cj_observation = 0.0
        self.wake.set()

    def close(self):
        with self.lock:
            self.closed = True
            for attempt in self.attempts.values():
                attempt.cancel = True
        self.wake.set()

    def release(self):
        self.close()
        with self.lock:
            self.released = True
        self.wake.set()

    def observe_cj(self):
        with self.lock:
            name, uid, released = self.cj_name, self.cj_uid, self.released
            now = time.monotonic()
            if released or not name or now < self.next_cj_observation:
                return
            self.next_cj_observation = now + _CJ_OBSERVATION_INTERVAL
        try:
            pod = self.backend.core_v1.read_namespaced_pod(
                name=name,
                namespace=self.backend.namespace,
                _request_timeout=self.backend.request_timeout,
            )
            if (
                pod.metadata.uid != uid
                or pod.status.phase in ("Succeeded", "Failed")
                or pod.metadata.deletion_timestamp
            ):
                self.release()
        except Exception as error:
            if getattr(error, "status", None) == 404:
                self.release()
            # An API outage is not evidence that CJ has exited.

    def run(self):
        while True:
            self.observe_cj()
            with self.lock:
                attempts = list(self.attempts.values())
            for attempt in attempts:
                with self.lock:
                    if attempt.status.settled:
                        continue
                    cancel = attempt.cancel or self.closed
                    if attempt.handle is None:
                        if cancel:
                            attempt.status = _terminal("task cancelled before Pod creation")
                            continue
                if attempt.handle is None:
                    try:
                        handle = self.backend.launch_task(attempt.request)
                    except TaskLaunchError as error:
                        handle = error.handle
                    except Exception as error:
                        with self.lock:
                            attempt.status = _terminal(f"task launch rejected: {type(error).__name__}: {error}")
                        continue
                    with self.lock:
                        attempt.handle = handle
                        attempt.execution_id = handle.execution_id
                        cancel = attempt.cancel or self.closed
                try:
                    if cancel:
                        attempt.handle.request_cancel()
                    status = attempt.handle.poll()
                    with self.lock:
                        previous = attempt.status
                        attempt.status = status
                        attempt.execution_id = attempt.handle.execution_id
                    if (
                        status.phase == TaskExecutionPhase.TERMINAL
                        and not status.settled
                        and status.failure_reason != previous.failure_reason
                    ):
                        logger.error("CP retains unresolved task %s: %s", attempt.execution_id, status.failure_reason)
                except Exception as error:
                    logger.warning("CP task observation failed for %s: %s", self.job_id, type(error).__name__)
            with self.lock:
                done = self.released and all(a.status.settled for a in self.attempts.values())
            if done:
                self.on_done()
                return
            self.wake.wait(self.backend.poll_interval)
            self.wake.clear()


class K8sTaskService:
    """One service on the CP Cell, with a separate capability and owner per job."""

    def __init__(self, cell):
        self.cell = cell
        self.lock = threading.Lock()
        self.jobs = {}
        cell.register_request_cb(channel=_CHANNEL, topic=_TOPIC, cb=self._request)

    @classmethod
    def get_or_create(cls, cell):
        lock = cell.__dict__.setdefault("_k8s_task_service_lock", threading.Lock())
        with lock:
            service = cell.__dict__.get("_k8s_task_service")
            if service is None:
                service = cls(cell)
                cell.__dict__["_k8s_task_service"] = service
            return service

    def add_job(self, backend, job_id, site_name, token):
        job = _Job(backend, job_id, site_name, token, FQCN.join([self.cell.get_fqcn(), job_id]))
        with self.lock:
            if job_id in self.jobs:
                raise ValueError("CP already owns this Kubernetes task job")
            self.jobs[job_id] = job

        def forget():
            with self.lock:
                if self.jobs.get(job_id) is job:
                    del self.jobs[job_id]

        job.on_done = forget
        thread = threading.Thread(target=job.run, name="k8s-task-owner", daemon=True)
        job.thread = thread
        thread.start()
        return job

    def _request(self, message):
        # The channel deliberately does not accept arbitrary component/request objects.
        try:
            if not isinstance(message.payload, str) or len(message.payload) > _MAX_MESSAGE:
                raise ValueError("task control requires bounded JSON")
            data = json.loads(message.payload)
            if not isinstance(data, dict):
                raise ValueError("task control requires a JSON object")
            job_id, token = data.get("job_id"), data.get("token")
            if not isinstance(job_id, str) or not isinstance(token, str):
                raise ValueError("missing task capability")
            with self.lock:
                job = self.jobs.get(job_id)
            if (
                job is None
                or not secrets.compare_digest(token, job.token)
                or message.get_header(MessageHeaderKey.ORIGIN) != job.origin
            ):
                return make_reply(ReturnCode.UNAUTHENTICATED)
            action = data.get("action")
            if action == "close" and set(data) == {"job_id", "token", "action"}:
                job.close()
                return make_reply(ReturnCode.OK, body=json.dumps({"closed": True}))
            fields = {"job_id", "token", "action", "task_id", "attempt_id"}
            if action == "launch":
                fields |= {"bootstrap", "cwd"}
            if set(data) != fields or action not in {"launch", "poll", "cancel"}:
                raise ValueError("unsupported task control fields or action")
            for key in ("task_id", "attempt_id"):
                if not isinstance(data[key], str) or not data[key] or len(data[key]) > 256 or "\x00" in data[key]:
                    raise ValueError("invalid attempt identity")
            identity = (data["task_id"], data["attempt_id"])
            with job.lock:
                attempt = job.attempts.get(identity)
                if attempt is None and action in {"launch", "cancel"} and len(job.attempts) >= job.backend.max_attempts:
                    # Retain every old identity to fence delayed/replayed launches.
                    # This includes cheap launches rejected before any Pod exists.
                    job.closed = True
                    for known in job.attempts.values():
                        known.cancel = True
                    job.wake.set()
                    raise ValueError("task attempt history limit reached; job task admission is closed")
                if action == "launch":
                    from nvflare.app_opt.task_launcher.k8s_launcher import WORKER_MODULE, _within

                    for key in ("bootstrap", "cwd"):
                        _within(job.backend.workspace_root, data[key])
                    request = TaskLaunchRequest(
                        job.job_id,
                        job.site_name,
                        *identity,
                        ("CP", "-m", WORKER_MODULE, "--bootstrap", data["bootstrap"], "--parent_pid", "1"),
                        cwd=data["cwd"],
                    )
                    if attempt is not None:
                        if attempt.request.argv != request.argv or attempt.request.cwd != request.cwd:
                            raise ValueError("attempt identity already bound to another request")
                    elif job.closed:
                        raise ValueError("job task admission is closed")
                    elif any(not a.status.settled for a in job.attempts.values()):
                        raise ValueError("job already owns an unsettled task")
                    else:
                        attempt = _Attempt(request)
                        job.attempts[identity] = attempt
                elif attempt is None:
                    if action != "cancel":
                        raise ValueError("unknown attempt")
                    if job.closed:
                        raise ValueError("job task admission is closed")
                    if job.cancel_fences >= _MAX_CANCEL_FENCES:
                        # Do not evict a fence: a delayed launch could then run.
                        # Seal admission instead, keeping existing cleanup live.
                        job.closed = True
                        for known in job.attempts.values():
                            known.cancel = True
                        job.wake.set()
                        raise ValueError("too many unknown task cancellations; job task admission is closed")
                    job.cancel_fences += 1
                    # Fence a launch delayed behind a cancellation/lost launch reply.
                    request = TaskLaunchRequest(job.job_id, job.site_name, *identity, ("cancelled",))
                    attempt = _Attempt(request, cancel=True, status=_terminal("task cancelled before launch"))
                    job.attempts[identity] = attempt
                if action == "cancel":
                    attempt.cancel = True
                result = _status_dict(attempt.status)
                result["execution_id"] = attempt.execution_id
            job.wake.set()
            return make_reply(ReturnCode.OK, body=json.dumps(result))
        except (ValueError, TypeError, KeyError) as error:
            return make_reply(ReturnCode.INVALID_REQUEST, error=str(error))


class K8sTaskClient:
    def __init__(self, cell, job_id, site_name, request_timeout, poll_interval, cleanup_timeout):
        self.cell, self.job_id, self.site_name = cell, job_id, site_name
        self.owner = os.environ.get(ENV_WORKSPACE_OWNER_FQCN)
        self.token = os.environ.get(ENV_WORKSPACE_TRANSFER_TOKEN)
        if not self.owner or not self.token:
            raise RuntimeError("Kubernetes task control requires CP-provisioned job credentials")
        self.request_timeout, self.poll_interval, self.cleanup_timeout = request_timeout, poll_interval, cleanup_timeout
        self.closed = False

    def call(self, action, **fields):
        payload = json.dumps(dict(job_id=self.job_id, token=self.token, action=action, **fields))
        reply = self.cell.send_request(
            target=self.owner,
            channel=_CHANNEL,
            topic=_TOPIC,
            request=new_cell_message({}, payload),
            timeout=self.request_timeout,
        )
        if reply.get_header(MessageHeaderKey.RETURN_CODE) != ReturnCode.OK:
            raise TaskLauncherError(
                f"CP task {action} rejected or unavailable: {reply.get_header(MessageHeaderKey.RETURN_CODE)}"
            )
        return json.loads(reply.payload)

    def launch(self, request):
        if self.closed:
            raise TaskLauncherError("Kubernetes task proxy is closed")
        if (request.job_id, request.site_name) != (self.job_id, self.site_name):
            raise ValueError("task request belongs to a different CJ")
        from nvflare.app_opt.task_launcher.k8s_launcher import WORKER_MODULE

        if (
            len(request.argv) != 7
            or request.argv[1:4] != ("-m", WORKER_MODULE, "--bootstrap")
            or request.argv[5] != "--parent_pid"
            or not request.argv[6].isdigit()
            or request.environment
            or not request.resources.is_empty()
        ):
            raise ValueError("Kubernetes requires the common staged worker request")
        handle = K8sRemoteTaskHandle(self, request)
        try:
            handle.observe(
                self.call(
                    "launch",
                    task_id=request.task_id,
                    attempt_id=request.attempt_id,
                    bootstrap=request.argv[4],
                    cwd=request.cwd,
                )
            )
        except Exception as error:
            # The request can have reached CP even when its reply was lost.
            raise TaskLaunchError("CP task launch reply unavailable", handle) from error
        return handle

    def close(self):
        self.closed = True
        self.call("close")


class K8sRemoteTaskHandle(TaskHandleSpec):
    def __init__(self, client, request):
        self.client, self._request = client, request
        self._execution_id = "k8s:pending"
        self._status = TaskExecutionStatus(TaskExecutionPhase.QUEUED)
        self._cancel_requested = False

    @property
    def request(self):
        return self._request

    @property
    def execution_id(self):
        return self._execution_id

    def observe(self, result):
        self._execution_id = result.pop("execution_id")
        result["phase"] = TaskExecutionPhase(result["phase"])
        self._status = TaskExecutionStatus(**result)
        return self._status

    def poll(self):
        try:
            return self.observe(
                self.client.call(
                    "cancel" if self._cancel_requested else "poll",
                    task_id=self.request.task_id,
                    attempt_id=self.request.attempt_id,
                )
            )
        except Exception as error:
            status = self._status
            return TaskExecutionStatus(
                status.phase,
                exit_code=status.exit_code,
                cancel_requested=self._cancel_requested,
                settled=status.settled,
                failure_reason=f"CP task observation unavailable: {type(error).__name__}",
                retryable=status.retryable,
            )

    def wait_for_settlement(self, timeout=None):
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            status = self.poll()
            if status.settled:
                return status
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("CP task settlement unconfirmed; cleanup remains owned by CP")
            time.sleep(self.client.poll_interval)

    def cancel(self):
        self._cancel_requested = True
        try:
            return self.wait_for_settlement(self.client.cleanup_timeout)
        except TimeoutError as error:
            raise TaskSettlementError(str(error), self._status) from error
