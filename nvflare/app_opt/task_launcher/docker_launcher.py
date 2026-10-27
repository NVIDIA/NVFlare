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

"""CJ proxy for a site-owned Docker task service. Never connects to Docker."""

import json
import math
import os
import threading
import time

from nvflare.apis.event_type import EventType
from nvflare.apis.task_launcher_spec import (
    TaskExecutionPhase,
    TaskExecutionStatus,
    TaskHandleSpec,
    TaskLaunchError,
    TaskLauncherSpec,
    TaskSettlementError,
)
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey, ReturnCode
from nvflare.fuel.f3.cellnet.fqcn import FQCN
from nvflare.fuel.f3.message import Message
from nvflare.private.fed.task_worker.protocol import _load_json

CHANNEL = "nvflare_docker_task"
SESSION_ENV = "NVFLARE_DOCKER_TASK_SESSION"
LISTENER_HOST_ENV = "NVFLARE_DOCKER_TASK_LISTENER_HOST"
MAX_CONTROL_BYTES = 16384
DEFAULT_MAX_RETAINED_ATTEMPTS = 10000


def _retained_limit(value):
    if type(value) is not int or value < 1:
        raise ValueError("max_retained_attempts must be a positive integer")
    return value


def encode_control(value):
    payload = json.dumps(value, allow_nan=False, separators=(",", ":"))
    if len(payload) > MAX_CONTROL_BYTES:
        raise ValueError("Docker task control message is too large")
    return payload


def decode_control(payload):
    if type(payload) is not str or len(payload) > MAX_CONTROL_BYTES or len(payload.encode("utf-8")) > MAX_CONTROL_BYTES:
        raise ValueError("Docker task control requires bounded JSON")

    value = _load_json(payload)
    if not isinstance(value, dict):
        raise ValueError("Docker task control requires a JSON object")
    return value


def _timeout(value, *, allow_none=False):
    if value is None and allow_none:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError("timeout must be finite and nonnegative")


class DockerTaskHandle(TaskHandleSpec):
    def __init__(self, launcher, request):
        self._launcher = launcher
        self._request = request
        self._execution_id = "docker:unresolved:" + request.attempt_id
        self._settled = None

    @property
    def request(self):
        return self._request

    @property
    def execution_id(self):
        return self._execution_id

    def _call(self, operation):
        if self._settled is not None:
            return self._settled
        response = self._launcher._call(operation, self.request)
        self._execution_id = response["execution_id"]
        values = dict(response["status"])
        values["phase"] = TaskExecutionPhase(values["phase"])
        status = TaskExecutionStatus(**values)
        if status.settled:
            self._settled = status
        if operation == "launch" and response.get("launch_failed"):
            raise RuntimeError(status.failure_reason or "Docker task failed during resource admission or launch")
        return status

    def poll(self):
        return self._call("poll")

    def cancel(self):
        try:
            status = self._call("cancel")
        except Exception as error:
            raise TaskSettlementError(
                "Docker task cancellation could not be confirmed",
                TaskExecutionStatus(TaskExecutionPhase.TERMINAL, cancel_requested=True, failure_reason=str(error)),
            ) from error
        if not status.settled:
            raise TaskSettlementError("Docker task settlement is unconfirmed", status)
        return status

    def wait_for_settlement(self, timeout=None):
        _timeout(timeout, allow_none=True)
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            status = self._call("settle")
            if status.settled:
                return status
            if status.phase == TaskExecutionPhase.TERMINAL and status.failure_reason:
                raise TaskSettlementError(status.failure_reason, status)
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("Docker task settlement timed out")
            time.sleep(0.05)


class DockerTaskLauncher(TaskLauncherSpec):
    """Only staged identity crosses the CP boundary; commands remain site-owned."""

    launch_mode = "docker"
    cleanup_owned_by_parent = True

    def __init__(self, rpc_timeout=30.0, max_retained_attempts=DEFAULT_MAX_RETAINED_ATTEMPTS):
        super().__init__()
        _timeout(rpc_timeout)
        if rpc_timeout == 0:
            raise ValueError("rpc_timeout must be positive")
        self.rpc_timeout = rpc_timeout
        self._cell = None
        self._session = None
        self._listener_host = None
        self._identities = set()
        self.max_retained_attempts = _retained_limit(max_retained_attempts)
        self._admission_closed = False
        self._lock = threading.Lock()

    def validate_site_config(self, site_config):
        """Require the matching CP authority in the canonical site file."""
        from nvflare.private.fed.client.task_execution_policy import site_job_launchers

        launchers = site_job_launchers(site_config)
        expected = "nvflare.app_opt.job_launcher.docker_launcher.ClientDockerJobLauncher"
        if len(launchers) != 1 or launchers[0][0] != expected:
            raise ValueError("Docker task execution requires exactly one matching Docker client JobLauncher")
        if not launchers[0][1].get("args", {}).get("task_worker_config"):
            raise ValueError("Docker task execution requires CP-owned task_worker_config")

    def get_worker_listener_host(self):
        if self._listener_host is None:
            raise RuntimeError("Docker task launcher has not bound its CP-issued worker listener host")
        return self._listener_host

    def handle_event(self, event_type, fl_ctx):
        if event_type == EventType.START_RUN:
            cell = fl_ctx.get_engine().get_cell()
            if self._cell is not None:
                if self._cell is not cell:
                    raise RuntimeError("Docker task launcher cannot move to a different CJ runtime")
                return
            session = os.environ.pop(SESSION_ENV, None)
            host = os.environ.pop(LISTENER_HOST_ENV, None)
            if cell is None or not session or not host:
                raise RuntimeError(
                    "Docker task execution requires a CP-issued launch session, listener host and CJ Cell"
                )
            if host in {"localhost", "127.0.0.1", "::1"} or any(char in host for char in "/:@ \t\r\n"):
                raise ValueError("Docker worker listener must use its CP-issued container DNS name")
            self._cell, self._session, self._listener_host = cell, session, host

    def _call(self, operation, request):
        if self._cell is None:
            raise RuntimeError("Docker task launcher has not bound its CJ runtime")
        reply = self._cell.send_request(
            target=FQCN.get_parent(self._cell.get_fqcn()),
            channel=CHANNEL,
            topic=operation,
            request=Message(
                payload=encode_control(
                    {
                        "session": self._session,
                        "identity": list(request.identity),
                        "gpu_count": request.resources.gpu_count,
                        "cpu_cores": request.resources.cpu_cores,
                        "memory_mb": request.resources.memory_mb,
                    }
                )
            ),
            timeout=self.rpc_timeout,
        )
        if reply.get_header(MessageHeaderKey.RETURN_CODE) != ReturnCode.OK:
            raise RuntimeError("Docker task authority did not acknowledge the request")
        return decode_control(reply.payload)

    def launch_task(self, request):
        with self._lock:
            if request.identity in self._identities:
                raise RuntimeError("task attempt already launched; resubmit its saved result")
            if self._admission_closed:
                raise RuntimeError("Docker task proxy admission is closed")
            if len(self._identities) >= self.max_retained_attempts:
                self._admission_closed = True
                raise RuntimeError("Docker task proxy retained-attempt limit reached; admission is closed")
            self._identities.add(request.identity)
        handle = DockerTaskHandle(self, request)
        try:
            handle._call("launch")
        except BaseException as error:
            # A timed-out RPC can still have created a container. Return ownership
            # through the error so the supervisor cancels/reconciles this identity.
            raise TaskLaunchError(f"Docker task launch failed or was not acknowledged: {error}", handle) from error
        return handle
