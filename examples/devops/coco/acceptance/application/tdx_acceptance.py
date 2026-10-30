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

import json
import re
import threading
from datetime import datetime, timezone
from pathlib import Path

from nvflare.apis.controller_spec import Task
from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.impl.controller import Controller
from nvflare.apis.shareable import Shareable, make_reply

TASK_NAME = "tdx_acceptance"
EXPECTED_VALUES = {"site-1": 3, "site-2": 7}
RESULT_FILE = "tdx_acceptance_result.json"


def validate_nonce(nonce):
    if not isinstance(nonce, str) or not re.fullmatch(r"[A-Za-z0-9_-]{16,128}", nonce):
        raise ValueError("nonce must be 16-128 ASCII letters, digits, underscores, or hyphens")
    return nonce


class AcceptanceExecutor(Executor):
    """Finite, image-baked computation; only designated protected sites may execute."""

    def __init__(self, nonce):
        super().__init__()
        self.nonce = validate_nonce(nonce)

    def execute(self, task_name, shareable, fl_ctx, abort_signal):
        if abort_signal.triggered:
            return make_reply(ReturnCode.TASK_ABORTED)
        if task_name != TASK_NAME:
            return make_reply(ReturnCode.TASK_UNKNOWN)
        site = fl_ctx.get_identity_name()
        if site not in EXPECTED_VALUES or shareable.get("nonce") != self.nonce:
            return make_reply(ReturnCode.EXECUTION_RESULT_ERROR)
        result = Shareable()
        result.update(client=site, nonce=self.nonce, value=EXPECTED_VALUES[site])
        return result


class AcceptanceController(Controller):
    """Validate transport-authenticated client identities and both nonce-bound results."""

    def __init__(self, nonce, timeout=540):
        super().__init__()
        self.nonce = validate_nonce(nonce)
        if type(timeout) is not int or not 1 <= timeout <= 540:
            raise ValueError("timeout must be an integer between 1 and 540 seconds")
        self.timeout = timeout
        self.values = {}
        self.errors = []
        self._lock = threading.Lock()

    def start_controller(self, fl_ctx):
        self.values = {}
        self.errors = []

    def stop_controller(self, fl_ctx):
        pass

    def _receive(self, client_task, fl_ctx):
        with self._lock:
            site = client_task.client.name
            response = client_task.result
            client_task.result = None
            if site not in EXPECTED_VALUES:
                self.errors.append(f"unexpected authenticated client: {site}")
            elif site in self.values:
                self.errors.append(f"duplicate response from {site}")
            elif not isinstance(response, Shareable) or response.get_return_code() != ReturnCode.OK:
                self.errors.append(f"unsuccessful response from {site}")
            elif (
                set(response) != {"__headers__", "client", "nonce", "value"}
                or response.get("client") != site
                or response.get("nonce") != self.nonce
                or type(response.get("value")) is not int
                or response.get("value") != EXPECTED_VALUES[site]
            ):
                self.errors.append(f"invalid identity, nonce, or value from {site}")
            else:
                self.values[site] = response["value"]

    def control_flow(self, abort_signal, fl_ctx):
        request = Shareable()
        request["nonce"] = self.nonce
        task = Task(name=TASK_NAME, data=request, timeout=self.timeout, result_received_cb=self._receive)
        try:
            self.broadcast_and_wait(
                task=task,
                targets=list(EXPECTED_VALUES),
                min_responses=2,
                wait_time_after_min_received=0,
                fl_ctx=fl_ctx,
                abort_signal=abort_signal,
            )
        except Exception:
            self.errors.append("task dispatch failed")
        with self._lock:
            if abort_signal.triggered:
                self.errors.append("job aborted")
            if self.values != EXPECTED_VALUES:
                self.errors.append("missing required protected-client responses")
            aggregate = sum(self.values.values())
            if aggregate != 10:
                self.errors.append("aggregate did not equal 10")
            result = {
                "schema_version": 1,
                "status": "failed" if self.errors else "passed",
                "job_id": fl_ctx.get_job_id(),
                "nonce": self.nonce,
                "values": dict(self.values),
                "aggregate": aggregate,
                "errors": list(self.errors),
                "completed_at": datetime.now(timezone.utc).isoformat(),
            }
        run_dir = Path(fl_ctx.get_workspace().get_run_dir(fl_ctx.get_job_id()))
        try:
            artifact = run_dir / RESULT_FILE
            temporary = run_dir / (RESULT_FILE + ".tmp")
            temporary.write_text(json.dumps(result, indent=2) + "\n")
            temporary.replace(artifact)
        except OSError:
            self.system_panic("could not persist acceptance result", fl_ctx)
            return
        if result["status"] != "passed":
            self.system_panic("TDX acceptance failed: " + "; ".join(result["errors"]), fl_ctx)

    def process_result_of_unknown_task(self, client, task_name, client_task_id, result, fl_ctx):
        with self._lock:
            self.errors.append(f"unexpected task result from {client.name}: {task_name}")
        self.system_panic("unexpected acceptance task result", fl_ctx)
