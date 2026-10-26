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

"""One long task that saves progress across repeated CJ crashes."""

import json
import os
import signal
from pathlib import Path

from nvflare.apis.controller_spec import Task
from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.impl.controller import Controller
from nvflare.apis.shareable import Shareable


class CheckpointExecutor(Executor):
    def execute(self, task_name, shareable, fl_ctx, abort_signal):
        run_dir = Path(fl_ctx.get_workspace().get_run_dir(fl_ctx.get_job_id()))
        checkpoint = run_dir / "recovery_checkpoint.json"
        state = json.loads(checkpoint.read_text()) if checkpoint.exists() else {"value": 0, "runs": []}
        state["runs"].append(
            {
                "pid": os.getpid(),
                "job": fl_ctx.get_job_id(),
                "task": fl_ctx.get_prop(FLContextKey.TASK_ID),
                "attempt": fl_ctx.get_prop(FLContextKey.TASK_ATTEMPT_ID),
            }
        )
        state["value"] += len(state["runs"])
        pending = checkpoint.with_suffix(".tmp")
        pending.write_text(json.dumps(state))
        pending.replace(checkpoint)
        if len(state["runs"]) <= shareable["crashes"]:
            os.kill(os.getpid(), signal.SIGKILL)
        return Shareable(state)


class RecoveryController(Controller):
    def __init__(self, crashes=4):
        super().__init__()
        self.crashes = crashes
        self.receipts = 0

    def start_controller(self, fl_ctx):
        pass

    def control_flow(self, abort_signal, fl_ctx):
        task = Task("train", Shareable({"crashes": self.crashes}), timeout=240, result_received_cb=self.received)
        self.broadcast_and_wait(task, fl_ctx, min_responses=1, abort_signal=abort_signal)
        if self.receipts != 1:
            self.system_panic(f"expected one result, got {self.receipts}", fl_ctx)

    def received(self, client_task, fl_ctx):
        self.receipts += 1
        result = client_task.result
        run_dir = Path(fl_ctx.get_workspace().get_run_dir(fl_ctx.get_job_id()))
        (run_dir / "recovery_result.json").write_text(
            json.dumps({"receipts": self.receipts, "value": result["value"], "runs": result["runs"]})
        )

    def stop_controller(self, fl_ctx):
        pass
