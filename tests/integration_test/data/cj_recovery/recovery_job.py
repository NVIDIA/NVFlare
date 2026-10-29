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

"""One long task that saves progress across repeated CJ or task worker crashes."""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from nvflare.apis.controller_spec import Task
from nvflare.apis.dxo import DXO, DataKind, from_shareable
from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.impl.controller import Controller
from nvflare.apis.shareable import Shareable
from nvflare.app_common.executors.client_api_executor import ClientAPIExecutor


class DetachedTrainerExecutor(ClientAPIExecutor):
    def __init__(self):
        super().__init__(
            execution_mode="external_process",
            command="python3 -c 'from recovery_job import run_detached_trainer; run_detached_trainer()'",
            launch_once=False,
            params_exchange_format="numpy",
            server_expected_format="numpy",
        )


def run_detached_trainer():
    import nvflare.client as flare
    from nvflare.client.cell.bootstrap import BOOTSTRAP_FILE_ENV_VAR, BootstrapKey

    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    flare.init()
    flare.receive()
    bootstrap = json.loads(Path(os.environ[BOOTSTRAP_FILE_ENV_VAR]).read_text())
    record = Path("detached_trainers.json")
    launched = json.loads(record.read_text()) if record.exists() else []
    overlap = False
    if launched:
        try:
            os.kill(launched[-1]["pid"], 0)
            overlap = True
        except ProcessLookupError:
            pass
    launched.append({"pid": os.getpid(), "cj_pid": bootstrap[BootstrapKey.CJ_PID], "overlap": overlap})
    pending = record.with_suffix(".tmp")
    pending.write_text(json.dumps(launched))
    pending.replace(record)
    if len(launched) == 1:
        os.kill(bootstrap[BootstrapKey.CJ_PID], signal.SIGKILL)
    time.sleep(15)  # The owner watchdog must kill the first trainer during this wait.


class CheckpointExecutor(Executor):
    def execute(self, task_name, shareable, fl_ctx, abort_signal):
        run_dir = Path(fl_ctx.get_workspace().get_run_dir(fl_ctx.get_job_id()))
        checkpoint = run_dir / "recovery_checkpoint.json"
        state = json.loads(checkpoint.read_text()) if checkpoint.exists() else {"value": 0, "runs": []}
        if state["runs"]:
            with_child = state["runs"][-1]
            try:
                os.kill(with_child["child_pid"], 0)
            except ProcessLookupError:
                pass
            else:
                raise RuntimeError("replacement started before the old child exited")
        state["runs"].append(
            {
                "pid": os.getpid(),
                "job": fl_ctx.get_job_id(),
                "task": fl_ctx.get_prop(FLContextKey.TASK_ID),
                "attempt": fl_ctx.get_prop(FLContextKey.TASK_ATTEMPT_ID),
            }
        )
        state["value"] += len(state["runs"])
        if len(state["runs"]) <= shareable["crashes"]:
            # A same-group child survives CJ's SIGKILL briefly, as an applet can.
            child = subprocess.Popen(
                [sys.executable, "-c", "import time; print('ready', flush=True); time.sleep(2)"],
                stdout=subprocess.PIPE,
                text=True,
            )
            assert child.stdout.readline().strip() == "ready"
            child.stdout.close()
            assert os.getpgid(child.pid) == os.getpgrp()
            state["runs"][-1]["child_pid"] = child.pid
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
        data = DXO(DataKind.WEIGHTS, {"w": [0.0]}, meta={"crashes": self.crashes}).to_shareable()
        data["crashes"] = self.crashes
        task = Task("train", data, timeout=240, result_received_cb=self.received)
        self.broadcast_and_wait(task, fl_ctx, min_responses=1, abort_signal=abort_signal)
        if self.receipts != 1:
            self.system_panic(f"expected one result, got {self.receipts}", fl_ctx)

    def received(self, client_task, fl_ctx):
        self.receipts += 1
        result = client_task.result
        if "value" not in result:
            result = from_shareable(result).meta["recovery_state"]
        run_dir = Path(fl_ctx.get_workspace().get_run_dir(fl_ctx.get_job_id()))
        (run_dir / "recovery_result.json").write_text(
            json.dumps({"receipts": self.receipts, "value": result["value"], "runs": result["runs"]})
        )

    def stop_controller(self, fl_ctx):
        pass
