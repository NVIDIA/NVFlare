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

"""Harmless jobs used by the secure server-command authorization regression."""

import json
import os
import time
from pathlib import Path

import nvflare
from nvflare.apis.controller_spec import Task
from nvflare.apis.executor import Executor
from nvflare.apis.impl.controller import Controller
from nvflare.apis.shareable import ReservedHeaderKey, Shareable
from nvflare.fuel.f3.cellnet.defs import CellChannel, MessageHeaderKey
from nvflare.fuel.f3.streaming.stream_const import StreamHeaderKey
from nvflare.private.defs import new_cell_message


def record(fl_ctx, name, data):
    run = Path(fl_ctx.get_workspace().get_run_dir(fl_ctx.get_job_id()))
    path = run / name
    pending = path.with_suffix(".tmp")
    pending.write_text(json.dumps(data))
    pending.replace(path)


class ControlExecutor(Executor):
    def execute(self, task_name, shareable, fl_ctx, abort_signal):
        if task_name == "attack":
            cell = fl_ctx.get_engine().get_cell()
            victim = shareable["victim"]
            assert victim != fl_ctx.get_job_id()
            target = f"server.{victim}"
            reply = cell.send_request(
                channel=CellChannel.SERVER_COMMAND,
                topic="abort",
                target=target,
                request=new_cell_message({}, Shareable()),
                timeout=10,
            )
            result = {
                "origin": cell.get_fqcn(),
                "target": target,
                "attacker_job": fl_ctx.get_job_id(),
                "victim_job": victim,
                "rc": reply.get_header(MessageHeaderKey.RETURN_CODE),
                "source": nvflare.__file__,
            }
            record(fl_ctx, "attack.json", result)
            return Shareable(result)
        run = Path(fl_ctx.get_workspace().get_run_dir(fl_ctx.get_job_id()))
        record(fl_ctx, "holding.json", {"job": fl_ctx.get_job_id(), "pid": os.getpid(), "source": nvflare.__file__})
        while not abort_signal.triggered and not (run / "release").exists():
            time.sleep(0.1)
        return Shareable({"job": fl_ctx.get_job_id(), "aborted": abort_signal.triggered})


class ControlController(Controller):
    def __init__(self, victim=""):
        super().__init__()
        self.victim = victim

    def start_controller(self, fl_ctx):
        pass

    def control_flow(self, abort_signal, fl_ctx):
        record(fl_ctx, "running.json", {"job": fl_ctx.get_job_id(), "pid": os.getpid(), "source": nvflare.__file__})
        data = Shareable({"victim": self.victim})
        task = Task("attack" if self.victim else "hold", data, timeout=90, result_received_cb=self.received)
        self.broadcast_and_wait(task, fl_ctx, min_responses=1, abort_signal=abort_signal)
        if self.victim:
            cell = fl_ctx.get_engine().get_cell()

            def claim_parent(request):
                if (
                    request.get_header(StreamHeaderKey.CHANNEL) == CellChannel.SERVER_COMMAND
                    and request.get_header(StreamHeaderKey.TOPIC) == "abort"
                ):
                    request.set_header(MessageHeaderKey.ORIGIN, "server")

            # Keep the job's real signed token; change only the claimed sender after transport stamping.
            cell.core_cell.add_outgoing_request_filter("*", "*", claim_parent)
            reply = cell.send_request(
                channel=CellChannel.SERVER_COMMAND,
                topic="abort",
                target=f"server.{self.victim}",
                request=new_cell_message({}, Shareable()),
                timeout=3,
            )
            record(
                fl_ctx,
                "forged_origin.json",
                {"sender": cell.get_fqcn(), "rc": reply.get_header(MessageHeaderKey.RETURN_CODE)},
            )

    def received(self, client_task, fl_ctx):
        record(fl_ctx, "received.json", {k: v for k, v in client_task.result.items() if k != ReservedHeaderKey.HEADERS})

    def stop_controller(self, fl_ctx):
        pass
