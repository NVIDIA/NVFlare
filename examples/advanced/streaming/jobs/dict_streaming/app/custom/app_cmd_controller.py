# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
import time

from nvflare.apis.controller_spec import Client, ClientTask, Task
from nvflare.apis.fl_context import FLContext
from nvflare.apis.impl.controller import Controller
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal
from nvflare.app_common.streamers.file_streamer import FileStreamer, StreamContext

from .defs import STREAM_CHANNEL, TOPIC_INITIAL_FILE


class AppCommandController(Controller):
    def __init__(self, cmd_timeout=2, task_check_period: float = 0.5):
        Controller.__init__(self, task_check_period=task_check_period)
        self.cmd_timeout = cmd_timeout
        self.app_done = False
        self.abort_signal = None

    def start_controller(self, fl_ctx: FLContext):
        engine = fl_ctx.get_engine()
        engine.register_app_command(
            topic="hello",
            cmd_func=self.handle_hello,
        )
        engine.register_app_command(
            topic="avg",
            cmd_func=self.handle_avg,
        )
        engine.register_app_command(
            topic="bye",
            cmd_func=self.handle_bye,
        )
        engine.register_app_command(
            topic="echo",
            cmd_func=self.handle_echo,
        )
        engine.register_app_command(
            topic="stream_file",
            cmd_func=self.handle_stream_file_cmd,
        )
        engine.register_app_command(
            topic="rtr_file",
            cmd_func=self.handle_rtr_file_cmd,
        )

        FileStreamer.register_stream_processing(
            fl_ctx, STREAM_CHANNEL, "*", stream_status_cb=self._file_received, file_type="echo"
        )

    def _file_received(
        self,
        stream_ctx: StreamContext,
        fl_ctx: FLContext,
        file_type: str,
    ):
        peer_ctx = fl_ctx.get_peer_context()
        assert isinstance(peer_ctx, FLContext)
        peer_name = peer_ctx.get_identity_name()
        self.log_info(fl_ctx, f"stream file received from {peer_name}: {stream_ctx=} {file_type=}")

    def stop_controller(self, fl_ctx: FLContext):
        self.app_done = True

    def handle_stream_file_cmd(self, topic: str, data, fl_ctx: FLContext) -> dict:
        full_file_name = data
        result = FileStreamer.stream_file(
            channel=STREAM_CHANNEL,
            topic=TOPIC_INITIAL_FILE,
            targets=[],
            file_name=full_file_name,
            fl_ctx=fl_ctx,
            stream_ctx={"cmd_topic": topic},
        )
        return {"result": result}

    def handle_rtr_file_cmd(self, topic: str, data, fl_ctx: FLContext) -> dict:
        self.log_info(fl_ctx, f"handle command: {topic=}")
        s = Shareable()
        s["file_name"] = data
        task = Task(name="rtr_file", data=s, timeout=self.cmd_timeout)
        self.broadcast_and_wait(
            task=task,
            fl_ctx=fl_ctx,
            min_responses=2,
            abort_signal=self.abort_signal,
        )
        client_resps = {}
        for ct in task.client_tasks:
            assert isinstance(ct, ClientTask)
            resp = ct.result
            if resp is None:
                resp = "no answer"
            else:
                assert isinstance(resp, Shareable)
                self.log_info(fl_ctx, f"got resp {resp} from client {ct.client.name}")
                resp = resp.get_return_code()
            client_resps[ct.client.name] = resp
        return {"status": "OK", "data": client_resps}

    def handle_echo(self, topic: str, data, fl_ctx: FLContext) -> dict:
        engine = fl_ctx.get_engine()
        clients = engine.get_clients()
        reqs = {}
        for c in clients:
            r = Shareable()
            r["data"] = c.name
            reqs[c.name] = r
        replies = engine.multicast_aux_requests(
            topic="echo",
            target_requests=reqs,
            timeout=self.cmd_timeout,
            fl_ctx=fl_ctx,
        )
        result = {}
        if replies:
            for k, s in replies.items():
                assert isinstance(s, Shareable)
                result[k] = s.get("data", "no data")
        return result

    def handle_bye(self, topic: str, data, fl_ctx: FLContext) -> dict:
        self.app_done = True
        return {"status": "OK"}

    def handle_hello(self, topic: str, data, fl_ctx: FLContext) -> dict:
        self.log_info(fl_ctx, f"handle command: {topic=}")
        s = Shareable()
        s["data"] = data
        task = Task(name="hello", data=s, timeout=self.cmd_timeout)
        self.broadcast_and_wait(
            task=task,
            fl_ctx=fl_ctx,
            min_responses=2,
            abort_signal=self.abort_signal,
        )
        client_resps = {}
        for ct in task.client_tasks:
            assert isinstance(ct, ClientTask)
            resp = ct.result
            if resp is None:
                resp = "no answer"
            else:
                self.log_info(fl_ctx, f"got resp {resp} from client {ct.client.name}")
                resp = resp.get("data")
                if not resp:
                    resp = "greetings!"
            client_resps[ct.client.name] = resp
        return {"status": "OK", "data": client_resps}

    def handle_avg(self, topic: str, data, fl_ctx: FLContext) -> dict:
        s = Shareable()
        s["data"] = data
        task = Task(name="avg", data=s, timeout=self.cmd_timeout)
        self.broadcast_and_wait(
            task=task,
            fl_ctx=fl_ctx,
            min_responses=2,
            abort_signal=self.abort_signal,
        )
        client_resps = {}
        total = 0.0
        count = 0
        for ct in task.client_tasks:
            assert isinstance(ct, ClientTask)
            resp = ct.result
            if resp is None:
                resp = 0.0
            else:
                self.log_info(fl_ctx, f"got resp {resp} from client {ct.client.name}")
                resp = resp.get("data")
                if not resp:
                    resp = 0.0
                else:
                    total += resp
                count += 1
            client_resps[ct.client.name] = resp
        client_resps["avg"] = 0.0 if count == 0 else total / count
        return {"status": "OK", "data": client_resps}

    def control_flow(self, abort_signal: Signal, fl_ctx: FLContext):
        self.abort_signal = abort_signal
        while not abort_signal.triggered and not self.app_done:
            time.sleep(1.0)

    def process_result_of_unknown_task(
        self, client: Client, task_name: str, client_task_id: str, result: Shareable, fl_ctx: FLContext
    ):
        pass
