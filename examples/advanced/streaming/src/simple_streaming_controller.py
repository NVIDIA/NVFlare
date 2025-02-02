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

from random import randbytes

from nvflare.apis.controller_spec import Client, ClientTask, Task
from nvflare.apis.event_type import EventType
from nvflare.apis.fl_context import FLContext
from nvflare.apis.impl.controller import Controller
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal
from nvflare.app_common.streamers.container_retriever import ContainerRetriever


class SimpleStreamingController(Controller):
    def __init__(self, dict_retriever_id=None, task_timeout=60, task_check_period: float = 0.5):
        Controller.__init__(self, task_check_period=task_check_period)
        self.dict_retriever_id = dict_retriever_id
        self.dict_retriever = None
        self.task_timeout = task_timeout

    def start_controller(self, fl_ctx: FLContext):
        model = self._get_test_model()
        self.dict_retriever.add_container("model", model)

    def stop_controller(self, fl_ctx: FLContext):
        pass

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.START_RUN:
            engine = fl_ctx.get_engine()
            if self.dict_retriever_id:
                c = engine.get_component(self.dict_retriever_id)
                if not isinstance(c, ContainerRetriever):
                    self.system_panic(
                        f"invalid dict_retriever {self.dict_retriever_id}, wrong type: {type(c)}",
                        fl_ctx,
                    )
                    return
                self.dict_retriever = c

    def control_flow(self, abort_signal: Signal, fl_ctx: FLContext):
        s = Shareable()
        s["name"] = "model"
        task = Task(name="retrieve_dict", data=s, timeout=self.task_timeout)
        self.broadcast_and_wait(
            task=task,
            fl_ctx=fl_ctx,
            min_responses=1,
            abort_signal=abort_signal,
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

    def process_result_of_unknown_task(
        self, client: Client, task_name: str, client_task_id: str, result: Shareable, fl_ctx: FLContext
    ):
        pass

    @staticmethod
    def _get_test_model() -> dict:
        model = {}
        for i in range(10):
            key = f"layer-{i}"
            model[key] = randbytes(1024)

        return model
