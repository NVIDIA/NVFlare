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

import numpy as np
import torch
from transformers import AutoModelForCausalLM

from nvflare.apis.controller_spec import Client, ClientTask, Task
from nvflare.apis.event_type import EventType
from nvflare.apis.fl_context import FLContext
from nvflare.apis.impl.controller import Controller
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal
from nvflare.app_common.streamers.container_retriever import ContainerRetriever
from nvflare.app_common.streamers.file_retriever import FileRetriever


class StreamingController(Controller):
    def __init__(self, retriever_mode=None, retriever_id=None, task_timeout=200, task_check_period: float = 0.5):
        Controller.__init__(self, task_check_period=task_check_period)
        self.retriever_mode = retriever_mode
        self.retriever_id = retriever_id
        self.retriever = None
        self.task_timeout = task_timeout

    def start_controller(self, fl_ctx: FLContext):
        self.file_name, self.model = self._get_test_model()
        if self.retriever_mode == "container":
            self.retriever.add_container("model", self.model)

    def stop_controller(self, fl_ctx: FLContext):
        pass

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        # perform initialization and checks
        if event_type == EventType.START_RUN:
            engine = fl_ctx.get_engine()
            if self.retriever_mode:
                c = engine.get_component(self.retriever_id)
                if self.retriever_mode == "container":
                    if not isinstance(c, ContainerRetriever):
                        self.system_panic(
                            f"invalid container_retriever {self.retriever_id}, wrong type: {type(c)}",
                            fl_ctx,
                        )
                        return
                    self.retriever = c
                elif self.retriever_mode == "file":
                    if not isinstance(c, FileRetriever):
                        self.system_panic(
                            f"invalid file_retriever {self.retriever_id}, wrong type: {type(c)}",
                            fl_ctx,
                        )
                        return
                    self.retriever = c
                else:
                    self.system_panic(
                        f"invalid retriever_mode {self.retriever_mode}",
                        fl_ctx,
                    )
                    return

    def control_flow(self, abort_signal: Signal, fl_ctx: FLContext):
        s = Shareable()
        # set shareable payload
        if self.retriever_mode == "container":
            s["model"] = "model"
        elif self.retriever_mode == "file":
            s["model"] = self.file_name
        else:
            s["model"] = self.model
        task = Task(name="retrieve_model", data=s, timeout=self.task_timeout)
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
    def _get_test_model():
        model_name = "meta-llama/llama-3.2-1b"
        # load model to dict
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map="auto",
            use_cache=False,
        )
        params = model.state_dict()
        for key in params:
            params[key] = params[key].cpu().numpy()

        # save params dict to a npz file
        file_name = "model.npz"
        np.savez(file_name, **params)

        return file_name, params
