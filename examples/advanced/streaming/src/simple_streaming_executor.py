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

from nvflare.apis.event_type import EventType
from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.streamers.container_retriever import ContainerRetriever


class SimpleStreamingExecutor(Executor):
    def __init__(self, dict_retriever_id=None):
        Executor.__init__(self)
        self.dict_retriever_id = dict_retriever_id
        self.dict_retriever = None

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

    def execute(self, task_name: str, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        self.log_info(fl_ctx, f"got task {task_name}")
        if task_name == "retrieve_dict":
            name = shareable.get("name")
            if not name:
                self.log_error(fl_ctx, "missing name in request")
                return make_reply(ReturnCode.BAD_TASK_DATA)
            if not self.dict_retriever:
                self.log_error(fl_ctx, "no container retriever")
                return make_reply(ReturnCode.SERVICE_UNAVAILABLE)

            assert isinstance(self.dict_retriever, ContainerRetriever)
            rc, result = self.dict_retriever.retrieve_container(
                from_site="server",
                fl_ctx=fl_ctx,
                timeout=10.0,
                name=name,
            )
            if rc != ReturnCode.OK:
                self.log_error(fl_ctx, f"failed to retrieve dict {name}: {rc}")
                return make_reply(rc)

            self.log_info(fl_ctx, f"received container type: {type(result)} size: {len(result)}")
            return make_reply(ReturnCode.OK)
        else:
            self.log_error(fl_ctx, f"got unknown task {task_name}")
            return make_reply(ReturnCode.TASK_UNKNOWN)
