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

import os

import numpy as np

from nvflare.apis.event_type import EventType
from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.streamers.container_retriever import ContainerRetriever
from nvflare.app_common.streamers.file_retriever import FileRetriever


class StreamingExecutor(Executor):
    def __init__(self, retriever_mode=None, retriever_id=None, task_timeout=200):
        Executor.__init__(self)
        self.retriever_mode = retriever_mode
        self.retriever_id = retriever_id
        self.retriever = None
        self.task_timeout = task_timeout

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

    def execute(self, task_name: str, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        self.log_info(fl_ctx, f"got task {task_name}")
        if task_name == "retrieve_model":
            model = shareable.get("model")
            if not model:
                self.log_error(fl_ctx, "missing model info in request")
                return make_reply(ReturnCode.BAD_TASK_DATA)

            if self.retriever_mode is None:
                self.log_info(fl_ctx, f"received container type: {type(model)} size: {len(model)}")
                return make_reply(ReturnCode.OK)
            elif self.retriever_mode == "container":
                rc, result = self.retriever.retrieve_container(
                    from_site="server",
                    fl_ctx=fl_ctx,
                    timeout=self.task_timeout,
                    name=model,
                )
                if rc != ReturnCode.OK:
                    self.log_error(fl_ctx, f"failed to retrieve {model}: {rc}")
                    return make_reply(rc)
                self.log_info(fl_ctx, f"received container type: {type(result)} size: {len(result)}")
                return make_reply(ReturnCode.OK)
            elif self.retriever_mode == "file":
                rc, result = self.retriever.retrieve_file(
                    from_site="server",
                    fl_ctx=fl_ctx,
                    timeout=self.task_timeout,
                    file_name=model,
                )
                if rc != ReturnCode.OK:
                    self.log_error(fl_ctx, f"failed to retrieve file {model}: {rc}")
                    return make_reply(rc)
                # rename the received file to its original name
                rename_path = os.path.join(os.path.dirname(result), model)
                os.rename(result, rename_path)
                self.log_info(fl_ctx, f"received file: {result}, renamed to: {rename_path}")
                # Load local model
                result = dict(np.load(rename_path))
                self.log_info(fl_ctx, f"loaded file content type: {type(result)} size: {len(result)}")

                return make_reply(ReturnCode.OK)
        else:
            self.log_error(fl_ctx, f"got unknown task {task_name}")
            return make_reply(ReturnCode.TASK_UNKNOWN)
