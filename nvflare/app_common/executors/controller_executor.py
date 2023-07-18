# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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
from nvflare.apis.fl_constant import FLContextKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.impl.client_controller import ClientController
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal


class ControllerExecutor(Executor):
    def __init__(self, controller_id: str):
        self.controller_id = controller_id

        self.controller = None

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.START_RUN:
            self.initialize(fl_ctx)
        elif event_type == EventType.END_RUN:
            self.finalize(fl_ctx)

    def initialize(self, fl_ctx: FLContext):
        engine = fl_ctx.get_engine()
        self.controller: ClientController = engine.get_component(self.controller_id)

        if not isinstance(self.controller, ClientController):
            raise RuntimeError(f"{self.controller_id} must be ClientController.")

    def finalize(self, fl_ctx: FLContext):
        pass

    def execute(self, task_name: str, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        try:
            self.controller.control_flow(abort_signal, fl_ctx)

            result = fl_ctx.get_prop(FLContextKey.TASK_RESULT)

            return result
        except:
            return make_reply(ReturnCode.EMPTY_RESULT)

