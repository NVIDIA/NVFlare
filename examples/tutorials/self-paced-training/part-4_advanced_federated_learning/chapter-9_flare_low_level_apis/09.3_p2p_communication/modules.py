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
from nvflare.apis.controller_spec import Task
from nvflare.apis.dxo import DXO, DataKind, from_shareable
from nvflare.apis.event_type import EventType
from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import ReservedKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.impl.controller import Controller
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal


class BasicController(Controller):
    def control_flow(self, abort_signal: Signal, fl_ctx: FLContext):
        self.broadcast_and_wait(
            task=Task(name="talk", data=Shareable()),
            targets=None,
            min_responses=0,
            fl_ctx=fl_ctx,
        )

    def start_controller(self, fl_ctx: FLContext):
        self.log_info(fl_ctx, "Starting the controller...")

    def stop_controller(self, fl_ctx: FLContext):
        self.log_info(fl_ctx, "Stopping the controller...")


class P2PExecutor(Executor):
    def execute(
        self,
        task_name: str,
        shareable: Shareable,
        fl_ctx: FLContext,
        abort_signal: Signal,
    ):
        if task_name == "talk":
            engine = fl_ctx.get_engine()
            identity_name = fl_ctx.get_identity_name()

            engine.send_aux_request(
                targets=[f"site-{i}" for i in range(3) if f"site-{i}" != identity_name],
                topic="hello",
                request=DXO(
                    data_kind=DataKind.APP_DEFINED,
                    data={
                        "message": f"Hello from {identity_name}",
                    },
                ).to_shareable(),
                timeout=0,
                fl_ctx=fl_ctx,
            )
            return make_reply(ReturnCode.OK)

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.START_RUN:
            engine = fl_ctx.get_engine()

            # Register the aux message handler
            engine.register_aux_message_handler(topic="hello", message_handle_func=self._handle_aux_request)

    def _handle_aux_request(self, topic: str, request: Shareable, fl_ctx: FLContext) -> Shareable:
        sender = request.get_peer_prop(key=ReservedKey.IDENTITY_NAME, default=None)
        received_message = from_shareable(request).data["message"]

        # log received message
        self.log_info(fl_ctx, f"Received message from {sender}: {received_message}")

        return make_reply(ReturnCode.OK)
