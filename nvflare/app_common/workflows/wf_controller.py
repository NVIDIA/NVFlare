# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

from typing import Dict

from nvflare.apis.fl_context import FLContext
from nvflare.apis.impl.controller import Controller
from nvflare.apis.signal import Signal
from nvflare.app_common.common_workflows.base_wf_controller import BaseWFController
from nvflare.app_common.workflows.wf_comm import WFCommAPI


class WFController(BaseWFController, Controller):
    def __init__(
        self,
        task_name: str,
        wf_class_path: str,
        wf_args: Dict,
        wf_fn_name: str = "run",
        task_timeout: int = 0,
        comm_msg_pull_interval: float = 0.2,
    ):
        super().__init__(task_name, wf_class_path, wf_args, wf_fn_name, task_timeout, comm_msg_pull_interval)

    def control_flow(self, abort_signal: Signal, fl_ctx: FLContext):
        self.start_workflow(abort_signal, fl_ctx)

    def publish_comm_api(self):
        super(WFController, self).publish_comm_api()
        comm_api: WFCommAPI = self.message_bus.receive_messages("wf_comm_api")
        comm_api.set_ctrl(self)
        self.message_bus.send_message("wf_comm_api", comm_api)
