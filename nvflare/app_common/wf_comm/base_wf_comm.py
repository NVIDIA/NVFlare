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
from abc import ABC
from typing import Dict, List, Optional, Tuple

from nvflare.apis.client import Client
from nvflare.apis.controller_spec import ClientTask, ControllerSpec, OperatorMethod, SendOrder, Task, TaskOperatorKey
from nvflare.apis.dxo import DXO, DataKind
from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.app_common.abstract.fl_model import FLModel
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.app_event_type import AppEventType
from nvflare.app_common.utils.fl_model_utils import FLModelUtils
from nvflare.app_common.wf_comm.wf_comm_api import WFCommAPI
from nvflare.app_common.wf_comm.wf_comm_api_spec import (
    DATA,
    MIN_RESPONSES,
    RESULT,
    SITE_NAMES,
    STATUS,
    TARGET_SITES,
    TASK_NAME,
)
from nvflare.app_common.wf_comm.wf_communicator_spec import WFCommunicatorSpec
from nvflare.app_common.workflows.error_handle_utils import ABORT_WHEN_IN_ERROR
from nvflare.fuel.message.data_bus import DataBus
from nvflare.fuel.message.event_manager import EventManager
from nvflare.security.logging import secure_format_traceback


class BaseWFCommunicator(FLComponent, WFCommunicatorSpec, ControllerSpec, ABC):
    def __init__(
        self,
        task_timeout: int = 0,
        result_pull_interval: float = 0.2,
    ):
        super().__init__()
        self.strategy_fn_name = "run"
        self.clients = None
        self.task_timeout = task_timeout

        self.result_pull_interval = result_pull_interval
        self.engine = None
        self.fl_ctx = None
        self.data_bus: Optional[DataBus] = None
        self.event_manager: Optional[EventManager] = None

    def start_controller(self, fl_ctx: FLContext):
        self.fl_ctx = fl_ctx
        self.log_info(fl_ctx, "Initializing controller workflow.")

        self.data_bus = DataBus()
        self.event_manager = EventManager(self.data_bus)

        self.engine = self.fl_ctx.get_engine()
        self.clients = self.engine.get_clients()
        self.publish_comm_api()
        self.log_info(fl_ctx, "workflow controller started")

    def publish_comm_api(self):
        comm_api = WFCommAPI()
        comm_api.meta.update({SITE_NAMES: self.get_site_names()})
        self.data_bus.send_message("wf_comm_api", comm_api)

    def start_workflow(self, abort_signal, fl_ctx):
        try:
            fl_ctx.set_prop("abort_signal", abort_signal)
            func = getattr(self.get_strategy(), self.strategy_fn_name)
            func()

        except Exception as e:
            error_msg = secure_format_traceback()
            self.log_error(fl_ctx, error_msg)
            self.system_panic(error_msg, fl_ctx=fl_ctx)

    def stop_controller(self, fl_ctx: FLContext):
        pass

    def process_result_of_unknown_task(
        self, client: Client, task_name: str, client_task_id: str, result: Shareable, fl_ctx: FLContext
    ):
        pass

    def broadcast_to_peers_and_wait(self, pay_load):
        abort_signal = self.fl_ctx.get_prop("abort_signal")
        current_round = self.prepare_round_info(self.fl_ctx, pay_load)
        task, min_responses, targets = self.get_payload_task(pay_load)

        self.fl_ctx.set_prop("task_name", task.name)

        self.broadcast_and_wait(
            task=task,
            targets=targets,
            min_responses=min_responses,
            wait_time_after_min_received=0,
            fl_ctx=self.fl_ctx,
            abort_signal=abort_signal,
        )
        self.fire_event(AppEventType.ROUND_DONE, self.fl_ctx)
        self.log_info(self.fl_ctx, f"Round {current_round} finished.")

    def broadcast_to_peers(self, pay_load):
        task, min_responses, targets = self.get_payload_task(pay_load)
        self.broadcast(
            task=task, fl_ctx=self.fl_ctx, targets=targets, min_responses=min_responses, wait_time_after_min_received=0
        )

    def send_to_peers(self, pay_load, send_order: SendOrder = SendOrder.SEQUENTIAL):
        task, _, targets = self.get_payload_task(pay_load)
        self.send(task=task, fl_ctx=self.fl_ctx, targets=targets, send_order=send_order, task_assignment_timeout=0)

    def send_to_peers_and_wait(self, pay_load, send_order: SendOrder = SendOrder.SEQUENTIAL):
        abort_signal = self.fl_ctx.get_prop("abort_signal")
        task, _, targets = self.get_payload_task(pay_load)
        self.send_and_wait(
            task=task,
            fl_ctx=self.fl_ctx,
            targets=targets,
            send_order=send_order,
            task_assignment_timeout=0,
            abort_signal=abort_signal,
        )

    def relay_to_peers_and_wait(self, pay_load, send_order: SendOrder = SendOrder.SEQUENTIAL):
        abort_signal = self.fl_ctx.get_prop("abort_signal")
        task, min_responses, targets = self.get_payload_task(pay_load)
        self.relay_and_wait(
            task=task,
            fl_ctx=self.fl_ctx,
            targets=targets,
            send_order=send_order,
            task_assignment_timeout=0,
            task_result_timeout=0,
            dynamic_targets=True,
            abort_signal=abort_signal,
        )

    def relay_to_peers(self, pay_load, send_order: SendOrder = SendOrder.SEQUENTIAL):
        task, min_responses, targets = self.get_payload_task(pay_load)
        self.relay(
            task=task,
            fl_ctx=self.fl_ctx,
            targets=targets,
            send_order=send_order,
            task_assignment_timeout=0,
            task_result_timeout=0,
            dynamic_targets=True,
        )

    def prepare_round_info(self, fl_ctx, pay_load):
        current_round = pay_load.get(AppConstants.CURRENT_ROUND, 0)
        start_round = pay_load.get(AppConstants.START_ROUND, 0)
        num_rounds = pay_load.get(AppConstants.NUM_ROUNDS, 1)

        fl_ctx.set_prop(AppConstants.CURRENT_ROUND, current_round, private=True, sticky=True)
        fl_ctx.set_prop(AppConstants.NUM_ROUNDS, num_rounds, private=True, sticky=True)
        fl_ctx.set_prop(AppConstants.START_ROUND, start_round, private=True, sticky=True)
        if current_round == start_round:
            self.fire_event(AppEventType.ROUND_STARTED, fl_ctx)
        return current_round

    def get_payload_task(self, pay_load) -> Tuple[Task, int, List[str]]:
        min_responses = pay_load.get(MIN_RESPONSES)
        current_round = pay_load.get(AppConstants.CURRENT_ROUND, 0)
        start_round = pay_load.get(AppConstants.START_ROUND, 0)
        num_rounds = pay_load.get(AppConstants.NUM_ROUNDS, 1)
        targets = pay_load.get(TARGET_SITES, self.get_site_names())
        task_name = pay_load.get(TASK_NAME)

        data = pay_load.get(DATA, {})
        data_shareable = self.get_shareable(data)
        data_shareable.set_header(AppConstants.START_ROUND, start_round)
        data_shareable.set_header(AppConstants.CURRENT_ROUND, current_round)
        data_shareable.set_header(AppConstants.NUM_ROUNDS, num_rounds)
        data_shareable.add_cookie(AppConstants.CONTRIBUTION_ROUND, current_round)

        operator = {
            TaskOperatorKey.OP_ID: task_name,
            TaskOperatorKey.METHOD: OperatorMethod.BROADCAST,
            TaskOperatorKey.TIMEOUT: self.task_timeout,
        }

        task = Task(
            name=task_name,
            data=data_shareable,
            operator=operator,
            props={},
            timeout=self.task_timeout,
            before_task_sent_cb=None,
            result_received_cb=self._result_received_cb,
        )

        return task, min_responses, targets

    def get_shareable(self, data):
        if isinstance(data, FLModel):
            data_shareable: Shareable = FLModelUtils.to_shareable(data)
        elif data is None:
            data_shareable = Shareable()
        else:
            dxo = DXO(DataKind.RAW, data=data, meta={})
            data_shareable = dxo.to_shareable()
        return data_shareable

    def _result_received_cb(self, client_task: ClientTask, fl_ctx: FLContext):

        self.log_info(
            fl_ctx, f"\n{client_task.client.name} task:'{client_task.task.name}' result callback received.\n\n"
        )

        client_name = client_task.client.name
        task_name = client_task.task.name
        result = client_task.result
        rc = result.get_return_code()
        results: Dict[str, any] = {STATUS: rc}

        if rc == ReturnCode.OK:
            self.log_info(fl_ctx, f"Received result entries from client:{client_name} for task {task_name}")
            fl_model = FLModelUtils.from_shareable(result)
            results[RESULT] = {client_name: fl_model}
            payload = {task_name: results}
            self.event_manager.fire_event("TASK_RESULT", payload)
        else:
            self.handle_client_errors(rc, client_task, fl_ctx)

        # Cleanup task result
        client_task.result = None

    def get_site_names(self):
        return [client.name for client in self.clients]

    def handle_client_errors(self, rc: str, client_task: ClientTask, fl_ctx: FLContext):
        abort = ABORT_WHEN_IN_ERROR[rc]
        if abort:
            self.log_error(fl_ctx, f"error code = {rc}")
            self.system_panic(
                f"Failed in client-site for {client_task.client.name} during task {client_task.task.name}.",
                fl_ctx=fl_ctx,
            )
            self.log_error(fl_ctx, f"Execution failed for {client_task.client.name}")
        else:
            raise ValueError(f"Execution result is not received for {client_task.client.name}")
