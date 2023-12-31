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

import time
from abc import ABC
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from typing import Dict, List, Tuple

from nvflare.apis.client import Client
from nvflare.apis.controller_spec import ClientTask, OperatorMethod, Task, TaskOperatorKey, ControllerSpec
from nvflare.apis.dxo import DXO, DataKind
from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal
from nvflare.app_common.abstract.fl_model import FLModel
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.app_event_type import AppEventType
from nvflare.app_common.utils.fl_model_utils import FLModelUtils
from nvflare.app_common.workflows.error_handle_utils import ABORT_WHEN_IN_ERROR
from nvflare.app_common.workflows.wf_comm.wf_comm_api import WFCommAPI
from nvflare.app_common.workflows.wf_comm.wf_comm_api_spec import (
    CMD,
    CMD_ABORT,
    CMD_BROADCAST,
    CMD_RELAY,
    CMD_SEND,
    CMD_STOP,
    DATA,
    MIN_RESPONSES,
    PAYLOAD,
    RESULT,
    SITE_NAMES,
    STATUS,
    TARGET_SITES,
)
from nvflare.app_common.workflows.wf_comm.wf_queue import WFQueue
from nvflare.app_common.workflows.wf_comm.wf_spec import WF
from nvflare.fuel.utils import class_utils
from nvflare.security.logging import secure_format_traceback


class BaseWFController(FLComponent, ControllerSpec, ABC):
    def __init__(
            self,
            task_name: str,
            wf_class_path: str,
            wf_args: Dict,
            task_timeout: int = 0,
            comm_msg_pull_interval: float = 0.2,
    ):
        super().__init__()

        self.clients = None
        self.task_timeout = task_timeout
        self.task_name = task_name
        self.comm_msg_pull_interval = comm_msg_pull_interval
        self.wf_class_path = wf_class_path
        self.wf_args = wf_args
        self.wf_queue: WFQueue = WFQueue(ctrl_queue=Queue(), result_queue=Queue())
        self.wf: WF = class_utils.instantiate_class(self.wf_class_path, self.wf_args)
        self._thread_pool_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix=self.__class__.__name__)

        self.engine = None
        self.fl_ctx = None

    def start_controller(self, fl_ctx: FLContext):
        self.fl_ctx = fl_ctx
        self.log_info(fl_ctx, "Initializing controller workflow.")
        self.engine = self.fl_ctx.get_engine()
        self.clients = self.engine.get_clients()

        self.setup_wf_queue()

        self.log_info(fl_ctx, "workflow controller started")

    def setup_wf_queue(self):
        wf_comm_api = self.find_wf_comm_in_wf()
        wf_comm_api.set_queue(self.wf_queue)
        wf_comm_api.set_result_pull_interval(self.comm_msg_pull_interval)
        wf_comm_api.meta.update({SITE_NAMES: self.get_site_names()})

    def find_wf_comm_in_wf(self):
        attr_objs = [getattr(self.wf, attr_name, None) for attr_name in dir(self.wf)]
        wf_comm_attrs = [attr for attr in attr_objs if isinstance(attr, WFCommAPI)]
        if wf_comm_attrs:
            return wf_comm_attrs[0]
        else:
            raise RuntimeError(f"missing required attribute with type of 'WFCommAPI' in {self.wf.__class__.__name__}")

    def start_workflow(self, abort_signal, fl_ctx):
        try:
            future = self._thread_pool_executor.submit(self.ctrl_msg_loop, fl_ctx=fl_ctx, abort_signal=abort_signal)
            self.wf.run()
            self.stop_msg_queue("job completed", fl_ctx)
            future.result()
        except Exception as e:
            error_msg = secure_format_traceback()
            self.log_error(fl_ctx, error_msg)
            self.system_panic(error_msg, fl_ctx=fl_ctx)
        finally:
            wait_time = self.comm_msg_pull_interval + 0.05
            self.stop_msg_queue("job finished", fl_ctx, wait_time)

    def stop_msg_queue(self, stop_message, fl_ctx, wait_time: float = 0):
        self.wf_queue.stop(stop_message)
        self.log_info(fl_ctx, stop_message)

        if wait_time > 0:
            self.log_info(fl_ctx, f"wait for {wait_time} sec")
            time.sleep(wait_time)

    def stop_controller(self, fl_ctx: FLContext):
        self.stop_msg_queue("job completed", fl_ctx)
        if self._thread_pool_executor:
            self._thread_pool_executor.shutdown()

    def process_result_of_unknown_task(
            self, client: Client, task_name: str, client_task_id: str, result: Shareable, fl_ctx: FLContext
    ):
        pass

    def ctrl_msg_loop(self, fl_ctx: FLContext, abort_signal: Signal):

        if self.wf_queue is None:
            raise ValueError("WFQueue must provided")

        try:
            while True:
                if abort_signal.triggered:
                    break
                if not self.wf_queue.has_ctrl_msg():
                    time.sleep(self.comm_msg_pull_interval)
                else:
                    item = self.wf_queue.get_ctrl_msg()
                    if item is None:
                        self.log_warning(fl_ctx, "Ignore 'None' ctrl comm message")
                        continue

                    cmd = item.get(CMD, None)

                    if cmd is None:
                        msg = f"get None command, expecting {CMD} key'"
                        self.log_error(fl_ctx, msg)
                        raise ValueError(msg)

                    elif cmd == CMD_STOP:
                        msg = item.get(PAYLOAD)
                        self.log_info(fl_ctx, f"receive {CMD_STOP} command, {msg}")
                        break

                    elif cmd == CMD_ABORT:
                        msg = item.get(PAYLOAD)
                        self.log_info(fl_ctx, f"receive {CMD_ABORT} command, {msg}")
                        raise RuntimeError(msg)

                    elif cmd == CMD_BROADCAST:
                        pay_load = item.get(PAYLOAD)

                        current_round = self.prepare_round_info(fl_ctx, pay_load)
                        task, min_responses, targets = self.get_payload_task(pay_load)

                        self.broadcast_and_wait(
                            task=task,
                            targets=targets,
                            min_responses=min_responses,
                            wait_time_after_min_received=0,
                            fl_ctx=fl_ctx,
                            abort_signal=abort_signal,
                        )
                        self.fire_event(AppEventType.ROUND_DONE, fl_ctx)
                        self.log_info(fl_ctx, f"Round {current_round} finished.")

                    elif cmd == CMD_RELAY:
                        pay_load = item.get(PAYLOAD)
                        current_round = self.prepare_round_info(fl_ctx, pay_load)
                        task, min_responses, targets = self.get_payload_task(pay_load)

                        self.relay_and_wait(
                            task=task,
                            targets=targets,
                            fl_ctx=fl_ctx,
                            abort_signal=abort_signal,
                        )
                        self.fire_event(AppEventType.ROUND_DONE, fl_ctx)
                        self.log_info(fl_ctx, f"Round {current_round} finished.")

                    elif cmd == CMD_SEND:
                        raise NotImplementedError
                    else:
                        abort_signal.trigger(f"Unknown command '{cmd}'")
                        raise ValueError(f"Unknown command '{cmd}'")

                    if abort_signal.triggered:
                        self.log_debug(self.fl_ctx, f"task {self.task_name} aborted")
                        break
        except Exception as e:
            error_msg = secure_format_traceback()
            self.wf_queue.ask_abort(error_msg)
            self.log_error(fl_ctx, error_msg)
            self.system_panic(error_msg, fl_ctx=fl_ctx)

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

        data = pay_load.get(DATA, {})
        data_shareable = self.get_shareable(data)
        data_shareable.set_header(AppConstants.START_ROUND, start_round)
        data_shareable.set_header(AppConstants.CURRENT_ROUND, current_round)
        data_shareable.set_header(AppConstants.NUM_ROUNDS, num_rounds)
        data_shareable.add_cookie(AppConstants.CONTRIBUTION_ROUND, current_round)

        operator = {
            TaskOperatorKey.OP_ID: self.task_name,
            TaskOperatorKey.METHOD: OperatorMethod.BROADCAST,
            TaskOperatorKey.TIMEOUT: self.task_timeout,
        }

        task = Task(
            name=self.task_name,
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

        self.log_info(fl_ctx, f"{client_task.client.name} task:'{client_task.task.name}' result callback received.\n")

        client_name = client_task.client.name
        task_name = client_task.task.name
        result = client_task.result
        rc = result.get_return_code()
        results: Dict[str, any] = {STATUS: rc}

        if rc == ReturnCode.OK:
            self.log_info(fl_ctx, f"Received result entries from client:{client_name} for task {task_name}")
            fl_model = FLModelUtils.from_shareable(result)
            results[RESULT] = {client_name: fl_model}
            payload = {CMD: RESULT, PAYLOAD: {task_name: results}}
            self.wf_queue.put_result(payload)
        else:
            self.handle_client_errors(rc, client_task, fl_ctx)

        # Cleanup task result
        client_task.result = None

    def get_site_names(self):
        return [client.name for client in self.clients]

    def handle_client_errors(self,
                             rc: str,
                             client_task: ClientTask,
                             fl_ctx: FLContext):
        abort = ABORT_WHEN_IN_ERROR[rc]
        if abort:
            self.wf_queue.ask_abort(f"error code {rc} occurred")
            self.log_error(fl_ctx, f"error code = {rc}")
            self.system_panic(
                    f"Failed in client-site for {client_task.client.name} during task {client_task.task.name}.",
                    fl_ctx=fl_ctx,
            )
            self.log_error(fl_ctx, f"Execution failed for {client_task.client.name}")
        else:
            raise ValueError(f"Execution result is not received for {client_task.client.name}")
