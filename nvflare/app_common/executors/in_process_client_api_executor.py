# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
import threading
import time
from typing import Optional

from nvflare.apis.event_type import EventType
from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import FLContextKey, FLMetaKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.apis.utils.analytix_utils import create_analytic_dxo
from nvflare.apis.workspace import Workspace
from nvflare.app_common.abstract.params_converter import ParamsConverter
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.executors.task_script_runner import TaskScriptRunner
from nvflare.app_common.tracking.tracker_types import ANALYTIC_EVENT_TYPE
from nvflare.app_common.widgets.streaming import send_analytic_dxo
from nvflare.client.api_spec import CLIENT_API_KEY
from nvflare.client.config import ConfigKey, ExchangeFormat, TransferType
from nvflare.client.in_process.api import (
    TOPIC_ABORT,
    TOPIC_GLOBAL_RESULT,
    TOPIC_LOCAL_RESULT,
    TOPIC_LOG_DATA,
    TOPIC_STOP,
    InProcessClientAPI,
)
from nvflare.fuel.data_event.data_bus import DataBus
from nvflare.fuel.data_event.event_manager import EventManager
from nvflare.fuel.utils.validation_utils import check_object_type
from nvflare.security.logging import secure_format_traceback


class InProcessClientAPIExecutor(Executor):
    def __init__(
        self,
        task_script_path: str,
        task_script_args: str = "",
        task_wait_time: Optional[float] = None,
        result_pull_interval: float = 0.5,
        log_pull_interval: Optional[float] = None,
        params_exchange_format: str = ExchangeFormat.NUMPY,
        params_transfer_type: TransferType = TransferType.FULL,
        from_nvflare_converter_id: Optional[str] = None,
        to_nvflare_converter_id: Optional[str] = None,
        train_with_evaluation: bool = True,
        train_task_name: str = AppConstants.TASK_TRAIN,
        evaluate_task_name: str = AppConstants.TASK_VALIDATION,
        submit_model_task_name: str = AppConstants.TASK_SUBMIT_MODEL,
    ):
        super(InProcessClientAPIExecutor, self).__init__()
        self._abort = False
        self._client_api = None
        self._result_pull_interval = result_pull_interval
        self._log_pull_interval = log_pull_interval
        self._params_exchange_format = params_exchange_format
        self._params_transfer_type = params_transfer_type

        if not task_script_path or not task_script_path.endswith(".py"):
            raise ValueError(f"invalid task_script_path '{task_script_path}'")

        # only support main() for backward compatibility
        self._task_script_path = task_script_path
        self._task_script_args = task_script_args
        self._task_wait_time = task_wait_time

        # flags to indicate whether the launcher side will send back trained model and/or metrics
        self._train_with_evaluation = train_with_evaluation
        self._train_task_name = train_task_name
        self._evaluate_task_name = evaluate_task_name
        self._submit_model_task_name = submit_model_task_name

        self._from_nvflare_converter_id = from_nvflare_converter_id
        self._from_nvflare_converter: Optional[ParamsConverter] = None
        self._to_nvflare_converter_id = to_nvflare_converter_id
        self._to_nvflare_converter: Optional[ParamsConverter] = None

        self._engine = None
        self._task_fn_thread = None
        self._log_thread = None
        self._data_bus = DataBus()
        self._event_manager = EventManager(self._data_bus)
        self._data_bus.subscribe([TOPIC_LOCAL_RESULT], self.local_result_callback)
        self._data_bus.subscribe([TOPIC_LOG_DATA], self.log_result_callback)
        self._data_bus.subscribe([TOPIC_ABORT, TOPIC_STOP], self.to_abort_callback)
        self.local_result = None
        self._fl_ctx = None
        self._task_fn_path = None
        self._task_fn_wrapper = None

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.START_RUN:
            super().handle_event(event_type, fl_ctx)
            self._engine = fl_ctx.get_engine()
            self._fl_ctx = fl_ctx
            self._init_converter(fl_ctx)

            workspace: Workspace = fl_ctx.get_prop(FLContextKey.WORKSPACE_OBJECT)
            job_id = fl_ctx.get_prop(FLContextKey.CURRENT_JOB_ID)
            custom_dir = workspace.get_app_custom_dir(job_id)
            self._task_fn_wrapper = TaskScriptRunner(
                custom_dir=custom_dir, script_path=self._task_script_path, script_args=self._task_script_args
            )

            self._task_fn_thread = threading.Thread(target=self._task_fn_wrapper.run)
            meta = self._prepare_task_meta(fl_ctx, None)
            self._client_api = InProcessClientAPI(task_metadata=meta, result_check_interval=self._result_pull_interval)
            self._client_api.init()
            self._data_bus.put_data(CLIENT_API_KEY, self._client_api)

            self._task_fn_thread.start()

        elif event_type == EventType.END_RUN:
            self._event_manager.fire_event(TOPIC_STOP, "END_RUN received")
            if self._task_fn_thread:
                self._task_fn_thread.join()

    def execute(self, task_name: str, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        self.log_info(fl_ctx, f"execute for task ({task_name})")
        try:
            fl_ctx.set_prop("abort_signal", abort_signal)

            meta = self._prepare_task_meta(fl_ctx, task_name)
            self._client_api.set_meta(meta)

            shareable.set_header(FLMetaKey.JOB_ID, fl_ctx.get_job_id())
            shareable.set_header(FLMetaKey.SITE_NAME, fl_ctx.get_identity_name())
            if self._from_nvflare_converter is not None:
                shareable = self._from_nvflare_converter.process(task_name, shareable, fl_ctx)

            self.log_info(fl_ctx, "send data to peer")

            self.send_data_to_peer(shareable, fl_ctx)

            # wait for result
            self.log_info(fl_ctx, "Waiting for result from peer")
            while True:
                if abort_signal.triggered or self._abort is True:
                    # notify peer that the task is aborted
                    self._event_manager.fire_event(TOPIC_ABORT, f"{task_name}' is aborted, abort_signal_triggered")
                    return make_reply(ReturnCode.TASK_ABORTED)

                if self.local_result:
                    result = self.local_result
                    self.local_result = None

                    if not isinstance(result, Shareable):
                        self.log_error(fl_ctx, f"bad task result from peer: expect Shareable but got {type(result)}")
                        return make_reply(ReturnCode.EXECUTION_EXCEPTION)

                    current_round = shareable.get_header(AppConstants.CURRENT_ROUND)
                    if current_round is not None:
                        result.set_header(AppConstants.CURRENT_ROUND, current_round)
                    if self._to_nvflare_converter is not None:
                        result = self._to_nvflare_converter.process(task_name, result, fl_ctx)
                    return result
                else:
                    self.log_debug(fl_ctx, f"waiting for result, sleep for {self._result_pull_interval} secs")
                    time.sleep(self._result_pull_interval)

        except Exception as e:
            self.log_error(fl_ctx, secure_format_traceback())
            self._event_manager.fire_event(TOPIC_ABORT, f"{task_name}' failed: {secure_format_traceback()}")
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)

    def _prepare_task_meta(self, fl_ctx, task_name):
        job_id = fl_ctx.get_job_id()
        site_name = fl_ctx.get_identity_name()
        meta = {
            FLMetaKey.SITE_NAME: site_name,
            FLMetaKey.JOB_ID: job_id,
            ConfigKey.TASK_NAME: task_name,
            ConfigKey.TASK_EXCHANGE: {
                ConfigKey.TRAIN_WITH_EVAL: self._train_with_evaluation,
                ConfigKey.EXCHANGE_FORMAT: self._params_exchange_format,
                ConfigKey.TRANSFER_TYPE: self._params_transfer_type,
                ConfigKey.TRAIN_TASK_NAME: self._train_task_name,
                ConfigKey.EVAL_TASK_NAME: self._evaluate_task_name,
                ConfigKey.SUBMIT_MODEL_TASK_NAME: self._submit_model_task_name,
            },
        }
        return meta

    def send_data_to_peer(self, shareable, fl_ctx: FLContext):
        self.log_info(fl_ctx, "sending payload to peer")
        self._event_manager.fire_event(TOPIC_GLOBAL_RESULT, shareable)

    def _init_converter(self, fl_ctx: FLContext):
        engine = fl_ctx.get_engine()
        from_nvflare_converter: ParamsConverter = engine.get_component(self._from_nvflare_converter_id)
        if from_nvflare_converter is not None:
            check_object_type(self._from_nvflare_converter_id, from_nvflare_converter, ParamsConverter)
            self._from_nvflare_converter = from_nvflare_converter

        to_nvflare_converter: ParamsConverter = engine.get_component(self._to_nvflare_converter_id)
        if to_nvflare_converter is not None:
            check_object_type(self._to_nvflare_converter_id, to_nvflare_converter, ParamsConverter)
            self._to_nvflare_converter = to_nvflare_converter

    def check_output_shareable(self, task_name: str, shareable, fl_ctx: FLContext):
        """Checks output shareable after execute."""
        if not isinstance(shareable, Shareable):
            msg = f"bad task result from peer: expect Shareable but got {type(shareable)}"
            self.log_error(fl_ctx, msg)
            raise ValueError(msg)

    def local_result_callback(self, topic, data, databus):
        if not isinstance(data, Shareable):
            msg = f"bad task result from peer: expect Shareable but got {type(data)}"
            self.logger(msg)
            raise ValueError(msg)

        self.local_result = data

    def log_result_callback(self, topic, data, databus):
        result = data
        if result and not isinstance(result, dict):
            raise ValueError(f"invalid result format, expecting Dict, but get {type(result)}")

        if "key" in result:
            result["tag"] = result.pop("key")
        dxo = create_analytic_dxo(**result)

        # fire_fed_event = True w/o fed_event_converter somehow did not work
        with self._engine.new_context() as fl_ctx:
            send_analytic_dxo(self, dxo=dxo, fl_ctx=fl_ctx, event_type=ANALYTIC_EVENT_TYPE, fire_fed_event=False)

    def to_abort_callback(self, topic, data, databus):
        self._abort = True
