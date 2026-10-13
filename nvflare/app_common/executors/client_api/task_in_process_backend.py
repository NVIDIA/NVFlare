# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Run one in-process Client API assignment on the task worker's main thread."""

import logging

from nvflare.apis.fl_constant import FLContextKey, FLMetaKey
from nvflare.apis.utils.analytix_utils import create_analytic_dxo
from nvflare.app_common.executors.client_api.backend_spec import ClientAPIBackendSpec
from nvflare.app_common.executors.client_api.script_utils import prepare_task_metadata
from nvflare.app_common.executors.client_api_executor import ClientAPIExecutor, ExecutionMode
from nvflare.app_common.executors.task_script_runner import TaskScriptRunner
from nvflare.client.api_spec import CLIENT_API_KEY
from nvflare.client.decomposers import register_framework_decomposers
from nvflare.client.in_process.adapters import TaskInProcessAdapter
from nvflare.client.in_process.api import InProcessClientAPI
from nvflare.fuel.data_event.data_bus import DataBus


class TaskInProcessBackend(ClientAPIBackendSpec):
    def __init__(self, *, on_started=None, on_result=None, on_log=None):
        self._on_started, self._on_result, self._on_log = on_started, on_result, on_log
        self._api = None
        self._runner = None
        self._executed = False

    def initialize(self, context, fl_ctx):
        if (
            type(context.executor) is not ClientAPIExecutor
            or context.executor.execution_mode != ExecutionMode.IN_PROCESS
        ):
            raise RuntimeError("task lifetime supports the in_process ClientAPIExecutor only")
        if not context.task_script_path or not context.task_script_path.endswith(".py"):
            raise ValueError("task Client API requires a .py script")
        try:
            self._task_name = fl_ctx.get_prop(FLContextKey.TASK_NAME)
            register_framework_decomposers(context.params_exchange_format, context.server_expected_format)
            metadata = prepare_task_metadata(context, fl_ctx.get_identity_name(), fl_ctx.get_job_id(), self._task_name)

            def log_handler(message):
                message["tag"] = message.pop("key")
                if self._on_log is not None:
                    self._on_log(create_analytic_dxo(**message))

            self._adapter = TaskInProcessAdapter(self._on_result, log_handler)
            self._api = InProcessClientAPI(metadata, adapter=self._adapter)
            self._api.init()
            self._api.configure_memory_management(context.memory_gc_rounds, context.cuda_empty_cache)
            self._runner = TaskScriptRunner(
                custom_dir=fl_ctx.get_workspace().get_app_custom_dir(fl_ctx.get_job_id()),
                script_path=context.task_script_path,
                script_args=context.task_script_args,
            )
        except BaseException:
            try:
                self.finalize(fl_ctx)
            except BaseException:
                logging.getLogger(__name__).exception("Task Client API cleanup failed during initialization")
            raise

    def execute(self, task_name, shareable, fl_ctx, abort_signal):
        if self._executed or task_name != self._task_name or self._api is None:
            raise RuntimeError("task Client API backend accepts only its one assigned task")
        if abort_signal.triggered:
            raise RuntimeError("task Client API assignment was aborted")
        self._executed = True
        shareable.set_header(FLMetaKey.JOB_ID, fl_ctx.get_job_id())
        shareable.set_header(FLMetaKey.SITE_NAME, fl_ctx.get_identity_name())
        self._adapter.stage_input(shareable)
        DataBus().put_data(CLIENT_API_KEY, self._api)
        if self._on_started is not None:
            self._on_started()
        try:
            self._runner.run()
        except SystemExit as e:
            if e.code not in (None, 0):
                raise
        return self._adapter.get_result()

    def abort(self, fl_ctx):
        if self._api is not None:
            self._api.stop = True

    def finalize(self, fl_ctx):
        api, self._api = self._api, None
        runner, self._runner = self._runner, None
        first_error = None
        try:
            if runner is not None:
                runner.release_runtime()
        except BaseException as error:
            first_error = error
        try:
            if api is not None:
                api.close()
        except BaseException as error:
            if first_error is None:
                first_error = error
            else:
                logging.getLogger(__name__).exception("Task Client API cleanup also failed")
        finally:
            bus = DataBus()
            if api is not None and bus.get_data(CLIENT_API_KEY) is api:
                bus.put_data(CLIENT_API_KEY, None)
        if first_error is not None:
            raise first_error
