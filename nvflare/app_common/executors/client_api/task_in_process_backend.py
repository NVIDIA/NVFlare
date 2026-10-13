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
    failure_is_fatal = True

    def __init__(self, *, on_started=None, on_result=None):
        self._on_started, self._on_result = on_started, on_result
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
                context.executor.fire_log_analytics(fl_ctx, create_analytic_dxo(**message))

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
            self.finalize(fl_ctx)
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
        if runner is not None:
            runner.release_runtime()
        if api is not None:
            api.close()
            bus = DataBus()
            if bus.get_data(CLIENT_API_KEY) is api:
                bus.put_data(CLIENT_API_KEY, None)
