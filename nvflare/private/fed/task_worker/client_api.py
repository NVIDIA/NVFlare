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

"""One-assignment Client API binding for a disposable script worker."""

from nvflare.apis.fl_constant import FLMetaKey
from nvflare.apis.shareable import Shareable
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.executors.client_api.backend_spec import ClientAPIBackendSpec
from nvflare.app_common.executors.client_api_executor import ClientAPIExecutor, ExecutionMode
from nvflare.app_common.executors.task_script_runner import TaskScriptRunner
from nvflare.client.api_spec import CLIENT_API_KEY
from nvflare.client.config import ConfigKey
from nvflare.client.decomposers import register_framework_decomposers
from nvflare.client.in_process.api import InProcessClientAPI
from nvflare.fuel.data_event.data_bus import DataBus

from .artifacts import FileTaskArtifactStore
from .protocol import TaskAttemptIdentity


class TaskClientAPI(InProcessClientAPI):
    """Reuse Client API model semantics while binding receive/send to local artifacts.

    The application runs on the worker's main thread. send() synchronously writes
    its durable result; the script's next is_running() returns False. Completion
    is committed separately after the script and compute component finalizers
    finish. No federation connection or job-long trainer thread is created.
    """

    def __init__(self, metadata: dict, store: FileTaskArtifactStore, identity: TaskAttemptIdentity, analytics=None):
        super().__init__(task_metadata=metadata)
        self._store = store
        self._identity = identity
        self._result_reference = None
        self._current_round = None
        self.analytics = [] if analytics is None else analytics

    def _subscribe_to_data_bus(self):
        # The worker owns one assignment; there are no asynchronous bus inputs.
        pass

    def stage_input(self, data: Shareable):
        self._current_round = data.get_header(AppConstants.CURRENT_ROUND)
        self._set_received_shareable(data)

    def _publish_result(self, shareable: Shareable):
        if self._result_reference is not None:
            raise RuntimeError("a task worker may send exactly one result")
        if self._current_round is not None:
            shareable.set_header(AppConstants.CURRENT_ROUND, self._current_round)
        self._result_reference = self._store.stage_script_result(self._identity, shareable)
        self.stop = True
        self.stop_reason = "task result handed off"

    def _publish_log(self, message: dict):
        self.analytics.append(message)

    def is_running(self) -> bool:
        return self._result_reference is None and super().is_running()

    def receive(self, timeout=None):
        if self._result_reference is not None:
            return None
        return super().receive(timeout)

    def get_result(self) -> Shareable:
        if self._result_reference is None:
            raise RuntimeError("Client API task script exited without sending a result")
        return self._store.read_script_result(self._identity, self._result_reference)


class TaskClientAPIBackend(ClientAPIBackendSpec):
    """Runtime-injected backend using the ordinary Executor lifecycle.

    Script exceptions fail the disposable attempt, including exceptions after
    send(). The generic worker commits a final Shareable only after task hooks
    and END_RUN finish; it does not dispatch on an Executor's concrete type.
    """

    failure_is_fatal = True

    def __init__(self, store: FileTaskArtifactStore, identity: TaskAttemptIdentity, analytics: list):
        self._store = store
        self._identity = identity
        self._analytics = analytics
        self._api = None
        self._runner = None
        self._executed = False

    def initialize(self, context, fl_ctx):
        executor = context.executor
        if type(executor) is not ClientAPIExecutor:
            raise RuntimeError("task lifetime does not yet support ClientAPIExecutor subclasses")
        if executor.execution_mode != ExecutionMode.IN_PROCESS:
            raise RuntimeError("the Process Client API task backend currently supports in_process scripts only")
        register_framework_decomposers(context.params_exchange_format, context.server_expected_format, executor.logger)
        metadata = {
            FLMetaKey.SITE_NAME: self._identity.site_name,
            FLMetaKey.JOB_ID: self._identity.job_id,
            ConfigKey.TASK_NAME: self._identity.task_name,
            ConfigKey.TASK_EXCHANGE: {
                ConfigKey.TRAIN_WITH_EVAL: context.train_with_evaluation,
                ConfigKey.EXCHANGE_FORMAT: context.params_exchange_format,
                ConfigKey.SERVER_EXPECTED_FORMAT: context.server_expected_format,
                ConfigKey.TRANSFER_TYPE: context.params_transfer_type,
                ConfigKey.TRAIN_TASK_NAME: context.train_task_name,
                ConfigKey.EVAL_TASK_NAME: context.evaluate_task_name,
                ConfigKey.SUBMIT_MODEL_TASK_NAME: context.submit_model_task_name,
            },
        }
        api = TaskClientAPI(metadata, self._store, self._identity, self._analytics)
        try:
            api.init()
            api.configure_memory_management(context.memory_gc_rounds, context.cuda_empty_cache)
            runner = TaskScriptRunner(
                custom_dir=fl_ctx.get_workspace().get_app_custom_dir(self._identity.job_id),
                script_path=context.task_script_path,
                script_args=context.task_script_args,
            )
        except BaseException:
            api.close()
            raise
        self._api = api
        self._runner = runner

    def execute(self, task_name, shareable, fl_ctx, abort_signal) -> Shareable:
        if self._api is None or self._runner is None:
            raise RuntimeError("task Client API backend is not initialized")
        if self._executed or task_name != self._identity.task_name:
            raise RuntimeError("task Client API backend accepts only its single assigned task")
        if abort_signal.triggered:
            raise RuntimeError("task Client API assignment was aborted")
        self._executed = True
        self._api.stage_input(shareable)
        DataBus().put_data(CLIENT_API_KEY, self._api)
        try:
            self._runner.run()
        except SystemExit as e:
            if e.code not in (None, 0):
                raise
        return self._api.get_result()

    def abort(self, fl_ctx):
        if self._api is not None:
            self._api.stop = True

    def finalize(self, fl_ctx):
        api = self._api
        self._api = None
        self._runner = None
        if api is not None:
            try:
                api.close()
            finally:
                bus = DataBus()
                if bus.get_data(CLIENT_API_KEY) is api:
                    bus.put_data(CLIENT_API_KEY, None)
