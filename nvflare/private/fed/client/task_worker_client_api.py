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

"""Client-specific context and Client API binding for a disposable task worker."""

import copy
from functools import partial

from nvflare.apis.analytix import ANALYTIC_EVENT_TYPE
from nvflare.apis.dxo import DataKind, from_shareable
from nvflare.apis.fl_constant import EventScope, FLContextKey, FLMetaKey
from nvflare.apis.shareable import Shareable
from nvflare.apis.task_state import get_task_state
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.executors.client_api.backend_spec import CLIENT_API_BACKEND_FACTORY, ClientAPIBackendSpec
from nvflare.app_common.executors.client_api.script_utils import (
    close_script_api,
    create_script_binding,
    prepare_task_metadata,
)
from nvflare.app_common.executors.client_api_executor import ClientAPIExecutor, ExecutionMode
from nvflare.client.api_spec import CLIENT_API_KEY
from nvflare.client.in_process.api import InProcessClientAPI
from nvflare.fuel.data_event.data_bus import DataBus
from nvflare.private.fed.task_worker.artifacts import FileTaskArtifactStore
from nvflare.private.fed.task_worker.protocol import TaskAttemptIdentity

CLIENT_TASK_CONTEXT_KEYS = {CLIENT_API_BACKEND_FACTORY, FLContextKey.CLIENT_NAME, FLContextKey.PROCESS_TYPE}


def bind_client_task_context(fl_ctx, identity, store, analytics):
    """Bind client services outside the role-neutral compute runtime."""
    fl_ctx.set_prop(FLContextKey.CLIENT_NAME, identity.site_name, private=True, sticky=True)
    fl_ctx.set_prop(FLContextKey.PROCESS_TYPE, "client_task_worker", private=True, sticky=True)
    fl_ctx.set_prop(
        CLIENT_API_BACKEND_FACTORY,
        partial(TaskClientAPIBackend, store, identity, analytics),
        private=True,
        sticky=True,
    )
    fl_ctx.get_engine().add_event_observer(partial(_capture_analytics, analytics))


def _capture_analytics(records, event_type, fl_ctx):
    # Only the standard analytics channel crosses this boundary. An application
    # event carrying a DXO must not become an arbitrary CJ lifecycle event.
    if event_type != ANALYTIC_EVENT_TYPE:
        return False
    data = fl_ctx.get_prop(FLContextKey.EVENT_DATA)
    if not isinstance(data, Shareable):
        return False
    try:
        dxo = from_shareable(data)
    except (ValueError, TypeError):
        return False
    if dxo.data_kind != DataKind.ANALYTIC:
        return False
    records.append(
        {
            "event_type": event_type,
            "data": copy.deepcopy(data),
            "federated": fl_ctx.get_prop(FLContextKey.EVENT_SCOPE) == EventScope.FEDERATION,
        }
    )
    return True


class TaskClientAPI(InProcessClientAPI):
    """Reuse Client API model semantics while binding receive/send to local artifacts.

    The application runs on the worker's main thread. send() synchronously writes
    its durable result; the script's next is_running() returns False. Completion
    is committed separately after the script and compute component finalizers
    finish. No federation connection or job-long trainer thread is created.
    """

    def __init__(
        self, metadata: dict, store: FileTaskArtifactStore, identity: TaskAttemptIdentity, analytics=None, state=None
    ):
        super().__init__(task_metadata=metadata, task_state=state)
        self._store = store
        self._identity = identity
        self._result_reference = None
        self._current_round = None
        self._result_wait_started = False
        self.analytics = [] if analytics is None else analytics

    def _subscribe_to_data_bus(self):
        # The worker owns one assignment; there are no asynchronous bus inputs.
        pass

    def stage_input(self, data: Shareable):
        self._current_round = data.get_header(AppConstants.CURRENT_ROUND)
        data.set_header(FLMetaKey.JOB_ID, self._identity.job_id)
        data.set_header(FLMetaKey.SITE_NAME, self._identity.site_name)
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

    def receive(self, timeout=None):
        if self._result_reference is not None:
            return None
        if not self._result_wait_started:
            self._store.mark_result_wait_started(self._identity)
            self._result_wait_started = True
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
        self._api, self._runner = create_script_binding(
            context,
            prepare_task_metadata(context, self._identity.site_name, self._identity.job_id, self._identity.task_name),
            fl_ctx.get_workspace().get_app_custom_dir(self._identity.job_id),
            lambda metadata: TaskClientAPI(
                metadata, self._store, self._identity, self._analytics, state=get_task_state(fl_ctx)
            ),
            executor.logger,
        )

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
        close_script_api(api, DataBus())
