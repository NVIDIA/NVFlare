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

"""Client-specific one-assignment script binding for a Process task worker."""

import copy
import json
import math
import os
import time
from functools import partial

from nvflare.apis.analytix import ANALYTIC_EVENT_TYPE
from nvflare.apis.dxo import DataKind, from_shareable
from nvflare.apis.fl_constant import EventScope, FLContextKey
from nvflare.apis.shareable import Shareable
from nvflare.apis.utils.analytix_utils import create_analytic_dxo
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.executors.client_api.backend_spec import CLIENT_API_BACKEND_FACTORY, ClientAPIBackendSpec
from nvflare.app_common.executors.client_api.script_utils import prepare_task_metadata
from nvflare.app_common.executors.client_api_executor import FED_ANALYTIC_EVENT_TYPE, ClientAPIExecutor, ExecutionMode
from nvflare.app_common.executors.task_script_runner import TaskScriptRunner
from nvflare.client.api_spec import CLIENT_API_KEY
from nvflare.client.decomposers import register_framework_decomposers
from nvflare.client.in_process.api import InProcessClientAPI
from nvflare.fuel.data_event.data_bus import DataBus
from nvflare.private.fed.task_worker.artifacts import require_eager_shareable
from nvflare.private.fed.task_worker.protocol import _atomic_publish, _open_regular

CLIENT_TASK_CONTEXT_KEYS = {CLIENT_API_BACKEND_FACTORY, FLContextKey.CLIENT_NAME, FLContextKey.PROCESS_TYPE}


def _mark_progress(store, identity, name):
    path = os.path.join(store.attempt_dir(identity), name)
    record = {"identity": identity.to_dict(), "monotonic": time.monotonic()}
    _atomic_publish(path, json.dumps(record, sort_keys=True).encode("utf-8"))


def _read_progress(store, identity, name):
    try:
        with _open_regular(os.path.join(store.attempt_dir(identity), name)) as stream:
            data = stream.read(65537)
    except FileNotFoundError:
        return None
    if len(data) > 65536:
        raise ValueError("Client API progress marker is too large")
    record = json.loads(data)
    if set(record) != {"identity", "monotonic"} or record["identity"] != identity.to_dict():
        raise ValueError("Client API progress marker has an invalid identity")
    value = record["monotonic"]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError("Client API progress marker has an invalid timestamp")
    return value


def result_wait_deadline(store, identity, timeout):
    """Read control progress only; completion/result reads require settlement."""
    if timeout is None:
        return None
    started = _read_progress(store, identity, "result_wait_started.json")
    if started is None:
        return None
    sent = _read_progress(store, identity, "script_result_sent.json")
    deadline = started + timeout
    return None if sent is not None and sent <= deadline else deadline


def bind_client_task_context(fl_ctx, identity, store, analytics):
    fl_ctx.set_prop(FLContextKey.CLIENT_NAME, identity.site_name, private=True, sticky=True)
    fl_ctx.set_prop(FLContextKey.PROCESS_TYPE, "client_task_worker", private=True, sticky=True)
    fl_ctx.set_prop(
        CLIENT_API_BACKEND_FACTORY, partial(TaskClientAPIBackend, store, identity, analytics), private=True, sticky=True
    )
    fl_ctx.get_engine().add_event_observer(partial(_capture_analytics, analytics))


def _capture_analytics(records, event_type, fl_ctx):
    if event_type not in (ANALYTIC_EVENT_TYPE, FED_ANALYTIC_EVENT_TYPE):
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
    """One receive/send using ordinary model conversion and rank semantics.

    send() hands off an eager snapshot locally. The worker's normal compute
    pipeline persists the final result after the script and finalizers succeed.
    Sending does not bypass those hooks or claim a server receipt.
    """

    def __init__(self, metadata, store, identity, log_handler):
        super().__init__(task_metadata=metadata)
        self._store = store
        self._identity = identity
        self._log_handler = log_handler
        self._result = None
        self._current_round = None

    def _subscribe_to_data_bus(self):
        # No asynchronous input: the selected worker owns exactly one task.
        pass

    def stage_input(self, data):
        self._current_round = data.get_header(AppConstants.CURRENT_ROUND)
        self._set_received_shareable(data)

    def _publish_result(self, shareable):
        if self._result is not None:
            raise RuntimeError("a task worker may send exactly one result")
        require_eager_shareable(shareable)
        self._result = copy.deepcopy(shareable)
        if self._current_round is not None:
            self._result.set_header(AppConstants.CURRENT_ROUND, self._current_round)
        _mark_progress(self._store, self._identity, "script_result_sent.json")
        self.stop = True
        self.stop_reason = "task result handed off"

    def _publish_log(self, message):
        message = copy.deepcopy(message)
        message["tag"] = message.pop("key")
        self._log_handler(create_analytic_dxo(**message))

    def is_running(self):
        if self._result is not None:
            return False
        return super().is_running()

    def get_result(self):
        if self._result is None:
            raise RuntimeError("Client API task script exited without sending a result")
        return self._result


class TaskClientAPIBackend(ClientAPIBackendSpec):
    failure_is_fatal = True

    def __init__(self, store, identity, analytics):
        self._store, self._identity, self._analytics = store, identity, analytics
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
            register_framework_decomposers(context.params_exchange_format, context.server_expected_format)
            metadata = prepare_task_metadata(
                context, self._identity.site_name, self._identity.job_id, self._identity.task_name
            )
            self._api = TaskClientAPI(
                metadata, self._store, self._identity, partial(context.executor.fire_log_analytics, fl_ctx)
            )
            self._api.init()
            self._api.configure_memory_management(context.memory_gc_rounds, context.cuda_empty_cache)
            self._runner = TaskScriptRunner(
                custom_dir=fl_ctx.get_workspace().get_app_custom_dir(self._identity.job_id),
                script_path=context.task_script_path,
                script_args=context.task_script_args,
            )
        except BaseException:
            self.finalize(fl_ctx)
            raise

    def execute(self, task_name, shareable, fl_ctx, abort_signal):
        if self._executed or task_name != self._identity.task_name or self._api is None:
            raise RuntimeError("task Client API backend accepts only its one assigned task")
        if abort_signal.triggered:
            raise RuntimeError("task Client API assignment was aborted")
        self._executed = True
        self._api.stage_input(shareable)
        DataBus().put_data(CLIENT_API_KEY, self._api)
        _mark_progress(self._store, self._identity, "result_wait_started.json")
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
        api, self._api = self._api, None
        runner, self._runner = self._runner, None
        if runner is not None:
            runner.release_runtime()
        if api is not None:
            api.close()
            bus = DataBus()
            if bus.get_data(CLIENT_API_KEY) is api:
                bus.put_data(CLIENT_API_KEY, None)
