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

"""Trusted worker wiring for the shared Client API, progress and analytics."""

import copy
import json
import logging
import math
import os
import time
from functools import partial

from nvflare.apis.analytix import ANALYTIC_EVENT_TYPE
from nvflare.apis.fl_constant import FLContextKey, FLMetaKey
from nvflare.apis.utils.analytix_utils import create_analytic_dxo
from nvflare.app_common.executors.client_api_executor import ClientAPIExecutor, ExecutionMode
from nvflare.app_common.executors.task_script_runner import TaskScriptRunner
from nvflare.client.api_spec import CLIENT_API_KEY
from nvflare.client.decomposers import register_framework_decomposers
from nvflare.client.in_process.adapters import TaskInProcessAdapter
from nvflare.client.in_process.api import InProcessClientAPI
from nvflare.fuel.data_event.data_bus import DataBus
from nvflare.private.fed.task_worker.artifacts import require_eager_shareable
from nvflare.private.fed.task_worker.protocol import _atomic_publish, _open_regular

CLIENT_TASK_CONTEXT_KEYS = {FLContextKey.CLIENT_NAME, FLContextKey.PROCESS_TYPE}


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


def result_wait_deadline(store, identity, timeout, finalization_timeout=30.0):
    """Bound the complete result wait, including script exit and finalizers."""
    started = _read_progress(store, identity, "result_wait_started.json")
    sent = _read_progress(store, identity, "script_result_sent.json")
    deadlines = []
    if started is not None and timeout is not None:
        deadlines.append(started + timeout)
    if sent is not None:
        deadlines.append(sent + finalization_timeout)
    return min(deadlines) if deadlines else None


def _mark_result(store, identity, shareable):
    require_eager_shareable(shareable)
    _mark_progress(store, identity, "script_result_sent.json")


class _ClientAPITaskRunner:
    """Run one Client API assignment directly on the worker's main thread."""

    def __init__(self, executor, *, on_started=None, on_result=None, on_log=None):
        self._context = executor._build_backend_context()
        self._on_started, self._on_result, self._on_log = on_started, on_result, on_log
        self._api = None
        self._runner = None
        self._executed = False

    def initialize(self, fl_ctx):
        context = self._context
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
            metadata = context.build_task_metadata(fl_ctx.get_identity_name(), fl_ctx.get_job_id(), self._task_name)

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
            raise RuntimeError("task Client API runner accepts only its one assigned task")
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


def bind_client_task_context(fl_ctx, identity, store, analytics):
    fl_ctx.set_prop(FLContextKey.CLIENT_NAME, identity.site_name, private=True, sticky=True)
    fl_ctx.set_prop(FLContextKey.PROCESS_TYPE, "client_task_worker", private=True, sticky=True)

    def capture_analytics(dxo):
        with fl_ctx.get_engine().operation():
            analytics.append(
                {"event_type": ANALYTIC_EVENT_TYPE, "data": copy.deepcopy(dxo.to_shareable()), "federated": False}
            )

    def adapt_executor(executor):
        if type(executor) is not ClientAPIExecutor:
            return executor
        return _ClientAPITaskRunner(
            executor,
            on_started=partial(_mark_progress, store, identity, "result_wait_started.json"),
            on_result=partial(_mark_result, store, identity),
            on_log=capture_analytics,
        )

    return adapt_executor
