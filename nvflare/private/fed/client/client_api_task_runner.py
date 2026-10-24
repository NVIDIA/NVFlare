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

"""Run Client API tasks on a worker with progress reporting and analytics capture."""

import json
import logging
import math
import os
import threading
import time

from nvflare.apis.analytix import ANALYTIC_EVENT_TYPE
from nvflare.apis.dxo import DataKind, from_shareable
from nvflare.apis.fl_constant import FLContextKey, FLMetaKey
from nvflare.apis.utils.analytix_utils import create_analytic_dxo
from nvflare.app_common.app_constant import AlgorithmConstants
from nvflare.app_common.executors.client_api_executor import ClientAPIExecutor, ExecutionMode
from nvflare.app_common.executors.task_script_runner import TaskScriptRunner
from nvflare.client.api_spec import CLIENT_API_KEY
from nvflare.client.config import ConfigKey
from nvflare.client.in_process.adapters import TASK_GROUP_STOP_REASON, TASK_RESULT_STOP_REASON, TaskInProcessAdapter
from nvflare.client.in_process.api import InProcessClientAPI
from nvflare.fuel.data_event.data_bus import DataBus
from nvflare.private.fed.task_worker.protocol import _atomic_publish, _open_regular

CLIENT_TASK_CONTEXT_KEYS = {FLContextKey.CLIENT_NAME, FLContextKey.PROCESS_TYPE}
_RESULT_POLL_INTERVAL = 0.05


def _mark_progress(store, identity, name):
    path = os.path.join(store.attempt_dir(identity), name)
    record = {"identity": identity.to_dict(), "monotonic": time.monotonic()}
    _atomic_publish(path, json.dumps(record, sort_keys=True).encode("utf-8"))


def _read_progress(attempt_dir, identity, name):
    try:
        with _open_regular(os.path.join(attempt_dir, name)) as stream:
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
    """Bound application execution, excluding framework-owned artifact I/O."""
    attempt_dir = store.attempt_dir(identity)
    if _read_progress(attempt_dir, identity, "script_finalized.json") is not None:
        return None
    started = _read_progress(attempt_dir, identity, "result_wait_started.json")
    sent = _read_progress(attempt_dir, identity, "script_result_sent.json")
    staging = _read_progress(attempt_dir, identity, "result_staging_started.json")
    if staging is not None and sent is None:
        return None
    staging_seconds = sent - staging if staging is not None and sent is not None else 0
    deadlines = []
    if started is not None and timeout is not None:
        deadlines.append(started + timeout + staging_seconds)
    if sent is not None:
        deadlines.append(sent + finalization_timeout)
    return min(deadlines) if deadlines else None


class _ClientAPITaskRunner:
    """Run Client API assignments on a worker: one on the main thread, or a session's worth on a script thread."""

    def __init__(
        self,
        executor,
        *,
        session=False,
        on_started=None,
        on_result=None,
        on_log=None,
        on_finished=None,
        checkpoint_paths=None,
    ):
        self._context = executor._build_backend_context()
        self._session = session
        self._on_started, self._on_result, self._on_log = on_started, on_result, on_log
        self._checkpoint_paths = checkpoint_paths
        self._on_finished = on_finished
        self._api = None
        self._runner = None
        self._executed = False
        self._thread = None
        self._script_error = None

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
            task_name = fl_ctx.get_prop(FLContextKey.TASK_NAME)
            metadata = context.build_task_metadata(fl_ctx.get_identity_name(), fl_ctx.get_job_id(), task_name)

            def log_handler(message):
                message["tag"] = message.pop("key")
                if self._on_log is not None:
                    self._on_log(create_analytic_dxo(**message))

            self._adapter = TaskInProcessAdapter(self._publish_result, log_handler)
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
        if self._api is None or (self._executed and not self._session):
            raise RuntimeError("task Client API runner accepts only its one assigned task")
        if abort_signal.triggered:
            raise RuntimeError("task Client API assignment was aborted")
        if task_name == self._context.submit_model_task_name:
            raise ValueError("task lifetime does not support local-model submission; use job lifetime")
        self._executed = True
        self._api.meta[ConfigKey.TASK_NAME] = task_name
        if self._checkpoint_paths is not None:
            self._api.system_info()["task_checkpoint"] = dict(self._checkpoint_paths())
        shareable.set_header(FLMetaKey.JOB_ID, fl_ctx.get_job_id())
        shareable.set_header(FLMetaKey.SITE_NAME, fl_ctx.get_identity_name())
        dxo = from_shareable(shareable)
        if AlgorithmConstants.SCAFFOLD_CTRL_GLOBAL in dxo.meta or (
            dxo.data_kind == DataKind.COLLECTION and AlgorithmConstants.SCAFFOLD_CTRL_GLOBAL in dxo.data
        ):
            raise ValueError("task lifetime does not restore SCAFFOLD client control variates; use job lifetime")
        del dxo
        self._adapter.stage_input(shareable)
        # The API owns the decoded model now. Worker stack frames retain this
        # Shareable until the script exits, so release its payload references.
        shareable.clear()
        DataBus().put_data(CLIENT_API_KEY, self._api)
        if self._on_started is not None:
            self._on_started()
        if not self._session:
            self._run_script()
            if self._script_error is not None:
                raise self._script_error
            return self._adapter.get_result()
        if self._thread is None:
            self._thread = threading.Thread(target=self._run_script, name="client-api-script", daemon=True)
            self._thread.start()
        while not self._adapter.wait_result(_RESULT_POLL_INTERVAL):
            if abort_signal.triggered:
                raise RuntimeError("task Client API assignment was aborted")
            if not self._thread.is_alive():
                if self._script_error is not None:
                    raise self._script_error
                raise RuntimeError("Client API task script exited without sending a result")
        return self._adapter.get_result()

    def _publish_result(self, shareable):
        reference = self._on_result(shareable) if self._on_result is not None else None
        if not self._session:
            # A one-shot script has nothing more to receive; let its loop exit.
            self._api.stop = True
            self._api.stop_reason = TASK_RESULT_STOP_REASON
        return reference

    def _run_script(self):
        try:
            self._runner.run()
        except SystemExit as e:
            if e.code not in (None, 0):
                self._script_error = e
        except BaseException as e:
            self._script_error = e

    def finalize(self, fl_ctx):
        api, self._api = self._api, None
        runner, self._runner = self._runner, None
        thread, self._thread = self._thread, None
        first_error = None
        if thread is not None and api is not None:
            api.stop = True
            api.stop_reason = TASK_GROUP_STOP_REASON
            thread.join()
            if self._script_error is not None:
                first_error = self._script_error
        try:
            if runner is not None:
                runner.release_runtime()
        except BaseException as error:
            if first_error is None:
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
        on_finished, self._on_finished = self._on_finished, None
        if on_finished is not None:
            on_finished()


def bind_client_task_context(fl_ctx, binding, store, session):
    """Bind the Client API runner to the worker's current attempt; ``binding`` moves between attempts."""
    fl_ctx.set_prop(FLContextKey.CLIENT_NAME, binding.identity.site_name, private=True, sticky=True)
    fl_ctx.set_prop(FLContextKey.PROCESS_TYPE, "client_task_worker", private=True, sticky=True)

    def capture_analytics(dxo):
        with fl_ctx.get_engine().operation():
            if binding.analytics is None:
                logging.getLogger(__name__).warning("dropping metric logged between task group attempts")
                return
            store.write_analytics_record(
                binding.analytics,
                {"event_type": ANALYTIC_EVENT_TYPE, "data": dxo.to_shareable(), "federated": False},
            )

    def save_result(shareable):
        with fl_ctx.get_engine().operation():
            _mark_progress(store, binding.identity, "result_staging_started.json")
            reference = store.stage_result(binding.identity, shareable)
            _mark_progress(store, binding.identity, "script_result_sent.json")
            return reference

    def adapt_executor(executor):
        if type(executor) is not ClientAPIExecutor:
            return executor
        return _ClientAPITaskRunner(
            executor,
            session=session,
            on_started=lambda: _mark_progress(store, binding.identity, "result_wait_started.json"),
            on_result=save_result,
            on_log=capture_analytics,
            on_finished=lambda: _mark_progress(store, binding.identity, "script_finalized.json"),
            checkpoint_paths=lambda: store.checkpoint_paths(binding.identity),
        )

    return adapt_executor
