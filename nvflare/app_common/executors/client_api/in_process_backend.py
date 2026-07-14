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

"""in_process backend for ClientAPIExecutor.

Ports InProcessClientAPIExecutor's DataBus machinery behind the frozen
ClientAPIBackendSpec surface, with the behavior-parity bar "nothing user-visible":
the trainer script still runs on a thread inside the CJ process, finds its
InProcessClientAPI via the DataBus CLIENT_API_KEY entry, receives tasks over
TOPIC_GLOBAL_RESULT, and returns results over TOPIC_LOCAL_RESULT.

Differences from the legacy executor (see docs/design/client_api_execution_modes.md):

- No ParamsConverters and no exchange-format/transfer-type knobs:
  the Client API boundary passes params through unconverted (ExchangeFormat.RAW)
  and V1 sends full params (TransferType.FULL); DIFF support returns with the
  model_registry transfer-type decision, and format conversion moves to
  send/receive filters at the client edge.
- LOG data is converted to analytics events through the executor-owned
  fire_log_analytics() (single analytics-event ownership point), not a direct
  send_analytic_dxo call.
- initialize() self-unwinds on failure and finalize() is idempotent and
  unsubscribes this backend's DataBus callbacks: the DataBus is a process
  singleton, so leaked subscriptions would survive into later jobs run in the
  same process (e.g. the simulator).
"""

import threading
import time
from typing import Optional

from nvflare.apis.fl_constant import FLContextKey, FLMetaKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.fl_exception import UnsafeJobError
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.apis.utils.analytix_utils import create_analytic_dxo
from nvflare.apis.workspace import Workspace
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.executors.client_api.backend_spec import ClientAPIBackendContext, ClientAPIBackendSpec
from nvflare.app_common.executors.client_api.single_backend import SingleBackendGuard
from nvflare.app_common.executors.task_script_runner import TaskScriptRunner
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
from nvflare.fuel.utils.log_utils import get_obj_logger
from nvflare.security.logging import secure_format_traceback

# Poll cadence for the result-wait loop and the trainer-side receive() checks.
# The legacy executor exposed this as result_pull_interval (default 0.5); the frozen
# surface deliberately drops the knob, so the default becomes the behavior.
_RESULT_POLL_INTERVAL = 0.5

# Bound for finalize()'s trainer-thread join. TOPIC_STOP is only observed at the trainer's
# next flare call; a trainer stuck in user code (a long GPU op, a loop that never checks
# flare.is_running()) may never observe it. With result_wait_timeout, execute() can return
# while the trainer is still alive, so an unbounded join here could hang CJ/simulator
# teardown forever.
_TRAINER_STOP_JOIN_TIMEOUT = 30.0
_NO_RESULT = object()

# One live in_process backend per DataBus (V1): the DataBus is a process singleton with a
# single well-known CLIENT_API_KEY and fixed topics, so two in_process ClientAPIExecutors
# in one job would silently overwrite each other. Shared mechanism with the external_process
# backend (keyed on the CJ Cell there); see single_backend.py.
_GUARD = SingleBackendGuard(
    mode="in_process",
    remedy="configure a single ClientAPIExecutor for all of its tasks (they share the process "
    "DataBus CLIENT_API_KEY and fixed topics)",
    # the DataBus is a process singleton, so the enforced scope is the process, not the job
    scope="process",
)


class InProcessBackend(ClientAPIBackendSpec):
    """Runs the trainer script on a thread in the CJ process, bridged over DataBus."""

    def __init__(self):
        super().__init__()
        # the spec is a plain ABC: fl_ctx-aware logging goes through the executor back-reference
        # (context.executor.log_*); this logger covers callback paths that have no fl_ctx
        self.logger = get_obj_logger(self)
        self._context: Optional[ClientAPIBackendContext] = None
        self._engine = None
        self._data_bus: Optional[DataBus] = None
        self._event_manager: Optional[EventManager] = None
        self._client_api: Optional[InProcessClientAPI] = None
        self._task_fn_thread: Optional[threading.Thread] = None
        self._local_result = _NO_RESULT
        self._abort = False
        self._abort_reason: Optional[str] = None
        self._finalized = False
        self._subscribed = False

    def initialize(self, context: ClientAPIBackendContext, fl_ctx: FLContext) -> None:
        self._context = context

        task_script_path = context.task_script_path
        if not task_script_path or not task_script_path.endswith(".py"):
            raise ValueError(f"invalid task_script_path '{task_script_path}': in_process mode requires a .py script")

        try:
            self._engine = fl_ctx.get_engine()

            self._data_bus = DataBus()
            # Claim the one-backend-per-DataBus slot BEFORE any subscribe/put_data, so a
            # rejected second backend (another in_process ClientAPIExecutor in the same job)
            # leaves the active backend's subscriptions and CLIENT_API_KEY untouched. The
            # DataBus is a process singleton with one well-known key + fixed topics, so two
            # backends would otherwise silently overwrite each other (last-writer-wins);
            # this fails the misconfiguration fast at START_RUN. See SingleBackendGuard.
            _GUARD.claim(self._data_bus, self)

            self._event_manager = EventManager(self._data_bus)
            self._data_bus.subscribe([TOPIC_LOCAL_RESULT], self._local_result_callback)
            self._data_bus.subscribe([TOPIC_LOG_DATA], self._log_result_callback)
            self._data_bus.subscribe([TOPIC_ABORT, TOPIC_STOP], self._to_abort_callback)
            self._subscribed = True

            workspace: Workspace = fl_ctx.get_prop(FLContextKey.WORKSPACE_OBJECT)
            job_id = fl_ctx.get_prop(FLContextKey.CURRENT_JOB_ID)
            custom_dir = workspace.get_app_custom_dir(job_id)
            task_fn_wrapper = TaskScriptRunner(
                custom_dir=custom_dir, script_path=task_script_path, script_args=context.task_script_args
            )

            meta = self._prepare_task_meta(fl_ctx, None)
            self._client_api = InProcessClientAPI(task_metadata=meta, result_check_interval=_RESULT_POLL_INTERVAL)
            self._client_api.init()
            if context.memory_gc_rounds > 0:
                self._client_api.configure_memory_management(
                    gc_rounds=context.memory_gc_rounds, cuda_empty_cache=context.cuda_empty_cache
                )
            # this is how the trainer script's flare.init() finds the API instance
            self._data_bus.put_data(CLIENT_API_KEY, self._client_api)

            # daemon: a deliberate divergence from the legacy executor (non-daemon). With
            # result_wait_timeout, execute() can return while the trainer is still running;
            # a trainer wedged in user code never observes TOPIC_STOP, and a non-daemon
            # thread would then block process exit even after finalize() gives up its
            # bounded join. finalize() still joins cooperatively first.
            self._task_fn_thread = threading.Thread(
                target=task_fn_wrapper.run, name="client_api_in_process_trainer", daemon=True
            )
            self._task_fn_thread.start()
        except BaseException:
            # contract (backend_spec): initialize() self-unwinds its partial setup on failure;
            # the executor does not call finalize() on a half-initialized backend. BaseException,
            # not Exception: a KeyboardInterrupt/SystemExit between the guard claim and here must
            # also release the process-scoped DataBus slot, or it stays blocked for the process.
            self._unwind()
            raise

    def execute(self, task_name: str, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        self._context.executor.log_info(fl_ctx, f"execute for task ({task_name})")

        if abort_signal.triggered:
            self._event_manager.fire_event(TOPIC_ABORT, f"'{task_name}' is aborted, abort_signal_triggered")
            return make_reply(ReturnCode.TASK_ABORTED)

        if not self._trainer_thread_is_alive():
            self._latch_abort("trainer thread exited without signaling ABORT")

        if self._abort:
            # An in-process trainer that aborted (script failure, prior timeout, STOP) is gone
            # for good -- the thread is never relaunched. Fail fast with an accurate return code:
            # TASK_ABORTED here would be misleading (this task was never delivered and the
            # abort_signal never triggered). The legacy executor could not reach this state in a
            # healthy job (it had no result-wait bound), so this is new, documented behavior.
            self._context.executor.log_error(
                fl_ctx,
                f"in-process trainer is no longer available (reason: {self._abort_reason}); "
                f"failing task '{task_name}'",
            )
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)

        # Drop any stale result from a previously timed-out task: a result published
        # concurrently with that task's timeout decision must not satisfy this task's wait.
        # No further correlation is needed: execute() runs one task at a time, and any
        # trainer abort latches _abort above, so at most one such straggler can exist.
        self._local_result = _NO_RESULT

        try:
            # kept from the legacy executor: some task scripts read this ad-hoc prop
            fl_ctx.set_prop("abort_signal", abort_signal)

            meta = self._prepare_task_meta(fl_ctx, task_name)
            self._client_api.set_meta(meta)

            shareable.set_header(FLMetaKey.JOB_ID, fl_ctx.get_job_id())
            shareable.set_header(FLMetaKey.SITE_NAME, fl_ctx.get_identity_name())

            self._context.executor.log_info(fl_ctx, "sending task data to in-process trainer")
            self._event_manager.fire_event(TOPIC_GLOBAL_RESULT, shareable)

            result_wait_timeout = self._context.result_wait_timeout
            # monotonic: a wall-clock step (NTP, VM resume) must not fire a spurious timeout,
            # which would kill the trainer for the rest of the job
            wait_start = time.monotonic()
            wait_deadline = None if result_wait_timeout is None else wait_start + result_wait_timeout
            self._context.executor.log_info(fl_ctx, "waiting for result from in-process trainer")
            while True:
                if abort_signal.triggered or self._abort:
                    # notify the trainer that the task is aborted
                    self._event_manager.fire_event(TOPIC_ABORT, f"'{task_name}' is aborted, abort_signal_triggered")
                    return make_reply(ReturnCode.TASK_ABORTED)

                if self._local_result is not _NO_RESULT:
                    result = self._local_result
                    self._local_result = _NO_RESULT

                    if not isinstance(result, Shareable):
                        self._context.executor.log_error(
                            fl_ctx, f"bad task result from trainer: expect Shareable but got {type(result)}"
                        )
                        return make_reply(ReturnCode.EXECUTION_EXCEPTION)

                    current_round = shareable.get_header(AppConstants.CURRENT_ROUND)
                    if current_round is not None:
                        result.set_header(AppConstants.CURRENT_ROUND, current_round)
                    return result

                if not self._trainer_thread_is_alive():
                    reason = "trainer thread exited before producing a result"
                    self._latch_abort(reason)
                    self._context.executor.log_error(fl_ctx, f"{reason} for task '{task_name}'")
                    return make_reply(ReturnCode.EXECUTION_EXCEPTION)

                now = time.monotonic()
                if wait_deadline is not None and now >= wait_deadline:
                    # the contract forbids waiting unbounded past the configured bound: tell the
                    # trainer to stop this task and fail the round
                    self._event_manager.fire_event(
                        TOPIC_ABORT, f"'{task_name}' timed out after {result_wait_timeout}s waiting for result"
                    )
                    self._context.executor.log_error(
                        fl_ctx, f"timed out after {result_wait_timeout}s waiting for '{task_name}' result"
                    )
                    return make_reply(ReturnCode.EXECUTION_EXCEPTION)

                sleep_time = _RESULT_POLL_INTERVAL
                if wait_deadline is not None:
                    sleep_time = min(sleep_time, wait_deadline - now)
                time.sleep(sleep_time)
        except UnsafeJobError:
            raise
        except Exception:
            self._context.executor.log_error(fl_ctx, secure_format_traceback())
            self._event_manager.fire_event(TOPIC_ABORT, f"'{task_name}' failed: {secure_format_traceback()}")
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)

    def handle_event(self, event_type: str, fl_ctx: FLContext) -> None:
        # no per-event behavior for in_process (START_RUN/END_RUN are handled by the
        # executor via initialize/finalize); contract: must not raise
        pass

    def finalize(self, fl_ctx: FLContext) -> None:
        # contract: idempotent and must not raise
        if self._finalized:
            return
        self._finalized = True
        # separate try blocks: a failure publishing the stop event must not skip the join
        try:
            if self._event_manager is not None:
                self._event_manager.fire_event(TOPIC_STOP, "END_RUN received")
        except Exception:
            self.logger.error(secure_format_traceback())
        try:
            thread = self._task_fn_thread
            if thread is not None and thread.is_alive():
                # bounded: TOPIC_STOP is only seen at the trainer's next flare call, and a
                # trainer stuck in user code may never make one -- do not hang teardown on it
                thread.join(timeout=_TRAINER_STOP_JOIN_TIMEOUT)
                if thread.is_alive():
                    self.logger.error(
                        f"in-process trainer thread did not stop within {_TRAINER_STOP_JOIN_TIMEOUT}s "
                        f"after TOPIC_STOP; abandoning it (daemon thread, will not block process exit; "
                        f"its API is closed, so any later send/log from it is dropped)"
                    )
        except Exception:
            self.logger.error(secure_format_traceback())
        finally:
            self._unwind()

    def _unwind(self) -> None:
        """Releases DataBus state so nothing leaks into later jobs (DataBus is a process singleton).

        Each step is individually best-effort: a failure in one must not skip the others.
        """
        # close the API FIRST: it detaches the API's own subscriptions (the singleton bus would
        # otherwise keep the dead instance subscribed to TOPIC_GLOBAL_RESULT, pinning each later
        # job's global model) and sets the closed gate that stops an abandoned trainer from
        # publishing into a successor job -- the one step that must not be skipped by a failure
        # elsewhere in the teardown
        if self._client_api is not None:
            try:
                self._client_api.close()
            except Exception:
                self.logger.error(secure_format_traceback())
            try:
                if self._data_bus is not None and self._data_bus.get_data(CLIENT_API_KEY) is self._client_api:
                    # only clear our own entry: a later backend may have installed its API
                    self._data_bus.put_data(CLIENT_API_KEY, None)
            except Exception:
                self.logger.error(secure_format_traceback())
        if self._data_bus is not None and self._subscribed:
            for topic, callback in (
                (TOPIC_LOCAL_RESULT, self._local_result_callback),
                (TOPIC_LOG_DATA, self._log_result_callback),
                (TOPIC_ABORT, self._to_abort_callback),
                (TOPIC_STOP, self._to_abort_callback),
            ):
                try:
                    self._data_bus.unsubscribe(topic, callback)
                except Exception:
                    self.logger.error(secure_format_traceback())
            self._subscribed = False
        self._client_api = None
        # release the one-backend-per-DataBus slot LAST, only if this backend holds it: a
        # rejected second backend's unwind must not evict the active backend's claim, and a
        # later sequential job (simulator reuse) must be able to reclaim. Runs on both the
        # finalize path (finalize -> _unwind) and the failed-init path.
        try:
            if self._data_bus is not None:
                _GUARD.release(self._data_bus, self)
        except Exception:
            self.logger.error(secure_format_traceback())

    def _prepare_task_meta(self, fl_ctx: FLContext, task_name: Optional[str]) -> dict:
        context = self._context
        return {
            FLMetaKey.SITE_NAME: fl_ctx.get_identity_name(),
            FLMetaKey.JOB_ID: fl_ctx.get_job_id(),
            ConfigKey.TASK_NAME: task_name,
            ConfigKey.TASK_EXCHANGE: {
                ConfigKey.TRAIN_WITH_EVAL: context.train_with_evaluation,
                # the Client API boundary passes params through unconverted; format conversion
                # happens in send/receive filters at the client edge, and DIFF returns with the
                # model-registry transfer-type decision
                ConfigKey.EXCHANGE_FORMAT: ExchangeFormat.RAW,
                ConfigKey.TRANSFER_TYPE: TransferType.FULL,
                ConfigKey.TRAIN_TASK_NAME: context.train_task_name,
                ConfigKey.EVAL_TASK_NAME: context.evaluate_task_name,
                ConfigKey.SUBMIT_MODEL_TASK_NAME: context.submit_model_task_name,
            },
        }

    def _trainer_thread_is_alive(self) -> bool:
        thread = self._task_fn_thread
        return thread is not None and thread.is_alive()

    def _latch_abort(self, reason: str) -> None:
        self._abort = True
        if self._abort_reason is None:
            self._abort_reason = reason

    def _local_result_callback(self, topic, data, databus):
        if not isinstance(data, Shareable):
            # do not raise into the trainer's send path; record the bad result and let the
            # execute() loop reply EXECUTION_EXCEPTION (a raise here would surface in the
            # trainer thread, not the CJ)
            self.logger.error(f"bad task result from trainer: expect Shareable but got {type(data)}")
        self._local_result = data

    def _log_result_callback(self, topic, data, databus):
        result = data
        if not isinstance(result, dict):
            self.logger.error(f"invalid result format, expecting Dict, but got {type(result)}")
            return

        try:
            if "key" in result:
                result["tag"] = result.pop("key")
            dxo = create_analytic_dxo(**result)

            # single analytics-event ownership point: the executor decides local vs fed fire path
            with self._engine.new_context() as fl_ctx:
                self._context.executor.fire_log_analytics(fl_ctx, dxo)
        except Exception:
            # DataBus callback failures otherwise disappear in the thread-pool Future, dropping the
            # metric without any useful diagnostic.
            self.logger.error(f"failed to process trainer LOG data: {secure_format_traceback()}")

    def _to_abort_callback(self, topic, data, databus):
        # keep the FIRST cause: later echoes (e.g. our own fail-fast fires) must not mask it
        self._latch_abort(f"{topic}: {data}")
