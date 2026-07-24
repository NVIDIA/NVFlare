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

"""External-process backend for ClientAPIExecutor.

The backend owns the launched trainer process/process group and its authenticated session with the
Client Job (CJ) Cell. Task and result Shareables use Cell/F3 directly, including lazy payload transfer. See
``docs/design/client_api_execution_modes.md`` and the trainer counterpart in
``nvflare/client/cell/api.py``.
"""

import os
import secrets
import signal
import subprocess
import threading
import time
import uuid
from typing import Any, Optional, Sequence, Tuple, Union

from nvflare.apis.fl_constant import FLContextKey, FLMetaKey, ReturnCode, ServerCommandNames
from nvflare.apis.fl_context import FLContext
from nvflare.apis.fl_exception import UnsafeJobError
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.apis.utils.analytix_utils import create_analytic_dxo
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.executors.client_api.backend_spec import ClientAPIBackendContext, ClientAPIBackendSpec
from nvflare.client.api_spec import CLIENT_API_TYPE_KEY
from nvflare.client.cell.bootstrap import (
    BOOTSTRAP_FILE_ENV_VAR,
    BOOTSTRAP_SCHEMA_VERSION,
    EXTERNAL_PROCESS_EXECUTION_MODE,
    BootstrapKey,
    write_bootstrap_config,
)
from nvflare.client.cell.decomposers import register_framework_decomposers
from nvflare.client.cell.defs import CHANNEL, PROTOCOL_VERSION, MsgKey, Topic
from nvflare.client.config import ConfigKey
from nvflare.fuel.f3.cellnet.defs import CellChannel, MessageHeaderKey
from nvflare.fuel.f3.cellnet.defs import ReturnCode as CellReturnCode
from nvflare.fuel.f3.cellnet.fqcn import FQCN
from nvflare.fuel.f3.cellnet.utils import make_reply as make_cell_reply
from nvflare.fuel.f3.cellnet.utils import new_cell_message
from nvflare.fuel.f3.streaming.download_service import DownloadService
from nvflare.fuel.f3.streaming.transfer_progress import DEFAULT_STREAMING_IDLE_TIMEOUT
from nvflare.fuel.utils.fobs import FOBSContextKey
from nvflare.fuel.utils.fobs.decomposers.via_downloader import RESULT_UPLOAD_TX_CREATED_CB_CTX_KEY
from nvflare.fuel.utils.log_utils import get_obj_logger
from nvflare.security.logging import secure_format_traceback
from nvflare.utils.job_launcher_utils import add_custom_dir_to_path
from nvflare.utils.process_utils import log_subprocess_output, prepare_subprocess_command

# Poll cadence for process-death detection; events wake successful waits immediately.
_RESULT_POLL_INTERVAL = 0.5
_HELLO_POLL_INTERVAL = 0.1

_DEFAULT_SHUTDOWN_TIMEOUT = 30.0

# The result reaper retries SHUTDOWN when send() is still settling.
_LIVE_RESULT_SHUTDOWN_ACK_TIMEOUT = 5.0

_LOG_THREAD_JOIN_TIMEOUT = 5.0

# A fresh FQCN prevents stale trainer cells from colliding with later launches.
_TRAINER_LEAF_PREFIX = "client_api_trainer"


def bootstrap_file_name(seq: int) -> str:
    """Return a launch-scoped bootstrap name so stale processes retain stale credentials."""
    return f"client_api_bootstrap_{seq}.json"


class _LaunchAborted(Exception):
    """The task's abort_signal triggered while waiting for a per-task trainer launch."""


# Conditions that can interrupt TASK_READY delivery.
_SEND_OK = "ok"
_SEND_ABORTED = "aborted"
_SEND_PROCESS_DEAD = "process_dead"
_SEND_SESSION_DEAD = "session_dead"
_SEND_CLOSED = "closed"

# Accepted lazy results keep their trainer source alive until flare.send() settles.
_NATURAL_EXIT_REAP_INTERVAL = 0.1
_SHUTDOWN_RETRY_INTERVAL = 1.0


class _TaskReadyCancelSignal(Signal):
    """Latches the first condition that must cancel a blocking TASK_READY request."""

    def __init__(self, cancel_cause_cb):
        super().__init__()
        self._cancel_cause_cb = cancel_cause_cb
        self._check_lock = threading.Lock()
        self.error = None

    @property
    def triggered(self):
        if super().triggered:
            return True
        with self._check_lock:
            if not super().triggered:
                try:
                    cause = self._cancel_cause_cb()
                except BaseException as e:
                    self.error = e
                    self.trigger(True)
                else:
                    if cause is not None:
                        self.trigger(cause)
            return super().triggered


class _TrainerSession:
    """One launched trainer process and its (at most one) authenticated protocol session."""

    def __init__(self, token: str, trainer_fqcn: str):
        self.token = token
        self.trainer_fqcn = trainer_fqcn
        self.session_id: Optional[str] = None
        self.ready = threading.Event()
        # latched when the HELLO of THIS launch's own process is rejected, so the launch
        # wait fails fast instead of waiting out launch_timeout
        self.reject_reason: Optional[str] = None
        self.bootstrap_path: Optional[str] = None
        self.process: Optional[subprocess.Popen] = None
        # POSIX process-group id, retained independently of the Popen leader handle so the
        # group can be probed/terminated even after the launcher itself exited
        self.pgid: Optional[int] = None
        self.log_thread: Optional[threading.Thread] = None
        # Conservative CJ-side latch: an accepted result may still be inside the
        # trainer's send() acknowledgement/payload barrier. SHUTDOWN reply truth clears
        # it once the trainer has crossed that barrier.
        self.result_source_live = threading.Event()
        self.reaper_thread: Optional[threading.Thread] = None
        self.shutdown_requested = threading.Event()
        self._shutdown_request_lock = threading.Lock()
        self._next_shutdown_retry = 0.0
        self._stop_lock = threading.Lock()
        self._cleanup_lock = threading.Lock()
        self._cleaned = False
        self._activity_lock = threading.Lock()
        self._last_peer_activity: Optional[float] = None

    def touch_peer_activity(self) -> None:
        with self._activity_lock:
            self._last_peer_activity = time.monotonic()

    def peer_silent_for(self) -> Optional[float]:
        with self._activity_lock:
            last_activity = self._last_peer_activity
        return None if last_activity is None else max(0.0, time.monotonic() - last_activity)


class _TaskContext:
    """Correlation state for the one task execute() is currently running."""

    def __init__(self, task_id: str):
        self.task_id = task_id
        self.result_ready = threading.Event()
        self.result: Optional[Shareable] = None


class ExternalProcessBackend(ClientAPIBackendSpec):
    """Launches and owns the external trainer process/group, bridged over the CJ cell."""

    def __init__(self):
        super().__init__()
        self.logger = get_obj_logger(self)
        self._context: Optional[ClientAPIBackendContext] = None
        self._engine = None
        self._cell = None
        self._cj_fqcn: Optional[str] = None
        self._connect_url: Optional[str] = None
        self._pass_through_routes: tuple[Tuple[str, str], ...] = ()
        self._owned_pass_through_routes: set[Tuple[str, str]] = set()
        self._owned_relay_pass_through_routes: set[Tuple[str, str]] = set()
        self._secure_mode = False
        self._job_id: Optional[str] = None
        self._site_name: Optional[str] = None
        self._app_dir: Optional[str] = None
        self._custom_dir: Optional[str] = None
        self._active_launch: Optional[_TrainerSession] = None
        self._launch_lock = threading.Lock()
        # Completed per-task launches can keep serving lazy results and remain owned through END_RUN.
        self._result_reapers = set()
        self._result_reapers_lock = threading.Lock()
        self._launch_seq = 0
        self._current_task: Optional[_TaskContext] = None
        self._task_lock = threading.Lock()
        self._execute_gate = threading.Lock()
        self._abort = False
        self._abort_reason: Optional[str] = None
        self._finalized = False
        # Cell request callbacks cannot be unregistered, so late messages need an explicit gate.
        self._closed = False

    # ------------------------------------------------------------------ lifecycle

    def initialize(self, context: ClientAPIBackendContext, fl_ctx: FLContext) -> None:
        self._context = context
        if not context.command:
            raise ValueError("external_process mode requires a non-empty command")

        try:
            self._engine = fl_ctx.get_engine()
            if self._engine is None:
                raise RuntimeError("no engine available in fl_ctx")
            cell = self._engine.get_cell()
            if cell is None:
                raise RuntimeError("no Cell available from the engine: external_process mode requires the CJ cell")
            self._cell = cell
            self._cj_fqcn = cell.get_fqcn()

            # SERVER_COMMAND also carries unrelated request types (notably
            # subordinate-client SUBMIT_UPDATE), so never opt in the whole channel.
            task_route = (CellChannel.SERVER_COMMAND, ServerCommandNames.GET_TASK)
            result_route = (CHANNEL, Topic.RESULT_READY)
            self._pass_through_routes = (task_route, result_route)
            self._secure_mode = bool(fl_ctx.get_prop(FLContextKey.SECURE_MODE, False))
            for route in self._pass_through_routes:
                if route not in cell.decode_pass_through_topics:
                    self._owned_pass_through_routes.add(route)
                cell.decode_pass_through_topics.add(route)
                if self._secure_mode:
                    if route not in cell.decode_pass_through_relay_topics:
                        self._owned_relay_pass_through_routes.add(route)
                    cell.decode_pass_through_relay_topics.add(route)

            workspace = fl_ctx.get_prop(FLContextKey.WORKSPACE_OBJECT)
            job_id = fl_ctx.get_prop(FLContextKey.CURRENT_JOB_ID)
            if workspace is None or not job_id:
                raise RuntimeError("workspace/job id not available in fl_ctx")
            self._job_id = job_id
            self._site_name = fl_ctx.get_identity_name()
            self._app_dir = workspace.get_app_dir(job_id)
            self._custom_dir = workspace.get_app_custom_dir(job_id)

            cell.make_internal_listener()
            connect_url = cell.get_internal_listener_url()
            if not connect_url:
                raise RuntimeError("CJ cell has no internal listener url for the trainer to connect to")
            self._connect_url = connect_url

            self._register_protocol_cbs(cell)
            register_framework_decomposers(context.params_exchange_format, context.server_expected_format, self.logger)

            context.executor.set_analytics_fire_fed_event(True)

            if context.launch_once:
                self._launch_trainer(timeout=context.launch_timeout)
        except BaseException:
            self._unwind()
            raise

    def execute(self, task_name: str, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        context = self._context
        executor = context.executor
        executor.log_info(fl_ctx, f"execute for task ({task_name})")

        if self._closed:
            executor.log_error(fl_ctx, f"backend is closed; failing task '{task_name}'")
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)

        # DO_TASK and CCWF can invoke the same executor concurrently; this backend admits one task at a time.
        if not self._execute_gate.acquire(blocking=False):
            if abort_signal.triggered:
                return make_reply(ReturnCode.TASK_ABORTED)
            executor.log_error(
                fl_ctx,
                f"a task is already executing on this external_process backend; rejecting concurrent "
                f"task '{task_name}' (one active task per trainer)",
            )
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)

        try:
            return self._execute_admitted_task(task_name, shareable, fl_ctx, abort_signal)
        finally:
            self._execute_gate.release()

    def _execute_admitted_task(
        self, task_name: str, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal
    ) -> Shareable:
        """Execute a task after admission, translating expected lifecycle failures to task replies."""
        context = self._context
        executor = context.executor

        if abort_signal.triggered:
            if context.launch_once:
                trainer = self._active_launch
                self._send_abort(trainer, f"'{task_name}' is aborted, abort_signal_triggered")
                if trainer is not None:
                    self._latch_abort(f"'{task_name}' aborted at entry, abort_signal_triggered")
            return make_reply(ReturnCode.TASK_ABORTED)

        launch_once = context.launch_once
        trainer = self._active_launch
        try:
            if launch_once:
                if self._abort:
                    executor.log_error(
                        fl_ctx,
                        f"external trainer is no longer available (reason: {self._abort_reason}); "
                        f"failing task '{task_name}'",
                    )
                    return make_reply(ReturnCode.EXECUTION_EXCEPTION)
                if trainer is None or not trainer.ready.is_set():
                    executor.log_error(fl_ctx, f"no established trainer session; failing task '{task_name}'")
                    return make_reply(ReturnCode.EXECUTION_EXCEPTION)
                liveness_error = self._trainer_liveness_error(trainer)
                if liveness_error:
                    self._latch_abort(liveness_error)
                    executor.log_error(fl_ctx, f"{liveness_error}; failing task '{task_name}'")
                    return make_reply(ReturnCode.EXECUTION_EXCEPTION)
            else:
                try:
                    trainer = self._launch_trainer(timeout=context.launch_timeout, abort_signal=abort_signal)
                except _LaunchAborted:
                    executor.log_info(fl_ctx, f"'{task_name}' aborted while launching the trainer")
                    return make_reply(ReturnCode.TASK_ABORTED)
                except Exception:
                    executor.log_error(fl_ctx, f"per-task trainer launch failed: {secure_format_traceback()}")
                    return make_reply(ReturnCode.EXECUTION_EXCEPTION)

            return self._run_task(trainer, task_name, shareable, fl_ctx, abort_signal)
        except UnsafeJobError:
            raise
        except Exception:
            executor.log_error(fl_ctx, secure_format_traceback())
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)
        finally:
            self._finish_task_trainer(trainer, launch_once)

    def _finish_task_trainer(self, trainer: Optional[_TrainerSession], launch_once: bool) -> None:
        """Retire a per-task trainer without letting cleanup failure mask the task result."""
        if launch_once or trainer is None:
            return
        try:
            if trainer.result_source_live.is_set():
                self._reap_trainer_after_result(trainer)
            else:
                self._stop_trainer(trainer, natural_exit_wait=self._shutdown_wait_bound())
        except Exception:
            self.logger.error(secure_format_traceback())

    def finalize(self, fl_ctx: FLContext) -> None:
        if self._finalized:
            return
        self._finalized = True
        # Serialize close with RESULT_READY's acceptance commit.
        with self._task_lock:
            self._closed = True
        # The same gate orders END_RUN against the launch-install-to-Popen window.
        admitted = self._execute_gate.acquire(timeout=self._shutdown_wait_bound())
        try:
            with self._launch_lock:
                trainer = self._active_launch
            if trainer is not None:
                if trainer.result_source_live.is_set():
                    # END_RUN must preserve CJ/F3 until the trainer's send barrier settles.
                    self._request_trainer_shutdown(trainer, wait_timeout=_LIVE_RESULT_SHUTDOWN_ACK_TIMEOUT)
                    self._reap_trainer_after_result(trainer)
                else:
                    self._stop_trainer(trainer, natural_exit_wait=self._shutdown_wait_bound())
            self._wait_for_result_reapers()
        except Exception:
            self.logger.error(secure_format_traceback())
        finally:
            self._disable_task_pass_through()
            if admitted:
                self._execute_gate.release()

    # ------------------------------------------------------------------ trainer management

    def _launch_trainer(self, timeout: Optional[float], abort_signal: Optional[Signal] = None) -> _TrainerSession:
        """Launch a trainer and establish its authenticated session, unwinding on failure."""
        token = secrets.token_urlsafe(32)
        with self._launch_lock:
            if self._closed:
                raise RuntimeError("backend is closed; not launching a trainer")
            self._launch_seq += 1
            seq = self._launch_seq
            trainer_fqcn = FQCN.join([self._cj_fqcn, f"{_TRAINER_LEAF_PREFIX}_{seq}"])
            trainer = _TrainerSession(token, trainer_fqcn)
            self._active_launch = trainer
        try:
            bootstrap_path = os.path.join(self._app_dir, bootstrap_file_name(seq))
            trainer.bootstrap_path = bootstrap_path
            write_bootstrap_config(
                bootstrap_path,
                {
                    BootstrapKey.SCHEMA_VERSION: BOOTSTRAP_SCHEMA_VERSION,
                    BootstrapKey.EXECUTION_MODE: EXTERNAL_PROCESS_EXECUTION_MODE,
                    BootstrapKey.CONNECT_URL: self._connect_url,
                    BootstrapKey.CJ_FQCN: self._cj_fqcn,
                    BootstrapKey.TRAINER_FQCN: trainer_fqcn,
                    BootstrapKey.LAUNCH_TOKEN: token,
                    BootstrapKey.JOB_ID: self._job_id,
                    BootstrapKey.SITE_NAME: self._site_name,
                    BootstrapKey.SECURE_MODE: self._secure_mode,
                    BootstrapKey.TASK_EXCHANGE: self._task_exchange_config(),
                    BootstrapKey.MEMORY_GC_ROUNDS: self._context.memory_gc_rounds,
                    BootstrapKey.CUDA_EMPTY_CACHE: self._context.cuda_empty_cache,
                },
            )

            env = os.environ.copy()
            env[BOOTSTRAP_FILE_ENV_VAR] = bootstrap_path
            env.pop(CLIENT_API_TYPE_KEY, None)
            add_custom_dir_to_path(self._custom_dir, env)

            # finalize() may close the backend after trainer installation but before Popen.
            if self._closed:
                raise RuntimeError("backend closed before trainer launch")

            # Never log the configured command: legacy/hand-written jobs may contain literal
            # credentials rather than site-resolved secret references.
            self.logger.info(f"launching external trainer (launch {seq})")
            process = subprocess.Popen(
                self._split_command(self._context.command),
                shell=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=self._app_dir,
                env=env,
                # own process group so orderly stop can signal the launched trainer group
                start_new_session=(os.name == "posix"),
            )
            with trainer._stop_lock:
                trainer.process = process
                if os.name == "posix":
                    # start_new_session made the child its own group leader (pgid == pid)
                    trainer.pgid = process.pid
                cleaned_during_launch = trainer._cleaned
            if cleaned_during_launch:
                # finalize() may have timed out on the execute gate and cleaned the
                # session while Popen was in flight, before it could see this handle.
                self._terminate_process_tree(trainer, grace=self._context.stop_grace_period)
                raise RuntimeError("backend closed during trainer launch")
            trainer.log_thread = threading.Thread(
                target=log_subprocess_output,
                args=(process, self.logger),
                name=f"client_api_trainer_log_{seq}",
                daemon=True,
            )
            trainer.log_thread.start()

            if self._closed:
                raise RuntimeError("backend closed during trainer launch")

            self._wait_for_hello(trainer, timeout, abort_signal)
            self.logger.info(
                f"trainer session established: launch={seq} fqcn={trainer_fqcn} session_id={trainer.session_id}"
            )
            return trainer
        except Exception:
            self._stop_trainer(trainer, natural_exit_wait=0.0)
            raise

    def _wait_for_hello(
        self, trainer: _TrainerSession, timeout: Optional[float], abort_signal: Optional[Signal] = None
    ) -> None:
        """Wait for an accepted HELLO, bounded by launch, process, abort, and close state."""
        deadline = None if timeout is None else time.monotonic() + timeout
        while not trainer.ready.wait(_HELLO_POLL_INTERVAL):
            if abort_signal is not None and abort_signal.triggered:
                raise _LaunchAborted("task aborted while waiting for the trainer HELLO")
            if self._closed:
                raise RuntimeError("backend closed while waiting for the trainer HELLO")
            if trainer.reject_reason:
                raise RuntimeError(f"trainer HELLO was rejected: {trainer.reject_reason}")
            if not self._process_group_alive(trainer):
                rc = trainer.process.poll() if trainer.process else None
                raise RuntimeError(f"trainer process group exited (rc={rc}) before completing the HELLO handshake")
            if deadline is not None and time.monotonic() >= deadline:
                raise RuntimeError(f"trainer did not complete the HELLO handshake within launch_timeout={timeout}s")

    def _stop_trainer(self, trainer: _TrainerSession, natural_exit_wait: float) -> None:
        """Stop a trainer gracefully, then terminate its process/group; idempotent and non-raising."""
        with trainer._stop_lock:
            if trainer._cleaned:
                return
            natural_exit_wait = max(0.0, natural_exit_wait)
            natural_exit_deadline = time.monotonic() + natural_exit_wait
            try:
                remaining = max(0.0, natural_exit_deadline - time.monotonic())
                self._request_trainer_shutdown(trainer, wait_timeout=remaining)
            except Exception:
                self.logger.error(secure_format_traceback())
            try:
                process = trainer.process
                remaining = max(0.0, natural_exit_deadline - time.monotonic())
                if process is not None and remaining > 0:
                    leader_exited = process.poll() is not None
                    if not leader_exited:
                        try:
                            process.wait(timeout=remaining)
                            leader_exited = True
                        except subprocess.TimeoutExpired:
                            pass
                    remaining = max(0.0, natural_exit_deadline - time.monotonic())
                    if leader_exited and os.name == "posix" and trainer.pgid is not None and remaining > 0:
                        # Launcher exit does not imply its worker group has exited.
                        self._await_group_exit(trainer, remaining)
            except Exception:
                self.logger.error(secure_format_traceback())
            try:
                self._terminate_process_tree(trainer, grace=self._context.stop_grace_period)
            except Exception:
                self.logger.error(secure_format_traceback())
            self._cleanup_trainer(trainer)

    def _request_trainer_shutdown(self, trainer: _TrainerSession, wait_timeout: float) -> None:
        """Send at most one orderly SHUTDOWN without taking ownership of process death."""
        with trainer._shutdown_request_lock:
            if trainer.shutdown_requested.is_set():
                return
            now = time.monotonic()
            if now < trainer._next_shutdown_retry:
                return
            trainer._next_shutdown_retry = now + _SHUTDOWN_RETRY_INTERVAL
            if trainer.session_id is None or not self._process_group_alive(trainer):
                return
            request = new_cell_message({}, {MsgKey.SESSION_ID: trainer.session_id, MsgKey.REASON: "shutdown requested"})
            try:
                if wait_timeout > 0:
                    reply = self._cell.send_request(
                        channel=CHANNEL,
                        topic=Topic.SHUTDOWN,
                        target=trainer.trainer_fqcn,
                        request=request,
                        timeout=wait_timeout,
                        optional=True,
                    )
                    if reply is None or reply.get_header(MessageHeaderKey.RETURN_CODE) != CellReturnCode.OK:
                        rc = None if reply is None else reply.get_header(MessageHeaderKey.RETURN_CODE)
                        self.logger.warning(f"trainer SHUTDOWN was not acknowledged (rc={rc})")
                        return
                    body = reply.payload
                    if isinstance(body, dict):
                        source_live = body.get(MsgKey.RESULT_SOURCE_LIVE)
                        if source_live is True:
                            trainer.result_source_live.set()
                        elif source_live is False:
                            trainer.result_source_live.clear()
                else:
                    send_errors = self._cell.fire_and_forget(
                        channel=CHANNEL,
                        topic=Topic.SHUTDOWN,
                        targets=[trainer.trainer_fqcn],
                        message=request,
                        optional=True,
                    )
                    send_error = send_errors.get(trainer.trainer_fqcn) if isinstance(send_errors, dict) else None
                    if send_error:
                        self.logger.warning(f"trainer SHUTDOWN was not delivered: {send_error}")
                        return
                trainer.shutdown_requested.set()
            except Exception:
                self.logger.error(secure_format_traceback())

    def _reap_trainer_after_result(self, trainer: _TrainerSession) -> None:
        """Reap a successful one-task trainer after it finishes serving its result."""
        with trainer._cleanup_lock:
            if trainer._cleaned or (trainer.reaper_thread is not None and trainer.reaper_thread.is_alive()):
                return
            trainer.reaper_thread = threading.Thread(
                target=self._wait_for_natural_exit_and_cleanup,
                args=(trainer,),
                name=f"client_api_trainer_reaper_{trainer.trainer_fqcn.rsplit('.', 1)[-1]}",
                daemon=True,
            )
            with self._result_reapers_lock:
                self._result_reapers.add(trainer)
                try:
                    # Registration and start are atomic to finalize(), which joins registered threads.
                    trainer.reaper_thread.start()
                except BaseException:
                    self._result_reapers.discard(trainer)
                    trainer.reaper_thread = None
                    raise

    def _wait_for_result_reapers(self) -> None:
        """Wait boundedly for accepted result sends, then force-clean wedged sources."""
        wait_bound = self._result_reaper_wait_bound()
        deadline = time.monotonic() + wait_bound
        while True:
            with self._result_reapers_lock:
                trainers = tuple(self._result_reapers)
            if not trainers:
                return
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            for trainer in trainers:
                reaper = trainer.reaper_thread
                if reaper is not None:
                    reaper.join(timeout=max(0.0, deadline - time.monotonic()))

        with self._result_reapers_lock:
            wedged = tuple(self._result_reapers)
        if wedged:
            self.logger.warning(
                f"timed out waiting {wait_bound}s for {len(wedged)} accepted result source(s); "
                "forcing trainer cleanup"
            )
        for trainer in wedged:
            self._stop_trainer(trainer, natural_exit_wait=0.0)
        for trainer in wedged:
            reaper = trainer.reaper_thread
            if reaper is not None:
                reaper.join(timeout=_LOG_THREAD_JOIN_TIMEOUT)

    def _wait_for_natural_exit_and_cleanup(self, trainer: _TrainerSession) -> None:
        disconnected_since = None
        disconnect_grace = (
            self._context.heartbeat_timeout
            if self._context.heartbeat_timeout > 0
            else self._result_source_disconnect_grace()
        )
        try:
            while self._process_group_alive(trainer):
                now = time.monotonic()
                if self._cell.is_cell_connected(trainer.trainer_fqcn):
                    disconnected_since = None
                elif disconnected_since is None:
                    disconnected_since = now
                elif now - disconnected_since >= disconnect_grace:
                    # Allow one reconnect lease for a transient reconnect.
                    self._stop_trainer(trainer, natural_exit_wait=0.0)
                    return
                if self._closed:
                    if not trainer.result_source_live.is_set():
                        self._stop_trainer(trainer, natural_exit_wait=self._result_source_disconnect_grace())
                        return
                    # SHUTDOWN cannot preempt an accepted result source still inside send().
                    self._request_trainer_shutdown(trainer, wait_timeout=_LIVE_RESULT_SHUTDOWN_ACK_TIMEOUT)
                time.sleep(_NATURAL_EXIT_REAP_INTERVAL)
            self._cleanup_trainer(trainer)
        except BaseException:
            self.logger.error(secure_format_traceback())
        finally:
            if not trainer._cleaned:
                self._stop_trainer(trainer, natural_exit_wait=0.0)
            with self._result_reapers_lock:
                self._result_reapers.discard(trainer)

    def _cleanup_trainer(self, trainer: _TrainerSession) -> None:
        """Release launch-scoped state after the process group is gone. Idempotent."""
        with trainer._cleanup_lock:
            if trainer._cleaned:
                return
            trainer._cleaned = True
        try:
            log_thread = trainer.log_thread
            if log_thread is not None and log_thread.is_alive():
                log_thread.join(timeout=_LOG_THREAD_JOIN_TIMEOUT)
        except Exception:
            self.logger.error(secure_format_traceback())
        trainer.token = ""
        trainer.session_id = None
        try:
            if trainer.bootstrap_path and os.path.exists(trainer.bootstrap_path):
                os.remove(trainer.bootstrap_path)
        except Exception as e:
            self.logger.debug(f"failed to remove {trainer.bootstrap_path}: {e}")
        with self._launch_lock:
            if self._active_launch is trainer:
                self._active_launch = None

    def _disable_task_pass_through(self) -> None:
        cell = self._cell
        if cell is not None:
            for route in self._owned_pass_through_routes:
                cell.decode_pass_through_topics.discard(route)
            for route in self._owned_relay_pass_through_routes:
                cell.decode_pass_through_relay_topics.discard(route)
        self._pass_through_routes = ()
        self._owned_pass_through_routes.clear()
        self._owned_relay_pass_through_routes.clear()

    def _process_group_alive(self, trainer: _TrainerSession) -> bool:
        """Return group liveness even when a launcher exits before its workers."""
        process = trainer.process
        if os.name != "posix" or trainer.pgid is None:
            return process is not None and process.poll() is None
        if process is not None:
            process.poll()
        try:
            os.killpg(trainer.pgid, 0)
            return True
        except ProcessLookupError:
            return False
        except Exception as e:
            # Probe failure must not let teardown abandon an owned group.
            self.logger.debug(f"cannot probe trainer process group {trainer.pgid}: {e}")
            return True

    def _trainer_liveness_error(self, trainer: _TrainerSession) -> Optional[str]:
        """Returns why an established trainer is unavailable, or None while it is live."""
        if not self._process_group_alive(trainer):
            rc = trainer.process.poll() if trainer.process else None
            return f"trainer process group exited (rc={rc})"
        heartbeat_timeout = self._context.heartbeat_timeout
        if heartbeat_timeout > 0 and trainer.ready.is_set():
            silent_for = trainer.peer_silent_for()
            if silent_for is not None and silent_for > heartbeat_timeout:
                return f"trainer heartbeat timed out after {silent_for:.1f}s " f"(timeout={heartbeat_timeout}s)"
        return None

    def _await_group_exit(self, trainer: _TrainerSession, timeout: float) -> bool:
        """Waits (bounded) for the whole process group to exit; reaps the leader."""
        deadline = time.monotonic() + timeout
        process = trainer.process
        while True:
            if process is not None and process.poll() is None:
                try:
                    process.wait(timeout=0.1)
                except subprocess.TimeoutExpired:
                    pass
            if not self._process_group_alive(trainer):
                return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.1)

    def _terminate_process_tree(self, trainer: _TrainerSession, grace: float) -> None:
        """Apply SIGTERM, bounded grace, then SIGKILL to the owned process group."""
        if not self._process_group_alive(trainer):
            return
        self.logger.info(f"terminating trainer process group (pgid={trainer.pgid}, grace={grace}s)")
        self._signal_process_tree(trainer, hard=False)
        if self._await_group_exit(trainer, grace):
            return
        self.logger.warning(f"trainer process group (pgid={trainer.pgid}) survived SIGTERM grace; killing")
        self._signal_process_tree(trainer, hard=True)
        if not self._await_group_exit(trainer, _LOG_THREAD_JOIN_TIMEOUT):
            self.logger.error(f"trainer process group (pgid={trainer.pgid}) did not die after SIGKILL")

    def _signal_process_tree(self, trainer: _TrainerSession, hard: bool) -> None:
        """Soft (SIGTERM/terminate) or hard (SIGKILL/kill) signal to the trainer process/group."""
        if os.name == "posix" and trainer.pgid is not None:
            try:
                os.killpg(trainer.pgid, signal.SIGKILL if hard else signal.SIGTERM)
                return
            except ProcessLookupError:
                return
            except Exception as e:
                self.logger.debug(f"failed to signal trainer process group: {e}")
        process = trainer.process
        if process is None or process.poll() is not None:
            return
        try:
            if hard:
                process.kill()
            else:
                process.terminate()
        except Exception as e:
            self.logger.debug(f"failed to signal trainer process: {e}")

    @staticmethod
    def _split_command(command: Union[str, Sequence[str]]) -> list[str]:
        """Prepare shell-free argv and resolve each secret as one argument."""
        return prepare_subprocess_command(command, posix=os.name == "posix")

    def _shutdown_wait_bound(self) -> float:
        shutdown_timeout = self._context.shutdown_timeout
        return _DEFAULT_SHUTDOWN_TIMEOUT if shutdown_timeout is None else shutdown_timeout

    def _result_source_disconnect_grace(self) -> float:
        """Return a nonzero disconnect grace for an accepted result source."""
        shutdown_bound = self._shutdown_wait_bound()
        return shutdown_bound if shutdown_bound > 0 else _DEFAULT_SHUTDOWN_TIMEOUT

    def _result_reaper_wait_bound(self) -> float:
        """Return the END_RUN backstop after transfer and disconnect budgets."""
        disconnect_grace = (
            self._context.heartbeat_timeout
            if self._context.heartbeat_timeout > 0
            else self._result_source_disconnect_grace()
        )
        return DEFAULT_STREAMING_IDLE_TIMEOUT + disconnect_grace + _LIVE_RESULT_SHUTDOWN_ACK_TIMEOUT

    def _unwind(self) -> None:
        """Releases partial setup after a failed initialize(). Best-effort per step."""
        self._closed = True
        try:
            trainer = self._active_launch
            if trainer is not None:
                self._stop_trainer(trainer, natural_exit_wait=0.0)
        except Exception:
            self.logger.error(secure_format_traceback())
        finally:
            self._disable_task_pass_through()

    # ------------------------------------------------------------------ task execution

    def _run_task(
        self, trainer: _TrainerSession, task_name: str, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal
    ) -> Shareable:
        context = self._context
        executor = context.executor
        launch_once = context.launch_once

        task = _TaskContext(task_id=uuid.uuid4().hex)

        shareable.set_header(FLMetaKey.JOB_ID, fl_ctx.get_job_id())
        shareable.set_header(FLMetaKey.SITE_NAME, fl_ctx.get_identity_name())

        with self._task_lock:
            self._current_task = task
        try:
            task_message = {
                MsgKey.SESSION_ID: trainer.session_id,
                MsgKey.TASK_ID: task.task_id,
                MsgKey.TASK_NAME: task_name,
                MsgKey.MODEL: shareable,
            }
            executor.log_info(fl_ctx, f"sending TASK_READY for '{task_name}' to trainer {trainer.trainer_fqcn}")
            send_status, reply = self._send_task_ready(trainer, task_message, abort_signal)
            if send_status == _SEND_ABORTED:
                self._send_abort(trainer, f"'{task_name}' is aborted, abort_signal_triggered")
                self._latch_abort(f"task '{task_name}' aborted during TASK_READY")
                return make_reply(ReturnCode.TASK_ABORTED)
            if send_status == _SEND_PROCESS_DEAD:
                reason = "trainer process exited while TASK_READY was pending"
                self._latch_abort(reason)
                executor.log_error(fl_ctx, f"{reason} for task '{task_name}'")
                return make_reply(ReturnCode.EXECUTION_EXCEPTION)
            if send_status == _SEND_SESSION_DEAD:
                reason = f"{reply} while TASK_READY was pending"
                self._latch_abort(reason)
                executor.log_error(fl_ctx, f"{reason} for task '{task_name}'")
                return make_reply(ReturnCode.EXECUTION_EXCEPTION)
            if send_status == _SEND_CLOSED:
                executor.log_error(fl_ctx, f"backend closed while TASK_READY was pending for '{task_name}'")
                return make_reply(ReturnCode.EXECUTION_EXCEPTION)
            reject_reason = self._check_task_accepted(reply)
            if reject_reason:
                executor.log_error(fl_ctx, f"trainer did not accept task '{task_name}': {reject_reason}")
                return make_reply(ReturnCode.EXECUTION_EXCEPTION)

            result_wait_timeout = context.result_wait_timeout
            wait_start = time.monotonic()
            wait_deadline = None if result_wait_timeout is None else wait_start + result_wait_timeout
            executor.log_info(fl_ctx, "waiting for result from external trainer")
            while True:
                if abort_signal.triggered or (launch_once and self._abort):
                    self._send_abort(trainer, f"'{task_name}' is aborted, abort_signal_triggered")
                    self._latch_abort(f"task '{task_name}' aborted")
                    return make_reply(ReturnCode.TASK_ABORTED)

                if task.result_ready.is_set():
                    break

                liveness_error = self._trainer_liveness_error(trainer)
                if liveness_error:
                    self._send_abort(trainer, liveness_error)
                    self._latch_abort(liveness_error)
                    executor.log_error(fl_ctx, f"{liveness_error} before task '{task_name}' produced a result")
                    return make_reply(ReturnCode.EXECUTION_EXCEPTION)

                now = time.monotonic()
                if wait_deadline is not None and now >= wait_deadline:
                    self._send_abort(
                        trainer, f"'{task_name}' timed out after {result_wait_timeout}s waiting for result"
                    )
                    self._latch_abort(f"result wait timed out for task '{task_name}'")
                    executor.log_error(
                        fl_ctx, f"timed out after {result_wait_timeout}s waiting for '{task_name}' result"
                    )
                    return make_reply(ReturnCode.EXECUTION_EXCEPTION)

                wait_time = _RESULT_POLL_INTERVAL
                if wait_deadline is not None:
                    wait_time = min(wait_time, wait_deadline - now)
                task.result_ready.wait(wait_time)

            # Preserve lazy references for ClientRunner forwarding.
            with self._task_lock:
                result = task.result
            if not isinstance(result, Shareable):
                executor.log_error(fl_ctx, f"bad task result from trainer: expect Shareable but got {type(result)}")
                return make_reply(ReturnCode.EXECUTION_EXCEPTION)

            current_round = shareable.get_header(AppConstants.CURRENT_ROUND)
            if current_round is not None:
                result.set_header(AppConstants.CURRENT_ROUND, current_round)
            return result
        finally:
            with self._task_lock:
                if self._current_task is task:
                    self._current_task = None

    def _send_task_ready(self, trainer: _TrainerSession, task_message: dict, abort_signal: Signal) -> Tuple[str, Any]:
        """Send TASK_READY with native cancellation for abort and liveness failures."""
        timeout = self._context.task_wait_timeout
        transfer_waiters = []

        def _on_transaction_created(transaction):
            transfer_waiters.append(DownloadService.get_transfer_waiter(transaction.tx_id))

        def _has_live_task_download():
            return any(not waiter.done() for waiter in tuple(transfer_waiters))

        def _cancel_cause():
            if abort_signal.triggered or (self._context.launch_once and self._abort):
                return _SEND_ABORTED, None
            if self._closed:
                return _SEND_CLOSED, None
            if not self._process_group_alive(trainer):
                return _SEND_PROCESS_DEAD, None
            # Transfer progress supersedes heartbeat expiry while materialization is active.
            if not _has_live_task_download():
                liveness_error = self._trainer_liveness_error(trainer)
                if liveness_error:
                    return _SEND_SESSION_DEAD, liveness_error
            return None

        cancel = _TaskReadyCancelSignal(_cancel_cause)
        try:
            reply = self._cell.send_request(
                channel=CHANNEL,
                topic=Topic.TASK_READY,
                target=trainer.trainer_fqcn,
                request=new_cell_message({}, task_message),
                timeout=timeout,
                abort_signal=cancel,
                progress_wait_cb=_has_live_task_download,
                receiver_ids=(trainer.trainer_fqcn,),
                fobs_ctx_props={
                    FOBSContextKey.STREAM_PROGRESS_CB: lambda **_kwargs: None,
                    RESULT_UPLOAD_TX_CREATED_CB_CTX_KEY: _on_transaction_created,
                },
            )
        except BaseException:
            cause = cancel.value
            self._delete_task_transfers(transfer_waiters)
            if cancel.error is not None:
                raise cancel.error
            if cause is not None:
                return cause
            raise

        cause = cancel.value
        if cancel.error is not None:
            self._delete_task_transfers(transfer_waiters)
            raise cancel.error
        if cause is not None:
            self._delete_task_transfers(transfer_waiters)
            return cause
        if self._check_task_accepted(reply) is not None:
            # A rejected task has no future consumer for its payload. Receiver
            # confirmation is asynchronous, so retire the source deterministically.
            self._delete_task_transfers(transfer_waiters)
        return _SEND_OK, reply

    @staticmethod
    def _delete_task_transfers(transfer_waiters) -> None:
        for waiter in transfer_waiters:
            try:
                DownloadService.delete_transaction(waiter.transaction_id)
            except Exception:
                # Preserve the task's original abort/transport error. The transaction's
                # own timeout remains the cleanup backstop if deletion itself fails.
                pass

    def _check_task_accepted(self, reply) -> Optional[str]:
        """Returns a rejection reason, or None when the trainer accepted the task."""
        if reply is None:
            return "no reply from trainer"
        rc = reply.get_header(MessageHeaderKey.RETURN_CODE)
        if rc != CellReturnCode.OK:
            return f"cell-level failure delivering TASK_READY: {rc}"
        body = reply.payload
        if not isinstance(body, dict):
            return f"invalid TASK_READY reply payload: expect dict but got {type(body)}"
        reply_topic = body.get(MsgKey.REPLY_TOPIC)
        if reply_topic != Topic.TASK_ACCEPTED:
            return f"trainer replied {reply_topic}: {body.get(MsgKey.REASON)}"
        return None

    # ------------------------------------------------------------------ control-plane handlers

    def _register_protocol_cbs(self, cell) -> None:
        # NOTE: cellnet request callbacks cannot be unregistered. The CJ cell is job-scoped
        # and every state-mutating handler is gated on self._closed after backend teardown.
        cell.register_request_cb(channel=CHANNEL, topic=Topic.HELLO, cb=self._handle_hello)
        cell.register_request_cb(channel=CHANNEL, topic=Topic.RESULT_READY, cb=self._handle_result_ready)
        cell.register_request_cb(channel=CHANNEL, topic=Topic.LOG, cb=self._handle_log)
        cell.register_request_cb(channel=CHANNEL, topic=Topic.HEARTBEAT, cb=self._handle_heartbeat)

    @staticmethod
    def _protocol_reply(reply_topic: str, **fields):
        body = {MsgKey.REPLY_TOPIC: reply_topic}
        body.update(fields)
        # semantic accept/reject rides an rc=OK reply body (see defs.Topic: reply-type
        # messages are modeled as request replies); cell-level rc is for transport faults
        return make_cell_reply(CellReturnCode.OK, body=body)

    def _handle_hello(self, request):
        """Validates HELLO per the V1 trusted-host proof: plain launch-token match, plus
        identity (prescribed FQCN), protocol version, job id, and rank-0 checks."""
        if self._closed:
            return self._protocol_reply(Topic.HELLO_REJECTED, **{MsgKey.REASON: "backend is closed"})
        payload = request.payload
        if not isinstance(payload, dict):
            return make_cell_reply(CellReturnCode.INVALID_REQUEST, error="HELLO payload must be a dict")

        origin = request.get_header(MessageHeaderKey.ORIGIN) or ""
        trainer = self._active_launch
        if trainer is None:
            return self._hello_reject(trainer, origin, "no active trainer launch", latch=False)

        # A foreign identity is not evidence that the prescribed trainer failed.
        claimed_fqcn = payload.get(MsgKey.TRAINER_FQCN)
        if origin != trainer.trainer_fqcn or claimed_fqcn != trainer.trainer_fqcn:
            return self._hello_reject(
                trainer,
                origin,
                f"unexpected trainer identity (origin={origin!r}, claimed={claimed_fqcn!r})",
                latch=False,
            )

        # Validate type before compare_digest so non-ASCII or forged proofs reject cleanly.
        proof = payload.get(MsgKey.PROOF)
        if (
            not isinstance(proof, str)
            or not trainer.token
            or not secrets.compare_digest(proof.encode("utf-8"), trainer.token.encode("utf-8"))
        ):
            return self._hello_reject(trainer, origin, "launch token mismatch", latch=True)

        if payload.get(MsgKey.PROTOCOL_VERSION) != PROTOCOL_VERSION:
            return self._hello_reject(
                trainer,
                origin,
                f"unsupported protocol version {payload.get(MsgKey.PROTOCOL_VERSION)!r} (expect {PROTOCOL_VERSION})",
                latch=True,
            )

        if payload.get(MsgKey.JOB_ID) != self._job_id:
            return self._hello_reject(trainer, origin, f"job id mismatch: {payload.get(MsgKey.JOB_ID)!r}", latch=True)

        if payload.get(MsgKey.SITE_NAME) != self._site_name:
            return self._hello_reject(
                trainer,
                origin,
                f"site name mismatch: {payload.get(MsgKey.SITE_NAME)!r}",
                latch=True,
            )

        # A nonzero rank does not latch rejection because rank zero may still connect.
        rank = payload.get(MsgKey.RANK)
        if str(rank) != "0":
            return self._hello_reject(trainer, origin, f"only rank 0 may connect (got rank {rank!r})", latch=False)

        # Concurrent duplicate HELLOs must receive the same session id.
        with self._launch_lock:
            if not trainer.ready.is_set():
                trainer.session_id = uuid.uuid4().hex
                trainer.ready.set()
                self.logger.info(f"HELLO accepted from {origin} (session_id={trainer.session_id})")
            trainer.touch_peer_activity()
        return self._protocol_reply(
            Topic.HELLO_ACCEPTED,
            **{
                MsgKey.SESSION_ID: trainer.session_id,
                MsgKey.JOB_ID: self._job_id,
                MsgKey.SITE_NAME: self._site_name,
                MsgKey.HEARTBEAT_INTERVAL: self._context.heartbeat_interval,
                MsgKey.HEARTBEAT_TIMEOUT: self._context.heartbeat_timeout,
            },
        )

    def _hello_reject(self, trainer: Optional[_TrainerSession], origin: str, reason: str, latch: bool):
        self.logger.warning(f"rejecting HELLO from {origin!r}: {reason}")
        if latch and trainer is not None and not trainer.ready.is_set() and trainer.reject_reason is None:
            trainer.reject_reason = reason
        return self._protocol_reply(Topic.HELLO_REJECTED, **{MsgKey.REASON: reason})

    def _validate_session_msg(self, request, payload) -> Tuple[Optional[_TrainerSession], Optional[str]]:
        """Binds a post-HELLO message to the current authenticated session."""
        trainer = self._active_launch
        if trainer is None or not trainer.ready.is_set() or trainer.session_id is None:
            return None, "no active trainer session"
        origin = request.get_header(MessageHeaderKey.ORIGIN) or ""
        if origin != trainer.trainer_fqcn:
            return None, f"unexpected origin {origin!r}"
        if payload.get(MsgKey.SESSION_ID) != trainer.session_id:
            return None, "stale or unknown session id"
        trainer.touch_peer_activity()
        return trainer, None

    def _handle_result_ready(self, request):
        """Accept a possibly lazy result; attach retries require paired receiver deduplication."""
        if self._closed:
            return self._protocol_reply(Topic.RESULT_REJECTED, **{MsgKey.REASON: "backend is closed"})
        payload = request.payload
        if not isinstance(payload, dict):
            return make_cell_reply(CellReturnCode.INVALID_REQUEST, error="RESULT_READY payload must be a dict")

        trainer, reject_reason = self._validate_session_msg(request, payload)
        if reject_reason:
            self.logger.warning(f"rejecting RESULT_READY: {reject_reason}")
            return self._protocol_reply(Topic.RESULT_REJECTED, **{MsgKey.REASON: reject_reason})

        task_id = payload.get(MsgKey.TASK_ID)
        result = payload.get(MsgKey.RESULT)
        # Validate and commit atomically against task retirement and teardown.
        with self._task_lock:
            if self._closed:
                return self._protocol_reply(Topic.RESULT_REJECTED, **{MsgKey.REASON: "backend is closed"})
            task = self._current_task
            if task is None or task_id != task.task_id:
                reason = f"no current task matching task_id {task_id!r}"
                self.logger.warning(f"rejecting RESULT_READY: {reason}")
                return self._protocol_reply(Topic.RESULT_REJECTED, **{MsgKey.REASON: reason})

            if task.result is not None:
                return self._protocol_reply(
                    Topic.RESULT_REJECTED,
                    **{MsgKey.REASON: "a result was already accepted for this task"},
                )
            if not isinstance(result, Shareable):
                return self._protocol_reply(
                    Topic.RESULT_REJECTED,
                    **{MsgKey.REASON: "invalid result envelope: Shareable result required"},
                )
            task.result = result
            # Acceptance precedes send settlement, even for an inline result.
            trainer.result_source_live.set()
            task.result_ready.set()
        return self._protocol_reply(Topic.RESULT_ACCEPTED)

    def _handle_log(self, request):
        """Route trainer LOG data without raising into the Cell dispatcher."""
        if self._closed:
            return None
        try:
            payload = request.payload
            if not isinstance(payload, dict):
                self.logger.error(f"invalid LOG data format, expecting Dict, but got {type(payload)}")
                return None
            trainer, reject_reason = self._validate_session_msg(request, payload)
            if reject_reason:
                self.logger.warning(f"dropping LOG data: {reject_reason}")
                return None
            record = {k: v for k, v in payload.items() if k != MsgKey.SESSION_ID}
            if "key" in record:
                record["tag"] = record.pop("key")
            dxo = create_analytic_dxo(**record)
            with self._engine.new_context() as fl_ctx:
                self._context.executor.fire_log_analytics(fl_ctx, dxo)
        except Exception:
            self.logger.error(f"failed to process trainer LOG data: {secure_format_traceback()}")
        return None

    def _handle_heartbeat(self, request):
        if self._closed:
            return self._protocol_reply(Topic.ERROR, **{MsgKey.REASON: "backend is closed"})
        payload = request.payload
        if not isinstance(payload, dict):
            return make_cell_reply(CellReturnCode.INVALID_REQUEST, error="HEARTBEAT payload must be a dict")
        trainer, reject_reason = self._validate_session_msg(request, payload)
        if reject_reason:
            self.logger.warning(f"rejecting HEARTBEAT: {reject_reason}")
            return self._protocol_reply(Topic.ERROR, **{MsgKey.REASON: reject_reason})
        return self._protocol_reply(Topic.HEARTBEAT, **{MsgKey.SESSION_ID: trainer.session_id})

    # ------------------------------------------------------------------ helpers

    def _send_abort(self, trainer: Optional[_TrainerSession], reason: str) -> None:
        if trainer is None or trainer.session_id is None:
            return
        try:
            self._cell.fire_and_forget(
                channel=CHANNEL,
                topic=Topic.ABORT,
                targets=[trainer.trainer_fqcn],
                message=new_cell_message({}, {MsgKey.SESSION_ID: trainer.session_id, MsgKey.REASON: reason}),
                optional=True,
            )
        except Exception:
            self.logger.error(secure_format_traceback())

    def _latch_abort(self, reason: str) -> None:
        self._abort = True
        if self._abort_reason is None:
            self._abort_reason = reason

    def _task_exchange_config(self) -> dict:
        context = self._context
        return {
            ConfigKey.TRAIN_WITH_EVAL: context.train_with_evaluation,
            ConfigKey.EXCHANGE_FORMAT: context.params_exchange_format,
            ConfigKey.SERVER_EXPECTED_FORMAT: context.server_expected_format,
            ConfigKey.TRANSFER_TYPE: context.params_transfer_type,
            ConfigKey.TRAIN_TASK_NAME: context.train_task_name,
            ConfigKey.EVAL_TASK_NAME: context.evaluate_task_name,
            ConfigKey.SUBMIT_MODEL_TASK_NAME: context.submit_model_task_name,
            ConfigKey.LAUNCH_ONCE: context.launch_once,
        }
