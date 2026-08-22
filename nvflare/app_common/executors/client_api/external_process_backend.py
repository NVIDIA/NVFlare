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

import ipaddress
import os
import secrets
import signal
import subprocess
import threading
import time
import uuid
from typing import Any, Optional, Sequence, Tuple, Union

from nvflare.apis.fl_constant import (
    CellMessageAuthHeaderKey,
    ConnectionSecurity,
    FLContextKey,
    FLMetaKey,
    ReturnCode,
    ServerCommandNames,
)
from nvflare.apis.fl_context import FLContext
from nvflare.apis.fl_exception import UnsafeJobError
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.executors.client_api.backend_spec import ClientAPIBackendContext
from nvflare.app_common.executors.client_api.cell_backend import CellBackendBase, CellSession, CellTask
from nvflare.client.api_spec import CLIENT_API_TYPE_KEY
from nvflare.client.cell.bootstrap import (
    BOOTSTRAP_FILE_ENV_VAR,
    BOOTSTRAP_SCHEMA_VERSION,
    EXTERNAL_PROCESS_EXECUTION_MODE,
    BootstrapKey,
    bootstrap_file_name,
    write_bootstrap_config,
)
from nvflare.client.cell.defs import CHANNEL, PROTOCOL_VERSION, SESSION_CONTROL_TIMEOUT, MsgKey, Topic
from nvflare.fuel.common.exit_codes import ProcessExitCode
from nvflare.fuel.f3.cellnet.defs import CellChannel, MessageHeaderKey
from nvflare.fuel.f3.cellnet.defs import ReturnCode as CellReturnCode
from nvflare.fuel.f3.cellnet.fqcn import FQCN
from nvflare.fuel.f3.cellnet.utils import make_reply as make_cell_reply
from nvflare.fuel.f3.cellnet.utils import new_cell_message
from nvflare.fuel.f3.drivers.driver_params import DriverParams
from nvflare.fuel.f3.drivers.net_utils import parse_url
from nvflare.fuel.f3.streaming.download_service import DownloadService
from nvflare.fuel.utils.fobs import FOBSContextKey
from nvflare.fuel.utils.fobs.decomposers.via_downloader import RESULT_UPLOAD_TX_CREATED_CB_CTX_KEY, LazyDownloadRef
from nvflare.security.logging import secure_format_exception, secure_format_traceback
from nvflare.utils.job_launcher_utils import add_custom_dir_to_path
from nvflare.utils.process_utils import log_subprocess_output, prepare_subprocess_command

# Poll cadence for process-death detection; events wake successful waits immediately.
_RESULT_POLL_INTERVAL = 0.5
_RESULT_SOURCE_FAILURE_DELIVERY_WAIT = 14.0
_HELLO_POLL_INTERVAL = 0.1

_DEFAULT_SHUTDOWN_TIMEOUT = SESSION_CONTROL_TIMEOUT

# The result reaper reserves TERM grace inside each session-scale cleanup budget
# and makes one final bounded state probe before force-cleaning a live source.
_LIVE_RESULT_SHUTDOWN_ACK_TIMEOUT = 5.0
_RESULT_REAPER_MAX_TOTAL_TIMEOUT = SESSION_CONTROL_TIMEOUT
_RESULT_REAPER_FORCE_TERM_GRACE = 5.0

_LOG_THREAD_JOIN_TIMEOUT = 5.0

# A fresh FQCN prevents stale trainer cells from colliding with later launches.
_TRAINER_LEAF_PREFIX = "client_api_trainer"


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

_TERMINAL_INTENT_ABORT = "abort"
_TERMINAL_INTENT_FAILURE = "failure"
_TERMINAL_INTENT_SHUTDOWN = "shutdown"


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


class _TrainerSession(CellSession):
    """One launched trainer process and its (at most one) authenticated protocol session."""

    def __init__(self, token: str, trainer_fqcn: str):
        super().__init__(trainer_fqcn)
        self.token = token
        # Latched when a token-authenticated HELLO is rejected so the launch wait fails
        # fast instead of waiting out launch_timeout.
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
        self.reaper_thread: Optional[threading.Thread] = None
        self.source_monitor_thread: Optional[threading.Thread] = None
        self._result_failure_lock = threading.Lock()
        self.result_source_refs = ()
        self.result_receiver_ids = ()
        self.result_failure_notified = False
        self.result_failure_delivery_done = threading.Event()
        self.result_failure_delivery_done.set()
        self.shutdown_requested = threading.Event()
        self._shutdown_request_lock = threading.Lock()
        self._next_shutdown_retry = 0.0
        self._stop_lock = threading.Lock()
        self._cleanup_lock = threading.Lock()
        self._cleaned = False


# Kept as a private compatibility alias for existing backend tests/extensions.
_TaskContext = CellTask


class ExternalProcessBackend(CellBackendBase):
    """Launches and owns the external trainer process/group, bridged over the CJ cell."""

    def __init__(self):
        super().__init__()
        self._connect_url: Optional[str] = None
        self._run_dir: Optional[str] = None
        self._app_dir: Optional[str] = None
        self._custom_dir: Optional[str] = None
        self._active_launch: Optional[_TrainerSession] = None
        self._launch_lock = threading.Lock()
        # A per-task launch can keep serving an accepted lazy result after the next
        # launch becomes active. Keep every owned protocol session addressable until
        # its process/result source is fully retired.
        self._protocol_sessions = {}
        # Completed per-task launches can keep serving lazy results and remain owned through END_RUN.
        self._result_reapers = set()
        self._result_reapers_lock = threading.Lock()
        self._launch_seq = 0
        self._abort = False
        self._abort_reason: Optional[str] = None
        self._run_abort_signal: Optional[Signal] = None
        self._lifecycle_lock = threading.RLock()
        self._terminal_intent: Optional[str] = None
        self._failure_panic_sent = False

    # ------------------------------------------------------------------ lifecycle

    def initialize(self, context: ClientAPIBackendContext, fl_ctx: FLContext) -> None:
        if not context.command:
            raise ValueError("external_process mode requires a non-empty command")

        try:
            self._run_abort_signal = fl_ctx.get_run_abort_signal()
            self._initialize_cell(
                context,
                fl_ctx,
                "external_process",
                pass_through_routes=(
                    (CellChannel.SERVER_COMMAND, ServerCommandNames.GET_TASK),
                    (CHANNEL, Topic.RESULT_READY),
                ),
                delegate_site_auth=True,
            )
            cell = self._cell
            # A managed trainer may fail immediately after RESULT_READY is accepted,
            # before this CJ starts consuming a peer's result refs. Register the
            # source-failure route at job initialization so such a notice is not lost.
            DownloadService.initialize(cell)

            workspace = fl_ctx.get_prop(FLContextKey.WORKSPACE_OBJECT)
            if workspace is None:
                raise RuntimeError("workspace/job id not available in fl_ctx")
            self._run_dir = workspace.get_run_dir(self._job_id)
            self._app_dir = workspace.get_app_dir(self._job_id)
            self._custom_dir = workspace.get_app_custom_dir(self._job_id)

            cell.make_internal_listener(
                scheme="tcp",
                resources={
                    DriverParams.HOST.value: "localhost",
                    DriverParams.LISTEN_HOST.value: "127.0.0.1",
                    DriverParams.CONNECTION_SECURITY.value: ConnectionSecurity.CLEAR,
                },
            )
            connect_url = cell.get_internal_listener_url()
            if not connect_url:
                raise RuntimeError("CJ cell has no internal listener url for the trainer to connect to")
            listener_params = cell.get_internal_listener_params() or {}
            connect_scheme = parse_url(connect_url).get(DriverParams.SCHEME.value)
            listener_scheme = listener_params.get(DriverParams.SCHEME.value)
            bind_host = listener_params.get(DriverParams.HOST.value)
            connection_security = listener_params.get(DriverParams.CONNECTION_SECURITY.value)
            try:
                loopback_bound = ipaddress.ip_address(bind_host).is_loopback
            except (TypeError, ValueError):
                loopback_bound = False
            if (
                connect_scheme != "tcp"
                or listener_scheme != "tcp"
                or connection_security != ConnectionSecurity.CLEAR
                or not loopback_bound
            ):
                raise RuntimeError(
                    "external_process trainer requires a clear TCP listener bound to loopback, but the CJ internal "
                    f"listener is incompatible: connect_scheme={connect_scheme!r}, listener_scheme={listener_scheme!r}, "
                    f"bind_host={bind_host!r}, connection_security={connection_security!r}"
                )
            self._connect_url = connect_url

            cell.register_request_cb(channel=CHANNEL, topic=Topic.HELLO, cb=self._handle_hello)
            cell.register_request_cb(channel=CHANNEL, topic=Topic.SESSION_READY, cb=self._handle_session_ready)

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

    def abort(self, fl_ctx: FLContext) -> None:
        """Notify the trainer of task cancellation and latch abortive run teardown.

        An accepted lazy result can outlive ``execute()`` while another client
        downloads it. Explicit run abort therefore carries a context marker so
        it remains distinguishable from normal END_RUN and CCWF task cancellation.
        Callers predating the marker retain the original run-abort behavior.
        """
        run_abort_requested = fl_ctx.get_prop(FLContextKey.RUN_ABORT_REQUESTED, True)
        if run_abort_requested:
            self._claim_abort_intent()
            self._latch_abort("run abort requested")
        with self._launch_lock:
            trainer = self._active_launch
        self._send_abort(trainer, "run aborted" if run_abort_requested else "task aborted")

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
                self._finish_task_trainer(trainer, launch_once=True)
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
            self._finish_task_trainer(trainer, launch_once, task_aborted=abort_signal.triggered)

    def _finish_task_trainer(
        self, trainer: Optional[_TrainerSession], launch_once: bool, task_aborted: bool = False
    ) -> None:
        """Retire a per-task trainer, or a persistent trainer after a terminal abort."""
        if trainer is None or (launch_once and not self._abort):
            return
        try:
            if launch_once:
                # A persistent trainer cannot serve another task after the backend latches
                # an abort. Stop it before execute() returns: abort teardown may destroy the
                # CJ process without delivering END_RUN to this executor.
                self._stop_trainer(trainer, natural_exit_wait=self._stop_wait_bound())
            elif trainer.result_accepted.is_set():
                self._reap_trainer_after_result(trainer)
            else:
                # A task-only cancellation must not latch run-wide abort intent, but
                # its one-task trainer still cannot be reused and must be stopped
                # without spending the ordinary natural-exit wait.
                natural_exit_wait = 0.0 if task_aborted else self._stop_wait_bound()
                self._stop_trainer(trainer, natural_exit_wait=natural_exit_wait)
        except Exception:
            self.logger.error(secure_format_traceback())

    def finalize(self, fl_ctx: FLContext) -> None:
        if self._finalized:
            return
        self._finalized = True
        # Serialize close with RESULT_READY's acceptance commit and with the
        # fatal-source/explicit-abort terminal decision.
        with self._task_lock:
            with self._lifecycle_lock:
                if self._terminal_intent is None:
                    self._terminal_intent = _TERMINAL_INTENT_SHUTDOWN
                self._closed = True
        # The same gate orders END_RUN against the launch-install-to-Popen window. Keep
        # this ordering bound on abort: a process handle may not have been installed yet.
        admitted = self._execute_gate.acquire(timeout=self._shutdown_wait_bound())
        # Cleanup is unconditional if the gate times out. _stop_trainer then waits on the
        # session stop lock until an in-flight Popen installs its process handle.
        try:
            with self._launch_lock:
                trainer = self._active_launch
            if trainer is not None:
                with self._result_reapers_lock:
                    reaper_owns_exit = trainer in self._result_reapers
                if reaper_owns_exit and not self._abort:
                    # A per-task trainer with an accepted result is already owned by
                    # its natural-exit reaper. Do not race that exit with another
                    # synchronous SHUTDOWN request during END_RUN.
                    pass
                elif trainer.result_source_live.is_set() and not self._abort:
                    # END_RUN must preserve CJ/F3 until the trainer's send barrier settles.
                    self._request_trainer_shutdown(trainer, wait_timeout=_LIVE_RESULT_SHUTDOWN_ACK_TIMEOUT)
                    self._reap_trainer_after_result(trainer)
                else:
                    self._stop_trainer(trainer, natural_exit_wait=self._stop_wait_bound())
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
            self._protocol_sessions[trainer_fqcn] = trainer
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
                    BootstrapKey.CJ_PID: os.getpid(),
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

            launch_blocked = False
            with trainer._stop_lock:
                # Serialize Popen and handle installation with teardown. Once Popen has
                # created a child, finalize must not return before that child is owned
                # and terminated.
                if self._closed or trainer._cleaned:
                    launch_blocked = True
                else:
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
                    trainer.process = process
                    if os.name == "posix":
                        # start_new_session made the child its own group leader (pgid == pid)
                        trainer.pgid = process.pid
            if launch_blocked:
                raise RuntimeError("backend closed before trainer launch")
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

    def _stop_trainer(
        self,
        trainer: _TrainerSession,
        natural_exit_wait: float,
        termination_grace: Optional[float] = None,
    ) -> None:
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
                grace = self._termination_grace() if termination_grace is None else max(0.0, termination_grace)
                self._terminate_process_tree(trainer, grace=grace)
            except Exception:
                self.logger.error(secure_format_traceback())
            self._cleanup_trainer(trainer)

    def _request_trainer_shutdown(
        self, trainer: _TrainerSession, wait_timeout: float, force_probe: bool = False
    ) -> None:
        """Request orderly SHUTDOWN and sample an accepted result source until it settles."""
        with trainer._shutdown_request_lock:
            if trainer.shutdown_requested.is_set():
                return
            now = time.monotonic()
            if not force_probe and now < trainer._next_shutdown_retry:
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
                            # Keep probing: this acknowledgement describes the current
                            # transfer barrier and is not a terminal SHUTDOWN acknowledgement.
                            return
                        elif source_live is False:
                            trainer.result_source_live.clear()
                            trainer.result_source_task_id = None
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
        """Wait boundedly for result sources without preempting a settled trainer's final work."""
        started = time.monotonic()
        settled_wait, settled_term_grace = self._settled_result_reaper_budget()
        live_wait = max(0.0, self._result_reaper_wait_bound() - settled_term_grace)
        live_deadline = started if self._abort else started + live_wait
        settled_deadlines = {}
        with self._result_reapers_lock:
            pending = set(self._result_reapers)

        # No new reaper can be admitted after finalize owns _execute_gate. Track
        # this fixed snapshot locally so an unexpectedly slow daemon cannot make
        # END_RUN loop forever after its bounded forced cleanup.
        while pending:
            for trainer in tuple(pending):
                reaper = trainer.reaper_thread
                if reaper is None or not reaper.is_alive():
                    pending.discard(trainer)
                    continue

                live = trainer.result_source_live.is_set()
                if live:
                    # If a later probe reports the source live again, its next
                    # settled observation must start a new natural-exit budget.
                    settled_deadlines.pop(trainer, None)
                    deadline = live_deadline
                else:
                    now = time.monotonic()
                    deadline = settled_deadlines.setdefault(
                        trainer,
                        now if self._abort else now + settled_wait,
                    )
                remaining = deadline - time.monotonic()
                if remaining > 0:
                    reaper.join(timeout=min(_NATURAL_EXIT_REAP_INTERVAL, remaining))
                    continue

                if live and not self._abort:
                    # Close the polling race at the hard deadline. The trainer publishes
                    # transfer-barrier completion before waiting for RESULT_SOURCE_SETTLED,
                    # so this probe can observe a clean source even if that request is queued.
                    self._request_trainer_shutdown(
                        trainer,
                        wait_timeout=_LIVE_RESULT_SHUTDOWN_ACK_TIMEOUT,
                        force_probe=True,
                    )
                    live = trainer.result_source_live.is_set()
                    if not live:
                        settled_deadlines[trainer] = time.monotonic() + settled_wait
                        continue

                self.logger.warning(
                    f"timed out waiting for accepted result source {trainer.trainer_fqcn} "
                    f"(live={live}); forcing trainer cleanup"
                )
                # Reserve the capped TERM grace inside the live-source cleanup bound.
                # A newly settled source instead receives its own natural-exit budget
                # before the same TERM grace is applied.
                grace = 0.0 if self._abort else settled_term_grace
                self._stop_trainer(trainer, natural_exit_wait=0.0, termination_grace=grace)
                reaper.join(timeout=_LOG_THREAD_JOIN_TIMEOUT)
                if reaper.is_alive():
                    self.logger.error(f"result-source reaper {reaper.name} did not stop after forced cleanup")
                pending.discard(trainer)

    def _wait_for_natural_exit_and_cleanup(self, trainer: _TrainerSession) -> None:
        disconnected_since = None
        fail_client_job = False
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
                    if self._abort:
                        # Abort teardown must never enter the normal accepted-source
                        # SHUTDOWN acknowledgement wait. The receiver cancellation
                        # may already have settled the source while its notification
                        # is still crossing the Cell.
                        self._stop_trainer(trainer, natural_exit_wait=0.0)
                        return
                    if not trainer.result_source_live.is_set():
                        if self._await_group_exit(trainer, self._result_source_disconnect_grace()):
                            self._cleanup_trainer(trainer)
                        else:
                            self._stop_trainer(trainer, natural_exit_wait=0.0)
                        return
                    # SHUTDOWN cannot preempt an accepted result source still inside send().
                    self._request_trainer_shutdown(trainer, wait_timeout=_LIVE_RESULT_SHUTDOWN_ACK_TIMEOUT)
                time.sleep(_NATURAL_EXIT_REAP_INTERVAL)
            failure_reason = self._trainer_exit_reason(trainer)
            fail_client_job = self._fail_accepted_result_source(
                trainer,
                trainer.result_source_task_id,
                failure_reason,
            )
            self._cleanup_trainer(trainer)
            if fail_client_job:
                self._fail_job_for_lost_result_source(failure_reason)
        except BaseException:
            self.logger.error(secure_format_traceback())
        finally:
            if not trainer._cleaned:
                self._stop_trainer(trainer, natural_exit_wait=0.0)
            with self._result_reapers_lock:
                self._result_reapers.discard(trainer)

    def _cleanup_trainer(self, trainer: _TrainerSession) -> None:
        """Release launch-scoped state after the process group is gone. Idempotent."""
        fail_client_job = False
        failure_reason = None
        if trainer.result_source_live.is_set() and not self._process_group_alive(trainer):
            failure_reason = self._trainer_exit_reason(trainer)
            fail_client_job = self._fail_accepted_result_source(
                trainer,
                trainer.result_source_task_id,
                failure_reason,
            )
        if trainer.result_failure_notified and not trainer.result_failure_delivery_done.is_set():
            trainer.result_failure_delivery_done.wait(_RESULT_SOURCE_FAILURE_DELIVERY_WAIT)
        with trainer._cleanup_lock:
            if trainer._cleaned:
                return
            # A reaped owned process group cannot remain a result source. Keep
            # this launch-scoped truth consistent even when its final SHUTDOWN
            # acknowledgement was lost during Cell/F3 teardown.
            trainer.result_source_live.clear()
            trainer.result_accepted.clear()
            trainer.result_source_task_id = None
            trainer._cleaned = True
        try:
            log_thread = trainer.log_thread
            if log_thread is not None and log_thread.is_alive():
                log_thread.join(timeout=_LOG_THREAD_JOIN_TIMEOUT)
        except Exception:
            self.logger.error(secure_format_traceback())
        trainer.token = ""
        try:
            if trainer.bootstrap_path and os.path.exists(trainer.bootstrap_path):
                os.remove(trainer.bootstrap_path)
        except Exception as e:
            self.logger.debug(f"failed to remove {trainer.bootstrap_path}: {e}")
        with self._launch_lock:
            trainer.session_id = None
            self._protocol_sessions.pop(trainer.trainer_fqcn, None)
            if self._active_launch is trainer:
                self._active_launch = None
        if fail_client_job:
            self._fail_job_for_lost_result_source(failure_reason)

    def _disable_task_pass_through(self) -> None:
        self._disable_pass_through()

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

    @staticmethod
    def _trainer_exit_reason(trainer: _TrainerSession) -> str:
        rc = trainer.process.poll() if trainer.process else None
        return f"accepted external result source died before transfer completion (rc={rc})"

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
        return prepare_subprocess_command(command)

    def _shutdown_wait_bound(self) -> float:
        shutdown_timeout = self._context.shutdown_timeout
        return _DEFAULT_SHUTDOWN_TIMEOUT if shutdown_timeout is None else shutdown_timeout

    def _stop_wait_bound(self) -> float:
        return 0.0 if self._abort else self._shutdown_wait_bound()

    def _termination_grace(self) -> float:
        return self._context.stop_grace_period

    def _result_source_disconnect_grace(self) -> float:
        """Return a nonzero disconnect grace for an accepted result source."""
        shutdown_bound = self._shutdown_wait_bound()
        return shutdown_bound if shutdown_bound > 0 else _DEFAULT_SHUTDOWN_TIMEOUT

    def _result_reaper_wait_bound(self) -> float:
        """Return the session-scale cleanup bound for a live result source.

        A source transaction has its own streaming idle timeout, but END_RUN cannot
        wait for that independently long timeout: the outer job process may tear down
        the CJ first and orphan an owned per-task trainer. The short SHUTDOWN request
        repeatedly samples whether send() still owns its transfer barrier, including
        one final sample at the deadline. This lets completed transfer cleanup settle
        the source even when its task-correlated settlement request is delayed. The
        separately configured natural-exit grace starts when the source is first
        observed settled.
        """
        return _RESULT_REAPER_MAX_TOTAL_TIMEOUT

    def _settled_result_reaper_budget(self) -> Tuple[float, float]:
        """Return natural-exit and TERM budgets for a settled one-task trainer.

        Reserve a small configured TERM grace inside a fixed total bound. This
        prevents END_RUN from cutting off normal post-send work at the live-source
        acknowledgement deadline while ensuring the CJ remains bounded.
        """
        term_grace = min(max(0.0, self._termination_grace()), _RESULT_REAPER_FORCE_TERM_GRACE)
        natural_cap = max(0.0, _RESULT_REAPER_MAX_TOTAL_TIMEOUT - term_grace)
        # An accepted result source needs a nonzero settlement grace even when the
        # general trainer shutdown timeout is configured as fire-and-forget (zero).
        natural_wait = min(self._result_source_disconnect_grace(), natural_cap)
        return natural_wait, term_grace

    def _unwind(self) -> None:
        """Releases partial setup after a failed initialize(). Best-effort per step."""
        with self._lifecycle_lock:
            if self._terminal_intent is None:
                self._terminal_intent = _TERMINAL_INTENT_SHUTDOWN
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

        task = CellTask(task_id=uuid.uuid4().hex)
        task.result_receiver_ids = self._get_result_receiver_ids(shareable, fl_ctx)

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

    @staticmethod
    def _get_result_receiver_ids(shareable: Shareable, fl_ctx: FLContext) -> tuple[str, ...]:
        receiver_ids = shareable.get_header(FOBSContextKey.RECEIVER_IDS)
        if isinstance(receiver_ids, str):
            receiver_ids = (receiver_ids,)
        if isinstance(receiver_ids, (list, tuple)):
            valid = tuple(dict.fromkeys(r for r in receiver_ids if isinstance(r, str) and not FQCN.validate(r)))
            if valid:
                return valid
        job_id = fl_ctx.get_job_id()
        return (FQCN.join([FQCN.ROOT_SERVER, job_id]),) if isinstance(job_id, str) and job_id else ()

    def _on_result_accepted(self, session: CellSession, task: CellTask, result: Shareable) -> None:
        if not isinstance(session, _TrainerSession):
            return
        try:
            refs = tuple(sorted(self._collect_result_source_refs(result, session.trainer_fqcn)))
        except BaseException:
            self.logger.error(secure_format_traceback())
            return
        if not refs or not task.result_receiver_ids:
            return
        with session._result_failure_lock:
            session.result_source_refs = refs
            session.result_receiver_ids = task.result_receiver_ids
            session.result_failure_notified = False
            session.result_failure_delivery_done.set()
            task_id = task.task_id
            monitor = threading.Thread(
                target=self._monitor_accepted_result_source,
                args=(session, task_id),
                name=f"client_api_result_source_{session.trainer_fqcn.rsplit('.', 1)[-1]}",
                daemon=True,
            )
            session.source_monitor_thread = monitor
            try:
                monitor.start()
            except BaseException:
                session.source_monitor_thread = None
                self.logger.error(secure_format_traceback())

    def _on_result_source_settled(self, session: CellSession, task_id: str) -> None:
        if not isinstance(session, _TrainerSession):
            return
        with session._result_failure_lock:
            session.result_source_refs = ()
            session.result_receiver_ids = ()

    @staticmethod
    def _collect_result_source_refs(value, source_fqcn: str, visited=None) -> set[str]:
        if isinstance(value, LazyDownloadRef):
            return {value.ref_id} if value.fqcn == source_fqcn and value.ref_id else set()
        if not isinstance(value, (dict, list, tuple, set)):
            return set()
        if visited is None:
            visited = set()
        value_id = id(value)
        if value_id in visited:
            return set()
        visited.add(value_id)
        items = (*value.keys(), *value.values()) if isinstance(value, dict) else value
        refs = set()
        for item in items:
            refs.update(ExternalProcessBackend._collect_result_source_refs(item, source_fqcn, visited))
        return refs

    def _monitor_accepted_result_source(self, trainer: _TrainerSession, task_id: str) -> None:
        disconnected_since = None
        try:
            while True:
                with trainer._result_failure_lock:
                    active = (
                        not trainer._cleaned
                        and trainer.result_source_live.is_set()
                        and trainer.result_source_task_id == task_id
                    )
                if not active:
                    return
                if not self._process_group_alive(trainer):
                    reason = self._trainer_exit_reason(trainer)
                    fail_client_job = self._fail_accepted_result_source(trainer, task_id, reason)
                    self._cleanup_trainer(trainer)
                    if fail_client_job:
                        self._fail_job_for_lost_result_source(reason)
                    return
                heartbeat_timeout = self._context.heartbeat_timeout
                silent_for = trainer.peer_silent_for() if heartbeat_timeout > 0 else None
                if silent_for is not None and silent_for > heartbeat_timeout:
                    reason = (
                        f"accepted external result source heartbeat timed out after {silent_for:.1f}s "
                        f"(timeout={heartbeat_timeout}s)"
                    )
                    fail_client_job = self._fail_accepted_result_source(trainer, task_id, reason)
                    self._stop_trainer(trainer, natural_exit_wait=0.0)
                    if fail_client_job:
                        self._fail_job_for_lost_result_source(reason)
                    return
                if self._cell.is_cell_connected(trainer.trainer_fqcn):
                    disconnected_since = None
                elif disconnected_since is None:
                    disconnected_since = time.monotonic()
                elif time.monotonic() - disconnected_since >= self._result_source_disconnect_grace():
                    reason = "accepted external result source disconnected before transfer completion"
                    fail_client_job = self._fail_accepted_result_source(trainer, task_id, reason)
                    self._stop_trainer(trainer, natural_exit_wait=0.0)
                    if fail_client_job:
                        self._fail_job_for_lost_result_source(reason)
                    return
                time.sleep(_RESULT_POLL_INTERVAL)
        except BaseException:
            self.logger.error(secure_format_traceback())

    def _fail_accepted_result_source(self, trainer: _TrainerSession, task_id: Optional[str], reason: str) -> bool:
        if not task_id:
            return False
        with trainer._result_failure_lock:
            if (
                trainer.result_failure_notified
                or not trainer.result_source_live.is_set()
                or trainer.result_source_task_id != task_id
            ):
                return False
            refs = trainer.result_source_refs
            receivers = trainer.result_receiver_ids
            if not refs or not receivers:
                return False
            fail_client_job = self._claim_result_source_failure()
            trainer.result_failure_notified = True
            trainer.result_failure_delivery_done.clear()
            trainer.result_source_live.clear()
            trainer.result_source_task_id = None
        self.logger.warning(
            f"notifying {len(receivers)} receiver(s) that accepted result source "
            f"{trainer.trainer_fqcn} failed for task {task_id}: {reason}"
        )
        try:
            errors = DownloadService.notify_source_failure(
                cell=self._cell,
                targets=receivers,
                source_fqcn=trainer.trainer_fqcn,
                ref_ids=refs,
                reason=reason,
                secure=self._secure_mode,
            )
            if isinstance(errors, dict):
                for target, error in errors.items():
                    if error:
                        self.logger.warning(f"failed to notify result receiver {target}: {error}")
        except Exception:
            self.logger.error(secure_format_traceback())
        finally:
            trainer.result_failure_delivery_done.set()
        return fail_client_job

    def _claim_abort_intent(self) -> bool:
        """Atomically let explicit abort win over a not-yet-classified source failure."""
        with self._lifecycle_lock:
            if self._terminal_intent is None:
                self._terminal_intent = _TERMINAL_INTENT_ABORT
            return self._terminal_intent == _TERMINAL_INTENT_ABORT

    def _claim_result_source_failure(self) -> bool:
        """Atomically classify source loss as fatal unless abort/shutdown already won."""
        with self._lifecycle_lock:
            if self._terminal_intent is not None or self._closed:
                return False
            abort_signal = self._run_abort_signal
            if abort_signal is not None and abort_signal.triggered:
                self._terminal_intent = _TERMINAL_INTENT_ABORT
                return False
            self._terminal_intent = _TERMINAL_INTENT_FAILURE
            return True

    def _fail_job_for_lost_result_source(self, reason: str) -> None:
        """Fail the run through normal CJ teardown with a reportable process code."""
        with self._lifecycle_lock:
            if self._terminal_intent != _TERMINAL_INTENT_FAILURE or self._failure_panic_sent:
                return
            self._failure_panic_sent = True
        self._latch_abort(reason)
        self.logger.critical(f"{reason}; failing the client job because the lazy result cannot be recovered")
        self._write_process_exit_code(ProcessExitCode.EXCEPTION)
        run_abort_signal = self._run_abort_signal
        if run_abort_signal is not None and not run_abort_signal.triggered:
            run_abort_signal.trigger(reason)
        try:
            with self._engine.new_context() as fl_ctx:
                self._context.executor.system_panic(reason, fl_ctx)
        except Exception:
            # The run signal above still releases ClientRunner so its finally path
            # can archive workspace results and preserve the reportable RC file.
            self.logger.error(f"failed to publish fatal result-source event: {secure_format_traceback()}")

    def _write_process_exit_code(self, return_code: int) -> None:
        """Preserve a reportable code across launchers that normalize nonzero child exits."""
        run_dir = self._run_dir
        if not run_dir:
            self.logger.error("cannot record client job failure: run directory is unavailable")
            return
        rc_file = os.path.join(run_dir, FLMetaKey.PROCESS_RC_FILE)
        try:
            with open(rc_file, "w", encoding="utf-8") as f:
                f.write(str(return_code))
        except Exception as e:
            self.logger.error(f"cannot record client job failure in {rc_file}: {secure_format_exception(e)}")

    def _send_task_ready(self, trainer: _TrainerSession, task_message: dict, abort_signal: Signal) -> Tuple[str, Any]:
        """Send TASK_READY until reply, cancelling on abort, closure, process death, or deadline."""
        max_timeout = self._context.task_wait_timeout
        transfer_waiters = []
        started = time.monotonic()

        def _on_transaction_created(transaction):
            transfer_waiters.append(DownloadService.get_transfer_waiter(transaction.tx_id))

        def _cancel_cause():
            if abort_signal.triggered or (self._context.launch_once and self._abort):
                return _SEND_ABORTED, None
            if self._closed:
                return _SEND_CLOSED, None
            if not self._process_group_alive(trainer):
                return _SEND_PROCESS_DEAD, None
            if max_timeout is not None and time.monotonic() - started >= max_timeout:
                return _SEND_SESSION_DEAD, f"TASK_READY timed out after {max_timeout}s"
            return None

        cancel = _TaskReadyCancelSignal(_cancel_cause)
        try:
            reply = self._cell.send_request(
                channel=CHANNEL,
                topic=Topic.TASK_READY,
                target=trainer.trainer_fqcn,
                request=new_cell_message({}, task_message),
                timeout=None,
                abort_signal=cancel,
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

        cause = cancel.value if cancel.triggered else None
        if cancel.error is not None:
            self._delete_task_transfers(transfer_waiters)
            raise cancel.error
        if cause is not None:
            self._delete_task_transfers(transfer_waiters)
            return cause
        if reply is not None and reply.get_header(MessageHeaderKey.RETURN_CODE) == CellReturnCode.OK:
            trainer.touch_peer_activity()
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

    def _get_protocol_session(self, origin: Optional[str] = None) -> Optional[_TrainerSession]:
        with self._launch_lock:
            if origin:
                return self._protocol_sessions.get(origin)
            return self._active_launch

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
            # Clear loopback transport cannot prove that an invalid-token sender is the launched trainer.
            return self._hello_reject(trainer, origin, "launch token mismatch", latch=False)

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

        # Concurrent duplicate HELLOs must receive the same session id. Do not
        # expose the trainer as ready yet: the reply carries delegated site
        # authentication, and only SESSION_READY proves that the trainer has
        # processed it and installed the outgoing auth-header filters.
        with self._launch_lock:
            if trainer.session_id is None:
                trainer.session_id = uuid.uuid4().hex
                self.logger.info(
                    f"HELLO accepted from {origin} (session_id={trainer.session_id}); awaiting SESSION_READY"
                )
            trainer.touch_peer_activity()
            session_id = trainer.session_id
        return self._protocol_reply(
            Topic.HELLO_ACCEPTED,
            **{
                MsgKey.SESSION_ID: session_id,
                MsgKey.JOB_ID: self._job_id,
                MsgKey.SITE_NAME: self._site_name,
                MsgKey.HEARTBEAT_INTERVAL: self._context.heartbeat_interval,
                MsgKey.HEARTBEAT_TIMEOUT: self._context.heartbeat_timeout,
                **self._session_security_payload(),
            },
        )

    def _handle_session_ready(self, request):
        """Complete HELLO only after the trainer has installed delegated authentication."""
        if self._closed:
            return self._protocol_reply(Topic.ERROR, **{MsgKey.REASON: "backend is closed"})
        payload = request.payload
        if not isinstance(payload, dict):
            return make_cell_reply(CellReturnCode.INVALID_REQUEST, error="SESSION_READY payload must be a dict")

        origin = request.get_header(MessageHeaderKey.ORIGIN) or ""
        session_id = None
        with self._launch_lock:
            trainer = self._active_launch
            if trainer is None or trainer.session_id is None:
                reason = "no accepted trainer session"
            elif origin != trainer.trainer_fqcn:
                reason = f"unexpected origin {origin!r}"
            elif payload.get(MsgKey.SESSION_ID) != trainer.session_id:
                reason = "stale or unknown session id"
            elif self._secure_mode and not self._delegated_auth_headers_match(request):
                reason = "delegated site authentication headers are not installed"
            else:
                reason = None
                session_id = trainer.session_id
                trainer.touch_peer_activity()
                if not trainer.ready.is_set():
                    trainer.ready.set()
                    self.logger.info(f"trainer readiness confirmed from {origin} (session_id={session_id})")

        if reason:
            self.logger.warning(f"rejecting SESSION_READY: {reason}")
            return self._protocol_reply(Topic.ERROR, **{MsgKey.REASON: reason})
        return self._protocol_reply(Topic.SESSION_READY, **{MsgKey.SESSION_ID: session_id})

    def _delegated_auth_headers_match(self, request) -> bool:
        expected = (
            (CellMessageAuthHeaderKey.CLIENT_NAME, self._site_name),
            (CellMessageAuthHeaderKey.TOKEN, self._site_auth_token),
            (CellMessageAuthHeaderKey.TOKEN_SIGNATURE, self._site_auth_token_signature),
        )
        for key, value in expected:
            actual = request.get_header(key)
            if not isinstance(actual, str) or not isinstance(value, str):
                return False
            if not secrets.compare_digest(actual.encode("utf-8"), value.encode("utf-8")):
                return False
        return True

    def _hello_reject(self, trainer: Optional[_TrainerSession], origin: str, reason: str, latch: bool):
        self.logger.warning(f"rejecting HELLO from {origin!r}: {reason}")
        if latch and trainer is not None and trainer.session_id is None and trainer.reject_reason is None:
            trainer.reject_reason = reason
        return self._protocol_reply(Topic.HELLO_REJECTED, **{MsgKey.REASON: reason})

    def _latch_abort(self, reason: str) -> None:
        with self._lifecycle_lock:
            self._abort = True
            if self._abort_reason is None:
                self._abort_reason = reason
