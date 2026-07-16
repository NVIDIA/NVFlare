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

"""external_process backend for ClientAPIExecutor.

Design: docs/design/client_api_execution_modes.md ("external_process", "Control Protocol",
"Payload Lifecycle State Machine"). The backend launches and owns the external trainer
process tree, and talks to it over the CJ cell using the implemented protocol vocabulary in
nvflare/client/cell/defs.py — no PipeHandler, no CellPipe, no MetricRelay:

- **Launch**: writes the bootstrap config (0600; fresh launch-scoped token, connect URL,
  prescribed trainer FQCN — see nvflare/client/cell/bootstrap.py), starts the configured
  command in its own process group, and waits for the trainer's HELLO. The token is a
  plain match over the localhost connection NVFlare itself created (V1 trusted host).
- **Control plane**: TASK_READY with TASK_ACCEPTED/TASK_FAILED replies,
  RESULT_READY with RESULT_ACCEPTED/RESULT_REJECTED replies, and LOG, HEARTBEAT, ABORT,
  SHUTDOWN as plain Cell messages. Process-group exit and the
  authenticated heartbeat lease are independent liveness signals: the former detects a
  dead trainer tree, while the latter detects a live tree whose Cell session is wedged.
- **Payload plane**: task and result Shareables ride directly in the Cell requests. Cell's
  FOBS encoder uses ViaDownloader for large tensors, so the existing F3 transaction is the
  only payload lifecycle. The CJ remains a pass-through hop, preserving ClientRunner's
  existing behavior: configured filters receive the payload representation delivered by
  the transport rather than triggering a new materialization policy here.
- **Teardown**: orderly stop is SHUTDOWN, a bounded wait for natural exit
  (shutdown_timeout), then SIGTERM -> stop_grace_period -> SIGKILL to the process GROUP.
  An accepted result is the one exception: teardown requests SHUTDOWN but preserves the
  trainer until its send barrier closes the Cell; an asynchronous reaper then cleans it
  up, while END_RUN preserves the CJ until natural exit. Process
  ownership is keyed off group liveness rather than the
  launcher handle (a torchrun/mpirun launcher can
  exit while its workers keep the group alive). Non-POSIX limitation: there is
  no group-liveness probe, so tree death is inferred from the launcher handle; both stop
  stages therefore use ``taskkill /T`` on Windows (soft, then ``/F``) so a launcher-only
  terminate cannot strand workers — full tree ownership there needs the Job-Object work
  the design assigns to that platform.

Like the in_process backend, LOG data is routed through the executor-owned
fire_log_analytics(); this backend selects the federation-scoped fire path
(set_analytics_fire_fed_event(True)) for parity with MetricRelay's ex-process behavior.

Client configuration permits one ClientAPIExecutor total per client job. That executor
routes all Client API task names through one selected backend and one control plane.

This backend is the CJ side of the protocol; its counterpart is the trainer-side Cell
engine (nvflare/client/cell/api.py), which reads the bootstrap config, HELLOs, and serves
tasks to user code.
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

# Poll cadence of the result-wait loop and the HELLO wait (result arrival wakes the loop
# early through the task's event; the poll only bounds process-death detection latency).
_RESULT_POLL_INTERVAL = 0.5
_HELLO_POLL_INTERVAL = 0.1

# Orderly-stop bound applied when the frozen surface's shutdown_timeout is None
# ("None means the backend default" per the executor docstring).
_DEFAULT_SHUTDOWN_TIMEOUT = 30.0

# A possibly-active accepted result must receive SHUTDOWN reliably: unlike ordinary
# zero-timeout teardown, the backend cannot force-stop it while send() is settling. Keep
# this control acknowledgement independently bounded and retry it from the result reaper.
_LIVE_RESULT_SHUTDOWN_ACK_TIMEOUT = 5.0

# Bound for joining the subprocess-stdout log thread after the process is gone.
_LOG_THREAD_JOIN_TIMEOUT = 5.0

# The prescribed FQCN leaf of the trainer's child cell: <cj_fqcn>.<prefix>_<launch_seq>.
# A fresh leaf per launch means a stale process's cell can never collide with (or receive
# messages meant for) the current launch's cell.
_TRAINER_LEAF_PREFIX = "client_api_trainer"


def bootstrap_file_name(seq: int) -> str:
    """The launch-scoped bootstrap file name, written into the job's app dir.

    Launch-scoped (not a fixed name) on purpose: the file delivers the launch token and
    prescribed FQCN, so a survivor of launch N-1 that re-reads its config file must find
    its OWN (invalidated) credentials — never launch N's. The file is removed once its
    launch stops.
    """
    return f"client_api_bootstrap_{seq}.json"


class _LaunchAborted(Exception):
    """The task's abort_signal triggered while waiting for a per-task trainer launch."""


# _send_task_ready outcomes (the request itself runs on a helper thread; these name the
# condition that ended the wait).
_SEND_OK = "ok"
_SEND_ABORTED = "aborted"
_SEND_PROCESS_DEAD = "process_dead"
_SEND_SESSION_DEAD = "session_dead"
_SEND_CLOSED = "closed"

# CHANNEL is streaming-capable (cellnet treats every non-excluded channel as one), so
# TASK_READY goes through Cell._send_request, whose timeout is a NO-PROGRESS bound and
# must be numeric (a None reaches conditional_wait and raises). This is the bound applied
# when the frozen surface's task_wait_timeout is None ("no timeout"): an hour of NO
# progress on a small control message is equivalent in practice, and the real bounds on
# the wait are abort/process-death/close, which cancel the request natively.
_TASK_READY_NO_PROGRESS_TIMEOUT = 3600.0

# Bounded join for the sender helper after its request was cancelled: the real streaming
# wait returns promptly on the cancel signal; a fake/legacy path that ignores it is
# abandoned (daemon) after this bound rather than blocking the task result.
_SENDER_CANCEL_JOIN_TIMEOUT = 1.0

# An accepted result keeps the trainer alive until flare.send() crosses its terminal
# barrier and returns. The trainer normally exits itself; this daemon only reaps its
# launch artifacts without racing the result acknowledgement or payload path.
_NATURAL_EXIT_REAP_INTERVAL = 0.1
_SHUTDOWN_RETRY_INTERVAL = 1.0


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
        # No per-task lock: all of a task's result/failure fields, the _current_task
        # pointer are guarded by the backend's single _task_lock, so current-task
        # validation and the acceptance/failure commit are one critical section (they
        # cannot interleave with teardown's clear).
        self.result_ready = threading.Event()
        self.result: Optional[Shareable] = None


class ExternalProcessBackend(ClientAPIBackendSpec):
    """Launches and owns the external trainer process tree, bridged over the CJ cell."""

    def __init__(self):
        super().__init__()
        # the spec is a plain ABC: fl_ctx-aware logging goes through the executor back-reference
        # (context.executor.log_*); this logger covers cell-callback paths that have no fl_ctx
        self.logger = get_obj_logger(self)
        self._context: Optional[ClientAPIBackendContext] = None
        self._engine = None
        self._cell = None
        self._cj_fqcn: Optional[str] = None
        self._connect_url: Optional[str] = None
        self._pass_through_route: Optional[Tuple[str, str]] = None
        self._owns_pass_through_route = False
        self._job_id: Optional[str] = None
        self._site_name: Optional[str] = None
        self._app_dir: Optional[str] = None
        self._custom_dir: Optional[str] = None
        self._session: Optional[_TrainerSession] = None
        self._session_lock = threading.Lock()
        # A per-task launch can be replaced in ``self._session`` while its accepted
        # result is still finishing send(). Keep explicit ownership of every such reaper
        # so END_RUN cannot return and tear down the CJ Cell underneath an older send.
        self._result_reapers = set()
        self._result_reapers_lock = threading.Lock()
        self._launch_seq = 0
        self._current_task: Optional[_TaskContext] = None
        self._task_lock = threading.Lock()
        # admission gate: one active execute() at a time (one active task per session).
        # try-acquired at execute() entry, released in its finally.
        self._execute_gate = threading.Lock()
        self._abort = False
        self._abort_reason: Optional[str] = None
        self._finalized = False
        # Gates every state-mutating cell callback after finalize()/unwind. Cellnet has no
        # unregister for request callbacks; the CJ cell is job-scoped so the registrations
        # die with it, and a later backend on the same cell would overwrite them — the gate
        # covers the window in between (a late message from an abandoned trainer is politely
        # rejected).
        self._closed = False

    # ------------------------------------------------------------------ lifecycle

    def initialize(self, context: ClientAPIBackendContext, fl_ctx: FLContext) -> None:
        self._context = context
        if not context.command:
            # the executor constructor already enforces this; kept as a defensive contract check
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

            # Decode only GET_TASK replies lazily in the CJ so an external trainer can
            # download directly from the original source. SERVER_COMMAND also carries
            # unrelated request types (notably subordinate-client SUBMIT_UPDATE), so a
            # channel-wide opt-in would silently expose lazy refs to their handlers.
            self._pass_through_route = (CellChannel.SERVER_COMMAND, ServerCommandNames.GET_TASK)
            self._owns_pass_through_route = self._pass_through_route not in cell.decode_pass_through_topics
            cell.decode_pass_through_topics.add(self._pass_through_route)

            workspace = fl_ctx.get_prop(FLContextKey.WORKSPACE_OBJECT)
            job_id = fl_ctx.get_prop(FLContextKey.CURRENT_JOB_ID)
            if workspace is None or not job_id:
                raise RuntimeError("workspace/job id not available in fl_ctx")
            self._job_id = job_id
            self._site_name = fl_ctx.get_identity_name()
            self._app_dir = workspace.get_app_dir(job_id)
            self._custom_dir = workspace.get_app_custom_dir(job_id)

            # idempotent if the cell already has one
            cell.make_internal_listener()
            connect_url = cell.get_internal_listener_url()
            if not connect_url:
                raise RuntimeError("CJ cell has no internal listener url for the trainer to connect to")
            self._connect_url = connect_url

            self._register_protocol_cbs(cell)
            # Register only decomposers required by the declared exchange pair. RAW disables
            # adaptation, so its concrete representation is handled opportunistically.
            register_framework_decomposers(context.params_exchange_format, context.server_expected_format, self.logger)

            # ex-process analytics parity with MetricRelay (fed_event=True): fire the
            # federation-scoped event directly. ConvertToFedEvent (if configured) only
            # re-fires LOCAL events, so this cannot double-deliver metrics.
            context.executor.set_analytics_fire_fed_event(True)

            if context.launch_once:
                self._start_session(timeout=context.launch_timeout)
        except BaseException:
            # contract (backend_spec): initialize() self-unwinds its partial setup on failure;
            # the executor does not call finalize() on a half-initialized backend. BaseException
            # ensures launch artifacts are also cleaned up for KeyboardInterrupt/SystemExit.
            self._unwind()
            raise

    def execute(self, task_name: str, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        context = self._context
        executor = context.executor
        executor.log_info(fl_ctx, f"execute for task ({task_name})")

        if self._closed:
            # END_RUN teardown already ran (or initialize unwound): a task racing finalize
            # must not launch a fresh trainer after the job's teardown finished
            executor.log_error(fl_ctx, f"backend is closed; failing task '{task_name}'")
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)

        # One active task per session (design invariant). ClientRunner does not itself
        # serialize executor calls, so a second task routed to this single external_process
        # executor while one is in flight would otherwise overwrite _current_task/_session —
        # the first task's RESULT_READY would then be rejected as "no current task". Admit
        # exactly one execute at a time; reject a concurrent second WITHOUT touching the
        # active one's state. (Non-blocking so a stuck task cannot pile up waiters.)
        if not self._execute_gate.acquire(blocking=False):
            # contract: a triggered abort_signal must yield TASK_ABORTED even here. Do NOT
            # touch the active task's session — this call never owned it.
            if abort_signal.triggered:
                return make_reply(ReturnCode.TASK_ABORTED)
            executor.log_error(
                fl_ctx,
                f"a task is already executing on this external_process backend; rejecting concurrent "
                f"task '{task_name}' (one active task per session)",
            )
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)

        if abort_signal.triggered:
            if context.launch_once:
                aborted_session = self._session
                self._send_abort(aborted_session, f"'{task_name}' is aborted, abort_signal_triggered")
                if aborted_session is not None:
                    # The wire ABORT ends the persistent trainer session; later tasks fail
                    # fast (parity with the in_process backend's abort-event latch).
                    self._latch_abort(f"'{task_name}' aborted at entry, abort_signal_triggered")
            # In launch-per-task mode this invocation owns no session yet. self._session
            # may still name the previous task's successful live result source, which must
            # not be aborted by a new task that never launched.
            self._execute_gate.release()
            return make_reply(ReturnCode.TASK_ABORTED)

        launch_once = context.launch_once
        session = self._session
        try:
            if launch_once:
                # An externally launched trainer whose session died (ABORT sent, process
                # exit, HELLO never completed) is gone for good in launch_once mode — the
                # process is never relaunched. Fail fast with an accurate return code
                # (mirrors the in_process backend's latched-abort behavior).
                if self._abort:
                    executor.log_error(
                        fl_ctx,
                        f"external trainer is no longer available (reason: {self._abort_reason}); "
                        f"failing task '{task_name}'",
                    )
                    return make_reply(ReturnCode.EXECUTION_EXCEPTION)
                if session is None or not session.ready.is_set():
                    executor.log_error(fl_ctx, f"no established trainer session; failing task '{task_name}'")
                    return make_reply(ReturnCode.EXECUTION_EXCEPTION)
                liveness_error = self._session_liveness_error(session)
                if liveness_error:
                    self._latch_abort(liveness_error)
                    executor.log_error(fl_ctx, f"{liveness_error}; failing task '{task_name}'")
                    return make_reply(ReturnCode.EXECUTION_EXCEPTION)
            else:
                try:
                    session = self._start_session(timeout=context.launch_timeout, abort_signal=abort_signal)
                except _LaunchAborted:
                    # contract: never hang (or fail inaccurately) past abort — the launch
                    # was already stopped by _start_session's unwind
                    executor.log_info(fl_ctx, f"'{task_name}' aborted while launching the trainer")
                    return make_reply(ReturnCode.TASK_ABORTED)
                except Exception:
                    executor.log_error(fl_ctx, f"per-task trainer launch failed: {secure_format_traceback()}")
                    return make_reply(ReturnCode.EXECUTION_EXCEPTION)

            return self._run_task(session, task_name, shareable, fl_ctx, abort_signal)
        except UnsafeJobError:
            raise
        except Exception:
            executor.log_error(fl_ctx, secure_format_traceback())
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)
        finally:
            # a raising per-task stop (e.g. deep in _sweep/_process_group_alive) must neither
            # skip the gate release (or every later task hits the busy rejection forever) nor
            # mask the task result — so catch+log the stop, then always release the gate.
            try:
                if not launch_once:
                    stale = session
                    if stale is not None:
                        if stale.result_source_live.is_set():
                            # RESULT_ACCEPTED can race its reply delivery and, when the
                            # result is streamed, the lazy payload transfer. Let the trainer
                            # gate its own exit on the complete send barrier and reap it
                            # asynchronously.
                            self._reap_session_after_result(stale)
                        else:
                            self._stop_session(stale, natural_exit_wait=self._shutdown_wait_bound())
            except Exception:
                self.logger.error(secure_format_traceback())
            finally:
                self._execute_gate.release()

    def finalize(self, fl_ctx: FLContext) -> None:
        # contract: idempotent and must not raise
        if self._finalized:
            return
        self._finalized = True
        # Order closure against RESULT_READY's acceptance commit. A callback that commits
        # first makes result_source_live visible to teardown; one that reaches the critical
        # section later is rejected and cannot create a source after finalize classified it.
        with self._task_lock:
            self._closed = True
        # Barrier on any in-flight execute(): with _closed set, a mid-launch execute bails at
        # the pre-Popen check and releases the gate, so acquiring it here means "no execute is
        # between session install and Popen". This prevents a trainer being launched after we
        # return, and lets a per-task execute finish its own unwind (removing its launch
        # artifacts) before finalize returns. Bounded: a long RUNNING
        # task holds the gate past this wait; we then proceed to stop its session (failing it),
        # matching END_RUN-mid-task, and that execute releases the gate on its own exit.
        admitted = self._execute_gate.acquire(timeout=self._shutdown_wait_bound())
        try:
            # read under the session lock: a concurrent _start_session installs its session
            # under this lock (after checking _closed), so teardown either sees the session
            # or the launch fails on the closed check — never an unseen orphan
            with self._session_lock:
                session = self._session
            if session is not None:
                if session.result_source_live.is_set():
                    # RESULT_ACCEPTED can precede reply delivery and, when present,
                    # lazy-transfer settlement. Ask the trainer to stop, but let its send
                    # barrier keep the process alive until it is safe. Preserve the CJ/Cell
                    # infrastructure
                    # until the reaper observes the trainer's truthful terminal signal
                    # (natural process/Cell exit). Returning with a daemon reaper is not
                    # safe: ClientRunner tears down DownloadService and the CJ Cell as soon
                    # as END_RUN completes. Stalled transfers are bounded by the source
                    # DownloadService's idle/receiver budgets, not by shutdown_timeout.
                    self._request_session_shutdown(session, wait_timeout=_LIVE_RESULT_SHUTDOWN_ACK_TIMEOUT)
                    self._reap_session_after_result(session)
                else:
                    self._stop_session(session, natural_exit_wait=self._shutdown_wait_bound())
            # ``self._session`` names only the newest launch. Earlier launch_per_task
            # sessions remain valid DownloadService sources until their own terminal
            # transfer outcome, so preserve the job Cell until all owned reapers finish.
            self._wait_for_result_reapers()
        except Exception:
            self.logger.error(secure_format_traceback())
        finally:
            self._disable_task_pass_through()
            if admitted:
                self._execute_gate.release()

    # ------------------------------------------------------------------ session management

    def _start_session(self, timeout: Optional[float], abort_signal: Optional[Signal] = None) -> _TrainerSession:
        """Writes a fresh bootstrap config, launches the trainer, and waits for its HELLO.

        Raises on any failure — _LaunchAborted if abort_signal triggered during the wait —
        after stopping whatever it started (the caller never inherits a half-launched
        session).
        """
        token = secrets.token_urlsafe(32)
        with self._session_lock:
            # serialized with finalize()'s session clear: a launch must not slip past a
            # concurrent END_RUN teardown unseen
            if self._closed:
                raise RuntimeError("backend is closed; not launching a trainer")
            self._launch_seq += 1
            seq = self._launch_seq
            trainer_fqcn = FQCN.join([self._cj_fqcn, f"{_TRAINER_LEAF_PREFIX}_{seq}"])
            session = _TrainerSession(token, trainer_fqcn)
            # install before launch so the HELLO handler resolves the current launch
            self._session = session
        try:
            bootstrap_path = os.path.join(self._app_dir, bootstrap_file_name(seq))
            session.bootstrap_path = bootstrap_path
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
                    BootstrapKey.TASK_EXCHANGE: self._task_exchange_config(),
                    BootstrapKey.MEMORY_GC_ROUNDS: self._context.memory_gc_rounds,
                    BootstrapKey.CUDA_EMPTY_CACHE: self._context.cuda_empty_cache,
                },
            )

            env = os.environ.copy()
            env[BOOTSTRAP_FILE_ENV_VAR] = bootstrap_path
            # The typed bootstrap is the launch selector. Do not inherit a legacy API-type
            # override that could conflict with it in the trainer process.
            env.pop(CLIENT_API_TYPE_KEY, None)
            add_custom_dir_to_path(self._custom_dir, env)

            # Last gate before spawning: finalize() may have set _closed after this session
            # was installed (above) but before this Popen. Bail now so no trainer process is
            # ever started after END_RUN teardown began. finalize() additionally barriers on
            # the execute gate, so it will not return until this unwind completes.
            if self._closed:
                raise RuntimeError("backend closed before trainer launch")

            # Never log the configured command: legacy/hand-written jobs may contain literal
            # credentials rather than site-resolved secret references.
            self.logger.info(f"launching external trainer (launch {seq})")
            session.process = subprocess.Popen(
                self._split_command(self._context.command),
                shell=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=self._app_dir,
                env=env,
                # own process group/session so orderly stop can signal the whole tree
                start_new_session=(os.name == "posix"),
            )
            if os.name == "posix":
                # start_new_session made the child its own group leader (pgid == pid)
                session.pgid = session.process.pid
            session.log_thread = threading.Thread(
                target=log_subprocess_output,
                args=(session.process, self.logger),
                name=f"client_api_trainer_log_{seq}",
                daemon=True,
            )
            session.log_thread.start()

            if self._closed:
                # finalize() may have swept between the closed check above and Popen;
                # do not inherit an orphan across END_RUN
                raise RuntimeError("backend closed during trainer launch")

            self._wait_for_hello(session, timeout, abort_signal)
            self.logger.info(
                f"trainer session established: launch={seq} fqcn={trainer_fqcn} session_id={session.session_id}"
            )
            return session
        except Exception:
            # no SHUTDOWN and no natural-exit wait: the trainer never completed HELLO,
            # so terminate the tree directly (bounded by stop_grace_period)
            self._stop_session(session, natural_exit_wait=0.0)
            raise

    def _wait_for_hello(
        self, session: _TrainerSession, timeout: Optional[float], abort_signal: Optional[Signal] = None
    ) -> None:
        """Waits for the launched process's accepted HELLO (this replaces the legacy
        external_pre_init_timeout / first-Pipe-heartbeat readiness wait).

        None timeout means no deadline (the frozen surface's launch_timeout contract);
        trainer process death, task abort, and backend close still bound the wait — the
        execute() contract forbids hanging past abort.
        """
        deadline = None if timeout is None else time.monotonic() + timeout
        while not session.ready.wait(_HELLO_POLL_INTERVAL):
            if abort_signal is not None and abort_signal.triggered:
                raise _LaunchAborted("task aborted while waiting for the trainer HELLO")
            if self._closed:
                raise RuntimeError("backend closed while waiting for the trainer HELLO")
            if session.reject_reason:
                raise RuntimeError(f"trainer HELLO was rejected: {session.reject_reason}")
            if not self._process_group_alive(session):
                rc = session.process.poll() if session.process else None
                raise RuntimeError(f"trainer process group exited (rc={rc}) before completing the HELLO handshake")
            if deadline is not None and time.monotonic() >= deadline:
                raise RuntimeError(f"trainer did not complete the HELLO handshake within launch_timeout={timeout}s")

    def _stop_session(self, session: _TrainerSession, natural_exit_wait: float) -> None:
        """Stops one trainer session and its process tree. Must not raise; idempotent.

        Order: bounded SHUTDOWN request/ack, remaining natural-exit wait, then SIGTERM ->
        stop_grace_period -> SIGKILL to the process group. The SHUTDOWN request time is
        charged against natural_exit_wait rather than extending teardown. A zero bound
        preserves immediate fire-and-forget notification. The token is invalidated so a
        surviving process can never authenticate against a later launch.
        """
        with session._stop_lock:
            if session._cleaned:
                return
            natural_exit_wait = max(0.0, natural_exit_wait)
            natural_exit_deadline = time.monotonic() + natural_exit_wait
            try:
                remaining = max(0.0, natural_exit_deadline - time.monotonic())
                self._request_session_shutdown(session, wait_timeout=remaining)
            except Exception:
                self.logger.error(secure_format_traceback())
            try:
                process = session.process
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
                    if leader_exited and os.name == "posix" and session.pgid is not None and remaining > 0:
                        # A launcher/torchrun leader may exit before its trainer workers.
                        # Preserve the same natural-exit grace for the whole owned group;
                        # otherwise a truthful RESULT_SOURCE_LIVE=False ACK can still be
                        # followed by an immediate SIGTERM while send() is returning.
                        self._await_group_exit(session, remaining)
            except Exception:
                self.logger.error(secure_format_traceback())
            try:
                self._terminate_process_tree(session, grace=self._context.stop_grace_period)
            except Exception:
                self.logger.error(secure_format_traceback())
            self._cleanup_session(session)

    def _request_session_shutdown(self, session: _TrainerSession, wait_timeout: float) -> None:
        """Send at most one orderly SHUTDOWN without taking ownership of process death."""
        with session._shutdown_request_lock:
            if session.shutdown_requested.is_set():
                return
            now = time.monotonic()
            if now < session._next_shutdown_retry:
                return
            session._next_shutdown_retry = now + _SHUTDOWN_RETRY_INTERVAL
            if session.session_id is None or not self._process_group_alive(session):
                return
            request = new_cell_message({}, {MsgKey.SESSION_ID: session.session_id, MsgKey.REASON: "shutdown requested"})
            try:
                if wait_timeout > 0:
                    reply = self._cell.send_request(
                        channel=CHANNEL,
                        topic=Topic.SHUTDOWN,
                        target=session.trainer_fqcn,
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
                            # The trainer is still inside send(); natural exit is the only
                            # safe proof that its reply/payload barrier has finished.
                            session.result_source_live.set()
                        elif source_live is False:
                            # The trainer serialized send-completion and SHUTDOWN under one
                            # lock. False proves the result transaction had settled before
                            # it handled this request, so ordinary process-exit grace applies.
                            session.result_source_live.clear()
                else:
                    send_errors = self._cell.fire_and_forget(
                        channel=CHANNEL,
                        topic=Topic.SHUTDOWN,
                        targets=[session.trainer_fqcn],
                        message=request,
                        optional=True,
                    )
                    send_error = send_errors.get(session.trainer_fqcn) if isinstance(send_errors, dict) else None
                    if send_error:
                        self.logger.warning(f"trainer SHUTDOWN was not delivered: {send_error}")
                        return
                session.shutdown_requested.set()
            except Exception:
                # A transient control-path failure must not abandon ownership. The result
                # reaper retries at a bounded cadence while preserving the data source;
                # Cell disconnect remains the proof that it is safe to terminate.
                self.logger.error(secure_format_traceback())

    def _reap_session_after_result(self, session: _TrainerSession) -> None:
        """Reap a successful one-task trainer after it finishes serving its result."""
        with session._cleanup_lock:
            if session._cleaned or (session.reaper_thread is not None and session.reaper_thread.is_alive()):
                return
            session.reaper_thread = threading.Thread(
                target=self._wait_for_natural_exit_and_cleanup,
                args=(session,),
                name=f"client_api_trainer_reaper_{session.trainer_fqcn.rsplit('.', 1)[-1]}",
                daemon=True,
            )
            # Register ownership before start: a very short-lived process may otherwise
            # finish and disappear before finalize() can observe this reaper.
            with self._result_reapers_lock:
                self._result_reapers.add(session)
                try:
                    # Keep registration and start atomic to finalize: Thread.join() on a
                    # registered-but-not-yet-started reaper raises RuntimeError.
                    session.reaper_thread.start()
                except BaseException:
                    self._result_reapers.discard(session)
                    session.reaper_thread = None
                    raise

    def _wait_for_result_reapers(self) -> None:
        """Wait boundedly for accepted result sends, then force-clean wedged sources."""
        wait_bound = self._result_reaper_wait_bound()
        deadline = time.monotonic() + wait_bound
        while True:
            with self._result_reapers_lock:
                sessions = tuple(self._result_reapers)
            if not sessions:
                return
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            for session in sessions:
                reaper = session.reaper_thread
                if reaper is not None:
                    reaper.join(timeout=max(0.0, deadline - time.monotonic()))

        with self._result_reapers_lock:
            wedged = tuple(self._result_reapers)
        if wedged:
            self.logger.warning(
                f"timed out waiting {wait_bound}s for {len(wedged)} accepted result source(s); "
                "forcing trainer cleanup"
            )
        for session in wedged:
            self._stop_session(session, natural_exit_wait=0.0)
        for session in wedged:
            reaper = session.reaper_thread
            if reaper is not None:
                reaper.join(timeout=_LOG_THREAD_JOIN_TIMEOUT)

    def _wait_for_natural_exit_and_cleanup(self, session: _TrainerSession) -> None:
        disconnected_since = None
        disconnect_grace = (
            self._context.heartbeat_timeout
            if self._context.heartbeat_timeout > 0
            else self._result_source_disconnect_grace()
        )
        try:
            while self._process_group_alive(session):
                now = time.monotonic()
                if self._cell.is_cell_connected(session.trainer_fqcn):
                    disconnected_since = None
                elif disconnected_since is None:
                    disconnected_since = now
                elif now - disconnected_since >= disconnect_grace:
                    # The trainer normally closes its Cell after the source transaction
                    # is terminal. Require a full session-lease interval before treating
                    # disconnect as terminal so a transient reconnect cannot corrupt a
                    # recoverable ViaDownloader transfer.
                    self._stop_session(session, natural_exit_wait=0.0)
                    return
                if self._closed:
                    if not session.result_source_live.is_set():
                        # A later per-task launch may already have replaced self._session,
                        # so finalize cannot discover every retired launch by that pointer.
                        # RESULT_SOURCE_LIVE=False proves send settled, but the user thread
                        # may still be returning from flare.send(). Give it normal/default
                        # process-exit grace before escalating to signals.
                        self._stop_session(session, natural_exit_wait=self._result_source_disconnect_grace())
                        return
                    # The trainer is still inside send(). SHUTDOWN is only an orderly
                    # intent here; send keeps serving until its receiver reaches a terminal
                    # outcome and then closes the trainer process itself.
                    self._request_session_shutdown(session, wait_timeout=_LIVE_RESULT_SHUTDOWN_ACK_TIMEOUT)
                time.sleep(_NATURAL_EXIT_REAP_INTERVAL)
            self._cleanup_session(session)
        except BaseException:
            self.logger.error(secure_format_traceback())
        finally:
            if not session._cleaned:
                # No exception in the sole ownership reaper may strand a process or its
                # launch token/bootstrap. _stop_session is idempotent and self-guarding.
                self._stop_session(session, natural_exit_wait=0.0)
            with self._result_reapers_lock:
                self._result_reapers.discard(session)

    def _cleanup_session(self, session: _TrainerSession) -> None:
        """Release launch-scoped state after the process group is gone. Idempotent."""
        with session._cleanup_lock:
            if session._cleaned:
                return
            session._cleaned = True
        try:
            log_thread = session.log_thread
            if log_thread is not None and log_thread.is_alive():
                log_thread.join(timeout=_LOG_THREAD_JOIN_TIMEOUT)
        except Exception:
            self.logger.error(secure_format_traceback())
        # Launch-scoped token invalidation: HELLO/RESULT_READY from a zombie of this
        # launch can never match again. Remove the launch's bootstrap file so the token
        # does not linger on disk. Process-group ownership remains in memory; no orphan
        # breadcrumb is emitted until there is an actual site-side consumer for one.
        session.token = ""
        session.session_id = None
        try:
            if session.bootstrap_path and os.path.exists(session.bootstrap_path):
                os.remove(session.bootstrap_path)
        except Exception as e:
            self.logger.debug(f"failed to remove {session.bootstrap_path}: {e}")
        with self._session_lock:
            if self._session is session:
                self._session = None

    def _disable_task_pass_through(self) -> None:
        cell = self._cell
        route = self._pass_through_route
        if cell is not None and route is not None and self._owns_pass_through_route:
            cell.decode_pass_through_topics.discard(route)
        self._pass_through_route = None
        self._owns_pass_through_route = False

    def _process_group_alive(self, session: _TrainerSession) -> bool:
        """True while any member of the launch's process group survives.

        The leader (launcher) exiting does not mean the tree is gone — a torchrun/mpirun
        launcher can die while its workers keep the group alive — so liveness keys off the
        GROUP where the platform lets us probe it, not off the Popen leader handle.
        """
        process = session.process
        if os.name != "posix" or session.pgid is None:
            # non-POSIX: only the launcher handle is observable (see the docstring's
            # Windows limitation); a dead leader is the best available "tree gone" signal
            return process is not None and process.poll() is None
        if process is not None:
            # reap the leader if it exited, so a zombie entry cannot hold the group open
            process.poll()
        try:
            os.killpg(session.pgid, 0)
            return True
        except ProcessLookupError:
            return False
        except Exception as e:
            # Cannot probe (for example due to permissions): assume alive so teardown
            # still attempts to signal the owned group rather than silently stranding it.
            self.logger.debug(f"cannot probe trainer process group {session.pgid}: {e}")
            return True

    def _session_liveness_error(self, session: _TrainerSession) -> Optional[str]:
        """Returns why an established session is unavailable, or None while it is live."""
        if not self._process_group_alive(session):
            rc = session.process.poll() if session.process else None
            return f"trainer process group exited (rc={rc})"
        heartbeat_timeout = self._context.heartbeat_timeout
        if heartbeat_timeout > 0 and session.ready.is_set():
            silent_for = session.peer_silent_for()
            if silent_for is not None and silent_for > heartbeat_timeout:
                return f"trainer heartbeat timed out after {silent_for:.1f}s " f"(timeout={heartbeat_timeout}s)"
        return None

    def _await_group_exit(self, session: _TrainerSession, timeout: float) -> bool:
        """Waits (bounded) for the whole process group to exit; reaps the leader."""
        deadline = time.monotonic() + timeout
        process = session.process
        while True:
            if process is not None and process.poll() is None:
                try:
                    process.wait(timeout=0.1)
                except subprocess.TimeoutExpired:
                    pass
            if not self._process_group_alive(session):
                return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.1)

    def _terminate_process_tree(self, session: _TrainerSession, grace: float) -> None:
        """SIGTERM to the process group, a bounded grace, then SIGKILL (design:
        "Process-tree termination"). Signaling keys off group liveness, not the leader:
        a launcher that already exited must not skip termination of surviving workers."""
        if not self._process_group_alive(session):
            return
        self.logger.info(f"terminating trainer process tree (pgid={session.pgid}, grace={grace}s)")
        self._signal_process_tree(session, hard=False)
        if self._await_group_exit(session, grace):
            return
        self.logger.warning(f"trainer process tree (pgid={session.pgid}) survived SIGTERM grace; killing")
        self._signal_process_tree(session, hard=True)
        if not self._await_group_exit(session, _LOG_THREAD_JOIN_TIMEOUT):
            self.logger.error(f"trainer process group (pgid={session.pgid}) did not die after SIGKILL")

    def _signal_process_tree(self, session: _TrainerSession, hard: bool) -> None:
        """Soft (SIGTERM/terminate) or hard (SIGKILL/kill) signal to the trainer's tree."""
        if os.name == "posix" and session.pgid is not None:
            try:
                os.killpg(session.pgid, signal.SIGKILL if hard else signal.SIGTERM)
                return
            except ProcessLookupError:
                return
            except Exception as e:
                self.logger.debug(f"failed to signal trainer process group: {e}")
        process = session.process
        if process is None or process.poll() is not None:
            return
        if os.name == "nt":
            # best available tree termination on Windows without a Job Object: /T covers
            # the tree at BOTH stages — a launcher-only terminate would let the launcher
            # exit while workers survive, and the leader handle is the only liveness
            # signal available here, so the tree would be inferred dead
            try:
                cmd = ["taskkill", "/T", "/PID", str(process.pid)]
                if hard:
                    cmd.insert(1, "/F")
                completed = subprocess.run(cmd, capture_output=True, timeout=10)
                if completed.returncode == 0 or process.poll() is not None:
                    return
                self.logger.debug(f"taskkill on trainer tree failed with rc={completed.returncode}")
            except Exception as e:
                self.logger.debug(f"taskkill on trainer tree failed: {e}")
        # last resort: the launcher process itself
        try:
            if hard:
                process.kill()
            else:
                process.terminate()
        except Exception as e:
            self.logger.debug(f"failed to signal trainer process: {e}")

    @staticmethod
    def _split_command(command: Union[str, Sequence[str]]) -> list[str]:
        """Prepare shell-free argv and resolve secret references after tokenization.

        Structured argv is already tokenized. Command strings retain the legacy platform
        behavior: Windows preserves backslashed paths and POSIX uses shell-style tokenization.
        Resolved secrets remain one argv item and are never reparsed as command syntax.
        """
        return prepare_subprocess_command(command, posix=os.name == "posix")

    def _shutdown_wait_bound(self) -> float:
        shutdown_timeout = self._context.shutdown_timeout
        return _DEFAULT_SHUTDOWN_TIMEOUT if shutdown_timeout is None else shutdown_timeout

    def _result_source_disconnect_grace(self) -> float:
        """Return a nonzero grace before treating source-Cell disconnect as terminal.

        ``ScriptRunner`` historically defaults ``shutdown_timeout`` to zero. That is a
        valid preference for ordinary process shutdown, but it cannot mean "kill on the
        first disconnected sample" while an accepted result can still be returning from
        send(). Use the backend default when no positive disconnect grace exists.
        """
        shutdown_bound = self._shutdown_wait_bound()
        return shutdown_bound if shutdown_bound > 0 else _DEFAULT_SHUTDOWN_TIMEOUT

    def _result_reaper_wait_bound(self) -> float:
        """Return the final END_RUN backstop for an accepted result source.

        Normal completion remains governed by the source transaction. This total bound is
        only the last defense against a connected trainer that remains live forever despite
        the transaction idle budget and orderly SHUTDOWN requests.
        """
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
            session = self._session
            if session is not None:
                self._stop_session(session, natural_exit_wait=0.0)
        except Exception:
            self.logger.error(secure_format_traceback())
        finally:
            self._disable_task_pass_through()

    # ------------------------------------------------------------------ task execution

    def _run_task(
        self, session: _TrainerSession, task_name: str, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal
    ) -> Shareable:
        context = self._context
        executor = context.executor
        launch_once = context.launch_once

        task = _TaskContext(task_id=uuid.uuid4().hex)
        # ``result_source_live`` is a conservative, session-monotonic send barrier. The
        # CJ may queue the next task after receiver confirmation but before asynchronous
        # source settlement lets the trainer return from the previous send(). Only a
        # synchronized SHUTDOWN ACK or process cleanup may clear it.

        shareable.set_header(FLMetaKey.JOB_ID, fl_ctx.get_job_id())
        shareable.set_header(FLMetaKey.SITE_NAME, fl_ctx.get_identity_name())

        # Publish correlation state before sending: Cell serializes the Shareable into this
        # request and materializes it at the trainer before TASK_ACCEPTED is returned.
        with self._task_lock:
            self._current_task = task
        try:
            task_message = {
                MsgKey.SESSION_ID: session.session_id,
                MsgKey.TASK_ID: task.task_id,
                MsgKey.TASK_NAME: task_name,
                MsgKey.MODEL: shareable,
            }
            executor.log_info(fl_ctx, f"sending TASK_READY for '{task_name}' to trainer {session.trainer_fqcn}")
            send_status, reply = self._send_task_ready(session, task_message, abort_signal)
            if send_status == _SEND_ABORTED:
                self._send_abort(session, f"'{task_name}' is aborted, abort_signal_triggered")
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

            # monotonic: a wall-clock step (NTP, VM resume) must not fire a spurious timeout
            result_wait_timeout = context.result_wait_timeout
            wait_start = time.monotonic()
            wait_deadline = None if result_wait_timeout is None else wait_start + result_wait_timeout
            executor.log_info(fl_ctx, "waiting for result from external trainer")
            while True:
                # the backend-level abort latch only concerns launch_once sessions (a
                # per-task launch is fresh state; a previous task's dead session must not
                # poison it)
                if abort_signal.triggered or (launch_once and self._abort):
                    self._send_abort(session, f"'{task_name}' is aborted, abort_signal_triggered")
                    # the wire ABORT ends the trainer's session (receive-side contract:
                    # flare.is_running() returns False), so the session is gone for good
                    self._latch_abort(f"task '{task_name}' aborted")
                    return make_reply(ReturnCode.TASK_ABORTED)

                if task.result_ready.is_set():
                    break

                liveness_error = self._session_liveness_error(session)
                if liveness_error:
                    self._send_abort(session, liveness_error)
                    self._latch_abort(liveness_error)
                    executor.log_error(fl_ctx, f"{liveness_error} before task '{task_name}' produced a result")
                    return make_reply(ReturnCode.EXECUTION_EXCEPTION)

                now = time.monotonic()
                if wait_deadline is not None and now >= wait_deadline:
                    self._send_abort(
                        session, f"'{task_name}' timed out after {result_wait_timeout}s waiting for result"
                    )
                    self._latch_abort(f"result wait timed out for task '{task_name}'")
                    executor.log_error(
                        fl_ctx, f"timed out after {result_wait_timeout}s waiting for '{task_name}' result"
                    )
                    return make_reply(ReturnCode.EXECUTION_EXCEPTION)

                wait_time = _RESULT_POLL_INTERVAL
                if wait_deadline is not None:
                    wait_time = min(wait_time, wait_deadline - now)
                # event-driven wake on result arrival; the timeout only bounds
                # process-death/abort detection latency
                task.result_ready.wait(wait_time)

            # RESULT_READY may carry lazy references. Return the Shareable unchanged so
            # ClientRunner retains its existing filter and forwarding semantics.
            with self._task_lock:
                result = task.result
            if not isinstance(result, Shareable):
                executor.log_error(fl_ctx, f"bad task result from trainer: expect Shareable but got {type(result)}")
                return make_reply(ReturnCode.EXECUTION_EXCEPTION)

            current_round = shareable.get_header(AppConstants.CURRENT_ROUND)
            if current_round is not None:
                # the round travels back on the result for workflow bookkeeping
                result.set_header(AppConstants.CURRENT_ROUND, current_round)
            return result
        finally:
            with self._task_lock:
                if self._current_task is task:
                    self._current_task = None

    def _send_task_ready(self, session: _TrainerSession, task_message: dict, abort_signal: Signal) -> Tuple[str, Any]:
        """Sends TASK_READY without hanging past abort, process death, or backend close.

        The request runs on a helper thread while this thread polls the liveness
        conditions the execute() contract demands. When one of them ends the wait, the
        request itself is cancelled through the streaming path's native ``abort_signal``
        (a dedicated cancel Signal, so a task abort maps onto it explicitly): the cell's
        request waiter is released instead of lingering until the request timeout, and
        the sender thread exits — it is joined boundedly and, only if something ignores
        the cancel, abandoned as a daemon whose late reply is inert (its holder is
        never read after a non-OK status).

        Returns:
            (_SEND_OK, reply) on a delivered request; (_SEND_ABORTED / _SEND_PROCESS_DEAD /
            _SEND_SESSION_DEAD / _SEND_CLOSED, reason) when the corresponding condition
            ended the wait. Only _SEND_SESSION_DEAD supplies a reason; the other non-OK
            outcomes return None as the second item.
        """
        timeout = self._context.task_wait_timeout
        if timeout is None:
            # the streaming request path requires a numeric (no-progress) timeout
            timeout = _TASK_READY_NO_PROGRESS_TIMEOUT
        cancel = Signal()
        holder = {}
        done = threading.Event()
        transactions = []

        def _on_transaction_created(transaction):
            transactions.append(transaction)

        def _has_live_task_download():
            # Cell calls this after a request-wait interval expires. Continue waiting
            # while the real task ViaDownloader transaction remains live; its own idle
            # timeout is the stall bound. Once it settles, a missing TASK_ACCEPTED is a
            # genuine request timeout.
            for transaction in tuple(transactions):
                try:
                    if not DownloadService.get_transfer_waiter(transaction.tx_id).done():
                        return True
                except Exception:
                    # A transaction deleted between snapshot and lookup is no longer live.
                    continue
            return False

        def _send():
            try:
                holder["reply"] = self._cell.send_request(
                    channel=CHANNEL,
                    topic=Topic.TASK_READY,
                    target=session.trainer_fqcn,
                    request=new_cell_message({}, task_message),
                    timeout=timeout,
                    abort_signal=cancel,
                    progress_wait_cb=_has_live_task_download,
                    receiver_ids=(session.trainer_fqcn,),
                    fobs_ctx_props={
                        FOBSContextKey.STREAM_PROGRESS_CB: lambda **_kwargs: None,
                        RESULT_UPLOAD_TX_CREATED_CB_CTX_KEY: _on_transaction_created,
                    },
                )
            except Exception as e:
                holder["error"] = e
            finally:
                done.set()

        sender = threading.Thread(target=_send, name="client_api_task_ready_send", daemon=True)
        sender.start()
        status = _SEND_OK
        while not done.wait(_RESULT_POLL_INTERVAL):
            if abort_signal.triggered or (self._context.launch_once and self._abort):
                status = _SEND_ABORTED
                break
            if self._closed:
                status = _SEND_CLOSED
                break
            if not self._process_group_alive(session):
                status = _SEND_PROCESS_DEAD
                break
            # The trainer callback runs only after Cell/FOBS materializes the task. During
            # an actual ViaDownloader transaction, that transaction's progress/idle policy
            # is the authoritative stall detector. Merely having TASK_READY pending is not
            # evidence of transfer progress: an inline request to a live-but-wedged Cell
            # must still be bounded by the heartbeat lease.
            if not _has_live_task_download():
                liveness_error = self._session_liveness_error(session)
                if liveness_error:
                    holder["liveness_error"] = liveness_error
                    status = _SEND_SESSION_DEAD
                    break
        if status != _SEND_OK:
            cancel.trigger(status)
            sender.join(timeout=_SENDER_CANCEL_JOIN_TIMEOUT)
            self._delete_transactions(transactions)
            return status, holder.get("liveness_error")
        error = holder.get("error")
        if error is not None:
            self._delete_transactions(transactions)
            raise error
        return _SEND_OK, holder.get("reply")

    @staticmethod
    def _delete_transactions(transactions) -> None:
        for transaction in transactions:
            try:
                DownloadService.delete_transaction(transaction.tx_id)
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
        # (it dies with the job), a later backend re-registering overwrites these entries,
        # and every state-mutating handler is gated on self._closed for the window in
        # between.
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
        session = self._session
        if session is None:
            return self._hello_reject(session, origin, "no active trainer launch", latch=False)

        # identity first: both the physical origin and the claimed identity must be the
        # FQCN this launch prescribed. A mismatch is a stale/foreign process — reject it
        # WITHOUT failing the current launch's HELLO wait.
        claimed_fqcn = payload.get(MsgKey.TRAINER_FQCN)
        if origin != session.trainer_fqcn or claimed_fqcn != session.trainer_fqcn:
            return self._hello_reject(
                session,
                origin,
                f"unexpected trainer identity (origin={origin!r}, claimed={claimed_fqcn!r})",
                latch=False,
            )

        # V1 external_process proof: the plain launch token carried as MsgKey.PROOF.
        # Compared as bytes: compare_digest raises TypeError on non-ASCII str input, and a
        # forged proof must yield a clean (latched) rejection, not a handler exception.
        proof = payload.get(MsgKey.PROOF)
        if (
            not isinstance(proof, str)
            or not session.token
            or not secrets.compare_digest(proof.encode("utf-8"), session.token.encode("utf-8"))
        ):
            return self._hello_reject(session, origin, "launch token mismatch", latch=True)

        if payload.get(MsgKey.PROTOCOL_VERSION) != PROTOCOL_VERSION:
            return self._hello_reject(
                session,
                origin,
                f"unsupported protocol version {payload.get(MsgKey.PROTOCOL_VERSION)!r} (expect {PROTOCOL_VERSION})",
                latch=True,
            )

        if payload.get(MsgKey.JOB_ID) != self._job_id:
            return self._hello_reject(session, origin, f"job id mismatch: {payload.get(MsgKey.JOB_ID)!r}", latch=True)

        if payload.get(MsgKey.SITE_NAME) != self._site_name:
            return self._hello_reject(
                session,
                origin,
                f"site name mismatch: {payload.get(MsgKey.SITE_NAME)!r}",
                latch=True,
            )

        # rank contract: only rank 0 connects; other ranks get the model via the training
        # framework's own collectives. Not latched: the real rank 0 may still HELLO.
        rank = payload.get(MsgKey.RANK)
        if str(rank) != "0":
            return self._hello_reject(session, origin, f"only rank 0 may connect (got rank {rank!r})", latch=False)

        # under the session lock: two concurrent HELLOs (dispatcher threads) must not both
        # mint a session id, or one reply would carry an id that instantly goes stale
        with self._session_lock:
            if not session.ready.is_set():
                session.session_id = uuid.uuid4().hex
                session.ready.set()
                self.logger.info(f"HELLO accepted from {origin} (session_id={session.session_id})")
            session.touch_peer_activity()
        # duplicate HELLO from the same authenticated trainer is idempotent: same session
        return self._protocol_reply(
            Topic.HELLO_ACCEPTED,
            **{
                MsgKey.SESSION_ID: session.session_id,
                MsgKey.JOB_ID: self._job_id,
                MsgKey.SITE_NAME: self._site_name,
                MsgKey.HEARTBEAT_INTERVAL: self._context.heartbeat_interval,
                MsgKey.HEARTBEAT_TIMEOUT: self._context.heartbeat_timeout,
            },
        )

    def _hello_reject(self, session: Optional[_TrainerSession], origin: str, reason: str, latch: bool):
        self.logger.warning(f"rejecting HELLO from {origin!r}: {reason}")
        if latch and session is not None and not session.ready.is_set() and session.reject_reason is None:
            # the launch's own process presented bad credentials: fail the launch wait fast
            session.reject_reason = reason
        return self._protocol_reply(Topic.HELLO_REJECTED, **{MsgKey.REASON: reason})

    def _validate_session_msg(self, request, payload) -> Tuple[Optional[_TrainerSession], Optional[str]]:
        """Binds a post-HELLO message to the current authenticated session."""
        session = self._session
        if session is None or not session.ready.is_set() or session.session_id is None:
            return None, "no active trainer session"
        origin = request.get_header(MessageHeaderKey.ORIGIN) or ""
        if origin != session.trainer_fqcn:
            return None, f"unexpected origin {origin!r}"
        if payload.get(MsgKey.SESSION_ID) != session.session_id:
            return None, "stale or unknown session id"
        session.touch_peer_activity()
        return session, None

    def _handle_result_ready(self, request):
        """Accept a result Shareable already decoded by Cell.

        PASS_THROUGH decoding can leave large values as lazy references here. They remain
        part of the Shareable returned to ClientRunner, matching the legacy subprocess path.

        RESULT_READY is currently sent once. Attach-mode redelivery tolerance should be
        added only as a sender-retry + receiver-dedup pair.
        """
        if self._closed:
            return self._protocol_reply(Topic.RESULT_REJECTED, **{MsgKey.REASON: "backend is closed"})
        payload = request.payload
        if not isinstance(payload, dict):
            return make_cell_reply(CellReturnCode.INVALID_REQUEST, error="RESULT_READY payload must be a dict")

        session, reject_reason = self._validate_session_msg(request, payload)
        if reject_reason:
            self.logger.warning(f"rejecting RESULT_READY: {reject_reason}")
            return self._protocol_reply(Topic.RESULT_REJECTED, **{MsgKey.REASON: reject_reason})

        task_id = payload.get(MsgKey.TASK_ID)
        result = payload.get(MsgKey.RESULT)
        # ONE critical section for current-task validation AND the acceptance commit. The
        # _run_task finally takes this same lock to clear the task, so if teardown wins the
        # task reads None below and the result is rejected instead of being accepted for a
        # retired task.
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
            # The RESULT_ACCEPTED reply itself can race END_RUN even for inline payloads.
            # Keep every accepted result behind the same send-completion barrier; the
            # synchronized SHUTDOWN reply clears this latch when send already settled.
            session.result_source_live.set()
            task.result_ready.set()
        return self._protocol_reply(Topic.RESULT_ACCEPTED)

    def _handle_log(self, request):
        """Routes trainer LOG data through the executor-owned fire_log_analytics().

        Contract: must not raise (a raise here would surface in the cell dispatcher, and
        the metric would vanish without a diagnostic)."""
        if self._closed:
            return None
        try:
            payload = request.payload
            if not isinstance(payload, dict):
                self.logger.error(f"invalid LOG data format, expecting Dict, but got {type(payload)}")
                return None
            session, reject_reason = self._validate_session_msg(request, payload)
            if reject_reason:
                # a zombie trainer's metrics must not land in the current job
                self.logger.warning(f"dropping LOG data: {reject_reason}")
                return None
            record = {k: v for k, v in payload.items() if k != MsgKey.SESSION_ID}
            if "key" in record:
                record["tag"] = record.pop("key")
            dxo = create_analytic_dxo(**record)
            # single analytics-event ownership point: the executor decides the fire path
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
        session, reject_reason = self._validate_session_msg(request, payload)
        if reject_reason:
            self.logger.warning(f"rejecting HEARTBEAT: {reject_reason}")
            return self._protocol_reply(Topic.ERROR, **{MsgKey.REASON: reject_reason})
        return self._protocol_reply(Topic.HEARTBEAT, **{MsgKey.SESSION_ID: session.session_id})

    # ------------------------------------------------------------------ helpers

    def _send_abort(self, session: Optional[_TrainerSession], reason: str) -> None:
        if session is None or session.session_id is None:
            return
        try:
            self._cell.fire_and_forget(
                channel=CHANNEL,
                topic=Topic.ABORT,
                targets=[session.trainer_fqcn],
                message=new_cell_message({}, {MsgKey.SESSION_ID: session.session_id, MsgKey.REASON: reason}),
                optional=True,
            )
        except Exception:
            self.logger.error(secure_format_traceback())

    def _latch_abort(self, reason: str) -> None:
        # keep the FIRST cause: later echoes must not mask it
        self._abort = True
        if self._abort_reason is None:
            self._abort_reason = reason

    def _task_exchange_config(self) -> dict:
        # Mirrors the in_process backend's TASK_EXCHANGE meta. The backend only transports
        # the declared representation contract; the trainer-side Client API performs the
        # receive/send adaptation and computes DIFF in the native exchange format.
        context = self._context
        return {
            ConfigKey.TRAIN_WITH_EVAL: context.train_with_evaluation,
            ConfigKey.EXCHANGE_FORMAT: context.params_exchange_format,
            ConfigKey.SERVER_EXPECTED_FORMAT: context.server_expected_format,
            ConfigKey.TRANSFER_TYPE: context.params_transfer_type,
            ConfigKey.TRAIN_TASK_NAME: context.train_task_name,
            ConfigKey.EVAL_TASK_NAME: context.evaluate_task_name,
            ConfigKey.SUBMIT_MODEL_TASK_NAME: context.submit_model_task_name,
            # Preserve the existing Client API lifecycle contract: a per-task trainer
            # closes its communication endpoint after its accepted result, while a
            # launch-once trainer remains connected for the next task.
            ConfigKey.LAUNCH_ONCE: context.launch_once,
        }
