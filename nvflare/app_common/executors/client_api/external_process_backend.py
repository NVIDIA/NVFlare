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
process tree, and talks to it over the CJ cell using the frozen protocol vocabulary in
nvflare/client/cell/defs.py — no PipeHandler, no CellPipe, no MetricRelay:

- **Launch**: writes the bootstrap config (0600; fresh launch-scoped token, connect URL,
  prescribed trainer FQCN — see nvflare/client/cell/bootstrap.py), starts the configured
  command in its own process group, and waits for the trainer's HELLO. The token is a
  plain match over the localhost connection NVFlare itself created (V1 trusted host);
  challenge-response proof is an attach-mode concern (design Appendix B).
- **Control plane**: TASK_READY/TASK_ACCEPTED, RESULT_READY/RESULT_ACCEPTED, TASK_FAILED,
  LOG, ABORT, SHUTDOWN, BYE as plain cell messages. There is deliberately NO session
  heartbeat protocol in this mode: liveness of a *transfer* is the payload layer's budget
  job, liveness of the *process* is the backend's (process handle; the trainer's cell
  connects directly to the CJ's internal listener, so process death also drops the
  connection). The heartbeat_interval/heartbeat_timeout knobs on the frozen executor
  surface are reserved for attach, which has no process handle.
- **Payload plane**: task payloads down and result payloads up move through the
  payload_transfer seam (nvflare/client/cell/payload_transfer.py, over the F3 payload
  layer's explicit DownloadService transactions). The backend mints
  the cross-attempt ``transfer_id``, carries it in the task message, and enforces
  one-live-attempt per logical transfer — a retry is a NEW attempt under a NEW tx_id, and
  creating a duplicate live attempt for a transfer_id raises. RESULT_ACCEPTED is a control
  ack, not payload completion: the result bytes are pulled through the seam after it, and
  a pull failure is confirmed to the trainer-side producer as FAILED by the layer itself.
- **Teardown**: orderly stop is SHUTDOWN, a bounded wait for natural exit
  (shutdown_timeout), then SIGTERM -> stop_grace_period -> SIGKILL to the process GROUP,
  keyed off group liveness rather than the launcher handle (a torchrun/mpirun launcher can
  exit while its workers keep the group alive). The pgid breadcrumb written at launch is
  removed only once the group is confirmed gone; its CP-side consumer (the orphan reaper
  of the design's "CJ Failure" section) is a follow-up PR. Non-POSIX limitation: there is
  no group-liveness probe, so tree death is inferred from the launcher handle; both stop
  stages therefore use ``taskkill /T`` on Windows (soft, then ``/F``) so a launcher-only
  terminate cannot strand workers — full tree ownership there needs the Job-Object work
  the design assigns to that platform. The trainer is never stopped while a pending payload
  attempt could still settle: live attempts get a bounded waiter wait (the waiter's None
  arm is treated as not-delivered) before termination.

Like the in_process backend, LOG data is routed through the executor-owned
fire_log_analytics(); this backend selects the federation-scoped fire path
(set_analytics_fire_fed_event(True)) for parity with MetricRelay's ex-process behavior.

V1 supports ONE external_process backend per client job: the protocol topics, prescribed
trainer FQCN leaves, and launch artifacts are all job-scoped, so a second
ClientAPIExecutor(execution_mode="external_process") in the same job would silently
cross-wire both trainers. A second backend is rejected deterministically at START_RUN
(system_panic) instead; jobs route all their tasks through a single executor.

This backend is the CJ side of the protocol; its counterpart is the trainer-side Cell
engine (nvflare/client/cell/api.py), which reads the bootstrap config, HELLOs, and serves
tasks to user code.
"""

import os
import secrets
import shlex
import signal
import subprocess
import threading
import time
import uuid
from collections import OrderedDict
from typing import Any, Optional, Tuple

from nvflare.apis.fl_constant import FLContextKey, FLMetaKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.fl_exception import UnsafeJobError
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.apis.utils.analytix_utils import create_analytic_dxo
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.executors.client_api.backend_spec import ClientAPIBackendContext, ClientAPIBackendSpec
from nvflare.app_common.executors.client_api.single_backend import SingleBackendGuard
from nvflare.app_common.launchers.subprocess_launcher import log_subprocess_output
from nvflare.client.api_spec import CLIENT_API_TYPE_KEY
from nvflare.client.cell.bootstrap import BOOTSTRAP_FILE_ENV_VAR, CELL_API_TYPE, BootstrapKey, write_bootstrap_config
from nvflare.client.cell.decomposers import register_framework_decomposers
from nvflare.client.cell.defs import CHANNEL, PROTOCOL_VERSION, MsgKey, Topic
from nvflare.client.cell.payload_transfer import PayloadTransferError, TaskPayloadAttempt, fetch_result_payload
from nvflare.client.config import ConfigKey, ExchangeFormat, TransferType
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey
from nvflare.fuel.f3.cellnet.defs import ReturnCode as CellReturnCode
from nvflare.fuel.f3.cellnet.fqcn import FQCN
from nvflare.fuel.f3.cellnet.utils import make_reply as make_cell_reply
from nvflare.fuel.f3.cellnet.utils import new_cell_message
from nvflare.fuel.utils.log_utils import get_obj_logger
from nvflare.security.logging import secure_format_traceback
from nvflare.utils.job_launcher_utils import add_custom_dir_to_path

# Poll cadence of the result-wait loop and the HELLO wait (result arrival wakes the loop
# early through the task's event; the poll only bounds process-death detection latency).
_RESULT_POLL_INTERVAL = 0.5
_HELLO_POLL_INTERVAL = 0.1

# Orderly-stop bound applied when the frozen surface's shutdown_timeout is None
# ("None means the backend default" per the executor docstring).
_DEFAULT_SHUTDOWN_TIMEOUT = 30.0

# Bounded wait for a still-live task payload attempt to settle before it is terminated
# (retire path / teardown). The waiter's None arm folds into "not delivered".
_ATTEMPT_SETTLE_WAIT = 5.0

# Bound for joining the subprocess-stdout log thread after the process is gone.
_LOG_THREAD_JOIN_TIMEOUT = 5.0

# The prescribed FQCN leaf of the trainer's child cell: <cj_fqcn>.<prefix>_<launch_seq>.
# A fresh leaf per launch means a stale process's cell can never collide with (or receive
# messages meant for) the current launch's cell.
_TRAINER_LEAF_PREFIX = "client_api_trainer"

# How many recently-accepted result receipts to retain for idempotent RESULT_READY retries.
# Task ids are sequential per session; a retry arrives shortly after its original, so a small
# ring covers the lost-reply window without unbounded growth over a long launch_once job.
_RESULT_RECEIPT_RING = 32

# One live external_process backend per CJ cell (V1): the protocol topics, trainer FQCN
# leaves, and launch artifacts are job-scoped, so a second backend on the same cell would
# overwrite the first's handlers and launch namespace. The shared guard mechanism is also
# used by the in_process backend (keyed on the DataBus); see single_backend.py.
_GUARD = SingleBackendGuard(
    mode="external_process",
    remedy="configure a single ClientAPIExecutor for all of its tasks (its protocol topics and "
    "launch artifacts are job-scoped)",
)


def pgid_file_name(seq: int) -> str:
    """The launch-scoped process-group breadcrumb name, written into the job's app dir.

    Lets the client CP (or manual cleanup) reap an orphaned trainer tree if the CJ dies
    (design: "CJ Failure (Owner Death)"). Launch-scoped so a group that survives SIGKILL
    keeps ITS breadcrumb even after later launches write theirs — a shared fixed name
    would lose exactly the recovery information a surviving group needs. Removed once the
    backend itself confirms the group stopped.
    """
    return f"client_api_trainer_{seq}.pgid"


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


class _TrainerSession:
    """One launched trainer process and its (at most one) authenticated protocol session."""

    def __init__(self, seq: int, token: str, trainer_fqcn: str):
        self.seq = seq
        self.token = token
        self.trainer_fqcn = trainer_fqcn
        self.session_id: Optional[str] = None
        self.ready = threading.Event()
        # latched when the HELLO of THIS launch's own process is rejected, so the launch
        # wait fails fast instead of waiting out launch_timeout
        self.reject_reason: Optional[str] = None
        self.bootstrap_path: Optional[str] = None
        self.pgid_path: Optional[str] = None
        self.process: Optional[subprocess.Popen] = None
        # POSIX process-group id, retained independently of the Popen leader handle so the
        # group can be probed/terminated even after the launcher itself exited
        self.pgid: Optional[int] = None
        self.log_thread: Optional[threading.Thread] = None

    def process_alive(self) -> bool:
        process = self.process
        return process is not None and process.poll() is None


class _TaskContext:
    """Correlation state for the one task execute() is currently running."""

    def __init__(self, task_id: str, task_name: str, transfer_id: str):
        self.task_id = task_id
        self.task_name = task_name
        self.transfer_id = transfer_id
        # No per-task lock: all of a task's result/failure fields, the _current_task
        # pointer, and the receipt ring are guarded by the backend's single _task_lock, so
        # current-task validation and the acceptance/failure commit are one critical section
        # (they cannot interleave with teardown's clear).
        self.result_ready = threading.Event()
        self.result_id: Optional[str] = None
        self.result_transfer_id: Optional[str] = None
        self.result_ref_ids: Optional[list] = None
        self.failed = threading.Event()
        self.failure_reason: Optional[str] = None


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
        self._job_id: Optional[str] = None
        self._site_name: Optional[str] = None
        self._app_dir: Optional[str] = None
        self._custom_dir: Optional[str] = None
        self._session: Optional[_TrainerSession] = None
        self._session_lock = threading.Lock()
        self._launch_seq = 0
        self._current_task: Optional[_TaskContext] = None
        self._task_lock = threading.Lock()
        # admission gate: one active execute() at a time (one active task per session).
        # try-acquired at execute() entry, released in its finally.
        self._execute_gate = threading.Lock()
        # Bounded receipts of accepted results (task_id -> result_id), so a RESULT_READY
        # retried after execute() cleared _current_task (a lost RESULT_ACCEPTED reply) is
        # still answered idempotently — the design's retry rule — instead of wrongly
        # rejected as "no current task". Guarded by _task_lock; oldest evicted past the ring.
        self._result_receipts: "OrderedDict[str, str]" = OrderedDict()
        # transfer_id -> live TaskPayloadAttempt: the one-live-attempt-per-transfer registry
        self._live_attempts = {}
        self._attempts_lock = threading.Lock()
        self._abort = False
        self._abort_reason: Optional[str] = None
        self._finalized = False
        # Gates every state-mutating cell callback after finalize()/unwind. Cellnet has no
        # unregister for request callbacks; the CJ cell is job-scoped so the registrations
        # die with it, and a later backend on the same cell would overwrite them — the gate
        # covers the window in between (a late message from an abandoned trainer is politely
        # rejected). The stateless BYE / TASK_PAYLOAD_READY acks are the ungated handlers.
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

            workspace = fl_ctx.get_prop(FLContextKey.WORKSPACE_OBJECT)
            job_id = fl_ctx.get_prop(FLContextKey.CURRENT_JOB_ID)
            if workspace is None or not job_id:
                raise RuntimeError("workspace/job id not available in fl_ctx")
            self._job_id = job_id
            self._site_name = fl_ctx.get_identity_name()
            self._app_dir = workspace.get_app_dir(job_id)
            self._custom_dir = workspace.get_app_custom_dir(job_id)

            # the trainer's child cell connects here (localhost listener owned by the CJ cell);
            # One external_process backend per CJ cell (V1). The protocol topics, the
            # prescribed trainer FQCN leaves, and the launch artifacts (bootstrap/pgid
            # files) are all job-scoped: a second backend would silently overwrite the
            # first's cell handlers and collide on its launch namespace, cross-wiring
            # both trainers. Reject deterministically at START_RUN (-> system_panic)
            # instead of hanging cross-wired at the first task.
            self._claim_cell(cell)

            # idempotent if the cell already has one
            cell.make_internal_listener()
            connect_url = cell.get_internal_listener_url()
            if not connect_url:
                raise RuntimeError("CJ cell has no internal listener url for the trainer to connect to")
            self._connect_url = connect_url

            self._register_protocol_cbs(cell)
            # register framework tensor decomposers in the CJ so a client-edge conversion
            # filter that produced framework-native tensors (e.g. torch) can be serialized to
            # the trainer; the trainer-side engine registers the same. Opportunistic: a
            # framework that is not installed is skipped, and numpy/FLModel need nothing extra.
            register_framework_decomposers(self.logger)

            # ex-process analytics parity with MetricRelay (fed_event=True): fire the
            # federation-scoped event directly. ConvertToFedEvent (if configured) only
            # re-fires LOCAL events, so this cannot double-deliver metrics.
            context.executor.set_analytics_fire_fed_event(True)

            if context.launch_once:
                self._start_session(timeout=context.launch_timeout)
        except BaseException:
            # contract (backend_spec): initialize() self-unwinds its partial setup on failure;
            # the executor does not call finalize() on a half-initialized backend. BaseException,
            # not Exception: a KeyboardInterrupt/SystemExit between the cell claim and here must
            # also release the guard slot and any launch artifacts already created.
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
            aborted_session = self._session
            self._send_abort(aborted_session, f"'{task_name}' is aborted, abort_signal_triggered")
            if aborted_session is not None:
                # the wire ABORT ends the trainer session; launch_once tasks after this
                # fail fast (parity with the in_process backend's abort-event latch)
                self._latch_abort(f"'{task_name}' aborted at entry, abort_signal_triggered")
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
                if not session.process_alive():
                    self._latch_abort("trainer process exited unexpectedly")
                    executor.log_error(fl_ctx, f"trainer process is not running; failing task '{task_name}'")
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
                    # per-task lifecycle: this launch's process stops with its task. The stop
                    # waits for pending payload attempts and natural exit before terminating
                    # the group, which is what replaces the legacy deferred-stop machinery.
                    stale = self._session
                    if stale is not None:
                        self._stop_session(stale, natural_exit_wait=self._shutdown_wait_bound())
            except Exception:
                self.logger.error(secure_format_traceback())
            finally:
                self._execute_gate.release()

    def handle_event(self, event_type: str, fl_ctx: FLContext) -> None:
        # no per-event behavior for external_process (START_RUN/END_RUN map to
        # initialize/finalize); contract: must not raise
        pass

    def finalize(self, fl_ctx: FLContext) -> None:
        # contract: idempotent and must not raise
        if self._finalized:
            return
        self._finalized = True
        self._closed = True
        # Barrier on any in-flight execute(): with _closed set, a mid-launch execute bails at
        # the pre-Popen check and releases the gate, so acquiring it here means "no execute is
        # between session install and Popen". This prevents a trainer being launched after we
        # return, and lets a per-task execute finish its own unwind (removing its launch
        # artifacts) before we release the cell to a successor job. Bounded: a long RUNNING
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
                self._stop_session(session, natural_exit_wait=self._shutdown_wait_bound())
        except Exception:
            self.logger.error(secure_format_traceback())
        finally:
            if admitted:
                self._execute_gate.release()
        self._sweep_live_attempts()
        self._release_cell()

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
            session = _TrainerSession(seq, token, trainer_fqcn)
            # install before launch so the HELLO handler resolves the current launch
            self._session = session
        try:
            bootstrap_path = os.path.join(self._app_dir, bootstrap_file_name(seq))
            session.bootstrap_path = bootstrap_path
            write_bootstrap_config(
                bootstrap_path,
                {
                    BootstrapKey.CONNECT_URL: self._connect_url,
                    BootstrapKey.CJ_FQCN: self._cj_fqcn,
                    BootstrapKey.TRAINER_FQCN: trainer_fqcn,
                    BootstrapKey.LAUNCH_TOKEN: token,
                    BootstrapKey.PROTOCOL_VERSION: PROTOCOL_VERSION,
                    BootstrapKey.JOB_ID: self._job_id,
                    BootstrapKey.SITE_NAME: self._site_name,
                    BootstrapKey.TASK_EXCHANGE: self._task_exchange_config(),
                    BootstrapKey.MEMORY_GC_ROUNDS: self._context.memory_gc_rounds,
                    BootstrapKey.CUDA_EMPTY_CACHE: self._context.cuda_empty_cache,
                },
            )

            env = os.environ.copy()
            env[BOOTSTRAP_FILE_ENV_VAR] = bootstrap_path
            # route the launched trainer's flare.init() to the Cell engine (CellClientAPI),
            # not the legacy CellPipe/FlareAgent ex-process stack
            env[CLIENT_API_TYPE_KEY] = CELL_API_TYPE
            add_custom_dir_to_path(self._custom_dir, env)

            # Last gate before spawning: finalize() may have set _closed after this session
            # was installed (above) but before this Popen. Bail now so no trainer process is
            # ever started after END_RUN teardown began. finalize() additionally barriers on
            # the execute gate, so it will not return until this unwind completes.
            if self._closed:
                raise RuntimeError("backend closed before trainer launch")

            self.logger.info(f"launching external trainer (launch {seq}): {self._context.command}")
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
            self._write_pgid_file(session)
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
            if not session.process_alive():
                rc = session.process.poll() if session.process else None
                raise RuntimeError(f"trainer process exited (rc={rc}) before completing the HELLO handshake")
            if deadline is not None and time.monotonic() >= deadline:
                raise RuntimeError(f"trainer did not complete the HELLO handshake within launch_timeout={timeout}s")

    def _stop_session(self, session: _TrainerSession, natural_exit_wait: float) -> None:
        """Stops one trainer session and its process tree. Must not raise; idempotent.

        Order: hold for pending payload attempts (bounded), SHUTDOWN, bounded natural-exit
        wait, then SIGTERM -> stop_grace_period -> SIGKILL to the process group. The token
        is invalidated so a surviving process can never authenticate against a later launch.
        """
        self._sweep_live_attempts()
        try:
            if session.session_id is not None and session.process_alive():
                self._cell.fire_and_forget(
                    channel=CHANNEL,
                    topic=Topic.SHUTDOWN,
                    targets=[session.trainer_fqcn],
                    message=new_cell_message(
                        {}, {MsgKey.SESSION_ID: session.session_id, MsgKey.REASON: "shutdown requested"}
                    ),
                    optional=True,
                )
        except Exception:
            self.logger.error(secure_format_traceback())
        try:
            process = session.process
            if process is not None and natural_exit_wait > 0:
                try:
                    process.wait(timeout=natural_exit_wait)
                except subprocess.TimeoutExpired:
                    pass
        except Exception:
            self.logger.error(secure_format_traceback())
        try:
            self._terminate_process_tree(session, grace=self._context.stop_grace_period)
        except Exception:
            self.logger.error(secure_format_traceback())
        try:
            log_thread = session.log_thread
            if log_thread is not None and log_thread.is_alive():
                log_thread.join(timeout=_LOG_THREAD_JOIN_TIMEOUT)
        except Exception:
            self.logger.error(secure_format_traceback())
        # launch-scoped token invalidation: HELLO/RESULT_READY from a zombie of this
        # launch can never match again. Remove the launch's bootstrap file (the token must
        # not linger on disk); the pgid breadcrumb goes only once the GROUP is confirmed
        # gone — a surviving tree keeps its breadcrumb so the CP reaper can still find it.
        session.token = ""
        session.session_id = None
        stale_files = [session.bootstrap_path]
        if not self._process_group_alive(session):
            stale_files.append(session.pgid_path)
        else:
            self.logger.warning(
                f"trainer process group (pgid={session.pgid}) may still be alive; keeping the pgid breadcrumb"
            )
        for stale_file in stale_files:
            try:
                if stale_file and os.path.exists(stale_file):
                    os.remove(stale_file)
            except Exception as e:
                self.logger.debug(f"failed to remove {stale_file}: {e}")
        with self._session_lock:
            if self._session is session:
                self._session = None

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
            # cannot probe (e.g. permissions): assume alive — erring this way keeps the
            # pgid breadcrumb for the CP reaper instead of silently dropping it
            self.logger.debug(f"cannot probe trainer process group {session.pgid}: {e}")
            return True

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
            # the breadcrumb is deliberately kept in this case (see _stop_session)
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
                subprocess.run(cmd, capture_output=True, timeout=10)
                return
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

    def _write_pgid_file(self, session: _TrainerSession) -> None:
        # Breadcrumb for orphan reaping if the CJ dies without running finalize (design:
        # "CJ Failure (Owner Death)"). NOTE: the CP-side reaper that consumes these files
        # is a follow-up PR of that design section — until it lands, the breadcrumb is
        # forward-looking (and still useful for manual cleanup/diagnosis).
        try:
            path = os.path.join(self._app_dir, pgid_file_name(session.seq))
            session.pgid_path = path
            with open(path, "w") as f:
                f.write(str(session.pgid if session.pgid is not None else session.process.pid))
        except Exception as e:
            self.logger.debug(f"failed to record trainer pgid: {e}")

    def _claim_cell(self, cell) -> None:
        """Claims the one-backend-per-cell slot; raises if another backend holds it.

        The slot is held from claim until _release_cell() (at the END of finalize()/
        _unwind()), NOT merely until _closed is set: a backend mid-teardown is still
        stopping its trainer and still owns launch seq 1's FQCN leaf, bootstrap/pgid
        artifacts, and registered handlers, so a second backend claiming that window would
        collide on all of them. See SingleBackendGuard for the shared rationale.
        """
        _GUARD.claim(cell, self)

    def _release_cell(self) -> None:
        """Releases the one-backend-per-cell slot if this backend holds it. Must not raise."""
        try:
            _GUARD.release(self._cell, self)
        except Exception:
            self.logger.error(secure_format_traceback())

    @staticmethod
    def _split_command(command: str):
        """POSIX: shlex argv. Windows: pass the string through — CreateProcess parses the
        command line natively, and POSIX shlex rules would corrupt backslashed paths
        (``python C:\\work\\train.py`` -> ``['python', 'C:worktrain.py']``)."""
        if os.name == "posix":
            return shlex.split(command)
        return command

    def _shutdown_wait_bound(self) -> float:
        shutdown_timeout = self._context.shutdown_timeout
        return _DEFAULT_SHUTDOWN_TIMEOUT if shutdown_timeout is None else shutdown_timeout

    def _unwind(self) -> None:
        """Releases partial setup after a failed initialize(). Best-effort per step."""
        self._closed = True
        try:
            session = self._session
            if session is not None:
                self._stop_session(session, natural_exit_wait=0.0)
        except Exception:
            self.logger.error(secure_format_traceback())
        self._sweep_live_attempts()
        # only releases a slot THIS backend claimed: a second backend whose claim was
        # rejected must not evict the first backend's claim on its way out
        self._release_cell()

    # ------------------------------------------------------------------ task execution

    def _run_task(
        self, session: _TrainerSession, task_name: str, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal
    ) -> Shareable:
        context = self._context
        executor = context.executor
        launch_once = context.launch_once

        task = _TaskContext(task_id=uuid.uuid4().hex, task_name=task_name, transfer_id=uuid.uuid4().hex)

        shareable.set_header(FLMetaKey.JOB_ID, fl_ctx.get_job_id())
        shareable.set_header(FLMetaKey.SITE_NAME, fl_ctx.get_identity_name())

        # task payload attempt: producer side of the payload seam; one-live-attempt per
        # transfer_id is enforced in _create_task_attempt (duplicates raise)
        attempt = self._create_task_attempt(task.transfer_id, shareable, session.trainer_fqcn)
        # publish the task for the RESULT_READY/TASK_FAILED handlers only once the attempt
        # exists, so a raising attempt creation cannot leak a half-registered task
        with self._task_lock:
            self._current_task = task
        try:
            task_message = {
                MsgKey.SESSION_ID: session.session_id,
                MsgKey.TASK_ID: task.task_id,
                MsgKey.TASK_NAME: task_name,
                MsgKey.TRANSFER_ID: task.transfer_id,
                MsgKey.MODEL: {MsgKey.REF_IDS: [attempt.ref_id]},
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

                if task.failed.is_set():
                    executor.log_error(fl_ctx, f"trainer failed task '{task_name}': {task.failure_reason}")
                    return make_reply(ReturnCode.EXECUTION_EXCEPTION)

                if attempt.failed():
                    # decisions come from the settled outcome, never from progress events
                    executor.log_error(
                        fl_ctx, f"task payload delivery failed for '{task_name}': {attempt.failure_reason()}"
                    )
                    return make_reply(ReturnCode.EXECUTION_EXCEPTION)

                if not session.process_alive():
                    rc = session.process.poll() if session.process else None
                    reason = f"trainer process exited (rc={rc}) before producing a result"
                    self._latch_abort(reason)
                    executor.log_error(fl_ctx, f"{reason} for task '{task_name}'")
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

            # RESULT_ACCEPTED (already sent by the handler) was a control ack only — the
            # payload moves here, and failure is confirmed to the producer by the layer.
            # result_ready is only set by the handler after it committed these fields under
            # _task_lock, so this read (same lock) sees them fully.
            with self._task_lock:
                result_ref_ids = list(task.result_ref_ids)
                result_transfer_id = task.result_transfer_id
            executor.log_info(fl_ctx, f"pulling result payload for '{task_name}' (transfer_id={result_transfer_id})")
            try:
                objs = fetch_result_payload(self._cell, session.trainer_fqcn, result_ref_ids, abort_signal)
            except PayloadTransferError as e:
                if abort_signal.triggered:
                    # the pull observed the abort (download_object fails the pull on a
                    # triggered signal): report the abort accurately and end the session,
                    # not a generic failure a workflow might retry against a dead session
                    self._send_abort(session, f"'{task_name}' is aborted, abort_signal_triggered")
                    self._latch_abort(f"task '{task_name}' aborted during result pull")
                    executor.log_info(fl_ctx, f"result pull for '{task_name}' aborted: {e}")
                    return make_reply(ReturnCode.TASK_ABORTED)
                executor.log_error(fl_ctx, f"result payload pull failed for '{task_name}': {e}")
                return make_reply(ReturnCode.EXECUTION_EXCEPTION)

            result = objs[0] if len(objs) == 1 else None
            if not isinstance(result, Shareable):
                executor.log_error(
                    fl_ctx, f"bad task result from trainer: expect one Shareable but got {[type(o) for o in objs]}"
                )
                return make_reply(ReturnCode.EXECUTION_EXCEPTION)

            current_round = shareable.get_header(AppConstants.CURRENT_ROUND)
            if current_round is not None:
                # the round travels back on the result for workflow bookkeeping
                result.set_header(AppConstants.CURRENT_ROUND, current_round)
            return result
        finally:
            self._retire_task_attempt(task.transfer_id, attempt)
            with self._task_lock:
                if self._current_task is task:
                    self._current_task = None
                # atomic with the clear: a RESULT_READY retried after this point finds the
                # receipt (below) instead of a wrongful "no current task"; before this point
                # it still matches the current task. Only an accepted result leaves a receipt.
                if task.result_id is not None:
                    self._record_result_receipt_locked(task.task_id, task.result_id)

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
            _SEND_CLOSED, None) when the corresponding condition ended the wait.
        """
        timeout = self._context.task_wait_timeout
        if timeout is None:
            # the streaming request path requires a numeric (no-progress) timeout
            timeout = _TASK_READY_NO_PROGRESS_TIMEOUT
        cancel = Signal()
        holder = {}
        done = threading.Event()

        def _send():
            try:
                holder["reply"] = self._cell.send_request(
                    channel=CHANNEL,
                    topic=Topic.TASK_READY,
                    target=session.trainer_fqcn,
                    request=new_cell_message({}, task_message),
                    timeout=timeout,
                    abort_signal=cancel,
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
            if not session.process_alive():
                status = _SEND_PROCESS_DEAD
                break
        if status != _SEND_OK:
            cancel.trigger(status)
            sender.join(timeout=_SENDER_CANCEL_JOIN_TIMEOUT)
            return status, None
        error = holder.get("error")
        if error is not None:
            raise error
        return _SEND_OK, holder.get("reply")

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

    # ------------------------------------------------------------------ payload attempts

    def _create_task_attempt(self, transfer_id: str, obj, receiver_fqcn: str) -> TaskPayloadAttempt:
        """Creates the (single) live payload attempt for a logical transfer.

        Contract: one live attempt per logical transfer is the BACKEND's responsibility. A
        retry must terminate the old attempt first and is a NEW attempt under a NEW
        (attempt-scoped, never reused) tx_id; a duplicate live attempt for the same
        transfer_id is a protocol bug and raises.
        """
        with self._attempts_lock:
            if self._closed:
                raise RuntimeError("backend is closed; not creating a payload attempt")
            if transfer_id in self._live_attempts:
                raise ValueError(
                    f"duplicate live payload attempt for transfer_id {transfer_id}: terminate the live "
                    f"attempt before retrying (a retry is a new attempt under a new tx_id)"
                )
            # reserve under the lock; create outside it (attempt creation talks to the service)
            self._live_attempts[transfer_id] = None
        try:
            attempt = TaskPayloadAttempt(self._cell, obj, receiver_fqcn)
        except Exception:
            with self._attempts_lock:
                self._live_attempts.pop(transfer_id, None)
            raise
        with self._attempts_lock:
            # a teardown sweep may have run between the reservation and here (it clears
            # the registry): the fresh attempt must not resurrect after END_RUN — publish
            # only if the reservation is still ours and the backend is still open
            if not self._closed and transfer_id in self._live_attempts:
                self._live_attempts[transfer_id] = attempt
                return attempt
        attempt.terminate()
        raise RuntimeError(f"backend closed while creating the payload attempt for transfer_id {transfer_id}")

    def _retire_task_attempt(self, transfer_id: str, attempt: TaskPayloadAttempt) -> None:
        """Settles or terminates the task's payload attempt at task end. Must not raise."""
        try:
            if not attempt.completed():
                # bounded, event-driven wait for the verdict; the waiter's None arm
                # (timeout or service shutdown) folds into delivered=False
                delivered = attempt.wait(timeout=_ATTEMPT_SETTLE_WAIT)
                if not delivered:
                    self.logger.warning(
                        f"task payload attempt {attempt.tx_id} (transfer_id={transfer_id}) was not certified "
                        f"delivered; terminating the attempt"
                    )
                    attempt.terminate()
        except Exception:
            self.logger.error(secure_format_traceback())
        finally:
            with self._attempts_lock:
                self._live_attempts.pop(transfer_id, None)

    def _sweep_live_attempts(self) -> None:
        """Terminates any attempts still live (teardown paths). Must not raise."""
        with self._attempts_lock:
            leaked = [(tid, attempt) for tid, attempt in self._live_attempts.items() if attempt is not None]
            self._live_attempts.clear()
        for transfer_id, attempt in leaked:
            try:
                if not attempt.completed():
                    if not attempt.wait(timeout=_ATTEMPT_SETTLE_WAIT):
                        self.logger.warning(
                            f"terminating still-live payload attempt {attempt.tx_id} (transfer_id={transfer_id})"
                        )
                attempt.terminate()
            except Exception:
                self.logger.error(secure_format_traceback())

    # ------------------------------------------------------------------ control-plane handlers

    def _register_protocol_cbs(self, cell) -> None:
        # NOTE: cellnet request callbacks cannot be unregistered. The CJ cell is job-scoped
        # (it dies with the job), a later backend re-registering overwrites these entries,
        # and every state-mutating handler is gated on self._closed for the window in
        # between (the stateless BYE / TASK_PAYLOAD_READY acks carry no gate).
        cell.register_request_cb(channel=CHANNEL, topic=Topic.HELLO, cb=self._handle_hello)
        cell.register_request_cb(channel=CHANNEL, topic=Topic.RESULT_READY, cb=self._handle_result_ready)
        cell.register_request_cb(channel=CHANNEL, topic=Topic.TASK_FAILED, cb=self._handle_task_failed)
        cell.register_request_cb(channel=CHANNEL, topic=Topic.LOG, cb=self._handle_log)
        cell.register_request_cb(channel=CHANNEL, topic=Topic.BYE, cb=self._handle_bye)
        # Topic.TASK_PAYLOAD_READY: the trainer's materialization-complete signal.
        # This backend has no heartbeat lease, so there is no materialization-phase
        # exemption to end — but the trainer engine sends it and must get a clean ack, not
        # a no-handler error.
        cell.register_request_cb(channel=CHANNEL, topic=Topic.TASK_PAYLOAD_READY, cb=self._handle_task_payload_ready)

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

        # V1 external_process proof: the plain launch token, carried as MsgKey.PROOF
        # (the token IS the proof on the trusted host; challenge-response is attach's).
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
        # duplicate HELLO from the same authenticated trainer is idempotent: same session
        return self._protocol_reply(
            Topic.HELLO_ACCEPTED,
            **{MsgKey.SESSION_ID: session.session_id, MsgKey.JOB_ID: self._job_id, MsgKey.SITE_NAME: self._site_name},
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
        return session, None

    def _handle_result_ready(self, request):
        """RESULT_READY -> RESULT_ACCEPTED is a control ack ONLY: it does not mean the
        payload transferred (the pull happens on the execute() thread afterwards)."""
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
        result_id = payload.get(MsgKey.RESULT_ID)
        transfer_id = payload.get(MsgKey.TRANSFER_ID)
        manifest = payload.get(MsgKey.MANIFEST)
        ref_ids = manifest.get(MsgKey.REF_IDS) if isinstance(manifest, dict) else None
        # ONE critical section for current-task validation AND the acceptance commit: a
        # split (validate here, commit under a different lock) lets teardown clear the task
        # between the two and record no receipt, so the handler would return RESULT_ACCEPTED
        # for a retired task the CJ never pulls — and the retry would then be rejected. The
        # _run_task finally (clear + conditional receipt) takes this same lock, so the two
        # are mutually exclusive: if teardown wins, the task reads None below and we reject.
        with self._task_lock:
            task = self._current_task
            if task is None or task_id != task.task_id:
                # Not the current task. It may be a retry of a result already accepted (and
                # consumed) for a now-cleared task — a lost RESULT_ACCEPTED reply. Answer
                # idempotently from the receipt ring instead of splitting the persistent
                # trainer from the CJ with a wrongful "no current task" rejection.
                accepted_result = self._result_receipts.get(task_id)
                if accepted_result is not None and accepted_result == result_id:
                    return self._protocol_reply(Topic.RESULT_ACCEPTED, **{MsgKey.RESULT_ID: result_id})
                if accepted_result is not None:
                    return self._protocol_reply(
                        Topic.RESULT_REJECTED,
                        **{MsgKey.REASON: f"a result ({accepted_result}) was already accepted for task {task_id!r}"},
                    )
                reason = f"no current task matching task_id {task_id!r}"
                self.logger.warning(f"rejecting RESULT_READY: {reason}")
                return self._protocol_reply(Topic.RESULT_REJECTED, **{MsgKey.REASON: reason})

            if task.result_id is not None:
                # duplicate RESULT_READY while the task is still current: idempotent by
                # result_id — reply the current state instead of a second result/transfer
                if result_id == task.result_id:
                    return self._protocol_reply(Topic.RESULT_ACCEPTED, **{MsgKey.RESULT_ID: result_id})
                return self._protocol_reply(
                    Topic.RESULT_REJECTED,
                    **{MsgKey.REASON: f"a result ({task.result_id}) was already accepted for this task"},
                )
            if not result_id or not transfer_id or not isinstance(ref_ids, list) or not ref_ids:
                return self._protocol_reply(
                    Topic.RESULT_REJECTED,
                    **{MsgKey.REASON: "invalid result envelope: result_id, transfer_id and manifest ref_ids required"},
                )
            if len(ref_ids) != 1:
                # V1 results are exactly one Shareable. Reject the malformed manifest here,
                # before any bytes move — the pull loop would otherwise download EVERY ref
                # only to have the single-Shareable check discard them all.
                return self._protocol_reply(
                    Topic.RESULT_REJECTED,
                    **{MsgKey.REASON: f"invalid result envelope: expected exactly one ref id, got {len(ref_ids)}"},
                )
            task.result_id = result_id
            task.result_transfer_id = transfer_id
            task.result_ref_ids = list(ref_ids)
            task.result_ready.set()
        return self._protocol_reply(Topic.RESULT_ACCEPTED, **{MsgKey.RESULT_ID: result_id})

    def _record_result_receipt_locked(self, task_id: str, result_id: str) -> None:
        """Records an accepted result for idempotent retries. Caller must hold _task_lock."""
        self._result_receipts.pop(task_id, None)  # move-to-end on re-record
        self._result_receipts[task_id] = result_id
        while len(self._result_receipts) > _RESULT_RECEIPT_RING:
            self._result_receipts.popitem(last=False)

    def _handle_task_failed(self, request):
        if self._closed:
            return make_cell_reply(CellReturnCode.OK)
        payload = request.payload
        if not isinstance(payload, dict):
            return make_cell_reply(CellReturnCode.INVALID_REQUEST, error="TASK_FAILED payload must be a dict")
        session, reject_reason = self._validate_session_msg(request, payload)
        if reject_reason:
            self.logger.warning(f"ignoring TASK_FAILED: {reject_reason}")
            return make_cell_reply(CellReturnCode.OK)
        task_id = payload.get(MsgKey.TASK_ID)
        # validate + commit in one _task_lock section (same discipline as _handle_result_ready),
        # so teardown cannot clear the task between the check and the failure mark
        with self._task_lock:
            task = self._current_task
            if task is None or task_id != task.task_id:
                self.logger.warning(f"ignoring TASK_FAILED: no current task matching task_id {task_id!r}")
                return make_cell_reply(CellReturnCode.OK)
            if task.failure_reason is None:
                task.failure_reason = str(payload.get(MsgKey.REASON))
            task.failed.set()
        return make_cell_reply(CellReturnCode.OK)

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

    def _handle_bye(self, request):
        # informational: the trainer announces orderly exit; process reaping is
        # authoritative, so this changes no state — which is why it carries no _closed
        # gate (see _register_protocol_cbs). Any future BYE behavior that mutates state
        # must add it.
        return make_cell_reply(CellReturnCode.OK)

    def _handle_task_payload_ready(self, request):
        # informational: the trainer materialized the task payload and
        # user code is training. No heartbeat lease exists in this backend, so nothing to
        # exempt/unexempt — log for observability and ack; stateless, hence no _closed gate.
        payload = request.payload if isinstance(request.payload, dict) else {}
        self.logger.info(f"trainer reports task payload ready (task_id={payload.get(MsgKey.TASK_ID)!r})")
        return make_cell_reply(CellReturnCode.OK)

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
        # mirrors the in_process backend's TASK_EXCHANGE meta: the Client API boundary
        # passes params through unconverted (format conversion belongs to send/receive
        # filters at the client edge; DIFF returns with the model-registry decision)
        context = self._context
        return {
            ConfigKey.TRAIN_WITH_EVAL: context.train_with_evaluation,
            ConfigKey.EXCHANGE_FORMAT: ExchangeFormat.RAW,
            ConfigKey.TRANSFER_TYPE: TransferType.FULL,
            ConfigKey.TRAIN_TASK_NAME: context.train_task_name,
            ConfigKey.EVAL_TASK_NAME: context.evaluate_task_name,
            ConfigKey.SUBMIT_MODEL_TASK_NAME: context.submit_model_task_name,
        }
