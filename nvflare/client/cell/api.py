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

"""Trainer-side Cell engine for the external_process Client API execution mode.

This is the counterpart of ExternalProcessBackend (the CJ side). It runs INSIDE the trainer
process that NVFlare launched, and is the Client API implementation that flare.init/receive/
send/log resolve to when the process was started with a Client API bootstrap config (the
NVFLARE_CLIENT_API_BOOTSTRAP env var; see nvflare/client/cell/bootstrap.py).

It replaces the legacy CellPipe + FlareAgent trainer stack with a direct Cell session using the
protocol vocabulary in nvflare/client/cell/defs.py. Task and result Shareables ride in
the Cell requests themselves: Cell/FOBS selects inline encoding or ViaDownloader references,
and the Client API keeps the original source alive until the ultimate receiver settles them:

- init(): read the bootstrap config, build a child Cell bound to the prescribed trainer FQCN and
  connected to the CJ's internal listener, register the control handlers (TASK_READY / ABORT /
  SHUTDOWN), and perform the HELLO handshake (launch-token proof; V1 trusted host).
- TASK_READY handler: the trainer Cell decodes/materializes the Shareable before invoking the
  handler, which validates and queues it before returning TASK_ACCEPTED.
- receive(): block until a materialized task is queued and return its FLModel.
- send(): send RESULT_READY with per-message pass-through enabled. The CJ accepts the lazy
  envelope and can forward its references so the declared downstream receiver downloads
  directly from this trainer. send() returns only after every created transfer reaches a
  successful terminal outcome.
- log(): send a LOG control message the executor converts to a fed analytics event.

V1 scope: rank 0 connects and drives the session; non-zero ranks get the model through the
training framework's own collectives and do not open a Client API session (the rank contract).
This engine therefore only builds the session for rank 0; other ranks get a passive API whose
receive() returns None and whose is_running() is False, so a plain (non-torchrun) trainer — the
primary external_process case — runs unchanged.
"""

import copy
import math
import os
import threading
import time
from queue import Empty, Queue
from typing import Any, Dict, Optional

from nvflare.apis.analytix import AnalyticsDataType
from nvflare.apis.fl_constant import FLMetaKey
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal
from nvflare.app_common.abstract.fl_model import FLModel, ParamsType
from nvflare.app_common.utils.fl_model_utils import FLModelUtils
from nvflare.client.api_spec import APISpec
from nvflare.client.cell.bootstrap import (
    BOOTSTRAP_FILE_ENV_VAR,
    BootstrapKey,
    get_bootstrap_client_api_type,
    read_bootstrap_config,
)
from nvflare.client.cell.decomposers import register_framework_decomposers
from nvflare.client.cell.defs import CHANNEL, PROTOCOL_VERSION, MsgKey, Topic
from nvflare.client.config import ConfigKey, ExchangeFormat, TransferType
from nvflare.client.params_conversion import convert_params
from nvflare.client.utils import DIFF_FUNCS
from nvflare.fuel.f3.cellnet.cell import Cell
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey
from nvflare.fuel.f3.cellnet.defs import ReturnCode as CellReturnCode
from nvflare.fuel.f3.cellnet.utils import make_reply as make_cell_reply
from nvflare.fuel.f3.cellnet.utils import new_cell_message
from nvflare.fuel.f3.streaming.byte_streamer import reliable_retry_scheduler
from nvflare.fuel.f3.streaming.download_service import DownloadService
from nvflare.fuel.f3.streaming.stream_utils import stream_shutdown
from nvflare.fuel.f3.streaming.transfer_progress import DEFAULT_STREAMING_IDLE_TIMEOUT, TransferProgressState
from nvflare.fuel.utils.fobs import FOBSContextKey
from nvflare.fuel.utils.fobs.decomposers.via_downloader import (
    RESULT_UPLOAD_PROGRESS_CTX_KEY,
    RESULT_UPLOAD_TX_CREATED_CB_CTX_KEY,
    ResultUploadProgressContextKey,
)
from nvflare.fuel.utils.log_utils import get_obj_logger

# How long HELLO retries for the CJ's HELLO_ACCEPTED before failing init.
_HELLO_TIMEOUT = 30.0
# Per-attempt timeout / backoff while the child cell's connection to the CJ comes up.
_HELLO_RETRY_INTERVAL = 1.0
# Poll cadence of receive()'s task-queue wait (the queue wakes it immediately; the poll only
# bounds abort/stop detection latency).
_RECEIVE_POLL_INTERVAL = 0.5
# Heartbeat requests are cancelled during shutdown and the daemon is joined boundedly.
_HEARTBEAT_JOIN_TIMEOUT = 1.0


class TrainerSessionError(Exception):
    """The trainer's Client API session ended (SHUTDOWN, ABORT, or CJ/cell loss)."""


def _shutdown_f3_streaming() -> None:
    """Retire the process-global F3 streaming services used by a bare trainer.

    Normal NVFlare entry points run under MainProcessMonitor, but an external trainer's
    entry point is the user's script and cannot be wrapped by it. Keep this list aligned
    with the process-global cleanup registrations in byte_streamer.py and stream_utils.py,
    plus DownloadService cleanup performed by the normal client/server runners. If a
    second standalone-process caller appears, promote this into an F3-owned, appropriately
    scoped lifecycle API instead of copying the list.

    Every stage is attempted even if an earlier one fails: one broken service must not
    strand another non-daemon pool. Stop transaction ownership first, then the retry
    dispatcher, before joining the executors needed by retry work. The underlying
    shutdown operations are idempotent, so a repeated caller retries a partial cleanup
    without a second completion latch.
    """
    errors = []
    for name, shutdown in (
        ("download service", DownloadService.shutdown),
        ("reliable retry scheduler", reliable_retry_scheduler.shutdown),
        ("stream executors", stream_shutdown),
    ):
        try:
            shutdown()
        except Exception as e:
            errors.append((name, e))
    if errors:
        names = ", ".join(name for name, _ in errors)
        raise RuntimeError(f"failed to stop F3 streaming services: {names}") from errors[0][1]


def _to_python_scalar(v: Any) -> Any:
    """Coerce a 0-d numpy scalar (np.float32, np.int64, ...) to a Python scalar.

    Trainers naturally produce numpy scalar metrics; the analytics DXO validation rejects
    numpy scalar types, so convert at the source. Non-numpy values and arrays pass through.
    """
    item = getattr(v, "item", None)
    if callable(item) and getattr(v, "shape", None) == ():
        return v.item()
    return v


class CellClientAPI(APISpec):
    """Client API implementation that speaks the external_process Cell protocol to the CJ."""

    def __init__(self, bootstrap_file: Optional[str] = None):
        """
        Args:
            bootstrap_file: path to the bootstrap config. Defaults to the path in the
                NVFLARE_CLIENT_API_BOOTSTRAP env var the backend set on the launched process.
        """
        super().__init__()
        self.logger = get_obj_logger(self)
        self._bootstrap_file = bootstrap_file or os.environ.get(BOOTSTRAP_FILE_ENV_VAR)
        if not self._bootstrap_file:
            raise RuntimeError(
                f"no Client API bootstrap config: set {BOOTSTRAP_FILE_ENV_VAR} or pass bootstrap_file "
                f"(the external_process backend writes it on the launched trainer)"
            )
        self._config = read_bootstrap_config(self._bootstrap_file)
        # Environment selection remains a compatibility path for untyped legacy files,
        # but any typed envelope must be complete and supported on every construction
        # path, including a backend-launched trainer.
        get_bootstrap_client_api_type(self._config, self._bootstrap_file)

        self._rank: Optional[str] = None
        self._is_control_rank = False
        self._cell: Optional[Cell] = None
        self._session_id: Optional[str] = None
        self._cj_fqcn: str = self._config[BootstrapKey.CJ_FQCN]
        self._trainer_fqcn: str = self._config[BootstrapKey.TRAINER_FQCN]
        self._job_id: str = self._config[BootstrapKey.JOB_ID]
        self._site_name: str = self._config[BootstrapKey.SITE_NAME]
        self._task_exchange: dict = self._config.get(BootstrapKey.TASK_EXCHANGE, {})
        # This is the same TASK_EXCHANGE lifecycle bit used by the legacy ex-process
        # Client API. Default to a persistent session for typed bootstrap files written
        # before the field was added, since closing after one result is irreversible.
        self._launch_once = bool(self._task_exchange.get(ConfigKey.LAUNCH_ONCE, True))
        self._memory_gc_rounds = int(self._config.get(BootstrapKey.MEMORY_GC_ROUNDS, 0))
        self._cuda_empty_cache = bool(self._config.get(BootstrapKey.CUDA_EMPTY_CACHE, False))

        # control state. _task_queue carries materialized tasks: Cell finishes FOBS decode
        # (including any ViaDownloader pulls) before it invokes the TASK_READY handler.
        self._task_queue: "Queue[dict]" = Queue()
        self._current_task: Optional[dict] = None
        self._result_receiver_ids = None
        self._fl_model: Optional[FLModel] = None
        self._receive_called = False
        self._abort = False
        self._abort_reason = ""
        # observed by incoming task payload pulls between chunk requests, so an abort/stop
        # ends an in-flight materialization instead of letting it run to completion for
        # nothing. Result publication deliberately has a separate cancellation signal:
        # an orderly SHUTDOWN can race the RESULT_ACCEPTED reply and must not tear down a
        # result source that the downstream receiver is still downloading.
        self._abort_signal = Signal()
        self._result_abort_signal = Signal()
        self._heartbeat_cancel = Signal()
        self._stopped = False
        self._closed = False
        self._lock = threading.Lock()
        self._shutdown_lock = threading.Lock()
        self._heartbeat_lock = threading.Lock()
        self._heartbeat_interval = 0.0
        self._heartbeat_timeout = 0.0
        self._last_cj_activity: Optional[float] = None
        self._heartbeat_stop = threading.Event()
        self._heartbeat_thread: Optional[threading.Thread] = None
        # While the trainer is serving an actual live DownloadService transaction, that
        # transaction's progress/idle policy is authoritative. Merely entering send() is
        # not enough: inline serialization or a wedged RESULT_READY request must remain
        # bounded by the CJ heartbeat lease.
        self._result_transactions = ()
        # Guarded by _lock. SHUTDOWN reads this in the same critical section in which it
        # sets _stopped, so either send() observes the stop and closes itself or the
        # SHUTDOWN reply tells the backend the process is already safe to stop.
        self._result_send_active = False
        self._params_conversion_state = {}

    # ------------------------------------------------------------------ lifecycle

    @property
    def closed(self) -> bool:
        """Whether this Cell API has been shut down."""
        return self._closed

    def init(self, rank: Optional[str] = None):
        self._rank = rank if rank is not None else os.environ.get("RANK", "0")
        self._is_control_rank = str(self._rank) == "0"
        if not self._is_control_rank:
            # non-zero ranks get the model via the framework's collectives; they do not open
            # a Client API session (rank contract). Leave the API passive.
            self.logger.info(f"rank {self._rank}: no Client API session (non-control rank)")
            return

        # Register the serialization decomposers the launched trainer process needs to decode
        # task payloads and encode results (the FL process/CJ registers these at startup; a
        # bare trainer subprocess must too). common_decomposers covers numpy/FLModel and the
        # download-ref DOT handlers used by the shared payload path. Mirrors FlareAgent.
        from nvflare.apis.utils.decomposers import flare_decomposers
        from nvflare.app_common.decomposers import common_decomposers

        flare_decomposers.register()
        common_decomposers.register()
        register_framework_decomposers(
            self._task_exchange.get(ConfigKey.EXCHANGE_FORMAT, ExchangeFormat.RAW),
            self._task_exchange.get(ConfigKey.SERVER_EXPECTED_FORMAT, ExchangeFormat.NUMPY),
            self.logger,
        )

        connect_url = self._config[BootstrapKey.CONNECT_URL]
        self._cell = Cell(
            fqcn=self._trainer_fqcn,
            root_url=None,
            secure=False,  # V1 trusted-host: the CJ's internal listener is a local connection
            credentials={},
            parent_url=connect_url,
            create_internal_listener=False,
        )
        # Direct task decode must observe a concurrent ABORT/SHUTDOWN while nested tensor
        # downloads are running. Adapter.call() derives each decode context from this Cell
        # context, so the signal reaches ViaDownloader without a ClientAPI-specific wrapper.
        self._cell.update_fobs_context({FOBSContextKey.ABORT_SIGNAL: self._abort_signal})
        self._register_control_cbs(self._cell)
        self._cell.start()
        try:
            self._hello()
            self._start_heartbeat()
        except Exception:
            self._stop_heartbeat()
            self._stop_cell()
            raise
        self.logger.info(f"trainer session established: fqcn={self._trainer_fqcn} session_id={self._session_id}")

    def _register_control_cbs(self, cell: Cell) -> None:
        cell.register_request_cb(channel=CHANNEL, topic=Topic.TASK_READY, cb=self._handle_task_ready)
        cell.register_request_cb(channel=CHANNEL, topic=Topic.ABORT, cb=self._handle_abort)
        cell.register_request_cb(channel=CHANNEL, topic=Topic.SHUTDOWN, cb=self._handle_shutdown)

    def _hello(self) -> None:
        # The child cell connects to the CJ's internal listener asynchronously after start();
        # the first HELLO can race that connection and come back unreachable. Retry with a
        # short backoff until the CJ replies (or _HELLO_TIMEOUT), so a connection still coming
        # up is not mistaken for a rejected handshake.
        deadline = time.monotonic() + _HELLO_TIMEOUT
        reply = None
        attempt = 0
        while True:
            attempt += 1
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TrainerSessionError(f"no HELLO reply from the CJ after {_HELLO_TIMEOUT}s (cell not connected)")
            reply = self._cell.send_request(
                channel=CHANNEL,
                topic=Topic.HELLO,
                target=self._cj_fqcn,
                request=new_cell_message(
                    {},
                    {
                        MsgKey.TRAINER_FQCN: self._trainer_fqcn,
                        MsgKey.PROOF: self._config[BootstrapKey.LAUNCH_TOKEN],
                        # Report the trainer code's compiled wire version. Protocol
                        # compatibility is negotiated by HELLO, not selected by bootstrap.
                        MsgKey.PROTOCOL_VERSION: PROTOCOL_VERSION,
                        MsgKey.JOB_ID: self._job_id,
                        MsgKey.SITE_NAME: self._site_name,
                        MsgKey.RANK: str(self._rank),
                    },
                ),
                timeout=min(_HELLO_RETRY_INTERVAL, remaining),
            )
            rc = None if reply is None else reply.get_header(MessageHeaderKey.RETURN_CODE)
            if rc == CellReturnCode.OK:
                break
            # transient: connection to the CJ not up yet (unreachable / no reply). Back off.
            self.logger.debug(f"HELLO attempt {attempt} not yet delivered (rc={rc}); retrying")
            time.sleep(min(_HELLO_RETRY_INTERVAL, max(0.0, deadline - time.monotonic())))
        body = reply.payload
        if not isinstance(body, dict) or body.get(MsgKey.REPLY_TOPIC) != Topic.HELLO_ACCEPTED:
            reason = body.get(MsgKey.REASON) if isinstance(body, dict) else body
            raise TrainerSessionError(f"HELLO not accepted: {reason}")
        self._session_id = body.get(MsgKey.SESSION_ID)
        if not self._session_id:
            raise TrainerSessionError("HELLO_ACCEPTED carried no session id")
        self._heartbeat_interval = self._valid_heartbeat_number(
            MsgKey.HEARTBEAT_INTERVAL, body.get(MsgKey.HEARTBEAT_INTERVAL), positive=True
        )
        self._heartbeat_timeout = self._valid_heartbeat_number(
            MsgKey.HEARTBEAT_TIMEOUT, body.get(MsgKey.HEARTBEAT_TIMEOUT), positive=False
        )
        if 0 < self._heartbeat_timeout <= self._heartbeat_interval:
            raise TrainerSessionError(
                f"invalid heartbeat policy: interval {self._heartbeat_interval} must be less than "
                f"timeout {self._heartbeat_timeout}"
            )
        self._note_cj_activity()

    # ------------------------------------------------------------------ receive

    def receive(self, timeout: Optional[float] = None) -> Optional[FLModel]:
        if not self._is_control_rank or self._closed:
            return None
        if self._abort:
            reason = self._abort_reason
            self.shutdown()
            raise TrainerSessionError(f"session aborted: {reason}")
        if self._stopped:
            # Standard trainers leave their loop after receive()/is_running() observes
            # SHUTDOWN. Close F3 synchronously on this user thread so interpreter teardown
            # does not wait forever for its worker pools.
            self.shutdown()
            return None
        if self._fl_model is not None:
            return self._fl_model

        try:
            entry = self._await_task(timeout)
        except TrainerSessionError:
            self.shutdown()
            raise
        if entry is None:
            if self._stopped and not self._closed:
                self.shutdown()
            return None

        task = entry["task"]
        self._current_task = task
        self._result_receiver_ids = entry.get("result_receiver_ids")
        self._fl_model = entry["model"]
        self._receive_called = True
        return self._fl_model

    def _await_task(self, timeout: Optional[float]) -> Optional[dict]:
        """Blocks for the next materialized task entry. Returns None on clean stop
        (SHUTDOWN) or timeout; raises on ABORT (the receive-side contract: abort surfaces
        as an exception out of a blocked receive)."""
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            if self._abort:
                raise TrainerSessionError(f"session aborted: {self._abort_reason}")
            if self._stopped or self._closed:
                return None
            wait = _RECEIVE_POLL_INTERVAL
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                wait = min(wait, remaining)
            try:
                return self._task_queue.get(timeout=wait)
            except Empty:
                continue

    # ------------------------------------------------------------------ send

    def send(self, model: FLModel, clear_cache: bool = True) -> None:
        if not self._is_control_rank or self._closed:
            return
        if not self._receive_called:
            raise RuntimeError('"receive" needs to be called before sending model!')
        self._check_session_alive()

        task = self._current_task
        if task is None:
            raise TrainerSessionError("send() called with no current task")

        if self._task_exchange.get(ConfigKey.TRANSFER_TYPE) == TransferType.DIFF:
            model = self._prepare_param_diff(model)
        if model.params is None and model.metrics is None:
            raise RuntimeError("the model to send does not have either params or metrics")

        # DIFF is computed above in the trainer-native representation. Adapt only a
        # shallow wire model so clear_cache=False leaves the user's FLModel native.
        wire_model = copy.copy(model)
        wire_model.params = convert_params(
            model.params,
            self._task_exchange.get(ConfigKey.EXCHANGE_FORMAT, ExchangeFormat.RAW),
            self._task_exchange.get(ConfigKey.SERVER_EXPECTED_FORMAT, ExchangeFormat.NUMPY),
            self._params_conversion_state,
            self.logger,
        )
        shareable = FLModelUtils.to_shareable(wire_model)
        transactions = []

        def _on_transaction_created(transaction):
            transactions.append(transaction)

        # A no-op progress callback requests transaction tracking from ViaDownloader.
        # DownloadService itself records per-receiver activity and terminal truth; the
        # callback is only the opt-in that exposes the created transaction to this sender.
        def _on_result_progress(**_kwargs):
            return None

        request = new_cell_message(
            {MessageHeaderKey.PASS_THROUGH: True},
            {
                MsgKey.SESSION_ID: self._session_id,
                MsgKey.TASK_ID: task.get(MsgKey.TASK_ID),
                MsgKey.RESULT: shareable,
            },
        )
        # These are the ultimate result receivers declared on the incoming task (if the
        # workflow knows them). Merely accepting the lazy envelope does not make the CJ a
        # DownloadService receiver; it remains an intermediate forwarding hop.
        result_receiver_ids = self._result_receiver_ids

        fobs_ctx_props = {
            FOBSContextKey.STREAM_PROGRESS_CB: _on_result_progress,
            RESULT_UPLOAD_TX_CREATED_CB_CTX_KEY: _on_transaction_created,
            RESULT_UPLOAD_PROGRESS_CTX_KEY: {
                ResultUploadProgressContextKey.JOB_ID: self._job_id,
                ResultUploadProgressContextKey.TASK_ID: task.get(MsgKey.TASK_ID),
                ResultUploadProgressContextKey.STREAMING_IDLE_TIMEOUT: DEFAULT_STREAMING_IDLE_TIMEOUT,
            },
        }

        def _has_live_result_transfer():
            for transaction in tuple(transactions):
                try:
                    if not DownloadService.get_transfer_waiter(transaction.tx_id).done():
                        return True
                except Exception:
                    # A transaction deleted between snapshot and lookup is no longer live.
                    continue
            return False

        result_accepted = False
        # Serialize result publication admission with SHUTDOWN. If shutdown wins, no
        # result has been published and send fails normally. If send wins, SHUTDOWN does
        # not cancel the result signal; send remains the terminal-transfer barrier.
        with self._lock:
            self._check_session_alive()
            self._result_send_active = True
        try:
            self._set_result_transactions(transactions)
            reply = self._cell.send_request(
                channel=CHANNEL,
                topic=Topic.RESULT_READY,
                target=self._cj_fqcn,
                request=request,
                timeout=_HELLO_TIMEOUT,
                abort_signal=self._result_abort_signal,
                progress_wait_cb=_has_live_result_transfer,
                num_receivers=len(result_receiver_ids) if result_receiver_ids else 1,
                receiver_ids=result_receiver_ids,
                fobs_ctx_props=fobs_ctx_props,
            )
            self._check_result_accepted(reply)
            result_accepted = True
            self._note_cj_activity()
            self._wait_for_result_transfers(transactions)
        except BaseException:
            self._delete_result_transactions(transactions)
            raise
        finally:
            try:
                self._set_result_transactions(())
                if clear_cache:
                    # Serialization is complete. Release both the submitted and received
                    # parameter sets; neither is needed after the actual transfer settles.
                    model.params = None
                    model.optimizer_params = None
                    received_model = self._fl_model
                    self._fl_model = None
                    if received_model is not None:
                        received_model.params = None
                        received_model.optimizer_params = None
                    self._receive_called = False
                    self._current_task = None
                    self._result_receiver_ids = None
                self._maybe_cleanup_memory()
            finally:
                # Pair this transition with _handle_shutdown under one lock. If SHUTDOWN
                # won first, this thread owns the close after the actual transfer settled.
                # If this transition won first, SHUTDOWN reports RESULT_SOURCE_LIVE=False
                # and the backend can stop the already-settled process after receiving ACK.
                with self._lock:
                    self._result_send_active = False
                    should_shutdown = self._stopped or (result_accepted and not self._launch_once)
                # A per-task trainer has completed its whole session only after the CJ
                # accepted the envelope AND the actual downstream receiver completed every
                # download. Close Cell/F3 synchronously before a one-shot script returns.
                if should_shutdown:
                    self.shutdown()

    def _wait_for_result_transfers(self, transactions) -> None:
        """Wait for strict terminal success of every result DownloadService transaction."""
        for transaction in tuple(transactions):
            waiter = DownloadService.get_transfer_waiter(transaction.tx_id)
            while True:
                outcome = waiter.wait(timeout=_RECEIVE_POLL_INTERVAL)
                if outcome is not None:
                    break
                if waiter.done():
                    raise TrainerSessionError(f"result transfer {transaction.tx_id} ended without a terminal outcome")
                if self._abort:
                    raise TrainerSessionError(f"session aborted while serving result: {self._abort_reason}")
                if self._closed:
                    raise TrainerSessionError("session closed while serving result")
            if outcome.status != TransferProgressState.COMPLETED:
                raise TrainerSessionError(
                    f"result transfer {transaction.tx_id} failed: status={outcome.status} reason={outcome.reason}"
                )

    @staticmethod
    def _delete_result_transactions(transactions) -> None:
        for transaction in transactions:
            try:
                DownloadService.delete_transaction(transaction.tx_id)
            except Exception:
                # Preserve the original send/abort failure; DownloadService cleanup is
                # best-effort and the transaction idle timeout remains a backstop.
                pass

    def _prepare_param_diff(self, model: FLModel) -> FLModel:
        exchange_format = self._task_exchange.get(ConfigKey.EXCHANGE_FORMAT, ExchangeFormat.RAW)
        diff_func = DIFF_FUNCS.get(exchange_format)
        if diff_func is None and exchange_format == ExchangeFormat.RAW:
            diff_func = DIFF_FUNCS.get(ExchangeFormat.NUMPY)
        if diff_func is None:
            raise RuntimeError(f"no default params diff function for {exchange_format}")
        if self._fl_model is None:
            raise RuntimeError("no received model")
        if self._fl_model.params is not None and model.params is not None and model.params_type == ParamsType.FULL:
            try:
                model.params = diff_func(original=self._fl_model.params, new=model.params)
                model.params_type = ParamsType.DIFF
            except Exception as e:
                raise RuntimeError(f"params diff function failed: {e}") from e
        return model

    def _check_result_accepted(self, reply) -> None:
        if reply is None:
            raise TrainerSessionError("no reply to RESULT_READY from the CJ")
        rc = reply.get_header(MessageHeaderKey.RETURN_CODE)
        if rc != CellReturnCode.OK:
            raise TrainerSessionError(f"cell-level failure on RESULT_READY: {rc}")
        body = reply.payload
        if not isinstance(body, dict) or body.get(MsgKey.REPLY_TOPIC) != Topic.RESULT_ACCEPTED:
            reason = body.get(MsgKey.REASON) if isinstance(body, dict) else body
            raise TrainerSessionError(f"result was rejected by the CJ: {reason}")

    # ------------------------------------------------------------------ log / info

    def log(self, key: str, value: Any, data_type: AnalyticsDataType, **kwargs):
        if self._closed or not self._is_control_rank:
            return
        if str(self._rank) != "0":
            raise RuntimeError("only rank 0 can call log!")
        try:
            self._cell.fire_and_forget(
                channel=CHANNEL,
                topic=Topic.LOG,
                targets=[self._cj_fqcn],
                message=new_cell_message(
                    {},
                    {
                        MsgKey.SESSION_ID: self._session_id,
                        "key": key,
                        # numpy scalar metrics (e.g. np.mean(...) -> np.float32) must go over
                        # the wire as a Python scalar: the analytics DXO validation on the CJ
                        # (create_analytic_dxo) rejects numpy scalar types.
                        "value": _to_python_scalar(value),
                        "data_type": data_type,
                        **kwargs,
                    },
                ),
                optional=True,
            )
        except Exception as e:
            self.logger.warning(f"failed to send LOG '{key}': {e}")

    def system_info(self) -> Dict:
        return {FLMetaKey.SITE_NAME: self._site_name, FLMetaKey.JOB_ID: self._job_id}

    def get_config(self) -> Dict:
        # Match the legacy ClientConfig shape consumed by existing trainers (for example
        # Lightning integrations call get_config()[TASK_EXCHANGE]). Do not expose Cell
        # addresses or the launch credential from the bootstrap file.
        return {
            ConfigKey.TASK_EXCHANGE: dict(self._task_exchange),
            FLMetaKey.JOB_ID: self._job_id,
            FLMetaKey.SITE_NAME: self._site_name,
            ConfigKey.MEMORY_GC_ROUNDS: self._memory_gc_rounds,
            ConfigKey.CUDA_EMPTY_CACHE: self._cuda_empty_cache,
        }

    def get_job_id(self) -> str:
        return self._job_id

    def get_site_name(self) -> str:
        return self._site_name

    def get_task_name(self) -> str:
        self._require_control_rank("get_task_name")
        task = self._current_task
        if task is None:
            raise RuntimeError("no current task")
        return task.get(MsgKey.TASK_NAME)

    def is_running(self) -> bool:
        # loop guard: False on any session end (abort/stop/closed). Otherwise block in
        # receive() for the next task; an abort arriving during that block returns False
        # here (the loop exits), while an explicit flare.receive()/send() still raises.
        if not self._is_control_rank or self._closed:
            return False
        if self._abort or self._stopped:
            self.shutdown()
            return False
        try:
            return self.receive() is not None
        except TrainerSessionError:
            self.shutdown()
            return False

    def is_train(self) -> bool:
        self._require_control_rank("is_train")
        return self._current_task_name() == self._task_exchange.get(ConfigKey.TRAIN_TASK_NAME)

    def is_evaluate(self) -> bool:
        self._require_control_rank("is_evaluate")
        return self._current_task_name() == self._task_exchange.get(ConfigKey.EVAL_TASK_NAME)

    def is_submit_model(self) -> bool:
        self._require_control_rank("is_submit_model")
        return self._current_task_name() == self._task_exchange.get(ConfigKey.SUBMIT_MODEL_TASK_NAME)

    def clear(self):
        self._fl_model = None
        self._receive_called = False

    def shutdown(self):
        """Stop this trainer session and retire its process-global F3 streaming runtime.

        A Cell Client API trainer is currently one-session-per-process. After shutdown,
        no further Cell Client API session can run in this interpreter.
        """
        with self._shutdown_lock:
            if not self._closed:
                self._closed = True
                self._stopped = True
                self._abort_signal.trigger("client api shutdown")
                self._result_abort_signal.trigger("client api shutdown")
                self._stop_heartbeat()
                self._stop_cell()
            try:
                # This API owns a dedicated trainer process, unlike CJ/server processes
                # whose MainProcessMonitor performs global F3 cleanup. Without this step,
                # non-daemon stream pools keep one-shot/torchrun trainers alive forever.
                # Retry this process-global cleanup on a repeated shutdown() if a previous
                # stage failed; the services' own shutdown operations are idempotent.
                _shutdown_f3_streaming()
            except Exception as e:
                self.logger.warning(f"failed to stop trainer streaming services: {e}")

    # ------------------------------------------------------------------ control handlers

    def _handle_task_ready(self, request):
        # TASK_READY is currently sent once. Attach-mode redelivery tolerance should be
        # added only as a sender-retry + receiver-dedup pair.
        payload = request.payload
        if not isinstance(payload, dict):
            return make_cell_reply(CellReturnCode.INVALID_REQUEST, error="TASK_READY payload must be a dict")
        reject_reason = self._validate_cj_control(request, payload)
        if reject_reason:
            return self._reply(Topic.TASK_FAILED, **{MsgKey.REASON: reject_reason})
        self._note_cj_activity()
        task_id = payload.get(MsgKey.TASK_ID)
        shareable = payload.get(MsgKey.MODEL)
        if not isinstance(shareable, Shareable):
            return self._reply(
                Topic.TASK_FAILED,
                **{
                    MsgKey.TASK_ID: task_id,
                    MsgKey.REASON: f"TASK_READY model must be Shareable, got {type(shareable)}",
                },
            )
        try:
            model = FLModelUtils.from_shareable(shareable)
            model.params = convert_params(
                model.params,
                self._task_exchange.get(ConfigKey.SERVER_EXPECTED_FORMAT, ExchangeFormat.NUMPY),
                self._task_exchange.get(ConfigKey.EXCHANGE_FORMAT, ExchangeFormat.RAW),
                self._params_conversion_state,
                self.logger,
            )
        except Exception as e:
            return self._reply(
                Topic.TASK_FAILED,
                **{MsgKey.TASK_ID: task_id, MsgKey.REASON: f"invalid task model: {e}"},
            )
        result_receiver_ids = self._normalize_result_receiver_ids(shareable.get_header(FOBSContextKey.RECEIVER_IDS))
        self._task_queue.put({"task": payload, "model": model, "result_receiver_ids": result_receiver_ids})
        return self._reply(Topic.TASK_ACCEPTED, **{MsgKey.TASK_ID: task_id})

    def _handle_abort(self, request):
        payload = request.payload if isinstance(request.payload, dict) else {}
        reject_reason = self._validate_cj_control(request, payload)
        if reject_reason:
            return make_cell_reply(CellReturnCode.INVALID_REQUEST, error=reject_reason)
        self._note_cj_activity()
        self._abort = True
        self._abort_reason = str(payload.get(MsgKey.REASON))
        self._abort_signal.trigger(self._abort_reason)
        self._result_abort_signal.trigger(self._abort_reason)
        self.logger.error(f"session aborted by CJ: {self._abort_reason}")
        return make_cell_reply(CellReturnCode.OK)

    def _handle_shutdown(self, request):
        payload = request.payload if isinstance(request.payload, dict) else {}
        reject_reason = self._validate_cj_control(request, payload)
        if reject_reason:
            return make_cell_reply(CellReturnCode.INVALID_REQUEST, error=reject_reason)
        self._note_cj_activity()
        with self._lock:
            self._stopped = True
            result_source_live = self._result_send_active
        # An incoming task materialization still in flight serves no one after an orderly
        # stop: receive() will return None and drop the entry. Do not trigger the result
        # signal here. SHUTDOWN can arrive after the CJ accepted a lazy result envelope but
        # before RESULT_ACCEPTED reaches send(); the result remains live until its actual
        # downstream DownloadService transaction settles.
        self._abort_signal.trigger("session shutdown")
        self.logger.info("session shutdown requested by CJ")
        return make_cell_reply(CellReturnCode.OK, body={MsgKey.RESULT_SOURCE_LIVE: result_source_live})

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _reply(reply_topic: str, **fields):
        body = {MsgKey.REPLY_TOPIC: reply_topic}
        body.update(fields)
        return make_cell_reply(CellReturnCode.OK, body=body)

    @staticmethod
    def _normalize_result_receiver_ids(receiver_ids):
        if receiver_ids is None:
            return None
        if isinstance(receiver_ids, str):
            values = [receiver_ids]
        else:
            try:
                values = list(receiver_ids)
            except TypeError:
                return None
        normalized = tuple(str(receiver_id) for receiver_id in values if receiver_id is not None and str(receiver_id))
        return normalized or None

    def _check_session_alive(self) -> None:
        if self._abort:
            raise TrainerSessionError(f"session aborted: {self._abort_reason}")
        if self._stopped or self._closed:
            raise TrainerSessionError("session stopped")

    def _validate_cj_control(self, request, payload: dict) -> Optional[str]:
        origin = request.get_header(MessageHeaderKey.ORIGIN) or ""
        if origin != self._cj_fqcn:
            return f"unexpected CJ origin {origin!r}"
        if payload.get(MsgKey.SESSION_ID) != self._session_id:
            return "stale or unknown session id"
        return None

    @staticmethod
    def _valid_heartbeat_number(name: str, value, positive: bool) -> float:
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
            or (positive and value <= 0)
            or (not positive and value < 0)
        ):
            relation = "> 0" if positive else ">= 0"
            raise TrainerSessionError(f"HELLO_ACCEPTED {name} must be a finite number {relation}, got {value!r}")
        return float(value)

    def _start_heartbeat(self) -> None:
        if self._heartbeat_timeout == 0:
            return
        thread = threading.Thread(target=self._heartbeat_loop, name="client_api_heartbeat", daemon=True)
        self._heartbeat_thread = thread
        thread.start()

    def _heartbeat_loop(self) -> None:
        while not self._heartbeat_stop.wait(self._heartbeat_interval):
            if self._closed or self._stopped or self._abort:
                return
            try:
                reply = self._cell.send_request(
                    channel=CHANNEL,
                    topic=Topic.HEARTBEAT,
                    target=self._cj_fqcn,
                    request=new_cell_message({}, {MsgKey.SESSION_ID: self._session_id}),
                    timeout=min(self._heartbeat_interval, self._heartbeat_timeout),
                    abort_signal=self._heartbeat_cancel,
                )
                if self._heartbeat_reply_valid(reply):
                    self._note_cj_activity()
            except Exception as e:
                self.logger.debug(f"heartbeat to CJ failed: {e}")

            silent_for = self._cj_silent_for()
            if silent_for > self._heartbeat_timeout and not self._has_live_result_transfer():
                self._mark_owner_lost(
                    f"CJ heartbeat timed out after {silent_for:.1f}s (timeout={self._heartbeat_timeout}s)"
                )
                return

    def _heartbeat_reply_valid(self, reply) -> bool:
        if reply is None or reply.get_header(MessageHeaderKey.RETURN_CODE) != CellReturnCode.OK:
            return False
        body = reply.payload
        return (
            isinstance(body, dict)
            and body.get(MsgKey.REPLY_TOPIC) == Topic.HEARTBEAT
            and body.get(MsgKey.SESSION_ID) == self._session_id
        )

    def _note_cj_activity(self) -> None:
        with self._heartbeat_lock:
            self._last_cj_activity = time.monotonic()

    def _cj_silent_for(self) -> float:
        with self._heartbeat_lock:
            last_activity = self._last_cj_activity
        return float("inf") if last_activity is None else max(0.0, time.monotonic() - last_activity)

    def _set_result_transactions(self, transactions) -> None:
        with self._heartbeat_lock:
            self._result_transactions = transactions

    def _has_live_result_transfer(self) -> bool:
        with self._heartbeat_lock:
            transactions = tuple(self._result_transactions)
        for transaction in transactions:
            try:
                if not DownloadService.get_transfer_waiter(transaction.tx_id).done():
                    return True
            except Exception:
                # A transaction deleted between snapshot and lookup is no longer live.
                continue
        return False

    def _mark_owner_lost(self, reason: str) -> None:
        with self._heartbeat_lock:
            if self._abort or self._stopped or self._closed:
                return
            self._abort = True
            self._abort_reason = reason
        self._abort_signal.trigger(reason)
        self._result_abort_signal.trigger(reason)
        self._heartbeat_stop.set()
        self.logger.error(reason)

    def _stop_heartbeat(self) -> None:
        self._heartbeat_stop.set()
        self._heartbeat_cancel.trigger("client api heartbeat stopped")
        thread = self._heartbeat_thread
        if thread is not None and thread is not threading.current_thread() and thread.is_alive():
            thread.join(timeout=_HEARTBEAT_JOIN_TIMEOUT)

    def _current_task_name(self) -> Optional[str]:
        task = self._current_task
        return task.get(MsgKey.TASK_NAME) if task else None

    def _require_control_rank(self, what: str) -> None:
        if str(self._rank) != "0":
            raise RuntimeError(f"only rank 0 can call {what}!")

    def _stop_cell(self) -> None:
        cell = self._cell
        if cell is not None:
            try:
                cell.stop()
            except Exception as e:
                self.logger.debug(f"failed to stop trainer cell: {e}")
