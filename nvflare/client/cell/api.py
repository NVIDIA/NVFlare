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

It replaces the legacy CellPipe + FlareAgent trainer stack with a direct Cell session on the
frozen protocol vocabulary in nvflare/client/cell/defs.py, and moves payloads through the same
payload_transfer seam the backend uses (so both sides share identical "returns == delivered"
semantics):

- init(): read the bootstrap config, build a child Cell bound to the prescribed trainer FQCN and
  connected to the CJ's internal listener, register the control handlers (TASK_READY / ABORT /
  SHUTDOWN), and perform the HELLO handshake (launch-token proof; V1 trusted host).
- TASK_READY handler: ack the control message, then EAGERLY pull the task payload on a
  materializer thread and send TASK_PAYLOAD_READY once it is stored. Eager, not deferred to
  receive(): the producer-side acquire budget expects the first pull promptly after TASK_READY,
  and user code may legitimately spend minutes between rounds (eval, checkpointing) before its
  next receive() — the payload must not wait on that.
- receive(): block until a materialized task is queued and return its FLModel.
- send(): publish the result as a producer-side payload attempt, send RESULT_READY with the
  manifest, and hold the attempt alive until the CJ's pull reaches its terminal outcome (the
  producer-liveness rule) before returning.
- log(): send a LOG control message the executor converts to a fed analytics event.

V1 scope: rank 0 connects and drives the session; non-zero ranks get the model through the
training framework's own collectives and do not open a Client API session (the rank contract).
This engine therefore only builds the session for rank 0; other ranks get a passive API whose
receive() returns None and whose is_running() is False, so a plain (non-torchrun) trainer — the
primary external_process case — runs unchanged.
"""

import os
import threading
import time
import uuid
from queue import Empty, Queue
from typing import Any, Dict, Optional

from nvflare.apis.analytix import AnalyticsDataType
from nvflare.apis.fl_constant import FLMetaKey
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal
from nvflare.app_common.abstract.fl_model import FLModel
from nvflare.app_common.utils.fl_model_utils import FLModelUtils
from nvflare.client.api_spec import APISpec
from nvflare.client.cell.bootstrap import BOOTSTRAP_FILE_ENV_VAR, BootstrapKey, read_bootstrap_config
from nvflare.client.cell.decomposers import register_framework_decomposers
from nvflare.client.cell.defs import CHANNEL, PROTOCOL_VERSION, MsgKey, Topic
from nvflare.client.cell.payload_transfer import (
    TRANSFER_TTL,
    PayloadTransferError,
    TaskPayloadAttempt,
    fetch_result_payload,
)
from nvflare.client.config import ConfigKey
from nvflare.fuel.f3.cellnet.cell import Cell
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey
from nvflare.fuel.f3.cellnet.defs import ReturnCode as CellReturnCode
from nvflare.fuel.f3.cellnet.utils import make_reply as make_cell_reply
from nvflare.fuel.f3.cellnet.utils import new_cell_message
from nvflare.fuel.utils.log_utils import get_obj_logger

# How long HELLO retries for the CJ's HELLO_ACCEPTED before failing init.
_HELLO_TIMEOUT = 30.0
# Per-attempt timeout / backoff while the child cell's connection to the CJ comes up.
_HELLO_RETRY_INTERVAL = 1.0
# Poll cadence of receive()'s task-queue wait (the queue wakes it immediately; the poll only
# bounds abort/stop detection latency).
_RECEIVE_POLL_INTERVAL = 0.5
# Producer-side hold: how long send() waits for the CJ to certify it pulled the result before
# giving up. Aligned with the attempt's own TTL backstop so this wait can never give up while
# the transfer is still legitimately live.
_RESULT_DELIVERY_TIMEOUT = TRANSFER_TTL
# Bounded linger after delivery so a lost terminal reply can still be replayed (contract).
_RESULT_DELIVERY_LINGER = 5.0


class TrainerSessionError(Exception):
    """The trainer's Client API session ended (SHUTDOWN, ABORT, or CJ/cell loss)."""


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

        self._rank: Optional[str] = None
        self._is_control_rank = False
        self._cell: Optional[Cell] = None
        self._session_id: Optional[str] = None
        self._cj_fqcn: str = self._config[BootstrapKey.CJ_FQCN]
        self._trainer_fqcn: str = self._config[BootstrapKey.TRAINER_FQCN]
        self._job_id: str = self._config[BootstrapKey.JOB_ID]
        self._site_name: str = self._config[BootstrapKey.SITE_NAME]
        self._task_exchange: dict = self._config.get(BootstrapKey.TASK_EXCHANGE, {})

        # control state. _task_queue carries MATERIALIZED tasks: the TASK_READY handler
        # starts an eager payload pull on a materializer thread, which enqueues
        # {task, model, error} once the pull settles; receive() only dequeues.
        self._task_queue: "Queue[dict]" = Queue()
        self._current_task: Optional[dict] = None
        self._accepted_task_id: Optional[str] = None  # TASK_READY redelivery idempotency (by task id)
        self._fl_model: Optional[FLModel] = None
        self._receive_called = False
        self._abort = False
        self._abort_reason = ""
        # observed by the payload pulls between chunk requests, so an abort/stop ends an
        # in-flight materialization instead of letting it run to completion for nothing
        self._abort_signal = Signal()
        self._stopped = False
        self._closed = False
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ lifecycle

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
        register_framework_decomposers(self.logger)

        connect_url = self._config[BootstrapKey.CONNECT_URL]
        self._cell = Cell(
            fqcn=self._trainer_fqcn,
            root_url=None,
            secure=False,  # V1 trusted-host: the CJ's internal listener is a local connection
            credentials={},
            parent_url=connect_url,
            create_internal_listener=False,
        )
        self._register_control_cbs(self._cell)
        self._cell.start()
        try:
            self._hello()
        except Exception:
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
                        MsgKey.PROTOCOL_VERSION: self._config.get(BootstrapKey.PROTOCOL_VERSION, PROTOCOL_VERSION),
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

    # ------------------------------------------------------------------ receive

    def receive(self, timeout: Optional[float] = None) -> Optional[FLModel]:
        if not self._is_control_rank or self._closed:
            return None
        if self._fl_model is not None:
            return self._fl_model

        entry = self._await_task(timeout)
        if entry is None:
            return None

        task = entry["task"]
        self._current_task = task
        error = entry["error"]
        if error is not None:
            # the materializer already sent TASK_FAILED to the CJ; surface it to user code
            raise TrainerSessionError(f"failed to materialize task '{task.get(MsgKey.TASK_NAME)}': {error}")

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

    def _materialize_and_enqueue(self, task: dict) -> None:
        """Body of the per-task materializer thread the TASK_READY handler starts.

        Pulls the payload eagerly, reports the payload state to the CJ (TASK_PAYLOAD_READY
        or TASK_FAILED), and queues the settled entry for receive(). Must not raise: this
        is a thread target, and an escaping exception would silently strand receive()."""
        try:
            model = self._materialize_task(task)
        except Exception as e:
            # PayloadTransferError, or anything unexpected: either way this task is dead.
            # TASK_FAILED tells the CJ (which fails the task there); the queued error entry
            # surfaces it to user code blocked in receive().
            reason = f"task payload download failed: {e}"
            self.logger.error(reason)
            self._send_task_failed(task, reason)
            self._task_queue.put({"task": task, "model": None, "error": str(e)})
            return
        self._send_task_payload_ready(task)
        self._task_queue.put({"task": task, "model": model, "error": None})

    def _materialize_task(self, task: dict) -> FLModel:
        model_ref = task.get(MsgKey.MODEL) or {}
        ref_ids = model_ref.get(MsgKey.REF_IDS) or []
        if not ref_ids:
            # inline / empty payload: an empty global model is valid (e.g. first round)
            return FLModelUtils.from_shareable(Shareable())
        objs = fetch_result_payload(self._cell, self._cj_fqcn, ref_ids, abort_signal=self._abort_signal)
        shareable = objs[0] if len(objs) == 1 else None
        if not isinstance(shareable, Shareable):
            raise PayloadTransferError(f"expected one Shareable task payload but got {[type(o) for o in objs]}")
        return FLModelUtils.from_shareable(shareable)

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

        shareable = FLModelUtils.to_shareable(model)
        transfer_id = uuid.uuid4().hex
        result_id = uuid.uuid4().hex
        attempt = TaskPayloadAttempt(self._cell, shareable, self._cj_fqcn)
        try:
            reply = self._cell.send_request(
                channel=CHANNEL,
                topic=Topic.RESULT_READY,
                target=self._cj_fqcn,
                request=new_cell_message(
                    {},
                    {
                        MsgKey.SESSION_ID: self._session_id,
                        MsgKey.TASK_ID: task.get(MsgKey.TASK_ID),
                        MsgKey.RESULT_ID: result_id,
                        MsgKey.TRANSFER_ID: transfer_id,
                        MsgKey.MANIFEST: {MsgKey.REF_IDS: [attempt.ref_id]},
                    },
                ),
                timeout=_HELLO_TIMEOUT,
            )
            self._check_result_accepted(reply)

            # producer-liveness rule: hold the result payload alive until the CJ's pull
            # reaches its terminal outcome (returns == delivered). None arm => not delivered.
            delivered = attempt.wait(timeout=_RESULT_DELIVERY_TIMEOUT, linger=_RESULT_DELIVERY_LINGER)
            if not delivered:
                raise TrainerSessionError("result was not certified delivered by the CJ")
        finally:
            attempt.terminate()
            if clear_cache:
                self._fl_model = None
                self._receive_called = False
                self._current_task = None
            self._maybe_cleanup_memory()

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
        return dict(self._task_exchange)

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
        if not self._is_control_rank or self._closed or self._abort or self._stopped:
            return False
        try:
            return self.receive() is not None
        except TrainerSessionError:
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
        if self._closed:
            return
        self._closed = True
        self._stopped = True
        self._abort_signal.trigger("client api shutdown")
        try:
            if self._cell is not None and self._session_id is not None:
                self._cell.fire_and_forget(
                    channel=CHANNEL,
                    topic=Topic.BYE,
                    targets=[self._cj_fqcn],
                    message=new_cell_message({}, {MsgKey.SESSION_ID: self._session_id}),
                    optional=True,
                )
        except Exception as e:
            self.logger.debug(f"failed to send BYE: {e}")
        self._stop_cell()

    # ------------------------------------------------------------------ control handlers

    def _handle_task_ready(self, request):
        payload = request.payload
        if not isinstance(payload, dict):
            return make_cell_reply(CellReturnCode.INVALID_REQUEST, error="TASK_READY payload must be a dict")
        if payload.get(MsgKey.SESSION_ID) != self._session_id:
            return self._reply(Topic.TASK_FAILED, **{MsgKey.REASON: "stale or unknown session id"})
        task_id = payload.get(MsgKey.TASK_ID)
        with self._lock:
            # TASK_READY redelivery is idempotent by task id (control protocol): a retry of
            # an already-accepted task is re-acked without starting a second pull
            is_new = self._accepted_task_id != task_id
            if is_new:
                self._accepted_task_id = task_id
        if is_new:
            # eager materialization off the cell callback thread: the ack below must not
            # wait on the payload pull (see the module docstring)
            threading.Thread(
                target=self._materialize_and_enqueue,
                args=(payload,),
                name="client_api_task_materializer",
                daemon=True,
            ).start()
        return self._reply(Topic.TASK_ACCEPTED, **{MsgKey.TASK_ID: task_id})

    def _handle_abort(self, request):
        payload = request.payload if isinstance(request.payload, dict) else {}
        if payload.get(MsgKey.SESSION_ID) == self._session_id:
            self._abort = True
            self._abort_reason = str(payload.get(MsgKey.REASON))
            self._abort_signal.trigger(self._abort_reason)
            self.logger.error(f"session aborted by CJ: {self._abort_reason}")
        return make_cell_reply(CellReturnCode.OK)

    def _handle_shutdown(self, request):
        payload = request.payload if isinstance(request.payload, dict) else {}
        if payload.get(MsgKey.SESSION_ID) == self._session_id:
            self._stopped = True
            # a materialization still in flight serves no one after an orderly stop either:
            # receive() will return None and drop the entry
            self._abort_signal.trigger("session shutdown")
            self.logger.info("session shutdown requested by CJ")
        return make_cell_reply(CellReturnCode.OK)

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _reply(reply_topic: str, **fields):
        body = {MsgKey.REPLY_TOPIC: reply_topic}
        body.update(fields)
        return make_cell_reply(CellReturnCode.OK, body=body)

    def _send_task_payload_ready(self, task: dict) -> None:
        try:
            self._cell.fire_and_forget(
                channel=CHANNEL,
                topic=Topic.TASK_PAYLOAD_READY,
                targets=[self._cj_fqcn],
                message=new_cell_message(
                    {}, {MsgKey.SESSION_ID: self._session_id, MsgKey.TASK_ID: task.get(MsgKey.TASK_ID)}
                ),
                optional=True,
            )
        except Exception as e:
            self.logger.debug(f"failed to send TASK_PAYLOAD_READY: {e}")

    def _send_task_failed(self, task: dict, reason: str) -> None:
        try:
            self._cell.fire_and_forget(
                channel=CHANNEL,
                topic=Topic.TASK_FAILED,
                targets=[self._cj_fqcn],
                message=new_cell_message(
                    {},
                    {
                        MsgKey.SESSION_ID: self._session_id,
                        MsgKey.TASK_ID: task.get(MsgKey.TASK_ID),
                        MsgKey.REASON: reason,
                    },
                ),
                optional=True,
            )
        except Exception as e:
            self.logger.debug(f"failed to send TASK_FAILED: {e}")

    def _check_session_alive(self) -> None:
        if self._abort:
            raise TrainerSessionError(f"session aborted: {self._abort_reason}")
        if self._stopped or self._closed:
            raise TrainerSessionError("session stopped")

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
