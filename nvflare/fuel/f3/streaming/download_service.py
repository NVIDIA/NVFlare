# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
import dataclasses
import functools
import threading
import time
import uuid
import weakref
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional, Tuple

from nvflare.apis.fl_constant import SystemConfigs
from nvflare.apis.signal import Signal
from nvflare.fuel.f3.cellnet.cell import Cell
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey, ReturnCode
from nvflare.fuel.f3.cellnet.utils import make_reply, new_cell_message
from nvflare.fuel.f3.message import Message
from nvflare.fuel.f3.streaming.transfer_outcome import (  # noqa: F401 (re-exported legacy names)
    DownloadStatus,
    RefOutcome,
    TransactionDoneStatus,
    TransferOutcome,
    compute_transfer_outcome,
    terminal_state_for_done_status,
)
from nvflare.fuel.f3.streaming.transfer_progress import TransferProgressState
from nvflare.fuel.utils.app_config_utils import get_positive_float_var
from nvflare.fuel.utils.config_service import ConfigService
from nvflare.fuel.utils.log_utils import get_obj_logger
from nvflare.fuel.utils.validation_utils import check_positive_number
from nvflare.security.logging import secure_format_exception

OBJ_DOWNLOADER_CHANNEL = "download_service__"
OBJ_DOWNLOADER_TOPIC = "download_service__download"

"""
This package provides a framework for building object downloading capability (file download, tensor download, etc.).

A large object takes a lot of memory space. Sending a large object in one message needs even more memory space since
the object needs to be serialized into large number of bytes. Additional memory space may still be needed for the
transport layer to send the message. If the message is to be sent to multiple endpoints, even more memory is needed.

Object Downloading can drastically reduce memory consumption:
- Instead of sending the large object in one message, it is divided into many smaller objects;
- Instead of pushing the message to the endpoints, each endpoint will come to request. This makes it more reliable when
different endpoints have different speed. 

Object Downloading works as follows:
- The sender prepares the object(s) for downloading. It first creates a transaction to get a tx_id. It then adds each
object (called Downloadable) to be downloaded to the transaction, and get a reference id (ref_id).
- The sender sends the ref_id(s) to all recipients through a separate message.
- Each recipient then calls the download_object function to download each referenced large object.

Note that the endpoint that received object refs may forward the refs to another endpoint, which then downloads the
referenced object(s).

To develop the downloading capability for a type of object (e.g. a file, a tensor state dict, etc.), you need to provide
the implementation of a Downloadable and a Consumer.
- On the sending side, the Downloadable is responsible for producing the next small object to be sent (a chunk of bytes;
a small subset of the large dict; etc.).
- On the receiving side, the Consumer is responsible for processing the received small objects (writing the received
bytes to a temp file; putting the received small dict to the end result; etc.).

One issue with object downloading is object life cycle management. Since the large objects to be downloaded are usually
temporary, you need to remove them when they are downloaded by all receivers. But the problem is that you don't know how
quickly each receiver can finish downloading these large objects. When a transaction contains multiple objects to be
downloaded, it's even harder to know it.

There are two ways to handle this issue: object downloaded callback, and transaction timeout.

You can implement the downloaded_to_one method for the Downloadable object. This method is called when the object is
downloaded to one receiver.

You can also implement the downloaded_to_all method for the Downloadable object. This method is called when the object 
is downloaded to all receivers.

Note that the downloaded_to_all method only works if you know how many receivers the object will be downloaded to!

You can always implement the transaction_done method for the Downloadable object. This method is called when the
transaction is done for some reason (normal completion or timeout).

Transaction timeout is the amount of time after the last downloading activity on any object in the
transaction from any receiver. For example, suppose you want to send 2 large files to 3 receivers, each time a download
request is received on any file from any of the 3 receivers, the last activity time of the transaction is updated to now.
If no downloading activity is received from any receiver on any objects in the transaction for the specified timeout,
the transaction is considered "timed out", and the transaction_done method is called for each Downloadable object 
added to the transaction.

Unlike with Object Streamer that the object owner pushes small objects to the recipients; with Object Downloader,
each recipient pulls the data from the object owner.
"""


class Downloadable(ABC):

    def __init__(self, obj: Any):
        self.base_obj = obj

    def set_transaction(self, tx_id: str, ref_id: str):
        """This method is called when the object is added to a transaction.
        You can use this method to keep transaction ID and/or ref ID for your own purpose.

        Args:
            tx_id: the ID of the transaction that the object has been added to.
            ref_id: ref ID generated for the object.

        Returns: None

        """
        pass

    @abstractmethod
    def produce(self, state: dict, requester: str) -> Tuple[str, Any, dict]:
        """Produce a small object to be sent (on object sender side).

        Args:
            state: current state of downloading, received from the downloading receiver
            requester: the FQCN of the receiver that is downloading

        Returns: a tuple of (return code, a small object to be sent, new state to be sent).

        """
        pass

    def downloaded_to_one(self, to_receiver: str, status: str):
        """Called when an object is downloaded to a receiver.

        Args:
            to_receiver: name of the receiver that the object has been completely downloaded to.
            status: the download status: DownloadStatus.SUCCESS or DownloadStatus.FAILED.

        Returns: None

        """
        pass

    def downloaded_to_all(self):
        """Called when the object is fully downloaded to all receivers."""
        pass

    def transaction_done(self, transaction_id: str, status: str):
        """Called when the transaction is finished.

        Args:
            transaction_id: ID of the transaction.
            status: completion status, a value defined in TransactionDoneStatus.

        Returns: None

        """
        pass

    def release(self):
        """Drop the infrastructure reference to the source object.

        Called by _Transaction.transaction_done() AFTER the transaction_done_cb
        fires.  Subclasses should override this to null their base_obj (or any
        other large reference) so the GC can reclaim the memory immediately.
        The default implementation is a no-op.
        """
        pass


class _PropKey:
    REF_ID = "ref_id"
    STATE = "state"
    DATA = "data"
    STATUS = "status"
    # Receiver-confirmed completion. All three keys are OPTIONAL on the wire so both
    # version skews interop with legacy peers: an old receiver never sends CONFIRM_CAPABLE and
    # gets today's producer-served semantics; an old producer never sends CONFIRM_EXPECTED so a
    # new receiver never confirms toward it.
    CONFIRM = "confirm"  # receiver -> producer: terminal receiver truth (a DownloadStatus value)
    CONFIRM_NONCE = "confirm_nonce"  # both ways: per-serve nonce binding a confirmation to ITS serve
    CONFIRM_CAPABLE = "confirm_capable"  # receiver -> producer, per request: will confirm if asked
    CONFIRM_EXPECTED = "confirm_expected"  # producer -> receiver, per reply: confirmations consumed


# Per-process kill-switch for receiver-confirmed completion (read once at first use; set the
# config var / env before process start and restart to change it). The wire behavior is doubly
# gated -- per-message capability advertisement AND this switch on each side -- so a field issue
# in a mixed-version fleet is mitigated by configuration + restart without a code revert.
RECEIVER_CONFIRM_CONFIG_VAR = "streaming_receiver_confirm_enabled"
_receiver_confirm_cached = None


def _receiver_confirm_enabled() -> bool:
    global _receiver_confirm_cached
    if _receiver_confirm_cached is None:
        try:
            _receiver_confirm_cached = bool(
                ConfigService.get_bool_var(
                    RECEIVER_CONFIRM_CONFIG_VAR, conf=SystemConfigs.APPLICATION_CONF, default=True
                )
            )
        except Exception:
            # unconfigured environments (e.g. bare unit tests) default to enabled
            _receiver_confirm_cached = True
    return _receiver_confirm_cached


# Per-(transfer, receiver) budgets. System defaults resolved from config vars; explicit
# per-transaction values win. None (unset everywhere) disables enforcement for that budget --
# the whole-transaction timeout then remains the only backstop, exactly today's behavior.
RECEIVER_ACQUIRE_TIMEOUT_CONFIG_VAR = "streaming_receiver_acquire_timeout"
RECEIVER_IDLE_TIMEOUT_CONFIG_VAR = "streaming_receiver_idle_timeout"

# How long settlement waits for in-flight operations to drain. Only a hung user
# callback can exhaust it; settlement then proceeds with a warning, and the id
# stays excluded until the leaked operation exits (see _leaked_ops).
OP_DRAIN_TIMEOUT = 60.0


def _resolve_receiver_budget(explicit, var_name: str):
    if explicit is not None:
        check_positive_number(var_name, explicit)
        return float(explicit)
    return get_positive_float_var(var_name, default=None)


class _Ref:

    def __init__(
        self,
        tx,
        obj: Downloadable,
        ref_id=None,
    ):
        if ref_id:
            # use provided ref_id
            self.rid = ref_id
        else:
            self.rid = "R" + str(uuid.uuid4())
        self.tx = tx
        self.obj = obj
        self.num_receivers_done = 0
        self.receiver_statuses = {}
        # producer-served terminal statuses awaiting the receiver's confirmation; only
        # finalized (confirmed or legacy-served) statuses live in receiver_statuses
        self._pending_confirms = {}
        # unconditional per-receiver liveness: receiver -> last activity timestamp,
        # updated on every request regardless of whether a progress_cb is configured -- so a
        # live receiver can no longer mask a stalled one behind the tx-wide last_active_time
        self._receiver_activity = {}
        self._created_time = time.time()
        self._downloaded_to_all_called = False
        self._receiver_progress = {}
        self._terminal_progress_state = None
        self._progress_lock = threading.Lock()

    def mark_active(self):
        self.tx.mark_active()

    def obj_downloaded(self, to_receiver: str, status: str):
        self._finalize_receiver(to_receiver, status)

    def _finalize_receiver(
        self, to_receiver: str, status: str, require_pending: bool = False, nonce: Optional[str] = None
    ) -> bool:
        # Recording is guarded so outcome snapshots never observe a half-updated map;
        # user callbacks run outside the lock. Dedup, pending-guard, pop, record, and
        # the all-done latch are ONE critical section, so a duplicate serve can never
        # resurrect a pending entry around a racing finalization.
        with self._progress_lock:
            if to_receiver in self.receiver_statuses:
                return False
            if require_pending:
                pending = self._pending_confirms.get(to_receiver)
                if pending is None or pending[1] != nonce:
                    # a legitimate confirmation always follows a provisional terminal serve on
                    # the CURRENT life of this ref and echoes that serve's nonce; anything else
                    # (unsolicited, or delayed across a ref_id reuse) must not certify -- or
                    # poison -- this transfer
                    self.tx.logger.warning(
                        f"dropping unsolicited/stale confirmation from {to_receiver} for ref {self.rid}"
                    )
                    return False
            self._pending_confirms.pop(to_receiver, None)

            self.receiver_statuses[to_receiver] = status
            self.num_receivers_done = len(self.receiver_statuses)

            assert isinstance(self.tx, _Transaction)
            all_done = not self._downloaded_to_all_called and self._completion_reached_locked()
            if all_done:
                self._downloaded_to_all_called = True

        # Guarded like the terminal callbacks in transaction_done: a raising user
        # callback on the serving path must not lose the EOF reply for this attempt,
        # and a raising downloaded_to_one must not skip downloaded_to_all (the
        # _downloaded_to_all_called latch above is already set and is never retried).
        assert isinstance(self.obj, Downloadable)
        _invoke_cb_safely(
            self.tx.logger,
            f"downloaded_to_one of {type(self.obj)} for ref {self.rid}",
            self.obj.downloaded_to_one,
            to_receiver,
            status,
        )

        if all_done:
            # this object is done for all receivers
            _invoke_cb_safely(
                self.tx.logger,
                f"downloaded_to_all of {type(self.obj)} for ref {self.rid}",
                self.obj.downloaded_to_all,
            )
        return True

    def _completion_reached_locked(self) -> bool:
        # Identity-aware when the transaction declared receiver_ids: completion means every
        # EXPECTED receiver is final. A status from an unexpected receiver never completes
        # the ref -- otherwise (expected "b" absent + unexpected "x" present) would certify
        # a delivery "b" never got. Count-based only when identities are unknown.
        expected = self.tx.receiver_ids
        if expected:
            return all(r in self.receiver_statuses for r in expected)
        return 0 < self.tx.num_receivers <= self.num_receivers_done

    def obj_served(self, to_receiver: str, status: str, expect_confirm: bool):
        """Records the producer-served terminal status for a receiver.

        Legacy receivers (expect_confirm=False) finalize immediately: served EOF/ERROR is the
        only truth available. Confirm-capable receivers are recorded as PROVISIONAL only -- the
        receiver's confirmation (obj_confirmed) finalizes them. This is what makes accounting
        retry-aware: while the record is provisional, a later serve for the same receiver
        overwrites it -- a lost terminal reply healed by a retry is not stuck at the first
        served status -- and the confirmation supersedes any provisional state (a receiver-side
        finalization failure after the last chunk turns a served-EOF SUCCESS into a confirmed
        FAILED). Once the receiver confirms, its status is final: a receiver that confirms
        FAILED has given up (it confirms only on its own terminal exits).

        Returns the per-serve nonce the confirmation must echo (None when finalized
        immediately or already final). The nonce binds a confirmation to THIS serve of
        THIS life of the ref: a stale confirmation from a previous life of a reused
        ref_id carries the wrong nonce and is dropped even if the new life has its own
        pending serve for the same receiver.
        """
        if not expect_confirm:
            self.obj_downloaded(to_receiver, status)
            return None
        nonce = uuid.uuid4().hex
        with self._progress_lock:
            if to_receiver in self.receiver_statuses:
                # already finalized -- a late duplicate serve must not resurrect a provisional
                return None
            self._pending_confirms[to_receiver] = (status, nonce)
        return nonce

    def obj_confirmed(self, to_receiver: str, status: str, nonce: Optional[str]) -> bool:
        """Records the receiver-confirmed terminal status. Receiver truth wins; first confirm is final.

        Accepted only when a provisional serve is pending for this receiver AND the
        confirmation echoes that serve's nonce. The nonce is what distinguishes ref
        lives: without it, a delayed confirmation from a previous life of a reused
        ref_id would be accepted whenever the new life happens to have its own pending
        serve for the same receiver -- certifying (or poisoning) a transfer it never saw.
        """
        if status not in (DownloadStatus.SUCCESS, DownloadStatus.FAILED):
            self.tx.logger.error(f"ignoring confirmation with invalid status '{status}' from {to_receiver}")
            return False
        accepted = self._finalize_receiver(to_receiver, status, require_pending=True, nonce=nonce)
        if accepted:
            # the receiver's truth is the terminal progress state for this receiver
            self.emit_progress(
                receiver_id=to_receiver,
                state=(
                    TransferProgressState.COMPLETED
                    if status == DownloadStatus.SUCCESS
                    else TransferProgressState.FAILED
                ),
                force=True,
            )
        return accepted

    def snapshot_receiver_statuses(self) -> dict:
        with self._progress_lock:
            return dict(self.receiver_statuses)

    def snapshot_pending_confirms(self) -> dict:
        with self._progress_lock:
            # public shape: receiver -> provisional status (the nonce is internal)
            return {r: v[0] for r, v in self._pending_confirms.items()}

    def mark_receiver_active(self, receiver: str):
        now = time.time()
        with self._progress_lock:
            self._receiver_activity[receiver] = now
        tx = self.tx
        with tx._stats_lock:
            # tx-level activity: budgets judge the whole transaction (see the
            # _receiver_last_active field comment)
            tx._acquired_receivers.add(receiver)
            tx._receiver_last_active[receiver] = now

    def snapshot_receiver_activity(self) -> dict:
        with self._progress_lock:
            return dict(self._receiver_activity)

    def enforce_budgets(
        self, now: float, acquire_timeout, idle_timeout, expected_receivers, tx_acquired=None, tx_last_active=None
    ) -> list:
        """Finalizes FAILED for receivers whose acquire or idle budget is exhausted.

        Candidates are the declared receiver_ids plus every receiver seen anywhere on the
        transaction. Idleness is judged on TRANSACTION-level activity (last request on ANY
        ref): a receiver that pulled a sibling ref and went silent has no per-ref timestamp
        here, and transaction-level acquisition exempts it from the acquire budget -- judging
        idle per-ref would let it escape both budgets and pin the producer to the full TTL.

        A budget failure counts toward completion (via _finalize_receiver), so the aggregate
        outcome resolves on the next monitor pass. This also bounds a lost fire-and-forget
        confirmation (fail-closed).

        Returns: list of (receiver, reason) that were failed on this pass.
        """
        tx_acquired = tx_acquired or set()
        tx_last_active = tx_last_active or {}
        with self._progress_lock:
            final = set(self.receiver_statuses)
        candidates = set(expected_receivers or ()) | tx_acquired
        failures = []
        for receiver in candidates:
            if receiver in final:
                continue
            last_active = tx_last_active.get(receiver)
            if last_active is None:
                # never pulled anywhere: only the acquire budget (needs declared identities)
                if acquire_timeout is not None and expected_receivers and receiver in expected_receivers:
                    waited = now - self._created_time
                    if waited > acquire_timeout:
                        failures.append(
                            (
                                receiver,
                                f"acquire budget exhausted: no pull within {acquire_timeout}s (waited {waited:.1f}s)",
                            )
                        )
            elif idle_timeout is not None:
                idle = now - last_active
                if idle > idle_timeout:
                    failures.append(
                        (
                            receiver,
                            f"idle budget exhausted: {idle:.1f}s since last transaction activity > {idle_timeout}s",
                        )
                    )
        enforced = []
        for receiver, reason in failures:
            with self.tx._stats_lock:
                if self.tx._receiver_last_active.get(receiver) != tx_last_active.get(receiver):
                    continue  # activity advanced past the snapshot: not actually idle
            # _finalize_receiver dedups and pops the pending-confirm entry itself;
            # a receiver finalized meanwhile (e.g. confirmed) wins -- truth over budget
            if not self._finalize_receiver(receiver, DownloadStatus.FAILED):
                continue
            self.tx.logger.warning(f"receiver {receiver} failed for ref {self.rid}: {reason}")
            self.emit_progress(receiver_id=receiver, state=TransferProgressState.FAILED, force=True)
            enforced.append((receiver, reason))
        return enforced

    def emit_progress(
        self,
        *,
        receiver_id: Optional[str],
        state: str = TransferProgressState.ACTIVE,
        bytes_delta: int = 0,
        items_delta: Optional[int] = None,
        force: bool = False,
    ):
        if not self.tx.progress_cb:
            return

        now = time.time()
        with self._progress_lock:
            event = self._make_progress_event_locked(
                receiver_id=receiver_id,
                state=state,
                bytes_delta=bytes_delta,
                items_delta=items_delta,
                force=force,
                timestamp=now,
            )
        if not event:
            return

        self.tx.emit_progress_event(event)

    def emit_terminal_progress_for_started_receivers(self, state: str):
        if not self.tx.progress_cb:
            return

        now = time.time()
        with self._progress_lock:
            self._terminal_progress_state = state
            receiver_ids = list(self._receiver_progress)
            events = [
                self._make_progress_event_locked(
                    receiver_id=receiver_id,
                    state=state,
                    force=True,
                    timestamp=now,
                )
                for receiver_id in receiver_ids
            ]

        for event in events:
            if event:
                self.tx.emit_progress_event(event)

    def _make_progress_event_locked(
        self,
        *,
        receiver_id: Optional[str],
        state: str,
        timestamp: float,
        bytes_delta: int = 0,
        items_delta: Optional[int] = None,
        force: bool = False,
    ):
        if self._terminal_progress_state and state not in TransferProgressState.TERMINAL_STATES:
            state = self._terminal_progress_state
            force = True
            bytes_delta = 0
            items_delta = None

        receiver_progress = self._receiver_progress.get(receiver_id)
        if receiver_progress is None:
            receiver_progress = _ReceiverProgress()
            self._receiver_progress[receiver_id] = receiver_progress

        if receiver_progress.terminal:
            return None

        first_emit = not receiver_progress.started
        if first_emit:
            receiver_progress.started = True

        if bytes_delta > 0:
            receiver_progress.bytes_done += bytes_delta
        if items_delta is not None and items_delta > 0:
            receiver_progress.items_done = (receiver_progress.items_done or 0) + items_delta

        counters_advanced = bytes_delta > 0 or (items_delta is not None and items_delta > 0)
        terminal = state in TransferProgressState.TERMINAL_STATES
        if (
            not force
            and not first_emit
            and not terminal
            and (not counters_advanced or timestamp - receiver_progress.last_emit_time < self.tx.progress_interval)
        ):
            return None

        receiver_progress.sequence += 1
        receiver_progress.last_emit_time = timestamp
        if terminal:
            receiver_progress.terminal = True

        return {
            "tx_id": self.tx.tid,
            "ref_id": self.rid,
            "receiver_id": receiver_id,
            "sequence": receiver_progress.sequence,
            "bytes_done": receiver_progress.bytes_done,
            "items_done": receiver_progress.items_done,
            "timestamp": timestamp,
            "state": state,
        }


class _ReceiverProgress:

    def __init__(self):
        self.sequence = 0
        self.bytes_done = 0
        self.items_done = None
        self.started = False
        self.terminal = False
        self.last_emit_time = 0.0


class ProduceRC:
    """Defines return code for the Downloadable object's 'produce' method."""

    OK = "ok"
    ERROR = "error"
    EOF = "eof"


def _invoke_cb_safely(logger, what: str, cb, *args, **kwargs):
    """Invoke a user callback without letting its exception escape.

    Termination callbacks run on the transaction monitor thread; a propagating
    exception would kill that thread and stop all future transactions from
    finishing or expiring, and would skip outcome recording and source release.
    """
    try:
        cb(*args, **kwargs)
    except Exception as ex:
        logger.warning(f"{what} failed: {secure_format_exception(ex)}")


class _FinishedRef:

    def __init__(self, receiver_statuses: dict[str, str], timestamp: float):
        self.receiver_statuses = receiver_statuses
        self.last_active_time = timestamp

    def expired(self, now: float, ttl: float) -> bool:
        return now - self.last_active_time > ttl


class _Transaction:

    def __init__(
        self,
        timeout: float,
        num_receivers: int,
        tx_id=None,
        transaction_done_cb=None,
        cb_kwargs=None,
        progress_cb: Optional[Callable] = None,
        progress_interval: float = 30.0,
        outcome_cb: Optional[Callable] = None,
        receiver_ids=None,
        min_receivers: Optional[int] = None,
        receiver_acquire_timeout: Optional[float] = None,
        receiver_idle_timeout: Optional[float] = None,
    ):
        """Constructor of the transaction object.

        Args:
            timeout: amount of time since last activity
            num_receivers: number of receivers. 0 means unknown/unbounded: such a
                transaction is never certified finished (is_finished() returns False) —
                it terminates via timeout or deletion, and its aggregate outcome can
                never be COMPLETED (all-receivers-success cannot be certified).
            tx_id: if provided, use it; otherwise create one
            outcome_cb: called with the aggregate TransferOutcome after transaction_done_cb fires
        """
        if tx_id:
            self.tid = tx_id
        else:
            self.tid = "T" + str(uuid.uuid4())
        self.timeout = timeout
        self.logger = get_obj_logger(self)

        # Expected receiver identities. Optional: when provided they enable the acquire
        # budget (a receiver that never issues its first pull can be failed) and, if
        # num_receivers is unknown (0), supply the receiver count.
        if receiver_ids:
            given = len(receiver_ids)
            receiver_ids = tuple(dict.fromkeys(str(r) for r in receiver_ids))  # dedup, keep order
            if len(receiver_ids) != given:
                # almost certainly a caller bug: it believes there are more distinct receivers
                self.logger.warning(
                    f"tx {self.tid}: duplicate receiver_ids deduplicated ({given} -> {len(receiver_ids)})"
                )
            if num_receivers and num_receivers != len(receiver_ids):
                raise ValueError(
                    f"num_receivers ({num_receivers}) does not match receiver_ids count ({len(receiver_ids)})"
                )
            num_receivers = len(receiver_ids)
            self.receiver_ids = receiver_ids
        else:
            self.receiver_ids = None
        if min_receivers is not None:
            if min_receivers <= 0:
                raise ValueError(f"min_receivers must be positive, got {min_receivers}")
            if num_receivers and min_receivers > num_receivers:
                raise ValueError(f"min_receivers ({min_receivers}) exceeds num_receivers ({num_receivers})")
        self.min_receivers = min_receivers
        self.receiver_acquire_timeout = _resolve_receiver_budget(
            receiver_acquire_timeout, RECEIVER_ACQUIRE_TIMEOUT_CONFIG_VAR
        )
        self.receiver_idle_timeout = _resolve_receiver_budget(receiver_idle_timeout, RECEIVER_IDLE_TIMEOUT_CONFIG_VAR)
        self.num_receivers = num_receivers
        self.last_active_time = time.time()
        self.start_time = time.time()
        self.total_bytes = 0
        self._stats_lock = threading.Lock()
        self.transaction_done_cb = transaction_done_cb
        self.cb_kwargs = cb_kwargs or {}
        self.outcome_cb = outcome_cb
        self.progress_cb = progress_cb
        if progress_interval < 0:
            raise ValueError(f"progress_interval must be non-negative, got {progress_interval}")
        self.progress_interval = float(progress_interval)
        self.refs = []
        self._refs_lock = threading.RLock()
        # The activity gate: serves, confirms, and monitor budget passes register as
        # in-flight operations; settlement closes and drains the gate before the
        # outcome snapshot, so late finishes are counted and nothing emits against a
        # settled transaction.
        self._ops_cond = threading.Condition(threading.Lock())
        self._active_ops = 0
        self._ops_closed = False
        # set in transaction_done's finally. A _leaked_ops marker is releasable only
        # when this is True AND no operations are in flight: "no ops" alone does not
        # mean quiet -- settlement callbacks can still be emitting.
        self._settlement_complete = False
        # receivers that have issued at least one pull on ANY ref (monotonic; the
        # transaction-level PAYLOAD_ACQUIRED fact the acquire budget and the facade read),
        # and each receiver's last activity anywhere on the transaction (what the idle
        # budget judges -- per-ref timestamps would let a multi-ref receiver escape)
        self._acquired_receivers = set()
        self._receiver_last_active = {}

    def mark_active(self):
        self.last_active_time = time.time()

    def add_total_bytes(self, byte_count: int):
        if byte_count <= 0:
            return
        with self._stats_lock:
            self.total_bytes += byte_count

    def get_total_bytes(self) -> int:
        with self._stats_lock:
            return self.total_bytes

    def add_object(
        self,
        obj: Downloadable,
        ref_id=None,
    ):
        """Adds a large object (to be downloaded) to the transaction; returns its ref."""
        with self._refs_lock:
            r = _Ref(self, obj, ref_id)
            self.refs.append(r)
            obj.set_transaction(self.tid, r.rid)
        return r

    def snapshot_refs(self):
        with self._refs_lock:
            return list(self.refs)

    def begin_op(self) -> bool:
        """Registers an in-flight operation (serve, confirm, budget pass).

        Returns False if the transaction is settling or settled: the caller must treat
        it as gone (same as a missing ref). Every True return must be paired with
        end_op(), normally via try/finally.
        """
        with self._ops_cond:
            if self._ops_closed:
                return False
            self._active_ops += 1
            return True

    def end_op(self):
        with self._ops_cond:
            self._active_ops -= 1
            if self._active_ops <= 0:
                self._ops_cond.notify_all()

    def _drain_ops(self, timeout: float) -> bool:
        """Closes the activity gate and waits for in-flight operations to finish."""
        deadline = time.time() + timeout
        with self._ops_cond:
            self._ops_closed = True
            while self._active_ops > 0:
                remaining = deadline - time.time()
                if remaining <= 0:
                    return False
                self._ops_cond.wait(remaining)
        return True

    @property
    def has_receiver_budgets(self) -> bool:
        return self.receiver_acquire_timeout is not None or self.receiver_idle_timeout is not None

    def enforce_receiver_budgets(self, now: float):
        """Evaluates per-receiver budgets across all refs; called by the monitor thread."""
        if not self.has_receiver_budgets:
            return
        with self._stats_lock:
            tx_acquired = set(self._acquired_receivers)
            tx_last_active = dict(self._receiver_last_active)
        for ref in self.snapshot_refs():
            assert isinstance(ref, _Ref)
            ref.enforce_budgets(
                now,
                self.receiver_acquire_timeout,
                self.receiver_idle_timeout,
                self.receiver_ids,
                tx_acquired=tx_acquired,
                tx_last_active=tx_last_active,
            )

    def is_finished(self):
        """Check whether every expected receiver has a final status on every ref.

        Identity-aware when receiver_ids were declared: statuses from unexpected
        receivers never finish a ref. A transaction with no refs yet is never
        finished (it terminates via timeout or deletion instead).
        """
        if self.num_receivers <= 0:
            return False

        refs = self.snapshot_refs()
        if not refs:
            return False
        for ref in refs:
            assert isinstance(ref, _Ref)
            with ref._progress_lock:
                if not ref._completion_reached_locked():
                    return False
        return True

    def transaction_done(self, status: str, on_outcome=None) -> TransferOutcome:
        """Settles the transaction; returns the aggregate TransferOutcome.

        COMPLETED only when every expected receiver succeeded — FINISHED alone does
        not certify that. Callback exceptions never propagate (they would kill the
        monitor thread and skip source release). on_outcome (records the outcome,
        releasing waiters) is invoked LAST, so acting on waiter.wait() returning can
        never preempt the callback chain or source release. Runs exactly once per
        transaction: every terminator unlinks it from _tx_table first.
        """
        # drain the activity gate first: in-flight results count in the verdict
        if not self._drain_ops(OP_DRAIN_TIMEOUT):
            self.logger.warning(
                f"tx {self.tid}: in-flight operations did not drain within {OP_DRAIN_TIMEOUT}s; settling anyway"
            )
        outcome = None
        try:
            refs = self.snapshot_refs()

            # Compute the aggregate outcome from locked per-receiver snapshots before any
            # user callback can observe (or mutate the world around) this transaction.
            outcome = compute_transfer_outcome(
                tx_id=self.tid,
                done_status=status,
                num_receivers=self.num_receivers,
                min_receivers=self.min_receivers,
                receiver_ids=self.receiver_ids,
                refs=[RefOutcome(ref_id=ref.rid, receiver_statuses=ref.snapshot_receiver_statuses()) for ref in refs],
                timestamp=time.time(),
            )

            progress_state = self._progress_state_for_transaction_status(status)
            if progress_state:
                for ref in refs:
                    ref.emit_terminal_progress_for_started_receivers(progress_state)

            elapsed = time.time() - self.start_time
            total_bytes = self.get_total_bytes()
            size_mb = total_bytes / (1024 * 1024)
            self.logger.info(
                f"[server] download tx {self.tid} done: status={status} elapsed={elapsed:.2f}s "
                f"size={size_mb:.1f}MB ({total_bytes:,} bytes)"
            )

            # Snapshot base_objs BEFORE the loop so the callback receives the
            # original objects.  obj.transaction_done() may clear the chunk cache
            # (CacheableObject.clear_cache()); the source object itself is released
            # via obj.release() AFTER the callback so the callback can still
            # observe it (e.g. for memory-GC notifications).
            base_objs = [ref.obj.base_obj for ref in refs]

            for ref in refs:
                obj = ref.obj
                assert isinstance(obj, Downloadable)
                _invoke_cb_safely(
                    self.logger,
                    f"transaction_done of {type(obj)} for tx {self.tid}",
                    obj.transaction_done,
                    self.tid,
                    status,
                )

            if self.transaction_done_cb:
                _invoke_cb_safely(
                    self.logger,
                    f"transaction done callback for tx {self.tid}",
                    self.transaction_done_cb,
                    self.tid,
                    status,
                    base_objs,
                    **self.cb_kwargs,
                )

            if self.outcome_cb:
                _invoke_cb_safely(self.logger, f"transfer outcome callback for tx {self.tid}", self.outcome_cb, outcome)

            # release after the callback (so it can still observe the sources); each
            # release is guarded so one raising release() cannot skip its siblings
            for ref in refs:
                _invoke_cb_safely(self.logger, f"release of {type(ref.obj)} for tx {self.tid}", ref.obj.release)
        finally:
            # in a finally: no exception above may leave the outcome unrecorded and
            # waiters pending
            if on_outcome and outcome is not None:
                _invoke_cb_safely(self.logger, f"outcome recording for tx {self.tid}", on_outcome, outcome)
            self._settlement_complete = True

        return outcome

    def emit_progress_event(self, event: dict):
        if not self.progress_cb:
            return

        try:
            self.progress_cb(**event)
        except Exception as ex:
            self.logger.warning(
                f"download source progress callback failed for ref={event.get('ref_id')}: "
                f"{secure_format_exception(ex)}"
            )

    @staticmethod
    def _progress_state_for_transaction_status(status: str) -> Optional[str]:
        return terminal_state_for_done_status(status)


class TransactionInfo:
    """This structure contains public info of a transaction:
    timeout value of the transaction;
    number of receivers that objects in the transaction will be downloaded to. 0 means unknown/unbounded
    (the transaction is never certified finished and terminates via timeout or deletion);
    objects that are added to the transaction.
    """

    def __init__(self, tx: _Transaction):
        self.timeout = tx.timeout
        self.num_receivers = tx.num_receivers
        self.objects = [r.obj for r in tx.snapshot_refs()]


class TransferWaiter:
    """The awaitable facade over a transaction's terminal transfer outcome.

    This is the "returns == delivered" primitive the upper layers (executor backends,
    trainer engine) consume: wait() blocks -- event-driven, no polling -- until the
    transaction's aggregate TransferOutcome is recorded, and the outcome is COMPLETED only
    when every expected receiver succeeded (receiver-confirmed where supported, budget- and
    TTL-bounded). It attaches to the outcome-recording path directly, so it composes with --
    and never replaces -- transaction_done_cb / outcome_cb / the FOBS-context
    DOWNLOAD_COMPLETE_CB chain.
    """

    def __init__(self, transaction_id: str, service=None):
        self.transaction_id = transaction_id
        self._service = service  # the DownloadService class that created this waiter
        self._event = threading.Event()
        self._outcome: Optional[TransferOutcome] = None

    def _resolve(self, outcome: Optional[TransferOutcome]):
        self._outcome = outcome
        self._event.set()

    @property
    def outcome(self) -> Optional[TransferOutcome]:
        """The terminal outcome, or None while the transfer is still in flight."""
        return self._outcome

    def done(self) -> bool:
        return self._event.is_set()

    def acquired_receivers(self) -> set:
        """Receivers that have issued at least one pull (the PAYLOAD_ACQUIRED signal, V1)."""
        service = self._service if self._service is not None else DownloadService
        return service.get_acquired_receivers(self.transaction_id)

    def wait(self, timeout: Optional[float] = None, linger: Optional[float] = None) -> Optional[TransferOutcome]:
        """Blocks until the terminal transfer outcome is recorded.

        Args:
            timeout: max seconds to wait. None waits indefinitely (callers should normally
                bound this; the transaction's own TTL and per-receiver budgets bound the
                producer side).
            linger: optional bounded post-completion linger, applied after any FINISHED
                outcome (completed or not). By termination time the sources are already
                released and the refs tombstoned; what the linger preserves is the PROCESS
                (and with it the tombstone window), so a receiver whose terminal EOF/ERROR
                reply was lost can still retry and be replayed its recorded status before
                the producer exits. Timed-out/deleted outcomes get no linger.

        Returns: the TransferOutcome; None if the wait timed out (transfer still in flight)
        or the service shut down before the transaction terminated.
        """
        if not self._event.wait(timeout):
            return None
        outcome = self._outcome
        if outcome is not None and linger and outcome.done_status == TransactionDoneStatus.FINISHED:
            time.sleep(linger)
        return outcome


class DownloadService:

    _init_lock = threading.Lock()
    _tx_table = {}
    _ref_table = {}
    # Ref tombstones let a client retry a lost/delayed EOF reply after the source
    # transaction has been cleaned up without turning a completed transfer into a fatal missing-ref error.
    _finished_refs = {}
    FINISHED_REFS_TTL = 1800.0
    # Terminal outcomes of finished/expired/deleted transactions, kept for a bounded
    # time so producers can query the aggregate result after termination. Guarded by
    # its own lock so outcome polling never contends with the chunk-serving _tx_lock.
    _tx_outcomes = {}
    # The transaction entitled to record the outcome for its tx_id (registered by
    # new_transaction, consumed by _record_outcome; object-identity checked, so a
    # recorder whose entry is gone drops its stale outcome). Also marks the id as
    # in-use while the transaction is settling. Guarded by _outcome_lock.
    _outcome_owners = {}
    # Waiters blocked on a transaction's terminal outcome (the awaitable facade). Guarded by
    # _outcome_lock; resolved inside _record_outcome so a waiter can never miss the outcome.
    _tx_waiters = {}
    # Termination markers: installed by _delete_tx at unlink (every terminator) and
    # releasable only when the transaction's settlement completed AND no in-flight
    # operations remain -- the id stays excluded from registration through the whole
    # termination window and any drain-leaked tail, even past receipt expiry (a
    # leaked emission must never land under a recycled id). Released by
    # _update_leaked_ops, the registration check, or the monitor reap.
    # Guarded by _tx_lock.
    _leaked_ops = {}
    _outcome_lock = threading.Lock()
    TX_OUTCOME_TTL = 1800.0
    _logger = None
    _tx_monitor = None
    _tx_lock = threading.Lock()
    _initialized_cells = weakref.WeakKeyDictionary()

    @classmethod
    def _initialize(cls, cell: Cell):
        with cls._init_lock:
            if not cls._logger:
                cls._logger = get_obj_logger(cls)

            if not cls._tx_monitor:
                cls._tx_monitor = threading.Thread(target=cls._monitor_tx, daemon=True)
                cls._tx_monitor.start()

            initialized = cls._initialized_cells.get(cell)
            if not initialized:
                # register CBs
                cell.register_request_cb(
                    channel=OBJ_DOWNLOADER_CHANNEL,
                    topic=OBJ_DOWNLOADER_TOPIC,
                    cb=cls._handle_download,
                )
                cls._initialized_cells[cell] = True

    @classmethod
    def new_transaction(
        cls,
        cell: Cell,
        timeout: float,
        num_receivers: int = 0,
        tx_id=None,
        transaction_done_cb=None,
        progress_cb: Optional[Callable] = None,
        progress_interval: float = 30.0,
        outcome_cb: Optional[Callable] = None,
        receiver_ids=None,
        min_receivers: Optional[int] = None,
        receiver_acquire_timeout: Optional[float] = None,
        receiver_idle_timeout: Optional[float] = None,
        **cb_kwargs,
    ):
        cls._initialize(cell)
        tx = _Transaction(
            timeout,
            num_receivers,
            tx_id,
            transaction_done_cb,
            cb_kwargs,
            progress_cb=progress_cb,
            progress_interval=progress_interval,
            outcome_cb=outcome_cb,
            receiver_ids=receiver_ids,
            min_receivers=min_receivers,
            receiver_acquire_timeout=receiver_acquire_timeout,
            receiver_idle_timeout=receiver_idle_timeout,
        )
        # tx_ids are ATTEMPT-SCOPED and single-use while known: a retry is a NEW
        # transaction with a new id (the stable cross-attempt identity is the
        # caller's application-level transfer id, which never enters this service),
        # so nothing a dying attempt emits can be confused with a live successor.
        # A duplicate id is rejected while live, settling, receipted
        # (TX_OUTCOME_TTL), or leaked (see _leaked_ops). Registration is one atomic
        # step -- ownership and table entry together, _tx_lock nesting _outcome_lock
        # (no path nests them in the reverse order).
        with cls._tx_lock:
            leaked = cls._leaked_ops.get(tx.tid)
            if leaked is not None:
                with leaked._ops_cond:
                    still_in_flight = leaked._active_ops > 0
                if still_in_flight or not leaked._settlement_complete:
                    raise ValueError(
                        f"transaction id {tx.tid} from a previous attempt has not fully terminated "
                        f"(settlement still running or operations still in flight): use a new id"
                    )
                cls._leaked_ops.pop(tx.tid, None)
            with cls._outcome_lock:
                # expire an unswept receipt inline: the exclusion window is exactly
                # TX_OUTCOME_TTL, not TTL plus a sweep cycle
                receipt = cls._tx_outcomes.get(tx.tid)
                if receipt is not None and receipt.expired(time.time(), cls.TX_OUTCOME_TTL):
                    cls._tx_outcomes.pop(tx.tid, None)
                    receipt = None
                if tx.tid in cls._tx_table or tx.tid in cls._outcome_owners or receipt is not None:
                    raise ValueError(
                        f"transaction id {tx.tid} is already in use: tx_ids are attempt-scoped and "
                        f"must be unique -- retry with a new id (correlate attempts with an "
                        f"application-level transfer id instead)"
                    )
                cls._outcome_owners[tx.tid] = tx
                cls._tx_table[tx.tid] = tx
        return tx.tid

    @classmethod
    def add_object(
        cls,
        transaction_id: str,
        obj: Downloadable,
        ref_id=None,
    ) -> str:
        if not isinstance(obj, Downloadable):
            raise ValueError(f"obj must be of type {Downloadable} but got {type(obj)}")

        with cls._tx_lock:
            tx = cls._tx_table.get(transaction_id)
            if not tx:
                raise ValueError(f"no such transaction {transaction_id}")

            assert isinstance(tx, _Transaction)
            ref = tx.add_object(obj, ref_id)
            cls._ref_table[ref.rid] = ref
            cls._finished_refs.pop(ref.rid, None)
        return ref.rid

    @classmethod
    def delete_transaction(cls, transaction_id: str):
        tx = None
        with cls._tx_lock:
            tx = cls._tx_table.get(transaction_id)
            if tx:
                cls._delete_tx(tx)

        if tx:
            tx.transaction_done(TransactionDoneStatus.DELETED, on_outcome=functools.partial(cls._record_outcome, tx=tx))
            cls._update_leaked_ops(tx)

    @classmethod
    def shutdown(cls):
        """Shuts down the service: terminates all transactions, drops all state."""
        # Table and ownership teardown are ONE atomic step (_tx_lock nesting
        # _outcome_lock): a registration landing between separate critical sections
        # would enter both tables and then lose its ownership, leaving a live
        # transaction that can never record.
        with cls._tx_lock:
            tx_list = list(cls._tx_table.values())
            for tx in tx_list:
                cls._delete_tx(tx)
            cls._finished_refs.clear()

            with cls._outcome_lock:
                # every id being torn down already carries its termination marker
                # (installed by _delete_tx at unlink), so clearing the ownership and
                # receipt exclusions here cannot expose an id mid-settlement.
                # Clearing ownership is what stops recording: a settlement mid-flight
                # on another thread finds its entry gone at _record_outcome and drops.
                cls._tx_outcomes.clear()
                cls._outcome_owners.clear()
                # waiters resolve to None rather than hang
                for waiters in cls._tx_waiters.values():
                    for waiter in waiters:
                        waiter._resolve(None)
                cls._tx_waiters.clear()

        with cls._init_lock:
            # Shutdown resets callback-registration state even when a cell is still
            # strongly held, so a later isolated service setup registers callbacks again.
            cls._initialized_cells.clear()

        for tx in tx_list:
            tx.transaction_done(TransactionDoneStatus.DELETED)
            cls._update_leaked_ops(tx)

    @classmethod
    def _update_leaked_ops(cls, tx: _Transaction):
        """Called by every terminator after settlement: release the termination marker
        (installed at unlink) once settlement completed and no operations remain, or
        sustain it while a drain-leaked operation is still in flight."""
        with tx._ops_cond:
            leaked = tx._active_ops > 0
        with cls._tx_lock:
            if leaked or not tx._settlement_complete:
                cls._leaked_ops[tx.tid] = tx
            elif cls._leaked_ops.get(tx.tid) is tx:
                cls._leaked_ops.pop(tx.tid, None)

    @classmethod
    def _reap_leaked_ops(cls):
        with cls._tx_lock:
            for tid in [t for t, tx in cls._leaked_ops.items() if tx._settlement_complete and tx._active_ops <= 0]:
                cls._leaked_ops.pop(tid, None)

    @classmethod
    def _delete_tx(cls, tx: _Transaction, tombstone_finished_refs: bool = False):
        cls._tx_table.pop(tx.tid, None)
        # install the termination marker at unlink, for EVERY terminator: it covers
        # the whole settlement window and the leaked-operation tail in one mechanism,
        # releasable only when settlement completed AND no operations are in flight
        # (_update_leaked_ops / the duplicate check / the monitor reap). Ownership and
        # receipt exclusions still exist but no longer carry the window alone.
        cls._leaked_ops[tx.tid] = tx

        # remove all refs
        now = time.time() if tombstone_finished_refs else None
        for r in tx.snapshot_refs():
            cls._ref_table.pop(r.rid, None)
            if tombstone_finished_refs:
                cls._finished_refs[r.rid] = _FinishedRef(r.snapshot_receiver_statuses(), now)
            else:
                cls._finished_refs.pop(r.rid, None)

    @classmethod
    def get_transfer_waiter(cls, transaction_id: str) -> TransferWaiter:
        """Returns an awaitable facade over the transaction's terminal outcome.

        Safe to call before or after termination: a waiter created after the outcome was
        recorded resolves immediately from the outcome table. tx_ids are attempt-scoped
        (never reused), so a waiter always resolves with the verdict of exactly the
        attempt it named; a retrying caller acquires a new waiter for the new attempt.
        """
        waiter = TransferWaiter(transaction_id, service=cls)
        with cls._outcome_lock:
            existing = cls._tx_outcomes.get(transaction_id)
            if existing is not None:
                # resolve even from an expired record: it is still the recorded truth
                waiter._resolve(existing)
                return waiter
            if transaction_id not in cls._outcome_owners:
                # unknown/expired/shut-down: nothing will ever record for this id --
                # resolve None now (waiters never hang); race-free under this lock
                waiter._resolve(None)
                return waiter
            cls._tx_waiters.setdefault(transaction_id, []).append(waiter)
        return waiter

    @classmethod
    def get_acquired_receivers(cls, transaction_id: str) -> set:
        """Receivers that have issued at least one pull on any ref of the transaction."""
        with cls._tx_lock:
            tx = cls._tx_table.get(transaction_id)
        if tx is None:
            return set()
        assert isinstance(tx, _Transaction)
        with tx._stats_lock:
            return set(tx._acquired_receivers)

    @classmethod
    def _record_outcome(cls, outcome: TransferOutcome, tx: _Transaction):
        # tx is required so no call site can opt out of the owner guard:
        # recording is legal only for the transaction that owns the outcome slot.
        with cls._outcome_lock:
            if cls._outcome_owners.get(outcome.tx_id) is not tx:
                # ownership consumed (prior record) or cleared (shutdown): stale, drop
                return
            cls._outcome_owners.pop(outcome.tx_id, None)
            # re-stamp at recording time: the TTL retention window starts when the
            # receipt becomes queryable, not when the verdict was computed -- a slow
            # settlement must not record a receipt that is already expired
            outcome = dataclasses.replace(outcome, timestamp=time.time())
            cls._tx_outcomes[outcome.tx_id] = outcome
            # resolve the awaitable facade: waiters are TransferWaiter objects (no user code
            # runs in _resolve), so setting them under the lock is safe and race-free
            for waiter in cls._tx_waiters.pop(outcome.tx_id, ()):
                waiter._resolve(outcome)

    @classmethod
    def get_transaction_outcome(cls, transaction_id: str) -> Optional[TransferOutcome]:
        """Get the aggregate terminal outcome of a terminated transaction.

        Returns None if the transaction is unknown, still active, or its outcome
        record has expired (TX_OUTCOME_TTL).
        """
        with cls._outcome_lock:
            outcome = cls._tx_outcomes.get(transaction_id)
            if outcome and outcome.expired(time.time(), cls.TX_OUTCOME_TTL):
                cls._tx_outcomes.pop(transaction_id, None)
                return None
        return outcome

    @classmethod
    def _expire_outcomes(cls, now: float):
        with cls._outcome_lock:
            # full scan: concurrent recorders (monitor + delete_transaction) can insert
            # slightly out of timestamp order, so an early-break is not safe; the scan
            # is one float comparison per record
            expired = [tid for tid, outcome in cls._tx_outcomes.items() if outcome.expired(now, cls.TX_OUTCOME_TTL)]
            for tid in expired:
                cls._tx_outcomes.pop(tid, None)

    @classmethod
    def _expire_finished_refs(cls, now: float):
        if not cls._finished_refs:
            return

        expired_refs = [
            rid for rid, finished_ref in cls._finished_refs.items() if finished_ref.expired(now, cls.FINISHED_REFS_TTL)
        ]
        for rid in expired_refs:
            cls._finished_refs.pop(rid, None)

    @classmethod
    def _get_finished_ref_status(cls, rid: str, requester: str) -> Optional[str]:
        now = time.time()
        finished_ref = cls._finished_refs.get(rid)
        if not finished_ref:
            return None

        if finished_ref.expired(now, cls.FINISHED_REFS_TTL):
            cls._finished_refs.pop(rid, None)
            return None

        status = finished_ref.receiver_statuses.get(requester)
        return status

    @classmethod
    def get_transaction_info(cls, transaction_id: str) -> Optional[TransactionInfo]:
        tx = cls._tx_table.get(transaction_id)
        if not tx:
            return None
        else:
            return TransactionInfo(tx)

    @classmethod
    def get_transaction_id(cls, ref_id: str) -> Optional[str]:
        ref = cls._ref_table.get(ref_id)
        if not ref:
            return None
        else:
            assert isinstance(ref, _Ref)
            return ref.tx.tid

    @classmethod
    def _handle_download(cls, request: Message) -> Message:
        requester = request.get_header(MessageHeaderKey.ORIGIN)
        payload = request.payload
        assert isinstance(payload, dict)
        rid = payload.get(_PropKey.REF_ID)
        if not rid:
            cls._logger.error(f"missing {_PropKey.REF_ID} in request from {requester}")
            return make_reply(ReturnCode.INVALID_REQUEST)

        confirm_status = payload.get(_PropKey.CONFIRM)
        if confirm_status is not None:
            return cls._handle_confirm(rid, requester, confirm_status, payload.get(_PropKey.CONFIRM_NONCE))

        current_state = payload.get(_PropKey.STATE)
        with cls._tx_lock:
            ref = cls._ref_table.get(rid)
            if ref is not None and not ref.tx.begin_op():
                # settling/settled: a serve must not start against it -- treat the ref
                # as already gone (tombstone/missing handling)
                ref = None
            if not ref:
                finished_status = cls._get_finished_ref_status(rid, requester)
                if finished_status == DownloadStatus.SUCCESS:
                    cls._logger.debug(f"finished ref {rid} from {requester} retried - returning EOF")
                    return make_reply(ReturnCode.OK, body={_PropKey.STATUS: ProduceRC.EOF})
                elif finished_status == DownloadStatus.FAILED:
                    cls._logger.debug(f"finished ref {rid} from {requester} retried - returning ERROR")
                    return make_reply(ReturnCode.OK, body={_PropKey.STATUS: ProduceRC.ERROR})

                cls._logger.error(f"no ref found for {rid} from {requester}")
                return make_reply(ReturnCode.INVALID_REQUEST)

        try:
            assert isinstance(ref, _Ref)
            ref.mark_active()
            ref.mark_receiver_active(requester)
            ref.emit_progress(receiver_id=requester, state=TransferProgressState.ACTIVE)
            tx = ref.tx
            assert isinstance(tx, _Transaction)

            # receiver-confirmed completion is armed only when the receiver advertised the
            # capability on this request AND the local kill-switch is on
            expect_confirm = bool(payload.get(_PropKey.CONFIRM_CAPABLE)) and _receiver_confirm_enabled()

            # Keep produce() outside the global transaction lock so slow chunk generation
            # does not block unrelated downloads. Timeout/delete cleanup can release the
            # source concurrently; if that happens, the produce exception is reported as
            # a download failure for this requester.
            try:
                rc, data, new_state = ref.obj.produce(current_state, requester)
            except Exception as ex:
                ref.emit_progress(receiver_id=requester, state=TransferProgressState.FAILED, force=True)
                cls._logger.error(
                    f"Object {type(ref.obj)} encountered exception when produce: {secure_format_exception(ex)}"
                )
                return make_reply(ReturnCode.PROCESS_EXCEPTION)

            if rc != ProduceRC.OK:
                # already done -- for a confirm-capable receiver this record is PROVISIONAL and the
                # receiver's confirmation finalizes it; for a legacy receiver it is final (today's
                # producer-served semantics)
                serve_nonce = ref.obj_served(
                    requester,
                    status=DownloadStatus.SUCCESS if rc == ProduceRC.EOF else DownloadStatus.FAILED,
                    expect_confirm=expect_confirm,
                )
                if expect_confirm and serve_nonce:
                    # provisional: the receiver's confirmation carries the terminal truth --
                    # do not latch a terminal progress state the confirm may contradict
                    ref.emit_progress(receiver_id=requester, state=TransferProgressState.ACTIVE, force=True)
                    body = {_PropKey.STATUS: rc, _PropKey.CONFIRM_EXPECTED: True, _PropKey.CONFIRM_NONCE: serve_nonce}
                else:
                    ref.emit_progress(
                        receiver_id=requester,
                        state=TransferProgressState.COMPLETED if rc == ProduceRC.EOF else TransferProgressState.FAILED,
                        force=True,
                    )
                    body = {_PropKey.STATUS: rc}
                return make_reply(ReturnCode.OK, body=body)
            else:
                # continue — accumulate bytes for timing summary in transaction_done()
                # CacheableObject returns a list of byte-chunks; FileDownloader returns raw bytes.
                # Sum chunk lengths for lists (len(list) counts items, not bytes).
                if data is not None:
                    bytes_delta = sum(len(c) for c in data) if isinstance(data, list) else len(data)
                    items_delta = len(data) if isinstance(data, list) else None
                    tx.add_total_bytes(bytes_delta)
                    ref.emit_progress(
                        receiver_id=requester,
                        state=TransferProgressState.ACTIVE,
                        bytes_delta=bytes_delta,
                        items_delta=items_delta,
                    )
                # no CONFIRM_EXPECTED on data chunks: the receiver only consumes it from the
                # terminal reply (confirms are sent only after terminal serves), so advertising
                # per chunk would be dead weight on the hottest wire message
                return make_reply(
                    ReturnCode.OK,
                    body={
                        _PropKey.STATUS: rc,
                        _PropKey.STATE: new_state,
                        _PropKey.DATA: data,
                    },
                )

        finally:
            ref.tx.end_op()

    @classmethod
    def _handle_confirm(cls, rid: str, requester: str, status: str, nonce: Optional[str]) -> Message:
        with cls._tx_lock:
            ref = cls._ref_table.get(rid)
            if ref is not None and not ref.tx.begin_op():
                # settling/settled: the outcome snapshot is being (or was) taken -- this
                # confirm can no longer influence it and must not emit against the tx
                ref = None
        if ref is None:
            # the transaction already terminated/cleaned up: its outcome was computed from what
            # was known then (fail-closed for unconfirmed receivers); a late confirm is dropped
            cls._logger.debug(f"late confirmation for unknown ref {rid} from {requester} dropped")
            return make_reply(ReturnCode.OK)
        assert isinstance(ref, _Ref)
        try:
            # deliberately no unconditional mark_active/mark_receiver_active: a stale or
            # unsolicited confirm must not extend the transaction TTL nor reset idle budgets
            if ref.obj_confirmed(requester, status, nonce):
                ref.mark_active()
        finally:
            ref.tx.end_op()
        return make_reply(ReturnCode.OK)

    @classmethod
    def _monitor_tx(cls):
        while True:
            now = time.time()

            # Per-receiver budget enforcement runs OUTSIDE _tx_lock: finalizing a
            # budget-failed receiver fires user callbacks (downloaded_to_one/all), which must
            # never run under the global lock. A budget failure recorded here flips
            # is_finished() so the classification pass below resolves the tx immediately.
            with cls._tx_lock:
                budget_txs = [tx for tx in cls._tx_table.values() if tx.has_receiver_budgets]
            for tx in budget_txs:
                with cls._tx_lock:
                    # dead or settling tx: skip (begin_op makes the table check binding)
                    live = cls._tx_table.get(tx.tid) is tx and tx.begin_op()
                if not live:
                    continue
                try:
                    tx.enforce_receiver_budgets(now)
                except Exception as ex:
                    cls._logger.error(
                        f"error enforcing receiver budgets for tx {tx.tid}: {secure_format_exception(ex)}"
                    )
                finally:
                    tx.end_op()

            expired_tx = []
            finished_tx = []
            with cls._tx_lock:
                for tid, tx in cls._tx_table.items():
                    assert isinstance(tx, _Transaction)

                    # check whether all refs are done
                    if tx.is_finished():
                        finished_tx.append(tx)
                    elif now - tx.last_active_time > tx.timeout:
                        expired_tx.append(tx)

                for tx in expired_tx:
                    assert isinstance(tx, _Transaction)
                    cls._delete_tx(tx)

                for tx in finished_tx:
                    cls._delete_tx(tx, tombstone_finished_refs=True)

                cls._expire_finished_refs(now)

            cls._expire_outcomes(now)
            cls._reap_leaked_ops()

            for tx in expired_tx:
                tx.transaction_done(
                    TransactionDoneStatus.TIMEOUT, on_outcome=functools.partial(cls._record_outcome, tx=tx)
                )
                cls._update_leaked_ops(tx)

            for tx in finished_tx:
                tx.transaction_done(
                    TransactionDoneStatus.FINISHED, on_outcome=functools.partial(cls._record_outcome, tx=tx)
                )
                cls._update_leaked_ops(tx)

            time.sleep(5.0)


class Consumer(ABC):

    def __init__(self):
        self.logger = get_obj_logger(self)

    @abstractmethod
    def consume(self, ref_id: str, state: dict, data: Any) -> dict:
        """Called to process the received data.

        Args:
            ref_id: ref id of the object being downloaded
            state: current state of downloading
            data: data to be processed

        Returns: new state to be sent back to the data owner.

        """
        pass

    @abstractmethod
    def download_completed(self, ref_id: str):
        """Called when the downloading is finished successfully.

        Args:
            ref_id: ref id of the object being downloaded

        Returns: None

        """
        pass

    @abstractmethod
    def download_failed(self, ref_id: str, reason: str):
        """Called when the downloading is finished unsuccessfully.

        Args:
            ref_id: ref id of the object being downloaded
            reason: explain the reason of failure

        Returns: None

        """
        pass


def download_object(
    from_fqcn: str,
    ref_id: str,
    per_request_timeout: float,
    cell: Cell,
    consumer: Consumer,
    secure=False,
    optional=False,
    abort_signal: Signal = None,
    max_retries: int = 3,
    progress_cb: Optional[Callable] = None,
    progress_interval: float = 30.0,
):
    """Download a large object from the object owner.

    Args:
        from_fqcn: the FQCN of the object owner
        ref_id: reference id of the object to be downloaded
        per_request_timeout: timeout for each request to the object owner.
        cell: the cell to be used for communication with the object owner.
        consumer: the Consumer object used for processing received data
        secure: use P2P private communication with the data owner
        optional: suppress log messages
        abort_signal: for signaling abort
        max_retries: max number of retries per request on TIMEOUT (default 3).
            Resending the same state causes the producer to re-generate the
            same chunk, so retry is data-safe.  Note: CacheableObject's
            _adjust_cache may run twice for the same state on retry, which
            can prematurely evict cache entries in multi-receiver scenarios
            but does not affect data correctness.

    Returns: None

    """
    logger = get_obj_logger(download_object)
    if max_retries < 0:
        raise ValueError(f"max_retries must be non-negative, got {max_retries}")
    consecutive_timeouts = 0
    total_bytes = 0
    total_items = None
    progress_sequence = 0
    last_progress_emit_time = 0.0
    download_start = time.time()
    # Track current download state (None = initial request).
    # On retry, resend the same state so producer re-generates the same chunk.
    current_state = None

    # Receiver-confirmed completion: we advertise the capability on every request (when the
    # kill-switch is on) and learn from each reply whether the producer consumes confirmations.
    confirm_enabled = _receiver_confirm_enabled()
    producer_expects_confirm = False
    confirm_nonce = None

    def _send_confirm(receiver_truth: str):
        # wire contract: a confirmation is sent ONLY after a producer-served terminal reply
        # (EOF/ERROR) -- the producer accepts a confirm only against its pending provisional
        # serve, so mid-stream failure exits do not confirm (budgets/TTL handle those)
        if not (confirm_enabled and producer_expects_confirm):
            return
        try:
            # fire-and-forget by design: a lost confirmation is backstopped producer-side by
            # per-receiver budgets / the transaction timeout, failing closed
            cell.fire_and_forget(
                channel=OBJ_DOWNLOADER_CHANNEL,
                topic=OBJ_DOWNLOADER_TOPIC,
                targets=from_fqcn,
                message=new_cell_message(
                    headers={},
                    payload={
                        _PropKey.REF_ID: ref_id,
                        _PropKey.CONFIRM: receiver_truth,
                        _PropKey.CONFIRM_NONCE: confirm_nonce,
                    },
                ),
                secure=secure,
                optional=optional,
            )
        except Exception as ex:
            logger.warning(f"failed to send download confirmation for ref={ref_id}: {secure_format_exception(ex)}")

    def _emit_progress(state: str, force: bool = False):
        nonlocal progress_sequence, last_progress_emit_time
        if not progress_cb:
            return

        now = time.time()
        if not force and now - last_progress_emit_time < progress_interval:
            return

        progress_sequence += 1
        last_progress_emit_time = now
        try:
            progress_cb(
                ref_id=ref_id,
                sequence=progress_sequence,
                bytes_done=total_bytes,
                items_done=total_items,
                timestamp=now,
                state=state,
            )
        except Exception as ex:
            logger.warning(f"download progress callback failed for ref={ref_id}: {secure_format_exception(ex)}")

    _emit_progress("start", force=True)

    while True:
        # Build a fresh request each iteration (including retries)
        # to avoid re-encoding an already-encoded message.
        request_payload = {_PropKey.REF_ID: ref_id}
        if confirm_enabled:
            request_payload[_PropKey.CONFIRM_CAPABLE] = True
        if current_state is not None:
            request_payload[_PropKey.STATE] = current_state
        request = new_cell_message(headers={}, payload=request_payload)

        start_time = time.time()
        reply = cell.send_request(
            channel=OBJ_DOWNLOADER_CHANNEL,
            target=from_fqcn,
            topic=OBJ_DOWNLOADER_TOPIC,
            request=request,
            timeout=per_request_timeout,
            secure=secure,
            optional=optional,
            abort_signal=abort_signal,
        )
        duration = time.time() - start_time

        if abort_signal and abort_signal.triggered:
            consumer.download_failed(ref_id, f"download aborted after {duration} secs")
            _emit_progress("aborted", force=True)
            return

        assert isinstance(reply, Message)
        rc = reply.get_header(MessageHeaderKey.RETURN_CODE)
        if rc != ReturnCode.OK:
            # Retry on TIMEOUT: streaming transport may intermittently lose
            # responses.  Resending the same state re-generates the same
            # chunk, making retry data-safe (see docstring for caveats).
            if rc == ReturnCode.TIMEOUT:
                if consecutive_timeouts < max_retries:
                    consecutive_timeouts += 1
                    backoff = min(2.0 * (2 ** (consecutive_timeouts - 1)), 60.0)
                    logger.warning(
                        f"[DOWNLOAD_RETRY] Request to {from_fqcn} timed out after {duration:.1f}s "
                        f"(ref={ref_id}, retry {consecutive_timeouts}/{max_retries}, "
                        f"backoff={backoff:.1f}s). Resending same state to re-request the chunk."
                    )
                    # Check abort signal before sleeping to minimise delay
                    if abort_signal and abort_signal.triggered:
                        consumer.download_failed(ref_id, f"download aborted after {duration} secs")
                        _emit_progress("aborted", force=True)
                        return
                    time.sleep(backoff)
                    if abort_signal and abort_signal.triggered:
                        consumer.download_failed(ref_id, f"download aborted after {duration} secs")
                        _emit_progress("aborted", force=True)
                        return
                    continue
                else:
                    logger.warning(
                        f"[DOWNLOAD_FAILED] Max retries ({max_retries}) exhausted for {from_fqcn}, "
                        f"ref={ref_id}. Giving up."
                    )
            consumer.download_failed(ref_id, f"error requesting data from {from_fqcn} after {duration} secs: {rc}")
            _emit_progress("failed", force=True)
            return

        # Log recovery if we were retrying
        if consecutive_timeouts > 0:
            logger.warning(
                f"[DOWNLOAD_RECOVERED] Download from {from_fqcn} recovered after "
                f"{consecutive_timeouts} timeout(s) (ref={ref_id})."
            )
        consecutive_timeouts = 0

        payload = reply.payload
        assert isinstance(payload, dict)
        if payload.get(_PropKey.CONFIRM_EXPECTED):
            producer_expects_confirm = True
            confirm_nonce = payload.get(_PropKey.CONFIRM_NONCE)
        status = payload.get(_PropKey.STATUS)
        if status == ProduceRC.EOF:
            elapsed = time.time() - download_start
            size_mb = total_bytes / (1024 * 1024)
            logger.info(
                f"[client] download ref={ref_id} done: elapsed={elapsed:.2f}s "
                f"size={size_mb:.1f}MB ({total_bytes:,} bytes)"
            )
            try:
                consumer.download_completed(ref_id)
            except Exception:
                # receiver-side finalization failed AFTER the last chunk (e.g. disk-offload
                # finalize): exactly what receiver-confirmed completion exists to surface --
                # the producer must not certify this receiver on its served EOF
                _send_confirm(DownloadStatus.FAILED)
                _emit_progress("failed", force=True)
                raise
            _send_confirm(DownloadStatus.SUCCESS)
            _emit_progress("completed", force=True)
            return
        elif status == ProduceRC.ERROR:
            _send_confirm(DownloadStatus.FAILED)
            consumer.download_failed(ref_id, f"producer error after {duration} secs")
            _emit_progress("failed", force=True)
            return

        # continue
        # CacheableObject sends a list of byte-chunks; FileDownloader sends raw bytes.
        data = payload.get(_PropKey.DATA)
        if data is not None:
            total_bytes += sum(len(c) for c in data) if isinstance(data, list) else len(data)
            if isinstance(data, list):
                total_items = (total_items or 0) + len(data)
        state = payload.get(_PropKey.STATE)
        try:
            new_state = consumer.consume(ref_id, state, data)
        except Exception as ex:
            consumer.download_failed(ref_id, f"exception when consuming data: {secure_format_exception(ex)}")
            _emit_progress("failed", force=True)
            return

        if not isinstance(new_state, dict):
            consumer.download_failed(ref_id, f"consumer error: new_state should be dict but got {type(new_state)}")
            _emit_progress("failed", force=True)
            return

        if abort_signal and abort_signal.triggered:
            consumer.download_failed(ref_id, "download aborted")
            _emit_progress("aborted", force=True)
            return

        _emit_progress("active")

        # Update state for next request
        current_state = new_state
