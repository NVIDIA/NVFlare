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
import functools
import threading
import time
import uuid
import weakref
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional, Tuple

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
from nvflare.fuel.utils.log_utils import get_obj_logger
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
        self._downloaded_to_all_called = False
        self._receiver_progress = {}
        self._terminal_progress_state = None
        self._progress_lock = threading.Lock()

    def mark_active(self):
        self.tx.mark_active()

    def obj_downloaded(self, to_receiver: str, status: str):
        # Status recording is guarded so terminal-outcome snapshots taken on the
        # monitor thread never observe a half-updated map; user callbacks run
        # outside the lock.
        with self._progress_lock:
            if to_receiver in self.receiver_statuses:
                return

            self.receiver_statuses[to_receiver] = status
            self.num_receivers_done = len(self.receiver_statuses)

            assert isinstance(self.tx, _Transaction)
            all_done = 0 < self.tx.num_receivers <= self.num_receivers_done and not self._downloaded_to_all_called
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

    def snapshot_receiver_statuses(self) -> dict:
        with self._progress_lock:
            return dict(self.receiver_statuses)

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
        self.logger = get_obj_logger(self)

    def mark_active(self):
        """Called to update the last active time of the transaction.

        Returns:

        """
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
        """Add a large object (to be downloaded) to the transaction.

        Args:
            obj: the large object to be downloaded
            ref_id: the ref id to be used, if specified

        Returns:

        """
        with self._refs_lock:
            r = _Ref(self, obj, ref_id)
            self.refs.append(r)
            obj.set_transaction(self.tid, r.rid)
        return r

    def snapshot_refs(self):
        with self._refs_lock:
            return list(self.refs)

    def timed_out(self):
        """Called when the transaction is timed out.

        Returns:

        """
        self.transaction_done(TransactionDoneStatus.TIMEOUT)

    def is_finished(self):
        """Check whether the transaction is finished (all objects are downloaded)."""
        if self.num_receivers <= 0:
            return False

        for ref in self.snapshot_refs():
            assert isinstance(ref, _Ref)
            if ref.num_receivers_done < self.num_receivers:
                return False
        return True

    def transaction_done(self, status: str, on_outcome=None) -> TransferOutcome:
        """Called when the transaction is finished.

        Returns the aggregate TransferOutcome (see transfer_outcome.py): COMPLETED only
        when every expected receiver succeeded — TransactionDoneStatus.FINISHED alone
        does not certify that. The existing transaction_done_cb contract is unchanged
        except that callback exceptions no longer propagate (they would kill the
        monitor thread and skip source release); outcome_cb (if set) fires after it
        with the computed outcome. on_outcome (used by DownloadService to record the
        outcome) is invoked right after the outcome is computed — before the
        potentially slow user callbacks — so pollers see the terminal outcome as soon
        as the transaction stops being active.
        """
        refs = self.snapshot_refs()

        # Compute (and record via on_outcome) the aggregate outcome first, from
        # locked per-receiver snapshots, before any user callback runs.
        outcome = compute_transfer_outcome(
            tx_id=self.tid,
            done_status=status,
            num_receivers=self.num_receivers,
            refs=[RefOutcome(ref_id=ref.rid, receiver_statuses=ref.snapshot_receiver_statuses()) for ref in refs],
            timestamp=time.time(),
        )
        if on_outcome:
            _invoke_cb_safely(self.logger, f"outcome recording for tx {self.tid}", on_outcome, outcome)

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

        # Release source objects after the callback so the callback can still
        # reference them.  This drops the last infrastructure reference to
        # large objects (e.g. numpy dicts) allowing GC to reclaim them.
        for ref in refs:
            ref.obj.release()

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
    # Current live incarnation per tx_id (registered by new_transaction), so a
    # transaction that terminates concurrently with a same-id retry cannot record
    # its outcome over the new incarnation. Guarded by _outcome_lock.
    _tx_incarnations = {}
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
        )
        with cls._outcome_lock:
            # a reused explicit tx_id must not surface the previous incarnation's
            # outcome: purge any recorded outcome and register this incarnation as
            # current so a concurrently-terminating older incarnation cannot record.
            # Registration happens BEFORE the tx becomes monitor-visible in _tx_table:
            # a tx the monitor can terminate is always incarnation-registered, so its
            # terminal outcome can never be dropped as stale, and no dead incarnation
            # entry can be left behind by a terminate-before-register interleaving.
            cls._tx_outcomes.pop(tx.tid, None)
            cls._tx_incarnations[tx.tid] = tx

        old_tx = None
        with cls._tx_lock:
            old_tx = cls._tx_table.get(tx.tid)
            if old_tx:
                # A retry reusing a tx_id supersedes a still-live prior transaction.
                # Retire it now: a plain overwrite would orphan it -- the monitor only
                # discovers transactions through _tx_table, so the old refs would stay
                # servable in _ref_table forever and the old sources would never be
                # released via transaction_done.
                cls._delete_tx(old_tx)
            cls._tx_table[tx.tid] = tx

        if old_tx:
            # terminal callbacks run outside the lock, as in delete_transaction(); the
            # retired transaction's outcome is dropped by the incarnation guard because
            # the new incarnation already owns the tid -- the retry is authoritative
            old_tx.transaction_done(
                TransactionDoneStatus.DELETED, on_outcome=functools.partial(cls._record_outcome, tx=old_tx)
            )
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

    @classmethod
    def shutdown(cls):
        """Shutdown and clean up resources.

        Returns: None

        """
        with cls._tx_lock:
            tx_list = list(cls._tx_table.values())
            for tx in tx_list:
                cls._delete_tx(tx)
            cls._finished_refs.clear()

        with cls._outcome_lock:
            # drop recorded outcomes and clear live incarnations. A monitor iteration
            # mid-termination that blocked on _outcome_lock finds its incarnation gone
            # once it acquires the lock, so its outcome drops in _record_outcome --
            # the incarnation guard alone gates post-shutdown recording.
            cls._tx_outcomes.clear()
            cls._tx_incarnations.clear()

        with cls._init_lock:
            # Shutdown resets callback-registration state even when a cell is still
            # strongly held, so a later isolated service setup registers callbacks again.
            cls._initialized_cells.clear()

        for tx in tx_list:
            tx.transaction_done(TransactionDoneStatus.DELETED)

    @classmethod
    def _delete_tx(cls, tx: _Transaction, tombstone_finished_refs: bool = False):
        cls._tx_table.pop(tx.tid, None)

        # remove all refs
        now = time.time() if tombstone_finished_refs else None
        for r in tx.snapshot_refs():
            cls._ref_table.pop(r.rid, None)
            if tombstone_finished_refs:
                cls._finished_refs[r.rid] = _FinishedRef(r.snapshot_receiver_statuses(), now)
            else:
                cls._finished_refs.pop(r.rid, None)

    @classmethod
    def _record_outcome(cls, outcome: TransferOutcome, tx: _Transaction):
        # tx is required so no call site can opt out of the incarnation guard:
        # recording is legal only for the live registered incarnation.
        with cls._outcome_lock:
            if cls._tx_incarnations.get(outcome.tx_id) is not tx:
                # This outcome belongs to a dead generation and must be dropped:
                #   - a newer same-tx_id incarnation (a retry) registered while this
                #     transaction was terminating, or
                #   - the incarnation was cleared by shutdown() (which also clears the
                #     outcome table) or consumed by a prior terminal record.
                # This single guard closes the stale-outcome race across shutdown and
                # re-init: a recorder that blocked on _outcome_lock during shutdown and
                # wins the lock afterward finds no live incarnation and drops here.
                return
            cls._tx_incarnations.pop(outcome.tx_id, None)
            cls._tx_outcomes[outcome.tx_id] = outcome

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

        current_state = payload.get(_PropKey.STATE)
        with cls._tx_lock:
            ref = cls._ref_table.get(rid)
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

        assert isinstance(ref, _Ref)
        ref.mark_active()
        ref.emit_progress(receiver_id=requester, state=TransferProgressState.ACTIVE)
        tx = ref.tx
        assert isinstance(tx, _Transaction)

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
            # already done
            ref.obj_downloaded(
                requester, status=DownloadStatus.SUCCESS if rc == ProduceRC.EOF else DownloadStatus.FAILED
            )
            ref.emit_progress(
                receiver_id=requester,
                state=TransferProgressState.COMPLETED if rc == ProduceRC.EOF else TransferProgressState.FAILED,
                force=True,
            )
            return make_reply(ReturnCode.OK, body={_PropKey.STATUS: rc})
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
            return make_reply(
                ReturnCode.OK,
                body={
                    _PropKey.STATUS: rc,
                    _PropKey.STATE: new_state,
                    _PropKey.DATA: data,
                },
            )

    @classmethod
    def _monitor_tx(cls):
        while True:
            now = time.time()
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

            for tx in expired_tx:
                tx.transaction_done(
                    TransactionDoneStatus.TIMEOUT, on_outcome=functools.partial(cls._record_outcome, tx=tx)
                )

            for tx in finished_tx:
                tx.transaction_done(
                    TransactionDoneStatus.FINISHED, on_outcome=functools.partial(cls._record_outcome, tx=tx)
                )

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
        status = payload.get(_PropKey.STATUS)
        if status == ProduceRC.EOF:
            elapsed = time.time() - download_start
            size_mb = total_bytes / (1024 * 1024)
            logger.info(
                f"[client] download ref={ref_id} done: elapsed={elapsed:.2f}s "
                f"size={size_mb:.1f}MB ({total_bytes:,} bytes)"
            )
            consumer.download_completed(ref_id)
            _emit_progress("completed", force=True)
            return
        elif status == ProduceRC.ERROR:
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
