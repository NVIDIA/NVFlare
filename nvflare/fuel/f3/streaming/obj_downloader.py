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
import threading
import time
import uuid
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple

from nvflare.apis.signal import Signal
from nvflare.fuel.f3.cellnet.cell import Cell
from nvflare.fuel.f3.streaming.download_service import Downloadable, DownloadService


class ObjectDownloader:
    """Defines a universal object downloader that can be used to download any Downloadable objects."""

    def __init__(
        self,
        tx,
        obj: Any,
        ref_id=None,
        obj_downloaded_cb=None,
        **cb_kwargs,
    ):
        if ref_id:
            # use provided ref_id
            self.rid = ref_id
        else:
            self.rid = "R" + str(uuid.uuid4())
        self.tx = tx
        self.obj = obj
        self.num_sites_done = 0
        self.obj_downloaded_cb = obj_downloaded_cb
        self.cb_kwargs = cb_kwargs

    def mark_active(self):
        self.tx.mark_active()

    def obj_downloaded(self, to_site: str, status: str):
        self.num_sites_done += 1
        if self.obj_downloaded_cb:
            self.obj_downloaded_cb(self.rid, to_site, status, self.obj, **self.cb_kwargs)

        assert isinstance(self.tx.producer, Producer)
        self.tx.producer.object_downloaded(self.rid, self.obj, to_site, status)

        assert isinstance(self.tx, _Transaction)
        if 0 < self.tx.num_receivers <= self.num_sites_done:
            # this object is done for all sites
            self.tx.producer.object_done(self.rid, self.obj)


class ProduceRC:
    """Defines return code for the Producer's produce method."""

    OK = "ok"
    ERROR = "error"
    EOF = "eof"


class DownloadStatus:
    SUCCESS = "success"
    FAILED = "failed"


class TransactionDoneStatus:
    FINISHED = "finished"
    TIMEOUT = "timeout"
    DELETED = "deleted"


class Producer(ABC):

    def __init__(self):
        self.logger = get_obj_logger(self)

    @abstractmethod
    def produce(self, ref_id: str, obj: Any, state: dict, requester: str) -> Tuple[str, Any, dict]:
        """Produce a small object to be sent (on object sender side).

        Args:
            ref_id: the ref id of the object being downloaded
            obj: the large object
            state: current state of downloading, received from the downloading site
            requester: the FQCN of the site that is downloading

        Returns: a tuple of (return code, a small object to be sent, new state to be sent).

        """
        pass

    def object_downloaded(self, ref_id: str, obj: Any, to_site: str, status: str):
        """Called when an object is downloaded to a site."""
        pass

    def object_done(self, ref_id: str, obj: Any):
        """Called when the object is fully downloaded to all sites."""
        pass

    def transaction_done(self, transaction_id: str, objs: List[Any], status: str):
        """Called when the transaction is finished."""
        pass


class _Transaction:

    def __init__(
        self,
        producer: Producer,
        timeout: float,
        num_receivers: int,
        tx_id=None,
        transaction_done_cb=None,
        progress_cb=None,
        progress_interval: float = 30.0,
        outcome_cb=None,
        **cb_kwargs,
    ):
        """Constructor of ObjectDownloader.

        Args:
            producer: the Producer object to produce small objects.
            timeout: amount of time since last activity
            num_receivers: number of receivers. 0 means unlimited.
            tx_id: if provided, use it; otherwise create one
            timeout_cb: the CB to be called when the transaction timed out
            **cb_kwargs: args to be passed to the timeout CB
        """
        check_callable("timeout_cb", timeout_cb)
        check_object_type("producer", producer, Producer)
        if tx_id:
            self.tid = tx_id
        else:
            self.tid = "T" + str(uuid.uuid4())
        self.producer = producer
        self.timeout = timeout
        self.num_receivers = num_receivers
        self.timeout_cb = timeout_cb
        self.cb_kwargs = cb_kwargs
        self.last_active_time = time.time()
        self.refs = []

    def produce(self, ref_id: str, obj: Any, state: dict, requester: str):
        """Called to produce the next small object to be sent.

        Args:
            ref_id: ref id of the object being downloaded
            obj: the large object being downloaded
            state: current state received from the downloading site (requester).
            requester: FQCN of the requester

        Returns:

        """
        return self.producer.produce(ref_id, obj, state, requester)

    def mark_active(self):
        """Called to update the last active time of the transaction.

        Returns:

        """
        self.last_active_time = time.time()

    def add_download_object(
        self,
        obj: Any,
        ref_id=None,
        obj_downloaded_cb=None,
        **cb_kwargs,
    ):
        """Add a large object (to be downloaded) to the transaction.

        Args:
            obj: the large object to be downloaded
            ref_id: the ref id to be used, if specified
            obj_downloaded_cb: the CB to be called when the object is fully downloaded
            **cb_kwargs: args to be passed to the CB.

        Returns:

        """
        r = _Ref(self, obj, ref_id, obj_downloaded_cb, **cb_kwargs)
        self.refs.append(r)
        return r

    def timed_out(self):
        """Called when the transaction is timed out.

        Returns:

        """
        if self.timeout_cb:
            self.timeout_cb(self.tid, [r.obj for r in self.refs], **self.cb_kwargs)

    def is_finished(self):
        """Check whether the transaction is finished (all objects are downloaded)."""
        if self.num_receivers <= 0:
            return False

        for ref in self.refs:
            assert isinstance(ref, _Ref)
            if ref.num_sites_done < self.num_receivers:
                return False
        return True

    def transaction_done(self, status: str):
        """Called when the transaction is finished."""
        self.producer.transaction_done(self.tid, [r.obj for r in self.refs], status)


class TransactionInfo:

    def __init__(self, tx: _Transaction):
        self.producer = tx.producer
        self.timeout = tx.timeout
        self.num_receivers = tx.num_receivers
        self.objects = [r.obj for r in tx.refs]


class ObjDownloader:

    _init_lock = threading.Lock()
    _tx_table = {}
    _ref_table = {}
    _logger = None
    _tx_monitor = None
    _tx_lock = threading.Lock()
    _initialized_cells = {}

    @classmethod
    def _initialize(cls, cell: Cell):
        with cls._init_lock:
            if not cls._logger:
                cls._logger = get_obj_logger(cls)

            if not cls._tx_monitor:
                cls._tx_monitor = threading.Thread(target=cls._monitor_tx, daemon=True)
                cls._tx_monitor.start()

            initialized = cls._initialized_cells.get(id(cell))
            if not initialized:
                # register CBs
                cell.register_request_cb(
                    channel=OBJ_DOWNLOADER_CHANNEL,
                    topic=OBJ_DOWNLOADER_TOPIC,
                    cb=cls._handle_download,
                )
                cls._initialized_cells[id(cell)] = True

    @classmethod
    def new_transaction(
        cls,
        cell: Cell,
        producer: Producer,
        timeout: float,
        num_receivers: int = 0,
        tx_id=None,
        timeout_cb=None,
        **cb_kwargs,
    ):
        cls._initialize(cell)
        tx = _Transaction(producer, timeout, num_receivers, tx_id, timeout_cb, **cb_kwargs)
        with cls._tx_lock:
            cls._tx_table[tx.tid] = tx
        return tx.tid

    @classmethod
    def add_download_object(
        cls,
        transaction_id: str,
        obj: Any,
        ref_id=None,
        obj_downloaded_cb=None,
        **cb_kwargs,
    ) -> str:
        if obj_downloaded_cb is not None:
            check_callable("obj_downloaded_cb", obj_downloaded_cb)

        tx = cls._tx_table.get(transaction_id)
        if not tx:
            raise ValueError(f"no such transaction {transaction_id}")

        assert isinstance(tx, _Transaction)
        ref = tx.add_download_object(obj, ref_id, obj_downloaded_cb, **cb_kwargs)
        with cls._tx_lock:
            cls._ref_table[ref.rid] = ref
        return ref.rid

    @classmethod
    def delete_transaction(cls, transaction_id: str, call_cb=False):
        with cls._tx_lock:
            tx = cls._tx_table.get(transaction_id)
            if tx:
                cls._delete_tx(tx, call_cb)
                tx.transaction_done(TransactionDoneStatus.DELETED)

    @classmethod
    def shutdown(cls):
        """Shutdown and clean up resources.

        Returns: None

        """
        with cls._tx_lock:
            tx_list = list(cls._tx_table.values())
            if tx_list:
                for tx in tx_list:
                    cls._delete_tx(tx, True)
                    tx.transaction_done(TransactionDoneStatus.DELETED)

    @classmethod
    def _delete_tx(cls, tx: _Transaction, call_cb=False):
        if call_cb:
            try:
                tx.timed_out()
            except Exception as ex:
                cls._logger.error(f"exception from timeout_cb: {secure_format_exception(ex)}")

        cls._tx_table.pop(tx.tid, None)

        # remove all refs
        for r in tx.refs:
            cls._ref_table.pop(r.rid, None)

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
            cls._logger.erro(f"missing {_PropKey.REF_ID} in request from {requester}")
            return make_reply(ReturnCode.INVALID_REQUEST)

        current_state = payload.get(_PropKey.STATE)
        with cls._tx_lock:
            ref = cls._ref_table.get(rid)
            if not ref:
                cls._logger.error(f"no ref found for {rid} from {requester}")
                return make_reply(ReturnCode.INVALID_REQUEST)

        assert isinstance(ref, _Ref)
        ref.mark_active()
        tx = ref.tx
        assert isinstance(tx, _Transaction)

        try:
            rc, data, new_state = tx.produce(rid, ref.obj, current_state, requester)
        except Exception as ex:
            cls._logger.error(f"Producer {type(tx.producer)} encountered exception: {secure_format_exception(ex)}")
            return make_reply(ReturnCode.PROCESS_EXCEPTION)

        if rc != ProduceRC.OK:
            # already done
            ref.obj_downloaded(
                requester, status=DownloadStatus.SUCCESS if rc == ProduceRC.EOF else DownloadStatus.FAILED
            )
            return make_reply(ReturnCode.OK, body={_PropKey.STATUS: rc})
        else:
            # continue
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
                    tx.transaction_done(TransactionDoneStatus.TIMEOUT)
                    cls._delete_tx(tx, True)

                for tx in finished_tx:
                    tx.transaction_done(TransactionDoneStatus.FINISHED)
                    cls._delete_tx(tx, False)

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
):
    """Download a large object from the object owner.

    Args:
        from_fqcn: the FQCN of the object owner
        ref_id: reference id of the object to be downloaded
        per_request_timeout: timeout for each request to the object owner.
        cell: the cell to be used for communication withe object owner.
        consumer: the Consumer object used for processing received data
        secure: use P2P private communication with the data owner
        optional: supress log messages
        abort_signal: for signaling abort

    Returns: None

    """
    request = new_cell_message(
        headers={},
        payload={
            _PropKey.REF_ID: ref_id,
        },
    )

    while True:
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

    def add_object(self, obj: Downloadable, ref_id=None) -> str:
        """Add a Downloadable object to the downloader.

        Args:
            obj: the Downloadable object to be added.
            ref_id: if specified, use it as the generated ref. If not specified, dynamically generates a ref ID.

        Returns: the ref ID for the object.

        """
        rid = DownloadService.add_object(
            transaction_id=self.tx_id,
            obj=obj,
            ref_id=ref_id,
        )
        return rid

    def delete_transaction(self):
        """Delete the download transaction forcefully.
        You call this method only if you want to stop the downloading process prematurely.

        Returns: None.

        """
        DownloadService.delete_transaction(self.tx_id)
