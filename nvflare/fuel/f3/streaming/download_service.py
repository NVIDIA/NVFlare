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
from typing import Any, Optional, Tuple

from nvflare.apis.signal import Signal
from nvflare.fuel.f3.cellnet.cell import Cell
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey, ReturnCode
from nvflare.fuel.f3.cellnet.utils import make_reply, new_cell_message
from nvflare.fuel.f3.message import Message
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
temporary, you need to remove them when they are downloaded by all sites. But the problem is that you don't know how
quickly each site can finish downloading these large objects. When a transaction contains multiple objects to be
downloaded, it's even harder to know it.

There are two ways to handle this issue: object downloaded callback, and transaction timeout.

You can implement the downloaded_to_one method for the Downloadable object. This method is called when the object is
downloaded to one site.

You can also implement the downloaded_to_all method for the Downloadable object. This method is called when the object 
is downloaded to all sites.

Note that the downloaded_to_all method only works if you know how many sites the object will be downloaded to!

You can always implement the transaction_done method for the Downloadable object. This method is called when the
transaction is done for some reason (normal completion or timeout).

Transaction timeout is the amount of time after the last downloading activity on any object in the
transaction from any site. For example, suppose you want to send 2 large files to 3 sites, each time a download
request is received on any file from any of the 3 sites, the last activity time of the transaction is updated to now.
If no downloading activity is received from any site on any objects in the transaction for the specified timeout,
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
            state: current state of downloading, received from the downloading site
            requester: the FQCN of the site that is downloading

        Returns: a tuple of (return code, a small object to be sent, new state to be sent).

        """
        pass

    def downloaded_to_one(self, to_site: str, status: str):
        """Called when an object is downloaded to a site.

        Args:
            to_site: name of the site that the object has been completely downloaded to.
            status: the download status: DownloadStatus.SUCCESS or DownloadStatus.FAILED.

        Returns: None

        """
        pass

    def downloaded_to_all(self):
        """Called when the object is fully downloaded to all sites."""
        pass

    def transaction_done(self, transaction_id: str, status: str):
        """Called when the transaction is finished.

        Args:
            transaction_id: ID of the transaction.
            status: completion status, a value defined in TransactionDoneStatus.

        Returns: None

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
        self.num_sites_done = 0

    def mark_active(self):
        self.tx.mark_active()

    def obj_downloaded(self, to_site: str, status: str):
        self.num_sites_done += 1

        assert isinstance(self.obj, Downloadable)
        self.obj.downloaded_to_one(to_site, status)

        assert isinstance(self.tx, _Transaction)
        if 0 < self.tx.num_receivers <= self.num_sites_done:
            # this object is done for all sites
            self.obj.downloaded_to_all()


class ProduceRC:
    """Defines return code for the Downloadable object's 'produce' method."""

    OK = "ok"
    ERROR = "error"
    EOF = "eof"


class DownloadStatus:
    """Constants for object download status."""

    SUCCESS = "success"
    FAILED = "failed"


class TransactionDoneStatus:
    """Constants for transaction completion status."""

    FINISHED = "finished"
    TIMEOUT = "timeout"
    DELETED = "deleted"


class _Transaction:

    def __init__(
        self,
        timeout: float,
        num_receivers: int,
        tx_id=None,
        transaction_done_cb=None,
        cb_kwargs=None,
    ):
        """Constructor of the transaction object.

        Args:
            timeout: amount of time since last activity
            num_receivers: number of receivers. 0 means unlimited.
            tx_id: if provided, use it; otherwise create one
        """
        if tx_id:
            self.tid = tx_id
        else:
            self.tid = "T" + str(uuid.uuid4())
        self.timeout = timeout
        self.num_receivers = num_receivers
        self.last_active_time = time.time()
        self.transaction_done_cb = transaction_done_cb
        self.cb_kwargs = cb_kwargs
        self.refs = []

    def mark_active(self):
        """Called to update the last active time of the transaction.

        Returns:

        """
        self.last_active_time = time.time()

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
        r = _Ref(self, obj, ref_id)
        self.refs.append(r)
        obj.set_transaction(self.tid, r.rid)
        return r

    def timed_out(self):
        """Called when the transaction is timed out.

        Returns:

        """
        self.transaction_done(TransactionDoneStatus.TIMEOUT)

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
        for ref in self.refs:
            obj = ref.obj
            assert isinstance(obj, Downloadable)
            obj.transaction_done(self.tid, status)

        if self.transaction_done_cb:
            self.transaction_done_cb(self.tid, status, [ref.obj.base_obj for ref in self.refs], **self.cb_kwargs)


class TransactionInfo:
    """This structure contains public info of a transaction:
    timeout value of the transaction;
    number of sites that objects in the transaction will be downloaded to. 0 means unknown.
    objects that are added to the transaction.
    """

    def __init__(self, tx: _Transaction):
        self.timeout = tx.timeout
        self.num_receivers = tx.num_receivers
        self.objects = [r.obj for r in tx.refs]


class DownloadService:

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
        timeout: float,
        num_receivers: int = 0,
        tx_id=None,
        transaction_done_cb=None,
        **cb_kwargs,
    ):
        cls._initialize(cell)
        tx = _Transaction(timeout, num_receivers, tx_id, transaction_done_cb, cb_kwargs)
        with cls._tx_lock:
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

        tx = cls._tx_table.get(transaction_id)
        if not tx:
            raise ValueError(f"no such transaction {transaction_id}")

        assert isinstance(tx, _Transaction)
        ref = tx.add_object(obj, ref_id)
        with cls._tx_lock:
            cls._ref_table[ref.rid] = ref
        return ref.rid

    @classmethod
    def delete_transaction(cls, transaction_id: str):
        with cls._tx_lock:
            tx = cls._tx_table.get(transaction_id)
            if tx:
                cls._delete_tx(tx)
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
                    cls._delete_tx(tx)
                    tx.transaction_done(TransactionDoneStatus.DELETED)

    @classmethod
    def _delete_tx(cls, tx: _Transaction):
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
            cls._logger.error(f"missing {_PropKey.REF_ID} in request from {requester}")
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
            rc, data, new_state = ref.obj.produce(current_state, requester)
        except Exception as ex:
            cls._logger.error(
                f"Object {type(ref.obj)} encountered exception when produce: {secure_format_exception(ex)}"
            )
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
                    cls._delete_tx(tx)

                for tx in finished_tx:
                    tx.transaction_done(TransactionDoneStatus.FINISHED)
                    cls._delete_tx(tx)

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
        cell: the cell to be used for communication with the object owner.
        consumer: the Consumer object used for processing received data
        secure: use P2P private communication with the data owner
        optional: suppress log messages
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
        duration = time.time() - start_time

        if abort_signal and abort_signal.triggered:
            consumer.download_failed(ref_id, f"download aborted after {duration} secs")
            return

        assert isinstance(reply, Message)
        rc = reply.get_header(MessageHeaderKey.RETURN_CODE)
        if rc != ReturnCode.OK:
            consumer.download_failed(ref_id, f"error requesting data from {from_fqcn} after {duration} secs: {rc}")
            return

        payload = reply.payload
        assert isinstance(payload, dict)
        status = payload.get(_PropKey.STATUS)
        if status == ProduceRC.EOF:
            consumer.download_completed(ref_id)
            return
        elif status == ProduceRC.ERROR:
            consumer.download_failed(ref_id, f"producer error after {duration} secs")
            return

        # continue
        data = payload.get(_PropKey.DATA)
        state = payload.get(_PropKey.STATE)
        try:
            new_state = consumer.consume(ref_id, state, data)
        except Exception as ex:
            consumer.download_failed(ref_id, f"exception when consuming data: {secure_format_exception(ex)}")
            return

        if not isinstance(new_state, dict):
            consumer.download_failed(ref_id, f"consumer error: new_state should be dict but got {type(new_state)}")
            return

        if abort_signal and abort_signal.triggered:
            consumer.download_failed(ref_id, "download aborted")
            return

        # ask for more
        request = new_cell_message(headers={}, payload={_PropKey.REF_ID: ref_id, _PropKey.STATE: new_state})
