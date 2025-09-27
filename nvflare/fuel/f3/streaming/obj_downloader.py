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
from typing import Any

from nvflare.apis.signal import Signal
from nvflare.fuel.f3.cellnet.cell import Cell
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey, ReturnCode
from nvflare.fuel.f3.cellnet.utils import make_reply, new_cell_message
from nvflare.fuel.f3.message import Message
from nvflare.fuel.utils.log_utils import get_obj_logger
from nvflare.fuel.utils.validation_utils import check_callable, check_object_type
from nvflare.security.logging import secure_format_exception

OBJ_DOWNLOADER_CHANNEL = "obj_downloader__"
OBJ_DOWNLOADER_TOPIC = "obj_downloader__download"

"""
This package provides a framework for building object downloading capability (e.g. file download).

A large object takes a lot of memory space. Sending a large object in one message needs even more memory space since
the object needs to be serialized into large number of bytes. Additional memory space may still be needed for the
transport layer to send the message. If the message is to be sent to multiple endpoints, even more memory is needed.

Object Downloading can drastically reduce memory consumption:
- Instead of sending the large object in one message, it is divided into many smaller objects;
- Instead of pushing the message to the endpoints, each endpoint will come to request. This makes it more reliable when
different endpoints have different speed.

Object Downloading works as follows:
- The sender prepares the object(s) for downloading. It first creates a transaction to get a tx_id. It then adds each
object to be downloaded to the transaction, and get a reference id (ref_id).
- The sender sends the ref_id(s) to all recipients through a separate message.
- Each recipient then calls the download_object function to download each referenced large object.

To develop the downloading capability for a type of object (e.g. a file, a large dict, etc.), you need to provide
the implementation of a Producer and a Consumer.
- On the sending side, the Producer is responsible for producing the next small object to be sent (a chunk of bytes;
a small subset of the large dict; etc.).
- On the receiving side, the Consumer is responsible for processing the received small objects (writing the received
bytes to a temp file; putting the received small dict to the end result; etc.).

One issue with object downloading is object life cycle management. Since the large objects to be downloaded are usually
temporary, you need to remove them when they are downloaded by all sites. But the problem is that you don't know how
quickly each site can finish downloading these large objects. When a transaction contains multiple objects to be
downloaded, it's even harder to know it.

There are two ways to handle this issue: object downloaded callback, and transaction timeout.

You can register an object_downloaded CB when adding an object to transaction. When the object is fully downloaded
to a site, this CB will be called. The obj_downloaded CB must follow this signature:

    downloaded_cb(ref_id: str, to_site: str, status: str, obj: Any, **cb_kwargs)

where ref_id is the reference id of the object;
to_site is the FQCN of the site that has just finished downloading;
status is the status of downloading, as defined in DownloadStatus class;
obj is the large object that was just downloaded;
cb_kwargs are the kw args registered with the CB.

Transaction timeout is the amount of time after the last downloading activity on any object in the
transaction from any site. For example, suppose you want to send 2 large files to 3 sites, each time a download
request is received on any file from any of the 3 sites, the last activity time of the transaction is updated to now.
If no downloading activity is received from any site on any objects in the transaction for the specified timeout,
the transaction is considered "timed out", and the timeout callback registered with the transaction is called.
The transaction timeout CB must follow this signature:

    timeout_cb(tx_id: str, objs: List[Any], **cb_kwargs)

where tx_id is the ID of the transaction;
objs is the list of large objects registered with the transaction;
cb_kwargs are the kw args registered with the CB.

You may need to use both mechanisms to fully take care of object life cycles. The object downloaded CB may never be
called since the site somehow didn't finish the downloading. In reality the timeout mechanism may be sufficient.

Unlike with Object Streamer that the object owner pushes small objects to the recipients; with Object Downloader,
each recipient pulls the data from the object owner.
"""


class _PropKey:
    REF_ID = "ref_id"
    STATE = "state"
    DATA = "data"
    STATUS = "status"


class _Ref:

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
        self.obj_downloaded_cb = obj_downloaded_cb
        self.cb_kwargs = cb_kwargs

    def mark_active(self):
        self.tx.mark_active()

    def obj_downloaded(self, to_site: str, status: str):
        if self.obj_downloaded_cb:
            self.obj_downloaded_cb(self.rid, to_site, status, self.obj, **self.cb_kwargs)


class ProduceRC:
    """Defines return code for the Producer's produce method."""

    OK = "ok"
    ERROR = "error"
    EOF = "eof"


class DownloadStatus:
    SUCCESS = "success"
    FAILED = "failed"


class Producer(ABC):

    def __init__(self):
        self.logger = get_obj_logger(self)

    @abstractmethod
    def produce(self, ref_id: str, obj: Any, state: dict, requester: str) -> (str, Any, dict):
        """Produce a small object to be sent (on object sender side).

        Args:
            ref_id: the ref id of the object being downloaded
            obj: the large object
            state: current state of downloading, received from the downloading site
            requester: the FQCN of the site that is downloading

        Returns: a tuple of (return code, a small object to be sent, new state to be sent).

        """
        pass


class _Transaction:

    def __init__(
        self,
        producer: Producer,
        timeout: float,
        tx_id=None,
        timeout_cb=None,
        **cb_kwargs,
    ):
        """Constructor of the transaction object.

        Args:
            producer: the Producer object to produce small objects.
            timeout: amount of time since last activity
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
    def new_transaction(cls, cell: Cell, producer: Producer, timeout: float, tx_id=None, timeout_cb=None, **cb_kwargs):
        cls._initialize(cell)
        tx = _Transaction(producer, timeout, tx_id, timeout_cb, **cb_kwargs)
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
            with cls._tx_lock:
                for tid, tx in cls._tx_table.items():
                    assert isinstance(tx, _Transaction)
                    if now - tx.last_active_time > tx.timeout:
                        expired_tx.append(tx)

                if expired_tx:
                    for tx in expired_tx:
                        cls._delete_tx(tx, True)
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
