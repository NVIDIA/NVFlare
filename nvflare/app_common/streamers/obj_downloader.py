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

from nvflare.fuel.f3.cellnet.cell import Cell
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey, ReturnCode
from nvflare.fuel.f3.cellnet.utils import make_reply, new_cell_message
from nvflare.fuel.f3.message import Message
from nvflare.fuel.utils.log_utils import get_obj_logger
from nvflare.fuel.utils.validation_utils import check_callable, check_object_type

OBJ_DOWNLOADER_CHANNEL = "obj_downloader__"
OBJ_DOWNLOADER_TOPIC = "obj_downloader__download"


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
        obj_downloaded_cb=None,
        **cb_kwargs,
    ):
        self.rid = "R" + str(uuid.uuid4())
        self.tx = tx
        self.obj = obj
        self.obj_downloaded_cb = obj_downloaded_cb
        self.cb_kwargs = cb_kwargs

    def mark_active(self):
        self.tx.mark_active()

    def obj_downloaded(self, to_site: str, rc: str):
        if self.obj_downloaded_cb:
            self.obj_downloaded_cb(to_site, rc, self.obj, **self.cb_kwargs)


class Producer(ABC):

    @abstractmethod
    def produce(self, obj, state: dict, requester, logger) -> (str, Any, dict):
        pass


class _Transaction:

    def __init__(self, producer: Producer, timeout: float, timeout_cb, **cb_kwargs):
        check_callable("timeout_cb", timeout_cb)
        check_object_type("producer", producer, Producer)
        self.tid = "T" + str(uuid.uuid4())
        self.producer = producer
        self.timeout = timeout
        self.timeout_cb = timeout_cb
        self.cb_kwargs = cb_kwargs
        self.last_active_time = time.time()
        self.refs = []

    def produce(self, obj, state, requester, logger):
        return self.producer.produce(obj, state, requester, logger)

    def mark_active(self):
        self.last_active_time = time.time()

    def add_download_object(
        self,
        obj: Any,
        obj_downloaded_cb=None,
        **cb_kwargs,
    ):
        r = _Ref(self, obj, obj_downloaded_cb, **cb_kwargs)
        self.refs.append(r)
        return r

    def timed_out(self):
        if self.timeout_cb:
            self.timeout_cb([r.obj for r in self.refs], **self.cb_kwargs)


class ProduceRC:
    OK = "OK"
    ERROR = "error"
    EOF = "eof"


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
    def new_transaction(cls, cell: Cell, producer: Producer, timeout: float, timeout_cb, **cb_kwargs):
        cls._initialize(cell)
        tx = _Transaction(producer, timeout, timeout_cb, **cb_kwargs)
        with cls._tx_lock:
            cls._tx_table[tx.tid] = tx
        return tx.tid

    @classmethod
    def add_download_object(
        cls,
        transaction_id: str,
        obj: Any,
        obj_downloaded_cb=None,
        **cb_kwargs,
    ) -> str:
        tx = cls._tx_table.get(transaction_id)
        if not tx:
            raise ValueError(f"no such transaction {transaction_id}")

        assert isinstance(tx, _Transaction)
        ref = tx.add_download_object(obj, obj_downloaded_cb, **cb_kwargs)
        with cls._tx_lock:
            cls._ref_table[ref.rid] = ref
        return ref.rid

    @classmethod
    def delete_transaction(cls, transaction_id: str):
        with cls._tx_lock:
            tx = cls._tx_table.get(transaction_id)
            if tx:
                cls._delete_tx(tx)

    @classmethod
    def _delete_tx(cls, tx: _Transaction):
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

        rc, chunk, new_state = tx.produce(ref.obj, current_state, requester, cls._logger)

        if rc != ProduceRC.OK:
            # already done
            ref.obj_downloaded(requester, rc)
            return make_reply(ReturnCode.OK, body={_PropKey.STATUS: rc})
        else:
            # continue
            return make_reply(
                ReturnCode.OK,
                body={
                    _PropKey.STATUS: rc,
                    _PropKey.STATE: new_state,
                    _PropKey.DATA: chunk,
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
                        cls._delete_tx(tx)
                        tx.timed_out()
            time.sleep(5.0)


class Consumer(ABC):

    @abstractmethod
    def consume(self, state, data) -> dict:
        pass

    @abstractmethod
    def download_completed(self):
        pass

    @abstractmethod
    def download_failed(self, reason: str):
        pass


def download_object(
    from_fqcn: str,
    ref_id: str,
    per_request_timeout: float,
    cell: Cell,
    consumer: Consumer,
    secure=False,
    optional=False,
    abort_signal=None,
):
    request = new_cell_message(
        headers={},
        payload={
            _PropKey.REF_ID: ref_id,
        },
    )

    while True:
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
        assert isinstance(reply, Message)
        rc = reply.get_header(MessageHeaderKey.RETURN_CODE)
        if rc != ReturnCode.OK:
            consumer.download_failed(f"error requesting data from {from_fqcn}: {rc}")
            return

        payload = reply.payload
        assert isinstance(payload, dict)
        status = payload.get(_PropKey.STATUS)
        if status == ProduceRC.EOF:
            consumer.download_completed()
            return
        elif status == ProduceRC.ERROR:
            consumer.download_failed("producer error")
            return

        # continue
        data = payload.get(_PropKey.DATA)
        state = payload.get(_PropKey.STATE)
        new_state = consumer.consume(state, data)
        if not isinstance(new_state, dict):
            consumer.download_failed("consumer error")
            return

        # ask for more
        request = new_cell_message(headers={}, payload={_PropKey.REF_ID: ref_id, _PropKey.STATE: new_state})
