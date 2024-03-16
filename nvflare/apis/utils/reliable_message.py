# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
import concurrent.futures
import threading
import time
import uuid

from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import ReservedHeaderKey, ReturnCode, Shareable, make_reply
from nvflare.apis.signal import Signal

# Operation Types
OP_REQUEST = "req"
OP_QUERY = "query"
OP_REPLY = "reply"

# Reliable Message headers
HEADER_OP = "rm.op"
HEADER_TOPIC = "rm.topic"
HEADER_TX = "rm.tx"
HEADER_TIMEOUT = "rm.timeout"
HEADER_STATUS = "rm.status"

# Status
STATUS_IN_PROCESS = "in_process"
STATUS_IN_REPLY = "in_reply"
STATUS_NOT_RECEIVED = "not_received"
STATUS_REPLIED = "replied"
STATUS_ABORTED = "aborted"

# Topics for Reliable Message
TOPIC_RELIABLE_REQUEST = "RM.RELIABLE_REQUEST"
TOPIC_RELIABLE_REPLY = "RM.RELIABLE_REPLY"


def _extract_result(reply: dict, target: str):
    if not isinstance(reply, dict):
        return None, None
    result = reply.get(target)
    if not result:
        return None, None
    return result, result.get_return_code()


def _status_reply(status: str):
    return make_reply(rc=ReturnCode.OK, headers={HEADER_STATUS: status})


def _error_reply(rc: str, error: str):
    return make_reply(rc, headers={ReservedHeaderKey.ERROR: error})


class _RequestReceiver:
    """This class handles reliable message request on the receiving end"""

    def __init__(self, topic, request_handler_f, executor):
        """The constructor

        Args:
            topic: The topic of the reliable message
            request_handler_f: The callback function to handle the request in the form of
                request_handler_f(topic: str, request: Shareable, fl_ctx:FLContext)
            executor: A ThreadPoolExecutor
        """
        self.topic = topic
        self.request_handler_f = request_handler_f
        self.executor = executor
        self.timeout = None
        self.rcv_time = None
        self.result = None
        self.source = None
        self.tx_id = None
        self.reply_time = None

    def process(self, request: Shareable, fl_ctx: FLContext) -> Shareable:
        self.tx_id = request.get_header(HEADER_TX)
        op = request.get_header(HEADER_OP)
        peer_ctx = fl_ctx.get_peer_context()
        assert isinstance(peer_ctx, FLContext)
        self.source = peer_ctx.get_identity_name()
        if op == OP_REQUEST:
            # it is possible that a new request for the same tx is received while we are processing the previous one
            if not self.rcv_time:
                self.rcv_time = time.time()
                self.timeout = request.get_header(HEADER_TIMEOUT)

                # start processing
                self.executor.submit(self._do_request, request, fl_ctx)
                return _status_reply(STATUS_IN_PROCESS)  # ack
            elif self.result:
                # we already finished processing - send the result back
                return self.result
            else:
                # we are still processing
                return _status_reply(STATUS_IN_PROCESS)
        elif op == OP_QUERY:
            if self.result:
                if self.reply_time:
                    # result already sent back successfully
                    return _status_reply(STATUS_REPLIED)
                elif self.replying:
                    # result is being sent
                    return _status_reply(STATUS_IN_REPLY)
                else:
                    # try to send the result again
                    return self.result
            else:
                # still in process
                if time.time() - self.rcv_time > self.timeout:
                    # the process is taking too much time
                    return _status_reply(STATUS_ABORTED)
                else:
                    return _status_reply(STATUS_IN_PROCESS)

    def _try_reply(self, fl_ctx: FLContext):
        engine = fl_ctx.get_engine()
        self.replying = True
        ack = engine.send_aux_request(
            targets=[self.source],
            topic=TOPIC_RELIABLE_REPLY,
            request=self.result,
            timeout=self.timeout,
            fl_ctx=fl_ctx,
        )
        self.replying = False
        _, rc = _extract_result(ack, self.source)
        if rc == ReturnCode.OK:
            # reply sent successfully!
            self.reply_time = time.time()

    def _do_request(self, request: Shareable, fl_ctx: FLContext):
        try:
            result = self.request_handler_f(self.topic, request, fl_ctx)
        except Exception as e:
            result = _error_reply(ReturnCode.EXECUTION_EXCEPTION, str(e))

        # send back
        result.set_header(HEADER_TX, self.tx_id)
        result.set_header(HEADER_OP, OP_REPLY)
        result.set_header(HEADER_TOPIC, self.topic)
        self.result = result
        self._try_reply(fl_ctx)


class _ReplyReceiver:
    def __init__(self, tx_id: str):
        self.tx_id = tx_id
        self.result = None
        self.result_ready = threading.Event()

    def process(self, reply: Shareable) -> Shareable:
        self.result = reply
        self.result_ready.set()
        return make_reply(ReturnCode.OK)


class ReliableMessage:

    _topic_to_handle = {}
    _req_receivers = {}  # tx id => receiver
    _enabled = False
    _executor = None
    _query_interval = 1.0
    _max_retries = 5
    _max_tx_time = 300.0  # 5 minutes
    _reply_receivers = {}  # tx id => receiver
    _tx_lock = threading.Lock()
    _shutdown_asked = False

    @classmethod
    def register_request_handler(cls, topic: str, handler_f):
        """Register a handler for the reliable message with this topic

        Args:
            topic: The topic of the reliable message
            handler_f: The callback function to handle the request in the form of
                handler_f(topic, request, fl_ctx)
        """
        if not cls._enabled:
            raise RuntimeError("ReliableMessage is not enabled. Please call ReliableMessage.enable() to enable it")
        if not callable(handler_f):
            raise TypeError(f"handler_f must be callable but {type(handler_f)}")
        cls._topic_to_handle[topic] = handler_f

    @classmethod
    def _receive_request(cls, topic: str, request: Shareable, fl_ctx: FLContext):
        tx_id = request.get_header(HEADER_TX)
        receiver = cls._req_receivers.get(tx_id)
        op = request.get_header(HEADER_OP)
        topic = request.get_header(HEADER_TOPIC)
        if op == OP_REQUEST:
            if not receiver:
                handler_f = cls._topic_to_handle.get(topic)
                if not handler_f:
                    # no handler registered for this topic!
                    return make_reply(ReturnCode.TOPIC_UNKNOWN)
                receiver = _RequestReceiver(topic, handler_f, cls._executor)
                with cls._tx_lock:
                    cls._req_receivers[tx_id] = receiver
            return receiver.process(request, fl_ctx)
        elif op == OP_QUERY:
            if not receiver:
                return _status_reply(STATUS_NOT_RECEIVED)  # meaning the request wasn't received
            else:
                return receiver.process(request, fl_ctx)
        else:
            return make_reply(rc=ReturnCode.BAD_REQUEST_DATA)

    @classmethod
    def _receive_reply(cls, topic: str, request: Shareable, fl_ctx: FLContext):
        tx_id = request.get_header(HEADER_TX)
        receiver = cls._reply_receivers.get(tx_id)
        if not receiver:
            return make_reply(ReturnCode.OK)
        else:
            return receiver.process(request)

    @classmethod
    def enable(cls, fl_ctx: FLContext, max_request_workers=20, query_interval=5, max_retries=5, max_tx_time=300.0):
        if cls._enabled:
            return

        cls._enabled = True
        cls._max_retries = max_retries
        cls._max_tx_time = max_tx_time
        cls._query_interval = query_interval
        cls._executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_request_workers)
        engine = fl_ctx.get_engine()
        engine.register_aux_message_handler(
            topic=TOPIC_RELIABLE_REQUEST,
            message_handle_func=cls._receive_request,
        )
        engine.register_aux_message_handler(
            topic=TOPIC_RELIABLE_REPLY,
            message_handle_func=cls._receive_reply,
        )
        t = threading.Thread(target=cls._monitor_req_receivers, daemon=True)
        t.start()

    @classmethod
    def _monitor_req_receivers(cls):
        while not cls._shutdown_asked:
            expired_receivers = []
            with cls._tx_lock:
                now = time.time()
                for tx_id, receiver in cls._req_receivers.items():
                    assert isinstance(receiver, _RequestReceiver)
                    if receiver.rcv_time and now - receiver.rcv_time > cls._max_tx_time:
                        expired_receivers.append(tx_id)

            if expired_receivers:
                with cls._tx_lock:
                    for tx_id in expired_receivers:
                        cls._req_receivers.pop(tx_id, None)

            time.sleep(2.0)

    @classmethod
    def shutdown(cls):
        cls._executor.shutdown(cancel_futures=True, wait=False)
        cls._shutdown_asked = True

    @classmethod
    def send_request(
        cls, target: str, topic: str, request: Shareable, timeout: float, abort_signal: Signal, fl_ctx: FLContext
    ) -> Shareable:
        tx_id = str(uuid.uuid4())
        receiver = _ReplyReceiver(tx_id)
        cls._reply_receivers[tx_id] = receiver
        request.set_header(HEADER_TX, tx_id)
        request.set_header(HEADER_OP, OP_REQUEST)
        request.set_header(HEADER_TOPIC, topic)
        request.set_header(HEADER_TIMEOUT, timeout)
        try:
            result = cls._send_request(target, request, timeout, abort_signal, fl_ctx, receiver)
        except Exception as e:
            result = _error_reply(ReturnCode.ERROR, str(e))
        cls._reply_receivers.pop(tx_id)
        return result

    @classmethod
    def _send_request(
        cls,
        target: str,
        request: Shareable,
        timeout: float,
        abort_signal: Signal,
        fl_ctx: FLContext,
        receiver: _ReplyReceiver,
    ) -> Shareable:
        engine = fl_ctx.get_engine()

        # keep sending the request until a positive ack or result is received
        num_tries = 0
        while True:
            if abort_signal and abort_signal.triggered:
                return make_reply(ReturnCode.TASK_ABORTED)

            ack = engine.send_aux_request(
                targets=[target],
                topic=TOPIC_RELIABLE_REQUEST,
                request=request,
                timeout=timeout,
                fl_ctx=fl_ctx,
            )
            ack, rc = _extract_result(ack, target)
            if ack and rc != ReturnCode.COMMUNICATION_ERROR:
                # is this result?
                op = ack.get_header(HEADER_OP)
                if op == OP_REPLY:
                    # the reply is already the result - we are done!
                    # this could happen when we didn't get positive ack for our first request, and the result was
                    # already produced when we did the 2nd request (this request).
                    return ack

                # the ack is a status report - check status
                status = ack.get_header(HEADER_STATUS)
                if status and status != STATUS_NOT_RECEIVED:
                    # status should never be STATUS_NOT_RECEIVED, unless there is a bug in the receiving logic
                    # STATUS_NOT_RECEIVED is only possible during "query" phase.
                    break

            # we didn't get a positive ack - wait a short time and re-send the request.
            num_tries += 1
            if num_tries > cls._max_retries:
                # enough tries
                return _error_reply(ReturnCode.COMMUNICATION_ERROR, f"Max send retries ({cls._max_retries}) reached")
            start = time.time()
            while time.time() - start < cls._query_interval:
                if abort_signal and abort_signal.triggered:
                    return make_reply(ReturnCode.TASK_ABORTED)
                time.sleep(0.1)

        return cls._query_result(target, timeout, abort_signal, fl_ctx, receiver)

    @classmethod
    def _query_result(
        cls,
        target: str,
        timeout: float,
        abort_signal: Signal,
        fl_ctx: FLContext,
        receiver: _ReplyReceiver,
    ) -> Shareable:

        # Querying phase - try to get result
        engine = fl_ctx.get_engine()
        query = Shareable()
        query.set_header(HEADER_TX, receiver.tx_id)
        query.set_header(HEADER_OP, OP_QUERY)

        num_tries = 0
        while True:
            if receiver.result_ready.wait(cls._query_interval):
                # we already received result sent by the target.
                # Note that we don't wait forever here - we only wait for _query_interval so we could
                # check other condition and/or send query to ask for result.
                return receiver.result

            if abort_signal and abort_signal.triggered:
                return make_reply(ReturnCode.TASK_ABORTED)

            # send a query. The ack of the query could be the result itself, or a status report.
            # Note: the ack could be the result because we failed to receive the result sent by the target earlier.
            ack = engine.send_aux_request(
                targets=[target],
                topic=TOPIC_RELIABLE_REQUEST,
                request=query,
                timeout=timeout,
                fl_ctx=fl_ctx,
            )
            ack, rc = _extract_result(ack, target)
            if ack and rc != ReturnCode.COMMUNICATION_ERROR:
                op = ack.get_header(HEADER_OP)
                if op == OP_REPLY:
                    # the ack is result itself!
                    return ack

                status = ack.get_header(HEADER_STATUS)
                if status == STATUS_NOT_RECEIVED:
                    # the receiver side lost context!
                    return _error_reply(ReturnCode.EXECUTION_EXCEPTION, "STATUS_NOT_RECEIVED")
                elif status == STATUS_ABORTED:
                    return _error_reply(ReturnCode.EXECUTION_EXCEPTION, "Aborted")
                else:
                    # the received is in process - do not increase num_tries here!
                    continue

            # retry query
            num_tries += 1
            if num_tries > cls._max_retries:
                return _error_reply(ReturnCode.COMMUNICATION_ERROR, f"Max query retries ({cls._max_retries}) reached")
