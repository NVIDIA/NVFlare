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
import logging
import threading
import time
import uuid

from nvflare.apis.fl_constant import ConfigVarName, SystemConfigs
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import ReservedHeaderKey, ReturnCode, Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.apis.utils.fl_context_utils import generate_log_message
from nvflare.fuel.utils.config_service import ConfigService
from nvflare.fuel.utils.validation_utils import check_positive_number
from nvflare.security.logging import secure_format_exception, secure_format_traceback

# Operation Types
OP_REQUEST = "req"
OP_QUERY = "query"
OP_REPLY = "reply"

# Reliable Message headers
HEADER_OP = "rm.op"
HEADER_TOPIC = "rm.topic"
HEADER_TX_ID = "rm.tx_id"
HEADER_PER_MSG_TIMEOUT = "rm.per_msg_timeout"
HEADER_TX_TIMEOUT = "rm.tx_timeout"
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

PROP_KEY_TX_ID = "RM.TX_ID"
PROP_KEY_TOPIC = "RM.TOPIC"
PROP_KEY_OP = "RM.OP"


def _extract_result(reply: dict, target: str):
    err_rc = ReturnCode.COMMUNICATION_ERROR
    if not isinstance(reply, dict):
        return make_reply(err_rc), err_rc
    result = reply.get(target)
    if not result:
        return make_reply(err_rc), err_rc
    return result, result.get_return_code()


def _status_reply(status: str):
    return make_reply(rc=ReturnCode.OK, headers={HEADER_STATUS: status})


def _error_reply(rc: str, error: str):
    return make_reply(rc, headers={ReservedHeaderKey.ERROR: error})


class _RequestReceiver:
    """This class handles reliable message request on the receiving end"""

    def __init__(self, topic, request_handler_f, executor, per_msg_timeout, tx_timeout):
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
        self.per_msg_timeout = per_msg_timeout
        self.tx_timeout = tx_timeout
        self.rcv_time = None
        self.result = None
        self.source = None
        self.tx_id = None
        self.reply_time = None
        self.replying = False

    def process(self, request: Shareable, fl_ctx: FLContext) -> Shareable:
        self.tx_id = request.get_header(HEADER_TX_ID)
        op = request.get_header(HEADER_OP)
        peer_ctx = fl_ctx.get_peer_context()
        assert isinstance(peer_ctx, FLContext)
        self.source = peer_ctx.get_identity_name()
        if op == OP_REQUEST:
            # it is possible that a new request for the same tx is received while we are processing the previous one
            if not self.rcv_time:
                self.rcv_time = time.time()
                self.per_msg_timeout = request.get_header(HEADER_PER_MSG_TIMEOUT)
                self.tx_timeout = request.get_header(HEADER_TX_TIMEOUT)

                # start processing
                ReliableMessage.debug(fl_ctx, f"started processing request of topic {self.topic}")
                self.executor.submit(self._do_request, request, fl_ctx)
                return _status_reply(STATUS_IN_PROCESS)  # ack
            elif self.result:
                # we already finished processing - send the result back
                ReliableMessage.info(fl_ctx, "resend result back to requester")
                return self.result
            else:
                # we are still processing
                ReliableMessage.info(fl_ctx, "got request - the request is being processed")
                return _status_reply(STATUS_IN_PROCESS)
        elif op == OP_QUERY:
            if self.result:
                if self.reply_time:
                    # result already sent back successfully
                    ReliableMessage.info(fl_ctx, "got query: we already replied successfully")
                    return _status_reply(STATUS_REPLIED)
                elif self.replying:
                    # result is being sent
                    ReliableMessage.info(fl_ctx, "got query: reply is being sent")
                    return _status_reply(STATUS_IN_REPLY)
                else:
                    # try to send the result again
                    ReliableMessage.info(fl_ctx, "got query: sending reply again")
                    return self.result
            else:
                # still in process
                if time.time() - self.rcv_time > self.tx_timeout:
                    # the process is taking too much time
                    ReliableMessage.error(fl_ctx, f"aborting processing since exceeded max tx time {self.tx_timeout}")
                    return _status_reply(STATUS_ABORTED)
                else:
                    ReliableMessage.debug(fl_ctx, "got query: request is in-process")
                    return _status_reply(STATUS_IN_PROCESS)

    def _try_reply(self, fl_ctx: FLContext):
        engine = fl_ctx.get_engine()
        self.replying = True
        start_time = time.time()
        ReliableMessage.debug(fl_ctx, f"try to send reply back to {self.source}: {self.per_msg_timeout=}")
        ack = engine.send_aux_request(
            targets=[self.source],
            topic=TOPIC_RELIABLE_REPLY,
            request=self.result,
            timeout=self.per_msg_timeout,
            fl_ctx=fl_ctx,
        )
        time_spent = time.time() - start_time
        self.replying = False
        _, rc = _extract_result(ack, self.source)
        if rc == ReturnCode.OK:
            # reply sent successfully!
            self.reply_time = time.time()
            ReliableMessage.debug(fl_ctx, f"sent reply successfully in {time_spent} secs")
        else:
            ReliableMessage.error(
                fl_ctx, f"failed to send reply in {time_spent} secs: {rc=}; will wait for requester to query"
            )

    def _do_request(self, request: Shareable, fl_ctx: FLContext):
        start_time = time.time()
        ReliableMessage.debug(fl_ctx, "invoking request handler")
        try:
            result = self.request_handler_f(self.topic, request, fl_ctx)
        except Exception as e:
            ReliableMessage.error(fl_ctx, f"exception processing request: {secure_format_traceback()}")
            result = _error_reply(ReturnCode.EXECUTION_EXCEPTION, secure_format_exception(e))

        # send back
        result.set_header(HEADER_TX_ID, self.tx_id)
        result.set_header(HEADER_OP, OP_REPLY)
        result.set_header(HEADER_TOPIC, self.topic)
        self.result = result
        ReliableMessage.debug(fl_ctx, f"finished request handler in {time.time()-start_time} secs")
        self._try_reply(fl_ctx)


class _ReplyReceiver:
    def __init__(self, tx_id: str, per_msg_timeout: float, tx_timeout: float):
        self.tx_id = tx_id
        self.tx_start_time = time.time()
        self.tx_timeout = tx_timeout
        self.per_msg_timeout = per_msg_timeout
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
    _reply_receivers = {}  # tx id => receiver
    _tx_lock = threading.Lock()
    _shutdown_asked = False
    _logger = logging.getLogger("ReliableMessage")

    @classmethod
    def register_request_handler(cls, topic: str, handler_f, fl_ctx: FLContext):
        """Register a handler for the reliable message with this topic

        Args:
            topic: The topic of the reliable message
            handler_f: The callback function to handle the request in the form of
                handler_f(topic, request, fl_ctx)
            fl_ctx: FL Context
        """
        if not cls._enabled:
            raise RuntimeError("ReliableMessage is not enabled. Please call ReliableMessage.enable() to enable it")
        if not callable(handler_f):
            raise TypeError(f"handler_f must be callable but {type(handler_f)}")
        cls._topic_to_handle[topic] = handler_f

        # ReliableMessage also sends aux message directly if tx_timeout is too small
        engine = fl_ctx.get_engine()
        engine.register_aux_message_handler(
            topic=topic,
            message_handle_func=handler_f,
        )

    @classmethod
    def _get_or_create_receiver(cls, topic: str, request: Shareable, handler_f) -> _RequestReceiver:
        tx_id = request.get_header(HEADER_TX_ID)
        if not tx_id:
            raise RuntimeError("missing tx_id in request")
        with cls._tx_lock:
            receiver = cls._req_receivers.get(tx_id)
            if not receiver:
                per_msg_timeout = request.get_header(HEADER_PER_MSG_TIMEOUT)
                if not per_msg_timeout:
                    raise RuntimeError("missing per_msg_timeout in request")
                tx_timeout = request.get_header(HEADER_TX_TIMEOUT)
                if not tx_timeout:
                    raise RuntimeError("missing tx_timeout in request")
                receiver = _RequestReceiver(topic, handler_f, cls._executor, per_msg_timeout, tx_timeout)
                cls._req_receivers[tx_id] = receiver
            return receiver

    @classmethod
    def _receive_request(cls, topic: str, request: Shareable, fl_ctx: FLContext):
        tx_id = request.get_header(HEADER_TX_ID)
        op = request.get_header(HEADER_OP)
        rm_topic = request.get_header(HEADER_TOPIC)
        fl_ctx.set_prop(key=PROP_KEY_TX_ID, value=tx_id, sticky=False, private=True)
        fl_ctx.set_prop(key=PROP_KEY_OP, value=op, sticky=False, private=True)
        fl_ctx.set_prop(key=PROP_KEY_TOPIC, value=rm_topic, sticky=False, private=True)
        cls.debug(fl_ctx, f"received aux msg ({topic=}) for RM request")

        if op == OP_REQUEST:
            handler_f = cls._topic_to_handle.get(rm_topic)
            if not handler_f:
                # no handler registered for this topic!
                cls.error(fl_ctx, f"no handler registered for request {rm_topic=}")
                return make_reply(ReturnCode.TOPIC_UNKNOWN)
            receiver = cls._get_or_create_receiver(rm_topic, request, handler_f)
            cls.debug(fl_ctx, f"received request {rm_topic=}")
            return receiver.process(request, fl_ctx)
        elif op == OP_QUERY:
            receiver = cls._req_receivers.get(tx_id)
            if not receiver:
                cls.error(fl_ctx, f"received query but the request ({rm_topic=}) is not received!")
                return _status_reply(STATUS_NOT_RECEIVED)  # meaning the request wasn't received
            else:
                return receiver.process(request, fl_ctx)
        else:
            cls.error(fl_ctx, f"received invalid op {op} for the request ({rm_topic=})")
            return make_reply(rc=ReturnCode.BAD_REQUEST_DATA)

    @classmethod
    def _receive_reply(cls, topic: str, request: Shareable, fl_ctx: FLContext):
        tx_id = request.get_header(HEADER_TX_ID)
        fl_ctx.set_prop(key=PROP_KEY_TX_ID, value=tx_id, private=True, sticky=False)
        cls.debug(fl_ctx, f"received aux msg ({topic=}) for RM reply")
        receiver = cls._reply_receivers.get(tx_id)
        if not receiver:
            cls.error(fl_ctx, "received reply but we are no longer waiting for it")
        else:
            assert isinstance(receiver, _ReplyReceiver)
            cls.debug(fl_ctx, f"received reply in {time.time()-receiver.tx_start_time} secs - set waiter")
            receiver.process(request)
        return make_reply(ReturnCode.OK)

    @classmethod
    def enable(cls, fl_ctx: FLContext):
        """Enable ReliableMessage. This method can be called multiple times, but only the 1st call has effect.

        Args:
            fl_ctx: FL Context

        Returns:

        """
        if cls._enabled:
            return

        cls._enabled = True
        max_request_workers = ConfigService.get_int_var(
            name=ConfigVarName.RM_MAX_REQUEST_WORKERS, conf=SystemConfigs.APPLICATION_CONF, default=20
        )
        query_interval = ConfigService.get_float_var(
            name=ConfigVarName.RM_QUERY_INTERVAL, conf=SystemConfigs.APPLICATION_CONF, default=2.0
        )

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
        cls._logger.info(f"enabled reliable message: {max_request_workers=} {query_interval=}")

    @classmethod
    def _monitor_req_receivers(cls):
        while not cls._shutdown_asked:
            expired_receivers = []
            with cls._tx_lock:
                now = time.time()
                for tx_id, receiver in cls._req_receivers.items():
                    assert isinstance(receiver, _RequestReceiver)
                    if receiver.rcv_time and now - receiver.rcv_time > 4 * receiver.tx_timeout:
                        cls._logger.info(f"detected expired request receiver {tx_id}")
                        expired_receivers.append(tx_id)

            if expired_receivers:
                with cls._tx_lock:
                    for tx_id in expired_receivers:
                        cls._req_receivers.pop(tx_id, None)

            time.sleep(2.0)
        cls._logger.info("shutdown reliable message monitor")

    @classmethod
    def shutdown(cls):
        """Shutdown ReliableMessage.

        Returns:

        """
        if not cls._shutdown_asked:
            cls._shutdown_asked = True
            cls._executor.shutdown(wait=False)
            cls._logger.info("ReliableMessage is shutdown")

    @classmethod
    def _log_msg(cls, fl_ctx: FLContext, msg: str):
        props = []
        tx_id = fl_ctx.get_prop(PROP_KEY_TX_ID)
        if tx_id:
            props.append(f"rm_tx={tx_id}")

        op = fl_ctx.get_prop(PROP_KEY_OP)
        if op:
            props.append(f"rm_op={op}")

        topic = fl_ctx.get_prop(PROP_KEY_TOPIC)
        if topic:
            props.append(f"rm_topic={topic}")

        rm_ctx = ""
        if props:
            rm_ctx = " ".join(props)

        if rm_ctx:
            msg = f"[{rm_ctx}] {msg}"
        return generate_log_message(fl_ctx, msg)

    @classmethod
    def info(cls, fl_ctx: FLContext, msg: str):
        cls._logger.info(cls._log_msg(fl_ctx, msg))

    @classmethod
    def error(cls, fl_ctx: FLContext, msg: str):
        cls._logger.error(cls._log_msg(fl_ctx, msg))

    @classmethod
    def debug(cls, fl_ctx: FLContext, msg: str):
        cls._logger.debug(cls._log_msg(fl_ctx, msg))

    @classmethod
    def send_request(
        cls,
        target: str,
        topic: str,
        request: Shareable,
        per_msg_timeout: float,
        tx_timeout: float,
        abort_signal: Signal,
        fl_ctx: FLContext,
    ) -> Shareable:
        """Send a reliable request.

        Args:
            target: the target cell of this request
            topic: topic of the request;
            request: the request to be sent
            per_msg_timeout: timeout when sending a message
            tx_timeout: the timeout of the whole transaction
            abort_signal: abort signal
            fl_ctx: the FL context

        Returns: reply from the peer.

        If tx_timeout is not specified or <= per_msg_timeout, the request will be sent only once without retrying.

        """
        check_positive_number("per_msg_timeout", per_msg_timeout)
        if tx_timeout:
            check_positive_number("tx_timeout", tx_timeout)

        if not tx_timeout or tx_timeout <= per_msg_timeout:
            # simple aux message
            cls.info(fl_ctx, f"send request with simple Aux Msg: {per_msg_timeout=} {tx_timeout=}")
            engine = fl_ctx.get_engine()
            reply = engine.send_aux_request(
                targets=[target],
                topic=topic,
                request=request,
                timeout=per_msg_timeout,
                fl_ctx=fl_ctx,
            )
            result, _ = _extract_result(reply, target)
            return result

        tx_id = str(uuid.uuid4())
        fl_ctx.set_prop(key=PROP_KEY_TX_ID, value=tx_id, private=True, sticky=False)
        cls.info(fl_ctx, f"send request with Reliable Msg {per_msg_timeout=} {tx_timeout=}")
        receiver = _ReplyReceiver(tx_id, per_msg_timeout, tx_timeout)
        cls._reply_receivers[tx_id] = receiver
        request.set_header(HEADER_TX_ID, tx_id)
        request.set_header(HEADER_OP, OP_REQUEST)
        request.set_header(HEADER_TOPIC, topic)
        request.set_header(HEADER_PER_MSG_TIMEOUT, per_msg_timeout)
        request.set_header(HEADER_TX_TIMEOUT, tx_timeout)
        try:
            result = cls._send_request(target, request, abort_signal, fl_ctx, receiver)
        except Exception as e:
            cls.error(fl_ctx, f"exception sending reliable message: {secure_format_traceback()}")
            result = _error_reply(ReturnCode.ERROR, secure_format_exception(e))
        cls._reply_receivers.pop(tx_id)
        return result

    @classmethod
    def _send_request(
        cls,
        target: str,
        request: Shareable,
        abort_signal: Signal,
        fl_ctx: FLContext,
        receiver: _ReplyReceiver,
    ) -> Shareable:
        engine = fl_ctx.get_engine()

        # keep sending the request until a positive ack or result is received
        tx_timeout = receiver.tx_timeout
        per_msg_timeout = receiver.per_msg_timeout
        num_tries = 0
        while True:
            if abort_signal and abort_signal.triggered:
                cls.info(fl_ctx, "send_request abort triggered")
                return make_reply(ReturnCode.TASK_ABORTED)

            if time.time() - receiver.tx_start_time >= receiver.tx_timeout:
                cls.error(fl_ctx, f"aborting send_request since exceeded {tx_timeout=}")
                return make_reply(ReturnCode.COMMUNICATION_ERROR)

            if num_tries > 0:
                cls.debug(fl_ctx, f"retry #{num_tries} sending request: {per_msg_timeout=}")

            ack = engine.send_aux_request(
                targets=[target],
                topic=TOPIC_RELIABLE_REQUEST,
                request=request,
                timeout=per_msg_timeout,
                fl_ctx=fl_ctx,
            )
            ack, rc = _extract_result(ack, target)
            if ack and rc not in [ReturnCode.COMMUNICATION_ERROR]:
                # is this result?
                op = ack.get_header(HEADER_OP)
                if op == OP_REPLY:
                    # the reply is already the result - we are done!
                    # this could happen when we didn't get positive ack for our first request, and the result was
                    # already produced when we did the 2nd request (this request).
                    cls.debug(fl_ctx, f"C1: received result in {time.time()-receiver.tx_start_time} seconds; {rc=}")
                    return ack

                # the ack is a status report - check status
                status = ack.get_header(HEADER_STATUS)
                if status and status != STATUS_NOT_RECEIVED:
                    # status should never be STATUS_NOT_RECEIVED, unless there is a bug in the receiving logic
                    # STATUS_NOT_RECEIVED is only possible during "query" phase.
                    cls.debug(fl_ctx, f"received status ack: {rc=} {status=}")
                    break

            if time.time() + cls._query_interval - receiver.tx_start_time >= tx_timeout:
                cls.error(fl_ctx, f"aborting send_request since it will exceed {tx_timeout=}")
                return make_reply(ReturnCode.COMMUNICATION_ERROR)

            # we didn't get a positive ack - wait a short time and re-send the request.
            cls.debug(fl_ctx, f"unsure the request was received ({rc=}): will retry in {cls._query_interval} secs")
            num_tries += 1
            start = time.time()
            while time.time() - start < cls._query_interval:
                if abort_signal and abort_signal.triggered:
                    cls.info(fl_ctx, "abort send_request triggered by signal")
                    return make_reply(ReturnCode.TASK_ABORTED)
                time.sleep(0.1)

        cls.debug(fl_ctx, "request was received by the peer - will query for result")
        return cls._query_result(target, abort_signal, fl_ctx, receiver)

    @classmethod
    def _query_result(
        cls,
        target: str,
        abort_signal: Signal,
        fl_ctx: FLContext,
        receiver: _ReplyReceiver,
    ) -> Shareable:
        tx_timeout = receiver.tx_timeout
        per_msg_timeout = receiver.per_msg_timeout

        # Querying phase - try to get result
        engine = fl_ctx.get_engine()
        query = Shareable()
        query.set_header(HEADER_TX_ID, receiver.tx_id)
        query.set_header(HEADER_OP, OP_QUERY)

        num_tries = 0
        last_query_time = 0
        short_wait = 0.1
        while True:
            if time.time() - receiver.tx_start_time > tx_timeout:
                cls.error(fl_ctx, f"aborted query since exceeded {tx_timeout=}")
                return _error_reply(ReturnCode.COMMUNICATION_ERROR, f"max tx timeout ({tx_timeout}) reached")

            if receiver.result_ready.wait(short_wait):
                # we already received result sent by the target.
                # Note that we don't wait forever here - we only wait for _query_interval, so we could
                # check other condition and/or send query to ask for result.
                cls.debug(fl_ctx, f"C2: received result in {time.time()-receiver.tx_start_time} seconds")
                return receiver.result

            if abort_signal and abort_signal.triggered:
                cls.info(fl_ctx, "aborted query triggered by abort signal")
                return make_reply(ReturnCode.TASK_ABORTED)

            if time.time() - last_query_time < cls._query_interval:
                # don't query too quickly
                continue

            # send a query. The ack of the query could be the result itself, or a status report.
            # Note: the ack could be the result because we failed to receive the result sent by the target earlier.
            num_tries += 1
            cls.debug(fl_ctx, f"query #{num_tries}: try to get result from {target}: {per_msg_timeout=}")
            ack = engine.send_aux_request(
                targets=[target],
                topic=TOPIC_RELIABLE_REQUEST,
                request=query,
                timeout=per_msg_timeout,
                fl_ctx=fl_ctx,
            )
            last_query_time = time.time()
            ack, rc = _extract_result(ack, target)
            if ack and rc not in [ReturnCode.COMMUNICATION_ERROR]:
                op = ack.get_header(HEADER_OP)
                if op == OP_REPLY:
                    # the ack is result itself!
                    cls.debug(fl_ctx, f"C3: received result in {time.time()-receiver.tx_start_time} seconds")
                    return ack

                status = ack.get_header(HEADER_STATUS)
                if status == STATUS_NOT_RECEIVED:
                    # the receiver side lost context!
                    cls.error(fl_ctx, f"peer {target} lost request!")
                    return _error_reply(ReturnCode.EXECUTION_EXCEPTION, "STATUS_NOT_RECEIVED")
                elif status == STATUS_ABORTED:
                    cls.error(fl_ctx, f"peer {target} aborted processing!")
                    return _error_reply(ReturnCode.EXECUTION_EXCEPTION, "Aborted")

                cls.debug(fl_ctx, f"will retry query in {cls._query_interval} secs: {rc=} {status=} {op=}")
            else:
                cls.debug(fl_ctx, f"will retry query in {cls._query_interval} secs: {rc=}")
