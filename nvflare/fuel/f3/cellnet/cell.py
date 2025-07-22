# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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
import copy
import threading
import uuid
from typing import Dict, List, Union

from nvflare.apis.signal import Signal
from nvflare.fuel.f3.cellnet.core_cell import CoreCell, TargetMessage
from nvflare.fuel.f3.cellnet.defs import CellChannel, MessageHeaderKey, MessagePropKey, MessageType, ReturnCode
from nvflare.fuel.f3.cellnet.utils import decode_payload, encode_payload, make_reply
from nvflare.fuel.f3.message import Message
from nvflare.fuel.f3.stream_cell import StreamCell
from nvflare.fuel.f3.streaming.stream_const import StreamHeaderKey
from nvflare.fuel.f3.streaming.stream_types import StreamFuture
from nvflare.fuel.utils.fobs import FOBSContextKey
from nvflare.fuel.utils.log_utils import get_obj_logger
from nvflare.fuel.utils.waiter_utils import WaiterRC, conditional_wait
from nvflare.security.logging import secure_format_exception

CHANNELS_TO_EXCLUDE = (
    CellChannel.CLIENT_MAIN,
    CellChannel.SERVER_MAIN,
    CellChannel.SERVER_PARENT_LISTENER,
    CellChannel.CLIENT_COMMAND,
    CellChannel.CLIENT_SUB_WORKER_COMMAND,
    CellChannel.MULTI_PROCESS_EXECUTOR,
    CellChannel.SIMULATOR_RUNNER,
    CellChannel.RETURN_ONLY,
)


def _is_stream_channel(channel: str) -> bool:
    if channel is None or channel == "":
        return False
    elif channel in CHANNELS_TO_EXCLUDE:
        return False
    # if not excluded, all channels supporting streaming capabilities
    return True


class SimpleWaiter:
    def __init__(self, req_id, result):
        super().__init__()
        self.req_id = req_id
        self.result = result
        self.receiving_future = None
        self.in_receiving = threading.Event()


class Adapter:
    def __init__(self, cb, my_info, cell):
        self.cb = cb
        self.my_info = my_info
        self.cell = cell
        self.logger = get_obj_logger(self)

    def call(self, future, *args, **kwargs):  # this will be called by StreamCell upon receiving the first byte of blob
        headers = future.headers
        stream_req_id = headers.get(StreamHeaderKey.STREAM_REQ_ID, "")
        origin = headers.get(MessageHeaderKey.ORIGIN, None)
        result = future.result()
        self.logger.debug(f"{stream_req_id=}: {headers=}, incoming data={result}")
        request = Message(headers, result)

        decode_payload(request, StreamHeaderKey.PAYLOAD_ENCODING, fobs_ctx=self.cell.get_fobs_context())

        channel = request.get_header(StreamHeaderKey.CHANNEL)
        request.set_header(MessageHeaderKey.CHANNEL, channel)
        topic = request.get_header(StreamHeaderKey.TOPIC)
        request.set_header(MessageHeaderKey.TOPIC, topic)
        self.logger.debug(f"Call back on {stream_req_id=}: {channel=}, {topic=}")

        req_id = request.get_header(MessageHeaderKey.REQ_ID, "")
        secure = request.get_header(MessageHeaderKey.SECURE, False)
        optional = request.get_header(MessageHeaderKey.OPTIONAL, False)
        self.logger.debug(f"{stream_req_id=}: on {channel=}, {topic=}")
        response = self.cb(request, *args, **kwargs)
        self.logger.debug(f"response available: {stream_req_id=}: on {channel=}, {topic=}")

        if not stream_req_id:
            # no need to reply!
            self.logger.debug("Do not send reply because there is no stream_req_id!")
            return

        response.add_headers(
            {
                MessageHeaderKey.REQ_ID: req_id,
                MessageHeaderKey.MSG_TYPE: MessageType.REPLY,
                StreamHeaderKey.STREAM_REQ_ID: stream_req_id,
            }
        )

        encode_payload(response, StreamHeaderKey.PAYLOAD_ENCODING, fobs_ctx=self.cell.get_fobs_context())
        self.logger.debug(f"sending: {stream_req_id=}: {response.headers=}, target={origin}")
        reply_future = self.cell.send_blob(
            CellChannel.RETURN_ONLY, f"{channel}:{topic}", origin, response, secure, optional
        )
        self.logger.debug(f"Done sending: {stream_req_id=}: {reply_future=}")


class Cell(StreamCell):
    def __init__(self, *args, **kwargs):
        self.core_cell = CoreCell(*args, **kwargs)
        super().__init__(self.core_cell)
        self.requests_dict = dict()
        self.logger = get_obj_logger(self)
        self.register_blob_cb(CellChannel.RETURN_ONLY, "*", self._process_reply)  # this should be one-time registration
        self.core_cell.update_fobs_context({FOBSContextKey.CELL: self})

    def update_fobs_context(self, props: dict):
        self.core_cell.update_fobs_context(props)

    def get_fobs_context(self, props: dict = None):
        """Return a new copy of the fobs context. If props is specified, they will be set into the context.

        Returns: a new copy of the fobs context

        """
        ctx = self.core_cell.get_fobs_context()
        if props:
            ctx.update(props)
        return ctx

    def __getattr__(self, func):
        """
        This method is called when Python cannot find an invoked method "x" of this class.
        Method "x" is one of the message sending methods (send_request, broadcast_request, etc.)
        In this method, we decide which method should be used instead, based on the "channel" of the message.
        - If the channel is stream channel, use the method "_x" of this class.
        - Otherwise, user the method "x" of the CoreCell.
        """

        def method(*args, **kwargs):
            self.logger.debug(f"__getattr__: {args=}, {kwargs=}")
            if _is_stream_channel(kwargs.get("channel")):
                self.logger.debug(f"calling cell {func}")
                return getattr(self, f"_{func}")(*args, **kwargs)
            if not hasattr(self.core_cell, func):
                raise AttributeError(f"'{func}' not in core_cell.")
            self.logger.debug(f"calling core_cell {func}")
            return getattr(self.core_cell, func)(*args, **kwargs)

        return method

    def _broadcast_request(
        self,
        channel: str,
        topic: str,
        targets: Union[str, List[str]],
        request: Message,
        timeout=None,
        secure=False,
        optional=False,
        abort_signal: Signal = None,
    ) -> Dict[str, Message]:
        """
        Send a message over a channel to specified destination cell(s), and wait for reply

        Args:
            channel: channel for the message
            topic: topic of the message
            targets: FQCN of the destination cell(s)
            request: message to be sent
            timeout: how long to wait for replies
            secure: End-end encryption
            optional: whether the message is optional
            abort_signal: signal to abort the message

        Returns: a dict of: cell_id => reply message

        """
        self.logger.debug(f"broadcast: {channel=}, {topic=}, {targets=}, {timeout=}")

        if isinstance(targets, str):
            targets = [targets]
        target_argument = {}
        fixed_dict = dict(channel=channel, topic=topic, timeout=timeout, secure=secure, optional=optional)
        results = dict()
        future_to_target = {}

        # encode the request now so each target thread won't need to do it again.
        self._encode_message(request, abort_signal)

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(targets)) as executor:
            self.logger.debug(f"broadcast to {targets=}")
            for t in targets:
                req = Message(copy.deepcopy(request.headers), request.payload)
                target_argument["request"] = TargetMessage(t, channel, topic, req).message
                target_argument["target"] = t
                target_argument["abort_signal"] = abort_signal
                target_argument.update(fixed_dict)
                f = executor.submit(self._send_one_request, **target_argument)
                future_to_target[f] = t
                self.logger.debug(f"submitted to {t} with {target_argument.keys()=}")
            for future in concurrent.futures.as_completed(future_to_target):
                target = future_to_target[future]
                self.logger.debug(f"{target} completed")
                try:
                    data = future.result()
                except Exception as exc:
                    self.logger.warning(f"{target} raises {exc}")
                    results[target] = make_reply(ReturnCode.TIMEOUT)
                else:
                    results[target] = data
                    self.logger.debug(f"{target=}: {data=}")
        self.logger.debug("About to return from broadcast_request")
        return results

    def _fire_and_forget(
        self,
        channel: str,
        topic: str,
        targets: Union[str, List[str]],
        message: Message,
        secure=False,
        optional=False,
    ) -> Dict[str, str]:
        """
        Send a message over a channel to specified destination cell(s), and do not wait for replies.

        Args:
            channel: channel for the message
            topic: topic of the message
            targets: one or more destination cell IDs. None means all.
            message: message to be sent
            secure: End-end encryption if True
            optional: whether the message is optional

        Returns: None

        """
        encode_payload(message, encoding_key=StreamHeaderKey.PAYLOAD_ENCODING, fobs_ctx=self.get_fobs_context())
        if isinstance(targets, str):
            targets = [targets]

        result = {}
        futures = {}
        for target in targets:
            future = self.send_blob(
                channel=channel, topic=topic, target=target, message=message, secure=secure, optional=optional
            )
            futures[target] = future
            result[target] = ""
        message.set_prop(MessagePropKey.FUTURES, futures)

        return result

    def _get_result(self, req_id):
        waiter = self.requests_dict.pop(req_id)
        return waiter.result

    def _check_error(self, future):
        if future.error:
            # must return a negative number
            return -1
        else:
            return WaiterRC.OK

    def _future_wait(self, future, timeout, abort_signal: Signal):
        # future could have an error!
        last_progress = 0
        while True:
            rc = conditional_wait(future.waiter, timeout, abort_signal, condition_cb=self._check_error, future=future)
            if rc == WaiterRC.IS_SET:
                # waiter has been set!
                break
            elif rc == WaiterRC.TIMEOUT:
                # timed out: check whether any progress has been made during this time
                current_progress = future.get_progress()
                if last_progress == current_progress:
                    # no progress in timeout secs: consider this to be a failure
                    return False
                else:
                    # good progress
                    self.logger.debug(f"{current_progress=}")
                    last_progress = current_progress
            else:
                # error condition: aborted or future error
                return False

        if future.error:
            return False
        else:
            return True

    def _encode_message(self, msg: Message, abort_signal) -> int:
        try:
            return encode_payload(
                msg,
                StreamHeaderKey.PAYLOAD_ENCODING,
                fobs_ctx=self.get_fobs_context({FOBSContextKey.ABORT_SIGNAL: abort_signal}),
            )
        except BaseException as exc:
            self.logger.error(f"Can't encode {msg=} {exc=}")
            raise exc

    def _send_request(
        self,
        channel,
        target,
        topic,
        request,
        timeout=10.0,
        secure=False,
        optional=False,
        abort_signal: Signal = None,
    ):
        """Stream one request to the target

        Args:
            channel: message channel name
            target: FQCN of the target cell
            topic: topic of the message
            request: request message
            timeout: how long to wait
            secure: is P2P security to be applied
            optional: is the message optional
            abort_signal: signal to abort the message

        Returns: reply data

        """
        self._encode_message(request, abort_signal)
        return self._send_one_request(channel, target, topic, request, timeout, secure, optional, abort_signal)

    def _send_one_request(
        self,
        channel,
        target,
        topic,
        request,
        timeout=10.0,
        secure=False,
        optional=False,
        abort_signal=None,
    ):
        req_id = str(uuid.uuid4())
        request.add_headers({StreamHeaderKey.STREAM_REQ_ID: req_id})

        # this future can be used to check sending progress, but not for checking return blob
        self.logger.debug(f"{req_id=}, {channel=}, {topic=}, {target=}, {timeout=}: send_request about to send_blob")

        waiter = SimpleWaiter(req_id=req_id, result=make_reply(ReturnCode.TIMEOUT))
        self.requests_dict[req_id] = waiter

        try:
            future = self.send_blob(
                channel=channel, topic=topic, target=target, message=request, secure=secure, optional=optional
            )

            self.logger.debug(f"{req_id=}: Waiting starts")

            # Three stages, sending, waiting for receiving first byte, receiving
            # sending with progress timeout
            self.logger.debug(f"{req_id=}: entering sending wait {timeout=}")
            sending_complete = self._future_wait(future, timeout, abort_signal)
            if not sending_complete:
                self.logger.debug(f"{req_id=}: sending timeout {timeout=}")
                return self._get_result(req_id)

            self.logger.debug(f"{req_id=}: sending complete")

            # waiting for receiving first byte
            self.logger.debug(f"{req_id=}: entering remote process wait {timeout=}")

            waiter_rc = conditional_wait(waiter.in_receiving, timeout, abort_signal)
            if waiter_rc != WaiterRC.IS_SET:
                self.logger.debug(f"{req_id=}: remote processing timeout {timeout=} {waiter_rc=}")
                return self._get_result(req_id)
            self.logger.debug(f"{req_id=}: in receiving")

            # receiving with progress timeout
            r_future = waiter.receiving_future
            self.logger.debug(f"{req_id=}: entering receiving wait {timeout=}")
            receiving_complete = self._future_wait(r_future, timeout, abort_signal)
            if not receiving_complete:
                self.logger.info(f"{req_id=}: receiving timeout {timeout=}")
                return self._get_result(req_id)
            self.logger.debug(f"{req_id=}: receiving complete")
            waiter.result = Message(r_future.headers, r_future.result())
            decode_payload(
                waiter.result,
                encoding_key=StreamHeaderKey.PAYLOAD_ENCODING,
                fobs_ctx=self.get_fobs_context(props={FOBSContextKey.ABORT_SIGNAL: abort_signal}),
            )
            self.logger.debug(f"{req_id=}: return result {waiter.result=}")
            return self._get_result(req_id)
        except Exception as ex:
            self.logger.error(f"exception sending request: {secure_format_exception(ex)}")
            return self._get_result(req_id)

    def _process_reply(self, future: StreamFuture):
        headers = future.headers
        req_id = headers.get(StreamHeaderKey.STREAM_REQ_ID, -1)
        self.logger.debug(f"{req_id=}: _process_reply")
        try:
            waiter = self.requests_dict[req_id]
        except KeyError as e:
            self.logger.warning(f"Receiving unknown {req_id=}, discarded: {e} headers: {headers}")
            return
        waiter.receiving_future = future
        waiter.in_receiving.set()

    def _register_request_cb(self, channel: str, topic: str, cb, *args, **kwargs):
        """
        Register a callback for handling request. The CB must follow request_cb_signature.

        Args:
            channel: the channel of the request
            topic: topic of the request
            cb:
            *args:
            **kwargs:

        Returns:

        """

        if not callable(cb):
            raise ValueError(f"specified request_cb {type(cb)} is not callable")

        # always register with core_cell since some requests (e.g. broadcast_multi_requests) will directly go
        # through the core_cell, even if the channel may be a stream channel (e.g. aux channel).
        self.core_cell.register_request_cb(channel, topic, cb, *args, **kwargs)
        if _is_stream_channel(channel):
            self.logger.info(f"Register blob CB for {channel=}, {topic=}")
            adapter = Adapter(cb, self.core_cell.my_info, self)
            self.register_blob_cb(channel, topic, adapter.call, *args, **kwargs)
