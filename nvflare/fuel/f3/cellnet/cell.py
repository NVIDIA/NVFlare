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
import logging
import threading
import uuid
from typing import Dict, List, Union

from nvflare.fuel.f3.cellnet.core_cell import CoreCell, TargetMessage
from nvflare.fuel.f3.cellnet.defs import CellChannel, MessageHeaderKey, MessageType, ReturnCode
from nvflare.fuel.f3.cellnet.utils import decode_payload, encode_payload, make_reply
from nvflare.fuel.f3.message import Message
from nvflare.fuel.f3.stream_cell import StreamCell
from nvflare.fuel.f3.streaming.stream_const import StreamHeaderKey
from nvflare.fuel.f3.streaming.stream_types import StreamFuture
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
        self.logger = logging.getLogger(self.__class__.__name__)

    def call(self, future):  # this will be called by StreamCell upon receiving the first byte of blob
        headers = future.headers
        stream_req_id = headers.get(StreamHeaderKey.STREAM_REQ_ID, "")
        origin = headers.get(MessageHeaderKey.ORIGIN, None)
        result = future.result()
        self.logger.debug(f"{stream_req_id=}: {headers=}, incoming data={result}")
        request = Message(headers, result)

        decode_payload(request, StreamHeaderKey.PAYLOAD_ENCODING)

        channel = request.get_header(StreamHeaderKey.CHANNEL)
        request.set_header(MessageHeaderKey.CHANNEL, channel)
        topic = request.get_header(StreamHeaderKey.TOPIC)
        request.set_header(MessageHeaderKey.TOPIC, topic)
        self.logger.debug(f"Call back on {stream_req_id=}: {channel=}, {topic=}")

        req_id = request.get_header(MessageHeaderKey.REQ_ID, "")
        secure = request.get_header(MessageHeaderKey.SECURE, False)
        optional = request.get_header(MessageHeaderKey.OPTIONAL, False)
        self.logger.debug(f"{stream_req_id=}: on {channel=}, {topic=}")
        response = self.cb(request)
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

        encode_payload(response, StreamHeaderKey.PAYLOAD_ENCODING)
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
        self.logger = logging.getLogger(self.__class__.__name__)
        self.register_blob_cb(CellChannel.RETURN_ONLY, "*", self._process_reply)  # this should be one-time registration

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

        Returns: a dict of: cell_id => reply message

        """
        self.logger.info(f"broadcast: {channel=}, {topic=}, {targets=}, {timeout=}")

        if isinstance(targets, str):
            targets = [targets]
        target_argument = {}
        fixed_dict = dict(channel=channel, topic=topic, timeout=timeout, secure=secure, optional=optional)
        results = dict()
        future_to_target = {}

        # encode the request now so each target thread won't need to do it again.
        self._encode_message(request)

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(targets)) as executor:
            self.logger.debug(f"broadcast to {targets=}")
            for t in targets:
                req = Message(copy.deepcopy(request.headers), request.payload)
                target_argument["request"] = TargetMessage(t, channel, topic, req).message
                target_argument["target"] = t
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
        self, channel: str, topic: str, targets: Union[str, List[str]], message: Message, secure=False, optional=False
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
        encode_payload(message, encoding_key=StreamHeaderKey.PAYLOAD_ENCODING)
        if isinstance(targets, str):
            targets = [targets]

        result = {}
        for target in targets:
            self.send_blob(
                channel=channel, topic=topic, target=target, message=message, secure=secure, optional=optional
            )
            result[target] = ""
        return result

    def _get_result(self, req_id):
        waiter = self.requests_dict.pop(req_id)
        return waiter.result

    def _future_wait(self, future, timeout):
        # future could have an error!
        last_progress = 0
        while not future.waiter.wait(timeout):
            if future.error:
                return False
            current_progress = future.get_progress()
            if last_progress == current_progress:
                return False
            else:
                self.logger.debug(f"{current_progress=}")
                last_progress = current_progress
        if future.error:
            return False
        else:
            return True

    def _encode_message(self, msg: Message):
        try:
            encode_payload(msg, StreamHeaderKey.PAYLOAD_ENCODING)
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

        Returns: reply data

        """
        self._encode_message(request)
        return self._send_one_request(channel, target, topic, request, timeout, secure, optional)

    def _send_one_request(
        self,
        channel,
        target,
        topic,
        request,
        timeout=10.0,
        secure=False,
        optional=False,
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
            sending_complete = self._future_wait(future, timeout)
            if not sending_complete:
                self.logger.debug(f"{req_id=}: sending timeout {timeout=}")
                return self._get_result(req_id)

            self.logger.debug(f"{req_id=}: sending complete")

            # waiting for receiving first byte
            self.logger.debug(f"{req_id=}: entering remote process wait {timeout=}")
            if not waiter.in_receiving.wait(timeout):
                self.logger.debug(f"{req_id=}: remote processing timeout {timeout=}")
                return self._get_result(req_id)
            self.logger.debug(f"{req_id=}: in receiving")

            # receiving with progress timeout
            r_future = waiter.receiving_future
            self.logger.debug(f"{req_id=}: entering receiving wait {timeout=}")
            receiving_complete = self._future_wait(r_future, timeout)
            if not receiving_complete:
                self.logger.info(f"{req_id=}: receiving timeout {timeout=}")
                return self._get_result(req_id)
            self.logger.debug(f"{req_id=}: receiving complete")
            waiter.result = Message(r_future.headers, r_future.result())
            decode_payload(waiter.result, encoding_key=StreamHeaderKey.PAYLOAD_ENCODING)
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
            self.logger.warning(f"Receiving unknown {req_id=}, discarded: {e}")
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
