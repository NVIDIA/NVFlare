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

import logging
import threading
import time
from collections import deque
from typing import Optional

from nvflare.apis.signal import Signal
from nvflare.fuel.utils.pipe.pipe import Message, Pipe, Topic
from nvflare.fuel.utils.validation_utils import (
    check_callable,
    check_non_negative_number,
    check_object_type,
    check_positive_number,
)
from nvflare.security.logging import secure_format_exception


class PipeHandler(object):
    """
    PipeHandler monitors a pipe for messages from the peer. It reads the pipe periodically and puts received data
    in a message queue in the order the data is received.

    If the received data indicates a peer status change (END, ABORT, GONE), the data is added the message queue if the
    status_cb is not registered. If the status_cb is registered, the data is NOT added the message queue. Instead,
    the status_cb is called with the status data.

    The PipeHandler should be used as follows:
      - The app creates a pipe and then creates the PipeHandler object for the pipe;
      - The app starts the PipeHandler. This step must be performed, or data in the pipe won't be read.
      - The app should call handler.get_next() periodically to process the message in the queue. This method may return
        None if there is no message in the queue. The app also must handle the status change event from the peer if it
        does not set the status_cb. The status change event has the special topic value of Topic.END or Topic.ABORT.
      - Optionally, the app can set a status_cb and handle the peer's status change immediately.
      - Stop the handler when the app is finished.

    NOTE: the handler uses a heartbeat mechanism to detect that the peer may be disconnected (gone). It sends
    a heartbeat message to the peer based on configured interval. It also expects heartbeats from the peer. If peer's
    heartbeat is not received for configured time, it will be treated as disconnected, and a GONE status is generated
    for the app to handle.

    """

    def __init__(
        self,
        pipe: Pipe,
        read_interval=0.1,
        heartbeat_interval=5.0,
        heartbeat_timeout=30.0,
        resend_interval=2.0,
        max_resends=5,
        default_request_timeout=5.0,
    ):
        """Constructor of the PipeHandler.

        Args:
            pipe (Pipe): the pipe to be monitored.
            read_interval (float): how often to read from the pipe.
            heartbeat_interval (float): how often to send a heartbeat to the peer.
            heartbeat_timeout (float): how long to wait for a heartbeat from the peer before treating the peer as gone,
                0 means DO NOT check for heartbeat.
            resend_interval (float): how often to resend a message if failing to send. None means no resend.
                Note that if the pipe does not support resending, then no resend.
            max_resends (int, optional): max number of resends. None means no limit.
            default_request_timeout (float): default timeout for request if timeout not specified.
        """
        check_positive_number("read_interval", read_interval)
        check_positive_number("heartbeat_interval", heartbeat_interval)
        check_non_negative_number("heartbeat_timeout", heartbeat_timeout)
        check_object_type("pipe", pipe, Pipe)

        if 0 < heartbeat_timeout <= heartbeat_interval:
            raise ValueError(f"heartbeat_interval {heartbeat_interval} must < heartbeat_timeout {heartbeat_timeout}")

        self.logger = logging.getLogger(self.__class__.__name__)
        self.pipe = pipe
        self.read_interval = read_interval
        self.heartbeat_interval = heartbeat_interval
        self.heartbeat_timeout = heartbeat_timeout
        self.default_request_timeout = default_request_timeout
        self.resend_interval = resend_interval
        self.max_resends = max_resends
        self.messages = deque([])
        self.reader = threading.Thread(target=self._read)
        self.reader.daemon = True
        self.asked_to_stop = False
        self.lock = threading.Lock()
        self.status_cb = None
        self.cb_args = None
        self.cb_kwargs = None
        self.msg_cb = None
        self.msg_cb_args = None
        self.msg_cb_kwargs = None
        self.peer_is_up_or_dead = threading.Event()
        self._pause = False
        self._last_heartbeat_received_time = None
        self._check_interval = 0.01
        self.heartbeat_sender = threading.Thread(target=self._heartbeat)
        self.heartbeat_sender.daemon = True

    def set_status_cb(self, cb, *args, **kwargs):
        """Set CB for status handling. When the peer status is changed (ABORT, END, GONE), this CB is called.
        If status CB is not set, the handler simply adds the status change event (topic) to the message queue.

        The status_cb must conform to this signature:

            cb(msg, *args, **kwargs)

        where the *args and *kwargs are ones passed to this call.
        The status_cb is called from the thread that reads from the pipe, hence it should be short-lived.
        Do not put heavy processing logic in the status_cb.

        Args:
            cb: the callback func
            *args: the args to be passed to the cb
            **kwargs: the kwargs to be passed to the cb

        Returns: None

        """
        check_callable("cb", cb)
        self.status_cb = cb
        self.cb_args = args
        self.cb_kwargs = kwargs

    def set_message_cb(self, cb, *args, **kwargs):
        """Set CB for message handling. When a regular message is received, this CB is called.
        If the msg CB is not set, the handler simply adds the received msg to the message queue.
        If the msg CB is set, the received msg will NOT be added to the message queue.

        The CB must conform to this signature:

            cb(msg, *args, **kwargs)

        where the *args and *kwargs are ones passed to this call.

        The CB is called from the thread that reads from the pipe, hence it should be short-lived.
        Do not put heavy processing logic in the CB.

        Args:
            cb: the callback func
            *args: the args to be passed to the cb
            **kwargs: the kwargs to be passed to the cb

        Returns: None

        """
        check_callable("cb", cb)
        self.msg_cb = cb
        self.msg_cb_args = args
        self.msg_cb_kwargs = kwargs

    def _send_to_pipe(self, msg: Message, timeout=None, abort_signal: Signal = None):
        pipe = self.pipe
        if not pipe:
            self.logger.error("cannot send message to pipe since it's already closed")
            return False

        if not timeout or not pipe.can_resend() or not self.resend_interval:
            if not timeout:
                timeout = self.default_request_timeout
            return pipe.send(msg, timeout)

        num_sends = 0
        while not self.asked_to_stop:
            sent = pipe.send(msg, timeout)
            num_sends += 1
            if sent:
                return sent

            if self.max_resends is not None and num_sends > self.max_resends:
                self.logger.error(f"abort sending after {num_sends} tries")
                return False

            if self.asked_to_stop:
                return False

            if abort_signal and abort_signal.triggered:
                return False

            # wait for resend_interval before resend, but return if asked_to_stop is set during the wait
            self.logger.info(f"will resend '{msg.topic}' in {self.resend_interval} secs")
            start_wait = time.time()
            while True:
                if self.asked_to_stop:
                    return False

                if abort_signal and abort_signal.triggered:
                    return False

                if time.time() - start_wait > self.resend_interval:
                    break
                time.sleep(0.1)
        return False

    def start(self):
        """Starts the PipeHandler.
        Note: before calling this method, the pipe managed by this PipeHandler must have been opened.
        """
        if self.reader and not self.reader.is_alive():
            self.reader.start()

        if self.heartbeat_sender and not self.heartbeat_sender.is_alive():
            self.heartbeat_sender.start()

    def stop(self, close_pipe=True):
        """Stops the handler and optionally close the monitored pipe.

        Args:
            close_pipe: whether to close the monitored pipe.
        """
        self.asked_to_stop = True
        self.peer_is_up_or_dead.clear()
        pipe = self.pipe
        self.pipe = None
        if pipe and close_pipe:
            pipe.close()

    @staticmethod
    def _make_event_message(topic: str, data):
        return Message.new_request(topic, data)

    def send_to_peer(self, msg: Message, timeout=None, abort_signal: Signal = None) -> bool:
        """Sends a message to peer.

        Args:
            msg: message to be sent
            timeout: how long to wait for the peer to read the data.
                If not specified, will use ``self.default_request_timeout``.
            abort_signal:

        Returns:
            Whether the peer has read the data.
        """
        if timeout is not None:
            check_positive_number("timeout", timeout)
        try:
            return self._send_to_pipe(msg, timeout, abort_signal)
        except BrokenPipeError:
            self._add_message(self._make_event_message(Topic.PEER_GONE, "send failed"))
            return False

    def notify_end(self, data):
        """Notifies the peer that the communication is ended normally."""
        p = self.pipe
        if p:
            try:
                # fire and forget
                p.send(self._make_event_message(Topic.END, data), 0.1)
            except Exception as ex:
                self.logger.debug(f"exception notify_end: {secure_format_exception(ex)}")

    def notify_abort(self, data):
        """Notifies the peer that the communication is aborted."""
        p = self.pipe
        if p:
            try:
                # fire and forget
                p.send(self._make_event_message(Topic.ABORT, data), 0.1)
            except Exception as ex:
                self.logger.debug(f"exception notify_abort: {secure_format_exception(ex)}")

    def _add_message(self, msg: Message):
        if msg.topic in [Topic.END, Topic.ABORT, Topic.PEER_GONE]:
            if self.status_cb is not None:
                self.status_cb(msg, *self.cb_args, **self.cb_kwargs)
                return
        else:
            if self.msg_cb is not None:
                self.msg_cb(msg, *self.msg_cb_args, **self.msg_cb_kwargs)
                return

        with self.lock:
            self.messages.append(msg)

    def _read(self):
        try:
            self._try_read()
        except Exception as e:
            self.logger.error(f"read error: {secure_format_exception(e)}")
            self._add_message(self._make_event_message(Topic.PEER_GONE, f"read error: {secure_format_exception(e)}"))

    def _try_read(self):
        self._last_heartbeat_received_time = time.time()
        while not self.asked_to_stop:
            time.sleep(self.read_interval)
            if self._pause:
                continue

            # we assign self.pipe to p and access pipe methods through p
            # this is because self.pipe could be set to None at any moment (e.g. the abort process could
            # stop the pipe handler at any time).
            p = self.pipe
            if not p:
                # the pipe handler is most likely stopped, but we leave it for the while loop to decide
                continue

            msg = p.receive()
            now = time.time()

            if msg:
                self._last_heartbeat_received_time = now
                # if receive any messages even if Topic is END or ABORT or PEER_GONE
                #    we still set peer_is_up_or_dead, as we no longer need to wait
                self.peer_is_up_or_dead.set()
                if msg.topic != Topic.HEARTBEAT and not self.asked_to_stop:
                    self._add_message(msg)
                if msg.topic in [Topic.END, Topic.ABORT]:
                    break
            else:
                # is peer gone?
                # ask the pipe for the last known active time of the peer
                last_peer_active_time = p.get_last_peer_active_time()
                if last_peer_active_time > self._last_heartbeat_received_time:
                    self._last_heartbeat_received_time = last_peer_active_time

                if (
                    self.heartbeat_timeout
                    and now - self._last_heartbeat_received_time > self.heartbeat_timeout
                    and not self.asked_to_stop
                ):
                    self._add_message(
                        self._make_event_message(
                            Topic.PEER_GONE, f"missing heartbeat after {self.heartbeat_timeout} secs"
                        )
                    )
                    break

        self.reader = None

    def _heartbeat(self):
        last_heartbeat_sent_time = 0.0
        while not self.asked_to_stop:
            if self._pause:
                time.sleep(self._check_interval)
                continue
            now = time.time()

            # send heartbeat to the peer
            if now - last_heartbeat_sent_time > self.heartbeat_interval:
                self.send_to_peer(self._make_event_message(Topic.HEARTBEAT, ""))
                last_heartbeat_sent_time = now

            time.sleep(self._check_interval)
        self.heartbeat_sender = None

    def get_next(self) -> Optional[Message]:
        """Gets the next message from the message queue.

        Returns:
            A Message at the top of the message queue.
            If the queue is empty, returns None.
        """
        if self.asked_to_stop:
            return None

        with self.lock:
            if self.messages:
                return self.messages.popleft()
            else:
                return None

    def pause(self):
        """Stops heartbeat checking and sending."""
        self._pause = True

    def resume(self):
        """Resumes heartbeat checking and sending."""
        if self._pause:
            self._pause = False
            self._last_heartbeat_received_time = time.time()
