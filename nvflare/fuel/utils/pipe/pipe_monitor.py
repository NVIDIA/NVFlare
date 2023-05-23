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

import threading
import time

from nvflare.fuel.utils import fobs
from nvflare.fuel.utils.pipe.pipe import Pipe
from nvflare.fuel.utils.validation_utils import check_callable, check_object_type, check_positive_number


class Topic(object):

    ABORT = "_Abort_"
    END = "_End_"
    HEARTBEAT = "_Heartbeat_"
    PEER_GONE = "_PeerGone_"


def _send_to_pipe(pipe: Pipe, topic: str, data, timeout=None):
    return pipe.send(topic, fobs.dumps(data), timeout)


def _receive_from_pipe(pipe: Pipe):
    topic, data = pipe.receive()
    if data:
        data = fobs.loads(data)
    return topic, data


class PipeMonitor(object):
    """
    PipeMonitor monitors a pipe for messages from the peer. It reads the pipe periodically and puts received data
    in a message queue in the order the data is received.

    If the received data indicates a peer status change (END, ABORT, GONE), the data is added the message queue if the
    status_cb is not registered. If the status_cb is registered, the data is NOT added the message queue. Instead,
    the status_cb is called with the status data.

    The PipeMonitor should be used as follows:
    - The app creates a pipe and then creates the PipeMonitor object for the pipe;
    - The app starts the PipeMonitor. This step must be performed, or data in the pipe won't be read.
    - The app should call monitor.get_next() periodically to process the message in the queue. This method may return
    None if there is no message in the queue. The app also must handle the status change event from the peer if it
    does not set the status_cb. The status change event has the special topic value of Topic.END or Topic.ABORT.
    - Optionally, the app can set a status_cb and handle the peer's status change immediately.
    - Stop the monitor when the app is finished.

    NOTE: the monitor uses a heartbeat mechanism to detect that the peer may be disconnected (gone). It sends
    a heartbeat message to the peer based on configured interval. It also expects heartbeats from the peer. If peer's
    heartbeat is not received for configured time, it will be treated as disconnected, and a GONE status is generated
    for the app to handle.

    """

    def __init__(self, pipe: Pipe, read_interval=0.1, heartbeat_interval=5.0, heartbeat_timeout=30.0):
        """
        Constructor of the PipeMonitor.

        Args:
            pipe: the pipe to be monitored
            read_interval: how often to read from the pipe
            heartbeat_interval: how often to send a heartbeat to the peer
            heartbeat_timeout: how long to wait for a heartbeat from the peer before treating the peer as gone
        """
        check_positive_number("read_interval", read_interval)
        check_positive_number("heartbeat_interval", heartbeat_interval)
        check_positive_number("heartbeat_timeout", heartbeat_timeout)
        check_object_type("pipe", pipe, Pipe)

        if heartbeat_interval >= heartbeat_timeout:
            raise ValueError(f"heartbeat_interval {heartbeat_interval} must < heartbeat_timeout {heartbeat_timeout}")

        self.pipe = pipe
        self.read_interval = read_interval
        self.heartbeat_interval = heartbeat_interval
        self.heartbeat_timeout = heartbeat_timeout
        self.messages = []
        self.reader = threading.Thread(target=self._read)
        self.reader.daemon = True
        self.asked_to_stop = False
        self.lock = threading.Lock()
        self.status_cb = None
        self.cb_args = None
        self.cb_kwargs = None

    def set_status_cb(self, cb, *args, **kwargs):
        """Set CB for status handling. When the peer status is changed (ABORT, END, GONE), this CB is called.
        If status CB is not set, the monitor simply adds the status change event (topic) to the message queue.

        The status_cb must conform to this signature:

            cb(topic, data, *args, **kwargs)

        where the *args and *kwargs are ones passed to this call.
        The status_cb is called from the thread that reads from the pipe, hence it should be short-lived.
        Do not put heavy processing logic in the status_cb.

        Args:
            cb:
            *args:
            **kwargs:

        Returns: None

        """
        check_callable("cb", cb)
        self.status_cb = cb
        self.cb_args = args
        self.cb_kwargs = kwargs

    def start(self):
        """
        Start the PipeMonitor.

        Returns:

        """
        if not self.reader.is_alive():
            self.reader.start()

    def stop(self, close_pipe=True):
        """
        Stop the monitor and optionally close the monitored pipe.

        Args:
            close_pipe: whether to close the monitored pipe.

        Returns:

        """
        self.asked_to_stop = True
        pipe = self.pipe
        self.pipe = None
        if pipe and close_pipe:
            pipe.close()

    def send_to_peer(self, topic, data, timeout=None):
        """Send a message to peer.

        Args:
            topic: topic of the message
            data: data of the message
            timeout: how long to wait for the peer to read the data. If not specified, return False immediately.

        Returns: whether the peer has read the data.

        """
        if timeout is not None:
            check_positive_number("timeout", timeout)
        try:
            return _send_to_pipe(self.pipe, topic, data, timeout)
        except BrokenPipeError:
            self._add_message(Topic.PEER_GONE, "")

    def notify_end(self, data=""):
        """Notify the peer that the communication is ended normally.

        Args:
            data: optional data to be sent

        Returns: None

        """
        self.send_to_peer(Topic.END, data)

    def notify_abort(self, data=""):
        """Notify the peer that the communication is aborted.

        Args:
            data: optional data to be sent

        Returns: None

        """
        self.send_to_peer(Topic.ABORT, data)

    def _add_message(self, topic, data):
        if topic in [Topic.END, Topic.ABORT, Topic.PEER_GONE]:
            if self.status_cb is not None:
                self.status_cb(topic, data, *self.cb_args, **self.cb_kwargs)
                return
        with self.lock:
            self.messages.append((topic, data))

    def _read(self):
        try:
            self._try_read()
        except BrokenPipeError:
            self._add_message(Topic.PEER_GONE, "pipe closed")
        except:
            self._add_message(Topic.PEER_GONE, "error")

    def _try_read(self):
        last_heartbeat_received_time = time.time()
        last_heartbeat_sent_time = 0.0
        while not self.asked_to_stop:
            now = time.time()
            topic, data = _receive_from_pipe(self.pipe)
            if topic is not None:
                last_heartbeat_received_time = now
                if topic != Topic.HEARTBEAT:
                    self._add_message(topic, data)
                if topic in [Topic.END, Topic.ABORT]:
                    break
            else:
                # is peer gone?
                if now - last_heartbeat_received_time > self.heartbeat_timeout:
                    self._add_message(Topic.PEER_GONE, "")
                    break

            # send heartbeat to the peer
            if now - last_heartbeat_sent_time > self.heartbeat_interval:
                _send_to_pipe(self.pipe, Topic.HEARTBEAT, "")
                last_heartbeat_sent_time = now

            time.sleep(self.read_interval)
        self.reader = None

    def get_next(self):
        """Get next message from the message queue.

        Returns: a tuple of (topic, data) at the top of the message queue. If the queue is empty, returns (None, None)

        """
        with self.lock:
            if len(self.messages) > 0:
                return self.messages.pop(0)
            else:
                return None, None
