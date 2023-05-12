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


class Topic(object):

    ABORT = "_Abort_"
    END_RUN = "_EndRun_"
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
    def __init__(self, pipe: Pipe, read_interval=0.1, heartbeat_interval=5.0, heartbeat_timeout=30.0):
        self.pipe = pipe
        self.read_interval = read_interval
        self.heartbeat_interval = heartbeat_interval
        self.heartbeat_timeout = heartbeat_timeout
        self.messages = []
        self.reader = threading.Thread(target=self._read)
        self.reader.daemon = True
        self.asked_to_stop = False
        self.lock = threading.Lock()

    def start(self):
        if not self.reader.is_alive():
            self.reader.start()

    def stop(self):
        self.asked_to_stop = True
        pipe = self.pipe
        self.pipe = None
        if pipe:
            pipe.close()

    def send_to_peer(self, topic, data, timeout=None):
        return _send_to_pipe(self.pipe, topic, data, timeout)

    def _add_message(self, topic, data):
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
                if topic in [Topic.END_RUN, Topic.ABORT]:
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
        with self.lock:
            if len(self.messages) > 0:
                return self.messages.pop(0)
            else:
                return None, None
