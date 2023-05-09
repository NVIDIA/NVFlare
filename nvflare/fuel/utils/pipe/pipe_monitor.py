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

    ABORT_TASK = "AbortTask"
    END_RUN = "EndRun"


def send_to_pipe(pipe: Pipe, topic: str, data, timeout=None):
    return pipe.send(topic, fobs.dumps(data), timeout)


def receive_from_pipe(pipe: Pipe):
    topic, data = pipe.receive()
    if data:
        data = fobs.loads(data)
    return topic, data


class PipeMonitor(object):
    def __init__(self, pipe: Pipe, task_status_cb, *args, **kwargs):
        self.pipe = pipe
        self.task_status_cb = task_status_cb
        self.cb_args = args
        self.cb_kwargs = kwargs
        self.messages = []
        self.reader = threading.Thread(target=self._read)
        self.asked_to_stop = False
        self.lock = threading.Lock()

    def start(self):
        if not self.reader.is_alive():
            self.reader.start()

    def stop(self):
        self.asked_to_stop = True
        reader = self.reader
        if reader and reader.is_alive():
            reader.join()

    def _read(self):
        while not self.asked_to_stop:
            topic, data = receive_from_pipe(self.pipe)
            if topic in [Topic.END_RUN, Topic.ABORT_TASK]:
                self.task_status_cb(topic, data, *self.cb_args, **self.cb_kwargs)
            elif topic is not None:
                with self.lock:
                    self.messages.append((topic, data))
            time.sleep(0.5)

        self.reader = None

    def get_next(self):
        with self.lock:
            if len(self.messages) > 0:
                return self.messages.pop(0)
            else:
                return None, None
