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

from queue import Empty, Full, Queue
from threading import Lock
from typing import Union

from nvflare.fuel.utils.constants import Mode
from nvflare.fuel.utils.pipe.pipe import Message, Pipe


class MemoryPipe(Pipe):
    PIPE_PAIRS = {}
    LOCK = Lock()

    def __init__(self, token: str, mode: Mode = Mode.ACTIVE):
        super().__init__(mode)
        self.token = token
        self.put_queue = None
        self.get_queue = None

    def open(self, name: str):
        with MemoryPipe.LOCK:
            if self.token in MemoryPipe.PIPE_PAIRS:
                x_queue, y_queue = MemoryPipe.PIPE_PAIRS[self.token]
            else:
                x_queue = Queue()
                y_queue = Queue()
                MemoryPipe.PIPE_PAIRS[self.token] = (x_queue, y_queue)
            if self.mode == Mode.ACTIVE:
                self.put_queue = x_queue
                self.get_queue = y_queue
            else:
                self.put_queue = y_queue
                self.get_queue = x_queue

    def clear(self):
        pass

    def close(self):
        with MemoryPipe.LOCK:
            if self.token in MemoryPipe.PIPE_PAIRS:
                MemoryPipe.PIPE_PAIRS.pop(self.token)

    def send(self, msg: Message, timeout=None) -> bool:
        try:
            self.put_queue.put(msg, block=False, timeout=timeout)
            return True
        except Full:
            return False

    def receive(self, timeout=None) -> Union[Message, None]:
        try:
            return self.get_queue.get(block=False, timeout=timeout)
        except Empty:
            return None

    def can_resend(self) -> bool:
        return False
