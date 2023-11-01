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
import queue


class QueueClosed(Exception):
    pass


class QQ:
    def __init__(self):
        self.q = queue.Queue()
        self.closed = False
        self.logger = logging.getLogger(self.__class__.__name__)

    def close(self):
        self.closed = True

    def append(self, i):
        if self.closed:
            raise QueueClosed("queue stopped")
        self.q.put_nowait(i)

    def __iter__(self):
        return self

    def __next__(self):
        if self.closed:
            raise StopIteration()
        while True:
            try:
                return self.q.get(block=True, timeout=0.1)
            except queue.Empty:
                if self.closed:
                    self.logger.debug("Queue closed - stop iteration")
                    raise StopIteration()
            except Exception as e:
                self.logger.error(f"queue exception {type(e)}")
                raise e
