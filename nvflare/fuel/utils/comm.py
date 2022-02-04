# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
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


class Queue(object):
    def __init__(self, name):
        """Queue object with basic functions.

        Args:
            name: name of queue
        """
        self.name = name
        self.items = []
        self._update_lock = threading.Lock()
        self.next_seq_no = 1

    def append(self, item):
        with self._update_lock:
            self.items.append(item)
            seq_no = self.next_seq_no
            self.next_seq_no += 1
            return seq_no

    def len(self):
        with self._update_lock:
            return len(self.items)

    def get(self, default=None):
        with self._update_lock:
            if len(self.items) > 0:
                result = self.items[0]
            else:
                result = default
        return result

    def pop(self, default=None):
        with self._update_lock:
            if len(self.items) > 0:
                result = self.items.pop(0)
            else:
                result = default
        return result

    def fill(self, data, limit):
        with self._update_lock:
            count = limit - len(self.items)
            if count > 0:
                for _ in range(count):
                    self.items.append(data)

    def clear(self):
        with self._update_lock:
            self.items = []


class Channel(object):
    def __init__(self, src, dest):
        """Channel object.

        Args:
            src: source
            dest: destination
        """
        self.src = src
        self.dest = dest
        self.req = Queue("req")
        self.reply = Queue("reply")

    def send(self, src, data):
        if src == self.src:
            self.req.append(data)
        elif src == self.dest:
            self.reply.append(data)

    def receive(self, src, default=None):
        if src == self.src:
            return self.reply.pop(default=default)
        elif src == self.dest:
            return self.req.pop(default=default)
        else:
            return default

    def __str__(self):
        return "Channel({}<=>{})".format(self.src.name, self.dest.name)
