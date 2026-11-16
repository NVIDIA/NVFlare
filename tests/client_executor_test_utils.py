# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Controlled process boundary for testing the real ProcessHandle lifecycle."""

import threading

from tests.timing_utils import join_thread


class ControlledProcessAdapter:
    def __init__(self, on_terminate, on_wait=None):
        self.on_terminate = on_terminate
        self.on_wait = on_wait
        self.exited = threading.Event()
        self.wait_entered = threading.Event()
        self.release_wait = threading.Event()
        self.release_wait.set()
        self.terminate_count = 0
        self.wait_count = 0
        self.lock = threading.Lock()

    def terminate(self):
        with self.lock:
            self.terminate_count += 1
            attempt = self.terminate_count
        self.on_terminate(self, attempt)

    def wait(self):
        with self.lock:
            self.wait_count += 1
            attempt = self.wait_count
        self.wait_entered.set()
        if self.on_wait:
            self.on_wait(self, attempt)
        self.exited.wait()
        self.release_wait.wait()

    def poll(self):
        return 9 if self.exited.is_set() else None


def join_cleanup_calls(*pairs):
    seen = set()
    for pair in pairs:
        for call in pair or ():
            if call is not None and id(call) not in seen:
                seen.add(id(call))
                if call.thread.ident is not None:
                    join_thread(call.thread)
