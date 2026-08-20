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

import threading


class CallbackAdmission:
    """Close callback admission and wait until every admitted callback returns."""

    def __init__(self):
        self._condition = threading.Condition()
        self._closed = False
        self._active = 0

    def enter(self) -> bool:
        with self._condition:
            if self._closed:
                return False
            self._active += 1
            return True

    def leave(self) -> None:
        with self._condition:
            if self._active <= 0:
                raise RuntimeError("callback admission leave without matching enter")
            self._active -= 1
            if self._active == 0:
                self._condition.notify_all()

    def close(self) -> None:
        with self._condition:
            self._closed = True

    def wait(self, timeout: float) -> bool:
        """Wait up to ``timeout`` seconds for admitted callbacks to return."""
        if timeout < 0:
            raise ValueError("timeout must not be negative")
        with self._condition:
            return self._condition.wait_for(lambda: self._active == 0, timeout=timeout)
