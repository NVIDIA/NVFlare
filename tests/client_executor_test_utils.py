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

from tests.timing_utils import WAIT_TIMEOUT


class ControlledCleanupDeadlines:
    """Expire caller waits only after the test observes its blocked backend boundary.

    Completion waits retain an independent real watchdog. Monitor waits and test
    readiness waits use their original events; only the product's caller budget
    is intercepted. No deadline depends on a worker being scheduled in 0.1s.
    """

    def __init__(self, monkeypatch, subject):
        self.timeout = 10.0
        self.cancel_pending = None
        self.observation_pending = None
        self.before_observation_result = None
        controller = self
        original_call = subject._CleanupCall
        original_state = subject._RunState

        class CleanupCall(original_call):
            def __init__(self, operation):
                super().__init__(operation)
                controller._watch(self.done, "cancel")

        class RunState(original_state):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                controller._watch(self.observation_done, "observation")

        monkeypatch.setattr(subject, "_CleanupCall", CleanupCall)
        monkeypatch.setattr(subject, "_RunState", RunState)

    def _watch(self, event, operation):
        original_wait = event.wait

        def wait(timeout=None):
            if timeout is None:
                result = original_wait(WAIT_TIMEOUT)
                assert result, f"unbounded {operation} wait exceeded the independent watchdog"
                return result
            if timeout != self.timeout:
                return original_wait(timeout)
            ready = getattr(self, operation + "_pending")
            if event.is_set():
                result = True
            elif ready is None:
                result = original_wait(WAIT_TIMEOUT)
                assert result, f"{operation} did not complete within the independent watchdog"
            else:
                assert ready.wait(WAIT_TIMEOUT), f"{operation} never reached its blocked boundary"
                result = event.is_set()
            if operation == "observation" and self.before_observation_result:
                self.before_observation_result()
            return result

        event.wait = wait


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
