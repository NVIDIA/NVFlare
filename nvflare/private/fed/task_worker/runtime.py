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

"""Private, role-neutral local services for one compute attempt."""

import threading
from contextlib import contextmanager

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContext, FLContextManager
from nvflare.apis.signal import Signal
from nvflare.apis.workspace import Workspace


class UnsupportedTaskRuntimeService(RuntimeError):
    """A compute component requested a service owned by its job runtime."""


class _TaskAbortSignal(Signal):
    """Serialize direct abort requests with the attempt's completion decision."""

    def __init__(self, runtime):
        super().__init__()
        self._runtime = runtime

    def trigger(self, value):
        with self._runtime._state_lock:
            self._runtime._require_open()
            self._runtime._fail_locked(RuntimeError("task runtime was aborted"))
            super().trigger(value)

    def reset(self, value=None):
        with self._runtime._state_lock:
            self._runtime._require_open()
            # Resetting the compatibility signal cannot undo an attempt failure.
            super().reset(value)


class TaskRuntime:
    """Own task-local context without a client/server engine contract.

    ``FLContext.get_engine()`` exposes this compatibility adapter to existing
    compute executors. It provides context, logging compatibility and abort;
    event dispatch and federation services stay with the calling job runtime.
    """

    def __init__(self, workspace: Workspace, identity_name: str, job_id: str, components=None):
        self._workspace = workspace
        # Only explicitly staged compute dependencies are available here.
        self._components = None if components is None else dict(components)
        self._state_lock = threading.Lock()
        self._completed = False
        self._active_calls = 0
        self._failure = None
        self.abort_signal = _TaskAbortSignal(self)
        self._context_manager = FLContextManager(
            engine=self,
            identity_name=identity_name,
            job_id=job_id,
            private_stickers={FLContextKey.RUN_ABORT_SIGNAL: self.abort_signal},
        )

    def get_workspace(self) -> Workspace:
        return self._workspace

    def new_context(self) -> FLContext:
        return self._context_manager.new_context()

    @contextmanager
    def operation(self):
        """Keep direct compute/analytics calls ordered with completion publication."""
        with self._state_lock:
            self._require_open()
            self._active_calls += 1
        try:
            yield
        except BaseException as error:
            self.fail(error)
            raise
        finally:
            with self._state_lock:
                self._active_calls -= 1

    def fail(self, error):
        with self._state_lock:
            self._require_open()
            self._fail_locked(error)

    def fire_event(self, event_type: str, fl_ctx: FLContext):
        """Compatibility for FLComponent panic/logging; never dispatch handlers."""
        if event_type == EventType.FATAL_SYSTEM_ERROR:
            message = str(fl_ctx.get_prop(FLContextKey.EVENT_DATA) or "fatal system error")
            self.fail(RuntimeError(f"task runtime received FATAL_SYSTEM_ERROR: {message}"))
        elif event_type not in (
            EventType.INFO_LOG_AVAILABLE,
            EventType.DEBUG_LOG_AVAILABLE,
            EventType.WARNING_LOG_AVAILABLE,
            EventType.ERROR_LOG_AVAILABLE,
            EventType.CRITICAL_LOG_AVAILABLE,
        ):
            error = UnsupportedTaskRuntimeService("task worker runtime does not provide event dispatch")
            self.fail(error)
            raise error
        # FLComponent logging already wrote the message before this notification.

    def _fail_locked(self, failure):
        if self._failure is None:
            self._failure = failure
        Signal.trigger(self.abort_signal, True)

    def _require_open(self):
        if self._completed:
            raise RuntimeError("task runtime has already completed")

    def raise_if_failed(self):
        """Surface a panic at execution boundaries, after finalizers can run."""
        with self._state_lock:
            self._raise_if_failed_locked()

    def _raise_if_failed_locked(self):
        if self._failure is not None:
            raise self._failure

    @contextmanager
    def completion_guard(self):
        """Order final publication against abort and direct failure reporting.

        Use only around the final filesystem publication, after serialization
        and verification. Fail closed if a direct operation is still running.
        Once publication wins, subsequent operations and abort requests are rejected.
        """
        with self._state_lock:
            self._require_open()
            self._raise_if_failed_locked()
            if self._active_calls:
                raise RuntimeError("task runtime operations are still active")
            yield
            self._completed = True

    @staticmethod
    def _unsupported(service: str):
        raise UnsupportedTaskRuntimeService(f"task worker runtime does not provide {service}")

    # Explicit compatibility guards keep existing compute components from
    # silently requesting network or job-lifetime services inside an attempt.
    def get_component(self, component_id=None):
        if self._components is None:
            self._unsupported("job component lookup")
        return self._components.get(component_id)

    def get_all_components(self):
        self._unsupported("job component lookup")

    def get_cell(self):
        self._unsupported("a federation Cell")

    def register_aux_message_handler(self, *args, **kwargs):
        self._unsupported("auxiliary message handlers")

    def send_aux_request(self, *args, **kwargs):
        self._unsupported("auxiliary federation requests")

    def multicast_aux_requests(self, *args, **kwargs):
        self._unsupported("multicast auxiliary federation requests")

    def fire_and_forget_aux_request(self, *args, **kwargs):
        self._unsupported("asynchronous auxiliary federation requests")

    def dispatch(self, *args, **kwargs):
        self._unsupported("auxiliary dispatch")

    def stream_objects(self, *args, **kwargs):
        self._unsupported("federation object streaming")

    def get_task_assignment(self, *args, **kwargs):
        self._unsupported("task acquisition")

    def send_task_result(self, *args, **kwargs):
        self._unsupported("remote result publication")

    def validate_targets(self, *args, **kwargs):
        self._unsupported("federation target validation")

    def get_widget(self, *args, **kwargs):
        self._unsupported("job widgets")

    def build_component(self, *args, **kwargs):
        self._unsupported("runtime component construction")

    def abort_app(self, *args, **kwargs):
        self._unsupported("job application control")
