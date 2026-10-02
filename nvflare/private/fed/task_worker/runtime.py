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

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_constant import EventScope, FLContextKey
from nvflare.apis.fl_context import FLContext, FLContextManager
from nvflare.apis.signal import Signal
from nvflare.apis.workspace import Workspace
from nvflare.private.event import fire_event


class UnsupportedTaskRuntimeService(RuntimeError):
    """A compute component requested a service owned by its job runtime."""


class TaskRuntime:
    """Own task-local context and components without a client/server engine contract.

    ``FLContext.get_engine()`` exposes this compatibility adapter to existing
    compute components. It reuses the context manager and event dispatcher;
    federation services and job supervision stay with the calling job runtime.
    Client metadata and Client API bindings are supplied outside this class.
    """

    def __init__(self, workspace: Workspace, identity_name: str, job_id: str):
        self._workspace = workspace
        self._components = {}
        self._handlers = []
        self._event_observers = []
        self._fatal_error = None
        self.abort_signal = Signal()
        self._context_manager = FLContextManager(
            engine=self,
            identity_name=identity_name,
            job_id=job_id,
            private_stickers={FLContextKey.RUN_ABORT_SIGNAL: self.abort_signal},
        )

    def set_compute_graph(self, components, executor):
        """Install only the attempt's graph, never its job's handler graph."""
        self._components = dict(components)
        self._handlers = [component for component in components.values() if isinstance(component, FLComponent)]
        self._handlers.append(executor)

    def get_component(self, component_id: str):
        return self._components.get(component_id)

    def get_all_components(self) -> dict:
        return dict(self._components)

    def get_workspace(self) -> Workspace:
        return self._workspace

    def new_context(self) -> FLContext:
        return self._context_manager.new_context()

    def add_event_observer(self, observer):
        """Bind a role-specific durable event sink without importing a job engine."""
        self._event_observers.append(observer)

    def fire_event(self, event_type: str, fl_ctx: FLContext):
        captured = False
        for observer in self._event_observers:
            captured = observer(event_type, fl_ctx) or captured
        if fl_ctx.get_prop(FLContextKey.EVENT_SCOPE) == EventScope.FEDERATION and not captured:
            self._unsupported("federated events")
        if event_type == EventType.FATAL_SYSTEM_ERROR and self._fatal_error is None:
            self._fatal_error = fl_ctx.get_prop(FLContextKey.EVENT_DATA) or "fatal system error"
            self.abort_signal.trigger(True)
        fire_event(event=event_type, handlers=self._handlers, ctx=fl_ctx)

    def raise_if_failed(self):
        """Surface a panic at execution boundaries, after finalizers can run."""
        if self._fatal_error is not None:
            raise RuntimeError(f"task runtime received FATAL_SYSTEM_ERROR: {self._fatal_error}")

    @staticmethod
    def _unsupported(service: str):
        raise UnsupportedTaskRuntimeService(f"task worker runtime does not provide {service}")

    # Explicit compatibility guards keep existing compute components from
    # silently requesting network or job-lifetime services inside an attempt.
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
