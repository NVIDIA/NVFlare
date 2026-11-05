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

from threading import Event

from nvflare.apis.fl_context import FLContext
from nvflare.apis.impl.controller import Controller
from nvflare.apis.signal import Signal
from nvflare.widgets.info_collector import InfoCollector


class AbortTestController(Controller):
    """Keep the authorization test job active until explicitly released or aborted."""

    RELEASE_TOPIC = "release_abort_test"

    def __init__(self):
        super().__init__()
        self._waiting = Event()
        self._released = Event()

    def start_controller(self, fl_ctx: FLContext):
        self._engine.register_app_command(self.RELEASE_TOPIC, self._release)

    def _release(self, topic, data, fl_ctx):
        self._released.set()
        return {"released": True}

    def control_flow(self, abort_signal: Signal, fl_ctx: FLContext):
        self._waiting.set()
        # The interval only bounds abort responsiveness. Elapsed time never opens the gate.
        while not abort_signal.triggered and not self._released.wait(0.1):
            pass

    def stop_controller(self, fl_ctx: FLContext):
        pass

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == InfoCollector.EVENT_TYPE_GET_STATS:
            collector = fl_ctx.get_prop(InfoCollector.CTX_KEY_STATS_COLLECTOR)
            if collector:
                phase = "initializing"
                if self._released.is_set():
                    phase = "released"
                elif self._waiting.is_set():
                    phase = "waiting_for_release"
                collector.add_info(self.name, {"phase": phase})
