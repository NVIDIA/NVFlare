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

from typing import List

from nvflare.apis.fl_constant import EventScope, FLContextKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.widgets.widget import Widget

FED_EVENT_PREFIX = "fed."


class ConvertToFedEvent(Widget):
    def __init__(self, events_to_convert: List[str], fed_event_prefix=FED_EVENT_PREFIX):
        """Converts local event to federated events.

        Args:
            events_to_convert (List[str]): A list of event names to be converted.
            fed_event_prefix (str): The prefix that will be added to the converted event's name.
        """
        super().__init__()
        self.events_to_convert = events_to_convert
        self.fed_event_prefix = fed_event_prefix

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type in self.events_to_convert:
            event_scope = fl_ctx.get_prop(key=FLContextKey.EVENT_SCOPE, default=EventScope.LOCAL)
            if event_scope == EventScope.FEDERATION:
                # already a fed event
                return
            data = fl_ctx.get_prop(FLContextKey.EVENT_DATA, None)
            if data is None:
                self.log_error(fl_ctx, "Missing event data.")
                return
            if not isinstance(data, Shareable):
                self.log_error(fl_ctx, f"Expect data to be shareable but got {type(data)}")
                return
            self.fire_fed_event(self.fed_event_prefix + event_type, data, fl_ctx)
