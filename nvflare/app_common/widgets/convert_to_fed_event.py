from typing import List

from nvflare.apis.fl_constant import EventScope, FLContextKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.widgets.widget import Widget

FED_EVENT_PREFIX = "fed."


class ConvertToFedEvent(Widget):
    def __init__(self, events_to_convert: List[str], fed_event_prefix=FED_EVENT_PREFIX):
        Widget.__init__(self)
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
