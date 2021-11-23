from abc import ABC, abstractmethod

from nvflare.apis.analytix import Data as AnalyticsData
from nvflare.apis.analytix import DataType as AnalyticsDataType
from nvflare.apis.event_type import EventType
from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_constant import FedEventHeader, FLContextKey, ReservedKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.widgets.widget import Widget

EVENT = "log_analytics"


class AnalyticsSender(Widget):
    def __init__(self, event=EVENT):
        super().__init__()
        self.event = event

    def write_scalar(self, fl_ctx, tag, scalar, **kwargs):
        self.write(fl_ctx, tag, scalar, data_type=AnalyticsDataType.SCALAR, kwargs=kwargs)

    def write_scalars(self, fl_ctx, tag, tag_scalar_dict, **kwargs):
        self.write(fl_ctx, tag, tag_scalar_dict, data_type=AnalyticsDataType.SCALARS, kwargs=kwargs)

    def write_image(self, fl_ctx, tag, image, **kwargs):
        self.write(fl_ctx, tag, image, data_type=AnalyticsDataType.IMAGE, kwargs=kwargs)

    def write_text(self, fl_ctx, tag, text, **kwargs):
        self.write(fl_ctx, tag, text, data_type=AnalyticsDataType.TEXT, kwargs=kwargs)

    def write(self, fl_ctx: FLContext, tag, value, data_type: AnalyticsDataType, kwargs=None):
        if not isinstance(fl_ctx, FLContext):
            raise TypeError("expect fl_ctx to be FLContext, but got {}".format(type(fl_ctx)))
        data = AnalyticsData(tag=tag, value=value, data_type=data_type, kwargs=kwargs)
        dxo = data.to_dxo()
        fl_ctx.set_prop(key=FLContextKey.EVENT_DATA, value=dxo.to_shareable(), private=True, sticky=False)
        self.fire_event(event_type=self.event, fl_ctx=fl_ctx)


class AnalyticsReceiver(Widget, ABC):
    def __init__(self, events=None):
        FLComponent.__init__(self)
        if events is None:
            events = [EVENT, f"fed.{EVENT}"]
        self.events = events

    @abstractmethod
    def initialize(self, fl_ctx: FLContext):
        pass

    @abstractmethod
    def save(self, fl_ctx: FLContext, shareable: Shareable, record_origin: str):
        pass

    @abstractmethod
    def finalize(self, fl_ctx: FLContext):
        pass

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.START_RUN:
            self.initialize(fl_ctx)
        elif event_type in self.events:
            data = fl_ctx.get_prop(FLContextKey.EVENT_DATA, None)
            if data is None:
                self.log_error(fl_ctx, "Missing event data.")
                return
            if not isinstance(data, Shareable):
                self.log_error(fl_ctx, f"Expect shareable but get {type(data)}")
                return
            record_origin = fl_ctx.get_identity_name()

            # if fed event use peer name to save
            if data.get_header(FedEventHeader.ORIGIN) is not None:
                peer_name = data.get_peer_prop(ReservedKey.IDENTITY_NAME, None)
                record_origin = peer_name

            if record_origin is None:
                self.log_error(fl_ctx, "record_origin can't be None.")
                return
            self.save(shareable=data, fl_ctx=fl_ctx, record_origin=record_origin)
        elif event_type == EventType.END_RUN:
            self.finalize(fl_ctx)
