from abc import ABC, abstractmethod
from typing import List, Optional

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
    def __init__(self, event: str = EVENT):
        """Sends analytic data.

        Args:
            event (str): The event that corresponding to this send.
        """
        super().__init__()
        self.event = event

    def write_scalar(self, fl_ctx: FLContext, tag: str, scalar: float, **kwargs):
        """Writes a scalar.

        Args:
            fl_ctx (FLContext): fl context info.
            tag (str): the tag associated with this value.
            scalar (float) : a scalar to write.
        """
        self.write(fl_ctx, tag, scalar, data_type=AnalyticsDataType.SCALAR, kwargs=kwargs)

    def write_scalars(self, fl_ctx: FLContext, tag: str, tag_scalar_dict: dict, **kwargs):
        """Writes scalars.

        Args:
            fl_ctx (FLContext): fl context info.
            tag (str): the tag associated with this dict.
            tag_scalar_dict (dict): A dictionary that contains tag and scalars to write.
        """
        self.write(fl_ctx, tag, tag_scalar_dict, data_type=AnalyticsDataType.SCALARS, kwargs=kwargs)

    def write_image(self, fl_ctx: FLContext, tag: str, image, **kwargs):
        """Writes an image.

        Args:
            fl_ctx (FLContext): fl context info.
            tag (str): the tag associated with this value.
            image: the image to write.
        """
        self.write(fl_ctx, tag, image, data_type=AnalyticsDataType.IMAGE, kwargs=kwargs)

    def write_text(self, fl_ctx: FLContext, tag: str, text: str, **kwargs):
        """Writes text.

        Args:
            fl_ctx (FLContext): fl context info.
            tag (str): the tag associated with this value.
            text: the text to write.
        """
        self.write(fl_ctx, tag, text, data_type=AnalyticsDataType.TEXT, kwargs=kwargs)

    def write(self, fl_ctx: FLContext, tag: str, value, data_type: AnalyticsDataType, kwargs=None):
        """Writes the analytic data.

        Args:
            fl_ctx (FLContext): fl context info.
            tag (str): the tag associated with this value.
            value: the analytic data.
            data_type (AnalyticsDataType): analytic data type.
            kwargs (dict): additional arguments to be passed into the receiver side's function.
        """
        if not isinstance(fl_ctx, FLContext):
            raise TypeError("expect fl_ctx to be FLContext, but got {}".format(type(fl_ctx)))
        data = AnalyticsData(tag=tag, value=value, data_type=data_type, kwargs=kwargs)
        dxo = data.to_dxo()
        fl_ctx.set_prop(key=FLContextKey.EVENT_DATA, value=dxo.to_shareable(), private=True, sticky=False)
        self.fire_event(event_type=self.event, fl_ctx=fl_ctx)


class AnalyticsReceiver(Widget, ABC):
    def __init__(self, events: Optional[List[str]] = None):
        """Receives analytic data.

        Args:
            events (optional, List[str]): A list of event that this receiver will handled.
        """
        FLComponent.__init__(self)
        if events is None:
            events = [EVENT, f"fed.{EVENT}"]
        self.events = events

    @abstractmethod
    def initialize(self, fl_ctx: FLContext):
        """Initializes the receiver.

        Args:
            fl_ctx (FLContext): fl context.
        """
        pass

    @abstractmethod
    def save(self, fl_ctx: FLContext, shareable: Shareable, record_origin: str):
        """Saves the received data.

        Args:
            fl_ctx (FLContext): fl context.
            shareable (Shareable): the received message.
            record_origin (str): the sender of this message / record.
        """
        pass

    @abstractmethod
    def finalize(self, fl_ctx: FLContext):
        """Finalizes the receiver.

        Args:
            fl_ctx (FLContext): fl context.

        """
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
