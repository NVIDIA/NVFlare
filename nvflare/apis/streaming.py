# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
from abc import ABC, abstractmethod
from builtins import dict as StreamContext
from typing import Any, Dict, Tuple

from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable


class StreamContextKey:
    CHANNEL = "__channel__"
    TOPIC = "__topic__"
    RC = "__RC__"


class ObjectProducer(ABC):
    @abstractmethod
    def produce(
        self,
        stream_ctx: StreamContext,
        fl_ctx: FLContext,
    ) -> Tuple[Shareable, float]:
        """Called to produce the next Shareable object to be sent.
        If this method needs to take long time, it should check the abort_signal in the fl_ctx frequently.
        If aborted it should return immediately.
        You can get the abort_signal by calling fl_ctx.get_run_abort_signal().

        Args:
            stream_ctx: stream context data
            fl_ctx: The FLContext object

        Returns: a tuple of (Shareable object to be sent, timeout for sending this object)

        """
        pass

    @abstractmethod
    def process_replies(
        self,
        replies: Dict[str, Shareable],
        stream_ctx: StreamContext,
        fl_ctx: FLContext,
    ) -> Any:
        """Called to process replies from receivers of the last Shareable object sent to them.

        Args:
            replies: replies from receivers. It's dict of site_name => reply
            stream_ctx: stream context data
            fl_ctx: the FLContext object

        Returns: Any object or None

        If None is returned, the streaming will continue; otherwise the streaming stops and the returned object is
        returned as the final result of the streaming.

        """
        pass


class ObjectConsumer(ABC):
    @abstractmethod
    def consume(
        self,
        shareable: Shareable,
        stream_ctx: StreamContext,
        fl_ctx: FLContext,
    ) -> Tuple[bool, Shareable]:
        """Consume the received Shareable object in the stream.

        Args:
            stream_ctx: the stream context data.
            shareable: the Shareable object to be processed
            fl_ctx: the FLContext object

        Returns: a tuple of (whether to continue streaming, reply message)

        Note: the channel and topic here are defined by the app. They are not the regular message headers
        (CHANNEL and TOPIC) defined in MessageHeaderKey.

        """
        pass

    def finalize(
        self,
        stream_ctx: StreamContext,
        fl_ctx: FLContext,
    ):
        """Called to finalize the generator.

        Args:
            stream_ctx: stream context
            fl_ctx: the FLContext object

        Returns: None

        This method is guaranteed to be called at the end of streaming.

        """
        pass


class ConsumerFactory(ABC):
    @abstractmethod
    def get_consumer(
        self,
        stream_ctx: StreamContext,
        fl_ctx: FLContext,
    ) -> ObjectConsumer:
        """Called to get an ObjectConsumer to process a new stream on the receiving side.
        This is called only when the 1st streaming object is received for each stream.

        Args:
            stream_ctx: the context of the stream
            fl_ctx: FLContext object

        Returns: an ObjectConsumer

        """
        pass

    def return_consumer(
        self,
        consumer: ObjectConsumer,
        stream_ctx: StreamContext,
        fl_ctx: FLContext,
    ):
        """Return the consumer back to the factory after a stream is finished on the receiving side.

        Args:
            consumer: the consumer to be returned
            stream_ctx: context of the stream
            fl_ctx: FLContext object

        Returns: None

        """
        pass


def stream_done_cb_signature(stream_ctx: StreamContext, fl_ctx: FLContext, **kwargs):
    """This is the signature of stream_done_cb.

    Args:
        stream_ctx: context of the stream
        fl_ctx: FLContext object
        **kwargs: the kwargs specified when registering the stream_done_cb.

    Returns: None

    """
    pass
