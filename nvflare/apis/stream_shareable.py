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
from typing import Any, Dict, Tuple

from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable


class StreamMetaKey:
    CHANNEL = "__channel__"
    TOPIC = "__topic__"
    RC = "__RC__"


class StreamMeta(dict):
    def get_channel(self):
        return self.get(StreamMetaKey.CHANNEL)

    def get_topic(self):
        return self.get(StreamMetaKey.TOPIC)

    def get_rc(self):
        return self.get(StreamMetaKey.RC)


class StreamShareableGenerator(ABC):
    @abstractmethod
    def get_next(
        self,
        stream_meta: StreamMeta,
        fl_ctx: FLContext,
    ) -> Tuple[Shareable, float]:
        """Called to generate next Shareable object to be sent.
        If this method needs to take long time, it should check the abort_signal in the fl_ctx frequently.
        If aborted it should return immediately.
        You can get the abort_signal by calling fl_ctx.get_run_abort_signal().

        Args:
            stream_meta: stream metadata
            fl_ctx: The FLContext object

        Returns: a tuple of (Shareable object to be sent, timeout for sending this object)

        """
        pass

    @abstractmethod
    def process_replies(
        self,
        replies: Dict[str, Shareable],
        stream_meta: StreamMeta,
        fl_ctx: FLContext,
    ) -> Any:
        """Called to process replies from receivers of the last Shareable object sent to them.

        Args:
            replies: replies from receivers. It's dict of site_name => reply
            stream_meta: stream metadata
            fl_ctx: the FLContext object

        Returns: Any object or None

        If None is returned, the streaming will continue; otherwise the streaming stops and the returned object is
        returned as the final result of the streaming.

        """
        pass


class StreamShareableProcessor(ABC):
    @abstractmethod
    def process(
        self,
        shareable: Shareable,
        stream_meta: StreamMeta,
        fl_ctx: FLContext,
    ) -> Tuple[bool, Shareable]:
        """Process received Shareable object in the stream.

        Args:
            stream_meta: the stream metadata.
            shareable: the Shareable object to be processed
            fl_ctx: the FLContext object

        Returns: a tuple of (whether to continue streaming, reply message)

        Note: the channel and topic here are defined by the app. They are not the regular message headers
        (CHANNEL and TOPIC) defined in MessageHeaderKey.

        """
        pass

    def finalize(
        self,
        stream_meta: StreamMeta,
        fl_ctx: FLContext,
    ):
        """Called to finalize the generator.

        Args:
            stream_meta: stream metadata
            fl_ctx: the FLContext object

        Returns: None

        This method is guaranteed to be called at the end of streaming.

        """
        pass


class StreamShareableProcessorFactory(ABC):
    @abstractmethod
    def get_processor(
        self,
        stream_meta: StreamMeta,
        fl_ctx: FLContext,
    ) -> StreamShareableProcessor:
        """Called to get a processor to process a new shareable stream on the receiving side.
        This is called only when the 1st streaming object is received for each stream.

        Args:
            stream_meta: the metadata of the stream
            fl_ctx: FLContext object

        Returns: a StreamShareableProcessor

        """
        pass

    def return_processor(
        self,
        processor: StreamShareableProcessor,
        stream_meta: StreamMeta,
        fl_ctx: FLContext,
    ):
        """Return the processor back to the factory after a stream is finished on the receiving side.

        Args:
            processor: the processor to return
            stream_meta: metadata of the stream
            fl_ctx: FLContext object

        Returns: None

        """
        pass


def stream_done_cb_signature(stream_meta: StreamMeta, fl_ctx: FLContext, **kwargs):
    """This is the signature of stream_done_cb.

    Args:
        stream_meta: metadata of the stream
        fl_ctx: FLContext object
        **kwargs: the kwargs specified when registering the stream_done_cb.

    Returns: None

    """
    pass
