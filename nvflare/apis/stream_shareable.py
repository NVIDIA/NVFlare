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
from nvflare.apis.signal import Signal


class StreamShareableGenerator(ABC):
    @abstractmethod
    def get_next(
        self,
        channel: str,
        topic: str,
        fl_ctx: FLContext,
        abort_signal: Signal,
    ) -> Tuple[Shareable, float]:
        pass

    @abstractmethod
    def process_replies(self, replies: Dict[str, Shareable], fl_ctx: FLContext, abort_signal: Signal) -> Any:
        pass


class StreamShareableProcessor(ABC):
    @abstractmethod
    def process(
        self,
        channel: str,
        topic: str,
        shareable: Shareable,
        fl_ctx: FLContext,
        abort_signal: Signal,
    ) -> Tuple[bool, Shareable]:
        """Process received Stream Msg.

        Args:
            channel: app channel of the msg.
            topic: app topic of the msg.
            shareable: the request to be processed
            fl_ctx:
            abort_signal: signal to abort processing

        Returns: a tuple of (whether to continue streaming, reply message)

        Note: the channel and topic here are defined by the app. They are not the regular message headers
        (CHANNEL and TOPIC) defined in MessageHeaderKey.

        """
        pass


class StreamShareableProcessorFactory(ABC):
    @abstractmethod
    def get_processor(self, channel: str, topic: str, shareable: Shareable) -> StreamShareableProcessor:
        """Get a processor to process a shareable stream.

        Args:
            channel: app channel
            topic: app topic
            shareable: the received msg

        Returns: a StreamShareableProcessor

        """
        pass
