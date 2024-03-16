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
from typing import Optional

from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal
from nvflare.fuel.f3.cellnet.fqcn import FQCN


class Sender(FLComponent, ABC):
    """An abstract class to send request"""

    @abstractmethod
    def send_request(
        self, target: str, topic: str, req: Shareable, timeout: float, fl_ctx: FLContext, abort_signal: Signal
    ) -> Optional[Shareable]:
        """Send a request to target. This is an abstract method. Derived class must implement this method

         Args:
             target: The destination
             topic: Topic for the request
             req: the request Shareable
             timeout: Timeout of the request in seconds
             fl_ctx: FLContext for the transaction
             abort_signal: used for checking whether the job is aborted.

        Returns:
            The reply in Shareable

        """
        pass

    def send_to_server(
        self, topic: str, req: Shareable, timeout: float, fl_ctx: FLContext, abort_signal: Signal
    ) -> Optional[Shareable]:
        """Send an XGB request to the server.

        Args:
            topic: The topic of the request
            req: the request Shareable
            timeout: The timeout value for the request
            fl_ctx: The FLContext for the request
            abort_signal: used for checking whether the job is aborted.

        Returns: reply from the server
        """

        return self.send_request(FQCN.ROOT_SERVER, topic, req, timeout, fl_ctx, abort_signal)


class SimpleSender(Sender):
    def __init__(self):
        super().__init__()

    def send_request(
        self, target: str, topic: str, req: Shareable, timeout: float, fl_ctx: FLContext, abort_signal: Signal
    ) -> Optional[Shareable]:

        engine = fl_ctx.get_engine()
        reply = engine.send_aux_request(
            targets=[target],
            topic=topic,
            request=req,
            timeout=timeout,
            fl_ctx=fl_ctx,
        )

        # send_aux_request returns multiple replies in a dict
        if reply:
            return reply.get(target)
        else:
            return None
