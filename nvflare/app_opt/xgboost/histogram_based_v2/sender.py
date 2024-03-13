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

from nvflare.apis.fl_component import FLComponent
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal
from nvflare.fuel.f3.cellnet.fqcn import FQCN


class Sender(FLComponent, ABC):
    """An abstract interface to send request to XGB server"""

    @abstractmethod
    def send_to_server(self, engine, topic: str, req: Shareable, timeout: float, abort_signal: Signal):
        """Send a request to the server.

        Args:
            engine: Client engine
            topic: Topic for the request
            req: the request Shareable
            timeout: Timeout of the request in seconds
            abort_signal: used for checking whether the job is aborted.

        Returns: reply from the server
        """
        pass


class SimpleSender(Sender):
    def __init__(self):
        super().__init__()

    def send_to_server(self, engine, topic: str, req: Shareable, timeout: float, abort_signal: Signal):

        server_name = FQCN.ROOT_SERVER
        with engine.new_context() as fl_ctx:
            return engine.send_aux_request(
                targets=[server_name],
                topic=topic,
                request=req,
                timeout=timeout,
                fl_ctx=fl_ctx,
            )
