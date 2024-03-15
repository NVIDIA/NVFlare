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
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal
from nvflare.apis.utils.reliable_message import ReliableMessage
from nvflare.apis.utils.sender import Sender
from nvflare.fuel.f3.cellnet.fqcn import FQCN


class ReliableSender(Sender):
    def __init__(self, max_request_workers=20, query_interval=5, max_retries=5, max_tx_time=300.0):
        """Constructor

        Args:
            max_request_workers: Number of concurrent request worker threads
            query_interval: Retry/query interval
            max_retries: Number of retries
            max_tx_time: Max transmitting time
        """

        super().__init__()
        self.max_request_workers = max_request_workers
        self.query_interval = query_interval
        self.max_retries = max_retries
        self.max_tx_time = max_tx_time
        self.enabled = False

    def send_request(
        self, target: str, topic: str, req: Shareable, timeout: float, fl_ctx: FLContext, abort_signal: Signal
    ) -> Shareable:

        if not self.enabled:
            ReliableMessage.enable(
                fl_ctx,
                max_request_workers=self.max_request_workers,
                query_interval=self.query_interval,
                max_retries=self.max_retries,
                max_tx_time=self.max_tx_time,
            )
            self.enabled = True

        return ReliableMessage.send_request(FQCN.ROOT_SERVER, topic, req, timeout, abort_signal, fl_ctx)
