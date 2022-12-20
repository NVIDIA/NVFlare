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

import threading

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.private.aux_runner import AuxRunner

from .client_engine_executor_spec import ClientEngineExecutorSpec


class ClientAuxRunner(AuxRunner):
    """ClientAuxRunner to send the aux messages to the server.

    Note: The ClientEngine must create a new ClientAuxRunner object for each RUN, and make sure
    it is added as an event handler!

    """

    def __init__(self):
        """To init the ClientAuxRunner."""
        AuxRunner.__init__(self)

    def send_aux_request(
            self,
            targets: list,
            topic: str,
            request: Shareable,
            timeout: float,
            fl_ctx: FLContext,
            bulk_send: bool
    ) -> dict:
        if not targets:
            targets = ["server"]
        return self.send_to_job_cell(
            targets=targets,
            topic=topic,
            request=request,
            timeout=timeout,
            fl_ctx=fl_ctx,
            bulk_send=bulk_send
        )
