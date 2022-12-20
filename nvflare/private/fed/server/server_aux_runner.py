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

from nvflare.apis.client import Client
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.private.aux_runner import AuxRunner


class ServerAuxRunner(AuxRunner):
    def __init__(self):
        """This class is for auxiliary channel communication on server side.

        Note: The ServerEngine must create a new ServerAuxRunner object for each RUN, and make sure
              it is added as an event handler.
        """
        AuxRunner.__init__(self)

    def send_aux_request(
            self,
            targets: [],
            topic: str,
            request: Shareable,
            timeout: float,
            fl_ctx: FLContext,
            bulk_send
    ) -> dict:
        """Send request through auxiliary channel.

        Args:
            targets (list): list of client names that the request will be sent to
            topic (str): topic of the request
            request (Shareable): request
            timeout (float): how long to wait for result. 0 means fire-and-forget
            fl_ctx (FLContext): the FL context
            bulk_send: whether to send in bulk

        Returns:
            A dict of results
        """
        engine = fl_ctx.get_engine()
        if not targets:
            # get all clients
            targets = engine.get_clients()

        target_names = []
        for t in targets:
            if isinstance(t, str):
                name = t
            elif isinstance(t, Client):
                name = t.name
            else:
                raise ValueError(f"invalid target {t} in list: got {type(t)}")

            if name.startswith("server"):
                raise ValueError(f"invalid target '{t}': cannot send to server itself")

            if name not in target_names:
                target_names.append(t)

        if not target_names:
            return {}

        clients, invalid_names = engine.validate_clients(target_names)
        if invalid_names:
            raise ValueError(f"invalid target(s): {invalid_names}")

        return self.send_to_job_cell(
            targets=target_names,
            timeout=timeout,
            topic=topic,
            request=request,
            fl_ctx=fl_ctx,
            bulk_send=bulk_send
        )
