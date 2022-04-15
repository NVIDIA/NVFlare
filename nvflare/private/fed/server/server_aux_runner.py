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
from nvflare.apis.shareable import ReservedHeaderKey, Shareable
from nvflare.private.aux_runner import AuxRunner


class ServerAuxRunner(AuxRunner):
    def __init__(self):
        """This class is for auxiliary channel communication on server side.

        Note: The ServerEngine must create a new ServerAuxRunner object for each RUN, and make sure
              it is added as an event handler.
        """
        AuxRunner.__init__(self)

    def send_aux_request(self, targets: [], topic: str, request: Shareable, timeout: float, fl_ctx: FLContext) -> dict:
        """Send request through auxiliary channel.

        Args:
            targets (list): list of client names that the request will be sent to
            topic (str): topic of the request
            request (Shareable): request
            timeout (float): how long to wait for result. 0 means fire-and-forget
            fl_ctx (FLContext): the FL context

        Returns:
            A dict of results
        """
        if not isinstance(request, Shareable):
            raise ValueError("invalid request type: expect Shareable but got {}".format(type(request)))

        if not targets:
            raise ValueError("targets must be specified")

        if targets is not None and not isinstance(targets, list):
            raise TypeError("targets must be a list of Client or str, but got {}".format(type(targets)))

        if not isinstance(topic, str):
            raise TypeError("invalid topic: expects str but got {}".format(type(topic)))

        if not topic:
            raise ValueError("invalid topic: must not be empty")

        if topic == self.TOPIC_BULK:
            raise ValueError('topic value "{}" is reserved'.format(topic))

        if not isinstance(timeout, float):
            raise TypeError("invalid timeout: expects float but got {}".format(type(timeout)))

        if timeout < 0:
            raise ValueError("invalid timeout value {}: must >= 0.0".format(timeout))

        if not isinstance(fl_ctx, FLContext):
            raise TypeError("invalid fl_ctx: expects FLContext but got {}".format(type(fl_ctx)))

        request.set_peer_props(fl_ctx.get_all_public_props())
        request.set_header(ReservedHeaderKey.TOPIC, topic)

        engine = fl_ctx.get_engine()
        # assert isinstance(engine, ServerEngineInternalSpec)

        target_names = []
        for t in targets:
            if isinstance(t, str):
                name = t
            elif isinstance(t, Client):
                name = t.name
            else:
                raise ValueError("invalid target in list: got {}".format(type(t)))

            if name not in target_names:
                target_names.append(t)

        clients, invalid_names = engine.validate_clients(target_names)
        if invalid_names:
            raise ValueError("invalid target(s): {}".format(invalid_names))
        valid_tokens = []
        for c in clients:
            if c.token not in valid_tokens:
                valid_tokens.append(c.token)

        replies = engine.parent_aux_send(
            targets=valid_tokens, topic=topic, request=request, timeout=timeout, fl_ctx=fl_ctx
        )

        return replies
