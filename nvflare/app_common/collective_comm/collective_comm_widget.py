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


from nvflare.apis.collective_comm_constants import (
    CollectiveCommEvent,
    CollectiveCommHandleError,
    CollectiveCommRequestTopic,
    CollectiveCommShareableHeader,
)
from nvflare.apis.event_type import EventType
from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.app_common.collective_comm.collective_functor import AllGatherFunctor, AllReduceFunctor, BroadcastFunctor


class CollectiveCommWidget(FLComponent):
    def __init__(self):
        super().__init__()
        self._world_size = 0
        self._sequence_number = None
        self._buffer = None
        self._function = {
            CollectiveCommRequestTopic.BROADCAST: BroadcastFunctor(),
            CollectiveCommRequestTopic.ALL_REDUCE: AllReduceFunctor(),
            CollectiveCommRequestTopic.ALL_GATHER: AllGatherFunctor(),
        }

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        engine = fl_ctx.get_engine()
        if event_type == EventType.START_RUN:
            for topic in self._function:
                engine.register_aux_message_handler(topic=topic, message_handle_func=self._handle_all_requests)

    def _handle_all_requests(self, topic: str, request: Shareable, fl_ctx: FLContext) -> Shareable:
        timeout = request.get_header(CollectiveCommShareableHeader.TIMEOUT)
        if timeout:
            self.fire_event(CollectiveCommEvent.FAILED, fl_ctx)
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)
        all_requests = request.get_header(CollectiveCommShareableHeader.ALL_REQUESTS)
        sequence_number = all_requests[0].get_header(CollectiveCommShareableHeader.SEQUENCE_NUMBER)
        self._world_size = all_requests[0].get_header(CollectiveCommShareableHeader.WORLD_SIZE)
        try:
            for r in all_requests:
                self._handle_request(topic=topic, request=r)
        except CollectiveCommHandleError:
            self.fire_event(CollectiveCommEvent.FAILED, fl_ctx)
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)
        result = Shareable()
        result.set_header(CollectiveCommShareableHeader.BUFFER, self._buffer)
        self._sequence_number = sequence_number + 1
        self._buffer = 0
        return result

    def _handle_request(self, topic: str, request: Shareable):
        sequence_number = request.get_header(CollectiveCommShareableHeader.SEQUENCE_NUMBER)

        # use the first sequence number as sequence number
        if self._sequence_number is None:
            self._sequence_number = sequence_number
            self._buffer = None

        if sequence_number != self._sequence_number:
            raise CollectiveCommHandleError("sequence number does not match")

        if topic not in self._function:
            raise CollectiveCommHandleError(f"topic {topic} is not supported.")
        func = self._function[topic]
        self._buffer = func(request=request, world_size=self._world_size, buffer=self._buffer)
