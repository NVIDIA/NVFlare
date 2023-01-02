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

from nvflare.fuel.f3.cellnet.cell import Cell, Message
from nvflare.private.defs import CellChannel, SessionTopic


class ServerSessionManager:

    def __init__(
            self,
            engine
    ):
        self.engine = engine
        cell = engine.get_cell()
        assert isinstance(cell, Cell)
        cell.set_message_interceptor(self._inspect_message)
        cell.register_request_cb(
            channel=CellChannel.SESSION,
            topic=SessionTopic.REGISTER,
            cb=self._do_login
        )

    def _inspect_message(self, message: Message):
        state_check = self.server_state.register(fl_ctx)
        if state_check.get(ACTION) == NIS:
            context.abort(
                grpc.StatusCode.FAILED_PRECONDITION,
                state_check.get(MESSAGE),
            )
        elif state_check.get(ACTION) == ABORT_RUN:
            context.abort(
                grpc.StatusCode.ABORTED,
                state_check.get(MESSAGE),
            )


