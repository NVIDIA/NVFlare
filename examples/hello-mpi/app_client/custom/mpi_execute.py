# Copyright (c) 2021, NVIDIA CORPORATION.
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


from nvflare.apis.collective_comm_constants import CollectiveCommRequestTopic, CollectiveCommShareableHeader
from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import FLContextKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal


class MPIExecutor(Executor):
    def execute(
        self,
        task_name: str,
        shareable: Shareable,
        fl_ctx: FLContext,
        abort_signal: Signal,
    ) -> Shareable:
        # retrieve model weights download from server's shareable
        if abort_signal.triggered:
            return make_reply(ReturnCode.TASK_ABORTED)

        if task_name != "mpi_train":
            return make_reply(ReturnCode.TASK_UNKNOWN)

        engine = fl_ctx.get_engine()
        client_name = fl_ctx.get_prop(FLContextKey.CLIENT_NAME)
        sequence_number = 0
        rounds = 10

        self.log_info(fl_ctx, f"Running MPIExecutor client training in {client_name}...")
        while not (abort_signal.triggered or sequence_number > rounds):
            request = Shareable()
            request.set_header(CollectiveCommShareableHeader.RANK, client_name)
            request.set_header(CollectiveCommShareableHeader.SEQUENCE_NUMBER, sequence_number)
            request.set_header(CollectiveCommShareableHeader.BUFFER, sequence_number)
            request.set_header(CollectiveCommShareableHeader.REDUCE_FUNCTION, "SUM")
            request.set_header(CollectiveCommShareableHeader.IS_COLLECTIVE_AUX, True)
            self.log_info(fl_ctx, f"Client {client_name} send request with sequence_number {sequence_number}")
            result: Shareable = engine.send_aux_request(
                topic=CollectiveCommRequestTopic.ALL_REDUCE, request=request, timeout=30.0, fl_ctx=fl_ctx
            )
            if result.get_return_code() != ReturnCode.OK:
                return make_reply(ReturnCode.EXECUTION_EXCEPTION)
            self.log_info(
                fl_ctx, f"Client {client_name} get result {result.get_header(CollectiveCommShareableHeader.BUFFER)}"
            )
            sequence_number += 1
        self.log_info(fl_ctx, "Training finished. Returning shareable...")
        model = Shareable()
        return model
