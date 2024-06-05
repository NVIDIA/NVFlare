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

import nvflare.app_opt.flower.proto.grpcadapter_pb2 as pb2
from nvflare.app_opt.flower.proto.grpcadapter_pb2_grpc import GrpcAdapterServicer
from nvflare.fuel.utils.obj_utils import get_logger


class EchoServicer(GrpcAdapterServicer):
    def __init__(self):
        self.logger = get_logger(self)

    def SendReceive(self, request: pb2.MessageContainer, context):
        msg_name = request.grpc_message_name
        headers = request.metadata
        content = request.grpc_message_content
        self.logger.info(f"got {msg_name=}: {headers=} {content=}")
        return pb2.MessageContainer(
            metadata=headers,
            grpc_message_name=msg_name,
            grpc_message_content=content,
        )
