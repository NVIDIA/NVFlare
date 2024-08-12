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
import grpc


def create_channel(server_addr, grpc_options, ready_timeout: float, test_only: bool):
    """Create gRPC channel and waits for the server to be ready

    Args:
        server_addr: the gRPC server address to connect to
        grpc_options: gRPC client connection options
        ready_timeout: how long to wait for the server to be ready
        test_only: whether for testing the server readiness only

    Returns: the gRPC channel created. Bit if test_only, the channel is closed and returns None.

    If the server does not become ready within ready_timeout, the RuntimeError exception will raise.

    """
    channel = grpc.insecure_channel(server_addr, options=grpc_options)

    # wait for channel ready
    try:
        grpc.channel_ready_future(channel).result(timeout=ready_timeout)
    except grpc.FutureTimeoutError:
        raise RuntimeError(f"cannot connect to server after {ready_timeout} seconds")

    if test_only:
        channel.close()
        channel = None
    return channel
