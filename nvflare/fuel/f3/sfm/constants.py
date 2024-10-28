# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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


class Types:
    DATA = 1
    FRAG = 2
    ACK = 3
    HELLO = 4
    READY = 5
    PING = 6
    PONG = 7


class HandshakeKeys:
    ENDPOINT_NAME = "endpoint_name"
    TIMESTAMP = "timestamp"


class Flags:
    # Out of band message
    OOB = 0x8000
    # ACK requested
    ACK = 0x4000
    # Request, message-id in the header
    REQ = 0x2000
    # Response, message-id in the header
    RESP = 0x1000
    # PUB/SUB message, topic is in the header
    PUB_SUB = 0x0800
