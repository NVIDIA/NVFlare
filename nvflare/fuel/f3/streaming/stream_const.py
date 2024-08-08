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

STREAM_PREFIX = "sm__"
STREAM_CHANNEL = STREAM_PREFIX + "STREAM"
STREAM_DATA_TOPIC = STREAM_PREFIX + "DATA"
STREAM_ACK_TOPIC = STREAM_PREFIX + "ACK"
STREAM_CERT_TOPIC = STREAM_PREFIX + "CERT"

# End of Stream indicator
EOS = bytes()


class StreamDataType:
    # Payload chunk
    CHUNK = 1
    # Final chunk, end of stream
    FINAL = 2
    # ACK with last received offset
    ACK = 3
    # Resume request
    RESUME = 4
    # Resume ack with offset to start
    RESUME_ACK = 5
    # Streaming failed
    ERROR = 6


class StreamHeaderKey:

    # Try to keep the key small to reduce the overhead
    STREAM_ID = STREAM_PREFIX + "id"
    DATA_TYPE = STREAM_PREFIX + "dt"
    SIZE = STREAM_PREFIX + "sz"
    SEQUENCE = STREAM_PREFIX + "sq"
    OFFSET = STREAM_PREFIX + "os"
    ERROR_MSG = STREAM_PREFIX + "em"
    CHANNEL = STREAM_PREFIX + "ch"
    FILE_NAME = STREAM_PREFIX + "fn"
    TOPIC = STREAM_PREFIX + "tp"
    OBJECT_STREAM_ID = STREAM_PREFIX + "os"
    OBJECT_INDEX = STREAM_PREFIX + "oi"
    STREAM_REQ_ID = STREAM_PREFIX + "ri"
    PAYLOAD_ENCODING = STREAM_PREFIX + "pe"
    OPTIONAL = STREAM_PREFIX + "op"
