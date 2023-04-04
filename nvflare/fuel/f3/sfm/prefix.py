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
import struct
from dataclasses import dataclass

from nvflare.fuel.f3.comm_error import CommError
from nvflare.fuel.f3.connection import BytesAlike

PREFIX_STRUCT = struct.Struct(">IHBBHHHH")
PREFIX_LEN = PREFIX_STRUCT.size


@dataclass
class Prefix:
    """Prefix is the 16-byte fixed header for the SFM frame, every frame must have this prefix.
    Beside all the other attributes, it provides framing for the message. Framing is needed if the
    frame is sent over byte streams like TCP or sockets.

    The 8 fields in the prefix are all integers encoded in big-endian,

        1. length(4): Total length of the frame.
        2. header_len(2): Length of the encoded headers
        3. type(1): Frame type (DATA, HELLO etc)
        4. reserved(1): Not used, 0
        5. flags(2): Attribute of the frame (OOB, ACK etc).
        6. app_id(2): Application ID to support multiple apps
        7. stream_id(2): Stream ID to connect all fragments of a stream
        8. sequence(2): A sequence number for each frame. Used to detect lost frames.

    """

    length: int = 0
    header_len: int = 0
    type: int = 0
    reserved: int = 0
    flags: int = 0
    app_id: int = 0
    stream_id: int = 0
    sequence: int = 0

    @staticmethod
    def from_bytes(buffer: bytes) -> "Prefix":
        if len(buffer) < PREFIX_LEN:
            raise CommError(CommError.BAD_DATA, "Prefix too short")

        return Prefix(*PREFIX_STRUCT.unpack_from(buffer, 0))

    def to_buffer(self, buffer: BytesAlike, offset: int):
        PREFIX_STRUCT.pack_into(
            buffer,
            offset,
            self.length,
            self.header_len,
            self.type,
            self.reserved,
            self.flags,
            self.app_id,
            self.stream_id,
            self.sequence,
        )
