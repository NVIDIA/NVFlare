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
import struct
from io import BytesIO
from typing import List

SIGNATURE = "NVDADAM1"  # DAM (Direct Accessible Marshalling) V1
PREFIX_LEN = 24

DATA_TYPE_INT = 1
DATA_TYPE_FLOAT = 2
DATA_TYPE_STRING = 3
DATA_TYPE_INT_ARRAY = 257
DATA_TYPE_FLOAT_ARRAY = 258


class DamEncoder:
    def __init__(self, data_set_id: int):
        self.data_set_id = data_set_id
        self.entries = []
        self.buffer = BytesIO()

    def add_int_array(self, value: List[int]):
        self.entries.append((DATA_TYPE_INT_ARRAY, value))

    def add_float_array(self, value: List[float]):
        self.entries.append((DATA_TYPE_FLOAT_ARRAY, value))

    def finish(self) -> bytes:
        size = PREFIX_LEN
        for entry in self.entries:
            size += 16
            size += len(entry[1]) * 8

        self.write_str(SIGNATURE)
        self.write_int64(size)
        self.write_int64(self.data_set_id)

        for entry in self.entries:
            data_type, value = entry
            self.write_int64(data_type)
            self.write_int64(len(value))

            for x in value:
                if data_type == DATA_TYPE_INT_ARRAY:
                    self.write_int64(x)
                else:
                    self.write_float(x)

        return self.buffer.getvalue()

    def write_int64(self, value: int):
        self.buffer.write(struct.pack("q", value))

    def write_float(self, value: float):
        self.buffer.write(struct.pack("d", value))

    def write_str(self, value: str):
        self.buffer.write(value.encode("utf-8"))


class DamDecoder:
    def __init__(self, buffer: bytes):
        self.buffer = buffer
        self.pos = 0
        if len(buffer) >= PREFIX_LEN:
            self.signature = self.read_string(8)
            self.size = self.read_int64()
            self.data_set_id = self.read_int64()
        else:
            self.signature = None
            self.size = 0
            self.data_set_id = 0

    def is_valid(self):
        return self.signature == SIGNATURE

    def get_data_set_id(self):
        return self.data_set_id

    def decode_int_array(self) -> List[int]:
        data_type = self.read_int64()
        if data_type != DATA_TYPE_INT_ARRAY:
            raise RuntimeError("Invalid data type for int array")

        num = self.read_int64()
        result = [0] * num
        for i in range(num):
            result[i] = self.read_int64()

        return result

    def decode_float_array(self):
        data_type = self.read_int64()
        if data_type != DATA_TYPE_FLOAT_ARRAY:
            raise RuntimeError("Invalid data type for float array")

        num = self.read_int64()
        result = [0.0] * num
        for i in range(num):
            result[i] = self.read_float()

        return result

    def read_string(self, length: int) -> str:
        result = self.buffer[self.pos : self.pos + length].decode("latin1")
        self.pos += length
        return result

    def read_int64(self) -> int:
        (result,) = struct.unpack_from("q", self.buffer, self.pos)
        self.pos += 8
        return result

    def read_float(self) -> float:
        (result,) = struct.unpack_from("d", self.buffer, self.pos)
        self.pos += 8
        return result
