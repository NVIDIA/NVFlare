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

# from __future__ import annotations
from nvflare.fuel.utils import fobs


class Learnable(dict):
    def to_bytes(self) -> bytes:
        """Method to serialize the Learnable object into bytes.

        Returns:
            object serialized in bytes.

        """
        return fobs.dumps(self)

    @classmethod
    def from_bytes(cls, data: bytes):
        """Method to convert the object bytes into Learnable object.

        Args:
            data: a bytes object

        Returns:
            an object loaded by FOBS from data

        """
        return fobs.loads(data)
