#  Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
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

class Frame:

    PREFIX_LEN = 12

    def __init__(self, payload: bytes = None, headers: dict = None):
        self.payload = payload
        self.headers = headers
        self.length = 0
        self.header_len = 0
        self.type = 0
        self.flags = 0
        self.app = 0
        self.sequence = 0

    def get_payload_len(self):
        return self.length - Frame.PREFIX_LEN - self.header_len

    def encode_header(self) -> bytes:
        """Encode prefix and header"""
        pass

    def decode_prefix(self, buffer: bytes):
        pass

    def decode_header(self, length: int, buffer: bytes):
        pass
