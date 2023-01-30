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

# Hard-coded stream ID to be used by packets before handshake
import threading
from typing import Optional, List

from nvflare.fuel.f3.endpoint import Endpoint
from nvflare.fuel.f3.sfm.sfm_conn import SfmConnection

RESERVED_STREAM_ID = 16


class SfmEndpoint:
    """An endpoint wrapper to keep SFM internal data"""

    def __init__(self, endpoint: Endpoint):
        self.endpoint = endpoint
        self.stream_id: int = RESERVED_STREAM_ID
        self.lock = threading.Lock()
        self.connections: List[SfmConnection] = []

    def add_connection(self, sfm_conn: SfmConnection):
        self.connections.append(sfm_conn)

    def get_connection(self, stream_id: int) -> Optional[SfmConnection]:
        if not self.connections:
            return None

        index = stream_id % len(self.connections)
        return self.connections[index]

    def next_stream_id(self) -> int:
        """Get next stream_id for the endpoint
        stream_id is used to assemble fragmented data
        """

        with self.lock:
            self.stream_id = (self.stream_id + 1) & 0xffff
            return self.stream_id
