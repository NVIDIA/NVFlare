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
import logging
import threading
from typing import List, Optional

from nvflare.fuel.f3.endpoint import Endpoint
from nvflare.fuel.f3.sfm.sfm_conn import SfmConnection

# Hard-coded stream ID to be used by packets before handshake
RESERVED_STREAM_ID = 16
MAX_CONN_PER_ENDPOINT = 1

log = logging.getLogger(__name__)


class SfmEndpoint:
    """An endpoint wrapper to keep SFM internal data"""

    def __init__(self, endpoint: Endpoint):
        self.endpoint = endpoint
        self.stream_id: int = RESERVED_STREAM_ID
        self.lock = threading.Lock()
        self.connections: List[SfmConnection] = []

    def add_connection(self, sfm_conn: SfmConnection):

        with self.lock:
            while len(self.connections) >= MAX_CONN_PER_ENDPOINT:
                first_conn = self.connections[0]
                first_conn.conn.close()
                self.connections.pop(0)
                log.info(
                    f"Connection {first_conn.get_name()} is evicted for {sfm_conn.get_name()} "
                    f"from endpoint {self.endpoint.name} for exceeding limit {MAX_CONN_PER_ENDPOINT}"
                )

            self.connections.append(sfm_conn)

    def remove_connection(self, sfm_conn: SfmConnection):

        if not self.connections:
            log.debug(
                f"Connection {sfm_conn.get_name()} is already removed. "
                f"No connections for endpoint {self.endpoint.name}"
            )
            return

        with self.lock:
            found_index = next(
                (index for index, conn in enumerate(self.connections) if conn.get_name() == sfm_conn.get_name()), None
            )

            if found_index is not None:
                self.connections.pop(found_index)
                log.debug(f"Connection {sfm_conn.get_name()} is removed from endpoint {self.endpoint.name}")
            else:
                log.debug(f"Connection {sfm_conn.get_name()} is already removed from endpoint {self.endpoint.name}")

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
            self.stream_id = (self.stream_id + 1) & 0xFFFF
            return self.stream_id
