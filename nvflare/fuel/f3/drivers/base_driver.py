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
import logging
import threading
from typing import Dict, Optional

from nvflare.fuel.f3.connection import Connection, ConnState
from nvflare.fuel.f3.drivers.driver import Driver, Connector, DriverParams

log = logging.getLogger(__name__)


# noinspection PyAbstractClass
class BaseDriver(Driver):
    """Common base class for all drivers
    It contains all the common connection management code
    """

    def __init__(self):
        super().__init__()
        self.connections: Dict[str, Connection] = {}
        self.connector: Optional[Connector] = None
        self.conn_lock = threading.Lock()

    def add_connection(self, conn: Connection):
        conn_props = conn.get_conn_properties()

        if log.isEnabledFor(logging.DEBUG):
            local_addr = conn_props.get(DriverParams.LOCAL_ADDR, "N/A")
            peer_addr = conn_props.get(DriverParams.PEER_ADDR, "N/A")
            log.debug(f"Connection created: {self.get_name()}:{conn.name}, Local: {local_addr} Peer: {peer_addr}")

        with self.conn_lock:
            self.connections[conn.name] = conn

        conn.state = ConnState.CONNECTED
        self._notify_monitor(conn)

    def close_connection(self, conn: Connection):
        log.debug(f"Connection removed: {self.get_name()}:{conn.name}")

        with self.conn_lock:
            self.connections.pop(conn.name)

        conn.state = ConnState.CLOSED
        self._notify_monitor(conn)

    def close_all(self):
        with self.conn_lock:
            for name in sorted(self.connections.keys()):
                conn = self.connections[name]
                log.debug(f"Closing connection: {self.get_name()}:{name}")
                conn.close()

    def _notify_monitor(self, conn: Connection):
        if not self.conn_monitor:
            log.error(f"Connection monitor not registered for driver {self.get_name()}")
        else:
            self.conn_monitor.state_change(conn)
