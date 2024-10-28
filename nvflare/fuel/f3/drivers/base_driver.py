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
from abc import ABC
from typing import Dict, Optional

from nvflare.fuel.f3.comm_error import CommError
from nvflare.fuel.f3.connection import Connection, ConnState
from nvflare.fuel.f3.drivers.driver import ConnectorInfo, Driver

log = logging.getLogger(__name__)


class BaseDriver(Driver, ABC):
    """Common base class for all drivers
    It contains all the common connection management code
    """

    def __init__(self):
        super().__init__()
        self.connections: Dict[str, Connection] = {}
        self.connector: Optional[ConnectorInfo] = None
        self.conn_lock = threading.Lock()

    def add_connection(self, conn: Connection):
        with self.conn_lock:
            self.connections[conn.name] = conn

        conn.state = ConnState.CONNECTED
        self._notify_monitor(conn)

        log.debug(f"Connection created: {self.get_name()}:{conn}")

    def close_connection(self, conn: Connection):
        log.debug(f"Connection removed: {self.get_name()}:{conn}")

        conn.state = ConnState.CLOSED
        self._notify_monitor(conn)

        with self.conn_lock:
            if not self.connections.pop(conn.name, None):
                log.debug(f"{conn.name} is already removed from driver")

    def close_all(self):
        with self.conn_lock:
            for name in sorted(self.connections.keys()):
                conn = self.connections[name]
                log.debug(f"Closing connection: {self.get_name()}:{conn}")
                conn.close()

    def _notify_monitor(self, conn: Connection):
        if not self.conn_monitor:
            raise CommError(CommError.ERROR, f"Connection monitor not registered for driver {self.get_name()}")

        self.conn_monitor.state_change(conn)
