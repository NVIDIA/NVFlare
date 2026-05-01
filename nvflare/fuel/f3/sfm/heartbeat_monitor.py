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
import time
from threading import Event, Thread
from typing import Dict

from nvflare.fuel.f3.comm_config import CommConfigurator
from nvflare.fuel.f3.drivers.driver_params import DriverCap
from nvflare.fuel.f3.sfm.constants import Types
from nvflare.fuel.f3.sfm.sfm_conn import SfmConnection

log = logging.getLogger(__name__)

HEARTBEAT_TICK = 5
DEFAULT_HEARTBEAT_INTERVAL = 60
DEFAULT_SEND_STALL_CONSECUTIVE_CHECKS = 3


class HeartbeatMonitor(Thread):
    def __init__(self, conns: Dict[str, SfmConnection]):
        Thread.__init__(self, name="hb_mon", daemon=True)
        self.conns = conns
        self.stopped = Event()
        self.curr_time = 0
        config = CommConfigurator()
        self.interval = config.get_heartbeat_interval(DEFAULT_HEARTBEAT_INTERVAL)
        self.send_stall_timeout = config.get_sfm_send_stall_timeout(45.0)
        self.close_stalled_connection = config.get_sfm_close_stalled_connection(False)
        self.stall_consecutive_checks = max(
            1, config.get_sfm_send_stall_consecutive_checks(DEFAULT_SEND_STALL_CONSECUTIVE_CHECKS)
        )
        self.stall_counts = {}
        if self.interval < HEARTBEAT_TICK:
            log.warning(f"Heartbeat interval is too small ({self.interval} < {HEARTBEAT_TICK})")

    def stop(self):
        self.stopped.set()

    def run(self):

        while not self.stopped.is_set():
            try:
                self.curr_time = time.time()
                self._check_heartbeat()
            except Exception as ex:
                log.error(f"Heartbeat check failed: {ex}")

            self.stopped.wait(HEARTBEAT_TICK)

        log.debug("Heartbeat monitor stopped")

    def _check_heartbeat(self):

        active_keys = set()
        for sfm_conn in self.conns.values():
            conn_key = sfm_conn.get_name() if hasattr(sfm_conn, "get_name") else str(id(sfm_conn))
            active_keys.add(conn_key)

            stall_sec = sfm_conn.get_send_stall_seconds()
            if stall_sec > self.send_stall_timeout:
                count = self.stall_counts.get(conn_key, 0) + 1
                self.stall_counts[conn_key] = count
                log.warning(
                    f"Detected stalled send on {sfm_conn.conn}: blocked {stall_sec:.1f}s "
                    f"({count}/{self.stall_consecutive_checks})"
                )
                if self.close_stalled_connection and count >= self.stall_consecutive_checks:
                    sfm_conn.conn.close()
                continue

            self.stall_counts[conn_key] = 0

            driver = sfm_conn.conn.connector.driver
            caps = driver.capabilities()
            if caps and not caps.get(DriverCap.SEND_HEARTBEAT.value, False):
                continue

            if self.curr_time - sfm_conn.last_activity > self.interval:
                sfm_conn.send_heartbeat(Types.PING)
                log.debug(f"Heartbeat sent to connection: {sfm_conn.conn}")

        stale_keys = [k for k in self.stall_counts.keys() if k not in active_keys]
        for k in stale_keys:
            self.stall_counts.pop(k, None)
