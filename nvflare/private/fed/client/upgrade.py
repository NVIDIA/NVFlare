# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
import math
import threading
import time

from nvflare.apis.fl_constant import ConnectionSecurity
from nvflare.apis.signal import Signal
from nvflare.fuel.f3.cellnet.fqcn import FQCN
from nvflare.fuel.f3.cellnet.identity import CellIdentityResolver
from nvflare.fuel.f3.communicator import Communicator, Mode
from nvflare.fuel.f3.drivers.driver_params import DriverParams
from nvflare.fuel.f3.drivers.net_utils import enhance_credential_info
from nvflare.fuel.f3.endpoint import Endpoint, EndpointMonitor, EndpointState
from nvflare.fuel.f3.mpm import MainProcessMonitor

log = logging.getLogger(__name__)
DEFAULT_UPGRADE_PROBE_INTERVAL = 60.0


class _ProbeMonitor(EndpointMonitor):
    def __init__(self, peer_fqcn):
        self.peer_fqcn = peer_fqcn
        self.done = threading.Event()
        self.compatible = False

    def state_change(self, endpoint):
        if endpoint.name == self.peer_fqcn and endpoint.state in (EndpointState.READY, EndpointState.ERROR):
            self.compatible = endpoint.state == EndpointState.READY
            self.done.set()


def wait_for_server(
    fqcn: str,
    peer_fqcn: str,
    url: str,
    secure: bool,
    credentials: dict,
    resources: dict,
    identity_map: dict,
    abort_signal: Signal,
    retry_interval: float = DEFAULT_UPGRADE_PROBE_INTERVAL,
    timeout: float = 10.0,
):
    """Wait for a compatible transport peer without creating a cell or registering.

    For relayed sites the peer is the relay; operators must upgrade the server
    before starting upgraded relays. Each attempt uses the site's existing TLS
    credentials and a distinct endpoint name, and sends no application messages.
    """
    if not math.isfinite(retry_interval) or retry_interval <= 0:
        raise ValueError("upgrade_probe_interval must be positive and finite")

    # Use the wire-compatible TCP driver with bounded connect and shutdown for short-lived probes.
    scheme, separator, address = url.partition("://")
    probe_url = {"atcp": "tcp", "satcp": "stcp"}.get(scheme, scheme) + separator + address

    credentials = dict(credentials)
    enhance_credential_info(credentials)
    conn_security = (resources or {}).get(
        DriverParams.CONNECTION_SECURITY, credentials.get(DriverParams.CONNECTION_SECURITY)
    )
    if conn_security:
        secure = conn_security != ConnectionSecurity.CLEAR
    resources = {
        **(resources or {}),
        DriverParams.CONNECT_TIMEOUT.value: timeout,
        DriverParams.QUIET_RECONNECT.value: True,
    }

    def cancelled():
        return abort_signal.triggered or MainProcessMonitor.is_stopping()

    probe_fqcn = FQCN.join([fqcn, "upgrade-probe"])
    waiting_logged = False
    while not cancelled():
        attempt_start = time.monotonic()
        monitor = _ProbeMonitor(peer_fqcn)
        probe = Communicator(
            Endpoint(probe_fqcn, conn_props=credentials),
            identity_resolver=CellIdentityResolver(local_fqcn=probe_fqcn, prefix_identity_map=identity_map),
        )
        probe.register_monitor(monitor)
        try:
            probe.add_connector(probe_url, Mode.ACTIVE, secure=secure, resources=resources)
            probe.start()
            while not cancelled() and time.monotonic() - attempt_start < timeout:
                if monitor.done.wait(0.1):
                    break
        finally:
            probe.stop()

        if monitor.compatible and not cancelled():
            log.info(f"Compatible server connection available at {url}; starting client {fqcn}")
            return

        if not waiting_logged:
            log.info(f"Waiting for a compatible server at {url}; probing every {retry_interval:g} seconds")
            waiting_logged = True
        while not cancelled() and time.monotonic() - attempt_start < retry_interval:
            time.sleep(0.1)

    raise RuntimeError("Client startup cancelled while waiting for a compatible server")
