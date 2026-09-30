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
from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional, Union

from cryptography import x509
from cryptography.hazmat.primitives import serialization
from cryptography.x509.oid import NameOID

from nvflare.fuel.f3.drivers.connector_info import ConnectorInfo, Mode
from nvflare.fuel.f3.drivers.driver_params import DriverParams

log = logging.getLogger(__name__)
lock = threading.Lock()
conn_count = 0

BytesAlike = Union[bytes, bytearray, memoryview, list]


def create_connection_name():
    global lock, conn_count
    with lock:
        conn_count += 1
    return "CN%05d" % conn_count


class ConnState(Enum):
    IDLE = 1  # Initial state
    CONNECTED = 2  # New connection
    CLOSED = 3  # Connection is closed


class FrameReceiver(ABC):
    @abstractmethod
    def process_frame(self, frame: BytesAlike):
        """Frame received callback

        Args:
            frame: The frame received

        Raises:
            CommError: If any error happens while processing the frame
        """
        pass


class Connection(ABC):
    """FCI connection spec. A connection is used to transfer opaque frames"""

    def __init__(self, connector: ConnectorInfo):
        self.name = create_connection_name()
        self.state = ConnState.IDLE
        self.frame_receiver = None
        self.connector = connector

    @abstractmethod
    def get_conn_properties(self) -> dict:
        """Get connection specific properties, like peer address, TLS certificate etc

        Raises:
            CommError: If any errors
        """
        pass

    @staticmethod
    def record_peer(conn_props: dict, peer_cert: Optional[bytes], secure: bool = False) -> None:
        """Record the authenticated peer: its certificate (PEER_CERT, DER) and the CN derived from it (PEER_CN).

        peer_cert is DER or PEM, as the transport provides it. A secure connection with no peer
        certificate (TLS without client authentication) reports PEER_CN "N/A".
        """
        if not peer_cert:
            if secure:
                conn_props[DriverParams.PEER_CN.value] = "N/A"
            return
        if peer_cert.startswith(b"-----BEGIN"):
            cert = x509.load_pem_x509_certificate(peer_cert)
        else:
            cert = x509.load_der_x509_certificate(peer_cert)
        common_names = cert.subject.get_attributes_for_oid(NameOID.COMMON_NAME)
        conn_props[DriverParams.PEER_CERT.value] = cert.public_bytes(serialization.Encoding.DER)
        conn_props[DriverParams.PEER_CN.value] = common_names[0].value if common_names else "N/A"

    @abstractmethod
    def close(self):
        """Close connection

        Raises:
            CommError: If any errors
        """
        pass

    @abstractmethod
    def send_frame(self, frame: BytesAlike):
        """Send a SFM frame through the connection to the remote endpoint.

        Args:
            frame: The frame to be sent

        Raises:
            CommError: If any error happens while sending the frame
        """
        pass

    def register_frame_receiver(self, receiver: FrameReceiver):
        """Register frame receiver

        Args:
            receiver: The frame receiver
        """
        self.frame_receiver = receiver

    def process_frame(self, frame: BytesAlike):
        """A convenience function to call frame receiver

        Args:
            frame: The frame to be processed

        Raises:
            CommError: If any error happens while processing the frame
        """

        if self.frame_receiver:
            self.frame_receiver.process_frame(frame)
        else:
            log.error(f"Frame receiver not registered for {self}")

    def __str__(self):

        if self.state != ConnState.CONNECTED:
            return f"[{self.name} Not Connected]"

        conn_props = self.get_conn_properties()
        local_addr = conn_props.get(DriverParams.LOCAL_ADDR, "N/A")
        peer_addr = conn_props.get(DriverParams.PEER_ADDR, "N/A")
        direction = "=>" if self.connector.mode == Mode.ACTIVE else "<="
        peer_cn = conn_props.get(DriverParams.PEER_CN, None)
        cn = " SSL " + peer_cn if peer_cn else ""
        return f"[{self.name} {local_addr} {direction} {peer_addr}{cn}]"
