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
import uuid

from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey
from nvflare.fuel.f3.streaming.stream_types import StreamError

from nvflare.fuel.f3.cellnet.fqcn import FQCN
from nvflare.fuel.f3.drivers.driver_params import DriverParams
from nvflare.fuel.f3.message import Message
from nvflare.fuel.f3.streaming.stream_const import STREAM_CHANNEL, STREAM_CERT_TOPIC

from nvflare.fuel.f3.cellnet.core_cell import CoreCell

log = logging.getLogger(__name__)

CERT_REQ_ID = "cert_req_id"
CERT_REPLY = "cert_reply"
CERT_ENDPOINT = "cert_endpoint"
CERT_CONTENT = "cert_content"
CERT_CA_CONTENT = "cert_ca_content"
CERT_REQ_TIMEOUT = 60


class CertificateManager:

    def __init__(self, core_cell: CoreCell):
        self.core_cell = core_cell
        self.cert_cache = {}
        self.requests = {}
        self.lock = threading.Lock()
        self.core_cell.register_request_cb(channel=STREAM_CHANNEL, topic=STREAM_CERT_TOPIC, cb=self._cert_handler)

        conn_props = core_cell.endpoint.conn_props
        ca_cert_path = conn_props.get(DriverParams.CA_CERT)
        server_cert_path = conn_props.get(DriverParams.SERVER_CERT)
        if server_cert_path:
            local_cert_path = server_cert_path
            local_key_path = conn_props.get(DriverParams.SERVER_KEY)
        else:
            local_cert_path = conn_props.get(DriverParams.CLIENT_CERT)
            local_key_path = conn_props.get(DriverParams.CLIENT_KEY)

        if not local_cert_path:
            log.debug("Certificate is not configured, secure message is not supported")

        self.ca_cert = self.read_file(ca_cert_path)
        self.local_cert = self.read_file(local_cert_path)
        self.local_key = self.read_file(local_key_path)

    def get_certificate(self, fqcn: str) -> bytes:

        if not self.local_cert:
            raise StreamError(f"This instance doesn't support certificate exchange, not running in secure mode")

        endpoint = FQCN.get_root(fqcn)
        cert = self.cert_cache.get(endpoint)
        if cert:
            return cert

        req_id = str(uuid.uuid4())
        req = {
            CERT_REQ_ID: req_id,
            CERT_REPLY: False,
            CERT_CONTENT: self.local_cert,
            CERT_CA_CONTENT: self.ca_cert
        }

        errors = self.core_cell.fire_and_forget(STREAM_CHANNEL, STREAM_CERT_TOPIC, endpoint, Message(None, req))
        error = errors.get(endpoint)
        if error:
            raise StreamError(f"Failed to send certificate request to  {endpoint}: {error}")

        with self.lock:
            event = threading.Event()
            self.requests[req_id] = (event, None)

        result = event.wait(CERT_REQ_TIMEOUT)
        with self.lock:
            _, reply = self.requests.pop(req_id)

        if not result:
            raise StreamError(f"Cert request to {endpoint} timed out after {CERT_REQ_TIMEOUT} seconds")

        self.cert_cache[endpoint] = reply
        return reply

    def get_local_cert(self):
        return self.local_cert

    def get_local_key(self):
        return self.local_key

    def get_ca_cert(self):
        return self.ca_cert

    @staticmethod
    def read_file(file_name: str):
        if not file_name:
            return None

        with open(file_name, "rb") as f:
            return f.read()

    def _cert_handler(self, message: Message):
        req = message.payload
        req_id = req.get(CERT_REQ_ID)
        cert = req.get(CERT_CONTENT)
        is_reply = req.get(CERT_REPLY)

        if is_reply:
            with self.lock:
                event, _ = self.requests.get(req_id)
                self.requests[req_id] = (event, cert)
            event.set()
        else:
            reply = {
                CERT_REQ_ID: req_id,
                CERT_REPLY: True,
                CERT_CONTENT: self.local_cert,
                CERT_CA_CONTENT: self.ca_cert
            }

            origin = message.get_header(MessageHeaderKey.ORIGIN)
            errors = self.core_cell.fire_and_forget(STREAM_CHANNEL, STREAM_CERT_TOPIC, origin,
                                                    Message(None, reply))
            error = errors.get(origin)
            if error:
                raise StreamError(f"Failed to send certificate response to  {origin}: {error}")
