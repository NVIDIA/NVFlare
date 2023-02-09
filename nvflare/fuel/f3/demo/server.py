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
import os
import time

from nvflare.fuel.f3.communicator import Communicator
from nvflare.fuel.f3.demo.callbacks import TimingReceiver, DemoEndpointMonitor, make_message, RequestReceiver, \
    AdHocReceiver
from nvflare.fuel.f3.drivers.connnector import Mode
from nvflare.fuel.f3.endpoint import Endpoint
from nvflare.fuel.f3.message import AppIds, Message

logging.basicConfig(level=logging.DEBUG)
formatter = logging.Formatter(fmt="%(relativeCreated)6d [%(threadName)-12s] [%(levelname)-5s] %(name)s: %(message)s")
handler = logging.StreamHandler()
handler.setFormatter(formatter)
root_log = logging.getLogger()
root_log.handlers.clear()
root_log.addHandler(handler)
log = logging.getLogger(__name__)

endpoints = []

cwd = os.getcwd()
conn_props = {
    "ca_cert": cwd + "/certs/ca.crt",
    "server_cert": cwd + "/certs/server.crt",
    "server_key": cwd + "/certs/server.key",
    "client_cert": cwd + "/certs/client.crt",
    "client_key": cwd + "/certs/client.key",
}

local_endpoint = Endpoint("demo.server", {"test": 456}, conn_props)
communicator = Communicator(local_endpoint)

listening_url = "grpc://localhost:4321"
handle1 = communicator.add_connector(listening_url, Mode.PASSIVE, True)

#connect_url = "otcp://localhost:1234"
#handle2 = communicator.add_connector(connect_url, Mode.ACTIVE)

communicator.register_monitor(DemoEndpointMonitor(local_endpoint.name, endpoints))
communicator.register_message_receiver(AppIds.CELL_NET, TimingReceiver())
communicator.register_message_receiver(AppIds.DEFAULT, RequestReceiver(communicator))
communicator.register_message_receiver(123, AdHocReceiver(communicator))
communicator.start()
log.info("Server is started")

count = 0
while count < 60:
    # Wait till one endpoint is available
    if endpoints:

        # Server can send message to client also
        msg1 = make_message(None, "Async message from server")
        communicator.send(endpoints[0], AppIds.CELL_NET, msg1)

        # Message can be empty
        msg2 = Message(None, None)
        communicator.send(endpoints[0], AppIds.CELL_NET, msg2)
        break

    time.sleep(1)
    count += 1

time.sleep(10)
communicator.remove_connector(handle1)
# communicator.remove_connector(handle2)
communicator.stop()
log.info("Server stopped!")
