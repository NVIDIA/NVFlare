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
import logging
import os
import threading
import time

from nvflare.fuel.f3.communicator import Communicator
from nvflare.fuel.f3.demo.callbacks import DemoEndpointMonitor, TimingReceiver, make_message, LoopbackReceiver
from nvflare.fuel.f3.drivers.connnector import Mode
from nvflare.fuel.f3.drivers.driver_params import DriverParams
from nvflare.fuel.f3.endpoint import Endpoint
from nvflare.fuel.f3.message import AppIds, Headers, Message

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
local_endpoint = Endpoint("demo.client", {"test": 123}, conn_props)

communicator = Communicator(local_endpoint)
communicator.register_message_receiver(10, LoopbackReceiver(communicator))
communicator.send(communicator.local_endpoint, 10, Message(None, "Test".encode("utf-8")))

connect_url = "grpc://localhost:4321"
handle1 = communicator.add_connector(connect_url, Mode.ACTIVE)

#listen_url = "otcp://localhost:1234"
#handle2 = communicator.add_connector(listen_url, Mode.PASSIVE)

resources = {
    DriverParams.SECURE: False,
    DriverParams.PORTS: "3000-6000",
}
# handle3, ad_hoc_url = communicator.start_listener("tcp", resources)

communicator.register_monitor(DemoEndpointMonitor(local_endpoint.name, endpoints))
communicator.register_message_receiver(AppIds.CELL_NET, TimingReceiver())
communicator.register_message_receiver(AppIds.DEFAULT, TimingReceiver())

communicator.start()
log.info("Client is started")

count = 0

while count < 5:

    if endpoints:
        name = endpoints[0].name
        log.info(f"Number of connections before ad-hoc {len(communicator.conn_manager.get_connections(name))}")
        msg1 = make_message(None, "Fire-forget message")
#        communicator.send(endpoints[0], 123, Message(None, ad_hoc_url.encode("utf-8")))
        communicator.send(endpoints[0], AppIds.CELL_NET, msg1)
        time.sleep(1)
        log.info(f"Number of connections after ad-hoc {len(communicator.conn_manager.get_connections(name))}")

        # MSG_ID can be used to match up response with request
        headers = Headers()
        headers[Headers.MSG_ID] = "1234"
        msg2 = make_message(headers, "Request with request_id")
        communicator.send(endpoints[0], AppIds.DEFAULT, msg2)
        time.sleep(1)
        break

    time.sleep(1)
    count += 1

time.sleep(10)
# communicator.remove_connector(handle1)
# communicator.remove_connector(handle2)
communicator.stop()
for thread in threading.enumerate():
    print(thread.name)
log.info("Client stopped!")
