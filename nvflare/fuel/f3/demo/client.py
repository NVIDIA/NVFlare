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
import time

from nvflare.fuel.f3.communicator import Communicator
from nvflare.fuel.f3.demo.callbacks import DemoEndpointMonitor, TimingReceiver, make_message
from nvflare.fuel.f3.drivers.http_driver import HttpDriver
from nvflare.fuel.f3.endpoint import Endpoint
from nvflare.fuel.f3.message import AppIds, Headers

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)
log = logging.getLogger(__name__)

endpoints = []
local_endpoint = Endpoint("demo.client", {"test": 123})

communicator = Communicator(local_endpoint)
conn_props = {
    "host": "localhost",
    "port": 4567
}

driver = HttpDriver(conn_props)
communicator.add_connector(driver)
communicator.register_monitor(DemoEndpointMonitor(local_endpoint.name, endpoints))
communicator.register_message_receiver(AppIds.CELL_NET, TimingReceiver())
communicator.register_message_receiver(AppIds.DEFAULT, TimingReceiver())
communicator.start()
log.info("Client is started")

count = 0
while count < 10:

    if endpoints:
        msg1 = make_message(None, "Fire-forget message")
        communicator.send(endpoints[0], AppIds.CELL_NET, msg1)
        time.sleep(1)

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
log.info("Client stopped!")
