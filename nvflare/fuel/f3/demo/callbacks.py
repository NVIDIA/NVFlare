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
from typing import List

import msgpack

from nvflare.fuel.f3.communicator import MessageReceiver, Communicator
from nvflare.fuel.f3.endpoint import EndpointMonitor, Endpoint, EndpointState
from nvflare.fuel.f3.message import Message, Headers

logging.basicConfig()
logging.getLogger().setLevel(logging.DEBUG)
log = logging.getLogger(__name__)

TIMESTAMP = "timestamp"
MESSAGE = "message"


def make_message(headers: Headers, msg: str) -> Message:
    data = {
        TIMESTAMP: time.time(),
        MESSAGE: msg}

    return Message(headers, msgpack.packb(data))


class DemoEndpointMonitor(EndpointMonitor):

    def __init__(self, name: str, endpoint_list: List[Endpoint]):
        self.name = name
        self.endpoint_list = endpoint_list

    def state_change(self, endpoint: Endpoint):
        if endpoint.state == EndpointState.READY:
            log.info(f"Endpoint {endpoint.name} is created")
            self.endpoint_list.append(endpoint)


class TimingReceiver(MessageReceiver):
    def process_message(self, endpoint: Endpoint, app_id: int, message: Message):

        if message.payload:
            data = msgpack.unpackb(message.payload)
            timestamp = data.get(TIMESTAMP)
            delay = time.time() - timestamp

            log.info(f"Message delay {delay:.6f} seconds Message: '{data.get(MESSAGE)}' Source: {endpoint.name}")
        else:
            log.info(f"Received empty message from {endpoint.name}")

        log.info(f"Headers: {message.headers}")


class RequestReceiver(MessageReceiver):

    def __init__(self, communicator: Communicator):
        self.communicator = communicator

    def process_message(self, endpoint: Endpoint, app_id: int, message: Message):

        data = msgpack.unpackb(message.payload)
        timestamp = data.get(TIMESTAMP)
        delay = time.time() - timestamp

        log.info(f"Message delay {delay:.6f} seconds Message: '{data.get(MESSAGE)}' Source: {endpoint.name}")
        log.info(f"Headers: {message.headers}")

        # Request header with MSG_ID is included with response
        response = make_message(message.headers, f"Response to message '{data.get(MESSAGE)}")
        self.communicator.send(endpoint, app_id, response)
