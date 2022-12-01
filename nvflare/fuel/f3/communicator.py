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
from typing import Optional

from nvflare.fuel.f3.endpoint import Endpoint, EndpointMonitor
from nvflare.fuel.f3.receiver import Receiver
from nvflare.fuel.f3.responder import Responder
from nvflare.fuel.f3.message import Message


class Communicator:

    def __init__(self, local_endpoint: Endpoint):
        self.local_endpoint = local_endpoint
        self.monitors = []

    def start(self):
        """Start the communicator and establishing all the connections

        Raises:
            CommError: If any error encountered while starting up
        """
        pass

    def stop(self):
        """Stop the communicator and shutdown all the connections

        Raises:
            CommError: If any error encountered while shutting down
        """
        pass

    def register_monitor(self, monitor: EndpointMonitor):
        """Register a monitor for endpoint lifecycle changes

        This monitor is notified for any state changes of all the endpoints

        Args:
            monitor: The class that receives the state change notification

        Raises:
            CommError: If any error happens while sending the request
        """
        pass

    def send(self, endpoint: Endpoint, app: int, message: Message, reliable=False):
        """Send a message to endpoint for app

        This method is similar to a HTTP request or RPC call.

        Args:
            endpoint: An endpoint to send the request to
            app: Application ID
            message: Message to send
            reliable: Reliable message, ack requested.

        Raises:
            CommError: If any error happens while sending the request
        """
        pass

    def register_receiver(self, endpoint: Optional[Endpoint], app: int, receiver: Receiver):
        """Register a receiver to process FCI message

         Args:
             endpoint: Endpoint of the message, None to handle messages from all endpoint
             app: Application ID
             receiver: The class to process the message

         Raises:
             CommError: If duplicate endpoint/app or responder is of wrong type
         """

        pass

    def request(self, endpoint: Endpoint, app: int, message: Message, timeout_ms=0) -> Message:
        """Send request to endpoint/channel and wait for response

        This method is similar to a HTTP or RPC call.

        Args:
            endpoint: An endpoint to send the request to
            app: Application ID
            message: Message to send
            timeout_ms: Timeout in milliseconds, 0 means system default

        Returns:
            The response message

        Raises:
            CommError: If any error happens while sending the request
        """
        pass

    def register_responder(self, endpoint: Optional[Endpoint], app: int, responder: Responder):
        """Register a responder to handle FCI request

         This is similar to HTTP server handler or gRPC servicer method

         Args:
             endpoint: Endpoint of the request, None to handle requests from all endpoints
             app: Application ID
             responder: The class to handle the request

         Raises:
             CommError: If duplicate endpoint/app or responder is of wrong type
         """

        pass

    def add_listener(self, url: str, parameters: dict):
        """Add a listener to wait for connections

         This is similar to HTTP server connection

         Args:
             url: The url to listen to in the form of https://0:8080/sfm
             parameters: Parameters for the listener

         Raises:
             CommError: If any errors
         """

        pass

    def add_connector(self, url: str, parameters: dict):
        """Add a connector to initiate connections

         This is similar to HTTP client connection

         Args:
             url: The url to make connection to in the form of https://server:8080/sfm
             parameters: Parameters for the listener

         Raises:
             CommError: If any errors
         """

        pass
