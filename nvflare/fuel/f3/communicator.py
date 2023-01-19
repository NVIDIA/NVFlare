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
#  limitations under the License
import logging
import os
from typing import Optional

from nvflare.fuel.f3 import drivers
from nvflare.fuel.f3.comm_error import CommError
from nvflare.fuel.f3.drivers.driver import Driver
from nvflare.fuel.f3.drivers.driver_manager import DriverManager
from nvflare.fuel.f3.endpoint import Endpoint, EndpointMonitor
from nvflare.fuel.f3.message import Message, MessageReceiver
from nvflare.fuel.f3.sfm.conn_manager import ConnManager, Mode

log = logging.getLogger(__name__)


class Communicator:
    """FCI main communication API"""

    def __init__(self, local_endpoint: Endpoint):
        self.local_endpoint = local_endpoint
        self.monitors = []
        self.conn_manager = ConnManager(local_endpoint)
        self.driver_mgr = DriverManager()

        # Load all the drivers in the drivers module
        self.driver_mgr.register_folder(os.path.dirname(drivers.__file__), drivers.__package__)

    def start(self):
        """Start the communicator and establishing all the connections

        Raises:
            CommError: If any error encountered while starting up
        """
        self.conn_manager.start()
        log.info(f"Communicator is started for local endpoint: {self.local_endpoint.name}")

    def stop(self):
        """Stop the communicator and shutdown all the connections

        Raises:
            CommError: If any error encountered while shutting down
        """
        self.conn_manager.stop()
        log.info(f"Communicator is stopped for local endpoint: {self.local_endpoint.name}")

    def register_monitor(self, monitor: EndpointMonitor):
        """Register a monitor for endpoint lifecycle changes

        This monitor is notified for any state changes of all the endpoints.
        Multiple monitors can be registered.

        Args:
            monitor: The class that receives the endpoint state change notification

        Raises:
            CommError: If any error happens while sending the request
        """
        self.conn_manager.add_endpoint_monitor(monitor)

    def get_endpoint(self, name: str) -> Optional[Endpoint]:
        """Find endpoint by name

        Args:
            name: Endpoint name

        Returns:
            The endpoint if found. None if not found

        """
        return self.conn_manager.find_endpoint(name)

    def send(self, endpoint: Endpoint, app_id: int, message: Message):
        """Send a message to endpoint for app_id, no response is expected

        Args:
            endpoint: An endpoint to send the request to
            app_id: Application ID
            message: Message to send

        Raises:
            CommError: If any error happens while sending the data
        """

        self.conn_manager.send_message(endpoint, app_id, message.headers, message.payload)

    def register_message_receiver(self, app_id: int, receiver: MessageReceiver):
        """Register a receiver to process FCI message for the app

        Args:
            app_id: Application ID
            receiver: The class to process the message

        Raises:
            CommError: If duplicate endpoint/app or responder is of wrong type
        """

        self.conn_manager.register_message_receiver(app_id, receiver)

    def add_connector(self, url: str, mode: Mode) -> str:
        """Load a connector. The driver is selected based on the URL

        Args:
            url: The url to listen on or connect to, like "https://0:443". Use 0 for empty host
            mode: Active for connecting, Passive for listening

        Returns:
            A handle that can be used to delete connector

        Raises:
            CommError: If any errors
        """

        return self._load_driver(url, mode)

    def start_listener(self, scheme: str, resources: dict) -> (str, str):
        """Add and start a connector in passive mode on an address selected by the driver.

        Args:
            scheme: Connection scheme, e.g. http, https
            resources: User specified resources like host and port ranges

        Returns:
            A tuple with connector handle and connect url

        Raises:
            CommError: If any errors like invalid host or port not available
        """

        driver_class = self.driver_mgr.find_driver_class(scheme)
        if not driver_class:
            raise CommError(CommError.NOT_SUPPORTED, f"No driver found for scheme {scheme}")

        connect_url, listening_url = driver_class.get_urls(scheme, resources)
        params = Driver.parse_url(listening_url)

        handle = self.add_connector_advanced(driver_class(), Mode.PASSIVE, params, True)

        return handle, connect_url

    def add_connector_advanced(self, driver: Driver, mode: Mode, params: dict, start: bool = False) -> str:
        """Add a connector using a specific driver instance.

        Args:
            driver: A transport driver instance
            mode: Active or passive
            params: Driver parameters
            start: Start the connector if true

        Returns:
            A handle that can be used to delete the connector

        Raises:
            CommError: If any errors
        """

        handle = self._update_conn_parameters(driver, params, mode)
        if not start:
            return handle

        listener = None
        for connector in self.conn_manager.connectors:
            if handle == connector.handle:
                listener = connector
                break

        if not listener:
            log.info(f"Connector {driver.get_name()}:{handle} is not found")
            return ""

        self.conn_manager.start_connector(listener)

        return handle

    def remove_connector(self, handle: str):
        """Remove the connector

        Args:
            handle: The connector handle

        Raises:
            CommError: If any errors
        """
        self.conn_manager.remove_connector(handle)

    # Internal methods

    def _update_conn_parameters(self, driver: Driver, params: dict, mode: Mode):

        if self.local_endpoint.conn_props:
            params.update(self.local_endpoint.conn_props)

        return self.conn_manager.add_connector(driver, params, mode)

    def _load_driver(self, url: str, mode: Mode) -> str:

        driver_class = self.driver_mgr.find_driver_class(url)
        if not driver_class:
            raise CommError(CommError.NOT_SUPPORTED, f"No driver found for URL {url}")

        params = Driver.parse_url(url)
        return self._update_conn_parameters(driver_class(), params, mode)
