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
import atexit
import logging
import os
import weakref
from typing import Optional

from nvflare.fuel.f3 import drivers
from nvflare.fuel.f3.comm_config import CommConfigurator
from nvflare.fuel.f3.comm_error import CommError
from nvflare.fuel.f3.drivers.driver import Driver
from nvflare.fuel.f3.drivers.driver_manager import DriverManager
from nvflare.fuel.f3.drivers.driver_params import DriverParams
from nvflare.fuel.f3.drivers.net_utils import parse_url
from nvflare.fuel.f3.endpoint import Endpoint, EndpointMonitor
from nvflare.fuel.f3.message import Message, MessageReceiver
from nvflare.fuel.f3.sfm.conn_manager import ConnManager, Mode
from nvflare.security.logging import secure_format_exception

log = logging.getLogger(__name__)
_running_instances = weakref.WeakSet()
driver_mgr = DriverManager()
driver_loaded = False


def load_comm_drivers():
    global driver_loaded

    # Load all the drivers in the drivers module
    driver_mgr.search_folder(os.path.dirname(drivers.__file__), drivers.__package__)

    # Load custom drivers
    driver_path = CommConfigurator().get_comm_driver_path(None)
    if not driver_path:
        return

    for path in driver_path.split(os.pathsep):
        log.debug(f"Custom driver folder {path} is searched")
        driver_mgr.search_folder(path, None)

    driver_loaded = True


class Communicator:
    """FCI (Flare Communication Interface) main communication API"""

    def __init__(self, local_endpoint: Endpoint):
        self.local_endpoint = local_endpoint
        self.monitors = []
        self.conn_manager = ConnManager(local_endpoint)
        self.stopped = False

    def start(self):
        """Start the communicator and establishing all the connections

        Raises:
            CommError: If any error encountered while starting up
        """
        self.conn_manager.start()
        log.debug(f"Communicator for local endpoint: {self.local_endpoint.name} is started")
        _running_instances.add(self)

    def stop(self):
        """Stop the communicator and shutdown all the connections

        Raises:
            CommError: If any error encountered while shutting down
        """
        if self.stopped:
            return

        self.conn_manager.stop()
        self.stopped = True
        try:
            _running_instances.remove(self)
        except KeyError as ex:
            log.error(
                f"Logical error, communicator {self.local_endpoint.name} is not started: {secure_format_exception(ex)}"
            )

        log.debug(f"Communicator endpoint: {self.local_endpoint.name} has stopped")

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

    def find_endpoint(self, name: str) -> Optional[Endpoint]:
        """Find endpoint by name

        Args:
            name: Endpoint name

        Returns:
            The endpoint if found. None if not found

        """
        return self.conn_manager.find_endpoint(name)

    def remove_endpoint(self, name: str):
        """Remove endpoint and close all the connections associated with it

        Args:
            name: Endpoint name

        """
        return self.conn_manager.remove_endpoint(name)

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
            receiver: The receiver to process the message

        Raises:
            CommError: If duplicate endpoint/app or receiver is of wrong type
        """

        self.conn_manager.register_message_receiver(app_id, receiver)

    def add_connector(self, url: str, mode: Mode, secure: bool = False) -> str:
        """Load a connector. The driver is selected based on the URL

        Args:
            url: The url to listen on or connect to, like "https://0:443". Use 0 for empty host
            mode: Active for connecting, Passive for listening
            secure: True if SSL is required.

        Returns:
            A handle that can be used to delete connector

        Raises:
            CommError: If any errors
        """

        if not driver_loaded:
            load_comm_drivers()

        driver_class = driver_mgr.find_driver_class(url)
        if not driver_class:
            raise CommError(CommError.NOT_SUPPORTED, f"No driver found for URL {url}")

        params = parse_url(url)
        return self.add_connector_advanced(driver_class(), mode, params, secure, False)

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

        if not driver_loaded:
            load_comm_drivers()

        driver_class = driver_mgr.find_driver_class(scheme)
        if not driver_class:
            raise CommError(CommError.NOT_SUPPORTED, f"No driver found for scheme {scheme}")

        connect_url, listening_url = driver_class.get_urls(scheme, resources)
        params = parse_url(listening_url)

        handle = self.add_connector_advanced(driver_class(), Mode.PASSIVE, params, False, True)

        return handle, connect_url

    def add_connector_advanced(
        self, driver: Driver, mode: Mode, params: dict, secure: bool, start: bool = False
    ) -> str:
        """Add a connector using a specific driver instance.

        Args:
            driver: A transport driver instance
            mode: Active or passive
            params: Driver parameters
            secure: SSL is required if true
            start: Start the connector if true

        Returns:
            A handle that can be used to delete the connector

        Raises:
            CommError: If any errors
        """

        if self.local_endpoint.conn_props:
            params.update(self.local_endpoint.conn_props)

        if secure:
            params[DriverParams.SECURE] = secure

        handle = self.conn_manager.add_connector(driver, params, mode)

        if not start:
            return handle

        connector = self.conn_manager.connectors.get(handle, None)

        if not connector:
            log.error(f"Connector {driver.get_name()}:{handle} is not found")
            raise CommError(CommError.ERROR, f"Logic error. Connector {driver.get_name()}:{handle} not found")

        self.conn_manager.start_connector(connector)

        return handle

    def remove_connector(self, handle: str):
        """Remove the connector

        Args:
            handle: The connector handle

        Raises:
            CommError: If any errors
        """
        self.conn_manager.remove_connector(handle)


def _exit_func():
    while _running_instances:
        c = next(iter(_running_instances))
        # This call will remove the entry from the set
        c.stop()
        log.debug(f"Communicator {c.local_endpoint.name} was left running, stopped on exit")


atexit.register(_exit_func)
