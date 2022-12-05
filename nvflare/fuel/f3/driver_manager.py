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

from typing import List, Union, Dict
from nvflare.fuel.utils.class_utils import get_class
from .driver import DriverSpec


class DriverResource:

    def __init__(
            self,
            class_path: str,
            properties: dict=None
    ):
        """

        Args:
            class_path: class path of the driver
            properties: additional info to assist creating new listeners or connectors. This info could include
            - domain, port numbers, and/or URLs allowed for incoming connections
            - whether listeners of this class could be used for external or internal connections
        """
        self.class_path = class_path
        self.properties = properties


class ConnPropKey:

    URL = "url"
    CONNECTION_TYPE = "conn_type"   # internal (to family) or external


class DriverManager:

    """
    Manages creation of drivers (listeners and connectors).

    Each Driver class must implement class methods for:
    - set_properties() to set additional properties
    - new_listener() to generate a listener object based on connection requirements
    - new_connector() to generate a connector object based on connection properties
    """

    def __init__(self):
        self.driver_classes = {}  # path => class
        # TBD: load default driver classes

    def add_resources(self, resources: List[DriverResource]):
        for r in resources:
            clazz = self.driver_classes.get(r.class_path)
            if not clazz:
                clazz = get_class(r.class_path)
                self.driver_classes[r.class_path] = clazz
            clazz.set_properties(r.properties)

    def get_listener(
            self,
            conn_requirements: dict) -> (Union[None, Dict], Union[None, DriverSpec]):
        """
        Try to get a listener that can satisfy the requirements specified in conn_requirements.
        The conn_requirements could specify very precise requirements like secure or not, scheme, domain, port;
        The conn_requirements could also only specify general requirements like connection type (ext or internal)

        Args:
            conn_requirements: required connection properties

        Returns: tuple of: dict of conn props, a listener or None.
        The returned conn props dict must contain sufficient info for others to make a connection:
        secure mode or not, scheme (http, tcp, ipc, etc), domain name, port number, etc.

        We simply try each driver class to get the required listener, until we get a listener.
        """
        for _, clazz in self.driver_classes.items():
            listener = clazz.new_listener(conn_requirements)
            if listener:
                return listener
        return None

    def get_connector(self, conn_props: dict):
        """
        Try to get a Connector for the specified conn_props, which contains info for making connection.

        Args:
            conn_props: dict that contains info for making a connection

        Returns: a connector or None

        We simply try each driver class to get a connector.
        """
        for _, clazz in self.driver_classes.items():
            connector = clazz.new_connector(conn_props)
            if connector:
                return connector
        return None
