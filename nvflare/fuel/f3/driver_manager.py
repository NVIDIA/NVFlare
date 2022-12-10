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

import copy
import traceback

from typing import Union
from nvflare.fuel.common.excepts import ConfigError
from nvflare.fuel.utils.class_utils import get_class
from nvflare.fuel.utils.config_service import ConfigService
from .constants import DriverUse, DriverRequirementKey, Visibility
from .driver import DriverSpec

_KEY_RESOURCES = "resources"
_KEY_INT_DRIVER = "internal_driver"
_KEY_EXT_DRIVER = "external_driver"
_KEY_NAME = "name"
_KEY_DRIVERS = "drivers"


class DriverManager:

    """
    Manages creation of drivers (listeners and connectors).

    Each Driver Class must implement class methods for:
    - get_name()
    - set_properties() to set additional properties
    - new_listener() to generate a listener object based on connection requirements
    - new_connector() to generate a connector object based on connection properties
    """

    DRIVERS = {
        "http": "nvflare.fuel.f3.drivers.http.HTTPDriver",
        "ipc": "nvflare.fuel.f3.drivers.ipc.IPCDriver",
    }

    driver_config_files = [
        "comm_config.json",
        "comm_config.json.default"
    ]

    def __init__(self):
        # set up default drivers
        self.driver_class_paths = copy.copy(self.DRIVERS)
        self.int_driver = "http"
        self.int_driver_resources = {
            "http": {
                "url": "http://localhost",
                "ports": ["any"]  # select a port randomly
            }
        }
        self.ext_driver = "http"
        self.ext_driver_resources = {}

        # load driver config if any
        config = None
        for file_name in self.driver_config_files:
            try:
                config = ConfigService.load_json(file_name)
            except FileNotFoundError:
                config = None

        if config:
            drivers_conf = config.get(_KEY_DRIVERS)
            if drivers_conf:
                if not isinstance(drivers_conf, dict):
                    raise ConfigError(f"'{_KEY_DRIVERS}' must be dict but got {type(drivers_conf)}")
                self.driver_class_paths.update(drivers_conf)

            int_driver_conf = self._validate_driver_config(config, _KEY_INT_DRIVER)
            if int_driver_conf:
                self.int_driver = int_driver_conf.get(_KEY_NAME)
                self.int_driver_resources = int_driver_conf.get(_KEY_RESOURCES)

            ext_driver_conf = self._validate_driver_config(config, _KEY_EXT_DRIVER)
            if ext_driver_conf:
                self.ext_driver = ext_driver_conf.get(_KEY_NAME)
                self.ext_driver_resources = ext_driver_conf.get(_KEY_RESOURCES)

        # load driver classes
        self.driver_classes = {}
        for name, class_path in self.driver_class_paths.items():
            self.driver_classes[name] = get_class(class_path)

    def _validate_driver_config(self, config: dict, key: str) -> Union[None, dict]:
        driver_config = config.get(key)
        if driver_config:
            if not isinstance(driver_config, dict):
                raise ConfigError(f"'{key}' must be dict but got {type(driver_config)}")
            driver_name = driver_config.get(_KEY_NAME)
            if not driver_name:
                raise ConfigError(f"missing '{_KEY_NAME}' in {key} config")
            if driver_name not in self.driver_class_paths:
                raise ConfigError(f"undefined driver '{driver_name}' specified in {key}")

            resources = driver_config.get(_KEY_RESOURCES)
            if resources:
                if not isinstance(resources, dict):
                    raise ConfigError(f"'{_KEY_RESOURCES}' in {key} must be dict but got {type(resources)}")
        return driver_config

    def _get_driver(self, active: bool, requirements):
        use = requirements.get(DriverRequirementKey.USE, DriverUse.BACKBONE)
        vis = requirements.get(DriverRequirementKey.VISIBILITY, Visibility.EXTERNAL)
        if use == DriverUse.BACKBONE:
            if vis == Visibility.EXTERNAL:
                driver_name = self.ext_driver
                resources = self.ext_driver_resources
            else:
                driver_name = self.int_driver
                resources = self.int_driver_resources
        else:
            driver_name = self.ext_driver
            resources = self.ext_driver_resources

        clazz = self.driver_classes.get(driver_name)
        if clazz is not None:
            try:
                return clazz(
                    active=active,
                    conn_requirements=requirements,
                    resources=resources
                )
            except:
                traceback.print_exc()
                return None
        return None

    def get_listener(
            self,
            requirements: dict) -> Union[None, DriverSpec]:
        """
        Try to get a listener that can satisfy the requirements specified in conn_requirements.
        The conn_requirements could specify very precise requirements like secure or not, scheme, domain, port;
        The conn_requirements could also only specify general requirements like connection type (ext or internal)

        Args:
            requirements: required connection properties

        Returns: tuple of: dict of conn props, a listener or None.
        The returned conn props dict must contain sufficient info for others to make a connection:
        secure mode or not, scheme (http, tcp, ipc, etc), domain name, port number, etc.

        We simply try each driver class to get the required listener, until we get a listener.
        """
        return self._get_driver(
            active=False,
            requirements=requirements,
        )

    def get_connector(
            self,
            requirements: dict) -> Union[None, DriverSpec]:
        """
        Try to get a Connector for the specified conn_requirements, which contains info for making connection.

        Args:
            requirements: dict that contains info for making a connection

        Returns: a connector or None

        We simply try each driver class to get a connector.
        """
        return self._get_driver(
            active=True,
            requirements=requirements,
        )
