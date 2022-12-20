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
import importlib
import inspect
import logging
import os
from typing import Union, Type, Optional

from nvflare.fuel.f3.drivers.driver import Driver

log = logging.getLogger(__name__)


class DriverManager:
    """Transport driver manager"""

    def __init__(self):
        # scheme-<
        self.drivers = {}

    def register(self, driver: Union[Driver, Type[Driver]]):
        """Register a driver with Driver Manager

        Args:
            driver: Driver to be registered. Driver can be either type or instance
        """

        if inspect.isclass(driver):
            cls = driver
            instance = driver()
        else:
            cls = driver.__class__
            instance = driver

        if not isinstance(instance, Driver):
            log.error(f"Class {cls.__name__} is not a transport driver, ignored")
            return

        for scheme in cls.supported_transports():
            key = scheme.lower()
            if key in self.drivers:
                log.error(f"Driver for scheme {scheme} is already registered, ignored")
            else:
                self.drivers[key] = instance
                log.debug(f"Driver {instance.get_name()} is registered for {scheme}")

    def register_folder(self, folder: str, package: str):
        """Scan the folder and register all drivers

        Args:
            folder: The folder to scan
            package: The root package for all the drivers
        """
        for file_name in os.listdir(folder):
            if file_name != "__init__.py" and file_name[-3:] == ".py":
                module = package + "." + file_name[:-3]
                imported = importlib.import_module(module)
                for _, cls_obj in inspect.getmembers(imported, inspect.isclass):
                    spec = inspect.getfullargspec(cls_obj.__init__)
                    # classes who are abstract or take extra args in __init__ can't be auto-registered
                    if issubclass(cls_obj, Driver) and not inspect.isabstract(cls_obj) and len(spec.args) == 1:
                        self.register(cls_obj)

    def find_driver(self, scheme_or_url: str) -> Optional[Driver]:
        """Find the driver instance based on scheme or URL

        Args:
            scheme_or_url: The scheme or the url

        Returns:
            The driver instance or None if not found
        """
        index = scheme_or_url.find(":")
        if index > 0:
            scheme = scheme_or_url[0:index]
        else:
            scheme = scheme_or_url

        return self.drivers.get(scheme.lower())
