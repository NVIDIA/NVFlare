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
import importlib
import inspect
import logging
import os
from typing import Optional, Type

from nvflare.fuel.f3.comm_error import CommError
from nvflare.fuel.f3.drivers.driver import Driver

log = logging.getLogger(__name__)


class DriverManager:
    """Transport driver manager"""

    def __init__(self):
        # scheme-<
        self.drivers = {}

    def register(self, driver_class: Type[Driver]):
        """Register a driver with Driver Manager

        Args:
            driver_class: Driver to be registered. Driver must be a subclass of Driver
        """

        if not inspect.isclass(driver_class):
            raise CommError(CommError.ERROR, f"Registrant must be class, not instance: {type(driver_class)}")

        if not issubclass(driver_class, Driver):
            raise CommError(CommError.ERROR, f"Class {driver_class.__name__} is not a transport driver")

        for scheme in driver_class.supported_transports():
            key = scheme.lower()
            if key in self.drivers:
                log.error(f"Driver for scheme {scheme} is already registered, ignored")
            else:
                self.drivers[key] = driver_class
                log.debug(f"Driver {driver_class.__name__} is registered for {scheme}")

    def register_folder(self, folder: str, package: str):
        """Scan the folder and register all drivers

        Args:
            folder: The folder to scan
            package: The root package for all the drivers
        """

        class_cache = set()

        for file_name in os.listdir(folder):
            if file_name != "__init__.py" and file_name[-3:] == ".py":
                module = package + "." + file_name[:-3]
                imported = importlib.import_module(module)
                for _, cls_obj in inspect.getmembers(imported, inspect.isclass):
                    if cls_obj.__name__ in class_cache:
                        continue
                    class_cache.add(cls_obj.__name__)

                    spec = inspect.getfullargspec(cls_obj.__init__)
                    # classes who are abstract or take extra args in __init__ can't be auto-registered
                    if issubclass(cls_obj, Driver) and not inspect.isabstract(cls_obj) and len(spec.args) == 1:
                        self.register(cls_obj)

    def find_driver_class(self, scheme_or_url: str) -> Optional[Type[Driver]]:
        """Find the driver class based on scheme or URL

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
