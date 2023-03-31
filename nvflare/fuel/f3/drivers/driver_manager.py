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
import sys
from typing import Optional, Type

from nvflare.fuel.f3.comm_error import CommError
from nvflare.fuel.f3.drivers.driver import Driver

log = logging.getLogger(__name__)


class DriverManager:
    """Transport driver manager"""

    def __init__(self):
        self.drivers = {}
        self.class_cache = set()

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

    def search_folder(self, folder: str, package: Optional[str]):
        """Search the folder recursively and register all drivers

        Args:
            folder: The folder to scan
            package: The root package for all the drivers. If none, the folder is the
            root of the packages
        """

        if package is None and folder not in sys.path:
            sys.path.append(folder)

        for root, dirs, files in os.walk(folder):
            for filename in files:
                if filename.endswith(".py"):
                    module = filename[:-3]
                    sub_folder = root[len(folder) :]
                    if sub_folder:
                        sub_folder = sub_folder.strip("/").replace("/", ".")

                    if sub_folder:
                        module = sub_folder + "." + module

                    if package:
                        module = package + "." + module

                    imported = importlib.import_module(module)
                    for _, cls_obj in inspect.getmembers(imported, inspect.isclass):
                        if cls_obj.__name__ in self.class_cache:
                            continue
                        self.class_cache.add(cls_obj.__name__)

                        if issubclass(cls_obj, Driver) and not inspect.isabstract(cls_obj):
                            spec = inspect.getfullargspec(cls_obj.__init__)
                            if len(spec.args) == 1:
                                self.register(cls_obj)
                            else:
                                # Can't handle argument in constructor
                                log.warning(f"Invalid driver, __init__ with extra arguments: {module}")

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
