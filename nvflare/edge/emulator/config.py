# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
import builtins
import importlib
import json
import os
import sys
from typing import Type

from nvflare.edge.emulator.device_task_processor import DeviceTaskProcessor


def load_class(class_path) -> Type:

    try:
        if "." in class_path:
            module_name, class_name = class_path.rsplit(".", 1)
            module = importlib.import_module(module_name)
            return getattr(module, class_name)
        else:
            return getattr(builtins, class_path)
    except Exception as ex:
        raise TypeError(f"Can't load class {class_path}: {ex}")


class ConfigParser:

    def __init__(self, config_file: str):
        self.processor = None
        self.endpoint = None
        self.num_devices = 16
        self.capabilities = {}

        self.parse(config_file)

    def get_processor(self):
        return self.processor

    def get_endpoint(self):
        return self.endpoint

    def get_num_devices(self):
        return self.num_devices

    def get_capabilities(self):
        return self.capabilities

    def parse(self, config_file: str):
        with open(config_file, "r") as f:
            config = json.load(f)

        # Load processor
        processor_config = config.get("processor", None)
        if processor_config is None:
            raise ValueError("processor is not defined in config file")

        path = processor_config.get("python_path", None)
        if not path:
            # If no python_path defined, use the folder where the config file is
            path = os.path.abspath(os.path.dirname(config_file))
        sys.path.append(path)

        path = processor_config.get("path")
        if path is None:
            raise ValueError("path for processor is not defined in config file")

        processor_args = processor_config.get("args", {})
        process_class = load_class(path)
        if not issubclass(process_class, DeviceTaskProcessor):
            raise TypeError(f"Processor {path} is not a subclass of DeviceTaskProcessor")

        self.processor = process_class(**processor_args)

        self.endpoint = config.get("endpoint", None)

        n = config.get("num_devices", None)
        if n:
            self.num_devices = n

        self.capabilities = config.get("capabilities", {})
