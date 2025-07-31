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
import re
import sys
from typing import Any, Type

from nvflare.edge.simulation.device_task_processor import DeviceTaskProcessor
from nvflare.fuel.utils.validation_utils import check_positive_int, check_positive_number, check_str

VAR_PATTERN = re.compile(r"\{(.*?)}")


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
        self.job_name = None
        self.get_job_timeout = None
        self.processor = None
        self.endpoint = None
        self.num_devices = 100
        self.num_workers = 10
        self.processor_class = None
        self.processor_args = None
        self.parse(config_file)

    def get_processor(self, variables: dict = None) -> DeviceTaskProcessor:

        if self.processor_args:
            args = self._variable_substitution(self.processor_args, variables)
        else:
            args = {}

        return self.processor_class(**args)

    def get_endpoint(self):
        return self.endpoint

    def get_num_devices(self):
        return self.num_devices

    def get_num_workers(self):
        return self.num_workers

    def get_job_name(self):
        return self.job_name

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

        self.processor_args = processor_config.get("args", {})
        self.processor_class = load_class(path)
        if not issubclass(self.processor_class, DeviceTaskProcessor):
            raise TypeError(f"Processor {path} is not a subclass of DeviceTaskProcessor")

        self.endpoint = config.get("endpoint", None)
        if self.endpoint is not None:
            check_str("endpoint", self.endpoint)

        self.job_name = config.get("job_name", None)
        check_str("job_name", self.job_name)

        n = config.get("num_devices", None)
        if n:
            check_positive_int("num_devices", n)
            self.num_devices = n

        n = config.get("num_workers", None)
        if n:
            check_positive_int("num_workers", n)
            self.num_workers = n

        n = config.get("get_job_timeout", 60.0)
        check_positive_number("get_job_timeout", n)
        self.get_job_timeout = n

    def _variable_substitution(self, args: Any, variables: dict) -> Any:
        if isinstance(args, dict):
            return {k: self._variable_substitution(v, variables) for k, v in args.items()}
        elif isinstance(args, list):
            return [self._variable_substitution(v, variables) for v in args]
        elif isinstance(args, str):
            result = args
            offset = 0
            for i, match in enumerate(VAR_PATTERN.finditer(result)):
                start, end = match.span()
                start += offset
                end += offset
                var = match.group(1)
                if var in variables:
                    var_value = variables.get(var)
                    result = result[:start] + var_value + result[end:]
                    offset += len(var_value) - (end - start)

            return result
        else:
            return args
