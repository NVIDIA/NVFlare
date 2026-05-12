# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

import inspect
import re
from functools import wraps

type_pattern_mapping = {
    "server": r"^(([a-zA-Z0-9]|[a-zA-Z0-9][a-zA-Z0-9\-]*[a-zA-Z0-9])\.)*([A-Za-z0-9]|[A-Za-z0-9][A-Za-z0-9\-]*[A-Za-z0-9])$",
    "host_name": r"^(([a-zA-Z0-9]|[a-zA-Z0-9][a-zA-Z0-9\-]*[a-zA-Z0-9])\.)*([A-Za-z0-9]|[A-Za-z0-9][A-Za-z0-9\-]*[A-Za-z0-9])$",
    "client": r"^[A-Za-z0-9-_]+$",
    "job_name": r"^[A-Za-z0-9][A-Za-z0-9._-]*$",
    "relay": r"^[A-Za-z0-9-_]+$",
    "admin": r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}$",
    "email": r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}$",
    "org": r"^[A-Za-z0-9_]+$",
    "simple_name": r"^[A-Za-z0-9_]+$",
    "study": r"^[a-z0-9](?:[a-z0-9_-]{0,61}[a-z0-9])?$",
    "site": r"^[A-Za-z0-9-_]+$",
}


def name_check(name: str, entity_type: str):
    regex_pattern = type_pattern_mapping.get(entity_type)
    if regex_pattern is None:
        return True, "entity_type={} not defined, unable to check name={}.".format(entity_type, name)
    if re.match(regex_pattern, name):
        return False, "name={} passed on regex_pattern={} check".format(name, regex_pattern)
    else:
        return True, "name={} is ill-formatted for entity_type={} based on regex_pattern={}".format(
            name, entity_type, regex_pattern
        )


def check_job_id(job_id: str):
    if not isinstance(job_id, str) or not job_id:
        raise ValueError("job_id must be a non-empty string")
    invalid, message = name_check(job_id, "job_name")
    if invalid:
        raise ValueError(f"invalid job_id '{job_id}': {message}")


def check_job_app_name(app_name: str):
    if not isinstance(app_name, str) or not app_name:
        raise ValueError("job app name must be a non-empty string")
    invalid, message = name_check(app_name, "job_name")
    if invalid:
        raise ValueError(f"invalid job app name '{app_name}': {message}")


def validate_class_methods_args(cls):
    for name, method in inspect.getmembers(cls, inspect.isfunction):
        if name != "__init_subclass__":
            setattr(cls, name, validate_args(method))
    return cls


def validate_args(method):
    signature = inspect.signature(method)

    @wraps(method)
    def wrapper(*args, **kwargs):
        bound_arguments = signature.bind(*args, **kwargs)
        for name, value in bound_arguments.arguments.items():
            annotation = signature.parameters[name].annotation
            if not (annotation is inspect.Signature.empty or isinstance(value, annotation)):
                raise TypeError(
                    "argument '{}' of {} must be {} but got {}".format(name, method, annotation, type(value))
                )
        return method(*args, **kwargs)

    return wrapper
