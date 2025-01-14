# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
from typing import Any

from nvflare.fuel.utils.validation_utils import check_str

from .data_bus import DataBus


def _scope_prop_key(scope_name: str, key: str):
    return f"{scope_name}::{key}"


def set_scope_property(scope_name: str, key: str, value: Any):
    """Save the specified property of the specified scope (globally).
    Args:
        scope_name: name of the scope
        key: key of the property to be saved
        value: value of property
    Returns: None
    """
    check_str("scope_name", scope_name)
    check_str("key", key)
    data_bus = DataBus()
    data_bus.put_data(_scope_prop_key(scope_name, key), value)


def get_scope_property(scope_name: str, key: str, default=None) -> Any:
    """Get the value of a specified property from the specified scope.
    Args:
        scope_name: name of the scope
        key: key of the scope
        default: value to return if property is not found
    Returns:
    """
    check_str("scope_name", scope_name)
    check_str("key", key)
    data_bus = DataBus()
    result = data_bus.get_data(_scope_prop_key(scope_name, key))
    if result is None:
        result = default
    return result
