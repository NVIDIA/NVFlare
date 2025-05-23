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
import pytest

from nvflare.apis.fl_constant import ConnectionSecurity
from nvflare.fuel.f3.comm_config_utils import requires_secure_connection
from nvflare.fuel.f3.drivers.driver_params import DriverParams

CS = DriverParams.CONNECTION_SECURITY.value
S = DriverParams.SECURE.value
IS = ConnectionSecurity.CLEAR
T = ConnectionSecurity.TLS
M = ConnectionSecurity.MTLS


class TestCommConfigUtils:
    @pytest.mark.parametrize(
        "resources, expected",
        [
            ({}, False),
            ({"x": 1, "y": 2}, False),
            ({S: True}, True),
            ({S: False}, False),
            ({CS: IS}, False),
            ({CS: T}, True),
            ({CS: M}, True),
            ({CS: M, S: False}, True),
            ({CS: M, S: True}, True),
            ({CS: T, S: False}, True),
            ({CS: T, S: True}, True),
            ({CS: IS, S: False}, False),
            ({CS: IS, S: True}, False),
        ],
    )
    def test_requires_secure_connection(self, resources, expected):
        result = requires_secure_connection(resources)
        assert result == expected
