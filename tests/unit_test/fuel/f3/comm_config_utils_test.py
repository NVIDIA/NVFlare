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


class TestCommConfigUtils:

    @pytest.mark.parametrize(
        "resources, expected",
        [
            ({}, False),
            ({"x": 1, "y": 2}, False),
            ({DriverParams.SECURE.value: True}, True),
            ({DriverParams.SECURE.value: False}, False),
            ({DriverParams.CONNECTION_SECURITY.value: ConnectionSecurity.INSECURE}, False),
            ({DriverParams.CONNECTION_SECURITY.value: ConnectionSecurity.TLS}, True),
            ({DriverParams.CONNECTION_SECURITY.value: ConnectionSecurity.MTLS}, True),
            (
                    {
                        DriverParams.CONNECTION_SECURITY.value: ConnectionSecurity.MTLS
                    },
                    True
            ),
            (
                    {
                        DriverParams.CONNECTION_SECURITY.value: ConnectionSecurity.TLS
                    },
                    True
            ),
            (
                    {
                        DriverParams.CONNECTION_SECURITY.value: ConnectionSecurity.MTLS,
                        DriverParams.SECURE.value: False,
                    },
                    True
            ),
            (
                    {
                        DriverParams.CONNECTION_SECURITY.value: ConnectionSecurity.TLS,
                        DriverParams.SECURE.value: False,
                    },
                    True
            ),
            (
                    {
                        DriverParams.CONNECTION_SECURITY.value: ConnectionSecurity.MTLS,
                        DriverParams.SECURE.value: True,
                    },
                    True
            ),
            (
                    {
                        DriverParams.CONNECTION_SECURITY.value: ConnectionSecurity.TLS,
                        DriverParams.SECURE.value: True,
                    },
                    True
            ),
            (
                    {
                        DriverParams.CONNECTION_SECURITY.value: ConnectionSecurity.INSECURE,
                        DriverParams.SECURE.value: True,
                    },
                    False
            ),
            (
                    {
                        DriverParams.CONNECTION_SECURITY.value: ConnectionSecurity.INSECURE,
                    },
                    False
            ),
        ],
    )
    def test_requires_secure_connection(self, resources, expected):
        result = requires_secure_connection(resources)
        assert result == expected
