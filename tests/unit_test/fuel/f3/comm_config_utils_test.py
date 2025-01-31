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

from nvflare.fuel.f3.comm_config_utils import requires_secure_connection


class TestCommConfigUtils:

    @pytest.mark.parametrize(
        "resources, expected",
        [
            ({}, False),
            ({"x": 1, "y": 2}, False),
            ({"secure": True}, True),
            ({"secure": False}, False),
            ({"connection_security": "insecure"}, False),
            ({"connection_security": "tls"}, True),
            ({"connection_security": "mtls"}, True),
            ({"connection_security": "mtls", "secure": False}, True),
            ({"connection_security": "mtls", "secure": True}, True),
            ({"connection_security": "tls", "secure": False}, True),
            ({"connection_security": "tls", "secure": True}, True),
            ({"connection_security": "insecure", "secure": False}, False),
            ({"connection_security": "insecure", "secure": True}, False),
        ],
    )
    def test_requires_secure_connection(self, resources, expected):
        result = requires_secure_connection(resources)
        assert result == expected
