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

from nvflare.fuel.utils.url_utils import make_url


class TestUrlUtils:
    @pytest.mark.parametrize(
        "scheme, address, secure, expected",
        [
            ("tcp", "xyz.com", False, "tcp://xyz.com"),
            ("tcp", "xyz.com:1234", False, "tcp://xyz.com:1234"),
            ("tcp", "xyz.com:1234", True, "stcp://xyz.com:1234"),
            ("grpc", "xyz.com", False, "grpc://xyz.com"),
            ("grpc", "xyz.com:1234", False, "grpc://xyz.com:1234"),
            ("grpc", "xyz.com:1234", True, "grpcs://xyz.com:1234"),
            ("http", "xyz.com", False, "http://xyz.com"),
            ("http", "xyz.com:1234", False, "http://xyz.com:1234"),
            ("http", "xyz.com:1234", True, "https://xyz.com:1234"),
            ("tcp", ("xyz.com",), False, "tcp://xyz.com"),
            ("tcp", ("xyz.com", 1234), False, "tcp://xyz.com:1234"),
            ("tcp", ["xyz.com"], False, "tcp://xyz.com"),
            ("tcp", ["xyz.com", 1234], False, "tcp://xyz.com:1234"),
            ("tcp", {"host": "xyz.com"}, False, "tcp://xyz.com"),
            ("tcp", {"host": "xyz.com", "port": 1234}, False, "tcp://xyz.com:1234"),
            ("stcp", {"host": "xyz.com"}, False, "tcp://xyz.com"),
            ("https", {"host": "xyz.com"}, False, "http://xyz.com"),
            ("grpcs", {"host": "xyz.com"}, False, "grpc://xyz.com"),
            ("stcp", {"host": "xyz.com"}, True, "stcp://xyz.com"),
            ("https", {"host": "xyz.com"}, True, "https://xyz.com"),
            ("grpcs", {"host": "xyz.com"}, True, "grpcs://xyz.com"),
        ],
    )
    def test_make_url(self, scheme, address, secure, expected):
        result = make_url(scheme, address, secure)
        assert result == expected

    @pytest.mark.parametrize(
        "scheme, address, secure",
        [
            ("tcp", "", False),
            ("abc", "xyz.com:1234", False),
            ("tcp", 1234, True),
            ("grpc", [], False),
            ("grpc", (), False),
            ("grpc", {}, True),
            ("http", [1234], False),
            ("http", [1234, "xyz.com"], False),
            ("http", ["xyz.com", 1234, 22], True),
            ("http", (1234,), False),
            ("http", (1234, "xyz.com"), False),
            ("http", ("xyz.com", 1234, 22), True),
            ("tcp", {"hosts": "xyz.com"}, False),
            ("tcp", {"host": "xyz.com", "port": 1234, "extra": 2323}, False),
        ],
    )
    def test_make_url_error(self, scheme, address, secure):
        with pytest.raises(ValueError):
            make_url(scheme, address, secure)
