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
from nvflare.fuel.f3.drivers.driver_params import DriverParams
from nvflare.fuel.f3.drivers.net_utils import encode_url, parse_url


class TestNetUtils:
    def test_encode_url(self):

        params = {
            DriverParams.SCHEME.value: "tcp",
            DriverParams.HOST.value: "flare.test.com",
            DriverParams.PORT.value: 1234,
            "b": "test value",
            "a": 123,
            "r": False,
        }

        url = encode_url(params)
        assert url == "tcp://flare.test.com:1234?b=test+value&a=123&r=False"

    def test_parse_url(self):
        url = "grpc://test.com:8002?a=123&b=test"
        params = parse_url(url)
        assert params.get(DriverParams.URL) == url
        assert int(params.get(DriverParams.PORT)) == 8002
        assert params.get("a") == "123"
        assert params.get("b") == "test"
