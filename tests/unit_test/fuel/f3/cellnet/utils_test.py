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
from nvflare.fuel.f3.cellnet.utils import format_size


class TestUtils:
    def test_format_size(self):

        assert format_size(0, False) == "0B"
        assert format_size(0, True) == "0B"

        assert format_size(1000, False) == "1KB"
        assert format_size(1000, True) == "1000B"
        assert format_size(1024, False) == "1KB"
        assert format_size(1024, True) == "1KiB"

        assert format_size(1000000, False) == "1MB"
        assert format_size(1000000, True) == "976.6KiB"
        assert format_size(1048576, False) == "1MB"
        assert format_size(1048576, True) == "1MiB"

        assert format_size(1000000000, False) == "1GB"
        assert format_size(1000000000, True) == "953.7MiB"
        assert format_size(1073741824, False) == "1.1GB"
        assert format_size(1073741824, True) == "1GiB"

        assert format_size(1000000000000, False) == "1TB"
        assert format_size(1000000000000, True) == "931.3GiB"
        assert format_size(1099511627776, False) == "1.1TB"
        assert format_size(1099511627776, True) == "1TiB"

        # Arbitrary large numbers
        assert format_size(10000000000000000000, False) == "10000PB"
        assert format_size(10000000000000000000, True) == "8881.8PiB"

        # Negative sizes
        assert format_size(-1099511627776, False) == "-1.1TB"

        # String value
        assert format_size("1099511627776", False) == "1.1TB"
