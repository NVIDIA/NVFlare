# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
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

from nvflare.fuel.sec import cve_checker


class TestCveChecker:

    @pytest.mark.parametrize(
        "version, expected_result",
        [
            ["OpenSSL 1.1.1q  5 Jul 2022", False],
            ["OpenSSL 1.1.2q  5 Jul 2022", True],
            ["OpenSSL 3.0.2 15 Mar 2022", True],
            ["OpenSSL 3.0.7  5 Oct 2022", False],
        ],
    )
    def test_openssl_versions(self, version,  expected_result):
        result = cve_checker.check_openssl(version)
        assert result == expected_result

