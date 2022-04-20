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

from nvflare.apis.utils.format_check import name_check


class TestNameCheck:
    @pytest.mark.parametrize("name, err_value", [["bad***", True], ["bad?!", True], ["bad{}", True], ["good", False]])
    def test_org(self, name, err_value):

        err, reason = name_check(name, "org")
        assert err == err_value

    @pytest.mark.parametrize(
        "name, err_value",
        [
            ["localhost", False],
            ["mylocalmachine", False],
            ["bad_name", True],
            [" badname", True],
            ["bad_name.com", True],
            ["good-name.com", False],
        ],
    )
    def test_server(self, name, err_value):

        err, reason = name_check(name, "server")
        assert err == err_value

    @pytest.mark.parametrize("name, err_value", [["*.-", True], ["good-name", False], ["good_name", False]])
    def test_client(self, name, err_value):

        err, reason = name_check(name, "client")
        assert err == err_value

    @pytest.mark.parametrize(
        "name, err_value", [["bad_email*", True], ["bad_email", True], ["bad_email@", True], ["bad_email@123", True]]
    )
    def test_admin(self, name, err_value):

        err, reason = name_check(name, "admin")
        assert err == err_value
        err, reason = name_check(name, "email")
        assert err == err_value
