# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

from nvflare.tool.cli_session import new_cli_session


def test_new_cli_session_delegates_to_secure_session_factory():
    from unittest.mock import MagicMock, patch

    fake_sess = MagicMock()
    with patch("nvflare.tool.cli_session.new_secure_session", return_value=fake_sess) as new_secure:
        returned = new_cli_session("user", "/tmp/startup", timeout=2.5, study="default")

    new_secure.assert_called_once_with(
        username="user",
        startup_kit_location="/tmp/startup",
        debug=False,
        study="default",
        timeout=2.5,
        auto_login_max_tries=1,
    )
    assert returned is fake_sess
