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

from unittest.mock import MagicMock, patch

import pytest

from nvflare.tool.cli_session import new_cli_session


def test_new_cli_session_returns_session_on_success():
    fake_sess = MagicMock()
    fake_sess.api = MagicMock()
    fake_sess.api.auto_login_max_tries = 15
    with patch("nvflare.tool.cli_session.Session", return_value=fake_sess) as mock_session:
        returned = new_cli_session("user", "/tmp/startup", timeout=2.5, study="default")

    mock_session.assert_called_once()
    assert fake_sess.api.auto_login_max_tries == 1
    fake_sess.api.set_command_timeout.assert_called_once_with(2.5)
    fake_sess.try_connect.assert_called_once_with(2.5)
    assert returned is fake_sess


def test_new_cli_session_closes_on_connect_failure():
    fake_sess = MagicMock()
    fake_sess.try_connect.side_effect = RuntimeError("boom")
    with patch("nvflare.tool.cli_session.Session", return_value=fake_sess):
        with pytest.raises(RuntimeError):
            new_cli_session("user", "/tmp/startup", timeout=1.0)

    fake_sess.close.assert_called_once()


def test_new_cli_session_tolerates_missing_api_helpers():
    fake_sess = MagicMock()
    fake_sess.api = None
    with patch("nvflare.tool.cli_session.Session", return_value=fake_sess):
        returned = new_cli_session("user", "/tmp/startup", timeout=3.0)

    fake_sess.try_connect.assert_called_once_with(3.0)
    assert returned is fake_sess
