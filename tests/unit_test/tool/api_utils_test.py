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

import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from nvflare.fuel.flare_api.api_spec import ClientInfo, ServerInfo
from nvflare.tool import cli_output


def test_wait_for_system_start_formats_client_list_and_ready_line(capsys, monkeypatch):
    from nvflare.tool.api_utils import wait_for_system_start

    monkeypatch.setattr(cli_output, "_output_format", "txt")
    sys_info = MagicMock(
        server_info=ServerInfo("running", None),
        client_info=[ClientInfo("site-1", None), ClientInfo("site-2", None)],
    )
    sess = MagicMock()
    sess.get_system_info.return_value = sys_info

    with patch("nvflare.tool.api_utils.Session", return_value=sess):
        assert (
            wait_for_system_start(
                2,
                "/tmp/prod",
                username="admin@nvidia.com",
                second_to_wait=0,
                timeout_in_sec=1,
                conn_timeout=0.1,
            )
            is sys_info
        )

    out = capsys.readouterr().out
    assert "Server info:" not in out
    assert "Client info" not in out
    assert "last_connect_time" not in out
    assert "trying to connect to server" not in out
    assert "Clients ready: 2/2 (site-1, site-2)" in out
    assert "\nReady to go.\n" in out


def test_wait_for_system_start_uses_ten_second_default_connection_budget(monkeypatch):
    from nvflare.tool.api_utils import wait_for_system_start

    monkeypatch.setattr(cli_output, "_output_format", "txt")
    sys_info = MagicMock(
        server_info=ServerInfo("running", None),
        client_info=[ClientInfo("site-1", None)],
    )
    sess = MagicMock()
    sess.get_system_info.return_value = sys_info

    with patch("nvflare.tool.api_utils.Session", return_value=sess):
        wait_for_system_start(1, "/tmp/prod", second_to_wait=0, timeout_in_sec=30)

    assert 9.0 < sess.try_connect.call_args.args[0] <= 10.0


def test_wait_for_system_start_reports_missing_expected_clients_concisely(capsys, monkeypatch):
    from nvflare.tool.api_utils import wait_for_system_start

    monkeypatch.setattr(cli_output, "_output_format", "txt")
    sys_info_waiting = MagicMock(
        server_info=ServerInfo("running", None),
        client_info=[ClientInfo("site-1", None)],
    )
    sys_info_ready = MagicMock(
        server_info=ServerInfo("running", None),
        client_info=[ClientInfo("site-1", None), ClientInfo("site-2", None)],
    )
    sess = MagicMock()
    sess.get_system_info.side_effect = [sys_info_waiting, sys_info_ready]

    with patch("nvflare.tool.api_utils.Session", return_value=sess):
        assert (
            wait_for_system_start(
                2,
                "/tmp/prod",
                username="admin@nvidia.com",
                second_to_wait=0,
                timeout_in_sec=1,
                poll_interval=0,
                conn_timeout=0.1,
                expected_clients=["site-1", "site-2"],
            )
            is sys_info_ready
        )

    out = capsys.readouterr().out
    assert "Waiting for clients: site-2 (1/2 ready)" in out
    assert "Clients ready: 2/2 (site-1, site-2)" in out


def test_wait_for_system_start_timeout_message_uses_expected_clients(monkeypatch):
    from nvflare.tool.api_utils import SystemStartTimeout, wait_for_system_start

    monkeypatch.setattr(cli_output, "_output_format", "txt")
    sys_info = MagicMock(
        server_info=ServerInfo("running", None),
        client_info=[ClientInfo("site-1", None)],
    )
    sess = MagicMock()
    sess.get_system_info.return_value = sys_info

    with patch("nvflare.tool.api_utils.Session", return_value=sess):
        with pytest.raises(SystemStartTimeout) as exc_info:
            wait_for_system_start(
                99,
                "/tmp/prod",
                username="admin@nvidia.com",
                second_to_wait=0,
                timeout_in_sec=0.01,
                poll_interval=0,
                conn_timeout=0.1,
                expected_clients=["site-1", "site-2"],
            )

    message = str(exc_info.value)
    assert "expected clients site-1, site-2" in message
    assert "99 clients" not in message


def test_wait_for_system_start_bounds_blocking_session_cleanup(monkeypatch):
    from nvflare.fuel.flare_api.api_spec import NoConnection
    from nvflare.tool.api_utils import SystemStartTimeout, wait_for_system_start

    monkeypatch.setattr(cli_output, "_output_format", "txt")
    release_close = threading.Event()
    close_started = threading.Event()

    class BlockingCloseSession:
        api = MagicMock()

        def __init__(self, **kwargs):
            pass

        def try_connect(self, timeout):
            raise NoConnection("server is not reachable")

        def close(self):
            close_started.set()
            release_close.wait(2)

    start = time.monotonic()
    try:
        with patch("nvflare.tool.api_utils.Session", BlockingCloseSession):
            with pytest.raises(SystemStartTimeout, match="session cleanup did not finish"):
                wait_for_system_start(
                    1,
                    "/tmp/prod",
                    second_to_wait=0,
                    timeout_in_sec=0.05,
                    poll_interval=0,
                    conn_timeout=0.01,
                )
        assert time.monotonic() - start < 0.5
        assert close_started.is_set()
    finally:
        release_close.set()
