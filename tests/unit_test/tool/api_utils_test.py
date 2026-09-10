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

    assert 29.0 < sess.try_connect.call_args.args[0] <= 30.0
    assert sess.try_connect.call_args.kwargs["connect_timeout"] == 10.0


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


@pytest.mark.parametrize("ready", [False, True])
def test_wait_for_system_start_bounds_blocking_session_cleanup(monkeypatch, ready):
    from nvflare.fuel.flare_api.api_spec import NoConnection
    from nvflare.tool.api_utils import SystemStartTimeout, wait_for_system_start

    monkeypatch.setattr(cli_output, "_output_format", "txt")
    release_close = threading.Event()
    close_started = threading.Event()
    close_finished = threading.Event()
    sys_info = MagicMock(client_info=[ClientInfo("site-1", None)])

    class BlockingCloseSession:
        api = MagicMock()

        def __init__(self, **kwargs):
            pass

        def try_connect(self, timeout, *, connect_timeout=None):
            if not ready:
                raise NoConnection("server is not reachable")

        def get_system_info(self):
            return sys_info

        def close(self):
            close_started.set()
            release_close.wait(2)
            close_finished.set()

    start = time.monotonic()
    try:
        with patch("nvflare.tool.api_utils.Session", BlockingCloseSession):
            kwargs = dict(second_to_wait=0, timeout_in_sec=0.05, poll_interval=0, conn_timeout=0.01)
            if ready:
                assert wait_for_system_start(1, "/tmp/prod", **kwargs) is sys_info
            else:
                with pytest.raises(SystemStartTimeout, match="session cleanup did not finish"):
                    wait_for_system_start(1, "/tmp/prod", **kwargs)
        assert time.monotonic() - start < 0.5
        assert close_started.is_set()
    finally:
        release_close.set()
        assert close_finished.wait(1)


@pytest.mark.parametrize("failure", ["stalled_login", "retry_sleep", "stalled_command_list"])
def test_readiness_deadline_bounds_real_admin_login(monkeypatch, failure):
    import json
    from types import SimpleNamespace

    from nvflare.fuel.flare_api.flare_api import Session
    from nvflare.fuel.hci.client.api import AdminAPI
    from nvflare.fuel.hci.client.api_spec import AdminConfigKey
    from nvflare.tool.api_utils import SystemStartTimeout, wait_for_system_start

    api = AdminAPI(
        admin_config={
            AdminConfigKey.CA_CERT: "ca.pem",
            AdminConfigKey.CLIENT_CERT: "client.pem",
            AdminConfigKey.CLIENT_KEY: "client.key",
        },
        user_name="admin",
        cmd_modules=[],
    )
    session = Session.__new__(Session)
    session.api = api
    request_budgets = []

    def stalled_request(**kwargs):
        if not api.in_logout:
            request_budgets.append(kwargs["timeout"])
            if failure == "stalled_command_list" and len(request_budgets) == 1:
                time.sleep(0.02)
                return SimpleNamespace(payload=json.dumps({"data": [{"type": "string", "data": "OK"}]}))
            if failure != "retry_sleep":
                time.sleep(kwargs["timeout"])
        return SimpleNamespace(payload=None)

    api.cell = MagicMock()
    api.cell.send_request.side_effect = stalled_request
    # Transport is already established; retain the real Session.try_connect,
    # AdminAPI.login, retry loop, and command path down to the network boundary.
    monkeypatch.setattr(api, "connect", lambda timeout: None)
    monkeypatch.setattr("nvflare.fuel.hci.client.api.IdentityAsserter", MagicMock())
    monkeypatch.setattr("nvflare.tool.api_utils.Session", lambda **kwargs: session)
    start = time.monotonic()
    with pytest.raises(SystemStartTimeout, match="admin login deadline reached"):
        wait_for_system_start(1, "/tmp/prod", second_to_wait=0, timeout_in_sec=0.1, poll_interval=0)
    assert time.monotonic() - start < 1.0
    assert len(request_budgets) == (2 if failure == "stalled_command_list" else 1)
    assert 0 < request_budgets[0] <= 0.1
    if failure == "stalled_command_list":
        assert 0 < request_budgets[1] < request_budgets[0]
    assert api._login_deadline is None


def test_readiness_rejects_status_received_after_deadline(monkeypatch):
    from nvflare.tool.api_utils import SystemStartTimeout, wait_for_system_start

    sess = MagicMock()

    def late_status():
        time.sleep(0.05)
        return MagicMock(client_info=[ClientInfo("site-1", None)])

    sess.get_system_info.side_effect = late_status
    monkeypatch.setattr("nvflare.tool.api_utils.Session", lambda **kwargs: sess)
    with pytest.raises(SystemStartTimeout, match="while checking system status"):
        wait_for_system_start(1, "/tmp/prod", second_to_wait=0, timeout_in_sec=0.02)


def test_slow_transport_leaves_outer_readiness_budget_for_real_login(monkeypatch):
    import json
    from types import SimpleNamespace

    from nvflare.fuel.flare_api import flare_api
    from nvflare.fuel.hci.client import api as admin_api
    from nvflare.fuel.hci.client.api_spec import AdminConfigKey
    from nvflare.tool import api_utils

    # Advance a virtual clock at the transport/network boundaries while retaining
    # the real Session, login requests, and login result processing.
    clock = [100.0]

    def advance(seconds):
        clock[0] += seconds

    timer = SimpleNamespace(monotonic=lambda: clock[0], time=lambda: clock[0], sleep=advance)
    for module in (api_utils, flare_api, admin_api):
        monkeypatch.setattr(module, "time", timer)
    api = admin_api.AdminAPI(
        admin_config={
            AdminConfigKey.CA_CERT: "ca.pem",
            AdminConfigKey.CLIENT_CERT: "client.pem",
            AdminConfigKey.CLIENT_KEY: "client.key",
        },
        user_name="admin",
        cmd_modules=[],
    )
    session = flare_api.Session.__new__(flare_api.Session)
    session.api = api
    requests = []

    def connect(timeout):
        assert timeout == 10.0
        advance(9.0)

    def respond(**kwargs):
        if api.in_logout:
            return SimpleNamespace(payload=None)
        requests.append(kwargs["timeout"])
        if len(requests) == 1:
            # Login needs more than the one second left from transport's cap.
            advance(min(2.0, kwargs["timeout"]))
            if kwargs["timeout"] < 2.0:
                return SimpleNamespace(payload=None)
            data = [{"type": "string", "data": "OK"}]
        else:
            data = [{"type": "table", "rows": []}]
        return SimpleNamespace(payload=json.dumps({"data": data}))

    api.cell = MagicMock()
    api.cell.send_request.side_effect = respond
    sys_info = MagicMock(client_info=[ClientInfo("site-1", None)])
    monkeypatch.setattr(api, "connect", connect)
    monkeypatch.setattr(admin_api, "IdentityAsserter", MagicMock())
    monkeypatch.setattr(session, "get_system_info", lambda: sys_info)
    monkeypatch.setattr(api_utils, "Session", lambda **kwargs: session)

    assert api_utils.wait_for_system_start(1, "/tmp/prod", second_to_wait=0, timeout_in_sec=30) is sys_info
    assert requests == [5.0, 5.0]
    assert clock[0] == 111.0
