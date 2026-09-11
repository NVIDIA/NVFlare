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
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from nvflare.fuel.common.excepts import ConfigError
from nvflare.fuel.flare_api.api_spec import ClientInfo, ServerInfo
from nvflare.fuel.hci.client.api import AdminCertAcquisitionError
from nvflare.fuel.sec.admin_cert_provider import AdminCertProviderRequestError
from nvflare.tool import cli_output


def _synchronize_probe_start(monkeypatch, entered):
    """Exclude worker scheduling/setup from the blocked-operation timeout test."""
    from nvflare.tool import api_utils

    started_at = None

    def elapsed():
        return 0.0 if started_at is None else time.monotonic() - started_at

    class SynchronizedThread(threading.Thread):
        def start(self):
            nonlocal started_at
            super().start()
            assert entered.wait(5), "readiness worker did not reach the tested phase"
            started_at = time.monotonic()

    # The deadline is computed before Thread.start(), so freeze only the probe's
    # clock until phase entry. Transport and streaming retain their real clocks.
    monkeypatch.setattr(api_utils, "time", SimpleNamespace(monotonic=elapsed, sleep=time.sleep))
    monkeypatch.setattr(api_utils, "threading", SimpleNamespace(Thread=SynchronizedThread, Event=threading.Event))
    return elapsed


@pytest.mark.parametrize("connection_options", [{}, {"conn_timeout": 0.1}])
def test_wait_for_system_start_formats_client_list_and_ready_line(capsys, monkeypatch, connection_options):
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
                timeout_in_sec=30,
                **connection_options,
            )
            is sys_info
        )

    sess.try_connect.assert_called_once_with(connection_options.get("conn_timeout", 10.0))
    out = capsys.readouterr().out
    assert "Connecting and logging in to the admin server" in out
    assert "seconds remaining" in out
    assert "deadline" not in out
    assert "Server info:" not in out
    assert "Client info" not in out
    assert "last_connect_time" not in out
    assert "trying to connect to server" not in out
    assert "Clients ready: 2/2 (site-1, site-2)" in out
    assert "\nReady to go.\n" in out


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
                timeout_in_sec=0.05,
                poll_interval=0.1,
                conn_timeout=0.1,
                expected_clients=["site-1", "site-2"],
            )

    message = str(exc_info.value)
    assert "expected clients site-1, site-2" in message
    assert "99 clients" not in message


@pytest.mark.parametrize("state", ["unreachable", "partial", "ready", "renewal", "config"])
def test_wait_for_system_start_bounds_blocking_session_cleanup(monkeypatch, state):
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

        def try_connect(self, timeout):
            if state == "config":
                raise ConfigError("invalid communication configuration")
            if state == "renewal":
                raise AdminCertAcquisitionError("certificate service unavailable")
            if state == "unreachable":
                raise NoConnection("server is not reachable")

        def get_system_info(self):
            return sys_info

        def close(self):
            close_started.set()
            release_close.wait(5)
            close_finished.set()

    elapsed = _synchronize_probe_start(monkeypatch, close_started)
    try:
        with patch("nvflare.tool.api_utils.Session", BlockingCloseSession):
            kwargs = dict(second_to_wait=0, timeout_in_sec=0.05, poll_interval=0, conn_timeout=0.01)
            if state == "ready":
                assert wait_for_system_start(1, "/tmp/prod", **kwargs) is sys_info
            elif state == "config":
                kwargs["timeout_in_sec"] = 30
                with pytest.raises(ConfigError, match="invalid communication configuration"):
                    wait_for_system_start(2, "/tmp/prod", **kwargs)
            else:
                with pytest.raises(SystemStartTimeout, match="Could not confirm") as exc_info:
                    wait_for_system_start(2, "/tmp/prod", **kwargs)
                assert type(exc_info.value) is SystemStartTimeout
                assert "closing" not in str(exc_info.value)
                if state == "partial":
                    assert "Last observation: Waiting for clients: 1/2 ready" in str(exc_info.value)
                elif state == "renewal":
                    assert "Last observation: certificate service unavailable" in str(exc_info.value)
                else:
                    assert "Last observation: server is not reachable" in str(exc_info.value)
        assert elapsed() < 0.5
        assert close_started.wait(1)
    finally:
        release_close.set()
        assert close_finished.wait(1)


@pytest.mark.parametrize("phase", ["construction", "connect", "status"])
def test_readiness_bounds_blocked_operations_and_cleans_up_after_return(monkeypatch, phase):
    from nvflare.tool import api_utils

    entered, release, closed = threading.Event(), threading.Event(), threading.Event()
    session = MagicMock()
    info = MagicMock(client_info=[ClientInfo("site-1", None)])
    session.get_system_info.return_value = info
    session.close.side_effect = closed.set

    def block():
        entered.set()
        assert release.wait(5)
        return info

    def construct(**kwargs):
        if phase == "construction":
            block()
        return session

    if phase == "connect":
        session.try_connect.side_effect = lambda timeout: block()
    elif phase == "status":
        session.get_system_info.side_effect = block
    monkeypatch.setattr(api_utils, "Session", construct)
    elapsed = _synchronize_probe_start(monkeypatch, entered)
    try:
        with pytest.raises(api_utils.SystemStartTimeout):
            api_utils.wait_for_system_start(1, "/tmp/prod", second_to_wait=0, timeout_in_sec=0.1)
        assert elapsed() < 0.5
        assert entered.is_set()
        assert not closed.is_set()
    finally:
        release.set()
        assert closed.wait(1)
    if phase != "status":
        session.get_system_info.assert_not_called()
    if phase == "construction":
        session.try_connect.assert_not_called()
    session.close.assert_called_once()


def test_slow_transport_leaves_outer_readiness_budget_for_real_login(monkeypatch):
    import json

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
    monkeypatch.setattr(api_utils, "time", timer)
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

    closed = threading.Event()
    api.cell = MagicMock()
    api.cell.stop.side_effect = closed.set
    api.cell.send_request.side_effect = respond
    sys_info = MagicMock(client_info=[ClientInfo("site-1", None)])
    monkeypatch.setattr(api, "connect", connect)
    monkeypatch.setattr(admin_api, "IdentityAsserter", MagicMock())
    monkeypatch.setattr(session, "get_system_info", lambda: sys_info)
    monkeypatch.setattr(api_utils, "Session", lambda **kwargs: session)

    assert api_utils.wait_for_system_start(1, "/tmp/prod", second_to_wait=0, timeout_in_sec=30) is sys_info
    assert closed.wait(1)
    assert requests == [5.0, 5.0]
    assert clock[0] == 111.0


@pytest.mark.parametrize("parameter", ["timeout_in_sec", "conn_timeout", "poll_interval", "second_to_wait"])
@pytest.mark.parametrize("value", [-1, float("inf"), float("nan"), "30", None, True])
def test_readiness_rejects_invalid_time_settings_before_any_work(parameter, value):
    from nvflare.tool.api_utils import wait_for_system_start

    with patch("nvflare.tool.api_utils.Session") as session, patch("nvflare.tool.api_utils.time.sleep") as sleep:
        with pytest.raises(ValueError, match=parameter + " must be a finite"):
            wait_for_system_start(1, "/tmp/prod", **{parameter: value})
    session.assert_not_called()
    sleep.assert_not_called()


@pytest.mark.parametrize("parameter", ["timeout_in_sec", "conn_timeout"])
def test_readiness_rejects_zero_timeout(parameter):
    from nvflare.tool.api_utils import wait_for_system_start

    with patch("nvflare.tool.api_utils.Session") as session, patch("nvflare.tool.api_utils.time.sleep") as sleep:
        with pytest.raises(ValueError, match=parameter + " must be a finite positive"):
            wait_for_system_start(1, "/tmp/prod", **{parameter: 0})
    session.assert_not_called()
    sleep.assert_not_called()


@pytest.mark.parametrize("failure_source", ["connection", "status_request", "status_parsing"])
def test_readiness_retries_value_error_during_connection_and_status(failure_source):
    from nvflare.tool.api_utils import wait_for_system_start

    sys_info = MagicMock(client_info=[ClientInfo("site-1", None)])
    sessions = [MagicMock(), MagicMock()]
    closed = threading.Event()
    sessions[-1].close.side_effect = closed.set
    for session in sessions:
        session.get_system_info.return_value = sys_info
    parsing_results = [["site-1"], ["site-1"]]
    error = ValueError("incomplete status response")
    if failure_source == "connection":
        sessions[0].try_connect.side_effect = error
    elif failure_source == "status_request":
        sessions[0].get_system_info.side_effect = error
    else:
        parsing_results[0] = error

    with (
        patch("nvflare.tool.api_utils.Session", side_effect=sessions) as factory,
        patch("nvflare.tool.api_utils._client_names", side_effect=parsing_results),
    ):
        assert wait_for_system_start(1, "/tmp/prod", second_to_wait=0, poll_interval=0) is sys_info

    assert closed.wait(1)
    assert factory.call_count == 2
    for session in sessions:
        session.try_connect.assert_called_once()
        session.close.assert_called_once()


@pytest.mark.parametrize(
    "error_type", [AssertionError, ValueError, TypeError, FileNotFoundError, RuntimeError, ConfigError]
)
def test_readiness_does_not_retry_session_constructor_configuration_errors(error_type):
    from nvflare.tool.api_utils import wait_for_system_start

    error = error_type("invalid admin configuration")
    with patch("nvflare.tool.api_utils.Session", side_effect=error) as factory:
        with pytest.raises(ConfigError, match="invalid admin configuration") as exc_info:
            wait_for_system_start(1, "/tmp/prod", second_to_wait=0)
    assert "Cannot initialize the admin session" in str(exc_info.value)
    factory.assert_called_once()


@pytest.mark.parametrize("phase", ["authentication", "login_stream", "command_list_stream", "login_retry"])
def test_readiness_bounds_real_authentication_and_stream_waits(monkeypatch, phase):
    import json

    from nvflare.fuel.f3.cellnet.cell import Cell
    from nvflare.fuel.f3.cellnet.defs import ReturnCode
    from nvflare.fuel.f3.cellnet.utils import make_reply
    from nvflare.fuel.flare_api.flare_api import Session
    from nvflare.fuel.hci.client import api as admin_api
    from nvflare.fuel.hci.client.api_spec import AdminConfigKey
    from nvflare.tool import api_utils

    api = admin_api.AdminAPI(
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
    entered, release, closed = threading.Event(), threading.Event(), threading.Event()
    cell = MagicMock()
    cell.stop.side_effect = closed.set
    calls = []
    progress = []

    def send_request(**kwargs):
        if api.in_logout:
            return make_reply(ReturnCode.OK)
        calls.append(kwargs["timeout"])
        if phase == "authentication":
            entered.set()
            assert release.wait(5)
            return make_reply(ReturnCode.TARGET_UNREACHABLE)
        if phase == "login_retry":
            return make_reply(ReturnCode.COMM_ERROR)
        if phase == "command_list_stream" and len(calls) == 1:
            return SimpleNamespace(payload=json.dumps({"data": [{"type": "string", "data": "OK"}]}))
        return stream._send_one_request(**kwargs)

    cell.send_request.side_effect = send_request
    if phase == "authentication":
        # Keep Session.try_connect, AdminAPI.connect and Authenticator's real
        # challenge/retry loop; only replace the network endpoint.
        monkeypatch.setattr(admin_api, "Cell", lambda **kwargs: cell)
        monkeypatch.setattr(admin_api, "NetAgent", MagicMock())
        monkeypatch.setattr("nvflare.private.fed.authenticator._get_client_ip", lambda: "127.0.0.1")
    else:
        # Keep real login, command-list retrieval, _send_one_request,
        # _future_wait and conditional_wait. Send progress extends idle waits.
        api.cell = cell
        api.set_command_timeout(0.01)
        api.auto_login_max_tries = 1
        if phase == "login_retry":

            def retry_sleep(delay):
                assert delay == admin_api.AUTO_LOGIN_INTERVAL
                entered.set()
                assert release.wait(5)

            # Exercise the real retry loop with its sleep held until teardown.
            monkeypatch.setattr(admin_api, "time", SimpleNamespace(time=time.time, sleep=retry_sleep))
        else:
            monkeypatch.setattr(admin_api, "AUTO_LOGIN_INTERVAL", 0)
        monkeypatch.setattr(admin_api, "IdentityAsserter", MagicMock())
        stream = Cell.__new__(Cell)
        stream.logger, stream.requests_dict = MagicMock(), {}

        def get_progress():
            progress.append(1)
            if len(progress) == 2:
                entered.set()
            return len(progress)

        future = SimpleNamespace(error=None, waiter=release, get_progress=get_progress)

        def send_blob(**kwargs):
            return future

        stream.send_blob = send_blob

    monkeypatch.setattr(api_utils, "Session", lambda **kwargs: session)
    elapsed = _synchronize_probe_start(monkeypatch, entered)
    try:
        with pytest.raises(api_utils.SystemStartTimeout, match="Could not confirm"):
            api_utils.wait_for_system_start(1, "/tmp/prod", second_to_wait=0, timeout_in_sec=0.1)
        assert elapsed() < 0.5
        assert entered.is_set()
        assert not closed.is_set()
        if phase == "authentication":
            assert calls == [5.0]  # The existing transport wait can outlast readiness.
        elif phase != "login_retry":
            assert len(progress) >= 2  # Actual idle waits renewed while sending.
            future.error = RuntimeError("test stream ended")
    finally:
        release.set()
        assert closed.wait(2)
    cell.stop.assert_called_once()


@pytest.mark.parametrize("operation", ["wait_for_shutdown", "shutdown"])
def test_slow_successful_transport_is_not_evidence_of_shutdown(monkeypatch, operation):

    from nvflare.fuel.flare_api import flare_api
    from nvflare.fuel.hci.client.api import APIStatus, ResultKey
    from nvflare.tool import api_utils

    clock = [100.0]

    def advance(seconds):
        clock[0] += seconds

    session = flare_api.Session.__new__(flare_api.Session)
    session.api = MagicMock(closed=False)
    session.api.connect.side_effect = lambda timeout: advance(timeout + 0.1)
    session.api.login.return_value = {ResultKey.STATUS: APIStatus.SUCCESS}
    session.get_system_info = MagicMock(return_value=MagicMock(server_info=ServerInfo("running", None)))
    monkeypatch.setattr(flare_api, "time", SimpleNamespace(time=lambda: clock[0], sleep=advance))
    if operation == "wait_for_shutdown":
        session._new_poll_session = lambda: session
        with pytest.raises(TimeoutError, match="server did not stop"):
            session._wait_for_server_down(2)
        session.get_system_info.assert_called()
    else:
        session.list_jobs, session.shutdown = MagicMock(return_value=[]), MagicMock()
        monkeypatch.setattr(api_utils, "Session", lambda **kwargs: session)
        result = api_utils.shutdown_system("/tmp/prod", wait=False)
        assert result["server_reachable"] is True
        assert result["already_stopped"] is False
        session.shutdown.assert_called_once()
    session.api.login.assert_called()


@pytest.mark.parametrize(
    "missing", ["startup_directory", "startup_folder", "site_config", "admin_section", "admin_file", "download_dir"]
)
def test_real_session_configuration_errors_fail_without_retry(tmp_path, missing):
    import json

    from nvflare.fuel.flare_api import flare_api
    from nvflare.tool import api_utils

    if missing != "startup_directory":
        (tmp_path / "admin").mkdir()
    if missing not in ("startup_directory", "startup_folder"):
        (tmp_path / "admin" / "startup").mkdir()
    if missing in ("admin_section", "admin_file", "download_dir"):
        (tmp_path / "admin" / "local").mkdir()
    if missing in ("admin_section", "download_dir"):
        config = {} if missing == "admin_section" else {"admin": {"upload_dir": "transfer", "download_dir": None}}
        (tmp_path / "admin" / "startup" / "fed_admin.json").write_text(json.dumps(config))

    with patch.object(api_utils, "Session", wraps=flare_api.Session) as factory:
        started = time.monotonic()
        with pytest.raises(ConfigError, match="Cannot initialize the admin session"):
            api_utils.wait_for_system_start(1, str(tmp_path), second_to_wait=0, timeout_in_sec=30)
        assert time.monotonic() - started < 2
    factory.assert_called_once()


@pytest.mark.parametrize("failure_type", [ConnectionError, AdminCertProviderRequestError])
@pytest.mark.parametrize("phase", ["initial", "renewal"])
@pytest.mark.parametrize("recovers", [True, False])
def test_certificate_acquisition_failures_retry_with_real_session(tmp_path, monkeypatch, phase, recovers, failure_type):
    import json

    from nvflare.fuel.flare_api import flare_api
    from nvflare.fuel.hci.client import api as admin_api
    from nvflare.fuel.sec.admin_cert_provider import AdminCertFiles
    from nvflare.tool import api_utils

    startup = tmp_path / "admin" / "startup"
    startup.mkdir(parents=True)
    (tmp_path / "admin" / "local").mkdir()
    config = {
        "admin": {
            "ca_cert": "rootCA.pem",
            "upload_dir": "transfer",
            "download_dir": "transfer",
            "admin_cert_provider": {"provider": "step_ca", "provider_config": {}},
        }
    }
    (startup / "fed_admin.json").write_text(json.dumps(config))
    expired = AdminCertFiles(client_key="old.key", client_cert="old.pem", expires_at=1)
    renewed = AdminCertFiles(client_key="new.key", client_cert="new.pem", expires_at=9999999999)
    entered = threading.Event()
    requests, workers, sessions = [], set(), []
    failed_at = 2 if phase == "renewal" else 1

    def obtain(**kwargs):
        workers.add(threading.current_thread())
        requests.append(kwargs)
        if len(requests) >= 2:
            entered.set()
        if phase == "renewal" and len(requests) == 1:
            return expired
        if len(requests) == failed_at or not recovers:
            raise failure_type("certificate service temporarily unavailable")
        return renewed

    def connect_cell(api, timeout):
        api.cell = MagicMock()

    def construct(**kwargs):
        session = flare_api.Session(**kwargs)
        sessions.append(session)
        return session

    info = MagicMock(client_info=[ClientInfo("site-1", None)])
    monkeypatch.setattr(admin_api, "obtain_admin_cert_files", obtain)
    monkeypatch.setattr(admin_api.AdminAPI, "_connect_cell", connect_cell)
    monkeypatch.setattr(admin_api.AdminAPI, "login", lambda self: {"status": admin_api.APIStatus.SUCCESS})
    monkeypatch.setattr(admin_api.AdminAPI, "logout", lambda self: self.close())
    monkeypatch.setattr(flare_api.Session, "get_system_info", lambda self: info)
    monkeypatch.setattr(api_utils, "Session", construct)
    elapsed = _synchronize_probe_start(monkeypatch, entered)
    try:
        kwargs = dict(second_to_wait=0, timeout_in_sec=0.1, poll_interval=0.01, secure_mode=True)
        if recovers:
            assert api_utils.wait_for_system_start(1, str(tmp_path), **kwargs) is info
            assert len(requests) == failed_at + 1
        else:
            with pytest.raises(api_utils.SystemStartTimeout, match="certificate service temporarily unavailable"):
                api_utils.wait_for_system_start(1, str(tmp_path), **kwargs)
        assert elapsed() < 0.5
    finally:
        for worker in workers:
            worker.join(1)
            assert not worker.is_alive()
    assert len(requests) >= 2
    assert all(session.api.closed for session in sessions)


@pytest.mark.parametrize(
    "failure, message",
    [
        ("ca_url", "ca_url is required"),
        ("provisioner", "provisioner is required"),
        ("provider", "built-in provider name"),
        ("provider_import", "cannot load admin certificate provider"),
        ("step_missing", "cannot execute"),
        ("step_not_executable", "cannot execute"),
        ("ca_unreadable", "Is a directory"),
        ("unknown_provisioner", "step ca certificate failed with exit code 1"),
        ("invalid_ttl", "step ca certificate failed with exit code 1"),
    ],
)
def test_real_provider_configuration_errors_fail_without_retry(tmp_path, monkeypatch, failure, message):
    import json

    from nvflare.fuel.flare_api import flare_api
    from nvflare.fuel.sec import admin_cert_provider
    from nvflare.tool import api_utils

    startup = tmp_path / "admin" / "startup"
    startup.mkdir(parents=True)
    (tmp_path / "admin" / "local").mkdir()
    root_ca = startup / "rootCA.pem"
    if failure == "ca_unreadable":
        root_ca.mkdir()
    else:
        root_ca.write_text("unused: acquisition fails before certificate verification")
    step_bin = tmp_path / "step"
    if failure == "step_not_executable":
        step_bin.write_text("not executable")
        step_bin.chmod(0o600)
    provider_config = {"ca_url": "https://ca.example.com", "provisioner": "admin", "step_bin": str(step_bin)}
    if failure in ("ca_url", "provisioner"):
        del provider_config[failure]
    rejected = failure in ("unknown_provisioner", "invalid_ttl")
    invocation_log = tmp_path / "step-invocations.json"
    if rejected:
        import sys

        flag, value = (
            ("--provisioner", "unknown-provisioner") if failure == "unknown_provisioner" else ("--not-after", "0s")
        )
        provider_config["provisioner" if failure == "unknown_provisioner" else "cert_ttl"] = value
        # A real subprocess models step's indistinguishable exit-1 failures.
        # It records the argv so the test verifies the configured value reaches it.
        step_bin.write_text(
            f"#!{sys.executable}\n"
            "import json, sys\n"
            f"with open({str(invocation_log)!r}, 'w') as out: json.dump(sys.argv[1:], out)\n"
            f"print({f'rejected {flag}: {value}'!r}, file=sys.stderr)\n"
            "sys.exit(1)\n"
        )
        step_bin.chmod(0o700)
    provider = {"provider": "step_ca", "provider_config": provider_config}
    if failure == "provider":
        provider["provider"] = "invalid-provider"
    elif failure == "provider_import":
        provider["provider"] = "missing_nvflare_cert_provider:obtain"
    config = {
        "admin": {
            "ca_cert": str(root_ca),
            "upload_dir": "transfer",
            "download_dir": "transfer",
            "admin_cert_provider": provider,
        }
    }
    (startup / "fed_admin.json").write_text(json.dumps(config))
    monkeypatch.setattr(admin_cert_provider, "_cache_base_dir", lambda: tmp_path / "cache")
    with patch.object(api_utils, "Session", wraps=flare_api.Session) as factory:
        started = time.monotonic()
        with pytest.raises(ConfigError, match=message) as exc_info:
            api_utils.wait_for_system_start(1, str(tmp_path), second_to_wait=0, timeout_in_sec=30, secure_mode=True)
        assert not isinstance(exc_info.value, AdminCertAcquisitionError)
        assert time.monotonic() - started < 2
    factory.assert_called_once()
    if rejected:
        argv = json.loads(invocation_log.read_text())
        assert argv[argv.index(flag) + 1] == value


@pytest.mark.parametrize("connector", ["internal", "adhoc"])
def test_real_connection_configuration_error_fails_without_retry(tmp_path, monkeypatch, connector):
    import json

    from nvflare.fuel.f3.comm_config import CommConfigurator
    from nvflare.fuel.flare_api import flare_api
    from nvflare.fuel.utils.config_service import ConfigService
    from nvflare.lighter.utils import Identity, generate_cert, generate_keys, serialize_cert, serialize_pri_key
    from nvflare.tool import api_utils

    startup = tmp_path / "admin" / "startup"
    startup.mkdir(parents=True)
    (tmp_path / "admin" / "local").mkdir()
    key, public_key = generate_keys()
    identity = Identity("admin", "test")
    cert = generate_cert(subject=identity, issuer=identity, signing_pri_key=key, subject_pub_key=public_key, ca=True)
    (startup / "rootCA.pem").write_bytes(serialize_cert(cert))
    (startup / "client.crt").write_bytes(serialize_cert(cert))
    (startup / "client.key").write_bytes(serialize_pri_key(key))
    config = {
        "admin": {
            "ca_cert": "rootCA.pem",
            "client_cert": "client.crt",
            "client_key": "client.key",
            "upload_dir": "transfer",
            "download_dir": "transfer",
        }
    }
    (startup / "fed_admin.json").write_text(json.dumps(config))
    (tmp_path / "comm_config.json").write_text(json.dumps({connector: "not a mapping"}))
    monkeypatch.setattr(ConfigService, "_config_path", [str(tmp_path)])
    monkeypatch.setattr(CommConfigurator, "_config_loaded", False)
    monkeypatch.setattr(CommConfigurator, "_configuration", None)
    closed = threading.Event()
    real_close = flare_api.Session.close

    def close(session):
        try:
            real_close(session)
        finally:
            closed.set()

    monkeypatch.setattr(flare_api.Session, "close", close)
    with patch.object(api_utils, "Session", wraps=flare_api.Session) as factory:
        started = time.monotonic()
        with pytest.raises(ConfigError, match=f"'{connector}' must be dict"):
            api_utils.wait_for_system_start(1, str(tmp_path), second_to_wait=0, timeout_in_sec=30, secure_mode=True)
        assert time.monotonic() - started < 2
        assert closed.wait(2)
    factory.assert_called_once()
