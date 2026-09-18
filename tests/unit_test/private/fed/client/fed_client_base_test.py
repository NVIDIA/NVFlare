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

import socket
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from nvflare.apis.fl_constant import ConnectionSecurity, ConnPropKey, SecureTrainConst
from nvflare.apis.signal import Signal
from nvflare.fuel.f3.cellnet.identity import CellIdentityResolver
from nvflare.fuel.f3.communicator import Communicator
from nvflare.fuel.f3.drivers.driver_params import DriverParams
from nvflare.fuel.f3.endpoint import Endpoint, EndpointState
from nvflare.lighter.utils import Identity, generate_cert, generate_keys, serialize_cert, serialize_pri_key
from nvflare.private.fed.client import fed_client_base, upgrade
from nvflare.private.fed.client.fed_client_base import FederatedClientBase

_SITE_ARGS = {
    SecureTrainConst.SSL_ROOT_CERT: "rootCA.pem",
    SecureTrainConst.SSL_CERT: "client.crt",
    SecureTrainConst.PRIVATE_KEY: "client.key",
}
# in a job process the configer has already made the job credential the ssl_cert / ssl_private_key
_JOB_ARGS = {
    SecureTrainConst.SSL_ROOT_CERT: "rootCA.pem",
    SecureTrainConst.SSL_CERT: "job.crt",
    SecureTrainConst.PRIVATE_KEY: "job.key",
}


@pytest.mark.parametrize("cancel", [False, True])
@pytest.mark.parametrize("initial_state", [EndpointState.ERROR, None])
def test_upgrade_probe_retries_and_cleans_up(monkeypatch, initial_state, cancel):
    signal = Signal()
    probes = [MagicMock(), MagicMock()]
    factory = MagicMock(side_effect=probes)
    monkeypatch.setattr(upgrade, "Communicator", factory)
    monkeypatch.setattr(upgrade.MainProcessMonitor, "_stopping", False)

    def report(probe, state):
        if state is not None:
            endpoint = Endpoint("server")
            endpoint.state = state
            probe.register_monitor.call_args.args[0].state_change(endpoint)

    probes[0].start.side_effect = lambda: report(probes[0], initial_state)
    probes[1].start.side_effect = lambda: signal.trigger(True) if cancel else report(probes[1], EndpointState.READY)
    with pytest.raises(RuntimeError, match="cancelled") if cancel else nullcontext():
        upgrade.wait_for_server(
            "site-1",
            "server",
            "tcp://localhost:8002",
            False,
            {},
            None,
            {},
            signal,
            retry_interval=0.01,
            timeout=0.01,
        )
    assert factory.call_count == 2
    for call, probe in zip(factory.call_args_list, probes):
        assert call.args[0].name == "site-1.upgrade-probe"
        probe.stop.assert_called_once()
        probe.send.assert_not_called()
        probe.register_message_receiver.assert_not_called()


def _make_client():
    client = FederatedClientBase.__new__(FederatedClientBase)
    client._shutdown_lock = threading.Lock()
    client.communicator = SimpleNamespace(heartbeat_done=False)
    client.cell = MagicMock()
    client.engine = None
    client.client_name = "site-1"
    client.logger = MagicMock()
    client.terminate = MagicMock()
    client.logout_client = MagicMock()
    return client


def _create_cell_credentials(monkeypatch, job_id, client_args, conn_props=None):
    captured = {}

    class _FakeCell:
        def __init__(self, **kwargs):
            assert job_id or "probe" in captured
            captured.update(kwargs)

        def start(self):
            pass

        def stop(self):
            pass

    if conn_props is None:
        conn_props = {ConnPropKey.CP_CONN_PROPS: {ConnPropKey.FQCN: "site-1", ConnPropKey.URL: "tcp://cp:1"}}
    monkeypatch.setattr(fed_client_base, "Cell", _FakeCell)
    monkeypatch.setattr(fed_client_base, "wait_for_server", lambda **kwargs: captured.update(probe=kwargs))
    monkeypatch.setattr(fed_client_base, "NetAgent", lambda cell: MagicMock())
    monkeypatch.setattr(fed_client_base.mpm, "add_cleanup_cb", lambda cb: None)
    monkeypatch.setattr(
        fed_client_base, "get_scope_property", lambda name, key, default=None: conn_props.get(key, default)
    )

    client = _make_client()
    client.secure_train = True
    client.client_args = dict(client_args)
    client.args = SimpleNamespace(job_id=job_id)
    client.abort_signal = Signal()
    client.communicator = MagicMock()
    client.engine_create_timeout = 1.0
    client.cell_check_frequency = 0.001
    client.engine = MagicMock()
    client.client_runner = MagicMock()

    client._create_cell("localhost:8002", "grpc")
    return captured


def test_cp_cell_uses_site_credential(monkeypatch):
    captured = _create_cell_credentials(monkeypatch, None, _SITE_ARGS)
    credentials = captured["credentials"]

    assert credentials[DriverParams.CLIENT_CERT.value] == "client.crt"
    assert credentials[DriverParams.CLIENT_KEY.value] == "client.key"
    assert DriverParams.SERVER_CERT.value not in credentials
    assert captured["probe"]["credentials"] == credentials
    assert captured["probe"]["fqcn"] == "site-1"
    assert captured["probe"]["peer_fqcn"] == "server"
    assert captured["probe"]["url"] == "grpc://localhost:8002"
    assert captured["probe"]["retry_interval"] == 60.0


def test_cj_cell_uses_job_credential_in_both_tls_roles(monkeypatch):
    captured = _create_cell_credentials(monkeypatch, "job-1", _JOB_ARGS)
    credentials = captured["credentials"]

    assert "probe" not in captured
    assert credentials[DriverParams.CLIENT_CERT.value] == "job.crt"
    assert credentials[DriverParams.CLIENT_KEY.value] == "job.key"
    assert credentials[DriverParams.SERVER_CERT.value] == "job.crt"
    assert credentials[DriverParams.SERVER_KEY.value] == "job.key"


@pytest.fixture
def probe_credentials(tmp_path, monkeypatch):
    monkeypatch.setattr(upgrade.MainProcessMonitor, "_stopping", False)
    root_key, root_pub = generate_keys()
    root = Identity("probe-test-ca")
    ca_path = tmp_path / "rootCA.pem"
    ca_path.write_bytes(serialize_cert(generate_cert(root, root, root_key, root_pub, ca=True)))
    credentials = []
    for name, role in [("localhost", "server"), ("site-1", "client")]:
        key, pub = generate_keys()
        cert_path, key_path = tmp_path / f"{role}.crt", tmp_path / f"{role}.key"
        cert_path.write_bytes(serialize_cert(generate_cert(Identity(name), root, root_key, pub)))
        key_path.write_bytes(serialize_pri_key(key))
        credentials.append({"ca_cert": str(ca_path), f"{role}_cert": str(cert_path), f"{role}_key": str(key_path)})
    return credentials


@pytest.mark.timeout(15)
def test_upgrade_probe_cancels_stalled_tls(probe_credentials):
    _, credentials = probe_credentials
    signal = Signal()
    with socket.socket() as listener, ThreadPoolExecutor(1) as executor:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        listener.settimeout(5)
        future = executor.submit(
            upgrade.wait_for_server,
            "site-1",
            "server",
            f"stcp://127.0.0.1:{listener.getsockname()[1]}",
            True,
            credentials,
            None,
            {},
            signal,
            60,
            0.3,
        )
        try:
            connection, _ = listener.accept()
            with connection:
                # Leave TCP open without answering TLS: stop() must not wait forever.
                signal.trigger(True)
                with pytest.raises(RuntimeError, match="cancelled"):
                    future.result(timeout=3)
                connection.settimeout(1)
                while connection.recv(4096):
                    pass  # Consume ClientHello and verify the probe closed its socket.
        finally:
            signal.trigger(True)


@pytest.mark.timeout(15)
@pytest.mark.parametrize("security", [ConnectionSecurity.MTLS, ConnectionSecurity.CLEAR])
def test_upgrade_probe_through_relay(monkeypatch, probe_credentials, security):
    server_credentials, client_credentials = probe_credentials
    server_credentials[DriverParams.CONNECTION_SECURITY] = security
    relay = Communicator(Endpoint("relay-a", conn_props=server_credentials), CellIdentityResolver("relay-a"))
    try:
        _, url, _ = relay.start_listener(
            "stcp" if security == ConnectionSecurity.MTLS else "tcp", {"host": "127.0.0.1"}
        )
        relay.start()
        captured = _create_cell_credentials(
            monkeypatch,
            None,
            {
                SecureTrainConst.SSL_ROOT_CERT: client_credentials["ca_cert"],
                SecureTrainConst.SSL_CERT: client_credentials["client_cert"],
                SecureTrainConst.PRIVATE_KEY: client_credentials["client_key"],
                ConnPropKey.CONNECTION_SECURITY: (
                    ConnectionSecurity.CLEAR if security == ConnectionSecurity.MTLS else ConnectionSecurity.MTLS
                ),
                "upgrade_probe_interval": 0.01,
            },
            {
                ConnPropKey.CP_CONN_PROPS: {ConnPropKey.FQCN: "relay-a.site-1"},
                ConnPropKey.RELAY_CONN_PROPS: {
                    ConnPropKey.FQCN: "relay-a",
                    ConnPropKey.URL: url,
                    ConnPropKey.AUTH_IDENTITY: "localhost",
                    ConnPropKey.CONNECTION_SECURITY: security,
                },
            },
        )
        assert captured["root_url"] is None
        probe_args = captured["probe"]
        signal = probe_args["abort_signal"]
        with ThreadPoolExecutor(1) as executor:
            try:
                executor.submit(upgrade.wait_for_server, **probe_args, timeout=1).result(timeout=5)
            finally:
                signal.trigger(True)
        assert relay.find_endpoint("relay-a.site-1.upgrade-probe") is not None
        assert relay.find_endpoint("relay-a.site-1") is None
    finally:
        relay.stop()


def test_send_request_before_shutdown_skips_after_close():
    client = _make_client()
    reply = MagicMock()
    client.cell.send_request.return_value = reply

    assert client.send_request_before_shutdown(topic="terminal_outcome") is reply

    client.close()

    assert client.communicator.heartbeat_done is True
    assert client.send_request_before_shutdown(topic="terminal_outcome") is None
    client.cell.send_request.assert_called_once_with(topic="terminal_outcome")
    client.logout_client.assert_called_once()


def test_close_waits_for_terminal_request_before_logout():
    client = _make_client()
    request_started = threading.Event()
    release_request = threading.Event()
    close_entered = threading.Event()

    def send_request(**_kwargs):
        request_started.set()
        assert release_request.wait(timeout=1.0)
        return MagicMock()

    client.cell.send_request.side_effect = send_request
    client.terminate.side_effect = close_entered.set

    request_thread = threading.Thread(target=client.send_request_before_shutdown, kwargs={"topic": "outcome"})
    request_thread.start()
    assert request_started.wait(timeout=1.0)

    close_thread = threading.Thread(target=client.close)
    close_thread.start()
    assert not close_entered.wait(timeout=0.1)
    client.logout_client.assert_not_called()

    release_request.set()
    request_thread.join(timeout=1.0)
    close_thread.join(timeout=1.0)

    assert not request_thread.is_alive()
    assert not close_thread.is_alive()
    assert client.communicator.heartbeat_done is True
    client.logout_client.assert_called_once()
