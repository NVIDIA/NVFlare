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

import asyncio
import socket
import threading
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import serialization

from nvflare.fuel.f3.drivers import aio_grpc_driver, grpc_driver
from nvflare.fuel.f3.drivers.connector_info import Mode
from nvflare.fuel.f3.drivers.driver_params import DriverParams
from nvflare.fuel.f3.drivers.grpc import utils as grpc_utils
from nvflare.fuel.f3.endpoint import Endpoint
from nvflare.fuel.f3.sfm.conn_manager import ConnManager
from nvflare.fuel.sec.cert_uri import CELL_URI_KIND, cert_uri
from nvflare.lighter.utils import Identity, generate_cert, generate_keys, serialize_cert, serialize_pri_key
from tests.timing_utils import CheckedThread, ManualClock


def _cert_pem(common_name: str, scoped: bool) -> bytes:
    key, pub_key = generate_keys()
    uri_names = [cert_uri(CELL_URI_KIND, "site-1.job-1")] if scoped else None
    cert = generate_cert(Identity(common_name), Identity(common_name), key, pub_key, uri_names=uri_names)
    return serialize_cert(cert)


def _der(pem: bytes) -> bytes:
    return x509.load_pem_x509_certificate(pem).public_bytes(serialization.Encoding.DER)


def _authenticated_context(cert_pem: bytes):
    context = MagicMock()
    context.peer.return_value = "ipv4:127.0.0.1:50001"
    context.auth_context.return_value = {"x509_common_name": [b"site-1"], "x509_pem_cert": [cert_pem]}
    return context


def _server():
    server = MagicMock()
    server.connector.params = {DriverParams.HOST.value: "localhost", DriverParams.PORT.value: "8002"}
    return server


def test_explicit_empty_grpc_proxy_disables_general_proxy(monkeypatch):
    monkeypatch.delenv("no_grpc_proxy", raising=False)
    monkeypatch.delenv("no_proxy", raising=False)
    monkeypatch.setenv("grpc_proxy", "")
    monkeypatch.setenv("https_proxy", "http://proxy.example:3128")

    assert grpc_utils._proxy_for_target("listener.example", 8443, None) is None

    monkeypatch.setenv("grpc_proxy", "http://proxy.example:3128")
    assert grpc_utils._proxy_for_target("listener.example", 8443, [("grpc.http_proxy", "")]) is None


def test_explicit_empty_grpc_bypass_does_not_use_general_bypass(monkeypatch):
    monkeypatch.setenv("grpc_proxy", "http://proxy.example:3128")
    monkeypatch.setenv("no_grpc_proxy", "")
    monkeypatch.setenv("no_proxy", "listener.example")

    assert grpc_utils._proxy_for_target("listener.example", 8443, None) == ("proxy.example", 3128, None)


def test_probe_connect_reserves_time_for_fallback_address(monkeypatch):
    clock = ManualClock(now=0)
    monkeypatch.setattr(grpc_utils, "time", clock)
    ipv6, ipv4 = MagicMock(), MagicMock()
    pending = MagicMock(closed=False)
    pending.resolve_addresses.return_value = [
        (socket.AF_INET6, socket.SOCK_STREAM, 0, "", ("::1", 8443, 0, 0)),
        (socket.AF_INET, socket.SOCK_STREAM, 0, "", ("127.0.0.1", 8443)),
    ]

    def stalled_address(address):
        clock.advance(ipv6.settimeout.call_args.args[0])
        raise TimeoutError

    ipv6.connect.side_effect = stalled_address
    monkeypatch.setattr(grpc_utils.socket, "socket", MagicMock(side_effect=[ipv6, ipv4]))
    assert grpc_utils._connect_raw("example", 8443, 10, pending) is ipv4
    ipv6.close.assert_called_once()
    ipv4.connect.assert_called_once_with(("127.0.0.1", 8443))
    ipv4.settimeout.assert_called_once_with(5.0)


def test_probe_deadline_is_shared_across_proxy_response_reads(monkeypatch):
    clock = ManualClock(now=0)
    monkeypatch.setattr(grpc_utils, "time", clock)
    raw, context = MagicMock(), MagicMock()
    monkeypatch.setattr(grpc_utils, "_connect_raw", MagicMock(return_value=raw))
    monkeypatch.setattr(
        grpc_utils,
        "_probe_config",
        MagicMock(return_value=(context, "listener", 443, "listener", 1.0, ("proxy", 80, None))),
    )
    timeouts = []

    def receive(size):
        timeout = raw.settimeout.call_args.args[0]
        timeouts.append(timeout)
        clock.advance(0.4)
        # Finite response on the old implementation, so a regression fails promptly.
        return b"x" if len(timeouts) < 4 else b"\r\n\r\n"

    raw.recv.side_effect = receive
    with pytest.raises(TimeoutError, match="certificate probe timed out"):
        grpc_utils.probe_grpc_peer_cert({})
    assert timeouts == pytest.approx([1.0, 0.6, 0.2])
    context.wrap_socket.assert_not_called()
    raw.close.assert_called_once()


def test_grpc_connector_stop_does_not_wait_for_stalled_dns(tmp_path, monkeypatch):
    ca_key, ca_public = generate_keys()
    ca_cert = generate_cert(Identity("rootCA"), Identity("rootCA"), ca_key, ca_public, ca=True)
    client_key, client_public = generate_keys()
    client_cert = generate_cert(Identity("site-1"), Identity("rootCA"), ca_key, client_public)
    for name, data in {
        "rootCA.pem": serialize_cert(ca_cert),
        "client.crt": serialize_cert(client_cert),
        "client.key": serialize_pri_key(client_key),
    }.items():
        (tmp_path / name).write_bytes(data)

    entered, release = threading.Event(), threading.Event()

    def stalled_lookup(*args, **kwargs):
        entered.set()
        release.wait(10)
        return []

    monkeypatch.setenv("no_grpc_proxy", "listener.invalid")
    manager = ConnManager(Endpoint("site-1"))
    driver = grpc_driver.GrpcDriver()
    manager.add_connector(
        driver,
        {
            DriverParams.SCHEME.value: "grpcs",
            DriverParams.HOST.value: "listener.invalid",
            DriverParams.PORT.value: 8443,
            DriverParams.CA_CERT.value: str(tmp_path / "rootCA.pem"),
            DriverParams.CLIENT_CERT.value: str(tmp_path / "client.crt"),
            DriverParams.CLIENT_KEY.value: str(tmp_path / "client.key"),
            DriverParams.CONNECT_TIMEOUT.value: None,
        },
        Mode.ACTIVE,
    )
    stopper = CheckedThread(target=manager.stop)
    stopped_promptly = False
    with patch.object(grpc_utils.socket, "getaddrinfo", side_effect=stalled_lookup):
        try:
            manager.start()
            assert entered.wait(3), "certificate probe did not reach DNS"
            stopper.start()
            stopper.join(timeout=3)
            stopped_promptly = not stopper.is_alive()
        finally:
            release.set()
            if stopper.ident is None:
                stopper.start()
            stopper.join(timeout=5)
            resolver = driver.pending_probes.resolver_job
            if resolver:
                resolver["thread"].join(timeout=5)
                assert not resolver["thread"].is_alive()
            assert not stopper.is_alive()
            stopper.raise_if_failed()
    assert stopped_promptly, "stalled DNS delayed connector shutdown"


@pytest.mark.parametrize("scoped", [True, False])
def test_grpc_servicer_records_peer_identity_of_authenticated_stream(scoped):
    servicer = grpc_driver.Servicer(_server())
    pem = _cert_pem("site-1", scoped)

    with (
        patch.object(grpc_driver, "StreamConnection") as connection_cls,
        patch.object(grpc_driver.threading, "Thread"),
    ):
        connection_cls.return_value.generate_output.return_value = iter([])
        list(servicer.Stream(iter([]), _authenticated_context(pem)))

    conn_props = connection_cls.call_args.args[2]
    assert conn_props[DriverParams.PEER_CN.value] == "site-1"
    assert conn_props[DriverParams.PEER_CERT.value] == _der(pem)


@pytest.mark.parametrize("scoped", [True, False])
def test_aio_grpc_servicer_records_peer_identity_of_authenticated_stream(scoped):
    servicer = aio_grpc_driver.Servicer(_server(), aio_ctx=MagicMock())
    pem = _cert_pem("site-1", scoped)

    async def consume():
        async for _ in servicer.Stream(iter([]), _authenticated_context(pem)):
            pass

    with patch.object(aio_grpc_driver, "AioStreamSession") as session_cls:
        session_cls.return_value.read_oq = AsyncMock(side_effect=asyncio.CancelledError())
        asyncio.run(consume())

    conn_props = session_cls.call_args.kwargs["conn_props"]
    assert conn_props[DriverParams.PEER_CN.value] == "site-1"
    assert conn_props[DriverParams.PEER_CERT.value] == _der(pem)
