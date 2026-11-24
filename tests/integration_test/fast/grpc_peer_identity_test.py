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

"""Live gRPC mTLS identity binding for both active driver implementations."""

import multiprocessing as mp
import select
import socket
import socketserver
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

import grpc
import pytest

from nvflare.fuel.f3.cellnet.cell import Cell
from nvflare.fuel.f3.cellnet.core_cell import CoreCell
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey, ReturnCode
from nvflare.fuel.f3.cellnet.identity import CellIdentityResolver
from nvflare.fuel.f3.drivers.connector_info import Mode
from nvflare.fuel.f3.drivers.driver_params import DriverParams
from nvflare.fuel.f3.drivers.grpc.utils import get_grpc_client_credentials, probe_grpc_peer_cert
from nvflare.fuel.f3.drivers.grpc_driver import GrpcDriver
from nvflare.fuel.f3.endpoint import Endpoint
from nvflare.fuel.f3.message import Message
from nvflare.fuel.f3.sfm.conn_manager import ConnManager
from nvflare.fuel.f3.stats_pool import StatsPoolManager
from nvflare.lighter.utils import Identity, generate_cert, generate_keys, serialize_cert, serialize_pri_key
from tests.job_cert_test_utils import _free_port
from tests.timing_utils import CheckedThread, stop_processes


def _write_pki(tmp_path, listener_cn):
    root_key, root_public = generate_keys()
    root_cert = generate_cert(Identity("rootCA"), Identity("rootCA"), root_key, root_public, ca=True)
    listener_key, listener_public = generate_keys()
    listener_cert = generate_cert(
        Identity(listener_cn), Identity("rootCA"), root_key, listener_public, server_default_host="localhost"
    )
    alternate_key, alternate_public = generate_keys()
    alternate_cert = generate_cert(
        Identity(listener_cn), Identity("rootCA"), root_key, alternate_public, server_default_host="localhost"
    )
    client_key, client_public = generate_keys()
    client_cert = generate_cert(Identity("site-1"), Identity("rootCA"), root_key, client_public)
    values = {
        "rootCA.pem": serialize_cert(root_cert),
        "server.crt": serialize_cert(listener_cert),
        "server.key": serialize_pri_key(listener_key),
        "alternate.crt": serialize_cert(alternate_cert),
        "alternate.key": serialize_pri_key(alternate_key),
        "client.crt": serialize_cert(client_cert),
        "client.key": serialize_pri_key(client_key),
    }
    for name, value in values.items():
        (tmp_path / name).write_bytes(value)
    return {name: str(tmp_path / name) for name in values}


@pytest.fixture
def isolated_cell_registry():
    cells, pools = dict(CoreCell.ALL_CELLS), dict(StatsPoolManager.pools)
    CoreCell.ALL_CELLS.clear()
    StatsPoolManager.pools.clear()
    try:
        yield
    finally:
        CoreCell.ALL_CELLS.clear()
        CoreCell.ALL_CELLS.update(cells)
        StatsPoolManager.pools.clear()
        StatsPoolManager.pools.update(pools)


def _run_server(root_url, pki, ready_q, stop_ev):
    server = None
    try:
        server = Cell(
            "server",
            root_url,
            secure=True,
            credentials={
                DriverParams.CA_CERT.value: pki["rootCA.pem"],
                DriverParams.SERVER_CERT.value: pki["server.crt"],
                DriverParams.SERVER_KEY.value: pki["server.key"],
                DriverParams.CONNECTION_SECURITY.value: "mtls",
            },
        )
        server.register_request_cb("identity", "echo", lambda request: Message(payload=request.payload))
        server.start()
        ready_q.put("ready")
        stop_ev.wait(30)
    except Exception:
        ready_q.put(traceback.format_exc())
    finally:
        if server:
            server.stop()


class _ConnectProxyHandler(socketserver.BaseRequestHandler):
    def handle(self):
        self.request.settimeout(5)
        request = b""
        while b"\r\n\r\n" not in request and len(request) < 8192:
            chunk = self.request.recv(8192 - len(request))
            if not chunk:
                return
            request += chunk
        method, target, _ = request.split(b"\r\n", 1)[0].split(b" ", 2)
        assert method == b"CONNECT"
        with self.server.count_lock:
            self.server.connection_count += 1
        host, port = target.decode("ascii").rsplit(":", 1)
        with socket.create_connection((host.strip("[]"), int(port)), timeout=5) as upstream:
            self.request.sendall(b"HTTP/1.1 200 Connection established\r\n\r\n")
            while True:
                readable, _, _ = select.select((self.request, upstream), (), (), 5)
                if not readable:
                    return
                for source in readable:
                    data = source.recv(65536)
                    if not data:
                        return
                    (upstream if source is self.request else self.request).sendall(data)


@contextmanager
def _http_proxy():
    proxy = socketserver.ThreadingTCPServer(("127.0.0.1", 0), _ConnectProxyHandler)
    proxy.daemon_threads = True
    proxy.count_lock = threading.Lock()
    proxy.connection_count = 0
    worker = threading.Thread(target=proxy.serve_forever)
    try:
        worker.start()
        yield proxy
    finally:
        proxy.shutdown()
        proxy.server_close()
        worker.join(timeout=5)
        assert not worker.is_alive()


@pytest.mark.parametrize("scheme", ["grpcs", "agrpcs"])
@pytest.mark.parametrize("listener_cn", ["server", "rogue-server"])
def test_active_grpc_mtls_requires_listener_cn(scheme, listener_cn, tmp_path, isolated_cell_registry, monkeypatch):
    identity_rejected = threading.Event()
    require_match = CellIdentityResolver.require_match

    def observe_rejection(resolver, fqcn, peer_cn, peer_desc, peer_cert=None):
        try:
            return require_match(resolver, fqcn, peer_cn, peer_desc, peer_cert)
        except ValueError:
            if fqcn == "server" and peer_cn == "rogue-server":
                identity_rejected.set()
            raise

    monkeypatch.setattr(CellIdentityResolver, "require_match", observe_rejection)
    pki = _write_pki(tmp_path, listener_cn)
    root_url = f"{scheme}://localhost:{_free_port()}"
    ctx = mp.get_context("spawn")
    ready_q, stop_ev = ctx.Queue(), ctx.Event()
    server = ctx.Process(target=_run_server, args=(root_url, pki, ready_q, stop_ev))
    client = None
    try:
        server.start()
        status = ready_q.get(timeout=20)
        assert status == "ready", status

        client = Cell(
            "site-1",
            root_url,
            secure=True,
            credentials={
                DriverParams.CA_CERT.value: pki["rootCA.pem"],
                DriverParams.CLIENT_CERT.value: pki["client.crt"],
                DriverParams.CLIENT_KEY.value: pki["client.key"],
                DriverParams.CONNECTION_SECURITY.value: "mtls",
                DriverParams.CONNECT_TIMEOUT.value: 3.0,
            },
            auth_identity_map={"server": "server"},
        )
        client.start()

        if listener_cn == "server":
            deadline = time.monotonic() + 6.0
            while time.monotonic() < deadline and not client.is_cell_connected("server"):
                time.sleep(0.05)
            assert client.is_cell_connected("server")
            reply = client.send_request("identity", "echo", "server", Message(payload="hello"), timeout=3.0)
            assert reply.get_header(MessageHeaderKey.RETURN_CODE) == ReturnCode.OK
            assert reply.payload == "hello"
        else:
            assert identity_rejected.wait(8), "client never rejected the listener certificate identity"
            assert not client.is_cell_connected("server")
            reply = client.send_request("identity", "echo", "server", Message(payload="hello"), timeout=3.0)
            assert reply.get_header(MessageHeaderKey.RETURN_CODE) != ReturnCode.OK
    finally:
        try:
            if client:
                client.stop()
        finally:
            stop_ev.set()
            stop_processes(server)


@pytest.mark.parametrize("scheme", ["grpcs", "agrpcs"])
@pytest.mark.parametrize("proxy_mode", ["connect", "wildcard", "bypass", "unsupported_scheme"])
def test_active_grpc_mtls_through_http_connect_proxy(scheme, proxy_mode, tmp_path, monkeypatch, isolated_cell_registry):
    pki = _write_pki(tmp_path, "server")
    root_url = f"{scheme}://localhost:{_free_port()}"
    ctx = mp.get_context("spawn")
    ready_q, stop_ev = ctx.Queue(), ctx.Event()
    server = ctx.Process(target=_run_server, args=(root_url, pki, ready_q, stop_ev))
    client = None
    try:
        server.start()
        assert ready_q.get(timeout=20) == "ready"
        with _http_proxy() as proxy:
            for name in ("grpc_proxy", "https_proxy", "http_proxy", "no_grpc_proxy", "no_proxy"):
                monkeypatch.delenv(name, raising=False)
            if proxy_mode == "unsupported_scheme":
                monkeypatch.setenv("https_proxy", f"https://127.0.0.1:{proxy.server_address[1]}")
            else:
                monkeypatch.setenv("grpc_proxy", f"http://127.0.0.1:{proxy.server_address[1]}")
            if proxy_mode in ("wildcard", "bypass"):
                monkeypatch.setenv("no_grpc_proxy", "*" if proxy_mode == "wildcard" else "localhost")
            client = Cell(
                "site-1",
                root_url,
                secure=True,
                credentials={
                    DriverParams.CA_CERT.value: pki["rootCA.pem"],
                    DriverParams.CLIENT_CERT.value: pki["client.crt"],
                    DriverParams.CLIENT_KEY.value: pki["client.key"],
                    DriverParams.CONNECTION_SECURITY.value: "mtls",
                    DriverParams.CONNECT_TIMEOUT.value: 3.0,
                },
                auth_identity_map={"server": "server"},
            )
            client.start()
            deadline = time.monotonic() + 6.0
            while time.monotonic() < deadline and not client.is_cell_connected("server"):
                time.sleep(0.05)
            assert client.is_cell_connected("server")
            reply = client.send_request("identity", "echo", "server", Message(payload="proxy"), timeout=3.0)
            assert reply.get_header(MessageHeaderKey.RETURN_CODE) == ReturnCode.OK
            assert reply.payload == "proxy"
            if proxy_mode in ("connect", "wildcard"):
                assert proxy.connection_count >= 2
            else:
                assert proxy.connection_count == 0
            client.stop()
            client = None
    finally:
        try:
            if client:
                client.stop()
        finally:
            stop_ev.set()
            stop_processes(server)


def test_stalled_grpc_certificate_probe_stops_with_connector(tmp_path, monkeypatch):
    pki = _write_pki(tmp_path, "server")
    monkeypatch.setenv("no_grpc_proxy", "127.0.0.1")
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    listener.settimeout(3)
    manager = ConnManager(Endpoint("site-1"))
    manager.add_connector(
        GrpcDriver(),
        {
            DriverParams.SCHEME.value: "grpcs",
            DriverParams.HOST.value: "127.0.0.1",
            DriverParams.PORT.value: listener.getsockname()[1],
            DriverParams.CA_CERT.value: pki["rootCA.pem"],
            DriverParams.CLIENT_CERT.value: pki["client.crt"],
            DriverParams.CLIENT_KEY.value: pki["client.key"],
            DriverParams.CONNECT_TIMEOUT.value: None,
        },
        Mode.ACTIVE,
    )
    stopper = CheckedThread(target=manager.stop)
    peer = None
    stopped_promptly = False
    try:
        manager.start()
        peer, _ = listener.accept()
        peer.settimeout(3)
        assert peer.recv(1), "TLS ClientHello was not sent"
        stopper.start()
        stopper.join(timeout=3)
        stopped_promptly = not stopper.is_alive()
    finally:
        if stopper.ident is None:
            stopper.start()
        if peer:
            peer.close()
        listener.close()
        stopper.join(timeout=5)
        assert not stopper.is_alive()
        stopper.raise_if_failed()
    assert stopped_promptly, "stalled TLS probe delayed cell shutdown"


@pytest.mark.parametrize("authority_option", ["grpc.default_authority", "grpc.ssl_target_name_override"])
def test_grpc_channel_rejects_listener_certificate_change_after_probe(tmp_path, authority_option):
    pki = _write_pki(tmp_path, "server")
    servers = []
    channels = []
    try:
        for cert_name, key_name in (("server.crt", "server.key"), ("alternate.crt", "alternate.key")):
            server = grpc.server(ThreadPoolExecutor(max_workers=1))
            credentials = grpc.ssl_server_credentials(
                [(Path(pki[key_name]).read_bytes(), Path(pki[cert_name]).read_bytes())],
                root_certificates=Path(pki["rootCA.pem"]).read_bytes(),
                require_client_auth=True,
            )
            port = server.add_secure_port("localhost:0", credentials)
            assert port
            server.start()
            servers.append((server, port))

        params = {
            DriverParams.HOST.value: "127.0.0.1",
            DriverParams.PORT.value: servers[0][1],
            DriverParams.CA_CERT.value: pki["rootCA.pem"],
            DriverParams.CLIENT_CERT.value: pki["client.crt"],
            DriverParams.CLIENT_KEY.value: pki["client.key"],
            DriverParams.CONNECT_TIMEOUT.value: 3.0,
        }
        authority = "localhost:8443"
        options = [(authority_option, authority)]
        peer_cert = probe_grpc_peer_cert(params, options)
        channel_credentials = get_grpc_client_credentials(params, peer_cert=peer_cert)
        for _, port in servers:
            channel = grpc.secure_channel(f"127.0.0.1:{port}", channel_credentials, options=options)
            channels.append(channel)
        grpc.channel_ready_future(channels[0]).result(timeout=3)
        with pytest.raises(grpc.FutureTimeoutError):
            grpc.channel_ready_future(channels[1]).result(timeout=2)
    finally:
        for channel in channels:
            channel.close()
        for server, _ in servers:
            server.stop(grace=0).wait(timeout=3)
