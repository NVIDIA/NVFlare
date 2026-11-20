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

import errno
import json
import socket
import ssl
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

import grpc
import pytest

from nvflare.tool.package_checker.check_rule import CHECK_PASSED, CheckServerAvailable
from nvflare.tool.package_checker.utils import try_bind_address
from tests.job_cert_test_utils import _write_pki


@pytest.fixture(scope="module")
def pki(tmp_path_factory):
    return _write_pki(str(tmp_path_factory.mktemp("pki")))


def client_probe(kit, scheme, port, security, **paths):
    startup = kit / "startup"
    startup.mkdir()
    config = {
        "client": {"connection_security": security, **paths},
        "servers": [{"service": {"scheme": scheme, "target": f"localhost:{port}"}}],
    }
    (startup / "fed_client.json").write_text(json.dumps(config))
    return CheckServerAvailable("Check server available", "client")(str(kit), None).problem


@contextmanager
def accept_probe(kit, scheme, security, **paths):
    with socket.socket() as listener, ThreadPoolExecutor(max_workers=1) as executor:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        listener.settimeout(5)
        result = executor.submit(client_probe, kit, scheme, listener.getsockname()[1], security, **paths)
        conn, _ = listener.accept()
        with conn:
            conn.settimeout(5)
            yield conn, result


@pytest.mark.parametrize("scheme", ["tcp", "http", "atcp", "satcp"])
def test_socket_probe_performs_mutual_tls(tmp_path, pki, scheme):
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(pki["server.crt"], pki["server.key"])
    context.load_verify_locations(pki["rootCA.pem"])
    context.verify_mode = ssl.CERT_REQUIRED
    with accept_probe(
        tmp_path,
        scheme,
        "mtls",
        ssl_root_cert=pki["rootCA.pem"],
        ssl_cert=pki["site-1.crt"],
        ssl_private_key=pki["site-1.key"],
    ) as (conn, result):
        with context.wrap_socket(conn, server_side=True) as tls:
            assert tls.getpeercert(binary_form=True)
    assert result.result() == CHECK_PASSED


@pytest.mark.parametrize("scheme", ["grpc", "satcp"])
def test_one_way_tls_probe_uses_custom_ca_without_client_key(tmp_path, pki, scheme):
    # The proxy's CA lives next to the configured project CA. No site key is installed.
    (tmp_path / "customRootCA.pem").write_bytes(Path(pki["rootCA.pem"]).read_bytes())
    with ThreadPoolExecutor(max_workers=1) as executor:
        server = grpc.server(executor)
        credentials = grpc.ssl_server_credentials(
            [(Path(pki["server.key"]).read_bytes(), Path(pki["server.crt"]).read_bytes())]
        )
        port = server.add_secure_port("127.0.0.1:0", credentials)
        server.start()
        try:
            assert (
                client_probe(tmp_path, scheme, port, "tls", ssl_root_cert=str(tmp_path / "project.pem")) == CHECK_PASSED
            )
        finally:
            assert server.stop(0).wait(10)


@pytest.mark.parametrize(
    "scheme,security,expected", [("atcp", "clear", True), ("satcp", "tls", False), ("satcp", "clear", False)]
)
def test_probe_against_plaintext_listener(tmp_path, pki, scheme, security, expected):
    with accept_probe(
        tmp_path,
        scheme,
        security,
        ssl_root_cert=pki["rootCA.pem"],
        ssl_cert=pki["site-1.crt"],
        ssl_private_key=pki["site-1.key"],
    ) as (conn, result):
        if not expected:
            conn.sendall(b"HTTP/1.0 400 Bad Request\r\n\r\n")
    assert (result.result() == CHECK_PASSED) is expected


@pytest.mark.parametrize("listener_host", ["127.0.0.1", "0.0.0.0"])
def test_port_probe_distinguishes_listener_from_time_wait(listener_host):
    with socket.socket() as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((listener_host, 0))
        listener.listen()
        listener.settimeout(5)
        port = listener.getsockname()[1]
        # Accept our control connection first: preflight may make its own connection to the listener.
        with socket.create_connection(("127.0.0.1", port), timeout=5) as client:
            conn, _ = listener.accept()
            with conn:
                assert try_bind_address("127.0.0.1", port).errno == errno.EADDRINUSE
            assert client.recv(1) == b""
    # The server closed first, so its local port still has TIME_WAIT and an ordinary bind fails.
    with socket.socket() as exclusive, pytest.raises(OSError) as err:
        exclusive.bind(("127.0.0.1", port))
    assert err.value.errno == errno.EADDRINUSE
    assert try_bind_address("127.0.0.1", port) is None
