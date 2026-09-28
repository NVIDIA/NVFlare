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

"""Real TLS transport tests, isolated in spawned processes like FLARE endpoints."""

import logging
import multiprocessing as mp
import os
import socket
import ssl
import time
import traceback
from pathlib import Path
from urllib.parse import urlparse

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization

from nvflare.fuel.f3.cellnet.cell import Cell
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey
from nvflare.fuel.f3.comm_config import CommConfigurator
from nvflare.fuel.f3.drivers.aio_context import AioContext
from nvflare.fuel.f3.drivers.connector_info import Mode
from nvflare.fuel.f3.drivers.net_utils import get_ssl_context
from nvflare.fuel.f3.message import Message
from nvflare.fuel.utils.config_service import ConfigService
from nvflare.lighter.utils import generate_keys, serialize_cert, serialize_pri_key
from tests.unit_test.security.certificate_renewal_test import issue, pki

pytestmark = pytest.mark.slow


def _endpoint(name, url, params, commands, results, directory, renewal=True):
    # Capture Python and native gRPC TLS errors in the child, independently of
    # pytest's setup/call capture boundaries and multiprocessing start method.
    stderr_fd = os.open(os.path.join(directory, "tls-errors.log"), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    os.dup2(stderr_fd, 2)
    os.close(stderr_fd)
    cell = None
    try:
        ConfigService.initialize(section_files={}, config_path=[directory])
        CommConfigurator.reset()
        cell = Cell(
            name,
            url,
            secure=True,
            credentials=params,
            auth_identity="localhost" if name == "server" else name,
            auth_identity_map={"server": "localhost", "site-1": "site-1"},
            certificate_renewal=renewal,
        )

        def work(request):
            time.sleep(2)
            return Message(payload=request.payload)

        cell.register_request_cb(channel="renewal_test", topic="work", cb=work)
        cell.start()
        results.put({"ready": os.getpid()})
        while True:
            command = commands.get(timeout=90)
            if command == "stop":
                break
            elif command == "tls_debug":
                # Observe asyncio's native TLS handshake failure, which some
                # transports close without flushing a TLS alert to the client.
                if url.startswith(("satcp:", "https:")):
                    logging.basicConfig()
                    context = AioContext.get_global_context()
                    context.logger.setLevel(logging.DEBUG)
                    context.get_event_loop().call_soon_threadsafe(context.get_event_loop().set_debug, True)
                results.put({"tls_debug": True})
            elif command == "cache_server":
                results.put({"certificate": cell.core_cell.cert_ex.get_certificate("server")})
            elif command == "disconnect":
                for conn in list(cell.core_cell.communicator.conn_manager.sfm_conns.values()):
                    conn.conn.close()
                results.put({"disconnected": True})
            elif command == "status":
                manager = cell.core_cell.communicator.conn_manager
                results.put(
                    {
                        "status": cell.core_cell.get_credential_status(),
                        "connections": [name for name, c in manager.sfm_conns.items() if c.sfm_endpoint],
                        "listeners": [
                            id(getattr(c.driver, "site", None) or getattr(c.driver, "server", None))
                            for c in manager.connectors.values()
                            if c.mode == Mode.PASSIVE
                        ],
                        "peer_certs": [
                            c.conn.get_conn_properties().get("peer_cert") for c in manager.sfm_conns.values()
                        ],
                        "pid": os.getpid(),
                    }
                )
            elif command in ("request", "plain_request", "encrypted_request"):
                sender = cell.core_cell if command == "plain_request" else cell
                reply = sender.send_request(
                    channel="renewal_test",
                    topic="work",
                    target="server",
                    request=Message(payload=b"weights" * 10000),
                    timeout=20,
                    secure=command == "encrypted_request",
                )
                results.put(
                    {
                        "rc": reply.get_header(MessageHeaderKey.RETURN_CODE),
                        "payload_ok": reply.payload == b"weights" * 10000,
                        "pid": os.getpid(),
                    }
                )
    except Exception:
        results.put({"error": traceback.format_exc()})
    finally:
        if cell:
            cell.stop()


def _get(queue):
    result = queue.get(timeout=35)
    assert "error" not in result, result.get("error")
    return result


def _server_certificate(url, client):
    address = urlparse(url)
    context = get_ssl_context(dict(client, scheme=address.scheme), False)
    context.set_alpn_protocols(["h2", "http/1.1"])
    with socket.create_connection((address.hostname, address.port), timeout=3) as raw:
        with context.wrap_socket(raw, server_hostname=address.hostname) as tls:
            return tls.getpeercert(binary_form=True)


@pytest.fixture
def endpoints(tmp_path, request):
    scheme, validity, client_renewal = request.param if isinstance(request.param, tuple) else (request.param, 30, True)
    server_dir, client_dir = tmp_path / "server", tmp_path / "client"
    params, root_key, root, server_key = pki(server_dir, seconds=validity)
    client_dir.mkdir()
    client_key, _ = generate_keys()
    (client_dir / "rootCA.pem").write_bytes(serialize_cert(root))
    (client_dir / "client.key").write_bytes(serialize_pri_key(client_key))
    (client_dir / "client.crt").write_bytes(
        issue(root_key, root, client_key, "site-1", validity if client_renewal else 3600)
    )
    client_params = {
        "ca_cert": str(client_dir / "rootCA.pem"),
        "client_cert": str(client_dir / "client.crt"),
        "client_key": str(client_dir / "client.key"),
        "connection_security": "mtls",
    }
    # A server's outgoing gRPC connection uses its own endpoint pair too.
    params.update(client_cert=params["server_cert"], client_key=params["server_key"])
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    url = f"{scheme}://localhost:{port}"
    ctx = mp.get_context("spawn")
    processes = []
    peers = []
    try:
        for name, credentials, directory in (("server", params, server_dir), ("site-1", client_params, client_dir)):
            commands, results = ctx.Queue(), ctx.Queue()
            process = ctx.Process(
                target=_endpoint,
                args=(name, url, credentials, commands, results, str(directory), name == "server" or client_renewal),
            )
            process.start()
            processes.append(process)
            peers.append((commands, results))
            assert _get(results)["ready"] == process.pid
        deadline = time.time() + 20
        while time.time() < deadline:
            peers[1][0].put("status")
            if _get(peers[1][1])["connections"]:
                break
            time.sleep(0.1)
        peers[1][0].put("request")
        assert _get(peers[1][1])["payload_ok"]
        yield peers, params, client_params, root_key, root, server_key, client_key, url
    finally:
        for commands, _ in peers:
            commands.put("stop")
        failures = []
        for process in processes:
            process.join(timeout=10)
            if process.is_alive():
                failures.append(f"process {process.pid} did not stop cleanly")
                process.terminate()
                process.join(timeout=5)
            elif process.exitcode != 0:
                failures.append(f"process {process.pid} exited with {process.exitcode}")
        assert not failures, failures


@pytest.mark.parametrize("endpoints", ["stcp", "satcp", "https", "grpcs", "agrpcs"], indirect=True)
@pytest.mark.parametrize(
    "request_type, renew_roles",
    [
        ("request", ("server", "client")),
        ("plain_request", ("server", "client")),
        ("encrypted_request", ("server", "client")),
        ("plain_request", ("server",)),
        ("plain_request", ("client",)),
    ],
)
def test_live_same_key_renewal_preserves_inflight_request_and_processes(endpoints, request_type, renew_roles):
    peers, server, client, root_key, root, server_key, client_key, url = endpoints
    initial = []
    for commands, results in peers:
        commands.put("status")
        initial.append(_get(results))
    # Renewal happens while a response is being produced; no application-level retry.
    peers[1][0].put(request_type)
    time.sleep(0.5)
    for params, key, name, role in (
        (server, server_key, "localhost", "server"),
        (client, client_key, "site-1", "client"),
    ):
        if role not in renew_roles:
            continue
        path = Path(params[f"{role}_cert"])
        candidate = path.with_suffix(".next")
        candidate.write_bytes(issue(root_key, root, key, name, 240))
        candidate.replace(path)
    reply = _get(peers[1][1])
    assert reply["payload_ok"], reply
    for i, (commands, results) in enumerate(peers):
        renewed = ("server", "client")[i] in renew_roles
        deadline = time.time() + 20
        while True:
            commands.put("status")
            current = _get(results)
            if (not renewed or current["status"][0]["last_observed"]) and current["connections"]:
                break
            assert time.time() < deadline, current
            time.sleep(0.2)
        assert current["pid"] == initial[i]["pid"]
        assert (current["status"][0]["fingerprint"] != initial[i]["status"][0]["fingerprint"]) == renewed
        assert current["connections"] == initial[i]["connections"]
        assert current["listeners"] == initial[i]["listeners"]
    if "server" in renew_roles:
        presented = x509.load_der_x509_certificate(_server_certificate(url, client))
        assert presented.fingerprint(hashes.SHA256()) == x509.load_pem_x509_certificate(
            Path(server["server_cert"]).read_bytes()
        ).fingerprint(hashes.SHA256())
    # Cross the original expiry without replacing the established TLS session.
    while time.time() <= initial[0]["status"][0]["expires"] + 1:
        time.sleep(0.2)
    peers[1][0].put(request_type)
    assert _get(peers[1][1])["payload_ok"]
    if request_type == "plain_request" and len(renew_roles) == 2:
        peers[1][0].put("disconnect")
        assert _get(peers[1][1])["disconnected"]
        expected_cert = x509.load_pem_x509_certificate(Path(client["client_cert"]).read_bytes()).public_bytes(
            serialization.Encoding.DER
        )
        deadline = time.time() + 20
        while True:
            current = []
            for commands, results in peers:
                commands.put("status")
                current.append(_get(results))
            if (
                all(
                    state["connections"] and set(state["connections"]).isdisjoint(initial[i]["connections"])
                    for i, state in enumerate(current)
                )
                and expected_cert in current[0]["peer_certs"]
            ):
                break
            assert time.time() < deadline, current
            time.sleep(0.2)
        for i, state in enumerate(current):
            assert state["listeners"] == initial[i]["listeners"]
        peers[1][0].put("plain_request")
        reply = _get(peers[1][1])
        assert reply["payload_ok"], reply


@pytest.mark.parametrize("endpoints", ["stcp", "satcp", "https", "grpcs", "agrpcs"], indirect=True)
def test_expiry_preserves_session_but_rejects_fresh_handshake(endpoints):
    peers, server, client, root_key, root, server_key, client_key, url = endpoints
    peers[0][0].put("status")
    initial = _get(peers[0][1])
    while time.time() <= initial["status"][0]["expires"] + 1:
        time.sleep(0.2)
    peers[1][0].put("plain_request")
    assert _get(peers[1][1])["payload_ok"]
    peers[0][0].put("status")
    expired = _get(peers[0][1])
    assert expired["connections"] == initial["connections"]
    assert expired["listeners"] == initial["listeners"]
    # Use a valid client certificate to isolate rejection of the expired server.
    Path(client["client_cert"]).write_bytes(issue(root_key, root, client_key, "site-1", 240))
    with pytest.raises(ssl.SSLCertVerificationError, match="expired"):
        _server_certificate(url, client)
    path = Path(server["server_cert"])
    candidate = path.with_suffix(".next")
    candidate.write_bytes(issue(root_key, root, server_key, "localhost", 240))
    candidate.replace(path)
    presented = x509.load_der_x509_certificate(_server_certificate(url, client))
    assert presented.fingerprint(hashes.SHA256()) == x509.load_pem_x509_certificate(path.read_bytes()).fingerprint(
        hashes.SHA256()
    )
    peers[0][0].put("status")
    recovered = _get(peers[0][1])
    assert recovered["pid"] == initial["pid"]
    assert recovered["listeners"] == initial["listeners"]


@pytest.mark.parametrize(
    "endpoints", [(scheme, 30, False) for scheme in ("stcp", "satcp", "https", "grpcs", "agrpcs")], indirect=True
)
def test_renewal_disabled_client_refreshes_expired_peer_certificate(endpoints):
    peers, server, _, root_key, root, key, _, _ = endpoints
    peers[1][0].put("cache_server")
    old = _get(peers[1][1])["certificate"]
    expiry = x509.load_pem_x509_certificate(old).not_valid_after_utc.timestamp()
    initial = []
    for commands, results in peers:
        commands.put("status")
        initial.append(_get(results))
    assert not initial[1]["status"], "client renewal must be disabled"
    path = Path(server["server_cert"])
    candidate = path.with_suffix(".next")
    renewed = issue(root_key, root, key, "localhost", 240)
    candidate.write_bytes(renewed)
    candidate.replace(path)
    while time.time() <= expiry + 2:
        time.sleep(0.2)
    # No previous encrypted reply: cached decryption keys cannot mask stale-cert validation.
    peers[1][0].put("encrypted_request")
    assert _get(peers[1][1])["payload_ok"]
    peers[1][0].put("cache_server")
    assert _get(peers[1][1])["certificate"] == renewed
    for i, (commands, results) in enumerate(peers):
        commands.put("status")
        assert _get(results)["pid"] == initial[i]["pid"]


@pytest.mark.parametrize("endpoints", ["stcp", "satcp", "https", "grpcs", "agrpcs"], indirect=True)
@pytest.mark.parametrize("version", [ssl.TLSVersion.TLSv1_2, ssl.TLSVersion.TLSv1_3])
def test_transport_rejects_missing_client_certificate(endpoints, version):
    peers, server, *_, url = endpoints
    peers[0][0].put("tls_debug")
    assert _get(peers[0][1])["tls_debug"]
    address = urlparse(url)
    context = ssl.create_default_context(cafile=server["ca_cert"])
    context.minimum_version = context.maximum_version = version
    context.set_alpn_protocols(["h2", "http/1.1"])
    with pytest.raises(ssl.SSLError) as rejected:
        with socket.create_connection((address.hostname, address.port), timeout=5) as sock:
            with context.wrap_socket(sock, server_hostname="localhost", suppress_ragged_eofs=False) as tls:
                # TLS 1.3 can return locally before the server rejects the empty
                # client Certificate message. Read the alert, not just handshake status.
                tls.recv(1)
    if rejected.value.reason == "UNEXPECTED_EOF_WHILE_READING":
        # EOF alone is not proof of mTLS rejection. Require the server TLS
        # implementation's explicit missing-client-certificate failure too.
        deadline = time.time() + 5
        evidence = ""
        while time.time() < deadline:
            evidence = (Path(server["server_cert"]).parent / "tls-errors.log").read_text()
            if "PEER_DID_NOT_RETURN_A_CERTIFICATE" in evidence:
                break
            time.sleep(0.1)
        assert "PEER_DID_NOT_RETURN_A_CERTIFICATE" in evidence
    else:
        assert rejected.value.reason in {"TLSV13_ALERT_CERTIFICATE_REQUIRED", "SSLV3_ALERT_HANDSHAKE_FAILURE"}
