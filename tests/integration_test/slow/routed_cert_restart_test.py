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

"""Encrypted messages recover when a peer changes keys behind a surviving relay."""

import multiprocessing
import time
import traceback

import pytest

from nvflare.fuel.f3.cellnet.core_cell import CoreCell
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey, ReturnCode
from nvflare.fuel.f3.message import Message
from nvflare.fuel.utils.network_utils import get_open_ports
from nvflare.lighter.utils import Identity, generate_cert, generate_keys, serialize_cert, serialize_pri_key

pytestmark = [pytest.mark.slow, pytest.mark.timeout(90)]
_TARGET = "relay-1.site-1"
_IDENTITIES = {"server": "server", "relay-1": "relay-1", _TARGET: "site-1"}


def _run_peer(fqcn, root_url, parent_url, credentials, ready, stop):
    cell = None
    try:
        cell = CoreCell(
            fqcn,
            root_url,
            secure=True,
            credentials=credentials,
            parent_url=parent_url,
            parent_resources={"connection_security": "mtls"},
            auth_identity_map=_IDENTITIES,
            max_timeout=3,
        )
        cell.connector_manager.adhoc_allowed = False
        if fqcn == "relay-1":
            cell.make_internal_listener("stcp", {"host": "localhost", "connection_security": "mtls"})

        def echo(request):
            return Message(
                headers={"request_route": [hop[0] for hop in request.get_header(MessageHeaderKey.ROUTE)]},
                payload=request.payload,
            )

        cell.register_request_cb("restart", "echo", echo)
        cell.start()
        upstream = "relay-1" if parent_url else "server"
        deadline = time.monotonic() + 10
        while not cell.is_cell_connected(upstream):
            if time.monotonic() >= deadline:
                raise RuntimeError(f"{fqcn} did not connect to {upstream}")
            time.sleep(0.05)
        ready.put({"url": cell.get_internal_listener_url()})
        stop.wait(60)
    except Exception:
        ready.put({"error": traceback.format_exc()})
        raise
    finally:
        if cell:
            cell.stop()


def test_encrypted_requests_recover_after_routed_peer_key_replacement(tmp_path):
    context = multiprocessing.get_context("spawn")
    root_key, root_pub = generate_keys()
    root = generate_cert(Identity("root"), Identity("root"), root_key, root_pub, ca=True)
    root_path = tmp_path / "rootCA.pem"
    root_path.write_bytes(serialize_cert(root))

    def credentials(name, generation=""):
        key, pub = generate_keys()
        cert = generate_cert(Identity(name), root.subject, root_key, pub, server_default_host="localhost")
        cert_path, key_path = tmp_path / f"{name}{generation}.crt", tmp_path / f"{name}{generation}.key"
        cert_path.write_bytes(serialize_cert(cert))
        key_path.write_bytes(serialize_pri_key(key))
        return {
            "ca_cert": str(root_path),
            "server_cert": str(cert_path),
            "server_key": str(key_path),
            "client_cert": str(cert_path),
            "client_key": str(key_path),
            "connection_security": "mtls",
        }, cert

    root_url = f"stcp://localhost:{get_open_ports(1)[0]}"
    server = CoreCell(
        "server",
        root_url,
        secure=True,
        credentials=credentials("server")[0],
        auth_identity_map=_IDENTITIES,
        max_timeout=3,
    )
    server.connector_manager.adhoc_allowed = False
    connections = []
    server.set_cell_connected_cb(lambda agent: connections.append(("ready", agent.info.fqcn)))
    server.set_cell_disconnected_cb(lambda agent: connections.append(("disconnected", agent.info.fqcn)))
    processes = []

    def start(fqcn, creds, parent_url=None):
        ready, stop = context.Queue(), context.Event()
        process = context.Process(
            target=_run_peer,
            args=(fqcn, None if parent_url else root_url, parent_url, creds, ready, stop),
        )
        process.start()
        processes.append((process, stop))
        result = ready.get(timeout=15)
        assert "error" not in result, result
        return process, stop, result["url"]

    def request():
        return server.send_request("restart", "echo", _TARGET, Message(payload=b"hello"), timeout=2, secure=True)

    def check_reply(reply):
        assert reply.get_header(MessageHeaderKey.RETURN_CODE) == ReturnCode.OK
        assert reply.payload == b"hello"
        assert reply.get_header("request_route") == ["server", "relay-1", _TARGET]
        assert [hop[0] for hop in reply.get_header(MessageHeaderKey.ROUTE)] == [_TARGET, "relay-1", "server"]

    try:
        server.start()
        relay, _, parent_url = start("relay-1", credentials("relay-1")[0])
        old_credentials, old_cert = credentials("site-1")
        peer, peer_stop, _ = start(_TARGET, old_credentials, parent_url)
        check_reply(request())
        assert server.credential_manager.get_certificate(_TARGET) == serialize_cert(old_cert)
        assert connections == [("ready", "relay-1")]

        peer_stop.set()
        peer.join(timeout=10)
        assert peer.exitcode == 0
        new_credentials, new_cert = credentials("site-1", "-replacement")
        assert old_cert.public_key().public_numbers() != new_cert.public_key().public_numbers()
        start(_TARGET, new_credentials, parent_url)
        assert relay.is_alive()
        assert connections == [("ready", "relay-1")]
        assert server.credential_manager.get_certificate(_TARGET) == serialize_cert(old_cert)

        # A request during recovery may fail. Real reciprocal certificate exchange
        # must repair the warm cache without reconnecting the server or relay.
        deadline = time.monotonic() + 10
        while True:
            reply = request()
            if reply.get_header(MessageHeaderKey.RETURN_CODE) == ReturnCode.OK:
                break
            assert time.monotonic() < deadline, reply.headers
        check_reply(reply)
        assert server.credential_manager.get_certificate(_TARGET) == serialize_cert(new_cert)
        check_reply(request())
        assert connections == [("ready", "relay-1")]
    finally:
        for process, stop in reversed(processes):
            stop.set()
            process.join(timeout=10)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
        server.stop()
    assert all(process.exitcode == 0 for process, _ in processes)
