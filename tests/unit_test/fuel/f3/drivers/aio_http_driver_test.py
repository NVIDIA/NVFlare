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
import base64
import os
import threading
from types import SimpleNamespace

import pytest
from aiohttp import web

from nvflare.fuel.f3.connection import ConnState
from nvflare.fuel.f3.drivers.aio_context import AioContext
from nvflare.fuel.f3.drivers.aio_http_driver import AioHttpDriver
from nvflare.fuel.f3.drivers.connector_info import ConnectorInfo, Mode
from nvflare.fuel.f3.drivers.net_utils import get_ssl_context
from nvflare.lighter.utils import Identity, generate_cert, generate_keys, serialize_cert, serialize_pri_key


@pytest.fixture
def tls_params(tmp_path):
    ca_key, ca_pub = generate_keys()
    ca = Identity("rootCA")
    ca_cert = generate_cert(ca, ca, ca_key, ca_pub, ca=True)
    ca_path = tmp_path / "rootCA.pem"
    ca_path.write_bytes(serialize_cert(ca_cert))
    params = {"scheme": "https", "ca_cert": str(ca_path)}
    for role in ("server", "client"):
        key, pub = generate_keys()
        cert = generate_cert(Identity(role), ca, ca_key, pub, server_default_host="localhost")
        for kind, data in (("cert", serialize_cert(cert)), ("key", serialize_pri_key(key))):
            path = tmp_path / f"{role}.{kind}"
            path.write_bytes(data)
            params[f"{role}_{kind}"] = str(path)
    return params


@pytest.mark.parametrize("bypass", [False, True], ids=["authenticated_proxy", "no_proxy"])
def test_https_proxy_preserves_mtls_and_frames(bypass, tls_params, tmp_path, monkeypatch):
    for name in list(os.environ):
        if name.lower().endswith("_proxy"):
            monkeypatch.delenv(name)
    monkeypatch.setenv("NETRC", str(tmp_path / "absent.netrc"))
    monkeypatch.setenv("no_proxy", "localhost" if bypass else "")

    async def run():
        context = AioContext("https_proxy_test")
        context.loop = asyncio.get_running_loop()
        context.ready.set()
        monkeypatch.setattr(AioContext, "get_global_context", lambda: context)
        received, peer_names, requests, tunnels = [], [], [], []

        async def echo(request):
            peer = request.transport.get_extra_info("peercert")
            peer_names.append(peer["subject"][0][0][1])
            ws = web.WebSocketResponse()
            await ws.prepare(request)
            message = await ws.receive(timeout=5)
            await ws.send_bytes(message.data)
            await ws.close()
            return ws

        app = web.Application()
        app.router.add_get("/f3", echo)
        runner = web.AppRunner(app, access_log=None)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0, ssl_context=get_ssl_context(tls_params, ssl_server=True))
        await site.start()
        server_port = site._server.sockets[0].getsockname()[1]

        async def tunnel(reader, writer):
            upstream = None
            try:
                requests.append(await reader.readuntil(b"\r\n\r\n"))
                remote_reader, upstream = await asyncio.open_connection("127.0.0.1", server_port)
                writer.write(b"HTTP/1.1 200 Connection Established\r\n\r\n")
                await writer.drain()

                async def relay(source, destination):
                    while data := await source.read(65536):
                        destination.write(data)
                        await destination.drain()
                    destination.close()

                await asyncio.gather(relay(reader, upstream), relay(remote_reader, writer))
            finally:
                writer.close()
                if upstream:
                    upstream.close()

        proxy = await asyncio.start_server(
            lambda r, w: tunnels.append(asyncio.create_task(tunnel(r, w))), "127.0.0.1", 0
        )
        try:
            proxy_port = proxy.sockets[0].getsockname()[1]
            monkeypatch.setenv("wss_proxy", f"http://user:password@127.0.0.1:{proxy_port}")
            driver = AioHttpDriver()

            def state_change(conn):
                if conn.state == ConnState.CONNECTED:
                    assert conn.get_conn_properties()["peer_cn"] == "server"
                    peer_port = server_port if bypass else proxy_port
                    assert conn.get_conn_properties()["peer_addr"] == f"127.0.0.1:{peer_port}"
                    conn.register_frame_receiver(SimpleNamespace(process_frame=received.append))
                    conn.send_frame(b"proxy-test")

            driver.register_conn_monitor(SimpleNamespace(state_change=state_change))
            params = {**tls_params, "host": "localhost", "port": server_port}
            connector = ConnectorInfo("test", driver, params, Mode.ACTIVE, 0, 0, False, threading.Event())
            await asyncio.to_thread(driver.connect, connector)
            await asyncio.gather(*tunnels)
            assert received == [b"proxy-test"]
            assert peer_names == ["client"]
            assert not driver.connections
            if bypass:
                assert requests == []
            else:
                (request,) = requests
                lines = request.split(b"\r\n")
                assert lines[0] == f"CONNECT localhost:{server_port} HTTP/1.1".encode()
                headers = dict(line.split(b": ", 1) for line in lines[1:] if line)
                assert headers[b"Proxy-Authorization"] == b"Basic " + base64.b64encode(b"user:password")
        finally:
            proxy.close()
            await proxy.wait_closed()
            await runner.cleanup()
            for task in tunnels:
                task.cancel()
            await asyncio.gather(*tunnels, return_exceptions=True)

    asyncio.run(asyncio.wait_for(run(), timeout=15))
