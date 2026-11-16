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

"""Real HTTP(S) driver receives frames exceeding aiohttp's default 4 MiB limit."""

import asyncio
import hashlib
import os
import threading
from types import SimpleNamespace

import pytest
from aiohttp import web

from nvflare.fuel.f3.connection import ConnState
from nvflare.fuel.f3.drivers.aio_context import AioContext
from nvflare.fuel.f3.drivers.aio_http_driver import MAX_FRAME_SIZE, AioHttpDriver
from nvflare.fuel.f3.drivers.connector_info import ConnectorInfo, Mode
from nvflare.fuel.f3.drivers.net_utils import get_ssl_context
from nvflare.lighter.utils import Identity, generate_cert, generate_keys, serialize_cert, serialize_pri_key


def _tls_params(tmp_path):
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


@pytest.mark.parametrize("secure", [False, True], ids=["http", "https_mtls"])
@pytest.mark.parametrize("size", [1024, 5 * 1024**2], ids=["small", "over_4mib"])
def test_driver_receives_server_frame(secure, size, tmp_path, monkeypatch):
    # Keep the test independent of the invoking machine's proxy settings.
    for name in list(os.environ):
        if name.lower().endswith("_proxy"):
            monkeypatch.delenv(name)
    monkeypatch.setenv("NETRC", str(tmp_path / "absent.netrc"))
    params = _tls_params(tmp_path) if secure else {"scheme": "http"}
    payload = b"frame-size-regression\x00" * (size // 22 + 1)
    payload = payload[:size]
    expected_digest = hashlib.sha256(payload).digest()

    async def run():
        context = AioContext("http_frame_size_test")
        context.loop = asyncio.get_running_loop()
        context.ready.set()
        monkeypatch.setattr(AioContext, "get_global_context", lambda: context)
        received, acknowledgments, callback_errors = [], [], []

        async def send(request):
            ws = web.WebSocketResponse(max_msg_size=MAX_FRAME_SIZE)
            await ws.prepare(request)
            try:
                await ws.send_bytes(payload)
                reply = await ws.receive(timeout=20)
                acknowledgments.append(reply.data)
            finally:
                await ws.close()
            return ws

        app = web.Application()
        app.router.add_get("/f3", send)
        runner = web.AppRunner(app, access_log=None)
        driver = None
        task = None
        try:
            await runner.setup()
            ssl_context = get_ssl_context(params, ssl_server=True) if secure else None
            site = web.TCPSite(runner, "127.0.0.1", 0, ssl_context=ssl_context)
            await site.start()
            port = site._server.sockets[0].getsockname()[1]
            driver = AioHttpDriver()

            def state_change(conn):
                if conn.state != ConnState.CONNECTED:
                    return

                def process_frame(frame):
                    try:
                        received.append(bytes(frame))
                        conn.send_frame(hashlib.sha256(frame).digest())
                    except Exception as ex:
                        callback_errors.append(ex)

                conn.register_frame_receiver(SimpleNamespace(process_frame=process_frame))

            driver.register_conn_monitor(SimpleNamespace(state_change=state_change))
            connector = ConnectorInfo(
                "test",
                driver,
                {**params, "host": "localhost", "port": port},
                Mode.ACTIVE,
                0,
                0,
                False,
                threading.Event(),
            )
            task = asyncio.create_task(asyncio.to_thread(driver.connect, connector))
            await asyncio.wait_for(asyncio.shield(task), timeout=30)
            assert not callback_errors
            assert received == [payload]
            assert acknowledgments == [expected_digest]
            assert not driver.connections
        finally:
            if driver:
                await driver._async_shutdown()
            if task:
                await asyncio.gather(task, return_exceptions=True)
            await runner.cleanup()

    asyncio.run(asyncio.wait_for(run(), timeout=60))
