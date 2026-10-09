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
import threading
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nvflare.fuel.f3.drivers.aio_conn import AioConnection
from nvflare.fuel.f3.drivers.aio_context import AioContext
from nvflare.fuel.f3.drivers.aio_tcp_driver import AioTcpDriver
from nvflare.fuel.f3.drivers.connector_info import ConnectorInfo, Mode


def test_shutdown_runs_on_aio_loop_and_is_idempotent(monkeypatch):
    aio_ctx = AioContext("aio_tcp_driver_test")
    loop_thread = threading.Thread(target=aio_ctx.run_aio_loop)
    loop_thread.start()
    assert aio_ctx.ready.wait(timeout=5)

    monkeypatch.setattr(AioContext, "get_global_context", lambda: aio_ctx)
    driver = AioTcpDriver()
    calls = []

    class _Server:
        def close(self):
            calls.append(("server.close", threading.get_ident()))

        async def wait_closed(self):
            calls.append(("server.wait_closed", threading.get_ident()))

    def close_all():
        calls.append(("close_all", threading.get_ident()))

    monkeypatch.setattr(driver, "close_all", close_all)
    driver.server = _Server()

    try:
        driver.shutdown().result(timeout=5)
        driver.shutdown().result(timeout=5)
    finally:
        aio_ctx.stop_aio_loop()
        loop_thread.join(timeout=5)

    assert not loop_thread.is_alive()
    assert calls == [
        ("server.close", loop_thread.ident),
        ("close_all", loop_thread.ident),
        ("server.wait_closed", loop_thread.ident),
        ("close_all", loop_thread.ident),
    ]
    assert driver.server is None


@pytest.mark.parametrize("target", ["connection", "listener"])
def test_shutdown_schedules_transport_close_on_owning_event_loop(target):
    context = MagicMock()
    if target == "connection":
        connection = AioConnection(MagicMock(), context, MagicMock(), None)
        connection.writer = transport = MagicMock()
        close = connection.close
    else:
        with patch("nvflare.fuel.f3.drivers.aio_tcp_driver.AioContext.get_global_context", return_value=context):
            driver = AioTcpDriver()
        driver.server = transport = MagicMock()
        transport.wait_closed = AsyncMock()
        close = driver.shutdown

    close()
    transport.close.assert_not_called()
    if target == "connection":
        loop = context.get_event_loop.return_value
        loop.call_soon_threadsafe.assert_called_once()
        loop.call_soon_threadsafe.call_args.args[0]()
    else:
        context.run_coro.assert_called_once()
        asyncio.run(context.run_coro.call_args.args[0])
        transport.wait_closed.assert_awaited_once()
    transport.close.assert_called_once()


def test_connection_registered_after_shutdown_is_closed():
    async def shutdown():
        context = MagicMock()
        context.get_event_loop.return_value = asyncio.get_running_loop()
        context.run_coro.side_effect = asyncio.create_task
        with patch.object(AioContext, "get_global_context", return_value=context):
            driver = AioTcpDriver()
        driver.connector = ConnectorInfo("test", driver, {}, Mode.PASSIVE, 0, 0, False, threading.Event())
        driver.register_conn_monitor(MagicMock())
        accepted, register, finished = asyncio.Event(), asyncio.Event(), asyncio.Event()
        writers = []

        async def delayed_registration(reader, writer):
            writers.append(writer)
            accepted.set()
            await register.wait()
            try:
                await driver._create_connection(reader, writer)
            finally:
                finished.set()

        server = await asyncio.start_server(delayed_registration, "127.0.0.1", 0)
        driver.server = server
        reader, peer = await asyncio.open_connection(*server.sockets[0].getsockname())
        try:
            await asyncio.wait_for(accepted.wait(), 2)
            completion = driver.shutdown()
            await asyncio.sleep(0)
            register.set()
            assert await asyncio.wait_for(reader.read(1), 2) == b""
            await asyncio.wait_for(server.wait_closed(), 2)
            await asyncio.wait_for(finished.wait(), 2)
            await asyncio.wait_for(completion, 2)
            assert not driver.connections
        finally:
            register.set()
            for writer in writers:
                writer.close()
                await writer.wait_closed()
            peer.close()
            await peer.wait_closed()
            server.close()
            await asyncio.wait_for(server.wait_closed(), 2)
            await asyncio.wait_for(finished.wait(), 2)

    asyncio.run(shutdown())


def test_listener_created_during_shutdown_is_closed():
    async def shutdown():
        context = MagicMock()
        context.get_event_loop.return_value = asyncio.get_running_loop()
        context.run_coro.side_effect = asyncio.create_task
        with patch.object(AioContext, "get_global_context", return_value=context):
            driver = AioTcpDriver()
        driver.connector = ConnectorInfo(
            "test", driver, {"scheme": "atcp"}, Mode.PASSIVE, 0, 0, False, threading.Event()
        )
        created, release = asyncio.Event(), asyncio.Event()
        servers = []
        start_server = asyncio.start_server

        async def delayed_start(*args, **kwargs):
            server = await start_server(*args, **kwargs)
            servers.append(server)
            created.set()
            await release.wait()
            return server

        with patch.object(asyncio, "start_server", delayed_start):
            listener = asyncio.create_task(driver._tcp_listen("127.0.0.1", 0))
            try:
                await asyncio.wait_for(created.wait(), 2)
                sockets = servers[0].sockets
                await driver.shutdown()
                release.set()
                await asyncio.wait_for(asyncio.shield(listener), 2)
                assert all(sock.fileno() == -1 for sock in sockets)
            finally:
                release.set()
                listener.cancel()
                await asyncio.gather(listener, return_exceptions=True)
                for server in servers:
                    server.close()
                    await server.wait_closed()

    asyncio.run(shutdown())
