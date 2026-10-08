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

"""Regression coverage for transport lifecycle fixes, independent of certificate renewal."""

import asyncio
import socket
import ssl
import threading
from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import grpc
import pytest
from aiohttp import web

from nvflare.fuel.f3.comm_error import CommError
from nvflare.fuel.f3.connection import ConnState
from nvflare.fuel.f3.drivers import aio_grpc_driver, grpc_driver
from nvflare.fuel.f3.drivers.aio_context import AioContext
from nvflare.fuel.f3.drivers.aio_http_driver import AioHttpDriver
from nvflare.fuel.f3.drivers.aio_tcp_driver import AioTcpDriver
from nvflare.fuel.f3.drivers.connector_info import ConnectorInfo, Mode
from nvflare.fuel.f3.drivers.file_driver import FileDriver
from nvflare.fuel.f3.drivers.tcp_driver import TcpDriver
from nvflare.fuel.f3.endpoint import Endpoint
from nvflare.fuel.f3.sfm.conn_manager import ConnManager


def test_close_all_does_not_hold_driver_lock():
    driver = TcpDriver()

    def close():
        assert driver.conn_lock.acquire(blocking=False)
        driver.conn_lock.release()
        driver.connections.clear()

    driver.connections["test"] = MagicMock(close=close)
    driver.close_all()
    assert not driver.connections


@pytest.mark.parametrize("operation", ["stop", "remove"])
def test_connector_shutdown_does_not_hold_manager_lock(operation):
    manager = ConnManager(Endpoint("test"))
    driver = TcpDriver()
    handle = manager.add_connector(driver, {"scheme": "tcp", "url": "tcp://localhost:1234"}, Mode.PASSIVE)

    def shutdown():
        assert manager.lock.acquire(blocking=False)
        manager.lock.release()

    try:
        with patch.object(driver, "shutdown", side_effect=shutdown):
            if operation == "remove":
                manager.remove_connector(handle)
            manager.stop()
    finally:
        manager.conn_mgr_executor.shutdown()
        manager.frame_mgr_executor.shutdown()


def test_connector_added_during_stop_is_rejected():
    manager = ConnManager(Endpoint("test"))
    driver = TcpDriver()
    params = {"scheme": "tcp", "url": "tcp://localhost:1234"}
    handle = manager.add_connector(driver, params, Mode.PASSIVE)
    manager.started = True
    late_driver = MagicMock()

    def shutdown():
        # stop() has taken its snapshot but has not shut down the executor yet.
        with pytest.raises(CommError, match="stopped") as exc:
            manager.add_connector(late_driver, params, Mode.ACTIVE)
        assert exc.value.code == CommError.CLOSED

    try:
        with patch.object(driver, "shutdown", side_effect=shutdown):
            manager.stop()
        assert list(manager.connectors) == [handle]
        late_driver.connect.assert_not_called()
    finally:
        # Release the retry loop if the regression permits the late connector.
        for connector in manager.connectors.values():
            connector.stopped.set()
        manager.conn_mgr_executor.shutdown()
        manager.frame_mgr_executor.shutdown()


def test_grpc_bind_failure_stops_server_and_propagates():
    connector = MagicMock(params={"scheme": "grpc", "host": "localhost", "port": 1234})
    with patch.object(grpc_driver.grpc, "server") as create:
        create.return_value.add_insecure_port.side_effect = RuntimeError("address unavailable")
        with pytest.raises(CommError, match="address unavailable"):
            grpc_driver.Server(MagicMock(), connector, 1, [])
        create.return_value.stop.assert_called_once_with(grace=0)


@pytest.mark.parametrize("asynchronous", [False, True])
def test_grpc_server_shutdown_is_repeatable(asynchronous):
    server = SimpleNamespace(grpc_server=MagicMock(), grpc_server_stop_grace=0, logger=MagicMock())
    native_server = server.grpc_server
    if asynchronous:
        native_server.stop = AsyncMock()

        async def shutdown():
            await aio_grpc_driver.Server.shutdown(server)
            await aio_grpc_driver.Server.shutdown(server)

        asyncio.run(shutdown())
        assert native_server.stop.await_count == 2
    else:
        grpc_driver.Server.shutdown(server)
        grpc_driver.Server.shutdown(server)
        assert native_server.stop.call_count == 2


def test_http_server_shutdown_is_repeatable():
    async def shutdown():
        context = MagicMock()
        context.get_event_loop.return_value = asyncio.get_running_loop()
        with patch("nvflare.fuel.f3.drivers.aio_http_driver.AioContext.get_global_context", return_value=context):
            driver = AioHttpDriver()
        driver.app = web.Application()
        driver.runner = runner = web.AppRunner(driver.app)
        await runner.setup()
        driver.site = web.TCPSite(runner, "127.0.0.1", 0)
        await driver.site.start()
        try:
            await driver._async_shutdown()
            await driver._async_shutdown()
            assert driver.stop_event.done()
            assert not runner.sites
        finally:
            await runner.cleanup()

    asyncio.run(shutdown())


@pytest.fixture
def listener_context(monkeypatch):
    context = AioContext("listener_shutdown_test")
    thread = threading.Thread(target=context.run_aio_loop, daemon=True)
    thread.start()
    assert context.ready.wait(5)
    monkeypatch.setattr(AioContext, "get_global_context", lambda: context)
    try:
        yield context
    finally:
        context.stop_aio_loop()
        thread.join(5)
        assert not thread.is_alive()


def _start_listener(driver, connector):
    errors = []

    def listen():
        try:
            driver.listen(connector)
        except Exception as ex:
            errors.append(ex)

    listener = threading.Thread(target=listen, daemon=True)
    listener.start()
    return listener, errors


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("pause_at", ["construction", "start"])
def test_grpc_listener_starting_during_shutdown_exits(asynchronous, pause_at, listener_context, monkeypatch):
    module = aio_grpc_driver if asynchronous else grpc_driver
    driver = module.AioGrpcDriver() if asynchronous else module.GrpcDriver()
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    params = {"scheme": "grpc", "host": "127.0.0.1", "port": port}
    connector = ConnectorInfo("test", driver, params, Mode.PASSIVE, 0, 0, False, threading.Event())
    entered, release = threading.Event(), threading.Event()
    server_class = module.Server
    servers = []

    def create_server(*args, **kwargs):
        server = server_class(*args, **kwargs)
        servers.append(server)
        if pause_at == "construction":
            entered.set()
            assert release.wait(5)
        else:
            start = server.grpc_server.start

            def delayed_start():
                entered.set()
                assert release.wait(5)
                start()

            async def delayed_async_start():
                await start()
                entered.set()
                assert await asyncio.to_thread(release.wait, 5)

            monkeypatch.setattr(server.grpc_server, "start", delayed_async_start if asynchronous else delayed_start)
        return server

    monkeypatch.setattr(module, "Server", create_server)
    listener, errors = _start_listener(driver, connector)
    try:
        assert entered.wait(5)
        connector.stopped.set()
        driver.shutdown()
        release.set()
        listener.join(3)
        assert not listener.is_alive(), "listener started after shutdown and never exited"
        assert not errors
        with socket.socket() as probe:
            probe.settimeout(1)
            assert probe.connect_ex(("127.0.0.1", port)) != 0
    finally:
        release.set()
        for server in servers:
            if asynchronous:
                listener_context.run_coro(server.shutdown()).result(5)
            else:
                server.shutdown()
        listener.join(5)
        assert not listener.is_alive()


@pytest.mark.parametrize("pause_at", ["before_bind", "after_bind"])
def test_http_listener_starting_during_shutdown_releases_socket(pause_at, listener_context, monkeypatch):
    driver = AioHttpDriver()
    params = {"scheme": "http", "host": "127.0.0.1", "port": 0}
    connector = ConnectorInfo("test", driver, params, Mode.PASSIVE, 0, 0, False, threading.Event())
    entered, release = threading.Event(), threading.Event()
    create_server = driver.loop.create_server
    servers, sockets = [], []

    async def delayed_create(*args, **kwargs):
        if pause_at == "before_bind":
            entered.set()
            assert await asyncio.to_thread(release.wait, 5)
        server = await create_server(*args, **kwargs)
        servers.append(server)
        sockets.extend(server.sockets)
        if pause_at == "after_bind":
            entered.set()
            assert await asyncio.to_thread(release.wait, 5)
        return server

    monkeypatch.setattr(driver.loop, "create_server", delayed_create)
    listener, errors = _start_listener(driver, connector)
    try:
        assert entered.wait(5)
        connector.stopped.set()
        listener_context.run_coro(driver._async_shutdown()).result(5)
        release.set()
        listener.join(3)
        assert not listener.is_alive()
        assert not errors
        assert sockets and all(sock.fileno() == -1 for sock in sockets)
        assert driver.site is None
        assert driver.runner is None
    finally:
        release.set()
        listener.join(5)

        async def cleanup():
            await driver._async_shutdown()
            for server in servers:
                server.close()
                await server.wait_closed()

        listener_context.run_coro(cleanup()).result(5)
        assert not listener.is_alive()


def test_overlapping_http_shutdowns_wait_for_cleanup(listener_context, monkeypatch):
    driver = AioHttpDriver()
    params = {"scheme": "http", "host": "127.0.0.1", "port": 0}
    connector = ConnectorInfo("test", driver, params, Mode.PASSIVE, 0, 0, False, threading.Event())
    started = threading.Event()
    cleaning, release = asyncio.Event(), asyncio.Event()
    start_site, cleanup_runner = web.TCPSite.start, web.AppRunner.cleanup

    async def start(site):
        await start_site(site)
        started.set()

    async def cleanup(runner):
        cleaning.set()
        await release.wait()
        await cleanup_runner(runner)

    monkeypatch.setattr(web.TCPSite, "start", start)
    monkeypatch.setattr(web.AppRunner, "cleanup", cleanup)
    listener, errors = _start_listener(driver, connector)

    async def overlap():
        first = asyncio.create_task(driver._async_shutdown())
        second = None
        try:
            await asyncio.wait_for(cleaning.wait(), 2)
            second = asyncio.create_task(driver._async_shutdown())
            # Let the second shutdown run while the first is paused in cleanup.
            await asyncio.sleep(0)
            assert not driver.stop_event.done(), "listener released before runner cleanup completed"
            assert listener.is_alive()
            assert not first.done()
            assert not second.done()
        finally:
            release.set()
            await asyncio.gather(*[task for task in (first, second) if task is not None])

    try:
        assert started.wait(5)
        connector.stopped.set()
        listener_context.run_coro(overlap()).result(5)
        listener.join(3)
        assert not listener.is_alive()
        assert not errors
        assert driver.stop_event.done()
    finally:
        listener_context.get_event_loop().call_soon_threadsafe(release.set)
        driver.shutdown()
        listener.join(5)
        assert not listener.is_alive()


@pytest.mark.parametrize("phase", ["handshake", "read"])
def test_http_connector_stop_cancels_inflight_work_before_executor_join(phase, listener_context):
    driver = AioHttpDriver()
    manager = ConnManager(Endpoint("http_shutdown_test"))
    entered = threading.Event()
    release = asyncio.Event()
    runner = None

    async def handler(request):
        if phase == "handshake":
            # Hold the HTTP upgrade open: close_all() has no connection to close.
            entered.set()
            await release.wait()
            return web.Response(status=503)
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        async for _message in ws:
            pass
        return ws

    async def setup():
        nonlocal runner
        app = web.Application()
        app.router.add_get("/f3", handler)
        runner = web.AppRunner(app, access_log=None)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        return next(iter(runner.addresses))[1]

    port = listener_context.run_coro(setup()).result(5)
    params = {"scheme": "http", "host": "127.0.0.1", "port": port, "url": f"http://127.0.0.1:{port}"}
    manager.add_connector(driver, params, Mode.ACTIVE)
    if phase == "read":
        notify = driver.add_connection

        def added(conn):
            admitted = notify(conn)
            entered.set()
            return admitted

        driver.add_connection = added
    manager.start()
    stopper = threading.Thread(target=manager.stop, daemon=True)
    try:
        assert entered.wait(5)
        stopper.start()
        stopper.join(3)
        assert not stopper.is_alive(), "HTTP connector is still blocking executor shutdown"
        assert not driver._connect_tasks
        assert not driver.connections
        listener_context.run_coro(driver._async_shutdown()).result(5)
    finally:

        async def cleanup():
            release.set()
            # Also release a broken implementation so the failed test cannot
            # leave a connection-manager executor alive in this pytest process.
            for task in list(driver._connect_tasks):
                task.cancel()
            await runner.cleanup()

        listener_context.run_coro(cleanup()).result(5)
        if stopper.ident is None:
            stopper.start()
        stopper.join(5)
        assert not stopper.is_alive()


def test_http_connect_scheduled_after_shutdown_does_not_open_a_session(listener_context):
    driver = AioHttpDriver()
    connector = ConnectorInfo(
        "test",
        driver,
        {"scheme": "http", "host": "127.0.0.1", "port": 1, "url": "http://127.0.0.1:1"},
        Mode.ACTIVE,
        0,
        0,
        False,
        threading.Event(),
    )
    listener_context.run_coro(driver._async_shutdown()).result(5)
    with patch("nvflare.fuel.f3.drivers.aio_http_driver.aiohttp.ClientSession") as session:
        driver.connect(connector)
    session.assert_not_called()


def test_http_upgrade_completed_after_stop_is_closed_before_admission(listener_context):
    driver = AioHttpDriver()
    driver.connector = MagicMock(stopped=threading.Event())
    driver.register_conn_monitor(MagicMock())
    websocket = MagicMock(close=AsyncMock())
    listener_context.run_coro(driver._async_shutdown()).result(5)
    listener_context.run_coro(driver._connection_handler(websocket)).result(5)
    websocket.close.assert_awaited_once()
    driver.conn_monitor.state_change.assert_not_called()
    assert not driver.connections


@pytest.mark.parametrize(
    "driver_class",
    [TcpDriver, AioTcpDriver, grpc_driver.GrpcDriver, aio_grpc_driver.AioGrpcDriver, AioHttpDriver, FileDriver],
)
def test_driver_shutdown_rejects_late_connection_without_unmatched_notification(driver_class, listener_context):
    driver = driver_class()
    monitor = MagicMock()
    driver.register_conn_monitor(monitor)
    conn = MagicMock(name="late_connection")
    conn.name = "late"
    conn.connector.stopped = threading.Event()

    def close():
        # Rejection must close outside the registry lock.
        assert driver.conn_lock.acquire(blocking=False)
        driver.conn_lock.release()

    conn.close.side_effect = close
    completion = driver.shutdown()
    if isinstance(completion, Future):
        completion.result(5)
    assert not driver.add_connection(conn)
    driver.close_connection(conn)
    conn.close.assert_called_once()
    monitor.state_change.assert_not_called()
    assert not driver.connections


def test_connection_close_notification_is_idempotent():
    driver = TcpDriver()
    states = []
    monitor = MagicMock()
    monitor.state_change.side_effect = lambda conn: states.append(conn.state)
    driver.register_conn_monitor(monitor)
    conn = MagicMock()
    conn.name = "normal"
    conn.connector.stopped = threading.Event()
    assert driver.add_connection(conn)
    driver.close_connection(conn)
    driver.close_connection(conn)
    assert states == [ConnState.CONNECTED, ConnState.CLOSED]


def test_manager_requests_all_shutdowns_and_waits_for_completion_before_join():
    manager = ConnManager(Endpoint("shutdown_order"))
    calls = []

    class Completion(Future):
        def result(self, timeout=None):
            assert calls[:2] == ["request_first", "request_second"]
            calls.append("completed")
            return super().result(timeout)

    completion = Completion()
    completion.set_result(None)
    for name in ("first", "second"):
        driver = MagicMock()

        def shutdown(name=name):
            calls.append(f"request_{name}")
            return completion if name == "first" else None

        driver.shutdown.side_effect = shutdown
        manager.add_connector(driver, {"scheme": "tcp", "url": "tcp://localhost:1"}, Mode.ACTIVE)
    with (
        patch.object(manager.conn_mgr_executor, "shutdown", side_effect=lambda wait: calls.append("join_connectors")),
        patch.object(manager.frame_mgr_executor, "shutdown", side_effect=lambda wait: calls.append("join_frames")),
    ):
        manager.stop()
    manager.conn_mgr_executor.shutdown()
    manager.frame_mgr_executor.shutdown()
    assert calls == ["request_first", "request_second", "completed", "join_connectors", "join_frames"]


def test_async_shutdown_waits_for_connect_finalizer(listener_context):
    driver = AioTcpDriver()
    connector = ConnectorInfo("test", driver, {}, Mode.ACTIVE, 0, 0, False, threading.Event())
    entered, finalizing = threading.Event(), threading.Event()
    release = asyncio.Event()

    async def connect():
        try:
            entered.set()
            await asyncio.Future()
        finally:
            finalizing.set()
            await release.wait()

    connection = listener_context.run_coro(driver._run_connect(connector, connect))
    completion = None
    try:
        assert entered.wait(5)
        completion = driver.shutdown()
        assert finalizing.wait(5)
        assert not completion.done(), "shutdown finished before the connect task's resource finalizer"
        assert not connection.done()
    finally:
        listener_context.get_event_loop().call_soon_threadsafe(release.set)
        if completion is not None:
            completion.result(5)
    assert connection.done()
    assert not driver._connect_tasks


@pytest.mark.parametrize("asynchronous", [False, True])
def test_tcp_shutdown_interrupts_pending_client_tls_handshake(asynchronous, listener_context, monkeypatch):
    driver = AioTcpDriver() if asynchronous else TcpDriver()
    manager = ConnManager(Endpoint("tcp_handshake_shutdown"))
    tls = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    tls.check_hostname = False
    tls.verify_mode = ssl.CERT_NONE
    module = "aio_tcp_driver" if asynchronous else "tcp_driver"
    monkeypatch.setattr(f"nvflare.fuel.f3.drivers.{module}.get_ssl_context", lambda *args, **kwargs: tls)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        listener.settimeout(5)
        port = listener.getsockname()[1]
        params = {"scheme": "stcp", "host": "127.0.0.1", "port": port, "url": f"stcp://localhost:{port}"}
        manager.add_connector(driver, params, Mode.ACTIVE)
        manager.start()
        peer, _ = listener.accept()
        stopper = threading.Thread(target=manager.stop, daemon=True)
        try:
            # Consume the ClientHello but never reply. There is no connection in
            # the driver's registry yet, so close_all alone cannot release it.
            peer.settimeout(5)
            assert peer.recv(4096)
            assert not driver.connections
            stopper.start()
            stopper.join(3)
            assert not stopper.is_alive(), "TLS negotiation blocked connector shutdown"
            assert peer.recv(1) == b""
            if asynchronous:
                assert not driver._connect_tasks
            else:
                assert not driver._pending_sockets
        finally:
            peer.close()
            if stopper.ident is None:
                stopper.start()
            stopper.join(5)
            assert not stopper.is_alive()


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("phase", ["before_admission", "read"])
def test_grpc_connector_shutdown_cannot_miss_late_admission(asynchronous, phase, listener_context, monkeypatch):
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    params = {"scheme": "grpc", "host": "127.0.0.1", "port": port, "url": f"grpc://127.0.0.1:{port}"}
    server = grpc_driver.GrpcDriver()
    server.register_conn_monitor(MagicMock())
    passive = ConnectorInfo("server", server, params, Mode.PASSIVE, 0, 0, False, threading.Event())
    listener, errors = _start_listener(server, passive)
    channel = grpc.insecure_channel(f"127.0.0.1:{port}")
    grpc.channel_ready_future(channel).result(5)
    channel.close()

    driver = aio_grpc_driver.AioGrpcDriver() if asynchronous else grpc_driver.GrpcDriver()
    manager = ConnManager(Endpoint("grpc_shutdown_test"))
    manager.add_connector(driver, params, Mode.ACTIVE)
    entered, release, requested = threading.Event(), threading.Event(), threading.Event()
    add_connection, shutdown = driver.add_connection, driver.shutdown
    stop_errors = []

    def delayed_admission(conn):
        entered.set()
        assert release.wait(5)
        return add_connection(conn)

    def request_shutdown():
        completion = shutdown()
        requested.set()
        return completion

    if phase == "before_admission":
        monkeypatch.setattr(driver, "add_connection", delayed_admission)
    else:
        session_class = aio_grpc_driver.AioStreamSession if asynchronous else grpc_driver.StreamConnection
        read_loop = session_class.read_loop

        def read(conn, messages):
            if conn.connector.driver is driver:
                entered.set()
            return read_loop(conn, messages)

        async def async_read(conn, messages):
            entered.set()
            return await read_loop(conn, messages)

        monkeypatch.setattr(session_class, "read_loop", async_read if asynchronous else read)
    monkeypatch.setattr(driver, "shutdown", request_shutdown)

    def stop():
        try:
            manager.stop()
        except Exception as ex:
            stop_errors.append(ex)

    manager.start()
    stopper = threading.Thread(target=stop, daemon=True)
    try:
        assert entered.wait(5)
        stopper.start()
        assert requested.wait(5)
        release.set()
        stopper.join(3)
        assert not stopper.is_alive(), "gRPC receive loop survived shutdown and blocked the connector join"
        assert not stop_errors
        assert not driver.connections
        assert not errors
    finally:
        # Release the old implementation too so a regression cannot pin pytest.
        release.set()
        driver.close_all()
        if stopper.ident is None:
            stopper.start()
        stopper.join(5)
        passive.stopped.set()
        server.shutdown()
        listener.join(5)
        assert not stopper.is_alive()
        assert not listener.is_alive()
