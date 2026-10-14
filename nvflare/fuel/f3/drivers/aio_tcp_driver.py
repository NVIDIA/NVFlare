# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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
import errno
import logging
import socket
from concurrent.futures import CancelledError
from typing import Any, Dict, List

from nvflare.fuel.f3.comm_error import CommError
from nvflare.fuel.f3.drivers.aio_base_driver import AioBaseDriver
from nvflare.fuel.f3.drivers.aio_conn import AioConnection
from nvflare.fuel.f3.drivers.aio_context import AioContext
from nvflare.fuel.f3.drivers.connector_info import ConnectorInfo, Mode
from nvflare.fuel.f3.drivers.driver_params import DriverCap, DriverParams
from nvflare.fuel.f3.drivers.net_utils import get_ssl_context
from nvflare.fuel.f3.drivers.tcp_driver import TcpDriver

log = logging.getLogger(__name__)

_RETRY_ACCEPT_ERRORS = frozenset(
    getattr(errno, name)
    for name in (
        "EAGAIN",
        "EWOULDBLOCK",
        "EINTR",
        "ECONNABORTED",
        # Linux can report an already-failed connection's network error from accept().
        "ENETDOWN",
        "EPROTO",
        "ENOPROTOOPT",
        "EHOSTDOWN",
        "ENONET",
        "EHOSTUNREACH",
        "EOPNOTSUPP",
        "ENETUNREACH",
    )
    if hasattr(errno, name)
)
_PAUSE_ACCEPT_ERRORS = frozenset(
    getattr(errno, name) for name in ("EMFILE", "ENFILE", "ENOBUFS", "ENOMEM") if hasattr(errno, name)
)
_ACCEPT_RETRY_DELAY = 1


class _TlsListener:
    """Accept raw sockets before handing them to asyncio's TLS transport."""

    def __init__(self, loop, sockets, on_accept):
        self.sockets = sockets
        self.on_accept = on_accept
        self._stopped = False
        self._failure = None
        self._tasks = [loop.create_task(self._accept_loop(loop, listener)) for listener in sockets]

    async def _accept_loop(self, loop, listener):
        try:
            while not self._stopped:
                try:
                    conn, _ = await loop.sock_accept(listener)
                except OSError as ex:
                    if self._stopped:
                        return
                    if isinstance(ex, ConnectionAbortedError) or ex.errno in _RETRY_ACCEPT_ERRORS:
                        log.debug("TLS listener retrying after accept error: %s", ex)
                        await asyncio.sleep(0)
                        continue
                    if ex.errno in _PAUSE_ACCEPT_ERRORS:
                        log.warning("TLS listener accept unavailable; retrying in one second: %s", ex)
                        await asyncio.sleep(_ACCEPT_RETRY_DELAY)
                        continue
                    self._failure = ex
                    log.exception("TLS listener stopped accepting connections")
                    self.close()
                    return
                except ValueError as ex:
                    if not self._stopped:
                        self._failure = ex
                        log.exception("TLS listener stopped accepting connections")
                        self.close()
                    return
                try:
                    self.on_accept(conn)
                except Exception:
                    conn.close()
                    log.exception("Failed to register accepted TLS socket")
        except asyncio.CancelledError:
            return

    def close(self):
        if self._stopped:
            return
        self._stopped = True
        for listener in self.sockets:
            listener.close()
        for task in self._tasks:
            task.cancel()

    async def wait_closed(self):
        await asyncio.gather(*self._tasks, return_exceptions=True)

    async def serve_forever(self):
        await self.wait_closed()
        if self._failure:
            raise self._failure


class AioTcpDriver(AioBaseDriver):
    def __init__(self):
        super().__init__()
        self.aio_ctx = AioContext.get_global_context()
        self.server = None
        self.ssl_context = None
        self._accept_tasks = set()
        self._pending_sockets = set()

    @staticmethod
    def supported_transports() -> List[str]:
        return ["atcp", "satcp"]

    @staticmethod
    def capabilities() -> Dict[str, Any]:
        return {DriverCap.SEND_HEARTBEAT.value: True, DriverCap.SUPPORT_SSL.value: True}

    def listen(self, connector: ConnectorInfo):
        self._run(connector, Mode.PASSIVE)

    def connect(self, connector: ConnectorInfo):
        self._run(connector, Mode.ACTIVE)

    def shutdown(self):
        self.stop_connection_admission()
        return self.aio_ctx.run_coro(self._async_shutdown())

    @staticmethod
    def get_urls(scheme: str, resources: dict) -> (str, str):
        return TcpDriver.get_urls(scheme, resources)

    # Internal methods

    def _run(self, connector: ConnectorInfo, mode: Mode):
        self.connector = connector
        if mode != self.connector.mode:
            raise CommError(CommError.ERROR, f"Connector mode doesn't match driver mode for {self.connector}")

        try:
            operation = (
                self._run_connect(connector, lambda: self._async_run(mode))
                if mode == Mode.ACTIVE
                else self._async_run(mode)
            )
            if self.is_stopping() or connector.stopped.is_set():
                operation.close()
                return
            self.aio_ctx.run_coro(operation).result()
        except CancelledError:
            log.debug(f"Connector {self.connector} is cancelled")

    async def _async_run(self, mode: Mode):

        params = self.connector.params
        host = params.get(DriverParams.HOST.value)
        port = params.get(DriverParams.PORT.value)

        if mode == Mode.ACTIVE:
            coroutine = self._tcp_connect(host, port)
        else:
            coroutine = self._tcp_listen(host, port)

        await coroutine

    async def _async_shutdown(self):
        async with self._shutdown_lock:
            server = self.server
            self.server = None
            if server:
                server.close()
            self.close_all()
            await self._cancel_connect_tasks()
            if isinstance(server, _TlsListener):
                # No accept callback can publish a socket after the accept
                # loops have finished. Their tasks own any pending raw sockets.
                await server.wait_closed()
            await self._cancel_accept_tasks()
            if server and not isinstance(server, _TlsListener):
                await server.wait_closed()

    async def _cancel_accept_tasks(self):
        tasks = list(self._accept_tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def _tls_protocol_factory(self):
        return asyncio.StreamReaderProtocol(asyncio.StreamReader(), self._create_connection)

    def _start_tls_socket(self, conn):
        if self.is_stopping() or self.connector.stopped.is_set():
            conn.close()
            return
        self._pending_sockets.add(conn)

        async def accept():
            try:
                if self.is_stopping() or self.connector.stopped.is_set():
                    return
                await asyncio.get_running_loop().connect_accepted_socket(
                    self._tls_protocol_factory, conn, ssl=self.ssl_context
                )
            except asyncio.CancelledError:
                conn.close()
                raise
            except Exception as ex:
                conn.close()
                log.debug(f"TLS accept failed: {ex}")
            finally:
                self._pending_sockets.discard(conn)
                if self.is_stopping() or self.connector.stopped.is_set():
                    conn.close()

        task = asyncio.create_task(accept())
        self._accept_tasks.add(task)

        def done(completed):
            self._accept_tasks.discard(completed)
            # Cancellation before the coroutine starts never runs its finally.
            if completed.cancelled():
                self._pending_sockets.discard(conn)
                conn.close()

        task.add_done_callback(done)

    async def _tcp_connect(self, host, port):
        self.ssl_context = get_ssl_context(self.connector.params, ssl_server=False)
        reader, writer = await asyncio.open_connection(host, port, ssl=self.ssl_context)
        await self._create_connection(reader, writer)

    async def _tcp_listen(self, host, port):
        self.ssl_context = get_ssl_context(self.connector.params, ssl_server=True)
        if self.ssl_context:
            loop = asyncio.get_running_loop()
            addresses = await loop.getaddrinfo(host or None, port, type=socket.SOCK_STREAM, flags=socket.AI_PASSIVE)
            sockets = []
            socket_error = None
            try:
                for family, sock_type, proto, _, address in addresses:
                    try:
                        listener = socket.socket(family, sock_type, proto)
                    except OSError as ex:
                        # A resolved address family may not be supported by this host.
                        socket_error = ex
                        continue
                    try:
                        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                        if family == socket.AF_INET6:
                            listener.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
                        listener.bind(address)
                        listener.listen(socket.SOMAXCONN)
                        listener.setblocking(False)
                    except BaseException:
                        listener.close()
                        raise
                    sockets.append(listener)
                if not sockets:
                    raise OSError(f"Could not create TLS listener on any address for {host}:{port}") from socket_error
            except BaseException:
                for listener in sockets:
                    listener.close()
                raise
            self.server = _TlsListener(loop, sockets, self._start_tls_socket)
        else:
            self.server = await asyncio.start_server(self._create_connection, host, port)
        if self.is_stopping() or self.connector.stopped.is_set():
            server = self.server
            self.server = None
            server.close()
            await server.wait_closed()
            return
        if isinstance(self.server, _TlsListener):
            server = self.server
            try:
                await server.serve_forever()
            finally:
                if self.server is server:
                    await self._async_shutdown()
        else:
            async with self.server:
                await self.server.serve_forever()

    async def _create_connection(self, reader, writer):
        conn = AioConnection(self.connector, self.aio_ctx, reader, writer, self.ssl_context is not None)
        try:
            if not self.add_connection(conn):
                return
            await conn.read_loop()
        finally:
            try:
                conn.close()
            finally:
                try:
                    self.close_connection(conn)
                finally:
                    # The read side has ended. A peer that stops reading can
                    # otherwise keep queued writes and wait_closed() pending.
                    writer.transport.abort()
            await writer.wait_closed()
