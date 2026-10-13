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
import logging
import select
import socket
import threading
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


class _TlsListener:
    """Accept raw sockets before handing them to asyncio's TLS transport."""

    def __init__(self, sockets, on_accept):
        self.sockets = sockets
        self.on_accept = on_accept
        self._stopped = threading.Event()
        self._closed = asyncio.Event()
        self._wake_reader, self._wake_writer = socket.socketpair()
        self._thread = threading.Thread(target=self._accept_loop, name="aio_tcp_tls_accept", daemon=True)
        self._thread.start()

    def _accept_loop(self):
        try:
            while not self._stopped.is_set():
                try:
                    readable, _, _ = select.select([self._wake_reader, *self.sockets], [], [])
                except (OSError, ValueError):
                    break
                if self._wake_reader in readable:
                    break
                for listener in readable:
                    try:
                        conn, _ = listener.accept()
                    except (BlockingIOError, OSError):
                        continue
                    try:
                        conn.setblocking(False)
                        self.on_accept(conn)
                    except Exception:
                        conn.close()
                        log.exception("Failed to register accepted TLS socket")
        finally:
            self._wake_reader.close()
            self._wake_writer.close()

    def close(self):
        if self._stopped.is_set():
            return
        self._stopped.set()
        for listener in self.sockets:
            listener.close()
        try:
            self._wake_writer.send(b"x")
        except OSError:
            pass
        self._closed.set()

    async def wait_closed(self):
        await asyncio.to_thread(self._thread.join)

    async def serve_forever(self):
        await self._closed.wait()


class AioTcpDriver(AioBaseDriver):
    def __init__(self):
        super().__init__()
        self.aio_ctx = AioContext.get_global_context()
        self.server = None
        self.ssl_context = None
        self._accept_tasks = set()
        self._pending_sockets = set()
        self._pending_lock = threading.Lock()

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
                # thread has joined. Close raw sockets before task cancellation.
                await server.wait_closed()
                with self._pending_lock:
                    pending = list(self._pending_sockets)
                for conn in pending:
                    conn.close()
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

    def _accept_tls_socket(self, conn):
        with self._pending_lock:
            if self.is_stopping() or self.connector.stopped.is_set():
                conn.close()
                return
            self._pending_sockets.add(conn)
        try:
            self.aio_ctx.get_event_loop().call_soon_threadsafe(self._start_tls_socket, conn)
        except Exception:
            with self._pending_lock:
                self._pending_sockets.discard(conn)
            conn.close()
            raise

    def _start_tls_socket(self, conn):
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
                with self._pending_lock:
                    self._pending_sockets.discard(conn)
                if self.is_stopping() or self.connector.stopped.is_set():
                    conn.close()

        task = asyncio.create_task(accept())
        self._accept_tasks.add(task)
        task.add_done_callback(self._accept_tasks.discard)

    async def _tcp_connect(self, host, port):
        self.ssl_context = get_ssl_context(self.connector.params, ssl_server=False)
        reader, writer = await asyncio.open_connection(host, port, ssl=self.ssl_context)
        await self._create_connection(reader, writer)

    async def _tcp_listen(self, host, port):
        self.ssl_context = get_ssl_context(self.connector.params, ssl_server=True)
        if self.ssl_context:
            loop = asyncio.get_running_loop()
            addresses = await loop.getaddrinfo(host, port, type=socket.SOCK_STREAM, flags=socket.AI_PASSIVE)
            sockets = []
            try:
                for family, sock_type, proto, _, address in addresses:
                    listener = socket.socket(family, sock_type, proto)
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
            except BaseException:
                for listener in sockets:
                    listener.close()
                raise
            self.server = _TlsListener(sockets, self._accept_tls_socket)
        else:
            self.server = await asyncio.start_server(self._create_connection, host, port)
        if self.is_stopping() or self.connector.stopped.is_set():
            self.server.close()
            await self.server.wait_closed()
            return
        if isinstance(self.server, _TlsListener):
            await self.server.serve_forever()
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
