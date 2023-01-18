# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
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
import time

import grpc
import logging
import threading
from typing import Union, List

from nvflare.fuel.f3.comm_config import CommConfigurator
from nvflare.fuel.f3.comm_error import CommError
from nvflare.fuel.f3.drivers.connection import Connection
from nvflare.fuel.f3.drivers.driver import Connector, DriverParams
from nvflare.fuel.f3.drivers.socket_driver import SocketDriver
from nvflare.fuel.f3.drivers import net_utils

from nvflare.fuel.f3.drivers.grpc.streamer_pb2_grpc import (
    StreamerServicer, add_StreamerServicer_to_server, StreamerStub
)
from .grpc.streamer_pb2 import Frame


MAX_MSG_SIZE = 1024 * 1024 * 1024    # 1G

GRPC_DEFAULT_OPTIONS = [
    ("grpc.max_send_message_length", MAX_MSG_SIZE),
    ("grpc.max_receive_message_length", MAX_MSG_SIZE),
]


class AioContext:

    def __init__(self, name):
        self.closed = False
        self.name = name
        self.thread = threading.Thread(target=self._run)
        self.loop = None
        self.ready = False
        self.logger = logging.getLogger(self.__class__.__name__)
        self.thread.start()

    def _run(self):
        self.logger.debug(f"{self.name}: started AioContext in thread {threading.current_thread().name}")
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.logger.debug(f"{self.name}: got loop: {id(self.loop)}")
        self.ready = True
        self.loop.run_until_complete(self._forever())

    async def _forever(self):
        self.logger.debug(f"{self.name}: run _forever in loop ")
        while not self.closed:
            await asyncio.sleep(0.5)

    def run_coro(self, coro):
        while not self.ready:
            self.logger.debug(f"{self.name}: waiting for loop to be ready")
            time.sleep(0.5)
        self.logger.debug(f"{self.name}: coro loop: {id(self.loop)}")
        asyncio.run_coroutine_threadsafe(coro, self.loop)

    def close(self):
        self.closed = True
        self.thread.join()


class AioStreamSession(Connection):

    seq_num = 0

    def __init__(
            self,
            side: str,
            aio_ctx: AioContext,
            connector: Connector,
            peer_address,
            context=None,
            channel=None):
        super().__init__(connector)
        self.side = side
        self.aio_ctx = aio_ctx
        self.logger = logging.getLogger(self.__class__.__name__)

        self.logger.debug(f"{side}: creating asyncio.Queue")
        self.oq = asyncio.Queue(16)
        self.logger.debug(f"{side}: got queue {id(self.oq)}")
        self.closing = False
        self.peer_address = peer_address
        self.context = context    # for server side
        self.channel = channel    # for client side
        self.lock = threading.Lock()
        # self.q_task = asyncio.create_task(self.oq.get())
        self.q_task = None

    def get_conn_properties(self) -> dict:
        addr = self.peer_address
        if isinstance(addr, tuple):
            return {"peer_host": addr[0], "peer_port": addr[1]}
        elif addr:
            return {"peer_addr": addr}
        else:
            return {}

    def close(self):
        self.closing = True
        with self.lock:
            if self.q_task:
                self.q_task.cancel()
                self.q_task = None
            if self.context:
                self.aio_ctx.run_coro(self.context.abort(grpc.StatusCode.CANCELLED, "service closed"))
                self.context = None
            if self.channel:
                self.aio_ctx.run_coro(self.channel.close())
                self.channel = None

    async def _send_frame(self, frame: Frame):
        self.logger.debug(f"{self.side}: sending frame {frame.seq}...")
        await self.oq.put(frame)
        self.logger.debug(f"{self.side}: sent frame {frame.seq}")

    def send_frame(self, frame: Union[bytes, bytearray, memoryview]):
        try:
            AioStreamSession.seq_num += 1
            seq = AioStreamSession.seq_num
            self.logger.debug(f"{self.side}: trying to queue frame #{seq}")
            self.aio_ctx.run_coro(self._send_frame(Frame(seq=seq, data=bytes(frame))))
            self.logger.debug(f"{self.side}: queued item to queue {id(self.oq)}")
        except BaseException as ex:
            raise CommError(CommError.ERROR, f"Error sending frame: {ex}")

    async def read_loop(self, msg_iter):
        ct = threading.current_thread()
        self.logger.debug(f"{self.side}: started read_loop in thread {ct.name}")
        try:
            async for f in msg_iter:
                if self.closing:
                    return

                assert isinstance(f, Frame)
                self.logger.debug(f"{self.side} in {ct.name}: incoming frame #{f.seq}")
                if self.frame_receiver:
                    self.frame_receiver.process_frame(f.data)
                else:
                    self.logger.error(f"{self.side}: Frame receiver not registered for connection: {self.name}")
        except BaseException as ex:
            self.logger.error(f"{self.side}: exception {type(ex)} in read_loop")
        self.logger.debug(f"{self.side} in {ct.name}: done read_loop")

    async def get_output_frame(self):
        return await self.q_task

    async def generate_output(self):
        ct = threading.current_thread()
        self.logger.debug(f"{self.side}: generate_output in thread {ct.name}")
        try:
            while True:
                item = await self.oq.get()
                assert isinstance(item, Frame)
                self.logger.debug(f"{self.side}: outgoing frame #{item.seq}")
                yield item
        except BaseException as ex:
            self.logger.debug(f"{self.side}: generate_output exception {type(ex)}")
        self.logger.debug(f"{self.side}: done generate_output")


class Servicer(StreamerServicer):

    def __init__(self, server, aio_ctx: AioContext):
        self.server = server
        self.aio_ctx = aio_ctx
        self.logger = logging.getLogger(self.__class__.__name__)

    async def _write_loop(self, connection, grpc_context):
        self.logger.debug("started _write_loop")
        try:
            while True:
                f = await connection.oq.get()
                await grpc_context.write(f)
        except BaseException as ex:
            self.logger.debug(f"_write_loop except: {type(ex)}")
        self.logger.debug("finished _write_loop")

    async def Stream(self, request_iterator, context):
        connection = None
        ct = threading.current_thread()
        try:
            self.logger.debug(f"SERVER started Stream CB in thread {ct.name}")
            connection = AioStreamSession(
                side="SERVER",
                aio_ctx=self.aio_ctx,
                connector=self.server.connector,
                peer_address=context.peer(),
                context=context)
            self.logger.debug(f"SERVER created connection in thread {ct.name}")
            self.server.driver.add_connection(connection)
            try:
                await asyncio.gather(
                    self._write_loop(connection, context),
                    connection.read_loop(request_iterator)
                )
            except BaseException as ex:
                self.logger.debug(f"await gather except: {type(ex)}")
            self.logger.debug(f"SERVER: done await gather in thread {ct.name}")

        except BaseException as ex:
            self.logger.error(f"Connection closed due to error: {ex}")
        finally:
            if connection:
                with connection.lock:
                    connection.context = None
                self.logger.debug(f"SERVER: closing connection {connection.name}")
                self.server.driver.close_connection(connection)
            self.logger.debug(f"SERVER: cleanly finished Stream CB in thread {ct.name}")


class Server:

    def __init__(
            self,
            driver,
            connector,
            aio_ctx: AioContext,
            options,
    ):
        self.driver = driver
        self.connector = connector
        self.grpc_server = grpc.aio.server(options=options)
        servicer = Servicer(self, aio_ctx)
        add_StreamerServicer_to_server(servicer, self.grpc_server)
        params = connector.params
        host = params.get(DriverParams.HOST.value)
        port = int(params.get(DriverParams.PORT.value))
        addr = f'{host}:{port}'
        self.grpc_server.add_insecure_port(addr)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.debug(f"added insecure address {addr}")

    async def start(self):
        self.logger.debug("starting grpc server")
        await self.grpc_server.start()
        await self.grpc_server.wait_for_termination()

    async def shutdown(self):
        await self.grpc_server.stop(grace=0.5)


class _ConnCtx:

    def __init__(self, conn):
        self.conn = conn


class AioGrpcDriver(SocketDriver):

    aio_ctx = None
    lock = threading.Lock()

    def __init__(self):
        SocketDriver.__init__(self)
        self.server = None
        self.options = GRPC_DEFAULT_OPTIONS
        self.logger = logging.getLogger(self.__class__.__name__)
        configurator = CommConfigurator()
        config = configurator.get_config()
        if config:
            my_params = config.get("grpc")
            if my_params:
                self.options = my_params.get("options")
        self.logger.debug(f"GRPC Config: options={self.options}")

    @staticmethod
    def _initialize_aio(name: str):
        with AioGrpcDriver.lock:
            if not AioGrpcDriver.aio_ctx:
                AioGrpcDriver.aio_ctx = AioContext(name)
            return AioGrpcDriver.aio_ctx

    @staticmethod
    def supported_transports() -> List[str]:
        return ["aio"]

    async def _start_server(self, connector: Connector, aio_ctx: AioContext):
        self.connector = connector
        self.server = Server(self, connector, aio_ctx, options=self.options)
        await self.server.start()

    def listen(self, connector: Connector):
        self.logger.debug(f"listen called from thread {threading.current_thread().name}")
        aio_ctx = self._initialize_aio("SERVER")
        aio_ctx.run_coro(self._start_server(connector, aio_ctx))

    async def _start_connect(self, connector: Connector, aio_ctx: AioContext, conn_ctx: _ConnCtx):
        self.logger.debug("Started _start_connect coro")
        params = connector.params
        host = params.get(DriverParams.HOST.value)
        port = params.get(DriverParams.PORT.value)
        address = f"{host}:{port}"

        self.logger.debug(f"CLIENT: trying to connect {address}")
        async with grpc.aio.insecure_channel(address, options=self.options) as channel:
            self.logger.debug(f"CLIENT: connected to {address}")
            stub = StreamerStub(channel)
            self.logger.debug("CLIENT: got stub!")
            connection = AioStreamSession(
                side="CLIENT",
                aio_ctx=aio_ctx,
                connector=connector,
                peer_address=address,
                channel=channel)
            self.logger.debug(f"CLIENT: created AioStreamSession {id(connection)}")
            try:
                self.logger.debug(f"CLIENT: start streaming on connection {id(connection)}")
                received = stub.Stream(connection.generate_output())
                conn_ctx.conn = connection
                await connection.read_loop(received)
                # try:
                #     async for f in received:
                #         assert isinstance(f, Frame)
                #         self.logger.debug(f"CLIENT: incoming frame #{f.seq}")
                #         if connection.frame_receiver:
                #             connection.frame_receiver.process_frame(f.data)
                #         else:
                #             self.logger.error(
                #                 f"CLIENT: Frame receiver not registered for connection: {connection.name}")
                # except BaseException as ex:
                #     self.logger.error(f"{connection.side}: exception {type(ex)} in read_loop")
                # self.logger.debug(f"{connection.side}: done read_loop")
            except BaseException as ex:
                self.logger.info(f"CLIENT: connection done: {type(ex)}")
        with connection.lock:
            connection.channel = None
        connection.close()
        self.logger.debug("CLIENT: finished connection")

    def connect(self, connector: Connector):
        self.logger.debug("CLIENT: connect called")
        aio_ctx = self._initialize_aio("CLIENT")
        conn_ctx = _ConnCtx(None)
        aio_ctx.run_coro(self._start_connect(connector, aio_ctx, conn_ctx))
        while not conn_ctx.conn:
            self.logger.debug("CLIENT: waiting for connection")
            time.sleep(0.5)

        self.logger.debug(f"CLIENT: got connection {id(conn_ctx.conn)}")
        self.add_connection(conn_ctx.conn)
        self.logger.debug(f"CLIENT: created new connection {id(conn_ctx.conn)}")

    def shutdown(self):
        for _, conn in self.connections.items():
            conn.close()

        if self.server:
            aio_ctx = self._initialize_aio("SERVER")
            aio_ctx.run_coro(self.server.shutdown())

    @staticmethod
    def get_urls(scheme: str, resources: dict) -> (str, str):
        secure = resources.get(DriverParams.SECURE)
        if secure:
            scheme = "aios"

        host = resources.get("host") if resources else None
        if not host:
            host = "localhost"

        port = net_utils.get_open_tcp_port(resources)
        if not port:
            raise CommError(CommError.BAD_CONFIG, "Can't find an open port in the specified range")

        # Always listen on all interfaces
        listening_url = f"{scheme}://0:{port}"
        connect_url = f"{scheme}://{host}:{port}"

        return connect_url, listening_url
