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
import traceback

import grpc
import logging
import threading
from typing import Union, List, Dict, Any

from nvflare.fuel.f3.comm_config import CommConfigurator
from nvflare.fuel.f3.comm_error import CommError
from nvflare.fuel.f3.connection import Connection
from nvflare.fuel.f3.drivers.driver import Connector, DriverParams, DriverCap
from nvflare.fuel.f3.drivers import net_utils

from nvflare.fuel.f3.drivers.grpc.streamer_pb2_grpc import (
    StreamerServicer, add_StreamerServicer_to_server, StreamerStub
)
from .base_driver import BaseDriver
from .grpc.streamer_pb2 import Frame
from .net_utils import get_address, ssl_required

MAX_MSG_SIZE = 1024 * 1024 * 1024    # 1G

GRPC_DEFAULT_OPTIONS = [
    ("grpc.max_send_message_length", MAX_MSG_SIZE),
    ("grpc.max_receive_message_length", MAX_MSG_SIZE),
]


class AioContext:

    counter_lock = threading.Lock()
    thread_count = 0

    def __init__(self, name):
        self.closed = False
        self.name = name
        with AioContext.counter_lock:
            AioContext.thread_count += 1
        thread_name = f"aio_ctx_{AioContext.thread_count}"
        self.thread = threading.Thread(target=self._run_aio_loop, name=thread_name)
        self.loop = None
        self.ready = False
        self.logger = logging.getLogger(self.__class__.__name__)
        self.thread.start()

    def _run_aio_loop(self):
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


class _ConnCtx:

    def __init__(self):
        self.conn = None
        self.error = None
        self.waiter = threading.Event()


class AioStreamSession(Connection):

    seq_num = 0

    def __init__(
            self,
            side: str,
            aio_ctx: AioContext,
            connector: Connector,
            conn_props: dict,
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
        self.conn_props = conn_props
        self.context = context    # for server side
        self.channel = channel    # for client side
        self.lock = threading.Lock()

    def get_conn_properties(self) -> dict:
        return self.conn_props

    def close(self):
        self.closing = True
        with self.lock:
            if self.context:
                self.aio_ctx.run_coro(self.context.abort(grpc.StatusCode.CANCELLED, "service closed"))
                self.context = None
            if self.channel:
                self.aio_ctx.run_coro(self.channel.close())
                self.channel = None

    # async def _send_frame(self, frame: Frame):
    #     self.logger.debug(f"{self.side}: sending frame {frame.seq}...")
    #     await self.oq.put(frame)
    #     self.logger.debug(f"{self.side}: sent frame {frame.seq}")

    def send_frame(self, frame: Union[bytes, bytearray, memoryview]):
        try:
            AioStreamSession.seq_num += 1
            seq = AioStreamSession.seq_num
            self.logger.debug(f"{self.side}: trying to queue frame #{seq}")
            f = Frame(seq=seq, data=bytes(frame))
            # self.aio_ctx.run_coro(self._send_frame(f))
            # asyncio.run_coroutine_threadsafe(self.oq.put(f), self.aio_ctx.loop)
            self.aio_ctx.run_coro(self.oq.put(f))
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
        except asyncio.CancelledError:
            self.logger.debug(f"{self.side}: RPC cancelled")
        except BaseException as ex:
            traceback.print_exc()
            self.logger.error(f"{self.side}: exception {type(ex)} in read_loop")
        self.logger.debug(f"{self.side}: in {ct.name}: done read_loop")

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
            conn_props = {DriverParams.PEER_ADDR.value: context.peer(),
                          DriverParams.LOCAL_ADDR.value: get_address(self.server.connector.params)}

            cn_names = context.auth_context().get("x509_common_name")
            if cn_names:
                conn_props[DriverParams.PEER_CN.value] = cn_names[0].decode("utf-8")

            connection = AioStreamSession(
                side="SERVER",
                aio_ctx=self.aio_ctx,
                connector=self.server.connector,
                conn_props=conn_props,
                context=context)
            self.logger.debug(f"SERVER created connection in thread {ct.name}")
            self.server.driver.add_connection(connection)
            try:
                await asyncio.gather(
                    self._write_loop(connection, context),
                    connection.read_loop(request_iterator)
                )
            except asyncio.CancelledError:
                self.logger.debug("SERVER: RPC cancelled")
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
            conn_ctx: _ConnCtx
    ):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.driver = driver
        self.connector = connector
        self.grpc_server = grpc.aio.server(options=options)
        servicer = Servicer(self, aio_ctx)
        add_StreamerServicer_to_server(servicer, self.grpc_server)
        params = connector.params
        host = params.get(DriverParams.HOST.value)
        if not host:
            host = 0
        port = int(params.get(DriverParams.PORT.value))
        addr = f'{host}:{port}'
        try:
            self.logger.debug(f"SERVER: connector params: {params}")
            self.logger.debug(f"SERVER: adding insecure port {addr}")

            secure = ssl_required(params)

            if secure:
                credentials = AioGrpcDriver.get_grpc_server_credentials(params)
                self.grpc_server.add_secure_port(addr, server_credentials=credentials)
            else:
                self.grpc_server.add_insecure_port(addr)
        except BaseException as ex:
            self.logger.error(f"cannot create SERVER to listen on {addr}: {type(ex)}")
            conn_ctx.error = f"cannot listen on {addr}: {type(ex)}: {ex}"
        self.logger.debug(f"added insecure address {addr}")

    async def start(self, conn_ctx: _ConnCtx):
        self.logger.debug("starting grpc server")
        try:
            await self.grpc_server.start()
            await self.grpc_server.wait_for_termination()
        except BaseException as ex:
            conn_ctx.error = f"cannot start server: {type(ex)}: {ex}"

    async def shutdown(self):
        await self.grpc_server.stop(grace=0.5)


class AioGrpcDriver(BaseDriver):

    aio_ctx = None
    lock = threading.Lock()

    def __init__(self):
        super().__init__()
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
        return ["grpc", "grpcs"]

    @staticmethod
    def capabilities() -> Dict[str, Any]:
        return {
            DriverCap.HEARTBEAT.value: True,
            DriverCap.SUPPORT_SSL.value: True
        }

    async def _start_server(self, connector: Connector, aio_ctx: AioContext, conn_ctx: _ConnCtx):
        self.connector = connector
        self.server = Server(self, connector, aio_ctx, options=self.options, conn_ctx=conn_ctx)
        if not conn_ctx.error:
            try:
                conn_ctx.conn = True
                await self.server.start(conn_ctx)
            except BaseException as ex:
                traceback.print_exc()
                conn_ctx.error = f"failed to start server: {type(ex)}: {ex}"
        conn_ctx.waiter.set()

    def listen(self, connector: Connector):
        self.logger.debug(f"listen called from thread {threading.current_thread().name}")
        aio_ctx = self._initialize_aio("SERVER")
        conn_ctx = _ConnCtx()
        aio_ctx.run_coro(self._start_server(connector, aio_ctx, conn_ctx))
        while not conn_ctx.conn and not conn_ctx.error:
            self.logger.debug("SERVER: waiting for server to be started")
            time.sleep(0.1)
        if conn_ctx.error:
            raise CommError(code=CommError.ERROR, message=conn_ctx.error)
        self.logger.debug("SERVER: waiting for server to finish")
        conn_ctx.waiter.wait()
        self.logger.debug("SERVER: server is done")

    async def _start_connect(self, connector: Connector, aio_ctx: AioContext, conn_ctx: _ConnCtx):
        self.logger.debug("Started _start_connect coro")
        params = connector.params
        address = get_address(params)

        self.logger.debug(f"CLIENT: trying to connect {address}")
        try:
            secure = ssl_required(params)
            if secure:
                grpc_channel = grpc.aio.secure_channel(address, options=self.options,
                                                       credentials=self.get_grpc_client_credentials(params))
            else:
                grpc_channel = grpc.aio.insecure_channel(address, options=self.options)

            async with grpc_channel as channel:
                self.logger.debug(f"CLIENT: connected to {address}")
                stub = StreamerStub(channel)
                self.logger.debug("CLIENT: got stub!")
                conn_props = {DriverParams.PEER_ADDR.value: address}
                connection = AioStreamSession(
                    side="CLIENT",
                    aio_ctx=aio_ctx,
                    connector=connector,
                    conn_props=conn_props,
                    channel=channel)
                self.logger.debug(f"CLIENT: created AioStreamSession {id(connection)}")
                try:
                    self.logger.debug(f"CLIENT: start streaming on connection {id(connection)}")
                    msg_iter = stub.Stream(connection.generate_output())
                    conn_ctx.conn = connection
                    await connection.read_loop(msg_iter)
                except asyncio.CancelledError:
                    self.logger.debug(f"CLIENT: RPC cancelled")
                except BaseException as ex:
                    traceback.print_exc()
                    self.logger.info(f"CLIENT: connection done: {type(ex)}")

            with connection.lock:
                connection.channel = None
            connection.close()
            self.logger.debug("CLIENT: finished connection")
        except asyncio.CancelledError:
            self.logger.debug("CLIENT: RPC cancelled")
        except BaseException as ex:
            traceback.print_exc()
            conn_ctx.error = f"connection error: {type(ex)}: {ex}"
        conn_ctx.waiter.set()

    def connect(self, connector: Connector):
        self.logger.debug("CLIENT: connect called")
        aio_ctx = self._initialize_aio("CLIENT")
        conn_ctx = _ConnCtx()
        aio_ctx.run_coro(self._start_connect(connector, aio_ctx, conn_ctx))
        time.sleep(0.2)
        while not conn_ctx.conn and not conn_ctx.error:
            self.logger.debug("CLIENT: waiting for connection")
            time.sleep(0.1)

        if conn_ctx.error:
            raise CommError(CommError.ERROR, conn_ctx.error)

        self.logger.debug(f"CLIENT: got connection {id(conn_ctx.conn)}")
        self.add_connection(conn_ctx.conn)
        self.logger.debug(f"CLIENT: created new connection {id(conn_ctx.conn)}")

        # wait for connection to be finished
        self.logger.debug(f"CLIENT: waiting for connection {id(conn_ctx.conn)} to finish")
        conn_ctx.waiter.wait()
        self.close_connection(conn_ctx.conn)
        self.logger.debug(f"CLIENT: connection {id(conn_ctx.conn)} is done")

    def shutdown(self):
        self.close_all()

        if self.server:
            aio_ctx = self._initialize_aio("SERVER")
            aio_ctx.run_coro(self.server.shutdown())

    @staticmethod
    def get_urls(scheme: str, resources: dict) -> (str, str):
        secure = resources.get(DriverParams.SECURE)
        if secure:
            scheme = "grpcs"

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

    @staticmethod
    def get_grpc_client_credentials(params: dict):

        root_cert = AioGrpcDriver.read_file(params.get(DriverParams.CA_CERT.value))
        cert_chain = AioGrpcDriver.read_file(params.get(DriverParams.CLIENT_CERT))
        private_key = AioGrpcDriver.read_file(params.get(DriverParams.CLIENT_KEY))

        return grpc.ssl_channel_credentials(
            certificate_chain=cert_chain, private_key=private_key, root_certificates=root_cert
        )

    @staticmethod
    def get_grpc_server_credentials(params: dict):

        root_cert = AioGrpcDriver.read_file(params.get(DriverParams.CA_CERT.value))
        cert_chain = AioGrpcDriver.read_file(params.get(DriverParams.SERVER_CERT))
        private_key = AioGrpcDriver.read_file(params.get(DriverParams.SERVER_KEY))

        return grpc.ssl_server_credentials(
            [(private_key, cert_chain)],
            root_certificates=root_cert,
            require_client_auth=True,
        )

    @staticmethod
    def read_file(file_name: str):
        if not file_name:
            return None

        with open(file_name, "rb") as f:
            return f.read()
