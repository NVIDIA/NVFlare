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
import threading
import time
from typing import Any, Dict, List

import grpc

from nvflare.fuel.f3.comm_config import CommConfigurator
from nvflare.fuel.f3.comm_error import CommError
from nvflare.fuel.f3.connection import BytesAlike, Connection
from nvflare.fuel.f3.drivers.aio_context import AioContext
from nvflare.fuel.f3.drivers.driver import ConnectorInfo
from nvflare.fuel.f3.drivers.grpc.streamer_pb2_grpc import (
    StreamerServicer,
    StreamerStub,
    add_StreamerServicer_to_server,
)
from nvflare.fuel.utils.obj_utils import get_logger
from nvflare.security.logging import secure_format_exception, secure_format_traceback

from .base_driver import BaseDriver
from .driver_params import DriverCap, DriverParams
from .grpc.streamer_pb2 import Frame
from .net_utils import MAX_FRAME_SIZE, get_address, get_tcp_urls, ssl_required

GRPC_DEFAULT_OPTIONS = [
    ("grpc.max_send_message_length", MAX_FRAME_SIZE),
    ("grpc.max_receive_message_length", MAX_FRAME_SIZE),
    ("grpc.keepalive_time_ms", 120000),
    ("grpc.http2.max_pings_without_data", 0),
]


class _ConnCtx:
    def __init__(self):
        self.conn = None
        self.error = None
        self.waiter = threading.Event()


class AioStreamSession(Connection):

    seq_num = 0

    def __init__(self, aio_ctx: AioContext, connector: ConnectorInfo, conn_props: dict, context=None, channel=None):
        super().__init__(connector)
        self.aio_ctx = aio_ctx
        self.logger = get_logger(self)

        self.oq = asyncio.Queue(16)
        self.closing = False
        self.conn_props = conn_props
        self.context = context  # for server side
        self.channel = channel  # for client side
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

    def send_frame(self, frame: BytesAlike):
        try:
            AioStreamSession.seq_num += 1
            seq = AioStreamSession.seq_num
            f = Frame(seq=seq, data=bytes(frame))
            self.aio_ctx.run_coro(self.oq.put(f))
        except Exception as ex:
            self.logger.debug(f"exception send_frame: {self}: {secure_format_exception(ex)}")
            if not self.closing:
                raise CommError(CommError.ERROR, f"Error sending frame on conn {self}: {secure_format_exception(ex)}")

    async def read_loop(self, msg_iter):
        ct = threading.current_thread()
        self.logger.debug(f"{self}: started read_loop in thread {ct.name}")
        try:
            async for f in msg_iter:
                if self.closing:
                    return
                self.process_frame(f.data)

        except grpc.aio.AioRpcError as error:
            if not self.closing:
                if error.code() == grpc.StatusCode.CANCELLED:
                    self.logger.debug(f"Connection {self} is closed by peer")
                else:
                    self.logger.debug(f"Connection {self} Error: {error.details()}")
                    self.logger.debug(secure_format_traceback())
            else:
                self.logger.debug(f"Connection {self} is closed locally")
        except Exception as ex:
            if not self.closing:
                self.logger.debug(f"{self}: exception {type(ex)} in read_loop: {secure_format_exception(ex)}")
                self.logger.debug(secure_format_traceback())

        self.logger.debug(f"{self}: in {ct.name}: done read_loop")

    async def generate_output(self):
        ct = threading.current_thread()
        self.logger.debug(f"{self}: generate_output in thread {ct.name}")
        try:
            while True:
                item = await self.oq.get()
                yield item
        except Exception as ex:
            if self.closing:
                self.logger.debug(f"{self}: connection closed by {type(ex)}: {secure_format_exception(ex)}")
            else:
                self.logger.debug(f"{self}: generate_output exception {type(ex)}: {secure_format_exception(ex)}")
            self.logger.debug(secure_format_traceback())

        self.logger.debug(f"{self}: done generate_output")


class Servicer(StreamerServicer):
    def __init__(self, server, aio_ctx: AioContext):
        self.server = server
        self.aio_ctx = aio_ctx
        self.logger = get_logger(self)

    async def _write_loop(self, connection, grpc_context):
        self.logger.debug("started _write_loop")
        try:
            while True:
                f = await connection.oq.get()
                await grpc_context.write(f)
        except Exception as ex:
            self.logger.debug(f"_write_loop except: {type(ex)}: {secure_format_exception(ex)}")
        self.logger.debug("finished _write_loop")

    async def Stream(self, request_iterator, context):
        connection = None
        ct = threading.current_thread()
        try:
            self.logger.debug(f"SERVER started Stream CB in thread {ct.name}")
            conn_props = {
                DriverParams.PEER_ADDR.value: context.peer(),
                DriverParams.LOCAL_ADDR.value: get_address(self.server.connector.params),
            }

            cn_names = context.auth_context().get("x509_common_name")
            if cn_names:
                conn_props[DriverParams.PEER_CN.value] = cn_names[0].decode("utf-8")

            connection = AioStreamSession(
                aio_ctx=self.aio_ctx,
                connector=self.server.connector,
                conn_props=conn_props,
                context=context,
            )
            self.logger.debug(f"SERVER created connection in thread {ct.name}")
            self.server.driver.add_connection(connection)
            try:
                await asyncio.gather(self._write_loop(connection, context), connection.read_loop(request_iterator))
            except asyncio.CancelledError:
                self.logger.debug("SERVER: RPC cancelled")
            except Exception as ex:
                self.logger.debug(f"await gather except: {type(ex)}: {secure_format_exception(ex)}")
            self.logger.debug(f"SERVER: done await gather in thread {ct.name}")

        except Exception as ex:
            self.logger.debug(f"Connection closed due to error: {secure_format_exception(ex)}")
        finally:
            if connection:
                with connection.lock:
                    connection.context = None
                self.logger.debug(f"SERVER: closing connection {connection.name}")
                self.server.driver.close_connection(connection)
            self.logger.debug(f"SERVER: cleanly finished Stream CB in thread {ct.name}")


class Server:
    def __init__(self, driver, connector, aio_ctx: AioContext, options, conn_ctx: _ConnCtx):
        self.logger = get_logger(self)
        self.driver = driver
        self.connector = connector
        self.grpc_server = grpc.aio.server(options=options)
        servicer = Servicer(self, aio_ctx)
        add_StreamerServicer_to_server(servicer, self.grpc_server)
        params = connector.params
        host = params.get(DriverParams.HOST.value)
        if not host:
            host = "0.0.0.0"
        port = int(params.get(DriverParams.PORT.value))
        addr = f"{host}:{port}"
        try:
            self.logger.debug(f"SERVER: connector params: {params}")

            secure = ssl_required(params)
            if secure:
                credentials = AioGrpcDriver.get_grpc_server_credentials(params)
                self.grpc_server.add_secure_port(addr, server_credentials=credentials)
            else:
                self.grpc_server.add_insecure_port(addr)
        except Exception as ex:
            conn_ctx.error = f"cannot listen on {addr}: {type(ex)}: {secure_format_exception(ex)}"
            self.logger.debug(conn_ctx.error)

    async def start(self, conn_ctx: _ConnCtx):
        self.logger.debug("starting grpc server")
        try:
            await self.grpc_server.start()
            await self.grpc_server.wait_for_termination()
        except Exception as ex:
            conn_ctx.error = f"cannot start server: {type(ex)}: {secure_format_exception(ex)}"
            raise ex

    async def shutdown(self):
        try:
            await self.grpc_server.stop(grace=0.5)
        except Exception as ex:
            self.logger.debug(f"exception shutdown server: {secure_format_exception(ex)}")


class AioGrpcDriver(BaseDriver):

    aio_ctx = None

    def __init__(self):
        super().__init__()
        self.server = None
        self.options = GRPC_DEFAULT_OPTIONS
        self.logger = get_logger(self)
        configurator = CommConfigurator()
        config = configurator.get_config()
        if config:
            my_params = config.get("grpc")
            if my_params:
                self.options = my_params.get("options")
        self.logger.debug(f"GRPC Config: options={self.options}")
        self.closing = False

    @staticmethod
    def supported_transports() -> List[str]:
        return ["grpc", "grpcs"]

    @staticmethod
    def capabilities() -> Dict[str, Any]:
        return {DriverCap.HEARTBEAT.value: True, DriverCap.SUPPORT_SSL.value: True}

    async def _start_server(self, connector: ConnectorInfo, aio_ctx: AioContext, conn_ctx: _ConnCtx):
        self.connector = connector
        self.server = Server(self, connector, aio_ctx, options=self.options, conn_ctx=conn_ctx)
        if not conn_ctx.error:
            try:
                conn_ctx.conn = True
                await self.server.start(conn_ctx)
            except Exception as ex:
                if not self.closing:
                    self.logger.debug(secure_format_traceback())
                conn_ctx.error = f"failed to start server: {type(ex)}: {secure_format_exception(ex)}"
        conn_ctx.waiter.set()

    def listen(self, connector: ConnectorInfo):
        self.logger.debug(f"listen called from thread {threading.current_thread().name}")
        self.connector = connector
        aio_ctx = AioContext.get_global_context()
        conn_ctx = _ConnCtx()
        aio_ctx.run_coro(self._start_server(connector, aio_ctx, conn_ctx))
        while not conn_ctx.conn and not conn_ctx.error:
            time.sleep(0.1)
        if conn_ctx.error:
            raise CommError(code=CommError.ERROR, message=conn_ctx.error)
        self.logger.debug("SERVER: waiting for server to finish")
        conn_ctx.waiter.wait()
        self.logger.debug("SERVER: server is done")

    async def _start_connect(self, connector: ConnectorInfo, aio_ctx: AioContext, conn_ctx: _ConnCtx):
        self.logger.debug("Started _start_connect coro")
        self.connector = connector
        params = connector.params
        address = get_address(params)

        self.logger.debug(f"CLIENT: trying to connect {address}")
        try:
            secure = ssl_required(params)
            if secure:
                grpc_channel = grpc.aio.secure_channel(
                    address, options=self.options, credentials=self.get_grpc_client_credentials(params)
                )
            else:
                grpc_channel = grpc.aio.insecure_channel(address, options=self.options)

            async with grpc_channel as channel:
                self.logger.debug(f"CLIENT: connected to {address}")
                stub = StreamerStub(channel)
                conn_props = {DriverParams.PEER_ADDR.value: address}

                if secure:
                    conn_props[DriverParams.PEER_CN.value] = "N/A"

                connection = AioStreamSession(
                    aio_ctx=aio_ctx, connector=connector, conn_props=conn_props, channel=channel
                )

                try:
                    self.logger.debug(f"CLIENT: start streaming on connection {connection}")
                    msg_iter = stub.Stream(connection.generate_output())
                    conn_ctx.conn = connection
                    await connection.read_loop(msg_iter)
                except asyncio.CancelledError as error:
                    self.logger.debug(f"CLIENT: RPC cancelled: {error}")
                except Exception as ex:
                    if self.closing:
                        self.logger.debug(
                            f"Connection {connection} closed by {type(ex)}: {secure_format_exception(ex)}"
                        )
                    else:
                        self.logger.debug(
                            f"Connection {connection} client read exception {type(ex)}: {secure_format_exception(ex)}"
                        )
                    self.logger.debug(secure_format_traceback())

            with connection.lock:
                connection.channel = None
            connection.close()
        except asyncio.CancelledError:
            self.logger.debug("CLIENT: RPC cancelled")
        except Exception as ex:
            conn_ctx.error = f"connection {connection} error: {type(ex)}: {secure_format_exception(ex)}"
            self.logger.debug(conn_ctx.error)
            self.logger.debug(secure_format_traceback())

        conn_ctx.waiter.set()

    def connect(self, connector: ConnectorInfo):
        self.logger.debug("CLIENT: connect called")
        aio_ctx = AioContext.get_global_context()
        conn_ctx = _ConnCtx()
        aio_ctx.run_coro(self._start_connect(connector, aio_ctx, conn_ctx))
        time.sleep(0.2)
        while not conn_ctx.conn and not conn_ctx.error:
            time.sleep(0.1)

        self.logger.debug("CLIENT: connect completed")
        if conn_ctx.error:
            raise CommError(CommError.ERROR, conn_ctx.error)

        self.add_connection(conn_ctx.conn)
        conn_ctx.waiter.wait()
        self.close_connection(conn_ctx.conn)

    def shutdown(self):
        if self.closing:
            return

        self.closing = True
        self.close_all()

        if self.server:
            aio_ctx = AioContext.get_global_context()
            aio_ctx.run_coro(self.server.shutdown())

    @staticmethod
    def get_urls(scheme: str, resources: dict) -> (str, str):
        secure = resources.get(DriverParams.SECURE)
        if secure:
            scheme = "grpcs"

        return get_tcp_urls(scheme, resources)

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
