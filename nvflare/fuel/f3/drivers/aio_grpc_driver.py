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
import os
import random
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
from .grpc.utils import get_grpc_client_credentials, get_grpc_server_credentials, use_aio_grpc
from .net_utils import MAX_FRAME_SIZE, get_address, get_tcp_urls, ssl_required

GRPC_DEFAULT_OPTIONS = [
    ("grpc.max_send_message_length", MAX_FRAME_SIZE),
    ("grpc.max_receive_message_length", MAX_FRAME_SIZE),
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
        self.read_task = None
        self.lock = threading.Lock()

        conf = CommConfigurator()
        if conf.get_bool_var("simulate_unstable_network", default=False):
            if context:
                # only server side
                self.disconn = threading.Thread(target=self._disconnect, daemon=True)
                self.disconn.start()

    def _disconnect(self):
        t = random.randint(10, 60)
        self.logger.info(f"will close connection after {t} secs")
        time.sleep(t)
        self.logger.info(f"close connection now after {t} secs")
        self.close()

    def get_conn_properties(self) -> dict:
        return self.conn_props

    async def _abort(self):
        try:
            self.context.abort(grpc.StatusCode.CANCELLED, "service closed")
        except:
            # ignore exception (if any) when aborting
            pass

    def close(self):
        self.closing = True
        with self.lock:
            if self.read_task:
                try:
                    self.logger.info("canceling read_output: connection is closed")
                    self.read_task.cancel()
                except Exception as ex:
                    self.logger.debug(f"exception cancelling read task: {secure_format_exception(ex)}")
                self.read_task = None

            if self.context:
                self.aio_ctx.run_coro(self._abort())
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
                item = await self.read_oq()
                yield item
        except Exception as ex:
            if self.closing:
                self.logger.debug(f"{self}: connection closed by {type(ex)}: {secure_format_exception(ex)}")
            else:
                self.logger.debug(f"{self}: generate_output exception {type(ex)}: {secure_format_exception(ex)}")
            self.logger.debug(secure_format_traceback())
            raise StopIteration()

    async def read_oq(self):
        # self.oq.get() does not return before an item is placed into the queue. This could cause it to wait for
        # a long time. If the connection is closed during this time, the coro won't be done immediately.
        # To be able to cancel the queue read operation, we wrap it into a task so that we can cancel it when
        # closing the connection. Once cancelled, "await task" will finish with asyncio.CancelledError exception.
        with self.lock:
            if self.closing:
                raise asyncio.CancelledError("cancelled read_oq: connection closed")

            # wrap the queue read into a task
            task = asyncio.create_task(self.oq.get())
            self.read_task = task
        return await task


class Servicer(StreamerServicer):
    def __init__(self, server, aio_ctx: AioContext):
        self.server = server
        self.aio_ctx = aio_ctx
        self.logger = get_logger(self)

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
            self.aio_ctx.run_coro(connection.read_loop(request_iterator))
            while True:
                item = await connection.read_oq()
                yield item
        except asyncio.CancelledError:
            self.logger.info("SERVER: RPC cancelled")
        except Exception as ex:
            self.logger.info(f"Connection {connection} closed due to: {secure_format_exception(ex)}")
            self.logger.debug(secure_format_traceback())
        finally:
            if connection:
                connection.close()
                self.logger.debug(f"SERVER: closing connection {connection}")
                self.server.driver.close_connection(connection)
            self.logger.info("SERVER: finished Stream CB")


class Server:
    def __init__(self, driver, connector, aio_ctx: AioContext, options, conn_ctx: _ConnCtx):
        self.logger = get_logger(self)
        self.driver = driver
        self.connector = connector
        self.grpc_server = grpc.aio.server(options=options)
        self.grpc_server_stop_grace = 0.5
        servicer = Servicer(self, aio_ctx)
        add_StreamerServicer_to_server(servicer, self.grpc_server)
        params = connector.params
        addr = get_address(params)
        try:
            self.logger.debug(f"SERVER: connector params: {params}")

            secure = ssl_required(params)
            if secure:
                credentials = get_grpc_server_credentials(params)
                self.grpc_server.add_secure_port(addr, server_credentials=credentials)
                self.logger.info(f"added secure port at {addr}")
            else:
                self.grpc_server.add_insecure_port(addr)
                self.logger.info(f"added insecure port at {addr}")
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
            await self.grpc_server.stop(grace=self.grpc_server_stop_grace)

            # Note that self.grpc_server.stop returns immediately. Since we gave 0.5 grace time for RPCs to end,
            # we wait here until RPCs are done or aborted.
            # Without this, we may run into "excepthook" error at the end of the program since the GRPC server isn't
            # properly shutdown.
            await asyncio.sleep(self.grpc_server_stop_grace)
            self.grpc_server = None
            self.logger.debug("GRPC Server is stopped!")
        except Exception as ex:
            self.logger.debug(f"exception shutdown server: {secure_format_exception(ex)}")


class AioGrpcDriver(BaseDriver):

    aio_ctx = None

    def __init__(self):
        super().__init__()
        # GRPC with fork issue: https://github.com/grpc/grpc/issues/28557
        os.environ["GRPC_ENABLE_FORK_SUPPORT"] = "False"

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
        if use_aio_grpc():
            return ["grpc", "grpcs"]
        else:
            return ["agrpc", "agrpcs"]

    @staticmethod
    def capabilities() -> Dict[str, Any]:
        return {DriverCap.SEND_HEARTBEAT.value: True, DriverCap.SUPPORT_SSL.value: True}

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
        connection = None
        try:
            secure = ssl_required(params)
            if secure:
                channel = grpc.aio.secure_channel(
                    address, options=self.options, credentials=get_grpc_client_credentials(params)
                )
                self.logger.info(f"created secure channel at {address}")
            else:
                channel = grpc.aio.insecure_channel(address, options=self.options)
                self.logger.info(f"created insecure channel at {address}")

            self.logger.debug(f"CLIENT: connected to {address}")
            stub = StreamerStub(channel)
            conn_props = {DriverParams.PEER_ADDR.value: address}

            if secure:
                conn_props[DriverParams.PEER_CN.value] = "N/A"

            connection = AioStreamSession(aio_ctx=aio_ctx, connector=connector, conn_props=conn_props, channel=channel)
            self.logger.debug(f"CLIENT: start streaming on connection {connection}")
            msg_iter = stub.Stream(connection.generate_output())
            conn_ctx.conn = connection
            await connection.read_loop(msg_iter)
        except asyncio.CancelledError:
            self.logger.info("CLIENT: RPC cancelled")
        except Exception as ex:
            conn_ctx.error = f"connection {connection} error: {type(ex)}: {secure_format_exception(ex)}"
            self.logger.debug(conn_ctx.error)
            self.logger.debug(secure_format_traceback())
        finally:
            if connection:
                connection.close()
        self.logger.info(f"finished connection {connection}")
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
            self.logger.debug("Start shutting down AIO grpc server ...")
            aio_ctx.run_coro(self.server.shutdown())
            time.sleep(self.server.grpc_server_stop_grace + 0.1)
            self.logger.debug("Finished shutting down AIO grpc server")

    @staticmethod
    def get_urls(scheme: str, resources: dict) -> (str, str):
        secure = resources.get(DriverParams.SECURE)
        if secure:
            if use_aio_grpc():
                scheme = "grpcs"
            else:
                scheme = "agrpcs"

        return get_tcp_urls(scheme, resources)
