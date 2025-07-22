# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
import concurrent.futures.thread
import threading

import grpc

from nvflare.edge.constants import EdgeApiStatus
from nvflare.fuel.f3.drivers.aio_context import AioContext
from nvflare.fuel.utils.log_utils import get_obj_logger
from nvflare.security.logging import secure_format_exception

from .edge_api_pb2 import Reply, Request
from .edge_api_pb2_grpc import EdgeApiServicer, add_EdgeApiServicer_to_server
from .query_handler import QueryHandler
from .utils import make_reply


class Servicer(EdgeApiServicer):

    def __init__(self, handler: QueryHandler, aio_ctx: AioContext, max_workers=100):
        self.logger = get_obj_logger(self)
        self.handler = handler
        self.aio_ctx = aio_ctx
        self.worker_pool = concurrent.futures.thread.ThreadPoolExecutor(max_workers=max_workers)

    async def Query(self, request: Request, context) -> Reply:
        try:
            loop = self.aio_ctx.get_event_loop()
            reply = await loop.run_in_executor(self.worker_pool, self.handler.handle_query, request)
            if not reply:
                raise RuntimeError("no result from QueryHandler.")
            return reply
        except Exception as ex:
            self.logger.error(f"error processing request: {secure_format_exception(ex)}")
            return make_reply(EdgeApiStatus.ERROR)


class EdgeApiServer:

    def __init__(
        self,
        handler: QueryHandler,
        address: str,
        grpc_options=None,
        max_workers=100,
        ssl_credentials=None,
    ):
        self.aio_ctx = AioContext.get_global_context()
        self.logger = get_obj_logger(self)
        self.handler = handler
        self.address = address
        self.grpc_options = grpc_options
        self.max_workers = max_workers
        self.grpc_server = None
        self.grpc_server_stop_grace = 0.5
        self.waiter = threading.Event()
        self.root_cert = None
        self.cert_chain = None
        self.private_key = None
        self.ssl_credentials = ssl_credentials

    async def _start(self):
        # Note: the AIO grpc server must be created in this coro, because it has to be created in the thread
        # that runs the event loop!
        self.logger.info("starting Edge API Server ...")
        self.grpc_server = grpc.aio.server(options=self.grpc_options)
        servicer = Servicer(self.handler, self.aio_ctx, self.max_workers)
        add_EdgeApiServicer_to_server(servicer, self.grpc_server)

        if self.ssl_credentials:
            # one-way SSL
            self.logger.info(f"adding secure port at {self.address} for 1-way ssl")
            self.grpc_server.add_secure_port(self.address, server_credentials=self.ssl_credentials)
            self.logger.info(f"added secure port at {self.address}")
        else:
            self.grpc_server.add_insecure_port(self.address)
            self.logger.info(f"added insecure port at {self.address}")

        self.logger.info("starting server engine")
        await self.grpc_server.start()
        self.logger.info("started server and wait for termination")
        await self.grpc_server.wait_for_termination()

    async def _shutdown(self):
        try:
            await self.grpc_server.stop(grace=self.grpc_server_stop_grace)

            # Note that self.grpc_server.stop returns immediately. Since we gave 0.5 grace time for RPCs to end,
            # we wait here until RPCs are done or aborted.
            # Without this, we may run into "excepthook" error at the end of the program since the GRPC server isn't
            # properly shutdown.
            await asyncio.sleep(self.grpc_server_stop_grace)
            self.grpc_server = None
            self.logger.debug("Server is stopped!")
        except Exception as ex:
            self.logger.debug(f"exception shutdown server: {secure_format_exception(ex)}")

    def start(self):
        self.aio_ctx.run_coro(self._start())
        self.logger.info("waiting for server to finish")
        self.waiter.wait()
        self.logger.info("server is done")

    def shutdown(self):
        self.aio_ctx.run_coro(self._shutdown())
        self.waiter.set()
        self.logger.info("Shutting Down Server")
