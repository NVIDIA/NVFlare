# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

import concurrent.futures as futures

import grpc
from flwr.proto.grpcadapter_pb2_grpc import GrpcAdapterServicer, add_GrpcAdapterServicer_to_server

from nvflare.app_opt.flower.defs import GRPC_DEFAULT_OPTIONS
from nvflare.fuel.utils.obj_utils import get_logger
from nvflare.fuel.utils.validation_utils import check_object_type, check_positive_int
from nvflare.security.logging import secure_format_exception


class GrpcServer:
    """This class implements a gRPC Server that is capable of processing Flower requests."""

    def __init__(self, addr, max_workers: int, grpc_options, servicer):
        """Constructor

        Args:
            addr: the listening address of the server
            max_workers: max number of workers
            grpc_options: gRPC options
            servicer: the servicer that is capable of processing Flower requests
        """
        if not grpc_options:
            grpc_options = GRPC_DEFAULT_OPTIONS

        check_object_type("servicer", servicer, GrpcAdapterServicer)
        check_positive_int("max_workers", max_workers)
        self.grpc_server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers), options=grpc_options)
        add_GrpcAdapterServicer_to_server(servicer, self.grpc_server)
        self.logger = get_logger(self)

        try:
            # TBD: will be enhanced to support secure port
            self.grpc_server.add_insecure_port(addr)
            self.logger.info(f"Flower gRPC Server: added insecure port at {addr}")
        except Exception as ex:
            self.logger.error(f"cannot listen on {addr}: {secure_format_exception(ex)}")

    def start(self, no_blocking=False):
        """Called to start the server

        Args:
            no_blocking: whether blocking the current thread and wait for server termination

        Returns: None

        """
        self.logger.info("starting Flower gRPC Server")
        self.grpc_server.start()
        if no_blocking:
            # don't wait for server termination
            return
        else:
            self.grpc_server.wait_for_termination()
            self.logger.info("Flower gRPC server terminated")

    def shutdown(self):
        """Shut down the gRPC server gracefully.

        Returns:

        """
        self.logger.info("shutting down Flower gRPC server")
        server = self.grpc_server
        self.grpc_server = None  # in case another thread calls shutdown at the same time
        if server:
            server.stop(grace=0.5)
