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

from concurrent import futures

import grpc
import logging
import threading
from typing import Union, List, Dict, Any

from nvflare.fuel.f3.comm_config import CommConfigurator
from nvflare.fuel.f3.comm_error import CommError
from nvflare.fuel.f3.connection import Connection
from nvflare.fuel.f3.drivers.driver import Connector
from .base_driver import BaseDriver
from .driver_params import DriverParams, DriverCap

from .grpc.qq import QQ
from nvflare.fuel.f3.drivers.grpc.streamer_pb2_grpc import (
    StreamerServicer, add_StreamerServicer_to_server, StreamerStub
)
from .grpc.streamer_pb2 import Frame
from .net_utils import get_address, get_tcp_urls

MAX_MSG_SIZE = 1024 * 1024 * 1024    # 1G

GRPC_DEFAULT_OPTIONS = [
    ("grpc.max_send_message_length", MAX_MSG_SIZE),
    ("grpc.max_receive_message_length", MAX_MSG_SIZE),
]


class StreamConnection(Connection):

    seq_num = 0

    def __init__(self, oq: QQ, connector: Connector, conn_props: dict, side: str, context=None, channel=None):
        super().__init__(connector)
        self.side = side
        self.oq = oq
        self.closing = False
        self.conn_props = conn_props
        self.context = context    # for server side
        self.channel = channel    # for client side
        self.lock = threading.Lock()
        self.logger = logging.getLogger(self.__class__.__name__)

    def get_conn_properties(self) -> dict:
        return self.conn_props

    def close(self):
        self.closing = True
        with self.lock:
            self.oq.close()
            if self.context:
                self.context.abort(grpc.StatusCode.CANCELLED, "service closed")
                self.context = None
            if self.channel:
                self.channel.close()
                self.channel = None

    def send_frame(self, frame: Union[bytes, bytearray, memoryview]):
        try:
            StreamConnection.seq_num += 1
            seq = StreamConnection.seq_num
            self.logger.debug(f"{self.side}: queued frame #{seq}")
            self.oq.append(Frame(seq=seq, data=bytes(frame)))
        except BaseException as ex:
            raise CommError(CommError.ERROR, f"Error sending frame: {ex}")

    def read_loop(self, msg_iter, q: QQ):
        ct = threading.current_thread()
        self.logger.debug(f"{self.side}: started read_loop in thread {ct.name}")
        try:
            for f in msg_iter:
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
            if q:
                self.logger.debug(f"{self.side}: closing queue")
                q.close()
        self.logger.debug(f"{self.side} in {ct.name}: done read_loop")

    def generate_output(self):
        ct = threading.current_thread()
        self.logger.debug(f"{self.side}: generate_output in thread {ct.name}")
        for i in self.oq:
            assert isinstance(i, Frame)
            self.logger.debug(f"{self.side}: outgoing frame #{i.seq}")
            yield i
        self.logger.debug(f"{self.side}: done generate_output in thread {ct.name}")


class Servicer(StreamerServicer):

    def __init__(self, server):
        self.server = server
        self.logger = logging.getLogger(self.__class__.__name__)

    def Stream(self, request_iterator, context):
        connection = None
        oq = QQ()
        t = None
        ct = threading.current_thread()
        try:
            self.logger.debug(f"SERVER started Stream CB in thread {ct.name}")
            conn_props = {DriverParams.PEER_ADDR.value: context.peer(),
                          DriverParams.LOCAL_ADDR.value: get_address(self.server.connector.params)}
            connection = StreamConnection(oq, self.server.connector, conn_props, "SERVER", context=context)
            self.logger.debug(f"SERVER created connection in thread {ct.name}")
            self.server.driver.add_connection(connection)
            self.logger.debug(f"SERVER created read_loop thread in thread {ct.name}")
            t = threading.Thread(target=connection.read_loop, args=(request_iterator, oq))
            t.start()

            # DO NOT use connection.generate_output()!
            self.logger.debug(f"SERVER: generate_output in thread {ct.name}")
            for i in oq:
                assert isinstance(i, Frame)
                self.logger.debug(f"SERVER: outgoing frame #{i.seq}")
                yield i
            self.logger.debug(f"SERVER: done generate_output in thread {ct.name}")

        except BaseException as ex:
            self.logger.error(f"Connection closed due to error: {ex}")
        finally:
            if t is not None:
                t.join()
            if connection:
                self.logger.debug(f"SERVER: closing connection {connection.name}")
                self.server.driver.close_connection(connection)
            self.logger.debug(f"SERVER: cleanly finished Stream CB in thread {ct.name}")


class Server:

    def __init__(
            self,
            driver,
            connector,
            max_workers,
            options,
    ):
        self.driver = driver
        self.connector = connector
        self.grpc_server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=max_workers),
            options=options
        )
        servicer = Servicer(self)
        add_StreamerServicer_to_server(servicer, self.grpc_server)

        params = connector.params
        host = params.get(DriverParams.HOST.value)
        port = int(params.get(DriverParams.PORT.value))
        if not host:
            host = 0
        self.grpc_server.add_insecure_port(f'{host}:{port}')

    def start(self):
        self.grpc_server.start()
        self.grpc_server.wait_for_termination()

    def shutdown(self):
        self.grpc_server.stop(grace=0.5)


class GrpcDriver(BaseDriver):

    def __init__(self):
        super().__init__()
        self.server = None
        self.max_workers = 10
        self.options = GRPC_DEFAULT_OPTIONS
        self.logger = logging.getLogger(self.__class__.__name__)
        configurator = CommConfigurator()
        config = configurator.get_config()
        if config:
            my_params = config.get("grpc")
            if my_params:
                self.max_workers = my_params.get("max_workers", 10)
                self.options = my_params.get("options")
        self.logger.debug(f"GRPC Config: max_workers={self.max_workers}, options={self.options}")

    @staticmethod
    def supported_transports() -> List[str]:
        return ["rpc"]

    @staticmethod
    def capabilities() -> Dict[str, Any]:
        return {
            DriverCap.HEARTBEAT.value: True,
            DriverCap.SUPPORT_SSL.value: False
        }

    def listen(self, connector: Connector):
        self.connector = connector
        self.server = Server(self, connector, max_workers=self.max_workers, options=self.options)
        self.server.start()

    def connect(self, connector: Connector):
        self.connector = connector
        params = connector.params
        address = get_address(params)

        channel = grpc.insecure_channel(address, options=self.options)
        stub = StreamerStub(channel)

        conn_props = {DriverParams.PEER_ADDR.value: address}

        oq = QQ()
        connection = StreamConnection(oq, connector, conn_props, "CLIENT", channel=channel)
        self.add_connection(connection)
        try:
            received = stub.Stream(connection.generate_output())
            connection.read_loop(received, oq)
        except BaseException as ex:
            self.logger.info(f"CLIENT: connection done: {type(ex)}")
        connection.close()
        self.close_connection(connection)
        self.logger.debug("CLIENT: finished connection")

    def shutdown(self):
        self.close_all()
        if self.server:
            self.server.shutdown()

    @staticmethod
    def get_urls(scheme: str, resources: dict) -> (str, str):
        return get_tcp_urls(scheme, resources)

