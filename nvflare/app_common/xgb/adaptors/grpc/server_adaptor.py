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

import nvflare.app_common.xgb.adaptors.grpc.proto.federated_pb2 as pb2
from nvflare.apis.fl_context import FLContext
from nvflare.app_common.xgb.adaptor import XGBServerAdaptor
from nvflare.app_common.xgb.adaptors.grpc.client import XGBClient
from nvflare.app_common.xgb.defs import Constant
from nvflare.app_common.xgb.process_manager import ProcessManager
from nvflare.fuel.f3.drivers.net_utils import get_open_tcp_port
from nvflare.fuel.utils.validation_utils import check_str


class GrpcServerAdaptor(XGBServerAdaptor):
    def __init__(
        self,
        run_xgb_server_cmd: str,
        xgb_server_addr=None,
        xgb_server_ready_timeout=Constant.XGB_SERVER_READY_TIMEOUT,
    ):
        XGBServerAdaptor.__init__(self)
        self.run_xgb_server_cmd = run_xgb_server_cmd
        self.xgb_server_addr = xgb_server_addr
        self.xgb_server_ready_timeout = xgb_server_ready_timeout
        self.internal_xgb_client = None
        self.xgb_server_manager = None
        check_str("run_xgb_server_cmd", run_xgb_server_cmd)

    def start(self, fl_ctx: FLContext):
        if not self.xgb_server_addr:
            # we dynamically create server address on localhost
            port = get_open_tcp_port(resources={})
            if not port:
                raise RuntimeError("failed to get a port for XGB server")
            self.xgb_server_addr = f"127.0.0.1:{port}"

        self.run_xgb_server_cmd = self.run_xgb_server_cmd.replace("$addr", self.xgb_server_addr)
        self.run_xgb_server_cmd = self.run_xgb_server_cmd.replace("$num_clients", str(self.world_size))

        self.xgb_server_manager = ProcessManager(
            name="XGBServer",
            start_cmd=self.run_xgb_server_cmd,
        )
        self.xgb_server_manager.start()

        # start XGB client
        self.internal_xgb_client = XGBClient(self.xgb_server_addr)
        self.internal_xgb_client.start(ready_timeout=self.xgb_server_ready_timeout)

    def stop(self, fl_ctx: FLContext):
        client = self.internal_xgb_client
        self.internal_xgb_client = None
        if client:
            self.log_info(fl_ctx, "Stopping internal XGB client")
            client.stop()

        mgr = self.xgb_server_manager
        self.xgb_server_manager = None
        if mgr:
            # stop the XGB server
            self.log_info(fl_ctx, "Stopping XGB Server Monitor")
            mgr.stop()

    def _is_stopped(self) -> (bool, int):
        if self.xgb_server_manager:
            return self.xgb_server_manager.is_stopped()
        else:
            return True, 0

    def all_gather(self, rank: int, seq: int, send_buf: bytes, fl_ctx: FLContext) -> bytes:
        assert isinstance(self.internal_xgb_client, XGBClient)
        result = self.internal_xgb_client.send_allgather(seq_num=seq, rank=rank, data=send_buf)
        if isinstance(result, pb2.AllgatherReply):
            return result.receive_buffer
        else:
            raise RuntimeError(f"bad result from XGB server: expect AllgatherReply but got {type(result)}")

    def all_gather_v(self, rank: int, seq: int, send_buf: bytes, fl_ctx: FLContext) -> bytes:
        assert isinstance(self.internal_xgb_client, XGBClient)
        result = self.internal_xgb_client.send_allgatherv(seq_num=seq, rank=rank, data=send_buf)
        if isinstance(result, pb2.AllgatherVReply):
            return result.receive_buffer
        else:
            raise RuntimeError(f"bad result from XGB server: expect AllgatherVReply but got {type(result)}")

    def all_reduce(
        self,
        rank: int,
        seq: int,
        data_type: int,
        reduce_op: int,
        send_buf: bytes,
        fl_ctx: FLContext,
    ) -> bytes:
        assert isinstance(self.internal_xgb_client, XGBClient)
        result = self.internal_xgb_client.send_allreduce(
            seq_num=seq,
            rank=rank,
            data=send_buf,
            data_type=data_type,
            reduce_op=reduce_op,
        )
        if isinstance(result, pb2.AllreduceReply):
            return result.receive_buffer
        else:
            raise RuntimeError(f"bad result from XGB server: expect AllreduceReply but got {type(result)}")

    def broadcast(self, rank: int, seq: int, root: int, send_buf: bytes, fl_ctx: FLContext) -> bytes:
        assert isinstance(self.internal_xgb_client, XGBClient)
        result = self.internal_xgb_client.send_broadcast(seq_num=seq, rank=rank, data=send_buf, root=root)
        if isinstance(result, pb2.BroadcastReply):
            return result.receive_buffer
        else:
            raise RuntimeError(f"bad result from XGB server: expect BroadcastReply but got {type(result)}")
