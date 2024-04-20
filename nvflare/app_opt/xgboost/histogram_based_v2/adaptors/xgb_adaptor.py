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
from abc import abstractmethod

from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.app_opt.xgboost.histogram_based_v2.defs import Constant
from nvflare.app_opt.xgboost.histogram_based_v2.sender import Sender
from nvflare.fuel.utils.validation_utils import check_non_negative_int, check_positive_int

from .adaptor import AppAdaptor

XGB_APP_NAME = "XGBoost"


class XGBServerAdaptor(AppAdaptor):
    """
    XGBServerAdaptor specifies commonly required methods for server adaptor implementations.
    """

    def __init__(self, in_process):
        AppAdaptor.__init__(self, XGB_APP_NAME, in_process)
        self.world_size = None

    def configure(self, config: dict, fl_ctx: FLContext):
        """Called by XGB Controller to configure the target.

        The world_size is a required config parameter.

        Args:
            config: config data
            fl_ctx: FL context

        Returns: None

        """
        ws = config.get(Constant.CONF_KEY_WORLD_SIZE)
        if not ws:
            raise RuntimeError("world_size is not configured")

        check_positive_int(Constant.CONF_KEY_WORLD_SIZE, ws)
        self.world_size = ws

    @abstractmethod
    def all_gather(self, rank: int, seq: int, send_buf: bytes, fl_ctx: FLContext) -> bytes:
        """Called by the XGB Controller to perform Allgather operation, per XGBoost spec.

        Args:
            rank: rank of the calling client
            seq: sequence number of the request
            send_buf: operation input data
            fl_ctx: FL context

        Returns: operation result

        """
        pass

    @abstractmethod
    def all_gather_v(self, rank: int, seq: int, send_buf: bytes, fl_ctx: FLContext) -> bytes:
        """Called by the XGB Controller to perform AllgatherV operation, per XGBoost spec.

        Args:
            rank: rank of the calling client
            seq: sequence number of the request
            send_buf: input data
            fl_ctx: FL context

        Returns: operation result

        """
        pass

    @abstractmethod
    def all_reduce(
        self,
        rank: int,
        seq: int,
        data_type: int,
        reduce_op: int,
        send_buf: bytes,
        fl_ctx: FLContext,
    ) -> bytes:
        """Called by the XGB Controller to perform Allreduce operation, per XGBoost spec.

        Args:
            rank: rank of the calling client
            seq: sequence number of the request
            data_type: data type of the input
            reduce_op: reduce operation to be performed
            send_buf: input data
            fl_ctx: FL context

        Returns: operation result

        """
        pass

    @abstractmethod
    def broadcast(self, rank: int, seq: int, root: int, send_buf: bytes, fl_ctx: FLContext) -> bytes:
        """Called by the XGB Controller to perform Broadcast operation, per XGBoost spec.

        Args:
            rank: rank of the calling client
            seq: sequence number of the request
            root: root rank of the broadcast
            send_buf: input data
            fl_ctx: FL context

        Returns: operation result

        """
        pass


class XGBClientAdaptor(AppAdaptor):
    """
    XGBClientAdaptor specifies commonly required methods for client adaptor implementations.
    """

    def __init__(self, in_process):
        """Constructor of XGBClientAdaptor"""
        AppAdaptor.__init__(self, XGB_APP_NAME, in_process)
        self.engine = None
        self.sender = None
        self.stopped = False
        self.rank = None
        self.num_rounds = None
        self.world_size = None

    def set_sender(self, sender: Sender):
        """Set the sender to be used to send XGB operation requests to the server.

        Args:
            sender: the sender to be set

        Returns: None

        """
        if not isinstance(sender, Sender):
            raise TypeError(f"sender must be Sender but got {type(sender)}")
        self.sender = sender

    def configure(self, config: dict, fl_ctx: FLContext):
        """Called by XGB Executor to configure the target.

        The rank, world size, and number of rounds are required config parameters.

        Args:
            config: config data
            fl_ctx: FL context

        Returns: None

        """
        ws = config.get(Constant.CONF_KEY_WORLD_SIZE)
        if not ws:
            raise RuntimeError("world_size is not configured")

        check_positive_int(Constant.CONF_KEY_WORLD_SIZE, ws)
        self.world_size = ws

        rank = config.get(Constant.CONF_KEY_RANK)
        if rank is None:
            raise RuntimeError("rank is not configured")

        check_non_negative_int(Constant.CONF_KEY_RANK, rank)
        self.rank = rank

        num_rounds = config.get(Constant.CONF_KEY_NUM_ROUNDS)
        if num_rounds is None:
            raise RuntimeError("num_rounds is not configured")

        check_positive_int(Constant.CONF_KEY_NUM_ROUNDS, num_rounds)
        self.num_rounds = num_rounds

    def _send_request(self, op: str, req: Shareable) -> (bytes, Shareable):
        """Send XGB operation request to the FL server via FLARE message.

        Args:
            op: the XGB operation
            req: operation data

        Returns: operation result

        """
        reply = self.sender.send_to_server(op, req, self.abort_signal)
        if isinstance(reply, Shareable):
            rcv_buf = reply.get(Constant.PARAM_KEY_RCV_BUF)
            return rcv_buf, reply
        else:
            raise RuntimeError(f"invalid reply for op {op}: expect Shareable but got {type(reply)}")

    def _send_all_gather(self, rank: int, seq: int, send_buf: bytes) -> (bytes, Shareable):
        """This method is called by a concrete client adaptor to send Allgather operation to the server.

        Args:
            rank: rank of the client
            seq: sequence number of the request
            send_buf: input data

        Returns: operation result

        """
        req = Shareable()
        req[Constant.PARAM_KEY_RANK] = rank
        req[Constant.PARAM_KEY_SEQ] = seq
        req[Constant.PARAM_KEY_SEND_BUF] = send_buf
        return self._send_request(Constant.OP_ALL_GATHER, req)

    def _send_all_gather_v(self, rank: int, seq: int, send_buf: bytes, headers=None) -> (bytes, Shareable):
        req = Shareable()
        self._add_headers(req, headers)
        req[Constant.PARAM_KEY_RANK] = rank
        req[Constant.PARAM_KEY_SEQ] = seq
        req[Constant.PARAM_KEY_SEND_BUF] = send_buf
        return self._send_request(Constant.OP_ALL_GATHER_V, req)

    def _do_all_gather_v(self, rank: int, seq: int, send_buf: bytes) -> (bytes, Shareable):
        """This method is called by a concrete client adaptor to send AllgatherV operation to the server.

        Args:
            rank: rank of the client
            seq: sequence number of the request
            send_buf: operation input

        Returns: operation result

        """
        fl_ctx = self.engine.new_context()
        fl_ctx.set_prop(key=Constant.PARAM_KEY_RANK, value=rank, private=True, sticky=False)
        fl_ctx.set_prop(key=Constant.PARAM_KEY_SEQ, value=seq, private=True, sticky=False)
        fl_ctx.set_prop(key=Constant.PARAM_KEY_SEND_BUF, value=send_buf, private=True, sticky=False)
        self.fire_event(Constant.EVENT_BEFORE_ALL_GATHER_V, fl_ctx)

        send_buf = fl_ctx.get_prop(Constant.PARAM_KEY_SEND_BUF)
        rcv_buf, reply = self._send_all_gather_v(
            rank=rank,
            seq=seq,
            send_buf=send_buf,
            headers=fl_ctx.get_prop(Constant.PARAM_KEY_HEADERS),
        )

        fl_ctx.set_prop(key=Constant.PARAM_KEY_RCV_BUF, value=rcv_buf, private=True, sticky=False)
        fl_ctx.set_prop(key=Constant.PARAM_KEY_REPLY, value=reply, private=True, sticky=False)
        self.fire_event(Constant.EVENT_AFTER_ALL_GATHER_V, fl_ctx)
        return fl_ctx.get_prop(Constant.PARAM_KEY_RCV_BUF)

    def _send_all_reduce(
        self, rank: int, seq: int, data_type: int, reduce_op: int, send_buf: bytes
    ) -> (bytes, Shareable):
        """This method is called by a concrete client adaptor to send Allreduce operation to the server.

        Args:
            rank: rank of the client
            seq: sequence number of the request
            data_type: data type of the input
            reduce_op: reduce operation to be performed
            send_buf: operation input

        Returns: operation result

        """
        req = Shareable()
        req[Constant.PARAM_KEY_RANK] = rank
        req[Constant.PARAM_KEY_SEQ] = seq
        req[Constant.PARAM_KEY_DATA_TYPE] = data_type
        req[Constant.PARAM_KEY_REDUCE_OP] = reduce_op
        req[Constant.PARAM_KEY_SEND_BUF] = send_buf
        return self._send_request(Constant.OP_ALL_REDUCE, req)

    def _send_broadcast(self, rank: int, seq: int, root: int, send_buf: bytes, headers=None) -> (bytes, Shareable):
        req = Shareable()
        self._add_headers(req, headers)
        req[Constant.PARAM_KEY_RANK] = rank
        req[Constant.PARAM_KEY_SEQ] = seq
        req[Constant.PARAM_KEY_ROOT] = root
        req[Constant.PARAM_KEY_SEND_BUF] = send_buf
        return self._send_request(Constant.OP_BROADCAST, req)

    def _do_broadcast(self, rank: int, seq: int, root: int, send_buf: bytes) -> bytes:
        """This method is called by a concrete client adaptor to send Broadcast operation to the server.

        Args:
            rank: rank of the client
            seq: sequence number of the request
            root: root rank of the broadcast
            send_buf: operation input

        Returns: operation result

        """
        fl_ctx = self.engine.new_context()
        fl_ctx.set_prop(key=Constant.PARAM_KEY_RANK, value=rank, private=True, sticky=False)
        fl_ctx.set_prop(key=Constant.PARAM_KEY_SEQ, value=seq, private=True, sticky=False)
        fl_ctx.set_prop(key=Constant.PARAM_KEY_ROOT, value=root, private=True, sticky=False)
        fl_ctx.set_prop(key=Constant.PARAM_KEY_SEND_BUF, value=send_buf, private=True, sticky=False)
        self.fire_event(Constant.EVENT_BEFORE_BROADCAST, fl_ctx)

        send_buf = fl_ctx.get_prop(Constant.PARAM_KEY_SEND_BUF)
        rcv_buf, reply = self._send_broadcast(
            rank=rank,
            seq=seq,
            root=root,
            send_buf=send_buf,
            headers=fl_ctx.get_prop(Constant.PARAM_KEY_HEADERS),
        )

        fl_ctx.set_prop(key=Constant.PARAM_KEY_RCV_BUF, value=rcv_buf, private=True, sticky=False)
        fl_ctx.set_prop(key=Constant.PARAM_KEY_REPLY, value=reply, private=True, sticky=False)
        self.fire_event(Constant.EVENT_AFTER_BROADCAST, fl_ctx)
        return fl_ctx.get_prop(Constant.PARAM_KEY_RCV_BUF)

    @staticmethod
    def _add_headers(req: Shareable, headers: dict):
        if not headers:
            return

        for k, v in headers.items():
            req.set_header(k, v)
