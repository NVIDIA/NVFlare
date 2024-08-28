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

from abc import ABC, abstractmethod
from typing import Tuple

from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.apis.utils.reliable_message import PROP_KEY_DEBUG_INFO, ReliableMessage
from nvflare.app_opt.xgboost.histogram_based_v2.defs import Constant
from nvflare.fuel.f3.cellnet.fqcn import FQCN
from nvflare.fuel.utils.validation_utils import check_non_negative_int, check_positive_int

from .adaptor import AppAdaptor

XGB_APP_NAME = "XGBoost"


class XGBServerAdaptor(AppAdaptor):
    """XGBServerAdaptor specifies commonly required methods for server adaptor implementations.

    For example, an XGB server could be run as a gRPC server process, or be run as part of the FLARE's FL server
    process. Similarly, an XGB client could be run as a gRPC client process, or be run as part of the
    FLARE's FL client process.

    Each type of XGB Target requires an appropriate adaptor to integrate it with FLARE's XGB Controller or Executor.

    """

    def __init__(self, in_process: bool):
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


class XGBClientAdaptor(AppAdaptor, ABC):
    """XGBClientAdaptor specifies commonly required methods for client adaptor implementations."""

    def __init__(self, in_process: bool, per_msg_timeout: float, tx_timeout: float):
        """Constructor of XGBClientAdaptor.

        Args:
            in_process (bool):
            per_msg_timeout (float): Number of seconds to wait for each message before timing out.
            tx_timeout (float): Timeout for the entire transaction.
        """
        AppAdaptor.__init__(self, XGB_APP_NAME, in_process)
        self.engine = None
        self.stopped = False
        self.rank = None
        self.num_rounds = None
        self.data_split_mode = None
        self.secure_training = None
        self.xgb_params = None
        self.xgb_options = None
        self.disable_version_check = None
        self.world_size = None
        self.per_msg_timeout = per_msg_timeout
        self.tx_timeout = tx_timeout

    def _check_rank(self, ranks: dict, site_name: str):
        if ranks is None or not isinstance(ranks, dict):
            raise RuntimeError(f"{Constant.CONF_KEY_CLIENT_RANKS} is not configured.")

        ws = len(ranks)
        if ws == 0:
            raise RuntimeError(f"{Constant.CONF_KEY_CLIENT_RANKS} length is 0.")
        self.world_size = ws

        rank = ranks.get(site_name, None)
        if rank is None:
            raise RuntimeError(f"rank is not configured ({site_name})")

        check_non_negative_int(f"{Constant.CONF_KEY_CLIENT_RANKS}[{site_name}]", rank)
        self.rank = rank

    def configure(self, config: dict, fl_ctx: FLContext):
        """Called by XGB Executor to configure the target.

        The rank, world size, and number of rounds are required config parameters.

        Args:
            config: config data
            fl_ctx: FL context

        Returns: None

        """
        ranks = config.get(Constant.CONF_KEY_CLIENT_RANKS, None)
        site_name = fl_ctx.get_identity_name()
        self._check_rank(ranks, site_name)

        num_rounds = config.get(Constant.CONF_KEY_NUM_ROUNDS)
        if num_rounds is None or num_rounds <= 0:
            raise RuntimeError("num_rounds is not configured or invalid value")

        check_positive_int(Constant.CONF_KEY_NUM_ROUNDS, num_rounds)
        self.num_rounds = num_rounds

        self.data_split_mode = config.get(Constant.CONF_KEY_DATA_SPLIT_MODE)
        if self.data_split_mode is None:
            raise RuntimeError("data_split_mode is not configured")
        fl_ctx.set_prop(key=Constant.PARAM_KEY_DATA_SPLIT_MODE, value=self.data_split_mode, private=True, sticky=True)

        self.secure_training = config.get(Constant.CONF_KEY_SECURE_TRAINING)
        if self.secure_training is None:
            raise RuntimeError("secure_training is not configured")
        fl_ctx.set_prop(key=Constant.PARAM_KEY_SECURE_TRAINING, value=self.secure_training, private=True, sticky=True)

        self.xgb_params = config.get(Constant.CONF_KEY_XGB_PARAMS)
        if not self.xgb_params:
            raise RuntimeError("xgb_params is not configured")

        self.xgb_options = config.get(Constant.CONF_KEY_XGB_OPTIONS, {})

        self.disable_version_check = config.get(Constant.CONF_KEY_DISABLE_VERSION_CHECK)
        if self.disable_version_check is None:
            raise RuntimeError("disable_version_check is not configured")
        fl_ctx.set_prop(
            key=Constant.PARAM_KEY_DISABLE_VERSION_CHECK, value=self.disable_version_check, private=True, sticky=True
        )

    def _send_request(self, op: str, req: Shareable) -> Tuple[bytes, Shareable]:
        """Send XGB operation request to the FL server via FLARE message.

        Args:
            op: the XGB operation
            req: operation data

        Returns: operation result

        """
        req.set_header(Constant.MSG_KEY_XGB_OP, op)

        with self.engine.new_context() as fl_ctx:
            debug_info = {
                "op": op,
                "rank": req[Constant.PARAM_KEY_RANK],
                "seq": req[Constant.PARAM_KEY_SEQ],
            }
            fl_ctx.set_prop(key=PROP_KEY_DEBUG_INFO, value=debug_info, private=True, sticky=False)
            reply = ReliableMessage.send_request(
                target=FQCN.ROOT_SERVER,
                topic=Constant.TOPIC_XGB_REQUEST,
                request=req,
                per_msg_timeout=self.per_msg_timeout,
                tx_timeout=self.tx_timeout,
                abort_signal=self.abort_signal,
                fl_ctx=fl_ctx,
            )

        if isinstance(reply, Shareable):
            rc = reply.get_return_code()
            if rc != ReturnCode.OK:
                raise RuntimeError(f"received error return code: {rc}")

            reply_op = reply.get_header(Constant.MSG_KEY_XGB_OP)
            if reply_op != op:
                raise RuntimeError(f"received op {reply_op} != expected op {op}")

            rcv_buf = reply.get(Constant.PARAM_KEY_RCV_BUF)
            return rcv_buf, reply
        else:
            raise RuntimeError(f"invalid reply for op {op}: expect Shareable but got {type(reply)}")

    def _send_all_gather(self, rank: int, seq: int, send_buf: bytes) -> Tuple[bytes, Shareable]:
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

    def _send_all_gather_v(self, rank: int, seq: int, send_buf: bytes, headers=None) -> Tuple[bytes, Shareable]:
        req = Shareable()
        self._add_headers(req, headers)
        req[Constant.PARAM_KEY_RANK] = rank
        req[Constant.PARAM_KEY_SEQ] = seq
        req[Constant.PARAM_KEY_SEND_BUF] = send_buf
        return self._send_request(Constant.OP_ALL_GATHER_V, req)

    def _do_all_gather_v(self, rank: int, seq: int, send_buf: bytes) -> Tuple[bytes, Shareable]:
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

    def _send_broadcast(self, rank: int, seq: int, root: int, send_buf: bytes, headers=None) -> Tuple[bytes, Shareable]:
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
