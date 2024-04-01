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

import threading
import time
from abc import ABC, abstractmethod
from typing import Tuple

from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal
from nvflare.apis.utils.reliable_message import ReliableMessage
from nvflare.app_opt.xgboost.histogram_based_v2.defs import Constant
from nvflare.app_opt.xgboost.histogram_based_v2.runner import XGBRunner
from nvflare.fuel.f3.cellnet.fqcn import FQCN
from nvflare.fuel.utils.validation_utils import check_non_negative_int, check_object_type, check_positive_int


class XGBAdaptor(ABC, FLComponent):
    """XGBAdaptors are used to integrate FLARE with XGBoost Target (Server or Client) in run time.

    For example, an XGB server could be run as a separate gRPC server process,
    or be run as part of the FLARE's FL server job process. Similarly, an XGB
    client could be run as a separate gRPC client process, or be run as part
    of the FLARE's FL client process.

    Each type of XGB Target requires an appropriate adaptor to integrate it with FLARE's XGB Controller or Executor.

    The XGBAdaptor class defines commonly required methods for all adaptor implementations.
    """

    def __init__(self):
        FLComponent.__init__(self)
        self.abort_signal = None
        self.xgb_runner = None

    def set_runner(self, runner: XGBRunner):
        """Sets the XGBRunner that will be used to run XGB processing logic.
        Note that the adaptor is only responsible for starting the runner.

        Args:
            runner: the runner to be set

        Returns: None

        """
        if not isinstance(runner, XGBRunner):
            raise TypeError(f"runner must be XGBRunner but got {type(runner)}")
        self.xgb_runner = runner

    def set_abort_signal(self, abort_signal: Signal):
        """Sets the abort_signal.

        The abort_signal is used by FLARE's XGB Controller/Executor.
        to tell the adaptor that the job has been aborted.

        Args:
            abort_signal: the abort signal assigned by the caller.

        Returns: None

        """
        check_object_type("abort_signal", abort_signal, Signal)
        self.abort_signal = abort_signal

    def initialize(self, fl_ctx: FLContext):
        """Initializes the adaptor.

        Args:
            fl_ctx: the FL context

        Returns: None

        """
        pass

    @abstractmethod
    def start(self, fl_ctx: FLContext):
        """Starts the target.
        If any error occurs when starting the target, this method should raise an exception.

        Args:
            fl_ctx: the FL context.

        Returns: None

        """
        pass

    @abstractmethod
    def stop(self, fl_ctx: FLContext):
        """Stops the target.
        If any error occurs when stopping the target, this method should raise an exception.

        Args:
            fl_ctx: the FL context.

        Returns: None

        """
        pass

    @abstractmethod
    def configure(self, config: dict, fl_ctx: FLContext):
        """Configures the target.
        If any error occurs, this method should raise an exception.

        Args:
            config: config data
            fl_ctx: the FL context

        Returns: None

        """
        pass

    @abstractmethod
    def _is_stopped(self) -> Tuple[bool, int]:
        """Checks if the target is stopped.

        Note:
            This method is not called by XGB Controller/Executor but is used by
            the monitor thread.
            A non-zero return code is considered an abnormal completion of the target.

        Returns:
            Tuple[bool, int]: A tuple indicating whether the target is stopped and the
            return code (if stopped).

        """
        pass

    def _monitor(self, fl_ctx: FLContext, target_stopped_cb):
        while True:
            if self.abort_signal.triggered:
                # asked to abort
                self.stop(fl_ctx)
                return

            stopped, rc = self._is_stopped()
            if stopped:
                # target already stopped - notify the caller
                target_stopped_cb(rc, fl_ctx)
                return

            time.sleep(0.1)

    def monitor_target(self, fl_ctx: FLContext, target_stopped_cb):
        """Starts a monitor thread to check and respond to the health of the target.

        The monitor thread periodically checks for the abort signal.
        If set, it triggers the `stop()` method to halt the target.

        Additionally, the monitor checks at intervals whether the target has already stopped
        (by invoking the `_is_stopped()` method). If the target is detected as stopped,
        the monitor calls the specified target_stopped_cb callback.

        Args:
            fl_ctx (FLContext): The Federated Learning context.
            target_stopped_cb (callable): The callback function to be executed when the target is stopped.

        Returns:
            None

        Raises:
            RuntimeError: If target_stopped_cb is not a callable function.

        Note:
            This method starts the monitor in a separate daemon thread to run concurrently.
        """
        if not callable(target_stopped_cb):
            raise RuntimeError(f"target_stopped_cb must be callable but got {type(target_stopped_cb)}")

        # start the monitor in a separate daemon thread!
        t = threading.Thread(target=self._monitor, args=(fl_ctx, target_stopped_cb), daemon=True)
        t.start()


class XGBServerAdaptor(XGBAdaptor, ABC):
    """
    XGBServerAdaptor specifies commonly required methods for server adaptor implementations.
    """

    def __init__(self):
        XGBAdaptor.__init__(self)
        self.world_size = None

    def configure(self, config: dict, fl_ctx: FLContext):
        """Configures the target.

        Args:
            config (dict): configuration to be used to configure the target.
            fl_ctx: FL context

        Returns:
            None

        Raises:
            RuntimeError: if world_size is not configured.

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

        Returns:
            operation result
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

        Returns:
            operation result
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

        Returns:
            operation result
        """
        pass


class XGBClientAdaptor(XGBAdaptor, ABC):
    """
    XGBClientAdaptor specifies commonly required methods for client adaptor implementations.
    """

    def __init__(self, per_msg_timeout: float, tx_timeout: float):
        """Constructor of XGBClientAdaptor"""
        XGBAdaptor.__init__(self)
        self.engine = None
        self.stopped = False
        self.rank = None
        self.num_rounds = None
        self.world_size = None
        self.per_msg_timeout = per_msg_timeout
        self.tx_timeout = tx_timeout

    def configure(self, config: dict, fl_ctx: FLContext):
        """Called by XGB Executor to configure the target.

        The rank, world size, and number of rounds are required config parameters.

        Args:
            config: config data
            fl_ctx: FL context

        Returns:
            None
        """
        self.engine = fl_ctx.get_engine()

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

    def _send_request(self, op: str, req: Shareable) -> bytes:
        """Send XGB operation request to the FL server via FLARE message.

        Args:
            op: the XGB operation
            req: operation data

        Returns:
            operation result
        """
        req.set_header(Constant.MSG_KEY_XGB_OP, op)

        with self.engine.new_context() as fl_ctx:
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
            if not isinstance(rcv_buf, bytes):
                raise RuntimeError(f"invalid rcv_buf for {op=}: expect bytes but got {type(rcv_buf)}")
            return rcv_buf
        else:
            raise RuntimeError(f"invalid reply for op {op}: expect Shareable but got {type(reply)}")

    def _send_all_gather(self, rank: int, seq: int, send_buf: bytes) -> bytes:
        """This method is called by a concrete client adaptor to send Allgather operation to the server.

        Args:
            rank: rank of the client
            seq: sequence number of the request
            send_buf: input data

        Returns:
            operation result
        """
        req = Shareable()
        req[Constant.PARAM_KEY_RANK] = rank
        req[Constant.PARAM_KEY_SEQ] = seq
        req[Constant.PARAM_KEY_SEND_BUF] = send_buf
        return self._send_request(Constant.OP_ALL_GATHER, req)

    def _send_all_reduce(self, rank: int, seq: int, data_type: int, reduce_op: int, send_buf: bytes) -> bytes:
        """This method is called by a concrete client adaptor to send Allreduce operation to the server.

        Args:
            rank: rank of the client
            seq: sequence number of the request
            data_type: data type of the input
            reduce_op: reduce operation to be performed
            send_buf: operation input

        Returns:
            operation result
        """
        req = Shareable()
        req[Constant.PARAM_KEY_RANK] = rank
        req[Constant.PARAM_KEY_SEQ] = seq
        req[Constant.PARAM_KEY_DATA_TYPE] = data_type
        req[Constant.PARAM_KEY_REDUCE_OP] = reduce_op
        req[Constant.PARAM_KEY_SEND_BUF] = send_buf
        return self._send_request(Constant.OP_ALL_REDUCE, req)

    def _send_broadcast(self, rank: int, seq: int, root: int, send_buf: bytes) -> bytes:
        """This method is called by a concrete client adaptor to send Broadcast operation to the server.

        Args:
            rank: rank of the client
            seq: sequence number of the request
            root: root rank of the broadcast
            send_buf: operation input

        Returns:
            operation result
        """
        req = Shareable()
        req[Constant.PARAM_KEY_RANK] = rank
        req[Constant.PARAM_KEY_SEQ] = seq
        req[Constant.PARAM_KEY_ROOT] = root
        req[Constant.PARAM_KEY_SEND_BUF] = send_buf
        return self._send_request(Constant.OP_BROADCAST, req)
