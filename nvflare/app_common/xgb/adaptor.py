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

import threading
import time
from abc import ABC, abstractmethod

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal
from nvflare.fuel.utils.validation_utils import (
    check_non_negative_int,
    check_object_type,
    check_positive_int,
    check_positive_number,
)

from .defs import Constant
from .sender import Sender


class XGBAdaptor(ABC, FLComponent):
    """
    XGBAdaptors are used to integrate FLARE with XGBoost Target (Server or Client) in run time.

    For example, an XGB server could be run as a gRPC server process, or be run as part of the FLARE's FL server
    process. Similarly, an XGB client could be run as a gRPC client process, or be run as part of the
    FLARE's FL client process.

    Each type of XGB Target requires an appropriate adaptor to integrate it with FLARE's XGB Controller or Executor.

    The XGBAdaptor class defines commonly required methods for all adaptor implementations.
    """

    def __init__(self):
        FLComponent.__init__(self)
        self.abort_signal = None

    def set_abort_signal(self, abort_signal: Signal):
        """Called by XGB Controller/Executor to set the abort_signal.

        The abort_signal is assigned by FLARE's XGB Controller/Executor. It is used by the Controller/Executor
        to tell the adaptor that the job has been aborted.

        Args:
            abort_signal: the abort signal assigned by the caller.

        Returns: None

        """
        check_object_type("abort_signal", abort_signal, Signal)
        self.abort_signal = abort_signal

    @abstractmethod
    def start(self, fl_ctx: FLContext):
        """Called by XGB Controller/Executor to start the target.
        If any error occurs when starting the target, this method should raise an exception.

        Args:
            fl_ctx: the FL context.

        Returns: None

        """
        pass

    @abstractmethod
    def stop(self, fl_ctx: FLContext):
        """Called by XGB Controller/Executor to stop the target.
        If any error occurs when stopping the target, this method should raise an exception.

        Args:
            fl_ctx: the FL context.

        Returns: None

        """
        pass

    @abstractmethod
    def configure(self, config: dict, fl_ctx: FLContext):
        """Called by XGB Controller/Executor to configure the adaptor.
        If any error occurs, this method should raise an exception.

        Args:
            config: config data
            fl_ctx: the FL context

        Returns: None

        """
        pass

    @abstractmethod
    def _is_stopped(self) -> (bool, int):
        """Called by the adaptor's monitor to know whether the target is stopped.
        Note that this method is not called by XGB Controller/Executor.

        Returns: a tuple of: whether the target is stopped, and return code (if stopped)

        Note that a non-zero return code is considered abnormal completion of the target.

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
        """Called by XGB Controller/Executor to monitor the health of the target.

        The monitor periodically checks the abort signal. Once set, it calls the adaptor's stop() method
        to stop the running of the target.

        The monitor also periodically checks whether the target is already stopped (by calling the is_stopped
        method). If the target is stopped, the monitor will call the specified target_stopped_cb.

        Args:
            fl_ctx: FL context
            target_stopped_cb: the callback function to be called when the target is stopped.

        Returns: None

        """
        if not callable(target_stopped_cb):
            raise RuntimeError(f"target_stopped_cb must be callable but got {type(target_stopped_cb)}")

        # start the monitor in a separate daemon thread!
        t = threading.Thread(target=self._monitor, args=(fl_ctx, target_stopped_cb), daemon=True)
        t.start()


class XGBServerAdaptor(XGBAdaptor):
    """
    XGBServerAdaptor specifies commonly required methods for server adaptor implementations.
    """

    def __init__(self):
        XGBAdaptor.__init__(self)
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


class XGBClientAdaptor(XGBAdaptor):
    """
    XGBClientAdaptor specifies commonly required methods for client adaptor implementations.
    """

    def __init__(self, req_timeout: float):
        """Constructor of XGBClientAdaptor

        Args:
            req_timeout: timeout of XGB requests sent to server
        """
        XGBAdaptor.__init__(self)
        check_positive_number("req_timeout", req_timeout)
        self.req_timeout = req_timeout
        self.engine = None
        self.sender = None
        self.stopped = False
        self.rank = None
        self.num_rounds = None

    def configure(self, config: dict, fl_ctx: FLContext):
        """Called by XGB Executor to configure the target.

        The rank and number of rounds are required config parameters.

        Args:
            config: config data
            fl_ctx: FL context

        Returns: None

        """
        rank = config.get(Constant.CONF_KEY_RANK)
        check_non_negative_int(Constant.CONF_KEY_RANK, rank)

        num_rounds = config.get(Constant.CONF_KEY_NUM_ROUNDS)
        check_positive_int(Constant.CONF_KEY_NUM_ROUNDS, num_rounds)

        self.rank = rank
        self.num_rounds = num_rounds

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        """Handle FL events.
        Listen to the START_RUN event and create the Sender object to be used to send requests to server.

        Args:
            event_type: event_type to be handled
            fl_ctx: FL context

        Returns:

        """
        if event_type == EventType.START_RUN:
            self.engine = fl_ctx.get_engine()
            self.sender = Sender(self.engine, self.req_timeout)

    def _send_request(self, op: str, req: Shareable) -> bytes:
        """Send XGB operation request to the FL server via FLARE message.

        Args:
            op: the XGB operation
            req: operation data

        Returns: operation result

        """
        reply = self.sender.send_to_server(op, req, self.abort_signal)
        if isinstance(reply, Shareable):
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

        Returns: operation result

        """
        req = Shareable()
        req[Constant.PARAM_KEY_RANK] = rank
        req[Constant.PARAM_KEY_SEQ] = seq
        req[Constant.PARAM_KEY_SEND_BUF] = send_buf
        return self._send_request(Constant.OP_ALL_GATHER, req)

    def _send_all_gather_v(self, rank: int, seq: int, send_buf: bytes) -> bytes:
        """This method is called by a concrete client adaptor to send AllgatherV operation to the server.

        Args:
            rank: rank of the client
            seq: sequence number of the request
            send_buf: operation input

        Returns: operation result

        """
        req = Shareable()
        req[Constant.PARAM_KEY_RANK] = rank
        req[Constant.PARAM_KEY_SEQ] = seq
        req[Constant.PARAM_KEY_SEND_BUF] = send_buf
        return self._send_request(Constant.OP_ALL_GATHER_V, req)

    def _send_all_reduce(self, rank: int, seq: int, data_type: int, reduce_op: int, send_buf: bytes) -> bytes:
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

    def _send_broadcast(self, rank: int, seq: int, root: int, send_buf: bytes) -> bytes:
        """This method is called by a concrete client adaptor to send Broadcast operation to the server.

        Args:
            rank: rank of the client
            seq: sequence number of the request
            root: root rank of the broadcast
            send_buf: operation input

        Returns: operation result

        """
        req = Shareable()
        req[Constant.PARAM_KEY_RANK] = rank
        req[Constant.PARAM_KEY_SEQ] = seq
        req[Constant.PARAM_KEY_ROOT] = root
        req[Constant.PARAM_KEY_SEND_BUF] = send_buf
        return self._send_request(Constant.OP_BROADCAST, req)
