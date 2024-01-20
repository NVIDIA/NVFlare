import threading
import time
from abc import ABC, abstractmethod

from nvflare.apis.signal import Signal
from nvflare.apis.shareable import Shareable
from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_context import FLContext
from nvflare.apis.event_type import EventType

from .sender import Sender
from .defs import Constant


class XGBBridge(ABC, FLComponent):

    def __init__(self):
        FLComponent.__init__(self)
        self.abort_signal = None

    def set_abort_signal(self, abort_signal: Signal):
        self.abort_signal = abort_signal

    @abstractmethod
    def start(self, fl_ctx: FLContext):
        pass

    @abstractmethod
    def stop(self, fl_ctx: FLContext):
        pass

    @abstractmethod
    def configure(self, config: dict, fl_ctx: FLContext):
        pass

    @abstractmethod
    def is_stopped(self) -> (bool, int):
        pass

    def _monitor(self, fl_ctx: FLContext, target_stopped_cb):
        while True:
            if self.abort_signal.triggered:
                # asked to abort
                self.stop(fl_ctx)
                return

            stopped, rc = self.is_stopped()
            if stopped:
                # target already stopped - notify the caller
                target_stopped_cb(rc, fl_ctx)
                return

            time.sleep(0.1)

    def monitor_target(self, fl_ctx: FLContext, target_stopped_cb):
        t = threading.Thread(target=self._monitor, args=(fl_ctx, target_stopped_cb), daemon=True)
        t.start()


class XGBServerBridge(XGBBridge):

    def __init__(self):
        XGBBridge.__init__(self)
        self.world_size = None

    def configure(self, config: dict, fl_ctx: FLContext):
        ws = config.get(Constant.CONF_KEY_WORLD_SIZE)
        if not ws:
            raise RuntimeError("world_size is not configured")

        if not isinstance(ws, int):
            raise RuntimeError(f"invalid world_size config: expect int but got {type(ws)}")

        if ws <= 0:
            raise RuntimeError(f"invalid world_size config: must > 0 but got {ws}")

        self.world_size = ws

    @abstractmethod
    def all_gather(self, rank: int, seq: int, send_buf: bytes, fl_ctx: FLContext) -> bytes:
        pass

    @abstractmethod
    def all_gather_v(self, rank: int, seq: int, send_buf: bytes, fl_ctx: FLContext) -> bytes:
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
        pass

    @abstractmethod
    def broadcast(self, rank: int, seq: int, root: int, send_buf: bytes, fl_ctx: FLContext) -> bytes:
        pass


class XGBClientBridge(XGBBridge):

    def __init__(self, req_timeout: float):
        XGBBridge.__init__(self)
        self.req_timeout = req_timeout
        self.engine = None
        self.sender = None
        self.stopped = False
        self.rank = None
        self.num_rounds = None

    def configure(self, config: dict, fl_ctx: FLContext):
        rank = config.get(Constant.CONF_KEY_RANK)
        if rank is None:
            raise RuntimeError("missing rank in config")

        if not isinstance(rank, int):
            raise RuntimeError(f"invalid rank in config - expect int but got {type(rank)}")

        if rank < 0:
            raise RuntimeError(f"invalid rank in config - must >= 0 but got {rank}")

        num_rounds = config.get(Constant.CONF_KEY_NUM_ROUNDS)
        if num_rounds is None:
            raise RuntimeError("missing num_rounds in config")

        if not isinstance(num_rounds, int):
            raise RuntimeError(f"invalid num_rounds in config - expect int but got {type(num_rounds)}")

        if num_rounds <= 0:
            raise RuntimeError(f"invalid num_rounds in config - must > 0 but got {num_rounds}")

        self.rank = rank
        self.num_rounds = num_rounds

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.START_RUN:
            self.engine = fl_ctx.get_engine()
            self.sender = Sender(self.engine, self.req_timeout)

    def _send_request(self, op: str, req: Shareable) -> bytes:
        reply = self.sender.send_to_server(op, req, self.abort_signal)
        if isinstance(reply, Shareable):
            rcv_buf = reply.get(Constant.PARAM_KEY_RCV_BUF)
            if not isinstance(rcv_buf, bytes):
                raise RuntimeError(f"invalid reply for op {op}: expect bytes but got {type(rcv_buf)}")
            return rcv_buf
        else:
            raise RuntimeError(f"invalid reply for op {op}: expect Shareable but got {type(reply)}")

    def send_all_gather(self, rank: int, seq: int, send_buf: bytes) -> bytes:
        req = Shareable()
        req[Constant.PARAM_KEY_RANK] = rank
        req[Constant.PARAM_KEY_SEQ] = seq
        req[Constant.PARAM_KEY_SEND_BUF] = send_buf
        return self._send_request(Constant.OP_ALL_GATHER, req)

    def send_all_gather_v(self, rank: int, seq: int, send_buf: bytes) -> bytes:
        req = Shareable()
        req[Constant.PARAM_KEY_RANK] = rank
        req[Constant.PARAM_KEY_SEQ] = seq
        req[Constant.PARAM_KEY_SEND_BUF] = send_buf
        return self._send_request(Constant.OP_ALL_GATHER_V, req)

    def send_all_reduce(self, rank: int, seq: int, data_type: int, reduce_op: int, send_buf: bytes) -> bytes:
        req = Shareable()
        req[Constant.PARAM_KEY_RANK] = rank
        req[Constant.PARAM_KEY_SEQ] = seq
        req[Constant.PARAM_KEY_DATA_TYPE] = data_type
        req[Constant.PARAM_KEY_REDUCE_OP] = reduce_op
        req[Constant.PARAM_KEY_SEND_BUF] = send_buf
        return self._send_request(Constant.OP_ALL_REDUCE, req)

    def send_broadcast(self, rank: int, seq: int, root: int, send_buf: bytes) -> bytes:
        req = Shareable()
        req[Constant.PARAM_KEY_RANK] = rank
        req[Constant.PARAM_KEY_SEQ] = seq
        req[Constant.PARAM_KEY_ROOT] = root
        req[Constant.PARAM_KEY_SEND_BUF] = send_buf
        return self._send_request(Constant.OP_BROADCAST, req)
