import ctypes
import threading
import time

from nvflare.app_common.xgb.defs import Constant
from nvflare.app_common.xgb.bridge import XGBClientBridge
from nvflare.apis.fl_context import FLContext
from nvflare.security.logging import secure_format_exception


class CClientBridge(XGBClientBridge):

    _xgb_lib = None
    _max_num_clients = 10

    @classmethod
    def _load_lib(cls, path: str):
        if cls._xgb_lib:
            # already loaded
            return

        cls._xgb_lib = ctypes.CDLL(path)

        cls._xgb_lib.xgbc_initialize.argtypes = [ctypes.c_int]
        cls._xgb_lib.xgbc_initialize.restype = None

        cls._xgb_lib.xgbc_new_client.argtypes = [ctypes.c_int]  # rank
        cls._xgb_lib.xgbc_new_client.restype = ctypes.c_int

        cls._xgb_lib.xgbc_get_pending_op.argtypes = [
            ctypes.c_int,   # rank
            ctypes.POINTER(ctypes.c_int),  # int* seq
            ctypes.POINTER(ctypes.POINTER(ctypes.c_ubyte)),  # unsigned char** send_buf
            ctypes.POINTER(ctypes.c_size_t),  # size_t* send_size
            ctypes.POINTER(ctypes.c_int),  # int* data_type
            ctypes.POINTER(ctypes.c_int),  # int* reduce_op
            ctypes.POINTER(ctypes.c_int),  # int* root
        ]
        cls._xgb_lib.xgbc_get_pending_op.restype = ctypes.c_int

        cls._xgb_lib.xgbc_reply.argtypes = [
            ctypes.c_int,                       # int op
            ctypes.c_int,                       # int rank
            ctypes.POINTER(ctypes.c_ubyte),     # unsigned char* rcv_buf
            ctypes.c_size_t,                    # size_t rcv_size
        ]
        cls._xgb_lib.xgbc_reply.restype = ctypes.c_int

        cls._xgb_lib.xgbc_start.argtypes = [ctypes.c_int, ctypes.c_int]  # rank, num_rounds
        cls._xgb_lib.xgbc_start.restype = ctypes.c_int

        cls._xgb_lib.xgbc_abort.argtypes = [ctypes.c_int]  # rank
        cls._xgb_lib.xgbc_start.restype = None

        cls._xgb_lib.xgbc_initialize(cls._max_num_clients)  # we may get _max_num_clients from config later

    def __init__(
            self,
            lib_path: str,
            req_timeout=10.0):
        XGBClientBridge.__init__(self, req_timeout)

        # TBD - need to handle c shared lib path with env vars or Pathlib
        self._load_lib(lib_path)
        self.stopped = False
        self.target_done = False
        self.target_rc = 0
        self.op_table = {
            Constant.OPCODE_ALL_GATHER: self._handle_all_gather,
            Constant.OPCODE_ALL_GATHER_V: self._handle_all_gather_v,
            Constant.OPCODE_ALL_REDUCE: self._handle_all_reduce,
            Constant.OPCODE_BROADCAST: self._handle_broadcast,
        }

    def configure(self, config: dict, fl_ctx: FLContext):
        super().configure(config, fl_ctx)
        rc = self._xgb_lib.xgbc_new_client(self.rank)
        if rc != 0:
            self.system_panic(
                reason=f"cannot create client for rank {self.rank}: {rc}",
                fl_ctx=fl_ctx,
            )
            return
        self.log_info(fl_ctx, f"created client for rank {self.rank}")

    def _get_pending_op(self) -> (int, dict):
        send_buf = ctypes.POINTER(ctypes.c_ubyte)()
        send_size = ctypes.c_size_t()
        seq = ctypes.c_int()
        data_type = ctypes.c_int()
        reduce_op = ctypes.c_int()
        root = ctypes.c_int()
        op = self._xgb_lib.xgbc_get_pending_op(
            self.rank, ctypes.byref(seq), ctypes.byref(send_buf), ctypes.byref(send_size),
            ctypes.byref(data_type), ctypes.byref(reduce_op), ctypes.byref(root),
        )
        if op in [Constant.OPCODE_DONE, Constant.OPCODE_NONE]:
            # no pending op
            return op, {}

        if op < 0:
            raise RuntimeError(f"error get_pending_op: {op}")

        props = {
            Constant.PARAM_KEY_SEQ: seq.value,
            Constant.PARAM_KEY_SEND_BUF: bytes(send_buf[0:send_size.value]),
            Constant.PARAM_KEY_DATA_TYPE: data_type.value,
            Constant.PARAM_KEY_REDUCE_OP: reduce_op.value,
            Constant.PARAM_KEY_ROOT: root.value,
        }
        return op, props

    def _send_reply(self, op: int, rcv_buf: bytes):
        cbuf = (ctypes.c_ubyte * len(rcv_buf)).from_buffer(bytearray(rcv_buf))
        rc = self._xgb_lib.xgbc_reply(op, self.rank, cbuf, len(rcv_buf))
        self.logger.info(f"xgbc_reply for op {op} rank {self.rank}: {rc}")
        if rc != 0:
            raise RuntimeError(f"error _send_reply: {rc}")

    def _start_xgb_client(self):
        self._xgb_lib.xgbc_start(self.rank, self.num_rounds)

    def start(self, fl_ctx: FLContext):
        # we start the XGB client in a separate process since we cannot block here
        t = threading.Thread(target=self._start_xgb_client, daemon=True)
        t.start()

        # start a thread to poll requests from the C client
        p = threading.Thread(target=self._poll_requests, daemon=True)
        p.start()
        self.log_info(fl_ctx, "started C client")

    def _handle_all_gather(self, op: int, params: dict):
        seq = params.get(Constant.PARAM_KEY_SEQ)
        send_buf = params.get(Constant.PARAM_KEY_SEND_BUF)
        self.logger.info(f"client {self.rank}: got all_gather {seq=} {len(send_buf)=}")
        return self.send_all_gather(self.rank, seq, send_buf)

    def _handle_all_gather_v(self, op: int, params: dict):
        seq = params.get(Constant.PARAM_KEY_SEQ)
        send_buf = params.get(Constant.PARAM_KEY_SEND_BUF)
        self.logger.info(f"client {self.rank}: got all_gather_v {seq=} {len(send_buf)=}")
        return self.send_all_gather_v(self.rank, seq, send_buf)

    def _handle_all_reduce(self, op: int, params: dict):
        seq = params.get(Constant.PARAM_KEY_SEQ)
        send_buf = params.get(Constant.PARAM_KEY_SEND_BUF)
        data_type = params.get(Constant.PARAM_KEY_DATA_TYPE)
        reduce_op = params.get(Constant.PARAM_KEY_REDUCE_OP)
        self.logger.info(f"client {self.rank}: got all_reduce {seq=} {len(send_buf)=}")
        return self.send_all_reduce(self.rank, seq, data_type, reduce_op, send_buf)

    def _handle_broadcast(self, op: int, params: dict):
        seq = params.get(Constant.PARAM_KEY_SEQ)
        send_buf = params.get(Constant.PARAM_KEY_SEND_BUF)
        root = params.get(Constant.PARAM_KEY_ROOT)
        self.logger.info(f"client {self.rank}: got broadcast {seq=} {len(send_buf)=}")
        return self.send_broadcast(self.rank, seq, root, send_buf)

    def _poll_requests(self):
        # poll requests from C side
        while True:
            if self.abort_signal.triggered or self.stopped:
                self.target_done = True
                return

            op, params = self._get_pending_op()
            if op == Constant.OPCODE_DONE:
                self.logger.info(f"End of client {self.rank}: C side stopped")
                self.target_rc = 0
                self.target_done = True
                return

            if op != Constant.OPCODE_NONE:
                self.logger.info(f"client {self.rank}: got op {op}")
                handler_f = self.op_table.get(op)
                if handler_f is None:
                    self.logger.error(f"no handler for opcode {op}")
                    self.target_rc = Constant.ERR_CLIENT_ERROR
                    self.target_done = True
                    return
                else:
                    try:
                        assert callable(handler_f)
                        rcv_buf = handler_f(op, params)
                        self._send_reply(op, rcv_buf)
                    except Exception as ex:
                        self.logger.error(f"exception handling all_gather: {secure_format_exception(ex)}")
                        self._stop_target()
                        self.target_rc = Constant.ERR_CLIENT_ERROR
                        self.target_done = True
                        return
            time.sleep(0.001)

    def is_stopped(self) -> (bool, int):
        return self.target_done, self.target_rc

    def _stop_target(self):
        if self.stopped:
            return
        self.logger.info(f"stopping C target of rank {self.rank}")
        self._xgb_lib.xgbc_abort(self.rank)
        self.stopped = True

    def stop(self, fl_ctx: FLContext):
        self._stop_target()
