import ctypes

from nvflare.app_common.xgb.bridge import XGBServerBridge
from nvflare.apis.fl_context import FLContext
from nvflare.app_common.xgb.defs import Constant
from nvflare.security.logging import secure_format_exception


class CServerBridge(XGBServerBridge):

    def __init__(
        self,
        lib_path: str,
    ):
        XGBServerBridge.__init__(self)
        self.stopped = False
        self.target_done = False
        self.target_rc = 0
        self.xgb_lib = ctypes.CDLL(lib_path)

        self.xgb_lib.xgbs_initialize.argtypes = [ctypes.c_int]
        self.xgb_lib.xgbs_initialize.restype = None

        self.xgb_lib.xgbs_all_gather.argtypes = [
            ctypes.c_int,                                       # int rank
            ctypes.c_int,                                       # int seq
            ctypes.POINTER(ctypes.c_ubyte),                     # unsigned char* send_buf
            ctypes.c_size_t,                                    # size_t send_size
            ctypes.POINTER(ctypes.POINTER(ctypes.c_ubyte)),     # unsigned char** rcv_buf
            ctypes.POINTER(ctypes.c_size_t),                    # size_t* rcv_size
        ]
        self.xgb_lib.xgbs_all_gather.restype = ctypes.c_int

        self.xgb_lib.xgbs_all_gather_v.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_ubyte),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.POINTER(ctypes.c_ubyte)),
            ctypes.POINTER(ctypes.c_size_t),
        ]
        self.xgb_lib.xgbs_all_gather_v.restype = ctypes.c_int

        self.xgb_lib.xgbs_all_reduce.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,   # int data_type
            ctypes.c_int,   # int reduce_op
            ctypes.POINTER(ctypes.c_ubyte),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.POINTER(ctypes.c_ubyte)),
            ctypes.POINTER(ctypes.c_size_t),
        ]
        self.xgb_lib.xgbs_all_reduce.restype = ctypes.c_int

        self.xgb_lib.xgbs_broadcast.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,  # int root
            ctypes.POINTER(ctypes.c_ubyte),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.POINTER(ctypes.c_ubyte)),
            ctypes.POINTER(ctypes.c_size_t),
        ]
        self.xgb_lib.xgbs_broadcast.restype = ctypes.c_int

        self.xgb_lib.xgbs_free_buf.argtypes = [
            ctypes.POINTER(ctypes.c_ubyte),
        ]
        self.xgb_lib.xgbs_free_buf.restype = None

        self.xgb_lib.xgbs_abort.argtypes = []
        self.xgb_lib.xgbs_abort.restype = None

    def start(self, fl_ctx: FLContext):
        try:
            self.xgb_lib.xgbs_initialize(self.world_size)
        except Exception as ex:
            self.system_panic(
                reason=f"cannot initialize C target: {secure_format_exception(ex)}",
                fl_ctx=fl_ctx,
            )
            return
        self.log_info(fl_ctx, "initialized C target")

    def stop(self, fl_ctx: FLContext):
        self._stop_target()

    def all_gather(self, rank: int, seq: int, send_buf: bytes, fl_ctx: FLContext) -> bytes:
        cbuf = (ctypes.c_ubyte * len(send_buf)).from_buffer(bytearray(send_buf))
        rcv_buf = ctypes.POINTER(ctypes.c_ubyte)()
        rcv_size = ctypes.c_size_t()
        rc = self.xgb_lib.xgbs_all_gather(
            rank, seq, cbuf, len(send_buf), ctypes.byref(rcv_buf), ctypes.byref(rcv_size))
        return self._process_result(rc, Constant.OP_ALL_GATHER, rank, seq, rcv_buf, rcv_size, fl_ctx)

    def all_gather_v(self, rank: int, seq: int, send_buf: bytes, fl_ctx: FLContext) -> bytes:
        cbuf = (ctypes.c_ubyte * len(send_buf)).from_buffer(bytearray(send_buf))
        rcv_buf = ctypes.POINTER(ctypes.c_ubyte)()
        rcv_size = ctypes.c_size_t()
        rc = self.xgb_lib.xgbs_all_gather_v(
            rank, seq, cbuf, len(send_buf), ctypes.byref(rcv_buf), ctypes.byref(rcv_size))
        return self._process_result(rc, Constant.OP_ALL_GATHER_V, rank, seq, rcv_buf, rcv_size, fl_ctx)

    def all_reduce(
            self,
            rank: int,
            seq: int,
            data_type: int,
            reduce_op: int,
            send_buf: bytes,
            fl_ctx: FLContext,
    ) -> bytes:
        cbuf = (ctypes.c_ubyte * len(send_buf)).from_buffer(bytearray(send_buf))
        rcv_buf = ctypes.POINTER(ctypes.c_ubyte)()
        rcv_size = ctypes.c_size_t()
        rc = self.xgb_lib.xgbs_all_reduce(
            rank, seq, data_type, reduce_op, cbuf, len(send_buf), ctypes.byref(rcv_buf), ctypes.byref(rcv_size))
        return self._process_result(rc, Constant.OP_ALL_REDUCE, rank, seq, rcv_buf, rcv_size, fl_ctx)

    def broadcast(self, rank: int, seq: int, root: int, send_buf: bytes, fl_ctx: FLContext) -> bytes:
        cbuf = (ctypes.c_ubyte * len(send_buf)).from_buffer(bytearray(send_buf))
        rcv_buf = ctypes.POINTER(ctypes.c_ubyte)()
        rcv_size = ctypes.c_size_t()
        rc = self.xgb_lib.xgbs_broadcast(
            rank, seq, root, cbuf, len(send_buf), ctypes.byref(rcv_buf), ctypes.byref(rcv_size))
        return self._process_result(rc, Constant.OP_BROADCAST, rank, seq, rcv_buf, rcv_size, fl_ctx)

    def _process_result(
            self,
            rc: int,
            op: str,
            rank: int,
            seq: int,
            rcv_buf,
            rcv_size,
            fl_ctx: FLContext):
        if rc != 0:
            self.log_error(fl_ctx, f"C target error {rc}: {op=} {rank=} {seq=}")
            self._stop_target()
            self.target_rc = Constant.ERR_TARGET_ERROR
            self.target_done = True
            raise RuntimeError("C target processing error")
        result = bytes(rcv_buf[0:rcv_size.value])
        self.xgb_lib.xgbs_free_buf(rcv_buf)
        return result

    def _stop_target(self):
        if self.stopped:
            return
        self.logger.info(f"stopping C target")
        self.xgb_lib.xgbs_abort()
        self.stopped = True

    def is_stopped(self) -> (bool, int):
        return self.target_done, self.target_rc
