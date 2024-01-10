from nvflare.apis.executor import Executor
from nvflare.apis.fl_context import FLContext
from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.signal import Signal
from nvflare.apis.event_type import EventType
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.fuel.utils.obj_utils import get_logger
from nvflare.fuel.f3.cellnet.fqcn import FQCN
from nvflare.security.logging import secure_format_exception
from nvflare.fuel.f3.drivers.net_utils import get_open_tcp_port
from .defs import Constant
from .process_manager import ProcessManager
import nvflare.app_common.xgb.proto.federated_pb2 as pb2
from nvflare.app_common.xgb.xgb.server import EchoServicer, XGBServer
import time
import threading


class Sender:

    def __init__(self, engine, timeout):
        self.engine = engine
        self.timeout = timeout
        self.logger = get_logger(self)

    def _extract_result(self, reply, expected_op):
        if not reply:
            return None
        if not isinstance(reply, dict):
            self.logger.error(f"expect reply to be a dict but got {type(reply)}")
            return None
        result = reply.get(FQCN.ROOT_SERVER)
        if not result:
            self.logger.error(f"no reply from {FQCN.ROOT_SERVER} for request {expected_op}")
            return None
        if not isinstance(result, Shareable):
            self.logger.error(f"expect result to be a Shareable but got {type(result)}")
            return None
        rc = result.get_return_code()
        if rc != ReturnCode.OK:
            self.logger.error(f"server failed to process request: {rc=}")
            return None
        reply_op = result.get(Constant.KEY_XGB_OP)
        if reply_op != expected_op:
            self.logger.error(f"received op {reply_op} != expected op {expected_op}")
            return None
        return result.get(Constant.KEY_XGB_MSG)

    def send_to_server(self, op: str, msg):
        req = Shareable()
        req[Constant.KEY_XGB_MSG] = msg
        req[Constant.KEY_XGB_OP] = op
        server_name = FQCN.ROOT_SERVER
        with self.engine.new_context() as fl_ctx:
            reply = self.engine.send_aux_request(
                targets=[server_name],
                topic=Constant.TOPIC_XGB_REQUEST,
                request=req,
                timeout=self.timeout,
                fl_ctx=fl_ctx
            )
        return self._extract_result(reply, op)


class LocalServicer(EchoServicer):
    def __init__(self, sender: Sender, comm_failed_cb, **cb_kwargs):
        EchoServicer.__init__(self)
        self.sender = sender
        self.comm_failed_cb = comm_failed_cb
        self.cb_kwargs = cb_kwargs
        self.is_broken = False

    def _process_request(self, op, request, reply_cls):
        self.logger.info(f"received request '{op}' from external XGB client")
        if self.is_broken:
            return None

        serialized_req = request.SerializeToString()
        reply = self.sender.send_to_server(op=op, msg=serialized_req)
        if reply:
            try:
                return reply_cls.FromString(reply)
            except Exception as ex:
                self.logger.error(f"cannot parse reply to {reply_cls}: {secure_format_exception(ex)}")
                reply = None
        if not reply:
            self.is_broken = True
            self.comm_failed_cb(**self.cb_kwargs)
        self.logger.info(f"sent response '{op}' back to external XGB client")
        return reply

    def Allgather(self, request: pb2.AllgatherRequest, context):
        return self._process_request(Constant.OP_ALL_GATHER, request, pb2.AllgatherReply)

    def AllgatherV(self, request: pb2.AllgatherVRequest, context):
        return self._process_request(Constant.OP_ALL_GATHER_V, request, pb2.AllgatherVReply)

    def Allreduce(self, request: pb2.AllreduceRequest, context):
        return self._process_request(Constant.OP_ALL_REDUCE, request, pb2.AllreduceReply)

    def Broadcast(self, request: pb2.BroadcastRequest, context):
        return self._process_request(Constant.OP_BROADCAST, request, pb2.BroadcastReply)


class XGBExecutor(Executor):

    def __init__(
            self,
            run_xgb_client_cmd: str,
            internal_server_addr=None,
            configure_task_name=Constant.CONFIG_TASK_NAME,
            start_task_name=Constant.START_TASK_NAME,
            grpc_options=None,
            req_timeout=10.0,
    ):
        Executor.__init__(self)
        self.xgb_client_manager = None
        self.internal_xgb_server = None
        self.internal_server_addr = internal_server_addr
        self.grpc_options = grpc_options
        self.req_timeout = req_timeout
        self.configure_task_name = configure_task_name
        self.start_task_name = start_task_name
        self.run_xgb_client_cmd = run_xgb_client_cmd
        self.asked_to_stop = False
        self.my_rank = None
        xgb_monitor = threading.Thread(target=self._monitor_xgb, daemon=True)
        xgb_monitor.start()

    def _monitor_xgb(self):
        while not self.asked_to_stop:
            time.sleep(0.5)
        self._stop_xgb()

    def _xgb_failed(self):
        self.asked_to_stop = True

    def _start_xgb(self, fl_ctx: FLContext):
        if not self.internal_server_addr:
            # dynamically determine address on localhost
            port = get_open_tcp_port(resources={})
            if not port:
                self.system_panic("failed to get a port for XGB server", fl_ctx)
                return
            self.internal_server_addr = f"127.0.0.1:{port}"

        self.run_xgb_client_cmd = self.run_xgb_client_cmd.replace("$addr", self.internal_server_addr)
        self.run_xgb_client_cmd = self.run_xgb_client_cmd.replace("$rank", str(self.my_rank))

        self.logger.info(f"Start internal server at {self.internal_server_addr}")
        sender = Sender(fl_ctx.get_engine(), self.req_timeout)
        servicer = LocalServicer(sender, self._xgb_failed)
        self.internal_xgb_server = XGBServer(self.internal_server_addr, 10, self.grpc_options, servicer)
        self.internal_xgb_server.start(no_blocking=True)
        self.logger.info(f"Started internal server at {self.internal_server_addr}")

        self.xgb_client_manager = ProcessManager(
            name="XGBClient",
            start_cmd=self.run_xgb_client_cmd,
            stopped_cb=self._xgb_client_stopped,
            fl_ctx=fl_ctx,
        )
        self.xgb_client_manager.start()
        self.logger.info(f"Started external XGB Client")

    def _xgb_client_stopped(self, rc, fl_ctx: FLContext):
        if rc != 0:
            self.log_error(fl_ctx, f"XGB Client stopped with RC {rc}")
        else:
            self.log_info(fl_ctx, "XGB Client Stopped")

        # tell server that XGB is done
        engine = fl_ctx.get_engine()
        req = Shareable()
        req[Constant.KEY_EXIT_CODE] = rc
        engine.send_aux_request(
            targets=[FQCN.ROOT_SERVER],
            topic=Constant.TOPIC_CLIENT_DONE,
            request=req,
            timeout=0,   # fire and forget
            fl_ctx=fl_ctx,
            optional=True,
        )

    def _stop_xgb(self):
        if self.xgb_client_manager:
            self.logger.info("Stop external XGB client")
            self.xgb_client_manager.stop()

        if self.internal_xgb_server:
            self.logger.info("Stop internal XGB Server")
            self.internal_xgb_server.shutdown()

        self.started = False

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.END_RUN:
            self.asked_to_stop = True

    def execute(self, task_name: str, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        if task_name == self.configure_task_name:
            ranks = shareable.get(Constant.KEY_CLIENT_RANKS)
            if not ranks:
                self.log_error(fl_ctx, f"missing {Constant.KEY_CLIENT_RANKS} from config")
                return make_reply(ReturnCode.BAD_TASK_DATA)

            if not isinstance(ranks, dict):
                self.log_error(fl_ctx, f"expect config data to be dict but got {ranks}")
                return make_reply(ReturnCode.BAD_TASK_DATA)

            me = fl_ctx.get_identity_name()
            self.my_rank = ranks.get(me)
            if self.my_rank is None:
                self.log_error(fl_ctx, f"missing rank for me ({me}) in config data")
                return make_reply(ReturnCode.BAD_TASK_DATA)

            self.log_info(fl_ctx, f"got my rank: {self.my_rank}")
            return make_reply(ReturnCode.OK)
        elif task_name == self.start_task_name:
            # create and start grpc server
            self._start_xgb(fl_ctx)
            return make_reply(ReturnCode.OK)
        else:
            self.log_error(fl_ctx, f"ignored unsupported {task_name}")
            return make_reply(ReturnCode.TASK_UNSUPPORTED)
