from nvflare.apis.signal import Signal
from nvflare.fuel.utils.obj_utils import get_logger
from nvflare.fuel.f3.cellnet.fqcn import FQCN
from nvflare.apis.shareable import Shareable, ReturnCode
from .defs import Constant


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
        reply_op = result.get_header(Constant.KEY_XGB_OP)
        if reply_op != expected_op:
            self.logger.error(f"received op {reply_op} != expected op {expected_op}")
            return None
        return result

    def send_to_server(self, op: str, req: Shareable, abort_signal: Signal):
        req.set_header(Constant.KEY_XGB_OP, op)

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
