from nvflare.apis.executor import Executor
from nvflare.apis.fl_context import FLContext
from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.signal import Signal
from nvflare.apis.event_type import EventType
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.app_common.xgb.bridge import XGBClientBridge
from nvflare.security.logging import secure_format_exception
from nvflare.fuel.f3.cellnet.fqcn import FQCN
from .defs import Constant


class XGBExecutor(Executor):

    def __init__(
            self,
            bridge_component_id: str,
            configure_task_name=Constant.CONFIG_TASK_NAME,
            start_task_name=Constant.START_TASK_NAME,
            req_timeout=10.0,
    ):
        Executor.__init__(self)
        self.bridge_component_id = bridge_component_id
        self.req_timeout = req_timeout
        self.configure_task_name = configure_task_name
        self.start_task_name = start_task_name
        self.bridge = None
        self.abort_signal = Signal()

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.START_RUN:
            engine = fl_ctx.get_engine()
            bridge = engine.get_component(self.bridge_component_id)
            if not bridge:
                self.system_panic(f"cannot get component for {self.bridge_component_id}", fl_ctx)
                return

            if not isinstance(bridge, XGBClientBridge):
                self.system_panic(
                    f"invalid component for {self.bridge_component_id}: expect XGBClientBridge but got {type(bridge)}",
                    fl_ctx)
                return

            bridge.set_abort_signal(self.abort_signal)
            self.bridge = bridge
        elif event_type == EventType.END_RUN:
            self.abort_signal.trigger(True)

    def execute(self, task_name: str, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        if task_name == self.configure_task_name:
            ranks = shareable.get(Constant.CONF_KEY_CLIENT_RANKS)
            if not ranks:
                self.log_error(fl_ctx, f"missing {Constant.CONF_KEY_CLIENT_RANKS} from config")
                return make_reply(ReturnCode.BAD_TASK_DATA)

            if not isinstance(ranks, dict):
                self.log_error(fl_ctx, f"expect config data to be dict but got {ranks}")
                return make_reply(ReturnCode.BAD_TASK_DATA)

            me = fl_ctx.get_identity_name()
            my_rank = ranks.get(me)
            if my_rank is None:
                self.log_error(fl_ctx, f"missing rank for me ({me}) in config data")
                return make_reply(ReturnCode.BAD_TASK_DATA)

            self.log_info(fl_ctx, f"got my rank: {my_rank}")

            num_rounds = shareable.get(Constant.CONF_KEY_NUM_ROUNDS)
            if not num_rounds:
                self.log_error(fl_ctx, f"missing {Constant.CONF_KEY_NUM_ROUNDS} from config")
                return make_reply(ReturnCode.BAD_TASK_DATA)

            assert isinstance(self.bridge, XGBClientBridge)
            self.bridge.configure(
                {
                    Constant.CONF_KEY_RANK: my_rank,
                    Constant.CONF_KEY_NUM_ROUNDS: num_rounds,
                 },
                fl_ctx)
            return make_reply(ReturnCode.OK)
        elif task_name == self.start_task_name:
            # start bridge
            try:
                self.bridge.start(fl_ctx)
            except Exception as ex:
                self.log_exception(fl_ctx, f"failed to start bridge: {secure_format_exception(ex)}")
                return make_reply(ReturnCode.EXECUTION_EXCEPTION)

            # start bridge monitor
            self.bridge.monitor_target(fl_ctx, self._notify_client_done)
            return make_reply(ReturnCode.OK)
        else:
            self.log_error(fl_ctx, f"ignored unsupported {task_name}")
            return make_reply(ReturnCode.TASK_UNSUPPORTED)

    def _notify_client_done(self, rc, fl_ctx: FLContext):
        if rc != 0:
            self.log_error(fl_ctx, f"XGB Client stopped with RC {rc}")
        else:
            self.log_info(fl_ctx, "XGB Client Stopped")

        # tell server that XGB is done
        engine = fl_ctx.get_engine()
        req = Shareable()
        req[Constant.CONF_KEY_EXIT_CODE] = rc
        engine.send_aux_request(
            targets=[FQCN.ROOT_SERVER],
            topic=Constant.TOPIC_CLIENT_DONE,
            request=req,
            timeout=0,   # fire and forget
            fl_ctx=fl_ctx,
            optional=True,
        )
