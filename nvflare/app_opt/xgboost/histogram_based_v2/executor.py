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

from nvflare.apis.event_type import EventType
from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import FLContextKey, ReservedKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_opt.xgboost.histogram_based_v2.adaptors.xgb_adaptor import XGBClientAdaptor
from nvflare.fuel.f3.cellnet.fqcn import FQCN
from nvflare.security.logging import secure_format_exception

from .defs import Constant


class XGBExecutor(Executor):
    def __init__(
        self,
        adaptor_component_id: str,
        configure_task_name=Constant.CONFIG_TASK_NAME,
        start_task_name=Constant.START_TASK_NAME,
        per_msg_timeout=10.0,
        tx_timeout=100.0,
    ):
        """Constructor

        Args:
            adaptor_component_id: the component ID of client target adaptor
            configure_task_name: name of the config task
            start_task_name: name of the start task
            per_msg_timeout: timeout for sending one message
            tx_timeout: transaction timeout
        """
        Executor.__init__(self)
        self.adaptor_component_id = adaptor_component_id
        self.per_msg_timeout = per_msg_timeout
        self.tx_timeout = tx_timeout
        self.configure_task_name = configure_task_name
        self.start_task_name = start_task_name
        self.adaptor = None

        # create the abort signal to be used for signaling the adaptor
        self.abort_signal = Signal()

    def get_adaptor(self, fl_ctx: FLContext):
        """Get adaptor to be used by this executor.
        This is the default implementation that gets the adaptor based on configured adaptor_component_id.
        A subclass of XGBExecutor may get adaptor in a different way.

        Args:
            fl_ctx: the FL context

        Returns: a XGBClientAdaptor object

        """
        engine = fl_ctx.get_engine()
        return engine.get_component(self.adaptor_component_id)

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.START_RUN:
            adaptor = self.get_adaptor(fl_ctx)
            if not adaptor:
                self.system_panic(f"cannot get component for {self.adaptor_component_id}", fl_ctx)
                return

            if not isinstance(adaptor, XGBClientAdaptor):
                self.system_panic(
                    f"invalid component '{self.adaptor_component_id}': expect XGBClientAdaptor but got {type(adaptor)}",
                    fl_ctx,
                )
                return

            adaptor.set_abort_signal(self.abort_signal)
            adaptor.initialize(fl_ctx)
            self.adaptor = adaptor
        elif event_type == Constant.EVENT_XGB_ABORTED:
            self._notify_client_done(Constant.EXIT_CODE_JOB_ABORT, fl_ctx)
        elif event_type == EventType.END_RUN:
            self.abort_signal.trigger(True)

    def execute(self, task_name: str, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        if task_name == self.configure_task_name:
            # there are two important config params for the client:
            #   the rank assigned to the client;
            #   number of rounds for training.
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

            # configure the XGB client target via the adaptor
            self.adaptor.configure(
                shareable,
                fl_ctx,
            )
            self.fire_event(Constant.EVENT_XGB_JOB_CONFIGURED, fl_ctx)
            config_error = fl_ctx.get_prop(Constant.PARAM_KEY_CONFIG_ERROR, None)
            if not config_error:
                return make_reply(ReturnCode.OK)
            else:
                self.log_error(fl_ctx, f"Config error: {config_error}")
                return make_reply(ReturnCode.SERVICE_UNAVAILABLE, {ReservedKey.EXCEPTIONS: config_error})
        elif task_name == self.start_task_name:
            # start adaptor
            try:
                self.adaptor.start(fl_ctx)
            except Exception as ex:
                self.log_exception(fl_ctx, f"failed to start adaptor: {secure_format_exception(ex)}")
                return make_reply(ReturnCode.EXECUTION_EXCEPTION)

            # start to monitor the XGB target via the adaptor
            self.adaptor.monitor_target(fl_ctx, self._notify_client_done)
            return make_reply(ReturnCode.OK)
        else:
            self.log_error(fl_ctx, f"ignored unsupported {task_name}")
            return make_reply(ReturnCode.TASK_UNSUPPORTED)

    def _notify_client_done(self, rc, fl_ctx: FLContext):
        """This is called when the XGB client target is done.
        We send a message to the FL server telling it that this client is done.

        Args:
            rc: the return code from the XGB client target
            fl_ctx: FL context

        Returns: None

        """
        if rc != 0:
            self.log_error(fl_ctx, f"XGB Client stopped with RC {rc}")
            error = fl_ctx.get_prop(FLContextKey.FATAL_SYSTEM_ERROR)
            error_msg = f", error: {error}" if error else ""
            self.system_panic(f"XGB Client stopped with non zero RC {rc}{error_msg}", fl_ctx)
        else:
            self.log_info(fl_ctx, "XGB Client Stopped")

        # tell server that this client is done
        engine = fl_ctx.get_engine()
        req = Shareable()
        req[Constant.MSG_KEY_EXIT_CODE] = rc
        engine.send_aux_request(
            targets=[FQCN.ROOT_SERVER],
            topic=Constant.TOPIC_CLIENT_DONE,
            request=req,
            timeout=0,  # fire and forget
            fl_ctx=fl_ctx,
            optional=True,
        )
