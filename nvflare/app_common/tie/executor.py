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
from abc import abstractmethod

from nvflare.apis.event_type import EventType
from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.tie.applet import Applet
from nvflare.app_common.tie.connector import Connector
from nvflare.fuel.f3.cellnet.fqcn import FQCN
from nvflare.security.logging import secure_format_exception

from .defs import Constant


class TieExecutor(Executor):
    def __init__(
        self,
        configure_task_name=Constant.CONFIG_TASK_NAME,
        start_task_name=Constant.START_TASK_NAME,
    ):
        """Constructor

        Args:
            configure_task_name: name of the config task
            start_task_name: name of the start task
        """
        Executor.__init__(self)
        self.configure_task_name = configure_task_name
        self.start_task_name = start_task_name
        self.connector = None
        self.engine = None

        # create the abort signal to be used for signaling the connector
        self.abort_signal = Signal()

    @abstractmethod
    def get_connector(self, fl_ctx: FLContext) -> Connector:
        """Called by the TieExecutor to get the Connector to be used by this executor.
        A subclass of TieExecutor must implement this method.

        Args:
            fl_ctx: the FL context

        Returns: a Connector object

        """
        pass

    @abstractmethod
    def get_applet(self, fl_ctx: FLContext) -> Applet:
        """Called by the TieExecutor to get the Applet to be used by this executor.
        A subclass of TieExecutor must implement this method.

        Args:
            fl_ctx: the FL context

        Returns: an Applet object

        """
        pass

    def configure(self, config: dict, fl_ctx: FLContext):
        """Called by the TieExecutor to configure the executor based on the config params received from the server.
        A subclass of TieExecutor should implement this method.

        Args:
            config: the config data
            fl_ctx: FL context

        Returns: None

        """
        pass

    def get_connector_config(self, fl_ctx: FLContext) -> dict:
        """Called by the TieExecutor to get config params for the connector.
        A subclass of TieExecutor should implement this method.
        Note that this method is always called after the "configure" method, hence it's possible to dynamically
        determine the connector's config based on the config params in the "configure" step.

        Args:
            fl_ctx: the FL context

        Returns: a dict of config params

        """
        return {}

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.START_RUN:
            self.engine = fl_ctx.get_engine()
            connector = self.get_connector(fl_ctx)
            if not connector:
                self.system_panic("cannot get connector", fl_ctx)
                return

            if not isinstance(connector, Connector):
                self.system_panic(
                    f"invalid connector: expect Connector but got {type(connector)}",
                    fl_ctx,
                )
                return

            applet = self.get_applet(fl_ctx)
            if not applet:
                self.system_panic("cannot get applet", fl_ctx)
                return

            if not isinstance(applet, Applet):
                self.system_panic(
                    f"invalid applet: expect Applet but got {type(applet)}",
                    fl_ctx,
                )
                return

            applet.initialize(fl_ctx)
            connector.set_abort_signal(self.abort_signal)
            connector.set_applet(applet)
            connector.initialize(fl_ctx)
            self.connector = connector
        elif event_type == EventType.FATAL_SYSTEM_ERROR:
            # notify server that the client is done
            self._notify_client_done(Constant.EXIT_CODE_FATAL_ERROR, fl_ctx)
        elif event_type == EventType.END_RUN:
            self.abort_signal.trigger(True)

    def execute(self, task_name: str, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        if task_name == self.configure_task_name:
            config = shareable.get(Constant.MSG_KEY_CONFIG)
            if config is None:
                self.log_error(fl_ctx, f"missing {Constant.MSG_KEY_CONFIG} from config")
                return make_reply(ReturnCode.BAD_TASK_DATA)

            self.configure(config, fl_ctx)

            # configure the connector
            connector_config = self.get_connector_config(fl_ctx)
            self.connector.configure(connector_config, fl_ctx)
            return make_reply(ReturnCode.OK)
        elif task_name == self.start_task_name:
            # start the connector
            try:
                self.connector.start(fl_ctx)
            except Exception as ex:
                self.log_exception(fl_ctx, f"failed to start connector: {secure_format_exception(ex)}")
                return make_reply(ReturnCode.EXECUTION_EXCEPTION)

            # start to monitor the connector and applet
            self.connector.monitor(fl_ctx, self._notify_client_done)
            return make_reply(ReturnCode.OK)
        else:
            self.log_error(fl_ctx, f"ignored unsupported {task_name}")
            return make_reply(ReturnCode.TASK_UNSUPPORTED)

    def _notify_client_done(self, rc, fl_ctx: FLContext):
        """This is called when app is done.
        We send a message to the FL server telling it that this client is done.

        Args:
            rc: the return/exit code
            fl_ctx: FL context

        Returns: None

        """
        if rc != 0:
            self.log_error(fl_ctx, f"App stopped with RC {rc}")
        else:
            self.log_info(fl_ctx, "App Stopped")

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
