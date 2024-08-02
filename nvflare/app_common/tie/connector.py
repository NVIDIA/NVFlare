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
from typing import Optional

from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal
from nvflare.apis.utils.reliable_message import ReliableMessage
from nvflare.app_common.tie.applet import Applet
from nvflare.app_common.tie.defs import Constant
from nvflare.fuel.f3.cellnet.fqcn import FQCN
from nvflare.fuel.utils.validation_utils import check_object_type


class Connector(ABC, FLComponent):
    """
    Connectors are used to integrate FLARE with an Applet (Server or Client) in run time.
    Each type of applet requires an appropriate connector to integrate it with FLARE's Controller or Executor.
    The Connector class defines commonly required methods for all Connector implementations.
    """

    def __init__(self):
        """Constructor of Connector"""
        FLComponent.__init__(self)
        self.abort_signal = None
        self.applet = None
        self.engine = None

    def set_applet(self, applet: Applet):
        """Set the applet that will be used to run app processing logic.
        Note that the connector is only responsible for starting the applet appropriately (in a separate thread or in a
        separate process).

        Args:
            applet: the applet to be set

        Returns: None

        """
        if not isinstance(applet, Applet):
            raise TypeError(f"applet must be Applet but got {type(applet)}")
        self.applet = applet

    def set_abort_signal(self, abort_signal: Signal):
        """Called by Controller/Executor to set the abort_signal.

        The abort_signal is assigned by FLARE Controller/Executor. It is used by the Controller/Executor
        to tell the connector that the job has been aborted.

        Args:
            abort_signal: the abort signal assigned by the caller.

        Returns: None

        """
        check_object_type("abort_signal", abort_signal, Signal)
        self.abort_signal = abort_signal

    def initialize(self, fl_ctx: FLContext):
        """Called by the Controller/Executor to initialize the connector.

        Args:
            fl_ctx: the FL context

        Returns: None

        """
        self.engine = fl_ctx.get_engine()

    @abstractmethod
    def start(self, fl_ctx: FLContext):
        """Called by Controller/Executor to start the connector.
        If any error occurs, this method should raise an exception.

        Args:
            fl_ctx: the FL context.

        Returns: None

        """
        pass

    @abstractmethod
    def stop(self, fl_ctx: FLContext):
        """Called by Controller/Executor to stop the connector.
        If any error occurs, this method should raise an exception.

        Args:
            fl_ctx: the FL context.

        Returns: None

        """
        pass

    @abstractmethod
    def configure(self, config: dict, fl_ctx: FLContext):
        """Called by Controller/Executor to configure the connector.
        If any error occurs, this method should raise an exception.

        Args:
            config: config data
            fl_ctx: the FL context

        Returns: None

        """
        pass

    def _is_stopped(self) -> (bool, int):
        """Called by the connector's monitor to know whether the connector is stopped.
        Note that this method is not called by Controller/Executor.

        Returns: a tuple of: whether the connector is stopped, and return code (if stopped)

        Note that a non-zero return code is considered abnormal completion of the connector.

        """
        return self.is_applet_stopped()

    def _monitor(self, fl_ctx: FLContext, connector_stopped_cb):
        while True:
            if self.abort_signal.triggered:
                # asked to abort
                self.stop(fl_ctx)
                return

            stopped, rc = self._is_stopped()
            if stopped:
                # connector already stopped - notify the caller
                connector_stopped_cb(rc, fl_ctx)
                return

            time.sleep(0.1)

    def monitor(self, fl_ctx: FLContext, connector_stopped_cb):
        """Called by Controller/Executor to monitor the health of the connector.

        The monitor periodically checks the abort signal. Once set, it calls the connector's stop() method
        to stop the running of the app.

        The monitor also periodically checks whether the connector is already stopped (by calling the is_stopped
        method). If the connector is stopped, the monitor will call the specified connector_stopped_cb.

        Args:
            fl_ctx: FL context
            connector_stopped_cb: the callback function to be called when the connector is stopped.

        Returns: None

        """
        if not callable(connector_stopped_cb):
            raise RuntimeError(f"connector_stopped_cb must be callable but got {type(connector_stopped_cb)}")

        # start the monitor in a separate daemon thread!
        t = threading.Thread(target=self._monitor, args=(fl_ctx, connector_stopped_cb), daemon=True)
        t.start()

    def start_applet(self, app_ctx: dict, fl_ctx: FLContext):
        """Start the applet set to the connector.

        Args:
            app_ctx: the contextual info for running the applet
            fl_ctx: FL context

        Returns: None

        """
        if not self.applet:
            raise RuntimeError("applet has not been set!")

        app_ctx[Constant.APP_CTX_FL_CONTEXT] = fl_ctx
        self.applet.start(app_ctx)

    def stop_applet(self, timeout=0.0) -> int:
        """Stop the running of the applet

        Returns: exit code of the applet

        """
        return self.applet.stop(timeout)

    def is_applet_stopped(self) -> (bool, int):
        """Check whether the applet is already stopped

        Returns: a tuple of (whether the applet is stopped, exit code)

        """
        applet = self.applet
        if applet:
            return applet.is_stopped()
        else:
            self.logger.warning("applet is not set with the connector")
            return True, 0

    def send_request(
        self,
        target: Optional[str],
        op: str,
        request: Shareable,
        per_msg_timeout: float,
        tx_timeout: float,
        fl_ctx: Optional[FLContext],
    ) -> Shareable:
        """Send app request to the specified target via FLARE ReliableMessage.

        Args:
            target: the destination of the request. If not specified, default to server.
            op: the operation
            request: operation data
            per_msg_timeout: per-message timeout
            tx_timeout: transaction timeout
            fl_ctx: FL context. If not provided, this method will create a new FL context.

        Returns:
            operation result
        """
        request.set_header(Constant.MSG_KEY_OP, op)
        if not target:
            target = FQCN.ROOT_SERVER

        if not fl_ctx:
            fl_ctx = self.engine.new_context()

        self.logger.debug(f"sending request with RM: {op=}")
        return ReliableMessage.send_request(
            target=target,
            topic=Constant.TOPIC_APP_REQUEST,
            request=request,
            per_msg_timeout=per_msg_timeout,
            tx_timeout=tx_timeout,
            abort_signal=self.abort_signal,
            fl_ctx=fl_ctx,
        )

    def process_app_request(self, op: str, req: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        """Called by Controller/Executor to process a request from an applet on another site.

        Args:
            op: the op code of the request
            req: the request to be sent
            fl_ctx: FL context
            abort_signal: abort signal that could be triggered during the request processing

        Returns: processing result as Shareable object

        """
        pass
