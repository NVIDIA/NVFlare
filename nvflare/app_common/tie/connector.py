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

import multiprocessing
import os
import sys
import threading
import time
from abc import ABC, abstractmethod
from typing import Optional

from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal
from nvflare.apis.utils.reliable_message import ReliableMessage
from nvflare.apis.workspace import Workspace
from nvflare.app_common.tie.applet import Applet
from nvflare.app_common.tie.defs import VALID_APPLET_ENV, Constant
from nvflare.fuel.f3.cellnet.fqcn import FQCN
from nvflare.fuel.utils.log_utils import add_log_file_handler, configure_logging
from nvflare.fuel.utils.validation_utils import check_object_type
from nvflare.security.logging import secure_format_exception, secure_log_traceback


class _AppletStarter:
    """This class is used to start applet. It is used when running the applet in a thread
    or in a separate process.
    """

    def __init__(self, applet: Applet, in_process: bool, workspace: Workspace, job_id: str):
        self.applet = applet
        self.in_process = in_process
        self.workspace = workspace
        self.job_id = job_id
        self.error = None
        self.started = True
        self.stopped = False
        self.exit_code = 0

    def start(self, ctx: dict):
        """Start the applet and wait for it to finish.

        Args:
            ctx:

        Returns:

        """
        try:
            if not self.in_process:
                # enable logging
                run_dir = self.workspace.get_run_dir(self.job_id)
                log_file_name = os.path.join(run_dir, "applet_log.txt")
                configure_logging(self.workspace)
                add_log_file_handler(log_file_name)
            self.applet.start(ctx)
            self.stopped = True
        except Exception as e:
            secure_log_traceback()
            self.error = f"Exception starting applet: {secure_format_exception(e)}"
            self.started = False
            self.exit_code = Constant.EXIT_CODE_CANT_START
            self.stopped = True
            if not self.in_process:
                # this is a separate process
                sys.exit(self.exit_code)


class Connector(ABC, FLComponent):
    """
    Connectors are used to integrate FLARE with an Applet (Server or Client) in run time.
    Each type of applet requires an appropriate connector to integrate it with FLARE's Controller or Executor.
    The Connector class defines commonly required methods for all Connector implementations.
    """

    def __init__(self, applet_env: str):
        """Constructor of Connector

        Args:
            applet_env: applet's running env
        """
        FLComponent.__init__(self)
        self.abort_signal = None
        self.applet = None
        self.applet_env = applet_env
        self.starter = None
        self.process = None
        self.engine = None

        if applet_env not in VALID_APPLET_ENV:
            raise ValueError(f"invalid applet_env {applet_env}: must be in {VALID_APPLET_ENV}")

        self.in_process = self.applet_env == Constant.APPLET_ENV_THREAD

    def set_applet(self, applet: Applet):
        """Set the applet that will be used to run app processing logic.
        Note that the connector is only responsible for starting the applet appropriately (in a thread or in a
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
        If self.in_process is True, then the applet will be started in a separate thread.
        If self.in_process is False, then the applet will be started in a separate process.

        Args:
            app_ctx: the contextual info for running the applet
            fl_ctx: FL context

        Returns: None

        """
        if not self.applet:
            raise RuntimeError("applet has not been set!")

        engine = fl_ctx.get_engine()
        workspace = engine.get_workspace()
        job_id = fl_ctx.get_job_id()

        if self.applet_env == Constant.APPLET_ENV_SELF:
            self.logger.info("starting applet by itself")
            self.applet.start(app_ctx)
            return

        starter = _AppletStarter(self.applet, self.in_process, workspace, job_id)
        if self.in_process:
            self.logger.info("starting applet in another thread")
            t = threading.Thread(
                target=starter.start,
                args=(app_ctx,),
                daemon=True,
                name="applet",
            )
            t.start()
            if not starter.started:
                self.logger.error(f"cannot start applet: {starter.error}")
                raise RuntimeError(starter.error)
            self.starter = starter
        else:
            # start as a separate local process
            self.logger.info("starting applet in another process")
            self.process = multiprocessing.Process(
                target=starter.start,
                args=(app_ctx,),
                daemon=True,
                name="applet",
            )
            self.process.start()

    def stop_applet(self, timeout=0.0):
        """Stop the running of the applet

        Returns: None

        """
        if self.applet_env == Constant.APPLET_ENV_SELF:
            self.applet.stop(timeout)
            return

        if self.in_process:
            applet = self.applet
            self.applet = None
            if applet:
                applet.stop(timeout)
        else:
            p = self.process
            self.process = None
            if p:
                p.kill()

    def is_applet_stopped(self) -> (bool, int):
        """Check whether the applet is already stopped

        Returns: a tuple of (whether the applet is stopped, exit code)

        """
        if self.applet_env == Constant.APPLET_ENV_SELF:
            if self.applet:
                return self.applet.is_stopped()
            else:
                self.logger.warning("applet is not set with the connector")
                return True, 0

        if self.in_process:
            if self.starter:
                if self.starter.stopped:
                    self.logger.info("starter is stopped!")
                    return True, self.starter.exit_code

            if self.applet:
                return self.applet.is_stopped()
            else:
                self.logger.warning("applet is not set with the connector")
                return True, 0
        else:
            if self.process:
                assert isinstance(self.process, multiprocessing.Process)
                ec = self.process.exitcode
                if ec is None:
                    return False, 0
                else:
                    return True, ec
            else:
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

        self.logger.debug(f"sending request with RM: {request=}; {type(request['flower.headers'])}")
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
