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
from abc import ABC, abstractmethod

from nvflare.apis.workspace import Workspace
from nvflare.fuel.utils.log_utils import add_log_file_handler, configure_logging
from nvflare.security.logging import secure_format_exception, secure_log_traceback

from .applet import Applet
from .defs import Constant


class _PyStarter:
    """This class is used to start the Python code of the applet. It is used when running the applet in a thread
    or in a separate process.
    """

    def __init__(self, in_process: bool, workspace: Workspace, job_id: str):
        self.in_process = in_process
        self.workspace = workspace
        self.job_id = job_id
        self.error = None
        self.started = True
        self.stopped = False
        self.exit_code = 0

    def start(self, run_func, ctx: dict):
        """Start the run_func and wait for it to finish.

        Args:
            ctx: run context

        Returns: None

        """
        try:
            if not self.in_process:
                # enable logging
                run_dir = self.workspace.get_run_dir(self.job_id)
                log_file_name = os.path.join(run_dir, "applet_log.txt")
                configure_logging(self.workspace)
                add_log_file_handler(log_file_name)
            run_func(ctx)

            # Note: run_func does not return until it runs to completion!
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


class PyApplet(Applet, ABC):
    def __init__(self, in_process: bool):
        """Constructor of PyApplet, which runs the applet's Python code in a separate thread or subprocess.

        Args:
            in_process: whether to run the applet code as separate thread within the same process or as a separate
                subprocess.
        """
        Applet.__init__(self)
        self.in_process = in_process
        self.starter = None
        self.process = None

    @abstractmethod
    def run_py(self, ctx: dict):
        """Subclass must implement this method to run the applet's python code.

        Args:
            ctx: the applet run context that contains execution env info

        Returns: None

        """
        pass

    def stop_py(self, timeout: float):
        """Stop the applet's python code, if possible.

        Returns:

        """
        pass

    @abstractmethod
    def is_py_stopped(self) -> (bool, int):
        """Check whether the applet's python code is halted already.
        Subclass must implement this method.

        Returns:

        """
        pass

    def start(self, ctx: dict):
        """Start the execution of the applet.

        Args:
            ctx: the applet run context

        Returns:

        """
        fl_ctx = ctx.get(Constant.APP_CTX_FL_CONTEXT)
        engine = fl_ctx.get_engine()
        workspace = engine.get_workspace()
        job_id = fl_ctx.get_job_id()

        starter = _PyStarter(self.in_process, workspace, job_id)
        if self.in_process:
            self.logger.info("starting applet in another thread")
            t = threading.Thread(
                target=starter.start,
                args=(
                    self.run_py,
                    ctx,
                ),
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
                args=(
                    self.run_py,
                    ctx,
                ),
                daemon=True,
                name="applet",
            )
            self.process.start()

    def stop(self, timeout=0.0):
        """Stop the applet

        Args:
            timeout: amount of time to wait for the applet to stop by itself. If the applet does not stop on
                its own within this time, we'll forcefully stop it by kill.

        Returns: None

        """
        if self.in_process:
            self.stop_py(timeout)
        else:
            p = self.process
            self.process = None
            if p:
                ec = p.exitcode
                if ec is None:
                    # the process is still running - kill it
                    p.kill()

    def is_stopped(self) -> (bool, int):
        if self.in_process:
            if self.starter:
                if self.starter.stopped:
                    self.logger.info("starter is stopped!")
                    return True, self.starter.exit_code

            return self.is_py_stopped()
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
