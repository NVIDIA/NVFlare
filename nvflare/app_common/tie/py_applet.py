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

from nvflare.apis.workspace import Workspace
from nvflare.fuel.utils.log_utils import add_log_file_handler, configure_logging
from nvflare.security.logging import secure_format_exception, secure_log_traceback

from .applet import Applet
from .defs import Constant


class PyRunner(ABC):

    """
    A PyApplet must return a light-weight PyRunner object to run the Python code of the external app.
    Since the runner could be running in a separate subprocess, the runner object must be pickleable!
    """

    @abstractmethod
    def start(self, app_ctx: dict):
        """Start the external app's Python code

        Args:
            app_ctx: the app's execution context

        Returns:

        """
        pass

    @abstractmethod
    def stop(self, timeout: float):
        """Stop the external app's python code

        Args:
            timeout: how long to wait for the app to stop before killing it

        Returns: None

        """
        pass

    @abstractmethod
    def is_stopped(self) -> (bool, int):
        """Check whether the app code is stopped

        Returns: a tuple of: whether the app is stopped, and exit code if stopped

        """
        pass


class _PyStarter:
    """This class is used to start the Python code of the applet. It is used when running the applet in a thread
    or in a separate process.
    """

    def __init__(self, runner: PyRunner, in_process: bool, workspace: Workspace, job_id: str):
        self.runner = runner
        self.in_process = in_process
        self.workspace = workspace
        self.job_id = job_id
        self.error = None
        self.started = True
        self.stopped = False
        self.exit_code = 0

    def start(self, app_ctx: dict):
        """Start the applet and wait for it to finish.

        Args:
            app_ctx: the app's execution context

        Returns: None

        """
        try:
            if not self.in_process:
                # enable logging
                run_dir = self.workspace.get_run_dir(self.job_id)
                log_file_name = os.path.join(run_dir, "applet_log.txt")
                configure_logging(self.workspace)
                add_log_file_handler(log_file_name)
            self.runner.start(app_ctx)

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
        self.runner = None

    @abstractmethod
    def get_runner(self, app_ctx: dict) -> PyRunner:
        """Subclass must implement this method to return a PyRunner.
        The returned PyRunner must be pickleable since it could be run in a separate subprocess!

        Args:
            app_ctx: the app context for the runner

        Returns: a PyRunner object

        """
        pass

    def start(self, app_ctx: dict):
        """Start the execution of the applet.

        Args:
            app_ctx: the app context

        Returns:

        """
        fl_ctx = app_ctx.get(Constant.APP_CTX_FL_CONTEXT)
        engine = fl_ctx.get_engine()
        workspace = engine.get_workspace()
        job_id = fl_ctx.get_job_id()
        runner = self.get_runner(app_ctx)

        if not isinstance(runner, PyRunner):
            raise RuntimeError(f"runner must be a PyRunner but got {type(runner)}")

        self.runner = runner
        self.starter = _PyStarter(runner, self.in_process, workspace, job_id)
        if self.in_process:
            self._start_in_thread(self.starter, app_ctx)
        else:
            self._start_in_process(self.starter, app_ctx)

    def _start_in_thread(self, starter, app_ctx: dict):
        """Start the applet in a separate thread."""
        self.logger.info("Starting applet in another thread")
        thread = threading.Thread(target=starter.start, args=(app_ctx,), daemon=True, name="applet")
        thread.start()
        if not self.starter.started:
            self.logger.error(f"Cannot start applet: {self.starter.error}")
            raise RuntimeError(self.starter.error)

    def _start_in_process(self, starter, app_ctx: dict):
        """Start the applet in a separate process."""
        # must remove Constant.APP_CTX_FL_CONTEXT from ctx because it's not pickleable!
        app_ctx.pop(Constant.APP_CTX_FL_CONTEXT, None)
        self.logger.info("Starting applet in another process")
        self.process = multiprocessing.Process(target=starter.start, args=(app_ctx,), daemon=True, name="applet")
        self.process.start()

    def stop(self, timeout=0.0) -> int:
        """Stop the applet

        Args:
            timeout: amount of time to wait for the applet to stop by itself. If the applet does not stop on
                its own within this time, we'll forcefully stop it by kill.

        Returns: None

        """
        if not self.runner:
            raise RuntimeError("PyRunner is not set")

        if self.in_process:
            self.runner.stop(timeout)
            return 0
        else:
            p = self.process
            self.process = None
            if p:
                assert isinstance(p, multiprocessing.Process)
                if p.exitcode is None:
                    # the process is still running
                    if timeout > 0:
                        # wait for the applet to stop by itself
                        start = time.time()
                        while time.time() - start < timeout:
                            if p.exitcode is not None:
                                # already stopped
                                self.logger.info(f"applet stopped (rc={p.exitcode}) after {time.time()-start} secs")
                                return p.exitcode
                            time.sleep(0.1)
                    self.logger.info("stopped applet by killing the process")
                    p.kill()
                    return -9

    def is_stopped(self) -> (bool, int):
        if not self.runner:
            raise RuntimeError("PyRunner is not set")

        if self.in_process:
            if self.starter:
                if self.starter.stopped:
                    self.logger.info("starter is stopped!")
                    return True, self.starter.exit_code
            return self.runner.is_stopped()
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
