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
import logging
import multiprocessing
import os
import sys
import threading
import time
from abc import ABC, abstractmethod
from typing import Tuple

from xgboost.core import XGBoostError

from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_context import FLContext
from nvflare.apis.signal import Signal
from nvflare.apis.workspace import Workspace
from nvflare.app_opt.xgboost.histogram_based_v2.defs import Constant
from nvflare.app_opt.xgboost.histogram_based_v2.runners.xgb_runner import AppRunner
from nvflare.fuel.utils.log_utils import add_log_file_handler, configure_logging
from nvflare.fuel.utils.validation_utils import check_object_type
from nvflare.security.logging import secure_format_exception, secure_log_traceback


class _RunnerStarter:
    """This small class is used to start XGB client runner. It is used when running the runner in a thread
    or in a separate process.

    """

    def __init__(self, app_name: str, runner, in_process: bool, workspace: Workspace, job_id: str):
        self.app_name = app_name
        self.runner = runner
        self.in_process = in_process
        self.workspace = workspace
        self.job_id = job_id
        self.error = None
        self.started = True
        self.stopped = False
        self.exit_code = 0
        self.logger = logging.getLogger(self.__class__.__name__)

    def start(self, ctx: dict):
        """Start the runner and wait for it to finish.

        Args:
            ctx:

        Returns:

        """
        try:
            if not self.in_process:
                # enable logging
                run_dir = self.workspace.get_run_dir(self.job_id)
                log_file_name = os.path.join(run_dir, f"{self.app_name}_log.txt")
                print(f"XGB Log: {log_file_name}")
                configure_logging(self.workspace)
                add_log_file_handler(log_file_name)
            self.runner.run(ctx)
            self.stopped = True
        except Exception as e:
            self.error = f"Exception starting {self.app_name} runner: {secure_format_exception(e)}"
            self.logger.error(self.error)
            # XGBoost already prints a traceback
            if not isinstance(e, XGBoostError):
                secure_log_traceback()
            self.started = False
            self.exit_code = Constant.EXIT_CODE_CANT_START
            self.stopped = True
            if not self.in_process:
                # this is a separate process
                sys.exit(self.exit_code)


class AppAdaptor(ABC, FLComponent):
    """AppAdaptors are used to integrate FLARE with App Target (Server or Client) in run time."""

    def __init__(self, app_name: str, in_process: bool):
        """Constructor of AppAdaptor.

        Args:
            app_name (str): The name of the application.
            in_process (bool): Whether to call the `AppRunner.run()` in the same process or not.
        """
        FLComponent.__init__(self)
        self.abort_signal = None
        self.app_runner = None
        self.app_name = app_name
        self.in_process = in_process
        self.starter = None
        self.process = None

    def set_runner(self, runner: AppRunner):
        """Set the App Runner that will be used to run app processing logic.
        Note that the adaptor is only responsible for starting the runner appropriately (in a thread or in a
        separate process).

        Args:
            runner (AppRunner): the runner to be set

        Returns: None

        """
        if not isinstance(runner, AppRunner):
            raise TypeError(f"runner must be AppRunner but got {type(runner)}")
        self.app_runner = runner

    def set_abort_signal(self, abort_signal: Signal):
        """Called by XGB Controller/Executor to set the abort_signal.

        The abort_signal is assigned by FLARE's XGB Controller/Executor. It is used by the Controller/Executor
        to tell the adaptor that the job has been aborted.

        Args:
            abort_signal: the abort signal assigned by the caller.

        Returns: None

        """
        check_object_type("abort_signal", abort_signal, Signal)
        self.abort_signal = abort_signal

    def initialize(self, fl_ctx: FLContext):
        """Called by the Controller/Executor to initialize the adaptor.

        Args:
            fl_ctx: the FL context

        Returns: None

        """
        pass

    @abstractmethod
    def start(self, fl_ctx: FLContext):
        """Called by XGB Controller/Executor to start the target.
        If any error occurs when starting the target, this method should raise an exception.

        Args:
            fl_ctx: the FL context.

        Returns: None

        """
        pass

    @abstractmethod
    def stop(self, fl_ctx: FLContext):
        """Called by XGB Controller/Executor to stop the target.
        If any error occurs when stopping the target, this method should raise an exception.

        Args:
            fl_ctx: the FL context.

        Returns: None

        """
        pass

    @abstractmethod
    def configure(self, config: dict, fl_ctx: FLContext):
        """Called by XGB Controller/Executor to configure the adaptor.
        If any error occurs, this method should raise an exception.

        Args:
            config: config data
            fl_ctx: the FL context

        Returns: None

        """
        pass

    @abstractmethod
    def _is_stopped(self) -> Tuple[bool, int]:
        """Called by the adaptor's monitor to know whether the target is stopped.
        Note that this method is not called by XGB Controller/Executor.

        Returns: a tuple of: whether the target is stopped, and return code (if stopped)

        Note that a non-zero return code is considered abnormal completion of the target.

        """
        pass

    def _monitor(self, fl_ctx: FLContext, target_stopped_cb):
        while True:
            if self.abort_signal.triggered:
                # asked to abort
                self.stop(fl_ctx)
                return

            stopped, rc = self._is_stopped()
            if stopped:
                # target already stopped - notify the caller
                target_stopped_cb(rc, fl_ctx)
                return

            time.sleep(0.1)

    def monitor_target(self, fl_ctx: FLContext, target_stopped_cb):
        """Called by XGB Controller/Executor to monitor the health of the target.

        The monitor periodically checks the abort signal. Once set, it calls the adaptor's stop() method
        to stop the running of the target.

        The monitor also periodically checks whether the target is already stopped (by calling the is_stopped
        method). If the target is stopped, the monitor will call the specified target_stopped_cb.

        Args:
            fl_ctx: FL context
            target_stopped_cb: the callback function to be called when the target is stopped.

        Returns: None

        """
        if not callable(target_stopped_cb):
            raise RuntimeError(f"target_stopped_cb must be callable but got {type(target_stopped_cb)}")

        # start the monitor in a separate daemon thread!
        t = threading.Thread(target=self._monitor, args=(fl_ctx, target_stopped_cb), daemon=True)
        t.start()

    def start_runner(self, run_ctx: dict, fl_ctx: FLContext):
        engine = fl_ctx.get_engine()
        workspace = engine.get_workspace()
        job_id = fl_ctx.get_job_id()
        starter = _RunnerStarter(self.app_name, self.app_runner, self.in_process, workspace, job_id)
        if self.in_process:
            self.logger.info(f"starting {self.app_name} Server in another thread")
            t = threading.Thread(
                target=starter.start,
                args=(run_ctx,),
                daemon=True,
                name=f"{self.app_name}_server_thread_runner",
            )
            t.start()
            if not starter.started:
                self.logger.error(f"cannot start {self.app_name} server: {starter.error}")
                raise RuntimeError(starter.error)
            self.starter = starter
        else:
            # start as a separate local process
            self.logger.info(f"starting {self.app_name} server in another process")
            self.process = multiprocessing.Process(
                target=starter.start,
                args=(run_ctx,),
                daemon=True,
                name=f"{self.app_name}_server_process_runner",
            )
            self.process.start()

    def stop_runner(self):
        if self.in_process:
            runner = self.app_runner
            self.app_runner = None
            if runner:
                runner.stop()
        else:
            p = self.process
            self.process = None
            if p:
                p.kill()

    def is_runner_stopped(self) -> Tuple[bool, int]:
        if self.in_process:
            if self.starter:
                if self.starter.stopped:
                    return True, self.starter.exit_code

            if self.app_runner:
                return self.app_runner.is_stopped()
            else:
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
