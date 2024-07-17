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
import os
import time
from abc import ABC, abstractmethod
from typing import Any

from nvflare.security.logging import secure_format_exception

from .applet import Applet
from .defs import Constant
from .process_mgr import start_process


class CLIApplet(Applet, ABC):
    def __init__(self):
        Applet.__init__(self)
        self._proc_mgr = None
        self._start_error = False

    @abstractmethod
    def get_command(self, ctx: dict) -> (str, str, dict, Any):
        """Subclass must implement this method to return the CLI command to be executed.

        Args:
            ctx: the applet context that contains execution env info

        Returns: a tuple of:
            command (str) - the CLI command to be executed
            current work dir - the current work dir for the command execution
            env - additional env vars to be added to system's env for the command execution
            log_file: the file for log messages. It can be a file object, full path to the file, or None.
                If none, no log file

        """
        pass

    def start(self, ctx: dict):
        cli_cmd, cli_cwd, cli_env, log_file = self.get_command(ctx)
        if not cli_cmd:
            raise RuntimeError("failed to get cli command from app context")

        env = os.environ.copy()
        if cli_env:
            if not isinstance(cli_env, dict):
                raise RuntimeError(f"expect cli env to be dict but got {type(cli_env)}")
            env.update(cli_env)

        try:
            self._proc_mgr = start_process(
                command=cli_cmd,
                cwd=cli_cwd,
                env=env,
                log_file=log_file,
            )
        except Exception as ex:
            self.logger.error(f"exception starting applet '{cli_cmd}': {secure_format_exception(ex)}")
            self._start_error = True

    def stop(self, timeout=0.0):
        mgr = self._proc_mgr
        self._proc_mgr = None

        if not mgr:
            return

        # wait for the applet to stop by itself
        start = time.time()
        while time.time() - start < timeout:
            rc = mgr.poll()
            if rc is not None:
                # already stopped
                self.logger.info(f"applet stopped ({rc=}) gracefully after {time.time()-start} seconds")
                break
            time.sleep(0.1)

        # have to kill the process after timeout
        rc = mgr.stop()
        if rc is None:
            self.logger.warn(f"killed the applet process after waiting {timeout} seconds")

    def is_stopped(self) -> (bool, int):
        if self._start_error:
            return True, Constant.EXIT_CODE_CANT_START

        if self._proc_mgr:
            return_code = self._proc_mgr.poll()
            if return_code is None:
                return False, 0
            else:
                return True, return_code
        else:
            return True, 0
