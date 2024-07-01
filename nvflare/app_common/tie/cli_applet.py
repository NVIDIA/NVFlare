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
import shlex
import subprocess
import time
from abc import ABC, abstractmethod

from nvflare.security.logging import secure_format_exception

from .applet import Applet
from .defs import Constant


class CLIApplet(Applet, ABC):
    def __init__(self):
        Applet.__init__(self)
        self._process = None
        self._start_error = False

    @abstractmethod
    def get_command(self, ctx: dict) -> (str, str, dict):
        """Subclass must implement this method to return the CLI command to be executed.

        Args:
            ctx: the applet context that contains execution env info

        Returns: a tuple of:
            command (str) - the CLI command to be executed
            current work dir - the current work dir for the command execution
            env - additional env vars to be added to system's env for the command execution

        """
        pass

    def start(self, ctx: dict):
        cli_cmd, cli_cwd, cli_env = self.get_command(ctx)
        if not cli_cmd:
            raise RuntimeError("failed to get cli command from app context")

        env = os.environ.copy()
        if cli_env:
            if not isinstance(cli_env, dict):
                raise RuntimeError(f"expect cli env to be dict but got {type(cli_env)}")
            env.update(cli_env)

        command_seq = shlex.split(cli_cmd)
        try:
            self._process = subprocess.Popen(
                command_seq,
                stderr=subprocess.STDOUT,
                cwd=cli_cwd,
                env=env,
            )
        except Exception as ex:
            self.logger.error(f"exception starting applet '{cli_cmd}': {secure_format_exception(ex)}")
            self._start_error = True

    def stop(self, timeout=0.0):
        p = self._process
        self._process = None

        if not p:
            return

        # wait for the applet to stop by itself
        start = time.time()
        while time.time() - start < timeout:
            rc = p.poll()
            if rc is not None:
                # already stopped
                self.logger.info(f"applet stopped ({rc=}) gracefully after {time.time()-start} seconds")
                return

            time.sleep(0.1)

        # have to kill the process after timeout
        self.logger.warn(f"killed the applet process after waiting {timeout} seconds")
        p.kill()

    def is_stopped(self) -> (bool, int):
        if self._start_error:
            return True, Constant.EXIT_CODE_CANT_START

        if self._process:
            return_code = self._process.poll()
            if return_code is None:
                return False, 0
            else:
                return True, return_code
        else:
            return True, 0
