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
import time
from abc import ABC, abstractmethod

from nvflare.security.logging import secure_format_exception

from .applet import Applet
from .defs import Constant
from .process_mgr import CommandDescriptor, start_process


class CLIApplet(Applet, ABC):
    def __init__(self):
        """Constructor of CLIApplet, which runs the applet as a subprocess started with CLI command."""
        Applet.__init__(self)
        self._proc_mgr = None
        self._start_error = False

    @abstractmethod
    def get_command(self, ctx: dict) -> CommandDescriptor:
        """Subclass must implement this method to return the CLI command to be executed.

        Args:
            ctx: the applet context that contains execution env info

        Returns: a CommandDescriptor to be executed

        """
        pass

    def start(self, ctx: dict):
        """Start the execution of the applet.

        Args:
            ctx:

        Returns:

        """
        cmd_desc = self.get_command(ctx)
        if not cmd_desc:
            raise RuntimeError("failed to get cli command from app context")

        fl_ctx = ctx.get(Constant.APP_CTX_FL_CONTEXT)
        try:
            self._proc_mgr = start_process(cmd_desc, fl_ctx)
        except Exception as ex:
            self.logger.error(f"exception starting applet '{cmd_desc.cmd}': {secure_format_exception(ex)}")
            self._start_error = True

    def stop(self, timeout=0.0):
        """Stop the applet

        Args:
            timeout: amount of time to wait for the applet to stop by itself. If the applet does not stop on
                its own within this time, we'll forcefully stop it by kill.

        Returns: None

        """
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
