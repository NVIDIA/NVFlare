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

import os
import shlex
import signal
import subprocess
import threading

from nvflare.fuel.utils.obj_utils import get_logger


class ProcessManager:
    """A ProcessManager is used to start a process on the same host. It also provides functions to check and stop
    the process.

    """

    def __init__(self, name: str, start_cmd: str):
        """Constructor

        Args:
            name: a logical name of the process
            start_cmd: the command for starting the process
        """
        self.name = name
        self.start_cmd = start_cmd
        self.process = None
        self.logger = get_logger(self)
        self.lock = threading.Lock()

    def start(self):
        """Start the process

        Returns:

        """
        with self.lock:
            if self.process:
                return
            self.logger.info(f"starting process {self.name}: {self.start_cmd}")
            self.process = subprocess.Popen(shlex.split(self.start_cmd), preexec_fn=os.setsid, env=os.environ.copy())
            self.logger.info(f"started process {self.name}: {self.start_cmd}")

    def is_stopped(self) -> (bool, int):
        """Check whether the process is already stopped

        Returns: a tuple of: whether the process is stopped, and the exit code if stopped.
        Exit code 0 means normal exit.

        """
        with self.lock:
            if not self.process:
                return True, 0

            rc = self.process.poll()
            if rc is None:
                # still running
                return False, 0
            else:
                # stopped
                self.process = None
                self.logger.info(f"process {self.name} is stopped with RC {rc}")
                return True, rc

    def stop(self):
        """Stop the process

        Returns: None

        """
        with self.lock:
            if self.process:
                # kill the process
                self.logger.info(f"stopping process {self.name}")
                try:
                    os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                except:
                    pass
                self.process = None
