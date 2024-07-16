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
import shlex
import subprocess


class ProcessManager:
    def __init__(self):
        self.process = None
        self.log_file = None

    def start(
        self,
        command: str,
        cwd=None,
        env=None,
        log_file=None,
    ):
        lf = log_file
        if log_file and isinstance(log_file, str):
            lf = open(log_file, "a")
            self.log_file = lf

        command_seq = shlex.split(command)
        self.process = subprocess.Popen(
            command_seq,
            stderr=subprocess.STDOUT,
            cwd=cwd,
            env=env,
            stdout=lf,
        )

    def poll(self):
        if not self.process:
            return 0
        return self.process.poll()

    def stop(self):
        rc = self.poll()
        if rc is None:
            # process is still alive
            try:
                self.process.kill()
            except Exception as ex:
                # ignore kill error
                pass

        # close the log file if any
        if self.log_file:
            self.log_file.close()
        return rc


def start_process(command: str, cwd=None, env=None, log_file=None) -> ProcessManager:
    mgr = ProcessManager()
    mgr.start(command, cwd, env, log_file)
    return mgr
