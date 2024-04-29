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
import builtins
import logging
import os
import sys
import traceback

print_fn = builtins.print


class TaskScriptRunner:
    logger = logging.getLogger(__name__)

    def __init__(self, script_path: str, script_args: str = None):
        """Wrapper for function given function path and args

        Args:
            script_path (str): script file name, such as train.py
            script_args (str, Optional): script arguments to pass in.
        """
        self.script_args = script_args
        self.client_api = None
        self.logger = logging.getLogger(self.__class__.__name__)
        self.script_path = self.get_script_full_path(script_path)

    def run(self):
        """Call the task_fn with any required arguments."""
        self.logger.info(f"\n start task run() with {self.script_path}")
        try:
            import runpy

            curr_argv = sys.argv
            builtins.print = log_print
            sys.argv = self.get_sys_argv()
            runpy.run_path(self.script_path, run_name="__main__")
            sys.argv = curr_argv

        except Exception as e:
            msg = traceback.format_exc()
            self.logger.error(msg)
            if self.client_api:
                self.client_api.exec_queue.ask_abort(msg)
            raise e
        finally:
            builtins.print = print_fn

    def get_sys_argv(self):
        args_list = [] if not self.script_args else self.script_args.split()
        return [self.script_path] + args_list

    def get_script_full_path(self, script_path) -> str:
        target_files = None
        for r, dirs, files in os.walk(os.getcwd()):
            target_files = [os.path.join(r, f) for f in files if f == script_path]
            if target_files:
                break
        if not target_files:
            raise ValueError(f"{script_path} is not found")
        return target_files[0]


def log_print(*args, logger=TaskScriptRunner.logger, **kwargs):
    # Create a message from print arguments
    message = " ".join(str(arg) for arg in args)
    logger.info(message)
