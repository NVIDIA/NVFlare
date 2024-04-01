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
import logging
import sys
import traceback
from typing import Dict

from nvflare.fuel.utils.function_utils import find_task_fn, require_arguments


class ExecTaskFuncWrapper:
    def __init__(self, task_fn_path: str, task_main_args: str = None, task_fn_args: Dict = None):
        """Wrapper for function given function path and args

        Args:
            task_fn_path (str): function path (ex: train.main, custom/train.main, custom.train.main).
            task_fn_args (Dict, optional): function arguments to pass in.
        """
        self.task_fn_path = task_fn_path
        self.task_main_args = task_main_args
        self.task_fn_args = task_fn_args
        self.client_api = None
        self.logger = logging.getLogger(self.__class__.__name__)
        self.task_fn = find_task_fn(task_fn_path)
        require_args, args_size, args_default_size = require_arguments(self.task_fn)
        self.check_fn_inputs(task_fn_path, require_args, args_size, args_default_size)
        self.task_fn_require_args = require_args

    def run(self):
        """Call the task_fn with any required arguments."""
        msg = f"\n start task run() with {self.task_fn_path}"
        msg = msg if not self.task_fn_require_args else msg + f", {self.task_fn_args}"
        self.logger.info(msg)
        try:
            if self.task_fn.__name__ == "main":
                curr_argv = sys.argv
                sys.argv = self.get_sys_argv()
                self.task_fn()
                sys.argv = curr_argv
            elif self.task_fn_require_args:
                self.task_fn(**self.task_fn_args)
            else:
                self.task_fn()
        except Exception as e:
            msg = traceback.format_exc()
            self.logger.error(msg)
            if self.client_api:
                self.client_api.exec_queue.ask_abort(msg)
            raise e

    def get_sys_argv(self):
        args_list = [] if not self.task_main_args else self.task_main_args.split()
        return [self.task_fn_path.rsplit(".", 1)[0].replace(".", "/") + ".py"] + args_list

    def check_fn_inputs(self, task_fn_path, require_args: bool, required_args_size: int, args_default_size: int):
        """Check if the provided task_fn_args are compatible with the task_fn."""
        if require_args:
            if not self.task_fn_args:
                raise ValueError(f"function '{task_fn_path}' requires arguments, but none provided")
            elif len(self.task_fn_args) < required_args_size - args_default_size:
                raise ValueError(
                    f"function '{task_fn_path}' requires {required_args_size} "
                    f"arguments, but {len(self.task_fn_args)} provided"
                )
        else:
            if self.task_fn_args and self.task_fn.__name__ != "main":
                msg = f"function '{task_fn_path}' does not require arguments, {self.task_fn_args} will be ignored"
                self.logger.warning(msg)
