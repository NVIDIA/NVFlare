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
import os
import runpy
import sys
import threading
import traceback

from nvflare.client.in_process.api import TOPIC_ABORT
from nvflare.fuel.data_event.data_bus import DataBus
from nvflare.fuel.data_event.event_manager import EventManager
from nvflare.fuel.utils.log_utils import get_module_logger
from nvflare.fuel.utils.secret_utils import resolve_secret_refs, split_command_preserving_secret_refs


class TaskScriptRunner:
    logger = get_module_logger(__module__, __qualname__)

    def __init__(self, custom_dir: str, script_path: str, script_args: str = None, redirect_print_to_log=True):
        """Wrapper for function given function path and args

        Args:
            custom_dir (str): site name
            script_path (str): script file name, such as train.py
            script_args (str, Optional): script arguments to pass in.
        """

        self.redirect_print_to_log = redirect_print_to_log
        self.event_manager = EventManager(DataBus())
        self.script_args = script_args
        self.custom_dir = custom_dir
        self.script_path = script_path
        self.script_full_path = self.get_script_full_path(self.custom_dir, self.script_path)
        self._runtime_lock = threading.Lock()
        self._runtime_released = False
        self._original_print = None
        self._original_argv = None
        self._original_argv_values = None
        self._task_argv = None

    def run(self):
        """Call the task_fn with any required arguments."""
        self.logger.info(f"start task run() with full path: {self.script_full_path}")
        try:
            if not self._activate_runtime():
                return
            runpy.run_path(self.script_full_path, run_name="__main__")
        except ImportError as ie:
            msg = "attempted relative import with no known parent package"
            if ie.msg == msg:
                xs = [p for p in sys.path if self.script_full_path.startswith(p)]
                import_base_path = max(xs, key=len)
                raise ImportError(
                    f"{ie.msg}, the relative import is not support. python import is based off the sys.path: {import_base_path}"
                )
            else:
                raise ie
        except Exception as e:
            msg = traceback.format_exc()
            self.logger.error(msg)
            self.logger.error("fire abort event")
            self.event_manager.fire_event(TOPIC_ABORT, f"'{self.script_full_path}' is aborted, {msg}")
            raise e
        finally:
            self.release_runtime()

    def _activate_runtime(self) -> bool:
        with self._runtime_lock:
            # finalize() can release a trainer before its thread reaches this point.
            if self._runtime_released:
                return False
            self._original_argv = sys.argv
            self._original_argv_values = list(sys.argv)
            self._task_argv = self.get_sys_argv()
            sys.argv = self._task_argv
            if self.redirect_print_to_log:
                self._original_print = builtins.print
                builtins.print = log_print
            return True

    def release_runtime(self):
        """Restore globals owned by this runner, even when its thread must be abandoned."""
        with self._runtime_lock:
            first_release = not self._runtime_released
            self._runtime_released = True
            if first_release and self._original_print is not None:
                builtins.print = self._original_print
            if self._original_argv is not None:
                self._original_argv[:] = self._original_argv_values
                if first_release and sys.argv is self._task_argv:
                    sys.argv = self._original_argv

    def get_sys_argv(self):
        # Preserve the runner's legacy whitespace splitting for existing arguments. Only quoted
        # spans containing a secret ref are grouped, so a composite such as
        # "Bearer ${secret:TOKEN}" remains one argument without changing unrelated backslashes.
        args_list = (
            []
            if not self.script_args
            else split_command_preserving_secret_refs(
                self.script_args,
                posix=False,
                group_secret_ref_quotes="${secret:" in self.script_args,
            )
        )
        # Resolve ${secret:ENV_VAR} references from this site's environment after splitting,
        # so injected values containing whitespace stay single arguments. The resolved values
        # exist only in the argv handed to the script and must never be logged.
        args_list = [resolve_secret_refs(arg) for arg in args_list]
        return [self.script_full_path] + args_list

    def get_script_full_path(self, custom_dir, script_path) -> str:
        if not custom_dir:
            raise ValueError("custom_dir must be not empty")
        if not script_path:
            raise ValueError("script_path must be not empty")

        target_file = None
        script_filename = os.path.basename(script_path)
        script_dirs = os.path.dirname(script_path)

        if os.path.isabs(script_path):
            if not os.path.isfile(script_path):
                raise ValueError(f"script_path='{script_path}' not found")
            return script_path

        for r, dirs, files in os.walk(custom_dir):
            for f in files:
                absolute_path = os.path.join(r, f)
                if absolute_path.endswith(os.sep + script_path):
                    target_file = absolute_path
                    break

                if not custom_dir and not script_dirs and f == script_filename:
                    target_file = absolute_path
                    break

            if target_file:
                break

        if not target_file:
            msg = f"Can not find {script_path}"
            self.event_manager.fire_event(TOPIC_ABORT, f"'{self.script_path}' is aborted, {msg}")
            raise ValueError(msg)
        return target_file


def log_print(*args, logger=TaskScriptRunner.logger, **kwargs):
    # Create a message from print arguments
    message = " ".join(str(arg) for arg in args)
    logger.info(message)
