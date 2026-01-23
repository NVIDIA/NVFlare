# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Subprocess Backend for forwarding calls to worker subprocesses.

This backend is used when Fox runs in subprocess mode (inprocess=False).
It wraps a SubprocessLauncher and forwards calls to the worker process
via CellNet. This is used by both:
- AppRunner (simulation path)
- CollabExecutor (FLARE path)

The design maintains symmetry between FlareBackend (for remote CellNet calls)
and SubprocessBackend (for local subprocess calls via CellNet).
"""

from nvflare.collab.api.backend import Backend
from nvflare.collab.api.call_opt import CallOption
from nvflare.collab.api.gcc import GroupCallContext


class SubprocessBackend(Backend):
    """Backend that forwards calls to a subprocess worker.

    This backend wraps a SubprocessLauncher and delegates calls to the worker
    subprocess. The worker runs the user's @fox.publish methods in a separate
    process (e.g., launched via torchrun for multi-GPU training).

    Architecture:
        SubprocessBackend -> SubprocessLauncher -> CellNet -> CollabWorker -> User's @fox.publish
    """

    def __init__(
        self,
        subprocess_launcher,
        abort_signal,
        thread_executor,
        target_name: str = "",
    ):
        """Initialize SubprocessBackend.

        Args:
            subprocess_launcher: The SubprocessLauncher that manages the worker.
            abort_signal: Signal to abort execution.
            thread_executor: ThreadPoolExecutor for async operations.
            target_name: Name of the target object (for logging/identification).
        """
        Backend.__init__(self, abort_signal)
        self.subprocess_launcher = subprocess_launcher
        self.thread_executor = thread_executor
        self.target_name = target_name

    def call_target(self, context, target_name: str, call_opt: CallOption, func_name: str, *args, **kwargs):
        """Forward a call to the subprocess worker.

        Args:
            context: Call context.
            target_name: Name of the target object.
            call_opt: Call options (timeout, etc.).
            func_name: Name of the function to call.
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            Result from the worker.
        """
        if self.abort_signal.triggered:
            from nvflare.apis.fl_exception import RunAborted

            return RunAborted("job is aborted")

        if not self.subprocess_launcher.is_ready():
            raise RuntimeError(f"Subprocess worker is not ready for {target_name}")

        try:
            # Forward call to subprocess via launcher
            # The launcher handles CellNet communication to the worker
            result = self.subprocess_launcher.call(
                func_name=func_name,
                args=args,
                kwargs=kwargs,
            )
            return result
        except Exception as e:
            self.logger.error(f"Call to subprocess failed: {e}")
            return e

    def call_target_in_group(self, gcc: GroupCallContext, func_name: str, *args, **kwargs):
        """Handle group call by submitting to thread executor."""
        self.thread_executor.submit(self._run_func_in_group, gcc, func_name, args, kwargs)

    def _run_func_in_group(self, gcc: GroupCallContext, func_name: str, args, kwargs):
        """Execute group call and set result."""
        try:
            result = self.call_target(
                context=gcc.context,
                target_name=gcc.target_name,
                call_opt=gcc.call_opt,
                func_name=func_name,
                *args,
                **kwargs,
            )
            gcc.send_completed()
            gcc.set_result(result)
        except Exception as ex:
            gcc.set_exception(ex)
