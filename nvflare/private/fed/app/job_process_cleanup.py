# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

from typing import Callable, Optional

from nvflare.fuel.f3.streaming.shutdown import shutdown_f3_streaming
from nvflare.private.fed.utils.fed_utils import security_close
from nvflare.security.logging import secure_format_exception

_COMMAND_CALLBACK_DRAIN_TIMEOUT = 5.0


def shutdown_job_process_runtime(
    stop_command_admission: Optional[Callable[[], None]],
    wait_for_command_callbacks: Optional[Callable[[float], bool]],
    stop_cell: Optional[Callable[[], None]],
    logger,
    before_streaming_shutdown: Optional[Callable[[], None]] = None,
) -> None:
    """Drain job-process communication before closing process security state."""

    def _run_stage(name: str, action: Optional[Callable[[], None]]) -> None:
        if action is None:
            return
        try:
            action()
        except Exception as e:
            if logger:
                logger.warning(f"failed to stop {name}: {secure_format_exception(e)}")

    # Reject new app commands before any final workspace publication. The
    # publication uses the process-global F3 pools, so it must complete before
    # those pools are irreversibly stopped. Teardown still runs if publication
    # raises, and the original publication error remains visible to the caller.
    _run_stage("command admission", stop_command_admission)
    try:
        if before_streaming_shutdown:
            before_streaming_shutdown()
    finally:
        # Drain every callback admitted before the gate, then stop transport so
        # any command blocked on Cell communication wakes. Audit/security state
        # must outlive all teardown stages.
        _run_stage("F3 streaming", shutdown_f3_streaming)
        _run_stage("Cell", stop_cell)
        if wait_for_command_callbacks:
            try:
                drained = wait_for_command_callbacks(_COMMAND_CALLBACK_DRAIN_TIMEOUT)
                if not drained and logger:
                    logger.warning(
                        f"timed out after {_COMMAND_CALLBACK_DRAIN_TIMEOUT} seconds waiting for command callbacks"
                    )
            except Exception as e:
                if logger:
                    logger.warning(f"failed to drain command callbacks: {secure_format_exception(e)}")
        _run_stage("security services", security_close)
