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

from typing import Any, Tuple

_ENABLE_TENSOR_DISK_OFFLOAD = "enable_tensor_disk_offload"


def apply_enable_tensor_disk_offload(
    engine,
    enabled: bool,
) -> Tuple[Any, bool]:
    """Apply enable_tensor_disk_offload to cell FOBS context.

    Returns:
      (previous value, applied flag).
    """
    if not engine:
        return None, False

    run_manager = getattr(engine, "run_manager", None)
    if run_manager and run_manager.cell:
        cell = run_manager.cell
    else:
        cell = engine.get_cell()
    if not cell:
        return None, False

    previous = cell.get_fobs_context().get(_ENABLE_TENSOR_DISK_OFFLOAD, False)
    if previous != enabled:
        cell.update_fobs_context({_ENABLE_TENSOR_DISK_OFFLOAD: enabled})
    return previous, True


def restore_enable_tensor_disk_offload(engine, previous_value: Any) -> None:
    """Restore prior enable_tensor_disk_offload value on a cell."""
    # previous_value is None only when apply was not executed because no
    # engine/cell was available; False is a valid prior value and must restore.
    if not engine or previous_value is None:
        return

    run_manager = getattr(engine, "run_manager", None)
    if run_manager and run_manager.cell:
        cell = run_manager.cell
    else:
        cell = engine.get_cell()
    if not cell:
        return

    current = cell.get_fobs_context().get(_ENABLE_TENSOR_DISK_OFFLOAD, False)
    if current != previous_value:
        cell.update_fobs_context({_ENABLE_TENSOR_DISK_OFFLOAD: previous_value})
