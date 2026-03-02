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

from typing import Any, Callable, Optional


_ENABLE_TENSOR_DISK_OFFLOAD = "enable_tensor_disk_offload"


def apply_enable_tensor_disk_offload(
    engine,
    enabled: bool,
    warning_fn: Optional[Callable[[str], None]] = None,
    info_fn: Optional[Callable[[str], None]] = None,
) -> Any:
    """Apply enable_tensor_disk_offload to cell FOBS context.

    Returns:
      previous value (or None when engine/cell is unavailable).
    """
    if not engine:
        if enabled and warning_fn:
            warning_fn("enable_tensor_disk_offload is enabled but no active engine is available; using in-memory download path")
        return None

    cell = engine.get_cell()
    if not cell:
        if enabled and warning_fn:
            warning_fn("enable_tensor_disk_offload is enabled but no active cell is available; using in-memory download path")
        return None

    previous = cell.get_fobs_context().get(_ENABLE_TENSOR_DISK_OFFLOAD, False)
    cell.update_fobs_context({_ENABLE_TENSOR_DISK_OFFLOAD: enabled})
    if info_fn:
        info_fn(f"_set_enable_tensor_disk_offload: {previous} -> {enabled}")
    return previous


def restore_enable_tensor_disk_offload(engine, previous_value: Any, info_fn: Optional[Callable[[str], None]] = None) -> None:
    """Restore prior enable_tensor_disk_offload value on a cell."""
    if not engine or previous_value is None:
        return

    cell = engine.get_cell()
    if not cell:
        return

    current = cell.get_fobs_context().get(_ENABLE_TENSOR_DISK_OFFLOAD, False)
    if current != previous_value:
        cell.update_fobs_context({_ENABLE_TENSOR_DISK_OFFLOAD: previous_value})
        if info_fn:
            info_fn(f"_restore_enable_tensor_disk_offload: {current} -> {previous_value}")
