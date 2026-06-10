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

import shutil
import tempfile
from dataclasses import dataclass
from typing import Any, Tuple

_ENABLE_TENSOR_DISK_OFFLOAD = "enable_tensor_disk_offload"
_TENSOR_DISK_OFFLOAD_ROOT_DIR = "tensor_disk_offload_root_dir"


@dataclass
class TensorDiskOffloadContext:
    previous_value: Any = None
    previous_root_dir: str = None
    root_dir: str = None
    applied: bool = False


def _get_cell(engine):
    if not engine:
        return None

    run_manager = getattr(engine, "run_manager", None)
    if run_manager and run_manager.cell:
        return run_manager.cell
    return engine.get_cell()


def apply_enable_tensor_disk_offload(
    engine,
    enabled: bool,
    root_dir: str = None,
) -> Tuple[Any, bool]:
    """Apply enable_tensor_disk_offload to cell FOBS context.

    Returns:
      (previous value, applied flag).
    """
    cell = _get_cell(engine)
    if not cell:
        return None, False

    previous = cell.get_fobs_context().get(_ENABLE_TENSOR_DISK_OFFLOAD, False)
    if enabled:
        if not root_dir:
            raise ValueError("root_dir must be provided when tensor disk offload is enabled")
        cell.update_fobs_context({_ENABLE_TENSOR_DISK_OFFLOAD: True, _TENSOR_DISK_OFFLOAD_ROOT_DIR: root_dir})
    elif previous != enabled:
        cell.update_fobs_context({_ENABLE_TENSOR_DISK_OFFLOAD: False})
    return previous, True


def restore_enable_tensor_disk_offload(
    engine,
    previous_value: Any,
    root_dir: str = None,
    previous_root_dir: str = None,
) -> None:
    """Restore prior enable_tensor_disk_offload value on a cell."""
    # previous_value is None only when apply was not executed because no
    # engine/cell was available; False is a valid prior value and must restore.
    if previous_value is None:
        return

    cell = _get_cell(engine)
    if not cell:
        return

    if root_dir:
        cell.update_fobs_context(
            {
                _ENABLE_TENSOR_DISK_OFFLOAD: previous_value,
                _TENSOR_DISK_OFFLOAD_ROOT_DIR: previous_root_dir,
            }
        )
        return

    current = cell.get_fobs_context().get(_ENABLE_TENSOR_DISK_OFFLOAD, False)
    if current != previous_value:
        cell.update_fobs_context({_ENABLE_TENSOR_DISK_OFFLOAD: previous_value})


def setup_tensor_disk_offload(engine, enabled: bool, job_id: str = "job") -> TensorDiskOffloadContext:
    """Apply tensor disk offload to the active cell FOBS context.

    Returns:
      Context needed to restore the prior setting and cleanup temporary files.
    """
    if not enabled:
        return TensorDiskOffloadContext()

    root_dir = tempfile.mkdtemp(prefix=f"nvflare_tensor_offload_{job_id}_")
    try:
        cell = _get_cell(engine)
        previous_root_dir = cell.get_fobs_context().get(_TENSOR_DISK_OFFLOAD_ROOT_DIR) if cell else None
        previous_value, applied = apply_enable_tensor_disk_offload(
            engine=engine,
            enabled=enabled,
            root_dir=root_dir,
        )
    except Exception:
        if root_dir:
            shutil.rmtree(root_dir, ignore_errors=True)
        raise
    return TensorDiskOffloadContext(
        previous_value=previous_value,
        previous_root_dir=previous_root_dir,
        root_dir=root_dir,
        applied=applied,
    )


def cleanup_tensor_disk_offload(engine, context: TensorDiskOffloadContext) -> None:
    """Restore tensor disk offload context and remove any temporary offload root."""
    if not context:
        return

    try:
        restore_enable_tensor_disk_offload(
            engine=engine,
            previous_value=context.previous_value,
            root_dir=context.root_dir,
            previous_root_dir=context.previous_root_dir,
        )
    finally:
        if context.root_dir:
            shutil.rmtree(context.root_dir, ignore_errors=True)
