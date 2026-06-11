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
from typing import Any, Optional

_ENABLE_TENSOR_DISK_OFFLOAD = "enable_tensor_disk_offload"
_TENSOR_DISK_OFFLOAD_ROOT_DIR = "tensor_disk_offload_root_dir"


@dataclass
class TensorDiskOffloadContext:
    previous_value: Any = None
    previous_root_dir: Optional[str] = None
    root_dir: Optional[str] = None
    applied: bool = False


def _get_cell(engine):
    if not engine:
        return None

    run_manager = getattr(engine, "run_manager", None)
    if run_manager and run_manager.cell:
        return run_manager.cell
    return engine.get_cell()


def setup_tensor_disk_offload(engine, enabled: bool, job_id: str = "job") -> TensorDiskOffloadContext:
    """Enable tensor disk offload in the active cell FOBS context.

    Returns:
      Context needed to restore the prior setting and cleanup temporary files.
    """
    if not enabled:
        return TensorDiskOffloadContext()

    cell = _get_cell(engine)
    if not cell:
        return TensorDiskOffloadContext()

    fobs_ctx = cell.get_fobs_context()
    previous_value = fobs_ctx.get(_ENABLE_TENSOR_DISK_OFFLOAD, False)
    previous_root_dir = fobs_ctx.get(_TENSOR_DISK_OFFLOAD_ROOT_DIR)
    root_dir = tempfile.mkdtemp(prefix=f"nvflare_tensor_offload_{job_id}_")
    try:
        cell.update_fobs_context({_ENABLE_TENSOR_DISK_OFFLOAD: True, _TENSOR_DISK_OFFLOAD_ROOT_DIR: root_dir})
    except Exception:
        shutil.rmtree(root_dir, ignore_errors=True)
        raise
    return TensorDiskOffloadContext(
        previous_value=previous_value,
        previous_root_dir=previous_root_dir,
        root_dir=root_dir,
        applied=True,
    )


def cleanup_tensor_disk_offload(engine, context: TensorDiskOffloadContext) -> None:
    """Restore the prior FOBS context values and remove any temporary offload root."""
    if not context:
        return

    try:
        if context.applied:
            cell = _get_cell(engine)
            if cell:
                cell.update_fobs_context(
                    {
                        _ENABLE_TENSOR_DISK_OFFLOAD: context.previous_value,
                        _TENSOR_DISK_OFFLOAD_ROOT_DIR: context.previous_root_dir,
                    }
                )
    finally:
        if context.root_dir:
            shutil.rmtree(context.root_dir, ignore_errors=True)
