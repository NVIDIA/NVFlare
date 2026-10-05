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
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Optional

from nvflare.fuel.utils.fobs import FOBSContextKey

_ENABLE_TENSOR_DISK_OFFLOAD = FOBSContextKey.TENSOR_DISK_OFFLOAD
_TENSOR_DISK_OFFLOAD_ROOT_DIR = "tensor_disk_offload_root_dir"
_TENSOR_DISK_OFFLOAD_CONTEXT = "tensor_disk_offload_context"


@dataclass
class TensorDiskOffloadContext:
    previous_value: Any = None
    previous_root_dir: Optional[str] = None
    root_dir: Optional[str] = None
    applied: bool = False
    previous_context: Optional["TensorDiskOffloadContext"] = None
    lock: Any = field(default_factory=Lock, repr=False)
    closed: bool = False


def _get_cell(engine):
    if not engine:
        return None

    run_manager = getattr(engine, "run_manager", None)
    if run_manager and run_manager.cell:
        return run_manager.cell
    return engine.get_cell()


def setup_tensor_disk_offload(
    engine, enabled: bool, job_id: str = "job", root_dir: Optional[str] = None
) -> TensorDiskOffloadContext:
    """Enable tensor disk offload in the active cell FOBS context.

    Args:
        engine: engine that owns the active Cell.
        enabled: whether to prepare disk-backed tensor downloads.
        job_id: identifier used to name the temporary offload root.
        root_dir: optional existing parent directory for the temporary root.

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
    offload_dir = tempfile.mkdtemp(prefix=f"nvflare_tensor_offload_{job_id}_", dir=root_dir)
    context = TensorDiskOffloadContext(
        previous_value=previous_value,
        previous_root_dir=previous_root_dir,
        root_dir=offload_dir,
        applied=True,
        previous_context=fobs_ctx.get(_TENSOR_DISK_OFFLOAD_CONTEXT),
    )
    try:
        cell.update_fobs_context(
            {
                _ENABLE_TENSOR_DISK_OFFLOAD: True,
                _TENSOR_DISK_OFFLOAD_ROOT_DIR: offload_dir,
                _TENSOR_DISK_OFFLOAD_CONTEXT: context,
            }
        )
    except Exception:
        shutil.rmtree(offload_dir, ignore_errors=True)
        raise
    return context


def cleanup_tensor_disk_offload(engine, context: TensorDiskOffloadContext) -> None:
    """Cancel active downloads, restore the prior FOBS context and remove the offload root."""
    if not context:
        return

    if context.root_dir:
        # Copies of the decode context share this guard. Close admission before
        # snapshotting consumers, including downloads paused before registration.
        with context.lock:
            context.closed = True
        try:
            from nvflare.app_opt.pt.tensor_downloader import cleanup_active_disk_tensor_downloads
        except ImportError:
            pass  # PyTorch is optional; without it there are no disk tensor consumers.
        else:
            cleanup_active_disk_tensor_downloads(
                reason="tensor disk offload ended before download completed", root_dir=context.root_dir
            )

    try:
        if context.applied:
            cell = _get_cell(engine)
            if cell:
                cell.update_fobs_context(
                    {
                        _ENABLE_TENSOR_DISK_OFFLOAD: context.previous_value,
                        _TENSOR_DISK_OFFLOAD_ROOT_DIR: context.previous_root_dir,
                        _TENSOR_DISK_OFFLOAD_CONTEXT: context.previous_context,
                    }
                )
    finally:
        if context.root_dir:
            shutil.rmtree(context.root_dir, ignore_errors=True)
