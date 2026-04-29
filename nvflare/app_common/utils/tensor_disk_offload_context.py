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

import logging
import shutil
import threading
from typing import Any, Tuple

logger = logging.getLogger(__name__)

_ENABLE_TENSOR_DISK_OFFLOAD = "enable_tensor_disk_offload"

# Module-level registry of disk-offload temp dirs created during the current
# process lifetime. Used as a safety net for cleanup paths where the natural
# GC of _TempDirRef does not fire (notably FedAvg.run() returning early on
# abort_signal — the partial state held by the controller's aggregation
# helper keeps LazyTensorDict references reachable, blocking GC).
_outstanding_temp_dirs = set()
_outstanding_lock = threading.Lock()


def register_offload_temp_dir(path: str) -> None:
    """Track a disk-offload temp dir so it can be swept up on workflow exit.

    Called by download_tensors_to_disk right after tempfile.mkdtemp().
    """
    with _outstanding_lock:
        _outstanding_temp_dirs.add(path)


def unregister_offload_temp_dir(path: str) -> None:
    """Remove a temp dir from the registry once it has been cleaned up via
    the natural _TempDirRef.__del__ / DiskTensorConsumer.download_failed
    path. Idempotent.
    """
    with _outstanding_lock:
        _outstanding_temp_dirs.discard(path)


def cleanup_all_outstanding_offload_temps() -> None:
    """Sweep any registered offload temp dirs that have not yet been cleaned.

    Intended to be called from a workflow's ``finally`` block as a safety net
    for exit paths where natural GC does not fire (abort_signal early return,
    unhandled exceptions, etc.). On normal completion the registry is
    typically empty by the time this runs because _TempDirRef.__del__ has
    already cleaned and unregistered each dir.
    """
    with _outstanding_lock:
        dirs = list(_outstanding_temp_dirs)
        _outstanding_temp_dirs.clear()
    for d in dirs:
        try:
            shutil.rmtree(d)
        except FileNotFoundError:
            pass
        except Exception as e:
            logger.warning("failed to cleanup outstanding offload temp dir '%s': %s", d, e)


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
