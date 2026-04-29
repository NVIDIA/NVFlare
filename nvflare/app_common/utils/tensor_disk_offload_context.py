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
import uuid
from typing import Any, Tuple

logger = logging.getLogger(__name__)

_ENABLE_TENSOR_DISK_OFFLOAD = "enable_tensor_disk_offload"
_DISK_OFFLOAD_REGISTRY_TOKEN = "tensor_disk_offload_registry_token"

# Module-level registry mapping an opaque scope token to the set of disk-offload
# temp dirs created within that scope. Used as a safety net for cleanup paths
# where the natural GC of _TempDirRef does not fire (notably FedAvg.run()
# returning early on abort_signal). The per-token scoping is important so that
# in environments where multiple FedAvg instances run concurrently in one
# Python process (e.g., NVFlare simulator mode), one instance's cleanup does
# not delete another instance's still-live temp dirs.
_outstanding_temp_dirs = {}  # type: dict; token (str) -> set[str]
_outstanding_lock = threading.Lock()


def new_registry_token() -> str:
    """Generate a fresh opaque token for a workflow run's registry scope."""
    return uuid.uuid4().hex


def register_offload_temp_dir(path: str, token: str) -> None:
    """Track a disk-offload temp dir under the given scope token.

    Called by download_tensors_to_disk right after tempfile.mkdtemp(). The
    token is read from the cell's FOBS context (set by
    apply_enable_tensor_disk_offload at the start of the workflow's run()).
    No-op if token is empty, allowing direct callers that do not opt into
    the registry (e.g., in tests) to skip tracking.
    """
    if not token:
        return
    with _outstanding_lock:
        _outstanding_temp_dirs.setdefault(token, set()).add(path)


def unregister_offload_temp_dir(path: str, token: str) -> None:
    """Remove a temp dir from the registry. Idempotent. No-op if token empty.

    Called by the natural cleanup path (_cleanup_temp_dir's finally block)
    so the registry stays consistent with on-disk state.
    """
    if not token:
        return
    with _outstanding_lock:
        scope = _outstanding_temp_dirs.get(token)
        if scope is not None:
            scope.discard(path)
            if not scope:
                del _outstanding_temp_dirs[token]


def cleanup_offload_temps_for_token(token: str) -> None:
    """Sweep any registered offload temp dirs under the given scope token.

    Intended to be called from a workflow's ``finally`` block as a safety net
    for exit paths where natural GC does not fire (abort_signal early return,
    unhandled exceptions, etc.). Only dirs registered under THIS token are
    removed; dirs under other tokens (concurrent workflows) are untouched.
    On normal completion the scope is typically empty already because
    _TempDirRef.__del__ has cleaned and unregistered each dir.
    """
    if not token:
        return
    with _outstanding_lock:
        dirs = list(_outstanding_temp_dirs.pop(token, ()))
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
    registry_token: str = "",
) -> Tuple[Any, bool]:
    """Apply enable_tensor_disk_offload to cell FOBS context.

    Also propagates registry_token so that download_tensors_to_disk can scope
    its registry entries to the calling workflow's run(). Pass an empty
    registry_token to opt out of registry tracking.

    Returns:
      (previous value of enable_tensor_disk_offload, applied flag).
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

    fobs_ctx = cell.get_fobs_context()
    previous = fobs_ctx.get(_ENABLE_TENSOR_DISK_OFFLOAD, False)
    updates = {}
    if previous != enabled:
        updates[_ENABLE_TENSOR_DISK_OFFLOAD] = enabled
    if registry_token and fobs_ctx.get(_DISK_OFFLOAD_REGISTRY_TOKEN, "") != registry_token:
        updates[_DISK_OFFLOAD_REGISTRY_TOKEN] = registry_token
    if updates:
        cell.update_fobs_context(updates)
    return previous, True


def restore_enable_tensor_disk_offload(engine, previous_value: Any) -> None:
    """Restore prior enable_tensor_disk_offload value and clear registry token.

    Clearing the token ensures any subsequent operation on this cell that is
    not driven by the just-finished workflow does not register into the now-
    completed token's scope.
    """
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

    fobs_ctx = cell.get_fobs_context()
    updates = {}
    if fobs_ctx.get(_ENABLE_TENSOR_DISK_OFFLOAD, False) != previous_value:
        updates[_ENABLE_TENSOR_DISK_OFFLOAD] = previous_value
    if fobs_ctx.get(_DISK_OFFLOAD_REGISTRY_TOKEN):
        updates[_DISK_OFFLOAD_REGISTRY_TOKEN] = ""
    if updates:
        cell.update_fobs_context(updates)
