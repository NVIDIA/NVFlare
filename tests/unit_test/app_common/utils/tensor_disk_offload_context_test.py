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

import os
import tempfile

import nvflare.app_common.utils.tensor_disk_offload_context as ctx
from nvflare.app_common.utils.tensor_disk_offload_context import (
    apply_enable_tensor_disk_offload,
    cleanup_all_outstanding_offload_temps,
    register_offload_temp_dir,
    restore_enable_tensor_disk_offload,
    unregister_offload_temp_dir,
)


class _MockCell:
    def __init__(self, enable_tensor_disk_offload: bool):
        self.ctx = {"enable_tensor_disk_offload": enable_tensor_disk_offload}
        self.update_calls = 0

    def get_fobs_context(self):
        return dict(self.ctx)

    def update_fobs_context(self, props: dict):
        self.update_calls += 1
        self.ctx.update(props)


class _MockEngine:
    def __init__(self, cell, run_manager=None):
        self.cell = cell
        self.run_manager = run_manager

    def get_cell(self):
        return self.cell


class _MockRunManager:
    def __init__(self, cell):
        self.cell = cell


def test_apply_returns_previous_and_updates():
    cell = _MockCell(enable_tensor_disk_offload=False)

    previous, applied = apply_enable_tensor_disk_offload(engine=_MockEngine(cell), enabled=True)

    assert previous is False
    assert applied is True
    assert cell.ctx["enable_tensor_disk_offload"] is True
    assert cell.update_calls == 1


def test_apply_skips_update_when_value_unchanged():
    cell = _MockCell(enable_tensor_disk_offload=True)

    previous, applied = apply_enable_tensor_disk_offload(engine=_MockEngine(cell), enabled=True)

    assert previous is True
    assert applied is True
    assert cell.ctx["enable_tensor_disk_offload"] is True
    assert cell.update_calls == 0


def test_restore_sets_previous_value():
    cell = _MockCell(enable_tensor_disk_offload=False)
    previous, _ = apply_enable_tensor_disk_offload(engine=_MockEngine(cell), enabled=True)

    restore_enable_tensor_disk_offload(_MockEngine(cell), previous)
    assert cell.ctx["enable_tensor_disk_offload"] is False


def test_apply_and_restore_noop_when_unavailable():
    previous, applied = apply_enable_tensor_disk_offload(engine=None, enabled=True)
    assert previous is None
    assert applied is False

    previous, applied = apply_enable_tensor_disk_offload(engine=_MockEngine(cell=None), enabled=True)
    assert previous is None
    assert applied is False

    restore_enable_tensor_disk_offload(None, False)
    restore_enable_tensor_disk_offload(None, None)


def test_apply_and_restore_use_run_manager_cell_when_available():
    parent_cell = _MockCell(enable_tensor_disk_offload=False)
    run_cell = _MockCell(enable_tensor_disk_offload=False)
    engine = _MockEngine(cell=parent_cell, run_manager=_MockRunManager(run_cell))

    previous, applied = apply_enable_tensor_disk_offload(engine=engine, enabled=True)

    assert previous is False
    assert applied is True
    assert run_cell.ctx["enable_tensor_disk_offload"] is True
    assert parent_cell.ctx["enable_tensor_disk_offload"] is False

    restore_enable_tensor_disk_offload(engine, previous)
    assert run_cell.ctx["enable_tensor_disk_offload"] is False


def _isolate_registry():
    """Snapshot + clear the module-level registry so each test starts clean."""
    with ctx._outstanding_lock:
        snapshot = set(ctx._outstanding_temp_dirs)
        ctx._outstanding_temp_dirs.clear()
    return snapshot


def _restore_registry(snapshot):
    with ctx._outstanding_lock:
        ctx._outstanding_temp_dirs.clear()
        ctx._outstanding_temp_dirs.update(snapshot)


def test_register_and_unregister_offload_temp_dir():
    saved = _isolate_registry()
    try:
        register_offload_temp_dir("/tmp/nvflare_tensors_aaa")
        register_offload_temp_dir("/tmp/nvflare_tensors_bbb")
        assert ctx._outstanding_temp_dirs == {"/tmp/nvflare_tensors_aaa", "/tmp/nvflare_tensors_bbb"}

        unregister_offload_temp_dir("/tmp/nvflare_tensors_aaa")
        assert ctx._outstanding_temp_dirs == {"/tmp/nvflare_tensors_bbb"}

        # Unregister a path that's not present should be a noop, not raise.
        unregister_offload_temp_dir("/tmp/nvflare_tensors_never_added")
        assert ctx._outstanding_temp_dirs == {"/tmp/nvflare_tensors_bbb"}
    finally:
        _restore_registry(saved)


def test_cleanup_all_outstanding_removes_existing_dirs_and_clears_registry():
    saved = _isolate_registry()
    created = []
    try:
        for _ in range(3):
            d = tempfile.mkdtemp(prefix="nvflare_tensors_test_")
            created.append(d)
            register_offload_temp_dir(d)
            # Drop a sentinel file so we can verify rmtree actually ran.
            with open(os.path.join(d, "sentinel"), "w") as f:
                f.write("x")

        for d in created:
            assert os.path.isdir(d)

        cleanup_all_outstanding_offload_temps()

        for d in created:
            assert not os.path.exists(d), f"{d} should have been removed"
        assert ctx._outstanding_temp_dirs == set()
    finally:
        for d in created:
            if os.path.exists(d):
                import shutil

                shutil.rmtree(d, ignore_errors=True)
        _restore_registry(saved)


def test_cleanup_all_outstanding_is_idempotent_and_handles_missing_dirs():
    saved = _isolate_registry()
    try:
        # Register a path that does not exist — cleanup should not raise.
        register_offload_temp_dir("/tmp/nvflare_tensors_nonexistent_1234567890")
        cleanup_all_outstanding_offload_temps()
        assert ctx._outstanding_temp_dirs == set()

        # Second call on empty registry — also a noop.
        cleanup_all_outstanding_offload_temps()
        assert ctx._outstanding_temp_dirs == set()
    finally:
        _restore_registry(saved)


def test_unregister_via_natural_cleanup_removes_from_registry():
    """When _cleanup_temp_dir runs (the natural path), the registry entry
    must also be removed so the safety-net sweep does not double-clean."""
    from nvflare.app_opt.pt.lazy_tensor_dict import _cleanup_temp_dir

    saved = _isolate_registry()
    d = None
    try:
        d = tempfile.mkdtemp(prefix="nvflare_tensors_test_natural_")
        register_offload_temp_dir(d)
        assert d in ctx._outstanding_temp_dirs

        _cleanup_temp_dir(d)
        assert not os.path.exists(d)
        assert d not in ctx._outstanding_temp_dirs
    finally:
        if d and os.path.exists(d):
            import shutil

            shutil.rmtree(d, ignore_errors=True)
        _restore_registry(saved)
