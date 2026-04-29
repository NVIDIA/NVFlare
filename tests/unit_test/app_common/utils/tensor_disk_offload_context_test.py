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
import shutil
import tempfile

import nvflare.app_common.utils.tensor_disk_offload_context as ctx
from nvflare.app_common.utils.tensor_disk_offload_context import (
    apply_enable_tensor_disk_offload,
    cleanup_offload_temps_for_token,
    new_registry_token,
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


_TOKEN_A = "test_token_alpha"
_TOKEN_B = "test_token_beta"


def _isolate_registry():
    """Snapshot + clear the module-level registry so each test starts clean."""
    with ctx._outstanding_lock:
        snapshot = {k: set(v) for k, v in ctx._outstanding_temp_dirs.items()}
        ctx._outstanding_temp_dirs.clear()
    return snapshot


def _restore_registry(snapshot):
    with ctx._outstanding_lock:
        ctx._outstanding_temp_dirs.clear()
        ctx._outstanding_temp_dirs.update(snapshot)


def test_new_registry_token_returns_unique_strings():
    a = new_registry_token()
    b = new_registry_token()
    assert isinstance(a, str) and a
    assert a != b


def test_register_and_unregister_offload_temp_dir():
    saved = _isolate_registry()
    try:
        register_offload_temp_dir("/tmp/nvflare_tensors_aaa", _TOKEN_A)
        register_offload_temp_dir("/tmp/nvflare_tensors_bbb", _TOKEN_A)
        assert ctx._outstanding_temp_dirs == {_TOKEN_A: {"/tmp/nvflare_tensors_aaa", "/tmp/nvflare_tensors_bbb"}}

        unregister_offload_temp_dir("/tmp/nvflare_tensors_aaa", _TOKEN_A)
        assert ctx._outstanding_temp_dirs == {_TOKEN_A: {"/tmp/nvflare_tensors_bbb"}}

        # Unregister a path that is not present is a noop.
        unregister_offload_temp_dir("/tmp/nvflare_tensors_never_added", _TOKEN_A)
        assert ctx._outstanding_temp_dirs == {_TOKEN_A: {"/tmp/nvflare_tensors_bbb"}}

        # Unregister the last entry — the empty token bucket is removed.
        unregister_offload_temp_dir("/tmp/nvflare_tensors_bbb", _TOKEN_A)
        assert ctx._outstanding_temp_dirs == {}
    finally:
        _restore_registry(saved)


def test_register_and_unregister_with_empty_token_are_noops():
    """Empty token = opt out of registry tracking. No raises, no entries."""
    saved = _isolate_registry()
    try:
        register_offload_temp_dir("/tmp/nvflare_tensors_xyz", "")
        assert ctx._outstanding_temp_dirs == {}
        unregister_offload_temp_dir("/tmp/nvflare_tensors_xyz", "")
        assert ctx._outstanding_temp_dirs == {}
    finally:
        _restore_registry(saved)


def test_cleanup_for_token_removes_dirs_and_clears_scope():
    saved = _isolate_registry()
    created = []
    try:
        for _ in range(3):
            d = tempfile.mkdtemp(prefix="nvflare_tensors_test_")
            created.append(d)
            register_offload_temp_dir(d, _TOKEN_A)
            with open(os.path.join(d, "sentinel"), "w") as f:
                f.write("x")

        for d in created:
            assert os.path.isdir(d)

        cleanup_offload_temps_for_token(_TOKEN_A)

        for d in created:
            assert not os.path.exists(d), f"{d} should have been removed"
        assert ctx._outstanding_temp_dirs == {}
    finally:
        for d in created:
            if os.path.exists(d):
                shutil.rmtree(d, ignore_errors=True)
        _restore_registry(saved)


def test_cleanup_for_token_is_idempotent_and_handles_missing_dirs():
    saved = _isolate_registry()
    try:
        register_offload_temp_dir("/tmp/nvflare_tensors_nonexistent_1234567890", _TOKEN_A)
        cleanup_offload_temps_for_token(_TOKEN_A)
        assert ctx._outstanding_temp_dirs == {}

        # Second call on empty token — also a noop.
        cleanup_offload_temps_for_token(_TOKEN_A)
        assert ctx._outstanding_temp_dirs == {}

        # Empty token argument — also a noop, never raises.
        cleanup_offload_temps_for_token("")
        assert ctx._outstanding_temp_dirs == {}
    finally:
        _restore_registry(saved)


def test_cleanup_for_token_isolates_other_tokens():
    """Two concurrent tokens. Cleaning up one must not touch the other.

    This is the exact scenario the per-token scoping was added to handle:
    in NVFlare simulator mode multiple FedAvg instances can run in the
    same process, and an abort on one must not delete temp dirs the
    other is still using.
    """
    saved = _isolate_registry()
    a_dir = b_dir = None
    try:
        a_dir = tempfile.mkdtemp(prefix="nvflare_tensors_test_iso_a_")
        b_dir = tempfile.mkdtemp(prefix="nvflare_tensors_test_iso_b_")
        register_offload_temp_dir(a_dir, _TOKEN_A)
        register_offload_temp_dir(b_dir, _TOKEN_B)

        cleanup_offload_temps_for_token(_TOKEN_A)

        assert not os.path.exists(a_dir), "TOKEN_A's dir should have been removed"
        assert os.path.exists(b_dir), "TOKEN_B's dir must NOT be touched"
        assert _TOKEN_A not in ctx._outstanding_temp_dirs
        assert ctx._outstanding_temp_dirs.get(_TOKEN_B) == {b_dir}
    finally:
        if a_dir and os.path.exists(a_dir):
            shutil.rmtree(a_dir, ignore_errors=True)
        if b_dir and os.path.exists(b_dir):
            shutil.rmtree(b_dir, ignore_errors=True)
        _restore_registry(saved)


def test_unregister_via_natural_cleanup_removes_from_registry():
    """When _cleanup_temp_dir runs (the natural path), the registry entry
    must also be removed so the safety-net sweep does not double-clean."""
    from nvflare.app_opt.pt.lazy_tensor_dict import _cleanup_temp_dir

    saved = _isolate_registry()
    d = None
    try:
        d = tempfile.mkdtemp(prefix="nvflare_tensors_test_natural_")
        register_offload_temp_dir(d, _TOKEN_A)
        assert d in ctx._outstanding_temp_dirs[_TOKEN_A]

        _cleanup_temp_dir(d, _TOKEN_A)
        assert not os.path.exists(d)
        assert _TOKEN_A not in ctx._outstanding_temp_dirs
    finally:
        if d and os.path.exists(d):
            shutil.rmtree(d, ignore_errors=True)
        _restore_registry(saved)


def test_apply_propagates_registry_token_to_fobs_context():
    cell = _MockCell(enable_tensor_disk_offload=False)
    apply_enable_tensor_disk_offload(engine=_MockEngine(cell), enabled=True, registry_token="my_token")
    assert cell.ctx["enable_tensor_disk_offload"] is True
    assert cell.ctx.get("tensor_disk_offload_registry_token") == "my_token"


def test_restore_clears_registry_token():
    cell = _MockCell(enable_tensor_disk_offload=False)
    apply_enable_tensor_disk_offload(engine=_MockEngine(cell), enabled=True, registry_token="my_token")
    assert cell.ctx.get("tensor_disk_offload_registry_token") == "my_token"
    restore_enable_tensor_disk_offload(_MockEngine(cell), previous_value=False)
    assert cell.ctx["enable_tensor_disk_offload"] is False
    assert cell.ctx.get("tensor_disk_offload_registry_token") == ""


def test_cleanup_retries_when_writer_keeps_writing(monkeypatch):
    """Reproduces the abort-path race: streaming threads keep writing chunks
    into the temp dir for a few seconds after cleanup_offload_temps_for_token
    starts. The first rmtree pass surfaces ENOTEMPTY; the retry loop must
    keep draining until the writer stops, then succeed.
    """
    saved = _isolate_registry()
    d = None
    try:
        d = tempfile.mkdtemp(prefix="nvflare_tensors_test_race_")
        for i in range(5):
            with open(os.path.join(d, f"chunk_{i}.safetensors"), "w") as f:
                f.write("x")
        register_offload_temp_dir(d, _TOKEN_A)

        # Simulate a writer that injects 3 new chunks AFTER each rmtree pass
        # (mimicking stream threads still flushing post-abort), then stops.
        injection_counter = {"n": 0}
        original_rmtree = ctx.shutil.rmtree

        def racy_rmtree(path, ignore_errors=False, onerror=None):
            original_rmtree(path, ignore_errors=ignore_errors, onerror=onerror)
            if injection_counter["n"] < 3 and not os.path.exists(path):
                os.makedirs(path, exist_ok=True)
                with open(os.path.join(path, f"late_chunk_{injection_counter['n']}"), "w") as f:
                    f.write("y")
                injection_counter["n"] += 1

        monkeypatch.setattr(ctx.shutil, "rmtree", racy_rmtree)
        # Speed up the test — don't actually sleep half a second per pass.
        monkeypatch.setattr(ctx, "_CLEANUP_RETRY_INTERVAL_SEC", 0.01)

        cleanup_offload_temps_for_token(_TOKEN_A)

        # After the writer stopped, the retry loop should have drained the dir.
        assert not os.path.exists(d), f"{d} should have been removed after retries"
        assert injection_counter["n"] == 3
        assert ctx._outstanding_temp_dirs == {}
    finally:
        if d and os.path.exists(d):
            shutil.rmtree(d, ignore_errors=True)
        _restore_registry(saved)


def test_cleanup_gives_up_after_deadline_when_writer_never_stops(monkeypatch, caplog):
    """If a writer never stops, cleanup must not block forever. After the
    deadline it logs a warning and returns; the registry is still cleared
    so subsequent runs don't double-track the same path.
    """
    saved = _isolate_registry()
    d = None
    try:
        d = tempfile.mkdtemp(prefix="nvflare_tensors_test_persistent_")
        register_offload_temp_dir(d, _TOKEN_A)
        original_rmtree = ctx.shutil.rmtree

        def never_empty_rmtree(path, ignore_errors=False, onerror=None):
            original_rmtree(path, ignore_errors=ignore_errors, onerror=onerror)
            os.makedirs(path, exist_ok=True)
            with open(os.path.join(path, "infinite_writer"), "w") as f:
                f.write("z")

        monkeypatch.setattr(ctx.shutil, "rmtree", never_empty_rmtree)
        monkeypatch.setattr(ctx, "_CLEANUP_TIMEOUT_SEC", 0.1)
        monkeypatch.setattr(ctx, "_CLEANUP_RETRY_INTERVAL_SEC", 0.01)

        with caplog.at_level("WARNING"):
            cleanup_offload_temps_for_token(_TOKEN_A)

        assert "after 0.1s of retries" in caplog.text
        assert d in caplog.text
        assert ctx._outstanding_temp_dirs == {}
    finally:
        if d and os.path.exists(d):
            shutil.rmtree(d, ignore_errors=True)
        _restore_registry(saved)


def test_cleanup_succeeds_first_pass_when_no_writer(monkeypatch):
    """Sanity check: the retry loop must not regress the no-race path.
    With no concurrent writer, the first rmtree pass succeeds and we don't
    sleep at all.
    """
    saved = _isolate_registry()
    d = None
    sleep_calls = []
    try:
        d = tempfile.mkdtemp(prefix="nvflare_tensors_test_nowriter_")
        with open(os.path.join(d, "chunk_0"), "w") as f:
            f.write("x")
        register_offload_temp_dir(d, _TOKEN_A)

        monkeypatch.setattr(ctx.time, "sleep", lambda s: sleep_calls.append(s))
        cleanup_offload_temps_for_token(_TOKEN_A)

        assert not os.path.exists(d)
        assert sleep_calls == [], "no retries should be needed without a concurrent writer"
        assert ctx._outstanding_temp_dirs == {}
    finally:
        if d and os.path.exists(d):
            shutil.rmtree(d, ignore_errors=True)
        _restore_registry(saved)
