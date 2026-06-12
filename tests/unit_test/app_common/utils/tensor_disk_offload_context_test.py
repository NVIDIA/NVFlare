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

from nvflare.app_common.utils.tensor_disk_offload_context import (
    _TENSOR_DISK_OFFLOAD_ROOT_DIR,
    cleanup_tensor_disk_offload,
    setup_tensor_disk_offload,
)


class _MockCell:
    def __init__(self, enable_tensor_disk_offload: bool, root_dir=None):
        self.ctx = {"enable_tensor_disk_offload": enable_tensor_disk_offload}
        if root_dir is not None:
            self.ctx[_TENSOR_DISK_OFFLOAD_ROOT_DIR] = root_dir
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


def _patch_mkdtemp(monkeypatch, tmp_path):
    root_dir = tmp_path / "nvflare_tensor_offload_root"

    def fake_mkdtemp(prefix):
        root_dir.mkdir()
        return str(root_dir)

    monkeypatch.setattr("nvflare.app_common.utils.tensor_disk_offload_context.tempfile.mkdtemp", fake_mkdtemp)
    return root_dir


def test_setup_records_previous_and_updates(tmp_path, monkeypatch):
    cell = _MockCell(enable_tensor_disk_offload=False)
    root_dir = _patch_mkdtemp(monkeypatch, tmp_path)

    context = setup_tensor_disk_offload(engine=_MockEngine(cell), enabled=True, job_id="job")

    assert context.applied is True
    assert context.previous_value is False
    assert context.previous_root_dir is None
    assert context.root_dir == str(root_dir)
    assert cell.ctx["enable_tensor_disk_offload"] is True
    assert cell.ctx[_TENSOR_DISK_OFFLOAD_ROOT_DIR] == str(root_dir)
    assert cell.update_calls == 1


def test_setup_disabled_does_not_touch_cell():
    cell = _MockCell(enable_tensor_disk_offload=True, root_dir="/tmp/owner")

    context = setup_tensor_disk_offload(engine=_MockEngine(cell), enabled=False)

    assert context.applied is False
    assert cell.ctx["enable_tensor_disk_offload"] is True
    assert cell.ctx[_TENSOR_DISK_OFFLOAD_ROOT_DIR] == "/tmp/owner"
    assert cell.update_calls == 0


def test_setup_enabled_without_cell_does_not_create_temp_dir(monkeypatch):
    def fail_mkdtemp(prefix):
        raise AssertionError("mkdtemp should not be called without an active cell")

    monkeypatch.setattr("nvflare.app_common.utils.tensor_disk_offload_context.tempfile.mkdtemp", fail_mkdtemp)

    for engine in (None, _MockEngine(cell=None)):
        context = setup_tensor_disk_offload(engine=engine, enabled=True)
        assert context.applied is False
        assert context.root_dir is None


def test_setup_cleanup_restores_previous_values(tmp_path, monkeypatch):
    cell = _MockCell(enable_tensor_disk_offload=True, root_dir="/tmp/owner")
    root_dir = _patch_mkdtemp(monkeypatch, tmp_path)

    context = setup_tensor_disk_offload(engine=_MockEngine(cell), enabled=True, job_id="job")
    assert context.applied is True
    assert cell.ctx["enable_tensor_disk_offload"] is True
    assert cell.ctx[_TENSOR_DISK_OFFLOAD_ROOT_DIR] == str(root_dir)

    cleanup_tensor_disk_offload(engine=_MockEngine(cell), context=context)

    assert cell.ctx["enable_tensor_disk_offload"] is True
    assert cell.ctx[_TENSOR_DISK_OFFLOAD_ROOT_DIR] == "/tmp/owner"
    assert not root_dir.exists()


def test_cleanup_noop_when_not_applied():
    cell = _MockCell(enable_tensor_disk_offload=True, root_dir="/tmp/owner")

    cleanup_tensor_disk_offload(engine=_MockEngine(cell), context=None)
    cleanup_tensor_disk_offload(engine=_MockEngine(cell), context=setup_tensor_disk_offload(engine=None, enabled=True))

    assert cell.ctx["enable_tensor_disk_offload"] is True
    assert cell.ctx[_TENSOR_DISK_OFFLOAD_ROOT_DIR] == "/tmp/owner"
    assert cell.update_calls == 0


def test_cleanup_removes_root_dir_when_cell_gone(tmp_path, monkeypatch):
    cell = _MockCell(enable_tensor_disk_offload=False)
    root_dir = _patch_mkdtemp(monkeypatch, tmp_path)

    context = setup_tensor_disk_offload(engine=_MockEngine(cell), enabled=True, job_id="job")
    assert root_dir.exists()

    cleanup_tensor_disk_offload(engine=_MockEngine(cell=None), context=context)

    assert not root_dir.exists()


def test_setup_cleanup_use_run_manager_cell_when_available(tmp_path, monkeypatch):
    parent_cell = _MockCell(enable_tensor_disk_offload=False)
    run_cell = _MockCell(enable_tensor_disk_offload=False)
    engine = _MockEngine(cell=parent_cell, run_manager=_MockRunManager(run_cell))
    root_dir = _patch_mkdtemp(monkeypatch, tmp_path)

    context = setup_tensor_disk_offload(engine=engine, enabled=True, job_id="job")

    assert context.applied is True
    assert run_cell.ctx["enable_tensor_disk_offload"] is True
    assert run_cell.ctx[_TENSOR_DISK_OFFLOAD_ROOT_DIR] == str(root_dir)
    assert parent_cell.ctx["enable_tensor_disk_offload"] is False

    cleanup_tensor_disk_offload(engine=engine, context=context)
    assert run_cell.ctx["enable_tensor_disk_offload"] is False
    assert parent_cell.update_calls == 0
