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
    apply_enable_tensor_disk_offload,
    restore_enable_tensor_disk_offload,
)


class _MockCell:
    def __init__(self, enable_tensor_disk_offload: bool):
        self.ctx = {"enable_tensor_disk_offload": enable_tensor_disk_offload}

    def get_fobs_context(self):
        return dict(self.ctx)

    def update_fobs_context(self, props: dict):
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
