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

from nvflare.app_common.ccwf.swarm_client_ctl import SwarmClientController


class _MockCell:
    def __init__(self, enable_tensor_disk_offload: bool):
        self.ctx = {"enable_tensor_disk_offload": enable_tensor_disk_offload}

    def get_fobs_context(self):
        return dict(self.ctx)

    def update_fobs_context(self, props: dict):
        self.ctx.update(props)


class _MockEngine:
    def __init__(self, cell):
        self.cell = cell

    def get_cell(self):
        return self.cell


class TestSwarmTensorDiskOffloadContext:
    def test_set_enable_tensor_disk_offload_true(self):
        ctl = object.__new__(SwarmClientController)
        ctl.enable_tensor_disk_offload = True
        cell = _MockCell(enable_tensor_disk_offload=False)
        ctl.engine = _MockEngine(cell)

        ctl._set_enable_tensor_disk_offload()
        assert cell.ctx["enable_tensor_disk_offload"] is True

    def test_set_enable_tensor_disk_offload_false(self):
        ctl = object.__new__(SwarmClientController)
        ctl.enable_tensor_disk_offload = False
        cell = _MockCell(enable_tensor_disk_offload=True)
        ctl.engine = _MockEngine(cell)

        ctl._set_enable_tensor_disk_offload()
        assert cell.ctx["enable_tensor_disk_offload"] is False

    def test_set_enable_tensor_disk_offload_without_cell(self):
        ctl = object.__new__(SwarmClientController)
        ctl.enable_tensor_disk_offload = True
        ctl.engine = _MockEngine(cell=None)

        ctl._set_enable_tensor_disk_offload()
        assert ctl.engine.get_cell() is None
