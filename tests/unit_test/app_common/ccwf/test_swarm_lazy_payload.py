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

from unittest.mock import patch

from nvflare.apis.fl_context import FLContext
from nvflare.app_common.ccwf.client_ctl import ClientSideController
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
    def test_finalize_restores_enable_tensor_disk_offload(self):
        ctl = object.__new__(SwarmClientController)
        ctl.enable_tensor_disk_offload = True
        cell = _MockCell(enable_tensor_disk_offload=False)
        ctl.engine = _MockEngine(cell)
        cell.ctx["enable_tensor_disk_offload"] = True
        ctl._previous_enable_tensor_disk_offload = False

        with patch.object(ClientSideController, "finalize", autospec=True) as super_finalize:
            ctl.finalize(FLContext())

        assert cell.ctx["enable_tensor_disk_offload"] is False
        assert ctl._previous_enable_tensor_disk_offload is None
        super_finalize.assert_called_once()

    def test_finalize_with_missing_cell(self):
        ctl = object.__new__(SwarmClientController)
        ctl.engine = _MockEngine(cell=None)
        ctl._previous_enable_tensor_disk_offload = False

        with patch.object(ClientSideController, "finalize", autospec=True) as super_finalize:
            ctl.finalize(FLContext())

        assert ctl._previous_enable_tensor_disk_offload is None
        super_finalize.assert_called_once()
