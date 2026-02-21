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
    def __init__(self, stream_to_disk: bool):
        self.ctx = {"stream_to_disk": stream_to_disk}

    def get_fobs_context(self):
        return dict(self.ctx)

    def update_fobs_context(self, props: dict):
        self.ctx.update(props)


class _MockEngine:
    def __init__(self, cell):
        self.cell = cell

    def get_cell(self):
        return self.cell


class TestSwarmStreamToDiskContext:
    def test_set_stream_to_disk_true(self):
        ctl = object.__new__(SwarmClientController)
        ctl.stream_to_disk = True
        cell = _MockCell(stream_to_disk=False)
        ctl.engine = _MockEngine(cell)

        ctl._set_stream_to_disk()
        assert cell.ctx["stream_to_disk"] is True

    def test_set_stream_to_disk_false(self):
        ctl = object.__new__(SwarmClientController)
        ctl.stream_to_disk = False
        cell = _MockCell(stream_to_disk=True)
        ctl.engine = _MockEngine(cell)

        ctl._set_stream_to_disk()
        assert cell.ctx["stream_to_disk"] is False

    def test_set_stream_to_disk_without_cell(self):
        ctl = object.__new__(SwarmClientController)
        ctl.stream_to_disk = True
        ctl.engine = _MockEngine(cell=None)

        ctl._set_stream_to_disk()
        assert ctl.engine.get_cell() is None
