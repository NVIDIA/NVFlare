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

from types import SimpleNamespace
from unittest.mock import MagicMock

from nvflare.private.fed.server.server_engine import ServerEngine


def _make_engine():
    args = SimpleNamespace(set=[])
    engine = ServerEngine(
        server=MagicMock(),
        args=args,
        client_manager=MagicMock(),
        snapshot_persistor=MagicMock(),
    )
    engine.logger = MagicMock()
    return engine


class TestServerEngineGetCell:
    def test_prefers_run_manager_cell(self):
        engine = _make_engine()
        parent_cell = MagicMock(name="parent_cell")
        run_cell = MagicMock(name="run_cell")
        engine.cell = parent_cell
        engine.run_manager = SimpleNamespace(cell=run_cell)

        assert engine.get_cell() is run_cell

    def test_falls_back_to_parent_cell_when_run_cell_missing(self):
        engine = _make_engine()
        parent_cell = MagicMock(name="parent_cell")
        engine.cell = parent_cell
        engine.run_manager = SimpleNamespace(cell=None)

        assert engine.get_cell() is parent_cell

    def test_returns_none_when_no_cells_available(self):
        engine = _make_engine()
        engine.cell = None
        engine.run_manager = None

        assert engine.get_cell() is None
