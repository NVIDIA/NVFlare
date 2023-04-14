# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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
from argparse import Namespace
from typing import Any

from nvflare.apis.analytix import AnalyticsDataType
from nvflare.apis.client import Client
from nvflare.apis.dxo import DXO, DataKind
from nvflare.apis.fl_constant import ReservedKey, ServerCommandKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.fl_snapshot import RunSnapshot
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal
from nvflare.apis.utils.decomposers import flare_decomposers
from nvflare.fuel.utils import fobs


class TestFlareDecomposers:

    ID1 = "abc"
    ID2 = "xyz"

    @classmethod
    def setup_class(cls):
        flare_decomposers.register()

    def test_nested_shareable(self):
        shareable = Shareable()
        shareable[ReservedKey.TASK_ID] = TestFlareDecomposers.ID1

        command_shareable = Shareable()
        command_shareable[ReservedKey.TASK_ID] = TestFlareDecomposers.ID2
        command_shareable.set_header(ServerCommandKey.SHAREABLE, shareable)
        new_command_shareable = self._run_fobs(command_shareable)
        assert new_command_shareable[ReservedKey.TASK_ID] == TestFlareDecomposers.ID2

        new_shareable = new_command_shareable.get_header(ServerCommandKey.SHAREABLE)
        assert new_shareable[ReservedKey.TASK_ID] == TestFlareDecomposers.ID1

    def test_fl_context(self):

        context = FLContext()
        context.set_prop("A", "test")
        context.set_prop("B", 123)

        new_context = self._run_fobs(context)

        assert new_context.get_prop("A") == context.get_prop("A")
        assert new_context.get_prop("B") == context.get_prop("B")

    def test_dxo(self):

        dxo = DXO(DataKind.WEIGHTS, {"A": 123})
        dxo.set_meta_prop("B", "test")

        new_dxo = self._run_fobs(dxo)

        assert new_dxo.data_kind == DataKind.WEIGHTS
        assert new_dxo.get_meta_prop("B") == "test"

    def test_client(self):

        client = Client("Name", "Token")
        client.set_prop("A", "test")

        new_client = self._run_fobs(client)

        assert new_client.name == client.name
        assert new_client.token == client.token
        assert new_client.get_prop("A") == client.get_prop("A")

    def test_run_snapshot(self):

        snapshot = RunSnapshot("Job-ID")
        snapshot.set_component_snapshot("comp_id", {"A": 123})

        new_snapshot = self._run_fobs(snapshot)

        assert new_snapshot.job_id == snapshot.job_id
        assert new_snapshot.get_component_snapshot("comp_id") == snapshot.get_component_snapshot("comp_id")

    def test_signal(self):

        signal = Signal()
        signal.trigger("test")

        new_signal = self._run_fobs(signal)

        assert new_signal.value == signal.value
        assert new_signal.trigger_time == signal.trigger_time
        assert new_signal.triggered == signal.triggered

    # The decomposer for the enum is auto-registered
    def test_analytics_data_type(self):

        adt = AnalyticsDataType.SCALARS

        new_adt = self._run_fobs(adt)

        assert new_adt == adt

    def test_namespace(self):

        ns = Namespace(a="foo", b=123)

        new_ns = self._run_fobs(ns)

        assert new_ns.a == ns.a
        assert new_ns.b == ns.b

    @staticmethod
    def _run_fobs(data: Any) -> Any:
        buf = fobs.dumps(data)
        return fobs.loads(buf)
