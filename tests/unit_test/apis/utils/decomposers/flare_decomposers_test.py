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
import filecmp
import os
import tempfile
import uuid
from argparse import Namespace
from typing import Any

import pytest

from nvflare.apis.analytix import AnalyticsDataType
from nvflare.apis.client import Client
from nvflare.apis.dxo import DXO, DataKind, from_shareable
from nvflare.apis.fl_constant import ReservedKey, ServerCommandKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.fl_snapshot import RunSnapshot
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal
from nvflare.apis.utils.decomposers import flare_decomposers
from nvflare.fuel.utils import fobs
from nvflare.fuel.utils.fobs.datum import Datum

LARGE_DATA = 5 * 1024 * 1024


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

    @pytest.mark.parametrize(
        "size",
        [100, 1000, LARGE_DATA],
    )
    def test_byte_stream(self, size):
        d = Shareable()
        d["x"] = os.urandom(size)
        d["y"] = "1234567890ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        d["u"] = ":;.'[]{}`~<>!@#$%^&*()-_+="
        d["v"] = "中文字母测试两岸猿声啼不住轻舟已过万重山"
        d["z"] = {
            "za": 12345,
            "zb": b"123456789012345678901234567890123456789012345678",
            "zc": "中文字母测试两岸猿声啼不住轻舟已过万重山:;.'[]{}`~<>!@#$%^&*()-_+=",
        }
        ds = fobs.dumps(d, max_value_size=10)
        dd = fobs.loads(ds)
        assert d == dd

    @pytest.mark.parametrize(
        "size",
        [100, 1000, LARGE_DATA],
    )
    def test_file_stream(self, size):
        d = Shareable()
        d["x"] = os.urandom(size)
        d["y"] = b"123456789012345678901234567890123456789012345678901234567890"
        d["z"] = {
            "za": 12345,
            "zb": b"123456789012345678901234567890123456789012345678",
        }

        with tempfile.TemporaryDirectory() as td:
            file_path = os.path.join(td, str(uuid.uuid4()))
            fobs.dumpf(d, file_path, max_value_size=15)
            df = fobs.loadf(file_path)
            assert df == d

    @pytest.mark.parametrize(
        "file_size",
        [100, 1000, LARGE_DATA],
    )
    def test_file_datum(self, file_size):
        d = Shareable()
        d["y"] = b"123456789012345678901234567890123456789012345678901234567890"
        with tempfile.TemporaryDirectory() as td:
            temp_file = os.path.join(td, str(uuid.uuid4()))
            with open(temp_file, "wb") as f:
                f.write(os.urandom(file_size))
            d["z"] = Datum.file_datum(temp_file)

            ds = fobs.dumps(d)
            dd = fobs.loads(ds)

            assert isinstance(dd, Shareable)
            assert d["y"] == dd["y"]
            datum = dd["z"]
            assert isinstance(datum, Datum)
            received_file_name = datum.value
            assert os.path.isfile(received_file_name)
            assert filecmp.cmp(received_file_name, temp_file)
            os.remove(received_file_name)

    @staticmethod
    def _dxo_equal(x: DXO, y: DXO):
        assert x.data_kind == y.data_kind and x.data == y.data and x.meta == y.meta

    def test_large_dxo(self):
        d = DXO(data_kind=DataKind.WEIGHTS, data={"x": os.urandom(LARGE_DATA)})
        ds = fobs.dumps(d)
        dd = fobs.loads(ds)
        self._dxo_equal(d, dd)

    def test_dxo_collection(self):
        d1 = DXO(data_kind=DataKind.WEIGHTS, data={"x": 1, "y": os.urandom(200)})

        d2 = DXO(data_kind=DataKind.WEIGHTS, data={"x": 3, "y": os.urandom(100)})

        d = DXO(data_kind=DataKind.COLLECTION, data={"x": d1, "y": d2})
        ds = fobs.dumps(d, max_value_size=20)
        dd = fobs.loads(ds)
        dd1 = dd.data["x"]
        dd2 = dd.data["y"]
        self._dxo_equal(dd1, d1)
        self._dxo_equal(dd2, d2)
        assert dd.data_kind == d.data_kind

    def test_dxo_shareable(self):
        dxo = DXO(
            data_kind=DataKind.WEIGHTS,
            data={"x": 1, "y": os.urandom(200), "z": "中文字母测试两岸猿声啼不住轻舟已过万重山"},
        )
        s1 = dxo.to_shareable()
        ds = fobs.dumps(s1, max_value_size=15)
        s2 = fobs.loads(ds)
        dxo2 = from_shareable(s2)
        self._dxo_equal(dxo, dxo2)

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
        buf = fobs.dumps(data, max_value_size=15)
        return fobs.loads(buf)
