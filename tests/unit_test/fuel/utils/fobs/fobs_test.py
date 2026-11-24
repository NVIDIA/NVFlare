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

import queue
from datetime import datetime
from typing import Any

import msgpack
import numpy as np
import pytest

from nvflare.apis.analytix import AnalyticsDataType
from nvflare.apis.dxo import DXO, DataKind, from_bytes, from_file
from nvflare.apis.utils.decomposers.flare_decomposers import DXODecomposer
from nvflare.app_common.abstract.statistics_spec import DataType, Feature, HistogramType
from nvflare.app_common.decomposers.numpy_decomposers import NumpyArrayDecomposer
from nvflare.app_common.statistics.statisitcs_objects_decomposer import FeatureDecomposer
from nvflare.fuel.f3.drivers.driver_params import DriverParams
from nvflare.fuel.utils import fobs
from nvflare.fuel.utils.buffer_list import ConsumableBufferList
from nvflare.fuel.utils.class_loader import get_class_name
from nvflare.fuel.utils.fobs import Decomposer
from nvflare.fuel.utils.fobs import fobs as fobs_impl
from nvflare.fuel.utils.fobs.datum import DatumManager
from nvflare.fuel.utils.fobs.decomposer import DataClassDecomposer, EnumTypeDecomposer


class TestFobs:

    NUMBER = 123456
    FLOAT = 123.456
    NAME = "FOBS Test"
    SET = {4, 5, 6}
    NOW = datetime.now()

    test_data = {
        "str": "Test string",
        "number": NUMBER,
        "float": FLOAT,
        "list": [7, 8, 9],
        "set": SET,
        "tuple": ("abc", "xyz"),
        "time": NOW,
    }

    def test_builtin(self):
        buf = fobs.dumps(TestFobs.test_data)
        data = fobs.loads(buf)
        assert data["number"] == TestFobs.NUMBER
        assert data["set"] == TestFobs.SET

    def test_unsupported_classes(self):
        with pytest.raises(Exception):
            # Queue contains collections.deque, which has no __dict__, can't be handled as Data Class
            unsupported_class = queue.Queue()
            try:
                fobs.dumps(unsupported_class)
            except Exception as ex:
                print(ex)
                raise ex

    def test_decomposers(self):
        test_class = ExampleClass(TestFobs.NUMBER)
        fobs.register(ExampleClassDecomposer)
        buf = fobs.dumps(test_class)
        new_class = fobs.loads(buf)
        assert new_class.number == TestFobs.NUMBER

    def test_decomposer_alias_uses_canonical_wire_type(self):
        fobs.reset()
        fobs.register(ExampleClassDecomposerWithAlias)
        buf = fobs.dumps(ExampleClassAlias(TestFobs.NUMBER))

        # A receiver needs only the canonical decomposer, not the sender's alias.
        fobs.reset()
        fobs.register(ExampleClassDecomposer)
        restored = fobs.loads(buf)

        assert type(restored) is ExampleClass
        assert restored.number == TestFobs.NUMBER

    def test_no_registration(self):
        test_class = ExampleClass(TestFobs.NUMBER)
        fobs.register(ExampleClassDecomposer)
        buf = fobs.dumps(test_class)
        # Unregister the decomposer
        fobs.reset()

        # ExampleClassDecomposer is not builtin, not allowed
        with pytest.raises(ValueError):
            new_class = fobs.loads(buf)
            assert new_class.number == TestFobs.NUMBER

    def test_whitelist_enforcement(self):
        fobs.reset()
        # Serialize ExampleDataClass (auto-registers it with DataClassDecomposer)
        test_class = ExampleDataClass(TestFobs.NAME)
        buf = fobs.dumps(test_class)
        fobs.reset()

        # After reset, ExampleDataClass is not in the whitelist.
        # DataClassDecomposer is a builtin decomposer so the decomposer check passes,
        # but the type whitelist check should block deserialization.
        with pytest.raises(ValueError, match="not allowed"):
            fobs.loads(buf)

        # After explicitly adding to the whitelist, deserialization should succeed.
        fobs.add_type_name_whitelist("tests.unit_test.fuel.utils.fobs.fobs_test.ExampleDataClass")
        new_class = fobs.loads(buf)
        assert new_class.name == TestFobs.NAME
        fobs.reset()

    def test_auto_registration(self):
        fobs.reset()
        test_class = ExampleDataClass(TestFobs.NAME)
        # pack() auto-registers a session-scoped DataClassDecomposer for ExampleDataClass.
        buf = fobs.dumps(test_class)
        new_class = fobs.loads(buf)
        assert new_class.name == TestFobs.NAME

    @pytest.mark.parametrize("register_sender", [False, True])
    def test_builtin_custom_decomposer_type_allowed_after_reset(self, isolated_fobs, register_sender):
        if register_sender:
            fobs.register(DXODecomposer)
        dxo = DXO(data_kind=DataKind.WEIGHTS, data={"w": 1.0})
        buf = fobs.dumps(dxo)

        # After reset, DXODecomposer will not be in _decomposers. The type name must still
        # pass the whitelist gate so the receiver can select its built-in decomposer.
        fobs.reset()
        restored = fobs.loads(buf)
        assert isinstance(restored, DXO)
        assert restored.data_kind == DataKind.WEIGHTS
        assert restored.data == {"w": 1.0}
        fobs.reset()

    def test_reserialize_lazily_decoded_builtin_values(self, isolated_fobs):
        fobs.reset()
        fobs.register(NumpyArrayDecomposer)
        fobs.register(FeatureDecomposer)
        original = DXO(DataKind.WEIGHTS, {"weights": np.arange(3), "feature": Feature("age", DataType.INT)})
        encoded = original.to_bytes()

        fobs.reset()
        decoded = from_bytes(encoded)
        assert get_class_name(np.ndarray) not in fobs_impl._decomposers
        assert get_class_name(Feature) not in fobs_impl._decomposers

        encoded_again = decoded.to_bytes()
        fobs.reset()
        restored = from_bytes(encoded_again)
        np.testing.assert_array_equal(restored.data["weights"], original.data["weights"])
        assert type(restored.data["feature"]) is Feature
        assert restored.data["feature"] == original.data["feature"]

    @pytest.mark.parametrize("register_receiver", [False, True])
    def test_legacy_dxo_bytes_and_file(self, isolated_fobs, tmp_path, register_receiver):
        fobs.reset()
        # Standalone senders previously used this codec when DXO was unregistered.
        fobs.register_data_classes(DXO)
        original = DXO(DataKind.COLLECTION, {"weights": DXO(DataKind.WEIGHTS, {"w": 1.0})}, {"round": 1})
        encoded = original.to_bytes()
        path = str(tmp_path / "legacy.dxo")
        original.to_file(path)

        fobs.reset()
        if register_receiver:
            fobs.register(DXODecomposer)
        type_name = get_class_name(DXO)
        registered = fobs_impl._decomposers.get(type_name)

        for restored in (from_bytes(encoded), from_file(path)):
            assert type(restored) is DXO
            assert restored.data_kind is DataKind.COLLECTION
            assert restored.meta == original.meta
            assert restored.data["weights"].to_dict() == original.data["weights"].to_dict()
        assert fobs_impl._decomposers.get(type_name) is registered

    def test_legacy_dxo_still_validates_fields(self, isolated_fobs):
        fobs.reset()
        fobs.register_data_classes(DXO)
        dxo = DXO(DataKind.WEIGHTS, {})
        dxo.data = "invalid weights"
        encoded = dxo.to_bytes()

        fobs.reset()
        with pytest.raises(ValueError, match="invalid DXO: invalid data"):
            from_bytes(encoded)

    def test_connector_params_enum_allowed_on_receiver(self):
        fobs.reset()
        connector_info = {"params": {DriverParams.SECURE: True, DriverParams.URL: "grpc://localhost:8002"}}
        buf = fobs.dumps(connector_info)

        # Simulate a separate receiver that has not registered DriverParams.
        fobs.reset()
        try:
            restored = fobs.loads(buf)
            assert restored == connector_info
            assert all(isinstance(key, DriverParams) for key in restored["params"])
        finally:
            fobs.reset()

    @pytest.mark.parametrize("enum_type", [AnalyticsDataType, HistogramType])
    def test_lazy_decode_keeps_process_registry_unchanged(self, isolated_fobs, enum_type):
        value = next(iter(enum_type))
        type_name = get_class_name(enum_type)
        frame = {
            fobs_impl.FOBS_TYPE: type_name,
            fobs_impl.FOBS_DECOMPOSER: get_class_name(DataClassDecomposer),
            fobs_impl.FOBS_DATA: {},
        }

        def decode():
            return fobs_impl.deserialize(msgpack.packb(frame), manager=DatumManager())

        with pytest.raises(ValueError, match="requires decomposer"):
            decode()
        assert type_name not in fobs_impl._decomposers

        frame[fobs_impl.FOBS_DECOMPOSER] = get_class_name(EnumTypeDecomposer)
        frame[fobs_impl.FOBS_DATA] = "unknown member"
        with pytest.raises(KeyError):
            decode()
        assert type_name not in fobs_impl._decomposers

        frame[fobs_impl.FOBS_DATA] = value.name
        assert decode() is value
        del frame[fobs_impl.FOBS_DECOMPOSER]
        assert decode() is value
        assert type_name not in fobs_impl._decomposers

    def test_register_rejects_non_decomposer_before_construction(self):
        class NotDecomposer:
            def __init__(self):
                pytest.fail("registration called a non-decomposer constructor")

        with pytest.raises(TypeError, match="is not a Decomposer"):
            fobs.register(NotDecomposer)

    def test_reset_mutates_whitelist_in_place(self):
        fobs.reset()
        whitelist_ref = fobs_impl._type_name_whitelist
        fobs.add_type_name_whitelist("tests.unit_test.fuel.utils.fobs.fobs_test.ExampleDataClass")
        assert "tests.unit_test.fuel.utils.fobs.fobs_test.ExampleDataClass" in whitelist_ref

        fobs.reset()
        assert fobs_impl._type_name_whitelist is whitelist_ref
        assert "tests.unit_test.fuel.utils.fobs.fobs_test.ExampleDataClass" not in whitelist_ref

    def test_buffer_list(self):
        buf = fobs.dumps(TestFobs.test_data, buffer_list=True)
        data = fobs.loads(buf)
        assert data["number"] == TestFobs.NUMBER

    def test_buffer_list_reassembles_externalized_bytes_from_network_chunks(self):
        expected = b"x" * 2048
        encoded = fobs.dumps({"blob": expected}, buffer_list=True, max_value_size=1024)
        wire_data = b"".join(encoded)
        network_chunks = [memoryview(wire_data)[start : start + 127] for start in range(0, len(wire_data), 127)]

        data = fobs.loads(network_chunks)

        assert type(data["blob"]) is bytes
        assert data["blob"] == expected
        assert network_chunks

    def test_buffer_list_releases_consumable_network_chunks(self):
        expected = b"x" * 2048
        encoded = fobs.dumps({"blob": expected}, buffer_list=True, max_value_size=1024)
        wire_data = b"".join(encoded)
        network_chunks = ConsumableBufferList(
            memoryview(wire_data)[start : start + 127] for start in range(0, len(wire_data), 127)
        )

        data = fobs.loads(network_chunks)

        assert data["blob"] == expected
        assert not network_chunks


class ExampleClass:
    def __init__(self, number):
        self.number = number


class ExampleDataClass:
    def __init__(self, name):
        self.name = name


class ExampleClassAlias(ExampleClass):
    pass


class ExampleClassDecomposer(Decomposer):
    def supported_type(self):
        return ExampleClass

    def decompose(self, target: ExampleClass, manager: DatumManager = None) -> Any:
        return target.number

    def recompose(self, data: Any, manager: DatumManager = None) -> ExampleClass:
        return ExampleClass(data)


class ExampleClassDecomposerWithAlias(ExampleClassDecomposer):
    def supported_aliases(self):
        return [ExampleClassAlias]
