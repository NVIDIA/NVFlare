# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
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
from datetime import datetime
from typing import Any, Type

import tensorflow as tf
import torch

from nvflare.apis.dxo import DXO, DataKind
from nvflare.fuel.utils import fobs, decomposers


class Simple:

    def __init__(self, num, name, timestamp):
        self.num = num
        self.name = name
        self.timestamp = timestamp

    def __eq__(self, other):
        return self.num == other.num and self.name == other.name and self.timestamp == other.timestamp


class FlareObject:

    def __init__(self, name, dxo_obj):
        self.name = name
        self.dxo_obj = dxo_obj


class SimpleDecomposer(fobs.Decomposer):

    @staticmethod
    def supported_type() -> Type[Any]:
        return Simple

    def decompose(self, obj) -> Any:
        return [obj.num, obj.name, obj.timestamp]

    def recompose(self, data: Any) -> Simple:
        return Simple(data[0], data[1], data[2])


class FlareObjectDecomposer(fobs.Decomposer):

    @staticmethod
    def supported_type() -> Type[Any]:
        return FlareObject

    def decompose(self, obj) -> Any:
        return [obj.name, obj.dxo_obj]

    def recompose(self, data: Any) -> FlareObject:
        return FlareObject(data[0], data[1])


class TestFobs:

    original_count: int
    flare_obj: FlareObject

    @classmethod
    def setup_class(cls):

        simple = Simple(123, "Simple Object", datetime.now())
        dxo = DXO(DataKind.MODEL, {"object": simple}, {"tuple": ("test metadata", 456)})
        cls.flare_obj = FlareObject("Test Flare Object", dxo)
        decomposers.register_all()

        cls.original_count = fobs.num_decomposers()
        fobs.register(SimpleDecomposer())
        fobs.register(FlareObjectDecomposer)

    def test_register(self):
        count = fobs.num_decomposers() - TestFobs.original_count
        assert count == 2, "Registered decomposers don't match"

    def test_full_cycle(self):
        data = fobs.serialize(TestFobs.flare_obj)
        new_obj = fobs.deserialize(data)
        self.compare(new_obj)

    def test_aliases(self):
        data = fobs.dumps(TestFobs.flare_obj)
        new_obj = fobs.loads(data)
        self.compare(new_obj)

    def test_pt_tensor(self):
        pt_tensor = torch.ones(2, 3, 4)
        data = fobs.serialize(pt_tensor)
        tensor = fobs.deserialize(data)
        assert torch.equal(pt_tensor, tensor), "PyTorch Tensors don't match"

    def test_tf_eager_tensor(self):
        # TF2 is in eager mode by default
        tf_tensor = tf.constant([[1.23, 2.71828, 3.1415], [123, 2456, 789]], dtype=tf.float64)
        data = fobs.serialize(tf_tensor)
        tensor = fobs.deserialize(data)
        result = tf.math.reduce_all(tf.equal(tf_tensor, tensor))
        assert result.numpy(), "TensorFlow EagerTensors don't match"

    @staticmethod
    def compare(obj: FlareObject):
        assert obj.name == TestFobs.flare_obj.name, "Names don't match"
        assert obj.dxo_obj.data == TestFobs.flare_obj.dxo_obj.data, "DXO data doesn't match"
        assert obj.dxo_obj.meta == TestFobs.flare_obj.dxo_obj.meta, "DXO metadata doesn't match"
