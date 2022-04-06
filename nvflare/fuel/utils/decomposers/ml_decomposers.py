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
"""Decomposers for types from Machine Learning frameworks and libraries."""
from abc import ABC
from io import BytesIO
from typing import Any

import numpy as np
import tensorflow as tf
import torch
from tensorflow.python.framework import ops

from nvflare.fuel.utils.fobs.decomposer import Decomposer


class NumpyTypeDecomposer(Decomposer, ABC):
    """Decomposer base class for all numpy types with item method."""

    def decompose(self, target: Any) -> Any:
        return target.item()

    def recompose(self, data: Any) -> np.ndarray:
        return self.supported_type()(data)


class Float64TypeDecomposer(NumpyTypeDecomposer):
    @staticmethod
    def supported_type():
        return np.float64


class Float32TypeDecomposer(NumpyTypeDecomposer):
    @staticmethod
    def supported_type():
        return np.float32


class Int64TypeDecomposer(NumpyTypeDecomposer):
    @staticmethod
    def supported_type():
        return np.int64


class Int32TypeDecomposer(NumpyTypeDecomposer):
    @staticmethod
    def supported_type():
        return np.int32


class ArrayDecomposer(Decomposer):
    @staticmethod
    def supported_type():
        return np.ndarray

    def decompose(self, target: np.ndarray) -> Any:
        stream = BytesIO()
        np.save(stream, target, allow_pickle=False)
        return stream.getvalue()

    def recompose(self, data: Any) -> np.ndarray:
        stream = BytesIO(data)
        return np.load(stream, allow_pickle=False)


class PtTensorDecomposer(Decomposer):
    @staticmethod
    def supported_type():
        return torch.Tensor

    def decompose(self, target: torch.Tensor) -> Any:
        stream = BytesIO()
        # torch.save uses Pickle so converting Tensor to ndarray first
        array = target.numpy()
        np.save(stream, array, allow_pickle=False)
        return stream.getvalue()

    def recompose(self, data: Any) -> torch.Tensor:
        stream = BytesIO(data)
        array = np.load(stream, allow_pickle=False)
        return torch.from_numpy(array)


class TfEagerTensorDecomposer(Decomposer):
    @staticmethod
    def supported_type():
        return ops.EagerTensor

    def decompose(self, target: ops.EagerTensor) -> Any:
        dt_value = target.dtype.as_datatype_enum
        data = tf.io.serialize_tensor(target).numpy()
        return [dt_value, data]

    def recompose(self, data: Any) -> ops.EagerTensor:
        dt = tf.DType(data[0])
        return tf.io.parse_tensor(data[1], dt)
