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
"""Decomposers for types from app_common and Machine Learning libraries."""
import os
from abc import ABC
from io import BytesIO
from typing import Any

import numpy as np

from nvflare.app_common.abstract.learnable import Learnable
from nvflare.app_common.abstract.model import ModelLearnable
from nvflare.app_common.widgets.event_recorder import _CtxPropReq, _EventReq, _EventStats
from nvflare.fuel.utils import fobs
from nvflare.fuel.utils.fobs import Decomposer
from nvflare.fuel.utils.fobs.decomposer import DictDecomposer


class ModelLearnableDecomposer(fobs.Decomposer):
    def supported_type(self):
        return ModelLearnable

    def decompose(self, target: ModelLearnable) -> Any:
        return target.copy()

    def recompose(self, data: Any) -> ModelLearnable:
        obj = ModelLearnable()
        for k, v in data.items():
            obj[k] = v
        return obj


class NumpyScalarDecomposer(fobs.Decomposer, ABC):
    """Decomposer base class for all numpy types with item method."""

    def decompose(self, target: Any) -> Any:
        return target.item()

    def recompose(self, data: Any) -> np.ndarray:
        return self.supported_type()(data)


class Float64ScalarDecomposer(NumpyScalarDecomposer):
    def supported_type(self):
        return np.float64


class Float32ScalarDecomposer(NumpyScalarDecomposer):
    def supported_type(self):
        return np.float32


class Int64ScalarDecomposer(NumpyScalarDecomposer):
    def supported_type(self):
        return np.int64


class Int32ScalarDecomposer(NumpyScalarDecomposer):
    def supported_type(self):
        return np.int32


class NumpyArrayDecomposer(Decomposer):
    def supported_type(self):
        return np.ndarray

    def decompose(self, target: np.ndarray) -> Any:
        stream = BytesIO()
        np.save(stream, target, allow_pickle=False)
        return stream.getvalue()

    def recompose(self, data: Any) -> np.ndarray:
        stream = BytesIO(data)
        return np.load(stream, allow_pickle=False)


def register():
    if register.registered:
        return

    fobs.register(DictDecomposer(Learnable))
    fobs.register(DictDecomposer(ModelLearnable))

    fobs.register_data_classes(
        _CtxPropReq,
        _EventReq,
        _EventStats,
    )

    fobs.register_folder(os.path.dirname(__file__), __package__)

    register.registered = True


register.registered = False
