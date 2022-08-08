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
"""Decomposers for types from app_common and Machine Learning libraries."""
import os
from abc import ABC
from io import BytesIO
from typing import Any

import numpy as np

from nvflare.app_common.abstract.learnable import Learnable
from nvflare.app_common.widgets.event_recorder import _CtxPropReq, _EventReq, _EventStats
from nvflare.fuel.utils import fobs


class LearnableDecomposer(fobs.Decomposer):
    @staticmethod
    def supported_type():
        return Learnable

    def decompose(self, target: Learnable) -> Any:
        return target.copy()

    def recompose(self, data: Any) -> Learnable:
        obj = Learnable()
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
    @staticmethod
    def supported_type():
        return np.float64


class Float32ScalarDecomposer(NumpyScalarDecomposer):
    @staticmethod
    def supported_type():
        return np.float32


class Int64ScalarDecomposer(NumpyScalarDecomposer):
    @staticmethod
    def supported_type():
        return np.int64


class Int32ScalarDecomposer(NumpyScalarDecomposer):
    @staticmethod
    def supported_type():
        return np.int32


class NumpyArrayDecomposer(fobs.Decomposer):
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


class CtxPropReqDecomposer(fobs.Decomposer):
    @staticmethod
    def supported_type():
        return _CtxPropReq

    def decompose(self, target: _CtxPropReq) -> Any:
        return [target.dtype, target.is_private, target.is_sticky, target.allow_none]

    def recompose(self, data: Any) -> _CtxPropReq:
        return _CtxPropReq(data[0], data[1], data[2], data[3])


class EventReqDecomposer(fobs.Decomposer):
    @staticmethod
    def supported_type():
        return _EventReq

    def decompose(self, target: _EventReq) -> Any:
        return [target.ctx_reqs, target.peer_ctx_reqs, target.ctx_block_list, target.peer_ctx_block_List]

    def recompose(self, data: Any) -> _EventReq:
        return _EventReq(data[0], data[1], data[2], data[3])


class EventStatsDecomposer(fobs.Decomposer):
    @staticmethod
    def supported_type():
        return _EventStats

    def decompose(self, target: _EventStats) -> Any:
        return [
            target.call_count,
            target.prop_missing,
            target.prop_none_value,
            target.prop_dtype_mismatch,
            target.prop_attr_mismatch,
            target.prop_block_list_violation,
            target.peer_ctx_missing,
        ]

    def recompose(self, data: Any) -> _EventStats:
        stats = _EventStats()
        stats.call_count = data[0]
        stats.prop_missing = data[1]
        stats.prop_none_value = data[2]
        stats.prop_dtype_mismatch = data[3]
        stats.prop_attr_mismatch = data[4]
        stats.prop_block_list_violation = data[5]
        stats.peer_ctx_missing = data[6]
        return stats


def register():
    if register.registered:
        return

    fobs.register_folder(os.path.dirname(__file__), __package__)
    register.registered = True


register.registered = False
