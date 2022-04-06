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
"""Decomposers for FLARE objects.

This module contains all the decomposers for commonly used FLARE objects.
This is only a temporary location. It should be moved to a different package
so FOBS doesn't depend on other FLARE packages to avoid circular dependencies.

"""
from typing import Any

from nvflare.apis.dxo import DXO
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.app_common.abstract.learnable import Learnable
from nvflare.fuel.utils.fobs.decomposer import Decomposer


class LearnableDecomposer(Decomposer):
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


class ShareableDecomposer(Decomposer):
    @staticmethod
    def supported_type():
        return Shareable

    def decompose(self, target: Shareable) -> Any:
        return target.copy()

    def recompose(self, data: Any) -> Shareable:
        obj = Shareable()
        for k, v in data.items():
            obj[k] = v
        return obj


class ContextDecomposer(Decomposer):
    @staticmethod
    def supported_type():
        return FLContext

    def decompose(self, target: FLContext) -> Any:
        return [target.model, target.props]

    def recompose(self, data: Any) -> FLContext:
        obj = FLContext()
        obj.model = data[0]
        obj.props = data[1]
        return obj


class DxoDecomposer(Decomposer):
    @staticmethod
    def supported_type():
        return DXO

    def decompose(self, target: DXO) -> Any:
        return [target.data_kind, target.data, target.meta]

    def recompose(self, data: Any) -> DXO:
        return DXO(data[0], data[1], data[2])
