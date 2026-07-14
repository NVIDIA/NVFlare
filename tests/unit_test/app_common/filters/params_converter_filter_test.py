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

"""Tests for the client-edge params-conversion filter (ParamsConverterFilter)."""

from nvflare.apis.dxo import DXO, DataKind
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.app_common.abstract.params_converter import ParamsConverter
from nvflare.app_common.filters.params_converter_filter import ParamsConverterFilter


class _DoublingConverter(ParamsConverter):
    def convert(self, params, fl_ctx):
        return {k: v * 2 for k, v in params.items()}


class TestParamsConverterFilter:
    def test_applies_converter_to_weights_dxo(self):
        f = ParamsConverterFilter(_DoublingConverter())
        dxo = DXO(data_kind=DataKind.WEIGHTS, data={"w": 3, "b": 5})
        out = f.process_dxo(dxo, Shareable(), FLContext())
        assert out.data == {"w": 6, "b": 10}

    def test_ignores_non_params_dxo(self):
        f = ParamsConverterFilter(_DoublingConverter())
        dxo = DXO(data_kind=DataKind.ANALYTIC, data={"x": 1})
        # not a params kind: passed through unchanged (returns None -> keep original)
        assert f.process_dxo(dxo, Shareable(), FLContext()) is None
