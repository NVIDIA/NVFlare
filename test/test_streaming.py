# Copyright (c) 2021, NVIDIA CORPORATION.
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

import pytest

from nvflare.apis.dxo import DXO, DataKind
from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_context import FLContext
from nvflare.app_common.widgets.streaming import send_analytic_dxo, write_image, write_scalar, write_scalars, write_text

INVALID_TEST_CASES = [
    (list(), dict(), FLContext(), TypeError, f"expect comp to be an instance of FLComponent, but got {type(list())}"),
    (FLComponent(), dict(), FLContext(), TypeError, f"expect dxo to be an instance of DXO, but got {type(dict())}"),
    (
        FLComponent(),
        DXO(data={"k": "v"}, data_kind=DataKind.ANALYTIC),
        list(),
        TypeError,
        f"expect fl_ctx to be an instance of FLContext, but got {type(list())}",
    ),
]

INVALID_WRITE_TEST_CASES = [
    (write_scalar, list(), 1.0, TypeError, f"expect tag to be an instance of str, but got {type(list())}"),
    (write_scalar, "tag", list(), TypeError, f"expect value to be an instance of float, but got {type(list())}"),
    (write_scalars, list(), 1.0, TypeError, f"expect tag to be an instance of str, but got {type(list())}"),
    (write_scalars, "tag", 1.0, TypeError, f"expect value to be an instance of dict, but got {type(1.0)}"),
    (write_text, list(), 1.0, TypeError, f"expect tag to be an instance of str, but got {type(list())}"),
    (write_text, "tag", 1.0, TypeError, f"expect value to be an instance of str, but got {type(1.0)}"),
    (write_image, list(), 1.0, TypeError, f"expect tag to be an instance of str, but got {type(list())}"),
]


class TestStreaming:
    @pytest.mark.parametrize("comp,dxo,fl_ctx,expected_error,expected_msg", INVALID_TEST_CASES)
    def test_invalid_send_analytic_dxo(self, comp, dxo, fl_ctx, expected_error, expected_msg):
        with pytest.raises(expected_error, match=expected_msg):
            send_analytic_dxo(comp=comp, dxo=dxo, fl_ctx=fl_ctx)

    @pytest.mark.parametrize("func,tag,value,expected_error,expected_msg", INVALID_WRITE_TEST_CASES)
    def test_invalid_write_func(self, func, tag, value, expected_error, expected_msg):
        with pytest.raises(expected_error, match=expected_msg):
            func(tag, value)
