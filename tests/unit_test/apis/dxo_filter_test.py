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

import pytest

from nvflare.apis.dxo import DXO, DataKind
from nvflare.apis.dxo_filter import DXOFilter
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable


class _NoOpDXOFilter(DXOFilter):
    """Minimal concrete DXOFilter for testing."""

    def process_dxo(self, dxo: DXO, shareable: Shareable, fl_ctx: FLContext):
        return dxo


class TestDXOFilterInit:
    def test_valid_data_kinds(self):
        f = _NoOpDXOFilter(
            supported_data_kinds=[DataKind.WEIGHTS, DataKind.WEIGHT_DIFF],
            data_kinds_to_filter=[DataKind.WEIGHTS],
        )
        assert f.data_kinds == [DataKind.WEIGHTS]

    def test_none_supported_kinds_accepts_all(self):
        f = _NoOpDXOFilter(supported_data_kinds=None, data_kinds_to_filter=None)
        assert f.data_kinds is None

    def test_unsupported_kind_raises_with_correct_message(self):
        """Error message must list supported_data_kinds, not data_kinds_to_filter."""
        supported = [DataKind.WEIGHTS, DataKind.WEIGHT_DIFF]
        bad_kinds = [DataKind.METRICS]
        with pytest.raises(ValueError) as exc_info:
            _NoOpDXOFilter(supported_data_kinds=supported, data_kinds_to_filter=bad_kinds)
        msg = str(exc_info.value)
        # The message must reference the supported kinds, not the invalid ones
        assert DataKind.WEIGHTS in msg
        assert DataKind.WEIGHT_DIFF in msg
        # And it must identify the bad kinds too
        assert DataKind.METRICS in msg

    def test_supported_kinds_must_be_list(self):
        with pytest.raises(ValueError, match="supported_data_kinds must be a list"):
            _NoOpDXOFilter(supported_data_kinds="WEIGHTS", data_kinds_to_filter=None)

    def test_data_kinds_to_filter_must_be_list(self):
        with pytest.raises(ValueError, match="data_kinds_to_filter must be a list"):
            _NoOpDXOFilter(supported_data_kinds=[DataKind.WEIGHTS], data_kinds_to_filter="WEIGHTS")

    def test_no_filter_kinds_defaults_to_supported(self):
        supported = [DataKind.WEIGHTS, DataKind.WEIGHT_DIFF]
        f = _NoOpDXOFilter(supported_data_kinds=supported, data_kinds_to_filter=None)
        assert f.data_kinds == supported
