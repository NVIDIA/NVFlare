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

from unittest.mock import MagicMock, patch

import pytest

from nvflare.client.cell.decomposers import register_framework_decomposers
from nvflare.client.config import ExchangeFormat


@pytest.mark.parametrize(
    "params_exchange_format,server_expected_format",
    [
        (ExchangeFormat.NUMPY, ExchangeFormat.NUMPY),
        (ExchangeFormat.PYTORCH, ExchangeFormat.NUMPY),
        (ExchangeFormat.KERAS_LAYER_WEIGHTS, ExchangeFormat.NUMPY),
    ],
)
def test_concrete_non_native_wire_format_does_not_import_framework(params_exchange_format, server_expected_format):
    with patch("builtins.__import__", wraps=__import__) as import_module:
        register_framework_decomposers(params_exchange_format, server_expected_format)

    assert not any(call.args[0] == "nvflare.app_opt.pt.decomposers" for call in import_module.call_args_list)


@pytest.mark.parametrize(
    "params_exchange_format,server_expected_format",
    [
        (ExchangeFormat.NUMPY, ExchangeFormat.PYTORCH),
        (ExchangeFormat.PYTORCH, ExchangeFormat.PYTORCH),
        (ExchangeFormat.RAW, ExchangeFormat.PYTORCH),
        (ExchangeFormat.PYTORCH, ExchangeFormat.RAW),
    ],
)
def test_declared_native_wire_format_fails_fast_when_decomposer_is_unavailable(
    params_exchange_format, server_expected_format
):
    real_import = __import__

    def import_module(name, *args, **kwargs):
        if name == "nvflare.app_opt.pt.decomposers":
            raise ImportError("missing framework")
        return real_import(name, *args, **kwargs)

    with (
        patch("builtins.__import__", side_effect=import_module),
        pytest.raises(RuntimeError, match="required by the declared pytorch wire format"),
    ):
        register_framework_decomposers(params_exchange_format, server_expected_format)


@pytest.mark.parametrize(
    "params_exchange_format,server_expected_format",
    [
        (ExchangeFormat.RAW, ExchangeFormat.NUMPY),
        (ExchangeFormat.NUMPY, ExchangeFormat.RAW),
        (ExchangeFormat.RAW, ExchangeFormat.RAW),
    ],
)
def test_raw_wire_format_registers_opportunistically(params_exchange_format, server_expected_format):
    logger = MagicMock()
    real_import = __import__

    def import_module(name, *args, **kwargs):
        if name == "nvflare.app_opt.pt.decomposers":
            raise ImportError("missing framework")
        return real_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=import_module):
        register_framework_decomposers(params_exchange_format, server_expected_format, logger)

    logger.debug.assert_called_once()
