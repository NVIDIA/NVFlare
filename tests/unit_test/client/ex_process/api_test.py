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

from unittest.mock import patch

from nvflare.app_common.abstract.params_converter import ParamsConverter
from nvflare.client.config import ClientConfig, ConfigKey, ExchangeFormat
from nvflare.client.converter_utils import create_default_params_converters


class _DummyConverter(ParamsConverter):
    def convert(self, params, fl_ctx):
        return params


def _fake_optional_import(module: str, name: str):
    class _LocalDummyConverter(_DummyConverter):
        pass

    _LocalDummyConverter.__name__ = name
    return _LocalDummyConverter, True


def test_create_params_converters_for_pytorch_format():
    config = ClientConfig(
        {
            ConfigKey.TASK_EXCHANGE: {
                ConfigKey.EXCHANGE_FORMAT: ExchangeFormat.PYTORCH,
                ConfigKey.SERVER_EXPECTED_FORMAT: ExchangeFormat.NUMPY,
                ConfigKey.TRAIN_TASK_NAME: "train",
                ConfigKey.EVAL_TASK_NAME: "validate",
                ConfigKey.SUBMIT_MODEL_TASK_NAME: "submit_model",
            }
        }
    )

    with patch("nvflare.client.converter_utils.optional_import", side_effect=_fake_optional_import):
        from_converter, to_converter = create_default_params_converters(
            server_expected_format=config.get_server_expected_format(),
            params_exchange_format=config.get_exchange_format(),
            train_task_name=config.get_train_task(),
            eval_task_name=config.get_eval_task(),
            submit_model_task_name=config.get_submit_model_task(),
        )

    assert from_converter is not None
    assert to_converter is not None
    assert from_converter.__class__.__name__ == "NumpyToPTParamsConverter"
    assert to_converter.__class__.__name__ == "PTToNumpyParamsConverter"
    assert from_converter.supported_tasks == ["train", "validate"]
    assert to_converter.supported_tasks == ["train", "submit_model"]


def test_create_params_converters_skips_when_server_expected_format_is_not_numpy():
    config = ClientConfig(
        {
            ConfigKey.TASK_EXCHANGE: {
                ConfigKey.EXCHANGE_FORMAT: ExchangeFormat.PYTORCH,
                ConfigKey.SERVER_EXPECTED_FORMAT: ExchangeFormat.RAW,
                ConfigKey.TRAIN_TASK_NAME: "train",
                ConfigKey.EVAL_TASK_NAME: "validate",
                ConfigKey.SUBMIT_MODEL_TASK_NAME: "submit_model",
            }
        }
    )

    from_converter, to_converter = create_default_params_converters(
        server_expected_format=config.get_server_expected_format(),
        params_exchange_format=config.get_exchange_format(),
        train_task_name=config.get_train_task(),
        eval_task_name=config.get_eval_task(),
        submit_model_task_name=config.get_submit_model_task(),
    )
    assert from_converter is None
    assert to_converter is None
