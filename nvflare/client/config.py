# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

from typing import Dict

from nvflare.app_common.abstract.fl_model import ParamsType

from .constants import ModelExchangeFormat


class ConfigKey:
    EXCHANGE_PATH = "exchange_path"
    EXCHANGE_FORMAT = "exchange_format"
    MODEL_CONVERTER = "model_converter"
    PARAMS_TYPE = "params_type"


class ClientConfig:
    """Config class used in nvflare.client module.

    Example:
        {
            "exchange_path": "./",
            "exchange_format": "pytorch",
            "params_type": "DIFF"
        }
    """

    def __init__(self, config: Dict):
        for required_key in (ConfigKey.EXCHANGE_PATH, ConfigKey.EXCHANGE_FORMAT, ConfigKey.PARAMS_TYPE):
            if required_key not in config:
                raise RuntimeError(f"Missing required_key: {required_key} in config.")

        config[ConfigKey.PARAMS_TYPE] = ParamsType(config[ConfigKey.PARAMS_TYPE])
        config[ConfigKey.EXCHANGE_FORMAT] = ModelExchangeFormat(config[ConfigKey.EXCHANGE_FORMAT])
        self.config = config

    def get_config(self):
        return self.config

    def get_exchange_path(self):
        return self.config[ConfigKey.EXCHANGE_PATH]

    def get_exchange_format(self) -> ModelExchangeFormat:
        return self.config[ConfigKey.EXCHANGE_FORMAT]

    def get_params_type(self, default=ParamsType.FULL) -> ParamsType:
        return self.config[ConfigKey.PARAMS_TYPE]
