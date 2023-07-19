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

import logging
from typing import Optional

from nvflare.app_common.abstract.fl_model import ParamsType
from nvflare.fuel.utils.config import Config
from nvflare.fuel.utils.config_service import ConfigService

from .constants import ModelExchangeFormat


class ConfigKey:
    EXCHANGE_PATH = "exchange_path"
    EXCHANGE_FORMAT = "exchange_format"
    MODEL_CONVERTER = "model_converter"
    PARAMS_TYPE = "params_type"
    PARAMS_DIFF_FUNC = "params_diff_func"


class ClientConfig:
    """Config class used in nvflare.client module.

    Example:
        {
            "exchange_path": "./",
            "exchange_format": "pytorch",
            "params_type": "DIFF",
            "params_diff_func": "numerical_params_diff"
        }

    Note that params_diff_func signature:
        params_diff_func(original model, new model) -> model difference
    """

    def __init__(self, config_file: str):
        self.logger = logging.getLogger(self.__class__.__name__)
        ConfigService.initialize({}, ["."])
        config: Optional[Config] = ConfigService.load_configuration(file_basename=config_file)
        self.config = None if config is None else config.to_dict()
        if self.get_params_type() == ParamsType.FULL and self.get_params_diff_func() is not None:
            raise ValueError(
                f"{ConfigKey.PARAMS_TYPE} is {ParamsType.FULL}, should not provide {ConfigKey.PARAMS_DIFF_FUNC}."
            )

    def get_config(self):
        return self.config

    def get_exchange_path(self, default="./"):
        return ConfigService.get_str_var(ConfigKey.EXCHANGE_PATH, self.config, default=default)

    def get_exchange_format(self, default=ModelExchangeFormat.PYTORCH) -> ModelExchangeFormat:
        return ModelExchangeFormat(ConfigService.get_str_var(ConfigKey.EXCHANGE_FORMAT, self.config, default=default))

    def get_params_type(self, default=ParamsType.FULL) -> ParamsType:
        return ParamsType(ConfigService.get_str_var(ConfigKey.PARAMS_TYPE, self.config, default=default))

    def get_params_diff_func(self, default=None):
        return ConfigService.get_str_var(ConfigKey.PARAMS_DIFF_FUNC, self.config, default=default)
