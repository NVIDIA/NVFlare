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

import json
from enum import Enum
from typing import Dict, Optional

from nvflare.app_common.model_exchange.constants import ModelExchangeFormat
from nvflare.fuel.utils.config_factory import ConfigFactory


class TransferType(str, Enum):
    FULL = "FULL"
    DIFF = "DIFF"


class ConfigKey:
    EXCHANGE_PATH = "exchange_path"
    EXCHANGE_FORMAT = "exchange_format"
    TRANSFER_TYPE = "transfer_type"
    GLOBAL_EVAL = "global_eval"
    TRAINING = "training"
    LAUNCH_ONCE = "launch_once"
    TOTAL_ROUNDS = "total_rounds"
    SITE_NAME = "site_name"
    JOB_ID = "job_id"


class ClientConfig:
    """Config class used in nvflare.client module.

    Example:
        {
            "exchange_path": "./",
            "exchange_format": "pytorch",
            "transfer_type": "FULL"
        }
    """

    def __init__(self, config: Optional[Dict] = None):
        if config is None:
            config = {}
        self.config = config
        if ConfigKey.EXCHANGE_FORMAT in self.config:
            self.config[ConfigKey.EXCHANGE_FORMAT] = ModelExchangeFormat(self.config[ConfigKey.EXCHANGE_FORMAT])

    def get_config(self):
        return self.config

    def get_exchange_path(self):
        return self.config[ConfigKey.EXCHANGE_PATH]

    def get_exchange_format(self) -> ModelExchangeFormat:
        return self.config[ConfigKey.EXCHANGE_FORMAT]

    def get_transfer_type(self):
        return self.config.get(ConfigKey.TRANSFER_TYPE, "FULL")

    def to_json(self, config_file: str):
        with open(config_file, "w") as f:
            json.dump(self.config, f)


def from_file(config_file: str):
    config = ConfigFactory.load_config(config_file)
    if config is None:
        raise RuntimeError(f"Load config file {config} failed.")

    return ClientConfig(config=config.to_dict())
