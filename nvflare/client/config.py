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
from typing import Dict, List, Optional

from nvflare.app_common.data_exchange.constants import ExchangeFormat
from nvflare.fuel.utils.config_factory import ConfigFactory


class TransferType(str, Enum):
    FULL = "FULL"
    DIFF = "DIFF"


class ConfigKey:
    EXCHANGE_PATH = "exchange_path"
    EXCHANGE_FORMAT = "exchange_format"
    TRANSFER_TYPE = "transfer_type"
    GLOBAL_EVAL = "global_eval"
    TRAIN_WITH_EVAL = "train_with_eval"
    TRAIN_TASK_NAME = "train_task_name"
    EVAL_TASK_NAME = "eval_task_name"
    SUBMIT_MODEL_TASK_NAME = "submit_model_task_name"
    PIPE_NAME = "pipe_name"
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
            self.config[ConfigKey.EXCHANGE_FORMAT] = ExchangeFormat(self.config[ConfigKey.EXCHANGE_FORMAT])

    def get_config(self):
        return self.config

    def get_exchange_path(self) -> str:
        return self.config[ConfigKey.EXCHANGE_PATH]

    def get_supported_topics(self) -> List[str]:
        return [
            self.config[k]
            for k in [ConfigKey.TRAIN_TASK_NAME, ConfigKey.EVAL_TASK_NAME, ConfigKey.SUBMIT_MODEL_TASK_NAME]
        ]

    def get_pipe_name(self) -> str:
        return self.config[ConfigKey.PIPE_NAME]

    def get_exchange_format(self) -> ExchangeFormat:
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
