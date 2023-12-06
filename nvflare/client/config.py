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

from nvflare.fuel.utils.config_factory import ConfigFactory


class ExchangeFormat:
    RAW = "raw"
    PYTORCH = "pytorch"
    NUMPY = "numpy"


class TransferType(str, Enum):
    FULL = "FULL"
    DIFF = "DIFF"


class ConfigKey:
    EXCHANGE_FORMAT = "exchange_format"
    TRANSFER_TYPE = "transfer_type"
    GLOBAL_EVAL = "global_eval"
    TRAIN_WITH_EVAL = "train_with_eval"
    TRAIN_TASK_NAME = "train_task_name"
    EVAL_TASK_NAME = "eval_task_name"
    SUBMIT_MODEL_TASK_NAME = "submit_model_task_name"
    PIPE_CHANNEL_NAME = "pipe_name"
    PIPE_CLASS = "pipe_class"
    PIPE_ARGS = "pipe_args"
    SITE_NAME = "SITE_NAME"
    JOB_ID = "JOB_ID"
    TASK_EXCHANGE = "TASK_EXCHANGE"
    METRICS_EXCHANGE = "METRICS_EXCHANGE"


class ClientConfig:
    """Config class used in nvflare.client module.

    Example:
        {
            "site_name": "site-1",
            "job_id": xxxxx,
            "task_exchange": {
                "exchange_format": "pytorch",
                "transfer_type": "FULL",
                "pipe_class": "FilePipe"
            },
            "metrics_exchange": {
                "pipe_class": "CellPipe"
            }
        }
    """

    def __init__(self, config: Optional[Dict] = None):
        if config is None:
            config = {}
        self.config = config

    def get_config(self):
        return self.config

    def get_pipe_channel_name(self, section: str) -> str:
        return self.config[section][ConfigKey.PIPE_CHANNEL_NAME]

    def get_pipe_args(self, section: str) -> dict:
        return self.config[section][ConfigKey.PIPE_ARGS]

    def get_pipe_class(self, section: str) -> str:
        return self.config[section][ConfigKey.PIPE_CLASS]

    def get_exchange_format(self) -> ExchangeFormat:
        return self.config[ConfigKey.TASK_EXCHANGE][ConfigKey.EXCHANGE_FORMAT]

    def get_transfer_type(self):
        return self.config.get(ConfigKey.TASK_EXCHANGE, {}).get(ConfigKey.TRANSFER_TYPE, "FULL")

    def get_train_task(self):
        return self.config[ConfigKey.TASK_EXCHANGE][ConfigKey.TRAIN_TASK_NAME]

    def get_eval_task(self):
        return self.config[ConfigKey.TASK_EXCHANGE][ConfigKey.EVAL_TASK_NAME]

    def get_submit_model_task(self):
        return self.config[ConfigKey.TASK_EXCHANGE][ConfigKey.SUBMIT_MODEL_TASK_NAME]

    def to_json(self, config_file: str):
        with open(config_file, "w") as f:
            json.dump(self.config, f)


def from_file(config_file: str):
    config = ConfigFactory.load_config(config_file)
    if config is None:
        raise RuntimeError(f"Load config file {config_file} failed")

    return ClientConfig(config=config.to_dict())
