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
import os
from typing import Dict, Optional

from nvflare.fuel.utils.config_factory import ConfigFactory


class ExchangeFormat:
    RAW = "raw"
    PYTORCH = "pytorch"
    NUMPY = "numpy"


class TransferType:
    FULL = "FULL"
    DIFF = "DIFF"


class ConfigKey:
    EXCHANGE_FORMAT = "exchange_format"
    TRANSFER_TYPE = "transfer_type"
    TRAIN_WITH_EVAL = "train_with_eval"
    TRAIN_TASK_NAME = "train_task_name"
    EVAL_TASK_NAME = "eval_task_name"
    SUBMIT_MODEL_TASK_NAME = "submit_model_task_name"
    PIPE_CHANNEL_NAME = "pipe_channel_name"
    PIPE = "pipe"
    CLASS_NAME = "CLASS_NAME"
    ARG = "ARG"
    TASK_NAME = "TASK_NAME"
    TASK_EXCHANGE = "TASK_EXCHANGE"
    METRICS_EXCHANGE = "METRICS_EXCHANGE"
    HEARTBEAT_TIMEOUT = "HEARTBEAT_TIMEOUT"


class ClientConfig:
    """Config class used in `nvflare.client` module.

    Note:
        The config has the following keys:

        .. code-block::

            EXCHANGE_FORMAT: Format to exchange, pytorch, raw, or numpy
            TRANSFER_TYPE: Either FULL or DIFF (means difference)
            TRAIN_WITH_EVAL: Whether train task needs to also do evaluation
            TRAIN_TASK_NAME: Name of the train task
            EVAL_TASK_NAME: Name of the evaluate task
            SUBMIT_MODEL_TASK_NAME: Name of the submit_model task
            PIPE_CHANNEL_NAME: Channel name of the pipe
            PIPE: pipe section
            CLASS_NAME: Class name
            ARG: Arguments
            SITE_NAME: Site name
            JOB_ID: Job id
            TASK_EXCHANGE: TASK_EXCHANGE section
            METRICS_EXCHANGE: METRICS_EXCHANGE section

    Example:
        The content of config looks like:

        .. code-block:: json

            {
              "METRICS_EXCHANGE": {
                "pipe_channel_name": "metric",
                "pipe": {
                  "CLASS_NAME": "nvflare.fuel.utils.pipe.cell_pipe.CellPipe",
                  "ARG": {
                    "mode": "ACTIVE",
                    "site_name": "site-1",
                    "token": "simulate_job",
                    "root_url": "tcp://0:51893",
                    "secure_mode": false,
                    "workspace_dir": "xxx"
                  }
                }
              },
              "SITE_NAME": "site-1",
              "JOB_ID": "simulate_job",
              "TASK_EXCHANGE": {
                "train_with_eval": true,
                "exchange_format": "numpy",
                "transfer_type": "DIFF",
                "train_task_name": "train",
                "eval_task_name": "validate",
                "submit_model_task_name": "submit_model",
                "pipe_channel_name": "task",
                "pipe": {
                  "CLASS_NAME": "nvflare.fuel.utils.pipe.cell_pipe.CellPipe",
                  "ARG": {
                    "mode": "ACTIVE",
                    "site_name": "site-1",
                    "token": "simulate_job",
                    "root_url": "tcp://0:51893",
                    "secure_mode": false,
                    "workspace_dir": "xxx"
                  }
                }
              }
            }

    """

    def __init__(self, config: Optional[Dict] = None):
        if config is None:
            config = {}
        self.config = config

    def get_config(self) -> Dict:
        return self.config

    def get_pipe_channel_name(self, section: str) -> str:
        return self.config[section][ConfigKey.PIPE_CHANNEL_NAME]

    def get_pipe_args(self, section: str) -> dict:
        return self.config[section][ConfigKey.PIPE][ConfigKey.ARG]

    def get_pipe_class(self, section: str) -> str:
        return self.config[section][ConfigKey.PIPE][ConfigKey.CLASS_NAME]

    def get_exchange_format(self) -> str:
        return self.config.get(ConfigKey.TASK_EXCHANGE, {}).get(ConfigKey.EXCHANGE_FORMAT, "")

    def get_transfer_type(self) -> str:
        return self.config.get(ConfigKey.TASK_EXCHANGE, {}).get(ConfigKey.TRANSFER_TYPE, "FULL")

    def get_train_task(self):
        return self.config.get(ConfigKey.TASK_EXCHANGE, {}).get(ConfigKey.TRAIN_TASK_NAME, "")

    def get_eval_task(self):
        return self.config.get(ConfigKey.TASK_EXCHANGE, {}).get(ConfigKey.EVAL_TASK_NAME, "")

    def get_submit_model_task(self):
        return self.config.get(ConfigKey.TASK_EXCHANGE, {}).get(ConfigKey.SUBMIT_MODEL_TASK_NAME, "")

    def get_heartbeat_timeout(self):
        # TODO decouple task and metric heartbeat timeouts
        return self.config.get(ConfigKey.TASK_EXCHANGE, {}).get(
            ConfigKey.HEARTBEAT_TIMEOUT,
            self.config.get(ConfigKey.METRICS_EXCHANGE, {}).get(ConfigKey.HEARTBEAT_TIMEOUT, 60),
        )

    def to_json(self, config_file: str):
        with open(config_file, "w") as f:
            json.dump(self.config, f, indent=2)


def from_file(config_file: str):
    config = ConfigFactory.load_config(config_file)
    if config is None:
        raise RuntimeError(f"Load config file {config_file} failed")

    return ClientConfig(config=config.to_dict())


def write_config_to_file(config_data: dict, config_file_path: str):
    """Writes client api config file.

    Args:
        config_data (dict): data to be updated.
        config_file_path (str): filepath to write.
    """
    if os.path.exists(config_file_path):
        client_config = from_file(config_file=config_file_path)
    else:
        client_config = ClientConfig()
    configuration = client_config.config
    configuration.update(config_data)
    client_config.to_json(config_file_path)
