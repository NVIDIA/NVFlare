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

from enum import Enum
from typing import Dict, Optional


class ExchangeFormat(str, Enum):
    RAW = "raw"
    PYTORCH = "pytorch"
    NUMPY = "numpy"
    KERAS_LAYER_WEIGHTS = "keras_layer_weights"


def normalize_exchange_format(value, name: str) -> ExchangeFormat:
    """Return a validated ExchangeFormat value for a config declaration."""

    try:
        return ExchangeFormat(value)
    except (TypeError, ValueError) as e:
        raise ValueError(f"invalid {name} {value!r}: must be one of {list(ExchangeFormat)}") from e


class TransferType(str, Enum):
    FULL = "FULL"
    DIFF = "DIFF"


class ConfigKey:
    EXCHANGE_FORMAT = "exchange_format"
    SERVER_EXPECTED_FORMAT = "server_expected_format"
    TRANSFER_TYPE = "transfer_type"
    TRAIN_WITH_EVAL = "train_with_eval"
    TRAIN_TASK_NAME = "train_task_name"
    EVAL_TASK_NAME = "eval_task_name"
    SUBMIT_MODEL_TASK_NAME = "submit_model_task_name"
    TASK_NAME = "TASK_NAME"
    TASK_EXCHANGE = "TASK_EXCHANGE"
    MEMORY_GC_ROUNDS = "memory_gc_rounds"
    CUDA_EMPTY_CACHE = "cuda_empty_cache"
    LAUNCH_ONCE = "launch_once"


class ClientConfig:
    """Task metadata used by the in-process Client API."""

    def __init__(self, config: Optional[Dict] = None):
        if config is None:
            config = {}
        self.config = config

    def get_config(self) -> Dict:
        return self.config

    def get_exchange_format(self) -> str:
        return self.config.get(ConfigKey.TASK_EXCHANGE, {}).get(ConfigKey.EXCHANGE_FORMAT, "")

    def get_server_expected_format(self) -> str:
        return self.config.get(ConfigKey.TASK_EXCHANGE, {}).get(ConfigKey.SERVER_EXPECTED_FORMAT, ExchangeFormat.NUMPY)

    def get_transfer_type(self) -> str:
        return self.config.get(ConfigKey.TASK_EXCHANGE, {}).get(ConfigKey.TRANSFER_TYPE, "FULL")

    def get_train_task(self):
        return self.config.get(ConfigKey.TASK_EXCHANGE, {}).get(ConfigKey.TRAIN_TASK_NAME, "")

    def get_eval_task(self):
        return self.config.get(ConfigKey.TASK_EXCHANGE, {}).get(ConfigKey.EVAL_TASK_NAME, "")

    def get_submit_model_task(self):
        return self.config.get(ConfigKey.TASK_EXCHANGE, {}).get(ConfigKey.SUBMIT_MODEL_TASK_NAME, "")
