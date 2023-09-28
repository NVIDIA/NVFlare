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

import copy
import os
import tempfile
import time
from typing import Dict, Optional

from nvflare.app_common.abstract.fl_model import FLModel, ParamsType
from nvflare.app_common.model_exchange.model_exchanger import ModelExchanger

from .config import ClientConfig, ConfigKey
from .constants import SYS_ATTRS
from .utils import DIFF_FUNCS


def get_data_file_name(config: dict):
    return os.path.abspath(
        os.path.join(
            config[ConfigKey.EXCHANGE_PATH], f"__nvclientapi_{config[ConfigKey.SITE_NAME]}_{config[ConfigKey.JOB_ID]}"
        )
    )


def save_data_atomic(data: int, data_file_path: str):
    temp_f = tempfile.NamedTemporaryFile(delete=False)

    try:
        with open(temp_f.name, "w") as f:
            f.write(str(data))
            f.flush()
            os.fsync(f.fileno())

        os.replace(temp_f.name, data_file_path)
    except Exception as e:
        print(f"XXX {e} XXX")
        os.remove(temp_f.name)


def read_data(data_file_path: str, read_timeout: float = 300.0):
    start_time = time.time()
    while True:
        if os.path.exists(data_file_path):
            with open(data_file_path, "r") as f:
                current_round = int(f.readlines()[0].strip())
                return current_round
        if time.time() - start_time > read_timeout:
            raise RuntimeError("Can't read data from rank 0.")
        time.sleep(1.0)


class ModelRegistry:
    """This class is used to remember attributes that need to share for a user code.

    For example, after "global_evaluate" we should remember the "metrics" value.
    And set that into the model that we want to submit after "train".

    For each user file:
        - we only need 1 model exchanger.
        - we only need to pull global model once

    """

    def __init__(self, config: ClientConfig, rank, model_exchanger: Optional[ModelExchanger] = None):
        self.model_exchanger = model_exchanger
        self.config = config

        self.cached_model: Optional[FLModel] = None
        self.cache_loaded = False
        self.metrics = None
        self.sys_info = {}
        for k, v in self.config.config.items():
            if k in SYS_ATTRS:
                self.sys_info[k] = v
        self.output_meta = {}
        self.rank = rank

    def receive(self):
        data_file_path = get_data_file_name(self.config.config)
        if not self.model_exchanger:
            current_round = read_data(data_file_path)
            self._set_model(FLModel(current_round=current_round))
        else:
            received_model = self.model_exchanger.receive_model()
            self._set_model(received_model)
            # write out only the current_round for other ranks
            save_data_atomic(received_model.current_round, data_file_path)

    def _set_model(self, model: FLModel):
        self.cached_model = model
        self.cache_loaded = True

    def get_model(self):
        if not self.cache_loaded:
            self.receive()
        return copy.deepcopy(self.cached_model)

    def get_sys_info(self) -> Dict:
        return self.sys_info

    def send(self, model: FLModel) -> None:
        if not self.model_exchanger:
            return None
        if self.config.get_transfer_type() == "DIFF":
            exchange_format = self.config.get_exchange_format()
            diff_func = DIFF_FUNCS.get(exchange_format, None)
            if diff_func is None:
                raise RuntimeError(f"no default params diff function for {exchange_format}")
            elif self.cached_model is None:
                raise RuntimeError("no received model")
            elif model.params is not None:
                if model.params_type == ParamsType.FULL:
                    try:
                        model.params = diff_func(original=self.cached_model.params, new=model.params)
                        model.params_type = ParamsType.DIFF
                    except Exception as e:
                        raise RuntimeError(f"params diff function failed: {e}")
            elif model.metrics is None:
                raise RuntimeError("the model to send does not have either params or metrics")
        self.model_exchanger.submit_model(model=model)
        data_file_path = get_data_file_name(self.config.config)
        if os.path.exists(data_file_path):
            os.remove(data_file_path)

    def clear(self):
        self.cached_model = None
        self.cache_loaded = False
        self.metrics = None

    def __str__(self):
        return f"{self.__class__.__name__}(config: {self.config.get_config()})"

    def __del__(self):
        if self.model_exchanger:
            self.model_exchanger.finalize(close_pipe=False)
