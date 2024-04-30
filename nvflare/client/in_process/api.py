# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
import os
import time
from typing import Any, Dict, Optional

from nvflare.apis.analytix import AnalyticsDataType
from nvflare.apis.fl_constant import FLMetaKey
from nvflare.apis.shareable import Shareable
from nvflare.app_common.abstract.fl_model import FLModel, ParamsType
from nvflare.app_common.utils.fl_model_utils import FLModelUtils
from nvflare.client.api_spec import APISpec
from nvflare.client.config import ClientConfig, ConfigKey
from nvflare.client.constants import SYS_ATTRS
from nvflare.client.utils import DIFF_FUNCS
from nvflare.fuel.data_event.data_bus import DataBus
from nvflare.fuel.data_event.event_manager import EventManager

TOPIC_LOG_DATA = "LOG_DATA"
TOPIC_STOP = "STOP"
TOPIC_ABORT = "ABORT"
TOPIC_LOCAL_RESULT = "LOCAL_RESULT"
TOPIC_GLOBAL_RESULT = "GLOBAL_RESULT"


class InProcessClientAPI(APISpec):
    def __init__(self, task_metadata: dict, result_check_interval: float = 2.0):
        """Initializes the InProcessClientAPI.

        Args:
            task_metadata (dict): task metadata, added to client_config.
            result_check_interval (float): how often to check if result is availabe.
        """
        self.data_bus = DataBus()
        self.data_bus.subscribe([TOPIC_GLOBAL_RESULT], self.__receive_callback)
        self.data_bus.subscribe([TOPIC_ABORT, TOPIC_STOP], self.__ask_to_abort)

        self.meta = task_metadata
        self.result_check_interval = result_check_interval

        self.fl_model = None
        self.sys_info = {}
        self.client_config: Optional[ClientConfig] = None
        self.logger = logging.getLogger(self.__class__.__name__)
        self.event_manager = EventManager(self.data_bus)
        self.abort_reason = ""
        self.stop_reason = ""
        self.abort = False
        self.stop = False
        self.rank = None

    def init(self, rank: Optional[str] = None, config: Optional[Dict] = None):
        """Initializes NVFlare Client API environment.

        Args:
            config (Union[str, Dict]): config dictionary.
            rank (str): local rank of the process.
                It is only useful when the training script has multiple worker processes. (for example multi GPU)
        """

        self.rank = rank
        if rank is None:
            self.rank = os.environ.get("RANK", "0")

        config = {} if config is None else config
        self.prepare_client_config(config)

        for k, v in self.client_config.config.items():
            if k in SYS_ATTRS:
                self.sys_info[k] = v

    def prepare_client_config(self, config):
        if isinstance(config, dict):
            client_config = ClientConfig(config=config)
        else:
            raise ValueError(f"config should be a dictionary, but got {type(config)}")

        if client_config.config:
            client_config.config.update(self.meta)
        else:
            client_config.config = self.meta
        self.client_config = client_config

    def set_meta(self, meta: dict):
        self.meta = meta

    def receive(self, timeout: Optional[float] = None) -> Optional[FLModel]:
        if self.fl_model:
            return self.fl_model

        while True:
            if not self.__continue_job():
                break

            if self.fl_model is None:
                self.logger.debug(f"no result global message available, sleep {self.result_check_interval} sec")
                time.sleep(self.result_check_interval)
            else:
                break

        return self.fl_model

    def send(self, model: FLModel, clear_cache: bool = True) -> None:
        if self.__continue_job():
            self.logger.info("send local model back to peer ")

        if self.client_config.get_transfer_type() == "DIFF":
            model = self._prepare_param_diff(model)

        shareable = FLModelUtils.to_shareable(model)
        self.event_manager.fire_event(TOPIC_LOCAL_RESULT, shareable)

        if clear_cache:
            self.fl_model = None

    def system_info(self) -> Dict:
        return self.sys_info

    def get_config(self) -> Dict:
        return self.client_config.get_config()

    def get_job_id(self) -> str:
        return self.meta[FLMetaKey.JOB_ID]

    def get_site_name(self) -> str:
        return self.meta[FLMetaKey.SITE_NAME]

    def get_task_name(self) -> str:
        if self.rank != "0":
            raise RuntimeError("only rank 0 can call get_task_name!")

        return self.meta[ConfigKey.TASK_NAME]

    def is_running(self) -> bool:
        if not self.__continue_job():
            return False
        else:
            self.receive()

        return self.fl_model is not None

    def is_train(self) -> bool:
        if self.rank != "0":
            raise RuntimeError("only rank 0 can call is_train!")
        return self.meta.get(ConfigKey.TASK_NAME) == self.client_config.get_train_task()

    def is_evaluate(self) -> bool:
        if self.rank != "0":
            raise RuntimeError("only rank 0 can call is_evaluate!")
        return self.meta.get(ConfigKey.TASK_NAME) == self.client_config.get_eval_task()

    def is_submit_model(self) -> bool:
        if self.rank != "0":
            raise RuntimeError("only rank 0 can call is_submit_model!")
        return self.meta.get(ConfigKey.TASK_NAME) == self.client_config.get_submit_model_task()

    def log(self, key: str, value: Any, data_type: AnalyticsDataType, **kwargs):
        if self.rank != "0":
            raise RuntimeError("only rank 0 can call log!")
        msg = dict(key=key, value=value, data_type=data_type, **kwargs)
        self.event_manager.fire_event(TOPIC_LOG_DATA, msg)

    def clear(self):
        self.fl_model = None

    def _prepare_param_diff(self, model: FLModel) -> FLModel:
        exchange_format = self.client_config.get_exchange_format()
        diff_func = DIFF_FUNCS.get(exchange_format, None)

        if diff_func is None:
            raise RuntimeError(f"no default params diff function for {exchange_format}")
        elif self.fl_model is None:
            raise RuntimeError("no received model")
        elif self.fl_model.params is not None:
            if model.params_type == ParamsType.FULL:
                try:
                    model.params = diff_func(original=self.fl_model.params, new=model.params)
                    model.params_type = ParamsType.DIFF
                except Exception as e:
                    raise RuntimeError(f"params diff function failed: {e}")

        if model.params is None and model.metrics is None:
            raise RuntimeError("the model to send does not have either params or metrics")

        return model

    def __receive_callback(self, topic, data, databus):

        if topic == TOPIC_GLOBAL_RESULT and not isinstance(data, Shareable):
            raise ValueError(f"expecting a Shareable, but got '{type(data)}'")

        fl_model = FLModelUtils.from_shareable(data)
        self.fl_model = fl_model

    def __ask_to_abort(self, topic, msg, databus):
        if topic == TOPIC_ABORT:
            self.abort = True
            self.abort_reason = msg
            self.logger.error(f"ask to abort job: reason: {msg}")
        elif topic == TOPIC_STOP:
            self.stop = True
            self.stop_reason = msg
            self.logger.warning(f"ask to stop job: reason: {msg}")

    def __continue_job(self) -> bool:
        if self.abort:
            raise RuntimeError(f"request to abort the job for reason {self.abort_reason}")
        if self.stop:
            self.logger.warning(f"request to stop the job for reason {self.stop_reason}")
            self.fl_model = None
            return False

        return True
