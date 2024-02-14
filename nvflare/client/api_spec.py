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

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from nvflare.apis.analytix import AnalyticsDataType
from nvflare.app_common.abstract.fl_model import FLModel

CLIENT_API_KEY = "CLIENT_API"
CLIENT_API_TYPE_KEY = "CLIENT_API_TYPE"


class APISpec(ABC):
    @abstractmethod
    def init(self, rank: Optional[str] = None):
        pass

    @abstractmethod
    def receive(self, timeout: Optional[float] = None) -> Optional[FLModel]:
        pass

    @abstractmethod
    def send(self, model: FLModel, clear_cache: bool = True) -> None:
        pass

    @abstractmethod
    def system_info(self) -> Dict:
        pass

    @abstractmethod
    def get_config(self) -> Dict:
        pass

    @abstractmethod
    def get_job_id(self) -> str:
        pass

    @abstractmethod
    def get_site_name(self) -> str:
        pass

    @abstractmethod
    def get_task_name(self) -> str:
        pass

    @abstractmethod
    def is_running(self) -> bool:
        pass

    @abstractmethod
    def is_train(self) -> bool:
        pass

    @abstractmethod
    def is_evaluate(self) -> bool:
        pass

    @abstractmethod
    def is_submit_model(self) -> bool:
        pass

    @abstractmethod
    def log(self, key: str, value: Any, data_type: AnalyticsDataType, **kwargs):
        pass

    @abstractmethod
    def clear(self):
        pass
