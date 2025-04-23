# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
import enum
import uuid
from abc import ABC, abstractmethod

from nvflare.edge.web.models.device_info import DeviceInfo
from nvflare.edge.web.models.task_response import TaskResponse
from nvflare.edge.web.models.user_info import UserInfo
from nvflare.fuel.utils.log_utils import get_obj_logger


class DeviceState(str, enum.Enum):
    IDLE = "idle"
    LEARNING = "learning"
    DONE = "done"


class SimulatedDevice(ABC):

    def __init__(self, device_id: str):
        self.device_id = device_id
        self.idle = True
        self.cookie = None
        self.state: DeviceState = DeviceState.IDLE
        self.logger = get_obj_logger(self)

    def get_device_info(self):
        return DeviceInfo(
            device_id=self.device_id,
            app_name="device_runner",
            app_version="1.0",
            platform="flare",
        )

    def get_user_info(self):
        return UserInfo(user_id=self.device_id)

    @abstractmethod
    def do_task(self, data: TaskResponse) -> dict:
        pass


class DeviceFactory(ABC):

    def make_device(self) -> SimulatedDevice:
        return SimulatedDevice(str(uuid.uuid4()))

    def shutdown(self):
        pass
