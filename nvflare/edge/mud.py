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

# Model Update Definitions (MUD) - defines common structures used for model updates
from typing import Dict, List, Union

from nvflare.apis.shareable import Shareable


class PropKey:
    MODEL_VERSION = "model_version"
    MODEL = "model"
    DEVICE_LIST_VERSION = "device_list_version"
    DEVICE_LIST = "device_list"
    DEVICE_ID = "device_id"
    CLIENT_NAME = "client_name"
    DEVICE_PROPS = "device_props"
    MODEL_UPDATES = "model_updates"
    DEVICE_LAST_ALIVE_TIME = "device_last_alive_time"


class BaseState:

    def __init__(self, model_version: int, model: Shareable, device_list_version: int, device_list: List[str]):
        """BaseState is the Base for on-device training

        Args:
            model_version:
            model:
            device_list_version:
            device_list:
        """
        if model_version > 0 and not model:
            raise ValueError(f"invalid model version {model_version} when no model is provided")

        if model_version <= 0 and model:
            raise ValueError(f"invalid model version {model_version} when model is provided")

        self.model_version = model_version
        self.model = model
        self.device_list_version = device_list_version
        self.device_list = device_list

    def is_device_in_list(self, device_id: str) -> bool:
        if not self.device_list_version:
            return False

        if "*" in self.device_list:
            # any device
            return True

        return device_id in self.device_list

    def to_shareable(self) -> Shareable:
        result = Shareable()
        result.set_header(PropKey.MODEL_VERSION, self.model_version)
        result[PropKey.MODEL] = self.model
        result.set_header(PropKey.DEVICE_LIST_VERSION, self.device_list_version)
        result[PropKey.DEVICE_LIST] = self.device_list
        return result

    @staticmethod
    def from_shareable(shareable: Shareable):
        model_version = shareable.get_header(PropKey.MODEL_VERSION)
        model = shareable.get(PropKey.MODEL)
        dev_list_version = shareable.get_header(PropKey.DEVICE_LIST_VERSION)
        dev_list = shareable.get(PropKey.DEVICE_LIST)
        return BaseState(model_version, model, dev_list_version, dev_list)


class DeviceInfo:

    def __init__(self, device_id: str, client_name: str, last_alive_time: float, props: Dict = None):
        self.device_id = device_id
        self.client_name = client_name
        self.last_alive_time = last_alive_time
        self.props = props

    def to_dict(self):
        return {
            PropKey.DEVICE_ID: self.device_id,
            PropKey.CLIENT_NAME: self.client_name,
            PropKey.DEVICE_LAST_ALIVE_TIME: self.last_alive_time,
            PropKey.DEVICE_PROPS: self.props,
        }

    @staticmethod
    def from_dict(d: Dict):
        return DeviceInfo(
            device_id=d.get(PropKey.DEVICE_ID),
            client_name=d.get(PropKey.CLIENT_NAME),
            last_alive_time=d.get(PropKey.DEVICE_LAST_ALIVE_TIME),
            props=d.get(PropKey.DEVICE_PROPS),
        )


class ModelUpdate:

    def __init__(self, model_version: int, update: Shareable, devices: Dict[str, float]):
        self.model_version = model_version
        self.update = update
        self.devices = devices

    def to_dict(self):
        return {
            PropKey.MODEL_VERSION: self.model_version,
            PropKey.MODEL: self.update,
            PropKey.DEVICE_LIST: self.devices,
        }

    @staticmethod
    def from_dict(d: Dict):
        return ModelUpdate(
            model_version=d.get(PropKey.MODEL_VERSION), update=d.get(PropKey.MODEL), devices=d.get(PropKey.DEVICE_LIST)
        )


class StateUpdateReport:

    def __init__(
        self,
        current_model_version: Union[int, None],
        current_device_list_version: Union[int, None],
        model_updates: Union[None, List[ModelUpdate]],
        devices: Union[List[DeviceInfo], None],
    ):
        """StateUpdateReport is sent from a client to its parent

        Args:
            current_model_version:
            current_device_list_version:
            model_updates:
            devices:
        """
        self.current_model_version = current_model_version
        self.current_device_list_version = current_device_list_version
        self.model_updates = model_updates
        self.devices = devices

    def to_shareable(self) -> Shareable:
        s = Shareable()
        s.set_header(PropKey.MODEL_VERSION, self.current_model_version)
        s.set_header(PropKey.DEVICE_LIST_VERSION, self.current_device_list_version)
        s[PropKey.MODEL_UPDATES] = [u.to_dict() for u in self.model_updates]
        s[PropKey.DEVICE_LIST] = [d.to_dict() for d in self.devices]
        return s

    @staticmethod
    def from_shareable(s: Shareable):
        return StateUpdateReport(
            current_model_version=s.get_header(PropKey.MODEL_VERSION),
            current_device_list_version=s.get_header(PropKey.DEVICE_LIST_VERSION),
            model_updates=[ModelUpdate.from_dict(d) for d in s[PropKey.MODEL_UPDATES]],
            devices=[DeviceInfo.from_dict(d) for d in s[PropKey.DEVICE_LIST]],
        )


class StateUpdateReply:

    def __init__(self, model_version: int, model: Shareable, device_list_version: int, device_list: List[str]):
        """StateUpdateReply is the reply to the child's StateUpdateReport.
        The child processes StateUpdateReply to adjust its Base State.

        Args:
            model_version:
            model:
            device_list_version:
            device_list:
        """
        self.model_version = model_version
        self.model = model
        self.device_list_version = device_list_version
        self.device_list = device_list

    def to_shareable(self) -> Shareable:
        result = Shareable()
        result.set_header(PropKey.MODEL_VERSION, self.model_version)
        result[PropKey.MODEL] = self.model
        result.set_header(PropKey.DEVICE_LIST_VERSION, self.device_list_version)
        result[PropKey.DEVICE_LIST] = self.device_list
        return result

    @staticmethod
    def from_shareable(shareable: Shareable):
        model_version = shareable.get_header(PropKey.MODEL_VERSION)
        model = shareable.get(PropKey.MODEL)
        dev_list_version = shareable.get_header(PropKey.DEVICE_LIST_VERSION)
        dev_list = shareable.get(PropKey.DEVICE_LIST)
        return StateUpdateReply(model_version, model, dev_list_version, dev_list)
