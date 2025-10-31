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

# Model Update Definitions (MUD) - defines common structures used for model update exchanges
from typing import Dict, Optional

from nvflare.apis.dxo import DXO
from nvflare.apis.shareable import Shareable
from nvflare.fuel.utils.validation_utils import check_non_negative_int, check_object_type, check_positive_int, check_str


class PropKey:
    MODEL_VERSION = "model_version"
    MODEL = "model"
    DEVICE_SELECTION_VERSION = "device_selection_version"
    DEVICE_SELECTION = "device_selection"
    SELECTION_ID = "selection_id"
    DEVICE_ID = "device_id"
    CLIENT_NAME = "client_name"
    DEVICE_PROPS = "device_props"
    MODEL_UPDATES = "model_updates"
    DEVICE_LAST_ALIVE_TIME = "device_last_alive_time"
    DEVICES = "devices"


class BaseState:

    def __init__(
        self,
        model_version: int,
        model: DXO,
        device_selection_version: int,
        device_selection: Dict[str, int],  # device_id => selection_id
    ):
        """BaseState is the Base for on-device training

        The device_selection is a dict of device_id => selection_id.

        Device selection is maintained by the Server. It records the device_id paired with a selection_id, which is a non-zero integer.
        A device can train only once with the same selection_id.

        Our default way is to assign selection_id is that when a device is added to the selection, set its selection_id to
        the global model version assigned to it. This means that if the device is selected for
        training again, it must be assigned a different global model - it does not make much sense to select a device for training
        again with the same model, because it will lead to almost the same update twice (unless the data on the device is changed during the two selections, which is unlikely).

        The devices in the current selection could have different selection_ids because they are selected at different time points.

        Args:
            model_version: version of model. 0 means no model available.
            model: data of the model.
            device_selection_version: version of device selection. 0 means no devices selected.
            device_selection: selected devices.
        """
        check_non_negative_int("model_version", model_version)
        check_non_negative_int("device_selection_version", device_selection_version)

        if model:
            check_object_type("model", model, DXO)

        if model_version > 0 and not model:
            raise ValueError(f"invalid model_version {model_version} when model is not provided")

        if model_version == 0 and model:
            raise ValueError(f"invalid model_version {model_version} when model is provided")

        if device_selection_version > 0 and not device_selection:
            raise ValueError(
                f"invalid device_selection_version {device_selection_version} when device_selection is not provided"
            )

        if device_selection_version == 0 and device_selection:
            raise ValueError(
                f"invalid device_selection_version {device_selection_version} when device_selection is provided"
            )

        if device_selection_version > 0:
            check_object_type("device_selection", device_selection, dict)

        self.model_version = model_version
        self.model = model
        self.device_selection_version = device_selection_version
        self.device_selection = device_selection
        self.converted_models = {}  # platform => model

    def set_converted_model(self, model, platform: str):
        """Set the model that is converted from the original model for the specified platform.

        Args:
            model: the converted model
            platform: the platform that the converted model will be used for

        Returns: None

        """
        self.converted_models[platform] = model

    def get_converted_model(self, platform: str):
        """Get the model that is converted for the platform.

        Args:
            platform: the platform of the model

        Returns: converted model if available; None otherwise.

        """
        return self.converted_models.get(platform)

    def is_device_selected(self, device_id: str, selection_id: int) -> (bool, int):
        """Determine whether the device should be selected for training.

        Args:
            device_id: the ID of the device
            selection_id: current selection ID of the device

        Returns: tuple of (whether the device is selected, new selection id)

        """
        if not self.device_selection_version:
            # no device selection available
            return False, 0

        # get the "selection id" of this device in the current device_selection.
        sid = self.device_selection.get(device_id)
        if not sid:
            # device ID "*" means every device
            sid = self.device_selection.get("*")

        if not sid or sid == selection_id:
            # either this device is not in the selection, or its selection id has not changed
            # we do not allow the device to participate in training!
            return False, 0
        else:
            # the device is in the selection AND its selection ID is changed
            # we allow the device to train
            return True, sid

    def to_shareable(self) -> Shareable:
        """Convert to Shareable for communication.

        Returns: a Shareable object

        """
        result = Shareable()

        result.set_header(PropKey.MODEL_VERSION, self.model_version)
        if self.model_version > 0:
            result[PropKey.MODEL] = self.model

        result.set_header(PropKey.DEVICE_SELECTION_VERSION, self.device_selection_version)
        if self.device_selection_version > 0:
            result[PropKey.DEVICE_SELECTION] = self.device_selection

        return result

    @staticmethod
    def from_shareable(shareable: Shareable):
        """Convert the specified Shareable object to BaseState object.

        Args:
            shareable: the Shareable object to be converted.

        Returns: a BaseState object

        """
        model_version = shareable.get_header(PropKey.MODEL_VERSION)
        if not isinstance(model_version, int):
            raise ValueError(f"prop {PropKey.MODEL_VERSION} must be int but got {type(model_version)}")

        if model_version > 0:
            model = shareable.get(PropKey.MODEL)
            if not isinstance(model, DXO):
                raise ValueError(f"prop {PropKey.MODEL} must be DXO but got {type(model)}")
        else:
            model = None

        dev_selection_version = shareable.get_header(PropKey.DEVICE_SELECTION_VERSION)
        if not isinstance(dev_selection_version, int):
            raise ValueError(
                f"prop {PropKey.DEVICE_SELECTION_VERSION} must be int but got {type(dev_selection_version)}"
            )

        if dev_selection_version > 0:
            dev_selection = shareable.get(PropKey.DEVICE_SELECTION)
            if not isinstance(dev_selection, dict):
                raise ValueError(f"prop {PropKey.DEVICE_SELECTION} must be dict but got {type(dev_selection)}")
        else:
            dev_selection = {}

        return BaseState(model_version, model, dev_selection_version, dev_selection)


class Device:

    def __init__(self, device_id: str, client_name: str, last_alive_time: float, props: Dict = None):
        """Device object keeps device information to be communicated to the Server.

        Args:
            device_id: unique ID of the device.
            client_name: name of the leaf client that the device belongs to.
            last_alive_time: last time when the device interacted with the client.
            props: additional properties of the device.
        """
        check_str("device_id", device_id)
        check_str("client_name", client_name)

        if props:
            check_object_type("props", props, dict)

        self.device_id = device_id
        self.client_name = client_name
        self.last_alive_time = last_alive_time
        self.props = props

    def to_dict(self) -> dict:
        """Convert to dict representation, mainly for communication.

        Returns: a dict object

        """
        return {
            PropKey.DEVICE_ID: self.device_id,
            PropKey.CLIENT_NAME: self.client_name,
            PropKey.DEVICE_LAST_ALIVE_TIME: self.last_alive_time,
            PropKey.DEVICE_PROPS: self.props,
        }

    @staticmethod
    def from_dict(d: Dict):
        """Convert the specified dict object to Device object

        Args:
            d: the dict object to be converted.

        Returns: a Device object

        """
        return Device(
            device_id=d.get(PropKey.DEVICE_ID),
            client_name=d.get(PropKey.CLIENT_NAME),
            last_alive_time=d.get(PropKey.DEVICE_LAST_ALIVE_TIME),
            props=d.get(PropKey.DEVICE_PROPS),
        )


class ModelUpdate:

    def __init__(self, model_version: int, update: Shareable, devices: Dict[str, float]):
        """ModelUpdate specifies information of a model update.

        Args:
            model_version: version of the model.
            update: update to the model.
            devices: devices that contributed to this update. It is a dict of device_id => update timestamp,
                which specifies at what time the device made the contribution.
        """
        check_positive_int("model_version", model_version)
        check_object_type("devices", devices, dict)
        check_object_type("update", update, Shareable)

        if not devices:
            raise ValueError("devices for ModelUpdate must not be empty")

        self.model_version = model_version
        self.update = update
        self.devices = devices

    def to_dict(self) -> dict:
        """Convert to dict representation, mainly for communication.

        Returns: a dict object

        """
        return {
            PropKey.MODEL_VERSION: self.model_version,
            PropKey.MODEL: self.update,
            PropKey.DEVICES: self.devices,
        }

    @staticmethod
    def from_dict(d: Dict):
        """Convert the specified dict object to ModelUpdate object

        Args:
            d: the dict to be converted.

        Returns: a ModelUpdate object

        """
        return ModelUpdate(
            model_version=d.get(PropKey.MODEL_VERSION),
            update=d.get(PropKey.MODEL),
            devices=d.get(PropKey.DEVICES),
        )


class StateUpdateReport:

    def __init__(
        self,
        current_model_version: int,
        current_device_selection_version: int,
        model_updates: Optional[Dict[int, ModelUpdate]],  # model_version => ModelUpdate
        available_devices: Optional[Dict[str, Device]],  # device_id => Device
    ):
        """StateUpdateReport is sent from a client to its parent to report its state update.

        Args:
            current_model_version: version of the child's current model.
            current_device_selection_version: version of the child's current device_selection.
            model_updates: 0 or more model updates.
            available_devices: 0 or more available devices.

        Notes:
            Multiple versions of models could be in training.
        """
        check_non_negative_int("current_model_version", current_model_version)
        check_non_negative_int("current_device_selection_version", current_device_selection_version)

        if model_updates:
            check_object_type("model_updates", model_updates, dict)

        if available_devices:
            check_object_type("available_devices", available_devices, dict)

        self.current_model_version = current_model_version
        self.current_device_selection_version = current_device_selection_version
        self.model_updates = model_updates
        self.available_devices = available_devices

    def to_shareable(self) -> Shareable:
        """Convert to Shareable object, mainly for communication.

        Returns: a Shareable object

        """
        s = Shareable()
        s.set_header(PropKey.MODEL_VERSION, self.current_model_version)
        s.set_header(PropKey.DEVICE_SELECTION_VERSION, self.current_device_selection_version)

        if self.model_updates:
            s[PropKey.MODEL_UPDATES] = {k: v.to_dict() for k, v in self.model_updates.items()}

        if self.available_devices:
            s[PropKey.DEVICES] = {k: v.to_dict() for k, v in self.available_devices.items()}

        return s

    @staticmethod
    def from_shareable(s: Shareable):
        model_version = s.get_header(PropKey.MODEL_VERSION)
        device_selection_version = s.get_header(PropKey.DEVICE_SELECTION_VERSION)

        mu = s.get(PropKey.MODEL_UPDATES)
        if mu:
            if not isinstance(mu, dict):
                raise ValueError(f"prop {PropKey.MODEL_UPDATES} must be dict but got {type(mu)}")
            model_updates = {k: ModelUpdate.from_dict(v) for k, v in mu.items()}
        else:
            model_updates = None

        devs = s.get(PropKey.DEVICES)
        if devs:
            if not isinstance(devs, dict):
                raise ValueError(f"prop {PropKey.DEVICES} must be dict but got {type(devs)}")
            available_devices = {k: Device.from_dict(v) for k, v in devs.items()}
        else:
            available_devices = {}

        return StateUpdateReport(
            current_model_version=model_version,
            current_device_selection_version=device_selection_version,
            model_updates=model_updates,
            available_devices=available_devices,
        )


class StateUpdateReply:

    def __init__(
        self,
        model_version: int,
        model: DXO,
        device_selection_version: int,
        device_selection: Optional[Dict[str, int]],
    ):
        """StateUpdateReply is the reply to the child's StateUpdateReport.
        The child processes StateUpdateReply to adjust its Base State.

        Args:
            model_version:
            model:
            device_selection_version:
            device_selection:
        """
        check_non_negative_int("model_version", model_version)
        check_non_negative_int("device_selection_version", device_selection_version)

        if device_selection:
            check_object_type("device_selection", device_selection, dict)

        if model:
            check_object_type("model", model, DXO)

        self.model_version = model_version
        self.model = model
        self.device_selection_version = device_selection_version
        self.device_selection = device_selection

    def to_shareable(self) -> Shareable:
        """Convert to Shareable object, mainly for communication.

        Returns: a Shareable object

        """
        result = Shareable()

        result.set_header(PropKey.MODEL_VERSION, self.model_version)
        if self.model:
            result[PropKey.MODEL] = self.model

        result.set_header(PropKey.DEVICE_SELECTION_VERSION, self.device_selection_version)
        if self.device_selection:
            result[PropKey.DEVICE_SELECTION] = self.device_selection

        return result

    @staticmethod
    def from_shareable(shareable: Shareable):
        """Convert the specified Shareable object to StateUpdateReply object

        Args:
            shareable: the object to be converted

        Returns: a StateUpdateReply object

        """
        model_version = shareable.get_header(PropKey.MODEL_VERSION)
        model = shareable.get(PropKey.MODEL)
        dev_selection_version = shareable.get_header(PropKey.DEVICE_SELECTION_VERSION)
        dev_selection = shareable.get(PropKey.DEVICE_SELECTION)
        return StateUpdateReply(model_version, model, dev_selection_version, dev_selection)
