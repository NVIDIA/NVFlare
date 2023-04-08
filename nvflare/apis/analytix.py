# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

from nvflare.apis.dxo import DXO, DataKind
from nvflare.app_common.tracking.tracker_types import TrackConst, Tracker

_DATA_TYPE_KEY = "analytics_data_type"
_KWARGS_KEY = "analytics_kwargs"


class AnalyticsDataType(Enum):
    SCALARS = "SCALARS"
    SCALAR = "SCALAR"
    IMAGE = "IMAGE"
    TEXT = "TEXT"
    LOG_RECORD = "LOG_RECORD"

    PARAMETER = "PARAMETER"
    PARAMETERS = "PARAMETERS"
    METRIC = "METRIC"
    METRICS = "METRICS"
    MODEL = "MODEL"

    #     # MLFLOW ONLY
    TAG = "TAG"
    TAGS = "TAGS"
    INIT_DATA = "INIT_DATA"


class AnalyticsData:
    def __init__(
        self,
        key: str,
        value,
        data_type: AnalyticsDataType,
        sender: Tracker = Tracker.TORCH_TB,
        kwargs: Optional[dict] = None,
        step: Optional[int] = None,
        path: Optional[str] = None,
    ):
        """This class defines AnalyticsData format.

        It is a wrapper to provide to/from DXO conversion.

        Args:
            key (str): tag name
            value: value
            data_type (AnalyticDataType): type of the analytic data.
            sender (Tracker): Syntax of Tracker such as Tensorboard or MLFLow
            kwargs (optional, dict): additional arguments to be passed.
        """
        self._validate_data_types(data_type, kwargs, key, value, step, path)
        self.tag = key
        self.value = value
        self.data_type = data_type
        self.kwargs = kwargs
        self.step = step
        self.path = path
        self.sender = sender

    def to_dxo(self):
        """Converts the AnalyticsData to DXO object.

        Returns:
            DXO object
        """

        data = {TrackConst.TRACK_KEY: self.tag, TrackConst.TRACK_VALUE: self.value}
        if self.step:
            data[TrackConst.GLOBAL_STEP_KEY] = self.step
        if self.path:
            data[TrackConst.PATH_KEY] = self.path
        if self.kwargs:
            data[TrackConst.KWARGS_KEY] = self.kwargs
        dxo = DXO(data_kind=DataKind.ANALYTIC, data=data)
        dxo.set_meta_prop(TrackConst.DATA_TYPE_KEY, self.data_type)
        dxo.set_meta_prop(TrackConst.TRACKER_KEY, self.sender)
        return dxo

    @classmethod
    def from_dxo(cls, dxo: DXO, receiver: Tracker = Tracker.TORCH_TB):
        """Generates the AnalyticsData from DXO object.

        Args:
            receiver: type experiment tacker, default to Tensorboard. TrackerType.TORCH_TB
            sender: type experiment tacker, default to Tensorboard. TrackerType.TORCH_TB
            dxo (DXO): The DXO object to convert.

        Returns:
            AnalyticsData object
        """
        if not isinstance(dxo, DXO):
            raise TypeError("expect dxo to be an instance of DXO, but got {}.".format(type(dxo)))

        if len(dxo.data) == 0:
            raise ValueError(
                "dxo does not have the correct format for AnalyticsData; expected dxo.data to be length > 0, but got 0"
            )
        data = dxo.data
        key = data[TrackConst.TRACK_KEY]
        value = data[TrackConst.TRACK_VALUE]
        step = data[TrackConst.GLOBAL_STEP_KEY] if TrackConst.GLOBAL_STEP_KEY in data else None
        path = data[TrackConst.PATH_KEY] if TrackConst.PATH_KEY in data else None
        kwargs = data[TrackConst.KWARGS_KEY] if TrackConst.KWARGS_KEY in data else None
        data_type = dxo.get_meta_prop(TrackConst.DATA_TYPE_KEY)
        sender = dxo.get_meta_prop(TrackConst.TRACKER_KEY)
        if sender is not None and sender != receiver:
            data_type = cls.convert_data_type(data_type, sender, receiver)
        return cls(key, value, data_type, sender, kwargs, step, path)

    def _validate_data_types(
        self,
        data_type: AnalyticsDataType,
        kwargs: Optional[Dict],
        key: str,
        value: any,
        step: Optional[int] = None,
        path: Optional[str] = None,
    ):
        if not isinstance(key, str):
            raise TypeError("expect tag to be an instance of str, but got {}.".format(type(key)))
        if not isinstance(data_type, AnalyticsDataType):
            raise TypeError(
                "expect data_type to be an instance of AnalyticsDataType, but got {}.".format(type(data_type))
            )
        if kwargs and not isinstance(kwargs, dict):
            raise TypeError("expect kwargs to be an instance of dict, but got {}.".format(type(kwargs)))
        if step:
            if not isinstance(step, int):
                raise TypeError("expect step to be an instance of int, but got {}.".format(type(step)))
            if step < 0:
                raise ValueError("expect step to be non-negative int, but got {}.".format(step))
        if path and not isinstance(path, str):
            raise TypeError("expect path to be an instance of str, but got {}.".format(type(step)))
        if data_type in [AnalyticsDataType.SCALAR, AnalyticsDataType.METRIC] and not isinstance(value, float):
            raise TypeError(f"expect '{key}' value to be an instance of float, but got '{type(value)}'.")
        elif data_type in [
            AnalyticsDataType.METRICS,
            AnalyticsDataType.PARAMETERS,
            AnalyticsDataType.SCALARS,
        ] and not isinstance(value, dict):
            raise TypeError(f"expect '{key}' value to be an instance of dict, but got '{type(value)}'.")
        elif data_type == AnalyticsDataType.TEXT and not isinstance(value, str):
            raise TypeError(f"expect '{key}' value to be an instance of str, but got '{type(value)}'.")
        elif data_type == AnalyticsDataType.TAGS and not isinstance(value, dict):
            raise TypeError(
                f"expect '{key}' data type expects value to be an instance of dict, but got '{type(value)}'"
            )

    @classmethod
    def convert_data_type(
        cls, sender_data_type: AnalyticsDataType, sender: Tracker, receiver: Tracker
    ) -> AnalyticsDataType:

        if sender == Tracker.TORCH_TB and receiver == Tracker.MLFLOW:
            if AnalyticsDataType.SCALAR == sender_data_type:
                return AnalyticsDataType.METRIC
            elif AnalyticsDataType.SCALARS == sender_data_type:
                return AnalyticsDataType.METRICS
            else:
                return sender_data_type

        if sender == Tracker.MLFLOW and receiver == Tracker.TORCH_TB:
            if AnalyticsDataType.PARAMETER == sender_data_type:
                return AnalyticsDataType.SCALAR
            elif AnalyticsDataType.PARAMETERS == sender_data_type:
                return AnalyticsDataType.SCALARS
            elif AnalyticsDataType.METRIC == sender_data_type:
                return AnalyticsDataType.SCALAR
            elif AnalyticsDataType.METRICS == sender_data_type:
                return AnalyticsDataType.SCALARS
            else:
                return sender_data_type
