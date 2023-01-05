# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
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
from typing import Optional

from nvflare.apis.dxo import DXO, DataKind
from nvflare.app_common.tracking.tracker_types import LogWriterName, TrackConst

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

    TAG = "TAG"
    TAGS = "TAGS"

    INIT = "INIT"


class AnalyticsData:
    def __init__(
        self, key: str, value, data_type: AnalyticsDataType, writer: LogWriterName = LogWriterName.TORCH_TB, **kwargs
    ):
        """This class defines AnalyticsData format.

        It is a wrapper to provide to/from DXO conversion.

        Args:
            key (str): tag name
            value: value
            data_type (AnalyticDataType): type of the analytic data.
            writer (LogWriterName): Syntax of Tracker such as Tensorboard or MLFLow
            kwargs (optional, dict): additional arguments to be passed.
        """
        self._validate_data_types(data_type, key, value, **kwargs)
        self.tag = key
        self.value = value
        self.data_type = data_type
        self.kwargs = kwargs
        self.writer = writer
        self.step = kwargs.get(TrackConst.GLOBAL_STEP_KEY, None)
        self.path = kwargs.get(TrackConst.PATH_KEY, None)

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
        dxo.set_meta_prop(TrackConst.WRITER_KEY, self.writer)
        return dxo

    @classmethod
    def from_dxo(cls, dxo: DXO, receiver: LogWriterName = LogWriterName.TORCH_TB):
        """Generates the AnalyticsData from DXO object.

        Args:
            receiver: type experiment tacker, default to Tensorboard. TrackerType.TORCH_TB
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
        kwargs = data[TrackConst.KWARGS_KEY] if TrackConst.KWARGS_KEY in data else None
        data_type = dxo.get_meta_prop(TrackConst.DATA_TYPE_KEY)
        writer = dxo.get_meta_prop(TrackConst.WRITER_KEY)
        if writer is not None and writer != receiver:
            data_type = cls.convert_data_type(data_type, writer, receiver)

        if not data_type:
            return None

        if not kwargs:
            return cls(key, value, data_type, writer)
        else:
            return cls(key, value, data_type, writer, **kwargs)

    def _validate_data_types(
        self,
        data_type: AnalyticsDataType,
        key: str,
        value: any,
        **kwargs,
    ):
        if not isinstance(key, str):
            raise TypeError("expect tag to be an instance of str, but got {}.".format(type(key)))
        if not isinstance(data_type, AnalyticsDataType):
            raise TypeError(
                "expect data_type to be an instance of AnalyticsDataType, but got {}.".format(type(data_type))
            )
        if kwargs and not isinstance(kwargs, dict):
            raise TypeError("expect kwargs to be an instance of dict, but got {}.".format(type(kwargs)))
        step = kwargs.get(TrackConst.GLOBAL_STEP_KEY, None)
        if step:
            if not isinstance(step, int):
                raise TypeError("expect step to be an instance of int, but got {}.".format(type(step)))
            if step < 0:
                raise ValueError("expect step to be non-negative int, but got {}.".format(step))

        path = kwargs.get(TrackConst.PATH_KEY, None)
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
        cls, writer_data_type: AnalyticsDataType, writer: LogWriterName, receiver: LogWriterName
    ) -> Optional[AnalyticsDataType]:

        # todo: better way to do this
        if writer == LogWriterName.TORCH_TB and receiver == LogWriterName.MLFLOW:
            if AnalyticsDataType.SCALAR == writer_data_type:
                return AnalyticsDataType.METRIC
            elif AnalyticsDataType.SCALARS == writer_data_type:
                return AnalyticsDataType.METRICS
            else:
                return writer_data_type

        elif writer == LogWriterName.TORCH_TB and receiver == LogWriterName.WANDB:
            if AnalyticsDataType.SCALAR == writer_data_type:
                return AnalyticsDataType.METRIC
            elif AnalyticsDataType.SCALARS == writer_data_type:
                return AnalyticsDataType.METRICS
            else:
                return writer_data_type

        if writer == LogWriterName.MLFLOW and receiver == LogWriterName.TORCH_TB:
            if AnalyticsDataType.TAG == writer_data_type:
                return None
            elif AnalyticsDataType.TAGS == writer_data_type:
                return None
            elif AnalyticsDataType.PARAMETER == writer_data_type:
                return None
            elif AnalyticsDataType.PARAMETERS == writer_data_type:
                return None
            elif AnalyticsDataType.METRIC == writer_data_type:
                return AnalyticsDataType.SCALAR
            elif AnalyticsDataType.METRICS == writer_data_type:
                return AnalyticsDataType.SCALARS
            else:
                return writer_data_type
        elif writer == LogWriterName.MLFLOW and receiver == LogWriterName.WANDB:
            if AnalyticsDataType.TAG == writer_data_type:
                return None
            elif AnalyticsDataType.TAGS == writer_data_type:
                return None
            else:
                return writer_data_type

        elif writer == LogWriterName.WANDB and receiver == LogWriterName.TORCH_TB:
            if AnalyticsDataType.INIT == writer_data_type:
                return None
            if AnalyticsDataType.METRICS == writer_data_type:
                return AnalyticsDataType.SCALARS
            else:
                return writer_data_type

        elif writer == LogWriterName.WANDB and receiver == LogWriterName.MLFLOW:
            if AnalyticsDataType.INIT == writer_data_type:
                return None
            else:
                return writer_data_type

    @classmethod
    def convert_kwargs(cls, kwargs, sender: LogWriterName, receiver: LogWriterName) -> dict:
        if not kwargs:
            return kwargs

        filtered_kwargs = {}
        if sender == LogWriterName.MLFLOW and receiver == LogWriterName.TORCH_TB:
            for k in kwargs:
                if k == TrackConst.GLOBAL_STEP_KEY:
                    filtered_kwargs[k] = kwargs[k]
        else:
            filtered_kwargs = kwargs

        return filtered_kwargs
