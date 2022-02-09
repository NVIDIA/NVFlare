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

_DATA_TYPE_KEY = "analytics_data_type"
_KWARGS_KEY = "analytics_kwargs"


class AnalyticsDataType(Enum):
    SCALARS = "SCALARS"
    SCALAR = "SCALAR"
    IMAGE = "IMAGE"
    TEXT = "TEXT"
    LOG_RECORD = "LOG_RECORD"


class AnalyticsData:
    def __init__(self, tag: str, value, data_type: AnalyticsDataType, kwargs: Optional[dict] = None):
        """This class defines AnalyticsData format.

        It is a wrapper to provide to/from DXO conversion.

        Args:
            tag (str): tag name
            value: value
            data_type (AnalyticDataType): type of the analytic data.
            kwargs (optional, dict): additional arguments to be passed.
        """
        if not isinstance(tag, str):
            raise TypeError("expect tag to be an instance of str, but got {}.".format(type(tag)))
        if not isinstance(data_type, AnalyticsDataType):
            raise TypeError(
                "expect data_type to be an instance of AnalyticsDataType, but got {}.".format(type(data_type))
            )
        if kwargs and not isinstance(kwargs, dict):
            raise TypeError("expect kwargs to be an instance of dict, but got {}.".format(type(kwargs)))
        if data_type == AnalyticsDataType.SCALAR and not isinstance(value, float):
            raise TypeError("expect value to be an instance of float, but got {}.".format(type(value)))
        elif data_type == AnalyticsDataType.SCALARS and not isinstance(value, dict):
            raise TypeError("expect value to be an instance of dict, but got {}.".format(type(value)))
        elif data_type == AnalyticsDataType.TEXT and not isinstance(value, str):
            raise TypeError("expect value to be an instance of str, but got {}.".format(type(value)))
        self.tag = tag
        self.value = value
        self.data_type = data_type
        self.kwargs = kwargs

    def to_dxo(self):
        """Converts the AnalyticsData to DXO object.

        Returns:
            DXO object
        """
        dxo = DXO(data_kind=DataKind.ANALYTIC, data={self.tag: self.value})
        dxo.set_meta_prop(_DATA_TYPE_KEY, self.data_type)
        dxo.set_meta_prop(_KWARGS_KEY, self.kwargs)
        return dxo

    @classmethod
    def from_dxo(cls, dxo: DXO):
        """Generates the AnalyticsData from DXO object.

        Args:
            dxo (DXO): The DXO object to convert.

        Returns:
            AnalyticsData object
        """
        if not isinstance(dxo, DXO):
            raise TypeError("expect dxo to be an instance of DXO, but got {}.".format(type(dxo)))

        if len(dxo.data) != 1:
            raise ValueError(
                "dxo does not have the correct format for AnalyticsData; expected dxo.data to be length 1, but got {}".format(
                    len(dxo.data)
                )
            )

        tag, value = list(dxo.data.items())[0]

        data_type = dxo.get_meta_prop(_DATA_TYPE_KEY)
        kwargs = dxo.get_meta_prop(_KWARGS_KEY)

        return cls(tag, value, data_type, kwargs)
