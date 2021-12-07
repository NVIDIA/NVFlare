# Copyright (c) 2021, NVIDIA CORPORATION.
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

DATA_TYPE_KEY = "analytics_data_type"
KWARGS_KEY = "analytics_kwargs"


class AnalyticsDataType(Enum):
    SCALARS = "SCALARS"
    SCALAR = "SCALAR"
    IMAGE = "IMAGE"
    TEXT = "TEXT"


class AnalyticsData:
    def __init__(self, tag: str, value, data_type: AnalyticsDataType, kwargs: Optional[dict]):
        """This class defines AnalyticsData format.

            It is a wrapper to provide from / to DXO conversion.
        Args:
            tag (str): tag name
            value: value
            data_type (AnalyticDataType): type of the analytic data.
            kwargs (optional, dict): additional arguments to be passed.
        """
        self.tag = tag
        self.value = value
        self.data_type = data_type
        self.kwargs = kwargs

    def to_dxo(self):
        """Converts the AnalyticsData to DXO object."""
        dxo = DXO(data_kind=DataKind.ANALYTIC, data={self.tag: self.value})
        dxo.set_meta_prop(DATA_TYPE_KEY, self.data_type)
        dxo.set_meta_prop(KWARGS_KEY, self.kwargs)
        return dxo

    @classmethod
    def from_dxo(cls, dxo: DXO):
        """Generates the AnalyticsData from DXO object.

        Args:
            dxo (DXO): The DXO object to convert.
        """
        if not isinstance(dxo, DXO):
            raise TypeError(f"dxo is not of type DXO, instead it has type {type(dxo)}.")

        if len(dxo.data) != 1:
            raise ValueError("dxo does not have the correct format for AnalyticsData.")

        tag, value = list(dxo.data.items())[0]

        data_type = dxo.get_meta_prop(DATA_TYPE_KEY)
        kwargs = dxo.get_meta_prop(KWARGS_KEY)
        if not isinstance(data_type, AnalyticsDataType):
            raise ValueError(f"data_type {data_type} is not supported.")

        return cls(tag, value, data_type, kwargs)
