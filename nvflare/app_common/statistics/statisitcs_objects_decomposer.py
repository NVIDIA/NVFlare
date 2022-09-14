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
from typing import Any, Type

from nvflare.app_common.abstract.statistics_spec import (
    Bin,
    BinRange,
    DataType,
    Feature,
    Histogram,
    HistogramType,
    MetricConfig,
)
from nvflare.fuel.utils import fobs


class MetricConfigDecomposer(fobs.Decomposer):
    @staticmethod
    def supported_type() -> Type[Any]:
        return MetricConfig

    def decompose(self, metric_config: MetricConfig) -> Any:
        return [metric_config.name, metric_config.config]

    def recompose(self, data: list) -> MetricConfig:
        return MetricConfig(data[0], data[1])


class DataTypeDecomposer(fobs.Decomposer):
    @staticmethod
    def supported_type() -> Type[Any]:
        return DataType

    def decompose(self, dt: DataType) -> Any:
        return dt.value

    def recompose(self, data: Any) -> DataType:
        return DataType(data)


class HistogramTypeDecomposer(fobs.Decomposer):
    @staticmethod
    def supported_type() -> Type[Any]:
        return HistogramType

    def decompose(self, ht: HistogramType) -> Any:
        return ht.value

    def recompose(self, data: Any) -> HistogramType:
        return HistogramType(data)


class FeatureDecomposer(fobs.Decomposer):
    @staticmethod
    def supported_type() -> Type[Any]:
        return Feature

    def decompose(self, f: Feature) -> Any:
        return [f.feature_name, f.data_type]

    def recompose(self, data: list) -> Feature:
        return Feature(data[0], data[1])


class BinDecomposer(fobs.Decomposer):
    @staticmethod
    def supported_type() -> Type[Any]:
        return Bin

    def decompose(self, b: Bin) -> Any:
        return [b.low_value, b.high_value, b.sample_count]

    def recompose(self, data: list) -> Bin:
        return Bin(data[0], data[1], data[2])


class BinRangeDecomposer(fobs.Decomposer):
    @staticmethod
    def supported_type() -> Type[Any]:
        return BinRange

    def decompose(self, b: BinRange) -> Any:
        return [b.min_value, b.max_value]

    def recompose(self, data: list) -> BinRange:
        return BinRange(data[0], data[1])


class HistogramDecomposer(fobs.Decomposer):
    @staticmethod
    def supported_type() -> Type[Any]:
        return Histogram

    def decompose(self, b: Histogram) -> Any:
        return [b.hist_type, b.bins, b.hist_name]

    def recompose(self, data: list) -> Histogram:
        return Histogram(data[0], data[1], data[2])


def fobs_registration():
    fobs.register(MetricConfigDecomposer)
    fobs.register(DataTypeDecomposer)
    fobs.register(FeatureDecomposer)
    fobs.register(HistogramTypeDecomposer)
    fobs.register(HistogramDecomposer)
    fobs.register(BinDecomposer)
    fobs.register(BinRangeDecomposer)
