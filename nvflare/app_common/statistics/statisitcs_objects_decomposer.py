# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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
    StatisticConfig,
)
from nvflare.fuel.utils import fobs
from nvflare.fuel.utils.fobs.datum import DatumManager


class StatisticConfigDecomposer(fobs.Decomposer):
    def supported_type(self) -> Type[Any]:
        return StatisticConfig

    def decompose(self, statistic_config: StatisticConfig, manager: DatumManager = None) -> Any:
        return [statistic_config.name, statistic_config.config]

    def recompose(self, data: list, manager: DatumManager = None) -> StatisticConfig:
        return StatisticConfig(data[0], data[1])


class FeatureDecomposer(fobs.Decomposer):
    def supported_type(self) -> Type[Any]:
        return Feature

    def decompose(self, f: Feature, manager: DatumManager = None) -> Any:
        return [f.feature_name, f.data_type]

    def recompose(self, data: list, manager: DatumManager = None) -> Feature:
        return Feature(data[0], data[1])


class BinDecomposer(fobs.Decomposer):
    def supported_type(self) -> Type[Any]:
        return Bin

    def decompose(self, b: Bin, manager: DatumManager = None) -> Any:
        return [b.low_value, b.high_value, b.sample_count]

    def recompose(self, data: list, manager: DatumManager = None) -> Bin:
        return Bin(data[0], data[1], data[2])


class BinRangeDecomposer(fobs.Decomposer):
    def supported_type(self) -> Type[Any]:
        return BinRange

    def decompose(self, b: BinRange, manager: DatumManager = None) -> Any:
        return [b.min_value, b.max_value]

    def recompose(self, data: list, manager: DatumManager = None) -> BinRange:
        return BinRange(data[0], data[1])


class HistogramDecomposer(fobs.Decomposer):
    def supported_type(self) -> Type[Any]:
        return Histogram

    def decompose(self, b: Histogram, manager: DatumManager = None) -> Any:
        return [b.hist_type, b.bins, b.hist_name]

    def recompose(self, data: list, manager: DatumManager = None) -> Histogram:
        return Histogram(data[0], data[1], data[2])


class HistogramTypeDecomposer(fobs.Decomposer):
    def supported_type(self) -> Type[HistogramType]:
        return HistogramType

    def decompose(self, target: HistogramType, manager: DatumManager = None) -> Any:
        return target.value

    def recompose(self, data: Any, manager: DatumManager = None) -> HistogramType:
        return HistogramType(data)


class DataTypeDecomposer(fobs.Decomposer):
    def supported_type(self) -> Type[DataType]:
        return DataType

    def decompose(self, target: DataType, manager: DatumManager = None) -> Any:
        return target.value

    def recompose(self, data: Any, manager: DatumManager = None) -> DataType:
        return DataType(data)


def fobs_registration():
    fobs.register(StatisticConfigDecomposer)
    fobs.register(FeatureDecomposer)
    fobs.register(HistogramDecomposer)
    fobs.register(BinDecomposer)
    fobs.register(BinRangeDecomposer)
    fobs.register(HistogramTypeDecomposer)
    fobs.register(DataTypeDecomposer)
