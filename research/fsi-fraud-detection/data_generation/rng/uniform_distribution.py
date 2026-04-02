# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Continuous uniform distribution sampler."""

from dataclasses import dataclass, field

from data_generation.rng.rng_base import RNGBase, RNGSampleConfig
from data_generation.rng.typedefs import SampleValueType, VectorSampleValueType


@dataclass
class UniformDistributionSamplingConfig(RNGSampleConfig):
    """Parameters for a continuous uniform distribution U(low, high).

    Attributes:
        low:  Lower bound (inclusive).
        high: Upper bound (exclusive).
    """

    low: float = field(default=0.0)
    high: float = field(default=1.0)


class UniformDistribution(RNGBase[UniformDistributionSamplingConfig]):
    """Sample from a continuous uniform distribution."""

    def __init__(self, seed: int = 42):
        super().__init__("uniform", seed=seed)

    def sample(
        self,
        *args,
        sample_config: UniformDistributionSamplingConfig | None = None,
        size: int = 1,
    ) -> SampleValueType | VectorSampleValueType:
        """Draw *size* samples from U(low, high).

        Raises:
            RuntimeError: If *sample_config* is not provided.
        """
        if sample_config is None:
            raise RuntimeError(
                "UniformDistributionSamplingConfig object with low and high values must be provided for uniform distribution sampling."
            )
        result = self.rng.uniform(low=sample_config.low, high=sample_config.high, size=size)
        return float(result[0]) if size == 1 else result
