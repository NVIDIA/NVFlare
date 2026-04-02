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

"""Gamma distribution sampler."""

from dataclasses import dataclass, field

from data_generation.rng.rng_base import RNGBase, RNGSampleConfig
from data_generation.rng.typedefs import SampleValueType, VectorSampleValueType


@dataclass
class GammaDistributionSamplingConfig(RNGSampleConfig):
    """Parameters for a gamma distribution Gamma(shape, scale).

    Attributes:
        shape: Shape parameter (k).  Must be positive.
        scale: Scale parameter (θ).  Must be positive.
    """

    shape: float = field(default=1.0)
    scale: float = field(default=1.0)


class GammaDistribution(RNGBase[GammaDistributionSamplingConfig]):
    """Sample from a gamma distribution."""

    def __init__(self, seed: int = 42):
        super().__init__("gamma", seed=seed)

    def sample(
        self,
        *args,
        sample_config: GammaDistributionSamplingConfig | None = None,
        size: int = 1,
    ) -> SampleValueType | VectorSampleValueType:
        """Draw *size* samples from Gamma(shape, scale).

        Raises:
            RuntimeError: If *sample_config* is not provided.
        """
        if sample_config is None:
            raise RuntimeError(
                "GammaDistributionSamplingConfig object with shape and scale values must be provided for gamma distribution sampling."
            )
        result = self.rng.gamma(shape=sample_config.shape, scale=sample_config.scale, size=size)
        return float(result[0]) if size == 1 else result
