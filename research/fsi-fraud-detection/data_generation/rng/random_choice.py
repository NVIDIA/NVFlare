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

"""Uniform random choice from a discrete set of options."""

from dataclasses import dataclass, field

from data_generation.rng.rng_base import RNGBase, RNGSampleConfig
from data_generation.rng.typedefs import SampleValueType, VectorSampleValueType


@dataclass
class RandomChoiceSamplingConfig(RNGSampleConfig):
    """Configuration for weighted random choice.

    Attributes:
        prob_distribution: Optional probability weights for each option.
            Must have the same length as the ``*args`` passed to ``sample``.
            When *None*, all options are equally likely.
    """

    prob_distribution: list[float] | None = field(default=None)


class RandomChoice(RNGBase[RandomChoiceSamplingConfig]):
    """Randomly select from a set of discrete options with optional weights."""

    def __init__(self, seed: int = 42) -> None:
        super().__init__("random_choice", seed)

    def sample(
        self,
        *args,
        sample_config: RandomChoiceSamplingConfig | None = None,
        size: int = 1,
    ) -> SampleValueType | VectorSampleValueType:
        """Draw *size* samples from *args*.

        Raises:
            RuntimeError: If no options are provided.
        """
        if not args:
            raise RuntimeError(
                "For random_choice distribution, you need to provide a non-empty list or tuple of options as arguments."
            )
        result = self.rng.choice(
            args,
            size=size,
            p=(sample_config.prob_distribution if sample_config and sample_config.prob_distribution else None),
        )
        return result[0] if size == 1 else result
