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

"""Faker-based synthetic data provider.

Wraps a ``faker.Faker`` instance with configurable locale and seed so that
the same provider can be shared across all Faker-backed attribute generators.
"""

from typing import override

import faker
from data_generation.synthetic_data_provider.synthetic_data_provider import SyntheticDataProvider


class FakerSyntheticDataProvider(SyntheticDataProvider[faker.Faker]):
    """Provide a seeded ``faker.Faker`` instance for reproducible synthetic data.

    Args:
        locale:        One or more Faker locale strings.
        seed:          Seed for Faker's internal RNG.
        use_weighting: Whether Faker should weight providers by locale frequency.
    """

    def __init__(
        self,
        locale: str | tuple[str, ...] = ("en", "en_US", "tr_TR"),
        seed: int = 42,
        use_weighting: bool = True,
    ):
        super().__init__()
        self._faker_data_provider: faker.Faker = faker.Faker(locale=locale, use_weighting=use_weighting)
        self._faker_data_provider.seed_instance(seed)

    @override
    def provide(self) -> faker.Faker:
        return self._faker_data_provider
