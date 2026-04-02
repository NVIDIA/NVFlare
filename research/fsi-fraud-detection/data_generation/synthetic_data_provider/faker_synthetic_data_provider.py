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
