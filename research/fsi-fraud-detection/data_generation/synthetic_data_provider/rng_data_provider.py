"""RNG-based synthetic data providers.

Each concrete class pairs a ``SyntheticDataProvider`` with a specific
``RNGBase`` subclass (random-choice, uniform, normal, …) so that attribute
generators can call ``provider.rng.sample(...)`` directly.
"""

from data_generation.synthetic_data_provider.synthetic_data_provider import (
    SyntheticDataProvider,
)
from data_generation.rng.random_choice import RandomChoice
from data_generation.rng.gamma_distribution import GammaDistribution
from data_generation.rng.normal_distribution import NormalDistribution
from data_generation.rng.lognormal_distribution import LogNormalDistribution
from data_generation.rng.uniform_distribution import UniformDistribution
from data_generation.rng.rng_base import RNGBase
from typing import override


class RNGSyntheticDataProvider[T: RNGBase](SyntheticDataProvider[T]):
    """Generic provider that wraps any ``RNGBase`` subclass.

    Args:
        rng_type: The concrete RNG class to instantiate.
        seed:     Seed forwarded to the RNG for reproducibility.
        **kwargs: Extra keyword arguments forwarded to *rng_type*.
    """

    def __init__(
        self,
        rng_type: type[T],
        seed: int = 42,
        **kwargs,
    ):
        super().__init__()
        self.rng = rng_type(seed=seed, **kwargs)

    @override
    def provide(self) -> T:
        return self.rng


class RandomChoiceDataProvider(RNGSyntheticDataProvider[RandomChoice]):
    """Provider for discrete random choice sampling."""

    def __init__(self, seed: int = 42):
        super().__init__(RandomChoice, seed=seed)

    @override
    def provide(self) -> RandomChoice:
        return self.rng


class LogNormalDistributionDataProvider(
    RNGSyntheticDataProvider[LogNormalDistribution]
):
    """Provider for log-normal distribution sampling."""

    def __init__(self, seed: int = 42):
        super().__init__(LogNormalDistribution, seed=seed)

    @override
    def provide(self) -> LogNormalDistribution:
        return self.rng


class NormalDistributionDataProvider(RNGSyntheticDataProvider[NormalDistribution]):
    """Provider for normal (Gaussian) distribution sampling."""

    def __init__(self, seed: int = 42):
        super().__init__(NormalDistribution, seed=seed)

    @override
    def provide(self) -> NormalDistribution:
        return self.rng


class GammaDistributionDataProvider(RNGSyntheticDataProvider[GammaDistribution]):
    """Provider for gamma distribution sampling."""

    def __init__(self, seed: int = 42):
        super().__init__(GammaDistribution, seed=seed)

    @override
    def provide(self) -> GammaDistribution:
        return self.rng


class UniformDistributionDataProvider(RNGSyntheticDataProvider[UniformDistribution]):
    """Provider for continuous uniform distribution sampling."""

    def __init__(self, seed: int = 42):
        super().__init__(UniformDistribution, seed=seed)

    @override
    def provide(self) -> UniformDistribution:
        return self.rng
