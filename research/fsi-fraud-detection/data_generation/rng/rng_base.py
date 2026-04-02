"""Abstract base class for reproducible random number generators.

Every concrete RNG subclass (normal, uniform, gamma, …) inherits from
``RNGBase`` and implements ``sample``.  The base class owns a
``numpy.random.Generator`` seeded for reproducibility.
"""

from abc import ABC, abstractmethod
import numpy as np
from data_generation.rng.typedefs import SampleValueType, VectorSampleValueType
from dataclasses import dataclass


@dataclass
class RNGSampleConfig:
    """Base configuration dataclass for RNG sampling parameters.

    Concrete distributions extend this with their own fields (e.g. mean, std_dev).
    """


class RNGBase[T: RNGSampleConfig](ABC):
    """Abstract base for seeded random number generators.

    Args:
        name: Human-readable distribution name (e.g. ``"normal"``).
        seed: Seed for ``numpy.random.default_rng`` to ensure reproducibility.
    """

    def __init__(self, name: str, seed: int = 42):
        self.name = name
        self.rng = np.random.default_rng(seed)

    @abstractmethod
    def sample(
        self, *args, sample_config: T | None = None, size: int = 1
    ) -> SampleValueType | VectorSampleValueType:
        """Draw one or more samples from the distribution.

        Args:
            *args: Distribution-specific positional arguments (e.g. choice options).
            sample_config: Optional typed configuration for the distribution.
            size: Number of samples.  Returns a scalar when ``size=1``,
                an array-like when ``size > 1``.

        Returns:
            A single scalar value or an array of values.
        """
        raise NotImplementedError("Subclasses must implement the sample method.")
