"""Normal (Gaussian) distribution sampler."""

from data_generation.rng.rng_base import RNGBase, RNGSampleConfig
from data_generation.rng.typedefs import SampleValueType, VectorSampleValueType

from dataclasses import dataclass, field


@dataclass
class NormalDistributionSamplingConfig(RNGSampleConfig):
    """Parameters for a normal distribution N(mean, std_dev²).

    Attributes:
        mean:    Arithmetic mean of the distribution.
        std_dev: Standard deviation of the distribution.
    """

    mean: float = field(default=0.0)
    std_dev: float = field(default=1.0)


class NormalDistribution(RNGBase[NormalDistributionSamplingConfig]):
    """Sample from a normal (Gaussian) distribution."""

    def __init__(self, seed: int = 42):
        super().__init__("normal", seed=seed)

    def sample(
        self,
        *args,
        sample_config: NormalDistributionSamplingConfig | None = None,
        size: int = 1,
    ) -> SampleValueType | VectorSampleValueType:
        """Draw *size* samples from N(mean, std_dev²).

        Raises:
            RuntimeError: If *sample_config* is not provided.
        """
        if sample_config is None:
            raise RuntimeError(
                "NormalDistributionSamplingConfig object with mean and std_dev values must be provided for normal distribution sampling."
            )
        result = self.rng.normal(
            loc=sample_config.mean, scale=sample_config.std_dev, size=size
        )
        return float(result[0]) if size == 1 else result
