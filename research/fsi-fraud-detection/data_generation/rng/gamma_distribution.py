"""Gamma distribution sampler."""

from data_generation.rng.rng_base import RNGBase, RNGSampleConfig
from data_generation.rng.typedefs import SampleValueType, VectorSampleValueType

from dataclasses import dataclass, field


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
        result = self.rng.gamma(
            shape=sample_config.shape, scale=sample_config.scale, size=size
        )
        return float(result[0]) if size == 1 else result
