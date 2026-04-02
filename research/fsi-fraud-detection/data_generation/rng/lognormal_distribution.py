"""Log-normal distribution sampler.

Supports two parameterisations:
  1. *Arithmetic* mean and std dev of the log-normal variable (default).
  2. *Log-space* \u03bc and \u03c3 of the underlying normal (``use_log_params=True``).
"""

from dataclasses import dataclass, field

import numpy as np
from data_generation.rng.rng_base import RNGBase, RNGSampleConfig
from data_generation.rng.typedefs import SampleValueType, VectorSampleValueType


@dataclass
class LogNormalDistributionSamplingConfig(RNGSampleConfig):
    """Parameters for a log-normal distribution.

    Attributes:
        mean:           Arithmetic mean (or log-space \u03bc when *use_log_params* is True).
        std_dev:        Arithmetic std dev (or log-space \u03c3 when *use_log_params* is True).
        use_log_params: Interpret *mean*/*std_dev* as the \u03bc/\u03c3 of the
            underlying normal distribution rather than the arithmetic moments.
    """

    mean: float = field(default=0.0)
    std_dev: float = field(default=1.0)
    use_log_params: bool = field(default=False)


def get_lognormal_params(target_mean: float, target_std: float) -> tuple[float, float]:
    """
    Calculates mu and sigma for np.random.lognormal
    given a desired arithmetic mean and standard deviation.
    """
    # Calculate variance (std^2)
    target_var = target_std**2

    # Calculate the underlying sigma (standard deviation)
    # sigma^2 = ln(1 + var/mean^2)
    sigma_sq = np.log(1 + (target_var / target_mean**2))
    sigma = np.sqrt(sigma_sq)

    # Calculate the underlying mu (mean)
    # mu = ln(mean) - sigma^2 / 2
    mu = np.log(target_mean) - (sigma_sq / 2.0)

    return mu, sigma


class LogNormalDistribution(RNGBase[LogNormalDistributionSamplingConfig]):
    """Sample from a log-normal distribution."""

    def __init__(self, seed: int = 42):
        super().__init__("lognormal", seed=seed)

    def sample(
        self,
        *args,
        sample_config: LogNormalDistributionSamplingConfig | None = None,
        size: int = 1,
    ) -> SampleValueType | VectorSampleValueType:
        """Draw *size* samples from a log-normal distribution.

        When ``sample_config.use_log_params`` is False (default), the arithmetic
        mean and std dev are converted to log-space parameters via
        ``get_lognormal_params`` before sampling.

        Raises:
            RuntimeError: If *sample_config* is not provided.
        """
        if sample_config is None:
            raise RuntimeError(
                "LogNormalDistributionSamplingConfig object with mean and std_dev values must be provided for lognormal distribution sampling."
            )

        mean, std_dev = (
            get_lognormal_params(
                target_mean=sample_config.mean,
                target_std=sample_config.std_dev,
            )
            if not sample_config.use_log_params
            else (sample_config.mean, sample_config.std_dev)
        )
        result = self.rng.lognormal(mean=mean, sigma=std_dev, size=size)
        return float(result[0]) if size == 1 else result
