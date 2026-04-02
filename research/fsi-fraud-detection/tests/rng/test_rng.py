import numpy as np
import pytest
from data_generation.rng.gamma_distribution import GammaDistribution, GammaDistributionSamplingConfig
from data_generation.rng.lognormal_distribution import (
    LogNormalDistribution,
    LogNormalDistributionSamplingConfig,
    get_lognormal_params,
)
from data_generation.rng.normal_distribution import NormalDistribution, NormalDistributionSamplingConfig
from data_generation.rng.random_choice import RandomChoice, RandomChoiceSamplingConfig
from data_generation.rng.rng_base import RNGBase, RNGSampleConfig
from data_generation.rng.uniform_distribution import UniformDistribution, UniformDistributionSamplingConfig

SEED = 42


# ---------------------------------------------------------------------------
# RNGBase
# ---------------------------------------------------------------------------


class TestRNGBase:
    def test_is_abstract(self) -> None:
        with pytest.raises(TypeError):
            RNGBase("test", seed=SEED)  # type: ignore[abstract]

    def test_sample_config_dataclass_default(self) -> None:
        cfg = RNGSampleConfig()
        assert isinstance(cfg, RNGSampleConfig)


# ---------------------------------------------------------------------------
# RandomChoice
# ---------------------------------------------------------------------------


class TestRandomChoice:
    @pytest.fixture
    def rng(self) -> RandomChoice:
        return RandomChoice(seed=SEED)

    def test_name(self, rng: RandomChoice) -> None:
        assert rng.name == "random_choice"

    def test_sample_returns_one_of_options(self, rng: RandomChoice) -> None:
        options = ("a", "b", "c")
        result = rng.sample(*options)
        assert result in options

    def test_deterministic_with_same_seed(self) -> None:
        r1 = RandomChoice(seed=SEED)
        r2 = RandomChoice(seed=SEED)
        results1 = [r1.sample("x", "y", "z") for _ in range(20)]
        results2 = [r2.sample("x", "y", "z") for _ in range(20)]
        assert results1 == results2

    def test_different_seeds_diverge(self) -> None:
        r1 = RandomChoice(seed=1)
        r2 = RandomChoice(seed=999)
        results1 = [r1.sample("x", "y", "z") for _ in range(50)]
        results2 = [r2.sample("x", "y", "z") for _ in range(50)]
        assert results1 != results2

    def test_respects_probability_distribution(self) -> None:
        rng = RandomChoice(seed=SEED)
        cfg = RandomChoiceSamplingConfig(prob_distribution=[1.0, 0.0, 0.0])
        for _ in range(20):
            assert rng.sample("a", "b", "c", sample_config=cfg) == "a"

    def test_uniform_when_no_config(self) -> None:
        rng = RandomChoice(seed=SEED)
        results = {str(rng.sample("a", "b")) for _ in range(200)}
        assert results == {"a", "b"}

    def test_raises_on_empty_args(self, rng: RandomChoice) -> None:
        with pytest.raises(RuntimeError, match="non-empty"):
            rng.sample()

    def test_single_option(self, rng: RandomChoice) -> None:
        assert rng.sample("only") == "only"

    def test_works_with_numeric_options(self, rng: RandomChoice) -> None:
        result = rng.sample(1, 2, 3)
        assert result in {1, 2, 3}

    def test_config_none_prob_treated_as_uniform(self, rng: RandomChoice) -> None:
        cfg = RandomChoiceSamplingConfig(prob_distribution=None)
        result = rng.sample("a", "b", sample_config=cfg)
        assert result in {"a", "b"}


# ---------------------------------------------------------------------------
# UniformDistribution
# ---------------------------------------------------------------------------


class TestUniformDistribution:
    @pytest.fixture
    def rng(self) -> UniformDistribution:
        return UniformDistribution(seed=SEED)

    def test_name(self, rng: UniformDistribution) -> None:
        assert rng.name == "uniform"

    def test_sample_within_bounds(self, rng: UniformDistribution) -> None:
        cfg = UniformDistributionSamplingConfig(low=5.0, high=10.0)
        for _ in range(100):
            val = rng.sample(sample_config=cfg)
            assert isinstance(val, float)
            assert 5.0 <= val < 10.0

    def test_deterministic_with_same_seed(self) -> None:
        r1 = UniformDistribution(seed=SEED)
        r2 = UniformDistribution(seed=SEED)
        cfg = UniformDistributionSamplingConfig(low=0.0, high=1.0)
        v1 = [r1.sample(sample_config=cfg) for _ in range(20)]
        v2 = [r2.sample(sample_config=cfg) for _ in range(20)]
        assert v1 == v2

    def test_raises_on_none_config(self, rng: UniformDistribution) -> None:
        with pytest.raises(RuntimeError, match="UniformDistributionSamplingConfig"):
            rng.sample(sample_config=None)

    def test_default_config_bounds(self, rng: UniformDistribution) -> None:
        cfg = UniformDistributionSamplingConfig()
        assert cfg.low == pytest.approx(0.0)
        assert cfg.high == pytest.approx(1.0)

    def test_narrow_range(self, rng: UniformDistribution) -> None:
        cfg = UniformDistributionSamplingConfig(low=100.0, high=100.001)
        val = rng.sample(sample_config=cfg)
        assert isinstance(val, float)
        assert 100.0 <= val < 100.001


# ---------------------------------------------------------------------------
# NormalDistribution
# ---------------------------------------------------------------------------


class TestNormalDistribution:
    @pytest.fixture
    def rng(self) -> NormalDistribution:
        return NormalDistribution(seed=SEED)

    def test_name(self, rng: NormalDistribution) -> None:
        assert rng.name == "normal"

    def test_sample_returns_float(self, rng: NormalDistribution) -> None:
        cfg = NormalDistributionSamplingConfig(mean=0.0, std_dev=1.0)
        val = rng.sample(sample_config=cfg)
        assert isinstance(val, float)

    def test_deterministic_with_same_seed(self) -> None:
        r1 = NormalDistribution(seed=SEED)
        r2 = NormalDistribution(seed=SEED)
        cfg = NormalDistributionSamplingConfig(mean=5.0, std_dev=2.0)
        v1 = [r1.sample(sample_config=cfg) for _ in range(20)]
        v2 = [r2.sample(sample_config=cfg) for _ in range(20)]
        assert v1 == v2

    def test_mean_converges_statistically(self, rng: NormalDistribution) -> None:
        cfg = NormalDistributionSamplingConfig(mean=50.0, std_dev=1.0)
        samples: list[float] = [float(rng.sample(sample_config=cfg)) for _ in range(10_000)]  # type: ignore[arg-type]
        assert np.mean(samples) == pytest.approx(50.0, abs=0.5)

    def test_raises_on_none_config(self, rng: NormalDistribution) -> None:
        with pytest.raises(RuntimeError, match="NormalDistributionSamplingConfig"):
            rng.sample(sample_config=None)

    def test_default_config_values(self) -> None:
        cfg = NormalDistributionSamplingConfig()
        assert cfg.mean == pytest.approx(0.0)
        assert cfg.std_dev == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# LogNormalDistribution — get_lognormal_params helper
# ---------------------------------------------------------------------------


class TestGetLognormalParams:
    def test_returns_two_floats(self) -> None:
        mu, sigma = get_lognormal_params(100.0, 50.0)
        assert isinstance(mu, float)
        assert isinstance(sigma, float)

    def test_roundtrip_mean(self) -> None:
        """Verify that the computed mu/sigma reproduce the target arithmetic mean."""
        target_mean, target_std = 500.0, 200.0
        mu, sigma = get_lognormal_params(target_mean, target_std)
        # E[X] = exp(mu + sigma^2 / 2) for lognormal
        computed_mean = np.exp(mu + sigma**2 / 2.0)
        assert computed_mean == pytest.approx(target_mean, rel=1e-6)

    def test_roundtrip_std(self) -> None:
        """Verify that the computed mu/sigma reproduce the target arithmetic std."""
        target_mean, target_std = 500.0, 200.0
        mu, sigma = get_lognormal_params(target_mean, target_std)
        computed_var = (np.exp(sigma**2) - 1) * np.exp(2 * mu + sigma**2)
        computed_std = np.sqrt(computed_var)
        assert computed_std == pytest.approx(target_std, rel=1e-6)


# ---------------------------------------------------------------------------
# LogNormalDistribution
# ---------------------------------------------------------------------------


class TestLogNormalDistribution:
    @pytest.fixture
    def rng(self) -> LogNormalDistribution:
        return LogNormalDistribution(seed=SEED)

    def test_name(self, rng: LogNormalDistribution) -> None:
        assert rng.name == "lognormal"

    def test_sample_returns_positive_float(self, rng: LogNormalDistribution) -> None:
        cfg = LogNormalDistributionSamplingConfig(mean=100.0, std_dev=50.0)
        val = rng.sample(sample_config=cfg)
        assert isinstance(val, float)
        assert val > 0.0

    def test_deterministic_with_same_seed(self) -> None:
        r1 = LogNormalDistribution(seed=SEED)
        r2 = LogNormalDistribution(seed=SEED)
        cfg = LogNormalDistributionSamplingConfig(mean=100.0, std_dev=50.0)
        v1 = [r1.sample(sample_config=cfg) for _ in range(20)]
        v2 = [r2.sample(sample_config=cfg) for _ in range(20)]
        assert v1 == v2

    def test_use_log_params_bypasses_conversion(self) -> None:
        """When use_log_params=True, mean/std_dev are passed directly as mu/sigma."""
        r1 = LogNormalDistribution(seed=SEED)
        r2 = LogNormalDistribution(seed=SEED)
        mu, sigma = 4.0, 0.5
        cfg_direct = LogNormalDistributionSamplingConfig(mean=mu, std_dev=sigma, use_log_params=True)
        cfg_converted = LogNormalDistributionSamplingConfig(mean=mu, std_dev=sigma, use_log_params=False)
        val_direct = r1.sample(sample_config=cfg_direct)
        val_converted = r2.sample(sample_config=cfg_converted)
        # They should differ because conversion changes the effective mu/sigma
        assert val_direct != val_converted

    def test_mean_converges_statistically(self) -> None:
        rng = LogNormalDistribution(seed=SEED)
        target_mean = 200.0
        cfg = LogNormalDistributionSamplingConfig(mean=target_mean, std_dev=50.0)
        samples: list[float] = [float(rng.sample(sample_config=cfg)) for _ in range(10_000)]  # type: ignore[arg-type]
        assert np.mean(samples) == pytest.approx(target_mean, rel=0.05)

    def test_raises_on_none_config(self, rng: LogNormalDistribution) -> None:
        with pytest.raises(RuntimeError, match="LogNormalDistributionSamplingConfig"):
            rng.sample(sample_config=None)

    def test_default_config_values(self) -> None:
        cfg = LogNormalDistributionSamplingConfig()
        assert cfg.mean == pytest.approx(0.0)
        assert cfg.std_dev == pytest.approx(1.0)
        assert cfg.use_log_params is False


# ---------------------------------------------------------------------------
# GammaDistribution
# ---------------------------------------------------------------------------


class TestGammaDistribution:
    @pytest.fixture
    def rng(self) -> GammaDistribution:
        return GammaDistribution(seed=SEED)

    def test_name(self, rng: GammaDistribution) -> None:
        assert rng.name == "gamma"

    def test_sample_returns_positive_float(self, rng: GammaDistribution) -> None:
        cfg = GammaDistributionSamplingConfig(shape=2.0, scale=3.0)
        val = rng.sample(sample_config=cfg)
        assert isinstance(val, float)
        assert val > 0.0

    def test_deterministic_with_same_seed(self) -> None:
        r1 = GammaDistribution(seed=SEED)
        r2 = GammaDistribution(seed=SEED)
        cfg = GammaDistributionSamplingConfig(shape=2.0, scale=3.0)
        v1 = [r1.sample(sample_config=cfg) for _ in range(20)]
        v2 = [r2.sample(sample_config=cfg) for _ in range(20)]
        assert v1 == v2

    def test_mean_converges_statistically(self) -> None:
        """Gamma mean = shape * scale."""
        rng = GammaDistribution(seed=SEED)
        shape, scale = 5.0, 2.0
        cfg = GammaDistributionSamplingConfig(shape=shape, scale=scale)
        samples: list[float] = [float(rng.sample(sample_config=cfg)) for _ in range(10_000)]  # type: ignore[arg-type]
        assert np.mean(samples) == pytest.approx(shape * scale, rel=0.05)

    def test_raises_on_none_config(self, rng: GammaDistribution) -> None:
        with pytest.raises(RuntimeError, match="GammaDistributionSamplingConfig"):
            rng.sample(sample_config=None)

    def test_default_config_values(self) -> None:
        cfg = GammaDistributionSamplingConfig()
        assert cfg.shape == pytest.approx(1.0)
        assert cfg.scale == pytest.approx(1.0)
