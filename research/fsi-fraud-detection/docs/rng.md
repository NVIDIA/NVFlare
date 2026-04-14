# RNG Architecture

## Design

All random number generation uses `numpy.random.Generator` (the modern NumPy
RNG API via `np.random.default_rng(seed)`), providing:

- Deterministic, seedable streams.
- Statistical independence between separately-seeded generators.
- No reliance on global state.

## Class Hierarchy

```uml
RNGBase[T: RNGSampleConfig]          (abstract)
  |-- RandomChoice                   discrete sampling from a finite set
  |-- UniformDistribution            continuous U(low, high)
  |-- NormalDistribution             Gaussian N(mean, std_dev^2)
  |-- LogNormalDistribution          log-normal with arithmetic or log-space params
  |-- GammaDistribution              Gamma(shape, scale)
```

### `RNGBase[T]`

Abstract base class, generic over a `RNGSampleConfig` subclass.

- `__init__(name, seed)` -- creates `self.rng = np.random.default_rng(seed)`.
- `sample(*args, sample_config, size)` -- abstract; returns
  `SampleValueType` (scalar) when `size=1`, `VectorSampleValueType`
  (ndarray) when `size > 1`.

### `RNGSampleConfig`

Base dataclass for distribution-specific parameters. Subclasses add fields:

| Config Class                          | Fields                                                           |
| ------------------------------------- | ---------------------------------------------------------------- |
| `RandomChoiceSamplingConfig`          | `prob_distribution: list[float]                         \| None` |
| `UniformDistributionSamplingConfig`   | `low: float`, `high: float`                                      |
| `NormalDistributionSamplingConfig`    | `mean: float`, `std_dev: float`                                  |
| `LogNormalDistributionSamplingConfig` | `mean: float`, `std_dev: float`, `use_log_params: bool`          |
| `GammaDistributionSamplingConfig`     | `shape: float`, `scale: float`                                   |

## Distribution Details

### RandomChoice

Wraps `rng.choice(args, size, p)`. Options are passed as positional
arguments; optional probability weights come from `sample_config`.

### UniformDistribution

Wraps `rng.uniform(low, high, size)`. Returns `float` for `size=1`,
`ndarray` otherwise.

### NormalDistribution

Wraps `rng.normal(loc, scale, size)`.

### LogNormalDistribution

Supports two parameterisations:

1. **Arithmetic** (default): `mean` and `std_dev` describe the desired
   arithmetic mean and standard deviation of the log-normal variable.
   Internally converted to log-space parameters via:

   ```txt
   sigma^2 = ln(1 + var / mean^2)
   mu      = ln(mean) - sigma^2 / 2
   ```

2. **Log-space** (`use_log_params=True`): `mean` and `std_dev` are used
   directly as mu and sigma of the underlying normal distribution.

Wraps `rng.lognormal(mean, sigma, size)`.

### GammaDistribution

Wraps `rng.gamma(shape, scale, size)`.

## Type Aliases

Defined in `data_generation/rng/typedefs.py`:

```python
type SampleValueType = str | int | float | bool | None
type VectorSampleValueType = list[SampleValueType] | tuple[SampleValueType, ...] | np.ndarray
```

The union return type reflects that `RandomChoice` can return strings (for
categorical sampling), while numeric distributions return floats or arrays.

## Reproducibility Model

Each RNG instance owns an independent `np.random.Generator`. Seeds are
derived deterministically from a global seed:

- **Data generation**: `provider_seed = global_seed + ds_idx * 100 + (i - 1)`
- **Shuffle**: `random_state = dataset_seed` (pandas `df.sample()`).
- **Anomaly injection**: `inject_all` creates one `default_rng(seed)` for
  sampling `fraud_row_frac` and `random_state`, then passes
  `anomaly_seed = seed + type_index` to each transformer.
- **Probability thinning**: `random_state = dataset_seed` (pandas `df.sample()`).

All RNG-dependent operations within a dataset use the same derived
`dataset_seed`, so changing the global seed consistently propagates to every
stochastic step. There are no hidden fixed constants.
