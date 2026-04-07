# Data Generation Architecture

## Overview

The data generation layer is a **declarative, attribute-driven system** for
producing synthetic payment DataFrames. Rather than writing imperative
row-by-row loops, the developer declares:

1. **What** columns to generate (attribute descriptors).
2. **How** each column gets its values (pluggable provider callables).
3. **Which** columns must exist before a given column can be computed
   (inter-attribute dependencies).

The framework resolves dependencies automatically, instantiates the correct
provider for each attribute, and generates the entire DataFrame
column-by-column in a single pass. This design follows the **Strategy
pattern**: each attribute delegates value generation to an interchangeable
provider (Faker, RNG sampler, static lookup, ...) selected at registration
time, making it straightforward to swap data sources or add new columns
without modifying the generation engine.

Because providers operate on the full column at once (returning a
`pd.Series` or `pd.DataFrame` rather than a single scalar), the system
naturally produces **vectorised** generation -- NumPy and pandas batch
operations replace per-row Python loops, yielding significant performance
gains for large datasets.

## Attribute Dependency Model

Attributes and their dependencies form a acyclic digraph (DAG).
Each attribute declares:

- One or more output column names.
- A list of prerequisite column names (may be empty).
- A callable that generates column values given a data provider and the
  partially-built DataFrame.

The graph is defined in `data_generation/attributes.py` through three
composable registration functions:

| Function                                   | Columns Registered                                                                                                                                                                                                                                   |
| ------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `get_per_participant_attributes(prefixes)` | Per-party columns for each prefix (DEBITOR, CREDITOR): username, name, email, phone, gender, account number, BIC, IP, comment, account type, address fields, DOB, currency, geo-coordinates, account timestamps, tower coordinates, activity events. |
| `get_payment_core_attributes(prefixes)`    | Payment-level: `PAYMENT_ID`, `PAYMENT_INIT_TIMESTAMP`, `PAYMENT_LAST_UPDATE_TIMESTAMP`, `PAYMENT_STATUS`.                                                                                                                                            |
| `get_payment_amount_attributes(prefixes)`  | Exchange rates (`{P0}_CCY_{P1}_CCY_RATE`) and amounts (`{P0}_AMOUNT`, `{P1}_AMOUNT`).                                                                                                                                                                |

Dependencies encode real-world constraints. For example:

- Currency depends on country (currency is derived from the participant's
  address country).
- Geo-coordinates depend on country (lat/lon are bounded to the country).
- Account creation timestamp depends on date of birth (account cannot predate
  birth).
- Payment init timestamp depends on both parties' last activity timestamps.
- Amounts depend on account type and exchange rate.

## Dependency Resolution

The DAG is resolved via Kahn's topological sort algorithm
(`dataset.topological_sort()`), which determines a valid generation order
such that every attribute is produced after its prerequisites:

1. Builds a reverse index mapping each column name to its producing
   attribute.
2. Computes in-degree for each attribute based on inter-attribute
   dependencies (not raw column dependencies).
3. BFS dequeues zero-in-degree attributes, appends them to the sorted order,
   and decrements dependents' in-degrees.
4. Raises `ValueError` if the sorted count does not match the graph size
   (cycle detected).

## Attribute Descriptors

### `PaymentDatasetAttribute[T]`

Wraps a single column name and its provider callable. A specialisation of
`PaymentDatasetAttributeGroup` with a one-element name tuple.

### `PaymentDatasetAttributeGroup[T]`

Wraps a tuple of column names and a single provider callable that produces
values for all of them at once. Used for correlated multi-column outputs
(e.g. latitude + longitude, exchange rate pairs, amounts).

Both types are generic over `T: SyntheticDataProvider`, which constrains
the provider type the callable expects.

The `emit()` method invokes the callable with:

```python
result = attribute_data_provider(
    provider=synthetic_data_provider,
    df=df,
    dependent_columns=dependent_columns,
    **kwargs,
)
```

## `generate()` Function

`dataset.generate()` is the entry point that ties the system together --
resolving dependencies, walking attributes in order, and delegating to
providers:

```python
def generate(
    dependency_graph: Mapping[PaymentDatasetAttributeGroup, list[str]],
    providers: Mapping[PaymentDatasetAttributeGroup, SyntheticDataProvider],
    n_rows: int,
    **kwargs,
) -> pd.DataFrame:
```

1. Resolves attribute dependencies via topological sort.
2. Creates an empty DataFrame with `n_rows` rows.
3. Iterates attributes in dependency order, calling `emit()` for each.
4. Each `emit()` call delegates to the attribute's provider callable,
   which generates values for the entire column (or column group) at once.
5. Returns the fully populated DataFrame.

Extra `**kwargs` (e.g. `country_static_data`, `uniform_dist_config_lat`)
are forwarded to every `emit()` call, allowing site-specific distribution
parameters to reach the provider callables.

## Data Providers

Providers are the pluggable data sources in the Strategy pattern. Each
attribute is bound to a provider type at registration time; the generation
engine resolves the correct instance and passes it to the attribute's
callable.

### `SyntheticDataProvider[T]`

Abstract base class. Concrete subclasses wrap a data source and expose it
via `provide() -> T`.

### `FakerSyntheticDataProvider`

Wraps `faker.Faker` with configurable locale(s) and seed.
`provide()` returns the Faker instance.

### RNG-Based Providers

`RNGSyntheticDataProvider[T: RNGBase]` pairs `SyntheticDataProvider` with
an `RNGBase` subclass. Concrete specialisations:

| Class                               | RNG                     | Use Case                      |
| ----------------------------------- | ----------------------- | ----------------------------- |
| `RandomChoiceDataProvider`          | `RandomChoice`          | Discrete categorical sampling |
| `UniformDistributionDataProvider`   | `UniformDistribution`   | Continuous uniform sampling   |
| `LogNormalDistributionDataProvider` | `LogNormalDistribution` | Log-normal amount generation  |

### `AttributeDataProviderProtocol[T]`

The callable protocol all column generators implement:

```python
class AttributeDataProviderProtocol[T: SyntheticDataProvider](Protocol):
    def __call__(
        self,
        provider: T,
        df: pd.DataFrame,
        dependent_columns: list[str] | None = None,
        **kwargs,
    ) -> pd.Series | pd.DataFrame: ...
```

### Helper Functions

`faker_synthetic_data_provider_helper_functions.py` contains all concrete
attribute generators. Each function:

- Receives a provider instance and the current DataFrame.
- Reads dependent columns when needed (e.g. country for currency lookup).
- Returns a `pd.Series` (single column) or `pd.DataFrame` (multi-column).
- Uses Faker for personal/identity data and RNG providers for numeric /
  timestamp data.

## Static Data

### `country_static_data`

Manages two reference datasets:

- **Country-currency map**: Maps ISO 3166-1 alpha-2 codes to ISO 4217
  currency codes via the `babel` library.
- **Currency exchange rates**: Pairwise rates for supported currencies,
  fetched from `CurrencyConverter` API at a fixed snapshot date
  (2019-01-01) and cached to a local CSV to avoid repeated network calls.

Supported countries: US, TR, AT, IE, PL, PT, GB, FR, IN.

### `field_static_data`

Constants for categorical fields:

- `PAYMENT_STATUS`: `(PENDING, PROCESSING, COMPLETED, FAILED, CANCELLED)`
- `ACCOUNT_TYPES`: `(SAVINGS, CHECKING, BUSINESS)`
- Prefix constants: `PAYMENT_CREDITOR_PREFIX`, `PAYMENT_DEBTOR_PREFIX`
