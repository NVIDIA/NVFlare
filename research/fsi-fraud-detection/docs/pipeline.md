# End-to-End Generation Pipeline

## Pipeline Stages

The dataset generation pipeline executes the following stages for each
dataset within each site:

```txt
Load Site YAML --> Build Providers --> Register Attributes & Dependencies
       |
       v
  Generate Normal Data (resolve attributes column-by-column)
       |
       v
  Shuffle Rows (random_state=dataset_seed)
       |
       v
  Add Fraud Columns (FRAUD_FLAG, TYPE{1-4}_ANOMALY)
       |
       v
  Inject Anomalies (sequential per fraud type)
       |
       v
  Apply Fraud Probability Thinning (random_state=dataset_seed)
       |
       v
  Write CSV
```

### 1. Configuration Loading

`load_site_config()` reads `config/{site}.yml` and extracts two top-level
keys:

- `anomaly_generation_config.field` -- distribution parameters for tower
  coordinates, normal/anomalous amounts, and perturbation factors.
- `dataset_generation_config` -- a list of dataset specifications (fraud
  rules, row counts, apply probability, filename labels, overlap fractions).

### 2. Provider Construction

`build_providers(seed)` instantiates three provider types, each seeded for
reproducibility:

| Provider                          | Wraps                     | Purpose                                                       |
| --------------------------------- | ------------------------- | ------------------------------------------------------------- |
| `FakerSyntheticDataProvider`      | `faker.Faker`             | Personal data (names, addresses, IBANs, ...)                  |
| `RandomChoiceDataProvider`        | `RandomChoice` RNG        | Discrete categorical sampling (gender, country, account type) |
| `UniformDistributionDataProvider` | `UniformDistribution` RNG | Continuous uniform sampling (tower perturbation)              |

Providers are recreated for each dataset with a unique seed
(`seed + ds_idx * 100 + (i - 1)`) to ensure RNG independence between
datasets.

### 3. Attribute Registration

`build_graph_and_providers()` registers all dataset attributes and their
inter-column dependencies using three composable functions:

1. `get_per_participant_attributes()` -- per-party columns for DEBITOR and
   CREDITOR (personal info, addresses, DOB, geo, tower coords, timestamps,
   account activity).
2. `get_payment_core_attributes()` -- payment-level columns (ID, init
   timestamp, last update timestamp, status).
3. `get_payment_amount_attributes()` -- exchange rates and amounts.

Each attribute is a descriptor that declares its output column name(s), the
columns it depends on, and a provider callable that generates values. The
provider type each attribute requires is resolved automatically via
`typing.get_type_hints()` on its callable.

See [data-generation.md](data-generation.md) for the full design.

### 4. Normal Data Generation

`dataset.generate()` resolves attribute dependencies (via topological sort)
and iterates through attributes in a valid generation order. Each
attribute's `emit()` method invokes its bound provider callable, which
produces a `pd.Series` (single column) or `pd.DataFrame` (multi-column
group) for the full batch of rows at once. This column-at-a-time,
vectorised approach avoids per-row iteration and leverages NumPy/pandas
batch operations for performance.

### 5. Row Shuffle

Rows are shuffled with `df.sample(frac=1, random_state=dataset_seed)` followed
by `reset_index(drop=True)`. The random state is the per-dataset seed, so each
dataset gets a distinct shuffle.

### 6. Fraud Column Initialisation

`add_fraud_columns(df)` appends five columns:

- `FRAUD_FLAG` (int, default 0)
- `TYPE1_ANOMALY` through `TYPE4_ANOMALY` (bool, default False)

### 7. Anomaly Injection

`inject_all()` iterates through the `fraud_insertion_rule_stack` and for each
fraud type:

1. Samples a `fraud_row_frac` from U(0.001, 0.01) if a single rule, or
   U(0.001, 0.005) if multiple rules.
2. Selects row indices via `_sample_indices()` with overlap control.
3. Applies the anomaly transformer to the selected subset.
4. Sets `FRAUD_FLAG = 1` on affected rows.

See [anomaly-injection.md](anomaly-injection.md) for details on each fraud
type and the overlap algorithm.

### 8. Probability Thinning

`apply_fraud_with_probability(df, prob, random_state=dataset_seed)` samples
`(1 - prob)` fraction of fraud rows and resets their `FRAUD_FLAG` to 0,
while keeping anomalous feature values intact. This creates "hard negatives"
that retain fraud-like data patterns without the fraud label. The
`random_state` is the per-dataset seed for consistent, per-dataset
reproducibility.

### 9. CSV Output

Each dataset is written to:

```shell
{output_dir}/{site_name}/{site_name}_[{fraud_types}]_[app_frac_{prob}]_[{overlap}]_[{label}_{i}].csv
```

Where:

- `{fraud_types}` = underscore-joined rule stack (e.g. `type1_type2_type3`),
  or `no_fraud`
- `{prob}` = apply probability (e.g. `0.9`)
- `{overlap}` = `pct_overlap_{N}` if > 0, else `no_overlap`
- `{label}_{i}` = dataset label and 1-based repetition index (e.g. `train_1`)

---

## CLI Reference

```shell
usage: main.py [-h] [-s SITE] [-o OUTPUT_DIR] [-S SEED]
               [-c | --checksum | --no-checksum]
               [-U | --generate-universal-set | --no-generate-universal-set]
               [-F UNIVERSAL_SET_FILE_PATH]
               [-C UNIVERSAL_SET_SAMPLE_ROW_COUNT]

Generate synthetic payment datasets with anomaly injection.

options:
  -h, --help                        show this help message and exit
  -s, --site SITE                   Site name to generate data for (repeatable).
                                    Maps to config/{site}.yml. If omitted, all
                                    sites in config/ are used.
  -o, --output-dir DIR              Root directory for CSV output (default: output/).
  -S, --seed SEED                   Global RNG seed for reproducibility (default: 42).
  -c, --checksum / --no-checksum    Write SHA256 checksums for produced files into
                                    each site directory as checksum_YYYYMMDD_HHMMSS.csv
                                    (default: enabled).
  -U, --generate-universal-set /    Combine all *scaling* CSVs across site directories
      --no-generate-universal-set   into a single dataset after generation (default: enabled).
  -F, --universal-set-file-path     Output file path for the universal scaling dataset.
                                    Relative paths are resolved against OUTPUT_DIR.
                                    (default: {output_dir}/universal_scaling_datasets_all_banks.csv)
  -C, --universal-set-sample-row-count
                                    Max rows sampled per site for the universal dataset.
                                    Use 0 to disable sampling (default: 10000).
```

### Examples

```bash
# Generate for all sites with default seed
uv run main.py

# Generate for two specific sites
uv run main.py -s siteA -s siteB -o datasets/

# Custom seed for reproducibility experiments
uv run main.py -s siteB -S 123 -o output/

# Skip checksums and universal scaling dataset
uv run main.py -o datasets/ --no-checksum --no-generate-universal-set

# Custom universal scaling dataset file and sample size
uv run main.py -o datasets/ -F datasets/scaling_all.csv -C 5000
```

### Post-Generation Outputs

Beyond the per-dataset CSVs, each run optionally produces:

**Checksums** (per site, `-c` flag, default on):

```text
{output_dir}/{site_name}/checksum_YYYYMMDD_HHMMSS.csv
```

Contains `File` and `SHA256` columns, one row per dataset generated in the
current run. Previous checksum files are not modified.

**Universal scaling dataset** (`-U` flag, default on):

```text
{output_dir}/universal_scaling_datasets_all_banks.csv  # default path
```

Combines all `*scaling*` CSVs found across `{output_dir}/*/` (excluding
`archive/` directories). Each source row is prefixed with a `SITE` column
identifying its origin. Up to `-C` rows are sampled per site (default 10 000;
use `0` for no limit).

### Seed Derivation

Each dataset receives a deterministic seed to ensure reproducibility while
avoiding correlated RNG streams:

```python
dataset_seed = global_seed + (dataset_config_index * 100) + (repetition_index - 1)
```

This seed is used for:

- Provider construction (Faker and RNG instances)
- Row shuffle (`random_state` in `df.sample()`)
- Anomaly injection (`inject_all` base seed)
- Probability thinning (`random_state` in `apply_fraud_with_probability`)

Using the same derived seed for all RNG-dependent operations within a dataset
ensures fully deterministic output with no hidden fixed constants.
