# Configuration Reference

## File Layout

Each site's configuration is a YAML file at `config/{site_name}.yml`. The
CLI discovers available sites by globbing `config/*.yml`. The filename stem
(without extension) becomes the site name used in output paths and CSV
filenames.

## Schema

A site config contains two top-level keys:

```yaml
anomaly_generation_config:
  field:
    # Distribution parameters for data generation and anomaly injection
    ...

dataset_generation_config:
  # List of dataset specifications
  - ...
```

---

## `anomaly_generation_config.field`

Defines distribution parameters used during both normal data generation
(tower coords, amounts) and anomaly injection (perturbation factors,
anomalous amounts).

### Tower Coordinates

```yaml
tower_lat:
  distributions:
    - key: uniform
      low: <float>    # e.g. -0.75
      high: <float>   # e.g. 1.25

tower_long:
  distributions:
    - key: uniform
      low: <float>    # e.g. -1.5
      high: <float>   # e.g. 2.0
```

Used by `UniformDistributionDataProvider` to generate tower latitude and
longitude perturbations around each participant's geo-coordinates.

### Anomalous Tower Perturbation (Type 1)

```yaml
anomalous_tower_NorE_perturbation:
  distributions:
    - key: uniform
      low: <float>    # e.g. -4.5
      high: <float>   # e.g. -4.5

anomalous_tower_SorW_perturbation:
  distributions:
    - key: uniform
      low: <float>
      high: <float>
```

Fed into `Type1Config` to control how far anomalous tower coordinates
deviate. Negative values push the perturbation factor to generate
coordinates far from the physical location.

### Normal Transaction Amounts

```yaml
normal_personal_acc_amount:
  distributions:
    - key: lognormal
      desired_mean: <float>   # e.g. 20000
      sigma: <float>          # e.g. 7500

normal_business_acc_amount:
  distributions:
    - key: lognormal
      desired_mean: <float>   # e.g. 80000
      sigma: <float>          # e.g. 10000
```

Parameters for the log-normal distribution of normal (non-anomalous)
transaction amounts, split by account type.

### Anomalous Transaction Amounts (Type 2)

```yaml
anomalous_personal_acc_amount:
  distributions:
    - key: lognormal
      desired_mean: <float>   # e.g. 75000
      sigma: <float>          # e.g. 5000

anomalous_business_acc_amount:
  distributions:
    - key: lognormal
      desired_mean: <float>   # e.g. 240000
      sigma: <float>          # e.g. 15000
```

Fed into `Type2Config` for anomalous amount generation. The desired mean
and sigma describe the target arithmetic distribution; internal conversion
to log-space parameters is handled by `get_lognormal_params()`.

---

## `dataset_generation_config`

A list of dataset specification objects. Each entry produces one or more CSV
files.

### Entry Schema

```yaml
- fraud_insertion_rule_stack: [type1, type2, type3]  # required
  num_datasets: 1                                     # default: 1
  max_num_rows: 100000                                # default: 3000
  apply_probability: 0.9                              # default: 1.0
  fname_label: "train"                                # default: ""
  fraud_overlap_frac: 0.1                             # default: -1
```

### Field Reference

| Field                        | Type      | Default    | Description                                                                                                                                                   |
| ---------------------------- | --------- | ---------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `fraud_insertion_rule_stack` | list[str] | (required) | Anomaly types to inject, in order. Valid: `type1`, `type2`, `type3`, `type4`.                                                                                 |
| `num_datasets`               | int       | 1          | Number of independent datasets to generate with this specification.                                                                                           |
| `max_num_rows`               | int       | 3000       | Number of rows per dataset.                                                                                                                                   |
| `apply_probability`          | float     | 1.0        | Probability of retaining fraud labels. 1.0 = no thinning; 0.9 = 10% of fraud rows get un-flagged.                                                             |
| `fname_label`                | str       | ""         | Label prefix for filename (e.g. "train", "eval", "scaling").                                                                                                  |
| `fraud_overlap_frac`         | float     | -1         | Controls overlap between anomaly types. <= 0 means no overlap (sample only from clean rows). > 0 means that fraction of already-fraud rows can be re-sampled. |

### Typical Configuration

A site typically defines 6 dataset groups:

1. **Training** (1 dataset, 100k rows) -- uses site-specific anomaly types
   with probability thinning (0.9) and moderate overlap (0.1).
2. **Scaling** (1 dataset, 25k rows) -- uses all four anomaly types with
   no thinning (1.0).
3. **Evaluation** (4 datasets, 25k rows each) -- different anomaly type
   combinations with varying apply probabilities and overlap fractions.

---

## Example

```yaml
anomaly_generation_config:
  field:
    tower_lat:
      distributions:
        - key: uniform
          low: -0.75
          high: 1.25
    tower_long:
      distributions:
        - key: uniform
          low: -1.5
          high: 2
    anomalous_tower_NorE_perturbation:
      distributions:
        - key: uniform
          low: -4.5
          high: -4.5
    anomalous_tower_SorW_perturbation:
      distributions:
        - key: uniform
          low: -4.5
          high: -4.5
    normal_personal_acc_amount:
      distributions:
        - key: lognormal
          desired_mean: 20000
          sigma: 7500
    anomalous_personal_acc_amount:
      distributions:
        - key: lognormal
          desired_mean: 75000
          sigma: 5000
    normal_business_acc_amount:
      distributions:
        - key: lognormal
          desired_mean: 80000
          sigma: 10000
    anomalous_business_acc_amount:
      distributions:
        - key: lognormal
          desired_mean: 240000
          sigma: 15000

dataset_generation_config:
  - fraud_insertion_rule_stack: [type1, type2, type3]
    num_datasets: 1
    max_num_rows: 100_000
    apply_probability: 0.9
    fname_label: "train"
    fraud_overlap_frac: 0.1
  - fraud_insertion_rule_stack: [type1, type2, type3, type4]
    num_datasets: 1
    apply_probability: 1.0
    fname_label: "scaling"
    max_num_rows: 25_000
    fraud_overlap_frac: 0.11
```
