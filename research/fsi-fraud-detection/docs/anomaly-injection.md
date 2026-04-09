# Anomaly Injection Framework

## Overview

The anomaly injection system lives in `data_generation/anomaly_transformers/`
and provides:

1. **Four anomaly type transformers** (type1 through type4), each
   implemented as a vectorised `apply()` function that mutates a DataFrame
   slice in bulk using NumPy array operations rather than per-row iteration.
   This mirrors the vectorised design of the data generation layer.
2. **An injection framework** (`inject`, `inject_all`) that handles row
   sampling, overlap control, and sequential application.
3. **Probability thinning** (`apply_fraud_with_probability`) that
   introduces label noise by un-flagging a fraction of fraud rows.

---

## Anomaly Types

### Type 1 -- Geo/Tower Location Mismatch

**Module**: `type1.py`

Perturbs tower coordinates so they are geographically far from the user's
physical location, simulating a scenario where the payment device is
connecting through a distant cell tower.

**Columns mutated**:

- `DEBITOR_TOWER_LATITUDE`, `DEBITOR_TOWER_LONGITUDE`
- `CREDITOR_TOWER_LATITUDE`, `CREDITOR_TOWER_LONGITUDE`

**Flag set**: `TYPE1_ANOMALY = True`

**Algorithm**: For each coordinate (lat/lon, for each party):

1. Computes the delta to the max and min bounds (90/-90 for lat, 180/-180
   for lon), offset by site-configurable perturbation factors.
2. Generates candidate coordinates in both directions (north/east and
   south/west).
3. If both directions are valid, randomly selects one; otherwise takes the
   valid direction.

**Configuration**: `Type1Config` with `nor_e_low`, `nor_e_high`,
`sor_w_low`, `sor_w_high` -- loaded from
`anomaly_generation_config.field.anomalous_tower_NorE_perturbation` and
`anomalous_tower_SorW_perturbation`.

### Type 2 -- Young Account + High Amount

**Module**: `type2.py`

Sets the debtor's account creation to within minutes of the payment
initiation and replaces the transaction amount with an anomalously high
value, simulating a newly-opened account making a suspiciously large
transaction.

**Columns mutated**:

- `DEBITOR_ACCOUNT_CREATE_TIMESTAMP` (set to minutes before payment)
- `DEBITOR_AMOUNT` (sampled from anomalous log-normal distribution)
- `CREDITOR_AMOUNT` (recomputed using exchange rate)

**Flag set**: `TYPE2_ANOMALY = True`

**Algorithm**:

1. Generates random time offsets (0-5 hours, 0-25 minutes, 1-50 seconds)
   and subtracts from payment init timestamp.
2. Samples anomalous amounts from a log-normal distribution parameterised
   by account type (personal vs. business).
3. Recomputes creditor amount as `debitor_amount * exchange_rate`.

**Configuration**: `Type2Config` with per-account-type log-normal
parameters -- loaded from `anomalous_personal_acc_amount` and
`anomalous_business_acc_amount`.

### Type 3 -- Stale Account Activity

**Module**: `type3.py`

Pushes the debtor's last activity 90-180 days before the payment, simulating
a dormant account suddenly initiating a transaction.

**Columns mutated**:

- `DEBITOR_ACCOUNT_LAST_ACTIVITY_TIMESTAMP`

**Flag set**: `TYPE3_ANOMALY = True`

**Algorithm**:

1. Samples days from {90, 120, 150, 180} plus random hours/minutes/seconds.
2. Subtracts the offset from payment init timestamp.
3. Clamps to not precede account creation timestamp (`np.maximum`).

**Configuration**: None required.

### Type 4 -- Unusually High Activity Events

**Module**: `type4.py`

Inflates the debtor's 30-day activity count well beyond normal ranges,
simulating automated transaction flooding.

**Columns mutated**:

- `DEBITOR_ACCOUNT_ACTIVITY_EVENTS_PAST_30D`

**Flag set**: `TYPE4_ANOMALY = True`

**Ranges by account type**:

| Account Type | Normal Range | Anomalous Range        |
| ------------ | ------------ | ---------------------- |
| BUSINESS     | (varies)     | 1,000,000 -- 5,000,000 |
| CHECKING     | (varies)     | 525 -- 2,000           |
| SAVINGS      | (varies)     | 75 -- 2,000            |

**Configuration**: None required.

---

## Injection Framework

### `add_fraud_columns(df)`

Initialises tracking columns on the DataFrame:

- `FRAUD_FLAG` = 0 (int)
- `TYPE1_ANOMALY` through `TYPE4_ANOMALY` = False (bool)

Must be called before any injection.

### `_sample_indices(df, fraudulent_frac, random_state, fraud_overlap_frac)`

Selects row indices for anomaly application with controlled overlap between
fraud types:

- **`fraud_overlap_frac <= 0`**: Samples only from rows where
  `FRAUD_FLAG == 0` (no overlap with existing fraud).
- **`fraud_overlap_frac > 0`**: Computes a total count
  (`ceil(len(df) * fraudulent_frac)`), then splits between
  already-flagged rows (`ceil(total * overlap_frac)`) and clean rows
  (`total - fraud_count`). Caps each to the available pool size.

Both sub-samples use the same `random_state` for reproducibility.

### `inject(df, anomaly_type, ...)`

Injects a single anomaly type:

1. Calls `_sample_indices()` to select target rows.
2. Copies the selected slice, applies the anomaly transformer's `apply()`
   function, and writes the result back.
3. Sets `FRAUD_FLAG = 1` on affected rows.

Parameters:

- `fraudulent_frac` -- fraction of total rows to make anomalous.
- `random_state` -- seed for index sampling.
- `fraud_overlap_frac` -- overlap control (see above).
- `anomaly_seed` -- seed passed to the transformer's RNG.
- `**kwargs` -- forwarded to the transformer (e.g. `config` for type1/type2).

### `inject_all(df, anomaly_types, configs, ...)`

Iterates through the anomaly type list and calls `inject()` for each:

```python
for i, atype in enumerate(anomaly_types):
    frac = fraudulent_frac or rng.uniform(0.001, 0.01 if single_rule else 0.005)
    rs = int(rng.integers(20, 50))
    inject(df, atype, fraudulent_frac=frac, random_state=rs, ...)
```

Key behaviors:

- **Single-rule stacks**: `fraud_row_frac` sampled from U(0.001, 0.01).
- **Multi-rule stacks**: Capped at U(0.001, 0.005) to prevent total fraud
  fraction from exceeding ~5%.
- Each type receives `anomaly_seed = base_seed + i` for variety.
- Application is sequential -- later fraud types see rows already flagged
  by earlier types, which interacts with the overlap logic.

### `apply_fraud_with_probability(df, prob, random_state=dataset_seed)`

Introduces label noise by un-flagging a random subset of fraud rows:

1. If `prob >= 1`, returns immediately (no thinning).
2. Samples `(1 - prob)` fraction of `FRAUD_FLAG == 1` rows using the
   provided `random_state` (callers pass the per-dataset seed).
3. Resets `FRAUD_FLAG` to 0 on those rows.
4. Anomaly type flags (`TYPE{1-4}_ANOMALY`) are **not** reset -- the feature
   values remain anomalous, creating "hard negatives" that challenge model
   training.

---

## Anomaly Type Registry

The `ANOMALY_TYPES` dict maps string keys to transformer functions:

```python
ANOMALY_TYPES = {
    "type1": type1.apply,
    "type2": type2.apply,
    "type3": type3.apply,
    "type4": type4.apply,
}
```

The `fraud_insertion_rule_stack` in site configs references these keys.
