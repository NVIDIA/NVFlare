"""Anomaly Type 3 — Stale Account Activity.

Pushes the debitor's last-activity timestamp 90-180 days before the payment,
simulating a dormant account suddenly initiating a transaction.
"""

import numpy as np
import pandas as pd

# Choices for the random time offset components.
_DAY_CHOICES = np.array([90, 120, 150, 180])
_HOUR_CHOICES = np.array([0, 1, 2, 3, 4, 5])
_MINUTE_CHOICES = np.array([0, 1, 4, 9, 16, 25])
_SECOND_CHOICES = np.array([1, 10, 20, 30, 40, 50])


def apply(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Apply type-3 anomaly to the given DataFrame rows (in-place).

    For each row, sets ``DEBITOR_ACCOUNT_LAST_ACTIVITY_TIMESTAMP`` to 90-180
    days before ``PAYMENT_INIT_TIMESTAMP``, clamped to not precede account
    creation.

    Args:
        df:   DataFrame slice containing the rows to mutate.
        seed: RNG seed for reproducibility.

    Returns:
        The mutated DataFrame (same object, modified in-place).
    """
    rng = np.random.default_rng(seed)
    n = len(df)

    payment_ts = df["PAYMENT_INIT_TIMESTAMP"].to_numpy(dtype=float)

    days = rng.choice(_DAY_CHOICES, size=n).astype(float)
    hours = rng.choice(_HOUR_CHOICES, size=n).astype(float)
    minutes = rng.choice(_MINUTE_CHOICES, size=n).astype(float)
    seconds = rng.choice(_SECOND_CHOICES, size=n).astype(float)

    offset_seconds = days * 86400 + hours * 3600 + minutes * 60 + seconds
    stale_ts = payment_ts - offset_seconds

    # Clamp: last activity cannot precede account creation
    create_ts = df["DEBITOR_ACCOUNT_CREATE_TIMESTAMP"].to_numpy(dtype=float)
    df["DEBITOR_ACCOUNT_LAST_ACTIVITY_TIMESTAMP"] = np.maximum(stale_ts, create_ts)

    df["TYPE3_ANOMALY"] = True
    return df
