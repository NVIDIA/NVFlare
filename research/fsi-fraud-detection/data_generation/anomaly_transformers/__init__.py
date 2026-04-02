"""Anomaly transformers and injection framework.

Each anomaly type is implemented as a module with an ``apply(df, ...)`` function
that mutates a DataFrame slice in bulk using vectorised operations.

The injection framework (``inject``, ``inject_all``) handles row sampling,
overlap control, and fraud-flag management.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from data_generation.anomaly_transformers import type1, type2, type3, type4
from data_generation.anomaly_transformers.type1 import Type1Config
from data_generation.anomaly_transformers.type2 import Type2Config

# Registry mapping anomaly-type name → apply function.
ANOMALY_TYPES: dict[str, Callable[..., pd.DataFrame]] = {
    "type1": type1.apply,
    "type2": type2.apply,
    "type3": type3.apply,
    "type4": type4.apply,
}

# Columns added by the injection framework.
FRAUD_COLUMNS = [
    "FRAUD_FLAG",
    "TYPE1_ANOMALY",
    "TYPE2_ANOMALY",
    "TYPE3_ANOMALY",
    "TYPE4_ANOMALY",
]


def add_fraud_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add fraud-tracking columns to a DataFrame (in-place)."""
    df["FRAUD_FLAG"] = 0
    df["TYPE1_ANOMALY"] = False
    df["TYPE2_ANOMALY"] = False
    df["TYPE3_ANOMALY"] = False
    df["TYPE4_ANOMALY"] = False
    return df


def _sample_indices(
    df: pd.DataFrame,
    fraudulent_frac: float = 0.3,
    random_state: int = 42,
    fraud_overlap_frac: float = 0.0,
) -> pd.Index:
    """Sample row indices with controlled overlap between anomaly types."""
    existing_fraud = df[df["FRAUD_FLAG"] == 1]
    existing_clean = df[df["FRAUD_FLAG"] == 0]

    if fraud_overlap_frac <= 0:
        return existing_clean.sample(
            frac=fraudulent_frac,
            random_state=random_state,
        ).index

    total = int(np.ceil(len(df) * fraudulent_frac))
    n_fraud = min(int(np.ceil(total * fraud_overlap_frac)), len(existing_fraud))
    n_clean = min(total - n_fraud, len(existing_clean))

    fraud_sample = existing_fraud.sample(n=n_fraud, random_state=random_state)
    clean_sample = existing_clean.sample(n=n_clean, random_state=random_state)
    return pd.concat([fraud_sample, clean_sample]).index


def inject(
    df: pd.DataFrame,
    anomaly_type: str,
    fraudulent_frac: float = 0.3,
    random_state: int = 42,
    fraud_overlap_frac: float = 0.1,
    anomaly_seed: int = 42,
    **kwargs: Any,
) -> pd.DataFrame:
    """Inject a single anomaly type into a DataFrame.

    Samples a fraction of rows, applies the anomaly transformer in bulk,
    and sets ``FRAUD_FLAG = 1`` on affected rows.

    Args:
        df:                 The full dataset DataFrame (mutated in-place).
        anomaly_type:       Key into ``ANOMALY_TYPES`` (e.g. ``"type1"``).
        fraudulent_frac:    Fraction of rows to turn anomalous.
        random_state:       Seed for row-sampling reproducibility.
        fraud_overlap_frac: Fraction of already-fraudulent rows eligible for
                            re-sampling (enables multi-type anomalies).
        anomaly_seed:       Seed passed to the anomaly transformer's RNG.
        **kwargs:           Extra arguments forwarded to the transformer
                            (e.g. ``config`` for type1/type2).

    Returns:
        The mutated DataFrame.
    """
    idx = _sample_indices(df, fraudulent_frac, random_state, fraud_overlap_frac)
    if idx.empty:
        return df

    apply_fn = ANOMALY_TYPES[anomaly_type]
    # Extract the slice, apply the transformer, write back
    subset = df.loc[idx].copy()
    subset = apply_fn(subset, seed=anomaly_seed, **kwargs)
    df.loc[idx] = subset
    df.loc[idx, "FRAUD_FLAG"] = 1
    return df


def inject_all(
    df: pd.DataFrame,
    anomaly_types: list[str],
    configs: dict[str, Any],
    fraudulent_frac: float | None = None,
    fraud_overlap_frac: float = 0.1,
    seed: int = 42,
) -> pd.DataFrame:
    """Inject multiple anomaly types sequentially.

    Args:
        df:                 The full dataset DataFrame (mutated in-place).
        anomaly_types:      List of anomaly type keys (e.g. ``["type1", "type2"]``).
        configs:            Mapping of anomaly type → config/kwargs dict.
                            For type1: ``{"config": Type1Config(...)}``.
                            For type2: ``{"config": Type2Config(...)}``.
                            Type3 and type4 need no config.
        fraudulent_frac:    Fraction per type.  If None, drawn randomly
                            from U(0.001, 0.01) per type (mimics original notebook).
        fraud_overlap_frac: Overlap fraction (see ``inject``).
        seed:               Base seed; each type gets ``seed + i`` for variety.

    Returns:
        The mutated DataFrame.
    """
    rng = np.random.default_rng(seed)
    single_rule = len(anomaly_types) == 1
    for i, atype in enumerate(anomaly_types):
        frac = fraudulent_frac or float(rng.uniform(0.001, 0.01) if single_rule else rng.uniform(0.001, 0.005))
        rs = int(rng.integers(20, 50))
        kwargs = configs.get(atype, {})
        df = inject(
            df,
            atype,
            fraudulent_frac=frac,
            random_state=rs,
            fraud_overlap_frac=fraud_overlap_frac,
            anomaly_seed=seed + i,
            **kwargs,
        )
    return df


def apply_fraud_with_probability(
    df: pd.DataFrame,
    prob: float = 0.9,
    random_state: int = 38,
) -> pd.DataFrame:
    """Randomly un-flag a fraction of fraud rows to add label noise.

    With probability ``1 - prob``, a fraud row has its ``FRAUD_FLAG`` reset to 0
    while keeping the anomalous feature values.  This creates "hard negatives"
    that confuse naive classifiers.

    Args:
        df:           DataFrame with ``FRAUD_FLAG`` column.
        prob:         Probability of keeping the fraud flag (default 0.9).
        random_state: Seed for reproducibility.

    Returns:
        The mutated DataFrame.
    """
    if prob >= 1:
        return df
    fraud_idx = (
        df[df["FRAUD_FLAG"] == 1]
        .sample(
            frac=(1 - prob),
            random_state=random_state,
        )
        .index
    )
    df.loc[fraud_idx, "FRAUD_FLAG"] = 0
    return df
