"""Anomaly Type 2 — Young Account + Anomalously High Amount.

Sets the debitor's account creation timestamp to within minutes of the payment
initiation, then replaces the transaction amount with an anomalously high value
drawn from a site-specific log-normal distribution (personal vs. business).
"""

from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd

from data_generation.rng.lognormal_distribution import (
    LogNormalDistribution,
    LogNormalDistributionSamplingConfig,
)


@dataclass
class Type2Config:
    """Configuration for anomalous amount generation.

    Attributes:
        personal_mean:  Desired arithmetic mean for personal account anomalous amounts.
        personal_sigma: Desired arithmetic std dev for personal account anomalous amounts.
        business_mean:  Desired arithmetic mean for business account anomalous amounts.
        business_sigma: Desired arithmetic std dev for business account anomalous amounts.
    """

    personal_mean: float
    personal_sigma: float
    business_mean: float
    business_sigma: float

    @classmethod
    def from_site_fields(cls, fields: dict) -> "Type2Config":
        personal = fields["anomalous_personal_acc_amount"]["distributions"][0]
        business = fields["anomalous_business_acc_amount"]["distributions"][0]
        return cls(
            personal_mean=personal["desired_mean"],
            personal_sigma=personal["sigma"],
            business_mean=business["desired_mean"],
            business_sigma=business["sigma"],
        )


# Choices for random time offsets (hours, minutes, seconds) that push
# account creation close to payment initiation.
_HOUR_CHOICES = np.array([0, 1, 2, 3, 4, 5])
_MINUTE_CHOICES = np.array([0, 1, 4, 9, 16, 25])
_SECOND_CHOICES = np.array([1, 10, 20, 30, 40, 50])


def apply(df: pd.DataFrame, config: Type2Config, seed: int = 42) -> pd.DataFrame:
    """Apply type-2 anomaly to the given DataFrame rows (in-place).

    For each row:
      1. Moves ``DEBITOR_ACCOUNT_CREATE_TIMESTAMP`` to within minutes of
         ``PAYMENT_INIT_TIMESTAMP``.
      2. Replaces ``DEBITOR_AMOUNT`` with a lognormal sample whose parameters
         depend on account type (personal vs business).
      3. Recomputes ``CREDITOR_AMOUNT`` using the exchange rate.

    Args:
        df:     DataFrame slice containing the rows to mutate.
        config: Lognormal distribution parameters from site config.
        seed:   RNG seed for reproducibility.

    Returns:
        The mutated DataFrame (same object, modified in-place).
    """
    rng = np.random.default_rng(seed)
    lognormal_rng = LogNormalDistribution(seed=seed)
    n = len(df)

    # --- 1. Push account creation close to payment initiation ---
    payment_ts = df["PAYMENT_INIT_TIMESTAMP"].to_numpy(dtype=float)
    hours = rng.choice(_HOUR_CHOICES, size=n).astype(float)
    minutes = rng.choice(_MINUTE_CHOICES, size=n).astype(float)
    seconds = rng.choice(_SECOND_CHOICES, size=n).astype(float)
    offset_seconds = hours * 3600 + minutes * 60 + seconds
    df["DEBITOR_ACCOUNT_CREATE_TIMESTAMP"] = payment_ts - offset_seconds

    # --- 2. Generate anomalous amounts by account type ---
    acct_types = df["DEBITOR_ACCOUNT_TYPE"].to_numpy()
    amounts = np.zeros(n, dtype=float)

    personal_mask = np.isin(acct_types, ["SAVINGS", "CHECKING"])
    business_mask = acct_types == "BUSINESS"

    n_personal = int(personal_mask.sum())
    if n_personal > 0:
        cfg = LogNormalDistributionSamplingConfig(
            mean=config.personal_mean,
            std_dev=config.personal_sigma,
        )
        amounts[personal_mask] = np.round(
            cast(np.ndarray, lognormal_rng.sample(sample_config=cfg, size=n_personal)),
            2,
        )

    n_business = int(business_mask.sum())
    if n_business > 0:
        cfg = LogNormalDistributionSamplingConfig(
            mean=config.business_mean,
            std_dev=config.business_sigma,
        )
        amounts[business_mask] = np.round(
            cast(np.ndarray, lognormal_rng.sample(sample_config=cfg, size=n_business)),
            2,
        )

    df["DEBITOR_AMOUNT"] = amounts

    # --- 3. Recompute creditor amount ---
    rates = df["DEBITOR_CCY_CREDITOR_CCY_RATE"].to_numpy(dtype=float)
    df["CREDITOR_AMOUNT"] = np.round(amounts * rates, 2)

    df["TYPE2_ANOMALY"] = True
    return df
