# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for individual anomaly type transformers (type1–type4)."""

import numpy as np
import pandas as pd
import pytest
from data_generation.anomaly_transformers import type1, type2, type3, type4
from data_generation.anomaly_transformers.type1 import Type1Config
from data_generation.anomaly_transformers.type2 import Type2Config

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def base_df() -> pd.DataFrame:
    """A small DataFrame with realistic columns for anomaly transformers."""
    rng = np.random.default_rng(99)
    n = 50
    acct_types = rng.choice(["SAVINGS", "CHECKING", "BUSINESS"], size=n)
    now_ts = 1_775_000_000.0
    return pd.DataFrame(
        {
            "DEBITOR_TOWER_LATITUDE": rng.uniform(30, 50, size=n),
            "DEBITOR_TOWER_LONGITUDE": rng.uniform(-100, -70, size=n),
            "CREDITOR_TOWER_LATITUDE": rng.uniform(30, 50, size=n),
            "CREDITOR_TOWER_LONGITUDE": rng.uniform(-100, -70, size=n),
            "DEBITOR_ACCOUNT_TYPE": acct_types,
            "DEBITOR_AMOUNT": rng.uniform(100, 5000, size=n).round(2),
            "CREDITOR_AMOUNT": rng.uniform(100, 5000, size=n).round(2),
            "DEBITOR_CCY_CREDITOR_CCY_RATE": rng.uniform(0.5, 2.0, size=n).round(4),
            "PAYMENT_INIT_TIMESTAMP": np.full(n, now_ts),
            "DEBITOR_ACCOUNT_CREATE_TIMESTAMP": rng.uniform(now_ts - 5 * 365 * 86400, now_ts - 86400, size=n),
            "DEBITOR_ACCOUNT_LAST_ACTIVITY_TIMESTAMP": rng.uniform(now_ts - 30 * 86400, now_ts - 86400, size=n),
            "DEBITOR_ACCOUNT_ACTIVITY_EVENTS_PAST_30D": rng.integers(1, 500, size=n),
        }
    )


@pytest.fixture()
def type1_config() -> Type1Config:
    return Type1Config(nor_e_low=-4.5, nor_e_high=-4.5, sor_w_low=-4.5, sor_w_high=-4.5)


@pytest.fixture()
def type2_config() -> Type2Config:
    return Type2Config(
        personal_mean=75_000,
        personal_sigma=5_000,
        business_mean=240_000,
        business_sigma=15_000,
    )


# ---------------------------------------------------------------------------
# Type 1 — Geo / Tower mismatch
# ---------------------------------------------------------------------------


class TestType1:
    def test_sets_anomaly_flag(self, base_df, type1_config):
        result = type1.apply(base_df.copy(), config=type1_config, seed=42)
        assert result["TYPE1_ANOMALY"].all()

    def test_tower_coords_changed(self, base_df, type1_config):
        original = base_df.copy()
        result = type1.apply(base_df.copy(), config=type1_config, seed=42)
        for col in [
            "DEBITOR_TOWER_LATITUDE",
            "DEBITOR_TOWER_LONGITUDE",
            "CREDITOR_TOWER_LATITUDE",
            "CREDITOR_TOWER_LONGITUDE",
        ]:
            assert not np.allclose(original[col], result[col]), f"{col} unchanged"

    def test_deterministic_with_same_seed(self, base_df, type1_config):
        r1 = type1.apply(base_df.copy(), config=type1_config, seed=7)
        r2 = type1.apply(base_df.copy(), config=type1_config, seed=7)
        pd.testing.assert_frame_equal(r1, r2)

    def test_different_seeds_diverge(self, base_df, type1_config):
        r1 = type1.apply(base_df.copy(), config=type1_config, seed=1)
        r2 = type1.apply(base_df.copy(), config=type1_config, seed=2)
        assert not np.allclose(r1["DEBITOR_TOWER_LATITUDE"], r2["DEBITOR_TOWER_LATITUDE"])

    def test_from_site_fields(self):
        fields = {
            "anomalous_tower_NorE_perturbation": {"distributions": [{"low": -3.0, "high": -2.0}]},
            "anomalous_tower_SorW_perturbation": {"distributions": [{"low": -5.0, "high": -4.0}]},
        }
        cfg = Type1Config.from_site_fields(fields)
        assert cfg.nor_e_low == -3.0
        assert cfg.nor_e_high == -2.0
        assert cfg.sor_w_low == -5.0
        assert cfg.sor_w_high == -4.0


# ---------------------------------------------------------------------------
# Type 2 — Young account + high amount
# ---------------------------------------------------------------------------


class TestType2:
    def test_sets_anomaly_flag(self, base_df, type2_config):
        result = type2.apply(base_df.copy(), config=type2_config, seed=42)
        assert result["TYPE2_ANOMALY"].all()

    def test_account_create_near_payment_init(self, base_df, type2_config):
        result = type2.apply(base_df.copy(), config=type2_config, seed=42)
        payment_ts = result["PAYMENT_INIT_TIMESTAMP"].to_numpy()
        create_ts = result["DEBITOR_ACCOUNT_CREATE_TIMESTAMP"].to_numpy()
        diff_hours = (payment_ts - create_ts) / 3600
        # Should be within ~6 hours (max offset: 5h + 25min + 50s)
        assert (diff_hours >= 0).all()
        assert (diff_hours < 7).all()

    def test_amounts_positive_and_large(self, base_df, type2_config):
        result = type2.apply(base_df.copy(), config=type2_config, seed=42)
        assert (result["DEBITOR_AMOUNT"] > 0).all()
        # Anomalous amounts should generally be larger than normal (mean=75k/240k)
        assert result["DEBITOR_AMOUNT"].median() > 10_000

    def test_creditor_amount_uses_exchange_rate(self, base_df, type2_config):
        result = type2.apply(base_df.copy(), config=type2_config, seed=42)
        expected = np.round(
            result["DEBITOR_AMOUNT"].to_numpy() * result["DEBITOR_CCY_CREDITOR_CCY_RATE"].to_numpy(),
            2,
        )
        np.testing.assert_array_almost_equal(result["CREDITOR_AMOUNT"].to_numpy(), expected, decimal=2)

    def test_deterministic_with_same_seed(self, base_df, type2_config):
        r1 = type2.apply(base_df.copy(), config=type2_config, seed=42)
        r2 = type2.apply(base_df.copy(), config=type2_config, seed=42)
        pd.testing.assert_frame_equal(r1, r2)

    def test_from_site_fields(self):
        fields = {
            "anomalous_personal_acc_amount": {"distributions": [{"desired_mean": 60_000, "sigma": 4_000}]},
            "anomalous_business_acc_amount": {"distributions": [{"desired_mean": 200_000, "sigma": 12_000}]},
        }
        cfg = Type2Config.from_site_fields(fields)
        assert cfg.personal_mean == 60_000
        assert cfg.business_sigma == 12_000


# ---------------------------------------------------------------------------
# Type 3 — Stale account activity
# ---------------------------------------------------------------------------


class TestType3:
    def test_sets_anomaly_flag(self, base_df):
        result = type3.apply(base_df.copy(), seed=42)
        assert result["TYPE3_ANOMALY"].all()

    def test_pushes_activity_back_90_to_180_days(self, base_df):
        # Use accounts created far enough in the past so clamping doesn't interfere
        df = base_df.copy()
        df["DEBITOR_ACCOUNT_CREATE_TIMESTAMP"] = df["PAYMENT_INIT_TIMESTAMP"] - 365 * 86400
        result = type3.apply(df, seed=42)
        payment_ts = result["PAYMENT_INIT_TIMESTAMP"].to_numpy()
        activity_ts = result["DEBITOR_ACCOUNT_LAST_ACTIVITY_TIMESTAMP"].to_numpy()
        diff_days = (payment_ts - activity_ts) / 86400
        # Should be at least 90 days back (minus a few hours of sub-day offset)
        assert (diff_days >= 89).all()
        # Should not exceed 180 days + ~6 hours
        assert (diff_days <= 181).all()

    def test_clamped_to_account_creation(self, base_df):
        # Force account creation very recently so clamping is exercised
        df = base_df.copy()
        df["DEBITOR_ACCOUNT_CREATE_TIMESTAMP"] = df["PAYMENT_INIT_TIMESTAMP"] - 10
        result = type3.apply(df, seed=42)
        create_ts = result["DEBITOR_ACCOUNT_CREATE_TIMESTAMP"].to_numpy()
        activity_ts = result["DEBITOR_ACCOUNT_LAST_ACTIVITY_TIMESTAMP"].to_numpy()
        assert (activity_ts >= create_ts).all()

    def test_deterministic_with_same_seed(self, base_df):
        r1 = type3.apply(base_df.copy(), seed=42)
        r2 = type3.apply(base_df.copy(), seed=42)
        pd.testing.assert_frame_equal(r1, r2)


# ---------------------------------------------------------------------------
# Type 4 — Unusually high activity events
# ---------------------------------------------------------------------------


class TestType4:
    def test_sets_anomaly_flag(self, base_df):
        result = type4.apply(base_df.copy(), seed=42)
        assert result["TYPE4_ANOMALY"].all()

    def test_business_events_in_range(self, base_df):
        df = base_df.copy()
        df["DEBITOR_ACCOUNT_TYPE"] = "BUSINESS"
        result = type4.apply(df, seed=42)
        events = result["DEBITOR_ACCOUNT_ACTIVITY_EVENTS_PAST_30D"].to_numpy()
        assert (events >= 1_000_000).all()
        assert (events < 5_000_001).all()

    def test_checking_events_in_range(self, base_df):
        df = base_df.copy()
        df["DEBITOR_ACCOUNT_TYPE"] = "CHECKING"
        result = type4.apply(df, seed=42)
        events = result["DEBITOR_ACCOUNT_ACTIVITY_EVENTS_PAST_30D"].to_numpy()
        assert (events >= 525).all()
        assert (events < 2_000).all()

    def test_savings_events_in_range(self, base_df):
        df = base_df.copy()
        df["DEBITOR_ACCOUNT_TYPE"] = "SAVINGS"
        result = type4.apply(df, seed=42)
        events = result["DEBITOR_ACCOUNT_ACTIVITY_EVENTS_PAST_30D"].to_numpy()
        assert (events >= 75).all()
        assert (events < 2_000).all()

    def test_deterministic_with_same_seed(self, base_df):
        r1 = type4.apply(base_df.copy(), seed=42)
        r2 = type4.apply(base_df.copy(), seed=42)
        pd.testing.assert_frame_equal(r1, r2)
