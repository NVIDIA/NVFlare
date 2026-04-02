"""Tests for the anomaly injection framework (__init__.py orchestration)."""

import numpy as np
import pandas as pd
import pytest

from data_generation.anomaly_transformers import (
    ANOMALY_TYPES,
    add_fraud_columns,
    inject,
    inject_all,
    apply_fraud_with_probability,
    Type1Config,
    Type2Config,
)


@pytest.fixture()
def fraud_ready_df() -> pd.DataFrame:
    """A DataFrame with fraud columns already added."""
    rng = np.random.default_rng(99)
    n = 200
    acct_types = rng.choice(["SAVINGS", "CHECKING", "BUSINESS"], size=n)
    now_ts = 1_775_000_000.0
    df = pd.DataFrame(
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
            "DEBITOR_ACCOUNT_CREATE_TIMESTAMP": rng.uniform(
                now_ts - 5 * 365 * 86400, now_ts - 86400, size=n
            ),
            "DEBITOR_ACCOUNT_LAST_ACTIVITY_TIMESTAMP": rng.uniform(
                now_ts - 30 * 86400, now_ts - 86400, size=n
            ),
            "DEBITOR_ACCOUNT_ACTIVITY_EVENTS_PAST_30D": rng.integers(1, 500, size=n),
        }
    )
    add_fraud_columns(df)
    return df


@pytest.fixture()
def type1_config() -> Type1Config:
    return Type1Config(nor_e_low=-4.5, nor_e_high=-4.5, sor_w_low=-4.5, sor_w_high=-4.5)


@pytest.fixture()
def type2_config() -> Type2Config:
    return Type2Config(
        personal_mean=75_000, personal_sigma=5_000,
        business_mean=240_000, business_sigma=15_000,
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_all_four_types_registered(self):
        assert set(ANOMALY_TYPES.keys()) == {"type1", "type2", "type3", "type4"}

    def test_values_are_callable(self):
        for fn in ANOMALY_TYPES.values():
            assert callable(fn)


# ---------------------------------------------------------------------------
# add_fraud_columns
# ---------------------------------------------------------------------------


class TestAddFraudColumns:
    def test_adds_expected_columns(self):
        df = pd.DataFrame({"A": [1, 2, 3]})
        add_fraud_columns(df)
        assert "FRAUD_FLAG" in df.columns
        for i in range(1, 5):
            assert f"TYPE{i}_ANOMALY" in df.columns

    def test_all_initial_values_clean(self):
        df = pd.DataFrame({"A": [1, 2, 3]})
        add_fraud_columns(df)
        assert (df["FRAUD_FLAG"] == 0).all()
        for i in range(1, 5):
            assert not df[f"TYPE{i}_ANOMALY"].any()


# ---------------------------------------------------------------------------
# inject (single type)
# ---------------------------------------------------------------------------


class TestInject:
    def test_sets_fraud_flag(self, fraud_ready_df, type1_config):
        inject(
            fraud_ready_df, "type1",
            fraudulent_frac=0.1, random_state=42,
            fraud_overlap_frac=0.0, config=type1_config,
        )
        assert (fraud_ready_df["FRAUD_FLAG"] == 1).sum() > 0

    def test_respects_fraction(self, fraud_ready_df, type1_config):
        inject(
            fraud_ready_df, "type1",
            fraudulent_frac=0.1, random_state=42,
            fraud_overlap_frac=0.0, config=type1_config,
        )
        fraud_count = (fraud_ready_df["FRAUD_FLAG"] == 1).sum()
        # ~10% of 200 = ~20 rows, allow reasonable tolerance
        assert 5 <= fraud_count <= 40

    def test_unknown_type_raises(self, fraud_ready_df):
        with pytest.raises(KeyError):
            inject(fraud_ready_df, "type99", fraudulent_frac=0.1)

    def test_returns_same_dataframe(self, fraud_ready_df, type1_config):
        result = inject(
            fraud_ready_df, "type1",
            fraudulent_frac=0.1, config=type1_config,
        )
        assert result is fraud_ready_df


# ---------------------------------------------------------------------------
# inject_all
# ---------------------------------------------------------------------------


class TestInjectAll:
    def test_multiple_types(self, fraud_ready_df, type1_config, type2_config):
        configs = {
            "type1": {"config": type1_config},
            "type2": {"config": type2_config},
        }
        inject_all(
            fraud_ready_df,
            anomaly_types=["type1", "type2"],
            configs=configs,
            seed=42,
        )
        assert fraud_ready_df["TYPE1_ANOMALY"].any()
        assert fraud_ready_df["TYPE2_ANOMALY"].any()
        assert (fraud_ready_df["FRAUD_FLAG"] == 1).sum() > 0

    def test_no_config_types(self, fraud_ready_df):
        inject_all(
            fraud_ready_df,
            anomaly_types=["type3", "type4"],
            configs={},
            seed=42,
        )
        assert fraud_ready_df["TYPE3_ANOMALY"].any()
        assert fraud_ready_df["TYPE4_ANOMALY"].any()

    def test_deterministic_with_same_seed(self, fraud_ready_df):
        df1 = fraud_ready_df.copy()
        df2 = fraud_ready_df.copy()
        inject_all(df1, ["type3", "type4"], configs={}, seed=7)
        inject_all(df2, ["type3", "type4"], configs={}, seed=7)
        pd.testing.assert_frame_equal(df1, df2)


# ---------------------------------------------------------------------------
# apply_fraud_with_probability
# ---------------------------------------------------------------------------


class TestApplyFraudWithProbability:
    def test_prob_1_keeps_all_fraud(self, fraud_ready_df, type1_config):
        inject(
            fraud_ready_df, "type1",
            fraudulent_frac=0.2, config=type1_config,
        )
        n_fraud_before = (fraud_ready_df["FRAUD_FLAG"] == 1).sum()
        apply_fraud_with_probability(fraud_ready_df, prob=1.0)
        n_fraud_after = (fraud_ready_df["FRAUD_FLAG"] == 1).sum()
        assert n_fraud_after == n_fraud_before

    def test_prob_0_removes_all_fraud(self, fraud_ready_df, type1_config):
        inject(
            fraud_ready_df, "type1",
            fraudulent_frac=0.2, config=type1_config,
        )
        apply_fraud_with_probability(fraud_ready_df, prob=0.0)
        assert (fraud_ready_df["FRAUD_FLAG"] == 1).sum() == 0

    def test_prob_between_reduces_fraud(self, fraud_ready_df, type1_config):
        inject(
            fraud_ready_df, "type1",
            fraudulent_frac=0.5, config=type1_config,
        )
        n_before = (fraud_ready_df["FRAUD_FLAG"] == 1).sum()
        apply_fraud_with_probability(fraud_ready_df, prob=0.5)
        n_after = (fraud_ready_df["FRAUD_FLAG"] == 1).sum()
        assert n_after < n_before
