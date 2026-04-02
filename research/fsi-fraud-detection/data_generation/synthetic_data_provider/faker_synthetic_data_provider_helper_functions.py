"""Vectorised attribute-generation functions for the payment dataset.

Every public function in this module conforms to
``AttributeDataProviderProtocol`` — it accepts a data provider, the
partially-built ``pd.DataFrame``, an optional list of dependent column names,
and arbitrary ``**kwargs``, returning either a ``pd.Series`` (single column)
or a ``pd.DataFrame`` (multi-column group).

Functions that only delegate to Faker iterate row-by-row internally but still
produce a ``pd.Series`` for a uniform downstream interface.  RNG-backed
functions use the ``size`` parameter for true vectorised sampling.
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from data_generation.synthetic_data_provider import (
    FakerSyntheticDataProvider,
    RandomChoiceDataProvider,
    UniformDistributionDataProvider,
)
from data_generation.rng.random_choice import RandomChoiceSamplingConfig
from data_generation.rng.uniform_distribution import UniformDistributionSamplingConfig
from data_generation.static_data import country_static_data
import data_generation.static_data.field_static_data as field_static_data


# ---------------------------------------------------------------------------
# Multi-column generators (return pd.DataFrame)
# ---------------------------------------------------------------------------


def date_of_birth(
    provider: FakerSyntheticDataProvider,
    df: pd.DataFrame,
    dependent_columns: list[str] | None = None,  # type: ignore[unused-argument]
    **kwargs,
) -> pd.DataFrame:
    """Generate date-of-birth columns (year, month, day) via Faker."""
    n = len(df)
    fake = provider.provide()
    years, months, days = [], [], []
    for _ in range(n):
        dob = fake.date_of_birth()
        years.append(dob.year)
        months.append(dob.month)
        days.append(dob.day)
    return pd.DataFrame(
        {"col_0": years, "col_1": months, "col_2": days}, index=df.index
    )


# ---------------------------------------------------------------------------
# Faker-backed single-column generators (return pd.Series)
# ---------------------------------------------------------------------------


def username(
    provider: FakerSyntheticDataProvider,
    df: pd.DataFrame,
    dependent_columns: list[str] | None = None,
    **kwargs,
) -> pd.Series:
    """Generate random usernames via Faker."""
    fake = provider.provide()
    return pd.Series([fake.user_name() for _ in range(len(df))], index=df.index)


def firstname(
    provider: FakerSyntheticDataProvider,
    df: pd.DataFrame,
    dependent_columns: list[str] | None = None,
    **kwargs,
) -> pd.Series:
    """Generate random first names via Faker."""
    fake = provider.provide()
    return pd.Series([fake.first_name() for _ in range(len(df))], index=df.index)


def lastname(
    provider: FakerSyntheticDataProvider,
    df: pd.DataFrame,
    dependent_columns: list[str] | None = None,
    **kwargs,
) -> pd.Series:
    """Generate random last names via Faker."""
    fake = provider.provide()
    return pd.Series([fake.last_name() for _ in range(len(df))], index=df.index)


def email(
    provider: FakerSyntheticDataProvider,
    df: pd.DataFrame,
    dependent_columns: list[str] | None = None,
    **kwargs,
) -> pd.Series:
    """Generate random email addresses via Faker."""
    fake = provider.provide()
    return pd.Series([fake.email() for _ in range(len(df))], index=df.index)


def phone_number(
    provider: FakerSyntheticDataProvider,
    df: pd.DataFrame,
    dependent_columns: list[str] | None = None,
    **kwargs,
) -> pd.Series:
    """Generate random phone numbers via Faker."""
    fake = provider.provide()
    return pd.Series([fake.phone_number() for _ in range(len(df))], index=df.index)


def account_number(
    provider: FakerSyntheticDataProvider,
    df: pd.DataFrame,
    dependent_columns: list[str] | None = None,
    **kwargs,
) -> pd.Series:
    """Generate random IBAN account numbers via Faker."""
    fake = provider.provide()
    return pd.Series([fake.iban() for _ in range(len(df))], index=df.index)


def bic_code(
    provider: FakerSyntheticDataProvider,
    df: pd.DataFrame,
    dependent_columns: list[str] | None = None,
    **kwargs,
) -> pd.Series:
    """Generate random ABA routing numbers via Faker."""
    fake = provider.provide()
    return pd.Series([fake.aba() for _ in range(len(df))], index=df.index)


def ip_address(
    provider: FakerSyntheticDataProvider,
    df: pd.DataFrame,
    dependent_columns: list[str] | None = None,
    **kwargs,
) -> pd.Series:
    """Generate random IPv4 addresses via Faker."""
    fake = provider.provide()
    return pd.Series([fake.ipv4(network=False) for _ in range(len(df))], index=df.index)


def comment(
    provider: FakerSyntheticDataProvider,
    df: pd.DataFrame,
    dependent_columns: list[str] | None = None,
    **kwargs,
) -> pd.Series:
    """Generate random short text comments (max 60 chars) via Faker."""
    fake = provider.provide()
    return pd.Series(
        [fake.text(max_nb_chars=60) for _ in range(len(df))], index=df.index
    )


def building_number(
    provider: FakerSyntheticDataProvider,
    df: pd.DataFrame,
    dependent_columns: list[str] | None = None,
    **kwargs,
) -> pd.Series:
    """Generate random building numbers via Faker."""
    fake = provider.provide()
    return pd.Series([fake.building_number() for _ in range(len(df))], index=df.index)


def street(
    provider: FakerSyntheticDataProvider,
    df: pd.DataFrame,
    dependent_columns: list[str] | None = None,
    **kwargs,
) -> pd.Series:
    """Generate random street names (name + suffix) via Faker."""
    fake = provider.provide()
    return pd.Series(
        [f"{fake.street_name()} {fake.street_suffix()}" for _ in range(len(df))],
        index=df.index,
    )


def city(
    provider: FakerSyntheticDataProvider,
    df: pd.DataFrame,
    dependent_columns: list[str] | None = None,
    **kwargs,
) -> pd.Series:
    """Generate random city names via Faker."""
    fake = provider.provide()
    return pd.Series([fake.city() for _ in range(len(df))], index=df.index)


def state(
    provider: FakerSyntheticDataProvider,
    df: pd.DataFrame,
    dependent_columns: list[str] | None = None,
    **kwargs,
) -> pd.Series:
    """Generate random state/province names via Faker."""
    fake = provider.provide()
    return pd.Series([fake.state() for _ in range(len(df))], index=df.index)


def zipcode(
    provider: FakerSyntheticDataProvider,
    df: pd.DataFrame,
    dependent_columns: list[str] | None = None,
    **kwargs,
) -> pd.Series:
    """Generate random postal codes via Faker."""
    fake = provider.provide()
    return pd.Series([fake.postcode() for _ in range(len(df))], index=df.index)


def payment_id(
    provider: FakerSyntheticDataProvider,
    df: pd.DataFrame,
    dependent_columns: list[str] | None = None,
    **kwargs,
) -> pd.Series:
    """Generate random UUID-based payment IDs via Faker."""
    fake = provider.provide()
    return pd.Series([str(fake.uuid4()) for _ in range(len(df))], index=df.index)


# ---------------------------------------------------------------------------
# RNG-backed single-column generators (return pd.Series)
# ---------------------------------------------------------------------------


def gender(
    provider: RandomChoiceDataProvider,
    df: pd.DataFrame,
    dependent_columns: list[str] | None = None,
    **kwargs,
) -> pd.Series:
    """Randomly sample gender values (M / F / O)."""
    cfg: RandomChoiceSamplingConfig | None = kwargs.get("random_choice_config")
    result = provider.rng.sample("M", "F", "O", sample_config=cfg, size=len(df))
    return pd.Series(result, index=df.index, dtype=str)


def country(
    provider: RandomChoiceDataProvider,
    df: pd.DataFrame,
    dependent_columns: list[str] | None = None,
    **kwargs,
) -> pd.Series:
    """Randomly sample a country code from the supported set."""
    cfg: RandomChoiceSamplingConfig | None = kwargs.get("random_choice_config")
    result = provider.rng.sample(
        *country_static_data.countries(), sample_config=cfg, size=len(df)
    )
    return pd.Series(result, index=df.index, dtype=str)


def account_type(
    provider: RandomChoiceDataProvider,
    df: pd.DataFrame,
    dependent_columns: list[str] | None = None,
    **kwargs,
) -> pd.Series:
    """Randomly sample an account type (SAVINGS / CHECKING / BUSINESS)."""
    cfg: RandomChoiceSamplingConfig | None = kwargs.get("random_choice_config")
    result = provider.rng.sample(
        *field_static_data.ACCOUNT_TYPES, sample_config=cfg, size=len(df)
    )
    return pd.Series(result, index=df.index, dtype=str)


def payment_status(
    provider: RandomChoiceDataProvider,
    df: pd.DataFrame,
    dependent_columns: list[str] | None = None,
    **kwargs,
) -> pd.Series:
    """Randomly sample a payment status."""
    cfg: RandomChoiceSamplingConfig | None = kwargs.get("random_choice_config")
    result = provider.rng.sample(
        *field_static_data.PAYMENT_STATUS, sample_config=cfg, size=len(df)
    )
    return pd.Series(result, index=df.index, dtype=str)


# ---------------------------------------------------------------------------
# Dependent single-column generators (return pd.Series)
# ---------------------------------------------------------------------------


def currency_from_country(
    provider: RandomChoiceDataProvider,
    df: pd.DataFrame,
    dependent_columns: list[str] | None = None,
    **kwargs,
) -> pd.Series:
    """Look up the currency for each row's country.

    Depends on:
        - ``dependent_columns[0]``: country column name.

    Requires ``country_static_data`` in **kwargs.
    """
    assert dependent_columns and len(dependent_columns) >= 1, (
        "dependent_columns must contain the country column name"
    )
    assert "country_static_data" in kwargs, (
        "country_static_data must be provided in kwargs"
    )
    _csd = kwargs["country_static_data"]
    country_col = dependent_columns[0]
    return df[country_col].map(lambda cty: country_static_data.currency(_csd, str(cty)))


def account_create_timestamp(
    provider: FakerSyntheticDataProvider,
    df: pd.DataFrame,
    dependent_columns: list[str] | None = None,
    **kwargs,
) -> pd.Series:
    """Generate account creation timestamps between DOB and 12 weeks ago.

    Depends on:
        - ``dependent_columns[0..2]``: birth_year, birth_month, birth_day.
    """
    assert dependent_columns and len(dependent_columns) >= 3, (
        "dependent_columns must contain birth_year, birth_month, birth_day column names"
    )
    yr_col, mo_col, dy_col = dependent_columns[:3]
    fake = provider.provide()
    cutoff = datetime.today() - timedelta(weeks=12)
    results = []
    for yr, mo, dy in zip(df[yr_col], df[mo_col], df[dy_col], strict=True):
        ts = fake.date_time_between(
            start_date=datetime(int(yr), int(mo), int(dy)),
            end_date=cutoff,
        ).timestamp()
        results.append(ts)
    return pd.Series(results, index=df.index, dtype=float)


def account_last_activity_timestamp(
    provider: FakerSyntheticDataProvider,
    df: pd.DataFrame,
    dependent_columns: list[str] | None = None,
    **kwargs,
) -> pd.Series:
    """Generate last-activity timestamps after account creation.

    Depends on:
        - ``dependent_columns[0]``: account_create_timestamp.
    """
    assert dependent_columns and len(dependent_columns) >= 1, (
        "dependent_columns must contain the account_create_timestamp column name"
    )
    create_ts_col = dependent_columns[0]
    fake = provider.provide()
    now = datetime.now()
    week_ago = now - timedelta(weeks=1)
    results = []
    for ts in df[create_ts_col]:
        start = max(datetime.fromtimestamp(float(ts)), week_ago)
        results.append(
            fake.date_time_between(start_date=start, end_date=now).timestamp()
        )
    return pd.Series(results, index=df.index, dtype=float)


def account_activity_events_past_30d(
    provider: UniformDistributionDataProvider,
    df: pd.DataFrame,
    dependent_columns: list[str] | None = None,
    **kwargs,
) -> pd.Series:
    """Generate account activity event counts based on account type.

    Depends on:
        - ``dependent_columns[0]``: account_type.
    """
    assert dependent_columns and len(dependent_columns) >= 1, (
        "dependent_columns must contain the account_type column name"
    )
    acct_col = dependent_columns[0]
    n = len(df)
    result = np.zeros(n, dtype=int)
    acct_types = df[acct_col].to_numpy()

    for acct_val, low, high in [
        ("BUSINESS", 50, 1_500_000),
        ("CHECKING", 2, 500),
        ("SAVINGS", 2, 50),
    ]:
        mask = acct_types == acct_val
        count = int(mask.sum())
        if count > 0:
            cfg = UniformDistributionSamplingConfig(low=float(low), high=float(high))
            vals = provider.rng.sample(sample_config=cfg, size=count)
            result[mask] = np.asarray(vals, dtype=int)

    return pd.Series(result, index=df.index, dtype=int)


def payment_init_timestamp(
    provider: FakerSyntheticDataProvider,
    df: pd.DataFrame,
    dependent_columns: list[str] | None = None,
    **kwargs,
) -> pd.Series:
    """Generate payment-initiation timestamps after both parties' last activity.

    Depends on:
        - ``dependent_columns[0]``: debitor account_last_activity_timestamp.
        - ``dependent_columns[1]``: creditor account_last_activity_timestamp.
    """
    assert dependent_columns and len(dependent_columns) >= 2, (
        "dependent_columns must contain debitor and creditor account_last_activity_timestamp column names"
    )
    deb_col, cred_col = dependent_columns[:2]
    fake = provider.provide()
    now = datetime.now()
    results = []
    for deb_ts, cred_ts in zip(df[deb_col], df[cred_col], strict=True):
        latest = max(float(deb_ts), float(cred_ts))
        start = datetime.fromtimestamp(latest) + timedelta(days=7)
        results.append(
            fake.date_time_between(start_date=start, end_date=now).timestamp()
        )
    return pd.Series(results, index=df.index, dtype=float)


def payment_last_update_timestamp(
    provider: FakerSyntheticDataProvider,
    df: pd.DataFrame,
    dependent_columns: list[str] | None = None,
    **kwargs,
) -> pd.Series:
    """Generate payment last-update timestamps after payment initiation.

    Depends on:
        - ``dependent_columns[0]``: payment_init_timestamp.
    """
    assert dependent_columns and len(dependent_columns) >= 1, (
        "dependent_columns must contain the payment_init_timestamp column name"
    )
    init_col = dependent_columns[0]
    fake = provider.provide()
    now = datetime.now()
    results = []
    for ts in df[init_col]:
        start = datetime.fromtimestamp(float(ts))
        results.append(
            fake.date_time_between(start_date=start, end_date=now).timestamp()
        )
    return pd.Series(results, index=df.index, dtype=float)


# ---------------------------------------------------------------------------
# Dependent multi-column generators (return pd.DataFrame)
# ---------------------------------------------------------------------------


def geo_coordinates(
    provider: FakerSyntheticDataProvider,
    df: pd.DataFrame,
    dependent_columns: list[str] | None = None,
    **kwargs,
) -> pd.DataFrame:
    """Generate (latitude, longitude) pairs based on the row's country code.

    Depends on:
        - ``dependent_columns[0]``: country column name.
    """
    assert dependent_columns and len(dependent_columns) >= 1, (
        "dependent_columns must contain the country column name"
    )
    country_col = dependent_columns[0]
    fake = provider.provide()
    lats, lons = [], []
    for cty in df[country_col]:
        ll = None
        while ll is None:
            ll = fake.local_latlng(country_code=str(cty))
        lats.append(float(ll[0]))
        lons.append(float(ll[1]))
    return pd.DataFrame({"col_0": lats, "col_1": lons}, index=df.index)


def tower_geo_coordinates(
    provider: UniformDistributionDataProvider,
    df: pd.DataFrame,
    dependent_columns: list[str] | None = None,
    **kwargs,
) -> pd.DataFrame:
    """Add uniform noise to user geo-coordinates to simulate cell-tower positions.

    Depends on:
        - ``dependent_columns[0]``: geo_latitude.
        - ``dependent_columns[1]``: geo_longitude.

    Accepts ``uniform_dist_config_lat`` and ``uniform_dist_config_lon`` in **kwargs.
    """
    assert dependent_columns and len(dependent_columns) >= 2, (
        "dependent_columns must contain geo_latitude, geo_longitude column names"
    )
    lat_col, lon_col = dependent_columns[:2]
    n = len(df)
    cfg_lat: UniformDistributionSamplingConfig | None = kwargs.get(
        "uniform_dist_config_lat"
    )
    cfg_lon: UniformDistributionSamplingConfig | None = kwargs.get(
        "uniform_dist_config_lon"
    )
    eps_lat = provider.rng.sample(sample_config=cfg_lat, size=n)
    eps_lon = provider.rng.sample(sample_config=cfg_lon, size=n)
    tower_lats = df[lat_col].to_numpy(dtype=float) + np.asarray(eps_lat, dtype=float)
    tower_lons = df[lon_col].to_numpy(dtype=float) + np.asarray(eps_lon, dtype=float)
    return pd.DataFrame({"col_0": tower_lats, "col_1": tower_lons}, index=df.index)


def currency_exchange_rates(
    provider: FakerSyntheticDataProvider,
    df: pd.DataFrame,
    dependent_columns: list[str] | None = None,
    **kwargs,
) -> pd.DataFrame:
    """Look up pairwise currency exchange rates for each row.

    Depends on:
        - ``dependent_columns[0]``: debitor currency.
        - ``dependent_columns[1]``: creditor currency.

    Requires ``country_static_data`` in **kwargs.

    Note:
        The *provider* parameter is unused — this function performs pure
        lookups against static data.  The type hint exists solely for
        auto-mapping dispatch in the provider routing logic.
    """
    assert dependent_columns and len(dependent_columns) >= 2, (
        "dependent_columns must contain both currency column names"
    )
    assert "country_static_data" in kwargs, (
        "country_static_data must be provided in kwargs"
    )
    _csd = kwargs["country_static_data"]
    ccy1_col, ccy2_col = dependent_columns[:2]

    rates_1, rates_2 = [], []
    for c1, c2 in zip(df[ccy1_col], df[ccy2_col], strict=True):
        rates_1.append(country_static_data.get_exchange_rate(_csd, str(c1), str(c2)))
        rates_2.append(country_static_data.get_exchange_rate(_csd, str(c2), str(c1)))
    return pd.DataFrame({"col_0": rates_1, "col_1": rates_2}, index=df.index)


def payment_amounts(
    provider: UniformDistributionDataProvider,
    df: pd.DataFrame,
    dependent_columns: list[str] | None = None,
    **kwargs,
) -> pd.DataFrame:
    """Generate debitor and creditor payment amounts.

    Depends on:
        - ``dependent_columns[0]``: account_type column.
        - ``dependent_columns[1]``: exchange_rate column.

    When ``lognormal_personal_amount_config`` and ``lognormal_business_amount_config``
    are provided in **kwargs (dicts with ``desired_mean`` and ``sigma`` keys),
    amounts are sampled from account-type-specific lognormal distributions.
    Otherwise falls back to uniform sampling via ``uniform_dist_config``.
    """
    assert dependent_columns and len(dependent_columns) >= 2, (
        "dependent_columns must contain account_type and exchange_rate column names"
    )
    acct_col, rate_col = dependent_columns[:2]
    n = len(df)

    personal_cfg = kwargs.get("lognormal_personal_amount_config")
    business_cfg = kwargs.get("lognormal_business_amount_config")

    if personal_cfg is not None and business_cfg is not None:
        from data_generation.rng.lognormal_distribution import get_lognormal_params

        numpy_rng = provider.rng.rng  # seeded np.random.Generator
        acct_types = df[acct_col].to_numpy()
        amounts = np.zeros(n, dtype=float)

        personal_mask = np.isin(acct_types, ["SAVINGS", "CHECKING"])
        business_mask = acct_types == "BUSINESS"

        p_mu, p_sigma = get_lognormal_params(
            personal_cfg["desired_mean"],
            personal_cfg["sigma"],
        )
        b_mu, b_sigma = get_lognormal_params(
            business_cfg["desired_mean"],
            business_cfg["sigma"],
        )

        n_personal = int(personal_mask.sum())
        if n_personal > 0:
            amounts[personal_mask] = numpy_rng.lognormal(p_mu, p_sigma, size=n_personal)

        n_business = int(business_mask.sum())
        if n_business > 0:
            amounts[business_mask] = numpy_rng.lognormal(b_mu, b_sigma, size=n_business)

        debitor_amounts = np.round(amounts, 2)
    else:
        cfg: UniformDistributionSamplingConfig | None = kwargs.get(
            "uniform_dist_config"
        )
        raw = provider.rng.sample(sample_config=cfg, size=n)
        debitor_amounts = np.round(np.asarray(raw, dtype=float), 2)

    exchange_rates = df[rate_col].to_numpy(dtype=float)
    creditor_amounts = np.round(debitor_amounts * exchange_rates, 2)
    return pd.DataFrame(
        {"col_0": debitor_amounts, "col_1": creditor_amounts}, index=df.index
    )
