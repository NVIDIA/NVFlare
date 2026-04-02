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

"""
Static reference data for FSI federated learning experiments.

Provides country-to-currency mappings and currency exchange rates for a fixed
set of countries (US, TR, AT, IE, PL, PT, GB, FR, IN). Exchange rates are
fetched once from the CurrencyConverter API (using a snapshot date of
2019-01-01) and cached to a local CSV file to avoid repeated network calls.

Usage::

    from data_generation.static_data import load_static_data, get_exchange_rates

    static = load_static_data("~/.cache/fsi_static")
    # static.country_currency_map  — DataFrame with columns [country, currency]
    # static.currency_exchange_rates — DataFrame with columns [CCY1, CCY2, RATE]

    rate = get_exchange_rates("USD", "EUR")

Public API:
    load_static_data(static_data_dir, force_rates_from_api) -> CountryStaticData
    get_exchange_rates(curr1, curr2) -> float
    CountryStaticData  — dataclass holding both DataFrames
    COUNTRIES          — tuple of supported ISO 3166-1 alpha-2 country codes
"""

import os
import tempfile
from dataclasses import dataclass
from datetime import date
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from babel import numbers
from currency_converter import CurrencyConverter

# Fixed snapshot date used for all exchange rate lookups; keeps generated data reproducible.
_SNAPSHOT_DATE = date(2019, 1, 1)

# Supported ISO 3166-1 alpha-2 country codes used throughout data generation.
COUNTRIES = ("US", "TR", "AT", "IE", "PL", "PT", "GB", "FR", "IN")


@dataclass
class CountryStaticData:
    """Holds all static reference data needed for transaction data generation."""

    country_currency_map: pd.DataFrame
    """DataFrame with columns [country, currency] mapping each country to its primary currency."""

    currency_exchange_rates: pd.DataFrame
    """DataFrame with columns [CCY1, CCY2, RATE] for all pairwise currency conversion rates."""


def _get_country_currency() -> pd.DataFrame:
    """Return a DataFrame mapping each country in COUNTRIES to its primary ISO 4217 currency code."""
    # get_territory_currencies returns a list; [0] selects the primary/dominant currency.
    return pd.DataFrame(
        {
            "country": COUNTRIES,
            "currency": [numbers.get_territory_currencies(cty)[0] for cty in COUNTRIES],
        }
    )


def _generate_exchange_rates(country_currency_map: pd.DataFrame) -> pd.DataFrame:
    """Build a DataFrame of pairwise exchange rates for all currencies in country_currency_map.

    Rates are fetched from the CurrencyConverter package using a fixed snapshot
    date (2019-01-01) for reproducibility. fallback_on_missing_rate allows the
    converter to use the nearest available date when an exact rate is absent.
    """
    print(
        "WARNING: Generating currency exchange rates from CurrencyConverter API. This may take a few seconds and requires an internet connection."
    )
    # fallback_on_missing_rate prevents errors for currencies with sparse historical data.
    curr_conv = CurrencyConverter(fallback_on_missing_rate=True)  # initialise the converter once; reused for all pairs
    currencies = country_currency_map["currency"].tolist()  # extract the ordered list of unique currency codes
    n = len(currencies)  # number of distinct currencies; determines matrix dimensions (n*n)

    # Pre-fill an n*n matrix with 1.0; diagonal entries (same currency) are already correct.
    rate_matrix = np.ones((n, n), dtype=float)  # shape (n, n), default 1.0 everywhere
    for (i, ccy1), (j, ccy2) in product(enumerate(currencies), repeat=2):  # iterate over all ordered pairs
        if i != j:  # skip diagonal — same-currency rate is always 1.0
            raw = curr_conv.convert(1, ccy1, ccy2, date=_SNAPSHOT_DATE)  # fetch rate: 1 ccy1 → ccy2
            # Guard against zero/None/negative rates that can occur on missing fallbacks.
            rate_matrix[i, j] = 1.0 if (not raw or raw <= 0.0) else raw  # sanitize invalid values to 1.0

    # Flatten the matrix into long format
    ccy1_col = np.repeat(
        currencies, n
    )  # repeat each currency n times to form the CCY1 column: [USD,USD,...,EUR,EUR,...]
    ccy2_col = np.tile(currencies, n)  # tile the full list n times to form the CCY2 column: [USD,EUR,...,USD,EUR,...]
    rate_col = rate_matrix.flatten().round(4)  # flatten row-major (matches repeat/tile order) and round to 4 d.p.

    return pd.DataFrame(
        {"CCY1": ccy1_col, "CCY2": ccy2_col, "RATE": rate_col}
    )  # assemble into a tidy long-format DataFrame


def get_exchange_rate(static_data: CountryStaticData, curr1: str, curr2: str) -> float:
    """Look up the exchange rate from curr1 to curr2.

    Returns 1.0 if the pair is not found in the rates table (safe no-op default).
    """
    result = static_data.currency_exchange_rates[
        (static_data.currency_exchange_rates["CCY1"] == curr1) & (static_data.currency_exchange_rates["CCY2"] == curr2)
    ]
    if not len(result):
        print(f"WARNING: Exchange rate not found for pair ({curr1}, {curr2}); defaulting to 1.0")
        return 1.0
    return float(result["RATE"].values[0])


def countries() -> tuple[str, ...]:
    """Return the tuple of supported country codes."""
    return COUNTRIES


def currency(static_data: CountryStaticData, country: str) -> str:
    """Return the currency code for a given country."""
    result = static_data.country_currency_map[static_data.country_currency_map["country"] == country]
    if not len(result):
        print(f"WARNING: Currency not found for country ({country}); defaulting to 'USD'")
        return "USD"
    return str(result["currency"].values[0])


def load_static_data(static_data_dir: str | Path, force_rates_from_api: bool = False) -> CountryStaticData:
    """Load (or generate) static reference data from disk.

    Args:
        static_data_dir: Directory where the CSV cache file is stored.
            Supports shell variables (``$VAR``) and ``~`` expansion.
        force_rates_from_api: When True, re-fetch rates from the API and
            overwrite the cached CSV even if it already exists.

    Returns:
        A CountryStaticData instance with the country-currency map and
        pairwise exchange rates.
    """
    # Resolve environment variables, ~, and relative segments to an absolute path.
    abs_path: Path = Path(os.path.expandvars(static_data_dir)).expanduser().resolve()

    country_currency_code = _get_country_currency()
    exchange_rates: pd.DataFrame | None = None

    currency_exchange_rates_file: Path = abs_path / "currency_exchange_rates.csv"
    if not currency_exchange_rates_file.is_file() or force_rates_from_api:
        print(
            "Static data cache file does not exist or force_rates_from_api is True. Creating new static data cache file."
        )
        abs_path.mkdir(parents=True, exist_ok=True)
        exchange_rates = _generate_exchange_rates(country_currency_code)
        # Write atomically: write to a temp file in the same directory, then rename.
        # This prevents a corrupt cache file if the process is interrupted mid-write.
        with tempfile.NamedTemporaryFile(mode="w", dir=abs_path, suffix=".csv", delete=False) as tmp:
            exchange_rates.to_csv(tmp, index=False)
            tmp_path = Path(tmp.name)
        tmp_path.replace(currency_exchange_rates_file)

    # exchange_rates is None only when the cache file already existed and was not regenerated.
    if exchange_rates is None:
        exchange_rates = pd.read_csv(currency_exchange_rates_file)

    return CountryStaticData(
        country_currency_map=country_currency_code,
        currency_exchange_rates=exchange_rates,
    )
