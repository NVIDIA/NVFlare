from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from data_generation.static_data.country_static_data import (
    COUNTRIES,
    CountryStaticData,
    _generate_exchange_rates,
    _get_country_currency,
    countries,
    currency,
    get_exchange_rate,
    load_static_data,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def country_currency_df() -> pd.DataFrame:
    """A small 3-country currency map for fast, deterministic tests."""
    return pd.DataFrame({"country": ["US", "GB", "IN"], "currency": ["USD", "GBP", "INR"]})


@pytest.fixture
def fake_exchange_rates(country_currency_df: pd.DataFrame) -> pd.DataFrame:
    """Deterministic exchange rates for the 3-currency fixture."""
    currencies: list[str] = country_currency_df["currency"].tolist()
    rows: list[dict[str, str | float]] = []
    for i, ccy1 in enumerate(currencies):
        for j, ccy2 in enumerate(currencies):
            rows.append(
                {
                    "CCY1": ccy1,
                    "CCY2": ccy2,
                    "RATE": 1.0 if i == j else round(0.5 + 0.1 * (i + j), 4),
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def static_data(country_currency_df: pd.DataFrame, fake_exchange_rates: pd.DataFrame) -> CountryStaticData:
    return CountryStaticData(
        country_currency_map=country_currency_df,
        currency_exchange_rates=fake_exchange_rates,
    )


# ---------------------------------------------------------------------------
# COUNTRIES constant
# ---------------------------------------------------------------------------


class TestCountries:
    def test_countries_is_tuple(self) -> None:
        assert isinstance(COUNTRIES, tuple)

    def test_countries_non_empty(self) -> None:
        assert len(COUNTRIES) > 0

    def test_countries_are_alpha2(self) -> None:
        for code in COUNTRIES:
            assert isinstance(code, str)
            assert len(code) == 2
            assert code == code.upper()

    def test_countries_helper_returns_same_tuple(self) -> None:
        assert countries() == COUNTRIES


# ---------------------------------------------------------------------------
# _get_country_currency
# ---------------------------------------------------------------------------


class TestGetCountryCurrency:
    def test_returns_dataframe(self) -> None:
        df = _get_country_currency()
        assert isinstance(df, pd.DataFrame)

    def test_has_expected_columns(self) -> None:
        df = _get_country_currency()
        assert list(df.columns) == ["country", "currency"]

    def test_row_count_matches_countries(self) -> None:
        df = _get_country_currency()
        assert len(df) == len(COUNTRIES)

    def test_country_column_matches_constant(self) -> None:
        df = _get_country_currency()
        assert tuple(df["country"]) == COUNTRIES

    def test_currencies_are_3_letter_codes(self) -> None:
        df = _get_country_currency()
        for ccy in df["currency"]:
            assert isinstance(ccy, str)
            assert len(ccy) == 3
            assert ccy == ccy.upper()

    def test_us_maps_to_usd(self) -> None:
        df = _get_country_currency()
        us_row = df[df["country"] == "US"]
        assert len(us_row) == 1
        assert us_row["currency"].values[0] == "USD"

    def test_gb_maps_to_gbp(self) -> None:
        df = _get_country_currency()
        gb_row = df[df["country"] == "GB"]
        assert len(gb_row) == 1
        assert gb_row["currency"].values[0] == "GBP"


# ---------------------------------------------------------------------------
# _generate_exchange_rates  (mocked — no network calls)
# ---------------------------------------------------------------------------


class TestGenerateExchangeRates:
    def test_returns_dataframe_with_correct_columns(self, country_currency_df: pd.DataFrame) -> None:
        with patch("data_generation.static_data.country_static_data.CurrencyConverter") as MockCC:
            mock_conv = MagicMock()
            mock_conv.convert.return_value = 1.25
            MockCC.return_value = mock_conv

            df = _generate_exchange_rates(country_currency_df)
            assert isinstance(df, pd.DataFrame)
            assert list(df.columns) == ["CCY1", "CCY2", "RATE"]

    def test_row_count_is_n_squared(self, country_currency_df: pd.DataFrame) -> None:
        n = len(country_currency_df)
        with patch("data_generation.static_data.country_static_data.CurrencyConverter") as MockCC:
            mock_conv = MagicMock()
            mock_conv.convert.return_value = 1.5
            MockCC.return_value = mock_conv

            df = _generate_exchange_rates(country_currency_df)
            assert len(df) == n * n

    def test_diagonal_rates_are_one(self, country_currency_df: pd.DataFrame) -> None:
        with patch("data_generation.static_data.country_static_data.CurrencyConverter") as MockCC:
            mock_conv = MagicMock()
            mock_conv.convert.return_value = 2.0
            MockCC.return_value = mock_conv

            df = _generate_exchange_rates(country_currency_df)
            diagonal = df[df["CCY1"] == df["CCY2"]]
            assert all(diagonal["RATE"].apply(lambda r: r == pytest.approx(1.0)))

    def test_off_diagonal_uses_converter_value(self, country_currency_df: pd.DataFrame) -> None:
        with patch("data_generation.static_data.country_static_data.CurrencyConverter") as MockCC:
            mock_conv = MagicMock()
            mock_conv.convert.return_value = 1.3456
            MockCC.return_value = mock_conv

            df = _generate_exchange_rates(country_currency_df)
            off_diag = df[df["CCY1"] != df["CCY2"]]
            assert all(off_diag["RATE"].apply(lambda r: r == pytest.approx(1.3456)))

    def test_negative_rate_sanitized_to_one(self, country_currency_df: pd.DataFrame) -> None:
        with patch("data_generation.static_data.country_static_data.CurrencyConverter") as MockCC:
            mock_conv = MagicMock()
            mock_conv.convert.return_value = -0.5
            MockCC.return_value = mock_conv

            df = _generate_exchange_rates(country_currency_df)
            assert all(df["RATE"].apply(lambda r: r == pytest.approx(1.0)))

    def test_zero_rate_sanitized_to_one(self, country_currency_df: pd.DataFrame) -> None:
        with patch("data_generation.static_data.country_static_data.CurrencyConverter") as MockCC:
            mock_conv = MagicMock()
            mock_conv.convert.return_value = 0.0
            MockCC.return_value = mock_conv

            df = _generate_exchange_rates(country_currency_df)
            assert all(df["RATE"].apply(lambda r: r == pytest.approx(1.0)))

    def test_none_rate_sanitized_to_one(self, country_currency_df: pd.DataFrame) -> None:
        with patch("data_generation.static_data.country_static_data.CurrencyConverter") as MockCC:
            mock_conv = MagicMock()
            mock_conv.convert.return_value = None
            MockCC.return_value = mock_conv

            df = _generate_exchange_rates(country_currency_df)
            assert all(df["RATE"].apply(lambda r: r == pytest.approx(1.0)))

    def test_rates_rounded_to_4_decimals(self, country_currency_df: pd.DataFrame) -> None:
        with patch("data_generation.static_data.country_static_data.CurrencyConverter") as MockCC:
            mock_conv = MagicMock()
            mock_conv.convert.return_value = 1.123456789
            MockCC.return_value = mock_conv

            df = _generate_exchange_rates(country_currency_df)
            off_diag = df[df["CCY1"] != df["CCY2"]]
            for rate in off_diag["RATE"]:
                assert rate == round(rate, 4)


# ---------------------------------------------------------------------------
# get_exchange_rate
# ---------------------------------------------------------------------------


class TestGetExchangeRate:
    def test_known_pair_returns_correct_rate(self, static_data: CountryStaticData) -> None:
        rate = get_exchange_rate(static_data, "USD", "GBP")
        expected = static_data.currency_exchange_rates[
            (static_data.currency_exchange_rates["CCY1"] == "USD")
            & (static_data.currency_exchange_rates["CCY2"] == "GBP")
        ]["RATE"].values[0]
        assert rate == float(expected)

    def test_same_currency_returns_one(self, static_data: CountryStaticData) -> None:
        assert get_exchange_rate(static_data, "USD", "USD") == pytest.approx(1.0)

    def test_missing_pair_returns_one(self, static_data: CountryStaticData) -> None:
        assert get_exchange_rate(static_data, "USD", "JPY") == pytest.approx(1.0)

    def test_return_type_is_float(self, static_data: CountryStaticData) -> None:
        assert isinstance(get_exchange_rate(static_data, "USD", "GBP"), float)


# ---------------------------------------------------------------------------
# currency()
# ---------------------------------------------------------------------------


class TestCurrency:
    def test_known_country_returns_currency(self, static_data: CountryStaticData) -> None:
        assert currency(static_data, "US") == "USD"

    def test_gb_returns_gbp(self, static_data: CountryStaticData) -> None:
        assert currency(static_data, "GB") == "GBP"

    def test_unknown_country_defaults_to_usd(self, static_data: CountryStaticData) -> None:
        assert currency(static_data, "ZZ") == "USD"

    def test_return_type_is_str(self, static_data: CountryStaticData) -> None:
        assert isinstance(currency(static_data, "US"), str)


# ---------------------------------------------------------------------------
# load_static_data
# ---------------------------------------------------------------------------


class TestLoadStaticData:
    def test_creates_cache_dir_and_csv(self, tmp_path: Path) -> None:
        cache_dir = tmp_path / "cache" / "static"
        with patch("data_generation.static_data.country_static_data._generate_exchange_rates") as mock_gen:
            mock_gen.return_value = pd.DataFrame({"CCY1": ["USD"], "CCY2": ["GBP"], "RATE": [1.3]})
            result = load_static_data(cache_dir)

        assert isinstance(result, CountryStaticData)
        assert (cache_dir / "currency_exchange_rates.csv").is_file()

    def test_returns_country_currency_map(self, tmp_path: Path) -> None:
        with patch("data_generation.static_data.country_static_data._generate_exchange_rates") as mock_gen:
            mock_gen.return_value = pd.DataFrame({"CCY1": ["USD"], "CCY2": ["GBP"], "RATE": [1.3]})
            result = load_static_data(tmp_path)

        assert list(result.country_currency_map.columns) == ["country", "currency"]
        assert len(result.country_currency_map) == len(COUNTRIES)

    def test_reads_from_cache_on_second_call(self, tmp_path: Path) -> None:
        with patch("data_generation.static_data.country_static_data._generate_exchange_rates") as mock_gen:
            mock_gen.return_value = pd.DataFrame({"CCY1": ["USD", "GBP"], "CCY2": ["GBP", "USD"], "RATE": [1.3, 0.77]})
            load_static_data(tmp_path)
            mock_gen.reset_mock()

            result = load_static_data(tmp_path)
            mock_gen.assert_not_called()

        assert isinstance(result, CountryStaticData)
        assert len(result.currency_exchange_rates) == 2

    def test_force_rates_from_api_regenerates(self, tmp_path: Path) -> None:
        with patch("data_generation.static_data.country_static_data._generate_exchange_rates") as mock_gen:
            mock_gen.return_value = pd.DataFrame({"CCY1": ["USD"], "CCY2": ["GBP"], "RATE": [1.3]})
            load_static_data(tmp_path)

            mock_gen.return_value = pd.DataFrame({"CCY1": ["USD"], "CCY2": ["GBP"], "RATE": [9.9]})
            result = load_static_data(tmp_path, force_rates_from_api=True)

        assert result.currency_exchange_rates["RATE"].values[0] == pytest.approx(9.9)

    def test_expands_home_dir(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HOME", str(tmp_path))
        cache_dir = "~/fsi_test_cache"
        with patch("data_generation.static_data.country_static_data._generate_exchange_rates") as mock_gen:
            mock_gen.return_value = pd.DataFrame({"CCY1": ["USD"], "CCY2": ["GBP"], "RATE": [1.0]})
            result = load_static_data(cache_dir)

        assert isinstance(result, CountryStaticData)
        assert (tmp_path / "fsi_test_cache" / "currency_exchange_rates.csv").is_file()


# ---------------------------------------------------------------------------
# CountryStaticData dataclass
# ---------------------------------------------------------------------------


class TestCountryStaticData:
    def test_dataclass_fields(self, static_data: CountryStaticData) -> None:
        assert hasattr(static_data, "country_currency_map")
        assert hasattr(static_data, "currency_exchange_rates")
        assert isinstance(static_data.country_currency_map, pd.DataFrame)
        assert isinstance(static_data.currency_exchange_rates, pd.DataFrame)
