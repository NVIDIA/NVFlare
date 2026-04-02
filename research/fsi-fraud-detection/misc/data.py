# from https://github.com/chesterxgchen/fsi_experiment/blob/main/aikya-phase1-sample/notebooks/PmtFraudSimulateFedLearnStory-v1.ipynb

import math
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler

amount_features = ["DEBITOR_AMOUNT_SCALED"]
age_synthetic_features = [
    "DEBITOR_ACCOUNT_AGE",
    "CREDITOR_ACCOUNT_AGE",
    "DEBITOR_ACCOUNT_DURATION_SINCE_LAST_ACTIVITY_SECS",
]
lat_long_synthetic_features = [
    "DEBITOR_PHY_AND_TOWER_DISTANCE",
    "CREDITOR_PHY_AND_TOWER_DISTANCE",
]

debitor_categorical_features = [
    "DEBITOR_ACCOUNT_TYPE_CHECKING",
    "DEBITOR_ACCOUNT_TYPE_SAVINGS",
    "DEBITOR_ACCOUNT_TYPE_BUSINESS",
]
dbtr_acc_type_encoder = OneHotEncoder(sparse_output=False, dtype=int).fit(
    np.array([["CHECKING"], ["SAVINGS"], ["BUSINESS"]])
)

creditor_categorical_features = [
    "CREDITOR_ACCOUNT_TYPE_CHECKING",
    "CREDITOR_ACCOUNT_TYPE_SAVINGS",
    "CREDITOR_ACCOUNT_TYPE_BUSINESS",
]
crdtr_acc_type_encoder = OneHotEncoder(sparse_output=False, dtype=int).fit(
    np.array([["CHECKING"], ["SAVINGS"], ["BUSINESS"]])
)

# Keep backward-compat alias used elsewhere in the codebase
categorical_features = debitor_categorical_features

activity_features = [
    "DEBITOR_ACCOUNT_ACTIVITY_EVENTS_PAST_30D",
]

# Per-account-type normal upper bounds derived from the data generator.
# ACTIVITY_RATIO > 1.0 signals TYPE4 fraud (unusually high activity for account type).
# AMOUNT_RATIO   > 1.0 signals elevated spend relative to account-type typical amount.
_ACTIVITY_NORMAL_UPPER = {"BUSINESS": 1_500_000.0, "CHECKING": 500.0, "SAVINGS": 50.0}
_AMOUNT_NORMAL_UPPER = {"BUSINESS": 100_000.0, "CHECKING": 20_000.0, "SAVINGS": 5_000.0}

ratio_features = [
    "DEBITOR_ACTIVITY_RATIO",  # activity / per-type normal upper  (TYPE4 signal)
    "DEBITOR_AMOUNT_RATIO",  # amount   / per-type normal upper  (TYPE2 signal)
]

numerical_features = (
    amount_features + age_synthetic_features + lat_long_synthetic_features + activity_features + ratio_features
)

all_data_features = [
    "FRAUD_FLAG",
    "DEBITOR_AMOUNT",
    "DEBITOR_GEO_LATITUDE",
    "DEBITOR_GEO_LONGITUDE",
    "DEBITOR_TOWER_LATITUDE",
    "DEBITOR_TOWER_LONGITUDE",
    "CREDITOR_GEO_LATITUDE",
    "CREDITOR_GEO_LONGITUDE",
    "CREDITOR_TOWER_LATITUDE",
    "CREDITOR_TOWER_LONGITUDE",
    "DEBITOR_ACCOUNT_CREATE_TIMESTAMP",
    "CREDITOR_ACCOUNT_CREATE_TIMESTAMP",
    "DEBITOR_ACCOUNT_LAST_ACTIVITY_TIMESTAMP",
    "DEBITOR_ACCOUNT_ACTIVITY_EVENTS_PAST_30D",
    "DEBITOR_ACCOUNT_TYPE",  # one-hot encoded → debitor_categorical_features
    "CREDITOR_ACCOUNT_TYPE",  # one-hot encoded → creditor_categorical_features
]
all_model_parameters = numerical_features + debitor_categorical_features + creditor_categorical_features
flag = "FRAUD_FLAG"

# print("Data Features: ", all_data_features, "\nModel Features: ", all_model_parameters, "\nFlag: ", flag)


def seconds_from_last_activity(dataset: pd.DataFrame, new_field: str, fields: list) -> pd.DataFrame:
    dataset.loc[:, new_field] = dataset.loc[:, fields].apply(
        lambda _: (datetime.fromtimestamp(_.iloc[0]) - datetime.fromtimestamp(_.iloc[1])).total_seconds(),
        axis=1,
    )
    return dataset


def log_normalize_columns(dataset: pd.DataFrame, fields: list) -> pd.DataFrame:
    dataset.loc[:, fields] = dataset.loc[:, fields].apply(lambda _: np.log(_))
    return dataset


def convert_timestamp_field_to_age(dataset: pd.DataFrame, timestamp_field: str, fields: list) -> pd.DataFrame:
    dataset.loc[:, timestamp_field] = dataset.loc[:, fields].apply(
        lambda _: ((datetime.fromtimestamp(_.iloc[0]) - datetime.fromtimestamp(_.iloc[1])).total_seconds() / 60),
        axis=1,
    )
    return dataset


def get_distance_from_lat_long(dataset: pd.DataFrame, distance_field: str, fields: list) -> pd.DataFrame:
    def calculate_geodesic_distance(lat1, lon1, lat2, lon2) -> float:
        # https://stackoverflow.com/a/19412565
        earth_radius = 3958.8 * 1.6  # in km
        lat1 = math.radians(lat1)
        lon1 = math.radians(lon1)
        lat2 = math.radians(lat2)
        lon2 = math.radians(lon2)
        lat_dist = lat2 - lat1
        lon_dist = lon2 - lon1
        a = math.sin(lat_dist / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(lon_dist / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        distance = earth_radius * c
        return float(distance)

    dataset.loc[:, distance_field] = dataset.loc[:, fields].apply(
        lambda row: calculate_geodesic_distance(row.iloc[0], row.iloc[1], row.iloc[2], row.iloc[3]),
        axis=1,
    )
    return dataset


def scale_data(df: pd.DataFrame) -> pd.DataFrame:
    scaler = RobustScaler()
    print(scaler)
    scaled_data = scaler.fit_transform(df)
    return pd.DataFrame(scaled_data, columns=df.columns, index=df.index)


def clean_dataframe(
    df: pd.DataFrame,
    dataset_name: str = "dataset",
    verbose: bool = True,
    subset: list | None = None,
) -> pd.DataFrame:
    """
    Clean dataframe by removing NaN and infinite values.

    Args:
        df: DataFrame to clean
        dataset_name: Name of the dataset for logging purposes
        verbose: Whether to print cleaning statistics
        subset: List of columns to check for NaN/inf. If None, checks all columns.
                Use this to avoid dropping rows due to NaN in irrelevant raw columns.

    Returns:
        Cleaned DataFrame with NaN and inf values removed
    """
    check_cols = subset if subset is not None else df.columns.tolist()
    df_check = df[check_cols]

    if verbose:
        # Check for NaN/inf values before cleaning
        nan_count = df_check.isna().sum().sum()
        inf_count = np.isinf(df_check.select_dtypes(include=[np.number])).sum().sum()
        print(
            f"{dataset_name} - NaN count: {nan_count}, Inf count: {inf_count} (checked {len(check_cols)} feature columns)"
        )
        print(f"{dataset_name} shape before cleaning: {df.shape}")

    # Replace inf with NaN in feature columns, then drop rows with NaN in those columns
    df_cleaned = df.copy()
    df_cleaned[check_cols] = df_cleaned[check_cols].replace([np.inf, -np.inf], np.nan)
    df_cleaned = df_cleaned.dropna(subset=check_cols)

    if verbose:
        print(f"{dataset_name} shape after cleaning: {df_cleaned.shape}")
        rows_removed = len(df) - len(df_cleaned)
        if rows_removed > 0:
            print(f"{dataset_name} - Removed {rows_removed} rows ({rows_removed / len(df) * 100:.2f}%)")

    return df_cleaned


def prepare_dataset(dataset: pd.DataFrame, scaler: StandardScaler | None = None) -> pd.DataFrame:
    dataset.loc[:, "DEBITOR_AMOUNT_SCALED"] = np.log1p(dataset.loc[:, "DEBITOR_AMOUNT"])
    dataset = convert_timestamp_field_to_age(
        dataset,
        "DEBITOR_ACCOUNT_AGE",
        ["PAYMENT_INIT_TIMESTAMP", "DEBITOR_ACCOUNT_CREATE_TIMESTAMP"],
    )
    dataset = convert_timestamp_field_to_age(
        dataset,
        "CREDITOR_ACCOUNT_AGE",
        ["PAYMENT_INIT_TIMESTAMP", "CREDITOR_ACCOUNT_CREATE_TIMESTAMP"],
    )
    dataset = get_distance_from_lat_long(
        dataset,
        "DEBITOR_PHY_AND_TOWER_DISTANCE",
        [
            "DEBITOR_GEO_LATITUDE",
            "DEBITOR_GEO_LONGITUDE",
            "DEBITOR_TOWER_LATITUDE",
            "DEBITOR_TOWER_LONGITUDE",
        ],
    )
    dataset = get_distance_from_lat_long(
        dataset,
        "CREDITOR_PHY_AND_TOWER_DISTANCE",
        [
            "CREDITOR_GEO_LATITUDE",
            "CREDITOR_GEO_LONGITUDE",
            "CREDITOR_TOWER_LATITUDE",
            "CREDITOR_TOWER_LONGITUDE",
        ],
    )
    dataset = seconds_from_last_activity(
        dataset,
        "DEBITOR_ACCOUNT_DURATION_SINCE_LAST_ACTIVITY_SECS",
        ["PAYMENT_INIT_TIMESTAMP", "DEBITOR_ACCOUNT_LAST_ACTIVITY_TIMESTAMP"],
    )

    dataset[debitor_categorical_features] = np.asarray(
        dbtr_acc_type_encoder.transform(dataset[["DEBITOR_ACCOUNT_TYPE"]])
    )
    dataset[creditor_categorical_features] = np.asarray(
        crdtr_acc_type_encoder.transform(dataset[["CREDITOR_ACCOUNT_TYPE"]])
    )

    # Ratio features: normalise by per-account-type normal upper bound.
    # Values > 1.0 indicate the metric exceeds the typical ceiling for that account type.
    act_upper = dataset["DEBITOR_ACCOUNT_TYPE"].map(_ACTIVITY_NORMAL_UPPER)
    amt_upper = dataset["DEBITOR_ACCOUNT_TYPE"].map(_AMOUNT_NORMAL_UPPER)
    dataset.loc[:, "DEBITOR_ACTIVITY_RATIO"] = dataset["DEBITOR_ACCOUNT_ACTIVITY_EVENTS_PAST_30D"] / act_upper
    dataset.loc[:, "DEBITOR_AMOUNT_RATIO"] = dataset["DEBITOR_AMOUNT"] / amt_upper

    dataset = log_normalize_columns(
        dataset,
        [
            "CREDITOR_ACCOUNT_AGE",
            "DEBITOR_ACCOUNT_AGE",
            "DEBITOR_ACCOUNT_DURATION_SINCE_LAST_ACTIVITY_SECS",
        ],
    )

    dataset = clean_dataframe(dataset, subset=all_model_parameters + [flag])

    # now all features should be numeric and unit-less. we can scale the data correctly now
    if isinstance(scaler, bool) and scaler:
        print("Using scale_data function")
        dataset.loc[:, numerical_features] = scale_data(dataset.loc[:, numerical_features])
    elif scaler:
        print(f"Using provided scaler...{scaler}")
        dataset.loc[:, numerical_features] = scaler.transform(dataset.loc[:, numerical_features])

    return dataset
