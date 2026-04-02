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

"""Dependency graph definitions for the synthetic payment dataset.

Each function builds an ``AttributeDependencyGraphType`` — a dict mapping an
attribute (or attribute group) to the list of column names it depends on.
Attributes with no dependencies map to an empty list and can be generated
in any order; dependent attributes must wait until their prerequisite columns
exist in the DataFrame.

Three functions compose the full graph:
    * ``get_per_participant_attributes`` — per-party columns (debitor & creditor)
    * ``get_payment_core_attributes``    — payment-level columns (ID, timestamps, status)
    * ``get_payment_amount_attributes``  — exchange rates & amounts
"""

import data_generation.synthetic_data_provider.faker_synthetic_data_provider_helper_functions as H
from data_generation.dataset_attribute import PaymentDatasetAttribute, PaymentDatasetAttributeGroup
from data_generation.synthetic_data_provider import (
    FakerSyntheticDataProvider,
    RandomChoiceDataProvider,
    UniformDistributionDataProvider,
)

type AttributeType = (
    PaymentDatasetAttribute[FakerSyntheticDataProvider]
    | PaymentDatasetAttribute[RandomChoiceDataProvider]
    | PaymentDatasetAttribute[UniformDistributionDataProvider]
    | PaymentDatasetAttributeGroup[FakerSyntheticDataProvider]
    | PaymentDatasetAttributeGroup[RandomChoiceDataProvider]
    | PaymentDatasetAttributeGroup[UniformDistributionDataProvider]
)
"""Union of all concrete attribute / attribute-group types."""

type AttributeDependencyGraphType = dict[AttributeType, list[str]]
"""Maps each attribute to the column names it depends on (empty list = independent)."""


def get_per_participant_attributes(
    participant_prefixes: tuple[str, str],
    dependency_graph: AttributeDependencyGraphType | None = None,
) -> AttributeDependencyGraphType:
    """Build the per-participant portion of the dependency graph.

    For each prefix (e.g. DEBITOR, CREDITOR) this registers personal info,
    address, account, DOB, currency, geo-coordinates, timestamps, tower
    coordinates, and activity event attributes.
    """
    dep_graph: AttributeDependencyGraphType = {} if dependency_graph is None else dependency_graph
    for participant_prefix in participant_prefixes:
        p = participant_prefix.upper()

        # --- independent attributes (no dependencies) ---
        dep_graph[PaymentDatasetAttribute(f"{p}_USERNAME", H.username)] = []
        dep_graph[PaymentDatasetAttribute(f"{p}_FIRST_NAME", H.firstname)] = []
        dep_graph[PaymentDatasetAttribute(f"{p}_LAST_NAME", H.lastname)] = []
        dep_graph[PaymentDatasetAttribute(f"{p}_EMAIL_ADDRESS", H.email)] = []
        dep_graph[PaymentDatasetAttribute(f"{p}_PHONE_NUMBER", H.phone_number)] = []
        dep_graph[PaymentDatasetAttribute(f"{p}_GENDER", H.gender)] = []
        dep_graph[PaymentDatasetAttribute(f"{p}_ACCOUNT_NUMBER", H.account_number)] = []
        dep_graph[PaymentDatasetAttribute(f"{p}_BIC_CODE", H.bic_code)] = []
        dep_graph[PaymentDatasetAttribute(f"{p}_IP_ADDRESS", H.ip_address)] = []
        dep_graph[PaymentDatasetAttribute(f"{p}_COMMENT", H.comment)] = []
        dep_graph[PaymentDatasetAttribute(f"{p}_ACCOUNT_TYPE", H.account_type)] = []

        # address attributes — also independent
        dep_graph[PaymentDatasetAttribute(f"{p}_ADDR_BUILDING", H.building_number)] = []
        dep_graph[PaymentDatasetAttribute(f"{p}_ADDR_STREET", H.street)] = []
        dep_graph[PaymentDatasetAttribute(f"{p}_ADDR_CITY", H.city)] = []
        dep_graph[PaymentDatasetAttribute(f"{p}_ADDR_STATE", H.state)] = []
        dep_graph[PaymentDatasetAttribute(f"{p}_ADDR_ZIPCODE", H.zipcode)] = []
        dep_graph[PaymentDatasetAttribute(f"{p}_ADDR_COUNTRY", H.country)] = []

        # date of birth — independent, multi-column
        dep_graph[
            PaymentDatasetAttributeGroup(
                (f"{p}_BIRTH_YEAR", f"{p}_BIRTH_MONTH", f"{p}_BIRTH_DAY"),
                H.date_of_birth,
            )
        ] = []

        # --- derived attributes ---

        # currency — depends on country
        dep_graph[PaymentDatasetAttribute(f"{p}_CURRENCY", H.currency_from_country)] = [
            f"{p}_ADDR_COUNTRY",
        ]

        # geo coordinates — depends on country
        dep_graph[
            PaymentDatasetAttributeGroup(
                (f"{p}_GEO_LATITUDE", f"{p}_GEO_LONGITUDE"),
                H.geo_coordinates,
            )
        ] = [f"{p}_ADDR_COUNTRY"]

        # account create timestamp — depends on DOB
        dep_graph[PaymentDatasetAttribute(f"{p}_ACCOUNT_CREATE_TIMESTAMP", H.account_create_timestamp)] = [
            f"{p}_BIRTH_YEAR",
            f"{p}_BIRTH_MONTH",
            f"{p}_BIRTH_DAY",
        ]

        # account last activity timestamp — depends on account create timestamp
        dep_graph[
            PaymentDatasetAttribute(f"{p}_ACCOUNT_LAST_ACTIVITY_TIMESTAMP", H.account_last_activity_timestamp)
        ] = [f"{p}_ACCOUNT_CREATE_TIMESTAMP"]

        # tower geo coordinates — depends on user geo coordinates
        dep_graph[
            PaymentDatasetAttributeGroup(
                (f"{p}_TOWER_LATITUDE", f"{p}_TOWER_LONGITUDE"),
                H.tower_geo_coordinates,
            )
        ] = [f"{p}_GEO_LATITUDE", f"{p}_GEO_LONGITUDE"]

        # account activity events past 30d — depends on account type
        dep_graph[
            PaymentDatasetAttribute(f"{p}_ACCOUNT_ACTIVITY_EVENTS_PAST_30D", H.account_activity_events_past_30d)
        ] = [f"{p}_ACCOUNT_TYPE"]

    return dep_graph


def get_payment_core_attributes(
    participant_prefixes: tuple[str, str],
    dependency_graph: AttributeDependencyGraphType | None = None,
) -> AttributeDependencyGraphType:
    """Register payment-level attributes: ID, timestamps, and status."""
    dep_graph: AttributeDependencyGraphType = {} if dependency_graph is None else dependency_graph
    p0, p1 = participant_prefixes[0].upper(), participant_prefixes[1].upper()

    dep_graph[PaymentDatasetAttribute("PAYMENT_ID", H.payment_id)] = []

    dep_graph[PaymentDatasetAttribute("PAYMENT_INIT_TIMESTAMP", H.payment_init_timestamp)] = [
        f"{p0}_ACCOUNT_LAST_ACTIVITY_TIMESTAMP",
        f"{p1}_ACCOUNT_LAST_ACTIVITY_TIMESTAMP",
    ]

    dep_graph[PaymentDatasetAttribute("PAYMENT_LAST_UPDATE_TIMESTAMP", H.payment_last_update_timestamp)] = [
        "PAYMENT_INIT_TIMESTAMP"
    ]

    dep_graph[PaymentDatasetAttribute("PAYMENT_STATUS", H.payment_status)] = []

    return dep_graph


def get_payment_amount_attributes(
    participant_prefixes: tuple[str, str],
    dependency_graph: AttributeDependencyGraphType | None = None,
) -> AttributeDependencyGraphType:
    """Register exchange-rate and payment-amount attributes."""
    dep_graph: AttributeDependencyGraphType = {} if dependency_graph is None else dependency_graph
    p0, p1 = participant_prefixes[0].upper(), participant_prefixes[1].upper()

    dep_graph[
        PaymentDatasetAttributeGroup(
            (f"{p0}_CCY_{p1}_CCY_RATE", f"{p1}_CCY_{p0}_CCY_RATE"),
            H.currency_exchange_rates,
        )
    ] = [f"{p0}_CURRENCY", f"{p1}_CURRENCY"]

    dep_graph[
        PaymentDatasetAttributeGroup(
            (f"{p0}_AMOUNT", f"{p1}_AMOUNT"),
            H.payment_amounts,
        )
    ] = [f"{p0}_ACCOUNT_TYPE", f"{p0}_CCY_{p1}_CCY_RATE"]

    return dep_graph
