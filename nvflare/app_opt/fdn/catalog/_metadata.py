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

"""Shared metadata-only connector validation and read accounting."""

import time
from collections.abc import Mapping
from pathlib import Path

from nvflare.app_common.fdn.catalog.common.schema import PROFILE_KEYS, CatalogInfo, ValidationError, validate
from nvflare.app_common.fdn.catalog.site.connector import ConnectorError


class ReadBudget:
    def __init__(self, limits):
        self.limits = limits
        self.deadline = time.monotonic() + limits.max_duration_s
        self.calls = 0

    def remaining(self):
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise ConnectorError("metadata read budget exceeded", budget=True)
        return remaining

    def request(self):
        self.remaining()
        self.calls += 1
        if self.calls > self.limits.max_calls:
            raise ConnectorError("metadata call budget exceeded", budget=True)

    def records(self, count):
        self.remaining()
        if count > self.limits.max_records:
            raise ConnectorError("metadata record budget exceeded", budget=True)


def local_path(dataset) -> Path:
    """The local path of a dataset whose ``uri`` is a file URI; a network URL cannot be read by a file connector."""
    if dataset.path is None:
        raise ConnectorError("this connector reads local files and needs a file:// uri")
    return Path(dataset.path)


def selection(options, allowed, required=()):
    """The dataset's options as a plain dict, checked against the keys this connector accepts. The dataset
    configuration API returns read-only mappings and tuples; "tables" comes back as a list."""
    if not isinstance(options, Mapping) or set(options) - set(allowed) or set(required) - set(options):
        raise ConnectorError("dataset options are invalid for this connector")
    options = dict(options)
    for key, value in options.items():
        if key == "tables":
            if (
                not isinstance(value, (list, tuple))
                or not value
                or not all(isinstance(name, str) and name for name in value)
                or len(set(value)) != len(value)
            ):
                raise ConnectorError("table selection must be a nonempty list of unique names")
            options[key] = list(value)
        elif key in ("schema", "catalog") and (not isinstance(value, str) or not value):
            raise ConnectorError("metadata selection names must be nonempty strings")
    return options


def build_catalog(dataset, source, tables, limits):
    """A complete, validated catalog of the structure the source reported. Descriptive and semantic fields stay
    unknown: connectors publish what the source itself exposes, nothing a site types in beside the data."""
    metadata = {
        "display_name": dataset,
        "description": None,
        "owner": None,
        "modalities": ["tabular"],
        "operations": [],
        "source_version": None,
        "lineage": [],
        "tables": tables,
        "resources": [],
        "semantic_profile": {key: None for key in PROFILE_KEYS},
        "provenance": {"source": source, "declaration_ref": None, "validation_ref": None},
    }
    try:
        catalog = CatalogInfo.from_wire({"dataset": dataset, "metadata": metadata, "observed_at": None})
    except ValidationError:
        raise ConnectorError("catalog metadata is invalid") from None
    if catalog.record_count() > limits.max_records:
        raise ConnectorError("metadata record budget exceeded", budget=True)
    try:
        validate(catalog, limits)
    except ValidationError:
        raise ConnectorError("catalog metadata is invalid") from None
    return catalog
