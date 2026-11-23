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

from nvflare.app_common.fdn.catalog.site.source_error import CatalogSourceError


class ReadBudget:
    def __init__(self, limits):
        self.limits = limits
        self.deadline = time.monotonic() + limits.max_duration_s
        self.calls = 0

    def remaining(self):
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise CatalogSourceError("metadata read budget exceeded", budget=True)
        return remaining

    def request(self):
        self.remaining()
        self.calls += 1
        if self.calls > self.limits.max_calls:
            raise CatalogSourceError("metadata call budget exceeded", budget=True)

    def records(self, count):
        self.remaining()
        if count > self.limits.max_records:
            raise CatalogSourceError("metadata record budget exceeded", budget=True)


def local_path(dataset) -> Path:
    """The local path of a dataset whose ``uri`` is a file URI; a network URL cannot be read by a file connector."""
    if dataset.path is None:
        raise CatalogSourceError("this connector reads local files and needs a file:// uri")
    return Path(dataset.path)


def selection(options, allowed, required=()):
    """The dataset's options as a plain dict, checked against the keys this connector accepts. The dataset
    configuration API returns read-only mappings and tuples; "tables" comes back as a list."""
    if not isinstance(options, Mapping) or set(options) - set(allowed) or set(required) - set(options):
        raise CatalogSourceError("dataset options are invalid for this connector")
    options = dict(options)
    for key, value in options.items():
        if key == "tables":
            if (
                not isinstance(value, (list, tuple))
                or not value
                or not all(isinstance(name, str) and name for name in value)
                or len(set(value)) != len(value)
            ):
                raise CatalogSourceError("table selection must be a nonempty list of unique names")
            options[key] = list(value)
        elif key in ("schema", "catalog") and (not isinstance(value, str) or not value):
            raise CatalogSourceError("metadata selection names must be nonempty strings")
    return options
