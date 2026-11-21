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

"""Builds the DatasetConfig a connector receives, through the dataset configuration API, as the executor does."""

from pathlib import Path

from nvflare.app_common.dataset_config.api import parse_study_config


def dataset_config(name="ds", type_="parquet", uri=None, path=None, options=None, secrets=None):
    """The dataset and its secret reader. ``path`` becomes a file:// uri; ``secrets`` maps keys to variable names."""
    entry = {"type": type_, "uri": uri if uri is not None else Path(path).resolve().as_uri()}
    if options is not None:
        entry["options"] = options
    if secrets is not None:
        entry["secrets"] = secrets
    config = parse_study_config(
        {"format_version": 1, "studies": {"study1": {"datasets": {name: entry}}}}, "study1", "datasets.yaml"
    )
    return config.get_dataset(name), lambda key: config.get_secret(name, key)


def read(connector, limits, **kwargs):
    dataset, secret = dataset_config(**kwargs)
    return connector.read_catalog(dataset, secret, limits)
