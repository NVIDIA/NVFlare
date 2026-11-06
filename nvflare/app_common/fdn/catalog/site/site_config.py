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

"""The site's own catalog configuration: which connector reads each of its datasets, and within what limits.

It lives next to the Study policy, in a location only the site owner controls. A job cannot supply or change any of
it. Secrets never appear here; a connector reads them from the Study runtime's environment.
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from nvflare.app_common.fdn.catalog.common.schema import DEFAULT_LIMITS, ReadLimits

SITE_CONFIG_FORMAT_VERSION = 1
SITE_CONFIG_FILE = "catalog_site.json"
PARQUET_CONNECTOR = "parquet"  # built in, so a site with only mounted Parquet data needs no connectors at all
SQLITE_CONNECTOR = "sqlite"
SQLITE_CONNECTOR_CLASS = "nvflare.app_opt.fdn.catalog.sqlite_connector.SQLiteConnector"
CSV_CONNECTOR_CLASS = "nvflare.app_opt.fdn.catalog.csv_connector.CSVConnector"
DEFAULT_CONNECTOR_CLASS = "nvflare.app_opt.fdn.catalog.parquet_connector.ParquetConnector"
_KEYS = frozenset({"format_version", "connectors", "default_connector", "studies", "limits", "installed_operations"})
_CONNECTOR_KEYS = frozenset({"class", "args"})
_BINDING_KEYS = frozenset({"connector", "options"})
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
# the dataset-name rule of the Study dataset mounts (nvflare/app_opt/job_launcher/study_data.py)
DATASET_NAME = re.compile(r"^[a-z0-9](?:[a-z0-9_-]{0,61}[a-z0-9])?$")


class SiteConfigError(Exception):
    """The site configuration is missing, unreadable or invalid. There is no fallback."""


@dataclass(frozen=True)
class ConnectorSpec:
    path: str  # a CatalogConnector class path
    args: dict = field(default_factory=dict)  # non-secret constructor arguments


@dataclass(frozen=True)
class DatasetBinding:
    connector: str  # the name of a connector in SiteConfig.connectors
    options: dict = field(default_factory=dict)  # non-secret, per dataset (a schema, a table list)


@dataclass(frozen=True)
class SiteConfig:
    connectors: dict[str, ConnectorSpec] = field(
        default_factory=lambda: {
            PARQUET_CONNECTOR: ConnectorSpec(DEFAULT_CONNECTOR_CLASS),
            SQLITE_CONNECTOR: ConnectorSpec(SQLITE_CONNECTOR_CLASS),
            "csv": ConnectorSpec(CSV_CONNECTOR_CLASS),
        }
    )
    default_connector: str = PARQUET_CONNECTOR  # reads a mounted dataset that has no binding
    studies: dict[str, dict[str, DatasetBinding]] = field(default_factory=dict)
    limits: ReadLimits = field(default_factory=lambda: ReadLimits(**DEFAULT_LIMITS))
    installed_operations: frozenset[str] = frozenset()

    def bindings(self, study: str) -> dict[str, DatasetBinding]:
        """The datasets the file binds to a connector for this Study."""
        return self.studies.get(study, {})


def _error(path: Path, message: str) -> SiteConfigError:
    return SiteConfigError(f"invalid site catalog config '{path}': {message}")


def _connector(path: Path, name, spec) -> ConnectorSpec:
    if not isinstance(name, str) or not _NAME.match(name):
        raise _error(path, "a connector name must be letters, digits, '.', '_' or '-'")
    if not isinstance(spec, dict) or set(spec) - _CONNECTOR_KEYS:
        raise _error(path, f"connector '{name}' must be an object with only {sorted(_CONNECTOR_KEYS)}")
    class_path = spec.get("class")
    if not isinstance(class_path, str) or not class_path:
        raise _error(path, f"connector '{name}' needs a class path")
    args = spec.get("args", {})
    if not isinstance(args, dict):
        raise _error(path, f"connector '{name}' args must be an object")
    return ConnectorSpec(class_path, args)


def _binding(path: Path, study, dataset, spec, connectors) -> DatasetBinding:
    where = f"study '{study}' dataset '{dataset}'"
    if not isinstance(dataset, str) or not DATASET_NAME.match(dataset):
        raise _error(path, f"study '{study}' has a dataset name that breaks the dataset-name rule")
    if not isinstance(spec, dict) or set(spec) - _BINDING_KEYS or "connector" not in spec:
        raise _error(path, f"{where} must be an object with a connector and optional options")
    if not isinstance(spec["connector"], str) or spec["connector"] not in connectors:
        raise _error(path, f"{where} names a connector that is not defined")
    options = spec.get("options", {})
    if not isinstance(options, dict):
        raise _error(path, f"{where} options must be an object")
    if options and connectors[spec["connector"]].path in {
        DEFAULT_CONNECTOR_CLASS,
        SQLITE_CONNECTOR_CLASS,
        CSV_CONNECTOR_CLASS,
    }:
        raise _error(path, f"{where}: built-in connectors accept no options; keep declarations beside source data")
    return DatasetBinding(spec["connector"], options)


def load_site_config(path: Path) -> SiteConfig:
    """Read and validate the file. Not cached: a replaced file is seen on the next read."""
    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise SiteConfigError(f"cannot read site catalog config '{path}': {type(e).__name__}") from None
    if not isinstance(data, dict):
        raise _error(path, "must be an object")
    unknown = set(data) - _KEYS
    if unknown:
        raise _error(path, f"unsupported keys {sorted(unknown)}")
    version = data.get("format_version")
    if type(version) is not int or version != SITE_CONFIG_FORMAT_VERSION:
        raise _error(path, f"format_version must be {SITE_CONFIG_FORMAT_VERSION}")

    named = data.get("connectors", {})
    if not isinstance(named, dict):
        raise _error(path, "connectors must be an object")
    connectors = SiteConfig().connectors | {n: _connector(path, n, s) for n, s in named.items()}
    default = data.get("default_connector", PARQUET_CONNECTOR)
    if not isinstance(default, str) or default not in connectors:
        raise _error(path, "default_connector names a connector that is not defined")

    studies = data.get("studies", {})
    if not isinstance(studies, dict):
        raise _error(path, "studies must be an object")
    bound = {}
    for study, datasets in studies.items():
        if not isinstance(study, str) or not study or not isinstance(datasets, dict):
            raise _error(path, "each study must map dataset names to bindings")
        bound[study] = {d: _binding(path, study, d, s, connectors) for d, s in datasets.items()}

    limits = data.get("limits", {})
    if (
        not isinstance(limits, dict)
        or set(limits) - set(DEFAULT_LIMITS)
        or not all(type(v) is int and v > 0 for v in limits.values())
    ):
        raise _error(path, f"limits must map {sorted(DEFAULT_LIMITS)} to positive integers")
    operations = data.get("installed_operations", [])
    if not isinstance(operations, list) or not all(isinstance(x, str) and x for x in operations):
        raise _error(path, "installed_operations must be a list of non-empty strings")
    return SiteConfig(
        connectors=connectors,
        default_connector=default,
        studies=bound,
        limits=ReadLimits(**{**DEFAULT_LIMITS, **limits}),
        installed_operations=frozenset(operations),
    )
