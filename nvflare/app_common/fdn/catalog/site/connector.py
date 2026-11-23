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

"""The site-side reader of one dataset's catalog. A connector is any subclass of CatalogConnector."""

from abc import ABC, abstractmethod
from collections.abc import Callable

from nvflare.apis.dataset_config_spec import DatasetConfig
from nvflare.app_common.fdn.catalog.common.schema import CatalogInfo, ReadLimits

# the connector selected by the dataset's site-configured `type`
BUILTIN_CONNECTORS = {
    "parquet": "nvflare.app_opt.fdn.catalog.parquet_connector.ParquetConnector",
    "csv": "nvflare.app_opt.fdn.catalog.csv_connector.CSVConnector",
    "sql": "nvflare.app_opt.fdn.catalog.sql_connector.SQLConnector",
    "unity-catalog": "nvflare.app_opt.fdn.catalog.unity_catalog_connector.UnityCatalogConnector",
}


class ConnectorError(Exception):
    """A safe, dataset-scoped failure. The message must contain no paths and no source text."""

    def __init__(self, safe_message: str, *, budget: bool = False):
        super().__init__(safe_message)
        # True: a budget stop, which is no evidence about the source.
        # False: the source is absent, inaccessible or invalid.
        self.budget = budget


class CatalogConnector(ABC):
    """Read metadata visible to the configured source identity or file projection.

    The source administrator controls publication through credentials, schemas and readable files. The executor
    validates the returned catalog but does not redact its tables, columns, resources, metadata or lineage.
    Constructed with the site's non-secret connector arguments, with no I/O.
    """

    # what the connector connects to ("parquet", ...); a safe label, never a path or endpoint
    source: str

    @abstractmethod
    def read_catalog(self, dataset: DatasetConfig, secret: Callable[[str], str], limits: ReadLimits) -> CatalogInfo:
        """One complete CatalogInfo for the dataset, or ConnectorError. Never a partial schema.

        Args:
            dataset: the dataset's configuration: its type, uri (and local path), and non-secret options.
            secret: returns one of this dataset's secrets by key, e.g. secret("password"); it cannot reach another
                dataset's secrets.
            limits: the remaining time and record budgets, and the call budget for this read.
        """

    def close(self) -> None:
        pass
