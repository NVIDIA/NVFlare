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
from pathlib import Path

from nvflare.app_common.fdn.catalog.common.schema import CatalogInfo, ReadLimits


class ConnectorError(Exception):
    """A safe, dataset-scoped failure. The message must contain no paths and no source text."""

    def __init__(self, safe_message: str, *, budget: bool = False):
        super().__init__(safe_message)
        # True: a budget or cancellation stop, which is no evidence about the source.
        # False: the source is absent, inaccessible or invalid.
        self.budget = budget


class CatalogConnector(ABC):
    # what the connector connects to ("parquet", ...); a safe label, never a path or endpoint
    source: str

    def __init__(self, datasets: dict[str, Path]):
        """datasets maps each current Study dataset name to its mount (empty for a Study without mounts). No I/O
        here."""
        self._datasets = dict(datasets)

    @abstractmethod
    def read_catalog(self, dataset: str, limits: ReadLimits, options: dict | None = None) -> CatalogInfo:
        """One complete CatalogInfo for the dataset, or ConnectorError. Never a partial schema. options are the
        non-secret settings the site's catalog_site.json gives this dataset (for example a schema or table list);
        a mounted dataset has none."""

    def close(self) -> None:
        pass
