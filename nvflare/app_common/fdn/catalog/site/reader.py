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

"""The user-supplied catalog reader, analogous to a federated statistics generator."""

from abc import ABC, abstractmethod
from pathlib import Path

from nvflare.apis.fl_component import FLComponent
from nvflare.app_common.fdn.catalog.common.schema import CatalogInfo, ReadLimits


class CatalogReader(FLComponent, ABC):
    @abstractmethod
    def read_catalog(
        self, dataset: str, mounts: dict[str, Path], limits: ReadLimits, options: dict | None = None
    ) -> CatalogInfo:
        """Read metadata locally. The executor supplies site-owned bindings and
        applies publication policy and validation to the returned CatalogInfo.
        Do not return data rows or credentials. Secrets belong in the site environment.
        Release resources opened for a read in a finally block.
        """


class ReaderConnector:
    """Bind a reader component to the current site's dataset paths."""

    def __init__(self, reader, mounts):
        self.reader = reader
        self.mounts = mounts

    def read_catalog(self, dataset, limits, options=None):
        return self.reader.read_catalog(dataset, self.mounts, limits, options)

    def close(self):
        pass
