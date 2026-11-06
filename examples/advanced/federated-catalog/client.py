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

"""Each site's reader generates its own catalog from local Parquet metadata."""

from nvflare.app_common.fdn.catalog.site.reader import CatalogReader
from nvflare.app_opt.fdn.catalog.parquet_connector import ParquetConnector


class ParquetCatalogReader(CatalogReader):
    def read_catalog(self, dataset, mounts, limits, options=None):
        connector = ParquetConnector(mounts)
        try:
            return connector.read_catalog(dataset, limits, options)
        finally:
            connector.close()
