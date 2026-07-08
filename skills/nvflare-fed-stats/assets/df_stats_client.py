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

"""Template for a tabular federated statistics client.

Adapt this template when generating the client:
- read the user's data inside ``load_data()``: reuse their loading logic when
  a script provides it, otherwise a plain pandas read; headerless data needs
  explicit user-supplied feature names via ``names=``;
- porting boundary: population-defining prep (cohort filters, derived
  columns, splits, missing-value encodings) is ported — report imputation,
  it shifts every statistic; summary computation (describe/agg/groupby
  aggregations, histogram/quantile/variance math) is deleted;
- return one DataFrame per dataset name (for example ``train`` and ``test``);
  the data-first default is one entry;
- parameterize the data location by site identity; do not hardcode one
  site's absolute path;
- do not port statistic computations from any script: ``DFStatisticsCore``
  computes every supported statistic from the returned DataFrames.
"""

from typing import Dict, Optional

import pandas as pd

from nvflare.apis.fl_context import FLContext
from nvflare.app_opt.statistics.df.df_core_statistics import DFStatisticsCore


class TabularStatistics(DFStatisticsCore):
    def __init__(self, filename: str, data_root_dir: str):
        """Local statistics generator for tabular data.

        Args:
            filename: per-site data file name under the site's data directory.
            data_root_dir: root directory holding one subdirectory per site.
        """
        super().__init__()
        self.filename = filename
        self.data_root_dir = data_root_dir
        self.data: Optional[Dict[str, pd.DataFrame]] = None

    def load_data(self, fl_ctx: FLContext) -> Dict[str, pd.DataFrame]:
        site_name = fl_ctx.get_identity_name()
        data_path = f"{self.data_root_dir}/{site_name}/{self.filename}"
        self.log_info(fl_ctx, f"loading data for site {site_name}")

        # ADAPTATION POINT: use the user's own loading logic when a script
        # provides it (their read_csv/read_parquet options, cleaning, and
        # dataset split); otherwise a plain read. Headerless data requires
        # user-supplied names: pd.read_csv(data_path, names=feature_names).
        df: pd.DataFrame = pd.read_csv(data_path)

        # ADAPTATION POINT: preserve a script's dataset split when it has
        # one; the data-first default returns {"data": df}.
        return {"data": df}

    def initialize(self, fl_ctx: FLContext):
        self.data = self.load_data(fl_ctx)
