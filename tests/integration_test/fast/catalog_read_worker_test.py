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

"""Catalog discovery crosses a real process boundary and returns only the projected metadata."""

import pytest

from nvflare.apis.dataset_config_spec import DatasetConfig
from nvflare.app_common.fdn.catalog.common.limits import ReadLimits
from nvflare.app_opt.fdn.catalog._metadata import ReadBudget
from nvflare.app_opt.fdn.catalog.odcs import read_odcs


def test_csv_worker_preserves_wide_headers_and_charges_each_batch(tmp_path):
    pytest.importorskip("datacontract.imports.csv_importer")
    path = tmp_path / "visits.csv"
    names = [f"visit_{i}" for i in range(129)]
    path.write_text(",".join(names) + "\n" + ",".join(["private"] * len(names)))
    dataset = DatasetConfig("study", "visits", "csv", path.as_uri(), str(path))
    budget = ReadBudget(ReadLimits(120, 4, 200))
    contract = read_odcs(dataset, None, budget, site="one")
    assert contract["schema"] == [{"name": "visits", "properties": [{"name": name} for name in names]}]
    assert budget.calls_left == 1
