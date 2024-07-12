# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
import pandas as pd
from hierarchical_stats.app.custom.hierarchical_stats import HierarchicalStats

from nvflare.apis.fl_context import FLContext


class TestHierarchicalStats:
    def setup_method(self) -> None:
        # mock the load_data with fake data
        self.local_stats_gen = HierarchicalStats()
        self.local_stats_gen.load_data = self.load_data
        self.local_stats_gen.initialize(fl_ctx=None)

    def teardown_method(self) -> None:
        pass

    def load_data(self, fl_ctx: FLContext = None):
        # initialize list of lists
        data = [[1, 0, 75.75], [1, 0, 65.77], [0, 1, 43.44]]
        df = pd.DataFrame(data, columns=["Pass", "Fail", "Percentage"])
        return {"default_set": df}

    def test_get_features(self):
        fs = self.local_stats_gen.features()
        assert len(fs.keys()) == 1
        assert "default_set" in fs.keys()
        assert [f.feature_name for f in fs["default_set"]] == ["Pass", "Fail", "Percentage"]

    def test_get_count(self):
        count = self.local_stats_gen.count("default_set", "Pass")
        assert count == 2
        count = self.local_stats_gen.count("default_set", "Fail")
        assert count == 1
        count = self.local_stats_gen.count("default_set", "Percentage")
        assert count == 3
