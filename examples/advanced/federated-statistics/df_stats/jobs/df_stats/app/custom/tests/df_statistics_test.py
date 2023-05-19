# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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
from df_stats.custom.df_statistics import DFStatistics

from nvflare.apis.fl_context import FLContext


class TestDFStatistics:
    def setup_method(self) -> None:
        # mock the load_data with fake data
        self.local_stats_gen = DFStatistics(data_path="data.csv")
        self.local_stats_gen.load_data = self.load_data
        self.local_stats_gen.initialize(fl_ctx=None)

    def teardown_method(self) -> None:
        pass

    def load_data(self, fl_ctx: FLContext = None):
        # initialize list of lists
        data = [["tom", 10, 4], ["nick", 15, 7], ["juli", 14, 8]]
        train_df = pd.DataFrame(data, columns=["Name", "Age", "Edu"])
        data = [["sam", 90, 20], ["jack", 75, 20], ["sara", 44, 13]]
        test_df = pd.DataFrame(data, columns=["Name", "Age", "Edu"])
        return {"train": train_df, "test": test_df}

    def test_get_features(self):
        fs = self.local_stats_gen.features()
        assert len(fs.keys()) == 2
        assert "train" in fs.keys()
        assert "test" in fs.keys()
        assert fs["train"] == fs["test"]
        assert [f.feature_name for f in fs["train"]] == ["Name", "Age", "Edu"]

    def test_get_count(self):
        count = self.local_stats_gen.count("train", "Age")
        assert count == 3
