# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

from typing import Dict

from nvflare.fuel.utils.import_utils import optional_import


class Visualization:
    def import_modules(self):
        display, import_flag = optional_import(module="IPython.display", name="display")
        if not import_flag:
            print(display.failure)
        pd, import_flag = optional_import(module="pandas")
        if not import_flag:
            print(pd.failure)
        return display, pd

    def show_stats(self, data, white_list_features=[]):
        display, pd = self.import_modules()
        all_features = [k for k in data]
        target_features = self._get_target_features(all_features, white_list_features)
        for feature in target_features:
            print(f"\n{feature}\n")
            feature_metrics = data[feature]
            df = pd.DataFrame.from_dict(feature_metrics)
            display(df)

    def show_histograms(self, data, display_format="sample_count", white_list_features=[], plot_type="both"):
        feature_dfs = self.get_histogram_dataframes(data, display_format, white_list_features)
        self.show_dataframe_plots(feature_dfs, plot_type)

    def show_dataframe_plots(self, feature_dfs, plot_type="both"):
        for feature in feature_dfs:
            df = feature_dfs[feature]
            if plot_type == "both":
                axes = df.plot.line(rot=40, title=feature)
                axes = df.plot.line(rot=40, subplots=True, title=feature)
            elif plot_type == "main":
                axes = df.plot.line(rot=40, title=feature)
            elif plot_type == "subplot":
                axes = df.plot.line(rot=40, subplots=True, title=feature)
            else:
                print(f"not supported plot type: '{plot_type}'")

    def get_histogram_dataframes(self, data, display_format="sample_count", white_list_features=[]) -> Dict:
        display, pd = self.import_modules()
        (hists, edges) = self._prepare_histogram_data(data, display_format, white_list_features)
        all_features = [k for k in edges]
        target_features = self._get_target_features(all_features, white_list_features)

        feature_dfs = {}
        for feature in target_features:
            hist_data = hists[feature]
            index = edges[feature]
            df = pd.DataFrame(hist_data, index=index)
            feature_dfs[feature] = df

        return feature_dfs

    def _prepare_histogram_data(self, data, display_format="sample_count", white_list_features=[]):
        all_features = [k for k in data]
        target_features = self._get_target_features(all_features, white_list_features)

        feature_hists = {}
        feature_edges = {}

        for feature in target_features:
            xs = data[feature]["histogram"]
            hists = {}
            feature_edges[feature] = []
            for i, ds in enumerate(xs):
                ds_hist = xs[ds]
                ds_bucket_counts = []

                for bucket in ds_hist:
                    if i == 0:
                        feature_edges[feature].append(bucket[0])
                    if display_format == "percent":
                        sum_value = self.sum_counts_in_histogram(ds_hist)
                        ds_bucket_counts.append(bucket[2] / sum_value)
                    else:
                        ds_bucket_counts.append(bucket[2])
                    hists[ds] = ds_bucket_counts
            feature_hists[feature] = hists

        return feature_hists, feature_edges

    def sum_counts_in_histogram(self, hist):
        sum_value = 0
        for bucket in hist:
            sum_value += bucket[2]
        return sum_value

    def _get_target_features(self, all_features, white_list_features=[]):
        target_features = white_list_features
        if not white_list_features:
            target_features = all_features
        return target_features
