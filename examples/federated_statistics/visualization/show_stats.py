# Copyright (c) 2022, NVIDIA CORPORATION.
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

"""
 Copy these codes to jupyter notebook to run
 
"""


def get_target_features(all_features, white_list_features=[]):
    target_features = white_list_features
    if not white_list_features:
        target_features = all_features
    return target_features


def show_stats(data, white_list_features=[]):
    all_features = [k for k in data]
    target_features = get_target_features(all_features, white_list_features)

    for feature in target_features:
        print(f"\n{feature}\n")
        feature_metrics = data[feature]
        df = pd.DataFrame.from_dict(feature_metrics)
        display(df)


def prepare_histogram_data(data, white_list_features=[]):
    all_features = [k for k in data]
    target_features = get_target_features(all_features, white_list_features)

    feature_hists = {}
    feature_edges = {}

    for feature in target_features:
        xs = data[feature]['histogram']
        hists = {}
        feature_edges[feature] = []
        for i, ds in enumerate(xs):
            ds_hist = xs[ds]
            ds_bucket_counts = []
            for bucket in ds_hist:
                if i == 0:
                    feature_edges[feature].append(bucket[0])
                ds_bucket_counts.append(bucket[2])
                hists[ds] = ds_bucket_counts
        feature_hists[feature] = hists

    return feature_hists, feature_edges


def show_histograms(data, white_list_features=[]):
    (hists, edges) = prepare_histogram_data(data)

    all_features = [k for k in edges]
    target_features = get_target_features(all_features, white_list_features)

    for feature in target_features:
        hist_data = hists[feature]
        index = edges[feature]
        df = pd.DataFrame(hist_data, index=index)
        axes = df.plot.bar(rot=30)
        axes = df.plot.bar(rot=30, subplots=True)
