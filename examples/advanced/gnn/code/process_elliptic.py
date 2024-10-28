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

import numpy as np
import torch


def process_ellipitc(df_features, df_edges, df_classes):
    # map "illicit" to label 1, "licit" to label 0, and "unknown" to label 2
    df_classes["class"] = df_classes["class"].map({1: 1, 2: 0, 3: 2})

    # merging dataframes
    df_merge = df_features.merge(df_classes, how="inner", on="txId")
    df_merge = df_merge.sort_values(by="txId").reset_index(drop=True)

    # map txIds to indexes
    nodes = df_merge["txId"].values
    map_id = {j: i for i, j in enumerate(nodes)}  # mapping nodes to indexes

    edges = df_edges.copy()
    edges.txId1 = edges.txId1.map(map_id)
    edges.txId2 = edges.txId2.map(map_id)
    edges = edges.astype(int)

    edge_index = np.array(edges.values).T

    # for undirected graph
    edge_index = torch.tensor(edge_index, dtype=torch.long).contiguous()
    weights = torch.tensor([1] * edge_index.shape[1], dtype=torch.double)

    # txIds mapped to corresponding indexes,
    # exclude "unknown" nodes for loss computation
    node_features = df_merge.drop(["txId"], axis=1).copy()
    classified_idx = node_features["class"].loc[node_features["class"] != 2].index
    unclassified_idx = node_features["class"].loc[node_features["class"] == 2].index

    # labels for classification
    labels = node_features["class"].values

    # features for nodes
    for key in node_features.keys():
        if "feature" not in key:
            node_features = node_features.drop([key], axis=1)
    node_features = torch.tensor(np.array(node_features.values))
    y_train = labels[classified_idx]

    return node_features, classified_idx, unclassified_idx, edge_index, weights, labels, y_train
