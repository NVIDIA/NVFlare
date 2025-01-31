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

from typing import Dict

import numpy as np
from sklearn.cluster import KMeans

from nvflare.apis.dxo import DXO, DataKind
from nvflare.apis.fl_context import FLContext
from nvflare.app_common.aggregators.assembler import Assembler
from nvflare.app_common.app_constant import AppConstants


class KMeansAssembler(Assembler):
    def __init__(self):
        super().__init__(data_kind=DataKind.WEIGHTS)
        # Aggregator needs to keep record of historical
        # center and count information for mini-batch kmeans
        self.center = None
        self.count = None
        self.n_cluster = 0

    def get_model_params(self, dxo: DXO):
        data = dxo.data
        return {"center": data["center"], "count": data["count"]}

    def assemble(self, data: Dict[str, dict], fl_ctx: FLContext) -> DXO:
        current_round = fl_ctx.get_prop(AppConstants.CURRENT_ROUND)
        if current_round == 0:
            # First round, collect the information regarding n_feature and n_cluster
            # Initialize the aggregated center and count to all zero
            client_0 = list(self.collection.keys())[0]
            self.n_cluster = self.collection[client_0]["center"].shape[0]
            n_feature = self.collection[client_0]["center"].shape[1]
            self.center = np.zeros([self.n_cluster, n_feature])
            self.count = np.zeros([self.n_cluster])
            # perform one round of KMeans over the submitted centers
            # to be used as the original center points
            # no count for this round
            center_collect = []
            for _, record in self.collection.items():
                center_collect.append(record["center"])
            centers = np.concatenate(center_collect)
            kmeans_center_initial = KMeans(n_clusters=self.n_cluster)
            kmeans_center_initial.fit(centers)
            self.center = kmeans_center_initial.cluster_centers_
        else:
            # Mini-batch k-Means step to assemble the received centers
            for center_idx in range(self.n_cluster):
                centers_global_rescale = self.center[center_idx] * self.count[center_idx]
                # Aggregate center, add new center to previous estimate, weighted by counts
                for _, record in self.collection.items():
                    centers_global_rescale += record["center"][center_idx] * record["count"][center_idx]
                    self.count[center_idx] += record["count"][center_idx]
                # Rescale to compute mean of all points (old and new combined)
                alpha = 1 / self.count[center_idx]
                centers_global_rescale *= alpha
                # Update the global center
                self.center[center_idx] = centers_global_rescale
        params = {"center": self.center}
        dxo = DXO(data_kind=self.expected_data_kind, data=params)

        return dxo
