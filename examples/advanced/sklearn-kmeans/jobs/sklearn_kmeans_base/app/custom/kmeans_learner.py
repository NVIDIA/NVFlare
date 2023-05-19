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

from typing import Optional, Tuple

from sklearn.cluster import KMeans, MiniBatchKMeans, kmeans_plusplus
from sklearn.metrics import homogeneity_score

from nvflare.apis.fl_context import FLContext
from nvflare.app_common.abstract.learner_spec import Learner
from nvflare.app_opt.sklearn.data_loader import load_data_for_range


class KMeansLearner(Learner):
    def __init__(
        self,
        data_path: str,
        train_start: int,
        train_end: int,
        valid_start: int,
        valid_end: int,
        random_state: int = None,
        max_iter: int = 1,
        n_init: int = 1,
        reassignment_ratio: int = 0,
    ):
        super().__init__()
        self.data_path = data_path
        self.train_start = train_start
        self.train_end = train_end
        self.valid_start = valid_start
        self.valid_end = valid_end

        self.random_state = random_state
        self.max_iter = max_iter
        self.n_init = n_init
        self.reassignment_ratio = reassignment_ratio
        self.train_data = None
        self.valid_data = None
        self.n_samples = None
        self.n_clusters = None

    def load_data(self) -> dict:
        train_data = load_data_for_range(self.data_path, self.train_start, self.train_end)
        valid_data = load_data_for_range(self.data_path, self.valid_start, self.valid_end)
        return {"train": train_data, "valid": valid_data}

    def initialize(self, parts: dict, fl_ctx: FLContext):
        data = self.load_data()
        self.train_data = data["train"]
        self.valid_data = data["valid"]
        # train data size, to be used for setting
        # NUM_STEPS_CURRENT_ROUND for potential use in aggregation
        self.n_samples = data["train"][-1]
        # note that the model needs to be created every round
        # due to the available API for center initialization

    def train(self, curr_round: int, global_param: Optional[dict], fl_ctx: FLContext) -> Tuple[dict, dict]:
        # get training data, note that clustering is unsupervised
        # so only x_train will be used
        (x_train, y_train, train_size) = self.train_data
        if curr_round == 0:
            # first round, compute initial center with kmeans++ method
            # model will be None for this round
            self.n_clusters = global_param["n_clusters"]
            center_local, _ = kmeans_plusplus(x_train, n_clusters=self.n_clusters, random_state=self.random_state)
            kmeans = None
            params = {"center": center_local, "count": None}
        else:
            center_global = global_param["center"]
            # following rounds, local training starting from global center
            kmeans = MiniBatchKMeans(
                n_clusters=self.n_clusters,
                batch_size=self.n_samples,
                max_iter=self.max_iter,
                init=center_global,
                n_init=self.n_init,
                reassignment_ratio=self.reassignment_ratio,
                random_state=self.random_state,
            )
            kmeans.fit(x_train)
            center_local = kmeans.cluster_centers_
            count_local = kmeans._counts
            params = {"center": center_local, "count": count_local}
        return params, kmeans

    def validate(self, curr_round: int, global_param: Optional[dict], fl_ctx: FLContext) -> Tuple[dict, dict]:
        # local validation with global center
        # fit a standalone KMeans with just the given center
        center_global = global_param["center"]
        kmeans_global = KMeans(n_clusters=self.n_clusters, init=center_global, n_init=1)
        kmeans_global.fit(center_global)
        # get validation data, both x and y will be used
        (x_valid, y_valid, valid_size) = self.valid_data
        y_pred = kmeans_global.predict(x_valid)
        homo = homogeneity_score(y_valid, y_pred)
        self.log_info(fl_ctx, f"Homogeneity {homo:.4f}")
        metrics = {"Homogeneity": homo}
        return metrics, kmeans_global

    def finalize(self, fl_ctx: FLContext) -> None:
        # freeing resources in finalize
        del self.train_data
        del self.valid_data
        self.log_info(fl_ctx, "Freed training resources")
