# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
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
from nvflare.apis.dxo import DXO, DataKind, from_shareable
from nvflare.apis.fl_constant import ReservedKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.app_common.abstract.aggregator import Aggregator
from nvflare.app_common.app_constant import AppConstants
from sklearn.cluster import KMeans


class _AccuItem(object):
    def __init__(self, client, center, count):
        self.client = client
        self.center = center
        self.count = count


class SKLearnKMeansModelAggregator(Aggregator):
    def __init__(self):
        """Perform accumulated aggregation for linear model parameters by sklearn."""
        super().__init__()
        self.expected_data_kind = DataKind.WEIGHTS
        self.center = None
        self.count = None
        self.n_cluster = 0
        self.accumulator = []
        self.logger.debug(f"expected data kind: {self.expected_data_kind}")

    def accept(self, shareable: Shareable, fl_ctx: FLContext) -> bool:
        """Store shareable and update aggregator's internal state

        Args:
            shareable: information from contributor
            fl_ctx: context provided by workflow

        Returns:
            The first boolean indicates if this shareable is accepted.
            The second boolean indicates if aggregate can be called.
        """
        try:
            dxo = from_shareable(shareable)
        except BaseException:
            self.log_exception(fl_ctx, "shareable data is not a valid DXO")
            return False

        contributor_name = shareable.get_peer_prop(
            key=ReservedKey.IDENTITY_NAME, default="?"
        )
        contribution_round = shareable.get_header(AppConstants.CONTRIBUTION_ROUND)

        rc = shareable.get_return_code()
        if rc and rc != ReturnCode.OK:
            self.log_warning(
                fl_ctx,
                f"Contributor {contributor_name} returned rc: {rc}. Disregarding contribution.",
            )
            return False

        if dxo.data_kind != self.expected_data_kind:
            self.log_error(
                fl_ctx,
                "expected {} but got {}".format(self.expected_data_kind, dxo.data_kind),
            )
            return False

        current_round = fl_ctx.get_prop(AppConstants.CURRENT_ROUND)
        self.log_debug(fl_ctx, f"current_round: {current_round}")
        if contribution_round != current_round:
            self.log_warning(
                fl_ctx,
                f"discarding DXO from {contributor_name} at round: "
                f"{contribution_round}. Current round is: {current_round}",
            )
            return False

        data = dxo.data
        if data is None:
            self.log_error(fl_ctx, "no data to aggregate")
            return False

        if not self._client_in_accumulator(contributor_name):
            self.accumulator.append(
                _AccuItem(contributor_name, data["center"], data["count"])
            )
            accepted = True
        else:
            self.log_info(
                fl_ctx,
                f"Discarded: Current round: {current_round} contributions already include client: {contributor_name}",
            )
            accepted = False
        return accepted

    def _client_in_accumulator(self, client_name):
        return any(client_name == item.client for item in self.accumulator)

    def aggregate(self, fl_ctx: FLContext) -> Shareable:
        """Called when workflow determines to generate shareable to send back to contributors

        Args:
            fl_ctx (FLContext): context provided by workflow

        Returns:
            Shareable: the weighted mean of accepted shareables from contributors
        """
        self.log_debug(fl_ctx, "Start aggregation")
        current_round = fl_ctx.get_prop(AppConstants.CURRENT_ROUND)
        site_num = len(self.accumulator)
        self.log_info(
            fl_ctx, f"aggregating {site_num} update(s) at round {current_round}"
        )

        if current_round == 0:
            # Fist round, collect the information regarding n_feature and n_cluster
            # Initialize the aggregated center and count to all zero
            self.n_cluster = self.accumulator[0].center.shape[0]
            n_feature = self.accumulator[0].center.shape[1]
            self.center = np.zeros([self.n_cluster, n_feature])
            self.count = np.zeros([self.n_cluster])
            # perform one round of KMeans over the submitted centers
            center_collect = []
            count_accu = np.zeros([self.n_cluster])
            for item in self.accumulator:
                center_collect.append(item.center)
                count_accu = count_accu + item.count
            centers = np.concatenate(center_collect)
            kmeans_center_initial = KMeans(n_clusters=self.n_cluster)
            kmeans_center_initial.fit(centers)
            self.center = kmeans_center_initial.cluster_centers_
            self.count = count_accu
        else:
            # Aggregate centers, starting with previous center
            # weight scaling recovered by previous count
            for center_idx in range(self.n_cluster):
                centers_global_rescale = (
                    self.center[center_idx] * self.count[center_idx]
                )
                # Aggregate center, add new center to previous estimate, weighted by counts
                for item in self.accumulator:
                    centers_global_rescale += (
                        item.center[center_idx] * item.count[center_idx]
                    )
                    self.count[center_idx] += item.count[center_idx]
                # Rescale to compute mean of all points (old and new combined)
                alpha = 1 / self.count[center_idx]
                centers_global_rescale *= alpha
                # Update the global center
                self.center[center_idx] = centers_global_rescale
        # Reset accumulator for next round,
        # but not the center and count, which will be used as the starting point of the next round
        self.accumulator = []
        self.log_debug(fl_ctx, "End aggregation")

        dxo = DXO(data_kind=self.expected_data_kind, data={"center": self.center})
        return dxo.to_shareable()
