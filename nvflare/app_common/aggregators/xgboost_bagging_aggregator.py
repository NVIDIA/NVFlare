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

import json
import os
import shutil

from nvflare.apis.dxo import DXO, DataKind, from_shareable
from nvflare.apis.fl_constant import FLContextKey, ReservedKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.app_common.abstract.aggregator import Aggregator
from nvflare.app_common.app_constant import AppConstants


class XGBoostBaggingAggregator(Aggregator):
    def __init__(
        self,
    ):
        """Perform bagging aggregation for XGBoost trees.

        The trees are pre-weighted during training.
        Bagging aggregation simply add the new trees to existing global model.

        Args:
            expected_data_kind:
                DataKind for DXO. Defaults to DataKind.XGB_MODEL
                Indicating the tree representation given by XGBoost
        """
        super().__init__()
        self.logger.debug(f"expected data kind: {DataKind.XGB_MODEL}")
        self.history = []
        self.local_models = []
        self.global_model = None
        self.expected_data_kind = DataKind.XGB_MODEL
        self.num_trees = 0
        
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

        contributor_name = shareable.get_peer_prop(key=ReservedKey.IDENTITY_NAME, default="?")
        contribution_round = shareable.get_header(AppConstants.CONTRIBUTION_ROUND)

        rc = shareable.get_return_code()
        if rc and rc != ReturnCode.OK:
            self.log_warning(fl_ctx, f"Contributor {contributor_name} returned rc: {rc}. Disregarding contribution.")
            return False

        if dxo.data_kind != self.expected_data_kind:
            self.log_error(fl_ctx, "expected {} but got {}".format(self.expected_data_kind, dxo.data_kind))
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

        for item in self.history:
            if contributor_name == item["contributor_name"]:
                prev_round = item["round"]
                self.log_warning(
                    fl_ctx,
                    f"discarding DXO from {contributor_name} at round: "
                    f"{contribution_round} as {prev_round} accepted already",
                )
                return False

        data = dxo.data
        if data is None:
            self.log_error(fl_ctx, "no data to aggregate")
            return False
        else:
            model_update = json.loads(data['model'])
            if not self.global_model:
                self.global_model = model_update
                # assume one tree per update
                self.num_trees = 1
            else:
                json_bagging = self.global_model
                 # Always 1 tree, so [0]
                append_info = model_update["learner"]["gradient_booster"]["model"]["trees"][0]
                append_info["id"] = self.num_trees
                json_bagging["learner"]["gradient_booster"]["model"]["trees"].append(append_info)
                json_bagging["learner"]["gradient_booster"]["model"]["tree_info"].append(0)
                self.num_trees += 1

            self.history.append(
                {
                    "contributor_name": contributor_name,
                    "round": contribution_round,
                }
            )
        return True

    def aggregate(self, fl_ctx: FLContext) -> Shareable:
        """Called when workflow determines to generate shareable to send back to contributors

        Args:
            fl_ctx (FLContext): context provided by workflow

        Returns:
            Shareable: the weighted mean of accepted shareables from contributors
        """

        self.log_debug(fl_ctx, "Start aggregation")
        current_round = fl_ctx.get_prop(AppConstants.CURRENT_ROUND)
        site_num = len(self.history)

        self.log_info(fl_ctx, f"aggregating {site_num} update(s) at round {current_round}")

        json_bagging = self.global_model
        json_bagging["learner"]["attributes"]["best_iteration"] = str(self.num_trees - 1)
        json_bagging["learner"]["attributes"]["best_ntree_limit"] = str(self.num_trees)
        json_bagging["learner"]["gradient_booster"]["model"]["gbtree_model_param"]["num_trees"] = str(self.num_trees)
        
        json_bagging["learner"]["gradient_booster"]["model"]["gbtree_model_param"]["num_parallel_tree"] = "1"
 
        as_bytearray = bytearray(json.dumps(json_bagging),'utf-8')
        
        self.history = []
        self.log_debug(fl_ctx, "End aggregation")

        dxo = DXO(data_kind=self.expected_data_kind, data={'model': as_bytearray})
        return dxo.to_shareable()
