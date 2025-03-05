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

from nvflare.apis.fl_context import FLContext
from nvflare.app_common.aggregators.intime_accumulate_model_aggregator import InTimeAccumulateWeightedAggregator

from .autofedrl_constants import AutoFedRLConstants


class AutoFedRLWeightedAggregator(InTimeAccumulateWeightedAggregator):
    """Perform accumulated weighted aggregation with support for updating aggregation weights.
            Used for Auto-FedRL implementation (https://arxiv.org/abs/2203.06338).

    Shares arguments with base class
    """

    def update_aggregation_weights(self, fl_ctx: FLContext):
        """Called when workflow determines to update aggregation weights

        Args:
            fl_ctx (FLContext): context provided by workflow

        Returns:
            None
        """
        received_aggregation_weights = None
        hps = fl_ctx.get_prop(AutoFedRLConstants.HYPERPARAMTER_COLLECTION)
        if hps is not None:
            received_aggregation_weights = hps.get("aw")
        # assign current aggregation weights to aggregator
        if received_aggregation_weights is not None:
            # TODO: Here, we assume contributor_name is "site-*".
            # this will be wrong if contributor_name is not in this pattern.
            aggregation_weights_dict = {
                f"site-{i + 1}": received_aggregation_weights[i] for i in range(len(received_aggregation_weights))
            }
            for key in self.expected_data_kind.keys():
                self.dxo_aggregators[key].aggregation_weights = aggregation_weights_dict
                self.log_info(fl_ctx, f"Assign current aggregation weights to aggregator: {key}")
        else:
            self.log_warning(fl_ctx, "Received aggregation weights are None.")
