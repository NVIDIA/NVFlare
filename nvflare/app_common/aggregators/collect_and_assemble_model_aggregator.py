# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

from typing import Optional

from nvflare.apis.dxo import from_shareable
from nvflare.apis.shareable import ReturnCode
from nvflare.app_common.abstract.fl_model import FLModel, ParamsType
from nvflare.app_common.aggregators.assembler import Assembler
from nvflare.app_common.aggregators.model_aggregator import ModelAggregator
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.utils.fl_model_utils import FLModelUtils


class CollectAndAssembleModelAggregator(ModelAggregator):
    """ModelAggregator adapter for CollectAndAssemble pattern.

    This aggregator bridges the gap between FLModel-based workflows (FedAvg)
    and Assembler-based custom aggregation logic (e.g., K-Means, SVM).

    It wraps an Assembler component and:
    1. Collects FLModel results from clients
    2. Converts them to the format expected by the Assembler
    3. Delegates aggregation to the Assembler
    4. Returns the aggregated result as FLModel

    This enables custom aggregation algorithms to work with the modern
    FedAvg workflow while maintaining InTime aggregation where possible.

    Args:
        assembler_id: ID of the Assembler component to use for aggregation.
    """

    def __init__(self, assembler_id: str):
        super().__init__()
        self.assembler_id = assembler_id
        self.assembler: Optional[Assembler] = None

    def accept_model(self, model: FLModel) -> None:
        """Accept one FLModel from a client.

        Args:
            model: FLModel received from a client
        """
        if not self.assembler:
            self.assembler = self.fl_ctx.get_engine().get_component(self.assembler_id)

        # Extract contributor name
        contributor_name = model.meta.get("client_name", "?")

        # Convert FLModel to Shareable to extract DXO
        shareable = FLModelUtils.to_shareable(model)

        # Check return code
        rc = shareable.get_return_code()
        if rc and rc != ReturnCode.OK:
            self.warning(f"Contributor {contributor_name} returned rc: {rc}. Disregarding contribution.")
            return

        # Get DXO from shareable
        try:
            dxo = from_shareable(shareable)
        except Exception:
            self.exception(f"Failed to convert shareable to DXO for {contributor_name}")
            return

        # Validate data kind
        expected_data_kind = self.assembler.get_expected_data_kind()
        if dxo.data_kind != expected_data_kind:
            self.error(f"Expected {expected_data_kind} but got {dxo.data_kind} from {contributor_name}")
            return

        # Check contribution round - get from FLModel, not shareable cookie
        current_round = self.fl_ctx.get_prop(AppConstants.CURRENT_ROUND)
        contribution_round = model.current_round
        if contribution_round is not None and contribution_round != current_round:
            self.warning(
                f"Discarding DXO from {contributor_name} at round {contribution_round}. "
                f"Current round is: {current_round}"
            )
            return

        # Add to assembler's collection
        collection = self.assembler.collection
        if contributor_name not in collection:
            collection[contributor_name] = self.assembler.get_model_params(dxo)
            self.info(f"Accepted contribution from {contributor_name}")
        else:
            self.info(f"Discarded: contributions already include client {contributor_name} at round {current_round}")

    def aggregate_model(self) -> FLModel:
        """Aggregate all accepted models using the Assembler.

        Returns:
            FLModel: Aggregated model
        """
        if not self.assembler:
            self.error("Assembler not initialized")
            return FLModel()

        current_round = self.fl_ctx.get_prop(AppConstants.CURRENT_ROUND)
        collection = self.assembler.collection
        site_num = len(collection)
        self.info(f"Aggregating {site_num} update(s) at round {current_round}")

        # Delegate to assembler
        dxo = self.assembler.assemble(data=collection, fl_ctx=self.fl_ctx)

        # Convert DXO to FLModel
        aggregated_model = FLModel(
            params=dxo.data,
            params_type=ParamsType.FULL,
            meta={"nr_aggregated": site_num, "current_round": current_round},
        )

        return aggregated_model

    def reset_stats(self) -> None:
        """Reset aggregation statistics for next round."""
        if self.assembler:
            self.assembler.reset()
