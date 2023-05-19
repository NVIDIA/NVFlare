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

from typing import Optional

from nvflare.apis.dxo import DXO, from_shareable
from nvflare.apis.fl_constant import ReservedKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.app_common.abstract.aggregator import Aggregator
from nvflare.app_common.aggregators.assembler import Assembler
from nvflare.app_common.app_constant import AppConstants


class CollectAndAssembleAggregator(Aggregator):
    """Perform collection and flexible assemble aggregation

    This is used for methods needing a special assemble mechanism on the client submissions.
    It first collects all submissions from clients, then delegates the assembling functionality to assembler,
    which is specific to a particular algorithm.
    Note that the aggregation in this case is not in-time, since the assembling function may not be arithmetic mean.
    """

    def __init__(self, assembler_id: str):
        super().__init__()
        self.assembler_id = assembler_id
        self.assembler: Optional[Assembler] = None

    def accept(self, shareable: Shareable, fl_ctx: FLContext) -> bool:
        if not self.assembler:
            self.assembler = fl_ctx.get_engine().get_component(self.assembler_id)
        contributor_name = shareable.get_peer_prop(key=ReservedKey.IDENTITY_NAME, default="?")
        dxo = self._get_contribution(shareable, fl_ctx)
        if dxo is None or dxo.data is None:
            self.log_error(fl_ctx, "no data to aggregate")
            return False
        data = dxo.data
        current_round = fl_ctx.get_prop(AppConstants.CURRENT_ROUND)
        return self._accept_contribution(contributor_name, current_round, data, fl_ctx)

    def _accept_contribution(self, contributor: str, current_round: int, data: dict, fl_ctx: FLContext) -> bool:
        collection = self.assembler.collection
        if contributor not in collection:
            collection[contributor] = self.assembler.get_model_params(data)
            accepted = True
        else:
            self.log_info(
                fl_ctx,
                f"Discarded: Current round: {current_round} " + f"contributions already include client: {contributor}",
            )
            accepted = False
        return accepted

    def _get_contribution(self, shareable: Shareable, fl_ctx: FLContext) -> Optional[DXO]:
        contributor_name = shareable.get_peer_prop(key=ReservedKey.IDENTITY_NAME, default="?")
        try:
            dxo = from_shareable(shareable)
        except Exception:
            self.log_exception(fl_ctx, "shareable data is not a valid DXO")
            return None

        rc = shareable.get_return_code()
        if rc and rc != ReturnCode.OK:
            self.log_warning(
                fl_ctx,
                f"Contributor {contributor_name} returned rc: {rc}. Disregarding contribution.",
            )
            return None
        expected_data_kind = self.assembler.get_expected_data_kind()
        if dxo.data_kind != expected_data_kind:
            self.log_error(
                fl_ctx,
                "expected {} but got {}".format(expected_data_kind, dxo.data_kind),
            )
            return None

        contribution_round = shareable.get_cookie(AppConstants.CONTRIBUTION_ROUND)
        current_round = fl_ctx.get_prop(AppConstants.CURRENT_ROUND)
        if contribution_round != current_round:
            self.log_warning(
                fl_ctx,
                f"discarding DXO from {contributor_name} at round: "
                f"{contribution_round}. Current round is: {current_round}",
            )
            return None

        return dxo

    def aggregate(self, fl_ctx: FLContext) -> Shareable:
        self.log_debug(fl_ctx, "Start aggregation")
        current_round = fl_ctx.get_prop(AppConstants.CURRENT_ROUND)
        collection = self.assembler.collection
        site_num = len(collection)
        self.log_info(fl_ctx, f"aggregating {site_num} update(s) at round {current_round}")

        model = self.assembler.assemble(data=collection, fl_ctx=fl_ctx)
        # Reset assembler for next round
        self.assembler.reset()
        self.log_debug(fl_ctx, "End aggregation")
        dxo = DXO(data_kind=self.assembler.get_expected_data_kind(), data=model)
        return dxo.to_shareable()
