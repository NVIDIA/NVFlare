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

from nvflare.apis.dxo import DXO, DataKind, from_shareable
from nvflare.apis.fl_constant import ReservedKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.app_common.abstract.aggregator import Aggregator
from nvflare.app_common.aggregators.dxo_aggregator import DXOAggregator
from nvflare.app_common.app_constant import AppConstants


class InTimeAccumulateWeightedAggregator(Aggregator):
    def __init__(self, exclude_vars=None, aggregation_weights=None, expected_data_kind=DataKind.WEIGHT_DIFF):
        """Perform accumulated weighted aggregation
        It parses the shareable and aggregates the contained DXO(s).

        Args:
            exclude_vars ([type], optional): regex to match excluded vars during aggregation. Defaults to None.
                                Can be one string or a dict of keys with regex strings corresponding to each aggregated
                                DXO when processing a DXO of `DataKind.COLLECTION`.
            aggregation_weights ([type], optional): dictionary to map contributor name to its aggregation weights.
                                Defaults to None.
                                Can be one dict or a dict of dicts corresponding to each aggregated DXO
                                when processing DXO of `DataKind.COLLECTION`.
            expected_data_kind: DataKind or dict of keys and matching DataKind entries
                                when processing DXO of `DataKind.COLLECTION`.
                                Only the keys in the dict will be processed.
        """
        super().__init__()
        self.logger.debug(f"exclude vars: {exclude_vars}")
        self.logger.debug(f"aggregation weights control: {aggregation_weights}")
        self.logger.debug(f"expected data kind: {expected_data_kind}")

        self._single_dxo_key = ""

        # Check expected data kind
        if isinstance(expected_data_kind, dict):
            for k, v in expected_data_kind.items():
                if v not in [DataKind.WEIGHT_DIFF, DataKind.WEIGHTS]:
                    raise ValueError(f"expected_data_kind[{k}] = {v} not in WEIGHT_DIFF or WEIGHTS")
            self.expected_data_kind = expected_data_kind
        else:
            if expected_data_kind not in [DataKind.WEIGHT_DIFF, DataKind.WEIGHTS]:
                raise ValueError(f"expected_data_kind={expected_data_kind} not in WEIGHT_DIFF or WEIGHTS")
            self.expected_data_kind = {self._single_dxo_key: expected_data_kind}

        # Check exclude_vars
        exclude_vars_dict = dict()
        for k in self.expected_data_kind.keys():
            if isinstance(exclude_vars, dict):
                if k in exclude_vars:
                    if not isinstance(exclude_vars[k], str):
                        raise ValueError(
                            f"exclude_vars[{k}] = {exclude_vars[k]} not a string but {type(exclude_vars[k])}! "
                            f"Expected type regex string."
                        )
                    exclude_vars_dict[k] = exclude_vars[k]
            else:
                # assume same exclude vars for each entry of DXO collection.
                exclude_vars_dict[k] = exclude_vars
        if self._single_dxo_key in self.expected_data_kind:
            exclude_vars_dict[self._single_dxo_key] = exclude_vars
        self.exclude_vars = exclude_vars_dict

        # Check aggregation weights
        aggregation_weights = aggregation_weights or {}
        aggregation_weights_dict = dict()
        for k in self.expected_data_kind.keys():
            if k in aggregation_weights:
                aggregation_weights_dict[k] = aggregation_weights[k]
            else:
                # assume same aggregation weights for each entry of DXO collection.
                aggregation_weights_dict[k] = aggregation_weights
        if self._single_dxo_key in self.expected_data_kind:
            aggregation_weights_dict[self._single_dxo_key] = aggregation_weights
        self.aggregation_weights = aggregation_weights_dict

        # Set up DXO aggregators
        self.dxo_aggregators = dict()
        for k in self.expected_data_kind.keys():
            self.dxo_aggregators.update(
                {
                    k: DXOAggregator(
                        exclude_vars=self.exclude_vars[k],
                        aggregation_weights=self.aggregation_weights[k],
                        expected_data_kind=self.expected_data_kind[k],
                        name_postfix=k,
                    )
                }
            )

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

        if dxo.data_kind not in (DataKind.WEIGHT_DIFF, DataKind.WEIGHTS, DataKind.COLLECTION):
            self.log_error(
                fl_ctx,
                f"cannot handle data kind {dxo.data_kind}, "
                f"expecting DataKind.WEIGHT_DIFF, DataKind.WEIGHTS, or DataKind.COLLECTION.",
            )
            return False

        contributor_name = shareable.get_peer_prop(key=ReservedKey.IDENTITY_NAME, default="?")
        contribution_round = shareable.get_header(AppConstants.CONTRIBUTION_ROUND)

        rc = shareable.get_return_code()
        if rc and rc != ReturnCode.OK:
            self.log_warning(fl_ctx, f"Contributor {contributor_name} returned rc: {rc}. Disregarding contribution.")
            return False

        # Accept expected DXO(s) in shareable
        n_accepted = 0
        for key in self.expected_data_kind.keys():
            if key == self._single_dxo_key:  # expecting a single DXO
                sub_dxo = dxo
            else:  # expecting a collection of DXOs
                sub_dxo = dxo.data.get(key)
            if not isinstance(sub_dxo, DXO):
                self.log_warning(fl_ctx, f"Collection does not contain DXO for key {key} but {type(sub_dxo)}.")
                continue

            accepted = self.dxo_aggregators[key].accept(
                dxo=sub_dxo, contributor_name=contributor_name, contribution_round=contribution_round, fl_ctx=fl_ctx
            )
            if not accepted:
                return False
            else:
                n_accepted += 1

        if n_accepted > 0:
            return True
        else:
            self.log_warning(fl_ctx, f"Did not accept any DXOs from {contributor_name} in round {contribution_round}!")
            return False

    def aggregate(self, fl_ctx: FLContext) -> Shareable:
        """Called when workflow determines to generate shareable to send back to contributors

        Args:
            fl_ctx (FLContext): context provided by workflow

        Returns:
            Shareable: the weighted mean of accepted shareables from contributors
        """

        self.log_debug(fl_ctx, "Start aggregation")
        result_dxo_dict = dict()
        # Aggregate the expected DXO(s)
        for key in self.expected_data_kind.keys():
            aggregated_dxo = self.dxo_aggregators[key].aggregate(fl_ctx)
            if key == self._single_dxo_key:  # return single DXO with aggregation results
                return aggregated_dxo.to_shareable()
            self.log_info(fl_ctx, f"Aggregated contributions matching key '{key}'.")
            result_dxo_dict.update({key: aggregated_dxo})
        # return collection of DXOs with aggregation results
        collection_dxo = DXO(data_kind=DataKind.COLLECTION, data=result_dxo_dict)
        return collection_dxo.to_shareable()
