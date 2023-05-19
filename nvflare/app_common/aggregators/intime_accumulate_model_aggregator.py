# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

from typing import Any, Dict, Union

from nvflare.apis.dxo import DXO, DataKind, from_shareable
from nvflare.apis.fl_constant import ReservedKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.app_common.abstract.aggregator import Aggregator
from nvflare.app_common.aggregators.dxo_aggregator import DXOAggregator
from nvflare.app_common.app_constant import AppConstants


def _is_nested_aggregation_weights(aggregation_weights):
    if not aggregation_weights:
        return False
    if not isinstance(aggregation_weights, dict):
        return False
    first_value = next(iter(aggregation_weights.items()))[1]
    if not isinstance(first_value, dict):
        return False
    return True


def _get_missing_keys(ref_dict: dict, dict_to_check: dict):
    result = []
    for k in ref_dict:
        if k not in dict_to_check:
            result.append(k)
    return result


class InTimeAccumulateWeightedAggregator(Aggregator):
    def __init__(
        self,
        exclude_vars: Union[str, Dict[str, str], None] = None,
        aggregation_weights: Union[Dict[str, Any], Dict[str, Dict[str, Any]], None] = None,
        expected_data_kind: Union[DataKind, Dict[str, DataKind]] = DataKind.WEIGHT_DIFF,
        weigh_by_local_iter: bool = True,
    ):
        """Perform accumulated weighted aggregation.

        This is often used as the default aggregation method and can be used for FedAvg. It parses the shareable and
        aggregates the contained DXO(s).

        Args:
            exclude_vars (Union[str, Dict[str, str]], optional):
                Regular expression string to match excluded vars during aggregation. Defaults to None.
                Can be one string or a dict of {dxo_name: regex strings} corresponding to each aggregated DXO
                when processing a DXO of `DataKind.COLLECTION`.
            aggregation_weights (Union[Dict[str, Any], Dict[str, Dict[str, Any]]], optional):
                Aggregation weight for each contributor. Defaults to None.
                Can be one dict of {contrib_name: aggr_weight} or a dict of dicts corresponding to each aggregated DXO
                when processing a DXO of `DataKind.COLLECTION`.
            expected_data_kind (Union[DataKind, Dict[str, DataKind]]):
                DataKind for DXO. Defaults to DataKind.WEIGHT_DIFF
                Can be one DataKind or a dict of {dxo_name: DataKind} corresponding to each aggregated DXO
                when processing a DXO of `DataKind.COLLECTION`. Only the keys in this dict will be processed.
            weigh_by_local_iter (bool, optional): Whether to weight the contributions by the number of iterations
                performed in local training in the current round. Defaults to `True`.
                Setting it to `False` can be useful in applications such as homomorphic encryption to reduce
                the number of computations on encrypted ciphertext.
                The aggregated sum will still be divided by the provided weights and `aggregation_weights` for the
                resulting weighted sum to be valid.
        """
        super().__init__()
        self.logger.debug(f"exclude vars: {exclude_vars}")
        self.logger.debug(f"aggregation weights control: {aggregation_weights}")
        self.logger.debug(f"expected data kind: {expected_data_kind}")

        self._single_dxo_key = ""
        self._weigh_by_local_iter = weigh_by_local_iter

        # Check expected data kind
        if isinstance(expected_data_kind, dict):
            for k, v in expected_data_kind.items():
                if v not in [DataKind.WEIGHT_DIFF, DataKind.WEIGHTS, DataKind.METRICS]:
                    raise ValueError(
                        f"expected_data_kind[{k}] = {v} is not {DataKind.WEIGHT_DIFF} or {DataKind.WEIGHTS} or {DataKind.METRICS}"
                    )
            self.expected_data_kind = expected_data_kind
        else:
            if expected_data_kind not in [DataKind.WEIGHT_DIFF, DataKind.WEIGHTS, DataKind.METRICS]:
                raise ValueError(
                    f"expected_data_kind = {expected_data_kind} is not {DataKind.WEIGHT_DIFF} or {DataKind.WEIGHTS} or {DataKind.METRICS}"
                )
            self.expected_data_kind = {self._single_dxo_key: expected_data_kind}

        # Check exclude_vars
        if exclude_vars:
            if not isinstance(exclude_vars, dict) and not isinstance(exclude_vars, str):
                raise ValueError(
                    f"exclude_vars = {exclude_vars} should be a regex string but got {type(exclude_vars)}."
                )
            if isinstance(exclude_vars, dict):
                missing_keys = _get_missing_keys(expected_data_kind, exclude_vars)
                if len(missing_keys) != 0:
                    raise ValueError(
                        "A dict exclude_vars should specify exclude_vars for every key in expected_data_kind. "
                        f"But missed these keys: {missing_keys}"
                    )

        exclude_vars_dict = dict()
        for k in self.expected_data_kind.keys():
            if isinstance(exclude_vars, dict):
                if k in exclude_vars:
                    if not isinstance(exclude_vars[k], str):
                        raise ValueError(
                            f"exclude_vars[{k}] = {exclude_vars[k]} should be a regex string but got {type(exclude_vars[k])}."
                        )
                    exclude_vars_dict[k] = exclude_vars[k]
            else:
                # assume same exclude vars for each entry of DXO collection.
                exclude_vars_dict[k] = exclude_vars
        if self._single_dxo_key in self.expected_data_kind:
            exclude_vars_dict[self._single_dxo_key] = exclude_vars
        self.exclude_vars = exclude_vars_dict

        # Check aggregation weights
        if _is_nested_aggregation_weights(aggregation_weights):
            missing_keys = _get_missing_keys(expected_data_kind, aggregation_weights)
            if len(missing_keys) != 0:
                raise ValueError(
                    "A dict of dict aggregation_weights should specify aggregation_weights "
                    f"for every key in expected_data_kind. But missed these keys: {missing_keys}"
                )

        aggregation_weights = aggregation_weights or {}
        aggregation_weights_dict = dict()
        for k in self.expected_data_kind.keys():
            if k in aggregation_weights:
                aggregation_weights_dict[k] = aggregation_weights[k]
            else:
                # assume same aggregation weights for each entry of DXO collection.
                aggregation_weights_dict[k] = aggregation_weights
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
                        weigh_by_local_iter=self._weigh_by_local_iter,
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
        except Exception:
            self.log_exception(fl_ctx, "shareable data is not a valid DXO")
            return False

        if dxo.data_kind not in (DataKind.WEIGHT_DIFF, DataKind.WEIGHTS, DataKind.METRICS, DataKind.COLLECTION):
            self.log_error(
                fl_ctx,
                f"cannot handle data kind {dxo.data_kind}, "
                f"expecting DataKind.WEIGHT_DIFF, DataKind.WEIGHTS, or DataKind.COLLECTION.",
            )
            return False

        contributor_name = shareable.get_peer_prop(key=ReservedKey.IDENTITY_NAME, default="?")
        contribution_round = shareable.get_cookie(AppConstants.CONTRIBUTION_ROUND)

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
