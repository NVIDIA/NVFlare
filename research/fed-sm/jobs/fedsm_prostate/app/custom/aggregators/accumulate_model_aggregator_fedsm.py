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

import re

import numpy as np

from nvflare.apis.dxo import DXO, DataKind, MetaKey, from_shareable
from nvflare.apis.fl_constant import ReservedKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.app_common.abstract.aggregator import Aggregator
from nvflare.app_common.app_constant import AppConstants


class _AccuItem(object):
    def __init__(self, client, data, steps):
        self.client = client
        self.data = data
        self.steps = steps


class AccumulateWeightedAggregatorFedSM(Aggregator):
    def __init__(
        self,
        soft_pull_lambda=0.7,
        exclude_vars_global=None,
        exclude_vars_person=None,
        exclude_vars_select=None,
        aggregation_weights=None,
    ):
        """FedSM aggregator.

        This aggregator performs two types of aggregation among received shareables from clients:
        weighted arithmetic average for model_global and model_select
        SoftPull aggregation for model_person

        Args:
            soft_pull_lambda: the control weight for generating person models via soft pull mechanism. Defaults to 0.3.
            exclude_vars (list, optional) for three models (_global/person/select): if not specified (None), all layers are included;
                    if list of variable/layer names, only specified variables are excluded;
                    if string containing regular expression (e.g. "conv"), only matched variables are being excluded.
                    Defaults to None.
            aggregation_weights (dict, optional): a mapping from client names to weights. Defaults to None.

        Raises:
            ValueError: if data_kind is neither WEIGHT_DIFF nor WEIGHTS
        """
        super().__init__()
        self.soft_pull_lambda = soft_pull_lambda
        self.exclude_vars_global = re.compile(exclude_vars_global) if exclude_vars_global else None
        self.exclude_vars_person = re.compile(exclude_vars_person) if exclude_vars_person else None
        self.exclude_vars_select = re.compile(exclude_vars_select) if exclude_vars_select else None
        self.aggregation_weights = aggregation_weights or {}
        self.logger.debug(f"aggregation weights control: {aggregation_weights}")
        # FedSM aggregator expects "COLLECTION" DXO containing all three models.
        self.expected_data_kind = DataKind.COLLECTION
        self.accumulator = []
        self.warning_count = dict()
        self.warning_limit = 10

    def accept(self, shareable: Shareable, fl_ctx: FLContext) -> bool:
        try:
            dxo = from_shareable(shareable)
        except:
            self.log_exception(fl_ctx, "shareable data is not a valid DXO")
            return False

        if dxo.data_kind != self.expected_data_kind:
            self.log_error(
                fl_ctx,
                f"FedSM aggregator expect {self.expected_data_kind} but got {dxo.data_kind}",
            )
            return False

        processed_algorithm = dxo.get_meta_prop(MetaKey.PROCESSED_ALGORITHM)
        if processed_algorithm is not None:
            self.log_error(fl_ctx, f"unable to accept shareable processed by {processed_algorithm}")
            return False

        current_round = fl_ctx.get_prop(AppConstants.CURRENT_ROUND)
        self.log_debug(fl_ctx, f"current_round: {current_round}")
        client_name = shareable.get_peer_prop(key=ReservedKey.IDENTITY_NAME, default="?")
        contribution_round = shareable.get_cookie(AppConstants.CONTRIBUTION_ROUND)

        rc = shareable.get_return_code()
        if rc and rc != ReturnCode.OK:
            self.log_info(
                fl_ctx,
                f"Client {client_name} returned rc: {rc}. Disregarding contribution.",
            )
            return False

        data = dxo.data
        if data is None:
            self.log_error(fl_ctx, "no data to aggregate")
            return False
        n_iter = dxo.get_meta_prop(MetaKey.NUM_STEPS_CURRENT_ROUND)

        if contribution_round == current_round:
            if not self._client_in_accumulator(client_name):
                self.accumulator.append(_AccuItem(client_name, data, n_iter))
                accepted = True
            else:
                self.log_info(
                    fl_ctx,
                    f"Discarded: Current round: {current_round} contributions already include client: {client_name}",
                )
                accepted = False
        else:
            self.log_info(
                fl_ctx,
                f"Discarded the contribution from {client_name} for round: {contribution_round}. Current round is: {current_round}",
            ),
            accepted = False
        return accepted

    def _client_in_accumulator(self, client_name):
        return any(client_name == item.client for item in self.accumulator)

    def aggregate(self, fl_ctx: FLContext) -> Shareable:
        """Aggregate model variables.

        This function is not thread-safe.


        Args:
            fl_ctx (FLContext): System-wide FL Context

        Returns:
            Shareable: Return True to indicates the current model is the best model so far.
        """
        current_round = fl_ctx.get_prop(AppConstants.CURRENT_ROUND)
        self.log_info(
            fl_ctx,
            f"aggregating {len(self.accumulator)} updates at round {current_round}",
        )

        aggregated_model_dict = {}

        # regular weighted average aggregation for global and selector models and parameters
        model_ids = [
            "global_weights",
            "select_weights",
            "select_exp_avg",
            "select_exp_avg_sq",
        ]
        select_models = {}
        for model_id in model_ids:
            acc_vars = [set(acc.data[model_id].data.keys()) for acc in self.accumulator]
            acc_vars = set.union(*acc_vars) if acc_vars else acc_vars

            # update vars that are not in exclude pattern
            if model_id == "global_weights":
                exclude_vars = self.exclude_vars_global
            elif model_id == "select_weights":
                exclude_vars = self.exclude_vars_select
            vars_to_aggregate = (
                [g_var for g_var in acc_vars if not exclude_vars.search(g_var)] if exclude_vars else acc_vars
            )

            aggregated_model = {}
            for v_name in vars_to_aggregate:
                n_local_iters, np_vars = [], []
                for item in self.accumulator:
                    client_name = item.client
                    data = item.data[model_id].data

                    n_iter = item.steps
                    if n_iter is None:
                        if self.warning_count.get(client_name, 0) <= self.warning_limit:
                            self.log_warning(
                                fl_ctx,
                                f"NUM_STEPS_CURRENT_ROUND missing in meta of shareable"
                                f" from {client_name} and set to default value, 1.0. "
                                f" This kind of message will show {self.warning_limit} times at most.",
                            )
                            if client_name in self.warning_count:
                                self.warning_count[client_name] = self.warning_count[client_name] + 1
                            else:
                                self.warning_count[client_name] = 0
                        n_iter = 1.0
                    if v_name not in data.keys():
                        continue  # this acc doesn't have the variable from client
                    float_n_iter = float(n_iter)
                    n_local_iters.append(float_n_iter)

                    weighted_value = data[v_name] * float_n_iter
                    np_vars.append(weighted_value)
                if not n_local_iters:
                    continue  # all acc didn't receive the variable from clients
                new_val = np.sum(np_vars, axis=0) / np.sum(n_local_iters)
                aggregated_model[v_name] = new_val
            # make aggregated weights a DXO and add to dict
            if model_id == "global_weights":
                dxo_weights = DXO(data_kind=DataKind.WEIGHT_DIFF, data=aggregated_model)
                aggregated_model_dict[model_id] = dxo_weights
            elif model_id == "select_weights":
                dxo_weights = DXO(data_kind=DataKind.WEIGHT_DIFF, data=aggregated_model)
                select_models[model_id] = dxo_weights
            else:
                dxo_weights = DXO(data_kind=DataKind.WEIGHTS, data=aggregated_model)
                select_models[model_id] = dxo_weights

        aggregated_model_dict["select_weights"] = select_models

        # SoftPull for personalized models
        # initialize the personalized model set dict
        aggregated_model = {}
        model_id = "person_weights"
        for item in self.accumulator:
            client_name = item.client
            aggregated_model[client_name] = {}
        # exclude specified vars
        acc_vars = [set(acc.data[model_id].data.keys()) for acc in self.accumulator]
        acc_vars = set.union(*acc_vars) if acc_vars else acc_vars
        exclude_vars = self.exclude_vars_select
        vars_to_aggregate = (
            [g_var for g_var in acc_vars if not exclude_vars.search(g_var)] if exclude_vars else acc_vars
        )

        # SoftPull aggregation, weighted with soft pull lambda
        for v_name in vars_to_aggregate:
            # np_vars for personalized model set is a dict for each client
            # initialize
            np_vars = {}
            for item in self.accumulator:
                client_name = item.client
                np_vars[client_name] = []

            # go over the accumulator for aggregation
            for item in self.accumulator:
                client_name = item.client
                data = item.data[model_id].data
                if v_name not in data.keys():
                    continue  # this acc doesn't have the variable from client

                for aggr_item in self.accumulator:
                    aggr_client_name = aggr_item.client
                    if aggr_client_name == client_name:
                        weighted_value = data[v_name] * self.soft_pull_lambda
                    else:
                        weighted_value = data[v_name] * (1 - self.soft_pull_lambda) / (len(self.accumulator) - 1)
                    np_vars[aggr_client_name].append(weighted_value)

            for item in self.accumulator:
                client_name = item.client
                new_val = np.sum(np_vars[client_name], axis=0)
                aggregated_model[client_name][v_name] = new_val
        # make aggregated weights a DXO and add to dict
        person_models = {}
        for client_name in aggregated_model.keys():
            dxo_weights = DXO(data_kind=DataKind.WEIGHTS, data=aggregated_model[client_name])
            person_models[client_name] = dxo_weights
        aggregated_model_dict["person_weights"] = person_models

        self.accumulator.clear()
        self.log_debug(fl_ctx, f"Model after aggregation: {aggregated_model_dict}")
        dxo = DXO(data_kind=self.expected_data_kind, data=aggregated_model_dict)
        return dxo.to_shareable()
