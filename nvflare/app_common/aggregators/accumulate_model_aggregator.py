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


class AccumulateWeightedAggregator(Aggregator):
    def __init__(self, exclude_vars=None, aggregation_weights=None, expected_data_kind="WEIGHT_DIFF"):
        """Fed average aggregator.

        This aggregator performs weighted arithmetic average among received shareables from clients.

        Args:
            exclude_vars (list, optional): if not specified (None), all layers are included;
                    if list of variable/layer names, only specified variables are excluded;
                    if string containing regular expression (e.g. "conv"), only matched variables are being excluded.
                    Defaults to None.
            aggregation_weights (dict, optional): a mapping from client names to weights. Defaults to None.
            expected_data_kind (str, optional): the data_kind this aggregator can process. Defaults to "WEIGHT_DIFF".

        Raises:
            ValueError: if data_kind is neither WEIGHT_DIFF nor WEIGHTS
        """
        super().__init__()
        self.exclude_vars = re.compile(exclude_vars) if exclude_vars else None
        self.aggregation_weights = aggregation_weights or {}
        if expected_data_kind not in [DataKind.WEIGHT_DIFF, DataKind.WEIGHTS]:
            raise ValueError(f"expected_data_kind={expected_data_kind} not in WEIGHT_DIFF or WEIGHTS")
        self.expected_data_kind = expected_data_kind
        self.logger.debug(f"aggregation weights control: {aggregation_weights}")

        self.accumulator = []

        self.warning_count = dict()
        self.warning_limit = 10

    def accept(self, shareable: Shareable, fl_ctx: FLContext) -> bool:
        try:
            dxo = from_shareable(shareable)
        except:
            self.log_exception(fl_ctx, "shareable data is not a valid DXO")
            return False

        if dxo.data_kind not in (DataKind.WEIGHT_DIFF, DataKind.WEIGHTS):
            self.log_error(fl_ctx, "cannot handle data kind {}".format(dxo.data_kind))
            return False

        if dxo.data_kind != self.expected_data_kind:
            self.log_error(fl_ctx, "expect {} but got {}".format(self.expected_data_kind, dxo.data_kind))
            return False

        processed_algorithm = dxo.get_meta_prop(MetaKey.PROCESSED_ALGORITHM)
        if processed_algorithm is not None:
            self.log_error(fl_ctx, f"unable to accept shareable processed by {processed_algorithm}")
            return False

        current_round = fl_ctx.get_prop(AppConstants.CURRENT_ROUND)
        self.log_debug(fl_ctx, f"current_round: {current_round}")
        client_name = shareable.get_peer_prop(key=ReservedKey.IDENTITY_NAME, default="?")
        contribution_round = shareable.get_header(AppConstants.CONTRIBUTION_ROUND)

        rc = shareable.get_return_code()
        if rc and rc != ReturnCode.OK:
            self.log_info(fl_ctx, f"Client {client_name} returned rc: {rc}. Disregarding contribution.")
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
                "Discarded the contribution from {} for round: {}. Current round is: {}".format(
                    client_name, contribution_round, current_round
                ),
            )
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
        self.log_info(fl_ctx, "aggregating {} updates at round {}".format(len(self.accumulator), current_round))

        # TODO: What if AppConstants.GLOBAL_MODEL is None?
        acc_vars = [set(acc.data.keys()) for acc in self.accumulator]
        acc_vars = set.union(*acc_vars) if acc_vars else acc_vars
        # update vars that are not in exclude pattern
        vars_to_aggregate = (
            [g_var for g_var in acc_vars if not self.exclude_vars.search(g_var)] if self.exclude_vars else acc_vars
        )

        clients_with_messages = []
        aggregated_model = {}
        for v_name in vars_to_aggregate:
            n_local_iters, np_vars = [], []
            for item in self.accumulator:
                client_name = item.client
                data = item.data
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
                aggregation_weight = self.aggregation_weights.get(client_name)
                if aggregation_weight is None:
                    if self.warning_count.get(client_name, 0) <= self.warning_limit:
                        self.log_warning(
                            fl_ctx,
                            f"Aggregation_weight missing for {client_name} and set to default value, 1.0"
                            f" This kind of message will show {self.warning_limit} times at most.",
                        )
                        if client_name in self.warning_count:
                            self.warning_count[client_name] = self.warning_count[client_name] + 1
                        else:
                            self.warning_count[client_name] = 0
                    aggregation_weight = 1.0

                weighted_value = data[v_name] * float_n_iter * aggregation_weight
                if client_name not in clients_with_messages:
                    if client_name in self.aggregation_weights.keys():
                        self.log_debug(fl_ctx, f"Client {client_name} use weight {aggregation_weight} for aggregation.")
                    else:
                        self.log_debug(
                            fl_ctx,
                            f"Client {client_name} not defined in the aggregation weight list. Use default value 1.0",
                        )
                    clients_with_messages.append(client_name)
                np_vars.append(weighted_value)
            if not n_local_iters:
                continue  # all acc didn't receive the variable from clients
            new_val = np.sum(np_vars, axis=0) / np.sum(n_local_iters)
            aggregated_model[v_name] = new_val

        self.accumulator.clear()

        self.log_debug(fl_ctx, f"Model after aggregation: {aggregated_model}")

        dxo = DXO(data_kind=self.expected_data_kind, data=aggregated_model)
        return dxo.to_shareable()
