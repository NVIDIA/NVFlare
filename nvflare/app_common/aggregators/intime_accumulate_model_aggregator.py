# Copyright (c) 2021, NVIDIA CORPORATION.
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

from nvflare.apis.dxo import DXO, DataKind, MetaKey, from_shareable
from nvflare.apis.fl_constant import ReservedKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.app_common.abstract.aggregator import Aggregator
from nvflare.app_common.app_constant import AppConstants


class InTimeAccumulateWeightedAggregator(Aggregator):
    def __init__(self, exclude_vars=None, aggregation_weights=None, expected_data_kind=DataKind.WEIGHT_DIFF):
        """Perform accumulated weighted aggregation
        It computes
        weighted_sum = sum(shareable*n_iteration*aggregation_weights) and
        sum_of_weights = sum(n_iteration)

        in accept function
        The aggregate function returns
        weighted_sum / sum_of_weights

        Args:
            exclude_vars ([type], optional): regex to match excluded vars during aggregation. Defaults to None.
            aggregation_weights ([type], optional): dictionary to map client name to its aggregation weights. Defaults to None.
        """
        super().__init__()
        self.exclude_vars = re.compile(exclude_vars) if exclude_vars else None
        self.aggregation_weights = aggregation_weights or {}
        if expected_data_kind not in [DataKind.WEIGHT_DIFF, DataKind.WEIGHTS]:
            raise ValueError(f"expected_data_kind={expected_data_kind} not in WEIGHT_DIFF or WEIGHTS")
        self.expected_data_kind = expected_data_kind
        self.logger.debug(f"aggregation weights control: {aggregation_weights}")
        self.reset_stats()
        self.warning_count = {}
        self.warning_limit = 10
        self.total = dict()
        self.counts = dict()
        self.history = list()

    def reset_stats(self):
        self.total = {}
        self.counts = {}
        self.history = []

    def accept(self, shareable: Shareable, fl_ctx: FLContext) -> bool:
        """Store shareable and update aggregator's internal state

        Args:
            shareable: information from client
            fl_ctx: context provided by workflow

        Returns:
            The first boolean indicates if this shareable is accepted.
            The second bollean indicates if aggregate can be called.
        """
        try:
            dxo = from_shareable(shareable)
        except:
            self.log_exception(fl_ctx, "shareable data is not a valid DXO")
            return False
        assert isinstance(dxo, DXO)

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
        if contribution_round != current_round:
            self.log_info(
                fl_ctx,
                f"discarding shareable from {client_name} at round: {contribution_round}. Current round is: {current_round}",
            )
            return False

        for item in self.history:
            if client_name == item["client_name"]:
                prev_round = item["round"]
                self.log_info(
                    fl_ctx,
                    f"discarding shareable from {client_name} at round: {contribution_round} as {prev_round} accepted already",
                )
                return False

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
        float_n_iter = float(n_iter)
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

        for k, v in data.items():
            if self.exclude_vars is not None and self.exclude_vars.search(k):
                continue
            weighted_value = v * aggregation_weight * float_n_iter
            current_total = self.total.get(k, None)
            if current_total is None:
                self.total[k] = weighted_value
                self.counts[k] = n_iter
            else:
                self.total[k] = current_total + weighted_value
                self.counts[k] = self.counts[k] + n_iter
        self.history.append(
            {
                "client_name": client_name,
                "round": contribution_round,
                "aggregation_weight": aggregation_weight,
                "n_iter": n_iter,
            }
        )
        self.log_debug(fl_ctx, "End accept")
        return True

    def aggregate(self, fl_ctx: FLContext) -> Shareable:
        """Called when workflow determines to generate shareable to send back to clients

        Args:
            fl_ctx (FLContext): context provided by workflow

        Returns:
            Shareable: the weighted mean of accepted shareables from clients
        """

        self.log_debug(fl_ctx, "Start aggregation")
        current_round = fl_ctx.get_prop(AppConstants.CURRENT_ROUND)
        self.log_info(fl_ctx, f"aggregating {len(self.history)} update(s) at round {current_round}")
        self.log_debug(fl_ctx, f"complete history {self.history}")
        aggregated_dict = {k: v / self.counts[k] for k, v in self.total.items()}
        self.reset_stats()
        self.log_debug(fl_ctx, "End aggregation")

        dxo = DXO(data_kind=self.expected_data_kind, data=aggregated_dict)
        return dxo.to_shareable()
