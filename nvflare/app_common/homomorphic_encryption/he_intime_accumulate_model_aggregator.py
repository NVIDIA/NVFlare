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
import time

import numpy as np
import tenseal as ts

import nvflare.app_common.homomorphic_encryption.he_constant as he
from nvflare.apis.dxo import DXO, DataKind, MetaKey, from_shareable
from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import ReservedKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.app_common.abstract.aggregator import Aggregator
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.homomorphic_encryption.homomorphic_encrypt import (
    count_encrypted_layers,
    load_tenseal_context_from_workspace,
)


class HEInTimeAccumulateWeightedAggregator(Aggregator):
    def __init__(
        self,
        exclude_vars=None,
        aggregation_weights=None,
        tenseal_context_file="server_context.tenseal",
        weigh_by_local_iter=False,
        expected_data_kind="WEIGHT_DIFF",
        expected_algorithm=he.HE_ALGORITHM_CKKS,
    ):
        """In time aggregator for `Shareables` encrypted using homomorphic encryption (HE) with TenSEAL https://github.com/OpenMined/TenSEAL.

        Args:
            exclude_vars ([list], optional): variable names that should be excluded from aggregation (use regular expression). Defaults to None.
            aggregation_weights ([dict], optional): dictionary of client aggregation. Defaults to None.
            tenseal_context_file (str, optional): [description]. Defaults to "server_context.tenseal".
            weigh_by_local_iter (bool, optional): If true, multiply client weights on first in encryption space
                                 (default: `False` which is recommended for HE, first multiply happens in `HEModelEncryptor`)].
            expected_data_kind (str, optional): the data_kind this aggregator can process. Defaults to "WEIGHT_DIFF".
            expected_algorithm ([str], optional): the HE algorithm it can process. Defaults to he.HE_ALGORITHM_CKKS.

        Raises:
            ValueError: mismatched data_kind or HE algorithm
        """
        super().__init__()
        self.tenseal_context = None
        self.tenseal_context_file = tenseal_context_file
        if expected_data_kind not in [DataKind.WEIGHT_DIFF, DataKind.WEIGHTS]:
            raise ValueError(f"expected_data_kind={expected_data_kind} not in WEIGHT_DIFF or WEIGHTS")
        self.expected_data_kind = expected_data_kind
        self.expected_algorithm = expected_algorithm
        if self.expected_algorithm != he.HE_ALGORITHM_CKKS:
            raise ValueError(f"expected algorithm {self.expected_algorithm} not supported")
        self.exclude_vars = re.compile(exclude_vars) if exclude_vars else None
        self.aggregation_weights = aggregation_weights or {}
        self.reset_stats()
        self.weigh_by_local_iter = weigh_by_local_iter
        self.logger.info(f"client weights control: {self.aggregation_weights}")
        if not self.weigh_by_local_iter:
            if self.aggregation_weights:
                self.logger.warning("aggregation_weights will be ignored if weigh_by_local_iter=False")
            self.logger.info("Only divide by sum of local (weighted) iterations.")
        self.warning_count = dict()
        self.warning_limit = 0

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.START_RUN:
            self.tenseal_context = load_tenseal_context_from_workspace(self.tenseal_context_file, fl_ctx)
        elif event_type == EventType.END_RUN:
            self.tenseal_context = None

    def reset_stats(self):
        self.total = dict()
        self.counts = dict()
        self.contribution_count = 0
        self.history = list()
        self.merged_encrypted_layers = dict()  # thread-safety is handled by workflow

    def accept(self, shareable: Shareable, fl_ctx: FLContext) -> bool:
        """Accepts and adds the client updates to current average in HE encrypted space.

        Args:
            shareable: a shareable from client
            fl_ctx: FL Contenxt associated with this shareable

        Returns:
            bool to indicate if this shareable is accepted.
        """
        dxo = from_shareable(shareable)
        if dxo.data_kind != self.expected_data_kind:
            self.log_error(
                fl_ctx,
                f"expected {self.expected_data_kind} type DXO only but received {dxo.data_kind}, skipping this shareable.",
            )
            return False

        enc_algo = dxo.get_meta_prop(key=MetaKey.PROCESSED_ALGORITHM, default=None)
        if enc_algo != self.expected_algorithm:
            self.log_error(fl_ctx, "unsupported encryption algorithm {enc_algo}")
            return False

        current_round = fl_ctx.get_prop(AppConstants.CURRENT_ROUND)
        client_name = shareable.get_peer_prop(ReservedKey.IDENTITY_NAME, "?")
        contribution_round = shareable.get_header(AppConstants.CONTRIBUTION_ROUND)

        rc = shareable.get_return_code()
        if rc and rc != ReturnCode.OK:
            self.log_debug(fl_ctx, f"Client {client_name} returned rc: {rc}. Disregarding contribution.")
            return False

        self.log_debug(fl_ctx, f"current_round: {current_round}")

        if contribution_round != current_round:
            self.log_debug(
                fl_ctx,
                "Discarded the contribution from {client_name} for round: {contribution_round}. Current round is: {current_round}",
            )
            return False

        start_time = time.time()

        for item in self.history:
            if client_name == item["client_name"]:
                prev_round = item["round"]
                self.log_info(
                    fl_ctx,
                    f"discarding shareable from {client_name} at round: {contribution_round} as {prev_round} accepted already",
                )
                return False

        self.log_info(fl_ctx, f"Adding contribution from {client_name}.")

        n_iter = dxo.get_meta_prop(key=MetaKey.NUM_STEPS_CURRENT_ROUND)
        if n_iter is None:
            if self.warning_count.get(client_name, 0) <= self.warning_limit:
                self.log_warning(
                    fl_ctx,
                    f"NUM_STEPS_CURRENT_ROUND missing"
                    f" from {client_name} and set to default value, 1.0. "
                    f" This kind of message will show {self.warning_limit} times at most.",
                )
                if client_name in self.warning_count:
                    self.warning_count[client_name] = self.warning_count[client_name] + 1
                else:
                    self.warning_count[client_name] = 0
            n_iter = 1.0
        float_n_iter = np.float(n_iter)

        aggregation_weight = self.aggregation_weights.get(client_name)
        if aggregation_weight is None:
            aggregation_weight = 1.0

        aggr_data = dxo.data
        encrypted_layers = dxo.get_meta_prop(MetaKey.PROCESSED_KEYS)
        # TODO: test support of different encrypted layers for different clients!

        if encrypted_layers is None:
            self.log_error(fl_ctx, "encrypted_layers is None!")
            return False

        for k, v in aggr_data.items():
            if self.exclude_vars is not None and self.exclude_vars.search(k):
                continue
            if encrypted_layers[k]:
                if self.weigh_by_local_iter:
                    weighted_value = ts.ckks_vector_from(self.tenseal_context, v) * (aggregation_weight * float_n_iter)
                else:
                    weighted_value = ts.ckks_vector_from(self.tenseal_context, v)
                self.merged_encrypted_layers[k] = True  # any client can set this true
            else:
                if self.weigh_by_local_iter:
                    weighted_value = v * (aggregation_weight * float_n_iter)
                else:
                    weighted_value = v
                if k not in self.merged_encrypted_layers:
                    self.merged_encrypted_layers[k] = False  # only set False if no other client set it to True
            current_total = self.total.get(k, None)
            if current_total is None:
                self.total[k] = weighted_value
                self.counts[k] = n_iter
            else:
                self.total[k] = current_total + weighted_value
                self.counts[k] = self.counts[k] + n_iter

        self.contribution_count += 1

        end_time = time.time()
        n_encrypted, n_total = count_encrypted_layers(self.merged_encrypted_layers)
        self.log_info(fl_ctx, f"{n_encrypted} of {n_total} layers encrypted")
        self.log_info(fl_ctx, f"Round {current_round} adding {client_name} time is {end_time - start_time} seconds")

        self.history.append(
            {
                "client_name": client_name,
                "round": contribution_round,
                "aggregation_weight": aggregation_weight,
                "n_iter": n_iter,
            }
        )
        return True

    def aggregate(self, fl_ctx: FLContext) -> Shareable:
        start_time = time.time()
        current_round = fl_ctx.get_prop(AppConstants.CURRENT_ROUND)

        aggregated_dict = dict()
        for k, v in self.total.items():
            aggregated_dict[k] = v * (1.0 / self.counts[k])
        end_time = time.time()
        self.log_info(
            fl_ctx,
            f"Aggregated {self.contribution_count} contributions for round {current_round} time is {end_time - start_time} seconds",
        )

        dxo = DXO(data_kind=self.expected_data_kind, data=aggregated_dict)
        dxo.set_meta_prop(MetaKey.PROCESSED_KEYS, self.merged_encrypted_layers)
        dxo.set_meta_prop(MetaKey.PROCESSED_ALGORITHM, self.expected_algorithm)
        n_encrypted, n_total = count_encrypted_layers(self.merged_encrypted_layers)
        self.log_info(fl_ctx, f"{n_encrypted} of {n_total} layers encrypted")

        fl_ctx.set_prop(AppConstants.DXO, dxo, private=True, sticky=False)

        self.reset_stats()  # only reset dictionary after adding merged_encrypted_layers to dictionary
        return dxo.to_shareable()
