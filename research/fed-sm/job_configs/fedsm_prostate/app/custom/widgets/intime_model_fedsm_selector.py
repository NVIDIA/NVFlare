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

import numpy as np

from nvflare.apis.dxo import DataKind, MetaKey, from_shareable
from nvflare.apis.event_type import EventType
from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_constant import FLContextKey, ReservedKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.app_event_type import AppEventType


class IntimeModelFedSMSelector(FLComponent):
    def __init__(self, weigh_by_local_iter=False, aggregation_weights=None):
        """Handler to determine if the model is globally best.
        Note that only model_global and model_select participate in this selection process, while personalized models do not
        Args:
            weigh_by_local_iter (bool, optional): whether the metrics should be weighted by trainer's iteration number. Defaults to False.
            aggregation_weights (dict, optional): a mapping of client name to float for aggregation. Defaults to None.
        """
        super().__init__()

        self.val_metric_global = self.best_val_metric_global = -np.inf
        self.val_metric_select = self.best_val_metric_select = -np.inf
        self.weigh_by_local_iter = weigh_by_local_iter
        self.validation_metric_name = MetaKey.INITIAL_METRICS
        self.aggregation_weights = aggregation_weights or {}

        self.logger.debug(f"model selection weights control: {aggregation_weights}")
        self._reset_stats()

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        """Perform the handler process based on the event_type.

        Args:
            event_type (str): event type delivered from workflow
            fl_ctx (FLContext): FL context, including peer context and other information
        """
        if event_type == EventType.START_RUN:
            self._startup(fl_ctx)
        elif event_type == EventType.BEFORE_PROCESS_SUBMISSION:
            self._before_accept(fl_ctx)
        elif event_type == AppEventType.BEFORE_AGGREGATION:
            self._before_aggregate(fl_ctx)

    def _startup(self, fl_ctx):
        self._reset_stats()

    def _reset_stats(self):
        self.validation_mertic_global_weighted_sum = 0
        self.validation_metric_global_sum_of_weights = 0
        self.validation_mertic_select_weighted_sum = 0
        self.validation_metric_select_sum_of_weights = 0

    def _before_accept(self, fl_ctx: FLContext):
        peer_ctx = fl_ctx.get_peer_context()
        shareable: Shareable = peer_ctx.get_prop(FLContextKey.SHAREABLE)
        try:
            dxo = from_shareable(shareable)
        except:
            self.log_exception(fl_ctx, "shareable data is not a valid DXO")
            return False

        # check data_kind
        if dxo.data_kind not in (
            DataKind.WEIGHT_DIFF,
            DataKind.WEIGHTS,
            DataKind.COLLECTION,
        ):
            self.log_debug(fl_ctx, "I cannot handle {}".format(dxo.data_kind))
            return False

        if dxo.data is None:
            self.log_debug(fl_ctx, "no data to filter")
            return False

        # DXO for FedSM is in "collection" format, containing three dxo objects (global_weights, person_weights, select_weights)
        # together with the meta information
        contribution_round = shareable.get_header(AppConstants.CONTRIBUTION_ROUND)
        client_name = shareable.get_peer_prop(ReservedKey.IDENTITY_NAME, default="?")

        current_round = fl_ctx.get_prop(AppConstants.CURRENT_ROUND)

        if current_round == 0:
            self.log_debug(fl_ctx, "skipping round 0")
            return False  # There is no aggregated model at round 0

        if contribution_round != current_round:
            self.log_debug(
                fl_ctx,
                f"discarding shareable from {client_name} for round: {contribution_round}. Current round is: {current_round}",
            )
            return False

        # validation metric is a list of two numbers, corresponding to the two models,
        # note that personalized model store locally, will not participate global selection and saving
        # [global_metric, select_metric]
        validation_metric = dxo.get_meta_prop(self.validation_metric_name)
        if validation_metric is None:
            self.log_debug(fl_ctx, f"validation metric not existing in {client_name}")
            return False
        else:
            self.log_info(
                fl_ctx,
                f"validation metric {validation_metric} from client {client_name}",
            )

        if self.weigh_by_local_iter:
            n_iter = dxo.get_meta_prop(MetaKey.NUM_STEPS_CURRENT_ROUND, 1.0)
        else:
            n_iter = 1.0

        aggregation_weights = self.aggregation_weights.get(client_name, 1.0)
        self.log_debug(fl_ctx, f"aggregation weight: {aggregation_weights}")

        self.validation_mertic_global_weighted_sum += validation_metric[0] * n_iter * aggregation_weights
        self.validation_metric_global_sum_of_weights += n_iter
        self.validation_mertic_select_weighted_sum += validation_metric[1] * n_iter * aggregation_weights
        self.validation_metric_select_sum_of_weights += n_iter
        return True

    def _before_aggregate(self, fl_ctx):
        if self.validation_metric_global_sum_of_weights == 0:
            self.log_debug(fl_ctx, "nothing accumulated for model_global")
            return False
        if self.validation_mertic_select_weighted_sum == 0:
            self.log_debug(fl_ctx, "nothing accumulated for model_selector")
            return False

        self.val_metric_global = (
            self.validation_mertic_global_weighted_sum / self.validation_metric_global_sum_of_weights
        )
        self.val_metric_select = (
            self.validation_mertic_select_weighted_sum / self.validation_metric_select_sum_of_weights
        )

        self.logger.debug(f"weighted validation metric {self.val_metric_global}")
        self.logger.debug(f"weighted validation metric {self.val_metric_select}")

        if self.val_metric_global > self.best_val_metric_global:
            self.best_val_metric_global = self.val_metric_global
            current_round = fl_ctx.get_prop(AppConstants.CURRENT_ROUND)
            self.log_info(
                fl_ctx,
                f"new best validation metric for global model at round {current_round}: {self.best_val_metric_global}",
            )
            # Fire event to notify that the current global model is a new best
            self.fire_event("fedsm_best_global_model_available", fl_ctx)

        if self.val_metric_select > self.best_val_metric_select:
            self.best_val_metric_select = self.val_metric_select
            current_round = fl_ctx.get_prop(AppConstants.CURRENT_ROUND)
            self.log_info(
                fl_ctx,
                f"new best validation metric for selector model at round {current_round}: {self.best_val_metric_select}",
            )
            # Fire event to notify that the current global model is a new best
            self.fire_event("fedsm_best_select_model_available", fl_ctx)

        self._reset_stats()
        return True
