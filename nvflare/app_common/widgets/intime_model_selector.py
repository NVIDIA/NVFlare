# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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
from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.app_event_type import AppEventType
from nvflare.widgets.widget import Widget


class IntimeModelSelector(Widget):
    def __init__(
        self, weigh_by_local_iter=False, aggregation_weights=None, validation_metric_name=MetaKey.INITIAL_METRICS
    ):
        """Handler to determine if the model is globally best.

        Args:
            weigh_by_local_iter (bool, optional): whether the metrics should be weighted by trainer's iteration number.
            aggregation_weights (dict, optional): a mapping of client name to float for aggregation. Defaults to None.
            validation_metric_name (str, optional): key used to save initial validation metric in the DXO meta properties (defaults to MetaKey.INITIAL_METRICS).
        """
        super().__init__()

        self.val_metric = self.best_val_metric = -np.inf
        self.weigh_by_local_iter = weigh_by_local_iter
        self.validation_metric_name = validation_metric_name
        self.aggregation_weights = aggregation_weights or {}

        self.logger.info(f"model selection weights control: {aggregation_weights}")
        self._reset_stats()

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.START_RUN:
            self._startup()
        elif event_type == AppEventType.ROUND_STARTED:
            self._reset_stats()
        elif event_type == AppEventType.BEFORE_CONTRIBUTION_ACCEPT:
            self._before_accept(fl_ctx)
        elif event_type == AppEventType.BEFORE_AGGREGATION:
            self._before_aggregate(fl_ctx)

    def _startup(self):
        self._reset_stats()

    def _reset_stats(self):
        self.validation_metric_weighted_sum = 0
        self.validation_metric_sum_of_weights = 0

    def _before_accept(self, fl_ctx: FLContext):
        peer_ctx = fl_ctx.get_peer_context()
        shareable: Shareable = peer_ctx.get_prop(FLContextKey.SHAREABLE)
        try:
            dxo = from_shareable(shareable)
        except:
            self.log_exception(fl_ctx, "shareable data is not a valid DXO")
            return False

        if dxo.data_kind not in (DataKind.WEIGHT_DIFF, DataKind.WEIGHTS, DataKind.COLLECTION):
            self.log_debug(fl_ctx, "cannot handle {}".format(dxo.data_kind))
            return False

        if dxo.data is None:
            self.log_debug(fl_ctx, "no data to filter")
            return False

        contribution_round = shareable.get_cookie(AppConstants.CONTRIBUTION_ROUND)
        client_name = peer_ctx.get_identity_name(default="?")

        current_round = fl_ctx.get_prop(AppConstants.CURRENT_ROUND)

        if current_round == 0:
            self.log_debug(fl_ctx, "skipping round 0")
            return False  # There is no aggregated model at round 0

        if contribution_round != current_round:
            self.log_warning(
                fl_ctx,
                f"discarding shareable from {client_name} for round: {contribution_round}. Current round is: {current_round}",
            )
            return False

        validation_metric = dxo.get_meta_prop(self.validation_metric_name)
        if validation_metric is None:
            self.log_debug(fl_ctx, f"validation metric not existing in {client_name}")
            return False
        else:
            self.log_info(fl_ctx, f"validation metric {validation_metric} from client {client_name}")

        if self.weigh_by_local_iter:
            n_iter = dxo.get_meta_prop(MetaKey.NUM_STEPS_CURRENT_ROUND, 1.0)
        else:
            n_iter = 1.0

        aggregation_weights = self.aggregation_weights.get(client_name, 1.0)
        self.log_debug(fl_ctx, f"aggregation weight: {aggregation_weights}")

        weight = n_iter * aggregation_weights
        self.validation_metric_weighted_sum += validation_metric * weight
        self.validation_metric_sum_of_weights += weight
        return True

    def _before_aggregate(self, fl_ctx):
        if self.validation_metric_sum_of_weights == 0:
            self.log_debug(fl_ctx, "nothing accumulated")
            return False
        self.val_metric = self.validation_metric_weighted_sum / self.validation_metric_sum_of_weights
        self.logger.debug(f"weighted validation metric {self.val_metric}")
        if self.val_metric > self.best_val_metric:
            self.best_val_metric = self.val_metric
            current_round = fl_ctx.get_prop(AppConstants.CURRENT_ROUND)
            self.log_info(fl_ctx, f"new best validation metric at round {current_round}: {self.best_val_metric}")

            # Fire event to notify that the current global model is a new best
            self.fire_event(AppEventType.GLOBAL_BEST_MODEL_AVAILABLE, fl_ctx)

        self._reset_stats()
        return True


class IntimeModelSelectionHandler(IntimeModelSelector):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger.warning("'IntimeModelSelectionHandler' was renamed to 'IntimeModelSelector'")
