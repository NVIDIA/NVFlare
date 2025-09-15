# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import time
from typing import Dict, Optional, Set

import numpy as np

from nvflare.apis.dxo import DXO, DataKind
from nvflare.apis.fl_context import FLContext
from nvflare.app_common.abstract.model import make_model_learnable
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.app_event_type import AppEventType
from nvflare.edge.aggregators.model_update_dxo import ModelUpdateDXOAggregator
from nvflare.edge.assessors.model_manager import ModelManager
from nvflare.edge.mud import ModelUpdate


class _ModelState:
    def __init__(self, aggr: ModelUpdateDXOAggregator):
        self.aggregator = aggr
        self.devices = {}
        self.last_update_time = None

    def accept(self, model_update: ModelUpdate, fl_ctx: FLContext):
        self.last_update_time = time.time()
        self.devices.update(model_update.devices)
        return self.aggregator.accept(model_update.update, fl_ctx)


class BuffModelManager(ModelManager):
    def __init__(
        self,
        num_updates_for_model: int,
        max_model_history: Optional[int] = None,
        global_lr: float = 1.0,
        staleness_weight: bool = False,
    ):
        """Initialize the ModelManager.
        The aggregation scheme and weights are calculated following FedBuff paper "Federated Learning with Buffered Asynchronous Aggregation".
        The staleness_weight can be enabled to apply staleness weighting to model updates.

        Special cases for max_model_history:
        - If None: Keep every model versions, only remove a version when all devices processing it reports back (version no longer related with any device_id in the current_selection from device_manager).

        Args:
            num_updates_for_model (int): Number of updates required before generating a new model version.
            max_model_history (int): Maximum number of historical model versions to keep in memory.
                - None (default): keep every version until all devices processing a particular version report back.
                - positive integer: keep only the latest n versions
            global_lr (float): Global learning rate for model aggregation, default is 1.0.
            staleness_weight (bool): Whether to apply staleness weighting to model updates, default is False.
        """

        super().__init__()
        self.num_updates_for_model = num_updates_for_model
        self.num_updates_counter = 0
        self.max_model_history = max_model_history
        self.global_lr = global_lr
        self.staleness_weight = staleness_weight

    def initialize_model(self, model: DXO, fl_ctx: FLContext):
        self.current_model = model
        # updates is a dict of model version to _ModelState
        self.updates[self.current_model_version] = _ModelState(ModelUpdateDXOAggregator())

    def prune_model_versions(self, versions_to_keep: Set[int], fl_ctx: FLContext) -> None:
        # go through all versions and remove the ones:
        # - either not in versions_to_keep
        # - or too old (current_model_version - v >= max_model_history)
        versions_to_remove = set()

        for v in self.updates.keys():
            if v not in versions_to_keep:
                versions_to_remove.add(v)
            if self.max_model_history and self.current_model_version - v >= self.max_model_history:
                versions_to_remove.add(v)

        # Remove the identified versions
        for v in versions_to_remove:
            self.log_info(fl_ctx, f"removed model version {v}")
            self.updates.pop(v)
        # log the current total number of model versions
        self.log_info(fl_ctx, f"current total number of active model versions: {len(self.updates)}")

    def generate_new_model(self, fl_ctx: FLContext) -> None:
        # New model generated based on the current global weights and all updates
        new_model = {}
        self.current_model_version += 1

        # counter to confirm the number of updates
        num_updates = 0

        if self.current_model_version == 1:
            # Initial global weights
            new_model = self.current_model.data
        else:
            # Aggregate all updates
            for v, ms in self.updates.items():
                if self.staleness_weight:
                    weight = 1 / (1 + (self.current_model_version - v) ** 0.5)
                else:
                    weight = 1.0
                aggr = ms.aggregator
                # Add the dict to new_model by multiplying the weight and dividing by the count
                update_dict = aggr.dict
                count = aggr.count

                if count > 0:
                    # aggregate updates
                    for key, value in update_dict.items():
                        # apply weight and divide by count
                        value = weight * value / count
                        # update the new model
                        if key not in new_model:
                            new_model[key] = value
                        else:
                            new_model[key] = new_model[key] + value

                # Reset aggr after counting its contribution
                ms.aggregator.reset(fl_ctx)
                num_updates += count

            # Add the aggregated updates to the current global weights
            global_weights = self.current_model.data
            for key, value in new_model.items():
                if key not in global_weights:
                    self.log_error(fl_ctx, f"key {key} not in new model")
                    continue
                new_model[key] = np.array(global_weights[key]) + value * self.global_lr

        # create the ModelState for the new model version
        self.updates[self.current_model_version] = _ModelState(ModelUpdateDXOAggregator())
        self.log_info(fl_ctx, f"generated new model version {self.current_model_version} with {num_updates} updates")

        # update the current model
        # convert new_model items from numpy arrays to lists for serialization
        new_model = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in new_model.items()}
        self.current_model = DXO(data_kind=DataKind.WEIGHTS, data=new_model)

        # reset the num_updates_counter
        self.num_updates_counter = 0

        # set fl_ctx and fire the event
        # wrap new_model to a learnable
        learnable = make_model_learnable(new_model, {})
        fl_ctx.set_prop(AppConstants.GLOBAL_MODEL, learnable, private=True, sticky=True)
        fl_ctx.set_prop(AppConstants.CURRENT_ROUND, self.current_model_version, private=True, sticky=True)
        self.fire_event(AppEventType.GLOBAL_WEIGHTS_UPDATED, fl_ctx)

    def process_updates(self, model_updates: Dict[int, ModelUpdate], fl_ctx: FLContext) -> bool:
        accepted = True
        for model_version, model_update in model_updates.items():
            if model_version <= 0:
                continue

            if not model_update:
                self.log_error(fl_ctx, f"bad child update version {model_version}: no update data")
                continue

            # Check if version is too old before accepting
            if self.max_model_history:
                # if max_model_history is set, output warning for updates that are too old
                if self.current_model_version - model_version >= self.max_model_history:
                    self.log_warning(
                        fl_ctx,
                        f"dropped child update version {model_version}. Current version {self.current_model_version}. Max history {self.max_model_history}",
                    )
                    continue

            # Accept the update and aggregate it to the corresponding model version
            model_state = self.updates.get(model_version)
            accepted = model_state.accept(model_update, fl_ctx)
            self.log_info(
                fl_ctx,
                f"processed child update V{model_version} with {len(model_update.devices)} devices: {accepted=}",
            )

            # update the global num_updates_counter
            self.num_updates_counter += len(model_update.devices)

        current_model_state = self.updates.get(self.current_model_version)
        if isinstance(current_model_state, _ModelState):
            if self.num_updates_counter >= self.num_updates_for_model:
                self.log_info(
                    fl_ctx,
                    f"Globally got {self.num_updates_counter} updates: generate new model version",
                )
                self.generate_new_model(fl_ctx)

        return accepted
