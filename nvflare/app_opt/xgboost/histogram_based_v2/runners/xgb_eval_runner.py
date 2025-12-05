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

import os
from typing import Tuple

import xgboost as xgb
from sklearn.metrics import roc_auc_score

from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_context import FLContext
from nvflare.app_opt.xgboost.data_loader import XGBDataLoader
from nvflare.app_opt.xgboost.histogram_based_v2.defs import Constant
from nvflare.app_opt.xgboost.histogram_based_v2.runners.xgb_runner import AppRunner
from nvflare.fuel.utils.log_utils import get_obj_logger


def _check_ctx(ctx: dict):
    required_ctx_keys = [
        Constant.RUNNER_CTX_CLIENT_NAME,
        Constant.RUNNER_CTX_RANK,
        Constant.RUNNER_CTX_WORLD_SIZE,
        Constant.RUNNER_CTX_SERVER_ADDR,
    ]
    for k in required_ctx_keys:
        if k not in ctx:
            raise RuntimeError(f"Missing {k} in context.")


class XGBEvalRunner(AppRunner, FLComponent):
    def __init__(
        self,
        data_loader_id: str,
        train_workspace_path: str,
    ):
        FLComponent.__init__(self)
        self.data_loader_id = data_loader_id
        self.train_workspace_path = train_workspace_path
        self.logger = get_obj_logger(self)
        self.fl_ctx = None

        self._client_name = None
        self._rank = None
        self._world_size = None
        self._data_split_mode = None
        self._server_addr = None
        self._data_loader = None
        self._stopped = False

    def initialize(self, fl_ctx: FLContext):
        self.fl_ctx = fl_ctx
        engine = fl_ctx.get_engine()
        self._data_loader = engine.get_component(self.data_loader_id)
        if not isinstance(self._data_loader, XGBDataLoader):
            self.system_panic(f"data_loader should be type XGBDataLoader but got {type(self._data_loader)}", fl_ctx)

    def _load_trained_model(self) -> xgb.core.Booster:
        """Load the trained model from the training workspace.

        Returns:
            A xgboost booster loaded from the trained model.
        """
        # Load the trained model from the training workspace
        model_path = os.path.join(self.train_workspace_path, f"{self._client_name}/simulate_job/model.json")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Trained model not found at {model_path}")

        bst = xgb.Booster({"nthread": 1})
        bst.load_model(model_path)
        self.logger.info(f"Loaded trained model from {model_path}")

        return bst

    def _evaluate_model(self, bst: xgb.core.Booster, val_data) -> float:
        """Evaluate the model and return metrics.

        Args:
            bst: The trained XGBoost model
            val_data: Validation data

        Returns:
            AUC score for the evaluation
        """
        # Make predictions
        preds = bst.predict(val_data)

        # Only label owner (rank 0) calculates and reports metrics
        if self._rank == 0:
            y_valid = val_data.get_label()
            auc_score = roc_auc_score(y_valid, preds)
            return auc_score
        else:
            # For non-label owners, just return 0 as they don't have labels
            return 0.0

    def run(self, ctx: dict):
        _check_ctx(ctx)
        self._client_name = ctx[Constant.RUNNER_CTX_CLIENT_NAME]
        self._rank = ctx[Constant.RUNNER_CTX_RANK]
        self._world_size = ctx[Constant.RUNNER_CTX_WORLD_SIZE]
        self._data_split_mode = ctx.get(Constant.RUNNER_CTX_DATA_SPLIT_MODE, 0)
        self._server_addr = ctx[Constant.RUNNER_CTX_SERVER_ADDR]

        self.logger.info(f"XGB eval, server address is {self._server_addr}")

        communicator_env = {
            "dmlc_communicator": "federated",
            "federated_server_address": f"{self._server_addr}",
            "federated_world_size": self._world_size,
            "federated_rank": self._rank,
        }

        # Plugins are required during training to enable federated communication and coordination between clients.
        # For inference, the model is already trained and only needs to be evaluated locally or collectively.
        # Therefore, plugin functionality is not needed for inference.

        self._data_loader.initialize(
            client_id=self._client_name, rank=self._rank, data_split_mode=self._data_split_mode
        )

        with xgb.collective.CommunicatorContext(**communicator_env):
            # Load the validation data. Dmatrix must be created with column split mode in CommunicatorContext for vertical FL
            _, val_data = self._data_loader.load_data()

            # Load the trained model
            bst = self._load_trained_model()

            # Evaluate the model
            auc_score = self._evaluate_model(bst, val_data)

            self.logger.info(f"AUC: {auc_score}")

            xgb.collective.communicator_print("Finished evaluation\n")

        self._stopped = True

    def stop(self):
        # currently no way to stop the runner
        pass

    def is_stopped(self) -> Tuple[bool, int]:
        return self._stopped, 0
