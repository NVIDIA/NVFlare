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
import os

import xgboost as xgb
from xgboost import callback

from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_context import FLContext
from nvflare.app_common.tracking.log_writer import LogWriter
from nvflare.app_opt.xgboost.data_loader import XGBDataLoader
from nvflare.app_opt.xgboost.histogram_based_v2.defs import Constant
from nvflare.app_opt.xgboost.histogram_based_v2.runners.xgb_runner import AppRunner
from nvflare.app_opt.xgboost.histogram_based_v2.tb import TensorBoardCallback
from nvflare.app_opt.xgboost.histogram_based_v2.xgb_params import XGBoostParams
from nvflare.app_opt.xgboost.metrics_cb import MetricsCallback
from nvflare.fuel.utils.import_utils import optional_import
from nvflare.fuel.utils.obj_utils import get_logger


class XGBClientRunner(AppRunner, FLComponent):
    def __init__(
        self,
        data_loader_id: str,
        early_stopping_rounds: int,
        xgb_params: dict,
        verbose_eval,
        use_gpus,
        model_file_name,
        metrics_writer_id: str = None,
    ):
        FLComponent.__init__(self)
        self.early_stopping_rounds = early_stopping_rounds
        self.xgb_params = xgb_params
        self.verbose_eval = verbose_eval
        self.use_gpus = use_gpus
        self.model_file_name = model_file_name
        self.data_loader_id = data_loader_id
        self.logger = get_logger(self)

        self._client_name = None
        self._rank = None
        self._world_size = None
        self._num_rounds = None
        self._server_addr = None
        self._data_loader = None
        self._tb_dir = None
        self._model_dir = None
        self._stopped = False
        self._metrics_writer_id = metrics_writer_id
        self._metrics_writer = None

    def initialize(self, fl_ctx: FLContext):
        engine = fl_ctx.get_engine()
        self._data_loader = engine.get_component(self.data_loader_id)
        if not isinstance(self._data_loader, XGBDataLoader):
            self.system_panic(f"data_loader should be type XGBDataLoader but got {type(self._data_loader)}", fl_ctx)

        if self._metrics_writer_id:
            self._metrics_writer = engine.get_component(self._metrics_writer_id)
            if not isinstance(self._metrics_writer, LogWriter):
                self.system_panic("writer should be type LogWriter", fl_ctx)

    def _xgb_train(self, params: XGBoostParams, train_data, val_data) -> xgb.core.Booster:
        """XGBoost training logic.

        Args:
            params (XGBoostParams): xgboost parameters.

        Returns:
            A xgboost booster.
        """
        # Specify validations set to watch performance
        watchlist = [(val_data, "eval"), (train_data, "train")]

        callbacks = [callback.EvaluationMonitor(rank=self._rank)]
        if self._metrics_writer:
            callbacks.append(MetricsCallback(self._metrics_writer))

        tensorboard, flag = optional_import(module="torch.utils.tensorboard")
        if flag and self._tb_dir:
            callbacks.append(TensorBoardCallback(self._tb_dir, tensorboard))

        # Run training, all the features in training API is available.
        bst = xgb.train(
            params.xgb_params,
            train_data,
            params.num_rounds,
            evals=watchlist,
            early_stopping_rounds=params.early_stopping_rounds,
            verbose_eval=params.verbose_eval,
            callbacks=callbacks,
        )
        return bst

    def run(self, ctx: dict):
        self._client_name = ctx.get(Constant.RUNNER_CTX_CLIENT_NAME)
        self._rank = ctx.get(Constant.RUNNER_CTX_RANK)
        self._world_size = ctx.get(Constant.RUNNER_CTX_WORLD_SIZE)
        self._num_rounds = ctx.get(Constant.RUNNER_CTX_NUM_ROUNDS)
        self._server_addr = ctx.get(Constant.RUNNER_CTX_SERVER_ADDR)
        # self._data_loader = ctx.get(Constant.RUNNER_CTX_DATA_LOADER)
        self._tb_dir = ctx.get(Constant.RUNNER_CTX_TB_DIR)
        self._model_dir = ctx.get(Constant.RUNNER_CTX_MODEL_DIR)

        if self.use_gpus:
            # mapping each rank to a GPU (can set to cuda:0 if simulating with only one gpu)
            self.logger.info(f"Training with GPU {self._rank}")
            self.xgb_params["device"] = f"cuda:{self._rank}"

        self.logger.info(f"Using xgb params: {self.xgb_params}")
        params = XGBoostParams(
            xgb_params=self.xgb_params,
            num_rounds=self._num_rounds,
            early_stopping_rounds=self.early_stopping_rounds,
            verbose_eval=self.verbose_eval,
        )

        self.logger.info(f"server address is {self._server_addr}")
        communicator_env = {
            "xgboost_communicator": "federated",
            "federated_server_address": f"{self._server_addr}",
            "federated_world_size": self._world_size,
            "federated_rank": self._rank,
            "plugin_name": "nvflare",
            "loader_params": {
                "LIBRARY_PATH": "/tmp",
            },
        }
        with xgb.collective.CommunicatorContext(**communicator_env):
            # Load the data. Dmatrix must be created with column split mode in CommunicatorContext for vertical FL
            train_data, val_data = self._data_loader.load_data(self._client_name)

            bst = self._xgb_train(params, train_data, val_data)

            # Save the model.
            bst.save_model(os.path.join(self._model_dir, self.model_file_name))
            xgb.collective.communicator_print("Finished training\n")

        self._stopped = True

    def stop(self):
        # currently no way to stop the runner
        pass

    def is_stopped(self) -> (bool, int):
        return self._stopped, 0
