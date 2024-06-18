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
from nvflare.apis.fl_constant import SystemConfigs
from nvflare.apis.fl_context import FLContext
from nvflare.app_common.tracking.log_writer import LogWriter
from nvflare.app_opt.xgboost.data_loader import XGBDataLoader
from nvflare.app_opt.xgboost.histogram_based_v2.defs import SECURE_TRAINING_MODES, Constant
from nvflare.app_opt.xgboost.histogram_based_v2.runners.xgb_runner import AppRunner
from nvflare.app_opt.xgboost.histogram_based_v2.tb import TensorBoardCallback
from nvflare.app_opt.xgboost.metrics_cb import MetricsCallback
from nvflare.fuel.utils.config_service import ConfigService
from nvflare.fuel.utils.import_utils import optional_import
from nvflare.fuel.utils.obj_utils import get_logger
from nvflare.utils.cli_utils import get_package_root

LOADER_PARAMS_LIBRARY_PATH = "LIBRARY_PATH"


class XGBClientRunner(AppRunner, FLComponent):
    def __init__(
        self,
        data_loader_id: str,
        model_file_name,
        metrics_writer_id: str = None,
    ):
        FLComponent.__init__(self)
        self.model_file_name = model_file_name
        self.data_loader_id = data_loader_id
        self.logger = get_logger(self)

        self._client_name = None
        self._rank = None
        self._world_size = None
        self._num_rounds = None
        self._training_mode = None
        self._xgb_params = None
        self._xgb_options = None
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

    def _xgb_train(self, num_rounds, xgb_params: dict, xgb_options: dict, train_data, val_data) -> xgb.core.Booster:
        """XGBoost training logic.

        Args:
            num_rounds: Number of rounds
            xgb_params (XGBoostParams): xgboost parameters.
            xgb_options: Other options needed by xgboost
            train_data: Training data
            val_data: Validation data

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

        early_stopping_rounds = xgb_options.get("early_stopping_rounds", 0)
        verbose_eval = xgb_options.get("verbose_eval", False)

        # Run training, all the features in training API is available.
        bst = xgb.train(
            xgb_params,
            train_data,
            num_rounds,
            evals=watchlist,
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=verbose_eval,
            callbacks=callbacks,
        )
        return bst

    def run(self, ctx: dict):
        self._client_name = ctx.get(Constant.RUNNER_CTX_CLIENT_NAME)
        self._rank = ctx.get(Constant.RUNNER_CTX_RANK)
        self._world_size = ctx.get(Constant.RUNNER_CTX_WORLD_SIZE)
        self._num_rounds = ctx.get(Constant.RUNNER_CTX_NUM_ROUNDS)
        self._training_mode = ctx.get(Constant.RUNNER_CTX_TRAINING_MODE)
        self._xgb_params = ctx.get(Constant.RUNNER_CTX_XGB_PARAMS)
        self._xgb_options = ctx.get(Constant.RUNNER_CTX_XGB_OPTIONS)
        self._server_addr = ctx.get(Constant.RUNNER_CTX_SERVER_ADDR)
        # self._data_loader = ctx.get(Constant.RUNNER_CTX_DATA_LOADER)
        self._tb_dir = ctx.get(Constant.RUNNER_CTX_TB_DIR)
        self._model_dir = ctx.get(Constant.RUNNER_CTX_MODEL_DIR)

        use_gpus = self._xgb_options.get("use_gpus", False)
        if use_gpus:
            # mapping each rank to a GPU (can set to cuda:0 if simulating with only one gpu)
            self.logger.info(f"Training with GPU {self._rank}")
            self._xgb_params["device"] = f"cuda:{self._rank}"

        self.logger.info(
            f"XGB training_mode: {self._training_mode} " f"params: {self._xgb_params} XGB options: {self._xgb_options}"
        )
        self.logger.info(f"server address is {self._server_addr}")

        communicator_env = {
            "xgboost_communicator": "federated",
            "federated_server_address": f"{self._server_addr}",
            "federated_world_size": self._world_size,
            "federated_rank": self._rank,
        }

        if self._training_mode not in SECURE_TRAINING_MODES:
            self.logger.info("XGBoost non-secure training")
        else:
            xgb_plugin_name = ConfigService.get_str_var(
                name="xgb_plugin_name", conf=SystemConfigs.RESOURCES_CONF, default="nvflare"
            )

            xgb_loader_params = ConfigService.get_dict_var(
                name="xgb_loader_params", conf=SystemConfigs.RESOURCES_CONF, default={}
            )

            # Library path is frequently used, add a scalar config var and overwrite what's in the dict
            xgb_library_path = ConfigService.get_str_var(name="xgb_library_path", conf=SystemConfigs.RESOURCES_CONF)
            if xgb_library_path:
                xgb_loader_params[LOADER_PARAMS_LIBRARY_PATH] = xgb_library_path

            lib_path = xgb_loader_params.get(LOADER_PARAMS_LIBRARY_PATH, None)
            if not lib_path:
                xgb_loader_params[LOADER_PARAMS_LIBRARY_PATH] = str(get_package_root() / "libs")

            xgb_proc_params = ConfigService.get_dict_var(
                name="xgb_proc_params", conf=SystemConfigs.RESOURCES_CONF, default={}
            )

            self.logger.info(
                f"XGBoost secure mode: {self._training_mode} plugin_name: {xgb_plugin_name} "
                f"proc_params: {xgb_proc_params} loader_params: {xgb_loader_params}"
            )

            communicator_env.update(
                {
                    "plugin_name": xgb_plugin_name,
                    "proc_params": xgb_proc_params,
                    "loader_params": xgb_loader_params,
                }
            )

        with xgb.collective.CommunicatorContext(**communicator_env):
            # Load the data. Dmatrix must be created with column split mode in CommunicatorContext for vertical FL
            train_data, val_data = self._data_loader.load_data(self._client_name, self._training_mode)

            bst = self._xgb_train(self._num_rounds, self._xgb_params, self._xgb_options, train_data, val_data)

            # Save the model.
            bst.save_model(os.path.join(self._model_dir, self.model_file_name))
            xgb.collective.communicator_print("Finished training\n")

        self._stopped = True

    def stop(self):
        # currently no way to stop the runner
        pass

    def is_stopped(self) -> (bool, int):
        return self._stopped, 0
