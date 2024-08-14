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
from typing import Tuple

import xgboost as xgb
from xgboost import callback

from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_constant import FLContextKey, SystemConfigs
from nvflare.apis.fl_context import FLContext
from nvflare.app_common.tracking.log_writer import LogWriter
from nvflare.app_opt.xgboost.data_loader import XGBDataLoader
from nvflare.app_opt.xgboost.histogram_based_v2.defs import Constant
from nvflare.app_opt.xgboost.histogram_based_v2.runners.xgb_runner import AppRunner
from nvflare.app_opt.xgboost.metrics_cb import MetricsCallback
from nvflare.fuel.utils.config_service import ConfigService
from nvflare.fuel.utils.obj_utils import get_logger
from nvflare.utils.cli_utils import get_package_root

PLUGIN_PARAM_KEY = "federated_plugin"
PLUGIN_KEY_NAME = "name"
PLUGIN_KEY_PATH = "path"
MODEL_FILE_NAME = "model.json"


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
        self.fl_ctx = None

        self._client_name = None
        self._rank = None
        self._world_size = None
        self._num_rounds = None
        self._split_mode = None
        self._secure_training = None
        self._xgb_params = None
        self._xgb_options = None
        self._server_addr = None
        self._data_loader = None
        self._model_dir = None
        self._stopped = False
        self._metrics_writer_id = metrics_writer_id
        self._metrics_writer = None

    def initialize(self, fl_ctx: FLContext):
        self.fl_ctx = fl_ctx
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
            xgb_params: The Boost parameters for XGBoost train method
            xgb_options: Other arguments needed by XGBoost
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

        early_stopping_rounds = xgb_options.get("early_stopping_rounds", 0)
        verbose_eval = xgb_options.get("verbose_eval", False)

        # Check for pre-trained model
        job_id = self.fl_ctx.get_prop(FLContextKey.CURRENT_JOB_ID)
        workspace = self.fl_ctx.get_prop(FLContextKey.WORKSPACE_OBJECT)
        custom_dir = workspace.get_app_custom_dir(job_id)
        model_file = os.path.join(custom_dir, MODEL_FILE_NAME)
        if os.path.isfile(model_file):
            self.logger.info(f"Pre-trained model is used: {model_file}")
            xgb_model = model_file
        else:
            xgb_model = None

        # Run training, all the features in training API is available.
        bst = xgb.train(
            xgb_params,
            train_data,
            num_rounds,
            evals=watchlist,
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=verbose_eval,
            callbacks=callbacks,
            xgb_model=xgb_model,
        )
        return bst

    def run(self, ctx: dict):
        self._client_name = ctx.get(Constant.RUNNER_CTX_CLIENT_NAME)
        self._rank = ctx.get(Constant.RUNNER_CTX_RANK)
        self._world_size = ctx.get(Constant.RUNNER_CTX_WORLD_SIZE)
        self._num_rounds = ctx.get(Constant.RUNNER_CTX_NUM_ROUNDS)
        self._split_mode = ctx.get(Constant.RUNNER_CTX_SPLIT_MODE)
        self._secure_training = ctx.get(Constant.RUNNER_CTX_SECURE_TRAINING)
        self._xgb_params = ctx.get(Constant.RUNNER_CTX_XGB_PARAMS)
        self._xgb_options = ctx.get(Constant.RUNNER_CTX_XGB_OPTIONS)
        self._server_addr = ctx.get(Constant.RUNNER_CTX_SERVER_ADDR)
        self._model_dir = ctx.get(Constant.RUNNER_CTX_MODEL_DIR)

        use_gpus = self._xgb_options.get("use_gpus", False)
        if use_gpus:
            # mapping each rank to a GPU (can set to cuda:0 if simulating with only one gpu)
            self.logger.info(f"Training with GPU {self._rank}")
            self._xgb_params["device"] = f"cuda:{self._rank}"

        self.logger.info(
            f"XGB split_mode: {self._split_mode} secure_training: {self._secure_training} "
            f"params: {self._xgb_params} XGB options: {self._xgb_options}"
        )

        self.logger.info(f"server address is {self._server_addr}")

        communicator_env = {
            "dmlc_communicator": "federated",
            "federated_server_address": f"{self._server_addr}",
            "federated_world_size": self._world_size,
            "federated_rank": self._rank,
        }

        if not self._secure_training:
            self.logger.info("XGBoost non-secure training")
        else:
            xgb_plugin_name = ConfigService.get_str_var(
                name="xgb_plugin_name", conf=SystemConfigs.RESOURCES_CONF, default=None
            )
            xgb_plugin_path = ConfigService.get_str_var(
                name="xgb_plugin_path", conf=SystemConfigs.RESOURCES_CONF, default=None
            )
            xgb_plugin_params: dict = ConfigService.get_dict_var(
                name=PLUGIN_PARAM_KEY, conf=SystemConfigs.RESOURCES_CONF, default={}
            )

            # path and name can be overwritten by scalar configuration
            if xgb_plugin_name:
                xgb_plugin_params[PLUGIN_KEY_NAME] = xgb_plugin_name

            if xgb_plugin_path:
                xgb_plugin_params[PLUGIN_KEY_PATH] = xgb_plugin_path

            # Set default plugin name
            if not xgb_plugin_params.get(PLUGIN_KEY_NAME):
                xgb_plugin_params[PLUGIN_KEY_NAME] = "cuda_paillier"

            if not xgb_plugin_params.get(PLUGIN_KEY_PATH):
                # This only works on Linux. Need to support other platforms
                lib_ext = "so"
                lib_name = f"lib{xgb_plugin_params[PLUGIN_KEY_NAME]}.{lib_ext}"
                xgb_plugin_params[PLUGIN_KEY_PATH] = str(get_package_root() / "libs" / lib_name)

            communicator_env[PLUGIN_PARAM_KEY] = xgb_plugin_params

        with xgb.collective.CommunicatorContext(**communicator_env):
            # Load the data. Dmatrix must be created with column split mode in CommunicatorContext for vertical FL
            train_data, val_data = self._data_loader.load_data(self._client_name, self._split_mode)

            bst = self._xgb_train(self._num_rounds, self._xgb_params, self._xgb_options, train_data, val_data)

            # Save the model.
            bst.save_model(os.path.join(self._model_dir, self.model_file_name))
            xgb.collective.communicator_print("Finished training\n")

        self._stopped = True

    def stop(self):
        # currently no way to stop the runner
        pass

    def is_stopped(self) -> Tuple[bool, int]:
        return self._stopped, 0
