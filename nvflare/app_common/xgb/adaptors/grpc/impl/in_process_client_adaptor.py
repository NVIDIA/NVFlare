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
import threading

import xgboost as xgb
from xgboost import callback

from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContext
from nvflare.app_common.xgb.adaptors.grpc.client_adaptor import GrpcClientAdaptor
from nvflare.app_common.xgb.data_loader import XGBDataLoader
from nvflare.app_common.xgb.tb import TensorBoardCallback
from nvflare.app_common.xgb.xgb_params import XGBoostParams
from nvflare.fuel.utils.import_utils import optional_import
from nvflare.security.logging import secure_format_exception, secure_log_traceback


class InProcessGrpcClientAdaptor(GrpcClientAdaptor):
    def __init__(
        self,
        early_stopping_rounds,
        xgb_params: dict,
        data_loader_id: str,
        verbose_eval=False,
        use_gpus=False,
        grpc_options=None,
        req_timeout=10.0,
        model_file_name="model.json",
    ):
        GrpcClientAdaptor.__init__(self, grpc_options, req_timeout)
        self.early_stopping_rounds = early_stopping_rounds
        self.xgb_params = xgb_params
        self.data_loader_id = data_loader_id
        self.verbose_eval = verbose_eval
        self.use_gpus = use_gpus
        self.model_file_name = model_file_name

        self._training_stopped = False
        self._asked_to_stop = False
        self._train_data = None
        self._val_data = None
        self._client_id = None
        self._data_loader = None
        self._app_dir = None
        self._workspace = None
        self._run_dir = None

    def initialize(self, fl_ctx: FLContext):
        self._client_id = fl_ctx.get_identity_name()
        engine = fl_ctx.get_engine()
        ws = engine.get_workspace()
        self._app_dir = ws.get_app_dir(fl_ctx.get_job_id())
        self._data_loader = engine.get_component(self.data_loader_id)
        if not isinstance(self._data_loader, XGBDataLoader):
            self.system_panic(f"data_loader should be type XGBDataLoader but got {type(self._data_loader)}", fl_ctx)

        self._workspace = fl_ctx.get_prop(FLContextKey.WORKSPACE_OBJECT)
        run_number = fl_ctx.get_prop(FLContextKey.CURRENT_RUN)
        self._run_dir = self._workspace.get_run_dir(run_number)

    def _xgb_train(self, params: XGBoostParams) -> xgb.core.Booster:
        """XGBoost training logic.

        Args:
            params (XGBoostParams): xgboost parameters.

        Returns:
            A xgboost booster.
        """
        # Load file, file will not be sharded in federated mode.
        dtrain = self._train_data
        dval = self._val_data

        # Specify validations set to watch performance
        watchlist = [(dval, "eval"), (dtrain, "train")]

        callbacks = [callback.EvaluationMonitor(rank=self.rank)]
        tensorboard, flag = optional_import(module="torch.utils.tensorboard")
        if flag and self._app_dir:
            callbacks.append(TensorBoardCallback(self._app_dir, tensorboard))

        # Run training, all the features in training API is available.
        bst = xgb.train(
            params.xgb_params,
            dtrain,
            params.num_rounds,
            evals=watchlist,
            early_stopping_rounds=params.early_stopping_rounds,
            verbose_eval=params.verbose_eval,
            callbacks=callbacks,
        )
        return bst

    def start_client(self, server_addr: str, port: int):
        t = threading.Thread(target=self._do_start_client, args=(server_addr,), daemon=True)
        t.start()

    def _do_start_client(self, server_addr: str):
        try:
            self._do_train(server_addr)
        except Exception as e:
            secure_log_traceback()
            self.logger.error(f"Exception happens when running xgb train: {secure_format_exception(e)}")
        self._training_stopped = True

    def _do_train(self, server_addr: str):
        if self.use_gpus:
            # mapping each rank to a GPU (can set to cuda:0 if simulating with only one gpu)
            self.logger.info(f"Training with GPU {self.rank}")
            self.xgb_params["device"] = f"cuda:{self.rank}"

        self.logger.info(f"Using xgb params: {self.xgb_params}")
        params = XGBoostParams(
            xgb_params=self.xgb_params,
            num_rounds=self.num_rounds,
            early_stopping_rounds=self.early_stopping_rounds,
            verbose_eval=self.verbose_eval,
        )

        self.logger.info(f"server address is {server_addr}")
        communicator_env = {
            "xgboost_communicator": "federated",
            "federated_server_address": f"{server_addr}",
            "federated_world_size": self.world_size,
            "federated_rank": self.rank,
        }
        with xgb.collective.CommunicatorContext(**communicator_env):
            # Load the data. Dmatrix must be created with column split mode in CommunicatorContext for vertical FL
            if not self._train_data or not self._val_data:
                self._train_data, self._val_data = self._data_loader.load_data(self._client_id)

            bst = self._xgb_train(params)

            # Save the model.
            bst.save_model(os.path.join(self._run_dir, self.model_file_name))
            xgb.collective.communicator_print("Finished training\n")

    def stop_client(self):
        # currently there is no way to stop the client while training
        self._asked_to_stop = True

    def is_client_stopped(self) -> (bool, int):
        return self._training_stopped, 0
