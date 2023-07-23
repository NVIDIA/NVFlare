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


import os
from abc import ABC, abstractmethod
from typing import Tuple

import xgboost as xgb
from xgboost import callback

from nvflare.apis.event_type import EventType
from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import FLContextKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.apis.workspace import Workspace
from nvflare.app_common.app_constant import AppConstants
from nvflare.security.logging import secure_format_exception, secure_log_traceback

from .constants import XGB_TRAIN_TASK, XGBShareableHeader
from .executor_spec import FedXGBHistogramExecutorSpec, XGBoostParams


class FedXGBHistogramExecutor(FedXGBHistogramExecutorSpec, Executor, ABC):
    """Federated XGBoost Executor Spec for histogram-base collaboration.

    This class implements the basic xgb_train logic, the subclass must implement load_data.
    """

    def __init__(self, num_rounds, early_stopping_round, xgboost_params: dict, verbose_eval=False):
        """Federated XGBoost Executor for histogram-base collaboration.

        This class sets up the training environment for Federated XGBoost.
        This is an abstract class, load_data method must be implemented by a subclass.
        This is the executor running on each NVFlare client, which starts XGBoost training.

        Args:
            num_rounds: number of boosting rounds
            early_stopping_round: early stopping round
            xgboost_params: This dict is passed to `xgboost.train()` as the first argument `params`.
                It contains all the Booster parameters.
                Please refer to XGBoost documentation for details:
                https://xgboost.readthedocs.io/en/stable/python/python_api.html#module-xgboost.training
            verbose_eval: verbose_eval in xgboost.train
        """
        super().__init__()

        self.num_rounds = num_rounds
        self.early_stopping_round = early_stopping_round
        self.verbose_eval = verbose_eval
        self.xgb_params = xgboost_params

        self.rank = None
        self.world_size = None
        self.client_id = None
        self._ca_cert_path = None
        self._client_key_path = None
        self._client_cert_path = None
        self._server_address = "localhost"

        self.dmat_train = None
        self.dmat_valid = None

    @abstractmethod
    def load_data(self) -> Tuple[xgb.core.DMatrix, xgb.core.DMatrix]:
        raise NotImplementedError

    def xgb_train(self, params: XGBoostParams) -> xgb.core.Booster:
        # Load file, file will not be sharded in federated mode.
        dtrain = self.dmat_train
        dval = self.dmat_valid

        # Specify validations set to watch performance
        watchlist = [(dval, "eval"), (dtrain, "train")]

        # Run training, all the features in training API is available.
        bst = xgb.train(
            params.xgb_params,
            dtrain,
            params.num_rounds,
            evals=watchlist,
            early_stopping_rounds=params.early_stopping_rounds,
            verbose_eval=params.verbose_eval,
            callbacks=[callback.EvaluationMonitor(rank=self.rank)],
        )

        return bst

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.START_RUN:
            engine = fl_ctx.get_engine()
            if engine.client.overseer_agent:
                sp = engine.client.overseer_agent.get_primary_sp()
                if sp and sp.primary is True:
                    self._server_address = sp.name
            self.client_id = fl_ctx.get_identity_name()
            self.log_info(fl_ctx, f"server address is {self._server_address}")

    def _get_certificates(self, fl_ctx: FLContext):
        workspace: Workspace = fl_ctx.get_prop(FLContextKey.WORKSPACE_OBJECT)
        bin_folder = workspace.get_startup_kit_dir()
        ca_cert_path = os.path.join(bin_folder, "rootCA.pem")
        if not os.path.exists(ca_cert_path):
            self.log_error(fl_ctx, "Missing ca certificate (rootCA.pem)")
            return False
        client_key_path = os.path.join(bin_folder, "client.key")
        if not os.path.exists(client_key_path):
            self.log_error(fl_ctx, "Missing client key (client.key)")
            return False
        client_cert_path = os.path.join(bin_folder, "client.crt")
        if not os.path.exists(client_cert_path):
            self.log_error(fl_ctx, "Missing client certificate (client.crt)")
            return False
        self._ca_cert_path = ca_cert_path
        self._client_key_path = client_key_path
        self._client_cert_path = client_cert_path
        return True

    def execute(self, task_name: str, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        self.log_info(fl_ctx, f"Client trainer got task: {task_name}")

        try:
            if task_name == XGB_TRAIN_TASK:
                return self.train(shareable, fl_ctx, abort_signal)
            else:
                self.log_error(fl_ctx, f"Could not handle task: {task_name}")
                return make_reply(ReturnCode.TASK_UNKNOWN)
        except Exception as e:
            # Task execution error, return EXECUTION_EXCEPTION Shareable
            self.log_exception(fl_ctx, f"learner execute exception: {secure_format_exception(e)}")
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)

    def train(self, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        """XGBoost training task pipeline which handles NVFlare specific tasks"""
        if abort_signal.triggered:
            return make_reply(ReturnCode.TASK_ABORTED)

        if self.dmat_train is None:
            self.dmat_train, self.dmat_valid = self.load_data()

        # Print round information
        current_round = shareable.get_header(AppConstants.CURRENT_ROUND)
        total_rounds = shareable.get_header(AppConstants.NUM_ROUNDS)
        client_name = fl_ctx.get_identity_name()
        self.log_info(fl_ctx, f"Client: {client_name} Round: {current_round}/{total_rounds}")

        rank_map = shareable.get_header(XGBShareableHeader.RANK_MAP)
        if client_name not in rank_map:
            self.log_error(fl_ctx, f"Train failed due to client {client_name} missing in rank_map: {rank_map}")
            return make_reply(ReturnCode.ERROR)

        world_size = shareable.get_header(XGBShareableHeader.WORLD_SIZE)
        if world_size is None:
            self.log_error(fl_ctx, f"Train failed in client {client_name}: missing xgboost world size in header.")
            return make_reply(ReturnCode.ERROR)

        xgb_fl_server_port = shareable.get_header(XGBShareableHeader.XGB_FL_SERVER_PORT)
        if xgb_fl_server_port is None:
            self.log_error(fl_ctx, f"Train failed in client {client_name}: missing xgboost FL server port in header.")
            return make_reply(ReturnCode.ERROR)

        secure_comm = shareable.get_header(XGBShareableHeader.XGB_FL_SERVER_SECURE)
        if secure_comm is None:
            self.log_error(fl_ctx, f"Train failed in client {client_name}: missing xgboost secure_comm in header.")
            return make_reply(ReturnCode.ERROR)

        self.rank = rank_map[client_name]
        self.world_size = world_size

        self.log_info(fl_ctx, f"Using xgb params: {self.xgb_params}")
        params = XGBoostParams(
            xgb_params=self.xgb_params,
            num_rounds=self.num_rounds,
            early_stopping_rounds=self.early_stopping_round,
            verbose_eval=self.verbose_eval,
        )

        communicator_env = {
            "xgboost_communicator": "federated",
            "federated_server_address": f"{self._server_address}:{xgb_fl_server_port}",
            "federated_world_size": self.world_size,
            "federated_rank": self.rank,
        }
        if secure_comm:
            if not self._get_certificates(fl_ctx):
                return make_reply(ReturnCode.ERROR)

            communicator_env["federated_server_cert"] = self._ca_cert_path
            communicator_env["federated_client_key"] = self._client_key_path
            communicator_env["federated_client_cert"] = self._client_cert_path

        try:
            with xgb.collective.CommunicatorContext(**communicator_env):
                bst = self.xgb_train(params)

                # Save the model.
                workspace = fl_ctx.get_prop(FLContextKey.WORKSPACE_OBJECT)
                run_number = fl_ctx.get_prop(FLContextKey.CURRENT_RUN)
                run_dir = workspace.get_run_dir(run_number)
                bst.save_model(os.path.join(run_dir, "test.model.json"))
                xgb.collective.communicator_print("Finished training\n")
        except Exception as e:
            secure_log_traceback()
            self.log_error(fl_ctx, f"Exception happens when running xgb train: {secure_format_exception(e)}")
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)

        return make_reply(ReturnCode.OK)
