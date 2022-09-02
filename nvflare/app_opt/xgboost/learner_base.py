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

import os
from abc import ABC, abstractmethod

from nvflare.apis.fl_constant import FLContextKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.apis.workspace import Workspace
from nvflare.app_common.abstract.learner_spec import Learner
from nvflare.app_common.app_constant import AppConstants

from .constants import XGBShareableHeader


class XGBoostParams:
    """Container for all XGBoost parameters"""

    def __init__(self, train_data: str, test_data: str, num_rounds=10, early_stopping_rounds=2):
        self.train_data = train_data
        self.test_data = test_data
        self.num_rounds = num_rounds
        self.early_stopping_rounds = early_stopping_rounds
        self.verbose_eval = False
        self.xgb_params: dict = {}
        self.rabit_env: list = []


def _get_server_address(fl_ctx: FLContext):
    try:
        engine = fl_ctx.get_engine()
        server_address = engine.client.overseer_agent.get_primary_sp().name
    except BaseException:
        return None
    return server_address


class XGBFedLearnerBase(Learner, ABC):
    def __init__(
        self, train_data: str, test_data: str, num_rounds: int = 10, early_stopping_round: int = 2, xgboost_params=None
    ):
        """Federated XGBoost Learner.
        This class sets up the training environment for Federated XGBoost. This is an abstract class and xgb_train
        method must be implemented by a subclass.

        Args:
            train_data: Data file name for training
            test_data: Data file name for testing
            num_rounds: number of boosting rounds
            early_stopping_round: early stopping round
            xgboost_params: parameters to passed in xgb

        """
        super().__init__()

        self.train_data = train_data
        self.test_data = test_data
        self.num_rounds = num_rounds
        self.early_stopping_round = early_stopping_round
        self.xgb_params = xgboost_params
        self.rank = None
        self.world_size = None
        self.client_id = None
        self._ca_cert_path = None
        self._client_key_path = None
        self._client_cert_path = None

    @abstractmethod
    def xgb_train(self, params: XGBoostParams, fl_ctx: FLContext) -> Shareable:
        """Main XGBoost training method"""
        pass

    def initialize(self, parts: dict, fl_ctx: FLContext):
        self.client_id = fl_ctx.get_identity_name()

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

    def train(self, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        """XGBoost training task pipeline which handles NVFlare specific tasks"""
        if abort_signal.triggered:
            return make_reply(ReturnCode.TASK_ABORTED)

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

        server_address = _get_server_address(fl_ctx)
        if server_address is None:
            self.log_error(fl_ctx, f"Train failed in client {client_name}: Can't get NVFlare server address")
            return make_reply(ReturnCode.ERROR)

        secure_comm = shareable.get_header(XGBShareableHeader.XGB_FL_SERVER_SECURE)
        if secure_comm is None:
            self.log_error(fl_ctx, f"Train failed in client {client_name}: missing xgboost secure_comm in header.")
            return make_reply(ReturnCode.ERROR)

        self.rank = rank_map[client_name]
        self.world_size = world_size

        params = XGBoostParams(
            self.train_data, self.test_data, self.num_rounds, early_stopping_rounds=self.early_stopping_round
        )

        rabit_env = [
            f"federated_server_address={server_address}:{xgb_fl_server_port}",
            f"federated_world_size={self.world_size}",
            f"federated_rank={self.rank}",
        ]
        if secure_comm:
            if not self._get_certificates(fl_ctx):
                return make_reply(ReturnCode.ERROR)

            rabit_env.extend(
                [
                    f"federated_server_cert={self._ca_cert_path}",
                    f"federated_client_key={self._client_key_path}",
                    f"federated_client_cert={self._client_cert_path}",
                ]
            )
        params.rabit_env = rabit_env

        self.log_info(fl_ctx, f"Using xgb params: {self.xgb_params}")
        params.xgb_params = self.xgb_params

        try:
            result = self.xgb_train(params, fl_ctx)
        except BaseException:
            self.log_error(fl_ctx, "Exception happens when running xgb train")
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)
        return result

    def validate(self, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:

        self.log_error(fl_ctx, "Validation is not supported for XGBoost")

        return make_reply(ReturnCode.TASK_UNSUPPORTED)
