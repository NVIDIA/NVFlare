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
from abc import ABC, abstractmethod

from nvflare.apis.collective_comm_constants import CollectiveCommShareableHeader
from nvflare.apis.fl_constant import FLContextKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.abstract.learner_spec import Learner
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.mpi_proxy.mpi_local_proxy import MpiLocalProxy

XGB_PREFIX = "xgb."


class XGBoostParams:
    """Container for all XGBoost parameters"""

    def __init__(self, train_data: str, test_data: str, num_rounds=10):
        self.train_data = train_data
        self.test_data = test_data
        self.num_rounds = num_rounds
        self.early_stopping_rounds = 2
        self.verbose_eval = False
        self.xgb_params: dict = {}
        self.rabit_env: list = []


class XGBoostFedLearnerBase(Learner, ABC):
    def __init__(
        self,
        train_data: str,
        test_data: str,
        num_rounds: int = 10,
        **kwargs
    ):
        """Federated XGBoost Learner.
        This class sets up the training environment for Federated XGBoost. This is an abstract class and xgb_train
        method must be implemented by a subclass.

        Args:
            train_data: Data file name for training
            test_data: Data file name for testing
        """
        super().__init__()

        self.train_data = train_data
        self.test_data = test_data
        self.num_rounds = num_rounds
        self.args = kwargs
        self.rank = None
        self.world_size = None
        self.client_id = None
        self.mpi_local_proxy = None

    @abstractmethod
    def xgb_train(self, params: XGBoostParams, fl_ctx: FLContext) -> Shareable:
        """Main XGBoost training method
        """
        pass

    def initialize(self, parts: dict, fl_ctx: FLContext):

        # get and print the args
        fl_args = fl_ctx.get_prop(FLContextKey.ARGS)
        self.client_id = fl_ctx.get_identity_name()
        self.log_info(
            fl_ctx,
            f"Client {self.client_id} initialized with args: \n {fl_args}",
        )

        self.mpi_local_proxy = MpiLocalProxy(fl_ctx)
        self.mpi_local_proxy.start()
        self.log_info(fl_ctx, f"MPI Proxy started on port: {self.mpi_local_proxy.port}")

    def finalize(self, fl_ctx: FLContext):
        if self.mpi_local_proxy:
            self.mpi_local_proxy.stop()

    def train(self, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        """XGBoost training task pipeline which handles NVFlare specific tasks
        """
        if abort_signal.triggered:
            return make_reply(ReturnCode.TASK_ABORTED)

        # Print round information
        current_round = shareable.get_header(AppConstants.CURRENT_ROUND)
        total_rounds = shareable.get_header(AppConstants.NUM_ROUNDS)
        client_name = fl_ctx.get_identity_name()
        self.log_info(fl_ctx, f"Client: {client_name} Round: {current_round + 1}/{total_rounds}")

        rank_map = shareable.get_header(CollectiveCommShareableHeader.RANK_MAP)
        if client_name not in rank_map:
            self.log_error(fl_ctx, f"Job aborted due to client {client_name} missing in rank_map: {rank_map}")
            return make_reply(ReturnCode.ERROR)

        self.rank = rank_map[client_name]
        self.world_size = shareable.get_header(CollectiveCommShareableHeader.WORLD_SIZE)

        params = XGBoostParams(self.train_data, self.test_data, self.num_rounds)

        rabit_env = [
            f"federated_server_address=localhost:{self.mpi_local_proxy.port}",
            f"federated_world_size={self.world_size}",
            f"federated_rank={self.rank}",
        ]
        params.rabit_env = rabit_env

        xgb_params = {}
        for key, value in self.args.items():
            if key.startswith(XGB_PREFIX):
                xgb_params[key[len(XGB_PREFIX)]] = value
        params.xgb_params = xgb_params

        return self.xgb_train(params, fl_ctx)

    def validate(self, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:

        self.log_error(fl_ctx, "Validation is not supported for XGBoost")

        return make_reply(ReturnCode.TASK_UNSUPPORTED)

