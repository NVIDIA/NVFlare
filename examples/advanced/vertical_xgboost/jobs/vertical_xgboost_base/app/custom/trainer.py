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

from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import FLContextKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_opt.xgboost.histogram_based.executor import TensorBoardCallback
from nvflare.fuel.utils.import_utils import optional_import


class SupportedTasks(object):
    TRAIN = "train"


class XGBoostTrainer(Executor):
    def __init__(
        self,
        server_address: str,
        world_size: int,
        data_loader_id: str,
        num_rounds: int,
        early_stopping_rounds: int,
        xgb_params: dict,
        server_cert_path: str = None,
        client_key_path: str = None,
        client_cert_path: str = None,
    ):
        """Trainer for federated XGBoost.

        Args:
            server_address: address for the gRPC server to connect to.
            world_size: the number of sites.
            data_loader_id: data_loader component id.
            num_rounds: number of boosting iterations.
            early_stopping_rounds: early stopping if no val improvement after this many rounds.
            xgb_params: dict of booster params for training.
            server_cert_path: the path to the server certificate file.
            client_key_path: the path to the client key file.
            client_cert_path: the path to the client certificate file.
        """
        super().__init__()
        self.server_address = server_address
        self.world_size = world_size
        self.data_loader_id = data_loader_id
        self.num_rounds = num_rounds
        self.early_stopping_rounds = early_stopping_rounds
        self.xgb_params = xgb_params
        self.server_cert_path = server_cert_path
        self.client_key_path = client_key_path
        self.client_cert_path = client_cert_path

    def initalize(self, fl_ctx: FLContext):
        engine = fl_ctx.get_engine()
        self.data_loader = engine.get_component(self.data_loader_id)
        self.app_dir = engine.get_workspace().get_app_dir(fl_ctx.get_job_id())
        self.client_id = fl_ctx.get_identity_name()
        self.rank = int(self.client_id.split("-")[1]) - 1

    def execute(self, task_name: str, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        self.log_info(fl_ctx, f"Executing {task_name}")
        try:
            if task_name == SupportedTasks.TRAIN:
                self.initalize(fl_ctx)
                self.train(fl_ctx)
                return make_reply(ReturnCode.OK)
            else:
                self.log_error(fl_ctx, f"{task_name} is not a supported task.")
                return make_reply(ReturnCode.TASK_UNKNOWN)
        except BaseException as e:
            self.log_exception(fl_ctx, f"Task {task_name} failed. Exception: {e.__str__()}")
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)

    def train(self, fl_ctx: FLContext):
        # using the xgboost federated learning plugin
        communicator_env = {
            "xgboost_communicator": "federated",
            "federated_server_address": self.server_address,
            "federated_world_size": self.world_size,
            "federated_rank": self.rank,
        }

        # if using ssl
        if self.server_cert_path and self.client_key_path and self.client_cert_path:
            communicator_env["federated_server_cert"] = self.server_cert_path
            communicator_env["federated_client_key"] = self.client_key_path
            communicator_env["federated_client_cert"] = self.client_cert_path

        with xgb.collective.CommunicatorContext(**communicator_env):
            # only one site holds the labels
            if self.rank == 0:
                label = "&label_column=0"
            else:
                label = ""

            train_path, test_path = self.data_loader.load_data(fl_ctx)
            dtrain = xgb.DMatrix(train_path + f"?format=csv{label}", data_split_mode=1)
            dtest = xgb.DMatrix(test_path + f"?format=csv{label}", data_split_mode=1)

            # specify validations set to watch performance
            watchlist = [(dtest, "eval"), (dtrain, "train")]

            callbacks = []
            tensorboard, flag = optional_import(module="torch.utils.tensorboard")
            if flag and self.app_dir:
                callbacks.append(TensorBoardCallback(self.app_dir, tensorboard))

            # train with booster params from config
            bst = xgb.train(
                self.xgb_params,
                dtrain,
                self.num_rounds,
                evals=watchlist,
                early_stopping_rounds=self.early_stopping_rounds,
                callbacks=callbacks,
            )

            # save the model
            workspace = fl_ctx.get_prop(FLContextKey.WORKSPACE_OBJECT)
            run_number = fl_ctx.get_prop(FLContextKey.CURRENT_RUN)
            run_dir = workspace.get_run_dir(run_number)
            bst.save_model(os.path.join(run_dir, "higgs.model.federated.vertical.json"))
            xgb.collective.communicator_print(f"{self.client_id} finished training\n")
