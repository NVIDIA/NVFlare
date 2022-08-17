# Copyright (c) 2021, NVIDIA CORPORATION.
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

from nvflare.apis.collective_comm_constants import CollectiveCommShareableHeader
from nvflare.apis.event_type import EventType
from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import FLContextKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.mpi_proxy.mpi_local_proxy import MpiLocalProxy


class MPIExecutor(Executor):
    def __init__(self, server_key_path, server_cert_path, client_cert_path, client_key_path):
        super().__init__()
        self._mpi_local_proxy = None
        self._server_key_path = server_key_path
        self._server_cert_path = server_cert_path
        self._client_cert_path = client_cert_path
        self._client_key_path = client_key_path

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.START_RUN:
            self._mpi_local_proxy = MpiLocalProxy(
                server_key_path=self._server_key_path,
                server_cert_path=self._server_cert_path,
                client_cert_path=self._client_cert_path,
                fl_context=fl_ctx,
            )
            self._mpi_local_proxy.start()
        elif event_type == EventType.END_RUN:
            if self._mpi_local_proxy:
                self._mpi_local_proxy.stop()

    def execute(self, task_name: str, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        self.log_info(fl_ctx, f"Executing {task_name}")
        try:
            if task_name == "mpi_train":
                self._do_training(shareable, fl_ctx)
                return make_reply(ReturnCode.OK)
            else:
                self.log_error(fl_ctx, f"{task_name} is not a supported task.")
                return make_reply(ReturnCode.TASK_UNKNOWN)
        except BaseException as e:
            self.log_exception(fl_ctx, f"Task {task_name} failed. Exception: {e.__str__()}")
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)

    def _do_training(self, shareable: Shareable, fl_ctx: FLContext):
        client_name = fl_ctx.get_prop(FLContextKey.CLIENT_NAME)
        rank = int(client_name.split("-")[1]) - 1
        world_size = shareable.get_header(CollectiveCommShareableHeader.WORLD_SIZE)
        rabit_env = [
            f"federated_server_address=localhost:{self._mpi_local_proxy.port}",
            f"federated_world_size={world_size}",
            f"federated_rank={rank}",
            f"federated_server_cert={self._server_cert_path}",
            f"federated_client_key={self._client_key_path}",
            f"federated_client_cert={self._client_cert_path}",
        ]
        with xgb.rabit.RabitContext([e.encode() for e in rabit_env]):
            # Load file, file will not be sharded in federated mode.
            dtrain = xgb.DMatrix("agaricus.txt.train")
            dtest = xgb.DMatrix("agaricus.txt.test")

            # Specify parameters via map, definition are same as c++ version
            param = {"max_depth": 2, "eta": 1, "objective": "binary:logistic"}

            # Specify validations set to watch performance
            watchlist = [(dtest, "eval"), (dtrain, "train")]
            num_round = 20

            # Run training, all the features in training API is available.
            bst = xgb.train(
                param,
                dtrain,
                num_round,
                evals=watchlist,
                early_stopping_rounds=2,
                verbose_eval=False,
                callbacks=[callback.EvaluationMonitor(rank=rank)],
            )

            # Save the model.
            workspace = fl_ctx.get_prop(FLContextKey.WORKSPACE_OBJECT)
            run_number = fl_ctx.get_prop(FLContextKey.CURRENT_RUN)
            run_dir = workspace.get_run_dir(run_number)
            bst.save_model(os.path.join(run_dir, "test.model.json"))
            xgb.rabit.tracker_print("Finished training\n")
