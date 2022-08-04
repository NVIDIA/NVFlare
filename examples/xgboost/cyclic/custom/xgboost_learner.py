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

import json
from abc import abstractmethod

import xgboost as xgb
from sklearn.metrics import roc_auc_score
from torch.utils.tensorboard import SummaryWriter

from nvflare.apis.dxo import DXO, DataKind, from_shareable
from nvflare.apis.fl_constant import FLContextKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.abstract.learner_spec import Learner
from nvflare.app_common.app_constant import AppConstants


class XGBoostLearner(Learner):
    def __init__(
        self,
        trees_per_round: int = 1,
        train_task_name: str = AppConstants.TASK_TRAIN,
    ):
        super().__init__()
        self.trees_per_round = trees_per_round
        self.train_task_name = train_task_name

    def initialize(self, parts: dict, fl_ctx: FLContext):
        # when a run starts, this is where the actual settings get initialized for learner

        # set the paths according to fl_ctx
        engine = fl_ctx.get_engine()
        ws = engine.get_workspace()
        app_dir = ws.get_app_dir(fl_ctx.get_job_id())

        # get and print the args
        fl_args = fl_ctx.get_prop(FLContextKey.ARGS)
        self.client_id = fl_ctx.get_identity_name()
        self.log_info(
            fl_ctx,
            f"Client {self.client_id} initialized with args: \n {fl_args}",
        )

        # set local tensorboard writer for local training info of current model
        self.writer = SummaryWriter(app_dir)
        # set the training-related contexts, this is task-specific
        self.train_config(fl_ctx)

    def train_config(self, fl_ctx: FLContext):
        """Traning configurations customized to individual tasks
        This can be specified / loaded in any ways
        as long as they are made available for further training and validation
        some potential items include but not limited to:

        self.data_path
        self.model_path
        self.site_id_index_mapping

        self.lr
        self.objective
        self.max_depth
        self.eval_metric
        self.nthread

        self.train_X
        self.train_y
        self.valid_X
        self.valid_y
        """
        raise NotImplementedError

    @abstractmethod
    def finalize(self, fl_ctx: FLContext):
        # collect threads, close files here
        pass

    def train(
        self,
        shareable: Shareable,
        fl_ctx: FLContext,
        abort_signal: Signal,
    ) -> Shareable:

        if abort_signal.triggered:
            return make_reply(ReturnCode.TASK_ABORTED)

        # retrieve current global model download from server's shareable
        dxo = from_shareable(shareable)
        model_global = dxo.data

        # xgboost parameters
        param = {}
        param["objective"] = self.objective
        param["eta"] = self.lr
        param["max_depth"] = self.max_depth
        param["eval_metric"] = self.eval_metric
        param["nthread"] = self.nthread

        if not model_global:
            self.log_info(
                fl_ctx,
                f"Client {self.client_id} initial training from scratch",
            )
            bst = xgb.train(
                param,
                self.dmat_train,
                num_boost_round=self.trees_per_round,
                evals=[(self.dmat_valid, "validate"), (self.dmat_train, "train")],
            )
        else:
            self.log_info(
                fl_ctx,
                f"Client {self.client_id} training from global model received",
            )
            # save it to local temp file
            with open(self.model_path, "w") as f:
                json.dump(model_global, f)
            # validate global model
            bst_global = xgb.Booster(param, model_file=self.model_path)
            y_pred = bst_global.predict(self.dmat_valid)
            auc = roc_auc_score(self.valid_y, y_pred)
            self.log_info(
                fl_ctx,
                f"Client {self.client_id} global AUC {auc}",
            )
            # train local model starting with global model
            bst = xgb.train(
                param,
                self.dmat_train,
                num_boost_round=self.trees_per_round,
                xgb_model=self.model_path,
                evals=[(self.dmat_valid, "validate"), (self.dmat_train, "train")],
            )

        # Validate model after training
        y_pred = bst.predict(self.dmat_valid)
        auc = roc_auc_score(self.valid_y, y_pred)
        self.log_info(
            fl_ctx,
            f"Client {self.client_id} AUC after training: {auc}",
        )
        self.writer.add_scalar("AUC", auc, bst.num_boosted_rounds() - 1)
        bst.save_model(self.model_path)

        # report updated model in shareable
        with open(self.model_path) as json_file:
            model_new = json.load(json_file)

        dxo = DXO(data_kind=DataKind.WEIGHTS, data=model_new)
        self.log_info(fl_ctx, "Local epochs finished. Returning shareable")
        new_shareable = dxo.to_shareable()

        self.writer.flush()
        return new_shareable
