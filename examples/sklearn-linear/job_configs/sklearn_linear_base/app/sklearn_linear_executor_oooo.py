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
import copy
import os
from abc import ABC, abstractmethod

from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score
from joblib import dump
import warnings
import numpy as np

from nvflare.apis.dxo import DXO, DataKind, MetaKey, from_shareable
from nvflare.apis.event_type import EventType
from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import FLContextKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.app_constant import AppConstants
#from nvflare.fuel.utils.import_utils import optional_import
from nvflare.security.logging import secure_format_exception

import tensorboard

class FedSKLearnLinearExecutor(Executor, ABC):
    def __init__(
            self,
            local_model_path: str = "model.joblib",
            global_model_path: str = "model_global.joblib",
            learning_rate: str = "constant",
            eta0: float = 1e-4,
            loss: str = "log",
            penalty: str = "l2",
            fit_intercept: int = 1,
            eval_metric: str = "auc",
            train_task_name: str = AppConstants.TASK_TRAIN,
    ):
        super().__init__()
        self.client_id = None
        self.writer = None

        self.local_model_path = local_model_path
        self.global_model_path = global_model_path

        self.learning_rate = learning_rate
        self.eta0 = eta0
        self.loss = loss
        self.penalty = penalty
        self.fit_intercept = fit_intercept

        self.eval_metric = eval_metric
        self.train_task_name = train_task_name

        self.local_model = None

        self.X_train = None
        self.y_train = None
        self.X_valid = None
        self.y_valid = None
        self.sample_size = None

    @abstractmethod
    def load_data(self):
        """Loads data customized to individual tasks.

        This can be specified / loaded in any ways
        as long as they are made available for training and validation

        Return:
            A tuple of (X_train, y_train, X_valid, y_valid, total_train_data_num)
        """
        raise NotImplementedError

    def initialize(self, fl_ctx: FLContext):
        # set the paths according to fl_ctx
        engine = fl_ctx.get_engine()
        ws = engine.get_workspace()
        app_dir = ws.get_app_dir(fl_ctx.get_job_id())
        self.local_model_path = os.path.join(app_dir, self.local_model_path)
        self.global_model_path = os.path.join(app_dir, self.global_model_path)

        # get and print the args
        fl_args = fl_ctx.get_prop(FLContextKey.ARGS)
        self.client_id = fl_ctx.get_identity_name()
        self.log_info(
            fl_ctx,
            f"Client {self.client_id} initialized with args: \n {fl_args}",
        )

        # set local tensorboard writer for local training info of current model
        #tensorboard, flag = optional_import(module="torch.utils.tensorboard")
        #if flag:
        self.writer = tensorboard.summary.Writer(app_dir)

        # load data and lr_scale, this is task/site-specific
        self.X_train, self.y_train, self.X_valid, self.y_valid, self.sample_size = self.load_data()

        # initialize model to all zero
        self.local_model = SGDClassifier(loss=self.loss, penalty=self.penalty, fit_intercept=bool(self.fit_intercept),
                                         learning_rate=self.learning_rate, eta0=self.eta0, max_iter=1, warm_start=True)
        n_classes = 2  # Binary classification
        n_features = self.X_train.shape[1]  # Number of features in dataset
        self.local_model.classes_ = np.array([i for i in range(n_classes)])
        self.local_model.coef_ = np.zeros((1, n_features))
        if self.fit_intercept:
            self.local_model.intercept_ = np.zeros((1,))

    def _get_model_parameters(self):
        if self.local_model.fit_intercept:
            params = {"coef": self.local_model.coef_, "intercept": self.local_model.intercept_}
        else:
            params = {"coef": self.local_model.coef_}
        return copy.deepcopy(params)

    def _set_model_params(self, params):
        self.local_model.coef_ = params["coef"]
        if self.local_model.fit_intercept:
            self.local_model.intercept_ = params["intercept"]

    def _local_validation(self, fl_ctx: FLContext, current_round):
        y_pred = self.local_model.predict(self.X_valid)
        auc = roc_auc_score(self.y_valid, y_pred)
        self.log_info(
            fl_ctx,
            f"Global AUC {auc} at Round {current_round}",
        )
        if self.writer:
            # note: writing auc before current training step, for passed in global model
            self.writer.add_scalar("AUC", auc, current_round)

    def _local_training(self, fl_ctx: FLContext):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.local_model.fit(self.X_train, self.y_train)

    def train(
            self,
            shareable: Shareable,
            fl_ctx: FLContext,
            abort_signal: Signal,
    ) -> Shareable:

        if abort_signal.triggered:
            self.finalize(fl_ctx)
            return make_reply(ReturnCode.TASK_ABORTED)

        # retrieve current global model download from server's shareable
        dxo = from_shareable(shareable)
        current_round = shareable.get_header(AppConstants.CURRENT_ROUND)
        global_param = dxo.data

        if not global_param:
            # First round
            self.log_info(
                fl_ctx,
                f"Client {self.client_id} initial training from scratch",
            )
        else:
            # Load global model param
            self._set_model_params(global_param)
            self.log_info(
                fl_ctx,
                f"Client {self.client_id} model updates with server model",
            )

        # validate global model
        self._local_validation(fl_ctx, current_round)
        # save global model
        dump(self.local_model, self.global_model_path)
        # train local model starting with global model
        self._local_training(fl_ctx)
        # save local model
        dump(self.local_model, self.local_model_path)
        params = self._get_model_parameters()
        # report updated model in shareable
        dxo = DXO(data_kind=DataKind.WEIGHTS, data=params)
        dxo.set_meta_prop(MetaKey.NUM_STEPS_CURRENT_ROUND, self.sample_size)
        self.log_info(fl_ctx, "Local epochs finished. Returning shareable")

        if self.writer:
            self.writer.flush()
        return dxo.to_shareable()

    def finalize(self, fl_ctx: FLContext):
        # freeing resources in finalize avoids seg fault during shutdown of gpu mode
        del self.local_model
        del self.X_train
        del self.y_train
        del self.X_valid
        del self.y_valid
        self.log_info(fl_ctx, "Freed training resources")

    def execute(self, task_name: str, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        self.log_info(fl_ctx, f"Client trainer got task: {task_name}")

        try:
            if task_name == "train":
                return self.train(shareable, fl_ctx, abort_signal)
            else:
                self.log_error(fl_ctx, f"Could not handle task: {task_name}")
                return make_reply(ReturnCode.TASK_UNKNOWN)
        except Exception as e:
            # Task execution error, return EXECUTION_EXCEPTION Shareable
            self.log_exception(fl_ctx, f"execute exception: {secure_format_exception(e)}")
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.START_RUN:
            self.initialize(fl_ctx)
        elif event_type == EventType.END_RUN:
            self.finalize(fl_ctx)
