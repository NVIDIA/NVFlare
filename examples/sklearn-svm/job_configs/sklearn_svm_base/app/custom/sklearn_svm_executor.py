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

import numpy as np
import tensorboard
from joblib import dump
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score

from nvflare.apis.dxo import DXO, DataKind, from_shareable
from nvflare.apis.event_type import EventType
from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import FLContextKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.app_constant import AppConstants
from nvflare.security.logging import secure_format_exception


class FedSKLearnSVMExecutor(Executor, ABC):
    def __init__(
        self,
        local_model_path: str = "model.joblib",
        global_model_path: str = "model_global.joblib",
        train_task_name: str = AppConstants.TASK_TRAIN,
    ):
        super().__init__()
        self.client_id = None
        self.writer = None

        self.local_model_path = local_model_path
        self.global_model_path = global_model_path
        self.train_task_name = train_task_name

        self.X_train = None
        self.y_train = None
        self.X_valid = None
        self.y_valid = None

        self.local_support_X = None
        self.local_support_y = None
        self.global_support_X = None
        self.global_support_y = None

    @abstractmethod
    def load_data(self):
        """Loads data customized to individual tasks.

        This can be specified / loaded in any ways
        as long as they are made available for training and validation

        Return:
            A tuple of (X_train, y_train, X_valid, y_valid)
        """
        raise NotImplementedError

    def initialize(self, fl_ctx: FLContext):
        # set the paths according to fl_ctx
        engine = fl_ctx.get_engine()
        ws = engine.get_workspace()
        app_dir = fl_ctx.get_prop(FLContextKey.APP_ROOT)
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
        self.writer = tensorboard.summary.Writer(app_dir)

        # load data, this is task/site-specific
        self.X_train, self.y_train, self.X_valid, self.y_valid = self.load_data()


    def _local_validation(self, fl_ctx: FLContext, current_round):
        # local validation with global center
        # fit a standalone SVM with the global support vectors
        svm_global = SVC(kernel='rbf')
        svm_global.fit(self.global_support_X, self.global_support_y)
        # save global model
        dump(svm_global, self.global_model_path)
        y_pred = svm_global.predict(self.X_valid)
        auc = roc_auc_score(self.y_valid, y_pred)
        self.log_info(
            fl_ctx,
            f"Global AUC Score {auc} at Round {current_round}",
        )
        if self.writer:
            # note: writing auc before current training step, for passed in global model
            self.writer.add_scalar("AUC", auc, current_round)

    def _local_training(self, fl_ctx: FLContext):
        # local training
        svm = SVC(kernel='rbf')
        svm.fit(self.X_train, self.y_train)
        # save local model
        dump(svm, self.local_model_path)
        index = svm.support_
        self.local_support_X = self.X_train[index]
        self.local_support_y = self.y_train[index]

    def train(
        self,
        shareable: Shareable,
        fl_ctx: FLContext,
        abort_signal: Signal,
    ) -> Shareable:

        if abort_signal.triggered:
            self.finalize(fl_ctx)
            return make_reply(ReturnCode.TASK_ABORTED)

        # retrieve current global center download from server's shareable
        dxo = from_shareable(shareable)
        current_round = shareable.get_header(AppConstants.CURRENT_ROUND)
        global_param = dxo.data

        if not global_param:
            # first round, train on local data
            self.log_info(
                fl_ctx,
                f"Client {self.client_id} initialization, perform local train",
            )
            self._local_training(fl_ctx)
        else:
            # receive global model, validate it locally
            self.global_support_X = global_param["support_X"]
            self.global_support_y = global_param["support_y"]
            self.log_info(
                fl_ctx,
                f"Client {self.client_id} gets server model",
            )
            # validate and save global model
            self._local_validation(fl_ctx, current_round)

        # report updated model in shareable
        params = {"support_X": self.local_support_X, "support_y": self.local_support_y}
        dxo = DXO(data_kind=DataKind.WEIGHTS, data=params)
        self.log_info(fl_ctx, "Local epochs finished. Returning shareable")

        if self.writer:
            self.writer.flush()
        return dxo.to_shareable()

    def finalize(self, fl_ctx: FLContext):
        # freeing resources in finalize
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
