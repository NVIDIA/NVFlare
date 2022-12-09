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
from nvflare.apis.dxo import DXO, DataKind, from_shareable
from nvflare.apis.event_type import EventType
from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import FLContextKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.app_constant import AppConstants
from nvflare.security.logging import secure_format_exception
from sklearn.cluster import KMeans, MiniBatchKMeans, kmeans_plusplus
from sklearn.metrics import homogeneity_score


class FedSKLearnKMeansExecutor(Executor, ABC):
    def __init__(
        self,
        local_model_path: str = "model.joblib",
        global_model_path: str = "model_global.joblib",
        n_clusters: int = 2,
        train_task_name: str = AppConstants.TASK_TRAIN,
    ):
        super().__init__()
        self.client_id = None
        self.writer = None

        self.local_model_path = local_model_path
        self.global_model_path = global_model_path

        self.n_clusters = n_clusters

        self.train_task_name = train_task_name

        self.X_train = None
        self.y_train = None
        self.X_valid = None
        self.y_valid = None
        self.n_samples = None

        self.center_global = None
        self.center_local = None
        self.count_local = None

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
        (
            self.X_train,
            self.y_train,
            self.X_valid,
            self.y_valid,
            self.n_samples,
        ) = self.load_data()

        # initialize model record (center and count) to all zero
        # note that the model needs to be created every round
        # due to the available API for center initialization
        # thus there is no self.local_model
        n_features = self.X_train.shape[1]
        self.center_local = np.zeros([self.n_clusters, n_features])
        self.count_local = np.zeros([self.n_clusters])
        self.center_global = np.zeros([self.n_clusters, n_features])

    def _local_validation(self, fl_ctx: FLContext, current_round):
        # local validation with global center
        # fit a standalone KMeans with just the given center
        kmeans_global = KMeans(
            n_clusters=self.n_clusters, init=self.center_global, n_init=1
        )
        kmeans_global.fit(self.center_global)
        # save global model
        dump(kmeans_global, self.global_model_path)
        y_pred = kmeans_global.predict(self.X_valid)
        homo = homogeneity_score(self.y_valid, y_pred)
        self.log_info(
            fl_ctx,
            f"Global Homogeneity Score {homo} at Round {current_round}",
        )
        if self.writer:
            # note: writing auc before current training step, for passed in global model
            self.writer.add_scalar("Homogeneity", homo, current_round)

    def _local_training(self, fl_ctx: FLContext):
        # local training from global center
        kmeans = MiniBatchKMeans(
            n_clusters=self.n_clusters,
            batch_size=self.n_samples,
            max_iter=1,
            init=self.center_global,
            n_init=1,
            reassignment_ratio=0,
        )
        kmeans.fit(self.X_train)
        # save local model
        dump(kmeans, self.local_model_path)
        self.center_local = kmeans.cluster_centers_
        self.count_local = kmeans._counts

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
            # validate and save global model
            self._local_validation(fl_ctx, current_round)
            # first round, compute initial center with kmeans++ method
            self.log_info(
                fl_ctx,
                f"Client {self.client_id} initialization, use kmeans++ to get center",
            )
            self.center_local, _ = kmeans_plusplus(
                self.X_train, n_clusters=self.n_clusters
            )
        else:
            # receive global center, initialize and train local model
            self.center_global = global_param["center"]
            self.log_info(
                fl_ctx,
                f"Client {self.client_id} center updates with server model",
            )
            # validate and save global model
            self._local_validation(fl_ctx, current_round)
            # train local model starting with global model
            self._local_training(fl_ctx)

        # report updated model in shareable
        params = {"center": self.center_local, "count": self.count_local}
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

    def execute(
        self,
        task_name: str,
        shareable: Shareable,
        fl_ctx: FLContext,
        abort_signal: Signal,
    ) -> Shareable:
        self.log_info(fl_ctx, f"Client trainer got task: {task_name}")

        try:
            if task_name == "train":
                return self.train(shareable, fl_ctx, abort_signal)
            else:
                self.log_error(fl_ctx, f"Could not handle task: {task_name}")
                return make_reply(ReturnCode.TASK_UNKNOWN)
        except Exception as e:
            # Task execution error, return EXECUTION_EXCEPTION Shareable
            self.log_exception(
                fl_ctx, f"execute exception: {secure_format_exception(e)}"
            )
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.START_RUN:
            self.initialize(fl_ctx)
        elif event_type == EventType.END_RUN:
            self.finalize(fl_ctx)
