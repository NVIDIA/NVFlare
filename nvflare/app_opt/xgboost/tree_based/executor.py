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

import json
import os

import xgboost as xgb

from nvflare.apis.dxo import DXO, DataKind, from_shareable
from nvflare.apis.event_type import EventType
from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import FLContextKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_opt.xgboost.data_loader import XGBDataLoader
from nvflare.app_opt.xgboost.tree_based.utils import update_model
from nvflare.fuel.utils.import_utils import optional_import
from nvflare.security.logging import secure_format_exception


class FedXGBTreeExecutor(Executor):
    def __init__(
        self,
        training_mode,
        lr_scale,
        data_loader_id: str,
        num_client_bagging: int = 1,
        lr_mode: str = "uniform",
        local_model_path: str = "model.json",
        global_model_path: str = "model_global.json",
        learning_rate: float = 0.1,
        objective: str = "binary:logistic",
        num_local_parallel_tree: int = 1,
        local_subsample: float = 1,
        max_depth: int = 8,
        eval_metric: str = "auc",
        nthread: int = 16,
        tree_method: str = "hist",
        train_task_name: str = AppConstants.TASK_TRAIN,
        use_gpus=False,
    ):
        super().__init__()
        self.client_id = None
        self.writer = None

        self.training_mode = training_mode
        self.num_client_bagging = num_client_bagging
        self.lr = None
        self.lr_scale = lr_scale
        self.base_lr = learning_rate
        self.lr_mode = lr_mode
        self.num_local_parallel_tree = num_local_parallel_tree
        self.local_subsample = local_subsample
        self.local_model_path = local_model_path
        self.global_model_path = global_model_path
        self.objective = objective
        self.max_depth = max_depth
        self.eval_metric = eval_metric
        self.nthread = nthread
        self.tree_method = tree_method
        self.train_task_name = train_task_name
        self.use_gpus = use_gpus
        self.num_local_round = 1

        self.bst = None
        self.global_model_as_dict = None
        self.config = None
        self.local_model = None

        self.data_loader_id = data_loader_id
        self.train_data = None
        self.val_data = None

        # use dynamic shrinkage - adjusted by personalized scaling factor
        if lr_mode not in ["uniform", "scaled"]:
            raise ValueError(f"Only support [uniform] or [scaled] mode, but got {lr_mode}")

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
        self.rank = int(self.client_id.split("-")[1]) - 1

        # set local tensorboard writer for local training info of current model
        tensorboard, flag = optional_import(module="torch.utils.tensorboard")
        if flag:
            self.writer = tensorboard.SummaryWriter(app_dir)

        if self.training_mode not in ["cyclic", "bagging"]:
            self.system_panic(f"Only support [cyclic] or [bagging] mode, but got {self.training_mode}", fl_ctx)
            return

        # load data and lr_scale, this is task/site-specific
        data_loader = engine.get_component(self.data_loader_id)
        if not isinstance(data_loader, XGBDataLoader):
            self.system_panic("data_loader should be type XGBDataLoader", fl_ctx)
        data_loader.initialize(
            client_id=self.client_id,
            rank=self.rank,
        )
        try:
            self.train_data, self.val_data = data_loader.load_data()
        except Exception as e:
            self.system_panic(f"load_data failed: {secure_format_exception(e)}", fl_ctx)
        self.lr = self._get_effective_learning_rate()

    def _get_effective_learning_rate(self):
        if self.training_mode == "bagging":
            # Bagging mode
            if self.lr_mode == "uniform":
                # uniform lr, global learning_rate scaled by num_client_bagging for bagging
                lr = self.base_lr / self.num_client_bagging
            else:
                # scaled lr, global learning_rate scaled by data size percentage
                lr = self.base_lr * self.lr_scale
        else:
            # Cyclic mode, directly use the base learning_rate
            lr = self.base_lr
        return lr

    def _get_xgb_train_params(self):
        params = {
            "objective": self.objective,
            "eta": self.lr,
            "max_depth": self.max_depth,
            "eval_metric": self.eval_metric,
            "nthread": self.nthread,
            "num_parallel_tree": self.num_local_parallel_tree,
            "subsample": self.local_subsample,
            "tree_method": self.tree_method,
        }
        return params

    def _local_boost_bagging(self, fl_ctx: FLContext):
        eval_results = self.bst.eval_set(
            evals=[(self.train_data, "train"), (self.val_data, "valid")], iteration=self.bst.num_boosted_rounds() - 1
        )
        self.log_info(fl_ctx, eval_results)
        auc = float(eval_results.split("\t")[2].split(":")[1])
        for i in range(self.num_local_round):
            self.bst.update(self.train_data, self.bst.num_boosted_rounds())

        # extract newly added self.num_local_round using xgboost slicing api
        bst = self.bst[self.bst.num_boosted_rounds() - self.num_local_round : self.bst.num_boosted_rounds()]

        self.log_info(
            fl_ctx,
            f"Global AUC {auc}",
        )
        if self.writer:
            # note: writing auc before current training step, for passed in global model
            self.writer.add_scalar(
                "AUC", auc, int((self.bst.num_boosted_rounds() - self.num_local_round - 1) / self.num_client_bagging)
            )
        return bst

    def _local_boost_cyclic(self, fl_ctx: FLContext):
        # Cyclic mode
        # starting from global model
        # return the whole boosting tree series
        self.bst.update(self.train_data, self.bst.num_boosted_rounds())
        eval_results = self.bst.eval_set(
            evals=[(self.train_data, "train"), (self.val_data, "valid")], iteration=self.bst.num_boosted_rounds() - 1
        )
        self.log_info(fl_ctx, eval_results)
        auc = float(eval_results.split("\t")[2].split(":")[1])
        self.log_info(
            fl_ctx,
            f"Client {self.client_id} AUC after training: {auc}",
        )
        if self.writer:
            self.writer.add_scalar("AUC", auc, self.bst.num_boosted_rounds() - 1)
        return self.bst

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
        model_update = dxo.data

        # xgboost parameters
        params = self._get_xgb_train_params()

        if self.use_gpus:
            # mapping each rank to a GPU (can set to cuda:0 if simulating with only one gpu)
            self.log_info(fl_ctx, f"Training with GPU {self.rank}")
            params["device"] = f"cuda:{self.rank}"

        if not self.bst:
            # First round
            self.log_info(
                fl_ctx,
                f"Client {self.client_id} initial training from scratch",
            )
            if not model_update:
                bst = xgb.train(
                    params,
                    self.train_data,
                    num_boost_round=self.num_local_round,
                    evals=[(self.val_data, "validate"), (self.train_data, "train")],
                )
            else:
                loadable_model = bytearray(model_update["model_data"])
                bst = xgb.train(
                    params,
                    self.train_data,
                    num_boost_round=self.num_local_round,
                    xgb_model=loadable_model,
                    evals=[(self.val_data, "validate"), (self.train_data, "train")],
                )
            self.config = bst.save_config()
            self.bst = bst
        else:
            self.log_info(
                fl_ctx,
                f"Client {self.client_id} model updates received from server",
            )
            if self.training_mode == "bagging":
                model_updates = model_update["model_data"]
                for update in model_updates:
                    self.global_model_as_dict = update_model(self.global_model_as_dict, json.loads(update))

                loadable_model = bytearray(json.dumps(self.global_model_as_dict), "utf-8")
            else:
                loadable_model = bytearray(model_update["model_data"])

            self.log_info(
                fl_ctx,
                f"Client {self.client_id} converted global model to json ",
            )

            self.bst.load_model(loadable_model)
            self.bst.load_config(self.config)

            self.log_info(
                fl_ctx,
                f"Client {self.client_id} loaded global model into booster ",
            )

            # train local model starting with global model
            if self.training_mode == "bagging":
                bst = self._local_boost_bagging(fl_ctx)
            else:
                bst = self._local_boost_cyclic(fl_ctx)

        self.local_model = bst.save_raw("json")

        # report updated model in shareable
        dxo = DXO(data_kind=DataKind.WEIGHTS, data={"model_data": self.local_model})
        self.log_info(fl_ctx, "Local epochs finished. Returning shareable")
        new_shareable = dxo.to_shareable()

        if self.writer:
            self.writer.flush()
        return new_shareable

    def finalize(self, fl_ctx: FLContext):
        # freeing resources in finalize avoids seg fault during shutdown of gpu mode
        del self.bst
        del self.train_data
        del self.val_data
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
