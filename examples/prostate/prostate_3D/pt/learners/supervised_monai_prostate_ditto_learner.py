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

import numpy as np
import torch
import torch.optim as optim
from monai.losses import DiceLoss
from monai.networks.nets.unet import UNet
from pt.helpers.supervised_pt_ditto import SupervisedPTDittoHelper
from pt.learners.supervised_monai_prostate_learner import SupervisedMonaiProstateLearner

from nvflare.apis.dxo import DXO, DataKind, MetaKey, from_shareable
from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.app_constant import AppConstants


class SupervisedMonaiProstateDittoLearner(SupervisedMonaiProstateLearner):
    def __init__(
        self,
        train_config_filename,
        aggregation_epochs: int = 1,
        ditto_model_epochs: int = 1,
        train_task_name: str = AppConstants.TASK_TRAIN,
    ):
        """Trainer for prostate segmentation task. It inherits from MONAI trainer.

        Args:
            train_config_filename: directory of config file.
            aggregation_epochs: the number of training epochs of global model for a round. Defaults to 1.
            ditto_model_epochs: the number of training epochs of personalized model for a round. Defaults to 1.
            train_task_name: name of the task to train the model.

        Returns:
            a Shareable with the updated local model after running `execute()`
        """
        SupervisedMonaiProstateLearner.__init__(
            self,
            train_config_filename=train_config_filename,
            aggregation_epochs=aggregation_epochs,
            train_task_name=train_task_name,
        )
        self.ditto_helper = None
        self.ditto_model_epochs = ditto_model_epochs

    def train_config(self, fl_ctx: FLContext):
        # Initialize superclass
        SupervisedMonaiProstateLearner.train_config(self, fl_ctx)

        engine = fl_ctx.get_engine()
        ws = engine.get_workspace()
        app_dir = ws.get_app_dir(fl_ctx.get_job_id())

        # Initialize PTDittoHelper
        ditto_model = UNet(
            dimensions=3,
            in_channels=1,
            out_channels=1,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
        ).to(self.device)
        ditto_optimizer = optim.SGD(ditto_model.parameters(), lr=self.config_info["ditto_learning_rate"], momentum=0.9)
        self.ditto_helper = SupervisedPTDittoHelper(
            criterion=DiceLoss(sigmoid=True),
            model=ditto_model,
            optimizer=ditto_optimizer,
            device=self.device,
            app_dir=app_dir,
            ditto_lambda=self.config_info["ditto_lambda"],
            model_epochs=self.ditto_model_epochs,
        )

    def train(
        self,
        shareable: Shareable,
        fl_ctx: FLContext,
        abort_signal: Signal,
    ) -> Shareable:
        """Training task pipeline for Ditto
        Get global model weights (potentially with HE)
        Prepare for fedprox loss
        Load Ditto personalized model info
        Local training reference model and personalized model
        Return updated weights of reference model (model_diff)
        """
        if abort_signal.triggered:
            return make_reply(ReturnCode.TASK_ABORTED)

        # get round information
        current_round = shareable.get_header(AppConstants.CURRENT_ROUND)
        total_rounds = shareable.get_header(AppConstants.NUM_ROUNDS)
        self.log_info(fl_ctx, f"Current/Total Round: {current_round + 1}/{total_rounds}")
        self.log_info(fl_ctx, f"Client identity: {fl_ctx.get_identity_name()}")

        # update local model weights with received weights
        dxo = from_shareable(shareable)
        global_weights = dxo.data

        # Before loading weights, tensors might need to be reshaped to support HE for secure aggregation.
        local_var_dict = self.model.state_dict()
        model_keys = global_weights.keys()
        for var_name in local_var_dict:
            if var_name in model_keys:
                weights = global_weights[var_name]
                try:
                    # reshape global weights to compute difference later on
                    global_weights[var_name] = np.reshape(weights, local_var_dict[var_name].shape)
                    # update the local dict
                    local_var_dict[var_name] = torch.as_tensor(global_weights[var_name])
                except Exception as e:
                    raise ValueError("Convert weight from {} failed with error: {}".format(var_name, str(e)))
        self.model.load_state_dict(local_var_dict)

        # Load Ditto personalized model
        self.ditto_helper.load_model(local_var_dict)

        # local steps
        epoch_len = len(self.train_loader)
        self.log_info(fl_ctx, f"Local steps per epoch: {epoch_len}")

        # make a copy of model_global as reference for
        # 1. FedProx loss of reference model
        # 2. Ditto loss of personalized model
        model_global = copy.deepcopy(self.model)
        for param in model_global.parameters():
            param.requires_grad = False

        # local train reference model
        self.local_train(
            fl_ctx=fl_ctx,
            train_loader=self.train_loader,
            model_global=model_global,
            abort_signal=abort_signal,
        )
        if abort_signal.triggered:
            return make_reply(ReturnCode.TASK_ABORTED)
        self.epoch_of_start_time += self.aggregation_epochs

        # local train ditto model
        self.ditto_helper.local_train(
            train_loader=self.train_loader, model_global=model_global, abort_signal=abort_signal, writer=self.writer
        )
        if abort_signal.triggered:
            return make_reply(ReturnCode.TASK_ABORTED)

        # local valid ditto model each round
        metric = self.local_valid(
            self.ditto_helper.model,
            self.valid_loader,
            abort_signal,
            tb_id="val_metric_per_model",
            record_epoch=self.ditto_helper.epoch_global,
        )
        if abort_signal.triggered:
            return make_reply(ReturnCode.TASK_ABORTED)
        self.log_info(fl_ctx, f"val_metric_per_model: {metric:.4f}")
        # save model
        self.ditto_helper.update_metric_save_model(metric=metric)

        # compute delta model, global model has the primary key set
        local_weights = self.model.state_dict()
        model_diff = {}
        for name in global_weights:
            if name not in local_weights:
                continue
            model_diff[name] = local_weights[name].cpu().numpy() - global_weights[name]
            if np.any(np.isnan(model_diff[name])):
                self.system_panic(f"{name} weights became NaN...", fl_ctx)
                return make_reply(ReturnCode.EXECUTION_EXCEPTION)

        # build the shareable
        dxo = DXO(data_kind=DataKind.WEIGHT_DIFF, data=model_diff)
        dxo.set_meta_prop(MetaKey.NUM_STEPS_CURRENT_ROUND, epoch_len)

        self.log_info(fl_ctx, "Local epochs finished. Returning shareable")
        return dxo.to_shareable()
