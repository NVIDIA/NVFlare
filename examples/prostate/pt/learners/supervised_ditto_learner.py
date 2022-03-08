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

import numpy as np
import torch

from nvflare.apis.dxo import DXO, DataKind, MetaKey, from_shareable
from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.app_constant import AppConstants

from pt.learners.supervised_learner import SupervisedLearner


class SupervisedDittoLearner(SupervisedLearner):
    def __init__(
        self,
        train_config_filename,
        aggregation_epochs: int = 1,
        local_model_epochs: int = 1,
        train_task_name: str = AppConstants.TASK_TRAIN,
        submit_model_task_name: str = AppConstants.TASK_SUBMIT_MODEL,
    ):
        """Simple Supervised Trainer with Ditto functionality.

        Args:
            train_config_filename: directory of config file.
            aggregation_epochs: the number of training epochs of global model for a round. Defaults to 1.
            local_model_epochs: the number of training epochs of local model for a round. Defaults to 1.
            train_task_name: name of the task to train the model.
            submit_model_task_name: name of the task to submit the best local model.

        Returns:
            a Shareable with the updated local model after running `execute()`
            or the best local model depending on the specified task.
        """
        super().__init__(
            train_config_filename=train_config_filename,
            aggregation_epochs=aggregation_epochs,
            train_task_name=train_task_name,
            submit_model_task_name=submit_model_task_name,
        )
        # trainer init happens at the very beginning, only the basic info regarding the trainer is set here
        # the actual run has not started at this point
        self.local_model_epochs = local_model_epochs

        # Ditto parameters
        self.ditto_lr_ref = None
        self.ditto_lr = None
        self.ditto_lambda = None
        self.ditto_criterion_prox = None

    def local_train(
        self,
        fl_ctx,
        train_loader,
        model_global,
        abort_signal: Signal,
        val_freq: int = 0,
    ):
        """
        val_freq: the validation interval for local training
        """
        # Train reference model for self.aggregation_epochs
        for epoch in range(self.aggregation_epochs):
            if abort_signal.triggered:
                return
            self.model_ref.train()
            self.log_info(
                fl_ctx,
                f"Local ref model epoch {self.client_id}: {epoch + 1}/{self.aggregation_epochs} (lr={self.ditto_lr_ref})",
            )
            for i, batch_data in enumerate(train_loader):
                if abort_signal.triggered:
                    return
                inputs = batch_data["image"].to(self.device)
                labels = batch_data["label"].to(self.device)

                # zero the parameter gradients
                self.optimizer_ref.zero_grad()
                # forward + backward + optimize
                outputs_ref = self.model_ref(inputs)
                loss_ref = self.criterion_ref(outputs_ref, labels)

                # FedProx loss term
                if self.fedproxloss_mu > 0:
                    fed_prox_loss = self.criterion_prox(self.model_ref, model_global)
                    loss_ref += fed_prox_loss

                loss_ref.backward()
                self.optimizer_ref.step()

        # Train local model for self.local_model_epochs, and keep track of curves
        for epoch in range(self.local_model_epochs):
            if abort_signal.triggered:
                return
            self.model.train()
            epoch_len = len(train_loader)
            self.epoch_global = self.epoch_of_start_time + epoch
            self.log_info(
                fl_ctx,
                f"Local model epoch {self.client_id}: {epoch + 1}/{self.local_model_epochs} (lr={self.ditto_lr})",
            )
            for i, batch_data in enumerate(train_loader):
                if abort_signal.triggered:
                    return
                inputs = batch_data["image"].to(self.device)
                labels = batch_data["label"].to(self.device)

                # zero the parameter gradients
                self.optimizer.zero_grad()
                # forward + backward + optimize
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                # add the prox loss term for ditto
                prox_loss = self.ditto_criterion_prox(self.model, model_global)
                loss += prox_loss

                loss.backward()
                self.optimizer.step()
                self.writer.add_scalar("train_loss", loss.item(), epoch_len * self.epoch_global + i)
            if val_freq > 0 and epoch % val_freq == 0:
                acc = self.local_valid(self.model, self.valid_loader, abort_signal, tb_id="val_acc_local_model")
                if acc > self.best_acc:
                    self.save_model(is_best=True)

    def train(
        self,
        shareable: Shareable,
        fl_ctx: FLContext,
        abort_signal: Signal,
    ) -> Shareable:
        # Check abort signal
        if abort_signal.triggered:
            return make_reply(ReturnCode.TASK_ABORTED)

        # get round information
        current_round = shareable.get_header(AppConstants.CURRENT_ROUND)
        total_rounds = shareable.get_header(AppConstants.NUM_ROUNDS)
        self.log_info(fl_ctx, f"Current/Total Round: {current_round + 1}/{total_rounds}")
        self.log_info(fl_ctx, f"Client identity: {fl_ctx.get_identity_name()}")

        # update reference model weights with received weights
        dxo = from_shareable(shareable)
        global_weights = dxo.data

        # Before loading weights, tensors might need to be reshaped to support HE for secure aggregation.
        # FL is performed on the reference model: model_ref
        local_var_dict = self.model_ref.state_dict()
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
        self.model_ref.load_state_dict(local_var_dict)

        # load local model from last round's record if model exist,
        # otherwise initialize from global model for the first round.
        if os.path.exists(self.local_model_file):
            model_data = torch.load(self.local_model_file)
            self.model.load_state_dict(model_data["model_weights"])
        else:
            self.model.load_state_dict(local_var_dict)

        # local steps
        epoch_len = len(self.train_loader)
        self.log_info(fl_ctx, f"Local steps per epoch: {epoch_len}")

        # make a copy of model_global as reference for:
        # Prox loss for local model training - required by Ditto
        # Prox loss for reference model training - if combining with FedProx
        model_global = copy.deepcopy(self.model_ref)
        for param in model_global.parameters():
            param.requires_grad = False

        # local train
        self.local_train(
            fl_ctx=fl_ctx,
            train_loader=self.train_loader,
            model_global=model_global,
            abort_signal=abort_signal,
            val_freq=0,
        )
        if abort_signal.triggered:
            return make_reply(ReturnCode.TASK_ABORTED)
        self.epoch_of_start_time += self.aggregation_epochs

        # perform valid after local train
        acc = self.local_valid(self.valid_loader, abort_signal, tb_id="val_acc_local_model")
        if abort_signal.triggered:
            return make_reply(ReturnCode.TASK_ABORTED)
        self.log_info(fl_ctx, f"val_acc_local_model: {acc:.4f}")

        # save model
        self.save_model(is_best=False)
        if acc > self.best_acc:
            self.save_model(is_best=True)

        # compute delta model, global model has the primary key set
        # Ditto sends the reference model back to server
        local_weights = self.model_ref.state_dict()
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
