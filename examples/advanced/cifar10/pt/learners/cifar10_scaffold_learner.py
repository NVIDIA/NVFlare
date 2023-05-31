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

import copy

import torch
from pt.learners.cifar10_learner import CIFAR10Learner

from nvflare.apis.dxo import DXO, DataKind, MetaKey, from_shareable
from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.app_constant import AlgorithmConstants, AppConstants
from nvflare.app_opt.pt.scaffold import PTScaffoldHelper, get_lr_values


class CIFAR10ScaffoldLearner(CIFAR10Learner):
    def __init__(
        self,
        train_idx_root: str = "./dataset",
        aggregation_epochs: int = 1,
        lr: float = 1e-2,
        fedproxloss_mu: float = 0.0,
        central: bool = False,
        analytic_sender_id: str = "analytic_sender",
        batch_size: int = 64,
        num_workers: int = 0,
    ):
        """Simple Scaffold CIFAR-10 Trainer.
        Implements the training algorithm proposed in
        Karimireddy et al. "SCAFFOLD: Stochastic Controlled Averaging for Federated Learning"
        (https://arxiv.org/abs/1910.06378) using functions implemented in `PTScaffoldHelper` class.

        Args:
            train_idx_root: directory with site training indices for CIFAR-10 data.
            aggregation_epochs: the number of training epochs for a round. Defaults to 1.
            lr: local learning rate. Float number. Defaults to 1e-2.
            fedproxloss_mu: weight for FedProx loss. Float number. Defaults to 0.0 (no FedProx).
            central: Bool. Whether to simulate central training. Default False.
            analytic_sender_id: id of `AnalyticsSender` if configured as a client component.
                If configured, TensorBoard events will be fired. Defaults to "analytic_sender".
            batch_size: batch size for training and validation.
            num_workers: number of workers for data loaders.

        Returns:
            a Shareable with the updated local model after running `execute()`
            or the best local model depending on the specified task.
        """

        CIFAR10Learner.__init__(
            self,
            train_idx_root=train_idx_root,
            aggregation_epochs=aggregation_epochs,
            lr=lr,
            fedproxloss_mu=fedproxloss_mu,
            central=central,
            analytic_sender_id=analytic_sender_id,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        self.scaffold_helper = PTScaffoldHelper()

    def initialize(self, parts: dict, fl_ctx: FLContext):
        # Initialize super class and SCAFFOLD
        CIFAR10Learner.initialize(self, parts=parts, fl_ctx=fl_ctx)
        self.scaffold_helper.init(model=self.model)

    def local_train(self, fl_ctx, train_loader, model_global, abort_signal: Signal, val_freq: int = 0):
        # local_train with SCAFFOLD steps
        c_global_para, c_local_para = self.scaffold_helper.get_params()
        for epoch in range(self.aggregation_epochs):
            if abort_signal.triggered:
                return
            self.model.train()
            epoch_len = len(train_loader)
            self.epoch_global = self.epoch_of_start_time + epoch
            self.log_info(fl_ctx, f"Local epoch {self.client_id}: {epoch + 1}/{self.aggregation_epochs} (lr={self.lr})")

            for i, (inputs, labels) in enumerate(train_loader):
                if abort_signal.triggered:
                    return
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                # zero the parameter gradients
                self.optimizer.zero_grad()
                # forward + backward + optimize
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                # FedProx loss term
                if self.fedproxloss_mu > 0:
                    fed_prox_loss = self.criterion_prox(self.model, model_global)
                    loss += fed_prox_loss

                loss.backward()
                self.optimizer.step()

                # SCAFFOLD step
                curr_lr = get_lr_values(self.optimizer)[0]
                self.scaffold_helper.model_update(
                    model=self.model, curr_lr=curr_lr, c_global_para=c_global_para, c_local_para=c_local_para
                )

                current_step = epoch_len * self.epoch_global + i
                self.writer.add_scalar("train_loss", loss.item(), current_step)

            if val_freq > 0 and epoch % val_freq == 0:
                acc = self.local_valid(self.valid_loader, abort_signal, tb_id="val_acc_local_model", fl_ctx=fl_ctx)
                if acc > self.best_acc:
                    self.save_model(is_best=True)

        # Update the SCAFFOLD terms
        self.scaffold_helper.terms_update(
            model=self.model,
            curr_lr=curr_lr,
            c_global_para=c_global_para,
            c_local_para=c_local_para,
            model_global=model_global,
        )

    def train(self, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        # return DXO with extra control differences for SCAFFOLD
        dxo_collection = from_shareable(shareable)
        if dxo_collection.data_kind != DataKind.COLLECTION:
            self.log_error(
                fl_ctx,
                f"SCAFFOLD learner expected shareable to contain a collection of two DXOs "
                f"but got data kind {dxo_collection.data_kind}.",
            )
            return make_reply(ReturnCode.ERROR)
        dxo_global_weights = dxo_collection.data.get(AppConstants.MODEL_WEIGHTS)
        dxo_global_ctrl_weights = dxo_collection.data.get(AlgorithmConstants.SCAFFOLD_CTRL_GLOBAL)
        if dxo_global_ctrl_weights is None:
            self.log_error(fl_ctx, "DXO collection doesn't contain the SCAFFOLD controls!")
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)
        # convert to tensor and load into c_global model
        global_ctrl_weights = dxo_global_ctrl_weights.data
        for k in global_ctrl_weights.keys():
            global_ctrl_weights[k] = torch.as_tensor(global_ctrl_weights[k])
        self.scaffold_helper.load_global_controls(weights=global_ctrl_weights)

        # modify shareable to only contain global weights
        shareable = dxo_global_weights.update_shareable(shareable)  # TODO: add set_dxo() method to Shareable

        # local training
        result_shareable = super().train(shareable, fl_ctx, abort_signal)
        if result_shareable.get_return_code() == ReturnCode.OK:
            # get DXO with weight updates from local training
            dxo_weights_diff = from_shareable(result_shareable)

            # Create a DXO collection with weights and scaffold controls
            dxo_weigths_diff_ctrl = DXO(data_kind=DataKind.WEIGHT_DIFF, data=self.scaffold_helper.get_delta_controls())
            # add same num steps as for model weights
            dxo_weigths_diff_ctrl.set_meta_prop(
                MetaKey.NUM_STEPS_CURRENT_ROUND, dxo_weights_diff.get_meta_prop(MetaKey.NUM_STEPS_CURRENT_ROUND)
            )

            collection_data = {
                AppConstants.MODEL_WEIGHTS: dxo_weights_diff,
                AlgorithmConstants.SCAFFOLD_CTRL_DIFF: dxo_weigths_diff_ctrl,
            }
            dxo = DXO(data_kind=DataKind.COLLECTION, data=collection_data)

            return dxo.to_shareable()
        else:
            return result_shareable

    def validate(self, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        dxo = from_shareable(shareable)

        # If collection, extract only global weights to validate
        if dxo.data_kind == DataKind.COLLECTION:
            # create a new shareable with only model weights
            shareable = copy.deepcopy(shareable)  # TODO: Is this the best way?
            dxo_global_weights = dxo.data.get(AppConstants.MODEL_WEIGHTS)
            shareable = dxo_global_weights.update_shareable(shareable)  # TODO: add set_dxo() method to Shareable

        return super().validate(shareable=shareable, fl_ctx=fl_ctx, abort_signal=abort_signal)
