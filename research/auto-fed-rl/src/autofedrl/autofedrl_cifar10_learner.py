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
import os

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from typing import Union

from pt.learners.cifar10_learner import CIFAR10Learner
from pt.utils.cifar10_data_utils import CIFAR10_ROOT
from pt.utils.cifar10_dataset import CIFAR10_Idx
from torchvision import datasets

from nvflare.apis.dxo import DXO, DataKind, MetaKey
from nvflare.apis.fl_constant import ReturnCode
from nvflare.app_common.app_constant import AppConstants, ValidateType
from nvflare.app_opt.pt.decomposers import TensorDecomposer
from nvflare.fuel.utils import fobs

from .autofedrl_constants import AutoFedRLConstants


class CIFAR10AutoFedRLearner(CIFAR10Learner):  # TODO: also support CIFAR10ScaffoldLearner
    def __init__(
        self,
        train_idx_root: str = "./dataset",
        aggregation_epochs: int = 1,  # TODO: Is this still being used?
        lr: float = 1e-2,
        fedproxloss_mu: float = 0.0,
        central: bool = False,
        analytic_sender_id: str = "analytic_sender",
        batch_size: int = 64,
        num_workers: int = 0,
        entropy_coeff: float = 1.0,
        entropy_threshold: float = 2.0,
    ):
        """Simple CIFAR-10 Trainer utilizing Auto-FedRL.

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
            entropy_coeff:  entropy cut-off.
            entropy_threshold:  entropy threshold.

        Returns:
            a Shareable with the updated local model after running `train()`,
            or validation metrics after calling `validate()`,
            or the best local model when calling `get_model_for_validation()`
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
        self.entropy_coeff = entropy_coeff
        self.entropy_threshold = entropy_threshold

        self.current_round = 0
        self.best_global_acc = 0

        # Use FOBS serializing/deserializing PyTorch tensors
        fobs.register(TensorDecomposer)

    def initialize(self):
        # Initialize super class
        CIFAR10Learner.initialize(self)
        # Enabling the Nesterov momentum can stabilize the training.
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9, weight_decay=0.0, nesterov=True)

    def _create_datasets(self):
        """To be called only after Cifar10DataSplitter downloaded the data and computed splits"""

        if self.train_dataset is None or self.train_loader is None:

            if not self.central:
                # Set datalist, here the path and filename are hard-coded, can also be fed as an argument
                site_idx_file_name = os.path.join(self.train_idx_root, self.client_id + ".npy")
                self.info(f"IndexList Path: {site_idx_file_name}")
                if os.path.exists(site_idx_file_name):
                    self.info("Loading subset index")
                    site_idx = np.load(site_idx_file_name).tolist()  # TODO: get from fl_ctx/shareable?
                else:
                    self.stop_task(f"No subset index found! File {site_idx_file_name} does not exist!")
                    return
                self.info(f"Client subset size: {len(site_idx)}")
            else:
                site_idx = None  # use whole training dataset if self.central=True

            self.debug(f"site_idx: {site_idx}")

            # Train set
            n_img_for_search = self.batch_size * 10
            self.train_dataset = CIFAR10_Idx(
                root=CIFAR10_ROOT,
                data_idx=site_idx[:],
                train=True,
                download=False,
                transform=self.transform_train,
            )
            self.train_loader = torch.utils.data.DataLoader(
                self.train_dataset, batch_size=self.batch_size, shuffle=True
            )
            # Val set for search
            self.val_dataset_for_search = CIFAR10_Idx(
                root=CIFAR10_ROOT,
                data_idx=site_idx[-n_img_for_search:],
                train=True,
                download=False,
                transform=self.transform_valid,
            )
            self.val_loader_search = torch.utils.data.DataLoader(
                self.val_dataset_for_search, batch_size=self.batch_size, shuffle=False
            )
            self.info(
                f"Split ({n_img_for_search}) images from {len(site_idx)} training images for Hyerparamters Search",
            )

        if self.valid_dataset is None or self.valid_loader is None:
            self.valid_dataset = datasets.CIFAR10(
                root=CIFAR10_ROOT,
                train=False,
                download=False,
                transform=self.transform_valid,
            )
            self.valid_loader = torch.utils.data.DataLoader(
                self.valid_dataset, batch_size=self.batch_size, shuffle=False
            )

    def local_train(self, train_loader, model_global, val_freq: int = 0):
        for epoch in range(self.aggregation_epochs):
            if self.is_aborted():
                return
            self.model.train()
            epoch_len = len(train_loader)
            self.epoch_global = self.epoch_of_start_time + epoch
            self.info(f"Local epoch {self.client_id}: {epoch + 1}/{self.aggregation_epochs} (lr={self.lr})")
            avg_loss = 0.0
            for i, (inputs, labels) in enumerate(train_loader):
                if self.is_aborted():
                    return
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                # Forward
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                # FedProx loss term
                if self.fedproxloss_mu > 0:
                    fed_prox_loss = self.criterion_prox(self.model, model_global)
                    loss += fed_prox_loss

                # entropy_cost
                if self.entropy_coeff > 0:
                    probs_output = torch.exp(outputs) / (torch.exp(outputs).sum(1).view(-1, 1))
                    entropy = -(probs_output * torch.log(probs_output)).sum(1).mean()
                    entropy_cost = self.entropy_coeff * F.relu(self.entropy_threshold - entropy)
                    loss += entropy_cost

                # Zero the parameter gradients
                self.optimizer.zero_grad()
                # Backward + Optimize
                loss.backward()
                self.optimizer.step()
                current_step = epoch_len * self.epoch_global + i
                avg_loss += loss.item()
            self.writer.add_scalar("train_loss", avg_loss / len(train_loader), current_step)
            if val_freq > 0 and epoch % val_freq == 0:
                acc = self.local_valid(self.valid_loader, tb_id="val_acc_local_model")
                if acc > self.best_acc:
                    self.best_acc = acc
                    self.save_model(is_best=True)

    def train(self, dxo: DXO) -> Union[str, DXO]:
        self._create_datasets()

        # Check abort signal
        if self.is_aborted():
            return ReturnCode.TASK_ABORTED

        # Get round information
        self.info(f"Current/Total Round: {self.current_round + 1}/{self.total_rounds}")
        self.info(f"Client identity: {self.site_name}")

        # Get lr and ne from server
        current_lr, current_ne = None, None
        hps = fobs.loads(self.get_shareable_header(AutoFedRLConstants.HYPERPARAMTER_COLLECTION))
        if hps is not None:
            current_lr = hps.get("lr")
            self.lr = current_lr
            current_ne = hps.get("ne")
        if current_lr is not None:
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = current_lr
            self.info(f"Received and override current learning rate as: {current_lr}")
        if current_ne is not None:
            self.aggregation_epochs = current_ne
            self.info(f"Received and override current number of local epochs: {current_ne}")

        # Update local model weights with received weights
        global_weights = dxo.data

        # Before loading weights, tensors might need to be reshaped to support HE for secure aggregation.
        local_var_dict = self.model.state_dict()
        model_keys = global_weights.keys()
        for var_name in local_var_dict:
            if var_name in model_keys:
                weights = global_weights[var_name]
                try:
                    # Reshape global weights to compute difference later on
                    global_weights[var_name] = np.reshape(weights, local_var_dict[var_name].shape)
                    # Update the local dict
                    local_var_dict[var_name] = torch.as_tensor(global_weights[var_name])
                except Exception as e:
                    raise ValueError(f"Convert weight from {var_name} failed!") from e
        self.model.load_state_dict(local_var_dict)

        # Local steps
        epoch_len = len(self.train_loader)
        self.info(f"Local steps per epoch: {epoch_len}")

        # Make a copy of model_global as reference for potential FedProx loss or SCAFFOLD
        model_global = copy.deepcopy(self.model)
        for param in model_global.parameters():
            param.requires_grad = False

        # Local train
        self.local_train(
            train_loader=self.train_loader,
            model_global=model_global,
            val_freq=1 if self.central else 0,
        )
        if self.is_aborted():
            return ReturnCode.TASK_ABORTED

        self.epoch_of_start_time += self.aggregation_epochs

        # Perform valid after local train
        acc = self.local_valid(self.valid_loader, tb_id="val_acc_local_model")
        if self.is_aborted():
            return ReturnCode.TASK_ABORTED
        self.info(f"val_acc_local_model: {acc:.4f}")

        # Save model
        self.save_model(is_best=False)
        if acc > self.best_acc:
            self.best_acc = acc
            self.save_model(is_best=True)

        # Compute delta model, global model has the primary key set
        local_weights = self.model.state_dict()
        model_diff = {}
        for name in global_weights:
            if name not in local_weights:
                continue
            model_diff[name] = np.subtract(local_weights[name].cpu().numpy(), global_weights[name], dtype=np.float32)
            if np.any(np.isnan(model_diff[name])):
                self.stop_task(f"{name} weights became NaN...")
                return ReturnCode.EXECUTION_EXCEPTION

        # Build the shareable
        dxo = DXO(data_kind=DataKind.WEIGHT_DIFF, data=model_diff)
        if hps.get("aw") is not None:
            # When search aggregation weights, we have to override it
            # to 1, since we will manually assign weights to aggregator.
            # Search space will discover which client is more informative.
            # It might not be related to the number of data in a client.
            dxo.set_meta_prop(MetaKey.NUM_STEPS_CURRENT_ROUND, 1)
        else:
            dxo.set_meta_prop(MetaKey.NUM_STEPS_CURRENT_ROUND, epoch_len)

        self.info("Local epochs finished. Returning shareable")
        return dxo

    def local_valid(self, valid_loader, tb_id=None, get_loss=False):
        self.model.eval()
        with torch.no_grad():
            correct, total = 0, 0
            for _, (inputs, labels) in enumerate(valid_loader):
                if self.is_aborted():
                    return None
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                _, pred_label = torch.max(outputs.data, 1)

                if get_loss:
                    # Return val loss instead of accuracy over the number batches
                    total += inputs.data.size()[0]
                    correct += loss.item()
                else:
                    total += inputs.data.size()[0]
                    correct += (pred_label == labels.data).sum().item()
            metric = correct / float(total)
            if get_loss:
                self.info(f"HP Search loss: {metric} of {total} batches on {self.site_name}")
            if tb_id:
                self.writer.add_scalar(tb_id, metric, self.current_round)
        return metric

    def validate(self, dxo: DXO) -> Union[str, DXO]:
        self._create_datasets()

        # Check abort signal
        if self.is_aborted():
            return ReturnCode.TASK_ABORTED

        # Get validation information
        self.info(f"Client identity: {self.site_name}")

        # Update local model weights with received weights
        global_weights = dxo.data

        # Before loading weights, tensors might need to be reshaped to support HE for secure aggregation.
        local_var_dict = self.model.state_dict()
        model_keys = global_weights.keys()
        n_loaded = 0
        for var_name in local_var_dict:
            if var_name in model_keys:
                weights = torch.as_tensor(global_weights[var_name], device=self.device)
                try:
                    # Update the local dict
                    local_var_dict[var_name] = torch.as_tensor(torch.reshape(weights, local_var_dict[var_name].shape))
                    n_loaded += 1
                except Exception as e:
                    raise ValueError(f"Convert weight from {var_name} failed!") from e
        self.model.load_state_dict(local_var_dict)
        if n_loaded == 0:
            raise ValueError(f"No weights loaded for validation! Received weight dict is {global_weights}")

        validate_type = self.get_shareable_header(AppConstants.VALIDATE_TYPE)
        model_owner = self.get_shareable_header(AppConstants.MODEL_OWNER)
        if validate_type == ValidateType.BEFORE_TRAIN_VALIDATE:
            # Perform valid before local train
            global_acc = self.local_valid(self.valid_loader, tb_id="val_acc_global_model")
            if self.is_aborted():
                return ReturnCode.TASK_ABORTED
            self.info(f"val_acc_global_model ({model_owner}): {global_acc}")

            if global_acc > self.best_global_acc:
                self.best_global_acc = global_acc
            # Log the best global model_accuracy
            self.writer.add_scalar("best_val_acc_global_model", self.best_global_acc, self.current_round)

            return DXO(data_kind=DataKind.METRICS, data={MetaKey.INITIAL_METRICS: global_acc}, meta={})

        elif validate_type == ValidateType.MODEL_VALIDATE:
            # Perform valid
            train_acc = self.local_valid(self.train_loader)
            if self.is_aborted():
                return ReturnCode.TASK_ABORTED
            self.info(f"training acc ({model_owner}): {train_acc}")

            val_acc = self.local_valid(self.valid_loader)
            if self.is_aborted():
                return ReturnCode.TASK_ABORTED
            self.info(f"validation acc ({model_owner}): {val_acc}")

            self.info("Evaluation finished. Returning shareable")

            val_results = {"train_accuracy": train_acc, "val_accuracy": val_acc}

            return DXO(data_kind=DataKind.METRICS, data=val_results)

        elif validate_type == AutoFedRLConstants.MODEL_VALIDATE_FOR_SEARCH:
            self.info(f"Evaluating model from {model_owner} on {self.site_name} for HP Search")

            val_loss_hp = self.local_valid(self.val_loader_search, get_loss=True)
            if self.is_aborted():
                return ReturnCode.TASK_ABORTED

            val_results = {"val_loss": val_loss_hp}

            return DXO(data_kind=DataKind.METRICS, data=val_results)
        else:
            return ReturnCode.VALIDATE_TYPE_UNKNOWN
