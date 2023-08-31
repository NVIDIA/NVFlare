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
from typing import Union

import numpy as np
import torch
import torch.optim as optim
from pt.cifar10_data_utils import CIFAR10_ROOT, CIFAR10_Idx
from pt.cifar10_net import Net
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets
from torchvision.transforms import Compose, Normalize, ToTensor

from nvflare.apis.fl_constant import FLMetaKey, ReturnCode
from nvflare.app_common.abstract.fl_model import FLModel, ParamsType
from nvflare.app_common.abstract.model_learner import ModelLearner
from nvflare.app_common.app_constant import AppConstants, ModelName, ValidateType
from nvflare.app_common.utils.fl_model_utils import FLModelUtils


class CIFAR10ModelLearner(ModelLearner):
    def __init__(
        self,
        train_idx_root: str = "./dataset",
        aggregation_epochs: int = 1,
        lr: float = 1e-2,
        analytic_sender_id: str = "analytic_sender",
        batch_size: int = 64,
        num_workers: int = 0,
    ):
        """Simple CIFAR-10 Trainer.

        Args:
            train_idx_root: directory with site training indices for CIFAR-10 data.
            aggregation_epochs: the number of training epochs for a round. Defaults to 1.
            lr: local learning rate. Float number. Defaults to 1e-2.
            analytic_sender_id: id of `AnalyticsSender` if configured as a client component.
                If configured, TensorBoard events will be fired. Defaults to "analytic_sender".
            batch_size: batch size for training and validation.
            num_workers: number of workers for data loaders.

        Returns:
            an FLModel with the updated local model differences after running `train()`, the metrics after `validate()`,
            or the best local model depending on the specified task.
        """
        super().__init__()
        self.train_idx_root = train_idx_root
        self.aggregation_epochs = aggregation_epochs
        self.lr = lr
        self.best_acc = 0.0
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.analytic_sender_id = analytic_sender_id

        self.epoch_of_start_time = 0
        self.epoch_global = 0

        self.local_model_file = None
        self.best_local_model_file = None
        self.writer = None
        self.device = None
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.train_dataset = None
        self.valid_dataset = None
        self.train_loader = None
        self.valid_loader = None

    def initialize(self):
        self.info(
            f"Client {self.site_name} initialized at \n {self.app_root} \n with args: {self.args}",
        )

        self.local_model_file = os.path.join(self.app_root, "local_model.pt")
        self.best_local_model_file = os.path.join(self.app_root, "best_local_model.pt")

        # Select local TensorBoard writer or event-based writer for streaming
        self.writer = self.get_component(
            self.analytic_sender_id
        )  # user configured config_fed_client.json for streaming
        if not self.writer:  # use local TensorBoard writer only
            self.writer = SummaryWriter(self.app_root)

        # set the training-related parameters
        # can be replaced by a config-style block
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = Net().to(self.device)
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
        self.criterion = torch.nn.CrossEntropyLoss()

        transforms = Compose(
            [
                ToTensor(),
                Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        # Set datalist, here the path and filename are hard-coded, can also be fed as an argument
        site_idx_file_name = os.path.join(self.train_idx_root, self.site_name + ".npy")
        self.info(f"IndexList Path: {site_idx_file_name}")
        if os.path.exists(site_idx_file_name):
            self.info("Loading subset index")
            site_idx = np.load(site_idx_file_name).tolist()  # TODO: get from server?
        else:
            self.stop_task(f"No subset index found! File {site_idx_file_name} does not exist!")
            return
        self.info(f"Client subset size: {len(site_idx)}")

        self.train_dataset = CIFAR10_Idx(
            root=CIFAR10_ROOT,
            data_idx=site_idx,
            train=True,
            download=False,
            transform=transforms,
        )
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers
        )

        self.valid_dataset = datasets.CIFAR10(
            root=CIFAR10_ROOT,
            train=False,
            download=False,
            transform=transforms,
        )
        self.valid_loader = torch.utils.data.DataLoader(
            self.valid_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers
        )

    def train(self, model: FLModel) -> Union[str, FLModel]:
        # get round information
        self.info(f"Current/Total Round: {self.current_round + 1}/{self.total_rounds}")
        self.info(f"Client identity: {self.site_name}")

        # update local model weights with received weights
        global_weights = model.params

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
                except BaseException as e:
                    raise ValueError(f"Convert weight from {var_name} failed") from e
        self.model.load_state_dict(local_var_dict)

        # local steps
        epoch_len = len(self.train_loader)
        self.info(f"Local steps per epoch: {epoch_len}")

        # make a copy of model_global as reference for potential FedProx loss or SCAFFOLD
        model_global = copy.deepcopy(self.model)
        for param in model_global.parameters():
            param.requires_grad = False

        # local train
        self.local_train(
            train_loader=self.train_loader,
            model_global=model_global,
            val_freq=0,
        )
        self.epoch_of_start_time += self.aggregation_epochs

        # perform valid after local train
        acc = self.local_valid(self.valid_loader, tb_id="val_acc_local_model")
        self.info(f"val_acc_local_model: {acc:.4f}")

        # save model
        self.save_model(is_best=False)
        if acc > self.best_acc:
            self.best_acc = acc
            self.save_model(is_best=True)

        # compute delta model, global model has the primary key set
        local_weights = self.model.state_dict()
        model_diff = {}
        for name in global_weights:
            if name not in local_weights:
                continue
            model_diff[name] = np.subtract(local_weights[name].cpu().numpy(), global_weights[name], dtype=np.float32)
            if np.any(np.isnan(model_diff[name])):
                self.stop_task(f"{name} weights became NaN...")
                return ReturnCode.EXECUTION_EXCEPTION

        # return an FLModel containing the model differences
        fl_model = FLModel(params_type=ParamsType.DIFF, params=model_diff)

        FLModelUtils.set_meta_prop(fl_model, FLMetaKey.NUM_STEPS_CURRENT_ROUND, epoch_len)
        self.info("Local epochs finished. Returning FLModel")
        return fl_model

    def local_train(self, train_loader, model_global, val_freq: int = 0):
        for epoch in range(self.aggregation_epochs):
            self.model.train()
            epoch_len = len(train_loader)
            self.epoch_global = self.epoch_of_start_time + epoch
            self.info(f"Local epoch {self.site_name}: {epoch + 1}/{self.aggregation_epochs} (lr={self.lr})")
            avg_loss = 0.0
            running_loss = 0.0
            for i, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                # zero the parameter gradients
                self.optimizer.zero_grad()
                # forward + backward + optimize
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                loss.backward()
                self.optimizer.step()
                current_step = epoch_len * self.epoch_global + i
                avg_loss += loss.item()
                running_loss += loss.cpu().detach().numpy() / inputs.size()[0]

                if i % 100 == 0:
                    self.info(f"Epoch: {epoch}/{self.aggregation_epochs}, Iteration: {i}, " f"Loss: {running_loss/100}")
                    running_loss = 0.0

            self.writer.add_scalar("train_loss", avg_loss / len(train_loader), current_step)
            if val_freq > 0 and epoch % val_freq == 0:
                acc = self.local_valid(self.valid_loader, tb_id="val_acc_local_model")
                if acc > self.best_acc:
                    self.best_acc = acc
                    self.save_model(is_best=True)

    def validate(self, model: FLModel) -> Union[str, FLModel]:
        # get validation information
        self.info(f"Client identity: {self.site_name}")

        # update local model weights with received weights
        global_weights = model.params

        # Before loading weights, tensors might need to be reshaped to support HE for secure aggregation.
        local_var_dict = self.model.state_dict()
        model_keys = global_weights.keys()
        n_loaded = 0
        for var_name in local_var_dict:
            if var_name in model_keys:
                weights = torch.as_tensor(global_weights[var_name], device=self.device)
                try:
                    # update the local dict
                    local_var_dict[var_name] = torch.as_tensor(torch.reshape(weights, local_var_dict[var_name].shape))
                    n_loaded += 1
                except BaseException as e:
                    raise ValueError(f"Convert weight from {var_name} failed") from e
        self.model.load_state_dict(local_var_dict)
        if n_loaded == 0:
            raise ValueError(f"No weights loaded for validation! Received weight dict is {1}")

        # get validation meta info
        validate_type = FLModelUtils.get_meta_prop(model, FLMetaKey.VALIDATE_TYPE, ValidateType.MODEL_VALIDATE)
        model_owner = self.get_shareable_header(AppConstants.MODEL_OWNER)

        # perform valid
        train_acc = self.local_valid(
            self.train_loader,
            tb_id="train_acc_global_model" if validate_type == ValidateType.BEFORE_TRAIN_VALIDATE else None,
        )
        self.info(f"training acc ({model_owner}): {train_acc:.4f}")

        val_acc = self.local_valid(
            self.valid_loader,
            tb_id="val_acc_global_model" if validate_type == ValidateType.BEFORE_TRAIN_VALIDATE else None,
        )
        self.info(f"validation acc ({model_owner}): {val_acc:.4f}")
        self.info("Evaluation finished. Returning result")

        if val_acc > self.best_acc:
            self.best_acc = val_acc
            self.save_model(is_best=True)

        val_results = {"train_accuracy": train_acc, "val_accuracy": val_acc}
        return FLModel(metrics=val_results)

    def local_valid(self, valid_loader, tb_id=None):
        self.model.eval()
        with torch.no_grad():
            correct, total = 0, 0
            for _i, (inputs, labels) in enumerate(valid_loader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, pred_label = torch.max(outputs.data, 1)

                total += inputs.data.size()[0]
                correct += (pred_label == labels.data).sum().item()
            metric = correct / float(total)
            if tb_id:
                self.writer.add_scalar(tb_id, metric, self.epoch_global)
        return metric

    def save_model(self, is_best=False):
        # save model
        model_weights = self.model.state_dict()
        save_dict = {"model_weights": model_weights, "epoch": self.epoch_global}
        if is_best:
            save_dict.update({"best_acc": self.best_acc})
            torch.save(save_dict, self.best_local_model_file)
        else:
            torch.save(save_dict, self.local_model_file)

    def get_model(self, model_name: str) -> Union[str, FLModel]:
        # Retrieve the best local model saved during training.
        if model_name == ModelName.BEST_MODEL:
            try:
                # load model to cpu as server might or might not have a GPU
                model_data = torch.load(self.best_local_model_file, map_location="cpu")
            except Exception as e:
                raise ValueError("Unable to load best model") from e

            # Create FLModel from model data.
            if model_data:
                # convert weights to numpy to support FOBS
                model_weights = model_data["model_weights"]
                for k, v in model_weights.items():
                    model_weights[k] = v.numpy()
                return FLModel(params_type=ParamsType.FULL, params=model_weights)
            else:
                # Set return code.
                self.error(f"best local model not found at {self.best_local_model_file}.")
                return ReturnCode.EXECUTION_RESULT_ERROR
        else:
            raise ValueError(f"Unknown model_type: {model_name}")  # Raised errors are caught in LearnerExecutor class.
