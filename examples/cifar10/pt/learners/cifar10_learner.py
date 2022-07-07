# Copyright (c) 2021-2022, NVIDIA CORPORATION.
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
import torch.optim as optim
from pt.networks.cifar10_nets import ModerateCNN
from pt.utils.cifar10_data_splitter import CIFAR10_ROOT
from pt.utils.cifar10_dataset import CIFAR10_Idx
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

from nvflare.apis.dxo import DXO, DataKind, MetaKey, from_shareable
from nvflare.apis.fl_constant import FLContextKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import ReservedHeaderKey, Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.abstract.learner_spec import Learner
from nvflare.app_common.abstract.model import ModelLearnableKey
from nvflare.app_common.app_constant import AppConstants, ModelName, ValidateType
from nvflare.app_common.pt.pt_fedproxloss import PTFedProxLoss


class CIFAR10Learner(Learner):  # also supports CIFAR10ScaffoldLearner
    def __init__(
        self,
        train_idx_root: str = "./dataset",
        aggregation_epochs: int = 1,
        train_task_name: str = AppConstants.TASK_TRAIN,
        submit_model_task_name: str = AppConstants.TASK_SUBMIT_MODEL,
        lr: float = 1e-2,
        fedproxloss_mu: float = 0.0,
        central: bool = False,
        analytic_sender_id: str = "analytic_sender",
        batch_size: int = 64,
        num_workers: int = 0,
    ):
        """Simple CIFAR-10 Trainer.

        Args:
            train_idx_root: directory with site training indices for CIFAR-10 data.
            aggregation_epochs: the number of training epochs for a round. Defaults to 1.
            train_task_name: name of the task to train the model.
            submit_model_task_name: name of the task to submit the best local model.
            lr: local learning rate. Float number. Defaults to 1e-2.
            fedproxloss_mu: weight for FedProx loss. Float number. Defaults to 0.0 (no FedProx).
            central: Bool. Whether to simulate central training. Default False.
            analytic_sender_id: id of `AnalyticsSender` if configured as a client component. If configured, TensorBoard events will be fired. Defaults to "analytic_sender".
            batch_size: batch size for training and validation.
            num_workers: number of workers for data loaders.

        Returns:
            a Shareable with the updated local model after running `execute()`
            or the best local model depending on the specified task.
        """
        super().__init__()
        # trainer init happens at the very beginning, only the basic info regarding the trainer is set here
        # the actual run has not started at this point
        self.train_idx_root = train_idx_root
        self.aggregation_epochs = aggregation_epochs
        self.train_task_name = train_task_name
        self.lr = lr
        self.fedproxloss_mu = fedproxloss_mu
        self.submit_model_task_name = submit_model_task_name
        self.best_acc = 0.0
        self.central = central
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.writer = None
        self.analytic_sender_id = analytic_sender_id

        # Epoch counter
        self.epoch_of_start_time = 0
        self.epoch_global = 0

        # following will be created in initialize() or later
        self.app_root = None
        self.client_id = None
        self.local_model_file = None
        self.best_local_model_file = None
        self.writer = None
        self.device = None
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.criterion_prox = None
        self.transform_train = None
        self.transform_valid = None
        self.train_dataset = None
        self.valid_dataset = None
        self.train_loader = None
        self.valid_loader = None

    def initialize(self, parts: dict, fl_ctx: FLContext):
        """
        Note: this code assumes a FL simulation setting
        Datasets will be initialized in train() and validate() when calling self._create_datasets()
        as we need to make sure that the server has already downloaded and split the data.
        """

        # when the run starts, this is where the actual settings get initialized for trainer

        # Set the paths according to fl_ctx
        self.app_root = fl_ctx.get_prop(FLContextKey.APP_ROOT)
        fl_args = fl_ctx.get_prop(FLContextKey.ARGS)
        self.client_id = fl_ctx.get_identity_name()
        self.log_info(
            fl_ctx,
            f"Client {self.client_id} initialized at \n {self.app_root} \n with args: {fl_args}",
        )

        self.local_model_file = os.path.join(self.app_root, "local_model.pt")
        self.best_local_model_file = os.path.join(self.app_root, "best_local_model.pt")

        # Select local TensorBoard writer or event-based writer for streaming
        self.writer = parts.get(self.analytic_sender_id)  # user configured config_fed_client.json for streaming
        if not self.writer:  # use local TensorBoard writer only
            self.writer = SummaryWriter(self.app_root)

        # set the training-related parameters
        # can be replaced by a config-style block
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = ModerateCNN().to(self.device)
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
        self.criterion = torch.nn.CrossEntropyLoss()
        if self.fedproxloss_mu > 0:
            self.log_info(fl_ctx, f"using FedProx loss with mu {self.fedproxloss_mu}")
            self.criterion_prox = PTFedProxLoss(mu=self.fedproxloss_mu)
        self.transform_train = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.ToPILImage(),
                transforms.Pad(4, padding_mode="reflect"),
                transforms.RandomCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                    std=[x / 255.0 for x in [63.0, 62.1, 66.7]],
                ),
            ]
        )
        self.transform_valid = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                    std=[x / 255.0 for x in [63.0, 62.1, 66.7]],
                ),
            ]
        )

    def _create_datasets(self, fl_ctx: FLContext):
        """To be called only after Cifar10DataSplitter downloaded the data and computed splits"""

        if self.train_dataset is None or self.train_loader is None:
            if not self.central:
                # Set datalist, here the path and filename are hard-coded, can also be fed as an argument
                site_idx_file_name = os.path.join(self.train_idx_root, self.client_id + ".npy")
                self.log_info(fl_ctx, f"IndexList Path: {site_idx_file_name}")
                if os.path.exists(site_idx_file_name):
                    self.log_info(fl_ctx, "Loading subset index")
                    site_idx = np.load(site_idx_file_name).tolist()  # TODO: get from fl_ctx/shareable?
                else:
                    self.system_panic(f"No subset index found! File {site_idx_file_name} does not exist!", fl_ctx)
                    return
                self.log_info(fl_ctx, f"Client subset size: {len(site_idx)}")
            else:
                site_idx = None  # use whole training dataset if self.central=True

            self.train_dataset = CIFAR10_Idx(
                root=CIFAR10_ROOT,
                data_idx=site_idx,
                train=True,
                download=False,
                transform=self.transform_train,
            )
            self.train_loader = torch.utils.data.DataLoader(
                self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers
            )

        if self.valid_dataset is None or self.valid_loader is None:
            self.valid_dataset = datasets.CIFAR10(
                root=CIFAR10_ROOT,
                train=False,
                download=False,
                transform=self.transform_valid,
            )
            self.valid_loader = torch.utils.data.DataLoader(
                self.valid_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers
            )

    def finalize(self, fl_ctx: FLContext):
        # collect threads, close files here
        pass

    def local_train(self, fl_ctx, train_loader, model_global, abort_signal: Signal, val_freq: int = 0):
        for epoch in range(self.aggregation_epochs):
            if abort_signal.triggered:
                return
            self.model.train()
            epoch_len = len(train_loader)
            self.epoch_global = self.epoch_of_start_time + epoch
            self.log_info(fl_ctx, f"Local epoch {self.client_id}: {epoch + 1}/{self.aggregation_epochs} (lr={self.lr})")
            avg_loss = 0.0
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
                current_step = epoch_len * self.epoch_global + i
                avg_loss += loss.item()
            self.writer.add_scalar("train_loss", avg_loss / len(train_loader), current_step)
            if val_freq > 0 and epoch % val_freq == 0:
                acc = self.local_valid(self.valid_loader, abort_signal, tb_id="val_acc_local_model", fl_ctx=fl_ctx)
                if acc > self.best_acc:
                    self.best_acc = acc
                    self.save_model(is_best=True)

    def save_model(self, is_best=False):
        # save model
        model_weights = self.model.state_dict()
        save_dict = {"model_weights": model_weights, "epoch": self.epoch_global}
        if is_best:
            save_dict.update({"best_acc": self.best_acc})
            torch.save(save_dict, self.best_local_model_file)
        else:
            torch.save(save_dict, self.local_model_file)

    def train(self, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        self._create_datasets(fl_ctx)

        # Check abort signal
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

        # local steps
        epoch_len = len(self.train_loader)
        self.log_info(fl_ctx, f"Local steps per epoch: {epoch_len}")

        # make a copy of model_global as reference for potential FedProx loss or SCAFFOLD
        model_global = copy.deepcopy(self.model)
        for param in model_global.parameters():
            param.requires_grad = False

        # local train
        self.local_train(
            fl_ctx=fl_ctx,
            train_loader=self.train_loader,
            model_global=model_global,
            abort_signal=abort_signal,
            val_freq=1 if self.central else 0,
        )
        if abort_signal.triggered:
            return make_reply(ReturnCode.TASK_ABORTED)
        self.epoch_of_start_time += self.aggregation_epochs

        # perform valid after local train
        acc = self.local_valid(self.valid_loader, abort_signal, tb_id="val_acc_local_model", fl_ctx=fl_ctx)
        if abort_signal.triggered:
            return make_reply(ReturnCode.TASK_ABORTED)
        self.log_info(fl_ctx, f"val_acc_local_model: {acc:.4f}")

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
            model_diff[name] = local_weights[name].cpu().numpy() - global_weights[name]
            if np.any(np.isnan(model_diff[name])):
                self.system_panic(f"{name} weights became NaN...", fl_ctx)
                return make_reply(ReturnCode.EXECUTION_EXCEPTION)

        # build the shareable
        dxo = DXO(data_kind=DataKind.WEIGHT_DIFF, data=model_diff)
        dxo.set_meta_prop(MetaKey.NUM_STEPS_CURRENT_ROUND, epoch_len)

        self.log_info(fl_ctx, "Local epochs finished. Returning shareable")
        return dxo.to_shareable()

    def get_model_for_validation(self, model_name: str, fl_ctx: FLContext) -> Shareable:
        # Retrieve the best local model saved during training.
        if model_name == ModelName.BEST_MODEL:
            model_data = None
            try:
                # load model to cpu as server might or might not have a GPU
                model_data = torch.load(self.best_local_model_file, map_location="cpu")
            except Exception as e:
                self.log_error(fl_ctx, f"Unable to load best model: {e}")

            # Create DXO and shareable from model data.
            if model_data:
                dxo = DXO(data_kind=DataKind.WEIGHTS, data=model_data["model_weights"])
                return dxo.to_shareable()
            else:
                # Set return code.
                self.log_error(fl_ctx, f"best local model not found at {self.best_local_model_file}.")
                return make_reply(ReturnCode.EXECUTION_RESULT_ERROR)
        else:
            raise ValueError(f"Unknown model_type: {model_name}")  # Raised errors are caught in LearnerExecutor class.

    def local_valid(self, valid_loader, abort_signal: Signal, tb_id=None, fl_ctx=None):
        self.model.eval()
        with torch.no_grad():
            correct, total = 0, 0
            for i, (inputs, labels) in enumerate(valid_loader):
                if abort_signal.triggered:
                    return None
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, pred_label = torch.max(outputs.data, 1)

                total += inputs.data.size()[0]
                correct += (pred_label == labels.data).sum().item()
            metric = correct / float(total)
            if tb_id:
                self.writer.add_scalar(tb_id, metric, self.epoch_global)
        return metric

    def validate(self, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        self._create_datasets(fl_ctx)

        # Check abort signal
        if abort_signal.triggered:
            return make_reply(ReturnCode.TASK_ABORTED)

        # get validation information
        self.log_info(fl_ctx, f"Client identity: {fl_ctx.get_identity_name()}")
        model_owner = shareable.get(ReservedHeaderKey.HEADERS).get(AppConstants.MODEL_OWNER)
        if model_owner:
            self.log_info(fl_ctx, f"Evaluating model from {model_owner} on {fl_ctx.get_identity_name()}")
        else:
            model_owner = "global_model"  # evaluating global model during training

        # update local model weights with received weights
        dxo = from_shareable(shareable)
        global_weights = dxo.data

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
                except Exception as e:
                    raise ValueError("Convert weight from {} failed with error: {}".format(var_name, str(e)))
        self.model.load_state_dict(local_var_dict)
        if n_loaded == 0:
            raise ValueError(f"No weights loaded for validation! Received weight dict is {global_weights}")

        validate_type = shareable.get_header(AppConstants.VALIDATE_TYPE)
        if validate_type == ValidateType.BEFORE_TRAIN_VALIDATE:
            # perform valid before local train
            global_acc = self.local_valid(self.valid_loader, abort_signal, tb_id="val_acc_global_model", fl_ctx=fl_ctx)
            if abort_signal.triggered:
                return make_reply(ReturnCode.TASK_ABORTED)
            self.log_info(fl_ctx, f"val_acc_global_model ({model_owner}): {global_acc}")

            return DXO(data_kind=DataKind.METRICS, data={MetaKey.INITIAL_METRICS: global_acc}, meta={}).to_shareable()

        elif validate_type == ValidateType.MODEL_VALIDATE:
            # perform valid
            train_acc = self.local_valid(self.train_loader, abort_signal)
            if abort_signal.triggered:
                return make_reply(ReturnCode.TASK_ABORTED)
            self.log_info(fl_ctx, f"training acc ({model_owner}): {train_acc}")

            val_acc = self.local_valid(self.valid_loader, abort_signal)
            if abort_signal.triggered:
                return make_reply(ReturnCode.TASK_ABORTED)
            self.log_info(fl_ctx, f"validation acc ({model_owner}): {val_acc}")

            self.log_info(fl_ctx, "Evaluation finished. Returning shareable")

            val_results = {"train_accuracy": train_acc, "val_accuracy": val_acc}

            metric_dxo = DXO(data_kind=DataKind.METRICS, data=val_results)
            return metric_dxo.to_shareable()

        else:
            return make_reply(ReturnCode.VALIDATE_TYPE_UNKNOWN)
