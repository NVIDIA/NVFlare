# Copyright (c) 2021, NVIDIA CORPORATION.
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
import json
import os

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from nvflare.apis.dxo import DXO, DataKind, MetaKey, from_shareable
from nvflare.apis.event_type import EventType
from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import FLContextKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.pt.pt_fedproxloss import PTFedProxLoss


class SupervisedTrainer(Executor):
    def __init__(
        self,
        train_config_filename,
        aggregation_epochs: int = 1,
        train_task_name: str = AppConstants.TASK_TRAIN,
        submit_model_task_name=AppConstants.TASK_SUBMIT_MODEL,
    ):
        """Simple Supervised Trainer.

        Args:
            train_config_filename: directory of config file.
            aggregation_epochs: the number of training epochs for a round.
                This parameter only works when `aggregation_iters` is 0. Defaults to 1.
            train_task_name: name of the task to train the model.
            submit_model_task_name: name of the task to submit the best local model.

        Returns:
            a Shareable with the updated local model after running `execute()`
            or the best local model depending on the specified task.
        """
        super().__init__()
        # trainer init happens at the very beginning, only the basic info regarding the trainer is set here
        # the actual run has not started at this point
        self.train_config_filename = train_config_filename
        self.aggregation_epochs = aggregation_epochs
        self.train_task_name = train_task_name
        self.submit_model_task_name = submit_model_task_name
        self.best_acc = 0.0

        # Epoch counter
        self.epoch_of_start_time = 0
        self.epoch_global = 0

        # FedProx related
        self.fedproxloss_mu = 0.0
        self.criterion_prox = None

    def _initialize_trainer(self, fl_ctx: FLContext):
        # when the run starts, this is where the actual settings get initialized for trainer

        # Set the paths according to fl_ctx
        engine = fl_ctx.get_engine()
        ws = engine.get_workspace()
        app_config_dir = ws.get_app_config_dir(fl_ctx.get_run_number())
        app_dir = ws.get_app_dir(fl_ctx.get_run_number())

        self.local_model_file = os.path.join(app_dir, "local_model.pt")
        self.best_local_model_file = os.path.join(app_dir, "best_local_model.pt")
        train_config_file_path = os.path.join(app_config_dir, self.train_config_filename)

        # get and print the args
        fl_args = fl_ctx.get_prop(FLContextKey.ARGS)
        self.client_id = fl_ctx.get_identity_name()
        self.log_info(
            fl_ctx,
            f"Client {self.client_id} initialized with args: \n {fl_args}",
        )

        # Set local tensorboard writer - to be replaced by event
        self.writer = SummaryWriter(app_dir)
        # Set the training-related contexts
        self.train_config(fl_ctx, train_config_file_path=train_config_file_path)

    def _extra_train_config(self, fl_ctx: FLContext, config_info: dict):
        """Additional monai traning configuration customized to individual tasks
        Need the following implementations for further training and validation:
        self.model
        self.device
        self.optimizer
        self.criterion
        self.transform_post
        self.train_loader
        self.valid_loader
        self.inferer
        self.valid_metric
        """
        pass

    def train_config(self, fl_ctx: FLContext, train_config_file_path: str):
        """Common monai traning configuration
        Individual training tasks can be customized by implementing `_extra_train_config`
        """

        # Load training configurations
        if not os.path.isfile(train_config_file_path):
            self.log_error(
                fl_ctx,
                f"Training configuration file does not exist at {train_config_file_path}",
            )
        with open(train_config_file_path) as file:
            config_info = json.load(file)

        self.fedproxloss_mu = config_info.get("fedproxloss_mu", 0.0)
        if self.fedproxloss_mu > 0:
            self.log_info(fl_ctx, f"Using FedProx loss with mu {self.fedproxloss_mu}")
            self.criterion_prox = PTFedProxLoss(mu=self.fedproxloss_mu)

        self._extra_train_config(fl_ctx, config_info)

    def _terminate_trainer(self):
        # collect threads, close files here
        pass

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        # the start and end of a run - only happen once
        if event_type == EventType.START_RUN:
            try:
                self._initialize_trainer(fl_ctx)
            except BaseException as e:
                error_msg = f"Exception in _initialize_trainer: {e}"
                self.log_exception(fl_ctx, error_msg)
                self.system_panic(error_msg, fl_ctx)
        elif event_type == EventType.END_RUN:
            self._terminate_trainer()

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
        for epoch in range(self.aggregation_epochs):
            if abort_signal.triggered:
                return
            self.model.train()
            epoch_len = len(train_loader)
            self.epoch_global = self.epoch_of_start_time + epoch
            self.log_info(
                fl_ctx,
                f"Local epoch {self.client_id}: {epoch + 1}/{self.aggregation_epochs} (lr={self.lr})",
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

                # FedProx loss term
                if self.fedproxloss_mu > 0:
                    fed_prox_loss = self.criterion_prox(self.model, model_global)
                    loss += fed_prox_loss

                loss.backward()
                self.optimizer.step()
                self.writer.add_scalar("train_loss", loss.item(), epoch_len * self.epoch_global + i)
            if val_freq > 0 and epoch % val_freq == 0:
                acc = self.local_valid(self.valid_loader, "val_acc_local_model", abort_signal)
                if acc > self.best_acc:
                    self.save_model(is_best=True)

    def local_valid(self, valid_loader, tb_id, abort_signal: Signal):
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

    def _train(
        self,
        shareable: Shareable,
        fl_ctx: FLContext,
        abort_signal: Signal,
    ) -> Shareable:
        # Check abort signal
        if abort_signal.triggered:
            return make_reply(ReturnCode.TASK_ABORTED)
        self.epoch_of_start_time += self.aggregation_epochs

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

        # make a copy of model_global as reference for potential FedProx loss
        if self.fedproxloss_mu > 0:
            model_global = copy.deepcopy(self.model)
            for param in model_global.parameters():
                param.requires_grad = False
        else:
            model_global = None

        # perform valid before local train
        global_acc = self.local_valid(self.valid_loader, "val_acc_global_model", abort_signal)
        if abort_signal.triggered:
            return make_reply(ReturnCode.TASK_ABORTED)
        self.log_info(fl_ctx, f"val_acc_global_model: {global_acc:.4f}")

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

        # perform valid after local train
        acc = self.local_valid(self.valid_loader, "val_acc_local_model", abort_signal)
        if abort_signal.triggered:
            return make_reply(ReturnCode.TASK_ABORTED)
        self.log_info(fl_ctx, f"val_acc_local_model: {acc:.4f}")

        # save model
        self.save_model(is_best=False)
        if acc > self.best_acc:
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
        dxo.set_meta_prop(MetaKey.INITIAL_METRICS, global_acc)

        self.log_info(fl_ctx, "Local epochs finished. Returning shareable")
        return dxo.to_shareable()

    def _submit_model(
        self,
        fl_ctx: FLContext,
        abort_signal: Signal,
    ) -> Shareable:
        # Retrieve the local model saved during training.
        model_data = None
        try:
            # load model to cpu to make it serializable
            model_data = torch.load(self.best_local_model_file, map_location="cpu")
        except Exception as e:
            self.log_error(fl_ctx, f"Unable to load best model: {e}")
            return make_reply(ReturnCode.EXEUTION_ERROR)

        # Checking abort signal
        if abort_signal.triggered:
            return make_reply(ReturnCode.TASK_ABORTED)

        # Create DXO and shareable from model data.
        if model_data:
            dxo = DXO(data_kind=DataKind.WEIGHTS, data=model_data["model_weights"])
            return dxo.to_shareable()
        else:
            # Set return code.
            self.log_error(fl_ctx, f"best local model not found at {self.best_local_model_file}.")
            return make_reply(ReturnCode.EXECUTION_RESULT_ERROR)

    def execute(
        self,
        task_name: str,
        shareable: Shareable,
        fl_ctx: FLContext,
        abort_signal: Signal,
    ) -> Shareable:
        self.log_info(fl_ctx, f"Task name: {task_name}")
        if task_name == self.train_task_name:
            return self._train(shareable=shareable, fl_ctx=fl_ctx, abort_signal=abort_signal)
        elif task_name == self.submit_model_task_name:
            return self._submit_model(fl_ctx=fl_ctx, abort_signal=abort_signal)
        else:
            return make_reply(ReturnCode.TASK_UNKNOWN)
