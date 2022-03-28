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
import json
import os

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from nvflare.apis.dxo import DXO, DataKind, MetaKey, from_shareable
from nvflare.apis.fl_constant import FLContextKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import ReservedHeaderKey, Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.app_constant import AppConstants, ModelName, ValidateType

from pt.learners.supervised_learner import SupervisedLearner


class SupervisedFedSMLearner(SupervisedLearner):
    def __init__(
        self,
        train_config_filename,
        aggregation_epochs: int = 1,
        train_task_name: str = AppConstants.TASK_TRAIN,
        submit_model_task_name: str = AppConstants.TASK_SUBMIT_MODEL,
    ):
        """ Supervised Trainer for FedSM algorithm.

        Args:
            train_config_filename: path of config file, containing basic configs for fedsm training

            aggregation_epochs: the number of training epochs for a round for all three models. Defaults to 1.
                Potentially, the main and selector models can have different epochs, we used the same in this study
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
        self.best_acc_global = 0.0
        self.best_acc_person = 0.0
        self.best_acc_select = 0.0

        # FedSM parameters
        self.lr_select = None

    def initialize(self, parts: dict, fl_ctx: FLContext):
        super().initialize(parts=parts, fl_ctx=fl_ctx)
        self.local_model_file_person = os.path.join(self.app_dir, "local_model_person.pt")
        self.local_model_file_select = os.path.join(self.app_dir, "local_model_select.pt")
        self.best_local_model_file_person = os.path.join(self.app_dir, "best_local_model_person.pt")
        self.best_local_model_file_select = os.path.join(self.app_dir, "best_local_model_select.pt")

    def save_model(self, model_id, is_best=False):
        # save model
        if model_id == "global":
            model_weights = self.model.state_dict()
            save_dict = {"model_weights": model_weights, "epoch": self.epoch_global}
            if is_best:
                save_dict.update({"best_acc": self.best_acc_global})
                torch.save(save_dict, self.best_local_model_file)
            else:
                torch.save(save_dict, self.local_model_file)
        elif model_id == "person":
            model_weights = self.model_person.state_dict()
            save_dict = {"model_weights": model_weights, "epoch": self.epoch_global}
            if is_best:
                save_dict.update({"best_acc": self.best_acc_person})
                torch.save(save_dict, self.best_local_model_file_person)
            else:
                torch.save(save_dict, self.local_model_file_person)
        elif model_id == "select":
            model_weights = self.model_select.state_dict()
            save_dict = {"model_weights": model_weights, "epoch": self.epoch_global}
            if is_best:
                save_dict.update({"best_acc": self.best_acc_select})
                torch.save(save_dict, self.best_local_model_file_select)
            else:
                torch.save(save_dict, self.local_model_file_select)

    def local_train(
        self,
        fl_ctx,
        train_loader,
        abort_signal: Signal,
        select_label,
    ):
        """
        three model trainings
        """
        for epoch in range(self.aggregation_epochs):
            if abort_signal.triggered:
                return
            self.model.train()
            self.model_person.train()
            self.model_select.train()

            epoch_len = len(train_loader)
            self.epoch_global = self.epoch_of_start_time + epoch
            self.log_info(
                fl_ctx,
                f"Local epoch {self.client_id}: {epoch + 1}/{self.aggregation_epochs} (lr_main={self.lr_main}, lr_select={self.lr_select})",
            )

            for i, batch_data in enumerate(train_loader):
                if abort_signal.triggered:
                    return
                inputs = batch_data["image"].to(self.device)
                labels = batch_data["label"].to(self.device)
                # construct vector of selector label
                labels_select = np.ones(inputs.size()[0]) * select_label
                labels_select = torch.tensor(labels_select).to(self.device)

                # zero the parameter gradients
                self.optimizer.zero_grad()
                self.optimizer_person.zero_grad()
                self.optimizer_select.zero_grad()

                # forward + backward + optimize
                outputs = self.model(inputs)
                outputs_person = self.model_person(inputs)
                outputs_select = self.model_person(outputs_select)

                loss = self.criterion(outputs, labels)
                loss_person = self.criterion_person(outputs_person, labels)
                loss_select = self.criterion(outputs_select, labels_select)

                loss.backward()
                loss_person.backward()
                loss_select.backward()

                self.optimizer.step()
                self.optimizer_person.step()
                self.optimizer_select.step()

                current_step = epoch_len * self.epoch_global + i
                self.writer.add_scalar("train_loss_global", loss.item(), current_step)
                self.writer.add_scalar("train_loss_personalized", loss_person.item(), current_step)
                self.writer.add_scalar("train_loss_selector", loss_select.item(), current_step)

    def local_valid(self, valid_loader, abort_signal: Signal, select_label=None, tb_id=None):
        self.model.eval()
        self.model_person.eval()
        self.model_select.eval()
        with torch.no_grad():
            total = 0
            correct_global = 0
            correct_person = 0
            correct_select = 0
            for i, (inputs, labels) in enumerate(valid_loader):
                if abort_signal.triggered:
                    return None
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs_global = self.model(inputs)
                _, pred_label_global = torch.max(outputs_global.data, 1)

                outputs_person = self.model_person(inputs)
                _, pred_label_person = torch.max(outputs_person.data, 1)

                outputs_select = self.model_select(inputs)
                pred_label_select = outputs_select.data

                total += inputs.data.size()[0]
                correct_global += (pred_label_global == labels.data).sum().item()
                correct_person += (pred_label_person == labels.data).sum().item()
                correct_select += (pred_label_select == select_label).sum().item()

            metric_global = correct_global / float(total)
            metric_person = correct_person / float(total)
            metric_select = correct_select / float(total)
            if tb_id:
                self.writer.add_scalar(tb_id + "global", metric_global, self.epoch_global)
                self.writer.add_scalar(tb_id + "personalized", metric_person, self.epoch_global)
                self.writer.add_scalar(tb_id + "selector", metric_select, self.epoch_global)

        return metric_global, metric_person, metric_select

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

        # update local model weights with received weights
        dxo = from_shareable(shareable)
        global_weights = dxo.data["_model_weights_"].data
        person_weights = dxo.data["person_weights"].data
        person_weights = person_weights["weights"]
        select_weights = dxo.data["select_weights"].data
        select_weights = select_weights["weights"]
        select_label = dxo.data["select_label"]

        # tensors might need to be reshaped to support HE for secure aggregation.
        # Loading global model weights
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
            raise ValueError(f"No global weights loaded for validation! Received weight dict is {global_weights}")
        # Loading personalized model weights
        local_var_dict = self.model_person.state_dict()
        model_keys = person_weights.keys()
        n_loaded = 0
        for var_name in local_var_dict:
            if var_name in model_keys:
                weights = torch.as_tensor(person_weights[var_name], device=self.device)
                try:
                    # update the local dict
                    local_var_dict[var_name] = torch.as_tensor(torch.reshape(weights, local_var_dict[var_name].shape))
                    n_loaded += 1
                except Exception as e:
                    raise ValueError("Convert weight from {} failed with error: {}".format(var_name, str(e)))
        self.model_person.load_state_dict(local_var_dict)
        if n_loaded == 0:
            raise ValueError(f"No personalized weights loaded for validation! Received weight dict is {person_weights}")
        # Loading selector model weights
        local_var_dict = self.model_select.state_dict()
        model_keys = select_weights.keys()
        n_loaded = 0
        for var_name in local_var_dict:
            if var_name in model_keys:
                weights = torch.as_tensor(select_weights[var_name], device=self.device)
                try:
                    # update the local dict
                    local_var_dict[var_name] = torch.as_tensor(torch.reshape(weights, local_var_dict[var_name].shape))
                    n_loaded += 1
                except Exception as e:
                    raise ValueError("Convert weight from {} failed with error: {}".format(var_name, str(e)))
        self.model_select.load_state_dict(local_var_dict)
        if n_loaded == 0:
            raise ValueError(f"No personalized weights loaded for validation! Received weight dict is {select_weights}")

        # local steps
        epoch_len = len(self.train_loader)
        self.log_info(fl_ctx, f"Local steps per epoch: {epoch_len}")

        # local train
        self.local_train(
            fl_ctx=fl_ctx,
            train_loader=self.train_loader,
            abort_signal=abort_signal,
            select_label=select_label
        )
        if abort_signal.triggered:
            return make_reply(ReturnCode.TASK_ABORTED)
        self.epoch_of_start_time += self.aggregation_epochs

        # perform valid after local train
        acc_global, acc_person, acc_select = self.local_valid(self.valid_loader, abort_signal,
                                                              select_label=select_label, tb_id="val_acc_after_train")
        if abort_signal.triggered:
            return make_reply(ReturnCode.TASK_ABORTED)
        self.log_info(fl_ctx, f"val_acc_after_train: global {acc_global:.4f}, personalized {acc_person:.4f}, selector {acc_select:.4f}")

        # save model
        self.save_model(model_id="global", is_best=False)
        self.save_model(model_id="person", is_best=False)
        self.save_model(model_id="select", is_best=False)
        if acc_global > self.best_acc_global:
            self.best_acc_global = acc_global
            self.save_model(model_id="global", is_best=True)
        if acc_person > self.best_acc_person:
            self.best_acc_person = acc_person
            self.save_model(model_id="person", is_best=True)
        if acc_select >= self.best_acc_select:
            self.best_acc_select = acc_select
            self.save_model(model_id="select", is_best=True)

        # compute delta models, initial models has the primary key set
        local_weights = self.model.state_dict()
        model_diff_global = {}
        for name in global_weights:
            if name not in local_weights:
                continue
            model_diff_global[name] = local_weights[name].cpu().numpy() - global_weights[name]
            if np.any(np.isnan(model_diff_global[name])):
                self.system_panic(f"{name} weights became NaN...", fl_ctx)
                return make_reply(ReturnCode.EXECUTION_EXCEPTION)

        local_weights = self.model_person.state_dict()
        model_diff_person = {}
        for name in person_weights:
            if name not in local_weights:
                continue
            model_diff_person[name] = local_weights[name].cpu().numpy() - person_weights[name]
            if np.any(np.isnan(model_diff_person[name])):
                self.system_panic(f"{name} weights became NaN...", fl_ctx)
                return make_reply(ReturnCode.EXECUTION_EXCEPTION)

        local_weights = self.model_select.state_dict()
        model_diff_select = {}
        for name in select_weights:
            if name not in local_weights:
                continue
            model_diff_select[name] = local_weights[name].cpu().numpy() - select_weights[name]
            if np.any(np.isnan(model_diff_select[name])):
                self.system_panic(f"{name} weights became NaN...", fl_ctx)
                return make_reply(ReturnCode.EXECUTION_EXCEPTION)

        # build the shareable
        dxo_global_weights = DXO(data_kind=DataKind.WEIGHT_DIFF, data=model_diff_global)
        dxo_person_weights = DXO(data_kind=DataKind.WEIGHT_DIFF, data=model_diff_person)
        dxo_select_weights = DXO(data_kind=DataKind.WEIGHT_DIFF, data=model_diff_select)

        dxo_dict = {
            AppConstants.MODEL_WEIGHTS: dxo_global_weights,
            "person_weights": dxo_person_weights,
            "select_weights": dxo_select_weights
        }
        dxo_collection = DXO(data_kind=DataKind.COLLECTION, data=dxo_dict)
        dxo_collection.set_meta_prop(MetaKey.NUM_STEPS_CURRENT_ROUND, epoch_len)

        self.log_info(fl_ctx, "Local epochs finished. Returning shareable")
        return dxo_collection.to_shareable()

    def get_model_for_validation(self, model_name: str, fl_ctx: FLContext) -> Shareable:
        # Retrieve the best local model saved during training.
        if model_name == ModelName.BEST_MODEL:
            model_data = None
            try:
                # load model to cpu as server might or might not have a GPU
                model_data = torch.load(self.best_local_model_file, map_location="cpu")
            except Exception as e:
                self.log_error(fl_ctx, f"Unable to load best model_global: {e}")
            model_data_person = None
            try:
                # load model to cpu as server might or might not have a GPU
                model_data_person = torch.load(self.best_local_model_file_person, map_location="cpu")
            except Exception as e:
                self.log_error(fl_ctx, f"Unable to load best model_person: {e}")
            model_data_select = None
            try:
                # load model to cpu as server might or might not have a GPU
                model_data_select = torch.load(self.best_local_model_file_select, map_location="cpu")
            except Exception as e:
                self.log_error(fl_ctx, f"Unable to load best model_select: {e}")

            # Create DXO and shareable from model data.
            if model_data and model_data_person and model_data_select:
                dxo_global_weights = DXO(data_kind=DataKind.WEIGHTS, data=model_data["model_weights"])
                dxo_select_weights = DXO(data_kind=DataKind.WEIGHTS, data=model_data_person["model_weights"])
                dxo_person_weights = DXO(data_kind=DataKind.WEIGHTS, data=model_data_select["model_weights"])

                dxo_dict = {
                    AppConstants.MODEL_WEIGHTS: dxo_global_weights,
                    "select_weights": dxo_select_weights,
                    "person_weights": dxo_person_weights,
                }
                dxo_collection = DXO(data_kind=DataKind.COLLECTION, data=dxo_dict)
                return dxo_collection.to_shareable()
            else:
                # Set return code.
                self.log_error(fl_ctx, f"best local model not found at {self.best_local_model_file}.")
                return make_reply(ReturnCode.EXECUTION_RESULT_ERROR)
        else:
            raise ValueError(f"Unknown model_type: {model_name}")  # Raised errors are caught in LearnerExecutor class.

    def validate(self, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        # Check abort signal
        if abort_signal.triggered:
            return make_reply(ReturnCode.TASK_ABORTED)

        # get validation information
        self.log_info(fl_ctx, f"Client identity: {fl_ctx.get_identity_name()}")
        model_owner = shareable.get(ReservedHeaderKey.HEADERS).get(AppConstants.MODEL_OWNER)
        if model_owner:
            self.log_info(fl_ctx, f"Evaluating model from {model_owner} on {fl_ctx.get_identity_name()}")
        else:
            model_owner = "model_from_server"

        # update local model weights with received weights
        # three models
        dxo = from_shareable(shareable)
        global_weights = dxo.data["_model_weights_"].data
        select_weights = dxo.data["select_weights"].data
        select_weights = select_weights["weights"]
        person_weights = dxo.data["person_weights"].data
        person_weights = person_weights["weights"]
        select_label = dxo.data["select_label"]

        # tensors might need to be reshaped to support HE for secure aggregation.
        # Loading global model weights
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
            raise ValueError(f"No global weights loaded for validation! Received weight dict is {global_weights}")
        # Loading personalized model weights
        local_var_dict = self.model_person.state_dict()
        model_keys = person_weights.keys()
        n_loaded = 0
        for var_name in local_var_dict:
            if var_name in model_keys:
                weights = torch.as_tensor(person_weights[var_name], device=self.device)
                try:
                    # update the local dict
                    local_var_dict[var_name] = torch.as_tensor(torch.reshape(weights, local_var_dict[var_name].shape))
                    n_loaded += 1
                except Exception as e:
                    raise ValueError("Convert weight from {} failed with error: {}".format(var_name, str(e)))
        self.model_person.load_state_dict(local_var_dict)
        if n_loaded == 0:
            raise ValueError(f"No personalized weights loaded for validation! Received weight dict is {person_weights}")
        # Loading selector model weights
        local_var_dict = self.model_select.state_dict()
        model_keys = select_weights.keys()
        n_loaded = 0
        for var_name in local_var_dict:
            if var_name in model_keys:
                weights = torch.as_tensor(select_weights[var_name], device=self.device)
                try:
                    # update the local dict
                    local_var_dict[var_name] = torch.as_tensor(torch.reshape(weights, local_var_dict[var_name].shape))
                    n_loaded += 1
                except Exception as e:
                    raise ValueError("Convert weight from {} failed with error: {}".format(var_name, str(e)))
        self.model_select.load_state_dict(local_var_dict)
        if n_loaded == 0:
            raise ValueError(f"No personalized weights loaded for validation! Received weight dict is {select_weights}")

        validate_type = shareable.get_header(AppConstants.VALIDATE_TYPE)
        if validate_type == ValidateType.BEFORE_TRAIN_VALIDATE:
            # perform valid before local train
            acc_global, acc_person, acc_select = self.local_valid(self.valid_loader, abort_signal, select_label=select_label, tb_id="val_acc_before_train")
            if abort_signal.triggered:
                return make_reply(ReturnCode.TASK_ABORTED)
            self.log_info(fl_ctx, f"val_acc_before_train ({model_owner}): global {acc_global:.4f}, personalized {acc_person:.4f}, selector {acc_select:.4f}")

            return DXO(data_kind=DataKind.METRICS, data={MetaKey.INITIAL_METRICS: [acc_global, acc_person, acc_select]}, meta={}).to_shareable()

        elif validate_type == ValidateType.MODEL_VALIDATE:
            # perform valid
            train_acc_global, train_acc_person, _ = self.local_valid(self.train_for_valid_loader, abort_signal)
            if abort_signal.triggered:
                return make_reply(ReturnCode.TASK_ABORTED)
            self.log_info(fl_ctx, f"training acc ({model_owner}): global {train_acc_global:.4f}, personalized {train_acc_person:.4f}")

            val_acc_global, val_acc_person, _ = self.local_valid(self.valid_loader, abort_signal)
            if abort_signal.triggered:
                return make_reply(ReturnCode.TASK_ABORTED)
            self.log_info(fl_ctx, f"validation acc ({model_owner}): global {val_acc_global:.4f}, personalized {val_acc_person:.4f}")

            self.log_info(fl_ctx, "Evaluation finished. Returning shareable")

            val_results = {"train_accuracy": [train_acc_global, train_acc_person], "val_accuracy": [val_acc_global, val_acc_person]}

            metric_dxo = DXO(data_kind=DataKind.METRICS, data=val_results)
            return metric_dxo.to_shareable()

        else:
            return make_reply(ReturnCode.VALIDATE_TYPE_UNKNOWN)
