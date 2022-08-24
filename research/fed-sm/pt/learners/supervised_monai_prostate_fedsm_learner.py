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
from pt.networks.vgg import vgg11
from pt.helpers.supervised_pt_fedsm import SupervisedPTFedSMHelper
from pt.learners.supervised_monai_prostate_learner import SupervisedMonaiProstateLearner

from nvflare.apis.dxo import DXO, DataKind, MetaKey, from_shareable
from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.app_constant import AppConstants, ValidateType


class SupervisedMonaiProstateFedSMLearner(SupervisedMonaiProstateLearner):
    def __init__(
            self,
            train_config_filename,
            aggregation_epochs: int = 1,
            fedsm_select_epochs: int = 1,
            train_task_name: str = AppConstants.TASK_TRAIN,
    ):
        """Trainer for prostate segmentation task. It inherits from MONAI trainer.

        Args:
            train_config_filename: directory of config file.
            fedsm_select_epochs: the number of training epochs of selector model. Defaults to 1.
            aggregation_epochs: the number of training epochs of both global and personalized models for a round. Defaults to 1.
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
        self.fedsm_person_model_epochs = aggregation_epochs
        self.fedsm_select_model_epochs = fedsm_select_epochs

    def train_config(self, fl_ctx: FLContext):
        # Initialize superclass
        SupervisedMonaiProstateLearner.train_config(self, fl_ctx)

        engine = fl_ctx.get_engine()
        ws = engine.get_workspace()
        app_dir = ws.get_app_dir(fl_ctx.get_job_id())

        # Initialize PTFedSMHelper
        # personalized and selector model training epoch
        # personalized model same as global model
        # selector model can be different from the other two task models
        fedsm_person_model = UNet(
            dimensions=2,
            in_channels=1,
            out_channels=1,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
        ).to(self.device)
        fedsm_select_model = vgg11(
            num_classes=self.config_info["select_num_classes"],
        ).to(self.device)
        fedsm_person_criterion = DiceLoss(sigmoid=True)
        fedsm_select_criterion = torch.nn.CrossEntropyLoss()
        fedsm_person_optimizer = optim.Adam(
            fedsm_person_model.parameters(), lr=self.lr
        )
        fedsm_select_optimizer = optim.Adam(
            fedsm_select_model.parameters(), lr=self.config_info["learning_rate_select"]
        )
        self.fedsm_helper = SupervisedPTFedSMHelper(
            person_model=fedsm_person_model,
            select_model=fedsm_select_model,
            person_criterion=fedsm_person_criterion,
            select_criterion=fedsm_select_criterion,
            person_optimizer=fedsm_person_optimizer,
            select_optimizer=fedsm_select_optimizer,
            device=self.device,
            app_dir=app_dir,
            person_model_epochs=self.fedsm_person_model_epochs,
            select_model_epochs=self.fedsm_select_model_epochs
        )

    def train(
        self,
        shareable: Shareable,
        fl_ctx: FLContext,
        abort_signal: Signal,
    ) -> Shareable:
        """Training task pipeline for FedSM
        Get global/client/selector model weights (potentially with HE)
        Prepare for fedprox loss - general
        Local training all three models
        Return updated weights of all three models (model_diff)
        """
        if abort_signal.triggered:
            return make_reply(ReturnCode.TASK_ABORTED)

        # get round information
        current_round = shareable.get_header(AppConstants.CURRENT_ROUND)
        total_rounds = shareable.get_header(AppConstants.NUM_ROUNDS)
        self.log_info(fl_ctx, f"Current/Total Round: {current_round + 1}/{total_rounds}")
        self.log_info(fl_ctx, f"Client identity: {fl_ctx.get_identity_name()}")

        # update local model weights with received weights for all three models
        dxo = from_shareable(shareable)
        global_weights = dxo.data["global_weights"].data
        global_weights = global_weights["weights"]
        person_weights = dxo.data["person_weights"].data
        person_weights = person_weights["weights"]
        select_weights = dxo.data["select_weights"].data
        select_weights = select_weights["weights"]
        select_label = dxo.data["select_label"]

        # tensors might need to be reshaped to support HE for secure aggregation.
        # Loading global model weights
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

        # Loading personalized model weights
        local_var_dict = self.fedsm_helper.person_model.state_dict()
        model_keys = person_weights.keys()
        for var_name in local_var_dict:
            if var_name in model_keys:
                weights = person_weights[var_name]
                try:
                    # reshape global weights to compute difference later on
                    person_weights[var_name] = np.reshape(weights, local_var_dict[var_name].shape)
                    # update the local dict
                    local_var_dict[var_name] = torch.as_tensor(person_weights[var_name])
                except Exception as e:
                    raise ValueError("Convert weight from {} failed with error: {}".format(var_name, str(e)))
        self.fedsm_helper.person_model.load_state_dict(local_var_dict)

        # Loading selector model weights
        local_var_dict = self.fedsm_helper.select_model.state_dict()
        model_keys = select_weights.keys()
        for var_name in local_var_dict:
            if var_name in model_keys:
                weights = select_weights[var_name]
                try:
                    # reshape global weights to compute difference later on
                    select_weights[var_name] = np.reshape(weights, local_var_dict[var_name].shape)
                    # update the local dict
                    local_var_dict[var_name] = torch.as_tensor(select_weights[var_name])
                except Exception as e:
                    raise ValueError("Convert weight from {} failed with error: {}".format(var_name, str(e)))
        self.fedsm_helper.select_model.load_state_dict(local_var_dict)

        # local steps
        epoch_len = len(self.train_loader)
        self.log_info(fl_ctx, f"Local steps per epoch: {epoch_len}")

        # make a copy of model_global as reference for
        # potential FedProx loss of global model
        model_global = copy.deepcopy(self.model)
        for param in model_global.parameters():
            param.requires_grad = False

        # local train global model
        self.local_train(
            fl_ctx=fl_ctx,
            train_loader=self.train_loader,
            model_global=model_global,
            abort_signal=abort_signal
        )
        if abort_signal.triggered:
            return make_reply(ReturnCode.TASK_ABORTED)
        self.epoch_of_start_time += self.aggregation_epochs

        # local train selector and personalized models
        self.fedsm_helper.local_train_select(
            train_loader=self.train_loader, select_label=select_label, abort_signal=abort_signal, writer=self.writer
        )
        if abort_signal.triggered:
            return make_reply(ReturnCode.TASK_ABORTED)
        self.fedsm_helper.local_train_person(
            train_loader=self.train_loader, abort_signal=abort_signal, writer=self.writer
        )
        if abort_signal.triggered:
            return make_reply(ReturnCode.TASK_ABORTED)

        # local valid for personalized model each round
        metric = self.local_valid(
            self.fedsm_helper.person_model,
            self.valid_loader,
            abort_signal,
            tb_id="val_metric_per_model",
            record_epoch=self.fedsm_helper.person_epoch_global,
        )
        if abort_signal.triggered:
            return make_reply(ReturnCode.TASK_ABORTED)
        self.log_info(fl_ctx, f"val_metric_per_model: {metric:.4f}")
        # save model
        self.fedsm_helper.update_metric_save_person_model(metric=metric)

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

        local_weights = self.fedsm_helper.person_model.state_dict()
        model_diff_person = {}
        for name in person_weights:
            if name not in local_weights:
                continue
            model_diff_person[name] = local_weights[name].cpu().numpy() - person_weights[name]
            if np.any(np.isnan(model_diff_person[name])):
                self.system_panic(f"{name} weights became NaN...", fl_ctx)
                return make_reply(ReturnCode.EXECUTION_EXCEPTION)

        local_weights = self.fedsm_helper.select_model.state_dict()
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
            "global_weights": dxo_global_weights,
            "person_weights": dxo_person_weights,
            "select_weights": dxo_select_weights
        }
        dxo_collection = DXO(data_kind=DataKind.COLLECTION, data=dxo_dict)
        dxo_collection.set_meta_prop(MetaKey.NUM_STEPS_CURRENT_ROUND, epoch_len)

        self.log_info(fl_ctx, "Local epochs finished. Returning shareable")
        return dxo_collection.to_shareable()

    def validate(self, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        """Validation task pipeline for FedSM
        Get global/selector model weights (potentially with HE)
        Validation the two models on local data (personalized model evaluated at end of each round)
        Return validation score for server-end best model selection
        """
        if abort_signal.triggered:
            return make_reply(ReturnCode.TASK_ABORTED)

        # validation on models from server
        # renamed to "models_from_server" to avoid confusion with "global_model"
        model_owner = "models_from_server"

        # update local model weights with received weights
        # three models
        dxo = from_shareable(shareable)
        global_weights = dxo.data["global_weights"].data
        global_weights = global_weights["weights"]
        select_weights = dxo.data["select_weights"].data
        select_weights = select_weights["weights"]
        select_label = dxo.data["select_label"]

        # Tensors might need to be reshaped to support HE for secure aggregation.
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
        # Loading selector model weights
        local_var_dict = self.fedsm_helper.select_model.state_dict()
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
        self.fedsm_helper.select_model.load_state_dict(local_var_dict)
        if n_loaded == 0:
            raise ValueError(f"No selector weights loaded for validation! Received weight dict is {select_weights}")

        # before_train_validate only, can extend to other validate types
        validate_type = shareable.get_header(AppConstants.VALIDATE_TYPE)
        if validate_type == ValidateType.BEFORE_TRAIN_VALIDATE:
            # perform valid before local train
            global_metric = self.local_valid(
                self.model, self.valid_loader, abort_signal, tb_id="val_metric_global_model", record_epoch=self.epoch_global
            )
            if abort_signal.triggered:
                return make_reply(ReturnCode.TASK_ABORTED)
            self.log_info(fl_ctx, f"val_metric_global_model ({model_owner}): {global_metric:.4f}")
            select_metric = self.fedsm_helper.local_valid_select(
                self.valid_loader, select_label, abort_signal,
                tb_id="val_metric_select_model", writer=self.writer, record_epoch=self.fedsm_helper.select_epoch_global
            )
            if abort_signal.triggered:
                return make_reply(ReturnCode.TASK_ABORTED)
            self.log_info(fl_ctx, f"val_metric_select_model ({model_owner}): {select_metric:.4f}")
            # validation metrics will be averaged with weights at server end for best model record
            # on the two models: global and selector
            metric_dxo = DXO(data_kind=DataKind.METRICS, data={MetaKey.INITIAL_METRICS: [global_metric, select_metric]}, meta={})
            metric_dxo.set_meta_prop(MetaKey.NUM_STEPS_CURRENT_ROUND, len(self.valid_loader))
            return metric_dxo.to_shareable()
        else:
            return make_reply(ReturnCode.VALIDATE_TYPE_UNKNOWN)
