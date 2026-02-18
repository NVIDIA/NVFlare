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

import numpy as np
import torch
from learners.supervised_monai_prostate_learner import SupervisedMonaiProstateLearner

from nvflare.apis.dxo import DXO, DataKind, MetaKey, from_shareable
from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.app_constant import AppConstants, ValidateType


class SupervisedMonaiProstateFedCELearner(SupervisedMonaiProstateLearner):
    def __init__(
        self,
        train_config_filename,
        aggregation_epochs: int = 1,
        train_task_name: str = AppConstants.TASK_TRAIN,
    ):
        """Trainer for prostate segmentation task. It inherits from MONAI trainer.

        Args:
            train_config_filename: directory of config_3 file.
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
        self.fedce_cos_sim = {}
        self.fedce_minus_val = {}
        self.model_last_round = None

    def train_config(self, fl_ctx: FLContext):
        # initialize superclass
        SupervisedMonaiProstateLearner.train_config(self, fl_ctx)
        # initialize last round model record
        self.model_last_round = copy.deepcopy(self.model)

    def get_minus_model(self, global_model, last_round_model, fedce_weight):
        minus_model = copy.deepcopy(global_model)
        for key in minus_model.state_dict().keys():
            temp = (global_model.state_dict()[key] - fedce_weight * last_round_model.state_dict()[key]) / (
                1 - fedce_weight
            )
            minus_model.state_dict()[key].data.copy_(temp)
        return minus_model

    def reshape_global_weights(self, local_var_dict: dict, global_weights: dict):
        model_keys = global_weights.keys()
        n_loaded = 0
        # tensors might need to be reshaped to support HE for secure aggregation.
        for var_name in local_var_dict:
            if var_name in model_keys:
                weights = global_weights[var_name]
                try:
                    # reshape global weights to compute difference later on
                    global_weights[var_name] = np.reshape(weights, local_var_dict[var_name].shape)
                    # update the local dict
                    local_var_dict[var_name] = torch.as_tensor(global_weights[var_name])
                    n_loaded += 1
                except Exception as e:
                    raise ValueError(f"Convert weight from {var_name} failed with error: {str(e)}")
        if n_loaded == 0:
            raise ValueError(f"No global weights loaded! Received weight dict is {global_weights}")
        return local_var_dict

    def compute_model_diff(self, initial_model: dict, end_model: dict, fl_ctx: FLContext):
        model_diff = {}
        for name in initial_model:
            if name not in end_model:
                continue
            model_diff[name] = np.subtract(end_model[name].cpu().numpy(), initial_model[name])
            if np.any(np.isnan(model_diff[name])):
                self.system_panic(f"{name} weights became NaN...", fl_ctx)
                return make_reply(ReturnCode.EXECUTION_EXCEPTION)
        return model_diff

    def train(
        self,
        shareable: Shareable,
        fl_ctx: FLContext,
        abort_signal: Signal,
    ) -> Shareable:
        """Training task pipeline for FedSM
        Get global/client/selector model weights (potentially with HE)
        Local training all three models
        Return updated weights of all three models (model_diff)
        together with the optimizer parameters of selector (model)
        """

        # get round information
        current_round = shareable.get_header(AppConstants.CURRENT_ROUND)
        total_rounds = shareable.get_header(AppConstants.NUM_ROUNDS)
        client_id = fl_ctx.get_identity_name()
        self.log_info(fl_ctx, f"Current/Total Round: {current_round + 1}/{total_rounds}")
        self.log_info(fl_ctx, f"Client identity: {client_id}")

        # get global model and FedCE coefficients from received dxo
        dxo = from_shareable(shareable)
        # global model weights
        global_weights = dxo.data
        local_var_dict = self.model.state_dict()
        local_var_dict = self.reshape_global_weights(local_var_dict, global_weights)
        # load global model weights to local model
        self.model.load_state_dict(local_var_dict)

        # as part of FedCE training, minus model validation is needed
        # before local training from global model
        # from second round
        if current_round > 0:
            # get FedCE coefficient
            fedce_coef = dxo.meta["fedce_coef"][client_id]
            # get fedce_minus model
            fedce_minus_model = self.get_minus_model(
                self.model,
                self.model_last_round,
                fedce_coef,
            )
            # validate minus model
            minus_metric = self.local_valid(
                fedce_minus_model,
                self.valid_loader,
                abort_signal,
                tb_id="val_metric_minus_model",
                current_round=current_round,
            )
            if abort_signal.triggered:
                return make_reply(ReturnCode.TASK_ABORTED)
            self.log_info(fl_ctx, f"val_metric_minus_model: {minus_metric:.4f}")
            # add to the record dict
            self.fedce_minus_val[current_round] = minus_metric
        else:
            fedce_coef = 0.0
            self.fedce_minus_val[0] = 0.0

        self.writer.add_scalar("FedCE_Coef", fedce_coef, current_round)

        # local training from global weights
        epoch_len = len(self.train_loader)
        self.log_info(fl_ctx, f"Local steps per epoch: {epoch_len}")
        self.local_train(
            fl_ctx=fl_ctx,
            train_loader=self.train_loader,
            abort_signal=abort_signal,
            current_round=current_round,
        )
        if abort_signal.triggered:
            return make_reply(ReturnCode.TASK_ABORTED)

        # compute delta models, initial models has the primary key set
        local_weights = self.model.state_dict()
        model_diff_global = self.compute_model_diff(global_weights, local_weights, fl_ctx)

        # update model_last_round
        self.model_last_round.load_state_dict(local_weights)

        # build the shareable
        dxo = DXO(data_kind=DataKind.WEIGHT_DIFF, data=model_diff_global)
        # use the historical mean of minus_val for FedCE
        minus_val = 1.0 - np.mean([self.fedce_minus_val[i] for i in range(current_round + 1)])
        dxo.set_meta_prop("fedce_minus_val", minus_val)
        self.log_info(fl_ctx, "Local epochs finished. Returning shareable")
        return dxo.to_shareable()

    def validate(self, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        """Validation task pipeline for FedSM
        Validate all three models: global/personal/selector
        Return validation score for server-end best model selection and record
        """
        # get round information
        current_round = shareable.get_header(AppConstants.CURRENT_ROUND)

        # validation on models from server
        # renamed to "models_from_server" to avoid confusion with "global_model"
        model_owner = "models_from_server"

        # update local model weights with received dxo
        dxo = from_shareable(shareable)
        # load global model weights
        global_weights = dxo.data
        local_var_dict = self.model.state_dict()
        local_var_dict = self.reshape_global_weights(local_var_dict, global_weights)
        self.model.load_state_dict(local_var_dict)

        # before_train_validate only, can extend to other validate types
        validate_type = shareable.get_header(AppConstants.VALIDATE_TYPE)
        if validate_type == ValidateType.BEFORE_TRAIN_VALIDATE:
            # perform valid before local train
            global_metric = self.local_valid(
                self.model,
                self.valid_loader,
                abort_signal,
                tb_id="val_metric_global_model",
                current_round=current_round,
            )
            if abort_signal.triggered:
                return make_reply(ReturnCode.TASK_ABORTED)
            self.log_info(fl_ctx, f"val_metric_global_model ({model_owner}): {global_metric:.4f}")

            # validation metrics will be averaged with weights at server end for best model record
            metric_dxo = DXO(
                data_kind=DataKind.METRICS,
                data={MetaKey.INITIAL_METRICS: global_metric},
            )
            metric_dxo.set_meta_prop(MetaKey.NUM_STEPS_CURRENT_ROUND, len(self.valid_loader))
            return metric_dxo.to_shareable()
        else:
            return make_reply(ReturnCode.VALIDATE_TYPE_UNKNOWN)
