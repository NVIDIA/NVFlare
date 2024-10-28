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

import numpy as np
import torch
import torch.optim as optim
from helpers.supervised_pt_fedsm import SupervisedPTFedSMHelper
from learners.supervised_monai_prostate_learner import SupervisedMonaiProstateLearner
from monai.losses import DiceLoss
from monai.networks.nets.unet import UNet
from networks.vgg import vgg11

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
        self.fedsm_person_model_epochs = aggregation_epochs
        self.fedsm_select_model_epochs = fedsm_select_epochs
        self.fedsm_helper = None

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
            spatial_dims=2,
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
        fedsm_person_optimizer = optim.Adam(fedsm_person_model.parameters(), lr=self.lr)
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
            select_model_epochs=self.fedsm_select_model_epochs,
        )

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
            model_diff[name] = np.subtract(end_model[name].cpu().numpy(), initial_model[name], dtype=np.float32)
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
        self.log_info(fl_ctx, f"Current/Total Round: {current_round + 1}/{total_rounds}")
        self.log_info(fl_ctx, f"Client identity: {fl_ctx.get_identity_name()}")

        # update local model parameters with received dxo
        dxo = from_shareable(shareable)
        # load global model weights
        global_weights = dxo.data["global_weights"].data["weights"]
        local_var_dict = self.model.state_dict()
        local_var_dict = self.reshape_global_weights(local_var_dict, global_weights)
        self.model.load_state_dict(local_var_dict)
        # load personalized model weights
        person_weights = dxo.data["person_weights"].data["weights"]
        local_var_dict = self.fedsm_helper.person_model.state_dict()
        local_var_dict = self.reshape_global_weights(local_var_dict, person_weights)
        self.fedsm_helper.person_model.load_state_dict(local_var_dict)
        # load selector model weights
        select_weights = dxo.data["select_weights"].data["weights"]
        local_var_dict = self.fedsm_helper.select_model.state_dict()
        local_var_dict = self.reshape_global_weights(local_var_dict, select_weights)
        self.fedsm_helper.select_model.load_state_dict(local_var_dict)
        # get selector label
        select_label = dxo.data["select_label"]
        # update Adam parameters
        if current_round > 0:
            # get weights from dxo collection
            global_exp_avg = dxo.data.get("select_exp_avg").data["weights"]
            global_exp_avg_sq = dxo.data.get("select_exp_avg_sq").data["weights"]
            # load parameters to optimizer
            local_optim_state_dict = self.fedsm_helper.select_optimizer.state_dict()
            for name in local_optim_state_dict["state"]:
                local_optim_state_dict["state"][name]["exp_avg"] = torch.as_tensor(global_exp_avg[str(name)])
                local_optim_state_dict["state"][name]["exp_avg_sq"] = torch.as_tensor(global_exp_avg_sq[str(name)])
            self.fedsm_helper.select_optimizer.load_state_dict(local_optim_state_dict)

        # local trainings for three models
        epoch_len = len(self.train_loader)
        self.log_info(fl_ctx, f"Local steps per epoch: {epoch_len}")
        # local train global model
        self.local_train(
            fl_ctx=fl_ctx,
            train_loader=self.train_loader,
            abort_signal=abort_signal,
            current_round=current_round,
        )
        if abort_signal.triggered:
            return make_reply(ReturnCode.TASK_ABORTED)
        # local train personalized model
        self.fedsm_helper.local_train_person(
            train_loader=self.train_loader,
            abort_signal=abort_signal,
            writer=self.writer,
            current_round=current_round,
        )
        if abort_signal.triggered:
            return make_reply(ReturnCode.TASK_ABORTED)
        # local train selector
        self.fedsm_helper.local_train_select(
            train_loader=self.train_loader,
            select_label=select_label,
            abort_signal=abort_signal,
            writer=self.writer,
            current_round=current_round,
        )
        if abort_signal.triggered:
            return make_reply(ReturnCode.TASK_ABORTED)

        # compute delta models, initial models has the primary key set
        local_weights = self.model.state_dict()
        model_diff_global = self.compute_model_diff(global_weights, local_weights, fl_ctx)
        local_weights = self.fedsm_helper.person_model.state_dict()
        model_person = local_weights
        for name in model_person:
            model_person[name] = model_person[name].cpu().numpy()
        local_weights = self.fedsm_helper.select_model.state_dict()
        model_diff_select = self.compute_model_diff(select_weights, local_weights, fl_ctx)
        # directly return the optimizer parameters
        optim_weights = self.fedsm_helper.select_optimizer.state_dict().get("state")
        exp_avg = {}
        exp_avg_sq = {}
        for name in optim_weights:
            exp_avg[str(name)] = optim_weights[name]["exp_avg"].cpu().numpy()
            exp_avg_sq[str(name)] = optim_weights[name]["exp_avg_sq"].cpu().numpy()

        # build the shareable
        dxo_dict = {
            "global_weights": DXO(data_kind=DataKind.WEIGHT_DIFF, data=model_diff_global),
            "person_weights": DXO(data_kind=DataKind.WEIGHTS, data=model_person),
            "select_weights": DXO(data_kind=DataKind.WEIGHT_DIFF, data=model_diff_select),
            "select_exp_avg": DXO(data_kind=DataKind.WEIGHTS, data=exp_avg),
            "select_exp_avg_sq": DXO(data_kind=DataKind.WEIGHTS, data=exp_avg_sq),
        }
        dxo_collection = DXO(data_kind=DataKind.COLLECTION, data=dxo_dict)
        dxo_collection.set_meta_prop(MetaKey.NUM_STEPS_CURRENT_ROUND, epoch_len)

        self.log_info(fl_ctx, "Local epochs finished. Returning shareable")
        return dxo_collection.to_shareable()

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
        global_weights = dxo.data["global_weights"].data["weights"]
        local_var_dict = self.model.state_dict()
        local_var_dict = self.reshape_global_weights(local_var_dict, global_weights)
        self.model.load_state_dict(local_var_dict)
        # load personalized model weights
        person_weights = dxo.data["person_weights"].data["weights"]
        local_var_dict = self.fedsm_helper.person_model.state_dict()
        local_var_dict = self.reshape_global_weights(local_var_dict, person_weights)
        self.fedsm_helper.person_model.load_state_dict(local_var_dict)
        # load selector model weights
        select_weights = dxo.data["select_weights"].data["weights"]
        local_var_dict = self.fedsm_helper.select_model.state_dict()
        local_var_dict = self.reshape_global_weights(local_var_dict, select_weights)
        self.fedsm_helper.select_model.load_state_dict(local_var_dict)
        # get selector label
        select_label = dxo.data["select_label"]

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

            person_metric = self.local_valid(
                self.fedsm_helper.person_model,
                self.valid_loader,
                abort_signal,
                tb_id="val_metric_person_model",
                current_round=current_round,
            )
            if abort_signal.triggered:
                return make_reply(ReturnCode.TASK_ABORTED)
            self.log_info(fl_ctx, f"val_metric_person_model ({model_owner}): {person_metric:.4f}")
            # save personalized model locally
            person_best = self.fedsm_helper.update_metric_save_person_model(
                current_round=current_round, metric=person_metric
            )
            if person_best:
                self.log_info(fl_ctx, "best personalized model available")

            select_metric = self.fedsm_helper.local_valid_select(
                self.valid_loader,
                select_label,
                abort_signal,
                tb_id="val_metric_select_model",
                writer=self.writer,
                current_round=current_round,
            )
            if abort_signal.triggered:
                return make_reply(ReturnCode.TASK_ABORTED)
            self.log_info(fl_ctx, f"val_metric_select_model ({model_owner}): {select_metric:.4f}")

            # validation metrics will be averaged with weights at server end for best model record
            # on the two models: global and selector
            # personalized metrics will not be averaged, send a flag to state the best model availability
            metric_dxo = DXO(
                data_kind=DataKind.METRICS,
                data={MetaKey.INITIAL_METRICS: [global_metric, select_metric, person_best]},
                meta={},
            )
            metric_dxo.set_meta_prop(MetaKey.NUM_STEPS_CURRENT_ROUND, len(self.valid_loader))
            return metric_dxo.to_shareable()
        else:
            return make_reply(ReturnCode.VALIDATE_TYPE_UNKNOWN)
