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
from collections import OrderedDict

import torch
from persistors.pt_fed_utils import PTModelPersistenceFormatManagerFedSM

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_context import FLContext
from nvflare.app_common.abstract.model import ModelLearnable
from nvflare.app_common.app_constant import DefaultCheckpointFileName
from nvflare.app_opt.pt.file_model_persistor import PTFileModelPersistor


class PTFileFedSMModelPersistor(PTFileModelPersistor):
    def __init__(
        self,
        client_ids,
        exclude_vars=None,
        model=None,
        model_selector=None,
        model_file_name=DefaultCheckpointFileName.GLOBAL_MODEL,
        best_model_file_name=DefaultCheckpointFileName.BEST_GLOBAL_MODEL,
        source_ckpt_file_full_name=None,
    ):
        """Persist a dict of personalized pytorch-based models to/from file system.

        Single model behavior is the same as PTFileModelPersistor,
        Instead of a single model, it creates a set of models with the same structure corresponding to each client

        Args:
            client_ids (list): the list of client ids
            exclude_vars (str, optional): regex expression specifying weight vars to be excluded from training. Defaults to None.
            model (str, optional): torch model object or component id of the model object. Defaults to None.
            model_selector (str, optional): torch model object or component id of the model_selector object. Defaults to None.
            global_model_file_name (str, optional): file name for saving global model. Defaults to DefaultCheckpointFileName.GLOBAL_MODEL.
            best_global_model_file_name (str, optional): file name for saving best global model. Defaults to DefaultCheckpointFileName.BEST_GLOBAL_MODEL.
            source_ckpt_file_full_name (str, optional): full file name for source model checkpoint file. Defaults to None.

        Raises:
            ValueError: when source_ckpt_file_full_name does not exist
        """
        super().__init__(
            exclude_vars=exclude_vars,
            model=model,
            global_model_file_name=model_file_name,
            best_global_model_file_name=best_model_file_name,
            source_ckpt_file_full_name=source_ckpt_file_full_name,
        )
        self.client_ids = client_ids
        self.model_selector = model_selector

    def _initialize(self, fl_ctx: FLContext):
        self.log_info(fl_ctx, "FedSM model persistor initialized")
        super()._initialize(fl_ctx=fl_ctx)

        # First convert str model description to model
        if isinstance(self.model_selector, str):
            # treat it as model component ID
            model_component_id = self.model_selector
            engine = fl_ctx.get_engine()
            self.model_selector = engine.get_component(model_component_id)
            if not self.model_selector:
                self.system_panic(
                    reason=f"cannot find model component '{model_component_id}'",
                    fl_ctx=fl_ctx,
                )
                return
            if not isinstance(self.model_selector, torch.nn.Module):
                self.system_panic(
                    reason=f"expect model component '{model_component_id}' to be torch.nn.Module but got {type(self.model_selector)}",
                    fl_ctx=fl_ctx,
                )
                return
        elif self.model_selector and not isinstance(self.model_selector, torch.nn.Module):
            self.system_panic(
                reason=f"expect model to be torch.nn.Module but got {type(self.model)}",
                fl_ctx=fl_ctx,
            )
            return

        # self.model and self.model_selector is only for getting the model structure from config_3
        # operations will be performed on a set of models, self.model_set_fedsm
        # consisting:
        # selector: selector model
        # global: global model
        # {client_id}: personalized models with same structure as global model
        self.model_set_fedsm = {}
        # initialize all models
        self.model_set_fedsm["select_weights"] = copy.deepcopy(self.model_selector)
        self.model_set_fedsm["select_exp_avg"] = OrderedDict()
        self.model_set_fedsm["select_exp_avg_sq"] = OrderedDict()
        self.model_set_fedsm["select_weights"] = copy.deepcopy(self.model_selector)
        self.model_set_fedsm["global_weights"] = copy.deepcopy(self.model)
        for id in self.client_ids:
            self.model_set_fedsm[id] = copy.deepcopy(self.model)

        fl_ctx.sync_sticky()

    def load_model(self, fl_ctx: FLContext) -> dict:
        """Convert initialised models into a dict of Learnable/Model format.

        Args:
            fl_ctx (FLContext): FL Context delivered by workflow

        Returns:
            Dict of models: a Dict of Learnable/Model object
        """
        src_file_name = None
        if self.source_ckpt_file_full_name:
            src_file_name = self.source_ckpt_file_full_name
        elif self.ckpt_preload_path:
            src_file_name = self.ckpt_preload_path

        # data is a dict of FedSM model set with other training-related items,
        # FedSM model set under a dict "model_set_fedsm"
        # containing select_weights, global_weights, and personal models under each client_id
        data = {"model_set_fedsm": {}}
        if src_file_name:
            try:
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                data = torch.load(src_file_name, map_location=device)
                # checkpoint may contain a dict "model_set_fedsm" of models indexed with model ids
                # 'optimizer', 'lr_scheduler', etc.
            except:
                self.log_exception(fl_ctx, f"error loading checkpoint from {src_file_name}")
                self.system_panic(reason="cannot load model checkpoint", fl_ctx=fl_ctx)
                return None
        else:
            # if no pretrained model provided, use the generated network weights from APP config_3
            # note that, if set "determinism" in the config_3, the init model weights will always be the same
            try:
                data["model_set_fedsm"]["select_weights"] = (
                    self.model_set_fedsm["select_weights"].state_dict()
                    if self.model_set_fedsm["select_weights"] is not None
                    else OrderedDict()
                )
                data["model_set_fedsm"]["select_exp_avg"] = OrderedDict()
                data["model_set_fedsm"]["select_exp_avg_sq"] = OrderedDict()
                data["model_set_fedsm"]["global_weights"] = (
                    self.model_set_fedsm["global_weights"].state_dict()
                    if self.model_set_fedsm["global_weights"] is not None
                    else OrderedDict()
                )
                for id in self.client_ids:
                    data["model_set_fedsm"][id] = (
                        self.model_set_fedsm[id].state_dict() if self.model_set_fedsm[id] is not None else OrderedDict()
                    )
            except:
                self.log_exception(fl_ctx, "error getting state_dict from model object")
                self.system_panic(reason="cannot create state_dict from model object", fl_ctx=fl_ctx)
                return None

        if self.model and self.model_selector:
            self.default_train_conf = {
                "train": {
                    "model": type(self.model).__name__,
                    "model_selector": type(self.model_selector).__name__,
                }
            }
        self.persistence_manager = PTModelPersistenceFormatManagerFedSM(
            data, default_train_conf=self.default_train_conf
        )
        learnable = self.persistence_manager.to_model_learnable(self.exclude_vars)
        return learnable

    def handle_event(self, event: str, fl_ctx: FLContext):
        if event == EventType.START_RUN:
            self._initialize(fl_ctx)

        model_list = ["global_weights", "select_weights"] + self.client_ids
        for model_id in model_list:
            if event == "fedsm_best_model_available_" + model_id:
                # save the current model as the best model
                best_ckpt_save_path = os.path.join(self.log_dir, model_id + "_" + self.best_global_model_file_name)
                self.save_best_model(model_id, best_ckpt_save_path)
                self.log_info(fl_ctx, f"new best model for {model_id} saved.")

    def save_best_model(self, model_id: str, save_path: str):
        save_dict = self.persistence_manager.get_single_model(model_id)
        torch.save(save_dict, save_path)

    def save_model(self, ml_dict: dict, fl_ctx: FLContext):
        self.persistence_manager.update(ml_dict)
        self.save_model_file(self._ckpt_save_path)

    def get_model(self, model_file, fl_ctx: FLContext) -> ModelLearnable:
        try:
            # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            # Use the "cpu" to load the global model weights, avoid GPU out of memory
            device = "cpu"
            location = os.path.join(self.log_dir, model_file)
            data = torch.load(location, map_location=device)
            persistence_manager = PTModelPersistenceFormatManagerFedSM(data, default_train_conf=self.default_train_conf)
            return persistence_manager.to_model_learnable(self.exclude_vars)
        except Exception:
            self.log_exception(fl_ctx, f"error loading checkpoint from {model_file}")
            return {}
