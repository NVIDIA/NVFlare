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

import json
import os
import re
from collections import OrderedDict

import copy

import torch

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContext
from nvflare.app_common.abstract.model import ModelLearnable
from nvflare.app_common.pt.pt_file_model_persistor import PTFileModelPersistor
from nvflare.app_common.app_constant import AppConstants, DefaultCheckpointFileName, EnvironmentKey
from nvflare.app_common.app_event_type import AppEventType
from nvflare.app_common.model_desc import ModelDescriptor

from pt.persistors.pt_fed_utils import PTModelPersistenceFormatManagerPersonalized


class PTFileModelPersistorPersonalized(PTFileModelPersistor):
    def __init__(
        self,
        client_ids,
        exclude_vars=None,
        model=None,
        model_file_name="FL_person_models.pt",
        best_model_file_name="best_FL_person_models.pt",
        source_ckpt_file_full_name=None,
    ):
        """Persist a dict of personalized pytorch-based models to/from file system.

        Single model behavior is the same as PTFileModelPersistor,
        Instead of a single model, it creates a set of models with the same structure corresponding to each client

        Args:
            client_ids (list): the list of client ids
            exclude_vars (str, optional): regex expression specifying weight vars to be excluded from training. Defaults to None.
            model (str, optional): torch model object or component id of the model object. Defaults to None.
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

    def _initialize(self, fl_ctx: FLContext):
        self.log_info(fl_ctx, "personalized persistor initialized")
        super()._initialize(fl_ctx=fl_ctx)
        # self.model is only for getting the model structure from config
        # all operations will be performed on self.personal_models
        # which is a dict of models with same structure as self.model
        self.personal_models = {}
        for id in self.client_ids:
            self.personal_models[id] = copy.deepcopy(self.model)

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
        # data is a dict of personalized models with other training-related items,
        # personalized models under a dict "personalized_models"
        data = {"personalized_models":{}}
        if src_file_name:
            try:
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                data = torch.load(src_file_name, map_location=device)
                # "checkpoint may contain a dict "personalized_models" of models indexed with client ids, 'optimizer', 'lr_scheduler', etc."
            except:
                self.log_exception(fl_ctx, "error loading checkpoint from {}".format(src_file_name))
                self.system_panic(reason="cannot load model checkpoint", fl_ctx=fl_ctx)
                return None
        else:
            # if no pretrained model provided, use the generated network weights from APP config
            # note that, if set "determinism" in the config, the init model weights will always be the same
            try:
                for id in self.client_ids:
                    data["personalized_models"][id] = self.personal_models[id].state_dict() if self.personal_models[id] is not None else OrderedDict()
            except:
                self.log_exception(fl_ctx, "error getting state_dict from model object")
                self.system_panic(reason="cannot create state_dict from model object", fl_ctx=fl_ctx)
                return None

        if self.model:
            self.default_train_conf = {"train": {"model": type(self.model).__name__}}

        self.persistence_manager = PTModelPersistenceFormatManagerPersonalized(data, default_train_conf=self.default_train_conf)
        return self.persistence_manager.to_model_learnable(self.exclude_vars)

    def handle_event(self, event: str, fl_ctx: FLContext):
        if event == EventType.START_RUN:
            self._initialize(fl_ctx)
        elif event == AppEventType.GLOBAL_BEST_MODEL_AVAILABLE:
            # save the current models as the best model
            self.save_model_file(self._best_ckpt_save_path)

    def save_model_file(self, save_path: str):
        save_dict = self.persistence_manager.to_persistence_dict()
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
            persistence_manager = PTModelPersistenceFormatManagerPersonalized(data, default_train_conf=self.default_train_conf)
            return persistence_manager.to_model_learnable(self.exclude_vars)
        except BaseException as e:
            self.log_exception(fl_ctx, "error loading checkpoint from {}".format(model_file))
            return {}

    def get_model_inventory(self, fl_ctx: FLContext) -> {str: ModelDescriptor}:
        model_inventory = {}
        location = os.path.join(self.log_dir, self.global_model_file_name)
        if os.path.exists(location):
            model_inventory[self.global_model_file_name] = ModelDescriptor(
                name=self.global_model_file_name,
                location=location,
                model_format=self.persistence_manager.get_persist_model_format(),
                props={},
            )

        location = os.path.join(self.log_dir, self.best_global_model_file_name)
        if os.path.exists(location):
            model_inventory[self.best_global_model_file_name] = ModelDescriptor(
                name=self.best_global_model_file_name,
                location=location,
                model_format=self.persistence_manager.get_persist_model_format(),
                props={},
            )

        return model_inventory
