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

import json
import os
import re
from collections import OrderedDict
from typing import Dict

import torch

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContext
from nvflare.app_common.abstract.model import ModelLearnable
from nvflare.app_common.abstract.model_persistor import ModelPersistor
from nvflare.app_common.app_constant import AppConstants, DefaultCheckpointFileName, EnvironmentKey
from nvflare.app_common.app_event_type import AppEventType
from nvflare.app_common.model_desc import ModelDescriptor
from nvflare.app_opt.pt.model_persistence_format_manager import PTModelPersistenceFormatManager


class PTFileModelPersistor(ModelPersistor):
    def __init__(
        self,
        exclude_vars=None,
        model=None,
        global_model_file_name=DefaultCheckpointFileName.GLOBAL_MODEL,
        best_global_model_file_name=DefaultCheckpointFileName.BEST_GLOBAL_MODEL,
        source_ckpt_file_full_name=None,
        filter_id: str = None,
    ):
        """Persist pytorch-based model to/from file system.

        This Model Persistor tries to load PT model data in the following three ways:

            1. Load from a specified source checkpoint file
            2. Load from a location from the app folder
            3. Load from a torch model object

        The Persistor tries method 1 first if the source_ckpt_file_full_name is specified;
        If source_ckpt_file_full_name is not specified, it tries method 2;
        If no checkpoint location is specified in the app folder, it tries method 3.

        Method 2 - Load from a location from the app folder

        It is assumed that the app folder must contain the environments.json file. Among other things, this
        JSON file must specify where to find the checkpoint file. It does so with two JSON elements:

            - APP_CKPT_DIR: specifies the folder (within the app) where the checkpoint file resides.
            - APP_CKPT: specifies the base file name of the checkpoint

        Here is an example of the environments.json content::

            {
                "APP_CKPT_DIR": "model",
                "APP_CKPT": "pretrained_model.pt"
            }

        In this example, the checkpoint file is located in the "model" folder within the app and is named
        pretrained_model.pt.

        Method 3 - Load from a torch model object. In this case, the 'model' arg must be a valid torch
        model, or the component ID of a valid torch model included in the "components" section of
        your config_fed_server.json.

        If all 3 methods fail, system_panic() is called.

        If checkpoint folder name is specified, then global model and best global model will be saved to it;
        Otherwise they will be saved directly in the app folder.

        The model is saved in a dict depending on the persistor you used. You might need to access it with
        ``model.load_state_dict(torch.load(path_to_model)["model"])`` as there is additional meta information together with the model weights.

        Args:
            exclude_vars (str, optional): regex expression specifying weight vars to be excluded from training. Defaults to None.
            model (str, optional): torch model object or component id of the model object. Defaults to None.
            global_model_file_name (str, optional): file name for saving global model. Defaults to DefaultCheckpointFileName.GLOBAL_MODEL.
            best_global_model_file_name (str, optional): file name for saving best global model. Defaults to DefaultCheckpointFileName.BEST_GLOBAL_MODEL.
            source_ckpt_file_full_name (str, optional): full file name for source model checkpoint file. Defaults to None.
            filter_id: Optional string that defines a filter component that is applied to prepare the model to be saved,
                e.g. for serialization of custom Python objects.
        Raises:
            ValueError: when source_ckpt_file_full_name does not exist
        """
        super().__init__(
            filter_id=filter_id,
        )
        self.exclude_vars = re.compile(exclude_vars) if exclude_vars else None
        self.model = model
        self.log_dir = None
        self.ckpt_preload_path = None
        self.persistence_manager = None
        self.ckpt_dir_env_key = EnvironmentKey.CHECKPOINT_DIR
        self.ckpt_file_name_env_key = EnvironmentKey.CHECKPOINT_FILE_NAME
        self.global_model_file_name = global_model_file_name
        self.best_global_model_file_name = best_global_model_file_name
        self.source_ckpt_file_full_name = source_ckpt_file_full_name

        self.default_train_conf = None

        if source_ckpt_file_full_name and not os.path.exists(source_ckpt_file_full_name):
            raise ValueError("specified source checkpoint model file {} does not exist")

    def _initialize(self, fl_ctx: FLContext):
        app_root = fl_ctx.get_prop(FLContextKey.APP_ROOT)
        env = None
        run_args = fl_ctx.get_prop(FLContextKey.ARGS)
        if run_args:
            env_config_file_name = os.path.join(app_root, run_args.env)
            if os.path.exists(env_config_file_name):
                try:
                    with open(env_config_file_name) as file:
                        env = json.load(file)
                except Exception:
                    self.system_panic(
                        reason="error opening env config file {}".format(env_config_file_name), fl_ctx=fl_ctx
                    )
                    return

        if env is not None:
            if env.get(self.ckpt_dir_env_key, None):
                fl_ctx.set_prop(AppConstants.LOG_DIR, env[self.ckpt_dir_env_key], private=True, sticky=True)
            if env.get(self.ckpt_file_name_env_key) is not None:
                fl_ctx.set_prop(
                    AppConstants.CKPT_PRELOAD_PATH, env[self.ckpt_file_name_env_key], private=True, sticky=True
                )

        log_dir = fl_ctx.get_prop(AppConstants.LOG_DIR)
        if log_dir:
            self.log_dir = os.path.join(app_root, log_dir)
        else:
            self.log_dir = app_root

        self._ckpt_save_path = os.path.join(self.log_dir, self.global_model_file_name)
        self._best_ckpt_save_path = os.path.join(self.log_dir, self.best_global_model_file_name)

        ckpt_preload_path = fl_ctx.get_prop(AppConstants.CKPT_PRELOAD_PATH)
        if ckpt_preload_path:
            self.ckpt_preload_path = os.path.join(app_root, ckpt_preload_path)

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        if isinstance(self.model, str):
            # treat it as model component ID
            model_component_id = self.model
            engine = fl_ctx.get_engine()
            self.model = engine.get_component(model_component_id)
            if not self.model:
                self.system_panic(reason="cannot find model component '{}'".format(model_component_id), fl_ctx=fl_ctx)
                return
            if not isinstance(self.model, torch.nn.Module):
                self.system_panic(
                    reason="expect model component '{}' to be torch.nn.Module but got {}".format(
                        model_component_id, type(self.model)
                    ),
                    fl_ctx=fl_ctx,
                )
                return
        elif self.model and not isinstance(self.model, torch.nn.Module):
            self.system_panic(
                reason="expect model to be torch.nn.Module but got {}".format(type(self.model)), fl_ctx=fl_ctx
            )
            return

        fl_ctx.sync_sticky()

    def load_model(self, fl_ctx: FLContext) -> ModelLearnable:
        """Convert initialised model into Learnable/Model format.

        Args:
            fl_ctx (FLContext): FL Context delivered by workflow

        Returns:
            Model: a Learnable/Model object
        """
        src_file_name = None
        if self.source_ckpt_file_full_name:
            src_file_name = self.source_ckpt_file_full_name
        elif self.ckpt_preload_path:
            src_file_name = self.ckpt_preload_path

        if src_file_name:
            try:
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                data = torch.load(src_file_name, map_location=device)
                # "checkpoint may contain 'model', 'optimizer', 'lr_scheduler', etc. or only contain model dict directly."
            except Exception:
                self.log_exception(fl_ctx, "error loading checkpoint from {}".format(src_file_name))
                self.system_panic(reason="cannot load model checkpoint", fl_ctx=fl_ctx)
                return None
        else:
            # if no pretrained model provided, use the generated network weights from APP config
            # note that, if set "determinism" in the config, the init model weights will always be the same
            try:
                data = self.model.state_dict() if self.model is not None else OrderedDict()
            except Exception:
                self.log_exception(fl_ctx, "error getting state_dict from model object")
                self.system_panic(reason="cannot create state_dict from model object", fl_ctx=fl_ctx)
                return None

        if self.model:
            self.default_train_conf = {"train": {"model": type(self.model).__name__}}

        self.persistence_manager = PTModelPersistenceFormatManager(data, default_train_conf=self.default_train_conf)
        return self.persistence_manager.to_model_learnable(self.exclude_vars)

    def handle_event(self, event: str, fl_ctx: FLContext):
        if event == EventType.START_RUN:
            self._initialize(fl_ctx)
        elif event == AppEventType.GLOBAL_BEST_MODEL_AVAILABLE:
            # save the current model as the best model!
            self.save_model_file(self._best_ckpt_save_path)

    def save_model_file(self, save_path: str):
        save_dict = self.persistence_manager.to_persistence_dict()
        torch.save(save_dict, save_path)

    def save_model(self, ml: ModelLearnable, fl_ctx: FLContext):
        self.persistence_manager.update(ml)
        self.save_model_file(self._ckpt_save_path)

    def get_model(self, model_file: str, fl_ctx: FLContext) -> ModelLearnable:
        try:
            # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            # Use the "cpu" to load the global model weights, avoid GPU out of memory
            device = "cpu"
            location = os.path.join(self.log_dir, model_file)
            data = torch.load(location, map_location=device)
            persistence_manager = PTModelPersistenceFormatManager(data, default_train_conf=self.default_train_conf)
            return persistence_manager.to_model_learnable(self.exclude_vars)
        except Exception:
            self.log_exception(fl_ctx, "error loading checkpoint from {}".format(model_file))
            return {}

    def get_model_inventory(self, fl_ctx: FLContext) -> Dict[str, ModelDescriptor]:
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
