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
from typing import Any, Dict, Optional, Union

import torch

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import FLContextKey, WorkspaceConstants
from nvflare.apis.fl_context import FLContext
from nvflare.app_common.abstract.model import ModelLearnable
from nvflare.app_common.abstract.model_persistor import ModelPersistor
from nvflare.app_common.app_constant import AppConstants, DefaultCheckpointFileName, EnvironmentKey
from nvflare.app_common.app_event_type import AppEventType
from nvflare.app_common.model_desc import ModelDescriptor
from nvflare.app_opt.pt.decomposers import TensorDecomposer
from nvflare.app_opt.pt.model_persistence_format_manager import PTModelPersistenceFormatManager
from nvflare.fuel.utils import fobs


class PTFileModelPersistor(ModelPersistor):
    def __init__(
        self,
        exclude_vars: Optional[str] = None,
        model: Optional[Union[torch.nn.Module, str, Dict[str, Any]]] = None,
        global_model_file_name: str = DefaultCheckpointFileName.GLOBAL_MODEL,
        best_global_model_file_name: str = DefaultCheckpointFileName.BEST_GLOBAL_MODEL,
        source_ckpt_file_full_name: Optional[str] = None,
        filter_id: Optional[str] = None,
        load_weights_only: bool = False,
        allow_numpy_conversion: bool = True,
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
            model: Model input. Can be one of:
                - torch.nn.Module: Direct model instance
                - str: Component ID of a model registered in config
                - dict: {"path": "fully.qualified.Class", "args": {...}} for dynamic instantiation
                Defaults to None.
            global_model_file_name (str, optional): file name for saving global model. Defaults to DefaultCheckpointFileName.GLOBAL_MODEL.
            best_global_model_file_name (str, optional): file name for saving best global model. Defaults to DefaultCheckpointFileName.BEST_GLOBAL_MODEL.
            source_ckpt_file_full_name (str, optional): full file name for source model checkpoint file. Defaults to None.
            filter_id: Optional string that defines a filter component that is applied to prepare the model to be saved,
                e.g. for serialization of custom Python objects.
            load_weights_only: Indicates whether torch's unpickler should be restricted to loading only tensors, primitive types, dictionaries
                and any types added via :func:`torch.serialization.add_safe_globals`. Defaults to False (<=PyTorch 2.6 behavior).
            allow_numpy_conversion (bool): If set to True, enables conversion between PyTorch tensors and NumPy arrays.
                PyTorch tensors will be converted to NumPy arrays during 'load_model',
                and NumPy arrays will be converted to PyTorch tensors during 'save_model'. Defaults to True.
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
        self.load_weights_only = load_weights_only
        self._allow_numpy_conversion = allow_numpy_conversion

        self.default_train_conf = None

        # Note: We don't validate existence here because the checkpoint path may be
        # a server-side path that doesn't exist on the job submission machine.
        # Existence is validated at runtime in load_model() when the job executes.

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

        if isinstance(self.model, dict):
            # Dict config: {"path": "module.Class", "args": {...}}
            # Dynamically instantiate the model class
            from nvflare.fuel.utils.class_utils import instantiate_class

            class_path = self.model.get("path")
            class_args = self.model.get("args", {})
            if not class_path:
                self.system_panic(reason="Dict model config must have 'path' key with class path", fl_ctx=fl_ctx)
                return
            try:
                self.model = instantiate_class(class_path, class_args)
            except Exception as e:
                self.system_panic(
                    reason=f"Failed to instantiate model class '{class_path}': {e}",
                    fl_ctx=fl_ctx,
                )
                return
            if not isinstance(self.model, torch.nn.Module):
                self.system_panic(
                    reason=f"expect model class '{class_path}' to be torch.nn.Module but got {type(self.model)}",
                    fl_ctx=fl_ctx,
                )
                return
        elif isinstance(self.model, str):
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
        fobs.register(TensorDecomposer)

    def load_model(self, fl_ctx: FLContext) -> ModelLearnable:
        """Convert initialised model into Learnable/Model format.

        Args:
            fl_ctx (FLContext): FL Context delivered by workflow

        Returns:
            Model: a Learnable/Model object
        """
        src_file_name = None
        if self.source_ckpt_file_full_name:
            if os.path.isabs(self.source_ckpt_file_full_name):
                ckpt_path = self.source_ckpt_file_full_name
            else:
                # Relative path: resolve against app's custom directory
                app_root = fl_ctx.get_prop(FLContextKey.APP_ROOT)
                ckpt_path = os.path.join(
                    app_root, WorkspaceConstants.CUSTOM_FOLDER_NAME, self.source_ckpt_file_full_name
                )
            # Checkpoint MUST exist at runtime (fail fast to catch config errors)
            if not os.path.exists(ckpt_path):
                self.system_panic(
                    reason=f"Source checkpoint not found: {ckpt_path}. " "Check that the checkpoint exists at runtime.",
                    fl_ctx=fl_ctx,
                )
                return None
            src_file_name = ckpt_path
        elif self.ckpt_preload_path:
            src_file_name = self.ckpt_preload_path

        if src_file_name:
            try:
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                data = torch.load(src_file_name, map_location=device, weights_only=self.load_weights_only)
                # "checkpoint may contain 'model', 'optimizer', 'lr_scheduler', etc. or only contain model dict directly."
            except Exception:
                self.log_exception(fl_ctx, "error loading checkpoint from {}".format(src_file_name))
                self.system_panic(reason="cannot load model checkpoint", fl_ctx=fl_ctx)
                return None
        else:
            # if no pretrained model provided, use the generated network weights from APP config
            # note that, if set "determinism" in the config, the init model weights will always be the same
            self.log_info(
                fl_ctx,
                f"Both source_ckpt_file_full_name and {AppConstants.CKPT_PRELOAD_PATH} are not provided. Using the default model weights initialized on the persistor side.",
                fire_event=False,
            )
            try:
                data = self.model.state_dict() if self.model is not None else OrderedDict()
            except Exception:
                self.log_exception(fl_ctx, "error getting state_dict from model object")
                self.system_panic(reason="cannot create state_dict from model object", fl_ctx=fl_ctx)
                return None

        if self.model:
            self.default_train_conf = {"train": {"model": type(self.model).__name__}}

        self.persistence_manager = PTModelPersistenceFormatManager(
            data, default_train_conf=self.default_train_conf, allow_numpy_conversion=self._allow_numpy_conversion
        )
        return self.persistence_manager.to_model_learnable(self.exclude_vars)

    def _get_persistence_manager(self, fl_ctx: FLContext):
        if not self.persistence_manager:
            self.load_model(fl_ctx)

        return self.persistence_manager

    def handle_event(self, event: str, fl_ctx: FLContext):
        if event == EventType.START_RUN:
            self._initialize(fl_ctx)
        elif event == AppEventType.GLOBAL_BEST_MODEL_AVAILABLE:
            # save the current model as the best model, or the global best model if available
            ml = fl_ctx.get_prop(AppConstants.GLOBAL_MODEL)
            if ml:
                self._get_persistence_manager(fl_ctx).update(ml)
            self.save_model_file(self._best_ckpt_save_path)

    def save_model_file(self, save_path: str):
        save_dict = self.persistence_manager.to_persistence_dict()
        torch.save(save_dict, save_path)

    def save_model(self, ml: ModelLearnable, fl_ctx: FLContext):
        self._get_persistence_manager(fl_ctx).update(ml)
        self.save_model_file(self._ckpt_save_path)

    def get_model(self, model_file: str, fl_ctx: FLContext) -> ModelLearnable:
        inventory = self.get_model_inventory(fl_ctx)
        if not inventory:
            return None

        desc = inventory.get(model_file)
        if not desc:
            return None

        location = desc.location
        return self._get_model_from_location(location, fl_ctx)

    def _get_model_from_location(self, location, fl_ctx):
        try:
            # Use the "cpu" to load the global model weights, avoid GPU out of memory
            device = "cpu"
            data = torch.load(location, map_location=device, weights_only=self.load_weights_only)
            persistence_manager = PTModelPersistenceFormatManager(
                data, default_train_conf=self.default_train_conf, allow_numpy_conversion=self._allow_numpy_conversion
            )
            return persistence_manager.to_model_learnable(self.exclude_vars)
        except Exception:
            self.log_exception(fl_ctx, "error loading checkpoint from {}".format(location))
            return None

    def get_model_inventory(self, fl_ctx: FLContext) -> Dict[str, ModelDescriptor]:
        model_inventory = {}

        # Include source checkpoint if provided (supports external/pre-trained models)
        if self.source_ckpt_file_full_name and os.path.exists(self.source_ckpt_file_full_name):
            _, tail = os.path.split(self.source_ckpt_file_full_name)
            model_inventory[tail] = ModelDescriptor(
                name=self.source_ckpt_file_full_name,
                location=self.source_ckpt_file_full_name,
                model_format=self._get_persistence_manager(fl_ctx).get_persist_model_format(),
                props={"source": "initial_ckpt"},
            )

        # Include training artifacts
        location = os.path.join(self.log_dir, self.global_model_file_name)
        if os.path.exists(location):
            _, tail = os.path.split(self.global_model_file_name)
            model_inventory[tail] = ModelDescriptor(
                name=self.global_model_file_name,
                location=location,
                model_format=self._get_persistence_manager(fl_ctx).get_persist_model_format(),
                props={},
            )

        location = os.path.join(self.log_dir, self.best_global_model_file_name)
        if os.path.exists(location):
            _, tail = os.path.split(self.best_global_model_file_name)
            model_inventory[tail] = ModelDescriptor(
                name=self.best_global_model_file_name,
                location=location,
                model_format=self._get_persistence_manager(fl_ctx).get_persist_model_format(),
                props={},
            )

        return model_inventory
