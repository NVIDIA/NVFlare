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

import os
import pickle
from typing import Dict

import numpy as np
from sklearn.neural_network import MLPClassifier

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContext
from nvflare.app_common.abstract.model import (
    ModelLearnable,
    ModelLearnableKey,
    make_model_learnable,
    validate_model_learnable,
)
from nvflare.app_common.app_constant import AppConstants, DefaultCheckpointFileName, EnvironmentKey
from nvflare.app_common.app_event_type import AppEventType
from nvflare.app_common.model_desc import ModelDescriptor
from nvflare.app_opt.pt.file_model_persistor import PTFileModelPersistor


class BioNeMoMLPModelPersistor(PTFileModelPersistor):
    def __init__(
        self,
        global_model_file_name=DefaultCheckpointFileName.GLOBAL_MODEL,
        best_global_model_file_name=DefaultCheckpointFileName.BEST_GLOBAL_MODEL,
        source_ckpt_file_full_name=None,
        filter_id: str = None,
    ):
        """Persist sklearn-based model to/from file system.

        Args:
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
        self.model = MLPClassifier(solver="adam", hidden_layer_sizes=(512, 256, 128), random_state=10, max_iter=1)
        self.log_dir = None
        self.ckpt_preload_path = None
        self.ckpt_dir_env_key = EnvironmentKey.CHECKPOINT_DIR
        self.ckpt_file_name_env_key = EnvironmentKey.CHECKPOINT_FILE_NAME
        self.global_model_file_name = global_model_file_name
        self.best_global_model_file_name = best_global_model_file_name
        self.source_ckpt_file_full_name = source_ckpt_file_full_name
        self.learned_weights = None

        self.default_train_conf = None

        if source_ckpt_file_full_name and not os.path.exists(source_ckpt_file_full_name):
            raise ValueError(f"specified source checkpoint model file {source_ckpt_file_full_name} does not exist")

    def _initialize(self, fl_ctx: FLContext):
        # To initialize the model, fit on some random data
        class_labels = [
            "Cell_membrane",
            "Cytoplasm",
            "Endoplasmic_reticulum",
            "Extracellular",
            "Golgi_apparatus",
            "Lysosome",
            "Mitochondrion",
            "Nucleus",
            "Peroxisome",
            "Plastid",
        ]
        _X, _y = [], []
        for label in class_labels:
            _X.append(np.random.rand(768))
            _y.append(label)
        self.model.fit(_X, _y)
        self.log_info(
            fl_ctx,
            f"MLPClassifier coefficients {[np.shape(x) for x in self.model.coefs_]}, intercepts {[np.shape(x) for x in self.model.intercepts_]}",
        )

        app_root = fl_ctx.get_prop(FLContextKey.APP_ROOT)
        log_dir = fl_ctx.get_prop(AppConstants.LOG_DIR)
        if log_dir:
            self.log_dir = os.path.join(app_root, log_dir)
        else:
            self.log_dir = app_root

        self._ckpt_save_path = os.path.join(self.log_dir, self.global_model_file_name)
        self._best_ckpt_save_path = os.path.join(self.log_dir, self.best_global_model_file_name)

    def load_model(self, fl_ctx: FLContext) -> ModelLearnable:
        """Convert initialised model into Learnable/Model format.

        Args:
            fl_ctx (FLContext): FL Context delivered by workflow

        Returns:
            Model: a Learnable/Model object
        """

        try:
            weights = {}
            for i, w in enumerate(self.model.coefs_):
                weights[f"coef_{i}"] = w
            for i, w in enumerate(self.model.intercepts_):
                weights[f"intercept_{i}"] = w
        except Exception:
            self.log_exception(fl_ctx, "error getting coefficients from model object")
            self.system_panic(reason="cannot create coefficients from model object", fl_ctx=fl_ctx)
            return None

        if self.model:
            self.default_train_conf = {"train": {"model": type(self.model).__name__}}

        self.learned_weights = weights

        return make_model_learnable(weights, meta_props=self.default_train_conf)

    def handle_event(self, event: str, fl_ctx: FLContext):
        if event == EventType.START_RUN:
            self._initialize(fl_ctx)
        elif event == AppEventType.GLOBAL_BEST_MODEL_AVAILABLE:
            # save the current model as the best model!
            self.save_model_file(self._best_ckpt_save_path)

    def save_model_file(self, save_path):
        pickle.dump(self.learned_weights, open(save_path, "wb"))

    def save_model(self, ml: ModelLearnable, fl_ctx: FLContext):
        err = validate_model_learnable(ml)
        if err:
            raise ValueError(err)

        # update with value of the model learnable
        self.learned_weights = ml.get(ModelLearnableKey.WEIGHTS, {})
        self.save_model_file(self._ckpt_save_path)

    def get_model(self, model_file: str, fl_ctx: FLContext) -> ModelLearnable:
        try:
            location = os.path.join(self.log_dir, model_file)
            weights = pickle.load(open(location, "rb"))

            return make_model_learnable(weights, meta_props=self.default_train_conf)
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
                model_format="WEIGHTS",
                props={},
            )

        location = os.path.join(self.log_dir, self.best_global_model_file_name)
        if os.path.exists(location):
            model_inventory[self.best_global_model_file_name] = ModelDescriptor(
                name=self.best_global_model_file_name,
                location=location,
                model_format="WEIGHTS",
                props={},
            )

        return model_inventory
