# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
"""CPU-normalizing model persistence for the Evo2 trainable-only state."""

from __future__ import annotations

import os
from pathlib import Path

import adapter_checkpoint
import torch

from nvflare.apis.fl_constant import FLContextKey, WorkspaceConstants
from nvflare.app_common.abstract.model import ModelLearnableKey
from nvflare.app_opt.pt.file_model_persistor import PTFileModelPersistor
from nvflare.app_opt.pt.model_persistence_format_manager import PTModelPersistenceFormatManager


class CPUTrainablePTFileModelPersistor(PTFileModelPersistor):
    """Keep the small server-side Evo2 state on CPU across NVFlare versions.

    Evo2 clients return CPU tensors to release their GPU processes promptly, so
    this example loads its source checkpoint directly onto CPU. Normalizing the
    returned learnable also strictly validates the LoRA/head-only federation boundary.
    """

    @staticmethod
    def _normalize_learnable(model_learnable, context):
        if model_learnable is None:
            return None
        model_learnable[ModelLearnableKey.WEIGHTS] = adapter_checkpoint.copy_trainable_state(
            model_learnable[ModelLearnableKey.WEIGHTS],
            context=context,
        )
        return model_learnable

    def load_model(self, fl_ctx):
        if self.source_ckpt_file_full_name:
            if os.path.isabs(self.source_ckpt_file_full_name):
                checkpoint_path = self.source_ckpt_file_full_name
            else:
                app_root = fl_ctx.get_prop(FLContextKey.APP_ROOT)
                checkpoint_path = os.path.join(
                    app_root,
                    WorkspaceConstants.CUSTOM_FOLDER_NAME,
                    self.source_ckpt_file_full_name,
                )
            if not os.path.isfile(checkpoint_path):
                self.system_panic(reason=f"Source checkpoint not found: {checkpoint_path}.", fl_ctx=fl_ctx)
                return None
            try:
                data = torch.load(checkpoint_path, map_location="cpu", weights_only=self.load_weights_only)
            except Exception:
                self.log_exception(fl_ctx, f"Error loading Evo2 checkpoint from {checkpoint_path}")
                self.system_panic(reason="cannot load Evo2 model checkpoint", fl_ctx=fl_ctx)
                return None
            self.persistence_manager = PTModelPersistenceFormatManager(
                data,
                default_train_conf=self.default_train_conf,
                allow_numpy_conversion=self._allow_numpy_conversion,
            )
            model_learnable = self.persistence_manager.to_model_learnable(self.exclude_vars)
        else:
            model_learnable = super().load_model(fl_ctx)
        return self._normalize_learnable(
            model_learnable,
            "Persisted Evo2 global trainable state",
        )

    def save_model_file(self, save_path: str):
        self.persistence_manager.var_dict = adapter_checkpoint.copy_trainable_state(
            self.persistence_manager.var_dict,
            context="Evo2 global trainable checkpoint",
        )
        super().save_model_file(save_path)

    def save_model(self, ml, fl_ctx):
        """Persist the latest global model and a round-named copy."""

        super().save_model(ml, fl_ctx)
        metadata = ml.get(ModelLearnableKey.META) or {}
        round_index = metadata.get("current_round")
        if type(round_index) is not int or round_index < 0:
            return
        latest_path = Path(self._ckpt_save_path)
        round_path = latest_path.with_name(f"{latest_path.stem}_round_{round_index:03d}{latest_path.suffix}")
        self.save_model_file(str(round_path))

    def get_model(self, model_file: str, fl_ctx):
        return self._normalize_learnable(
            super().get_model(model_file, fl_ctx),
            "Retrieved Evo2 global trainable state",
        )
