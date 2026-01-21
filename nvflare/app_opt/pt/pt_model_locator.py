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

import os
from typing import Dict, List, Union

import torch

from nvflare.apis.dxo import DXO
from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContext
from nvflare.app_common.abstract.model import model_learnable_to_dxo
from nvflare.app_common.abstract.model_locator import ModelLocator
from nvflare.app_opt.pt.model_persistence_format_manager import PTModelPersistenceFormatManager
from nvflare.security.logging import secure_format_exception


class PTModelLocator(ModelLocator):
    """ModelLocator for loading pre-trained PyTorch models for cross-site evaluation.

    This locator loads PyTorch checkpoint files (.pt) from a specified directory
    for standalone cross-site evaluation without training. Similar to NPModelLocator
    but for PyTorch models.
    """

    def __init__(
        self,
        model_dir: str = "models",
        model_name: Union[str, Dict[str, str]] = None,
        exclude_vars: str = None,
        load_weights_only: bool = False,
    ):
        """Initialize PTModelLocator.

        Args:
            model_dir: Directory containing pre-trained models (relative to run directory or absolute).
                Defaults to "models".
            model_name: Dict mapping model identifiers to filenames, e.g.,
                {"model_1": "server_1.pt", "model_2": "server_2.pt"}.
                If string, uses it as single model filename with key "server".
                If None, defaults to {"server": "server.pt"}.
            exclude_vars: Regex pattern for variables to exclude when loading. Defaults to None.
            load_weights_only: Whether to restrict unpickling to tensors/primitives only. Defaults to False.
        """
        super().__init__()

        self.model_dir = model_dir
        self.exclude_vars = exclude_vars
        self.load_weights_only = load_weights_only

        # Normalize model_name to dict
        if model_name is None:
            self.model_name = {"server": "server.pt"}
        elif isinstance(model_name, str):
            self.model_name = {"server": model_name}
        elif isinstance(model_name, dict):
            self.model_name = model_name
        else:
            raise ValueError(f"model_name must be str or Dict[str, str], but got: {type(model_name)}")

    def get_model_names(self, fl_ctx: FLContext) -> List[str]:
        """Returns the list of model names for cross-site evaluation.

        Args:
            fl_ctx: FL Context object.

        Returns:
            List[str]: List of model names.
        """
        return list(self.model_name.keys())

    def locate_model(self, model_name: str, fl_ctx: FLContext) -> DXO:
        """Locate and load a specific model by name.

        Args:
            model_name: Name of the model to load.
            fl_ctx: FL Context object.

        Returns:
            DXO containing the model weights, or None if loading fails.
        """
        if model_name not in self.model_name:
            self.log_error(fl_ctx, f"Model '{model_name}' not found in model_name mapping")
            return None

        try:
            # Build the full path to the model file
            if os.path.isabs(self.model_dir):
                # Absolute path - use directly
                model_path = self.model_dir
            else:
                # Relative path - resolve relative to run directory
                engine = fl_ctx.get_engine()
                job_id = fl_ctx.get_prop(FLContextKey.CURRENT_RUN)
                run_dir = engine.get_workspace().get_run_dir(job_id)
                model_path = os.path.join(run_dir, self.model_dir)

            model_file = self.model_name[model_name]
            model_load_path = os.path.join(model_path, model_file)

            # Load the PyTorch checkpoint
            self.log_info(fl_ctx, f"Loading model '{model_name}' from {model_load_path}")

            # Load with CPU to avoid GPU memory issues
            data = torch.load(model_load_path, map_location="cpu", weights_only=self.load_weights_only)

            # Convert to ModelLearnable using PTModelPersistenceFormatManager
            persistence_manager = PTModelPersistenceFormatManager(data)
            model_learnable = persistence_manager.to_model_learnable(self.exclude_vars)

            # Convert to DXO
            dxo = model_learnable_to_dxo(model_learnable)

            self.log_info(fl_ctx, f"Successfully loaded model '{model_name}'")
            return dxo

        except Exception as e:
            self.log_error(fl_ctx, f"Failed to load model '{model_name}': {secure_format_exception(e)}")
            return None
