# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

from typing import Any, Dict, Optional, Union

import torch.nn as nn

from nvflare.app_common.abstract.model_locator import ModelLocator
from nvflare.app_common.abstract.model_persistor import ModelPersistor
from nvflare.app_opt.pt import PTFileModelPersistor
from nvflare.app_opt.pt.file_model_locator import PTFileModelLocator
from nvflare.job_config.api import validate_object_for_job


class PTModel:
    def __init__(
        self,
        model: Union[nn.Module, Dict[str, Any]],
        persistor: Optional[ModelPersistor] = None,
        locator: Optional[ModelLocator] = None,
        allow_numpy_conversion: bool = True,
        initial_ckpt: Optional[str] = None,
    ):
        """PyTorch model wrapper.

        Supports two input modes:
        1. nn.Module instance - existing behavior
        2. Dict config {"path": "module.Class", "args": {...}} - new

        Note: PyTorch requires model for architecture because .pt/.pth files
        only contain state_dict (weights), not model architecture.

        Args:
            model: Model input (required), can be:
                - nn.Module: Model instance (existing behavior)
                - dict: {"path": "fully.qualified.Class", "args": {...}}
            persistor (optional, ModelPersistor): Custom persistor. If None, creates default.
            locator (optional, ModelLocator): Custom locator. If None, creates default.
            allow_numpy_conversion (bool): If True, enables conversion between PyTorch
                tensors and NumPy arrays. Defaults to True.
            initial_ckpt (str, optional): Absolute path to checkpoint file.
                May not exist locally (server-side path). Used to load pre-trained weights.
        """
        self.model = model
        self.initial_ckpt = initial_ckpt

        if persistor:
            validate_object_for_job("persistor", persistor, ModelPersistor)
        self.persistor = persistor
        if locator:
            validate_object_for_job("locator", locator, ModelLocator)
        self.locator = locator
        self.allow_numpy_conversion = allow_numpy_conversion

        # Extract model class path if dict config
        self.model_class_path = None
        self.model_args = None
        if isinstance(model, dict):
            self.model_class_path = model.get("path")
            self.model_args = model.get("args", {})
            if not self.model_class_path:
                raise ValueError("Dict model config must have 'path' key with fully qualified class path.")

    def add_to_fed_job(self, job, ctx):
        """This method is used by Job API.

        Args:
            job: the Job object to add to
            ctx: Job Context

        Returns:
            dictionary of ids of component added
        """
        # Handle nn.Module instance
        if isinstance(self.model, nn.Module):
            persistor = self._create_persistor_for_module()
            persistor_id = job.add_component(comp_id="persistor", obj=persistor, ctx=ctx)

            locator = self.locator if self.locator else PTFileModelLocator(pt_persistor_id=persistor_id)
            locator_id = job.add_component(comp_id="locator", obj=locator, ctx=ctx)
            return {"persistor_id": persistor_id, "locator_id": locator_id}

        # Handle dict config {"path": "...", "args": {...}}
        elif isinstance(self.model, dict):
            persistor = self._create_persistor_for_dict_config()
            persistor_id = job.add_component(comp_id="persistor", obj=persistor, ctx=ctx)

            locator = self.locator if self.locator else PTFileModelLocator(pt_persistor_id=persistor_id)
            locator_id = job.add_component(comp_id="locator", obj=locator, ctx=ctx)
            return {"persistor_id": persistor_id, "locator_id": locator_id}

        else:
            raise ValueError(
                f"Unable to add {self.model} to job. Expected nn.Module or dict config, " f"but got {type(self.model)}."
            )

    def _create_persistor_for_module(self) -> ModelPersistor:
        """Create persistor for nn.Module model."""
        if self.persistor:
            return self.persistor

        persistor_kwargs = {
            "model": self.model,
            "allow_numpy_conversion": self.allow_numpy_conversion,
        }

        # Add checkpoint path if provided
        if self.initial_ckpt:
            persistor_kwargs["source_ckpt_file_full_name"] = self.initial_ckpt

        return PTFileModelPersistor(**persistor_kwargs)

    def _create_persistor_for_dict_config(self) -> ModelPersistor:
        """Create persistor for dict config model."""
        if self.persistor:
            return self.persistor

        # For dict config, pass full dict with path and args (resolved at runtime via instantiate_class)
        persistor_kwargs = {
            "model": self.model,  # Pass full dict {"path": "...", "args": {...}}
            "allow_numpy_conversion": self.allow_numpy_conversion,
        }

        # Add checkpoint path if provided
        if self.initial_ckpt:
            persistor_kwargs["source_ckpt_file_full_name"] = self.initial_ckpt

        return PTFileModelPersistor(**persistor_kwargs)
