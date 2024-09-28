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

from typing import Optional

import torch.nn as nn

from nvflare.app_common.app_constant import DefaultCheckpointFileName, EnvironmentKey
from nvflare.app_opt.pt import PTFileModelPersistor
from nvflare.app_opt.pt.file_model_locator import PTFileModelLocator


class PTFileModelPersistorArgs:
    def __init__(
        self,
        exclude_vars=None,
        global_model_file_name=DefaultCheckpointFileName.GLOBAL_MODEL,
        best_global_model_file_name=DefaultCheckpointFileName.BEST_GLOBAL_MODEL,
        source_ckpt_file_full_name=None,
        filter_id: str = None,
    ):
        self.exclude_vars = exclude_vars
        self.filter_id = filter_id
        self.ckpt_dir_env_key = EnvironmentKey.CHECKPOINT_DIR
        self.ckpt_file_name_env_key = EnvironmentKey.CHECKPOINT_FILE_NAME
        self.global_model_file_name = global_model_file_name
        self.best_global_model_file_name = best_global_model_file_name
        self.source_ckpt_file_full_name = source_ckpt_file_full_name


class PTModel:
    def __init__(self, model, model_persistor_args: Optional[PTFileModelPersistorArgs] = None):
        """PyTorch model wrapper.

        If model is an nn.Module, add a PTFileModelPersistor with the model and a PTFileModelLocator.

        Args:
            model (any): model
        """
        self.model = model
        self.model_persistor_args = model_persistor_args if model_persistor_args else PTFileModelPersistorArgs()

    def add_to_fed_job(self, job, ctx):
        """This method is used by Job API.

        Args:
            job: the Job object to add to
            ctx: Job Context

        Returns:
            dictionary of ids of component added
        """
        if isinstance(self.model, nn.Module):  # if model, create a PT persistor
            persistor = PTFileModelPersistor(
                exclude_vars=self.model_persistor_args.exclude_vars,
                model=self.model,
                global_model_file_name=self.model_persistor_args.global_model_file_name,
                best_global_model_file_name=self.model_persistor_args.best_global_model_file_name,
                source_ckpt_file_full_name=self.model_persistor_args.source_ckpt_file_full_name,
                filter_id=self.model_persistor_args.filter_id,
            )
            persistor_id = job.add_component(comp_id="persistor", obj=persistor, ctx=ctx)

            locator = PTFileModelLocator(pt_persistor_id=persistor_id)
            locator_id = job.add_component(comp_id="locator", obj=locator, ctx=ctx)
            return {"persistor_id": persistor_id, "locator_id": locator_id}
        else:
            raise ValueError(f"Unsupported type for model: {type(self.model)}.")
