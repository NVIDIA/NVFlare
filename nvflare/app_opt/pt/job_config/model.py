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

import torch.nn as nn

from nvflare.app_opt.pt import PTFileModelPersistor
from nvflare.app_opt.pt.file_model_locator import PTFileModelLocator


class PTModel:
    def __init__(self, model):
        """PyTorch model wrapper.

        If model is an nn.Module, add a PTFileModelPersistor with the model and a TFModelPersistor.

        Args:
            model (any): model
        """
        self.model = model

    def add_to_fed_job(self, job, ctx):
        """This method is used by Job API.

        Args:
            job: the Job object to add to
            ctx: Job Context

        Returns:
            dictionary of ids of component added
        """
        if isinstance(self.model, nn.Module):  # if model, create a PT persistor
            persistor = PTFileModelPersistor(model=self.model)
            persistor_id = job.add_component(comp_id="persistor", obj=persistor, ctx=ctx)

            locator = PTFileModelLocator(pt_persistor_id=persistor_id)
            locator_id = job.add_component(comp_id="locator", obj=locator, ctx=ctx)
            return {"persistor_id": persistor_id, "locator_id": locator_id}
        else:
            raise ValueError(
                f"Unable to add {self.model} to job with PTFileModelPersistor. Expected nn.Module but got {type(self.model)}."
            )
