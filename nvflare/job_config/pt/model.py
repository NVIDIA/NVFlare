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
from nvflare.fuel.utils.import_utils import optional_import

torch, torch_ok = optional_import(module="torch")
if torch_ok:
    import torch.nn as nn

    from nvflare.app_opt.pt import PTFileModelPersistor
    from nvflare.app_opt.pt.file_model_locator import PTFileModelLocator


class PTModel:
    def __init__(self, model):
        self.model = model

    def add_to_fed_job(self, job, ctx):
        """This method is required by Job API.

        Args:
            job: the Job object to add to
            ctx: Job Context

        Returns:

        """
        if torch_ok and isinstance(self.model, nn.Module):  # if model, create a PT persistor
            component = PTFileModelPersistor(model=self.model)
            job.add_component(comp_id="persistor", obj=component, ctx=ctx)

            component = PTFileModelLocator(pt_persistor_id="persistor")
            job.add_component(comp_id="model_locator", obj=component, ctx=ctx)
