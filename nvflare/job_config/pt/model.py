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


class Wrap:
    def __init__(
        self,
        model,
        persistor_id="persistor",
        model_locator_id="model_locator",
    ):
        self.model = model
        self.persistor_id = persistor_id
        self.model_locator_id = model_locator_id

    def add_to_fed_job(self, job, ctx):
        """This method is required by Job API.

        Args:
            job: the Job object to add to
            ctx: Job Context

        Returns:

        """
        if torch_ok and isinstance(self.model, nn.Module):  # if model, create a PT persistor
            component = PTFileModelPersistor(model=self.model)
            job.add_component(self.persistor_id, component, ctx)

            component = PTFileModelLocator(pt_persistor_id=self.persistor_id)
            job.add_component(self.model_locator_id, component, ctx)
