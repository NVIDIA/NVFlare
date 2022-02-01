# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
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

from typing import List

from nvflare.apis.dxo import DXO
from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_context import FLContext


class ModelLocator(FLComponent):
    def get_model_names(self, fl_ctx: FLContext) -> List[str]:
        """List the name of the models.

        Args:
            fl_ctx (FLContext): FL Context object

        Returns:
            List[str]: List of names for models
        """
        pass

    def locate_model(self, model_name, fl_ctx: FLContext) -> DXO:
        """Locate a single model by it's name.

        Args:
            model_name (str): Name of the model.
            fl_ctx (FLContext): FL Context object.

        Returns:
            DXO: a DXO object
        """
        pass
