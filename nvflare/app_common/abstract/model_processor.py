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

from abc import ABC, abstractmethod

from nvflare.apis.fl_context import FLContext


class ModelProcessor(ABC):
    @abstractmethod
    def extract_model(self, network, multi_processes: bool, model_vars: dict, fl_ctx: FLContext) -> dict:
        """Call to extract the current model from the training network.

        Args:
            network: training network
            multi_processes: boolean to indicates if it's a multi-processes
            model_vars: global model dict
            fl_ctx: FLContext

        Returns:
            a dictionary representing the model
        """
        pass

    @abstractmethod
    def apply_model(self, network, multi_processes: bool, model_params: dict, fl_ctx: FLContext, options=None):
        """Call to apply the model parameters to the training network.

        Args:
            network: training network
            multi_processes: boolean to indicates if it's a multi-processes
            model_params: model parameters to apply
            fl_ctx: FLContext
            options: optional information that can be used for this process

        """
        pass
