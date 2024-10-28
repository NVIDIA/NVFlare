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

from typing import Union

from nvflare.app_common.abstract.fl_model import FLModel
from nvflare.app_common.utils.fl_component_wrapper import FLComponentWrapper


class ModelLearner(FLComponentWrapper):
    STATE = None

    def __init__(self):
        super().__init__()

    def train(self, model: FLModel) -> Union[str, FLModel]:
        """Called by the framework to perform training. Can be called many times during the lifetime of the Learner.

        Args:
            model: the training input model

        Returns: train result as a FLModel if successful; or return code as str if failed.

        """
        pass

    def get_model(self, model_name: str) -> Union[str, FLModel]:
        """Called by the framework to return the trained model from the Learner.

        Args:
            model_name: type of the model for validation

        Returns: trained model; or return code if failed

        """
        pass

    def validate(self, model: FLModel) -> Union[str, FLModel]:
        """Called by the framework to validate the model with the specified weights in dxo

        Args:
            model: the FLModel object that contains model weights to be validated

        Returns: validation metrics in FLModel if successful; or return code if failed.

        """
        pass

    def configure(self, model: FLModel):
        """Called by the framework to configure the Learner with the config info in the model

        Args:
            model: the object that contains config parameters

        Returns: None

        """
        pass
