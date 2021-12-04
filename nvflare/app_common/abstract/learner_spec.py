# Copyright (c) 2021, NVIDIA CORPORATION.
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

from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_context import FLContext
from nvflare.app_common.abstract.learnable import Learnable as ModelLearnable


class Learner(FLComponent):

    def initialize(self, parts: dict, fl_ctx: FLContext):
        """
        Initialize the Learner object. This is called before the Learner can train or validate.
        This is called only once.

        Args:
            parts: components to be used by the Trainer
            fl_ctx: FLContext of the running environment

        Returns:

        """
        pass

    def train(self, data: dict, fl_ctx: FLContext) -> dict:
        """
        Called to perform training. Can be called many times during the life time of the Learner.

        Args:
            data: the training input data (e.g. model weights)
            fl_ctx: FLContext of the running environment

        Returns: trained result

        """
        pass

    def get_best_model(self, fl_ctx: FLContext) -> ModelLearnable:
        """
        Called to return the best trained model from the Learner.

        Args:
            fl_ctx: FLContext of the running environment

        Returns: best trained model

        """
        pass

    def validate(self, data: dict, fl_ctx: FLContext) -> dict:
        """
        Called to perform validation. Can be called many times during the life time of the Learner.

        Args:
            data: the training input data (e.g. model weights)
            fl_ctx: FLContext of the running environment

        Returns: validate result

        """
        pass

    def abort(self, fl_ctx: FLContext):
        """
        Called (from another thread) to abort the current task (validate or train)
        Note: this is to abort the current task only, not the Trainer. After aborting, the Learner.
        may still be called to perform another task.

        Args:
            fl_ctx: FLContext of the running environment

        Returns:

        """
        pass

    def finalize(self, fl_ctx: FLContext):
        """
        Called to finalize the Learner (close/release resources gracefully).
        After this call, the Learner will be destroyed.

        Args:
            fl_ctx: FLContext of the running environment

        Returns:

        """
        pass
