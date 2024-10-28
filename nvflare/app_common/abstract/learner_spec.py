# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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
from nvflare.apis.shareable import ReturnCode, Shareable, make_reply
from nvflare.apis.signal import Signal


class Learner(FLComponent):
    def initialize(self, parts: dict, fl_ctx: FLContext):
        """Initialize the Learner object. This is called before the Learner can train or validate.

        This is called only once.

        Args:
            parts: components to be used by the Trainer
            fl_ctx: FLContext of the running environment
        """
        pass

    def train(self, data: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        """Called to perform training. Can be called many times during the lifetime of the Learner.

        Args:
            data: the training input data (e.g. model weights)
            fl_ctx: FLContext of the running environment
            abort_signal: signal to abort the train

        Returns: train result in Shareable

        """
        return make_reply(ReturnCode.TASK_UNSUPPORTED)

    def get_model_for_validation(self, model_name: str, fl_ctx: FLContext) -> Shareable:
        """Called to return the trained model from the Learner.

        Args:
            model_name: type of the model for validation
            fl_ctx: FLContext of the running environment

        Returns: trained model for validation

        """
        return make_reply(ReturnCode.TASK_UNSUPPORTED)

    def validate(self, data: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        """Called to perform validation. Can be called many times during the lifetime of the Learner.

        Args:
            data: the training input data (e.g. model weights)
            fl_ctx: FLContext of the running environment
            abort_signal: signal to abort the train

        Returns: validate result in Shareable

        """
        return make_reply(ReturnCode.TASK_UNSUPPORTED)

    def abort(self, fl_ctx: FLContext):
        """Called (from another thread) to abort the current task (validate or train).

        Note: this is to abort the current task only, not the Trainer. After aborting, the Learner.
        may still be called to perform another task.

        Args:
            fl_ctx: FLContext of the running environment

        """
        pass

    def finalize(self, fl_ctx: FLContext):
        """Called to finalize the Learner (close/release resources gracefully).

        After this call, the Learner will be destroyed.

        Args:
            fl_ctx: FLContext of the running environment

        """
        pass
