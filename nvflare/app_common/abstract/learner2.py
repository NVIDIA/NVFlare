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

from abc import abstractmethod
from typing import Union

from nvflare.apis.dxo import DXO
from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_context import FLContext


class Learner2(FLComponent):

    STATE = None

    def __init__(self):
        super().__init__()
        self.engine = None
        self.fl_ctx = None
        self.workspace = None
        self.shareable = None
        self.args = None
        self.site_name = None
        self.job_id = None
        self.app_root = None
        self.abort_signal = None
        self.current_round = 0
        self.total_rounds = 0

    def is_aborted(self) -> bool:
        return self.abort_signal and self.abort_signal.triggered

    def get_shareable_header(self, key: str, default=None):
        if not self.shareable:
            return default
        return self.shareable.get_header(key, default)

    def get_context_prop(self, key: str, default=None):
        if not self.fl_ctx:
            return default
        assert isinstance(self.fl_ctx, FLContext)
        return self.fl_ctx.get_prop(key, default)

    def debug(self, msg: str):
        self.log_debug(self.fl_ctx, msg)

    def info(self, msg: str):
        self.log_info(self.fl_ctx, msg)

    def error(self, msg: str):
        self.log_error(self.fl_ctx, msg)

    def warning(self, msg: str):
        self.log_warning(self.fl_ctx, msg)

    def exception(self, msg: str):
        self.log_exception(self.fl_ctx, msg)

    def critical(self, msg: str):
        self.log_critical(self.fl_ctx, msg)

    def panic(self, reason: str):
        self.system_panic(reason, self.fl_ctx)

    def initialize(self, parts: dict):
        """Initialize the Learner object. This is called before the Learner can train or validate.

        This is called only once.

        Args:
            parts: components to be used by the Trainer
        """
        pass

    @abstractmethod
    def train(self, dxo: DXO) -> Union[str, DXO]:
        """Called to perform training. Can be called many times during the lifetime of the Learner.

        Args:
            dxo: the training input data (e.g. model weights)

        Returns: train result as a DXO if successful; or return code as str if failed.

        """
        pass

    def get_model_for_validation(self, model_name: str) -> Union[str, DXO]:
        """Called to return the trained model from the Learner.

        Args:
            model_name: type of the model for validation

        Returns: trained model for validation

        """
        pass

    def validate_before_train(self, dxo: DXO) -> Union[str, DXO]:
        """Validate the current model with the specified weights in dxo before training.

        Args:
            dxo: the DXO object that contains model weights

        Returns: validation metrics in DXO if successful; or return code if failed.

        """
        pass

    def validate(self, dxo: DXO, validate_type: str, model_owner: str) -> Union[str, DXO]:
        """Validate the model with the specified weights in dxo

        Args:
            dxo: the DXO object that contains model weights
            validate_type: type of validation
            model_owner: the owner of the model weights

        Returns: validation metrics in DXO if successful; or return code if failed.

        """
        pass

    def abort(self):
        """Called (from another thread) to abort the current task (validate or train).

        Note: this is to abort the current task only, not the Trainer. After aborting, the Learner may still be called
        to perform another task.
        """
        pass

    def finalize(self):
        """Called to finalize the Learner (close/release resources gracefully).

        After this call, the Learner will be destroyed.

        Args:

        """
        pass
