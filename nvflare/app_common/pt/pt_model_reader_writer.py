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

import logging
from pathlib import Path

from nvflare.apis.fl_context import FLContext
from nvflare.app_common.abstract.model_processor import ModelProcessor
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.pt.pt_fed_utils import feed_vars


class PTModelReaderWriter(ModelProcessor):
    def __init__(self):
        self._name = self.__class__.__name__
        self.logger = logging.getLogger(self._name)

    def initialize(self, trainer):
        pass

    def extract_model(self, model_vars, fl_ctx: FLContext):
        # net = self.fitter.net
        net = fl_ctx.get_prop(AppConstants.MODEL_NETWORK)
        # if self.fitter.multi_gpu:
        if fl_ctx.get_prop(AppConstants.MULTI_GPU):
            net = net.module
        local_state_dict = net.state_dict()

        self.logger.debug("setup local_model_dict")
        local_model_dict = {}
        for var_name in local_state_dict:
            try:
                local_model_dict[var_name] = local_state_dict[var_name].cpu().numpy()
            except Exception as e:
                raise ValueError("Did not work:", str(e))
        self.logger.debug(f"local_model_dict {len(local_model_dict)}")

        return local_model_dict

    def apply_model(self, model_params, fl_ctx: FLContext, options=None):
        """Set the local model according to model_data

        Args:
            model_params: model data information
            fl_ctx (FLContext): FL Context delivered by workflow
            options: . Defaults to None.

        Raises:
            RuntimeError: Raised when being unable to apply model_params to the network

        Returns:
            a list of ops applied to model
        """
        try:
            # net = self.fitter.net
            net = fl_ctx.get_prop(AppConstants.MODEL_NETWORK)
            # if self.fitter.multi_gpu:
            if fl_ctx.get_prop(AppConstants.MULTI_GPU):
                net = net.module
            assign_ops, updated_local_model = feed_vars(net, model_params)
            self.logger.debug(f"assign_ops: {len(assign_ops)}")
            self.logger.debug(f"updated_local_model: {len(updated_local_model)}")
            # self.fitter.net.load_state_dict(updated_local_model)
            net.load_state_dict(updated_local_model)
            return assign_ops
        except Exception as e:
            raise RuntimeError("load_state_dict Exception:", str(e))

    def get_local_models(self, model_params=None):
        """Get the local models

        Args:
            model_params: model data information

        Returns:
            a dictionary from filename to bytes
        """
        if not model_params or "model_log_dir" not in model_params or "model_name" not in model_params:
            return

        model_dir = Path(model_params["model_log_dir"])
        model_name = model_params["model_name"]

        # Find all checkpoint files
        model_file = Path(model_dir, model_name)
        if not model_file.exists():
            return None

        # Read checkpoint files into bytes
        buffer_dict = {}
        with model_file.open("rb") as file_bytes:
            buffer_dict[model_file.name] = file_bytes.read()

        return buffer_dict
