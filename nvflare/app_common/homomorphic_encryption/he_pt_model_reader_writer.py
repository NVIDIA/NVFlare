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

import numpy as np

from nvflare.apis.fl_context import FLContext
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.pt.pt_fed_utils import feed_vars
from nvflare.app_common.pt.pt_model_reader_writer import PTModelReaderWriter


class HEPTModelReaderWriter(PTModelReaderWriter):
    def apply_model(self, model_params, fl_ctx: FLContext, options=None):
        """
            Write global model back to local model.
            Needed to extract local parameter shape to reshape decrypted vectors.

        Args:
        :param model_params: a ModelData message
        :return: True if the local model changed
        """
        try:
            # net = self.fitter.net
            net = fl_ctx.get_prop(AppConstants.MODEL_NETWORK)
            # if self.fitter.multi_gpu:
            if fl_ctx.get_prop(AppConstants.MULTI_GPU):
                net = net.module

            # reshape decrypted parameters
            local_var_dict = net.state_dict()
            for var_name in local_var_dict:
                if var_name in model_params:
                    try:
                        self.logger.debug(
                            f"Reshaping {var_name}: {np.shape(model_params[var_name])} to"
                            f" {local_var_dict[var_name].shape}",
                        )
                        model_params[var_name] = np.reshape(model_params[var_name], local_var_dict[var_name].shape)
                    except Exception as e:
                        raise RuntimeError(f"{self._name} reshaping Exception: {str(e)}")

            assign_ops, updated_local_model = feed_vars(net, model_params)
            self.logger.debug(f"assign_ops: {len(assign_ops)}")
            self.logger.debug(f"updated_local_model: {len(updated_local_model)}")
            net.load_state_dict(updated_local_model)
            return assign_ops
        except BaseException as e:
            raise RuntimeError(f"{self._name} apply_model Exception: {str(e)}")
