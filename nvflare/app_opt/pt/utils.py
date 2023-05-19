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

import logging

import torch
import torch.nn as nn

from nvflare.security.logging import secure_format_exception


def feed_vars(model: nn.Module, model_params):
    """Feed variable values from model_params to pytorch state_dict.

    Args:
        model (nn.Module): the local pytorch model
        model_params: a ModelData message

    Returns:
        a list of params and a dictionary of vars to params
    """
    _logger = logging.getLogger("AssignVariables")
    _logger.debug("AssignVariables...")

    to_assign = []
    n_ext = len(model_params)
    _logger.debug(f"n_ext {n_ext}")

    local_var_dict = model.state_dict()
    for var_name in local_var_dict:
        try:
            if var_name in tuple(model_params):
                nd = model_params[var_name]
                to_assign.append(nd)
                local_var_dict[var_name] = torch.as_tensor(
                    nd
                )  # update local state dict TODO: enable setting of datatype
        except Exception as e:
            _logger.error(f"feed_vars Exception: {secure_format_exception(e)}")
            raise RuntimeError(secure_format_exception(e))

    _logger.debug("Updated local variables to be assigned.")

    n_assign = len(to_assign)
    _logger.info(f"Vars {n_ext} of {n_assign} assigned.")
    return to_assign, local_var_dict
