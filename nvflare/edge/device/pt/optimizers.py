# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
import copy

import torch.optim as optim

"""
In a typical training loop, the optimizer is created dynamically from model params and other parameters.
The OptimizerWrapper classes defined here simply hold default parameters specified in config.
When used in training loop, the Trainer will call the "get" method of the OptimizerWrapper to get the actual optimizer
defined in torch.optim.
"""


class AdamOptimizerWrapper:

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def get(self, model_params, **kwargs):
        # args could be changed when the actual optimizer is needed
        all_args = copy.copy(self.kwargs)
        all_args.update(kwargs)
        return optim.Adam(model_params, **all_args)


class SGDOptimizerWrapper:

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def get(self, model_params, **kwargs):
        all_args = copy.copy(self.kwargs)
        all_args.update(kwargs)
        return optim.SGD(model_params, **all_args)
