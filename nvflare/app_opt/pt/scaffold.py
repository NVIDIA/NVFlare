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

# The SCAFFOLD-related functions are based on https://github.com/Xtra-Computing/NIID-Bench

# MIT License
#
# Copyright (c) 2021 Yiqun Diao, Qinbin Li
#
# Copyright (c) 2020 International Business Machines
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import copy

import torch
from torch.optim import Optimizer


def get_lr_values(optimizer: Optimizer):
    """
    This function is used to get the learning rates of the optimizer.
    """
    return [group["lr"] for group in optimizer.state_dict()["param_groups"]]


class PTScaffoldHelper(object):
    """Helper to be used with SCAFFOLD components.
    Implements the functions used for the algorithm proposed in
    Karimireddy et al. "SCAFFOLD: Stochastic Controlled Averaging for Federated Learning"
    (https://arxiv.org/abs/1910.06378) using PyTorch.
    SCAFFOLD-related functions are based on https://github.com/Xtra-Computing/NIID-Bench.
    See also Li et al. "Federated Learning on Non-IID Data Silos: An Experimental Study"
    (https://arxiv.org/abs/2102.02079).
    """

    def __init__(self):
        # SCAFFOLD control terms
        self.cnt = 0
        self.c_global = None
        self.c_local = None
        self.c_delta_para = None

    def init(self, model):
        # create models for SCAFFOLD correction terms
        self.c_global = copy.deepcopy(model)
        self.c_local = copy.deepcopy(model)
        # Initialize correction term with zeros
        c_init_para = model.state_dict()
        for k in c_init_para.keys():
            c_init_para[k] = torch.zeros_like(c_init_para[k])
        self.c_global.load_state_dict(c_init_para)
        self.c_local.load_state_dict(c_init_para)

    def get_params(self):
        self.cnt = 0
        # Adapted from https://github.com/Xtra-Computing/NIID-Bench/blob/main/experiments.py#L371
        c_global_para = self.c_global.state_dict()
        c_local_para = self.c_local.state_dict()
        return c_global_para, c_local_para

    def model_update(self, model, curr_lr, c_global_para, c_local_para):
        # Update model using scaffold controls
        # See https://github.com/Xtra-Computing/NIID-Bench/blob/main/experiments.py#L391
        net_para = model.state_dict()
        for key in net_para:
            net_para[key] = net_para[key] - curr_lr * (c_global_para[key] - c_local_para[key])
        model.load_state_dict(net_para)

        self.cnt += 1

    def terms_update(self, model, curr_lr, c_global_para, c_local_para, model_global):
        # Update the local scaffold controls
        # See https://github.com/Xtra-Computing/NIID-Bench/blob/main/experiments.py#L403

        c_new_para = self.c_local.state_dict()
        self.c_delta_para = copy.deepcopy(self.c_local.state_dict())
        global_model_para = model_global.state_dict()
        net_para = model.state_dict()
        for key in net_para:
            c_new_para[key] = (
                c_new_para[key] - c_global_para[key] + (global_model_para[key] - net_para[key]) / (self.cnt * curr_lr)
            )
            self.c_delta_para[key] = (c_new_para[key] - c_local_para[key]).cpu().numpy()
        self.c_local.load_state_dict(c_new_para)

    def load_global_controls(self, weights):
        self.c_global.load_state_dict(weights)

    def get_delta_controls(self):
        if self.c_delta_para is None:
            raise ValueError("c_delta_para hasn't been computed yet!")
        return self.c_delta_para
