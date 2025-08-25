# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

import torch
import torch.nn as nn


class HyperAggWeight(nn.Module):
    """Hyper Conflict-Averse Aggregation for encoders"""

    def __init__(self, K, init_alpha=1):
        super(HyperAggWeight, self).__init__()
        self.K = K

        # define parameters
        self.alpha = nn.Parameter(torch.ones(K) * init_alpha)

    def forward(self, flatten_last_param_list, flatten_delta, flatten_delta_update):
        flatten_new_param_list = copy.deepcopy(flatten_last_param_list)
        assert self.K == len(flatten_last_param_list)  # number of encoders

        # cut weight into [0, 1]
        alpha = torch.clamp(self.alpha, 0, 1)
        for i in range(self.K):
            flatten_new_param_list[i] += flatten_delta[i] + alpha[i] * flatten_delta_update

        return flatten_new_param_list


class HyperCrossAttention(nn.Module):
    """Hyper Cross-Attention Aggregation for decoders"""

    def __init__(self, model, K, init_beta=1):
        super(HyperCrossAttention, self).__init__()
        self.K = K

        # get layer names
        self.layer_names = []
        for name, _ in model.named_parameters():
            self.layer_names.append(".".join(name.split('.')[1:-1]))
        self.layer_names = sorted(set(self.layer_names))
        self.beta_names = [name.replace('.', '_') for name in self.layer_names]

        # define parameters
        self.beta = nn.ParameterDict()
        for name in self.beta_names:
            self.beta[name] = nn.Parameter(torch.ones(K) * init_beta)  # layer-wise

    def forward(self, last_param_dict_list, delta_dict_list):
        new_param_dict_list = copy.deepcopy(last_param_dict_list)
        assert self.K == len(last_param_dict_list)  # number of decoders

        for name in self.layer_names:
            # cut weight into [0, 1]
            layer_beta = torch.clamp(self.beta[name.replace('.', '_')], 0, 1)
            # get keys of each parameter in the layer (weight & bias)
            layer_keys = []
            for key in delta_dict_list[0].keys():
                if name in key:
                    layer_keys.append(key)

            for key in layer_keys:
                cross_delta = torch.stack([delta_dict_list[j][key].reshape(-1) for j in range(self.K)])
                for i in range(self.K):
                    self_delta = delta_dict_list[i][key].reshape(1, -1)
                    cross_attn_delta = CrossAttention(self_delta, cross_delta, cross_delta)

                    beta = layer_beta[i]
                    ori_shape = delta_dict_list[i][key].shape
                    new_delta = delta_dict_list[i][key] + beta * cross_attn_delta.reshape(ori_shape)
                    new_param_dict_list[i][key] += new_delta

        return new_param_dict_list


def CrossAttention(q, k, v):
    scale = q.size(-1) ** -0.5
    attn = (q @ k.transpose(-2, -1)) * scale
    attn = nn.Softmax(dim=-1)(attn)
    out = attn @ v

    return out
