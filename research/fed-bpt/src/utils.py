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

# Part of this code is adopted from BBT (https://github.com/txsun1997/Black-Box-Tuning)

# MIT License
#
# Copyright (c) 2022 Tianxiang Sun
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

REDUCE_FN_MAPPINGS = {"sum": torch.sum, "mean": torch.mean, "none": lambda x: x}


def hinge_loss(logit, target, margin, reduction="sum"):
    """
    Args:
        logit (torch.Tensor): (N, C, d_1, d_2, ..., d_K)
        target (torch.Tensor): (N, d_1, d_2, ..., d_K)
        margin (float):
    """
    target = target.unsqueeze(1)
    tgt_logit = torch.gather(logit, dim=1, index=target)
    loss = logit - tgt_logit + margin
    loss = torch.masked_fill(loss, loss < 0, 0)
    loss = torch.scatter(loss, dim=1, index=target, value=0)
    reduce_fn = REDUCE_FN_MAPPINGS[reduction]
    return reduce_fn(loss)


def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg
