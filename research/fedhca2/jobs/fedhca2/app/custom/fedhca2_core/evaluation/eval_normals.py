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

import torch


def normalize_tensor(input_tensor, dim):
    norm = torch.norm(input_tensor, p='fro', dim=dim, keepdim=True)
    zero_mask = norm == 0
    norm[zero_mask] = 1
    out = input_tensor.div(norm)
    out[zero_mask.expand_as(out)] = 0
    return out


class NormalsMeter(object):

    def __init__(self, ignore_index=255):
        self.ignore_index = ignore_index
        self.sum_deg_diff = 0
        self.total = 0

    @torch.no_grad()
    def update(self, pred, gt):
        pred = pred.permute(0, 3, 1, 2)  # [B, C, H, W]
        pred = 2 * pred / 255 - 1
        valid_mask = (gt != self.ignore_index).all(dim=1)

        pred = normalize_tensor(pred, dim=1)
        gt = normalize_tensor(gt, dim=1)
        deg_diff = torch.rad2deg(2 * torch.atan2(torch.norm(pred - gt, dim=1), torch.norm(pred + gt, dim=1)))
        deg_diff = torch.masked_select(deg_diff, valid_mask)

        self.sum_deg_diff += torch.sum(deg_diff).item()
        self.total += deg_diff.numel()

    def get_score(self):
        eval_result = {'mErr': (self.sum_deg_diff / self.total)}

        return eval_result
