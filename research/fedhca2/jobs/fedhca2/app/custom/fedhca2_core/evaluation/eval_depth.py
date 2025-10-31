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

import numpy as np
import torch


class DepthMeter(object):
    def __init__(self, dataname):
        self.dataname = dataname
        self.total_rmses = 0.0
        self.abs_rel = 0.0
        self.n_valid = 0.0
        self.max_depth = 80.0
        self.min_depth = 0.0

    @torch.no_grad()
    def update(self, pred, gt):
        pred, gt = pred.squeeze(), gt.squeeze()

        # Determine valid mask
        mask = torch.logical_and(gt < self.max_depth, gt > self.min_depth)
        self.n_valid += mask.float().sum().item()  # Valid pixels per image

        # Only positive depth values are possible
        # pred = torch.clamp(pred, min=1e-9)
        gt[gt <= 0] = 1e-9
        pred[pred <= 0] = 1e-9

        rmse_tmp = torch.pow(gt[mask] - pred[mask], 2)
        self.total_rmses += rmse_tmp.sum().item()
        self.abs_rel += (torch.abs(gt[mask] - pred[mask]) / gt[mask]).sum().item()

    def reset(self):
        self.total_rmses = 0.0
        self.abs_rel = 0.0
        self.n_valid = 0.0

    def get_score(self):
        if self.dataname == 'nyud':
            eval_result = {'RMSE': np.sqrt(self.total_rmses / self.n_valid)}
        else:
            raise NotImplementedError

        return eval_result
