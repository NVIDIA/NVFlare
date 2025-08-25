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
from fedhca2_core.losses import BalancedBCELoss


class EdgeMeter(object):

    def __init__(self, dataname, ignore_index=255):
        if dataname == 'pascalcontext':
            pos_weight = 0.95
        elif dataname == 'nyud':
            pos_weight = 0.95
        else:
            raise NotImplementedError

        self.loss = 0
        self.n = 0
        self.loss_function = BalancedBCELoss(pos_weight=pos_weight, ignore_index=ignore_index)
        self.ignore_index = ignore_index

    @torch.no_grad()
    def update(self, pred, gt):
        pred, gt = pred.squeeze(), gt.squeeze()
        valid_mask = gt != self.ignore_index
        pred = pred[valid_mask]
        gt = gt[valid_mask]

        pred = pred.float().squeeze() / 255.0
        loss = self.loss_function(pred, gt).item()
        numel = gt.numel()
        self.n += numel
        self.loss += numel * loss

    def reset(self):
        self.loss = 0
        self.n = 0

    def get_score(self):
        eval_dict = {'loss': (self.loss / self.n)}

        return eval_dict
