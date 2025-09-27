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


class HumanPartsMeter(object):

    def __init__(self, ignore_index=255):
        self.n_parts = 7
        self.ignore_index = ignore_index
        self.tp = [0] * self.n_parts
        self.fp = [0] * self.n_parts
        self.fn = [0] * self.n_parts

    @torch.no_grad()
    def update(self, pred, gt):
        pred, gt = pred.squeeze(), gt.squeeze()
        valid = gt != self.ignore_index

        for i_part in range(self.n_parts):
            tmp_gt = gt == i_part
            tmp_pred = pred == i_part
            self.tp[i_part] += torch.sum(tmp_gt & tmp_pred & (valid)).item()
            self.fp[i_part] += torch.sum(~tmp_gt & tmp_pred & (valid)).item()
            self.fn[i_part] += torch.sum(tmp_gt & ~tmp_pred & (valid)).item()

    def reset(self):
        self.tp = [0] * self.n_parts
        self.fp = [0] * self.n_parts
        self.fn = [0] * self.n_parts

    def get_score(self):
        jac = [0] * self.n_parts
        for i_part in range(self.n_parts):
            jac[i_part] = float(self.tp[i_part]) / max(float(self.tp[i_part] + self.fp[i_part] + self.fn[i_part]), 1e-8)

        eval_result = {'mIoU': (np.mean(jac) * 100)}

        return eval_result
