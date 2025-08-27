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
import torch.nn as nn
import torch.nn.functional as F

# Loss functions and hyperparameters
PASCAL_LOSS_CONFIG = {
    'semseg': {
        'loss_function': 'CELoss',
        'weight': 1
    },
    'human_parts': {
        'loss_function': 'CELoss',
        'weight': 2
    },
    'normals': {
        'loss_function': 'L1Loss',
        'parameters': {
            'normalize': True
        },
        'weight': 10
    },
    'sal': {
        'loss_function': 'CELoss',
        'parameters': {
            'balanced': True
        },
        'weight': 5
    },
    'edge': {
        'loss_function': 'BalancedBCELoss',
        'parameters': {
            'pos_weight': 0.95
        },
        'weight': 50
    }
}

NYUD_LOSS_CONFIG = {
    'semseg': {
        'loss_function': 'CELoss',
        'weight': 1
    },
    'normals': {
        'loss_function': 'L1Loss',
        'parameters': {
            'normalize': True
        },
        'weight': 10
    },
    'edge': {
        'loss_function': 'BalancedBCELoss',
        'parameters': {
            'pos_weight': 0.95
        },
        'weight': 50
    },
    'depth': {
        'loss_function': 'L1Loss',
        'weight': 1
    }
}


class BalancedBCELoss(nn.Module):
    # Edge Detection

    def __init__(self, pos_weight=0.95, ignore_index=255):
        super().__init__()
        self.pos_weight = pos_weight
        self.ignore_index = ignore_index

    def forward(self, output, label):
        mask = (label != self.ignore_index)
        masked_output = torch.masked_select(output, mask)  # 1-d tensor
        masked_label = torch.masked_select(label, mask)  # 1-d tensor

        # pos weight: w, neg weight: 1-w
        w = torch.tensor(self.pos_weight, device=output.device)
        factor = 1. / (1 - w)
        loss = F.binary_cross_entropy_with_logits(masked_output, masked_label, pos_weight=w * factor)
        loss /= factor

        return loss


class CELoss(nn.Module):
    # Semantic Segmentation, Human Parts Segmentation, Saliency Detection

    def __init__(self, balanced=False, ignore_index=255):
        super(CELoss, self).__init__()
        self.ignore_index = ignore_index
        self.balanced = balanced

    def forward(self, output, label):
        label = torch.squeeze(label, dim=1).long()

        if self.balanced:
            mask = (label != self.ignore_index)
            masked_label = torch.masked_select(label, mask)
            assert torch.max(masked_label) < 2  # binary

            num_labels_neg = torch.sum(1.0 - masked_label)
            num_total = torch.numel(masked_label)
            pos_weight = num_labels_neg / num_total
            class_weight = torch.stack((1. - pos_weight, pos_weight), dim=0)
            loss = F.cross_entropy(output, label, weight=class_weight, ignore_index=self.ignore_index, reduction='sum')
        else:
            loss = F.cross_entropy(output, label, ignore_index=self.ignore_index, reduction='sum')

        n_valid = (label != self.ignore_index).sum()
        loss /= max(n_valid, 1)

        return loss


class L1Loss(nn.Module):
    # Normals Estimation, Depth Estimation

    def __init__(self, normalize=False, ignore_index=255):
        super(L1Loss, self).__init__()
        self.normalize = normalize
        self.ignore_index = ignore_index

    def forward(self, output, label):
        if self.normalize:
            # Normalize to unit vector
            output = F.normalize(output, p=2, dim=1)

        mask = (label != self.ignore_index).all(dim=1, keepdim=True)
        masked_output = torch.masked_select(output, mask)
        masked_label = torch.masked_select(label, mask)

        loss = F.l1_loss(masked_output, masked_label, reduction='sum')
        n_valid = torch.sum(mask).item()
        loss /= max(n_valid, 1)

        return loss


def get_loss_functions(task_loss_config):
    """
    Get loss function for each task
    """
    key2loss = {
        "CELoss": CELoss,
        "BalancedBCELoss": BalancedBCELoss,
        "L1Loss": L1Loss,
    }

    # Get loss function for each task
    loss_fx = key2loss[task_loss_config['loss_function']]
    if 'parameters' in task_loss_config:
        loss_ft = loss_fx(**task_loss_config['parameters'])
    else:
        loss_ft = loss_fx()

    return loss_ft


class MultiTaskLoss(nn.Module):
    """
    Multi-Task loss with different loss functions and weights
    """
    def __init__(self, tasks, loss_ft, loss_weights):
        super(MultiTaskLoss, self).__init__()
        assert (set(tasks) == set(loss_ft.keys()))
        assert (set(tasks) == set(loss_weights.keys()))
        self.tasks = tasks
        self.loss_ft = loss_ft
        self.loss_weights = loss_weights

    def forward(self, pred, gt, tasks):
        out = {task: self.loss_ft[task](pred[task], gt[task]) for task in tasks}
        out['total'] = torch.sum(torch.stack([self.loss_weights[t] * out[t] for t in tasks]))

        return out


def get_criterion(dataname, tasks):
    if dataname == 'pascalcontext':
        losses_config = PASCAL_LOSS_CONFIG
    elif dataname == 'nyud':
        losses_config = NYUD_LOSS_CONFIG
    else:
        raise NotImplementedError

    loss_ft = torch.nn.ModuleDict({task: get_loss_functions(losses_config[task]) for task in tasks})
    loss_weights = {task: losses_config[task]['weight'] for task in tasks}

    return MultiTaskLoss(tasks, loss_ft, loss_weights)
