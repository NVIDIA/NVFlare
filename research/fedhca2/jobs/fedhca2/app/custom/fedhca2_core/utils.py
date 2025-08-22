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

import os
import random

import numpy as np
import torch
import torch.nn.functional as F

from .datasets.custom_transforms import get_transformations
from .datasets.utils.configs import NUM_TRAIN_IMAGES, TEST_SCALE, TRAIN_SCALE


def get_st_config(dataset_configs, local_rank=0):
    """
    Get single-task client configs
    """
    st_configs = {}
    for data_config in dataset_configs:
        dataname = data_config['dataname']
        train_transforms = get_transformations(TRAIN_SCALE[dataname], train=True)
        val_transforms = get_transformations(TEST_SCALE[dataname], train=False)

        # number of clients is defined in task_dict
        task_dict = data_config['task_dict']
        n_clients = sum(task_dict.values())
        if local_rank == 0:
            print("Training %d single-task models on %s" % (n_clients, dataname))

        task_list = []
        for task_name in task_dict:
            task_list += [task_name] * task_dict[task_name]

        # random partition of dataset
        idxs = np.random.permutation(NUM_TRAIN_IMAGES[dataname])
        batch_idxs = np.array_split(idxs, n_clients)
        net_task_dataidx_map = [{'task_list': [task_list[i]], 'dataidx': batch_idxs[i]} for i in range(n_clients)]

        st_configs[dataname] = data_config  # defined in yml
        st_configs[dataname]['n_clients'] = n_clients
        st_configs[dataname]['train_transforms'] = train_transforms
        st_configs[dataname]['val_transforms'] = val_transforms
        st_configs[dataname]['net_task_dataidx_map'] = net_task_dataidx_map

    return st_configs


def get_mt_config(dataset_configs, local_rank=0):
    """
    Get multi-task client configs
    """
    mt_configs = {}
    for data_config in dataset_configs:
        dataname = data_config['dataname']
        train_transforms = get_transformations(TRAIN_SCALE[dataname], train=True)
        val_transforms = get_transformations(TEST_SCALE[dataname], train=False)

        # number of models is defined in client_num
        n_clients = data_config['client_num']
        if local_rank == 0:
            print("Training %d multi-task models on %s" % (n_clients, dataname))

        task_dict = data_config['task_dict']
        task_list = []
        for task_name in task_dict:
            task_list += [task_name] * (task_dict[task_name] > 0)

        # random partition of dataset
        idxs = np.random.permutation(NUM_TRAIN_IMAGES[dataname])
        batch_idxs = np.array_split(idxs, n_clients)
        net_task_dataidx_map = [{'task_list': task_list, 'dataidx': batch_idxs[i]} for i in range(n_clients)]

        mt_configs[dataname] = data_config  # defined in yml
        mt_configs[dataname]['n_clients'] = n_clients
        mt_configs[dataname]['train_transforms'] = train_transforms
        mt_configs[dataname]['val_transforms'] = val_transforms
        mt_configs[dataname]['net_task_dataidx_map'] = net_task_dataidx_map

    return mt_configs


class RunningMeter(object):
    def __init__(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def create_dir(directory):
    """
    Create required directory if it does not exist
    """
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def create_results_dir(results_dir, exp_name):
    """
    Create required results directory if it does not exist
    """
    exp_dir = os.path.join(results_dir, exp_name)
    checkpoint_dir = os.path.join(exp_dir, 'checkpoints')
    create_dir(results_dir)
    create_dir(exp_dir)
    create_dir(checkpoint_dir)
    return exp_dir, checkpoint_dir


def create_pred_dir(results_dir, exp_name, all_nets):
    """
    Create required prediction directory if it does not exist
    """
    exp_dir = os.path.join(results_dir, exp_name)
    checkpoint_dir = os.path.join(exp_dir, 'checkpoints')
    pred_dir = os.path.join(exp_dir, 'predictions')
    create_dir(pred_dir)

    for idx in range(len(all_nets)):
        for task in all_nets[idx]['tasks']:
            task_dir = os.path.join(pred_dir, str(idx) + '_' + task)
            create_dir(task_dir)
            if task == 'edge':
                create_dir(os.path.join(task_dir, 'img'))

    return checkpoint_dir, pred_dir


def get_loss_metric(loss_meter, tasks, prefix, idx):
    """
    Get loss statistics
    """
    if len(tasks) == 1:
        mt = False
        statistics = {}
    else:
        mt = True
        statistics = {prefix + '/' + str(idx) + '_loss_sum': 0.0}

    for task in tasks:
        if mt:
            statistics[prefix + '/' + str(idx) + '_loss_sum'] += loss_meter[task].avg
        statistics[prefix + '/' + str(idx) + '_' + task] = loss_meter[task].avg
        loss_meter[task].reset()

    return statistics


def to_cuda(batch):
    """
    Move batch to GPU
    """
    if type(batch) is dict:
        out = {}
        for k, v in batch.items():
            if k == 'meta':
                out[k] = v
            else:
                out[k] = to_cuda(v)
        return out
    elif type(batch) is torch.Tensor:
        return batch.cuda(non_blocking=True)
    elif type(batch) is list:
        return [to_cuda(v) for v in batch]
    else:
        return batch


def get_output(output, task):
    """
    Get output prediction in the required range and format
    """
    if task in {'normals'}:
        output = output.permute(0, 2, 3, 1)
        output = (F.normalize(output, p=2, dim=3) + 1.0) * 255 / 2.0

    elif task in {'semseg', 'human_parts'}:
        output = output.permute(0, 2, 3, 1)
        _, output = torch.max(output, dim=3)

    elif task in {'edge'}:
        output = output.permute(0, 2, 3, 1)
        output = torch.sigmoid(output).squeeze(-1) * 255

    elif task in {'sal'}:
        output = output.permute(0, 2, 3, 1)
        output = F.softmax(output, dim=3)[:, :, :, 1] * 255

    elif task in {'depth'}:
        output.clamp_(min=0.0)
        output = output.permute(0, 2, 3, 1).squeeze(-1)

    else:
        raise NotImplementedError

    return output


def move_ckpt(ckpt_dict, device):
    """
    Move checkpoint tensors to device
    """
    if isinstance(ckpt_dict, list):
        for i in range(len(ckpt_dict)):
            for key in ckpt_dict[i].keys():
                ckpt_dict[i][key] = ckpt_dict[i][key].to(device)
    elif isinstance(ckpt_dict, dict):
        for key in ckpt_dict.keys():
            ckpt_dict[key] = ckpt_dict[key].to(device)
    return ckpt_dict


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
