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

from ..utils import get_output
from .save_img import save_img


class PerformanceMeter(object):
    """
    A general performance meter which shows performance across one or more tasks
    """

    def __init__(self, dataname, tasks):
        self.tasks = tasks
        self.meters = {t: get_single_task_meter(dataname, t) for t in self.tasks}

    def reset(self):
        for t in self.tasks:
            self.meters[t].reset()

    def update(self, pred, gt):
        for t in self.tasks:
            self.meters[t].update(pred[t], gt[t])

    def get_score(self):
        eval_dict = {}
        for t in self.tasks:
            eval_dict[t] = self.meters[t].get_score()

        return eval_dict


def get_single_task_meter(dataname, task):
    """
    Retrieve a meter to measure the single-task performance
    """

    if task == 'semseg':
        from .eval_semseg import SemsegMeter

        return SemsegMeter(dataname)

    elif task == 'human_parts':
        from .eval_human_parts import HumanPartsMeter

        return HumanPartsMeter()

    elif task == 'normals':
        from .eval_normals import NormalsMeter

        return NormalsMeter()

    elif task == 'sal':
        from .eval_sal import SaliencyMeter

        return SaliencyMeter()

    elif task == 'depth':
        from .eval_depth import DepthMeter

        return DepthMeter(dataname)

    elif task == 'edge':  # Single task performance meter uses the loss (True evaluation is based on seism evaluation)
        from .eval_edge import EdgeMeter

        return EdgeMeter(dataname)

    else:
        raise NotImplementedError


def predict(dataname, meta, outputs, task, pred_dir, idx):
    """
    Get predictions and save predicted images
    :param str dataname: Dataset name
    :param dict meta: Metadata from the dataset, containing image names and sizes
    :param dict outputs: Model outputs
    :param str task: Task name
    :param str pred_dir: Directory to save the predictions
    """

    output_task = get_output(outputs[task], task)
    preds = []
    for i in range(output_task.size(0)):
        # Cut image borders (padding area)
        pred = output_task[i]  # H, W or H, W, C
        ori_dim = (int(meta['size'][i][0]), int(meta['size'][i][1]))
        curr_dim = tuple(pred.shape[:2])

        if ori_dim != curr_dim:
            # Height and width of border
            delta_h = max(curr_dim[0] - ori_dim[0], 0)
            delta_w = max(curr_dim[1] - ori_dim[1], 0)

            # Location of original image
            loc_h = [delta_h // 2, (delta_h // 2) + ori_dim[0]]
            loc_w = [delta_w // 2, (delta_w // 2) + ori_dim[1]]

            pred = pred[loc_h[0] : loc_h[1], loc_w[0] : loc_w[1]]

        pred = pred.cpu().numpy()
        preds.append(pred)

    save_img(dataname, meta['file_name'], preds, task, pred_dir, str(idx))
