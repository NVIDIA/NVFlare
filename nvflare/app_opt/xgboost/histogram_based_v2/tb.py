# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

import xgboost.callback


class TensorBoardCallback(xgboost.callback.TrainingCallback):
    def __init__(self, app_dir: str, tensorboard):
        xgboost.callback.TrainingCallback.__init__(self)
        self.train_writer = tensorboard.SummaryWriter(log_dir=os.path.join(app_dir, "train-auc/"))
        self.val_writer = tensorboard.SummaryWriter(log_dir=os.path.join(app_dir, "val-auc/"))

    def after_iteration(self, model, epoch: int, evals_log: xgboost.callback.TrainingCallback.EvalsLog):
        if not evals_log:
            return False

        for data, metric in evals_log.items():
            for metric_name, log in metric.items():
                score = log[-1][0] if isinstance(log[-1], tuple) else log[-1]
                if data == "train":
                    self.train_writer.add_scalar(metric_name, score, epoch)
                else:
                    self.val_writer.add_scalar(metric_name, score, epoch)
        return False
