# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
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
import torch.optim as optim
from monai.losses import DiceLoss
from monai.networks.nets.unet import UNet

from nvflare.apis.fl_context import FLContext
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.pt.pt_fedproxloss import PTFedProxLoss
from nvflare.apis.signal import Signal

from pt.learners.supervised_fedsm_learner import SupervisedFedSMLearner
from pt.learners.supervised_prostate_learner import SupervisedProstateLearner
from pt.networks.vgg import vgg11

import numpy as np


def Accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class ProstateFedSMLearner(SupervisedFedSMLearner, SupervisedProstateLearner):
    def __init__(
        self,
        train_config_filename,
        select_num_classes,
        aggregation_epochs: int = 1,
        train_task_name: str = AppConstants.TASK_TRAIN,
        submit_model_task_name: str = AppConstants.TASK_SUBMIT_MODEL,
    ):
        """Trainer for prostate segmentation task. It inherits from MONAI trainer.

        Args:
            train_config_filename: directory of config file.
            train_task_name: name of the task to train the model.
            aggregation_epochs: the number of training epochs of global model for a round. Defaults to 1.
            select_num_classes: the number of classes for selector model. Defaults to 1.
            submit_model_task_name: name of the task to submit the best local model.

        Returns:
            a Shareable with the updated local model after running `execute()`
            or the best local model depending on the specified task.
        """
        super().__init__(
            train_config_filename=train_config_filename,
            aggregation_epochs=aggregation_epochs,
            train_task_name=train_task_name,
            submit_model_task_name=submit_model_task_name,
        )
        self.select_num_classes = select_num_classes

    def _extra_train_config(self, fl_ctx: FLContext, config_info: str):
        # Get the config_info
        super()._extra_train_config(fl_ctx, config_info)
        # FedSM and prostate specific
        self.lr_select = config_info["learning_rate_select"]
        self.bs_slice_select = 12

        # Additional personalized and select model/criterion/optimizer
        self.model_person = UNet(
            dimensions=3,
            in_channels=1,
            out_channels=1,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
        ).to(self.device)
        self.criterion_person = DiceLoss(sigmoid=True)
        self.optimizer_person = optim.SGD(self.model_person.parameters(), lr=self.lr, momentum=0.9)

        self.model_select = vgg11(
            num_classes=self.select_num_classes,
        ).to(self.device)
        self.criterion_select = torch.nn.CrossEntropyLoss()
        self.optimizer_select = optim.SGD(self.model_select.parameters(), lr=self.lr_select, momentum=0.9)

    def local_train(
        self,
        fl_ctx,
        train_loader,
        abort_signal: Signal,
        select_label,
    ):
        """
        three model trainings
        """
        for epoch in range(self.aggregation_epochs):
            if abort_signal.triggered:
                return
            self.model.train()
            self.model_person.train()
            self.model_select.train()

            epoch_len = len(train_loader)
            self.epoch_global = self.epoch_of_start_time + epoch
            self.log_info(
                fl_ctx,
                f"Local epoch {self.client_id}: {epoch + 1}/{self.aggregation_epochs} (lr_main={self.lr}, lr_select={self.lr_select})",
            )

            for i, batch_data in enumerate(train_loader):
                if abort_signal.triggered:
                    return
                # input and expected output
                images = batch_data["image"].to(self.device)
                labels = batch_data["label"].to(self.device)
                # pick select_bs of 2D slices at random location as input to 2D selector model
                slice_ct = images.size()[-1]
                rand_idx = np.random.randint(slice_ct, size=self.bs_slice_select)
                rand_idx = torch.tensor(rand_idx).to(self.device)
                images_select = torch.index_select(images, -1, rand_idx)
                # re-organize: move last axis (slices) to first, and merge with original batch
                images_select = torch.moveaxis(images_select, -1, 0)
                images_select = images_select.flatten(0, 1)
                # generate label vector: image batch_size * slice batch_size, same label
                labels_select = np.ones(images.size()[0]*self.bs_slice_select) * select_label
                labels_select = torch.tensor(labels_select, dtype=torch.int64).to(self.device)

                # zero the parameter gradients
                self.optimizer.zero_grad()
                self.optimizer_person.zero_grad()
                self.optimizer_select.zero_grad()

                # forward + backward + optimize
                outputs = self.model(images)
                outputs_person = self.model_person(images)
                outputs_select = self.model_select(images_select)

                loss = self.criterion(outputs, labels)
                loss_person = self.criterion_person(outputs_person, labels)
                loss_select = self.criterion_select(outputs_select, labels_select)

                loss.backward()
                loss_person.backward()
                loss_select.backward()

                self.optimizer.step()
                self.optimizer_person.step()
                self.optimizer_select.step()

                current_step = epoch_len * self.epoch_global + i
                self.writer.add_scalar("train_loss_global", loss.item(), current_step)
                self.writer.add_scalar("train_loss_personalized", loss_person.item(), current_step)
                self.writer.add_scalar("train_loss_selector", loss_select.item(), current_step)
                
    def local_valid(self, valid_loader, abort_signal: Signal, select_label=None, tb_id=None):
        self.model.eval()
        self.model_person.eval()
        if select_label is not None:
            self.model_select.eval()
        with torch.no_grad():
            metric_score_global = 0
            metric_score_person = 0
            metric_score_select = 0
            for i, batch_data in enumerate(valid_loader):
                if abort_signal.triggered:
                    return self._abort_execution()
                # input and expected output
                images = batch_data["image"].to(self.device)
                labels = batch_data["label"].to(self.device)
                # inference
                outputs_global = self.inferer(images, self.model)
                outputs_global = self.transform_post(outputs_global)
                outputs_person = self.inferer(images, self.model_person)
                outputs_person = self.transform_post(outputs_person)
                # compute metric
                metric = self.valid_metric(y_pred=outputs_global, y=labels)
                metric_score_global += metric.item()
                metric = self.valid_metric(y_pred=outputs_person, y=labels)
                metric_score_person += metric.item()

                # validate the selector model
                if select_label is not None:
                    # pick select_bs of 2D slices at random location as input to 2D selector model
                    slice_ct = images.size()[-1]
                    rand_idx = np.random.randint(slice_ct, size=self.bs_slice_select)
                    rand_idx = torch.tensor(rand_idx).to(self.device)
                    images_select = torch.index_select(images, -1, rand_idx)
                    # re-organize: move last axis (slices) to first, and merge with original batch
                    images_select = torch.moveaxis(images_select, -1, 0)
                    images_select = images_select.flatten(0, 1)

                    # generate label vector: image batch_size * slice batch_size, same label
                    select = np.ones(images.size()[0]*self.bs_slice_select) * select_label
                    select = torch.tensor(select).to(self.device)

                    # inference
                    outputs_select = self.model_select(images_select)

                    # compute metric
                    metric = Accuracy(outputs_select, select, topk=(1, ))
                    metric_score_select += metric[0].item()
                else:
                    metric_score_select = 0

            # compute mean dice/acc over whole validation set
            metric_score_global /= len(valid_loader)
            metric_score_person /= len(valid_loader)
            metric_score_select /= len(valid_loader)

            if tb_id:
                self.writer.add_scalar(tb_id + "_global", metric_score_global, self.epoch_of_start_time)
                self.writer.add_scalar(tb_id + "_personalized", metric_score_person, self.epoch_of_start_time)
                self.writer.add_scalar(tb_id + "_selector", metric_score_select, self.epoch_of_start_time)
        return metric_score_global, metric_score_person, metric_score_select
