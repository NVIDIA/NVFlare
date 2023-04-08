# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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
from abc import abstractmethod

import numpy as np
import torch

from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.shareable import make_reply
from nvflare.apis.signal import Signal


def AccuracyTopK(output, target, topk=(1,)):
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


class PTFedSMHelper(object):
    def __init__(
        self,
        person_model,
        select_model,
        person_criterion,
        select_criterion,
        person_optimizer,
        select_optimizer,
        device,
        app_dir,
        person_model_epochs: int = 1,
        select_model_epochs: int = 1,
    ):
        """Helper to be used with FedSM components.
        Implements the functions used for the algorithm proposed in
        Xu et al. "Closing the Generalization Gap of Cross-silo Federated Medical Image Segmentation"
        (https://arxiv.org/abs/2203.10144) using PyTorch.

        Args:
            person/select_model: the personalized and selector models
            person/select_criterion: loss criterion
            person/select_optimizer: training optimizer the two models
            device: device for model training
            app_dir: needed for local personalized model saving
            person/select_model_epochs: total training epochs each round

        Returns:
            None
        """

        self.person_model = person_model
        self.select_model = select_model
        self.person_criterion = person_criterion
        self.select_criterion = select_criterion
        self.person_optimizer = person_optimizer
        self.select_optimizer = select_optimizer
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        self.person_model_epochs = person_model_epochs
        self.select_model_epochs = select_model_epochs
        # check criterion, model, and optimizer type
        if not isinstance(self.person_model, torch.nn.Module):
            raise ValueError(f"person_model component must be torch model. But got: {type(self.person_model)}")
        if not isinstance(self.select_model, torch.nn.Module):
            raise ValueError(f"select_model component must be torch model. But got: {type(self.select_model)}")
        if not isinstance(self.person_criterion, torch.nn.modules.loss._Loss):
            raise ValueError(f"person_criterion component must be torch loss. But got: {type(self.person_criterion)}")
        if not isinstance(self.select_criterion, torch.nn.modules.loss._Loss):
            raise ValueError(f"select_criterion component must be torch loss. But got: {type(self.select_criterion)}")
        if not isinstance(self.person_optimizer, torch.optim.Optimizer):
            raise ValueError(
                f"person_optimizer component must be torch optimizer. But got: {type(self.person_optimizer)}"
            )
        if not isinstance(self.select_optimizer, torch.optim.Optimizer):
            raise ValueError(
                f"select_optimizer component must be torch optimizer. But got: {type(self.select_optimizer)}"
            )
        if not isinstance(self.device, torch.device):
            raise ValueError(f"device component must be torch device. But got: {type(self.device)}")

        # initialize other recording related parameters
        # save personalized model to local file
        # note: global and selector model saved on server
        self.person_best_metric = 0
        self.person_model_file_path = os.path.join(app_dir, "personalized_model.pt")
        self.best_person_model_file_path = os.path.join(app_dir, "best_personalized_model.pt")

    def save_person_model(self, current_round, is_best=False):
        # save personalized model locally
        model_weights = self.person_model.state_dict()
        save_dict = {"model": model_weights, "epoch": current_round}
        if is_best:
            save_dict.update({"best_metric": self.person_best_metric})
            torch.save(save_dict, self.best_person_model_file_path)
        else:
            torch.save(save_dict, self.person_model_file_path)

    def local_train_select(self, train_loader, select_label, abort_signal: Signal, writer, current_round):
        # Train selector model in full batch manner, and keep track of curves
        for epoch in range(self.select_model_epochs):
            if abort_signal.triggered:
                return make_reply(ReturnCode.TASK_ABORTED)
            self.select_model.train()
            epoch_len = len(train_loader)
            epoch_global = current_round * self.select_model_epochs + epoch
            for i, batch_data in enumerate(train_loader):
                if abort_signal.triggered:
                    return make_reply(ReturnCode.TASK_ABORTED)
                inputs = batch_data["image"].to(self.device)
                # construct vector of selector label
                labels = np.ones(inputs.size()[0], dtype=np.int64) * select_label
                labels = torch.tensor(labels).to(self.device)
                # forward + backward
                outputs = self.select_model(inputs)
                loss = self.select_criterion(outputs, labels)
                loss.backward()
                current_step = epoch_len * epoch_global + i
                writer.add_scalar("train_loss_selector", loss.item(), current_step)
            # Full batch training, 1 step per epoch
            self.select_optimizer.step()
            self.select_optimizer.zero_grad()

    def local_valid_select(
        self,
        valid_loader,
        select_label,
        abort_signal: Signal,
        tb_id=None,
        writer=None,
        current_round=None,
    ):
        # Validate selector model
        self.select_model.eval()
        with torch.no_grad():
            metric = 0
            for i, batch_data in enumerate(valid_loader):
                if abort_signal.triggered:
                    return make_reply(ReturnCode.TASK_ABORTED)
                # input and expected output
                images = batch_data["image"].to(self.device)
                # generate label vector: image batch_size, same label
                select = np.ones(images.size()[0], dtype=np.int64) * select_label
                select = torch.tensor(select).to(self.device)
                # inference
                outputs = self.select_model(images)
                # compute metric
                metric_score = AccuracyTopK(outputs, select, topk=(1,))
                metric += metric_score[0].item()
            # compute mean acc over whole validation set
            metric /= len(valid_loader)
            # tensorboard record id, add to record if provided
            if tb_id:
                writer.add_scalar(tb_id, metric, current_round)
        return metric

    def update_metric_save_person_model(self, current_round, metric):
        self.save_person_model(current_round, is_best=False)
        if metric > self.person_best_metric:
            self.person_best_metric = metric
            self.save_person_model(current_round, is_best=True)
            return 1
        else:
            return 0

    @abstractmethod
    def local_train_person(self, train_loader, abort_signal: Signal, writer):
        # Train personal model for self.model_epochs, and keep track of curves
        # This part is task dependent, need customization
        raise NotImplementedError
