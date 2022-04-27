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

import os

import torch

from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.shareable import make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.pt.pt_fedproxloss import PTFedProxLoss


class PTDittoHelper(object):
    """Helper to be used with Ditto components.
    Implements the functions used for the algorithm proposed in
    Li et al. "Ditto: Fair and Robust Federated Learning Through Personalization"
    (https://arxiv.org/abs/2012.04221) using PyTorch.
    """

    def __init__(self, ditto_lambda, criterion, model, optimizer, model_epochs, app_dir, device):
        self.criterion = criterion
        self.model = model
        self.optimizer = optimizer
        self.model_epochs = model_epochs
        self.device = device
        # initialize Ditto criterion
        self.prox_criterion = PTFedProxLoss(mu=ditto_lambda)
        # initialize other recording related parameters
        self.epoch_global = 0
        self.epoch_of_start_time = 0
        self.best_metric: int = 0
        self.model_file_path = os.path.join(app_dir, "personalized_model.pt")
        self.best_model_file_path = os.path.join(app_dir, "best_personalized_model.pt")

    def load_model(self, global_weights):
        # load local model from last round's record if model exist,
        # otherwise initialize from global model weights for the first round.
        if os.path.exists(self.model_file_path):
            model_data = torch.load(self.model_file_path)
            self.model.load_state_dict(model_data["model"])
            self.epoch_of_start_time = model_data["epoch"]
        else:
            self.model.load_state_dict(global_weights)
            self.epoch_of_start_time = 0
        if os.path.exists(self.best_model_file_path):
            model_data = torch.load(self.best_model_file_path)
            self.best_metric = model_data["best_metric"]

    def save_model(self, is_best=False):
        # save personalized model locally
        model_weights = self.model.state_dict()
        save_dict = {"model": model_weights, "epoch": self.epoch_global}
        if is_best:
            save_dict.update({"best_metric": self.best_metric})
            torch.save(save_dict, self.best_model_file_path)
        else:
            torch.save(save_dict, self.model_file_path)

    def local_train(self, train_loader, model_global, abort_signal: Signal, writer):
        # Train personal model for self.model_epochs, and keep track of curves
        for epoch in range(self.model_epochs):
            if abort_signal.triggered:
                return make_reply(ReturnCode.TASK_ABORTED)
            self.model.train()
            epoch_len = len(train_loader)
            self.epoch_global = self.epoch_of_start_time + epoch + 1
            for i, batch_data in enumerate(train_loader):
                if abort_signal.triggered:
                    return make_reply(ReturnCode.TASK_ABORTED)
                inputs = batch_data["image"].to(self.device)
                labels = batch_data["label"].to(self.device)

                # forward + backward + optimize
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                # add the Ditto prox loss term for Ditto
                loss = self.prox_criterion(self.model, model_global)
                loss += loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                current_step = epoch_len * self.epoch_global + i
                writer.add_scalar("train_loss_ditto", loss.item(), current_step)

    def update_metric_save_model(self, metric):
        self.save_model(is_best=False)
        if metric > self.best_metric:
            self.best_metric = metric
            self.save_model(is_best=True)
