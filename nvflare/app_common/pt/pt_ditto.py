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


class PTDittoHelper(object):
    """Helper to be used with Ditto components.
    Implements the functions used for the algorithm proposed in
    Li et al. "Ditto: Fair and Robust Federated Learning Through Personalization"
    (https://arxiv.org/abs/2012.04221) using PyTorch.
    """

    def __init__(self):
        # Ditto control parameters and personalized model settings:
        # optimization parameter
        self.ditto_lr = None
        self.ditto_lambda = None
        self.ditto_criterion = None
        self.ditto_prox_criterion = None
        # personalized model and optimizer
        self.ditto_model = None
        self.ditto_optimizer = None
        # personalized model training epoch
        self.ditto_model_epochs = None
        self.ditto_epoch_global = None
        self.ditto_epoch_of_start_time = None
        # save to local file
        self.ditto_best_metric = None
        self.ditto_model_file_path = None
        self.best_ditto_model_file_path = None
        # device
        self.device = None

    def init(self, app_dir):
        self.ditto_model_file_path = os.path.join(app_dir, "personalized_model.pt")
        self.best_ditto_model_file_path = os.path.join(app_dir, "best_personalized_model.pt")

    def load_ditto_model(self, global_weights):
        # load local model from last round's record if model exist,
        # otherwise initialize from global model weights for the first round.
        if os.path.exists(self.ditto_model_file_path):
            model_data = torch.load(self.ditto_model_file_path)
            self.ditto_model.load_state_dict(model_data["model"])
            self.ditto_epoch_of_start_time = model_data["epoch"]
        else:
            self.ditto_model.load_state_dict(global_weights)
            self.ditto_epoch_of_start_time = 0
        if os.path.exists(self.best_ditto_model_file_path):
            model_data = torch.load(self.best_ditto_model_file_path)
            self.ditto_best_metric = model_data["best_metric"]

    def save_ditto_model(self, is_best=False):
        # save personalized model locally
        model_weights = self.ditto_model.state_dict()
        save_dict = {"model": model_weights, "epoch": self.ditto_epoch_global}
        if is_best:
            save_dict.update({"best_metric": self.ditto_best_metric})
            torch.save(save_dict, self.best_ditto_model_file_path)
        else:
            torch.save(save_dict, self.ditto_model_file_path)

    def local_train_ditto(self, train_loader, model_global, abort_signal: Signal, writer):
        # Train personal model for self.ditto_model_epochs, and keep track of curves
        for epoch in range(self.ditto_model_epochs):
            if abort_signal.triggered:
                return make_reply(ReturnCode.TASK_ABORTED)
            self.ditto_model.train()
            epoch_len = len(train_loader)
            self.ditto_epoch_global = self.ditto_epoch_of_start_time + epoch + 1
            for i, batch_data in enumerate(train_loader):
                if abort_signal.triggered:
                    return make_reply(ReturnCode.TASK_ABORTED)
                inputs = batch_data["image"].to(self.device)
                labels = batch_data["label"].to(self.device)

                # forward + backward + optimize
                outputs = self.ditto_model(inputs)
                loss = self.ditto_criterion(outputs, labels)

                # add the Ditto prox loss term for Ditto
                ditto_loss = self.ditto_prox_criterion(self.ditto_model, model_global)
                loss += ditto_loss

                self.ditto_optimizer.zero_grad()
                loss.backward()
                self.ditto_optimizer.step()

                current_step = epoch_len * self.ditto_epoch_global + i
                writer.add_scalar("train_loss_ditto", loss.item(), current_step)
