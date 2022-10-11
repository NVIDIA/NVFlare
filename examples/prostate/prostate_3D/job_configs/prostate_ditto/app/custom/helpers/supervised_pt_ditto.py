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


from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.shareable import make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.pt.pt_ditto import PTDittoHelper


class SupervisedPTDittoHelper(PTDittoHelper):
    """Helper to be used with Ditto components under supervised training specs."""

    def __init__(self, criterion, model, optimizer, device, app_dir, ditto_lambda, model_epochs):
        super().__init__(criterion, model, optimizer, device, app_dir, ditto_lambda, model_epochs)

    def local_train(self, train_loader, model_global, abort_signal: Signal, writer):
        # Train personal model for self.model_epochs, and keep track of curves
        # This part is task dependent, need customization
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
                loss_ditto = self.prox_criterion(self.model, model_global)
                loss += loss_ditto

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                current_step = epoch_len * self.epoch_global + i
                writer.add_scalar("train_loss_ditto", loss.item(), current_step)
