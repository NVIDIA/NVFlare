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


import torch.optim as optim
from monai.losses import DiceLoss
from monai.networks.nets.unet import UNet

from nvflare.apis.fl_context import FLContext
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.pt.pt_fedproxloss import PTFedProxLoss

from pt.learners.supervised_ditto_learner import SupervisedDittoLearner
from pt.learners.supervised_prostate_learner import SupervisedProstateLearner

class ProstateDittoLearner(SupervisedDittoLearner, SupervisedProstateLearner):
    def __init__(
        self,
        train_config_filename,
        aggregation_epochs: int = 1,
        local_model_epochs: int = 1,
        train_task_name: str = AppConstants.TASK_TRAIN,
        submit_model_task_name: str = AppConstants.TASK_SUBMIT_MODEL,
    ):
        """Trainer for prostate segmentation task. It inherits from MONAI trainer.

        Args:
            train_config_filename: directory of config file.
            aggregation_epochs: the number of training epochs of global model for a round. Defaults to 1.
            local_model_epochs: the number of training epochs of local model for a round. Defaults to 1.
            train_task_name: name of the task to train the model.
            submit_model_task_name: name of the task to submit the best local model.

        Returns:
            a Shareable with the updated local model after running `execute()`
            or the best local model depending on the specified task.
        """
        super().__init__(
            train_config_filename=train_config_filename,
            aggregation_epochs=aggregation_epochs,
            local_model_epochs=local_model_epochs,
            train_task_name=train_task_name,
            submit_model_task_name=submit_model_task_name,
        )

    def _extra_train_config(self, fl_ctx: FLContext, config_info: str):
        # Get the config_info
        super()._extra_train_config(fl_ctx, config_info)
        # Ditto specific
        self.ditto_lr_ref = config_info["ref_learning_rate"]
        self.ditto_lambda = config_info["ditto_lambda"]

        # Additional ref model/criterion/optimizer
        self.model_ref = UNet(
            dimensions=3,
            in_channels=1,
            out_channels=1,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
        ).to(self.device)
        self.criterion_ref = DiceLoss(sigmoid=True)
        self.ditto_criterion_prox = PTFedProxLoss(mu=self.ditto_lambda)
        self.optimizer_ref = optim.SGD(self.model_ref.parameters(), lr=self.ditto_lr_ref, momentum=0.9)
