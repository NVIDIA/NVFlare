# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
from torch.utils.tensorboard import SummaryWriter

from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContext
from nvflare.app_common.abstract.model import ModelLearnableKey
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.app_event_type import AppEventType
from nvflare.edge.models.model import XorNet
from nvflare.widgets.widget import Widget

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GlobalEvaluator(Widget):
    def __init__(self):
        super().__init__()
        self.model = XorNet()
        self.tb_writer = None
        self.register_event_handler(AppEventType.AFTER_SHAREABLE_TO_LEARNABLE, self.evaluate)
        self.register_event_handler(AppEventType.TRAINING_STARTED, self.initiate_tb)

    def _eval_model(self) -> float:
        # XOR task
        test_data = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
        test_labels = torch.tensor([0, 1, 1, 0], dtype=torch.long)
        test_data, test_labels = test_data.to(DEVICE), test_labels.to(DEVICE)
        self.model.to(DEVICE)
        with torch.no_grad():
            outputs = self.model(test_data)
            # calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == test_labels).sum().item()
            total = test_labels.size(0)
            acc = 100 * correct / total
        return {"accuracy": acc}

    def initiate_tb(self, _event_type: str, fl_ctx: FLContext):
        # Initiate tensorboard at server
        app_root = fl_ctx.get_prop(FLContextKey.APP_ROOT)
        self.tb_writer = SummaryWriter(log_dir=app_root)

    def evaluate(self, _event_type: str, fl_ctx: FLContext):
        global_model = fl_ctx.get_prop(AppConstants.GLOBAL_MODEL)
        current_round = fl_ctx.get_prop(AppConstants.CURRENT_ROUND)
        # Load the model weights
        global_weights = global_model[ModelLearnableKey.WEIGHTS]
        # Convert numpy weights to torch weights
        global_weights = {k: torch.from_numpy(v) for k, v in global_weights.items()}
        self.model.load_state_dict(global_weights)
        # Evaluate the model
        metric = self._eval_model()
        for key, value in metric.items():
            self.tb_writer.add_scalar(key, value, current_round)
