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
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContext
from nvflare.app_common.abstract.model import ModelLearnableKey
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.app_event_type import AppEventType
from nvflare.edge.models.model import Cifar10ConvNet
from nvflare.widgets.widget import Widget

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GlobalEvaluator(Widget):
    def __init__(self, data_root: str):
        super().__init__()
        self.model = Cifar10ConvNet()
        self.tb_writer = None
        self.data_root = data_root
        self.criterion = nn.CrossEntropyLoss()
        self.test_loader = None

        self.register_event_handler(AppEventType.AFTER_SHAREABLE_TO_LEARNABLE, self.evaluate)
        self.register_event_handler(AppEventType.TRAINING_STARTED, self.initiate_tb)

    def _create_data_loader(self):
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

        test_set = torchvision.datasets.CIFAR10(root=self.data_root, train=False, download=True, transform=transform)
        self.test_loader = torch.utils.data.DataLoader(
            test_set, batch_size=self.batch_size, shuffle=False, num_workers=2
        )

    def _eval_model(self) -> dict:
        if self.test_loader is None:
            self._create_data_loader()

        self.model.to(DEVICE)
        self.model.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        avg_loss = total_loss / len(self.test_loader)

        return {"accuracy": accuracy, "loss": avg_loss}

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
        metrics = self._eval_model()
        for key, value in metrics.items():
            self.tb_writer.add_scalar(key, value, current_round)
