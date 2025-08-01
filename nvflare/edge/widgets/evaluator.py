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
import importlib
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContext
from nvflare.app_common.abstract.model import ModelLearnableKey
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.app_event_type import AppEventType
from nvflare.widgets.widget import Widget

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GlobalEvaluator(Widget):
    def __init__(
        self,
        model_path: Union[str, nn.Module],
        eval_frequency: int = 1,
        torchvision_dataset: Optional[Dict] = None,
        custom_dataset: Optional[Dict] = None,
    ):
        """Initialize the evaluator with either a dataset path or custom dataset.

        Args:
            model_path: PyTorch model to evaluate
            torchvision_dataset: Torchvision dataset (for standard datasets like CIFAR10)
            custom_dataset: Dictionary containing 'data' and 'labels' tensors
        """
        super().__init__()
        if torchvision_dataset is None and custom_dataset is None:
            raise ValueError("Must provide either torchvision_dataset or custom_dataset")
        if torchvision_dataset is not None and custom_dataset is not None:
            raise ValueError("Cannot provide both torchvision_dataset and custom_dataset")

        if isinstance(model_path, nn.Module):
            pass
        elif not isinstance(model_path, str):
            raise ValueError(f"model_path must be either a Pytorch model or class path, but got {type(model_path)}")

        self.model_path = model_path
        self.eval_frequency = eval_frequency
        self.torchvision_dataset = torchvision_dataset
        self.custom_dataset = custom_dataset
        self.batch_size = 4
        self.model = None
        self.data_loader = None
        self.tb_writer = None

        self.register_event_handler(EventType.START_RUN, self._initialize)
        self.register_event_handler(AppEventType.GLOBAL_WEIGHTS_UPDATED, self.evaluate)

    def _load_model(self, model_path: str, fl_ctx: FLContext) -> Any:
        """Load model from model path.

        Args:
            model_path (str): model path in format "module.submodule.ClassName"
            fl_ctx (FLContext): FL context

        Returns:
            model class instance
        """
        try:
            module_path, class_name = model_path.rsplit(".", 1)
            module = importlib.import_module(module_path)
            model_class = getattr(module, class_name)
            return model_class
        except Exception as e:
            self.system_panic(
                reason=f"Failed to load model class from '{model_path}': {str(e)}",
                fl_ctx=fl_ctx,
            )
            return None

    def _create_data_loader(self):
        if self.torchvision_dataset is not None:
            # Define transform
            transform = transforms.Compose([transforms.ToTensor()])
            # For torchvision datasets (e.g., CIFAR10)
            # first check the keys of "name" and "path"
            if "name" not in self.torchvision_dataset or "path" not in self.torchvision_dataset:
                raise ValueError("torchvision_dataset must contain 'name' and 'path' keys")
            dataset_name = self.torchvision_dataset["name"]
            dataset_path = self.torchvision_dataset["path"]
            # then check if the dataset_name is a valid torchvision dataset
            if dataset_name not in torchvision.datasets.__dict__:
                raise ValueError(f"Invalid torchvision dataset: {dataset_name}")
            # then get the dataset class
            dataset_class = getattr(torchvision.datasets, dataset_name)
            test_set = dataset_class(root=dataset_path, train=False, download=True, transform=transform)
            # then create the data loader
            self.data_loader = torch.utils.data.DataLoader(
                test_set, batch_size=self.batch_size, shuffle=False, num_workers=2
            )
        else:
            # For custom datasets (e.g., XOR)
            data = self.custom_dataset["data"]
            labels = self.custom_dataset["label"]
            # Convert data and labels from list to tensors
            data = torch.tensor(data, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.long)
            if len(data) != len(labels):
                raise ValueError("data and labels must have the same length")
            # Create a TensorDataset
            dataset = torch.utils.data.TensorDataset(data, labels)
            # if length of data smaller than batch size, set batch size to length of data
            if len(data) < self.batch_size:
                self.batch_size = len(data)
            self.data_loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size)

    def _eval_model(self) -> Dict[str, float]:
        self.model.to(DEVICE)
        self.model.eval()

        correct = 0
        total = 0
        with torch.no_grad():
            for data, labels in self.data_loader:
                data, labels = data.to(DEVICE), labels.to(DEVICE)
                outputs = self.model(data)
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        accuracy = 100 * correct / total
        return {"accuracy": accuracy}

    def _initialize(self, _event_type: str, fl_ctx: FLContext):
        # Initialize the model
        if isinstance(self.model_path, str):
            # load the model
            model_class = self._load_model(self.model_path, fl_ctx)
            self.model = model_class()
        else:
            # the model_path is nn.Module
            self.model = self.model_path

        # Initialize the data loader
        self._create_data_loader()
        # Initialize the tensorboard writer
        app_root = fl_ctx.get_prop(FLContextKey.APP_ROOT)
        self.tb_writer = SummaryWriter(log_dir=app_root)

    def evaluate(self, _event_type: str, fl_ctx: FLContext):
        global_model = fl_ctx.get_prop(AppConstants.GLOBAL_MODEL)
        current_round = fl_ctx.get_prop(AppConstants.CURRENT_ROUND)
        # Load the model weights
        global_weights = global_model[ModelLearnableKey.WEIGHTS]
        # Convert weights from list to torch tensors
        global_weights = {k: torch.tensor(v) for k, v in global_weights.items()}
        self.model.load_state_dict(global_weights)
        # Evaluate the model according to the evaluation frequency
        if current_round % self.eval_frequency == 0:
            metrics = self._eval_model()
            for key, value in metrics.items():
                self.tb_writer.add_scalar(key, value, current_round)
