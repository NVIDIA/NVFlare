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
from concurrent.futures import ThreadPoolExecutor
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
from nvflare.fuel.utils.validation_utils import check_positive_int
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
            eval_frequency: Frequency of evaluation (evaluate every N rounds)
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

        # Validate eval_frequency - positive integer
        check_positive_int("eval_frequency", eval_frequency)

        self.model_path = model_path
        self.eval_frequency = eval_frequency
        self.torchvision_dataset = torchvision_dataset
        self.custom_dataset = custom_dataset
        self.batch_size = 4
        self.model = None
        self.data_loader = None
        self.tb_writer = None

        # Initialize thread pool, single worker to ensure evaluations are sequential
        # to avoid model version conflicts / extra GPU memory usage to sync multiple models
        self._thread_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="GlobalEvaluator")

        # Register event handlers
        self.register_event_handler(EventType.START_RUN, self._initialize)
        self.register_event_handler(AppEventType.GLOBAL_WEIGHTS_UPDATED, self.evaluate)
        self.register_event_handler(EventType.END_RUN, self._handle_end_run)

    def _load_model(self, model_path: str, fl_ctx: FLContext) -> Any:
        """Load model from model path.

        Args:
            model_path (str): model path in format "module.submodule.ClassName"
            fl_ctx (FLContext): FL context

        Returns:
            model class instance
        """
        try:
            self.logger.info(f"Loading model class from path: {model_path}")
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
            # For torchvision datasets (e.g., CIFAR10) that have a Bool `train` argument
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
            self.logger.info(
                f"Created torchvision data loader with {len(test_set)} samples, batch size {self.batch_size}"
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
                self.logger.info(f"Adjusted batch size to {self.batch_size} (data size)")
            self.data_loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size)
            self.logger.info(f"Created custom data loader with {len(data)} samples, batch size {self.batch_size}")

    def _eval_model(self) -> Dict[str, float]:
        if self.data_loader is None:
            self.logger.warning("Data loader not available for evaluation")
            return {"accuracy": 0.0}

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

        accuracy = 100 * correct / total if total > 0 else 0.0
        return {"accuracy": accuracy}

    def _evaluate_async(self, global_weights: Dict, current_round: int, evaluation_id: str):
        """Run evaluation in a separate thread."""
        try:
            # Safety check for model
            if self.model is None:
                self.logger.warning(f"Model not available for evaluation round {current_round}")
                return

            self.logger.info(f"Starting evaluation {evaluation_id} for round {current_round}")

            # Load the model weights
            global_weights_tensors = {k: torch.tensor(v) for k, v in global_weights.items()}
            self.model.load_state_dict(global_weights_tensors)

            # Evaluate the model
            metrics = self._eval_model()

            # Write metrics to tensorboard, starting from 0
            if self.tb_writer:
                for key, value in metrics.items():
                    self.tb_writer.add_scalar(key, value, current_round - 1)

            self.logger.info(f"Evaluation {evaluation_id} for round {current_round} completed with metrics: {metrics}")
        except Exception as e:
            # Log any errors that occur during evaluation
            self.logger.error(f"Error during evaluation for round {current_round}: {str(e)}")
        finally:
            self.logger.info(f"Evaluation {evaluation_id} cleanup completed")

    def _initialize(self, _event_type: str, fl_ctx: FLContext):
        # Initialize the model
        if isinstance(self.model_path, str):
            # load the model
            model_class = self._load_model(self.model_path, fl_ctx)
            if model_class is None:
                self.system_panic(
                    reason="Failed to load model class during initialization",
                    fl_ctx=fl_ctx,
                )
                return
            self.model = model_class()
        else:
            # the model_path is nn.Module
            self.model = self.model_path

        # Initialize the data loader
        self._create_data_loader()
        # Initialize the tensorboard writer
        app_root = fl_ctx.get_prop(FLContextKey.APP_ROOT)
        self.tb_writer = SummaryWriter(log_dir=app_root)

    def _is_initialized(self) -> bool:
        """Check if the evaluator is properly initialized."""
        return (
            self.model is not None
            and self.data_loader is not None
            and self.tb_writer is not None
            and self._thread_pool is not None
        )

    def evaluate(self, _event_type: str, fl_ctx: FLContext):
        # Safety check - ensure we're initialized
        if not self._is_initialized():
            self.logger.warning("Evaluator not initialized, skipping evaluation")
            return

        global_model = fl_ctx.get_prop(AppConstants.GLOBAL_MODEL)
        current_round = fl_ctx.get_prop(AppConstants.CURRENT_ROUND)

        # Check if evaluation should be performed
        if current_round % self.eval_frequency != 0:
            return

        # Get the global weights
        global_weights = global_model[ModelLearnableKey.WEIGHTS]

        # Create unique evaluation ID
        evaluation_id = f"eval_round_{current_round}"

        # Submit evaluation to thread pool
        future = self._thread_pool.submit(self._evaluate_async, global_weights, current_round, evaluation_id)
        self.logger.info(f"Submitted evaluation {evaluation_id} for round {current_round}")

    def _handle_end_run(self, _event_type: str, fl_ctx: FLContext):
        """Handle the END_RUN event to ensure proper cleanup."""
        self.logger.info("END_RUN Event received, starting shutdown process...")
        self.logger.info("Waiting for all evaluations to complete...")
        self._thread_pool.shutdown(wait=True)
        self.logger.info("Thread pool shutdown completed")

        # Close tensorboard writer
        if self.tb_writer:
            self.tb_writer.close()
            self.logger.info("Tensorboard writer closed")

        self.logger.info("Shutdown process completed")
